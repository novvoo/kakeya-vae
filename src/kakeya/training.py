"""Training engine and run artifact management."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from kakeya.config import ExperimentConfig
from kakeya.data import get_mnist_dataloaders
from kakeya.models import VAE, Discriminator
from kakeya.objectives import compute_objective


@dataclass
class TrainingResult:
    model: VAE
    history: dict[str, Any]
    test_z: np.ndarray
    test_labels: np.ndarray
    run_dir: Path


EpochCallback = Callable[
    [int, int, dict[str, float], dict[str, float], Path], None
]


def train_model(
    config: ExperimentConfig,
    *,
    device: str | torch.device | None = None,
    progress: bool = True,
    epoch_callback: EpochCallback | None = None,
) -> TrainingResult:
    seed_everything(config.seed)
    resolved_device = torch.device(device or _default_device())
    train_loader, test_loader = get_mnist_dataloaders(
        config.batch_size,
        config.data_dir,
        num_workers=config.num_workers,
        download=config.download,
        seed=config.seed,
        train_limit=config.train_limit,
        test_limit=config.test_limit,
    )
    model = VAE(latent_dim=config.latent_dim).to(resolved_device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    discriminator: Discriminator | None = None
    discriminator_optimizer: optim.Optimizer | None = None
    if config.method == "factor_vae":
        discriminator = Discriminator(config.latent_dim).to(resolved_device)
        discriminator_optimizer = optim.Adam(
            discriminator.parameters(), lr=config.learning_rate * 0.1
        )

    run_dir = _create_run_directory(config)
    history: dict[str, Any] = {"epoch": [], "train": {}, "validation": {}}
    best_validation_loss = float("inf")
    last_z = np.empty((0, config.latent_dim), dtype=np.float32)
    last_labels = np.empty((0,), dtype=np.int64)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            resolved_device,
            config,
            discriminator,
            discriminator_optimizer,
            progress=progress,
        )
        validation_metrics, last_z, last_labels = validate(
            model,
            test_loader,
            resolved_device,
            config,
            discriminator,
            progress=progress,
        )
        history["epoch"].append(epoch)
        _append_metrics(history["train"], train_metrics)
        _append_metrics(history["validation"], validation_metrics)
        if epoch_callback is not None:
            epoch_callback(
                epoch,
                config.epochs,
                train_metrics,
                validation_metrics,
                run_dir,
            )

        validation_loss = validation_metrics["total"]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            _save_checkpoint(
                run_dir / "checkpoints" / "best.pt", model, config, epoch
            )

    _save_checkpoint(
        run_dir / "checkpoints" / "final.pt", model, config, config.epochs
    )
    _write_json(run_dir / "metrics" / "history.json", history)
    np.save(run_dir / "embeddings" / "z.npy", last_z)
    np.save(run_dir / "embeddings" / "labels.npy", last_labels)
    _write_manifest(run_dir, config, resolved_device)
    return TrainingResult(model, history, last_z, last_labels, run_dir)


def train_epoch(
    model: VAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
    discriminator: Discriminator | None = None,
    discriminator_optimizer: optim.Optimizer | None = None,
    *,
    progress: bool = True,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    for data, _ in tqdm(dataloader, desc="train", leave=False, disable=not progress):
        data = data.to(device, non_blocking=True)

        if discriminator is not None and discriminator_optimizer is not None:
            _train_discriminator(model, discriminator, discriminator_optimizer, data)

        optimizer.zero_grad(set_to_none=True)
        output = compute_objective(model, data, config.method, **config.objective)
        total = output.total
        if discriminator is not None:
            _set_requires_grad(discriminator, False)
            tc_estimate = discriminator(output.z).mean()
            total = total + float(config.objective.get("gamma", 10.0)) * tc_estimate
            _set_requires_grad(discriminator, True)
            output.components["total_correlation"] = float(tc_estimate.detach())
            output.components["total"] = float(total.detach())
        total.backward()
        optimizer.step()
        _accumulate(totals, output.components)
    return _average(totals, len(dataloader))


def validate(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    discriminator: Discriminator | None = None,
    *,
    progress: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    totals: dict[str, float] = {}
    all_z: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for data, labels in tqdm(
            dataloader, desc="validate", leave=False, disable=not progress
        ):
            data = data.to(device, non_blocking=True)
            output = compute_objective(model, data, config.method, **config.objective)
            if discriminator is not None:
                tc_estimate = discriminator(output.z).mean()
                total = output.total + float(
                    config.objective.get("gamma", 10.0)
                ) * tc_estimate
                output.components["total_correlation"] = float(tc_estimate)
                output.components["total"] = float(total)
            _accumulate(totals, output.components)
            # Deterministic posterior means make cross-run evaluation comparable.
            all_z.append(output.mu.cpu().numpy())
            all_labels.append(labels.numpy())
    return (
        _average(totals, len(dataloader)),
        np.concatenate(all_z),
        np.concatenate(all_labels),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_discriminator(
    model: VAE,
    discriminator: Discriminator,
    optimizer: optim.Optimizer,
    data: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        mu, log_var = model.encode(data)
        z = model.reparameterize(mu, log_var)
    permuted = torch.stack(
        [z[torch.randperm(z.size(0), device=z.device), index] for index in range(z.size(1))],
        dim=1,
    )
    joint_logits = discriminator(z.detach())
    permuted_logits = discriminator(permuted.detach())
    loss = F.binary_cross_entropy_with_logits(
        joint_logits, torch.ones_like(joint_logits)
    ) + F.binary_cross_entropy_with_logits(
        permuted_logits, torch.zeros_like(permuted_logits)
    )
    loss.backward()
    optimizer.step()


def _create_run_directory(config: ExperimentConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = config.output_dir / config.method
    run_dir = base / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{timestamp}-{suffix}"
        suffix += 1
    for directory in ("checkpoints", "embeddings", "metrics", "reports"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_checkpoint(
    path: Path, model: VAE, config: ExperimentConfig, epoch: int
) -> None:
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": epoch},
        path,
    )


def _write_manifest(
    run_dir: Path, config: ExperimentConfig, device: torch.device
) -> None:
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "torch_version": torch.__version__,
            "config": config.to_dict(),
        },
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _accumulate(totals: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + value


def _average(totals: dict[str, float], batches: int) -> dict[str, float]:
    if batches == 0:
        raise ValueError("dataloader must contain at least one batch")
    return {key: value / batches for key, value in totals.items()}


def _append_metrics(history: dict[str, list[float]], values: dict[str, float]) -> None:
    for key, value in values.items():
        history.setdefault(key, []).append(value)


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
