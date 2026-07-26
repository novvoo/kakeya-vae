"""High-level experiment orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from kakeya.config import ExperimentConfig, load_config
from kakeya.data import get_mnist_dataloaders
from kakeya.evaluation import evaluate_all
from kakeya.image_codec import train_image_codec
from kakeya.training import train_model


def run_experiment(
    config: ExperimentConfig,
    *,
    device: str | None = None,
    progress: bool = True,
) -> tuple[dict[str, Any], Path]:
    if config.method == "image_codec":
        resolved_device = torch.device(device or _default_device())
        image_result = train_image_codec(config, resolved_device)
        metrics = dict(image_result.metrics)
        metrics["method"] = config.method
        metrics_path = image_result.run_dir / "metrics" / "summary.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metrics, image_result.run_dir

    training_result = train_model(config, device=device, progress=progress)
    resolved_device = next(training_result.model.parameters()).device
    _, test_loader = get_mnist_dataloaders(
        config.batch_size,
        config.data_dir,
        num_workers=config.num_workers,
        download=False,
        seed=config.seed,
        test_limit=config.test_limit,
    )
    metrics = evaluate_all(
        training_result.model,
        test_loader,
        training_result.test_z,
        training_result.test_labels,
        resolved_device,
    )
    metrics["method"] = config.method
    metrics_path = training_result.run_dir / "metrics" / "summary.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, training_result.run_dir


def run_config_files(
    paths: Iterable[str | Path],
    *,
    device: str | None = None,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    configs = [load_config(path) for path in paths]
    if not configs:
        raise ValueError("at least one config file is required")

    all_results: dict[str, dict[str, Any]] = {}
    for config in configs:
        metrics, _ = run_experiment(config, device=device, progress=progress)
        all_results[config.method] = metrics

    comparison_dir = configs[0].output_dir / "comparisons"
    from kakeya.reporting import generate_comparison_report

    generate_comparison_report(all_results, comparison_dir)
    return all_results


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
