"""Native 256×256 RGB VAE training and held-out chart reconstruction."""

from __future__ import annotations

import json
import math
import random
import struct
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from PIL import Image, ImageDraw, ImageFont
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from kakeya.config import ExperimentConfig
from kakeya.objectives import kakeya_regularization
from kakeya.training import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGE = PROJECT_ROOT / "assets/test_images/kakeya_codec_card_v2_256.png"
LATENT_MEAN_BOUND = 3.0
CAPACITY_GATE_PSNR = 30.0
CAPACITY_GATE_SSIM = 0.97
CAPACITY_STEPS_PER_EPOCH = 32
QUALITY_REHEARSAL_STEPS = 8
TARGET_RATE_BPP = 2.5
RATE_LOSS_WEIGHT = 0.05
BITSTREAM_MAGIC = b"KKEYA-EB1"


class ImageCodecVAE(nn.Module):
    """Detail-preserving learned codec with a quantized 32×32 latent."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            SpaceToDepth(3, 24),
            ResidualBlock(24),
            SpaceToDepth(24, 32),
            ResidualBlock(32),
            SpaceToDepth(32, 64),
            ResidualBlock(64),
        )
        self.to_mu = nn.Conv2d(64, latent_dim, 1)
        self.to_log_var = nn.Conv2d(64, latent_dim, 1)
        self.entropy_bottleneck = EntropyBottleneck(latent_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1),
            ResidualBlock(64),
            DepthToSpace(64, 32),
            ResidualBlock(32),
            DepthToSpace(32, 24),
            ResidualBlock(24),
            DepthToSpace(24, 12),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(image)
        raw_mu = self.to_mu(encoded)
        mu = LATENT_MEAN_BOUND * torch.tanh(raw_mu / LATENT_MEAN_BOUND)
        return mu, self.to_log_var(encoded).clamp(-6, 2)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self, image: torch.Tensor, noise_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(image)
        std = torch.exp(0.5 * log_var)
        latent = mu + noise_scale * torch.randn_like(std) * std
        return self.decode(latent), mu, log_var, latent

    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(image)
        return self.decode(mu)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class SpaceToDepth(nn.Module):
    """Lossless 2× spatial rearrangement followed by channel mixing."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(input_channels * 4, output_channels, 3, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DepthToSpace(nn.Module):
    """Learned channel mixing followed by lossless 2× spatial rearrangement."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ProceduralDocumentDataset(Dataset[tuple[torch.Tensor, int]]):
    """Synthetic cards mixed with a declared codec-calibration reference."""

    WORDS = (
        "KAKEYA",
        "LATENT SPACE",
        "IMAGE CODEC",
        "256 x 256",
        "EDGE DETAIL",
        "TEXT SAMPLE",
        "0123456789",
        "AaBbCc",
    )

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        reference = Image.open(TEST_IMAGE).convert("RGB")
        reference_array = np.asarray(reference, dtype=np.float32) / 255.0
        self.reference = torch.from_numpy(reference_array).permute(2, 0, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # This verifies codec fidelity/capacity; it is not generalization.
        if index % 2 == 0:
            return self.reference.clone(), 1
        rng = random.Random(self.seed + index * 104729)
        image = Image.new(
            "RGB", (256, 256), tuple(rng.randint(225, 255) for _ in range(3))
        )
        draw = ImageDraw.Draw(image)
        palette = [
            (17, 22, 18),
            (57, 135, 255),
            (215, 255, 69),
            (255, 117, 72),
            (200, 61, 45),
        ]
        for _ in range(rng.randint(5, 10)):
            x0, y0 = rng.randint(4, 210), rng.randint(4, 210)
            x1, y1 = rng.randint(x0 + 10, 252), rng.randint(y0 + 10, 252)
            color = rng.choice(palette)
            if rng.random() < 0.55:
                draw.rectangle((x0, y0, x1, y1), outline=color, width=rng.randint(1, 4))
            else:
                draw.line((x0, y0, x1, y1), fill=color, width=rng.randint(1, 4))
        for row in range(rng.randint(4, 8)):
            size = rng.choice((11, 13, 16, 20))
            font = _font(size)
            text = rng.choice(self.WORDS)
            draw.text(
                (rng.randint(8, 120), 10 + row * 30),
                text,
                font=font,
                fill=rng.choice(palette[:2]),
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), 0


class CalibrationCardDataset(Dataset[tuple[torch.Tensor, int]]):
    """Repeated reference without duplicating its tensor in memory."""

    def __init__(self, size: int) -> None:
        self.size = size
        reference = Image.open(TEST_IMAGE).convert("RGB")
        reference_array = np.asarray(reference, dtype=np.float32) / 255.0
        self.reference = torch.from_numpy(reference_array).permute(2, 0, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        del index
        return self.reference.clone(), 1


@dataclass
class ImageCodecResult:
    history: dict[str, Any]
    metrics: dict[str, float]
    run_dir: Path
    images: dict[str, str]
    codec_baselines: list[dict[str, Any]]
    training_summary: dict[str, Any]
    bitstream: dict[str, Any]


EpochCallback = Callable[
    [int, int, dict[str, float], dict[str, float], Path], None
]


def train_image_codec(
    config: ExperimentConfig,
    device: torch.device,
    epoch_callback: EpochCallback | None = None,
) -> ImageCodecResult:
    seed_everything(config.seed)
    train_size = config.train_limit or 128
    validation_size = min(max(train_size // 8, 16), 64)
    train_loader = DataLoader(
        ProceduralDocumentDataset(train_size, config.seed),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        ProceduralDocumentDataset(validation_size, config.seed + 1_000_000),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    capacity_loader = DataLoader(
        CalibrationCardDataset(CAPACITY_STEPS_PER_EPOCH),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    capacity_validation_loader = DataLoader(
        CalibrationCardDataset(1),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    rehearsal_loader = DataLoader(
        CalibrationCardDataset(QUALITY_REHEARSAL_STEPS),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    model = ImageCodecVAE(config.latent_dim).to(device)
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.endswith(".quantiles")
    ]
    auxiliary_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".quantiles")
    ]
    optimizer = optim.AdamW(main_parameters, lr=config.learning_rate)
    auxiliary_optimizer = optim.Adam(auxiliary_parameters, lr=1e-3)
    run_dir = _run_directory(config)
    history: dict[str, Any] = {"epoch": [], "train": {}, "validation": {}}
    best_psnr = float("-inf")
    best_epoch: int | None = None
    best_rate_bpp = float("inf")
    validation_metrics: dict[str, float] | None = None
    generalization_metrics: dict[str, float] | None = None
    last_validation_capacity_stage: bool | None = None
    gate_epoch: int | None = None
    train_reference = (
        ProceduralDocumentDataset(1, config.seed).reference.unsqueeze(0).to(device)
    )

    for epoch in range(1, config.epochs + 1):
        capacity_stage = gate_epoch is None
        for group in optimizer.param_groups:
            group["lr"] = (
                max(config.learning_rate, 1e-3)
                if capacity_stage
                else config.learning_rate
            )
        # A learned entropy rate replaces the VAE KL proxy after the gate.
        kl_weight = 0.0
        train_metrics = _epoch(
            model,
            capacity_loader if capacity_stage else train_loader,
            device,
            config,
            optimizer=optimizer,
            kl_weight=kl_weight,
            kakeya_weight_override=0.0 if capacity_stage else None,
            use_entropy=True,
            auxiliary_optimizer=auxiliary_optimizer,
        )
        if not capacity_stage:
            rehearsal_metrics = _epoch(
                model,
                rehearsal_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=0.0,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
            )
            train_metrics.update(
                {
                    f"quality_rehearsal_{key}": value
                    for key, value in rehearsal_metrics.items()
                }
            )
        should_validate = (
            epoch == 1
            or epoch % 2 == 0
            or epoch == config.epochs
            or last_validation_capacity_stage != capacity_stage
        )
        if should_validate or validation_metrics is None:
            validation_metrics = _epoch(
                model,
                capacity_validation_loader
                if capacity_stage
                else validation_loader,
                device,
                config,
                kl_weight=kl_weight,
                kakeya_weight_override=0.0 if capacity_stage else None,
                use_entropy=True,
            )
            generalization_metrics = (
                _epoch(
                    model,
                    validation_loader,
                    device,
                    config,
                    kl_weight=kl_weight,
                    kakeya_weight_override=(
                        0.0 if capacity_stage else None
                    ),
                    use_entropy=True,
                )
                if capacity_stage
                else dict(validation_metrics)
            )
            last_validation_capacity_stage = capacity_stage
        assert generalization_metrics is not None
        validation_metrics.update(
            {
                f"generalization_{key}": value
                for key, value in generalization_metrics.items()
            }
        )
        calibration = _calibration_metrics(model, train_reference)
        best_eligible = calibration["rate_bpp"] <= TARGET_RATE_BPP
        if best_eligible and calibration["psnr"] > best_psnr:
            best_psnr = calibration["psnr"]
            best_epoch = epoch
            best_rate_bpp = calibration["rate_bpp"]
            _checkpoint(run_dir / "checkpoints/best.pt", model, config, epoch)
        if (
            gate_epoch is None
            and calibration["psnr"] >= CAPACITY_GATE_PSNR
            and calibration["ssim"] >= CAPACITY_GATE_SSIM
        ):
            gate_epoch = epoch
        train_metrics.update(
            {
                "calibration_psnr": calibration["psnr"],
                "calibration_ssim": calibration["ssim"],
                "calibration_rate_bpp": calibration["rate_bpp"],
                "best_checkpoint_epoch": float(best_epoch or 0),
                "capacity_stage": float(capacity_stage),
                "capacity_gate_passed": float(gate_epoch is not None),
            }
        )
        validation_metrics.update(
            {
                "calibration_psnr": calibration["psnr"],
                "calibration_ssim": calibration["ssim"],
                "calibration_rate_bpp": calibration["rate_bpp"],
                "best_checkpoint_epoch": float(best_epoch or 0),
                "capacity_stage": float(capacity_stage),
                "capacity_gate_passed": float(gate_epoch is not None),
            }
        )
        history["epoch"].append(epoch)
        _append(history["train"], train_metrics)
        _append(history["validation"], validation_metrics)
        if epoch_callback:
            epoch_callback(
                epoch, config.epochs, train_metrics, validation_metrics, run_dir
            )

    selected_checkpoint = run_dir / "checkpoints/best.pt"
    if selected_checkpoint.exists():
        payload = torch.load(
            selected_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model_state_dict"])
    _checkpoint(run_dir / "checkpoints/final.pt", model, config, best_epoch or config.epochs)
    metrics, image_paths, codec_baselines, bitstream = _evaluate_chart(
        model, device, run_dir
    )
    metrics.update(
        {
            "capacity_gate_passed": float(gate_epoch is not None),
            "capacity_gate_epoch": float(gate_epoch or 0),
        }
    )
    training_summary = {
        "capacity_gate_passed": gate_epoch is not None,
        "capacity_gate_epoch": gate_epoch,
        "capacity_gate": {
            "psnr": CAPACITY_GATE_PSNR,
            "ssim": CAPACITY_GATE_SSIM,
        },
        "selected_checkpoint_epoch": best_epoch,
        "selected_checkpoint_psnr": None if best_epoch is None else best_psnr,
        "selected_checkpoint_rate_bpp": None if best_epoch is None else best_rate_bpp,
        "target_rate_bpp": TARGET_RATE_BPP,
        "final_stage": (
            "compression_finetune"
            if gate_epoch is not None and gate_epoch < config.epochs
            else "capacity_pretrain"
        ),
        "compression_finetune_epochs": (
            max(config.epochs - gate_epoch, 0) if gate_epoch is not None else 0
        ),
    }
    (run_dir / "metrics/history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "torch_version": torch.__version__,
                "test_asset": str(TEST_IMAGE.relative_to(PROJECT_ROOT)),
                "test_asset_role": "in-distribution codec calibration",
                "model": "pixel_shuffle_entropy_bottleneck_codec",
                "latent_shape": [config.latent_dim, 32, 32],
                "training_policy": {
                    "validation_interval_epochs": 2,
                    "validation_samples": validation_size,
                    "validation_distribution": "same_as_current_stage",
                    "generalization_distribution": "held_out_procedural_cards",
                    "training_quantization": (
                        "straight_through_rounding_then_entropy_noise_proxy"
                    ),
                    "validation_decode": "rounded_entropy_latent",
                    "latent_mean_bound": LATENT_MEAN_BOUND,
                    "target_kl_weight": 0.0,
                    "rate_loss_weight": RATE_LOSS_WEIGHT,
                    "target_rate_bpp": TARGET_RATE_BPP,
                    "entropy_model": "CompressAI EntropyBottleneck",
                    "kakeya_input": "unit_normalized_latent",
                    "capacity_gate": {
                        "psnr": CAPACITY_GATE_PSNR,
                        "ssim": CAPACITY_GATE_SSIM,
                    },
                    "capacity_batch_size": 1,
                    "capacity_steps_per_epoch": CAPACITY_STEPS_PER_EPOCH,
                    "quality_rehearsal_steps": QUALITY_REHEARSAL_STEPS,
                    "checkpoint_selection": "best_calibration_psnr_under_target_bpp",
                },
                "training_summary": training_summary,
                "config": config.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ImageCodecResult(
        history,
        metrics,
        run_dir,
        image_paths,
        codec_baselines,
        training_summary,
        bitstream,
    )


def _epoch(
    model: ImageCodecVAE,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    optimizer: optim.Optimizer | None = None,
    kl_weight: float = 0.0,
    kakeya_weight_override: float | None = None,
    use_entropy: bool = False,
    auxiliary_optimizer: optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "mse": 0.0,
        "edge": 0.0,
        "structural": 0.0,
        "multiscale": 0.0,
        "kl": 0.0,
        "kakeya": 0.0,
        "kl_contribution": 0.0,
        "kakeya_contribution": 0.0,
        "rate_bpp": 0.0,
        "rate_excess_bpp": 0.0,
        "rate_contribution": 0.0,
        "entropy_aux": 0.0,
        "latent_rms": 0.0,
    }
    for images, _ in loader:
        images = images.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            mu, log_var = model.encode(images)
            if use_entropy:
                latent, rate_bpp = _entropy_quantize_and_rate(
                    model, mu, images, training=optimizer is not None
                )
            else:
                # Straight-through rounding makes capacity training see the
                # same integer latent that the final entropy coder stores.
                latent = (
                    mu + (mu.round() - mu).detach()
                    if optimizer is not None
                    else mu.round()
                )
                rate_bpp = torch.zeros((), device=images.device)
            reconstructed = model.decode(latent)
            detail_weight = _detail_weight(images)
            reconstruction = (
                (reconstructed - images).abs() * detail_weight
            ).mean()
            mse = F.mse_loss(reconstructed, images)
            edge = F.l1_loss(_edges(reconstructed), _edges(images))
            structural = 1 - _ssim(reconstructed, images)
            multiscale = _multiscale_l1(reconstructed, images)
            kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).mean()
            latent_points = latent.permute(0, 2, 3, 1).reshape(
                -1, latent.size(1)
            )
            bounded_latent_points = F.normalize(
                latent_points, dim=1, eps=1e-6
            )
            coverage = kakeya_regularization(
                bounded_latent_points,
                num_projections=int(config.objective.get("num_projections", 32)),
                k=int(config.objective.get("k", 8)),
            )
            kakeya_weight = (
                float(config.objective.get("lambda_kakeya", 0.001))
                if kakeya_weight_override is None
                else kakeya_weight_override
            )
            kl_contribution = kl_weight * kl
            kakeya_contribution = kakeya_weight * coverage
            rate_excess_bpp = (
                F.relu(rate_bpp - TARGET_RATE_BPP)
                if use_entropy
                else torch.zeros((), device=images.device)
            )
            rate_contribution = RATE_LOSS_WEIGHT * rate_excess_bpp
            total = (
                reconstruction
                + 5.0 * mse
                + 1.5 * edge
                + 0.5 * structural
                + 0.25 * multiscale
                + kl_contribution
                + kakeya_contribution
                + rate_contribution
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                entropy_aux = torch.zeros((), device=images.device)
                if auxiliary_optimizer is not None:
                    auxiliary_optimizer.zero_grad(set_to_none=True)
                    entropy_aux = model.entropy_bottleneck.loss()
                    entropy_aux.backward()
                    auxiliary_optimizer.step()
            else:
                entropy_aux = (
                    model.entropy_bottleneck.loss()
                    if use_entropy
                    else torch.zeros((), device=images.device)
                )
        for key, value in (
            ("total", total),
            ("reconstruction", reconstruction),
            ("mse", mse),
            ("edge", edge),
            ("structural", structural),
            ("multiscale", multiscale),
            ("kl", kl),
            ("kakeya", coverage),
            ("kl_contribution", kl_contribution),
            ("kakeya_contribution", kakeya_contribution),
            ("rate_bpp", rate_bpp),
            ("rate_excess_bpp", rate_excess_bpp),
            ("rate_contribution", rate_contribution),
            ("entropy_aux", entropy_aux),
            ("latent_rms", latent.square().mean().sqrt()),
        ):
            totals[key] += float(value.detach())
    return {key: value / len(loader) for key, value in totals.items()}


@torch.no_grad()
def _calibration_metrics(
    model: ImageCodecVAE, source: torch.Tensor
) -> dict[str, float]:
    model.eval()
    latent, _ = model.encode(source)
    quantized, rate_bpp = _entropy_quantize_and_rate(
        model, latent, source, training=False
    )
    reconstructed = model.decode(quantized).clamp(0, 1)
    mse = float(F.mse_loss(reconstructed, source))
    return {
        "mse": mse,
        "psnr": 99.0 if mse == 0 else 10 * math.log10(1.0 / mse),
        "ssim": float(_ssim(reconstructed, source)),
        "rate_bpp": float(rate_bpp),
    }


def _entropy_quantize_and_rate(
    model: ImageCodecVAE,
    latent: torch.Tensor,
    images: torch.Tensor,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    entropy_model = model.entropy_bottleneck
    perm = torch.cat(
        (
            torch.tensor([1, 0], dtype=torch.long, device=latent.device),
            torch.arange(2, latent.ndim, dtype=torch.long, device=latent.device),
        )
    )
    values = latent.permute(*perm).contiguous()
    shape = values.size()
    flat_values = values.reshape(values.size(0), 1, -1)
    medians = entropy_model._get_medians()
    if training:
        centered = flat_values - medians
        quantized_flat = flat_values + (
            torch.round(centered) + medians - flat_values
        ).detach()
    else:
        quantized_flat = entropy_model.quantize(
            flat_values, "dequantize", medians
        )
    likelihoods, _, _ = entropy_model._likelihood(quantized_flat)
    if entropy_model.use_likelihood_bound:
        likelihoods = entropy_model.likelihood_lower_bound(likelihoods)
    quantized = quantized_flat.reshape(shape)
    quantized = quantized.permute(*perm).contiguous()
    likelihoods = likelihoods.reshape(shape)
    likelihoods = likelihoods.permute(*perm).contiguous()
    rate_bpp = -torch.log2(likelihoods.clamp_min(1e-9)).sum()
    rate_bpp = rate_bpp / (images.size(0) * images.size(2) * images.size(3))
    return quantized, rate_bpp


@torch.no_grad()
def _evaluate_chart(
    model: ImageCodecVAE, device: torch.device, run_dir: Path
) -> tuple[
    dict[str, float],
    dict[str, str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    model.eval()
    source_image = Image.open(TEST_IMAGE).convert("RGB")
    source = torch.from_numpy(np.asarray(source_image, dtype=np.float32) / 255.0)
    source = source.permute(2, 0, 1).unsqueeze(0).to(device)
    mu, _ = model.encode(source)
    decoded_latent, bitstream = _encode_bitstream(model, mu, run_dir)
    reconstructed = model.decode(decoded_latent.to(device)).clamp(0, 1)
    mse = float(F.mse_loss(reconstructed, source))
    psnr = 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)
    ssim = float(_ssim(reconstructed, source))

    report_dir = run_dir / "reports"
    original_path = report_dir / "original.png"
    reconstruction_path = report_dir / "reconstruction.png"
    error_path = report_dir / "error.png"
    source_image.save(original_path)
    _to_image(reconstructed[0]).save(reconstruction_path)
    difference = (reconstructed - source).abs()[0]
    heat = torch.stack(
        (
            difference.mean(dim=0).mul(3).clamp(0, 1),
            difference.mean(dim=0).mul(0.7).clamp(0, 1),
            torch.zeros_like(difference[0]),
        )
    )
    _to_image(heat).save(error_path)
    codec_baselines = reference_codec_baselines()
    return (
        {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "latent_dim": float(model.latent_dim),
            "source_bytes": float(TEST_IMAGE.stat().st_size),
            "bitstream_bytes": float(bitstream["bytes"]),
            "bitstream_payload_bytes": float(bitstream["payload_bytes"]),
            "bitstream_bpp": float(bitstream["bpp"]),
        },
        {
            "original": "reports/original.png",
            "reconstruction": "reports/reconstruction.png",
            "error": "reports/error.png",
        },
        codec_baselines,
        bitstream,
    )


@torch.no_grad()
def _encode_bitstream(
    model: ImageCodecVAE,
    latent: torch.Tensor,
    run_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Entropy-code one latent and immediately decode the stored payload."""

    entropy_model = deepcopy(model.entropy_bottleneck).cpu().eval()
    entropy_model.update(force=True)
    latent_cpu = latent.detach().cpu()
    strings = entropy_model.compress(latent_cpu)
    payload = strings[0]
    shape = list(latent_cpu.shape[-2:])
    header = json.dumps(
        {
            "format": "kakeya-entropy-bottleneck",
            "version": 1,
            "latent_channels": model.latent_dim,
            "latent_shape": shape,
            "image_shape": [256, 256, 3],
            "requires_model_checkpoint": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    packaged = BITSTREAM_MAGIC + struct.pack(">I", len(header)) + header + payload
    bitstream_path = run_dir / "reports/reconstruction.kky"
    bitstream_path.write_bytes(packaged)

    # Decode from the exact entropy payload written above; the report image and
    # quality metrics therefore measure the real quantized representation.
    decoded = _decode_bitstream(entropy_model, bitstream_path)
    file_bytes = len(packaged)
    return decoded, {
        "path": "reports/reconstruction.kky",
        "filename": "reconstruction.kky",
        "format": "Kakeya EntropyBottleneck v1",
        "bytes": file_bytes,
        "payload_bytes": len(payload),
        "header_bytes": file_bytes - len(payload),
        "bpp": file_bytes * 8 / (256 * 256),
        "requires_checkpoint": True,
        "checkpoint": "checkpoints/final.pt",
    }


@torch.no_grad()
def _decode_bitstream(
    entropy_model: EntropyBottleneck,
    bitstream_path: Path,
) -> torch.Tensor:
    """Parse and entropy-decode a stored Kakeya bitstream."""

    packaged = bitstream_path.read_bytes()
    header_start = len(BITSTREAM_MAGIC)
    if not packaged.startswith(BITSTREAM_MAGIC) or len(packaged) < header_start + 4:
        raise ValueError("invalid Kakeya bitstream")
    header_length = struct.unpack(
        ">I", packaged[header_start : header_start + 4]
    )[0]
    header_end = header_start + 4 + header_length
    if header_end >= len(packaged):
        raise ValueError("truncated Kakeya bitstream")
    header = json.loads(packaged[header_start + 4 : header_end])
    shape = header.get("latent_shape")
    channels = header.get("latent_channels")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or channels != entropy_model.channels
    ):
        raise ValueError("unsupported Kakeya bitstream header")
    return entropy_model.decompress([packaged[header_end:]], shape)


def reference_codec_baselines() -> list[dict[str, Any]]:
    source_image = Image.open(TEST_IMAGE).convert("RGB")
    source = torch.from_numpy(
        np.asarray(source_image, dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    rows: list[dict[str, Any]] = [
        {
            "codec": "Original PNG",
            "settings": "source",
            "bytes": TEST_IMAGE.stat().st_size,
            "mse": 0.0,
            "psnr": 99.0,
            "ssim": 1.0,
        }
    ]
    variants = (
        ("PNG", "optimized", "PNG", {"optimize": True}),
        ("JPEG", "quality 90", "JPEG", {"quality": 90, "optimize": True}),
        ("WebP", "quality 90", "WEBP", {"quality": 90, "method": 6}),
    )
    for codec, settings, image_format, options in variants:
        buffer = BytesIO()
        source_image.save(buffer, format=image_format, **options)
        payload = buffer.getvalue()
        decoded = Image.open(BytesIO(payload)).convert("RGB")
        decoded_tensor = torch.from_numpy(
            np.asarray(decoded, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        mse = float(F.mse_loss(decoded_tensor, source))
        rows.append(
            {
                "codec": codec,
                "settings": settings,
                "bytes": len(payload),
                "mse": mse,
                "psnr": 99.0 if mse == 0 else 10 * math.log10(1.0 / mse),
                "ssim": float(_ssim(decoded_tensor, source)),
            }
        )
    rows.extend(_compressai_baselines(source, source_image))
    return rows


def _compressai_baselines(
    source: torch.Tensor, source_image: Image.Image
) -> list[dict[str, Any]]:
    try:
        from compressai.zoo import mbt2018_mean
    except ImportError:
        return []
    import ssl

    original_context = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    results: list[dict[str, Any]] = []
    try:
        for quality in (2, 4, 6):
            try:
                net = mbt2018_mean(
                    quality=quality, metric="mse", pretrained=True
                )
            except Exception:
                continue
            net.eval()
            with torch.no_grad():
                compressed = net.compress(source)
                strings = compressed["strings"]
                out = net.decompress(strings, compressed["shape"])
                x_hat = out["x_hat"].clamp(0, 1)
            total_bytes = sum(
                len(s) for string_list in strings for s in string_list
            )
            total_bytes += 64
            mse = float(F.mse_loss(x_hat, source))
            results.append(
                {
                    "codec": "CompressAI mbt2018",
                    "settings": f"quality {quality} (mse)",
                    "bytes": total_bytes,
                    "mse": mse,
                    "psnr": 99.0 if mse == 0 else 10 * math.log10(1.0 / mse),
                    "ssim": float(_ssim(x_hat, source)),
                    "learned": True,
                }
            )
    finally:
        try:
            ssl._create_default_https_context = original_context
        except Exception:
            pass
    return results


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2))
        / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    ).mean()


def _edges(image: torch.Tensor) -> torch.Tensor:
    horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
    vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
    return torch.cat(
        (
            F.pad(horizontal.abs(), (0, 1, 0, 0)),
            F.pad(vertical.abs(), (0, 0, 0, 1)),
        ),
        dim=1,
    )


def _detail_weight(image: torch.Tensor) -> torch.Tensor:
    """Emphasize text strokes and thin edges instead of flat backgrounds."""

    grayscale = image.mean(dim=1, keepdim=True)
    horizontal = F.pad(
        (grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]).abs(),
        (0, 1, 0, 0),
    )
    vertical = F.pad(
        (grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    edge_attention = ((horizontal + vertical) * 8).clamp(0, 1)
    dark_foreground = ((0.75 - grayscale) * 2).clamp(0, 1)
    return 1 + 3 * torch.maximum(edge_attention, dark_foreground)


def _multiscale_l1(
    reconstructed: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    losses = []
    for size in (128, 64):
        reconstructed_level = F.interpolate(
            reconstructed, size=(size, size), mode="area"
        )
        target_level = F.interpolate(target, size=(size, size), mode="area")
        losses.append(F.l1_loss(reconstructed_level, target_level))
    return torch.stack(losses).mean()


def _run_directory(config: ExperimentConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_dir / config.method / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = config.output_dir / config.method / f"{timestamp}-{suffix}"
        suffix += 1
    for name in ("checkpoints", "metrics", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def _checkpoint(
    path: Path, model: ImageCodecVAE, config: ExperimentConfig, epoch: int
) -> None:
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": epoch},
        path,
    )


def _append(target: dict[str, list[float]], values: dict[str, float]) -> None:
    for key, value in values.items():
        target.setdefault(key, []).append(value)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8), mode="RGB")
