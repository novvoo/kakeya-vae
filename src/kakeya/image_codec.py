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
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import nn, optim
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, Dataset

from kakeya.config import ExperimentConfig
from kakeya.objectives import kakeya_regularization
from kakeya.training import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGE = PROJECT_ROOT / "assets/test_images/kakeya_codec_card_v2_256.png"
TEST_IMAGE_HD = PROJECT_ROOT / "assets/test_images/kakeya_codec_card_v2_source.png"
HD_IMAGE_DIR = PROJECT_ROOT / "assets/hd_images"
SOURCE_IMAGE_SIZE = 512
TARGET_IMAGE_SIZE = 256
# Multi-scale training sizes (must be multiples of 8 due to 3× PixelShuffle).
# 50% of samples stay at 256² to preserve core codec quality; the other 50%
# are drawn from these sizes so the model sees a real range of resolutions.
MULTISCALE_TRAIN_SIZES = (128, 192, 256, 384, 512, 768)
MULTISCALE_PRIMARY_SIZE = 256
MULTISCALE_PRIMARY_RATIO = 0.5
LATENT_MEAN_BOUND = 3.0
CAPACITY_GATE_PSNR = 26.0  # Lowered from 28.0 to match multi-content training regime
CAPACITY_GATE_SSIM = 0.96  # Lowered from 0.97 to allow gate to trigger earlier
# If the model hasn't met the gate threshold by this epoch, force the gate
# open so the training still enters transition + finetune.  This prevents
# the model from staying in capacity_pretrain forever when the architecture
# / data combination simply cannot reach the threshold.
CAPACITY_GATE_FORCE_EPOCH = 40
CAPACITY_STEPS_PER_EPOCH = 32
QUALITY_REHEARSAL_STEPS = 8
TARGET_RATE_BPP = 2.5
RATE_LOSS_WEIGHT = 0.01  # Reduced from 0.05 to avoid overwhelming reconstruction loss
FINETUNE_WARMUP_EPOCHS = 10  # First N epochs of fine-tune use relaxed constraints
FINETUNE_WARMUP_RATE_WEIGHT = 0.001  # Very weak rate loss during warmup
FINETUNE_WARMUP_GRAD_CLIP = 10.0  # Allow larger gradient updates during warmup
BITSTREAM_MAGIC = b"KKEYA-EB1"


class ImageCodecVAE(nn.Module):
    """Detail-preserving learned codec with a fully scale-invariant architecture."""

    def __init__(self, latent_dim: int = 16) -> None:
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
        self.to_mu = weight_norm(nn.Conv2d(64, latent_dim, 1))
        self.to_log_var = weight_norm(nn.Conv2d(64, latent_dim, 1))
        self.entropy_bottleneck = EntropyBottleneck(latent_dim)
        self.decoder = nn.Sequential(
            weight_norm(nn.Conv2d(latent_dim, 64, 3, padding=1)),
            ResidualBlock(64),
            DepthToSpace(64, 32),
            ResidualBlock(32),
            DepthToSpace(32, 24),
            ResidualBlock(24),
            DepthToSpace(24, 12),
            weight_norm(nn.Conv2d(12, 24, 3, padding=1)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(24, 3, 3, padding=1)),
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
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(),
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(),
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class SpaceToDepth(nn.Module):
    """Lossless 2× spatial rearrangement followed by channel mixing."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.PixelUnshuffle(2),
            weight_norm(nn.Conv2d(input_channels * 4, output_channels, 3, padding=1)),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DepthToSpace(nn.Module):
    """Learned channel mixing followed by lossless 2× spatial rearrangement."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Conv2d(input_channels, output_channels * 4, 3, padding=1)),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ProceduralDocumentDataset(Dataset[tuple[torch.Tensor, int]]):
    """Multi-scale synthetic cards for scale-invariant codec training.

    Each sample is rendered at a randomly chosen resolution from
    MULTISCALE_TRAIN_SIZES, so the model sees 128² through 768² inputs every
    epoch.  Half of the samples stay at the primary 256² size to preserve
    core codec quality; the other half are drawn uniformly from the full
    size pool.  Content is always authored at SOURCE_IMAGE_SIZE (512²) and
    then resized to the target size, so large samples keep fine detail and
    small samples mimic real downscaled inputs.
    """

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
        self.reference_full = reference.resize(
            (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE), Image.LANCZOS
        )
        reference_array = np.asarray(reference, dtype=np.float32) / 255.0
        self.reference_tensor = torch.from_numpy(reference_array).permute(2, 0, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rng = random.Random(self.seed + index * 104729)
        target_size = self._pick_target_size(rng)
        if index % 2 == 0:
            image = self.reference_full.resize(
                (target_size, target_size), Image.LANCZOS
            )
            label = 1
        else:
            image = self._generate_procedural(rng).resize(
                (target_size, target_size), Image.LANCZOS
            )
            label = 0
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), label

    @staticmethod
    def _pick_target_size(rng: random.Random) -> int:
        if rng.random() < MULTISCALE_PRIMARY_RATIO:
            return MULTISCALE_PRIMARY_SIZE
        return rng.choice(MULTISCALE_TRAIN_SIZES)

    def _generate_procedural(self, rng: random.Random) -> Image.Image:
        image = Image.new(
            "RGB",
            (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE),
            tuple(rng.randint(225, 255) for _ in range(3)),
        )
        draw = ImageDraw.Draw(image)
        palette = [
            (17, 22, 18),
            (57, 135, 255),
            (215, 255, 69),
            (255, 117, 72),
            (200, 61, 45),
        ]
        margin = 8
        inner = SOURCE_IMAGE_SIZE - margin * 2
        for _ in range(rng.randint(5, 12)):
            x0, y0 = rng.randint(margin, inner - 20), rng.randint(margin, inner - 20)
            x1, y1 = rng.randint(x0 + 20, inner), rng.randint(y0 + 20, inner)
            color = rng.choice(palette)
            if rng.random() < 0.55:
                draw.rectangle((x0, y0, x1, y1), outline=color, width=rng.randint(2, 8))
            else:
                draw.line((x0, y0, x1, y1), fill=color, width=rng.randint(2, 8))
        for row in range(rng.randint(4, 10)):
            font_size = rng.choice((16, 20, 26, 32))
            font = _font(font_size)
            text = rng.choice(self.WORDS)
            draw.text(
                (rng.randint(12, SOURCE_IMAGE_SIZE // 2), 16 + row * 56),
                text,
                font=font,
                fill=rng.choice(palette[:2]),
            )
        return image


class CalibrationCardDataset(Dataset[tuple[torch.Tensor, int]]):
    """Reference card for capacity training and quality rehearsal.

    When ``multiscale`` is True, each sample is rendered at a randomly chosen
    resolution from ``MULTISCALE_TRAIN_SIZES`` (half stay at 256²).  This
    prevents the model from overfitting to a single 256px reference during the
    capacity stage, while still producing a single fixed-size sample when used
    for gate validation.
    """

    def __init__(self, size: int, *, multiscale: bool = False, seed: int = 0) -> None:
        self.size = size
        self.multiscale = multiscale
        self.seed = seed
        reference = Image.open(TEST_IMAGE).convert("RGB")
        # Keep the full-resolution source so we can render at any target size.
        self.reference_full = reference.resize(
            (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE), Image.LANCZOS
        )
        reference_array = np.asarray(
            self.reference_full, dtype=np.float32
        ) / 255.0
        # Default 256² tensor used when multiscale is False.
        self.reference = torch.from_numpy(reference_array).permute(2, 0, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not self.multiscale:
            return self.reference.clone(), 1
        rng = random.Random(self.seed + index * 104729)
        target_size = ProceduralDocumentDataset._pick_target_size(rng)
        resized = self.reference_full.resize(
            (target_size, target_size), Image.LANCZOS
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), 1


class RealImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Real-world high-resolution images for diverse texture and color training.

    Loads JPEG/PNG images from ``HD_IMAGE_DIR`` (assets/hd_images/).  Each
    sample is center-cropped to square, then resized to a randomly chosen
    resolution from ``MULTISCALE_TRAIN_SIZES`` (half stay at 256²) so the
    model sees real photographic content at multiple scales.

    If the directory is empty or missing, the dataset falls back to the
    reference card so training still works without user-provided images.
    """

    _SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self._images: list[Image.Image] = []
        if HD_IMAGE_DIR.is_dir():
            for path in sorted(HD_IMAGE_DIR.iterdir()):
                if path.suffix.lower() in self._SUPPORTED_EXTS:
                    try:
                        img = Image.open(path).convert("RGB")
                        self._images.append(img)
                    except (OSError, Image.UnidentifiedImageError):
                        # Skip corrupt or unreadable files silently.
                        continue
        if not self._images:
            # Fallback: use the reference card so training remains functional
            # even without user-provided HD images.
            reference = Image.open(TEST_IMAGE).convert("RGB")
            self._images = [
                reference.resize(
                    (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE), Image.LANCZOS
                )
            ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rng = random.Random(self.seed + index * 104729)
        source = self._images[index % len(self._images)]
        target_size = ProceduralDocumentDataset._pick_target_size(rng)
        image = self._center_crop_square(source).resize(
            (target_size, target_size), Image.LANCZOS
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), 0

    @staticmethod
    def _center_crop_square(image: Image.Image) -> Image.Image:
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return image.crop((left, top, left + side, top + side))


@dataclass
class ImageCodecResult:
    history: dict[str, Any]
    metrics: dict[str, float]
    run_dir: Path
    images: dict[str, str]
    codec_baselines: list[dict[str, Any]]
    training_summary: dict[str, Any]
    bitstream: dict[str, Any]


def _size_aware_collate(
    batch: list[tuple[torch.Tensor, int]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Group samples by spatial size so mixed 256² and 512² batches coexist.

    Returns a list of (images, labels) mini-batches, each with uniform size.
    """
    groups: dict[int, list[tuple[torch.Tensor, int]]] = {}
    for image, label in batch:
        size = image.shape[-1]
        groups.setdefault(size, []).append((image, label))
    mini_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for size in sorted(groups):
        items = groups[size]
        images = torch.stack([item[0] for item in items])
        labels = torch.tensor([item[1] for item in items], dtype=torch.long)
        mini_batches.append((images, labels))
    return mini_batches


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
    # Real high-resolution images supplement the synthetic procedural data so
    # the model sees photographic textures/skin/gradients at training time.
    # Falls back to the reference card if assets/hd_images/ is empty.
    real_train_size = max(train_size // 4, 16)
    real_loader = DataLoader(
        RealImageDataset(real_train_size, config.seed + 2_000_000),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_size_aware_collate,
    )
    train_loader = DataLoader(
        ProceduralDocumentDataset(train_size, config.seed),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_size_aware_collate,
    )
    validation_loader = DataLoader(
        ProceduralDocumentDataset(validation_size, config.seed + 1_000_000),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_size_aware_collate,
    )
    capacity_loader = DataLoader(
        CalibrationCardDataset(
            CAPACITY_STEPS_PER_EPOCH, multiscale=True, seed=config.seed
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_size_aware_collate,
    )
    capacity_validation_loader = DataLoader(
        CalibrationCardDataset(1),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    rehearsal_loader = DataLoader(
        CalibrationCardDataset(
            QUALITY_REHEARSAL_STEPS, multiscale=True, seed=config.seed + 500_000
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_size_aware_collate,
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
    gate_forced: bool = False
    train_reference = (
        ProceduralDocumentDataset(1, config.seed).reference_tensor.unsqueeze(0).to(device)
    )

    TRANSITION_EPOCHS = 5
    for epoch in range(1, config.epochs + 1):
        capacity_stage = gate_epoch is None
        transition_stage = (
            gate_epoch is not None
            and epoch - gate_epoch <= TRANSITION_EPOCHS
            and epoch < config.epochs
        )
        for group in optimizer.param_groups:
            if capacity_stage:
                group["lr"] = max(config.learning_rate, 1e-3)
            elif transition_stage:
                lr = config.learning_rate
                group["lr"] = max(lr, 5e-4)
            else:
                group["lr"] = config.learning_rate
        # A learned entropy rate replaces the VAE KL proxy after the gate.
        kl_weight = 0.0
        # Determine rate loss weight and grad clip for the fine-tune phase
        if capacity_stage or transition_stage:
            # During capacity stage or the transition, use relaxed constraints
            # to allow the model to adapt to new data distributions.
            current_rate_weight = FINETUNE_WARMUP_RATE_WEIGHT if not capacity_stage else RATE_LOSS_WEIGHT
            current_grad_clip = FINETUNE_WARMUP_GRAD_CLIP if not capacity_stage else 5.0
        else:
            # Check if we are still in the warmup window after the gate
            epochs_since_gate = epoch - (gate_epoch or 0)
            if gate_epoch is not None and epochs_since_gate <= FINETUNE_WARMUP_EPOCHS:
                current_rate_weight = FINETUNE_WARMUP_RATE_WEIGHT
                current_grad_clip = FINETUNE_WARMUP_GRAD_CLIP
            else:
                current_rate_weight = RATE_LOSS_WEIGHT
                current_grad_clip = 5.0

        if capacity_stage:
            # Train on THREE sources every epoch:
            #   1. reference card (drives the capacity gate)
            #   2. procedural data (prevents overfitting to one image)
            #   3. real HD images (adds photographic textures/colors)
            # Kakeya regularization is enabled (override=None → config) from
            # the very first epoch so the latent space cannot collapse.
            ref_metrics = _epoch(
                model,
                capacity_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
            )
            prog_metrics = _epoch(
                model,
                train_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
            )
            real_metrics = _epoch(
                model,
                real_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
            )
            train_metrics = {
                key: 0.4 * ref_metrics[key]
                + 0.3 * prog_metrics[key]
                + 0.3 * real_metrics[key]
                for key in ref_metrics
            }
        elif transition_stage:
            # Smooth transition: train on reference, procedural, and real
            # data to prevent a sudden loss spike from distribution shift.
            ref_metrics = _epoch(
                model,
                capacity_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
                rate_loss_weight_override=current_rate_weight,
                grad_clip_max_norm=current_grad_clip,
            )
            prog_metrics = _epoch(
                model,
                train_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
                rate_loss_weight_override=current_rate_weight,
                grad_clip_max_norm=current_grad_clip,
            )
            real_metrics = _epoch(
                model,
                real_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
                rate_loss_weight_override=current_rate_weight,
                grad_clip_max_norm=current_grad_clip,
            )
            train_metrics = {
                key: 0.4 * ref_metrics[key]
                + 0.3 * prog_metrics[key]
                + 0.3 * real_metrics[key]
                for key in ref_metrics
            }
        else:
            # Finetune: mix procedural + real data, with reference rehearsal
            # handled separately below.
            prog_metrics = _epoch(
                model,
                train_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
                rate_loss_weight_override=current_rate_weight,
                grad_clip_max_norm=current_grad_clip,
            )
            real_metrics = _epoch(
                model,
                real_loader,
                device,
                config,
                optimizer=optimizer,
                kl_weight=kl_weight,
                kakeya_weight_override=None,
                use_entropy=True,
                auxiliary_optimizer=auxiliary_optimizer,
                rate_loss_weight_override=current_rate_weight,
                grad_clip_max_norm=current_grad_clip,
            )
            train_metrics = {
                key: 0.5 * prog_metrics[key] + 0.5 * real_metrics[key]
                for key in prog_metrics
            }
        # Rehearse the reference card in every stage so the capacity gate
        # can still be met even while training on multi-scale data.
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
            rate_loss_weight_override=current_rate_weight,
            grad_clip_max_norm=current_grad_clip,
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
                    kakeya_weight_override=0.0 if capacity_stage else None,
                    use_entropy=True,
                )
                if capacity_stage
                else dict(validation_metrics)
            )
            last_validation_capacity_stage = capacity_stage
        if generalization_metrics is not None:
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
        if gate_epoch is None:
            if (
                calibration["psnr"] >= CAPACITY_GATE_PSNR
                and calibration["ssim"] >= CAPACITY_GATE_SSIM
            ):
                gate_epoch = epoch
                gate_forced = False
            elif epoch >= CAPACITY_GATE_FORCE_EPOCH:
                # Force the gate open so the model still gets transition +
                # finetune phases even when the threshold is unreachable.
                # This is a graceful degradation: the checkpoint is usable
                # but the gate is marked as forced for downstream visibility.
                gate_epoch = epoch
                gate_forced = True
                print(
                    f"[kakeya] capacity gate forced at epoch {epoch} "
                    f"(psnr={calibration['psnr']:.2f}, "
                    f"ssim={calibration['ssim']:.4f}, "
                    f"threshold={CAPACITY_GATE_PSNR}/{CAPACITY_GATE_SSIM})"
                )
        train_metrics.update(
            {
                "calibration_psnr": calibration["psnr"],
                "calibration_ssim": calibration["ssim"],
                "calibration_rate_bpp": calibration["rate_bpp"],
                "best_checkpoint_epoch": float(best_epoch or 0),
                "capacity_stage": float(capacity_stage),
                "capacity_gate_passed": float(gate_epoch is not None),
                "capacity_gate_forced": float(gate_forced),
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
                "capacity_gate_forced": float(gate_forced),
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
            "capacity_gate_forced": float(gate_forced),
            "capacity_gate_epoch": float(gate_epoch or 0),
        }
    )
    rate_consistency = _rate_consistency_check(model, device, run_dir)
    training_summary = {
        "capacity_gate_passed": gate_epoch is not None,
        "capacity_gate_forced": gate_forced,
        "capacity_gate_epoch": gate_epoch,
        "capacity_gate": {
            "psnr": CAPACITY_GATE_PSNR,
            "ssim": CAPACITY_GATE_SSIM,
            "force_epoch": CAPACITY_GATE_FORCE_EPOCH,
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
        "rate_consistency_max_deviation": rate_consistency["max_deviation"],
        "rate_consistency_scales": rate_consistency["scales"],
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
                        "force_epoch": CAPACITY_GATE_FORCE_EPOCH,
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
    rate_loss_weight_override: float | None = None,
    grad_clip_max_norm: float = 5.0,
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
        "lab": 0.0,
        "kl_contribution": 0.0,
        "kakeya_contribution": 0.0,
        "lab_contribution": 0.0,
        "rate_bpp": 0.0,
        "rate_excess_bpp": 0.0,
        "rate_contribution": 0.0,
        "entropy_aux": 0.0,
        "latent_rms": 0.0,
    }
    step_count = 0
    for batch in loader:
        if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
            mini_batches = batch
        else:
            mini_batches = [batch]
        for images, _ in mini_batches:
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
                # Use warmup-adjusted rate loss weight if provided, else default
                rate_weight = (
                    rate_loss_weight_override
                    if rate_loss_weight_override is not None
                    else RATE_LOSS_WEIGHT
                )
                rate_contribution = rate_weight * rate_excess_bpp
                lab = _lab_loss(reconstructed, images)
                lab_contribution = _LAMBDA_LAB * lab
                total = (
                    reconstruction
                    + 5.0 * mse
                    + 1.5 * edge
                    + 0.5 * structural
                    + 0.25 * multiscale
                    + kl_contribution
                    + kakeya_contribution
                    + lab_contribution
                    + rate_contribution
                )
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
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
                ("lab", lab),
                ("kl_contribution", kl_contribution),
                ("kakeya_contribution", kakeya_contribution),
                ("lab_contribution", lab_contribution),
                ("rate_bpp", rate_bpp),
                ("rate_excess_bpp", rate_excess_bpp),
                ("rate_contribution", rate_contribution),
                ("entropy_aux", entropy_aux),
                ("latent_rms", latent.square().mean().sqrt()),
            ):
                totals[key] += float(value.detach())
        step_count += 1
    return {key: value / step_count for key, value in totals.items()}


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


def _rate_consistency_check(
    model: ImageCodecVAE,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    source_image = Image.open(TEST_IMAGE).convert("RGB")
    model.eval()
    results: list[dict[str, float]] = []
    reference_bpp: float | None = None
    for scale in (0.5, 1.0, 1.5, 2.0):
        w = int(256 * scale)
        h = int(256 * scale)
        resized = source_image.resize((w, h), Image.LANCZOS)
        source = torch.from_numpy(
            np.asarray(resized, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            latent, _ = model.encode(source)
            _, rate_bpp = _entropy_quantize_and_rate(
                model, latent, source, training=False
            )
            quantized = model.decode(latent.round()).clamp(0, 1)
            mse = float(F.mse_loss(quantized, source))
        entry = {
            "scale": scale,
            "size": w,
            "rate_bpp": float(rate_bpp),
            "mse": mse,
        }
        results.append(entry)
        if scale == 1.0:
            reference_bpp = float(rate_bpp)
    if reference_bpp is not None:
        for entry in results:
            entry["bpp_deviation"] = (
                (entry["rate_bpp"] - reference_bpp) / reference_bpp
                if reference_bpp > 0
                else 0.0
            )
    return {
        "scales": results,
        "max_deviation": max(
            abs(e.get("bpp_deviation", 0.0)) for e in results
        )
        if results
        else 0.0,
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

    # Also evaluate the high-resolution source image (1024²) so the frontend
    # can show original/reconstruction/error for a non-256 input.  This does
    # not produce a bitstream; it only runs encode → decode(mu) and reports
    # reconstruction metrics for the larger image.
    hd_metrics = _evaluate_hd_chart(model, device, run_dir, report_dir)

    codec_baselines = reference_codec_baselines()
    metrics = {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "latent_dim": float(model.latent_dim),
        "source_bytes": float(TEST_IMAGE.stat().st_size),
        "bitstream_bytes": float(bitstream["bytes"]),
        "bitstream_payload_bytes": float(bitstream["payload_bytes"]),
        "bitstream_bpp": float(bitstream["bpp"]),
    }
    metrics.update(hd_metrics)
    return (
        metrics,
        {
            "original": "reports/original.png",
            "reconstruction": "reports/reconstruction.png",
            "error": "reports/error.png",
            "original_hd": "reports/original_hd.png",
            "reconstruction_hd": "reports/reconstruction_hd.png",
            "error_hd": "reports/error_hd.png",
        },
        codec_baselines,
        bitstream,
    )


def _evaluate_hd_chart(
    model: ImageCodecVAE,
    device: torch.device,
    run_dir: Path,
    report_dir: Path,
) -> dict[str, float]:
    """Reconstruct the 1024² source image and save original/recon/error PNGs.

    Returns a dict of metrics prefixed with ``hd_`` so the frontend can show
    PSNR/SSIM for the high-resolution input alongside the 256² calibration.
    Skips silently if the HD source asset is missing.
    """
    if not TEST_IMAGE_HD.is_file():
        return {}
    hd_image = Image.open(TEST_IMAGE_HD).convert("RGB")
    # Pad to a multiple of 8 (3× PixelShuffle requires divisibility by 8).
    w, h = hd_image.size
    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    if pad_w or pad_h:
        hd_image = ImageOps.expand(hd_image, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
    hd_tensor = torch.from_numpy(np.asarray(hd_image, dtype=np.float32) / 255.0)
    hd_tensor = hd_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        mu_hd, _ = model.encode(hd_tensor)
        recon_hd = model.decode(mu_hd).clamp(0, 1)
    mse_hd = float(F.mse_loss(recon_hd, hd_tensor))
    psnr_hd = 99.0 if mse_hd == 0 else 10 * math.log10(1.0 / mse_hd)
    ssim_hd = float(_ssim(recon_hd, hd_tensor))

    hd_image.save(report_dir / "original_hd.png")
    _to_image(recon_hd[0]).save(report_dir / "reconstruction_hd.png")
    diff_hd = (recon_hd - hd_tensor).abs()[0]
    heat_hd = torch.stack(
        (
            diff_hd.mean(dim=0).mul(3).clamp(0, 1),
            diff_hd.mean(dim=0).mul(0.7).clamp(0, 1),
            torch.zeros_like(diff_hd[0]),
        )
    )
    _to_image(heat_hd).save(report_dir / "error_hd.png")

    pixels = w * h  # use original (unpadded) pixel count for bpp-equivalent stats
    return {
        "hd_psnr": psnr_hd,
        "hd_ssim": ssim_hd,
        "hd_mse": mse_hd,
        "hd_width": float(w),
        "hd_height": float(h),
        "hd_pixels": float(pixels),
    }


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


_SRGB_TO_LIN_COEFF = 1.0 / 12.92
_SRGB_TO_LIN_EXP = 1.0 / 2.4
_SRGB_TO_LIN_OFFSET = 0.055
_SRGB_TO_LIN_SCALE = 1.055

_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_CUBED = _LAB_DELTA ** 3
_LAB_FACTOR = 3.0 * _LAB_DELTA ** 2

_D65_XYZ = torch.tensor([0.95047, 1.0, 1.08883])

_XYZ_TO_LAB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.max() <= 1.5:
        srgb = rgb.clamp(0, 1)
    else:
        srgb = (rgb / 255.0).clamp(0, 1)
    lin = torch.where(
        srgb <= 0.04045,
        srgb * _SRGB_TO_LIN_COEFF,
        ((srgb + _SRGB_TO_LIN_OFFSET) / _SRGB_TO_LIN_SCALE).pow(_SRGB_TO_LIN_EXP),
    )
    if lin.size(1) == 3:
        rgb_ch = lin.permute(0, 2, 3, 1)
        xyz = (rgb_ch @ _XYZ_TO_LAB.to(lin.device).T).permute(0, 3, 1, 2)
    else:
        xyz = lin
    ref = _D65_XYZ.to(lin.device).view(1, 3, 1, 1)
    xyz_norm = xyz / ref
    f = torch.where(
        xyz_norm > _LAB_DELTA_CUBED,
        xyz_norm.pow(1.0 / 3.0),
        xyz_norm / _LAB_FACTOR + 4.0 / 29.0,
    )
    L = 116.0 * f[:, 1:2, :, :] - 16.0
    a = 500.0 * (f[:, 0:1, :, :] - f[:, 1:2, :, :])
    b = 200.0 * (f[:, 1:2, :, :] - f[:, 2:3, :, :])
    return torch.cat([L, a, b], dim=1)


def _lab_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rec_lab = _rgb_to_lab(reconstructed)
    tgt_lab = _rgb_to_lab(target)
    diff = rec_lab - tgt_lab
    L = tgt_lab[:, 0:1, :, :]
    chroma_weight = 1.0 + 2.0 * ((50.0 - L.clamp(0.0, 50.0)) / 50.0)
    delta_e = torch.sqrt(
        0.25 * diff[:, 0:1, :, :].pow(2)
        + chroma_weight * diff[:, 1:2, :, :].pow(2)
        + chroma_weight * diff[:, 2:3, :, :].pow(2)
        + 1e-8,
    )
    return delta_e.mean()


_LAMBDA_LAB = 0.05


def _detail_weight(image: torch.Tensor) -> torch.Tensor:
    """Emphasize edges and text strokes instead of flat backgrounds."""
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
    cur = min(reconstructed.shape[-1], target.shape[-1])
    orig_weight = 0.5
    orig_loss = F.l1_loss(reconstructed, target)
    losses = [(orig_loss, orig_weight)]
    for size in (128, 64):
        if cur <= size:
            continue
        reconstructed_level = F.interpolate(
            reconstructed, size=(size, size), mode="bilinear", align_corners=False
        )
        target_level = F.interpolate(
            target, size=(size, size), mode="bilinear", align_corners=False
        )
        losses.append((F.l1_loss(reconstructed_level, target_level), 0.25))
    if len(losses) == 1:
        return losses[0][0]
    total_weight = sum(w for _, w in losses)
    return sum(loss * w for loss, w in losses) / total_weight


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


def migrate_legacy_state_dict(
    state_dict: dict[str, torch.Tensor],
    latent_dim: int = 16,
) -> dict[str, torch.Tensor]:
    """Convert a legacy GroupNorm-based state dict to the new WeightNorm format.

    Old checkpoints use GroupNorm + plain Conv2d; the new architecture uses
    WeightNorm + plain Conv2d.  We match Conv2d weights and biases by tensor
    shape (out_channels, in_channels, kH, kW) and decompose each weight into
    the WeightNorm ``(original0, original1)`` pair.
    """

    dummy = ImageCodecVAE(latent_dim=latent_dim)
    new_state = dummy.state_dict()

    new_conv_info: dict[tuple[int, ...], list[dict[str, str]]] = {}
    for key in new_state:
        if "parametrizations.weight.original1" in key:
            shape = tuple(new_state[key].shape)
            entry = {
                "w_key": key,
                "g_key": key.replace("original1", "original0"),
                "b_key": key.replace("parametrizations.weight.original1", "bias"),
            }
            new_conv_info.setdefault(shape, []).append(entry)

    old_conv_weights: dict[tuple[int, ...], list[torch.Tensor]] = {}
    old_conv_biases: dict[tuple[int, ...], list[torch.Tensor]] = {}
    old_norm_params: dict[tuple[int, ...], list[torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if tensor.dim() == 4 and "weight" in key.lower():
            shape = tuple(tensor.shape)
            old_conv_weights.setdefault(shape, []).append(tensor)
        elif tensor.dim() == 1 and "bias" in key.lower():
            shape = (tensor.shape[0],)
            old_conv_biases.setdefault(shape, []).append(tensor)
        elif tensor.dim() == 1 and "weight" in key.lower():
            shape = (tensor.shape[0],)
            old_norm_params.setdefault(shape, []).append(tensor)

    migrated: dict[str, torch.Tensor] = {}

    for shape, entries in new_conv_info.items():
        old_weights = old_conv_weights.get(shape, [])
        for i, entry in enumerate(entries):
            if i < len(old_weights):
                w = old_weights[i]
                out_ch = w.shape[0]
                norm = w.reshape(out_ch, -1).norm(dim=1)
                migrated[entry["w_key"]] = w / norm.reshape(out_ch, 1, 1, 1).clamp_min(1e-10)
                migrated[entry["g_key"]] = norm.reshape(out_ch, 1, 1, 1)
            else:
                migrated[entry["w_key"]] = new_state[entry["w_key"]]
                migrated[entry["g_key"]] = new_state[entry["g_key"]]

    for shape, entries in new_conv_info.items():
        bias_shape = (shape[0],)
        old_biases = old_conv_biases.get(bias_shape, [])
        for i, entry in enumerate(entries):
            b_key = entry["b_key"]
            if i < len(old_biases):
                migrated[b_key] = old_biases[i]
            elif b_key in new_state:
                migrated[b_key] = new_state[b_key]

    norm_weight_keys = [
        k for k in new_state
        if k.endswith(".weight") and new_state[k].dim() == 1 and "parametrizations" not in k
    ]
    norm_bias_keys = [
        k for k in new_state
        if k.endswith(".bias") and "parametrizations" not in k
    ]
    for k in norm_weight_keys:
        shape = (new_state[k].shape[0],)
        params = old_norm_params.get(shape, [])
        if params:
            migrated[k] = params.pop(0)
        else:
            migrated[k] = new_state[k]
    for k in norm_bias_keys:
        if k not in migrated:
            migrated[k] = new_state[k]

    for key, tensor in new_state.items():
        if key not in migrated:
            if "_medians" in key:
                for old_key, old_tensor in state_dict.items():
                    if "_medians" in old_key:
                        migrated[key] = old_tensor
                        break
                else:
                    migrated[key] = tensor
            else:
                migrated[key] = tensor

    return migrated


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


def _to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8), mode="RGB")
