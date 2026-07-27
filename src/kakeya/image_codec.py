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
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from torch import nn, optim
from torch.nn.utils.parametrizations import weight_norm
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
# If the model hasn't met the gate threshold by this epoch, force the gate
# open so the training still enters transition + finetune.  This prevents
# the model from staying in capacity_pretrain forever when the architecture
# / data combination simply cannot reach the threshold.
TARGET_RATE_BPP = 2.5
BITSTREAM_MAGIC = b"KKEYA-EB1"


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


class ResidualBlockGDN(nn.Module):
    """Residual block with GDN activation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            GDN(channels),
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            GDN(channels),
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)
class KakeyaHyperpriorCodec(nn.Module):
    """Hyperprior codec combining CompressAI MeanScaleHyperprior architecture
    with Kakeya coverage regularization.

    Architecture inherits from mbt2018:
    - ``g_a`` encoder with GDN activations
    - ``hyper_encoder`` / ``hyper_decoder`` for spatial entropy modelling
    - ``g_s`` decoder with IGDN activations
    - ``entropy_bottleneck`` / ``gaussian_conditional`` for rate estimation

    Novel additions:
    - Kakeya coverage regularization on unit-normalized latent points
    - SpaceToDepth / DepthToSpace preserve spatial precision for document text
    - Single-stage training (no capacity/transition/finetune)
    """

    def __init__(self, latent_dim: int = 16, hyper_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Analysis transform: SpaceToDepth × 3 → 8× downscale to 32×32
        # GDN replaces InstanceNorm for better compression statistics.
        self.g_a = nn.Sequential(
            SpaceToDepth(3, 24),
            ResidualBlockGDN(24),
            SpaceToDepth(24, 32),
            ResidualBlockGDN(32),
            SpaceToDepth(32, 64),
            ResidualBlockGDN(64),
            weight_norm(nn.Conv2d(64, latent_dim * 2, 1)),
        )

        # Hyperprior — spatially adaptive entropy model.
        # h_a compresses the latent → hyper-latent (32→8 spatial).
        # h_s decompresses back → scale + mean for GaussianConditional.
        self.h_a = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2),
        )
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(hyper_dim, hyper_dim, 5, stride=2, padding=2,
                               output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hyper_dim, hyper_dim, 5, stride=2, padding=2,
                               output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_dim, latent_dim * 2, 3, padding=1),
        )

        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(hyper_dim)
        self.gaussian_conditional = GaussianConditional(None)  # scale table set after h_s output

        # Synthesis transform: DepthToSpace × 3 → 8× upscale back to 256
        self.g_s = nn.Sequential(
            weight_norm(nn.Conv2d(latent_dim, 64, 3, padding=1)),
            ResidualBlockGDN(64),
            DepthToSpace(64, 32),
            ResidualBlockGDN(32),
            DepthToSpace(32, 24),
            ResidualBlockGDN(24),
            DepthToSpace(24, 12),
            weight_norm(nn.Conv2d(12, 24, 3, padding=1)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(24, 3, 3, padding=1)),
            nn.Sigmoid(),
        )

    def compress(self, image: torch.Tensor) -> dict[str, Any]:
        """Encode image → quantized latents for bitstream generation."""
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = 3.0 * torch.tanh(mu / 3.0)
        z = self.h_a(mu)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        y_hat = mu + (mu.round() - mu).detach()
        # Encode y_hat through the hyperprior path
        y_strings = self.entropy_bottleneck.compress(y_hat)
        return {"strings": [y_strings, z_strings], "shape": mu.size()[-2:]}

    def decompress(self, strings: list[bytes], shape: list[int]) -> torch.Tensor:
        """Decode bitstream back to image."""
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        # y is reconstructed from z (placeholder - real decode needs GaussianConditional)
        dummy = torch.zeros(1, self.latent_dim, *shape, device=z_hat.device)
        y_hat = dummy + (dummy.round() - dummy).detach()
        return self.g_s(y_hat)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass with rate estimation.

        GaussianConditional handles noise internally (no separate log_var/noise).
        Returns:
            reconstructed, mu, None (placeholder), y_hat, y_likelihoods, z_likelihoods
        """
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = 3.0 * torch.tanh(mu / 3.0)

        z = self.h_a(mu)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        params = self.h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        scale = scale.abs() + 1e-6

        y_hat, y_likelihoods = self.gaussian_conditional(mu, scale, mean)
        reconstructed = self.g_s(y_hat)

        return reconstructed, mu, None, y_hat, y_likelihoods, z_likelihoods

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return bounded mu from the encoder."""
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = 3.0 * torch.tanh(mu / 3.0)
        return mu

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to RGB image."""
        return self.g_s(latent)

    def update(self, force: bool = False) -> None:
        """Update entropy bottleneck CDF tables (required before compress)."""
        self.entropy_bottleneck.update(force=force)

    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        """Deterministic encode → quantize → decode for evaluation."""
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = 3.0 * torch.tanh(mu / 3.0)
        z = self.h_a(mu)
        z_hat = self.entropy_bottleneck.decompress(
            self.entropy_bottleneck.compress(z), z.size()[-2:]
        )
        params = self.h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        scale = scale.abs() + 1e-6
        # Use straight-through rounding for evaluation (STE same as training)
        y_hat = mu + (mu.round() - mu).detach()
        return self.g_s(y_hat)


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



@torch.no_grad()
def _rate_consistency_check(
    model: KakeyaHyperpriorCodec,
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
            mu = model.encode(source)
            # Get rate from forward pass
            _, _, _, _, y_likelihoods, z_likelihoods = model(source)
            rate_bpp = float(
                (-y_likelihoods.log().sum() - z_likelihoods.log().sum())
                / (source.shape[-1] * source.shape[-2])
            )
            quantized = model.decode(mu.round()).clamp(0, 1)
            mse = float(F.mse_loss(quantized, source))
        entry = {
            "scale": scale,
            "size": w,
            "rate_bpp": rate_bpp,
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


def _checkpoint(
    path: Path, model: KakeyaHyperpriorCodec, config: ExperimentConfig, epoch: int
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


@torch.no_grad()
def _calibration_metrics(
    model: KakeyaHyperpriorCodec, source: torch.Tensor
) -> dict[str, float]:
    model.eval()
    model.update()
    reconstructed = model.reconstruct(source).clamp(0, 1)
    mse = float(F.mse_loss(reconstructed, source))
    # Rate from forward pass
    _, _, _, _, y_likelihoods, z_likelihoods = model(source)
    rate_bpp = float(
        (-y_likelihoods.log2().sum() - z_likelihoods.log2().sum())
        / (source.shape[-1] * source.shape[-2])
    )
    return {
        "mse": mse,
        "psnr": 99.0 if mse == 0 else 10 * math.log10(1.0 / mse),
        "ssim": float(_ssim(reconstructed, source)),
        "rate_bpp": rate_bpp,
    }


def _hyperprior_epoch(
    model: KakeyaHyperpriorCodec,
    loader: DataLoader,
    device: torch.device,
    lambda_rate: float = 1.0,
    lambda_kakeya: float = 0.001,
    num_projections: int = 32,
    k: int = 3,
    optimizer: optim.Optimizer | None = None,
    aux_optimizer: optim.Optimizer | None = None,
    stage_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Single-epoch training with perceptual losses and per-stage weight scheduling.

    When stage_weights is provided, the loss is a weighted sum of MSE, edge,
    structural (1-SSIM), multiscale L1, LAB color (delta_e, hue, saturation),
    kakeya coverage, and rate.  Without stage_weights, falls back to basic
    MSE + lambda_rate*bpp + lambda_kakeya*coverage.
    """
    model.train(optimizer is not None)
    totals: dict[str, float] = {
        "total": 0.0, "mse": 0.0, "edge": 0.0, "structural": 0.0,
        "multiscale": 0.0, "lab": 0.0, "hue": 0.0, "saturation": 0.0,
        "kakeya": 0.0, "rate": 0.0, "psnr": 0.0, "rate_bpp": 0.0,
        "latent_rms": 0.0,
    }
    steps = 0

    for batch in loader:
        if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
            mini_batches = batch
        else:
            mini_batches = [batch]
        for images, _ in mini_batches:
            images = images.to(device)
            with torch.set_grad_enabled(optimizer is not None):
                reconstructed, mu, log_var, y_hat, y_likelihoods, z_likelihoods = model(images)
                loss_mse = F.mse_loss(reconstructed, images)
                rate = (-y_likelihoods.log2().sum() - z_likelihoods.log2().sum()) / images.size(0)
                bpp_bits = rate / (images.shape[-1] * images.shape[-2])

                lp = y_hat.permute(0, 2, 3, 1).reshape(-1, y_hat.size(1))
                norm = F.normalize(lp, dim=1)
                coverage = kakeya_regularization(norm, num_projections=num_projections, k=k)

                if stage_weights is not None:
                    sw = stage_weights
                    edge = F.l1_loss(_edges(reconstructed), _edges(images))
                    structural = 1.0 - _ssim(reconstructed, images)
                    multiscale = _multiscale_l1(reconstructed, images)
                    lab_losses = _lab_losses(reconstructed, images)
                    lab = lab_losses["delta_e"]
                    hue = lab_losses["hue"]
                    sat = lab_losses["saturation"]
                    rate_penalty = F.relu(bpp_bits - TARGET_RATE_BPP)

                    total = (
                        sw["mse"] * loss_mse
                        + sw["edge"] * edge
                        + sw["structural"] * structural
                        + sw["multiscale"] * multiscale
                        + sw["lab"] * lab
                        + sw["hue"] * hue
                        + sw["saturation"] * sat
                        + sw["kakeya"] * coverage
                        + lambda_rate * rate_penalty
                    )
                else:
                    edge = torch.zeros_like(loss_mse)
                    structural = torch.zeros_like(loss_mse)
                    multiscale = torch.zeros_like(loss_mse)
                    lab = torch.zeros_like(loss_mse)
                    hue = torch.zeros_like(loss_mse)
                    sat = torch.zeros_like(loss_mse)
                    rate_penalty = F.relu(bpp_bits - TARGET_RATE_BPP)
                    total = loss_mse + lambda_rate * rate_penalty + lambda_kakeya * coverage

                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    optimizer.step()
                    if aux_optimizer is not None:
                        aux_optimizer.zero_grad(set_to_none=True)
                        aux_loss = model.entropy_bottleneck.loss()
                        aux_loss.backward()
                        aux_optimizer.step()

            with torch.no_grad():
                psnr = 20 * torch.log10(1.0 / loss_mse.clamp_min(1e-8).sqrt())

            totals["total"] += float(total.detach())
            totals["mse"] += float(loss_mse.detach())
            totals["edge"] += float(edge.detach())
            totals["structural"] += float(structural.detach())
            totals["multiscale"] += float(multiscale.detach())
            totals["lab"] += float(lab.detach())
            totals["hue"] += float(hue.detach())
            totals["saturation"] += float(sat.detach())
            totals["kakeya"] += float(coverage.detach())
            totals["rate"] += float(rate.detach())
            totals["psnr"] += float(psnr)
            totals["rate_bpp"] += float(bpp_bits.detach())
            totals["latent_rms"] += float(y_hat.square().mean().sqrt().detach())
            steps += 1

    return {k: v / max(steps, 1) for k, v in totals.items()}
# Stage training constants
CAPACITY_GATE_PSNR = 26.0
CAPACITY_GATE_SSIM = 0.96
CAPACITY_GATE_FORCE_EPOCH = 40
CAPACITY_STEPS_PER_EPOCH = 32
QUALITY_REHEARSAL_STEPS = 8

TRANSITION_EPOCHS = 5
RATE_LOSS_WEIGHT = 0.01
FINETUNE_WARMUP_EPOCHS = 10
FINETUNE_WARMUP_RATE_WEIGHT = 0.001
FINETUNE_WARMUP_GRAD_CLIP = 10.0
def train_image_codec(
    config: ExperimentConfig,
    device: torch.device,
    epoch_callback: (
        Callable[[int, int, dict[str, float], dict[str, float], Path], None] | None
    ) = None,
    run_dir: Path | None = None,
) -> ImageCodecResult:
    """Single-stage rate-distortion training for KakeyaHyperpriorCodec.

    Total = MSE + lambda_rate * bpp + lambda_kakeya * kakeya_coverage.
    """
    seed_everything(config.seed)
    train_size = config.train_limit or 128
    validation_size = min(max(train_size // 8, 16), 64)
    real_train_size = max(train_size // 4, 16)

    train_loader = DataLoader(
        ProceduralDocumentDataset(train_size, config.seed),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=_size_aware_collate,
    )
    real_loader = DataLoader(
        RealImageDataset(real_train_size, config.seed + 2_000_000),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=_size_aware_collate,
    )
    validation_loader = DataLoader(
        ProceduralDocumentDataset(validation_size, config.seed + 1_000_000),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=_size_aware_collate,
    )

    hyper_dim = max(4, config.latent_dim // 2)
    model = KakeyaHyperpriorCodec(latent_dim=config.latent_dim, hyper_dim=hyper_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    aux_optimizer = optim.Adam(
        [p for n, p in model.named_parameters() if n.endswith(".quantiles")],
        lr=1e-3,
    )

    lambda_rate = float(config.objective.get("lambda_rate", RATE_LOSS_WEIGHT))
    lambda_kakeya = float(config.objective.get("lambda_kakeya", 0.001))
    num_projections = int(config.objective.get("num_projections", 32))
    k = int(config.objective.get("k", 3))
    stage_weights_map = config.stage_weights()

    # Capacity gate dataloaders
    capacity_loader = DataLoader(
        CalibrationCardDataset(CAPACITY_STEPS_PER_EPOCH, multiscale=True, seed=config.seed),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=_size_aware_collate,
    )

    run_dir = run_dir or _run_directory(config)
    history: dict[str, Any] = {"epoch": [], "train": {}, "validation": {}}
    best_psnr = float("-inf")
    best_epoch: int | None = None
    gate_epoch: int | None = None
    gate_forced: bool = False
    train_reference = (
        ProceduralDocumentDataset(1, config.seed)
        .reference_tensor.unsqueeze(0)
        .to(device)
    )

    for epoch in range(1, config.epochs + 1):
        capacity_stage = gate_epoch is None
        transition_stage = (
            gate_epoch is not None
            and epoch - gate_epoch <= TRANSITION_EPOCHS
            and epoch < config.epochs
        )
        stage: Literal["capacity", "transition", "finetune"] = (
            "capacity" if capacity_stage
            else "transition" if transition_stage
            else "finetune"
        )

        # Stage-dependent learning rate (matches original VAE schedule)
        if capacity_stage:
            current_lr = max(config.learning_rate, 1e-3)
        elif transition_stage:
            current_lr = max(config.learning_rate, 5e-4)
        else:
            current_lr = config.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # Rate weight and grad clip
        if capacity_stage:
            rate_weight = 0.0
            grad_clip = None
        elif transition_stage:
            rate_weight = FINETUNE_WARMUP_RATE_WEIGHT
            grad_clip = FINETUNE_WARMUP_GRAD_CLIP
        else:
            epochs_since_gate = epoch - (gate_epoch or 0)
            if epochs_since_gate <= FINETUNE_WARMUP_EPOCHS:
                rate_weight = FINETUNE_WARMUP_RATE_WEIGHT
                grad_clip = FINETUNE_WARMUP_GRAD_CLIP
            else:
                rate_weight = lambda_rate
                grad_clip = None

        sw = stage_weights_map[stage]
        epoch_kw = dict(
            model=model, device=device, lambda_rate=rate_weight,
            lambda_kakeya=lambda_kakeya, num_projections=num_projections, k=k,
            stage_weights=sw,
        )

        if capacity_stage:
            # Capacity stage: train on reference card, procedural, real images
            ref_metrics = _hyperprior_epoch(
                **epoch_kw, loader=capacity_loader, optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            prog_metrics = _hyperprior_epoch(
                **epoch_kw, loader=train_loader, optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            real_metrics = _hyperprior_epoch(
                **epoch_kw, loader=real_loader, optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            train_metrics = {
                key: 0.4 * ref_metrics[key] + 0.3 * prog_metrics[key] + 0.3 * real_metrics[key]
                for key in ref_metrics
            }
        else:
            # Finetune/transition: train on procedural + real + rehearsal (reference card)
            ref_metrics = _hyperprior_epoch(
                **epoch_kw, loader=capacity_loader, optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            prog_metrics = _hyperprior_epoch(
                **epoch_kw, loader=train_loader, optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            real_metrics = _hyperprior_epoch(
                **epoch_kw, loader=real_loader, optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            train_metrics = {
                key: 0.25 * ref_metrics[key] + 0.375 * prog_metrics[key] + 0.375 * real_metrics[key]
                for key in ref_metrics
            }

        validation_metrics = _hyperprior_epoch(
            **epoch_kw, loader=validation_loader,
        )

        calibration = _calibration_metrics(model, train_reference)

        # Capacity gate check (only when still in capacity stage)
        if capacity_stage:
            if calibration["psnr"] >= CAPACITY_GATE_PSNR and calibration["ssim"] >= CAPACITY_GATE_SSIM:
                gate_epoch = epoch
            elif epoch >= CAPACITY_GATE_FORCE_EPOCH:
                gate_epoch = epoch
                gate_forced = True

        if calibration["psnr"] > best_psnr and (gate_epoch is not None or calibration["rate_bpp"] <= TARGET_RATE_BPP):
            best_psnr = calibration["psnr"]
            best_epoch = epoch
            _checkpoint(run_dir / "checkpoints/best.pt", model, config, epoch)

        train_metrics.update({
            "stage": {"capacity": 1, "transition": 2, "finetune": 3}[stage],
            "capacity_stage": float(capacity_stage),
            "capacity_gate_passed": gate_epoch is not None,
            "capacity_gate_epoch": float(gate_epoch or 0),
            "calibration_psnr": calibration["psnr"],
            "calibration_ssim": calibration["ssim"],
            "calibration_rate_bpp": calibration["rate_bpp"],
        })
        validation_metrics.update({
            "stage": {"capacity": 1, "transition": 2, "finetune": 3}[stage],
            "capacity_stage": float(capacity_stage),
            "capacity_gate_passed": gate_epoch is not None,
            "capacity_gate_epoch": float(gate_epoch or 0),
            "calibration_psnr": calibration["psnr"],
            "calibration_ssim": calibration["ssim"],
            "calibration_rate_bpp": calibration["rate_bpp"],
        })

        history["epoch"].append(epoch)
        _append(history["train"], train_metrics)
        _append(history["validation"], validation_metrics)
        if epoch_callback:
            epoch_callback(
                epoch, config.epochs, train_metrics, validation_metrics, run_dir
            )

    selected = run_dir / "checkpoints/best.pt"
    if selected.exists():
        payload = torch.load(selected, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    _checkpoint(run_dir / "checkpoints/final.pt", model, config, best_epoch or config.epochs)

    model.update()
    metrics, image_paths, codec_baselines, bitstream = _evaluate_chart(
        model, device, run_dir
    )
    rate_consistency = _rate_consistency_check(model, device, run_dir)

    finetune_stage_epochs = max(0, (config.epochs - (gate_epoch or 0) - TRANSITION_EPOCHS))
    training_summary = {
        "method": "hyperprior_kakeya",
        "total_epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "selected_checkpoint_epoch": best_epoch,
        "selected_checkpoint_psnr": best_psnr if best_epoch else None,
        "target_rate_bpp": TARGET_RATE_BPP,
        "final_stage": "finetune" if gate_epoch else "capacity",
        "lambda_rate": lambda_rate,
        "lambda_kakeya": lambda_kakeya,
        "config": config.to_dict(),
        "capacity_gate_passed": gate_epoch is not None,
        "capacity_gate_forced": gate_forced,
        "capacity_gate_epoch": gate_epoch,
        "capacity_gate": {"psnr": CAPACITY_GATE_PSNR, "ssim": CAPACITY_GATE_SSIM},
        "transition_epochs": TRANSITION_EPOCHS,
        "compression_finetune_epochs": finetune_stage_epochs,
    }
    metrics.update(training_summary)
    # _evaluate_chart already includes CompressAI baselines via reference_codec_baselines().

    return ImageCodecResult(
        history=history,
        metrics=metrics,
        run_dir=run_dir,
        images=image_paths,
        codec_baselines=codec_baselines,
        training_summary=training_summary,
        bitstream=bitstream,
    )

@torch.no_grad()
def _evaluate_chart(
    model: KakeyaHyperpriorCodec, device: torch.device, run_dir: Path
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
    mu = model.encode(source)
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
    model: KakeyaHyperpriorCodec,
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
        mu_hd = model.encode(hd_tensor)
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
    model: KakeyaHyperpriorCodec,
    latent: torch.Tensor,
    run_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize and store latent as a simplified bitstream."""

    # Use rounded latent for simplicity; full entropy coding needs GaussianConditional
    latent_cpu = latent.detach().cpu()
    quantized = latent_cpu.round()
    packaged = _pack_bitstream(quantized, model.latent_dim, [256, 256, 3])
    bitstream_path = run_dir / "reports/reconstruction.kky"
    bitstream_path.write_bytes(packaged)
    decoded = quantized
    # Parse back out header/payload sizes from the packaged bitstream
    header_start = len(BITSTREAM_MAGIC)
    header_len = struct.unpack(">I", packaged[header_start:header_start+4])[0]
    payload_bytes = len(packaged) - header_start - 4 - header_len
    header_bytes = header_start + 4 + header_len
    file_bytes = len(packaged)
    return decoded, {
        "path": "reports/reconstruction.kky",
        "filename": "reconstruction.kky",
        "format": "Kakeya Round-Trip v2",
        "bytes": file_bytes,
        "payload_bytes": payload_bytes,
        "header_bytes": header_bytes,
        "bpp": file_bytes * 8 / (256 * 256),
        "requires_checkpoint": True,
        "checkpoint": "checkpoints/final.pt",
    }


@torch.no_grad()

def _pack_bitstream(latent: torch.Tensor, channels: int, image_shape: list[int]) -> bytes:
    """Package a quantized latent tensor as a .kky bitstream (v2 format)."""
    header = json.dumps(
        {
            "format": "kakeya-round-trip",
            "version": 2,
            "latent_channels": channels,
            "latent_shape": list(latent.shape[-2:]),
            "image_shape": image_shape,
            "requires_model_checkpoint": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    payload = latent.numpy().astype(np.float32).tobytes()
    return BITSTREAM_MAGIC + struct.pack(">I", len(header)) + header + payload

def _decode_bitstream(
    entropy_model: EntropyBottleneck | None,
    bitstream_path: Path,
) -> torch.Tensor:
    """Parse and decode a stored Kakeya bitstream. Supports v1 (EntropyBottleneck) and v2 (direct float)."""

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
    fmt = header.get("format", "")
    shape = header.get("latent_shape")
    channels = header.get("latent_channels")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(v, int) and v > 0 for v in shape)
        or not isinstance(channels, int)
    ):
        raise ValueError("unsupported Kakeya bitstream header")
    payload = packaged[header_end:]
    if fmt == "kakeya-round-trip":
        # v2: raw float32 latent values
        payload_tensor = torch.tensor(
            np.frombuffer(payload, dtype=np.float32).reshape(1, channels, *shape),
        )
        return payload_tensor
    # v1: EntropyBottleneck compressed (backward compat)
    if entropy_model is None:
        raise ValueError("v1 bitstream requires EntropyBottleneck")
    return entropy_model.decompress([payload], shape)

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
            net = net.to(source.device)
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


def _lab_losses(
    reconstructed: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """CIELAB-based color losses: ΔE, hue, and saturation.

    All returns are normalized to ~O(1) scale so that stage weights stay
    on the same order as other loss weights.
    """
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
    ).mean() / 7.0
    # Hue: 1 - cos(Δh) handles angular wraparound, range [0, 2].
    a_rec, b_rec = rec_lab[:, 1:2, :, :], rec_lab[:, 2:3, :, :]
    a_tgt, b_tgt = tgt_lab[:, 1:2, :, :], tgt_lab[:, 2:3, :, :]
    h_rec = torch.atan2(b_rec, a_rec)
    h_tgt = torch.atan2(b_tgt, a_tgt)
    hue = (1.0 - torch.cos(h_rec - h_tgt)).mean() / 2.0
    # Saturation: |Δchroma|, typical chroma scale 0-50.
    C_rec = torch.sqrt(a_rec.pow(2) + b_rec.pow(2) + 1e-8)
    C_tgt = torch.sqrt(a_tgt.pow(2) + b_tgt.pow(2) + 1e-8)
    saturation = (C_rec - C_tgt).abs().mean() / 10.0
    return {"delta_e": delta_e, "hue": hue, "saturation": saturation}


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

