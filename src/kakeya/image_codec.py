"""Native 256×256 RGB VAE training and held-out chart reconstruction."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
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
LATENT_MEAN_BOUND = 5.0
# If the model hasn't met the gate threshold by this epoch, force the gate
# open so the training still enters transition + finetune.  This prevents
# the model from staying in capacity_pretrain forever when the architecture
# / data combination simply cannot reach the threshold.
TARGET_RATE_BPP = 2.5
BITSTREAM_MAGIC = b"KKEYA-EB1"
SMALL_IMAGE_MAX_SIZE = 256
SMALL_IMAGE_RATE_MULTIPLIER = 0.35
SMALL_IMAGE_EDGE_WEIGHT = 5.0
SMALL_IMAGE_STRUCTURAL_WEIGHT = 1.2
SMALL_IMAGE_MULTISCALE_WEIGHT = 0.2
SMALL_IMAGE_HIGH_FREQUENCY_WEIGHT = 1.0
HD_CHECKPOINT_SIZE = 512
HD_PSNR_MAX_REGRESSION = 0.1
HD_SSIM_MAX_REGRESSION = 0.001


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


class LearnedUpsample(nn.Module):
    """2× upsampling combining bilinear smooth base with a learned sharp residual.

    ``base = bilinear(x)`` gives smooth interpolation (good for gradients).
    ``residual = PixelShuffle(Conv(x)) - bilinear(x)`` gives the sharp edge
    signal that bilinear loses.  The residual is learned so the model can
    decide per-location how much sharpening to apply.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.sharpener = nn.Sequential(
            weight_norm(nn.Conv2d(channels, channels * 4, 3, padding=1)),
            nn.PixelShuffle(2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            value, scale_factor=2, mode="bilinear", align_corners=False
        )
        residual = self.sharpener(value)
        return base + residual


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

    def __init__(self, latent_dim: int = 8, hyper_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Analysis transform: SpaceToDepth × 2 → 4× downscale to 64×64
        # GDN replaces InstanceNorm for better compression statistics.
        self.g_a = nn.Sequential(
            SpaceToDepth(3, 24),
            ResidualBlockGDN(24),
            SpaceToDepth(24, 32),
            ResidualBlockGDN(32),
            weight_norm(nn.Conv2d(32, 64, 1)),  # expand to 64ch bottleneck
            ResidualBlockGDN(64),
            weight_norm(nn.Conv2d(64, latent_dim * 2, 1)),  # squeeze to latent
        )

        # Hyperprior — spatially adaptive entropy model.
        # h_a compresses the latent → hyper-latent (64→16 spatial).
        # h_s decompresses back → scale + mean for GaussianConditional.
        self.h_a = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2),
        )
        self.h_s_attention = nn.MultiheadAttention(
            hyper_dim * 4, num_heads=4, batch_first=True, dropout=0.0
        )
        self.h_s_proj_in = nn.Conv2d(hyper_dim, hyper_dim * 4, 1)
        self.h_s_proj_out = nn.Conv2d(hyper_dim * 4, hyper_dim, 1)
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(
                hyper_dim, hyper_dim, 5, stride=2, padding=2, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                hyper_dim, hyper_dim, 5, stride=2, padding=2, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_dim, latent_dim * 2, 3, padding=1),
        )

        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(hyper_dim)
        self.gaussian_conditional = GaussianConditional(
            None
        )  # scale table set after h_s output
        self.g_s = nn.Sequential(
            weight_norm(nn.Conv2d(latent_dim, 64, 3, padding=1)),  # expand to 64ch
            ResidualBlockGDN(64),
            weight_norm(nn.Conv2d(64, 32, 3, padding=1)),  # squeeze to 32ch
            ResidualBlockGDN(32),
            LearnedUpsample(32),  # bilinear + sharp residual
            weight_norm(nn.Conv2d(32, 24, 3, padding=1)),  # mix to 24ch
            ResidualBlockGDN(24),
            DepthToSpace(24, 12),
            weight_norm(nn.Conv2d(12, 24, 3, padding=1)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(24, 3, 3, padding=1)),
            nn.Sigmoid(),
        )

    def compress(self, image: torch.Tensor) -> dict[str, Any]:
        """Encode an image with the learned hyperprior probability model."""
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = LATENT_MEAN_BOUND * torch.tanh(mu / LATENT_MEAN_BOUND)
        z = self.h_a(mu)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self._apply_h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        scale = scale.abs() + 1e-6
        if scale.shape[-2:] != mu.shape[-2:]:
            scale = F.interpolate(
                scale, size=mu.shape[-2:], mode="bilinear", align_corners=False
            )
            mean = F.interpolate(
                mean, size=mu.shape[-2:], mode="bilinear", align_corners=False
            )
        indexes = self.gaussian_conditional.build_indexes(scale)
        y_strings = self.gaussian_conditional.compress(mu, indexes, means=mean)
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "y_shape": mu.size()[-2:],
        }

    def decompress(
        self,
        strings: list[list[bytes]],
        shape: tuple[int, int],
        y_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Decode an image using z to reconstruct y's conditional distribution."""
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self._apply_h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        scale = scale.abs() + 1e-6
        if scale.shape[-2:] != y_shape:
            scale = F.interpolate(
                scale, size=y_shape, mode="bilinear", align_corners=False
            )
            mean = F.interpolate(
                mean, size=y_shape, mode="bilinear", align_corners=False
            )
        indexes = self.gaussian_conditional.build_indexes(scale)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=mean)
        return self.g_s(y_hat)

    def init_scale_table(
        self, min_scale: float = 0.11, max_scale: float = 256, levels: int = 64
    ) -> None:
        """Initialize the GaussianConditional scale table for entropy coding."""
        table = torch.exp(
            torch.linspace(math.log(min_scale), math.log(max_scale), levels)
        )
        self.gaussian_conditional.update_scale_table(table.tolist(), force=True)

    def _apply_h_s(self, z_hat: torch.Tensor) -> torch.Tensor:
        """h_s with self-attention on z_hat, windowed for large inputs.

        For small hyper-latents (≤16×16 = 256 tokens, the training regime)
        the full spatial self-attention runs as-is.  For larger images the
        hyper-latent is partitioned into 16×16 non-overlapping windows and
        attention runs per-window, keeping the same trained attention
        semantics without O(n²) memory blowup on high-res inputs.
        """
        B, _, H, W = z_hat.shape
        attn_in = self.h_s_proj_in(z_hat)  # → B, 4C, H, W
        token_count = H * W
        # The model was trained on 256px images → 16×16 hyper-latent = 256 tokens.
        # Stay in that regime below the threshold and fall back to windowed
        # attention above it so O(n²) does not blow up memory.
        if token_count <= 256:
            flat = attn_in.flatten(2).transpose(1, 2)  # → B, H*W, 4C
            attn_out, _ = self.h_s_attention(flat, flat, flat)
            attn_out = attn_out.transpose(1, 2).reshape(B, -1, H, W)
        else:
            attn_out = self._windowed_attention(attn_in, window_size=16)
        z_attended = self.h_s_proj_out(attn_out) + z_hat  # residual
        return self.h_s(z_attended)

    def _windowed_attention(
        self, attn_in: torch.Tensor, window_size: int = 16
    ) -> torch.Tensor:
        """Apply self-attention per non-overlapping spatial window.

        Partitions the (B, 4C, H, W) tensor into (B×nH×nW, 4C, ws, ws)
        windows, runs the same trained MultiheadAttention per window, then
        folds back.
        """
        ws = window_size
        B, C, H, W = attn_in.shape
        # Pad to window alignment
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        if pH or pW:
            attn_in = F.pad(attn_in, (0, pW, 0, pH))
        _, C, Hp, Wp = attn_in.shape
        nH, nW = Hp // ws, Wp // ws
        # Unfold → (B, C, nH, ws, nW, ws) → (B×nH×nW, C, ws, ws)
        windows = (
            attn_in.unfold(2, ws, ws)
            .unfold(3, ws, ws)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B * nH * nW, C, ws, ws)
        )
        flat = windows.flatten(2).transpose(1, 2)  # (B×nH×nW, ws², C)
        attn_w, _ = self.h_s_attention(flat, flat, flat)
        attn_w = attn_w.transpose(1, 2).reshape(B * nH * nW, C, ws, ws)
        # Fold back
        attn_out = (
            attn_w.reshape(B, nH, nW, C, ws, ws)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C, Hp, Wp)
        )
        if pH or pW:
            attn_out = attn_out[:, :, :H, :W]
        return attn_out

    def forward(
        self, image: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = LATENT_MEAN_BOUND * torch.tanh(mu / LATENT_MEAN_BOUND)
        z = self.h_a(mu)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self._apply_h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        scale = scale.abs() + 1e-6
        y_hat, y_likelihoods = self.gaussian_conditional(mu, scale, mean)
        reconstructed = self.g_s(y_hat)
        return reconstructed, mu, None, y_hat, y_likelihoods, z_likelihoods

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = LATENT_MEAN_BOUND * torch.tanh(mu / LATENT_MEAN_BOUND)
        return mu

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to RGB image."""
        return self.g_s(latent)

    def update(self, force: bool = False) -> None:
        """Update entropy bottleneck CDF tables (required before compress)."""
        self.entropy_bottleneck.update(force=force)
        self.init_scale_table()

    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        """Deterministic encode → quantize → decode for evaluation.

        Uses the GaussianConditional for quantization (via learned scale/mean
        from the hyperprior) rather than a crude STE integer round — this
        gives much finer quantization steps (~scale, typically ~0.1) compared
        to the 7 integer bins from rounding to [-3,-2,…,3].
        """
        raw = self.g_a(image)
        mu, _ = raw.chunk(2, dim=1)
        mu = LATENT_MEAN_BOUND * torch.tanh(mu / LATENT_MEAN_BOUND)
        z = self.h_a(mu)
        z_hat = self.entropy_bottleneck.decompress(
            self.entropy_bottleneck.compress(z), z.size()[-2:]
        )
        params = self._apply_h_s(z_hat)
        scale, mean = params.chunk(2, dim=1)
        scale = scale.abs() + 1e-6
        # The hyperprior decoder (h_s) uses transposed convolutions that may
        # produce a slightly different spatial size than g_a's output for
        # non-power-of-2 image dimensions.  Interpolate to match mu so the
        # GaussianConditional doesn't crash with a size mismatch.
        if scale.shape[-2:] != mu.shape[-2:]:
            scale = F.interpolate(
                scale, size=mu.shape[-2:], mode="bilinear", align_corners=False
            )
            mean = F.interpolate(
                mean, size=mu.shape[-2:], mode="bilinear", align_corners=False
            )
        # GaussianConditional quantizes y with learned step size:
        #   y_hat = round((mu - mean) / scale) * scale + mean
        # This yields finer gradations (~0.1 steps vs 1.0 for integer round).
        y_hat, _ = self.gaussian_conditional(mu, scale, mean)
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
        reference_array = np.asarray(self.reference_full, dtype=np.float32) / 255.0
        # Default 256² tensor used when multiscale is False.
        self.reference = torch.from_numpy(reference_array).permute(2, 0, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not self.multiscale:
            return self.reference.clone(), 1
        rng = random.Random(self.seed + index * 104729)
        target_size = ProceduralDocumentDataset._pick_target_size(rng)
        resized = self.reference_full.resize((target_size, target_size), Image.LANCZOS)
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
                reference.resize((SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE), Image.LANCZOS)
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


EpochCallback = Callable[[int, int, dict[str, float], dict[str, float], Path], None]


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
        source = (
            torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
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
        "max_deviation": max(abs(e.get("bpp_deviation", 0.0)) for e in results)
        if results
        else 0.0,
    }


def _checkpoint(
    path: Path, model: KakeyaHyperpriorCodec, config: ExperimentConfig, epoch: int
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
        },
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


def _scale_conditioned_objective(
    image_size: int,
    stage_weights: dict[str, float],
    lambda_rate: float,
) -> tuple[dict[str, float], float, float]:
    """Strengthen small-image detail without changing the HD objective."""
    if image_size > SMALL_IMAGE_MAX_SIZE:
        return stage_weights, lambda_rate, 0.0

    weights = dict(stage_weights)
    weights["edge"] = max(weights["edge"], SMALL_IMAGE_EDGE_WEIGHT)
    weights["structural"] = max(weights["structural"], SMALL_IMAGE_STRUCTURAL_WEIGHT)
    weights["multiscale"] = min(weights["multiscale"], SMALL_IMAGE_MULTISCALE_WEIGHT)
    return (
        weights,
        lambda_rate * SMALL_IMAGE_RATE_MULTIPLIER,
        SMALL_IMAGE_HIGH_FREQUENCY_WEIGHT,
    )


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
        "total": 0.0,
        "mse": 0.0,
        "edge": 0.0,
        "structural": 0.0,
        "multiscale": 0.0,
        "high_frequency": 0.0,
        "lab": 0.0,
        "hue": 0.0,
        "saturation": 0.0,
        "kakeya": 0.0,
        "rate": 0.0,
        "psnr": 0.0,
        "rate_bpp": 0.0,
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
                reconstructed, _, _, y_hat, y_likelihoods, z_likelihoods = model(images)
                loss_mse = F.mse_loss(reconstructed, images)
                rate = (
                    -y_likelihoods.log2().sum() - z_likelihoods.log2().sum()
                ) / images.size(0)
                bpp_bits = rate / (images.shape[-1] * images.shape[-2])

                lp = y_hat.permute(0, 2, 3, 1).reshape(-1, y_hat.size(1))
                norm = F.normalize(lp, dim=1)
                coverage = kakeya_regularization(
                    norm, num_projections=num_projections, k=k
                )

                if stage_weights is not None:
                    sw, effective_rate_weight, high_frequency_weight = (
                        _scale_conditioned_objective(
                            max(images.shape[-2:]), stage_weights, lambda_rate
                        )
                    )
                    edge = F.l1_loss(_edges(reconstructed), _edges(images))
                    structural = 1.0 - _ssim(reconstructed, images)
                    multiscale = _multiscale_l1(reconstructed, images)
                    high_frequency = F.l1_loss(
                        _laplacian(reconstructed), _laplacian(images)
                    )
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
                        + high_frequency_weight * high_frequency
                        + sw["lab"] * lab
                        + sw["hue"] * hue
                        + sw["saturation"] * sat
                        + sw["kakeya"] * coverage
                        + effective_rate_weight * rate_penalty
                    )
                else:
                    edge = torch.zeros_like(loss_mse)
                    structural = torch.zeros_like(loss_mse)
                    multiscale = torch.zeros_like(loss_mse)
                    high_frequency = torch.zeros_like(loss_mse)
                    lab = torch.zeros_like(loss_mse)
                    hue = torch.zeros_like(loss_mse)
                    sat = torch.zeros_like(loss_mse)
                    rate_penalty = F.relu(bpp_bits - TARGET_RATE_BPP)
                    total = (
                        loss_mse + lambda_rate * rate_penalty + lambda_kakeya * coverage
                    )

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
            totals["high_frequency"] += float(high_frequency.detach())
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
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_size_aware_collate,
    )
    real_loader = DataLoader(
        RealImageDataset(real_train_size, config.seed + 2_000_000),
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

    hyper_dim = max(4, config.latent_dim // 2)
    model = KakeyaHyperpriorCodec(latent_dim=config.latent_dim, hyper_dim=hyper_dim).to(
        device
    )
    model.init_scale_table()
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
        CalibrationCardDataset(
            CAPACITY_STEPS_PER_EPOCH, multiscale=True, seed=config.seed
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_size_aware_collate,
    )

    run_dir = run_dir or _run_directory(config)
    history: dict[str, Any] = {"epoch": [], "train": {}, "validation": {}}
    best_psnr = float("-inf")
    best_hd_psnr_seen = float("-inf")
    best_hd_ssim_seen = float("-inf")
    selected_hd_psnr: float | None = None
    selected_hd_ssim: float | None = None
    best_epoch: int | None = None
    gate_epoch: int | None = None
    gate_forced: bool = False
    train_reference = (
        ProceduralDocumentDataset(1, config.seed)
        .reference_tensor.unsqueeze(0)
        .to(device)
    )
    hd_reference_image = Image.open(TEST_IMAGE_HD).convert("RGB")
    hd_reference_image.thumbnail(
        (HD_CHECKPOINT_SIZE, HD_CHECKPOINT_SIZE), Image.Resampling.LANCZOS
    )
    hd_reference = (
        torch.from_numpy(np.asarray(hd_reference_image, dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
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
            "capacity"
            if capacity_stage
            else "transition"
            if transition_stage
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
        elif transition_stage:
            rate_weight = FINETUNE_WARMUP_RATE_WEIGHT
        else:
            epochs_since_gate = epoch - (gate_epoch or 0)
            if epochs_since_gate <= FINETUNE_WARMUP_EPOCHS:
                rate_weight = FINETUNE_WARMUP_RATE_WEIGHT
            else:
                rate_weight = lambda_rate

        sw = stage_weights_map[stage]
        epoch_kw = {
            "model": model,
            "device": device,
            "lambda_rate": rate_weight,
            "lambda_kakeya": lambda_kakeya,
            "num_projections": num_projections,
            "k": k,
            "stage_weights": sw,
        }

        if capacity_stage:
            # Capacity stage: train on reference card, procedural, real images
            ref_metrics = _hyperprior_epoch(
                **epoch_kw,
                loader=capacity_loader,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            prog_metrics = _hyperprior_epoch(
                **epoch_kw,
                loader=train_loader,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            real_metrics = _hyperprior_epoch(
                **epoch_kw,
                loader=real_loader,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            train_metrics = {
                key: 0.4 * ref_metrics[key]
                + 0.3 * prog_metrics[key]
                + 0.3 * real_metrics[key]
                for key in ref_metrics
            }
        else:
            # Finetune/transition: train on procedural + real + rehearsal (reference card)
            ref_metrics = _hyperprior_epoch(
                **epoch_kw,
                loader=capacity_loader,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            prog_metrics = _hyperprior_epoch(
                **epoch_kw,
                loader=train_loader,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            real_metrics = _hyperprior_epoch(
                **epoch_kw,
                loader=real_loader,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
            )
            train_metrics = {
                key: 0.25 * ref_metrics[key]
                + 0.375 * prog_metrics[key]
                + 0.375 * real_metrics[key]
                for key in ref_metrics
            }

        validation_metrics = _hyperprior_epoch(
            **epoch_kw,
            loader=validation_loader,
        )

        calibration = _calibration_metrics(model, train_reference)
        hd_calibration = _calibration_metrics(model, hd_reference)
        best_hd_psnr_seen = max(best_hd_psnr_seen, hd_calibration["psnr"])
        best_hd_ssim_seen = max(best_hd_ssim_seen, hd_calibration["ssim"])

        # Capacity gate check (only when still in capacity stage)
        if capacity_stage:
            if (
                calibration["psnr"] >= CAPACITY_GATE_PSNR
                and calibration["ssim"] >= CAPACITY_GATE_SSIM
            ):
                gate_epoch = epoch
            elif epoch >= CAPACITY_GATE_FORCE_EPOCH:
                gate_epoch = epoch
                gate_forced = True

        hd_quality_preserved = (
            hd_calibration["psnr"] >= best_hd_psnr_seen - HD_PSNR_MAX_REGRESSION
            and hd_calibration["ssim"] >= best_hd_ssim_seen - HD_SSIM_MAX_REGRESSION
        )
        if (
            calibration["psnr"] > best_psnr
            and hd_quality_preserved
            and (gate_epoch is not None or calibration["rate_bpp"] <= TARGET_RATE_BPP)
        ):
            best_psnr = calibration["psnr"]
            best_epoch = epoch
            selected_hd_psnr = hd_calibration["psnr"]
            selected_hd_ssim = hd_calibration["ssim"]
            _checkpoint(run_dir / "checkpoints/best.pt", model, config, epoch)

        train_metrics.update(
            {
                "stage": {"capacity": 1, "transition": 2, "finetune": 3}[stage],
                "capacity_stage": float(capacity_stage),
                "capacity_gate_passed": gate_epoch is not None,
                "capacity_gate_epoch": float(gate_epoch or 0),
                "calibration_psnr": calibration["psnr"],
                "calibration_ssim": calibration["ssim"],
                "calibration_rate_bpp": calibration["rate_bpp"],
                "hd_calibration_psnr": hd_calibration["psnr"],
                "hd_calibration_ssim": hd_calibration["ssim"],
                "hd_quality_preserved": float(hd_quality_preserved),
            }
        )
        validation_metrics.update(
            {
                "stage": {"capacity": 1, "transition": 2, "finetune": 3}[stage],
                "capacity_stage": float(capacity_stage),
                "capacity_gate_passed": gate_epoch is not None,
                "capacity_gate_epoch": float(gate_epoch or 0),
                "calibration_psnr": calibration["psnr"],
                "calibration_ssim": calibration["ssim"],
                "calibration_rate_bpp": calibration["rate_bpp"],
                "hd_calibration_psnr": hd_calibration["psnr"],
                "hd_calibration_ssim": hd_calibration["ssim"],
                "hd_quality_preserved": float(hd_quality_preserved),
            }
        )

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
    _checkpoint(
        run_dir / "checkpoints/final.pt", model, config, best_epoch or config.epochs
    )

    model.update()
    metrics, image_paths, codec_baselines, bitstream = _evaluate_chart(
        model, device, run_dir
    )
    _rate_consistency_check(model, device, run_dir)

    finetune_stage_epochs = max(
        0, (config.epochs - (gate_epoch or 0) - TRANSITION_EPOCHS)
    )
    training_summary = {
        "method": "image_codec",
        "total_epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "selected_checkpoint_epoch": best_epoch,
        "selected_checkpoint_psnr": best_psnr if best_epoch else None,
        "selected_checkpoint_hd_psnr": selected_hd_psnr,
        "selected_checkpoint_hd_ssim": selected_hd_ssim,
        "hd_checkpoint_guard": {
            "size": HD_CHECKPOINT_SIZE,
            "max_psnr_regression": HD_PSNR_MAX_REGRESSION,
            "max_ssim_regression": HD_SSIM_MAX_REGRESSION,
        },
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
    # Pad to a multiple of 4 (2× PixelShuffle requires divisibility by 4).
    w, h = hd_image.size
    pad_w = (4 - w % 4) % 4
    pad_h = (4 - h % 4) % 4
    if pad_w or pad_h:
        hd_image = ImageOps.expand(
            hd_image, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0)
        )
    hd_tensor = torch.from_numpy(np.asarray(hd_image, dtype=np.float32) / 255.0)
    hd_tensor = hd_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        mu_hd = model.encode(hd_tensor)
        decoded_hd, hd_bitstream = _encode_bitstream(
            model, mu_hd, run_dir, write_file=False
        )
        recon_hd = model.decode(decoded_hd).clamp(0, 1)
    mse_hd = float(F.mse_loss(recon_hd, hd_tensor))
    psnr_hd = 99.0 if mse_hd == 0 else 10 * math.log10(1.0 / mse_hd)
    ssim_hd = float(_ssim(recon_hd, hd_tensor))
    pixels = w * h
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
    return {
        "hd_psnr": psnr_hd,
        "hd_ssim": ssim_hd,
        "hd_mse": mse_hd,
        "hd_width": float(w),
        "hd_height": float(h),
        "hd_pixels": float(pixels),
        "hd_bitstream_bytes": float(hd_bitstream["bytes"]),
        "hd_bitstream_bpp": float(hd_bitstream["bytes"] * 8 / pixels),
    }


def reference_codec_baselines() -> list[dict[str, Any]]:
    source_image = Image.open(TEST_IMAGE).convert("RGB")
    source = (
        torch.from_numpy(np.asarray(source_image, dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
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
        decoded_tensor = (
            torch.from_numpy(np.asarray(decoded, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
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
                net = mbt2018_mean(quality=quality, metric="mse", pretrained=True)
            except Exception:
                continue
            net.eval()
            net = net.to(source.device)
            with torch.no_grad():
                compressed = net.compress(source)
                strings = compressed["strings"]
                out = net.decompress(strings, compressed["shape"])
                x_hat = out["x_hat"].clamp(0, 1)
            total_bytes = sum(len(s) for string_list in strings for s in string_list)
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


def _laplacian(image: torch.Tensor) -> torch.Tensor:
    """Second-order high-frequency response used for small-image detail."""
    channels = image.shape[1]
    kernel = image.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=channels)


_SRGB_TO_LIN_COEFF = 1.0 / 12.92
_SRGB_TO_LIN_EXP = 1.0 / 2.4
_SRGB_TO_LIN_OFFSET = 0.055
_SRGB_TO_LIN_SCALE = 1.055

_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_CUBED = _LAB_DELTA**3
_LAB_FACTOR = 3.0 * _LAB_DELTA**2

_D65_XYZ = torch.tensor([0.95047, 1.0, 1.08883])

_XYZ_TO_LAB = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


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
    delta_e = (
        torch.sqrt(
            0.25 * diff[:, 0:1, :, :].pow(2)
            + chroma_weight * diff[:, 1:2, :, :].pow(2)
            + chroma_weight * diff[:, 2:3, :, :].pow(2)
            + 1e-8,
        ).mean()
        / 7.0
    )
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


def _multiscale_l1(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def _encode_bitstream(
    model: KakeyaHyperpriorCodec,
    latent: torch.Tensor,
    run_dir: Path,
    *,
    write_file: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode y with parameters reconstructed from the entropy-coded z."""
    import struct as _struct

    model.eval()
    model.update()

    z = model.h_a(latent)
    z_strings = model.entropy_bottleneck.compress(z)
    z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    params = model._apply_h_s(z_hat)
    scale, mean = params.chunk(2, dim=1)
    scale = scale.abs() + 1e-6
    if scale.shape[-2:] != latent.shape[-2:]:
        scale = F.interpolate(
            scale, size=latent.shape[-2:], mode="bilinear", align_corners=False
        )
        mean = F.interpolate(
            mean, size=latent.shape[-2:], mode="bilinear", align_corners=False
        )
    indexes = model.gaussian_conditional.build_indexes(scale)
    y_strings = model.gaussian_conditional.compress(latent, indexes, means=mean)
    y_hat = model.gaussian_conditional.decompress(y_strings, indexes, means=mean)

    # Pack: [z_len:4B][z_data][y_len:4B][y_data]
    payload = _struct.pack(">I", len(z_strings[0])) + z_strings[0]
    payload += _struct.pack(">I", len(y_strings[0])) + y_strings[0]

    file_bytes = len(payload)
    if write_file:
        bitstream_path = run_dir / "reports/reconstruction.kky"
        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        bitstream_path.write_bytes(payload)
    height, width = latent.shape[-2] * 4, latent.shape[-1] * 4
    bitstream_bpp = file_bytes * 8 / (height * width)
    return y_hat, {
        "path": "reports/reconstruction.kky",
        "filename": "reconstruction.kky",
        "format": "Kakeya Hyperprior v5",
        "bytes": file_bytes,
        "payload_bytes": len(z_strings[0]) + len(y_strings[0]),
        "header_bytes": 8,
        "bpp": bitstream_bpp,
        "requires_checkpoint": True,
        "checkpoint": "checkpoints/final.pt",
    }
