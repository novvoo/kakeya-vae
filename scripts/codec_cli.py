"""Command-line encode/decode tool for Kakeya learned image codec.

Usage:
  python scripts/codec_cli.py encode --checkpoint runs/image_codec/TIMESTAMP/checkpoints/final.pt \
      --input input.png --output output.kky
  python scripts/codec_cli.py decode --checkpoint runs/image_codec/TIMESTAMP/checkpoints/final.pt \
      --input output.kky --output reconstruction.png
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kakeya.image_codec import BITSTREAM_MAGIC, ImageCodecVAE  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> ImageCodecVAE:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = payload["config"].get("latent_dim", 8)
    model = ImageCodecVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def encode_image(
    model: ImageCodecVAE,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict[str, float]:
    source_image = Image.open(input_path).convert("RGB")
    if source_image.size != (256, 256):
        source_image = source_image.resize((256, 256), Image.LANCZOS)
    source = torch.from_numpy(np.asarray(source_image, dtype=np.float32) / 255.0)
    source = source.permute(2, 0, 1).unsqueeze(0).to(device)

    mu, _ = model.encode(source)
    entropy_model = deepcopy(model.entropy_bottleneck).cpu().eval()
    entropy_model.update(force=True)
    latent_cpu = mu.detach().cpu()
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
    output_path.write_bytes(packaged)

    decoded_latent = entropy_model.decompress([payload], shape).to(device)
    reconstructed = model.decode(decoded_latent).clamp(0, 1)
    mse = float(torch.nn.functional.mse_loss(reconstructed, source))
    psnr = 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)

    file_bytes = len(packaged)
    bpp = file_bytes * 8 / (256 * 256)
    return {"bytes": file_bytes, "bpp": bpp, "psnr": psnr}


@torch.no_grad()
def decode_image(
    model: ImageCodecVAE,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    packaged = input_path.read_bytes()
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
        or not all(isinstance(v, int) and v > 0 for v in shape)
        or channels != model.latent_dim
    ):
        raise ValueError(
            f"bitstream latent_channels={channels} does not match model latent_dim={model.latent_dim}"
        )

    entropy_model = deepcopy(model.entropy_bottleneck).cpu().eval()
    entropy_model.update(force=True)
    decoded_latent = entropy_model.decompress([packaged[header_end:]], shape).to(device)
    reconstructed = model.decode(decoded_latent).clamp(0, 1)

    array = reconstructed[0].detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8), mode="RGB").save(
        output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kakeya learned image codec CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode a PNG image to .kky")
    encode_parser.add_argument("--checkpoint", required=True, type=Path)
    encode_parser.add_argument("--input", required=True, type=Path)
    encode_parser.add_argument("--output", required=True, type=Path)

    decode_parser = subparsers.add_parser("decode", help="Decode a .kky file to PNG")
    decode_parser.add_argument("--checkpoint", required=True, type=Path)
    decode_parser.add_argument("--input", required=True, type=Path)
    decode_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.command == "encode":
        stats = encode_image(model, args.input, args.output, device)
        print(f"Encoded: {args.output}")
        print(f"  Size:    {stats['bytes']} bytes ({stats['bpp']:.4f} bpp)")
        print(f"  PSNR:    {stats['psnr']:.2f} dB")
    elif args.command == "decode":
        decode_image(model, args.input, args.output, device)
        print(f"Decoded: {args.output}")


if __name__ == "__main__":
    main()
