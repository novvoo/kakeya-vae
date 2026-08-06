"""Command-line encode/decode tool for the Kakeya learned image codec.

Supports the v9 architecture (Kakeya Hyperprior v11 bitstream) with the
channel-wise autoregressive entropy model (Minnen 2020). The Detail latent
is split into two channel groups: group 1 is decoded from the hyperprior,
group 2 is conditioned on group 1's quantized values via
``ChannelGroupContext``.

The ``.kky`` file produced by this CLI is self-describing and differs from
the internal ``reports/reconstruction.kky`` written by ``_encode_bitstream``
(which is a bare 4-segment payload paired with ``final.pt`` in memory).
This CLI format embeds a JSON header so it can be decoded standalone:

    BITSTREAM_MAGIC + [header_len:4B] + JSON_header + [z][y1][y2][base]

where each segment is prefixed by a 4-byte big-endian length.

Usage:
  python scripts/codec_cli.py encode --checkpoint runs/.../checkpoints/final.pt \
      --input input.png --output output.kky
  python scripts/codec_cli.py decode --checkpoint runs/.../checkpoints/final.pt \
      --input output.kky --output reconstruction.png
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kakeya.image_codec import (  # noqa: E402
    BITSTREAM_MAGIC,
    CODEC_ALIGNMENT,
    CODEC_ARCHITECTURE_VERSION,
    CODEC_BITSTREAM_FORMAT,
    KakeyaHyperpriorCodec,
    load_codec_model_state,
)


def load_model(checkpoint_path: Path, device: torch.device) -> KakeyaHyperpriorCodec:
    """Load a KakeyaHyperpriorCodec from a training checkpoint.

    Rejects checkpoints whose architecture version predates the current
    channel-grouped hyperprior (v9 / Kakeya Hyperprior v11).
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    ckpt_version = architecture.get("version")
    if ckpt_version != CODEC_ARCHITECTURE_VERSION:
        raise SystemExit(
            f"检查点架构版本 v{ckpt_version} 与当前主干 v{CODEC_ARCHITECTURE_VERSION}"
            f"（{CODEC_BITSTREAM_FORMAT}，含通道维分组上下文）不兼容，"
            "请用当前主干重新训练"
        )
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    latent_dim = config.get("latent_dim", 8) if isinstance(config, dict) else 8
    hyper_dim = int(architecture.get("hyper_dim", max(8, latent_dim)))
    model = KakeyaHyperpriorCodec(latent_dim=latent_dim, hyper_dim=hyper_dim).to(device)
    load_codec_model_state(model, payload["model_state_dict"])
    model.init_scale_table()
    model.update()
    model.eval()
    return model


def _pad_to_alignment(image: Image.Image) -> tuple[Image.Image, int, int]:
    """Replicate edges so Haar / SCH windows stay aligned at boundaries."""
    w, h = image.size
    pad_w = (CODEC_ALIGNMENT - w % CODEC_ALIGNMENT) % CODEC_ALIGNMENT
    pad_h = (CODEC_ALIGNMENT - h % CODEC_ALIGNMENT) % CODEC_ALIGNMENT
    if not pad_w and not pad_h:
        return image, 0, 0
    padded = Image.fromarray(
        np.pad(np.asarray(image), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    )
    return padded, pad_w, pad_h


@torch.no_grad()
def encode_image(
    model: KakeyaHyperpriorCodec,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict[str, float]:
    """Encode a PNG/JPG image into a self-describing ``.kky`` bitstream."""
    source_image = Image.open(input_path).convert("RGB")
    original_w, original_h = source_image.size
    if original_w < 16 or original_h < 16:
        raise SystemExit("图片尺寸过小（下限 16×16）")
    if original_w > 4096 or original_h > 4096:
        raise SystemExit("图片尺寸过大（上限 4096×4096），请先缩小后再试")

    aligned_image, pad_w, pad_h = _pad_to_alignment(source_image)
    source = torch.from_numpy(np.asarray(aligned_image, dtype=np.float32) / 255.0)
    source = source.permute(2, 0, 1).unsqueeze(0).to(device)

    compressed = model.compress(source)
    y_strings1, y_strings2, z_strings, base_strings = compressed["strings"]
    z_shape = list(compressed["shape"])
    y_shape = list(compressed["y_shape"])
    base_shape = list(compressed["base_shape"])

    # 4-segment payload: [z_len:4B][z][y1_len:4B][y1][y2_len:4B][y2][base_len:4B][base]
    payload = b""
    for segment in (z_strings[0], y_strings1[0], y_strings2[0], base_strings[0]):
        payload += struct.pack(">I", len(segment)) + segment

    header = json.dumps(
        {
            "format": CODEC_BITSTREAM_FORMAT,
            "architecture_version": CODEC_ARCHITECTURE_VERSION,
            "latent_channels": model.latent_dim,
            "z_shape": z_shape,
            "y_shape": y_shape,
            "base_shape": base_shape,
            "image_shape": [original_w, original_h, 3],
            "padding": [pad_w, pad_h],
            "requires_model_checkpoint": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    packaged = (
        BITSTREAM_MAGIC + struct.pack(">I", len(header)) + header + payload
    )
    output_path.write_bytes(packaged)

    # Round-trip in-memory to report PSNR on the (padded) reconstruction.
    reconstructed = model.decompress(
        [y_strings1, y_strings2, z_strings, base_strings],
        tuple(z_shape),
        tuple(y_shape),
        tuple(base_shape),
    ).clamp(0, 1)
    mse = float(torch.nn.functional.mse_loss(reconstructed, source))
    psnr = 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)

    file_bytes = len(packaged)
    bpp = file_bytes * 8 / (original_w * original_h)
    return {"bytes": file_bytes, "bpp": bpp, "psnr": psnr}


@torch.no_grad()
def decode_image(
    model: KakeyaHyperpriorCodec,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    """Decode a ``.kky`` bitstream back to a PNG image."""
    packaged = input_path.read_bytes()
    magic_len = len(BITSTREAM_MAGIC)
    if not packaged.startswith(BITSTREAM_MAGIC) or len(packaged) < magic_len + 4:
        raise SystemExit("无效的 Kakeya 码流（magic 不匹配）")
    header_len = struct.unpack(">I", packaged[magic_len : magic_len + 4])[0]
    header_end = magic_len + 4 + header_len
    if header_end >= len(packaged):
        raise SystemExit("无效的 Kakeya 码流（header 截断）")
    header = json.loads(packaged[magic_len + 4 : header_end])
    if header.get("format") != CODEC_BITSTREAM_FORMAT:
        raise SystemExit(
            f"码流格式 {header.get('format')} 与当前 {CODEC_BITSTREAM_FORMAT} 不兼容"
        )
    z_shape = tuple(header["z_shape"])
    y_shape = tuple(header["y_shape"])
    base_shape = tuple(header["base_shape"])
    original_w, original_h, _ = header["image_shape"]
    pad_w, pad_h = header.get("padding", [0, 0])

    # Read 4 length-prefixed segments: z, y1, y2, base.
    cursor = header_end
    segments: list[bytes] = []
    for _ in range(4):
        if cursor + 4 > len(packaged):
            raise SystemExit("无效的 Kakeya 码流（段长度截断）")
        seg_len = struct.unpack(">I", packaged[cursor : cursor + 4])[0]
        cursor += 4
        if cursor + seg_len > len(packaged):
            raise SystemExit("无效的 Kakeya 码流（段数据截断）")
        segments.append(packaged[cursor : cursor + seg_len])
        cursor += seg_len
    z_bytes, y1_bytes, y2_bytes, base_bytes = segments

    # model.decompress expects strings as [[per-batch bytes]]; batch size is 1.
    strings = [[y1_bytes], [y2_bytes], [z_bytes], [base_bytes]]
    reconstructed = model.decompress(strings, z_shape, y_shape, base_shape).clamp(0, 1)

    # Strip the alignment padding that was added during encode.
    if pad_w or pad_h:
        reconstructed = reconstructed[:, :, :original_h, :original_w]
    array = reconstructed[0].detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8), mode="RGB").save(
        output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kakeya learned image codec CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode a PNG/JPG image to .kky")
    encode_parser.add_argument("--checkpoint", required=True, type=Path)
    encode_parser.add_argument("--input", required=True, type=Path)
    encode_parser.add_argument("--output", required=True, type=Path)

    decode_parser = subparsers.add_parser("decode", help="Decode a .kky file to PNG")
    decode_parser.add_argument("--checkpoint", required=True, type=Path)
    decode_parser.add_argument("--input", required=True, type=Path)
    decode_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = load_model(args.checkpoint, device)

    if args.command == "encode":
        stats = encode_image(model, args.input, args.output, device)
        print(f"Encoded: {args.output}")
        print(f"  Format:  {CODEC_BITSTREAM_FORMAT} (v{CODEC_ARCHITECTURE_VERSION})")
        print(f"  Size:    {stats['bytes']} bytes ({stats['bpp']:.4f} bpp)")
        print(f"  PSNR:    {stats['psnr']:.2f} dB")
    elif args.command == "decode":
        decode_image(model, args.input, args.output, device)
        print(f"Decoded: {args.output}")


if __name__ == "__main__":
    main()
