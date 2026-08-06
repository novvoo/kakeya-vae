"""End-to-end tests for scripts/codec_cli.py (v9 / Kakeya Hyperprior v11).

Covers:
- load_model rejects checkpoints with incompatible architecture versions.
- encode → decode round-trip produces a same-size RGB PNG.
- the self-describing ``.kky`` header records the v11 format and 4 segments.
- decode rejects a truncated payload with a clear error.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# scripts/ is not a package; import by path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.codec_cli as cli  # noqa: E402
from kakeya.config import ExperimentConfig  # noqa: E402
from kakeya.image_codec import (  # noqa: E402
    BITSTREAM_MAGIC,
    CODEC_ARCHITECTURE_VERSION,
    CODEC_BITSTREAM_FORMAT,
    KakeyaHyperpriorCodec,
    _checkpoint,
)

_TEST_IMAGE = _PROJECT_ROOT / "assets/test_images/kakeya_codec_card_v2_256.png"


def _make_checkpoint(tmp_path: Path, *, latent_dim: int = 8, version: int | None = None) -> Path:
    """Write a KakeyaHyperpriorCodec checkpoint; optionally fake an old version."""
    model = KakeyaHyperpriorCodec(latent_dim=latent_dim)
    model.init_scale_table()
    model.update()
    config = ExperimentConfig(method="image_codec")
    ckpt_path = tmp_path / "final.pt"
    _checkpoint(ckpt_path, model, config, epoch=1)
    if version is not None and version != CODEC_ARCHITECTURE_VERSION:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        payload["architecture"]["version"] = version
        torch.save(payload, ckpt_path)
    return ckpt_path


def test_load_model_rejects_old_architecture_version(tmp_path: Path) -> None:
    ckpt_path = _make_checkpoint(tmp_path, version=8)

    with pytest.raises(SystemExit, match="v8.*不兼容"):
        cli.load_model(ckpt_path, torch.device("cpu"))


def test_load_model_accepts_current_version(tmp_path: Path) -> None:
    ckpt_path = _make_checkpoint(tmp_path)

    model = cli.load_model(ckpt_path, torch.device("cpu"))

    assert isinstance(model, KakeyaHyperpriorCodec)
    assert model.group1_channels == 4
    assert model.group2_channels == 4


def test_encode_decode_round_trip_produces_same_size_png(tmp_path: Path) -> None:
    ckpt_path = _make_checkpoint(tmp_path)
    model = cli.load_model(ckpt_path, torch.device("cpu"))
    out_kky = tmp_path / "out.kky"
    out_png = tmp_path / "recon.png"

    stats = cli.encode_image(model, _TEST_IMAGE, out_kky, torch.device("cpu"))
    assert out_kky.is_file()
    assert stats["bytes"] == out_kky.stat().st_size
    assert stats["bpp"] > 0

    cli.decode_image(model, out_kky, out_png, torch.device("cpu"))
    assert out_png.is_file()

    original = Image.open(_TEST_IMAGE).convert("RGB")
    recon = Image.open(out_png).convert("RGB")
    assert recon.size == original.size == (256, 256)


def test_kky_header_records_v11_format_and_four_segments(tmp_path: Path) -> None:
    ckpt_path = _make_checkpoint(tmp_path)
    model = cli.load_model(ckpt_path, torch.device("cpu"))
    out_kky = tmp_path / "out.kky"
    cli.encode_image(model, _TEST_IMAGE, out_kky, torch.device("cpu"))

    packaged = out_kky.read_bytes()
    magic_len = len(BITSTREAM_MAGIC)
    assert packaged.startswith(BITSTREAM_MAGIC)
    header_len = struct.unpack(">I", packaged[magic_len : magic_len + 4])[0]
    import json

    header = json.loads(packaged[magic_len + 4 : magic_len + 4 + header_len])
    assert header["format"] == CODEC_BITSTREAM_FORMAT
    assert header["architecture_version"] == CODEC_ARCHITECTURE_VERSION == 9
    assert header["latent_channels"] == 8
    assert header["image_shape"] == [256, 256, 3]
    assert header["padding"] == [0, 0]

    # After the header, 4 length-prefixed segments must follow.
    cursor = magic_len + 4 + header_len
    segments = []
    for _ in range(4):
        seg_len = struct.unpack(">I", packaged[cursor : cursor + 4])[0]
        cursor += 4
        segments.append(packaged[cursor : cursor + seg_len])
        cursor += seg_len
    assert cursor == len(packaged), "trailing bytes after 4 segments"
    assert all(len(seg) > 0 for seg in segments), "every segment must be non-empty"


def test_decode_rejects_truncated_payload(tmp_path: Path) -> None:
    ckpt_path = _make_checkpoint(tmp_path)
    model = cli.load_model(ckpt_path, torch.device("cpu"))
    out_kky = tmp_path / "out.kky"
    cli.encode_image(model, _TEST_IMAGE, out_kky, torch.device("cpu"))

    # Truncate the file so the 4th segment is missing.
    truncated = tmp_path / "truncated.kky"
    truncated.write_bytes(out_kky.read_bytes()[: out_kky.stat().st_size - 50])

    with pytest.raises(SystemExit, match="截断"):
        cli.decode_image(model, truncated, tmp_path / "out.png", torch.device("cpu"))


def test_decode_rejects_wrong_format_header(tmp_path: Path) -> None:
    ckpt_path = _make_checkpoint(tmp_path)
    model = cli.load_model(ckpt_path, torch.device("cpu"))
    out_kky = tmp_path / "out.kky"
    cli.encode_image(model, _TEST_IMAGE, out_kky, torch.device("cpu"))

    # Rewrite the header's format field to simulate an incompatible stream.
    import json

    packaged = out_kky.read_bytes()
    magic_len = len(BITSTREAM_MAGIC)
    header_len = struct.unpack(">I", packaged[magic_len : magic_len + 4])[0]
    header_end = magic_len + 4 + header_len
    header = json.loads(packaged[magic_len + 4 : header_end])
    header["format"] = "Kakeya Hyperprior v999"
    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    new_packaged = (
        BITSTREAM_MAGIC
        + struct.pack(">I", len(new_header))
        + new_header
        + packaged[header_end:]
    )
    bad_kky = tmp_path / "bad.kky"
    bad_kky.write_bytes(new_packaged)

    with pytest.raises(SystemExit, match="不兼容"):
        cli.decode_image(model, bad_kky, tmp_path / "out.png", torch.device("cpu"))
