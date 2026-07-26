import torch

from kakeya.image_codec import (
    BITSTREAM_MAGIC,
    DepthToSpace,
    ImageCodecVAE,
    SpaceToDepth,
    _encode_bitstream,
    reference_codec_baselines,
)
from kakeya.models import VAE


def test_vae_supports_custom_hidden_dimensions() -> None:
    model = VAE(latent_dim=8, hidden_dims=(16, 32, 64, 128))
    x = torch.rand(2, 1, 32, 32)

    reconstruction, mu, log_var, z = model(x)

    assert reconstruction.shape == x.shape
    assert mu.shape == log_var.shape == z.shape == (2, 8)


def test_reconstruction_is_deterministic_by_default() -> None:
    model = VAE(latent_dim=4)
    model.eval()
    x = torch.rand(1, 1, 32, 32)

    assert torch.equal(model.reconstruct(x), model.reconstruct(x))


def test_image_codec_uses_native_rgb_256_shape() -> None:
    model = ImageCodecVAE(latent_dim=8)
    model.eval()
    image = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        mu, log_var = model.encode(image)
        reconstruction = model.reconstruct(image)

    assert reconstruction.shape == image.shape
    assert mu.shape == log_var.shape == (1, 8, 32, 32)
    assert mu.abs().max() <= 3.0
    assert log_var.min() >= -6.0
    assert log_var.max() <= 2.0


def test_image_codec_zero_noise_matches_mean_reconstruction() -> None:
    model = ImageCodecVAE(latent_dim=4)
    model.eval()
    image = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        sampled, _, _, latent = model(image, noise_scale=0.0)
        mean_reconstruction = model.reconstruct(image)

    assert torch.equal(sampled, mean_reconstruction)
    assert torch.isfinite(latent).all()


def test_space_depth_blocks_preserve_expected_shapes() -> None:
    down = SpaceToDepth(3, 12)
    up = DepthToSpace(12, 3)
    image = torch.rand(1, 3, 32, 32)

    assert down(image).shape == (1, 12, 16, 16)
    assert up(down(image)).shape == image.shape


def test_entropy_bottleneck_produces_a_decodable_payload() -> None:
    model = ImageCodecVAE(latent_dim=4).eval()
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        latent, _ = model.encode(image)
        model.entropy_bottleneck.update(force=True)
        strings = model.entropy_bottleneck.compress(latent)
        decoded = model.entropy_bottleneck.decompress(
            strings, latent.shape[-2:]
        )

    assert strings[0]
    assert decoded.shape == latent.shape
    assert torch.equal(decoded, latent.round())


def test_packaged_bitstream_is_written_and_decoded(tmp_path) -> None:
    model = ImageCodecVAE(latent_dim=4).eval()
    (tmp_path / "reports").mkdir()
    latent = torch.randn(1, 4, 32, 32)

    decoded, metadata = _encode_bitstream(model, latent, tmp_path)
    packaged = (tmp_path / metadata["path"]).read_bytes()

    assert packaged.startswith(BITSTREAM_MAGIC)
    assert metadata["bytes"] == len(packaged)
    assert metadata["payload_bytes"] < metadata["bytes"]
    assert decoded.shape == latent.shape


def test_reference_codec_baselines_use_same_image() -> None:
    baselines = reference_codec_baselines()

    assert [item["codec"] for item in baselines] == [
        "Original PNG",
        "PNG",
        "JPEG",
        "WebP",
    ]
    assert baselines[0]["ssim"] == 1.0
    assert all(item["bytes"] > 0 for item in baselines)
