from pathlib import Path

import pytest
import torch

from kakeya.image_codec import (
    DepthToSpace,
    KakeyaHyperpriorCodec,
    SpaceToDepth,
    _encode_bitstream,
    _laplacian,
    _scale_conditioned_objective,
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


def test_image_codec_forward_pass() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=8)
    model.eval()
    image = torch.rand(1, 3, 256, 256)

    mu = model.encode(image)
    recon, _, _, _, yl, _ = model(image)
    assert recon.shape == image.shape
    assert mu.shape == (1, 8, 64, 64)
    assert mu.abs().max() <= 3.0
    assert yl.shape == (1, 8, 64, 64)
    assert torch.isfinite(yl).all()


def test_image_codec_finite_reconstruction() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4)
    model.eval()
    model.update()
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        recon = model.reconstruct(image)
    assert recon.shape == (1, 3, 256, 256)
    assert torch.isfinite(recon).all()


def test_space_depth_blocks_preserve_expected_shapes() -> None:
    down = SpaceToDepth(3, 12)
    up = DepthToSpace(12, 3)
    image = torch.rand(1, 3, 32, 32)

    assert down(image).shape == (1, 12, 16, 16)
    assert up(down(image)).shape == image.shape


def test_hyperprior_rate_is_computable() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4)
    image = torch.rand(1, 3, 256, 256)
    _, _, _, _, yl, zl = model(image)
    total_rate = (-yl.log().sum() - zl.log().sum()) / image.size(0)
    assert total_rate > 0
    assert torch.isfinite(total_rate)


def test_model_reconstruct_is_deterministic() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4).eval()
    model.update()
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        a = model.reconstruct(image)
        b = model.reconstruct(image)
    assert torch.equal(a, b)
    assert a.shape == (1, 3, 256, 256)


def test_hyperprior_bitstream_round_trip_uses_conditional_model(
    tmp_path: Path,
) -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4).eval()
    model.update()
    image = torch.rand(1, 3, 32, 32)
    latent = model.encode(image)

    decoded, metadata = _encode_bitstream(model, latent, tmp_path)

    assert decoded.shape == latent.shape
    assert torch.isfinite(decoded).all()
    assert metadata["format"] == "Kakeya Hyperprior v5"
    assert metadata["bytes"] == (tmp_path / "reports/reconstruction.kky").stat().st_size
    assert metadata["bytes"] > metadata["header_bytes"]


def test_small_image_objective_preserves_hd_weights() -> None:
    weights = {
        "edge": 2.0,
        "structural": 0.6,
        "multiscale": 0.4,
    }

    small_weights, small_rate, high_frequency = _scale_conditioned_objective(
        256, weights, 0.01
    )
    hd_weights, hd_rate, hd_high_frequency = _scale_conditioned_objective(
        512, weights, 0.01
    )

    assert small_weights == {
        "edge": 5.0,
        "structural": 1.2,
        "multiscale": 0.2,
    }
    assert small_rate == pytest.approx(0.0035)
    assert high_frequency == 1.0
    assert hd_weights is weights
    assert hd_rate == 0.01
    assert hd_high_frequency == 0.0


def test_laplacian_emphasizes_edges_not_flat_regions() -> None:
    flat = torch.ones(1, 3, 16, 16)
    edge = flat.clone()
    edge[:, :, :, 8:] = 0

    flat_response = _laplacian(flat)[:, :, 1:-1, 1:-1]
    edge_response = _laplacian(edge)[:, :, 1:-1, 1:-1]

    assert torch.count_nonzero(flat_response) == 0
    assert edge_response.abs().sum() > 0


def test_kakeya_hyperprior_forward_pass() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4)
    x = torch.rand(1, 3, 128, 128)
    recon, mu, _, _, y_likelihoods, _ = model(x)
    assert recon.shape == (1, 3, 128, 128)
    assert mu.shape == (1, 4, 32, 32)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(y_likelihoods).all()
