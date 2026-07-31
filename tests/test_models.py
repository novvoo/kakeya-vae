from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kakeya.image_codec import (
    BlendedInstanceNorm,
    DepthToSpace,
    KakeyaHyperpriorCodec,
    SpaceToDepth,
    _encode_bitstream,
    _hyperprior_epoch,
    _lab_losses,
    _laplacian,
    _optimizer_parameter_groups,
    _rgb_to_lab,
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


@pytest.mark.parametrize("value", [0.0, 1e-8, 0.5, 1.0])
def test_rgb_to_lab_has_finite_gradients_at_neutral_values(value: float) -> None:
    image = torch.full((1, 3, 4, 4), value, requires_grad=True)

    _rgb_to_lab(image).sum().backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_rgb_to_lab_matches_standard_d65_reference_colors() -> None:
    # CIE 1976 L*a*b*, 2° observer, D65 white point.  These reference
    # values make the sRGB transfer exponent, RGB-to-XYZ matrix, and white
    # point observable instead of testing only self-consistency.
    srgb = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).view(5, 3, 1, 1)
    expected_lab = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [100.0000, 0.0000, 0.0000],
            [53.2408, 80.0925, 67.2032],
            [87.7347, -86.1827, 83.1793],
            [32.2970, 79.1875, -107.8602],
        ]
    )

    actual_lab = _rgb_to_lab(srgb).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(actual_lab, expected_lab, rtol=0.0, atol=0.02)


def test_lab_losses_have_finite_gradients_for_achromatic_pixels() -> None:
    reconstructed = torch.zeros(1, 3, 4, 4, requires_grad=True)
    target = torch.zeros_like(reconstructed)

    sum(_lab_losses(reconstructed, target).values()).backward()

    assert reconstructed.grad is not None
    assert torch.isfinite(reconstructed.grad).all()


def test_kakeya_hyperprior_forward_pass() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4)
    x = torch.rand(1, 3, 128, 128)
    recon, mu, _, _, y_likelihoods, _ = model(x)
    assert recon.shape == (1, 3, 128, 128)
    assert mu.shape == (1, 4, 32, 32)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(y_likelihoods).all()


def test_custom_backbone_detail_head_receives_gradients() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)
    image = torch.rand(1, 3, 32, 32)

    reconstruction, *_ = model(image)
    reconstruction.mean().backward()

    assert model.hyper_dim == 8
    assert model.detail_head[-1].weight.grad is not None
    assert torch.isfinite(model.detail_head[-1].weight.grad).all()
    assert model.g_a[-1].out_channels == 4


def test_blended_instance_norm_preserves_raw_statistics_path() -> None:
    layer = BlendedInstanceNorm(3, initial_strength=0.9)
    value = torch.rand(2, 3, 8, 8) * 2 + 3

    output = layer(value)
    expected = value + layer.strength * (layer.norm(value) - value)
    output.mean().backward()

    assert torch.allclose(output, expected)
    assert float(layer.strength.mean().detach()) == pytest.approx(0.9)
    assert layer.strength_logit.grad is not None
    assert torch.isfinite(layer.strength_logit.grad).all()


def test_entropy_quantiles_are_only_in_auxiliary_optimizer() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)

    main_parameters, auxiliary_parameters = _optimizer_parameter_groups(model)

    assert auxiliary_parameters
    assert not (
        {id(p) for p in main_parameters} & {id(p) for p in auxiliary_parameters}
    )
    assert {id(p) for p in auxiliary_parameters} == {
        id(p) for name, p in model.named_parameters() if name.endswith(".quantiles")
    }


def test_entropy_cdf_remains_finite_after_training_step() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)
    main_parameters, auxiliary_parameters = _optimizer_parameter_groups(model)
    optimizer = torch.optim.AdamW(main_parameters, lr=1e-3)
    auxiliary_optimizer = torch.optim.Adam(auxiliary_parameters, lr=1e-3)
    images = torch.rand(1, 3, 32, 32)
    loader = DataLoader(TensorDataset(images, torch.zeros(1)), batch_size=1)

    metrics = _hyperprior_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        aux_optimizer=auxiliary_optimizer,
    )
    model.update(force=True)

    assert torch.isfinite(torch.tensor(metrics["total"]))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
