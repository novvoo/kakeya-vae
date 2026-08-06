from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import kakeya.image_codec as image_codec_module
from kakeya.image_codec import (
    BASE_LATENT_CHANNELS,
    CODEC_ARCHITECTURE_VERSION,
    CODEC_BITSTREAM_FORMAT,
    ChannelGroupContext,
    CONTEXT_GROUP_COUNT,
    TRAIN_MIX_CYCLE,
    BlendedInstanceNorm,
    DepthToSpace,
    FixedHaarAnalysis,
    FixedHaarSynthesis,
    KakeyaHyperpriorCodec,
    LightweightSCHBlock,
    SpaceToDepth,
    WindowChannelAttention,
    _checkpoint,
    _compose_base_detail,
    _detail_ycocg,
    _encode_bitstream,
    _flat_region_high_frequency_loss,
    _hyperprior_epoch,
    _kakeya_dimension_proxy,
    _kakeya_tube_loss,
    _kakeya_tube_responses,
    _lab_losses,
    _laplacian,
    _low_frequency_base,
    _optimizer_parameter_groups,
    _restore_low_frequency_base,
    _rgb_to_lab,
    _rgb_to_ycocg,
    _scale_conditioned_objective,
    _training_step_count,
    _ycocg_to_rgb,
    load_codec_model_state,
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
    recon, _, _, _, yl, _, base_likelihoods = model(image)
    assert recon.shape == image.shape
    assert mu.shape == (1, 8, 64, 64)
    assert mu.abs().max() <= 3.0
    assert yl.shape == (1, 8, 64, 64)
    assert torch.isfinite(yl).all()
    assert base_likelihoods.shape == (1, BASE_LATENT_CHANNELS, 32, 32)


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


def test_fixed_haar_transform_is_exactly_reversible() -> None:
    analysis = FixedHaarAnalysis(3)
    synthesis = FixedHaarSynthesis(3)
    image = torch.randn(2, 3, 32, 48)

    coefficients = analysis(image)
    reconstructed = synthesis(coefficients)

    assert coefficients.shape == (2, 12, 16, 24)
    assert not list(analysis.parameters())
    assert not list(synthesis.parameters())
    torch.testing.assert_close(reconstructed, image, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("pattern", "expected_band"),
    [
        (torch.tensor([[1.0, -1.0], [1.0, -1.0]]), 1),
        (torch.tensor([[1.0, 1.0], [-1.0, -1.0]]), 2),
        (torch.tensor([[1.0, -1.0], [-1.0, 1.0]]), 3),
    ],
)
def test_haar_subbands_separate_edge_directions(
    pattern: torch.Tensor, expected_band: int
) -> None:
    image = pattern.repeat(4, 4).view(1, 1, 8, 8)
    coefficients = FixedHaarAnalysis(1)(image)
    energies = coefficients.square().mean(dim=(0, 2, 3))

    assert int(energies.argmax()) == expected_band
    assert energies[expected_band] > 0


def test_lightweight_sch_supports_non_aligned_spatial_shapes() -> None:
    block = LightweightSCHBlock(64, window_size=4)
    value = torch.randn(2, 64, 10, 14, requires_grad=True)

    output = block(value)
    output.mean().backward()

    assert output.shape == value.shape
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_window_channel_attention_chunking_preserves_results() -> None:
    unchunked = WindowChannelAttention(8, window_size=4, max_windows_per_chunk=64)
    chunked = WindowChannelAttention(8, window_size=4, max_windows_per_chunk=1)
    chunked.load_state_dict(unchunked.state_dict())
    value = torch.randn(1, 8, 10, 14)

    with torch.no_grad():
        expected = unchunked(value)
        actual = chunked(value)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_training_mix_is_real_optimizer_step_ratio() -> None:
    assert len(TRAIN_MIX_CYCLE) == 10
    assert TRAIN_MIX_CYCLE.count("reference") == 4
    assert TRAIN_MIX_CYCLE.count("procedural") == 3
    assert TRAIN_MIX_CYCLE.count("real") == 3
    assert _training_step_count(128) == 100
    assert _training_step_count(32) == 30


def test_hyperprior_rate_is_computable() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4)
    image = torch.rand(1, 3, 256, 256)
    _, _, _, _, yl, zl, cl = model(image)
    total_rate = (-yl.log().sum() - zl.log().sum() - cl.log().sum()) / image.size(0)
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
    decoded, metadata = _encode_bitstream(model, image, tmp_path)

    assert decoded.shape == image.shape
    assert torch.isfinite(decoded).all()
    assert metadata["format"] == "Kakeya Hyperprior v11"
    assert metadata["bytes"] == (tmp_path / "reports/reconstruction.kky").stat().st_size
    assert metadata["bytes"] > metadata["header_bytes"]
    assert metadata["base_bytes"] > 4


def test_base_stream_restores_low_frequency_luminance_and_chroma() -> None:
    target = torch.zeros(1, 3, 32, 32)
    target[:, 0] = 0.2
    target[:, 1] = 0.7
    target[:, 2] = 0.3
    base = torch.full_like(target, 0.4)
    decoded_base = torch.nn.functional.adaptive_avg_pool2d(
        _rgb_to_ycocg(target), (4, 4)
    )

    restored = _restore_low_frequency_base(base, decoded_base)
    base_error = torch.nn.functional.l1_loss(_rgb_to_ycocg(base), _rgb_to_ycocg(target))
    restored_error = torch.nn.functional.l1_loss(
        _rgb_to_ycocg(restored), _rgb_to_ycocg(target)
    )
    target_luminance = (target[:, 0] + 2 * target[:, 1] + target[:, 2]) / 4
    restored_luminance = (restored[:, 0] + 2 * restored[:, 1] + restored[:, 2]) / 4

    assert restored_error < base_error * 0.01
    torch.testing.assert_close(restored_luminance, target_luminance)


def test_ycocg_base_fusion_preserves_detail_high_pass() -> None:
    reconstructed = torch.full((1, 3, 32, 32), 0.4)
    checkerboard = ((torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2).float()
    reconstructed += (checkerboard * 0.1 - 0.05)[None, None]
    decoded_base = torch.zeros(1, 3, 4, 4)
    decoded_base[:, 0] = 0.5

    restored = _restore_low_frequency_base(reconstructed, decoded_base)
    detail_ycocg = _rgb_to_ycocg(reconstructed)
    detail_low = torch.nn.functional.interpolate(
        _low_frequency_base(reconstructed, (4, 4)),
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    )
    decoded_up = torch.nn.functional.interpolate(
        decoded_base, size=(32, 32), mode="bilinear", align_corners=False
    )

    torch.testing.assert_close(
        _rgb_to_ycocg(restored) - decoded_up,
        detail_ycocg - detail_low,
        atol=1e-6,
        rtol=1e-5,
    )


def test_ycocg_transform_is_reversible() -> None:
    image = torch.rand(2, 3, 17, 19)
    torch.testing.assert_close(_ycocg_to_rgb(_rgb_to_ycocg(image)), image)


def test_detail_signal_excludes_absolute_brightness() -> None:
    image = torch.rand(1, 3, 32, 32) * 0.6 + 0.2
    brighter = image + 0.1

    torch.testing.assert_close(
        _detail_ycocg(brighter), _detail_ycocg(image), atol=1e-6, rtol=1e-5
    )


def test_laplacian_base_detail_split_is_exactly_reversible() -> None:
    image = torch.rand(1, 3, 32, 32) * 0.8 + 0.1
    base = _low_frequency_base(image, (4, 4))
    detail = _detail_ycocg(image)

    torch.testing.assert_close(
        _compose_base_detail(base, detail), image, atol=1e-6, rtol=1e-5
    )


def test_detail_can_compensate_the_decoded_base_reference() -> None:
    image = torch.rand(1, 3, 32, 32) * 0.6 + 0.2
    decoded_base = _low_frequency_base(image, (4, 4)).clone()
    decoded_base[:, 0] += 0.01

    detail = _detail_ycocg(image, decoded_base)

    torch.testing.assert_close(
        _compose_base_detail(decoded_base, detail), image, atol=1e-6, rtol=1e-5
    )


def test_forward_encodes_detail_against_quantized_base() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8).eval()
    image = torch.rand(1, 3, 32, 32) * 0.6 + 0.2
    base = model._base_analysis(image)
    base_hat, _ = model.base_entropy_bottleneck(base)
    expected = _detail_ycocg(image, model.base_synthesis(base_hat))
    captured: list[torch.Tensor] = []
    handle = model.g_a[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )

    with torch.no_grad():
        model(image)
    handle.remove()

    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)


def test_learned_base_branch_starts_as_full_ycocg_transform() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)
    image = torch.rand(1, 3, 32, 32)
    expected = torch.nn.functional.avg_pool2d(
        _rgb_to_ycocg(image), kernel_size=8, stride=8
    )

    encoded = model._base_analysis(image)
    decoded = model.base_synthesis(encoded)
    decoded.mean().backward()

    torch.testing.assert_close(decoded, expected)
    assert model.base_analysis.projection.weight.grad is not None
    assert model.base_analysis.refinement[-1].weight.grad is not None
    assert model.base_synthesis.refinement[-1].weight.grad is not None
    assert torch.isfinite(model.base_analysis.projection.weight.grad).all()
    assert torch.isfinite(model.base_analysis.refinement[-1].weight.grad).all()
    assert torch.isfinite(model.base_synthesis.refinement[-1].weight.grad).all()


def test_codec_state_loader_rejects_missing_learned_parameters() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)
    state = dict(model.state_dict())
    state.pop("g_a.1.net.1.bias")

    with pytest.raises(RuntimeError, match="missing=.*g_a.1.net.1.bias"):
        load_codec_model_state(model, state)


def test_codec_state_loader_rebuilds_only_entropy_buffers() -> None:
    source = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)
    target = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)

    load_codec_model_state(target, source.state_dict())

    torch.testing.assert_close(target.g_a[1].net[1].weight, source.g_a[1].net[1].weight)


def test_small_image_objective_preserves_hd_weights() -> None:
    weights = {
        "edge": 2.0,
        "structural": 0.6,
        "multiscale": 0.4,
    }

    small_weights, small_rate, high_frequency, flat_region = (
        _scale_conditioned_objective(256, weights, 0.01)
    )
    hd_weights, hd_rate, hd_high_frequency, hd_flat_region = (
        _scale_conditioned_objective(512, weights, 0.01)
    )

    assert small_weights == {
        "edge": 4.0,
        "structural": 1.2,
        "multiscale": 0.2,
    }
    assert small_rate == pytest.approx(0.0035)
    assert high_frequency == 0.5
    assert flat_region == 0.5
    assert hd_weights is weights
    assert hd_rate == 0.01
    assert hd_high_frequency == 0.0
    assert hd_flat_region == 0.0


def test_laplacian_emphasizes_edges_not_flat_regions() -> None:
    flat = torch.ones(1, 3, 16, 16)
    edge = flat.clone()
    edge[:, :, :, 8:] = 0

    flat_response = _laplacian(flat)[:, :, 1:-1, 1:-1]
    edge_response = _laplacian(edge)[:, :, 1:-1, 1:-1]

    assert torch.count_nonzero(flat_response) == 0
    assert edge_response.abs().sum() > 0


def test_flat_region_loss_penalizes_ringing_and_has_finite_gradients() -> None:
    target = torch.full((1, 3, 16, 16), 0.5)
    checkerboard = ((torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2).float()
    reconstructed = (target + (checkerboard * 0.04 - 0.02)[None, None]).requires_grad_()

    loss = _flat_region_high_frequency_loss(reconstructed, target)
    loss.backward()

    assert loss > 0
    assert reconstructed.grad is not None
    assert torch.isfinite(reconstructed.grad).all()


def test_kakeya_tube_loss_is_zero_for_exact_reconstruction() -> None:
    image = torch.rand(1, 3, 32, 32)

    components = _kakeya_tube_loss(image, image, num_directions=12, num_scales=3)

    for value in components.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-7, rtol=0)


def test_kakeya_tube_response_detects_line_orientation() -> None:
    image = torch.ones(1, 3, 32, 32)
    image[:, :, 15:17, :] = 0

    response = _kakeya_tube_responses(image, 12, 1)[0]
    peak_by_direction = response.amax(dim=(-2, -1))[0]

    assert peak_by_direction[0] > peak_by_direction[6] * 2


def test_kakeya_tube_loss_penalizes_unsupported_lines() -> None:
    target = torch.full((1, 3, 32, 32), 0.5)
    reconstructed = target.clone()
    reconstructed[:, :, 15:17, :] = 0.1

    components = _kakeya_tube_loss(
        reconstructed, target, num_directions=12, num_scales=3
    )

    assert components["leakage"] > 0
    assert components["total"] > components["tube"]


def test_kakeya_tube_loss_backpropagates_without_dimension_gradient() -> None:
    target = torch.ones(1, 3, 32, 32)
    target[:, :, :, 15:17] = 0
    reconstructed = (target * 0.9 + 0.05).requires_grad_()

    components = _kakeya_tube_loss(
        reconstructed, target, num_directions=8, num_scales=2
    )
    components["total"].backward()
    dimension = _kakeya_dimension_proxy(reconstructed)

    assert reconstructed.grad is not None
    assert torch.isfinite(reconstructed.grad).all()
    assert torch.isfinite(dimension)
    assert not dimension.requires_grad


def test_hd_epoch_does_not_evaluate_kakeya_tube_loss(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("HD training must not evaluate the small-image tube loss")

    monkeypatch.setattr(image_codec_module, "_kakeya_tube_loss", fail_if_called)
    monkeypatch.setattr(image_codec_module, "_kakeya_dimension_proxy", fail_if_called)
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4)
    images = torch.rand(1, 3, 288, 288)
    loader = DataLoader(TensorDataset(images, torch.zeros(1)), batch_size=1)

    metrics = _hyperprior_epoch(model, loader, torch.device("cpu"))

    assert metrics["kakeya"] == 0.0
    assert metrics["kakeya_dimension"] == 0.0


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
    recon, mu, _, _, y_likelihoods, _, base_likelihoods = model(x)
    assert recon.shape == (1, 3, 128, 128)
    assert mu.shape == (1, 4, 32, 32)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(y_likelihoods).all()
    assert torch.isfinite(base_likelihoods).all()


def test_custom_backbone_detail_head_receives_gradients() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=8)
    image = torch.rand(1, 3, 32, 32)

    reconstruction, *_ = model(image)
    reconstruction.mean().backward()

    assert model.hyper_dim == 8
    assert model.detail_coefficient_refinement[-1].weight.grad is not None
    assert torch.isfinite(model.detail_coefficient_refinement[-1].weight.grad).all()
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


# ---------------------------------------------------------------------------
# Channel-wise autoregressive context (Minnen 2020) — v9 architecture
# ---------------------------------------------------------------------------


def test_channel_group_context_splits_latent_into_equal_groups() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=8, hyper_dim=8)

    assert CONTEXT_GROUP_COUNT == 2
    assert model.group1_channels == 4
    assert model.group2_channels == 4
    assert model.group1_channels + model.group2_channels == model.latent_dim


def test_channel_group_context_rejects_undivisible_latent_dim() -> None:
    with pytest.raises(ValueError, match="divisible by CONTEXT_GROUP_COUNT"):
        KakeyaHyperpriorCodec(latent_dim=7)


def test_channel_group_context_forward_shapes_and_positive_scale() -> None:
    g1, g2 = 3, 5
    ctx = ChannelGroupContext(g1, g2)
    y_hat_g1 = torch.randn(2, g1, 8, 10)
    hyper_scale_g2 = torch.rand(2, g2, 8, 10) + 0.1
    hyper_mean_g2 = torch.randn(2, g2, 8, 10)

    scale_g2, mean_g2 = ctx(y_hat_g1, hyper_scale_g2, hyper_mean_g2)

    assert scale_g2.shape == mean_g2.shape == (2, g2, 8, 10)
    assert (scale_g2 > 0).all()
    assert torch.isfinite(scale_g2).all()
    assert torch.isfinite(mean_g2).all()


def test_channel_group_context_fusion_starts_identity_initialized() -> None:
    """At init, fusion passes hyperprior params through → group 2 starts from v8 behavior.

    This is the critical fix for the 0806 regression: the old zero-init made
    scale_g2≈1e-6 (naked quantization) for 40 epochs. Identity init makes
    scale_g2≈hyper_scale_g2 and mean_g2≈hyper_mean_g2 from step 1.
    """
    g1, g2 = 4, 4
    ctx = ChannelGroupContext(g1, g2)
    # context_proj is zero-init so context contributes nothing at start.
    assert torch.allclose(ctx.context_proj.weight, torch.zeros_like(ctx.context_proj.weight))
    assert torch.allclose(ctx.context_proj.bias, torch.zeros_like(ctx.context_proj.bias))

    y_hat_g1 = torch.randn(1, g1, 4, 4)
    hyper_scale_g2 = torch.rand(1, g2, 4, 4) + 0.1
    hyper_mean_g2 = torch.randn(1, g2, 4, 4)

    scale_g2, mean_g2 = ctx(y_hat_g1, hyper_scale_g2, hyper_mean_g2)

    # Identity init + zero context_proj → scale_g2 ≈ |hyper_scale_g2| + 1e-6,
    # mean_g2 ≈ hyper_mean_g2. Since hyper_scale_g2 > 0, |x|+1e-6 ≈ x.
    torch.testing.assert_close(
        scale_g2, hyper_scale_g2 + 1e-6, atol=1e-7, rtol=1e-6
    )
    torch.testing.assert_close(mean_g2, hyper_mean_g2, atol=1e-7, rtol=1e-6)


def test_channel_group_context_context_net_is_single_3x3_conv() -> None:
    """v9.1 reduces context_net from two 5×5 convs to one 3×3 to curb overfitting."""
    g1, g2 = 4, 4
    ctx = ChannelGroupContext(g1, g2)

    # context_net is a single weight_norm-wrapped Conv2d (no Sequential).
    assert isinstance(ctx.context_net, nn.Module)
    underlying = ctx.context_net
    # The weight_norm parametrization wraps a Conv2d; check kernel size.
    assert hasattr(underlying, "weight")
    assert underlying.weight.shape == (g2 * 2, g1, 3, 3), (
        f"expected (g2*2, g1, 3, 3), got {underlying.weight.shape}"
    )


def test_channel_group_context_backpropagates_to_context_net() -> None:
    g1, g2 = 4, 4
    ctx = ChannelGroupContext(g1, g2)
    y_hat_g1 = torch.randn(1, g1, 8, 8, requires_grad=True)
    hyper_scale_g2 = torch.rand(1, g2, 8, 8) + 0.1
    hyper_mean_g2 = torch.randn(1, g2, 8, 8)

    scale_g2, mean_g2 = ctx(y_hat_g1, hyper_scale_g2, hyper_mean_g2)
    (scale_g2.mean() + mean_g2.mean()).backward()

    # fusion is a plain Conv2d (leaf tensor) so its weight grad is populated.
    assert ctx.fusion.weight.grad is not None
    assert torch.isfinite(ctx.fusion.weight.grad).all()
    # context_proj is also a plain Conv2d; its grad should be populated.
    assert ctx.context_proj.weight.grad is not None
    assert torch.isfinite(ctx.context_proj.weight.grad).all()
    # y_hat_g1 receives gradient through context_net.
    assert y_hat_g1.grad is not None
    assert torch.isfinite(y_hat_g1.grad).all()


def test_forward_uses_two_pass_channel_group_quantization() -> None:
    """forward() must split y_hat into group1 (hyperprior) + group2 (context)."""
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4).eval()
    model.update()
    image = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        reconstructed, mu, _, y_hat, y_likelihoods, _, _ = model(image)

    g1 = model.group1_channels
    assert y_hat.shape == mu.shape == (1, 4, 8, 8)
    assert y_likelihoods.shape == mu.shape
    # Likelihoods for group 1 and group 2 are concatenated along channels.
    assert y_likelihoods[:, :g1].shape == (1, g1, 8, 8)
    assert y_likelihoods[:, g1:].shape == (1, model.group2_channels, 8, 8)
    assert torch.isfinite(y_hat).all()


def test_compress_returns_four_segment_strings() -> None:
    """compress() must produce [y_strings1, y_strings2, z_strings, base_strings]."""
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4).eval()
    model.update()
    image = torch.rand(1, 3, 32, 32)

    compressed = model.compress(image)

    assert len(compressed["strings"]) == 4
    y_strings1, y_strings2, z_strings, base_strings = compressed["strings"]
    assert len(y_strings1[0]) > 0
    assert len(y_strings2[0]) > 0
    assert len(z_strings[0]) > 0
    assert len(base_strings[0]) > 0
    assert isinstance(compressed["shape"], tuple)
    assert isinstance(compressed["y_shape"], tuple)
    assert isinstance(compressed["base_shape"], tuple)


def test_compress_decompress_round_trip_matches_reconstruct() -> None:
    """Range-coded compress → decompress must match reconstruct() deterministically."""
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4).eval()
    model.update()
    image = torch.rand(1, 3, 32, 32)

    compressed = model.compress(image)
    decoded = model.decompress(
        compressed["strings"],
        compressed["shape"],
        compressed["y_shape"],
        compressed["base_shape"],
    )
    with torch.no_grad():
        replay = model.reconstruct(image)

    torch.testing.assert_close(decoded, replay, atol=1e-6, rtol=1e-5)
    assert decoded.shape == image.shape


def test_decompress_rejects_wrong_string_count() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4).eval()
    model.update()
    image = torch.rand(1, 3, 32, 32)
    compressed = model.compress(image)
    # Drop the last segment (base_strings) to simulate a v10 3-segment payload.
    truncated = compressed["strings"][:3]

    with pytest.raises((ValueError, TypeError)):
        model.decompress(
            truncated,
            compressed["shape"],
            compressed["y_shape"],
            compressed["base_shape"],
        )


def test_checkpoint_architecture_records_channel_group_metadata(tmp_path) -> None:
    """_checkpoint must persist v9 architecture + channel-group metadata."""
    from kakeya.config import ExperimentConfig

    model = KakeyaHyperpriorCodec(latent_dim=8, hyper_dim=8)
    config = ExperimentConfig(method="image_codec")
    ckpt_path = tmp_path / "final.pt"
    _checkpoint(ckpt_path, model, config, epoch=1)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]

    assert architecture["version"] == CODEC_ARCHITECTURE_VERSION == 9
    assert architecture["bitstream_format"] == CODEC_BITSTREAM_FORMAT
    assert architecture["channel_group_context"] == "minnen_2020"
    assert architecture["context_group_count"] == CONTEXT_GROUP_COUNT == 2
    assert architecture["group1_channels"] == model.group1_channels == 4
    assert architecture["group2_channels"] == model.group2_channels == 4
    assert architecture["hyper_dim"] == model.hyper_dim
