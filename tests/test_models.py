import torch

from kakeya.image_codec import (
    BITSTREAM_MAGIC,
    DepthToSpace,
    KakeyaHyperpriorCodec,
    SpaceToDepth,
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
    recon, _, _, y_hat, yl, zl = model(image)
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


def test_kakeya_hyperprior_forward_pass() -> None:
    model = KakeyaHyperpriorCodec(latent_dim=4, hyper_dim=4)
    x = torch.rand(1, 3, 128, 128)
    (recon, mu, log_var, y_hat, y_likelihoods, z_likelihoods) = model(x)
    assert recon.shape == (1, 3, 128, 128)
    assert mu.shape == (1, 4, 32, 32)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(y_likelihoods).all()
