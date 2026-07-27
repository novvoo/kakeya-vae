import numpy as np
import torch

from kakeya.evaluation import compute_dimension_metrics
from kakeya.models import VAE
from kakeya.objectives import compute_objective, kakeya_regularization


def test_kakeya_regularizer_has_minimization_sign() -> None:
    torch.manual_seed(0)
    compact = torch.randn(128, 8) * 0.01
    torch.manual_seed(1)
    spread = torch.randn(128, 8)

    torch.manual_seed(2)
    compact_loss = kakeya_regularization(compact)
    torch.manual_seed(2)
    spread_loss = kakeya_regularization(spread)

    assert spread_loss < compact_loss


def test_all_objectives_are_finite() -> None:
    model = VAE(latent_dim=4, hidden_dims=(8, 16, 32, 64))
    x = torch.rand(3, 1, 32, 32)
    methods = (
        "baseline",
        "beta_vae",
        "beta_tcvae",
        "factor_vae",
    )

    for method in methods:
        result = compute_objective(model, x, method)
        assert torch.isfinite(result.total)


def test_participation_ratio_matches_effective_dimension() -> None:
    rng = np.random.default_rng(0)
    isotropic = rng.normal(size=(20_000, 8))

    metrics = compute_dimension_metrics(isotropic)

    assert 7.8 < metrics["effective_dim_pr"] <= 8.0
