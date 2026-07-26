"""Loss functions and objective composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, float]
    reconstruction: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    z: torch.Tensor


def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(reconstruction, target, reduction="sum") / target.size(0)


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp()) / mu.size(0)


def total_correlation(
    mu: torch.Tensor, log_var: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    return (_log_aggregate_posterior(z, mu, log_var) - _log_product_marginals(
        z, mu, log_var
    )).mean()


def dimensional_kl(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + log_var - mu.square() - log_var.exp())).mean(dim=0)


def kakeya_regularization(
    z: torch.Tensor, num_projections: int = 64, k: int = 100
) -> torch.Tensor:
    """Encourage coverage along random directions.

    The negative sign is intentional: optimizers minimize the objective, so a
    negative mean gap maximizes projection coverage. The previous implementation
    added a positive gap and therefore collapsed the projections.
    """

    if z.size(0) < 2:
        return z.new_zeros(())
    directions = torch.randn(num_projections, z.size(1), device=z.device)
    directions = F.normalize(directions, dim=1)
    sorted_projections = torch.sort(z @ directions.T, dim=0).values
    gaps = sorted_projections.diff(dim=0)
    top_k = min(k, gaps.size(0))
    return -torch.topk(gaps, k=top_k, dim=0).values.mean()


def polynomial_kakeya_regularization(
    z: torch.Tensor, degree: int = 3, num_projections: int = 64
) -> torch.Tensor:
    if degree <= 0:
        raise ValueError("degree must be positive")
    normalized_z = F.normalize(z, dim=1)
    directions = F.normalize(
        torch.randn(num_projections, z.size(1), device=z.device), dim=1
    )
    projections = normalized_z @ directions.T
    polynomial_features = torch.cat(
        [projections.pow(power) for power in range(1, degree + 1)], dim=1
    )
    return -polynomial_features.var(dim=0, unbiased=False).mean()


def compute_objective(
    model: torch.nn.Module,
    x: torch.Tensor,
    method: str = "baseline",
    **parameters: float,
) -> LossOutput:
    reconstruction, mu, log_var, z = model(x)
    recon = reconstruction_loss(reconstruction, x)
    kl = kl_divergence(mu, log_var)
    tc = z.new_zeros(())
    regularizer = z.new_zeros(())

    if method == "baseline":
        total = recon + kl
    elif method == "beta_vae":
        total = recon + float(parameters.get("beta", 4.0)) * kl
    elif method == "beta_tcvae":
        tc = total_correlation(mu, log_var, z)
        beta = float(parameters.get("beta", 4.0))
        total = recon + kl + (beta - 1.0) * tc
    elif method == "factor_vae":
        # The discriminator-based TC term is added by the training engine.
        total = recon + kl
    elif method == "poly_kakeya":
        regularizer = polynomial_kakeya_regularization(
            z,
            degree=int(parameters.get("degree", 3)),
            num_projections=int(parameters.get("num_projections", 64)),
        )
        total = recon + kl + float(parameters.get("lambda_kakeya", 1.0)) * regularizer
    else:
        raise ValueError(f"Unknown method: {method}")

    return LossOutput(
        total=total,
        components={
            "total": float(total.detach()),
            "reconstruction": float(recon.detach()),
            "kl": float(kl.detach()),
            "total_correlation": float(tc.detach()),
            "kakeya": float(regularizer.detach()),
        },
        reconstruction=reconstruction,
        mu=mu,
        log_var=log_var,
        z=z,
    )


def _log_aggregate_posterior(
    z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    batch_size, latent_dim = z.shape
    difference = z[:, None, :] - mu[None, :, :]
    log_probability = -0.5 * (
        latent_dim * np.log(2 * np.pi)
        + log_var[None, :, :].sum(dim=2)
        + (difference.square() * torch.exp(-log_var[None, :, :])).sum(dim=2)
    )
    return torch.logsumexp(log_probability, dim=1) - np.log(batch_size)


def _log_product_marginals(
    z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    difference = z[:, None, :] - mu[None, :, :]
    per_dimension = -0.5 * (
        np.log(2 * np.pi)
        + log_var[None, :, :]
        + difference.square() * torch.exp(-log_var[None, :, :])
    )
    return (
        torch.logsumexp(per_dimension, dim=1) - np.log(z.size(0))
    ).sum(dim=1)


# Compatibility alias for old callers.
compute_losses = compute_objective
