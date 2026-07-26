import math

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from kakeya.config import ExperimentConfig
from kakeya.models import VAE, Discriminator
from kakeya.training import train_epoch


def test_factor_vae_training_step_is_finite() -> None:
    config = ExperimentConfig(
        method="factor_vae",
        epochs=1,
        batch_size=4,
        latent_dim=4,
        objective={"gamma": 2.0},
    )
    model = VAE(latent_dim=4, hidden_dims=(8, 16, 32, 64))
    discriminator = Discriminator(latent_dim=4, hidden_dim=16)
    loader = DataLoader(
        TensorDataset(torch.rand(4, 1, 32, 32), torch.zeros(4, dtype=torch.long)),
        batch_size=4,
    )

    metrics = train_epoch(
        model,
        loader,
        Adam(model.parameters()),
        torch.device("cpu"),
        config,
        discriminator,
        Adam(discriminator.parameters()),
        progress=False,
    )

    assert all(math.isfinite(value) for value in metrics.values())
