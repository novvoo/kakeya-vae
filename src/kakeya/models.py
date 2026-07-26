"""Neural-network modules used by the experiments."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one channel size")

        modules: list[nn.Module] = []
        current_channels = in_channels
        for channels in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2),
                )
            )
            current_channels = channels

        self.encoder = nn.Sequential(*modules)
        self.output_channels = hidden_dims[-1]
        self.output_size = 2
        flattened_size = self.output_channels * self.output_size**2
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(flattened_size, latent_dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(encoded), self.fc_log_var(encoded)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        out_channels: int = 1,
        hidden_dims: Sequence[int] = (256, 128, 64, 32),
        initial_size: int = 2,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one channel size")

        self.initial_channels = hidden_dims[0]
        self.initial_size = initial_size
        self.decoder_input = nn.Linear(
            latent_dim, self.initial_channels * initial_size**2
        )

        modules: list[nn.Module] = []
        for input_channels, output_channels in itertools.pairwise(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        input_channels,
                        output_channels,
                        3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_input(z)
        decoded = decoded.view(
            -1, self.initial_channels, self.initial_size, self.initial_size
        )
        return self.final_layer(self.decoder(decoded))


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = ConvDecoder(
            latent_dim, in_channels, tuple(reversed(hidden_dims))
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruct(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, log_var)
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
