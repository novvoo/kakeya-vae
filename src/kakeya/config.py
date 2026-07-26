"""Typed experiment configuration and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_METHODS = {
    "baseline",
    "beta_vae",
    "beta_tcvae",
    "factor_vae",
    "image_codec",
    "poly_kakeya",
}


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    epochs: int = 25
    latent_dim: int = 32
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42
    data_dir: Path = Path("data")
    output_dir: Path = Path("runs")
    num_workers: int = 0
    train_limit: int | None = None
    test_limit: int | None = None
    download: bool = True
    objective: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method {self.method!r}; "
                f"choose one of {sorted(SUPPORTED_METHODS)}"
            )
        for name in ("epochs", "latent_dim", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("train_limit", "test_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    values["data_dir"] = Path(values.get("data_dir", "data"))
    values["output_dir"] = Path(values.get("output_dir", "runs"))
    return ExperimentConfig(**values)
