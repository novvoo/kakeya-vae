"""Typed experiment configuration and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
DEFAULT_STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    "capacity": {
        "mse": 3.0,
        "edge": 1.0,
        "structural": 0.2,
        "multiscale": 0.2,
        "lab": 0.05,
        "hue": 0.05,
        "saturation": 0.05,
        "kakeya": 0.002,
    },
    "transition": {
        "mse": 2.0,
        "edge": 1.5,
        "structural": 0.4,
        "multiscale": 0.3,
        "lab": 0.08,
        "hue": 0.08,
        "saturation": 0.08,
        "kakeya": 0.001,
    },
    "finetune": {
        "mse": 3.0,
        "edge": 2.0,
        "structural": 0.6,
        "multiscale": 0.4,
        "lab": 0.12,
        "hue": 0.06,
        "saturation": 0.08,
        "kakeya": 0.0005,
    },
}


SUPPORTED_METHODS = {
    "baseline",
    "beta_vae",
    "beta_tcvae",
    "factor_vae",
    "hyperprior_kakeya",
    "image_codec",
}


# Default per-stage loss weights for image_codec training.  Every loss
# direction participates from the first epoch; stages only differ in how
# much weight each loss carries.  Users can override any subset of these
# via config.objective.stage_weights in YAML / API.
#
# Rationale:
#   capacity   — build latent structure, light pixel constraints
#   transition — smooth ramp to pixel accuracy after gate opens
#   finetune   — maximize perceptual quality (SSIM, edges, color)
@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    epochs: int = 25
    latent_dim: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42
    data_dir: Path = Path("data")
    output_dir: Path = Path("runs")
    num_workers: int = 0
    train_limit: int | None = None
    test_limit: int | None = None
    download: bool = True
    # objective is an open map so that method-specific parameters (e.g.
    # num_projections, k) and the nested stage_weights table can both live
    # here without schema churn.  stage_weights, when present, must be a
    # mapping of stage -> {loss_name: weight}; missing stages / losses fall
    # back to DEFAULT_STAGE_WEIGHTS.
    objective: dict[str, Any] = field(default_factory=dict)

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

    def stage_weights(self) -> dict[str, dict[str, float]]:
        """Return per-stage loss weights, merging user overrides from config.objective.stage_weights."""
        user_overrides = self.objective.get("stage_weights", {})
        merged: dict[str, dict[str, float]] = {}
        for stage, defaults in DEFAULT_STAGE_WEIGHTS.items():
            overrides = user_overrides.get(stage, {})
            merged[stage] = {**defaults, **overrides}
        return merged


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    values["data_dir"] = Path(values.get("data_dir", "data"))
    values["output_dir"] = Path(values.get("output_dir", "runs"))
    return ExperimentConfig(**values)
