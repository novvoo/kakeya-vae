import pytest

from kakeya.config import ExperimentConfig


def test_legacy_vae_methods_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported method"):
        ExperimentConfig(method="factor_vae")
