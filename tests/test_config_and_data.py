from pathlib import Path

from kakeya.config import load_config
from kakeya.data import TrustedMNIST


def test_config_loads_paths() -> None:
    config = load_config(Path("configs/image-codec.yaml"))

    assert config.method == "image_codec"
    assert config.data_dir == Path("data")
    assert config.objective["num_projections"] == 32


def test_mnist_uses_only_pytorch_mirror() -> None:
    assert TrustedMNIST.mirrors == [
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ]
