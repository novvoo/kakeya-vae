"""Dataset construction with a maintained and checksum-verified MNIST source."""

from __future__ import annotations

import os
import random
import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import ClassVar

import certifi
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity, extract_archive


class TrustedMNIST(MNIST):
    """MNIST using only PyTorch's maintained OSSCI mirror.

    ``torchvision.datasets.MNIST`` still lists the obsolete Yann LeCun HTTP
    endpoint as a fallback. Keeping only the maintained PyTorch mirror avoids a
    misleading 404 while retaining TorchVision's MD5 integrity verification.
    """

    mirrors: ClassVar[list[str]] = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ]

    @property
    def raw_folder(self) -> str:
        # Stay compatible with TorchVision's standard on-disk layout even
        # though this safety-focused subclass has a different Python name.
        return str(Path(self.root) / "MNIST" / "raw")

    @property
    def processed_folder(self) -> str:
        return str(Path(self.root) / "MNIST" / "processed")

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        for filename, md5 in self.resources:
            archive_path = Path(self.raw_folder) / filename
            if not check_integrity(str(archive_path), md5):
                _secure_download(self.mirrors[0] + filename, archive_path, md5)
            extract_archive(str(archive_path), self.raw_folder)


def get_mnist_dataloaders(
    batch_size: int = 128,
    data_dir: str | Path = "data",
    *,
    num_workers: int = 0,
    download: bool = True,
    seed: int = 42,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    root = Path(data_dir)
    train_dataset = TrustedMNIST(
        root=root, train=True, transform=transform, download=download
    )
    test_dataset = TrustedMNIST(
        root=root, train=False, transform=transform, download=download
    )
    generator = torch.Generator().manual_seed(seed)
    if train_limit is not None:
        indices = torch.randperm(len(train_dataset), generator=generator)[
            :train_limit
        ].tolist()
        train_dataset = Subset(
            train_dataset, indices
        )
    if test_limit is not None:
        indices = torch.randperm(len(test_dataset), generator=generator)[
            :test_limit
        ].tolist()
        test_dataset = Subset(
            test_dataset, indices
        )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker if num_workers else None,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=len(train_dataset) % batch_size == 1,
        **common,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _secure_download(url: str, destination: Path, md5: str) -> None:
    """Download with a current CA bundle and atomically publish the file."""

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        url, headers={"User-Agent": "kakeya-vae/0.1"}
    )
    try:
        with (
            urllib.request.urlopen(request, context=context, timeout=60) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if not check_integrity(str(temporary_path), md5):
            raise RuntimeError(f"Checksum verification failed for {url}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
