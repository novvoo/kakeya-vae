"""Post-training metrics for latent-space and reconstruction quality."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from kakeya.models import VAE


def compute_dimension_metrics(
    z: np.ndarray, threshold: float = 1e-3
) -> dict[str, float | int | list[float]]:
    variances = np.var(z, axis=0)
    spectrum = np.sort(variances)[::-1]
    total_variance = float(spectrum.sum())
    active_dims = int(np.count_nonzero(variances > threshold))
    if total_variance > 0:
        effective_dim = total_variance**2 / float(np.square(spectrum).sum())
        top_k_ratio = float(spectrum[: min(5, len(spectrum))].sum() / total_variance)
    else:
        effective_dim = 0.0
        top_k_ratio = 0.0
    return {
        "active_dims": active_dims,
        "dim_utilization": active_dims / len(variances),
        "effective_dim_pr": effective_dim,
        "pca_top5_ratio": top_k_ratio,
        "variance_spectrum": spectrum.tolist(),
    }


def compute_reconstruction_metrics(
    model: VAE, dataloader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    mse_values: list[np.ndarray] = []
    bce_values: list[np.ndarray] = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            reconstruction = model.reconstruct(data)
            mse_values.append(
                torch.mean((data - reconstruction).square(), dim=(1, 2, 3))
                .cpu()
                .numpy()
            )
            bce_values.append(
                F.binary_cross_entropy(reconstruction, data, reduction="none")
                .mean(dim=(1, 2, 3))
                .cpu()
                .numpy()
            )
    mse = np.concatenate(mse_values)
    bce = np.concatenate(bce_values)
    return {
        "mse": float(mse.mean()),
        "mse_std": float(mse.std()),
        "bce": float(bce.mean()),
    }


def compute_downstream_classification(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, float | int]:
    x_train, x_test, y_train, y_test = train_test_split(
        z,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    classifier = LinearSVC(max_iter=10_000, dual=False, random_state=random_state)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "num_classes": len(np.unique(labels)),
    }


def evaluate_all(
    model: VAE,
    dataloader: DataLoader,
    z: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, float | int | list[float]]:
    results = compute_dimension_metrics(z)
    results.update(compute_reconstruction_metrics(model, dataloader, device))
    results.update(compute_downstream_classification(z, labels))
    return results
