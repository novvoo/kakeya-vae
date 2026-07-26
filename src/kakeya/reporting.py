"""Comparison report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS: dict[str, tuple[str, bool]] = {
    "mse": ("Reconstruction MSE", False),
    "bce": ("Reconstruction BCE", False),
    "active_dims": ("Active dimensions", True),
    "dim_utilization": ("Dimension utilization", True),
    "effective_dim_pr": ("Effective dimension (PR)", True),
    "pca_top5_ratio": ("Top-5 variance ratio", False),
    "accuracy": ("Linear classification accuracy", True),
}


def generate_comparison_report(
    results: dict[str, dict[str, Any]], output_dir: str | Path
) -> Path:
    if not results:
        raise ValueError("results must not be empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    methods = list(results)
    available_metrics = [
        metric
        for metric in METRICS
        if any(metric in method_results for method_results in results.values())
    ]

    lines = ["VAE regularization comparison", "=" * 80, ""]
    header = ["metric", *methods]
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))
    for metric in available_metrics:
        label, _ = METRICS[metric]
        values = [results[method].get(metric) for method in methods]
        lines.append(
            " | ".join(
                [label, *[_format_value(value) for value in values]]
            )
        )

    report_path = destination / "comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_metrics(results, available_metrics, destination)
    _plot_spectra(results, destination)
    return report_path


def _plot_metrics(
    results: dict[str, dict[str, Any]], metrics: list[str], destination: Path
) -> None:
    if not metrics:
        return
    methods = list(results)
    columns = 3
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, metric in zip(axes_array, metrics):
        label, higher_is_better = METRICS[metric]
        values = [float(results[method].get(metric, np.nan)) for method in methods]
        bars = axis.bar(methods, values)
        axis.set_title(f"{label} ({'higher' if higher_is_better else 'lower'} is better)")
        axis.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
            )
    for axis in axes_array[len(metrics) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(destination / "metric-comparison.png", dpi=150)
    plt.close(figure)


def _plot_spectra(results: dict[str, dict[str, Any]], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = False
    for method, method_results in results.items():
        spectrum = np.asarray(method_results.get("variance_spectrum", []))
        total = spectrum.sum()
        if spectrum.size and total > 0:
            axis.plot(np.arange(1, len(spectrum) + 1), np.cumsum(spectrum) / total, label=method)
            plotted = True
    if plotted:
        axis.set_xlabel("Number of latent dimensions")
        axis.set_ylabel("Cumulative variance ratio")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(destination / "variance-spectrum.png", dpi=150)
    plt.close(figure)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
