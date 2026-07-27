"""Subprocess entry point used by the web experiment service."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from kakeya.config import ExperimentConfig
from kakeya.data import get_mnist_dataloaders
from kakeya.evaluation import evaluate_all
from kakeya.image_codec import train_image_codec, _run_directory


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False),
        flush=True,
    )


def run(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    config_values = dict(spec["config"])
    config_values["data_dir"] = Path(config_values.get("data_dir", "data"))
    config_values["output_dir"] = Path(config_values.get("output_dir", "runs"))
    config = ExperimentConfig(**config_values)
    requested_device = spec.get("device", "auto")
    device = _resolve_device(requested_device)

    emit(
        "started",
        method=config.method,
        epochs=config.epochs,
        device=str(device),
        message="训练进程已启动",
    )

    def on_epoch(
        epoch: int,
        total_epochs: int,
        train: dict[str, float],
        validation: dict[str, float],
        run_dir: Path,
    ) -> None:
        stage_num = train.get("stage", 1)
        phase = "容量预训练" if stage_num == 1 else "过渡阶段" if stage_num == 2 else "压缩微调"
        gate = " · 闸门已通过" if train.get("capacity_gate_passed") else ""
        emit(
            "epoch",
            epoch=epoch,
            total_epochs=total_epochs,
            progress=epoch / total_epochs,
            train=train,
            validation=validation,
            run_dir=str(run_dir),
            message=f"{phase} · 第 {epoch}/{total_epochs} 轮完成{gate}",
        )

    run_dir = _run_directory(config)
    try:
        image_result = train_image_codec(
            config,
            device,
            epoch_callback=on_epoch,
            run_dir=run_dir,
        )
        dashboard = {
            "config": config.to_dict(),
            "history": image_result.history,
            "metrics": image_result.metrics,
            "latent": [],
            "runtime": {"device": str(device)},
            "image_codec": {
                "image_size": 256,
                "test_asset": "Kakeya Codec Test Card v2",
                "test_role": "in_distribution_calibration",
                "latent_shape": [config.latent_dim, 32, 32],
                "images": image_result.images,
                "training": image_result.training_summary,
                "bitstream": image_result.bitstream,
            },
            "codec_baselines": image_result.codec_baselines,
        }
        dashboard_path = run_dir / "reports" / "dashboard.json"
        dashboard_path.write_text(
            json.dumps(dashboard, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit(
            "completed",
            progress=1.0,
            run_dir=str(run_dir),
            result_path=str(dashboard_path),
            metrics=image_result.metrics,
            message="训练完成（KakeyaHyperpriorCodec）",
        )
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        error_dashboard = {
            "config": config.to_dict(),
            "history": {"epoch": [], "train": {}, "validation": {}},
            "metrics": {"psnr": 0, "ssim": 0},
            "latent": [],
            "runtime": {"device": str(device)},
            "image_codec": {
                "image_size": 256,
                "test_asset": "Kakeya Codec Test Card v2",
                "test_role": "in_distribution_calibration",
                "latent_shape": [config.latent_dim, 32, 32],
                "images": {},
                "training": {
                    "error": tb,
                    "capacity_gate_passed": False,
                    "capacity_gate": {"psnr": 0, "ssim": 0},
                    "final_stage": "failed",
                },
                "bitstream": {},
            },
            "codec_baselines": [],
            "error": tb,
        }
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        dashboard_path = run_dir / "reports" / "dashboard.json"
        dashboard_path.write_text(
            json.dumps(error_dashboard, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit(
            "failed",
            error=str(exc),
            traceback=tb,
            message="训练失败",
        )
        raise

    result = train_model(
        config,
        device=device,
        progress=False,
        epoch_callback=on_epoch,
    )
    resolved_device = next(result.model.parameters()).device
    emit("evaluating", message="正在计算最终指标")
    _, test_loader = get_mnist_dataloaders(
        config.batch_size,
        config.data_dir,
        num_workers=config.num_workers,
        download=False,
        seed=config.seed,
        test_limit=config.test_limit,
    )
    metrics = evaluate_all(
        result.model,
        test_loader,
        result.test_z,
        result.test_labels,
        resolved_device,
    )
    dashboard = _dashboard_payload(
        config,
        result.history,
        metrics,
        result.test_z,
        result.test_labels,
        str(resolved_device),
    )
    dashboard_path = result.run_dir / "reports" / "dashboard.json"
    dashboard_path.write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    emit(
        "completed",
        progress=1.0,
        run_dir=str(result.run_dir),
        result_path=str(dashboard_path),
        metrics=metrics,
        message="训练与评估已完成",
    )
    return 0


def _dashboard_payload(
    config: ExperimentConfig,
    history: dict[str, Any],
    metrics: dict[str, Any],
    z: np.ndarray,
    labels: np.ndarray,
    device: str,
) -> dict[str, Any]:
    sample_size = min(800, len(z))
    indices = np.linspace(0, len(z) - 1, sample_size, dtype=int)
    sampled_z = z[indices]
    projected = PCA(n_components=2, random_state=config.seed).fit_transform(sampled_z)
    latent = [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "label": int(label),
        }
        for point, label in zip(projected, labels[indices])
    ]
    return {
        "config": config.to_dict(),
        "history": history,
        "metrics": metrics,
        "latent": latent,
        "runtime": {"device": device},
    }


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args(argv)
    try:
        return run(args.spec)
    except KeyboardInterrupt:
        emit("cancelled", message="训练已停止")
        return 130
    except (OSError, ValueError, RuntimeError, TypeError, LookupError) as error:
        emit(
            "failed",
            error=str(error),
            traceback=traceback.format_exc(),
            message="训练失败",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
