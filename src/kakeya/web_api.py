"""FastAPI application for the browser-based experiment console."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from kakeya.config import DEFAULT_STAGE_WEIGHTS, ExperimentConfig
from kakeya.image_codec import TEST_IMAGE
from kakeya.job_manager import JobManager

manager = JobManager()


class ExperimentRequest(BaseModel):
    method: Literal["image_codec"] = "image_codec"
    epochs: Annotated[int, Field(ge=1, le=500)] = 80
    latent_dim: Annotated[int, Field(ge=2, le=256)] = 8
    batch_size: Annotated[int, Field(ge=1, le=2048)] = 4
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.0005
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 42
    num_workers: Annotated[int, Field(ge=0, le=32)] = 0
    train_limit: Annotated[int, Field(ge=0, le=60_000)] = 128
    test_limit: Annotated[int, Field(ge=0, le=10_000)] = 0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_projections: Annotated[int, Field(ge=4, le=1024)] = 32
    k: Annotated[int, Field(ge=1, le=4096)] = 3
    lambda_rate: Annotated[float, Field(gt=0, le=100)] = 0.01
    lambda_kakeya: Annotated[float, Field(ge=0, le=10)] = 0.001

    @model_validator(mode="after")
    def validate_device(self) -> ExperimentRequest:
        import torch

        minimum_samples = 32 if self.method == "image_codec" else 100
        if self.train_limit and self.train_limit < minimum_samples:
            raise ValueError(f"训练样本上限应为 0 或至少 {minimum_samples}")
        if self.test_limit and self.test_limit < 100:
            raise ValueError("测试样本上限应为 0 或至少 100")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前环境没有可用的 CUDA 设备")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("当前环境没有可用的 MPS 设备")
        if self.method == "image_codec" and self.latent_dim > 32:
            raise ValueError("空间潜在通道应在 2 到 32 之间")
        return self

    def experiment_config(self) -> dict[str, Any]:
        objective: dict[str, Any] = {
            "num_projections": min(self.num_projections, 128),
            "k": min(self.k, max(self.batch_size - 1, 1)),
            "lambda_rate": self.lambda_rate,
            "lambda_kakeya": self.lambda_kakeya,
        }
        # Default stage_weights are managed server-side via DEFAULT_STAGE_WEIGHTS.
        # Users can override specific weights via the objective.stage_weights field
        # when constructing requests programmatically.  The web UI sends the defaults
        # to make them visible before training starts.
        stage_weights = (
            DEFAULT_STAGE_WEIGHTS.copy() if self.method == "image_codec" else {}
        )
        if stage_weights:
            objective["stage_weights"] = stage_weights
        config = ExperimentConfig(
            method=self.method,
            epochs=self.epochs,
            latent_dim=self.latent_dim,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            num_workers=self.num_workers,
            train_limit=self.train_limit or None,
            test_limit=self.test_limit or None,
            data_dir="data",
            output_dir="runs",
            download=True,
            objective=objective,
        )
        return config.to_dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    yield
    manager.shutdown()


app = FastAPI(
    title="Kakeya Lab API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            os.environ.get("KAKEYA_UI_ORIGIN", "http://localhost:3000"),
        }
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/environment")
def environment() -> dict[str, Any]:
    return manager.environment()


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    return {
        "method": "image_codec",
        "stage_weights": DEFAULT_STAGE_WEIGHTS,
    }


@app.get("/api/test-image")
def test_image() -> FileResponse:
    return FileResponse(TEST_IMAGE, media_type="image/png", filename=TEST_IMAGE.name)


@app.post("/api/environment/install", status_code=202)
def install_dependencies() -> dict[str, str]:
    manager.install_dependencies()
    return {"status": "accepted"}


@app.get("/api/experiments")
def list_experiments() -> list[dict[str, Any]]:
    return manager.list()


@app.post("/api/experiments", status_code=202)
def create_experiment(request: ExperimentRequest) -> dict[str, Any]:
    record = manager.create(request.experiment_config(), request.device)
    return record.public()


@app.get("/api/experiments/{job_id}")
def get_experiment(job_id: str) -> dict[str, Any]:
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return record.public()


@app.post("/api/experiments/{job_id}/stop", status_code=202)
def stop_experiment(job_id: str) -> dict[str, Any]:
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return manager.stop(job_id).public()


@app.get("/api/experiments/{job_id}/result")
def experiment_result(job_id: str) -> dict[str, Any]:
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    result = manager.result(job_id)
    if result is None:
        raise HTTPException(status_code=409, detail="实验结果尚未生成")
    return result


@app.get("/api/experiments/{job_id}/image/{kind}")
def experiment_image(
    job_id: str,
    kind: Literal[
        "original",
        "reconstruction",
        "error",
        "original_hd",
        "reconstruction_hd",
        "error_hd",
    ],
) -> FileResponse:
    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    result = manager.result(job_id)
    image_codec = result.get("image_codec", {}) if result else {}
    relative_path = image_codec.get("images", {}).get(kind)
    if not relative_path:
        raise HTTPException(status_code=404, detail="该实验没有图像重建结果")
    run_dir = (manager.project_root / record.run_dir).resolve()
    image_path = (run_dir / relative_path).resolve()
    if run_dir not in image_path.parents or not image_path.is_file():
        raise HTTPException(status_code=404, detail="图像结果不存在")
    return FileResponse(image_path, media_type="image/png")


@app.post("/api/experiments/{job_id}/regenerate", status_code=200)
def regenerate_experiment(job_id: str) -> dict[str, Any]:
    """Re-run chart reconstruction on the trained checkpoint.

    Loads the model checkpoint, runs encode → decode on the standard
    test chart (256² and HD), overwrites the static report images and
    bitstream, then returns updated metrics.  Idempotent — does not
    re-train.
    """
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from kakeya.image_codec import (
        TEST_IMAGE,
        KakeyaHyperpriorCodec,
        _encode_bitstream,
        _evaluate_hd_chart,
        _ssim,
        _to_image,
    )

    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    run_dir = (manager.project_root / record.run_dir).resolve()
    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail="检查点不存在，无法重新生成")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    architecture = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    if architecture.get("version") != 3:
        raise HTTPException(
            status_code=409,
            detail="该检查点属于旧主干，请用当前多尺度主干重新训练",
        )
    latent_dim = config.get("latent_dim", 8) if isinstance(config, dict) else 8
    model = KakeyaHyperpriorCodec(
        latent_dim=latent_dim,
        hyper_dim=int(architecture.get("hyper_dim", max(8, latent_dim))),
    ).to(device)

    skip_keys = {
        "entropy_bottleneck._offset",
        "entropy_bottleneck._quantized_cdf",
        "entropy_bottleneck._cdf_length",
        "y_entropy_bottleneck._offset",
        "y_entropy_bottleneck._quantized_cdf",
        "y_entropy_bottleneck._cdf_length",
        "gaussian_conditional._offset",
        "gaussian_conditional._quantized_cdf",
        "gaussian_conditional._cdf_length",
        "gaussian_conditional.scale_table",
    }
    filtered_sd = {
        k: v for k, v in payload["model_state_dict"].items() if k not in skip_keys
    }
    model.load_state_dict(filtered_sd, strict=False)
    model.init_scale_table()
    model.update()
    model.eval()

    model.to(device)

    # --- 256² reconstruction (same as _evaluate_chart) ---
    source_image = Image.open(TEST_IMAGE).convert("RGB")
    source = torch.from_numpy(np.asarray(source_image, dtype=np.float32) / 255.0)
    source = source.permute(2, 0, 1).unsqueeze(0).to(device)
    mu = model.encode(source)
    decoded_latent, bitstream = _encode_bitstream(model, mu, run_dir)
    reconstructed = model.decode(decoded_latent.to(device)).clamp(0, 1)
    mse = float(F.mse_loss(reconstructed, source))
    psnr = 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)
    ssim = float(_ssim(reconstructed, source))

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    original_path = report_dir / "original.png"
    reconstruction_path = report_dir / "reconstruction.png"
    error_path = report_dir / "error.png"
    source_image.save(original_path)
    _to_image(reconstructed[0]).save(reconstruction_path)
    difference = (reconstructed - source).abs()[0]
    heat = torch.stack(
        (
            difference.mean(dim=0).mul(3).clamp(0, 1),
            difference.mean(dim=0).mul(0.7).clamp(0, 1),
            torch.zeros_like(difference[0]),
        )
    )
    _to_image(heat).save(error_path)

    # --- HD reconstruction ---
    hd_metrics = _evaluate_hd_chart(model, device, run_dir, report_dir)

    metrics: dict[str, Any] = {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "latent_dim": float(model.latent_dim),
        "source_bytes": float(TEST_IMAGE.stat().st_size),
        "bitstream_bytes": float(bitstream["bytes"]),
        "bitstream_payload_bytes": float(bitstream["payload_bytes"]),
        "bitstream_bpp": float(bitstream["bpp"]),
    }
    metrics.update(hd_metrics)

    # --- Update saved dashboard.json ---
    old = manager.result(job_id)
    if old is not None:
        old["metrics"] = metrics
        if "image_codec" in old:
            old["image_codec"]["images"] = {
                "original": "reports/original.png",
                "reconstruction": "reports/reconstruction.png",
                "error": "reports/error.png",
                "original_hd": "reports/original_hd.png",
                "reconstruction_hd": "reports/reconstruction_hd.png",
                "error_hd": "reports/error_hd.png",
            }
            old["image_codec"]["bitstream"] = bitstream
        dash_path = (run_dir / "reports" / "dashboard.json").resolve()
        if run_dir in dash_path.parents:
            dash_path.write_text(
                json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return {"metrics": metrics, "bitstream": bitstream}


@app.get("/api/experiments/{job_id}/artifact/bitstream")
def experiment_bitstream(job_id: str) -> FileResponse:
    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    result = manager.result(job_id)
    bitstream = result.get("image_codec", {}).get("bitstream", {}) if result else {}
    relative_path = bitstream.get("path")
    if not relative_path:
        raise HTTPException(status_code=404, detail="该实验没有压缩码流")
    run_dir = (manager.project_root / record.run_dir).resolve()
    artifact_path = (run_dir / relative_path).resolve()
    if run_dir not in artifact_path.parents or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="压缩码流不存在")
    return FileResponse(
        artifact_path,
        media_type="application/octet-stream",
        filename=bitstream.get("filename", "reconstruction.kky"),
    )


@app.get("/api/experiments/{job_id}/artifact/checkpoint")
def experiment_checkpoint(job_id: str) -> FileResponse:
    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    result = manager.result(job_id)
    bitstream = result.get("image_codec", {}).get("bitstream", {}) if result else {}
    relative_path = bitstream.get("checkpoint", "checkpoints/final.pt")
    run_dir = (manager.project_root / record.run_dir).resolve()
    artifact_path = (run_dir / relative_path).resolve()
    if run_dir not in artifact_path.parents or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="模型检查点不存在")
    return FileResponse(
        artifact_path,
        media_type="application/octet-stream",
        filename="final.pt",
    )


@app.post("/api/experiments/{job_id}/reconstruct")
async def reconstruct_uploaded(
    job_id: str, file: Annotated[UploadFile, File(...)]
) -> dict[str, Any]:
    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    run_dir = (manager.project_root / record.run_dir).resolve()
    checkpoint_path = run_dir / "checkpoints/final.pt"
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail="模型检查点不存在")
    image_bytes = await file.read()
    return _do_reconstruct(checkpoint_path, image_bytes)


@app.post("/api/reconstruct-custom")
async def reconstruct_custom_checkpoint(
    checkpoint: Annotated[UploadFile, File(...)],
    image: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    checkpoint_bytes = await checkpoint.read()
    image_bytes = await image.read()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(checkpoint_bytes)
        temp_path = Path(f.name)
    try:
        return _do_reconstruct(temp_path, image_bytes)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _do_reconstruct(checkpoint_path: Path, image_bytes: bytes) -> dict[str, Any]:
    import base64
    import io
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F

    from kakeya.image_codec import KakeyaHyperpriorCodec, _encode_bitstream

    try:
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析上传的图片")

    w, h = source_image.size
    if w > 4096 or h > 4096:
        raise HTTPException(
            status_code=400, detail="图片尺寸过大（上限 4096×4096），请先缩小后再试"
        )
    if w < 16 or h < 16:
        raise HTTPException(status_code=400, detail="图片尺寸过小（下限 16×16）")

    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    if pad_w or pad_h:
        padded = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
        padded.paste(source_image, (0, 0))
        source_image = padded

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        raise HTTPException(status_code=400, detail="无法加载模型检查点")

    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    architecture = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    if architecture.get("version") != 3:
        raise HTTPException(
            status_code=409,
            detail="该检查点属于旧主干，请用当前多尺度主干重新训练",
        )
    latent_dim = config.get("latent_dim", 8) if isinstance(config, dict) else 8
    try:
        model = KakeyaHyperpriorCodec(
            latent_dim=latent_dim,
            hyper_dim=int(architecture.get("hyper_dim", max(8, latent_dim))),
        ).to(device)
        # Exclude CDF buffers that mismatch shapes; update() repopulates them.
        skip_keys = {
            "entropy_bottleneck._offset",
            "entropy_bottleneck._quantized_cdf",
            "entropy_bottleneck._cdf_length",
            "y_entropy_bottleneck._offset",
            "y_entropy_bottleneck._quantized_cdf",
            "y_entropy_bottleneck._cdf_length",
            "gaussian_conditional._offset",
            "gaussian_conditional._quantized_cdf",
            "gaussian_conditional._cdf_length",
            "gaussian_conditional.scale_table",
        }
        filtered_sd = {
            k: v for k, v in payload["model_state_dict"].items() if k not in skip_keys
        }
        model.load_state_dict(filtered_sd, strict=False)
        model.init_scale_table()
        model.update()
        model.eval()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="模型结构不匹配，请确认是 Kakeya image_codec 训练的 checkpoint",
        )

    source = torch.from_numpy(np.asarray(source_image, dtype=np.float32) / 255.0)
    source = source.permute(2, 0, 1).unsqueeze(0).to(device)

    # Whole-image encode / compress / decode path.  The multi-scale trained
    # model handles any resolution directly, so no tiling or rescaling is
    # needed; padding to a multiple of 8 (above) keeps the encoder happy.
    with torch.no_grad():
        latent = model.encode(source)

    with tempfile.TemporaryDirectory() as temp_dir:
        decoded_latent, bitstream = _encode_bitstream(
            model, latent, Path(temp_dir), write_file=False
        )
    with torch.no_grad():
        reconstructed = model.decode(decoded_latent).clamp(0, 1)
    bitstream_bytes = int(bitstream["bytes"])
    bpp = bitstream_bytes * 8 / (w * h)

    if pad_w or pad_h:
        reconstructed = reconstructed[:, :, :h, :w]
        source = source[:, :, :h, :w]

    mse = float(F.mse_loss(reconstructed, source))
    psnr = 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)
    ssim = float(_ssim_torch(reconstructed, source))
    diff = (reconstructed - source).abs()[0]
    heat = torch.stack(
        (
            diff.mean(dim=0).mul(3).clamp(0, 1),
            diff.mean(dim=0).mul(0.7).clamp(0, 1),
            torch.zeros_like(diff[0]),
        )
    )

    def _to_png(tensor: torch.Tensor) -> str:
        arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "original": _to_png(source[0]),
        "reconstruction": _to_png(reconstructed[0]),
        "error": _to_png(heat),
        "metrics": {
            "mse": mse,
            "psnr": float(psnr),
            "ssim": ssim,
            "bitstream_bytes": bitstream_bytes,
            "bpp": float(bpp),
            "downscaled": False,
        },
    }


def _ssim_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2))
        / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    ).mean()


@app.post("/api/experiments/{job_id}/artifact/open-checkpoint-dir")
def open_checkpoint_dir(job_id: str) -> dict[str, Any]:
    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    run_dir = (manager.project_root / record.run_dir).resolve()
    checkpoints_dir = (run_dir / "checkpoints").resolve()
    if run_dir not in checkpoints_dir.parents or not checkpoints_dir.is_dir():
        raise HTTPException(status_code=404, detail="checkpoints 目录不存在")
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", str(checkpoints_dir)])
    elif system == "Windows":
        os.startfile(str(checkpoints_dir))  # type: ignore[attr-defined]
    elif system == "Linux":
        subprocess.Popen(["xdg-open", str(checkpoints_dir)])
    else:
        raise HTTPException(status_code=400, detail="不支持的操作系统")
    return {"ok": True, "path": str(checkpoints_dir)}


@app.get("/api/experiments/{job_id}/events")
async def experiment_events(job_id: str, request: Request) -> StreamingResponse:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="实验不存在")

    async def stream():
        previous = ""
        terminal_sent = False
        while not await request.is_disconnected():
            record = manager.get(job_id)
            if record is None:
                break
            snapshot = record.public()
            serialized = json.dumps(snapshot, ensure_ascii=False)
            if serialized != previous:
                yield f"event: snapshot\ndata: {serialized}\n\n"
                previous = serialized
            if snapshot["status"] in {"completed", "failed", "cancelled"}:
                if terminal_sent:
                    break
                terminal_sent = True
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    uvicorn.run(
        "kakeya.web_api:app",
        host=os.environ.get("KAKEYA_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("KAKEYA_API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
