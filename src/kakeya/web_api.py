"""FastAPI application for the browser-based experiment console."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from kakeya.config import ExperimentConfig
from kakeya.image_codec import TEST_IMAGE
from kakeya.job_manager import JobManager

manager = JobManager()


class ExperimentRequest(BaseModel):
    method: Literal["image_codec"] = "image_codec"
    epochs: Annotated[int, Field(ge=1, le=500)] = 50
    latent_dim: Annotated[int, Field(ge=2, le=256)] = 8
    batch_size: Annotated[int, Field(ge=1, le=2048)] = 4
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.0005
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 42
    num_workers: Annotated[int, Field(ge=0, le=32)] = 0
    train_limit: Annotated[int, Field(ge=0, le=60_000)] = 128
    test_limit: Annotated[int, Field(ge=0, le=10_000)] = 0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    beta: Annotated[float, Field(gt=0, le=100)] = 4.0
    gamma: Annotated[float, Field(ge=0, le=1000)] = 10.0
    lambda_kakeya: Annotated[float, Field(ge=0, le=1000)] = 0.001
    num_projections: Annotated[int, Field(ge=4, le=1024)] = 32
    k: Annotated[int, Field(ge=1, le=4096)] = 3
    degree: Annotated[int, Field(ge=1, le=10)] = 3

    @model_validator(mode="after")
    def validate_device(self) -> ExperimentRequest:
        import torch

        minimum_samples = 32 if self.method == "image_codec" else 100
        if self.train_limit and self.train_limit < minimum_samples:
            raise ValueError(
                f"训练样本上限应为 0 或至少 {minimum_samples}"
            )
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
        objective: dict[str, float | int] = {
            "lambda_kakeya": min(self.lambda_kakeya, 0.1),
            "num_projections": min(self.num_projections, 128),
            "k": min(self.k, max(self.batch_size - 1, 1)),
        }
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
def experiment_image(job_id: str, kind: Literal["original", "reconstruction", "error"]) -> FileResponse:
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


@app.get("/api/experiments/{job_id}/artifact/bitstream")
def experiment_bitstream(job_id: str) -> FileResponse:
    record = manager.get(job_id)
    if record is None or not record.run_dir:
        raise HTTPException(status_code=404, detail="实验不存在")
    result = manager.result(job_id)
    bitstream = (
        result.get("image_codec", {}).get("bitstream", {}) if result else {}
    )
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
    bitstream = (
        result.get("image_codec", {}).get("bitstream", {}) if result else {}
    )
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
