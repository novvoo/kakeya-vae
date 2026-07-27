"""Thread-safe subprocess management for web-triggered experiments."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "jobs"


@dataclass
class JobRecord:
    id: str
    config: dict[str, Any]
    device: str
    status: str = "queued"
    epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    message: str = "等待训练进程"
    error: str | None = None
    traceback: str | None = None
    run_dir: str | None = None
    result_path: str | None = None
    metrics: dict[str, Any] | None = None
    series: list[dict[str, Any]] = field(default_factory=list)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=300))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config,
            "device": self.device,
            "status": self.status,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "run_dir": self.run_dir,
            "metrics": dict(self.metrics) if self.metrics else None,
            "series": list(self.series),
            "logs": list(self.logs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_result": bool(self.result_path),
        }


class JobManager:
    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._install_process: subprocess.Popen[str] | None = None
        self._install_status = "idle"
        self._install_logs: deque[str] = deque(maxlen=200)
        self._install_error: str | None = None
        self._load_existing_runs()

    def create(self, config: dict[str, Any], device: str) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id,
            config=config,
            device=device,
            total_epochs=int(config["epochs"]),
        )
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        spec_path = RUNTIME_DIR / f"{job_id}.json"
        spec_path.write_text(
            json.dumps({"config": config, "device": device}, ensure_ascii=False),
            encoding="utf-8",
        )
        with self._lock:
            self._jobs[job_id] = record
        thread = threading.Thread(
            target=self._launch_worker,
            args=(record, spec_path),
            daemon=True,
            name=f"kakeya-job-{job_id}",
        )
        thread.start()
        return record

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(
                self._jobs.values(), key=lambda item: item.created_at, reverse=True
            )
            return [record.public() for record in records]

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stop(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            process = record.process
            if record.status not in {"queued", "running", "evaluating"}:
                return record
            record.status = "stopping"
            record.message = "正在强制停止训练进程"
            record.logs.append("已向整个训练进程组发送终止请求")
            record.updated_at = _now()
        if process is not None and process.poll() is None:
            threading.Thread(
                target=self._stop_worker,
                args=(record, process),
                daemon=True,
                name=f"kakeya-stop-{job_id}",
            ).start()
        elif process is None:
            # A queued worker has not spawned yet. _launch_worker observes the
            # stopping state and finalizes it without creating a subprocess.
            record.logs.append("训练进程尚未启动，已取消排队任务")
        return record

    def result(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            result_path = record.result_path if record else None
        if not result_path:
            return None
        path = Path(result_path).resolve()
        runs_root = (PROJECT_ROOT / "runs").resolve()
        if runs_root not in path.parents:
            raise ValueError("result path is outside the run directory")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("config", {}).get("method") == "image_codec"
            and "codec_baselines" not in payload
        ):
            from kakeya.image_codec import reference_codec_baselines

            payload["codec_baselines"] = reference_codec_baselines()
        return payload

    def environment(self) -> dict[str, Any]:
        packages = {}
        for module_name, distribution_name in (
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("compressai", "compressai"),
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("sklearn", "scikit-learn"),
        ):
            installed = importlib.util.find_spec(module_name) is not None
            version = None
            if installed:
                try:
                    version = importlib.metadata.version(distribution_name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            packages[distribution_name] = {
                "installed": installed,
                "version": version,
            }
        device = {
            "recommended": "cpu",
            "mps_available": False,
            "cuda_available": False,
            "label": "CPU",
        }
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available():
                device.update(
                    {
                        "recommended": "cuda",
                        "cuda_available": True,
                        "label": f"CUDA · {torch.cuda.get_device_name(0)}",
                    }
                )
            elif torch.backends.mps.is_available():
                device.update(
                    {
                        "recommended": "mps",
                        "mps_available": True,
                        "label": "Apple MPS",
                    }
                )
        with self._lock:
            return {
                "ready": all(item["installed"] for item in packages.values()),
                "python": sys.version.split()[0],
                "packages": packages,
                "install_status": self._install_status,
                "install_logs": list(self._install_logs),
                "install_error": self._install_error,
                "device": device,
            }

    def install_dependencies(self) -> None:
        with self._lock:
            if (
                self._install_process is not None
                and self._install_process.poll() is None
            ):
                return
            self._install_status = "running"
            self._install_logs.clear()
            self._install_error = None
        threading.Thread(
            target=self._run_install,
            daemon=True,
            name="kakeya-dependency-install",
        ).start()

    def shutdown(self) -> None:
        with self._lock:
            processes = [
                record.process
                for record in self._jobs.values()
                if record.process is not None and record.process.poll() is None
            ]
        for process in processes:
            _terminate_process_tree(process, grace_seconds=0.5)

    def _launch_worker(self, record: JobRecord, spec_path: Path) -> None:
        environment = os.environ.copy()
        source_path = str(PROJECT_ROOT / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [source_path, environment.get("PYTHONPATH")])
        )
        environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        command = [
            sys.executable,
            "-u",
            "-m",
            "kakeya.web_worker",
            str(spec_path),
        ]
        try:
            with self._lock:
                if record.status == "stopping":
                    record.status = "cancelled"
                    record.message = "训练已停止"
                    record.updated_at = _now()
                    return
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            with self._lock:
                record.process = process
                should_terminate = record.status == "stopping"
                if not should_terminate:
                    record.status = "running"
                record.updated_at = _now()
            if should_terminate:
                threading.Thread(
                    target=self._stop_worker,
                    args=(record, process),
                    daemon=True,
                    name=f"kakeya-stop-{record.id}",
                ).start()
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if line:
                    self._handle_line(record, line)
            return_code = process.wait()
            with self._lock:
                if record.status == "stopping":
                    record.status = "cancelled"
                    record.message = "训练已停止"
                elif record.status not in {"completed", "failed", "cancelled"}:
                    record.status = "failed"
                    record.error = f"训练进程异常退出（代码 {return_code}）"
                    record.message = "训练失败"
                record.updated_at = _now()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with self._lock:
                record.status = "failed"
                record.error = str(error)
                record.message = "无法启动训练进程"
                record.updated_at = _now()
        finally:
            spec_path.unlink(missing_ok=True)

    def _load_existing_runs(self) -> None:
        runs_root = PROJECT_ROOT / "runs"
        if not runs_root.exists():
            return
        for dashboard_path in sorted(
            runs_root.glob("*/*/reports/dashboard.json"), reverse=True
        ):
            try:
                payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
                config = payload["config"]
                if config.get("method") != "image_codec":
                    continue
                history = payload["history"]
                epochs = history.get("epoch", [])
                train_history = history.get("train", {})
                validation_history = history.get("validation", {})
                series = [
                    {
                        "epoch": epoch,
                        "train": {
                            key: values[index]
                            for key, values in train_history.items()
                            if isinstance(values, list) and index < len(values)
                        },
                        "validation": {
                            key: values[index]
                            for key, values in validation_history.items()
                            if isinstance(values, list) and index < len(values)
                        },
                    }
                    for index, epoch in enumerate(epochs)
                ]
                run_dir = dashboard_path.parents[1]
                has_error = bool(payload.get("error"))
                record = JobRecord(
                    id=f"saved-{run_dir.parent.name}-{run_dir.name}",
                    config=config,
                    device="saved",
                    status="failed" if has_error else "completed",
                    epoch=len(epochs),
                    total_epochs=int(config.get("epochs", len(epochs))),
                    progress=1.0 if not has_error else 0.0,
                    message=payload.get("error", "已从磁盘加载实验结果"),
                    error=payload.get("error") if has_error else None,
                    run_dir=str(run_dir.relative_to(PROJECT_ROOT)),
                    result_path=str(dashboard_path),
                    metrics=payload.get("metrics"),
                    series=series,
                    created_at=datetime.fromtimestamp(
                        dashboard_path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    updated_at=datetime.fromtimestamp(
                        dashboard_path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                )
                record.logs.append("已从磁盘加载实验结果" if not has_error else "训练失败记录加载")
                self._jobs[record.id] = record
            except (KeyError, IndexError, OSError, ValueError, json.JSONDecodeError):
                continue

    def _handle_line(self, record: JobRecord, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            with self._lock:
                record.logs.append(line)
                record.updated_at = _now()
            return

        event_type = event.get("event")
        with self._lock:
            message = event.get("message")
            if message:
                record.logs.append(message)
                if record.status != "stopping":
                    record.message = message
            # Once a stop is requested, late epoch/completed events must not
            # revive the job while the process group is being terminated.
            if record.status == "stopping" and event_type not in {
                "cancelled",
                "failed",
            }:
                record.updated_at = _now()
                return
            if event_type == "started":
                record.status = "running"
                record.device = event.get("device", record.device)
            elif event_type == "epoch":
                record.status = "running"
                record.epoch = int(event["epoch"])
                record.total_epochs = int(event["total_epochs"])
                record.progress = float(event["progress"])
                record.run_dir = event.get("run_dir")
                record.series.append(
                    {
                        "epoch": record.epoch,
                        "train": event["train"],
                        "validation": event["validation"],
                    }
                )
            elif event_type == "evaluating":
                record.status = "evaluating"
            elif event_type == "completed":
                record.status = "completed"
                record.progress = 1.0
                record.run_dir = event.get("run_dir")
                record.result_path = event.get("result_path")
                record.metrics = event.get("metrics")
            elif event_type == "cancelled":
                record.status = "cancelled"
            elif event_type == "failed":
                record.status = "failed"
                record.error = event.get("error")
                record.traceback = event.get("traceback")
            record.updated_at = _now()

    def _stop_worker(
        self, record: JobRecord, process: subprocess.Popen[str]
    ) -> None:
        forced = _terminate_process_tree(process)
        with self._lock:
            if forced:
                record.logs.append("普通停止超时，已强制终止整个训练进程组")
            else:
                record.logs.append("训练进程已响应终止请求")
            if record.status == "stopping":
                record.status = "cancelled"
                record.message = (
                    "训练已强制停止" if forced else "训练已停止"
                )
            record.updated_at = _now()

    def _run_install(self) -> None:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(PROJECT_ROOT),
            "--disable-pip-version-check",
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._install_process = process
            assert process.stdout is not None
            for line in process.stdout:
                with self._lock:
                    self._install_logs.append(line.rstrip())
            return_code = process.wait()
            with self._lock:
                if return_code == 0:
                    self._install_status = "completed"
                else:
                    self._install_status = "failed"
                    self._install_error = (
                        f"依赖安装进程退出，代码 {return_code}"
                    )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with self._lock:
                self._install_status = "failed"
                self._install_error = str(error)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminate_process_tree(
    process: subprocess.Popen[Any], *, grace_seconds: float = 2.0
) -> bool:
    """Terminate an isolated worker process group, escalating to kill."""

    if process.poll() is not None:
        return False

    _signal_process_tree(process, force=False)
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is not None:
        return False

    _signal_process_tree(process, force=True)
    kill_deadline = time.monotonic() + max(grace_seconds, 0.5)
    while process.poll() is None and time.monotonic() < kill_deadline:
        time.sleep(0.05)
    return True


def _signal_process_tree(
    process: subprocess.Popen[Any], *, force: bool
) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            if force:
                process.kill()
            else:
                process.terminate()
            return
        os.killpg(
            process.pid,
            signal.SIGKILL if force else signal.SIGTERM,
        )
    except ProcessLookupError:
        return
