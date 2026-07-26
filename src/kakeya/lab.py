"""Reliable launcher for the local Kakeya experiment platform."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
LOCAL_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
REQUIRED_PYTHON_MODULES = (
    "certifi",
    "fastapi",
    "matplotlib",
    "numpy",
    "sklearn",
    "torch",
    "torchvision",
    "tqdm",
    "uvicorn",
    "yaml",
)


@dataclass(frozen=True)
class Service:
    name: str
    process: subprocess.Popen[bytes]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kakeya-lab",
        description="启动 Kakeya 网页实验台和本地训练服务",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="启动前安装或更新 Python 与前端依赖",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="只安装依赖并检查环境，不启动服务（需要与 --install 一起使用）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只执行环境检查，不启动服务",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="构建并启动生产版前端；默认使用开发服务器",
    )
    parser.add_argument("--api-host", default="127.0.0.1", help="训练 API 监听地址")
    parser.add_argument("--api-port", type=int, default=8000, help="训练 API 端口")
    parser.add_argument("--ui-host", default="localhost", help="网页监听地址")
    parser.add_argument("--ui-port", type=int, default=3000, help="网页端口")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="服务就绪后不自动打开浏览器",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90,
        help="等待服务启动的最长秒数，默认 90",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.setup_only and not args.install:
        raise SystemExit("--setup-only 需要与 --install 一起使用")
    _validate_port(args.api_port, "--api-port")
    _validate_port(args.ui_port, "--ui-port")
    if args.api_port == args.ui_port and _same_local_host(
        args.api_host, args.ui_host
    ):
        raise SystemExit("API 与网页不能使用同一端口")

    print("\nKakeya 潜空间实验台")
    print("=" * 52)
    if args.install:
        install_dependencies()

    report = check_environment()
    print_environment_report(report)
    if not report["ready"]:
        print("\n环境尚未就绪。请运行：")
        print("  python start_lab.py --install")
        return 2
    if args.check or args.setup_only:
        return 0

    occupied = []
    if not port_available(args.api_host, args.api_port):
        occupied.append(f"API {args.api_host}:{args.api_port}")
    if not port_available(args.ui_host, args.ui_port):
        occupied.append(f"网页 {args.ui_host}:{args.ui_port}")
    if occupied:
        print("\n无法启动，以下地址已被占用：")
        for item in occupied:
            print(f"  - {item}")
        print("请停止已有服务，或使用 --api-port / --ui-port 更换端口。")
        return 2

    api_url = _public_url(args.api_host, args.api_port)
    ui_url = _public_url(args.ui_host, args.ui_port)
    services: list[Service] = []
    try:
        services.append(
            Service(
                "训练 API",
                _start_api(args.api_host, args.api_port, ui_url),
            )
        )
        _wait_for_url(f"{api_url}/api/health", services, args.timeout, "训练 API")
        print(f"[OK] 训练 API 已就绪：{api_url}")

        if args.production:
            print("[..] 正在构建生产版网页")
            _run_checked(
                [_npm_command(), "run", "build"],
                cwd=FRONTEND_ROOT,
                label="前端构建",
                env=_frontend_environment(api_url),
            )
        services.append(
            Service(
                "网页",
                _start_frontend(
                    args.ui_host,
                    args.ui_port,
                    api_url,
                    production=args.production,
                ),
            )
        )
        _wait_for_url(ui_url, services, args.timeout, "网页")
        print(f"[OK] 网页已就绪：{ui_url}")
        print("\n按 Ctrl+C 可同时停止网页与训练服务。\n")
        if not args.no_browser:
            webbrowser.open(ui_url)
        return _monitor(services)
    except (RuntimeError, KeyboardInterrupt) as error:
        if isinstance(error, RuntimeError):
            print(f"\n启动失败：{error}")
            return 1
        print("\n正在停止服务…")
        return 0
    finally:
        stop_services(services)


def install_dependencies() -> None:
    npm = _require_executable("npm", "Node.js/npm")
    node = _require_executable("node", "Node.js")
    node_version = _executable_version(node, "--version")
    if not _version_at_least(node_version, (22, 13, 0)):
        raise RuntimeError(
            f"Node.js {node_version or '未知'} 版本过低，需要 22.13 或更高版本"
        )
    print("[..] 安装 Python 项目依赖")
    _run_checked(
        [sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
        cwd=PROJECT_ROOT,
        label="Python 依赖安装",
    )
    print("[..] 安装前端依赖")
    _run_checked(
        [npm, "install"],
        cwd=FRONTEND_ROOT,
        label="前端依赖安装",
    )


def check_environment() -> dict[str, object]:
    missing_modules = [
        name
        for name in REQUIRED_PYTHON_MODULES
        if importlib.util.find_spec(name) is None
    ]
    npm = shutil.which("npm")
    node = shutil.which("node")
    node_version = _executable_version(node, "--version") if node else None
    npm_version = _executable_version(npm, "--version") if npm else None
    frontend_installed = (FRONTEND_ROOT / "node_modules").is_dir()
    frontend_manifest = (FRONTEND_ROOT / "package.json").is_file()

    accelerator = "CPU"
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            accelerator = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            accelerator = "Apple MPS"

    python_supported = sys.version_info >= (3, 10)
    node_supported = _version_at_least(node_version, (22, 13, 0))
    ready = (
        python_supported
        and not missing_modules
        and bool(npm)
        and bool(node)
        and node_supported
        and frontend_manifest
        and frontend_installed
    )
    return {
        "ready": ready,
        "python": sys.version.split()[0],
        "python_supported": python_supported,
        "missing_modules": missing_modules,
        "node": node_version,
        "node_supported": node_supported,
        "npm": npm_version,
        "frontend_manifest": frontend_manifest,
        "frontend_installed": frontend_installed,
        "accelerator": accelerator,
    }


def print_environment_report(report: dict[str, object]) -> None:
    print(f"[{'OK' if report['python_supported'] else '!!'}] Python {report['python']}")
    node_status = "OK" if report["node_supported"] else "!!"
    print(f"[{node_status}] Node.js {report['node'] or '未安装'}（需要 >= 22.13）")
    print(f"[{'OK' if report['npm'] else '!!'}] npm {report['npm'] or '未安装'}")
    print(f"[{'OK' if not report['missing_modules'] else '!!'}] Python 依赖", end="")
    missing = report["missing_modules"]
    print(f"：缺少 {', '.join(missing)}" if missing else "：已就绪")
    frontend_ready = report["frontend_manifest"] and report["frontend_installed"]
    print(f"[{'OK' if frontend_ready else '!!'}] 前端依赖", end="")
    print("：已就绪" if frontend_ready else "：尚未安装")
    print(f"[OK] 训练设备：{report['accelerator']}")


def port_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
        return True
    except (OSError, socket.gaierror):
        return False


def stop_services(services: Sequence[Service]) -> None:
    for service in reversed(services):
        process = service.process
        if process.poll() is not None:
            continue
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 8
    for service in reversed(services):
        process = service.process
        if process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()


def _start_api(host: str, port: int, ui_url: str) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment.update(
        {
            "KAKEYA_API_HOST": host,
            "KAKEYA_API_PORT": str(port),
            "KAKEYA_UI_ORIGIN": ui_url,
            "PYTHONUNBUFFERED": "1",
        }
    )
    return _popen(
        [sys.executable, "-m", "kakeya.web_api"],
        cwd=PROJECT_ROOT,
        env=environment,
    )


def _start_frontend(
    host: str,
    port: int,
    api_url: str,
    *,
    production: bool,
) -> subprocess.Popen[bytes]:
    script = "start" if production else "dev"
    return _popen(
        [
            _npm_command(),
            "run",
            script,
            "--",
            "--host",
            host,
            "--port",
            str(port),
            "--strictPort",
        ],
        cwd=FRONTEND_ROOT,
        env=_frontend_environment(api_url),
    )


def _frontend_environment(api_url: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["NEXT_PUBLIC_KAKEYA_API_URL"] = api_url
    return environment


def _popen(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        start_new_session=os.name == "posix",
    )


def _monitor(services: Sequence[Service]) -> int:
    while True:
        for service in services:
            return_code = service.process.poll()
            if return_code is not None:
                print(f"\n{service.name} 已退出（代码 {return_code}）。")
                return return_code or 1
        time.sleep(0.5)


def _wait_for_url(
    url: str,
    services: Sequence[Service],
    timeout: float,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        for service in services:
            return_code = service.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"{service.name} 在启动过程中退出（代码 {return_code}）"
                )
        try:
            # Local readiness checks must not be routed through a configured
            # corporate/system HTTP proxy.
            with LOCAL_URL_OPENER.open(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.4)
    suffix = f"：{last_error}" if last_error else ""
    raise RuntimeError(f"等待{label}就绪超时{suffix}")


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    label: str,
    env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(list(command), cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label}失败（代码 {result.returncode}）")


def _require_executable(name: str, description: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"未找到 {description}，请先安装后重试")
    return path


def _npm_command() -> str:
    return _require_executable("npm", "Node.js/npm")


def _executable_version(path: str | None, flag: str) -> str | None:
    if path is None:
        return None
    try:
        result = subprocess.run(
            [path, flag],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip().lstrip("v") or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _version_at_least(
    value: str | None, minimum: tuple[int, int, int]
) -> bool:
    if value is None:
        return False
    try:
        parts = tuple(int(part) for part in value.split(".")[:3])
    except ValueError:
        return False
    return parts + (0,) * (3 - len(parts)) >= minimum


def _validate_port(port: int, option: str) -> None:
    if not 1 <= port <= 65535:
        raise SystemExit(f"{option} 必须在 1 到 65535 之间")


def _same_local_host(left: str, right: str) -> bool:
    local_names = {"localhost", "127.0.0.1", "0.0.0.0"}
    return left == right or left in local_names and right in local_names


def _public_url(host: str, port: int) -> str:
    public_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{public_host}:{port}"


if __name__ == "__main__":
    raise SystemExit(main())
