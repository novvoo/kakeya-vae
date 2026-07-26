"""Command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from kakeya.runner import run_config_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Kakeya VAE experiments")
    parser.add_argument("configs", nargs="+", help="YAML experiment config files")
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), help="override automatic device"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable tqdm progress bars"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_config_files(
        args.configs, device=args.device, progress=not args.no_progress
    )
    return 0
