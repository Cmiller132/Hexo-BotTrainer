"""CLI entry point for config-driven model training.

Expected use:

    hexo-train-model path/to/train.toml
    python -m hexo_train.cli.train_model path/to/train.yaml

The CLI intentionally stays thin. All lifecycle decisions belong to
`TrainingPipeline`, which keeps command parsing separate from orchestration.

This file should only translate command-line arguments into a pipeline call.
Any decision about epochs, checkpoints, samples, or model behavior belongs in
config normalization or the pipeline itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _prepend_workspace_package_paths() -> None:
    """Prefer sibling workspace packages when running the CLI from a repo checkout."""

    package_dir = Path(__file__).resolve()
    packages_dir = package_dir.parents[4]
    if not (packages_dir / "hexo_models").is_dir():
        return
    for relative in (
        ("hexo_models", "python"),
        ("hexo_train", "python"),
        ("hexo_runner", "python"),
        ("hexo_engine", "python"),
        ("hexo_utils", "python"),
    ):
        path = str(packages_dir.joinpath(*relative))
        if path not in sys.path:
            sys.path.insert(0, path)


_prepend_workspace_package_paths()

from hexo_train.pipeline import TrainingPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the single public training command."""

    parser = argparse.ArgumentParser(
        prog="hexo-train-model",
        description="Run a Hexo model training pipeline from YAML or TOML config.",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to a .yaml, .yml, or .toml training config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run the training pipeline, and return a status code."""

    args = build_parser().parse_args(argv)
    TrainingPipeline().run(args.config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
