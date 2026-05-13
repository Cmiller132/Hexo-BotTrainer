"""Shared training orchestration package.

`hexo_train` owns config loading, run setup, stage orchestration, checkpoints,
and diagnostics. Model packages still own model architecture, sample decoding,
losses, target semantics, and model-specific training behavior.
"""

from __future__ import annotations

from .config import TrainingConfig, load_training_config
from .context import RunContext
from .pipeline import TrainingPipeline
from .registry import load_model_plugin

__all__ = [
    "RunContext",
    "TrainingConfig",
    "TrainingPipeline",
    "load_model_plugin",
    "load_training_config",
]
