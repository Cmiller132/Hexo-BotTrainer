"""Training pipeline boundary for the ResNet model family.

Training code owns how ResNet samples become examples, policy/value targets,
loss inputs, optimizer steps, checkpoint writes, and model-specific metrics.
Shared sample helpers may select records, but target semantics live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Inputs for one ResNet training run."""

    model: object
    sample_source: object
    config: object
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary emitted after training."""

    checkpoint: object | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by ResNet training code.")


def train(run: TrainingRun) -> TrainingResult:
    """Train a ResNet model from model-owned sample records."""

    _not_implemented("train")
