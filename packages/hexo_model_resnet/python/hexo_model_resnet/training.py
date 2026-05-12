"""Training pipeline boundary for the ResNet model family.

Training code owns how replay becomes ResNet examples, policy/value targets,
loss inputs, optimizer steps, checkpoint writes, and model-specific metrics.
Shared replay helpers may sample records, but target semantics live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Inputs for one ResNet training run."""

    model: object
    replay_source: object
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
    """Train a ResNet model from replay records."""

    _not_implemented("train")
