"""Checkpoint helpers for the ResNet model family.

Checkpoint semantics are model-owned because architecture config, optimizer
state, and tensor names are specific to this package. Runner code should treat
checkpoint references as opaque inputs to player/model loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Location and metadata for a ResNet checkpoint."""

    path: Path
    step: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by checkpoint IO.")


def load_checkpoint(ref: CheckpointRef, *, map_location: object | None = None) -> object:
    """Load a model package checkpoint."""

    _not_implemented("load_checkpoint")


def save_checkpoint(model: object, ref: CheckpointRef, *, metadata: Mapping[str, Any] | None = None) -> None:
    """Save a model package checkpoint."""

    _not_implemented("save_checkpoint")
