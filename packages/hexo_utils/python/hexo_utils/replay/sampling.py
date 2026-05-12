"""Replay sampling boundary.

Sampling utilities should select replay records and preserve schema metadata.
Model packages are responsible for turning sampled records into architecture
specific tensors, masks, policy targets, and value targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Sequence


@dataclass(frozen=True, slots=True)
class ReplaySampleRequest:
    """Request for a reusable replay sampler."""

    count: int
    seed: int | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A sampled set of replay records plus provenance metadata."""

    records: Sequence[object]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by replay sampling helpers.")


def sample_replay_records(source: object, request: ReplaySampleRequest) -> ReplayBatch:
    """Sample replay records without constructing model-specific examples."""

    _not_implemented("sample_replay_records")
