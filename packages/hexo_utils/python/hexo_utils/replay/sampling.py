"""Replay sampling boundary.

Sampling utilities should select replay records and preserve schema metadata.
Model packages are responsible for turning sampled records into architecture
specific tensors, masks, policy targets, and value targets.

Sampling stays model-consistent by returning canonical replay decisions plus
schema metadata. Each model package provides the reader/mapper that converts
those decisions into its own tensors and may declare required extension
namespaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Sequence


@dataclass(frozen=True, slots=True)
class ReplaySampleRequest:
    """Request for a reusable replay sampler."""

    count: int
    seed: int | None = None
    required_extensions: Sequence[str] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A sampled set of canonical records plus provenance metadata."""

    records: Sequence[object]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by replay sampling helpers.")


def sample_replay_records(source: object, request: ReplaySampleRequest) -> ReplayBatch:
    """Sample canonical replay records without constructing model tensors.

    The eventual sampler should:

    1. read durable game records by schema version;
    2. select decision records deterministically from `seed` and filters;
    3. keep legal-action ordering and action ids intact;
    4. include only requested model extension namespaces when asked;
    5. return records for model packages to parse into their own batches.
    """

    _not_implemented("sample_replay_records")
