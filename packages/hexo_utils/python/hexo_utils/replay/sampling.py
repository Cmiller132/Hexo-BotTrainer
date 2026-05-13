"""Replay sampling boundary.

Sampling utilities should select replay records and preserve schema metadata.
Model packages are responsible for turning sampled records into architecture
specific tensors, masks, policy targets, and value targets.

Sampling stays model-consistent by returning training replay decisions plus
schema metadata and references to core game records. Each model package
provides the reader/mapper that converts those decisions into its own tensors
and may declare required extension namespaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Sequence

from hexo_utils.encoding import D6Symmetry


@dataclass(frozen=True, slots=True)
class ReplaySampleRequest:
    """Request for a reusable replay sampler."""

    count: int
    seed: int | None = None
    required_extensions: Sequence[str] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A sampled set of training replay records plus provenance metadata."""

    records: Sequence[object]
    symmetries: Sequence[D6Symmetry] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by replay sampling helpers.")


def sample_replay_records(source: object, request: ReplaySampleRequest) -> ReplayBatch:
    """Sample training replay records without constructing model tensors.

    The eventual sampler should:

    1. read core game records and replay records by schema version;
    2. select decision records deterministically from `seed` and filters;
    3. choose one deterministic D6 symmetry per sampled decision;
    4. keep legal-action ordering and action ids intact;
    5. include only requested model extension namespaces when asked;
    6. return records and symmetries for model packages to parse into batches.
    """

    _not_implemented("sample_replay_records")
