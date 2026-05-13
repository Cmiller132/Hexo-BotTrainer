"""Sample writing placeholder.

Model packages write trainable samples during self-play. This writer is the
shared storage surface they eventually write through; the model still owns the
payload schema and when pending samples become finalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .store import SampleStore


@dataclass(frozen=True, slots=True)
class SampleWriteResult:
    """Summary of samples appended to a store."""

    count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def append_samples(
    store: SampleStore,
    records: Sequence[object],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SampleWriteResult:
    """Append finalized sample records.

    This is a placeholder; future code should write chunked columnar data and
    update manifests without interpreting model-owned payloads.
    """

    _ = store
    return SampleWriteResult(count=len(records), metadata=dict(metadata or {}))
