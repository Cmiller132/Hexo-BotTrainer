"""Neutral sample-store placeholder.

The sample store is shared mechanics: where chunks live, how they are opened,
and how later indexing/sampling code can find them. It does not know how a
model decodes records into tensors or what a target means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SampleStore:
    """Location and metadata for trainable sample chunks."""

    path: Path
    mode: str = "append"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def open_sample_store(
    path: str | Path,
    *,
    mode: str = "append",
    metadata: Mapping[str, Any] | None = None,
) -> SampleStore:
    """Open or create the directory that will contain sample chunks."""

    store_path = Path(path)
    store_path.mkdir(parents=True, exist_ok=True)
    return SampleStore(
        path=store_path,
        mode=mode,
        metadata=dict(metadata or {}),
    )
