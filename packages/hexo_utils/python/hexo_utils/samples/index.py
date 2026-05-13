"""Sample index placeholder.

An index lets training choose rows without scanning every sample chunk. The
index remains storage/provenance mechanics; models still decode selected rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .store import SampleStore


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """Searchable summary over finalized sample chunks."""

    store: SampleStore
    sample_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def refresh_sample_index(store: SampleStore) -> SampleIndex:
    """Refresh and return the sample index for a store."""

    return SampleIndex(
        store=store,
        sample_count=0,
        metadata={"note": "Indexing is a placeholder until chunk IO exists."},
    )
