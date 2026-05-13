"""Sample-window placeholder.

A sample window is the slice of indexed samples visible to a training run. This
supports KataGo-style rolling buffers without making utils understand model
targets or tensor layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .index import SampleIndex


@dataclass(frozen=True, slots=True)
class SampleWindow:
    """A selectable training window over an index."""

    index: SampleIndex
    window_size: int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_sample_window(
    index: SampleIndex,
    *,
    window_size: int | None = None,
    seed: int | None = None,
) -> SampleWindow:
    """Build a deterministic training window description."""

    return SampleWindow(
        index=index,
        window_size=window_size,
        seed=seed,
        metadata={"note": "Window selection is a placeholder."},
    )
