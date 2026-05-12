"""Replay schema versioning helpers.

Replay records combine engine history, runner metadata, common policy logits,
and optional model-owned extensions. This module holds shared schema
identifiers without owning storage policy or model-specific training targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReplaySchema:
    """Version metadata attached to replay files and batches."""

    name: str = "hexo.replay"
    version: int = REPLAY_SCHEMA_VERSION
    engine_version: str | None = None
    extensions: Mapping[str, int] = field(default_factory=dict)
