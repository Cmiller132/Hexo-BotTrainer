"""Training sample schema versioning helpers.

Sample records contain trainable self-play outputs plus optional model-owned
payloads. This module holds shared schema identifiers without owning
model-specific training targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


SAMPLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SampleSchema:
    """Version metadata attached to sample files and batches."""

    name: str = "hexo.samples"
    version: int = SAMPLE_SCHEMA_VERSION
    engine_version: str | None = None
    extensions: Mapping[str, int] = field(default_factory=dict)
