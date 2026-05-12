"""Model-specific diagnostics.

Diagnostics are produced by this model package and transported opaquely by the
runner. They can include policy summaries, value estimates, search statistics,
or training metrics without becoming runner concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DecisionDiagnostics:
    """Diagnostics attached to one model-backed decision."""

    policy_entropy: float | None = None
    value: float | None = None
    search_visits: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingDiagnostics:
    """Diagnostics emitted during or after training."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
