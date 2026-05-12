"""Session setup contracts.

The runner session binds players, seeds, scenarios, and run metadata before any
game loop starts. Engine state creation and player initialization hang off this
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Inputs needed to create one runner session."""

    players: Sequence[object]
    seed: int | None = None
    scenario: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Context shared with players during initialization."""

    session_id: str
    seed: int | None
    engine_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def create_session_context(spec: SessionSpec) -> SessionContext:
    """Create the context object passed to players before the loop starts."""

    raise NotImplementedError("Session creation will be wired to engine setup.")
