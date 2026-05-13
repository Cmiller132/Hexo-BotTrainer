"""Session setup contracts for one engine-backed game.

`match.py` creates a session, then hands the initialized context to the shared
loop. Session setup owns engine state creation, player initialization, seed and
scenario metadata, and any future resource handles needed by parallel batch
runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hexo_engine import EngineStateRef

from .player import RunnerPlayer


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Inputs needed to create one runner session."""

    players: Sequence[RunnerPlayer]
    seed: int | None = None
    scenario: object | None = None
    mode: str = "match"
    is_evaluation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Initialized context consumed by the game loop and players."""

    session_id: str
    game_id: str
    seed: int | None
    engine_state: EngineStateRef
    players: Sequence[RunnerPlayer]
    mode: str = "match"
    is_evaluation: bool = False
    engine_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def create_session_context(spec: SessionSpec) -> SessionContext:
    """Create engine state and the context passed to players before the loop.

    The implementation should avoid process-global mutable state so many
    sessions can be created safely by batch workers. It should create or load
    the `EngineStateRef`, ask each player to initialize against the resulting
    context, and return the initialized players with the state handle.
    """

    raise NotImplementedError("Session creation will be wired to engine setup.")
