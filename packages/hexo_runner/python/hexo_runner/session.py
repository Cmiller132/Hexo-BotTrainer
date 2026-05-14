"""Session setup contracts for one engine-backed game.

`match.py` creates a session, then hands the initialized context to the shared
loop. Session setup owns engine state creation, player initialization, seed and
scenario metadata, and any future resource handles needed by parallel batch
runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hexo_engine import EngineStateRef, engine_metadata, new_game

from .player import RunnerPlayer


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Inputs needed to create one engine-backed game.

    `run_match` receives this from the caller. The runner passes `seed` and
    `scenario` through to `hexo_engine.new_game`; it does not interpret game
    rules or scenario contents itself.
    """

    game_id: str
    seed: int | None = None
    scenario: object | None = None
    mode: str = "match"
    is_evaluation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


SessionSpec = GameSpec


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Initialized context consumed by the game loop and players.

    `state_ref` is the primary authoritative engine state. The loop keeps it and
    applies real moves to it. Per-decision players receive cloned state refs,
    not this primary handle.
    """

    session_id: str
    game_id: str
    seed: int | None
    state_ref: EngineStateRef
    players: Sequence[RunnerPlayer]
    mode: str = "match"
    is_evaluation: bool = False
    engine_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def create_session_context(spec: GameSpec, players: Sequence[RunnerPlayer]) -> SessionContext:
    """Create engine state and the context passed to players before the loop.

    This is the boundary where the runner asks the engine for the primary game
    state. Everything in `SessionContext` is setup/provenance data used by the
    loop and by `RunnerPlayer.initialize`.
    """

    # Public engine API call that creates the authoritative game state.
    state_ref = new_game(seed=spec.seed, scenario=spec.scenario)
    return SessionContext(
        session_id=spec.game_id,
        game_id=spec.game_id,
        seed=spec.seed,
        state_ref=state_ref,
        players=tuple(players),
        mode=spec.mode,
        is_evaluation=spec.is_evaluation,
        engine_metadata=engine_metadata(),
        metadata=spec.metadata,
    )
