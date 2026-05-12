"""Core runner loop boundary.

The loop asks the engine for context, asks the active player for a decision,
submits the chosen action back to the engine, records events, and stops on a
terminal result or runner policy failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LoopOptions:
    """Runner loop options that are independent of model internals."""

    emit_replay: bool = True
    stop_on_player_error: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GameLoop:
    """Coordinator for one engine-backed game."""

    def __init__(self, players: Sequence[object], options: LoopOptions | None = None) -> None:
        self.players = tuple(players)
        self.options = options or LoopOptions()

    def run(self, session_context: object) -> object:
        """Run a game using engine state and runner player contracts."""

        raise NotImplementedError("GameLoop.run will be wired to the engine API.")
