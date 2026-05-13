"""Core runner loop boundary for one game.

`match.py` is the public entry point for a single game. It creates a session
and calls this loop. The loop asks the engine for context, asks the active
player for a decision, submits the chosen action back to the engine, emits
events/records, and stops on terminal state or runner policy failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .records import EventSink, GameResult, RecordSink
from .session import SessionContext


@dataclass(frozen=True, slots=True)
class LoopOptions:
    """Runner loop options that are independent of model internals."""

    emit_replay: bool = True
    stop_on_player_error: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GameLoop:
    """Coordinator for one engine-backed game."""

    def __init__(
        self,
        options: LoopOptions | None = None,
        *,
        event_sink: EventSink | None = None,
        record_sink: RecordSink | None = None,
    ) -> None:
        self.options = options or LoopOptions()
        self.event_sink = event_sink
        self.record_sink = record_sink

    def run(self, session_context: SessionContext) -> GameResult:
        """Run a game using `session_context.engine_state` and runner players."""

        raise NotImplementedError("GameLoop.run will be wired to the engine API.")
