"""Ephemeral runner event shapes.

Events are emitted around player initialization, decision requests, accepted
engine transitions, player errors, and final summaries. They are useful for
logging, live observers, and feeding record writers, but they are not the
durable replay schema by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol


class EventKind(StrEnum):
    """Stable event names produced by runner loops and modes."""

    SESSION_STARTED = "session_started"
    DECISION_REQUESTED = "decision_requested"
    ACTION_APPLIED = "action_applied"
    PLAYER_ERROR = "player_error"
    SESSION_FINISHED = "session_finished"


@dataclass(frozen=True, slots=True)
class RunnerEvent:
    """One event emitted by the runner."""

    kind: EventKind
    game_id: str
    turn_index: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    """Observer for live runner events."""

    def emit(self, event: RunnerEvent) -> None:
        """Handle one event without taking ownership of durable storage."""
