"""Runner event shapes.

Events are emitted around player initialization, decision requests, accepted
engine transitions, player errors, and final summaries. They carry runner
metadata and opaque player/model diagnostics without interpreting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


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
