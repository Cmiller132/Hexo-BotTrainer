"""Runner result summaries.

Results describe the outcome of games and batches from the runner point of
view: participants, status, terminal state, record references, and optional
post-game analysis summaries. Detailed move history belongs in game records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence


class GameStatus(StrEnum):
    """Runner-level game completion status."""

    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class GameResult:
    """Summary for one game."""

    game_id: str
    status: GameStatus
    terminal: object | None = None
    winner: object | None = None
    record_ref: object | None = None
    analysis: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Aggregate summary for many games."""

    games: Sequence[GameResult]
    metadata: Mapping[str, Any] = field(default_factory=dict)
