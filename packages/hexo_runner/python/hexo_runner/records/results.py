"""Runner result summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .record import AbortRecord


class GameStatus(StrEnum):
    """Runner-level game completion status."""

    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class GameResult:
    """Summary for one game."""

    game_id: str
    status: GameStatus
    terminal: Mapping[str, Any] | None = None
    winner: object | None = None
    record_ref: object | None = None
    turns: int = 0
    duration_ms: float = 0.0
    abort: AbortRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Summary for a local batch run."""

    batch_id: str
    total_games: int
    completed: int
    aborted: int
    worker_count: int
    duration_ms: float
    record_refs: Sequence[object] = ()
    aborts: Sequence[AbortRecord] = ()
    results: Sequence[GameResult] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
