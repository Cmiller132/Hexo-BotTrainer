"""Durable runner game record boundary.

The runner records what happened while a game runs. It does not reinterpret
game legality, model tensors, or policy semantics; it joins engine transition
records, runner metadata, common policy outputs, and opaque model extension
records into a durable game record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class GameRecordEntry:
    """Durable record for one accepted decision/transition."""

    game_id: str
    turn_index: int
    player_id: str
    engine_record: object
    policy_record: object | None = None
    model_extension_records: Sequence[object] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Complete durable record for one game."""

    game_id: str
    entries: Sequence[GameRecordEntry]
    terminal: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RecordSink(Protocol):
    """Destination for game record entries."""

    def write_entry(self, entry: GameRecordEntry) -> None:
        """Persist or forward one record entry as the game runs."""

    def close_game(self, game_id: str, terminal: object | None = None) -> object:
        """Finalize a game record and return a storage reference or manifest."""


class RecordAnalyzer(Protocol):
    """Post-game analyzer for durable records."""

    def analyze(self, record: GameRecord) -> Mapping[str, Any]:
        """Return derived statistics without mutating the source record."""
