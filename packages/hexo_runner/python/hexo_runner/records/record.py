"""Compact durable game records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..player import PlayerIdentity

GAME_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AbortRecord:
    """Abort information for fail-loud runner outcomes."""

    stage: str
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class PlayerRecord:
    """Player identity and role for one game."""

    player_id: str
    role: str
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionRecordV1:
    """Compact accepted-action record."""

    index: int
    player_id: str
    player_role: str
    action_id: str
    action: Mapping[str, Any]
    decision_ms: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    transition: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """Simple wall-clock timing summary."""

    duration_ms: float


@dataclass(frozen=True, slots=True)
class GameRecordV1:
    """Compact replayable game record."""

    schema_version: int
    game_id: str
    seed: int | None
    scenario: object | None
    engine: Mapping[str, Any]
    players: Sequence[PlayerRecord]
    actions: Sequence[ActionRecordV1]
    status: str
    terminal: Mapping[str, Any] | None
    abort: AbortRecord | None
    timing: TimingRecord
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        game_id: str,
        seed: int | None,
        scenario: object | None,
        engine: Mapping[str, Any],
        players: Sequence[PlayerRecord],
        actions: Sequence[ActionRecordV1],
        status: str,
        terminal: Mapping[str, Any] | None,
        abort: AbortRecord | None,
        duration_ms: float,
        metadata: Mapping[str, Any],
    ) -> "GameRecordV1":
        return cls(
            schema_version=GAME_RECORD_SCHEMA_VERSION,
            game_id=game_id,
            seed=seed,
            scenario=_jsonable(scenario),
            engine=_jsonable(engine),
            players=tuple(players),
            actions=tuple(actions),
            status=status,
            terminal=_jsonable(terminal),
            abort=abort,
            timing=TimingRecord(duration_ms=duration_ms),
            metadata=_jsonable(metadata),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """Compatibility view for older manual sinks.

    New durable records should use `GameRecordV1`.
    """

    game_id: str
    turn_index: int
    player_id: str
    action: object
    terminal: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Compatibility alias shape for complete game records."""

    game_id: str
    players: Sequence[PlayerIdentity]
    entries: Sequence[PositionRecord]
    seed: int | None = None
    terminal: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RecordSink(Protocol):
    """Destination for compact game records."""

    def write_game(self, record: GameRecordV1) -> object:
        """Persist one complete compact game record."""


class MemoryRecordSink:
    """In-memory sink useful for tests and manual controllers."""

    def __init__(self) -> None:
        self.records: list[GameRecordV1] = []

    def write_game(self, record: GameRecordV1) -> object:
        self.records.append(record)
        return {"game_id": record.game_id, "actions": len(record.actions), "status": record.status}


class JsonlRecordSink:
    """Append one JSON game record per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_game(self, record: GameRecordV1) -> object:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        return {"path": str(self.path), "game_id": record.game_id, "status": record.status}


class RecordAnalyzer(Protocol):
    """Post-game analyzer for durable records."""

    def analyze(self, record: GameRecord) -> Mapping[str, Any]:
        """Return derived statistics without mutating the source record."""


def _jsonable(value: object) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
