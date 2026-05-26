"""Compact durable game records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Protocol

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
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "seed": self.seed,
            "scenario": _jsonable(self.scenario),
            "engine": _jsonable(self.engine),
            "players": [_player_to_dict(player) for player in self.players],
            "actions": [_action_to_dict(action) for action in self.actions],
            "status": self.status,
            "terminal": _jsonable(self.terminal),
            "abort": _abort_to_dict(self.abort),
            "timing": {"duration_ms": self.timing.duration_ms},
            "metadata": _jsonable(self.metadata),
        }


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

    def __init__(self, path: str | Path, *, flush_on_write: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_on_write = flush_on_write
        self._handle: TextIOWrapper | None = None

    def write_game(self, record: GameRecordV1) -> object:
        handle = self._open()
        handle.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        if self.flush_on_write:
            handle.flush()
        return {"path": str(self.path), "game_id": record.game_id, "status": record.status}

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None

    def __enter__(self) -> JsonlRecordSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open(self) -> TextIOWrapper:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle


def _jsonable(value: object) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _player_to_dict(player: PlayerRecord) -> dict[str, Any]:
    return {
        "player_id": player.player_id,
        "role": player.role,
        "label": player.label,
        "metadata": _jsonable(player.metadata),
    }


def _action_to_dict(action: ActionRecordV1) -> dict[str, Any]:
    return {
        "index": action.index,
        "player_id": action.player_id,
        "player_role": action.player_role,
        "action_id": action.action_id,
        "action": _jsonable(action.action),
        "decision_ms": action.decision_ms,
        "diagnostics": _jsonable(action.diagnostics),
        "transition": _jsonable(action.transition),
    }


def _abort_to_dict(abort: AbortRecord | None) -> dict[str, str] | None:
    if abort is None:
        return None
    return {
        "stage": abort.stage,
        "exception_type": abort.exception_type,
        "message": abort.message,
    }
