"""Compact binary Hexo runner records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import hexo_engine as engine
from hexo_engine.types import pack_coord_id, unpack_coord_id

HEXO_RECORD_MAGIC = b"HEXOREC1"
HEXO_RECORD_SCHEMA_VERSION = 1

_GAME_MARKER = b"G"
_STATUS_COMPLETED = 1
_STATUS_ABORTED = 2


@dataclass(frozen=True, slots=True)
class AbortRecord:
    """Abort information for fail-loud runner outcomes."""

    stage: str
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class HexoRecordPlayer:
    """Player identity stored once in a HexoRecordFile header."""

    player_id: str
    role: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class HexoRecord:
    """Replay-core data for one Hexo game."""

    game_id: str
    seed: int | None
    scenario: object | None
    status: str
    action_ids: tuple[int, ...]
    abort: AbortRecord | None = None
    winner: str | None = None
    placements: int | None = None

    def replay(self) -> engine.TerminalResult | None:
        """Replay accepted actions through the authoritative engine."""

        state = engine.new_game(seed=self.seed, scenario=self.scenario)
        for action_id in self.action_ids:
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(action_id)))
        return engine.terminal(state)


class HexoRecordGameWriter:
    """Append-only writer for one game inside a HexoRecordFile."""

    def __init__(self, record_file: "HexoRecordFile", game_id: str, seed: int | None, scenario: object | None) -> None:
        self._record_file = record_file
        self.game_id = game_id
        self.seed = seed
        self.scenario = _jsonable_scenario(scenario)
        self._action_ids: list[int] = []
        self._finished = False

    @property
    def action_count(self) -> int:
        return len(self._action_ids)

    def record_action(self, action: engine.PlacementAction | int) -> None:
        self._ensure_open()
        if isinstance(action, int):
            action_id = int(action)
        elif isinstance(action, engine.PlacementAction):
            action_id = pack_coord_id(action.coord)
        else:
            raise TypeError(f"unsupported record action type: {type(action).__name__}")
        self._action_ids.append(action_id)

    def finish_completed(self, winner: object, placements: int) -> object:
        self._ensure_open()
        record = HexoRecord(
            game_id=self.game_id,
            seed=self.seed,
            scenario=self.scenario,
            status="completed",
            action_ids=tuple(self._action_ids),
            winner=str(winner) if winner is not None else None,
            placements=int(placements),
        )
        self._record_file._write_record(record)
        self._finished = True
        return {"path": str(self._record_file.path), "game_id": self.game_id, "status": record.status}

    def finish_aborted(self, abort: AbortRecord) -> object:
        self._ensure_open()
        record = HexoRecord(
            game_id=self.game_id,
            seed=self.seed,
            scenario=self.scenario,
            status="aborted",
            action_ids=tuple(self._action_ids),
            abort=abort,
        )
        self._record_file._write_record(record)
        self._finished = True
        return {"path": str(self._record_file.path), "game_id": self.game_id, "status": record.status}

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError(f"record writer for {self.game_id!r} is already finished")


class HexoRecordFile:
    """Reader/writer for the binary Hexo runner record file format."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str,
        engine_metadata: Mapping[str, Any] | None = None,
        players: Sequence[object] = (),
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self._handle: BinaryIO | None = None
        self.engine_metadata: dict[str, Any] = {}
        self.players: tuple[HexoRecordPlayer, ...] = ()
        self._data_offset = 0

        if mode == "w":
            if engine_metadata is None:
                raise ValueError("engine_metadata is required when creating a HexoRecordFile")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.engine_metadata = _header_engine_metadata(engine_metadata)
            self.players = _header_players(players)
            self._handle = self.path.open("wb")
            self._write_header()
        elif mode == "r":
            self._handle = self.path.open("rb")
            self._read_header()
        else:
            raise ValueError(f"unsupported HexoRecordFile mode: {mode!r}")

    @classmethod
    def create(
        cls,
        path: str | Path,
        engine_metadata: Mapping[str, Any],
        players: Sequence[object],
    ) -> "HexoRecordFile":
        return cls(path, mode="w", engine_metadata=engine_metadata, players=players)

    @classmethod
    def open(cls, path: str | Path) -> "HexoRecordFile":
        return cls(path, mode="r")

    def begin_game(self, game_id: str, seed: int | None = None, scenario: object | None = None) -> HexoRecordGameWriter:
        if self.mode != "w":
            raise RuntimeError("cannot write games to a read-only HexoRecordFile")
        return HexoRecordGameWriter(self, game_id, seed, scenario)

    def iter_records(self) -> tuple[HexoRecord, ...]:
        if self.mode != "r":
            self.close()
            reader = type(self).open(self.path)
            try:
                return reader.iter_records()
            finally:
                reader.close()

        handle = self._require_handle()
        handle.seek(self._data_offset)
        records: list[HexoRecord] = []
        while True:
            marker = handle.read(1)
            if marker == b"":
                return tuple(records)
            if marker != _GAME_MARKER:
                raise ValueError(f"invalid HexoRecordFile game marker: {marker!r}")
            payload = handle.read(_read_varint(handle))
            records.append(_decode_game_payload(payload))

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "HexoRecordFile":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _write_record(self, record: HexoRecord) -> None:
        handle = self._require_handle()
        payload = _encode_game_payload(record)
        handle.write(_GAME_MARKER)
        _write_varint(handle, len(payload))
        handle.write(payload)

    def _write_header(self) -> None:
        handle = self._require_handle()
        handle.write(HEXO_RECORD_MAGIC)
        _write_varint(handle, HEXO_RECORD_SCHEMA_VERSION)
        _write_varint(handle, int(self.engine_metadata.get("rules_version", 0)))
        _write_string(handle, str(self.engine_metadata.get("backend", "")))
        _write_varint(handle, len(self.players))
        for player in self.players:
            _write_string(handle, player.player_id)
            _write_string(handle, player.role)
            _write_optional_string(handle, player.label)
        self._data_offset = handle.tell()

    def _read_header(self) -> None:
        handle = self._require_handle()
        magic = handle.read(len(HEXO_RECORD_MAGIC))
        if magic != HEXO_RECORD_MAGIC:
            raise ValueError(f"not a HexoRecordFile: {self.path}")
        schema = _read_varint(handle)
        if schema != HEXO_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported HexoRecordFile schema version: {schema}")
        rules_version = _read_varint(handle)
        backend = _read_string(handle)
        players = []
        for _ in range(_read_varint(handle)):
            players.append(HexoRecordPlayer(_read_string(handle), _read_string(handle), _read_optional_string(handle)))
        self.engine_metadata = {"rules_version": rules_version, "backend": backend}
        self.players = tuple(players)
        self._data_offset = handle.tell()

    def _require_handle(self) -> BinaryIO:
        if self._handle is None:
            raise RuntimeError("HexoRecordFile is closed")
        return self._handle


def _encode_game_payload(record: HexoRecord) -> bytes:
    buffer = bytearray()
    _append_string(buffer, record.game_id)
    _append_optional_int(buffer, record.seed)
    _append_optional_json(buffer, record.scenario)
    buffer.append(_STATUS_COMPLETED if record.status == "completed" else _STATUS_ABORTED)
    _append_varint(buffer, len(record.action_ids))
    for action_id in record.action_ids:
        _append_u32(buffer, action_id)
    _append_optional_string(buffer, record.winner)
    _append_optional_int(buffer, record.placements)
    if record.abort is None:
        buffer.append(0)
    else:
        buffer.append(1)
        _append_string(buffer, record.abort.stage)
        _append_string(buffer, record.abort.exception_type)
        _append_string(buffer, record.abort.message)
    return bytes(buffer)


def _decode_game_payload(payload: bytes) -> HexoRecord:
    cursor = _PayloadCursor(payload)
    game_id = cursor.read_string()
    seed = cursor.read_optional_int()
    scenario = cursor.read_optional_json()
    status_byte = cursor.read_byte()
    if status_byte not in (_STATUS_COMPLETED, _STATUS_ABORTED):
        raise ValueError(f"invalid HexoRecord status byte: {status_byte}")
    action_ids = tuple(cursor.read_u32() for _ in range(cursor.read_varint()))
    winner = cursor.read_optional_string()
    placements = cursor.read_optional_int()
    abort = None
    if cursor.read_bool():
        abort = AbortRecord(cursor.read_string(), cursor.read_string(), cursor.read_string())
    cursor.finish()
    return HexoRecord(
        game_id=game_id,
        seed=seed,
        scenario=scenario,
        status="completed" if status_byte == _STATUS_COMPLETED else "aborted",
        action_ids=action_ids,
        abort=abort,
        winner=winner,
        placements=placements,
    )


def _header_engine_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rules_version": int(metadata.get("rules_version", 0)),
        "backend": str(metadata.get("backend", "")),
    }


def _header_players(players: Sequence[object]) -> tuple[HexoRecordPlayer, ...]:
    roles = ("player0", "player1")
    out = []
    for index, player in enumerate(players):
        identity = getattr(player, "identity", player)
        out.append(
            HexoRecordPlayer(
                player_id=str(getattr(identity, "player_id")),
                role=roles[index] if index < len(roles) else f"player{index}",
                label=getattr(identity, "label", None),
            )
        )
    return tuple(out)


def _jsonable_scenario(scenario: object | None) -> object | None:
    if scenario is None:
        return None
    try:
        encoded = json.dumps(scenario, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("HexoRecordFile scenarios must be JSON-serializable") from exc
    return json.loads(encoded)


def _append_optional_json(buffer: bytearray, value: object | None) -> None:
    if value is None:
        buffer.append(0)
        return
    buffer.append(1)
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    _append_bytes(buffer, encoded)


def _append_optional_int(buffer: bytearray, value: int | None) -> None:
    if value is None:
        buffer.append(0)
        return
    buffer.append(1)
    _append_varint(buffer, _zigzag_encode(int(value)))


def _append_optional_string(buffer: bytearray, value: str | None) -> None:
    if value is None:
        buffer.append(0)
        return
    buffer.append(1)
    _append_string(buffer, value)


def _append_string(buffer: bytearray, value: str) -> None:
    _append_bytes(buffer, value.encode("utf-8"))


def _append_bytes(buffer: bytearray, value: bytes) -> None:
    _append_varint(buffer, len(value))
    buffer.extend(value)


def _append_u32(buffer: bytearray, value: int) -> None:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"packed action id outside u32 range: {value}")
    buffer.extend(int(value).to_bytes(4, "little", signed=False))


def _append_varint(buffer: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    while value >= 0x80:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    buffer.append(value)


def _write_varint(handle: BinaryIO, value: int) -> None:
    buffer = bytearray()
    _append_varint(buffer, value)
    handle.write(buffer)


def _write_string(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    _write_varint(handle, len(encoded))
    handle.write(encoded)


def _write_optional_string(handle: BinaryIO, value: str | None) -> None:
    handle.write(b"\x00" if value is None else b"\x01")
    if value is not None:
        _write_string(handle, value)


def _read_varint(handle: BinaryIO) -> int:
    shift = 0
    value = 0
    while True:
        raw = handle.read(1)
        if raw == b"":
            raise EOFError("unexpected EOF while reading varint")
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value
        shift += 7
        if shift > 63:
            raise ValueError("varint is too large")


def _read_string(handle: BinaryIO) -> str:
    length = _read_varint(handle)
    encoded = handle.read(length)
    if len(encoded) != length:
        raise EOFError("unexpected EOF while reading string")
    return encoded.decode("utf-8")


def _read_optional_string(handle: BinaryIO) -> str | None:
    flag = handle.read(1)
    if flag == b"\x00":
        return None
    if flag != b"\x01":
        raise ValueError("invalid optional string flag")
    return _read_string(handle)


def _zigzag_encode(value: int) -> int:
    return value * 2 if value >= 0 else (-value * 2) - 1


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


class _PayloadCursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def read_byte(self) -> int:
        if self.offset >= len(self.payload):
            raise EOFError("unexpected EOF while reading byte")
        value = self.payload[self.offset]
        self.offset += 1
        return value

    def read_bool(self) -> bool:
        value = self.read_byte()
        if value not in (0, 1):
            raise ValueError("invalid boolean flag")
        return bool(value)

    def read_varint(self) -> int:
        shift = 0
        value = 0
        while True:
            byte = self.read_byte()
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7
            if shift > 63:
                raise ValueError("varint is too large")

    def read_u32(self) -> int:
        end = self.offset + 4
        if end > len(self.payload):
            raise EOFError("unexpected EOF while reading u32")
        value = int.from_bytes(self.payload[self.offset : end], "little", signed=False)
        self.offset = end
        return value

    def read_bytes(self) -> bytes:
        length = self.read_varint()
        end = self.offset + length
        if end > len(self.payload):
            raise EOFError("unexpected EOF while reading bytes")
        value = self.payload[self.offset : end]
        self.offset = end
        return value

    def read_string(self) -> str:
        return self.read_bytes().decode("utf-8")

    def read_optional_string(self) -> str | None:
        return self.read_string() if self.read_bool() else None

    def read_optional_int(self) -> int | None:
        return _zigzag_decode(self.read_varint()) if self.read_bool() else None

    def read_optional_json(self) -> object | None:
        if not self.read_bool():
            return None
        return json.loads(self.read_bytes().decode("utf-8"))

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise ValueError("trailing bytes in HexoRecord payload")
