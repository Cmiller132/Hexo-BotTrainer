"""Lightweight Python types for the engine API boundary.

These objects describe the data Python callers should exchange with
`hexo_engine.api`. The actual rule interpretation stays in Rust; these types are
only handles, identifiers, and transport shapes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, TypeAlias


ActionId = str
LegalActionId = int
_COORD_OFFSET = 1 << 15
_COORD_MIN = -(1 << 15)
_COORD_MAX = (1 << 15) - 1


class Player(StrEnum):
    """Canonical player labels exposed to Python callers."""

    PLAYER_0 = "player0"
    PLAYER_1 = "player1"


class TurnPhase(StrEnum):
    """Rust-like phase of the autoregressive Hexo turn."""

    OPENING = "Opening"
    FIRST_STONE = "FirstStone"
    SECOND_STONE = "SecondStone"


@dataclass(frozen=True, slots=True)
class AxialCoord:
    """Axial hex coordinate passed through the Python API."""

    q: int
    r: int


@dataclass(frozen=True, slots=True)
class PlacementAction:
    """One single-placement action submitted to the Rust engine."""

    coord: AxialCoord


Action: TypeAlias = PlacementAction


class LegalActions(Sequence[PlacementAction]):
    """Deterministic legal action view backed by compact engine action IDs."""

    __slots__ = ("_ids", "_id_set")

    def __init__(self, action_ids: Sequence[int]) -> None:
        self._ids = tuple(int(action_id) for action_id in action_ids)
        self._id_set = frozenset(self._ids)

    @property
    def action_ids(self) -> tuple[LegalActionId, ...]:
        """Compact deterministic legal action IDs."""

        return self._ids

    def coords(self) -> tuple[AxialCoord, ...]:
        """Legal coordinates without wrapping them in action objects."""

        return tuple(unpack_coord_id(action_id) for action_id in self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[PlacementAction]:
        for action_id in self._ids:
            yield PlacementAction(unpack_coord_id(action_id))

    def __getitem__(self, index: int | slice) -> PlacementAction | tuple[PlacementAction, ...]:
        if isinstance(index, slice):
            return tuple(PlacementAction(unpack_coord_id(action_id)) for action_id in self._ids[index])
        return PlacementAction(unpack_coord_id(self._ids[index]))

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, PlacementAction):
            return False
        return pack_coord_id(item.coord) in self._id_set


def pack_coord_id(coord: AxialCoord) -> LegalActionId:
    """Pack a coordinate the same way the Rust legal move store does."""

    q = _checked_coord_component(coord.q)
    r = _checked_coord_component(coord.r)
    return ((q + _COORD_OFFSET) << 16) | (r + _COORD_OFFSET)


def unpack_coord_id(action_id: LegalActionId) -> AxialCoord:
    """Unpack a compact legal action ID into an axial coordinate."""

    action_id = int(action_id)
    q = (action_id >> 16) - _COORD_OFFSET
    r = (action_id & 0xFFFF) - _COORD_OFFSET
    return AxialCoord(q=q, r=r)


def _checked_coord_component(value: int) -> int:
    value = int(value)
    if value < _COORD_MIN or value > _COORD_MAX:
        raise ValueError(f"coordinate component outside i16 range: {value}")
    return value


@dataclass(frozen=True, slots=True)
class PythonTerminal:
    """Read-only Python mirror of Rust `GameOutcome`."""

    winner: Player
    placements: int


@dataclass(frozen=True, slots=True)
class PythonMoveRecord:
    """Read-only Python mirror of Rust `MoveRecord`."""

    player: Player
    placements: tuple[AxialCoord, ...]


@dataclass(frozen=True, slots=True)
class PythonPlacementRecord:
    """Read-only Python mirror of Rust `PlacementRecord`."""

    player: Player
    coord: AxialCoord
    phase: TurnPhase
    placement_index: int
    first_stone: AxialCoord | None = None


@dataclass(frozen=True, slots=True)
class PythonWindowKey:
    """Read-only Python mirror of Rust `WindowKey`."""

    start: AxialCoord
    axis: str


@dataclass(frozen=True, slots=True)
class PythonWindowEntry:
    """Read-only Python mirror of Rust `WindowEntry`."""

    key: PythonWindowKey
    masks: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PythonWindowStore:
    """Read-only Python mirror of Rust `WindowStore`."""

    entries: tuple[PythonWindowEntry, ...] = ()

    @property
    def len(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries


@dataclass(frozen=True, slots=True)
class PythonBoard:
    """Read-only Python mirror of Rust `Board`."""

    stones: tuple[tuple[AxialCoord, Player], ...]
    occupied: tuple[AxialCoord, ...]
    legal: tuple[AxialCoord, ...]
    windows: PythonWindowStore


@dataclass(frozen=True, slots=True)
class PythonHexoState:
    """Read-only Python mirror of Rust `HexoState`."""

    board: PythonBoard
    current_player: Player
    phase: TurnPhase
    placements_made: int
    terminal: PythonTerminal | None
    last_turn: PythonMoveRecord | None
    placement_history: tuple[PythonPlacementRecord, ...]
    first_stone: AxialCoord | None = None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Result returned after the engine accepts and applies an action."""

    next_player: Player | None
    terminal: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Terminal state summary reported by the engine."""

    winner: Player | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
