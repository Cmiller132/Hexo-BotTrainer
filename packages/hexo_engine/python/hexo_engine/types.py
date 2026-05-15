"""Lightweight Python types for the engine API boundary.

These objects describe the data Python callers should exchange with
`hexo_engine.api`. The actual rule interpretation stays in Rust; these types are
only handles, identifiers, and transport shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, TypeAlias


ActionId = str


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
