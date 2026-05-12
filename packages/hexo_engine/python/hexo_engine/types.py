"""Lightweight Python types for the engine API boundary.

These objects describe the data Python callers should exchange with
`hexo_engine.api`. The actual rule interpretation stays in Rust; these types are
only handles, identifiers, and transport shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, TypeAlias


StateId = str
ActionId = str
TacticalSummary = Mapping[str, Any]


class Player(StrEnum):
    """Canonical player labels exposed to Python callers."""

    PLAYER_0 = "player0"
    PLAYER_1 = "player1"


class TurnPlacement(StrEnum):
    """Placement slot inside the current logical turn."""

    PLACEMENT_0 = "placement_0"
    PLACEMENT_1 = "placement_1"


@dataclass(frozen=True, slots=True)
class AxialCoord:
    """Axial hex coordinate passed through the Python API."""

    q: int
    r: int

@dataclass(frozen=True, slots=True)
class PlacementAction:
    """One single-placement action submitted to the Rust engine."""

    coord: AxialCoord


@dataclass(frozen=True, slots=True)
class PairAction:
    """Unordered two-placement convenience action.

    The input tuple names the two requested cells; it does not imply
    application order. The engine boundary should resolve the pair into single
    placements deterministically for the current state and record the resolved
    order. If the first resolved placement wins the game, the second placement
    is discarded and must not be applied.
    """

    placements: tuple[AxialCoord, AxialCoord]


Action: TypeAlias = PlacementAction | PairAction


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Replayable state payload produced and consumed by the engine."""

    version: int
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Result returned after the engine accepts and applies an action."""

    snapshot: EngineSnapshot
    next_player: Player | None
    terminal: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Terminal state summary reported by the engine."""

    winner: Player | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
