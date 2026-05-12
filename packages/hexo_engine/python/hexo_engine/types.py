"""Lightweight Python types for the engine API boundary.

These objects describe the data Python callers should exchange with
`hexo_engine.api`. The actual rule interpretation stays in Rust; these types are
only handles, identifiers, and transport shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


StateId = str
ActionId = str
TacticalSummary = Mapping[str, Any]


class Player(StrEnum):
    """Canonical player labels exposed to Python callers."""

    ONE = "one"
    TWO = "two"


class TurnPhase(StrEnum):
    """High-level turn phase reported by the engine."""

    PLACE = "place"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class AxialCoord:
    """Axial hex coordinate passed through the Python API."""

    q: int
    r: int


@dataclass(frozen=True, slots=True)
class Action:
    """A player action submitted to the Rust engine for validation."""

    coord: AxialCoord


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
