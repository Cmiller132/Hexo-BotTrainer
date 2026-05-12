"""Thin Python API over the Rust engine.

This module is the host-facing surface for game creation, legal action lookup,
state transitions, snapshots, tactics, and stable identities. It should remain a
wrapper over the Rust rules authority instead of growing a parallel Python rules
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from .errors import EngineUnavailableError
from .types import (
    Action,
    ActionId,
    EngineSnapshot,
    Player,
    StateId,
    TacticalSummary,
    TerminalResult,
    TransitionResult,
    TurnPlacement,
)


@dataclass(frozen=True, slots=True)
class EngineStateRef:
    """Opaque reference to Rust-owned engine state."""

    inner: object | None = None


def _not_bound(operation: str) -> NoReturn:
    raise EngineUnavailableError(
        f"{operation} requires the Rust engine Python binding to be connected."
    )


def new_game(*, seed: int | None = None, scenario: object | None = None) -> EngineStateRef:
    """Create a new Rust-owned game state."""

    _not_bound("new_game")


def load_snapshot(snapshot: EngineSnapshot) -> EngineStateRef:
    """Load a replayable engine snapshot into Rust-owned state."""

    _not_bound("load_snapshot")


def snapshot(state: EngineStateRef) -> EngineSnapshot:
    """Return a replayable snapshot for the current state."""

    _not_bound("snapshot")


def current_player(state: EngineStateRef) -> Player:
    """Return the player to act, as reported by the engine."""

    _not_bound("current_player")


def turn_placement(state: EngineStateRef) -> TurnPlacement:
    """Return the current placement slot, as reported by the engine."""

    _not_bound("turn_placement")


def legal_actions(state: EngineStateRef) -> list[Action]:
    """Return legal actions from the Rust rules authority."""

    _not_bound("legal_actions")


def validate_action(state: EngineStateRef, action: Action) -> None:
    """Ask Rust to validate an action, raising on illegal input."""

    _not_bound("validate_action")


def apply_action(state: EngineStateRef, action: Action) -> TransitionResult:
    """Apply an accepted action through the Rust engine."""

    _not_bound("apply_action")


def terminal(state: EngineStateRef) -> TerminalResult | None:
    """Return terminal information when the game is complete."""

    _not_bound("terminal")


def tactics(state: EngineStateRef) -> TacticalSummary:
    """Return rules-derived tactical facts for models, runner, or search."""

    _not_bound("tactics")


def state_id(state: EngineStateRef) -> StateId:
    """Return the stable identity for state caches and diagnostics."""

    _not_bound("state_id")


def action_id(action: Action) -> ActionId:
    """Return the stable identity for an action."""

    _not_bound("action_id")


def engine_metadata() -> dict[str, Any]:
    """Return engine version and binding metadata once the bridge exists."""

    _not_bound("engine_metadata")
