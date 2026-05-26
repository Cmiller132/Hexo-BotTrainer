"""Python API wrapper over the Rust Hexo engine bridge."""

from __future__ import annotations

from typing import Any, Mapping

from .errors import EngineUnavailableError, IllegalActionError
from .types import (
    Action,
    ActionId,
    AxialCoord,
    LegalActionId,
    LegalActions,
    Player,
    PlacementAction,
    PythonBoard,
    PythonHexoState,
    PythonMoveRecord,
    PythonPlacementRecord,
    PythonTerminal,
    PythonWindowEntry,
    PythonWindowKey,
    PythonWindowStore,
    TerminalResult,
    TransitionResult,
    TurnPhase,
)

try:
    from . import _rust
except ImportError as exc:  # pragma: no cover - exercised only in broken installs.
    _rust = None
    _RUST_IMPORT_ERROR = exc
else:
    _RUST_IMPORT_ERROR = None


HexoState = _rust.HexoState if _rust is not None else object


def new_game(*, seed: int | None = None, scenario: object | None = None) -> HexoState:
    """Create a new Rust-owned game state."""

    return _bridge().new_game(seed, scenario)


def clone_state(state: HexoState) -> HexoState:
    """Return an independent mutable Rust state clone."""

    return _bridge().clone_state(state)


def current_player(state: HexoState) -> Player:
    """Return the player to act."""

    return Player(_bridge().current_player(state))


def legal_actions(state: HexoState) -> LegalActions:
    """Return deterministic legal single-placement actions."""

    return LegalActions(_bridge().legal_action_ids(state))


def legal_action_ids(state: HexoState) -> tuple[LegalActionId, ...]:
    """Return compact deterministic legal action IDs."""

    return tuple(int(action_id) for action_id in _bridge().legal_action_ids(state))


def legal_action_count(state: HexoState) -> int:
    """Return the number of legal single-placement actions."""

    return int(_bridge().legal_action_count(state))


def is_legal_action(state: HexoState, action: Action) -> bool:
    """Return whether an action is legal in the current state."""

    coord = _placement_coord(action)
    return bool(_bridge().is_legal_action(state, coord.q, coord.r))


def apply_action(state: HexoState, action: Action) -> TransitionResult:
    """Apply an action through the Rust engine."""

    coord = _placement_coord(action)
    try:
        payload = _bridge().apply_action(state, coord.q, coord.r)
    except ValueError as exc:
        raise IllegalActionError(str(exc)) from exc

    return TransitionResult(
        next_player=Player(payload["next_player"]) if payload.get("next_player") else None,
        terminal=bool(payload["terminal"]),
        metadata=dict(payload.get("metadata", {})),
    )


def terminal(state: HexoState) -> TerminalResult | None:
    """Return terminal information when the game is complete."""

    return _terminal(_bridge().terminal(state))


def to_python_state(state: HexoState) -> PythonHexoState:
    """Return a read-only Python mirror of the Rust state."""

    payload = _bridge().to_python_state(state)
    board = payload["board"]
    return PythonHexoState(
        board=PythonBoard(
            stones=tuple(
                (_coord(item["coord"]), Player(item["player"]))
                for item in board["stones"]
            ),
            occupied=tuple(_coord(item) for item in board["occupied"]),
            legal=tuple(_coord(item) for item in board["legal"]),
            windows=PythonWindowStore(
                entries=tuple(_window_entry(item) for item in board["windows"])
            ),
        ),
        current_player=Player(payload["current_player"]),
        phase=TurnPhase(payload["phase"]),
        placements_made=int(payload["placements_made"]),
        terminal=_terminal(payload.get("terminal")),
        last_turn=_move_record(payload.get("last_turn")),
        placement_history=tuple(
            _placement_record(item) for item in payload["placement_history"]
        ),
        first_stone=_coord_or_none(payload.get("first_stone")),
    )


def action_id(action: Action) -> ActionId:
    """Return the stable identity for an action."""

    coord = _placement_coord(action)
    return _bridge().action_id(coord.q, coord.r)


def engine_metadata() -> dict[str, Any]:
    """Return engine and bridge metadata."""

    return dict(_bridge().engine_metadata())


def _bridge() -> Any:
    if _rust is None:
        raise EngineUnavailableError(
            f"hexo_engine Rust bridge is unavailable: {_RUST_IMPORT_ERROR}"
        )
    return _rust


def _placement_coord(action: Action) -> AxialCoord:
    if isinstance(action, PlacementAction):
        return action.coord
    raise IllegalActionError(f"Unsupported action type: {type(action).__name__}")


def _coord(payload: Mapping[str, Any]) -> AxialCoord:
    return AxialCoord(q=int(payload["q"]), r=int(payload["r"]))


def _coord_or_none(payload: Mapping[str, Any] | None) -> AxialCoord | None:
    if payload is None:
        return None
    return _coord(payload)


def _terminal(payload: Mapping[str, Any] | None) -> TerminalResult | None:
    if payload is None:
        return None
    return TerminalResult(
        winner=Player(payload["winner"]) if payload.get("winner") else None,
        reason=str(payload["reason"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _window_entry(payload: Mapping[str, Any]) -> PythonWindowEntry:
    return PythonWindowEntry(
        key=PythonWindowKey(
            start=_coord(payload["start"]),
            axis=str(payload["axis"]),
        ),
        masks=(int(payload["masks"][0]), int(payload["masks"][1])),
    )


def _move_record(payload: Mapping[str, Any] | None) -> PythonMoveRecord | None:
    if payload is None:
        return None
    return PythonMoveRecord(
        player=Player(payload["player"]),
        placements=tuple(_coord(item) for item in payload["placements"]),
    )


def _placement_record(payload: Mapping[str, Any]) -> PythonPlacementRecord:
    return PythonPlacementRecord(
        player=Player(payload["player"]),
        coord=_coord(payload["coord"]),
        phase=TurnPhase(payload["phase"]),
        placement_index=int(payload["placement_index"]),
        first_stone=_coord_or_none(payload.get("first_stone")),
    )
