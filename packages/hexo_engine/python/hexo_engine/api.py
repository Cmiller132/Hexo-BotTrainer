"""Thin Python API over the Hexo engine boundary.

The Rust bridge is still being wired. Until it is available, this module keeps a
small Python implementation behind the same public API so callers can build
against the intended engine contract instead of duplicating rules in UI code.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import EngineUnavailableError
from .types import (
    Action,
    ActionId,
    AxialCoord,
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


LEGAL_RADIUS = 8
WIN_LENGTH = 6
DIRECTIONS = ((1, 0), (0, 1), (1, -1))
AXES = (("Q", (1, 0)), ("R", (0, 1)), ("QR", (1, -1)))


class _TurnPlacement(Enum):
    PLACEMENT_0 = "placement_0"
    PLACEMENT_1 = "placement_1"


@dataclass(frozen=True, slots=True)
class HexoState:
    """Opaque reference to engine-owned Hexo state."""

    inner: object | None = None


@dataclass(frozen=True, slots=True, order=True)
class _Coord:
    q: int
    r: int


@dataclass(frozen=True, slots=True)
class _PlacementRecord:
    player: Player
    coord: _Coord
    phase: _TurnPlacement
    index: int


class _EngineState:
    def __init__(self) -> None:
        self.stones: dict[_Coord, Player] = {}
        self.legal: set[_Coord] = set()
        self.history: list[_PlacementRecord] = []
        self.current_player = Player.PLAYER_0
        self.turn_slot = _TurnPlacement.PLACEMENT_0
        self.first_stone: _Coord | None = None
        self.outcome: TerminalResult | None = None


def _state(ref: HexoState) -> _EngineState:
    if isinstance(ref.inner, _EngineState):
        return ref.inner
    raise EngineUnavailableError("HexoState is not backed by a usable engine state.")


def new_game(*, seed: int | None = None, scenario: object | None = None) -> HexoState:
    """Create a new engine-owned game state."""

    return HexoState(_EngineState())


def clone_state(state: HexoState) -> HexoState:
    """Return an independent mutable copy of an engine state."""

    return HexoState(_clone_engine_state(_state(state)))


def current_player(state: HexoState) -> Player:
    """Return the player to act, as reported by the engine."""

    return _state(state).current_player


def legal_actions(state: HexoState) -> list[Action]:
    """Return legal actions from the Rust rules authority."""

    return [PlacementAction(AxialCoord(coord.q, coord.r)) for coord in _legal_coords(_state(state))]


def apply_action(state: HexoState, action: Action) -> TransitionResult:
    """Apply an accepted action through the Rust engine."""

    engine_state = _state(state)
    for coord in _validated_action_coords(engine_state, action):
        _apply_placement(engine_state, coord)
        if engine_state.outcome is not None:
            break

    return TransitionResult(
        next_player=None if engine_state.outcome else engine_state.current_player,
        terminal=engine_state.outcome is not None,
        metadata={
            "placements_made": len(engine_state.history),
        },
    )


def terminal(state: HexoState) -> TerminalResult | None:
    """Return terminal information when the game is complete."""

    return _state(state).outcome


def to_python_state(state: HexoState) -> PythonHexoState:
    """Return a read-only Python mirror of the engine-owned Hexo state."""

    engine_state = _state(state)
    occupied = tuple(_axial(record.coord) for record in engine_state.history)
    return PythonHexoState(
        board=PythonBoard(
            stones=tuple(
                sorted(
                    ((_axial(coord), player) for coord, player in engine_state.stones.items()),
                    key=lambda item: (item[0].q, item[0].r),
                )
            ),
            occupied=occupied,
            legal=tuple(_axial(coord) for coord in _legal_coords(engine_state)),
            windows=PythonWindowStore(entries=_python_window_entries(engine_state)),
        ),
        current_player=engine_state.current_player,
        phase=_turn_phase(engine_state),
        placements_made=len(engine_state.history),
        terminal=_python_terminal(engine_state.outcome),
        last_turn=_python_last_turn(engine_state),
        placement_history=tuple(
            _python_record(record, index, engine_state.history)
            for index, record in enumerate(engine_state.history)
        ),
        first_stone=_axial(engine_state.first_stone) if engine_state.first_stone is not None else None,
    )


def action_id(action: Action) -> ActionId:
    """Return the stable identity for an action."""

    return ";".join(f"{coord.q},{coord.r}" for coord in _action_coords(action))


def engine_metadata() -> dict[str, Any]:
    """Return engine version and binding metadata once the bridge exists."""

    return {"engine_api": True, "backend": "python-fallback"}


def _clone_engine_state(source: _EngineState) -> _EngineState:
    clone = _EngineState()
    clone.stones = dict(source.stones)
    clone.legal = set(source.legal)
    clone.history = list(source.history)
    clone.current_player = source.current_player
    clone.turn_slot = source.turn_slot
    clone.first_stone = source.first_stone
    clone.outcome = deepcopy(source.outcome)
    return clone


def _axial(coord: _Coord) -> AxialCoord:
    return AxialCoord(q=coord.q, r=coord.r)


def _turn_phase(state: _EngineState) -> TurnPhase:
    if not state.history:
        return TurnPhase.OPENING
    if state.turn_slot == _TurnPlacement.PLACEMENT_0:
        return TurnPhase.FIRST_STONE
    return TurnPhase.SECOND_STONE


def _placement_phase(record: _PlacementRecord, index: int) -> TurnPhase:
    if index == 0:
        return TurnPhase.OPENING
    if record.phase == _TurnPlacement.PLACEMENT_0:
        return TurnPhase.FIRST_STONE
    return TurnPhase.SECOND_STONE


def _python_terminal(outcome: TerminalResult | None) -> PythonTerminal | None:
    if outcome is None or outcome.winner is None:
        return None
    return PythonTerminal(
        winner=outcome.winner,
        placements=int(outcome.metadata.get("placements", 0)),
    )


def _python_last_turn(state: _EngineState) -> PythonMoveRecord | None:
    if not state.history:
        return None
    record = state.history[-1]
    if len(state.history) == 1:
        return PythonMoveRecord(player=record.player, placements=(_axial(record.coord),))
    if record.phase == _TurnPlacement.PLACEMENT_1 and len(state.history) >= 2:
        previous = state.history[-2]
        return PythonMoveRecord(
            player=record.player,
            placements=(_axial(previous.coord), _axial(record.coord)),
        )
    return PythonMoveRecord(player=record.player, placements=(_axial(record.coord),))


def _python_record(
    record: _PlacementRecord,
    index: int,
    history: list[_PlacementRecord],
) -> PythonPlacementRecord:
    phase = _placement_phase(record, index)
    first_stone = None
    if phase == TurnPhase.SECOND_STONE and index > 0:
        first_stone = _axial(history[index - 1].coord)
    return PythonPlacementRecord(
        player=record.player,
        coord=_axial(record.coord),
        phase=phase,
        placement_index=record.index,
        first_stone=first_stone,
    )


def _python_window_entries(state: _EngineState) -> tuple[PythonWindowEntry, ...]:
    entries: list[PythonWindowEntry] = []
    for start, axis_name, vector in _known_window_keys(state):
        cells = [_add(start, vector, index) for index in range(WIN_LENGTH)]
        entries.append(
            PythonWindowEntry(
                key=PythonWindowKey(start=_axial(start), axis=axis_name),
                masks=(
                    _mask_for_player(state, cells, Player.PLAYER_0),
                    _mask_for_player(state, cells, Player.PLAYER_1),
                ),
            )
        )
    return tuple(entries)


def _action_coords(action: Action) -> list[_Coord]:
    if isinstance(action, PlacementAction):
        return [_Coord(action.coord.q, action.coord.r)]
    from .errors import IllegalActionError

    raise IllegalActionError(f"Unsupported action type: {type(action).__name__}")


def _validated_action_coords(state: _EngineState, action: Action) -> list[_Coord]:
    coords = _action_coords(action)
    trial = _clone_engine_state(state)
    accepted: list[_Coord] = []
    for coord in coords:
        if coord not in set(_legal_coords(trial)):
            from .errors import IllegalActionError

            raise IllegalActionError(f"{coord.q},{coord.r} is not legal.")
        _apply_placement(trial, coord)
        accepted.append(coord)
        if trial.outcome is not None:
            break
    return accepted


def _legal_coords(state: _EngineState) -> list[_Coord]:
    if state.outcome is not None:
        return []
    if not state.history:
        return [_Coord(0, 0)] if _Coord(0, 0) not in state.stones else []
    return sorted(state.legal)


def _apply_placement(state: _EngineState, coord: _Coord) -> None:
    player = state.current_player
    phase = state.turn_slot
    state.stones[coord] = player
    state.history.append(
        _PlacementRecord(
            player=player,
            coord=coord,
            phase=phase,
            index=len(state.history) + 1,
        )
    )
    _update_legal(state, coord)

    if _has_six_in_line(state, coord, player):
        state.outcome = TerminalResult(
            winner=player,
            reason="six_in_line",
            metadata={"placements": len(state.history)},
        )
        return

    if len(state.history) == 1:
        state.current_player = Player.PLAYER_1
        state.turn_slot = _TurnPlacement.PLACEMENT_0
        state.first_stone = None
    elif phase == _TurnPlacement.PLACEMENT_0:
        state.turn_slot = _TurnPlacement.PLACEMENT_1
        state.first_stone = coord
    else:
        state.current_player = Player.PLAYER_0 if player == Player.PLAYER_1 else Player.PLAYER_1
        state.turn_slot = _TurnPlacement.PLACEMENT_0
        state.first_stone = None


def _update_legal(state: _EngineState, coord: _Coord) -> None:
    state.legal.discard(coord)
    for candidate in _coords_within_radius(coord, LEGAL_RADIUS):
        if candidate not in state.stones:
            state.legal.add(candidate)


def _coords_within_radius(center: _Coord, radius: int) -> list[_Coord]:
    coords: list[_Coord] = []
    for dq in range(-radius, radius + 1):
        r_min = max(-radius, -dq - radius)
        r_max = min(radius, -dq + radius)
        for dr in range(r_min, r_max + 1):
            coords.append(_Coord(center.q + dq, center.r + dr))
    return coords


def _has_six_in_line(state: _EngineState, coord: _Coord, player: Player) -> bool:
    for dq, dr in DIRECTIONS:
        count = 1
        count += _count_direction(state, coord, player, dq, dr)
        count += _count_direction(state, coord, player, -dq, -dr)
        if count >= WIN_LENGTH:
            return True
    return False


def _count_direction(state: _EngineState, coord: _Coord, player: Player, dq: int, dr: int) -> int:
    count = 0
    cursor = _Coord(coord.q + dq, coord.r + dr)
    while state.stones.get(cursor) == player:
        count += 1
        cursor = _Coord(cursor.q + dq, cursor.r + dr)
    return count


def _known_window_keys(state: _EngineState) -> list[tuple[_Coord, str, tuple[int, int]]]:
    keys: set[tuple[_Coord, str, tuple[int, int]]] = set()
    for coord in state.stones:
        for axis_name, vector in AXES:
            for offset in range(WIN_LENGTH):
                keys.add((_add(coord, vector, -offset), axis_name, vector))
    return sorted(keys, key=lambda item: (item[1], item[0].q, item[0].r))


def _mask_for_player(state: _EngineState, cells: list[_Coord], player: Player) -> int:
    mask = 0
    for index, coord in enumerate(cells):
        if state.stones.get(coord) == player:
            mask |= 1 << index
    return mask


def _add(coord: _Coord, vector: tuple[int, int], scale: int) -> _Coord:
    return _Coord(coord.q + vector[0] * scale, coord.r + vector[1] * scale)
