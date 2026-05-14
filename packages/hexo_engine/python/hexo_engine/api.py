"""Thin Python API over the Hexo engine boundary.

The Rust bridge is still being wired. Until it is available, this module keeps a
small Python implementation behind the same public API so callers can build
against the intended engine contract instead of duplicating rules in UI code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from .errors import EngineUnavailableError
from .types import (
    Action,
    ActionId,
    AxialCoord,
    EngineSnapshot,
    PairAction,
    Player,
    PlacementAction,
    StateId,
    TacticalSummary,
    TerminalResult,
    TransitionResult,
    TurnPlacement,
)


SNAPSHOT_VERSION = 1
FRONTIER_RADIUS = 8
WIN_LENGTH = 6
DIRECTIONS = ((1, 0), (0, 1), (1, -1))
AXES = (("Q", (1, 0)), ("R", (0, 1)), ("QR", (1, -1)))


@dataclass(frozen=True, slots=True)
class EngineStateRef:
    """Opaque reference to engine-owned state."""

    inner: object | None = None


@dataclass(frozen=True, slots=True, order=True)
class _Coord:
    q: int
    r: int


@dataclass(frozen=True, slots=True)
class _PlacementRecord:
    player: Player
    coord: _Coord
    phase: TurnPlacement
    index: int


class _EngineState:
    def __init__(self) -> None:
        self.stones: dict[_Coord, Player] = {}
        self.frontier: set[_Coord] = set()
        self.history: list[_PlacementRecord] = []
        self.current_player = Player.PLAYER_0
        self.turn_slot = TurnPlacement.PLACEMENT_0
        self.first_stone: _Coord | None = None
        self.outcome: TerminalResult | None = None
        self.last_window_update: dict[str, list[dict[str, object]]] = {
            "changed": [],
            "threats": [],
            "winning_windows": [],
        }


def _state(ref: EngineStateRef) -> _EngineState:
    if isinstance(ref.inner, _EngineState):
        return ref.inner
    raise EngineUnavailableError("EngineStateRef is not backed by a usable engine state.")


def new_game(*, seed: int | None = None, scenario: object | None = None) -> EngineStateRef:
    """Create a new engine-owned game state."""

    return EngineStateRef(_EngineState())


def load_snapshot(snapshot: EngineSnapshot) -> EngineStateRef:
    """Load a replayable engine snapshot into engine-owned state."""

    if snapshot.version != SNAPSHOT_VERSION:
        from .errors import IncompatibleSnapshotError

        raise IncompatibleSnapshotError(
            f"Snapshot version {snapshot.version} is not supported by this engine."
        )

    state = _EngineState()
    for raw in snapshot.payload.get("placements", []):
        if isinstance(raw, Mapping):
            q = int(raw["q"])
            r = int(raw["r"])
        else:
            q = int(raw[0])
            r = int(raw[1])
        coord = _Coord(q, r)
        if coord not in set(_legal_coords(state)):
            from .errors import SnapshotError

            raise SnapshotError(f"Snapshot contains illegal placement {q},{r}.")
        _apply_placement(state, coord)
    return EngineStateRef(state)


def snapshot(state: EngineStateRef) -> EngineSnapshot:
    """Return a replayable snapshot for the current state."""

    engine_state = _state(state)
    return EngineSnapshot(
        version=SNAPSHOT_VERSION,
        payload={
            "placements": [
                {"q": record.coord.q, "r": record.coord.r} for record in engine_state.history
            ],
            "history": [_record_payload(record) for record in engine_state.history],
        },
    )


def current_player(state: EngineStateRef) -> Player:
    """Return the player to act, as reported by the engine."""

    return _state(state).current_player


def turn_placement(state: EngineStateRef) -> TurnPlacement:
    """Return the current placement slot, as reported by the engine."""

    return _state(state).turn_slot


def legal_actions(state: EngineStateRef) -> list[Action]:
    """Return legal actions from the Rust rules authority."""

    return [PlacementAction(AxialCoord(coord.q, coord.r)) for coord in _legal_coords(_state(state))]


def validate_action(state: EngineStateRef, action: Action) -> None:
    """Ask Rust to validate an action, raising on illegal input."""

    engine_state = _state(state)
    for coord in _action_coords(action):
        if coord not in set(_legal_coords(engine_state)):
            from .errors import IllegalActionError

            raise IllegalActionError(f"{coord.q},{coord.r} is not legal.")


def apply_action(state: EngineStateRef, action: Action) -> TransitionResult:
    """Apply an accepted action through the Rust engine."""

    engine_state = _state(state)
    for coord in _action_coords(action):
        validate_action(state, PlacementAction(AxialCoord(coord.q, coord.r)))
        _apply_placement(engine_state, coord)
        if engine_state.outcome is not None:
            break

    return TransitionResult(
        snapshot=snapshot(state),
        next_player=None if engine_state.outcome else engine_state.current_player,
        terminal=engine_state.outcome is not None,
        metadata={
            "turn_placement": engine_state.turn_slot.value,
            "placements": [_record_payload(record) for record in engine_state.history],
        },
    )


def terminal(state: EngineStateRef) -> TerminalResult | None:
    """Return terminal information when the game is complete."""

    return _state(state).outcome


def game_state(state: EngineStateRef) -> dict[str, object]:
    """Return the engine-owned state shape without dashboard interpretation."""

    engine_state = _state(state)
    return {
        "current_player": _raw_player(engine_state.current_player),
        "phase": _raw_turn_phase(engine_state.turn_slot, engine_state.first_stone, len(engine_state.history)),
        "placements_made": len(engine_state.history),
        "terminal": _raw_terminal(engine_state.outcome),
        "last_turn": _raw_last_turn(engine_state),
        "placement_history": [
            _raw_record(record, engine_state.history[index - 1] if index else None)
            for index, record in enumerate(engine_state.history)
        ],
        "board": {
            "stones": [
                {"coord": _coord_payload(record.coord), "stone": _raw_player(record.player)}
                for record in engine_state.history
            ],
            "occupied": [_coord_payload(record.coord) for record in engine_state.history],
            "frontier": [_coord_payload(coord) for coord in sorted(engine_state.frontier)],
        },
    }


def tactics(state: EngineStateRef) -> TacticalSummary:
    """Return the raw window store/update data maintained by the engine."""

    engine_state = _state(state)
    entries = _raw_window_entries(engine_state)
    return {
        "window_store": {
            "len": len(entries),
            "is_empty": not entries,
            "entries": entries,
        },
        "last_update": engine_state.last_window_update,
    }


def state_id(state: EngineStateRef) -> StateId:
    """Return the stable identity for state caches and diagnostics."""

    payload = "|".join(
        f"{record.player.value}:{record.coord.q},{record.coord.r}" for record in _state(state).history
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def action_id(action: Action) -> ActionId:
    """Return the stable identity for an action."""

    return ";".join(f"{coord.q},{coord.r}" for coord in _action_coords(action))


def engine_metadata() -> dict[str, Any]:
    """Return engine version and binding metadata once the bridge exists."""

    return {"engine_api": True, "backend": "python-fallback", "snapshot_version": SNAPSHOT_VERSION}


def _action_coords(action: Action) -> list[_Coord]:
    if isinstance(action, PlacementAction):
        return [_Coord(action.coord.q, action.coord.r)]
    if isinstance(action, PairAction):
        return [_Coord(coord.q, coord.r) for coord in action.placements]
    from .errors import IllegalActionError

    raise IllegalActionError(f"Unsupported action type: {type(action).__name__}")


def _legal_coords(state: _EngineState) -> list[_Coord]:
    if state.outcome is not None:
        return []
    if not state.history:
        return [_Coord(0, 0)] if _Coord(0, 0) not in state.stones else []
    return sorted(state.frontier)


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
    _update_frontier(state, coord)
    state.last_window_update = _raw_window_update(state, coord, player)

    if _has_six_in_line(state, coord, player):
        state.outcome = TerminalResult(
            winner=player,
            reason="six_in_line",
            metadata={"placements": len(state.history)},
        )
        return

    if len(state.history) == 1:
        state.current_player = Player.PLAYER_1
        state.turn_slot = TurnPlacement.PLACEMENT_0
        state.first_stone = None
    elif phase == TurnPlacement.PLACEMENT_0:
        state.turn_slot = TurnPlacement.PLACEMENT_1
        state.first_stone = coord
    else:
        state.current_player = Player.PLAYER_0 if player == Player.PLAYER_1 else Player.PLAYER_1
        state.turn_slot = TurnPlacement.PLACEMENT_0
        state.first_stone = None


def _update_frontier(state: _EngineState, coord: _Coord) -> None:
    state.frontier.discard(coord)
    for candidate in _coords_within_radius(coord, FRONTIER_RADIUS):
        if candidate not in state.stones:
            state.frontier.add(candidate)


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


def _coord_payload(coord: _Coord | None) -> dict[str, int] | None:
    if coord is None:
        return None
    return {"q": coord.q, "r": coord.r}


def _record_payload(record: _PlacementRecord) -> dict[str, object]:
    return {
        "q": record.coord.q,
        "r": record.coord.r,
        "player": record.player.value,
        "phase": record.phase.value,
        "index": record.index,
    }


def _raw_player(player: Player) -> str:
    return "Player0" if player == Player.PLAYER_0 else "Player1"


def _raw_turn_phase(placement: TurnPlacement, first: _Coord | None, history_len: int) -> object:
    if history_len == 0:
        return "Opening"
    if placement == TurnPlacement.PLACEMENT_0:
        return "FirstStone"
    return {"SecondStone": {"first": _coord_payload(first)}}


def _raw_terminal(outcome: TerminalResult | None) -> dict[str, object] | None:
    if outcome is None or outcome.winner is None:
        return None
    return {
        "winner": _raw_player(outcome.winner),
        "placements": outcome.metadata.get("placements"),
    }


def _raw_record(record: _PlacementRecord, previous: _PlacementRecord | None) -> dict[str, object]:
    first = previous.coord if record.phase == TurnPlacement.PLACEMENT_1 and previous else record.coord
    return {
        "player": _raw_player(record.player),
        "coord": _coord_payload(record.coord),
        "phase": _raw_turn_phase(record.phase, first, record.index - 1),
        "placement_index": record.index,
    }


def _raw_last_turn(state: _EngineState) -> dict[str, object] | None:
    if not state.history:
        return None
    if len(state.history) == 1:
        record = state.history[-1]
        return {"player": _raw_player(record.player), "placements": [_coord_payload(record.coord)]}
    record = state.history[-1]
    if record.phase == TurnPlacement.PLACEMENT_1 and len(state.history) >= 2:
        previous = state.history[-2]
        return {
            "player": _raw_player(record.player),
            "placements": [_coord_payload(previous.coord), _coord_payload(record.coord)],
        }
    return {"player": _raw_player(record.player), "placements": [_coord_payload(record.coord)]}


def _raw_window_entries(state: _EngineState) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for start, axis_name, vector in _known_window_keys(state):
        cells = [_add(start, vector, index) for index in range(WIN_LENGTH)]
        entries.append(
            {
                "key": {
                    "start": _coord_payload(start),
                    "axis": axis_name,
                },
                "masks": [
                    _mask_for_player(state, cells, Player.PLAYER_0),
                    _mask_for_player(state, cells, Player.PLAYER_1),
                ],
            }
        )
    return entries


def _raw_window_update(state: _EngineState, coord: _Coord, player: Player) -> dict[str, list[dict[str, object]]]:
    changed: list[dict[str, object]] = []
    threats: list[dict[str, object]] = []
    winning_windows: list[dict[str, object]] = []
    for axis_name, vector in AXES:
        for offset in range(WIN_LENGTH):
            start = _add(coord, vector, -offset)
            key = {"start": _coord_payload(start), "axis": axis_name}
            changed.append(key)
            cells = [_add(start, vector, index) for index in range(WIN_LENGTH)]
            mask = _mask_for_player(state, cells, player)
            if mask.bit_count() >= 4 and _active_player(
                _mask_for_player(state, cells, Player.PLAYER_0).bit_count(),
                _mask_for_player(state, cells, Player.PLAYER_1).bit_count(),
            ) == player:
                threats.append(key)
            if mask == (1 << WIN_LENGTH) - 1:
                winning_windows.append(key)
    return {"changed": changed, "threats": threats, "winning_windows": winning_windows}


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


def _active_player(player0_count: int, player1_count: int) -> Player | None:
    if player0_count and not player1_count:
        return Player.PLAYER_0
    if player1_count and not player0_count:
        return Player.PLAYER_1
    return None


def _add(coord: _Coord, vector: tuple[int, int], scale: int) -> _Coord:
    return _Coord(coord.q + vector[0] * scale, coord.r + vector[1] * scale)
