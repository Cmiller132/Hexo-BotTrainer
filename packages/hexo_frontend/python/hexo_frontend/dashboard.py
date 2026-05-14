"""Frontend-only shaping for the local manual match dashboard."""

from __future__ import annotations

from typing import Any


WIN_LENGTH = 6
AXIS_VECTORS = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}
WINDOW_MASK = (1 << WIN_LENGTH) - 1


def dashboard_state(raw: dict[str, object]) -> dict[str, object]:
    """Translate raw match/engine data into the browser dashboard shape."""

    engine_state = _mapping(raw.get("engine_state"))
    raw_tactics = _mapping(raw.get("tactics"))
    legal = list(raw.get("legal_actions") or [])
    placements = [_placement(record) for record in engine_state.get("placement_history", [])]
    terminal = _mapping(raw.get("terminal")) if raw.get("terminal") else None
    tactics = _dashboard_tactics(raw_tactics, engine_state, legal)

    return {
        "current_player": _player(engine_state.get("current_player")),
        "phase": _phase(engine_state.get("phase")),
        "first_stone": _first_stone(engine_state.get("phase")),
        "winner": _player(terminal.get("winner")) if terminal else None,
        "terminal_reason": terminal.get("reason") if terminal else None,
        "placements": placements,
        "legal": legal,
        "legal_count": len(legal),
        "tactics": tactics,
        "snapshot": raw.get("snapshot"),
    }


def _dashboard_tactics(
    raw_tactics: dict[str, Any],
    engine_state: dict[str, Any],
    legal: list[object],
) -> dict[str, object]:
    legal_coords = {_coord_key(_coord(coord)) for coord in legal}
    stone_owner = {
        _coord_key(_coord(stone.get("coord"))): _player(stone.get("stone"))
        for stone in _mapping(engine_state.get("board")).get("stones", [])
    }
    entries = _mapping(raw_tactics.get("window_store")).get("entries", [])
    windows = [_window(entry, stone_owner, legal_coords) for entry in entries]
    threats = [window for window in windows if window["is_threat"]]
    winning_windows = [window for window in windows if window["is_win"]]
    immediate_wins = _move_facts(windows, legal_coords, want_win=True)
    must_blocks = _must_blocks(windows, legal_coords)

    return {
        "raw": raw_tactics,
        "windows": windows,
        "window_count": len(windows),
        "threats": threats,
        "threat_count": len(threats),
        "winning_windows": winning_windows,
        "immediate_wins": immediate_wins,
        "must_blocks": must_blocks,
        "summary": {
            "active": sum(1 for window in windows if window["is_active"]),
            "blocked": sum(1 for window in windows if window["is_blocked"]),
            "threats": len(threats),
            "wins": len(winning_windows),
            "immediate_wins": len(immediate_wins),
            "must_blocks": len(must_blocks),
        },
    }


def _window(entry: object, stone_owner: dict[tuple[int, int], str | None], legal: set[tuple[int, int]]) -> dict[str, object]:
    data = _mapping(entry)
    key = _mapping(data.get("key"))
    start = _coord(key.get("start"))
    axis = str(key.get("axis"))
    masks = list(data.get("masks") or [0, 0])
    p0_mask = int(masks[0] if len(masks) > 0 else 0)
    p1_mask = int(masks[1] if len(masks) > 1 else 0)
    p0_count = p0_mask.bit_count()
    p1_count = p1_mask.bit_count()
    active_player = _active_player(p0_count, p1_count)
    threat_player = active_player if active_player and max(p0_count, p1_count) >= 4 else None
    own_count = max(p0_count, p1_count)
    cells = [_add(start, AXIS_VECTORS[axis], index) for index in range(WIN_LENGTH)]
    empty = [cell for cell in cells if _coord_key(cell) not in stone_owner]
    blockable = [cell for cell in empty if _coord_key(cell) in legal]

    return {
        "id": _window_id(start, axis),
        "key": {"start": start, "axis": axis},
        "axis": axis,
        "cells": [
            {"q": cell["q"], "r": cell["r"], "owner": stone_owner.get(_coord_key(cell)), "index": index}
            for index, cell in enumerate(cells)
        ],
        "mask": {
            "player0": p0_mask,
            "player1": p1_mask,
            "occupied": p0_mask | p1_mask,
            "empty": (~(p0_mask | p1_mask)) & WINDOW_MASK,
        },
        "counts": {
            "player0": p0_count,
            "player1": p1_count,
            "empty": len(empty),
            "occupied": p0_count + p1_count,
        },
        "active_player": active_player,
        "threat_player": threat_player,
        "player": threat_player or active_player,
        "own_count": own_count,
        "is_active": active_player is not None,
        "is_blocked": p0_count > 0 and p1_count > 0,
        "is_threat": threat_player is not None,
        "is_win": active_player is not None and own_count >= WIN_LENGTH,
        "severity": "win" if own_count >= WIN_LENGTH else "direct" if own_count == 5 else "threat" if own_count >= 4 else "active",
        "stone_cells": {
            "player0": [_mask_cell(cells, index) for index in range(WIN_LENGTH) if p0_mask & (1 << index)],
            "player1": [_mask_cell(cells, index) for index in range(WIN_LENGTH) if p1_mask & (1 << index)],
        },
        "empty_cells": empty,
        "blockable_cells": blockable,
        "blockable_now": bool(blockable),
    }


def _move_facts(windows: list[dict[str, object]], legal: set[tuple[int, int]], *, want_win: bool) -> list[dict[str, object]]:
    facts: dict[tuple[str, tuple[int, int]], set[str]] = {}
    target = WIN_LENGTH if want_win else WIN_LENGTH - 1
    for window in windows:
        if window["is_blocked"]:
            continue
        player = str(window["active_player"] or "")
        if not player:
            continue
        own_count = int(window["own_count"])
        for empty in window["empty_cells"]:
            coord = _coord(empty)
            key = _coord_key(coord)
            if key in legal and own_count + 1 >= target:
                facts.setdefault((player, key), set()).add(str(window["id"]))
    return [
        {"player": player, "q": coord[0], "r": coord[1], "window_ids": sorted(window_ids)}
        for (player, coord), window_ids in sorted(facts.items())
    ]


def _must_blocks(windows: list[dict[str, object]], legal: set[tuple[int, int]]) -> list[dict[str, object]]:
    blocks: dict[tuple[str, tuple[int, int]], set[str]] = {}
    for window in windows:
        if int(window["own_count"]) != 5 or not window["active_player"]:
            continue
        blocker = "player1" if window["active_player"] == "player0" else "player0"
        for empty in window["empty_cells"]:
            coord = _coord(empty)
            key = _coord_key(coord)
            if key in legal:
                blocks.setdefault((blocker, key), set()).add(str(window["id"]))
    return [
        {"player": player, "q": coord[0], "r": coord[1], "window_ids": sorted(window_ids)}
        for (player, coord), window_ids in sorted(blocks.items())
    ]


def _placement(raw: object) -> dict[str, object]:
    record = _mapping(raw)
    coord = _coord(record.get("coord"))
    return {
        "q": coord["q"],
        "r": coord["r"],
        "player": _player(record.get("player")),
        "phase": _phase(record.get("phase")),
        "index": int(record.get("placement_index") or 0),
    }


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coord(value: object) -> dict[str, int]:
    data = _mapping(value)
    return {"q": int(data.get("q", 0)), "r": int(data.get("r", 0))}


def _coord_key(coord: dict[str, int]) -> tuple[int, int]:
    return (coord["q"], coord["r"])


def _add(coord: dict[str, int], vector: tuple[int, int], scale: int) -> dict[str, int]:
    return {"q": coord["q"] + vector[0] * scale, "r": coord["r"] + vector[1] * scale}


def _mask_cell(cells: list[dict[str, int]], index: int) -> dict[str, int]:
    return cells[index]


def _window_id(start: dict[str, int], axis: str) -> str:
    return f"{axis}:{start['q']},{start['r']}"


def _player(value: object) -> str | None:
    return {"Player0": "player0", "Player1": "player1", "player0": "player0", "player1": "player1"}.get(str(value))


def _phase(value: object) -> str:
    if value == "Opening":
        return "opening"
    if value == "FirstStone":
        return "first_stone"
    return "second_stone" if isinstance(value, dict) and "SecondStone" in value else "first_stone"


def _first_stone(value: object) -> dict[str, int] | None:
    if isinstance(value, dict) and "SecondStone" in value:
        return _coord(_mapping(value["SecondStone"]).get("first"))
    return None


def _active_player(p0_count: int, p1_count: int) -> str | None:
    if p0_count and not p1_count:
        return "player0"
    if p1_count and not p0_count:
        return "player1"
    return None
