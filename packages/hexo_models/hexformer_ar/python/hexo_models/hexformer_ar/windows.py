"""Window-token and tactical summary features for Hexformer AR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .coordinates import Axial, as_axial, line_cells, pack_action_id


@dataclass(frozen=True, slots=True)
class TacticalWindow:
    start: Axial
    axis: str
    masks: tuple[int, int]
    cells: tuple[Axial, ...]
    counts: tuple[int, int]
    empty_cells: tuple[Axial, ...]
    empty_action_ids: tuple[int, ...]
    threat_player: str | None
    immediate_win_action_ids: tuple[int, ...] = ()
    must_block_action_ids: tuple[int, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TacticalSummary:
    windows: tuple[TacticalWindow, ...]
    tactical_action_ids: tuple[int, ...]
    immediate_win_action_ids: tuple[int, ...]
    must_block_action_ids: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


def build_tactical_summary(python_state: object, legal_action_ids: Sequence[int]) -> TacticalSummary:
    legal_set = {int(action_id) for action_id in legal_action_ids}
    current = _player_label(getattr(python_state, "current_player", "player0"))
    opponent = "player1" if current == "player0" else "player0"
    windows: list[TacticalWindow] = []
    tactical_ids: set[int] = set()
    win_ids: set[int] = set()
    block_ids: set[int] = set()

    window_store = getattr(getattr(python_state, "board", object()), "windows", None)
    entries = getattr(window_store, "entries", ())
    for entry in entries:
        key = getattr(entry, "key", entry)
        start = as_axial(getattr(key, "start"))
        axis = str(getattr(key, "axis"))
        masks = tuple(int(item) for item in getattr(entry, "masks", (0, 0)))
        cells = line_cells(start, axis)
        occupied_mask = masks[0] | masks[1]
        empty = tuple(cell for index, cell in enumerate(cells) if not (occupied_mask & (1 << index)))
        empty_ids = tuple(pack_action_id(cell) for cell in empty if pack_action_id(cell) in legal_set)
        counts = (masks[0].bit_count(), masks[1].bit_count())
        threat_player = None
        immediate: tuple[int, ...] = ()
        must_block: tuple[int, ...] = ()

        if counts[0] > 0 and counts[1] == 0 and counts[0] >= 4:
            threat_player = "player0"
        elif counts[1] > 0 and counts[0] == 0 and counts[1] >= 4:
            threat_player = "player1"

        if threat_player is not None:
            tactical_ids.update(empty_ids)
            if threat_player == current and counts[0 if current == "player0" else 1] >= 5:
                immediate = empty_ids
                win_ids.update(immediate)
            if threat_player == opponent and counts[0 if opponent == "player0" else 1] >= 5:
                must_block = empty_ids
                block_ids.update(must_block)

        windows.append(
            TacticalWindow(
                start=start,
                axis=axis,
                masks=(masks[0], masks[1]),
                cells=cells,
                counts=counts,
                empty_cells=empty,
                empty_action_ids=empty_ids,
                threat_player=threat_player,
                immediate_win_action_ids=immediate,
                must_block_action_ids=must_block,
                metadata={"current_player": current},
            )
        )

    return TacticalSummary(
        windows=tuple(sorted(windows, key=lambda item: (item.start.q, item.start.r, item.axis))),
        tactical_action_ids=tuple(sorted(tactical_ids)),
        immediate_win_action_ids=tuple(sorted(win_ids)),
        must_block_action_ids=tuple(sorted(block_ids)),
        metadata={
            "window_count": len(windows),
            "tactical_action_count": len(tactical_ids),
            "immediate_win_count": len(win_ids),
            "must_block_count": len(block_ids),
        },
    )


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
