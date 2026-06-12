"""Incremental standing-win ("win-in-1") tracker for Hexo self-play games.

Hexo is hex-grid Connect6: pure placement, win = 6-in-a-row along one of the
three hex axes Q=(1,0), R=(0,1), QR=(1,-1). A "standing win cell" for a player
is an EMPTY cell whose placement completes 6-in-a-row for that player, i.e. the
single missing cell of a 6-cell window holding 5 stones of that player and none
of the opponent.

The tracker mirrors the engine-verified window bookkeeping of
``scripts/_wf_r4_structure.py`` (its missed-win classification was validated
47/47 against the real Rust engine on main_3 marathon games): per placement,
only the 6 windows per axis that pass through the placed cell change (18 window
updates per stone), so maintaining the standing-win sets is O(1) per placement.
Cells leave the standing sets automatically when they become occupied or when
an opponent stone poisons the window, because every window containing the
placed cell is re-contributed.

Used by the self-play ``frozen_win_override`` lever (see ``selfplay.py``): in
crop-frozen marathon games a standing win whose every completion cell lies
outside the radius-20 input crop is invisible to the search for BOTH players;
the override plays it directly after engine verification.
"""

from __future__ import annotations

from collections import Counter

# The three hex axes a 6-in-a-row can lie on (matches engine tactics.rs and
# scripts/_wf_r4_structure.py).
AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))

_FULL = 0b111111
_POPCOUNT = tuple(bin(value).count("1") for value in range(64))
_PLAYER_INDEX = {"player0": 0, "player1": 1, 0: 0, 1: 1}


class IncrementalWinTracker:
    """Per-game incremental 6-cell-window tracker of standing win cells.

    ``add_stone(q, r, player)`` ingests one placement (players are ``0``/``1``
    or ``"player0"``/``"player1"``; every stone of the game must be fed in
    order). ``standing_win_cells(player)`` returns the set of empty ``(q, r)``
    cells that would complete 6-in-a-row for that player if placed now.
    """

    __slots__ = ("_masks", "_win_cells", "_stones")

    def __init__(self) -> None:
        # (axis_index, start_q, start_r) -> [player0_bitmask, player1_bitmask]
        # over the window's 6 cells (bit k = start + k * axis_vector).
        self._masks: dict[tuple[int, int, int], list[int]] = {}
        # player -> Counter{(q, r): number of windows this empty cell completes}
        self._win_cells: tuple[Counter, Counter] = (Counter(), Counter())
        self._stones = 0

    def __len__(self) -> int:
        return self._stones

    def add_stone(self, q: int, r: int, player: int | str) -> None:
        """Apply one placement for ``player`` and update the standing-win sets."""

        owner = _PLAYER_INDEX[player]
        q = int(q)
        r = int(r)
        masks = self._masks
        for axis_index, (vq, vr) in enumerate(AXES):
            for k in range(6):
                key = (axis_index, q - k * vq, r - k * vr)
                mask = masks.get(key)
                if mask is None:
                    mask = [0, 0]
                    masks[key] = mask
                else:
                    self._contribute(key, mask, -1)
                mask[owner] |= 1 << k
                self._contribute(key, mask, +1)
        self._stones += 1

    def standing_win_cells(self, player: int | str) -> set[tuple[int, int]]:
        """Empty cells whose placement completes 6-in-a-row for ``player`` now."""

        return set(self._win_cells[_PLAYER_INDEX[player]])

    def _contribute(self, key: tuple[int, int, int], mask: list[int], sign: int) -> None:
        """Add/remove (``sign`` +1/-1) one window's standing-win contributions.

        A window contributes one standing win cell for player ``p`` exactly when
        it holds 5 of ``p``'s stones and none of the opponent's; the cell is the
        window's single missing slot (necessarily empty: it holds no stone of
        either player). Ported from ``scripts/_wf_r4_structure.WinTracker._contrib``.
        """

        for player in (0, 1):
            if mask[1 - player] != 0:
                continue
            if _POPCOUNT[mask[player]] != 5:
                continue
            missing_bit = mask[player] ^ _FULL
            k = missing_bit.bit_length() - 1
            axis_index, start_q, start_r = key
            vq, vr = AXES[axis_index]
            cell = (start_q + k * vq, start_r + k * vr)
            win_cells = self._win_cells[player]
            win_cells[cell] += sign
            if win_cells[cell] == 0:
                del win_cells[cell]
