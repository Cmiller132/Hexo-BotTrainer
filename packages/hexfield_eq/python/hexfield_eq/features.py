"""Node features (F = 25) from stored position facts.

Expands feature rows from raw facts (the hexfield_compact_v1 columns): the
placement history (q, r, owner, placement_index), phase, and first stone. The
25 planes are: 11 kept scalars (indices 0-10; index 9 = distance-to-nearest-
stone), 12 graded per-(cell, axis) window planes (indices 11-22: 4 quantities
own_line / opp_line / own_live / opp_live x 3 axes Q/R/QR), and 2 scalar fork
planes (indices 23-24). The graded window planes are recomputed here from the
stored placement history (see ``window_features_for_cell``); the retired binary
hot / standing-win planes of the hexfield lineage are gone (see
docs/PLAN_D6_EQUIVARIANT_REWRITE.md §3, constants.py plane map).

Off-board / edge windows (design decision, OWNED here — not a bug): a length-6
window through a support cell that runs off the legal support is treated as
clean-and-empty — an absent cell contributes 0 own and 0 opp stones (the plan's
``None => counts 0 => clean+empty`` rule). So an edge cell's off-board windows
still count toward own_live / opp_live openness even though such a window can
never actually complete a line. This is a DELIBERATE, player-symmetric choice
(own and opp are treated identically), owned by plan §1.1; the featurizer does
not distinguish an off-board cell from an interior empty cell, and neither does
the Rust ``WindowStore`` path, so the two featurizers stay in exact parity.

Turn structure is deterministic (1-then-2-2-2...), so each history record's
phase and player are derived from its ordinal position; the shard schema
stores neither.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import (
    DIST_SCALE,
    F_DIST_TO_STONE,
    F_EMPTY,
    F_FIRST_STONE,
    F_LEGAL,
    F_OPP_FORK,
    F_OPP_LAST_TURN,
    F_OPP_RECENCY,
    F_OPP_STONE,
    F_OWN_FORK,
    F_OWN_LINE_Q,
    F_OWN_RECENCY,
    F_OWN_STONE,
    F_PHASE_SECOND,
    F_PLAYER_COLOUR,
    FORK_LINE_THRESHOLD,
    FORK_NORM,
    LINE_NORM,
    LIVE_NORM,
    NUM_FEATURES,
    RAY_REACH,
    RAYLEN_SLOTS,
    WINDOW_LEN,
)
from .geometry import apply_d6
from .support import Support, build_support

PHASE_OPENING = "Opening"
PHASE_FIRST = "FirstStone"
PHASE_SECOND = "SecondStone"

# Win axes: Q=(1,0), R=(0,1), QR=(1,-1).
AXIS_DELTAS: dict[str, tuple[int, int]] = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}


@dataclass(frozen=True)
class PositionFacts:
    """Raw facts for one decision state, side-relative where applicable.

    records: chronological placement history (q, r, owner, placement_index);
    owner is 0/1 (player0/player1). placements_made == len(records). The graded
    per-axis window planes are recomputed from ``records`` at feature-build time
    (no stored hot/win cell lists).
    """

    records: tuple[tuple[int, int, int, int], ...]
    current_player: int
    phase: str
    first_stone: tuple[int, int] | None

    @property
    def placements_made(self) -> int:
        return len(self.records)

    def stones(self) -> list[tuple[int, int]]:
        return [(q, r) for q, r, _owner, _idx in self.records]


def record_phase(ordinal: int) -> str:
    """Phase of the history record at chronological position ``ordinal``."""

    if ordinal == 0:
        return PHASE_OPENING
    return PHASE_FIRST if (ordinal - 1) % 2 == 0 else PHASE_SECOND


def record_player(ordinal: int) -> int:
    """Player (0/1) of the history record at position ``ordinal``."""

    if ordinal == 0:
        return 0
    return 1 if ((ordinal - 1) // 2) % 2 == 0 else 0


# --- graded per-axis window features --------------------------------------------

# Win axes in canonical order [Q, R, QR], matching Axis::ALL in the Rust store.
_AXES: tuple[tuple[int, int], ...] = (
    AXIS_DELTAS["Q"],
    AXIS_DELTAS["R"],
    AXIS_DELTAS["QR"],
)


def window_features_for_cell(
    owner_at: dict[tuple[int, int], int],
    xq: int,
    xr: int,
    me: int,
    other: int,
) -> list[float]:
    """The 14 graded window-feature values for one cell, plane order
    ``[own_line Q,R,QR, opp_line Q,R,QR, own_live Q,R,QR, opp_live Q,R,QR,
    own_fork, opp_fork]``.

    A literal transcription of the Rust ``window_feature_row`` (features.rs):
    for each axis and each of the 6 length-``WINDOW_LEN`` windows through the
    cell, count own/opp stones; a window is clean-for-me when ``opp == 0`` and
    clean-for-opp when ``own == 0``. ``line`` is the max clean count (``/5``),
    ``live`` the clean-window count (``/6``). The empty-at-x gate (require the
    cell empty in the window) is vacuous — always true for an empty cell,
    dropped for a stone — so it is folded into ``is_empty`` and never changes a
    value; it is kept explicit to mirror the Rust path. ``fork`` is
    ``|{axis : raw line >= FORK_LINE_THRESHOLD}| / 3``.
    """

    is_empty = (xq, xr) not in owner_at
    own_line_raw = [0, 0, 0]
    opp_line_raw = [0, 0, 0]
    out = [0.0] * 14
    for ai, (dq, dr) in enumerate(_AXES):
        own_max = 0
        opp_max = 0
        own_live = 0
        opp_live = 0
        for offset in range(WINDOW_LEN):
            sq = xq - dq * offset
            sr = xr - dr * offset
            own_c = 0
            opp_c = 0
            for i in range(WINDOW_LEN):
                owner = owner_at.get((sq + dq * i, sr + dr * i))
                if owner is None:
                    continue
                if owner == me:
                    own_c += 1
                elif owner == other:
                    opp_c += 1
            # empty_at_x == is_empty (x sits at position `offset`); the gate
            # never fires, but is written out for exact parity with Rust.
            empty_at_x = is_empty
            if is_empty and not empty_at_x:
                continue
            if opp_c == 0:
                own_live += 1
                if own_c > own_max:
                    own_max = own_c
            if own_c == 0:
                opp_live += 1
                if opp_c > opp_max:
                    opp_max = opp_c
        own_line_raw[ai] = own_max
        opp_line_raw[ai] = opp_max
        out[ai] = own_max / LINE_NORM
        out[3 + ai] = opp_max / LINE_NORM
        out[6 + ai] = own_live / LIVE_NORM
        out[9 + ai] = opp_live / LIVE_NORM
    out[12] = sum(1 for c in own_line_raw if c >= FORK_LINE_THRESHOLD) / FORK_NORM
    out[13] = sum(1 for c in opp_line_raw if c >= FORK_LINE_THRESHOLD) / FORK_NORM
    return out


def ray_lengths_for_cell(
    owner_at: dict[tuple[int, int], int],
    support,
    xq: int,
    xr: int,
    me: int,
) -> list[int]:
    """The RAYLEN_SLOTS side-relative ray lengths for one cell, flat index
    ``side*6 + axis*2 + dir`` with side in {own=0, opp=1} (own = the side to
    move), axis in [Q, R, QR] order, dir in {+=0, -=1}.

    A literal transcription of the Rust ``ray_length_row`` (features.rs), the
    L1 walk of docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md: walk j = 1..RAY_REACH
    from the cell; a cell off the support stops the walk (unattendable); an
    anti-side stone is INCLUDED (terminal blocker) then stops it; own-side
    stones and empties pass through. The occupancy of the cell itself is never
    consulted. ``support`` is any container answering ``(q, r) in support``.
    """

    out = [0] * RAYLEN_SLOTS
    for side in range(2):
        anti = (1 - me) if side == 0 else me
        for ai, (dq, dr) in enumerate(_AXES):
            for di, sign in ((0, 1), (1, -1)):
                length = 0
                for j in range(1, RAY_REACH + 1):
                    y = (xq + sign * dq * j, xr + sign * dr * j)
                    if y not in support:
                        break
                    length = j
                    if owner_at.get(y) == anti:
                        break
                out[side * 6 + ai * 2 + di] = length
    return out


def build_ray_lengths(facts: PositionFacts, sup: Support) -> np.ndarray:
    """(N, RAYLEN_SLOTS) uint8 ray lengths in support node order (the Python
    oracle for the serve/train Rust walks' 3-way parity tests)."""

    owner_at = {(q, r): owner for q, r, owner, _ in facts.records}
    me = facts.current_player
    out = np.zeros((sup.num_nodes, RAYLEN_SLOTS), dtype=np.uint8)
    coords = sup.coords
    for row in range(sup.num_nodes):
        out[row] = ray_lengths_for_cell(
            owner_at, sup.index, int(coords[row][0]), int(coords[row][1]), me
        )
    return out


def _fill_window_features(
    facts: PositionFacts, sup: Support, feats: np.ndarray
) -> None:
    """Populate the 12 axis planes + 2 fork planes (indices 11-24) in place."""

    me = facts.current_player
    other = 1 - facts.current_player
    owner_at = {(q, r): owner for q, r, owner, _ in facts.records}
    coords = sup.coords
    for row in range(sup.num_nodes):
        vals = window_features_for_cell(
            owner_at, int(coords[row][0]), int(coords[row][1]), me, other
        )
        feats[row, F_OWN_LINE_Q : F_OWN_LINE_Q + 12] = vals[:12]
        feats[row, F_OWN_FORK] = vals[12]
        feats[row, F_OPP_FORK] = vals[13]


# --- feature build ----------------------------------------------------------------


def build_features(facts: PositionFacts, sup: Support) -> np.ndarray:
    """(N, 25) float32 feature matrix in support node order."""

    n = sup.num_nodes
    feats = np.zeros((n, NUM_FEATURES), dtype=np.float32)

    placements_made = facts.placements_made
    for q, r, owner, placement_index in facts.records:
        row = sup.index[(q, r)]
        if owner == facts.current_player:
            feats[row, F_OWN_STONE] = 1.0
            recency_plane = F_OWN_RECENCY
        else:
            feats[row, F_OPP_STONE] = 1.0
            recency_plane = F_OPP_RECENCY
        # age = placements_made - placement_index; recency weight = 1/(1+age).
        age = placements_made - placement_index
        weight = 1.0 / (1.0 + float(age))
        feats[row, recency_plane] = max(feats[row, recency_plane], weight)

    feats[:, F_EMPTY] = 1.0 - feats[:, F_OWN_STONE] - feats[:, F_OPP_STONE]
    feats[: sup.legal_count, F_LEGAL] = 1.0

    if facts.phase == PHASE_SECOND:
        feats[:, F_PHASE_SECOND] = 1.0
        if facts.first_stone is not None:
            feats[sup.index[facts.first_stone], F_FIRST_STONE] = 1.0

    if facts.current_player == 0:
        feats[:, F_PLAYER_COLOUR] = 1.0

    _fill_window_features(facts, sup, feats)

    feats[:, F_DIST_TO_STONE] = sup.dist.astype(np.float32) / DIST_SCALE

    for cell in _opp_last_turn_cells(facts):
        feats[sup.index[cell], F_OPP_LAST_TURN] = 1.0

    return feats


def _opp_last_turn_cells(facts: PositionFacts) -> list[tuple[int, int]]:
    """Cells of the opponent's most recent full turn.

    Scans history in reverse. The first opponent SecondStone record returns
    both its own cell and the previous record's cell (its first-stone
    companion); an opponent Opening record returns its single cell; opponent
    FirstStone records are skipped."""

    opponent = 1 - facts.current_player
    records = facts.records
    for ordinal in range(len(records) - 1, -1, -1):
        if record_player(ordinal) != opponent:
            continue
        phase = record_phase(ordinal)
        q, r, _owner, _idx = records[ordinal]
        if phase == PHASE_SECOND:
            fq, fr, _o, _i = records[ordinal - 1]
            return [(fq, fr), (q, r)]
        if phase == PHASE_OPENING:
            return [(q, r)]
    return []


def build_position(facts: PositionFacts) -> tuple[Support, np.ndarray]:
    """Support + features for one decision state."""

    sup = build_support(facts.stones())
    return sup, build_features(facts, sup)


# --- D6 augmentation ----------------------------------------------------------------


def transform_facts(facts: PositionFacts, sym: int) -> PositionFacts:
    """Apply D6 transform ``sym`` to every stored coordinate fact.

    Transforms the coordinates in records and first_stone; all other fields are
    copied unchanged. Support, node order, neighbour table, and the graded
    window features are rebuilt from the transformed records."""

    def t(cell: tuple[int, int]) -> tuple[int, int]:
        return apply_d6(sym, cell[0], cell[1])

    return replace(
        facts,
        records=tuple(
            (*t((q, r)), owner, idx) for q, r, owner, idx in facts.records
        ),
        first_stone=t(facts.first_stone) if facts.first_stone is not None else None,
    )
