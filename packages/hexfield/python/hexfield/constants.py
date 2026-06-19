"""hexfield constants — feature indices, geometry radii, bias-table layout.

Single source for every magic number shared by the Python featurizer, the
model, the wire ABI, and (via parity fixtures) the Rust serve-time featurizer.
"""

from __future__ import annotations

# --- engine-contract geometry -------------------------------------------------
# Legality: empty ∧ hex-dist <= LEGAL_RADIUS of any stone (engine legal.rs
# LEGAL_RADIUS == 8); Opening => forced {(0, 0)}. The halo is exactly the
# distance-(LEGAL_RADIUS + 1) shell (a property test, not a construction step).
LEGAL_RADIUS = 8
HALO_DIST = LEGAL_RADIUS + 1

# Fixed direction order D: the rotate60 orbit of (1, 0).
# rot60(D[i]) == D[(i + 1) % 6]; reflect(D[i]) == D[5 - i] (tests only).
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)

# Packed action id: ((q + 2^15) << 16) | (r + 2^15) — byte-identical to engine
# legal.rs pack_coord and hexo_engine.types.pack_coord_id (cross-checked in
# tests). Integer order of ids == ascending signed (q, r) order.
COORD_OFFSET = 1 << 15

# Missing-neighbour sentinel on the u16 wire (`nbr` ABI buffer). The Python
# featurizer uses -1; the wire/batching layer maps -1 -> the padded zero row.
NBR_SENTINEL_U16 = 0xFFFF

# --- node features (F = 15) ---------------------------------------------------
# Indices 0-12 are the plane semantics, with index 11 = distance-to-nearest-stone.
# Indices 13-14 are the engine-exact standing-win planes.
F_OWN_STONE = 0
F_OPP_STONE = 1
F_EMPTY = 2
F_LEGAL = 3
F_PHASE_SECOND = 4
F_FIRST_STONE = 5
F_PLAYER_COLOUR = 6
F_OWN_RECENCY = 7
F_OPP_RECENCY = 8
F_OPP_HOT = 9
F_OWN_HOT = 10
F_DIST_TO_STONE = 11
F_OPP_LAST_TURN = 12
F_OPP_WIN_NOW = 13
F_OWN_WIN_NOW = 14
NUM_FEATURES = 15

# Window thresholds: hot == the TSS threat definition (count >= 4, one concept
# repo-wide, threats_shared.rs); standing win == count == 5 (its single empty
# is a win-in-1 cell). The hot gate is exact, not a heuristic: the first
# possible count-4 single-colour window appears after placement 7.
HOT_MIN_COUNT = 4
WIN_NOW_COUNT = 5
HOT_MIN_PLACEMENTS = 7
WINDOW_LEN = 6

# dist_to_stone feature scaling: stones -> 0, legal in (0, 1], halo -> 1.125
# exactly (9/8, exactly representable in f16). Ply 0 => 0 everywhere.
DIST_SCALE = float(LEGAL_RADIUS)
HALO_DIST_FEATURE = HALO_DIST / DIST_SCALE  # 1.125

# --- heads / targets ------------------------------------------------------------
VALUE_BINS = 65
MOVES_LEFT_CAP = 209  # v3: measured p99.5 of main_2 recorded moves_left (max 254)

# --- trunk ----------------------------------------------------------------------
CHANNELS = 96
NUM_TOKENS = 8
ATTENTION_HEADS = 4
HEAD_DIM = CHANNELS // ATTENTION_HEADS  # 24
MLP_RATIO = 2

# --- relative-position bias table (per-block learned tables) --------------------
# rows 0-216:  exact axial offsets with hex-dist <= 8 (the 217-offset disk LUT)
# rows 217-224: on-win-axis ring buckets, hex-dist 9-16
# rows 225-232: off-axis ring buckets, hex-dist 9-16
# row  233:    far bucket, hex-dist >= 17
# rows 234/235/236: (query=cell,key=token) / (query=token,key=cell) / (token,token)
BIAS_DISK_RADIUS = LEGAL_RADIUS
BIAS_EXACT_ROWS = 217
BIAS_RING_MIN = 9
BIAS_RING_MAX = 16
BIAS_ON_AXIS_BASE = 217
BIAS_OFF_AXIS_BASE = 225
BIAS_FAR_ROW = 233
BIAS_CELL_TOKEN_ROW = 234
BIAS_TOKEN_CELL_ROW = 235
BIAS_TOKEN_TOKEN_ROW = 236
BIAS_ROWS = 237
