"""hexfield constants: feature indices, geometry radii, bias-table layout.

Shared by the Python featurizer, the model, the wire ABI, and (via parity
fixtures) the Rust serve-time featurizer.
"""

from __future__ import annotations

import os

# --- engine-contract geometry -------------------------------------------------
# Legality: empty and hex-dist <= LEGAL_RADIUS of any stone; opening move is
# forced to {(0, 0)}. LEGAL_RADIUS matches engine legal.rs. The halo is the
# distance-(LEGAL_RADIUS + 1) shell.
LEGAL_RADIUS = 8
HALO_DIST = LEGAL_RADIUS + 1

# Fixed direction order D: the rotate60 orbit of (1, 0).
# rot60(D[i]) == D[(i + 1) % 6]; reflect(D[i]) == D[5 - i].
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)

# Packed action id: ((q + 2^15) << 16) | (r + 2^15). Matches engine legal.rs
# pack_coord and hexo_engine.types.pack_coord_id. Integer order of ids equals
# ascending signed (q, r) order.
COORD_OFFSET = 1 << 15

# Missing-neighbour sentinel on the u16 wire (`nbr` ABI buffer). The Python
# featurizer uses -1; the wire/batching layer maps -1 to the padded zero row.
NBR_SENTINEL_U16 = 0xFFFF

# --- node features (F = 25) ---------------------------------------------------
# Planes 0-10 are the 11 kept scalars (index 9 = distance-to-nearest-stone);
# planes 11-22 are the 12 graded per-(cell, axis) window planes (4 quantities x
# 3 axes Q/R/QR, each quantity 3 contiguous slots so a D6 axis-permutation acts
# on 3-slot blocks); planes 23-24 are the 2 scalar fork planes. The binary
# hot/standing-win planes of the hexfield lineage are retired (see
# docs/PLAN_D6_EQUIVARIANT_REWRITE.md §3).
F_OWN_STONE = 0
F_OPP_STONE = 1
F_EMPTY = 2
F_LEGAL = 3
F_PHASE_SECOND = 4
F_FIRST_STONE = 5
F_PLAYER_COLOUR = 6
F_OWN_RECENCY = 7
F_OPP_RECENCY = 8
F_DIST_TO_STONE = 9
F_OPP_LAST_TURN = 10
# Graded per-axis window planes. Each quantity spans 3 contiguous slots ordered
# by axis [Q, R, QR], so `BASE + axis_index` selects the axis plane.
F_OWN_LINE_Q = 11
F_OWN_LINE_R = 12
F_OWN_LINE_QR = 13
F_OPP_LINE_Q = 14
F_OPP_LINE_R = 15
F_OPP_LINE_QR = 16
F_OWN_LIVE_Q = 17
F_OWN_LIVE_R = 18
F_OWN_LIVE_QR = 19
F_OPP_LIVE_Q = 20
F_OPP_LIVE_R = 21
F_OPP_LIVE_QR = 22
F_OWN_FORK = 23
F_OPP_FORK = 24
NUM_FEATURES = 25

# Length of the single-colour win/threat windows scanned for the graded planes.
WINDOW_LEN = 6

# Side-relative ray lengths (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L0):
# per cell u8[RAYLEN_SLOTS], flat index side*6 + axis*2 + dir with side in
# {own=0, opp=1}, axis in [Q, R, QR] order, dir in {+=0, -=1}. Values
# 0..RAY_REACH; the reach is the window-6 geometry made exact (a length-6
# window through x extends at most 5 cells along the axis), not a knob.
# Matches rust/src/constants.rs.
RAYLEN_SLOTS = 12
RAY_REACH = WINDOW_LEN - 1

# Graded-feature normalizers (match rust/src/constants.rs):
#   line count / 5 (a clean window holds at most 5 own stones in a decision
#   state — 6 is a played win), live window count / 6 (6 windows per cell per
#   axis), fork axis count / 3 (3 axes). A raw per-axis line count >=
#   FORK_LINE_THRESHOLD marks that axis as forking.
LINE_NORM = 5.0
LIVE_NORM = 6.0
FORK_NORM = 3.0
FORK_LINE_THRESHOLD = 3

# dist_to_stone feature scaling: stones -> 0, legal in (0, 1], halo -> 1.125
# (9/8, exactly representable in f16). Ply 0 => 0 everywhere.
DIST_SCALE = float(LEGAL_RADIUS)
HALO_DIST_FEATURE = HALO_DIST / DIST_SCALE  # 1.125

# --- heads / targets ------------------------------------------------------------
VALUE_BINS = 65
MOVES_LEFT_CAP = 209

# --- trunk ----------------------------------------------------------------------
# Trunk width from env HEXFIELD_EQ_CHANNELS (default 96), read once at import.
# Must be divisible by ATTENTION_HEADS. A checkpoint loads only into a net built
# at the same width.
#
# ENV NAMESPACE: this package deliberately reads HEXFIELD_EQ_* arch env names,
# NOT the live hexfield lineage's HEXFIELD_* names, so a process importing both
# trunks (a mixed eval / dashboard debug worker) cannot cross-configure them
# with one value ("BUGS_FOUND" env-collision item). The checkpoint meta is the
# authoritative arch self-description (see model.arch_meta); these envs only pick
# the default build.
CHANNELS = int(os.environ.get("HEXFIELD_EQ_CHANNELS", "96"))
# Summary tokens. The equivariant rewrite prunes the two dead tokens 6 & 7 (read
# by no head: value/aux/moves-left consume tokens 0..5), so NUM_TOKENS 8 -> 6
# (docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md §6, plan §1.4).
NUM_TOKENS = 6
# Head count from env HEXFIELD_EQ_ATTENTION_HEADS (default 3 — the equivariant
# build's coset split: heads = [D6:K] = 12/4 = 3 win-axes, head_dim = 4*C_ORBIT,
# and at c=192 head_dim lands on 64 where attention kernels are fast). d=32 at
# c=96 is still a valid fast-path head_dim.
ATTENTION_HEADS = int(os.environ.get("HEXFIELD_EQ_ATTENTION_HEADS", "3"))
if CHANNELS % ATTENTION_HEADS != 0:
    raise ValueError(
        f"HEXFIELD_EQ_CHANNELS={CHANNELS} not divisible by "
        f"HEXFIELD_EQ_ATTENTION_HEADS={ATTENTION_HEADS}"
    )
HEAD_DIM = CHANNELS // ATTENTION_HEADS  # 24 at c=96, 32 at c=128, 64 at c=192/h=3
MLP_RATIO = 2

# --- D6-equivariant trunk knobs (hexfield_eq) -----------------------------------
# The equivariant rewrite (docs/PLAN_D6_EQUIVARIANT_REWRITE.md) tiles the trunk
# width C into GROUP_ORDER fibers of C_ORBIT channels each, so
# C == GROUP_ORDER * C_ORBIT. Read once at import, matching the CHANNELS /
# ATTENTION_HEADS / TRUNK env convention above.
#
#   HEXFIELD_EQ_GROUP_ORDER — regular-representation order:
#     12 = full D6 (6 rotations x reflection; the Phase-3 target),
#      6 = C6 rotations only (reserved),
#      1 = non-equivariant passthrough (the Phase-6 A/B ablation AND the Phase-0
#          scaffold DEFAULT, so the copied dense trunk still builds unchanged).
#   HEXFIELD_EQ_C_ORBIT — per-fiber width. Unset -> derived (CHANNELS //
#     GROUP_ORDER for GROUP_ORDER>1, else CHANNELS).
#
# Phase 3b: the full D6 (order-12) regular-representation tie is the DEFAULT
# build. The tied trunk (equivariant.py + model.py) generates dense weights each
# forward from small orbit base params. GROUP_ORDER=1 stays a non-equivariant
# passthrough (the Phase-6 A/B ablation and the copied dense trunk).
GROUP_ORDER = int(os.environ.get("HEXFIELD_EQ_GROUP_ORDER", "12"))
if GROUP_ORDER not in (1, 6, 12):
    raise ValueError(
        f"HEXFIELD_EQ_GROUP_ORDER={GROUP_ORDER} unsupported; use 1 (passthrough), "
        "6 (C6, reserved), or 12 (full D6)"
    )

_C_ORBIT_ENV = os.environ.get("HEXFIELD_EQ_C_ORBIT")
if GROUP_ORDER > 1:
    # Equivariant build: enforce the C = GROUP_ORDER * C_ORBIT divisibility, plus
    # the kernel fast-path alignment constraints (C % 16 == 0 and
    # head_dim in {16,32,64,128}) so a tied trunk lands on the Triton fast path.
    if CHANNELS % GROUP_ORDER != 0:
        raise ValueError(
            f"HEXFIELD_EQ_CHANNELS={CHANNELS} not divisible by "
            f"HEXFIELD_EQ_GROUP_ORDER={GROUP_ORDER} (need C = GROUP_ORDER * C_ORBIT)"
        )
    C_ORBIT = int(_C_ORBIT_ENV) if _C_ORBIT_ENV is not None else CHANNELS // GROUP_ORDER
    if C_ORBIT < 1 or GROUP_ORDER * C_ORBIT != CHANNELS:
        raise ValueError(
            f"HEXFIELD_EQ_C_ORBIT={C_ORBIT} inconsistent: "
            f"GROUP_ORDER*C_ORBIT={GROUP_ORDER * C_ORBIT} != "
            f"HEXFIELD_EQ_CHANNELS={CHANNELS}"
        )
    if CHANNELS % 16 != 0:
        raise ValueError(
            f"HEXFIELD_EQ_CHANNELS={CHANNELS} must be a multiple of 16 for the "
            f"equivariant build (HEXFIELD_EQ_GROUP_ORDER={GROUP_ORDER})"
        )
    if HEAD_DIM not in (16, 32, 64, 96, 128):
        raise ValueError(
            f"head_dim={HEAD_DIM} (CHANNELS/ATTENTION_HEADS) must be one of "
            "{16,32,64,96,128} for the equivariant build (96 = the pre-authorized "
            "C=288 width contingency)"
        )
    if HEAD_DIM == 96:
        # Pre-authorized underfit contingency (C=288, C_ORBIT=24): permitted,
        # but the bespoke Triton attention kernel's fast path only engages at
        # head_dim in {16,32,64,128}; serve falls back to sdpa/flex. Loud so a
        # run at this width knows what it opted out of.
        import warnings

        warnings.warn(
            "HEXFIELD_EQ head_dim=96 (C=288 contingency): the bespoke Triton "
            "attention fast path will NOT engage; sdpa/flex serve paths only "
            "(docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md).",
            stacklevel=1,
        )
    # The equivariant multi-head split aligns heads to the 3 left cosets of the
    # order-4 subgroup K = stab(Q-axis): heads == [D6:K] == 3 (the 3 win-axes),
    # head_dim == |K|*C_ORBIT == 4*C_ORBIT (docs/DERIVATION §4). Only 12 (full D6)
    # is implemented; 6 (C6) is reserved.
    if GROUP_ORDER != 12:
        raise NotImplementedError(
            f"HEXFIELD_EQ_GROUP_ORDER={GROUP_ORDER} equivariant trunk not "
            "implemented; use 12 (full D6) or 1 (passthrough)"
        )
    # heads == 3 is STRUCTURAL under full D6 (GROUP_ORDER == 12), not a free
    # knob: the multi-head split IS the 3 left cosets of the order-4 stabilizer
    # K = stab(Q-axis), so heads == [D6:K] == 12/4 == 3 win-axes and
    # head_dim == |K|*C_ORBIT == 4*C_ORBIT (docs/DERIVATION §4). The derivation
    # forbids any other head count; reject it at import with a clear message.
    if ATTENTION_HEADS != 3:
        raise ValueError(
            f"HEXFIELD_EQ_ATTENTION_HEADS={ATTENTION_HEADS} must be 3 for the "
            "equivariant build (GROUP_ORDER=12): heads are structural — the "
            "3 left cosets of K=stab(Q-axis), i.e. the 3 win-axes ([D6:K]=3). "
            "Set HEXFIELD_EQ_ATTENTION_HEADS=3 or leave it unset (default 3)."
        )
    if HEAD_DIM != 4 * C_ORBIT:
        raise ValueError(
            f"head_dim={HEAD_DIM} must equal 4*C_ORBIT={4 * C_ORBIT} "
            "(|K|=4 coset split) for the equivariant build"
        )
else:
    # Passthrough (GROUP_ORDER == 1): the fiber IS the full width; no equivariant
    # alignment constraints apply. This is the Phase-0 scaffold default.
    C_ORBIT = int(_C_ORBIT_ENV) if _C_ORBIT_ENV is not None else CHANNELS
    if C_ORBIT != CHANNELS:
        raise ValueError(
            f"HEXFIELD_EQ_C_ORBIT={C_ORBIT} must equal "
            f"HEXFIELD_EQ_CHANNELS={CHANNELS} when HEXFIELD_EQ_GROUP_ORDER=1 "
            "(passthrough)"
        )

# Trunk block order from env HEXFIELD_EQ_TRUNK (default = the main_1..main_6
# layout). 'C' = ConvBlock (two hex convs), 'A' = attention block, 'L' = ray
# attention block (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L); the layout
# must end with 'A' (ln_final consumes the joint [tokens; cells] sequence the
# last attention block produced). main_7 uses "CCACCACCACCACCA" (CC A x5).
# A checkpoint loads only into a net built with the same layout.
TRUNK_LAYOUT = os.environ.get("HEXFIELD_EQ_TRUNK", "CCCACCCACCA")
if (
    not TRUNK_LAYOUT
    or set(TRUNK_LAYOUT) - {"C", "A", "L"}
    or not TRUNK_LAYOUT.endswith("A")
):
    raise ValueError(
        f"HEXFIELD_EQ_TRUNK={TRUNK_LAYOUT!r} must be a non-empty string of "
        "'C'/'A'/'L' ending with 'A'"
    )

# --- ray attention (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L) -----------
# L blocks run 6 heads = 3 win-axis cosets x {own, opp} orbit-halves (plan L4).
# The count is STRUCTURAL under the equivariant build (the own/opp split rides
# the orbit index, NEVER the K-slots — a slot split silently breaks
# equivariance), not an env knob; the ATTENTION_HEADS == 3 import check above
# governs A blocks only (per-block-type head counts). head_dim_L =
# CHANNELS / RAY_HEADS (= 2*C_ORBIT in the equivariant build).
RAY_HEADS = 6
if "L" in TRUNK_LAYOUT:
    if CHANNELS % RAY_HEADS != 0:
        raise ValueError(
            f"HEXFIELD_EQ_CHANNELS={CHANNELS} not divisible by RAY_HEADS="
            f"{RAY_HEADS} (required for an 'L' trunk layout)"
        )
    if GROUP_ORDER == 12 and C_ORBIT % 2 != 0:
        raise ValueError(
            f"HEXFIELD_EQ_C_ORBIT={C_ORBIT} must be even for an 'L' layout: the "
            "own/opp sub-head split is along the orbit index (head_dim_L = "
            "2*C_ORBIT; plan L4)"
        )
# HEXFIELD_EQ_RAY_BLOCKERS ("0"/"1", default "1"): 1 = game-live rays (walk
# truncated at anti-side stones, read from the raylen wire data); 0 = geometric
# rays (pure axis-disk-RAY_REACH attention, computable from coords alone — the
# plan L6 attribution control). A mask-build variant, not a state-dict change.
_RAY_BLOCKERS_ENV = os.environ.get("HEXFIELD_EQ_RAY_BLOCKERS", "1")
if _RAY_BLOCKERS_ENV not in ("0", "1"):
    raise ValueError(
        f"HEXFIELD_EQ_RAY_BLOCKERS={_RAY_BLOCKERS_ENV!r} must be '0' or '1'"
    )
RAY_BLOCKERS = _RAY_BLOCKERS_ENV == "1"

# --- register lane (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase R) -----------
# HEXFIELD_EQ_REG_LANE ("0"/"1", default "0"): attach a RegisterRefresh (one-way
# sigmoid-gated SUM cross-attention, tokens <- cells) at the exit of every C
# block. HEXFIELD_EQ_REG_TOK_READ ("0"/"1", default "0"): the cells <- tokens
# broadcast read at C-block entry; only meaningful with the lane on. Both are
# arch knobs (they change the state-dict key set), so they take the
# HEXFIELD_EQ_* namespace and ride the checkpoint meta (model.arch_meta).
_REG_LANE_ENV = os.environ.get("HEXFIELD_EQ_REG_LANE", "0")
if _REG_LANE_ENV not in ("0", "1"):
    raise ValueError(f"HEXFIELD_EQ_REG_LANE={_REG_LANE_ENV!r} must be '0' or '1'")
REG_LANE = _REG_LANE_ENV == "1"
_REG_TOK_READ_ENV = os.environ.get("HEXFIELD_EQ_REG_TOK_READ", "0")
if _REG_TOK_READ_ENV not in ("0", "1"):
    raise ValueError(
        f"HEXFIELD_EQ_REG_TOK_READ={_REG_TOK_READ_ENV!r} must be '0' or '1'"
    )
REG_TOK_READ = _REG_TOK_READ_ENV == "1"
if REG_TOK_READ and not REG_LANE:
    raise ValueError(
        "HEXFIELD_EQ_REG_TOK_READ=1 requires HEXFIELD_EQ_REG_LANE=1 (the read is "
        "an arm of the register lane, not a standalone mechanism)"
    )
# Fixed scale on the sigmoid-gated SUM aggregation (plan R1): matched-set sizes
# are tens of cells, so scaled updates land O(1)-O(10). A constant, not an env
# knob.
REG_SUM_SCALE = 1.0 / 32.0

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
