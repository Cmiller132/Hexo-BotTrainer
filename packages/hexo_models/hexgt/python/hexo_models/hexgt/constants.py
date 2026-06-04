"""hexgt (Model 2) tensor + graph constants.

These values define the Python side of the dynamic-graph model contract. The
Rust candidate/window/graph builder (`hexgt/rust/src/*.rs`, added in Phase 1)
carries matching values; any change to node/edge type ids, the value-bin count,
or the node-feature layout must update both halves and the boundary tests.

The model is a *truly dynamic* typed heterogeneous graph: there is no board
size, no `BOARD_AREA`, no fixed candidate count. Counts vary per position. See
`HEXFORMER_REWRITE_PLAN.md` §3.4/§6 and `HEXGT_DECISIONS.md`.
"""

from __future__ import annotations

# --- Value head (reused verbatim from dense_cnn, pipeline-convenient) ---------
VALUE_BINS = 65

# --- Node types ---------------------------------------------------------------
# A node carries exactly one type. SIDE is the per-position context/goal hub.
NODE_TYPE_SIDE = 0
NODE_TYPE_STONE = 1
NODE_TYPE_CANDIDATE = 2
NODE_TYPE_WINDOW = 3
NUM_NODE_TYPES = 4

# --- Edge types (bounded construction, §6.3 — NO same-axis cliques) -----------
# Routed through window nodes as the line/co-linearity hub. Stored symmetric for
# the MVP (each structural edge emitted in both directions with the same type).
EDGE_TYPE_ADJACENCY = 0        # node <-> node within hex-distance 1 (<=6/node)
EDGE_TYPE_STONE_WINDOW = 1     # window <-> its one-color stones (<=6/window)
EDGE_TYPE_CANDIDATE_WINDOW = 2 # window <-> its empty cells (<=6/window)
EDGE_TYPE_RECENCY = 3          # stone <-> immediately-preceding/following stone
EDGE_TYPE_CONTEXT = 4          # side/goal hub <-> all nodes (1 hub)
NUM_EDGE_TYPES = 5

# --- Tactical-window token taxonomy (§5): count-3/4/5 active windows ----------
# owner in {current, opponent}, count in {3, 4, 5}. Used by feature encoding and
# (later) the active-window token nodes.
WINDOW_COUNT_MIN_TOKEN = 3
WINDOW_COUNT_MAX_TOKEN = 5
WINDOW_LEN = 6                 # engine win length (six-in-a-line)

# --- Candidate-set rule (§4): active windows UNION n-radius --------------------
# `n` is the single tunable neighborhood radius (same hex-distance metric as the
# engine's LEGAL_RADIUS = 8). Practical range ~[2, 4].
#
# DECISION (user, informed by the Phase-1 coverage sweep): default n = 3. n=2
# covers ~91% of dense_cnn's strong moves by visit mass, n=3 ~92%, n=4 ~92%; the
# remaining ~8% are FAR "spread" placements (68% at hex-distance 6-8) that only
# n=8 (≈ the full legal set) reaches. Those far moves are dense_cnn's LEARNED
# novel-far-placement strategy and are NOT needed for high-level play, so we do
# NOT widen the radius to fit them. Recorded positions whose played move / policy
# mass falls outside the n=3 candidate set are handled by DATASET PRUNING in the
# BC/sample-gen path (see expand.py + HexgtSampleConfig.bc_prune_max_dropped_mass),
# NOT by enlarging the candidate set. The "up to 8" framing is dropped.
DEFAULT_CANDIDATE_RADIUS = 3
MIN_CANDIDATE_RADIUS = 2
MAX_CANDIDATE_RADIUS = 4
ENGINE_LEGAL_RADIUS = 8

# --- Turn phase ids (engine placement phase within a turn) --------------------
# A turn places up to two stones; the phase indexes where we are in it. Used by
# the side-node phase one-hot (F_SIDE_PHASE_ONEHOT, width NUM_TURN_PHASES) and by
# the phase-aware v2/v3 candidate features in features.py.
PHASE_OPENING = 0
PHASE_FIRST_STONE = 1
PHASE_SECOND_STONE = 2
NUM_TURN_PHASES = 3

# --- Node feature layout (gap A) — D6-INVARIANT (see d6.py / features.py) ------
# Unified per-node feature vector of fixed width NODE_FEATURE_DIM. Slots are
# type-routed: a node only fills the slots relevant to its type, the rest are 0.
#
# ALL features are D6-INVARIANT: there are NO raw axial coords and NO window axis
# labels (both rotate under D6). Geometry is carried by the graph STRUCTURE
# (adjacency / window-membership edges) + the invariant per-edge hex-distance
# (see EDGE_ATTR_DIM). This makes the model D6-invariant by construction, so the
# equivariance test passes exactly for all 12 elements with no augmentation.
# `center_distance` (max-norm from the opening (0,0)) is D6-invariant (D6 fixes
# the origin) and is the one coord-derived scalar retained.
NODE_FEATURE_DIM = 32
COORD_SCALE = 16.0            # normalization scale for distances / counts
COUNT_SCALE = 32.0           # normalization scale for stone/move counts

F_TYPE_ONEHOT = 0            # [0:4)  node-type one-hot (side, stone, candidate, window)
F_OWNER_OWN = 4             # 1 if node belongs to side-to-move (stone/window)
F_OWNER_OPP = 5             # 1 if node belongs to the opponent
F_CENTER_DISTANCE = 6      # max-norm(coord) / COORD_SCALE (D6-invariant)
F_STONE_RECENCY = 7        # stone hist_idx normalized to [0,1]
F_WIN_COUNT_ONEHOT = 8     # [8:11) window count one-hot (3,4,5)
F_WIN_EMPTY_CELLS = 11     # window empty-cell count / WINDOW_LEN
F_CAND_COMPLETE_OWN = 12   # candidate completes a count-5 own window (winning move)
F_CAND_COMPLETE_OPP = 13   # candidate completes a count-5 opp window (must-block)
F_CAND_NWIN_OWN = 14       # candidate: # active own windows through this cell (norm)
F_CAND_NWIN_OPP = 15       # candidate: # active opp windows through this cell (norm)
F_SIDE_PHASE_ONEHOT = 16   # [16:19) side node: turn-phase one-hot (Opening/First/Second)
F_SIDE_STONES_OWN = 19     # side node: own stone count (norm)
F_SIDE_STONES_OPP = 20     # side node: opp stone count (norm)
F_SIDE_MOVE_NUMBER = 21    # side node: move number (norm)

# --- Tactical feature-set v2 (added into the reserved range; D6-INVARIANT) -----
# Activated by FEATURE_SCHEMA_VERSION 2. These occupy slots [22:30) of the SAME
# fixed NODE_FEATURE_DIM=32 vector, so the unified `node_in` projection keeps
# shape (token_dim, 32): a checkpoint trained before v2 LOADS WITHOUT SHAPE
# REJECTION, and the ZERO-INIT LAYER-EXPANSION (architecture.zero_init_expanded_
# feature_columns) zeroes exactly these input-weight COLUMNS once at the
# introducing resume so the model output is byte-identical to that checkpoint,
# then learns them as training continues. Splitting candidate window counts by
# owner+count generalizes F_CAND_NWIN_{OWN,OPP} (their per-count breakdown; the
# three own bins sum to the own nwin count, likewise opp — gated by tests).
#
# D6-invariance: window counts through a cell are preserved by every D6 element
# (windows/owners/counts map bijectively, the cell maps to its image); hex
# distance is a D6 invariant; the placement flag is phase-derived (D6 fixes
# phase). So all eight are invariant and the equivariance test holds unchanged.
F_CAND_OWN_WIN3 = 22       # candidate: # active OWN count-3 windows through cell (norm)
F_CAND_OWN_WIN4 = 23       # candidate: # active OWN count-4 windows through cell (norm)
F_CAND_OWN_WIN5 = 24       # candidate: # active OWN count-5 windows through cell (norm)
F_CAND_OPP_WIN3 = 25       # candidate: # active OPP count-3 windows through cell (norm)
F_CAND_OPP_WIN4 = 26       # candidate: # active OPP count-4 windows through cell (norm)
F_CAND_OPP_WIN5 = 27       # candidate: # active OPP count-5 windows through cell (norm)
F_CAND_DIST_FIRST = 28     # candidate: hex-dist to THIS turn's first stone (0 on the
                           #   turn's first placement) / COORD_SCALE (D6-invariant)
F_SIDE_IS_SECOND = 29      # side node: 1.0 if the current placement is the turn's
                           #   SECOND stone (phase == SecondStone), else 0.0

# --- Tactical feature-set v3 (slots [30:32); D6-INVARIANT, PHASE-AWARE) --------
# Activated by FEATURE_SCHEMA_VERSION 3. These are PHASE-AWARE win-now flags,
# distinct from the v2 count-{3,4,5} window-count splits (which only COUNT windows)
# and from F_CAND_COMPLETE_{OWN,OPP} (which fire only at count-5). They give the
# net the count-4-with-correct-phase tactical signal the count-5-only completion
# flags miss (HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md PART 1 (c)). Same zero-init
# layer-expansion path as v2 (architecture.zero_init_expanded_feature_columns).
#
# D6-invariance: window owner/count/membership through a cell maps bijectively
# under every D6 element; the placement budget B is phase-derived (D6 fixes phase).
# Both flags are invariant, so the equivariance test holds unchanged.
F_CAND_WIN_NOW_OWN = 30    # candidate: placing here is (part of) an OWN win THIS turn
                           #   -- own count-5 (any B), or own count-4 only at FirstStone
                           #   (B==2). A count-4 at SecondStone is NOT a win (TEST G).
F_CAND_OPP_THREAT = 31     # candidate: cell is an empty of an active OPPONENT >=4 window
                           #   (count-4 OR count-5) -- the must-answer block-cell set (the
                           #   opponent's own turn always has B==2, so count-4 is a genuine
                           #   immediate threat). Count-4-inclusive analog of COMPLETE_OPP.

# Feature schema version. Bumped whenever a NEW always-zero-until-now slot is
# activated, so the RL resume path knows to zero-init its node_in columns ONCE
# (checkpoints predating the bump report a lower version / none -> treated as 1).
FEATURE_SCHEMA_VERSION = 3
# The slots activated in v2 — the zero-init layer-expansion target (node_in input
# columns to zero at the introducing resume). Order is irrelevant (a column set).
NEW_FEATURE_SLOTS_V2 = (
    F_CAND_OWN_WIN3, F_CAND_OWN_WIN4, F_CAND_OWN_WIN5,
    F_CAND_OPP_WIN3, F_CAND_OPP_WIN4, F_CAND_OPP_WIN5,
    F_CAND_DIST_FIRST, F_SIDE_IS_SECOND,
)
# The slots activated in v3 (the phase-aware win-now flags).
NEW_FEATURE_SLOTS_V3 = (F_CAND_WIN_NOW_OWN, F_CAND_OPP_THREAT)


def feature_slots_after(version: int) -> tuple[int, ...]:
    """Node-feature columns introduced in schema versions strictly greater than
    `version` — the zero-init layer-expansion target when resuming from a
    checkpoint stamped `version`. A checkpoint at v2 needs only the v3 slots
    zeroed (its v2 columns are trained); a v1/none checkpoint needs v2 + v3.
    """

    slots: list[int] = []
    if version < 2:
        slots.extend(NEW_FEATURE_SLOTS_V2)
    if version < 3:
        slots.extend(NEW_FEATURE_SLOTS_V3)
    return tuple(slots)

# --- Edge attributes (D6-invariant) -------------------------------------------
# Per-edge feature vector: edge-type one-hot ++ endpoint hex-distance (invariant).
EDGE_ATTR_DIM = NUM_EDGE_TYPES + 1  # NUM_EDGE_TYPES defined above
EDGE_ATTR_DIST = NUM_EDGE_TYPES     # offset of the hex-distance scalar

# --- Default GNN/transformer hyperparameters (gap H) — sized to ~2.1M params ---
# Verified by sum(p.numel()) in Phase 3 to land within ~10% of the running 96x8
# baseline (~2.1M). token_dim 168 / ffn 336, 3 GNN + 3 transformer layers.
DEFAULT_TOKEN_DIM = 168
DEFAULT_GNN_LAYERS = 3
DEFAULT_CTX_LAYERS = 3
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_FFN_DIM = 336
# PMA value-head seed count k (HEXGT_PMA_VALUE_HEAD_PLAN.md): the value readout is
# [SIDE | PMA_k]. Owner's chosen build uses k=2 (the doc's analytic default is 1).
DEFAULT_VALUE_PMA_SEEDS = 2
