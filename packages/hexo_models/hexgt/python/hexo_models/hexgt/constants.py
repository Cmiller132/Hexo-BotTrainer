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
# `n` is the single tunable neighborhood radius, range [2, 8] (same hex-distance
# metric as the engine's LEGAL_RADIUS = 8).
#
# Phase-1 finding (HEXGT_DECISIONS.md): the plan's n=2 default covers only
# ~85-92% of dense_cnn's strong played/visited moves; the misses are FAR spread
# plays (68% at hex-distance 6-8), so ~100% coverage requires n=8 (= the full
# engine legal set). To avoid handicapping the move vocabulary vs dense_cnn, the
# MVP default is n=8 (candidate_set ≡ legal set, 100% coverage). The graph's
# value-add over dense_cnn is its tactical STRUCTURE (window-hub tokens, typed
# edges), validated by the no-explosion gate, not candidate pruning. `n` stays
# the single knob to sweep the coverage/throughput frontier down in Phase 5.
DEFAULT_CANDIDATE_RADIUS = 8
MIN_CANDIDATE_RADIUS = 2
MAX_CANDIDATE_RADIUS = 8
ENGINE_LEGAL_RADIUS = 8

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
# [22:32) reserved.

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
