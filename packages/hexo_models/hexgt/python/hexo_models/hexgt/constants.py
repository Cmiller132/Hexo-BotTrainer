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

# --- Node feature layout (gap A) — PROVISIONAL, finalized in Phase 4 expand ---
# Unified per-node feature vector of fixed width NODE_FEATURE_DIM. Slots are
# type-routed: a node only fills the slots relevant to its type, the rest are 0.
# The contract only requires the Rust/Python graph builder and the model's input
# projection to agree on this layout. Coordinates are center-relative axial,
# scaled by COORD_SCALE so the unbounded board never saturates a fixed range.
NODE_FEATURE_DIM = 32
COORD_SCALE = 16.0            # divide relative axial coords by this before use

# Named feature offsets (half-open ranges where noted).
F_TYPE_ONEHOT = 0            # [0:4)  node-type one-hot
F_OWNER_OWN = 4             # 1 if node belongs to side-to-move (stone/window)
F_OWNER_OPP = 5             # 1 if node belongs to the opponent
F_REL_Q = 6                # center-relative axial q / COORD_SCALE
F_REL_R = 7                # center-relative axial r / COORD_SCALE
F_REL_S = 8                # center-relative axial s (= -q-r) / COORD_SCALE
F_STONE_RECENCY = 9        # stone hist_idx normalized to [0,1]
F_WIN_COUNT_ONEHOT = 10    # [10:13) window count one-hot (3,4,5)
F_WIN_AXIS_ONEHOT = 13     # [13:16) window axis one-hot (Q,R,QR)
F_WIN_EMPTY_CELLS = 16     # window empty-cell count / WINDOW_LEN
F_CAND_NWIN_OWN = 17       # candidate: # active own windows through this cell (norm)
F_CAND_NWIN_OPP = 18       # candidate: # active opp windows through this cell (norm)
F_CAND_COMPLETE_OWN = 19   # candidate completes a count-5 own window (winning move)
F_CAND_COMPLETE_OPP = 20   # candidate completes a count-5 opp window (must-block)
F_SIDE_TO_MOVE = 21        # side node: side-to-move indicator
F_SIDE_PHASE_ONEHOT = 22   # [22:25) side node: turn-phase one-hot (Opening/First/Second)
F_SIDE_STONES_OWN = 25     # side node: own stone count (norm)
F_SIDE_STONES_OPP = 26     # side node: opp stone count (norm)
F_SIDE_MOVE_NUMBER = 27    # side node: move number (norm)
# [28:32) reserved for Phase 4 finalization.

# --- Default GNN/transformer hyperparameters (gap H) — sized in Phase 3 -------
# Provisional defaults aiming at the ~2.1M-param budget (verified in Phase 3).
DEFAULT_TOKEN_DIM = 128
DEFAULT_GNN_LAYERS = 3
DEFAULT_CTX_LAYERS = 3
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_FFN_DIM = 256
