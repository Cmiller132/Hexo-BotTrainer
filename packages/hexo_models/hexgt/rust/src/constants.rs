//! hexgt MCTS native constants.
//!
//! Unlike dense_cnn, hexgt has no plane/crop tensor dimensions in Rust — the
//! model input is the typed graph built by `candidates.rs` and featurized in
//! Python. These constants only bound the search batching + eval cache, mirroring
//! dense_cnn's `MODEL1_EVAL_*` so the two families share self-play behavior.

/// Default candidate-set neighborhood radius `n` (mirror of
/// `python/.../constants.py::DEFAULT_CANDIDATE_RADIUS`). Move vocabulary = the
/// n-radius candidate set; this is the only knob threaded into both sample-gen
/// and live search.
pub(crate) const HEXGT_DEFAULT_CANDIDATE_RADIUS: i16 = 3;

/// Max unique leaf states evaluated in one Python callback chunk (Torch memory +
/// callback latency are better behaved when large batches are chunked).
pub(crate) const HEXGT_EVAL_CHUNK_STATES: usize = 1024;

/// Default transposition/eval cache bound (FIFO eviction beyond this).
pub(crate) const HEXGT_EVAL_CACHE_MAX_STATES: usize = 1_048_576;

/// Strict upper bound on the number of active roots per `search` call.
pub(crate) const HEXGT_ACTIVE_ROOT_LIMIT: usize = 1024;

// --- Node/edge feature layout (MUST mirror python/.../constants.py + features.py).
// Any change here is a model-weights + replay-schema change; the parity test
// (test_hexgt_featurize_parity) guards Rust<->Python equality.
pub(crate) const NODE_FEATURE_DIM: usize = 32;
pub(crate) const NUM_NODE_TYPES: i64 = 4;
// i64 sibling of `candidates::NUM_EDGE_TYPES` (usize), for edge-type clamp
// arithmetic in the featurizer. Both mirror python `NUM_EDGE_TYPES`.
pub(crate) const NUM_EDGE_TYPES: i64 = 5;
pub(crate) const EDGE_ATTR_DIM: usize = 6; // NUM_EDGE_TYPES + 1
pub(crate) const EDGE_ATTR_DIST: usize = 5; // offset of the hex-distance scalar
pub(crate) const COORD_SCALE: f32 = 16.0;
pub(crate) const COUNT_SCALE: f32 = 32.0;
pub(crate) const WINDOW_LEN_F: f32 = 6.0;

// node type ids (mirror candidates.rs NODE_* / python NODE_TYPE_*)
pub(crate) const FT_NODE_SIDE: i64 = 0;
pub(crate) const FT_NODE_STONE: i64 = 1;
pub(crate) const FT_NODE_CANDIDATE: i64 = 2;
pub(crate) const FT_NODE_WINDOW: i64 = 3;
pub(crate) const FT_EDGE_CANDIDATE_WINDOW: i64 = 2;

// feature slot offsets
pub(crate) const F_TYPE_ONEHOT: usize = 0; // [0:4)
pub(crate) const F_OWNER_OWN: usize = 4;
pub(crate) const F_OWNER_OPP: usize = 5;
pub(crate) const F_CENTER_DISTANCE: usize = 6;
pub(crate) const F_STONE_RECENCY: usize = 7;
pub(crate) const F_WIN_COUNT_ONEHOT: usize = 8; // [8:11)
pub(crate) const F_WIN_EMPTY_CELLS: usize = 11;
pub(crate) const F_CAND_COMPLETE_OWN: usize = 12;
pub(crate) const F_CAND_COMPLETE_OPP: usize = 13;
pub(crate) const F_CAND_NWIN_OWN: usize = 14;
pub(crate) const F_CAND_NWIN_OPP: usize = 15;
pub(crate) const F_SIDE_PHASE_ONEHOT: usize = 16; // [16:19)
pub(crate) const F_SIDE_STONES_OWN: usize = 19;
pub(crate) const F_SIDE_STONES_OPP: usize = 20;
pub(crate) const F_SIDE_MOVE_NUMBER: usize = 21;
// Tactical feature-set v2 ([22:30) of the same 32-wide vector; D6-invariant).
// MUST mirror python/.../constants.py (gated by test_hexgt_feature_buffer).
pub(crate) const F_CAND_OWN_WIN3: usize = 22; // own count-3 windows through cell (norm)
pub(crate) const F_CAND_OWN_WIN4: usize = 23; // own count-4
pub(crate) const F_CAND_OWN_WIN5: usize = 24; // own count-5
pub(crate) const F_CAND_OPP_WIN3: usize = 25; // opp count-3
pub(crate) const F_CAND_OPP_WIN4: usize = 26; // opp count-4
pub(crate) const F_CAND_OPP_WIN5: usize = 27; // opp count-5
pub(crate) const F_CAND_DIST_FIRST: usize = 28; // hex-dist to this turn's first stone (norm)
pub(crate) const F_SIDE_IS_SECOND: usize = 29; // side: 1 if current placement is 2nd stone
// Tactical feature-set v3 ([30:32) of the same 32-wide vector; D6-invariant,
// PHASE-AWARE win-now flags). Distinct from the v2 count-{3,4,5} window-count
// SPLITS (F_CAND_*_WIN{3,4,5}, which only count windows) and from F_CAND_COMPLETE_*
// (which fire only at count-5). MUST mirror python/.../constants.py (gated by
// test_hexgt_feature_buffer).
pub(crate) const F_CAND_WIN_NOW_OWN: usize = 30; // placing here is (part of) an OWN win
                                                 // THIS turn: own count-5 (any B), or own
                                                 // count-4 only at FirstStone (B==2). NOT a
                                                 // win for a count-4 at SecondStone (TEST G).
pub(crate) const F_CAND_OPP_THREAT: usize = 31;  // cell is an empty of an active OPPONENT
                                                 // >=4 window (count-4 OR count-5): the
                                                 // must-answer block-cell set (the opponent's
                                                 // own turn always has B==2, so count-4 is a
                                                 // genuine immediate threat). Count-4-inclusive
                                                 // analog of F_CAND_COMPLETE_OPP (count-5 only).
