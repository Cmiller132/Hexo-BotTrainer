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
