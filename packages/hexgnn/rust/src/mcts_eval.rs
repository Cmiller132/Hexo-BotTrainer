//! Python/PyTorch evaluator adapter for hexgnn MCTS.
//!
//! The search tree (copied from dense_cnn) asks for batches of leaf states; this
//! module is the boundary to the torch GNN evaluator:
//!
//! 1. Hash incoming engine states with `hexo_utils` (transposition + duplicate
//!    coalescing), reusing cached evaluations.
//! 2. Build the hexgnn typed-graph facts for each unique leaf **in Rust** from the
//!    leaf `HexoState` (no Py->Rust reclone) via the shared `candidates` builder,
//!    so search inputs == training inputs.
//! 3. Call the Python evaluator (`HexgnnInference.evaluate_graph_facts`).
//! 4. Parse the per-candidate byte contract (values + CSR candidate ids + priors)
//!    back into `RustEvaluation`, intersecting with engine legality and
//!    DESCENDING-sorting + normalizing the priors (the contract the copied tree
//!    relies on — interior nodes materialize edges strictly highest-prior-first).
//!
//! Unlike dense_cnn there are no planes/crop here; the move vocabulary is the
//! n-radius candidate set. `n` is threaded from the session so training support
//! == search expansion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hexo_engine::{pack_coord, HexoState as RustHexoState, PackedCoord};
use hexo_utils::{hash_state, StateHash};

use super::constants::HEXGT_EVAL_CHUNK_STATES;
use super::features::{collated_to_py_dict, featurize_collate_states, CollatedFeatures};

#[derive(Clone, Debug)]
pub(crate) struct RustEvaluation {
    /// Scalar value from the perspective of the evaluated state's current player.
    pub(crate) value: f32,
    /// Number of in-set legal candidate moves (== `priors.len()`; the candidate
    /// set IS the move vocabulary, so no out-of-set tail like dense_cnn's crop).
    pub(crate) legal_action_count: usize,
    /// One prior per legal candidate, ranked DESCENDING + normalized to sum 1.0.
    pub(crate) priors: Vec<(PackedCoord, f32)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EvaluationStats {
    pub(crate) requested_states: usize,
    pub(crate) cache_hits: usize,
    pub(crate) duplicate_hits: usize,
    pub(crate) unique_states: usize,
    pub(crate) evaluator_chunks: usize,
    pub(crate) encoded_states: usize,
    pub(crate) encoded_legal_actions: usize,
    pub(crate) max_chunk_states: usize,
    pub(crate) max_chunk_legal_actions: usize,
    pub(crate) input_bytes: usize,
    pub(crate) legal_index_bytes: usize,
    pub(crate) value_bytes: usize,
    pub(crate) prior_bytes: usize,
    pub(crate) cache_inserts: usize,
    pub(crate) cache_insert_skipped: usize,
    pub(crate) cache_size_peak: usize,
    pub(crate) encoding_seconds: f64,
    pub(crate) evaluator_seconds: f64,
    pub(crate) parse_seconds: f64,
    pub(crate) payload_seconds: f64,
    pub(crate) dedup_seconds: f64,
    pub(crate) cache_insert_seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RustEvaluationCache {
    entries: HashMap<StateHash, Arc<RustEvaluation>>,
    insertion_order: VecDeque<StateHash>,
}

impl RustEvaluationCache {
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, key: &StateHash) -> Option<Arc<RustEvaluation>> {
        self.entries.get(key).map(Arc::clone)
    }

    fn insert_bounded(&mut self, key: StateHash, evaluation: Arc<RustEvaluation>, max_states: usize) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key, evaluation);
            return;
        }
        debug_assert!(max_states > 0);
        while self.entries.len() >= max_states {
            let Some(evicted) = self.insertion_order.pop_front() else {
                break;
            };
            if self.entries.remove(&evicted).is_some() {
                break;
            }
        }
        self.insertion_order.push_back(key);
        self.entries.insert(key, evaluation);
    }
}

pub(crate) type SharedEvaluationCache = Arc<Mutex<RustEvaluationCache>>;
pub(crate) type SharedEvaluationStats = Arc<Mutex<EvaluationStats>>;

pub(crate) fn new_shared_evaluation_cache() -> SharedEvaluationCache {
    Arc::new(Mutex::new(RustEvaluationCache::default()))
}

pub(crate) fn new_shared_evaluation_stats() -> SharedEvaluationStats {
    Arc::new(Mutex::new(EvaluationStats::default()))
}

#[inline]
fn lock_cache(cache: &SharedEvaluationCache) -> std::sync::MutexGuard<'_, RustEvaluationCache> {
    cache.lock().expect("evaluation cache mutex poisoned")
}

#[inline]
fn lock_stats(stats: &SharedEvaluationStats) -> std::sync::MutexGuard<'_, EvaluationStats> {
    stats.lock().expect("evaluation stats mutex poisoned")
}

pub(crate) fn state_hash(state: &RustHexoState) -> StateHash {
    hash_state(state)
}

pub(crate) struct RustEvaluationRequest<'a> {
    pub(crate) state: &'a RustHexoState,
    pub(crate) state_hash: StateHash,
}

/// Depth of the INTRA-BATCH featurize<->forward pipeline (`evaluate_state_refs_-
/// uncached`): how many leaf chunks the GIL-free rayon featurizer may run ahead of
/// the GPU forward within ONE eval. The producer (featurizer) is GIL-free; the
/// consumer (torch forward) holds the GIL and is the blocking, GPU-bound stage. The
/// sync_channel capacity == this depth, so it doubles as the backpressure bound
/// that caps queued-feature memory to `depth` chunks.
///
/// depth=2 is plain double-buffering (featurize chunk N+1 while the GPU forwards
/// chunk N). We default to 3 so the featurizer can run ~2 chunks ahead, keeping the
/// forward pipe fed across chunk boundaries / start-of-batch ramp. Memory stays
/// bounded by the channel capacity. Tunable via `HEXGT_EVAL_PIPELINE_DEPTH` (>=1).
///
/// NOTE: the hot self-play path (`run_searches_to_targets`) does NOT route through
/// this intra-batch channel — at active=512/visits=512 each eval's unique-leaf
/// count is a single chunk, so the channel never engaged and the GPU sat 50-66%
/// idle. That path instead uses the CROSS-batch `prepare_eval_refs`/
/// `finish_eval_prepared` split, which featurizes the NEXT batch on a worker thread
/// while the GPU forwards the CURRENT batch (fixed depth-2 cross-batch overlap).
/// This knob still governs the per-search ROOT eval (`evaluate_states_cached`),
/// which can legitimately span multiple chunks.
fn eval_pipeline_depth() -> usize {
    std::env::var("HEXGT_EVAL_PIPELINE_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&d| d >= 1)
        .unwrap_or(3)
}

/// Max unique leaves featurized + forwarded per pipeline chunk. Defaults to
/// `HEXGT_EVAL_CHUNK_STATES`; overridable via the same-named env var for tuning
/// and for tests that exercise the multi-chunk pipeline with small batches.
fn eval_chunk_states() -> usize {
    std::env::var("HEXGT_EVAL_CHUNK_STATES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| c >= 1)
        .unwrap_or(HEXGT_EVAL_CHUNK_STATES)
}

/// Eval concurrency model. Evaluating a batch of unique leaves is two stages:
///
///   PRODUCER (GIL-free): dedup vs the shared cache, then featurize the misses
///     (`build_graph` + node featurize + collate) — pure Rust, rayon-parallel,
///     touches no Python and no MCTS tree.
///   CONSUMER (GIL-bound): build the zero-copy payload dict, run the torch forward
///     (GPU-bound, holds the GIL), parse per-candidate priors, insert into cache.
///
/// The two halves are split across `prepare_eval_refs` / `finish_eval_prepared`
/// (carried by `PreparedEval`, which owns its data — no borrow of the leaf states)
/// so the caller (`run_searches_to_targets`) can run the PRODUCER for the NEXT
/// batch on a worker thread WHILE this GIL thread runs the CONSUMER (forward) for
/// the CURRENT batch. The worker NEVER touches Python — only the consumer calls
/// the evaluator — so the GIL stays on one thread and rayon featurization runs
/// alongside it, instead of the per-chunk GIL hand-off that previously collapsed
/// throughput (HEXGT_DECISIONS Phase-5d). This is the cross-batch realization of
/// the intra-batch featurize<->forward pipeline.
///
/// One featurized chunk awaiting the GPU forward: the collated graph buffers plus
/// the precomputed per-state engine legality sets (one per state, used by the
/// consumer's prior finalize). Produced GIL-free by the rayon featurizer and
/// handed to `consume_featurized_chunk` later. Every field is OWNED (no borrow of
/// the source `HexoState`s), so a `FeaturizedChunk` — and the `PreparedEval` that
/// holds it — is free to outlive the featurize call and cross an iteration/scope
/// boundary. That is what makes cross-batch featurize<->forward overlap sound: the
/// prepared next batch no longer borrows the leaves it was built from.
pub(crate) struct FeaturizedChunk {
    collated: CollatedFeatures,
    legal_sets: Vec<HashSet<PackedCoord>>,
}

/// GIL-FREE producer: featurize every unique leaf state into one `FeaturizedChunk`
/// per `eval_chunk_states()` slice. This is the pure-Rust, rayon-parallel half of
/// the eval — it touches NO Python and NO MCTS tree, so it can run on a worker
/// thread while the GIL thread is busy running a different batch's torch forward
/// (cross-batch featurize<->forward overlap; see `run_searches_to_targets`).
fn featurize_unique_states(
    states: &[&RustHexoState],
    n: i16,
    stats: Option<&SharedEvaluationStats>,
) -> Vec<FeaturizedChunk> {
    let chunk_size = eval_chunk_states();
    states
        .chunks(chunk_size)
        .map(|chunk| FeaturizedChunk {
            collated: featurize_leaf_chunk(chunk, n, stats),
            // Legality is pure-Rust/GIL-free, so fold it into the producer here
            // rather than reaching for the live state in the GIL-bound consumer.
            legal_sets: chunk.iter().copied().map(legal_move_set).collect(),
        })
        .collect()
}

/// GIL-BOUND consumer: run the torch forward + parse for each pre-featurized
/// chunk, in order. The featurization (and legality) has already happened on
/// another thread, so all this contributes is the GPU-bound forward (+ parse).
fn consume_featurized_chunks(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    chunks: Vec<FeaturizedChunk>,
    total_states: usize,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let mut evaluations = Vec::with_capacity(total_states);
    for chunk in chunks {
        evaluations.extend(consume_featurized_chunk(
            py,
            evaluator,
            chunk.collated,
            &chunk.legal_sets,
            stats,
        )?);
    }
    debug_assert_eq!(evaluations.len(), total_states, "featurized chunks must tile unique states");
    Ok(evaluations)
}

/// SYNCHRONOUS (single-thread) evaluate of unique leaves: featurize -> forward ->
/// parse. Used by the per-search root eval (`evaluate_states_cached`), where there
/// is no adjacent batch to overlap with. For multi-chunk batches it still runs the
/// intra-batch featurize<->forward channel pipeline (a worker featurizes chunk N+1
/// GIL-free while this GIL thread forwards chunk N), bounded by
/// `eval_pipeline_depth()` so queued-feature memory stays capped. The hot self-play
/// path does NOT use this — it uses the `prepare_eval_refs`/`finish_eval_prepared`
/// split for cross-batch overlap (see `run_searches_to_targets`).
fn evaluate_state_refs_uncached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    n: i16,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let chunk_size = eval_chunk_states();
    // Single chunk: nothing to overlap, featurize + consume inline.
    if states.len() <= chunk_size {
        let collated = featurize_leaf_chunk(states, n, stats);
        let legal_sets: Vec<HashSet<PackedCoord>> =
            states.iter().copied().map(legal_move_set).collect();
        return consume_featurized_chunk(py, evaluator, collated, &legal_sets, stats);
    }

    let chunks: Vec<&[&RustHexoState]> = states.chunks(chunk_size).collect();
    let chunks_ref = &chunks;
    let mut evaluations = Vec::with_capacity(states.len());
    std::thread::scope(|scope| -> PyResult<()> {
        // Bounded channel = backpressure: the featurizer runs at most `depth`
        // chunks ahead, capping queued-feature memory. The worker featurizes +
        // precomputes legality (both GIL-free); the GIL thread forwards + parses.
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, CollatedFeatures, Vec<HashSet<PackedCoord>>)>(
            eval_pipeline_depth(),
        );
        scope.spawn(move || {
            for (index, chunk) in chunks_ref.iter().enumerate() {
                let collated = featurize_leaf_chunk(chunk, n, stats);
                let legal_sets: Vec<HashSet<PackedCoord>> =
                    chunk.iter().copied().map(legal_move_set).collect();
                if tx.send((index, collated, legal_sets)).is_err() {
                    break; // consumer hit an error and dropped the receiver
                }
            }
        });
        for expected in 0..chunks_ref.len() {
            let (index, collated, legal_sets) = rx
                .recv()
                .expect("hexgnn eval featurizer thread terminated before sending all chunks");
            debug_assert_eq!(index, expected, "pipelined eval chunks must arrive in order");
            evaluations.extend(consume_featurized_chunk(py, evaluator, collated, &legal_sets, stats)?);
        }
        Ok(())
    })?;
    Ok(evaluations)
}

/// PRODUCER half: featurize + collate one chunk of unique leaves (pure Rust,
/// rayon-parallel, GIL-free). Records the featurization (encoding) time.
fn featurize_leaf_chunk(
    states: &[&RustHexoState],
    n: i16,
    stats: Option<&SharedEvaluationStats>,
) -> CollatedFeatures {
    let encoding_started = Instant::now();
    let collated = featurize_collate_states(states, n);
    if let Some(stats) = stats {
        let mut s = lock_stats(stats);
        s.evaluator_chunks += 1;
        s.encoded_states += states.len();
        s.max_chunk_states = s.max_chunk_states.max(states.len());
        s.encoding_seconds += encoding_started.elapsed().as_secs_f64();
    }
    collated
}

/// CONSUMER half: pack the featurized chunk into the zero-copy payload, run the
/// Python/Torch forward, and parse per-candidate priors back to `RustEvaluation`
/// (legality intersect + descending sort + normalize). Holds the GIL. `legal_sets`
/// is the precomputed per-row engine legality (computed GIL-free upstream), one
/// entry per state in this chunk, so the consumer needs no `HexoState` reference.
fn consume_featurized_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    mut collated: CollatedFeatures,
    legal_sets: &[HashSet<PackedCoord>],
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let payload_started = Instant::now();
    // The candidate ids/CSR row lengths stay Rust-side: priors come back in the
    // same packed candidate order, so we zip positionally with no id round-trip.
    let candidate_ids: Vec<PackedCoord> = std::mem::take(&mut collated.candidate_ids)
        .iter()
        .map(|&v| v as u32)
        .collect();
    let candidate_counts = std::mem::take(&mut collated.candidate_counts);
    let num_candidates = collated.num_candidates;
    let payload = collated_to_py_dict(py, collated, false)?;
    if let Some(stats) = stats {
        lock_stats(stats).payload_seconds += payload_started.elapsed().as_secs_f64();
    }

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }

    let parse_started = Instant::now();
    let values = read_values(&output, legal_sets.len())?;
    let priors_obj = output.get_item("priors_bytes").map_err(|_| {
        PyValueError::new_err("hexgnn evaluator output missing required priors_bytes")
    })?;
    let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
    require_exact_bytes("priors_bytes", prior_bytes.len(), num_candidates, 4)?;
    if candidate_counts.len() != legal_sets.len() {
        return Err(PyValueError::new_err(format!(
            "hexgnn collation produced {} candidate rows for {} states",
            candidate_counts.len(),
            legal_sets.len()
        )));
    }

    let mut evaluations = Vec::with_capacity(legal_sets.len());
    let mut total_candidates = 0usize;
    let mut offset = 0usize;
    for (row_index, legal_set) in legal_sets.iter().enumerate() {
        let count = candidate_counts[row_index];
        let id_row = &candidate_ids[offset..offset + count];
        let mut prior_row = Vec::with_capacity(count);
        for i in 0..count {
            prior_row.push(read_f32(prior_bytes, offset + i).unwrap_or(0.0));
        }
        offset += count;
        let value = read_value_checked(values[row_index], row_index)?;
        let priors = finalize_candidate_row(legal_set, id_row, &prior_row, row_index)?;
        total_candidates += priors.len();
        evaluations.push(RustEvaluation {
            value,
            legal_action_count: priors.len(),
            priors,
        });
    }
    if let Some(stats) = stats {
        let mut s = lock_stats(stats);
        s.parse_seconds += parse_started.elapsed().as_secs_f64();
        s.encoded_legal_actions += total_candidates;
        s.max_chunk_legal_actions = s.max_chunk_legal_actions.max(total_candidates);
    }
    Ok(evaluations)
}

pub(crate) fn evaluate_states_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
    n: i16,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let requests: Vec<_> = states
        .iter()
        .map(|state| RustEvaluationRequest {
            state,
            state_hash: state_hash(state),
        })
        .collect();
    evaluate_state_refs_cached(py, evaluator, &requests, n, cache, stats, cache_max_states)
}

/// Synchronous cached eval (dedup -> uncached featurize/forward -> cache insert).
/// This is the single-threaded form used by the per-search root eval. The hot
/// self-play path instead uses `prepare_eval_refs` + `finish_eval_prepared` so the
/// featurize half overlaps the previous batch's forward on a worker thread; this
/// function is the equivalent composition for when there is no batch to overlap.
pub(crate) fn evaluate_state_refs_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    n: i16,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let (mut result_slots, slot_to_unique, unique_keys, unique_states) =
        dedup_requests(requests, cache, stats);

    if !unique_states.is_empty() {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_states.len();
        }
        let unique_evals = evaluate_state_refs_uncached(py, evaluator, &unique_states, n, stats)?;
        let unique_evals = arc_and_insert(unique_evals, &unique_keys, cache, stats, cache_max_states);
        scatter_unique_into_slots(&mut result_slots, &slot_to_unique, &unique_evals);
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexgnn evaluation slot must be populated"))
        .collect())
}

/// Dedup a request batch against the shared cache. Returns (pre-filled result
/// slots with cache hits, request->unique map, per-unique state hashes, unique
/// state refs in unique order). Pure Rust / GIL-free: the only shared resource is
/// the thread-safe cache mutex, so this is safe to call on a worker thread.
fn dedup_requests<'a>(
    requests: &[RustEvaluationRequest<'a>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
) -> (
    Vec<Option<Arc<RustEvaluation>>>,
    Vec<Option<usize>>,
    Vec<StateHash>,
    Vec<&'a RustHexoState>,
) {
    let mut result_slots: Vec<Option<Arc<RustEvaluation>>> = vec![None; requests.len()];
    // Pre-size the per-batch scratch from the request count (capacity only — the
    // dedup logic and resulting unique order are unchanged). The unique sets are at
    // most `requests.len()` long; reserving avoids regrowth/rehash mid-loop.
    let mut unique_states: Vec<&RustHexoState> = Vec::with_capacity(requests.len());
    let mut unique_keys: Vec<StateHash> = Vec::with_capacity(requests.len());
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::with_capacity(requests.len());
    let mut slot_to_unique: Vec<Option<usize>> = vec![None; requests.len()];
    if let Some(stats) = stats {
        lock_stats(stats).requested_states += requests.len();
    }

    let dedup_started = Instant::now();
    {
        let cached = lock_cache(cache);
        if let Some(stats) = stats {
            let mut s = lock_stats(stats);
            s.cache_size_peak = s.cache_size_peak.max(cached.len());
        }
        for (index, request) in requests.iter().enumerate() {
            let key = request.state_hash;
            if let Some(cached_eval) = cached.get(&key) {
                result_slots[index] = Some(cached_eval);
                if let Some(stats) = stats {
                    lock_stats(stats).cache_hits += 1;
                }
                continue;
            }
            // Single map probe (was contains_key + get): identical control flow,
            // one fewer hash/lookup on the duplicate-leaf path.
            if let Some(&dup_index) = unique_index_by_key.get(&key) {
                slot_to_unique[index] = Some(dup_index);
                if let Some(stats) = stats {
                    lock_stats(stats).duplicate_hits += 1;
                }
                continue;
            }
            unique_index_by_key.insert(key, unique_states.len());
            unique_keys.push(key);
            slot_to_unique[index] = Some(unique_states.len());
            unique_states.push(request.state);
        }
    }
    if let Some(stats) = stats {
        lock_stats(stats).dedup_seconds += dedup_started.elapsed().as_secs_f64();
    }
    (result_slots, slot_to_unique, unique_keys, unique_states)
}

/// Wrap fresh evaluations in `Arc`, shrink their prior vectors, and insert them
/// into the shared cache under their unique keys (GIL not required).
fn arc_and_insert(
    unique_evals: Vec<RustEvaluation>,
    unique_keys: &[StateHash],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> Vec<Arc<RustEvaluation>> {
    let unique_evals: Vec<Arc<RustEvaluation>> = unique_evals
        .into_iter()
        .map(|mut eval| {
            eval.priors.shrink_to_fit();
            Arc::new(eval)
        })
        .collect();
    let insert_started = Instant::now();
    let mut cached = lock_cache(cache);
    let mut inserted = 0usize;
    for (key, evaluation) in unique_keys.iter().copied().zip(unique_evals.iter()) {
        cached.insert_bounded(key, Arc::clone(evaluation), cache_max_states);
        inserted += 1;
    }
    if let Some(stats) = stats {
        let mut s = lock_stats(stats);
        s.cache_inserts += inserted;
        s.cache_size_peak = s.cache_size_peak.max(cached.len());
        s.cache_insert_seconds += insert_started.elapsed().as_secs_f64();
    }
    unique_evals
}

/// Scatter the per-unique evaluations back into the per-request result slots
/// (cache hits are already populated and skipped).
fn scatter_unique_into_slots(
    result_slots: &mut [Option<Arc<RustEvaluation>>],
    slot_to_unique: &[Option<usize>],
    unique_evals: &[Arc<RustEvaluation>],
) {
    for (index, unique_index) in slot_to_unique.iter().enumerate() {
        if result_slots[index].is_some() {
            continue;
        }
        if let Some(unique_index) = unique_index {
            result_slots[index] = Some(Arc::clone(&unique_evals[*unique_index]));
        }
    }
}

/// Dedup + featurize result of one batch, held between the GIL-FREE producer
/// (`prepare_eval_refs`) and the GIL-BOUND consumer (`finish_eval_prepared`).
///
/// FULLY OWNED — it does NOT borrow the leaf `HexoState`s it was built from. The
/// only thing the consumer needed from the states (engine legality) is precomputed
/// GIL-free into each `FeaturizedChunk`, and the candidate-graph buffers are owned
/// copies. That is what lets the pipeline (`run_searches_to_targets`) prepare the
/// NEXT batch on a worker thread, return it across the `thread::scope` boundary,
/// and finish it on the next loop iteration without any self-referential borrow of
/// `next_leaves`.
pub(crate) struct PreparedEval {
    /// Cache hits are pre-filled; misses/dups are `None` until `finish` fills them.
    result_slots: Vec<Option<Arc<RustEvaluation>>>,
    /// request index -> unique index (or None for cache hits already in a slot).
    slot_to_unique: Vec<Option<usize>>,
    /// State hash per unique leaf, in unique order (for cache insert after forward).
    unique_keys: Vec<StateHash>,
    /// Number of unique misses == total rows the featurized chunks cover.
    unique_count: usize,
    /// Pre-featurized chunks tiling the unique misses; empty if every request hit
    /// the cache or deduped (then `finish` has no forward to run).
    chunks: Vec<FeaturizedChunk>,
}

/// GIL-FREE prepare half: dedup the requests against the shared cache, gather the
/// unique misses, and featurize them. Touches only the (thread-safe) cache mutex
/// and pure-Rust featurization — no Python, no MCTS tree — so it is safe to run on
/// a worker thread concurrently with another batch's torch forward.
pub(crate) fn prepare_eval_refs(
    requests: &[RustEvaluationRequest<'_>],
    n: i16,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
) -> PreparedEval {
    let (result_slots, slot_to_unique, unique_keys, unique_states) =
        dedup_requests(requests, cache, stats);

    let unique_count = unique_states.len();
    let chunks = if unique_states.is_empty() {
        Vec::new()
    } else {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_count;
        }
        featurize_unique_states(&unique_states, n, stats)
    };

    PreparedEval {
        result_slots,
        slot_to_unique,
        unique_keys,
        unique_count,
        chunks,
    }
}

/// GIL-BOUND finish half: run the torch forward + parse for the pre-featurized
/// chunks, insert the fresh evaluations into the shared cache, and scatter every
/// request slot (cache hits, dups, and fresh misses) into the final result vector.
pub(crate) fn finish_eval_prepared(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    prepared: PreparedEval,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let PreparedEval {
        mut result_slots,
        slot_to_unique,
        unique_keys,
        unique_count,
        chunks,
    } = prepared;

    if unique_count > 0 {
        let unique_evals = consume_featurized_chunks(py, evaluator, chunks, unique_count, stats)?;
        let unique_evals = arc_and_insert(unique_evals, &unique_keys, cache, stats, cache_max_states);
        scatter_unique_into_slots(&mut result_slots, &slot_to_unique, &unique_evals);
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexgnn evaluation slot must be populated"))
        .collect())
}

/// Engine legality (set of legal placement coords) for one state. Pure Rust /
/// GIL-free, so it is precomputed in the prepare half and carried into the
/// GIL-bound finalize half, decoupling `finalize_candidate_row` from the live
/// `HexoState` references (which lets `PreparedEval` be fully owned).
fn legal_move_set(state: &RustHexoState) -> HashSet<PackedCoord> {
    let mut legal_cells = Vec::with_capacity(state.legal_move_count());
    state.write_legal_moves(&mut legal_cells);
    // Reserve the final set up front (the set is only probed via `.contains`, so
    // its internal order is irrelevant — pre-sizing changes capacity only and the
    // membership is identical) instead of growing/rehashing while inserting.
    let mut set = HashSet::with_capacity(legal_cells.len());
    set.extend(legal_cells.into_iter().map(pack_coord));
    set
}

/// Intersect the evaluator's candidate row with engine legality, drop bad/dup
/// priors, then DESCENDING-sort + normalize (the copied tree's prior contract).
/// `legal_set` is the precomputed legality for this row's state.
fn finalize_candidate_row(
    legal_set: &HashSet<PackedCoord>,
    candidate_row: &[PackedCoord],
    prior_row: &[f32],
    row_index: usize,
) -> PyResult<Vec<(PackedCoord, f32)>> {
    let mut seen = HashSet::with_capacity(candidate_row.len());
    let mut priors: Vec<(PackedCoord, f32)> = Vec::with_capacity(candidate_row.len());
    let mut total = 0.0f32;
    for (action_id, prior) in candidate_row.iter().copied().zip(prior_row.iter().copied()) {
        if !legal_set.contains(&action_id) || !seen.insert(action_id) {
            continue;
        }
        if !prior.is_finite() || prior < 0.0 {
            continue;
        }
        priors.push((action_id, prior));
        total += prior;
    }
    if priors.is_empty() || total <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "hexgnn evaluator produced no legal positive-mass candidate priors for non-terminal row {row_index}"
        )));
    }
    priors.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    for entry in priors.iter_mut() {
        entry.1 /= total;
    }
    Ok(priors)
}

fn read_values(output: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f32>> {
    if let Ok(values_obj) = output.get_item("values_bytes") {
        let bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
        require_exact_bytes("values_bytes", bytes.len(), expected, 4)?;
        let mut values = Vec::with_capacity(expected);
        for index in 0..expected {
            values.push(read_f32(bytes, index).unwrap_or(0.0));
        }
        return Ok(values);
    }
    let values_obj = output.get_item("values")?;
    let mut values = Vec::with_capacity(expected);
    for item in values_obj.try_iter()? {
        values.push(item?.extract::<f32>()?);
    }
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "hexgnn evaluator returned {} values for {} states",
            values.len(),
            expected
        )));
    }
    Ok(values)
}

fn read_value_checked(value: f32, row_index: usize) -> PyResult<f32> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "hexgnn values row {row_index} must be finite, got {value}"
        )));
    }
    Ok(value.clamp(-1.0, 1.0))
}

fn require_exact_bytes(
    name: &str,
    actual_bytes: usize,
    expected_items: usize,
    bytes_per_item: usize,
) -> PyResult<()> {
    let Some(expected_bytes) = expected_items.checked_mul(bytes_per_item) else {
        return Err(PyValueError::new_err(format!("{name} expected byte count overflow")));
    };
    if actual_bytes != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "{name} has {actual_bytes} bytes, expected {expected_bytes}"
        )));
    }
    Ok(())
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}
