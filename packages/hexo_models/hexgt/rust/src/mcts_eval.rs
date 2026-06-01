//! Python/PyTorch evaluator adapter for hexgt MCTS.
//!
//! The search tree (copied from dense_cnn) asks for batches of leaf states; this
//! module is the boundary to the torch GNN evaluator:
//!
//! 1. Hash incoming engine states with `hexo_utils` (transposition + duplicate
//!    coalescing), reusing cached evaluations.
//! 2. Build the hexgt typed-graph facts for each unique leaf **in Rust** from the
//!    leaf `HexoState` (no Py->Rust reclone) via the shared `candidates` builder,
//!    so search inputs == training inputs.
//! 3. Call the Python evaluator (`HexgtInference.evaluate_graph_facts`).
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
use super::features::{collated_to_py_dict, featurize_collate_states};

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

/// Evaluate a chunk of unique leaf states: build graph-facts -> Python forward ->
/// parse per-candidate priors. `n` is the candidate radius (move vocabulary).
fn evaluate_state_refs_uncached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    n: i16,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    if states.len() > HEXGT_EVAL_CHUNK_STATES {
        let mut evaluations = Vec::with_capacity(states.len());
        for chunk in states.chunks(HEXGT_EVAL_CHUNK_STATES) {
            evaluations.extend(evaluate_states_chunk(py, evaluator, chunk, n, stats)?);
        }
        return Ok(evaluations);
    }
    evaluate_states_chunk(py, evaluator, states, n, stats)
}

fn evaluate_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    n: i16,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let encoding_started = Instant::now();
    // Featurize + collate the whole leaf batch IN RUST (rayon across graphs:
    // build_graph + D6-invariant featurization + disjoint-graph collation),
    // emitted to Python as zero-copy buffers it just frombuffer+forwards. This
    // replaces the former per-graph Python featurization (the ~88% bottleneck,
    // HEXGT_DECISIONS Phase-5c). The GIL is released for the rayon work.
    // NOTE: do NOT release the GIL here. This callback runs inside the MCTS
    // select<->eval pipeline (which is itself driven across rayon threads); a
    // py.detach() per chunk triggers GIL hand-off contention that collapsed
    // throughput ~9x. featurize_collate_states still parallelizes the per-graph
    // featurization across rayon internally (pure-Rust workers need no GIL).
    let mut collated = featurize_collate_states(states, n);
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
        let mut s = lock_stats(stats);
        s.evaluator_chunks += 1;
        s.encoded_states += states.len();
        s.max_chunk_states = s.max_chunk_states.max(states.len());
        s.encoding_seconds += encoding_started.elapsed().as_secs_f64();
    }

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }

    let parse_started = Instant::now();
    let values = read_values(&output, states.len())?;
    let priors_obj = output.get_item("priors_bytes").map_err(|_| {
        PyValueError::new_err("hexgt evaluator output missing required priors_bytes")
    })?;
    let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
    require_exact_bytes("priors_bytes", prior_bytes.len(), num_candidates, 4)?;
    if candidate_counts.len() != states.len() {
        return Err(PyValueError::new_err(format!(
            "hexgt collation produced {} candidate rows for {} states",
            candidate_counts.len(),
            states.len()
        )));
    }

    let mut evaluations = Vec::with_capacity(states.len());
    let mut total_candidates = 0usize;
    let mut offset = 0usize;
    for (row_index, state) in states.iter().enumerate() {
        let count = candidate_counts[row_index];
        let id_row = &candidate_ids[offset..offset + count];
        let mut prior_row = Vec::with_capacity(count);
        for i in 0..count {
            prior_row.push(read_f32(prior_bytes, offset + i).unwrap_or(0.0));
        }
        offset += count;
        let value = read_value_checked(values[row_index], row_index)?;
        let priors = finalize_candidate_row(state, id_row, &prior_row, row_index)?;
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

pub(crate) fn evaluate_state_refs_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    n: i16,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let mut result_slots: Vec<Option<Arc<RustEvaluation>>> = vec![None; requests.len()];
    let mut unique_states: Vec<&RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();
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
            if unique_index_by_key.contains_key(&key) {
                slot_to_unique[index] = unique_index_by_key.get(&key).copied();
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

    if !unique_states.is_empty() {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_states.len();
        }
        let unique_evals = evaluate_state_refs_uncached(py, evaluator, &unique_states, n, stats)?;
        let unique_evals: Vec<Arc<RustEvaluation>> = unique_evals
            .into_iter()
            .map(|mut eval| {
                eval.priors.shrink_to_fit();
                Arc::new(eval)
            })
            .collect();
        {
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
        }
        for (index, unique_index) in slot_to_unique.into_iter().enumerate() {
            if result_slots[index].is_some() {
                continue;
            }
            if let Some(unique_index) = unique_index {
                result_slots[index] = Some(Arc::clone(&unique_evals[unique_index]));
            }
        }
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexgt evaluation slot must be populated"))
        .collect())
}

/// Intersect the evaluator's candidate row with engine legality, drop bad/dup
/// priors, then DESCENDING-sort + normalize (the copied tree's prior contract).
fn finalize_candidate_row(
    state: &RustHexoState,
    candidate_row: &[PackedCoord],
    prior_row: &[f32],
    row_index: usize,
) -> PyResult<Vec<(PackedCoord, f32)>> {
    let mut legal_cells = Vec::with_capacity(state.legal_move_count());
    state.write_legal_moves(&mut legal_cells);
    let legal_set: HashSet<PackedCoord> = legal_cells.into_iter().map(pack_coord).collect();

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
            "hexgt evaluator produced no legal positive-mass candidate priors for non-terminal row {row_index}"
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
            "hexgt evaluator returned {} values for {} states",
            values.len(),
            expected
        )));
    }
    Ok(values)
}

fn read_value_checked(value: f32, row_index: usize) -> PyResult<f32> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "hexgt values row {row_index} must be finite, got {value}"
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
