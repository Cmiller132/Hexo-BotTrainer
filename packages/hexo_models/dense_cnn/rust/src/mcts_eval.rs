//! Python/PyTorch evaluator adapter for dense CNN MCTS.
//!
//! MCTS owns the tree and game-state mutations, while PyTorch remains the neural
//! evaluator. This file is the boundary between those worlds:
//!
//! 1. Hash incoming engine states with `hexo_utils`.
//! 2. Reuse cached evaluations and coalesce duplicate uncached states.
//! 3. Encode unique states into strict tensor byte payloads.
//! 4. Call `DenseCNNInference.evaluate_model1_payload`.
//! 5. Parse exact value/prior byte buffers back into `RustEvaluation`.
//! 6. Validate legality, finiteness, uniqueness, and prior mass before search
//!    sees the result.
//!
//! No fallback evaluator format is accepted here. If Python returns malformed
//! bytes, the PyO3 boundary raises `PyValueError`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::Instant;

use hexo_engine::{pack_coord, HexoState as RustHexoState, PackedCoord};
use hexo_utils::{hash_state, StateHash};

use super::constants::*;
use super::encoding::{
    encode_model1_state_for_mcts, encode_model1_state_half_for_mcts, model1_coord_from_flat,
};

#[derive(Clone, Debug)]
pub(crate) struct RustEvaluation {
    /// Scalar value from the perspective of the evaluated state's current player.
    pub(crate) value: f32,
    /// Total legal moves in the authoritative engine state, including moves
    /// outside the dense crop or outside a candidate-limited result.
    pub(crate) legal_action_count: usize,
    /// Ranked visible priors. Hidden legal actions are represented later by
    /// `mcts_tree` using `legal_action_count`.
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
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RustEvaluationCache {
    entries: HashMap<StateHash, RustEvaluation>,
    insertion_order: VecDeque<StateHash>,
    candidate_limit_initialized: bool,
    candidate_limit: Option<usize>,
}

impl RustEvaluationCache {
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
    }

    fn ensure_candidate_limit(&mut self, candidate_limit: Option<usize>) {
        if !self.candidate_limit_initialized {
            self.candidate_limit_initialized = true;
            self.candidate_limit = candidate_limit;
            return;
        }
        if self.candidate_limit != candidate_limit {
            self.clear();
            self.candidate_limit = candidate_limit;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, key: &StateHash) -> Option<&RustEvaluation> {
        self.entries.get(key)
    }

    fn insert_bounded(&mut self, key: StateHash, evaluation: RustEvaluation, max_states: usize) {
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

pub(crate) type SharedEvaluationCache = Rc<RefCell<RustEvaluationCache>>;
pub(crate) type SharedEvaluationStats = Rc<RefCell<EvaluationStats>>;

pub(crate) fn new_shared_evaluation_cache() -> SharedEvaluationCache {
    Rc::new(RefCell::new(RustEvaluationCache::default()))
}

pub(crate) fn new_shared_evaluation_stats() -> SharedEvaluationStats {
    Rc::new(RefCell::new(EvaluationStats::default()))
}

pub(crate) fn state_hash(state: &RustHexoState) -> StateHash {
    hash_state(state)
}

pub(crate) struct RustEvaluationRequest<'a> {
    pub(crate) state: &'a RustHexoState,
    pub(crate) state_hash: StateHash,
}

fn evaluate_model1_state_refs(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    stats: Option<&SharedEvaluationStats>,
    prior_candidate_limit: Option<usize>,
) -> PyResult<Vec<RustEvaluation>> {
    // Keep Python callback batches bounded. The caller may request many leaf
    // states at once, but Torch memory and callback latency are better behaved
    // when large batches are chunked here.
    if states.len() > MODEL1_EVAL_CHUNK_STATES {
        let mut evaluations = Vec::with_capacity(states.len());
        for chunk in states.chunks(MODEL1_EVAL_CHUNK_STATES) {
            evaluations.extend(evaluate_model1_states_chunk(
                py,
                evaluator,
                chunk,
                stats,
                prior_candidate_limit,
            )?);
        }
        return Ok(evaluations);
    }
    evaluate_model1_states_chunk(py, evaluator, states, stats, prior_candidate_limit)
}

fn evaluate_model1_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    stats: Option<&SharedEvaluationStats>,
    prior_candidate_limit: Option<usize>,
) -> PyResult<Vec<RustEvaluation>> {
    // Candidate-limited mode reduces traffic by sending half-precision inputs
    // and asking Python to choose top-k priors from the legal plane. Full-prior
    // mode sends explicit legal flats and expects one prior per flat.
    let use_half_inputs = prior_candidate_limit.is_some();
    let use_legal_plane_mask = prior_candidate_limit.is_some();
    let encoding_started = Instant::now();
    let encoded: Vec<_> = if use_half_inputs {
        states
            .par_iter()
            .map(|state| encode_model1_state_half_for_mcts(state))
            .collect()
    } else {
        states
            .par_iter()
            .map(|state| encode_model1_state_for_mcts(state, !use_legal_plane_mask))
            .collect()
    };
    let plane_values_per_row = MODEL1_INPUT_CHANNELS * MODEL1_BOARD_AREA;
    let mut planes = Vec::new();
    let mut half_planes = Vec::new();
    if use_half_inputs {
        half_planes = vec![0u16; encoded.len() * plane_values_per_row];
        half_planes
            .par_chunks_mut(plane_values_per_row)
            .zip(encoded.par_iter())
            .for_each(|(target, row)| target.copy_from_slice(&row.half_planes));
    } else {
        planes = vec![0.0f32; encoded.len() * plane_values_per_row];
        planes
            .par_chunks_mut(plane_values_per_row)
            .zip(encoded.par_iter())
            .for_each(|(target, row)| target.copy_from_slice(&row.planes));
    }
    let mut legal_flat_indices = Vec::new();
    let mut legal_row_offsets = Vec::new();
    if !use_legal_plane_mask {
        legal_row_offsets = Vec::with_capacity(encoded.len() + 1);
        legal_row_offsets.push(0i64);
        for row in &encoded {
            legal_flat_indices.extend_from_slice(&row.legal_flat_indices);
            legal_row_offsets.push(legal_flat_indices.len() as i64);
        }
    }

    let byte_len = if use_half_inputs {
        half_planes.len() * std::mem::size_of::<u16>()
    } else {
        planes.len() * std::mem::size_of::<f32>()
    };
    let bytes = if use_half_inputs {
        unsafe { std::slice::from_raw_parts(half_planes.as_ptr() as *const u8, byte_len) }
    } else {
        unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const u8, byte_len) }
    };
    let flat_byte_len = legal_flat_indices.len() * std::mem::size_of::<i64>();
    let flat_bytes = unsafe {
        std::slice::from_raw_parts(legal_flat_indices.as_ptr() as *const u8, flat_byte_len)
    };
    if let Some(stats) = stats {
        let mut stats = stats.borrow_mut();
        stats.evaluator_chunks += 1;
        stats.encoded_states += encoded.len();
        stats.encoded_legal_actions += legal_flat_indices.len();
        stats.max_chunk_states = stats.max_chunk_states.max(encoded.len());
        stats.max_chunk_legal_actions = stats.max_chunk_legal_actions.max(legal_flat_indices.len());
        stats.input_bytes += byte_len;
        stats.legal_index_bytes += flat_byte_len;
        stats.encoding_seconds += encoding_started.elapsed().as_secs_f64();
    }
    let payload = PyDict::new(py);
    // This dictionary is the native evaluator ABI consumed by
    // `DenseCNNInference.evaluate_model1_payload`.
    payload.set_item("inputs", PyBytes::new(py, bytes))?;
    if use_half_inputs {
        payload.set_item("input_dtype", "float16")?;
    }
    payload.set_item(
        "shape",
        (
            encoded.len(),
            MODEL1_INPUT_CHANNELS,
            MODEL1_BOARD_SIZE,
            MODEL1_BOARD_SIZE,
        ),
    )?;
    if let Some(limit) = prior_candidate_limit {
        payload.set_item("max_prior_candidates", limit)?;
        payload.set_item("legal_mask_from_inputs", true)?;
    } else {
        payload.set_item("legal_flat_indices_bytes", PyBytes::new(py, flat_bytes))?;
        payload.set_item("legal_row_offsets", PyTuple::new(py, legal_row_offsets)?)?;
    }

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        stats.borrow_mut().evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }
    let values_obj = output.get_item("values_bytes").map_err(|_| {
        PyValueError::new_err("dense_cnn evaluator output missing required values_bytes")
    })?;
    let priors_obj = output.get_item("priors_bytes").map_err(|_| {
        PyValueError::new_err("dense_cnn evaluator output missing required priors_bytes")
    })?;
    let value_bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
    let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
    require_exact_bytes("values_bytes", value_bytes.len(), encoded.len(), 4)?;
    if let Some(stats) = stats {
        let mut stats = stats.borrow_mut();
        stats.value_bytes += value_bytes.len();
        stats.prior_bytes += prior_bytes.len();
    }
    let mut evaluations = Vec::with_capacity(encoded.len());
    if prior_candidate_limit.is_some() {
        // Python returns selected crop flats in candidate-limited mode. Convert
        // each flat back to a packed action id, then prove it is legal in the
        // encoded engine state before accepting the prior.
        let flats_obj = output.get_item("selected_flat_indices_bytes").map_err(|_| {
            PyValueError::new_err(
                "candidate-limited dense_cnn evaluator output missing selected_flat_indices_bytes",
            )
        })?;
        let selected_offsets_obj = output.get_item("selected_row_offsets").map_err(|_| {
            PyValueError::new_err(
                "candidate-limited dense_cnn evaluator output missing selected_row_offsets",
            )
        })?;
        let flat_bytes = flats_obj.downcast::<PyBytes>()?.as_bytes();
        let selected_offsets = selected_offsets_obj.extract::<Vec<usize>>()?;
        validate_row_offsets("selected_row_offsets", &selected_offsets, encoded.len())?;
        let selected_count = selected_offsets.last().copied().unwrap_or(0);
        require_exact_bytes("priors_bytes", prior_bytes.len(), selected_count, 4)?;
        require_exact_bytes(
            "selected_flat_indices_bytes",
            flat_bytes.len(),
            selected_count,
            8,
        )?;
        for (row_index, row) in encoded.iter().enumerate() {
            let value = read_value(value_bytes, row_index)?;
            let start = selected_offsets[row_index];
            let end = selected_offsets[row_index + 1];
            let mut row_priors = Vec::with_capacity(end - start);
            for selected_index in start..end {
                let flat = read_i64_required("selected_flat_indices_bytes", flat_bytes, selected_index)?;
                if flat < 0 {
                    return Err(PyValueError::new_err(format!(
                        "selected_flat_indices_bytes row {row_index} entry {selected_index} is negative"
                    )));
                }
                let Some(coord) = model1_coord_from_flat(flat as usize, row.center) else {
                    return Err(PyValueError::new_err(format!(
                        "selected_flat_indices_bytes row {row_index} entry {selected_index} is outside the dense crop"
                    )));
                };
                let action_id = pack_coord(coord);
                if !row.legal_action_ids.contains(&action_id) {
                    return Err(PyValueError::new_err(format!(
                        "selected_flat_indices_bytes row {row_index} entry {selected_index} maps to illegal action {action_id}"
                    )));
                }
                let prior = read_prior(prior_bytes, selected_index, row_index)?;
                row_priors.push((action_id, prior));
            }
            finalize_model_priors(
                &mut row_priors,
                row.all_legal_action_count,
                prior_candidate_limit,
                true,
                row_index,
            )?;
            evaluations.push(RustEvaluation {
                value,
                legal_action_count: row.all_legal_action_count,
                priors: row_priors,
            });
        }
    } else {
        // Full-prior mode is positional: priors_bytes item N corresponds to the
        // Nth legal action id written into the row-offset payload sent above.
        let mut prior_offset = 0usize;
        let expected_prior_count: usize =
            encoded.iter().map(|row| row.legal_action_ids.len()).sum();
        require_exact_bytes("priors_bytes", prior_bytes.len(), expected_prior_count, 4)?;
        for (row_index, row) in encoded.iter().enumerate() {
            let value = read_value(value_bytes, row_index)?;
            let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
            for action_id in row.legal_action_ids.iter().copied() {
                let prior = read_prior(prior_bytes, prior_offset, row_index)?;
                row_priors.push((action_id, prior));
                prior_offset += 1;
            }
            finalize_model_priors(
                &mut row_priors,
                row.all_legal_action_count,
                prior_candidate_limit,
                false,
                row_index,
            )?;
            evaluations.push(RustEvaluation {
                value,
                legal_action_count: row.all_legal_action_count,
                priors: row_priors,
            });
        }
    }
    Ok(evaluations)
}

pub(crate) fn evaluate_model1_states_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    prior_candidate_limit: Option<usize>,
    cache_max_states: usize,
) -> PyResult<Vec<RustEvaluation>> {
    let requests: Vec<_> = states
        .iter()
        .map(|state| RustEvaluationRequest {
            state,
            state_hash: state_hash(state),
        })
        .collect();
    evaluate_model1_state_refs_cached(
        py,
        evaluator,
        &requests,
        cache,
        stats,
        prior_candidate_limit,
        cache_max_states,
    )
}

pub(crate) fn evaluate_model1_state_refs_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    prior_candidate_limit: Option<usize>,
    cache_max_states: usize,
) -> PyResult<Vec<RustEvaluation>> {
    // Cache slots preserve caller order. `unique_states` contains only misses,
    // and `slot_to_unique` maps duplicate misses back to their first occurrence.
    let mut result_slots: Vec<Option<RustEvaluation>> = vec![None; requests.len()];
    let mut unique_states: Vec<&RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();
    let mut slot_to_unique: Vec<Option<usize>> = vec![None; requests.len()];
    if let Some(stats) = stats {
        stats.borrow_mut().requested_states += requests.len();
    }

    cache
        .borrow_mut()
        .ensure_candidate_limit(prior_candidate_limit);

    {
        let cached = cache.borrow();
        if let Some(stats) = stats {
            let mut stats = stats.borrow_mut();
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
        }
        for (index, request) in requests.iter().enumerate() {
            let key = request.state_hash;
            if let Some(cached_eval) = cached.get(&key) {
                result_slots[index] = Some(cached_eval.clone());
                if let Some(stats) = stats {
                    stats.borrow_mut().cache_hits += 1;
                }
                continue;
            }
            if unique_index_by_key.contains_key(&key) {
                slot_to_unique[index] = unique_index_by_key.get(&key).copied();
                if let Some(stats) = stats {
                    stats.borrow_mut().duplicate_hits += 1;
                }
                continue;
            }
            unique_index_by_key.insert(key, unique_states.len());
            unique_keys.push(key);
            slot_to_unique[index] = Some(unique_states.len());
            unique_states.push(request.state);
        }
    }

    if !unique_states.is_empty() {
        if let Some(stats) = stats {
            stats.borrow_mut().unique_states += unique_states.len();
        }
        let unique_evals = evaluate_model1_state_refs(
            py,
            evaluator,
            &unique_states,
            stats,
            prior_candidate_limit,
        )?;
        {
            let mut cached = cache.borrow_mut();
            let mut inserted = 0usize;
            for (offset, (key, evaluation)) in unique_keys
                .iter()
                .copied()
                .zip(unique_evals.iter())
                .enumerate()
            {
                let _ = offset;
                cached.insert_bounded(key, evaluation.clone(), cache_max_states);
                inserted += 1;
            }
            if let Some(stats) = stats {
                let mut stats = stats.borrow_mut();
                stats.cache_inserts += inserted;
                stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
            }
        }
        for (index, unique_index) in slot_to_unique.into_iter().enumerate() {
            if result_slots[index].is_some() {
                continue;
            }
            if let Some(unique_index) = unique_index {
                result_slots[index] = Some(unique_evals[unique_index].clone());
            }
        }
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every model1 evaluation slot must be populated"))
        .collect())
}

fn finalize_model_priors(
    priors: &mut Vec<(PackedCoord, f32)>,
    legal_action_count: usize,
    prior_candidate_limit: Option<usize>,
    already_ranked: bool,
    row_index: usize,
) -> PyResult<()> {
    // This is the last validation step before priors become tree edges. The tree
    // assumes every visible prior is finite, nonnegative, unique, legal, and has
    // positive total mass for nonterminal rows.
    if legal_action_count == 0 {
        if priors.is_empty() {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for terminal row {row_index}",
            priors.len()
        )));
    }
    let limit = match prior_candidate_limit {
        Some(0) => {
            return Err(PyValueError::new_err(
                "prior_candidate_limit must be > 0 when provided",
            ));
        }
        Some(count) => count.min(legal_action_count),
        None => legal_action_count,
    };

    if priors.is_empty() {
        return Err(PyValueError::new_err(format!(
            "evaluator returned no priors for non-terminal row {row_index}"
        )));
    }
    if priors.len() > limit {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for row {row_index}, above limit {limit}",
            priors.len()
        )));
    }
    let mut seen = HashSet::with_capacity(priors.len());
    for (action_id, prior) in priors.iter().copied() {
        if !seen.insert(action_id) {
            return Err(PyValueError::new_err(format!(
                "evaluator returned duplicate action {action_id} in row {row_index}"
            )));
        }
        if !prior.is_finite() || prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "evaluator returned invalid prior {prior} for action {action_id} in row {row_index}"
            )));
        }
    }
    let total: f32 = priors.iter().map(|(_, prior)| *prior).sum();
    if !priors.is_empty() && total <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "evaluator returned zero total prior mass for row {row_index}"
        )));
    }

    if !already_ranked {
        priors.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
    }
    Ok(())
}

fn require_exact_bytes(
    name: &str,
    actual_bytes: usize,
    expected_items: usize,
    bytes_per_item: usize,
) -> PyResult<()> {
    // Byte-length checks keep payload corruption from turning into accidental
    // reinterpretation of shorter or longer buffers.
    let Some(expected_bytes) = expected_items.checked_mul(bytes_per_item) else {
        return Err(PyValueError::new_err(format!(
            "{name} expected byte count overflow"
        )));
    };
    if actual_bytes != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "{name} has {actual_bytes} bytes, expected {expected_bytes}"
        )));
    }
    Ok(())
}

fn validate_row_offsets(name: &str, offsets: &[usize], rows: usize) -> PyResult<()> {
    if offsets.len() != rows + 1 {
        return Err(PyValueError::new_err(format!(
            "{name} has {} entries, expected {}",
            offsets.len(),
            rows + 1
        )));
    }
    if offsets.first().copied().unwrap_or(1) != 0 {
        return Err(PyValueError::new_err(format!("{name} must start at 0")));
    }
    for pair in offsets.windows(2) {
        if pair[1] < pair[0] {
            return Err(PyValueError::new_err(format!(
                "{name} must be monotonically nondecreasing"
            )));
        }
    }
    Ok(())
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_value(bytes: &[u8], index: usize) -> PyResult<f32> {
    let value = read_f32_required("values_bytes", bytes, index)?;
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(PyValueError::new_err(format!(
            "values_bytes row {index} must be finite and in [-1, 1], got {value}"
        )));
    }
    Ok(value)
}

fn read_prior(bytes: &[u8], index: usize, row_index: usize) -> PyResult<f32> {
    let value = read_f32_required("priors_bytes", bytes, index)?;
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "priors_bytes row {row_index} entry {index} must be finite and >= 0, got {value}"
        )));
    }
    Ok(value)
}

fn read_f32_required(name: &str, bytes: &[u8], index: usize) -> PyResult<f32> {
    read_f32(bytes, index).ok_or_else(|| {
        PyValueError::new_err(format!(
            "{name} missing f32 at item index {index}"
        ))
    })
}

fn read_i64(bytes: &[u8], index: usize) -> Option<i64> {
    let start = index.checked_mul(8)?;
    let chunk = bytes.get(start..start + 8)?;
    Some(i64::from_ne_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ]))
}

fn read_i64_required(name: &str, bytes: &[u8], index: usize) -> PyResult<i64> {
    read_i64(bytes, index).ok_or_else(|| {
        PyValueError::new_err(format!(
            "{name} missing i64 at item index {index}"
        ))
    })
}
