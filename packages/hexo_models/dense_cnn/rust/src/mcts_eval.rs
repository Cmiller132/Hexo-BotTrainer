//! Python/PyTorch evaluator adapter for dense CNN MCTS.
//!
//! MCTS owns the tree and game-state mutations, while PyTorch remains the neural
//! evaluator. This file is the boundary between those worlds: it encodes engine
//! states into the dense-cnn tensor payload, calls the Python evaluator once per
//! batch, and caches exact model evaluations by the history-sensitive state
//! identity derived in `hexo_utils`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use hexo_engine::{pack_coord, HexoState as RustHexoState, PackedCoord};
use hexo_utils::{hash_state, StateHash};

use super::constants::*;
use super::encoding::{encode_model1_state, model1_coord_from_flat};

#[derive(Clone, Debug)]
pub(crate) struct RustEvaluation {
    pub(crate) value: f32,
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
}

pub(crate) type SharedEvaluationCache = Rc<RefCell<HashMap<StateHash, RustEvaluation>>>;
pub(crate) type SharedEvaluationStats = Rc<RefCell<EvaluationStats>>;

#[pyclass(unsendable)]
pub(crate) struct Model1MctsEvaluationCache {
    cache: SharedEvaluationCache,
    max_states: usize,
}

#[pymethods]
impl Model1MctsEvaluationCache {
    #[new]
    #[pyo3(signature = (max_states=None))]
    fn new(max_states: Option<usize>) -> Self {
        Self {
            cache: new_shared_evaluation_cache(),
            max_states: max_states.unwrap_or(MODEL1_EVAL_CACHE_MAX_STATES).max(1),
        }
    }

    fn clear(&self) {
        self.cache.borrow_mut().clear();
    }

    fn len(&self) -> usize {
        self.cache.borrow().len()
    }

    fn max_states(&self) -> usize {
        self.max_states
    }
}

impl Model1MctsEvaluationCache {
    pub(crate) fn shared_cache(&self) -> SharedEvaluationCache {
        self.cache.clone()
    }

    pub(crate) fn max_state_count(&self) -> usize {
        self.max_states
    }
}

pub(crate) fn new_shared_evaluation_cache() -> SharedEvaluationCache {
    Rc::new(RefCell::new(HashMap::new()))
}

pub(crate) fn new_shared_evaluation_stats() -> SharedEvaluationStats {
    Rc::new(RefCell::new(EvaluationStats::default()))
}

pub(crate) fn state_hash(state: &RustHexoState) -> StateHash {
    hash_state(state)
}

fn evaluate_model1_states(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
    stats: Option<&SharedEvaluationStats>,
    prior_candidate_limit: Option<usize>,
) -> PyResult<Vec<RustEvaluation>> {
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
    states: &[RustHexoState],
    stats: Option<&SharedEvaluationStats>,
    prior_candidate_limit: Option<usize>,
) -> PyResult<Vec<RustEvaluation>> {
    let encoded: Vec<_> = states.iter().map(encode_model1_state).collect();
    let mut planes = Vec::with_capacity(encoded.len() * MODEL1_INPUT_CHANNELS * MODEL1_BOARD_AREA);
    let mut legal_flat_indices = Vec::new();
    let mut legal_row_offsets = Vec::with_capacity(encoded.len() + 1);
    legal_row_offsets.push(0i64);
    for row in &encoded {
        planes.extend_from_slice(&row.planes);
        legal_flat_indices.extend_from_slice(&row.legal_flat_indices);
        legal_row_offsets.push(legal_flat_indices.len() as i64);
    }

    let byte_len = planes.len() * std::mem::size_of::<f32>();
    let bytes = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const u8, byte_len) };
    let use_legal_plane_mask = prior_candidate_limit.is_some();
    let flat_byte_len = if use_legal_plane_mask {
        0
    } else {
        legal_flat_indices.len() * std::mem::size_of::<i64>()
    };
    let flat_bytes = if use_legal_plane_mask {
        &[][..]
    } else {
        unsafe {
            std::slice::from_raw_parts(legal_flat_indices.as_ptr() as *const u8, flat_byte_len)
        }
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
    }
    let payload = PyDict::new(py);
    payload.set_item("inputs", PyBytes::new(py, bytes))?;
    payload.set_item(
        "shape",
        (
            encoded.len(),
            MODEL1_INPUT_CHANNELS,
            MODEL1_BOARD_SIZE,
            MODEL1_BOARD_SIZE,
        ),
    )?;
    payload.set_item("legal_flat_indices_bytes", PyBytes::new(py, flat_bytes))?;
    payload.set_item("legal_row_offsets", PyTuple::new(py, legal_row_offsets)?)?;
    if let Some(limit) = prior_candidate_limit {
        payload.set_item("max_prior_candidates", limit.max(1))?;
        payload.set_item("legal_mask_from_inputs", true)?;
    }

    let output = evaluator.call1((payload,))?;
    if let (Ok(values_obj), Ok(priors_obj)) = (
        output.get_item("values_bytes"),
        output.get_item("priors_bytes"),
    ) {
        let value_bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
        let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
        if let Some(stats) = stats {
            let mut stats = stats.borrow_mut();
            stats.value_bytes += value_bytes.len();
            stats.prior_bytes += prior_bytes.len();
        }
        let mut evaluations = Vec::with_capacity(encoded.len());
        if let (Ok(flats_obj), Ok(selected_offsets_obj)) = (
            output.get_item("selected_flat_indices_bytes"),
            output.get_item("selected_row_offsets"),
        ) {
            let flat_bytes = flats_obj.downcast::<PyBytes>()?.as_bytes();
            let selected_offsets = selected_offsets_obj.extract::<Vec<usize>>()?;
            for (index, row) in encoded.iter().enumerate() {
                let value = read_f32(value_bytes, index).unwrap_or(0.0).clamp(-1.0, 1.0);
                let start = selected_offsets.get(index).copied().unwrap_or(0);
                let end = selected_offsets
                    .get(index + 1)
                    .copied()
                    .unwrap_or(start)
                    .max(start);
                let mut row_priors = Vec::with_capacity(end.saturating_sub(start));
                for selected_index in start..end {
                    let Some(flat) = read_i64(flat_bytes, selected_index) else {
                        continue;
                    };
                    if flat < 0 {
                        continue;
                    }
                    let Some(coord) = model1_coord_from_flat(flat as usize, row.center) else {
                        continue;
                    };
                    let prior = read_f32(prior_bytes, selected_index)
                        .unwrap_or(0.0)
                        .max(0.0);
                    row_priors.push((pack_coord(coord), prior));
                }
                evaluations.push(RustEvaluation {
                    value,
                    priors: row_priors,
                });
            }
        } else if let (Ok(ordinals_obj), Ok(selected_offsets_obj)) = (
            output.get_item("selected_legal_ordinals_bytes"),
            output.get_item("selected_row_offsets"),
        ) {
            let ordinal_bytes = ordinals_obj.downcast::<PyBytes>()?.as_bytes();
            let selected_offsets = selected_offsets_obj.extract::<Vec<usize>>()?;
            for (index, row) in encoded.iter().enumerate() {
                let value = read_f32(value_bytes, index).unwrap_or(0.0).clamp(-1.0, 1.0);
                let start = selected_offsets.get(index).copied().unwrap_or(0);
                let end = selected_offsets
                    .get(index + 1)
                    .copied()
                    .unwrap_or(start)
                    .max(start);
                let mut row_priors = Vec::with_capacity(end.saturating_sub(start));
                for selected_index in start..end {
                    let Some(ordinal) = read_i64(ordinal_bytes, selected_index) else {
                        continue;
                    };
                    if ordinal < 0 {
                        continue;
                    }
                    let ordinal = ordinal as usize;
                    let Some(action_id) = row.legal_action_ids.get(ordinal).copied() else {
                        continue;
                    };
                    let prior = read_f32(prior_bytes, selected_index)
                        .unwrap_or(0.0)
                        .max(0.0);
                    row_priors.push((action_id, prior));
                }
                evaluations.push(RustEvaluation {
                    value,
                    priors: row_priors,
                });
            }
        } else {
            let mut prior_offset = 0usize;
            for (index, row) in encoded.iter().enumerate() {
                let value = read_f32(value_bytes, index).unwrap_or(0.0).clamp(-1.0, 1.0);
                let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
                for action_id in row.legal_action_ids.iter().copied() {
                    let prior = read_f32(prior_bytes, prior_offset).unwrap_or(0.0).max(0.0);
                    row_priors.push((action_id, prior));
                    prior_offset += 1;
                }
                evaluations.push(RustEvaluation {
                    value,
                    priors: row_priors,
                });
            }
        }
        return Ok(evaluations);
    }

    let values = output.get_item("values")?;
    let priors = output.get_item("priors")?;
    let mut evaluations = Vec::with_capacity(encoded.len());
    for (index, row) in encoded.iter().enumerate() {
        let value = values.get_item(index)?.extract::<f32>()?.clamp(-1.0, 1.0);
        let prior_row = priors.get_item(index)?;
        let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
        for (action_id, prior_item) in row
            .legal_action_ids
            .iter()
            .copied()
            .zip(prior_row.try_iter()?)
        {
            row_priors.push((action_id, prior_item?.extract::<f32>()?.max(0.0)));
        }
        evaluations.push(RustEvaluation {
            value,
            priors: row_priors,
        });
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
    let mut result_slots: Vec<Option<RustEvaluation>> = vec![None; states.len()];
    let mut unique_states: Vec<RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();
    let mut slot_to_unique: Vec<Option<usize>> = vec![None; states.len()];
    if let Some(stats) = stats {
        stats.borrow_mut().requested_states += states.len();
    }

    {
        let cached = cache.borrow();
        if let Some(stats) = stats {
            let mut stats = stats.borrow_mut();
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
        }
        for (index, state) in states.iter().enumerate() {
            let key = state_hash(state);
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
            unique_states.push(state.clone());
        }
    }

    if !unique_states.is_empty() {
        if let Some(stats) = stats {
            stats.borrow_mut().unique_states += unique_states.len();
        }
        let unique_evals =
            evaluate_model1_states(py, evaluator, &unique_states, stats, prior_candidate_limit)?;
        {
            let mut cached = cache.borrow_mut();
            let mut inserted = 0usize;
            for (offset, (key, evaluation)) in unique_keys
                .iter()
                .copied()
                .zip(unique_evals.iter())
                .enumerate()
            {
                if cached.len() >= cache_max_states {
                    if let Some(stats) = stats {
                        stats.borrow_mut().cache_insert_skipped += unique_keys.len() - offset;
                    }
                    break;
                }
                cached.insert(key, evaluation.clone());
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

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_i64(bytes: &[u8], index: usize) -> Option<i64> {
    let start = index.checked_mul(8)?;
    let chunk = bytes.get(start..start + 8)?;
    Some(i64::from_ne_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ]))
}
