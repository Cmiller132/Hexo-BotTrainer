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

use hexo_engine::{HexoState as RustHexoState, PackedCoord};
use hexo_utils::{hash_state, StateHash};

use crate::constants::*;
use crate::encoding::encode_model1_state;

#[derive(Clone, Debug)]
pub(crate) struct RustEvaluation {
    pub(crate) value: f32,
    pub(crate) priors: Vec<(PackedCoord, f32)>,
}

pub(crate) type SharedEvaluationCache = Rc<RefCell<HashMap<StateHash, RustEvaluation>>>;

pub(crate) fn new_shared_evaluation_cache() -> SharedEvaluationCache {
    Rc::new(RefCell::new(HashMap::new()))
}

pub(crate) fn state_hash(state: &RustHexoState) -> StateHash {
    hash_state(state)
}

fn evaluate_model1_states(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
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
    let flat_byte_len = legal_flat_indices.len() * std::mem::size_of::<i64>();
    let flat_bytes = unsafe {
        std::slice::from_raw_parts(legal_flat_indices.as_ptr() as *const u8, flat_byte_len)
    };
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

    let output = evaluator.call1((payload,))?;
    if let (Ok(values_obj), Ok(priors_obj)) = (
        output.get_item("values_bytes"),
        output.get_item("priors_bytes"),
    ) {
        let value_bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
        let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
        let mut evaluations = Vec::with_capacity(encoded.len());
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
        return Ok(evaluations);
    }

    let values = output.get_item("values")?;
    let priors = output.get_item("priors")?;
    let mut evaluations = Vec::with_capacity(encoded.len());
    for (index, row) in encoded.iter().enumerate() {
        let value = values.get_item(index)?.extract::<f32>()?.clamp(-1.0, 1.0);
        let prior_row = priors.get_item(index)?;
        let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
        for (action_id, prior_item) in row.legal_action_ids.iter().copied().zip(prior_row.try_iter()?) {
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
) -> PyResult<Vec<RustEvaluation>> {
    let mut result_slots: Vec<Option<RustEvaluation>> = vec![None; states.len()];
    let mut unique_states: Vec<RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();

    {
        let cached = cache.borrow();
        for (index, state) in states.iter().enumerate() {
            let key = state_hash(state);
            if let Some(cached_eval) = cached.get(&key) {
                result_slots[index] = Some(cached_eval.clone());
                continue;
            }
            if unique_index_by_key.contains_key(&key) {
                continue;
            }
            unique_index_by_key.insert(key, unique_states.len());
            unique_keys.push(key);
            unique_states.push(state.clone());
        }
    }

    if !unique_states.is_empty() {
        let unique_evals = evaluate_model1_states(py, evaluator, &unique_states)?;
        {
            let mut cached = cache.borrow_mut();
            for (key, evaluation) in unique_keys.into_iter().zip(unique_evals.into_iter()) {
                cached.insert(key, evaluation);
            }
        }
        let cached = cache.borrow();
        for (index, state) in states.iter().enumerate() {
            if result_slots[index].is_some() {
                continue;
            }
            let key = state_hash(state);
            if let Some(evaluation) = cached.get(&key) {
                result_slots[index] = Some(evaluation.clone());
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
