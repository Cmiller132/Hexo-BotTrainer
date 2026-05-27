//! Python/PyTorch evaluator adapter for Hexformer MCTS.
//!
//! The search tree asks for batches of leaf states; this module serializes their
//! packed histories/legal actions, calls the Python inference callback, and
//! validates candidate priors before they enter the tree.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};

use hexo_engine::{pack_coord, HexoState as RustHexoState, PackedCoord};

#[derive(Clone, Debug)]
pub(crate) struct RustEvaluation {
    pub(crate) value: f32,
    pub(crate) priors: Vec<(PackedCoord, f32)>,
}

fn evaluate_states(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
) -> PyResult<Vec<RustEvaluation>> {
    let history_rows = PyList::empty(py);
    let legal_rows = PyList::empty(py);
    for state in states {
        let history: Vec<PackedCoord> = state
            .placement_history()
            .iter()
            .map(|record| pack_coord(record.coord))
            .collect();
        let mut legal = Vec::new();
        state.write_legal_action_ids(&mut legal);
        history_rows.append(PyTuple::new(py, history)?)?;
        legal_rows.append(PyTuple::new(py, legal)?)?;
    }

    let payload = PyDict::new(py);
    payload.set_item("history_rows", history_rows)?;
    payload.set_item("legal_action_ids", legal_rows)?;

    let output = evaluator.call1((payload,))?;
    parse_evaluation_output(&output, states)
}

pub(crate) fn evaluate_states_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
    cache: &mut HashMap<Vec<PackedCoord>, RustEvaluation>,
) -> PyResult<Vec<RustEvaluation>> {
    let mut result_slots: Vec<Option<RustEvaluation>> = vec![None; states.len()];
    let mut unique_states: Vec<RustHexoState> = Vec::new();
    let mut unique_keys: Vec<Vec<PackedCoord>> = Vec::new();
    let mut unique_index_by_key: HashMap<Vec<PackedCoord>, usize> = HashMap::new();

    for (index, state) in states.iter().enumerate() {
        let key = state_cache_key(state);
        if let Some(cached) = cache.get(&key) {
            result_slots[index] = Some(cached.clone());
            continue;
        }
        if unique_index_by_key.contains_key(&key) {
            continue;
        }
        unique_index_by_key.insert(key.clone(), unique_states.len());
        unique_keys.push(key);
        unique_states.push(state.clone());
    }

    if !unique_states.is_empty() {
        let unique_evals = evaluate_states(py, evaluator, &unique_states)?;
        for (key, evaluation) in unique_keys.into_iter().zip(unique_evals.into_iter()) {
            cache.insert(key, evaluation);
        }
    }

    for (index, state) in states.iter().enumerate() {
        if result_slots[index].is_some() {
            continue;
        }
        let key = state_cache_key(state);
        if let Some(evaluation) = cache.get(&key) {
            result_slots[index] = Some(evaluation.clone());
        }
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexformer evaluation slot must be populated"))
        .collect())
}

fn parse_evaluation_output(
    output: &Bound<'_, PyAny>,
    states: &[RustHexoState],
) -> PyResult<Vec<RustEvaluation>> {
    let values = read_values(output, states.len())?;
    let candidate_rows = read_candidate_rows(output)?;
    let prior_rows = read_prior_rows(output, &candidate_rows)?;
    if candidate_rows.len() != states.len() || prior_rows.len() != states.len() {
        return Err(PyValueError::new_err(format!(
            "hexformer evaluator returned {} candidate rows and {} prior rows for {} states",
            candidate_rows.len(),
            prior_rows.len(),
            states.len()
        )));
    }
    for (index, (candidates, priors)) in candidate_rows.iter().zip(prior_rows.iter()).enumerate() {
        if candidates.len() != priors.len() {
            return Err(PyValueError::new_err(format!(
                "hexformer evaluator row {index} returned {} candidates but {} priors",
                candidates.len(),
                priors.len()
            )));
        }
    }

    let mut evaluations = Vec::with_capacity(states.len());
    for (index, state) in states.iter().enumerate() {
        let mut legal = Vec::new();
        state.write_legal_action_ids(&mut legal);
        let legal_set: HashSet<_> = legal.into_iter().collect();
        let mut seen = HashSet::new();
        let mut priors = Vec::new();
        for (action_id, prior) in candidate_rows[index]
            .iter()
            .copied()
            .zip(prior_rows[index].iter().copied())
        {
            if !legal_set.contains(&action_id) || !seen.insert(action_id) {
                continue;
            }
            priors.push((action_id, clean_prior(prior)));
        }
        evaluations.push(RustEvaluation {
            value: values[index].clamp(-1.0, 1.0),
            priors,
        });
    }
    Ok(evaluations)
}

fn read_values(output: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f32>> {
    if let Ok(values_obj) = output.get_item("values_bytes") {
        let bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
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
            "hexformer evaluator returned {} values for {} states",
            values.len(),
            expected
        )));
    }
    Ok(values)
}

fn read_candidate_rows(output: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<PackedCoord>>> {
    if let (Ok(bytes_obj), Ok(offsets_obj)) = (
        output.get_item("candidate_action_ids_bytes"),
        output.get_item("candidate_row_offsets"),
    ) {
        let bytes = bytes_obj.downcast::<PyBytes>()?.as_bytes();
        let offsets = offsets_obj
            .try_iter()?
            .map(|item| item?.extract::<usize>())
            .collect::<PyResult<Vec<_>>>()?;
        let mut rows = Vec::with_capacity(offsets.len().saturating_sub(1));
        for window in offsets.windows(2) {
            let start = window[0];
            let end = window[1];
            let mut row = Vec::with_capacity(end.saturating_sub(start));
            for index in start..end {
                row.push(read_packed_coord(bytes, index).unwrap_or(0));
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    let rows_obj = output.get_item("candidate_action_ids")?;
    let mut rows = Vec::new();
    for row in rows_obj.try_iter()? {
        let row = row?;
        let mut ids = Vec::new();
        for item in row.try_iter()? {
            ids.push(item?.extract::<PackedCoord>()?);
        }
        rows.push(ids);
    }
    Ok(rows)
}

fn read_prior_rows(
    output: &Bound<'_, PyAny>,
    candidate_rows: &[Vec<PackedCoord>],
) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(priors_obj) = output.get_item("priors_bytes") {
        let bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
        let mut offset = 0usize;
        let mut rows = Vec::with_capacity(candidate_rows.len());
        for candidates in candidate_rows {
            let mut row = Vec::with_capacity(candidates.len());
            for _ in candidates {
                row.push(read_f32(bytes, offset).unwrap_or(0.0));
                offset += 1;
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    let rows_obj = output.get_item("priors")?;
    let mut rows = Vec::new();
    for row in rows_obj.try_iter()? {
        let row = row?;
        let mut priors = Vec::new();
        for item in row.try_iter()? {
            priors.push(item?.extract::<f32>()?);
        }
        rows.push(priors);
    }
    Ok(rows)
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_packed_coord(bytes: &[u8], index: usize) -> Option<PackedCoord> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn state_cache_key(state: &RustHexoState) -> Vec<PackedCoord> {
    state
        .placement_history()
        .iter()
        .map(|record| pack_coord(record.coord))
        .collect()
}

fn clean_prior(prior: f32) -> f32 {
    if prior.is_finite() {
        prior.max(0.0)
    } else {
        0.0
    }
}
