//! Evaluator payload ABI (spec §5.2) + cached batch evaluation.
//!
//! Request: one dict per flush; CSR flat-concat over support nodes; rows
//! pre-sorted by support size DESCENDING (stable by request index) so Python
//! grouping is contiguous slicing; the dedup slot-map restores caller order on
//! reply. Reply: dense_cnn's two-key contract byte-identical
//! (values_bytes f32 x B, priors_bytes f32 x sum(L_g) positional over each
//! row's legal prefix) plus the optional moves_left_bytes (f32 x B, decoded
//! decisions [0, 512]) when `request_moves_left` is set.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use half::f16;
use hexo_engine::{pack_coord, HexoState as RustHexoState, PackedCoord};
use hexo_utils::StateHash;

use crate::cache::{
    lock_cache, lock_stats, RustEvaluation, RustEvaluationRequest, SharedEvaluationCache,
    SharedEvaluationStats,
};
use crate::constants::NUM_FEATURES;
use crate::features::build_features;
use crate::support::build_support;

pub const ABI_VERSION: u32 = 1;
pub const NBR_SENTINEL: u16 = 0xFFFF;
/// Keep evaluator batches bounded (same intent as dense's chunking).
pub const EVAL_CHUNK_STATES: usize = 1024;

fn evaluate_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let encoding_started = Instant::now();
    // Featurize each row, then order rows by support size DESCENDING (stable
    // by request index). Rust keeps the per-row sorted legal action ids; they
    // never cross the boundary — priors return positionally over the prefix.
    struct Row {
        request_index: usize,
        legal_ids: Vec<PackedCoord>,
        coords_qr: Vec<i16>,
        nbr_local: Vec<u16>,
        feats: Vec<f16>,
        num_nodes: usize,
    }
    let mut rows: Vec<Row> = states
        .iter()
        .enumerate()
        .map(|(request_index, state)| {
            let sup = build_support(state);
            let feats32 = build_features(state, &sup);
            let mut feats = vec![f16::ZERO; feats32.len()];
            for (dst, src) in feats.iter_mut().zip(feats32.iter()) {
                *dst = f16::from_f32(*src);
            }
            let mut coords_qr = Vec::with_capacity(sup.num_nodes() * 2);
            for c in &sup.coords {
                coords_qr.push(c.q);
                coords_qr.push(c.r);
            }
            let mut nbr_local = Vec::with_capacity(sup.num_nodes() * 6);
            for row in &sup.nbr {
                for &j in row {
                    nbr_local.push(if j < 0 { NBR_SENTINEL } else { j as u16 });
                }
            }
            if sup.num_nodes() > NBR_SENTINEL as usize {
                return Err(PyValueError::new_err(format!(
                    "support of {} nodes exceeds the u16 neighbour wire limit",
                    sup.num_nodes()
                )));
            }
            let legal_ids: Vec<PackedCoord> = sup.coords[..sup.legal_count]
                .iter()
                .map(|&c| pack_coord(c))
                .collect();
            Ok(Row {
                request_index,
                legal_ids,
                coords_qr,
                nbr_local,
                feats,
                num_nodes: sup.num_nodes(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    rows.sort_by(|a, b| {
        b.num_nodes
            .cmp(&a.num_nodes)
            .then_with(|| a.request_index.cmp(&b.request_index))
    });

    let total_nodes: usize = rows.iter().map(|r| r.num_nodes).sum();
    let b = rows.len();
    let mut node_feats: Vec<f16> = Vec::with_capacity(total_nodes * NUM_FEATURES);
    let mut node_qr: Vec<i16> = Vec::with_capacity(total_nodes * 2);
    let mut nbr: Vec<u16> = Vec::with_capacity(total_nodes * 6);
    let mut node_row_offsets: Vec<i64> = Vec::with_capacity(b + 1);
    let mut legal_counts: Vec<i32> = Vec::with_capacity(b);
    node_row_offsets.push(0);
    for row in &rows {
        node_feats.extend_from_slice(&row.feats);
        node_qr.extend_from_slice(&row.coords_qr);
        nbr.extend_from_slice(&row.nbr_local);
        legal_counts.push(row.legal_ids.len() as i32);
        node_row_offsets.push(node_row_offsets.last().unwrap() + row.num_nodes as i64);
    }

    fn bytes_of<'py, T>(py: Python<'py>, data: &[T]) -> Bound<'py, PyBytes> {
        let len = std::mem::size_of_val(data);
        let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
        PyBytes::new(py, raw)
    }

    let payload = PyDict::new(py);
    payload.set_item("abi", ABI_VERSION)?;
    payload.set_item("shape", (b, total_nodes))?;
    payload.set_item("node_feats", bytes_of(py, &node_feats))?;
    payload.set_item("node_qr", bytes_of(py, &node_qr))?;
    payload.set_item("node_row_offsets", node_row_offsets.clone())?;
    payload.set_item("nbr", bytes_of(py, &nbr))?;
    payload.set_item("legal_counts", bytes_of(py, &legal_counts))?;
    payload.set_item("request_moves_left", request_moves_left)?;
    if let Some(stats) = stats {
        let mut stats = lock_stats(stats);
        stats.evaluator_chunks += 1;
        stats.encoded_states += b;
        stats.encoded_nodes += total_nodes;
        stats.max_chunk_states = stats.max_chunk_states.max(b);
        stats.input_bytes += node_feats.len() * 2 + node_qr.len() * 2 + nbr.len() * 2;
        stats.encoding_seconds += encoding_started.elapsed().as_secs_f64();
    }

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }

    let parse_started = Instant::now();
    let values_obj = output
        .get_item("values_bytes")
        .map_err(|_| PyValueError::new_err("hexfield evaluator output missing values_bytes"))?;
    let priors_obj = output
        .get_item("priors_bytes")
        .map_err(|_| PyValueError::new_err("hexfield evaluator output missing priors_bytes"))?;
    let value_bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
    let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
    require_exact_bytes("values_bytes", value_bytes.len(), b, 4)?;
    let expected_priors: usize = rows.iter().map(|r| r.legal_ids.len()).sum();
    require_exact_bytes("priors_bytes", prior_bytes.len(), expected_priors, 4)?;
    let moves_left_bytes: Option<Vec<u8>> = if request_moves_left {
        let obj = output.get_item("moves_left_bytes").map_err(|_| {
            PyValueError::new_err(
                "hexfield evaluator output missing moves_left_bytes (request_moves_left was set)",
            )
        })?;
        let bytes = obj.downcast::<PyBytes>()?.as_bytes().to_vec();
        require_exact_bytes("moves_left_bytes", bytes.len(), b, 4)?;
        Some(bytes)
    } else {
        None
    };
    if let Some(stats) = stats {
        let mut stats = lock_stats(stats);
        stats.value_bytes += value_bytes.len();
        stats.prior_bytes += prior_bytes.len();
    }

    // Parse per (sorted) row, then restore caller order.
    let mut by_request: Vec<Option<RustEvaluation>> = (0..states.len()).map(|_| None).collect();
    let mut offset = 0usize;
    for (sorted_index, row) in rows.iter().enumerate() {
        let value = read_value(value_bytes, sorted_index)?;
        let mut priors = Vec::with_capacity(row.legal_ids.len());
        for (k, &action_id) in row.legal_ids.iter().enumerate() {
            let prior = read_prior(prior_bytes, offset + k, sorted_index)?;
            priors.push((action_id, prior));
        }
        offset += row.legal_ids.len();
        finalize_priors(&mut priors, row.legal_ids.len(), sorted_index)?;
        let moves_left = match &moves_left_bytes {
            Some(bytes) => {
                let ml = read_f32_required("moves_left_bytes", bytes, sorted_index)?;
                if !ml.is_finite() || !(0.0..=512.0).contains(&ml) {
                    return Err(PyValueError::new_err(format!(
                        "moves_left_bytes row {sorted_index} must be in [0, 512], got {ml}"
                    )));
                }
                Some(ml)
            }
            None => None,
        };
        by_request[row.request_index] = Some(RustEvaluation {
            value,
            legal_action_count: row.legal_ids.len(),
            priors,
            moves_left,
        });
    }
    if let Some(stats) = stats {
        lock_stats(stats).parse_seconds += parse_started.elapsed().as_secs_f64();
    }
    Ok(by_request
        .into_iter()
        .map(|item| item.expect("every payload row parsed"))
        .collect())
}

fn evaluate_state_refs(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    if states.len() > EVAL_CHUNK_STATES {
        let mut evaluations = Vec::with_capacity(states.len());
        for chunk in states.chunks(EVAL_CHUNK_STATES) {
            evaluations.extend(evaluate_states_chunk(
                py,
                evaluator,
                chunk,
                request_moves_left,
                stats,
            )?);
        }
        return Ok(evaluations);
    }
    evaluate_states_chunk(py, evaluator, states, request_moves_left, stats)
}

/// Cache-checked, duplicate-coalescing batch evaluation preserving caller
/// order (port of dense's evaluate_model1_state_refs_cached).
pub fn evaluate_state_refs_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
    request_moves_left: bool,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let mut result_slots: Vec<Option<Arc<RustEvaluation>>> = vec![None; requests.len()];
    let mut unique_states: Vec<&RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();
    let mut slot_to_unique: Vec<Option<usize>> = vec![None; requests.len()];
    if let Some(stats) = stats {
        lock_stats(stats).requested_states += requests.len();
    }

    {
        let cached = lock_cache(cache);
        if let Some(stats) = stats {
            let mut stats = lock_stats(stats);
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
        }
        for (index, request) in requests.iter().enumerate() {
            let key = request.state_hash;
            if let Some(cached_eval) = cached.get(&key) {
                // A cached eval without moves_left cannot serve a request that
                // needs it; treat as a miss so the reply carries the field.
                if !request_moves_left || cached_eval.moves_left.is_some() {
                    result_slots[index] = Some(cached_eval);
                    if let Some(stats) = stats {
                        lock_stats(stats).cache_hits += 1;
                    }
                    continue;
                }
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

    if !unique_states.is_empty() {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_states.len();
        }
        let unique_evals =
            evaluate_state_refs(py, evaluator, &unique_states, request_moves_left, stats)?;
        let unique_evals: Vec<Arc<RustEvaluation>> = unique_evals
            .into_iter()
            .map(|mut eval| {
                eval.priors.shrink_to_fit();
                Arc::new(eval)
            })
            .collect();
        {
            let mut cached = lock_cache(cache);
            let mut inserted = 0usize;
            for (key, evaluation) in unique_keys.iter().copied().zip(unique_evals.iter()) {
                cached.insert_bounded(key, Arc::clone(evaluation), cache_max_states);
                inserted += 1;
            }
            if let Some(stats) = stats {
                let mut stats = lock_stats(stats);
                stats.cache_inserts += inserted;
                stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
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
        .map(|item| item.expect("every hexfield evaluation slot must be populated"))
        .collect())
}

/// Validate + descending-sort + normalize (dense finalize_model_priors port;
/// `legal_action_count == priors.len()` always holds here — the vocabulary IS
/// the legal set).
fn finalize_priors(
    priors: &mut Vec<(PackedCoord, f32)>,
    legal_action_count: usize,
    row_index: usize,
) -> PyResult<()> {
    if legal_action_count == 0 {
        if priors.is_empty() {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for terminal row {row_index}",
            priors.len()
        )));
    }
    if priors.is_empty() {
        return Err(PyValueError::new_err(format!(
            "evaluator returned no priors for non-terminal row {row_index}"
        )));
    }
    let mut seen = HashSet::with_capacity(priors.len());
    let mut total = 0.0f32;
    for (action_id, prior) in priors.iter().copied() {
        if !seen.insert(action_id) {
            return Err(PyValueError::new_err(format!(
                "duplicate action {action_id} in row {row_index}"
            )));
        }
        if !prior.is_finite() || prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "invalid prior {prior} for action {action_id} in row {row_index}"
            )));
        }
        total += prior;
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "zero total prior mass for row {row_index}"
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
    Ok(())
}

fn require_exact_bytes(
    name: &str,
    actual_bytes: usize,
    expected_items: usize,
    bytes_per_item: usize,
) -> PyResult<()> {
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

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_f32_required(name: &str, bytes: &[u8], index: usize) -> PyResult<f32> {
    read_f32(bytes, index)
        .ok_or_else(|| PyValueError::new_err(format!("{name} missing f32 at item index {index}")))
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
