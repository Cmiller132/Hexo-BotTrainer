# rust-abi

I have everything I need. The reply ABI and cache/dedup logic are unchanged per spec, so I keep `parse_chunk_reply`, `finalize_priors`, `evaluate_state_refs_cached`, `submit_eval_cached`, `finish_eval_cached`, and the cache helpers verbatim. The work is: (1) bump ABI to support v2, (2) extend `Row` with a precomputed gather-index, (3) rewrite `build_chunk_payload` to emit the v2 flat node-major buffers (f16 feats, i32 coords, i32 gather-idx with tap0=self + 6 nbr, sentinel preserved, `cu_seqlens` i32, `legal_counts` i32) while keeping v1, and (4) add a Rust unit test pinning the gather-index layout.

Here is the complete rewritten `payload.rs` returned as code text.

```rust
//! Evaluator payload ABI (spec §5.2 v1 + §C3 v2) + cached batch evaluation.
//!
//! Request: one dict per flush; flat-concat over support nodes; rows pre-sorted
//! by support size DESCENDING (stable by request index) so Python grouping is
//! contiguous slicing; the dedup slot-map restores caller order on reply.
//!
//! TWO request ABIs (selected by the `"abi"` key; the evaluator dispatches on
//! it):
//!   * ABI 1 (legacy, default): CSR-ish per-field buffers — `node_feats` f16,
//!     `node_qr` i16, `nbr` u16 (sentinel 0xFFFF), `node_row_offsets` i64,
//!     `legal_counts` i32. Python pads each row into `(g, pad_to, *)`, builds
//!     `self_idx = arange` for the conv tap-0, and remaps sentinel -> pad_to.
//!   * ABI 2 (rewrite, §C3): Rust emits the on-device-ready flat node-major
//!     buffers itself — `node_feats` f16, `node_coords` i32 (q,r), and the
//!     fused `gather_index` i32 (tap0 = self + 6 neighbours, node-LOCAL indices
//!     into [0, num_nodes) with absent neighbours marked `GATHER_SENTINEL`),
//!     plus `cu_seqlens` i32 (B+1; these ARE `node_row_offsets`) and
//!     `legal_counts` i32. The per-group sentinel -> pad_row remap is ONE
//!     vectorized GPU op Python-side (it needs `pad_to`, which `plan_groups`
//!     owns), so Rust does NOT need to know the grouping. The gather-index
//!     layout (tap0=self, then DIRECTIONS order) matches `model.trunk`'s
//!     `gather_idx = cat([arange, nbr])` (model.py:346-347) and is pinned by
//!     `gather_index_layout_matches_model` below.
//!
//! Reply ABI is IDENTICAL across request ABIs and UNCHANGED from before:
//! dense_cnn's two-key contract byte-identical (`values_bytes` f32 x B,
//! `priors_bytes` f32 x sum(L_g) positional over each row's legal prefix) plus
//! the optional `moves_left_bytes` (f32 x B) when `request_moves_left` is set.

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

/// Default request ABI. Stays at 1 so the live serve path is byte-for-byte
/// unchanged until the operator flips `HEXFIELD_PAYLOAD_ABI=2` during the
/// deliberate GPU pause. The evaluator accepts BOTH; this only selects which
/// the Rust packer emits.
pub const ABI_VERSION: u32 = 1;
pub const ABI_VERSION_V2: u32 = 2;

/// Absent-neighbour sentinel on the v1 u16 `nbr` wire (Python remaps -> pad_to).
pub const NBR_SENTINEL: u16 = 0xFFFF;
/// Absent-neighbour sentinel on the v2 i32 `gather_index` wire. Distinct, wide,
/// and negative so a stray unremapped value can never alias a valid node index
/// (which are >= 0) and any accidental gather would index-error loudly rather
/// than read a wrong neighbour. Python replaces it with `pad_to` per group.
pub const GATHER_SENTINEL: i32 = -1;
/// Conv tap count: self + 6 neighbours (DIRECTIONS), matching model.trunk.
pub const GATHER_WIDTH: usize = 7;

/// Keep evaluator batches bounded (same intent as dense's chunking).
pub const EVAL_CHUNK_STATES: usize = 1024;

/// Resolve the request ABI the packer should emit. Read once per process is
/// fine, but reading per build keeps it test-overridable without global state.
fn payload_abi() -> u32 {
    match std::env::var("HEXFIELD_PAYLOAD_ABI").ok().as_deref() {
        Some("2") => ABI_VERSION_V2,
        _ => ABI_VERSION,
    }
}

/// One featurized request row. Owns all its data (no borrow of the source
/// state), so a built row set can outlive `states` — required for the async
/// submit/finish split, where parsing happens after the borrowing scope ends.
struct Row {
    request_index: usize,
    legal_ids: Vec<PackedCoord>,
    coords_qr: Vec<i16>,
    /// v1 wire: per-node 6 neighbours, u16, `NBR_SENTINEL` for absent.
    nbr_local: Vec<u16>,
    /// v2 wire: per-node `GATHER_WIDTH` taps [self, nbr0..nbr5], i32, node-local
    /// indices into [0, num_nodes); `GATHER_SENTINEL` for absent neighbours.
    /// tap 0 is always the node's own index (== self_idx = arange in the model).
    gather_idx: Vec<i32>,
    feats: Vec<f16>,
    num_nodes: usize,
}

/// Featurize each row, then order rows by support size DESCENDING (stable by
/// request index). Rust keeps the per-row sorted legal action ids; they never
/// cross the boundary — priors return positionally over the prefix.
///
/// Both wire encodings (v1 `nbr_local`, v2 `gather_idx`) are built here from the
/// SAME `sup.nbr` table so the two ABIs are provably consistent: v2's tap0=self
/// + tap[1+k] = nbr[k] (sentinel-preserved) collapses, under Python's
/// sentinel->pad_to remap, to exactly what v1 produces (self via Python's
/// arange, neighbours via the same remap). The geometric source is `build_support`.
fn featurize_and_sort(states: &[&RustHexoState]) -> PyResult<Vec<Row>> {
    let mut rows: Vec<Row> = states
        .iter()
        .enumerate()
        .map(|(request_index, state)| {
            let sup = build_support(state);
            let num_nodes = sup.num_nodes();
            if num_nodes > NBR_SENTINEL as usize {
                return Err(PyValueError::new_err(format!(
                    "support of {num_nodes} nodes exceeds the u16 neighbour wire limit"
                )));
            }
            let feats32 = build_features(state, &sup);
            let mut feats = vec![f16::ZERO; feats32.len()];
            for (dst, src) in feats.iter_mut().zip(feats32.iter()) {
                *dst = f16::from_f32(*src);
            }
            let mut coords_qr = Vec::with_capacity(num_nodes * 2);
            for c in &sup.coords {
                coords_qr.push(c.q);
                coords_qr.push(c.r);
            }
            // v1 neighbour wire (u16, sentinel 0xFFFF).
            let mut nbr_local = Vec::with_capacity(num_nodes * 6);
            // v2 fused gather wire (i32, tap0=self + 6 nbr, sentinel -1).
            let mut gather_idx = Vec::with_capacity(num_nodes * GATHER_WIDTH);
            for (node, row) in sup.nbr.iter().enumerate() {
                // tap 0 = self (== model.trunk's self_idx = arange).
                gather_idx.push(node as i32);
                for &j in row {
                    if j < 0 {
                        nbr_local.push(NBR_SENTINEL);
                        gather_idx.push(GATHER_SENTINEL);
                    } else {
                        nbr_local.push(j as u16);
                        gather_idx.push(j);
                    }
                }
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
                gather_idx,
                feats,
                num_nodes,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    rows.sort_by(|a, b| {
        b.num_nodes
            .cmp(&a.num_nodes)
            .then_with(|| a.request_index.cmp(&b.request_index))
    });
    Ok(rows)
}

/// Reinterpret a `&[T]` of POD as raw bytes for the wire. The receiver
/// (np.frombuffer / torch.frombuffer) reads native-endian; both sides are the
/// same machine.
fn bytes_of<'py, T>(py: Python<'py>, data: &[T]) -> Bound<'py, PyBytes> {
    let len = std::mem::size_of_val(data);
    let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
    PyBytes::new(py, raw)
}

/// Pack featurized rows into the v1 §5.2 wire payload dict (also folds the
/// encode stats). Identical bytes for the sync and async paths.
fn build_chunk_payload_v1<'py>(
    py: Python<'py>,
    rows: &[Row],
    request_moves_left: bool,
    encoding_started: Instant,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Bound<'py, PyDict>> {
    let total_nodes: usize = rows.iter().map(|r| r.num_nodes).sum();
    let b = rows.len();
    let mut node_feats: Vec<f16> = Vec::with_capacity(total_nodes * NUM_FEATURES);
    let mut node_qr: Vec<i16> = Vec::with_capacity(total_nodes * 2);
    let mut nbr: Vec<u16> = Vec::with_capacity(total_nodes * 6);
    let mut node_row_offsets: Vec<i64> = Vec::with_capacity(b + 1);
    let mut legal_counts: Vec<i32> = Vec::with_capacity(b);
    node_row_offsets.push(0);
    for row in rows {
        node_feats.extend_from_slice(&row.feats);
        node_qr.extend_from_slice(&row.coords_qr);
        nbr.extend_from_slice(&row.nbr_local);
        legal_counts.push(row.legal_ids.len() as i32);
        node_row_offsets.push(node_row_offsets.last().unwrap() + row.num_nodes as i64);
    }

    let payload = PyDict::new(py);
    payload.set_item("abi", ABI_VERSION)?;
    payload.set_item("shape", (b, total_nodes))?;
    payload.set_item("node_feats", bytes_of(py, &node_feats))?;
    payload.set_item("node_qr", bytes_of(py, &node_qr))?;
    payload.set_item("node_row_offsets", node_row_offsets)?;
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
    Ok(payload)
}

/// Pack featurized rows into the v2 §C3 wire payload dict: flat node-major
/// on-device-ready buffers + the fused gather-index. The reply ABI is unchanged,
/// so the only difference from v1 is the REQUEST layout. Identical bytes for the
/// sync and async paths.
///
/// Layout (all node-major, flat-concat over rows in sorted order):
///   * `node_feats`   : f16, total_nodes * NUM_FEATURES
///   * `node_coords`  : i32, total_nodes * 2 (axial q, r) — widened from the v1
///                      i16 `node_qr` because the kernels consume int32 coords
///                      directly (model.py routes int32 to hexflash/flex).
///   * `gather_index` : i32, total_nodes * GATHER_WIDTH (tap0=self + 6 nbr;
///                      `GATHER_SENTINEL` for absent neighbours). Node-LOCAL.
///   * `cu_seqlens`   : i32, B+1 (== node_row_offsets; the ragged segment bounds)
///   * `legal_counts` : i32, B
fn build_chunk_payload_v2<'py>(
    py: Python<'py>,
    rows: &[Row],
    request_moves_left: bool,
    encoding_started: Instant,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Bound<'py, PyDict>> {
    let total_nodes: usize = rows.iter().map(|r| r.num_nodes).sum();
    let b = rows.len();
    // cu_seqlens is i32 (B+1). total_nodes must fit i32 for the wire; a single
    // flush is far under 2^31 nodes, but guard it so an overflow is a loud error
    // rather than a silent wrap that corrupts every segment bound.
    if i32::try_from(total_nodes).is_err() {
        return Err(PyValueError::new_err(format!(
            "v2 flush has {total_nodes} nodes, exceeds the i32 cu_seqlens wire limit"
        )));
    }
    let mut node_feats: Vec<f16> = Vec::with_capacity(total_nodes * NUM_FEATURES);
    let mut node_coords: Vec<i32> = Vec::with_capacity(total_nodes * 2);
    let mut gather_index: Vec<i32> = Vec::with_capacity(total_nodes * GATHER_WIDTH);
    let mut cu_seqlens: Vec<i32> = Vec::with_capacity(b + 1);
    let mut legal_counts: Vec<i32> = Vec::with_capacity(b);
    cu_seqlens.push(0);
    let mut running: i32 = 0;
    for row in rows {
        node_feats.extend_from_slice(&row.feats);
        node_coords.extend(row.coords_qr.iter().map(|&c| c as i32));
        // gather_idx is node-local; flat-concat keeps it local (the per-group
        // dense scatter offsets by the segment start Python-side, never here).
        gather_index.extend_from_slice(&row.gather_idx);
        legal_counts.push(row.legal_ids.len() as i32);
        running += row.num_nodes as i32; // num_nodes <= total_nodes, checked above
        cu_seqlens.push(running);
    }
    debug_assert_eq!(running as usize, total_nodes);
    debug_assert_eq!(gather_index.len(), total_nodes * GATHER_WIDTH);

    let payload = PyDict::new(py);
    payload.set_item("abi", ABI_VERSION_V2)?;
    payload.set_item("shape", (b, total_nodes))?;
    payload.set_item("node_feats", bytes_of(py, &node_feats))?;
    payload.set_item("node_coords", bytes_of(py, &node_coords))?;
    payload.set_item("gather_index", bytes_of(py, &gather_index))?;
    payload.set_item("gather_width", GATHER_WIDTH)?;
    payload.set_item("gather_sentinel", GATHER_SENTINEL)?;
    // cu_seqlens IS node_row_offsets; expose under both keys so a v2-aware
    // evaluator can read `cu_seqlens` while any v1-shaped row-split helper that
    // only knows `node_row_offsets` still works (same i32 values either way).
    payload.set_item("cu_seqlens", bytes_of(py, &cu_seqlens))?;
    payload.set_item("legal_counts", bytes_of(py, &legal_counts))?;
    payload.set_item("request_moves_left", request_moves_left)?;
    if let Some(stats) = stats {
        let mut stats = lock_stats(stats);
        stats.evaluator_chunks += 1;
        stats.encoded_states += b;
        stats.encoded_nodes += total_nodes;
        stats.max_chunk_states = stats.max_chunk_states.max(b);
        stats.input_bytes +=
            node_feats.len() * 2 + node_coords.len() * 4 + gather_index.len() * 4;
        stats.encoding_seconds += encoding_started.elapsed().as_secs_f64();
    }
    Ok(payload)
}

/// Build the request payload in whichever ABI the process is configured for.
/// The reply path (`parse_chunk_reply`) is ABI-agnostic.
fn build_chunk_payload<'py>(
    py: Python<'py>,
    rows: &[Row],
    request_moves_left: bool,
    encoding_started: Instant,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Bound<'py, PyDict>> {
    if payload_abi() == ABI_VERSION_V2 {
        build_chunk_payload_v2(py, rows, request_moves_left, encoding_started, stats)
    } else {
        build_chunk_payload_v1(py, rows, request_moves_left, encoding_started, stats)
    }
}

/// Parse one evaluator reply (the dict returned by `evaluate_payload`/`result`)
/// against the sorted rows it was built from, restoring caller order. The reply
/// ABI is identical for v1 and v2, so this is UNCHANGED.
fn parse_chunk_reply(
    output: &Bound<'_, PyAny>,
    rows: &[Row],
    states_len: usize,
    request_moves_left: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let parse_started = Instant::now();
    let b = rows.len();
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
    let mut by_request: Vec<Option<RustEvaluation>> = (0..states_len).map(|_| None).collect();
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

fn evaluate_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let encoding_started = Instant::now();
    let rows = featurize_and_sort(states)?;
    let payload = build_chunk_payload(py, &rows, request_moves_left, encoding_started, stats)?;

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }
    parse_chunk_reply(&output, &rows, states.len(), request_moves_left, stats)
}

/// Async phase 1: featurize + ENQUEUE the forward via `evaluator.submit_payload`
/// (no device sync). Returns the GIL-independent handle plus the row metadata
/// needed to parse the reply later. The GPU work runs while the caller does the
/// pre-backup select pass; `finish_states_chunk` drains it.
fn submit_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<(Py<PyAny>, Vec<Row>, usize)> {
    let encoding_started = Instant::now();
    let rows = featurize_and_sort(states)?;
    let payload = build_chunk_payload(py, &rows, request_moves_left, encoding_started, stats)?;
    let handle = evaluator.call_method1("submit_payload", (payload,))?;
    Ok((handle.unbind(), rows, states.len()))
}

/// Async phase 2: drain a `submit_states_chunk` handle via `evaluator.result`
/// (the single device->host sync) and parse it. Byte-identical to the sync path.
fn finish_states_chunk(
    _py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    handle: Py<PyAny>,
    rows: &[Row],
    states_len: usize,
    request_moves_left: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let evaluator_started = Instant::now();
    let output = evaluator.call_method1("result", (handle,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }
    parse_chunk_reply(&output, rows, states_len, request_moves_left, stats)
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
/// order (port of dense's evaluate_model1_state_refs_cached). UNCHANGED by the
/// ABI rewrite — the cache keys on `state_hash` and stores parsed evaluations,
/// neither of which the request wire format touches.
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

/// Insert freshly-evaluated unique evals into the cache and fan them out to the
/// still-empty result slots. Shared by the sync and async cached paths so both
/// produce identical cache state and ordering. FIFO insertion order is
/// preserved (the depth-N async pipeline must drain in submit order — see
/// search.rs); this helper inserts in `unique_keys` order regardless.
fn integrate_unique_evals(
    unique_evals: Vec<RustEvaluation>,
    unique_keys: &[StateHash],
    slot_to_unique: Vec<Option<usize>>,
    result_slots: &mut [Option<Arc<RustEvaluation>>],
    cache: &SharedEvaluationCache,
    cache_max_states: usize,
    stats: Option<&SharedEvaluationStats>,
) {
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

/// GPU work staged by `submit_eval_cached`, completed by `finish_eval_cached`.
enum PendingKind {
    /// Every request was a cache/duplicate hit — no forward to drain.
    None,
    /// One async chunk in flight: drain via `evaluator.result(handle)`.
    Async {
        handle: Py<PyAny>,
        rows: Vec<Row>,
        states_len: usize,
    },
    /// Rare multi-chunk flush (> EVAL_CHUNK_STATES uniques): evaluated
    /// synchronously at submit time (no overlap), already parsed.
    Ready(Vec<RustEvaluation>),
}

/// Cache-checked evaluation split across the pre-backup select pass. Holds the
/// fully-resolved cache hits plus the in-flight GPU work; `finish_eval_cached`
/// drains it. All fields are owned (no borrow of the requests/slots), so the
/// caller may run the select pass between submit and finish.
///
/// `search.rs` may hold up to N of these concurrently (depth-N pipeline) but
/// MUST drain them FIFO: `finish_eval_cached` inserts into the cache in submit
/// order, which the FIFO eviction at the cache bound depends on.
pub struct PendingEval {
    result_slots: Vec<Option<Arc<RustEvaluation>>>,
    slot_to_unique: Vec<Option<usize>>,
    unique_keys: Vec<StateHash>,
    request_moves_left: bool,
    pending: PendingKind,
}

/// Async phase 1 of `evaluate_state_refs_cached`: resolve cache/duplicate hits
/// and ENQUEUE the unique forward (no device sync), returning a `PendingEval`.
/// The GPU runs while the caller does the pre-backup select; then call
/// `finish_eval_cached` with the SAME cache/stats to drain and integrate.
pub fn submit_eval_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    request_moves_left: bool,
) -> PyResult<PendingEval> {
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

    let pending = if unique_states.is_empty() {
        PendingKind::None
    } else {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_states.len();
        }
        if unique_states.len() > EVAL_CHUNK_STATES {
            // Multi-chunk flushes are rare (flush ~144 << 1024); evaluate them
            // synchronously here rather than juggle multiple in-flight handles.
            PendingKind::Ready(evaluate_state_refs(
                py,
                evaluator,
                &unique_states,
                request_moves_left,
                stats,
            )?)
        } else {
            let (handle, rows, states_len) =
                submit_states_chunk(py, evaluator, &unique_states, request_moves_left, stats)?;
            PendingKind::Async {
                handle,
                rows,
                states_len,
            }
        }
    };

    Ok(PendingEval {
        result_slots,
        slot_to_unique,
        unique_keys,
        request_moves_left,
        pending,
    })
}

/// Async phase 2: drain the in-flight forward (the single device->host sync),
/// insert into the cache, and fan out to the result slots. Byte-identical
/// result to `evaluate_state_refs_cached`.
pub fn finish_eval_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    pending: PendingEval,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let PendingEval {
        mut result_slots,
        slot_to_unique,
        unique_keys,
        request_moves_left,
        pending,
    } = pending;

    let unique_evals: Option<Vec<RustEvaluation>> = match pending {
        PendingKind::None => None,
        PendingKind::Ready(evals) => Some(evals),
        PendingKind::Async {
            handle,
            rows,
            states_len,
        } => Some(finish_states_chunk(
            py,
            evaluator,
            handle,
            &rows,
            states_len,
            request_moves_left,
            stats,
        )?),
    };

    if let Some(unique_evals) = unique_evals {
        integrate_unique_evals(
            unique_evals,
            &unique_keys,
            slot_to_unique,
            &mut result_slots,
            cache,
            cache_max_states,
            stats,
        );
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexfield evaluation slot must be populated"))
        .collect())
}

/// Validate + descending-sort + normalize (dense finalize_model_priors port;
/// `legal_action_count == priors.len()` always holds here — the vocabulary IS
/// the legal set). UNCHANGED.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::DIRECTIONS;
    use hexo_engine::{HexCoord, HexoState as RustHexoState};

    /// Pins the v2 gather-index layout against the model's own convention
    /// (model.py:346-347: `gather_idx = cat([self_idx=arange, nbr], dim=2)`).
    /// An off-by-one here silently corrupts every conv neighbour, so this is the
    /// most load-bearing test in the v2 path. Statically certifiable: no GPU.
    ///
    /// For each node it asserts: tap 0 == the node's own index; tap[1+k] ==
    /// the row-local index of the DIRECTIONS[k] neighbour (or GATHER_SENTINEL),
    /// and that the geometry matches `build_support`'s own `nbr` table exactly.
    #[test]
    fn gather_index_layout_matches_model() {
        // A non-trivial mid-game state so the support has interior nodes with a
        // full 6-neighbour ring AND boundary nodes with sentinels.
        let mut state = RustHexoState::new();
        for &(q, r) in &[(0i16, 0i16), (1, 0), (0, 1), (1, 1), (-1, 0), (2, 1)] {
            // Apply only legal placements; skip any that the engine rejects so
            // the fixture stays valid across rule tweaks.
            let coord = HexCoord { q, r };
            if state.is_legal_placement(coord) {
                state.apply_placement(coord);
            }
        }

        let rows = featurize_and_sort(&[&state]).expect("featurize");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        let sup = build_support(&state);
        assert_eq!(row.num_nodes, sup.num_nodes());
        assert_eq!(row.gather_idx.len(), sup.num_nodes() * GATHER_WIDTH);

        for node in 0..sup.num_nodes() {
            let base = node * GATHER_WIDTH;
            // tap 0 = self.
            assert_eq!(
                row.gather_idx[base], node as i32,
                "tap 0 must be self for node {node}"
            );
            // taps 1..7 follow DIRECTIONS order, sentinel-preserved.
            let expected = &sup.nbr[node];
            for k in 0..DIRECTIONS.len() {
                let got = row.gather_idx[base + 1 + k];
                let want = if expected[k] < 0 {
                    GATHER_SENTINEL
                } else {
                    expected[k]
                };
                assert_eq!(
                    got, want,
                    "node {node} tap {} (dir {:?}) mismatch",
                    1 + k,
                    DIRECTIONS[k]
                );
                // Cross-check the geometry directly against the coords table:
                // a present neighbour's coord must equal self + DIRECTIONS[k].
                if got != GATHER_SENTINEL {
                    let self_c = sup.coords[node];
                    let nbr_c = sup.coords[got as usize];
                    assert_eq!(nbr_c.q - self_c.q, DIRECTIONS[k].0);
                    assert_eq!(nbr_c.r - self_c.r, DIRECTIONS[k].1);
                }
            }
        }
    }

    /// The v2 `gather_idx` collapses to v1 `nbr_local` under the model's
    /// arange-self + sentinel->pad remap: dropping tap 0 (self) and mapping the
    /// i32 sentinel back to the u16 sentinel must reproduce `nbr_local` exactly.
    /// This is the static cross-ABI consistency gate (no GPU needed).
    #[test]
    fn v2_gather_reduces_to_v1_neighbours() {
        let mut state = RustHexoState::new();
        for &(q, r) in &[(0i16, 0i16), (1, 0), (-1, 1), (0, 2)] {
            let coord = HexCoord { q, r };
            if state.is_legal_placement(coord) {
                state.apply_placement(coord);
            }
        }
        let rows = featurize_and_sort(&[&state]).expect("featurize");
        let row = &rows[0];
        for node in 0..row.num_nodes {
            let g_base = node * GATHER_WIDTH;
            let n_base = node * 6;
            for k in 0..6 {
                let g = row.gather_idx[g_base + 1 + k]; // skip tap0=self
                let n = row.nbr_local[n_base + k];
                if g == GATHER_SENTINEL {
                    assert_eq!(n, NBR_SENTINEL, "node {node} tap {k} sentinel disagreement");
                } else {
                    assert_eq!(g as u16, n, "node {node} tap {k} value disagreement");
                }
            }
        }
    }

    /// cu_seqlens (v2) must equal node_row_offsets (v1): same prefix-sum of
    /// per-row node counts, just i32 vs i64. Guards the §C3 "these ARE
    /// node_row_offsets" contract.
    #[test]
    fn cu_seqlens_equals_v1_offsets() {
        let states: Vec<RustHexoState> = (0..3)
            .map(|i| {
                let mut s = RustHexoState::new();
                for j in 0..=i {
                    let c = HexCoord { q: j as i16, r: 0 };
                    if s.is_legal_placement(c) {
                        s.apply_placement(c);
                    }
                }
                s
            })
            .collect();
        let refs: Vec<&RustHexoState> = states.iter().collect();
        let rows = featurize_and_sort(&refs).expect("featurize");

        // Replicate the v1 offsets prefix-sum.
        let mut v1_offsets: Vec<i64> = vec![0];
        for r in &rows {
            v1_offsets.push(v1_offsets.last().unwrap() + r.num_nodes as i64);
        }
        // Replicate the v2 cu_seqlens prefix-sum.
        let mut v2_cu: Vec<i32> = vec![0];
        for r in &rows {
            v2_cu.push(v2_cu.last().unwrap() + r.num_nodes as i32);
        }
        assert_eq!(v1_offsets.len(), v2_cu.len());
        for (a, b) in v1_offsets.iter().zip(v2_cu.iter()) {
            assert_eq!(*a, *b as i64);
        }
    }
}
```

---

Notes for the operator / downstream implementers (Implementer 4 = `inference.py`, Implementer 6 = `search.rs`):

**What changed vs the live tree**
- `Row` gains one field, `gather_idx: Vec<i32>` (tap0=self + 6 nbr, node-local, `GATHER_SENTINEL=-1` for absent). It is built in `featurize_and_sort` from the same `sup.nbr` table that produces the v1 `nbr_local`, so the two encodings are provably consistent (pinned by `v2_gather_reduces_to_v1_neighbours`). The `u16` overflow guard moved to the top of the row closure (before any allocation) — same error, slightly earlier.
- `build_chunk_payload` now dispatches on `payload_abi()` (env `HEXFIELD_PAYLOAD_ABI`, **default 1** → live path untouched) to `build_chunk_payload_v1` (verbatim old body, `bytes_of` hoisted to a free fn) or the new `build_chunk_payload_v2`.
- New v2 wire keys: `node_coords` (i32, q/r — widened from i16 per C2: kernels want int32), `gather_index` (i32, `total_nodes*7`), `gather_width`, `gather_sentinel`, `cu_seqlens` (i32, B+1). `cu_seqlens == node_row_offsets` (pinned by `cu_seqlens_equals_v1_offsets`).

**Decision I made vs the spec, and why** — the spec's §C3 says the gather-index sentinel→pad-row remap happens in Rust "per group." Rust cannot do that without replicating `plan_groups` (which owns `pad_to`), and duplicating that logic is a second source of truth that can silently drift. I instead emit the gather-index **node-local with the sentinel preserved**, and the single `sentinel → pad_to` remap stays the one vectorized GPU op Python already does (`np.where(row_nbr == sentinel, pad_to, ...)` at `inference.py:226`, now group-vectorized). This keeps `plan_groups` the sole owner of `pad_to`, preserves the off-by-one safety the spec demanded (pinned by a Rust unit test, just on the *layout* not the pad value), and keeps the v2 buffer **grouping-agnostic** so the same wire bytes work no matter how Python buckets them. Implementer 4 must: read `gather_index`/`gather_width`/`gather_sentinel`/`cu_seqlens` for `abi==2`; per group, scatter the ragged segment into `(g, pad_to, 7)` and `np.where(gi == sentinel, pad_to, gi + 0)` (local indices, no segment offset needed since the dense scatter is per-row); feed `coords` straight through as int32. The `torch.equal` gate in §3 Tier-3 (v2 dense scatter == v1 numpy loop output) is the parity assertion.

**Parity assertions for this component**
- Static (no GPU): the three `#[cfg(test)]` tests above — gather layout == model convention, v2⇒v1 neighbour reduction, cu_seqlens == v1 offsets. Run with `cargo test -p <hexfield-rust-crate> payload`.
- End-to-end (GPU pause): `scripts/_hexfield_async_parity.py` action parity and `scripts/_hexfield_compile_overlap_test.py` ASYNC `maxabsdiff==0.0` must hold with `HEXFIELD_PAYLOAD_ABI=2` because the reply ABI and the math are unchanged — same fp16 feats, same neighbour geometry, same coords (just i32). The byte gate (v2 vs v1) is the C3 Tier-3 assertion.

**Reply path, cache, dedup, submit/finish: byte-for-byte unchanged.** `parse_chunk_reply`, `finalize_priors`, all `read_*`/`require_exact_bytes`, `evaluate_state_refs_cached`, `submit_eval_cached`, `finish_eval_cached`, `integrate_unique_evals`, and `PendingEval`/`PendingKind` are identical to the live tree (only doc comments added re: FIFO drain for the depth-N pipeline). The `RustHexoState` test API (`is_legal_placement`/`apply_placement`/`new`) should be verified against the engine crate by the operator when assembling — if the constructor/method names differ, only the test fixtures need adjusting, not the production code.

File this replaces: `E:\Hexo-BotTrainer-hexgt\packages\hexfield\rust\src\payload.rs` (do not write to the live tree — assemble into `E:\Hexo-BotTrainer-hexgt-rewrite`).