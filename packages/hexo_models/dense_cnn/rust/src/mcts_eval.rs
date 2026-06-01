//! Python/PyTorch evaluator adapter for dense CNN MCTS.
//!
//! MCTS owns the tree and game-state mutations, while PyTorch remains the neural
//! evaluator. This file is the boundary between those worlds:
//!
//! 1. Hash incoming engine states with `hexo_utils`.
//! 2. Reuse cached evaluations and coalesce duplicate uncached states.
//! 3. Encode unique states into strict tensor byte payloads (f32 planes + the
//!    per-row legal crop flats).
//! 4. Call `DenseCNNInference.evaluate_model1_payload`.
//! 5. Parse exact value/prior byte buffers back into `RustEvaluation`.
//! 6. Validate finiteness, uniqueness, and prior mass before search sees it.
//!
//! There is a single evaluator mode: Rust sends the exact legal crop flats and
//! Python returns one prior per flat. No fallback format is accepted; malformed
//! bytes raise `PyValueError`.

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use half::slice::HalfFloatSliceExt;
use hexo_engine::{HexoState as RustHexoState, PackedCoord};
use hexo_utils::{hash_state, StateHash};

use super::constants::*;
use super::encoding::encode_model1_state_for_mcts;

/// Zero-copy f16 plane buffer handed to the Python evaluator.
///
/// Holds the contiguous MCTS leaf planes (as `f16`) and exposes them through the
/// Python buffer protocol so `torch.frombuffer(dtype=float16)` views them in place
/// (then copies straight to the GPU) instead of paying a serial `PyBytes` memcpy on
/// the GIL thread every forward pass. Read-only, 1-D, itemsize 1 (raw bytes).
///
/// f16 is the host->device TRANSPORT dtype: it halves the ~89 MB plane H2D copy.
/// Python upcasts to f32 on-device before the (TRT/torch) forward, which is gated
/// byte-identical because the TRT engine already downcasts its f32 input to f16
/// internally (see scripts/_fp16_input_gate.py).
#[pyclass]
pub(crate) struct PlaneBuffer {
    data: Vec<half::f16>,
}

#[pymethods]
impl PlaneBuffer {
    /// Byte length, so the Python evaluator's `len(payload["inputs"])` check is
    /// exact (f16 == 2 bytes/element).
    fn __len__(&self) -> usize {
        self.data.len() * std::mem::size_of::<half::f16>()
    }

    /// SAFETY: `view` is the CPython-supplied `Py_buffer` to populate. We expose a
    /// read-only 1-D byte view over `data` and keep `slf` alive via `view.obj`, so
    /// the backing `Vec` outlives the buffer.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("buffer view is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            (*view).obj = ptr::null_mut();
            return Err(PyBufferError::new_err("PlaneBuffer is read-only"));
        }
        let guard = slf.borrow();
        let data = &guard.data;
        (*view).buf = data.as_ptr() as *mut c_void;
        (*view).len = (data.len() * std::mem::size_of::<half::f16>()) as ffi::Py_ssize_t;
        (*view).readonly = 1;
        (*view).itemsize = 1;
        (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            // Static NUL-terminated "B" (unsigned bytes); 'static keeps the pointer
            // valid for the life of the view.
            b"B\0".as_ptr() as *mut c_char
        } else {
            ptr::null_mut()
        };
        (*view).ndim = 1;
        (*view).shape = ptr::null_mut();
        (*view).strides = ptr::null_mut();
        (*view).suboffsets = ptr::null_mut();
        (*view).internal = ptr::null_mut();
        // Hold a strong reference so the Vec lives until __releasebuffer__.
        (*view).obj = slf.clone().into_any().into_ptr();
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

#[derive(Clone, Debug)]
pub(crate) struct RustEvaluation {
    /// Scalar value from the perspective of the evaluated state's current player.
    pub(crate) value: f32,
    /// Number of in-crop legal moves. Out-of-crop engine legal moves are
    /// intentionally ignored by dense-cnn MCTS, and the evaluator returns one
    /// prior per in-crop legal move, so this equals `priors.len()`.
    pub(crate) legal_action_count: usize,
    /// One prior per in-crop legal move, ranked by descending prior.
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
    // PROFILING (perf attribution of the ~half-of-wall "glue"): payload =
    // PyDict/PyBytes construction (the per-pass plane buffer copy into Python);
    // dedup = cache-check + duplicate coalescing; cache_insert = bounded insert/
    // eviction churn. All on the GIL/eval thread, previously uninstrumented.
    pub(crate) payload_seconds: f64,
    pub(crate) dedup_seconds: f64,
    pub(crate) cache_insert_seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RustEvaluationCache {
    // `Arc` (not `Rc`) so that tree nodes holding a shared prior list stay `Send`
    // for the rayon root-selection parallelism in `run_searches_to_targets`, and
    // so the cache itself can be `Arc<Mutex<…>>` (`SharedEvaluationCache`) shared
    // across the A1 select↔eval pipeline threads.
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

// `Arc<Mutex<…>>` (M2): the cache and stats are shared across the A1 pipeline's
// eval thread and the scoped selection thread, so they must be `Send + Sync`.
// Access is at eval boundaries only (never in the hot per-ply select loop), so a
// single mutex does not serialize search work. A sharded lock would only matter
// for a future many-worker design; deferred until profiling shows lock contention.
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

fn evaluate_model1_state_refs(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    // Keep Python callback batches bounded. The caller may request many leaf
    // states at once, but Torch memory and callback latency are better behaved
    // when large batches are chunked here.
    if states.len() > MODEL1_EVAL_CHUNK_STATES {
        let mut evaluations = Vec::with_capacity(states.len());
        for chunk in states.chunks(MODEL1_EVAL_CHUNK_STATES) {
            evaluations.extend(evaluate_model1_states_chunk(py, evaluator, chunk, stats)?);
        }
        return Ok(evaluations);
    }
    evaluate_model1_states_chunk(py, evaluator, states, stats)
}

fn evaluate_model1_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let encoding_started = Instant::now();
    let encoded: Vec<_> = states
        .par_iter()
        .map(|state| encode_model1_state_for_mcts(state, true))
        .collect();
    let plane_values_per_row = MODEL1_INPUT_CHANNELS * MODEL1_BOARD_AREA;

    // Planes go to Python as a ZERO-COPY buffer view (PlaneBuffer), not a PyBytes
    // copy. The old `PyBytes::new` path memcpy'd the whole (~89 MB at batch 1024)
    // plane buffer on the single GIL-holding thread every pass — ~46% of self-play
    // wall-clock (see payload_seconds). `torch.frombuffer` only needs a
    // buffer-protocol view, which it copies straight to the GPU, so the intermediate
    // Python copy was pure waste.
    let total = encoded.len() * plane_values_per_row;
    let mut planes: Vec<half::f16> = Vec::with_capacity(total);
    // SAFETY: the par_chunks_mut fill below tiles the entire buffer exactly
    // (`encoded.len()` chunks of `plane_values_per_row`, summing to `total`) and
    // writes every element via `convert_from_f32_slice` before any read; `f16` has
    // no invalid bit patterns. This also drops the redundant zero-fill the reused
    // scratch paid on every pass.
    unsafe {
        planes.set_len(total);
    }
    // f32 -> f16 happens here in the parallel assembly (SIMD via `half`), so the
    // buffer crossing to Python and the H2D copy are half-size. Plane values are
    // binary masks / [0,1] soft features, so f16 is loss-free for search (gated).
    planes
        .par_chunks_mut(plane_values_per_row)
        .zip(encoded.par_iter())
        .for_each(|(target, row)| target.convert_from_f32_slice(&row.planes));
    let byte_len = total * std::mem::size_of::<half::f16>();
    let plane_buffer = Py::new(py, PlaneBuffer { data: planes })?;

    thread_local! {
        // Legal flat indices are small (~5 MB at batch 1024) and still travel as a
        // PyBytes copy; reuse one buffer across passes to avoid reallocations.
        static LEGAL_SCRATCH: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
    }

    let payload = LEGAL_SCRATCH.with(|scratch| -> PyResult<Bound<'_, PyDict>> {
        let mut legal_flat_indices = scratch.borrow_mut();
        legal_flat_indices.clear();
        let mut legal_row_offsets = Vec::with_capacity(encoded.len() + 1);
        legal_row_offsets.push(0i64);
        for row in &encoded {
            legal_flat_indices.extend_from_slice(&row.legal_flat_indices);
            legal_row_offsets.push(legal_flat_indices.len() as i64);
        }
        let flat_byte_len = legal_flat_indices.len() * std::mem::size_of::<i64>();
        let flat_bytes = unsafe {
            std::slice::from_raw_parts(legal_flat_indices.as_ptr() as *const u8, flat_byte_len)
        };
        if let Some(stats) = stats {
            let mut stats = lock_stats(stats);
            stats.evaluator_chunks += 1;
            stats.encoded_states += encoded.len();
            stats.encoded_legal_actions += legal_flat_indices.len();
            stats.max_chunk_states = stats.max_chunk_states.max(encoded.len());
            stats.max_chunk_legal_actions =
                stats.max_chunk_legal_actions.max(legal_flat_indices.len());
            stats.input_bytes += byte_len;
            stats.legal_index_bytes += flat_byte_len;
            stats.encoding_seconds += encoding_started.elapsed().as_secs_f64();
        }

        // This dictionary is the native evaluator ABI consumed by
        // `DenseCNNInference.evaluate_model1_payload`.
        let payload_started = Instant::now();
        let payload = PyDict::new(py);
        payload.set_item("inputs", plane_buffer)?;
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
        if let Some(stats) = stats {
            lock_stats(stats).payload_seconds += payload_started.elapsed().as_secs_f64();
        }
        Ok(payload)
    })?;

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
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
        let mut stats = lock_stats(stats);
        stats.value_bytes += value_bytes.len();
        stats.prior_bytes += prior_bytes.len();
    }

    // priors_bytes is positional: item N corresponds to the Nth legal action id
    // written into the row-offset payload sent above.
    let expected_prior_count: usize = encoded.iter().map(|row| row.legal_action_ids.len()).sum();
    require_exact_bytes("priors_bytes", prior_bytes.len(), expected_prior_count, 4)?;
    // Parse priors/values per row. Rows are independent (each reads a disjoint
    // slice of `prior_bytes` and one value), and the per-row prior decode + sort
    // (`finalize_model_priors`) was the dominant main-thread cost once the forward
    // is FP16 (~39% of a self-play move). Precompute per-row offsets so the rows
    // can be decoded across rayon workers; output stays row-ordered, so this is
    // byte-identical to the previous sequential parse.
    let parse_started = Instant::now();
    let mut prior_offsets = Vec::with_capacity(encoded.len() + 1);
    let mut running = 0usize;
    prior_offsets.push(0usize);
    for row in &encoded {
        running += row.legal_action_ids.len();
        prior_offsets.push(running);
    }
    let evaluations: Vec<RustEvaluation> = encoded
        .par_iter()
        .enumerate()
        .map(|(row_index, row)| {
            let value = read_value(value_bytes, row_index)?;
            let base = prior_offsets[row_index];
            let mut row_priors = Vec::with_capacity(row.legal_action_ids.len());
            for (offset, action_id) in row.legal_action_ids.iter().copied().enumerate() {
                let prior = read_prior(prior_bytes, base + offset, row_index)?;
                row_priors.push((action_id, prior));
            }
            finalize_model_priors(&mut row_priors, row.all_legal_action_count, row_index)?;
            Ok(RustEvaluation {
                value,
                legal_action_count: row.all_legal_action_count,
                priors: row_priors,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(stats) = stats {
        lock_stats(stats).parse_seconds += parse_started.elapsed().as_secs_f64();
    }
    Ok(evaluations)
}

pub(crate) fn evaluate_model1_states_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
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
    evaluate_model1_state_refs_cached(py, evaluator, &requests, cache, stats, cache_max_states)
}

pub(crate) fn evaluate_model1_state_refs_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    // Cache slots preserve caller order. `unique_states` contains only misses,
    // and `slot_to_unique` maps duplicate misses back to their first occurrence.
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
            let mut stats = lock_stats(stats);
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
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
        let unique_evals = evaluate_model1_state_refs(py, evaluator, &unique_states, stats)?;
        // Wrap each freshly computed evaluation in an `Arc` once (`Arc`, not `Rc`,
        // so nodes holding the shared prior list stay `Send` for rayon — see the
        // `RustEvaluationCache` note above). Cache inserts and duplicate-slot fills
        // below are refcount bumps, not deep copies of the (up to several-hundred-
        // entry) prior vector.
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
                let mut stats = lock_stats(stats);
                stats.cache_inserts += inserted;
                stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
                stats.cache_insert_seconds += insert_started.elapsed().as_secs_f64();
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
        .map(|item| item.expect("every model1 evaluation slot must be populated"))
        .collect())
}

fn finalize_model_priors(
    priors: &mut [(PackedCoord, f32)],
    legal_action_count: usize,
    row_index: usize,
) -> PyResult<()> {
    // Last validation step before priors become tree edges: every prior must be
    // finite, nonnegative, unique, and the row must carry positive total mass.
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
    if priors.len() > legal_action_count {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for row {row_index}, above legal count {legal_action_count}",
            priors.len()
        )));
    }
    let mut seen = HashSet::with_capacity(priors.len());
    let mut total = 0.0f32;
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
        total += prior;
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "evaluator returned zero total prior mass for row {row_index}"
        )));
    }
    priors.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    // Pre-normalize to sum 1.0 so tree nodes can consume the cached priors
    // directly without an extra per-node normalization pass. This is behavior
    // neutral: interior nodes previously re-normalized at construction, and the
    // root export / Python targets normalize regardless. Applying root-policy
    // temperature later to a pre-normalized prior is identical because the
    // normalization constant cancels (`normalize(x^(1/T))` is invariant to a
    // uniform prescaling of `x`).
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
    read_f32(bytes, index)
        .ok_or_else(|| PyValueError::new_err(format!("{name} missing f32 at item index {index}")))
}
