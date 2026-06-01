//! Rust featurizer — the parallel replacement for the per-graph Python
//! `features.build_graph_tensors` + `collate.collate_graphs` (the ~88% self-play
//! bottleneck; see HEXGT_DECISIONS.md Phase-5c). Featurizes each position's
//! `PositionGraph` into the D6-invariant node/edge features and collates a batch
//! into one disjoint graph, all in Rust with rayon across graphs (GIL released),
//! emitting the collated batch as ZERO-COPY buffer-protocol views the Python
//! evaluator wraps with `torch.frombuffer` (one H2D copy, no Python-side copy —
//! mirrors dense_cnn's `PlaneBuffer`).
//!
//! MUST stay byte-identical to the Python featurizer+collator — guarded by
//! `tests/test_hexgt_feature_buffer.py` (a mismatch silently poisons the model,
//! the dense_cnn D6-augmentation bug class). Any change to the feature layout
//! updates constants.rs + python/.../constants.py + features.py together.

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use hexo_engine::{HexoState as RustHexoState, TurnPhase};

use super::candidates::{build_graph, PositionGraph};
use super::constants::*;

/// Per-graph featurized arrays (pre-collation).
pub(crate) struct GraphFeat {
    pub(crate) node_feat: Vec<f32>, // num_nodes * NODE_FEATURE_DIM
    pub(crate) node_type: Vec<i64>, // num_nodes
    pub(crate) edge_src: Vec<i64>,  // num_edges (graph-local)
    pub(crate) edge_dst: Vec<i64>,
    pub(crate) edge_type: Vec<i64>,
    pub(crate) edge_attr: Vec<f32>, // num_edges * EDGE_ATTR_DIM
    pub(crate) candidate_nodes: Vec<i64>, // graph-local node indices
    pub(crate) candidate_ids: Vec<i64>,
    pub(crate) num_nodes: usize,
}

/// Featurize one position's graph (mirror of `features.build_graph_tensors`).
pub(crate) fn featurize_position_graph(g: &PositionGraph, placements: i32, phase: u8) -> GraphFeat {
    let n = g.node_count();
    let mut feat = vec![0f32; n * NODE_FEATURE_DIM];
    let denom_recency = placements.max(1) as f32;

    let mut stones_own = 0usize;
    let mut stones_opp = 0usize;
    for i in 0..n {
        let base = i * NODE_FEATURE_DIM;
        let t = g.node_type[i] as i64;
        if (0..NUM_NODE_TYPES).contains(&t) {
            feat[base + F_TYPE_ONEHOT + t as usize] = 1.0;
        }
        let owner = g.node_owner[i];
        if owner == 0 {
            feat[base + F_OWNER_OWN] = 1.0;
        } else if owner == 1 {
            feat[base + F_OWNER_OPP] = 1.0;
        }
        let q = g.node_q[i] as i32;
        let r = g.node_r[i] as i32;
        let s = -q - r;
        let cd = q.abs().max(r.abs()).max(s.abs()) as f32;
        feat[base + F_CENTER_DISTANCE] = cd / COORD_SCALE;

        if t == FT_NODE_STONE {
            feat[base + F_STONE_RECENCY] = (g.node_recency[i].max(0) as f32) / denom_recency;
            if owner == 0 {
                stones_own += 1;
            } else if owner == 1 {
                stones_opp += 1;
            }
        } else if t == FT_NODE_WINDOW {
            let widx = ((g.node_wcount[i] as i64) - 3).clamp(0, 2) as usize;
            feat[base + F_WIN_COUNT_ONEHOT + widx] = 1.0;
            feat[base + F_WIN_EMPTY_CELLS] = (g.node_wempty[i] as f32) / WINDOW_LEN_F;
        }
    }

    let ph = (phase as i64).clamp(0, 2) as usize;
    for i in 0..n {
        if g.node_type[i] as i64 == FT_NODE_SIDE {
            let base = i * NODE_FEATURE_DIM;
            feat[base + F_SIDE_PHASE_ONEHOT + ph] = 1.0;
            feat[base + F_SIDE_STONES_OWN] = stones_own as f32 / COUNT_SCALE;
            feat[base + F_SIDE_STONES_OPP] = stones_opp as f32 / COUNT_SCALE;
            feat[base + F_SIDE_MOVE_NUMBER] = placements as f32 / COUNT_SCALE;
        }
    }

    // edges
    let e = g.edge_src.len();
    let mut edge_attr = vec![0f32; e * EDGE_ATTR_DIM];
    let mut edge_src = Vec::with_capacity(e);
    let mut edge_dst = Vec::with_capacity(e);
    let mut edge_type = Vec::with_capacity(e);
    for k in 0..e {
        let su = g.edge_src[k] as usize;
        let du = g.edge_dst[k] as usize;
        edge_src.push(g.edge_src[k] as i64);
        edge_dst.push(g.edge_dst[k] as i64);
        let et = (g.edge_type[k] as i64).clamp(0, NUM_EDGE_TYPES_F - 1) as usize;
        edge_type.push(g.edge_type[k] as i64);
        edge_attr[k * EDGE_ATTR_DIM + et] = 1.0;
        let dq = g.node_q[su] as i32 - g.node_q[du] as i32;
        let dr = g.node_r[su] as i32 - g.node_r[du] as i32;
        let ds = -dq - dr;
        let dist = dq.abs().max(dr.abs()).max(ds.abs()) as f32;
        edge_attr[k * EDGE_ATTR_DIM + EDGE_ATTR_DIST] = dist / COORD_SCALE;
    }

    // candidate tactical features from candidate<->window edges (window is src).
    if e > 0 {
        for k in 0..e {
            if g.edge_type[k] as i64 == FT_EDGE_CANDIDATE_WINDOW {
                let wsrc = g.edge_src[k] as usize;
                if g.node_type[wsrc] as i64 == FT_NODE_WINDOW {
                    let cdst = g.edge_dst[k] as usize;
                    let cbase = cdst * NODE_FEATURE_DIM;
                    let own = g.node_owner[wsrc];
                    let cnt = g.node_wcount[wsrc] as i64;
                    if own == 0 {
                        feat[cbase + F_CAND_NWIN_OWN] += 1.0;
                        if cnt == 5 {
                            feat[cbase + F_CAND_COMPLETE_OWN] = 1.0;
                        }
                    } else if own == 1 {
                        feat[cbase + F_CAND_NWIN_OPP] += 1.0;
                        if cnt == 5 {
                            feat[cbase + F_CAND_COMPLETE_OPP] = 1.0;
                        }
                    }
                }
            }
        }
        for i in 0..n {
            feat[i * NODE_FEATURE_DIM + F_CAND_NWIN_OWN] /= COORD_SCALE;
            feat[i * NODE_FEATURE_DIM + F_CAND_NWIN_OPP] /= COORD_SCALE;
        }
    }

    GraphFeat {
        node_feat: feat,
        node_type: g.node_type.iter().map(|&t| t as i64).collect(),
        edge_src,
        edge_dst,
        edge_type,
        edge_attr,
        candidate_nodes: g.candidate_nodes.iter().map(|&c| c as i64).collect(),
        candidate_ids: g.candidate_ids.iter().map(|&c| c as i64).collect(),
        num_nodes: n,
    }
}

/// Collated disjoint-graph batch buffers (mirror of `collate.collate_graphs`).
/// `edge_index` is packed as `[src.., dst..]` (length 2*E) so Python recovers the
/// (2, E) tensor with one `reshape`. `candidate_counts` are the per-graph CSR row
/// lengths (the Rust search path zips priors back to its own `candidate_ids`).
pub(crate) struct CollatedFeatures {
    pub(crate) node_feat: Vec<f32>,
    pub(crate) node_type: Vec<i64>,
    pub(crate) node_graph: Vec<i64>,
    pub(crate) edge_index: Vec<i64>, // 2 * num_edges
    pub(crate) edge_type: Vec<i64>,
    pub(crate) edge_attr: Vec<f32>,
    pub(crate) candidate_index: Vec<i64>,
    pub(crate) candidate_graph: Vec<i64>,
    pub(crate) candidate_ids: Vec<i64>,
    pub(crate) candidate_counts: Vec<usize>,
    pub(crate) num_graphs: usize,
    pub(crate) num_nodes: usize,
    pub(crate) num_edges: usize,
    pub(crate) num_candidates: usize,
}

pub(crate) fn collate(feats: &[GraphFeat]) -> CollatedFeatures {
    let num_nodes: usize = feats.iter().map(|g| g.num_nodes).sum();
    let num_edges: usize = feats.iter().map(|g| g.edge_type.len()).sum();
    let num_candidates: usize = feats.iter().map(|g| g.candidate_ids.len()).sum();

    let mut node_feat = Vec::with_capacity(num_nodes * NODE_FEATURE_DIM);
    let mut node_type = Vec::with_capacity(num_nodes);
    let mut node_graph = Vec::with_capacity(num_nodes);
    let mut edge_src = Vec::with_capacity(num_edges);
    let mut edge_dst = Vec::with_capacity(num_edges);
    let mut edge_type = Vec::with_capacity(num_edges);
    let mut edge_attr = Vec::with_capacity(num_edges * EDGE_ATTR_DIM);
    let mut candidate_index = Vec::with_capacity(num_candidates);
    let mut candidate_graph = Vec::with_capacity(num_candidates);
    let mut candidate_ids = Vec::with_capacity(num_candidates);
    let mut candidate_counts = Vec::with_capacity(feats.len());

    let mut node_offset: i64 = 0;
    for (gid, g) in feats.iter().enumerate() {
        let gid = gid as i64;
        node_feat.extend_from_slice(&g.node_feat);
        node_type.extend_from_slice(&g.node_type);
        node_graph.extend(std::iter::repeat(gid).take(g.num_nodes));
        for k in 0..g.edge_src.len() {
            edge_src.push(g.edge_src[k] + node_offset);
            edge_dst.push(g.edge_dst[k] + node_offset);
        }
        edge_type.extend_from_slice(&g.edge_type);
        edge_attr.extend_from_slice(&g.edge_attr);
        for &c in &g.candidate_nodes {
            candidate_index.push(c + node_offset);
        }
        candidate_graph.extend(std::iter::repeat(gid).take(g.candidate_nodes.len()));
        candidate_ids.extend_from_slice(&g.candidate_ids);
        candidate_counts.push(g.candidate_ids.len());
        node_offset += g.num_nodes as i64;
    }

    // edge_index = [all src, all dst]; Python `reshape(2, num_edges)`.
    let mut edge_index = edge_src;
    edge_index.extend_from_slice(&edge_dst);

    CollatedFeatures {
        node_feat,
        node_type,
        node_graph,
        edge_index,
        edge_type,
        edge_attr,
        candidate_index,
        candidate_graph,
        candidate_ids,
        candidate_counts,
        num_graphs: feats.len(),
        num_nodes,
        num_edges,
        num_candidates,
    }
}

/// Featurize + collate a batch of Rust states (GIL-free: build_graph + featurize +
/// collate run under rayon). Used by the MCTS eval (Rust states) — the caller
/// supplies the live states; per-state (placements, phase) come from each state.
pub(crate) fn featurize_collate_states(states: &[&RustHexoState], n: i16) -> CollatedFeatures {
    let feats: Vec<GraphFeat> = states
        .par_iter()
        .map(|s| {
            let g = build_graph(s, n);
            let placements = s.placements_made() as i32;
            let phase = match s.phase() {
                TurnPhase::Opening => 0u8,
                TurnPhase::FirstStone => 1u8,
                TurnPhase::SecondStone { .. } => 2u8,
            };
            featurize_position_graph(&g, placements, phase)
        })
        .collect();
    collate(&feats)
}

// --- Zero-copy buffer-protocol pyclasses (mirror dense_cnn's PlaneBuffer) -----

/// Read-only 1-D byte view over an owned `Vec<f32>` so
/// `torch.frombuffer(buf, dtype=torch.float32)` views it in place.
#[pyclass]
pub(crate) struct HexgtF32Buffer {
    data: Vec<f32>,
}

#[pymethods]
impl HexgtF32Buffer {
    fn __len__(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (buf, len) = {
            let g = slf.borrow();
            (
                g.data.as_ptr() as *mut c_void,
                g.data.len() * std::mem::size_of::<f32>(),
            )
        };
        fill_byte_view(view, flags, buf, len, slf.clone().into_any())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// Read-only 1-D byte view over an owned `Vec<i64>` for
/// `torch.frombuffer(buf, dtype=torch.int64)`.
#[pyclass]
pub(crate) struct HexgtI64Buffer {
    data: Vec<i64>,
}

#[pymethods]
impl HexgtI64Buffer {
    fn __len__(&self) -> usize {
        self.data.len() * std::mem::size_of::<i64>()
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (buf, len) = {
            let g = slf.borrow();
            (
                g.data.as_ptr() as *mut c_void,
                g.data.len() * std::mem::size_of::<i64>(),
            )
        };
        fill_byte_view(view, flags, buf, len, slf.clone().into_any())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// Shared `__getbuffer__` body: a read-only, 1-D, itemsize-1 ("B") view of `len`
/// bytes at `buf`, keeping `owner` alive for the life of the view.
unsafe fn fill_byte_view(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    buf: *mut c_void,
    len: usize,
    owner: Bound<'_, PyAny>,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("buffer view is null"));
    }
    if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
        (*view).obj = ptr::null_mut();
        return Err(PyBufferError::new_err("hexgt buffer is read-only"));
    }
    (*view).buf = buf;
    (*view).len = len as ffi::Py_ssize_t;
    (*view).readonly = 1;
    (*view).itemsize = 1;
    (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
        b"B\0".as_ptr() as *mut c_char
    } else {
        ptr::null_mut()
    };
    (*view).ndim = 1;
    (*view).shape = ptr::null_mut();
    (*view).strides = ptr::null_mut();
    (*view).suboffsets = ptr::null_mut();
    (*view).internal = ptr::null_mut();
    (*view).obj = owner.into_ptr();
    Ok(())
}

/// Pack a collated batch into the Python evaluator payload, MOVING each flat
/// buffer into a zero-copy pyclass. `include_candidate_ids` adds the CSR action
/// ids (needed by the parity test + training expand; the search path holds them
/// in Rust and skips them). Consumes `c`.
pub(crate) fn collated_to_py_dict<'py>(
    py: Python<'py>,
    c: CollatedFeatures,
    include_candidate_ids: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("num_graphs", c.num_graphs)?;
    d.set_item("total_nodes", c.num_nodes)?;
    d.set_item("total_edges", c.num_edges)?;
    d.set_item("total_candidates", c.num_candidates)?;
    d.set_item("feat_dim", NODE_FEATURE_DIM)?;
    d.set_item("attr_dim", EDGE_ATTR_DIM)?;
    d.set_item("node_feat", Py::new(py, HexgtF32Buffer { data: c.node_feat })?)?;
    d.set_item("node_type", Py::new(py, HexgtI64Buffer { data: c.node_type })?)?;
    d.set_item("node_graph", Py::new(py, HexgtI64Buffer { data: c.node_graph })?)?;
    d.set_item("edge_index", Py::new(py, HexgtI64Buffer { data: c.edge_index })?)?;
    d.set_item("edge_type", Py::new(py, HexgtI64Buffer { data: c.edge_type })?)?;
    d.set_item("edge_attr", Py::new(py, HexgtF32Buffer { data: c.edge_attr })?)?;
    d.set_item(
        "candidate_index",
        Py::new(py, HexgtI64Buffer { data: c.candidate_index })?,
    )?;
    d.set_item(
        "candidate_graph",
        Py::new(py, HexgtI64Buffer { data: c.candidate_graph })?,
    )?;
    if include_candidate_ids {
        d.set_item(
            "candidate_ids",
            Py::new(py, HexgtI64Buffer { data: c.candidate_ids })?,
        )?;
    }
    Ok(d)
}

/// PyO3 entry: featurize+collate a batch of live engine states at radius `n`,
/// returning the zero-copy packed batch payload. Used by the parity test + the
/// trainer's parallel expand. rayon featurization runs with the GIL released.
#[pyfunction]
#[pyo3(signature = (states, n))]
pub fn hexgt_featurize_states(py: Python<'_>, states: &Bound<'_, PyAny>, n: i64) -> PyResult<Py<PyAny>> {
    let owned = super::state::states_from_py_states(py, states)?;
    let refs: Vec<&RustHexoState> = owned.iter().collect();
    let collated = py.detach(|| featurize_collate_states(&refs, n as i16));
    Ok(collated_to_py_dict(py, collated, true)?.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexgt_featurize_states, module)?)?;
    module.add_class::<HexgtF32Buffer>()?;
    module.add_class::<HexgtI64Buffer>()?;
    Ok(())
}
