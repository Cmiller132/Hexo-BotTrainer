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
//! `tests/test_hexgnn_feature_buffer.py` (a mismatch silently poisons the model,
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

/// Minimum number of consecutive graphs handled by a single rayon featurize task.
/// build_graph+featurize for one opening graph (~150-250 nodes) is far cheaper than
/// a rayon task spawn/steal, so a 1-graph-per-task split oversubscribes the pool;
/// grouping this many per task amortizes the scheduling overhead. Affects ONLY work
/// granularity (scheduling), never the output order or values — the indexed
/// `par_iter().collect()` stays in input order.
const FEATURIZE_RAYON_MIN_LEN: usize = 8;

/// Per-graph featurized arrays (pre-collation).
pub(crate) struct GraphFeat {
    pub(crate) node_feat: Vec<f32>, // num_nodes * NODE_FEATURE_DIM
    pub(crate) node_type: Vec<i64>, // num_nodes
    pub(crate) edge_src: Vec<i64>,  // num_edges (graph-local)
    pub(crate) edge_dst: Vec<i64>,
    pub(crate) edge_type: Vec<i64>,
    pub(crate) edge_dir: Vec<i64>, // num_edges; adjacency 0..5, else -1
    pub(crate) edge_attr: Vec<f32>, // num_edges * EDGE_ATTR_DIM
    pub(crate) candidate_nodes: Vec<i64>, // graph-local node indices
    pub(crate) candidate_ids: Vec<i64>,
    pub(crate) num_nodes: usize,
}

/// Featurize one position's graph (mirror of `features.build_graph_tensors`).
pub(crate) fn featurize_position_graph(
    g: &PositionGraph,
    placements: i32,
    phase: u8,
    first_stone: Option<(i16, i16)>,
) -> GraphFeat {
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
            // Per-STONE window-count features (sparse rewrite): active windows
            // through this stone's cell, split by owner, normalized like the
            // candidate nwin (/COORD_SCALE). Sourced from node_wins (folded from the
            // removed STONE_WINDOW hub edges).
            let w = &g.node_wins[i];
            feat[base + F_STONE_NWIN_OWN] = (w.stone_nwin_own as f32) / COORD_SCALE;
            feat[base + F_STONE_NWIN_OPP] = (w.stone_nwin_opp as f32) / COORD_SCALE;
        } else if t == FT_NODE_CANDIDATE {
            // v2: hex-distance from this candidate to THIS turn's first stone.
            // 0 (left zero) when there is no first stone yet — i.e. on the turn's
            // FIRST placement (Opening/FirstStone). D6-invariant (a hex distance).
            if let Some((fq, fr)) = first_stone {
                let dq = q - fq as i32;
                let dr = r - fr as i32;
                let dd = -dq - dr;
                let d = dq.abs().max(dr.abs()).max(dd.abs()) as f32;
                feat[base + F_CAND_DIST_FIRST] = d / COORD_SCALE;
            }
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
            // v2: 1 when the current placement is the turn's SECOND stone.
            feat[base + F_SIDE_IS_SECOND] = if ph == 2 { 1.0 } else { 0.0 };
        }
    }

    // edges
    let e = g.edge_src.len();
    let mut edge_attr = vec![0f32; e * EDGE_ATTR_DIM];
    let mut edge_src = Vec::with_capacity(e);
    let mut edge_dst = Vec::with_capacity(e);
    let mut edge_type = Vec::with_capacity(e);
    let mut edge_dir = Vec::with_capacity(e);
    for k in 0..e {
        let su = g.edge_src[k] as usize;
        let du = g.edge_dst[k] as usize;
        edge_src.push(g.edge_src[k] as i64);
        edge_dst.push(g.edge_dst[k] as i64);
        let et = (g.edge_type[k] as i64).clamp(0, NUM_EDGE_TYPES - 1) as usize;
        edge_type.push(g.edge_type[k] as i64);
        // Per-edge hex-direction index (adjacency 0..5, else -1); i8 -> i64.
        edge_dir.push(g.edge_dir[k] as i64);
        edge_attr[k * EDGE_ATTR_DIM + et] = 1.0;
        let dq = g.node_q[su] as i32 - g.node_q[du] as i32;
        let dr = g.node_r[su] as i32 - g.node_r[du] as i32;
        let ds = -dq - dr;
        let dist = dq.abs().max(dr.abs()).max(ds.abs()) as f32;
        edge_attr[k * EDGE_ATTR_DIM + EDGE_ATTR_DIST] = dist / COORD_SCALE;
    }

    // Candidate tactical window-count features (sparse rewrite): sourced directly
    // from the per-node `node_wins` accumulators (computed in build_graph from the
    // active-window tokens), replacing the removed candidate<->window hub edges.
    // The accumulated counts/flags are IDENTICAL to what the old window->candidate
    // edge loop produced, so the encoded candidate features are byte-identical; the
    // only difference is the source of the counts (tokens vs hub edges). Counts are
    // populated by build_graph on candidate cells only (stones use the stone_nwin_*
    // fields, encoded above), so iterating all nodes is safe (non-candidates have 0).
    for i in 0..n {
        let b = i * NODE_FEATURE_DIM;
        let w = &g.node_wins[i];
        feat[b + F_CAND_NWIN_OWN] = (w.nwin_own as f32) / COORD_SCALE;
        feat[b + F_CAND_NWIN_OPP] = (w.nwin_opp as f32) / COORD_SCALE;
        // v2 per-count splits share the NWIN normalization (so the three own bins
        // sum to NWIN_OWN, the three opp bins to NWIN_OPP — gated by tests).
        feat[b + F_CAND_OWN_WIN3] = (w.own_win3 as f32) / COORD_SCALE;
        feat[b + F_CAND_OWN_WIN4] = (w.own_win4 as f32) / COORD_SCALE;
        feat[b + F_CAND_OWN_WIN5] = (w.own_win5 as f32) / COORD_SCALE;
        feat[b + F_CAND_OPP_WIN3] = (w.opp_win3 as f32) / COORD_SCALE;
        feat[b + F_CAND_OPP_WIN4] = (w.opp_win4 as f32) / COORD_SCALE;
        feat[b + F_CAND_OPP_WIN5] = (w.opp_win5 as f32) / COORD_SCALE;
        if w.complete_own {
            feat[b + F_CAND_COMPLETE_OWN] = 1.0;
        }
        if w.complete_opp {
            feat[b + F_CAND_COMPLETE_OPP] = 1.0;
        }
        // v3: phase-aware own win-now — own count-5 (any B) OR own count-4 only at
        // FirstStone (ph == 1). Reconstructed from the per-count own accumulators.
        if w.own_win5 > 0 || (w.own_win4 > 0 && ph == 1) {
            feat[b + F_CAND_WIN_NOW_OWN] = 1.0;
        }
        // v3: any opponent >=4 window through this empty cell is a must-answer
        // immediate threat (the block-cell set). Equivalent to opp_win4>0||opp_win5>0.
        if w.opp_threat {
            feat[b + F_CAND_OPP_THREAT] = 1.0;
        }
    }

    GraphFeat {
        node_feat: feat,
        node_type: g.node_type.iter().map(|&t| t as i64).collect(),
        edge_src,
        edge_dst,
        edge_type,
        edge_dir,
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
    pub(crate) edge_dir: Vec<i64>, // num_edges; adjacency 0..5, else -1
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

/// Disjoint mutable destination slices into the collated buffers for ONE graph.
/// Writing through these is data-race-free because every graph owns a non-
/// overlapping region (the offsets partition each buffer exactly).
struct GraphDst<'a> {
    node_feat: &'a mut [f32],
    node_type: &'a mut [i64],
    node_graph: &'a mut [i64],
    edge_src: &'a mut [i64], // first half of edge_index
    edge_dst: &'a mut [i64], // second half of edge_index
    edge_type: &'a mut [i64],
    edge_dir: &'a mut [i64],
    edge_attr: &'a mut [f32],
    candidate_index: &'a mut [i64],
    candidate_graph: &'a mut [i64],
    candidate_ids: &'a mut [i64],
}

/// Carve a preallocated buffer into per-graph sub-slices of the given lengths.
/// Cheap pointer arithmetic (no copies); the returned slices tile `buf` exactly.
fn carve<'a, T>(buf: &'a mut [T], lens: impl Iterator<Item = usize>) -> Vec<&'a mut [T]> {
    let mut rest = buf;
    let mut out = Vec::new();
    for len in lens {
        let (head, tail) = rest.split_at_mut(len);
        out.push(head);
        rest = tail;
    }
    out
}

/// Collate per-graph features into one disjoint packed graph, in PARALLEL.
///
/// The serial concatenation this replaces was ~90% of the featurization wall (it
/// is memory-latency-bound: it gathers from ~`num_graphs` separately-allocated,
/// cache-cold `GraphFeat`s). We instead size the output buffers up front from a
/// serial prefix-sum of per-graph counts, carve each buffer into disjoint
/// per-graph destination slices, then scatter every graph's data into its own
/// slice across the rayon pool. Output bytes are identical to the serial form
/// (gated by `tests/test_hexgnn_feature_buffer.py`).
pub(crate) fn collate(feats: &[GraphFeat]) -> CollatedFeatures {
    let num_graphs = feats.len();

    // Per-graph node base (exclusive prefix sum) — the only offset the scatter
    // needs, to rebase each graph's local edge/candidate node indices into the
    // packed numbering. Buffer positions are handled by `carve` below.
    let mut node_offsets = Vec::with_capacity(num_graphs);
    let mut node_acc = 0usize;
    for g in feats {
        node_offsets.push(node_acc as i64);
        node_acc += g.num_nodes;
    }
    let num_nodes = node_acc;
    let num_edges: usize = feats.iter().map(|g| g.edge_src.len()).sum();
    let num_candidates: usize = feats.iter().map(|g| g.candidate_nodes.len()).sum();
    let candidate_counts: Vec<usize> = feats.iter().map(|g| g.candidate_ids.len()).collect();

    // Allocate the collated buffers once, at exact final size.
    let mut node_feat = vec![0f32; num_nodes * NODE_FEATURE_DIM];
    let mut node_type = vec![0i64; num_nodes];
    let mut node_graph = vec![0i64; num_nodes];
    // edge_index packs [all src .. , all dst ..] so Python recovers (2, E) with
    // one reshape; split it into the two halves before carving per-graph slices.
    let mut edge_index = vec![0i64; 2 * num_edges];
    let mut edge_type = vec![0i64; num_edges];
    let mut edge_dir = vec![0i64; num_edges];
    let mut edge_attr = vec![0f32; num_edges * EDGE_ATTR_DIM];
    let mut candidate_index = vec![0i64; num_candidates];
    let mut candidate_graph = vec![0i64; num_candidates];
    let mut candidate_ids = vec![0i64; num_candidates];

    let (edge_src_buf, edge_dst_buf) = edge_index.split_at_mut(num_edges);
    let mut node_feat_dst = carve(&mut node_feat, feats.iter().map(|g| g.num_nodes * NODE_FEATURE_DIM));
    let mut node_type_dst = carve(&mut node_type, feats.iter().map(|g| g.num_nodes));
    let mut node_graph_dst = carve(&mut node_graph, feats.iter().map(|g| g.num_nodes));
    let mut edge_src_dst = carve(edge_src_buf, feats.iter().map(|g| g.edge_src.len()));
    let mut edge_dst_dst = carve(edge_dst_buf, feats.iter().map(|g| g.edge_dst.len()));
    let mut edge_type_dst = carve(&mut edge_type, feats.iter().map(|g| g.edge_type.len()));
    let mut edge_dir_dst = carve(&mut edge_dir, feats.iter().map(|g| g.edge_dir.len()));
    let mut edge_attr_dst = carve(&mut edge_attr, feats.iter().map(|g| g.edge_attr.len()));
    let mut cand_index_dst = carve(&mut candidate_index, feats.iter().map(|g| g.candidate_nodes.len()));
    let mut cand_graph_dst = carve(&mut candidate_graph, feats.iter().map(|g| g.candidate_nodes.len()));
    let mut cand_ids_dst = carve(&mut candidate_ids, feats.iter().map(|g| g.candidate_ids.len()));

    // Zip the per-graph destination slices into one handle per graph, so a single
    // parallel pass fills every buffer (each graph touches only its own region).
    let mut dsts: Vec<GraphDst<'_>> = node_feat_dst
        .drain(..)
        .zip(node_type_dst.drain(..))
        .zip(node_graph_dst.drain(..))
        .zip(edge_src_dst.drain(..))
        .zip(edge_dst_dst.drain(..))
        .zip(edge_type_dst.drain(..))
        .zip(edge_dir_dst.drain(..))
        .zip(edge_attr_dst.drain(..))
        .zip(cand_index_dst.drain(..))
        .zip(cand_graph_dst.drain(..))
        .zip(cand_ids_dst.drain(..))
        .map(
            |((((((((((nf, nt), ng), es), ed), et), edir), ea), ci), cg), cid)| GraphDst {
                node_feat: nf,
                node_type: nt,
                node_graph: ng,
                edge_src: es,
                edge_dst: ed,
                edge_type: et,
                edge_dir: edir,
                edge_attr: ea,
                candidate_index: ci,
                candidate_graph: cg,
                candidate_ids: cid,
            },
        )
        .collect();

    dsts.par_iter_mut()
        .zip(feats.par_iter())
        .zip(node_offsets.par_iter())
        .enumerate()
        // Same rationale as featurize: opening graphs are tiny, so coarsen the
        // per-graph scatter grain to avoid oversubscribing the pool. Each graph
        // still writes ONLY its own disjoint output region, so grain affects
        // scheduling only — the collated bytes are identical to a 1-per-task split.
        .with_min_len(FEATURIZE_RAYON_MIN_LEN)
        .for_each(|(gid, ((dst, g), &node_offset))| {
            let gid = gid as i64;
            dst.node_feat.copy_from_slice(&g.node_feat);
            dst.node_type.copy_from_slice(&g.node_type);
            dst.node_graph.fill(gid);
            for (out, &v) in dst.edge_src.iter_mut().zip(&g.edge_src) {
                *out = v + node_offset;
            }
            for (out, &v) in dst.edge_dst.iter_mut().zip(&g.edge_dst) {
                *out = v + node_offset;
            }
            dst.edge_type.copy_from_slice(&g.edge_type);
            dst.edge_dir.copy_from_slice(&g.edge_dir);
            dst.edge_attr.copy_from_slice(&g.edge_attr);
            for (out, &c) in dst.candidate_index.iter_mut().zip(&g.candidate_nodes) {
                *out = c + node_offset;
            }
            dst.candidate_graph.fill(gid);
            dst.candidate_ids.copy_from_slice(&g.candidate_ids);
        });

    CollatedFeatures {
        node_feat,
        node_type,
        node_graph,
        edge_index,
        edge_type,
        edge_dir,
        edge_attr,
        candidate_index,
        candidate_graph,
        candidate_ids,
        candidate_counts,
        num_graphs,
        num_nodes,
        num_edges,
        num_candidates,
    }
}

/// Featurize + collate a batch of Rust states (GIL-free: build_graph + featurize +
/// collate run under rayon). Used by the MCTS eval (Rust states) — the caller
/// supplies the live states; per-state (placements, phase) come from each state.
pub(crate) fn featurize_collate_states(states: &[&RustHexoState], n: i16) -> CollatedFeatures {
    // Opening graphs are tiny (~150-250 nodes) and a batch can hold thousands of
    // them, so a 1-graph-per-rayon-task split oversubscribes the pool with work
    // items smaller than the spawn/steal overhead. `with_min_len` groups several
    // consecutive graphs into one task to amortize that overhead. It only changes
    // how the (indexed) parallel iterator is SPLIT, not the element order or the
    // per-element computation, so `.collect()` still yields the graphs in input
    // order and every `GraphFeat` is byte-identical to the 1-per-task form.
    let feats: Vec<GraphFeat> = states
        .par_iter()
        .with_min_len(FEATURIZE_RAYON_MIN_LEN)
        .map(|s| {
            let g = build_graph(s, n);
            let placements = s.placements_made() as i32;
            let (phase, first_stone) = match s.phase() {
                TurnPhase::Opening => (0u8, None),
                TurnPhase::FirstStone => (1u8, None),
                TurnPhase::SecondStone { first } => (2u8, Some((first.q, first.r))),
            };
            featurize_position_graph(&g, placements, phase, first_stone)
        })
        .collect();
    collate(&feats)
}

// --- Zero-copy buffer-protocol pyclasses (mirror dense_cnn's PlaneBuffer) -----

/// Read-only 1-D byte view over an owned `Vec<f32>` so
/// `torch.frombuffer(buf, dtype=torch.float32)` views it in place.
#[pyclass]
pub(crate) struct HexgnnF32Buffer {
    data: Vec<f32>,
}

#[pymethods]
impl HexgnnF32Buffer {
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
pub(crate) struct HexgnnI64Buffer {
    data: Vec<i64>,
}

#[pymethods]
impl HexgnnI64Buffer {
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
        return Err(PyBufferError::new_err("hexgnn buffer is read-only"));
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
    d.set_item("node_feat", Py::new(py, HexgnnF32Buffer { data: c.node_feat })?)?;
    d.set_item("node_type", Py::new(py, HexgnnI64Buffer { data: c.node_type })?)?;
    d.set_item("node_graph", Py::new(py, HexgnnI64Buffer { data: c.node_graph })?)?;
    d.set_item("edge_index", Py::new(py, HexgnnI64Buffer { data: c.edge_index })?)?;
    d.set_item("edge_type", Py::new(py, HexgnnI64Buffer { data: c.edge_type })?)?;
    // Per-edge hex-direction index (int64; adjacency 0..5, else -1).
    d.set_item("edge_dir", Py::new(py, HexgnnI64Buffer { data: c.edge_dir })?)?;
    d.set_item("edge_attr", Py::new(py, HexgnnF32Buffer { data: c.edge_attr })?)?;
    d.set_item(
        "candidate_index",
        Py::new(py, HexgnnI64Buffer { data: c.candidate_index })?,
    )?;
    d.set_item(
        "candidate_graph",
        Py::new(py, HexgnnI64Buffer { data: c.candidate_graph })?,
    )?;
    if include_candidate_ids {
        d.set_item(
            "candidate_ids",
            Py::new(py, HexgnnI64Buffer { data: c.candidate_ids })?,
        )?;
    }
    Ok(d)
}

/// PyO3 entry: featurize+collate a batch of live engine states at radius `n`,
/// returning the zero-copy packed batch payload. Used by the parity test + the
/// trainer's parallel expand. rayon featurization runs with the GIL released.
#[pyfunction]
#[pyo3(signature = (states, n))]
pub fn hexgnn_featurize_states(py: Python<'_>, states: &Bound<'_, PyAny>, n: i64) -> PyResult<Py<PyAny>> {
    let owned = super::state::states_from_py_states(py, states)?;
    let refs: Vec<&RustHexoState> = owned.iter().collect();
    let collated = py.detach(|| featurize_collate_states(&refs, n as i16));
    Ok(collated_to_py_dict(py, collated, true)?.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexgnn_featurize_states, module)?)?;
    module.add_class::<HexgnnF32Buffer>()?;
    module.add_class::<HexgnnI64Buffer>()?;
    Ok(())
}
