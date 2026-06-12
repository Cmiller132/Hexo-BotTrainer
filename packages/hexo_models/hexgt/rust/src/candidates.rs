//! Shared candidate-set + active-window + bounded-graph construction.
//!
//! This is the SINGLE Rust path (HEXFORMER_REWRITE_PLAN.md §4.5) used by both
//! sample-gen (expand-time recompute, Phase 4) and live MCTS (Phase 5), so the
//! move vocabulary at a node is identical in training data and in play.
//!
//! Candidate set (§4):
//!   candidate_set = { empty cells in ANY active window of either player }      (A)
//!                 ∪ { empty cells within hex-distance <= n of ANY stone that
//!                     are NOT "dead" }                                          (B)
//! with `n` the single tunable radius (default 3, practical range [2, 4] — see
//! python/.../constants.py DEFAULT_CANDIDATE_RADIUS; the early "up to 8" framing
//! was dropped by the coverage-sweep decision). A radius-n cell
//! is "dead" (and dropped) only when EVERY length-6 line through it is BLOCKED
//! (contains both colors -> uncompletable); a cell on any active or open line is
//! kept. Dead cells are useless moves; in practice they are rare (mostly dense
//! endgames) and carry ~0 of a strong player's visit mass (validated).
//!
//! Tactical-window tokens (§5): the count-3/4/5 active windows of both colors.
//!
//! Bounded typed edges (§6.3) — line/co-linearity is routed THROUGH window nodes
//! as hubs; there are NO all-pairs same-axis cliques. Total edges are linear in
//! (#nodes + #window_tokens).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use hexo_engine::{pack_coord, Axis, HexCoord, HexoState as RustHexoState, TurnPhase};

// Node type ids (mirror of hexgt/python/.../constants.py).
pub const NODE_SIDE: u8 = 0;
pub const NODE_STONE: u8 = 1;
pub const NODE_CANDIDATE: u8 = 2;
pub const NODE_WINDOW: u8 = 3;

// Edge type ids (mirror of constants.py).
pub const EDGE_ADJACENCY: u8 = 0;
pub const EDGE_STONE_WINDOW: u8 = 1;
pub const EDGE_CANDIDATE_WINDOW: u8 = 2;
pub const EDGE_RECENCY: u8 = 3;
pub const EDGE_CONTEXT: u8 = 4;
pub const NUM_EDGE_TYPES: usize = 5;

/// The six axial hex-neighbor directions.
const HEX_DIRS: [(i16, i16); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

fn axis_index(axis: Axis) -> u8 {
    match axis {
        Axis::Q => 0,
        Axis::R => 1,
        Axis::QR => 2,
    }
}

/// Iterate every coordinate within `radius` hex steps of `center` (cube diamond).
/// Re-derived locally because `coords_within_radius` is not exported by the engine.
fn coords_within_radius(center: HexCoord, radius: i16) -> Vec<HexCoord> {
    let mut out = Vec::new();
    for dq in -radius..=radius {
        let r_min = (-radius).max(-dq - radius);
        let r_max = radius.min(-dq + radius);
        for dr in r_min..=r_max {
            out.push(HexCoord {
                q: center.q + dq,
                r: center.r + dr,
            });
        }
    }
    out
}

/// True when some length-6 line through `c` is ENTIRELY empty — i.e. `c` still
/// has an OPEN/completable line and is NOT boxed in entirely by blocked (both-
/// color, uncompletable) lines. Used by the n-radius dead-cell exclusion: a
/// candidate is kept if it is in an active window OR has any open line; it is
/// dropped only when every line through it is blocked. Scans the board directly
/// (the three axes Q/R/QR), so it needs no canonical window keys.
fn has_open_window(state: &RustHexoState, c: HexCoord) -> bool {
    const WIN: i16 = 6; // length-6 windows (fixed game constant)
    let board = state.board();
    for axis in Axis::ALL {
        let v = axis.vector();
        // The six windows containing `c` along this axis place `c` at offset `o`
        // (0..6); the window's cells are `c + (k - o) * v` for k in 0..6.
        for o in 0..WIN {
            let mut all_empty = true;
            for k in 0..WIN {
                if !board.is_cell_empty(c + v.scale(k - o)) {
                    all_empty = false;
                    break;
                }
            }
            if all_empty {
                return true;
            }
        }
    }
    false
}

/// One count-3/4/5 active-window token (§5).
pub struct WindowToken {
    pub owner_is_current: bool,
    pub count: u8,
    pub axis: Axis,
    pub anchor: HexCoord,
    pub empty_count: u8,
    /// Stone cells (of the active player) inside this window.
    pub stone_cells: Vec<HexCoord>,
    /// Empty cells inside this window (candidate-edge endpoints).
    pub empty_cells: Vec<HexCoord>,
}

/// The bounded typed graph for one position.
pub struct PositionGraph {
    // Parallel per-node arrays (node 0 is always the SIDE hub).
    pub node_type: Vec<u8>,
    pub node_q: Vec<i16>,
    pub node_r: Vec<i16>,
    pub node_owner: Vec<i8>, // 0 current, 1 opponent, -1 none/side
    pub node_recency: Vec<i32>, // stone placement_index, else -1
    pub node_wcount: Vec<u8>, // window count (3/4/5), else 0
    // UNUSED(2026-06-12): no reader found in packages/tests/scripts — neither
    // featurizer (features.py / features.rs) consumes node_waxis (axis labels are
    // deliberately excluded for D6 invariance), and the hexgnn fork dropped the
    // column entirely. Built + serialized into every graph_facts payload for
    // nothing; kept to avoid changing the payload shape on the halted lineage.
    pub node_waxis: Vec<i8>, // window axis index, else -1
    pub node_wempty: Vec<u8>, // window empty-cell count, else 0
    // Candidate node indices, in deterministic (q, r) order == CSR/legal order.
    pub candidate_nodes: Vec<u32>,
    // Candidate packed action ids, aligned 1:1 with `candidate_nodes`.
    pub candidate_ids: Vec<u32>,
    // Directed edges (each structural edge emitted in both directions, same type).
    pub edge_src: Vec<u32>,
    pub edge_dst: Vec<u32>,
    pub edge_type: Vec<u8>,
    pub edge_counts: [usize; NUM_EDGE_TYPES],
    pub used_legal_fallback: bool,
}

impl PositionGraph {
    pub fn node_count(&self) -> usize {
        self.node_type.len()
    }
    pub fn stone_count(&self) -> usize {
        self.node_type.iter().filter(|&&t| t == NODE_STONE).count()
    }
    pub fn candidate_count(&self) -> usize {
        self.candidate_nodes.len()
    }
    pub fn window_count(&self) -> usize {
        self.node_type.iter().filter(|&&t| t == NODE_WINDOW).count()
    }
}

/// Component (A) ∪ (B): the candidate empty-cell set, deduped and (q,r)-sorted.
/// Falls back to the engine legal moves only if the union is empty on a
/// non-terminal position (should be the pre-opening empty board only).
fn candidate_cells(state: &RustHexoState, n: i16) -> (Vec<HexCoord>, bool) {
    let board = state.board();
    let mut set: std::collections::BTreeSet<(i16, i16)> = std::collections::BTreeSet::new();

    // (A) empty cells of every ACTIVE window (either color, any count >= 1).
    let mut active_empty: std::collections::BTreeSet<(i16, i16)> = std::collections::BTreeSet::new();
    for entry in board.windows().entries() {
        if entry.is_active() {
            for cell in entry.empty_cells() {
                active_empty.insert((cell.q, cell.r));
            }
        }
    }
    set.extend(active_empty.iter().copied());
    // (B) empty cells within hex-distance n of any stone, EXCLUDING "dead" cells:
    // a cell is dead when EVERY length-6 line through it is BLOCKED (both colors,
    // uncompletable). Such a cell can never start or complete a line, so it is a
    // useless move and dropped. It is KEPT if it is in an active window (already
    // a live line) OR has any open/completable line (`has_open_window`).
    for &stone in board.occupied_cells() {
        for cell in coords_within_radius(stone, n) {
            if !board.is_cell_empty(cell) {
                continue;
            }
            let key = (cell.q, cell.r);
            if active_empty.contains(&key) || has_open_window(state, cell) {
                set.insert(key);
            }
            // else: boxed in entirely by blocked lines -> dead -> dropped
        }
    }

    let mut used_fallback = false;
    if set.is_empty() && !state.is_terminal() {
        // Safety net (§4.4): never diverge silently from legality.
        used_fallback = true;
        let mut legal = Vec::with_capacity(state.legal_move_count());
        state.write_legal_moves(&mut legal);
        for cell in legal {
            set.insert((cell.q, cell.r));
        }
    }

    let cells = set.into_iter().map(|(q, r)| HexCoord { q, r }).collect();
    (cells, used_fallback)
}

/// The count-3/4/5 active-window tokens of both colors (§5), deterministically
/// ordered by (axis, start.q, start.r).
fn window_tokens(state: &RustHexoState) -> Vec<WindowToken> {
    let current = state.current_player();
    let mut tokens: Vec<WindowToken> = Vec::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        let count = entry.count(owner);
        if !(3..=5).contains(&count) {
            continue;
        }
        let key = entry.key();
        tokens.push(WindowToken {
            owner_is_current: owner == current,
            count,
            axis: key.axis,
            anchor: key.start,
            empty_count: entry.empty_mask().count_ones() as u8,
            stone_cells: entry.stone_cells(owner),
            empty_cells: entry.empty_cells(),
        });
    }
    tokens.sort_by_key(|t| (axis_index(t.axis), t.anchor.q, t.anchor.r));
    tokens
}

/// Build the full bounded typed graph for `state` at radius `n`.
pub fn build_graph(state: &RustHexoState, n: i16) -> PositionGraph {
    let board = state.board();
    let current = state.current_player();

    let (cands, used_legal_fallback) = candidate_cells(state, n);
    let tokens = window_tokens(state);

    // --- nodes ---------------------------------------------------------------
    let mut node_type = Vec::new();
    let mut node_q = Vec::new();
    let mut node_r = Vec::new();
    let mut node_owner = Vec::new();
    let mut node_recency = Vec::new();
    let mut node_wcount = Vec::new();
    let mut node_waxis = Vec::new();
    let mut node_wempty = Vec::new();

    let mut push_node =
        |t: u8, c: HexCoord, owner: i8, recency: i32, wc: u8, wa: i8, we: u8| -> u32 {
            node_type.push(t);
            node_q.push(c.q);
            node_r.push(c.r);
            node_owner.push(owner);
            node_recency.push(recency);
            node_wcount.push(wc);
            node_waxis.push(wa);
            node_wempty.push(we);
            (node_type.len() - 1) as u32
        };

    // node 0: SIDE hub.
    let side_idx = push_node(NODE_SIDE, HexCoord::ZERO, -1, -1, 0, -1, 0);

    // STONE nodes (sorted by (q,r)); record recency from placement_history.
    let mut recency_of: HashMap<(i16, i16), i32> = HashMap::new();
    for rec in state.placement_history() {
        recency_of.insert((rec.coord.q, rec.coord.r), rec.placement_index as i32);
    }
    let mut occupied: Vec<HexCoord> = board.occupied_cells().to_vec();
    occupied.sort_by_key(|c| (c.q, c.r));
    let mut spatial: HashMap<(i16, i16), u32> = HashMap::new();
    let mut stone_idx: HashMap<(i16, i16), u32> = HashMap::new();
    for c in &occupied {
        let owner = match board.get(*c) {
            Some(p) if p == current => 0i8,
            Some(_) => 1i8,
            None => -1i8,
        };
        let rec = *recency_of.get(&(c.q, c.r)).unwrap_or(&-1);
        let idx = push_node(NODE_STONE, *c, owner, rec, 0, -1, 0);
        spatial.insert((c.q, c.r), idx);
        stone_idx.insert((c.q, c.r), idx);
    }

    // CANDIDATE nodes (already (q,r)-sorted); CSR order == this order.
    let mut candidate_nodes = Vec::with_capacity(cands.len());
    let mut candidate_ids = Vec::with_capacity(cands.len());
    let mut cand_idx: HashMap<(i16, i16), u32> = HashMap::new();
    for c in &cands {
        let idx = push_node(NODE_CANDIDATE, *c, -1, -1, 0, -1, 0);
        candidate_nodes.push(idx);
        candidate_ids.push(pack_coord(*c));
        spatial.insert((c.q, c.r), idx);
        cand_idx.insert((c.q, c.r), idx);
    }

    // WINDOW nodes.
    let mut window_node_idx = Vec::with_capacity(tokens.len());
    for t in &tokens {
        let owner = if t.owner_is_current { 0 } else { 1 };
        let idx = push_node(
            NODE_WINDOW,
            t.anchor,
            owner,
            -1,
            t.count,
            axis_index(t.axis) as i8,
            t.empty_count,
        );
        window_node_idx.push(idx);
    }

    // --- edges (directed; emit both directions per structural edge) ----------
    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    let mut edge_type = Vec::new();
    let mut edge_counts = [0usize; NUM_EDGE_TYPES];
    let mut add_undirected = |a: u32, b: u32, ty: u8| {
        edge_src.push(a);
        edge_dst.push(b);
        edge_type.push(ty);
        edge_src.push(b);
        edge_dst.push(a);
        edge_type.push(ty);
        edge_counts[ty as usize] += 2;
    };

    // CONTEXT: SIDE hub <-> every other node.
    let total_nodes = node_type.len() as u32;
    for idx in 0..total_nodes {
        if idx != side_idx {
            add_undirected(side_idx, idx, EDGE_CONTEXT);
        }
    }

    // RECENCY: consecutive stones in placement order (chain).
    let mut hist: Vec<&hexo_engine::PlacementRecord> = state.placement_history().iter().collect();
    hist.sort_by_key(|r| r.placement_index);
    for pair in hist.windows(2) {
        let a = stone_idx.get(&(pair[0].coord.q, pair[0].coord.r));
        let b = stone_idx.get(&(pair[1].coord.q, pair[1].coord.r));
        if let (Some(&a), Some(&b)) = (a, b) {
            add_undirected(a, b, EDGE_RECENCY);
        }
    }

    // STONE<->WINDOW and CANDIDATE<->WINDOW membership (window hub, <=6 each).
    for (ti, t) in tokens.iter().enumerate() {
        let widx = window_node_idx[ti];
        for sc in &t.stone_cells {
            if let Some(&s) = stone_idx.get(&(sc.q, sc.r)) {
                add_undirected(widx, s, EDGE_STONE_WINDOW);
            }
        }
        for ec in &t.empty_cells {
            if let Some(&c) = cand_idx.get(&(ec.q, ec.r)) {
                add_undirected(widx, c, EDGE_CANDIDATE_WINDOW);
            }
        }
    }

    // ADJACENCY: spatial nodes (stones + candidates) within hex-distance 1.
    // Emit once per unordered pair (when neighbor index > self) to avoid dupes.
    let mut spatial_nodes: Vec<(HexCoord, u32)> = spatial
        .iter()
        .map(|(&(q, r), &idx)| (HexCoord { q, r }, idx))
        .collect();
    spatial_nodes.sort_by_key(|(_, idx)| *idx);
    for (coord, idx) in &spatial_nodes {
        for (dq, dr) in HEX_DIRS {
            let nb = (coord.q + dq, coord.r + dr);
            if let Some(&nidx) = spatial.get(&nb) {
                if nidx > *idx {
                    add_undirected(*idx, nidx, EDGE_ADJACENCY);
                }
            }
        }
    }

    PositionGraph {
        node_type,
        node_q,
        node_r,
        node_owner,
        node_recency,
        node_wcount,
        node_waxis,
        node_wempty,
        candidate_nodes,
        candidate_ids,
        edge_src,
        edge_dst,
        edge_type,
        edge_counts,
        used_legal_fallback,
    }
}

/// Pack a built `PositionGraph` + its state context into the Python graph-facts
/// dict consumed by `features.build_graph_tensors`. Shared by the PyO3
/// `hexgt_graph_facts` entry (Py state in) and the MCTS eval (Rust leaf states,
/// no Py->Rust reclone), so training inputs == search inputs byte-for-byte.
pub(crate) fn position_graph_to_py_dict(
    py: Python<'_>,
    state: &RustHexoState,
    g: &PositionGraph,
    n: i64,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("n", n)?;
    dict.set_item("used_legal_fallback", g.used_legal_fallback)?;

    // Global context for the SIDE node + sample targets.
    let meta = PyDict::new(py);
    let phase_idx: u8 = match state.phase() {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { .. } => 2,
    };
    // This turn's first-stone coord on the SECOND placement (else None) — the
    // Python featurizer needs it for F_CAND_DIST_FIRST (Rust gets it from the
    // state directly). None on the turn's first placement.
    let first_stone: Option<(i16, i16)> = match state.phase() {
        TurnPhase::SecondStone { first } => Some((first.q, first.r)),
        _ => None,
    };
    meta.set_item("phase", phase_idx)?;
    meta.set_item("placements", state.placements_made())?;
    meta.set_item("current_player", state.current_player().index())?;
    meta.set_item("is_terminal", state.is_terminal())?;
    meta.set_item("first_stone", first_stone)?;
    dict.set_item("meta", meta)?;

    let counts = PyDict::new(py);
    counts.set_item("side", 1)?;
    counts.set_item("stone", g.stone_count())?;
    counts.set_item("candidate", g.candidate_count())?;
    counts.set_item("window", g.window_count())?;
    counts.set_item("total_nodes", g.node_count())?;
    dict.set_item("node_counts", counts)?;

    let ecounts = PyDict::new(py);
    ecounts.set_item("adjacency", g.edge_counts[EDGE_ADJACENCY as usize])?;
    ecounts.set_item("stone_window", g.edge_counts[EDGE_STONE_WINDOW as usize])?;
    ecounts.set_item("candidate_window", g.edge_counts[EDGE_CANDIDATE_WINDOW as usize])?;
    ecounts.set_item("recency", g.edge_counts[EDGE_RECENCY as usize])?;
    ecounts.set_item("context", g.edge_counts[EDGE_CONTEXT as usize])?;
    ecounts.set_item("total_edges", g.edge_src.len())?;
    dict.set_item("edge_counts", ecounts)?;

    dict.set_item("candidate_ids", g.candidate_ids.clone())?;
    dict.set_item("candidate_nodes", g.candidate_nodes.clone())?;

    // Typed node arrays.
    let nodes = PyDict::new(py);
    nodes.set_item("node_type", g.node_type.clone())?;
    nodes.set_item("node_q", g.node_q.clone())?;
    nodes.set_item("node_r", g.node_r.clone())?;
    nodes.set_item("node_owner", g.node_owner.clone())?;
    nodes.set_item("node_recency", g.node_recency.clone())?;
    nodes.set_item("node_wcount", g.node_wcount.clone())?;
    // UNUSED(2026-06-12): no Python consumer reads "node_waxis" (see the field
    // declaration above for the grep evidence).
    nodes.set_item("node_waxis", g.node_waxis.clone())?;
    nodes.set_item("node_wempty", g.node_wempty.clone())?;
    dict.set_item("nodes", nodes)?;

    // Typed edge arrays.
    let edges = PyDict::new(py);
    edges.set_item("edge_src", g.edge_src.clone())?;
    edges.set_item("edge_dst", g.edge_dst.clone())?;
    edges.set_item("edge_type", g.edge_type.clone())?;
    dict.set_item("edges", edges)?;

    // Window tokens (owner_is_current, count, axis_index, anchor_q, anchor_r, empty_count).
    let toks = PyList::empty(py);
    for t in window_tokens(state) {
        toks.append((
            t.owner_is_current,
            t.count,
            axis_index(t.axis),
            t.anchor.q,
            t.anchor.r,
            t.empty_count,
        ))?;
    }
    dict.set_item("window_tokens", toks)?;

    Ok(dict.into_any().unbind())
}

// --- PyO3 surface ------------------------------------------------------------

/// Candidate packed action ids for `state` at radius `n` (deterministic order).
#[pyfunction]
#[pyo3(signature = (state, n))]
pub fn hexgt_candidate_ids(py: Python<'_>, state: &Bound<'_, PyAny>, n: i64) -> PyResult<Vec<u32>> {
    let state = super::state::state_from_py_state(py, state)?;
    let (cells, _) = candidate_cells(&state, n as i16);
    Ok(cells.into_iter().map(pack_coord).collect())
}

/// Full graph facts for `state` at radius `n`: counts (for the no-explosion
/// gate), candidate ids/order, window tokens, and the typed node/edge arrays
/// (reused by the Phase-4 expand step).
#[pyfunction]
#[pyo3(signature = (state, n))]
pub fn hexgt_graph_facts(py: Python<'_>, state: &Bound<'_, PyAny>, n: i64) -> PyResult<Py<PyAny>> {
    let state = super::state::state_from_py_state(py, state)?;
    let g = build_graph(&state, n as i16);
    position_graph_to_py_dict(py, &state, &g, n)
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexgt_candidate_ids, module)?)?;
    module.add_function(wrap_pyfunction!(hexgt_graph_facts, module)?)?;
    Ok(())
}
