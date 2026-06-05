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
//! with `n` the single tunable radius (default 2, range [2, 8]). A radius-n cell
//! is "dead" (and dropped) only when EVERY length-6 line through it is BLOCKED
//! (contains both colors -> uncompletable); a cell on any active or open line is
//! kept. Dead cells are useless moves; in practice they are rare (mostly dense
//! endgames) and carry ~0 of a strong player's visit mass (validated).
//!
//! Tactical-window tokens (§5): the count-3/4/5 active windows of both colors.
//!
//! Bounded typed edges (§6.3). SPARSE REWRITE: the WINDOW nodes and their
//! STONE_WINDOW / CANDIDATE_WINDOW hub edges have been REMOVED — window-count
//! information is folded into per-node FEATURES (computed here from
//! `window_tokens`, encoded in features.rs / features.py). Only ADJACENCY (+ the
//! per-edge `edge_dir` index) and RECENCY edges remain; the SIDE node is
//! edge-isolated. Total edges are linear in #nodes.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use hexo_engine::{pack_coord, Axis, HexCoord, HexoState as RustHexoState, TurnPhase};

// Node type ids (mirror of hexgnn/python/.../constants.py).
pub const NODE_SIDE: u8 = 0;
pub const NODE_STONE: u8 = 1;
pub const NODE_CANDIDATE: u8 = 2;
// NODE_WINDOW (=3) RETIRED in the sparse rewrite (window nodes never emitted); the
// id is reserved (NUM_NODE_TYPES stays 4) so the type one-hot width is unchanged.

// Edge type ids (mirror of constants.py). STONE_WINDOW/CANDIDATE_WINDOW/CONTEXT are
// RETIRED (never emitted) but keep their ids so NUM_EDGE_TYPES / the edge-type
// one-hot width / the relational weight shape are byte-identical to pre-rewrite.
pub const EDGE_ADJACENCY: u8 = 0;
pub const EDGE_RECENCY: u8 = 3;
pub const NUM_EDGE_TYPES: usize = 5;

/// The six axial hex-neighbor directions.
const HEX_DIRS: [(i16, i16); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

/// Canonical hex-direction order for the per-edge `edge_dir` index. This MUST
/// EXACTLY MATCH the Python steerable layer's basis
/// `_HEX_DIRS_AXIAL = ((1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1))`: for a directed
/// adjacency edge A->B, `edge_dir` is the index `d` with
/// `(B.q-A.q, B.r-A.r) == DIR_ORDER[d]`. Independent of `HEX_DIRS` ordering.
const DIR_ORDER: [(i16, i16); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

/// Index of the axial delta `(dq, dr)` in `DIR_ORDER`, or `-1` if it is not a
/// unit hex step (i.e. not one of the six neighbor directions).
fn dir_index(dq: i16, dr: i16) -> i8 {
    for (i, &(cq, cr)) in DIR_ORDER.iter().enumerate() {
        if cq == dq && cr == dr {
            return i as i8;
        }
    }
    -1
}

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

/// Per-node window-count facts derived from the active-window tokens (sparse
/// rewrite: these replace the removed STONE_WINDOW / CANDIDATE_WINDOW hub edges).
/// Parallel to the per-node arrays. For CANDIDATE nodes they reproduce EXACTLY the
/// values the old candidate<->window edge loop accumulated; for STONE nodes the
/// per-owner nwin counts are the new per-stone window features. All zero on the
/// SIDE node (and unused fields stay zero on stones / candidates respectively).
#[derive(Clone, Default)]
pub struct WinCounts {
    // Candidate accumulators (per empty cell, over the windows it belongs to):
    pub nwin_own: u16,
    pub nwin_opp: u16,
    pub own_win3: u16,
    pub own_win4: u16,
    pub own_win5: u16,
    pub opp_win3: u16,
    pub opp_win4: u16,
    pub opp_win5: u16,
    pub complete_own: bool, // any own count-5 window through cell
    pub complete_opp: bool, // any opp count-5 window through cell
    pub opp_threat: bool,   // any opp count>=4 window through cell
    // Stone accumulators (per stone cell): active windows of each owner through it.
    pub stone_nwin_own: u16,
    pub stone_nwin_opp: u16,
}

/// The bounded typed graph for one position.
pub struct PositionGraph {
    // Parallel per-node arrays (node 0 is always the SIDE hub).
    pub node_type: Vec<u8>,
    pub node_q: Vec<i16>,
    pub node_r: Vec<i16>,
    pub node_owner: Vec<i8>, // 0 current, 1 opponent, -1 none/side
    pub node_recency: Vec<i32>, // stone placement_index, else -1
    // Per-node window-count facts (sparse rewrite; one entry per node). Folds the
    // removed window-hub edges into per-node features (see WinCounts).
    pub node_wins: Vec<WinCounts>,
    // Candidate node indices, in deterministic (q, r) order == CSR/legal order.
    pub candidate_nodes: Vec<u32>,
    // Candidate packed action ids, aligned 1:1 with `candidate_nodes`.
    pub candidate_ids: Vec<u32>,
    // Directed edges (each structural edge emitted in both directions, same type).
    pub edge_src: Vec<u32>,
    pub edge_dst: Vec<u32>,
    pub edge_type: Vec<u8>,
    // Per-directed-edge hex-direction index in DIR_ORDER for ADJACENCY edges
    // (which of the 6 axial directions src->dst points), else -1. Parallel to
    // edge_src/edge_dst/edge_type (same length, same order).
    pub edge_dir: Vec<i8>,
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

    let mut push_node = |t: u8, c: HexCoord, owner: i8, recency: i32| -> u32 {
        node_type.push(t);
        node_q.push(c.q);
        node_r.push(c.r);
        node_owner.push(owner);
        node_recency.push(recency);
        (node_type.len() - 1) as u32
    };

    // node 0: SIDE hub.
    let side_idx = push_node(NODE_SIDE, HexCoord::ZERO, -1, -1);

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
        let idx = push_node(NODE_STONE, *c, owner, rec);
        spatial.insert((c.q, c.r), idx);
        stone_idx.insert((c.q, c.r), idx);
    }

    // CANDIDATE nodes (already (q,r)-sorted); CSR order == this order.
    let mut candidate_nodes = Vec::with_capacity(cands.len());
    let mut candidate_ids = Vec::with_capacity(cands.len());
    let mut cand_idx: HashMap<(i16, i16), u32> = HashMap::new();
    for c in &cands {
        let idx = push_node(NODE_CANDIDATE, *c, -1, -1);
        candidate_nodes.push(idx);
        candidate_ids.push(pack_coord(*c));
        spatial.insert((c.q, c.r), idx);
        cand_idx.insert((c.q, c.r), idx);
    }

    let num_nodes = node_type.len();

    // --- per-node window-count features (sparse rewrite: replaces the removed
    // STONE_WINDOW / CANDIDATE_WINDOW hub edges). For each active-window token we
    // visit its empty cells (candidate accumulators) and its stone cells (per-stone
    // accumulators). The candidate accumulation reproduces EXACTLY what the old
    // window->candidate edge loop did (same per-count increments + complete/threat
    // flags), so the encoded candidate features are byte-identical; the per-stone
    // own/opp nwin counts are the NEW per-stone window features.
    let mut node_wins = vec![WinCounts::default(); num_nodes];
    for t in &tokens {
        for ec in &t.empty_cells {
            if let Some(&ci) = cand_idx.get(&(ec.q, ec.r)) {
                let w = &mut node_wins[ci as usize];
                if t.owner_is_current {
                    w.nwin_own += 1;
                    match t.count {
                        3 => w.own_win3 += 1,
                        4 => w.own_win4 += 1,
                        5 => w.own_win5 += 1,
                        _ => {}
                    }
                    if t.count == 5 {
                        w.complete_own = true;
                    }
                } else {
                    w.nwin_opp += 1;
                    match t.count {
                        3 => w.opp_win3 += 1,
                        4 => w.opp_win4 += 1,
                        5 => w.opp_win5 += 1,
                        _ => {}
                    }
                    if t.count == 5 {
                        w.complete_opp = true;
                    }
                    if t.count >= 4 {
                        w.opp_threat = true;
                    }
                }
            }
        }
        for sc in &t.stone_cells {
            if let Some(&si) = stone_idx.get(&(sc.q, sc.r)) {
                let w = &mut node_wins[si as usize];
                if t.owner_is_current {
                    w.stone_nwin_own += 1;
                } else {
                    w.stone_nwin_opp += 1;
                }
            }
        }
    }

    // --- edges (directed; emit both directions per structural edge) ----------
    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    let mut edge_type = Vec::new();
    let mut edge_dir = Vec::new();
    let mut edge_counts = [0usize; NUM_EDGE_TYPES];
    // Push both directed edges of a structural edge, recording a direction index
    // for each (adjacency: the real DIR_ORDER index of the a->b / b->a delta;
    // all other edge types: -1).
    let mut add_undirected_dir = |a: u32, b: u32, ty: u8, dir_ab: i8, dir_ba: i8| {
        edge_src.push(a);
        edge_dst.push(b);
        edge_type.push(ty);
        edge_dir.push(dir_ab);
        edge_src.push(b);
        edge_dst.push(a);
        edge_type.push(ty);
        edge_dir.push(dir_ba);
        edge_counts[ty as usize] += 2;
    };

    // CONTEXT hub REMOVED (hexgnn sparse rewrite, phase 1; design #4: no global
    // hub). The SIDE hub <-> every-node fan-out was 2*(N-1) edges (~27% of midgame
    // edges) and memory-bound scatter dominated the forward. Per design #4 the
    // global scalars (phase / move-number / stone counts) ride on the SIDE node's
    // OWN features and feed the value head directly (the value readout still reads
    // the SIDE row + the PMA pool over all nodes), so no per-node global broadcast
    // edge is materialized. D6-INVARIANT (removing an invariant edge class). The
    // SIDE node is now edge-isolated; the message passing leaves an edgeless node's
    // embedding = norm(node_in(side_features)), exactly the intended "global scalars
    // -> value head" path. `side_idx` retained for the value readout.
    let _ = side_idx;

    // RECENCY: consecutive stones in placement order (chain).
    let mut hist: Vec<&hexo_engine::PlacementRecord> = state.placement_history().iter().collect();
    hist.sort_by_key(|r| r.placement_index);
    for pair in hist.windows(2) {
        let a = stone_idx.get(&(pair[0].coord.q, pair[0].coord.r));
        let b = stone_idx.get(&(pair[1].coord.q, pair[1].coord.r));
        if let (Some(&a), Some(&b)) = (a, b) {
            add_undirected_dir(a, b, EDGE_RECENCY, -1, -1);
        }
    }

    // STONE<->WINDOW and CANDIDATE<->WINDOW membership edges REMOVED (sparse
    // rewrite): the window nodes are gone and the membership information is folded
    // into the per-node `node_wins` features above. `tokens` is still consumed (for
    // those features and the py-dict window_tokens list); the candidate SET is
    // unchanged.

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
                    // src=*idx (coord), dst=nidx (nb): a->b delta is (dq,dr),
                    // b->a delta is (-dq,-dr). Indices into DIR_ORDER.
                    let dir_ab = dir_index(dq, dr);
                    let dir_ba = dir_index(-dq, -dr);
                    add_undirected_dir(*idx, nidx, EDGE_ADJACENCY, dir_ab, dir_ba);
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
        node_wins,
        candidate_nodes,
        candidate_ids,
        edge_src,
        edge_dst,
        edge_type,
        edge_dir,
        edge_counts,
        used_legal_fallback,
    }
}

/// Pack a built `PositionGraph` + its state context into the Python graph-facts
/// dict consumed by `features.build_graph_tensors`. Shared by the PyO3
/// `hexgnn_graph_facts` entry (Py state in) and the MCTS eval (Rust leaf states,
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
    // window NODES removed (sparse rewrite): always 0 (kept for dict-shape compat).
    counts.set_item("window", 0)?;
    counts.set_item("total_nodes", g.node_count())?;
    dict.set_item("node_counts", counts)?;

    let ecounts = PyDict::new(py);
    ecounts.set_item("adjacency", g.edge_counts[EDGE_ADJACENCY as usize])?;
    // STONE_WINDOW / CANDIDATE_WINDOW / CONTEXT edges removed: always 0.
    ecounts.set_item("stone_window", 0)?;
    ecounts.set_item("candidate_window", 0)?;
    ecounts.set_item("recency", g.edge_counts[EDGE_RECENCY as usize])?;
    ecounts.set_item("context", 0)?;
    ecounts.set_item("total_edges", g.edge_src.len())?;
    dict.set_item("edge_counts", ecounts)?;

    dict.set_item("candidate_ids", g.candidate_ids.clone())?;
    dict.set_item("candidate_nodes", g.candidate_nodes.clone())?;

    // Typed node arrays. The former window-node columns (node_wcount/waxis/wempty)
    // are GONE; instead the per-node window-count features (sparse rewrite) are
    // exposed as parallel per-node arrays the Python featurizer reads directly
    // (replacing the old candidate<->window edge accumulation). Candidate columns
    // reproduce the old edge-loop values exactly; the stone_nwin_* columns are the
    // new per-stone window features.
    let nodes = PyDict::new(py);
    nodes.set_item("node_type", g.node_type.clone())?;
    nodes.set_item("node_q", g.node_q.clone())?;
    nodes.set_item("node_r", g.node_r.clone())?;
    nodes.set_item("node_owner", g.node_owner.clone())?;
    nodes.set_item("node_recency", g.node_recency.clone())?;
    nodes.set_item(
        "node_nwin_own",
        g.node_wins.iter().map(|w| w.nwin_own).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_nwin_opp",
        g.node_wins.iter().map(|w| w.nwin_opp).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_own_win3",
        g.node_wins.iter().map(|w| w.own_win3).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_own_win4",
        g.node_wins.iter().map(|w| w.own_win4).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_own_win5",
        g.node_wins.iter().map(|w| w.own_win5).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_opp_win3",
        g.node_wins.iter().map(|w| w.opp_win3).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_opp_win4",
        g.node_wins.iter().map(|w| w.opp_win4).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_opp_win5",
        g.node_wins.iter().map(|w| w.opp_win5).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_complete_own",
        g.node_wins.iter().map(|w| w.complete_own).collect::<Vec<bool>>(),
    )?;
    nodes.set_item(
        "node_complete_opp",
        g.node_wins.iter().map(|w| w.complete_opp).collect::<Vec<bool>>(),
    )?;
    nodes.set_item(
        "node_opp_threat",
        g.node_wins.iter().map(|w| w.opp_threat).collect::<Vec<bool>>(),
    )?;
    nodes.set_item(
        "node_stone_nwin_own",
        g.node_wins.iter().map(|w| w.stone_nwin_own).collect::<Vec<u16>>(),
    )?;
    nodes.set_item(
        "node_stone_nwin_opp",
        g.node_wins.iter().map(|w| w.stone_nwin_opp).collect::<Vec<u16>>(),
    )?;
    dict.set_item("nodes", nodes)?;

    // Typed edge arrays.
    let edges = PyDict::new(py);
    edges.set_item("edge_src", g.edge_src.clone())?;
    edges.set_item("edge_dst", g.edge_dst.clone())?;
    edges.set_item("edge_type", g.edge_type.clone())?;
    // Per-directed-edge hex-direction index (adjacency: 0..5; else -1).
    edges.set_item("edge_dir", g.edge_dir.clone())?;
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
pub fn hexgnn_candidate_ids(py: Python<'_>, state: &Bound<'_, PyAny>, n: i64) -> PyResult<Vec<u32>> {
    let state = super::state::state_from_py_state(py, state)?;
    let (cells, _) = candidate_cells(&state, n as i16);
    Ok(cells.into_iter().map(pack_coord).collect())
}

/// Full graph facts for `state` at radius `n`: counts (for the no-explosion
/// gate), candidate ids/order, window tokens, and the typed node/edge arrays
/// (reused by the Phase-4 expand step).
#[pyfunction]
#[pyo3(signature = (state, n))]
pub fn hexgnn_graph_facts(py: Python<'_>, state: &Bound<'_, PyAny>, n: i64) -> PyResult<Py<PyAny>> {
    let state = super::state::state_from_py_state(py, state)?;
    let g = build_graph(&state, n as i16);
    position_graph_to_py_dict(py, &state, &g, n)
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexgnn_candidate_ids, module)?)?;
    module.add_function(wrap_pyfunction!(hexgnn_graph_facts, module)?)?;
    Ok(())
}
