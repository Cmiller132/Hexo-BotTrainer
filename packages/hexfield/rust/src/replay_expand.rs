//! Train-read row expansion kernel (Rust + rayon, GIL-free).
//!
//! Per row: a D6 transform of every stored coordinate fact, a depth-(radius+1)
//! multi-source BFS support build, the feature build, and the legal-slot policy
//! projection. Runs across `facts.par_iter()` under `py.detach` (GIL released),
//! order-preserving via `collect`, exposing the stacked result as zero-copy
//! buffers consumed Python-side.
//!
//! This mirrors the Python expansion chain (`support.py::_build_support`,
//! `features.py::build_features`, `samples.py::expand_sample`/`_legal_slot`,
//! `geometry.py::apply_d6`), not the serve-time `support.rs`/`features.rs`. The
//! train path has only the stored `hexfield_compact_v1` facts: the unified
//! placement history, the phase, the first stone, and the pre-computed hot /
//! standing-win cell lists. Legality is derived in closed form
//! (`empty ∧ dist <= radius`); the hot/win cells are read from the shard and
//! D6-transformed (`transform_facts`), not recomputed from windows. The
//! element-wise parity test across all 12 D6 values and off-legal radius-4 rows
//! is `tests/katago_buffer/test_p7_rust_parity.py`.
//!
//! Off-legal handling: an off-legal SELF policy target flags the row invalid in
//! the returned `valid` mask (not dropped in-worker) when `tolerate_off_legal`;
//! otherwise it is a hard error. The caller filters survivors / permutes /
//! truncates on the main thread.
//!
//! Determinism: the per-row `d6: i32[n]` vector is pre-drawn on the main thread
//! and passed positionally; the kernel makes no rng call. `par_iter().collect()`
//! preserves input order, so the output does not depend on worker count.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;

use crate::constants::{
    DIRECTIONS, DIST_SCALE, F_DIST_TO_STONE, F_EMPTY, F_FIRST_STONE, F_LEGAL, F_OPP_HOT,
    F_OPP_LAST_TURN, F_OPP_RECENCY, F_OPP_STONE, F_OPP_WIN_NOW, F_OWN_HOT, F_OWN_RECENCY,
    F_OWN_STONE, F_OWN_WIN_NOW, F_PHASE_SECOND, F_PLAYER_COLOUR, NUM_FEATURES,
};

// hexfield_compact_v1 phase enum: index 2 == "SecondStone".
const PHASE_SECOND_STONE: u8 = 2;
// moves_left is normalized to [-1, 1] over [0, MOVES_LEFT_CAP]. Must match the
// Python constant.
const MOVES_LEFT_CAP: f32 = 209.0;
// Packed action id = ((q+2^15)<<16) | (r+2^15).
const COORD_OFFSET: i32 = 1 << 15;

// =============================================================================
// Geometry (mirrors geometry.py) — integer math, no floats.
// =============================================================================

/// `geometry.rot60`: (-r, q+r).
#[inline]
fn rot60(q: i32, r: i32) -> (i32, i32) {
    (-r, q + r)
}

/// `geometry.reflect`: (q, -q-r).
#[inline]
fn reflect(q: i32, r: i32) -> (i32, i32) {
    (q, -q - r)
}

/// `geometry.apply_d6`: index 0-5 == rot60^i; 6-11 == rot60^(i-6) ∘ reflect.
/// Reflect first when index>=6, then rotate.
#[inline]
fn apply_d6(index: i32, q: i32, r: i32) -> (i32, i32) {
    let (mut q, mut r) = (q, r);
    let mut idx = index;
    if idx >= 6 {
        let (nq, nr) = reflect(q, r);
        q = nq;
        r = nr;
        idx -= 6;
    }
    for _ in 0..idx {
        let (nq, nr) = rot60(q, r);
        q = nq;
        r = nr;
    }
    (q, r)
}

/// `geometry.unpack_action_id`: inverse of pack_action_id.
#[inline]
fn unpack_action_id(action_id: u32) -> (i32, i32) {
    let q = ((action_id >> 16) & 0xFFFF) as i32 - COORD_OFFSET;
    let r = (action_id & 0xFFFF) as i32 - COORD_OFFSET;
    (q, r)
}

// =============================================================================
// Per-row stored facts, copied out of the byte buffers on the main thread so
// workers own their data (no borrow of the source buffers inside par_iter).
// =============================================================================

struct RowFacts {
    // (q, r, owner, placement_index)
    records: Vec<(i32, i32, u8, u32)>,
    current_player: u8,
    phase: u8,
    first_stone: Option<(i32, i32)>,
    own_hot: Vec<(i32, i32)>,
    opp_hot: Vec<(i32, i32)>,
    own_win: Vec<(i32, i32)>,
    opp_win: Vec<(i32, i32)>,
    // (action_id, weight)
    policy: Vec<(u32, f32)>,
    q_policy: Vec<(u32, f32)>,
    opp_policy: Vec<(u32, f32)>,
    policy_surprise: f32,
    value: f32,
    // (H,) stvalue + mask blocks
    stvalue: Vec<f32>,
    stvalue_mask: Vec<f32>,
    moves_left: f32,
    // 1 == completed game (grounded outcome), 0 == truncated (no engine winner).
    // Gates the value/stvalue/cell_q heads to zero loss.
    outcome_valid: u8,
    // 1 == full row (policy/opp/soft/cell_q), 0 == fast (value-only).
    policy_valid: u8,
}

/// One expanded row's flat arrays. Invalid rows carry zero-length node/policy
/// vecs and `valid=false`.
struct RowOut {
    valid: bool,
    legal_count: i32,
    stone_count: i32,
    halo_count: i32,
    // node-major: coords (2N), dist (N), nbr (6N), feats (NUM_FEATURES*N)
    coords: Vec<i32>,
    dist: Vec<i32>,
    nbr: Vec<i32>,
    feats: Vec<f32>,
    // legal-prefix targets (legal_count each)
    policy: Vec<f32>,
    opp_policy: Vec<f32>,
    cell_q: Vec<f32>,
    cell_q_mask: Vec<f32>,
    policy_surprise: f32,
    // opp_coverage is emitted as f64: it is an f64 ratio of f64-accumulated sums,
    // matching the Python float computation. Other scalars stay f32.
    opp_coverage: f64,
    value: f32,
    value_mask: f32,
    policy_valid: f32,
    stvalue: Vec<f32>,
    stvalue_mask: Vec<f32>,
    moves_left: f32,
    moves_left_mask: f32,
}

// =============================================================================
// Support build (mirrors support.py::_build_support) — closed-form legality.
// =============================================================================

struct Support {
    /// [legal | stones | halo], each ascending by (q, r).
    coords: Vec<(i32, i32)>,
    legal_count: usize,
    stone_count: usize,
    halo_count: usize,
    dist: Vec<i32>,
    index: HashMap<(i32, i32), usize>,
}

impl Support {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.coords.len()
    }
}

/// Mirrors `support.py::_build_support(stones)`. `stones` is the D6-transformed
/// stone coordinate list (order irrelevant — deduped into a set). `radius`/`halo`
/// correspond to `_SUPPORT_RADIUS`/`_SUPPORT_HALO`.
fn build_support(stones: &[(i32, i32)], radius: i32, halo: i32) -> Support {
    if stones.is_empty() {
        // Ply 0: origin + its 6 halo neighbours (7 nodes, 1 legal), all dist 0.
        // coords = [(0,0)] + sorted(DIRECTIONS).
        let mut dirs: Vec<(i32, i32)> = DIRECTIONS.iter().map(|&(dq, dr)| (dq as i32, dr as i32)).collect();
        dirs.sort();
        let mut coords = Vec::with_capacity(7);
        coords.push((0, 0));
        coords.extend(dirs);
        let dist = vec![0i32; coords.len()];
        let index = build_index(&coords);
        return Support {
            coords,
            legal_count: 1,
            stone_count: 0,
            halo_count: 6,
            dist,
            index,
        };
    }

    // Multi-source BFS depth `halo` (== radius+1) from the stones. dist is
    // seeded from the deduped stone set, so duplicates do not double-seed.
    let mut dist_map: HashMap<(i32, i32), i32> = HashMap::with_capacity(stones.len() * 16);
    let mut frontier: std::collections::VecDeque<(i32, i32)> =
        std::collections::VecDeque::with_capacity(stones.len() * 8);
    let mut stone_set: HashMap<(i32, i32), ()> = HashMap::with_capacity(stones.len());
    for &s in stones {
        if stone_set.insert(s, ()).is_none() {
            dist_map.insert(s, 0);
            frontier.push_back(s);
        }
    }
    while let Some(cell) = frontier.pop_front() {
        let d = dist_map[&cell];
        if d == halo {
            continue;
        }
        let (q, r) = cell;
        for &(dq, dr) in &DIRECTIONS {
            let nxt = (q + dq as i32, r + dr as i32);
            if !dist_map.contains_key(&nxt) {
                dist_map.insert(nxt, d + 1);
                frontier.push_back(nxt);
            }
        }
    }

    // legal = empty ∧ dist <= radius (NOT a stone); stones = sorted set;
    // halo = dist == halo. Each segment ascending by (q, r).
    let mut legal: Vec<(i32, i32)> = dist_map
        .iter()
        .filter(|(c, &d)| d <= radius && !stone_set.contains_key(*c))
        .map(|(&c, _)| c)
        .collect();
    legal.sort();
    let mut stones_sorted: Vec<(i32, i32)> = stone_set.keys().copied().collect();
    stones_sorted.sort();
    let mut halo_cells: Vec<(i32, i32)> = dist_map
        .iter()
        .filter(|(_, &d)| d == halo)
        .map(|(&c, _)| c)
        .collect();
    halo_cells.sort();

    let legal_count = legal.len();
    let stone_count = stones_sorted.len();
    let halo_count = halo_cells.len();
    let mut coords = legal;
    coords.extend(stones_sorted);
    coords.extend(halo_cells);
    let dist: Vec<i32> = coords.iter().map(|c| dist_map[c]).collect();
    let index = build_index(&coords);
    Support {
        coords,
        legal_count,
        stone_count,
        halo_count,
        dist,
        index,
    }
}

#[inline]
fn build_index(coords: &[(i32, i32)]) -> HashMap<(i32, i32), usize> {
    let mut index = HashMap::with_capacity(coords.len());
    for (i, &c) in coords.iter().enumerate() {
        index.insert(c, i);
    }
    index
}

/// Mirrors `support._neighbor_table`: (N,6) row-local neighbour index per
/// DIRECTIONS, -1 when absent. Returned node-major flat (row*6 + k).
fn neighbor_table(coords: &[(i32, i32)], index: &HashMap<(i32, i32), usize>) -> Vec<i32> {
    let n = coords.len();
    let mut nbr = vec![-1i32; n * 6];
    for (row, &(q, r)) in coords.iter().enumerate() {
        for (k, &(dq, dr)) in DIRECTIONS.iter().enumerate() {
            if let Some(&j) = index.get(&(q + dq as i32, r + dr as i32)) {
                nbr[row * 6 + k] = j as i32;
            }
        }
    }
    nbr
}

// =============================================================================
// Phase / player ordinal derivation (mirrors features.py::record_phase/
// record_player).
// =============================================================================

const REC_PHASE_OPENING: u8 = 0;
const REC_PHASE_FIRST: u8 = 1;
const REC_PHASE_SECOND: u8 = 2;

#[inline]
fn record_phase(ordinal: usize) -> u8 {
    if ordinal == 0 {
        return REC_PHASE_OPENING;
    }
    if (ordinal - 1) % 2 == 0 {
        REC_PHASE_FIRST
    } else {
        REC_PHASE_SECOND
    }
}

#[inline]
fn record_player(ordinal: usize) -> i32 {
    if ordinal == 0 {
        return 0;
    }
    if ((ordinal - 1) / 2) % 2 == 0 {
        1
    } else {
        0
    }
}

/// Mirrors `features._opp_last_turn_cells`: reversed-history scan over the
/// records (records carry the D6-transformed coords already).
fn opp_last_turn_cells(records: &[(i32, i32, u8, u32)], current_player: i32) -> Vec<(i32, i32)> {
    let opponent = 1 - current_player;
    let n = records.len();
    for ordinal in (0..n).rev() {
        if record_player(ordinal) != opponent {
            continue;
        }
        let phase = record_phase(ordinal);
        let (q, r, _o, _i) = records[ordinal];
        if phase == REC_PHASE_SECOND {
            // ordinal-1 is the opponent's first-stone companion (ordinal >= 1
            // here since phase is SecondStone).
            let (fq, fr, _o2, _i2) = records[ordinal - 1];
            return vec![(fq, fr), (q, r)];
        }
        if phase == REC_PHASE_OPENING {
            return vec![(q, r)];
        }
        // FirstStone: skip (mid-turn).
    }
    Vec::new()
}

// =============================================================================
// Feature build (mirrors features.py::build_features) from D6-transformed facts.
// =============================================================================

/// Build the (N*NUM_FEATURES) node-major feature matrix. `records` carries the
/// D6-transformed coords; `first_stone`/hot/win cells are likewise transformed.
///
/// A cell absent from the support is surfaced as `ExpandErr::Hard` (a clean
/// error return) rather than a panic crossing the rayon/FFI boundary.
fn build_features(
    sup: &Support,
    records: &[(i32, i32, u8, u32)],
    current_player: i32,
    phase: u8,
    first_stone: Option<(i32, i32)>,
    own_hot: &[(i32, i32)],
    opp_hot: &[(i32, i32)],
    own_win: &[(i32, i32)],
    opp_win: &[(i32, i32)],
) -> Result<Vec<f32>, ExpandErr> {
    let n = sup.num_nodes();
    let mut feats = vec![0f32; n * NUM_FEATURES];
    let placements_made = records.len() as i64;

    let lookup = |sup: &Support, cell: (i32, i32), what: &str| -> Result<usize, ExpandErr> {
        sup.index
            .get(&cell)
            .copied()
            .ok_or_else(|| ExpandErr::Hard(format!("{what} cell {cell:?} missing from support")))
    };

    // Stones + recency. age = placements_made - placement_index;
    // weight = 1/(1+age); max-accumulate.
    for &(q, r, owner, placement_index) in records {
        let row = lookup(sup, (q, r), "stone")?;
        let recency_plane = if owner as i32 == current_player {
            feats[row * NUM_FEATURES + F_OWN_STONE] = 1.0;
            F_OWN_RECENCY
        } else {
            feats[row * NUM_FEATURES + F_OPP_STONE] = 1.0;
            F_OPP_RECENCY
        };
        let age = placements_made - placement_index as i64;
        // Computed in f64 then cast to f32 (matches `1.0 / (1.0 + float(age))`
        // in Python). f64-then-cast avoids the last-ULP difference an f32-direct
        // computation would have for non-dyadic ratios.
        let weight = (1.0f64 / (1.0 + age as f64)) as f32;
        let off = row * NUM_FEATURES + recency_plane;
        if weight > feats[off] {
            feats[off] = weight;
        }
    }

    // EMPTY = 1 - own - opp; LEGAL on the legal prefix.
    for row in 0..n {
        let own = feats[row * NUM_FEATURES + F_OWN_STONE];
        let opp = feats[row * NUM_FEATURES + F_OPP_STONE];
        feats[row * NUM_FEATURES + F_EMPTY] = 1.0 - own - opp;
    }
    for row in 0..sup.legal_count {
        feats[row * NUM_FEATURES + F_LEGAL] = 1.0;
    }

    // Phase-second + first-stone.
    if phase == PHASE_SECOND_STONE {
        for row in 0..n {
            feats[row * NUM_FEATURES + F_PHASE_SECOND] = 1.0;
        }
        if let Some(fs) = first_stone {
            let row = lookup(sup, fs, "first_stone")?;
            feats[row * NUM_FEATURES + F_FIRST_STONE] = 1.0;
        }
    }

    // Player colour.
    if current_player == 0 {
        for row in 0..n {
            feats[row * NUM_FEATURES + F_PLAYER_COLOUR] = 1.0;
        }
    }

    // Hot / standing-win cells — stored cells, transformed.
    for &cell in opp_hot {
        let row = lookup(sup, cell, "opp_hot")?;
        feats[row * NUM_FEATURES + F_OPP_HOT] = 1.0;
    }
    for &cell in own_hot {
        let row = lookup(sup, cell, "own_hot")?;
        feats[row * NUM_FEATURES + F_OWN_HOT] = 1.0;
    }
    for &cell in opp_win {
        let row = lookup(sup, cell, "opp_win")?;
        feats[row * NUM_FEATURES + F_OPP_WIN_NOW] = 1.0;
    }
    for &cell in own_win {
        let row = lookup(sup, cell, "own_win")?;
        feats[row * NUM_FEATURES + F_OWN_WIN_NOW] = 1.0;
    }

    // dist_to_stone: dist / DIST_SCALE.
    for row in 0..n {
        feats[row * NUM_FEATURES + F_DIST_TO_STONE] = sup.dist[row] as f32 / DIST_SCALE;
    }

    // Opponent last full turn.
    for cell in opp_last_turn_cells(records, current_player) {
        let row = lookup(sup, cell, "opp_last_turn")?;
        feats[row * NUM_FEATURES + F_OPP_LAST_TURN] = 1.0;
    }

    Ok(feats)
}

// =============================================================================
// Policy projection (mirrors samples.py::_legal_slot + expand_sample).
// =============================================================================

/// Mirrors `samples._legal_slot`: unpack the action id, apply D6, look up the
/// support slot; None when off-support or not in the legal prefix.
#[inline]
fn legal_slot(sup: &Support, sym: i32, action_id: u32) -> Option<usize> {
    let (q, r) = unpack_action_id(action_id);
    let (tq, tr) = apply_d6(sym, q, r);
    match sup.index.get(&(tq, tr)) {
        Some(&slot) if slot < sup.legal_count => Some(slot),
        _ => None,
    }
}

/// Expand one row under symmetry `sym` (mirrors samples.py::expand_sample).
/// Returns `Err(ExpandErr::OffLegal)` for a tolerated off-legal SELF policy
/// target (flag invalid), `Err(ExpandErr::Hard)` for a hard error to raise.
/// Numeric errors (non-finite / negative / zero-mass policy) always hard-error.
fn expand_one(
    facts: &RowFacts,
    sym: i32,
    radius: i32,
    halo: i32,
    horizons_len: usize,
    tolerate_off_legal: bool,
) -> Result<RowOut, ExpandErr> {
    // (1) Transform every stored coordinate fact (transform_facts).
    let records: Vec<(i32, i32, u8, u32)> = facts
        .records
        .iter()
        .map(|&(q, r, o, p)| {
            let (tq, tr) = apply_d6(sym, q, r);
            (tq, tr, o, p)
        })
        .collect();
    let first_stone = facts.first_stone.map(|(q, r)| apply_d6(sym, q, r));
    let own_hot = transform_cells(&facts.own_hot, sym);
    let opp_hot = transform_cells(&facts.opp_hot, sym);
    let own_win = transform_cells(&facts.own_win, sym);
    let opp_win = transform_cells(&facts.opp_win, sym);

    // (2) Support from transformed stones.
    let stones: Vec<(i32, i32)> = records.iter().map(|&(q, r, _, _)| (q, r)).collect();
    let sup = build_support(&stones, radius, halo);
    let legal_count = sup.legal_count;

    // (3) Features.
    let feats = build_features(
        &sup,
        &records,
        facts.current_player as i32,
        facts.phase,
        first_stone,
        &own_hot,
        &opp_hot,
        &own_win,
        &opp_win,
    )?;

    // (4) Self policy projection. Off-legal is a hard error unless
    // `tolerate_off_legal`, in which case it flags the row invalid.
    let mut policy = vec![0f32; legal_count];
    let mut total = 0.0f32;
    for &(action_id, w) in &facts.policy {
        if !w.is_finite() || w < 0.0 {
            return Err(ExpandErr::Hard(
                "policy weights must be finite and nonnegative".to_string(),
            ));
        }
        match legal_slot(&sup, sym, action_id) {
            Some(slot) => {
                policy[slot] += w;
                total += w;
            }
            None => {
                if tolerate_off_legal {
                    return Err(ExpandErr::OffLegal);
                }
                return Err(ExpandErr::Hard(format!(
                    "policy target action {action_id} is off the legal set (hard error)"
                )));
            }
        }
    }
    if facts.policy_valid != 0 && total <= 0.0 {
        return Err(ExpandErr::Hard(
            "policy target must carry positive mass".to_string(),
        ));
    }

    // (5) Opp policy projection: drop off-legal, track coverage. Off-legal does
    // not raise here.
    //
    // `opp[slot] += w` accumulates in f32 (matching numpy's in-place float32
    // add). The coverage scalars `opp_total`/`opp_kept` accumulate in f64
    // (matching Python floats, where `w = float(weight)` promotes the f32 weight
    // to f64), so `opp_kept / opp_total` is computed in f64.
    let mut opp = vec![0f32; legal_count];
    let mut opp_total = 0.0f64;
    let mut opp_kept = 0.0f64;
    for &(action_id, w) in &facts.opp_policy {
        if !w.is_finite() || w < 0.0 {
            return Err(ExpandErr::Hard(
                "opp policy weights must be finite and nonnegative".to_string(),
            ));
        }
        opp_total += w as f64;
        if let Some(slot) = legal_slot(&sup, sym, action_id) {
            opp[slot] += w;
            opp_kept += w as f64;
        }
    }
    let opp_coverage: f64 = if opp_total > 0.0 { opp_kept / opp_total } else { 1.0 };

    // (5b) Per-cell Q projection: scalar assign + presence mask. Off-legal is
    // dropped (does not raise); q must be finite and in [-1, 1].
    let mut cell_q = vec![0f32; legal_count];
    let mut cell_q_mask = vec![0f32; legal_count];
    for &(action_id, q) in &facts.q_policy {
        if !q.is_finite() || q < -1.0 || q > 1.0 {
            return Err(ExpandErr::Hard(
                "cell_q targets must be finite and in [-1, 1]".to_string(),
            ));
        }
        if let Some(slot) = legal_slot(&sup, sym, action_id) {
            cell_q[slot] = q;        // scalar assign (one action -> one distinct cell)
            cell_q_mask[slot] = 1.0;
        }
    }

    // (6) STV + moves_left — D6-invariant. Unmasked stvalue columns
    // (mask <= 0.0) are re-zeroed; masked columns are kept as stored.
    let mut stvalue = facts.stvalue[..horizons_len].to_vec();
    let mut stvalue_mask = facts.stvalue_mask[..horizons_len].to_vec();
    for c in 0..horizons_len {
        if !(stvalue_mask[c] > 0.0) {
            stvalue[c] = 0.0;
        }
    }
    // Computed in f64 then cast to f32 (matches `2.0 * min(1.0,
    // float(moves_left)/MOVES_LEFT_CAP) - 1.0` in Python), avoiding the
    // double-rounding an f32-direct computation would introduce.
    let (moves_left, moves_left_mask) = if facts.moves_left >= 0.0 {
        let m = 2.0f64 * (facts.moves_left as f64 / MOVES_LEFT_CAP as f64).min(1.0) - 1.0;
        (m as f32, 1.0f32)
    } else {
        (0.0f32, 0.0f32)
    };

    // (7) Truncated-game outcome masking. A truncated row (outcome_valid==0) has
    // no grounded terminal outcome, so value_mask, stvalue_mask, and cell_q_mask
    // are zeroed (gating the value/stvalue/cell_q heads to zero loss). The
    // policy/opp_policy heads and moves_left (masked via its -1 sentinel above)
    // are unaffected. The stvalue/cell_q target arrays are left as built; only
    // the masks are zeroed. Completed rows (outcome_valid==1) keep
    // value_mask=1.0 and the presence masks as built.
    let value_mask = if facts.outcome_valid == 0 {
        for c in 0..horizons_len {
            stvalue_mask[c] = 0.0;
        }
        for m in cell_q_mask.iter_mut() {
            *m = 0.0;
        }
        0.0f32
    } else {
        1.0f32
    };
    // Fast (value-only) rows: mask cell_q (search-distribution head). policy/
    // opp_policy/soft_policy are gated by policy_valid at the loss; cell_q is
    // gated by its presence mask, so zeroing it here is the operative gate.
    let policy_valid = if facts.policy_valid == 0 {
        for m in cell_q_mask.iter_mut() {
            *m = 0.0;
        }
        0.0f32
    } else {
        1.0f32
    };

    let nbr = neighbor_table(&sup.coords, &sup.index);
    let mut coords_flat = Vec::with_capacity(sup.num_nodes() * 2);
    for &(q, r) in &sup.coords {
        coords_flat.push(q);
        coords_flat.push(r);
    }

    Ok(RowOut {
        valid: true,
        legal_count: legal_count as i32,
        stone_count: sup.stone_count as i32,
        halo_count: sup.halo_count as i32,
        coords: coords_flat,
        dist: sup.dist,
        nbr,
        feats,
        policy,
        opp_policy: opp,
        cell_q,
        cell_q_mask,
        policy_surprise: facts.policy_surprise,
        opp_coverage,
        value: facts.value,
        value_mask,
        policy_valid,
        stvalue,
        stvalue_mask,
        moves_left,
        moves_left_mask,
    })
}

enum ExpandErr {
    /// Off-legal SELF policy target under tolerate_off_legal: flag row invalid.
    OffLegal,
    /// A hard error to surface to Python.
    Hard(String),
}

#[inline]
fn transform_cells(cells: &[(i32, i32)], sym: i32) -> Vec<(i32, i32)> {
    // The transformed cells are not sorted: each cell sets feats[row]=1
    // independently, so ordering does not affect the produced feature values.
    cells.iter().map(|&(q, r)| apply_d6(sym, q, r)).collect()
}

// =============================================================================
// Zero-copy output buffers (PlaneBuffer ABI; see serve_pack.rs).
// =============================================================================

macro_rules! out_buffer {
    ($name:ident, $ty:ty) => {
        #[pyclass]
        pub struct $name {
            data: Vec<$ty>,
        }
        #[pymethods]
        impl $name {
            fn __len__(&self) -> usize {
                self.data.len() * std::mem::size_of::<$ty>()
            }
            /// SAFETY: read-only 1-D byte view over `data`, keeping `slf` alive.
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
                    return Err(PyBufferError::new_err("buffer is read-only"));
                }
                let guard = slf.borrow();
                let data = &guard.data;
                (*view).buf = data.as_ptr() as *mut c_void;
                (*view).len = (data.len() * std::mem::size_of::<$ty>()) as ffi::Py_ssize_t;
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
                (*view).obj = slf.clone().into_any().into_ptr();
                Ok(())
            }
            unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
        }
    };
}

out_buffer!(RxF32Buf, f32);
out_buffer!(RxF64Buf, f64);
out_buffer!(RxI32Buf, i32);
out_buffer!(RxI64Buf, i64);
out_buffer!(RxU8Buf, u8);

// =============================================================================
// Column extraction (reinterpret the PackedWindow byte buffers + CSR offsets).
// =============================================================================

/// Reinterpret a `&[u8]` as a typed slice (native endianness). Length-checked
/// against `count`.
fn as_typed<'a, T: Copy>(bytes: &'a [u8], count: usize, name: &str) -> PyResult<&'a [T]> {
    let want = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| PyValueError::new_err(format!("{name}: length overflow")))?;
    if bytes.len() != want {
        return Err(PyValueError::new_err(format!(
            "{name}: {} bytes, expected {} ({} items)",
            bytes.len(),
            want,
            count
        )));
    }
    // SAFETY: length checked; T is POD with no invalid bit patterns (i16/u8/u16/
    // u32/f32/i64); PyBytes is malloc-aligned and the arrays are contiguous; the
    // source byte buffers are alive for the call.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) })
}

/// Pull a column's bytes from the dict and reinterpret to a typed slice.
fn col_typed<'a, T: Copy>(
    columns: &'a Bound<'_, PyDict>,
    key: &str,
    count: usize,
) -> PyResult<&'a [T]> {
    let item = columns
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("columns missing '{key}'")))?;
    let bytes = item.downcast::<PyBytes>()?.as_bytes();
    // Reinterpret over the borrowed bytes. The Bound keeps the PyBytes alive for
    // the duration of `columns` (the caller holds it), so the slice is valid.
    let typed = as_typed::<T>(bytes, count, key)?;
    // Transmute the lifetime to 'a (tied to `columns`): the PyBytes objects live
    // in the dict for the whole call, so this is sound.
    Ok(unsafe { std::mem::transmute::<&[T], &'a [T]>(typed) })
}

// =============================================================================
// Entry point — expand a window's rows under their pre-drawn D6.
// =============================================================================

/// Expand the rows named by `row_index` (into the packed window columns) under
/// their pre-drawn `d6` symmetries, in parallel and GIL-free.
///
/// `columns` is a dict of the `hexfield_compact_v1` window columns, each value a
/// `bytes` object (native endianness). Required keys (all `bytes` unless noted):
///   scalars[n]:  current_player(u8), phase(u8), value(f32), moves_left(f32),
///                policy_surprise(f32), outcome_valid(u8),
///                first_q(i16), first_r(i16), first_present(u8)
///   blocks[n*H]: stvalue(f32), stvalue_mask(f32)
///   hist CSR:    hist_qr(i16, 2*L), hist_owner(u8, L), hist_pidx(u16, L),
///                hist_off(i64, n+1)
///   pol/opp CSR: pol_act(u32), pol_w(f32), pol_off(i64, n+1);
///                opp_act(u32), opp_w(f32), opp_off(i64, n+1)
///   cell CSR:    {own,opp}_{hot,win}_qr(i16, 2*L) + matching _off(i64, n+1)
/// `n` is the window row count (the column length); `row_index: i64[r]` selects
/// the subset to expand (aligned 1:1 with `d6: i32[r]`). `horizons_len` is H,
/// `support_radius` the model radius (== `HEXFIELD_SUPPORT_RADIUS`).
///
/// Returns a dict of zero-copy buffers + per-(expanded-row) CSR offsets:
///   valid(RxU8Buf[r]), legal_count/stone_count/halo_count(RxI32Buf[r]),
///   node_off(i64[r+1]), pol_off_out(i64[r+1]),
///   coords(RxI32Buf, 2*ΣN), dist(RxI32Buf, ΣN), nbr(RxI32Buf, 6*ΣN),
///   feats(RxF32Buf, NUM_FEATURES*ΣN),
///   policy(RxF32Buf, ΣL), opp_policy(RxF32Buf, ΣL),
///   opp_coverage/value/value_mask/moves_left/moves_left_mask(RxF32Buf[r]),
///   stvalue(RxF32Buf, r*H), stvalue_mask(RxF32Buf, r*H).
#[pyfunction]
#[pyo3(signature = (columns, n, row_index, d6, horizons_len, support_radius, tolerate_off_legal))]
pub fn expand_shard_train<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    n: usize,
    row_index: Vec<i64>,
    d6: Vec<i32>,
    horizons_len: usize,
    support_radius: i32,
    tolerate_off_legal: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let r = row_index.len();
    if d6.len() != r {
        return Err(PyValueError::new_err(format!(
            "d6 length {} != row_index length {r}",
            d6.len()
        )));
    }
    let halo = support_radius + 1;

    // --- reinterpret the scalar + block columns (length n) -------------------
    let current_player = col_typed::<u8>(columns, "current_player", n)?;
    let phase = col_typed::<u8>(columns, "phase", n)?;
    let value = col_typed::<f32>(columns, "value", n)?;
    let policy_surprise = col_typed::<f32>(columns, "policy_surprise", n)?;
    let moves_left = col_typed::<f32>(columns, "moves_left", n)?;
    let first_q = col_typed::<i16>(columns, "first_q", n)?;
    let first_r = col_typed::<i16>(columns, "first_r", n)?;
    let first_present = col_typed::<u8>(columns, "first_present", n)?;
    // outcome_valid[i] (u8): 1 completed / 0 truncated. Gates the value/stvalue/
    // cell_q heads to zero loss for truncated rows.
    let outcome_valid = col_typed::<u8>(columns, "outcome_valid", n)?;
    // policy_valid[i] (u8): 1 full / 0 fast. Gates policy/opp/soft/cell_q heads.
    let policy_valid = col_typed::<u8>(columns, "policy_valid", n)?;
    let stvalue = col_typed::<f32>(columns, "stvalue", n * horizons_len)?;
    let stvalue_mask = col_typed::<f32>(columns, "stvalue_mask", n * horizons_len)?;

    // --- CSR offset arrays (length n+1) --------------------------------------
    let hist_off = col_typed::<i64>(columns, "hist_off", n + 1)?;
    let pol_off = col_typed::<i64>(columns, "pol_off", n + 1)?;
    let opp_off = col_typed::<i64>(columns, "opp_off", n + 1)?;
    let own_hot_off = col_typed::<i64>(columns, "own_hot_off", n + 1)?;
    let opp_hot_off = col_typed::<i64>(columns, "opp_hot_off", n + 1)?;
    let own_win_off = col_typed::<i64>(columns, "own_win_off", n + 1)?;
    let opp_win_off = col_typed::<i64>(columns, "opp_win_off", n + 1)?;

    // --- CSR data arrays (length from the offset tails) ----------------------
    let hist_total = *hist_off.last().unwrap() as usize;
    let pol_total = *pol_off.last().unwrap() as usize;
    let opp_total = *opp_off.last().unwrap() as usize;
    let hist_qr = col_typed::<i16>(columns, "hist_qr", 2 * hist_total)?;
    let hist_owner = col_typed::<u8>(columns, "hist_owner", hist_total)?;
    let hist_pidx = col_typed::<u16>(columns, "hist_pidx", hist_total)?;
    let pol_act = col_typed::<u32>(columns, "pol_act", pol_total)?;
    let pol_w = col_typed::<f32>(columns, "pol_w", pol_total)?;
    let q_pol_q = col_typed::<f32>(columns, "q_pol_q", pol_total)?;
    let opp_act = col_typed::<u32>(columns, "opp_act", opp_total)?;
    let opp_w = col_typed::<f32>(columns, "opp_w", opp_total)?;
    let own_hot_qr = col_typed::<i16>(columns, "own_hot_qr", 2 * *own_hot_off.last().unwrap() as usize)?;
    let opp_hot_qr = col_typed::<i16>(columns, "opp_hot_qr", 2 * *opp_hot_off.last().unwrap() as usize)?;
    let own_win_qr = col_typed::<i16>(columns, "own_win_qr", 2 * *own_win_off.last().unwrap() as usize)?;
    let opp_win_qr = col_typed::<i16>(columns, "opp_win_qr", 2 * *opp_win_off.last().unwrap() as usize)?;

    // --- materialize per-row facts on the main thread so workers own their
    // data (no borrow of the PyBytes inside par_iter) -------------------------
    let qr_pairs = |flat: &[i16], a: usize, b: usize| -> Vec<(i32, i32)> {
        (a..b)
            .map(|k| (flat[2 * k] as i32, flat[2 * k + 1] as i32))
            .collect()
    };

    let mut facts: Vec<RowFacts> = Vec::with_capacity(r);
    for &ri64 in &row_index {
        let i = ri64 as usize;
        if i >= n {
            return Err(PyValueError::new_err(format!(
                "row_index entry {i} out of range for n={n}"
            )));
        }
        let h0 = hist_off[i] as usize;
        let h1 = hist_off[i + 1] as usize;
        let records: Vec<(i32, i32, u8, u32)> = (h0..h1)
            .map(|k| {
                (
                    hist_qr[2 * k] as i32,
                    hist_qr[2 * k + 1] as i32,
                    hist_owner[k],
                    hist_pidx[k] as u32,
                )
            })
            .collect();
        let p0 = pol_off[i] as usize;
        let p1 = pol_off[i + 1] as usize;
        let policy: Vec<(u32, f32)> = (p0..p1).map(|k| (pol_act[k], pol_w[k])).collect();
        let q_policy: Vec<(u32, f32)> = (p0..p1).map(|k| (pol_act[k], q_pol_q[k])).collect();
        let o0 = opp_off[i] as usize;
        let o1 = opp_off[i + 1] as usize;
        let opp_policy: Vec<(u32, f32)> = (o0..o1).map(|k| (opp_act[k], opp_w[k])).collect();
        let first_stone = if first_present[i] == 1 {
            Some((first_q[i] as i32, first_r[i] as i32))
        } else {
            None
        };
        let stv = stvalue[i * horizons_len..(i + 1) * horizons_len].to_vec();
        let stv_mask = stvalue_mask[i * horizons_len..(i + 1) * horizons_len].to_vec();
        facts.push(RowFacts {
            records,
            current_player: current_player[i],
            phase: phase[i],
            first_stone,
            own_hot: qr_pairs(own_hot_qr, own_hot_off[i] as usize, own_hot_off[i + 1] as usize),
            opp_hot: qr_pairs(opp_hot_qr, opp_hot_off[i] as usize, opp_hot_off[i + 1] as usize),
            own_win: qr_pairs(own_win_qr, own_win_off[i] as usize, own_win_off[i + 1] as usize),
            opp_win: qr_pairs(opp_win_qr, opp_win_off[i] as usize, opp_win_off[i + 1] as usize),
            policy,
            q_policy,
            opp_policy,
            policy_surprise: policy_surprise[i],
            value: value[i],
            stvalue: stv,
            stvalue_mask: stv_mask,
            moves_left: moves_left[i],
            outcome_valid: outcome_valid[i],
            policy_valid: policy_valid[i],
        });
    }

    // --- expand in parallel under py.detach (GIL released) -------------------
    // par_iter().collect() preserves input order, so the output does not depend
    // on worker count. A Hard error in any row aborts the whole call (surfaced
    // by first index in the loop below).
    let results: Vec<Result<RowOut, ExpandErr>> = py.detach(|| {
        facts
            .par_iter()
            .zip(d6.par_iter())
            .map(|(f, &sym)| expand_one(f, sym, support_radius, halo, horizons_len, tolerate_off_legal))
            .collect()
    });

    // Surface the first hard error in row order.
    let mut rows: Vec<RowOut> = Vec::with_capacity(r);
    for res in results {
        match res {
            Ok(row) => rows.push(row),
            Err(ExpandErr::OffLegal) => rows.push(RowOut {
                valid: false,
                legal_count: 0,
                stone_count: 0,
                halo_count: 0,
                coords: Vec::new(),
                dist: Vec::new(),
                nbr: Vec::new(),
                feats: Vec::new(),
                policy: Vec::new(),
                opp_policy: Vec::new(),
                cell_q: Vec::new(),
                cell_q_mask: Vec::new(),
                policy_surprise: 0.0,
                opp_coverage: 1.0,
                value: 0.0,
                value_mask: 0.0,
                policy_valid: 0.0,
                stvalue: vec![0.0; horizons_len],
                stvalue_mask: vec![0.0; horizons_len],
                moves_left: 0.0,
                moves_left_mask: 0.0,
            }),
            Err(ExpandErr::Hard(msg)) => return Err(PyValueError::new_err(msg)),
        }
    }

    // --- serial order-preserving concat into the flat output buffers ---------
    let total_nodes: usize = rows.iter().map(|x| x.coords.len() / 2).sum();
    let total_legal: usize = rows.iter().map(|x| x.policy.len()).sum();

    let mut valid = Vec::with_capacity(r);
    let mut legal_count = Vec::with_capacity(r);
    let mut stone_count = Vec::with_capacity(r);
    let mut halo_count = Vec::with_capacity(r);
    let mut node_off = Vec::with_capacity(r + 1);
    let mut pol_off_out = Vec::with_capacity(r + 1);
    let mut coords = Vec::with_capacity(total_nodes * 2);
    let mut dist = Vec::with_capacity(total_nodes);
    let mut nbr = Vec::with_capacity(total_nodes * 6);
    let mut feats = Vec::with_capacity(total_nodes * NUM_FEATURES);
    let mut policy = Vec::with_capacity(total_legal);
    let mut opp_policy = Vec::with_capacity(total_legal);
    let mut cell_q = Vec::with_capacity(total_legal);
    let mut cell_q_mask = Vec::with_capacity(total_legal);
    let mut policy_surprise_out = Vec::with_capacity(r);
    let mut opp_coverage: Vec<f64> = Vec::with_capacity(r);
    let mut value_out = Vec::with_capacity(r);
    let mut value_mask_out = Vec::with_capacity(r);
    let mut policy_valid_out = Vec::with_capacity(r);
    let mut moves_left_out = Vec::with_capacity(r);
    let mut moves_left_mask = Vec::with_capacity(r);
    let mut stvalue_out = Vec::with_capacity(r * horizons_len);
    let mut stvalue_mask_out = Vec::with_capacity(r * horizons_len);

    node_off.push(0i64);
    pol_off_out.push(0i64);
    for row in &rows {
        valid.push(if row.valid { 1u8 } else { 0u8 });
        legal_count.push(row.legal_count);
        stone_count.push(row.stone_count);
        halo_count.push(row.halo_count);
        coords.extend_from_slice(&row.coords);
        dist.extend_from_slice(&row.dist);
        nbr.extend_from_slice(&row.nbr);
        feats.extend_from_slice(&row.feats);
        policy.extend_from_slice(&row.policy);
        opp_policy.extend_from_slice(&row.opp_policy);
        cell_q.extend_from_slice(&row.cell_q);
        cell_q_mask.extend_from_slice(&row.cell_q_mask);
        policy_surprise_out.push(row.policy_surprise);
        opp_coverage.push(row.opp_coverage);
        value_out.push(row.value);
        value_mask_out.push(row.value_mask);
        policy_valid_out.push(row.policy_valid);
        moves_left_out.push(row.moves_left);
        moves_left_mask.push(row.moves_left_mask);
        stvalue_out.extend_from_slice(&row.stvalue);
        stvalue_mask_out.extend_from_slice(&row.stvalue_mask);
        node_off.push(node_off.last().unwrap() + (row.coords.len() / 2) as i64);
        pol_off_out.push(pol_off_out.last().unwrap() + row.policy.len() as i64);
    }

    let out = PyDict::new(py);
    out.set_item("valid", Py::new(py, RxU8Buf { data: valid })?)?;
    out.set_item("legal_count", Py::new(py, RxI32Buf { data: legal_count })?)?;
    out.set_item("stone_count", Py::new(py, RxI32Buf { data: stone_count })?)?;
    out.set_item("halo_count", Py::new(py, RxI32Buf { data: halo_count })?)?;
    out.set_item("node_off", Py::new(py, RxI64Buf { data: node_off })?)?;
    out.set_item("pol_off", Py::new(py, RxI64Buf { data: pol_off_out })?)?;
    out.set_item("coords", Py::new(py, RxI32Buf { data: coords })?)?;
    out.set_item("dist", Py::new(py, RxI32Buf { data: dist })?)?;
    out.set_item("nbr", Py::new(py, RxI32Buf { data: nbr })?)?;
    out.set_item("feats", Py::new(py, RxF32Buf { data: feats })?)?;
    out.set_item("policy", Py::new(py, RxF32Buf { data: policy })?)?;
    out.set_item("opp_policy", Py::new(py, RxF32Buf { data: opp_policy })?)?;
    out.set_item("opp_coverage", Py::new(py, RxF64Buf { data: opp_coverage })?)?;
    out.set_item("value", Py::new(py, RxF32Buf { data: value_out })?)?;
    out.set_item("value_mask", Py::new(py, RxF32Buf { data: value_mask_out })?)?;
    out.set_item("policy_valid", Py::new(py, RxF32Buf { data: policy_valid_out })?)?;
    out.set_item("moves_left", Py::new(py, RxF32Buf { data: moves_left_out })?)?;
    out.set_item("moves_left_mask", Py::new(py, RxF32Buf { data: moves_left_mask })?)?;
    out.set_item("stvalue", Py::new(py, RxF32Buf { data: stvalue_out })?)?;
    out.set_item("stvalue_mask", Py::new(py, RxF32Buf { data: stvalue_mask_out })?)?;
    out.set_item("cell_q", Py::new(py, RxF32Buf { data: cell_q })?)?;
    out.set_item("cell_q_mask", Py::new(py, RxF32Buf { data: cell_q_mask })?)?;
    out.set_item("policy_surprise", Py::new(py, RxF32Buf { data: policy_surprise_out })?)?;
    out.set_item("num_rows", r)?;
    out.set_item("num_features", NUM_FEATURES)?;
    Ok(out)
}
