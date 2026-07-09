//! Node features (F = NUM_FEATURES = 25) computed from the engine state.
//! Planes 0-10 are the 11 kept scalars, 11-22 the 12 graded per-axis window
//! planes, 23-24 the 2 fork scalars. A matching implementation lives in
//! python/hexfield_eq/features.py.

use hexo_engine::{Axis, HexCoord, HexoState as RustHexoState, Player, TurnPhase, WindowKey, WindowStore};

use crate::constants::*;
use crate::support::Support;

pub fn axis_delta(axis: Axis) -> (i16, i16) {
    match axis {
        Axis::Q => (1, 0),
        Axis::R => (0, 1),
        Axis::QR => (1, -1),
    }
}

/// The 14 graded window-feature values for one cell `x`, in plane order:
/// `[own_line Q,R,QR, opp_line Q,R,QR, own_live Q,R,QR, opp_live Q,R,QR,
/// own_fork, opp_fork]`.
///
/// For each axis and each of the 6 length-`WINDOW_LEN` windows through `x`, read
/// `own = count(me)`, `opp = count(other)` from the incremental store (a window
/// absent from the store has no stones ⇒ counts 0). A window is *clean for me*
/// when `opp == 0` and *clean for opp* when `own == 0`; `line` is the max clean
/// count (`/5`), `live` the clean-window count (`/6`). The `empty-at-x` gate
/// (require `x` empty in the window) is vacuous — always true for an empty `x`,
/// dropped for a stone `x` — and is applied faithfully so the Rust and Python
/// paths are literal transcriptions of each other. `fork` is `|{axis : raw line
/// >= FORK_LINE_THRESHOLD}| / 3`.
///
/// Shared by the serve featurizer (below) and the train-time expand kernel
/// (`replay_expand.rs`), which supplies a store built from raw placement facts
/// via `WindowStore::from_placements`.
pub(crate) fn window_feature_row(
    windows: &WindowStore,
    x: HexCoord,
    is_empty: bool,
    me: Player,
    other: Player,
) -> [f32; 14] {
    let mut out = [0f32; 14];
    let mut own_line_raw = [0u32; 3];
    let mut opp_line_raw = [0u32; 3];
    for (ai, axis) in Axis::ALL.iter().enumerate() {
        let vec = axis.vector();
        let mut own_max = 0u32;
        let mut opp_max = 0u32;
        let mut own_live = 0u32;
        let mut opp_live = 0u32;
        for offset in 0..WINDOW_LEN {
            let start = x - vec.scale(offset as i16);
            let (own_c, opp_c, empty_at_x) = match windows.entry(WindowKey { start, axis: *axis }) {
                Some(entry) => (
                    entry.count(me) as u32,
                    entry.count(other) as u32,
                    ((entry.empty_mask() >> offset) & 1) != 0,
                ),
                None => (0, 0, true),
            };
            // empty-at-x gate: require x empty in the window for an empty cell
            // (always satisfied); dropped for a stone cell.
            if is_empty && !empty_at_x {
                continue;
            }
            if opp_c == 0 {
                own_live += 1;
                if own_c > own_max {
                    own_max = own_c;
                }
            }
            if own_c == 0 {
                opp_live += 1;
                if opp_c > opp_max {
                    opp_max = opp_c;
                }
            }
        }
        own_line_raw[ai] = own_max;
        opp_line_raw[ai] = opp_max;
        out[ai] = own_max as f32 / LINE_NORM;
        out[3 + ai] = opp_max as f32 / LINE_NORM;
        out[6 + ai] = own_live as f32 / LIVE_NORM;
        out[9 + ai] = opp_live as f32 / LIVE_NORM;
    }
    let own_fork = own_line_raw.iter().filter(|&&c| c >= FORK_LINE_THRESHOLD).count();
    let opp_fork = opp_line_raw.iter().filter(|&&c| c >= FORK_LINE_THRESHOLD).count();
    out[12] = own_fork as f32 / FORK_NORM;
    out[13] = opp_fork as f32 / FORK_NORM;
    out
}

/// Side-relative ray lengths for one cell (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md
/// L1 walk), flat u8[RAYLEN_SLOTS] indexed `side*6 + axis*2 + dir` with side in
/// {own=0, opp=1} (own = the side to move), axis in Axis::ALL order, dir in
/// {+=0, -=1}. For each (side, axis, dir): walk `j = 1..=RAY_REACH` from `x`;
/// a cell off the support stops the walk (unattendable); an anti-side stone is
/// INCLUDED (terminal blocker) then stops it; own-side stones and empties pass
/// through. The occupancy of `x` itself is never consulted.
///
/// `on_support` is support membership and `owner` the stone owner (player index
/// 0/1) at a cell; `me` is the side to move's player index. Shared by the serve
/// featurizer (`build_ray_lengths` below) and the train-time expand kernel
/// (`replay_expand.rs`), which supplies closures over its reconstructed board.
pub(crate) fn ray_length_row<S, O>(on_support: &S, owner: &O, xq: i32, xr: i32, me: u8) -> [u8; RAYLEN_SLOTS]
where
    S: Fn(i32, i32) -> bool,
    O: Fn(i32, i32) -> Option<u8>,
{
    let mut out = [0u8; RAYLEN_SLOTS];
    for side in 0..2usize {
        let anti = if side == 0 { 1 - me } else { me };
        for (ai, axis) in Axis::ALL.iter().enumerate() {
            let (dq, dr) = axis_delta(*axis);
            let (dq, dr) = (dq as i32, dr as i32);
            for (di, sign) in [(0usize, 1i32), (1usize, -1i32)] {
                let mut len = 0u8;
                for j in 1..=(RAY_REACH as i32) {
                    let yq = xq + sign * dq * j;
                    let yr = xr + sign * dr * j;
                    if !on_support(yq, yr) {
                        break;
                    }
                    len = j as u8;
                    if owner(yq, yr) == Some(anti) {
                        break;
                    }
                }
                out[side * 6 + ai * 2 + di] = len;
            }
        }
    }
    out
}

/// (N * RAYLEN_SLOTS) node-major ray lengths in support node order, from the
/// live engine board (serve twin of the `replay_expand.rs` walk).
pub fn build_ray_lengths(state: &RustHexoState, sup: &Support) -> Vec<u8> {
    let n = sup.num_nodes();
    let mut out = vec![0u8; n * RAYLEN_SLOTS];
    let board = state.board();
    let me = if state.current_player() == Player::Player0 { 0u8 } else { 1u8 };
    let on_support = |q: i32, r: i32| sup.index.contains_key(&(q as i16, r as i16));
    let owner = |q: i32, r: i32| {
        board
            .get(HexCoord { q: q as i16, r: r as i16 })
            .map(|p| if p == Player::Player0 { 0u8 } else { 1u8 })
    };
    for row in 0..n {
        let x = sup.coords[row];
        let vals = ray_length_row(&on_support, &owner, x.q as i32, x.r as i32, me);
        out[row * RAYLEN_SLOTS..(row + 1) * RAYLEN_SLOTS].copy_from_slice(&vals);
    }
    out
}

/// (N * NUM_FEATURES) node-major feature matrix in support node order.
pub fn build_features(state: &RustHexoState, sup: &Support) -> Vec<f32> {
    let n = sup.num_nodes();
    let mut feats = vec![0f32; n * NUM_FEATURES];
    let set = |feats: &mut Vec<f32>, row: usize, plane: usize, value: f32| {
        feats[row * NUM_FEATURES + plane] = value;
    };
    let current = state.current_player();
    let placements_made = state.placements_made();

    // Stones and recency. age = placements_made - placement_index (saturating);
    // recency weight = 1/(1+age), max-accumulated per cell. Divide in f64 then
    // cast to f32 to match the train expand path (replay_expand.rs) and the Python
    // oracle (features.build_features), both of which compute in f64 and store
    // into a float32 array; a direct f32 divide can differ by a ULP for non-dyadic
    // ratios (e.g. 1/3) (BUGS_FOUND.md serve-recency-dtype item).
    for record in state.placement_history().iter() {
        let row = sup.row(record.coord).expect("stone missing from support");
        let (stone_plane, recency_plane) = if record.player == current {
            (F_OWN_STONE, F_OWN_RECENCY)
        } else {
            (F_OPP_STONE, F_OPP_RECENCY)
        };
        set(&mut feats, row, stone_plane, 1.0);
        let age = placements_made.saturating_sub(record.placement_index);
        let weight = (1.0f64 / (1.0 + age as f64)) as f32;
        let offset = row * NUM_FEATURES + recency_plane;
        feats[offset] = feats[offset].max(weight);
    }

    for row in 0..n {
        let own = feats[row * NUM_FEATURES + F_OWN_STONE];
        let opp = feats[row * NUM_FEATURES + F_OPP_STONE];
        feats[row * NUM_FEATURES + F_EMPTY] = 1.0 - own - opp;
        feats[row * NUM_FEATURES + F_DIST_TO_STONE] = sup.dist[row] as f32 / DIST_SCALE;
    }
    for row in 0..sup.legal_count {
        set(&mut feats, row, F_LEGAL, 1.0);
    }

    match state.phase() {
        TurnPhase::SecondStone { first } => {
            for row in 0..n {
                set(&mut feats, row, F_PHASE_SECOND, 1.0);
            }
            let row = sup.row(first).expect("first stone missing from support");
            set(&mut feats, row, F_FIRST_STONE, 1.0);
        }
        TurnPhase::Opening | TurnPhase::FirstStone => {}
    }

    if current == Player::Player0 {
        for row in 0..n {
            set(&mut feats, row, F_PLAYER_COLOUR, 1.0);
        }
    }

    fill_window_features(state, current, sup, &mut feats);
    fill_opponent_last_turn(state, current, sup, &mut feats);

    feats
}

/// Graded per-axis window planes (11-24), read from the engine's incremental
/// window store. Per support cell, `window_feature_row` scans the 6 windows on
/// each axis and emits own/opp line + live per axis plus own/opp fork.
fn fill_window_features(
    state: &RustHexoState,
    current: Player,
    sup: &Support,
    feats: &mut Vec<f32>,
) {
    let windows = state.board().windows();
    let me = current;
    let other = current.other();
    for row in 0..sup.num_nodes() {
        let x = sup.coords[row];
        // A support cell is empty iff no stone occupies it (legal/halo cells).
        let is_empty = state.board().get(x).is_none();
        let vals = window_feature_row(windows, x, is_empty, me, other);
        let base = row * NUM_FEATURES;
        for k in 0..12 {
            feats[base + F_OWN_LINE_Q + k] = vals[k];
        }
        feats[base + F_OWN_FORK] = vals[12];
        feats[base + F_OPP_FORK] = vals[13];
    }
}

/// Cells of the opponent's most recent full turn.
fn fill_opponent_last_turn(
    state: &RustHexoState,
    current: Player,
    sup: &Support,
    feats: &mut Vec<f32>,
) {
    let opponent = current.other();
    for record in state.placement_history().iter().rev() {
        if record.player != opponent {
            continue;
        }
        match record.phase {
            TurnPhase::SecondStone { first } => {
                for coord in [first, record.coord] {
                    let row = sup.row(coord).expect("last-turn cell missing from support");
                    feats[row * NUM_FEATURES + F_OPP_LAST_TURN] = 1.0;
                }
                return;
            }
            TurnPhase::Opening => {
                let row = sup
                    .row(record.coord)
                    .expect("last-turn cell missing from support");
                feats[row * NUM_FEATURES + F_OPP_LAST_TURN] = 1.0;
                return;
            }
            TurnPhase::FirstStone => {}
        }
    }
}
