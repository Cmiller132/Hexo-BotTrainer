//! vcf.rs — EXPLORATORY depth-bounded forcing-move (VCF/VCDT) solver prototype
//! for Phase 6 of the TSS plan. This is a BENCHMARK/EXPLORATION artifact: it is
//! NOT wired into the live MCTS (the plan defers the deep solver). It exists to
//! measure the cost and rough correctness of a multi-ply forcing search so the
//! adopt/defer decision is data-driven.
//!
//! Structure: an AND-OR forcing search restricted to threat moves (the only way
//! to bound Connect6's 2-stone branching).
//!   - OR node (attacker to move): proven win if the attacker has a win-now this
//!     turn; else try each FORCING attacker placement (one that creates/keeps a
//!     >=4 threat), win if ANY line wins.
//!   - AND node (defender to move): if the defender has its own win-now the attack
//!     is refuted on this line; if the attacker's threats are unanswerable within
//!     the defender's placements (min hitting set > B) the attacker wins; else the
//!     defender must try every neutralizing (hitting) placement — the attacker
//!     wins only if EVERY defense still loses.
//!
//! SOUNDNESS CAVEAT (the central finding): the defender move set is restricted to
//! threat-neutralizing cells, so a defender COUNTER-THREAT / quiet refutation is
//! not considered. A "win" from this prototype is therefore NOT a sound proof —
//! it can over-claim. This is the classic forcing-search incompleteness the plan
//! flags, and it is exactly why a deep solver is risky to wire into the value
//! signal. The benchmark below quantifies both the cost and (via a defensible
//! position) whether the over-claim actually materializes.

use std::time::Instant;

use hexo_engine::{
    apply_placement, HexCoord, HexoState as RustHexoState, Placement, Player,
};

use super::threats::analyze;

pub(crate) struct VcfResult {
    pub(crate) win: bool,
    pub(crate) nodes: u64,
    pub(crate) hit_limit: bool,
}

struct Budget {
    nodes: u64,
    node_cap: u64,
    deadline: Instant,
    hit_limit: bool,
}

/// Empties of `player`'s active windows with count >= `min_count`, deduplicated
/// and capped. For the attacker (`min_count = 3`) these are the forcing moves
/// (placing makes a >=4 threat); for the defender (`min_count = 4`) these are the
/// neutralizing/hitting cells against the attacker's immediate threats.
fn threat_cells(state: &RustHexoState, player: Player, min_count: u8, cap: usize) -> Vec<HexCoord> {
    let mut cells: Vec<HexCoord> = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.active_player() == Some(player) && entry.count(player) >= min_count {
            for c in entry.empty_cells() {
                if !cells.contains(&c) {
                    cells.push(c);
                    if cells.len() >= cap {
                        return cells;
                    }
                }
            }
        }
    }
    cells
}

fn prove(state: &RustHexoState, attacker: Player, ply_budget: u32, b: &mut Budget) -> bool {
    b.nodes += 1;
    if b.nodes >= b.node_cap || Instant::now() >= b.deadline {
        b.hit_limit = true;
        return false; // unknown -> not proven
    }
    let stm = state.current_player();
    let a = analyze(state);
    if stm == attacker {
        if a.own_win_now {
            return true;
        }
        if ply_budget == 0 {
            return false;
        }
        // Forcing attacker placements: cells extending an attacker window to >=4.
        for mv in threat_cells(state, attacker, 3, 24) {
            let mut child = state.clone();
            match apply_placement(&mut child, Placement { coord: mv }) {
                Ok(res) => {
                    if let Some(outcome) = res.outcome {
                        if outcome.winner == attacker {
                            return true;
                        }
                        continue; // attacker move ended the game without winning
                    }
                    // Keep only forcing lines: after the move the attacker must
                    // still have an active >=4 threat (so the defender is forced),
                    // unless control is still the attacker's (mid-turn build-up).
                    let child_stm = child.current_player();
                    let keep = child_stm == attacker
                        || child
                            .board()
                            .windows()
                            .threat_entries(attacker)
                            .next()
                            .is_some();
                    if keep && prove(&child, attacker, ply_budget - 1, b) {
                        return true;
                    }
                }
                Err(_) => continue, // illegal candidate (shouldn't happen for empties)
            }
        }
        false
    } else {
        // Defender to move.
        if a.own_win_now {
            return false; // defender escapes with its own win -> attack refuted
        }
        if a.forced_loss() {
            return true; // unanswerable conjunction -> attacker wins
        }
        if ply_budget == 0 {
            return false;
        }
        // Defender must neutralize the attacker's immediate threats; try every
        // hitting cell. The attacker wins iff every defense still loses.
        let defenses = threat_cells(state, attacker, 4, 24);
        if defenses.is_empty() {
            return false; // no immediate threat to force a response -> can't prove
        }
        for mv in defenses {
            let mut child = state.clone();
            match apply_placement(&mut child, Placement { coord: mv }) {
                Ok(res) => {
                    if let Some(outcome) = res.outcome {
                        // defender completing 6 would be a defender win (rare);
                        // treat any non-attacker outcome as a refutation.
                        if outcome.winner != attacker {
                            return false;
                        }
                    }
                    if !prove(&child, attacker, ply_budget - 1, b) {
                        return false; // this defense refutes the attack
                    }
                }
                Err(_) => continue,
            }
        }
        true
    }
}

/// Run the bounded forcing search for the side to move at `state`.
pub(crate) fn solve(
    state: &RustHexoState,
    ply_budget: u32,
    node_cap: u64,
    time_limit_ms: u64,
) -> VcfResult {
    let attacker = state.current_player();
    let mut budget = Budget {
        nodes: 0,
        node_cap,
        deadline: Instant::now() + std::time::Duration::from_millis(time_limit_ms),
        hit_limit: false,
    };
    // The placement budget B (placements remaining this turn) is phase-derived;
    // see `threats::placements_remaining`, used by the threat analysis below.
    let win = prove(state, attacker, ply_budget, &mut budget);
    VcfResult {
        win,
        nodes: budget.nodes,
        hit_limit: budget.hit_limit,
    }
}

// --- PyO3 benchmark/diagnostic hook -------------------------------------------

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Benchmark hook: run the forcing solver on a live engine state and return
/// {win, nodes, micros, hit_limit}. Exploratory only — not used by search.
#[pyfunction]
#[pyo3(signature = (state, ply_budget=8, node_cap=2_000_000, time_limit_ms=200))]
fn hexgnn_vcf_solve(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    ply_budget: u32,
    node_cap: u64,
    time_limit_ms: u64,
) -> PyResult<Py<PyAny>> {
    let s = super::state::state_from_py_state(py, state)?;
    let t0 = Instant::now();
    let res = solve(&s, ply_budget, node_cap, time_limit_ms);
    let micros = t0.elapsed().as_micros() as u64;
    let d = PyDict::new(py);
    d.set_item("win", res.win)?;
    d.set_item("nodes", res.nodes)?;
    d.set_item("micros", micros)?;
    d.set_item("hit_limit", res.hit_limit)?;
    Ok(d.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexgnn_vcf_solve, module)?)?;
    Ok(())
}
