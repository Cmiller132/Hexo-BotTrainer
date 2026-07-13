//! Test-gated CPU benchmark for sizing deterministic production caps.
//!
//! Run with:
//! `cargo test --release -p hexfield_eq tss_bench_report -- --ignored --nocapture`
//! Wall-clock measurement exists only in this ignored harness; no solve path
//! observes time.

use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::TssSolver;

#[derive(Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn random_state(seed: u64, stones: usize) -> HexoState {
    let mut rng = XorShift64(seed | 1);
    let mut state = HexoState::new();
    while state.board().len() < stones && !state.is_terminal() {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        if legal.is_empty() {
            break;
        }
        let coord = legal[(rng.next() as usize) % legal.len()];
        apply_placement(&mut state, Placement { coord }).unwrap();
    }
    state
}

fn line_biased_state(seed: u64, stones: usize) -> HexoState {
    let mut rng = XorShift64(seed | 1);
    let mut state = HexoState::new();
    while state.board().len() < stones && !state.is_terminal() {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        if legal.is_empty() {
            break;
        }
        let mut shortlist = Vec::new();
        let sample_count = legal.len().min(16);
        while shortlist.len() < sample_count {
            let coord = legal[(rng.next() as usize) % legal.len()];
            if !shortlist.contains(&coord) {
                shortlist.push(coord);
            }
        }
        shortlist.sort_by_key(|coord| (coord.q, coord.r));

        let mut best: Option<(u64, HexCoord)> = None;
        for coord in shortlist {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord }) else {
                continue;
            };
            let score = if result.outcome.is_some() {
                None // keep every timed corpus state nonterminal
            } else {
                let mut threats = 0u64;
                let mut count_three = 0u64;
                let mut max_count = 0u64;
                for entry in state.board().windows().entries() {
                    let Some(owner) = entry.active_player() else {
                        continue;
                    };
                    let count = u64::from(entry.count(owner));
                    threats += u64::from(count >= 4);
                    count_three += u64::from(count == 3);
                    max_count = max_count.max(count);
                }
                Some(64 * threats + 8 * count_three + max_count)
            };
            state.undo(delta);
            if let Some(score) = score {
                let candidate = (score, coord);
                if best.is_none_or(|(old_score, old_coord)| {
                    score > old_score
                        || (score == old_score && (coord.q, coord.r) < (old_coord.q, old_coord.r))
                }) {
                    best = Some(candidate);
                }
            }
        }
        let Some((_score, coord)) = best else {
            break;
        };
        apply_placement(&mut state, Placement { coord }).unwrap();
    }
    state
}

fn replay_prefix(history: &[(i16, i16)], stones: usize) -> Option<HexoState> {
    if stones > history.len() {
        return None;
    }
    let mut state = HexoState::new();
    for &(q, r) in &history[..stones] {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord { q, r },
            },
        )
        .ok()?;
    }
    (!state.is_terminal()).then_some(state)
}

const LINE_BIASED: &[(i16, i16)] = &[
    (0, 0),
    (-1, 0),
    (0, -1),
    (-2, -3),
    (-1, -3),
    (-2, 1),
    (-3, 1),
    (0, -3),
    (1, -3),
    (-4, 2),
    (2, -4),
    (1, 4),
    (2, 4),
    (-5, 2),
    (2, -5),
    (3, 4),
    (4, 1),
    (-6, 3),
    (3, -6),
    (4, 2),
    (4, 3),
    (-7, 3),
    (3, -7),
    (1, 7),
    (2, 6),
    (-1, 2),
    (2, -1),
    (3, 5),
];

/// Prints stable, machine-readable bucket rows.  Timing noise affects only the
/// reported throughput, never the solver results or corpus.
#[test]
#[ignore = "timing harness; run explicitly in --release mode"]
fn tss_bench_report() {
    let caps = SolveCaps {
        node_cap: 100,
        tt_bytes_cap: 64 << 10,
    };
    println!(
        "TSS_BENCH_CONFIG node_cap={} tt_bytes_cap={} uniform_positions_per_bucket=2 line_biased_positions_per_bucket=2",
        caps.node_cap, caps.tt_bytes_cap
    );

    for stones in [3usize, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28] {
        let mut corpus: Vec<HexoState> = (0..2u64)
            .map(|sample| {
                random_state(
                    0xD1B5_4A32_D192_ED03u64 ^ (stones as u64).wrapping_mul(0x9E37_79B9) ^ sample,
                    stones,
                )
            })
            .collect();
        corpus.extend((0..2u64).map(|sample| {
            line_biased_state(
                0xA076_1D64_78BD_642Fu64 ^ (stones as u64).wrapping_mul(0xE703_7ED1) ^ sample,
                stones,
            )
        }));
        if let Some(line_biased) = replay_prefix(LINE_BIASED, stones) {
            corpus.push(line_biased);
        }
        assert!(corpus
            .iter()
            .all(|state| state.board().len() == stones && !state.is_terminal()));

        // One untimed warm-up keeps code-page effects out of the first bucket.
        let _ = TssSolver::default().solve(&corpus[0], &caps);
        let start = Instant::now();
        let mut nodes = 0u64;
        let mut unknown = 0usize;
        let threatful = corpus
            .iter()
            .filter(|state| state.board().windows().has_threats())
            .count();
        for state in &corpus {
            let result = TssSolver::default().solve(state, &caps);
            nodes = nodes.saturating_add(result.stats.nodes);
            unknown += usize::from(result.status == ProofStatus::Unknown);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let nodes_per_sec = if elapsed > 0.0 {
            nodes as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "TSS_BENCH stones_on_board={} positions={} threatful={} nodes={} seconds={:.6} nodes_per_sec={:.1} unknown_rate={:.6}",
            stones,
            corpus.len(),
            threatful,
            nodes,
            elapsed,
            nodes_per_sec,
            unknown as f64 / corpus.len() as f64,
        );
    }
}
