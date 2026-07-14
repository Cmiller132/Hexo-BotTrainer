//! Test-gated CPU benchmark for sizing deterministic production caps.
//!
//! Run with:
//! `cargo test --release -p hexfield_eq tss_bench_report -- --ignored --nocapture`
//! Wall-clock measurement exists only in this ignored harness; no solve path
//! observes time.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::TssSolver;
use crate::tss_verify::d6_transform_coord;

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

// Exact histories from tests/test_hexfield_eq_tss_shadow.py.  Four D6 images
// of each give a compact but adversarial 12-position extension without
// changing the underlying game histories.
const FORCED_DEFENSE_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (0, 8),
    (2, 7),
    (1, 0),
    (2, 0),
    (4, 6),
    (6, 5),
    (3, 0),
    (4, 0),
];

const DEEP_WIN_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (0, 8),
    (2, 7),
    (1, 0),
    (2, 0),
    (4, 6),
    (6, 5),
    (0, 4),
    (1, 4),
    (8, 4),
    (10, 3),
    (2, 4),
    (16, 0),
    (12, 2),
    (14, 1),
];

const FORCED_LOSS_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (0, 8),
    (2, 7),
    (1, 0),
    (2, 0),
    (4, 6),
    (6, 5),
    (3, 0),
    (0, 4),
    (8, 4),
    (10, 3),
    (1, 4),
    (2, 4),
    (12, 2),
    (14, 1),
    (3, 4),
    (16, 0),
];

const CURATED_D6_IMAGES: [u8; 4] = [0, 1, 6, 7];
const SYNTHETIC_STONE_BUCKETS: [usize; 14] = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28];

struct BenchPosition {
    name: String,
    symmetry: i16,
    threatful: bool,
    state: HexoState,
}

struct BenchBucket {
    name: String,
    corpus: &'static str,
    family: &'static str,
    stones: usize,
    positions: Vec<BenchPosition>,
}

fn bench_position(name: String, symmetry: i16, state: HexoState) -> BenchPosition {
    // Classification is intentionally done at corpus-build time, outside
    // every timed region.
    let threatful = state.board().windows().has_threats();
    BenchPosition {
        name,
        symmetry,
        threatful,
        state,
    }
}

fn synthetic_bucket(stones: usize) -> BenchBucket {
    let mut positions = Vec::with_capacity(5);
    for sample in 0..2u64 {
        positions.push(bench_position(
            format!("synthetic_s{stones:02}_uniform_{sample}"),
            -1,
            random_state(
                0xD1B5_4A32_D192_ED03u64 ^ (stones as u64).wrapping_mul(0x9E37_79B9) ^ sample,
                stones,
            ),
        ));
    }
    for sample in 0..2u64 {
        positions.push(bench_position(
            format!("synthetic_s{stones:02}_line_biased_{sample}"),
            -1,
            line_biased_state(
                0xA076_1D64_78BD_642Fu64 ^ (stones as u64).wrapping_mul(0xE703_7ED1) ^ sample,
                stones,
            ),
        ));
    }
    let line_biased = replay_prefix(LINE_BIASED, stones)
        .expect("every configured synthetic bucket fits the fixed replay");
    positions.push(bench_position(
        format!("synthetic_s{stones:02}_replay_prefix"),
        -1,
        line_biased,
    ));
    assert!(positions
        .iter()
        .all(|position| position.state.board().len() == stones && !position.state.is_terminal()));
    BenchBucket {
        name: format!("synthetic_s{stones:02}"),
        corpus: "synthetic",
        family: "none",
        stones,
        positions,
    }
}

fn replay_d6(history: &[(i16, i16)], symmetry: u8) -> HexoState {
    let mut state = HexoState::new();
    for &(q, r) in history {
        let coord =
            d6_transform_coord(HexCoord { q, r }, symmetry).expect("configured D6 image is valid");
        apply_placement(&mut state, Placement { coord })
            .expect("D6 image of an exact fixture history stays legal");
    }
    state
}

fn curated_bucket(name: &'static str, family: &'static str, history: &[(i16, i16)]) -> BenchBucket {
    let positions: Vec<_> = CURATED_D6_IMAGES
        .into_iter()
        .map(|symmetry| {
            bench_position(
                format!("{name}_d6_{symmetry:02}"),
                i16::from(symmetry),
                replay_d6(history, symmetry),
            )
        })
        .collect();
    assert!(positions.iter().all(|position| {
        position.state.board().len() == history.len() && !position.state.is_terminal()
    }));
    BenchBucket {
        name: name.to_owned(),
        corpus: "curated",
        family,
        stones: history.len(),
        positions,
    }
}

fn build_buckets() -> Vec<BenchBucket> {
    let mut buckets: Vec<_> = SYNTHETIC_STONE_BUCKETS
        .into_iter()
        .map(synthetic_bucket)
        .collect();
    buckets.extend([
        curated_bucket(
            "curated_forced_defense_s09",
            "FORCED_DEFENSE",
            FORCED_DEFENSE_MOVES,
        ),
        curated_bucket("curated_deep_win_s15", "DEEP_WIN", DEEP_WIN_MOVES),
        curated_bucket("curated_forced_loss_s17", "FORCED_LOSS", FORCED_LOSS_MOVES),
    ]);
    buckets
}

#[derive(Default)]
struct TimedTotals {
    elapsed: Duration,
    nodes: u64,
    tt_hits: u64,
    peak_tt_bytes: u64,
    wins: usize,
    losses: usize,
    unknown: usize,
}

fn record_status(totals: &mut TimedTotals, status: ProofStatus) {
    match status {
        ProofStatus::Win => totals.wins += 1,
        ProofStatus::Loss => totals.losses += 1,
        ProofStatus::Unknown => totals.unknown += 1,
    }
}

fn time_bucket(solver: &mut TssSolver, bucket: &BenchBucket, caps: &SolveCaps) -> TimedTotals {
    let mut totals = TimedTotals::default();
    for position in &bucket.positions {
        let start = Instant::now();
        let result = solver.solve(&position.state, caps);
        let elapsed = start.elapsed();
        // Only the solver call is inside the measured interval.  Accounting,
        // status classification, and corpus inspection happen afterward.
        totals.elapsed += elapsed;
        totals.nodes = totals.nodes.saturating_add(result.stats.nodes);
        totals.tt_hits = totals.tt_hits.saturating_add(result.stats.tt_hits);
        totals.peak_tt_bytes = totals.peak_tt_bytes.max(result.stats.peak_tt_bytes);
        record_status(&mut totals, result.status);
    }
    totals
}

fn nodes_per_second(nodes: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        nodes as f64 / seconds
    } else {
        0.0
    }
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn median_ns(sorted_ns: &[u128]) -> u128 {
    let middle = sorted_ns.len() / 2;
    if sorted_ns.len() % 2 == 1 {
        sorted_ns[middle]
    } else {
        (sorted_ns[middle - 1] + sorted_ns[middle]) / 2
    }
}

#[test]
fn curated_bench_corpus_matches_shadow_fixture_families() {
    let buckets = build_buckets();
    let curated: Vec<_> = buckets
        .iter()
        .filter(|bucket| bucket.corpus == "curated")
        .collect();
    assert_eq!(curated.len(), 3);
    assert_eq!(
        curated
            .iter()
            .map(|bucket| bucket.positions.len())
            .sum::<usize>(),
        12
    );
    assert_eq!(
        curated
            .iter()
            .map(|bucket| bucket.stones)
            .collect::<Vec<_>>(),
        [9, 15, 17]
    );
    for bucket in curated {
        for position in &bucket.positions {
            if bucket.family == "DEEP_WIN" {
                // The lambda-two fixture is intentionally lambda-one quiet at
                // its root; its two count-three lines become threat-dense one
                // claimant turn later.
                assert!(!position.threatful);
            } else {
                assert!(position.threatful);
            }
        }
    }
}

/// Prints stable, machine-readable bucket rows.  Timing noise affects only the
/// reported throughput, never the solver results or corpus.
#[test]
#[ignore = "timing harness; run explicitly in --release mode"]
fn tss_bench_report() {
    let bucket_caps = SolveCaps {
        node_cap: 100,
        tt_bytes_cap: 64 << 10,
        semantic_horizon: u32::MAX,
    };
    let full_caps = SolveCaps {
        node_cap: 2_000,
        tt_bytes_cap: 64 << 10,
        semantic_horizon: u32::MAX,
    };
    let buckets = build_buckets();
    let synthetic_positions = buckets
        .iter()
        .filter(|bucket| bucket.corpus == "synthetic")
        .map(|bucket| bucket.positions.len())
        .sum::<usize>();
    let curated_positions = buckets
        .iter()
        .filter(|bucket| bucket.corpus == "curated")
        .map(|bucket| bucket.positions.len())
        .sum::<usize>();
    println!(
        "TSS_BENCH_CONFIG schema=2 bucket_node_cap={} full_solve_node_cap={} tt_bytes_cap={} synthetic_buckets={} synthetic_positions={} curated_buckets=3 curated_positions={} uniform_positions_per_bucket=2 line_biased_positions_per_bucket=2 replay_positions_per_bucket=1 cache_handle_scope=per_pass_reused",
        bucket_caps.node_cap,
        full_caps.node_cap,
        bucket_caps.tt_bytes_cap,
        SYNTHETIC_STONE_BUCKETS.len(),
        synthetic_positions,
        curated_positions,
    );

    // A separate solver warms code pages without seeding either timed pass's
    // reusable cache state.
    let mut warmup_solver = TssSolver::default();
    black_box(warmup_solver.solve(&buckets[0].positions[0].state, &bucket_caps));

    // Cap-100 bucket rows retain the old synthetic workload for direct
    // before/after comparison and add one row for each curated family.  One
    // solver handle is deliberately reused throughout this pass so a shared
    // TT implementation is exercised exactly as a long-lived caller uses it.
    let mut bucket_solver = TssSolver::default();
    for bucket in &buckets {
        let totals = time_bucket(&mut bucket_solver, bucket, &bucket_caps);
        let threatful = bucket
            .positions
            .iter()
            .filter(|position| position.threatful)
            .count();
        let nodes_per_sec = nodes_per_second(totals.nodes, totals.elapsed);
        println!(
            "TSS_BENCH_BUCKET bucket={} corpus={} family={} stones_on_board={} node_cap={} positions={} threatful={} nodes={} tt_hits={} peak_tt_bytes={} solve_ns={} solve_seconds={:.9} nodes_per_sec={:.1} wins={} losses={} unknown={} unknown_rate={:.6} gate_nodes_per_sec_20k={}",
            bucket.name,
            bucket.corpus,
            bucket.family,
            bucket.stones,
            bucket_caps.node_cap,
            bucket.positions.len(),
            threatful,
            totals.nodes,
            totals.tt_hits,
            totals.peak_tt_bytes,
            totals.elapsed.as_nanos(),
            totals.elapsed.as_secs_f64(),
            nodes_per_sec,
            totals.wins,
            totals.losses,
            totals.unknown,
            totals.unknown as f64 / bucket.positions.len() as f64,
            nodes_per_sec >= 20_000.0,
        );
    }

    // The full-solve gate is one cap-2000 solve for every position in the
    // extended corpus.  Per-position rows make regressions attributable; the
    // final median is the acceptance number.  A fresh handle is reused for
    // this whole pass, so cap-100 discoveries cannot seed it.
    let mut full_solver = TssSolver::default();
    let mut latencies_ns = Vec::with_capacity(synthetic_positions + curated_positions);
    let mut full_totals = TimedTotals::default();
    for bucket in &buckets {
        for position in &bucket.positions {
            let start = Instant::now();
            let result = full_solver.solve(&position.state, &full_caps);
            let elapsed = start.elapsed();
            latencies_ns.push(elapsed.as_nanos());
            full_totals.elapsed += elapsed;
            full_totals.nodes = full_totals.nodes.saturating_add(result.stats.nodes);
            full_totals.tt_hits = full_totals.tt_hits.saturating_add(result.stats.tt_hits);
            full_totals.peak_tt_bytes = full_totals.peak_tt_bytes.max(result.stats.peak_tt_bytes);
            record_status(&mut full_totals, result.status);
            println!(
                "TSS_BENCH_SOLVE bucket={} corpus={} family={} position={} symmetry={} stones_on_board={} threatful={} node_cap={} nodes={} tt_hits={} peak_tt_bytes={} solve_ns={} solve_ms={:.6} status={}",
                bucket.name,
                bucket.corpus,
                bucket.family,
                position.name,
                position.symmetry,
                bucket.stones,
                position.threatful,
                full_caps.node_cap,
                result.stats.nodes,
                result.stats.tt_hits,
                result.stats.peak_tt_bytes,
                elapsed.as_nanos(),
                elapsed.as_secs_f64() * 1_000.0,
                status_name(result.status),
            );
        }
    }
    latencies_ns.sort_unstable();
    let median_ns = median_ns(&latencies_ns);
    let p95_index = (latencies_ns.len() * 95).div_ceil(100).saturating_sub(1);
    let p95_ns = latencies_ns[p95_index];
    let max_ns = *latencies_ns.last().expect("extended corpus is nonempty");
    println!(
        "TSS_BENCH_CAP2000_SUMMARY positions={} synthetic_positions={} curated_positions={} nodes={} tt_hits={} peak_tt_bytes={} solve_ns={} nodes_per_sec={:.1} wins={} losses={} unknown={} median_ns={} median_ms={:.6} p95_ns={} p95_ms={:.6} max_ns={} max_ms={:.6} gate_median_le_10ms={}",
        latencies_ns.len(),
        synthetic_positions,
        curated_positions,
        full_totals.nodes,
        full_totals.tt_hits,
        full_totals.peak_tt_bytes,
        full_totals.elapsed.as_nanos(),
        nodes_per_second(full_totals.nodes, full_totals.elapsed),
        full_totals.wins,
        full_totals.losses,
        full_totals.unknown,
        median_ns,
        median_ns as f64 / 1_000_000.0,
        p95_ns,
        p95_ns as f64 / 1_000_000.0,
        max_ns,
        max_ns as f64 / 1_000_000.0,
        median_ns <= 10_000_000,
    );
}

/// Zone A/B report: the whole corpus solved twice per cap (zone OFF vs zone
/// ON, side-heuristics off — the production deploy shape) at several node
/// caps, THROUGH the production `tss_solve_verified` path. Routing through
/// the wrapper matters twice over: it applies the production semantic
/// horizons (+8/+12 zone ladder vs the flat +12) — a raw `u32::MAX` horizon
/// makes the zone generator structurally inert (defender budget >= 6 at
/// every node => full legal set) — and it includes the mandatory verify pass,
/// so the timing is the cost production actually pays per solve. The rows
/// answer the deployment questions directly: (1) does the zoned path decide
/// MORE positions within the same budget ("search deeper per node"), (2)
/// what happens to wall time per decision, and (3) does the zone generator
/// engage at all (zone_nodes > 0).
#[test]
#[ignore = "timing harness; run explicitly in --release mode"]
// `tree` (the production wrapper) is python-gated, so this harness needs the
// feature: run with the PYO3_PYTHON + RUSTFLAGS link recipe from the runbook.
#[cfg(feature = "python")]
fn tss_bench_zone_ab() {
    use crate::tree::{tss_solve_verified, TssCounters};

    let buckets = build_buckets();
    for cap in [500u64, 2_000, 8_000] {
        for (label, zone_on) in [("off", false), ("on", true)] {
            let mut solver = TssSolver::default();
            let zone = crate::tss_core::ZoneSearchCaps {
                enabled: zone_on,
                stale_area_filter: false,
                count2_threshold: false,
                pair_commutation: false,
            };
            let mut counters = TssCounters::default();
            let mut wins = 0u64;
            let mut losses = 0u64;
            let mut unknown = 0u64;
            let mut elapsed = std::time::Duration::ZERO;
            let mut latencies_ns = Vec::new();
            for bucket in &buckets {
                for position in &bucket.positions {
                    let start = Instant::now();
                    let solved = tss_solve_verified(
                        &position.state,
                        cap,
                        crate::tss_core::SolveGoal::Both,
                        zone,
                        &mut solver,
                        &mut counters,
                    );
                    let took = start.elapsed();
                    latencies_ns.push(took.as_nanos());
                    elapsed += took;
                    match solved.status {
                        crate::tss_core::ProofStatus::Win => wins += 1,
                        crate::tss_core::ProofStatus::Loss => losses += 1,
                        crate::tss_core::ProofStatus::Unknown => unknown += 1,
                    }
                }
            }
            assert_eq!(
                counters.deep_verify_failed, 0,
                "verify failures during the zone A/B bench"
            );
            latencies_ns.sort_unstable();
            let decided = wins + losses;
            println!(
                "TSS_BENCH_ZONE_AB cap={} zone={} positions={} decided={} wins={} losses={} unknown={} nodes={} zone_nodes={} horizon_retry={} nodes_per_sec={:.1} total_ms={:.3} median_ms={:.6} p95_ms={:.6}",
                cap,
                label,
                latencies_ns.len(),
                decided,
                wins,
                losses,
                unknown,
                counters.deep_nodes,
                counters.zone_nodes,
                counters.horizon_retry,
                nodes_per_second(counters.deep_nodes, elapsed),
                elapsed.as_secs_f64() * 1_000.0,
                median_ns(&latencies_ns) as f64 / 1_000_000.0,
                latencies_ns[(latencies_ns.len() * 95).div_ceil(100).saturating_sub(1)] as f64
                    / 1_000_000.0,
            );
        }
    }
}
