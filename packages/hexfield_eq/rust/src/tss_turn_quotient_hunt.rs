//! NQ4 search-space quotient measurement harness.
//!
//! This is deliberately an ignored, single-threaded measurement test. It does
//! not alter solver choices: all counters live behind the test-only
//! `TSS_TURN_QUOTIENT_TELEMETRY` switch.

use std::collections::HashMap;
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, TurnPhase};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps, ZoneSearchCaps};
use crate::tss_solver::{
    take_quotient_telemetry_report, QuotientTelemetryReport, TssSolver, WidthOptions,
};
use crate::tss_verify::TssVerifier;

const DEFAULT_TT_BYTES: usize = 512 << 20;
const HUMAN_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const HUMAN_MIN_STONES: u32 = 20;
const DOUBLE_FORK_COMPACT: &[(i16, i16)] = &[
    (0, 0),
    (-1, 0),
    (4, 1),
    (1, 0),
    (2, 0),
    (4, 2),
    (4, 3),
    (3, 0),
    (4, 6),
    (4, 4),
    (4, 5),
    (1, 3),
    (2, 3),
    (2, 1),
    (5, 5),
    (3, 3),
    (0, 4),
    (6, 2),
    (-1, 5),
    (0, 5),
    (0, 6),
    (7, 6),
    (1, 6),
    (5, 7),
    (6, 7),
    (6, 6),
    (3, 6),
    (7, 7),
    (5, 6),
    (-1, 6),
    (1, 4),
    (6, 5),
    (7, 4),
    (7, 3),
    (7, 5),
    (-1, 2),
];

#[derive(Clone)]
struct CorpusPosition {
    id: String,
    expect_win: bool,
    state: HexoState,
}

#[derive(Clone)]
struct HumanGame {
    moves: Vec<(i16, i16)>,
}

#[derive(Clone, Copy)]
struct HumanRoot {
    game: usize,
    prefix: usize,
}

#[derive(Default)]
struct Aggregate {
    roots: u64,
    nodes: u64,
    tt_hits: u64,
    wins: u64,
    losses: u64,
    unknowns: u64,
    telemetry: QuotientTelemetryReport,
}

impl Aggregate {
    fn push(&mut self, status: ProofStatus, nodes: u64, tt_hits: u64, q: &QuotientTelemetryReport) {
        self.roots += 1;
        self.nodes = self.nodes.saturating_add(nodes);
        self.tt_hits = self.tt_hits.saturating_add(tt_hits);
        match status {
            ProofStatus::Win => self.wins += 1,
            ProofStatus::Loss => self.losses += 1,
            ProofStatus::Unknown => self.unknowns += 1,
        }
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {$(
                self.telemetry.$field = self.telemetry.$field.saturating_add(q.$field);
            )+};
        }
        add!(
            retained_entries,
            indexed_entries,
            tt_hits,
            d6_index_duplicates,
            d6_index_denominator,
            expanded_unique_positions,
            d6_expanded_duplicates,
            d6_canonicalization_calls,
            d6_canonicalization_nanos,
            horizon_queries,
            horizon_exact_hits,
            horizon_clock_misses,
            horizon_monotone_hits,
            horizon_position_clock_entries,
            horizon_multi_clock_positions,
            horizon_positions,
            horizon_sound_wins,
            horizon_sound_refutations,
            horizon_staged_cutoffs_excluded,
            commutation_eligible_nodes,
            commutation_independent_nodes,
            commutation_shared_window,
            commutation_legality_coupling,
            commutation_threat_response,
        );
    }

    fn print(&self, group: &str) {
        let q = &self.telemetry;
        println!(
            "TQ_SUMMARY group={group} roots={} nodes={} tt_entries={} retained_entries={} tt_hits={} wins={} losses={} unknowns={} d6_tt_dup={} d6_tt_den={} d6_exp_dup={} d6_exp_den={} d6_calls={} d6_ns={} horizon_queries={} horizon_exact_hits={} horizon_misses={} horizon_monotone_hits={} horizon_clock_entries={} horizon_multi_positions={} horizon_positions={} horizon_wins={} horizon_refutations={} staged_cutoffs_excluded={} commute_eligible={} commute_independent={} commute_shared_window={} commute_legality={} commute_threat_response={}",
            self.roots,
            self.nodes,
            q.indexed_entries,
            q.retained_entries,
            self.tt_hits,
            self.wins,
            self.losses,
            self.unknowns,
            q.d6_index_duplicates,
            q.d6_index_denominator,
            q.d6_expanded_duplicates,
            q.expanded_unique_positions,
            q.d6_canonicalization_calls,
            q.d6_canonicalization_nanos,
            q.horizon_queries,
            q.horizon_exact_hits,
            q.horizon_clock_misses,
            q.horizon_monotone_hits,
            q.horizon_position_clock_entries,
            q.horizon_multi_clock_positions,
            q.horizon_positions,
            q.horizon_sound_wins,
            q.horizon_sound_refutations,
            q.horizon_staged_cutoffs_excluded,
            q.commutation_eligible_nodes,
            q.commutation_independent_nodes,
            q.commutation_shared_window,
            q.commutation_legality_coupling,
            q.commutation_threat_response,
        );
    }
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn replay(moves: &[(i16, i16)]) -> HexoState {
    let mut state = HexoState::new();
    for &(q, r) in moves {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(q, r),
            },
        )
        .unwrap_or_else(|error| panic!("illegal replay at ({q},{r}): {error:?}"));
    }
    state
}

fn forcing_corpus() -> Vec<CorpusPosition> {
    let path = format!(
        "{}/rust/corpus/forcing_corpus_moves.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("read forcing corpus");
    let mut positions = Vec::new();
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() || header.starts_with('#') {
            continue;
        }
        let mut id = String::new();
        let mut expect_win = false;
        let mut nstones = 0usize;
        for token in header.split_whitespace().skip(1) {
            let (key, value) = token.split_once('=').expect("forcing k=v");
            match key {
                "id" => id = value.to_owned(),
                "expect" => expect_win = value == "WIN",
                "nstones" => nstones = value.parse().expect("numeric nstones"),
                _ => {}
            }
        }
        let mut moves = Vec::with_capacity(nstones);
        for _ in 0..nstones {
            let mut fields = lines.next().expect("forcing move").split_whitespace();
            moves.push((
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            ));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"));
        positions.push(CorpusPosition {
            id,
            expect_win,
            state: replay(&moves),
        });
    }
    assert_eq!(positions.len(), 19);
    positions
}

fn parse_ints(text: &str) -> Vec<i16> {
    let mut values = Vec::new();
    let mut token = String::new();
    for ch in text.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            token.push(ch);
        } else if !token.is_empty() {
            values.push(token.parse().expect("i16 token"));
            token.clear();
        }
    }
    if !token.is_empty() {
        values.push(token.parse().expect("i16 token"));
    }
    values
}

fn human_games() -> Vec<HumanGame> {
    let path = std::env::var("TSS_TURN_QUOTIENT_HUMAN_CORPUS").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl".to_owned()
    });
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read human corpus {path}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let after = line.split_once("\"moves\":").expect("moves field").1;
            let mut depth = 0i32;
            let mut started = false;
            let mut end = 0usize;
            for (index, byte) in after.bytes().enumerate() {
                match byte {
                    b'[' => {
                        started = true;
                        depth += 1;
                    }
                    b']' if started => {
                        depth -= 1;
                        if depth == 0 {
                            end = index;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let values = parse_ints(&after[..=end]);
            HumanGame {
                moves: values
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect(),
            }
        })
        .collect()
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn human_roots(games: &[HumanGame], sample_n: usize) -> Vec<HumanRoot> {
    let mut roots = Vec::new();
    for (game_index, game) in games.iter().enumerate() {
        let mut state = HexoState::new();
        for (prefix, &(q, r)) in game.moves.iter().enumerate() {
            if !state.is_terminal()
                && matches!(state.phase(), TurnPhase::FirstStone)
                && state.placements_made() >= HUMAN_MIN_STONES
            {
                roots.push(HumanRoot {
                    game: game_index,
                    prefix,
                });
            }
            if state.is_terminal() {
                break;
            }
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal human replay");
        }
    }
    let mut rng = XorShift(HUMAN_SEED | 1);
    for index in (1..roots.len()).rev() {
        let other = (rng.next() % (index as u64 + 1)) as usize;
        roots.swap(index, other);
    }
    roots.truncate(sample_n);
    roots
}

fn assert_no_win_loss_flip(seen: &mut HashMap<String, ProofStatus>, id: &str, status: ProofStatus) {
    if let Some(previous) = seen.insert(id.to_owned(), status) {
        assert!(
            !matches!(
                (previous, status),
                (ProofStatus::Win, ProofStatus::Loss) | (ProofStatus::Loss, ProofStatus::Win)
            ),
            "WIN-vs-LOSS anomaly for {id}: {previous:?} -> {status:?}"
        );
    }
}

fn solve_row(
    id: &str,
    group: &str,
    state: &HexoState,
    caps: SolveCaps,
    width: WidthOptions,
    zone: ZoneSearchCaps,
    aggregate: &mut Aggregate,
    seen: &mut HashMap<String, ProofStatus>,
) -> (ProofStatus, u64, u64) {
    let mut solver = TssSolver::default();
    solver.set_width_options(width);
    solver.set_zone_options(zone);
    let started = Instant::now();
    let result = solver.solve(state, &caps);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    if let Some(cert) = result.cert.as_ref() {
        assert!(
            TssVerifier.verify(state, cert, result.status),
            "certificate verification failed for {id}"
        );
    }
    assert_no_win_loss_flip(seen, id, result.status);
    let telemetry = take_quotient_telemetry_report().unwrap_or_default();
    println!(
        "TQ_ROW group={group} id={id} cap={} horizon={} status={} nodes={} tt_entries={} retained_entries={} tt_hits={} ms={elapsed_ms:.3}",
        caps.node_cap,
        caps.semantic_horizon,
        status_name(result.status),
        result.stats.nodes,
        telemetry.indexed_entries,
        telemetry.retained_entries,
        result.stats.tt_hits,
    );
    aggregate.push(
        result.status,
        result.stats.nodes,
        result.stats.tt_hits,
        &telemetry,
    );
    (result.status, result.stats.nodes, result.stats.tt_hits)
}

#[test]
#[ignore = "NQ4 measurement campaign; release-only, serialized, <=10 minutes"]
fn turn_quotient_campaign() {
    let tt_bytes_cap = std::env::var("TSS_TURN_QUOTIENT_TT_BYTES")
        .ok()
        .map(|value| value.parse().expect("numeric TT bytes"))
        .unwrap_or(DEFAULT_TT_BYTES);
    let corpus = forcing_corpus();
    let identity_caps = SolveCaps {
        node_cap: 10_000,
        tt_bytes_cap,
        semantic_horizon: u32::MAX,
    };

    // Required behavior-identity tripwire: same cold solve, telemetry OFF/ON.
    std::env::remove_var("TSS_TURN_QUOTIENT_TELEMETRY");
    let mut off_solver = TssSolver::default();
    off_solver.set_width_options(WidthOptions::vcf_pair_complete());
    let off = off_solver.solve(&corpus[0].state, &identity_caps);
    std::env::set_var("TSS_TURN_QUOTIENT_TELEMETRY", "1");
    let mut on_solver = TssSolver::default();
    on_solver.set_width_options(WidthOptions::vcf_pair_complete());
    let on = on_solver.solve(&corpus[0].state, &identity_caps);
    assert_eq!(off.status, on.status, "telemetry changed verdict");
    assert_eq!(
        off.stats.nodes, on.stats.nodes,
        "telemetry changed node count"
    );
    assert_eq!(
        off.stats.tt_hits, on.stats.tt_hits,
        "telemetry changed TT hits"
    );
    println!(
        "TQ_IDENTITY id={} status={} nodes={} tt_hits={} result=PASS",
        corpus[0].id,
        status_name(on.status),
        on.stats.nodes,
        on.stats.tt_hits,
    );
    let _ = take_quotient_telemetry_report();

    let mut seen = HashMap::new();
    let mut all = Aggregate::default();
    for cap in [10_000u64, 100_000] {
        let group = format!("forcing_{cap}");
        let mut aggregate = Aggregate::default();
        for position in &corpus {
            let (status, _, _) = solve_row(
                &position.id,
                &group,
                &position.state,
                SolveCaps {
                    node_cap: cap,
                    tt_bytes_cap,
                    semantic_horizon: u32::MAX,
                },
                WidthOptions::vcf_pair_complete(),
                ZoneSearchCaps::default(),
                &mut aggregate,
                &mut seen,
            );
            assert!(
                position.expect_win || status != ProofStatus::Win,
                "NO forcing row {} became WIN",
                position.id
            );
        }
        aggregate.print(&group);
        all.roots += aggregate.roots;
        all.nodes += aggregate.nodes;
        all.tt_hits += aggregate.tt_hits;
        all.wins += aggregate.wins;
        all.losses += aggregate.losses;
        all.unknowns += aggregate.unknowns;
        all.telemetry = {
            let mut merged = all.telemetry.clone();
            let mut helper = Aggregate {
                telemetry: merged,
                ..Aggregate::default()
            };
            helper.push(ProofStatus::Unknown, 0, 0, &aggregate.telemetry);
            merged = helper.telemetry;
            merged
        };
    }

    let compact = replay(DOUBLE_FORK_COMPACT);
    let mut compact_aggregate = Aggregate::default();
    solve_row(
        "double_fork_compact",
        "double_fork_compact",
        &compact,
        SolveCaps {
            node_cap: 100_000,
            tt_bytes_cap,
            semantic_horizon: 45,
        },
        WidthOptions::round3_consume(),
        ZoneSearchCaps {
            enabled: true,
            stale_area_filter: false,
            count2_threshold: true,
            pair_commutation: false,
        },
        &mut compact_aggregate,
        &mut seen,
    );
    compact_aggregate.print("double_fork_compact");

    let games = human_games();
    let roots = human_roots(&games, 100);
    assert_eq!(roots.len(), 100, "human sample must contain 100 roots");
    let mut human_aggregate = Aggregate::default();
    for (rank, root) in roots.iter().enumerate() {
        let state = replay(&games[root.game].moves[..root.prefix]);
        solve_row(
            &format!("human_{rank:03}_g{}_p{}", root.game, root.prefix),
            "human_100_cap10000",
            &state,
            SolveCaps {
                node_cap: 10_000,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            },
            WidthOptions::vcf_pair_complete(),
            ZoneSearchCaps::default(),
            &mut human_aggregate,
            &mut seen,
        );
    }
    human_aggregate.print("human_100_cap10000");
    all.print("forcing_all_rungs");
    println!("TQ_DONE result=PASS anomalies=0");
}
