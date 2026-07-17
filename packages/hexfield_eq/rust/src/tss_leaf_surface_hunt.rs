//! Phase-3 trainer-leaf configuration campaign.
//!
//! Test-only empirical harness. Run explicitly with one test thread; it owns
//! the four process-level TSS lever variables for the duration of the test.

use std::ffi::OsString;
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{KReplyShadowRecord, TssSolver, WidthOptions};
use crate::tss_verify::TssVerifier;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const GAME_COUNT: usize = 50;
const WINDOW_STATES: usize = 6;
const MIN_START_PLY: usize = 8;
const TT_BYTES: usize = 256 * 1024;
const CAPS: [u64; 3] = [500, 2_000, 8_000];
const HORIZONS: [u32; 2] = [8, 16];
const FLAGS: [&str; 5] = [
    "TSS_LAZY_FRONTIER",
    "TSS_INTERIOR_CENSUS_GATE",
    "TSS_SHARED_FRAGMENTS",
    "TSS_K_REPLY_CONSUME",
    "TSS_K_REPLY_SHADOW",
];

#[derive(Clone, Debug)]
struct Game {
    hash: String,
    moves: Vec<(i16, i16)>,
    winner: i8,
}

#[derive(Clone)]
struct Batch {
    hash: String,
    start_ply: usize,
    states: Vec<HexoState>,
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

struct EnvGuard(Vec<(&'static str, Option<OsString>)>);

impl EnvGuard {
    fn set(enabled: &[&'static str]) -> Self {
        let old = FLAGS
            .iter()
            .map(|&name| (name, std::env::var_os(name)))
            .collect::<Vec<_>>();
        for name in FLAGS {
            std::env::remove_var(name);
        }
        for &name in enabled {
            std::env::set_var(name, "1");
        }
        Self(old)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..) {
            if let Some(value) = value {
                std::env::set_var(name, value);
            } else {
                std::env::remove_var(name);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Config {
    A,
    B,
    C,
    D,
    E,
    F,
}

impl Config {
    fn name(self) -> &'static str {
        match self {
            Self::A => "A_NARROW",
            Self::B => "B_WIDE_BARE",
            Self::C => "C_WIDE_LAZY",
            Self::D => "D_WIDE_LAZY_GATE",
            Self::E => "E_WIDE_LAZY_GATE_FRAG",
            Self::F => "F_E_PLUS_K_LITERAL",
        }
    }

    fn flags(self) -> &'static [&'static str] {
        match self {
            Self::A | Self::B => &[],
            Self::C => &["TSS_LAZY_FRONTIER"],
            Self::D => &["TSS_LAZY_FRONTIER", "TSS_INTERIOR_CENSUS_GATE"],
            Self::E => &[
                "TSS_LAZY_FRONTIER",
                "TSS_INTERIOR_CENSUS_GATE",
                "TSS_SHARED_FRAGMENTS",
            ],
            Self::F => &[
                "TSS_LAZY_FRONTIER",
                "TSS_INTERIOR_CENSUS_GATE",
                "TSS_SHARED_FRAGMENTS",
                "TSS_K_REPLY_CONSUME",
                "TSS_K_REPLY_SHADOW",
            ],
        }
    }

    fn make_solver(self) -> TssSolver {
        let mut solver = TssSolver::default();
        if self != Self::A {
            solver.set_width_options(WidthOptions::vcf_pair_complete());
        }
        solver
    }
}

#[derive(Default)]
struct Cell {
    status: Vec<ProofStatus>,
    solve_ns: Vec<u64>,
    cold_ns: Vec<u64>,
    warm_ns: Vec<u64>,
    batch_ns: Vec<u64>,
    wins: u64,
    losses: u64,
    unknowns: u64,
    verified: u64,
    nodes: u64,
    expansions: u64,
    tt_hits: u64,
    tt_entries_max: u64,
    peak_tt_bytes: u64,
    tt_evictions: u64,
    tt_rejections: u64,
    fragment_lookups: u64,
    fragment_hits: u64,
    fragment_imports: u64,
    fragment_entries_max: u64,
    fragment_bytes_max: u64,
    fragment_admissions: u64,
    fragment_replacements: u64,
    fragment_refusals: u64,
    gate_evaluations: u64,
    gate_dismissals: u64,
    gate_nanos: u64,
    shared_reconfigs_max: u64,
    fragment_reconfigs_max: u64,
    shared_slots_max: u64,
    fragment_slots_max: u64,
    k_fires: u64,
    k_urgent: u64,
    k_consumed: u64,
    k_full: Vec<u64>,
    k_retained: Vec<u64>,
}

fn parse_ints(slice: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in slice.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.parse().expect("i16 corpus token"));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.parse().expect("i16 corpus token"));
    }
    out
}

fn parse_line(line: &str) -> Option<Game> {
    let hash_key = "\"game_hash\":\"";
    let hash_tail = &line[line.find(hash_key)? + hash_key.len()..];
    let hash = hash_tail[..hash_tail.find('"')?].to_string();

    let moves_key = "\"moves\":";
    let after = &line[line.find(moves_key)? + moves_key.len()..];
    let start = after.find('[')?;
    let mut depth = 0i32;
    let mut end = None;
    for (index, byte) in after.as_bytes().iter().copied().enumerate().skip(start) {
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let ints = parse_ints(&after[start..=end?]);
    let moves = ints
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();

    let winner_key = "\"winner\":";
    let winner_tail = &line[line.find(winner_key)? + winner_key.len()..];
    let mut winner = String::new();
    for ch in winner_tail.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            winner.push(ch);
        } else if !winner.is_empty() {
            break;
        }
    }
    Some(Game {
        hash,
        moves,
        winner: winner.parse().ok()?,
    })
}

fn corpus_path() -> String {
    std::env::var("TSS_LEAF_SURFACE_CORPUS").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl".to_string()
    })
}

fn load_batches() -> Vec<Batch> {
    let path = corpus_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read leaf-surface corpus {path}: {error}"));
    let games = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_line(line).expect("valid human-corpus row"))
        .filter(|game| {
            matches!(game.winner, -1 | 1) && game.moves.len() >= MIN_START_PLY + WINDOW_STATES
        })
        .collect::<Vec<_>>();
    assert!(games.len() >= GAME_COUNT);

    let mut order = (0..games.len()).collect::<Vec<_>>();
    let mut rng = XorShift(SEED | 1);
    for index in (1..order.len()).rev() {
        let other = (rng.next() % (index as u64 + 1)) as usize;
        order.swap(index, other);
    }

    let mut batches = Vec::with_capacity(GAME_COUNT);
    for &game_index in order.iter().take(GAME_COUNT) {
        let game = &games[game_index];
        let max_start = game.moves.len() - WINDOW_STATES;
        let choices = max_start - MIN_START_PLY + 1;
        let start_ply = MIN_START_PLY + (rng.next() % choices as u64) as usize;
        let mut state = HexoState::new();
        let mut states = Vec::with_capacity(WINDOW_STATES);
        for (ply, &(q, r)) in game.moves.iter().enumerate() {
            if (start_ply..start_ply + WINDOW_STATES).contains(&ply) {
                assert!(!state.is_terminal(), "sampled prefix must be nonterminal");
                states.push(state.clone());
            }
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap_or_else(|error| panic!("illegal replay {} ply {ply}: {error}", game.hash));
            if states.len() == WINDOW_STATES {
                break;
            }
        }
        assert_eq!(states.len(), WINDOW_STATES);
        batches.push(Batch {
            hash: game.hash.clone(),
            start_ply,
            states,
        });
    }

    println!(
        "LEAF_SURFACE_SETUP corpus={} eligible_games={} games={} window_states={} solves_per_cell={} seed=0x{:016X} tt_bytes={} caps={:?} horizons={:?}",
        path,
        games.len(),
        batches.len(),
        WINDOW_STATES,
        batches.len() * WINDOW_STATES,
        SEED,
        TT_BYTES,
        CAPS,
        HORIZONS,
    );
    for (index, batch) in batches.iter().enumerate() {
        println!(
            "LEAF_SURFACE_BATCH index={} game_hash={} start_ply={} end_ply={}",
            index,
            batch.hash,
            batch.start_ply,
            batch.start_ply + WINDOW_STATES - 1
        );
    }
    batches
}

fn percentile(values: &[u64], quantile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

fn absorb_shadow(cell: &mut Cell, records: &[KReplyShadowRecord]) {
    cell.k_fires = cell.k_fires.saturating_add(records.len() as u64);
    for record in records {
        if record.urgent {
            cell.k_urgent = cell.k_urgent.saturating_add(1);
            cell.k_full.push(record.full_quiet as u64);
            cell.k_retained
                .push(record.k_reply.expect("urgent K-reply record") as u64);
        }
        if record.consumed {
            cell.k_consumed = cell.k_consumed.saturating_add(1);
        }
    }
}

fn run_cell(
    batches: &[Batch],
    config: Config,
    cap: u64,
    horizon: u32,
    previous: Option<&[ProofStatus]>,
) -> Cell {
    let _env = EnvGuard::set(config.flags());
    let mut cell = Cell::default();
    let mut solve_index = 0usize;
    for (batch_index, batch) in batches.iter().enumerate() {
        let mut solver = config.make_solver();
        let fragments_before = solver.shared_fragment_store_snapshot();
        let batch_started = Instant::now();
        for (within_batch, state) in batch.states.iter().enumerate() {
            let caps = SolveCaps {
                node_cap: cap,
                tt_bytes_cap: TT_BYTES,
                semantic_horizon: state.placements_made().saturating_add(horizon),
            };
            let started = Instant::now();
            let result = solver.solve_goal(state, &caps, SolveGoal::Win);
            let elapsed = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
            if within_batch == 0 {
                cell.cold_ns.push(elapsed);
            } else {
                cell.warm_ns.push(elapsed);
            }
            cell.solve_ns.push(elapsed);

            if let Some(prior) = previous.map(|statuses| statuses[solve_index]) {
                match (prior, result.status) {
                    (ProofStatus::Unknown, _) => {}
                    (hard, now) if hard == now => {}
                    (hard, ProofStatus::Unknown) => panic!(
                        "STOP nonmonotone hard->UNKNOWN config={} cap={} horizon={} batch={} within={} prior={:?}",
                        config.name(), cap, horizon, batch_index, within_batch, hard
                    ),
                    (hard, now) => panic!(
                        "STOP WIN/LOSS contradiction config={} cap={} horizon={} batch={} within={} prior={:?} now={:?}",
                        config.name(), cap, horizon, batch_index, within_batch, hard, now
                    ),
                }
            }

            match result.status {
                ProofStatus::Win => cell.wins += 1,
                ProofStatus::Loss => cell.losses += 1,
                ProofStatus::Unknown => cell.unknowns += 1,
            }
            if result.status != ProofStatus::Unknown {
                let cert = result.cert.as_ref().unwrap_or_else(|| {
                    panic!(
                        "STOP hard result without cert config={} cap={} horizon={} batch={} within={}",
                        config.name(), cap, horizon, batch_index, within_batch
                    )
                });
                assert!(
                    TssVerifier.verify(state, cert, result.status),
                    "STOP verifier rejected config={} cap={} horizon={} batch={} within={}",
                    config.name(),
                    cap,
                    horizon,
                    batch_index,
                    within_batch
                );
                cell.verified += 1;
            }

            cell.status.push(result.status);
            cell.nodes = cell.nodes.saturating_add(result.stats.nodes);
            cell.expansions = cell.expansions.saturating_add(result.stats.expansions);
            cell.tt_hits = cell.tt_hits.saturating_add(result.stats.tt_hits);
            cell.tt_entries_max = cell.tt_entries_max.max(result.stats.tt_entries);
            cell.peak_tt_bytes = cell.peak_tt_bytes.max(result.stats.peak_tt_bytes);
            cell.tt_evictions = cell.tt_evictions.saturating_add(result.stats.tt_evictions);
            cell.tt_rejections = cell
                .tt_rejections
                .saturating_add(result.stats.tt_admission_rejections);
            cell.fragment_lookups = cell
                .fragment_lookups
                .saturating_add(result.stats.fragment_lookups);
            cell.fragment_hits = cell
                .fragment_hits
                .saturating_add(result.stats.fragment_hits);
            cell.fragment_imports = cell
                .fragment_imports
                .saturating_add(result.stats.fragment_imports);
            cell.fragment_entries_max = cell
                .fragment_entries_max
                .max(result.stats.fragment_store_entries);
            cell.fragment_bytes_max = cell
                .fragment_bytes_max
                .max(result.stats.fragment_store_bytes);
            cell.gate_evaluations = cell
                .gate_evaluations
                .saturating_add(result.stats.interior_gate_evaluations);
            cell.gate_dismissals = cell
                .gate_dismissals
                .saturating_add(result.stats.interior_gate_dismissals);
            cell.gate_nanos = cell
                .gate_nanos
                .saturating_add(result.stats.interior_gate_nanos);

            let reuse = solver.leaf_surface_reuse_snapshot();
            cell.shared_reconfigs_max =
                cell.shared_reconfigs_max.max(reuse.shared_reconfigurations);
            cell.fragment_reconfigs_max = cell
                .fragment_reconfigs_max
                .max(reuse.fragment_reconfigurations);
            cell.shared_slots_max = cell.shared_slots_max.max(reuse.shared_slots);
            cell.fragment_slots_max = cell.fragment_slots_max.max(reuse.fragment_slots);
            let expected_shared = u64::from(config == Config::A);
            let expected_fragment = u64::from(matches!(config, Config::E | Config::F));
            assert_eq!(
                reuse.shared_reconfigurations,
                expected_shared,
                "persistent shared TT reconfigured after solve: {} batch={} within={}",
                config.name(),
                batch_index,
                within_batch
            );
            assert_eq!(
                reuse.fragment_reconfigurations,
                expected_fragment,
                "fragment partition reconfigured after solve: {} batch={} within={}",
                config.name(),
                batch_index,
                within_batch
            );
            if config == Config::F {
                assert!(
                    solver.k_reply_shadow().is_empty(),
                    "wide PN unexpectedly entered narrow K-reply fallback"
                );
            }
            solve_index += 1;
        }
        cell.batch_ns
            .push(batch_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
        let fragments_after = solver.shared_fragment_store_snapshot();
        cell.fragment_admissions = cell.fragment_admissions.saturating_add(
            fragments_after
                .admissions
                .saturating_sub(fragments_before.admissions),
        );
        cell.fragment_replacements = cell.fragment_replacements.saturating_add(
            fragments_after
                .replacements
                .saturating_sub(fragments_before.replacements),
        );
        cell.fragment_refusals = cell.fragment_refusals.saturating_add(
            fragments_after
                .refusals
                .saturating_sub(fragments_before.refusals),
        );
    }
    cell
}

fn print_cell(config: Config, cap: u64, horizon: u32, cell: &Cell) {
    let solves = cell.status.len() as u64;
    let verdicts = cell.wins + cell.losses;
    let total_ns = cell.solve_ns.iter().copied().sum::<u64>();
    let batch_total_ns = cell.batch_ns.iter().copied().sum::<u64>();
    println!(
        "LEAF_SURFACE_CELL config={} cap={} horizon={} solves={} wins={} losses={} unknowns={} verdicts={} verdict_rate={:.6} verified={} median_us={:.3} p90_us={:.3} solve_total_ms={:.3} batch_median_ms={:.3} batch_p90_ms={:.3} batch_total_ms={:.3}",
        config.name(),
        cap,
        horizon,
        solves,
        cell.wins,
        cell.losses,
        cell.unknowns,
        verdicts,
        verdicts as f64 / solves as f64,
        cell.verified,
        percentile(&cell.solve_ns, 0.50) as f64 / 1_000.0,
        percentile(&cell.solve_ns, 0.90) as f64 / 1_000.0,
        total_ns as f64 / 1_000_000.0,
        percentile(&cell.batch_ns, 0.50) as f64 / 1_000_000.0,
        percentile(&cell.batch_ns, 0.90) as f64 / 1_000_000.0,
        batch_total_ns as f64 / 1_000_000.0,
    );
    println!(
        "LEAF_SURFACE_TT config={} cap={} horizon={} nodes={} expansions={} tt_hits={} tt_entries_max={} peak_tt_bytes={} pressure={:.6} evictions={} admission_rejections={}",
        config.name(),
        cap,
        horizon,
        cell.nodes,
        cell.expansions,
        cell.tt_hits,
        cell.tt_entries_max,
        cell.peak_tt_bytes,
        cell.peak_tt_bytes as f64 / TT_BYTES as f64,
        cell.tt_evictions,
        cell.tt_rejections,
    );
    println!(
        "LEAF_SURFACE_LEVERS config={} cap={} horizon={} gate_evaluations={} gate_dismissals={} gate_ms={:.3} fragment_lookups={} fragment_hits={} fragment_imports={} fragment_entries_max={} fragment_bytes_max={} fragment_admissions={} fragment_replacements={} fragment_refusals={}",
        config.name(),
        cap,
        horizon,
        cell.gate_evaluations,
        cell.gate_dismissals,
        cell.gate_nanos as f64 / 1_000_000.0,
        cell.fragment_lookups,
        cell.fragment_hits,
        cell.fragment_imports,
        cell.fragment_entries_max,
        cell.fragment_bytes_max,
        cell.fragment_admissions,
        cell.fragment_replacements,
        cell.fragment_refusals,
    );
    println!(
        "LEAF_SURFACE_REUSE config={} cap={} horizon={} cold_median_us={:.3} successor_median_us={:.3} shared_reconfigs_max={} fragment_reconfigs_max={} shared_slots_max={} fragment_slots_max={} result=PASS",
        config.name(),
        cap,
        horizon,
        percentile(&cell.cold_ns, 0.50) as f64 / 1_000.0,
        percentile(&cell.warm_ns, 0.50) as f64 / 1_000.0,
        cell.shared_reconfigs_max,
        cell.fragment_reconfigs_max,
        cell.shared_slots_max,
        cell.fragment_slots_max,
    );
}

fn run_k_probe(batches: &[Batch], consume: bool) -> Cell {
    let flags = if consume {
        [
            "TSS_INTERIOR_CENSUS_GATE",
            "TSS_K_REPLY_CONSUME",
            "TSS_K_REPLY_SHADOW",
        ]
        .as_slice()
    } else {
        ["TSS_INTERIOR_CENSUS_GATE", "TSS_K_REPLY_SHADOW"].as_slice()
    };
    let _env = EnvGuard::set(flags);
    let mut cell = Cell::default();
    for batch in batches {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::round3_consume());
        let batch_started = Instant::now();
        for (within, state) in batch.states.iter().enumerate() {
            let caps = SolveCaps {
                node_cap: 2_000,
                tt_bytes_cap: TT_BYTES,
                semantic_horizon: state.placements_made().saturating_add(8),
            };
            let started = Instant::now();
            let result = solver.solve_goal(state, &caps, SolveGoal::Win);
            let elapsed = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
            cell.solve_ns.push(elapsed);
            if within == 0 {
                cell.cold_ns.push(elapsed);
            } else {
                cell.warm_ns.push(elapsed);
            }
            match result.status {
                ProofStatus::Win => cell.wins += 1,
                ProofStatus::Loss => cell.losses += 1,
                ProofStatus::Unknown => cell.unknowns += 1,
            }
            if result.status != ProofStatus::Unknown {
                assert!(TssVerifier.verify(
                    state,
                    result.cert.as_ref().expect("K probe hard cert"),
                    result.status
                ));
                cell.verified += 1;
            }
            cell.status.push(result.status);
            cell.nodes = cell.nodes.saturating_add(result.stats.nodes);
            cell.expansions = cell.expansions.saturating_add(result.stats.expansions);
            cell.gate_evaluations = cell
                .gate_evaluations
                .saturating_add(result.stats.interior_gate_evaluations);
            cell.gate_dismissals = cell
                .gate_dismissals
                .saturating_add(result.stats.interior_gate_dismissals);
            absorb_shadow(&mut cell, solver.k_reply_shadow());
        }
        cell.batch_ns
            .push(batch_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
    }
    cell
}

#[test]
#[ignore = "Phase-3 trainer-leaf surface campaign; run explicitly with --test-threads=1"]
fn leaf_surface_campaign() {
    let batches = load_batches();
    let mut matrix = Vec::new();
    for horizon in HORIZONS {
        for cap in CAPS {
            let mut previous: Option<Vec<ProofStatus>> = None;
            let mut e_status = None;
            for config in [Config::A, Config::B, Config::C, Config::D, Config::E] {
                let cell = run_cell(&batches, config, cap, horizon, previous.as_deref());
                print_cell(config, cap, horizon, &cell);
                previous = Some(cell.status.clone());
                if config == Config::E {
                    e_status = Some(cell.status.clone());
                }
                matrix.push((horizon, cap, config, cell));
            }

            if cap == 2_000 && horizon == 8 {
                let f = run_cell(&batches, Config::F, cap, horizon, e_status.as_deref());
                assert_eq!(f.status, e_status.expect("E status retained for literal F"));
                print_cell(Config::F, cap, horizon, &f);
                matrix.push((horizon, cap, Config::F, f));
            }
        }
    }

    let k_off = run_k_probe(&batches, false);
    let k_on = run_k_probe(&batches, true);
    assert_eq!(
        k_off.status, k_on.status,
        "STOP K-reply probe verdict difference"
    );
    let off_total = k_off.solve_ns.iter().copied().sum::<u64>();
    let on_total = k_on.solve_ns.iter().copied().sum::<u64>();
    println!(
        "LEAF_SURFACE_K_PROBE route=round3_narrow_compat batches={} solves={} cap=2000 horizon=8 off_verdicts={} on_verdicts={} off_median_us={:.3} on_median_us={:.3} off_p90_us={:.3} on_p90_us={:.3} off_total_ms={:.3} on_total_ms={:.3} wall_delta_pct={:.6} off_fires={} on_fires={} on_urgent={} on_consumed={} full_median={} retained_median={} result=PASS",
        batches.len(),
        k_off.status.len(),
        k_off.wins + k_off.losses,
        k_on.wins + k_on.losses,
        percentile(&k_off.solve_ns, 0.50) as f64 / 1_000.0,
        percentile(&k_on.solve_ns, 0.50) as f64 / 1_000.0,
        percentile(&k_off.solve_ns, 0.90) as f64 / 1_000.0,
        percentile(&k_on.solve_ns, 0.90) as f64 / 1_000.0,
        off_total as f64 / 1_000_000.0,
        on_total as f64 / 1_000_000.0,
        if off_total == 0 { 0.0 } else { (on_total as f64 / off_total as f64 - 1.0) * 100.0 },
        k_off.k_fires,
        k_on.k_fires,
        k_on.k_urgent,
        k_on.k_consumed,
        percentile(&k_on.k_full, 0.50),
        percentile(&k_on.k_retained, 0.50),
    );

    assert_eq!(matrix.len(), HORIZONS.len() * CAPS.len() * 5 + 1);
    println!(
        "LEAF_SURFACE_DONE cells={} hard_certificates_verified={} contradictions=0 monotonicity=PASS reuse=PASS",
        matrix.len(),
        matrix.iter().map(|(_, _, _, cell)| cell.verified).sum::<u64>()
            + k_off.verified
            + k_on.verified,
    );
}
