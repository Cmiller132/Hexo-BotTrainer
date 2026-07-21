//! NQ4 search-space quotient and G2R9 shared-fragment measurement harness.
//!
//! These are deliberately ignored, single-threaded measurement tests. NQ4's
//! counters do not alter solver choices; the G2R9 lanes intentionally A/B the
//! default-off shared-fragment policy through its test-only deterministic
//! setter.

use std::collections::HashMap;
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase, WindowKey};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps, SolveStats, ZoneSearchCaps};
use crate::tss_solver::{
    take_quotient_telemetry_report, QuotientTelemetryReport, SharedFragmentStoreSnapshot,
    TssSolver, WidthOptions,
};
use crate::tss_verify::{CertNode, TssCertificate, TssVerifier};

const DEFAULT_TT_BYTES: usize = 512 << 20;
const DEFAULT_LAZY_EQ_TT_BYTES: usize = 2 << 30;
const DEFAULT_SHARED_FRAGMENT_TT_BYTES: usize = 512 << 20;
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

#[derive(Default)]
struct LazyAggregate {
    roots: u64,
    nodes: u64,
    indexed_entries: u64,
    retained_entries: u64,
    peak_tt_bytes: u64,
    tt_hits: u64,
    elapsed_ms: f64,
}

impl LazyAggregate {
    fn push(&mut self, run: &LazyRun) {
        self.roots = self.roots.saturating_add(1);
        self.nodes = self.nodes.saturating_add(run.nodes);
        self.indexed_entries = self
            .indexed_entries
            .saturating_add(run.telemetry.indexed_entries);
        self.retained_entries = self
            .retained_entries
            .saturating_add(run.telemetry.retained_entries);
        self.peak_tt_bytes = self.peak_tt_bytes.saturating_add(run.peak_tt_bytes);
        self.tt_hits = self.tt_hits.saturating_add(run.tt_hits);
        self.elapsed_ms += run.elapsed_ms;
    }

    fn print(&self, cohort: &str, mode: &str) {
        println!(
            "LF_EQ_SUMMARY cohort={cohort} mode={mode} roots={} nodes={} indexed_entries={} retained_entries={} peak_tt_bytes={} tt_hits={} ms={:.3}",
            self.roots,
            self.nodes,
            self.indexed_entries,
            self.retained_entries,
            self.peak_tt_bytes,
            self.tt_hits,
            self.elapsed_ms,
        );
    }
}

struct LazyRun {
    status: ProofStatus,
    nodes: u64,
    peak_tt_bytes: u64,
    tt_hits: u64,
    cert: Option<TssCertificate>,
    cert_bytes: Vec<u8>,
    telemetry: QuotientTelemetryReport,
    elapsed_ms: f64,
}

#[derive(Clone)]
struct SharedFragmentCase {
    cohort: String,
    id: String,
    state: HexoState,
    caps: SolveCaps,
    width: WidthOptions,
    zone: ZoneSearchCaps,
    /// The forcing corpus's five NO controls must never become WIN. A
    /// verifier-accepted WIN there is still a campaign-stopping oracle
    /// disagreement, even if both A/B modes happen to reproduce it.
    forbid_win: bool,
}

struct SharedFragmentRun {
    status: ProofStatus,
    strict_verified_hard: bool,
    stats: SolveStats,
    store: SharedFragmentStoreSnapshot,
    elapsed_ms: f64,
}

#[derive(Default)]
struct SharedFragmentImprovementCensus {
    count: u64,
    expansions_saved: i128,
}

impl SharedFragmentImprovementCensus {
    fn record(&mut self, expansions_saved: i128) {
        self.count = self.count.saturating_add(1);
        self.expansions_saved += expansions_saved;
    }

    fn print(&self, lane: &str, lazy: bool) {
        println!(
            "SF_IMPROVEMENT_CENSUS lane={lane} lazy={} count={} expansions_saved={}",
            on_off(lazy),
            self.count,
            self.expansions_saved,
        );
    }
}

#[derive(Default)]
struct SharedFragmentAggregate {
    roots: u64,
    nodes: u64,
    fragment_lookups: u64,
    fragment_hits: u64,
    fragment_imports: u64,
    store_entries_sum: u64,
    store_bytes_sum: u64,
    max_store_entries: u64,
    max_store_bytes: u64,
    elapsed_ms: f64,
}

impl SharedFragmentAggregate {
    fn push(&mut self, run: &SharedFragmentRun) {
        self.roots = self.roots.saturating_add(1);
        self.nodes = self.nodes.saturating_add(run.stats.nodes);
        self.fragment_lookups = self
            .fragment_lookups
            .saturating_add(run.stats.fragment_lookups);
        self.fragment_hits = self.fragment_hits.saturating_add(run.stats.fragment_hits);
        self.fragment_imports = self
            .fragment_imports
            .saturating_add(run.stats.fragment_imports);
        self.store_entries_sum = self.store_entries_sum.saturating_add(run.store.entries);
        self.store_bytes_sum = self.store_bytes_sum.saturating_add(run.store.bytes);
        self.max_store_entries = self.max_store_entries.max(run.store.entries);
        self.max_store_bytes = self.max_store_bytes.max(run.store.bytes);
        self.elapsed_ms += run.elapsed_ms;
    }

    fn print(&self, lazy: bool, fragments: bool, phase: &str) {
        let hit_rate = ratio(self.fragment_hits, self.fragment_lookups);
        println!(
            "SF_SUMMARY lazy={} fragments={} phase={phase} roots={} nodes={} expansions={} ms={:.3} fragment_lookups={} fragment_hits={} fragment_hit_rate={hit_rate:.6} fragment_imports={} store_entries_sum={} store_bytes_sum={} max_store_entries={} max_store_bytes={}",
            on_off(lazy),
            on_off(fragments),
            self.roots,
            self.nodes,
            self.nodes,
            self.elapsed_ms,
            self.fragment_lookups,
            self.fragment_hits,
            self.fragment_imports,
            self.store_entries_sum,
            self.store_bytes_sum,
            self.max_store_entries,
            self.max_store_bytes,
        );
    }
}

/// Restores the process-global lazy-frontier setting even on an ordinary test
/// return. G2R9 campaign commands are required to be serialized; this guard
/// also keeps later ignored tests in that process from inheriting a mode.
struct LazyFrontierEnvGuard {
    previous: Option<std::ffi::OsString>,
}

impl LazyFrontierEnvGuard {
    fn new() -> Self {
        Self {
            previous: std::env::var_os("TSS_LAZY_FRONTIER"),
        }
    }

    fn set(&self, enabled: bool) {
        if enabled {
            std::env::set_var("TSS_LAZY_FRONTIER", "1");
        } else {
            std::env::remove_var("TSS_LAZY_FRONTIER");
        }
    }
}

impl Drop for LazyFrontierEnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => std::env::set_var("TSS_LAZY_FRONTIER", value),
            None => std::env::remove_var("TSS_LAZY_FRONTIER"),
        }
    }
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

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn signed_delta_percent(on: u64, off: u64) -> f64 {
    if off == 0 {
        if on == 0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (on as f64 - off as f64) * 100.0 / off as f64
    }
}

fn signed_delta_percent_f64(on: f64, off: f64) -> f64 {
    if off == 0.0 {
        if on == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (on - off) * 100.0 / off
    }
}

/// `TSS_SHARED_FRAGMENT_LAZY_MODE=off|on|both` lets a regeneration command
/// split the two composition lanes without changing what either lane tests.
fn shared_fragment_lazy_modes(default_both: bool) -> Vec<bool> {
    match std::env::var("TSS_SHARED_FRAGMENT_LAZY_MODE")
        .unwrap_or_else(|_| (if default_both { "both" } else { "off" }).to_owned())
        .as_str()
    {
        "off" => vec![false],
        "on" => vec![true],
        "both" => vec![false, true],
        value => panic!("TSS_SHARED_FRAGMENT_LAZY_MODE must be off, on, or both; got {value:?}"),
    }
}

fn shared_fragment_reduced_budgets() -> Vec<usize> {
    let value =
        std::env::var("TSS_SHARED_FRAGMENT_REDUCED_TT_BYTES").unwrap_or_else(|_| "both".to_owned());
    if value == "both" {
        return vec![512usize << 20, 1usize << 30];
    }
    let budgets = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.parse::<usize>()
                .expect("numeric TSS_SHARED_FRAGMENT_REDUCED_TT_BYTES")
        })
        .collect::<Vec<_>>();
    assert!(
        !budgets.is_empty(),
        "TSS_SHARED_FRAGMENT_REDUCED_TT_BYTES is empty"
    );
    budgets
}

fn shared_fragment_reduced_ladder() -> Vec<u64> {
    let mut ladder = match std::env::var("TSS_SHARED_FRAGMENT_REDUCED_LADDER") {
        Ok(value) => value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| {
                item.parse::<u64>()
                    .expect("numeric TSS_SHARED_FRAGMENT_REDUCED_LADDER")
            })
            .collect::<Vec<_>>(),
        Err(_) => vec![10_000, 100_000, 1_000_000, 20_000_000],
    };
    assert!(
        !ladder.is_empty(),
        "TSS_SHARED_FRAGMENT_REDUCED_LADDER is empty"
    );
    assert!(
        ladder.windows(2).all(|pair| pair[0] < pair[1]),
        "TSS_SHARED_FRAGMENT_REDUCED_LADDER must be strictly increasing"
    );
    let max_cap = std::env::var("TSS_SHARED_FRAGMENT_REDUCED_MAX_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric reduced-TT max cap"))
        .unwrap_or(u64::MAX);
    ladder.retain(|cap| *cap <= max_cap);
    assert!(
        !ladder.is_empty(),
        "TSS_SHARED_FRAGMENT_REDUCED_MAX_CAP removed every ladder rung"
    );
    ladder
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

fn shared_fragment_cases(tt_bytes_cap: usize) -> Vec<SharedFragmentCase> {
    let corpus = forcing_corpus();
    let mut cases = Vec::with_capacity(139);
    for cap in [10_000u64, 100_000] {
        for position in &corpus {
            cases.push(SharedFragmentCase {
                cohort: format!("forcing_{cap}"),
                id: position.id.clone(),
                state: position.state.clone(),
                caps: SolveCaps {
                    node_cap: cap,
                    tt_bytes_cap,
                    semantic_horizon: u32::MAX,
                },
                width: WidthOptions::vcf_pair_complete(),
                zone: ZoneSearchCaps::default(),
                forbid_win: !position.expect_win,
            });
        }
    }

    cases.push(SharedFragmentCase {
        cohort: "double_fork_compact".to_owned(),
        id: "double_fork_compact".to_owned(),
        state: replay(DOUBLE_FORK_COMPACT),
        caps: SolveCaps {
            node_cap: 100_000,
            tt_bytes_cap,
            semantic_horizon: 45,
        },
        width: WidthOptions::round3_consume(),
        zone: ZoneSearchCaps {
            enabled: true,
            stale_area_filter: false,
            count2_threshold: true,
            pair_commutation: false,
        },
        forbid_win: false,
    });

    let games = human_games();
    let roots = human_roots(&games, 100);
    assert_eq!(roots.len(), 100, "human sample must contain 100 roots");
    for (rank, root) in roots.into_iter().enumerate() {
        cases.push(SharedFragmentCase {
            cohort: "human_100_cap10000".to_owned(),
            id: format!("human_{rank:03}_g{}_p{}", root.game, root.prefix),
            state: replay(&games[root.game].moves[..root.prefix]),
            caps: SolveCaps {
                node_cap: 10_000,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            },
            width: WidthOptions::vcf_pair_complete(),
            zone: ZoneSearchCaps::default(),
            forbid_win: false,
        });
    }
    assert_eq!(cases.len(), 139, "G2R9 campaign root count drifted");
    cases
}

fn configured_fragment_solver(
    fragments: bool,
    width: WidthOptions,
    zone: ZoneSearchCaps,
) -> TssSolver {
    let mut solver = TssSolver::default();
    // The production path reads TSS_SHARED_FRAGMENTS once in Default. The
    // test-only setter gives one serialized process a deterministic A/B pair
    // without racing process-global environment changes.
    solver.set_shared_fragments_for_test(fragments);
    solver.set_width_options(width);
    solver.set_zone_options(zone);
    solver
}

fn solve_shared_fragment_once(
    solver: &mut TssSolver,
    case: &SharedFragmentCase,
    fragments: bool,
    phase: &str,
    lazy: bool,
) -> SharedFragmentRun {
    let started = Instant::now();
    let result = solver.solve(&case.state, &case.caps);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    let certificate_verified = result
        .cert
        .as_ref()
        .map(|cert| TssVerifier.verify(&case.state, cert, result.status));
    if result.status != ProofStatus::Unknown {
        assert!(
            result.cert.is_some(),
            "SF_STOP hard verdict without certificate: id={} fragments={} phase={phase} lazy={} status={}",
            case.id,
            on_off(fragments),
            on_off(lazy),
            status_name(result.status),
        );
        assert_eq!(
            certificate_verified,
            Some(true),
            "SF_STOP unverified hard verdict: id={} fragments={} phase={phase} lazy={} status={}",
            case.id,
            on_off(fragments),
            on_off(lazy),
            status_name(result.status),
        );
    }
    if certificate_verified.is_some() {
        assert!(
            certificate_verified == Some(true),
            "SF_STOP verifier rejection: id={} fragments={} phase={phase} lazy={} status={}",
            case.id,
            on_off(fragments),
            on_off(lazy),
            status_name(result.status),
        );
    }
    assert!(
        !(case.forbid_win && result.status == ProofStatus::Win),
        "SF_STOP forcing NO control became WIN: id={} fragments={} phase={phase} lazy={}",
        case.id,
        on_off(fragments),
        on_off(lazy),
    );
    SharedFragmentRun {
        status: result.status,
        strict_verified_hard: result.status != ProofStatus::Unknown
            && certificate_verified == Some(true),
        stats: result.stats,
        store: solver.shared_fragment_store_snapshot(),
        elapsed_ms,
    }
}

fn assert_fragment_verdict_identity(
    case: &SharedFragmentCase,
    lazy: bool,
    label: &str,
    left: ProofStatus,
    right: ProofStatus,
) {
    assert_eq!(
        left,
        right,
        "SF_STOP verdict flip: cohort={} id={} cap={} horizon={} lazy={} comparison={label} left={} right={}",
        case.cohort,
        case.id,
        case.caps.node_cap,
        case.caps.semantic_horizon,
        on_off(lazy),
        status_name(left),
        status_name(right),
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_fragment_warm_contract(
    cohort: &str,
    root: &str,
    rung: u64,
    horizon: u32,
    lazy: bool,
    label: &str,
    off_status: ProofStatus,
    on_status: ProofStatus,
    on_strict_verified_hard: bool,
    off_expansions: u64,
    on_expansions: u64,
) -> Option<i128> {
    if off_status == on_status {
        return None;
    }

    let permitted = off_status == ProofStatus::Unknown
        && matches!(on_status, ProofStatus::Win | ProofStatus::Loss)
        && on_strict_verified_hard;
    assert!(
        permitted,
        "SF_STOP warm verdict contract violation: cohort={cohort} root={root} rung={rung} horizon={horizon} lazy={} comparison={label} off={} on={} on_strict_verified_hard={on_strict_verified_hard}",
        on_off(lazy),
        status_name(off_status),
        status_name(on_status),
    );

    let expansions_saved = i128::from(off_expansions) - i128::from(on_expansions);
    println!(
        "SF_MONOTONE_IMPROVEMENT cohort={cohort} root={root} rung={rung} horizon={horizon} lazy={} comparison={label} off_verdict={} on_verdict={} strict_verifier=PASS off_expansions={off_expansions} on_expansions={on_expansions} expansions_saved={expansions_saved}",
        on_off(lazy),
        status_name(off_status),
        status_name(on_status),
    );
    Some(expansions_saved)
}

#[allow(clippy::too_many_arguments)]
fn print_shared_fragment_row(
    case: &SharedFragmentCase,
    lazy: bool,
    off_cold: &SharedFragmentRun,
    off_warm: &SharedFragmentRun,
    on_cold: &SharedFragmentRun,
    on_warm: &SharedFragmentRun,
) {
    println!(
        "SF_ROW cohort={} id={} cap={} horizon={} tt_bytes_cap={} lazy={} off_cold_status={} off_warm_status={} on_cold_status={} on_warm_status={} off_cold_nodes={} off_cold_expansions={} off_warm_nodes={} off_warm_expansions={} on_cold_nodes={} on_cold_expansions={} on_warm_nodes={} on_warm_expansions={} off_cold_ms={:.3} off_warm_ms={:.3} on_cold_ms={:.3} on_warm_ms={:.3} on_cold_lookups={} on_cold_hits={} on_cold_imports={} on_warm_lookups={} on_warm_hits={} on_warm_hit_rate={:.6} on_warm_imports={} store_entries={} store_bytes={} store_peak_bytes={} stored_nodes={} stored_edges={} admissions={} replacements={} refusals={}",
        case.cohort,
        case.id,
        case.caps.node_cap,
        case.caps.semantic_horizon,
        case.caps.tt_bytes_cap,
        on_off(lazy),
        status_name(off_cold.status),
        status_name(off_warm.status),
        status_name(on_cold.status),
        status_name(on_warm.status),
        off_cold.stats.nodes,
        off_cold.stats.nodes,
        off_warm.stats.nodes,
        off_warm.stats.nodes,
        on_cold.stats.nodes,
        on_cold.stats.nodes,
        on_warm.stats.nodes,
        on_warm.stats.nodes,
        off_cold.elapsed_ms,
        off_warm.elapsed_ms,
        on_cold.elapsed_ms,
        on_warm.elapsed_ms,
        on_cold.stats.fragment_lookups,
        on_cold.stats.fragment_hits,
        on_cold.stats.fragment_imports,
        on_warm.stats.fragment_lookups,
        on_warm.stats.fragment_hits,
        ratio(
            on_warm.stats.fragment_hits,
            on_warm.stats.fragment_lookups
        ),
        on_warm.stats.fragment_imports,
        on_warm.store.entries,
        on_warm.store.bytes,
        on_warm.store.peak_bytes,
        on_warm.store.stored_nodes,
        on_warm.store.stored_edges,
        on_warm.store.admissions,
        on_warm.store.replacements,
        on_warm.store.refusals,
    );
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

/// The certificate has no public wire codec. This test-only encoder covers
/// every field explicitly so R-LF1 compares deterministic bytes as well as the
/// type's structural equality.
fn certificate_bytes(cert: &Option<TssCertificate>) -> Vec<u8> {
    fn put_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn put_len(out: &mut Vec<u8>, value: usize) {
        out.extend_from_slice(&(value as u64).to_le_bytes());
    }
    fn put_player(out: &mut Vec<u8>, player: Player) {
        out.push(match player {
            Player::Player0 => 0,
            Player::Player1 => 1,
        });
    }
    fn put_coord(out: &mut Vec<u8>, coord: HexCoord) {
        out.extend_from_slice(&coord.q.to_le_bytes());
        out.extend_from_slice(&coord.r.to_le_bytes());
    }
    fn put_window(out: &mut Vec<u8>, window: WindowKey) {
        put_coord(out, window.start);
        out.push(window.axis.index());
    }

    let mut out = Vec::new();
    let Some(cert) = cert else {
        out.push(0);
        return out;
    };
    out.push(1);
    put_len(&mut out, cert.root.occupancy.len());
    for coord in &cert.root.occupancy {
        put_coord(&mut out, *coord);
    }
    put_len(&mut out, cert.root.owners.len());
    for owner in &cert.root.owners {
        put_player(&mut out, *owner);
    }
    put_player(&mut out, cert.root.current_player);
    match cert.root.phase {
        TurnPhase::Opening => out.push(0),
        TurnPhase::FirstStone => out.push(1),
        TurnPhase::SecondStone { first } => {
            out.push(2);
            put_coord(&mut out, first);
        }
    }
    put_u32(&mut out, cert.root.placements_made);
    match cert.root.terminal {
        None => out.push(0),
        Some(outcome) => {
            out.push(1);
            put_player(&mut out, outcome.winner);
            put_u32(&mut out, outcome.placements);
        }
    }
    put_player(&mut out, cert.claimant);
    put_u32(&mut out, cert.root_node);
    put_len(&mut out, cert.nodes.len());
    for node in &cert.nodes {
        match node {
            CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => {
                panic!("legacy canonical encoder is legacy-only")
            }
            CertNode::OrCompletion {
                mv,
                witness,
                completion_ply,
            } => {
                out.push(0);
                put_coord(&mut out, *mv);
                put_window(&mut out, *witness);
                put_u32(&mut out, *completion_ply);
            }
            CertNode::Win {
                witness,
                count,
                budget,
                resolution_ply,
            } => {
                out.push(1);
                put_window(&mut out, *witness);
                out.push(*count);
                out.push(*budget);
                put_u32(&mut out, *resolution_ply);
            }
            CertNode::Loss {
                witnesses,
                resolution_ply,
            } => {
                out.push(2);
                put_len(&mut out, witnesses.len());
                for witness in witnesses {
                    put_window(&mut out, *witness);
                }
                put_u32(&mut out, *resolution_ply);
            }
            CertNode::Choice { mv, child } => {
                out.push(3);
                put_coord(&mut out, *mv);
                put_u32(&mut out, *child);
            }
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                out.push(4);
                put_len(&mut out, edges.len());
                for edge in edges {
                    put_coord(&mut out, edge.mv);
                    put_u32(&mut out, edge.child);
                }
                out.push(u8::from(*implicit_dispatch));
                match zone {
                    None => out.push(0),
                    Some(zone) => {
                        out.push(1);
                        put_u32(&mut out, zone.d);
                        put_u32(&mut out, zone.build_horizon);
                    }
                }
                put_len(&mut out, commutations.len());
                for item in commutations {
                    put_coord(&mut out, item.first);
                    put_coord(&mut out, item.omitted_second);
                    put_u32(&mut out, item.first_child);
                    put_u32(&mut out, item.mirror_child);
                }
            }
        }
    }
    put_u32(&mut out, cert.semantic_horizon);
    out
}

fn solve_lazy_row(
    id: &str,
    state: &HexoState,
    caps: &SolveCaps,
    width: WidthOptions,
    zone: ZoneSearchCaps,
    lazy: bool,
) -> LazyRun {
    if lazy {
        std::env::set_var("TSS_LAZY_FRONTIER", "1");
    } else {
        std::env::remove_var("TSS_LAZY_FRONTIER");
    }
    let mut solver = TssSolver::default();
    solver.set_width_options(width);
    solver.set_zone_options(zone);
    let started = Instant::now();
    let result = solver.solve(state, caps);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    if let Some(cert) = result.cert.as_ref() {
        assert!(
            TssVerifier.verify(state, cert, result.status),
            "R-LF1 certificate verification failed for {id} lazy={lazy}"
        );
    }
    let telemetry = take_quotient_telemetry_report().unwrap_or_default();
    let cert_bytes = certificate_bytes(&result.cert);
    LazyRun {
        status: result.status,
        nodes: result.stats.nodes,
        peak_tt_bytes: result.stats.peak_tt_bytes,
        tt_hits: result.stats.tt_hits,
        cert: result.cert,
        cert_bytes,
        telemetry,
        elapsed_ms,
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_lazy_row(
    cohort: &str,
    id: &str,
    state: &HexoState,
    caps: SolveCaps,
    width: WidthOptions,
    zone: ZoneSearchCaps,
    off_aggregate: &mut LazyAggregate,
    on_aggregate: &mut LazyAggregate,
) {
    let off = solve_lazy_row(id, state, &caps, width, zone, false);
    let on = solve_lazy_row(id, state, &caps, width, zone, true);
    let context = format!(
        "cohort={cohort} id={id} cap={} horizon={} tt_bytes_cap={}",
        caps.node_cap, caps.semantic_horizon, caps.tt_bytes_cap
    );
    assert_eq!(off.status, on.status, "R-LF1 verdict mismatch: {context}");
    assert_eq!(off.cert, on.cert, "R-LF1 certificate mismatch: {context}");
    assert_eq!(
        off.cert_bytes, on.cert_bytes,
        "R-LF1 certificate-byte mismatch: {context}"
    );
    assert_eq!(off.nodes, on.nodes, "R-LF1 node mismatch: {context}");
    println!(
        "LF_EQ_ROW cohort={cohort} id={id} status={} nodes={} cert_bytes={} off_indexed={} on_indexed={} off_retained={} on_retained={} off_peak_tt_bytes={} on_peak_tt_bytes={} off_tt_hits={} on_tt_hits={} off_ms={:.3} on_ms={:.3}",
        status_name(on.status),
        on.nodes,
        on.cert_bytes.len(),
        off.telemetry.indexed_entries,
        on.telemetry.indexed_entries,
        off.telemetry.retained_entries,
        on.telemetry.retained_entries,
        off.peak_tt_bytes,
        on.peak_tt_bytes,
        off.tt_hits,
        on.tt_hits,
        off.elapsed_ms,
        on.elapsed_ms,
    );
    off_aggregate.push(&off);
    on_aggregate.push(&on);
}

fn shared_fragment_mutation_control(corpus: &[CorpusPosition], tt_bytes_cap: usize, lazy: bool) {
    let first = corpus
        .iter()
        .find(|position| position.id == "0hz3hty")
        .expect("mutation-control first root");
    let different = corpus
        .iter()
        .find(|position| position.id == "8is963b")
        .expect("mutation-control different root");
    let first_case = SharedFragmentCase {
        cohort: "mutation_control".to_owned(),
        id: first.id.clone(),
        state: first.state.clone(),
        caps: SolveCaps {
            node_cap: 100_000,
            tt_bytes_cap,
            semantic_horizon: u32::MAX,
        },
        width: WidthOptions::vcf_pair_complete(),
        zone: ZoneSearchCaps::default(),
        forbid_win: !first.expect_win,
    };
    let different_case = SharedFragmentCase {
        cohort: "mutation_control".to_owned(),
        id: different.id.clone(),
        state: different.state.clone(),
        caps: first_case.caps,
        width: first_case.width,
        zone: first_case.zone,
        forbid_win: !different.expect_win,
    };

    let mut warm_on = configured_fragment_solver(
        true,
        WidthOptions::vcf_pair_complete(),
        ZoneSearchCaps::default(),
    );
    let seeded = solve_shared_fragment_once(&mut warm_on, &first_case, true, "seed", lazy);
    let mutated =
        solve_shared_fragment_once(&mut warm_on, &different_case, true, "different", lazy);

    let mut fresh_on = configured_fragment_solver(
        true,
        WidthOptions::vcf_pair_complete(),
        ZoneSearchCaps::default(),
    );
    let fresh_on_result =
        solve_shared_fragment_once(&mut fresh_on, &different_case, true, "fresh", lazy);
    let mut fresh_off = configured_fragment_solver(
        false,
        WidthOptions::vcf_pair_complete(),
        ZoneSearchCaps::default(),
    );
    let fresh_off_result =
        solve_shared_fragment_once(&mut fresh_off, &different_case, false, "fresh", lazy);

    assert_fragment_verdict_identity(
        &different_case,
        lazy,
        "fresh-off-vs-fresh-on",
        fresh_off_result.status,
        fresh_on_result.status,
    );
    assert_fragment_verdict_identity(
        &different_case,
        lazy,
        "seeded-different-vs-fresh-on",
        mutated.status,
        fresh_on_result.status,
    );
    assert_fragment_verdict_identity(
        &different_case,
        lazy,
        "seeded-different-vs-fresh-off",
        mutated.status,
        fresh_off_result.status,
    );
    println!(
        "SF_MUTATION lazy={} seed_id={} seed_status={} different_id={} status={} fresh_on_status={} fresh_off_status={} different_nodes={} different_lookups={} different_hits={} different_imports={} store_entries={} store_bytes={} cold_contract=PASS result=PASS",
        on_off(lazy),
        first_case.id,
        status_name(seeded.status),
        different_case.id,
        status_name(mutated.status),
        status_name(fresh_on_result.status),
        status_name(fresh_off_result.status),
        mutated.stats.nodes,
        mutated.stats.fragment_lookups,
        mutated.stats.fragment_hits,
        mutated.stats.fragment_imports,
        mutated.store.entries,
        mutated.store.bytes,
    );
}

#[test]
#[ignore = "G2R9 shared-fragment soundness/warm campaign; release-only and serialized"]
fn shared_fragment_soundness_and_warm_campaign() {
    let tt_bytes_cap = std::env::var("TSS_SHARED_FRAGMENT_TT_BYTES")
        .ok()
        .map(|value| value.parse().expect("numeric shared-fragment TT bytes"))
        .unwrap_or(DEFAULT_SHARED_FRAGMENT_TT_BYTES);
    let lazy_guard = LazyFrontierEnvGuard::new();
    let mut cases = shared_fragment_cases(tt_bytes_cap);
    let case_filter = std::env::var("TSS_SHARED_FRAGMENT_CASE_ID").ok();
    if let Some(id) = case_filter.as_deref() {
        cases.retain(|case| case.id == id);
        assert!(!cases.is_empty(), "unknown shared-fragment case id {id}");
    }
    let corpus = forcing_corpus();

    for lazy in shared_fragment_lazy_modes(true) {
        lazy_guard.set(lazy);
        let mut off_cold_total = SharedFragmentAggregate::default();
        let mut off_warm_total = SharedFragmentAggregate::default();
        let mut on_cold_total = SharedFragmentAggregate::default();
        let mut on_warm_total = SharedFragmentAggregate::default();
        let mut improvements = SharedFragmentImprovementCensus::default();

        for case in &cases {
            let mut off_solver = configured_fragment_solver(false, case.width, case.zone);
            let off_cold = solve_shared_fragment_once(&mut off_solver, case, false, "cold", lazy);
            let off_warm = solve_shared_fragment_once(&mut off_solver, case, false, "warm", lazy);

            let mut on_solver = configured_fragment_solver(true, case.width, case.zone);
            let on_cold = solve_shared_fragment_once(&mut on_solver, case, true, "cold", lazy);
            let on_warm = solve_shared_fragment_once(&mut on_solver, case, true, "warm", lazy);

            // Emit the complete A/B row before tripwires so any mandated stop
            // still leaves enough evidence for a single-root blocker report.
            print_shared_fragment_row(case, lazy, &off_cold, &off_warm, &on_cold, &on_warm);
            assert_fragment_verdict_identity(
                case,
                lazy,
                "off-cold-vs-off-warm-baseline",
                off_cold.status,
                off_warm.status,
            );
            assert_fragment_verdict_identity(
                case,
                lazy,
                "off-cold-vs-on-cold",
                off_cold.status,
                on_cold.status,
            );
            if let Some(expansions_saved) = assert_fragment_warm_contract(
                &case.cohort,
                &case.id,
                case.caps.node_cap,
                case.caps.semantic_horizon,
                lazy,
                "off-warm-vs-on-warm",
                off_warm.status,
                on_warm.status,
                on_warm.strict_verified_hard,
                off_warm.stats.nodes,
                on_warm.stats.nodes,
            ) {
                improvements.record(expansions_saved);
            }
            assert_eq!(
                off_cold.store,
                SharedFragmentStoreSnapshot::default(),
                "flag-off fragment store changed for {}",
                case.id,
            );
            assert!(on_cold.store.enabled && on_warm.store.enabled);

            off_cold_total.push(&off_cold);
            off_warm_total.push(&off_warm);
            on_cold_total.push(&on_cold);
            on_warm_total.push(&on_warm);
        }

        off_cold_total.print(lazy, false, "cold");
        off_warm_total.print(lazy, false, "warm");
        on_cold_total.print(lazy, true, "cold");
        on_warm_total.print(lazy, true, "warm");
        println!(
            "SF_DELTA lazy={} roots={} cold_nodes_delta_pct={:.3} cold_expansions_delta_pct={:.3} cold_ms_delta_pct={:.3} warm_nodes_delta_pct={:.3} warm_expansions_delta_pct={:.3} warm_ms_delta_pct={:.3} on_warm_vs_on_cold_nodes_delta_pct={:.3} on_warm_vs_on_cold_ms_delta_pct={:.3} warm_fragment_hit_rate={:.6}",
            on_off(lazy),
            cases.len(),
            signed_delta_percent(on_cold_total.nodes, off_cold_total.nodes),
            signed_delta_percent(on_cold_total.nodes, off_cold_total.nodes),
            signed_delta_percent_f64(on_cold_total.elapsed_ms, off_cold_total.elapsed_ms),
            signed_delta_percent(on_warm_total.nodes, off_warm_total.nodes),
            signed_delta_percent(on_warm_total.nodes, off_warm_total.nodes),
            signed_delta_percent_f64(on_warm_total.elapsed_ms, off_warm_total.elapsed_ms),
            signed_delta_percent(on_warm_total.nodes, on_cold_total.nodes),
            signed_delta_percent_f64(on_warm_total.elapsed_ms, on_cold_total.elapsed_ms),
            ratio(on_warm_total.fragment_hits, on_warm_total.fragment_lookups),
        );
        improvements.print("soundness_warm", lazy);
        let mutation = if case_filter.is_none() {
            shared_fragment_mutation_control(&corpus, tt_bytes_cap, lazy);
            "PASS"
        } else {
            "SKIP_FILTERED"
        };
        println!(
            "SF_CAMPAIGN_DONE lazy={} roots={} cold_verdict_identity=PASS flag_off_baseline_identity=PASS warm_monotone_contract=PASS monotone_improvements={} improvement_expansions_saved={} certificates=PASS forcing_no=PASS mutation={mutation}",
            on_off(lazy),
            cases.len(),
            improvements.count,
            improvements.expansions_saved,
        );
    }
}

struct ReducedFragmentRun {
    final_status: ProofStatus,
    closure_cap: Option<u64>,
    rungs: Vec<ReducedFragmentRung>,
    aggregate: SharedFragmentAggregate,
    final_store: SharedFragmentStoreSnapshot,
}

#[derive(Clone, Copy)]
struct ReducedFragmentRung {
    cap: u64,
    status: ProofStatus,
    strict_verified_hard: bool,
    expansions: u64,
}

fn reduced_fragment_case(
    position: &CorpusPosition,
    tt_bytes_cap: usize,
    node_cap: u64,
) -> SharedFragmentCase {
    SharedFragmentCase {
        cohort: "reduced_tt".to_owned(),
        id: position.id.clone(),
        state: position.state.clone(),
        caps: SolveCaps {
            node_cap,
            tt_bytes_cap,
            semantic_horizon: u32::MAX,
        },
        width: WidthOptions::vcf_pair_complete(),
        zone: ZoneSearchCaps::default(),
        forbid_win: !position.expect_win,
    }
}

fn run_reduced_fragment_ladder(
    position: &CorpusPosition,
    tt_bytes_cap: usize,
    ladder: &[u64],
    fragments: bool,
    lazy: bool,
) -> ReducedFragmentRun {
    let width = WidthOptions::vcf_pair_complete();
    let zone = ZoneSearchCaps::default();
    let mut solver = configured_fragment_solver(fragments, width, zone);
    let mut aggregate = SharedFragmentAggregate::default();
    let mut rungs = Vec::new();
    let mut closure_cap = None;
    let mut final_status = ProofStatus::Unknown;
    let mut final_store = solver.shared_fragment_store_snapshot();

    for (index, &node_cap) in ladder.iter().enumerate() {
        let case = reduced_fragment_case(position, tt_bytes_cap, node_cap);
        let phase = if index == 0 {
            "cold"
        } else {
            "progressive_warm"
        };
        let run = solve_shared_fragment_once(&mut solver, &case, fragments, phase, lazy);
        assert!(
            !(position.expect_win && run.status == ProofStatus::Loss),
            "SF_STOP reduced-TT WIN row returned LOSS: id={} cap={node_cap} fragments={} lazy={}",
            position.id,
            on_off(fragments),
            on_off(lazy),
        );
        println!(
            "SF_REDUCED_RUNG id={} tt_bytes_cap={} cap={node_cap} lazy={} fragments={} phase={phase} status={} nodes={} expansions={} ms={:.3} lookups={} hits={} hit_rate={:.6} imports={} store_entries={} store_bytes={} store_peak_bytes={} admissions={} replacements={} refusals={}",
            position.id,
            tt_bytes_cap,
            on_off(lazy),
            on_off(fragments),
            status_name(run.status),
            run.stats.nodes,
            run.stats.nodes,
            run.elapsed_ms,
            run.stats.fragment_lookups,
            run.stats.fragment_hits,
            ratio(run.stats.fragment_hits, run.stats.fragment_lookups),
            run.stats.fragment_imports,
            run.store.entries,
            run.store.bytes,
            run.store.peak_bytes,
            run.store.admissions,
            run.store.replacements,
            run.store.refusals,
        );
        final_status = run.status;
        final_store = run.store;
        rungs.push(ReducedFragmentRung {
            cap: node_cap,
            status: run.status,
            strict_verified_hard: run.strict_verified_hard,
            expansions: run.stats.nodes,
        });
        aggregate.push(&run);
        if run.status == ProofStatus::Win {
            closure_cap = Some(node_cap);
            break;
        }
        if run.status == ProofStatus::Loss {
            break;
        }
    }

    ReducedFragmentRun {
        final_status,
        closure_cap,
        rungs,
        aggregate,
        final_store,
    }
}

fn run_reduced_flag_off_baseline(
    position: &CorpusPosition,
    tt_bytes_cap: usize,
    node_cap: u64,
    lazy: bool,
) -> SharedFragmentRun {
    let case = reduced_fragment_case(position, tt_bytes_cap, node_cap);
    let mut solver = configured_fragment_solver(false, case.width, case.zone);
    let run = solve_shared_fragment_once(&mut solver, &case, false, "fresh_cold_baseline", lazy);
    assert_eq!(
        run.store,
        SharedFragmentStoreSnapshot::default(),
        "flag-off fragment store changed for reduced baseline {} cap={node_cap}",
        position.id,
    );
    println!(
        "SF_REDUCED_FLAG_OFF_BASELINE id={} tt_bytes_cap={} cap={node_cap} lazy={} status={} expansions={} ms={:.3} result=PASS",
        position.id,
        tt_bytes_cap,
        on_off(lazy),
        status_name(run.status),
        run.stats.nodes,
        run.elapsed_ms,
    );
    run
}

#[test]
#[ignore = "G2R9 0l reduced-TT saturation campaign; release-only and serialized"]
fn shared_fragment_reduced_tt_campaign() {
    let ladder = shared_fragment_reduced_ladder();
    let selected_ids = std::env::var("TSS_SHARED_FRAGMENT_HEAVY_IDS")
        .unwrap_or_else(|_| "0l4291i_live".to_owned())
        .split(',')
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(!selected_ids.is_empty(), "heavy-row selection is empty");
    let corpus = forcing_corpus();
    let selected = selected_ids
        .iter()
        .map(|id| {
            corpus
                .iter()
                .find(|position| &position.id == id)
                .unwrap_or_else(|| panic!("unknown heavy forcing row {id}"))
        })
        .collect::<Vec<_>>();
    let budgets = shared_fragment_reduced_budgets();
    let lazy_guard = LazyFrontierEnvGuard::new();
    let mut total_improvements = SharedFragmentImprovementCensus::default();
    let mut warm_comparisons = 0u64;
    let mut flag_off_baseline_comparisons = 0u64;

    for lazy in shared_fragment_lazy_modes(false) {
        lazy_guard.set(lazy);
        for &tt_bytes_cap in &budgets {
            for position in &selected {
                assert!(
                    position.expect_win,
                    "reduced-TT closure campaign expects a WIN row: {}",
                    position.id
                );
                let off = run_reduced_fragment_ladder(position, tt_bytes_cap, &ladder, false, lazy);
                let on = run_reduced_fragment_ladder(position, tt_bytes_cap, &ladder, true, lazy);
                let mut improvements = SharedFragmentImprovementCensus::default();
                for off_rung in &off.rungs {
                    if let Some(on_rung) = on.rungs.iter().find(|rung| rung.cap == off_rung.cap) {
                        if off_rung.cap == ladder[0] {
                            assert_eq!(
                                off_rung.status,
                                on_rung.status,
                                "SF_STOP reduced-TT cold verdict flip: id={} cap={} tt_bytes_cap={tt_bytes_cap} lazy={} off={} on={}",
                                position.id,
                                off_rung.cap,
                                on_off(lazy),
                                status_name(off_rung.status),
                                status_name(on_rung.status),
                            );
                        } else {
                            let baseline = run_reduced_flag_off_baseline(
                                position,
                                tt_bytes_cap,
                                off_rung.cap,
                                lazy,
                            );
                            assert_eq!(
                                baseline.status,
                                off_rung.status,
                                "SF_STOP reduced-TT flag-off baseline flip: id={} cap={} tt_bytes_cap={tt_bytes_cap} lazy={} cold={} progressive_warm={}",
                                position.id,
                                off_rung.cap,
                                on_off(lazy),
                                status_name(baseline.status),
                                status_name(off_rung.status),
                            );
                            flag_off_baseline_comparisons =
                                flag_off_baseline_comparisons.saturating_add(1);
                            warm_comparisons = warm_comparisons.saturating_add(1);
                            if let Some(expansions_saved) = assert_fragment_warm_contract(
                                "reduced_tt",
                                &position.id,
                                off_rung.cap,
                                u32::MAX,
                                lazy,
                                "off-progressive-warm-vs-on-progressive-warm",
                                off_rung.status,
                                on_rung.status,
                                on_rung.strict_verified_hard,
                                off_rung.expansions,
                                on_rung.expansions,
                            ) {
                                improvements.record(expansions_saved);
                                total_improvements.record(expansions_saved);
                            }
                        }
                    }
                }
                let newly_closed = on.closure_cap.is_some() && off.closure_cap.is_none();
                let closed_earlier = match (off.closure_cap, on.closure_cap) {
                    (Some(off_cap), Some(on_cap)) => on_cap < off_cap,
                    (None, Some(_)) => true,
                    _ => false,
                };
                let off_final_cap = off.rungs.last().map_or(0, |rung| rung.cap);
                let on_final_cap = on.rungs.last().map_or(0, |rung| rung.cap);
                println!(
                    "SF_REDUCED_SUMMARY id={} tt_bytes_cap={} lazy={} configured_max_cap={} off_final_cap={} on_final_cap={} off_rungs={} on_rungs={} paired_rungs={} off_status={} on_status={} off_closure_cap={} on_closure_cap={} newly_closed={} closed_earlier={} monotone_improvements={} improvement_expansions_saved={} off_ladder_expansions={} on_ladder_expansions={} ladder_work_delta_pct={:.3} off_ladder_ms={:.3} on_ladder_ms={:.3} ladder_ms_delta_pct={:.3} on_lookups={} on_hits={} on_hit_rate={:.6} on_imports={} on_store_entries={} on_store_bytes={} on_store_peak_bytes={} on_admissions={} on_replacements={} on_refusals={}",
                    position.id,
                    tt_bytes_cap,
                    on_off(lazy),
                    ladder.last().copied().unwrap_or_default(),
                    off_final_cap,
                    on_final_cap,
                    off.rungs.len(),
                    on.rungs.len(),
                    off.rungs
                        .iter()
                        .filter(|off_rung| on.rungs.iter().any(|rung| rung.cap == off_rung.cap))
                        .count(),
                    status_name(off.final_status),
                    status_name(on.final_status),
                    off.closure_cap.unwrap_or_default(),
                    on.closure_cap.unwrap_or_default(),
                    newly_closed,
                    closed_earlier,
                    improvements.count,
                    improvements.expansions_saved,
                    off.aggregate.nodes,
                    on.aggregate.nodes,
                    signed_delta_percent(on.aggregate.nodes, off.aggregate.nodes),
                    off.aggregate.elapsed_ms,
                    on.aggregate.elapsed_ms,
                    signed_delta_percent_f64(
                        on.aggregate.elapsed_ms,
                        off.aggregate.elapsed_ms
                    ),
                    on.aggregate.fragment_lookups,
                    on.aggregate.fragment_hits,
                    ratio(
                        on.aggregate.fragment_hits,
                        on.aggregate.fragment_lookups
                    ),
                    on.aggregate.fragment_imports,
                    on.final_store.entries,
                    on.final_store.bytes,
                    on.final_store.peak_bytes,
                    on.final_store.admissions,
                    on.final_store.replacements,
                    on.final_store.refusals,
                );
                println!(
                    "SF_REDUCED_IMPROVEMENT_CENSUS id={} tt_bytes_cap={} lazy={} count={} expansions_saved={}",
                    position.id,
                    tt_bytes_cap,
                    on_off(lazy),
                    improvements.count,
                    improvements.expansions_saved,
                );
            }
        }
    }
    let warm_contract = if warm_comparisons == 0 {
        "NOT_EXERCISED"
    } else {
        "PASS"
    };
    let flag_off_baseline = if flag_off_baseline_comparisons == 0 {
        "NOT_EXERCISED"
    } else {
        "PASS"
    };
    println!(
        "SF_REDUCED_DONE rows={} budgets={} max_cap={} cold_verdict_identity=PASS flag_off_baseline_identity={flag_off_baseline} flag_off_baseline_comparisons={} warm_monotone_contract={warm_contract} warm_comparisons={} monotone_improvements={} improvement_expansions_saved={} certificates=PASS forcing_no=NOT_APPLICABLE_EXPECT_WIN_ONLY result=PASS",
        selected.len(),
        budgets.len(),
        ladder.last().copied().unwrap_or_default(),
        flag_off_baseline_comparisons,
        warm_comparisons,
        total_improvements.count,
        total_improvements.expansions_saved,
    );
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

#[test]
#[ignore = "R-LF1 equivalence campaign; release-only and serialized"]
fn lazy_frontier_equivalence_campaign() {
    let tt_bytes_cap = std::env::var("TSS_LAZY_FRONTIER_TT_BYTES")
        .ok()
        .map(|value| value.parse().expect("numeric lazy-frontier TT bytes"))
        .unwrap_or(DEFAULT_LAZY_EQ_TT_BYTES);
    std::env::set_var("TSS_TURN_QUOTIENT_TELEMETRY", "1");
    std::env::remove_var("TSS_LAZY_FRONTIER");

    let corpus = forcing_corpus();
    for cap in [10_000u64, 100_000] {
        let cohort = format!("forcing_{cap}");
        let mut off = LazyAggregate::default();
        let mut on = LazyAggregate::default();
        for position in &corpus {
            compare_lazy_row(
                &cohort,
                &position.id,
                &position.state,
                SolveCaps {
                    node_cap: cap,
                    tt_bytes_cap,
                    semantic_horizon: u32::MAX,
                },
                WidthOptions::vcf_pair_complete(),
                ZoneSearchCaps::default(),
                &mut off,
                &mut on,
            );
        }
        off.print(&cohort, "off");
        on.print(&cohort, "on");
    }

    let compact = replay(DOUBLE_FORK_COMPACT);
    let mut compact_off = LazyAggregate::default();
    let mut compact_on = LazyAggregate::default();
    compare_lazy_row(
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
        &mut compact_off,
        &mut compact_on,
    );
    compact_off.print("double_fork_compact", "off");
    compact_on.print("double_fork_compact", "on");

    let games = human_games();
    let roots = human_roots(&games, 20);
    assert_eq!(roots.len(), 20, "human sample must contain 20 roots");
    let mut human_off = LazyAggregate::default();
    let mut human_on = LazyAggregate::default();
    for (rank, root) in roots.iter().enumerate() {
        let state = replay(&games[root.game].moves[..root.prefix]);
        compare_lazy_row(
            "human_20_cap10000",
            &format!("human_{rank:03}_g{}_p{}", root.game, root.prefix),
            &state,
            SolveCaps {
                node_cap: 10_000,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            },
            WidthOptions::vcf_pair_complete(),
            ZoneSearchCaps::default(),
            &mut human_off,
            &mut human_on,
        );
    }
    human_off.print("human_20_cap10000", "off");
    human_on.print("human_20_cap10000", "on");

    std::env::remove_var("TSS_LAZY_FRONTIER");
    std::env::remove_var("TSS_TURN_QUOTIENT_TELEMETRY");
    println!("LF_EQ_DONE result=PASS node_identity=exact certificate_bytes=exact");
}
