//! NQ6 interior census-gate and PN-seed measurement harness.
//!
//! The workload and human-root sampler are derived from the sibling NQ4
//! `tss_turn_quotient_hunt.rs` at commit 2430fc47. Census evaluation is owned
//! by the test-only solver hook and copies the exact reviewed Contract-8.1
//! `WindowStore::entries()` recipe with provenance there.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, TurnPhase};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal, ZoneSearchCaps};
use crate::tss_solver::{
    begin_pn_init_telemetry, take_pn_init_telemetry_report, PnInitTelemetryMode,
    PnInitTelemetryNode, PnInitTelemetryOutcome, PnInitTelemetryReport, TssSolver, WidthOptions,
};
use crate::tss_verify::TssVerifier;

const DEFAULT_TT_BYTES: usize = 512 << 20;
const RELATIVE_HORIZON: u32 = 16;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SeedKind {
    LiveGe4,
    LiveGe3,
    DisjointTwoGap,
}

impl SeedKind {
    const ALL: [Self; 3] = [Self::LiveGe4, Self::LiveGe3, Self::DisjointTwoGap];

    fn name(self) -> &'static str {
        match self {
            Self::LiveGe4 => "live_ge4",
            Self::LiveGe3 => "live_ge3",
            Self::DisjointTwoGap => "disjoint_two_gap",
        }
    }

    fn value(self, node: &PnInitTelemetryNode) -> u32 {
        match self {
            Self::LiveGe4 => node.live_ge4,
            Self::LiveGe3 => node.live_ge3,
            Self::DisjointTwoGap => node.disjoint_two_gap,
        }
    }
}

#[derive(Default)]
struct Counterfactual {
    candidate_nodes: u64,
    incremental_saved: u64,
    proven_contradictions: u64,
}

#[derive(Default)]
struct PnReplayAggregate {
    solved_roots: u64,
    unique_nodes: u64,
    classified_nodes: u64,
    correlation_rows: Vec<(u32, u32, u32, bool)>,
    actual_replay_nodes: u64,
    seeded_replay_nodes: BTreeMap<SeedKind, u64>,
}

#[derive(Default)]
struct Aggregate {
    roots: u64,
    solver_nodes: u64,
    expansions: u64,
    wins: u64,
    losses: u64,
    unknowns: u64,
    eligible_gate_nodes: u64,
    gated_nodes: u64,
    subtree_saved: u64,
    soundness_checked_on_solved_roots: u64,
    census_costs: Vec<u64>,
    slack: BTreeMap<(u8, u32, u8, u8), u64>,
    counterfactual: BTreeMap<(u8, u32, u8), Counterfactual>,
    pn: PnReplayAggregate,
}

#[derive(Default)]
struct LiveAggregate {
    roots: u64,
    wins: u64,
    losses: u64,
    unknowns: u64,
    nodes: u64,
    expansions: u64,
    tt_entries: u64,
    tt_hits: u64,
    wall_nanos: u128,
    gate_evaluations: u64,
    gate_dismissals: u64,
    gate_nanos: u64,
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
    let path = std::env::var("TSS_PN_INIT_HUMAN_CORPUS").unwrap_or_else(|_| {
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

fn inherited_mask(
    report: &PnInitTelemetryReport,
    predicate: impl Fn(&PnInitTelemetryNode) -> bool,
) -> Vec<bool> {
    let mut mask = vec![false; report.nodes.len()];
    for node in &report.nodes {
        let inherited = node
            .parent_serial
            .and_then(|parent| mask.get(parent as usize).copied())
            .unwrap_or(false);
        mask[node.serial as usize] = inherited || predicate(node);
    }
    mask
}

fn phase_name(code: u8) -> &'static str {
    match code {
        1 => "FS",
        2 => "SS",
        _ => "OPEN",
    }
}

fn analyze_trace(report: &PnInitTelemetryReport, solved: bool, aggregate: &mut Aggregate) {
    for (index, node) in report.nodes.iter().enumerate() {
        assert_eq!(node.serial as usize, index, "telemetry serial drift");
        if let Some(parent) = node.parent_serial {
            assert!(parent < node.serial, "telemetry parent must precede child");
        }
    }

    let actual_saved = inherited_mask(report, |node| node.gate);
    aggregate.expansions += report.nodes.len() as u64;
    aggregate.subtree_saved += actual_saved.iter().filter(|saved| **saved).count() as u64;

    for node in &report.nodes {
        let eligible = node.win_arm
            && node.h_rem.is_some_and(|h| h <= 8)
            && node.census.is_some()
            && node.coordinate_safe;
        aggregate.eligible_gate_nodes += u64::from(eligible);
        aggregate.gated_nodes += u64::from(node.gate);
        if eligible {
            aggregate.census_costs.push(node.census_scan_nanos);
        }
        if solved && node.gate {
            aggregate.soundness_checked_on_solved_roots += 1;
        }
        if node.gate && node.outcome == PnInitTelemetryOutcome::Proven {
            panic!(
                "SOUNDNESS FINDING: gated node contained a WIN certificate: {}",
                node.frozen_state
                    .as_deref()
                    .unwrap_or("missing frozen state")
            );
        }
        if node.win_arm {
            if let (Some(h), Some(c), Some(lb)) = (node.h_rem, node.census, node.lb_plies) {
                if h <= RELATIVE_HORIZON {
                    *aggregate
                        .slack
                        .entry((node.phase_code, h, c, lb))
                        .or_default() += 1;
                }
            }
        }
    }

    for phase in [1u8, 2] {
        for horizon in 9..=16u32 {
            for census_max in 0..=4u8 {
                let hypothetical = inherited_mask(report, |node| {
                    node.win_arm
                        && node.phase_code == phase
                        && node.coordinate_safe
                        && node.h_rem.is_some_and(|h| h <= horizon)
                        && node.census.is_some_and(|c| c <= census_max)
                });
                let row = aggregate
                    .counterfactual
                    .entry((phase, horizon, census_max))
                    .or_default();
                row.candidate_nodes += report
                    .nodes
                    .iter()
                    .filter(|node| {
                        node.win_arm
                            && node.phase_code == phase
                            && node.coordinate_safe
                            && node.h_rem.is_some_and(|h| h <= horizon)
                            && node.census.is_some_and(|c| c <= census_max)
                    })
                    .count() as u64;
                row.incremental_saved += hypothetical
                    .iter()
                    .zip(&actual_saved)
                    .filter(|(hypothetical, actual)| **hypothetical && !**actual)
                    .count() as u64;
                row.proven_contradictions += report
                    .nodes
                    .iter()
                    .filter(|node| {
                        node.win_arm
                            && node.phase_code == phase
                            && node.coordinate_safe
                            && node.h_rem.is_some_and(|h| h <= horizon)
                            && node.census.is_some_and(|c| c <= census_max)
                            && node.outcome == PnInitTelemetryOutcome::Proven
                    })
                    .count() as u64;
            }
        }
    }
}

fn rank_values(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut begin = 0usize;
    while begin < order.len() {
        let mut end = begin + 1;
        while end < order.len() && values[order[end]] == values[order[begin]] {
            end += 1;
        }
        let rank = (begin + end - 1) as f64 / 2.0;
        for &index in &order[begin..end] {
            ranks[index] = rank;
        }
        begin = end;
    }
    ranks
}

fn spearman(features: &[u32], proven: &[bool]) -> f64 {
    if features.len() < 2 || features.len() != proven.len() {
        return f64::NAN;
    }
    let x = rank_values(
        &features
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
    );
    let y = rank_values(
        &proven
            .iter()
            .map(|value| if *value { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let covariance = x
        .iter()
        .zip(&y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance_x = x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let variance_y = y.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    covariance / (variance_x * variance_y).sqrt()
}

fn shadow_replay_cost(
    id: u64,
    seed: Option<SeedKind>,
    nodes: &HashMap<u64, &PnInitTelemetryNode>,
    children: &HashMap<u64, Vec<u64>>,
    memo: &mut HashMap<u64, u64>,
) -> u64 {
    if let Some(cost) = memo.get(&id) {
        return *cost;
    }
    let Some(node) = nodes.get(&id).copied() else {
        return 0;
    };
    let mut child_ids = children.get(&id).cloned().unwrap_or_default();
    child_ids.sort_by_key(|child| {
        let child_node = nodes[child];
        (
            seed.map(|kind| Reverse(kind.value(child_node))),
            child_node.serial,
        )
    });
    let cost = match (node.win_arm, node.outcome) {
        (true, PnInitTelemetryOutcome::Proven) => {
            let mut cost = 1u64;
            for child in child_ids {
                cost = cost.saturating_add(shadow_replay_cost(child, seed, nodes, children, memo));
                if nodes[&child].outcome == PnInitTelemetryOutcome::Proven {
                    break;
                }
            }
            cost
        }
        (false, PnInitTelemetryOutcome::Refuted) => {
            let mut cost = 1u64;
            for child in child_ids {
                cost = cost.saturating_add(shadow_replay_cost(child, seed, nodes, children, memo));
                if nodes[&child].outcome == PnInitTelemetryOutcome::Refuted {
                    break;
                }
            }
            cost
        }
        _ => 1u64.saturating_add(
            child_ids
                .into_iter()
                .map(|child| shadow_replay_cost(child, seed, nodes, children, memo))
                .sum::<u64>(),
        ),
    };
    memo.insert(id, cost);
    cost
}

fn analyze_pn_replay(
    report: &PnInitTelemetryReport,
    solved: bool,
    aggregate: &mut PnReplayAggregate,
) {
    let wide = report
        .nodes
        .iter()
        .filter(|node| node.mode == PnInitTelemetryMode::WidePn)
        .collect::<Vec<_>>();
    if wide.is_empty() {
        return;
    }
    let mut unique = HashMap::<u64, &PnInitTelemetryNode>::new();
    for node in &wide {
        unique.entry(node.engine_node).or_insert(node);
    }
    aggregate.unique_nodes += unique.len() as u64;
    for node in unique.values() {
        match node.outcome {
            PnInitTelemetryOutcome::Proven => {
                aggregate.classified_nodes += 1;
                aggregate.correlation_rows.push((
                    node.live_ge4,
                    node.live_ge3,
                    node.disjoint_two_gap,
                    true,
                ));
            }
            PnInitTelemetryOutcome::Refuted => {
                aggregate.classified_nodes += 1;
                aggregate.correlation_rows.push((
                    node.live_ge4,
                    node.live_ge3,
                    node.disjoint_two_gap,
                    false,
                ));
            }
            PnInitTelemetryOutcome::Unknown => {}
        }
    }
    if !solved || !unique.contains_key(&0) {
        return;
    }

    let mut parent_of = HashMap::<u64, u64>::new();
    for node in &wide {
        let Some(parent_serial) = node.parent_serial else {
            continue;
        };
        let Some(parent) = report.nodes.get(parent_serial as usize) else {
            continue;
        };
        if parent.mode == PnInitTelemetryMode::WidePn && parent.engine_node != node.engine_node {
            parent_of
                .entry(node.engine_node)
                .or_insert(parent.engine_node);
        }
    }
    let mut children = HashMap::<u64, Vec<u64>>::new();
    for (child, parent) in parent_of {
        children.entry(parent).or_default().push(child);
    }
    for child_ids in children.values_mut() {
        child_ids.sort_unstable_by_key(|id| unique[id].serial);
        child_ids.dedup();
    }

    aggregate.solved_roots += 1;
    aggregate.actual_replay_nodes +=
        shadow_replay_cost(0, None, &unique, &children, &mut HashMap::new());
    for kind in SeedKind::ALL {
        *aggregate.seeded_replay_nodes.entry(kind).or_default() +=
            shadow_replay_cost(0, Some(kind), &unique, &children, &mut HashMap::new());
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
) -> ProofStatus {
    begin_pn_init_telemetry();
    let mut solver = TssSolver::default();
    solver.set_width_options(width);
    solver.set_zone_options(zone);
    let started = Instant::now();
    let result = solver.solve_goal(state, &caps, SolveGoal::Win);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    let report = take_pn_init_telemetry_report().expect("telemetry report");
    if let Some(cert) = result.cert.as_ref() {
        assert!(
            TssVerifier.verify(state, cert, result.status),
            "certificate verification failed for {id}"
        );
    }
    assert_ne!(
        result.status,
        ProofStatus::Loss,
        "WIN-only solve returned LOSS"
    );
    aggregate.roots += 1;
    aggregate.solver_nodes += result.stats.nodes;
    match result.status {
        ProofStatus::Win => aggregate.wins += 1,
        ProofStatus::Loss => aggregate.losses += 1,
        ProofStatus::Unknown => aggregate.unknowns += 1,
    }
    analyze_trace(&report, result.status == ProofStatus::Win, aggregate);
    analyze_pn_replay(
        &report,
        result.status == ProofStatus::Win,
        &mut aggregate.pn,
    );
    println!(
        "PNI_ROW group={group} id={id} cap={} horizon={} status={} solver_nodes={} expansions={} gated={} subtree_saved={} ms={elapsed_ms:.3}",
        caps.node_cap,
        caps.semantic_horizon,
        status_name(result.status),
        result.stats.nodes,
        report.nodes.len(),
        report.nodes.iter().filter(|node| node.gate).count(),
        inherited_mask(&report, |node| node.gate)
            .iter()
            .filter(|saved| **saved)
            .count(),
    );
    result.status
}

fn solve_live_row(
    id: &str,
    group: &str,
    state: &HexoState,
    caps: SolveCaps,
    width: WidthOptions,
    zone: ZoneSearchCaps,
    aggregate: &mut LiveAggregate,
) -> ProofStatus {
    let mut solver = TssSolver::default();
    solver.set_width_options(width);
    solver.set_zone_options(zone);
    let started = Instant::now();
    let result = solver.solve_goal(state, &caps, SolveGoal::Win);
    let elapsed = started.elapsed();
    if let Some(cert) = result.cert.as_ref() {
        assert!(
            TssVerifier.verify(state, cert, result.status),
            "live certificate verification failed for {id}"
        );
    }
    assert_ne!(
        result.status,
        ProofStatus::Loss,
        "live WIN-only solve returned LOSS"
    );
    aggregate.roots = aggregate.roots.saturating_add(1);
    aggregate.nodes = aggregate.nodes.saturating_add(result.stats.nodes);
    aggregate.expansions = aggregate.expansions.saturating_add(result.stats.expansions);
    aggregate.tt_entries = aggregate.tt_entries.saturating_add(result.stats.tt_entries);
    aggregate.tt_hits = aggregate.tt_hits.saturating_add(result.stats.tt_hits);
    aggregate.wall_nanos = aggregate.wall_nanos.saturating_add(elapsed.as_nanos());
    aggregate.gate_evaluations = aggregate
        .gate_evaluations
        .saturating_add(result.stats.interior_gate_evaluations);
    aggregate.gate_dismissals = aggregate
        .gate_dismissals
        .saturating_add(result.stats.interior_gate_dismissals);
    aggregate.gate_nanos = aggregate
        .gate_nanos
        .saturating_add(result.stats.interior_gate_nanos);
    match result.status {
        ProofStatus::Win => aggregate.wins = aggregate.wins.saturating_add(1),
        ProofStatus::Loss => aggregate.losses = aggregate.losses.saturating_add(1),
        ProofStatus::Unknown => aggregate.unknowns = aggregate.unknowns.saturating_add(1),
    }
    println!(
        "IG_ROW group={group} id={id} cap={} horizon={} status={} nodes={} expansions={} tt_entries={} tt_hits={} gate_evals={} gate_dismissals={} gate_us={:.3} ms={:.3}",
        caps.node_cap,
        caps.semantic_horizon,
        status_name(result.status),
        result.stats.nodes,
        result.stats.expansions,
        result.stats.tt_entries,
        result.stats.tt_hits,
        result.stats.interior_gate_evaluations,
        result.stats.interior_gate_dismissals,
        result.stats.interior_gate_nanos as f64 / 1_000.0,
        elapsed.as_secs_f64() * 1e3,
    );
    result.status
}

fn print_live_aggregate(group: &str, aggregate: &LiveAggregate) {
    let mode = if std::env::var_os("TSS_INTERIOR_CENSUS_GATE").is_some_and(|value| value == "1") {
        "on"
    } else {
        "off"
    };
    println!(
        "IG_SUMMARY mode={mode} group={group} roots={} verdicts={}/{}/{} nodes={} expansions={} tt_entries={} tt_hits={} gate_evals={} gate_dismissals={} gate_ms={:.3} wall_ms={:.3}",
        aggregate.roots,
        aggregate.wins,
        aggregate.losses,
        aggregate.unknowns,
        aggregate.nodes,
        aggregate.expansions,
        aggregate.tt_entries,
        aggregate.tt_hits,
        aggregate.gate_evaluations,
        aggregate.gate_dismissals,
        aggregate.gate_nanos as f64 / 1e6,
        aggregate.wall_nanos as f64 / 1e6,
    );
}

fn percentile(values: &mut [u64], numerator: usize, denominator: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) * numerator) / denominator;
    values[index]
}

fn print_aggregate(group: &str, aggregate: &mut Aggregate) {
    let gated_fraction = if aggregate.eligible_gate_nodes == 0 {
        0.0
    } else {
        aggregate.gated_nodes as f64 / aggregate.eligible_gate_nodes as f64
    };
    let subtree_fraction = if aggregate.expansions == 0 {
        0.0
    } else {
        aggregate.subtree_saved as f64 / aggregate.expansions as f64
    };
    let mean_ns = if aggregate.census_costs.is_empty() {
        0.0
    } else {
        aggregate.census_costs.iter().sum::<u64>() as f64 / aggregate.census_costs.len() as f64
    };
    let median_ns = percentile(&mut aggregate.census_costs.clone(), 1, 2);
    let p95_ns = percentile(&mut aggregate.census_costs.clone(), 95, 100);
    println!(
        "PNI_SUMMARY group={group} roots={} verdicts={}/{}/{} solver_nodes={} expansions={} eligible={} gated={} gated_fraction={gated_fraction:.6} subtree_saved={} subtree_fraction={subtree_fraction:.6} census_evals={} census_mean_us={:.3} census_median_us={:.3} census_p95_us={:.3} soundness_checked={} findings=0",
        aggregate.roots,
        aggregate.wins,
        aggregate.losses,
        aggregate.unknowns,
        aggregate.solver_nodes,
        aggregate.expansions,
        aggregate.eligible_gate_nodes,
        aggregate.gated_nodes,
        aggregate.subtree_saved,
        aggregate.census_costs.len(),
        mean_ns / 1_000.0,
        median_ns as f64 / 1_000.0,
        p95_ns as f64 / 1_000.0,
        aggregate.soundness_checked_on_solved_roots,
    );

    for phase in [1u8, 2] {
        for horizon in 0..=16u32 {
            let mut total = 0u64;
            let mut gt = 0u64;
            let mut eq = 0u64;
            let mut hist = [0u64; 6];
            for (&(row_phase, row_h, census, lb), &count) in &aggregate.slack {
                if row_phase != phase || row_h != horizon {
                    continue;
                }
                total += count;
                hist[census as usize] += count;
                if u32::from(lb) > horizon {
                    gt += count;
                } else if u32::from(lb) == horizon {
                    eq += count;
                }
            }
            if total > 0 {
                println!(
                    "PNI_SLACK group={group} phase={} h={horizon} nodes={total} lb_gt={gt} lb_eq={eq} lb_lt={} census_hist={hist:?}",
                    phase_name(phase),
                    total - gt - eq,
                );
            }
        }
    }

    for (&(phase, horizon, census_max), row) in &aggregate.counterfactual {
        println!(
            "PNI_CF group={group} phase={} h={horizon} cmax={census_max} candidate_nodes={} incremental_subtree_saved={} proven_contradictions={}",
            phase_name(phase),
            row.candidate_nodes,
            row.incremental_saved,
            row.proven_contradictions,
        );
    }

    let proven = aggregate
        .pn
        .correlation_rows
        .iter()
        .map(|row| row.3)
        .collect::<Vec<_>>();
    for (index, kind) in SeedKind::ALL.into_iter().enumerate() {
        let features = aggregate
            .pn
            .correlation_rows
            .iter()
            .map(|row| match index {
                0 => row.0,
                1 => row.1,
                _ => row.2,
            })
            .collect::<Vec<_>>();
        println!(
            "PNI_SEED group={group} seed={} unique_nodes={} classified={} spearman_proven={:.6} solved_roots={} actual_replay_nodes={} seeded_replay_nodes={}",
            kind.name(),
            aggregate.pn.unique_nodes,
            aggregate.pn.classified_nodes,
            spearman(&features, &proven),
            aggregate.pn.solved_roots,
            aggregate.pn.actual_replay_nodes,
            aggregate
                .pn
                .seeded_replay_nodes
                .get(&kind)
                .copied()
                .unwrap_or(0),
        );
    }
}

#[test]
#[ignore = "NQ6 measurement campaign; release-only, serialized, <=10 minutes"]
fn pn_init_campaign() {
    let tt_bytes_cap = std::env::var("TSS_PN_INIT_TT_BYTES")
        .ok()
        .map(|value| value.parse().expect("numeric TT bytes"))
        .unwrap_or(DEFAULT_TT_BYTES);
    let human_n = std::env::var("TSS_PN_INIT_HUMAN_N")
        .ok()
        .map(|value| value.parse().expect("numeric human sample"))
        .unwrap_or(100usize);
    let corpus = forcing_corpus();
    let identity_caps = SolveCaps {
        node_cap: 10_000,
        tt_bytes_cap,
        semantic_horizon: corpus[0].state.placements_made() + RELATIVE_HORIZON,
    };

    let mut off_solver = TssSolver::default();
    off_solver.set_width_options(WidthOptions::vcf_pair_complete());
    let off = off_solver.solve_goal(&corpus[0].state, &identity_caps, SolveGoal::Win);
    begin_pn_init_telemetry();
    let mut on_solver = TssSolver::default();
    on_solver.set_width_options(WidthOptions::vcf_pair_complete());
    let on = on_solver.solve_goal(&corpus[0].state, &identity_caps, SolveGoal::Win);
    let identity_report = take_pn_init_telemetry_report().expect("identity telemetry");
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
        "PNI_IDENTITY id={} status={} nodes={} tt_hits={} expansions={} result=PASS",
        corpus[0].id,
        status_name(on.status),
        on.stats.nodes,
        on.stats.tt_hits,
        identity_report.nodes.len(),
    );

    for cap in [10_000u64, 100_000] {
        let group = format!("forcing_{cap}");
        let mut aggregate = Aggregate::default();
        for position in &corpus {
            let status = solve_row(
                &position.id,
                &group,
                &position.state,
                SolveCaps {
                    node_cap: cap,
                    tt_bytes_cap,
                    semantic_horizon: position.state.placements_made() + RELATIVE_HORIZON,
                },
                WidthOptions::vcf_pair_complete(),
                ZoneSearchCaps::default(),
                &mut aggregate,
            );
            assert!(
                position.expect_win || status != ProofStatus::Win,
                "forcing NO row {} became WIN",
                position.id
            );
        }
        print_aggregate(&group, &mut aggregate);
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
    );
    print_aggregate("double_fork_compact", &mut compact_aggregate);

    let games = human_games();
    let roots = human_roots(&games, human_n);
    assert_eq!(roots.len(), human_n, "human sample size drift");
    let mut human_aggregate = Aggregate::default();
    for (rank, root) in roots.iter().enumerate() {
        let state = replay(&games[root.game].moves[..root.prefix]);
        solve_row(
            &format!("human_{rank:03}_g{}_p{}", root.game, root.prefix),
            &format!("human_{human_n}_cap10000"),
            &state,
            SolveCaps {
                node_cap: 10_000,
                tt_bytes_cap,
                semantic_horizon: state.placements_made() + RELATIVE_HORIZON,
            },
            WidthOptions::vcf_pair_complete(),
            ZoneSearchCaps::default(),
            &mut human_aggregate,
        );
    }
    print_aggregate(&format!("human_{human_n}_cap10000"), &mut human_aggregate);
    println!("PNI_DONE result=PASS anomalies=0 soundness_findings=0");
}

#[test]
#[ignore = "R-IG1 live on/off campaign; release-only, serialized, <=10 minutes"]
fn interior_gate_live_campaign() {
    let tt_bytes_cap = std::env::var("TSS_PN_INIT_TT_BYTES")
        .ok()
        .map(|value| value.parse().expect("numeric TT bytes"))
        .unwrap_or(DEFAULT_TT_BYTES);
    let human_n = std::env::var("TSS_PN_INIT_HUMAN_N")
        .ok()
        .map(|value| value.parse().expect("numeric human sample"))
        .unwrap_or(100usize);
    let corpus = forcing_corpus();

    for cap in [10_000u64, 100_000] {
        let group = format!("forcing_{cap}");
        let mut aggregate = LiveAggregate::default();
        for position in &corpus {
            let status = solve_live_row(
                &position.id,
                &group,
                &position.state,
                SolveCaps {
                    node_cap: cap,
                    tt_bytes_cap,
                    semantic_horizon: position.state.placements_made() + RELATIVE_HORIZON,
                },
                WidthOptions::vcf_pair_complete(),
                ZoneSearchCaps::default(),
                &mut aggregate,
            );
            assert!(
                position.expect_win || status != ProofStatus::Win,
                "forcing NO row {} became WIN",
                position.id
            );
        }
        print_live_aggregate(&group, &aggregate);
    }

    let compact = replay(DOUBLE_FORK_COMPACT);
    let mut compact_aggregate = LiveAggregate::default();
    solve_live_row(
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
    );
    print_live_aggregate("double_fork_compact", &compact_aggregate);

    let games = human_games();
    let roots = human_roots(&games, human_n);
    assert_eq!(roots.len(), human_n, "human sample size drift");
    let mut human_aggregate = LiveAggregate::default();
    for (rank, root) in roots.iter().enumerate() {
        let state = replay(&games[root.game].moves[..root.prefix]);
        solve_live_row(
            &format!("human_{rank:03}_g{}_p{}", root.game, root.prefix),
            &format!("human_{human_n}_cap10000"),
            &state,
            SolveCaps {
                node_cap: 10_000,
                tt_bytes_cap,
                semantic_horizon: state.placements_made() + RELATIVE_HORIZON,
            },
            WidthOptions::vcf_pair_complete(),
            ZoneSearchCaps::default(),
            &mut human_aggregate,
        );
    }
    print_live_aggregate(&format!("human_{human_n}_cap10000"), &human_aggregate);
    println!("IG_DONE result=PASS certificates=VERIFIED forcing_anomalies=0");
}
