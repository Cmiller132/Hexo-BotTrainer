//! R-CREL-5 Phase-3 leaf-relevance cohort-construction audit.
//!
//! This ignored, cfg(test)-only harness freezes actual sibling frontier leaves
//! from the selected wide/gated solve trees. It deliberately stops at the
//! pre-registered zoned-coverage gate: economics arms are not run when fewer
//! than 20% of admitted source certificates contain a zoned Universal.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, TurnPhase};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal, ZoneSearchCaps};
use crate::tss_solver::{
    begin_pn_init_leaf_state_telemetry, take_pn_init_telemetry_report, TssSolver, WidthOptions,
};
use crate::tss_verify::{CertNode, RootBinding, TssCertificate, TssVerifier};

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const GAME_COUNT: usize = 50;
const WINDOW_STATES: usize = 6;
const MIN_START_PLY: usize = 8;
const NODE_CAP: u64 = 500;
const TT_BYTES: usize = 256 * 1024;
const HORIZONS: [u32; 2] = [8, 16];
const FLAGS: [&str; 4] = [
    "TSS_LAZY_FRONTIER",
    "TSS_INTERIOR_CENSUS_GATE",
    "TSS_SHARED_FRAGMENTS",
    "TSS_K_REPLY_CONSUME",
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

#[derive(Clone)]
struct SiblingPair {
    cluster: String,
    ancestor_id: u64,
    ancestor_hash: u64,
    source_hash: u64,
    reuse_hash: u64,
    source: HexoState,
    reuse: HexoState,
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
    fn selected_profile() -> Self {
        let old = FLAGS
            .iter()
            .map(|&name| (name, std::env::var_os(name)))
            .collect::<Vec<_>>();
        for name in FLAGS {
            std::env::remove_var(name);
        }
        std::env::set_var("TSS_LAZY_FRONTIER", "1");
        std::env::set_var("TSS_INTERIOR_CENSUS_GATE", "1");
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

fn parse_ints(slice: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in slice.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            out.push(current.parse().expect("i16 corpus token"));
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current.parse().expect("i16 corpus token"));
    }
    out
}

fn parse_line(line: &str) -> Option<Game> {
    let hash_key = "\"game_hash\":\"";
    let hash_tail = &line[line.find(hash_key)? + hash_key.len()..];
    let hash = hash_tail[..hash_tail.find('"')?].to_owned();

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
        "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl".to_owned()
    })
}

fn load_batches() -> (usize, Vec<Batch>) {
    let path = corpus_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read leaf corpus {path}: {error}"));
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
    (games.len(), batches)
}

fn make_solver() -> TssSolver {
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::vcf_pair_complete());
    solver.set_zone_options(ZoneSearchCaps::default());
    solver
}

fn caps(state: &HexoState, horizon: u32) -> SolveCaps {
    SolveCaps {
        node_cap: NODE_CAP,
        tt_bytes_cap: TT_BYTES,
        semantic_horizon: state.placements_made().saturating_add(horizon),
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn state_key(state: &HexoState) -> String {
    let mut stones = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&coord| {
            let owner = state.board().get(coord).expect("occupied owner");
            (coord.q, coord.r, owner.index())
        })
        .collect::<Vec<_>>();
    stones.sort_unstable();
    let phase = match state.phase() {
        TurnPhase::Opening => "O".to_owned(),
        TurnPhase::FirstStone => "F".to_owned(),
        TurnPhase::SecondStone { first } => format!("S:{},{}", first.q, first.r),
    };
    format!(
        "p={};c={};phase={phase};terminal={:?};stones={stones:?}",
        state.placements_made(),
        state.current_player().index(),
        state.terminal(),
    )
}

fn state_hash(state: &HexoState) -> u64 {
    fnv1a(state_key(state).as_bytes())
}

fn zoned_nodes(cert: &TssCertificate) -> usize {
    cert.nodes
        .iter()
        .filter(|node| matches!(node, CertNode::Universal { zone: Some(_), .. }))
        .count()
}

fn sibling_candidates(
    cluster: &str,
    report: &crate::tss_solver::PnInitTelemetryReport,
) -> Vec<SiblingPair> {
    let mut leaves = Vec::<(u64, String, HexoState)>::new();
    for node in &report.nodes {
        if !matches!(
            node.final_node_tag,
            "proven_leaf" | "depth_cutoff" | "refuted"
        ) {
            continue;
        }
        let Some(state) = node.captured_state.as_ref() else {
            continue;
        };
        leaves.push((node.serial, state_key(state), state.clone()));
    }
    leaves.sort_by(|left, right| (left.1.as_str(), left.0).cmp(&(right.1.as_str(), right.0)));
    leaves.dedup_by(|left, right| left.1 == right.1);

    let lineage = |mut serial: u64| {
        let mut path = Vec::new();
        loop {
            path.push(serial);
            let Some(parent) = report
                .nodes
                .get(serial as usize)
                .and_then(|node| node.parent_serial)
            else {
                break;
            };
            assert!(parent < serial, "telemetry ancestry must move backward");
            serial = parent;
        }
        path.reverse();
        path
    };

    let mut by_ancestor = BTreeMap::<u64, SiblingPair>::new();
    for left_index in 0..leaves.len() {
        for right_index in left_index + 1..leaves.len() {
            let (_, _, left) = &leaves[left_index];
            let (_, _, right) = &leaves[right_index];
            if left.placements_made() != right.placements_made()
                || left.current_player() != right.current_player()
                || left.phase() != right.phase()
            {
                continue;
            }
            let left_path = lineage(leaves[left_index].0);
            let right_path = lineage(leaves[right_index].0);
            let common = left_path
                .iter()
                .zip(&right_path)
                .take_while(|(a, b)| a == b)
                .map(|(serial, _)| *serial)
                .last();
            let Some(ancestor_id) = common else {
                continue;
            };
            // Distinct leaves whose paths diverge immediately below their LCA
            // are sibling leaves under that actual solve-tree ancestor.
            if ancestor_id == leaves[left_index].0 || ancestor_id == leaves[right_index].0 {
                continue;
            }
            let source = left.clone();
            let reuse = right.clone();
            let candidate = SiblingPair {
                cluster: cluster.to_owned(),
                ancestor_id,
                ancestor_hash: report
                    .nodes
                    .get(ancestor_id as usize)
                    .and_then(|node| node.captured_state.as_ref())
                    .map(state_hash)
                    .unwrap_or(0),
                source_hash: state_hash(&source),
                reuse_hash: state_hash(&reuse),
                source,
                reuse,
            };
            by_ancestor.entry(ancestor_id).or_insert(candidate);
        }
    }

    let mut pairs = by_ancestor.into_values().collect::<Vec<_>>();
    for pair in &pairs {
        let ancestor_hash = report
            .nodes
            .get(pair.ancestor_id as usize)
            .and_then(|node| node.captured_state.as_ref())
            .map(state_hash)
            .unwrap_or(0);
        assert_eq!(pair.ancestor_hash, ancestor_hash);
    }
    pairs.sort_by_key(|pair| {
        (
            pair.cluster.clone(),
            pair.ancestor_id,
            pair.source_hash,
            pair.reuse_hash,
        )
    });
    pairs
}

#[test]
#[ignore = "R-CREL-5 cohort gate; release, serial, --nocapture"]
fn crel_leaf_cohort_gate() {
    let _env = EnvGuard::selected_profile();
    let (eligible_games, batches) = load_batches();
    let mut abort_gate_fired = false;
    println!(
        "CREL_LEAF_META shadow_only=true selected_reservation_mib=8 fanout=1 corpus={} eligible_games={} games={} window_states={} roots={} seed=0x{SEED:016X} node_cap={NODE_CAP} tt_bytes={TT_BYTES} horizons={HORIZONS:?} width=vcf_pair_complete lazy_frontier=true interior_census_gate=true shared_fragments=false k_reply=false zone_options=default_off cluster=common_ancestor",
        corpus_path(),
        eligible_games,
        batches.len(),
        batches.len() * WINDOW_STATES,
        batches.len() * WINDOW_STATES,
    );
    for (batch_index, batch) in batches.iter().enumerate() {
        println!(
            "CREL_LEAF_BATCH index={batch_index} game_hash={} start_ply={} end_ply={}",
            batch.hash,
            batch.start_ply,
            batch.start_ply + WINDOW_STATES - 1,
        );
    }

    for horizon in HORIZONS {
        let mut discovery_roots = 0usize;
        let mut discovery_hard = 0usize;
        let mut discovery_zoned = 0usize;
        let mut leaf_events = 0usize;
        let mut sibling_groups = BTreeSet::<String>::new();
        let mut candidates = Vec::new();

        for (batch_index, batch) in batches.iter().enumerate() {
            let mut solver = make_solver();
            for (within_batch, state) in batch.states.iter().enumerate() {
                begin_pn_init_leaf_state_telemetry();
                let result = solver.solve_goal(state, &caps(state, horizon), SolveGoal::Win);
                let report = take_pn_init_telemetry_report().expect("active leaf telemetry");
                discovery_roots += 1;
                if result.status != ProofStatus::Unknown {
                    let cert = result.cert.as_ref().expect("hard discovery certificate");
                    assert!(TssVerifier.verify(state, cert, result.status));
                    discovery_hard += 1;
                    discovery_zoned += usize::from(zoned_nodes(cert) > 0);
                }
                leaf_events += report
                    .nodes
                    .iter()
                    .filter(|node| {
                        matches!(
                            node.final_node_tag,
                            "proven_leaf" | "depth_cutoff" | "refuted"
                        )
                    })
                    .count();
                let cluster = format!(
                    "h{horizon}:b{batch_index}:w{within_batch}:{}:p{}",
                    batch.hash,
                    batch.start_ply + within_batch
                );
                let found = sibling_candidates(&cluster, &report);
                for pair in &found {
                    sibling_groups.insert(format!("{}:a{}", pair.cluster, pair.ancestor_id));
                }
                candidates.extend(found);
            }
        }

        // At most one deterministic adjacent pair per common ancestor enters
        // source acquisition. This preserves the ancestor as the bootstrap
        // cluster and prevents high-fanout ancestors from dominating.
        let mut seen_ancestors = BTreeSet::new();
        candidates.retain(|pair| seen_ancestors.insert((pair.cluster.clone(), pair.ancestor_id)));

        let mut cohort_pairs = 0usize;
        let mut source_unknown = 0usize;
        let mut source_zoned = 0usize;
        let mut reuse_hard = 0usize;
        let mut hard_without_strict = 0usize;
        for pair in &candidates {
            let mut source_solver = make_solver();
            let source_result = source_solver.solve_goal(
                &pair.source,
                &caps(&pair.source, horizon),
                SolveGoal::Win,
            );
            if source_result.status == ProofStatus::Unknown {
                source_unknown += 1;
                continue;
            }
            let source_cert = source_result
                .cert
                .as_ref()
                .expect("hard source certificate");
            let source_strict = TssVerifier.verify(&pair.source, source_cert, source_result.status);
            assert!(source_strict, "strict source admission failed");
            hard_without_strict += usize::from(!source_strict);

            // The warm store is populated before the non-parent sibling solve.
            let warm_store = vec![source_cert.clone()];
            let zoned_present = warm_store.iter().any(|cert| zoned_nodes(cert) > 0);
            source_zoned += usize::from(zoned_present);

            let mut reuse_solver = make_solver();
            let reuse_result =
                reuse_solver.solve_goal(&pair.reuse, &caps(&pair.reuse, horizon), SolveGoal::Win);
            if reuse_result.status != ProofStatus::Unknown {
                let cert = reuse_result.cert.as_ref().expect("hard reuse certificate");
                assert!(TssVerifier.verify(&pair.reuse, cert, reuse_result.status));
                reuse_hard += 1;
            }
            cohort_pairs += 1;
            println!(
                "CREL_LEAF_PAIR horizon={horizon} cluster={} ancestor_event={} ancestor_hash={:016X} source_hash={:016X} reuse_hash={:016X} source_ply={} reuse_ply={} equal_root={} direct_parent=false source_status={:?} reuse_status={:?} warm_store_certs={} zoned_present={} hard_without_strict=0",
                pair.cluster,
                pair.ancestor_id,
                pair.ancestor_hash,
                pair.source_hash,
                pair.reuse_hash,
                pair.source.placements_made(),
                pair.reuse.placements_made(),
                RootBinding::from_state(&pair.source) == RootBinding::from_state(&pair.reuse),
                source_result.status,
                reuse_result.status,
                warm_store.len(),
                zoned_present,
            );
        }
        assert_eq!(hard_without_strict, 0);
        let coverage = if cohort_pairs == 0 {
            0.0
        } else {
            source_zoned as f64 / cohort_pairs as f64
        };
        println!(
            "CREL_LEAF_COHORT horizon={horizon} discovery_roots={discovery_roots} discovery_hard={discovery_hard} discovery_zoned={discovery_zoned} leaf_events={leaf_events} sibling_groups={} candidate_pairs={} source_unknown={source_unknown} cohort_pairs={cohort_pairs} reuse_hard={reuse_hard} positive_zoned={source_zoned} zoned_coverage_fraction={coverage:.6} zoned_coverage_percent={:.3} hard_without_strict=0",
            sibling_groups.len(),
            candidates.len(),
            coverage * 100.0,
        );
        println!(
            "CREL_LEAF_GATE horizon={horizon} threshold=0.200000 observed={coverage:.6} pass={} action={}",
            coverage >= 0.20,
            if coverage >= 0.20 {
                "continue_to_selected_cell_measurement"
            } else {
                "abort_do_not_measure"
            },
        );
        abort_gate_fired |= coverage < 0.20;
    }
    if abort_gate_fired {
        println!(
            "CREL_LEAF_VERDICT verdict=ABORT criterion=positive_zoned_coverage_below_20_percent economics_arms_run=false hard_without_strict=0 strict_verifier_unchanged=true"
        );
    } else {
        println!(
            "CREL_LEAF_VERDICT verdict=READY_FOR_MEASUREMENT criterion=positive_zoned_coverage_at_least_20_percent economics_arms_run=false hard_without_strict=0 strict_verifier_unchanged=true"
        );
    }
}
