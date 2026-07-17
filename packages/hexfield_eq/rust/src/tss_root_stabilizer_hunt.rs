//! Candidate-3 opening-atlas root-stabilizer sizing campaign.
//!
//! Everything in this module is `cfg(test)`. The consuming arm is additionally
//! gated by `TSS_ROOT_STABILIZER_CONSUME=1` inside the test-only wide solver.

use std::collections::HashMap;
use std::ffi::OsString;
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{
    take_root_stabilizer_report, RootStabilizerReport, TssSolver, WidthOptions,
};
use crate::tss_verify::{d6_transform_coord, TssVerifier, D6_SYMMETRY_COUNT};

const DEFAULT_CORPUS: &str =
    "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl";
const OFFICIAL_TT_BYTES: usize = 1 << 30;
const ROOT_FLAGS: [&str; 3] = [
    "TSS_ROOT_STABILIZER_TELEMETRY",
    "TSS_ROOT_STABILIZER_CONSUME",
    "TSS_ROOT_STABILIZER_INJECT_INCONSISTENCY",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FamilyKey([(i16, i16); 2]);

impl FamilyKey {
    fn label(self) -> String {
        let [a, b] = self.0;
        format!("({},{});({},{})", a.0, a.1, b.0, b.1)
    }
}

#[derive(Clone, Debug)]
struct Family {
    key: FamilyKey,
    games: u64,
    representative_order: [(i16, i16); 2],
    stabilizer: usize,
}

#[derive(Clone)]
struct Run {
    status: ProofStatus,
    nodes: u64,
    expansions: u64,
    elapsed_nanos: u64,
    report: RootStabilizerReport,
}

struct RootEnvGuard(Vec<(&'static str, Option<OsString>)>);

impl RootEnvGuard {
    fn set(consume: bool, inject: bool) -> Self {
        let old = ROOT_FLAGS
            .iter()
            .map(|&name| (name, std::env::var_os(name)))
            .collect::<Vec<_>>();
        for name in ROOT_FLAGS {
            std::env::remove_var(name);
        }
        std::env::set_var("TSS_ROOT_STABILIZER_TELEMETRY", "1");
        if consume {
            std::env::set_var("TSS_ROOT_STABILIZER_CONSUME", "1");
        }
        if inject {
            std::env::set_var("TSS_ROOT_STABILIZER_INJECT_INCONSISTENCY", "1");
        }
        Self(old)
    }
}

impl Drop for RootEnvGuard {
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

fn parse_moves(line: &str) -> Option<Vec<(i16, i16)>> {
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
    Some(
        ints.chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect(),
    )
}

fn transform_pair(pair: [(i16, i16); 2], symmetry: u8) -> Option<[(i16, i16); 2]> {
    Some([
        {
            let coord = d6_transform_coord(HexCoord::new(pair[0].0, pair[0].1), symmetry)?;
            (coord.q, coord.r)
        },
        {
            let coord = d6_transform_coord(HexCoord::new(pair[1].0, pair[1].1), symmetry)?;
            (coord.q, coord.r)
        },
    ])
}

fn canonical_family(pair: [(i16, i16); 2]) -> (FamilyKey, [(i16, i16); 2]) {
    let mut best = None::<(FamilyKey, [(i16, i16); 2])>;
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let ordered = transform_pair(pair, symmetry).expect("opening coordinates fit D6");
        let mut unordered = ordered;
        unordered.sort_unstable();
        let candidate = (FamilyKey(unordered), ordered);
        if best.as_ref().is_none_or(|old| candidate < *old) {
            best = Some(candidate);
        }
    }
    best.expect("D6 contains identity")
}

fn geometric_stabilizer(key: FamilyKey) -> usize {
    (0..D6_SYMMETRY_COUNT)
        .filter(|&symmetry| {
            let mut pair = transform_pair(key.0, symmetry).expect("opening coordinates fit D6");
            pair.sort_unstable();
            FamilyKey(pair) == key
        })
        .count()
}

fn load_families() -> Vec<Family> {
    let path =
        std::env::var("TSS_ROOT_STABILIZER_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS.to_owned());
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read root-stabilizer corpus {path}: {error}"));
    let mut games = 0u64;
    let mut counts = HashMap::<FamilyKey, (u64, [(i16, i16); 2])>::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let moves = parse_moves(line).expect("valid human-corpus moves");
        if moves.len() < 3 {
            continue;
        }
        games += 1;
        let (key, ordered) = canonical_family([moves[1], moves[2]]);
        let entry = counts.entry(key).or_insert((0, ordered));
        entry.0 += 1;
        if ordered < entry.1 {
            entry.1 = ordered;
        }
    }
    assert_eq!(games, 6_902, "eligible human-corpus game count drifted");
    let mut families = counts
        .into_iter()
        .map(|(key, (games, representative_order))| Family {
            key,
            games,
            representative_order,
            stabilizer: geometric_stabilizer(key),
        })
        .collect::<Vec<_>>();
    families.sort_by(|left, right| {
        right
            .games
            .cmp(&left.games)
            .then_with(|| left.key.cmp(&right.key))
    });
    assert_eq!(
        families.len(),
        262,
        "canonical opening-family count drifted"
    );
    families
}

fn opening_state(family: &Family) -> HexoState {
    let mut state = HexoState::new();
    for (q, r) in [
        (0, 0),
        family.representative_order[0],
        family.representative_order[1],
    ] {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(q, r),
            },
        )
        .unwrap_or_else(|error| panic!("replay family {}: {error}", family.key.label()));
    }
    state
}

fn transformed_state(state: &HexoState, symmetry: u8) -> HexoState {
    let mut transformed = HexoState::new();
    for record in state.placement_history() {
        let coord = d6_transform_coord(record.coord, symmetry).expect("D6 coordinate fits");
        apply_placement(&mut transformed, Placement { coord }).expect("transformed move legal");
    }
    transformed
}

fn run(state: &HexoState, caps: &SolveCaps, consume: bool, inject: bool) -> Run {
    let _guard = RootEnvGuard::set(consume, inject);
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::round3_consume());
    let started = Instant::now();
    let result = solver.solve_goal(state, caps, SolveGoal::Win);
    let elapsed_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if result.status != ProofStatus::Unknown {
        let cert = result
            .cert
            .as_ref()
            .expect("hard result carries a certificate");
        assert!(
            TssVerifier.verify(state, cert, result.status),
            "strict verifier rejected root-stabilizer hard result"
        );
    }
    let report = take_root_stabilizer_report().expect("root telemetry report");
    Run {
        status: result.status,
        nodes: result.stats.nodes,
        expansions: result.stats.expansions,
        elapsed_nanos,
        report,
    }
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn parse_caps() -> Vec<u64> {
    std::env::var("TSS_ROOT_STABILIZER_CAPS")
        .unwrap_or_else(|_| "10000,100000".to_owned())
        .split(',')
        .map(|value| value.trim().parse().expect("numeric root-stabilizer cap"))
        .collect()
}

fn census_rows(families: &[Family]) {
    let mut groups = HashMap::<usize, (usize, u64)>::new();
    for family in families {
        let group = groups.entry(family.stabilizer).or_default();
        group.0 += 1;
        group.1 += family.games;
    }
    for stabilizer in [1usize, 2, 4] {
        let (family_count, games) = groups[&stabilizer];
        println!(
            "RS_CENSUS stabilizer={stabilizer} families={family_count} games={games} game_pct={:.4}",
            100.0 * games as f64 / 6_902.0
        );
    }
    let nontrivial_games = families
        .iter()
        .filter(|family| family.stabilizer > 1)
        .map(|family| family.games)
        .sum::<u64>();
    let weighted_ceiling = families
        .iter()
        .map(|family| family.games as f64 * (1.0 - 1.0 / family.stabilizer as f64))
        .sum::<f64>();
    println!(
        "RS_CENSUS_DONE families={} games=6902 nontrivial_games={nontrivial_games} nontrivial_game_pct={:.4} weighted_root_child_removal_ceiling_pct={:.4}",
        families.len(),
        100.0 * nontrivial_games as f64 / 6_902.0,
        100.0 * weighted_ceiling / 6_902.0,
    );
    for (rank, family) in families.iter().take(20).enumerate() {
        println!(
            "RS_FAMILY rank={} key={} games={} game_pct={:.6} stabilizer={}",
            rank + 1,
            family.key.label(),
            family.games,
            100.0 * family.games as f64 / 6_902.0,
            family.stabilizer,
        );
    }
}

#[test]
fn root_stabilizer_shadow_is_default_off_and_fail_closed() {
    let families = load_families();
    let state = opening_state(&families[0]);
    let caps = SolveCaps {
        node_cap: 128,
        tt_bytes_cap: 4 << 20,
        semantic_horizon: u32::MAX,
    };

    for name in ROOT_FLAGS {
        std::env::remove_var(name);
    }
    let mut plain_solver = TssSolver::default();
    plain_solver.set_width_options(WidthOptions::round3_consume());
    let plain = plain_solver.solve_goal(&state, &caps, SolveGoal::Win);
    assert!(take_root_stabilizer_report().is_none());

    let shadow = run(&state, &caps, false, false);
    assert_eq!(shadow.status, plain.status);
    assert_eq!(shadow.nodes, plain.stats.nodes);
    assert_eq!(shadow.expansions, plain.stats.expansions);
    assert!(shadow.report.eligible);
    assert!(shadow.report.complete_binding_checked);
    assert!(!shadow.report.fail_closed);
    assert_eq!(shadow.report.stabilizer.len(), families[0].stabilizer);
    assert!(shadow.report.orbit_count <= shadow.report.raw_children);

    let fault = run(&state, &caps, true, true);
    assert!(fault.report.fail_closed);
    assert!(!fault.report.consumed);
    assert_eq!(fault.status, shadow.status);
    assert_eq!(fault.nodes, shadow.nodes);
    assert_eq!(fault.expansions, shadow.expansions);
}

#[test]
fn root_stabilizer_all_transforms_partition_and_verify() {
    let families = load_families();
    let state = opening_state(&families[0]);
    let caps = SolveCaps {
        node_cap: 128,
        tt_bytes_cap: 4 << 20,
        semantic_horizon: u32::MAX,
    };
    let mut expected = None;
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let transformed = transformed_state(&state, symmetry);
        let baseline = run(&transformed, &caps, false, false);
        let consumed = run(&transformed, &caps, true, false);
        assert_eq!(baseline.status, consumed.status, "symmetry={symmetry}");
        assert!(!baseline.report.fail_closed, "symmetry={symmetry}");
        assert!(!consumed.report.fail_closed, "symmetry={symmetry}");
        let shape = (
            baseline.report.stabilizer.len(),
            baseline.report.raw_children,
            baseline.report.orbit_count,
            baseline.report.fixed_children,
        );
        assert_eq!(*expected.get_or_insert(shape), shape, "symmetry={symmetry}");
    }
}

#[test]
#[ignore = "Candidate-3 top-10 root-universe shadow census"]
fn root_stabilizer_atlas_shadow_campaign() {
    let families = load_families();
    census_rows(&families);
    let cap = std::env::var("TSS_ROOT_STABILIZER_SHADOW_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric shadow cap"))
        .unwrap_or(128);
    let tt_bytes = std::env::var("TSS_BACKWALK_TT_BYTES")
        .ok()
        .map(|value| value.parse::<usize>().expect("numeric TT bytes"))
        .unwrap_or(OFFICIAL_TT_BYTES);
    for (rank, family) in families.iter().take(10).enumerate() {
        let state = opening_state(family);
        let result = run(
            &state,
            &SolveCaps {
                node_cap: cap,
                tt_bytes_cap: tt_bytes,
                semantic_horizon: u32::MAX,
            },
            false,
            false,
        );
        assert!(result.report.eligible && result.report.complete_binding_checked);
        assert!(!result.report.fail_closed);
        let visited_orbits = result
            .report
            .orbits
            .iter()
            .filter(|orbit| orbit.descents != 0)
            .count();
        let below_expansions = result
            .report
            .orbits
            .iter()
            .map(|orbit| orbit.expansions)
            .sum::<u64>();
        let below_nanos = result
            .report
            .orbits
            .iter()
            .map(|orbit| orbit.wall_nanos)
            .sum::<u64>();
        println!(
            "RS_SHADOW rank={} key={} games={} cap={cap} stabilizer={} raw_children={} orbits={} orbits_removed={} fixed_children={} root_generation_ms={:.3} visited_orbits={visited_orbits} below_expansions={below_expansions} below_wall_ms={:.3} status={} total_expansions={} total_ms={:.3}",
            rank + 1,
            family.key.label(),
            family.games,
            result.report.stabilizer.len(),
            result.report.raw_children,
            result.report.orbit_count,
            result
                .report
                .raw_children
                .saturating_sub(result.report.orbit_count),
            result.report.fixed_children,
            result.report.root_generation_nanos as f64 / 1e6,
            below_nanos as f64 / 1e6,
            status_name(result.status),
            result.expansions,
            result.elapsed_nanos as f64 / 1e6,
        );
    }
    println!("RS_SHADOW_DONE families=10 cap={cap} strict_verify=PASS fail_closed=PASS");
}

#[test]
#[ignore = "Candidate-3 A-0 top-10 1-GiB sizing campaign"]
fn tss_root_stabilizer_atlas_campaign() {
    assert_eq!(
        std::env::var("TSS_BACKWALK_TT_BYTES").ok().as_deref(),
        Some("1073741824"),
        "official campaign requires the 1-GiB TT profile"
    );
    for flag in [
        "TSS_LAZY_FRONTIER",
        "TSS_INTERIOR_CENSUS_GATE",
        "TSS_INCR_DEFENDER",
    ] {
        assert_eq!(
            std::env::var(flag).ok().as_deref(),
            Some("1"),
            "official campaign requires {flag}=1"
        );
    }
    let tt_bytes = std::env::var("TSS_BACKWALK_TT_BYTES")
        .expect("official TT bytes")
        .parse::<usize>()
        .expect("numeric official TT bytes");
    assert_eq!(tt_bytes, OFFICIAL_TT_BYTES);
    let family_limit = std::env::var("TSS_ROOT_STABILIZER_FAMILIES")
        .ok()
        .map(|value| value.parse::<usize>().expect("numeric family limit"))
        .unwrap_or(10);
    let transform_limit = std::env::var("TSS_ROOT_STABILIZER_TRANSFORMS")
        .ok()
        .map(|value| value.parse::<u8>().expect("numeric transform limit"))
        .unwrap_or(D6_SYMMETRY_COUNT);
    assert!((1..=D6_SYMMETRY_COUNT).contains(&transform_limit));
    let caps = parse_caps();
    let families = load_families();
    assert!(family_limit <= families.len());
    census_rows(&families);
    println!(
        "RS_SETUP families={family_limit} transforms={transform_limit} caps={caps:?} tt_bytes={tt_bytes} lazy=1 interior_gate=1 incr_defender=1"
    );

    for cap in caps {
        let solve_caps = SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes,
            semantic_horizon: u32::MAX,
        };
        let mut family_before = 0.0f64;
        let mut family_after = 0.0f64;
        let mut weighted_before = 0.0f64;
        let mut weighted_after = 0.0f64;
        let mut total_before_expansions = 0u64;
        let mut total_after_expansions = 0u64;

        for (family_index, family) in families.iter().take(family_limit).enumerate() {
            let root = opening_state(family);
            let mut baseline_status = None;
            let mut consumed_status = None;
            let mut before_ns = 0u64;
            let mut after_ns = 0u64;
            let mut before_expansions = 0u64;
            let mut after_expansions = 0u64;
            let mut canonical_report = None::<RootStabilizerReport>;

            for symmetry in 0..transform_limit {
                let transformed = transformed_state(&root, symmetry);
                let (baseline, consumed) = if symmetry % 2 == 0 {
                    (
                        run(&transformed, &solve_caps, false, false),
                        run(&transformed, &solve_caps, true, false),
                    )
                } else {
                    let consumed = run(&transformed, &solve_caps, true, false);
                    let baseline = run(&transformed, &solve_caps, false, false);
                    (baseline, consumed)
                };
                assert!(baseline.report.eligible && consumed.report.eligible);
                assert!(
                    baseline.report.complete_binding_checked
                        && consumed.report.complete_binding_checked
                );
                assert!(!baseline.report.fail_closed, "baseline symmetry={symmetry}");
                assert!(!consumed.report.fail_closed, "consume symmetry={symmetry}");
                assert_eq!(baseline.status, consumed.status, "symmetry={symmetry}");
                assert_eq!(
                    (
                        baseline.report.stabilizer.len(),
                        baseline.report.raw_children,
                        baseline.report.orbit_count,
                        baseline.report.fixed_children,
                    ),
                    (
                        consumed.report.stabilizer.len(),
                        consumed.report.raw_children,
                        consumed.report.orbit_count,
                        consumed.report.fixed_children,
                    ),
                    "A/B orbit shape mismatch symmetry={symmetry}"
                );
                let shape = (
                    baseline.report.stabilizer.len(),
                    baseline.report.raw_children,
                    baseline.report.orbit_count,
                    baseline.report.fixed_children,
                );
                if let Some(report) = canonical_report.as_ref() {
                    assert_eq!(
                        shape,
                        (
                            report.stabilizer.len(),
                            report.raw_children,
                            report.orbit_count,
                            report.fixed_children,
                        ),
                        "D6 orbit shape mismatch symmetry={symmetry}"
                    );
                } else {
                    canonical_report = Some(baseline.report.clone());
                }
                assert_eq!(
                    *baseline_status.get_or_insert(baseline.status),
                    baseline.status,
                    "baseline D6 verdict mismatch symmetry={symmetry}"
                );
                assert_eq!(
                    *consumed_status.get_or_insert(consumed.status),
                    consumed.status,
                    "consume D6 verdict mismatch symmetry={symmetry}"
                );
                before_ns = before_ns.saturating_add(baseline.elapsed_nanos);
                after_ns = after_ns.saturating_add(consumed.elapsed_nanos);
                before_expansions = before_expansions.saturating_add(baseline.expansions);
                after_expansions = after_expansions.saturating_add(consumed.expansions);
                println!(
                    "RS_RUN rank={} key={} games={} cap={cap} symmetry={symmetry} stabilizer={} raw_children={} orbits={} fixed_children={} baseline_status={} consume_status={} baseline_ms={:.3} consume_ms={:.3} baseline_expansions={} consume_expansions={} root_gen_ms={:.3} fail_closed=0 strict_verify=PASS",
                    family_index + 1,
                    family.key.label(),
                    family.games,
                    baseline.report.stabilizer.len(),
                    baseline.report.raw_children,
                    baseline.report.orbit_count,
                    baseline.report.fixed_children,
                    status_name(baseline.status),
                    status_name(consumed.status),
                    baseline.elapsed_nanos as f64 / 1e6,
                    consumed.elapsed_nanos as f64 / 1e6,
                    baseline.expansions,
                    consumed.expansions,
                    baseline.report.root_generation_nanos as f64 / 1e6,
                );
            }

            let report = canonical_report.expect("canonical root report");
            let mut zero_work_orbits = 0usize;
            for (orbit, row) in report.orbits.iter().enumerate() {
                if row.descents == 0 && row.expansions == 0 && row.wall_nanos == 0 {
                    zero_work_orbits += 1;
                    continue;
                }
                println!(
                    "RS_ORBIT rank={} key={} cap={cap} orbit={orbit} representative={} members={} fixed={} descents={} expansions={} wall_ms={:.3}",
                    family_index + 1,
                    family.key.label(),
                    row.representative,
                    row.members.join("|"),
                    u8::from(row.members.len() == 1),
                    row.descents,
                    row.expansions,
                    row.wall_nanos as f64 / 1e6,
                );
            }
            println!(
                "RS_ORBIT_ZERO rank={} key={} cap={cap} zero_work_orbits={zero_work_orbits} total_orbits={}",
                family_index + 1,
                family.key.label(),
                report.orbit_count,
            );
            let transforms = f64::from(transform_limit);
            let mean_before = before_ns as f64 / transforms;
            let mean_after = after_ns as f64 / transforms;
            let wall_delta_pct = if mean_before == 0.0 {
                0.0
            } else {
                100.0 * (mean_before - mean_after) / mean_before
            };
            family_before += mean_before;
            family_after += mean_after;
            weighted_before += mean_before * family.games as f64;
            weighted_after += mean_after * family.games as f64;
            total_before_expansions = total_before_expansions.saturating_add(before_expansions);
            total_after_expansions = total_after_expansions.saturating_add(after_expansions);
            println!(
                "RS_ATTRIBUTION rank={} key={} games={} cap={cap} stabilizer={} raw_children={} orbits={} orbits_removed={} fixed_children={} baseline_status={} consume_status={} baseline_mean_ms={:.3} consume_mean_ms={:.3} wall_delta_pct={wall_delta_pct:.4} baseline_expansions={} consume_expansions={} expansion_delta_pct={:.4}",
                family_index + 1,
                family.key.label(),
                family.games,
                report.stabilizer.len(),
                report.raw_children,
                report.orbit_count,
                report.raw_children.saturating_sub(report.orbit_count),
                report.fixed_children,
                status_name(baseline_status.expect("baseline status")),
                status_name(consumed_status.expect("consume status")),
                mean_before / 1e6,
                mean_after / 1e6,
                before_expansions,
                after_expansions,
                if before_expansions == 0 { 0.0 } else { 100.0 * (before_expansions as f64 - after_expansions as f64) / before_expansions as f64 },
            );
        }

        let family_delta_pct = 100.0 * (family_before - family_after) / family_before;
        let game_delta_pct = 100.0 * (weighted_before - weighted_after) / weighted_before;
        println!(
            "RS_AGGREGATE cap={cap} families={family_limit} transforms={transform_limit} family_weighted_baseline_ms={:.3} family_weighted_consume_ms={:.3} family_weighted_wall_delta_pct={family_delta_pct:.4} game_weighted_baseline_ms={:.3} game_weighted_consume_ms={:.3} game_weighted_wall_delta_pct={game_delta_pct:.4} baseline_expansions={total_before_expansions} consume_expansions={total_after_expansions} verdict_identity=PASS strict_verify=PASS differential_12=PASS fail_closed=PASS",
            family_before / 1e6,
            family_after / 1e6,
            weighted_before / 1e6,
            weighted_after / 1e6,
        );
    }
    println!("RS_CAMPAIGN_DONE verdict_identity=PASS strict_verify=PASS fail_closed=PASS");
}
