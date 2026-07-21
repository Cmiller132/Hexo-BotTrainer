//! Default-off J2near matched-cap and broader-coverage measurement harness.
//! The ignored tests write machine-readable rows under `.scratch/j2near_ab`.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};
use serde_json::{json, Value};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::TssVerifier;

#[derive(Clone)]
struct Position {
    id: String,
    set: String,
    state: HexoState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Outcome {
    status: &'static str,
    nodes: u64,
    verified: bool,
    verify_failed: u64,
    wall_nanos: u64,
}

fn root_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "win",
        ProofStatus::Loss => "loss",
        ProofStatus::Unknown => "unknown",
    }
}

fn load_set(name: &str) -> Vec<Position> {
    let path = root_dir()
        .join("scripts/tss_harness/sets")
        .join(format!("{name}.jsonl"));
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    BufReader::new(file)
        .lines()
        .filter_map(|line| {
            let line = line.expect("read set row");
            if line.trim().is_empty() {
                return None;
            }
            let row: Value = serde_json::from_str(&line).expect("parse set row");
            let id = row["pos_id"].as_str().expect("pos_id").to_owned();
            let mut state = HexoState::new();
            for pair in row["moves"].as_array().expect("moves") {
                let pair = pair.as_array().expect("move pair");
                let coord = HexCoord::new(
                    i16::try_from(pair[0].as_i64().expect("q")).expect("q fits i16"),
                    i16::try_from(pair[1].as_i64().expect("r")).expect("r fits i16"),
                );
                apply_placement(&mut state, Placement { coord }).expect("legal frozen position");
            }
            Some(Position {
                id,
                set: name.to_owned(),
                state,
            })
        })
        .collect()
}

fn grind_ids() -> HashSet<String> {
    let path = root_dir().join("raws/lanec_labels.jsonl");
    BufReader::new(File::open(&path).expect("open grind labels"))
        .lines()
        .filter_map(|line| {
            let row: Value = serde_json::from_str(&line.expect("read label row")).expect("label json");
            (row["source"].as_str() == Some("grind"))
                .then(|| row["pos_id"].as_str().expect("grind pos_id").to_owned())
        })
        .collect()
}

fn all_frozen() -> Vec<Position> {
    ["selfplay_v1", "human_v1", "puzzle_v3"]
        .into_iter()
        .flat_map(load_set)
        .collect()
}

fn make_solver(j2near: bool) -> TssSolver {
    let mut solver = TssSolver::default();
    solver.configure_leaf_profile();
    solver.set_width_options(if j2near {
        WidthOptions::vcf_pair_j2near()
    } else {
        WidthOptions::vcf_pair_complete()
    });
    solver.set_dual_pass(true);
    solver
}

fn solve_one(solver: &mut TssSolver, state: &HexoState, cap: u64, tt_bytes: usize) -> Outcome {
    let started = Instant::now();
    let result = solver.solve(
        state,
        &SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes,
            semantic_horizon: u32::MAX,
        },
    );
    let wall_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let verified = result.cert.as_ref().is_some_and(|cert| {
        TssVerifier.verify(state, cert, result.status)
    });
    let verify_failed = u64::from(result.status != ProofStatus::Unknown && !verified);
    Outcome {
        status: status_name(result.status),
        nodes: result.stats.nodes,
        verified,
        verify_failed,
        wall_nanos,
    }
}

fn run_repetitions(
    positions: &[Position],
    cap: u64,
    tt_bytes: usize,
    repetitions: usize,
) -> (Vec<Vec<Outcome>>, Vec<Vec<Outcome>>) {
    let mut off = vec![Vec::with_capacity(repetitions); positions.len()];
    let mut on = vec![Vec::with_capacity(repetitions); positions.len()];
    for repetition in 0..repetitions {
        for &j2near in if repetition % 2 == 0 { &[false, true] } else { &[true, false] } {
            let sink = if j2near { &mut on } else { &mut off };
            let mut solver = make_solver(j2near);
            for (index, position) in positions.iter().enumerate() {
                let result = solve_one(&mut solver, &position.state, cap, tt_bytes);
                assert_eq!(result.verify_failed, 0, "{} verifier failure", position.id);
                sink[index].push(result);
            }
        }
    }
    for (index, position) in positions.iter().enumerate() {
        for runs in [&off[index], &on[index]] {
            let first = &runs[0];
            assert!(runs.iter().all(|run| {
                run.status == first.status
                    && run.nodes == first.nodes
                    && run.verified == first.verified
                    && run.verify_failed == first.verify_failed
            }), "{} nondeterministic verdict or node count", position.id);
        }
    }
    (off, on)
}

fn output_writer(name: &str) -> BufWriter<File> {
    let dir = root_dir().join(".scratch/j2near_ab");
    fs::create_dir_all(&dir).expect("create J2near scratch output");
    BufWriter::new(File::create(dir.join(name)).expect("create J2near output"))
}

fn write_rows(
    writer: &mut impl Write,
    positions: &[Position],
    cap: u64,
    tt_bytes: usize,
    off: &[Vec<Outcome>],
    on: &[Vec<Outcome>],
) {
    for (index, position) in positions.iter().enumerate() {
        let left = &off[index][0];
        let right = &on[index][0];
        let row = json!({
            "set": position.set,
            "pos_id": position.id,
            "cap": cap,
            "tt_bytes": tt_bytes,
            "off_status": left.status,
            "on_status": right.status,
            "off_verified": left.verified,
            "on_verified": right.verified,
            "off_nodes": left.nodes,
            "on_nodes": right.nodes,
            "off_verify_failed": left.verify_failed,
            "on_verify_failed": right.verify_failed,
            "off_wall_nanos": off[index].iter().map(|run| run.wall_nanos).collect::<Vec<_>>(),
            "on_wall_nanos": on[index].iter().map(|run| run.wall_nanos).collect::<Vec<_>>(),
        });
        serde_json::to_writer(&mut *writer, &row).expect("write A/B row");
        writeln!(writer).expect("finish A/B row");
    }
}

fn archived_identity() -> BTreeMap<(String, String), (&'static str, u64)> {
    let base = root_dir().join(
        "scripts/tss_harness/harness_runs/20260720_231040_dualpass_adoption",
    );
    let mut out = BTreeMap::new();
    for set in ["selfplay_v1", "human_v1", "puzzle_v3"] {
        let path = base.join(format!("records_dualpass_adoption_{set}.jsonl"));
        for line in BufReader::new(File::open(&path).expect("open identity archive")).lines() {
            let row: Value = serde_json::from_str(&line.expect("read identity row"))
                .expect("identity json");
            let status = match row["status"].as_str().expect("identity status") {
                "win" => "win",
                "loss" => "loss",
                "unknown" => "unknown",
                other => panic!("unexpected identity status {other}"),
            };
            out.insert(
                (
                    set.to_owned(),
                    row["pos_id"].as_str().expect("identity pos_id").to_owned(),
                ),
                (status, row["cost"].as_u64().expect("identity cost")),
            );
        }
    }
    out
}

#[test]
#[ignore = "three-repeat matched-cap J2near A/B over frozen cohorts"]
fn tss_j2near_matched_ab() {
    let repetitions = std::env::var("TSS_J2NEAR_REPETITIONS")
        .ok()
        .map(|value| value.parse::<usize>().expect("numeric repetitions"))
        .unwrap_or(3);
    assert!(repetitions >= 3, "wall claims require at least three repetitions");
    let frozen = all_frozen();
    assert_eq!(frozen.len(), 6_443);
    let mut writer = output_writer("matched.jsonl");
    for cap in [500, 2_000] {
        let (off, on) = run_repetitions(&frozen, cap, 256 << 10, repetitions);
        if cap == 500 {
            let archive = archived_identity();
            assert_eq!(archive.len(), frozen.len());
            for (position, runs) in frozen.iter().zip(&off) {
                let expected = archive
                    .get(&(position.set.clone(), position.id.clone()))
                    .expect("position in identity archive");
                assert_eq!(
                    (runs[0].status, runs[0].nodes),
                    *expected,
                    "{} flag-off identity mismatch",
                    position.id,
                );
            }
        }
        write_rows(&mut writer, &frozen, cap, 256 << 10, &off, &on);
        writer.flush().expect("flush frozen A/B");
        println!("J2NEAR_AB cohort=frozen cap={cap} rows={} repetitions={repetitions}", frozen.len());
    }

}

#[test]
#[ignore = "three-repeat matched-cap J2near A/B over the 248 grind roots"]
fn tss_j2near_grind_ab() {
    let repetitions = std::env::var("TSS_J2NEAR_REPETITIONS")
        .ok()
        .map(|value| value.parse::<usize>().expect("numeric repetitions"))
        .unwrap_or(3);
    assert!(repetitions >= 3, "wall claims require at least three repetitions");
    let ids = grind_ids();
    let mut by_id = BTreeMap::new();
    for position in all_frozen() {
        by_id.insert(position.id.clone(), position);
    }
    let grinds = ids
        .iter()
        .map(|id| by_id.get(id).unwrap_or_else(|| panic!("missing grind {id}")).clone())
        .collect::<Vec<_>>();
    assert_eq!(grinds.len(), 248);
    let (off, on) = run_repetitions(&grinds, 50_000, 256 << 20, repetitions);
    let mut writer = output_writer("grind.jsonl");
    write_rows(&mut writer, &grinds, 50_000, 256 << 20, &off, &on);
    writer.flush().expect("flush grind A/B");
    println!("J2NEAR_AB cohort=grind cap=50000 rows={} repetitions={repetitions}", grinds.len());
}

#[test]
#[ignore = "full-population puzzle_v3 cap-100k broader J2near win check"]
fn tss_j2near_broader_unknown_check() {
    let population = load_set("puzzle_v3");
    assert_eq!(population.len(), 468);
    let mut off_solver = make_solver(false);
    let mut unknown = Vec::new();
    let mut off = Vec::new();
    for position in &population {
        let result = solve_one(&mut off_solver, &position.state, 100_000, 256 << 20);
        assert_eq!(result.verify_failed, 0, "{} verifier failure", position.id);
        if result.status == "unknown" {
            unknown.push(position.clone());
            off.push(vec![result]);
        }
    }
    let mut on_solver = make_solver(true);
    let on = unknown
        .iter()
        .map(|position| {
            let result = solve_one(&mut on_solver, &position.state, 100_000, 256 << 20);
            assert_eq!(result.verify_failed, 0, "{} verifier failure", position.id);
            vec![result]
        })
        .collect::<Vec<_>>();
    let mut writer = output_writer("broader_puzzle_unknown_100k.jsonl");
    write_rows(&mut writer, &unknown, 100_000, 256 << 20, &off, &on);
    writer.flush().expect("flush broader check");
    let upgrades = unknown
        .iter()
        .enumerate()
        .filter(|(index, _)| off[*index][0].status == "unknown" && on[*index][0].status == "win")
        .map(|(_, position)| position.id.as_str())
        .collect::<Vec<_>>();
    println!("J2NEAR_BROADER frame=all_flag_off_unknown population=468 unknowns={} upgrades={} ids={upgrades:?}", unknown.len(), upgrades.len());
}
