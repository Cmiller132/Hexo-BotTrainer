//! Phase-B test-only trajectory corpus harness.
//!
//! The ignored battery replays the 248 frozen grind roots, creates a fresh
//! solver for every row, and drives only the WIN goal at the preregistered
//! 5,000-node / 256-KiB / unbounded-horizon shape. Snapshot emission itself
//! lives beside `WidePnSearch` and is additionally env-gated.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::TssSolver;
use crate::tss_verify::TssVerifier;

fn json_string(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\"");
    let tail = line.split_once(&needle)?.1;
    let tail = tail.split_once(':')?.1.trim_start();
    let quoted = tail.strip_prefix('"')?;
    let end = quoted.find('"')?;
    Some(quoted[..end].to_owned())
}

fn json_bool(line: &str, field: &str) -> Option<bool> {
    let needle = format!("\"{field}\"");
    let tail = line.split_once(&needle)?.1;
    let tail = tail.split_once(':')?.1.trim_start();
    if tail.starts_with("true") {
        Some(true)
    } else if tail.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_moves(line: &str) -> Vec<(i16, i16)> {
    let tail = line.split_once("\"moves\"").expect("moves field").1;
    let array = tail.split_once(':').expect("moves colon").1.trim_start();
    let mut depth = 0i32;
    let mut end = None;
    for (index, byte) in array.bytes().enumerate() {
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(index + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let array = &array[..end.expect("complete moves array")];
    let mut ints = Vec::new();
    let mut token = String::new();
    for ch in array.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            token.push(ch);
        } else if !token.is_empty() {
            ints.push(token.parse::<i16>().expect("move integer"));
            token.clear();
        }
    }
    assert_eq!(ints.len() % 2, 0, "q/r pairs");
    ints.chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

fn replay(id: &str, moves: &[(i16, i16)]) -> HexoState {
    let mut state = HexoState::new();
    for &(q, r) in moves {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(q, r),
            },
        )
        .unwrap_or_else(|error| panic!("{id}: illegal replay at ({q},{r}): {error:?}"));
    }
    state
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "win",
        ProofStatus::Loss => "loss",
        ProofStatus::Unknown => "unknown",
    }
}

#[test]
#[ignore = "serialized release-only 248-root Phase-B trajectory battery"]
fn triage_phase_b_battery() {
    let labels_path = std::env::var("TSS_TRIAGE_LABELS")
        .expect("TSS_TRIAGE_LABELS must name lanec_labels.jsonl");
    let positions_path = std::env::var("TSS_TRIAGE_POSITIONS")
        .expect("TSS_TRIAGE_POSITIONS must name selfplay_positions.jsonl");
    let trajectory_path = std::env::var("TSS_TRACE_TRAJECTORY")
        .expect("TSS_TRACE_TRAJECTORY must name the output JSONL");
    let results_path = std::env::var("TSS_TRIAGE_RESULTS")
        .expect("TSS_TRIAGE_RESULTS must name the result JSONL");

    let mut labels = HashMap::<String, String>::new();
    for line in std::fs::read_to_string(&labels_path)
        .unwrap_or_else(|error| panic!("read {labels_path}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        if json_string(line, "source").as_deref() != Some("grind") {
            continue;
        }
        let id = json_string(line, "pos_id").expect("label pos_id");
        let class = match json_string(line, "status").as_deref() {
            Some("win") => "provable",
            Some("unknown") if json_bool(line, "tt_saturation_suspect") == Some(true) => {
                "cap_bound"
            }
            Some("unknown") => "exhaust",
            other => panic!("{id}: unsupported label status {other:?}"),
        };
        assert!(labels.insert(id, class.to_owned()).is_none(), "duplicate label");
    }
    let counts = labels.values().fold(HashMap::<&str, usize>::new(), |mut out, class| {
        *out.entry(class.as_str()).or_default() += 1;
        out
    });
    assert_eq!(labels.len(), 248, "frozen grind cardinality");
    assert_eq!(counts.get("provable"), Some(&57));
    assert_eq!(counts.get("exhaust"), Some(&97));
    assert_eq!(counts.get("cap_bound"), Some(&94));

    let wanted = labels.keys().cloned().collect::<HashSet<_>>();
    let mut states = HashMap::<String, HexoState>::new();
    for line in std::fs::read_to_string(&positions_path)
        .unwrap_or_else(|error| panic!("read {positions_path}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let id = json_string(line, "id").expect("position id");
        if wanted.contains(&id) {
            states.insert(id.clone(), replay(&id, &json_moves(line)));
        }
    }
    assert_eq!(states.len(), labels.len(), "every label has a replay state");

    let mut ids = labels.keys().cloned().collect::<Vec<_>>();
    ids.sort();
    let caps = SolveCaps {
        node_cap: 5_000,
        tt_bytes_cap: 256 << 10,
        semantic_horizon: u32::MAX,
    };
    let mut results = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&results_path)
            .unwrap_or_else(|error| panic!("open {results_path}: {error}")),
    );
    let started_all = Instant::now();
    let mut total_nodes = 0u64;
    for (index, id) in ids.iter().enumerate() {
        std::env::set_var("TSS_TRACE_SOLVE_ID", id);
        let state = states.get(id).expect("loaded state");
        let mut solver = TssSolver::default();
        solver.configure_leaf_profile();
        let started = Instant::now();
        let result = solver.solve_goal(state, &caps, SolveGoal::Win);
        let wall_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let verified = result
            .cert
            .as_ref()
            .is_some_and(|cert| TssVerifier.verify(state, cert, result.status));
        assert_eq!(
            result.status != ProofStatus::Unknown,
            verified,
            "{id}: every verdict has exactly one verified certificate"
        );
        assert!(result.stats.nodes <= caps.node_cap, "{id}: node cap");
        assert!(
            result.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64,
            "{id}: TT cap"
        );
        total_nodes = total_nodes.saturating_add(result.stats.nodes);
        let id_json = serde_json::to_string(id).expect("serialize id");
        let class_json = serde_json::to_string(labels.get(id).expect("class")).expect("class");
        writeln!(
            results,
            "{{\"solve_id\":{id_json},\"class\":{class_json},\"status\":\"{}\",\"verified\":{verified},\"nodes\":{},\"expansions\":{},\"tt_entries\":{},\"peak_tt_bytes\":{},\"wall_nanos\":{wall_nanos}}}",
            status_name(result.status),
            result.stats.nodes,
            result.stats.expansions,
            result.stats.tt_entries,
            result.stats.peak_tt_bytes,
        )
        .expect("write result row");
        results.flush().expect("flush result row");
        println!(
            "TRIAGE_B_ROW index={}/{} id={id} class={} status={} nodes={} expansions={} wall_ms={:.3}",
            index + 1,
            ids.len(),
            labels[id],
            status_name(result.status),
            result.stats.nodes,
            result.stats.expansions,
            wall_nanos as f64 / 1e6,
        );
    }
    std::env::remove_var("TSS_TRACE_SOLVE_ID");
    println!(
        "TRIAGE_B_DONE positions={} total_nodes={total_nodes} wall_s={:.3} trajectory={} results={}",
        ids.len(),
        started_all.elapsed().as_secs_f64(),
        trajectory_path,
        results_path,
    );
}
