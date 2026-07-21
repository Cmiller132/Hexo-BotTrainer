//! Test-only G2 consume step-zero measurement.
//!
//! The decisive numerator is deliberately conservative: if a root reaches an
//! exact eligible unforced defender site, or any unforced site whose negative
//! classification is not kill-grade certified, the root enters `E` and its
//! entire solver-node total is charged. Forced FHW sites are counted
//! separately. Nothing in this module is compiled into a non-test build.

use std::cell::RefCell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StepZeroObservation {
    pub exact_eligible_unforced: u64,
    pub indeterminate_unforced: u64,
    pub forced_fhw: u64,
}

impl StepZeroObservation {
    fn root_in_e(self) -> bool {
        self.exact_eligible_unforced != 0 || self.indeterminate_unforced != 0
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<StepZeroObservation>> = const { RefCell::new(None) };
}

pub(crate) fn begin_observation() {
    ACTIVE.with(|slot| {
        let previous = slot.replace(Some(StepZeroObservation::default()));
        assert!(previous.is_none(), "step-zero observation cannot nest");
    });
}

/// Record the exact future unforced hook. A positive from the existing frozen
/// producer-side predicate is useful telemetry. A negative is intentionally
/// forced indeterminate: the current shared-helper classifier has not met R4's
/// independent no-false-negative certification contract.
pub(crate) fn observe_unforced(producer_eligible: bool) {
    ACTIVE.with(|slot| {
        if let Some(observation) = slot.borrow_mut().as_mut() {
            if producer_eligible {
                observation.exact_eligible_unforced =
                    observation.exact_eligible_unforced.saturating_add(1);
            } else {
                observation.indeterminate_unforced =
                    observation.indeterminate_unforced.saturating_add(1);
            }
        }
    });
}

pub(crate) fn observe_forced() {
    ACTIVE.with(|slot| {
        if let Some(observation) = slot.borrow_mut().as_mut() {
            observation.forced_fhw = observation.forced_fhw.saturating_add(1);
        }
    });
}

pub(crate) fn finish_observation() -> StepZeroObservation {
    ACTIVE.with(|slot| {
        slot.replace(None)
            .expect("step-zero observation must be started")
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::time::Instant;

    use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use super::{begin_observation, finish_observation};
    use crate::tss_core::{ProofStatus, SolveCaps, SolveGoal};
    use crate::tss_solver::TssSolver;

    const SELFPLAY: &str = include_str!("../../../../scripts/tss_harness/sets/selfplay_v1.jsonl");
    const HUMAN: &str = include_str!("../../../../scripts/tss_harness/sets/human_v1.jsonl");
    const PUZZLE: &str = include_str!("../../../../scripts/tss_harness/sets/puzzle_v3.jsonl");
    const LANE_C: &str = include_str!("../../../../raws/lanec_labels.jsonl");
    const FORCING: &str = include_str!("../corpus/forcing_corpus_moves.txt");

    #[derive(Clone, Debug, Deserialize)]
    struct FrozenRow {
        pos_id: String,
        source: String,
        moves: Vec<[i16; 2]>,
        #[serde(default)]
        meta: Value,
    }

    #[derive(Clone, Debug, Deserialize)]
    struct LaneCRow {
        pos_id: String,
        source: String,
    }

    #[derive(Clone, Debug)]
    struct WorkItem {
        dataset: &'static str,
        pos_id: String,
        source: String,
        cluster: String,
        moves: Vec<[i16; 2]>,
    }

    #[derive(Debug, Serialize)]
    struct RawRecord<'a> {
        profile: &'a str,
        dataset: &'a str,
        pos_id: &'a str,
        source: &'a str,
        cluster: &'a str,
        cap: u64,
        tt_bytes: usize,
        nodes: u64,
        status: &'a str,
        wall_nanos: u64,
        exact_eligible_unforced_occurrences: u64,
        indeterminate_unforced_occurrences: u64,
        forced_fhw_occurrences: u64,
        root_in_e: bool,
    }

    fn parse_rows(text: &str) -> Vec<FrozenRow> {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("valid frozen JSONL row"))
            .collect()
    }

    fn cluster_for(row: &FrozenRow) -> String {
        if let Some(game) = row.meta.get("game").and_then(Value::as_u64) {
            return format!("selfplay_game_{game}");
        }
        if let Some((prefix, _)) = row.pos_id.rsplit_once("_p") {
            if row.pos_id.starts_with("human_") {
                return prefix.to_owned();
            }
            if row.pos_id.starts_with("sp_") {
                let mut fields = row.pos_id.split('_');
                if let (Some("sp"), Some(game)) = (fields.next(), fields.next()) {
                    return format!("selfplay_game_{game}");
                }
            }
        }
        // The frozen puzzle rows do not carry an atlas-family field. Keep each
        // named puzzle root separate rather than inventing a family relation.
        format!("{}:{}", row.source, row.pos_id)
    }

    fn forcing_items() -> Vec<WorkItem> {
        let mut items = Vec::new();
        let mut lines = FORCING.lines();
        while let Some(header) = lines.next() {
            let header = header.trim();
            if header.is_empty() {
                continue;
            }
            assert!(header.starts_with("POS "), "bad forcing header: {header}");
            let mut id = None;
            let mut nstones = None;
            for field in header.split_whitespace().skip(1) {
                let (key, value) = field.split_once('=').expect("forcing k=v field");
                match key {
                    "id" => id = Some(value.to_owned()),
                    "nstones" => nstones = Some(value.parse::<usize>().expect("nstones")),
                    _ => {}
                }
            }
            let id = id.expect("forcing id");
            let moves = (0..nstones.expect("forcing nstones"))
                .map(|_| {
                    let mut fields = lines.next().expect("forcing move").split_whitespace();
                    let q = fields.next().expect("q").parse().expect("numeric q");
                    let r = fields.next().expect("r").parse().expect("numeric r");
                    assert!(fields.next().is_none(), "extra forcing move field");
                    [q, r]
                })
                .collect();
            assert_eq!(lines.next().map(str::trim), Some("END"));
            items.push(WorkItem {
                dataset: "forcing_corpus",
                pos_id: format!("forcing_{id}"),
                source: "forcing".to_owned(),
                cluster: format!("forcing:{id}"),
                moves,
            });
        }
        items
    }

    fn labeling_items() -> Vec<WorkItem> {
        let mut items = Vec::new();
        for (dataset, text) in [
            ("selfplay_v1", SELFPLAY),
            ("human_v1", HUMAN),
            ("puzzle_v3", PUZZLE),
        ] {
            items.extend(parse_rows(text).into_iter().map(|row| WorkItem {
                dataset,
                cluster: cluster_for(&row),
                pos_id: row.pos_id,
                source: row.source,
                moves: row.moves,
            }));
        }
        items.extend(forcing_items());
        assert_eq!(items.len(), 6_462, "frozen Labeling-2k frame");
        items
    }

    fn grind_items() -> Vec<WorkItem> {
        let grind_ids: HashSet<String> = LANE_C
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<LaneCRow>(line).expect("valid lane-C JSONL"))
            .filter(|row| row.source == "grind")
            .map(|row| row.pos_id)
            .collect();
        assert_eq!(grind_ids.len(), 248, "frozen lane-C grind ids");
        let items: Vec<_> = parse_rows(SELFPLAY)
            .into_iter()
            .filter(|row| grind_ids.contains(&row.pos_id))
            .map(|row| WorkItem {
                dataset: "grind_248",
                cluster: cluster_for(&row),
                pos_id: row.pos_id,
                source: "grind".to_owned(),
                moves: row.moves,
            })
            .collect();
        assert_eq!(items.len(), 248, "all grind ids resolve in selfplay_v1");
        items
    }

    fn replay(item: &WorkItem) -> HexoState {
        let mut state = HexoState::new();
        for &[q, r] in &item.moves {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap_or_else(|error| {
                panic!("{} illegal replay at ({q},{r}): {error:?}", item.pos_id)
            });
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
    fn uncertified_negative_is_forced_indeterminate() {
        begin_observation();
        super::observe_unforced(false);
        let observation = finish_observation();
        assert_eq!(observation.exact_eligible_unforced, 0);
        assert_eq!(observation.indeterminate_unforced, 1);
        assert_eq!(observation.forced_fhw, 0);
        assert!(observation.root_in_e());
    }

    #[test]
    fn forced_fhw_is_reported_separately_from_unforced_e() {
        begin_observation();
        super::observe_forced();
        let observation = finish_observation();
        assert_eq!(observation.exact_eligible_unforced, 0);
        assert_eq!(observation.indeterminate_unforced, 0);
        assert_eq!(observation.forced_fhw, 1);
        assert!(!observation.root_in_e());
    }

    #[test]
    fn frozen_measurement_frames_resolve_to_preregistered_sizes() {
        assert_eq!(labeling_items().len(), 6_462);
        assert_eq!(grind_items().len(), 248);
    }

    /// Run with `G2_STEP_ZERO_PROFILE=labeling-2k|atlas-50k` and
    /// `G2_STEP_ZERO_LOG=<path>`. Ignored so ordinary test suites never pay for
    /// the measurement.
    #[test]
    #[ignore = "owner-authorized fixed step-zero measurement"]
    fn g2_step_zero_measurement() {
        assert!(cfg!(all(target_arch = "x86_64", target_os = "windows")));
        let target_dir = std::env::var("CARGO_TARGET_DIR").expect("CARGO_TARGET_DIR is required");
        assert!(
            target_dir.replace('\\', "/").ends_with("/.target-g2c"),
            "measurement must use the lane target dir: {target_dir}"
        );
        let profile = std::env::var("G2_STEP_ZERO_PROFILE").expect("measurement profile");
        let output = std::env::var("G2_STEP_ZERO_LOG").expect("measurement log path");
        let (items, cap, tt_bytes) = match profile.as_str() {
            "labeling-2k" => (labeling_items(), 2_000, 256 << 10),
            "atlas-50k" => (grind_items(), 50_000, 256 << 20),
            _ => panic!("unknown step-zero profile: {profile}"),
        };

        let mut writer = BufWriter::new(File::create(&output).expect("create raw JSONL"));
        let mut total_nodes = 0u64;
        let mut e_nodes = 0u64;
        let mut e_roots = 0u64;
        let mut exact_sites = 0u64;
        let mut indeterminate_sites = 0u64;
        let mut forced_sites = 0u64;

        for item in &items {
            let state = replay(item);
            let mut solver = TssSolver::default();
            solver.configure_leaf_profile();
            solver.set_dual_pass(true);
            solver.set_loss_reserve_nodes(0);
            solver.set_group2(false);
            let caps = SolveCaps {
                node_cap: cap,
                tt_bytes_cap: tt_bytes,
                semantic_horizon: u32::MAX,
            };
            begin_observation();
            let started = Instant::now();
            let result = solver.solve_goal(&state, &caps, SolveGoal::Both);
            let wall_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let observation = finish_observation();
            let root_in_e = observation.root_in_e();
            total_nodes = total_nodes.saturating_add(result.stats.nodes);
            if root_in_e {
                e_roots = e_roots.saturating_add(1);
                e_nodes = e_nodes.saturating_add(result.stats.nodes);
            }
            exact_sites = exact_sites.saturating_add(observation.exact_eligible_unforced);
            indeterminate_sites =
                indeterminate_sites.saturating_add(observation.indeterminate_unforced);
            forced_sites = forced_sites.saturating_add(observation.forced_fhw);
            serde_json::to_writer(
                &mut writer,
                &RawRecord {
                    profile: &profile,
                    dataset: item.dataset,
                    pos_id: &item.pos_id,
                    source: &item.source,
                    cluster: &item.cluster,
                    cap,
                    tt_bytes,
                    nodes: result.stats.nodes,
                    status: status_name(result.status),
                    wall_nanos,
                    exact_eligible_unforced_occurrences: observation.exact_eligible_unforced,
                    indeterminate_unforced_occurrences: observation.indeterminate_unforced,
                    forced_fhw_occurrences: observation.forced_fhw,
                    root_in_e,
                },
            )
            .expect("write raw record");
            writer.write_all(b"\n").expect("write JSONL newline");
        }
        writer.flush().expect("flush raw JSONL");
        let u_nodes = if total_nodes == 0 {
            0.0
        } else {
            e_nodes as f64 / total_nodes as f64
        };
        eprintln!(
            "G2_STEP_ZERO_SUMMARY profile={profile} roots={} e_roots={e_roots} total_nodes={total_nodes} e_nodes={e_nodes} u_nodes={u_nodes:.9} exact_sites={exact_sites} indeterminate_sites={indeterminate_sites} forced_sites={forced_sites}",
            items.len()
        );
    }
}
