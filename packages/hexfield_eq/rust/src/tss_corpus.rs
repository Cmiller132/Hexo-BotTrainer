//! VCF-width acceptance gate: the external forcing corpus must fully solve.
//!
//! Corpus: `rust/corpus/forcing_corpus_moves.txt` (19 positions, replay-
//! ordered; from the owner's hexo-solver idtt/dfpn/pdspn race corpus).
//! Expected: 14 WIN (attacker has a forced win), 5 NO (no forced win —
//! LOSS or UNKNOWN both acceptable; WIN on a NO entry is a soundness
//! failure and fails the test immediately).
//!
//! This test drives the WIDE (pair-complete) attacker universe via
//! `TssSolver::set_width_options` — the API this branch exists to build.
//! Narrow-mode behavior must remain byte-identical (existing tests + bench).
//!
//! The canonical 512 MiB default, 2 GiB official profile, and fixed node
//! ladder are documented in `docs/TSS_RUNBOOK.md`. Run the official gate with
//! `TSS_BACKWALK_TT_BYTES=2147483648` and:
//! `cargo test --release -p hexfield_eq tss_corpus_check -- --ignored
//! --test-threads=1 --nocapture`

use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::{
    CertVerify, ClosureDebtStats, DeepSolve, ProofStatus, SolveCaps, SolveGoal, ThresholdScaleStats,
};
use crate::tss_solver::{
    round3_shadow_certificate, CapResumeError, CapResumeSession, TssSolver, WidthOptions,
};
use crate::tss_verify::TssVerifier;

const DEFAULT_TSS_TEST_TT_BYTES: usize = 512 << 20;

/// Test-harness resource override shared by both ignored corpus helpers. See
/// `docs/TSS_RUNBOOK.md` for the single profile story and official gate.
fn test_tt_bytes_cap() -> usize {
    std::env::var("TSS_BACKWALK_TT_BYTES")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("numeric TSS_BACKWALK_TT_BYTES")
        })
        .unwrap_or(DEFAULT_TSS_TEST_TT_BYTES)
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

pub(crate) struct CorpusPosition {
    pub(crate) id: String,
    pub(crate) expect_win: bool,
    pub(crate) state: HexoState,
}

struct ForcingLine {
    id: String,
    moves: Vec<HexCoord>,
}

pub(crate) fn load_corpus() -> Vec<CorpusPosition> {
    let path = std::env::var("TSS_CORPUS_FILE").unwrap_or_else(|_| {
        format!(
            "{}/rust/corpus/forcing_corpus_moves.txt",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let text = std::fs::read_to_string(&path).expect("read corpus file");
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        assert!(header.starts_with("POS "), "bad header: {header}");
        let mut id = String::new();
        let mut attacker = 0usize;
        let mut expect = String::new();
        let mut nstones = 0usize;
        for tok in header.split_whitespace().skip(1) {
            let (k, v) = tok.split_once('=').expect("k=v token");
            match k {
                "id" => id = v.to_string(),
                "attacker" => attacker = v.parse().unwrap(),
                "expect" => expect = v.to_string(),
                "nstones" => nstones = v.parse().unwrap(),
                _ => {}
            }
        }
        let mut state = HexoState::new();
        for _ in 0..nstones {
            let line = lines.next().expect("stone line");
            let mut it = line.split_whitespace();
            let q: i16 = it.next().unwrap().parse().unwrap();
            let r: i16 = it.next().unwrap().parse().unwrap();
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord { q, r },
                },
            )
            .unwrap_or_else(|e| panic!("{id}: illegal replay at ({q},{r}): {e:?}"));
        }
        assert_eq!(
            lines.next().map(str::trim),
            Some("END"),
            "{id}: missing END"
        );
        assert!(
            !state.is_terminal(),
            "{id}: replay reached a terminal state"
        );
        let expected_player = if attacker == 0 {
            Player::Player0
        } else {
            Player::Player1
        };
        assert_eq!(
            state.current_player(),
            expected_player,
            "{id}: side-to-move mismatch after replay"
        );
        out.push(CorpusPosition {
            id,
            expect_win: expect == "WIN",
            state,
        });
    }
    assert_eq!(out.len(), 19, "expected all 19 corpus positions");
    out
}

fn load_forcing_lines() -> Vec<ForcingLine> {
    let path = format!(
        "{}/rust/corpus/forcing_corpus_lines.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("read forcing lines file");
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        assert!(header.starts_with("LINE "), "bad line header: {header}");
        let mut id = String::new();
        let mut nmoves = 0usize;
        for tok in header.split_whitespace().skip(1) {
            let (key, value) = tok.split_once('=').expect("line k=v token");
            match key {
                "id" => id = value.to_string(),
                "nmoves" => nmoves = value.parse().expect("numeric nmoves"),
                _ => {}
            }
        }
        let mut moves = Vec::with_capacity(nmoves);
        for _ in 0..nmoves {
            let line = lines.next().expect("forcing-line move");
            let mut fields = line.split_whitespace();
            let q = fields.next().expect("move q").parse().expect("numeric q");
            let r = fields.next().expect("move r").parse().expect("numeric r");
            assert!(fields.next().is_none(), "extra forcing-line move field");
            moves.push(HexCoord { q, r });
        }
        assert_eq!(
            lines.next().map(str::trim),
            Some("END"),
            "{id}: missing END"
        );
        out.push(ForcingLine { id, moves });
    }
    assert_eq!(out.len(), 14, "expected all 14 WIN reference lines");
    out
}

#[test]
#[ignore = "acceptance gate; run explicitly in --release"]
fn tss_corpus_check() {
    let corpus = load_corpus();
    let tt_bytes_cap = test_tt_bytes_cap();
    let shared_fragments = std::env::var("TSS_SHARED_FRAGMENTS").ok().as_deref() == Some("1");
    let lazy_frontier = std::env::var("TSS_LAZY_FRONTIER").ok().as_deref() == Some("1");
    let interior_gate = std::env::var("TSS_INTERIOR_CENSUS_GATE").ok().as_deref() == Some("1");
    let k_reply_consume = std::env::var("TSS_K_REPLY_CONSUME").ok().as_deref() == Some("1");
    let cap_resume = std::env::var("TSS_CAP_RESUME").ok().as_deref() == Some("1");
    let live_ge3_seed = std::env::var("TSS_LIVE_GE3_SEED").ok().as_deref() == Some("1");
    let closure_counters = std::env::var("TSS_CLOSURE_COUNTERS").ok().as_deref() == Some("1");
    let threshold_counters = std::env::var("TSS_THRESHOLD_COUNTERS").ok().as_deref() == Some("1");
    let threshold_delta = std::env::var("TSS_THRESHOLD_DELTA").unwrap_or_else(|_| "off".to_owned());
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_SHARED_FRAGMENTS") {
        assert_eq!(
            expected,
            if shared_fragments { "1" } else { "0" },
            "TSS_SHARED_FRAGMENTS does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_LAZY_FRONTIER") {
        assert_eq!(
            expected,
            if lazy_frontier { "1" } else { "0" },
            "TSS_LAZY_FRONTIER does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE") {
        assert_eq!(
            expected,
            if interior_gate { "1" } else { "0" },
            "TSS_INTERIOR_CENSUS_GATE does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_K_REPLY_CONSUME") {
        assert_eq!(
            expected,
            if k_reply_consume { "1" } else { "0" },
            "TSS_K_REPLY_CONSUME does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_CAP_RESUME") {
        assert_eq!(
            expected,
            if cap_resume { "1" } else { "0" },
            "TSS_CAP_RESUME does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_LIVE_GE3_SEED") {
        assert_eq!(
            expected,
            if live_ge3_seed { "1" } else { "0" },
            "TSS_LIVE_GE3_SEED does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_CLOSURE_COUNTERS") {
        assert_eq!(
            expected,
            if closure_counters { "1" } else { "0" },
            "TSS_CLOSURE_COUNTERS does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_THRESHOLD_COUNTERS") {
        assert_eq!(
            expected,
            if threshold_counters { "1" } else { "0" },
            "TSS_THRESHOLD_COUNTERS does not match gate expectation",
        );
    }
    if let Ok(expected) = std::env::var("TSS_CORPUS_EXPECT_THRESHOLD_DELTA") {
        assert_eq!(
            expected, threshold_delta,
            "TSS_THRESHOLD_DELTA does not match gate expectation",
        );
    }
    println!(
        "CORPUS_MODE shared_fragments={} lazy_frontier={} interior_gate={} k_reply_consume={} cap_resume={} live_ge3_seed={} closure_counters={} threshold_counters={} threshold_delta={} tt_bytes_cap={tt_bytes_cap}",
        if shared_fragments { "on" } else { "off" },
        if lazy_frontier { "on" } else { "off" },
        if interior_gate { "on" } else { "off" },
        if k_reply_consume { "on" } else { "off" },
        if cap_resume { "on" } else { "off" },
        if live_ge3_seed { "on" } else { "off" },
        if closure_counters { "on" } else { "off" },
        if threshold_counters { "on" } else { "off" },
        threshold_delta,
    );
    let selected_ids = std::env::var("TSS_CORPUS_ID").ok().map(|value| {
        let mut ids = value
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        ids.sort();
        ids.dedup();
        assert!(!ids.is_empty(), "TSS_CORPUS_ID must name a corpus entry");
        ids
    });
    // WIN entries climb the full ladder; NO entries stop at 1M (they only
    // must never come back WIN).
    // TSS_CORPUS_MAX_CAP is a debugging guard for addendum-compliant <=100k
    // iteration.  It is unset in the acceptance gate, preserving its ladder.
    let max_cap = std::env::var("TSS_CORPUS_MAX_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric TSS_CORPUS_MAX_CAP"))
        .unwrap_or(u64::MAX);
    let ladder = [10_000, 100_000, 1_000_000, 20_000_000]
        .into_iter()
        .filter(|cap| *cap <= max_cap)
        .collect::<Vec<_>>();
    assert!(
        !ladder.is_empty(),
        "TSS_CORPUS_MAX_CAP must be at least 10000"
    );

    let mut failures: Vec<String> = Vec::new();
    let mut selected = 0usize;
    let mut fragment_lookups = 0u64;
    let mut fragment_hits = 0u64;
    let mut fragment_imports = 0u64;
    let mut max_fragment_store_entries = 0u64;
    let mut max_fragment_store_bytes = 0u64;
    let mut resume_total_ms = 0.0f64;
    let mut resume_total_reentries = 0u64;
    let mut closure_total = ClosureDebtStats::default();
    let mut threshold_total = ThresholdScaleStats::default();
    let mut stage_refresh_total = 0u64;
    let mut live_ge3_seed_scans = 0u64;
    let mut live_ge3_seed_nanos = 0u64;
    for pos in &corpus {
        if selected_ids
            .as_ref()
            .is_some_and(|ids| !ids.iter().any(|id| id == &pos.id))
        {
            continue;
        }
        selected += 1;
        let mut final_status = ProofStatus::Unknown;
        let resume_solver = cap_resume.then(|| {
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            solver
        });
        let mut resume_session = None::<CapResumeSession>;
        let mut resume_unsupported = false;
        let mut resume_root_ms = 0.0f64;
        let mut resume_root_reentries = 0u64;
        for (i, cap) in ladder.iter().enumerate() {
            if !pos.expect_win && *cap > 1_000_000 {
                break;
            }
            let caps = SolveCaps {
                node_cap: *cap,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            };
            let t0 = Instant::now();
            let (result, resume_meta) = if let Some(solver) = resume_solver.as_ref() {
                if resume_session.is_none() && !resume_unsupported {
                    match CapResumeSession::new(solver, &pos.state, &caps, SolveGoal::Both) {
                        Ok(session) => resume_session = Some(session),
                        Err(CapResumeError::UnsupportedProfile) => {
                            resume_unsupported = true;
                            println!(
                                "CAP_RESUME_FALLBACK id={} cap={cap} reason=no_unfinished_wide_frontier",
                                pos.id
                            );
                        }
                        Err(error) => {
                            panic!("{}: cap-resume session creation: {error:?}", pos.id)
                        }
                    }
                }
                if let Some(session) = resume_session.as_mut() {
                    let advance = session
                        .advance_to_node_cap(solver, &pos.state, &caps, SolveGoal::Both)
                        .unwrap_or_else(|error| {
                            panic!("{} cap={cap}: cap-resume advance: {error:?}", pos.id)
                        });
                    let meta = (
                        advance.root_pn,
                        advance.root_dn,
                        advance.stage_depth,
                        advance.advances,
                        advance.reentries,
                    );
                    (advance.result, Some(meta))
                } else {
                    let mut fresh_solver = TssSolver::default();
                    fresh_solver.set_width_options(WidthOptions::vcf_pair_complete());
                    (fresh_solver.solve(&pos.state, &caps), None)
                }
            } else {
                let mut solver = TssSolver::default();
                solver.set_width_options(WidthOptions::vcf_pair_complete());
                (solver.solve(&pos.state, &caps), None)
            };
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if cap_resume {
                resume_root_ms += ms;
                resume_total_ms += ms;
            }
            assert!(
                result.status == ProofStatus::Unknown || result.cert.is_some(),
                "{}: hard {} verdict without certificate at cap={cap}",
                pos.id,
                status_name(result.status),
            );
            if let Some(cert) = &result.cert {
                assert!(
                    TssVerifier.verify(&pos.state, cert, result.status),
                    "{}: strict verifier rejected returned {} certificate at cap={cap}",
                    pos.id,
                    status_name(result.status),
                );
            }
            fragment_lookups = fragment_lookups.saturating_add(result.stats.fragment_lookups);
            fragment_hits = fragment_hits.saturating_add(result.stats.fragment_hits);
            fragment_imports = fragment_imports.saturating_add(result.stats.fragment_imports);
            max_fragment_store_entries =
                max_fragment_store_entries.max(result.stats.fragment_store_entries);
            max_fragment_store_bytes =
                max_fragment_store_bytes.max(result.stats.fragment_store_bytes);
            println!(
                "CORPUS id={} cap={cap} status={} expect={} nodes={} expansions={} tt_entries={} tt_hits={} tt_bytes_cap={} peak_tt_bytes={} stage_refreshes={} gate_evals={} gate_dismissals={} gate_us={:.3} seed_scans={} seed_ms={:.3} ms={ms:.1}",
                pos.id,
                status_name(result.status),
                if pos.expect_win { "WIN" } else { "NO" },
                result.stats.nodes,
                result.stats.expansions,
                result.stats.tt_entries,
                result.stats.tt_hits,
                tt_bytes_cap,
                result.stats.peak_tt_bytes,
                result.stats.stage_refreshes,
                result.stats.interior_gate_evaluations,
                result.stats.interior_gate_dismissals,
                result.stats.interior_gate_nanos as f64 / 1_000.0,
                result.stats.live_ge3_seed_scans,
                result.stats.live_ge3_seed_nanos as f64 / 1_000_000.0,
            );
            stage_refresh_total = stage_refresh_total.saturating_add(result.stats.stage_refreshes);
            live_ge3_seed_scans =
                live_ge3_seed_scans.saturating_add(result.stats.live_ge3_seed_scans);
            live_ge3_seed_nanos =
                live_ge3_seed_nanos.saturating_add(result.stats.live_ge3_seed_nanos);
            closure_total.merge(result.stats.closure_debt);
            threshold_total.merge(result.stats.threshold_scale);
            let closure = result.stats.closure_debt;
            println!(
                "CLOSURE_ROW id={} cap={cap} evaluated={} accepted={} retained={} selected={} linked={} expanded={} winning_choices={} winning_rank_bins={:?} reveal_evaluated={} reveal_prefix={} pair_ms={:.3} gate_ms={:.3} second_ms={:.3} eval_ms={:.3} dedup_ms={:.3} avoid_second_ms={:.3} avoid_eval_ms={:.3} avoid_dedup_ms={:.3}",
                pos.id,
                closure.pairs_evaluated,
                closure.pairs_accepted,
                closure.pairs_retained,
                closure.pairs_selected,
                closure.pairs_linked,
                closure.pairs_expanded,
                closure.winning_choice_nodes,
                closure.winning_rank_bins,
                closure.reveal_pair_evaluated,
                closure.reveal_pair_prefix,
                closure.pair_generation_nanos as f64 / 1e6,
                closure.gate_build_nanos as f64 / 1e6,
                closure.second_candidate_nanos as f64 / 1e6,
                closure.pair_evaluation_nanos as f64 / 1e6,
                closure.dedup_nanos as f64 / 1e6,
                closure.avoidable_second_candidate_nanos as f64 / 1e6,
                closure.avoidable_pair_evaluation_nanos as f64 / 1e6,
                closure.avoidable_dedup_nanos as f64 / 1e6,
            );
            let threshold = result.stats.threshold_scale;
            println!(
                "THRESHOLD_ROW id={} cap={cap} visits={} revisits={} threshold_crosses={} reselections={} sibling_switches={} residencies={} residency_expansions={} expansion_bins={:?} descent_ms={:.3} state_ms={:.3}",
                pos.id,
                threshold.recursive_node_visits,
                threshold.expanded_node_revisits,
                threshold.threshold_cross_returns,
                threshold.same_parent_reselections,
                threshold.sibling_switches,
                threshold.residencies,
                threshold.residency_expansions,
                threshold.residency_expansion_bins,
                threshold.descent_nanos as f64 / 1e6,
                threshold.state_apply_undo_nanos as f64 / 1e6,
            );
            if let Some((pn, dn, stage_depth, advances, reentries)) = resume_meta {
                println!(
                    "CAP_RESUME_PROFILE id={} cap={cap} pn={pn} dn={dn} status={} cumulative_expansions={} incremental_ms={ms:.3} root_cumulative_ms={resume_root_ms:.3} stage_depth={stage_depth} advances={advances} reentries={reentries}",
                    pos.id,
                    status_name(result.status),
                    result.stats.expansions,
                );
                resume_root_reentries = reentries;
            }
            println!(
                "FRAGMENT_PROFILE id={} cap={cap} lookups={} hits={} imports={} store_entries={} store_bytes={}",
                pos.id,
                result.stats.fragment_lookups,
                result.stats.fragment_hits,
                result.stats.fragment_imports,
                result.stats.fragment_store_entries,
                result.stats.fragment_store_bytes,
            );
            let (pair_ms, defender_ms, regen_ms, expand_ms, refresh_ms, insert_ms) =
                crate::tss_solver::wide_gen_profile();
            println!(
                "GEN_PROFILE pair_ms={pair_ms} defender_ms={defender_ms} regen_ms={regen_ms} expand_ms={expand_ms} refresh_ms={refresh_ms} insert_ms={insert_ms}"
            );
            final_status = result.status;
            if result.status != ProofStatus::Unknown || i == ladder.len() - 1 {
                break;
            }
        }
        resume_total_reentries = resume_total_reentries.saturating_add(resume_root_reentries);
        if pos.expect_win && final_status != ProofStatus::Win {
            failures.push(format!(
                "{}: expected WIN, got {}",
                pos.id,
                status_name(final_status)
            ));
        }
        if !pos.expect_win && final_status == ProofStatus::Win {
            failures.push(format!("{}: SOUNDNESS: WIN on a NO position", pos.id));
        }
    }
    if let Some(ids) = &selected_ids {
        assert_eq!(
            selected,
            ids.len(),
            "TSS_CORPUS_ID contained an unknown corpus entry"
        );
    }
    let fragment_hit_rate = if fragment_lookups == 0 {
        0.0
    } else {
        fragment_hits as f64 * 100.0 / fragment_lookups as f64
    };
    println!(
        "CORPUS_FRAGMENTS lookups={fragment_lookups} hits={fragment_hits} hit_rate_pct={fragment_hit_rate:.3} imports={fragment_imports} max_store_entries={max_fragment_store_entries} max_store_bytes={max_fragment_store_bytes}"
    );
    println!(
        "CLOSURE_DONE evaluated={} accepted={} retained={} selected={} linked={} expanded={} winning_choices={} winning_rank_bins={:?} reveal_evaluated={} reveal_prefix={} pair_ms={:.3} gate_ms={:.3} second_ms={:.3} eval_ms={:.3} dedup_ms={:.3} avoid_second_ms={:.3} avoid_eval_ms={:.3} avoid_dedup_ms={:.3} stage_refreshes={} seed_scans={} seed_ms={:.3}",
        closure_total.pairs_evaluated,
        closure_total.pairs_accepted,
        closure_total.pairs_retained,
        closure_total.pairs_selected,
        closure_total.pairs_linked,
        closure_total.pairs_expanded,
        closure_total.winning_choice_nodes,
        closure_total.winning_rank_bins,
        closure_total.reveal_pair_evaluated,
        closure_total.reveal_pair_prefix,
        closure_total.pair_generation_nanos as f64 / 1e6,
        closure_total.gate_build_nanos as f64 / 1e6,
        closure_total.second_candidate_nanos as f64 / 1e6,
        closure_total.pair_evaluation_nanos as f64 / 1e6,
        closure_total.dedup_nanos as f64 / 1e6,
        closure_total.avoidable_second_candidate_nanos as f64 / 1e6,
        closure_total.avoidable_pair_evaluation_nanos as f64 / 1e6,
        closure_total.avoidable_dedup_nanos as f64 / 1e6,
        stage_refresh_total,
        live_ge3_seed_scans,
        live_ge3_seed_nanos as f64 / 1e6,
    );
    println!(
        "THRESHOLD_DONE visits={} revisits={} threshold_crosses={} reselections={} sibling_switches={} residencies={} residency_expansions={} expansion_bins={:?} descent_ms={:.3} state_ms={:.3}",
        threshold_total.recursive_node_visits,
        threshold_total.expanded_node_revisits,
        threshold_total.threshold_cross_returns,
        threshold_total.same_parent_reselections,
        threshold_total.sibling_switches,
        threshold_total.residencies,
        threshold_total.residency_expansions,
        threshold_total.residency_expansion_bins,
        threshold_total.descent_nanos as f64 / 1e6,
        threshold_total.state_apply_undo_nanos as f64 / 1e6,
    );
    println!(
        "CORPUS_DONE failures={} shared_fragments={} lazy_frontier={} interior_gate={} k_reply_consume={} cap_resume={} resume_wall_ms={resume_total_ms:.3} resume_reentries={resume_total_reentries}",
        failures.len(),
        if shared_fragments { "on" } else { "off" },
        if lazy_frontier { "on" } else { "off" },
        if interior_gate { "on" } else { "off" },
        if k_reply_consume { "on" } else { "off" },
        if cap_resume { "on" } else { "off" },
    );
    assert!(
        failures.is_empty(),
        "corpus acceptance failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "round-3 all-19 shadow coverage"]
fn tss_round3_shadow_forcing_coverage() {
    let cap = std::env::var("TSS_R3_SHADOW_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric TSS_R3_SHADOW_CAP"))
        .unwrap_or(1_000_000);
    let tt_bytes_cap = test_tt_bytes_cap();
    let corpus = load_corpus();
    assert_eq!(corpus.len(), 19);
    for pos in corpus {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::round3_shadow());
        let result = solver.solve(
            &pos.state,
            &SolveCaps {
                node_cap: cap,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            },
        );
        let mut quiet_turns = 0usize;
        let mut quiet_legal_edges = 0usize;
        let mut zone_nodes = 0usize;
        let mut zone_cells = 0usize;
        let mut legal_cells = 0usize;
        if let Some(cert) = &result.cert {
            assert!(TssVerifier.verify(&pos.state, cert, result.status));
            let report = round3_shadow_certificate(&pos.state, cert)
                .expect("finder shadow replay must accept finder certificate");
            quiet_turns = report.quiet_turns;
            quiet_legal_edges = report.quiet_legal_edges;
            zone_nodes = report.zones.len();
            zone_cells = report.zones.iter().map(|zone| zone.zone.len()).sum();
            legal_cells = report.zones.iter().map(|zone| zone.full_legal).sum();
        }
        println!(
            "R3_SHADOW id={} source=forcing status={} nodes={} quiet_fires={} quiet_legal_edges={} zone_nodes={} zone_cells={} full_legal_cells={} cert={}",
            pos.id,
            status_name(result.status),
            result.stats.nodes,
            quiet_turns,
            quiet_legal_edges,
            zone_nodes,
            zone_cells,
            legal_cells,
            result.cert.is_some(),
        );
        if !pos.expect_win {
            assert_ne!(result.status, ProofStatus::Win, "NO row became WIN");
        }
    }
}

/// Walk a selected reference line backward from the last non-terminal attacker
/// turn.  The first UNKNOWN checkpoint localizes the missing search mechanism.
///
/// Run with `TSS_CORPUS_ID=xsnfyll ... tss_corpus_backward_walk -- --ignored
/// --nocapture`.  The literal full-line state is intentionally skipped because
/// every fixture's final coordinate is terminal and terminal roots are outside
/// the solver API's pre-move contract.
#[test]
#[ignore = "reference-line debugging helper; run explicitly in --release"]
fn tss_corpus_backward_walk() {
    let tt_bytes_cap = test_tt_bytes_cap();
    let selected =
        std::env::var("TSS_CORPUS_ID").expect("set TSS_CORPUS_ID to one of the 14 WIN positions");
    let corpus = load_corpus();
    let lines = load_forcing_lines();
    let position = corpus
        .iter()
        .find(|position| position.id == selected)
        .unwrap_or_else(|| panic!("unknown corpus id: {selected}"));
    assert!(
        position.expect_win,
        "{selected}: NO positions have no WIN line"
    );
    let line = lines
        .iter()
        .find(|line| line.id == selected)
        .unwrap_or_else(|| panic!("missing forcing line for {selected}"));

    let attacker = position.state.current_player();
    let mut replay = position.state.clone();
    let mut prefix_states = vec![(0usize, replay.clone())];
    for (index, &coord) in line.moves.iter().enumerate() {
        apply_placement(&mut replay, Placement { coord })
            .unwrap_or_else(|error| panic!("{selected}: illegal line move {coord:?}: {error:?}"));
        let prefix = index + 1;
        if replay.is_terminal() {
            assert_eq!(prefix, line.moves.len(), "{selected}: line ended early");
            break;
        }
        prefix_states.push((prefix, replay.clone()));
    }

    if let Ok(requested) = std::env::var("TSS_BACKWALK_PREFIX") {
        let requested = requested.parse::<usize>().expect("numeric backwalk prefix");
        let (_, state) = prefix_states
            .iter()
            .find(|(prefix, _)| *prefix == requested)
            .unwrap_or_else(|| panic!("missing nonterminal prefix {requested}"));
        let mut exact_state = state.clone();
        if let Ok(extra) = std::env::var("TSS_BACKWALK_EXTRA") {
            for encoded in extra.split(';').filter(|value| !value.is_empty()) {
                let (q, r) = encoded
                    .split_once(',')
                    .unwrap_or_else(|| panic!("bad TSS_BACKWALK_EXTRA coord: {encoded}"));
                let coord = HexCoord {
                    q: q.parse().expect("numeric extra q"),
                    r: r.parse().expect("numeric extra r"),
                };
                apply_placement(&mut exact_state, Placement { coord }).unwrap_or_else(|error| {
                    panic!("{selected}: illegal extra move {coord:?}: {error:?}")
                });
            }
        }
        let goal = if exact_state.current_player() == attacker {
            SolveGoal::Win
        } else {
            SolveGoal::Loss
        };
        let node_cap = std::env::var("TSS_BACKWALK_CAP")
            .ok()
            .map(|value| value.parse::<u64>().expect("numeric backwalk cap"))
            .unwrap_or(10_000);
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let result = solver.solve_goal(
            &exact_state,
            &SolveCaps {
                node_cap,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            },
            goal,
        );
        println!(
            "BACKWALK_EXACT id={selected} prefix={requested} status={} nodes={} tt_hits={} tt_bytes_cap={} peak_tt_bytes={}",
            status_name(result.status),
            result.stats.nodes,
            result.stats.tt_hits,
            tt_bytes_cap,
            result.stats.peak_tt_bytes,
        );
        let expected = if goal == SolveGoal::Win {
            ProofStatus::Win
        } else {
            ProofStatus::Loss
        };
        assert_eq!(result.status, expected);
        return;
    }

    let checkpoints = prefix_states
        .iter()
        .filter(|(prefix, state)| {
            *prefix == 0
                || (state.current_player() == attacker
                    && matches!(state.phase(), TurnPhase::FirstStone))
        })
        .collect::<Vec<_>>();
    for &&(prefix, ref state) in checkpoints.iter().rev() {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let caps = SolveCaps {
            node_cap: 10_000,
            tt_bytes_cap,
            semantic_horizon: u32::MAX,
        };
        let t0 = Instant::now();
        let result = solver.solve(&state, &caps);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "BACKWALK id={selected} prefix={prefix} status={} nodes={} tt_hits={} tt_bytes_cap={} peak_tt_bytes={} ms={ms:.1}",
            status_name(result.status),
            result.stats.nodes,
            result.stats.tt_hits,
            tt_bytes_cap,
            result.stats.peak_tt_bytes,
        );
        if result.status != ProofStatus::Win {
            // Probe the four individual placements leading to the already-
            // proven next attacker checkpoint.  This distinguishes missing
            // first/second-stone generation from a defender-universal gap.
            for (probe_prefix, probe_state) in prefix_states
                .iter()
                .filter(|(probe_prefix, _)| *probe_prefix > prefix && *probe_prefix <= prefix + 4)
            {
                let goal = if probe_state.current_player() == attacker {
                    SolveGoal::Win
                } else {
                    SolveGoal::Loss
                };
                let expected = if goal == SolveGoal::Win {
                    ProofStatus::Win
                } else {
                    ProofStatus::Loss
                };
                let mut probe_solver = TssSolver::default();
                probe_solver.set_width_options(WidthOptions::vcf_pair_complete());
                let probe = probe_solver.solve_goal(probe_state, &caps, goal);
                println!(
                    "BACKWALK_PROBE id={selected} prefix={probe_prefix} status={} expected={} nodes={} tt_hits={} tt_bytes_cap={} peak_tt_bytes={}",
                    status_name(probe.status),
                    status_name(expected),
                    probe.stats.nodes,
                    probe.stats.tt_hits,
                    tt_bytes_cap,
                    probe.stats.peak_tt_bytes,
                );
            }
            panic!("{selected}: first failing backward checkpoint is prefix {prefix}");
        }
    }
}
