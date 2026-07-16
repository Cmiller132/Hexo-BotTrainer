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
//! Run with:
//! `cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --nocapture`

use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{round3_shadow_certificate, TssSolver, WidthOptions};
use crate::tss_verify::TssVerifier;

const DEFAULT_TSS_TEST_TT_BYTES: usize = 512 << 20;

/// Test-harness resource override shared by both ignored corpus helpers.
/// Production callers and the default 512 MiB test profile are unchanged.
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

struct CorpusPosition {
    id: String,
    expect_win: bool,
    state: HexoState,
}

struct ForcingLine {
    id: String,
    moves: Vec<HexCoord>,
}

fn load_corpus() -> Vec<CorpusPosition> {
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
    for pos in &corpus {
        if selected_ids
            .as_ref()
            .is_some_and(|ids| !ids.iter().any(|id| id == &pos.id))
        {
            continue;
        }
        selected += 1;
        let mut final_status = ProofStatus::Unknown;
        for (i, cap) in ladder.iter().enumerate() {
            if !pos.expect_win && *cap > 1_000_000 {
                break;
            }
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            let caps = SolveCaps {
                node_cap: *cap,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            };
            let t0 = Instant::now();
            let result = solver.solve(&pos.state, &caps);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "CORPUS id={} cap={cap} status={} expect={} nodes={} tt_hits={} tt_bytes_cap={} peak_tt_bytes={} ms={ms:.1}",
                pos.id,
                status_name(result.status),
                if pos.expect_win { "WIN" } else { "NO" },
                result.stats.nodes,
                result.stats.tt_hits,
                tt_bytes_cap,
                result.stats.peak_tt_bytes,
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
    println!("CORPUS_DONE failures={}", failures.len());
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
