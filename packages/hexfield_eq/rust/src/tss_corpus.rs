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

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player};

use crate::tss_core::{DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{TssSolver, WidthOptions};

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

fn load_corpus() -> Vec<CorpusPosition> {
    let path = std::env::var("TSS_CORPUS_FILE").unwrap_or_else(|_| {
        format!(
            "{}/corpus/forcing_corpus_moves.txt",
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
            apply_placement(&mut state, Placement { coord: HexCoord { q, r } })
                .unwrap_or_else(|e| panic!("{id}: illegal replay at ({q},{r}): {e:?}"));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"), "{id}: missing END");
        assert!(!state.is_terminal(), "{id}: replay reached a terminal state");
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

#[test]
#[ignore = "acceptance gate; run explicitly in --release"]
fn tss_corpus_check() {
    let corpus = load_corpus();
    // WIN entries climb the full ladder; NO entries stop at 1M (they only
    // must never come back WIN).
    let ladder: [u64; 4] = [10_000, 100_000, 1_000_000, 20_000_000];

    let mut failures: Vec<String> = Vec::new();
    for pos in &corpus {
        let mut final_status = ProofStatus::Unknown;
        for (i, cap) in ladder.iter().enumerate() {
            if !pos.expect_win && *cap > 1_000_000 {
                break;
            }
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            let caps = SolveCaps {
                node_cap: *cap,
                tt_bytes_cap: 512 << 20,
                semantic_horizon: u32::MAX,
            };
            let t0 = Instant::now();
            let result = solver.solve(&pos.state, &caps);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "CORPUS id={} cap={cap} status={} expect={} nodes={} tt_hits={} ms={ms:.1}",
                pos.id,
                status_name(result.status),
                if pos.expect_win { "WIN" } else { "NO" },
                result.stats.nodes,
                result.stats.tt_hits,
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
    println!("CORPUS_DONE failures={}", failures.len());
    assert!(
        failures.is_empty(),
        "corpus acceptance failures:\n{}",
        failures.join("\n")
    );
}
