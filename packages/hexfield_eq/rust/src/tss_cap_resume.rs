//! R-CR1 test-only cap-ladder continuation campaign.

use std::time::Instant;

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_corpus::load_corpus;
use crate::tss_residue::{self, ResidueCategory, ResidueJobKey, ResidueJobOutcome};
use crate::tss_solver::{CapResumeError, CapResumeSession, TssSolver, WidthOptions};
use crate::tss_verify::TssVerifier;

const REQUIRED_IDS: [&str; 7] = [
    "0l4291i_live",
    "94gnnol",
    "lz60mfb",
    "mvp2lvc",
    "hayes_20260712_turn16",
    "hayes_20260712_placement31",
    "xsnfyll",
];

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("numeric {name}"))
        })
        .unwrap_or(default)
}

fn selected_ids() -> Vec<String> {
    let selected = std::env::var("TSS_CAP_RESUME_ID")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| REQUIRED_IDS.iter().map(|id| (*id).to_owned()).collect());
    assert!(!selected.is_empty(), "TSS_CAP_RESUME_ID selected no rows");
    selected
}

#[test]
#[ignore = "R-CR1 milestone identity and uninterrupted-final campaign"]
fn tss_cap_resume_campaign() {
    assert_ne!(
        std::env::var("TSS_SHARED_FRAGMENTS").ok().as_deref(),
        Some("1"),
        "R-CR1 excludes shared fragments"
    );
    let tt_bytes_cap =
        usize::try_from(env_u64("TSS_CAP_RESUME_TT_BYTES", 1 << 30)).expect("TT cap fits usize");
    let max_cap = env_u64("TSS_CAP_RESUME_MAX_CAP", 20_000_000);
    let ladder = [10_000, 100_000, 1_000_000, 20_000_000]
        .into_iter()
        .filter(|cap| *cap <= max_cap)
        .collect::<Vec<_>>();
    assert!(!ladder.is_empty(), "max cap must include 10k");
    let selected = selected_ids();
    let corpus = load_corpus();
    let mut rows = 0usize;
    for position in corpus
        .iter()
        .filter(|position| selected.iter().any(|id| id == &position.id))
    {
        rows += 1;
        let profile_caps = SolveCaps {
            node_cap: ladder[0],
            tt_bytes_cap,
            semantic_horizon: u32::MAX,
        };
        let mut resume_solver = TssSolver::default();
        resume_solver.set_width_options(WidthOptions::vcf_pair_complete());
        let mut session = CapResumeSession::new(
            &resume_solver,
            &position.state,
            &profile_caps,
            SolveGoal::Both,
        )
        .unwrap_or_else(|error| panic!("{}: session creation: {error:?}", position.id));

        for &cap in &ladder {
            let caps = SolveCaps {
                node_cap: cap,
                ..profile_caps
            };
            let mut fresh_solver = TssSolver::default();
            fresh_solver.set_width_options(WidthOptions::vcf_pair_complete());
            let fresh_started = Instant::now();
            let fresh = fresh_solver.solve(&position.state, &caps);
            let fresh_ms = fresh_started.elapsed().as_secs_f64() * 1e3;
            let (fresh_pn, fresh_dn) = fresh_solver
                .last_wide_root_numbers()
                .unwrap_or_else(|| panic!("{} cap={cap}: no fresh root PN/DN", position.id));
            if let Some(cert) = &fresh.cert {
                assert!(TssVerifier.verify(&position.state, cert, fresh.status));
            }

            let resume_started = Instant::now();
            let resumed = session
                .advance_to_node_cap(&resume_solver, &position.state, &caps, SolveGoal::Both)
                .unwrap_or_else(|error| {
                    panic!("{} cap={cap}: resume advance: {error:?}", position.id)
                });
            let resume_ms = resume_started.elapsed().as_secs_f64() * 1e3;
            if let Some(cert) = &resumed.result.cert {
                assert!(TssVerifier.verify(&position.state, cert, resumed.result.status));
            }
            let identity = fresh.status == resumed.result.status
                && fresh_pn == resumed.root_pn
                && fresh_dn == resumed.root_dn;
            let expansion_delta =
                i128::from(resumed.result.stats.expansions) - i128::from(fresh.stats.expansions);
            println!(
                "CAP_IDENTITY id={} cap={cap} fresh_pn={fresh_pn} fresh_dn={fresh_dn} fresh_status={} fresh_expansions={} fresh_ms={fresh_ms:.3} resumed_pn={} resumed_dn={} resumed_status={} resumed_cumulative_expansions={} expansion_delta={} resumed_ms={resume_ms:.3} stage_depth={} advances={} reentries={} fresh_cert_nodes={} resumed_cert_nodes={} identity={}",
                position.id,
                status_name(fresh.status),
                fresh.stats.expansions,
                resumed.root_pn,
                resumed.root_dn,
                status_name(resumed.result.status),
                resumed.result.stats.expansions,
                expansion_delta,
                resumed.stage_depth,
                resumed.advances,
                resumed.reentries,
                fresh.cert.as_ref().map_or(0, |cert| cert.nodes.len()),
                resumed
                    .result
                    .cert
                    .as_ref()
                    .map_or(0, |cert| cert.nodes.len()),
                if identity { "PASS" } else { "FAIL" },
            );
            assert!(
                identity,
                "{} cap={cap}: milestone identity failed",
                position.id
            );
            assert_eq!(
                fresh.status, resumed.result.status,
                "{} cap={cap}: uninterrupted outcome contradicted resumed outcome",
                position.id
            );
        }
    }
    assert_eq!(rows, selected.len(), "unknown TSS_CAP_RESUME_ID");
    println!(
        "CAP_IDENTITY_DONE rows={rows} rungs={} tt_bytes_cap={tt_bytes_cap} max_cap={max_cap} result=PASS",
        ladder.len()
    );
}

#[test]
fn cap_resume_discards_on_binding_or_cap_mismatch() {
    tss_residue::begin_job(ResidueJobKey {
        profile: "unit".to_owned(),
        row: "cap_resume_discard".to_owned(),
        cap_rung: 20,
        horizon_rung: "binding_mismatch".to_owned(),
        horizon: u32::MAX,
        resume: true,
        repetition: 0,
    });
    let position = load_corpus()
        .into_iter()
        .find(|position| position.id == "xsnfyll")
        .expect("xsn fixture");
    let caps = SolveCaps {
        node_cap: 10,
        tt_bytes_cap: 1 << 20,
        semantic_horizon: u32::MAX,
    };
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::vcf_pair_complete());
    let mut session =
        CapResumeSession::new(&solver, &position.state, &caps, SolveGoal::Both).unwrap();
    session
        .advance_to_node_cap(&solver, &position.state, &caps, SolveGoal::Both)
        .unwrap();
    assert!(matches!(
        session.advance_to_node_cap(&solver, &position.state, &caps, SolveGoal::Both),
        Err(CapResumeError::NonMonotoneNodeCap)
    ));
    assert!(matches!(
        session.advance_to_node_cap(
            &solver,
            &position.state,
            &SolveCaps {
                node_cap: 20,
                ..caps
            },
            SolveGoal::Both,
        ),
        Err(CapResumeError::Discarded)
    ));

    let mut session =
        CapResumeSession::new(&solver, &position.state, &caps, SolveGoal::Both).unwrap();
    let changed = SolveCaps {
        node_cap: 20,
        semantic_horizon: u32::MAX - 1,
        ..caps
    };
    assert!(matches!(
        session.advance_to_node_cap(&solver, &position.state, &changed, SolveGoal::Both),
        Err(CapResumeError::BindingMismatch)
    ));
    assert!(matches!(
        session.advance_to_node_cap(&solver, &position.state, &changed, SolveGoal::Both),
        Err(CapResumeError::Discarded)
    ));
    let residue = tss_residue::end_job(ResidueJobOutcome::default());
    assert!(
        residue.valid,
        "{}",
        residue.invalid_reason.unwrap_or_default()
    );
    assert!(
        residue.category_ns[ResidueCategory::CapResumeOverhead as usize] > 0,
        "binding/cap rejection must remain visible as resume orchestration"
    );
}
