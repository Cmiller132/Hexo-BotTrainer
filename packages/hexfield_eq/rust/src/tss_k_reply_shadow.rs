//! Round-7/8 test-only measurement and verdict-identity harness for the proven
//! Q8 reply-survival kernel. Production consumption lives in `tss_solver`;
//! telemetry and campaign fixtures remain behind `cfg(test)` here.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::time::{Duration, Instant};

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::{CertVerify, DeepResult, DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{k_reply_kernel, KReplyShadowRecord, TssSolver, WidthOptions};
use crate::tss_verify::{TssCertificate, TssVerifier};

const MASTER_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DEFAULT_TT_BYTES: usize = 256 << 20;
const HUMAN_QUOTAS: [usize; 3] = [67, 67, 66];

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().unwrap_or_else(|_| panic!("numeric {name}")))
        .unwrap_or(default)
}

fn tt_bytes() -> usize {
    usize::try_from(env_u64("TSS_R7_TT_BYTES", DEFAULT_TT_BYTES as u64))
        .expect("TSS_R7_TT_BYTES fits usize")
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

fn assert_kernel_matches_full_scan(
    state: &HexoState,
    claimant: Player,
    legal: &[HexCoord],
    kernel: &crate::tss_solver::KReplyKernel,
) {
    let eligible = state.terminal().is_none()
        && state.current_player() == claimant
        && matches!(state.phase(), TurnPhase::SecondStone { .. });
    let defender = claimant.other();
    let defender_windows = eligible
        .then(|| {
            state
                .board()
                .windows()
                .entries()
                .filter(|entry| {
                    entry.active_player() == Some(defender)
                        && matches!(entry.count(defender), 4 | 5)
                })
                .map(|entry| entry.key())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let urgent = !defender_windows.is_empty();
    let expected = if !eligible {
        Vec::new()
    } else if urgent {
        let win_now_windows = state
            .board()
            .windows()
            .entries()
            .filter(|entry| entry.active_player() == Some(claimant) && entry.count(claimant) == 5)
            .map(|entry| entry.key())
            .collect::<Vec<_>>();
        legal
            .iter()
            .copied()
            .filter(|coord| {
                win_now_windows.iter().any(|window| window.contains(*coord))
                    || defender_windows
                        .iter()
                        .all(|window| window.contains(*coord))
            })
            .collect()
    } else {
        legal.to_vec()
    };
    assert_eq!(kernel.eligible, eligible, "Q8 full-scan eligibility");
    assert_eq!(kernel.urgent, urgent, "Q8 full-scan urgency");
    assert_eq!(
        kernel.retained(legal),
        expected.as_slice(),
        "Q8 exact live index must equal the full WindowStore scan"
    );
}

#[derive(Default)]
struct ShadowSummary {
    fires: usize,
    urgent: usize,
    urgent_quiet: Vec<usize>,
    urgent_kernel: Vec<usize>,
    retention: Vec<f64>,
    proved_urgent_wins: usize,
    hits: usize,
    consumed: usize,
}

impl ShadowSummary {
    fn absorb(&mut self, class: &str, records: &[KReplyShadowRecord]) {
        self.fires += records.len();
        for record in records {
            if record.consumed {
                self.consumed += 1;
                assert_eq!(
                    record.consumed_matches_shadow,
                    Some(true),
                    "consumed Q8 set must equal the independently recorded shadow set"
                );
            }
            if !record.urgent {
                continue;
            }
            self.urgent += 1;
            let kernel = record.k_reply.expect("urgent Q8 record has kernel size");
            self.urgent_quiet.push(record.full_quiet);
            self.urgent_kernel.push(kernel);
            self.retention.push(if record.full_quiet == 0 {
                0.0
            } else {
                kernel as f64 / record.full_quiet as f64
            });
            if record.proved_win {
                self.proved_urgent_wins += 1;
                if record.winning_edge_in_k == Some(true) {
                    self.hits += 1;
                } else {
                    panic!(
                        "Q8_COUNTEREXAMPLE class={class} edge={:?} full_quiet={} k_reply={} position={:?}",
                        record.winning_edge, record.full_quiet, kernel, record.position
                    );
                }
            }
        }
    }

    fn print(&self, class: &str) {
        self.print_line(class);
        let mut pairs = BTreeMap::<(usize, usize), usize>::new();
        for (&quiet, &kernel) in self.urgent_quiet.iter().zip(&self.urgent_kernel) {
            *pairs.entry((quiet, kernel)).or_default() += 1;
        }
        println!("R7_PAIRS class={class} pairs={pairs:?}");
    }

    fn print_line(&self, class: &str) {
        let urgent_fraction = if self.fires == 0 {
            0.0
        } else {
            self.urgent as f64 / self.fires as f64
        };
        let hit_rate = if self.proved_urgent_wins == 0 {
            1.0
        } else {
            self.hits as f64 / self.proved_urgent_wins as f64
        };
        println!(
            "R7_SUMMARY class={class} fires={} urgent={} consumed={} urgent_fraction={urgent_fraction:.6} quiet_median={} quiet_p90={} k_median={} k_p90={} retention_median={:.6} retention_p90={:.6} proved_urgent_wins={} hits={} hit_rate={hit_rate:.6}",
            self.fires,
            self.urgent,
            self.consumed,
            percentile_usize(&self.urgent_quiet, 0.50),
            percentile_usize(&self.urgent_quiet, 0.90),
            percentile_usize(&self.urgent_kernel, 0.50),
            percentile_usize(&self.urgent_kernel, 0.90),
            percentile_f64(&self.retention, 0.50),
            percentile_f64(&self.retention, 0.90),
            self.proved_urgent_wins,
            self.hits,
        );
        assert_eq!(self.hits, self.proved_urgent_wins, "Q8 hit rate drift");
    }
}

fn percentile_usize(values: &[usize], quantile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index]
}

fn percentile_f64(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index]
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn restore_shadow_env(previous: Option<OsString>) {
    if let Some(value) = previous {
        std::env::set_var("TSS_K_REPLY_SHADOW", value);
    } else {
        std::env::remove_var("TSS_K_REPLY_SHADOW");
    }
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var_os(name);
        if let Some(value) = value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.previous.take() {
            std::env::set_var(self.name, value);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

struct ProfileRun {
    result: DeepResult<TssCertificate>,
    shadow: Vec<KReplyShadowRecord>,
    wall: Duration,
}

fn run_consume_profile(state: &HexoState, caps: &SolveCaps, consume_q8: bool) -> ProfileRun {
    // Keep the OFF timing/search path genuinely telemetry-free. ON records
    // the kernel it already computes for consumption, including agreement.
    let capture_shadow = std::env::var("TSS_R8_CAPTURE_SHADOW").as_deref() != Ok("0");
    let _shadow = EnvVarGuard::set(
        "TSS_K_REPLY_SHADOW",
        (consume_q8 && capture_shadow).then_some("1"),
    );
    let _consume = EnvVarGuard::set("TSS_K_REPLY_CONSUME", consume_q8.then_some("1"));
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::round3_consume());
    let started = Instant::now();
    let result = solver.solve(state, caps);
    let wall = started.elapsed();
    ProfileRun {
        result,
        shadow: solver.k_reply_shadow().to_vec(),
        wall,
    }
}

fn verify_profile(class: &str, id: &str, state: &HexoState, run: &ProfileRun) {
    if run.result.status == ProofStatus::Unknown {
        assert!(
            run.result.cert.is_none(),
            "{class}/{id}: UNKNOWN must not carry a certificate"
        );
    } else {
        let cert = run
            .result
            .cert
            .as_ref()
            .unwrap_or_else(|| panic!("{class}/{id}: hard verdict missing certificate"));
        assert!(
            TssVerifier.verify(state, cert, run.result.status),
            "{class}/{id}: emitted certificate rejected"
        );
    }
}

fn cert_fingerprint(cert: Option<&TssCertificate>) -> u64 {
    let Some(cert) = cert else {
        return 0;
    };
    format!("{cert:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn paired_identity(
    class: &str,
    id: &str,
    state: &HexoState,
    caps: &SolveCaps,
) -> (ProfileRun, ProfileRun, bool) {
    let off = run_consume_profile(state, caps, false);
    let on = run_consume_profile(state, caps, true);
    verify_profile(class, id, state, &off);
    verify_profile(class, id, state, &on);
    assert_eq!(
        on.result.status,
        off.result.status,
        "STOP K_REPLY_VERDICT_DIFFERENCE class={class} id={id} cap={} off={} on={}",
        caps.node_cap,
        status_name(off.result.status),
        status_name(on.result.status),
    );
    let cert_equal = match (&off.result.cert, &on.result.cert) {
        (Some(off), Some(on)) => off == on,
        (None, None) => true,
        _ => false,
    };
    assert!(
        on.shadow
            .iter()
            .all(|record| record.consumed_matches_shadow != Some(false)),
        "{class}/{id}: consumed/shadow candidate disagreement"
    );
    println!(
        "R8_IDENTITY class={class} id={id} cap={} status={} off_nodes={} on_nodes={} off_ms={:.3} on_ms={:.3} off_fires={} on_fires={} on_urgent={} cert_equal={cert_equal}",
        caps.node_cap,
        status_name(on.result.status),
        off.result.stats.nodes,
        on.result.stats.nodes,
        off.wall.as_secs_f64() * 1e3,
        on.wall.as_secs_f64() * 1e3,
        off.shadow.len(),
        on.shadow.len(),
        on.shadow.iter().filter(|record| record.urgent).count(),
    );
    (off, on, cert_equal)
}

#[derive(Default)]
struct CohortIdentity {
    roots: usize,
    cert_pairs: usize,
    cert_equal: usize,
    off_nodes: u64,
    on_nodes: u64,
    off_wall: Duration,
    on_wall: Duration,
    off_shadow: ShadowSummary,
    on_shadow: ShadowSummary,
}

impl CohortIdentity {
    fn absorb(&mut self, class: &str, off: &ProfileRun, on: &ProfileRun, cert_equal: bool) {
        self.roots += 1;
        self.off_nodes = self.off_nodes.saturating_add(off.result.stats.nodes);
        self.on_nodes = self.on_nodes.saturating_add(on.result.stats.nodes);
        self.off_wall += off.wall;
        self.on_wall += on.wall;
        if off.result.cert.is_some() && on.result.cert.is_some() {
            self.cert_pairs += 1;
            self.cert_equal += usize::from(cert_equal);
        }
        self.off_shadow.absorb(class, &off.shadow);
        self.on_shadow.absorb(class, &on.shadow);
    }

    fn print(&self, class: &str) {
        let node_delta = self.on_nodes as i128 - self.off_nodes as i128;
        let node_pct = if self.off_nodes == 0 {
            0.0
        } else {
            node_delta as f64 / self.off_nodes as f64 * 100.0
        };
        let off_ms = self.off_wall.as_secs_f64() * 1e3;
        let on_ms = self.on_wall.as_secs_f64() * 1e3;
        let wall_pct = if off_ms == 0.0 {
            0.0
        } else {
            (on_ms - off_ms) / off_ms * 100.0
        };
        println!(
            "R8_COHORT class={class} roots={} off_nodes={} on_nodes={} node_delta={node_delta} node_pct={node_pct:.6} off_ms={off_ms:.3} on_ms={on_ms:.3} wall_pct={wall_pct:.6} cert_pairs={} cert_equal={} off_fires={} on_fires={} off_urgent={} on_urgent={} consumed={}",
            self.roots,
            self.off_nodes,
            self.on_nodes,
            self.cert_pairs,
            self.cert_equal,
            self.off_shadow.fires,
            self.on_shadow.fires,
            self.off_shadow.urgent,
            self.on_shadow.urgent,
            self.on_shadow.consumed,
        );
        self.off_shadow.print_line(&format!("{class}_off"));
        self.on_shadow.print_line(&format!("{class}_on"));
    }
}

#[test]
fn tss_round7_k_reply_frozen_witness() {
    let moves = [
        (0, 0),
        (-1, 0),
        (1, -1),
        (1, 0),
        (2, 0),
        (2, -2),
        (3, -3),
        (3, 0),
        (4, 6),
        (4, -4),
        (5, -5),
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
        (6, 0),
    ];
    let root = replay(&moves);
    let claimant = root.current_player();
    assert_eq!(claimant, Player::Player0);
    assert!(matches!(
        root.phase(),
        TurnPhase::SecondStone { first } if first == HexCoord::new(6, 0)
    ));
    let mut legal = Vec::new();
    root.write_legal_moves(&mut legal);
    assert_eq!(legal.len(), 538);

    let remote = HexCoord::new(6, -6);
    let kernel = k_reply_kernel(&root, claimant, &legal);
    assert_kernel_matches_full_scan(&root, claimant, &legal, &kernel);
    assert!(kernel.urgent, "frozen node must be Q8-urgent");
    assert_eq!(
        kernel.retained(&legal),
        &[remote],
        "Q8 kernel must remain singleton"
    );
    assert!(kernel.cells.contains(&remote));

    let defender = claimant.other();
    let mut eliminated = 0usize;
    for alternative in legal.iter().copied().filter(|coord| *coord != remote) {
        let mut child = root.clone();
        let attack = apply_placement(&mut child, Placement { coord: alternative })
            .expect("enumerated attacker alternative");
        assert!(
            attack.outcome.is_none(),
            "alternative unexpectedly wins now"
        );
        let defense = apply_placement(&mut child, Placement { coord: remote })
            .expect("remote completion remains legal");
        assert_eq!(
            defense.outcome.map(|outcome| outcome.winner),
            Some(defender)
        );
        eliminated += 1;
    }
    assert_eq!(eliminated, 537);
}

#[test]
fn tss_round8_k_reply_trigger_matrix() {
    let opening = HexoState::new();
    let mut opening_legal = Vec::new();
    opening.write_legal_moves(&mut opening_legal);
    let opening_kernel = k_reply_kernel(&opening, opening.current_player(), &opening_legal);
    assert_kernel_matches_full_scan(
        &opening,
        opening.current_player(),
        &opening_legal,
        &opening_kernel,
    );
    assert!(!opening_kernel.eligible);
    assert!(!opening_kernel.urgent);
    assert!(
        opening_kernel.retained(&opening_legal).is_empty(),
        "Q8 must not widen to Opening"
    );

    let mut first_stone = opening;
    apply_placement(
        &mut first_stone,
        Placement {
            coord: HexCoord::ZERO,
        },
    )
    .expect("legal opening");
    assert!(matches!(first_stone.phase(), TurnPhase::FirstStone));
    let mut first_legal = Vec::new();
    first_stone.write_legal_moves(&mut first_legal);
    let first_kernel = k_reply_kernel(&first_stone, first_stone.current_player(), &first_legal);
    assert_kernel_matches_full_scan(
        &first_stone,
        first_stone.current_player(),
        &first_legal,
        &first_kernel,
    );
    assert!(!first_kernel.eligible);
    assert!(!first_kernel.urgent);
    assert!(
        first_kernel.retained(&first_legal).is_empty(),
        "Q8 must not widen to FirstStone"
    );

    let first = first_legal[0];
    let claimant = first_stone.current_player();
    apply_placement(&mut first_stone, Placement { coord: first }).expect("legal first placement");
    assert!(matches!(
        first_stone.phase(),
        TurnPhase::SecondStone { first: stored } if stored == first
    ));
    let mut second_legal = Vec::new();
    first_stone.write_legal_moves(&mut second_legal);
    let quiet_kernel = k_reply_kernel(&first_stone, claimant, &second_legal);
    assert_kernel_matches_full_scan(&first_stone, claimant, &second_legal, &quiet_kernel);
    assert!(quiet_kernel.eligible);
    assert!(!quiet_kernel.urgent);
    assert_eq!(
        quiet_kernel.retained(&second_legal),
        second_legal.as_slice(),
        "nonurgent SecondStone must return full Legal(P)"
    );
    let wrong_claimant = k_reply_kernel(&first_stone, claimant.other(), &second_legal);
    assert_kernel_matches_full_scan(
        &first_stone,
        claimant.other(),
        &second_legal,
        &wrong_claimant,
    );
    assert!(!wrong_claimant.eligible);
    assert!(!wrong_claimant.urgent);
    assert!(
        wrong_claimant.retained(&second_legal).is_empty(),
        "Q8 identity gate must reject the wrong claimant"
    );
}

#[test]
#[ignore = "round-7 telemetry on/off identity on double_fork_compact corpus row"]
fn tss_round7_k_reply_identity() {
    let previous = std::env::var_os("TSS_K_REPLY_SHADOW");
    let caps = SolveCaps {
        node_cap: 10_000,
        tt_bytes_cap: tt_bytes(),
        semantic_horizon: u32::MAX,
    };
    let state = crate::tss_spare_corpus::mining_candidate("double_fork_compact");
    std::env::remove_var("TSS_K_REPLY_SHADOW");
    let mut off_solver = TssSolver::default();
    off_solver.set_width_options(WidthOptions::round3_consume());
    let off = off_solver.solve(&state, &caps);
    assert!(off_solver.k_reply_shadow().is_empty());

    std::env::set_var("TSS_K_REPLY_SHADOW", "1");
    let mut on_solver = TssSolver::default();
    on_solver.set_width_options(WidthOptions::round3_consume());
    let on = on_solver.solve(&state, &caps);
    assert_eq!(on.status, off.status, "double_fork_compact status identity");
    assert_eq!(
        on.stats.nodes, off.stats.nodes,
        "double_fork_compact node identity"
    );
    assert_eq!(
        on.stats.tt_hits, off.stats.tt_hits,
        "double_fork_compact TT identity"
    );
    assert_eq!(
        on.cert, off.cert,
        "double_fork_compact certificate identity"
    );
    let fires = on_solver.k_reply_shadow().len();
    assert!(fires > 0, "identity row must exercise quiet fallback");
    restore_shadow_env(previous);
    println!(
        "R7_IDENTITY id=double_fork_compact nodes={} fires={fires} status_node_tt_cert=identical",
        on.stats.nodes
    );
}

#[test]
#[ignore = "round-8 Q8 consumption identity and local reduction on double_fork_compact"]
fn tss_round8_k_reply_double_fork_identity() {
    let state = crate::tss_spare_corpus::mining_candidate("double_fork_compact");
    let caps = SolveCaps {
        node_cap: env_u64("TSS_R3_CAP", 10_000),
        tt_bytes_cap: tt_bytes(),
        semantic_horizon: 45,
    };
    let (off, on, cert_equal) =
        paired_identity("double_fork_compact", "double_fork_compact", &state, &caps);
    assert_eq!(off.result.status, ProofStatus::Win);
    assert_eq!(off.result.stats.nodes, 409);
    assert_eq!(on.result.status, ProofStatus::Win);
    let urgent = on
        .shadow
        .iter()
        .find(|record| record.urgent)
        .expect("double_fork_compact must consume its urgent fallback node");
    assert_eq!((urgent.full_quiet, urgent.k_reply), (478, Some(1)));
    assert!(urgent.consumed);
    assert_eq!(urgent.consumed_matches_shadow, Some(true));

    let mut cohort = CohortIdentity::default();
    cohort.absorb("double_fork_compact", &off, &on, cert_equal);
    cohort.print("double_fork_compact");
}

#[test]
#[ignore = "round-7 all-19 Q8 telemetry at the 10k/100k ladder rungs"]
fn tss_round7_k_reply_forcing_measurement() {
    assert!(
        std::env::var_os("TSS_K_REPLY_SHADOW").is_some(),
        "set TSS_K_REPLY_SHADOW=1"
    );
    let mut summary = ShadowSummary::default();
    let tt_bytes_cap = tt_bytes();
    let selected = std::env::var("TSS_R7_CORPUS_ID").ok().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>()
    });
    let mut rows = 0usize;
    for position in crate::tss_corpus::load_corpus() {
        if selected
            .as_ref()
            .is_some_and(|ids| !ids.iter().any(|id| id == &position.id))
        {
            continue;
        }
        rows += 1;
        let cap = forcing_telemetry_cap(&position.id);
        eprintln!("R7_ROW_START class=forcing19 id={} cap={cap}", position.id);
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::round3_consume());
        let result = solver.solve(
            &position.state,
            &SolveCaps {
                node_cap: cap,
                tt_bytes_cap,
                semantic_horizon: u32::MAX,
            },
        );
        summary.absorb("forcing19", solver.k_reply_shadow());
        println!(
            "R7_ROW class=forcing19 id={} cap={cap} status={} nodes={} fires={} urgent={}",
            position.id,
            status_name(result.status),
            result.stats.nodes,
            solver.k_reply_shadow().len(),
            solver
                .k_reply_shadow()
                .iter()
                .filter(|record| record.urgent)
                .count(),
        );
        eprintln!(
            "R7_ROW_DONE class=forcing19 id={} cap={cap} status={} nodes={} fires={} urgent={}",
            position.id,
            status_name(result.status),
            result.stats.nodes,
            solver.k_reply_shadow().len(),
            solver
                .k_reply_shadow()
                .iter()
                .filter(|record| record.urgent)
                .count(),
        );
        assert!(
            position.expect_win || result.status != ProofStatus::Win,
            "{}: Q8 telemetry found WIN on official NO row",
            position.id
        );
    }
    assert!(rows > 0, "TSS_R7_CORPUS_ID selected no official row");
    if let Some(ids) = selected {
        assert_eq!(
            rows,
            ids.len(),
            "TSS_R7_CORPUS_ID contains unknown/duplicate id"
        );
    }
    summary.print("forcing19");
}

/// Use each row's first documented closing rung. Rows whose official rung is
/// 1M/20M (and NO rows that normally climb to 1M) are clamped to 100k for this
/// telemetry-only pass, as required by the round-7 work order.
fn forcing_telemetry_cap(id: &str) -> u64 {
    match id {
        "0hz3hty"
        | "8is963b"
        | "acly7kb"
        | "dy3dg99"
        | "g2xx6wl"
        | "hu01jk4"
        | "jh7yo7y"
        | "jnzzmcm"
        | "xsnfyll"
        | "strongloss_b_prefix8" => 10_000,
        _ => 100_000,
    }
}

#[test]
#[ignore = "round-8 paired Q8 consumption identity on forcing-19 at 10k/100k"]
fn tss_round8_k_reply_forcing_identity() {
    let single_profile = std::env::var("TSS_R8_PROFILE").ok().map(|profile| {
        assert!(matches!(profile.as_str(), "off" | "on"));
        profile
    });
    let selected = std::env::var("TSS_R8_CORPUS_ID").ok().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>()
    });
    let mut cohort = CohortIdentity::default();
    let mut rows = 0usize;
    for position in crate::tss_corpus::load_corpus() {
        if selected
            .as_ref()
            .is_some_and(|ids| !ids.iter().any(|id| id == &position.id))
        {
            continue;
        }
        rows += 1;
        let cap = forcing_telemetry_cap(&position.id);
        eprintln!("R8_ROW_START class=forcing19 id={} cap={cap}", position.id);
        let caps = SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes(),
            semantic_horizon: u32::MAX,
        };
        if let Some(profile) = single_profile.as_deref() {
            let consume = profile == "on";
            let run = run_consume_profile(&position.state, &caps, consume);
            verify_profile("forcing19", &position.id, &position.state, &run);
            assert!(
                position.expect_win || run.result.status != ProofStatus::Win,
                "{}: Q8 {profile} profile produced WIN on an official NO row",
                position.id
            );
            println!(
                "R8_SINGLE class=forcing19 profile={profile} id={} cap={cap} status={} nodes={} ms={:.3} fires={} urgent={} cert_fp={:016x}",
                position.id,
                status_name(run.result.status),
                run.result.stats.nodes,
                run.wall.as_secs_f64() * 1e3,
                run.shadow.len(),
                run.shadow.iter().filter(|record| record.urgent).count(),
                cert_fingerprint(run.result.cert.as_ref()),
            );
            continue;
        }
        let (off, on, cert_equal) =
            paired_identity("forcing19", &position.id, &position.state, &caps);
        assert!(
            position.expect_win || on.result.status != ProofStatus::Win,
            "{}: Q8 consumption produced WIN on an official NO row",
            position.id
        );
        cohort.absorb("forcing19", &off, &on, cert_equal);
        eprintln!(
            "R8_ROW_DONE class=forcing19 id={} cap={cap} status={} off_nodes={} on_nodes={}",
            position.id,
            status_name(on.result.status),
            off.result.stats.nodes,
            on.result.stats.nodes,
        );
    }
    assert!(rows > 0, "TSS_R8_CORPUS_ID selected no official row");
    if let Some(ids) = selected {
        assert_eq!(
            rows,
            ids.len(),
            "TSS_R8_CORPUS_ID contains unknown/duplicate id"
        );
    } else {
        assert_eq!(rows, 19, "forcing identity campaign must cover all rows");
    }
    if single_profile.is_none() {
        cohort.print("forcing19");
    }
}

#[test]
#[ignore = "round-7 double_fork_compact Q8 telemetry"]
fn tss_round7_k_reply_double_fork_measurement() {
    assert!(
        std::env::var_os("TSS_K_REPLY_SHADOW").is_some(),
        "set TSS_K_REPLY_SHADOW=1"
    );
    let cap = env_u64("TSS_R3_CAP", 10_000);
    let state = crate::tss_spare_corpus::mining_candidate("double_fork_compact");
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::round3_consume());
    let result = solver.solve(
        &state,
        &SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes(),
            semantic_horizon: 45,
        },
    );
    assert_eq!(result.status, ProofStatus::Win);
    assert_eq!(result.stats.nodes, 409);
    assert!(result.cert.as_ref().is_some_and(|cert| TssVerifier.verify(
        &state,
        cert,
        result.status
    )));
    let mut summary = ShadowSummary::default();
    summary.absorb("double_fork_compact", solver.k_reply_shadow());
    summary.print("double_fork_compact");
}

struct HumanGame {
    moves: Vec<(i16, i16)>,
    winner: i8,
}

fn parse_ints(slice: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in slice.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            out.push(current.parse().expect("i16 token"));
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current.parse().expect("i16 token"));
    }
    out
}

fn parse_human_game(line: &str) -> Option<HumanGame> {
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
    let numbers = parse_ints(&after[start..=end?]);
    let moves = numbers
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    let winner_key = "\"winner\":";
    let winner_after = &line[line.find(winner_key)? + winner_key.len()..];
    let mut token = String::new();
    for ch in winner_after.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            token.push(ch);
        } else if !token.is_empty() {
            break;
        }
    }
    Some(HumanGame {
        moves,
        winner: token.parse().ok()?,
    })
}

fn load_human_corpus() -> Vec<HumanGame> {
    let path = std::env::var("TSS_R7_HUMAN_CORPUS").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl".to_string()
    });
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read human corpus {path}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_human_game(line).expect("valid human-corpus row"))
        .collect()
}

#[derive(Clone, Copy)]
struct HumanCandidate {
    game: usize,
    prefix: usize,
    band: usize,
}

fn band_of(ply: u32) -> usize {
    if ply <= 12 {
        0
    } else if ply <= 40 {
        1
    } else {
        2
    }
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

fn human_sample(games: &[HumanGame], seed: u64) -> Vec<HumanCandidate> {
    let mut candidates = Vec::new();
    for (game, row) in games.iter().enumerate() {
        if !matches!(row.winner, -1 | 1) {
            continue;
        }
        let mut state = HexoState::new();
        for (prefix, &(q, r)) in row.moves.iter().enumerate() {
            if !state.is_terminal() && matches!(state.phase(), TurnPhase::FirstStone) {
                candidates.push(HumanCandidate {
                    game,
                    prefix,
                    band: band_of(state.placements_made()),
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
            .expect("legal human-corpus replay");
        }
    }

    let mut sample = Vec::with_capacity(HUMAN_QUOTAS.iter().sum());
    for (band, quota) in HUMAN_QUOTAS.into_iter().enumerate() {
        let mut band_candidates = candidates
            .iter()
            .copied()
            .filter(|candidate| candidate.band == band)
            .collect::<Vec<_>>();
        let mut rng = XorShift((seed ^ (band as u64).wrapping_mul(0x9E37_79B9)) | 1);
        for index in (1..band_candidates.len()).rev() {
            let selected = (rng.next() % (index as u64 + 1)) as usize;
            band_candidates.swap(index, selected);
        }
        assert!(band_candidates.len() >= quota);
        sample.extend(band_candidates.into_iter().take(quota));
    }
    sample
}

#[test]
#[ignore = "round-7 deterministic 200-root human-corpus Q8 telemetry"]
fn tss_round7_k_reply_human_measurement() {
    assert!(
        std::env::var_os("TSS_K_REPLY_SHADOW").is_some(),
        "set TSS_K_REPLY_SHADOW=1"
    );
    let seed = env_u64("TSS_R7_HUMAN_SEED", MASTER_SEED);
    let cap = env_u64("TSS_R7_HUMAN_CAP", 10_000);
    let games = load_human_corpus();
    let sample = human_sample(&games, seed);
    assert_eq!(sample.len(), 200);
    println!(
        "R7_HUMAN_SETUP roots={} quotas={HUMAN_QUOTAS:?} seed={seed} seed_hex=0x{seed:016X} cap={cap}",
        sample.len()
    );

    let mut all = ShadowSummary::default();
    let mut bands = [
        ShadowSummary::default(),
        ShadowSummary::default(),
        ShadowSummary::default(),
    ];
    let mut statuses = [0usize; 3];
    for (index, candidate) in sample.into_iter().enumerate() {
        let state = replay(&games[candidate.game].moves[..candidate.prefix]);
        assert!(matches!(state.phase(), TurnPhase::FirstStone));
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::round3_consume());
        let result = solver.solve(
            &state,
            &SolveCaps {
                node_cap: cap,
                tt_bytes_cap: tt_bytes(),
                semantic_horizon: state.placements_made().saturating_add(50),
            },
        );
        statuses[match result.status {
            ProofStatus::Win => 0,
            ProofStatus::Loss => 1,
            ProofStatus::Unknown => 2,
        }] += 1;
        all.absorb("human200", solver.k_reply_shadow());
        bands[candidate.band].absorb("human200", solver.k_reply_shadow());
        println!(
            "R7_HUMAN_ROOT index={index} band={} ply={} status={} nodes={} fires={} urgent={}",
            candidate.band,
            state.placements_made(),
            status_name(result.status),
            result.stats.nodes,
            solver.k_reply_shadow().len(),
            solver
                .k_reply_shadow()
                .iter()
                .filter(|record| record.urgent)
                .count(),
        );
    }
    println!(
        "R7_HUMAN_STATUS win={} loss={} unknown={}",
        statuses[0], statuses[1], statuses[2]
    );
    all.print("human200");
    for (band, summary) in bands.iter().enumerate() {
        summary.print(match band {
            0 => "human_ply_le_12",
            1 => "human_ply_13_40",
            _ => "human_ply_gt_40",
        });
    }
}

#[test]
#[ignore = "round-8 paired Q8 consumption identity on fixed-seed 200 human roots"]
fn tss_round8_k_reply_human_identity() {
    let seed = env_u64("TSS_R7_HUMAN_SEED", MASTER_SEED);
    let cap = env_u64("TSS_R7_HUMAN_CAP", 10_000);
    let games = load_human_corpus();
    let sample = human_sample(&games, seed);
    assert_eq!(sample.len(), 200);
    println!(
        "R8_HUMAN_SETUP roots={} quotas={HUMAN_QUOTAS:?} seed={seed} seed_hex=0x{seed:016X} cap={cap}",
        sample.len()
    );

    let mut all = CohortIdentity::default();
    let mut bands = [
        CohortIdentity::default(),
        CohortIdentity::default(),
        CohortIdentity::default(),
    ];
    let mut statuses = [0usize; 3];
    for (index, candidate) in sample.into_iter().enumerate() {
        let state = replay(&games[candidate.game].moves[..candidate.prefix]);
        assert!(matches!(state.phase(), TurnPhase::FirstStone));
        let caps = SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes(),
            semantic_horizon: state.placements_made().saturating_add(50),
        };
        let id = format!(
            "root{index}_band{}_ply{}",
            candidate.band,
            state.placements_made()
        );
        let (off, on, cert_equal) = paired_identity("human200", &id, &state, &caps);
        statuses[match on.result.status {
            ProofStatus::Win => 0,
            ProofStatus::Loss => 1,
            ProofStatus::Unknown => 2,
        }] += 1;
        all.absorb("human200", &off, &on, cert_equal);
        bands[candidate.band].absorb("human200", &off, &on, cert_equal);
    }
    println!(
        "R8_HUMAN_STATUS win={} loss={} unknown={}",
        statuses[0], statuses[1], statuses[2]
    );
    all.print("human200");
    for (band, cohort) in bands.iter().enumerate() {
        cohort.print(match band {
            0 => "human_ply_le_12",
            1 => "human_ply_13_40",
            _ => "human_ply_gt_40",
        });
    }
}
