//! Round-7 test-only measurement harness for the proven Q8 reply-survival
//! kernel. The production library has no corresponding module or telemetry
//! state: `lib.rs` includes this file only under `cfg(test)`.

use std::collections::BTreeMap;
use std::ffi::OsString;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{k_reply_kernel, KReplyShadowRecord, TssSolver, WidthOptions};
use crate::tss_verify::TssVerifier;

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

#[derive(Default)]
struct ShadowSummary {
    fires: usize,
    urgent: usize,
    urgent_quiet: Vec<usize>,
    urgent_kernel: Vec<usize>,
    retention: Vec<f64>,
    proved_urgent_wins: usize,
    hits: usize,
}

impl ShadowSummary {
    fn absorb(&mut self, class: &str, records: &[KReplyShadowRecord]) {
        self.fires += records.len();
        for record in records {
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
            "R7_SUMMARY class={class} fires={} urgent={} urgent_fraction={urgent_fraction:.6} quiet_median={} quiet_p90={} k_median={} k_p90={} retention_median={:.6} retention_p90={:.6} proved_urgent_wins={} hits={} hit_rate={hit_rate:.6}",
            self.fires,
            self.urgent,
            percentile_usize(&self.urgent_quiet, 0.50),
            percentile_usize(&self.urgent_quiet, 0.90),
            percentile_usize(&self.urgent_kernel, 0.50),
            percentile_usize(&self.urgent_kernel, 0.90),
            percentile_f64(&self.retention, 0.50),
            percentile_f64(&self.retention, 0.90),
            self.proved_urgent_wins,
            self.hits,
        );
        let mut pairs = BTreeMap::<(usize, usize), usize>::new();
        for (&quiet, &kernel) in self.urgent_quiet.iter().zip(&self.urgent_kernel) {
            *pairs.entry((quiet, kernel)).or_default() += 1;
        }
        println!("R7_PAIRS class={class} pairs={pairs:?}");
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
    assert!(kernel.urgent, "frozen node must be Q8-urgent");
    assert_eq!(
        kernel.cells,
        vec![remote],
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
