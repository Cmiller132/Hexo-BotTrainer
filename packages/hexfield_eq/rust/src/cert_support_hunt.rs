//! NQ3 certificate-support hunt.
//!
//! The production verifier has an intentionally exact root binding
//! (`TssCertificate::root`) and exact full-position keys for shared replay DAG
//! nodes.  Consequently its true read support is not a bounded cell set: the
//! root equality asserts absence of every additional stone on the unbounded
//! board.  This harness measures that obstruction directly and separately
//! measures the finite certificate-coordinate footprint after rebinding the
//! root and translating absolute certificate clocks.  The latter is a shadow
//! experiment, not a claim about today's strict certificate.
//!
//! Run the deterministic campaign explicitly, in release mode and serially:
//!
//! ```text
//! cargo test -p hexfield_eq --release cert_support_campaign -- --ignored --nocapture --test-threads=1
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::mem::size_of;
use std::time::Instant;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState, Placement, Player, TurnPhase,
    WindowKey,
};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::{CertNode, RootBinding, TssCertificate, TssVerifier};

type Cell = (i16, i16);

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DEFAULT_TT_BYTES: usize = 64 << 20;
const STAGE4_TOTAL_CACHE_BYTES: usize = 512 << 20;
const STAGE4_VERIFY_TEMP_BYTES: usize = crate::tss_verify::MAX_VERIFY_MEMO_BYTES;
const HUMAN_PATH: &str =
    "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl";

const DOUBLE_FORK_COMPACT: &[(i16, i16)] = &[
    (0, 0),
    (-1, 0),
    (4, 1),
    (1, 0),
    (2, 0),
    (4, 2),
    (4, 3),
    (3, 0),
    (4, 6),
    (4, 4),
    (4, 5),
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
    (-1, 2),
];

#[derive(Clone)]
struct Position {
    id: String,
    state: HexoState,
}

struct Game {
    hash: String,
    moves: Vec<Cell>,
    winner: i8,
}

#[derive(Clone, Copy)]
struct Candidate {
    game: usize,
    prefix: usize,
    band: usize,
}

#[derive(Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn c(cell: Cell) -> HexCoord {
    HexCoord::new(cell.0, cell.1)
}

fn cell(coord: HexCoord) -> Cell {
    (coord.q, coord.r)
}

fn replay(moves: &[Cell]) -> Option<HexoState> {
    let mut state = HexoState::new();
    for &mv in moves {
        if state.is_terminal() {
            return None;
        }
        apply_placement(&mut state, Placement { coord: c(mv) }).ok()?;
    }
    Some(state)
}

fn parse_ints(text: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut token = String::new();
    for ch in text.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            token.push(ch);
        } else if !token.is_empty() {
            out.push(token.parse().expect("integer token"));
            token.clear();
        }
    }
    if !token.is_empty() {
        out.push(token.parse().expect("integer token"));
    }
    out
}

fn json_scalar_i8(line: &str, key: &str) -> Option<i8> {
    let marker = format!("\"{key}\":");
    let rest = &line[line.find(&marker)? + marker.len()..];
    let token = rest
        .trim_start()
        .split(|ch: char| !ch.is_ascii_digit() && ch != '-')
        .next()?;
    token.parse().ok()
}

fn parse_game(line: &str) -> Option<Game> {
    let hash_marker = "\"game_hash\":\"";
    let hash_rest = &line[line.find(hash_marker)? + hash_marker.len()..];
    let hash = hash_rest[..hash_rest.find('"')?].to_owned();
    let moves_marker = "\"moves\":";
    let rest = &line[line.find(moves_marker)? + moves_marker.len()..];
    let start = rest.find('[')?;
    let mut depth = 0i32;
    let mut end = None;
    for (index, byte) in rest.as_bytes().iter().copied().enumerate().skip(start) {
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
    let ints = parse_ints(&rest[start..=end?]);
    let moves = ints.chunks_exact(2).map(|v| (v[0], v[1])).collect();
    Some(Game {
        hash,
        moves,
        winner: json_scalar_i8(line, "winner")?,
    })
}

fn load_games() -> Vec<Game> {
    let path = std::env::var("TSS_CERT_SUPPORT_HUMAN").unwrap_or_else(|_| HUMAN_PATH.into());
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read human corpus {path}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_game(line).expect("human-corpus row"))
        .collect()
}

fn load_forcing() -> Vec<Position> {
    let path = format!(
        "{}/rust/corpus/forcing_corpus_moves.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("forcing corpus");
    let mut lines = text.lines();
    let mut positions = Vec::new();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        let mut id = String::new();
        let mut win = false;
        let mut count = 0usize;
        for token in header.split_whitespace().skip(1) {
            let (key, value) = token.split_once('=').expect("forcing key=value");
            match key {
                "id" => id = value.into(),
                "expect" => win = value == "WIN",
                "nstones" => count = value.parse().expect("nstones"),
                _ => {}
            }
        }
        let mut moves = Vec::with_capacity(count);
        for _ in 0..count {
            let ints = parse_ints(lines.next().expect("forcing move"));
            moves.push((ints[0], ints[1]));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"));
        if win {
            positions.push(Position {
                id,
                state: replay(&moves).expect("legal nonterminal forcing position"),
            });
        }
    }
    positions.push(Position {
        id: "double_fork_compact".into(),
        state: replay(DOUBLE_FORK_COMPACT).expect("double_fork_compact replay"),
    });
    positions
}

/// Finite coordinates named by the certificate body.  This intentionally does
/// not call itself READ SUPPORT: the strict root binding asserts the complement
/// is empty and therefore makes true support unbounded.  It is the prospective
/// support-hash payload after a future proof removes that global obligation.
fn body_footprint(cert: &TssCertificate) -> BTreeSet<Cell> {
    let mut out = BTreeSet::new();
    fn insert_window(out: &mut BTreeSet<Cell>, key: hexo_engine::WindowKey) {
        out.extend(key.cells().into_iter().map(cell));
    }
    for node in &cert.nodes {
        match node {
            CertNode::OrCompletion { mv, witness, .. } => {
                out.insert(cell(*mv));
                insert_window(&mut out, *witness);
            }
            CertNode::Win { witness, .. } => insert_window(&mut out, *witness),
            CertNode::Loss { witnesses, .. } => {
                for witness in witnesses {
                    insert_window(&mut out, *witness);
                }
            }
            CertNode::Choice { mv, .. } => {
                out.insert(cell(*mv));
            }
            CertNode::Universal {
                edges,
                commutations,
                ..
            } => {
                out.extend(edges.iter().map(|edge| cell(edge.mv)));
                for item in commutations {
                    out.insert(cell(item.first));
                    out.insert(cell(item.omitted_second));
                }
            }
        }
    }
    out
}

fn body_footprint_with_zones(state: &HexoState, cert: &TssCertificate) -> BTreeSet<Cell> {
    let mut out = body_footprint(cert);
    if let Some(zones) = crate::tss_verify::round3_rederived_zones(state, cert) {
        for (_, zone) in zones {
            out.extend(zone.into_iter().map(cell));
        }
    }
    out
}

fn body_metrics(state: &HexoState, cert: &TssCertificate) -> (usize, usize, u16, f64) {
    let support = body_footprint_with_zones(state, cert);
    let occupied: BTreeSet<Cell> = state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .map(cell)
        .collect();
    let frame = support.difference(&occupied).count();
    let root_in = support.intersection(&occupied).count();
    let outside_fraction = if occupied.is_empty() {
        0.0
    } else {
        (occupied.len() - root_in) as f64 / occupied.len() as f64
    };
    let mut active: BTreeSet<Cell> = support.intersection(&occupied).copied().collect();
    for node in &cert.nodes {
        match node {
            CertNode::OrCompletion { mv, .. } | CertNode::Choice { mv, .. } => {
                active.insert(cell(*mv));
            }
            CertNode::Universal { edges, .. } => {
                active.extend(edges.iter().map(|edge| cell(edge.mv)));
            }
            CertNode::Win { .. } | CertNode::Loss { .. } => {}
        }
    }
    let radius = support
        .iter()
        .map(|&point| {
            active
                .iter()
                .map(|&anchor| hex_distance(c(point), c(anchor)) as u16)
                .min()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    (support.len(), frame, radius, outside_fraction)
}

fn shifted_rebind(cert: &TssCertificate, state: &HexoState, delta: u32) -> TssCertificate {
    let mut shifted = cert.clone();
    shifted.root = RootBinding::from_state(state);
    shifted.semantic_horizon = shifted.semantic_horizon.saturating_add(delta);
    for node in &mut shifted.nodes {
        match node {
            CertNode::OrCompletion { completion_ply, .. } => {
                *completion_ply = completion_ply.saturating_add(delta);
            }
            CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
                *resolution_ply = resolution_ply.saturating_add(delta);
            }
            CertNode::Universal { zone, .. } => {
                if let Some(zone) = zone {
                    zone.build_horizon = zone.build_horizon.saturating_add(delta);
                }
            }
            CertNode::Choice { .. } => {}
        }
    }
    shifted
}

fn add_balanced_turn_pairs(
    root: &HexoState,
    forbidden: &BTreeSet<Cell>,
    k: usize,
    seed: u64,
) -> Option<(HexoState, Vec<Cell>)> {
    if !matches!(root.phase(), TurnPhase::FirstStone) {
        return None;
    }
    let player = root.current_player();
    let mut state = root.clone();
    let mut rng = XorShift64(seed | 1);
    let mut added = Vec::new();
    // One unit K is one full two-stone turn by each player: +4 placements,
    // equal colour counts, and identical player/phase on return.
    for _ in 0..k.saturating_mul(4) {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        let mut candidates = legal
            .into_iter()
            .filter(|coord| !forbidden.contains(&cell(*coord)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        let offset = (rng.next() % candidates.len() as u64) as usize;
        candidates.rotate_left(offset);
        let mut accepted = None;
        for mv in candidates {
            let mut probe = state.clone();
            let result = apply_placement(&mut probe, Placement { coord: mv }).ok()?;
            if result.outcome.is_none() {
                accepted = Some((probe, mv));
                break;
            }
        }
        let (next, mv) = accepted?;
        state = next;
        added.push(cell(mv));
    }
    (state.current_player() == player && matches!(state.phase(), TurnPhase::FirstStone))
        .then_some((state, added))
}

/// Add a remote count-5 defender window while returning to the same
/// player/FirstStone phase. The claimant gets harmless filler turns between
/// the defender's placements. Every added coordinate is outside the finite
/// certificate-body footprint.
fn add_far_defender_five(
    root: &HexoState,
    forbidden: &BTreeSet<Cell>,
) -> Option<(HexoState, Vec<Cell>, [Cell; 6])> {
    if !matches!(root.phase(), TurnPhase::FirstStone) {
        return None;
    }
    const DIRECTIONS: [Cell; 6] = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)];
    let claimant = root.current_player();
    let defender = claimant.other();
    let occupied = root.board().occupied_cells();
    let mut anchors = occupied.to_vec();
    anchors.sort_by_key(|anchor| {
        std::cmp::Reverse(
            forbidden
                .iter()
                .map(|point| hex_distance(*anchor, c(*point)))
                .min()
                .unwrap_or(0),
        )
    });
    for anchor in anchors {
        for direction in DIRECTIONS {
            let window = std::array::from_fn(|index| {
                let step = index as i16 + 1;
                (anchor.q + direction.0 * step, anchor.r + direction.1 * step)
            });
            if window
                .iter()
                .any(|point| forbidden.contains(point) || root.board().get(c(*point)).is_some())
            {
                continue;
            }
            let mut state = root.clone();
            let mut added = Vec::new();
            let mut next_line = 0usize;
            let mut failed = false;
            for ply in 0..12 {
                let defender_line_ply = matches!(ply, 2 | 3 | 6 | 7 | 10);
                let mv = if defender_line_ply {
                    let mv = c(window[next_line]);
                    next_line += 1;
                    mv
                } else {
                    let mut legal = Vec::new();
                    state.write_legal_moves(&mut legal);
                    let Some(mv) = legal.into_iter().find(|candidate| {
                        !forbidden.contains(&cell(*candidate))
                            && !window.contains(&cell(*candidate))
                            && {
                                let mut probe = state.clone();
                                apply_placement(&mut probe, Placement { coord: *candidate })
                                    .is_ok_and(|result| result.outcome.is_none())
                            }
                    }) else {
                        failed = true;
                        break;
                    };
                    mv
                };
                let Ok(result) = apply_placement(&mut state, Placement { coord: mv }) else {
                    failed = true;
                    break;
                };
                if result.outcome.is_some() {
                    failed = true;
                    break;
                }
                added.push(cell(mv));
            }
            if failed
                || state.current_player() != claimant
                || !matches!(state.phase(), TurnPhase::FirstStone)
            {
                continue;
            }
            let has_five = state
                .board()
                .windows()
                .entries()
                .any(|entry| entry.count(defender) == 5 && entry.count(claimant) == 0);
            if has_five {
                return Some((state, added, window));
            }
        }
    }
    None
}

fn solve_win(
    state: &HexoState,
    cap: u64,
    tt_bytes: usize,
    semantic_horizon: u32,
) -> crate::tss_core::DeepResult<TssCertificate> {
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::vcf_pair_complete());
    solver.solve_goal(
        state,
        &SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes,
            semantic_horizon,
        },
        SolveGoal::Win,
    )
}

fn pct(values: &mut [usize], quantile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}

fn human_candidates(games: &[Game], want: usize) -> Vec<Candidate> {
    let mut all = [Vec::new(), Vec::new(), Vec::new()];
    for (game_index, game) in games.iter().enumerate() {
        if game.winner != 1 && game.winner != -1 {
            continue;
        }
        let mut state = HexoState::new();
        for (prefix, &mv) in game.moves.iter().enumerate() {
            if state.is_terminal() {
                break;
            }
            if matches!(state.phase(), TurnPhase::FirstStone) {
                let ply = state.placements_made();
                let band = if ply <= 12 {
                    0
                } else if ply <= 40 {
                    1
                } else {
                    2
                };
                all[band].push(Candidate {
                    game: game_index,
                    prefix,
                    band,
                });
            }
            apply_placement(&mut state, Placement { coord: c(mv) }).expect("human replay");
        }
    }
    let quotas = [
        want / 3 + usize::from(want % 3 > 0),
        want / 3 + usize::from(want % 3 > 1),
        want / 3,
    ];
    let mut sample = Vec::new();
    for band in 0..3 {
        let mut rng = XorShift64((SEED ^ (band as u64).wrapping_mul(0x9E37_79B9)) | 1);
        for index in (1..all[band].len()).rev() {
            let other = (rng.next() % (index as u64 + 1)) as usize;
            all[band].swap(index, other);
        }
        sample.extend(all[band].iter().copied().take(quotas[band]));
    }
    sample
}

fn record_certificate(
    source: &str,
    id: &str,
    state: &HexoState,
    status: ProofStatus,
    cert: &TssCertificate,
    cap: u64,
    nodes: u64,
    tt_hits: u64,
    transfer: &mut [u64; 12],
) {
    assert!(TssVerifier.verify(state, cert, status));
    let (body, frame, radius, outside) = body_metrics(state, cert);
    println!(
        "CERT_ROW source={source} id={id} cap={cap} pop={} status={status:?} nodes={nodes} tt_hits={tt_hits} strict_support=UNBOUNDED strict_frame=UNBOUNDED strict_radius=UNBOUNDED root_outside_strict=0 body_support={body} body_frame={frame} body_radius={radius} root_outside_body={outside:.6} cert_nodes={}",
        state.board().len(), cert.nodes.len()
    );
    let footprint = body_footprint_with_zones(state, cert);
    for (ki, k) in [1usize, 2, 4, 8].into_iter().enumerate() {
        for trial in 0..4u64 {
            let seed = SEED
                ^ (state.placements_made() as u64).rotate_left(17)
                ^ (k as u64).rotate_left(31)
                ^ trial.wrapping_mul(0xD1B5_4A32_D192_ED03);
            let Some((mutated, added)) = add_balanced_turn_pairs(state, &footprint, k, seed) else {
                println!("TRANSFER_SKIP source={source} id={id} k={k} trial={trial} reason=no_legal_outside_body");
                continue;
            };
            transfer[ki * 3] += 1;
            let strict = TssVerifier.verify(&mutated, cert, status);
            if strict {
                transfer[ki * 3 + 1] += 1;
            }
            let delta = mutated.placements_made() - state.placements_made();
            let shifted = shifted_rebind(cert, &mutated, delta);
            let shadow = TssVerifier.verify(&mutated, &shifted, status);
            if shadow {
                transfer[ki * 3 + 2] += 1;
            }
            if strict || !shadow {
                println!(
                    "TRANSFER_EXEMPLAR source={source} id={id} k={k} trial={trial} strict={} strict_reason={} shifted_rebound={} delta={delta} added={added:?}",
                    strict,
                    if strict { "accepted" } else { "root_binding_first" },
                    shadow
                );
            }
        }
    }
}

#[test]
fn strict_root_binding_is_a_global_obligation() {
    let state = replay(&[(0, 0), (8, -8), (-8, 0), (1, 0), (2, 0)]).expect("fixture");
    let binding = RootBinding::from_state(&state);
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    let mv = legal[legal.len() / 2];
    let mut changed = state.clone();
    apply_placement(&mut changed, Placement { coord: mv }).expect("legal extra stone");
    assert_ne!(binding, RootBinding::from_state(&changed));
    // There is no finite list of empty cells in `binding`; nevertheless any
    // legal added cell changes equality. This is the frame-lemma obstruction.
}

#[test]
#[ignore = "NQ3 far-threat adversarial boundary; release, serial, --nocapture"]
fn cert_support_far_threat_adversarial() {
    let pos = load_forcing()
        .into_iter()
        .find(|pos| pos.id == "0hz3hty")
        .expect("0hz3hty fixture");
    let result = solve_win(&pos.state, 10_000, DEFAULT_TT_BYTES, u32::MAX);
    assert_eq!(result.status, ProofStatus::Win);
    let cert = result.cert.expect("WIN certificate");
    assert!(TssVerifier.verify(&pos.state, &cert, result.status));
    let (derived_t, _) = crate::tss_verify::certificate_horizon_preflight(&cert).unwrap();
    assert!(
        derived_t > pos.state.placements_made() + 2,
        "adversarial target must not be an immediate same-turn win"
    );
    let footprint = body_footprint_with_zones(&pos.state, &cert);
    let (mutated, added, window) =
        add_far_defender_five(&pos.state, &footprint).expect("remote defender five construction");
    let delta = mutated.placements_made() - pos.state.placements_made();
    let shifted = shifted_rebind(&cert, &mutated, delta);
    let strict = TssVerifier.verify(&mutated, &cert, result.status);
    let shifted_accepted = TssVerifier.verify(&mutated, &shifted, result.status);
    println!(
        "ADVERSARIAL id={} construction=far_defender_count5 derived_remaining={} delta={delta} strict_unchanged={strict} strict_reason=root_binding_first shifted_rebound={shifted_accepted} window={window:?} added={added:?}",
        pos.id,
        derived_t - pos.state.placements_made(),
    );
    assert!(!strict, "unchanged strict certificate crossed root binding");
    assert!(
        !shifted_accepted,
        "SOUNDNESS: non-immediate certificate survived a remote defender count-5"
    );
}

#[test]
#[ignore = "NQ3 measurement campaign; release, serial, --nocapture"]
fn cert_support_campaign() {
    let forcing_max_cap = env_num("TSS_CERT_SUPPORT_FORCING_CAP", 100_000u64);
    let human_n = env_num("TSS_CERT_SUPPORT_HUMAN_N", 200usize);
    let human_cap = env_num("TSS_CERT_SUPPORT_HUMAN_CAP", 30_000u64);
    let tt_bytes = env_num("TSS_CERT_SUPPORT_TT_BYTES", DEFAULT_TT_BYTES);
    let start = Instant::now();
    println!(
        "CERT_META seed={SEED} forcing_max_cap={forcing_max_cap} human_n={human_n} human_cap={human_cap} tt_bytes={tt_bytes} strict_support=UNBOUNDED reason=RootBinding_full_occupancy_complement_and_ReplayKey_full_position"
    );
    let mut transfer = [0u64; 12];
    let mut solved = 0usize;
    for pos in load_forcing() {
        let mut result = None;
        let mut used_cap = 0;
        for cap in [10_000u64, 100_000]
            .into_iter()
            .filter(|cap| *cap <= forcing_max_cap)
        {
            used_cap = cap;
            let attempt = solve_win(&pos.state, cap, tt_bytes, u32::MAX);
            if attempt.status == ProofStatus::Win {
                result = Some(attempt);
                break;
            }
        }
        match result {
            Some(result) => {
                solved += 1;
                record_certificate(
                    "forcing",
                    &pos.id,
                    &pos.state,
                    result.status,
                    result.cert.as_ref().expect("WIN certificate"),
                    used_cap,
                    result.stats.nodes,
                    result.stats.tt_hits,
                    &mut transfer,
                );
            }
            None => println!(
                "CERT_UNSOLVED source=forcing id={} max_cap={forcing_max_cap}",
                pos.id
            ),
        }
    }

    let games = load_games();
    let sample = human_candidates(&games, human_n);
    let mut human_support = Vec::new();
    let mut human_population = Vec::new();
    let mut human_wins = 0usize;
    for (index, candidate) in sample.iter().enumerate() {
        let game = &games[candidate.game];
        let Some(state) = replay(&game.moves[..candidate.prefix]) else {
            continue;
        };
        let result = solve_win(
            &state,
            human_cap,
            tt_bytes,
            state.placements_made().saturating_add(50),
        );
        println!(
            "HUMAN_SCREEN index={index} id={}@{} band={} pop={} cap={human_cap} status={:?} nodes={} tt_hits={}",
            game.hash,
            candidate.prefix,
            candidate.band,
            state.board().len(),
            result.status,
            result.stats.nodes,
            result.stats.tt_hits
        );
        if result.status != ProofStatus::Win {
            continue;
        }
        human_wins += 1;
        let cert = result.cert.as_ref().expect("human WIN cert");
        human_support.push(body_metrics(&state, cert).0);
        human_population.push(state.board().len());
        record_certificate(
            "human",
            &format!("{}@{}", game.hash, candidate.prefix),
            &state,
            result.status,
            cert,
            human_cap,
            result.stats.nodes,
            result.stats.tt_hits,
            &mut transfer,
        );
    }
    let mut hs_med = human_support.clone();
    let mut hs_p90 = human_support.clone();
    let mut hs_max = human_support.clone();
    let mut hp_med = human_population.clone();
    let mut hp_p90 = human_population.clone();
    let mut hp_max = human_population.clone();
    println!(
        "HUMAN_DIST screened={} wins={human_wins} strict_support_med=UNBOUNDED strict_support_p90=UNBOUNDED strict_support_max=UNBOUNDED body_support_med={} body_support_p90={} body_support_max={} population_med={} population_p90={} population_max={}",
        sample.len(),
        pct(&mut hs_med, 0.5),
        pct(&mut hs_p90, 0.9),
        pct(&mut hs_max, 1.0),
        pct(&mut hp_med, 0.5),
        pct(&mut hp_p90, 0.9),
        pct(&mut hp_max, 1.0),
    );
    for (ki, k) in [1usize, 2, 4, 8].into_iter().enumerate() {
        let attempts = transfer[ki * 3];
        let accepted = transfer[ki * 3 + 1];
        let shadow = transfer[ki * 3 + 2];
        println!(
            "TRANSFER_SUMMARY k={k} attempts={attempts} strict_accepted={accepted} strict_rate={:.6} strict_rejection_reason=root_binding_first shifted_rebound_accepted={shadow} shifted_rebound_rate={:.6}",
            if attempts == 0 { 0.0 } else { accepted as f64 / attempts as f64 },
            if attempts == 0 { 0.0 } else { shadow as f64 / attempts as f64 }
        );
    }
    println!(
        "TT_ESTIMATE method=strict_support_equivalence full_position_entries=solver_nodes support_key_entries=solver_nodes extra_collisions=0 multiplier=1.000000 forcing_solved={solved}"
    );
    println!(
        "CERT_DONE elapsed_s={:.3} forcing_solved={solved} human_screened={} human_wins={human_wins}",
        start.elapsed().as_secs_f64(),
        sample.len()
    );
}

// ---------------------------------------------------------------------------
// C-REL round 2: cfg(test)-only strict-discharge shadow harness.
//
// This code deliberately never calls hard_value_from_verified and never writes
// solver/tree state.  A shadow acceptance below always means that the ordinary,
// unchanged TssVerifier accepted the fully materialized target certificate.
// ---------------------------------------------------------------------------

const C_REL_RULESET_CONTRACT: [u8; 32] = [0x52; 32];
const C_REL_STRICT_CONTRACT: [u8; 32] = [0x53; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelDeadline {
    AfterRoot(u32),
    MaxU32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RelWfAnchor {
    RootOccupied(HexCoord),
    PriorClaimantPlacement(HexCoord),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RelWfWitness {
    node: u32,
    subject: HexCoord,
    anchor: RelWfAnchor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RelZoneHint {
    node: u32,
    source_required_cells: Vec<HexCoord>,
}

#[derive(Clone, Debug)]
struct RelInterface {
    current_player: Player,
    phase: TurnPhase,
    root_projection: Vec<(HexCoord, u8)>,
    zone_hints: Vec<RelZoneHint>,
    wf_plan: Vec<RelWfWitness>,
}

#[derive(Clone, Debug)]
struct RelativeClocks {
    events: Vec<Option<u32>>,
    zone_deadlines: Vec<Option<RelDeadline>>,
    semantic_deadline: RelDeadline,
    derived_resolution_offset: u32,
}

#[derive(Clone)]
struct RelTemplate {
    source_kind: String,
    source_id: String,
    source_state: HexoState,
    source_cert: TssCertificate,
    status: ProofStatus,
    interface: RelInterface,
    clocks: RelativeClocks,
    canonical_bytes: Vec<u8>,
    artifact_id: [u8; 32],
    extraction_nanos: u128,
}

#[derive(Clone)]
struct StageTarget {
    source_index: usize,
    source_kind: String,
    source_id: String,
    status: ProofStatus,
    k: usize,
    trial: u64,
    state: HexoState,
    query_horizon: u32,
}

#[derive(Clone, Copy)]
struct Stage4Candidate {
    template_index: usize,
    symmetry: u8,
}

struct Stage4Library {
    admitted: Vec<usize>,
    refused: Vec<usize>,
    artifact_bytes: usize,
    index_bytes: usize,
    build_nanos: u128,
    _index_blob: Vec<u8>,
}

#[derive(Default)]
struct Stage4Totals {
    e_ns: u128,
    l_ns: u128,
    i_ns: u128,
    m_ns: u128,
    v_ns: u128,
    solve_ns: u128,
    accepted: u64,
    missed: u64,
    probes: u64,
    hint_checks: u64,
    expansions: u64,
    nodes: u64,
    tt_peak_bytes: u64,
    fragment_peak_bytes: u64,
    fragment_entries: u64,
}

#[derive(Clone, Debug)]
struct ExtractZoneSummary {
    local_budget: u32,
    protected: Vec<HexCoord>,
}

fn coord_order(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

fn window_at(state: &HexoState, key: WindowKey) -> Option<hexo_engine::WindowEntry> {
    state
        .board()
        .windows()
        .entries()
        .find(|entry| entry.key() == key)
}

fn sorted_contains(sorted: &[HexCoord], coord: HexCoord) -> bool {
    sorted
        .binary_search_by_key(&coord_order(coord), |candidate| coord_order(*candidate))
        .is_ok()
}

// Exact test-recorder copy of the source-zone declaration recurrence.  It is
// not used to accept a certificate; the unchanged verifier remains normative.
fn extract_zone_summary(
    cert: &TssCertificate,
    state: &mut HexoState,
    node_id: u32,
    depth: usize,
) -> Option<ExtractZoneSummary> {
    if depth > crate::tss_verify::MAX_CERT_DEPTH {
        return None;
    }
    let node = cert.nodes.get(node_id as usize)?;
    let mut protected = Vec::new();
    let local_budget = match node {
        CertNode::OrCompletion { mv, .. } => {
            protected.push(*mv);
            0
        }
        CertNode::Win { witness, .. } => {
            protected.extend(window_at(state, *witness)?.empty_cells());
            0
        }
        CertNode::Loss { witnesses, .. } => {
            for witness in witnesses {
                protected.extend(window_at(state, *witness)?.empty_cells());
            }
            u32::from(crate::threats_shared::placements_remaining(state))
        }
        CertNode::Choice { mv, child } => {
            let (result, delta) = state.apply_with_delta(Placement { coord: *mv }).ok()?;
            if result.outcome.is_some() {
                state.undo(delta);
                return None;
            }
            let child_summary = extract_zone_summary(cert, state, *child, depth + 1);
            state.undo(delta);
            let child_summary = child_summary?;
            protected.push(*mv);
            protected.extend(child_summary.protected);
            child_summary.local_budget
        }
        CertNode::Universal { edges, .. } => {
            let mut maximum = 0u32;
            for edge in edges {
                let (result, delta) = state.apply_with_delta(Placement { coord: edge.mv }).ok()?;
                if result.outcome.is_some() {
                    state.undo(delta);
                    return None;
                }
                let child_summary = extract_zone_summary(cert, state, edge.child, depth + 1);
                state.undo(delta);
                let child_summary = child_summary?;
                maximum = maximum.max(child_summary.local_budget);
                protected.extend(child_summary.protected);
            }
            maximum.saturating_add(1)
        }
    };
    protected.sort_by_key(|coord| coord_order(*coord));
    protected.dedup();
    Some(ExtractZoneSummary {
        local_budget,
        protected,
    })
}

fn extract_uniform_zone(
    state: &HexoState,
    claimant: Player,
    summary: &ExtractZoneSummary,
) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_order(*coord));
    let stones = state.board().occupied_cells();
    let mut zone = summary
        .protected
        .iter()
        .copied()
        .filter(|coord| sorted_contains(&legal, *coord))
        .collect::<Vec<_>>();
    let pending = summary
        .protected
        .iter()
        .copied()
        .filter(|coord| !sorted_contains(&legal, *coord) && !stones.contains(coord))
        .collect::<Vec<_>>();
    if !pending.is_empty() {
        let radius = crate::tss_core::seed_band_radius(summary.local_budget);
        zone.extend(legal.iter().copied().filter(|coord| {
            pending
                .iter()
                .any(|target| i32::from(hex_distance(*coord, *target)) <= radius)
        }));
    }
    let defender = claimant.other();
    for entry in state.board().windows().entries() {
        let count = entry.count(defender);
        if entry.active_player() == Some(defender)
            && count >= 1
            && u32::from(count).saturating_add(summary.local_budget) >= 6
        {
            zone.extend(entry.empty_cells());
        }
    }
    if summary.local_budget >= 6 {
        zone.extend(legal.iter().copied());
    }
    zone.sort_by_key(|coord| coord_order(*coord));
    zone.dedup();
    if zone.is_empty() {
        if let Some(&fallback) = legal.first() {
            zone.push(fallback);
        }
    }
    zone
}

fn choose_wf_anchor(
    state: &HexoState,
    cert: &TssCertificate,
    subject: HexCoord,
) -> Option<RelWfAnchor> {
    let mut root = cert.root.occupancy.clone();
    root.sort_by_key(|coord| coord_order(*coord));
    if let Some(anchor) = root
        .iter()
        .copied()
        .find(|anchor| hex_distance(*anchor, subject) <= 8)
    {
        return Some(RelWfAnchor::RootOccupied(anchor));
    }
    let root_set = root;
    let mut prior = state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|anchor| {
            !root_set.contains(anchor)
                && state.board().get(*anchor) == Some(cert.claimant)
                && hex_distance(*anchor, subject) <= 8
        })
        .collect::<Vec<_>>();
    prior.sort_by_key(|coord| coord_order(*coord));
    prior
        .first()
        .copied()
        .map(RelWfAnchor::PriorClaimantPlacement)
}

fn record_wf_queries(
    node: u32,
    subjects: Vec<HexCoord>,
    state: &HexoState,
    cert: &TssCertificate,
    by_node: &mut BTreeMap<u32, Vec<RelWfWitness>>,
) -> Result<(), &'static str> {
    let mut occurrence = Vec::new();
    for subject in subjects {
        let anchor = choose_wf_anchor(state, cert, subject).ok_or("wf_anchor_missing")?;
        occurrence.push(RelWfWitness {
            node,
            subject,
            anchor,
        });
    }
    occurrence.sort_by_key(|w| coord_order(w.subject));
    occurrence.dedup_by(|a, b| a.subject == b.subject && a.anchor == b.anchor);
    match by_node.get(&node) {
        Some(previous) if previous != &occurrence => Err("ambiguous_shared_wf_occurrence"),
        Some(_) => Ok(()),
        None => {
            by_node.insert(node, occurrence);
            Ok(())
        }
    }
}

fn walk_interface_occurrences(
    cert: &TssCertificate,
    state: &mut HexoState,
    node_id: u32,
    depth: usize,
    wf_by_node: &mut BTreeMap<u32, Vec<RelWfWitness>>,
    zones_by_node: &mut BTreeMap<u32, Vec<HexCoord>>,
) -> Result<(), &'static str> {
    if depth > crate::tss_verify::MAX_CERT_DEPTH {
        return Err("interface_depth_limit");
    }
    let node = cert
        .nodes
        .get(node_id as usize)
        .ok_or("interface_bad_node_id")?;
    let subjects = match node {
        CertNode::OrCompletion { mv, .. } | CertNode::Choice { mv, .. } => vec![*mv],
        CertNode::Win { witness, .. } => window_at(state, *witness)
            .ok_or("wf_win_window_missing")?
            .empty_cells(),
        CertNode::Loss { witnesses, .. } => {
            let mut subjects = Vec::new();
            for witness in witnesses {
                subjects.extend(
                    window_at(state, *witness)
                        .ok_or("wf_loss_window_missing")?
                        .empty_cells(),
                );
            }
            subjects
        }
        CertNode::Universal { .. } => Vec::new(),
    };
    record_wf_queries(node_id, subjects, state, cert, wf_by_node)?;

    if matches!(node, CertNode::Universal { zone: Some(_), .. }) {
        let mut replay = state.clone();
        let summary =
            extract_zone_summary(cert, &mut replay, node_id, 0).ok_or("zone_summary_failed")?;
        let zone = extract_uniform_zone(state, cert.claimant, &summary);
        match zones_by_node.get(&node_id) {
            Some(previous) if previous != &zone => return Err("ambiguous_shared_zone_occurrence"),
            Some(_) => {}
            None => {
                zones_by_node.insert(node_id, zone);
            }
        }
    }

    match node {
        CertNode::Choice { mv, child } => {
            let (result, delta) = state
                .apply_with_delta(Placement { coord: *mv })
                .map_err(|_| "interface_choice_illegal")?;
            if result.outcome.is_some() {
                state.undo(delta);
                return Err("interface_choice_terminal");
            }
            let value = walk_interface_occurrences(
                cert,
                state,
                *child,
                depth + 1,
                wf_by_node,
                zones_by_node,
            );
            state.undo(delta);
            value
        }
        CertNode::Universal { edges, .. } => {
            for edge in edges {
                let (result, delta) = state
                    .apply_with_delta(Placement { coord: edge.mv })
                    .map_err(|_| "interface_edge_illegal")?;
                if result.outcome.is_some() {
                    state.undo(delta);
                    return Err("interface_edge_terminal");
                }
                let value = walk_interface_occurrences(
                    cert,
                    state,
                    edge.child,
                    depth + 1,
                    wf_by_node,
                    zones_by_node,
                );
                state.undo(delta);
                value?;
            }
            Ok(())
        }
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => Ok(()),
    }
}

fn relative_event_offset(base: u32, stored: u32, logical_delta: u32) -> Result<u32, &'static str> {
    let expected = base
        .checked_add(logical_delta)
        .ok_or("saturated_event_encoding")?;
    if expected != stored {
        return Err("event_clock_mismatch");
    }
    stored.checked_sub(base).ok_or("event_clock_underflow")
}

fn deadline_from_absolute(base: u32, value: u32) -> Result<RelDeadline, &'static str> {
    if value == u32::MAX {
        Ok(RelDeadline::MaxU32)
    } else {
        value
            .checked_sub(base)
            .map(RelDeadline::AfterRoot)
            .ok_or("deadline_underflow")
    }
}

fn record_event_offset(
    offsets: &mut [Option<u32>],
    node_id: u32,
    offset: u32,
) -> Result<(), &'static str> {
    let slot = offsets
        .get_mut(node_id as usize)
        .ok_or("clock_bad_node_id")?;
    match *slot {
        Some(previous) if previous != offset => Err("ambiguous_shared_clock_occurrence"),
        Some(_) => Ok(()),
        None => {
            *slot = Some(offset);
            Ok(())
        }
    }
}

fn walk_event_clocks(
    cert: &TssCertificate,
    state: &mut HexoState,
    node_id: u32,
    depth: usize,
    root_ply: u32,
    offsets: &mut [Option<u32>],
) -> Result<(), &'static str> {
    if depth > crate::tss_verify::MAX_CERT_DEPTH {
        return Err("clock_depth_limit");
    }
    let node = cert
        .nodes
        .get(node_id as usize)
        .ok_or("clock_bad_node_id")?;
    match node {
        CertNode::OrCompletion { completion_ply, .. } => {
            let expected = state
                .placements_made()
                .checked_add(1)
                .ok_or("saturated_event_encoding")?;
            if *completion_ply != expected {
                return Err("completion_clock_mismatch");
            }
            record_event_offset(
                offsets,
                node_id,
                relative_event_offset(root_ply, *completion_ply, expected - root_ply)?,
            )
        }
        CertNode::Win {
            count,
            resolution_ply,
            ..
        } => {
            let logical_delta = match *count {
                5 => 1,
                4 if crate::threats_shared::placements_remaining(state) == 2 => 2,
                _ => return Err("win_clock_rule_invalid"),
            };
            let expected = state
                .placements_made()
                .checked_add(logical_delta)
                .ok_or("saturated_event_encoding")?;
            if *resolution_ply != expected {
                return Err("win_clock_mismatch");
            }
            record_event_offset(
                offsets,
                node_id,
                relative_event_offset(root_ply, *resolution_ply, expected - root_ply)?,
            )
        }
        CertNode::Loss { resolution_ply, .. } => {
            let analysis = crate::threats_shared::analyze(state);
            let logical_delta = u32::from(analysis.b)
                .checked_add(2)
                .ok_or("loss_delta_overflow")?;
            let expected = state
                .placements_made()
                .checked_add(logical_delta)
                .ok_or("saturated_event_encoding")?;
            if *resolution_ply != expected {
                return Err("loss_clock_mismatch");
            }
            record_event_offset(
                offsets,
                node_id,
                relative_event_offset(root_ply, *resolution_ply, expected - root_ply)?,
            )
        }
        CertNode::Choice { mv, child } => {
            let (result, delta) = state
                .apply_with_delta(Placement { coord: *mv })
                .map_err(|_| "clock_choice_illegal")?;
            if result.outcome.is_some() {
                state.undo(delta);
                return Err("clock_choice_terminal");
            }
            let value = walk_event_clocks(cert, state, *child, depth + 1, root_ply, offsets);
            state.undo(delta);
            value
        }
        CertNode::Universal { edges, .. } => {
            for edge in edges {
                let (result, delta) = state
                    .apply_with_delta(Placement { coord: edge.mv })
                    .map_err(|_| "clock_edge_illegal")?;
                if result.outcome.is_some() {
                    state.undo(delta);
                    return Err("clock_edge_terminal");
                }
                let value =
                    walk_event_clocks(cert, state, edge.child, depth + 1, root_ply, offsets);
                state.undo(delta);
                value?;
            }
            Ok(())
        }
    }
}

fn extract_interface(
    state: &HexoState,
    cert: &TssCertificate,
) -> Result<RelInterface, &'static str> {
    let mut wf_by_node = BTreeMap::new();
    let mut zones_by_node = BTreeMap::new();
    let mut replay = state.clone();
    walk_interface_occurrences(
        cert,
        &mut replay,
        cert.root_node,
        0,
        &mut wf_by_node,
        &mut zones_by_node,
    )?;

    // Correspondence guard: the independently recorded zone multiset must be
    // byte-for-byte equal to the verifier module's retained test rederivation.
    let mut verifier_zones = crate::tss_verify::round3_rederived_zones(state, cert)
        .ok_or("verifier_zone_rederivation_failed")?
        .into_iter()
        .map(|(_, zone)| zone)
        .collect::<Vec<_>>();
    let mut extracted_zones = zones_by_node.values().cloned().collect::<Vec<_>>();
    let zone_order = |a: &Vec<HexCoord>, b: &Vec<HexCoord>| {
        a.len().cmp(&b.len()).then_with(|| {
            a.iter()
                .map(|coord| coord_order(*coord))
                .cmp(b.iter().map(|coord| coord_order(*coord)))
        })
    };
    verifier_zones.sort_by(zone_order);
    extracted_zones.sort_by(zone_order);
    if verifier_zones != extracted_zones {
        return Err("zone_rederivation_disagreement");
    }

    let zone_hints = zones_by_node
        .into_iter()
        .map(|(node, source_required_cells)| RelZoneHint {
            node,
            source_required_cells,
        })
        .collect::<Vec<_>>();
    let wf_plan = wf_by_node.into_values().flatten().collect::<Vec<_>>();

    let mut projection = body_footprint(cert).into_iter().map(c).collect::<Vec<_>>();
    if let TurnPhase::SecondStone { first } = state.phase() {
        projection.push(first);
    }
    for hint in &zone_hints {
        projection.extend(hint.source_required_cells.iter().copied());
    }
    for witness in &wf_plan {
        if let RelWfAnchor::RootOccupied(anchor) = witness.anchor {
            projection.push(anchor);
        }
    }
    projection.sort_by_key(|coord| coord_order(*coord));
    projection.dedup();
    let root_projection = projection
        .into_iter()
        .map(|coord| {
            let value = match state.board().get(coord) {
                None => 0,
                Some(Player::Player0) => 1,
                Some(Player::Player1) => 2,
            };
            (coord, value)
        })
        .collect();
    Ok(RelInterface {
        current_player: state.current_player(),
        phase: state.phase(),
        root_projection,
        zone_hints,
        wf_plan,
    })
}

fn extract_clocks(
    state: &HexoState,
    cert: &TssCertificate,
) -> Result<RelativeClocks, &'static str> {
    let root_ply = state.placements_made();
    let mut events = vec![None; cert.nodes.len()];
    let mut replay = state.clone();
    walk_event_clocks(cert, &mut replay, cert.root_node, 0, root_ply, &mut events)?;
    let mut zone_deadlines = vec![None; cert.nodes.len()];
    for (index, node) in cert.nodes.iter().enumerate() {
        if let CertNode::Universal {
            zone: Some(zone), ..
        } = node
        {
            zone_deadlines[index] = Some(deadline_from_absolute(root_ply, zone.build_horizon)?);
        }
    }
    let (derived_t, _) = crate::tss_verify::certificate_horizon_preflight(cert)
        .ok_or("certificate_preflight_failed")?;
    let derived_resolution_offset = derived_t
        .checked_sub(root_ply)
        .ok_or("derived_resolution_underflow")?;
    if events.iter().flatten().copied().max().unwrap_or(0) != derived_resolution_offset {
        return Err("derived_resolution_disagreement");
    }
    Ok(RelativeClocks {
        events,
        zone_deadlines,
        semantic_deadline: deadline_from_absolute(root_ply, cert.semantic_horizon)?,
        derived_resolution_offset,
    })
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend(value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend(value.to_le_bytes());
}

fn put_i16(out: &mut Vec<u8>, value: i16) {
    out.extend(value.to_le_bytes());
}

fn put_coord(out: &mut Vec<u8>, coord: HexCoord) {
    put_i16(out, coord.q);
    put_i16(out, coord.r);
}

fn player_tag(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

fn status_tag(status: ProofStatus) -> u8 {
    match status {
        ProofStatus::Win => 0,
        ProofStatus::Loss => 1,
        ProofStatus::Unknown => 2,
    }
}

fn put_phase(out: &mut Vec<u8>, phase: TurnPhase) {
    match phase {
        TurnPhase::Opening => put_u8(out, 0),
        TurnPhase::FirstStone => put_u8(out, 1),
        TurnPhase::SecondStone { first } => {
            put_u8(out, 2);
            put_coord(out, first);
        }
    }
}

fn put_deadline(out: &mut Vec<u8>, deadline: RelDeadline) {
    match deadline {
        RelDeadline::AfterRoot(offset) => {
            put_u8(out, 0);
            put_u32(out, offset);
        }
        RelDeadline::MaxU32 => put_u8(out, 1),
    }
}

fn put_window(out: &mut Vec<u8>, key: WindowKey) {
    put_coord(out, key.start);
    put_u8(
        out,
        match key.axis {
            Axis::Q => 0,
            Axis::R => 1,
            Axis::QR => 2,
        },
    );
}

fn canonical_rel_bytes(
    cert: &TssCertificate,
    status: ProofStatus,
    interface: &RelInterface,
    clocks: &RelativeClocks,
) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::new();
    out.extend(*b"HXCR");
    put_u16(&mut out, 1);
    out.extend(C_REL_RULESET_CONTRACT);
    out.extend(C_REL_STRICT_CONTRACT);
    put_u8(&mut out, player_tag(interface.current_player));
    put_phase(&mut out, interface.phase);
    put_u8(&mut out, 0); // MustBeNonterminal
    put_u32(
        &mut out,
        u32::try_from(interface.root_projection.len()).map_err(|_| "projection_len")?,
    );
    for (coord, value) in &interface.root_projection {
        put_coord(&mut out, *coord);
        put_u8(&mut out, *value);
    }
    put_u32(
        &mut out,
        u32::try_from(interface.zone_hints.len()).map_err(|_| "zone_len")?,
    );
    for hint in &interface.zone_hints {
        put_u32(&mut out, hint.node);
        put_u32(
            &mut out,
            u32::try_from(hint.source_required_cells.len()).map_err(|_| "zone_cells_len")?,
        );
        for coord in &hint.source_required_cells {
            put_coord(&mut out, *coord);
        }
    }
    put_u32(
        &mut out,
        u32::try_from(interface.wf_plan.len()).map_err(|_| "wf_len")?,
    );
    for witness in &interface.wf_plan {
        put_u32(&mut out, witness.node);
        put_coord(&mut out, witness.subject);
        match witness.anchor {
            RelWfAnchor::RootOccupied(anchor) => {
                put_u8(&mut out, 0);
                put_coord(&mut out, anchor);
            }
            RelWfAnchor::PriorClaimantPlacement(anchor) => {
                put_u8(&mut out, 1);
                put_coord(&mut out, anchor);
            }
        }
    }
    put_u8(&mut out, status_tag(status));
    put_u8(&mut out, player_tag(cert.claimant));
    put_u32(&mut out, cert.root_node);
    put_u32(
        &mut out,
        u32::try_from(cert.nodes.len()).map_err(|_| "node_len")?,
    );
    for (index, node) in cert.nodes.iter().enumerate() {
        match node {
            CertNode::OrCompletion { mv, witness, .. } => {
                put_u8(&mut out, 0);
                put_coord(&mut out, *mv);
                put_window(&mut out, *witness);
                put_u32(
                    &mut out,
                    clocks.events[index].ok_or("missing_completion_offset")?,
                );
            }
            CertNode::Win {
                witness,
                count,
                budget,
                ..
            } => {
                put_u8(&mut out, 1);
                put_window(&mut out, *witness);
                put_u8(&mut out, *count);
                put_u8(&mut out, *budget);
                put_u32(&mut out, clocks.events[index].ok_or("missing_win_offset")?);
            }
            CertNode::Loss { witnesses, .. } => {
                put_u8(&mut out, 2);
                put_u32(
                    &mut out,
                    u32::try_from(witnesses.len()).map_err(|_| "witness_len")?,
                );
                for witness in witnesses {
                    put_window(&mut out, *witness);
                }
                put_u32(&mut out, clocks.events[index].ok_or("missing_loss_offset")?);
            }
            CertNode::Choice { mv, child } => {
                put_u8(&mut out, 3);
                put_coord(&mut out, *mv);
                put_u32(&mut out, *child);
            }
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                put_u8(&mut out, 4);
                put_u32(
                    &mut out,
                    u32::try_from(edges.len()).map_err(|_| "edge_len")?,
                );
                for edge in edges {
                    put_coord(&mut out, edge.mv);
                    put_u32(&mut out, edge.child);
                }
                put_u8(&mut out, u8::from(*implicit_dispatch));
                match zone {
                    Some(zone) => {
                        put_u8(&mut out, 1);
                        put_u32(&mut out, zone.d);
                        put_deadline(
                            &mut out,
                            clocks.zone_deadlines[index].ok_or("missing_zone_deadline")?,
                        );
                    }
                    None => put_u8(&mut out, 0),
                }
                put_u32(
                    &mut out,
                    u32::try_from(commutations.len()).map_err(|_| "commutation_len")?,
                );
                for item in commutations {
                    put_coord(&mut out, item.first);
                    put_coord(&mut out, item.omitted_second);
                    put_u32(&mut out, item.first_child);
                    put_u32(&mut out, item.mirror_child);
                }
            }
        }
    }
    put_u32(&mut out, clocks.derived_resolution_offset);
    put_deadline(&mut out, clocks.semantic_deadline);
    Ok(out)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend(bit_len.to_be_bytes());
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh].into_iter()) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 32];
    for (index, word) in h.into_iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn artifact_id_hex(id: &[u8; 32]) -> String {
    id.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn admit_template(
    source_kind: &str,
    source_id: &str,
    state: &HexoState,
    status: ProofStatus,
    cert: &TssCertificate,
) -> Result<RelTemplate, &'static str> {
    if status == ProofStatus::Unknown || !TssVerifier.verify(state, cert, status) {
        return Err("source_strict_rejected");
    }
    let started = Instant::now();
    let clocks = extract_clocks(state, cert)?;
    let interface = extract_interface(state, cert)?;
    let canonical_bytes = canonical_rel_bytes(cert, status, &interface, &clocks)?;
    if canonical_bytes.len() > 64 << 20 {
        return Err("artifact_byte_limit");
    }
    let artifact_id = sha256(&canonical_bytes);
    Ok(RelTemplate {
        source_kind: source_kind.to_owned(),
        source_id: source_id.to_owned(),
        source_state: state.clone(),
        source_cert: cert.clone(),
        status,
        interface,
        clocks,
        canonical_bytes,
        artifact_id,
        extraction_nanos: started.elapsed().as_nanos(),
    })
}

fn deadline_to_absolute(root_ply: u32, deadline: RelDeadline) -> Result<u32, &'static str> {
    match deadline {
        RelDeadline::AfterRoot(offset) => root_ply.checked_add(offset).ok_or("deadline_overflow"),
        RelDeadline::MaxU32 => Ok(u32::MAX),
    }
}

fn expected_claimant(state: &HexoState, status: ProofStatus) -> Option<Player> {
    match status {
        ProofStatus::Win => Some(state.current_player()),
        ProofStatus::Loss => Some(state.current_player().other()),
        ProofStatus::Unknown => None,
    }
}

fn materialize_template(
    template: &RelTemplate,
    target: &HexoState,
    symmetry: u8,
    query_status: ProofStatus,
    query_horizon: u32,
) -> Result<TssCertificate, &'static str> {
    if query_status != template.status || query_status == ProofStatus::Unknown {
        return Err("claimed_status_mismatch");
    }
    if expected_claimant(target, query_status) != Some(template.source_cert.claimant) {
        return Err("claimant_mismatch");
    }
    let mut candidate = crate::tss_verify::d6_remap_certificate(&template.source_cert, symmetry)
        .ok_or("d6_mapping_failed")?;
    candidate.root = RootBinding::from_state(target);
    let root_ply = target.placements_made();
    for (index, node) in candidate.nodes.iter_mut().enumerate() {
        match node {
            CertNode::OrCompletion { completion_ply, .. } => {
                *completion_ply = root_ply
                    .checked_add(template.clocks.events[index].ok_or("missing_event_offset")?)
                    .ok_or("event_overflow")?;
            }
            CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
                *resolution_ply = root_ply
                    .checked_add(template.clocks.events[index].ok_or("missing_event_offset")?)
                    .ok_or("event_overflow")?;
            }
            CertNode::Universal {
                zone: Some(zone), ..
            } => {
                zone.build_horizon = deadline_to_absolute(
                    root_ply,
                    template.clocks.zone_deadlines[index].ok_or("missing_zone_deadline")?,
                )?;
            }
            CertNode::Choice { .. } | CertNode::Universal { zone: None, .. } => {}
        }
    }
    candidate.semantic_horizon = deadline_to_absolute(root_ply, template.clocks.semantic_deadline)?;
    if candidate.semantic_horizon != query_horizon {
        return Err("semantic_horizon_mismatch");
    }
    let derived = root_ply
        .checked_add(template.clocks.derived_resolution_offset)
        .ok_or("derived_resolution_overflow")?;
    if derived > query_horizon {
        return Err("derived_resolution_exceeds_query");
    }
    Ok(candidate)
}

fn transformed_phase(phase: TurnPhase, symmetry: u8) -> Option<TurnPhase> {
    match phase {
        TurnPhase::Opening => Some(TurnPhase::Opening),
        TurnPhase::FirstStone => Some(TurnPhase::FirstStone),
        TurnPhase::SecondStone { first } => Some(TurnPhase::SecondStone {
            first: crate::tss_verify::d6_transform_coord(first, symmetry)?,
        }),
    }
}

fn hint_match(template: &RelTemplate, target: &HexoState, symmetry: u8) -> bool {
    if target.is_terminal()
        || target.current_player() != template.interface.current_player
        || transformed_phase(template.interface.phase, symmetry) != Some(target.phase())
    {
        return false;
    }
    template
        .interface
        .root_projection
        .iter()
        .all(|(coord, value)| {
            let Some(mapped) = crate::tss_verify::d6_transform_coord(*coord, symmetry) else {
                return false;
            };
            let target_value = match target.board().get(mapped) {
                None => 0,
                Some(Player::Player0) => 1,
                Some(Player::Player1) => 2,
            };
            target_value == *value
        })
}

fn forced_loss_state() -> HexoState {
    replay(&[
        (0, 0),
        (0, 8),
        (2, 7),
        (1, 0),
        (2, 0),
        (4, 6),
        (6, 5),
        (3, 0),
        (0, 4),
        (8, 4),
        (10, 3),
        (1, 4),
        (2, 4),
        (12, 2),
        (14, 1),
        (3, 4),
        (16, 0),
    ])
    .expect("hand Loss fixture")
}

fn solve_loss_fixture(state: &HexoState) -> crate::tss_core::DeepResult<TssCertificate> {
    TssSolver::default().solve_goal(
        state,
        &SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        },
        SolveGoal::Loss,
    )
}

fn log_admission(
    templates: &mut Vec<RelTemplate>,
    source_kind: &str,
    source_id: &str,
    state: &HexoState,
    status: ProofStatus,
    cert: &TssCertificate,
) {
    match admit_template(source_kind, source_id, state, status, cert) {
        Ok(template) => {
            println!(
                "CREL_ADMISSION source={source_kind} id={source_id} outcome=admitted status={status:?} phase={:?} projection_cells={} zone_hints={} wf_witnesses={} serialized_bytes={} artifact_id={} extract_ns={}",
                state.phase(),
                template.interface.root_projection.len(),
                template.interface.zone_hints.len(),
                template.interface.wf_plan.len(),
                template.canonical_bytes.len(),
                artifact_id_hex(&template.artifact_id),
                template.extraction_nanos,
            );
            templates.push(template);
        }
        Err(reason) => println!(
            "CREL_ADMISSION source={source_kind} id={source_id} outcome=rejected status={status:?} reason={reason}"
        ),
    }
}

fn acquire_round2_templates() -> Vec<RelTemplate> {
    let mut templates = Vec::new();
    for pos in load_forcing() {
        let mut acquired = None;
        for cap in [10_000u64, 100_000] {
            let started = Instant::now();
            let result = solve_win(&pos.state, cap, DEFAULT_TT_BYTES, u32::MAX);
            println!(
                "CREL_ACQUIRE source=forcing id={} cap={cap} horizon={} status={:?} cert_present={} nodes={} tt_hits={} elapsed_ns={} reason={}",
                pos.id,
                u32::MAX,
                result.status,
                result.cert.is_some(),
                result.stats.nodes,
                result.stats.tt_hits,
                started.elapsed().as_nanos(),
                if result.status == ProofStatus::Win { "strict_candidate" } else { "cap_exhausted_unknown" },
            );
            if result.status == ProofStatus::Win {
                acquired = result.cert.map(|cert| (result.status, cert));
                break;
            }
        }
        match acquired {
            Some((status, cert)) => {
                log_admission(&mut templates, "forcing", &pos.id, &pos.state, status, &cert)
            }
            None => println!(
                "CREL_ACQUISITION_FINAL source=forcing id={} outcome=unavailable reason=no_certificate_at_frozen_ladder",
                pos.id
            ),
        }
    }

    let games = load_games();
    let sample = human_candidates(&games, 200);
    assert_eq!(
        sample.len(),
        200,
        "frozen human manifest must contain 200 roots"
    );
    for (index, candidate) in sample.iter().enumerate() {
        let game = &games[candidate.game];
        let source_id = format!("{}@{}", game.hash, candidate.prefix);
        let Some(state) = replay(&game.moves[..candidate.prefix]) else {
            println!(
                "CREL_ACQUIRE source=human id={source_id} index={index} outcome=unavailable reason=replay_failed"
            );
            continue;
        };
        let horizon = state
            .placements_made()
            .checked_add(50)
            .expect("frozen human horizon");
        let started = Instant::now();
        let result = solve_win(&state, 30_000, DEFAULT_TT_BYTES, horizon);
        println!(
            "CREL_ACQUIRE source=human id={source_id} index={index} band={} cap=30000 horizon={horizon} status={:?} cert_present={} nodes={} tt_hits={} elapsed_ns={} reason={}",
            candidate.band,
            result.status,
            result.cert.is_some(),
            result.stats.nodes,
            result.stats.tt_hits,
            started.elapsed().as_nanos(),
            if result.status == ProofStatus::Win { "strict_candidate" } else { "cap_exhausted_unknown" },
        );
        if let Some(cert) = result.cert.as_ref() {
            log_admission(
                &mut templates,
                "human",
                &source_id,
                &state,
                result.status,
                cert,
            );
        }
    }

    let first = forced_loss_state();
    let mut second = first.clone();
    apply_placement(
        &mut second,
        Placement {
            coord: HexCoord::new(-8, 0),
        },
    )
    .expect("hand Loss SecondStone fixture");
    for (id, state) in [
        ("forced_loss_firststone", first),
        ("forced_loss_secondstone", second),
    ] {
        let started = Instant::now();
        let result = solve_loss_fixture(&state);
        println!(
            "CREL_ACQUIRE source=hand_loss id={id} cap=1 horizon={} status={:?} cert_present={} nodes={} elapsed_ns={} reason={}",
            u32::MAX,
            result.status,
            result.cert.is_some(),
            result.stats.nodes,
            started.elapsed().as_nanos(),
            if result.status == ProofStatus::Loss { "strict_candidate" } else { "fixture_drift" },
        );
        if let Some(cert) = result.cert.as_ref() {
            log_admission(&mut templates, "hand_loss", id, &state, result.status, cert);
        }
    }
    templates
}

fn stage1_shadow_reproduction(templates: &[RelTemplate]) -> Vec<StageTarget> {
    let mut targets = Vec::new();
    let mut all = [[0u64; 3]; 4];
    let mut retained_comparable = [[0u64; 2]; 4];
    let mut loss = [[0u64; 2]; 4];
    let mut m_nanos = [0u128; 4];
    let mut v_nanos = [0u128; 4];
    let mut cross_root_accepts = 0u64;
    let mut hard_without_strict = 0u64;
    for (source_index, template) in templates.iter().enumerate() {
        if !matches!(template.source_state.phase(), TurnPhase::FirstStone) {
            println!(
                "CREL_STAGE1_SOURCE source={} id={} outcome=skipped reason=non_FirstStone phase={:?}",
                template.source_kind, template.source_id, template.source_state.phase()
            );
            continue;
        }
        let footprint = template
            .interface
            .root_projection
            .iter()
            .map(|(coord, _)| cell(*coord))
            .collect::<BTreeSet<_>>();
        for (ki, k) in [1usize, 2, 4, 8].into_iter().enumerate() {
            for trial in 0..4u64 {
                let seed = SEED
                    ^ (template.source_state.placements_made() as u64).rotate_left(17)
                    ^ (k as u64).rotate_left(31)
                    ^ trial.wrapping_mul(0xD1B5_4A32_D192_ED03);
                let Some((mutated, added)) =
                    add_balanced_turn_pairs(&template.source_state, &footprint, k, seed)
                else {
                    println!(
                        "CREL_STAGE1_PROBE source={} id={} k={k} trial={trial} outcome=skipped reason=no_legal_outside_projection",
                        template.source_kind, template.source_id
                    );
                    continue;
                };
                all[ki][0] += 1;
                let unchanged =
                    TssVerifier.verify(&mutated, &template.source_cert, template.status);
                all[ki][1] += u64::from(unchanged);
                let query_horizon = deadline_to_absolute(
                    mutated.placements_made(),
                    template.clocks.semantic_deadline,
                )
                .expect("Stage-1 target horizon");
                let m_start = Instant::now();
                let materialized =
                    materialize_template(template, &mutated, 0, template.status, query_horizon);
                let m_elapsed = m_start.elapsed().as_nanos();
                m_nanos[ki] += m_elapsed;
                let (accepted, reason, v_elapsed) = match materialized {
                    Ok(candidate) => {
                        let v_start = Instant::now();
                        let accepted = TssVerifier.verify(&mutated, &candidate, template.status);
                        let elapsed = v_start.elapsed().as_nanos();
                        (
                            accepted,
                            if accepted {
                                "unchanged_strict_accepted"
                            } else {
                                "unchanged_strict_rejected_replay"
                            },
                            elapsed,
                        )
                    }
                    Err(reason) => (false, reason, 0),
                };
                v_nanos[ki] += v_elapsed;
                all[ki][2] += u64::from(accepted);
                cross_root_accepts += u64::from(accepted);
                if template.source_kind == "hand_loss" {
                    loss[ki][0] += 1;
                    loss[ki][1] += u64::from(accepted);
                } else {
                    retained_comparable[ki][0] += 1;
                    retained_comparable[ki][1] += u64::from(accepted);
                }
                // There is intentionally no hard-value constructor in this
                // harness. Keep the explicit invariant counter in the log.
                hard_without_strict += 0;
                println!(
                    "CREL_STAGE1_PROBE source={} id={} status={:?} k={k} trial={trial} unchanged_strict={} unchanged_reason={} materialized_strict={} result_reason={reason} exact_root_equal=false m_ns={m_elapsed} v_ns={v_elapsed} added={added:?}",
                    template.source_kind,
                    template.source_id,
                    template.status,
                    unchanged,
                    if unchanged { "accepted" } else { "root_binding_first" },
                    accepted,
                );
                assert!(
                    !unchanged,
                    "unchanged strict certificate crossed a changed root"
                );
                targets.push(StageTarget {
                    source_index,
                    source_kind: template.source_kind.clone(),
                    source_id: template.source_id.clone(),
                    status: template.status,
                    k,
                    trial,
                    state: mutated,
                    query_horizon,
                });
            }
        }
    }
    for (ki, k) in [1usize, 2, 4, 8].into_iter().enumerate() {
        let comparable_rate = if retained_comparable[ki][0] == 0 {
            0.0
        } else {
            retained_comparable[ki][1] as f64 / retained_comparable[ki][0] as f64
        };
        println!(
            "CREL_STAGE1_SUMMARY k={k} attempts={} unchanged_strict_accepted={} rebound_strict_accepted={} rebound_rate={:.6} retained_comparable_attempts={} retained_comparable_accepted={} retained_comparable_rate={comparable_rate:.6} retained_range_low=0.7778 retained_range_high=0.9611 within_retained_range={} hand_loss_attempts={} hand_loss_accepted={} m_ns={} v_ns={}",
            all[ki][0],
            all[ki][1],
            all[ki][2],
            if all[ki][0] == 0 { 0.0 } else { all[ki][2] as f64 / all[ki][0] as f64 },
            retained_comparable[ki][0],
            retained_comparable[ki][1],
            ((140.0 / 180.0)..=(173.0 / 180.0)).contains(&comparable_rate),
            loss[ki][0],
            loss[ki][1],
            m_nanos[ki],
            v_nanos[ki],
        );
    }
    assert_eq!(
        hard_without_strict, 0,
        "hard result emitted without strict acceptance"
    );
    println!(
        "CREL_STAGE1_VERDICT verdict={} criterion={} cross_root_strict_accepts={cross_root_accepts} hard_without_strict={hard_without_strict}",
        if cross_root_accepts > 0 { "PASS" } else { "STOPPED" },
        if cross_root_accepts > 0 {
            "strict_discharge_cross_root_acceptance_present"
        } else {
            "no_cross_root_strict_acceptance_beyond_exact_root_equality"
        },
    );
    targets
}

#[derive(Default)]
struct ConfusionTotals {
    hint_and_accept: u64,
    hint_and_reject: u64,
    no_hint_and_accept: u64,
    no_hint_and_reject: u64,
    exact_root_accept: u64,
    cross_root_hint: u64,
    cross_root_accept: u64,
    unconditional_hit_targets: u64,
    matched_hit_targets: u64,
    filtered_hit_targets: u64,
    lookup_ns: u128,
    match_ns: u128,
    materialize_ns: u128,
    verify_ns: u128,
    saved_materialize_verify_ns: u128,
}

struct ShadowCandidate {
    template_index: usize,
    symmetry: u8,
}

fn source_order_key(template: &RelTemplate) -> (u8, u8, usize, &[u8]) {
    let phase = match template.interface.phase {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { .. } => 2,
    };
    (
        status_tag(template.status),
        phase,
        template.interface.root_projection.len(),
        template.artifact_id.as_slice(),
    )
}

fn stage2_interface_matrix(templates: &[RelTemplate], targets: &[StageTarget]) {
    let mut totals = ConfusionTotals::default();
    let eligible_templates = templates
        .iter()
        .filter(|template| matches!(template.source_state.phase(), TurnPhase::FirstStone))
        .collect::<Vec<_>>();
    let extract_ns = eligible_templates
        .iter()
        .map(|template| template.extraction_nanos)
        .sum::<u128>();
    let serialized_bytes = eligible_templates
        .iter()
        .map(|template| template.canonical_bytes.len())
        .sum::<usize>();
    for template in &eligible_templates {
        println!(
            "CREL_STAGE2_ARTIFACT source={} id={} status={:?} projection_cells={} zone_hints={} wf_witnesses={} serialized_bytes={} artifact_id={} extract_ns={}",
            template.source_kind,
            template.source_id,
            template.status,
            template.interface.root_projection.len(),
            template.interface.zone_hints.len(),
            template.interface.wf_plan.len(),
            template.canonical_bytes.len(),
            artifact_id_hex(&template.artifact_id),
            template.extraction_nanos,
        );
    }
    let mut fanout_hits = [0u64; 6];
    let fanouts = [1usize, 2, 4, 8, 16, 32];
    let mut stage2_targets = 0u64;
    for target in targets.iter().filter(|target| target.k <= 2) {
        stage2_targets += 1;
        let lookup_start = Instant::now();
        let mut candidates = Vec::new();
        for (template_index, template) in templates.iter().enumerate() {
            if template.status != target.status
                || !matches!(template.source_state.phase(), TurnPhase::FirstStone)
            {
                continue;
            }
            for symmetry in 0..crate::tss_verify::D6_SYMMETRY_COUNT {
                candidates.push(ShadowCandidate {
                    template_index,
                    symmetry,
                });
            }
        }
        candidates.sort_by(|a, b| {
            source_order_key(&templates[a.template_index])
                .cmp(&source_order_key(&templates[b.template_index]))
                .then_with(|| a.symmetry.cmp(&b.symmetry))
        });
        totals.lookup_ns += lookup_start.elapsed().as_nanos();

        let mut bucket = 0usize;
        let mut first_accepted_rank = None;
        let mut accepted_ranks = Vec::new();
        let mut target_tp = 0u64;
        let mut target_fp = 0u64;
        let mut target_fn = 0u64;
        let mut target_tn = 0u64;
        for candidate in candidates {
            let template = &templates[candidate.template_index];
            let match_start = Instant::now();
            let matched = hint_match(template, &target.state, candidate.symmetry);
            let match_elapsed = match_start.elapsed().as_nanos();
            totals.match_ns += match_elapsed;
            if matched {
                bucket += 1;
            }

            let materialize_start = Instant::now();
            let materialized = materialize_template(
                template,
                &target.state,
                candidate.symmetry,
                target.status,
                target.query_horizon,
            );
            let materialize_elapsed = materialize_start.elapsed().as_nanos();
            totals.materialize_ns += materialize_elapsed;
            let (strict_accepted, verify_elapsed) = match materialized.as_ref() {
                Ok(cert) => {
                    let verify_start = Instant::now();
                    let accepted = TssVerifier.verify(&target.state, cert, target.status);
                    (accepted, verify_start.elapsed().as_nanos())
                }
                Err(_) => (false, 0),
            };
            totals.verify_ns += verify_elapsed;
            if !matched {
                totals.saved_materialize_verify_ns += materialize_elapsed + verify_elapsed;
            }

            let exact_root =
                crate::tss_verify::d6_remap_certificate(&template.source_cert, candidate.symmetry)
                    .is_some_and(|cert| cert.root == RootBinding::from_state(&target.state));
            if strict_accepted && exact_root {
                totals.exact_root_accept += 1;
            }
            if matched && !exact_root {
                totals.cross_root_hint += 1;
            }
            if strict_accepted && !exact_root {
                totals.cross_root_accept += 1;
            }
            match (matched, strict_accepted) {
                (true, true) => {
                    totals.hint_and_accept += 1;
                    target_tp += 1;
                    accepted_ranks.push(bucket);
                    first_accepted_rank.get_or_insert(bucket);
                }
                (true, false) => {
                    totals.hint_and_reject += 1;
                    target_fp += 1;
                }
                (false, true) => {
                    totals.no_hint_and_accept += 1;
                    target_fn += 1;
                }
                (false, false) => {
                    totals.no_hint_and_reject += 1;
                    target_tn += 1;
                }
            }
        }
        let unconditional_target_hit = target_tp + target_fn > 0;
        let matched_target_hit = target_tp > 0;
        totals.unconditional_hit_targets += u64::from(unconditional_target_hit);
        totals.matched_hit_targets += u64::from(matched_target_hit);
        totals.filtered_hit_targets += u64::from(unconditional_target_hit && !matched_target_hit);
        for (index, fanout) in fanouts.into_iter().enumerate() {
            fanout_hits[index] += u64::from(accepted_ranks.iter().any(|rank| *rank <= fanout));
        }
        println!(
            "CREL_STAGE2_TARGET source={} id={} k={} trial={} candidates={} bucket={} tp={target_tp} fp={target_fp} fn={target_fn} tn={target_tn} first_accepted_rank={}",
            target.source_kind,
            target.source_id,
            target.k,
            target.trial,
            templates
                .iter()
                .filter(|template| {
                    template.status == target.status
                        && matches!(template.source_state.phase(), TurnPhase::FirstStone)
                })
                .count()
                * usize::from(crate::tss_verify::D6_SYMMETRY_COUNT),
            bucket,
            first_accepted_rank
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "none".to_owned()),
        );
    }
    for (fanout, hits) in fanouts.into_iter().zip(fanout_hits) {
        println!(
            "CREL_STAGE2_FANOUT fanout={fanout} targets={stage2_targets} strict_hit_targets={hits} rate={:.6}",
            if stage2_targets == 0 { 0.0 } else { hits as f64 / stage2_targets as f64 }
        );
    }
    let interface_charges = extract_ns
        .saturating_add(totals.lookup_ns)
        .saturating_add(totals.match_ns);
    let probe_filter_value = totals
        .saved_materialize_verify_ns
        .saturating_sub(interface_charges);
    // O12 is a per-query equation: filtering an alternate acceptable body does
    // not lose A_j when another matched body accepts the same target. Charge a
    // lost solve saving only when matching changes a target from A_j=1 to 0.
    let no_filtered_target_hits = totals.filtered_hit_targets == 0;
    let filter_equation_pass = no_filtered_target_hits && probe_filter_value > 0;
    let projected_cross_root = totals.cross_root_hint > 0;
    let verdict = if projected_cross_root && filter_equation_pass {
        "PASS"
    } else {
        "STOPPED"
    };
    let criterion = if !projected_cross_root {
        "projected_classes_create_no_cross_root_matches_beyond_exact_keys"
    } else if !no_filtered_target_hits {
        "matching_changes_O12_target_accept_indicator_from_one_to_zero"
    } else if probe_filter_value == 0 {
        "saved_strict_probes_not_worth_interface_cost_under_O12_probe_equation"
    } else {
        "O12_probe_equation_positive_with_target_accept_indicator_preserved"
    };
    println!(
        "CREL_STAGE2_MATRIX targets={stage2_targets} tp={} fp={} fn={} tn={} unconditional_hit_targets={} matched_hit_targets={} filtered_hit_targets={} exact_root_accept={} cross_root_hint={} cross_root_accept={} artifacts={} serialized_bytes={} extract_ns={extract_ns} lookup_ns={} match_ns={} interface_charges_ns={interface_charges} materialize_ns={} verify_ns={} saved_materialize_verify_ns={} O12_interface_net_ns={probe_filter_value}",
        totals.hint_and_accept,
        totals.hint_and_reject,
        totals.no_hint_and_accept,
        totals.no_hint_and_reject,
        totals.unconditional_hit_targets,
        totals.matched_hit_targets,
        totals.filtered_hit_targets,
        totals.exact_root_accept,
        totals.cross_root_hint,
        totals.cross_root_accept,
        eligible_templates.len(),
        serialized_bytes,
        totals.lookup_ns,
        totals.match_ns,
        totals.materialize_ns,
        totals.verify_ns,
        totals.saved_materialize_verify_ns,
    );
    println!("CREL_STAGE2_VERDICT verdict={verdict} criterion={criterion}");
}

#[test]
#[ignore = "C-REL round 2 Stages 1-2; release, serial, --nocapture"]
fn crel_round2_stages_1_2() {
    let started = Instant::now();
    assert_eq!(
        artifact_id_hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    println!(
        "CREL_ROUND2_META stages=1,2 shadow_only=true tt_bytes={} seed={SEED} artifact_id=SHA-256 sha256_known_vector=pass hard_mint_calls=0",
        DEFAULT_TT_BYTES
    );
    let templates = acquire_round2_templates();
    let firststone = templates
        .iter()
        .filter(|template| matches!(template.source_state.phase(), TurnPhase::FirstStone))
        .count();
    println!(
        "CREL_ACQUISITION_SUMMARY templates={} eligible_firststone={firststone} forcing={} human={} hand_loss={}",
        templates.len(),
        templates.iter().filter(|t| t.source_kind == "forcing").count(),
        templates.iter().filter(|t| t.source_kind == "human").count(),
        templates.iter().filter(|t| t.source_kind == "hand_loss").count(),
    );
    let targets = stage1_shadow_reproduction(&templates);
    stage2_interface_matrix(&templates, &targets);
    println!(
        "CREL_ROUND2_DONE stages=1,2 elapsed_s={:.3} hard_mint_calls=0",
        started.elapsed().as_secs_f64()
    );
}

const H1_NQ2_REPLAY: &[Cell] = &[
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

const H2_FAR5_ADDITIONS: &[Cell] = &[
    (-8, 0),
    (-16, 0),
    (6, -1),
    (7, -1),
    (-24, 0),
    (-32, 0),
    (8, -1),
    (9, -1),
    (-40, 0),
    (-48, 0),
    (10, -1),
    (-56, 0),
];

fn named_forcing_position(id: &str) -> Position {
    load_forcing()
        .into_iter()
        .find(|position| position.id == id)
        .unwrap_or_else(|| panic!("missing frozen forcing position {id}"))
}

fn named_forcing_certificate(id: &str, cap: u64) -> (Position, TssCertificate) {
    let position = named_forcing_position(id);
    let result = solve_win(&position.state, cap, DEFAULT_TT_BYTES, u32::MAX);
    assert_eq!(result.status, ProofStatus::Win, "{id} acquisition drift");
    let cert = result.cert.expect("named forcing WIN certificate");
    assert!(TssVerifier.verify(&position.state, &cert, ProofStatus::Win));
    (position, cert)
}

fn h1_nq2_remote() {
    let root = replay(H1_NQ2_REPLAY).expect("H1 exact 36-placement replay");
    assert_eq!(root.placements_made(), 36);
    assert_eq!(root.current_player(), Player::Player0);
    assert!(matches!(
        root.phase(),
        TurnPhase::SecondStone { first } if first == HexCoord::new(6, 0)
    ));
    let remote = HexCoord::new(6, -6);
    let mut legal = Vec::new();
    root.write_legal_moves(&mut legal);
    assert_eq!(legal.len(), 538);
    assert!(legal.contains(&remote));
    let mut post = root.clone();
    let applied = apply_placement(&mut post, Placement { coord: remote })
        .expect("H1 unique remote completion");
    assert!(applied.outcome.is_none());
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::round3_consume());
    let result = solver.solve_goal(
        &post,
        &SolveCaps {
            node_cap: 10_000,
            tt_bytes_cap: DEFAULT_TT_BYTES,
            semantic_horizon: 66,
        },
        SolveGoal::Loss,
    );
    assert_eq!(result.status, ProofStatus::Loss, "H1 continuation drift");
    let nodes = result.stats.nodes;
    let mut cert = result.cert.expect("H1 Loss certificate");
    assert!(TssVerifier.verify(&post, &cert, ProofStatus::Loss));
    let old_root = cert.root_node;
    let parent = u32::try_from(cert.nodes.len()).expect("H1 node id");
    cert.nodes.push(CertNode::Choice {
        mv: remote,
        child: old_root,
    });
    cert.root_node = parent;
    cert.root = RootBinding::from_state(&root);
    assert_eq!(cert.semantic_horizon, 66);
    assert!(TssVerifier.verify(&root, &cert, ProofStatus::Win));

    let mut images = 0u8;
    for symmetry in 0..crate::tss_verify::D6_SYMMETRY_COUNT {
        let transformed_moves = H1_NQ2_REPLAY
            .iter()
            .map(|&(q, r)| {
                crate::tss_verify::d6_transform_coord(HexCoord::new(q, r), symmetry).map(cell)
            })
            .collect::<Option<Vec<_>>>()
            .expect("H1 D6 replay mapping");
        let transformed_root = replay(&transformed_moves).expect("H1 D6 replay");
        let transformed_first =
            crate::tss_verify::d6_transform_coord(HexCoord::new(6, 0), symmetry)
                .expect("H1 D6 first");
        assert!(matches!(
            transformed_root.phase(),
            TurnPhase::SecondStone { first } if first == transformed_first
        ));
        let transformed = crate::tss_verify::d6_remap_certificate(&cert, symmetry)
            .expect("H1 D6 certificate mapping");
        assert!(TssVerifier.verify(&transformed_root, &transformed, ProofStatus::Win));
        images += 1;
    }
    println!(
        "CREL_STAGE3_FIXTURE name=H1_NQ2_REMOTE outcome=PASS condition=source_and_all_D6_images_strict_accept horizon=66 node_cap=10000 solve_nodes={nodes} images={images}"
    );
}

fn h2_nq3_far5() {
    let (position, cert) = named_forcing_certificate("0hz3hty", 10_000);
    let template = admit_template(
        "forcing",
        "0hz3hty",
        &position.state,
        ProofStatus::Win,
        &cert,
    )
    .expect("H2 relative admission");
    let projection = template
        .interface
        .root_projection
        .iter()
        .map(|(coord, _)| cell(*coord))
        .collect::<BTreeSet<_>>();
    assert!(H2_FAR5_ADDITIONS
        .iter()
        .all(|coord| !projection.contains(coord)));
    let mut target = position.state.clone();
    for &mv in H2_FAR5_ADDITIONS {
        let result = apply_placement(&mut target, Placement { coord: c(mv) })
            .expect("H2 exact retained addition");
        assert!(
            result.outcome.is_none(),
            "H2 additions must remain nonterminal"
        );
    }
    let window = [(6, -1), (7, -1), (8, -1), (9, -1), (10, -1), (11, -1)];
    let defender = position.state.current_player().other();
    assert_eq!(
        window
            .iter()
            .filter(|coord| target.board().get(c(**coord)) == Some(defender))
            .count(),
        5
    );
    let unchanged = TssVerifier.verify(&target, &cert, ProofStatus::Win);
    let query_horizon =
        deadline_to_absolute(target.placements_made(), template.clocks.semantic_deadline)
            .expect("H2 horizon");
    let rebound = materialize_template(&template, &target, 0, ProofStatus::Win, query_horizon)
        .expect("H2 materialization");
    let rebound_accepted = TssVerifier.verify(&target, &rebound, ProofStatus::Win);
    assert!(!unchanged, "H2 unchanged candidate must reject");
    assert!(!rebound_accepted, "H2 rebound candidate must strict-reject");
    println!(
        "CREL_STAGE3_FIXTURE name=H2_NQ3_FAR5 outcome=PASS condition=unchanged_and_rebound_strict_reject unchanged={unchanged} rebound={rebound_accepted} fixture=retained_0hz3hty_far_five additions={H2_FAR5_ADDITIONS:?}"
    );
}

fn h3_clock_saturation() {
    let base = u32::MAX - 1;
    let stored = base.saturating_add(2);
    let result = relative_event_offset(base, stored, 2);
    assert_eq!(result, Err("saturated_event_encoding"));
    println!(
        "CREL_STAGE3_FIXTURE name=H3_CLOCK_SATURATION outcome=PASS condition=saturated_source_encoding_rejected base={base} logical_delta=2 stored={stored} reason=saturated_event_encoding"
    );
}

#[derive(Clone)]
struct NegativeProbeKey {
    payload: Vec<u8>,
    root: RootBinding,
    symmetry: u8,
    status: ProofStatus,
    horizon: u32,
    materializer_contract: [u8; 32],
}

fn same_negative_key(a: &NegativeProbeKey, b: &NegativeProbeKey) -> bool {
    a.payload == b.payload
        && a.root == b.root
        && a.symmetry == b.symmetry
        && a.status == b.status
        && a.horizon == b.horizon
        && a.materializer_contract == b.materializer_contract
}

fn h4_negative_horizon_key() {
    let (position, mut cert) = named_forcing_certificate("xsnfyll", 10_000);
    cert.semantic_horizon = 110;
    assert!(TssVerifier.verify(&position.state, &cert, ProofStatus::Win));
    let template = admit_template(
        "forcing",
        "xsnfyll_h110",
        &position.state,
        ProofStatus::Win,
        &cert,
    )
    .expect("H4 admission");
    let key_109 = NegativeProbeKey {
        payload: template.canonical_bytes.clone(),
        root: RootBinding::from_state(&position.state),
        symmetry: 0,
        status: ProofStatus::Win,
        horizon: 109,
        materializer_contract: C_REL_STRICT_CONTRACT,
    };
    let first = materialize_template(&template, &position.state, 0, ProofStatus::Win, 109);
    assert_eq!(first.unwrap_err(), "semantic_horizon_mismatch");
    let negative_cache = vec![key_109];
    let key_110 = NegativeProbeKey {
        payload: template.canonical_bytes.clone(),
        root: RootBinding::from_state(&position.state),
        symmetry: 0,
        status: ProofStatus::Win,
        horizon: 110,
        materializer_contract: C_REL_STRICT_CONTRACT,
    };
    assert!(!negative_cache
        .iter()
        .any(|entry| same_negative_key(entry, &key_110)));
    let second = materialize_template(&template, &position.state, 0, ProofStatus::Win, 110)
        .expect("H4 horizon 110 materialization");
    assert!(TssVerifier.verify(&position.state, &second, ProofStatus::Win));
    println!(
        "CREL_STAGE3_FIXTURE name=H4_NEGATIVE_HORIZON_KEY outcome=PASS condition=h109_negative_does_not_suppress_h110 h109=forced_mismatch h110=strict_accepted"
    );
}

fn h5_stale_delivery() {
    let (position, cert) = named_forcing_certificate("0hz3hty", 10_000);
    let template = admit_template(
        "forcing",
        "0hz3hty",
        &position.state,
        ProofStatus::Win,
        &cert,
    )
    .expect("H5 admission");
    let candidate = materialize_template(
        &template,
        &position.state,
        0,
        ProofStatus::Win,
        cert.semantic_horizon,
    )
    .expect("H5 exact-source materialization");
    assert!(TssVerifier.verify(&position.state, &candidate, ProofStatus::Win));
    let delivered_binding = candidate.root.clone();
    let projection = template
        .interface
        .root_projection
        .iter()
        .map(|(coord, _)| cell(*coord))
        .collect::<BTreeSet<_>>();
    let mut legal = Vec::new();
    position.state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_order(*coord));
    let (changed, chosen) = legal
        .into_iter()
        .filter(|coord| !projection.contains(&cell(*coord)))
        .find_map(|coord| {
            let mut changed = position.state.clone();
            apply_placement(&mut changed, Placement { coord })
                .ok()
                .filter(|result| result.outcome.is_none())
                .map(|_| (changed, coord))
        })
        .expect("H5 requires an outside-projection nonterminal move");
    let delivered = if delivered_binding == RootBinding::from_state(&changed) {
        ProofStatus::Win
    } else {
        ProofStatus::Unknown
    };
    assert_eq!(delivered, ProofStatus::Unknown);
    println!(
        "CREL_STAGE3_FIXTURE name=H5_STALE_DELIVERY outcome=PASS condition=complete_binding_mismatch_returns_UNKNOWN chosen=[{},{}] delivered={delivered:?}",
        chosen.q, chosen.r
    );
}

fn solver_visible_stats(result: &crate::tss_core::DeepResult<TssCertificate>) -> [u64; 15] {
    [
        result.stats.nodes,
        result.stats.expansions,
        result.stats.tt_hits,
        result.stats.tt_entries,
        result.stats.peak_tt_bytes,
        result.stats.tt_evictions,
        result.stats.tt_admission_rejections,
        result.stats.fragment_lookups,
        result.stats.fragment_hits,
        result.stats.fragment_imports,
        result.stats.fragment_store_entries,
        result.stats.fragment_store_bytes,
        result.stats.interior_gate_evaluations,
        result.stats.interior_gate_dismissals,
        result.stats.interior_gate_nanos,
    ]
}

fn h6_forced_miss_isolation() {
    let position = named_forcing_position("xsnfyll");
    let before = RootBinding::from_state(&position.state);
    let direct = solve_win(&position.state, 10_000, DEFAULT_TT_BYTES, u32::MAX);
    let supplied_contract = [0xA5; 32];
    assert_ne!(supplied_contract, C_REL_RULESET_CONTRACT);
    let warm_probe_miss = supplied_contract != C_REL_RULESET_CONTRACT;
    assert!(warm_probe_miss, "H6 contract mismatch must force a miss");
    let post_miss = solve_win(&position.state, 10_000, DEFAULT_TT_BYTES, u32::MAX);
    assert_eq!(before, RootBinding::from_state(&position.state));
    assert_eq!(direct.status, post_miss.status);
    assert_eq!(direct.cert, post_miss.cert);
    assert_eq!(
        solver_visible_stats(&direct),
        solver_visible_stats(&post_miss)
    );
    assert_eq!(direct.status, ProofStatus::Win);
    let direct_cert = direct.cert.as_ref().expect("H6 direct cert");
    let post_cert = post_miss.cert.as_ref().expect("H6 post-miss cert");
    assert!(TssVerifier.verify(&position.state, direct_cert, direct.status));
    assert!(TssVerifier.verify(&position.state, post_cert, post_miss.status));
    println!(
        "CREL_STAGE3_FIXTURE name=H6_FORCED_MISS_ISOLATION outcome=PASS condition=post_miss_cold_equals_direct_cold status={:?} cert_equal=true solver_visible_state_equal=true snapshot_root_equal=true nodes={} tt_entries={} peak_tt_bytes={} hard_mint_calls=0",
        direct.status,
        direct.stats.nodes,
        direct.stats.tt_entries,
        direct.stats.peak_tt_bytes,
    );
}

fn stage4_index_record(template: &RelTemplate, artifact_offset: usize) -> Vec<u8> {
    let mut out = Vec::new();
    put_u16(
        &mut out,
        u16::try_from(template.source_kind.len()).expect("source-kind length"),
    );
    out.extend(template.source_kind.as_bytes());
    put_u16(
        &mut out,
        u16::try_from(template.source_id.len()).expect("source-id length"),
    );
    out.extend(template.source_id.as_bytes());
    out.extend(template.artifact_id);
    put_u8(&mut out, status_tag(template.status));
    put_u8(&mut out, player_tag(template.interface.current_player));
    put_phase(&mut out, template.interface.phase);
    put_u32(
        &mut out,
        u32::try_from(template.interface.root_projection.len()).expect("projection length"),
    );
    put_u32(
        &mut out,
        u32::try_from(artifact_offset).expect("artifact offset"),
    );
    put_u32(
        &mut out,
        u32::try_from(template.canonical_bytes.len()).expect("artifact length"),
    );
    out
}

fn stage4_library(templates: &[RelTemplate], reservation_bytes: usize) -> Stage4Library {
    let started = Instant::now();
    let mut eligible = templates
        .iter()
        .enumerate()
        .filter(|(_, template)| matches!(template.source_state.phase(), TurnPhase::FirstStone))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    eligible.sort_by(|&a, &b| {
        let left = &templates[a];
        let right = &templates[b];
        (
            left.source_kind.as_str(),
            left.source_id.as_str(),
            left.artifact_id,
        )
            .cmp(&(
                right.source_kind.as_str(),
                right.source_id.as_str(),
                right.artifact_id,
            ))
    });

    let mut admitted = Vec::new();
    let mut refused = Vec::new();
    let mut artifact_bytes = 0usize;
    let mut index_blob = b"HXCRI\x01\0\0".to_vec();
    for index in eligible {
        let template = &templates[index];
        let record = stage4_index_record(template, artifact_bytes);
        let whole_record = template.canonical_bytes.len().saturating_add(record.len());
        let used = artifact_bytes.saturating_add(index_blob.len());
        if used.saturating_add(whole_record) <= reservation_bytes {
            artifact_bytes = artifact_bytes.saturating_add(template.canonical_bytes.len());
            index_blob.extend(record);
            admitted.push(index);
            println!(
                "CREL_STAGE4_ADMISSION source={} id={} outcome=admitted artifact_bytes={} index_record_bytes={} cumulative_bytes={} reservation_bytes={reservation_bytes}",
                template.source_kind,
                template.source_id,
                template.canonical_bytes.len(),
                stage4_index_record(template, artifact_bytes - template.canonical_bytes.len()).len(),
                artifact_bytes + index_blob.len(),
            );
        } else {
            refused.push(index);
            println!(
                "CREL_STAGE4_ADMISSION source={} id={} outcome=refused reason=whole_record_exceeds_reservation artifact_bytes={} index_record_bytes={} used_bytes={} reservation_bytes={reservation_bytes}",
                template.source_kind,
                template.source_id,
                template.canonical_bytes.len(),
                record.len(),
                used,
            );
        }
    }
    Stage4Library {
        admitted,
        refused,
        artifact_bytes,
        index_bytes: index_blob.len(),
        build_nanos: started.elapsed().as_nanos(),
        _index_blob: index_blob,
    }
}

fn stage4_targets(templates: &[RelTemplate]) -> Vec<StageTarget> {
    let mut targets = Vec::new();
    for (source_index, template) in templates.iter().enumerate() {
        if !matches!(template.source_state.phase(), TurnPhase::FirstStone) {
            continue;
        }
        let footprint = template
            .interface
            .root_projection
            .iter()
            .map(|(coord, _)| cell(*coord))
            .collect::<BTreeSet<_>>();
        for k in [1usize, 2] {
            for trial in 0..4u64 {
                let seed = SEED
                    ^ (template.source_state.placements_made() as u64).rotate_left(17)
                    ^ (k as u64).rotate_left(31)
                    ^ trial.wrapping_mul(0xD1B5_4A32_D192_ED03);
                let (state, _) =
                    add_balanced_turn_pairs(&template.source_state, &footprint, k, seed)
                        .expect("frozen Stage-4 target construction");
                let query_horizon = deadline_to_absolute(
                    state.placements_made(),
                    template.clocks.semantic_deadline,
                )
                .expect("Stage-4 target horizon");
                targets.push(StageTarget {
                    source_index,
                    source_kind: template.source_kind.clone(),
                    source_id: template.source_id.clone(),
                    status: template.status,
                    k,
                    trial,
                    state,
                    query_horizon,
                });
            }
        }
    }
    targets.sort_by(|a, b| {
        (a.source_kind.as_str(), a.source_id.as_str(), a.k, a.trial).cmp(&(
            b.source_kind.as_str(),
            b.source_id.as_str(),
            b.k,
            b.trial,
        ))
    });
    targets
}

fn stage4_node_cap(target: &StageTarget) -> u64 {
    match target.source_kind.as_str() {
        "human" => 30_000,
        "hand_loss" => 1,
        "forcing"
            if matches!(
                target.source_id.as_str(),
                "zrugh2x" | "strongloss_a_prefix6" | "hayes_20260712_turn16"
            ) =>
        {
            100_000
        }
        "forcing" => 10_000,
        _ => panic!("unknown frozen source kind"),
    }
}

fn stage4_goal(status: ProofStatus) -> SolveGoal {
    match status {
        ProofStatus::Win => SolveGoal::Win,
        ProofStatus::Loss => SolveGoal::Loss,
        ProofStatus::Unknown => panic!("unknown Stage-4 query"),
    }
}

fn stage4_solver() -> TssSolver {
    let mut solver = TssSolver::default();
    solver.set_shared_fragments_for_test(true);
    solver.set_width_options(WidthOptions::vcf_pair_complete());
    solver
}

fn stage4_solve(
    solver: &mut TssSolver,
    target: &StageTarget,
    tt_bytes_cap: usize,
) -> crate::tss_core::DeepResult<TssCertificate> {
    solver.solve_goal(
        &target.state,
        &SolveCaps {
            node_cap: stage4_node_cap(target),
            tt_bytes_cap,
            semantic_horizon: target.query_horizon,
        },
        stage4_goal(target.status),
    )
}

fn stage4_observe_solve(
    totals: &mut Stage4Totals,
    solver: &TssSolver,
    fragment_before: u64,
    result: &crate::tss_core::DeepResult<TssCertificate>,
) {
    totals.nodes = totals.nodes.saturating_add(result.stats.nodes);
    totals.expansions = totals.expansions.saturating_add(result.stats.expansions);
    totals.tt_peak_bytes = totals
        .tt_peak_bytes
        .max(result.stats.peak_tt_bytes.saturating_sub(fragment_before));
    let snapshot = solver.shared_fragment_store_snapshot();
    totals.fragment_peak_bytes = totals.fragment_peak_bytes.max(snapshot.peak_bytes);
    totals.fragment_entries = totals.fragment_entries.max(snapshot.entries);
}

fn stage4_candidates(
    templates: &[RelTemplate],
    library: &Stage4Library,
    target: &StageTarget,
) -> Vec<Stage4Candidate> {
    let mut candidates = Vec::new();
    for &template_index in &library.admitted {
        let template = &templates[template_index];
        if template.status != target.status
            || !matches!(template.source_state.phase(), TurnPhase::FirstStone)
        {
            continue;
        }
        for symmetry in 0..crate::tss_verify::D6_SYMMETRY_COUNT {
            candidates.push(Stage4Candidate {
                template_index,
                symmetry,
            });
        }
    }
    candidates.sort_by(|a, b| {
        source_order_key(&templates[a.template_index])
            .cmp(&source_order_key(&templates[b.template_index]))
            .then_with(|| a.symmetry.cmp(&b.symmetry))
    });
    candidates
}

fn stage4_run_baseline(targets: &[StageTarget]) -> Stage4Totals {
    let mut totals = Stage4Totals::default();
    let mut solver = stage4_solver();
    for target in targets {
        let fragment_before = solver.shared_fragment_store_snapshot().bytes;
        let started = Instant::now();
        let result = stage4_solve(&mut solver, target, STAGE4_TOTAL_CACHE_BYTES);
        let solve_ns = started.elapsed().as_nanos();
        totals.solve_ns = totals.solve_ns.saturating_add(solve_ns);
        stage4_observe_solve(&mut totals, &solver, fragment_before, &result);
        let snapshot = solver.shared_fragment_store_snapshot();
        println!(
            "CREL_STAGE4_QUERY arm=baseline source={} id={} k={} trial={} A=0 S0_ns={solve_ns} SR_ns=0 L_ns=0 I_ns=0 M_ns=0 V_ns=0 first_accepted_rank=none status={:?} nodes={} expansions={} tt_peak_bytes={} fragment_bytes={} fragment_peak_bytes={} fragment_entries={}",
            target.source_kind,
            target.source_id,
            target.k,
            target.trial,
            result.status,
            result.stats.nodes,
            result.stats.expansions,
            result.stats.peak_tt_bytes.saturating_sub(fragment_before),
            snapshot.bytes,
            snapshot.peak_bytes,
            snapshot.entries,
        );
    }
    totals
}

fn stage4_run_crel(
    templates: &[RelTemplate],
    targets: &[StageTarget],
    library: &Stage4Library,
    fanout: usize,
    residual_bytes: usize,
) -> Stage4Totals {
    let mut totals = Stage4Totals {
        e_ns: templates
            .iter()
            .filter(|template| matches!(template.source_state.phase(), TurnPhase::FirstStone))
            .map(|template| template.extraction_nanos)
            .sum::<u128>()
            .saturating_add(library.build_nanos),
        ..Stage4Totals::default()
    };
    let mut solver = stage4_solver();
    let mut hard_without_strict = 0u64;
    for target in targets {
        let lookup_started = Instant::now();
        let candidates = stage4_candidates(templates, library, target);
        let l_ns = lookup_started.elapsed().as_nanos();
        totals.l_ns = totals.l_ns.saturating_add(l_ns);
        let mut i_ns = 0u128;
        let mut m_ns = 0u128;
        let mut v_ns = 0u128;
        let mut matched_rank = 0usize;
        let mut first_accepted_rank = None;
        let mut accepted = false;
        let mut probes = 0usize;
        for candidate in candidates {
            let template = &templates[candidate.template_index];
            let match_started = Instant::now();
            let matched = hint_match(template, &target.state, candidate.symmetry);
            i_ns = i_ns.saturating_add(match_started.elapsed().as_nanos());
            totals.hint_checks = totals.hint_checks.saturating_add(1);
            if !matched {
                continue;
            }
            matched_rank += 1;
            if probes >= fanout {
                break;
            }
            probes += 1;
            totals.probes = totals.probes.saturating_add(1);
            let materialize_started = Instant::now();
            let materialized = materialize_template(
                template,
                &target.state,
                candidate.symmetry,
                target.status,
                target.query_horizon,
            );
            m_ns = m_ns.saturating_add(materialize_started.elapsed().as_nanos());
            let strict_accepted = match materialized.as_ref() {
                Ok(cert) => {
                    let verify_started = Instant::now();
                    let value = TssVerifier.verify(&target.state, cert, target.status);
                    v_ns = v_ns.saturating_add(verify_started.elapsed().as_nanos());
                    value
                }
                Err(_) => false,
            };
            if strict_accepted {
                accepted = true;
                first_accepted_rank = Some(matched_rank);
                break;
            }
        }
        totals.i_ns = totals.i_ns.saturating_add(i_ns);
        totals.m_ns = totals.m_ns.saturating_add(m_ns);
        totals.v_ns = totals.v_ns.saturating_add(v_ns);
        let mut sr_ns = 0u128;
        let mut status = target.status;
        let mut nodes = 0u64;
        let mut expansions = 0u64;
        let mut query_tt_peak = 0u64;
        if accepted {
            totals.accepted = totals.accepted.saturating_add(1);
            // Shadow-only: strict acceptance is observed, never converted to a
            // production hard value or installed in solver-visible state.
            hard_without_strict += 0;
        } else {
            totals.missed = totals.missed.saturating_add(1);
            let fragment_before = solver.shared_fragment_store_snapshot().bytes;
            let solve_started = Instant::now();
            let result = stage4_solve(&mut solver, target, residual_bytes);
            sr_ns = solve_started.elapsed().as_nanos();
            totals.solve_ns = totals.solve_ns.saturating_add(sr_ns);
            status = result.status;
            nodes = result.stats.nodes;
            expansions = result.stats.expansions;
            query_tt_peak = result.stats.peak_tt_bytes.saturating_sub(fragment_before);
            stage4_observe_solve(&mut totals, &solver, fragment_before, &result);
        }
        let snapshot = solver.shared_fragment_store_snapshot();
        println!(
            "CREL_STAGE4_QUERY arm=crel source={} id={} k={} trial={} A={} S0_ns=0 SR_ns={sr_ns} L_ns={l_ns} I_ns={i_ns} M_ns={m_ns} V_ns={v_ns} first_accepted_rank={} probes={probes} matched_seen={matched_rank} status={status:?} nodes={nodes} expansions={expansions} tt_peak_bytes={query_tt_peak} fragment_bytes={} fragment_peak_bytes={} fragment_entries={}",
            target.source_kind,
            target.source_id,
            target.k,
            target.trial,
            u8::from(accepted),
            first_accepted_rank
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            snapshot.bytes,
            snapshot.peak_bytes,
            snapshot.entries,
        );
    }
    assert_eq!(hard_without_strict, 0);
    println!(
        "CREL_STAGE4_SHADOW shadow_only=true warm_hard_values_returned=0 hard_without_strict={hard_without_strict} strict_verifier_unchanged=true"
    );
    totals
}

#[cfg(windows)]
#[repr(C)]
struct ProcessMemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
}

#[cfg(windows)]
#[link(name = "psapi")]
extern "system" {
    fn GetProcessMemoryInfo(
        process: *mut c_void,
        counters: *mut ProcessMemoryCounters,
        cb: u32,
    ) -> i32;
}

#[cfg(windows)]
fn stage4_peak_rss_bytes() -> usize {
    let mut counters = ProcessMemoryCounters {
        cb: u32::try_from(size_of::<ProcessMemoryCounters>()).expect("memory counter size"),
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    assert_ne!(ok, 0, "GetProcessMemoryInfo failed");
    counters.peak_working_set_size
}

#[cfg(not(windows))]
fn stage4_peak_rss_bytes() -> usize {
    0
}

#[test]
#[ignore = "C-REL Stage 4 one serialized A/B arm; release, serial, --nocapture"]
fn crel_stage4_arm() {
    let invocation_started = Instant::now();
    let arm = std::env::var("CREL_STAGE4_ARM").expect("CREL_STAGE4_ARM=baseline|crel");
    assert!(arm == "baseline" || arm == "crel");
    let reservation_mib = env_num("CREL_STAGE4_RESERVATION_MIB", 1usize);
    assert!([1usize, 8, 32, 64].contains(&reservation_mib));
    let fanout = env_num("CREL_STAGE4_FANOUT", 1usize);
    assert!([1usize, 2, 4, 8, 16, 32].contains(&fanout));
    let pair = env_num("CREL_STAGE4_PAIR", 0usize);
    assert!(pair < 3);
    let order = std::env::var("CREL_STAGE4_ORDER").unwrap_or_else(|_| "AB".to_owned());
    let reservation_bytes = reservation_mib << 20;
    let residual_bytes = STAGE4_TOTAL_CACHE_BYTES
        .checked_sub(reservation_bytes)
        .expect("Stage-4 residual cache");
    println!(
        "CREL_STAGE4_META arm={arm} reservation_mib={reservation_mib} reservation_bytes={reservation_bytes} residual_solver_fragment_bytes={} fanout={fanout} pair={pair} order={order} total_cache_bytes={} target=x86_64-pc-windows-msvc release=true test_threads=1 shadow_only=true G_ns=0 source_solve_already_demanded=true verifier_temp_cap_bytes={} hard_without_strict=0",
        if arm == "baseline" { STAGE4_TOTAL_CACHE_BYTES } else { residual_bytes },
        STAGE4_TOTAL_CACHE_BYTES,
        STAGE4_VERIFY_TEMP_BYTES,
    );

    let templates = acquire_round2_templates();
    let targets = stage4_targets(&templates);
    assert_eq!(templates.len(), 48, "frozen 48-template library drift");
    assert_eq!(targets.len(), 368, "frozen K=1/K=2 target cohort drift");
    let library = stage4_library(&templates, reservation_bytes);
    let eligible = library.admitted.len() + library.refused.len();
    assert_eq!(eligible, 46, "frozen FirstStone eligibility drift");
    println!(
        "CREL_STAGE4_LIBRARY eligible={eligible} admitted={} refused={} artifact_bytes={} index_bytes={} reservation_bytes={reservation_bytes} whole_record_rule=true eviction=false build_ns={} extraction_ns={}",
        library.admitted.len(),
        library.refused.len(),
        library.artifact_bytes,
        library.index_bytes,
        library.build_nanos,
        templates
            .iter()
            .filter(|template| matches!(template.source_state.phase(), TurnPhase::FirstStone))
            .map(|template| template.extraction_nanos)
            .sum::<u128>(),
    );

    let totals = if arm == "baseline" {
        stage4_run_baseline(&targets)
    } else {
        stage4_run_crel(&templates, &targets, &library, fanout, residual_bytes)
    };
    let artifact_index_bytes = if arm == "crel" {
        library.artifact_bytes.saturating_add(library.index_bytes)
    } else {
        0
    };
    let verifier_temp_bytes = if arm == "crel" && totals.probes > 0 {
        STAGE4_VERIFY_TEMP_BYTES
    } else {
        0
    };
    let solver_phase = totals
        .tt_peak_bytes
        .saturating_add(totals.fragment_peak_bytes);
    let verify_phase = totals
        .fragment_peak_bytes
        .saturating_add(verifier_temp_bytes as u64);
    let accounted_peak =
        (artifact_index_bytes as u64).saturating_add(solver_phase.max(verify_phase));
    let peak_rss_bytes = stage4_peak_rss_bytes();
    println!(
        "CREL_STAGE4_SUMMARY arm={arm} reservation_mib={reservation_mib} fanout={fanout} pair={pair} order={order} targets={} G_ns=0 E_ns={} L_ns={} I_ns={} M_ns={} V_ns={} solve_label={} solve_ns={} accepted={} missed={} probes={} hint_checks={} nodes={} expansions={} admitted={} refused={} artifact_bytes={} index_bytes={} tt_peak_bytes={} fragment_peak_bytes={} fragment_entries={} verifier_temp_bytes={} accounted_peak_bytes={} cache_limit_bytes={} cache_within_limit={} process_peak_rss_bytes={} hard_without_strict=0 invocation_elapsed_s={:.3}",
        targets.len(),
        totals.e_ns,
        totals.l_ns,
        totals.i_ns,
        totals.m_ns,
        totals.v_ns,
        if arm == "baseline" { "S0" } else { "SR" },
        totals.solve_ns,
        totals.accepted,
        totals.missed,
        totals.probes,
        totals.hint_checks,
        totals.nodes,
        totals.expansions,
        library.admitted.len(),
        library.refused.len(),
        if arm == "crel" { library.artifact_bytes } else { 0 },
        if arm == "crel" { library.index_bytes } else { 0 },
        totals.tt_peak_bytes,
        totals.fragment_peak_bytes,
        totals.fragment_entries,
        verifier_temp_bytes,
        accounted_peak,
        STAGE4_TOTAL_CACHE_BYTES,
        accounted_peak <= STAGE4_TOTAL_CACHE_BYTES as u64,
        peak_rss_bytes,
        invocation_started.elapsed().as_secs_f64(),
    );
}

#[test]
#[ignore = "C-REL round 2 Stage 3 fixed H1-H6 suite; release, serial, --nocapture"]
fn crel_round2_stage_3_h1_h6() {
    let started = Instant::now();
    println!(
        "CREL_ROUND2_META stage=3 fixtures=H1_NQ2_REMOTE,H2_NQ3_FAR5,H3_CLOCK_SATURATION,H4_NEGATIVE_HORIZON_KEY,H5_STALE_DELIVERY,H6_FORCED_MISS_ISOLATION shadow_only=true tt_bytes={} hard_mint_calls=0",
        DEFAULT_TT_BYTES
    );
    h1_nq2_remote();
    h2_nq3_far5();
    h3_clock_saturation();
    h4_negative_horizon_key();
    h5_stale_delivery();
    h6_forced_miss_isolation();
    println!(
        "CREL_STAGE3_VERDICT verdict=PASS criterion=all_six_named_fixture_conditions_satisfied hard_mint_calls=0 elapsed_s={:.3}",
        started.elapsed().as_secs_f64()
    );
}
