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

use std::collections::BTreeSet;
use std::time::Instant;

use hexo_engine::{apply_placement, hex_distance, HexCoord, HexoState, Placement, TurnPhase};

use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::{CertNode, RootBinding, TssCertificate, TssVerifier};

type Cell = (i16, i16);

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DEFAULT_TT_BYTES: usize = 64 << 20;
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
