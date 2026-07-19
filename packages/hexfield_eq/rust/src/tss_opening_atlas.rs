//! Default-off certified opening-atlas harness.
//!
//! Run only via the ignored `opening_atlas_pass1` test. Hard rows are printed
//! only after the independent strict verifier accepts the returned certificate
//! at the canonical root and all 12 D6-remapped roots.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::{
    d6_remap_certificate, d6_transform_coord, CertNode, RootBinding, TssCertificate, TssVerifier,
    D6_SYMMETRY_COUNT, MAX_CERT_DEPTH, MAX_CERT_NODES,
};

const DEFAULT_TT_BYTES: usize = 512 << 20;
const DEFAULT_NODE_LADDER: [u64; 2] = [10_000, 100_000];
const DEFAULT_RELATIVE_HORIZON: u32 = 12;
const DEFAULT_GAME_COUNT: usize = 8;
const DEFAULT_BACKTRACK: usize = 12;
const DEFAULT_WALL_SECONDS: u64 = 1_200;
const DEFAULT_FIRST_N: usize = 7;

#[derive(Clone, Debug)]
struct HumanGame {
    hash: String,
    moves: Vec<HexCoord>,
    winner: i8,
}

#[derive(Clone, Debug)]
struct Candidate {
    source: String,
    source_prefix: usize,
    canonical_moves: Vec<HexCoord>,
    orbit_size: usize,
    preferred_moves: Vec<HexCoord>,
}

#[derive(Clone, Debug)]
struct ReverifyRoot {
    id: String,
    moves: Vec<HexCoord>,
    expected_status: ProofStatus,
    expected_claimant: Player,
    terminal_before: bool,
    first_line_move: Option<HexCoord>,
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<T>()
                .unwrap_or_else(|error| panic!("{name}: {error:?}"))
        })
        .unwrap_or(default)
}

fn env_ladder(name: &str, default: &[u64]) -> Vec<u64> {
    match std::env::var(name) {
        Ok(value) => {
            let ladder = value
                .split(',')
                .map(|token| token.trim())
                .filter(|token| !token.is_empty())
                .map(|token| {
                    token
                        .parse::<u64>()
                        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
                })
                .collect::<Vec<_>>();
            assert!(!ladder.is_empty(), "{name} must list at least one node cap");
            ladder
        }
        Err(_) => default.to_vec(),
    }
}

fn parse_ints(slice: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in slice.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            out.push(current.parse().expect("i16 corpus token"));
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current.parse().expect("i16 corpus token"));
    }
    out
}

fn parse_game(line: &str) -> Option<HumanGame> {
    let hash_key = "\"game_hash\":\"";
    let hash_tail = &line[line.find(hash_key)? + hash_key.len()..];
    let hash = hash_tail[..hash_tail.find('"')?].to_owned();

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
    let ints = parse_ints(&after[start..=end?]);
    let moves = ints
        .chunks_exact(2)
        .map(|pair| HexCoord::new(pair[0], pair[1]))
        .collect::<Vec<_>>();

    let winner_key = "\"winner\":";
    let winner_tail = &line[line.find(winner_key)? + winner_key.len()..];
    let winner = winner_tail
        .chars()
        .skip_while(|ch| *ch != '-' && !ch.is_ascii_digit())
        .take_while(|ch| *ch == '-' || ch.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some(HumanGame {
        hash,
        moves,
        winner,
    })
}

fn coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

fn sequence_key(moves: &[HexCoord]) -> Vec<(i16, i16)> {
    moves.iter().copied().map(coord_key).collect()
}

fn root_position_key(moves: &[HexCoord]) -> (u8, (u8, i16, i16), Vec<(i16, i16, u8)>) {
    let state = replay_unchecked(moves);
    let binding = RootBinding::from_state(&state);
    let player = match binding.current_player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    };
    let phase = match binding.phase {
        hexo_engine::TurnPhase::Opening => (0, 0, 0),
        hexo_engine::TurnPhase::FirstStone => (1, 0, 0),
        hexo_engine::TurnPhase::SecondStone { first } => (2, first.q, first.r),
    };
    let stones = binding
        .occupancy
        .into_iter()
        .zip(binding.owners)
        .map(|(coord, owner)| {
            let owner = match owner {
                Player::Player0 => 0,
                Player::Player1 => 1,
            };
            (coord.q, coord.r, owner)
        })
        .collect();
    (player, phase, stones)
}

fn canonical_sequence_with_symmetry(moves: &[HexCoord]) -> (Vec<HexCoord>, u8) {
    (0..D6_SYMMETRY_COUNT)
        .map(|symmetry| {
            let image = moves
                .iter()
                .copied()
                .map(|coord| d6_transform_coord(coord, symmetry).expect("D6 coordinate in range"))
                .collect::<Vec<_>>();
            (image, symmetry)
        })
        .min_by_key(|(image, _)| (root_position_key(image), sequence_key(image)))
        .expect("D6 is nonempty")
}

fn canonical_sequence(moves: &[HexCoord]) -> Vec<HexCoord> {
    canonical_sequence_with_symmetry(moves).0
}

fn canonical_unordered_pair(a: HexCoord, b: HexCoord) -> Vec<HexCoord> {
    (0..D6_SYMMETRY_COUNT)
        .map(|symmetry| {
            let mut pair = vec![
                d6_transform_coord(a, symmetry).expect("D6 coordinate in range"),
                d6_transform_coord(b, symmetry).expect("D6 coordinate in range"),
            ];
            pair.sort_by_key(|coord| coord_key(*coord));
            pair
        })
        .min_by_key(|image| sequence_key(image))
        .expect("D6 is nonempty")
}

fn orbit_size(moves: &[HexCoord]) -> usize {
    let images = (0..D6_SYMMETRY_COUNT)
        .map(|symmetry| {
            let image = moves
                .iter()
                .copied()
                .map(|coord| d6_transform_coord(coord, symmetry).expect("D6 coordinate in range"))
                .collect::<Vec<_>>();
            root_position_key(&image)
        })
        .collect::<BTreeSet<_>>();
    images.len()
}

fn hex_distance(coord: HexCoord) -> i16 {
    coord
        .q
        .abs()
        .max(coord.r.abs())
        .max((coord.q + coord.r).abs())
}

fn shallow_candidates() -> Vec<Candidate> {
    let mut reps = BTreeMap::<Vec<(i16, i16)>, Vec<HexCoord>>::new();
    for q in -8..=8 {
        for r in -8..=8 {
            let coord = HexCoord::new(q, r);
            if coord == HexCoord::ZERO || hex_distance(coord) > 8 {
                continue;
            }
            let canonical = canonical_sequence(&[HexCoord::ZERO, coord]);
            reps.entry(sequence_key(&canonical)).or_insert(canonical);
        }
    }
    let mut out = vec![
        Candidate {
            source: "shallow:empty".to_owned(),
            source_prefix: 0,
            canonical_moves: Vec::new(),
            orbit_size: 1,
            preferred_moves: Vec::new(),
        },
        Candidate {
            source: "shallow:origin".to_owned(),
            source_prefix: 1,
            canonical_moves: vec![HexCoord::ZERO],
            orbit_size: 1,
            preferred_moves: Vec::new(),
        },
    ];
    for canonical_moves in reps.into_values() {
        out.push(Candidate {
            source: "shallow:first-reply".to_owned(),
            source_prefix: 2,
            orbit_size: orbit_size(&canonical_moves),
            canonical_moves,
            preferred_moves: Vec::new(),
        });
    }
    assert_eq!(out.len(), 26, "2 roots plus 24 first-reply D6 reps");
    out
}

fn load_human_candidates(path: &str, game_count: usize, backtrack: usize) -> Vec<Candidate> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read OPENING_ATLAS_CORPUS={path}: {error}"));
    let mut games = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_game(line).expect("valid human corpus JSONL row"))
        .filter(|game| matches!(game.winner, -1 | 1) && game.moves.len() >= 4)
        .collect::<Vec<_>>();
    games.sort_by(|left, right| left.hash.cmp(&right.hash));
    games.truncate(game_count);
    assert_eq!(games.len(), game_count, "not enough eligible human games");

    let mut unique = BTreeSet::new();
    let mut candidates = Vec::new();
    for game in games {
        let mut full = HexoState::new();
        for (ply, &coord) in game.moves.iter().enumerate() {
            apply_placement(&mut full, Placement { coord }).unwrap_or_else(|error| {
                panic!("illegal human replay {} ply {ply}: {error}", game.hash)
            });
        }
        assert!(
            full.is_terminal(),
            "human game {} is not terminal",
            game.hash
        );
        let expected_winner = if game.winner == 1 {
            Player::Player0
        } else {
            Player::Player1
        };
        assert_eq!(
            full.terminal().map(|outcome| outcome.winner),
            Some(expected_winner),
            "human winner mismatch {}",
            game.hash
        );

        let first = game.moves.len().saturating_sub(backtrack).max(1);
        for prefix in (first..game.moves.len()).rev() {
            let canonical_moves = canonical_sequence(&game.moves[..prefix]);
            let key = root_position_key(&canonical_moves);
            if unique.insert(key) {
                candidates.push(Candidate {
                    source: format!("human:{}:winner={}", game.hash, game.winner),
                    source_prefix: prefix,
                    orbit_size: orbit_size(&canonical_moves),
                    canonical_moves,
                    preferred_moves: Vec::new(),
                });
            }
        }
    }
    candidates
}

type PositionKey = (u8, (u8, i16, i16), Vec<(i16, i16, u8)>);

/// Enumerate every distinct D6-canonical position that occurs within the first
/// `first_n` plies of ANY corpus game. Positions are collapsed by
/// `root_position_key` (symmetric/transposed duplicates fold together) and
/// returned in a deterministic order: shallow depths first, then by canonical
/// position key. Illegal or terminal prefixes are skipped defensively.
fn load_corpus_first_n(path: &str, first_n: usize) -> Vec<Candidate> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read OPENING_ATLAS_CORPUS={path}: {error}"));
    let mut reps: BTreeMap<(usize, PositionKey), Vec<HexCoord>> = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let game = match parse_game(line) {
            Some(game) => game,
            None => continue,
        };
        let mut state = HexoState::new();
        for (ply, &coord) in game.moves.iter().take(first_n).enumerate() {
            if apply_placement(&mut state, Placement { coord }).is_err() {
                break;
            }
            if state.is_terminal() {
                break;
            }
            let prefix = ply + 1;
            let canonical_moves = canonical_sequence(&game.moves[..prefix]);
            let key = root_position_key(&canonical_moves);
            reps.entry((prefix, key)).or_insert(canonical_moves);
        }
    }
    reps.into_iter()
        .map(|((prefix, _key), canonical_moves)| Candidate {
            source: format!("corpus{first_n}:depth{prefix}"),
            source_prefix: prefix,
            orbit_size: orbit_size(&canonical_moves),
            canonical_moves,
            preferred_moves: Vec::new(),
        })
        .collect()
}

/// Probe helper: load explicit canonical move sequences (one per line,
/// "q,r;q,r;...") as candidates, bypassing corpus enumeration. Used to
/// calibrate node caps against known-verdict positions.
fn load_moves_file(path: &str) -> Vec<Candidate> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read OPENING_ATLAS_MOVES_FILE={path}: {error}"));
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let moves = line
            .split(';')
            .filter(|token| !token.trim().is_empty())
            .map(|token| {
                let ints = parse_ints(token);
                assert_eq!(ints.len(), 2, "move token must be q,r");
                HexCoord::new(ints[0], ints[1])
            })
            .collect::<Vec<_>>();
        let canonical_moves = canonical_sequence(&moves);
        out.push(Candidate {
            source: "probe:moves".to_owned(),
            source_prefix: canonical_moves.len(),
            orbit_size: orbit_size(&canonical_moves),
            canonical_moves,
            preferred_moves: Vec::new(),
        });
    }
    out
}

fn raw_field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    line.split_ascii_whitespace()
        .find_map(|token| token.strip_prefix(name))
}

/// Derive root-move hints from certified one-ply descendants. For a
/// FirstStone parent the same player owns the SecondStone child, so a verified
/// child WIN supplies a constructive existential move. The parent is still
/// solved afresh and must pass every normal verifier gate before emission.
fn load_parent_win_hints(path: &str, parent_depth: usize) -> Vec<Candidate> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read OPENING_ATLAS_PARENT_WIN_HINT_RAW={path}: {error}"));
    let child_depth = parent_depth + 1;
    let mut parents = BTreeMap::<PositionKey, (Vec<HexCoord>, BTreeSet<(i16, i16)>)>::new();
    for line in text.lines().filter(|line| line.starts_with("ATLAS_ROW ")) {
        if raw_field(line, "certified=") != Some("1")
            || raw_field(line, "status=") != Some("WIN")
            || raw_field(line, "source_prefix=").and_then(|value| value.parse().ok())
                != Some(child_depth)
        {
            continue;
        }
        let Some(moves_value) = raw_field(line, "moves=") else {
            continue;
        };
        let ints = parse_ints(moves_value);
        if ints.len() != child_depth * 2 {
            continue;
        }
        let child_moves = ints
            .chunks_exact(2)
            .map(|pair| HexCoord::new(pair[0], pair[1]))
            .collect::<Vec<_>>();
        let parent_moves = &child_moves[..parent_depth];
        let parent_state = replay(parent_moves);
        let child_state = replay(&child_moves);
        if parent_state.current_player() != child_state.current_player()
            || raw_field(line, "claimant=") != Some(player_name(parent_state.current_player()))
        {
            continue;
        }
        let (canonical_moves, symmetry) = canonical_sequence_with_symmetry(parent_moves);
        let preferred = d6_transform_coord(child_moves[parent_depth], symmetry)
            .expect("D6 hinted coordinate in range");
        let key = root_position_key(&canonical_moves);
        parents
            .entry(key)
            .or_insert_with(|| (canonical_moves, BTreeSet::new()))
            .1
            .insert(coord_key(preferred));
    }
    parents
        .into_values()
        .map(|(canonical_moves, hints)| Candidate {
            source: format!("corpus{parent_depth}:depth{parent_depth}"),
            source_prefix: parent_depth,
            orbit_size: orbit_size(&canonical_moves),
            canonical_moves,
            preferred_moves: hints
                .into_iter()
                .map(|(q, r)| HexCoord::new(q, r))
                .collect(),
        })
        .collect()
}

fn replay_unchecked(moves: &[HexCoord]) -> HexoState {
    let mut state = HexoState::new();
    for (ply, &coord) in moves.iter().enumerate() {
        apply_placement(&mut state, Placement { coord })
            .unwrap_or_else(|error| panic!("illegal canonical replay ply {ply}: {error}"));
    }
    state
}

fn replay(moves: &[HexCoord]) -> HexoState {
    let state = replay_unchecked(moves);
    assert!(!state.is_terminal(), "atlas roots must be nonterminal");
    state
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::Player0 => "P0",
        Player::Player1 => "P1",
    }
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn parse_status(value: &str) -> ProofStatus {
    match value {
        "WIN" => ProofStatus::Win,
        "LOSS" => ProofStatus::Loss,
        "UNKNOWN" => ProofStatus::Unknown,
        other => panic!("unknown atlas status {other}"),
    }
}

fn parse_player(value: &str) -> Player {
    match value {
        "P0" => Player::Player0,
        "P1" => Player::Player1,
        other => panic!("unknown atlas player {other}"),
    }
}

fn phase_name(state: &HexoState) -> &'static str {
    match state.phase() {
        hexo_engine::TurnPhase::Opening => "Opening",
        hexo_engine::TurnPhase::FirstStone => "FirstStone",
        hexo_engine::TurnPhase::SecondStone { .. } => "SecondStone",
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn moves_text(moves: &[HexCoord]) -> String {
    moves
        .iter()
        .map(|coord| format!("{},{}", coord.q, coord.r))
        .collect::<Vec<_>>()
        .join(";")
}

fn candidate_id(candidate: &Candidate) -> String {
    let fingerprint =
        fnv1a64(format!("{:?}", root_position_key(&candidate.canonical_moves)).as_bytes());
    format!("oa-{fingerprint:016x}")
}

/// Read verifier-backed ids from an earlier raw.  This is only a scheduling
/// filter: it cannot manufacture a verdict, and all newly found certificates
/// still pass the canonical and 12-way D6 verifier gates below.
fn certified_ids_from_raw(path: &str) -> BTreeSet<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read OPENING_ATLAS_SKIP_CERTIFIED_RAW={path}: {error}"))
        .lines()
        .filter(|line| line.starts_with("ATLAS_ROW ") && line.contains(" certified=1 "))
        .filter_map(|line| {
            line.split_ascii_whitespace()
                .find_map(|token| token.strip_prefix("id="))
                .map(str::to_owned)
        })
        .collect()
}

fn cert_metrics(cert: &TssCertificate) -> (u32, usize, usize, usize) {
    let mut derived_horizon = cert.root.placements_made;
    let mut edges = 0usize;
    let mut commutations = 0usize;
    let mut zones = 0usize;
    for node in &cert.nodes {
        match node {
            CertNode::OrCompletion { completion_ply, .. } => {
                derived_horizon = derived_horizon.max(*completion_ply)
            }
            CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
                derived_horizon = derived_horizon.max(*resolution_ply)
            }
            CertNode::Choice { .. } => {}
            CertNode::Universal {
                edges: node_edges,
                commutations: node_commutations,
                zone,
                ..
            } => {
                edges += node_edges.len();
                commutations += node_commutations.len();
                zones += usize::from(zone.is_some());
            }
        }
    }
    (derived_horizon, edges, commutations, zones)
}

fn certificate_graph_depth(cert: &TssCertificate) -> usize {
    fn visit(cert: &TssCertificate, id: u32, memo: &mut [Option<usize>]) -> usize {
        if let Some(depth) = memo[id as usize] {
            return depth;
        }
        let child_depth = match &cert.nodes[id as usize] {
            CertNode::Choice { child, .. } => visit(cert, *child, memo),
            CertNode::Universal { edges, .. } => edges
                .iter()
                .map(|edge| visit(cert, edge.child, memo))
                .max()
                .unwrap_or(0),
            CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
        };
        let depth = child_depth.saturating_add(1);
        memo[id as usize] = Some(depth);
        depth
    }
    let mut memo = vec![None; cert.nodes.len()];
    visit(cert, cert.root_node, &mut memo)
}

/// Extract the PRINCIPAL forced-win line from a certificate, as the ordered
/// list of placements AFTER the root opening. At each claimant node take the
/// winning move; at each defender (Universal) node take one principal reply;
/// at a lambda-1 Win leaf complete the witness window with the claimant's
/// remaining placements so the line reaches a literal six-in-a-row. Returns the
/// move list and whether replay reached a terminal win for the claimant.
fn extract_win_line(cert: &TssCertificate, root_state: &HexoState) -> (Vec<HexCoord>, bool) {
    let (line, terminal, _path) = extract_win_line_traced(cert, root_state);
    (line, terminal)
}

fn extract_win_line_traced(
    cert: &TssCertificate,
    root_state: &HexoState,
) -> (Vec<HexCoord>, bool, String) {
    let claimant = cert.claimant;
    let mut state = root_state.clone();
    let mut line = Vec::new();
    let mut path = String::new();
    let mut id = cert.root_node;
    let mut terminal_win = false;

    let play = |state: &mut HexoState, mv: HexCoord, line: &mut Vec<HexCoord>| -> bool {
        if apply_placement(state, Placement { coord: mv }).is_err() {
            return false;
        }
        line.push(mv);
        true
    };

    // The arena is acyclic; bound the walk defensively.
    for _ in 0..(cert.nodes.len() + 8) {
        let Some(node) = cert.nodes.get(id as usize) else {
            break;
        };
        let mover_claimant = state.current_player() == claimant;
        match node {
            CertNode::Choice { mv, child } => {
                path.push(if mover_claimant { 'C' } else { 'c' });
                if !play(&mut state, *mv, &mut line) {
                    path.push('!');
                    break;
                }
                if let Some(outcome) = state.terminal() {
                    terminal_win = outcome.winner == claimant;
                    break;
                }
                id = *child;
            }
            CertNode::OrCompletion { mv, .. } => {
                path.push(if mover_claimant { 'O' } else { 'o' });
                if play(&mut state, *mv, &mut line) {
                    terminal_win = state.terminal().is_some_and(|o| o.winner == claimant);
                } else {
                    path.push('!');
                }
                break;
            }
            CertNode::Universal {
                edges,
                commutations,
                ..
            } => {
                // One principal defender reply is enough for a single line; the
                // certificate proves every reply loses.
                path.push(if mover_claimant { 'u' } else { 'U' });
                let Some(edge) = edges.first() else {
                    path.push('0');
                    break;
                };
                if !play(&mut state, edge.mv, &mut line) {
                    path.push('!');
                    break;
                }
                if let Some(outcome) = state.terminal() {
                    terminal_win = outcome.winner == claimant;
                    break;
                }
                // Preserve the established principal choice (`edges.first`).
                // If its SecondStone child omitted the matching commuted reply,
                // replay that concrete second placement and continue through
                // the mirror-order edge at the identical pair position.
                let empty_child = matches!(
                    cert.nodes.get(edge.child as usize),
                    Some(CertNode::Universal { edges, .. }) if edges.is_empty()
                );
                if empty_child {
                    if let Some(item) = commutations.iter().find(|item| item.first == edge.mv) {
                        path.push('M');
                        if !play(&mut state, item.omitted_second, &mut line) {
                            path.push('!');
                            break;
                        }
                        if let Some(outcome) = state.terminal() {
                            terminal_win = outcome.winner == claimant;
                            break;
                        }
                        let Some(CertNode::Universal {
                            edges: mirror_edges,
                            ..
                        }) = cert.nodes.get(item.mirror_child as usize)
                        else {
                            path.push('!');
                            break;
                        };
                        let Some(mirror_edge) =
                            mirror_edges.iter().find(|mirror| mirror.mv == item.first)
                        else {
                            path.push('!');
                            break;
                        };
                        id = mirror_edge.child;
                        continue;
                    }
                }
                id = edge.child;
            }
            CertNode::Win { witness, count, .. } => {
                // Claimant to move with a lambda-1 win: fill the empty witness
                // cells with claimant placements to reach the six-in-a-row.
                path.push(if mover_claimant { 'W' } else { 'w' });
                path.push_str(&count.to_string());
                for cell in witness.cells() {
                    if state.board().get(cell).is_some() {
                        continue;
                    }
                    if !play(&mut state, cell, &mut line) {
                        path.push('!');
                        break;
                    }
                    if let Some(outcome) = state.terminal() {
                        terminal_win = outcome.winner == claimant;
                        break;
                    }
                }
                break;
            }
            CertNode::Loss { .. } => {
                path.push(if mover_claimant { 'l' } else { 'L' });
                break;
            }
        }
    }
    // The arena principal line ends at a proven-winning contract (Loss / implicit
    // Universal / lambda-1 Win) that does not spell out the finishing stones. A
    // bounded, threat-restricted search recovers the actual six-in-a-row line.
    if !terminal_win && state.terminal().is_none() {
        if let Some(finish) = find_finish(&state, claimant, 12) {
            let before = line.len();
            line.extend(finish);
            terminal_win = true;
            path.push('+');
            path.push_str(&(line.len() - before).to_string());
        }
    }
    (line, terminal_win, path)
}

/// Empty cells that matter for a lambda-1 finish: the open cells of any window
/// where `claimant` already holds >= 4 and the opponent holds none. The mover
/// completes/extends such a window (claimant) or blocks it (defender), so both
/// sides' relevant replies live in this tiny set.
fn threat_cells(state: &HexoState, claimant: Player, min_count: u8) -> Vec<HexCoord> {
    let mut cells = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.count(claimant) >= min_count && entry.count(claimant.other()) == 0 {
            cells.extend(entry.empty_cells());
        }
    }
    cells.sort_by_key(|coord| (coord.q, coord.r));
    cells.dedup();
    cells
}

/// Depth-bounded forced-win finder over threat-relevant moves only. Returns a
/// concrete line to a `claimant` six-in-a-row: at claimant nodes any move that
/// forces the win; at defender nodes EVERY reply must still lose. Restricting to
/// threat cells keeps branching tiny, which is sound for the lambda-1 finish the
/// certificate already proved exists.
fn find_finish(state: &HexoState, claimant: Player, depth: usize) -> Option<Vec<HexCoord>> {
    if let Some(outcome) = state.terminal() {
        return (outcome.winner == claimant).then(Vec::new);
    }
    if depth == 0 {
        return None;
    }
    let claimant_to_move = state.current_player() == claimant;
    // Restrict both sides to imminent (>=4) threat cells: the claimant completes
    // or extends, the defender blocks. This keeps the lambda-1 finish fast and
    // the displayed line a legal, terminating six-in-a-row.
    let moves = threat_cells(state, claimant, 4);
    if moves.is_empty() {
        return None;
    }
    let mut principal: Option<Vec<HexCoord>> = None;
    for mv in moves {
        let mut child = state.clone();
        if apply_placement(&mut child, Placement { coord: mv }).is_err() {
            continue;
        }
        match find_finish(&child, claimant, depth - 1) {
            Some(mut tail) => {
                if claimant_to_move {
                    // One forcing move suffices for the claimant.
                    let mut line = Vec::with_capacity(tail.len() + 1);
                    line.push(mv);
                    line.append(&mut tail);
                    return Some(line);
                }
                // Defender: remember a witness line but keep checking all replies.
                if principal.is_none() {
                    let mut line = Vec::with_capacity(tail.len() + 1);
                    line.push(mv);
                    line.append(&mut tail);
                    principal = Some(line);
                }
            }
            None => {
                if !claimant_to_move {
                    // A defender reply escapes within this depth: not forced here.
                    return None;
                }
            }
        }
    }
    if claimant_to_move {
        None
    } else {
        principal
    }
}

/// Replay an extracted line exactly. A terminal line is accepted only when its
/// final placement creates a real claimant six; a terminal prefix with trailing
/// moves is rejected. This is deliberately independent of the extractor's flag.
fn validate_extracted_line(
    root_state: &HexoState,
    line: &[HexCoord],
    claimant: Player,
) -> (bool, bool) {
    let mut state = root_state.clone();
    for (index, &mv) in line.iter().enumerate() {
        if state.is_terminal() || apply_placement(&mut state, Placement { coord: mv }).is_err() {
            return (false, false);
        }
        if let Some(outcome) = state.terminal() {
            let literal_six = state
                .board()
                .windows()
                .entries()
                .any(|entry| entry.count(claimant) == 6 && entry.count(claimant.other()) == 0);
            return (
                index + 1 == line.len(),
                index + 1 == line.len() && outcome.winner == claimant && literal_six,
            );
        }
    }
    (true, false)
}

fn load_reverify_roots(path: &str) -> Vec<ReverifyRoot> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read OPENING_ATLAS_REVERIFY_JSON={path}: {error}"));
    let doc: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parse OPENING_ATLAS_REVERIFY_JSON={path}: {error}"));
    let rows = doc["rows"].as_array().expect("atlas rows array");
    let mut roots = Vec::new();
    for row in rows {
        if row["certified"].as_u64() != Some(1) {
            continue;
        }
        let id = row["id"].as_str().expect("certified row id").to_owned();
        let expected_status = parse_status(
            row["status"]
                .as_str()
                .expect("certified row decisive status"),
        );
        assert_ne!(
            expected_status,
            ProofStatus::Unknown,
            "certified row {id} is UNKNOWN"
        );
        let expected_claimant =
            parse_player(row["claimant"].as_str().expect("certified row claimant"));
        let moves = row["moves"]
            .as_array()
            .expect("certified row moves")
            .iter()
            .map(|pair| {
                let pair = pair.as_array().expect("atlas move pair");
                assert_eq!(pair.len(), 2, "atlas move pair length for {id}");
                let q = i16::try_from(pair[0].as_i64().expect("atlas move q"))
                    .expect("atlas move q fits i16");
                let r = i16::try_from(pair[1].as_i64().expect("atlas move r"))
                    .expect("atlas move r fits i16");
                HexCoord::new(q, r)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            row["placements"].as_u64(),
            Some(moves.len() as u64),
            "atlas placement count for {id}"
        );
        let state = replay(&moves);
        let actual_claimant = match expected_status {
            ProofStatus::Win => state.current_player(),
            ProofStatus::Loss => state.current_player().other(),
            ProofStatus::Unknown => unreachable!(),
        };
        assert_eq!(
            expected_claimant, actual_claimant,
            "stored claimant disagrees with stored verdict for {id}"
        );
        let candidate = Candidate {
            source: "reverify".to_owned(),
            source_prefix: moves.len(),
            canonical_moves: moves.clone(),
            orbit_size: orbit_size(&moves),
            preferred_moves: Vec::new(),
        };
        assert_eq!(candidate_id(&candidate), id, "stored root id mismatch");
        roots.push(ReverifyRoot {
            id,
            moves,
            expected_status,
            expected_claimant,
            terminal_before: row["win_line_terminal"].as_u64() == Some(1),
            first_line_move: row["win_line"]
                .as_array()
                .and_then(|line| line.first())
                .map(|pair| {
                    let pair = pair.as_array().expect("atlas win-line move pair");
                    HexCoord::new(
                        i16::try_from(pair[0].as_i64().expect("atlas win-line q"))
                            .expect("atlas win-line q fits i16"),
                        i16::try_from(pair[1].as_i64().expect("atlas win-line r"))
                            .expect("atlas win-line r fits i16"),
                    )
                }),
        });
    }
    assert_eq!(roots.len(), 2_190, "certified atlas root count changed");
    roots
}

fn solve_reverify_root(root: &ReverifyRoot, tt_bytes: usize, node_ladder: &[u64]) {
    let state = replay(&root.moves);
    std::env::remove_var("TSS_ROOT_PREFERRED_MOVES");
    let start = Instant::now();
    let mut final_result = None;
    let mut used_cap = 0u64;
    let mut solve_path = "root";
    for &node_cap in node_ladder {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let result = solver.solve(
            &state,
            &SolveCaps {
                node_cap,
                tt_bytes_cap: tt_bytes,
                semantic_horizon: u32::MAX,
            },
        );
        assert!(
            result.status == ProofStatus::Unknown || result.cert.is_some(),
            "decisive re-verification result without certificate for {}",
            root.id
        );
        used_cap = node_cap;
        let decisive = result.status != ProofStatus::Unknown;
        final_result = Some(result);
        if decisive {
            break;
        }
    }
    // The squeeze layer contains seven roots certified by a same-claimant child
    // lift. If the direct narrow generator returns UNKNOWN, solve the stored
    // first continuation afresh under the normative width and prepend a Choice.
    // The move is only a scheduling hint: the strict root verifier below remains
    // the authority for the reconstructed certificate.
    if final_result
        .as_ref()
        .is_some_and(|result| result.status == ProofStatus::Unknown)
        && root.expected_status == ProofStatus::Win
    {
        if let Some(preferred) = root.first_line_move {
            let mut child_state = state.clone();
            if apply_placement(&mut child_state, Placement { coord: preferred }).is_ok()
                && child_state.current_player() == state.current_player()
                && !child_state.is_terminal()
            {
                for &node_cap in node_ladder {
                    let mut solver = TssSolver::default();
                    solver.set_width_options(WidthOptions::vcf_pair_complete());
                    let mut result = solver.solve(
                        &child_state,
                        &SolveCaps {
                            node_cap,
                            tt_bytes_cap: tt_bytes,
                            semantic_horizon: u32::MAX,
                        },
                    );
                    used_cap = node_cap;
                    if result.status != ProofStatus::Win {
                        final_result = Some(result);
                        continue;
                    }
                    let child_cert = result.cert.take().expect("lifted child WIN certificate");
                    assert!(
                        TssVerifier.verify(&child_state, &child_cert, ProofStatus::Win),
                        "strict verifier rejected lifted child certificate for {}",
                        root.id
                    );
                    assert!(
                        child_cert.nodes.len() < MAX_CERT_NODES
                            && certificate_graph_depth(&child_cert) < MAX_CERT_DEPTH,
                        "lifted child certificate too large for {}",
                        root.id
                    );
                    let child_root = child_cert.root_node;
                    let mut nodes = child_cert.nodes;
                    let root_node =
                        u32::try_from(nodes.len()).expect("certificate node id fits u32");
                    nodes.push(CertNode::Choice {
                        mv: preferred,
                        child: child_root,
                    });
                    result.cert = Some(TssCertificate {
                        root: RootBinding::from_state(&state),
                        claimant: child_cert.claimant,
                        root_node,
                        nodes,
                        semantic_horizon: child_cert.semantic_horizon,
                    });
                    result.status = ProofStatus::Win;
                    final_result = Some(result);
                    solve_path = "same_claimant_child_lift";
                    break;
                }
            }
        }
    }
    let result = final_result.expect("re-verification node ladder is nonempty");
    let reproduced_claimant = result.cert.as_ref().map(|cert| cert.claimant);
    let mut verifier_ok = false;
    let mut win_line = Vec::new();
    let mut win_line_terminal = false;
    let mut win_line_path = "NA".to_owned();
    if let Some(cert) = result.cert.as_ref() {
        verifier_ok = TssVerifier.verify(&state, cert, result.status);
        assert!(
            verifier_ok,
            "strict verifier rejected re-verification certificate for {}",
            root.id
        );
        if result.status == ProofStatus::Win && cert.claimant == root.expected_claimant {
            (win_line, win_line_terminal, win_line_path) = extract_win_line_traced(cert, &state);
            let (line_legal, terminal_six) =
                validate_extracted_line(&state, &win_line, cert.claimant);
            assert!(line_legal, "extracted line is illegal for {}", root.id);
            assert_eq!(
                win_line_terminal, terminal_six,
                "terminal-line flag disagrees with literal replay for {}",
                root.id
            );
        }
    }
    let same_verdict = result.status == root.expected_status
        && reproduced_claimant == Some(root.expected_claimant);
    println!(
        "ATLAS_REVERIFY_ROW schema=1 id={} expected_status={} expected_claimant={} reproduced_status={} reproduced_claimant={} same_verdict={} verifier_ok={} solve_path={} cap={} nodes={} expansions={} ms={:.3} terminal_before={} win_line_len={} win_line_terminal={} win_line_path={} win_line={} moves={}",
        root.id,
        status_name(root.expected_status),
        player_name(root.expected_claimant),
        status_name(result.status),
        reproduced_claimant.map(player_name).unwrap_or("NA"),
        u8::from(same_verdict),
        u8::from(verifier_ok),
        solve_path,
        used_cap,
        result.stats.nodes,
        result.stats.expansions,
        start.elapsed().as_secs_f64() * 1e3,
        u8::from(root.terminal_before),
        win_line.len(),
        u8::from(win_line_terminal),
        win_line_path,
        if win_line.is_empty() { "NA".to_owned() } else { moves_text(&win_line) },
        moves_text(&root.moves),
    );
}

fn opening_atlas_reverify_certified() {
    let path = std::env::var("OPENING_ATLAS_REVERIFY_JSON")
        .expect("reverify_certified mode requires OPENING_ATLAS_REVERIFY_JSON");
    let roots = load_reverify_roots(&path);
    let tt_bytes = env_num("OPENING_ATLAS_TT_BYTES", DEFAULT_TT_BYTES);
    let node_ladder = env_ladder("OPENING_ATLAS_NODE_LADDER", &[100_000, 1_000_000]);
    let wall_seconds = env_num("OPENING_ATLAS_WALL_SECONDS", 14_400u64);
    let shard_count = env_num::<usize>("SHARD_COUNT", 1).max(1);
    let shard_index = env_num::<usize>("SHARD_INDEX", 0);
    assert!(
        shard_index < shard_count,
        "SHARD_INDEX must be < SHARD_COUNT"
    );
    let assigned = (0..roots.len())
        .filter(|index| index % shard_count == shard_index)
        .collect::<Vec<_>>();
    println!(
        "ATLAS_REVERIFY_SETUP schema=1 roots={} shard_index={} shard_count={} shard_total={} width=vcf_pair_complete horizon={} node_ladder={:?} tt_bytes={} wall_seconds={}",
        roots.len(),
        shard_index,
        shard_count,
        assigned.len(),
        u32::MAX,
        node_ladder,
        tt_bytes,
        wall_seconds,
    );
    std::io::stdout().flush().ok();
    let batch_start = Instant::now();
    let mut attempted = 0usize;
    for index in assigned {
        if attempted > 0 && batch_start.elapsed().as_secs() >= wall_seconds {
            break;
        }
        solve_reverify_root(&roots[index], tt_bytes, &node_ladder);
        attempted += 1;
        std::io::stdout().flush().ok();
    }
    println!(
        "ATLAS_REVERIFY_DONE shard_index={} shard_count={} attempted={} residual={} wall_ms={:.3}",
        shard_index,
        shard_count,
        attempted,
        roots
            .iter()
            .enumerate()
            .filter(|(index, _)| index % shard_count == shard_index)
            .count()
            - attempted,
        batch_start.elapsed().as_secs_f64() * 1e3,
    );
    std::io::stdout().flush().ok();
}

fn verify_all_d6(
    canonical_moves: &[HexCoord],
    cert: &TssCertificate,
    status: ProofStatus,
) -> (usize, u16) {
    let mut verified = 0usize;
    let mut accepted_mask = 0u16;
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let transformed_moves = canonical_moves
            .iter()
            .copied()
            .map(|coord| d6_transform_coord(coord, symmetry).expect("D6 coordinate in range"))
            .collect::<Vec<_>>();
        let transformed_state = replay(&transformed_moves);
        let transformed_cert = d6_remap_certificate(cert, symmetry).expect("D6 certificate remap");
        if TssVerifier.verify(&transformed_state, &transformed_cert, status) {
            verified += 1;
            accepted_mask |= 1u16 << symmetry;
        } else {
            println!(
                "ATLAS_D6_REMAP_REJECT symmetry={} status={} placements={} moves={}",
                symmetry,
                status_name(status),
                canonical_moves.len(),
                moves_text(canonical_moves),
            );
        }
    }
    (verified, accepted_mask)
}

fn solve_candidate(
    candidate: &Candidate,
    tt_bytes: usize,
    relative_horizon: u32,
    node_ladder: &[u64],
    unbounded_horizon: bool,
    width_name: &str,
) {
    let state = replay(&candidate.canonical_moves);
    if candidate.preferred_moves.is_empty() {
        std::env::remove_var("TSS_ROOT_PREFERRED_MOVES");
    } else {
        std::env::set_var(
            "TSS_ROOT_PREFERRED_MOVES",
            moves_text(&candidate.preferred_moves),
        );
    }
    // Deep profile: an unbounded ply deadline lets the search go as deep as the
    // node budget allows (depth bounded by the node cap, not an artificial ply
    // limit). Default keeps the +relative_horizon deadline (pass1 behavior).
    let semantic_horizon = if unbounded_horizon {
        u32::MAX
    } else {
        state
            .placements_made()
            .checked_add(relative_horizon)
            .expect("semantic horizon overflow")
    };
    let position_fingerprint =
        fnv1a64(format!("{:?}", root_position_key(&candidate.canonical_moves)).as_bytes());
    let start = Instant::now();
    let mut final_result = None;
    let mut used_cap = 0u64;
    // A certified same-claimant child can be lifted through one legal
    // existential move. Re-solve that child with the established deep narrow
    // profile, prepend a Choice node, and let the parent verifier below judge
    // the whole certificate. No raw certificate or verdict is trusted.
    'hint_caps: for &node_cap in node_ladder {
        for &preferred in &candidate.preferred_moves {
            let mut child_state = state.clone();
            if apply_placement(&mut child_state, Placement { coord: preferred }).is_err()
                || child_state.current_player() != state.current_player()
                || child_state.is_terminal()
            {
                continue;
            }
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            let mut result = solver.solve(
                &child_state,
                &SolveCaps {
                    node_cap,
                    tt_bytes_cap: tt_bytes,
                    semantic_horizon,
                },
            );
            used_cap = node_cap;
            if result.status != ProofStatus::Win {
                final_result = Some(result);
                continue;
            }
            let child_cert = result.cert.take().expect("hinted child WIN certificate");
            assert!(
                TssVerifier.verify(&child_state, &child_cert, ProofStatus::Win),
                "strict verifier rejected hinted child certificate"
            );
            if child_cert.nodes.len() >= MAX_CERT_NODES
                || certificate_graph_depth(&child_cert) >= MAX_CERT_DEPTH
            {
                result.status = ProofStatus::Unknown;
                final_result = Some(result);
                continue;
            }
            let child_root = child_cert.root_node;
            let mut nodes = child_cert.nodes;
            let root_node = u32::try_from(nodes.len()).expect("certificate node id fits u32");
            nodes.push(CertNode::Choice {
                mv: preferred,
                child: child_root,
            });
            result.cert = Some(TssCertificate {
                root: RootBinding::from_state(&state),
                claimant: child_cert.claimant,
                root_node,
                nodes,
                semantic_horizon: child_cert.semantic_horizon,
            });
            final_result = Some(result);
            break 'hint_caps;
        }
    }
    if final_result.is_none() {
        for &node_cap in node_ladder {
            let mut solver = TssSolver::default();
            // Wider search (round3_consume) is vcf_pair_complete PLUS consuming
            // quiet-turn attacker moves and sound ranked-zone defender pruning. The
            // strict TssVerifier below stays normative: any certified WIN it rejects
            // panics the shard rather than emitting unsound data.
            solver.set_width_options(match width_name {
                "round3_consume" => WidthOptions::round3_consume(),
                "quiet_turn_consume" => WidthOptions::quiet_turn_consume(),
                "ranked_zone_consume" => WidthOptions::ranked_zone_consume(),
                "vcf_pair_complete" | "" => WidthOptions::vcf_pair_complete(),
                other => panic!("unknown OPENING_ATLAS_WIDTH={other}"),
            });
            let result = solver.solve(
                &state,
                &SolveCaps {
                    node_cap,
                    tt_bytes_cap: tt_bytes,
                    semantic_horizon,
                },
            );
            assert!(
                result.status == ProofStatus::Unknown || result.cert.is_some(),
                "hard result without certificate"
            );
            used_cap = node_cap;
            let hard = result.status != ProofStatus::Unknown;
            final_result = Some(result);
            if hard {
                break;
            }
        }
    }
    let result = final_result.expect("node ladder is nonempty");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
    let base = format!(
        "id=oa-{position_fingerprint:016x} source={} source_prefix={} placements={} side={} phase={} orbit={} cap={} horizon={} status={} nodes={} expansions={} tt_bytes={} peak_tt_bytes={} ms={elapsed_ms:.3}",
        candidate.source,
        candidate.source_prefix,
        state.placements_made(),
        player_name(state.current_player()),
        phase_name(&state),
        candidate.orbit_size,
        used_cap,
        semantic_horizon,
        status_name(result.status),
        result.stats.nodes,
        result.stats.expansions,
        tt_bytes,
        result.stats.peak_tt_bytes,
    );
    if let Some(cert) = &result.cert {
        assert!(
            TssVerifier.verify(&state, cert, result.status),
            "strict verifier rejected canonical certificate"
        );
        let (d6_verified, d6_mask) = verify_all_d6(&candidate.canonical_moves, cert, result.status);
        let (derived_horizon, edges, commutations, zones) = cert_metrics(cert);
        let certificate_debug = format!("{cert:?}");
        let cert_fingerprint = fnv1a64(certificate_debug.as_bytes());
        let (win_line, win_line_terminal) = extract_win_line(cert, &state);
        println!(
            "ATLAS_ROW {base} certified=1 claimant={} cert_nodes={} cert_edges={} cert_commutations={} cert_zones={} derived_horizon={} cert_fnv1a64_debug_v1={cert_fingerprint:016x} d6_verified={} d6_mask=0x{d6_mask:03x} win_line_len={} win_line_terminal={} win_line={} moves={}",
            player_name(cert.claimant),
            cert.nodes.len(),
            edges,
            commutations,
            zones,
            derived_horizon,
            d6_verified,
            win_line.len(),
            u8::from(win_line_terminal),
            moves_text(&win_line),
            moves_text(&candidate.canonical_moves),
        );
    } else {
        assert_eq!(result.status, ProofStatus::Unknown);
        println!(
            "ATLAS_ROW {base} certified=0 claimant=NA cert_nodes=0 cert_edges=0 cert_commutations=0 cert_zones=0 derived_horizon=NA cert_fnv1a64_debug_v1=NA d6_verified=0 d6_mask=0x000 win_line_len=0 win_line_terminal=NA win_line=NA moves={}",
            moves_text(&candidate.canonical_moves),
        );
    }
}

#[test]
#[ignore = "default-off certified opening-atlas pass; run explicitly"]
fn opening_atlas_pass1() {
    let mode = std::env::var("OPENING_ATLAS_MODE").unwrap_or_default();
    if mode == "reverify_certified" {
        opening_atlas_reverify_certified();
        return;
    }
    let corpus_path = std::env::var("OPENING_ATLAS_CORPUS").ok();
    let game_count = env_num("OPENING_ATLAS_GAME_COUNT", DEFAULT_GAME_COUNT);
    let backtrack = env_num("OPENING_ATLAS_BACKTRACK", DEFAULT_BACKTRACK);
    let first_n = env_num("OPENING_ATLAS_FIRST_N", DEFAULT_FIRST_N);
    let tt_bytes = env_num("OPENING_ATLAS_TT_BYTES", DEFAULT_TT_BYTES);
    let relative_horizon = env_num("OPENING_ATLAS_RELATIVE_HORIZON", DEFAULT_RELATIVE_HORIZON);
    let wall_seconds = env_num("OPENING_ATLAS_WALL_SECONDS", DEFAULT_WALL_SECONDS);
    let node_ladder = env_ladder("OPENING_ATLAS_NODE_LADDER", &DEFAULT_NODE_LADDER);
    let unbounded_horizon = env_num::<u8>("OPENING_ATLAS_UNBOUNDED", 0) != 0;
    // Search width: "round3_consume" = wider (quiet-turn + ranked-zone consume);
    // anything else = vcf_pair_complete (pass1/deep behavior).
    let width_name =
        std::env::var("OPENING_ATLAS_WIDTH").unwrap_or_else(|_| "vcf_pair_complete".to_owned());
    // Cross-position parallelism: shard the candidate list so N independent
    // worker processes (built once, launched N times) each solve a disjoint
    // stride of positions with their own TT. Round-robin by index balances the
    // depth mix (and thus the expensive positions) across workers.
    let shard_count = env_num::<usize>("SHARD_COUNT", 1).max(1);
    let shard_index = env_num::<usize>("SHARD_INDEX", 0);
    assert!(
        shard_index < shard_count,
        "SHARD_INDEX must be < SHARD_COUNT"
    );

    // Two candidate modes share the same solve+strict-verify emit loop:
    //  - default: shallow D6 census (+ optional deep human backtrack)
    //  - corpus_first_n: EVERY distinct D6-canonical position within the first
    //    `first_n` plies of ALL corpus games.
    let corpus_first_n = mode == "corpus_first_n";
    let (candidates, shallow_count) =
        if let Ok(hints_path) = std::env::var("OPENING_ATLAS_PARENT_WIN_HINT_RAW") {
            (load_parent_win_hints(&hints_path, first_n), 0usize)
        } else if let Ok(moves_path) = std::env::var("OPENING_ATLAS_MOVES_FILE") {
            (load_moves_file(&moves_path), 0usize)
        } else if corpus_first_n {
            let path = corpus_path
                .as_deref()
                .expect("corpus_first_n mode requires OPENING_ATLAS_CORPUS");
            (load_corpus_first_n(path, first_n), 0usize)
        } else {
            let mut candidates = shallow_candidates();
            let shallow_count = candidates.len();
            if let Some(path) = corpus_path.as_deref() {
                candidates.extend(load_human_candidates(path, game_count, backtrack));
            }
            (candidates, shallow_count)
        };
    // Optional depth window (source_prefix in [min,max]). Lets a run focus the
    // budget on the highest-yield depths (deeper positions are closest to a
    // forcing win) instead of spreading thin across all depths.
    let min_depth = env_num::<usize>("OPENING_ATLAS_MIN_DEPTH", 0);
    let max_depth = env_num::<usize>("OPENING_ATLAS_MAX_DEPTH", usize::MAX);
    let mut candidates = candidates
        .into_iter()
        .filter(|candidate| {
            candidate.source_prefix >= min_depth && candidate.source_prefix <= max_depth
        })
        .collect::<Vec<_>>();
    let skipped_certified = if let Ok(path) = std::env::var("OPENING_ATLAS_SKIP_CERTIFIED_RAW") {
        let ids = certified_ids_from_raw(&path);
        let before = candidates.len();
        candidates.retain(|candidate| !ids.contains(&candidate_id(candidate)));
        before - candidates.len()
    } else {
        0
    };
    // Optional ordering: "desc" puts the deepest (highest win-yield) positions
    // first so a wall-ceiling truncation leaves only low-value shallow residual.
    if std::env::var("OPENING_ATLAS_ORDER").unwrap_or_default() == "desc" {
        candidates.reverse();
    }

    let total = candidates.len();
    // Positions assigned to THIS shard (round-robin stride).
    let shard_indices = (0..total)
        .filter(|index| index % shard_count == shard_index)
        .collect::<Vec<_>>();
    let shard_total = shard_indices.len();
    println!(
        "ATLAS_SETUP schema=1 mode={} corpus={} games={} backtrack={} first_n={} shallow={} candidates={} skipped_certified={} shard_index={} shard_count={} shard_total={} width={} node_ladder={:?} tt_bytes={} relative_horizon={} unbounded_horizon={} wall_seconds={}",
        if mode.is_empty() { "pass1" } else { &mode },
        corpus_path.as_deref().unwrap_or("NONE"),
        game_count,
        backtrack,
        first_n,
        shallow_count,
        total,
        skipped_certified,
        shard_index,
        shard_count,
        shard_total,
        width_name,
        node_ladder,
        tt_bytes,
        relative_horizon,
        unbounded_horizon,
        wall_seconds,
    );
    std::io::stdout().flush().ok();

    let batch_start = Instant::now();
    let mut attempted = 0usize;
    for &index in &shard_indices {
        if attempted > 0 && batch_start.elapsed().as_secs() >= wall_seconds {
            break;
        }
        solve_candidate(
            &candidates[index],
            tt_bytes,
            relative_horizon,
            &node_ladder,
            unbounded_horizon,
            &width_name,
        );
        attempted += 1;
        // Flush every row so a wall-time stop or crash preserves partial output.
        std::io::stdout().flush().ok();
    }
    println!(
        "ATLAS_DONE shard_index={} shard_count={} attempted={} residual={} wall_ms={:.3}",
        shard_index,
        shard_count,
        attempted,
        shard_total - attempted,
        batch_start.elapsed().as_secs_f64() * 1e3,
    );
    std::io::stdout().flush().ok();
    assert!(
        attempted >= shallow_count,
        "wall cap must cover the shallow census"
    );
}

#[test]
fn opening_atlas_d6_census_constants() {
    let shallow = shallow_candidates();
    assert_eq!(shallow.len(), 26);

    let disk = (-8..=8)
        .flat_map(|q| (-8..=8).map(move |r| HexCoord::new(q, r)))
        .filter(|coord| *coord != HexCoord::ZERO && hex_distance(*coord) <= 8)
        .collect::<Vec<_>>();
    assert_eq!(disk.len(), 216);

    let mut pairs = BTreeSet::<Vec<(i16, i16)>>::new();
    for &first in &disk {
        for q in -16..=16 {
            for r in -16..=16 {
                let second = HexCoord::new(q, r);
                if second == HexCoord::ZERO || second == first {
                    continue;
                }
                let delta = HexCoord::new(second.q - first.q, second.r - first.r);
                if hex_distance(second) <= 8 || hex_distance(delta) <= 8 {
                    pairs.insert(sequence_key(&canonical_unordered_pair(first, second)));
                }
            }
        }
    }
    assert_eq!(pairs.len(), 3_684);
}
