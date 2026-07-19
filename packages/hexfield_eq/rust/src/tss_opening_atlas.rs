//! Default-off certified opening-atlas harness.
//!
//! Run only via the ignored `opening_atlas_pass1` test. Hard rows are printed
//! only after the independent strict verifier accepts the returned certificate
//! at the canonical root and all 12 D6-remapped roots.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player};

use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::{
    d6_remap_certificate, d6_transform_coord, CertNode, RootBinding, TssCertificate, TssVerifier,
    D6_SYMMETRY_COUNT,
};

const DEFAULT_TT_BYTES: usize = 512 << 20;
const DEFAULT_NODE_LADDER: [u64; 2] = [10_000, 100_000];
const DEFAULT_RELATIVE_HORIZON: u32 = 12;
const DEFAULT_GAME_COUNT: usize = 8;
const DEFAULT_BACKTRACK: usize = 12;
const DEFAULT_WALL_SECONDS: u64 = 1_200;

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

fn canonical_sequence(moves: &[HexCoord]) -> Vec<HexCoord> {
    (0..D6_SYMMETRY_COUNT)
        .map(|symmetry| {
            moves
                .iter()
                .copied()
                .map(|coord| d6_transform_coord(coord, symmetry).expect("D6 coordinate in range"))
                .collect::<Vec<_>>()
        })
        .min_by_key(|image| (root_position_key(image), sequence_key(image)))
        .expect("D6 is nonempty")
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
        },
        Candidate {
            source: "shallow:origin".to_owned(),
            source_prefix: 1,
            canonical_moves: vec![HexCoord::ZERO],
            orbit_size: 1,
        },
    ];
    for canonical_moves in reps.into_values() {
        out.push(Candidate {
            source: "shallow:first-reply".to_owned(),
            source_prefix: 2,
            orbit_size: orbit_size(&canonical_moves),
            canonical_moves,
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
                });
            }
        }
    }
    candidates
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

fn solve_candidate(candidate: &Candidate, tt_bytes: usize, relative_horizon: u32) {
    let state = replay(&candidate.canonical_moves);
    let semantic_horizon = state
        .placements_made()
        .checked_add(relative_horizon)
        .expect("semantic horizon overflow");
    let position_fingerprint =
        fnv1a64(format!("{:?}", root_position_key(&candidate.canonical_moves)).as_bytes());
    let start = Instant::now();
    let mut final_result = None;
    let mut used_cap = 0u64;
    for node_cap in DEFAULT_NODE_LADDER {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
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
        println!(
            "ATLAS_ROW {base} certified=1 claimant={} cert_nodes={} cert_edges={} cert_commutations={} cert_zones={} derived_horizon={} cert_fnv1a64_debug_v1={cert_fingerprint:016x} d6_verified={} d6_mask=0x{d6_mask:03x} moves={}",
            player_name(cert.claimant),
            cert.nodes.len(),
            edges,
            commutations,
            zones,
            derived_horizon,
            d6_verified,
            moves_text(&candidate.canonical_moves),
        );
    } else {
        assert_eq!(result.status, ProofStatus::Unknown);
        println!(
            "ATLAS_ROW {base} certified=0 claimant=NA cert_nodes=0 cert_edges=0 cert_commutations=0 cert_zones=0 derived_horizon=NA cert_fnv1a64_debug_v1=NA d6_verified=0 d6_mask=0x000 moves={}",
            moves_text(&candidate.canonical_moves),
        );
    }
}

#[test]
#[ignore = "default-off certified opening-atlas pass; run explicitly"]
fn opening_atlas_pass1() {
    let corpus_path = std::env::var("OPENING_ATLAS_CORPUS").ok();
    let game_count = env_num("OPENING_ATLAS_GAME_COUNT", DEFAULT_GAME_COUNT);
    let backtrack = env_num("OPENING_ATLAS_BACKTRACK", DEFAULT_BACKTRACK);
    let tt_bytes = env_num("OPENING_ATLAS_TT_BYTES", DEFAULT_TT_BYTES);
    let relative_horizon = env_num("OPENING_ATLAS_RELATIVE_HORIZON", DEFAULT_RELATIVE_HORIZON);
    let wall_seconds = env_num("OPENING_ATLAS_WALL_SECONDS", DEFAULT_WALL_SECONDS);

    let mut candidates = shallow_candidates();
    let shallow_count = candidates.len();
    if let Some(path) = corpus_path.as_deref() {
        candidates.extend(load_human_candidates(path, game_count, backtrack));
    }
    let total = candidates.len();
    println!(
        "ATLAS_SETUP schema=1 corpus={} games={} backtrack={} shallow={} candidates={} node_ladder={:?} tt_bytes={} relative_horizon={} wall_seconds={}",
        corpus_path.as_deref().unwrap_or("NONE"),
        game_count,
        backtrack,
        shallow_count,
        total,
        DEFAULT_NODE_LADDER,
        tt_bytes,
        relative_horizon,
        wall_seconds,
    );

    let batch_start = Instant::now();
    let mut attempted = 0usize;
    for candidate in &candidates {
        if attempted > 0 && batch_start.elapsed().as_secs() >= wall_seconds {
            break;
        }
        solve_candidate(candidate, tt_bytes, relative_horizon);
        attempted += 1;
    }
    println!(
        "ATLAS_DONE attempted={} residual={} wall_ms={:.3}",
        attempted,
        total - attempted,
        batch_start.elapsed().as_secs_f64() * 1e3,
    );
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
