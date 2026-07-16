//! Group-2 spare-turn corpus acceptance and regeneration helpers.
//!
//! The permanent corpus lives in `rust/corpus/spare_corpus_moves.txt`.  The
//! ignored mining helper is retained so every frozen position can be audited
//! against the independent exhaustive reference solver without a separate
//! binary or engine-only instrumentation.

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{
    lambda1_status, CertVerify, DeepSolve, ProofStatus, SolveCaps, ZoneSearchCaps,
};
use crate::tss_reference_fast::{FastOrderingHint, FastReferenceConfig, FastReferenceResult};
use crate::tss_solver::{compact_certificate, round3_shadow_certificate, TssSolver, WidthOptions};
use crate::tss_verify::{
    certificate_horizon_preflight, d6_transform_coord, round3_rederived_zones, CertNode,
    TssVerifier,
};

const DEEP_WIN_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (0, 8),
    (2, 7),
    (1, 0),
    (2, 0),
    (4, 6),
    (6, 5),
    (0, 4),
    (1, 4),
    (8, 4),
    (10, 3),
    (2, 4),
    (16, 0),
    (12, 2),
    (14, 1),
];

const DEEP_UNIVERSAL_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (-1, 0),
    (0, -1),
    (-2, -3),
    (-1, -3),
    (-2, 1),
    (-3, 1),
    (0, -3),
    (1, -3),
    (-4, 2),
    (2, -4),
    (1, 4),
    (2, 4),
    (-5, 2),
    (2, -5),
    (3, 4),
    (4, 1),
    (-6, 3),
    (3, -6),
    (4, 2),
    (4, 3),
    (-7, 3),
    (3, -7),
    (1, 7),
    (2, 6),
    (-1, 2),
    (2, -1),
    (3, 5),
];

const COMPACT_URGENT_SPARE_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (-1, 3),
    (-3, 3),
    (-1, 0),
    (1, 0),
    (3, -2),
    (3, -1),
    (2, 0),
    (1, 4),
    (-2, 2),
    (-4, 4),
    (2, 4),
    (3, 4),
    (3, 1),
    (3, 2),
    (4, 1),
    (4, 2),
    (3, 3),
    (-2, 0),
    (4, 3),
    (1, 7),
    (-4, 1),
    (-3, -1),
    (2, 6),
    (-5, 5),
    (-4, -2),
    (0, -4),
    (3, 5),
];

const UNCAPPED_JUNCTION_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (4, -1),
    (-1, 5),
    (3, 3),
    (4, 3),
    (6, 0),
    (1, -7),
    (5, 3),
    (-3, 3),
    (-1, -6),
    (-2, -1),
    (-4, 3),
    (-5, 3),
    (-3, 6),
    (-7, 2),
    (0, 6),
    (0, 7),
    (-6, 0),
    (-5, 1),
    (0, 8),
    (0, -1),
    (7, -5),
    (3, 2),
    (0, -2),
    (1, -3),
    (5, -4),
    (6, -6),
    (1, 0),
    (-4, 0),
    (-6, 7),
    (2, -3),
    (1, 1),
];

const HUMAN_6A5A_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (0, -1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-2, 1),
    (0, 1),
    (-1, -1),
    (-2, 0),
    (-1, 2),
    (-3, 1),
];

const HUMAN_2A94_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (0, -1),
    (1, -1),
    (2, 0),
    (4, -2),
    (2, -1),
    (1, 0),
    (2, -2),
    (4, -3),
    (1, -2),
    (3, -2),
    (4, -1),
    (1, -3),
];

const HUMAN_FEAA_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (-2, 2),
    (0, 2),
    (-4, 2),
    (-2, 0),
    (-3, 1),
    (-3, 2),
    (-3, 3),
    (-3, 0),
    (-1, 0),
    (-2, 1),
    (-4, 3),
    (1, -2),
];

const HUMAN_5801_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (1, 0),
    (0, 2),
    (-2, 2),
    (2, -2),
    (1, -1),
    (1, 1),
    (-2, 4),
    (1, 2),
    (3, -1),
    (-2, 3),
    (1, -2),
    (4, -2),
];

const SPARE_TEMPO_PREFIX_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (1, 4),
    (2, 4),
    (-2, -3),
    (-1, -3),
    (3, 4),
    (4, 1),
    (0, -3),
    (2, -3),
    (4, 2),
    (4, 3),
    (3, -3),
    (-3, 1),
    (1, 7),
    (2, 6),
    (1, -2),
    (-4, 2),
    (3, 5),
    (-1, 0),
    (2, -4),
    (-5, 2),
    (0, -1),
    (-2, 1),
];

const DOUBLE_FORK_SPARE_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (-1, 0),
    (4, -5),
    (1, 0),
    (2, 0),
    (4, -4),
    (4, -3),
    (3, 0),
    (4, -6),
    (4, -2),
    (4, -1),
    (1, 3),
    (2, 3),
    (2, -5),
    (-1, 1),
    (3, 3),
    (0, 4),
    (7, 1),
    (8, -3),
    (0, 5),
    (0, 6),
    (4, 4),
    (5, 1),
    (5, 7),
    (6, 7),
    (5, 4),
    (2, 9),
    (7, 7),
    (4, 8),
    (8, 10),
    (7, 8),
    (4, 9),
    (4, 10),
    (0, -2),
    (3, 8),
    (2, -3),
];

const DOUBLE_FORK_COMPACT_MOVES: &[(i16, i16)] = &[
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

const DOUBLE_FORK_DENSE_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (4, 0),
    (-1, 1),
    (1, 0),
    (2, 0),
    (-1, 2),
    (-1, 3),
    (3, 0),
    (-1, 6),
    (-1, 4),
    (-1, 5),
    (3, 2),
    (4, 2),
    (0, 6),
    (0, 5),
    (5, 2),
    (2, 3),
    (1, 3),
    (3, 6),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (4, 5),
    (3, 5),
    (4, 4),
    (4, 1),
    (5, 4),
    (5, 3),
    (2, 1),
    (1, 1),
    (3, 1),
];

const DOUBLE_FORK_ORDERED_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (4, 0),
    (-1, 1),
    (1, 0),
    (2, 0),
    (-1, 2),
    (-1, 3),
    (3, 0),
    (-1, 6),
    (-1, 4),
    (-1, 5),
    (3, 2),
    (4, 2),
    (8, 0),
    (8, 4),
    (5, 2),
    (2, 3),
    (8, 6),
    (4, 3),
    (2, 4),
    (2, 5),
    (0, 2),
    (0, 5),
    (6, 5),
    (7, 5),
    (4, 1),
    (7, 2),
    (8, 5),
    (6, 4),
    (2, 1),
    (3, 6),
    (7, 3),
    (8, 2),
    (1, 6),
    (8, 1),
    (3, 3),
];

const SHARED_TARGET_SPARE_MOVES: &[(i16, i16)] = &[
    (0, 0),
    (5, -4),
    (-2, 2),
    (1, 0),
    (2, -1),
    (-1, 0),
    (-1, -4),
    (3, -2),
    (4, -3),
    (0, 4),
    (-3, -2),
    (-2, 1),
    (-1, 1),
    (3, 0),
    (-3, -3),
    (-3, 2),
    (-3, 3),
    (-2, 3),
    (-2, 0),
    (-3, 4),
    (0, -1),
    (5, -3),
    (3, -3),
    (1, -2),
    (2, -2),
    (-2, -1),
    (1, 4),
    (-2, -3),
];

fn replay(history: &[(i16, i16)]) -> HexoState {
    let mut state = HexoState::new();
    for &(q, r) in history {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(q, r),
            },
        )
        .unwrap_or_else(|error| panic!("illegal replay at ({q},{r}): {error:?}"));
    }
    assert!(!state.is_terminal());
    state
}

fn replay_d6(history: &[(i16, i16)], symmetry: u8) -> HexoState {
    let transformed = history
        .iter()
        .map(|&(q, r)| {
            let coord =
                d6_transform_coord(HexCoord::new(q, r), symmetry).expect("valid D6 transform");
            (coord.q, coord.r)
        })
        .collect::<Vec<_>>();
    replay(&transformed)
}

fn double_fork_dense_accelerated() -> HexoState {
    let (&first, prefix) = DOUBLE_FORK_DENSE_MOVES
        .split_last()
        .expect("dense fixture has a saved first stone");
    let mut moves = prefix.to_vec();
    // Two complete cycles preload one endpoint on each axis of both latent
    // fork centers.  Once either center survives the defender's spare, it
    // exposes four one-cell completions; the defender can cover only two.
    // The interleaved P1 placements avoid the urgent row and all four fork
    // axes.  Replaying the saved `(3,1)` then restores the original P0
    // SecondStone root and its mandatory `(-1,0)` completion.
    moves.extend_from_slice(&[
        (6, 2),
        (2, 6),
        (0, 3),
        (1, 4),
        (1, 5),
        (5, 1),
        (4, 6),
        (6, 4),
        first,
    ]);
    replay(&moves)
}

fn replaced(history: &[(i16, i16)], index: usize, coord: (i16, i16)) -> HexoState {
    let mut moves = history.to_vec();
    moves[index] = coord;
    replay(&moves)
}

fn deep_triple_block() -> HexoState {
    let mut moves = DEEP_UNIVERSAL_MOVES.to_vec();
    // Break the three remote count-three families that otherwise offer direct
    // k>B shortcuts, leaving the known horizontal k<B continuation intact.
    moves[17] = (0, 4);
    moves[21] = (0, 8);
    moves[25] = (4, 0);
    replay(&moves)
}

fn deep_quad_block() -> HexoState {
    let mut moves = DEEP_UNIVERSAL_MOVES.to_vec();
    // Adjacent-end blockers kill every six-window containing each of the
    // three direct contiguous count-three families. `(4,4)` is shared by all
    // three, so four defender fillers suffice while the r=-3 branch survives.
    moves[17] = (0, 4);
    moves[21] = (4, 4);
    moves[22] = (4, 0);
    moves[25] = (0, 8);
    replay(&moves)
}

fn deep_urgent_spare() -> HexoState {
    let mut moves = DEEP_UNIVERSAL_MOVES.to_vec();
    // P1's gapped q=2 five forces P0's remaining stone to `(2,-3)`. The
    // blocker at `(-3,-3)` makes P0's resulting count-five family one-hit,
    // so the reply node is the desired k=1<B=2 boundary. The three remote
    // P0 count-three families remain available after P1's spare placement.
    moves[17] = (-3, -3);
    moves[21] = (2, -2);
    moves[25] = (2, 0);
    replay(&moves)
}

fn urgent_uncapped_junction() -> HexoState {
    let mut moves = UNCAPPED_JUNCTION_MOVES.to_vec();
    // P1's horizontal gapped five makes `(1,2)` mandatory, removing the
    // otherwise-direct `(0,3)` junction shortcut. The mandatory block also
    // completes P0's single-window pin and exposes the k=1 spare boundary.
    moves[1] = (-2, 2);
    moves[2] = (-1, 2);
    moves[5] = (0, 2);
    moves[6] = (2, 2);
    replay(&moves)
}

fn human_6a5a_spare_edge() -> HexoState {
    let mut moves = HUMAN_6A5A_MOVES.to_vec();
    moves[10] = (-4, 0);
    moves.push((-1, -2));
    replay(&moves)
}

fn shared_target_block_endpoints() -> HexoState {
    let mut moves = SHARED_TARGET_SPARE_MOVES.to_vec();
    moves[25] = (-3, 5);
    moves[26] = (4, -2);
    replay(&moves)
}

fn deep_pruned_latents() -> HexoState {
    let mut moves = DEEP_UNIVERSAL_MOVES.to_vec();
    // Replace one claimant stone from each direct count-three shortcut with
    // an isolated, legal radius-eight filler. The horizontal count-four at
    // r=-3 and its known k<B continuation remain unchanged.
    moves[11] = (-8, 8);
    moves[16] = (8, 0);
    moves[23] = (8, -8);
    replay(&moves)
}

fn forcing_prefix(source_id: &str, nstones: usize) -> HexoState {
    let text = include_str!("../corpus/forcing_corpus_moves.txt");
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        if !header.starts_with("POS ") {
            continue;
        }
        let mut id = "";
        let mut count = 0usize;
        for field in header.split_whitespace().skip(1) {
            let (key, value) = field.split_once('=').expect("forcing k=v field");
            match key {
                "id" => id = value,
                "nstones" => count = value.parse().expect("numeric forcing nstones"),
                _ => {}
            }
        }
        let mut history = Vec::with_capacity(count);
        for _ in 0..count {
            let mut fields = lines.next().expect("forcing stone").split_whitespace();
            history.push((
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            ));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"));
        if id == source_id {
            assert!(nstones <= history.len(), "prefix exceeds source history");
            return replay(&history[..nstones]);
        }
    }
    panic!("unknown forcing source id: {source_id}")
}

fn mining_candidate(id: &str) -> HexoState {
    if let Some(encoded) = id.strip_prefix("forcing_prefix:") {
        let (source_id, nstones) = encoded
            .rsplit_once(':')
            .expect("forcing_prefix:<id>:<nstones>");
        return forcing_prefix(
            source_id,
            nstones.parse().expect("numeric forcing prefix length"),
        );
    }
    match id {
        "deep_win_seed" => replay(DEEP_WIN_MOVES),
        "deep_universal" => replay(DEEP_UNIVERSAL_MOVES),
        // Block the fixture's shorter direct lambda-one root move `(4,4)` by
        // replacing one Player1 filler. The three replacements test whether
        // the known full-universal `(2,-3)` branch survives independently of
        // the chosen remote provenance stone.
        "deep_block18" => replaced(DEEP_UNIVERSAL_MOVES, 17, (4, 4)),
        "deep_block22" => replaced(DEEP_UNIVERSAL_MOVES, 21, (4, 4)),
        "deep_block26" => replaced(DEEP_UNIVERSAL_MOVES, 25, (4, 4)),
        "deep_triple_block" => deep_triple_block(),
        "deep_quad_block" => deep_quad_block(),
        "deep_urgent_spare" => deep_urgent_spare(),
        "compact_urgent_spare" => replay(COMPACT_URGENT_SPARE_MOVES),
        "uncapped_junction" => replay(UNCAPPED_JUNCTION_MOVES),
        "urgent_uncapped_junction" => urgent_uncapped_junction(),
        "deep_pruned_latents" => deep_pruned_latents(),
        "human_6a5a" => replay(HUMAN_6A5A_MOVES),
        "human_6a5a_block_q" => replaced(HUMAN_6A5A_MOVES, 10, (-4, 0)),
        "human_6a5a_spare_edge" => human_6a5a_spare_edge(),
        "human_2a94" => replay(HUMAN_2A94_MOVES),
        "human_feaa" => replay(HUMAN_FEAA_MOVES),
        "human_5801" => replay(HUMAN_5801_MOVES),
        "spare_tempo_prefix" => replay(SPARE_TEMPO_PREFIX_MOVES),
        "double_fork_spare" => replay(DOUBLE_FORK_SPARE_MOVES),
        "double_fork_compact" => replay(DOUBLE_FORK_COMPACT_MOVES),
        "double_fork_compact_rot1" => replay_d6(DOUBLE_FORK_COMPACT_MOVES, 1),
        "double_fork_compact_reflect" => replay_d6(DOUBLE_FORK_COMPACT_MOVES, 6),
        "double_fork_dense" => replay(DOUBLE_FORK_DENSE_MOVES),
        "double_fork_dense_accelerated" => double_fork_dense_accelerated(),
        "double_fork_ordered" => replay(DOUBLE_FORK_ORDERED_MOVES),
        "shared_target_spare" => replay(SHARED_TARGET_SPARE_MOVES),
        "shared_target_block4" => replaced(SHARED_TARGET_SPARE_MOVES, 26, (4, -2)),
        "shared_target_block_endpoints" => shared_target_block_endpoints(),
        _ => panic!("unknown TSS_SPARE_MINE_ID: {id}"),
    }
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn fast_reference_config() -> FastReferenceConfig {
    let tt_bytes_cap = std::env::var("TSS_REFERENCE_FAST_TT_BYTES")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("numeric TSS_REFERENCE_FAST_TT_BYTES")
        })
        .unwrap_or(512 << 20);
    FastReferenceConfig {
        tt_bytes_cap,
        d6_canonical: std::env::var_os("TSS_REFERENCE_FAST_D6").is_some(),
        ordering_hint: FastOrderingHint::None,
    }
}

fn print_fast_result(id: &str, plies: u32, result: &FastReferenceResult) {
    println!(
        "SPARE_MINE id={id} profile=reference_fast status={} nodes={} tt_hits={} tt_entries={} tt_bytes={} tt_clears={} wall_ms={} plies={plies}",
        status_name(result.status),
        result.nodes,
        result.tt_hits,
        result.tt_entries,
        result.tt_accounted_bytes,
        result.tt_clears,
        result.elapsed.as_millis(),
    );
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

fn random_legal_state(seed: u64, placements: usize) -> HexoState {
    let mut rng = XorShift64(seed ^ 0xD1B5_4A32_D192_ED03);
    let mut state = HexoState::new();
    for _ in 0..placements {
        let legal = crate::tss_reference::legal_moves(&state);
        if legal.is_empty() {
            break;
        }
        let coord = legal[(rng.next() as usize) % legal.len()];
        let placed = apply_placement(&mut state, Placement { coord }).unwrap();
        if placed.outcome.is_some() {
            break;
        }
    }
    state
}

fn randomized_four_line(seed: u64, claimant: hexo_engine::Player) -> HexoState {
    use hexo_engine::Player;

    let mut rng = XorShift64(seed ^ 0x94D0_49BB_1331_11EB);
    let target_len = match claimant {
        Player::Player0 => 11,
        Player::Player1 => 9,
    };
    let mut state = HexoState::new();
    let mut claimant_stones = 0usize;
    while state.placements_made() < target_len {
        let mover = state.current_player();
        let forced = if mover == claimant {
            let coord = match (claimant, claimant_stones) {
                (Player::Player0, 0) => Some(HexCoord::ZERO),
                (Player::Player0, 1..=3) => Some(HexCoord::new(claimant_stones as i16, 0)),
                (Player::Player1, 0..=3) => Some(HexCoord::new(-(claimant_stones as i16) - 1, 0)),
                _ => None,
            };
            claimant_stones += 1;
            coord
        } else {
            None
        };

        let coord = forced.unwrap_or_else(|| {
            if state.phase() == hexo_engine::TurnPhase::Opening {
                return HexCoord::ZERO;
            }
            let candidates = crate::tss_reference::legal_moves(&state)
                .into_iter()
                .filter(|coord| coord.r != 0 || coord.q < -8 || coord.q > 8)
                .collect::<Vec<_>>();
            candidates[(rng.next() as usize) % candidates.len()]
        });
        let placed = apply_placement(&mut state, Placement { coord }).unwrap();
        assert!(placed.outcome.is_none(), "random tactical setup won early");
    }
    assert_eq!(state.current_player(), claimant);
    assert_eq!(state.phase(), hexo_engine::TurnPhase::FirstStone);
    state
}

fn assert_fast_matches_stock(
    id: &str,
    state: &HexoState,
    plies: u32,
    counts: &mut [usize; 3],
    movers: &mut [usize; 2],
) {
    let stock = crate::tss_reference::solve(state, plies);
    let fast = crate::tss_reference_fast::solve(
        state,
        plies,
        FastReferenceConfig {
            tt_bytes_cap: 64 << 20,
            d6_canonical: true,
            ordering_hint: FastOrderingHint::None,
        },
    );
    assert_eq!(
        fast.status, stock.status,
        "REFERENCE_DISAGREEMENT id={id} plies={plies} stock={stock:?} fast={fast:?}"
    );
    counts[match stock.status {
        ProofStatus::Win => 0,
        ProofStatus::Loss => 1,
        ProofStatus::Unknown => 2,
    }] += 1;
    movers[state.current_player().index()] += 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpareExpectation {
    WinPending,
    No,
}

struct SpareCorpusPosition {
    id: String,
    expect: SpareExpectation,
    reference_plies: u32,
    oracle: ProofStatus,
    state: HexoState,
}

fn load_spare_corpus() -> Vec<SpareCorpusPosition> {
    let text = include_str!("../corpus/spare_corpus_moves.txt");
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() || header.starts_with('#') {
            continue;
        }
        assert!(header.starts_with("POS "), "bad spare header: {header}");
        let mut id = String::new();
        let mut expect = None;
        let mut oracle = None;
        let mut reference_plies = 0u32;
        let mut attacker = 0usize;
        let mut nstones = 0usize;
        for field in header.split_whitespace().skip(1) {
            let (key, value) = field.split_once('=').expect("spare k=v field");
            match key {
                "id" => id = value.to_string(),
                "expect" => {
                    expect = Some(match value {
                        "WIN_PENDING" => SpareExpectation::WinPending,
                        "NO" => SpareExpectation::No,
                        _ => panic!("unknown spare expectation: {value}"),
                    })
                }
                "oracle" => {
                    oracle = Some(match value {
                        "WIN" => ProofStatus::Win,
                        "LOSS" => ProofStatus::Loss,
                        "UNKNOWN" => ProofStatus::Unknown,
                        _ => panic!("unknown spare oracle status: {value}"),
                    })
                }
                "reference_plies" => reference_plies = value.parse().unwrap(),
                "attacker" => attacker = value.parse().unwrap(),
                "nstones" => nstones = value.parse().unwrap(),
                _ => panic!("unknown spare header field: {key}"),
            }
        }
        let mut state = HexoState::new();
        for _ in 0..nstones {
            let mut fields = lines.next().expect("spare stone").split_whitespace();
            let q = fields.next().unwrap().parse().unwrap();
            let r = fields.next().unwrap().parse().unwrap();
            assert!(fields.next().is_none(), "extra spare stone field");
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap_or_else(|error| panic!("{id}: illegal replay at ({q},{r}): {error:?}"));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"));
        assert!(!state.is_terminal());
        assert_eq!(state.current_player().index(), attacker);
        out.push(SpareCorpusPosition {
            id,
            expect: expect.expect("spare expectation"),
            reference_plies,
            oracle: oracle.expect("spare oracle status"),
            state,
        });
    }
    out
}

#[test]
#[ignore = "Group-2 spare-turn acceptance gate"]
fn tss_spare_corpus_check() {
    let cap = std::env::var("TSS_SPARE_CORPUS_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric TSS_SPARE_CORPUS_CAP"))
        .unwrap_or(1_000_000);
    let tt_bytes_cap = std::env::var("TSS_BACKWALK_TT_BYTES")
        .ok()
        .map(|value| value.parse::<usize>().expect("numeric TT bytes"))
        .unwrap_or(512 << 20);
    let selected = std::env::var("TSS_SPARE_CORPUS_ID").ok();
    let mut ran = 0usize;
    for position in load_spare_corpus() {
        if selected
            .as_ref()
            .is_some_and(|wanted| !wanted.split(',').any(|id| id == position.id))
        {
            continue;
        }
        let started = std::time::Instant::now();
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let result = solver.solve(
            &position.state,
            &SolveCaps {
                node_cap: cap,
                tt_bytes_cap,
                semantic_horizon: position.state.placements_made() + position.reference_plies,
            },
        );
        println!(
            "SPARE_CORPUS id={} expect={:?} oracle={} horizon={} cap={} status={} nodes={} legal={} wall_ms={}",
            position.id,
            position.expect,
            status_name(position.oracle),
            position.state.placements_made() + position.reference_plies,
            cap,
            status_name(result.status),
            result.stats.nodes,
            position.state.legal_move_count(),
            started.elapsed().as_millis(),
        );
        match position.expect {
            SpareExpectation::WinPending => assert_ne!(
                result.status,
                ProofStatus::Loss,
                "{}: certified LOSS on oracle WIN row",
                position.id
            ),
            SpareExpectation::No => assert_ne!(
                result.status,
                ProofStatus::Win,
                "{}: false WIN on stock-reference NO control",
                position.id
            ),
        }
        ran += 1;
    }
    assert!(ran > 0, "spare corpus selection matched no rows");
}

/// Mandatory gate before `tss_reference_fast` may supply corpus ground truth.
#[test]
#[ignore = "serialized exact-reference differential gate"]
fn tss_reference_fast_differential() {
    let mut counts = [0usize; 3];
    let mut movers = [0usize; 2];

    let round1 = [
        (
            "compact_urgent_spare",
            mining_candidate("compact_urgent_spare"),
            2,
        ),
        (
            "strongloss_a_backoff_7",
            forcing_prefix("strongloss_a_prefix6", 7),
            2,
        ),
        (
            "spare_tempo_prefix",
            mining_candidate("spare_tempo_prefix"),
            2,
        ),
    ];
    for (id, state, plies) in &round1 {
        assert_fast_matches_stock(id, state, *plies, &mut counts, &mut movers);
    }

    // Horizon semantics: identical positions at two distinct budgets.
    for (id, state, _) in &round1 {
        for plies in [0, 1] {
            assert_fast_matches_stock(
                &format!("{id}_h{plies}"),
                state,
                plies,
                &mut counts,
                &mut movers,
            );
        }
    }

    // 120 ordinary fixed-seed playout positions, distributed over both
    // player identities and depth-zero/depth-one UNKNOWN leaves.
    for seed in 1..=120u64 {
        let placements = 1 + (seed as usize % 12);
        let state = random_legal_state(seed, placements);
        let plies = (seed % 2) as u32;
        assert_fast_matches_stock(
            &format!("random_{seed:03}"),
            &state,
            plies,
            &mut counts,
            &mut movers,
        );
    }

    // 80 randomized tactical positions have a four-line for the mover. At
    // depth one the result is UNKNOWN; at depth two it is WIN. Both players
    // occur at both horizons, and filler choices are fixed by the seed.
    for seed in 1..=40u64 {
        for claimant in [hexo_engine::Player::Player0, hexo_engine::Player::Player1] {
            let state = randomized_four_line(seed, claimant);
            let plies = 1 + (seed % 2) as u32;
            assert_fast_matches_stock(
                &format!("tactical_{claimant:?}_{seed:02}"),
                &state,
                plies,
                &mut counts,
                &mut movers,
            );
        }
    }

    let total = counts.into_iter().sum::<usize>();
    println!(
        "REFERENCE_FAST_DIFFERENTIAL cases={total} win={} loss={} unknown={} mover_p0={} mover_p1={}",
        counts[0], counts[1], counts[2], movers[0], movers[1]
    );
    assert!(total >= 200);
    assert!(counts[0] > 0 && counts[2] > 0);
    assert!(movers.into_iter().all(|count| count > 0));
}

/// Exact range decomposition of the compact witness's first Universal node.
#[test]
#[ignore = "serialized compact exact-oracle branch batch"]
fn tss_reference_fast_compact_branch_batch() {
    let start = std::env::var("TSS_SPARE_BRANCH_START")
        .expect("set TSS_SPARE_BRANCH_START")
        .parse::<usize>()
        .expect("numeric branch start");
    let end = std::env::var("TSS_SPARE_BRANCH_END")
        .expect("set TSS_SPARE_BRANCH_END")
        .parse::<usize>()
        .expect("numeric branch end");
    let mut root = mining_candidate("double_fork_compact");
    let claimant = root.current_player();
    let (placed, root_delta) = root
        .apply_with_delta(Placement {
            coord: HexCoord::new(4, 0),
        })
        .expect("compact exact root move is legal");
    assert!(placed.outcome.is_none());
    let legal = crate::tss_reference_fast::full_legal_moves(&root);
    assert_eq!(
        legal.len(),
        478,
        "frozen compact post-root defender frontier"
    );
    assert!(start < end && end <= legal.len());

    let mut nodes = 0u64;
    let mut hits = 0u64;
    let mut wall_ms = 0u128;
    for (index, &coord) in legal[start..end].iter().enumerate() {
        let absolute_index = start + index;
        let (reply, delta) = root
            .apply_with_delta(Placement { coord })
            .expect("independent compact defender move is legal");
        let result = if reply.outcome.is_some() {
            FastReferenceResult {
                status: ProofStatus::Loss,
                nodes: 1,
                tt_hits: 0,
                tt_entries: 0,
                tt_accounted_bytes: 0,
                tt_clears: 0,
                elapsed: std::time::Duration::ZERO,
            }
        } else {
            crate::tss_reference_fast::solve_for_player(
                &root,
                claimant,
                7,
                FastReferenceConfig {
                    tt_bytes_cap: 256 << 20,
                    d6_canonical: false,
                    ordering_hint: FastOrderingHint::DoubleForkCompact,
                },
            )
        };
        root.undo(delta);
        println!(
            "SPARE_COMPACT_BRANCH index={absolute_index} move=({}, {}) status={} nodes={} tt_hits={} wall_ms={}",
            coord.q,
            coord.r,
            status_name(result.status),
            result.nodes,
            result.tt_hits,
            result.elapsed.as_millis(),
        );
        assert_eq!(
            result.status,
            ProofStatus::Win,
            "compact exact branch refutes candidate at index={absolute_index} move={coord:?}"
        );
        nodes = nodes.saturating_add(result.nodes);
        hits = hits.saturating_add(result.tt_hits);
        wall_ms = wall_ms.saturating_add(result.elapsed.as_millis());
    }
    root.undo(root_delta);
    println!(
        "SPARE_COMPACT_BRANCH_BATCH start={start} end={end} status=WIN nodes={nodes} tt_hits={hits} summed_wall_ms={wall_ms}"
    );
}

/// Candidate triage. This deliberately runs one selected solve at a time.
/// It is a regeneration aid, not part of the acceptance gate.
#[test]
#[ignore = "serialized Group-2 corpus mining helper"]
fn tss_spare_mine_candidate() {
    let id = std::env::var("TSS_SPARE_MINE_ID").expect("set TSS_SPARE_MINE_ID");
    let state = mining_candidate(&id);
    let cap = std::env::var("TSS_SPARE_MINE_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric TSS_SPARE_MINE_CAP"))
        .unwrap_or(500_000);
    let tt_bytes_cap = std::env::var("TSS_BACKWALK_TT_BYTES")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("numeric TSS_BACKWALK_TT_BYTES")
        })
        .unwrap_or(512 << 20);
    let caps = SolveCaps {
        node_cap: cap,
        tt_bytes_cap,
        semantic_horizon: u32::MAX,
    };

    if std::env::var_os("TSS_SPARE_LIST_WINDOWS").is_some() {
        let claimant = state.current_player();
        for entry in state.board().windows().entries() {
            if entry.active_player() == Some(claimant) && entry.count(claimant) >= 3 {
                println!(
                    "SPARE_ROOT_WINDOW id={id} count={} key={:?} empties={:?}",
                    entry.count(claimant),
                    entry.key(),
                    entry.empty_cells(),
                );
            }
        }
    }

    if let Ok(encoded) = std::env::var("TSS_SPARE_PROBE_ROOT") {
        let (q, r) = encoded.split_once(',').expect("TSS_SPARE_PROBE_ROOT=q,r");
        let coord = HexCoord::new(
            q.parse().expect("numeric probe q"),
            r.parse().expect("numeric probe r"),
        );
        let mut child = state.clone();
        let placed = apply_placement(&mut child, Placement { coord }).expect("legal root probe");
        let analysis = (!child.is_terminal()).then(|| crate::threats_shared::analyze(&child));
        println!(
            "SPARE_ROOT_PROBE id={id} move=({}, {}) outcome={:?} phase={:?} b={:?} k={:?} threats={:?} own_win_now={:?} lambda1={}",
            coord.q,
            coord.r,
            placed.outcome,
            child.phase(),
            analysis.as_ref().map(|value| value.b),
            analysis.as_ref().and_then(|value| value.min_hitting_set),
            analysis.as_ref().map(|value| value.opp_threat_count),
            analysis.as_ref().map(|value| value.own_win_now),
            status_name(lambda1_status(&child)),
        );
    }

    if std::env::var_os("TSS_SPARE_LIST_ROOT_L1").is_some() {
        let mut work = state.clone();
        let root_player = work.current_player();
        let legal = crate::tss_reference::legal_moves(&work);
        for coord in legal {
            let (placed, delta) = work
                .apply_with_delta(Placement { coord })
                .expect("reference legal root move");
            let analysis = (!work.is_terminal()).then(|| crate::threats_shared::analyze(&work));
            let root_wins = placed
                .outcome
                .is_some_and(|outcome| outcome.winner == root_player)
                || lambda1_status(&work) == ProofStatus::Loss;
            if root_wins {
                println!(
                    "SPARE_ROOT_L1 id={id} move=({}, {}) terminal={} child_b={:?} child_k={:?} child_threats={:?} child_own_win_now={:?}",
                    coord.q,
                    coord.r,
                    work.is_terminal(),
                    analysis.as_ref().map(|value| value.b),
                    analysis.as_ref().and_then(|value| value.min_hitting_set),
                    analysis.as_ref().map(|value| value.opp_threat_count),
                    analysis.as_ref().map(|value| value.own_win_now),
                );
            }
            work.undo(delta);
        }
    }

    if let Ok(value) = std::env::var("TSS_SPARE_REFERENCE_PLIES") {
        let plies = value
            .parse::<u32>()
            .expect("numeric TSS_SPARE_REFERENCE_PLIES");
        let reference = crate::tss_reference::solve(&state, plies);
        println!(
            "SPARE_MINE id={id} profile=reference status={} nodes={} plies={plies}",
            status_name(reference.status),
            reference.nodes,
        );
        return;
    }

    if let Ok(value) = std::env::var("TSS_SPARE_FAST_PLIES") {
        let plies = value.parse::<u32>().expect("numeric TSS_SPARE_FAST_PLIES");
        let mut config = fast_reference_config();
        if id.starts_with("double_fork_compact") {
            config.ordering_hint = FastOrderingHint::DoubleForkCompact;
        }
        let result = crate::tss_reference_fast::solve(&state, plies, config);
        print_fast_result(&id, plies, &result);
        return;
    }

    let mut solver = TssSolver::default();
    let result = solver.solve(&state, &caps);
    let exact_t = result
        .cert
        .as_ref()
        .and_then(certificate_horizon_preflight)
        .map(|(horizon, _)| horizon);
    let root_node = result
        .cert
        .as_ref()
        .map(|cert| &cert.nodes[cert.root_node as usize]);
    println!(
        "SPARE_MINE id={id} profile=default status={} nodes={} root_ply={} horizon={exact_t:?} root_node={root_node:?}",
        status_name(result.status),
        result.stats.nodes,
        state.placements_made(),
    );
    if std::env::var_os("TSS_SPARE_SUMMARIZE_DEFAULT_CERT").is_some() {
        let mut choices = std::collections::BTreeMap::<(i16, i16), usize>::new();
        let mut universal_nodes = 0usize;
        let mut universal_edges = 0usize;
        if let Some(cert) = &result.cert {
            for node in &cert.nodes {
                match node {
                    CertNode::Choice { mv, .. } | CertNode::OrCompletion { mv, .. } => {
                        *choices.entry((mv.q, mv.r)).or_default() += 1;
                    }
                    CertNode::Universal { edges, .. } => {
                        universal_nodes += 1;
                        universal_edges += edges.len();
                    }
                    CertNode::Win { .. } | CertNode::Loss { .. } => {}
                }
            }
        }
        let mut choices = choices.into_iter().collect::<Vec<_>>();
        choices.sort_by_key(|(coord, count)| (std::cmp::Reverse(*count), *coord));
        println!(
            "SPARE_DEFAULT_CERT_SUMMARY id={id} universal_nodes={universal_nodes} universal_edges={universal_edges} top_choices={:?}",
            &choices[..choices.len().min(32)]
        );
    }

    let mut wide = TssSolver::default();
    wide.set_width_options(WidthOptions::vcf_pair_complete());
    let wide_result = wide.solve(
        &state,
        &SolveCaps {
            semantic_horizon: exact_t.unwrap_or(u32::MAX),
            ..caps
        },
    );
    let wide_horizon = wide_result
        .cert
        .as_ref()
        .and_then(certificate_horizon_preflight)
        .map(|(horizon, _)| horizon);
    let wide_root = wide_result
        .cert
        .as_ref()
        .map(|cert| &cert.nodes[cert.root_node as usize]);
    println!(
        "SPARE_MINE id={id} profile=vcf_pair_complete status={} nodes={} horizon={wide_horizon:?} root_node={wide_root:?}",
        status_name(wide_result.status),
        wide_result.stats.nodes,
    );
    if std::env::var_os("TSS_SPARE_PRINT_CERT").is_some() {
        println!("SPARE_WIDE_CERT id={id} cert={:?}", wide_result.cert);
    }

    if std::env::var_os("TSS_SPARE_REFERENCE").is_some() {
        let exact_t = exact_t.expect("reference mining requires a hard default certificate");
        let reference = crate::tss_reference::solve(&state, exact_t - state.placements_made());
        println!(
            "SPARE_MINE id={id} profile=reference status={} nodes={} plies={}",
            status_name(reference.status),
            reference.nodes,
            exact_t - state.placements_made(),
        );
    }

    // Keep CertNode imported and its schema compiled into this helper while
    // mining output is still intentionally generic.
    let _ = std::mem::size_of::<CertNode>();
}

/// Step-1 SHADOW gate. The historical finder supplies the completed strategy
/// from which D10 live roles can be derived; the wide shadow profile itself is
/// required to remain byte-for-byte identical to ordinary wide mode.
#[test]
#[ignore = "round-3 shadow coverage and default-off identity"]
fn tss_round3_shadow_spare_coverage() {
    let state = mining_candidate("double_fork_compact");
    let caps = SolveCaps {
        node_cap: 100_000,
        tt_bytes_cap: 512 << 20,
        semantic_horizon: 45,
    };

    let mut historical = TssSolver::default();
    let historical_result = historical.solve(&state, &caps);
    assert_eq!(historical_result.status, ProofStatus::Win);
    let cert = historical_result
        .cert
        .as_ref()
        .expect("historical WIN cert");
    assert!(TssVerifier.verify(&state, cert, ProofStatus::Win));
    let report = round3_shadow_certificate(&state, cert).expect("shadow certificate replay");
    println!(
        "R3_SHADOW id=double_fork_compact source=historical status=WIN nodes={} quiet_fires={} quiet_legal_edges={} zone_nodes={}",
        historical_result.stats.nodes,
        report.quiet_turns,
        report.quiet_legal_edges,
        report.zones.len(),
    );
    for (index, zone) in report.zones.iter().enumerate() {
        println!(
            "R3_SHADOW_ZONE id=double_fork_compact index={index} ply={} b={} k={:?} B={} zone={} legal={} ratio={:.6} z_dir={} z_seed={} z_touch={} z_virgin={} represented_in_zone={} best_rank={:?} worst_rank={:?}",
            zone.ply,
            zone.b,
            zone.k,
            zone.local_budget,
            zone.zone.len(),
            zone.full_legal,
            zone.zone.len() as f64 / zone.full_legal as f64,
            zone.z_dir,
            zone.z_seed,
            zone.z_touch,
            zone.z_virgin,
            zone.represented_in_zone,
            zone.best_represented_rank,
            zone.worst_represented_rank,
        );
    }
    assert!(report.quiet_turns >= 1, "witness must contain a quiet turn");
    assert!(
        !report.zones.is_empty(),
        "witness must contain k<b AND nodes"
    );
    assert!(report
        .zones
        .iter()
        .any(|zone| zone.full_legal == 478 && zone.zone.len() < zone.full_legal));
    let verifier_zones = round3_rederived_zones(&state, cert).expect("independent zone replay");
    let finder_zones = report
        .zones
        .iter()
        .map(|zone| (zone.ply, zone.zone.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        verifier_zones, finder_zones,
        "finder/verifier shadow zones must match byte-for-byte"
    );

    let mut ordinary_wide = TssSolver::default();
    ordinary_wide.set_width_options(WidthOptions::vcf_pair_complete());
    let ordinary = ordinary_wide.solve(&state, &caps);
    let mut shadow_wide = TssSolver::default();
    shadow_wide.set_width_options(WidthOptions::round3_shadow());
    let shadow = shadow_wide.solve(&state, &caps);
    assert_eq!(shadow.status, ordinary.status);
    assert_eq!(shadow.stats.nodes, ordinary.stats.nodes);
    assert_eq!(shadow.cert, ordinary.cert);

    for control in load_spare_corpus() {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::round3_shadow());
        let control_caps = SolveCaps {
            node_cap: 10_000,
            tt_bytes_cap: 512 << 20,
            semantic_horizon: u32::MAX,
        };
        let result = solver.solve(&control.state, &control_caps);
        println!(
            "R3_SHADOW id={} source=spare_control status={} nodes={} quiet_fires=0 zone_nodes=0 cert={}",
            control.id,
            status_name(result.status),
            result.stats.nodes,
            result.cert.is_some(),
        );
        assert_ne!(result.status, ProofStatus::Win, "NO control became WIN");
    }
}

#[test]
#[ignore = "round-3 independent verifier and mandatory mutation controls"]
fn tss_round3_verifier_mutations() {
    let state = mining_candidate("double_fork_compact");
    let mut solver = TssSolver::default();
    solver.set_zone_options(ZoneSearchCaps {
        enabled: true,
        stale_area_filter: false,
        count2_threshold: true,
        pair_commutation: false,
    });
    let result = solver.solve(
        &state,
        &SolveCaps {
            node_cap: 100_000,
            tt_bytes_cap: 512 << 20,
            semantic_horizon: 45,
        },
    );
    assert_eq!(result.status, ProofStatus::Win);
    let cert = result.cert.expect("zoned finder certificate");
    println!(
        "R3_VERIFY_BASE status=WIN nodes={} cert_nodes={} verifier={}",
        result.stats.nodes,
        cert.nodes.len(),
        TssVerifier.verify(&state, &cert, ProofStatus::Win),
    );
    assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));

    let finder = round3_shadow_certificate(&state, &cert).expect("finder zone replay");
    let verifier = round3_rederived_zones(&state, &cert).expect("verifier zone replay");
    assert_eq!(
        finder
            .zones
            .iter()
            .map(|zone| (zone.ply, zone.zone.clone()))
            .collect::<Vec<_>>(),
        verifier,
        "zoned certificate finder/verifier sets differ"
    );

    let reject = |label: &str, mutated: &crate::tss_verify::TssCertificate| {
        let accepted = TssVerifier.verify(&state, mutated, ProofStatus::Win);
        println!("R3_VERIFY_MUTATION label={label} accepted={accepted}");
        assert!(!accepted, "mutation {label} was accepted");
    };
    let compact_from = |mutated: &mut crate::tss_verify::TssCertificate, root| {
        let (nodes, root_node) =
            compact_certificate(&mutated.nodes, root).expect("compact mutation");
        mutated.nodes = nodes;
        mutated.root_node = root_node;
    };

    let (root_move, zone_id) = match &cert.nodes[cert.root_node as usize] {
        CertNode::Choice { mv, child } => (*mv, *child),
        other => panic!("expected quiet root Choice, got {other:?}"),
    };
    let zone_edges = match &cert.nodes[zone_id as usize] {
        CertNode::Universal {
            edges,
            zone: Some(_),
            ..
        } => edges,
        other => panic!("expected post-quiet zone Universal, got {other:?}"),
    };
    assert!(zone_edges.len() > 2);

    let mut zone_state = state.clone();
    apply_placement(&mut zone_state, Placement { coord: root_move }).expect("quiet root replay");
    let zone_budget = match &cert.nodes[zone_id as usize] {
        CertNode::Universal {
            zone: Some(zone), ..
        } => zone.d,
        _ => unreachable!(),
    };
    let defender = cert.claimant.other();
    let mut touch = Vec::new();
    for entry in zone_state.board().windows().entries() {
        let count = entry.count(defender);
        if entry.active_player() == Some(defender)
            && count >= 1
            && u32::from(count).saturating_add(zone_budget) >= 6
        {
            touch.extend(entry.empty_cells());
        }
    }
    touch.sort_by_key(|coord| (coord.q, coord.r));
    touch.dedup();
    touch.retain(|cell| zone_edges.iter().any(|edge| edge.mv == *cell));
    assert!(
        touch.len() >= 2,
        "fixture must expose two stable Z_touch cells"
    );

    let mut omitted_zone_cell = cert.clone();
    if let CertNode::Universal { edges, .. } = &mut omitted_zone_cell.nodes[zone_id as usize] {
        let index = edges
            .iter()
            .position(|edge| edge.mv == touch[0])
            .expect("first touch edge");
        edges.remove(index);
    }
    let omitted_root = omitted_zone_cell.root_node;
    compact_from(&mut omitted_zone_cell, omitted_root);
    reject("omitted_zone_cell", &omitted_zone_cell);

    let mut omitted_defender_edge = cert.clone();
    if let CertNode::Universal { edges, .. } = &mut omitted_defender_edge.nodes[zone_id as usize] {
        let index = edges
            .iter()
            .position(|edge| edge.mv == touch[1])
            .expect("second touch edge");
        edges.remove(index);
    }
    let omitted_root = omitted_defender_edge.root_node;
    compact_from(&mut omitted_defender_edge, omitted_root);
    reject("omitted_defender_edge", &omitted_defender_edge);

    let mut dropped_quiet_edge = cert.clone();
    compact_from(&mut dropped_quiet_edge, zone_id);
    reject("dropped_quiet_edge", &dropped_quiet_edge);

    let mut wrong_budget = cert.clone();
    if let CertNode::Universal {
        zone: Some(zone), ..
    } = &mut wrong_budget.nodes[zone_id as usize]
    {
        zone.d = zone.d.saturating_add(1);
    }
    reject("wrong_budget", &wrong_budget);

    let mut wrong_horizon = cert.clone();
    wrong_horizon.semantic_horizon = wrong_horizon.semantic_horizon.saturating_add(1);
    reject("wrong_horizon", &wrong_horizon);

    let mut forged_leaf = cert.clone();
    let leaf = forged_leaf
        .nodes
        .iter_mut()
        .find(|node| {
            matches!(
                node,
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. }
            )
        })
        .expect("certificate leaf");
    match leaf {
        CertNode::OrCompletion { completion_ply, .. } => {
            *completion_ply = completion_ply.saturating_add(1)
        }
        CertNode::Win { count, .. } => *count = count.saturating_sub(1),
        CertNode::Loss { resolution_ply, .. } => *resolution_ply = resolution_ply.saturating_add(1),
        CertNode::Choice { .. } | CertNode::Universal { .. } => unreachable!(),
    }
    reject("forged_leaf", &forged_leaf);

    let represented = zone_edges.iter().map(|edge| edge.mv).collect::<Vec<_>>();
    let mut full_legal = Vec::new();
    zone_state.write_legal_moves(&mut full_legal);
    let outside = full_legal
        .into_iter()
        .find(|cell| !represented.contains(cell))
        .expect("zone must omit some legal cells");
    let mut out_of_zone = cert.clone();
    if let CertNode::Universal { edges, .. } = &mut out_of_zone.nodes[zone_id as usize] {
        let index = edges
            .iter()
            .position(|edge| edge.mv == touch[0])
            .expect("substituted touch edge");
        edges[index].mv = outside;
    }
    reject("out_of_zone_substitution", &out_of_zone);
}

#[test]
#[ignore = "round-3 consume witness ladder"]
fn tss_round3_consume_witness() {
    let cap = std::env::var("TSS_R3_CAP")
        .ok()
        .map(|value| value.parse::<u64>().expect("numeric TSS_R3_CAP"))
        .unwrap_or(10_000);
    let tt_bytes_cap = std::env::var("TSS_BACKWALK_TT_BYTES")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("numeric TSS_BACKWALK_TT_BYTES")
        })
        .unwrap_or(512 << 20);
    let state = mining_candidate("double_fork_compact");
    let mut solver = TssSolver::default();
    solver.set_width_options(WidthOptions::round3_consume());
    let started = std::time::Instant::now();
    let result = solver.solve(
        &state,
        &SolveCaps {
            node_cap: cap,
            tt_bytes_cap,
            semantic_horizon: 45,
        },
    );
    let verified = result
        .cert
        .as_ref()
        .is_some_and(|cert| TssVerifier.verify(&state, cert, result.status));
    println!(
        "R3_CONSUME id=double_fork_compact cap={cap} status={} nodes={} tt_hits={} peak_tt_bytes={} wall_ms={} verified={verified}",
        status_name(result.status),
        result.stats.nodes,
        result.stats.tt_hits,
        result.stats.peak_tt_bytes,
        started.elapsed().as_millis(),
    );
    assert_eq!(result.status, ProofStatus::Win);
    assert!(verified);
}
