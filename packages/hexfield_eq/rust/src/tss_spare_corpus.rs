//! Group-2 spare-turn corpus acceptance and regeneration helpers.
//!
//! The permanent corpus lives in `rust/corpus/spare_corpus_moves.txt`.  The
//! ignored mining helper is retained so every frozen position can be audited
//! against the independent exhaustive reference solver without a separate
//! binary or engine-only instrumentation.

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

use crate::tss_core::{lambda1_status, DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::{certificate_horizon_preflight, d6_transform_coord, CertNode};

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
            let coord = d6_transform_coord(HexCoord::new(q, r), symmetry)
                .expect("valid D6 transform");
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
