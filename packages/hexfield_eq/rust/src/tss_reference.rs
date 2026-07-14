//! Independent, exhaustive reference search for the proof-carrying TSS.
//!
//! This module intentionally does not use the engine's incremental legal-move
//! store, six-cell window store, threat analysis, or either TSS move generator.
//! Legal placements are reconstructed from occupancy and the radius-eight
//! rule, and wins are found by reading board owners along the three line axes.
//! The only engine operation used while searching is make/unmake.  Keeping
//! these mechanisms independent makes differential tests useful against a
//! common-mode move-generation or terminal-detection defect.

use std::collections::BTreeSet;

use hexo_engine::{HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::ProofStatus;

const LEGAL_RADIUS: i32 = 8;
const WIN_LENGTH: i32 = 6;
const AXES: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];

/// Outcome of an exhaustive search through `ply_budget` single placements.
///
/// `status` is always from the identity of the player to move in the supplied
/// root state. `nodes` counts visited positions, including the root and
/// terminal/horizon leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReferenceResult {
    pub(crate) status: ProofStatus,
    pub(crate) nodes: u64,
}

/// Exhaustive, depth-limited, three-valued minimax over every legal placement.
///
/// A single clone creates the mutable search root; recursion itself uses only
/// `apply_with_delta`/`undo`.  At the horizon a non-winning position is
/// `Unknown`.  There is deliberately no threat pruning, transposition table,
/// incremental legal enumeration, or window-based evaluation here.
pub(crate) fn solve(state: &HexoState, ply_budget: u32) -> ReferenceResult {
    let root_player = state.current_player();
    let mut working = state.clone();
    let mut nodes = 0;
    let status = minimax(&mut working, root_player, ply_budget, &mut nodes);
    ReferenceResult { status, nodes }
}

/// Reconstruct the authoritative single-placement legal set independently.
///
/// For a non-opening state the rules admit every empty cell within hex distance
/// eight of at least one occupied cell. A `BTreeSet<(q, r)>` both deduplicates
/// overlapping neighborhoods and fixes lexicographic iteration order.
pub(crate) fn legal_moves(state: &HexoState) -> Vec<HexCoord> {
    if state.is_terminal() {
        return Vec::new();
    }

    if state.phase() == TurnPhase::Opening {
        return if state.board().is_cell_empty(HexCoord::ZERO) {
            vec![HexCoord::ZERO]
        } else {
            Vec::new()
        };
    }

    let mut ordered = BTreeSet::new();
    for &stone in state.board().occupied_cells() {
        // Axial coordinates (dq, dr) lie in a radius-R hex exactly when
        // max(|dq|, |dr|, |dq + dr|) <= R. These bounds enumerate that set
        // without calling the engine's coords_within_radius helper.
        for dq in -LEGAL_RADIUS..=LEGAL_RADIUS {
            let dr_min = (-LEGAL_RADIUS).max(-dq - LEGAL_RADIUS);
            let dr_max = LEGAL_RADIUS.min(-dq + LEGAL_RADIUS);
            for dr in dr_min..=dr_max {
                let q = i32::from(stone.q) + dq;
                let r = i32::from(stone.r) + dr;
                let (Ok(q), Ok(r)) = (i16::try_from(q), i16::try_from(r)) else {
                    continue;
                };
                let candidate = HexCoord { q, r };
                if state.board().is_cell_empty(candidate) {
                    ordered.insert((q, r));
                }
            }
        }
    }

    ordered
        .into_iter()
        .map(|(q, r)| HexCoord { q, r })
        .collect()
}

/// Scan owners directly for a contiguous run of at least six stones.
///
/// Reachable engine states cannot contain lines for both players: application
/// stops on the first winning placement. The stable Player0-first tie break is
/// therefore relevant only to malformed, externally manufactured states.
pub(crate) fn direct_winner(state: &HexoState) -> Option<Player> {
    [Player::Player0, Player::Player1]
        .into_iter()
        .find(|&player| has_six(state, player))
}

fn has_six(state: &HexoState, player: Player) -> bool {
    for &start in state.board().occupied_cells() {
        if state.board().get(start) != Some(player) {
            continue;
        }
        for (dq, dr) in AXES {
            let mut complete = true;
            for offset in 1..WIN_LENGTH {
                let Some(coord) = offset_coord(start, dq, dr, offset) else {
                    complete = false;
                    break;
                };
                if state.board().get(coord) != Some(player) {
                    complete = false;
                    break;
                }
            }
            if complete {
                return true;
            }
        }
    }
    false
}

fn offset_coord(start: HexCoord, dq: i32, dr: i32, offset: i32) -> Option<HexCoord> {
    let q = i32::from(start.q) + dq * offset;
    let r = i32::from(start.r) + dr * offset;
    Some(HexCoord {
        q: i16::try_from(q).ok()?,
        r: i16::try_from(r).ok()?,
    })
}

fn minimax(
    state: &mut HexoState,
    root_player: Player,
    plies_left: u32,
    nodes: &mut u64,
) -> ProofStatus {
    *nodes = nodes.saturating_add(1);

    // Evaluation ignores ApplyResult::outcome and the engine terminal winner;
    // the independent board scan is the sole source of a hard leaf value.
    if let Some(winner) = direct_winner(state) {
        return if winner == root_player {
            ProofStatus::Win
        } else {
            ProofStatus::Loss
        };
    }
    if plies_left == 0 {
        return ProofStatus::Unknown;
    }

    let mut moves = legal_moves(state);
    if moves.is_empty() {
        // Hexo has no ordinary draw and its board is unbounded. This branch is
        // only reachable for an inconsistent non-winning terminal-like state,
        // so it cannot support a hard claim.
        return ProofStatus::Unknown;
    }

    // This is the load-bearing identity rule: FirstStone -> SecondStone keeps
    // the same player, so node type comes from identity rather than ply parity.
    let maximizing_root = state.current_player() == root_player;
    order_direct_extensions(state, state.current_player(), &mut moves);
    let mut saw_unknown = false;

    for coord in moves {
        let (_ignored_result, delta) = state
            .apply_with_delta(Placement { coord })
            .expect("independent legal enumerator produced an illegal placement");
        let child = minimax(state, root_player, plies_left - 1, nodes);
        state.undo(delta);

        if maximizing_root {
            if child == ProofStatus::Win {
                return ProofStatus::Win;
            }
        } else if child == ProofStatus::Loss {
            return ProofStatus::Loss;
        }
        saw_unknown |= child == ProofStatus::Unknown;
    }

    if saw_unknown {
        ProofStatus::Unknown
    } else if maximizing_root {
        // Every root-player option was a proven loss.
        ProofStatus::Loss
    } else {
        // Every opponent option was a proven win for the root player.
        ProofStatus::Win
    }
}

/// Independent alpha-beta move ordering: placements extending the mover's
/// longest contiguous line are tried first.  The legal set is unchanged and a
/// universal result still examines every child; this can only reduce the work
/// before an existential terminal witness is found.
fn order_direct_extensions(state: &HexoState, player: Player, moves: &mut [HexCoord]) {
    moves.sort_by_key(|coord| {
        (
            std::cmp::Reverse(direct_extension_length(state, player, *coord)),
            coord.q,
            coord.r,
        )
    });
}

fn direct_extension_length(state: &HexoState, player: Player, coord: HexCoord) -> u8 {
    let mut best = 1u8;
    for (dq, dr) in AXES {
        let mut length = 1u8;
        for sign in [-1, 1] {
            for distance in 1..WIN_LENGTH {
                let Some(cell) = offset_coord(coord, dq * sign, dr * sign, distance) else {
                    break;
                };
                if state.board().get(cell) != Some(player) {
                    break;
                }
                length = length.saturating_add(1);
            }
        }
        best = best.max(length);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::is_legal_placement;

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

    fn play(state: &mut HexoState, coord: HexCoord) {
        let _ = state.apply_with_delta(Placement { coord }).unwrap();
    }

    fn play_all(coords: &[HexCoord]) -> HexoState {
        let mut state = HexoState::new();
        for &coord in coords {
            play(&mut state, coord);
        }
        state
    }

    fn assert_same_state(actual: &HexoState, expected: &HexoState) {
        assert_eq!(actual.current_player(), expected.current_player());
        assert_eq!(actual.phase(), expected.phase());
        assert_eq!(actual.placements_made(), expected.placements_made());
        assert_eq!(actual.terminal(), expected.terminal());
        assert_eq!(actual.last_turn(), expected.last_turn());
        assert_eq!(actual.placement_history(), expected.placement_history());
        assert_eq!(
            actual.board().occupied_cells(),
            expected.board().occupied_cells()
        );
        for &coord in expected.board().occupied_cells() {
            assert_eq!(actual.board().get(coord), expected.board().get(coord));
        }
        assert_eq!(legal_moves(actual), legal_moves(expected));
    }

    #[test]
    fn independent_legal_moves_match_engine_on_seeded_states() {
        let mut saw = [false; 3];
        let mut checked = 0usize;

        for seed in 1..=12u64 {
            let mut rng = XorShift64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut state = HexoState::new();
            for _ in 0..36 {
                saw[match state.phase() {
                    TurnPhase::Opening => 0,
                    TurnPhase::FirstStone => 1,
                    TurnPhase::SecondStone { .. } => 2,
                }] = true;

                let independent = legal_moves(&state);
                assert_eq!(
                    independent.len(),
                    state.legal_move_count(),
                    "seed={seed}, ply={checked}"
                );
                assert!(independent
                    .windows(2)
                    .all(|pair| (pair[0].q, pair[0].r) < (pair[1].q, pair[1].r)));
                for &coord in &independent {
                    assert!(
                        is_legal_placement(&state, coord).is_ok(),
                        "reference-only move {coord:?}, seed={seed}, ply={checked}"
                    );
                }
                checked += 1;

                if independent.is_empty() {
                    break;
                }
                let coord = independent[(rng.next() as usize) % independent.len()];
                let (result, _delta) = state.apply_with_delta(Placement { coord }).unwrap();
                if result.outcome.is_some() {
                    assert!(legal_moves(&state).is_empty());
                    assert_eq!(state.legal_move_count(), 0);
                    break;
                }
            }
        }

        assert!(saw.into_iter().all(|phase| phase));
        assert!(checked >= 300);
    }

    #[test]
    fn player_identity_is_fixed_across_two_stone_turn() {
        // P1 has four stones blocked at the right by P0. From FirstStone P1
        // needs (-5, 0) and (-6, 0), so depth one is unresolved while depth
        // two is a win. Both choice nodes belong to P1.
        let state = play_all(&[
            HexCoord::ZERO,
            HexCoord::new(-1, 0),
            HexCoord::new(-2, 0),
            HexCoord::new(0, 5),
            HexCoord::new(0, 6),
            HexCoord::new(-3, 0),
            HexCoord::new(-4, 0),
            HexCoord::new(1, 5),
            HexCoord::new(1, 6),
        ]);
        assert_eq!(state.current_player(), Player::Player1);
        assert_eq!(state.phase(), TurnPhase::FirstStone);
        assert_eq!(solve(&state, 1).status, ProofStatus::Unknown);
        assert_eq!(solve(&state, 2).status, ProofStatus::Win);
    }

    #[test]
    fn direct_scanner_finds_six_on_each_axis() {
        let base = [
            HexCoord::ZERO,
            HexCoord::new(0, 5),
            HexCoord::new(0, 6),
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(1, 6),
            HexCoord::new(1, 7),
            HexCoord::new(3, 0),
            HexCoord::new(4, 0),
            HexCoord::new(2, 7),
            HexCoord::new(3, 7),
            HexCoord::new(5, 0),
        ];
        let transforms: [fn(HexCoord) -> HexCoord; 3] = [
            |c| c,
            |c| HexCoord::new(c.r, c.q),
            |c| HexCoord::new(c.q, -c.q - c.r),
        ];

        for transform in transforms {
            let coords: Vec<_> = base.into_iter().map(transform).collect();
            let state = play_all(&coords);
            assert!(state.is_terminal());
            assert_eq!(direct_winner(&state), Some(Player::Player0));
            let result = solve(&state, 0);
            assert_eq!(result.status, ProofStatus::Win);
            assert_eq!(result.nodes, 1);
        }

        assert_eq!(direct_winner(&HexoState::new()), None);
        assert_eq!(solve(&HexoState::new(), 0).status, ProofStatus::Unknown);
    }

    #[test]
    fn recursive_make_unmake_restores_exact_public_state() {
        let mut state = play_all(&[
            HexCoord::ZERO,
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(0, 2),
            HexCoord::new(0, 3),
        ]);
        let before = state.clone();
        let root_player = state.current_player();
        let mut nodes = 0;
        let _ = minimax(&mut state, root_player, 2, &mut nodes);
        assert!(nodes > 100);
        assert_same_state(&state, &before);
    }
}
