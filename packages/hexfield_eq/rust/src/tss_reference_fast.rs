//! Test-only exact reference search with a bounded transposition table.
//!
//! This is deliberately independent of `tss_solver`: it reconstructs the
//! complete legal set from occupancy, scans wins directly from board owners,
//! builds its own exact position key, and applies the same three-valued
//! depth-limited recurrence as `tss_reference`.  The only search reductions
//! are exact transposition reuse and mover-value short circuits. Move ordering
//! changes work, never the legal universe.

use std::collections::{BTreeSet, HashMap};
use std::mem::size_of;
use std::time::{Duration, Instant};

use hexo_engine::{HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::ProofStatus;

const LEGAL_RADIUS: i32 = 8;
const WIN_LENGTH: i32 = 6;
const AXES: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];
const MAX_TT_BYTES: usize = 2 << 30;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FastReferenceConfig {
    pub(crate) tt_bytes_cap: usize,
    pub(crate) d6_canonical: bool,
    pub(crate) ordering_hint: FastOrderingHint,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum FastOrderingHint {
    #[default]
    None,
    DoubleForkCompact,
}

impl Default for FastReferenceConfig {
    fn default() -> Self {
        Self {
            tt_bytes_cap: 512 << 20,
            d6_canonical: false,
            ordering_hint: FastOrderingHint::None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FastReferenceResult {
    pub(crate) status: ProofStatus,
    pub(crate) nodes: u64,
    pub(crate) tt_hits: u64,
    pub(crate) tt_entries: usize,
    pub(crate) tt_accounted_bytes: usize,
    pub(crate) tt_clears: u64,
    pub(crate) elapsed: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactKey {
    stones: Box<[u64]>,
    plies_left: u32,
    current_player: u8,
    phase: PhaseKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum PhaseKey {
    Opening,
    FirstStone,
    SecondStone { q: i16, r: i16 },
}

#[derive(Clone, Copy, Debug)]
struct TtEntry {
    status: ProofStatus,
}

struct Search {
    root_player: Player,
    config: FastReferenceConfig,
    tt: HashMap<ExactKey, TtEntry>,
    tt_accounted_bytes: usize,
    nodes: u64,
    tt_hits: u64,
    tt_clears: u64,
}

pub(crate) fn solve(
    state: &HexoState,
    ply_budget: u32,
    config: FastReferenceConfig,
) -> FastReferenceResult {
    solve_for_player(state, state.current_player(), ply_budget, config)
}

pub(crate) fn solve_for_player(
    state: &HexoState,
    root_player: Player,
    ply_budget: u32,
    mut config: FastReferenceConfig,
) -> FastReferenceResult {
    config.tt_bytes_cap = config.tt_bytes_cap.min(MAX_TT_BYTES);
    let started = Instant::now();
    let mut search = Search {
        root_player,
        config,
        tt: HashMap::new(),
        tt_accounted_bytes: 0,
        nodes: 0,
        tt_hits: 0,
        tt_clears: 0,
    };
    let mut working = state.clone();
    let mut frontier = ExactFrontier::from_state(&working);
    let status = search.minimax(&mut working, &mut frontier, ply_budget);
    FastReferenceResult {
        status,
        nodes: search.nodes,
        tt_hits: search.tt_hits,
        tt_entries: search.tt.len(),
        tt_accounted_bytes: search.tt_accounted_bytes,
        tt_clears: search.tt_clears,
        elapsed: started.elapsed(),
    }
}

pub(crate) fn full_legal_moves(state: &HexoState) -> Vec<HexCoord> {
    legal_moves(state)
}

impl Search {
    fn minimax(
        &mut self,
        state: &mut HexoState,
        frontier: &mut ExactFrontier,
        plies_left: u32,
    ) -> ProofStatus {
        self.nodes = self.nodes.saturating_add(1);

        if let Some(winner) = direct_winner(state) {
            return if winner == self.root_player {
                ProofStatus::Win
            } else {
                ProofStatus::Loss
            };
        }
        if plies_left == 0 {
            return ProofStatus::Unknown;
        }

        let key = exact_key(state, plies_left, self.config.d6_canonical);
        if let Some(entry) = self.tt.get(&key) {
            self.tt_hits = self.tt_hits.saturating_add(1);
            return entry.status;
        }

        let mut moves = frontier.legal_moves(state);
        if moves.is_empty() {
            return ProofStatus::Unknown;
        }

        let mover = state.current_player();
        let maximizing_root = mover == self.root_player;
        if plies_left == 1 {
            let status = if moves
                .iter()
                .any(|&coord| direct_extension_length(state, mover, coord) >= WIN_LENGTH as u8)
            {
                if maximizing_root {
                    ProofStatus::Win
                } else {
                    ProofStatus::Loss
                }
            } else {
                ProofStatus::Unknown
            };
            self.insert_exact(key, status);
            return status;
        }
        if maximizing_root {
            order_moves(state, mover, &mut moves, self.config.ordering_hint);
        }
        let mut saw_unknown = false;

        let status = if maximizing_root {
            let mut answer = ProofStatus::Loss;
            for coord in moves {
                let frontier_delta = frontier.apply(state, coord);
                let (_ignored, delta) = state
                    .apply_with_delta(Placement { coord })
                    .expect("fast reference legal enumerator produced an illegal placement");
                let child = self.minimax(state, frontier, plies_left - 1);
                state.undo(delta);
                frontier.undo(state, frontier_delta);
                if child == ProofStatus::Win {
                    answer = ProofStatus::Win;
                    break;
                }
                saw_unknown |= child == ProofStatus::Unknown;
            }
            if answer == ProofStatus::Win {
                answer
            } else if saw_unknown {
                ProofStatus::Unknown
            } else {
                ProofStatus::Loss
            }
        } else {
            let mut answer = ProofStatus::Win;
            for coord in moves {
                let frontier_delta = frontier.apply(state, coord);
                let (_ignored, delta) = state
                    .apply_with_delta(Placement { coord })
                    .expect("fast reference legal enumerator produced an illegal placement");
                let child = self.minimax(state, frontier, plies_left - 1);
                state.undo(delta);
                frontier.undo(state, frontier_delta);
                if child == ProofStatus::Loss {
                    answer = ProofStatus::Loss;
                    break;
                }
                saw_unknown |= child == ProofStatus::Unknown;
            }
            if answer == ProofStatus::Loss {
                answer
            } else if saw_unknown {
                ProofStatus::Unknown
            } else {
                ProofStatus::Win
            }
        };

        self.insert_exact(key, status);
        status
    }

    fn insert_exact(&mut self, key: ExactKey, status: ProofStatus) {
        // Conservative accounting includes the key payload, map bucket,
        // allocator metadata, and slack for HashMap growth. Refusing a cache
        // insertion cannot alter the recurrence's value.
        let accounted = key
            .stones
            .len()
            .saturating_mul(size_of::<u64>())
            .saturating_add(size_of::<ExactKey>())
            .saturating_add(size_of::<TtEntry>())
            .saturating_add(96);
        if accounted > self.config.tt_bytes_cap {
            return;
        }
        if self
            .tt_accounted_bytes
            .checked_add(accounted)
            .is_none_or(|next| next > self.config.tt_bytes_cap)
        {
            self.tt.clear();
            self.tt_accounted_bytes = 0;
            self.tt_clears = self.tt_clears.saturating_add(1);
        }
        self.tt.insert(key, TtEntry { status });
        self.tt_accounted_bytes += accounted;
    }
}

#[derive(Debug)]
struct ExactFrontier {
    legal: BTreeSet<(i16, i16)>,
    cover_count: HashMap<(i16, i16), u16>,
}

#[derive(Clone, Copy, Debug)]
struct FrontierDelta {
    coord: (i16, i16),
    removed_cover: Option<u16>,
}

impl ExactFrontier {
    fn from_state(state: &HexoState) -> Self {
        let mut frontier = Self {
            legal: BTreeSet::new(),
            cover_count: HashMap::new(),
        };
        for &stone in state.board().occupied_cells() {
            frontier.add_radius(state, stone, None);
        }
        frontier
    }

    fn legal_moves(&self, state: &HexoState) -> Vec<HexCoord> {
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
        self.legal.iter().map(|&(q, r)| HexCoord { q, r }).collect()
    }

    fn apply(&mut self, state: &HexoState, coord: HexCoord) -> FrontierDelta {
        let encoded = (coord.q, coord.r);
        let removed_cover = self.cover_count.remove(&encoded);
        self.legal.remove(&encoded);
        self.add_radius(state, coord, Some(coord));
        FrontierDelta {
            coord: encoded,
            removed_cover,
        }
    }

    fn undo(&mut self, restored_state: &HexoState, delta: FrontierDelta) {
        let coord = HexCoord::new(delta.coord.0, delta.coord.1);
        for_each_radius(coord, |candidate| {
            if candidate == coord || !restored_state.board().is_cell_empty(candidate) {
                return;
            }
            let encoded = (candidate.q, candidate.r);
            let count = self
                .cover_count
                .get_mut(&encoded)
                .expect("frontier undo found a missing cover count");
            *count -= 1;
            if *count == 0 {
                self.cover_count.remove(&encoded);
                self.legal.remove(&encoded);
            }
        });
        if let Some(count) = delta.removed_cover {
            self.cover_count.insert(delta.coord, count);
            self.legal.insert(delta.coord);
        }
    }

    fn add_radius(&mut self, state: &HexoState, center: HexCoord, skip: Option<HexCoord>) {
        for_each_radius(center, |candidate| {
            if Some(candidate) == skip || !state.board().is_cell_empty(candidate) {
                return;
            }
            let encoded = (candidate.q, candidate.r);
            let count = self.cover_count.entry(encoded).or_insert(0);
            *count = count
                .checked_add(1)
                .expect("radius-eight cover count fits u16");
            self.legal.insert(encoded);
        });
    }
}

fn for_each_radius(center: HexCoord, mut visit: impl FnMut(HexCoord)) {
    // The center itself is included; callers explicitly skip it when the
    // newly occupied cell must not enter the legal set.
    for dq in -LEGAL_RADIUS..=LEGAL_RADIUS {
        let dr_min = (-LEGAL_RADIUS).max(-dq - LEGAL_RADIUS);
        let dr_max = LEGAL_RADIUS.min(-dq + LEGAL_RADIUS);
        for dr in dr_min..=dr_max {
            let q = i32::from(center.q) + dq;
            let r = i32::from(center.r) + dr;
            let (Ok(q), Ok(r)) = (i16::try_from(q), i16::try_from(r)) else {
                continue;
            };
            visit(HexCoord { q, r });
        }
    }
}

/// Reconstruct the full legal set without using the engine legal-move store.
fn legal_moves(state: &HexoState) -> Vec<HexCoord> {
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
        for dq in -LEGAL_RADIUS..=LEGAL_RADIUS {
            let dr_min = (-LEGAL_RADIUS).max(-dq - LEGAL_RADIUS);
            let dr_max = LEGAL_RADIUS.min(-dq + LEGAL_RADIUS);
            for dr in dr_min..=dr_max {
                let q = i32::from(stone.q) + dq;
                let r = i32::from(stone.r) + dr;
                let (Ok(q), Ok(r)) = (i16::try_from(q), i16::try_from(r)) else {
                    continue;
                };
                let coord = HexCoord { q, r };
                if state.board().is_cell_empty(coord) {
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

fn direct_winner(state: &HexoState) -> Option<Player> {
    [Player::Player0, Player::Player1]
        .into_iter()
        .find(|&player| has_six(state, player))
}

fn has_six(state: &HexoState, player: Player) -> bool {
    state.board().occupied_cells().iter().copied().any(|start| {
        state.board().get(start) == Some(player)
            && AXES.into_iter().any(|(dq, dr)| {
                (1..WIN_LENGTH).all(|offset| {
                    offset_coord(start, dq, dr, offset)
                        .is_some_and(|coord| state.board().get(coord) == Some(player))
                })
            })
    })
}

fn offset_coord(start: HexCoord, dq: i32, dr: i32, offset: i32) -> Option<HexCoord> {
    let q = i32::from(start.q) + dq * offset;
    let r = i32::from(start.r) + dr * offset;
    Some(HexCoord {
        q: i16::try_from(q).ok()?,
        r: i16::try_from(r).ok()?,
    })
}

fn order_moves(state: &HexoState, mover: Player, moves: &mut [HexCoord], hint: FastOrderingHint) {
    if matches!(hint, FastOrderingHint::DoubleForkCompact) {
        const PREFERRED: [HexCoord; 7] = [
            HexCoord { q: 4, r: 0 },
            HexCoord { q: 4, r: 7 },
            HexCoord { q: 0, r: 3 },
            HexCoord { q: 0, r: 8 },
            HexCoord { q: 0, r: 7 },
            HexCoord { q: 9, r: 7 },
            HexCoord { q: 4, r: 8 },
        ];
        let mut front = 0usize;
        for preferred in PREFERRED {
            if let Some(offset) = moves[front..].iter().position(|&coord| coord == preferred) {
                moves.swap(front, front + offset);
                front += 1;
            }
        }
        return;
    }
    moves.sort_by_cached_key(|coord| {
        (
            std::cmp::Reverse(direct_extension_length(state, mover, *coord)),
            std::cmp::Reverse(direct_extension_length(state, mover.other(), *coord)),
            coord.q,
            coord.r,
        )
    });
}

fn direct_extension_length(state: &HexoState, player: Player, coord: HexCoord) -> u8 {
    AXES.into_iter()
        .map(|(dq, dr)| {
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
            length
        })
        .max()
        .unwrap_or(1)
}

fn exact_key(state: &HexoState, plies_left: u32, d6_canonical: bool) -> ExactKey {
    let frames = if d6_canonical { 12 } else { 1 };
    let mut best: Option<(Box<[u64]>, PhaseKey)> = None;
    for frame in 0..frames {
        let Some(phase) = transform_phase(state.phase(), frame) else {
            continue;
        };
        let mut stones = Vec::with_capacity(state.board().occupied_cells().len());
        let mut valid = true;
        for &coord in state.board().occupied_cells() {
            let Some(transformed) = transform_coord(coord, frame) else {
                valid = false;
                break;
            };
            let owner = state
                .board()
                .get(coord)
                .expect("occupied coordinate has an owner");
            stones.push(pack_stone(transformed, owner));
        }
        if !valid {
            continue;
        }
        stones.sort_unstable();
        let candidate = (stones.into_boxed_slice(), phase);
        if best.as_ref().is_none_or(|current| candidate < *current) {
            best = Some(candidate);
        }
    }
    let (stones, phase) = best.expect("identity frame is always representable");
    ExactKey {
        stones,
        plies_left,
        current_player: state.current_player().index() as u8,
        phase,
    }
}

fn pack_stone(coord: HexCoord, owner: Player) -> u64 {
    (u64::from(coord.q as u16) << 17) | (u64::from(coord.r as u16) << 1) | owner.index() as u64
}

fn transform_phase(phase: TurnPhase, frame: u8) -> Option<PhaseKey> {
    Some(match phase {
        TurnPhase::Opening => PhaseKey::Opening,
        TurnPhase::FirstStone => PhaseKey::FirstStone,
        TurnPhase::SecondStone { first } => {
            let first = transform_coord(first, frame)?;
            PhaseKey::SecondStone {
                q: first.q,
                r: first.r,
            }
        }
    })
}

/// Independent axial-coordinate D6 action (six rotations, then reflection).
fn transform_coord(coord: HexCoord, frame: u8) -> Option<HexCoord> {
    let mut q = i32::from(coord.q);
    let mut r = i32::from(coord.r);
    if frame >= 6 {
        std::mem::swap(&mut q, &mut r);
    }
    for _ in 0..(frame % 6) {
        (q, r) = (-r, q + r);
    }
    Some(HexCoord {
        q: i16::try_from(q).ok()?,
        r: i16::try_from(r).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::apply_placement;

    fn replay(coords: &[(i16, i16)]) -> HexoState {
        let mut state = HexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap();
        }
        state
    }

    #[test]
    #[ignore = "test-only exact-oracle validation; run explicitly"]
    fn exact_key_ignores_placement_order_but_not_phase_first() {
        let a = replay(&[(0, 0), (1, 0), (2, 0), (0, 2), (0, 3)]);
        let b = replay(&[(0, 0), (2, 0), (1, 0), (0, 3), (0, 2)]);
        assert_eq!(exact_key(&a, 3, false), exact_key(&b, 3, false));

        let first_a = replay(&[(0, 0), (1, 0)]);
        let first_b = replay(&[(0, 0), (2, 0)]);
        assert_ne!(exact_key(&first_a, 3, false), exact_key(&first_b, 3, false));
    }

    #[test]
    #[ignore = "test-only exact-oracle validation; run explicitly"]
    fn d6_key_matches_all_independent_images() {
        let state = replay(&[(0, 0), (1, 0), (2, -1), (0, 2), (-1, 3)]);
        let expected = exact_key(&state, 4, true);
        for frame in 0..12 {
            let image = state
                .placement_history()
                .iter()
                .map(|record| transform_coord(record.coord, frame).unwrap())
                .collect::<Vec<_>>();
            let mut transformed = HexoState::new();
            for coord in image {
                apply_placement(&mut transformed, Placement { coord }).unwrap();
            }
            assert_eq!(exact_key(&transformed, 4, true), expected);
        }
    }

    #[test]
    #[ignore = "test-only exact-oracle validation; run explicitly"]
    fn small_recurrence_matches_stock_reference() {
        let states = [
            replay(&[(0, 0)]),
            replay(&[(0, 0), (1, 0)]),
            replay(&[(0, 0), (1, 0), (2, 0)]),
        ];
        for state in states {
            for depth in 0..=2 {
                assert_eq!(
                    solve(
                        &state,
                        depth,
                        FastReferenceConfig {
                            tt_bytes_cap: 8 << 20,
                            d6_canonical: true,
                            ordering_hint: FastOrderingHint::None,
                        },
                    )
                    .status,
                    crate::tss_reference::solve(&state, depth).status,
                );
            }
        }
    }

    #[test]
    #[ignore = "test-only exact-oracle validation; run explicitly"]
    fn incremental_frontier_matches_full_rebuild_through_undo() {
        let mut state = replay(&[(0, 0), (1, 0), (2, -1), (0, 2), (-1, 3)]);
        let before = state.clone();
        let mut frontier = ExactFrontier::from_state(&state);
        assert_eq!(frontier.legal_moves(&state), legal_moves(&state));

        for _ in 0..2 {
            let coord = frontier.legal_moves(&state)[37];
            let frontier_delta = frontier.apply(&state, coord);
            let (_result, state_delta) = state.apply_with_delta(Placement { coord }).unwrap();
            assert_eq!(frontier.legal_moves(&state), legal_moves(&state));
            state.undo(state_delta);
            frontier.undo(&state, frontier_delta);
            assert_eq!(frontier.legal_moves(&state), legal_moves(&state));
        }
        assert_eq!(state.placement_history(), before.placement_history());
    }
}
