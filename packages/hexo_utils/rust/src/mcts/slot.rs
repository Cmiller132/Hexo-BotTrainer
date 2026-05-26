//! Resident game slots for future batched self-play and MCTS orchestration.
//!
//! This module intentionally does not schedule inference or run MCTS. It gives
//! future batching code a compact Rust-owned place to keep many active games,
//! their accepted action history, and an optional search-tree placeholder.

use crate::mcts::tree::SearchTree;
use hexo_engine::{
    apply_placement, unpack_coord, GameOutcome, HexoState, MoveError, PackedCoord, Placement,
};

/// Stable handle for a slot inside `GameSlotArena`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GameSlotId(pub usize);

/// Lifecycle state for one resident game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameSlotStatus {
    Active,
    Completed(GameOutcome),
    Aborted,
}

/// One Rust-owned resident game prepared for future batched MCTS.
#[derive(Clone, Debug)]
pub struct GameSlot {
    game_id: String,
    seed: Option<u64>,
    state: HexoState,
    status: GameSlotStatus,
    accepted_actions: Vec<PackedCoord>,
    search_tree: Option<SearchTree>,
}

impl GameSlot {
    /// Create a fresh active game slot.
    pub fn new(game_id: impl Into<String>, seed: Option<u64>) -> Self {
        Self {
            game_id: game_id.into(),
            seed,
            state: HexoState::new(),
            status: GameSlotStatus::Active,
            accepted_actions: Vec::new(),
            search_tree: None,
        }
    }

    /// Stable caller-owned game id.
    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    /// Optional seed recorded by the caller.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Current slot lifecycle state.
    pub fn status(&self) -> GameSlotStatus {
        self.status
    }

    /// Read-only access to the authoritative state stored in this slot.
    pub fn state(&self) -> &HexoState {
        &self.state
    }

    /// Clone the current state for player/search ownership.
    pub fn clone_decision_state(&self) -> HexoState {
        self.state.clone()
    }

    /// Deterministic packed legal actions from the real engine state.
    pub fn legal_action_ids(&self) -> Vec<PackedCoord> {
        let mut actions = Vec::with_capacity(self.state.legal_move_count());
        self.state.write_legal_action_ids(&mut actions);
        actions
    }

    /// Apply a packed action through the authoritative engine.
    pub fn apply_packed_action(
        &mut self,
        action_id: PackedCoord,
    ) -> Result<Option<GameOutcome>, MoveError> {
        let action = Placement {
            coord: unpack_coord(action_id),
        };
        match apply_placement(&mut self.state, action) {
            Ok(result) => {
                self.accepted_actions.push(action_id);
                if let Some(outcome) = result.outcome {
                    self.status = GameSlotStatus::Completed(outcome);
                    Ok(Some(outcome))
                } else {
                    Ok(None)
                }
            }
            Err(error) => {
                self.status = GameSlotStatus::Aborted;
                Err(error)
            }
        }
    }

    /// True when the authoritative game state is terminal.
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Accepted packed action history.
    pub fn accepted_actions(&self) -> &[PackedCoord] {
        &self.accepted_actions
    }

    /// Optional future search-tree placeholder.
    pub fn search_tree(&self) -> Option<&SearchTree> {
        self.search_tree.as_ref()
    }

    /// Mutable optional future search-tree placeholder.
    pub fn search_tree_mut(&mut self) -> Option<&mut SearchTree> {
        self.search_tree.as_mut()
    }

    /// Replace the optional search-tree placeholder.
    pub fn set_search_tree(&mut self, search_tree: Option<SearchTree>) {
        self.search_tree = search_tree;
    }

    /// Reset this allocation for another game.
    pub fn reset_for_reuse(&mut self, game_id: impl Into<String>, seed: Option<u64>) {
        self.game_id = game_id.into();
        self.seed = seed;
        self.state = HexoState::new();
        self.status = GameSlotStatus::Active;
        self.accepted_actions.clear();
        self.search_tree = None;
    }
}

/// Reusable arena for many resident games.
#[derive(Clone, Debug, Default)]
pub struct GameSlotArena {
    slots: Vec<Option<GameSlot>>,
    free_list: Vec<usize>,
    active_len: usize,
}

impl GameSlotArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty arena with preallocated slot capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            active_len: 0,
        }
    }

    /// Insert a new active game and return its stable slot id.
    pub fn insert(&mut self, game_id: impl Into<String>, seed: Option<u64>) -> GameSlotId {
        let slot = GameSlot::new(game_id, seed);
        self.active_len += 1;
        if let Some(index) = self.free_list.pop() {
            self.slots[index] = Some(slot);
            GameSlotId(index)
        } else {
            let index = self.slots.len();
            self.slots.push(Some(slot));
            GameSlotId(index)
        }
    }

    /// Return an immutable slot reference.
    pub fn get(&self, id: GameSlotId) -> Option<&GameSlot> {
        self.slots.get(id.0).and_then(Option::as_ref)
    }

    /// Return a mutable slot reference.
    pub fn get_mut(&mut self, id: GameSlotId) -> Option<&mut GameSlot> {
        self.slots.get_mut(id.0).and_then(Option::as_mut)
    }

    /// Release a slot allocation for future reuse.
    pub fn release(&mut self, id: GameSlotId) -> Option<GameSlot> {
        let slot = self.slots.get_mut(id.0)?.take()?;
        self.free_list.push(id.0);
        self.active_len -= 1;
        Some(slot)
    }

    /// Number of active slots.
    pub fn active_len(&self) -> usize {
        self.active_len
    }

    /// True if no slots are active.
    pub fn is_empty(&self) -> bool {
        self.active_len == 0
    }

    /// Allocated slot capacity, including reusable free slots.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{pack_coord, HexCoord, Player};

    fn play_to_player0_win(slot: &mut GameSlot) {
        for coord in [
            HexCoord::new(0, 0),
            HexCoord::new(0, 1),
            HexCoord::new(0, 2),
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(1, 1),
            HexCoord::new(1, 2),
            HexCoord::new(3, 0),
            HexCoord::new(4, 0),
            HexCoord::new(2, 1),
            HexCoord::new(2, 2),
            HexCoord::new(5, 0),
        ] {
            slot.apply_packed_action(pack_coord(coord)).unwrap();
        }
    }

    #[test]
    fn legal_ids_come_from_real_engine_state() {
        let slot = GameSlot::new("game", Some(7));
        assert_eq!(slot.legal_action_ids(), vec![pack_coord(HexCoord::ZERO)]);
    }

    #[test]
    fn applying_packed_action_mutates_only_that_slot() {
        let mut left = GameSlot::new("left", None);
        let right = GameSlot::new("right", None);

        left.apply_packed_action(pack_coord(HexCoord::ZERO))
            .unwrap();

        assert_eq!(left.accepted_actions().len(), 1);
        assert_eq!(right.accepted_actions().len(), 0);
        assert_eq!(left.state().placements_made(), 1);
        assert_eq!(right.state().placements_made(), 0);
    }

    #[test]
    fn terminal_status_and_history_are_recorded() {
        let mut slot = GameSlot::new("win", None);
        play_to_player0_win(&mut slot);

        assert!(slot.is_terminal());
        assert_eq!(slot.accepted_actions().len(), 12);
        assert!(matches!(
            slot.status(),
            GameSlotStatus::Completed(GameOutcome {
                winner: Player::Player0,
                placements: 12,
            })
        ));
    }

    #[test]
    fn arena_reuses_slots_without_leaking_prior_game_state() {
        let mut arena = GameSlotArena::with_capacity(2);
        let first = arena.insert("first", Some(1));
        arena
            .get_mut(first)
            .unwrap()
            .apply_packed_action(pack_coord(HexCoord::ZERO))
            .unwrap();
        let released = arena.release(first).unwrap();
        assert_eq!(released.accepted_actions().len(), 1);

        let second = arena.insert("second", Some(2));
        assert_eq!(first, second);
        let slot = arena.get(second).unwrap();
        assert_eq!(slot.game_id(), "second");
        assert_eq!(slot.seed(), Some(2));
        assert_eq!(slot.accepted_actions().len(), 0);
        assert_eq!(slot.state().placements_made(), 0);
        assert_eq!(arena.active_len(), 1);
        assert_eq!(arena.capacity(), 1);
    }
}
