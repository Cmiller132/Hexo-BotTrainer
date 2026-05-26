//! Resident game slots for future batched self-play and MCTS orchestration.
//!
//! This module intentionally does not schedule inference or run MCTS. It gives
//! future batching code a compact Rust-owned place to keep many active games,
//! their accepted action history, and an optional search-tree placeholder.

use crate::mcts::tree::SearchTree;
use hexo_engine::{
    apply_placement, unpack_coord, GameOutcome, HexoState, MoveError, PackedCoord, Placement,
    Player,
};
use std::mem::{size_of, size_of_val};

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

/// Conservative memory accounting for one resident game slot.
///
/// This reports the slot struct itself plus heap allocations owned directly by
/// the slot scaffolding. Private engine allocations inside `HexoState` are
/// represented by the inline `HexoState` size only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GameSlotMemoryUsage {
    pub inline_bytes: usize,
    pub game_id_bytes: usize,
    pub accepted_actions_bytes: usize,
    pub abort_error_bytes: usize,
    pub search_tree_bytes: usize,
}

impl GameSlotMemoryUsage {
    /// Total tracked bytes for this slot.
    pub fn total_bytes(self) -> usize {
        self.inline_bytes
            + self.game_id_bytes
            + self.accepted_actions_bytes
            + self.abort_error_bytes
            + self.search_tree_bytes
    }
}

/// Conservative memory accounting for an arena and its resident slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GameSlotArenaMemoryUsage {
    pub inline_bytes: usize,
    pub slot_storage_bytes: usize,
    pub free_list_bytes: usize,
    pub resident_slot_heap_bytes: usize,
}

impl GameSlotArenaMemoryUsage {
    /// Total tracked bytes for this arena.
    pub fn total_bytes(self) -> usize {
        self.inline_bytes
            + self.slot_storage_bytes
            + self.free_list_bytes
            + self.resident_slot_heap_bytes
    }
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
    abort_error: Option<String>,
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
            abort_error: None,
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

    /// Player who must choose the next single placement.
    pub fn current_player(&self) -> Player {
        self.state.current_player()
    }

    /// Total stones placed so far.
    pub fn placements_made(&self) -> u32 {
        self.state.placements_made()
    }

    /// Terminal outcome from the authoritative state, if any.
    pub fn terminal(&self) -> Option<GameOutcome> {
        self.state.terminal()
    }

    /// Clone the current state for player/search ownership.
    pub fn clone_decision_state(&self) -> HexoState {
        self.state.clone()
    }

    /// Deterministic packed legal actions from the real engine state.
    pub fn legal_action_ids(&self) -> Vec<PackedCoord> {
        if self.status != GameSlotStatus::Active {
            return Vec::new();
        }

        let mut actions = Vec::with_capacity(self.state.legal_move_count());
        self.state.write_legal_action_ids(&mut actions);
        actions
    }

    /// Apply a packed action through the authoritative engine.
    pub fn apply_packed_action(
        &mut self,
        action_id: PackedCoord,
    ) -> Result<Option<GameOutcome>, MoveError> {
        match self.status {
            GameSlotStatus::Active => {}
            GameSlotStatus::Completed(_) | GameSlotStatus::Aborted => {
                return Err(MoveError::TerminalState);
            }
        }

        let action = Placement {
            coord: unpack_coord(action_id),
        };
        match apply_placement(&mut self.state, action) {
            Ok(result) => {
                self.accepted_actions.push(action_id);
                self.clear_search_state();
                self.abort_error = None;
                if let Some(outcome) = result.outcome {
                    self.status = GameSlotStatus::Completed(outcome);
                    Ok(Some(outcome))
                } else {
                    Ok(None)
                }
            }
            Err(error) => {
                self.status = GameSlotStatus::Aborted;
                self.abort_error = Some(error.to_string());
                self.clear_search_state();
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

    /// Drop any stored search state placeholder.
    pub fn clear_search_state(&mut self) {
        self.search_tree = None;
    }

    /// Abort this resident game with caller-provided diagnostic text.
    pub fn abort(&mut self, error: impl Into<String>) {
        self.status = GameSlotStatus::Aborted;
        self.abort_error = Some(error.into());
        self.clear_search_state();
    }

    /// Abort/error diagnostic recorded for this slot, if any.
    pub fn abort_error(&self) -> Option<&str> {
        self.abort_error.as_deref()
    }

    /// Conservative accounting for bytes tracked by the slot scaffolding.
    pub fn memory_usage(&self) -> GameSlotMemoryUsage {
        GameSlotMemoryUsage {
            inline_bytes: size_of_val(self),
            game_id_bytes: self.game_id.capacity(),
            accepted_actions_bytes: self.accepted_actions.capacity() * size_of::<PackedCoord>(),
            abort_error_bytes: self
                .abort_error
                .as_ref()
                .map_or(0, |error| error.capacity()),
            search_tree_bytes: self.search_tree.as_ref().map_or(0, search_tree_heap_bytes),
        }
    }

    /// Total tracked bytes for this slot.
    pub fn memory_usage_bytes(&self) -> usize {
        self.memory_usage().total_bytes()
    }

    /// Reset this allocation for another game.
    pub fn reset_for_reuse(&mut self, game_id: impl Into<String>, seed: Option<u64>) {
        self.game_id = game_id.into();
        self.seed = seed;
        self.state = HexoState::new();
        self.status = GameSlotStatus::Active;
        self.accepted_actions.clear();
        self.search_tree = None;
        self.abort_error = None;
    }

    fn heap_usage_bytes(&self) -> usize {
        self.memory_usage().total_bytes() - size_of_val(self)
    }
}

/// Reusable arena for many resident games.
#[derive(Clone, Debug, Default)]
pub struct GameSlotArena {
    slots: Vec<Option<GameSlot>>,
    free_list: Vec<usize>,
    resident_len: usize,
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
            resident_len: 0,
        }
    }

    /// Insert a new active game and return its stable slot id.
    pub fn insert(&mut self, game_id: impl Into<String>, seed: Option<u64>) -> GameSlotId {
        let slot = GameSlot::new(game_id, seed);
        self.resident_len += 1;
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

    /// Iterate active slot ids and immutable slot references.
    pub fn iter_active(&self) -> impl Iterator<Item = (GameSlotId, &GameSlot)> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let slot = slot.as_ref()?;
            (slot.status() == GameSlotStatus::Active).then_some((GameSlotId(index), slot))
        })
    }

    /// Iterate ids for slots whose lifecycle status is active.
    pub fn iter_active_ids(&self) -> impl Iterator<Item = GameSlotId> + '_ {
        self.iter_active().map(|(id, _)| id)
    }

    /// Collect ids for slots whose lifecycle status is active.
    pub fn active_ids(&self) -> Vec<GameSlotId> {
        self.iter_active_ids().collect()
    }

    /// Release a slot allocation for future reuse.
    pub fn release(&mut self, id: GameSlotId) -> Option<GameSlot> {
        let slot = self.slots.get_mut(id.0)?.take()?;
        self.free_list.push(id.0);
        self.resident_len -= 1;
        Some(slot)
    }

    /// Release all completed or aborted slots for future reuse.
    pub fn release_completed(&mut self) -> Vec<(GameSlotId, GameSlot)> {
        let mut released = Vec::new();
        for index in 0..self.slots.len() {
            let should_release = self.slots[index]
                .as_ref()
                .is_some_and(|slot| slot.status() != GameSlotStatus::Active);
            if should_release {
                if let Some(slot) = self.slots[index].take() {
                    self.free_list.push(index);
                    self.resident_len -= 1;
                    released.push((GameSlotId(index), slot));
                }
            }
        }
        released
    }

    /// Number of slots whose lifecycle status is active.
    pub fn active_len(&self) -> usize {
        self.iter_active().count()
    }

    /// Number of resident slots, including completed or aborted games awaiting release.
    pub fn resident_len(&self) -> usize {
        self.resident_len
    }

    /// True if no slots have active lifecycle status.
    pub fn is_empty(&self) -> bool {
        self.active_len() == 0
    }

    /// Allocated slot positions, including reusable free slots.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Reserved slot storage before the arena must reallocate.
    pub fn reserved_capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Number of reusable free slot positions.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Conservative accounting for bytes tracked by the arena scaffolding.
    pub fn memory_usage(&self) -> GameSlotArenaMemoryUsage {
        GameSlotArenaMemoryUsage {
            inline_bytes: size_of_val(self),
            slot_storage_bytes: self.slots.capacity() * size_of::<Option<GameSlot>>(),
            free_list_bytes: self.free_list.capacity() * size_of::<usize>(),
            resident_slot_heap_bytes: self
                .slots
                .iter()
                .filter_map(Option::as_ref)
                .map(GameSlot::heap_usage_bytes)
                .sum(),
        }
    }

    /// Total tracked bytes for this arena.
    pub fn memory_usage_bytes(&self) -> usize {
        self.memory_usage().total_bytes()
    }
}

fn search_tree_heap_bytes(search_tree: &SearchTree) -> usize {
    search_tree.nodes.capacity() * size_of::<crate::mcts::tree::Node>()
        + search_tree
            .nodes
            .iter()
            .map(|node| node.edges.capacity() * size_of::<crate::mcts::tree::Edge>())
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::tree::{Edge, Node};
    use hexo_engine::{apply_placement, pack_coord, HexCoord, Placement, Player, TurnPhase};

    fn search_tree_placeholder() -> SearchTree {
        let mut tree = SearchTree::with_root(Node::new(Player::Player0, TurnPhase::Opening));
        tree.nodes[0].edges.push(Edge::new(HexCoord::ZERO, 1.0));
        tree
    }

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
        let mut slot = GameSlot::new("game", Some(7));
        assert_eq!(slot.legal_action_ids(), vec![pack_coord(HexCoord::ZERO)]);

        slot.apply_packed_action(pack_coord(HexCoord::ZERO))
            .unwrap();
        let legal_ids = slot.legal_action_ids();
        assert_eq!(legal_ids.len(), 216);
        assert!(legal_ids.contains(&pack_coord(HexCoord::new(-8, 0))));
        assert!(!legal_ids.contains(&pack_coord(HexCoord::ZERO)));
    }

    #[test]
    fn clone_decision_state_is_isolated_from_slot_state() {
        let slot = GameSlot::new("game", None);
        let mut cloned = slot.clone_decision_state();

        apply_placement(
            &mut cloned,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .unwrap();

        assert_eq!(slot.placements_made(), 0);
        assert_eq!(slot.current_player(), Player::Player0);
        assert_eq!(cloned.placements_made(), 1);
        assert_eq!(cloned.current_player(), Player::Player1);
    }

    #[test]
    fn applying_packed_action_mutates_only_that_slot() {
        let mut left = GameSlot::new("left", None);
        let right = GameSlot::new("right", None);

        left.apply_packed_action(pack_coord(HexCoord::ZERO))
            .unwrap();

        assert_eq!(left.accepted_actions().len(), 1);
        assert_eq!(right.accepted_actions().len(), 0);
        assert_eq!(left.placements_made(), 1);
        assert_eq!(right.placements_made(), 0);
        assert_eq!(left.current_player(), Player::Player1);
    }

    #[test]
    fn terminal_status_and_history_are_recorded() {
        let mut slot = GameSlot::new("win", None);
        play_to_player0_win(&mut slot);

        assert!(slot.is_terminal());
        assert_eq!(slot.accepted_actions().len(), 12);
        assert_eq!(
            slot.terminal(),
            Some(GameOutcome {
                winner: Player::Player0,
                placements: 12,
            })
        );
        assert!(matches!(
            slot.status(),
            GameSlotStatus::Completed(GameOutcome {
                winner: Player::Player0,
                placements: 12,
            })
        ));
    }

    #[test]
    fn post_terminal_action_is_rejected_without_mutating_history() {
        let mut slot = GameSlot::new("win", None);
        play_to_player0_win(&mut slot);

        let accepted_before = slot.accepted_actions().len();
        let result = slot.apply_packed_action(pack_coord(HexCoord::new(6, 0)));

        assert_eq!(result, Err(MoveError::TerminalState));
        assert_eq!(slot.accepted_actions().len(), accepted_before);
        assert_eq!(slot.placements_made(), 12);
    }

    #[test]
    fn abort_state_records_error_and_rejects_moves() {
        let mut slot = GameSlot::new("abort", None);
        slot.set_search_tree(Some(search_tree_placeholder()));

        slot.abort("cancelled by caller");

        assert_eq!(slot.status(), GameSlotStatus::Aborted);
        assert_eq!(slot.abort_error(), Some("cancelled by caller"));
        assert!(slot.search_tree().is_none());
        assert_eq!(
            slot.apply_packed_action(pack_coord(HexCoord::ZERO)),
            Err(MoveError::TerminalState)
        );
        assert_eq!(slot.accepted_actions().len(), 0);
        assert!(slot.legal_action_ids().is_empty());
    }

    #[test]
    fn illegal_action_aborts_and_stores_engine_error() {
        let mut slot = GameSlot::new("illegal", None);
        let result = slot.apply_packed_action(pack_coord(HexCoord::new(1, 0)));

        assert_eq!(result, Err(MoveError::IllegalOpening));
        assert_eq!(slot.status(), GameSlotStatus::Aborted);
        assert_eq!(
            slot.abort_error(),
            Some("opening placement must be at (0, 0)")
        );
        assert_eq!(slot.accepted_actions().len(), 0);
    }

    #[test]
    fn clear_search_state_and_memory_accounting_are_explicit() {
        let mut slot = GameSlot::new("memory", None);
        let baseline = slot.memory_usage();
        assert_eq!(baseline.total_bytes(), slot.memory_usage_bytes());
        assert!(baseline.inline_bytes >= size_of::<GameSlot>());
        assert_eq!(baseline.search_tree_bytes, 0);

        slot.set_search_tree(Some(search_tree_placeholder()));
        let with_tree = slot.memory_usage();
        assert!(with_tree.search_tree_bytes >= size_of::<Node>());

        slot.clear_search_state();
        assert!(slot.search_tree().is_none());
        assert_eq!(slot.memory_usage().search_tree_bytes, 0);
    }

    #[test]
    fn arena_reuses_slots_without_leaking_prior_game_state() {
        let mut arena = GameSlotArena::with_capacity(2);
        let first = arena.insert("first", Some(1));
        {
            let slot = arena.get_mut(first).unwrap();
            slot.apply_packed_action(pack_coord(HexCoord::ZERO))
                .unwrap();
            slot.set_search_tree(Some(search_tree_placeholder()));
            slot.abort("stale abort");
        }

        let released = arena.release_completed();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].0, first);
        assert_eq!(released[0].1.accepted_actions().len(), 1);
        assert_eq!(released[0].1.status(), GameSlotStatus::Aborted);
        assert_eq!(arena.free_count(), 1);

        let second = arena.insert("second", Some(2));
        assert_eq!(first, second);
        let slot = arena.get(second).unwrap();
        assert_eq!(slot.game_id(), "second");
        assert_eq!(slot.seed(), Some(2));
        assert_eq!(slot.status(), GameSlotStatus::Active);
        assert_eq!(slot.abort_error(), None);
        assert!(slot.search_tree().is_none());
        assert_eq!(slot.accepted_actions().len(), 0);
        assert_eq!(slot.placements_made(), 0);
        assert_eq!(arena.active_len(), 1);
        assert_eq!(arena.resident_len(), 1);
        assert_eq!(arena.capacity(), 1);
        assert_eq!(arena.free_count(), 0);
        assert!(arena.reserved_capacity() >= 2);
    }

    #[test]
    fn arena_iteration_tracks_only_active_slots() {
        let mut arena = GameSlotArena::new();
        let active = arena.insert("active", None);
        let aborted = arena.insert("aborted", None);
        let completed = arena.insert("completed", None);

        arena.get_mut(aborted).unwrap().abort("stop");
        play_to_player0_win(arena.get_mut(completed).unwrap());

        assert_eq!(arena.iter_active_ids().collect::<Vec<_>>(), vec![active]);
        assert_eq!(arena.active_ids(), vec![active]);
        let active_games = arena
            .iter_active()
            .map(|(id, slot)| (id, slot.game_id().to_owned()))
            .collect::<Vec<_>>();
        assert_eq!(active_games, vec![(active, "active".to_owned())]);
        assert_eq!(arena.active_len(), 1);
        assert_eq!(arena.resident_len(), 3);
        assert_eq!(arena.free_count(), 0);

        let released = arena.release_completed();
        let released_ids = released.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert_eq!(released_ids, vec![aborted, completed]);
        assert_eq!(arena.active_ids(), vec![active]);
        assert_eq!(arena.resident_len(), 1);
        assert_eq!(arena.free_count(), 2);
        assert!(arena.memory_usage_bytes() >= arena.memory_usage().inline_bytes);
    }
}
