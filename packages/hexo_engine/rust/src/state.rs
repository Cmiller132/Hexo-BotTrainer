//! Game state and phase-aware move application.
//!
//! This is the heart of the rule engine. Hexo turns are represented
//! autoregressively:
//! - `Opening`: Player 0 places the center stone.
//! - `FirstStone`: current player places the first stone of a normal turn.
//! - `SecondStone`: the same player places the second stone, then turn passes.
//!
//! A win is checked after every single placement. If the first stone of a
//! two-stone turn wins, the second stone is never played.

use super::board::Board;
use super::coord::HexCoord;
use super::error::{MoveError, StateLoadError};
use super::rules::is_legal_placement;
use super::snapshot::{StateSnapshot, HEXO_STATE_SNAPSHOT_VERSION};
use super::tactics::WindowUpdate;
use serde::{Deserialize, Serialize};

/// Player identifier and stone owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    Player0,
    Player1,
}

impl Player {
    /// Return the opponent.
    pub fn other(self) -> Self {
        match self {
            Self::Player0 => Self::Player1,
            Self::Player1 => Self::Player0,
        }
    }

    /// Stable zero-based index for arrays and tensors.
    pub fn index(self) -> usize {
        match self {
            Self::Player0 => 0,
            Self::Player1 => 1,
        }
    }
}

/// Where the current player is inside the autoregressive turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TurnPhase {
    /// Game start. Only Player 0 at `(0, 0)` is legal.
    Opening,
    /// First placement of a normal two-stone turn.
    FirstStone,
    /// Second placement of the same turn; stores the first coordinate so the
    /// same cell cannot be reused and encoders can mark it.
    SecondStone { first: HexCoord },
}

/// One single-stone action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Placement {
    pub coord: HexCoord,
}

/// Terminal result. Hexo has no normal draw in this prototype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameOutcome {
    /// Winning player.
    pub winner: Player,
    /// Number of stones placed when the game ended.
    pub placements: u32,
}

/// Flat history record for encoders and training samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementRecord {
    /// Player who placed the stone.
    pub player: Player,
    /// Coordinate that was placed.
    pub coord: HexCoord,
    /// Phase before the stone was placed.
    pub phase: TurnPhase,
    /// One-based placement count after this stone is applied.
    pub placement_index: u32,
}

/// Human-sized record of the most recent logical turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveRecord {
    /// Player who took the turn.
    pub player: Player,
    /// One coordinate for opening, two coordinates for a full normal turn.
    pub placements: Vec<HexCoord>,
}

/// Complete Hexo game state.
#[derive(Clone, Debug)]
pub struct HexoState {
    /// Sparse unlimited board.
    board: Board,
    /// Player who chooses the next placement.
    current_player: Player,
    /// Current point in the opening/first/second placement sequence.
    phase: TurnPhase,
    /// Total number of stones placed.
    placements_made: u32,
    /// Set once a player has six in a line.
    terminal: Option<GameOutcome>,
    /// Most recent logical turn progress.
    last_turn: Option<MoveRecord>,
    /// Full single-placement history for encoding recent stones.
    placement_history: Vec<PlacementRecord>,
}

/// Summary returned after applying one placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyResult {
    /// Coordinate that was placed.
    pub placed: HexCoord,
    /// Player who placed the stone.
    pub player: Player,
    /// Phase before applying the placement.
    pub phase_before: TurnPhase,
    /// Phase after applying the placement. Unchanged if the move ended game.
    pub phase_after: TurnPhase,
    /// Terminal outcome if this placement won immediately.
    pub outcome: Option<GameOutcome>,
    /// Windows changed by this placement plus any threat/win windows.
    pub window_update: WindowUpdate,
}

impl Default for HexoState {
    fn default() -> Self {
        Self::new()
    }
}

impl HexoState {
    /// Create the initial empty game state.
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            current_player: Player::Player0,
            phase: TurnPhase::Opening,
            placements_made: 0,
            terminal: None,
            last_turn: None,
            placement_history: Vec::new(),
        }
    }

    /// Read-only access to board occupancy.
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Player who must choose the next single placement.
    pub fn current_player(&self) -> Player {
        self.current_player
    }

    /// Current turn phase.
    pub fn phase(&self) -> TurnPhase {
        self.phase
    }

    /// Total stones placed so far.
    pub fn placements_made(&self) -> u32 {
        self.placements_made
    }

    /// Terminal result, if the game has ended.
    pub fn terminal(&self) -> Option<GameOutcome> {
        self.terminal
    }

    /// True once no more moves should be generated.
    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// Most recent logical turn progress.
    pub fn last_turn(&self) -> Option<&MoveRecord> {
        self.last_turn.as_ref()
    }

    /// Complete single-placement history.
    pub fn placement_history(&self) -> &[PlacementRecord] {
        &self.placement_history
    }

    /// Export a compact snapshot that can be passed to `load_state`.
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot::new(
            self.placement_history
                .iter()
                .map(|record| record.coord)
                .collect(),
        )
    }

    /// Append a single-stone history entry after placement succeeds.
    fn push_history(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        self.placement_history.push(PlacementRecord {
            player,
            coord,
            phase,
            placement_index: self.placements_made,
        });
    }

    fn record_turn_progress(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        let placements = match phase {
            TurnPhase::Opening | TurnPhase::FirstStone => vec![coord],
            TurnPhase::SecondStone { first } => vec![first, coord],
        };
        self.last_turn = Some(MoveRecord { player, placements });
    }
}

/// Build authoritative state by replaying a validated startup/resume snapshot.
pub fn load_state(snapshot: &StateSnapshot) -> Result<HexoState, StateLoadError> {
    if snapshot.rules_version != HEXO_STATE_SNAPSHOT_VERSION {
        return Err(StateLoadError::UnsupportedSnapshotVersion {
            found: snapshot.rules_version,
            expected: HEXO_STATE_SNAPSHOT_VERSION,
        });
    }

    let mut state = HexoState::new();

    for (index, coord) in snapshot.placements.iter().copied().enumerate() {
        apply_placement(&mut state, Placement { coord })
            .map_err(|source| StateLoadError::IllegalPlacement { index, source })?;
    }

    Ok(state)
}

/// Apply one single-stone placement and advance the phase machine.
///
/// The function performs the full rule sequence:
/// 1. Validate the coordinate against the current phase.
/// 2. Place the stone for the current player.
/// 3. Record history.
/// 4. Check for an immediate six-in-line win.
/// 5. If not terminal, advance phase/current player.
pub fn apply_placement(
    state: &mut HexoState,
    placement: Placement,
) -> Result<ApplyResult, MoveError> {
    is_legal_placement(state, placement.coord)?;

    let player = state.current_player;
    let phase_before = state.phase;
    let window_update = state.board.place(placement.coord, player)?;
    state.placements_made += 1;
    state.push_history(player, placement.coord, phase_before);
    state.record_turn_progress(player, placement.coord, phase_before);

    // The board updates all affected six-cell windows during placement, so win
    // detection is now an O(18) incremental check rather than a separate line
    // scan through the board.
    if window_update.has_win() {
        let outcome = GameOutcome {
            winner: player,
            placements: state.placements_made,
        };
        state.terminal = Some(outcome);
        return Ok(ApplyResult {
            placed: placement.coord,
            player,
            phase_before,
            phase_after: state.phase,
            outcome: Some(outcome),
            window_update,
        });
    }

    match phase_before {
        TurnPhase::Opening => {
            // Opening is a special one-stone turn by Player 0. After it,
            // Player 1 starts the first normal two-stone turn.
            state.current_player = Player::Player1;
            state.phase = TurnPhase::FirstStone;
        }
        TurnPhase::FirstStone => {
            // The same player remains to place the second stone.
            state.phase = TurnPhase::SecondStone {
                first: placement.coord,
            };
        }
        TurnPhase::SecondStone { .. } => {
            // A normal two-stone turn is complete, so control passes.
            state.current_player = player.other();
            state.phase = TurnPhase::FirstStone;
        }
    }

    Ok(ApplyResult {
        placed: placement.coord,
        player,
        phase_before,
        phase_after: state.phase,
        outcome: None,
        window_update,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::HEXO_STATE_SNAPSHOT_VERSION;
    use serde_json::json;

    fn sample_state() -> HexoState {
        let mut state = HexoState::new();
        for coord in [
            HexCoord::ZERO,
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(0, 1),
            HexCoord::new(0, 2),
        ] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    fn assert_same_public_state(left: &HexoState, right: &HexoState) {
        assert_eq!(left.current_player(), right.current_player());
        assert_eq!(left.phase(), right.phase());
        assert_eq!(left.placements_made(), right.placements_made());
        assert_eq!(left.terminal(), right.terminal());
        assert_eq!(left.last_turn(), right.last_turn());
        assert_eq!(left.placement_history(), right.placement_history());
        assert_eq!(
            left.board().occupied_cells(),
            right.board().occupied_cells()
        );
        for coord in left.board().occupied_cells() {
            assert_eq!(left.board().get(*coord), right.board().get(*coord));
        }
    }

    #[test]
    fn snapshot_uses_canonical_shape() {
        let state = sample_state();
        let snapshot = state.snapshot();
        let value = serde_json::to_value(&snapshot).unwrap();

        assert_eq!(value["rules_version"], json!(HEXO_STATE_SNAPSHOT_VERSION));
        assert_eq!(value["placements"].as_array().unwrap().len(), 5);
        assert!(value.get("board").is_none());
    }

    #[test]
    fn load_state_replays_snapshot() {
        let state = sample_state();
        let snapshot = state.snapshot();

        let decoded = load_state(&snapshot).unwrap();

        assert_same_public_state(&decoded, &state);
    }

    #[test]
    fn load_state_rejects_unsupported_snapshot_version() {
        let state = sample_state();
        let mut snapshot = state.snapshot();
        snapshot.rules_version = HEXO_STATE_SNAPSHOT_VERSION + 1;

        assert!(matches!(
            load_state(&snapshot),
            Err(StateLoadError::UnsupportedSnapshotVersion { .. })
        ));
    }

    #[test]
    fn load_state_rejects_illegal_snapshot_placement() {
        let state = sample_state();
        let mut snapshot = state.snapshot();
        snapshot.placements[1] = snapshot.placements[0];

        assert!(matches!(
            load_state(&snapshot),
            Err(StateLoadError::IllegalPlacement {
                index: 1,
                source: MoveError::Occupied(coord),
            }) if coord == HexCoord::ZERO
        ));
    }
}
