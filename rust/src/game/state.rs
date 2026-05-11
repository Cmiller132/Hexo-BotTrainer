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

use super::board::{Board, MoveError};
use super::coord::HexCoord;
use super::rules::is_legal_placement;
use super::win::is_winning_placement;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

/// Flat history record for encoders and replay samples.
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// Last completed turn, useful for debugging and optional features.
    last_turn: Option<MoveRecord>,
    /// Full single-placement history for encoding recent stones.
    placement_history: Vec<PlacementRecord>,
    /// Deterministic hash of the public state used by MCTS nodes.
    ///
    /// This is named zobrist for the intended role, but currently uses a simple
    /// deterministic hash over state fields. A true incremental Zobrist table
    /// can replace `refresh_hash` later without changing callers.
    zobrist_hash: u64,
}

/// Summary returned after applying one placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl Default for HexoState {
    fn default() -> Self {
        Self::new()
    }
}

impl HexoState {
    /// Create the initial empty game state.
    pub fn new() -> Self {
        let mut state = Self {
            board: Board::new(),
            current_player: Player::Player0,
            phase: TurnPhase::Opening,
            placements_made: 0,
            terminal: None,
            last_turn: None,
            placement_history: Vec::new(),
            zobrist_hash: 0,
        };
        state.refresh_hash();
        state
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

    /// Alias for `terminal`, used by self-play code.
    pub fn outcome(&self) -> Option<GameOutcome> {
        self.terminal
    }

    /// True once no more moves should be generated.
    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// Last completed logical turn.
    pub fn last_turn(&self) -> Option<&MoveRecord> {
        self.last_turn.as_ref()
    }

    /// Complete single-placement history.
    pub fn placement_history(&self) -> &[PlacementRecord] {
        &self.placement_history
    }

    /// State hash used to label MCTS nodes.
    pub fn zobrist_hash(&self) -> u64 {
        self.zobrist_hash
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

    /// Recompute the current state hash from scratch.
    fn refresh_hash(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.current_player.hash(&mut hasher);
        self.phase.hash(&mut hasher);
        self.placements_made.hash(&mut hasher);
        for coord in self.board.occupied_cells() {
            coord.hash(&mut hasher);
            self.board.get(*coord).hash(&mut hasher);
        }
        self.zobrist_hash = hasher.finish();
    }
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
    state.board.place(placement.coord, player)?;
    state.placements_made += 1;
    state.push_history(player, placement.coord, phase_before);

    // Hexo wins immediately after any single placement, including the first
    // stone of a normal two-stone turn.
    if is_winning_placement(&state.board, placement.coord, player) {
        let outcome = GameOutcome {
            winner: player,
            placements: state.placements_made,
        };
        state.terminal = Some(outcome);
        state.refresh_hash();
        return Ok(ApplyResult {
            placed: placement.coord,
            player,
            phase_before,
            phase_after: state.phase,
            outcome: Some(outcome),
        });
    }

    match phase_before {
        TurnPhase::Opening => {
            // Opening is a special one-stone turn by Player 0. After it,
            // Player 1 starts the first normal two-stone turn.
            state.last_turn = Some(MoveRecord {
                player,
                placements: vec![placement.coord],
            });
            state.current_player = Player::Player1;
            state.phase = TurnPhase::FirstStone;
        }
        TurnPhase::FirstStone => {
            // The same player remains to place the second stone.
            state.last_turn = Some(MoveRecord {
                player,
                placements: vec![placement.coord],
            });
            state.phase = TurnPhase::SecondStone {
                first: placement.coord,
            };
        }
        TurnPhase::SecondStone { first } => {
            // A normal two-stone turn is complete, so control passes.
            state.last_turn = Some(MoveRecord {
                player,
                placements: vec![first, placement.coord],
            });
            state.current_player = player.other();
            state.phase = TurnPhase::FirstStone;
        }
    }

    state.refresh_hash();
    Ok(ApplyResult {
        placed: placement.coord,
        player,
        phase_before,
        phase_after: state.phase,
        outcome: None,
    })
}
