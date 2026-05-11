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
use super::windows::WindowUpdate;
use serde::{de, Deserialize, Deserializer, Serialize};

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
#[derive(Clone, Debug, Serialize)]
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
    /// Deterministic Zobrist-style hash of the public state used by MCTS nodes.
    ///
    /// The board is unbounded, so keys are generated on demand from fixed
    /// splitmix64 seeds instead of being stored in a finite table.
    zobrist_hash: u64,
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
    /// Window ids changed by this placement plus any threat/win ids.
    pub window_update: WindowUpdate,
}

const ZOBRIST_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const ZOBRIST_TAG_STONE: u64 = 0x4845_584f_5354_4f4e;
const ZOBRIST_TAG_CURRENT_PLAYER: u64 = 0x4845_584f_4355_5252;
const ZOBRIST_TAG_PHASE: u64 = 0x4845_584f_5048_4153;
const ZOBRIST_TAG_TERMINAL: u64 = 0x4845_584f_5445_524d;
const ZOBRIST_TAG_PLACEMENT_COUNT: u64 = 0x4845_584f_434f_554e;
const ZOBRIST_TAG_HISTORY: u64 = 0x4845_584f_4849_5354;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn zobrist_key(tag: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut value = ZOBRIST_SEED ^ tag;
    value = splitmix64(value ^ a);
    value = splitmix64(value ^ b.rotate_left(21));
    splitmix64(value ^ c.rotate_left(42))
}

fn player_component(player: Player) -> u64 {
    player.index() as u64
}

fn coord_component(coord: HexCoord) -> u64 {
    ((coord.q as u16 as u64) << 16) | coord.r as u16 as u64
}

fn phase_component(phase: TurnPhase) -> u64 {
    match phase {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { first } => 2 ^ coord_component(first).rotate_left(17),
    }
}

fn zobrist_stone_key(player: Player, coord: HexCoord) -> u64 {
    zobrist_key(
        ZOBRIST_TAG_STONE,
        player_component(player),
        coord_component(coord),
        0,
    )
}

fn zobrist_current_player_key(player: Player) -> u64 {
    zobrist_key(ZOBRIST_TAG_CURRENT_PLAYER, player_component(player), 0, 0)
}

fn zobrist_phase_key(phase: TurnPhase) -> u64 {
    zobrist_key(ZOBRIST_TAG_PHASE, phase_component(phase), 0, 0)
}

fn zobrist_terminal_key(terminal: Option<GameOutcome>) -> u64 {
    match terminal {
        Some(outcome) => zobrist_key(
            ZOBRIST_TAG_TERMINAL,
            player_component(outcome.winner),
            outcome.placements as u64,
            0,
        ),
        None => 0,
    }
}

fn zobrist_placement_count_key(placements_made: u32) -> u64 {
    zobrist_key(ZOBRIST_TAG_PLACEMENT_COUNT, placements_made as u64, 0, 0)
}

fn zobrist_history_key(record: PlacementRecord) -> u64 {
    zobrist_key(
        ZOBRIST_TAG_HISTORY,
        record.placement_index as u64,
        player_component(record.player) ^ coord_component(record.coord).rotate_left(7),
        phase_component(record.phase),
    )
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
        state.reset_zobrist_hash();
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

    /// Most recent logical turn progress.
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

    fn record_turn_progress(&mut self, player: Player, coord: HexCoord, phase: TurnPhase) {
        let placements = match phase {
            TurnPhase::Opening | TurnPhase::FirstStone => vec![coord],
            TurnPhase::SecondStone { first } => vec![first, coord],
        };
        self.last_turn = Some(MoveRecord { player, placements });
    }

    fn zobrist_metadata_hash(&self) -> u64 {
        zobrist_current_player_key(self.current_player)
            ^ zobrist_phase_key(self.phase)
            ^ zobrist_terminal_key(self.terminal)
            ^ zobrist_placement_count_key(self.placements_made)
    }

    /// Recompute the current state hash from scratch.
    fn recompute_zobrist_hash(&self) -> u64 {
        let mut hash = self.zobrist_metadata_hash();
        for coord in self.board.occupied_cells() {
            if let Some(stone) = self.board.get(*coord) {
                hash ^= zobrist_stone_key(stone, *coord);
            }
        }
        for record in &self.placement_history {
            hash ^= zobrist_history_key(*record);
        }
        hash
    }

    fn reset_zobrist_hash(&mut self) {
        self.zobrist_hash = self.recompute_zobrist_hash();
    }

    fn debug_assert_zobrist_consistent(&self) {
        debug_assert_eq!(self.zobrist_hash, self.recompute_zobrist_hash());
    }
}

#[derive(Deserialize)]
struct RawHexoState {
    board: Board,
    current_player: Player,
    phase: TurnPhase,
    placements_made: u32,
    terminal: Option<GameOutcome>,
    last_turn: Option<MoveRecord>,
    placement_history: Vec<PlacementRecord>,
    #[serde(default, rename = "zobrist_hash")]
    _zobrist_hash: Option<u64>,
}

impl<'de> Deserialize<'de> for HexoState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawHexoState::deserialize(deserializer)?;
        deserialize_validated_state(raw).map_err(de::Error::custom)
    }
}

fn deserialize_validated_state(raw: RawHexoState) -> Result<HexoState, String> {
    let rebuilt = replay_history(&raw.placement_history)?;
    validate_deserialized_state(&raw, &rebuilt)?;
    Ok(rebuilt)
}

fn replay_history(history: &[PlacementRecord]) -> Result<HexoState, String> {
    let mut state = HexoState::new();

    for (index, record) in history.iter().copied().enumerate() {
        let expected_player = state.current_player;
        let expected_phase = state.phase;
        let expected_placement_index = state.placements_made + 1;

        if record.player != expected_player {
            return Err(format!(
                "placement history record {index} has player {:?}, expected {:?}",
                record.player, expected_player
            ));
        }
        if record.phase != expected_phase {
            return Err(format!(
                "placement history record {index} has phase {:?}, expected {:?}",
                record.phase, expected_phase
            ));
        }
        if record.placement_index != expected_placement_index {
            return Err(format!(
                "placement history record {index} has placement_index {}, expected {}",
                record.placement_index, expected_placement_index
            ));
        }

        apply_placement(
            &mut state,
            Placement {
                coord: record.coord,
            },
        )
        .map_err(|error| format!("placement history record {index} is illegal: {error}"))?;
    }

    Ok(state)
}

fn validate_deserialized_state(raw: &RawHexoState, rebuilt: &HexoState) -> Result<(), String> {
    if !boards_match(&raw.board, &rebuilt.board) {
        return Err("serialized board does not match placement history replay".to_owned());
    }
    if raw.current_player != rebuilt.current_player {
        return Err(format!(
            "serialized current_player {:?} does not match replayed {:?}",
            raw.current_player, rebuilt.current_player
        ));
    }
    if raw.phase != rebuilt.phase {
        return Err(format!(
            "serialized phase {:?} does not match replayed {:?}",
            raw.phase, rebuilt.phase
        ));
    }
    if raw.placements_made != rebuilt.placements_made {
        return Err(format!(
            "serialized placements_made {} does not match replayed {}",
            raw.placements_made, rebuilt.placements_made
        ));
    }
    if raw.terminal != rebuilt.terminal {
        return Err(format!(
            "serialized terminal {:?} does not match replayed {:?}",
            raw.terminal, rebuilt.terminal
        ));
    }
    if raw.last_turn != rebuilt.last_turn {
        return Err(format!(
            "serialized last_turn {:?} does not match replayed {:?}",
            raw.last_turn, rebuilt.last_turn
        ));
    }
    if raw.placement_history != rebuilt.placement_history {
        return Err("serialized placement_history does not match replayed history".to_owned());
    }

    Ok(())
}

fn boards_match(left: &Board, right: &Board) -> bool {
    left.occupied_cells() == right.occupied_cells()
        && left
            .occupied_cells()
            .iter()
            .all(|coord| left.get(*coord) == right.get(*coord))
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
    let old_metadata_hash = state.zobrist_metadata_hash();
    let window_update = state.board.place(placement.coord, player)?;
    state.zobrist_hash ^= old_metadata_hash;
    state.zobrist_hash ^= zobrist_stone_key(player, placement.coord);
    state.placements_made += 1;
    state.push_history(player, placement.coord, phase_before);
    let history_record = *state
        .placement_history
        .last()
        .expect("history is pushed immediately before hashing");
    state.zobrist_hash ^= zobrist_history_key(history_record);
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
        state.zobrist_hash ^= state.zobrist_metadata_hash();
        state.debug_assert_zobrist_consistent();
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

    state.zobrist_hash ^= state.zobrist_metadata_hash();
    state.debug_assert_zobrist_consistent();
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
    fn deserialization_replays_history_and_recomputes_zobrist_hash() {
        let state = sample_state();
        let mut value = serde_json::to_value(&state).unwrap();
        value["zobrist_hash"] = json!(0);

        let decoded: HexoState = serde_json::from_value(value).unwrap();

        assert_same_public_state(&decoded, &state);
        assert_eq!(decoded.zobrist_hash(), state.zobrist_hash());
    }

    #[test]
    fn deserialization_rejects_placement_count_that_disagrees_with_replay() {
        let state = sample_state();
        let mut value = serde_json::to_value(&state).unwrap();
        value["placements_made"] = json!(99);

        let error = serde_json::from_value::<HexoState>(value).unwrap_err();

        assert!(error.to_string().contains("placements_made"));
    }

    #[test]
    fn deserialization_rejects_history_that_does_not_match_phase_machine() {
        let state = sample_state();
        let mut value = serde_json::to_value(&state).unwrap();
        value["placement_history"][0]["phase"] = json!("FirstStone");

        let error = serde_json::from_value::<HexoState>(value).unwrap_err();

        assert!(error.to_string().contains("phase"));
    }
}
