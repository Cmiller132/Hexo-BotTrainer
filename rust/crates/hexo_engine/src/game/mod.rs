//! Hexo rule engine.
//!
//! This module keeps all game-specific logic together: axial coordinates,
//! sparse board storage, turn phases, legality checks, and incremental window
//! tracking. Search and training code should call this module instead of
//! duplicating game rules.

pub mod board;
pub mod coord;
pub mod rules;
pub mod state;
pub mod windows;

pub use board::{Board, MoveError, Stone, LEGAL_FRONTIER_RADIUS};
pub use coord::{hex_distance, HexCoord, AXES};
pub use rules::{is_legal_placement, legal_placements};
pub use state::{
    apply_placement, ApplyResult, GameOutcome, HexoState, MoveRecord, Placement, PlacementRecord,
    Player, TurnPhase,
};
pub use windows::{
    find_threats, Axis, Threat, WindowEntry, WindowId, WindowKey, WindowStore, WindowUpdate,
    WINDOW_LEN,
};
