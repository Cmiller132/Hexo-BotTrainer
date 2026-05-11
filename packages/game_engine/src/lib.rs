//! Hexo rule engine.
//!
//! This crate owns the authoritative game state and state transitions. Model,
//! search, replay, and Python bridge code live outside this crate so the rules
//! layer stays small, deterministic, and easy to audit.

pub mod game;

pub use game::{
    apply_placement, find_threats, hex_distance, is_legal_placement, legal_placements, ApplyResult,
    Axis, Board, GameOutcome, HexCoord, HexoState, MoveError, MoveRecord, Placement,
    PlacementRecord, Player, Stone, Threat, TurnPhase, WindowEntry, WindowId, WindowKey,
    WindowStore, WindowUpdate, AXES, LEGAL_FRONTIER_RADIUS, WINDOW_LEN,
};
