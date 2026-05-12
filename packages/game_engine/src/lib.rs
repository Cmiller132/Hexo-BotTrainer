//! Hexo rule engine.
//!
//! This crate owns the authoritative game state and state transitions. Model,
//! search, replay, and Python bridge code live outside this crate so the rules
//! layer stays small, deterministic, and easy to audit.

pub mod game;

pub use game::{
    apply_placement, find_all_threats, find_threats, hex_distance, is_legal_placement,
    legal_placements, load_state, ApplyResult, Axis, Board, GameOutcome, HexCoord, HexoState,
    MoveError, MoveRecord, Placement, PlacementRecord, Player, StateLoadError, StateSnapshot,
    Stone, Threat, TurnPhase, WindowEntry, WindowKey, WindowKeyList, WindowStore, WindowUpdate,
    LEGAL_FRONTIER_RADIUS, WINDOWS_PER_PLACEMENT, WINDOW_LEN,
};
