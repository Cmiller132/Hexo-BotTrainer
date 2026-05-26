//! Hexo rule engine.
//!
//! This crate owns the authoritative game state and state transitions. Model,
//! search, and sample code live outside this crate so the rules layer stays
//! small, deterministic, and easy to audit.

pub mod board;
pub mod coord;
pub mod error;
pub mod legal;
pub mod rules;
pub mod snapshot;
pub mod state;
pub mod tactics;

#[cfg(feature = "python")]
pub mod pybridge;

pub use board::{Board, BoardDelta, Stone};
pub use coord::{hex_distance, HexCoord};
pub use error::{MoveError, StateLoadError};
pub use legal::{
    pack_coord, unpack_coord, LegalMoveDelta, LegalMoveStore, PackedCoord, LEGAL_RADIUS,
};
pub use rules::is_legal_placement;
pub use snapshot::StateSnapshot;
pub use state::{
    apply_placement, load_state, ApplyDelta, ApplyResult, GameOutcome, HexoState, MoveRecord,
    Placement, PlacementRecord, Player, TurnPhase,
};
pub use tactics::{
    Axis, WindowEntry, WindowKey, WindowKeyList, WindowStore, WindowStoreDelta, WindowUpdate,
    WINDOWS_PER_PLACEMENT, WINDOW_LEN,
};
