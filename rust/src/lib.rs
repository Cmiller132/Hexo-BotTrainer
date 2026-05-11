//! Public entry point for the Hexo RL Rust crate.
//!
//! The crate is intentionally small and single-crate for the prototype. The
//! important boundary is conceptual:
//! - `game` owns rules and state transitions.
//! - `mcts` owns search over single stone placements.
//! - `encode`, `sample`, and `selfplay` connect the game/search core to RL data.

pub mod encode;
pub mod game;
pub mod mcts;
pub mod sample;
pub mod selfplay;

#[cfg(feature = "python")]
pub mod pybridge;

pub use game::{
    apply_placement, legal_placements, Board, GameOutcome, HexCoord, HexoState, MoveError,
    Placement, Player, Stone, TurnPhase,
};
