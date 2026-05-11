//! Shared model/search utilities for Hexo models.
//!
//! This crate depends on `game_engine` for authoritative rules and state
//! transitions. It owns neural-network encoding, MCTS, replay samples,
//! self-play orchestration, and optional Python bindings.

pub mod encode;
pub mod mcts;
pub mod position;
pub mod sample;
pub mod selfplay;

#[cfg(feature = "python")]
pub mod pybridge;

pub use encode::{
    encode_state, legal_placements_in_crop, planes, EncodedState, DEFAULT_CROP_SIZE, PLANE_COUNT,
};
pub use mcts::{
    run_mcts, Evaluation, Evaluator, MctsConfig, NetworkOutput, PolicyPrior, SearchError,
    SearchResult, StateEvaluator, UniformEvaluator,
};
pub use position::SearchPosition;
pub use sample::{normalize_visit_policy, ReplaySample, TurnPhaseLabel, RULES_VERSION};
pub use selfplay::{
    attach_final_values, play_selfplay_game, SelfplayConfig, SelfplayError, SelfplayGame,
};
