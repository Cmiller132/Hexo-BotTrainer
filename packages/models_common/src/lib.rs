//! Shared model/search utilities for Hexo models.
//!
//! This crate depends on `game_engine` for authoritative rules and state
//! transitions. During the runner redesign it keeps the stable pieces:
//! neural-network encoding, MCTS, and a minimal optional Python extension.
//! Replay and self-play modules are placeholders for the next explicit
//! contract.

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
pub use sample::{ReplayBatchDraft, REPLAY_SCHEMA_DRAFT};
pub use selfplay::{SelfplayCycleDraft, SelfplayPlan};
