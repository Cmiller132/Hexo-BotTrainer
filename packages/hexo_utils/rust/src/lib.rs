//! Shared model/search utilities for Hexo models.
//!
//! This crate depends on `hexo_engine` for authoritative rules and state
//! transitions. It keeps the shared pieces: neural-network encoding, MCTS, and
//! replay contract helpers.

pub mod encoder;
pub mod mcts;
pub mod position;
pub mod replay;

#[cfg(feature = "python")]
pub mod pybridge;

pub use encoder::{
    encode_state, legal_placements_in_crop, planes, EncodedState, DEFAULT_CROP_SIZE, PLANE_COUNT,
};
pub use mcts::{
    run_mcts, Evaluation, Evaluator, MctsConfig, NetworkOutput, PolicyPrior, SearchError,
    SearchResult, StateEvaluator, UniformEvaluator,
};
pub use position::SearchPosition;
pub use replay::{ReplayBatchDraft, REPLAY_SCHEMA_DRAFT};
