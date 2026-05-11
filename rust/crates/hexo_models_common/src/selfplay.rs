//! CPU self-play loop.
//!
//! This module glues game state, MCTS, and replay sampling together. It is
//! generic over `StateEvaluator`, so it can run with `UniformEvaluator` now and
//! a Python-backed neural evaluator later.

use crate::mcts::{run_mcts, MctsConfig, SearchError, StateEvaluator};
use crate::sample::ReplaySample;
use hexo_engine::{apply_placement, GameOutcome, HexoState, MoveError, Placement, Player};

/// Game-level self-play limits.
#[derive(Clone, Debug)]
pub struct SelfplayConfig {
    /// Stop a non-terminal game after this many placements.
    pub max_placements: u32,
    /// Encoder crop size for generated replay samples.
    pub crop_size: usize,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        Self {
            max_placements: 300,
            crop_size: 31,
        }
    }
}

/// Complete self-play result.
#[derive(Clone, Debug)]
pub struct SelfplayGame {
    /// Placement-level replay samples.
    pub samples: Vec<ReplaySample>,
    /// Winner if the game ended naturally before the max-placement cap.
    pub outcome: Option<GameOutcome>,
    /// Number of stones placed.
    pub placements_made: u32,
}

/// Errors that can interrupt self-play.
#[derive(Debug)]
pub enum SelfplayError {
    /// Search failed, usually because no legal action existed.
    Search(SearchError),
    /// MCTS selected an action rejected by the game rules.
    IllegalMove(MoveError),
}

impl From<SearchError> for SelfplayError {
    fn from(value: SearchError) -> Self {
        Self::Search(value)
    }
}

impl From<MoveError> for SelfplayError {
    fn from(value: MoveError) -> Self {
        Self::IllegalMove(value)
    }
}

/// Play one game using MCTS for every single placement.
///
/// The loop records a replay sample before applying each selected action. Once
/// the game ends, final value targets are attached to every pending sample.
pub fn play_selfplay_game<E>(
    game_config: &SelfplayConfig,
    mcts_config: &MctsConfig,
    evaluator: &mut E,
) -> Result<SelfplayGame, SelfplayError>
where
    E: StateEvaluator,
{
    let mut state = HexoState::new();
    let mut samples = Vec::new();

    while state.terminal().is_none() && state.placements_made() < game_config.max_placements {
        // Search and sample at the current autoregressive decision point.
        let search = run_mcts(&state, evaluator, mcts_config)?;
        samples.push(ReplaySample::from_search(
            &state,
            &search.visit_policy,
            search.root_value,
            mcts_config.crop_size,
        ));

        // Apply only the selected single-stone action. The state machine handles
        // whether the next decision is same player or opponent.
        apply_placement(
            &mut state,
            Placement {
                coord: search.selected_action,
            },
        )?;
    }

    let outcome = state.terminal();
    attach_final_values(&mut samples, outcome.as_ref());

    Ok(SelfplayGame {
        samples,
        outcome,
        placements_made: state.placements_made(),
    })
}

/// Fill value targets from the final outcome.
pub fn attach_final_values(samples: &mut [ReplaySample], outcome: Option<&GameOutcome>) {
    for sample in samples {
        sample.value_target = outcome
            .map(|outcome| final_value_for_player(outcome, sample.current_player))
            .unwrap_or(0.0);
    }
}

/// Convert a terminal winner to a sample-perspective target.
fn final_value_for_player(outcome: &GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}
