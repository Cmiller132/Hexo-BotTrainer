//! Replay sample structures.
//!
//! Self-play records one sample per placement decision, not one sample per
//! two-stone turn. That keeps the policy target aligned with the MCTS action
//! space: a single crop cell.

use crate::encode::{encode_state, legal_placements_in_crop, EncodedState, DEFAULT_CROP_SIZE};
use game_engine::{HexCoord, HexoState, Player, TurnPhase};
use serde::{Deserialize, Serialize};

/// Rules schema version written into replay data.
pub const RULES_VERSION: u32 = 1;

/// Serializable phase label that omits the first-stone coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnPhaseLabel {
    Opening,
    FirstStone,
    SecondStone,
}

impl From<TurnPhase> for TurnPhaseLabel {
    fn from(value: TurnPhase) -> Self {
        match value {
            TurnPhase::Opening => Self::Opening,
            TurnPhase::FirstStone => Self::FirstStone,
            TurnPhase::SecondStone { .. } => Self::SecondStone,
        }
    }
}

/// One training record for one placement decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplaySample {
    /// Game identifier, currently always `"hexo"`.
    pub game: String,
    /// Rule/schema version for future compatibility.
    pub rules_version: u32,
    /// Encoded neural-network input before the action.
    pub state: EncodedState,
    /// Player whose perspective the state/value target use.
    pub current_player: Player,
    /// Phase label before the action.
    pub phase: TurnPhaseLabel,
    /// Legal actions before the action.
    pub legal_actions: Vec<HexCoord>,
    /// Normalized MCTS visit distribution.
    pub policy_target: Vec<(HexCoord, f32)>,
    /// Final game outcome from `current_player` perspective.
    pub value_target: f32,
    /// MCTS root value estimate before final outcome is known.
    pub root_value: f32,
    /// Number of stones already on board before this decision.
    pub placements_made: u32,
}

impl ReplaySample {
    /// Build a pending sample from a search result.
    ///
    /// `value_target` is filled later after the self-play game ends.
    pub fn from_search(
        state: &HexoState,
        visit_policy: &[(HexCoord, u32)],
        root_value: f32,
        crop_size: usize,
    ) -> Self {
        let encoded = encode_state(state, crop_size);
        let mut legal_actions = Vec::new();
        legal_placements_in_crop(state, &encoded, &mut legal_actions);
        let cropped_visit_policy: Vec<_> = visit_policy
            .iter()
            .copied()
            .filter(|(coord, _)| encoded.index_of_coord(*coord).is_some())
            .collect();

        Self {
            game: "hexo".to_owned(),
            rules_version: RULES_VERSION,
            state: encoded,
            current_player: state.current_player(),
            phase: TurnPhaseLabel::from(state.phase()),
            legal_actions,
            policy_target: normalize_visit_policy(&cropped_visit_policy),
            value_target: 0.0,
            root_value,
            placements_made: state.placements_made(),
        }
    }

    /// Convenience constructor using the development crop size.
    pub fn with_default_crop(
        state: &HexoState,
        visit_policy: &[(HexCoord, u32)],
        root_value: f32,
    ) -> Self {
        Self::from_search(state, visit_policy, root_value, DEFAULT_CROP_SIZE)
    }
}

/// Convert root visit counts to probabilities.
pub fn normalize_visit_policy(visit_policy: &[(HexCoord, u32)]) -> Vec<(HexCoord, f32)> {
    let total: u32 = visit_policy.iter().map(|(_, visits)| *visits).sum();
    if total == 0 {
        return Vec::new();
    }

    visit_policy
        .iter()
        .filter_map(|(coord, visits)| {
            if *visits == 0 {
                None
            } else {
                Some((*coord, *visits as f32 / total as f32))
            }
        })
        .collect()
}
