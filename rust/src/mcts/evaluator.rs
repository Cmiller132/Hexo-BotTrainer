//! Evaluator abstraction used by MCTS.
//!
//! Rust search only needs two model outputs:
//! - policy logits over crop cells
//! - scalar value from the current player's perspective
//!
//! The trait boundary keeps PyTorch/model details out of the game and search
//! code. `UniformEvaluator` is enough to exercise MCTS before Python inference
//! is connected.

use crate::encode::EncodedState;
use crate::game::{HexCoord, HexoState};

/// Raw neural-network-style output for one encoded state.
#[derive(Clone, Debug)]
pub struct NetworkOutput {
    /// Flat policy logits, indexed the same way as `EncodedState`.
    pub policy_logits: Vec<f32>,
    /// Value in [-1, 1] from the state's current player perspective.
    pub value: f32,
}

/// Normalized prior probability for one legal placement.
#[derive(Clone, Copy, Debug)]
pub struct PolicyPrior {
    /// Legal single-stone action.
    pub action: HexCoord,
    /// Probability mass assigned to that action.
    pub prior: f32,
}

/// Search-ready evaluation for one state.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Priors only for legal actions.
    pub priors: Vec<PolicyPrior>,
    /// Leaf value from current player perspective.
    pub value: f32,
}

/// Batch evaluator interface intended for neural inference.
pub trait Evaluator {
    /// Evaluate encoded states and return one output per input state.
    fn evaluate_batch(&mut self, states: &[EncodedState]) -> Vec<NetworkOutput>;
}

/// State-aware evaluator interface consumed directly by MCTS.
///
/// Implementors may use the full `HexoState` and legal action list for
/// handcrafted evaluators. The blanket impl below adapts neural-style
/// `Evaluator`s by masking logits to legal actions.
pub trait StateEvaluator {
    fn evaluate_state(
        &mut self,
        state: &HexoState,
        encoded: &EncodedState,
        legal_actions: &[HexCoord],
    ) -> Evaluation;
}

impl<T> StateEvaluator for T
where
    T: Evaluator,
{
    /// Convert flat crop logits into priors over the provided legal actions.
    fn evaluate_state(
        &mut self,
        _state: &HexoState,
        encoded: &EncodedState,
        legal_actions: &[HexCoord],
    ) -> Evaluation {
        let outputs = self.evaluate_batch(std::slice::from_ref(encoded));
        let output = outputs.into_iter().next().unwrap_or(NetworkOutput {
            policy_logits: Vec::new(),
            value: 0.0,
        });

        Evaluation {
            priors: legal_policy_from_logits(encoded, legal_actions, &output.policy_logits),
            value: output.value.clamp(-1.0, 1.0),
        }
    }
}

/// Baseline evaluator: every legal move is equally plausible and value is zero.
#[derive(Clone, Debug, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate_batch(&mut self, states: &[EncodedState]) -> Vec<NetworkOutput> {
        states
            .iter()
            .map(|state| NetworkOutput {
                policy_logits: vec![0.0; state.cell_count()],
                value: 0.0,
            })
            .collect()
    }
}

/// Convert model logits into a softmax over legal coordinates.
///
/// Legal moves outside the current crop receive logit 0.0 for now. The encoder
/// crop is expected to be large enough in normal configs; this fallback keeps
/// the prototype robust while that assumption is being refined.
fn legal_policy_from_logits(
    encoded: &EncodedState,
    legal_actions: &[HexCoord],
    logits: &[f32],
) -> Vec<PolicyPrior> {
    if legal_actions.is_empty() {
        return Vec::new();
    }

    let mut scored = Vec::with_capacity(legal_actions.len());
    let mut max_logit = f32::NEG_INFINITY;

    for &action in legal_actions {
        let logit = encoded
            .index_of_coord(action)
            .and_then(|index| logits.get(index).copied())
            .unwrap_or(0.0);
        max_logit = max_logit.max(logit);
        scored.push((action, logit));
    }

    let mut normalizer = 0.0;
    let mut priors = Vec::with_capacity(scored.len());

    for (action, logit) in scored {
        let prior = (logit - max_logit).exp();
        normalizer += prior;
        priors.push(PolicyPrior { action, prior });
    }

    if normalizer <= f32::EPSILON || !normalizer.is_finite() {
        let prior = 1.0 / legal_actions.len() as f32;
        return legal_actions
            .iter()
            .copied()
            .map(|action| PolicyPrior { action, prior })
            .collect();
    }

    for prior in &mut priors {
        prior.prior /= normalizer;
    }

    priors
}
