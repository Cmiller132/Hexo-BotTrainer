//! Autoregressive PUCT MCTS.
//!
//! Each edge is a single stone placement. A two-stone Hexo turn is therefore
//! searched as `FirstStone -> SecondStone -> opponent FirstStone`, which avoids
//! pair-action explosion and matches the training target shape.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use rand::Rng;

use crate::encode::{encode_state, legal_placements_in_crop};
use crate::mcts::evaluator::{Evaluation, StateEvaluator};
use crate::mcts::tree::{Edge, Node, NodeId, SearchTree};
use crate::position::SearchPosition;
use game_engine::{GameOutcome, HexCoord, HexoState, MoveError, Player};

/// Tunable search parameters for one root search.
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    /// Number of simulations to run from the root.
    pub visits: u32,
    /// Exploration strength in the PUCT formula.
    pub c_puct: f32,
    /// Encoder crop size used when evaluating leaf states.
    pub crop_size: usize,
    /// Root action sampling temperature. `0.0` means choose max visits.
    pub temperature: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            visits: 64,
            c_puct: 1.5,
            crop_size: 31,
            temperature: 0.0,
        }
    }
}

/// Output of one root search.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Action chosen for actual play.
    pub selected_action: HexCoord,
    /// Visit counts for each root action; this becomes the policy target.
    pub visit_policy: Vec<(HexCoord, u32)>,
    /// Root mean value from the root current-player perspective.
    pub root_value: f32,
}

/// Recoverable search failures.
#[derive(Debug)]
pub enum SearchError {
    /// Root state had no legal actions.
    NoLegalActions,
    /// A selected tree action failed game-rule validation.
    IllegalMove(MoveError),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::NoLegalActions => write!(f, "MCTS root has no legal actions"),
            SearchError::IllegalMove(error) => write!(f, "MCTS selected an illegal move: {error:?}"),
        }
    }
}

impl Error for SearchError {}

impl From<MoveError> for SearchError {
    fn from(value: MoveError) -> Self {
        Self::IllegalMove(value)
    }
}

/// Run MCTS from `root_state` and return a selected single-stone placement.
pub fn run_mcts<E>(
    root_state: &HexoState,
    evaluator: &mut E,
    config: &MctsConfig,
) -> Result<SearchResult, SearchError>
where
    E: StateEvaluator,
{
    // The root represents the exact state passed by the caller. It is expanded
    // once before simulations so root edges exist for policy extraction even if
    // visit count is tiny.
    let root = Node::new(
        state_hash(root_state),
        root_state.current_player(),
        root_state.phase(),
    );
    let mut tree = SearchTree::with_root(root);

    expand_node(&mut tree, 0, root_state, evaluator, config.crop_size)?;

    if tree.nodes[0].edges.is_empty() {
        return Err(SearchError::NoLegalActions);
    }

    // Each simulation starts from a fresh clone of the root state and walks the
    // tree by applying actions to that clone.
    for _ in 0..config.visits.max(1) {
        run_simulation(root_state, &mut tree, evaluator, config)?;
    }

    let root = &tree.nodes[0];
    let mut visit_policy: Vec<(HexCoord, u32)> = root
        .edges
        .iter()
        .map(|edge| (edge.action, edge.visits))
        .collect();
    visit_policy.sort_by(|a, b| compare_coord(a.0, b.0));

    let selected_action =
        select_root_action(root, config.temperature).ok_or(SearchError::NoLegalActions)?;

    Ok(SearchResult {
        selected_action,
        visit_policy,
        root_value: root.value(),
    })
}

/// Execute one selection/expansion/evaluation/backup simulation.
fn run_simulation<E>(
    root_state: &HexoState,
    tree: &mut SearchTree,
    evaluator: &mut E,
    config: &MctsConfig,
) -> Result<(), SearchError>
where
    E: StateEvaluator,
{
    let mut position = SearchPosition::from_root(root_state);
    let mut node_id = 0;
    let mut path: Vec<(NodeId, usize)> = Vec::new();

    let (leaf_player, leaf_value) = loop {
        // Terminal leaves get exact values instead of evaluator estimates.
        if let Some(outcome) = position.state().terminal() {
            let player = position.state().current_player();
            break (player, terminal_value_for_player(&outcome, player));
        }

        if !tree.nodes[node_id].expanded {
            let evaluation =
                expand_node(tree, node_id, position.state(), evaluator, config.crop_size)?;
            break (position.state().current_player(), evaluation.value);
        }

        if tree.nodes[node_id].edges.is_empty() {
            break (position.state().current_player(), 0.0);
        }

        // Selection: choose one edge with PUCT and apply its placement to the
        // local state clone.
        let edge_index = select_child(tree, node_id, config.c_puct);
        let action = tree.nodes[node_id].edges[edge_index].action;
        position.place_coord(action)?;
        path.push((node_id, edge_index));

        // If the child already exists, keep descending. Otherwise create it and
        // evaluate/expand that new leaf.
        if let Some(child_id) = tree.nodes[node_id].edges[edge_index].child {
            node_id = child_id;
            continue;
        }

        let child = Node::new(
            state_hash(position.state()),
            position.state().current_player(),
            position.state().phase(),
        );
        let child_id = tree.add_node(child);
        tree.nodes[node_id].edges[edge_index].child = Some(child_id);
        node_id = child_id;

        if let Some(outcome) = position.state().terminal() {
            tree.nodes[node_id].expanded = true;
            break (
                position.state().current_player(),
                terminal_value_for_player(&outcome, position.state().current_player()),
            );
        }

        let evaluation = expand_node(tree, node_id, position.state(), evaluator, config.crop_size)?;
        break (position.state().current_player(), evaluation.value);
    };

    backup(tree, node_id, &path, leaf_player, leaf_value);
    Ok(())
}

/// Expand a node by generating legal placements and evaluator priors.
fn expand_node<E>(
    tree: &mut SearchTree,
    node_id: NodeId,
    state: &HexoState,
    evaluator: &mut E,
    crop_size: usize,
) -> Result<Evaluation, SearchError>
where
    E: StateEvaluator,
{
    if state.terminal().is_some() {
        tree.nodes[node_id].expanded = true;
        tree.nodes[node_id].edges.clear();
        return Ok(Evaluation {
            priors: Vec::new(),
            value: 0.0,
        });
    }

    let encoded = encode_state(state, crop_size);
    let mut legal = Vec::new();
    legal_placements_in_crop(state, &encoded, &mut legal);

    if legal.is_empty() {
        tree.nodes[node_id].expanded = true;
        tree.nodes[node_id].edges.clear();
        return Ok(Evaluation {
            priors: Vec::new(),
            value: 0.0,
        });
    }

    legal.sort_by(|a, b| compare_coord(*a, *b));
    legal.dedup();

    // Encoding is only needed at expansion time. Legal actions are filtered to
    // the same crop so search, policy priors, and replay targets agree.
    let evaluation = evaluator.evaluate_state(state, &encoded, &legal);

    let mut edges = Vec::with_capacity(legal.len());
    for action in legal {
        let prior = evaluation
            .priors
            .iter()
            .find(|candidate| candidate.action == action)
            .map(|candidate| candidate.prior)
            .unwrap_or(0.0);
        edges.push(Edge::new(action, prior));
    }
    normalize_edge_priors(&mut edges);

    tree.nodes[node_id].edges = edges;
    tree.nodes[node_id].expanded = true;

    Ok(evaluation)
}

/// Choose the child edge with the highest PUCT score.
fn select_child(tree: &SearchTree, node_id: NodeId, c_puct: f32) -> usize {
    let node = &tree.nodes[node_id];
    let parent_visits = node.visits.max(1) as f32;
    let exploration_scale = parent_visits.sqrt();

    node.edges
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a_score = puct_score(a, c_puct, exploration_scale);
            let b_score = puct_score(b, c_puct, exploration_scale);
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.visits.cmp(&a.visits))
                .then_with(|| compare_coord(b.action, a.action))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// PUCT = mean value + prior-weighted exploration bonus.
fn puct_score(edge: &Edge, c_puct: f32, exploration_scale: f32) -> f32 {
    edge.q_value() + c_puct * edge.prior * exploration_scale / (1.0 + edge.visits as f32)
}

/// Back up a leaf value through the selected path.
///
/// Important Hexo detail: values are stored from each node's `player_to_act`
/// perspective. We do not flip signs per stone placement. We only flip when the
/// stored value's player differs from the node's player. This naturally handles
/// `FirstStone -> SecondStone` as same-player and `SecondStone -> opponent` as
/// opposite-player.
fn backup(
    tree: &mut SearchTree,
    leaf_node: NodeId,
    path: &[(NodeId, usize)],
    leaf_player: Player,
    leaf_value: f32,
) {
    // Update edge statistics from their parent node perspective.
    for &(node_id, edge_index) in path {
        let player = tree.nodes[node_id].player_to_act;
        let value = value_for_player(leaf_value, leaf_player, player);
        let edge = &mut tree.nodes[node_id].edges[edge_index];
        edge.visits += 1;
        edge.value_sum += value;
    }

    // Update every internal node on the selected path.
    for &(node_id, _) in path {
        let player = tree.nodes[node_id].player_to_act;
        let value = value_for_player(leaf_value, leaf_player, player);
        let node = &mut tree.nodes[node_id];
        node.visits += 1;
        node.value_sum += value;
    }

    // Update the leaf node itself so its value is meaningful if revisited.
    let player = tree.nodes[leaf_node].player_to_act;
    let value = value_for_player(leaf_value, leaf_player, player);
    let node = &mut tree.nodes[leaf_node];
    node.visits += 1;
    node.value_sum += value;
}

/// Convert a value from one player's perspective to another's.
fn value_for_player(value: f32, value_player: Player, target_player: Player) -> f32 {
    if value_player == target_player {
        value
    } else {
        -value
    }
}

/// Exact terminal value from `player` perspective.
fn terminal_value_for_player(outcome: &GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}

/// Ensure edge priors form a valid probability distribution.
fn normalize_edge_priors(edges: &mut [Edge]) {
    let total: f32 = edges.iter().map(|edge| edge.prior.max(0.0)).sum();

    if total <= f32::EPSILON || !total.is_finite() {
        let prior = 1.0 / edges.len().max(1) as f32;
        for edge in edges {
            edge.prior = prior;
        }
        return;
    }

    for edge in edges {
        edge.prior = edge.prior.max(0.0) / total;
    }
}

/// Select the final root action from visit counts.
fn select_root_action(root: &Node, temperature: f32) -> Option<HexCoord> {
    if temperature <= f32::EPSILON {
        return root
            .edges
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| compare_coord(b.action, a.action))
            })
            .map(|edge| edge.action);
    }

    let inv_temp = 1.0 / temperature.max(1.0e-3);
    let weights: Vec<f32> = root
        .edges
        .iter()
        .map(|edge| (edge.visits as f32).max(0.0).powf(inv_temp))
        .collect();
    let total: f32 = weights.iter().sum();

    if total <= f32::EPSILON || !total.is_finite() {
        return root.edges.first().map(|edge| edge.action);
    }

    let mut threshold = rand::thread_rng().gen_range(0.0..total);
    for (edge, weight) in root.edges.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Some(edge.action);
        }
    }

    root.edges.last().map(|edge| edge.action)
}

/// Current state hash accessor.
fn state_hash(state: &HexoState) -> u64 {
    state.zobrist_hash()
}

/// Deterministic coordinate ordering for stable policies and tie-breaks.
fn compare_coord(a: HexCoord, b: HexCoord) -> Ordering {
    a.q.cmp(&b.q).then_with(|| a.r.cmp(&b.r))
}
