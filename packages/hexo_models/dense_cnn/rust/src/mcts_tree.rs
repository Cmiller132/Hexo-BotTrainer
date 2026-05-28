//! Dense CNN PUCT tree mechanics.
//!
//! The tree mirrors the usual KataGo-style ownership pattern at a smaller
//! scale: a search owns one root state, an arena of nodes, an exact hash table
//! for already-expanded states, and root visit accounting. Nodes intentionally store only statistics and outgoing edges;
//! traversal recreates leaf states from a root clone plus selected actions so
//! the implementation stays memory-light and easy to inspect.

use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use hexo_engine::{
    apply_placement, unpack_coord, GameOutcome, HexCoord, HexoState as RustHexoState, PackedCoord,
    Placement, Player,
};
use hexo_utils::StateHash;

use super::mcts_eval::{state_hash, RustEvaluation};
use super::state::move_error;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProgressiveWideningConfig {
    pub(crate) root_initial_actions: usize,
    pub(crate) child_initial_actions: usize,
    pub(crate) growth_interval: f32,
    pub(crate) growth_base: f32,
}

impl ProgressiveWideningConfig {
    pub(crate) fn new(
        root_initial_actions: usize,
        child_initial_actions: usize,
        growth_interval: f32,
        growth_base: f32,
    ) -> Option<Self> {
        let root_initial_actions = root_initial_actions.max(1);
        let child_initial_actions = child_initial_actions.max(1);
        let growth_interval = if growth_interval.is_finite() && growth_interval > 0.0 {
            growth_interval
        } else {
            256.0
        };
        let growth_base = if growth_base.is_finite() && growth_base > 1.0 {
            growth_base
        } else {
            1.3
        };
        Some(Self {
            root_initial_actions,
            child_initial_actions,
            growth_interval,
            growth_base,
        })
    }

    fn edge_limit(&self, visits: u32, total_actions: usize, is_root: bool) -> usize {
        let initial_actions = if is_root {
            self.root_initial_actions
        } else {
            self.child_initial_actions
        };
        if total_actions <= initial_actions {
            return total_actions;
        }
        // Chaslot et al.'s progressive unpruning keeps the k_init highest
        // heuristic children, then unprunes k children when parent visits pass
        // A * B^(k-k_init). The dense CNN policy prior is the heuristic.
        let visits = visits as f32;
        if visits < self.growth_interval {
            return initial_actions;
        }
        let extra = (visits / self.growth_interval)
            .log(self.growth_base)
            .floor()
            .max(0.0) as usize;
        (initial_actions + extra).min(total_actions)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RustEdge {
    pub(crate) action_id: PackedCoord,
    pub(crate) action: HexCoord,
    pub(crate) prior: f32,
    pub(crate) visits: u32,
    pub(crate) value_sum: f32,
    pub(crate) pending: u32,
    pub(crate) child: Option<usize>,
}

impl RustEdge {
    pub(crate) fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RustPriorCandidate {
    pub(crate) action_id: PackedCoord,
    pub(crate) prior: f32,
}

impl RustPriorCandidate {
    fn into_edge(self) -> RustEdge {
        RustEdge {
            action_id: self.action_id,
            action: unpack_coord(self.action_id),
            prior: self.prior,
            visits: 0,
            value_sum: 0.0,
            pending: 0,
            child: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RustNode {
    pub(crate) state_hash: StateHash,
    pub(crate) player: Player,
    pub(crate) eval_value: f32,
    pub(crate) total_legal_actions: usize,
    pub(crate) visits: u32,
    pub(crate) value_sum: f32,
    pub(crate) edges: Vec<RustEdge>,
    // KataGo-style staged children: keep legal policy candidates compact and
    // materialize an edge only when PUCT actually selects that move.
    pub(crate) unexpanded_priors: Vec<RustPriorCandidate>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RustSearchDiagnostics {
    pub(crate) node_count: usize,
    pub(crate) active_edge_count: usize,
    pub(crate) hidden_prior_count: usize,
    pub(crate) root_active_edges: usize,
    pub(crate) root_hidden_priors: usize,
    pub(crate) max_active_edges_per_node: usize,
    pub(crate) max_hidden_priors_per_node: usize,
    pub(crate) widened_edges_total: usize,
}

impl RustNode {
    pub(crate) fn value(&self) -> f32 {
        if self.visits == 0 {
            self.eval_value
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn hidden_action_count(&self) -> usize {
        self.total_legal_actions
            .saturating_sub(self.edges.len())
            .max(self.unexpanded_priors.len())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RustSearch {
    pub(crate) root_state: RustHexoState,
    pub(crate) root_hash: StateHash,
    pub(crate) nodes: Vec<RustNode>,
    pub(crate) node_table: HashMap<StateHash, usize>,
    pub(crate) target_visits: u32,
    pub(crate) completed_visits: u32,
    widening: Option<ProgressiveWideningConfig>,
    active_edge_count: usize,
    hidden_prior_count: usize,
    max_active_edges_per_node: usize,
    max_hidden_priors_per_node: usize,
    widened_edges_total: usize,
}

pub(crate) struct RustSelectedLeaf {
    pub(crate) path: Vec<(usize, usize)>,
    pub(crate) state: RustHexoState,
    pub(crate) state_hash: StateHash,
    pub(crate) parent_node: usize,
    pub(crate) edge_index: usize,
    pub(crate) terminal: Option<GameOutcome>,
    pub(crate) existing_node: Option<usize>,
}

pub(crate) struct RustLeaf {
    pub(crate) root_index: usize,
    pub(crate) parent_node: usize,
    pub(crate) edge_index: usize,
    pub(crate) path: Vec<(usize, usize)>,
    pub(crate) state: RustHexoState,
    pub(crate) state_hash: StateHash,
}

impl RustSearch {
    pub(crate) fn new(
        root_state: RustHexoState,
        evaluation: &RustEvaluation,
        target_visits: u32,
        widening: Option<ProgressiveWideningConfig>,
    ) -> Self {
        let root_hash = state_hash(&root_state);
        let root_node = node_from_evaluation(root_hash, &root_state, evaluation, widening, true);
        let root_hidden = root_node.hidden_action_count();
        let mut node_table = HashMap::new();
        node_table.insert(root_hash, 0);
        Self {
            root_state,
            root_hash,
            nodes: vec![root_node],
            node_table,
            target_visits,
            completed_visits: 0,
            widening,
            active_edge_count: 0,
            hidden_prior_count: root_hidden,
            max_active_edges_per_node: 0,
            max_hidden_priors_per_node: root_hidden,
            widened_edges_total: 0,
        }
    }

    pub(crate) fn root_edges_empty(&self) -> bool {
        self.nodes[0].edges.is_empty() && self.nodes[0].hidden_action_count() == 0
    }

    pub(crate) fn needs_visits(&self) -> bool {
        self.completed_visits < self.target_visits && !self.root_edges_empty()
    }

    pub(crate) fn remaining_visits(&self) -> u32 {
        self.target_visits.saturating_sub(self.completed_visits)
    }

    pub(crate) fn set_additional_visits(&mut self, visits: u32) {
        self.target_visits = self.completed_visits.saturating_add(visits.max(1));
    }

    pub(crate) fn root(&self) -> &RustNode {
        debug_assert_eq!(self.nodes[0].state_hash, self.root_hash);
        &self.nodes[0]
    }

    pub(crate) fn root_edge_visits(&self) -> Vec<(PackedCoord, u32)> {
        self.root()
            .edges
            .iter()
            .map(|edge| (edge.action_id, edge.visits))
            .collect()
    }

    pub(crate) fn add_node_from_eval(
        &mut self,
        state: &RustHexoState,
        hash: StateHash,
        evaluation: &RustEvaluation,
    ) -> usize {
        if let Some(existing) = self.node_table.get(&hash).copied() {
            return existing;
        }
        let id = self.nodes.len();
        let node = node_from_evaluation(hash, state, evaluation, self.widening, false);
        let hidden = node.hidden_action_count();
        self.hidden_prior_count += hidden;
        self.max_hidden_priors_per_node = self.max_hidden_priors_per_node.max(hidden);
        self.nodes.push(node);
        self.node_table.insert(hash, id);
        id
    }

    pub(crate) fn select_pending_leaf(
        &mut self,
        c_puct: f32,
    ) -> PyResult<Option<RustSelectedLeaf>> {
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;
        let mut current_hash = self.root_hash;

        loop {
            let Some(edge_index) = self.select_or_materialize_edge(node_id, c_puct, &state) else {
                let Some(parent_node) = last_parent else {
                    return Ok(None);
                };
                let edge_index = last_edge.expect("edge index exists with parent");
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node,
                    edge_index,
                    terminal: None,
                    existing_node: Some(node_id),
                }));
            };

            let edge = &self.nodes[node_id].edges[edge_index];
            if edge.pending > 0 && edge.child.is_none() {
                return Ok(None);
            }

            let action = edge.action;
            let child = edge.child;
            apply_placement(&mut state, Placement { coord: action }).map_err(move_error)?;
            current_hash = state_hash(&state);
            path.push((node_id, edge_index));
            last_parent = Some(node_id);
            last_edge = Some(edge_index);

            if let Some(child_id) = child {
                node_id = child_id;
                continue;
            }

            if let Some(child_id) = self.node_table.get(&current_hash).copied() {
                self.nodes[node_id].edges[edge_index].child = Some(child_id);
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node: node_id,
                    edge_index,
                    terminal: None,
                    existing_node: Some(child_id),
                }));
            }

            return Ok(Some(RustSelectedLeaf {
                path,
                state: state.clone(),
                state_hash: current_hash,
                parent_node: node_id,
                edge_index,
                terminal: state.terminal(),
                existing_node: None,
            }));
        }
    }

    fn eligible_unmaterialized_count(&mut self, node_id: usize, state: &RustHexoState) -> usize {
        self.refresh_total_legal_actions(node_id, state);
        let node = &self.nodes[node_id];
        let Some(config) = self.widening else {
            if !node.unexpanded_priors.is_empty() {
                return node.unexpanded_priors.len();
            }
            return node.hidden_action_count();
        };
        let initial_limit = if node_id == 0 {
            config.root_initial_actions
        } else {
            config.child_initial_actions
        }
        .min(node.total_legal_actions);
        if node.edges.len() < initial_limit && !node.unexpanded_priors.is_empty() {
            return (initial_limit - node.edges.len()).min(node.unexpanded_priors.len());
        }
        let hidden_count = node.hidden_action_count();
        if hidden_count == 0 {
            return 0;
        }
        let edge_limit = config.edge_limit(node.visits, node.total_legal_actions, node_id == 0);
        if edge_limit <= node.edges.len() {
            return 0;
        }
        (edge_limit - node.edges.len()).min(hidden_count)
    }

    fn select_or_materialize_edge(
        &mut self,
        node_id: usize,
        c_puct: f32,
        state: &RustHexoState,
    ) -> Option<usize> {
        let node = &self.nodes[node_id];
        let exploration_scale = c_puct * (node.visits.max(1) as f32).sqrt();
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value() + edge.prior * exploration_scale / (1.0 + edge.visits as f32);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        let eligible_new = self.eligible_unmaterialized_count(node_id, state);
        if eligible_new > 0 {
            if let Some(candidate) = self.nodes[node_id].unexpanded_priors.last() {
                let score = candidate.prior * exploration_scale;
                let candidate_key = (usize::MAX, score, 0, candidate.action_id);
                let replace = match best {
                    Some(current) => {
                        compare_edge_score(candidate_key, current) == Ordering::Greater
                    }
                    None => true,
                };
                if replace {
                    let candidate = self.nodes[node_id]
                        .unexpanded_priors
                        .pop()
                        .expect("last prior candidate exists");
                    let edge_index = self.nodes[node_id].edges.len();
                    self.nodes[node_id].edges.push(candidate.into_edge());
                    self.record_materialized_edge(node_id);
                    return Some(edge_index);
                }
            } else if let Some(action_id) = next_lazy_legal_action_id(state, &self.nodes[node_id]) {
                let fallback_prior = fallback_prior(&self.nodes[node_id]);
                let score = fallback_prior * exploration_scale;
                let candidate_key = (usize::MAX, score, 0, action_id);
                let replace = match best {
                    Some(current) => {
                        compare_edge_score(candidate_key, current) == Ordering::Greater
                    }
                    None => true,
                };
                if replace {
                    let edge_index = self.nodes[node_id].edges.len();
                    self.nodes[node_id].edges.push(
                        RustPriorCandidate {
                            action_id,
                            prior: fallback_prior,
                        }
                        .into_edge(),
                    );
                    self.record_materialized_edge(node_id);
                    return Some(edge_index);
                }
            }
        }

        best.map(|item| item.0)
    }

    pub(crate) fn apply_virtual_visit(&mut self, path: &[(usize, usize)]) {
        self.completed_visits = self.completed_visits.saturating_add(1);
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].edges[edge_index].visits += 1;
        }
    }

    pub(crate) fn backup_virtual(
        &mut self,
        path: &[(usize, usize)],
        leaf_player: Player,
        leaf_value: f32,
    ) {
        for &(node_id, edge_index) in path {
            let value = if self.nodes[node_id].player == leaf_player {
                leaf_value
            } else {
                -leaf_value
            };
            self.nodes[node_id].value_sum += value;
            self.nodes[node_id].edges[edge_index].value_sum += value;
        }
    }

    pub(crate) fn mark_pending(&mut self, node_id: usize, edge_index: usize, delta: i32) {
        let edge = &mut self.nodes[node_id].edges[edge_index];
        if delta >= 0 {
            edge.pending = edge.pending.saturating_add(delta as u32);
        } else {
            edge.pending = edge.pending.saturating_sub((-delta) as u32);
        }
    }

    fn record_materialized_edge(&mut self, node_id: usize) {
        self.active_edge_count += 1;
        self.hidden_prior_count = self.hidden_prior_count.saturating_sub(1);
        self.widened_edges_total += 1;
        self.max_active_edges_per_node = self
            .max_active_edges_per_node
            .max(self.nodes[node_id].edges.len());
    }

    fn refresh_total_legal_actions(&mut self, node_id: usize, state: &RustHexoState) {
        let node = &self.nodes[node_id];
        if !node.unexpanded_priors.is_empty() || node.edges.len() < node.total_legal_actions {
            return;
        }
        let refreshed = state.legal_move_count().max(node.total_legal_actions);
        if refreshed <= node.total_legal_actions {
            return;
        }
        let added_hidden = refreshed - node.total_legal_actions;
        self.nodes[node_id].total_legal_actions = refreshed;
        self.hidden_prior_count += added_hidden;
        self.max_hidden_priors_per_node = self
            .max_hidden_priors_per_node
            .max(self.nodes[node_id].hidden_action_count());
    }

    pub(crate) fn diagnostics(&self) -> RustSearchDiagnostics {
        RustSearchDiagnostics {
            node_count: self.nodes.len(),
            active_edge_count: self.active_edge_count,
            hidden_prior_count: self.hidden_prior_count,
            root_active_edges: self.nodes.first().map(|node| node.edges.len()).unwrap_or(0),
            root_hidden_priors: self
                .nodes
                .first()
                .map(RustNode::hidden_action_count)
                .unwrap_or(0),
            max_active_edges_per_node: self.max_active_edges_per_node,
            max_hidden_priors_per_node: self.max_hidden_priors_per_node,
            widened_edges_total: self.widened_edges_total,
            ..RustSearchDiagnostics::default()
        }
    }

    pub(crate) fn advance_root(&mut self, action_id: PackedCoord) -> PyResult<bool> {
        let Some((edge_index, edge)) = self
            .nodes
            .first()
            .and_then(|node| {
                node.edges
                    .iter()
                    .enumerate()
                    .find(|(_, edge)| edge.action_id == action_id)
            })
            .map(|(index, edge)| (index, edge.clone()))
        else {
            return Ok(false);
        };
        let Some(child_id) = edge.child else {
            return Ok(false);
        };

        let mut new_root_state = self.root_state.clone();
        apply_placement(&mut new_root_state, Placement { coord: edge.action })
            .map_err(move_error)?;
        if new_root_state.terminal().is_some() {
            return Ok(false);
        }

        let mut old_to_new = HashMap::new();
        let mut nodes = Vec::new();
        clone_subtree_nodes(child_id, &self.nodes, &mut old_to_new, &mut nodes);
        if nodes.is_empty() {
            return Ok(false);
        }

        let root_hash = state_hash(&new_root_state);
        nodes[0].state_hash = root_hash;
        if edge.visits > nodes[0].visits {
            nodes[0].visits = edge.visits;
            nodes[0].value_sum = -edge.value_sum;
        }
        let mut node_table = HashMap::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            node_table.insert(node.state_hash, index);
        }

        self.root_state = new_root_state;
        self.root_hash = root_hash;
        self.nodes = nodes;
        self.node_table = node_table;
        self.target_visits = 0;
        self.completed_visits = self.nodes[0]
            .edges
            .iter()
            .fold(self.nodes[0].visits, |total, edge| total.max(edge.visits));
        self.recompute_accounting();
        let _ = edge_index;
        Ok(true)
    }

    fn recompute_accounting(&mut self) {
        self.active_edge_count = 0;
        self.hidden_prior_count = 0;
        self.max_active_edges_per_node = 0;
        self.max_hidden_priors_per_node = 0;
        for node in &self.nodes {
            let active = node.edges.len();
            let hidden = node.hidden_action_count();
            self.active_edge_count += active;
            self.hidden_prior_count += hidden;
            self.max_active_edges_per_node = self.max_active_edges_per_node.max(active);
            self.max_hidden_priors_per_node = self.max_hidden_priors_per_node.max(hidden);
        }
        self.widened_edges_total = self.active_edge_count;
    }
}

fn clone_subtree_nodes(
    old_id: usize,
    old_nodes: &[RustNode],
    old_to_new: &mut HashMap<usize, usize>,
    new_nodes: &mut Vec<RustNode>,
) -> usize {
    if let Some(new_id) = old_to_new.get(&old_id).copied() {
        return new_id;
    }
    let new_id = new_nodes.len();
    old_to_new.insert(old_id, new_id);
    let mut node = old_nodes[old_id].clone();
    for edge in &mut node.edges {
        edge.child = None;
    }
    new_nodes.push(node);

    for (edge_index, old_edge) in old_nodes[old_id].edges.iter().enumerate() {
        if let Some(old_child) = old_edge.child {
            let new_child = clone_subtree_nodes(old_child, old_nodes, old_to_new, new_nodes);
            new_nodes[new_id].edges[edge_index].child = Some(new_child);
        }
    }
    new_id
}

fn node_from_evaluation(
    state_hash: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    widening: Option<ProgressiveWideningConfig>,
    is_root: bool,
) -> RustNode {
    let mut candidates: Vec<_> = evaluation
        .priors
        .iter()
        .map(|(action_id, prior)| RustPriorCandidate {
            action_id: *action_id,
            prior: sanitize_prior(*prior),
        })
        .collect();
    candidates.sort_by(compare_prior_candidate);
    let mut seen_actions = HashSet::new();
    candidates.retain(|candidate| seen_actions.insert(candidate.action_id));
    candidates.reverse();
    let total_legal_actions = if is_root {
        evaluation.legal_action_count.max(candidates.len())
    } else {
        candidates.len()
    };
    let _ = (widening, is_root);
    RustNode {
        state_hash,
        player: state.current_player(),
        eval_value: evaluation.value.clamp(-1.0, 1.0),
        total_legal_actions,
        visits: 0,
        value_sum: 0.0,
        edges: Vec::new(),
        unexpanded_priors: candidates,
    }
}

fn fallback_prior(node: &RustNode) -> f32 {
    let edge_min = node
        .edges
        .iter()
        .map(|edge| edge.prior)
        .filter(|prior| prior.is_finite() && *prior > 0.0)
        .fold(f32::INFINITY, f32::min);
    let hidden_min = node
        .unexpanded_priors
        .iter()
        .map(|candidate| candidate.prior)
        .filter(|prior| prior.is_finite() && *prior > 0.0)
        .fold(f32::INFINITY, f32::min);
    let min_positive = edge_min.min(hidden_min);
    if min_positive.is_finite() {
        (min_positive * 0.01).max(1.0e-8)
    } else if node.total_legal_actions > 0 {
        1.0 / node.total_legal_actions as f32
    } else {
        0.0
    }
}

fn next_lazy_legal_action_id(state: &RustHexoState, node: &RustNode) -> Option<PackedCoord> {
    if node.edges.len() >= node.total_legal_actions {
        return None;
    }
    let mut legal_action_ids = Vec::with_capacity(node.total_legal_actions);
    state.write_legal_action_ids(&mut legal_action_ids);
    legal_action_ids.into_iter().find(|action_id| {
        !node.edges.iter().any(|edge| edge.action_id == *action_id)
            && !node
                .unexpanded_priors
                .iter()
                .any(|candidate| candidate.action_id == *action_id)
    })
}

fn compare_prior_candidate(left: &RustPriorCandidate, right: &RustPriorCandidate) -> Ordering {
    right
        .prior
        .partial_cmp(&left.prior)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.action_id.cmp(&right.action_id))
}

fn sanitize_prior(prior: f32) -> f32 {
    if prior.is_finite() && prior > 0.0 {
        prior
    } else {
        0.0
    }
}

fn compare_edge_score(
    left: (usize, f32, u32, PackedCoord),
    right: (usize, f32, u32, PackedCoord),
) -> Ordering {
    left.1
        .partial_cmp(&right.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| right.2.cmp(&left.2))
        .then_with(|| right.3.cmp(&left.3))
}

pub(crate) fn terminal_value(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}

pub(crate) fn select_root_action(
    node: &RustNode,
    temperature: f32,
    seed: u64,
) -> Option<PackedCoord> {
    if node.edges.is_empty() {
        return None;
    }
    if temperature <= 1.0e-6 {
        return node
            .edges
            .iter()
            .max_by_key(|edge| (edge.visits, std::cmp::Reverse(edge.action_id)))
            .map(|edge| edge.action_id);
    }
    let inv_temperature = 1.0 / temperature.max(1.0e-3);
    let mut total = 0.0f64;
    let mut weights = Vec::with_capacity(node.edges.len());
    for edge in &node.edges {
        let weight = (edge.visits.max(1) as f32).powf(inv_temperature) as f64;
        total += weight;
        weights.push(weight);
    }
    let mut threshold = random_unit(seed) * total;
    for (edge, weight) in node.edges.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Some(edge.action_id);
        }
    }
    node.edges.last().map(|edge| edge.action_id)
}

fn random_unit(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use hexo_engine::{pack_coord, HexCoord, HexoState as RustHexoState};

    use super::*;

    fn evaluation_with_priors(count: usize) -> RustEvaluation {
        let priors = (0..count)
            .map(|index| {
                (
                    pack_coord(HexCoord {
                        q: index as i16,
                        r: 0,
                    }),
                    index as f32 + 1.0,
                )
            })
            .collect();
        RustEvaluation {
            value: 0.0,
            legal_action_count: count,
            priors,
        }
    }

    #[test]
    fn progressive_widening_starts_with_top_initial_priors() {
        let config = ProgressiveWideningConfig::new(128, 32, 40.0, 1.3);
        let state = RustHexoState::new();
        let node = node_from_evaluation(0, &state, &evaluation_with_priors(200), config, true);

        assert_eq!(node.edges.len(), 0);
        assert_eq!(node.unexpanded_priors.len(), 200);
        assert!(node
            .unexpanded_priors
            .iter()
            .rev()
            .take(128)
            .any(|prior| prior.action_id == pack_coord(HexCoord { q: 199, r: 0 })));
        assert!(!node
            .unexpanded_priors
            .iter()
            .rev()
            .take(128)
            .any(|prior| prior.action_id == pack_coord(HexCoord { q: 0, r: 0 })));
    }

    #[test]
    fn progressive_widening_materializes_edges_lazily_as_visits_grow() {
        let config = ProgressiveWideningConfig::new(128, 32, 40.0, 1.3);
        let state = RustHexoState::new();
        let mut search = RustSearch::new(state, &evaluation_with_priors(300), 128, config);

        let root_state = search.root_state.clone();
        assert_eq!(search.eligible_unmaterialized_count(0, &root_state), 128);
        for _ in 0..128 {
            let root_state = search.root_state.clone();
            let edge_index = search
                .select_or_materialize_edge(0, 1.5, &root_state)
                .unwrap();
            search.nodes[0].edges[edge_index].pending = 1;
        }
        assert_eq!(search.nodes[0].edges.len(), 128);
        let root_state = search.root_state.clone();
        assert_eq!(search.eligible_unmaterialized_count(0, &root_state), 0);
        search.nodes[0].visits = 52;
        let root_state = search.root_state.clone();
        assert_eq!(search.eligible_unmaterialized_count(0, &root_state), 1);
        let root_state = search.root_state.clone();
        assert!(search
            .select_or_materialize_edge(0, 1.5, &root_state)
            .is_some());
        assert_eq!(search.nodes[0].edges.len(), 129);
        assert_eq!(search.nodes[0].unexpanded_priors.len(), 171);
    }

    #[test]
    fn progressive_widening_uses_smaller_child_frontier() {
        let config = ProgressiveWideningConfig::new(128, 32, 40.0, 1.3);
        let state = RustHexoState::new();
        let mut search = RustSearch::new(state.clone(), &evaluation_with_priors(200), 128, config);

        let child_id = search.add_node_from_eval(&state, 1, &evaluation_with_priors(200));
        let root_state = search.root_state.clone();
        assert_eq!(
            search.eligible_unmaterialized_count(child_id, &root_state),
            32
        );
        let node = &search.nodes[child_id];
        assert_eq!(node.edges.len(), 0);
        assert_eq!(node.unexpanded_priors.len(), 200);
        assert!(node
            .unexpanded_priors
            .iter()
            .rev()
            .take(32)
            .any(|prior| prior.action_id == pack_coord(HexCoord { q: 199, r: 0 })));
        assert!(!node
            .unexpanded_priors
            .iter()
            .rev()
            .take(32)
            .any(|prior| prior.action_id == pack_coord(HexCoord { q: 100, r: 0 })));
    }
}
