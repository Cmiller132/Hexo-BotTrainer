//! Dense CNN PUCT tree mechanics.
//!
//! The tree mirrors the usual KataGo-style ownership pattern at a smaller
//! scale: a search owns one root state, an arena of nodes, an exact hash table
//! for already-expanded states, root visit accounting, and a shared evaluator
//! cache handle. Nodes intentionally store only statistics and outgoing edges;
//! traversal recreates leaf states from a root clone plus selected actions so
//! the implementation stays memory-light and easy to inspect.

use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

use hexo_engine::{
    apply_placement, unpack_coord, GameOutcome, HexCoord, HexoState as RustHexoState,
    PackedCoord, Placement, Player,
};
use hexo_utils::StateHash;

use crate::mcts_eval::{state_hash, RustEvaluation, SharedEvaluationCache};
use crate::state::move_error;

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
pub(crate) struct RustNode {
    pub(crate) state_hash: StateHash,
    pub(crate) player: Player,
    pub(crate) eval_value: f32,
    pub(crate) visits: u32,
    pub(crate) value_sum: f32,
    pub(crate) edges: Vec<RustEdge>,
}

impl RustNode {
    pub(crate) fn value(&self) -> f32 {
        if self.visits == 0 {
            self.eval_value
        } else {
            self.value_sum / self.visits as f32
        }
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
    pub(crate) evaluator_cache: SharedEvaluationCache,
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
        evaluator_cache: SharedEvaluationCache,
    ) -> Self {
        let root_hash = state_hash(&root_state);
        let root_node = node_from_evaluation(root_hash, &root_state, evaluation);
        let mut node_table = HashMap::new();
        node_table.insert(root_hash, 0);
        Self {
            root_state,
            root_hash,
            nodes: vec![root_node],
            node_table,
            target_visits,
            completed_visits: 0,
            evaluator_cache,
        }
    }

    pub(crate) fn root_edges_empty(&self) -> bool {
        self.nodes[0].edges.is_empty()
    }

    pub(crate) fn needs_visits(&self) -> bool {
        self.completed_visits < self.target_visits && !self.root_edges_empty()
    }

    pub(crate) fn remaining_visits(&self) -> u32 {
        self.target_visits.saturating_sub(self.completed_visits)
    }

    pub(crate) fn root(&self) -> &RustNode {
        debug_assert_eq!(self.nodes[0].state_hash, self.root_hash);
        &self.nodes[0]
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
        self.nodes.push(node_from_evaluation(hash, state, evaluation));
        self.node_table.insert(hash, id);
        id
    }

    pub(crate) fn select_pending_leaf(&mut self, c_puct: f32) -> PyResult<Option<RustSelectedLeaf>> {
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;
        let mut current_hash = self.root_hash;

        loop {
            let node = &self.nodes[node_id];
            if node.edges.is_empty() {
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
            }

            let Some(edge_index) = self.select_edge(node_id, c_puct) else {
                return Ok(None);
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

    fn select_edge(&self, node_id: usize, c_puct: f32) -> Option<usize> {
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
        best.map(|item| item.0)
    }

    pub(crate) fn apply_virtual_visit(&mut self, path: &[(usize, usize)]) {
        self.completed_visits = self.completed_visits.saturating_add(1);
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].edges[edge_index].visits += 1;
        }
    }

    pub(crate) fn backup_virtual(&mut self, path: &[(usize, usize)], leaf_player: Player, leaf_value: f32) {
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
}

fn node_from_evaluation(
    state_hash: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
) -> RustNode {
    let mut edges: Vec<_> = evaluation
        .priors
        .iter()
        .map(|(action_id, prior)| RustEdge {
            action_id: *action_id,
            action: unpack_coord(*action_id),
            prior: prior.max(0.0),
            visits: 0,
            value_sum: 0.0,
            pending: 0,
            child: None,
        })
        .collect();
    normalize_rust_priors(&mut edges);
    RustNode {
        state_hash,
        player: state.current_player(),
        eval_value: evaluation.value.clamp(-1.0, 1.0),
        visits: 0,
        value_sum: 0.0,
        edges,
    }
}

fn normalize_rust_priors(edges: &mut [RustEdge]) {
    let total: f32 = edges.iter().map(|edge| edge.prior.max(0.0)).sum();
    if total <= f32::EPSILON || !total.is_finite() {
        let prior = if edges.is_empty() {
            0.0
        } else {
            1.0 / edges.len() as f32
        };
        for edge in edges {
            edge.prior = prior;
        }
        return;
    }
    for edge in edges {
        edge.prior = edge.prior.max(0.0) / total;
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

pub(crate) fn select_root_action(node: &RustNode, temperature: f32, seed: u64) -> Option<PackedCoord> {
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
