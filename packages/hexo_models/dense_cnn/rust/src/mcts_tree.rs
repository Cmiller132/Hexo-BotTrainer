//! Dense CNN PUCT tree mechanics.
//!
//! The tree mirrors the usual KataGo-style ownership pattern at a smaller
//! scale: a search owns one root state, an arena of nodes, an exact hash table
//! for already-expanded states, and root visit accounting. Nodes intentionally
//! store only statistics and outgoing edges; traversal recreates leaf states
//! from a root clone plus selected actions so the implementation stays
//! memory-light and keeps move legality inside `hexo_engine`.
//!
//! This module does not call Python and does not encode tensors. It consumes
//! validated `RustEvaluation` values from `mcts_eval`, stages every in-crop legal
//! prior as a lazy candidate, and materializes an edge only when PUCT selects it.
//! Every legal in-crop move is a candidate; there is no progressive widening or
//! candidate cap.

use pyo3::exceptions::PyValueError;
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
pub(crate) struct RootDirichletNoise {
    /// Total Dirichlet concentration spread across legal moves; the per-action
    /// concentration is `total_alpha / legal_count` (KataGo's scheme, so the
    /// noise strength does not depend on how many moves are legal).
    pub(crate) total_alpha: f32,
    pub(crate) fraction: f32,
    pub(crate) seed: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct RustEdge {
    /// Packed engine coordinate used by Python payloads and visit policies.
    pub(crate) action_id: PackedCoord,
    /// Unpacked coordinate used by `hexo_engine::apply_placement`.
    pub(crate) action: HexCoord,
    /// PUCT prior mass for this materialized edge.
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

    fn value_or_fpu(&self, parent_value: f32, fpu_reduction: f32) -> f32 {
        if self.visits == 0 {
            parent_value - fpu_reduction
        } else {
            self.value()
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
    pub(crate) root_active_edges: usize,
    pub(crate) max_active_edges_per_node: usize,
}

impl RustNode {
    pub(crate) fn value(&self) -> f32 {
        if self.visits == 0 {
            self.eval_value
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn has_actions(&self) -> bool {
        !self.edges.is_empty() || !self.unexpanded_priors.is_empty()
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
    fpu_reduction: f32,
    active_edge_count: usize,
    max_active_edges_per_node: usize,
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
        fpu_reduction: f32,
        root_policy_temperature: f32,
        root_noise: Option<RootDirichletNoise>,
    ) -> PyResult<Self> {
        // The root node starts with priors staged but no active edges. Edges are
        // materialized lazily by `select_or_materialize_edge` according to PUCT
        // score. Root-policy temperature and Dirichlet noise apply at the root.
        let root_hash = state_hash(&root_state);
        let root_node = node_from_evaluation(
            root_hash,
            &root_state,
            evaluation,
            Some(root_policy_temperature),
            root_noise,
        )?;
        let mut node_table = HashMap::new();
        node_table.insert(root_hash, 0);
        Ok(Self {
            root_state,
            root_hash,
            nodes: vec![root_node],
            node_table,
            target_visits,
            completed_visits: 0,
            fpu_reduction,
            active_edge_count: 0,
            max_active_edges_per_node: 0,
        })
    }

    pub(crate) fn apply_root_dirichlet_noise(&mut self, noise: RootDirichletNoise) {
        let root = &mut self.nodes[0];
        let count = root.edges.len() + root.unexpanded_priors.len();
        if count == 0 || noise.total_alpha <= 0.0 || noise.fraction <= 0.0 {
            return;
        }
        let samples = dirichlet_samples(count, noise);
        let visible_total: f32 = root
            .edges
            .iter()
            .map(|edge| edge.prior)
            .chain(
                root.unexpanded_priors
                    .iter()
                    .map(|candidate| candidate.prior),
            )
            .filter(|prior| prior.is_finite())
            .sum();
        let fraction = noise.fraction;
        let mut sample_index = 0usize;
        for edge in &mut root.edges {
            edge.prior =
                (1.0 - fraction) * edge.prior + fraction * samples[sample_index] * visible_total;
            sample_index += 1;
        }
        for candidate in &mut root.unexpanded_priors {
            candidate.prior = (1.0 - fraction) * candidate.prior
                + fraction * samples[sample_index] * visible_total;
            sample_index += 1;
        }
        root.unexpanded_priors.sort_by(compare_prior_candidate);
        root.unexpanded_priors.reverse();
    }

    pub(crate) fn root_edges_empty(&self) -> bool {
        !self.nodes[0].has_actions()
    }

    pub(crate) fn needs_visits(&self) -> bool {
        self.completed_visits < self.target_visits && !self.root_edges_empty()
    }

    pub(crate) fn remaining_visits(&self) -> u32 {
        self.target_visits.saturating_sub(self.completed_visits)
    }

    pub(crate) fn set_additional_visits(&mut self, visits: u32) {
        self.target_visits = self.completed_visits.saturating_add(visits);
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
    ) -> PyResult<usize> {
        if let Some(existing) = self.node_table.get(&hash).copied() {
            return Ok(existing);
        }
        let id = self.nodes.len();
        let node = node_from_evaluation(hash, state, evaluation, None, None)?;
        self.nodes.push(node);
        self.node_table.insert(hash, id);
        Ok(id)
    }

    pub(crate) fn select_pending_leaf(
        &mut self,
        c_puct: f32,
    ) -> PyResult<Option<RustSelectedLeaf>> {
        // Recreate the selected leaf state by replaying edge actions from the
        // root clone. Nodes store search statistics only, so the engine remains
        // the single source of truth for move application and terminal checks.
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;
        let mut current_hash = self.root_hash;

        loop {
            let Some(edge_index) = self.select_or_materialize_edge(node_id, c_puct) else {
                // No edge can be selected from this node. If we reached it via a
                // parent edge, return the existing node value for backup.
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
                // Another virtual batch already selected this edge and is
                // waiting for evaluation, so avoid duplicating the same pending
                // leaf in this batch.
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

    fn select_or_materialize_edge(&mut self, node_id: usize, c_puct: f32) -> Option<usize> {
        // First score already-materialized edges. Then compare the best existing
        // edge against the next staged prior candidate, materializing that
        // candidate only if its prior/exploration score wins.
        let node = &self.nodes[node_id];
        let exploration_scale = c_puct * (node.visits.max(1) as f32).sqrt();
        let parent_value = node.value();
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value_or_fpu(parent_value, self.fpu_reduction)
                + edge.prior * exploration_scale / (1.0 + edge.visits as f32);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        if let Some(candidate) = self.nodes[node_id].unexpanded_priors.last() {
            let score = candidate.prior * exploration_scale;
            let candidate_key = (usize::MAX, score, 0, candidate.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate_key, current) == Ordering::Greater,
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
        }

        best.map(|item| item.0)
    }

    pub(crate) fn apply_virtual_visit(&mut self, path: &[(usize, usize)], virtual_loss: f32) {
        // Virtual visits reserve a path during batched leaf gathering. The
        // later backup adds back the virtual loss and the real leaf value.
        self.completed_visits = self.completed_visits.saturating_add(1);
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].value_sum -= virtual_loss;
            self.nodes[node_id].edges[edge_index].visits += 1;
            self.nodes[node_id].edges[edge_index].value_sum -= virtual_loss;
        }
    }

    pub(crate) fn backup_virtual(
        &mut self,
        path: &[(usize, usize)],
        leaf_player: Player,
        leaf_value: f32,
        virtual_loss: f32,
    ) {
        // Values are stored from each node player's perspective, so a leaf value
        // is negated whenever the node player differs from the evaluated player.
        for &(node_id, edge_index) in path {
            let value = if self.nodes[node_id].player == leaf_player {
                leaf_value
            } else {
                -leaf_value
            };
            self.nodes[node_id].value_sum += value + virtual_loss;
            self.nodes[node_id].edges[edge_index].value_sum += value + virtual_loss;
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
        self.max_active_edges_per_node = self
            .max_active_edges_per_node
            .max(self.nodes[node_id].edges.len());
    }

    pub(crate) fn diagnostics(&self) -> RustSearchDiagnostics {
        RustSearchDiagnostics {
            node_count: self.nodes.len(),
            active_edge_count: self.active_edge_count,
            root_active_edges: self.nodes.first().map(|node| node.edges.len()).unwrap_or(0),
            max_active_edges_per_node: self.max_active_edges_per_node,
        }
    }

    pub(crate) fn advance_root(&mut self, action_id: PackedCoord) -> PyResult<bool> {
        // Promote the selected child subtree after a move. If the child was not
        // expanded or the move ends the game, the session simply drops the tree.
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
        self.max_active_edges_per_node = 0;
        for node in &self.nodes {
            let active = node.edges.len();
            self.active_edge_count += active;
            self.max_active_edges_per_node = self.max_active_edges_per_node.max(active);
        }
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
    root_policy_temperature: Option<f32>,
    root_noise: Option<RootDirichletNoise>,
) -> PyResult<RustNode> {
    // The evaluator returns one prior per in-crop legal move, so every legal
    // candidate is staged here. Root-policy temperature softens the prior at the
    // root before Dirichlet noise; both are skipped for interior nodes.
    let mut candidates: Vec<_> = evaluation
        .priors
        .iter()
        .map(|(action_id, prior)| RustPriorCandidate {
            action_id: *action_id,
            prior: *prior,
        })
        .collect();
    candidates.sort_by(compare_prior_candidate);
    let mut seen_actions = HashSet::new();
    candidates.retain(|candidate| seen_actions.insert(candidate.action_id));
    if let Some(temperature) = root_policy_temperature {
        apply_root_policy_temperature(&mut candidates, temperature);
    }
    normalize_candidate_priors(&mut candidates)?;
    if let Some(noise) = root_noise {
        apply_dirichlet_noise(&mut candidates, noise);
    }
    candidates.sort_by(compare_prior_candidate);
    candidates.reverse();
    Ok(RustNode {
        state_hash,
        player: state.current_player(),
        eval_value: evaluation.value,
        visits: 0,
        value_sum: 0.0,
        edges: Vec::new(),
        unexpanded_priors: candidates,
    })
}

fn apply_root_policy_temperature(candidates: &mut [RustPriorCandidate], temperature: f32) {
    // Raise each prior to 1 / temperature (KataGo root policy softmax temperature).
    // temperature > 1 flattens the prior so search explores beyond the model's
    // current favorite; temperature == 1 is a no-op.
    if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < 1.0e-6 {
        return;
    }
    let inverse = 1.0 / temperature;
    for candidate in candidates.iter_mut() {
        if candidate.prior.is_finite() && candidate.prior > 0.0 {
            candidate.prior = candidate.prior.powf(inverse);
        }
    }
}

fn apply_dirichlet_noise(candidates: &mut [RustPriorCandidate], noise: RootDirichletNoise) {
    if candidates.is_empty() || noise.total_alpha <= 0.0 || noise.fraction <= 0.0 {
        return;
    }
    let fraction = noise.fraction;
    let samples = dirichlet_samples(candidates.len(), noise);
    for (candidate, sampled) in candidates.iter_mut().zip(samples) {
        candidate.prior = (1.0 - fraction) * candidate.prior + fraction * sampled;
    }
}

fn dirichlet_samples(count: usize, noise: RootDirichletNoise) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    // KataGo spreads a total concentration across legal moves: per-action alpha
    // is total_alpha / legal_count.
    let per_action_alpha = (noise.total_alpha as f64 / count as f64).max(1.0e-6);
    let mut sampler = DirichletSampler::new(noise.seed);
    let mut samples = Vec::with_capacity(count);
    let mut total = 0.0f64;
    for _ in 0..count {
        let value = sampler.gamma(per_action_alpha);
        samples.push(value);
        total += value;
    }
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / count as f32; count];
    }
    samples
        .into_iter()
        .map(|sample| (sample / total) as f32)
        .collect()
}

struct DirichletSampler {
    state: u64,
}

impl DirichletSampler {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xD1B5_4A32_D192_ED03,
        }
    }

    fn uniform_open(&mut self) -> f64 {
        random_unit(self.next_u64()).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform_open();
        let u2 = self.uniform_open();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let boosted = self.gamma(alpha + 1.0);
            return boosted * self.uniform_open().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.uniform_open();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v3;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }
}

fn compare_prior_candidate(left: &RustPriorCandidate, right: &RustPriorCandidate) -> Ordering {
    right
        .prior
        .partial_cmp(&left.prior)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.action_id.cmp(&right.action_id))
}

fn normalize_candidate_priors(candidates: &mut [RustPriorCandidate]) -> PyResult<()> {
    let mut total = 0.0f32;
    for candidate in candidates.iter() {
        if !candidate.prior.is_finite() || candidate.prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "prior for action {} must be finite and >= 0",
                candidate.action_id
            )));
        }
        total += candidate.prior;
    }
    if candidates.is_empty() {
        return Ok(());
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "candidate priors must contain positive mass",
        ));
    }
    for candidate in candidates {
        candidate.prior /= total;
    }
    Ok(())
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
    fn root_stages_every_legal_prior_without_a_cap() {
        let state = RustHexoState::new();
        let node = node_from_evaluation(0, &state, &evaluation_with_priors(200), Some(1.0), None).unwrap();
        assert_eq!(node.edges.len(), 0);
        assert_eq!(node.unexpanded_priors.len(), 200);
        // Highest prior is staged last so PUCT pops it first.
        assert_eq!(
            node.unexpanded_priors.last().unwrap().action_id,
            pack_coord(HexCoord { q: 199, r: 0 })
        );
    }

    #[test]
    fn edges_materialize_lazily_in_prior_order() {
        let state = RustHexoState::new();
        let mut search =
            RustSearch::new(state, &evaluation_with_priors(8), 128, 0.20, 1.0, None).unwrap();
        for _ in 0..8 {
            let edge_index = search.select_or_materialize_edge(0, 1.5).unwrap();
            search.nodes[0].edges[edge_index].pending = 1;
        }
        assert_eq!(search.nodes[0].edges.len(), 8);
        assert!(search.nodes[0].unexpanded_priors.is_empty());
        // First materialized edge is the highest-prior move.
        assert_eq!(
            search.nodes[0].edges[0].action_id,
            pack_coord(HexCoord { q: 7, r: 0 })
        );
    }

    #[test]
    fn root_policy_temperature_flattens_priors() {
        let state = RustHexoState::new();
        let sharp = node_from_evaluation(0, &state, &evaluation_with_priors(4), Some(1.0), None).unwrap();
        let flat = node_from_evaluation(0, &state, &evaluation_with_priors(4), Some(2.0), None).unwrap();
        let sharp_top = sharp.unexpanded_priors.last().unwrap().prior;
        let flat_top = flat.unexpanded_priors.last().unwrap().prior;
        // Temperature > 1 reduces the gap between the top prior and uniform.
        assert!(flat_top < sharp_top);
    }
}
