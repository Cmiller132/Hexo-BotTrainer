//! hexfield PUCT tree — from-scratch port of the as-built dense_cnn tree
//! semantics (mcts_tree.rs is the semantic reference; the M5/M6 stub-evaluator
//! differential harness pins this implementation against it bit-for-bit in
//! `search_parity_mode`), plus the §5.4 divergence machinery:
//!
//! - per-edge sum-of-squares accumulator (LCB selection; inert in parity)
//! - per-node/per-edge (ml_sum, ml_weight) moves-left stats (§5.4.4; inert
//!   when the utility is off)
//! - visit-scaled c_puct schedule (§5.4.3; off => the caller's static c)
//!
//! No crop exists: every engine-legal move is a candidate by construction, so
//! TSS injection is total (the dense call-site crop filter is deleted).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hexo_engine::{
    apply_placement, pack_coord, unpack_coord, GameOutcome, HexCoord, HexoState as RustHexoState,
    PackedCoord, Placement, Player,
};
use hexo_utils::StateHash;

use crate::cache::{state_hash, RustEvaluation};
use crate::state::move_error;
use crate::threats_shared as threats;

#[derive(Clone, Copy, Debug)]
pub struct RootDirichletNoise {
    pub total_alpha: f32,
    pub fraction: f32,
    pub seed: u64,
}

/// §5.4 divergence toggles + constants. `parity()` (== `search_parity_mode`)
/// forces all four off; production defaults ship all four ON (spec §5.4).
#[derive(Clone, Copy, Debug)]
pub struct Divergences {
    pub lcb_move_selection: bool,
    pub lcb_z: f32,
    pub lcb_min_visits: u32,
    pub lcb_visit_fraction: f32,
    pub early_stop: bool,
    pub full_visit_floor: f32,
    pub visit_scaled_c_puct: bool,
    pub c_scale: f32,
    pub c_base: f32,
    pub moves_left_utility: bool,
    pub ml_weight: f32,
    pub ml_scale: f32,
    pub ml_q_gate: f32,
}

impl Divergences {
    pub fn parity() -> Self {
        Self {
            lcb_move_selection: false,
            lcb_z: 1.6,
            lcb_min_visits: 8,
            lcb_visit_fraction: 0.1,
            early_stop: false,
            full_visit_floor: 0.75,
            visit_scaled_c_puct: false,
            c_scale: 0.45,
            c_base: 500.0,
            moves_left_utility: false,
            ml_weight: 0.03,
            ml_scale: 32.0,
            ml_q_gate: 0.6,
        }
    }

    pub fn production() -> Self {
        Self {
            lcb_move_selection: true,
            early_stop: true,
            visit_scaled_c_puct: true,
            moves_left_utility: true,
            ..Self::parity()
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustEdge {
    pub action_id: PackedCoord,
    pub action: HexCoord,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
    /// Sum of squared REAL backup values (LCB sigma; virtual losses excluded —
    /// a schema addition that is inert in parity mode).
    pub value_sq_sum: f32,
    /// Moves-left stats accumulated on real backups (§5.4.4; inert when off).
    pub ml_sum: f32,
    pub ml_weight: f32,
    pub pending: u32,
    pub child: Option<usize>,
    pub forced: bool,
}

impl RustEdge {
    pub fn value(&self) -> f32 {
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

    pub fn ml_mean(&self) -> Option<f32> {
        if self.ml_weight > 0.0 {
            Some(self.ml_sum / self.ml_weight)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustPriorCandidate {
    pub action_id: PackedCoord,
    pub prior: f32,
}

impl RustPriorCandidate {
    fn into_edge(self) -> RustEdge {
        RustEdge {
            action_id: self.action_id,
            action: unpack_coord(self.action_id),
            prior: self.prior,
            visits: 0,
            value_sum: 0.0,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Widening {
    pub mass: f32,
    pub min_children: usize,
    pub max_children: usize,
}

#[derive(Clone, Debug)]
pub enum NodePriors {
    /// Interior nodes share the cache's DESCENDING normalized prior vector;
    /// the next unexpanded candidate is `priors[edges.len()]`.
    Shared(Arc<RustEvaluation>),
    /// Root nodes: owned ASCENDING candidate list (highest popped from back).
    Owned(Vec<RustPriorCandidate>),
}

#[derive(Clone, Debug)]
pub struct RustNode {
    pub state_hash: StateHash,
    pub player: Player,
    pub eval_value: f32,
    /// Evaluator moves-left decode for this node's state (decisions).
    pub eval_ml: Option<f32>,
    pub visits: u32,
    pub value_sum: f32,
    pub ml_sum: f32,
    pub ml_weight: f32,
    pub edges: Vec<RustEdge>,
    pub priors: NodePriors,
    pub max_eligible_children: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RustSearchDiagnostics {
    pub node_count: usize,
    pub active_edge_count: usize,
    pub hidden_prior_count: usize,
    pub root_active_edges: usize,
    pub root_hidden_priors: usize,
    pub max_active_edges_per_node: usize,
    pub max_hidden_priors_per_node: usize,
    pub active_edge_bytes: usize,
    pub hidden_prior_bytes: usize,
    pub shared_prior_nodes: usize,
    pub shared_prior_refs: usize,
}

impl RustNode {
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            self.eval_value
        } else {
            self.value_sum / self.visits as f32
        }
    }

    pub fn ml_mean(&self) -> Option<f32> {
        if self.ml_weight > 0.0 {
            Some(self.ml_sum / self.ml_weight)
        } else {
            self.eval_ml
        }
    }

    fn has_actions(&self) -> bool {
        !self.edges.is_empty() || self.remaining_prior_count() > 0
    }

    pub fn remaining_prior_count(&self) -> usize {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors.len().saturating_sub(self.edges.len()),
            NodePriors::Owned(unexpanded) => unexpanded.len(),
        }
    }

    fn peek_next_candidate(&self) -> Option<(PackedCoord, f32)> {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors.get(self.edges.len()).copied(),
            NodePriors::Owned(unexpanded) => unexpanded
                .last()
                .map(|candidate| (candidate.action_id, candidate.prior)),
        }
    }

    fn materialize_next_candidate(&mut self) -> RustEdge {
        match &mut self.priors {
            NodePriors::Owned(unexpanded) => unexpanded
                .pop()
                .expect("last prior candidate exists")
                .into_edge(),
            NodePriors::Shared(eval) => {
                let (action_id, prior) = eval.priors[self.edges.len()];
                RustPriorCandidate { action_id, prior }.into_edge()
            }
        }
    }

    pub fn remaining_priors(&self) -> Vec<(PackedCoord, f32)> {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors[self.edges.len().min(eval.priors.len())..]
                .iter()
                .copied()
                .collect(),
            NodePriors::Owned(unexpanded) => unexpanded
                .iter()
                .map(|candidate| (candidate.action_id, candidate.prior))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustSearch {
    pub root_state: RustHexoState,
    pub root_hash: StateHash,
    pub nodes: Vec<RustNode>,
    pub node_table: HashMap<StateHash, usize>,
    pub target_visits: u32,
    pub completed_visits: u32,
    fpu_reduction: f32,
    root_fpu_reduction: f32,
    widening: Widening,
    forced_playout_k: f32,
    pub tss_enabled: bool,
    pub divergences: Divergences,
    /// Set when an early-stop fired for this search (telemetry).
    pub early_stopped: bool,
    active_edge_count: usize,
    max_active_edges_per_node: usize,
}

pub struct RustSelectedLeaf {
    pub path: Vec<(usize, usize)>,
    pub state: RustHexoState,
    pub state_hash: StateHash,
    pub parent_node: usize,
    pub edge_index: usize,
    pub terminal: Option<GameOutcome>,
    pub existing_node: Option<usize>,
}

pub struct RustLeaf {
    pub root_index: usize,
    pub parent_node: usize,
    pub edge_index: usize,
    pub path: Vec<(usize, usize)>,
    pub state: RustHexoState,
    pub state_hash: StateHash,
}

impl RustSearch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        root_state: RustHexoState,
        evaluation: &RustEvaluation,
        target_visits: u32,
        fpu_reduction: f32,
        root_fpu_reduction: f32,
        root_policy_temperature: f32,
        root_noise: Option<RootDirichletNoise>,
        widening: Widening,
        forced_playout_k: f32,
        tss_enabled: bool,
        divergences: Divergences,
    ) -> PyResult<Self> {
        let root_hash = state_hash(&root_state);
        let root_node = owned_root_from_evaluation(
            root_hash,
            &root_state,
            evaluation,
            Some(root_policy_temperature),
            root_noise,
            widening,
            tss_enabled,
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
            root_fpu_reduction,
            widening,
            forced_playout_k,
            tss_enabled,
            divergences,
            early_stopped: false,
            active_edge_count: 0,
            max_active_edges_per_node: 0,
        })
    }

    pub fn set_forced_playout_k(&mut self, k: f32) {
        self.forced_playout_k = k;
    }

    pub fn set_tss_enabled(&mut self, enabled: bool) {
        self.tss_enabled = enabled;
    }

    pub fn set_root_fpu_reduction(&mut self, value: f32) {
        self.root_fpu_reduction = value;
    }

    pub fn set_divergences(&mut self, divergences: Divergences) {
        self.divergences = divergences;
    }

    /// Root-policy softmax temperature on a REUSED root (raw eval priors),
    /// applied before noise, at most once per root lifetime — dense semantics
    /// verbatim (the quarantined production default passes 1.0 == no-op).
    pub fn apply_root_policy_temperature(&mut self, temperature: f32) {
        if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < 1.0e-6 {
            return;
        }
        self.ensure_root_owned();
        let root = &mut self.nodes[0];
        let NodePriors::Owned(unexpanded) = &mut root.priors else {
            return;
        };
        let inverse = 1.0 / temperature;
        let mut total = 0.0f32;
        for edge in root.edges.iter_mut() {
            if edge.prior.is_finite() && edge.prior > 0.0 {
                edge.prior = edge.prior.powf(inverse);
                total += edge.prior;
            }
        }
        for candidate in unexpanded.iter_mut() {
            if candidate.prior.is_finite() && candidate.prior > 0.0 {
                candidate.prior = candidate.prior.powf(inverse);
                total += candidate.prior;
            }
        }
        if total > 0.0 {
            for edge in root.edges.iter_mut() {
                if edge.prior.is_finite() && edge.prior > 0.0 {
                    edge.prior /= total;
                }
            }
            for candidate in unexpanded.iter_mut() {
                if candidate.prior.is_finite() && candidate.prior > 0.0 {
                    candidate.prior /= total;
                }
            }
        }
    }

    fn ensure_root_owned(&mut self) {
        let root = &mut self.nodes[0];
        let owned = match &root.priors {
            NodePriors::Owned(_) => return,
            NodePriors::Shared(eval) => {
                let start = root.edges.len().min(eval.priors.len());
                let mut unexpanded: Vec<RustPriorCandidate> = eval.priors[start..]
                    .iter()
                    .map(|&(action_id, prior)| RustPriorCandidate { action_id, prior })
                    .collect();
                unexpanded.reverse();
                unexpanded.shrink_to_fit();
                unexpanded
            }
        };
        root.priors = NodePriors::Owned(owned);
    }

    pub fn apply_root_dirichlet_noise(&mut self, noise: RootDirichletNoise) {
        self.ensure_root_owned();
        let root = &mut self.nodes[0];
        let NodePriors::Owned(unexpanded) = &mut root.priors else {
            return;
        };
        let count = root.edges.len() + unexpanded.len();
        if count == 0 || noise.total_alpha <= 0.0 || noise.fraction <= 0.0 {
            return;
        }
        let samples = dirichlet_samples(count, noise);
        let visible_total: f32 = root
            .edges
            .iter()
            .map(|edge| edge.prior)
            .chain(unexpanded.iter().map(|candidate| candidate.prior))
            .filter(|prior| prior.is_finite())
            .sum();
        let fraction = noise.fraction;
        let mut sample_index = 0usize;
        for edge in &mut root.edges {
            edge.prior =
                (1.0 - fraction) * edge.prior + fraction * samples[sample_index] * visible_total;
            sample_index += 1;
        }
        for candidate in unexpanded.iter_mut() {
            candidate.prior = (1.0 - fraction) * candidate.prior
                + fraction * samples[sample_index] * visible_total;
            sample_index += 1;
        }
        unexpanded.sort_by(compare_prior_candidate);
        unexpanded.reverse();
    }

    pub fn root_edges_empty(&self) -> bool {
        !self.nodes[0].has_actions()
    }

    pub fn needs_visits(&self) -> bool {
        self.completed_visits < self.target_visits && !self.root_edges_empty()
    }

    pub fn remaining_visits(&self) -> u32 {
        self.target_visits.saturating_sub(self.completed_visits)
    }

    pub fn set_additional_visits(&mut self, visits: u32) {
        self.target_visits = self.completed_visits.saturating_add(visits);
        self.early_stopped = false;
    }

    pub fn root(&self) -> &RustNode {
        debug_assert_eq!(self.nodes[0].state_hash, self.root_hash);
        &self.nodes[0]
    }

    pub fn root_edge_visits(&self) -> Vec<(PackedCoord, u32)> {
        self.root()
            .edges
            .iter()
            .map(|edge| (edge.action_id, edge.visits))
            .collect()
    }

    pub fn add_node_from_eval(
        &mut self,
        state: &RustHexoState,
        hash: StateHash,
        evaluation: Arc<RustEvaluation>,
    ) -> PyResult<usize> {
        if let Some(existing) = self.node_table.get(&hash).copied() {
            return Ok(existing);
        }
        let id = self.nodes.len();
        // TSS expansion injection. Injection is TOTAL by construction here:
        // every tactical cell is engine-legal (hex-dist <= 5 of a stone) and
        // the candidate vocabulary is the full legal set — the dense crop
        // filter does not exist.
        let tactical = if self.tss_enabled {
            threats::tactical_cells(state)
        } else {
            Vec::new()
        };
        let node = if tactical.is_empty() {
            shared_from_cache(hash, state, evaluation, self.widening)
        } else {
            owned_with_injection_from_eval(hash, state, &evaluation, self.widening, &tactical)
        };
        let injected_edges = node.edges.len();
        self.nodes.push(node);
        self.node_table.insert(hash, id);
        self.active_edge_count += injected_edges;
        self.max_active_edges_per_node = self.max_active_edges_per_node.max(injected_edges);
        Ok(id)
    }

    /// §5.4.3: exploration constant for a node with `visits` — static c in
    /// parity, else c_init + c_scale * ln((n + c_base) / c_base).
    fn c_for(&self, c_puct: f32, visits: u32) -> f32 {
        if !self.divergences.visit_scaled_c_puct {
            return c_puct;
        }
        c_puct
            + self.divergences.c_scale
                * ((visits as f32 + self.divergences.c_base) / self.divergences.c_base).ln()
    }

    /// §5.4.4 moves-left selection bonus for one edge (0 when off / ungated /
    /// no stats): -w * g(Q_e) * tanh((M_e - M_node) / m_scale), win-side gate
    /// g = 1 iff Q_e > ml_q_gate. Delegates to the same core the M6 property
    /// tests exercise.
    fn ml_bonus(&self, node: &RustNode, edge: &RustEdge) -> f32 {
        if !self.divergences.moves_left_utility {
            return 0.0;
        }
        if edge.visits == 0 {
            return 0.0;
        }
        let (Some(m_edge), Some(m_node)) = (edge.ml_mean(), node.ml_mean()) else {
            return 0.0;
        };
        crate::search::debug_ml_bonus(
            edge.value(),
            m_edge,
            m_node,
            self.divergences.ml_weight,
            self.divergences.ml_scale,
            self.divergences.ml_q_gate,
        )
    }

    pub fn select_pending_leaf(&mut self, c_puct: f32) -> PyResult<Option<RustSelectedLeaf>> {
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;
        let mut current_hash = self.root_hash;

        loop {
            let Some(edge_index) = self.select_or_materialize_edge(node_id, c_puct) else {
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

    fn select_or_materialize_edge(&mut self, node_id: usize, c_puct: f32) -> Option<usize> {
        // TSS forced edges get a guaranteed first visit before normal PUCT.
        for (index, edge) in self.nodes[node_id].edges.iter().enumerate() {
            if edge.forced && edge.visits == 0 && !(edge.pending > 0 && edge.child.is_none()) {
                return Some(index);
            }
        }

        let node = &self.nodes[node_id];
        let exploration_scale =
            self.c_for(c_puct, node.visits) * (node.visits.max(1) as f32).sqrt();
        let parent_value = node.value();
        let fpu_reduction = if node_id == 0 {
            self.root_fpu_reduction
        } else {
            self.fpu_reduction
        };
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value_or_fpu(parent_value, fpu_reduction)
                + edge.prior * exploration_scale / (1.0 + edge.visits as f32)
                + self.ml_bonus(node, edge);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        let can_widen =
            self.nodes[node_id].edges.len() < self.nodes[node_id].max_eligible_children;
        if can_widen {
            if let Some((action_id, prior)) = self.nodes[node_id].peek_next_candidate() {
                let score = prior * exploration_scale;
                let candidate_key = (usize::MAX, score, 0, action_id);
                let replace = match best {
                    Some(current) => compare_edge_score(candidate_key, current) == Ordering::Greater,
                    None => true,
                };
                if replace {
                    let edge_index = self.nodes[node_id].edges.len();
                    let edge = self.nodes[node_id].materialize_next_candidate();
                    self.nodes[node_id].edges.push(edge);
                    self.record_materialized_edge(node_id);
                    return Some(edge_index);
                }
            }
        }

        if node_id == 0 && self.forced_playout_k > 0.0 {
            if let Some(forced) = self.forced_root_edge() {
                return Some(forced);
            }
        }

        best.map(|item| item.0)
    }

    fn forced_root_edge(&self) -> Option<usize> {
        let root = &self.nodes[0];
        let root_visits = root.visits.max(1) as f32;
        let k = self.forced_playout_k;
        let mut best: Option<(usize, f32)> = None;
        for (index, edge) in root.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            if !(edge.prior.is_finite() && edge.prior > 0.0) {
                continue;
            }
            let n_forced = (k * edge.prior * root_visits).sqrt();
            let deficit = n_forced - edge.visits as f32;
            if deficit > 0.0 {
                let replace = match best {
                    Some((_, best_deficit)) => deficit > best_deficit,
                    None => true,
                };
                if replace {
                    best = Some((index, deficit));
                }
            }
        }
        best.map(|(index, _)| index)
    }

    pub fn apply_virtual_visit(&mut self, path: &[(usize, usize)], virtual_loss: f32) {
        self.completed_visits = self.completed_visits.saturating_add(1);
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].value_sum -= virtual_loss;
            self.nodes[node_id].edges[edge_index].visits += 1;
            self.nodes[node_id].edges[edge_index].value_sum -= virtual_loss;
        }
    }

    /// Real backup (adds back the virtual loss). `leaf_ml` is the moves-left
    /// estimate at the leaf in DECISIONS (terminals contribute exact path
    /// distance — 0 at the terminal itself; the off-by-one is handled by the
    /// per-step increment below). ML stats are side-agnostic.
    pub fn backup_virtual(
        &mut self,
        path: &[(usize, usize)],
        leaf_player: Player,
        leaf_value: f32,
        virtual_loss: f32,
        leaf_ml: Option<f32>,
    ) {
        let depth = path.len();
        for (step, &(node_id, edge_index)) in path.iter().enumerate() {
            let value = if self.nodes[node_id].player == leaf_player {
                leaf_value
            } else {
                -leaf_value
            };
            let node = &mut self.nodes[node_id];
            node.value_sum += value + virtual_loss;
            if let Some(ml) = leaf_ml {
                let ml_here = ml + (depth - step) as f32;
                node.ml_sum += ml_here;
                node.ml_weight += 1.0;
                let edge = &mut node.edges[edge_index];
                edge.value_sum += value + virtual_loss;
                edge.value_sq_sum += value * value;
                edge.ml_sum += ml_here - 1.0; // the edge's child sits one decision deeper
                edge.ml_weight += 1.0;
            } else {
                let edge = &mut node.edges[edge_index];
                edge.value_sum += value + virtual_loss;
                edge.value_sq_sum += value * value;
            }
        }
    }

    pub fn mark_pending(&mut self, node_id: usize, edge_index: usize, delta: i32) {
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

    pub fn diagnostics(&self) -> RustSearchDiagnostics {
        let mut hidden_prior_count = 0usize;
        let mut max_hidden_priors_per_node = 0usize;
        let mut shared_prior_nodes = 0usize;
        let mut shared_prior_refs = 0usize;
        for node in &self.nodes {
            match &node.priors {
                NodePriors::Owned(unexpanded) => {
                    hidden_prior_count += unexpanded.len();
                    max_hidden_priors_per_node = max_hidden_priors_per_node.max(unexpanded.len());
                }
                NodePriors::Shared(_) => {
                    shared_prior_nodes += 1;
                    shared_prior_refs += node.remaining_prior_count();
                }
            }
        }
        RustSearchDiagnostics {
            node_count: self.nodes.len(),
            active_edge_count: self.active_edge_count,
            hidden_prior_count,
            root_active_edges: self.nodes.first().map(|node| node.edges.len()).unwrap_or(0),
            root_hidden_priors: self
                .nodes
                .first()
                .map(|node| node.remaining_prior_count())
                .unwrap_or(0),
            max_active_edges_per_node: self.max_active_edges_per_node,
            max_hidden_priors_per_node,
            active_edge_bytes: self.active_edge_count * std::mem::size_of::<RustEdge>(),
            hidden_prior_bytes: hidden_prior_count * std::mem::size_of::<RustPriorCandidate>(),
            shared_prior_nodes,
            shared_prior_refs,
        }
    }

    pub fn advance_root(&mut self, action_id: PackedCoord) -> PyResult<bool> {
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
            // INHERITED-FROM-DENSE (audit 2026-06-13, intentional): the edge
            // value_sum is the child value from the PARENT's perspective, so the
            // strictly-correct promoted-root value would negate ONLY when the
            // child's side-to-move differs from the parent's — which is NOT the
            // case for a FirstStone->SecondStone promotion (same player). dense
            // mcts_tree.rs negates unconditionally; hexfield reproduces it
            // verbatim so the M5/M6 differential-parity gate holds bit-for-bit.
            // The effect is bounded: this only seeds the promoted root's FPU
            // baseline / first reported value and is overwritten by fresh
            // backups within the next search. Owner decision to "fix" (negate
            // conditionally on player flip, diverging from dense) is deferred;
            // it is not a hexfield-introduced regression and matches the
            // workspace's most successful lineage.
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

/// TSS injection split (dense semantics, additive cap) — but with NO crop
/// filter: every tactical cell is in the candidate set by construction.
fn split_tactical(
    candidates: Vec<RustPriorCandidate>,
    tactical: &[HexCoord],
    nucleus: usize,
) -> (Vec<RustEdge>, Vec<RustPriorCandidate>, usize) {
    if tactical.is_empty() {
        return (Vec::new(), candidates, nucleus);
    }
    let tac: HashSet<PackedCoord> = tactical.iter().map(|c| pack_coord(*c)).collect();
    let mut by_prior: Vec<usize> = (0..candidates.len()).collect();
    by_prior.sort_by(|&a, &c| {
        candidates[c]
            .prior
            .partial_cmp(&candidates[a].prior)
            .unwrap_or(Ordering::Equal)
    });
    let nucleus_set: HashSet<PackedCoord> = by_prior
        .iter()
        .take(nucleus)
        .map(|&i| candidates[i].action_id)
        .collect();
    let mut forced = Vec::new();
    let mut rest = Vec::with_capacity(candidates.len());
    let mut extra_beyond_nucleus = 0usize;
    for candidate in candidates {
        if tac.contains(&candidate.action_id) {
            if !nucleus_set.contains(&candidate.action_id) {
                extra_beyond_nucleus += 1;
            }
            let mut edge = candidate.into_edge();
            edge.forced = true;
            forced.push(edge);
        } else {
            rest.push(candidate);
        }
    }
    let cap = nucleus + extra_beyond_nucleus;
    (forced, rest, cap)
}

fn owned_with_injection_from_eval(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    widening: Widening,
    tactical: &[HexCoord],
) -> RustNode {
    let nucleus = nucleus_count_pairs(&evaluation.priors, widening);
    let mut candidates: Vec<RustPriorCandidate> = evaluation
        .priors
        .iter()
        .map(|&(action_id, prior)| RustPriorCandidate { action_id, prior })
        .collect();
    candidates.reverse();
    let (edges, rest, max_eligible_children) = split_tactical(candidates, tactical, nucleus);
    RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges,
        priors: NodePriors::Owned(rest),
        max_eligible_children,
    }
}

fn shared_from_cache(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: Arc<RustEvaluation>,
    widening: Widening,
) -> RustNode {
    let max_eligible_children = nucleus_count_pairs(&evaluation.priors, widening);
    RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges: Vec::new(),
        priors: NodePriors::Shared(evaluation),
        max_eligible_children,
    }
}

fn owned_root_from_evaluation(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    root_policy_temperature: Option<f32>,
    root_noise: Option<RootDirichletNoise>,
    widening: Widening,
    tss_enabled: bool,
) -> PyResult<RustNode> {
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
        apply_root_policy_temperature_to(&mut candidates, temperature);
    }
    normalize_candidate_priors(&mut candidates)?;
    if let Some(noise) = root_noise {
        apply_dirichlet_noise(&mut candidates, noise);
    }
    candidates.sort_by(compare_prior_candidate);
    candidates.reverse();
    let nucleus = nucleus_count(&candidates, widening);
    let tactical = if tss_enabled {
        threats::tactical_cells(state)
    } else {
        Vec::new()
    };
    let (edges, mut candidates, max_eligible_children) =
        split_tactical(candidates, &tactical, nucleus);
    candidates.shrink_to_fit();
    Ok(RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges,
        priors: NodePriors::Owned(candidates),
        max_eligible_children,
    })
}

fn nucleus_count(candidates: &[RustPriorCandidate], widening: Widening) -> usize {
    nucleus_count_values(
        candidates.iter().map(|candidate| candidate.prior).collect(),
        widening,
    )
}

fn nucleus_count_pairs(priors: &[(PackedCoord, f32)], widening: Widening) -> usize {
    nucleus_count_values(priors.iter().map(|(_, prior)| *prior).collect(), widening)
}

fn nucleus_count_values(mut priors: Vec<f32>, widening: Widening) -> usize {
    let total = priors.len();
    if total == 0 {
        return 0;
    }
    let lo = widening.min_children.max(1).min(total);
    let hi = widening.max_children.max(lo).min(total);
    if lo >= hi {
        return hi;
    }
    priors.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let mut cumulative = 0.0f32;
    let mut count = 0usize;
    for prior in priors {
        cumulative += prior;
        count += 1;
        if cumulative >= widening.mass {
            break;
        }
    }
    count.clamp(lo, hi)
}

fn apply_root_policy_temperature_to(candidates: &mut [RustPriorCandidate], temperature: f32) {
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

pub fn terminal_value(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}

pub fn random_unit(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}
