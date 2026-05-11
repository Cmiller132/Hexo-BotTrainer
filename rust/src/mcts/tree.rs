//! Lightweight in-memory MCTS tree.
//!
//! Nodes are stored in a `Vec` and referenced by integer ids. Each edge is one
//! single placement action, matching Hexo's autoregressive turn model.

use crate::game::{HexCoord, Player, TurnPhase};

/// Index into `SearchTree.nodes`.
pub type NodeId = usize;

/// Directed action from a parent node to an optional child node.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Single-stone placement represented by this edge.
    pub action: HexCoord,
    /// Prior probability from the evaluator.
    pub prior: f32,
    /// Number of simulations that selected this edge.
    pub visits: u32,
    /// Sum of backed-up values from the parent node's perspective.
    pub value_sum: f32,
    /// Lazily-created child node reached after applying `action`.
    pub child: Option<NodeId>,
}

impl Edge {
    /// Create an unvisited action edge.
    pub fn new(action: HexCoord, prior: f32) -> Self {
        Self {
            action,
            prior,
            visits: 0,
            value_sum: 0.0,
            child: None,
        }
    }

    /// Mean backed-up value for this edge.
    pub fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// One MCTS node representing a complete Hexo state.
#[derive(Clone, Debug)]
pub struct Node {
    /// Hash of the represented state, useful for debugging/transpositions later.
    pub state_hash: u64,
    /// Player to act at this state.
    pub player_to_act: Player,
    /// Turn phase at this state.
    pub phase: TurnPhase,
    /// Number of simulations backed up into this node.
    pub visits: u32,
    /// Sum of node values from `player_to_act` perspective.
    pub value_sum: f32,
    /// Legal action edges. Empty until the node is expanded.
    pub edges: Vec<Edge>,
    /// True once legal actions and priors have been populated.
    pub expanded: bool,
}

impl Node {
    /// Create an unexpanded search node.
    pub fn new(state_hash: u64, player_to_act: Player, phase: TurnPhase) -> Self {
        Self {
            state_hash,
            player_to_act,
            phase,
            visits: 0,
            value_sum: 0.0,
            edges: Vec::new(),
            expanded: false,
        }
    }

    /// Mean value for the node's player-to-act.
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// Arena-style storage for all nodes in one MCTS run.
#[derive(Clone, Debug, Default)]
pub struct SearchTree {
    pub nodes: Vec<Node>,
}

impl SearchTree {
    /// Start a tree with a single root node.
    pub fn with_root(root: Node) -> Self {
        Self { nodes: vec![root] }
    }

    /// Push a new node and return its id.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
}
