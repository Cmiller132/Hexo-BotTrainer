//! Independent verifier and replayable certificate format for deep TSS.
//!
//! This module deliberately does not depend on `tss_solver`.  A certificate is
//! checked by replaying every represented move through `HexoState` and by using
//! only the shared, one-turn lambda-1 analysis for leaves and instant dispatch.

use std::mem::size_of;

use hexo_engine::{
    GameOutcome, HexCoord, HexoState as RustHexoState, Placement, Player, TurnPhase,
};

use crate::threats_shared;
use crate::tss_core::{CertVerify, ProofStatus};

/// Maximum number of arena nodes accepted from one certificate.
pub const MAX_CERT_NODES: usize = 100_000;
/// Maximum total number of explicitly represented universal edges.
pub const MAX_CERT_EDGES: usize = 1_000_000;
/// Maximum replay depth.  This is also a guard against adversarially deep DAGs.
pub const MAX_CERT_DEPTH: usize = 256;
/// Maximum number of root stones encoded in a certificate binding.
pub const MAX_CERT_ROOT_STONES: usize = 1_000_000;
/// Fixed verifier memo ceiling.  The verifier trait has no solve caps, so its
/// replay cache has its own hard byte bound rather than borrowing a solver TT
/// budget.
pub const MAX_VERIFY_MEMO_BYTES: usize = 64 << 20;
/// Number of rotations/reflections in the dihedral symmetry group of the hex.
pub const D6_SYMMETRY_COUNT: u8 = 12;

/// Compact arena index.  IDs always index `TssCertificate::nodes` directly.
pub type CertNodeId = u32;

/// Exact, history-independent binding of a certificate to its root position.
/// `occupancy` is lexicographically sorted by `(q, r)` and `owners[i]` owns
/// `occupancy[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootBinding {
    pub occupancy: Vec<HexCoord>,
    pub owners: Vec<Player>,
    pub current_player: Player,
    pub phase: TurnPhase,
    pub placements_made: u32,
    pub terminal: Option<GameOutcome>,
}

impl RootBinding {
    /// Construct the canonical full-position binding used by certificates.
    pub fn from_state(state: &RustHexoState) -> Self {
        let mut stones: Vec<(HexCoord, Player)> = state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .map(|coord| {
                let owner = state
                    .board()
                    .get(coord)
                    .expect("occupied_cells and Board::get must agree");
                (coord, owner)
            })
            .collect();
        stones.sort_by_key(|(coord, _)| coord_key(*coord));
        let (occupancy, owners) = stones.into_iter().unzip();
        Self {
            occupancy,
            owners,
            current_player: state.current_player(),
            phase: state.phase(),
            placements_made: state.placements_made(),
            terminal: state.terminal(),
        }
    }
}

/// One explicitly searched move at a universal (opponent) node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertEdge {
    pub mv: HexCoord,
    pub child: CertNodeId,
}

/// A proof arena node.  Nodes prove that `TssCertificate::claimant` wins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertNode {
    /// The replayed state is terminal with the claimant as winner.
    Terminal,
    /// The shared one-turn analysis proves the claimant wins from this state.
    Lambda1,
    /// A claimant move selecting one winning continuation.
    Choice { mv: HexCoord, child: CertNodeId },
    /// All listed opponent moves are replayed.  When `implicit_dispatch` is
    /// true, the unlisted complement must satisfy the forced-boundary rule and
    /// is individually checked by applying the move and invoking lambda-1.
    Universal {
        edges: Vec<CertEdge>,
        implicit_dispatch: bool,
    },
}

/// Replayable proof that `claimant` wins from the exactly bound root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TssCertificate {
    pub root: RootBinding,
    pub claimant: Player,
    pub root_node: CertNodeId,
    pub nodes: Vec<CertNode>,
}

/// Independent checker for [`TssCertificate`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TssVerifier;

impl CertVerify for TssVerifier {
    type Cert = TssCertificate;

    fn verify(&self, state: &RustHexoState, cert: &Self::Cert, claimed: ProofStatus) -> bool {
        if claimed == ProofStatus::Unknown || cert.root != RootBinding::from_state(state) {
            return false;
        }

        // Win/Loss is from the root side-to-move perspective, while the arena
        // itself is uniformly a winning strategy for the named claimant.
        let expected_claimant = match claimed {
            ProofStatus::Win => state.current_player(),
            ProofStatus::Loss => state.current_player().other(),
            ProofStatus::Unknown => return false,
        };
        if cert.claimant != expected_claimant || !validate_arena(cert) {
            return false;
        }

        let mut replay = state.clone();
        let Some(mut memo) = ReplayMemo::new(cert) else {
            return false;
        };
        verify_node(
            cert,
            cert.root_node,
            &mut replay,
            cert.claimant,
            0,
            &mut memo,
        )
    }
}

/// A full-position identity used only to make replay of shared DAG nodes both
/// bounded and sound.  A shared arena node may only denote one exact state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReplayKey {
    stones: Vec<(i16, i16, u8)>,
    current_player: u8,
    phase: PhaseKey,
    placements_made: u32,
    terminal: Option<(u8, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PhaseKey {
    Opening,
    FirstStone,
    SecondStone(i16, i16),
}

impl ReplayKey {
    fn from_state(state: &RustHexoState) -> Self {
        let binding = RootBinding::from_state(state);
        let stones = binding
            .occupancy
            .iter()
            .copied()
            .zip(binding.owners.iter().copied())
            .map(|(c, p)| (c.q, c.r, player_key(p)))
            .collect();
        let phase = match binding.phase {
            TurnPhase::Opening => PhaseKey::Opening,
            TurnPhase::FirstStone => PhaseKey::FirstStone,
            TurnPhase::SecondStone { first } => PhaseKey::SecondStone(first.q, first.r),
        };
        Self {
            stones,
            current_player: player_key(binding.current_player),
            phase,
            placements_made: binding.placements_made,
            terminal: binding
                .terminal
                .map(|outcome| (player_key(outcome.winner), outcome.placements)),
        }
    }

    fn heap_bytes(&self) -> usize {
        self.stones
            .capacity()
            .saturating_mul(size_of::<(i16, i16, u8)>())
            .saturating_add(32)
    }
}

struct ReplayMemo {
    /// A node is accepted only when every occurrence reaches the same full
    /// position.  Once verified there, subsequent transposition edges reuse it.
    states: Vec<Option<(ReplayKey, bool)>>,
    shared: Vec<bool>,
    bytes: usize,
}

impl ReplayMemo {
    fn new(cert: &TssCertificate) -> Option<Self> {
        let nodes = cert.nodes.len();
        let mut indegree = vec![0u32; nodes];
        for node in &cert.nodes {
            match node {
                CertNode::Choice { child, .. } => {
                    indegree[*child as usize] = indegree[*child as usize].saturating_add(1);
                }
                CertNode::Universal { edges, .. } => {
                    for edge in edges {
                        indegree[edge.child as usize] =
                            indegree[edge.child as usize].saturating_add(1);
                    }
                }
                CertNode::Terminal | CertNode::Lambda1 => {}
            }
        }
        let shared: Vec<bool> = indegree.into_iter().map(|count| count > 1).collect();
        let mut states = Vec::with_capacity(nodes);
        states.resize_with(nodes, || None);
        let bytes = states
            .capacity()
            .checked_mul(size_of::<Option<(ReplayKey, bool)>>())?
            .checked_add(shared.capacity().checked_mul(size_of::<bool>())?)?
            .checked_add(64)?;
        (bytes <= MAX_VERIFY_MEMO_BYTES).then_some(Self {
            states,
            shared,
            bytes,
        })
    }

    fn get(&self, id: CertNodeId) -> Option<&(ReplayKey, bool)> {
        if !self.shared.get(id as usize).copied().unwrap_or(false) {
            return None;
        }
        self.states.get(id as usize)?.as_ref()
    }

    fn is_shared(&self, id: CertNodeId) -> bool {
        self.shared.get(id as usize).copied().unwrap_or(false)
    }

    fn insert(&mut self, id: CertNodeId, key: ReplayKey, result: bool) -> bool {
        if !self.shared.get(id as usize).copied().unwrap_or(false) {
            return true;
        }
        let Some(slot) = self.states.get_mut(id as usize) else {
            return false;
        };
        if slot.is_some() {
            return false;
        }
        let new_bytes = self.bytes.saturating_add(key.heap_bytes());
        if new_bytes > MAX_VERIFY_MEMO_BYTES {
            return false;
        }
        *slot = Some((key, result));
        self.bytes = new_bytes;
        true
    }
}

fn verify_node(
    cert: &TssCertificate,
    id: CertNodeId,
    state: &mut RustHexoState,
    claimant: Player,
    depth: usize,
    memo: &mut ReplayMemo,
) -> bool {
    if depth > MAX_CERT_DEPTH {
        return false;
    }

    let replay_key = memo.is_shared(id).then(|| ReplayKey::from_state(state));
    if let (Some(key), Some((seen, result))) = (replay_key.as_ref(), memo.get(id)) {
        return seen == key && *result;
    }

    // The graph is already known acyclic, so inserting after evaluation cannot
    // recurse back to this node.  Failed nodes are memoized as well.
    let node = &cert.nodes[id as usize];
    let result = match node {
        CertNode::Terminal => state
            .terminal()
            .is_some_and(|outcome| outcome.winner == claimant),
        CertNode::Lambda1 => lambda1_proves_claimant(state, claimant),
        CertNode::Choice { mv, child } => {
            state.current_player() == claimant
                && !state.is_terminal()
                && with_move(state, *mv, |child_state, _| {
                    verify_node(cert, *child, child_state, claimant, depth + 1, memo)
                })
        }
        CertNode::Universal {
            edges,
            implicit_dispatch,
        } => verify_universal(
            cert,
            state,
            claimant,
            edges,
            *implicit_dispatch,
            depth,
            memo,
        ),
    };
    if let Some(key) = replay_key {
        if !memo.insert(id, key, result) {
            return false;
        }
    }
    result
}

fn verify_universal(
    cert: &TssCertificate,
    state: &mut RustHexoState,
    claimant: Player,
    edges: &[CertEdge],
    implicit_dispatch: bool,
    depth: usize,
    memo: &mut ReplayMemo,
) -> bool {
    if state.is_terminal() || state.current_player() == claimant {
        return false;
    }

    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_key(*coord));

    // Duplicate and illegal explicit moves are rejected rather than silently
    // coalesced.  This also makes complement coverage unambiguous.
    let mut explicit_moves: Vec<HexCoord> = edges.iter().map(|edge| edge.mv).collect();
    explicit_moves.sort_by_key(|coord| coord_key(*coord));
    if explicit_moves.windows(2).any(|pair| pair[0] == pair[1])
        || explicit_moves.iter().any(|mv| {
            legal
                .binary_search_by_key(&coord_key(*mv), |c| coord_key(*c))
                .is_err()
        })
    {
        return false;
    }

    let boundary = dispatch_boundary(state, claimant);
    if implicit_dispatch && boundary.is_none() {
        // In particular, a spare-stone node may never advertise an implicit
        // complement even if this particular certificate happened to list all
        // of its legal moves.
        return false;
    }

    for edge in edges {
        if !with_move(state, edge.mv, |child_state, _| {
            verify_node(cert, edge.child, child_state, claimant, depth + 1, memo)
        }) {
            return false;
        }
    }

    for mv in legal {
        if explicit_moves
            .binary_search_by_key(&coord_key(mv), |c| coord_key(*c))
            .is_ok()
        {
            continue;
        }
        if !implicit_dispatch {
            return false;
        }
        let hitting_universe = boundary.as_ref().expect("checked above");
        if hitting_universe
            .binary_search_by_key(&coord_key(mv), |c| coord_key(*c))
            .is_ok()
        {
            // Every hitting-universe move is a genuine defense and must have
            // an explicit searched child.
            return false;
        }
        if !with_move(state, mv, |child_state, outcome| match outcome {
            Some(outcome) => outcome.winner == claimant,
            None => lambda1_proves_claimant(child_state, claimant),
        }) {
            return false;
        }
    }
    true
}

/// Return the independently collected hitting-cell universe exactly when the
/// parent is at the sound instant-dispatch boundary.
fn dispatch_boundary(state: &RustHexoState, claimant: Player) -> Option<Vec<HexCoord>> {
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let analysis = threats_shared::analyze(state);
    if analysis.opp_threat_count == 0
        || analysis.own_win_now
        || analysis.min_hitting_set != Some(analysis.b)
    {
        return None;
    }

    // At a universal node the claimant is the opponent of the mover.  Collect
    // its active-window empties directly from the engine, independently of any
    // solver candidate list or stored coverage claim.
    let mut cells = Vec::new();
    for (owner, entry) in state.board().windows().threats() {
        if owner == claimant {
            cells.extend(entry.empty_cells());
        }
    }
    cells.sort_by_key(|coord| coord_key(*coord));
    cells.dedup();
    (!cells.is_empty()).then_some(cells)
}

fn lambda1_proves_claimant(state: &RustHexoState, claimant: Player) -> bool {
    // `analyze` is a forward, one-turn argument; a terminal fact is represented
    // by `CertNode::Terminal` and must not be reinterpreted as a lambda leaf.
    // Its soundness contract is post-opening, so Opening is rejected even
    // though every reachable Opening state is currently threat-free.
    if state.is_terminal() || matches!(state.phase(), TurnPhase::Opening) {
        return false;
    }
    let Some(verdict) = threats_shared::analyze(state).verdict() else {
        return false;
    };
    let proved_winner = if verdict > 0.0 {
        state.current_player()
    } else {
        state.current_player().other()
    };
    proved_winner == claimant
}

fn with_move(
    state: &mut RustHexoState,
    mv: HexCoord,
    verify_child: impl FnOnce(&mut RustHexoState, Option<GameOutcome>) -> bool,
) -> bool {
    let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
        return false;
    };
    let accepted = verify_child(state, result.outcome);
    state.undo(delta);
    accepted
}

fn validate_arena(cert: &TssCertificate) -> bool {
    if cert.root.occupancy.len() != cert.root.owners.len()
        || cert.root.occupancy.len() > MAX_CERT_ROOT_STONES
        || cert.nodes.is_empty()
        || cert.nodes.len() > MAX_CERT_NODES
        || cert.root_node as usize >= cert.nodes.len()
    {
        return false;
    }

    let mut edge_count = 0usize;
    for node in &cert.nodes {
        match node {
            CertNode::Choice { child, .. } => {
                if *child as usize >= cert.nodes.len() {
                    return false;
                }
            }
            CertNode::Universal { edges, .. } => {
                edge_count = match edge_count.checked_add(edges.len()) {
                    Some(count) if count <= MAX_CERT_EDGES => count,
                    _ => return false,
                };
                if edges
                    .iter()
                    .any(|edge| edge.child as usize >= cert.nodes.len())
                {
                    return false;
                }
                let mut moves: Vec<_> = edges.iter().map(|edge| edge.mv).collect();
                moves.sort_by_key(|coord| coord_key(*coord));
                if moves.windows(2).any(|pair| pair[0] == pair[1]) {
                    return false;
                }
            }
            CertNode::Terminal | CertNode::Lambda1 => {}
        }
    }

    // Three-colour DFS over the entire arena catches cycles even in components
    // unreachable from the declared root.
    let mut colours = vec![0u8; cert.nodes.len()];
    for start in 0..cert.nodes.len() {
        if colours[start] == 0 && !acyclic_from(cert, start, &mut colours) {
            return false;
        }
    }

    // A separate reachability pass rejects every orphan, including an acyclic
    // but otherwise well-formed component.
    let mut seen = vec![false; cert.nodes.len()];
    let mut stack = vec![cert.root_node as usize];
    while let Some(id) = stack.pop() {
        if seen[id] {
            continue;
        }
        seen[id] = true;
        push_children(&cert.nodes[id], &mut stack);
    }
    seen.into_iter().all(|reachable| reachable)
}

fn acyclic_from(cert: &TssCertificate, start: usize, colours: &mut [u8]) -> bool {
    // `(node, exiting)` avoids verifier call-stack exhaustion on malformed
    // certificates while retaining ordinary three-colour DFS semantics.
    let mut stack = vec![(start, false)];
    while let Some((id, exiting)) = stack.pop() {
        if exiting {
            colours[id] = 2;
            continue;
        }
        match colours[id] {
            1 => return false,
            2 => continue,
            _ => {}
        }
        colours[id] = 1;
        stack.push((id, true));
        let mut children = Vec::new();
        push_children(&cert.nodes[id], &mut children);
        // Reverse only to preserve certificate order in the conceptual DFS.
        for child in children.into_iter().rev() {
            match colours[child] {
                1 => return false,
                0 => stack.push((child, false)),
                _ => {}
            }
        }
    }
    true
}

fn push_children(node: &CertNode, out: &mut Vec<usize>) {
    match node {
        CertNode::Choice { child, .. } => out.push(*child as usize),
        CertNode::Universal { edges, .. } => {
            out.extend(edges.iter().map(|edge| edge.child as usize));
        }
        CertNode::Terminal | CertNode::Lambda1 => {}
    }
}

#[inline]
fn coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

#[inline]
fn player_key(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

/// Apply one of the 12 D6 symmetries to an axial coordinate.
///
/// IDs `0..=5` are rotations by repeated `(-r, q+r)`.  IDs `6..=11`
/// first reflect by `(q, -q-r)` and then apply the corresponding rotation.
/// `None` is returned for an invalid ID or if an intermediate coordinate does
/// not fit in the engine's `i16` representation.
pub fn d6_transform_coord(coord: HexCoord, symmetry: u8) -> Option<HexCoord> {
    if symmetry >= D6_SYMMETRY_COUNT {
        return None;
    }
    let mut q = i32::from(coord.q);
    let mut r = i32::from(coord.r);
    if symmetry >= 6 {
        r = q.checked_neg()?.checked_sub(r)?;
        i16::try_from(r).ok()?;
    }
    for _ in 0..(symmetry % 6) {
        let next_q = r.checked_neg()?;
        let next_r = q.checked_add(r)?;
        i16::try_from(next_q).ok()?;
        i16::try_from(next_r).ok()?;
        q = next_q;
        r = next_r;
    }
    Some(HexCoord {
        q: i16::try_from(q).ok()?,
        r: i16::try_from(r).ok()?,
    })
}

/// Remap every coordinate in a certificate under one D6 symmetry.
/// Arena IDs, player identities, counts, and terminal facts are invariant.
pub fn d6_remap_certificate(cert: &TssCertificate, symmetry: u8) -> Option<TssCertificate> {
    if symmetry >= D6_SYMMETRY_COUNT {
        return None;
    }

    let mut stones: Vec<(HexCoord, Player)> = cert
        .root
        .occupancy
        .iter()
        .copied()
        .zip(cert.root.owners.iter().copied())
        .map(|(coord, owner)| Some((d6_transform_coord(coord, symmetry)?, owner)))
        .collect::<Option<_>>()?;
    if stones.len() != cert.root.occupancy.len()
        || cert.root.occupancy.len() != cert.root.owners.len()
    {
        return None;
    }
    stones.sort_by_key(|(coord, _)| coord_key(*coord));
    if stones.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return None;
    }
    let (occupancy, owners) = stones.into_iter().unzip();
    let phase = match cert.root.phase {
        TurnPhase::Opening => TurnPhase::Opening,
        TurnPhase::FirstStone => TurnPhase::FirstStone,
        TurnPhase::SecondStone { first } => TurnPhase::SecondStone {
            first: d6_transform_coord(first, symmetry)?,
        },
    };
    let root = RootBinding {
        occupancy,
        owners,
        current_player: cert.root.current_player,
        phase,
        placements_made: cert.root.placements_made,
        terminal: cert.root.terminal,
    };

    let nodes = cert
        .nodes
        .iter()
        .map(|node| match node {
            CertNode::Terminal => Some(CertNode::Terminal),
            CertNode::Lambda1 => Some(CertNode::Lambda1),
            CertNode::Choice { mv, child } => Some(CertNode::Choice {
                mv: d6_transform_coord(*mv, symmetry)?,
                child: *child,
            }),
            CertNode::Universal {
                edges,
                implicit_dispatch,
            } => Some(CertNode::Universal {
                edges: edges
                    .iter()
                    .map(|edge| {
                        Some(CertEdge {
                            mv: d6_transform_coord(edge.mv, symmetry)?,
                            child: edge.child,
                        })
                    })
                    .collect::<Option<_>>()?,
                implicit_dispatch: *implicit_dispatch,
            }),
        })
        .collect::<Option<_>>()?;

    Some(TssCertificate {
        root,
        claimant: cert.claimant,
        root_node: cert.root_node,
        nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::apply_placement;

    fn terminal_player0_state(sequence_symmetry: u8) -> RustHexoState {
        let sequence = [
            HexCoord::new(0, 0),
            HexCoord::new(0, 1),
            HexCoord::new(0, 2),
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(1, 1),
            HexCoord::new(1, 2),
            HexCoord::new(3, 0),
            HexCoord::new(4, 0),
            HexCoord::new(2, 1),
            HexCoord::new(2, 2),
            HexCoord::new(5, 0),
        ];
        let mut state = RustHexoState::new();
        for coord in sequence {
            let coord = d6_transform_coord(coord, sequence_symmetry).unwrap();
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        assert_eq!(
            state.terminal().map(|outcome| outcome.winner),
            Some(Player::Player0)
        );
        state
    }

    fn terminal_cert(state: &RustHexoState) -> TssCertificate {
        TssCertificate {
            root: RootBinding::from_state(state),
            claimant: Player::Player0,
            root_node: 0,
            nodes: vec![CertNode::Terminal],
        }
    }

    #[test]
    fn terminal_certificate_is_bound_to_status_and_exact_root() {
        let state = terminal_player0_state(0);
        let cert = terminal_cert(&state);
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Unknown));

        let mut corrupt = cert.clone();
        corrupt.root.placements_made -= 1;
        assert!(!TssVerifier.verify(&state, &corrupt, ProofStatus::Win));
    }

    #[test]
    fn arena_rejects_orphans_cycles_and_invalid_ids() {
        let state = terminal_player0_state(0);
        let base = terminal_cert(&state);

        let mut orphan = base.clone();
        orphan.nodes.push(CertNode::Terminal);
        assert!(!TssVerifier.verify(&state, &orphan, ProofStatus::Win));

        let mut cyclic = base.clone();
        cyclic.root_node = 1;
        cyclic.nodes.push(CertNode::Choice {
            mv: HexCoord::ZERO,
            child: 1,
        });
        assert!(!TssVerifier.verify(&state, &cyclic, ProofStatus::Win));

        let mut invalid = base;
        invalid.root_node = 99;
        assert!(!TssVerifier.verify(&state, &invalid, ProofStatus::Win));
    }

    #[test]
    fn opening_lambda_and_oversized_certificate_are_rejected() {
        let opening = RustHexoState::new();
        let lambda = TssCertificate {
            root: RootBinding::from_state(&opening),
            claimant: Player::Player0,
            root_node: 0,
            nodes: vec![CertNode::Lambda1],
        };
        assert!(!TssVerifier.verify(&opening, &lambda, ProofStatus::Win));

        let terminal = terminal_player0_state(0);
        let mut oversized = terminal_cert(&terminal);
        oversized.nodes = vec![CertNode::Terminal; MAX_CERT_NODES + 1];
        assert!(!TssVerifier.verify(&terminal, &oversized, ProofStatus::Win));
    }

    #[test]
    fn all_d6_remaps_replay_against_transformed_roots() {
        let state = terminal_player0_state(0);
        let cert = terminal_cert(&state);
        let probe = HexCoord::new(2, 1);
        let mut images = Vec::new();
        for symmetry in 0..D6_SYMMETRY_COUNT {
            images.push(d6_transform_coord(probe, symmetry).unwrap());
            let transformed_state = terminal_player0_state(symmetry);
            let transformed_cert = d6_remap_certificate(&cert, symmetry).unwrap();
            assert_eq!(
                transformed_cert.root,
                RootBinding::from_state(&transformed_state)
            );
            assert!(TssVerifier.verify(&transformed_state, &transformed_cert, ProofStatus::Win));
        }
        images.sort_by_key(|coord| coord_key(*coord));
        images.dedup();
        assert_eq!(images.len(), D6_SYMMETRY_COUNT as usize);
        assert!(d6_transform_coord(probe, D6_SYMMETRY_COUNT).is_none());
    }
}
