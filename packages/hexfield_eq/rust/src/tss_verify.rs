//! Independent verifier and replayable certificate format for deep TSS.
//!
//! This module deliberately does not depend on `tss_solver`.  A certificate is
//! checked by replaying every represented move through `HexoState` and by using
//! only the shared, one-turn lambda-1 analysis for leaves and instant dispatch.

use std::mem::size_of;

use hexo_engine::{
    hex_distance, Axis, GameOutcome, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowKey,
};

use crate::threats_shared;
use crate::tss_core::{CertVerify, ProofStatus};

/// Maximum number of arena nodes accepted from one certificate.
pub const MAX_CERT_NODES: usize = 100_000;
/// Maximum total number of explicitly represented universal edges.
pub const MAX_CERT_EDGES: usize = 1_000_000;
/// Maximum total witness identities carried by typed leaves. Window keys are
/// compact, but LOSS families are attacker-controlled certificate data.
pub const MAX_CERT_WITNESSES: usize = 1_000_000;
pub const MAX_CERT_COMMUTATIONS: usize = 1_000_000;
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

/// P3 same-turn commutation evidence attached to the turn-start Universal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CertCommutation {
    pub first: HexCoord,
    pub omitted_second: HexCoord,
    pub first_child: CertNodeId,
    pub mirror_child: CertNodeId,
}

/// Horizon-dependent data carried by a defender zone node.  `d` is evidence
/// only: the verifier always recomputes the exact remaining defender budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneInfo {
    pub d: u32,
    /// Semantic deadline against which this D-dependent zone was built.
    pub build_horizon: u32,
}

/// A proof arena node.  Nodes prove that `TssCertificate::claimant` wins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertNode {
    /// Claimant placement that completes its named window immediately.
    OrCompletion {
        mv: HexCoord,
        witness: WindowKey,
        completion_ply: u32,
    },
    /// Claimant-to-move lambda-1 win with exact count/budget evidence.
    Win {
        witness: WindowKey,
        count: u8,
        budget: u8,
        resolution_ply: u32,
    },
    /// Defender-to-move adaptive lambda-1 loss contract.
    Loss {
        witnesses: Vec<WindowKey>,
        resolution_ply: u32,
    },
    /// A claimant move selecting one winning continuation.
    Choice { mv: HexCoord, child: CertNodeId },
    /// All listed opponent moves are replayed. When `implicit_dispatch` is
    /// true, every extendable-hit kernel cell must be represented explicitly
    /// or by a parent-validated same-turn commutation. The unrepresented
    /// complement is individually checked by the debug oracle by applying the
    /// move and invoking lambda-1.
    Universal {
        edges: Vec<CertEdge>,
        implicit_dispatch: bool,
        zone: Option<ZoneInfo>,
        commutations: Vec<CertCommutation>,
    },
}

/// Replayable proof that `claimant` wins from the exactly bound root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TssCertificate {
    pub root: RootBinding,
    pub claimant: Player,
    pub root_node: CertNodeId,
    pub nodes: Vec<CertNode>,
    /// Caller-supplied absolute deadline.  The verifier derives the
    /// certificate's actual T as the maximum exact leaf resolution and merely
    /// checks that derived value against this external cap.
    pub semantic_horizon: u32,
}

/// Independent checker for [`TssCertificate`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TssVerifier;

#[cfg(test)]
impl TssVerifier {
    pub(crate) fn verify_with_dispatch_oracle(
        &self,
        state: &RustHexoState,
        cert: &TssCertificate,
        claimed: ProofStatus,
    ) -> bool {
        verify_certificate(state, cert, claimed, true)
    }
}

impl CertVerify for TssVerifier {
    type Cert = TssCertificate;

    fn verify(&self, state: &RustHexoState, cert: &Self::Cert, claimed: ProofStatus) -> bool {
        verify_certificate(state, cert, claimed, false)
    }
}

fn verify_certificate(
    state: &RustHexoState,
    cert: &TssCertificate,
    claimed: ProofStatus,
    dispatch_oracle: bool,
) -> bool {
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

    let Some(meta) = certificate_metadata(cert) else {
        return false;
    };
    if meta.derived_t > cert.semantic_horizon
        || meta
            .zone_build_t
            .is_some_and(|build_t| meta.derived_t > build_t)
        || (meta.has_zone && state.is_terminal())
    {
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
        dispatch_oracle,
        &meta,
        &[],
    )
}

struct CertificateMetadata {
    derived_t: u32,
    has_zone: bool,
    zone_build_t: Option<u32>,
    cores: Vec<Vec<HexCoord>>,
    root_stones: Vec<HexCoord>,
}

fn certificate_metadata(cert: &TssCertificate) -> Option<CertificateMetadata> {
    let mut derived_t = 0u32;
    let mut has_zone = false;
    let mut zone_build_t: Option<u32> = None;
    for node in &cert.nodes {
        match node {
            CertNode::OrCompletion { completion_ply, .. } => {
                derived_t = derived_t.max(*completion_ply);
            }
            CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
                derived_t = derived_t.max(*resolution_ply);
            }
            CertNode::Universal { zone, .. } => {
                has_zone |= zone.is_some();
                if let Some(zone) = zone {
                    zone_build_t = Some(
                        zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                    );
                }
            }
            CertNode::Choice { .. } => {}
        }
    }
    let mut cores = vec![None; cert.nodes.len()];
    // Depth-bounded like `verify_node`: metadata construction must never
    // out-resource the replay it precedes (a valid acyclic million-node Choice
    // chain would otherwise overflow the stack here before verification could
    // reject it at MAX_CERT_DEPTH).
    fn build(
        cert: &TssCertificate,
        id: CertNodeId,
        memo: &mut [Option<Vec<HexCoord>>],
        depth: usize,
    ) -> Option<Vec<HexCoord>> {
        if depth > MAX_CERT_DEPTH {
            return None;
        }
        if let Some(core) = memo.get(id as usize)?.as_ref() {
            return Some(core.clone());
        }
        let mut core = Vec::new();
        match cert.nodes.get(id as usize)? {
            CertNode::OrCompletion { mv, witness, .. } => {
                core.push(*mv);
                core.extend(witness.cells());
            }
            CertNode::Win { witness, .. } => core.extend(witness.cells()),
            CertNode::Loss { witnesses, .. } => {
                for witness in witnesses {
                    core.extend(witness.cells());
                }
            }
            CertNode::Choice { mv, child } => {
                core.push(*mv);
                core.extend(build(cert, *child, memo, depth + 1)?);
            }
            CertNode::Universal { edges, .. } => {
                for edge in edges {
                    core.extend(build(cert, edge.child, memo, depth + 1)?);
                }
            }
        }
        core.sort_by_key(|coord| coord_key(*coord));
        core.dedup();
        memo[id as usize] = Some(core.clone());
        Some(core)
    }
    build(cert, cert.root_node, &mut cores, 0)?;
    let cores = cores.into_iter().collect::<Option<Vec<_>>>()?;
    Some(CertificateMetadata {
        derived_t,
        has_zone,
        zone_build_t,
        cores,
        root_stones: cert.root.occupancy.clone(),
    })
}

/// Cheap structural preflight used by the solver wrapper before verification.
/// This does not establish truth; it only derives the certificate's exact
/// semantic deadline and whether any AND node used the zone theorem.
pub(crate) fn certificate_horizon_preflight(cert: &TssCertificate) -> Option<(u32, bool)> {
    certificate_metadata(cert).map(|meta| (meta.derived_t, meta.has_zone))
}

/// A full-position identity used only to make replay of shared DAG nodes both
/// bounded and sound.  A shared arena node may only denote one exact state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReplayKey {
    stones: Vec<(i16, i16, u8)>,
    /// Same-position Universal nodes can have different obligations when a
    /// parent P3 commutation supplies an omitted reply. Bind that context into
    /// memo identity so a permissive occurrence cannot discharge a stricter
    /// one.
    allowed_commuted: Vec<(i16, i16)>,
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
    fn from_state_with_allowed(state: &RustHexoState, allowed_commuted: &[HexCoord]) -> Self {
        let binding = RootBinding::from_state(state);
        let stones = binding
            .occupancy
            .iter()
            .copied()
            .zip(binding.owners.iter().copied())
            .map(|(c, p)| (c.q, c.r, player_key(p)))
            .collect();
        let mut allowed_commuted = allowed_commuted
            .iter()
            .map(|coord| (coord.q, coord.r))
            .collect::<Vec<_>>();
        allowed_commuted.sort_unstable();
        let phase = match binding.phase {
            TurnPhase::Opening => PhaseKey::Opening,
            TurnPhase::FirstStone => PhaseKey::FirstStone,
            TurnPhase::SecondStone { first } => PhaseKey::SecondStone(first.q, first.r),
        };
        Self {
            stones,
            allowed_commuted,
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
            .saturating_add(
                self.allowed_commuted
                    .capacity()
                    .saturating_mul(size_of::<(i16, i16)>()),
            )
            .saturating_add(usize::from(!self.allowed_commuted.is_empty()).saturating_mul(32))
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
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
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
    dispatch_oracle: bool,
    meta: &CertificateMetadata,
    allowed_commuted: &[HexCoord],
) -> bool {
    if depth > MAX_CERT_DEPTH {
        return false;
    }

    let replay_key = memo
        .is_shared(id)
        .then(|| ReplayKey::from_state_with_allowed(state, allowed_commuted));
    if let (Some(key), Some((seen, result))) = (replay_key.as_ref(), memo.get(id)) {
        return seen == key && *result;
    }

    // The graph is already known acyclic, so inserting after evaluation cannot
    // recurse back to this node.  Failed nodes are memoized as well.
    let node = &cert.nodes[id as usize];
    let result = match node {
        CertNode::OrCompletion {
            mv,
            witness,
            completion_ply,
        } => verify_or_completion(state, claimant, *mv, *witness, *completion_ply, meta),
        CertNode::Win {
            witness,
            count,
            budget,
            resolution_ply,
        } => verify_win_leaf(
            state,
            claimant,
            *witness,
            *count,
            *budget,
            *resolution_ply,
            meta,
        ),
        CertNode::Loss {
            witnesses,
            resolution_ply,
        } => verify_loss_leaf(state, claimant, witnesses, *resolution_ply, meta),
        CertNode::Choice { mv, child } => {
            state.current_player() == claimant
                && !state.is_terminal()
                && attacker_placement_wf(state, claimant, *mv, meta)
                && with_move(state, *mv, |child_state, outcome| {
                    if outcome.is_some() {
                        return false;
                    }
                    verify_node(
                        cert,
                        *child,
                        child_state,
                        claimant,
                        depth + 1,
                        memo,
                        dispatch_oracle,
                        meta,
                        &[],
                    )
                })
        }
        CertNode::Universal {
            edges,
            implicit_dispatch,
            zone,
            commutations,
        } => verify_universal(
            cert,
            state,
            claimant,
            edges,
            *implicit_dispatch,
            *zone,
            commutations,
            depth,
            memo,
            dispatch_oracle,
            meta,
            id,
            allowed_commuted,
        ),
    };
    if let Some(key) = replay_key {
        if !memo.insert(id, key, result) {
            return false;
        }
    }
    result
}

fn window_entry(state: &RustHexoState, key: WindowKey) -> Option<hexo_engine::WindowEntry> {
    state
        .board()
        .windows()
        .entries()
        .find(|entry| entry.key() == key)
}

fn attacker_placement_wf(
    state: &RustHexoState,
    claimant: Player,
    mv: HexCoord,
    meta: &CertificateMetadata,
) -> bool {
    state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|stone| state.board().get(*stone) == Some(claimant))
        .chain(meta.root_stones.iter().copied())
        .any(|anchor| hex_distance(anchor, mv) <= 8)
}

fn verify_or_completion(
    state: &mut RustHexoState,
    claimant: Player,
    mv: HexCoord,
    witness: WindowKey,
    completion_ply: u32,
    meta: &CertificateMetadata,
) -> bool {
    state.current_player() == claimant
        && !state.is_terminal()
        && witness.contains(mv)
        && attacker_placement_wf(state, claimant, mv, meta)
        && completion_ply == state.placements_made().saturating_add(1)
        && completion_ply <= meta.derived_t
        && with_move(state, mv, |child_state, outcome| {
            outcome.is_some_and(|outcome| outcome.winner == claimant)
                && window_entry(child_state, witness).is_some_and(|entry| {
                    entry.count(claimant) == 6 && entry.count(claimant.other()) == 0
                })
        })
}

fn verify_win_leaf(
    state: &RustHexoState,
    claimant: Player,
    witness: WindowKey,
    count: u8,
    budget: u8,
    resolution_ply: u32,
    meta: &CertificateMetadata,
) -> bool {
    if state.is_terminal() || state.current_player() != claimant {
        return false;
    }
    let actual_budget = threats_shared::placements_remaining(state);
    let Some(entry) = window_entry(state, witness) else {
        return false;
    };
    let expected_resolution = match count {
        5 => state.placements_made().saturating_add(1),
        4 if actual_budget == 2 => state.placements_made().saturating_add(2),
        _ => return false,
    };
    budget == actual_budget
        && entry.count(claimant) == count
        && entry.count(claimant.other()) == 0
        && resolution_ply == expected_resolution
        && resolution_ply <= meta.derived_t
        && entry
            .empty_cells()
            .into_iter()
            .all(|mv| attacker_placement_wf(state, claimant, mv, meta))
}

fn family_hitting_exceeds(witnesses: &[Vec<HexCoord>], b: u8) -> bool {
    let mut universe = witnesses.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| coord_key(*coord));
    universe.dedup();
    if witnesses.iter().any(Vec::is_empty) {
        return true;
    }
    if b >= 1
        && universe
            .iter()
            .any(|a| witnesses.iter().all(|w| w.contains(a)))
    {
        return false;
    }
    if b >= 2 {
        for (index, a) in universe.iter().enumerate() {
            for b_cell in &universe[index..] {
                if witnesses
                    .iter()
                    .all(|w| w.contains(a) || w.contains(b_cell))
                {
                    return false;
                }
            }
        }
    }
    true
}

fn verify_loss_leaf(
    state: &RustHexoState,
    claimant: Player,
    witnesses: &[WindowKey],
    resolution_ply: u32,
    meta: &CertificateMetadata,
) -> bool {
    if state.is_terminal() || state.current_player() == claimant || witnesses.is_empty() {
        return false;
    }
    let analysis = threats_shared::analyze(state);
    if analysis.own_win_now {
        return false;
    }
    let mut empties = Vec::with_capacity(witnesses.len());
    for &key in witnesses {
        let Some(entry) = window_entry(state, key) else {
            return false;
        };
        if entry.active_player() != Some(claimant) || entry.count(claimant) < 4 {
            return false;
        }
        let cells = entry.empty_cells();
        if !cells
            .iter()
            .copied()
            .all(|mv| attacker_placement_wf(state, claimant, mv, meta))
        {
            return false;
        }
        empties.push(cells);
    }
    let expected = state
        .placements_made()
        .saturating_add(u32::from(analysis.b))
        .saturating_add(2);
    family_hitting_exceeds(&empties, analysis.b)
        && resolution_ply == expected
        && resolution_ply <= meta.derived_t
}

fn validate_commutations(
    cert: &TssCertificate,
    state: &mut RustHexoState,
    edges: &[CertEdge],
    commutations: &[CertCommutation],
) -> Option<Vec<(HexCoord, Vec<HexCoord>)>> {
    if commutations.is_empty() {
        return Some(Vec::new());
    }
    if !matches!(state.phase(), TurnPhase::FirstStone)
        || threats_shared::placements_remaining(state) != 2
    {
        return None;
    }
    let mut grouped: Vec<(HexCoord, Vec<HexCoord>)> = Vec::new();
    let mut seen = Vec::new();
    for item in commutations {
        if coord_key(item.omitted_second) >= coord_key(item.first)
            || seen.contains(&(item.first, item.omitted_second))
        {
            return None;
        }
        seen.push((item.first, item.omitted_second));
        let first_edge = edges.iter().find(|edge| edge.mv == item.first)?;
        let mirror_edge = edges.iter().find(|edge| edge.mv == item.omitted_second)?;
        if first_edge.child != item.first_child || mirror_edge.child != item.mirror_child {
            return None;
        }
        let CertNode::Universal {
            edges: first_replies,
            ..
        } = cert.nodes.get(item.first_child as usize)?
        else {
            return None;
        };
        let CertNode::Universal {
            edges: mirror_replies,
            ..
        } = cert.nodes.get(item.mirror_child as usize)?
        else {
            return None;
        };
        if first_replies
            .iter()
            .any(|edge| edge.mv == item.omitted_second)
            || !mirror_replies.iter().any(|edge| edge.mv == item.first)
        {
            return None;
        }
        for mv in [item.first, item.omitted_second] {
            if !with_move(state, mv, |child, outcome| {
                outcome.is_none() && matches!(child.phase(), TurnPhase::SecondStone { .. })
            }) {
                return None;
            }
        }
        let pair_outcome = |a: HexCoord, b: HexCoord| {
            let mut replay = state.clone();
            let first = replay.apply_with_delta(Placement { coord: a }).ok()?.0;
            if first.outcome.is_some() {
                return None;
            }
            Some(
                replay
                    .apply_with_delta(Placement { coord: b })
                    .ok()?
                    .0
                    .outcome,
            )
        };
        let forward = pair_outcome(item.first, item.omitted_second)?;
        let mirror = pair_outcome(item.omitted_second, item.first)?;
        if forward != mirror {
            return None;
        }
        match grouped.iter_mut().find(|(first, _)| *first == item.first) {
            Some((_, omitted)) => omitted.push(item.omitted_second),
            None => grouped.push((item.first, vec![item.omitted_second])),
        }
    }
    for (_, omitted) in &mut grouped {
        omitted.sort_by_key(|coord| coord_key(*coord));
    }
    Some(grouped)
}

fn verify_universal(
    cert: &TssCertificate,
    state: &mut RustHexoState,
    claimant: Player,
    edges: &[CertEdge],
    implicit_dispatch: bool,
    zone: Option<ZoneInfo>,
    commutations: &[CertCommutation],
    depth: usize,
    memo: &mut ReplayMemo,
    dispatch_oracle: bool,
    meta: &CertificateMetadata,
    node_id: CertNodeId,
    allowed_commuted: &[HexCoord],
) -> bool {
    if state.is_terminal()
        || state.current_player() == claimant
        || threats_shared::analyze(state).own_win_now
    {
        return false;
    }
    // Duplicate explicit moves are rejected rather than silently coalesced.
    // Legality is independently established by the replay below.
    let mut explicit_moves: Vec<HexCoord> = edges.iter().map(|edge| edge.mv).collect();
    explicit_moves.sort_by_key(|coord| coord_key(*coord));
    if explicit_moves.windows(2).any(|pair| pair[0] == pair[1]) {
        return false;
    }
    let mut allowed = allowed_commuted.to_vec();
    allowed.sort_by_key(|coord| coord_key(*coord));
    if allowed.windows(2).any(|pair| pair[0] == pair[1])
        || allowed.iter().any(|mv| explicit_moves.contains(mv))
        || allowed.iter().any(|mv| {
            let mut probe = state.clone();
            probe.apply_with_delta(Placement { coord: *mv }).is_err()
        })
    {
        return false;
    }
    let mut represented = explicit_moves.clone();
    represented.extend(allowed.iter().copied());
    represented.sort_by_key(|coord| coord_key(*coord));
    // Empty nested nodes are meaningful only when a validated parent
    // commutation supplies their entire same-turn obligation.
    if represented.is_empty() || (zone.is_some() && !allowed.is_empty()) {
        return false;
    }

    let boundary = dispatch_boundary(state, claimant);
    let child_commutations = match validate_commutations(cert, state, edges, commutations) {
        Some(value) => value,
        None => return false,
    };
    if implicit_dispatch && boundary.is_none() {
        // In particular, a spare-stone node may never advertise an implicit
        // complement even if this particular certificate happened to list all
        // of its legal moves.
        return false;
    }

    if implicit_dispatch {
        // T6 kernel staple: at a checked post-opening ¬own_win_now, tau=b
        // boundary, only cells extendable to a size-b transversal can retain a
        // live defense. Requiring the independently derived kernel is the
        // complete obligation; certificates may explicitly prove any superset.
        let kernel = boundary.as_ref().expect("checked above");
        if kernel.iter().any(|mv| {
            represented
                .binary_search_by_key(&coord_key(*mv), |c| coord_key(*c))
                .is_err()
        }) {
            return false;
        }
    } else if let Some(zone) = zone {
        if !verify_zone_node(state, claimant, &explicit_moves, zone, meta, node_id) {
            return false;
        }
    } else {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        legal.sort_by_key(|coord| coord_key(*coord));
        if represented != legal {
            return false;
        }
    }

    for edge in edges {
        if !with_move(state, edge.mv, |child_state, outcome| {
            if outcome.is_some() {
                return false;
            }
            verify_node(
                cert,
                edge.child,
                child_state,
                claimant,
                depth + 1,
                memo,
                dispatch_oracle,
                meta,
                child_commutations
                    .iter()
                    .find(|(first, _)| *first == edge.mv)
                    .map(|(_, omitted)| omitted.as_slice())
                    .unwrap_or(&[]),
            )
        }) {
            return false;
        }
    }

    if implicit_dispatch && dispatch_oracle {
        // Paired debug oracle: validate every omitted nonkernel move with the
        // per-move lambda-1 staple. Production never enters this arm.
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        let kernel = boundary.as_ref().expect("checked above");
        for mv in legal {
            if represented
                .binary_search_by_key(&coord_key(mv), |c| coord_key(*c))
                .is_ok()
            {
                continue;
            }
            if kernel
                .binary_search_by_key(&coord_key(mv), |c| coord_key(*c))
                .is_ok()
                || !with_move(state, mv, |child_state, outcome| match outcome {
                    Some(outcome) => outcome.winner == claimant,
                    None => lambda1_proves_claimant(child_state, claimant),
                })
            {
                return false;
            }
        }
    }
    true
}

fn remaining_defender_placements(
    state: &RustHexoState,
    claimant: Player,
    horizon: u32,
) -> Option<u32> {
    // A valid zone node exists only for defender budgets 0..=5 (the solver
    // takes the full legal set at d >= 6), so once the count passes that band
    // the exact value can no longer matter — bail rather than walk a
    // corrupted/adversarial horizon (`u32::MAX` would otherwise spin billions
    // of iterations outside every node cap). `None` rejects the node, which
    // is always sound.
    const DEFENDER_BUDGET_BAIL: u32 = 8;
    let mut ply = state.placements_made();
    if horizon < ply {
        return None;
    }
    let mut player = state.current_player();
    let mut phase = state.phase();
    let mut count = 0u32;
    while ply < horizon {
        if player != claimant {
            count = count.checked_add(1)?;
            if count > DEFENDER_BUDGET_BAIL {
                return None;
            }
        }
        match phase {
            TurnPhase::Opening => {
                player = player.other();
                phase = TurnPhase::FirstStone;
            }
            TurnPhase::FirstStone => {
                phase = TurnPhase::SecondStone {
                    first: HexCoord::ZERO,
                }
            }
            TurnPhase::SecondStone { .. } => {
                player = player.other();
                phase = TurnPhase::FirstStone;
            }
        }
        ply = ply.checked_add(1)?;
    }
    Some(count)
}

fn set_contains(sorted: &[HexCoord], coord: HexCoord) -> bool {
    sorted
        .binary_search_by_key(&coord_key(coord), |candidate| coord_key(*candidate))
        .is_ok()
}

fn verify_zone_node(
    state: &RustHexoState,
    claimant: Player,
    explicit: &[HexCoord],
    zone: ZoneInfo,
    meta: &CertificateMetadata,
    node_id: CertNodeId,
) -> bool {
    if matches!(state.phase(), TurnPhase::Opening)
        || state.current_player() == claimant
        || threats_shared::analyze(state).own_win_now
        || explicit.is_empty()
    {
        return false;
    }
    let Some(d) = remaining_defender_placements(state, claimant, meta.derived_t) else {
        return false;
    };
    if zone.d != d {
        return false;
    }

    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_key(*coord));
    if explicit.iter().any(|mv| !set_contains(&legal, *mv)) {
        return false;
    }
    if d >= 6 {
        return legal.iter().all(|mv| set_contains(explicit, *mv));
    }

    // Z1: all current claimant-threat empties are searched.
    for (owner, entry) in state.board().windows().threats() {
        if owner == claimant
            && entry
                .empty_cells()
                .into_iter()
                .any(|mv| set_contains(&legal, mv) && !set_contains(explicit, mv))
        {
            return false;
        }
    }

    // Z2: final-DAG core plus the defender completion guard.
    let Some(core) = meta.cores.get(node_id as usize) else {
        return false;
    };
    let defender = claimant.other();
    let mut protected = core.clone();
    for entry in state.board().windows().entries() {
        if entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)).saturating_add(d) >= 6
        {
            protected.extend(entry.empty_cells());
        }
    }
    protected.sort_by_key(|coord| coord_key(*coord));
    protected.dedup();
    for &cell in &protected {
        if set_contains(&legal, cell) && !set_contains(explicit, cell) {
            return false;
        }
    }

    // Z5: if protected territory is not yet legal or occupied, search every
    // currently legal cell within the full 8*D chain radius.
    let stones = state.board().occupied_cells();
    let pending = protected
        .iter()
        .copied()
        .filter(|cell| !set_contains(&legal, *cell) && !stones.contains(cell))
        .collect::<Vec<_>>();
    if !pending.is_empty() {
        let radius = i32::try_from(d.saturating_mul(8)).unwrap_or(i32::MAX);
        for &cell in &legal {
            if pending
                .iter()
                .any(|target| i32::from(hex_distance(cell, *target)) <= radius)
                && !set_contains(explicit, cell)
            {
                return false;
            }
        }
    }
    true
}

/// Return the independently derived extendable-hit kernel exactly when the
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

    // At a universal node the claimant is the opponent of the mover. Collect
    // its active-window empty sets directly from the engine, independently of
    // any solver candidate list or stored coverage claim.
    let family = state
        .board()
        .windows()
        .threats()
        .filter_map(|(owner, entry)| (owner == claimant).then(|| entry.empty_cells()))
        .collect::<Vec<_>>();
    let kernel = extendable_hit_kernel_for_family(&family, analysis.b);
    (!kernel.is_empty()).then_some(kernel)
}

fn extendable_hit_kernel_for_family(family: &[Vec<HexCoord>], budget: u8) -> Vec<HexCoord> {
    let mut universe = family.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| coord_key(*coord));
    universe.dedup();
    match budget {
        1 => universe
            .into_iter()
            .filter(|cell| family.iter().all(|threat| threat.contains(cell)))
            .collect(),
        2 => universe
            .iter()
            .copied()
            .filter(|cell| {
                universe.iter().copied().any(|mate| {
                    mate != *cell
                        && family
                            .iter()
                            .all(|threat| threat.contains(cell) || threat.contains(&mate))
                })
            })
            .collect(),
        // Connect-6 dispatch boundaries only have one or two placements. Keep
        // an independently safe full-universe fallback for future phases.
        _ => universe,
    }
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
    let mut witness_count = 0usize;
    let mut commutation_count = 0usize;
    for node in &cert.nodes {
        match node {
            CertNode::Choice { child, .. } => {
                if *child as usize >= cert.nodes.len() {
                    return false;
                }
            }
            CertNode::Universal {
                edges,
                commutations,
                ..
            } => {
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
                commutation_count = match commutation_count.checked_add(commutations.len()) {
                    Some(count) if count <= MAX_CERT_COMMUTATIONS => count,
                    _ => return false,
                };
                if commutations.iter().any(|item| {
                    item.first_child as usize >= cert.nodes.len()
                        || item.mirror_child as usize >= cert.nodes.len()
                }) {
                    return false;
                }
                let mut moves: Vec<_> = edges.iter().map(|edge| edge.mv).collect();
                moves.sort_by_key(|coord| coord_key(*coord));
                if moves.windows(2).any(|pair| pair[0] == pair[1]) {
                    return false;
                }
            }
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => {
                witness_count = match witness_count.checked_add(1) {
                    Some(count) if count <= MAX_CERT_WITNESSES => count,
                    _ => return false,
                };
            }
            CertNode::Loss { witnesses, .. } => {
                witness_count = match witness_count.checked_add(witnesses.len()) {
                    Some(count) if count <= MAX_CERT_WITNESSES => count,
                    _ => return false,
                };
            }
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
        CertNode::Universal {
            edges,
            commutations,
            ..
        } => {
            out.extend(edges.iter().map(|edge| edge.child as usize));
            out.extend(
                commutations
                    .iter()
                    .flat_map(|item| [item.first_child as usize, item.mirror_child as usize]),
            );
        }
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
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

fn d6_transform_window(key: WindowKey, symmetry: u8) -> Option<WindowKey> {
    let first = d6_transform_coord(key.coord_at(0), symmetry)?;
    let second = d6_transform_coord(key.coord_at(1), symmetry)?;
    let dq = i32::from(second.q) - i32::from(first.q);
    let dr = i32::from(second.r) - i32::from(first.r);
    let axis = match (dq, dr) {
        (1, 0) => {
            return Some(WindowKey {
                start: first,
                axis: Axis::Q,
            })
        }
        (0, 1) => {
            return Some(WindowKey {
                start: first,
                axis: Axis::R,
            })
        }
        (1, -1) => {
            return Some(WindowKey {
                start: first,
                axis: Axis::QR,
            })
        }
        (-1, 0) => Axis::Q,
        (0, -1) => Axis::R,
        (-1, 1) => Axis::QR,
        _ => return None,
    };
    Some(WindowKey {
        start: d6_transform_coord(key.coord_at(5), symmetry)?,
        axis,
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
            CertNode::OrCompletion {
                mv,
                witness,
                completion_ply,
            } => Some(CertNode::OrCompletion {
                mv: d6_transform_coord(*mv, symmetry)?,
                witness: d6_transform_window(*witness, symmetry)?,
                completion_ply: *completion_ply,
            }),
            CertNode::Win {
                witness,
                count,
                budget,
                resolution_ply,
            } => Some(CertNode::Win {
                witness: d6_transform_window(*witness, symmetry)?,
                count: *count,
                budget: *budget,
                resolution_ply: *resolution_ply,
            }),
            CertNode::Loss {
                witnesses,
                resolution_ply,
            } => Some(CertNode::Loss {
                witnesses: witnesses
                    .iter()
                    .map(|key| d6_transform_window(*key, symmetry))
                    .collect::<Option<_>>()?,
                resolution_ply: *resolution_ply,
            }),
            CertNode::Choice { mv, child } => Some(CertNode::Choice {
                mv: d6_transform_coord(*mv, symmetry)?,
                child: *child,
            }),
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
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
                zone: *zone,
                commutations: commutations
                    .iter()
                    .map(|commutation| {
                        Some(CertCommutation {
                            first: d6_transform_coord(commutation.first, symmetry)?,
                            omitted_second: d6_transform_coord(
                                commutation.omitted_second,
                                symmetry,
                            )?,
                            first_child: commutation.first_child,
                            mirror_child: commutation.mirror_child,
                        })
                    })
                    .collect::<Option<_>>()?,
            }),
        })
        .collect::<Option<_>>()?;

    Some(TssCertificate {
        root,
        claimant: cert.claimant,
        root_node: cert.root_node,
        nodes,
        semantic_horizon: cert.semantic_horizon,
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

    fn win_now_state(sequence_symmetry: u8) -> RustHexoState {
        let sequence = [
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
            (8, 4),
            (10, 3),
        ];
        let mut state = RustHexoState::new();
        for (q, r) in sequence {
            let coord = d6_transform_coord(HexCoord::new(q, r), sequence_symmetry).unwrap();
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        assert!(!state.is_terminal());
        state
    }

    fn win_cert(state: &RustHexoState, symmetry: u8) -> TssCertificate {
        let witness = d6_transform_window(
            WindowKey {
                start: HexCoord::ZERO,
                axis: Axis::Q,
            },
            symmetry,
        )
        .unwrap();
        let resolution_ply = state.placements_made() + 1;
        TssCertificate {
            root: RootBinding::from_state(state),
            claimant: Player::Player0,
            root_node: 0,
            nodes: vec![CertNode::Win {
                witness,
                count: 5,
                budget: 2,
                resolution_ply,
            }],
            semantic_horizon: resolution_ply,
        }
    }

    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap();
        }
        state
    }

    fn commutation_fixture(
        state: &RustHexoState,
        first: HexCoord,
        omitted: HexCoord,
    ) -> (TssCertificate, Vec<CertEdge>, CertCommutation) {
        let leaf = CertNode::Win {
            witness: WindowKey {
                start: HexCoord::ZERO,
                axis: Axis::Q,
            },
            count: 5,
            budget: 2,
            resolution_ply: 1,
        };
        let first_child = CertNode::Universal {
            edges: vec![CertEdge {
                mv: HexCoord::new(99, 99),
                child: 2,
            }],
            implicit_dispatch: false,
            zone: None,
            commutations: Vec::new(),
        };
        let mirror_child = CertNode::Universal {
            edges: vec![CertEdge {
                mv: first,
                child: 2,
            }],
            implicit_dispatch: false,
            zone: None,
            commutations: Vec::new(),
        };
        let cert = TssCertificate {
            root: RootBinding::from_state(state),
            claimant: state.current_player().other(),
            root_node: 0,
            nodes: vec![first_child, mirror_child, leaf],
            semantic_horizon: u32::MAX,
        };
        let edges = vec![
            CertEdge {
                mv: first,
                child: 0,
            },
            CertEdge {
                mv: omitted,
                child: 1,
            },
        ];
        let item = CertCommutation {
            first,
            omitted_second: omitted,
            first_child: 0,
            mirror_child: 1,
        };
        (cert, edges, item)
    }

    fn xsnfyll_forced_defender_fixture() -> RustHexoState {
        let mut state = replay(&[
            (0, 0),
            (-1, 0),
            (1, -2),
            (-2, 0),
            (1, 0),
            (0, -2),
            (1, -3),
            (0, -3),
            (2, -5),
            (2, -4),
            (1, -4),
            (3, -4),
            (3, -2),
        ]);
        for coord in [HexCoord::new(-1, -1), HexCoord::new(1, -5)] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    #[test]
    fn verifier_extendable_hit_kernel_matches_k1_and_k2_algebra() {
        let a = HexCoord::new(0, 0);
        let b = HexCoord::new(1, 0);
        let c = HexCoord::new(2, 0);
        let z = HexCoord::new(3, 0);
        assert_eq!(
            extendable_hit_kernel_for_family(&[vec![a, b], vec![a, c]], 1),
            vec![a]
        );
        assert_eq!(
            extendable_hit_kernel_for_family(&[vec![a, z], vec![a, c], vec![b]], 2),
            vec![a, b]
        );
    }

    #[test]
    fn verifier_derives_xsnfyll_k2_independently() {
        let state = xsnfyll_forced_defender_fixture();
        assert_eq!(state.current_player(), Player::Player0);
        assert_eq!(
            dispatch_boundary(&state, Player::Player1),
            Some(vec![
                HexCoord::new(1, -6),
                HexCoord::new(3, -5),
                HexCoord::new(4, -6),
            ])
        );
    }

    #[test]
    fn implicit_nested_universal_accepts_only_complete_commuted_kernel() {
        let mut state = xsnfyll_forced_defender_fixture();
        let claimant = Player::Player1;
        let first = dispatch_boundary(&state, claimant).unwrap()[0];
        apply_placement(&mut state, Placement { coord: first }).unwrap();
        assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }));
        let kernel = dispatch_boundary(&state, claimant).expect("forced K1 after a K2 first reply");

        let cert = TssCertificate {
            root: RootBinding::from_state(&state),
            claimant,
            root_node: 0,
            nodes: Vec::new(),
            semantic_horizon: u32::MAX,
        };
        let meta = CertificateMetadata {
            derived_t: 0,
            has_zone: false,
            zone_build_t: None,
            cores: vec![Vec::new()],
            root_stones: RootBinding::from_state(&state).occupancy,
        };
        let verify = |allowed: &[HexCoord], zone: Option<ZoneInfo>, oracle: bool| {
            let mut replay = state.clone();
            let mut memo = ReplayMemo::new(&cert).unwrap();
            verify_universal(
                &cert,
                &mut replay,
                claimant,
                &[],
                true,
                zone,
                &[],
                0,
                &mut memo,
                oracle,
                &meta,
                0,
                allowed,
            )
        };
        assert!(verify(&kernel, None, false));
        assert!(verify(&kernel, None, true));
        assert!(!verify(&[], None, false));
        assert!(!verify(
            &kernel,
            Some(ZoneInfo {
                d: 1,
                build_horizon: state.placements_made() + 1,
            }),
            false,
        ));
    }

    #[test]
    fn replay_memo_key_binds_commuted_context() {
        let state = replay(&[(0, 0), (0, 8), (2, 7)]);
        let a = HexCoord::new(1, 0);
        let b = HexCoord::new(2, 0);
        assert_ne!(
            ReplayKey::from_state_with_allowed(&state, &[]),
            ReplayKey::from_state_with_allowed(&state, &[a]),
        );
        assert_eq!(
            ReplayKey::from_state_with_allowed(&state, &[a, b]),
            ReplayKey::from_state_with_allowed(&state, &[b, a]),
        );
    }

    #[test]
    fn p3_commutation_condition_matrix() {
        let quiet = replay(&[(0, 0), (0, 8), (2, 7)]);
        assert!(matches!(quiet.phase(), TurnPhase::FirstStone));
        let mut legal = Vec::new();
        quiet.write_legal_moves(&mut legal);
        legal.sort_by_key(|coord| coord_key(*coord));
        let omitted = legal[0];
        let first = *legal.last().unwrap();
        let (cert, edges, item) = commutation_fixture(&quiet, first, omitted);
        assert!(validate_commutations(&cert, &mut quiet.clone(), &edges, &[item]).is_some());

        let mut absent_mirror = cert.clone();
        {
            let CertNode::Universal { edges, .. } = &mut absent_mirror.nodes[1] else {
                unreachable!()
            };
            edges.clear();
        }
        assert!(validate_commutations(
            &absent_mirror,
            &mut quiet.clone(),
            &edges_for(first, omitted),
            &[item],
        )
        .is_none());

        let reversed = CertCommutation {
            first: omitted,
            omitted_second: first,
            first_child: 1,
            mirror_child: 0,
        };
        assert!(validate_commutations(&cert, &mut quiet.clone(), &edges, &[reversed]).is_none());

        let wrong_binding = CertCommutation {
            mirror_child: 0,
            ..item
        };
        assert!(
            validate_commutations(&cert, &mut quiet.clone(), &edges, &[wrong_binding]).is_none()
        );

        // Newly legal second cells have no turn-start mirror and therefore
        // cannot be commutation-omitted.
        let mut after_first = quiet.clone();
        apply_placement(&mut after_first, Placement { coord: first }).unwrap();
        let mut after_legal = Vec::new();
        after_first.write_legal_moves(&mut after_legal);
        if let Some(new_cell) = after_legal.into_iter().find(|mv| !legal.contains(mv)) {
            let (new_cert, new_edges, new_item) = commutation_fixture(&quiet, first, new_cell);
            assert!(
                validate_commutations(&new_cert, &mut quiet.clone(), &new_edges, &[new_item],)
                    .is_none()
            );
        }

        // Singleton-terminal prefixes are excluded.
        let singleton = win_now_state(0);
        let winning = HexCoord::new(5, 0);
        let mut singleton_legal = Vec::new();
        singleton.write_legal_moves(&mut singleton_legal);
        let other = singleton_legal
            .into_iter()
            .find(|mv| *mv != winning)
            .unwrap();
        let (single_cert, single_edges, single_item) =
            commutation_fixture(&singleton, winning, other);
        assert!(validate_commutations(
            &single_cert,
            &mut singleton.clone(),
            &single_edges,
            &[single_item],
        )
        .is_none());

        // Joint-second wins are allowed: neither singleton terminates, and
        // the verifier binds the materialized mirror rather than comparing
        // terminal PositionKeys.
        let joint = replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 1),
            (8, 4),
            (10, 3),
        ]);
        let (joint_cert, joint_edges, joint_item) =
            commutation_fixture(&joint, HexCoord::new(5, 0), HexCoord::new(4, 0));
        assert!(validate_commutations(
            &joint_cert,
            &mut joint.clone(),
            &joint_edges,
            &[joint_item],
        )
        .is_some());
    }

    #[test]
    fn zone_mutations_reject_d6_late_core_band_opening_and_own_win() {
        let quiet = replay(&[(0, 0), (0, 8), (2, 7)]);
        let claimant = quiet.current_player().other();
        let mut legal = Vec::new();
        quiet.write_legal_moves(&mut legal);
        legal.sort_by_key(|coord| coord_key(*coord));
        let omitted = legal[0];

        let short_t = quiet.placements_made() + 1;
        let short_d = remaining_defender_placements(&quiet, claimant, short_t).unwrap();
        assert_eq!(short_d, 1);
        let stones = quiet.board().occupied_cells();
        let pending = (-16..=16)
            .flat_map(|dq| (-16..=16).map(move |dr| HexCoord::new(omitted.q + dq, omitted.r + dr)))
            .find(|cell| {
                !legal.contains(cell) && !stones.contains(cell) && hex_distance(*cell, omitted) <= 8
            })
            .expect("quiet frontier needs a nonlegal cell within the Z5 radius");
        let band_meta = CertificateMetadata {
            derived_t: short_t,
            has_zone: true,
            zone_build_t: Some(short_t),
            cores: vec![vec![pending]],
            root_stones: RootBinding::from_state(&quiet).occupancy,
        };
        let explicit = legal.iter().copied().skip(1).collect::<Vec<_>>();
        assert!(verify_zone_node(
            &quiet,
            claimant,
            &legal,
            ZoneInfo {
                d: short_d,
                build_horizon: short_t,
            },
            &band_meta,
            0,
        ));
        assert!(!verify_zone_node(
            &quiet,
            claimant,
            &explicit,
            ZoneInfo {
                d: short_d,
                build_horizon: short_t,
            },
            &band_meta,
            0,
        ));

        let late_core_meta = CertificateMetadata {
            cores: vec![vec![omitted]],
            ..band_meta
        };
        assert!(!verify_zone_node(
            &quiet,
            claimant,
            &explicit,
            ZoneInfo {
                d: short_d,
                build_horizon: short_t,
            },
            &late_core_meta,
            0,
        ));

        let long_t = quiet.placements_made() + 12;
        let long_d = remaining_defender_placements(&quiet, claimant, long_t).unwrap();
        assert!(long_d >= 6);
        let d6_meta = CertificateMetadata {
            derived_t: long_t,
            has_zone: true,
            zone_build_t: Some(long_t),
            cores: vec![Vec::new()],
            root_stones: RootBinding::from_state(&quiet).occupancy,
        };
        assert!(verify_zone_node(
            &quiet,
            claimant,
            &legal,
            ZoneInfo {
                d: long_d,
                build_horizon: long_t,
            },
            &d6_meta,
            0,
        ));
        assert!(!verify_zone_node(
            &quiet,
            claimant,
            &explicit,
            ZoneInfo {
                d: long_d,
                build_horizon: long_t,
            },
            &d6_meta,
            0,
        ));

        let opening = RustHexoState::new();
        let opening_meta = CertificateMetadata {
            derived_t: 1,
            has_zone: true,
            zone_build_t: Some(1),
            cores: vec![Vec::new()],
            root_stones: Vec::new(),
        };
        assert!(!verify_zone_node(
            &opening,
            opening.current_player().other(),
            &[HexCoord::ZERO],
            ZoneInfo {
                d: 1,
                build_horizon: 1,
            },
            &opening_meta,
            0,
        ));

        let own_win = win_now_state(0);
        let own_meta = CertificateMetadata {
            derived_t: own_win.placements_made() + 1,
            has_zone: true,
            zone_build_t: Some(own_win.placements_made() + 1),
            cores: vec![Vec::new()],
            root_stones: RootBinding::from_state(&own_win).occupancy,
        };
        assert!(!verify_zone_node(
            &own_win,
            own_win.current_player().other(),
            &[HexCoord::new(5, 0)],
            ZoneInfo {
                d: 1,
                build_horizon: own_win.placements_made() + 1,
            },
            &own_meta,
            0,
        ));
    }

    fn edges_for(first: HexCoord, omitted: HexCoord) -> Vec<CertEdge> {
        vec![
            CertEdge {
                mv: first,
                child: 0,
            },
            CertEdge {
                mv: omitted,
                child: 1,
            },
        ]
    }

    #[test]
    fn typed_win_certificate_is_bound_to_status_and_exact_root() {
        let state = win_now_state(0);
        let cert = win_cert(&state, 0);
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Unknown));

        let mut corrupt = cert.clone();
        corrupt.root.placements_made -= 1;
        assert!(!TssVerifier.verify(&state, &corrupt, ProofStatus::Win));

        let mut wrong_count = cert.clone();
        let CertNode::Win { count, .. } = &mut wrong_count.nodes[0] else {
            unreachable!()
        };
        *count = 4;
        assert!(!TssVerifier.verify(&state, &wrong_count, ProofStatus::Win));

        let mut wrong_resolution = cert.clone();
        let CertNode::Win { resolution_ply, .. } = &mut wrong_resolution.nodes[0] else {
            unreachable!()
        };
        *resolution_ply += 1;
        wrong_resolution.semantic_horizon += 1;
        assert!(!TssVerifier.verify(&state, &wrong_resolution, ProofStatus::Win));
    }

    #[test]
    fn typed_completion_and_loss_mutations_are_rejected() {
        let state = win_now_state(0);
        let witness = WindowKey {
            start: HexCoord::ZERO,
            axis: Axis::Q,
        };
        let completion_ply = state.placements_made() + 1;
        let completion = TssCertificate {
            root: RootBinding::from_state(&state),
            claimant: state.current_player(),
            root_node: 0,
            nodes: vec![CertNode::OrCompletion {
                mv: HexCoord::new(5, 0),
                witness,
                completion_ply,
            }],
            semantic_horizon: completion_ply,
        };
        assert!(TssVerifier.verify(&state, &completion, ProofStatus::Win));

        let mut no_completion = completion.clone();
        let CertNode::OrCompletion { mv, .. } = &mut no_completion.nodes[0] else {
            unreachable!()
        };
        *mv = HexCoord::new(5, 1);
        assert!(!TssVerifier.verify(&state, &no_completion, ProofStatus::Win));

        let mut outside_witness = completion.clone();
        let CertNode::OrCompletion { witness, .. } = &mut outside_witness.nodes[0] else {
            unreachable!()
        };
        witness.start = HexCoord::new(0, 1);
        assert!(!TssVerifier.verify(&state, &outside_witness, ProofStatus::Win));

        let mut wrong_ply = completion.clone();
        let CertNode::OrCompletion { completion_ply, .. } = &mut wrong_ply.nodes[0] else {
            unreachable!()
        };
        *completion_ply += 1;
        wrong_ply.semantic_horizon += 1;
        assert!(!TssVerifier.verify(&state, &wrong_ply, ProofStatus::Win));

        let loss_state = replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (0, 4),
            (8, 4),
            (10, 3),
            (1, 4),
            (2, 4),
            (12, 2),
            (14, 1),
            (3, 4),
            (16, 0),
        ]);
        let claimant = loss_state.current_player().other();
        let mut witnesses = loss_state
            .board()
            .windows()
            .threats()
            .filter(|(owner, _)| *owner == claimant)
            .map(|(_, entry)| entry.key())
            .collect::<Vec<_>>();
        witnesses.sort_by_key(|key| (key.start.q, key.start.r, key.axis as u8));
        let b = threats_shared::placements_remaining(&loss_state);
        let resolution = loss_state.placements_made() + u32::from(b) + 2;
        let loss = TssCertificate {
            root: RootBinding::from_state(&loss_state),
            claimant,
            root_node: 0,
            nodes: vec![CertNode::Loss {
                witnesses,
                resolution_ply: resolution,
            }],
            semantic_horizon: resolution,
        };
        assert!(TssVerifier.verify(&loss_state, &loss, ProofStatus::Loss));

        let mut corrupt_family = loss.clone();
        let CertNode::Loss { witnesses, .. } = &mut corrupt_family.nodes[0] else {
            unreachable!()
        };
        witnesses.truncate(1);
        assert!(!TssVerifier.verify(&loss_state, &corrupt_family, ProofStatus::Loss));

        let mut corrupt_resolution = loss.clone();
        let CertNode::Loss { resolution_ply, .. } = &mut corrupt_resolution.nodes[0] else {
            unreachable!()
        };
        *resolution_ply -= 1;
        assert!(!TssVerifier.verify(&loss_state, &corrupt_resolution, ProofStatus::Loss));

        let mut external_horizon = loss.clone();
        external_horizon.semantic_horizon = resolution - 1;
        assert!(!TssVerifier.verify(&loss_state, &external_horizon, ProofStatus::Loss));
    }

    #[test]
    fn arena_rejects_orphans_cycles_and_invalid_ids() {
        let state = win_now_state(0);
        let base = win_cert(&state, 0);

        let mut orphan = base.clone();
        orphan.nodes.push(base.nodes[0].clone());
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
    fn opening_terminal_root_and_oversized_certificate_are_rejected() {
        let opening = RustHexoState::new();
        let fake = TssCertificate {
            root: RootBinding::from_state(&opening),
            claimant: Player::Player0,
            root_node: 0,
            nodes: vec![CertNode::Win {
                witness: WindowKey {
                    start: HexCoord::ZERO,
                    axis: Axis::Q,
                },
                count: 5,
                budget: 1,
                resolution_ply: 1,
            }],
            semantic_horizon: 1,
        };
        assert!(!TssVerifier.verify(&opening, &fake, ProofStatus::Win));

        let terminal = terminal_player0_state(0);
        let mut terminal_root = fake.clone();
        terminal_root.root = RootBinding::from_state(&terminal);
        assert!(!TssVerifier.verify(&terminal, &terminal_root, ProofStatus::Win));

        let state = win_now_state(0);
        let mut oversized = win_cert(&state, 0);
        oversized.nodes = vec![oversized.nodes[0].clone(); MAX_CERT_NODES + 1];
        assert!(!TssVerifier.verify(&state, &oversized, ProofStatus::Win));
    }

    #[test]
    fn all_d6_remaps_replay_against_transformed_roots() {
        let state = win_now_state(0);
        let cert = win_cert(&state, 0);
        let probe = HexCoord::new(2, 1);
        let mut images = Vec::new();
        for symmetry in 0..D6_SYMMETRY_COUNT {
            images.push(d6_transform_coord(probe, symmetry).unwrap());
            let transformed_state = win_now_state(symmetry);
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
