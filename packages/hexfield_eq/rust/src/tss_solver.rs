//! Proof-carrying forced-tree solver.
//!
//! The search fixes a `claimant` player and constructs a winning strategy for
//! that identity.  Nodes owned by the claimant are existential; nodes owned by
//! the other player are universal.  This is deliberately not negamax: a
//! `FirstStone` placement leaves the same player to place `SecondStone`.
//!
//! The implementation is a deterministic, proof-number-ordered depth-first
//! AND/OR proof constructor.  It is equivalent to the proof side of df-pn for
//! the three-valued interface used here: the most promising (lowest initial
//! proof-number) OR child is expanded first, while every child of an AND node
//! must produce a proof.  Failure or any resource exhaustion is `Unknown`; a
//! failed restricted attack is never interpreted as a proof for the opponent.

use std::cmp::Reverse;
use std::mem::size_of;

use hexo_engine::{HexCoord, HexoState as RustHexoState, Placement, Player, TurnPhase};

use crate::threats_shared as threats;
use crate::tss_core::{DeepResult, DeepSolve, ProofStatus, SolveCaps, SolveStats};
use crate::tss_verify::{
    CertEdge, CertNode, CertNodeId, RootBinding, TssCertificate, MAX_CERT_DEPTH, MAX_CERT_EDGES,
    MAX_CERT_NODES, MAX_CERT_ROOT_STONES,
};

/// A second, fixed guard in addition to `SolveCaps::node_cap`.  It bounds stack
/// depth even when a caller supplies an accidentally enormous node cap.
const MAX_SEARCH_DEPTH: usize = MAX_CERT_DEPTH;

/// Conservative allocator-header charge used by the explicit TT accounting.
const ALLOC_OVERHEAD: usize = 32;
/// The direct table never reserves more than this many inline slots.
const MAX_TT_SLOTS: usize = 1 << 20;
/// Expected bytes per slot.  Entries with larger position keys simply make the
/// table stop accepting replacements before the caller's byte cap is crossed.
const TARGET_BYTES_PER_TT_SLOT: usize = 256;

/// Stateless public solver.  All mutable search state is allocated and freed
/// inside one `solve`, so interleaved calls cannot affect one another.
#[derive(Clone, Debug)]
pub(crate) struct TssSolver {
    tt_enabled: bool,
    hash_mask: u64,
}

impl Default for TssSolver {
    fn default() -> Self {
        Self {
            tt_enabled: true,
            hash_mask: u64::MAX,
        }
    }
}

impl TssSolver {
    #[cfg(test)]
    fn without_tt() -> Self {
        Self {
            tt_enabled: false,
            hash_mask: u64::MAX,
        }
    }

    /// Test hook: masking every hash to zero forces all positions into one
    /// bucket.  Full-key equality must still prevent a value-bearing false hit.
    #[cfg(test)]
    fn with_hash_mask(hash_mask: u64) -> Self {
        Self {
            tt_enabled: true,
            hash_mask,
        }
    }
}

impl DeepSolve for TssSolver {
    type Cert = TssCertificate;

    fn solve(&mut self, state: &RustHexoState, caps: &SolveCaps) -> DeepResult<Self::Cert> {
        if caps.node_cap == 0 || state.board().len() > MAX_CERT_ROOT_STONES {
            return unknown(SolveStats::default());
        }

        // A root lambda-one/terminal result is both common and symmetric
        // between the primal and dual claims.  Count it as one examined node,
        // then avoid spending half the budget merely to rediscover a LOSS.
        let mut stats = SolveStats {
            nodes: 1,
            ..SolveStats::default()
        };
        if let Some((claimant, leaf)) = immediate_winner(state) {
            let cert = TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node: 0,
                nodes: vec![leaf],
            };
            let status = status_for_claimant(state.current_player(), claimant);
            return DeepResult {
                status,
                cert: Some(cert),
                stats,
            };
        }

        // Reserve deterministic budgets for both directions.  A restricted OR
        // failure is not a LOSS, so the opponent gets an independent dual run.
        let remaining = caps.node_cap - 1;
        let primal_cap = (remaining + 1) / 2;
        let dual_cap = remaining / 2;
        let root_player = state.current_player();

        if primal_cap > 0 {
            let attempt = self.prove_for(state, root_player, primal_cap, caps.tt_bytes_cap);
            merge_stats(&mut stats, attempt.stats);
            if let Some(cert) = attempt.cert {
                return DeepResult {
                    status: ProofStatus::Win,
                    cert: Some(cert),
                    stats,
                };
            }
        }

        if dual_cap > 0 {
            let attempt = self.prove_for(state, root_player.other(), dual_cap, caps.tt_bytes_cap);
            merge_stats(&mut stats, attempt.stats);
            if let Some(cert) = attempt.cert {
                return DeepResult {
                    status: ProofStatus::Loss,
                    cert: Some(cert),
                    stats,
                };
            }
        }

        unknown(stats)
    }
}

impl TssSolver {
    fn prove_for(
        &self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        tt_bytes_cap: usize,
    ) -> AttemptResult {
        let mut work = state.clone();
        let entry_key = PositionKey::from_state(&work);
        let mut context = SearchContext::new(
            node_cap,
            if self.tt_enabled { tt_bytes_cap } else { 0 },
            self.hash_mask,
        );
        let proof = context.prove(&mut work, claimant, 0);

        // Every recursive path uses LIFO make/unmake, including cap exits.
        debug_assert_eq!(entry_key, PositionKey::from_state(&work));

        let stats = SolveStats {
            nodes: context.nodes,
            tt_hits: context.tt_hits,
            peak_tt_bytes: context.tt.peak_bytes as u64,
        };
        let cert = proof.and_then(|root| {
            compact_certificate(&context.arena, root).map(|(nodes, root_node)| TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node,
                nodes,
            })
        });
        AttemptResult { cert, stats }
    }
}

struct AttemptResult {
    cert: Option<TssCertificate>,
    stats: SolveStats,
}

fn unknown<C>(stats: SolveStats) -> DeepResult<C> {
    DeepResult {
        status: ProofStatus::Unknown,
        cert: None,
        stats,
    }
}

fn merge_stats(total: &mut SolveStats, part: SolveStats) {
    total.nodes = total.nodes.saturating_add(part.nodes);
    total.tt_hits = total.tt_hits.saturating_add(part.tt_hits);
    total.peak_tt_bytes = total.peak_tt_bytes.max(part.peak_tt_bytes);
}

fn status_for_claimant(root_player: Player, claimant: Player) -> ProofStatus {
    if root_player == claimant {
        ProofStatus::Win
    } else {
        ProofStatus::Loss
    }
}

/// Return the player proved to win at this node and the corresponding compact
/// leaf.  Lambda-one is intentionally unavailable at Opening (the shared
/// theorem is post-opening), although reachable Opening currently has no
/// threats and therefore cannot produce a verdict anyway.
fn immediate_winner(state: &RustHexoState) -> Option<(Player, CertNode)> {
    if let Some(outcome) = state.terminal() {
        return Some((outcome.winner, CertNode::Terminal));
    }
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let analysis = threats::analyze(state);
    if analysis.own_win_now {
        Some((state.current_player(), CertNode::Lambda1))
    } else if analysis.forced_loss() {
        Some((state.current_player().other(), CertNode::Lambda1))
    } else {
        None
    }
}

struct SearchContext {
    node_cap: u64,
    nodes: u64,
    tt_hits: u64,
    hit_limit: bool,
    arena: Vec<CertNode>,
    edge_count: usize,
    tt: BoundedTt,
}

impl SearchContext {
    fn new(node_cap: u64, tt_bytes_cap: usize, hash_mask: u64) -> Self {
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt: BoundedTt::new(tt_bytes_cap, hash_mask),
        }
    }

    fn prove(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        depth: usize,
    ) -> Option<CertNodeId> {
        if depth > MAX_SEARCH_DEPTH || self.nodes >= self.node_cap {
            self.hit_limit = true;
            return None;
        }
        self.nodes += 1;

        let key = PositionKey::from_state(state);
        if let Some(node) = self.tt.lookup(&key, claimant) {
            if (node as usize) < self.arena.len() {
                self.tt_hits += 1;
                return Some(node);
            }
        }

        if let Some((winner, leaf)) = immediate_winner(state) {
            if winner != claimant {
                return None;
            }
            let node = self.alloc_node(leaf, 0)?;
            self.tt.insert(key, claimant, node);
            return Some(node);
        }

        let node = if state.current_player() == claimant {
            self.prove_choice(state, claimant, depth)?
        } else {
            self.prove_universal(state, claimant, depth)?
        };
        self.tt.insert(key, claimant, node);
        Some(node)
    }

    fn prove_choice(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        depth: usize,
    ) -> Option<CertNodeId> {
        // Descending line count is the static proof-number initialization:
        // completions before four-builds before three-builds.  The coordinate
        // tie break makes the order independent of WindowStore hash iteration.
        for candidate in ordered_threat_creating_moves(state, claimant) {
            let Ok((_result, delta)) = state.apply_with_delta(Placement {
                coord: candidate.coord,
            }) else {
                continue;
            };
            let child = self.prove(state, claimant, depth + 1);
            state.undo(delta);

            if let Some(child) = child {
                return self.alloc_node(
                    CertNode::Choice {
                        mv: candidate.coord,
                        child,
                    },
                    1,
                );
            }
            if self.hit_limit {
                return None;
            }
        }
        // Exhausting a restricted attacker set only says that this attack
        // generator found no proof.  It is deliberately not a disproof.
        None
    }

    fn prove_universal(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        depth: usize,
    ) -> Option<CertNodeId> {
        let mut all_legal = Vec::new();
        state.write_legal_moves(&mut all_legal);
        if all_legal.is_empty() {
            return None;
        }

        let analysis = threats::analyze(state);
        let initially_forced = !matches!(state.phase(), TurnPhase::Opening)
            && analysis.opp_threat_count > 0
            && !analysis.own_win_now
            && analysis.min_hitting_set == Some(analysis.b);

        let mut implicit_dispatch = initially_forced;
        let mut explicit = if initially_forced {
            hitting_universe(state, claimant)
        } else {
            all_legal.clone()
        };
        explicit.sort_by_key(|coord| (coord.q, coord.r));
        explicit.dedup();

        // Staple-check every proposed omission before relying on L1.  If a
        // future engine/threat change invalidates the premise, fall back to a
        // fully explicit universal node; never silently drop that placement.
        if implicit_dispatch {
            for &mv in &all_legal {
                if explicit
                    .binary_search_by_key(&(mv.q, mv.r), |c| (c.q, c.r))
                    .is_ok()
                {
                    continue;
                }
                let Ok((_result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                    implicit_dispatch = false;
                    break;
                };
                let dispatch_winner = immediate_winner(state).map(|(winner, _)| winner);
                state.undo(delta);
                if dispatch_winner != Some(claimant) {
                    implicit_dispatch = false;
                    break;
                }
            }
            if !implicit_dispatch {
                explicit = all_legal.clone();
            }
        }

        let frame = canonical_frame(state);
        explicit.sort_by_key(|coord| canonical_coord_key(frame, *coord));

        let mut edges = Vec::with_capacity(explicit.len());
        for mv in explicit {
            let Ok((_result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                return None;
            };
            let child = self.prove(state, claimant, depth + 1);
            state.undo(delta);
            let child = child?; // Unknown poisons the universal claim.
            edges.push(CertEdge { mv, child });
        }

        let explicit_edge_count = edges.len();

        self.alloc_node(
            CertNode::Universal {
                edges,
                implicit_dispatch,
            },
            explicit_edge_count,
        )
    }

    fn alloc_node(&mut self, node: CertNode, added_edges: usize) -> Option<CertNodeId> {
        if self.arena.len() >= MAX_CERT_NODES
            || self.edge_count.saturating_add(added_edges) > MAX_CERT_EDGES
        {
            self.hit_limit = true;
            return None;
        }
        let id = u32::try_from(self.arena.len()).ok()?;
        self.arena.push(node);
        self.edge_count += added_edges;
        Some(id)
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    coord: HexCoord,
    /// Maximum pre-placement stone count among active claimant windows that
    /// this coordinate extends.  Larger means a lower initial proof number.
    strength: u8,
    priority_class: u8,
    child_threats: usize,
}

/// Exact OR restriction: every returned placement changes an active claimant
/// length-six window with at least three stones into a >=4 threat (or a win).
/// Omitting all other claimant moves can only miss a winning proof.
fn threat_creating_moves(state: &RustHexoState, claimant: Player) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.active_player() != Some(claimant) {
            continue;
        }
        let strength = entry.count(claimant);
        if strength < 3 {
            continue;
        }
        for coord in entry.empty_cells() {
            if let Some(existing) = candidates.iter_mut().find(|item| item.coord == coord) {
                existing.strength = existing.strength.max(strength);
            } else {
                candidates.push(Candidate {
                    coord,
                    strength,
                    priority_class: u8::MAX,
                    child_threats: 0,
                });
            }
        }
    }
    candidates.sort_by_key(|item| (Reverse(item.strength), item.coord.q, item.coord.r));
    candidates
}

/// Proof-number initialization for OR children.  This probes each already-
/// selected forcing move with make/unmake, but never adds or removes a move:
/// immediate proofs rank first, then children that hand the opponent a fully
/// forced defense, then mid-turn builds ordered by resulting threat count.
fn ordered_threat_creating_moves(state: &mut RustHexoState, claimant: Player) -> Vec<Candidate> {
    let mut candidates = threat_creating_moves(state, claimant);
    for candidate in &mut candidates {
        let Ok((_result, delta)) = state.apply_with_delta(Placement {
            coord: candidate.coord,
        }) else {
            candidate.priority_class = u8::MAX;
            continue;
        };
        let immediate = immediate_winner(state).map(|(winner, _)| winner) == Some(claimant);
        candidate.child_threats = state
            .board()
            .windows()
            .threats()
            .filter(|(owner, _)| *owner == claimant)
            .count();
        candidate.priority_class = if immediate {
            0
        } else if state.current_player() != claimant {
            let analysis = threats::analyze(state);
            if analysis.opp_threat_count > 0
                && !analysis.own_win_now
                && analysis.min_hitting_set == Some(analysis.b)
            {
                1
            } else if analysis.opp_threat_count > 0 {
                3
            } else {
                4
            }
        } else {
            2
        };
        state.undo(delta);
    }
    let frame = canonical_frame(state);
    candidates.sort_by_key(|item| {
        let canonical = canonical_coord_key(frame, item.coord);
        (
            item.priority_class,
            Reverse(item.child_threats),
            Reverse(item.strength),
            canonical.0,
            canonical.1,
        )
    });
    candidates
}

/// Union of empties of every live claimant threat.  At a defender node this is
/// the L1 hitting-cell universe, not a selected minimal hitting set.
fn hitting_universe(state: &RustHexoState, claimant: Player) -> Vec<HexCoord> {
    let mut cells = Vec::new();
    for (owner, entry) in state.board().windows().threats() {
        if owner == claimant {
            cells.extend(entry.empty_cells());
        }
    }
    cells.sort_by_key(|coord| (coord.q, coord.r));
    cells.dedup();
    cells
}

/// Choose the lexicographically least D6 image of the full semantic position.
/// Search ties are compared in this frame, so rotating/reflection-transforming
/// an input cannot change which proof-cost class is expanded first merely due
/// to raw `(q,r)` order.  The TT remains uncanonicalized and still uses exact
/// raw-position equality.
fn canonical_frame(state: &RustHexoState) -> u8 {
    let mut best: Option<((u8, i32, i32), Vec<(i32, i32, u8)>, u8)> = None;
    for symmetry in 0..12u8 {
        let phase = match state.phase() {
            TurnPhase::Opening => (0, 0, 0),
            TurnPhase::FirstStone => (1, 0, 0),
            TurnPhase::SecondStone { first } => {
                let (q, r) = d6_coord_i32(first, symmetry);
                (2, q, r)
            }
        };
        let mut stones: Vec<_> = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                let (q, r) = d6_coord_i32(coord, symmetry);
                let owner = state.board().get(coord).expect("occupied cell has owner");
                (q, r, player_code(owner))
            })
            .collect();
        stones.sort_unstable();
        let candidate = (phase, stones, symmetry);
        if best.as_ref().is_none_or(|(best_phase, best_stones, _)| {
            (&candidate.0, &candidate.1) < (best_phase, best_stones)
        }) {
            best = Some(candidate);
        }
    }
    best.expect("D6 contains identity").2
}

fn canonical_coord_key(frame: u8, coord: HexCoord) -> (i32, i32) {
    d6_coord_i32(coord, frame)
}

fn d6_coord_i32(coord: HexCoord, symmetry: u8) -> (i32, i32) {
    let mut q = i32::from(coord.q);
    let mut r = i32::from(coord.r);
    if symmetry >= 6 {
        r = -q - r;
    }
    for _ in 0..(symmetry % 6) {
        (q, r) = (-r, q + r);
    }
    (q, r)
}

// === Full-key transposition table ==========================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KeyStone {
    q: i16,
    r: i16,
    owner: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyPhase {
    Opening,
    FirstStone,
    SecondStone { q: i16, r: i16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KeyTerminal {
    winner: u8,
    placements: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PositionKey {
    stones: Vec<KeyStone>,
    current_player: u8,
    phase: KeyPhase,
    placements_made: u32,
    terminal: Option<KeyTerminal>,
}

impl PositionKey {
    fn from_state(state: &RustHexoState) -> Self {
        let mut stones: Vec<KeyStone> = state
            .board()
            .occupied_cells()
            .iter()
            .map(|coord| KeyStone {
                q: coord.q,
                r: coord.r,
                owner: player_code(state.board().get(*coord).expect("occupied cell has owner")),
            })
            .collect();
        stones.sort_by_key(|stone| (stone.q, stone.r, stone.owner));
        let phase = match state.phase() {
            TurnPhase::Opening => KeyPhase::Opening,
            TurnPhase::FirstStone => KeyPhase::FirstStone,
            TurnPhase::SecondStone { first } => KeyPhase::SecondStone {
                q: first.q,
                r: first.r,
            },
        };
        let terminal = state.terminal().map(|outcome| KeyTerminal {
            winner: player_code(outcome.winner),
            placements: outcome.placements,
        });
        Self {
            stones,
            current_player: player_code(state.current_player()),
            phase,
            placements_made: state.placements_made(),
            terminal,
        }
    }

    fn stable_hash(&self) -> u64 {
        // FNV-1a is used only for bucket selection.  Equality below, never this
        // 64-bit value, authorizes a proof hit.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for &byte in bytes {
                *hash ^= u64::from(byte);
                *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        feed(&mut hash, &[self.current_player]);
        feed(&mut hash, &self.placements_made.to_le_bytes());
        match self.phase {
            KeyPhase::Opening => feed(&mut hash, &[0]),
            KeyPhase::FirstStone => feed(&mut hash, &[1]),
            KeyPhase::SecondStone { q, r } => {
                feed(&mut hash, &[2]);
                feed(&mut hash, &q.to_le_bytes());
                feed(&mut hash, &r.to_le_bytes());
            }
        }
        match self.terminal {
            None => feed(&mut hash, &[0]),
            Some(terminal) => {
                feed(&mut hash, &[1, terminal.winner]);
                feed(&mut hash, &terminal.placements.to_le_bytes());
            }
        }
        for stone in &self.stones {
            feed(&mut hash, &stone.q.to_le_bytes());
            feed(&mut hash, &stone.r.to_le_bytes());
            feed(&mut hash, &[stone.owner]);
        }
        hash
    }

    fn heap_bytes(&self) -> usize {
        self.stones
            .capacity()
            .saturating_mul(size_of::<KeyStone>())
            .saturating_add(ALLOC_OVERHEAD)
    }
}

fn player_code(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

struct TtEntry {
    hash: u64,
    key: PositionKey,
    claimant: Player,
    node: CertNodeId,
}

struct BoundedTt {
    slots: Vec<Option<TtEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
}

impl BoundedTt {
    fn new(cap: usize, hash_mask: u64) -> Self {
        let slot_count = (cap / TARGET_BYTES_PER_TT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
            };
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = slots
            .capacity()
            .saturating_mul(size_of::<Option<TtEntry>>())
            .saturating_add(ALLOC_OVERHEAD);
        if base > cap {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
            };
        }
        Self {
            slots,
            cap,
            current_bytes: base,
            peak_bytes: base,
            hash_mask,
        }
    }

    fn lookup(&self, key: &PositionKey, claimant: Player) -> Option<CertNodeId> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.claimant == claimant && entry.key == *key)
            .then_some(entry.node)
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, node: CertNodeId) {
        if self.slots.is_empty() {
            return;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(|entry| entry.key.heap_bytes())
            .unwrap_or(0);
        let new_heap = key.heap_bytes();
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(new_heap);
        if candidate_bytes > self.cap {
            return;
        }
        self.slots[index] = Some(TtEntry {
            hash,
            key,
            claimant,
            node,
        });
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
    }
}

/// Remove abandoned OR branches from the certificate arena and remap every
/// reachable child.  The resulting certificate has no orphan nodes, which the
/// independent verifier requires.
fn compact_certificate(
    arena: &[CertNode],
    root: CertNodeId,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    fn copy(
        old: CertNodeId,
        arena: &[CertNode],
        remap: &mut [Option<CertNodeId>],
        visiting: &mut [bool],
        out: &mut Vec<CertNode>,
    ) -> Option<CertNodeId> {
        let index = old as usize;
        if index >= arena.len() || visiting[index] {
            return None;
        }
        if let Some(mapped) = remap[index] {
            return Some(mapped);
        }
        visiting[index] = true;
        let mapped_node = match &arena[index] {
            CertNode::Terminal => CertNode::Terminal,
            CertNode::Lambda1 => CertNode::Lambda1,
            CertNode::Choice { mv, child } => CertNode::Choice {
                mv: *mv,
                child: copy(*child, arena, remap, visiting, out)?,
            },
            CertNode::Universal {
                edges,
                implicit_dispatch,
            } => {
                let mut mapped_edges = Vec::with_capacity(edges.len());
                for edge in edges {
                    mapped_edges.push(CertEdge {
                        mv: edge.mv,
                        child: copy(edge.child, arena, remap, visiting, out)?,
                    });
                }
                CertNode::Universal {
                    edges: mapped_edges,
                    implicit_dispatch: *implicit_dispatch,
                }
            }
        };
        visiting[index] = false;
        let mapped = u32::try_from(out.len()).ok()?;
        out.push(mapped_node);
        remap[index] = Some(mapped);
        Some(mapped)
    }

    if arena.len() > MAX_CERT_NODES {
        return None;
    }
    let mut remap = vec![None; arena.len()];
    let mut visiting = vec![false; arena.len()];
    let mut nodes = Vec::new();
    let root_node = copy(root, arena, &mut remap, &mut visiting, &mut nodes)?;
    Some((nodes, root_node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tss_core::CertVerify;
    use crate::tss_reference;
    use crate::tss_verify::{
        d6_remap_certificate, d6_transform_coord, TssVerifier, D6_SYMMETRY_COUNT,
    };
    use hexo_engine::apply_placement;

    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord { q, r },
                },
            )
            .unwrap();
        }
        state
    }

    fn forced_loss_fixture() -> RustHexoState {
        replay(&[
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
        ])
    }

    fn quiet_fixture() -> RustHexoState {
        replay(&[(0, 0), (0, 8), (2, 7)])
    }

    fn forced_defense_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
        ])
    }

    fn win_now_fixture() -> RustHexoState {
        let mut state = forced_defense_fixture();
        for coord in [HexCoord::new(8, 4), HexCoord::new(10, 3)] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    fn spare_tempo_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (1, 4),
            (2, 4),
            (-2, -3),
            (-1, -3),
            (3, 4),
            (4, 1),
            (0, -3),
            (2, -3),
            (4, 2),
            (4, 3),
            (3, -3),
            (-3, 1),
            (1, 7),
            (2, 6),
            (1, -2),
            (-4, 2),
            (3, 5),
            (-1, 0),
            (2, -4),
            (-5, 2),
            (0, -1),
            (-2, 1),
            (2, -5),
            (-6, 3),
        ])
    }

    fn deep_universal_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (-1, 0),
            (0, -1),
            (-2, -3),
            (-1, -3),
            (-2, 1),
            (-3, 1),
            (0, -3),
            (1, -3),
            (-4, 2),
            (2, -4),
            (1, 4),
            (2, 4),
            (-5, 2),
            (2, -5),
            (3, 4),
            (4, 1),
            (-6, 3),
            (3, -6),
            (4, 2),
            (4, 3),
            (-7, 3),
            (3, -7),
            (1, 7),
            (2, 6),
            (-1, 2),
            (2, -1),
            (3, 5),
        ])
    }

    fn deep_universal_certificate(state: &RustHexoState) -> TssCertificate {
        // The default proof ordering finds the shorter direct lambda-one move
        // (4,4).  Staple the also-valid longer forcing choice (2,-3) so the
        // harness exercises a real universal strategy subtree.
        let mut work = state.clone();
        let (_placed, delta) = work
            .apply_with_delta(Placement {
                coord: HexCoord::new(2, -3),
            })
            .unwrap();
        let mut context = SearchContext::new(500_000, 8 << 20, u64::MAX);
        let child = context
            .prove(&mut work, Player::Player0, 1)
            .expect("validated forcing branch must prove");
        work.undo(delta);
        let root = context
            .alloc_node(
                CertNode::Choice {
                    mv: HexCoord::new(2, -3),
                    child,
                },
                1,
            )
            .unwrap();
        let (nodes, root_node) = compact_certificate(&context.arena, root).unwrap();
        TssCertificate {
            root: RootBinding::from_state(state),
            claimant: Player::Player0,
            root_node,
            nodes,
        }
    }

    fn transformed_state(state: &RustHexoState, symmetry: u8) -> RustHexoState {
        let mut transformed = RustHexoState::new();
        for record in state.placement_history() {
            let coord = d6_transform_coord(record.coord, symmetry).unwrap();
            apply_placement(&mut transformed, Placement { coord }).unwrap();
        }
        transformed
    }

    fn assert_exact_state(left: &RustHexoState, right: &RustHexoState) {
        assert_eq!(left.current_player(), right.current_player());
        assert_eq!(left.phase(), right.phase());
        assert_eq!(left.placements_made(), right.placements_made());
        assert_eq!(left.terminal(), right.terminal());
        assert_eq!(left.last_turn(), right.last_turn());
        assert_eq!(left.placement_history(), right.placement_history());
        assert_eq!(
            left.board().occupied_cells(),
            right.board().occupied_cells()
        );
        for &coord in left.board().occupied_cells() {
            assert_eq!(left.board().get(coord), right.board().get(coord));
        }
        let mut left_legal = Vec::new();
        let mut right_legal = Vec::new();
        left.write_legal_action_ids(&mut left_legal);
        right.write_legal_action_ids(&mut right_legal);
        assert_eq!(left_legal, right_legal);
        fn windows(state: &RustHexoState) -> Vec<(u8, i16, i16, u8, u8)> {
            let mut entries: Vec<_> = state
                .board()
                .windows()
                .entries()
                .map(|entry| {
                    let key = entry.key();
                    (
                        key.axis.index(),
                        key.start.q,
                        key.start.r,
                        entry.mask(Player::Player0),
                        entry.mask(Player::Player1),
                    )
                })
                .collect();
            entries.sort_unstable();
            entries
        }
        assert_eq!(windows(left), windows(right));
    }

    #[test]
    fn root_lambda_loss_has_dual_certificate() {
        let state = forced_loss_fixture();
        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 0,
        };
        let result = TssSolver::default().solve(&state, &caps);
        assert_eq!(result.status, ProofStatus::Loss);
        let cert = result.cert.unwrap();
        assert_eq!(cert.claimant, state.current_player().other());
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Loss));
    }

    #[test]
    fn full_key_rejects_forced_hash_collision() {
        let left = replay(&[(0, 0), (1, 0)]);
        let right = replay(&[(0, 0), (0, 1)]);
        let left_key = PositionKey::from_state(&left);
        let right_key = PositionKey::from_state(&right);
        assert_ne!(left_key, right_key);

        let mut tt = BoundedTt::new(4096, 0);
        tt.insert(left_key.clone(), Player::Player0, 7);
        assert_eq!(tt.lookup(&left_key, Player::Player0), Some(7));
        assert_eq!(tt.lookup(&right_key, Player::Player0), None);
        tt.insert(right_key.clone(), Player::Player0, 9);
        assert_eq!(tt.lookup(&left_key, Player::Player0), None);
        assert_eq!(tt.lookup(&right_key, Player::Player0), Some(9));
    }

    #[test]
    fn tt_allocation_never_exceeds_cap() {
        let states = [
            replay(&[(0, 0)]),
            replay(&[(0, 0), (1, 0)]),
            replay(&[(0, 0), (0, 1), (1, 0)]),
        ];
        for cap in [0, 1, 255, 256, 1024, 4096] {
            let mut tt = BoundedTt::new(cap, u64::MAX);
            let base = if tt.slots.is_empty() {
                0
            } else {
                tt.slots.capacity() * size_of::<Option<TtEntry>>() + ALLOC_OVERHEAD
            };
            assert_eq!(tt.current_bytes, base);
            assert_eq!(tt.peak_bytes, base);
            let mut expected_peak = base;
            for (node, state) in states.iter().enumerate() {
                tt.insert(
                    PositionKey::from_state(state),
                    state.current_player(),
                    node as u32,
                );
                let exact_accounted = base
                    + tt.slots
                        .iter()
                        .flatten()
                        .map(|entry| entry.key.heap_bytes())
                        .sum::<usize>();
                expected_peak = expected_peak.max(exact_accounted);
                assert_eq!(tt.current_bytes, exact_accounted);
                assert_eq!(tt.peak_bytes, expected_peak);
                assert!(tt.current_bytes <= cap);
                assert!(tt.peak_bytes <= cap);
            }
        }
    }

    #[test]
    fn zero_node_cap_is_unknown_and_certless() {
        let state = forced_loss_fixture();
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 0,
                tt_bytes_cap: usize::MAX,
            },
        );
        assert_eq!(result.status, ProofStatus::Unknown);
        assert!(result.cert.is_none());
        assert_eq!(result.stats.nodes, 0);
    }

    #[test]
    fn solver_configurations_are_deterministic_on_hard_leaf() {
        let state = forced_loss_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 4096,
        };
        let a = TssSolver::without_tt().solve(&state, &caps);
        let b = TssSolver::with_hash_mask(0).solve(&state, &caps);
        let c = TssSolver::default().solve(&state, &caps);
        assert_eq!(a.status, b.status);
        assert_eq!(b.status, c.status);
        assert_eq!(a.cert, b.cert);
        assert_eq!(b.cert, c.cert);
    }

    #[test]
    fn deep_win_contains_verified_universal_coverage() {
        let state = deep_universal_fixture();
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 500_000,
                tt_bytes_cap: 8 << 20,
            },
        );
        assert_eq!(result.status, ProofStatus::Win);
        let cert = deep_universal_certificate(&state);
        assert!(cert
            .nodes
            .iter()
            .any(|node| matches!(node, CertNode::Universal { .. })));
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));
    }

    #[test]
    fn curated_differential_and_every_hard_certificate_verifies() {
        let fixtures = [
            (quiet_fixture(), ProofStatus::Unknown, 1u64, None),
            (forced_defense_fixture(), ProofStatus::Unknown, 1, None),
            (win_now_fixture(), ProofStatus::Win, 1, Some(2)),
            // The exact Python forced-loss fixture is retained here, while a
            // zero cap exercises the permitted Unknown differential path.  Its
            // hard dual certificate is tested above; the independently
            // exhaustive hard comparison uses its SecondStone child below so
            // default debug CI does not enumerate ~10^6 two-stone defenses.
            (forced_loss_fixture(), ProofStatus::Unknown, 0, None),
            (spare_tempo_fixture(), ProofStatus::Unknown, 1, None),
        ];
        for (state, expected, node_cap, reference_depth) in fixtures {
            let caps = SolveCaps {
                node_cap,
                tt_bytes_cap: 0,
            };
            let result = TssSolver::default().solve(&state, &caps);
            assert_eq!(result.status, expected);
            if result.status == ProofStatus::Unknown {
                assert!(result.cert.is_none());
                continue;
            }
            let cert = result.cert.as_ref().expect("hard result needs cert");
            assert!(TssVerifier.verify(&state, cert, result.status));
            let reference = tss_reference::solve(&state, reference_depth.unwrap());
            assert_eq!(
                reference.status, result.status,
                "reference disagreement after {} nodes",
                reference.nodes
            );
        }
    }

    #[test]
    fn differential_forced_loss_after_first_defender_stone() {
        let mut state = forced_loss_fixture();
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(-8, 0),
            },
        )
        .unwrap();
        assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }));
        assert!(threats::analyze(&state).forced_loss());
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 1,
                tt_bytes_cap: 0,
            },
        );
        assert_eq!(result.status, ProofStatus::Loss);
        assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
        let reference = tss_reference::solve(&state, 3);
        assert_eq!(reference.status, ProofStatus::Loss);
    }

    #[test]
    fn seeded_random_differential_covers_all_phases_and_dense_positions() {
        #[derive(Clone, Copy)]
        struct XorShift(u64);
        impl XorShift {
            fn next(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
        }

        let mut corpus = vec![win_now_fixture()]; // threat-dense endgame anchor
        let mut saw_phase = [false; 3];
        for seed in 1..=6u64 {
            let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut state = RustHexoState::new();
            for ply in 0..20usize {
                let phase_index = match state.phase() {
                    TurnPhase::Opening => 0,
                    TurnPhase::FirstStone => 1,
                    TurnPhase::SecondStone { .. } => 2,
                };
                saw_phase[phase_index] = true;
                if ply < 3 || ply % 5 == seed as usize % 5 {
                    corpus.push(state.clone());
                }
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                if legal.is_empty() {
                    break;
                }
                let coord = legal[(rng.next() as usize) % legal.len()];
                let result = apply_placement(&mut state, Placement { coord }).unwrap();
                if result.outcome.is_some() {
                    break;
                }
            }
        }
        assert!(saw_phase.into_iter().all(|seen| seen));
        assert!(corpus
            .iter()
            .any(|state| state.board().windows().has_threats()));

        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 1024,
        };
        let mut hard = 0usize;
        for state in corpus {
            let result = TssSolver::default().solve(&state, &caps);
            let Some(cert) = result.cert.as_ref() else {
                assert_eq!(result.status, ProofStatus::Unknown);
                continue;
            };
            hard += 1;
            assert!(TssVerifier.verify(&state, cert, result.status));
            let b = threats::placements_remaining(&state) as u32;
            let depth = if result.status == ProofStatus::Win {
                b
            } else {
                b + 2
            };
            assert_eq!(tss_reference::solve(&state, depth).status, result.status);
        }
        assert!(
            hard >= 1,
            "dense anchor must produce at least one hard proof"
        );
    }

    #[test]
    fn spare_stone_counter_threat_cannot_be_implicitly_dispatched() {
        let state = spare_tempo_fixture();
        let root = threats::analyze(&state);
        assert_eq!(state.phase(), TurnPhase::FirstStone);
        assert_eq!(root.b, 2);
        assert_eq!(root.min_hitting_set, Some(1));
        assert!(root.opp_threat_count > 0);
        assert!(!root.own_win_now);

        // Y is outside the sole hitting cell G, yet it creates a counter-fork;
        // lambda-one correctly leaves the child unresolved instead of
        // dispatching it as a loss.
        let counter = HexCoord::new(4, 4);
        let guard = HexCoord::new(1, -3);
        assert!(
            ordered_threat_creating_moves(&mut state.clone(), Player::Player1)
                .iter()
                .any(|candidate| candidate.coord == counter)
        );
        let mut line = state.clone();
        apply_placement(&mut line, Placement { coord: counter }).unwrap();
        let after_counter = threats::analyze(&line);
        assert_eq!(line.phase(), TurnPhase::SecondStone { first: counter });
        assert_eq!(after_counter.verdict(), None);
        assert_eq!(after_counter.min_hitting_set, Some(1));
        apply_placement(&mut line, Placement { coord: guard }).unwrap();
        assert!(threats::analyze(&line).forced_loss());

        // A malicious P0 strategy certificate that advertises dispatch at the
        // k<B root is rejected before any child can be trusted.
        let bogus = TssCertificate {
            root: RootBinding::from_state(&state),
            claimant: Player::Player0,
            root_node: 0,
            nodes: vec![CertNode::Universal {
                edges: Vec::new(),
                implicit_dispatch: true,
            }],
        };
        assert!(!TssVerifier.verify(&state, &bogus, ProofStatus::Loss));
    }

    #[test]
    fn certificate_mutations_are_rejected() {
        let state = deep_universal_fixture();
        let cert = deep_universal_certificate(&state);
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));

        let mut corrupt_move = cert.clone();
        let root = corrupt_move.root_node as usize;
        let CertNode::Choice { mv, .. } = &mut corrupt_move.nodes[root] else {
            panic!("deep cert root must be Choice");
        };
        *mv = HexCoord::ZERO; // occupied at the root
        assert!(!TssVerifier.verify(&state, &corrupt_move, ProofStatus::Win));

        let mut dropped_child = cert.clone();
        let universal = dropped_child
            .nodes
            .iter_mut()
            .find_map(|node| match node {
                CertNode::Universal { edges, .. } if !edges.is_empty() => Some(edges),
                _ => None,
            })
            .expect("fixture must contain a non-empty universal node");
        universal.remove(0);
        assert!(!TssVerifier.verify(&state, &dropped_child, ProofStatus::Win));

        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!TssVerifier.verify(&quiet_fixture(), &cert, ProofStatus::Win));

        let mut wrong_witness = cert.clone();
        wrong_witness.root.phase = TurnPhase::SecondStone {
            first: HexCoord::new(99, 99),
        };
        assert!(!TssVerifier.verify(&state, &wrong_witness, ProofStatus::Win));

        let mut cycle = cert.clone();
        let root = cycle.root_node;
        let CertNode::Choice { child, .. } = &mut cycle.nodes[root as usize] else {
            unreachable!();
        };
        *child = root;
        assert!(!TssVerifier.verify(&state, &cycle, ProofStatus::Win));
    }

    #[test]
    fn d6_status_and_certificate_replay_all_twelve_symmetries() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 64 << 10,
        };
        let base = TssSolver::default().solve(&state, &caps);
        assert_eq!(base.status, ProofStatus::Win);
        let deep_cert = deep_universal_certificate(&state);

        for symmetry in 0..D6_SYMMETRY_COUNT {
            let transformed = transformed_state(&state, symmetry);
            let result = TssSolver::default().solve(&transformed, &caps);
            assert_eq!(result.status, base.status, "symmetry={symmetry}");
            let cert = result.cert.as_ref().unwrap();
            assert!(TssVerifier.verify(&transformed, cert, result.status));

            let remapped = d6_remap_certificate(&deep_cert, symmetry).unwrap();
            assert!(TssVerifier.verify(&transformed, &remapped, ProofStatus::Win));
        }
    }

    #[test]
    fn tt_disabled_tiny_large_and_interleaved_solves_match() {
        let state = deep_universal_fixture();
        let caps_for = |bytes| SolveCaps {
            node_cap: 64,
            tt_bytes_cap: bytes,
        };
        let off = TssSolver::without_tt().solve(&state, &caps_for(0));
        let tiny = TssSolver::default().solve(&state, &caps_for(256));
        let large = TssSolver::default().solve(&state, &caps_for(8 << 20));
        assert_eq!(off.status, ProofStatus::Win);
        assert_eq!(off.status, tiny.status);
        assert_eq!(tiny.status, large.status);
        for result in [&off, &tiny, &large] {
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
        }
        assert_eq!(off.stats.peak_tt_bytes, 0);
        assert!(tiny.stats.peak_tt_bytes <= 256);
        assert!(large.stats.peak_tt_bytes <= (8 << 20));

        let mut solver = TssSolver::with_hash_mask(0);
        let first = solver.solve(&state, &caps_for(4096));
        let _other = solver.solve(&forced_loss_fixture(), &caps_for(4096));
        let repeated = solver.solve(&state, &caps_for(4096));
        assert_eq!(first.status, repeated.status);
        assert_eq!(first.cert, repeated.cert);
        assert_eq!(first.stats.nodes, repeated.stats.nodes);
        assert_eq!(first.stats.tt_hits, repeated.stats.tt_hits);
        assert_eq!(first.stats.peak_tt_bytes, repeated.stats.peak_tt_bytes);
    }

    #[test]
    fn make_unmake_round_trips_on_proof_and_cap_exit() {
        let original = deep_universal_fixture();
        for cap in [2, 500_000] {
            let mut work = original.clone();
            let before = work.clone();
            let mut context = SearchContext::new(cap, 64 << 10, u64::MAX);
            let _ = context.prove(&mut work, Player::Player0, 0);
            assert_exact_state(&work, &before);
        }
    }

    #[test]
    fn node_cap_only_moves_results_toward_unknown() {
        let state = deep_universal_fixture();
        let small = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 3,
                tt_bytes_cap: 4096,
            },
        );
        let enough = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 5,
                tt_bytes_cap: 4096,
            },
        );
        assert_eq!(small.status, ProofStatus::Unknown);
        assert!(small.cert.is_none());
        assert_eq!(enough.status, ProofStatus::Win);
        assert!(TssVerifier.verify(&state, enough.cert.as_ref().unwrap(), enough.status));
        assert!(small.stats.nodes <= 3);
        assert!(enough.stats.nodes <= 5);
        assert!(small.stats.peak_tt_bytes <= 4096);
        assert!(enough.stats.peak_tt_bytes <= 4096);
    }
}
