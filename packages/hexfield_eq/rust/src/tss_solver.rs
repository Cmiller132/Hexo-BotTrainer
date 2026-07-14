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
use std::collections::HashSet;
use std::mem::size_of;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowKey,
};

use crate::threats_shared as threats;
use crate::tss_core::{
    DeepResult, DeepSolve, ProofStatus, SolveCaps, SolveGoal, SolveStats, ZoneSearchCaps,
};
use crate::tss_verify::{
    CertCommutation, CertEdge, CertNode, CertNodeId, RootBinding, TssCertificate, ZoneInfo,
    MAX_CERT_DEPTH, MAX_CERT_EDGES, MAX_CERT_NODES, MAX_CERT_ROOT_STONES,
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
/// Shared entries own certificate fragments and are consequently wider than
/// solve-local entries.  The target only determines direct-table density; the
/// exact retained capacities below remain the authoritative byte accounting.
const TARGET_BYTES_PER_SHARED_TT_SLOT: usize = 512;
/// Internal positive fragments are promoted only while cheap to compact.  A
/// successful attempt root is offered regardless of these two tuning limits.
const MAX_PROMOTED_FRAGMENT_NODES: usize = 128;
const MAX_PROMOTED_FRAGMENT_EDGES: usize = 512;
/// Solve-local TT sentinel for a fully explored restricted position with no
/// proof in the current wide/depth-bounded attempt. Certificate IDs can never
/// approach this value (`MAX_CERT_NODES` is 100k).
const LOCAL_TT_FAILED: CertNodeId = CertNodeId::MAX;

/// Optional attacker-universe expansions.  The default is deliberately the
/// historical narrow generator so production callers retain byte-identical
/// search behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WidthOptions {
    vcf_pair_complete: bool,
}

impl WidthOptions {
    pub(crate) fn vcf_pair_complete() -> Self {
        Self {
            vcf_pair_complete: true,
        }
    }
}

/// Reusable proof-carrying solver.  Its shared TT retains only complete,
/// self-contained positive proof fragments; solve-local arena IDs never cross
/// an attempt boundary.
#[derive(Debug)]
pub(crate) struct TssSolver {
    tt_enabled: bool,
    hash_mask: u64,
    shared_tt: SharedProofCache,
    zone: ZoneSearchCaps,
    width: WidthOptions,
}

impl Default for TssSolver {
    fn default() -> Self {
        Self {
            tt_enabled: true,
            hash_mask: u64::MAX,
            shared_tt: SharedProofCache::new(0, u64::MAX),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
        }
    }
}

impl TssSolver {
    /// Set the zone/commutation options for subsequent solves. Changing the
    /// options DROPS the persistent positive-fragment cache: cached fragments
    /// are verified proofs either way, but their node-cost provenance belongs
    /// to the profile that built them — reusing them across an ON→OFF flip
    /// contaminates A/B node counts and conditional determinism (Codex
    /// review, profile isolation). Same-options calls keep the warm cache.
    pub(crate) fn set_zone_options(&mut self, zone: ZoneSearchCaps) {
        if self.zone != zone {
            self.shared_tt.clear();
        }
        self.zone = zone;
    }

    /// Set the attacker-width profile for subsequent solves.  As with zone
    /// options, changing profiles drops cached positive fragments so their
    /// node-cost provenance cannot leak between narrow and wide searches.
    pub(crate) fn set_width_options(&mut self, width: WidthOptions) {
        if self.width != width {
            self.shared_tt.clear();
        }
        self.width = width;
    }

    #[cfg(test)]
    fn without_tt() -> Self {
        Self {
            tt_enabled: false,
            hash_mask: u64::MAX,
            shared_tt: SharedProofCache::new(0, u64::MAX),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
        }
    }

    /// Test hook: masking every hash to zero forces all positions into one
    /// bucket.  Full-key equality must still prevent a value-bearing false hit.
    #[cfg(test)]
    fn with_hash_mask(hash_mask: u64) -> Self {
        Self {
            tt_enabled: true,
            hash_mask,
            shared_tt: SharedProofCache::new(0, hash_mask),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
        }
    }

    /// Solve only for the requested root-perspective side(s).  One-sided modes
    /// receive the entire remaining node budget; the legacy trait entry point
    /// below delegates to `Both`.
    pub(crate) fn solve_goal(
        &mut self,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
    ) -> DeepResult<TssCertificate> {
        let effective_tt_cap = if self.tt_enabled {
            caps.tt_bytes_cap
        } else {
            0
        };
        let (local_tt_cap, shared_tt_cap) = split_tt_cap(effective_tt_cap);
        self.shared_tt.reconfigure(shared_tt_cap, self.hash_mask);

        let initial_stats = SolveStats {
            peak_tt_bytes: self.shared_tt.current_bytes as u64,
            ..SolveStats::default()
        };
        if caps.node_cap == 0
            || caps.semantic_horizon < state.placements_made()
            || state.board().len() > MAX_CERT_ROOT_STONES
        {
            return unknown(initial_stats);
        }

        // A root lambda-one/terminal result is both common and symmetric
        // between the primal and dual claims.  Count it as one examined node,
        // but filter it when the caller explicitly requested only the other
        // perspective.
        let mut stats = SolveStats {
            nodes: 1,
            ..initial_stats
        };
        if let Some((claimant, leaf)) = immediate_winner(state) {
            if node_resolution(&leaf) > caps.semantic_horizon {
                return unknown(stats);
            }
            let status = status_for_claimant(state.current_player(), claimant);
            if goal_accepts(goal, status) {
                let cert = TssCertificate {
                    root: RootBinding::from_state(state),
                    claimant,
                    root_node: 0,
                    nodes: vec![leaf],
                    semantic_horizon: caps.semantic_horizon,
                };
                return DeepResult {
                    status,
                    cert: Some(cert),
                    stats,
                };
            }
            return unknown(stats);
        }

        let remaining = caps.node_cap - 1;
        let (primal_cap, dual_cap) = match goal {
            SolveGoal::Win => (remaining, 0),
            SolveGoal::Loss => (0, remaining),
            SolveGoal::Both => ((remaining + 1) / 2, remaining / 2),
        };
        let root_player = state.current_player();

        if primal_cap > 0 {
            let attempt = self.prove_for(
                state,
                root_player,
                primal_cap,
                local_tt_cap,
                caps.semantic_horizon,
                self.zone,
                self.width,
            );
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
            let attempt = self.prove_for(
                state,
                root_player.other(),
                dual_cap,
                local_tt_cap,
                caps.semantic_horizon,
                self.zone,
                self.width,
            );
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

impl DeepSolve for TssSolver {
    type Cert = TssCertificate;

    fn solve(&mut self, state: &RustHexoState, caps: &SolveCaps) -> DeepResult<Self::Cert> {
        self.solve_goal(state, caps, SolveGoal::Both)
    }
}

impl TssSolver {
    fn prove_for(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
    ) -> AttemptResult {
        if !width.vcf_pair_complete {
            return self.prove_for_at_depth(
                state,
                claimant,
                node_cap,
                local_tt_cap,
                semantic_horizon,
                zone,
                width,
                MAX_SEARCH_DEPTH,
            );
        }

        const DEPTH_LADDER: [usize; 5] = [16, 32, 64, 128, MAX_SEARCH_DEPTH];
        let mut stats = SolveStats::default();
        for (index, depth_cap) in DEPTH_LADDER.into_iter().enumerate() {
            let remaining = node_cap.saturating_sub(stats.nodes);
            if remaining == 0 {
                break;
            }
            let attempts_left = (DEPTH_LADDER.len() - index) as u64;
            let attempt_cap = remaining / attempts_left;
            let attempt = self.prove_for_at_depth(
                state,
                claimant,
                attempt_cap,
                local_tt_cap,
                semantic_horizon,
                zone,
                width,
                depth_cap,
            );
            merge_stats(&mut stats, attempt.stats);
            if attempt.cert.is_some() {
                return AttemptResult {
                    cert: attempt.cert,
                    stats,
                };
            }
        }
        AttemptResult { cert: None, stats }
    }

    #[allow(clippy::too_many_arguments)]
    fn prove_for_at_depth(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        depth_cap: usize,
    ) -> AttemptResult {
        let mut work = state.clone();
        let entry_key = PositionKey::from_state(&work);
        let root_ply = state.placements_made();
        let mut context = SearchContext::with_shared(
            node_cap,
            local_tt_cap,
            self.hash_mask,
            &mut self.shared_tt,
            root_ply,
            semantic_horizon,
            zone,
            width,
            depth_cap,
        );
        let proof = context.prove(&mut work, claimant, root_ply, None);

        // Every recursive path uses LIFO make/unmake, including cap exits.
        debug_assert_eq!(entry_key, PositionKey::from_state(&work));

        let stats = SolveStats {
            nodes: context.nodes,
            tt_hits: context.tt_hits,
            peak_tt_bytes: context.peak_tt_bytes as u64,
        };
        let cert = proof.and_then(|root| {
            let (nodes, root_node) = compact_certificate(&context.arena, root)?;
            if context.can_admit_compact(&entry_key, &nodes) {
                if let Some(cached) = CachedProof::from_compact(nodes.clone(), root_node) {
                    context.insert_shared(entry_key.clone(), claimant, cached);
                }
            }
            let mut cert = TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node,
                nodes,
                semantic_horizon,
            };
            rebase_zone_distances(&mut cert, state)?;
            Some(cert)
        });
        let stats = SolveStats {
            peak_tt_bytes: context.peak_tt_bytes as u64,
            ..stats
        };
        AttemptResult { cert, stats }
    }
}

/// Imported zone fragments may have been built at a larger admissible
/// horizon. Their searched set remains sound at a smaller T, but the carried
/// D evidence must be relabelled to the assembled certificate's exact build
/// horizon before solver-side preflight and independent verification.
fn rebase_zone_distances(cert: &mut TssCertificate, root: &RustHexoState) -> Option<()> {
    let mut states = vec![None; cert.nodes.len()];
    let mut stack = vec![(cert.root_node, root.clone())];
    while let Some((id, state)) = stack.pop() {
        let slot = states.get_mut(id as usize)?;
        if let Some(seen) = slot.as_ref() {
            if PositionKey::from_state(seen) != PositionKey::from_state(&state) {
                return None;
            }
            continue;
        }
        *slot = Some(state.clone());
        match cert.nodes.get(id as usize)? {
            CertNode::Choice { mv, child } => {
                let mut next = state;
                let result = apply_placement(&mut next, Placement { coord: *mv }).ok()?;
                if result.outcome.is_some() {
                    return None;
                }
                stack.push((*child, next));
            }
            CertNode::Universal { edges, .. } => {
                for edge in edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    stack.push((edge.child, next));
                }
            }
            _ => {}
        }
    }
    for (index, state) in states.into_iter().enumerate() {
        let Some(state) = state else {
            return None;
        };
        if let CertNode::Universal {
            zone: Some(zone), ..
        } = cert.nodes.get_mut(index)?
        {
            zone.d = remaining_defender_placements_for_horizon(
                &state,
                cert.claimant,
                cert.semantic_horizon,
            )?;
        }
    }
    Some(())
}

fn split_tt_cap(total: usize) -> (usize, usize) {
    let shared = total / 2;
    (total - shared, shared)
}

fn goal_accepts(goal: SolveGoal, status: ProofStatus) -> bool {
    matches!(
        (goal, status),
        (SolveGoal::Both, ProofStatus::Win | ProofStatus::Loss)
            | (SolveGoal::Win, ProofStatus::Win)
            | (SolveGoal::Loss, ProofStatus::Loss)
    )
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
    if state.is_terminal() {
        return None;
    }
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let analysis = threats::analyze(state);
    let winner = winner_from_analysis(state, &analysis)?;
    typed_lambda_leaf(state, winner, &analysis).map(|leaf| (winner, leaf))
}

fn window_key_order(key: WindowKey) -> (u8, i16, i16) {
    (key.axis.index(), key.start.q, key.start.r)
}

fn typed_lambda_leaf(
    state: &RustHexoState,
    winner: Player,
    analysis: &threats::ThreatAnalysis,
) -> Option<CertNode> {
    if winner == state.current_player() {
        let mut candidates = state
            .board()
            .windows()
            .entries()
            .filter(|entry| {
                entry.active_player() == Some(winner)
                    && (entry.count(winner) == 5 || (analysis.b == 2 && entry.count(winner) == 4))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|entry| {
            (
                std::cmp::Reverse(entry.count(winner)),
                window_key_order(entry.key()),
            )
        });
        let witness = candidates.first().copied()?;
        let count = witness.count(winner);
        let extra = if count == 5 { 1 } else { 2 };
        Some(CertNode::Win {
            witness: witness.key(),
            count,
            budget: analysis.b,
            resolution_ply: state.placements_made().saturating_add(extra),
        })
    } else {
        let mut witnesses = state
            .board()
            .windows()
            .threats()
            .filter_map(|(owner, entry)| (owner == winner).then_some(entry.key()))
            .collect::<Vec<_>>();
        witnesses.sort_by_key(|key| window_key_order(*key));
        witnesses.dedup();
        (!witnesses.is_empty()).then_some(CertNode::Loss {
            witnesses,
            resolution_ply: state
                .placements_made()
                .saturating_add(u32::from(analysis.b))
                .saturating_add(2),
        })
    }
}

fn node_resolution(node: &CertNode) -> u32 {
    match node {
        CertNode::OrCompletion { completion_ply, .. } => *completion_ply,
        CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
            *resolution_ply
        }
        CertNode::Choice { .. } | CertNode::Universal { .. } => 0,
    }
}

fn winner_from_analysis(
    state: &RustHexoState,
    analysis: &threats::ThreatAnalysis,
) -> Option<Player> {
    if analysis.own_win_now {
        Some(state.current_player())
    } else if analysis.forced_loss() {
        Some(state.current_player().other())
    } else {
        None
    }
}

struct SearchContext<'a> {
    node_cap: u64,
    nodes: u64,
    tt_hits: u64,
    hit_limit: bool,
    arena: Vec<CertNode>,
    edge_count: usize,
    tt: BoundedTt,
    shared_tt: Option<&'a mut SharedProofCache>,
    peak_tt_bytes: usize,
    /// Absolute placement index at the attempt root.  Structural depth is
    /// derived from the separately threaded ply clock.
    root_ply: u32,
    semantic_horizon: u32,
    clock_is_absolute: bool,
    zone: ZoneSearchCaps,
    width: WidthOptions,
    depth_cap: usize,
}

#[derive(Clone, Debug)]
struct PairContext {
    first: HexCoord,
    turn_start_legal: Vec<HexCoord>,
}

impl SearchContext<'static> {
    fn new(node_cap: u64, tt_bytes_cap: usize, hash_mask: u64) -> Self {
        let tt = BoundedTt::new(tt_bytes_cap, hash_mask);
        let peak_tt_bytes = tt.current_bytes;
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt,
            shared_tt: None,
            peak_tt_bytes,
            root_ply: 0,
            semantic_horizon: u32::MAX,
            clock_is_absolute: false,
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            depth_cap: MAX_SEARCH_DEPTH,
        }
    }
}

impl<'a> SearchContext<'a> {
    fn with_shared(
        node_cap: u64,
        tt_bytes_cap: usize,
        hash_mask: u64,
        shared_tt: &'a mut SharedProofCache,
        root_ply: u32,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        depth_cap: usize,
    ) -> Self {
        let tt = BoundedTt::new(tt_bytes_cap, hash_mask);
        let peak_tt_bytes = tt.current_bytes.saturating_add(shared_tt.current_bytes);
        let shared_tt = (!shared_tt.slots.is_empty()).then_some(shared_tt);
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt,
            shared_tt,
            peak_tt_bytes,
            root_ply,
            semantic_horizon,
            clock_is_absolute: true,
            zone,
            width,
            depth_cap,
        }
    }

    fn prove(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        let depth = if self.clock_is_absolute {
            debug_assert_eq!(state.placements_made(), ply);
            usize::try_from(ply.checked_sub(self.root_ply)?).ok()?
        } else {
            ply as usize
        };
        if self.clock_is_absolute && ply > self.semantic_horizon {
            return None;
        }
        if depth > self.depth_cap {
            if !self.width.vcf_pair_complete {
                self.hit_limit = true;
            }
            return None;
        }
        if self.nodes >= self.node_cap {
            self.hit_limit = true;
            return None;
        }
        self.nodes += 1;

        let key = PositionKey::from_state(state);
        if pair.is_none() {
            if let Some(node) = self.tt.lookup(&key, claimant) {
                if node == LOCAL_TT_FAILED && self.width.vcf_pair_complete {
                    self.tt_hits += 1;
                    return None;
                }
                if (node as usize) < self.arena.len() {
                    self.tt_hits += 1;
                    return Some(node);
                }
            }
            if let Some(node) = self.lookup_shared(&key, claimant, depth) {
                self.tt.insert(key, claimant, node);
                self.tt_hits += 1;
                self.observe_tt_bytes();
                return Some(node);
            }
        }

        if let Some(outcome) = state.terminal() {
            let _ = outcome;
            // A claimant completion is represented at its parent by the typed
            // OrCompletion leaf; defender-terminal edges are not certifiable.
            return None;
        }

        // Analyze each non-terminal node exactly once.  Universal dispatch
        // consumes this same immutable result instead of repeating the scan.
        let analysis = threats::analyze(state);
        if !matches!(state.phase(), TurnPhase::Opening) {
            if let Some(winner) = winner_from_analysis(state, &analysis) {
                if winner != claimant {
                    return None;
                }
                let leaf = typed_lambda_leaf(state, winner, &analysis)?;
                if node_resolution(&leaf) > self.semantic_horizon {
                    return None;
                }
                let node = self.alloc_node(leaf, 0)?;
                self.remember_proof(key, claimant, node);
                return Some(node);
            }
        }

        let node = if state.current_player() == claimant {
            self.prove_choice(state, claimant, ply, pair)
        } else {
            self.prove_universal(state, claimant, ply, &analysis, pair)
        };
        let Some(node) = node else {
            if self.width.vcf_pair_complete && !self.hit_limit && pair.is_none() {
                self.tt.insert(key, claimant, LOCAL_TT_FAILED);
                self.observe_tt_bytes();
            }
            return None;
        };
        if pair.is_none() {
            self.remember_proof(key, claimant, node);
        }
        Some(node)
    }

    fn prove_choice(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        // Descending line count is the static proof-number initialization:
        // completions before four-builds before three-builds.  The coordinate
        // tie break makes the order independent of WindowStore hash iteration.
        let mut candidates = ordered_threat_creating_moves_with_width(state, claimant, self.width);
        if self.width.vcf_pair_complete {
            if let Some(pair) = pair {
                candidates.retain(|candidate| pair_candidate_allowed(candidate.coord, pair));
            }
        }
        let turn_start_candidates = (self.width.vcf_pair_complete
            && pair.is_none()
            && matches!(state.phase(), TurnPhase::FirstStone)
            && threats::placements_remaining(state) == 2)
            .then(|| {
                let mut coords = candidates
                    .iter()
                    .map(|candidate| candidate.coord)
                    .collect::<Vec<_>>();
                coords.sort_by_key(|coord| raw_coord_key(*coord));
                coords
            });
        for candidate in candidates {
            let Ok((result, delta)) = state.apply_with_delta(Placement {
                coord: candidate.coord,
            }) else {
                continue;
            };
            if result
                .outcome
                .is_some_and(|outcome| outcome.winner == claimant)
            {
                let mut witnesses = state
                    .board()
                    .windows()
                    .entries()
                    .filter(|entry| {
                        entry.key().contains(candidate.coord)
                            && entry.count(claimant) == 6
                            && entry.count(claimant.other()) == 0
                    })
                    .map(|entry| entry.key())
                    .collect::<Vec<_>>();
                witnesses.sort_by_key(|key| window_key_order(*key));
                let witness = witnesses.first().copied();
                state.undo(delta);
                let completion_ply = ply.checked_add(1)?;
                if completion_ply > self.semantic_horizon {
                    return None;
                }
                return self.alloc_node(
                    CertNode::OrCompletion {
                        mv: candidate.coord,
                        witness: witness?,
                        completion_ply,
                    },
                    0,
                );
            }
            let pair_context = turn_start_candidates.as_ref().and_then(|turn_start_legal| {
                (matches!(state.phase(), TurnPhase::SecondStone { .. })).then(|| PairContext {
                    first: candidate.coord,
                    turn_start_legal: turn_start_legal.clone(),
                })
            });
            let child = self.prove(state, claimant, ply.checked_add(1)?, pair_context.as_ref());
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
        ply: u32,
        analysis: &threats::ThreatAnalysis,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        let implicit_dispatch = !matches!(state.phase(), TurnPhase::Opening)
            && analysis.opp_threat_count > 0
            && !analysis.own_win_now
            && analysis.min_hitting_set == Some(analysis.b);

        // At the proved L1 boundary U3 lets the verifier theorem-dismiss the
        // complement without enumerating it.  At spare nodes the default-off
        // U1 generator is consumable only because U2 re-derives the zone.
        let zone = (!implicit_dispatch
            && self.zone.enabled
            && !matches!(state.phase(), TurnPhase::Opening))
        .then(|| {
            remaining_defender_placements_for_horizon(state, claimant, self.semantic_horizon).map(
                |d| ZoneInfo {
                    d,
                    build_horizon: self.semantic_horizon,
                },
            )
        })
        .flatten();
        let mut explicit = if implicit_dispatch {
            hitting_universe(state, claimant)
        } else if let Some(zone) = zone {
            zone_initial_candidates(state, claimant, zone.d, self.zone)
        } else {
            let mut all_legal = Vec::new();
            state.write_legal_moves(&mut all_legal);
            if all_legal.is_empty() {
                return None;
            }
            all_legal
        };
        if !implicit_dispatch && zone.is_none() {
            if let Some(pair) = pair {
                restrict_pair_candidates(&mut explicit, pair);
            }
        }

        if implicit_dispatch {
            let frame = canonical_frame(state);
            explicit.sort_by_key(|coord| canonical_coord_key(frame, *coord));
        } else {
            // At spare-budget nodes every legal move remains explicit, but
            // likely defenses are searched first so a refutation can stop the
            // lazy child loop before distant quiet moves are materialized.
            let hitting = hitting_universe(state, claimant);
            let frame = canonical_frame(state);
            explicit.sort_by_key(|coord| {
                let hits = hitting.contains(coord);
                (!hits, canonical_coord_key(frame, *coord))
            });
        }
        if explicit.is_empty() {
            return None;
        }

        let turn_start_legal = ((self.zone.pair_commutation || self.width.vcf_pair_complete)
            && pair.is_none()
            && matches!(state.phase(), TurnPhase::FirstStone)
            && threats::placements_remaining(state) == 2)
            .then(|| {
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                legal.sort_by_key(|coord| raw_coord_key(*coord));
                legal
            });
        let mut edges = Vec::with_capacity(explicit.len());
        for &mv in &explicit {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                return None;
            };
            let pair_context = turn_start_legal.as_ref().and_then(|legal| {
                (result.outcome.is_none() && matches!(state.phase(), TurnPhase::SecondStone { .. }))
                    .then(|| PairContext {
                        first: mv,
                        turn_start_legal: legal.clone(),
                    })
            });
            let child = self.prove(state, claimant, ply.checked_add(1)?, pair_context.as_ref());
            state.undo(delta);
            let child = child?; // Unknown poisons the universal claim.
            edges.push(CertEdge { mv, child });
        }

        if let Some(zone) = zone {
            loop {
                let required =
                    zone_certificate_extras(state, claimant, zone.d, &edges, &self.arena)?;
                let mut added = required
                    .into_iter()
                    .filter(|mv| !explicit.contains(mv))
                    .collect::<Vec<_>>();
                if added.is_empty() {
                    break;
                }
                let frame = canonical_frame(state);
                added.sort_by_key(|coord| canonical_coord_key(frame, *coord));
                for mv in added {
                    let Ok((_result, delta)) = state.apply_with_delta(Placement { coord: mv })
                    else {
                        return None;
                    };
                    let child = self.prove(state, claimant, ply.checked_add(1)?, None);
                    state.undo(delta);
                    let child = child?;
                    explicit.push(mv);
                    edges.push(CertEdge { mv, child });
                }
            }
        }

        let commutations = turn_start_legal
            .as_ref()
            .map(|legal| pair_commutations(legal, &edges, &self.arena))
            .unwrap_or_default();
        let explicit_edge_count = edges.len();

        self.alloc_node(
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
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

    fn lookup_shared(
        &mut self,
        key: &PositionKey,
        claimant: Player,
        depth: usize,
    ) -> Option<CertNodeId> {
        let proof = self.shared_tt.as_ref()?.lookup_cloned(key, claimant)?;
        self.import_cached_proof(proof, depth)
    }

    fn remember_proof(&mut self, key: PositionKey, claimant: Player, node: CertNodeId) {
        self.tt.insert(key.clone(), claimant, node);
        // Persistent promotion is aimed at reusable forcing structure.  A
        // factual leaf is cheaper to re-establish than to compact, allocate,
        // and retain, while every non-leaf fragment still owns its leaves.
        // The solve root is offered separately after final compaction.
        let promotes_structure = matches!(
            self.arena.get(node as usize),
            Some(CertNode::Choice { .. } | CertNode::Universal { .. })
        );
        if promotes_structure
            && self
                .shared_tt
                .as_ref()
                .is_some_and(|shared| shared.could_admit_minimal(&key))
        {
            if let Some(proof) = CachedProof::from_arena_limited(
                &self.arena,
                node,
                MAX_PROMOTED_FRAGMENT_NODES,
                MAX_PROMOTED_FRAGMENT_EDGES,
            ) {
                self.insert_shared(key, claimant, proof);
            }
        }
        self.observe_tt_bytes();
    }

    fn insert_shared(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) {
        if let Some(shared) = self.shared_tt.as_deref_mut() {
            shared.insert(key, claimant, proof);
        }
        self.observe_tt_bytes();
    }

    fn can_admit_compact(&self, key: &PositionKey, nodes: &[CertNode]) -> bool {
        self.shared_tt
            .as_ref()
            .is_some_and(|shared| shared.could_admit_compact(key, nodes))
    }

    /// Import is atomic with respect to the live arena: every structural and
    /// resource check happens against the owned clone before any node is
    /// appended.  A fragment that does not fit is merely a cache miss.
    fn import_cached_proof(&mut self, mut proof: CachedProof, depth: usize) -> Option<CertNodeId> {
        proof.validate()?;
        if proof.resolution_t > self.semantic_horizon
            || proof
                .zone_build_t
                .is_some_and(|build_t| self.semantic_horizon > build_t)
            || depth.checked_add(proof.height)? > MAX_SEARCH_DEPTH
            || self.arena.len().checked_add(proof.nodes.len())? > MAX_CERT_NODES
            || self.edge_count.checked_add(proof.explicit_edges)? > MAX_CERT_EDGES
        {
            return None;
        }

        let base = self.arena.len();
        let final_len = base.checked_add(proof.nodes.len())?;
        u32::try_from(final_len).ok()?;
        for node in &mut proof.nodes {
            remap_node_ids_with_offset(node, base, final_len)?;
        }
        let root = offset_node_id(proof.root_node, base, final_len)?;
        self.arena.append(&mut proof.nodes);
        self.edge_count += proof.explicit_edges;
        Some(root)
    }

    fn observe_tt_bytes(&mut self) {
        let shared = self
            .shared_tt
            .as_ref()
            .map(|cache| cache.current_bytes)
            .unwrap_or(0);
        self.peak_tt_bytes = self
            .peak_tt_bytes
            .max(self.tt.current_bytes.saturating_add(shared));
    }
}

fn arena_subtree_contains_zone(arena: &[CertNode], root: CertNodeId) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.get(id as usize) {
            Some(CertNode::Choice { child, .. }) => stack.push(*child),
            Some(CertNode::Universal { edges, zone, .. }) => {
                if zone.is_some() {
                    return true;
                }
                stack.extend(edges.iter().map(|edge| edge.child));
            }
            Some(_) => {}
            None => return true,
        }
    }
    false
}

fn pair_commutations(
    turn_start_legal: &[HexCoord],
    parent_edges: &[CertEdge],
    arena: &[CertNode],
) -> Vec<CertCommutation> {
    let mut result = Vec::new();
    for first_edge in parent_edges {
        let Some(CertNode::Universal {
            edges: first_replies,
            ..
        }) = arena.get(first_edge.child as usize)
        else {
            continue;
        };
        for &omitted_second in turn_start_legal {
            if raw_coord_key(omitted_second) >= raw_coord_key(first_edge.mv)
                || first_replies.iter().any(|edge| edge.mv == omitted_second)
            {
                continue;
            }
            let Some(mirror_edge) = parent_edges.iter().find(|edge| edge.mv == omitted_second)
            else {
                continue;
            };
            let Some(CertNode::Universal {
                edges: mirror_replies,
                ..
            }) = arena.get(mirror_edge.child as usize)
            else {
                continue;
            };
            if mirror_replies.iter().any(|edge| edge.mv == first_edge.mv) {
                result.push(CertCommutation {
                    first: first_edge.mv,
                    omitted_second,
                    first_child: first_edge.child,
                    mirror_child: mirror_edge.child,
                });
            }
        }
    }
    result.sort_by_key(|item| {
        (
            raw_coord_key(item.first),
            raw_coord_key(item.omitted_second),
        )
    });
    result
}

fn restrict_pair_candidates(candidates: &mut Vec<HexCoord>, pair: &PairContext) {
    candidates.retain(|mv| pair_candidate_allowed(*mv, pair));
}

fn pair_candidate_allowed(mv: HexCoord, pair: &PairContext) -> bool {
    raw_coord_key(mv) > raw_coord_key(pair.first) || !pair.turn_start_legal.contains(&mv)
}

#[derive(Clone)]
struct Candidate {
    coord: HexCoord,
    /// Maximum pre-placement stone count among active claimant windows that
    /// this coordinate extends.  Larger means a lower initial proof number.
    strength: u8,
    priority_class: u8,
    child_threats: usize,
    /// This move occupies an empty of an active defender count-four/five
    /// window.  Wide mode must retain such tempo-preserving blocks even when
    /// the cell is not yet in a claimant-owned window.
    defender_block: bool,
    /// Distinct count-two windows through this cell.  In pair-complete mode
    /// this is the primary ordering key within the newly admitted tier.
    pair_start_degree: usize,
    /// Distinct local count-one windows through this cell.  This orders the r3
    /// escalation tier after pair starts.
    seed_degree: usize,
    /// Nearest claimant stone, used only to break widened-tier ordering ties.
    own_proximity: i16,
    /// Count-three claimant windows this placement turns into live threats.
    /// Their pre-placement empties let SecondStone reply forcedness be derived
    /// without rescanning or mutating the engine state.
    created_threats: Vec<Vec<HexCoord>>,
}

struct CandidateBatch {
    candidates: Vec<Candidate>,
    claimant_threats: Vec<Vec<HexCoord>>,
    defender_threats: Vec<Vec<HexCoord>>,
}

/// Exact OR restriction: every returned placement changes an active claimant
/// length-six window with at least three stones into a >=4 threat (or a win).
/// Omitting all other claimant moves can only miss a winning proof.
fn threat_creating_moves(state: &RustHexoState, claimant: Player) -> CandidateBatch {
    threat_creating_moves_with_threshold(state, claimant, 3)
}

fn threat_creating_moves_with_threshold(
    state: &RustHexoState,
    claimant: Player,
    minimum_strength: u8,
) -> CandidateBatch {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut claimant_threats = Vec::new();
    let mut defender_threats = Vec::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        if entry.count(owner) >= 4 {
            if owner == claimant {
                claimant_threats.push(entry.empty_cells());
            } else {
                defender_threats.push(entry.empty_cells());
            }
        }
        if owner != claimant {
            continue;
        }
        let strength = entry.count(claimant);
        if strength < minimum_strength {
            continue;
        }
        let empties = entry.empty_cells();
        for &coord in &empties {
            if strength == 1
                && !state
                    .board()
                    .occupied_cells()
                    .iter()
                    .any(|stone| hex_distance(coord, *stone) <= 3)
            {
                continue;
            }
            let created = (strength == 3).then(|| {
                empties
                    .iter()
                    .copied()
                    .filter(|empty| *empty != coord)
                    .collect::<Vec<_>>()
            });
            if let Some(existing) = candidates.iter_mut().find(|item| item.coord == coord) {
                existing.strength = existing.strength.max(strength);
                if strength == 2 {
                    existing.pair_start_degree += 1;
                } else if strength == 1 {
                    existing.seed_degree += 1;
                }
                if let Some(created) = created {
                    existing.created_threats.push(created);
                }
            } else {
                candidates.push(Candidate {
                    coord,
                    strength,
                    priority_class: u8::MAX,
                    child_threats: 0,
                    defender_block: false,
                    pair_start_degree: usize::from(strength == 2),
                    seed_degree: usize::from(strength == 1),
                    own_proximity: i16::MAX,
                    created_threats: created.into_iter().collect(),
                });
            }
        }
    }
    if minimum_strength < 3 {
        for coord in defender_threats.iter().flatten().copied() {
            if let Some(existing) = candidates.iter_mut().find(|item| item.coord == coord) {
                existing.defender_block = true;
            } else {
                candidates.push(Candidate {
                    coord,
                    strength: 0,
                    priority_class: u8::MAX,
                    child_threats: 0,
                    defender_block: true,
                    pair_start_degree: 0,
                    seed_degree: 0,
                    own_proximity: i16::MAX,
                    created_threats: Vec::new(),
                });
            }
        }
    }
    candidates.sort_by_key(|item| (Reverse(item.strength), item.coord.q, item.coord.r));
    CandidateBatch {
        candidates,
        claimant_threats,
        defender_threats,
    }
}

/// Static proof-number initialization derived from WindowStore membership.
/// The candidate set is unchanged.  A count-four extension is an immediate
/// lambda-one proof after a one-stone remainder; otherwise same-turn builds
/// precede replies, and newly created threat-window count orders each class.
fn ordered_threat_creating_moves(state: &RustHexoState, claimant: Player) -> Vec<Candidate> {
    ordered_threat_creating_moves_with_width(state, claimant, WidthOptions::default())
}

fn ordered_threat_creating_moves_with_width(
    state: &RustHexoState,
    claimant: Player,
    width: WidthOptions,
) -> Vec<Candidate> {
    let CandidateBatch {
        mut candidates,
        claimant_threats,
        defender_threats,
    } = if width.vcf_pair_complete {
        threat_creating_moves_with_threshold(state, claimant, 1)
    } else {
        threat_creating_moves(state, claimant)
    };
    for candidate in &mut candidates {
        candidate.child_threats = claimant_threats.len() + candidate.created_threats.len();
        if width.vcf_pair_complete && candidate.strength <= 2 {
            candidate.own_proximity = state
                .board()
                .occupied_cells()
                .iter()
                .copied()
                .filter(|coord| state.board().get(*coord) == Some(claimant))
                .map(|coord| hex_distance(candidate.coord, coord))
                .min()
                .unwrap_or(i16::MAX);
        }
        candidate.priority_class = if candidate.defender_block && candidate.strength < 4 {
            match state.phase() {
                TurnPhase::FirstStone => 1,
                TurnPhase::SecondStone { .. } => 2,
                TurnPhase::Opening => 3,
            }
        } else {
            match state.phase() {
                TurnPhase::FirstStone if candidate.strength >= 4 => 0,
                TurnPhase::FirstStone => 2,
                TurnPhase::SecondStone { .. } => {
                    post_turn_reply_priority(candidate, &claimant_threats, &defender_threats)
                }
                TurnPhase::Opening => 3,
            }
        };
    }
    if candidates.len() <= 1 {
        return candidates;
    }
    let frame = canonical_frame(state);
    if width.vcf_pair_complete {
        candidates.sort_by_key(|item| {
            let width_tier = match (item.defender_block, item.strength) {
                (true, _) | (_, 3..) => 0,
                (_, 2) => 1,
                _ => 2,
            };
            let canonical = canonical_coord_key(frame, item.coord);
            (
                width_tier,
                if width_tier == 0 {
                    item.priority_class
                } else {
                    0
                },
                Reverse(match width_tier {
                    0 => item.child_threats,
                    1 => item.pair_start_degree,
                    _ => item.seed_degree,
                }),
                Reverse(if width_tier == 0 { item.strength } else { 0 }),
                if width_tier == 0 {
                    0
                } else {
                    item.own_proximity
                },
                canonical.0,
                canonical.1,
            )
        });
    } else {
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
    }
    candidates
}

/// Reconstruct exactly the child threat-cost class after a claimant's
/// SecondStone at a reachable unresolved search node, without mutating the
/// engine state.  (`prove` removes terminal/lambda-one parents first, so a
/// strength-five completion and its off-path ordering distinctions cannot
/// reach this function.)  Window masks change only in windows containing
/// `coord`: claimant windows gain one bit and defender windows become blocked.
/// The returned classes match the former child `analyze` probe: immediate
/// claimant proof (0), fully forced two-hit reply (1), or a reply with
/// spare/counter-winning budget (3).
fn post_turn_reply_priority(
    candidate: &Candidate,
    claimant_threats: &[Vec<HexCoord>],
    defender_threats: &[Vec<HexCoord>],
) -> u8 {
    // The child is FirstStone (B=2), so any defender count-four/five not
    // blocked by this placement is win-now.  Child analysis gives it
    // precedence over claimant threats.
    if defender_threats
        .iter()
        .any(|empties| !empties.contains(&candidate.coord))
    {
        return 3;
    }
    match min_hitting_set_at_most_two(
        claimant_threats,
        &candidate.created_threats,
        candidate.coord,
    ) {
        None => 0,
        Some(2) => 1,
        Some(_) => 3,
    }
}

fn min_hitting_set_at_most_two(
    existing: &[Vec<HexCoord>],
    created: &[Vec<HexCoord>],
    placed: HexCoord,
) -> Option<u8> {
    if existing.is_empty() && created.is_empty() {
        return Some(0);
    }
    let sets = || existing.iter().chain(created.iter());
    if sets().any(|set| !set.iter().any(|coord| *coord != placed)) {
        return None;
    }
    let mut universe = Vec::new();
    for set in sets() {
        for &coord in set {
            if coord != placed && !universe.contains(&coord) {
                universe.push(coord);
            }
        }
    }
    if universe
        .iter()
        .any(|coord| sets().all(|set| set.contains(coord)))
    {
        return Some(1);
    }
    for left in 0..universe.len() {
        for right in (left + 1)..universe.len() {
            if sets().all(|set| set.contains(&universe[left]) || set.contains(&universe[right])) {
                return Some(2);
            }
        }
    }
    None
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

fn remaining_defender_placements_for_horizon(
    state: &RustHexoState,
    claimant: Player,
    horizon: u32,
) -> Option<u32> {
    // Zone machinery only distinguishes budgets 0..=5 (>= 6 takes the full
    // legal set) and production horizons sit ~12 plies out, so any count past
    // this band signals a corrupted/degenerate horizon — bail (None => no
    // zone, full legal set) instead of walking a `u32::MAX` horizon.
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

fn all_incident_windows_two_coloured(state: &RustHexoState, cell: HexCoord) -> bool {
    for axis in Axis::ALL {
        for offset in 0..6i16 {
            let key = WindowKey {
                start: cell - axis.vector().scale(offset),
                axis,
            };
            let Some(entry) = state
                .board()
                .windows()
                .entries()
                .find(|entry| entry.key() == key)
            else {
                return false;
            };
            if entry.count(Player::Player0) == 0 || entry.count(Player::Player1) == 0 {
                return false;
            }
        }
    }
    true
}

fn zone_initial_candidates(
    state: &RustHexoState,
    claimant: Player,
    d: u32,
    options: ZoneSearchCaps,
) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| (coord.q, coord.r));
    if d >= 6 {
        return legal;
    }
    let defender = claimant.other();
    let mut out = hitting_universe(state, claimant);
    for entry in state.board().windows().entries() {
        let attacker_term = entry.active_player() == Some(claimant)
            && entry.count(claimant) >= if options.count2_threshold { 2 } else { 1 };
        let defender_term = entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)) >= 6u32.saturating_sub(d);
        if attacker_term || defender_term {
            out.extend(entry.empty_cells());
        }
    }
    out.sort_by_key(|coord| (coord.q, coord.r));
    out.dedup();
    out.retain(|cell| {
        legal
            .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
            .is_ok()
    });
    if options.stale_area_filter {
        out.retain(|cell| !all_incident_windows_two_coloured(state, *cell));
    }
    let hitting = hitting_universe(state, claimant);
    let frame = canonical_frame(state);
    out.sort_by_key(|coord| (!hitting.contains(coord), canonical_coord_key(frame, *coord)));
    out
}

fn arena_core(arena: &[CertNode], root: CertNodeId, out: &mut Vec<HexCoord>) -> Option<()> {
    match arena.get(root as usize)? {
        CertNode::OrCompletion { mv, witness, .. } => {
            out.push(*mv);
            out.extend(witness.cells());
        }
        CertNode::Win { witness, .. } => out.extend(witness.cells()),
        CertNode::Loss { witnesses, .. } => {
            for witness in witnesses {
                out.extend(witness.cells());
            }
        }
        CertNode::Choice { mv, child } => {
            out.push(*mv);
            arena_core(arena, *child, out)?;
        }
        CertNode::Universal { edges, .. } => {
            for edge in edges {
                arena_core(arena, edge.child, out)?;
            }
        }
    }
    Some(())
}

fn zone_certificate_extras(
    state: &RustHexoState,
    claimant: Player,
    d: u32,
    edges: &[CertEdge],
    arena: &[CertNode],
) -> Option<Vec<HexCoord>> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| (coord.q, coord.r));
    let mut protected = Vec::new();
    for edge in edges {
        arena_core(arena, edge.child, &mut protected)?;
    }
    let defender = claimant.other();
    for entry in state.board().windows().entries() {
        if entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)).saturating_add(d) >= 6
        {
            protected.extend(entry.empty_cells());
        }
    }
    protected.sort_by_key(|coord| (coord.q, coord.r));
    protected.dedup();
    let stones = state.board().occupied_cells();
    let pending = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
                .is_err()
                && !stones.contains(cell)
        })
        .collect::<Vec<_>>();
    let mut required = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
                .is_ok()
        })
        .collect::<Vec<_>>();
    if !pending.is_empty() {
        let radius = i32::try_from(d.saturating_mul(8)).unwrap_or(i32::MAX);
        required.extend(legal.iter().copied().filter(|cell| {
            pending
                .iter()
                .any(|target| i32::from(hex_distance(*cell, *target)) <= radius)
        }));
    }
    required.sort_by_key(|coord| (coord.q, coord.r));
    required.dedup();
    Some(required)
}

/// Choose the lexicographically least D6 image of the full semantic position.
/// Search ties are compared in this frame, so rotating/reflection-transforming
/// an input cannot change which proof-cost class is expanded first merely due
/// to raw `(q,r)` order.  The TT remains uncanonicalized and still uses exact
/// raw-position equality.
fn canonical_frame(state: &RustHexoState) -> u8 {
    let stone_count = state.board().occupied_cells().len();
    let mut best_phase: Option<(u8, i32, i32)> = None;
    let mut best_stones = Vec::with_capacity(stone_count);
    let mut candidate_stones = Vec::with_capacity(stone_count);
    let mut best_symmetry = 0;
    for symmetry in 0..12u8 {
        let phase = match state.phase() {
            TurnPhase::Opening => (0, 0, 0),
            TurnPhase::FirstStone => (1, 0, 0),
            TurnPhase::SecondStone { first } => {
                let (q, r) = d6_coord_i32(first, symmetry);
                (2, q, r)
            }
        };
        candidate_stones.clear();
        candidate_stones.extend(state.board().occupied_cells().iter().map(|&coord| {
            let (q, r) = d6_coord_i32(coord, symmetry);
            let owner = state.board().get(coord).expect("occupied cell has owner");
            (q, r, player_code(owner))
        }));
        candidate_stones.sort_unstable();
        if best_phase
            .as_ref()
            .is_none_or(|best| (&phase, &candidate_stones) < (best, &best_stones))
        {
            best_phase = Some(phase);
            best_symmetry = symmetry;
            std::mem::swap(&mut best_stones, &mut candidate_stones);
        }
    }
    debug_assert!(best_phase.is_some(), "D6 contains identity");
    best_symmetry
}

fn canonical_coord_key(frame: u8, coord: HexCoord) -> (i32, i32) {
    d6_coord_i32(coord, frame)
}

fn raw_coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedProof {
    nodes: Vec<CertNode>,
    root_node: CertNodeId,
    explicit_edges: usize,
    /// Maximum number of certificate edges below `root_node`.
    height: usize,
    /// Maximum exact resolution label over all contained typed leaves.
    resolution_t: u32,
    /// Minimum zone-build deadline over all contained zoned components.
    zone_build_t: Option<u32>,
}

impl CachedProof {
    fn from_arena_limited(
        arena: &[CertNode],
        root: CertNodeId,
        max_nodes: usize,
        max_edges: usize,
    ) -> Option<Self> {
        let (nodes, root_node) = compact_certificate_limited(arena, root, max_nodes, max_edges)?;
        Self::from_compact(nodes, root_node)
    }

    fn from_compact(nodes: Vec<CertNode>, root_node: CertNodeId) -> Option<Self> {
        if nodes.is_empty() || root_node as usize >= nodes.len() {
            return None;
        }
        let mut heights = vec![0usize; nodes.len()];
        let mut explicit_edges = 0usize;
        let mut resolution_t = 0u32;
        let mut zone_build_t: Option<u32> = None;
        for (index, node) in nodes.iter().enumerate() {
            resolution_t = resolution_t.max(node_resolution(node));
            if let CertNode::Universal {
                zone: Some(zone), ..
            } = node
            {
                zone_build_t = Some(
                    zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                );
            }
            heights[index] = match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
                CertNode::Choice { child, .. } => {
                    let child = *child as usize;
                    if child >= index {
                        return None;
                    }
                    heights[child].checked_add(1)?
                }
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    explicit_edges = explicit_edges.checked_add(edges.len())?;
                    let mut height = 0usize;
                    for edge in edges {
                        let child = edge.child as usize;
                        if child >= index {
                            return None;
                        }
                        height = height.max(heights[child].checked_add(1)?);
                    }
                    for item in commutations {
                        for child in [item.first_child, item.mirror_child] {
                            let child = child as usize;
                            if child >= index {
                                return None;
                            }
                            height = height.max(heights[child].checked_add(1)?);
                        }
                    }
                    height
                }
            };
        }
        let height = heights[root_node as usize];
        let proof = Self {
            nodes,
            root_node,
            explicit_edges,
            height,
            resolution_t,
            zone_build_t,
        };
        proof.validate()?;
        Some(proof)
    }

    fn validate(&self) -> Option<()> {
        if self.nodes.is_empty()
            || self.nodes.len() > MAX_CERT_NODES
            || self.root_node as usize >= self.nodes.len()
            || self.height > MAX_CERT_DEPTH
            || self.explicit_edges > MAX_CERT_EDGES
        {
            return None;
        }
        let rebuilt = Self::from_compact_unchecked_metadata(&self.nodes, self.root_node)?;
        (rebuilt
            == (
                self.explicit_edges,
                self.height,
                self.resolution_t,
                self.zone_build_t,
            ))
            .then_some(())
    }

    fn from_compact_unchecked_metadata(
        nodes: &[CertNode],
        root_node: CertNodeId,
    ) -> Option<(usize, usize, u32, Option<u32>)> {
        let mut heights = vec![0usize; nodes.len()];
        let mut explicit_edges = 0usize;
        let mut resolution_t = 0u32;
        let mut zone_build_t: Option<u32> = None;
        for (index, node) in nodes.iter().enumerate() {
            resolution_t = resolution_t.max(node_resolution(node));
            if let CertNode::Universal {
                zone: Some(zone), ..
            } = node
            {
                zone_build_t = Some(
                    zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                );
            }
            heights[index] = match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
                CertNode::Choice { child, .. } => {
                    let child = *child as usize;
                    if child >= index {
                        return None;
                    }
                    heights[child].checked_add(1)?
                }
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    explicit_edges = explicit_edges.checked_add(edges.len())?;
                    let mut height = 0usize;
                    for edge in edges {
                        let child = edge.child as usize;
                        if child >= index {
                            return None;
                        }
                        height = height.max(heights[child].checked_add(1)?);
                    }
                    for item in commutations {
                        for child in [item.first_child, item.mirror_child] {
                            let child = child as usize;
                            if child >= index {
                                return None;
                            }
                            height = height.max(heights[child].checked_add(1)?);
                        }
                    }
                    height
                }
            };
        }
        Some((
            explicit_edges,
            heights[root_node as usize],
            resolution_t,
            zone_build_t,
        ))
    }

    fn heap_bytes(&self) -> usize {
        let mut bytes = allocation_bytes(self.nodes.capacity(), size_of::<CertNode>());
        for node in &self.nodes {
            match node {
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    bytes = bytes.saturating_add(allocation_bytes(
                        edges.capacity(),
                        size_of::<CertEdge>(),
                    ));
                    bytes = bytes.saturating_add(allocation_bytes(
                        commutations.capacity(),
                        size_of::<CertCommutation>(),
                    ));
                }
                // Adaptive LOSS contracts own a witness vector too — omitting
                // it made the cap admission understate real heap (Codex
                // review, cache accounting).
                CertNode::Loss { witnesses, .. } => {
                    bytes = bytes.saturating_add(allocation_bytes(
                        witnesses.capacity(),
                        size_of::<WindowKey>(),
                    ));
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
        bytes
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SharedTtEntry {
    hash: u64,
    key: PositionKey,
    claimant: Player,
    proof: CachedProof,
}

impl SharedTtEntry {
    fn heap_bytes(&self) -> usize {
        self.key
            .heap_bytes()
            .saturating_add(self.proof.heap_bytes())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SharedProofCache {
    slots: Vec<Option<SharedTtEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
}

impl SharedProofCache {
    fn new(cap: usize, hash_mask: u64) -> Self {
        let slot_count = (cap / TARGET_BYTES_PER_SHARED_TT_SLOT).min(MAX_TT_SLOTS);
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
        let base = allocation_bytes(slots.capacity(), size_of::<Option<SharedTtEntry>>());
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

    fn reconfigure(&mut self, cap: usize, hash_mask: u64) {
        if self.cap != cap || self.hash_mask != hash_mask {
            *self = Self::new(cap, hash_mask);
        } else {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Drop every retained fragment (profile isolation on option changes).
    fn clear(&mut self) {
        *self = Self::new(self.cap, self.hash_mask);
    }

    fn lookup_cloned(&self, key: &PositionKey, claimant: Player) -> Option<CachedProof> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.claimant == claimant && entry.key == *key)
            .then(|| entry.proof.clone())
    }

    fn could_admit_minimal(&self, key: &PositionKey) -> bool {
        self.could_admit_heap(key, allocation_bytes(1, size_of::<CertNode>()))
    }

    fn could_admit_compact(&self, key: &PositionKey, nodes: &[CertNode]) -> bool {
        let mut proof_heap = allocation_bytes(nodes.len(), size_of::<CertNode>());
        for node in nodes {
            match node {
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    proof_heap = proof_heap
                        .saturating_add(allocation_bytes(edges.len(), size_of::<CertEdge>()));
                    proof_heap = proof_heap.saturating_add(allocation_bytes(
                        commutations.len(),
                        size_of::<CertCommutation>(),
                    ));
                }
                CertNode::Loss { witnesses, .. } => {
                    proof_heap = proof_heap.saturating_add(allocation_bytes(
                        witnesses.len(),
                        size_of::<WindowKey>(),
                    ));
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
        self.could_admit_heap(key, proof_heap)
    }

    fn could_admit_heap(&self, key: &PositionKey, proof_heap: usize) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(SharedTtEntry::heap_bytes)
            .unwrap_or(0);
        self.current_bytes
            .saturating_sub(old_heap)
            .saturating_add(key.heap_bytes())
            .saturating_add(proof_heap)
            <= self.cap
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) {
        if self.slots.is_empty() || proof.validate().is_none() {
            return;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(SharedTtEntry::heap_bytes)
            .unwrap_or(0);
        let entry = SharedTtEntry {
            hash,
            key,
            claimant,
            proof,
        };
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(entry.heap_bytes());
        if candidate_bytes > self.cap {
            return;
        }
        self.slots[index] = Some(entry);
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
    }

    #[cfg(test)]
    fn recomputed_bytes(&self) -> usize {
        let base = allocation_bytes(self.slots.capacity(), size_of::<Option<SharedTtEntry>>());
        base.saturating_add(
            self.slots
                .iter()
                .flatten()
                .map(SharedTtEntry::heap_bytes)
                .sum::<usize>(),
        )
    }
}

fn allocation_bytes(capacity: usize, element_size: usize) -> usize {
    if capacity == 0 {
        0
    } else {
        capacity
            .saturating_mul(element_size)
            .saturating_add(ALLOC_OVERHEAD)
    }
}

fn offset_node_id(id: CertNodeId, base: usize, final_len: usize) -> Option<CertNodeId> {
    let index = id as usize;
    let mapped = base.checked_add(index)?;
    (mapped < final_len)
        .then(|| u32::try_from(mapped).ok())
        .flatten()
}

fn remap_node_ids_with_offset(node: &mut CertNode, base: usize, final_len: usize) -> Option<()> {
    match node {
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
        CertNode::Choice { child, .. } => {
            *child = offset_node_id(*child, base, final_len)?;
        }
        CertNode::Universal {
            edges,
            commutations,
            ..
        } => {
            for edge in edges {
                edge.child = offset_node_id(edge.child, base, final_len)?;
            }
            for item in commutations {
                item.first_child = offset_node_id(item.first_child, base, final_len)?;
                item.mirror_child = offset_node_id(item.mirror_child, base, final_len)?;
            }
        }
    }
    Some(())
}

/// Remove abandoned OR branches from the certificate arena and remap every
/// reachable child.  The resulting certificate has no orphan nodes, which the
/// independent verifier requires.
fn compact_certificate(
    arena: &[CertNode],
    root: CertNodeId,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    compact_certificate_limited(arena, root, MAX_CERT_NODES, MAX_CERT_EDGES)
}

fn compact_certificate_limited(
    arena: &[CertNode],
    root: CertNodeId,
    max_nodes: usize,
    max_edges: usize,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    fn copy(
        old: CertNodeId,
        arena: &[CertNode],
        remap: &mut [Option<CertNodeId>],
        visiting: &mut [bool],
        out: &mut Vec<CertNode>,
        edge_count: &mut usize,
        max_nodes: usize,
        max_edges: usize,
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
            CertNode::OrCompletion {
                mv,
                witness,
                completion_ply,
            } => CertNode::OrCompletion {
                mv: *mv,
                witness: *witness,
                completion_ply: *completion_ply,
            },
            CertNode::Win {
                witness,
                count,
                budget,
                resolution_ply,
            } => CertNode::Win {
                witness: *witness,
                count: *count,
                budget: *budget,
                resolution_ply: *resolution_ply,
            },
            CertNode::Loss {
                witnesses,
                resolution_ply,
            } => CertNode::Loss {
                witnesses: witnesses.clone(),
                resolution_ply: *resolution_ply,
            },
            CertNode::Choice { mv, child } => CertNode::Choice {
                mv: *mv,
                child: copy(
                    *child, arena, remap, visiting, out, edge_count, max_nodes, max_edges,
                )?,
            },
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                *edge_count = edge_count.checked_add(edges.len())?;
                if *edge_count > max_edges {
                    return None;
                }
                let mut mapped_edges = Vec::with_capacity(edges.len());
                for edge in edges {
                    mapped_edges.push(CertEdge {
                        mv: edge.mv,
                        child: copy(
                            edge.child, arena, remap, visiting, out, edge_count, max_nodes,
                            max_edges,
                        )?,
                    });
                }
                let mut mapped_commutations = Vec::with_capacity(commutations.len());
                for item in commutations {
                    mapped_commutations.push(CertCommutation {
                        first: item.first,
                        omitted_second: item.omitted_second,
                        first_child: copy(
                            item.first_child,
                            arena,
                            remap,
                            visiting,
                            out,
                            edge_count,
                            max_nodes,
                            max_edges,
                        )?,
                        mirror_child: copy(
                            item.mirror_child,
                            arena,
                            remap,
                            visiting,
                            out,
                            edge_count,
                            max_nodes,
                            max_edges,
                        )?,
                    });
                }
                CertNode::Universal {
                    edges: mapped_edges,
                    implicit_dispatch: *implicit_dispatch,
                    zone: *zone,
                    commutations: mapped_commutations,
                }
            }
        };
        visiting[index] = false;
        if out.len() >= max_nodes {
            return None;
        }
        let mapped = u32::try_from(out.len()).ok()?;
        out.push(mapped_node);
        remap[index] = Some(mapped);
        Some(mapped)
    }

    if arena.len() > MAX_CERT_NODES || max_nodes > MAX_CERT_NODES || max_edges > MAX_CERT_EDGES {
        return None;
    }
    let mut remap = vec![None; arena.len()];
    let mut visiting = vec![false; arena.len()];
    let mut nodes = Vec::new();
    let mut edge_count = 0usize;
    let root_node = copy(
        root,
        arena,
        &mut remap,
        &mut visiting,
        &mut nodes,
        &mut edge_count,
        max_nodes,
        max_edges,
    )?;
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
    use hexo_engine::{apply_placement, WindowStore};

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

    fn cache_test_leaf() -> CertNode {
        CertNode::Win {
            witness: WindowKey {
                start: HexCoord::ZERO,
                axis: hexo_engine::Axis::Q,
            },
            count: 5,
            budget: 2,
            resolution_ply: 1,
        }
    }

    /// Geometry-only scaffold for the proof program's adversarial positions.
    /// P2 materializes these ownership maps into matched-horizon solver roots;
    /// keeping the canonical stone sets here makes mutations independent of a
    /// particular legal replay ordering.
    struct ZoneFixtureSpec {
        stones: Vec<(HexCoord, Player)>,
        focus: HexCoord,
    }

    fn g1_junction_spec() -> ZoneFixtureSpec {
        let attacker = Player::Player0;
        let defender = attacker.other();
        let arms = [
            (3, 0),
            (4, 0),
            (5, 0),
            (-3, 0),
            (-4, 0),
            (-5, 0),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, -3),
            (0, -4),
            (0, -5),
        ];
        let pin = [(8, -2), (11, -5), (12, -6), (13, -7)];
        let caps = [(6, 0), (-6, 0), (0, 6), (0, -6)];
        let mut stones = arms
            .into_iter()
            .chain(pin)
            .map(|(q, r)| (HexCoord::new(q, r), attacker))
            .collect::<Vec<_>>();
        stones.extend(
            caps.into_iter()
                .map(|(q, r)| (HexCoord::new(q, r), defender)),
        );
        ZoneFixtureSpec {
            stones,
            focus: HexCoord::ZERO,
        }
    }

    fn g3_counterfork_spec() -> ZoneFixtureSpec {
        let attacker = Player::Player0;
        let defender = attacker.other();
        let defender_arms = [
            (8, 0),
            (9, 0),
            (10, 0),
            (5, 3),
            (5, 4),
            (5, 5),
            (8, -3),
            (9, -4),
            (10, -5),
            (5, 0),
        ];
        let attacker_scaffold = [(-9, 4), (-8, 4), (-6, 4), (-4, 4)];
        let mut stones = defender_arms
            .into_iter()
            .map(|(q, r)| (HexCoord::new(q, r), defender))
            .collect::<Vec<_>>();
        stones.extend(
            attacker_scaffold
                .into_iter()
                .map(|(q, r)| (HexCoord::new(q, r), attacker)),
        );
        ZoneFixtureSpec {
            stones,
            focus: HexCoord::new(5, 0),
        }
    }

    #[test]
    fn zone_adversary_geometry_scaffolds_match_python_reference() {
        let g1 = g1_junction_spec();
        let g1_store = WindowStore::from_placements(&g1.stones);
        let junction_routes = g1_store
            .entries()
            .filter(|entry| {
                entry.key().contains(g1.focus)
                    && entry.count(Player::Player0) == 3
                    && entry.count(Player::Player1) == 0
            })
            .count();
        assert_eq!(junction_routes, 4, "G1 must retain all four junction arms");
        let pins = g1_store
            .entries()
            .filter(|entry| {
                entry.count(Player::Player0) == 4
                    && entry.count(Player::Player1) == 0
                    && entry.empty_cells().len() == 2
            })
            .count();
        assert_eq!(pins, 1, "G1 pin is deliberately a single live window");

        let g3 = g3_counterfork_spec();
        let g3_store = WindowStore::from_placements(&g3.stones);
        let fork_windows = g3_store
            .entries()
            .filter(|entry| {
                entry.key().contains(g3.focus)
                    && entry.count(Player::Player1) == 4
                    && entry.count(Player::Player0) == 0
            })
            .map(|entry| entry.empty_cells())
            .collect::<Vec<_>>();
        assert_eq!(fork_windows.len(), 3);
        for left in 0..fork_windows.len() {
            for right in left + 1..fork_windows.len() {
                assert!(
                    fork_windows[left]
                        .iter()
                        .all(|cell| !fork_windows[right].contains(cell)),
                    "G3 threat empties must be pairwise disjoint"
                );
            }
        }
    }

    #[test]
    fn pair_generator_uses_frozen_parent_order_and_keeps_newly_legal_cells() {
        let lower = HexCoord::new(-2, 0);
        let first = HexCoord::new(0, 0);
        let higher = HexCoord::new(2, 0);
        let newly_legal_lower = HexCoord::new(-1, -1);
        let pair = PairContext {
            first,
            turn_start_legal: vec![lower, first, higher],
        };
        let mut candidates = vec![lower, higher, newly_legal_lower];
        restrict_pair_candidates(&mut candidates, &pair);
        assert!(!candidates.contains(&lower));
        assert!(candidates.contains(&higher));
        assert!(candidates.contains(&newly_legal_lower));
    }

    #[test]
    fn pair_complete_width_adds_every_count_two_cell_at_both_turn_plies() {
        let first = pair_width_first_stone_fixture();
        assert_eq!(first.current_player(), Player::Player0);
        assert_eq!(first.phase(), TurnPhase::FirstStone);
        let mut second = first.clone();
        apply_placement(
            &mut second,
            Placement {
                coord: HexCoord::new(2, 0),
            },
        )
        .unwrap();
        assert_eq!(
            second.phase(),
            TurnPhase::SecondStone {
                first: HexCoord::new(2, 0)
            }
        );

        let width = WidthOptions::vcf_pair_complete();
        let mut saw_narrow_candidate = false;
        for state in [&first, &second] {
            let narrow = ordered_threat_creating_moves(state, Player::Player0);
            let wide = ordered_threat_creating_moves_with_width(state, Player::Player0, width);
            saw_narrow_candidate |= !narrow.is_empty();

            let narrow_coords = narrow.iter().map(|item| item.coord).collect::<Vec<_>>();
            let wide_narrow_coords = wide
                .iter()
                .filter(|item| item.strength >= 3)
                .map(|item| item.coord)
                .collect::<Vec<_>>();
            assert_eq!(wide_narrow_coords, narrow_coords);
            assert!(wide.iter().any(|item| item.strength == 2));
            assert!(wide
                .iter()
                .skip(narrow.len())
                .all(|item| item.strength <= 2));
            let first_seed = wide.iter().position(|item| item.strength == 1).unwrap();
            assert!(wide[..first_seed].iter().all(|item| item.strength >= 2));
            assert!(wide[first_seed..].iter().all(|item| item.strength == 1));
            assert!(wide[first_seed..].iter().all(|item| {
                state
                    .board()
                    .occupied_cells()
                    .iter()
                    .any(|stone| hex_distance(item.coord, *stone) <= 3)
            }));

            for entry in state.board().windows().entries().filter(|entry| {
                entry.active_player() == Some(Player::Player0) && entry.count(Player::Player0) == 2
            }) {
                for coord in entry.empty_cells() {
                    assert!(wide.iter().any(|item| item.coord == coord));
                }
            }
        }
        assert!(saw_narrow_candidate);
    }

    #[test]
    fn pair_complete_count_two_order_prefers_forks_then_proximity() {
        let state = pair_width_first_stone_fixture();
        let candidates = ordered_threat_creating_moves_with_width(
            &state,
            Player::Player0,
            WidthOptions::vcf_pair_complete(),
        );
        let frame = canonical_frame(&state);
        let pair_starts = candidates
            .iter()
            .filter(|item| item.strength == 2)
            .collect::<Vec<_>>();
        assert!(pair_starts.len() > 1);
        assert!(pair_starts.windows(2).all(|pair| {
            let key = |item: &Candidate| {
                let canonical = canonical_coord_key(frame, item.coord);
                (
                    Reverse(item.pair_start_degree),
                    item.own_proximity,
                    canonical.0,
                    canonical.1,
                )
            };
            key(pair[0]) <= key(pair[1])
        }));
    }

    #[test]
    fn pair_complete_width_keeps_defender_threat_blocks() {
        let state = forced_defense_fixture();
        let claimant = state.current_player();
        let defender = claimant.other();
        let blocks = hitting_universe(&state, defender);
        assert!(!blocks.is_empty());
        let candidates = ordered_threat_creating_moves_with_width(
            &state,
            claimant,
            WidthOptions::vcf_pair_complete(),
        );
        for block in blocks {
            assert!(candidates
                .iter()
                .any(|candidate| candidate.coord == block && candidate.defender_block));
        }
    }

    #[test]
    fn zone_generator_is_deterministic_and_never_count_truncates() {
        let state = quiet_fixture();
        let claimant = state.current_player().other();
        let d = 1;
        let caps = ZoneSearchCaps {
            enabled: true,
            stale_area_filter: false,
            count2_threshold: false,
            pair_commutation: false,
        };
        let first = zone_initial_candidates(&state, claimant, d, caps);
        let second = zone_initial_candidates(&state, claimant, d, caps);
        assert_eq!(first, second);
        assert!(!first.is_empty());
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        assert!(first.len() <= legal.len());
        assert!(first.iter().all(|mv| legal.contains(mv)));

        let d6 = zone_initial_candidates(&state, claimant, 6, caps);
        assert_eq!(d6.len(), legal.len(), "D>=6 must use the full legal set");
        assert!(legal.iter().all(|mv| d6.contains(mv)));
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

    fn pair_width_first_stone_fixture() -> RustHexoState {
        replay(&[(0, 0), (0, 8), (2, 7), (1, 0), (4, 6), (6, 5), (8, 4)])
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
            .prove(&mut work, Player::Player0, 1, None)
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
            semantic_horizon: u32::MAX,
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
            semantic_horizon: u32::MAX,
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
                semantic_horizon: u32::MAX,
            },
        );
        assert_eq!(result.status, ProofStatus::Unknown);
        assert!(result.cert.is_none());
        assert_eq!(result.stats.nodes, 0);
    }

    #[test]
    fn semantic_horizon_is_an_absolute_placement_deadline() {
        let state = deep_universal_fixture();
        let root_ply = state.placements_made();
        let result = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 500_000,
                tt_bytes_cap: 0,
                semantic_horizon: root_ply,
            },
        );
        assert_eq!(result.status, ProofStatus::Unknown);
        assert!(result.cert.is_none());

        let expired = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 500_000,
                tt_bytes_cap: 0,
                semantic_horizon: root_ply - 1,
            },
        );
        assert_eq!(expired.status, ProofStatus::Unknown);
        assert_eq!(expired.stats.nodes, 0);
    }

    #[test]
    fn solver_configurations_are_deterministic_on_hard_leaf() {
        let state = forced_loss_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 4096,
            semantic_horizon: u32::MAX,
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
                semantic_horizon: u32::MAX,
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
    fn curated_deep_branch_zone_or_dispatch_is_reference_consistent() {
        let state = deep_universal_fixture();
        let base = deep_universal_certificate(&state);
        let (exact_t, _) = crate::tss_verify::certificate_horizon_preflight(&base).unwrap();

        fn collect_states(
            cert: &TssCertificate,
            id: CertNodeId,
            state: &RustHexoState,
            out: &mut [Option<RustHexoState>],
        ) {
            if out[id as usize].is_some() {
                return;
            }
            out[id as usize] = Some(state.clone());
            match &cert.nodes[id as usize] {
                CertNode::Choice { mv, child } => {
                    let mut next = state.clone();
                    apply_placement(&mut next, Placement { coord: *mv }).unwrap();
                    collect_states(cert, *child, &next, out);
                }
                CertNode::Universal { edges, .. } => {
                    for edge in edges {
                        let mut next = state.clone();
                        apply_placement(&mut next, Placement { coord: edge.mv }).unwrap();
                        collect_states(cert, edge.child, &next, out);
                    }
                }
                _ => {}
            }
        }
        let mut states = vec![None; base.nodes.len()];
        collect_states(&base, base.root_node, &state, &mut states);

        let mut accepted = None;
        for index in 0..base.nodes.len() {
            let Some(node_state) = states[index].as_ref() else {
                continue;
            };
            let CertNode::Universal {
                edges,
                implicit_dispatch: false,
                ..
            } = &base.nodes[index]
            else {
                continue;
            };
            let Some(d) =
                remaining_defender_placements_for_horizon(node_state, base.claimant, exact_t)
            else {
                continue;
            };
            for drop_index in 0..edges.len() {
                let mut nodes = base.nodes.clone();
                let CertNode::Universal { edges, zone, .. } = &mut nodes[index] else {
                    unreachable!()
                };
                *zone = Some(ZoneInfo {
                    d,
                    build_horizon: exact_t,
                });
                edges.remove(drop_index);
                let Some((nodes, root_node)) = compact_certificate(&nodes, base.root_node) else {
                    continue;
                };
                let candidate = TssCertificate {
                    root: base.root.clone(),
                    claimant: base.claimant,
                    root_node,
                    nodes,
                    semantic_horizon: exact_t,
                };
                if TssVerifier.verify(&state, &candidate, ProofStatus::Win) {
                    accepted = Some(candidate);
                    break;
                }
            }
            if accepted.is_some() {
                break;
            }
        }
        let cert = if let Some(cert) = accepted {
            cert
        } else {
            // This curated forcing line's final core can cover every explicit
            // reply. It still exercises the complete zone verifier; omission
            // behavior is covered by the generator-set and mutation tests.
            let mut nodes = base.nodes.clone();
            let Some((index, d)) = states.iter().enumerate().find_map(|(index, replay)| {
                let replay = replay.as_ref()?;
                matches!(
                    base.nodes[index],
                    CertNode::Universal {
                        implicit_dispatch: false,
                        ..
                    }
                )
                .then(|| {
                    remaining_defender_placements_for_horizon(replay, base.claimant, exact_t)
                        .map(|d| (index, d))
                })
                .flatten()
            }) else {
                // This line is dispatch-only; the U3 paired oracle is its
                // applicable certificate gate.
                return;
            };
            let CertNode::Universal { zone, .. } = &mut nodes[index] else {
                unreachable!()
            };
            *zone = Some(ZoneInfo {
                d,
                build_horizon: exact_t,
            });
            TssCertificate {
                root: base.root.clone(),
                claimant: base.claimant,
                root_node: base.root_node,
                nodes,
                semantic_horizon: exact_t,
            }
        };
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Win));
        assert_eq!(
            tss_reference::solve(&state, exact_t - state.placements_made()).status,
            ProofStatus::Win
        );
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
                semantic_horizon: u32::MAX,
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
                semantic_horizon: u32::MAX,
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
            semantic_horizon: u32::MAX,
        };
        let mut hard = 0usize;
        for state in &corpus {
            let result = TssSolver::default().solve(state, &caps);
            let Some(cert) = result.cert.as_ref() else {
                assert_eq!(result.status, ProofStatus::Unknown);
                continue;
            };
            hard += 1;
            assert!(TssVerifier.verify(state, cert, result.status));
            let b = threats::placements_remaining(state) as u32;
            let depth = if result.status == ProofStatus::Win {
                b
            } else {
                b + 2
            };
            assert_eq!(tss_reference::solve(state, depth).status, result.status);
        }
        assert!(
            hard >= 1,
            "dense anchor must produce at least one hard proof"
        );

        // Repeat the one-sided differential with U1 enabled at a matched
        // semantic horizon. Unknown remains an allowed restricted-search
        // outcome; every hard claim must verify and match the exhaustive
        // reference at the exact derived deadline.
        let mut zoned_hard = 0usize;
        for state in &corpus {
            let mut solver = TssSolver::default();
            solver.set_zone_options(ZoneSearchCaps {
                enabled: true,
                stale_area_filter: false,
                count2_threshold: false,
                pair_commutation: false,
            });
            let mut zone_caps = SolveCaps {
                node_cap: 16,
                tt_bytes_cap: 4096,
                semantic_horizon: state.placements_made().saturating_add(4),
            };
            let mut result = solver.solve(&state, &zone_caps);
            if let Some(cert) = result.cert.as_ref() {
                if let Some((exact_t, true)) =
                    crate::tss_verify::certificate_horizon_preflight(cert)
                {
                    if exact_t != zone_caps.semantic_horizon {
                        zone_caps.semantic_horizon = exact_t;
                        result = solver.solve(state, &zone_caps);
                    }
                }
            }
            let Some(cert) = result.cert.as_ref() else {
                assert_eq!(result.status, ProofStatus::Unknown);
                continue;
            };
            zoned_hard += 1;
            assert!(TssVerifier.verify(state, cert, result.status));
            let (exact_t, _) = crate::tss_verify::certificate_horizon_preflight(cert).unwrap();
            assert_eq!(
                tss_reference::solve(state, exact_t - state.placements_made()).status,
                result.status,
            );
        }
        assert!(zoned_hard >= 1);
    }

    #[test]
    fn dispatch_theorem_and_per_move_staple_oracles_agree() {
        let caps = SolveCaps {
            node_cap: 2_000,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let mut corpus = Vec::new();
        let mut random = 0x5a17_1e5u64;
        for _ in 0..16 {
            let mut state = RustHexoState::new();
            for ply in 0..18 {
                if ply >= 8 && ply % 3 == 0 {
                    corpus.push(state.clone());
                }
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                if legal.is_empty() {
                    break;
                }
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let coord = legal[(random as usize) % legal.len()];
                if apply_placement(&mut state, Placement { coord })
                    .unwrap()
                    .outcome
                    .is_some()
                {
                    break;
                }
            }
        }
        corpus.extend([
            forced_defense_fixture(),
            deep_universal_fixture(),
            spare_tempo_fixture(),
        ]);
        let mut checked = 0usize;
        for state in corpus {
            let result = TssSolver::default().solve(&state, &caps);
            let Some(cert) = result.cert.as_ref() else {
                continue;
            };
            let theorem = TssVerifier.verify(&state, cert, result.status);
            let per_move = TssVerifier.verify_with_dispatch_oracle(&state, cert, result.status);
            assert_eq!(theorem, per_move, "paired dispatch oracle divergence");
            assert!(theorem, "solver-produced hard certificate must verify");
            checked += 1;
        }
        assert!(checked > 0, "paired corpus must contain hard certificates");
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
                zone: None,
                commutations: Vec::new(),
            }],
            semantic_horizon: u32::MAX,
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
            semantic_horizon: u32::MAX,
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
            semantic_horizon: u32::MAX,
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
        for result in [&first, &repeated] {
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
        }
        // Cache history may change discovery cost, but never hard-verdict
        // validity.  A fixed initial cache state remains deterministic below.
        assert!(first.stats.peak_tt_bytes <= 4096);
        assert!(repeated.stats.peak_tt_bytes <= 4096);
    }

    #[test]
    fn shared_tt_warm_and_cold_verdicts_verify() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let cold = TssSolver::default().solve(&state, &caps);
        let mut warm_solver = TssSolver::default();
        let first = warm_solver.solve(&state, &caps);
        let warm = warm_solver.solve(&state, &caps);
        for result in [&cold, &first, &warm] {
            assert_eq!(result.status, ProofStatus::Win);
            assert!(TssVerifier.verify(&state, result.cert.as_ref().unwrap(), result.status));
            assert!(result.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        }
        assert!(warm.stats.tt_hits > 0);
        assert!(warm.stats.nodes < first.stats.nodes);
    }

    #[test]
    fn shared_tt_survives_origin_arena_drop() {
        let state = quiet_fixture();
        let key = PositionKey::from_state(&state);
        let mut cache = SharedProofCache::new(4096, u64::MAX);
        {
            let arena = vec![cache_test_leaf()];
            let proof = CachedProof::from_arena_limited(&arena, 0, 4, 4).unwrap();
            cache.insert(key.clone(), Player::Player0, proof);
        }
        let proof = cache.lookup_cloned(&key, Player::Player0).unwrap();
        let mut context = SearchContext::new(4, 0, u64::MAX);
        assert_eq!(context.import_cached_proof(proof, 0), Some(0));
        assert!(matches!(context.arena.as_slice(), [CertNode::Win { .. }]));
    }

    #[test]
    fn shared_tt_parent_proof_reuses_descendant() {
        let parent = deep_universal_fixture();
        let claimant = parent.current_player();
        let mut descendant_root = parent.clone();
        descendant_root
            .apply_with_delta(Placement {
                coord: HexCoord::new(2, -3),
            })
            .unwrap();
        assert_ne!(descendant_root.current_player(), claimant);

        let generous = SolveCaps {
            node_cap: 500_000,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let (local_cap, shared_cap) = split_tt_cap(generous.tt_bytes_cap);
        let mut solver = TssSolver::default();
        solver.shared_tt.reconfigure(shared_cap, u64::MAX);
        let reply = {
            let mut work = descendant_root.clone();
            let mut context = SearchContext::with_shared(
                generous.node_cap,
                local_cap,
                u64::MAX,
                &mut solver.shared_tt,
                descendant_root.placements_made(),
                generous.semantic_horizon,
                ZoneSearchCaps::default(),
                WidthOptions::default(),
                MAX_SEARCH_DEPTH,
            );
            let root = context
                .prove(&mut work, claimant, descendant_root.placements_made(), None)
                .unwrap();
            let CertNode::Universal { edges, .. } = &context.arena[root as usize] else {
                panic!("forcing descendant root must be universal");
            };
            edges
                .iter()
                .find(|edge| {
                    matches!(
                        context.arena[edge.child as usize],
                        CertNode::Choice { .. } | CertNode::Universal { .. }
                    )
                })
                .expect("fixture needs a cached non-leaf grandchild")
                .mv
        };

        // Re-root one level below the SearchContext root.  This key was
        // promoted while proving an internal universal edge, so an exact-root
        // result memo cannot satisfy the lookup.
        let mut descendant = descendant_root.clone();
        descendant
            .apply_with_delta(Placement { coord: reply })
            .unwrap();
        let descendant_key = PositionKey::from_state(&descendant);
        assert!(solver
            .shared_tt
            .lookup_cloned(&descendant_key, claimant)
            .is_some());

        let expected = status_for_claimant(descendant.current_player(), claimant);
        let goal = match expected {
            ProofStatus::Win => SolveGoal::Win,
            ProofStatus::Loss => SolveGoal::Loss,
            ProofStatus::Unknown => unreachable!(),
        };
        let tiny = SolveCaps {
            node_cap: 2,
            tt_bytes_cap: generous.tt_bytes_cap,
            semantic_horizon: u32::MAX,
        };
        let cold = TssSolver::default().solve_goal(&descendant, &tiny, goal);
        assert_eq!(cold.status, ProofStatus::Unknown);
        let result = solver.solve_goal(&descendant, &tiny, goal);
        assert_eq!(result.status, expected);
        assert!(result.stats.tt_hits > 0);
        assert!(TssVerifier.verify(&descendant, result.cert.as_ref().unwrap(), result.status));
    }

    #[test]
    fn shared_tt_forced_collision_a_b_a_stays_valid() {
        let a = deep_universal_fixture();
        let b = transformed_state(&a, 1);
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        let mut solver = TssSolver::with_hash_mask(0);
        let first_a = solver.solve(&a, &caps);
        let a_key = PositionKey::from_state(&a);
        let b_key = PositionKey::from_state(&b);
        let claimant = a.current_player();
        assert!(solver.shared_tt.lookup_cloned(&a_key, claimant).is_some());
        let first_b = solver.solve(&b, &caps);
        assert!(solver.shared_tt.lookup_cloned(&a_key, claimant).is_none());
        assert!(solver.shared_tt.lookup_cloned(&b_key, claimant).is_some());
        let second_a = solver.solve(&a, &caps);
        for (state, result) in [(&a, &first_a), (&b, &first_b), (&a, &second_a)] {
            assert_eq!(result.status, ProofStatus::Win);
            assert!(TssVerifier.verify(state, result.cert.as_ref().unwrap(), result.status));
        }
        assert!(first_a.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        assert!(first_b.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
        assert!(second_a.stats.peak_tt_bytes <= caps.tt_bytes_cap as u64);
    }

    #[test]
    fn shared_tt_claimant_isolation() {
        let state = quiet_fixture();
        let key = PositionKey::from_state(&state);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let mut cache = SharedProofCache::new(4096, 0);
        cache.insert(key.clone(), Player::Player0, proof);
        assert!(cache.lookup_cloned(&key, Player::Player0).is_some());
        assert!(cache.lookup_cloned(&key, Player::Player1).is_none());
    }

    #[test]
    fn width_profile_change_drops_shared_fragments_but_same_profile_keeps_them() {
        let state = quiet_fixture();
        let key = PositionKey::from_state(&state);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let mut solver = TssSolver::default();
        solver.shared_tt.reconfigure(4096, u64::MAX);
        solver.shared_tt.insert(key.clone(), Player::Player0, proof);
        assert!(solver.shared_tt.current_bytes > 0);

        solver.set_width_options(WidthOptions::default());
        assert!(solver.shared_tt.current_bytes > 0);
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        assert!(solver
            .shared_tt
            .lookup_cloned(&key, Player::Player0)
            .is_none());
        assert!(solver.shared_tt.slots.iter().all(Option::is_none));
        assert_eq!(
            solver.shared_tt.current_bytes,
            solver.shared_tt.recomputed_bytes()
        );
    }

    #[test]
    fn shared_tt_sustained_churn_respects_cap() {
        let mut cache = SharedProofCache::new(4096, u64::MAX);
        let proof = CachedProof::from_compact(vec![cache_test_leaf()], 0).unwrap();
        let base = deep_universal_fixture();
        for round in 0..200usize {
            let state = transformed_state(&base, (round % D6_SYMMETRY_COUNT as usize) as u8);
            cache.insert(
                PositionKey::from_state(&state),
                state.current_player(),
                proof.clone(),
            );
            assert_eq!(cache.current_bytes, cache.recomputed_bytes());
            assert!(cache.current_bytes <= cache.cap);
            assert!(cache.peak_bytes <= cache.cap);
        }
    }

    #[test]
    fn shared_tt_large_tiny_zero_reconfiguration() {
        let state = deep_universal_fixture();
        let mut solver = TssSolver::default();
        let large = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 8 << 20,
            semantic_horizon: u32::MAX,
        };
        assert_eq!(solver.solve(&state, &large).status, ProofStatus::Win);
        assert!(solver.shared_tt.current_bytes > 0);

        let tiny = SolveCaps {
            node_cap: 0,
            tt_bytes_cap: 1024,
            semantic_horizon: u32::MAX,
        };
        let tiny_result = solver.solve(&state, &tiny);
        assert_eq!(tiny_result.status, ProofStatus::Unknown);
        assert_eq!(solver.shared_tt.cap, split_tt_cap(tiny.tt_bytes_cap).1);
        assert!(solver.shared_tt.current_bytes <= solver.shared_tt.cap);
        assert!(tiny_result.stats.peak_tt_bytes <= tiny.tt_bytes_cap as u64);

        let zero = SolveCaps {
            node_cap: 0,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let zero_result = solver.solve(&state, &zero);
        assert_eq!(zero_result.status, ProofStatus::Unknown);
        assert!(solver.shared_tt.slots.is_empty());
        assert_eq!(solver.shared_tt.current_bytes, 0);
        assert_eq!(zero_result.stats.peak_tt_bytes, 0);
    }

    #[test]
    fn shared_tt_import_preflight_is_atomic() {
        let chain = CachedProof::from_compact(
            vec![
                cache_test_leaf(),
                CertNode::Choice {
                    mv: HexCoord::new(1, 0),
                    child: 0,
                },
                CertNode::Choice {
                    mv: HexCoord::new(2, 0),
                    child: 1,
                },
            ],
            2,
        )
        .unwrap();

        let mut depth_context = SearchContext::new(4, 0, u64::MAX);
        assert!(depth_context
            .import_cached_proof(chain.clone(), MAX_SEARCH_DEPTH - 1)
            .is_none());
        assert!(depth_context.arena.is_empty());

        let mut node_context = SearchContext::new(4, 0, u64::MAX);
        node_context.arena = vec![cache_test_leaf(); MAX_CERT_NODES - 2];
        let before_nodes = node_context.arena.len();
        assert!(node_context.import_cached_proof(chain, 0).is_none());
        assert_eq!(node_context.arena.len(), before_nodes);

        let edge_proof = CachedProof::from_compact(
            vec![
                cache_test_leaf(),
                CertNode::Universal {
                    edges: vec![CertEdge {
                        mv: HexCoord::new(1, 0),
                        child: 0,
                    }],
                    implicit_dispatch: false,
                    zone: None,
                    commutations: Vec::new(),
                },
            ],
            1,
        )
        .unwrap();
        let mut edge_context = SearchContext::new(4, 0, u64::MAX);
        edge_context.edge_count = MAX_CERT_EDGES;
        assert!(edge_context.import_cached_proof(edge_proof, 0).is_none());
        assert!(edge_context.arena.is_empty());
        assert_eq!(edge_context.edge_count, MAX_CERT_EDGES);
    }

    #[test]
    fn zone_cache_composition_refuses_slow_sibling_horizon() {
        let mut quick = cache_test_leaf();
        if let CertNode::Win { resolution_ply, .. } = &mut quick {
            *resolution_ply = 4;
        }
        let mut slow = cache_test_leaf();
        if let CertNode::Win { resolution_ply, .. } = &mut slow {
            *resolution_ply = 8;
        }
        // The quick branch was zoned at T=6. Flattening it beside a slower
        // T=8 sibling must retain min(zone-build)=6 and max(resolution)=8.
        let composite = CachedProof::from_compact(
            vec![
                quick,
                CertNode::Universal {
                    edges: vec![CertEdge {
                        mv: HexCoord::new(1, 0),
                        child: 0,
                    }],
                    implicit_dispatch: false,
                    zone: Some(ZoneInfo {
                        d: 1,
                        build_horizon: 6,
                    }),
                    commutations: Vec::new(),
                },
                slow,
                CertNode::Universal {
                    edges: vec![
                        CertEdge {
                            mv: HexCoord::new(2, 0),
                            child: 1,
                        },
                        CertEdge {
                            mv: HexCoord::new(3, 0),
                            child: 2,
                        },
                    ],
                    implicit_dispatch: false,
                    zone: None,
                    commutations: Vec::new(),
                },
            ],
            3,
        )
        .unwrap();
        assert_eq!(composite.resolution_t, 8);
        assert_eq!(composite.zone_build_t, Some(6));

        let mut rejected = SearchContext::new(32, 0, u64::MAX);
        rejected.semantic_horizon = 8;
        assert!(rejected.import_cached_proof(composite.clone(), 0).is_none());
        assert!(rejected.arena.is_empty(), "import preflight must be atomic");

        let mut accepted = SearchContext::new(32, 0, u64::MAX);
        accepted.semantic_horizon = 6;
        // Resolution 8 is independently too late, so a containing proof can
        // never smuggle this malformed composite through at either horizon.
        assert!(accepted.import_cached_proof(composite, 0).is_none());
    }

    #[test]
    fn shared_tt_never_caches_unknown() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 64 << 10,
            semantic_horizon: u32::MAX,
        };
        let mut solver = TssSolver::default();
        let result = solver.solve(&state, &caps);
        assert_eq!(result.status, ProofStatus::Unknown);
        let key = PositionKey::from_state(&state);
        for claimant in [state.current_player(), state.current_player().other()] {
            assert!(solver.shared_tt.lookup_cloned(&key, claimant).is_none());
        }
    }

    #[test]
    fn shared_tt_conditional_determinism() {
        let a = deep_universal_fixture();
        let b = transformed_state(&a, 2);
        let caps = SolveCaps {
            node_cap: 64,
            tt_bytes_cap: 64 << 10,
            semantic_horizon: u32::MAX,
        };
        let run = |solver: &mut TssSolver| {
            let first = solver.solve(&a, &caps);
            let second = solver.solve(&b, &caps);
            (
                first.status,
                first.cert,
                first.stats.nodes,
                first.stats.tt_hits,
                second.status,
                second.cert,
                second.stats.nodes,
                second.stats.tt_hits,
            )
        };
        let mut left = TssSolver::default();
        let mut right = TssSolver::default();
        assert_eq!(run(&mut left), run(&mut right));
        assert_eq!(left.shared_tt, right.shared_tt);
    }

    #[test]
    fn solve_goal_filters_root_facts() {
        let loss = forced_loss_fixture();
        let win = win_now_fixture();
        let caps = SolveCaps {
            node_cap: 1,
            tt_bytes_cap: 4096,
            semantic_horizon: u32::MAX,
        };
        let loss_filtered = TssSolver::default().solve_goal(&loss, &caps, SolveGoal::Win);
        assert_eq!(loss_filtered.status, ProofStatus::Unknown);
        assert!(loss_filtered.cert.is_none());
        let loss_kept = TssSolver::default().solve_goal(&loss, &caps, SolveGoal::Loss);
        assert_eq!(loss_kept.status, ProofStatus::Loss);
        assert!(TssVerifier.verify(&loss, loss_kept.cert.as_ref().unwrap(), loss_kept.status));
        assert_eq!(
            TssSolver::default()
                .solve_goal(&win, &caps, SolveGoal::Loss)
                .status,
            ProofStatus::Unknown
        );
    }

    #[test]
    fn solve_goal_one_sided_gets_full_budget() {
        let state = deep_universal_fixture();
        let caps = SolveCaps {
            node_cap: 3,
            tt_bytes_cap: 4096,
            semantic_horizon: u32::MAX,
        };
        let both = TssSolver::default().solve_goal(&state, &caps, SolveGoal::Both);
        let win = TssSolver::default().solve_goal(&state, &caps, SolveGoal::Win);
        assert_eq!(both.status, ProofStatus::Unknown);
        assert_eq!(win.status, ProofStatus::Win);
        assert!(TssVerifier.verify(&state, win.cert.as_ref().unwrap(), win.status));
    }

    #[test]
    fn quiet_no_hitting_universal_is_d6_cap_stable() {
        let state = quiet_fixture();
        let caps = SolveCaps {
            node_cap: 4,
            tt_bytes_cap: 0,
            semantic_horizon: u32::MAX,
        };
        let expected = TssSolver::default()
            .solve_goal(&state, &caps, SolveGoal::Loss)
            .status;
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let transformed = transformed_state(&state, symmetry);
            let status = TssSolver::default()
                .solve_goal(&transformed, &caps, SolveGoal::Loss)
                .status;
            assert_eq!(status, expected, "symmetry={symmetry}");
        }
    }

    #[test]
    fn make_unmake_round_trips_on_proof_and_cap_exit() {
        let original = deep_universal_fixture();
        for cap in [2, 500_000] {
            let mut work = original.clone();
            let before = work.clone();
            let mut context = SearchContext::new(cap, 64 << 10, u64::MAX);
            let _ = context.prove(&mut work, Player::Player0, 0, None);
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
                semantic_horizon: u32::MAX,
            },
        );
        let enough = TssSolver::default().solve(
            &state,
            &SolveCaps {
                node_cap: 5,
                tt_bytes_cap: 4096,
                semantic_horizon: u32::MAX,
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
