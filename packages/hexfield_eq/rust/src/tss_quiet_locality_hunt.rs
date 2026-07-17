//! NQ2 empirical hunt: lambda-2 connector locality.
//!
//! This module is entirely `#[cfg(test)]` and default-ignored. It does not
//! change any production search behavior. It mines verifier-accepted wide
//! certificates, isolates every QUIET attacker turn (a two-placement turn that
//! leaves the defender unforced), and measures where the quiet placements sit
//! relative to the threat structures they eventually serve.
//!
//! Quiet-turn definition used here (stated explicitly for the report): after a
//! nonterminal SecondStone placement, the turn is QUIET iff the engine's
//! `turn_forces_small_defender_reply` predicate is false.  A turn is forcing
//! iff the claimant wins under threat analysis, or (outside Opening) there is
//! at least one claimant >=4 threat, the defender cannot win now, and the
//! minimum hitting set equals the defender's remaining budget.  The weaker
//! "no claimant >=4 window" property is recorded separately as `strict_quiet`.
//! `round3_shadow_certificate` independently cross-checks the engine verdict.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState, Placement, Player, TurnPhase,
    WindowKey,
};

use crate::threats_shared as threats;
use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{round3_shadow_certificate, TssSolver, WidthOptions};
use crate::tss_verify::{
    d6_remap_certificate, d6_transform_coord, CertNode, CertNodeId, RootBinding, TssCertificate,
    TssVerifier, D6_SYMMETRY_COUNT, MAX_CERT_DEPTH,
};

// ------------------------------------------------------------------------
// Small deterministic RNG (sampling only; never on a solved path).
// ------------------------------------------------------------------------
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

fn free_ram_gb() -> f64 {
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB",
        ])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse::<f64>()
            .unwrap_or(64.0),
        Err(_) => 64.0,
    }
}

// ------------------------------------------------------------------------
// JSON helpers (hand-rolled: serde_json is not a dependency of this crate).
// ------------------------------------------------------------------------

/// Parse the first balanced `[...]` array following `"<key>":` as a flat list
/// of signed integers (works for `[[q,r],...]` too -- we just collect the ints
/// and chunk into pairs).
fn parse_pairs_after(line: &str, key: &str) -> Option<Vec<(i16, i16)>> {
    let needle = format!("\"{key}\":");
    let m = line.find(&needle)?;
    let after = &line[m + needle.len()..];
    let start = after.find('[')?;
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let arr = &after[start..=end?];
    let nums = parse_ints(arr);
    Some(nums.chunks_exact(2).map(|c| (c[0], c[1])).collect())
}

fn parse_ints(s: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.parse().expect("numeric"));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.parse().expect("numeric"));
    }
    out
}

fn parse_string_field(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let m = line.find(&needle)?;
    let after = &line[m + needle.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn parse_int_field(line: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\":");
    let m = line.find(&needle)?;
    let after = &line[m + needle.len()..];
    let mut s = String::new();
    for ch in after.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            s.push(ch);
        } else if !s.is_empty() {
            break;
        }
    }
    s.parse().ok()
}

// ------------------------------------------------------------------------
// Window-family algebra.
// ------------------------------------------------------------------------

#[derive(Clone)]
struct Family {
    cells: Vec<HexCoord>,
    max_count: u8,
}

/// Connected components of `windows` under cell-overlap (`WindowKey::intersects`).
/// Each component's cell-set is the union of all six-cell windows in it;
/// `max_count` is the largest per-window attacker count carried in.
fn build_families(windows: &[(WindowKey, u8)]) -> Vec<Family> {
    let n = windows.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != c {
            let nxt = parent[c];
            parent[c] = r;
            c = nxt;
        }
        r
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if windows[i].0.intersects(windows[j].0) {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut groups: HashMap<usize, (Vec<HexCoord>, u8)> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        let entry = groups.entry(root).or_insert_with(|| (Vec::new(), 0u8));
        entry.0.extend(windows[i].0.cells());
        entry.1 = entry.1.max(windows[i].1);
    }
    groups
        .into_values()
        .map(|(mut cells, max_count)| {
            cells.sort_by_key(|c| (c.q, c.r));
            cells.dedup();
            Family { cells, max_count }
        })
        .collect()
}

/// Live attacker windows at `state`: active (only-attacker) windows with
/// `count(attacker) >= 1`. `active_player == Some(claimant)` already implies
/// `count(defender) == 0`.
fn live_attacker_windows(state: &HexoState, claimant: Player) -> Vec<(WindowKey, u8)> {
    state
        .board()
        .windows()
        .entries()
        .filter(|e| e.active_player() == Some(claimant) && e.count(claimant) >= 1)
        .map(|e| (e.key(), e.count(claimant)))
        .collect()
}

fn attacker_stones(state: &HexoState, claimant: Player) -> Vec<HexCoord> {
    state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|&c| state.board().get(c) == Some(claimant))
        .collect()
}

/// Distance from `cell` to each family (min over the family's cells), sorted.
fn family_distances(cell: HexCoord, families: &[Family]) -> Vec<i32> {
    let mut d: Vec<i32> = families
        .iter()
        .map(|f| {
            f.cells
                .iter()
                .map(|&fc| i32::from(hex_distance(cell, fc)))
                .min()
                .unwrap_or(i32::MAX)
        })
        .collect();
    d.sort_unstable();
    d
}

/// Number of attacker windows with `count(defender) == 0` that contain `cell`,
/// bucketed by the attacker count AFTER placing at `cell` (index 0 => new
/// count 1, .. index 4 => new count 5). Also returns the max new count and the
/// count of pre-placement attacker-active windows through the cell.
fn window_incidence(
    state: &HexoState,
    claimant: Player,
    cell: HexCoord,
) -> ([usize; 5], u8, usize, usize) {
    let defender = claimant.other();
    let mut by_new_count = [0usize; 5];
    let mut max_new = 0u8;
    let mut pre_active = 0usize; // windows through cell already attacker-active
    let mut contested = 0usize; // windows through cell with a defender stone
    for entry in state.board().windows().entries() {
        if !entry.key().contains(cell) {
            continue;
        }
        if entry.count(defender) > 0 {
            contested += 1;
            continue;
        }
        // cnt_D == 0 window; placing here makes it (or keeps it) attacker-owned.
        let a = entry.count(claimant);
        let new = a + 1;
        if a >= 1 {
            pre_active += 1;
        }
        if (1..=5).contains(&new) {
            by_new_count[(new - 1) as usize] += 1;
        }
        max_new = max_new.max(new);
    }
    (by_new_count, max_new, pre_active, contested)
}

// ------------------------------------------------------------------------
// Subtree winning-window collection (structural, DAG-memoized).
// ------------------------------------------------------------------------

/// All window keys that constitute an eventual attacker win anywhere in the
/// subtree rooted at `id`: `OrCompletion.witness`, `Win.witness`, and
/// `Loss.witnesses`.
fn subtree_winning_windows(
    cert: &TssCertificate,
    id: CertNodeId,
    memo: &mut Vec<Option<Vec<WindowKey>>>,
    depth: usize,
) -> Vec<WindowKey> {
    if depth > MAX_CERT_DEPTH {
        return Vec::new();
    }
    if let Some(v) = memo.get(id as usize).and_then(|o| o.as_ref()) {
        return v.clone();
    }
    let mut acc: Vec<WindowKey> = Vec::new();
    match &cert.nodes[id as usize] {
        CertNode::OrCompletion { witness, .. } => acc.push(*witness),
        CertNode::Win { witness, .. } => acc.push(*witness),
        CertNode::Loss { witnesses, .. } => acc.extend(witnesses.iter().copied()),
        CertNode::Choice { child, .. } => {
            acc.extend(subtree_winning_windows(cert, *child, memo, depth + 1));
        }
        CertNode::Universal { edges, .. } => {
            for e in edges {
                acc.extend(subtree_winning_windows(cert, e.child, memo, depth + 1));
            }
        }
    }
    acc.sort_by_key(|w| (w.start.q, w.start.r, w.axis.index()));
    acc.dedup();
    if (id as usize) < memo.len() {
        memo[id as usize] = Some(acc.clone());
    }
    acc
}

// ------------------------------------------------------------------------
// Per-placement measurement.
// ------------------------------------------------------------------------

#[derive(Clone)]
struct Measures {
    placement: HexCoord,
    pre_ply: u32,
    phase: &'static str,
    stone_role: &'static str,
    d_stone: i32,
    // served (future) family distances
    n_served_families: usize,
    served_dists: Vec<i32>, // sorted
    d_used: i32,
    d_two: i32,
    // current-node structure
    node_full_legal: usize,
    incidence_by_new_count: [usize; 5],
    max_new_count: u8,
    pre_active_windows: usize,
    contested_windows: usize,
    // served-family connector incidence: how many served families have a live
    // attacker window through the cell
    served_families_through_cell: usize,
    // live-family connector incidence at the node
    live_families_through_cell: usize,
    subclass: &'static str,
    reduces_joint_completion: bool,
    // candidate C(P) universe sizes at this OR node + whether the actual cell hits
    cand: Vec<(&'static str, usize, bool)>,
}

/// Compute all measures for a placement at `cell` given the pre-placement
/// `state` (attacker to move), the served families (future completions), and
/// the served winning-window keys (to test served connector incidence).
fn measure_placement(
    state: &HexoState,
    claimant: Player,
    cell: HexCoord,
    stone_role: &'static str,
    served_families: &[Family],
    served_windows: &[WindowKey],
) -> Measures {
    let phase = match state.phase() {
        TurnPhase::FirstStone => "FirstStone",
        TurnPhase::SecondStone { .. } => "SecondStone",
        TurnPhase::Opening => "Opening",
    };

    // d_stone
    let stones = attacker_stones(state, claimant);
    let d_stone = stones
        .iter()
        .map(|&s| i32::from(hex_distance(cell, s)))
        .min()
        .map(|x| x as i32)
        .unwrap_or(-1);

    // served-family distances
    let served_dists = family_distances(cell, served_families);
    let n_served_families = served_families.len();
    let d_used = served_dists.first().copied().unwrap_or(-1);
    let d_two = served_dists.get(1).copied().unwrap_or(-1);

    // window incidence
    let (incidence_by_new_count, max_new_count, pre_active_windows, contested_windows) =
        window_incidence(state, claimant, cell);

    // live families at the node
    let live = live_attacker_windows(state, claimant);
    let live_families = build_families(&live);
    // how many live families does the cell touch (cell in one of the family's
    // live windows)? Compute using live windows through cell grouped by family.
    let live_families_through_cell = count_families_through_cell(cell, &live, &live_families);

    // served connector incidence: served windows through cell that are also
    // live-active now, grouped into served families.
    let served_windows_pairs: Vec<(WindowKey, u8)> =
        served_windows.iter().map(|&w| (w, 6u8)).collect();
    let served_families_through_cell =
        count_families_through_cell(cell, &served_windows_pairs, served_families);

    // reduces joint completion: the single placement advances >=2 distinct
    // live families at once (cell lies in a live window of >=2 families).
    let reduces_joint_completion = live_families_through_cell >= 2;

    // subclass
    let subclass = if served_families_through_cell >= 2 {
        "connector"
    } else if pre_active_windows == 0 {
        "remote_seed"
    } else if (2..=3).contains(&max_new_count) {
        "pair_build"
    } else {
        "other"
    };

    // node full legal + candidate universes
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    let node_full_legal = legal.len();
    let cand = candidate_universes(&legal, cell, &live, &live_families, &stones);

    Measures {
        placement: cell,
        pre_ply: state.placements_made(),
        phase,
        stone_role,
        d_stone,
        n_served_families,
        served_dists,
        d_used,
        d_two,
        node_full_legal,
        incidence_by_new_count,
        max_new_count,
        pre_active_windows,
        contested_windows,
        served_families_through_cell,
        live_families_through_cell,
        subclass,
        reduces_joint_completion,
        cand,
    }
}

/// Number of distinct families whose windows (from `windows`) contain `cell`.
fn count_families_through_cell(
    cell: HexCoord,
    windows: &[(WindowKey, u8)],
    families: &[Family],
) -> usize {
    // A cell "touches" a family if it is inside one of that family's windows.
    // We test membership via the family's cell-set: a window through the cell
    // contributes its cells to exactly one family, and the family's cell-set
    // contains the cell. So: count families whose cell-set contains the cell
    // AND at least one contributing window actually passes through the cell.
    let mut hit = 0usize;
    for f in families {
        if !f.cells.contains(&cell) {
            continue;
        }
        // Confirm a source window through the cell belongs to this family.
        let belongs = windows
            .iter()
            .any(|(w, _)| w.contains(cell) && w.cells().iter().all(|c| f.cells.contains(c)));
        if belongs {
            hit += 1;
        }
    }
    hit
}

/// Candidate certified-universe C(P) sizes at an OR node, computed purely from
/// the CURRENT position (live attacker families), plus whether `chosen` (the
/// actual quiet placement) is inside each candidate.
fn candidate_universes(
    legal: &[HexCoord],
    chosen: HexCoord,
    live_windows: &[(WindowKey, u8)],
    live_families: &[Family],
    attacker_stones: &[HexCoord],
) -> Vec<(&'static str, usize, bool)> {
    let in_live = |cell: HexCoord| -> bool { live_windows.iter().any(|(w, _)| w.contains(cell)) };
    let adj_stone = |cell: HexCoord, k: i32| -> bool {
        attacker_stones
            .iter()
            .any(|&s| i32::from(hex_distance(cell, s)) <= k)
    };
    // C(P) refinements: membership in a live attacker window AND within k of
    // an attacker stone. `join_adj1` is the report's conjectured universe;
    // `join_adj2` is the separately audited weaker adjacency tier.
    let join_adj1 = |cell: HexCoord| -> bool { in_live(cell) && adj_stone(cell, 1) };
    let join_adj2 = |cell: HexCoord| -> bool { in_live(cell) && adj_stone(cell, 2) };
    let fams_within = |cell: HexCoord, k: i32| -> usize {
        live_families
            .iter()
            .filter(|f| {
                f.cells
                    .iter()
                    .any(|&fc| i32::from(hex_distance(cell, fc)) <= k)
            })
            .count()
    };
    let pair_within = |cell: HexCoord, k: i32| -> bool {
        live_families.iter().any(|f| {
            f.max_count >= 2
                && f.cells
                    .iter()
                    .any(|&fc| i32::from(hex_distance(cell, fc)) <= k)
        })
    };

    // (name, predicate result for a cell) evaluated over the legal universe.
    let eval = |cell: HexCoord| -> [bool; 10] {
        [
            in_live(cell),
            fams_within(cell, 0) >= 2,
            fams_within(cell, 1) >= 2,
            fams_within(cell, 2) >= 2,
            pair_within(cell, 1),
            pair_within(cell, 2),
            adj_stone(cell, 1),
            adj_stone(cell, 2),
            join_adj2(cell),
            join_adj1(cell),
        ]
    };
    let names = [
        "join_live",
        "in2fam_k0",
        "near2fam_k1",
        "near2fam_k2",
        "nearpair_k1",
        "nearpair_k2",
        "adj_stone_k1",
        "adj_stone_k2",
        "join_adj2",
        "join_adj1",
    ];
    const NC: usize = 10;
    let mut sizes = [0usize; NC];
    for &c in legal {
        let e = eval(c);
        for i in 0..NC {
            if e[i] {
                sizes[i] += 1;
            }
        }
    }
    let hits = eval(chosen);
    (0..NC).map(|i| (names[i], sizes[i], hits[i])).collect()
}

// ------------------------------------------------------------------------
// Certificate walk: emit quiet placements.
// ------------------------------------------------------------------------

struct Walk<'a> {
    cert: &'a TssCertificate,
    claimant: Player,
    win_memo: Vec<Option<Vec<WindowKey>>>,
    out: Vec<QuietTurn>,
    quiet_turns: usize,
    visits: u64,
    budget_hit: bool,
}

/// Safety valve: a shared DAG expanded as a tree can be exponential. Certs from
/// the fast wide profile are small, but bound the traversal so no pathological
/// record can hang the sweep.
const WALK_VISIT_BUDGET: u64 = 40_000_000;

struct QuietTurn {
    turn_index: usize,
    forcing_threats_after: usize,
    strict_quiet: bool,
    served_family_count: usize,
    placements: Vec<Measures>,
}

/// Served families for the turn whose FirstStone node is `first_id` (structural,
/// identical for both stones of the turn).
fn served_for_turn(walk: &mut Walk, first_id: CertNodeId) -> (Vec<Family>, Vec<WindowKey>) {
    let wins = subtree_winning_windows(walk.cert, first_id, &mut walk.win_memo, 0);
    let pairs: Vec<(WindowKey, u8)> = wins.iter().map(|&w| (w, 6u8)).collect();
    (build_families(&pairs), wins)
}

fn count_forcing_threats(state: &HexoState, claimant: Player) -> usize {
    state
        .board()
        .windows()
        .entries()
        .filter(|e| e.active_player() == Some(claimant) && e.count(claimant) >= 4)
        .count()
}

/// Faithful replica of the engine's `turn_forces_small_defender_reply`
/// (private in `tss_solver`). `state` is the post-turn position with the
/// DEFENDER to move. Returns true when the attacker's turn forces a tight
/// defender reply (a win now, or an exactly-b hitting obligation). The shadow
/// census labels a completed SecondStone turn quiet exactly when this is false;
/// the consume fallback itself is not gated by this predicate.
fn engine_turn_forces_small_reply(state: &HexoState, claimant: Player) -> bool {
    let a = threats::analyze(state);
    let winner = if a.own_win_now {
        Some(state.current_player())
    } else if a.forced_loss() {
        Some(state.current_player().other())
    } else {
        None
    };
    winner == Some(claimant)
        || (!matches!(state.phase(), TurnPhase::Opening)
            && a.opp_threat_count > 0
            && !a.own_win_now
            && a.min_hitting_set == Some(a.b))
}

#[allow(clippy::too_many_arguments)]
fn walk_node(
    walk: &mut Walk,
    id: CertNodeId,
    state: &mut HexoState,
    depth: usize,
    // Pending FirstStone measures + served data, waiting on SecondStone verdict.
    pending: Option<(Measures, Vec<Family>, Vec<WindowKey>, CertNodeId)>,
) -> Option<()> {
    if depth > MAX_CERT_DEPTH {
        return None;
    }
    walk.visits += 1;
    if walk.visits > WALK_VISIT_BUDGET {
        walk.budget_hit = true;
        return Some(()); // stop expanding; keep whatever was recorded
    }
    let node = walk.cert.nodes.get(id as usize)?.clone();
    match node {
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {
            // Leaves. A pending FirstStone that reaches a leaf without a
            // SecondStone Choice is not a quiet two-stone turn; drop it.
            Some(())
        }
        CertNode::Choice { mv, child } => {
            let phase = state.phase();
            match phase {
                TurnPhase::FirstStone => {
                    // First stone of an attacker turn. Compute its measures now
                    // (served families are structural: subtree of THIS node).
                    let (served_families, served_windows) = served_for_turn(walk, id);
                    let m = measure_placement(
                        state,
                        walk.claimant,
                        mv,
                        "first",
                        &served_families,
                        &served_windows,
                    );
                    let (_res, delta) = state.apply_with_delta(Placement { coord: mv }).ok()?;
                    let r = walk_node(
                        walk,
                        child,
                        state,
                        depth + 1,
                        Some((m, served_families, served_windows, id)),
                    );
                    state.undo(delta);
                    r
                }
                TurnPhase::SecondStone { .. } => {
                    // Second stone completes the attacker turn. Determine quiet.
                    // Served families for THIS turn: if we have a pending first
                    // stone, reuse its (structural) served data; otherwise the
                    // turn's first stone is pre-root -- use this node's subtree.
                    let (served_families, served_windows) = match &pending {
                        Some((_, fams, wins, _)) => (fams.clone(), wins.clone()),
                        None => served_for_turn(walk, id),
                    };
                    let m2 = measure_placement(
                        state,
                        walk.claimant,
                        mv,
                        "second",
                        &served_families,
                        &served_windows,
                    );
                    let (res, delta) = state.apply_with_delta(Placement { coord: mv }).ok()?;
                    if res.outcome.is_some() {
                        // A winning placement must be an OrCompletion, not a
                        // Choice; mirror the engine shadow walk and abort.
                        state.undo(delta);
                        return None;
                    }
                    let forcing = count_forcing_threats(state, walk.claimant);
                    // Primary criterion = the engine's quiet_turn_or_edges gate.
                    let is_quiet = !engine_turn_forces_small_reply(state, walk.claimant);
                    if is_quiet {
                        walk.quiet_turns += 1;
                        let turn_index = walk.quiet_turns;
                        let mut placements = Vec::new();
                        let mut served_family_count = served_families.len();
                        if let Some((m1, fams, _, _)) = &pending {
                            served_family_count = fams.len();
                            placements.push(m1.clone());
                        }
                        placements.push(m2);
                        walk.out.push(QuietTurn {
                            turn_index,
                            forcing_threats_after: forcing,
                            strict_quiet: forcing == 0,
                            served_family_count,
                            placements,
                        });
                    }
                    let r = walk_node(walk, child, state, depth + 1, None);
                    state.undo(delta);
                    r
                }
                TurnPhase::Opening => {
                    let (_res, delta) = state.apply_with_delta(Placement { coord: mv }).ok()?;
                    let r = walk_node(walk, child, state, depth + 1, None);
                    state.undo(delta);
                    r
                }
            }
        }
        CertNode::Universal { edges, .. } => {
            for e in &edges {
                let (_res, delta) = state.apply_with_delta(Placement { coord: e.mv }).ok()?;
                let r = walk_node(walk, e.child, state, depth + 1, None);
                state.undo(delta);
                r?;
            }
            Some(())
        }
    }
}

fn mine_certificate<'a>(root: &HexoState, cert: &'a TssCertificate) -> Option<Walk<'a>> {
    let mut walk = Walk {
        cert,
        claimant: cert.claimant,
        win_memo: vec![None; cert.nodes.len()],
        out: Vec::new(),
        quiet_turns: 0,
        visits: 0,
        budget_hit: false,
    };
    let mut state = root.clone();
    walk_node(&mut walk, cert.root_node, &mut state, 0, None)?;
    Some(walk)
}

// ------------------------------------------------------------------------
// JSONL emission.
// ------------------------------------------------------------------------

fn ints_json(v: &[i32]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{x}");
    }
    s.push(']');
    s
}

fn usizes_json(v: &[usize]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{x}");
    }
    s.push(']');
    s
}

fn pairs_json(v: &[(i16, i16)]) -> String {
    let mut s = String::from("[");
    for (i, (q, r)) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "[{q},{r}]");
    }
    s.push(']');
    s
}

fn cand_json(cand: &[(&'static str, usize, bool)]) -> String {
    let mut s = String::from("{");
    for (i, (name, size, hit)) in cand.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "\"{name}\":{{\"size\":{size},\"hit\":{hit}}}");
    }
    s.push('}');
    s
}

#[allow(clippy::too_many_arguments)]
fn emit_records(
    sink: &mut dyn std::io::Write,
    source: &str,
    spec_id: &str,
    replay: &[(i16, i16)],
    claimant: Player,
    root_ply: u32,
    walk: &Walk,
) {
    for turn in &walk.out {
        for m in &turn.placements {
            let line = format!(
                "{{\"source\":\"{source}\",\"spec_id\":\"{spec_id}\",\"claimant\":{claimant},\"root_ply\":{root_ply},\"turn_index\":{turn_index},\"stone_role\":\"{role}\",\"placement\":[{pq},{pr}],\"pre_ply\":{pre_ply},\"phase\":\"{phase}\",\"forcing_threats_after_turn\":{forcing},\"strict_quiet\":{strict},\"d_used\":{d_used},\"d_two\":{d_two},\"d_stone\":{d_stone},\"n_served_families\":{nfam},\"served_dists\":{sdists},\"served_families_through_cell\":{sftc},\"live_families_through_cell\":{lftc},\"subclass\":\"{subclass}\",\"reduces_joint_completion\":{rjc},\"max_new_count\":{maxnew},\"pre_active_windows\":{preact},\"contested_windows\":{contested},\"incidence_by_new_count\":{incid},\"node_full_legal\":{legal},\"candidates\":{cand},\"replay\":{replay}}}",
                claimant = claimant.index(),
                turn_index = turn.turn_index,
                role = m.stone_role,
                pq = m.placement.q,
                pr = m.placement.r,
                pre_ply = m.pre_ply,
                phase = m.phase,
                forcing = turn.forcing_threats_after,
                strict = turn.strict_quiet,
                d_used = m.d_used,
                d_two = m.d_two,
                d_stone = m.d_stone,
                nfam = m.n_served_families,
                sdists = ints_json(&m.served_dists),
                sftc = m.served_families_through_cell,
                lftc = m.live_families_through_cell,
                subclass = m.subclass,
                rjc = m.reduces_joint_completion,
                maxnew = m.max_new_count,
                preact = m.pre_active_windows,
                contested = m.contested_windows,
                incid = usizes_json(&m.incidence_by_new_count),
                legal = m.node_full_legal,
                cand = cand_json(&m.cand),
                replay = pairs_json(replay),
            );
            let _ = writeln!(sink, "{line}");
        }
        let _ = turn.served_family_count;
    }
}

// ------------------------------------------------------------------------
// Replay helpers.
// ------------------------------------------------------------------------

/// Reconstruct the (q,r) move history that produced `state`, for a reproducible
/// `replay` field. For spare specimens the `spec_id` alone also regenerates the
/// position via `mining_candidate`.
fn state_replay(state: &HexoState) -> Vec<(i16, i16)> {
    state
        .placement_history()
        .iter()
        .map(|r| (r.coord.q, r.coord.r))
        .collect()
}

fn replay_moves(moves: &[(i16, i16)]) -> Option<HexoState> {
    let mut state = HexoState::new();
    for &(q, r) in moves {
        if state.is_terminal() {
            return None;
        }
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(q, r),
            },
        )
        .ok()?;
    }
    if state.is_terminal() {
        return None;
    }
    Some(state)
}

fn d6_moves(moves: &[(i16, i16)], sym: u8) -> Option<Vec<(i16, i16)>> {
    moves
        .iter()
        .map(|&(q, r)| d6_transform_coord(HexCoord::new(q, r), sym).map(|c| (c.q, c.r)))
        .collect()
}

// ------------------------------------------------------------------------
// Solve + report a single specimen.
// ------------------------------------------------------------------------

struct SolveOutcome {
    status: ProofStatus,
    nodes: u64,
    cert_nodes: usize,
    verified: bool,
    quiet_turns_mine: usize,
    quiet_turns_engine: usize,
    quiet_placements: usize,
}

/// Solve width profile. `vcf_pair_complete` is the fast established wide engine
/// (the profile the leaf-width records were mined with); after the ordinary
/// attacker frontier fails, `round3_consume` additionally enumerates the full
/// legal fallback without consulting the quiet predicate. It is ~2 orders of
/// magnitude heavier because branching can explode to the full frontier.
/// Locality measurement and the
/// `|legal|` universe count are position properties, identical either way, so
/// the corpus sweeps use the fast profile and consume is reserved for
/// specimens that genuinely need it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SolveMode {
    Vcf,
    Consume,
    /// Try the fast VCF profile first; if it does not return a certified WIN
    /// under the configured limits, fall back to consume. This heuristically
    /// targets candidate quiet-required positions without interpreting a VCF
    /// miss as proof that no pure-forcing win exists.
    TwoStage,
}

#[allow(clippy::too_many_arguments)]
fn solve_and_mine(
    sink: &mut dyn std::io::Write,
    source: &str,
    spec_id: &str,
    replay: &[(i16, i16)],
    state: &HexoState,
    cap: u64,
    tt_bytes: usize,
    horizon: u32,
    mode: SolveMode,
) -> SolveOutcome {
    let caps = SolveCaps {
        node_cap: cap,
        tt_bytes_cap: tt_bytes,
        semantic_horizon: horizon,
    };
    let result = match mode {
        SolveMode::Vcf => {
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::vcf_pair_complete());
            solver.solve(state, &caps)
        }
        SolveMode::Consume => {
            let mut solver = TssSolver::default();
            solver.set_width_options(WidthOptions::round3_consume());
            solver.solve(state, &caps)
        }
        SolveMode::TwoStage => {
            let mut vcf = TssSolver::default();
            vcf.set_width_options(WidthOptions::vcf_pair_complete());
            let vcf_result = vcf.solve(state, &caps);
            if vcf_result.status == ProofStatus::Win {
                vcf_result
            } else {
                let mut consume = TssSolver::default();
                consume.set_width_options(WidthOptions::round3_consume());
                consume.solve(state, &caps)
            }
        }
    };
    let mut outcome = SolveOutcome {
        status: result.status,
        nodes: result.stats.nodes,
        cert_nodes: result.cert.as_ref().map(|c| c.nodes.len()).unwrap_or(0),
        verified: false,
        quiet_turns_mine: 0,
        quiet_turns_engine: 0,
        quiet_placements: 0,
    };
    if let Some(cert) = &result.cert {
        outcome.verified = TssVerifier.verify(state, cert, result.status);
        if outcome.verified {
            if let Some(rep) = round3_shadow_certificate(state, cert) {
                outcome.quiet_turns_engine = rep.quiet_turns;
            }
            if let Some(walk) = mine_certificate(state, cert) {
                if walk.budget_hit {
                    eprintln!("QL_WARN spec_id={spec_id} walk_visit_budget_hit=true (partial)");
                }
                outcome.quiet_turns_mine = walk.out.len();
                outcome.quiet_placements = walk.out.iter().map(|t| t.placements.len()).sum();
                emit_records(
                    sink,
                    source,
                    spec_id,
                    replay,
                    cert.claimant,
                    state.placements_made(),
                    &walk,
                );
            }
        }
    }
    outcome
}

fn mode_from_env(default: SolveMode) -> SolveMode {
    match std::env::var("QL_MODE").ok().as_deref() {
        Some("consume") => SolveMode::Consume,
        Some("vcf") => SolveMode::Vcf,
        Some("twostage") => SolveMode::TwoStage,
        _ => default,
    }
}

fn open_sink(path_key: &str, default: &str) -> std::io::BufWriter<std::fs::File> {
    let path = std::env::var(path_key).unwrap_or_else(|_| default.to_string());
    let file = std::fs::File::create(&path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    eprintln!("QL_SINK key={path_key} path={path}");
    std::io::BufWriter::new(file)
}

// ------------------------------------------------------------------------
// TEST 1: canonical specimen + double-fork geometry variants.
// ------------------------------------------------------------------------

#[test]
#[ignore = "NQ2 quiet-locality: canonical + constructed double-fork family variants"]
fn quiet_locality_specimens() {
    let cap: u64 = std::env::var("QL_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let tt_bytes: usize = std::env::var("QL_TT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256 << 20);
    let horizon_slack: u32 = std::env::var("QL_HORIZON_SLACK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let ram = free_ram_gb();
    eprintln!("QL_SPECIMENS_SETUP cap={cap} tt_bytes={tt_bytes} slack={horizon_slack} free_ram_gb={ram:.2}");
    assert!(ram > 9.0, "insufficient free RAM: {ram:.2} GiB");

    let mut sink = open_sink(
        "QL_OUT_SPECIMENS",
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-quiet-locality/QL_SPECIMENS.jsonl",
    );

    // The spare-corpus family: hand-built positions engineered so the win
    // REQUIRES a non-forcing quiet connector / spare move (double fork, shared
    // target, deep junction, urgent-spare). These are the consume-solvable /
    // not-VCF-solvable population where quiet turns actually live -- unlike the
    // leaf-width records, which are VCF (all-forcing) wins with zero quiet
    // turns. Overridable via QL_SPEC_IDS (comma-separated).
    const DEFAULT_IDS: &[&str] = &[
        "double_fork_compact",
        "double_fork_spare",
        "double_fork_dense",
        "double_fork_dense_accelerated",
        "double_fork_ordered",
        "shared_target_spare",
        "shared_target_block4",
        "shared_target_block_endpoints",
        "deep_win_seed",
        "deep_universal",
        "deep_block18",
        "deep_block22",
        "deep_block26",
        "deep_triple_block",
        "deep_quad_block",
        "deep_urgent_spare",
        "compact_urgent_spare",
        "uncapped_junction",
        "urgent_uncapped_junction",
        "deep_pruned_latents",
        "human_6a5a",
        "human_6a5a_block_q",
        "human_6a5a_spare_edge",
        "human_2a94",
        "human_feaa",
        "human_5801",
        "spare_tempo_prefix",
    ];
    let ids: Vec<String> = match std::env::var("QL_SPEC_IDS") {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => DEFAULT_IDS.iter().map(|s| s.to_string()).collect(),
    };
    // Also include a couple of D6 images of the canonical fork to confirm the
    // measures are D6-covariant.
    let d6_anchor = std::env::var_os("QL_NO_D6").is_none();

    let mut work: Vec<(String, HexoState)> = Vec::new();
    for id in &ids {
        let state = crate::tss_spare_corpus::mining_candidate(id);
        work.push((id.clone(), state));
    }
    if d6_anchor {
        for sym in [1u8, 6u8] {
            let st = crate::tss_spare_corpus::mining_candidate("double_fork_compact");
            let moves = state_replay(&st);
            if let Some(m) = d6_moves(&moves, sym) {
                if let Some(s) = replay_moves(&m) {
                    work.push((format!("double_fork_compact_d6_{sym:02}"), s));
                }
            }
        }
    }

    for (id, state) in &work {
        let replay = state_replay(state);
        let horizon = state.placements_made().saturating_add(horizon_slack);
        let out = solve_and_mine(
            &mut sink,
            "specimen",
            id,
            &replay,
            state,
            cap,
            tt_bytes,
            horizon,
            mode_from_env(SolveMode::Consume),
        );
        eprintln!(
            "QL_SPEC id={id} status={:?} nodes={} cert_nodes={} verified={} quiet_mine={} quiet_engine={} quiet_placements={}",
            out.status,
            out.nodes,
            out.cert_nodes,
            out.verified,
            out.quiet_turns_mine,
            out.quiet_turns_engine,
            out.quiet_placements,
        );
    }
    sink.flush().ok();
}

// ------------------------------------------------------------------------
// TEST 2: leaf-width records.
// ------------------------------------------------------------------------

#[test]
#[ignore = "NQ2 quiet-locality: 122 leaf-width wide-only-win records"]
fn quiet_locality_leafwidth() {
    let cap: u64 = std::env::var("QL_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let tt_bytes: usize = std::env::var("QL_TT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256 << 20);
    let horizon_slack: u32 = std::env::var("QL_HORIZON_SLACK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let limit: usize = std::env::var("QL_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let path = std::env::var("QL_LEAFW_PATH").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-leaf-width/LEAF_WIDTH_RECORDS.jsonl"
            .to_string()
    });
    let ram = free_ram_gb();
    eprintln!("QL_LEAFW_SETUP cap={cap} tt_bytes={tt_bytes} slack={horizon_slack} limit={limit} free_ram_gb={ram:.2} path={path}");
    assert!(ram > 9.0, "insufficient free RAM: {ram:.2} GiB");

    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut sink = open_sink(
        "QL_OUT_LEAFW",
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-quiet-locality/QL_LEAFWIDTH.jsonl",
    );

    let mut n = 0usize;
    let mut solved_win = 0usize;
    let mut with_quiet = 0usize;
    let mut total_quiet_placements = 0usize;
    let mut replay_ok = 0usize;
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if n >= limit {
            break;
        }
        n += 1;
        let prefix = parse_pairs_after(line, "prefix").expect("prefix array");
        let game_hash =
            parse_string_field(line, "game_hash").unwrap_or_else(|| format!("rec{idx}"));
        let ply = parse_int_field(line, "ply").unwrap_or(-1);
        let spec_id = format!("{game_hash}:ply{ply}");
        let Some(state) = replay_moves(&prefix) else {
            eprintln!("QL_LEAFW_SKIP idx={idx} id={spec_id} reason=replay_terminal");
            continue;
        };
        replay_ok += 1;
        let horizon = state.placements_made().saturating_add(horizon_slack);
        let out = solve_and_mine(
            &mut sink,
            "leafwidth",
            &spec_id,
            &prefix,
            &state,
            cap,
            tt_bytes,
            horizon,
            mode_from_env(SolveMode::Vcf),
        );
        if out.status == ProofStatus::Win {
            solved_win += 1;
        }
        if out.quiet_turns_mine > 0 {
            with_quiet += 1;
        }
        total_quiet_placements += out.quiet_placements;
        eprintln!(
            "QL_LEAFW idx={idx} id={spec_id} status={:?} nodes={} cert_nodes={} verified={} quiet_mine={} quiet_engine={} quiet_placements={}",
            out.status, out.nodes, out.cert_nodes, out.verified, out.quiet_turns_mine, out.quiet_turns_engine, out.quiet_placements,
        );
    }
    sink.flush().ok();
    eprintln!(
        "QL_LEAFW_DONE records={n} replay_ok={replay_ok} solved_win={solved_win} with_quiet={with_quiet} total_quiet_placements={total_quiet_placements}"
    );
}

// ------------------------------------------------------------------------
// TEST 3: human corpus sampling.
// ------------------------------------------------------------------------

#[test]
#[ignore = "NQ2 quiet-locality: human-corpus attacker-node sample"]
fn quiet_locality_human() {
    let cap: u64 = std::env::var("QL_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let tt_bytes: usize = std::env::var("QL_TT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256 << 20);
    let horizon_slack: u32 = std::env::var("QL_HORIZON_SLACK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let sample: usize = std::env::var("QL_SAMPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let seed: u64 = std::env::var("QL_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xC0FFEE);
    let min_ply: u32 = std::env::var("QL_MIN_PLY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(14);
    let path = std::env::var("QL_HUMAN_PATH").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl".to_string()
    });
    let ram = free_ram_gb();
    eprintln!("QL_HUMAN_SETUP cap={cap} tt_bytes={tt_bytes} slack={horizon_slack} sample={sample} seed={seed} min_ply={min_ply} free_ram_gb={ram:.2} path={path}");
    assert!(ram > 9.0, "insufficient free RAM: {ram:.2} GiB");

    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    // Tail window: only sample FirstStone attacker nodes where the side to move
    // is the eventual WINNER and the node is within `tail` plies of the game's
    // end. Late winning-side nodes are where forced wins actually exist; a
    // random midgame node almost never has one, so consume would burn caps for
    // nothing. The winner is read from the TRUE engine terminal (robust to any
    // winner-int convention).
    let tail: u32 = std::env::var("QL_TAIL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    struct Node {
        moves: Vec<(i16, i16)>, // prefix
        game_hash: String,
        ply: u32,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut decisive_games = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let moves = match parse_pairs_after(line, "moves") {
            Some(m) => m,
            None => continue,
        };
        let game_hash = parse_string_field(line, "game_hash").unwrap_or_default();
        // Replay fully to find the true terminal winner and total plies.
        let mut end = HexoState::new();
        let mut applied = 0usize;
        for &(q, r) in &moves {
            if end.is_terminal() {
                break;
            }
            if apply_placement(
                &mut end,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .is_err()
            {
                break;
            }
            applied += 1;
        }
        let Some(outcome) = end.terminal() else {
            continue; // non-terminal / illegal game: skip
        };
        decisive_games += 1;
        let winner = outcome.winner;
        let total_plies = end.placements_made();
        // Re-walk to collect winning-side FirstStone nodes in the tail window.
        let mut state = HexoState::new();
        for (i, &(q, r)) in moves.iter().enumerate().take(applied) {
            if state.is_terminal() {
                break;
            }
            let ply = state.placements_made();
            if matches!(state.phase(), TurnPhase::FirstStone)
                && state.current_player() == winner
                && ply >= min_ply
                && ply + tail >= total_plies
                && i + 1 < moves.len()
            {
                nodes.push(Node {
                    moves: moves[..i].to_vec(),
                    game_hash: game_hash.clone(),
                    ply,
                });
            }
            if apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .is_err()
            {
                break;
            }
        }
    }
    eprintln!(
        "QL_HUMAN_POOL decisive_games={decisive_games} tail={tail} candidate_nodes={}",
        nodes.len()
    );
    let pool = nodes.len();
    let mut rng = XorShift(seed | 1);
    for i in (1..nodes.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        nodes.swap(i, j);
    }
    nodes.truncate(sample);

    let mut sink = open_sink(
        "QL_OUT_HUMAN",
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-quiet-locality/QL_HUMAN.jsonl",
    );

    let mut solved_win = 0usize;
    let mut with_quiet = 0usize;
    let mut total_quiet_placements = 0usize;
    for (k, node) in nodes.iter().enumerate() {
        let Some(state) = replay_moves(&node.moves) else {
            continue;
        };
        let spec_id = format!("{}:ply{}", node.game_hash, node.ply);
        let horizon = state.placements_made().saturating_add(horizon_slack);
        let out = solve_and_mine(
            &mut sink,
            "human",
            &spec_id,
            &node.moves,
            &state,
            cap,
            tt_bytes,
            horizon,
            mode_from_env(SolveMode::TwoStage),
        );
        if out.status == ProofStatus::Win {
            solved_win += 1;
        }
        if out.quiet_turns_mine > 0 {
            with_quiet += 1;
            total_quiet_placements += out.quiet_placements;
            eprintln!(
                "QL_HUMAN k={k} id={spec_id} status={:?} nodes={} verified={} quiet_mine={} quiet_engine={} quiet_placements={}",
                out.status, out.nodes, out.verified, out.quiet_turns_mine, out.quiet_turns_engine, out.quiet_placements,
            );
        }
    }
    sink.flush().ok();
    eprintln!(
        "QL_HUMAN_DONE pool={pool} sampled={} solved_win={solved_win} with_quiet={with_quiet} total_quiet_placements={total_quiet_placements}",
        nodes.len()
    );
}

// ------------------------------------------------------------------------
// TEST 4: adversarial remote defensive-tempo counterexamples.
// ------------------------------------------------------------------------

fn urgent_defender_empty_sets(state: &HexoState, attacker: Player) -> Vec<Vec<HexCoord>> {
    let defender = attacker.other();
    let mut threats = state
        .board()
        .windows()
        .entries()
        .filter(|entry| entry.active_player() == Some(defender) && entry.count(defender) >= 4)
        .map(|entry| {
            let mut empties = entry.empty_cells();
            empties.sort_by_key(|cell| (cell.q, cell.r));
            empties
        })
        .collect::<Vec<_>>();
    threats.sort_by_key(|empties| {
        empties
            .first()
            .map(|cell| (empties.len(), cell.q, cell.r))
            .unwrap_or((0, 0, 0))
    });
    threats
}

fn reply_kernel_parts(
    state: &HexoState,
    attacker: Player,
) -> (Vec<HexCoord>, Vec<HexCoord>, Vec<HexCoord>) {
    let threats = urgent_defender_empty_sets(state, attacker);
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);

    let mut win_now = Vec::new();
    for &cell in &legal {
        let mut child = state.clone();
        let result = apply_placement(&mut child, Placement { coord: cell })
            .expect("enumerated reply-kernel move is legal");
        if result
            .outcome
            .is_some_and(|outcome| outcome.winner == attacker)
        {
            win_now.push(cell);
        }
    }
    let block_all = legal
        .iter()
        .copied()
        .filter(|cell| threats.iter().all(|empties| empties.contains(cell)))
        .collect::<Vec<_>>();
    let mut kernel = win_now.clone();
    kernel.extend(block_all.iter().copied());
    kernel.sort_by_key(|cell| (cell.q, cell.r));
    kernel.dedup();
    (win_now, block_all, kernel)
}

fn assert_every_reply_kernel_omission_loses(state: &HexoState) {
    let attacker = state.current_player();
    assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }));
    let defender = attacker.other();
    let threats = urgent_defender_empty_sets(state, attacker);
    assert!(!threats.is_empty(), "fixture must be urgent");
    let (_, _, kernel) = reply_kernel_parts(state, attacker);
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    for cell in legal {
        if kernel.contains(&cell) {
            continue;
        }
        let missed = threats
            .iter()
            .find(|empties| !empties.contains(&cell))
            .expect("non-kernel non-win move misses an urgent window");
        let mut child = state.clone();
        let attack = apply_placement(&mut child, Placement { coord: cell })
            .expect("enumerated attacker completion");
        assert!(attack.outcome.is_none(), "Win1 move omitted from kernel");
        assert_eq!(child.current_player(), defender);
        assert!(matches!(child.phase(), TurnPhase::FirstStone));
        let mut won = false;
        for &reply in missed {
            let result = apply_placement(&mut child, Placement { coord: reply })
                .expect("urgent defender completion remains legal");
            if let Some(outcome) = result.outcome {
                assert_eq!(outcome.winner, defender);
                won = true;
                break;
            }
        }
        assert!(won, "defender did not complete the missed urgent window");
    }
}

#[test]
#[ignore = "NQ2 Q8: urgent reply-kernel adversarial phase matrix"]
fn quiet_locality_adversarial_reply_kernel() {
    // One count-four window with two distinct empties. Every omitted SecondStone
    // completion lets the defender fill both cells in its next two-stone turn.
    let count_four = [
        (0, 0),
        (0, 2),
        (1, 2),
        (-1, 2),
        (-3, -1),
        (2, 2),
        (3, 2),
        (1, -3),
        (4, -1),
        (-3, 3),
        (4, -3),
        (5, 1),
    ];
    let state = replay_moves(&count_four).expect("count-four Q8 replay");
    assert_eq!(
        state.phase(),
        TurnPhase::SecondStone {
            first: HexCoord::new(5, 1)
        }
    );
    let threats = urgent_defender_empty_sets(&state, Player::Player0);
    assert_eq!(
        threats,
        vec![vec![HexCoord::new(4, 2), HexCoord::new(5, 2)]]
    );
    let (win_now, block_all, kernel) = reply_kernel_parts(&state, Player::Player0);
    assert!(win_now.is_empty());
    assert_eq!(block_all, vec![HexCoord::new(4, 2), HexCoord::new(5, 2)]);
    assert_eq!(kernel, block_all);
    assert_every_reply_kernel_omission_loses(&state);

    // Four overlapping urgent windows (two count-five and two shifted
    // count-four windows) have one common empty, so their exact intersection is
    // a singleton.
    let overlap = [
        (0, 0),
        (-3, 2),
        (-2, 2),
        (-4, 2),
        (2, -4),
        (-1, 2),
        (0, 2),
        (-4, -1),
        (-3, -3),
        (1, 2),
        (2, -3),
        (4, -3),
        (5, -1),
        (2, -2),
        (2, -1),
        (5, 2),
        (4, 4),
        (2, 0),
        (2, 1),
        (-2, 5),
    ];
    let state = replay_moves(&overlap).expect("overlap Q8 replay");
    assert_eq!(
        state.phase(),
        TurnPhase::SecondStone {
            first: HexCoord::new(-2, 5)
        }
    );
    assert_eq!(urgent_defender_empty_sets(&state, Player::Player0).len(), 4);
    let (win_now, block_all, kernel) = reply_kernel_parts(&state, Player::Player0);
    assert!(win_now.is_empty());
    assert_eq!(block_all, vec![HexCoord::new(2, 2)]);
    assert_eq!(kernel, block_all);
    assert_every_reply_kernel_omission_loses(&state);

    // Disjoint defender threats have empty BlockAll, but the stored first stone
    // has created an attacker count-five. Its immediate completion must survive
    // through the Win1 arm of the union.
    let disjoint_win_now = [
        (0, 0),
        (0, 2),
        (0, 3),
        (1, 0),
        (2, 5),
        (0, 4),
        (-3, 1),
        (2, 0),
        (3, 5),
        (0, 5),
        (-3, 2),
        (3, 0),
        (4, 5),
        (0, 6),
        (-3, 3),
        (5, 5),
        (-4, -2),
        (-3, 4),
        (-3, 5),
        (4, 0),
    ];
    let state = replay_moves(&disjoint_win_now).expect("disjoint Win1 Q8 replay");
    assert_eq!(
        state.phase(),
        TurnPhase::SecondStone {
            first: HexCoord::new(4, 0)
        }
    );
    let (win_now, block_all, kernel) = reply_kernel_parts(&state, Player::Player0);
    assert!(urgent_defender_empty_sets(&state, Player::Player0).len() >= 2);
    assert!(block_all.is_empty());
    let completion = HexCoord::new(5, 0);
    assert!(win_now.contains(&completion));
    assert!(kernel.contains(&completion));
    assert_every_reply_kernel_omission_loses(&state);

    // Phase guard: the same predicate is not complete at FirstStone because the
    // attacker may spend its first placement outside K_reply and win with its
    // stored second placement before the defender receives a turn.
    let first_stone = [
        (0, 0),
        (0, 3),
        (1, 3),
        (1, 0),
        (2, 0),
        (2, 3),
        (3, 3),
        (3, 0),
        (-1, 3),
        (-1, 0),
        (5, -2),
    ];
    let mut state = replay_moves(&first_stone).expect("FirstStone guard replay");
    assert!(matches!(state.phase(), TurnPhase::FirstStone));
    let (_, _, raw_kernel) = reply_kernel_parts(&state, Player::Player0);
    let first = HexCoord::new(4, 0);
    assert!(!raw_kernel.contains(&first));
    let result =
        apply_placement(&mut state, Placement { coord: first }).expect("winning pair first");
    assert!(result.outcome.is_none());
    assert_eq!(state.phase(), TurnPhase::SecondStone { first });
    let result = apply_placement(
        &mut state,
        Placement {
            coord: HexCoord::new(5, 0),
        },
    )
    .expect("winning pair second");
    assert_eq!(
        result.outcome.map(|outcome| outcome.winner),
        Some(Player::Player0)
    );
}

#[test]
#[ignore = "NQ2 quiet-locality: overlap-family remote-seed counterexample"]
fn quiet_locality_adversarial_family_geometry() {
    // P0 stones (0,0),(1,0),(2,0) give an old horizontal live window.  The
    // remote P0 placement (5,3) is in no old P0-live window.  It nevertheless
    // creates a count-one vertical window through (5,0), which overlaps the
    // old horizontal window.  Thus every old window keeps its exact count and
    // every born window has completion distance five, but "only NEW families"
    // is false for the harness's overlap-component definition of Family.
    let replay = [(0, 0), (0, 8), (1, 7), (1, 0), (2, 0), (2, 7), (3, 7)];
    let mut state = replay_moves(&replay).expect("remote-family replay");
    let claimant = state.current_player();
    assert_eq!(claimant, Player::Player0);
    assert!(matches!(state.phase(), TurnPhase::FirstStone));

    let remote = HexCoord::new(5, 3);
    let before = live_attacker_windows(&state, claimant);
    assert!(!before.iter().any(|(window, _)| window.contains(remote)));

    let result = apply_placement(&mut state, Placement { coord: remote })
        .expect("remote placement is engine-legal");
    assert!(result.outcome.is_none());
    let after = live_attacker_windows(&state, claimant);

    for (window, old_count) in &before {
        let new_count = after
            .iter()
            .find_map(|(candidate, count)| (candidate == window).then_some(*count))
            .expect("old live window remains live");
        assert_eq!(new_count, *old_count, "remote changed an old live window");
    }

    let born = after
        .iter()
        .filter(|(window, count)| {
            *count == 1 && window.contains(remote) && !before.iter().any(|(old, _)| old == window)
        })
        .map(|(window, _)| *window)
        .collect::<Vec<_>>();
    assert_eq!(before.len(), 44, "old-live census drift");
    assert_eq!(born.len(), 16, "born delta-five census drift");
    assert!(born.contains(&WindowKey {
        start: HexCoord::new(5, 0),
        axis: Axis::R,
    }));
    assert!(born
        .iter()
        .any(|new_window| before.iter().any(|(old, _)| new_window.intersects(*old))));
    eprintln!(
        "QL_ADV_FAMILY replay={} remote=[{},{}] old_windows={} born_delta5={} overlap_merge=true",
        pairs_json(&replay),
        remote.q,
        remote.r,
        before.len(),
        born.len(),
    );
}

#[test]
#[ignore = "NQ2 quiet-locality: targeted required-remote quiet-win search"]
fn quiet_locality_adversarial_required_remote() {
    const FROZEN_ID: &str = "trapped_origin_diag_build6";
    let cap: u64 = std::env::var("QL_ADV_CAP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200_000);
    let tt_bytes: usize = std::env::var("QL_TT_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256 << 20);
    let horizon_slack: u32 = std::env::var("QL_ADV_HORIZON_SLACK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30);
    let ram = free_ram_gb();
    eprintln!(
        "QL_ADV_SETUP cap={cap} tt_bytes={tt_bytes} slack={horizon_slack} free_ram_gb={ram:.2}"
    );
    assert!(ram > 9.0, "insufficient free RAM: {ram:.2} GiB");

    #[derive(Clone)]
    struct AdversarialCase {
        id: &'static str,
        replacements: Vec<(usize, (i16, i16))>,
        remote: (i16, i16),
    }

    // Each mutation starts from the verifier-accepted double_fork_compact
    // replay.  The first four variants translate P1's five-in-a-row and use a
    // nonterminal P0 FirstStone build before the remaining stone must block a
    // remote completion.  The final variant is a separate compact row
    // mutation retaining the original first stone.
    let cases = vec![
        AdversarialCase {
            id: "trapped_origin_diag_build6",
            replacements: vec![
                (2, (1, -1)),
                (5, (2, -2)),
                (6, (3, -3)),
                (9, (4, -4)),
                (10, (5, -5)),
                (35, (6, 0)),
            ],
            remote: (6, -6),
        },
        AdversarialCase {
            id: "trapped_origin_diag_build31",
            replacements: vec![
                (2, (1, -1)),
                (5, (2, -2)),
                (6, (3, -3)),
                (9, (4, -4)),
                (10, (5, -5)),
                (35, (3, 1)),
            ],
            remote: (6, -6),
        },
        AdversarialCase {
            id: "trapped_origin_vertical_build6",
            replacements: vec![
                (2, (0, -1)),
                (5, (0, -2)),
                (6, (0, -3)),
                (9, (0, -4)),
                (10, (0, -5)),
                (35, (6, 0)),
            ],
            remote: (0, -6),
        },
        AdversarialCase {
            id: "split_far_row_build6",
            replacements: vec![
                (2, (-7, -2)),
                (5, (-8, -2)),
                (6, (-9, -2)),
                (9, (-11, -2)),
                (10, (-12, -2)),
                (35, (6, 0)),
            ],
            remote: (-10, -2),
        },
        AdversarialCase {
            id: "compact_row2",
            replacements: vec![(2, (0, 2)), (6, (1, 2)), (9, (2, 2)), (10, (3, 2))],
            remote: (5, 2),
        },
    ];

    let base = crate::tss_spare_corpus::mining_candidate("double_fork_compact");
    let base_replay = state_replay(&base);
    assert_eq!(base_replay.len(), 36);
    let expected = [
        (2usize, (4, 1)),
        (5, (4, 2)),
        (6, (4, 3)),
        (9, (4, 4)),
        (10, (4, 5)),
        (35, (-1, 2)),
    ];
    for (index, coord) in expected {
        assert_eq!(base_replay[index], coord, "base replay drift at {index}");
    }

    let mut structural = 0usize;
    let mut quiet = 0usize;
    let mut solver_unknown = 0usize;
    let mut solver_hard_nonloss = 0usize;
    let mut witness: Option<&'static str> = None;

    for case in &cases {
        let mut replay = base_replay.clone();
        for &(index, replacement) in &case.replacements {
            replay[index] = replacement;
        }
        let Some(root) = replay_moves(&replay) else {
            eprintln!("QL_ADV_CASE id={} replay_valid=false", case.id);
            continue;
        };
        let claimant = root.current_player();
        assert_eq!(claimant, Player::Player0);
        assert!(matches!(root.phase(), TurnPhase::SecondStone { .. }));
        let defender = claimant.other();
        let remote = HexCoord::new(case.remote.0, case.remote.1);

        let mut legal = Vec::new();
        root.write_legal_moves(&mut legal);
        assert!(legal.contains(&remote), "{} remote is not legal", case.id);
        let live = live_attacker_windows(&root, claimant);
        let live_families = build_families(&live);
        let stones = attacker_stones(&root, claimant);
        let d_stone = stones
            .iter()
            .map(|stone| i32::from(hex_distance(remote, *stone)))
            .min()
            .expect("claimant has stones");
        let candidates = candidate_universes(&legal, remote, &live, &live_families, &stones);
        let hit = |name: &str| {
            candidates
                .iter()
                .find_map(|(candidate, _, is_hit)| (*candidate == name).then_some(*is_hit))
                .expect("named candidate")
        };
        let candidate_size = |name: &str| {
            candidates
                .iter()
                .find_map(|(candidate, size, _)| (*candidate == name).then_some(*size))
                .expect("named candidate")
        };
        assert!(!hit("join_live"), "{} unexpectedly joins live", case.id);
        assert!(!hit("join_adj2"), "{} unexpectedly hits join_adj2", case.id);
        assert!(!hit("join_adj1"), "{} unexpectedly hits join_adj1", case.id);
        assert!(d_stone > 1);
        if case.id == FROZEN_ID {
            let frozen_replay = [
                (0, 0),
                (-1, 0),
                (1, -1),
                (1, 0),
                (2, 0),
                (2, -2),
                (3, -3),
                (3, 0),
                (4, 6),
                (4, -4),
                (5, -5),
                (1, 3),
                (2, 3),
                (2, 1),
                (5, 5),
                (3, 3),
                (0, 4),
                (6, 2),
                (-1, 5),
                (0, 5),
                (0, 6),
                (7, 6),
                (1, 6),
                (5, 7),
                (6, 7),
                (6, 6),
                (3, 6),
                (7, 7),
                (5, 6),
                (-1, 6),
                (1, 4),
                (6, 5),
                (7, 4),
                (7, 3),
                (7, 5),
                (6, 0),
            ];
            assert_eq!(replay.as_slice(), frozen_replay.as_slice());
            assert_eq!(root.placements_made(), 36);
            assert_eq!(
                root.phase(),
                TurnPhase::SecondStone {
                    first: HexCoord::new(6, 0)
                }
            );
            assert_eq!(remote, HexCoord::new(6, -6));
            assert_eq!(legal.len(), 538);
            assert_eq!(d_stone, 6);
            for (name, size) in [
                ("join_live", 141usize),
                ("join_adj2", 75),
                ("join_adj1", 38),
                ("adj_stone_k2", 93),
                ("adj_stone_k1", 39),
            ] {
                assert_eq!(candidate_size(name), size, "{name} census drift");
                assert!(!hit(name), "frozen remote entered {name}");
            }
            let threats = urgent_defender_empty_sets(&root, claimant);
            assert_eq!(
                threats,
                vec![vec![remote], vec![remote, HexCoord::new(7, -7)],]
            );
            let (win_now, block_all, kernel) = reply_kernel_parts(&root, claimant);
            assert!(win_now.is_empty());
            assert_eq!(block_all, vec![remote]);
            assert_eq!(kernel, vec![remote]);
        }
        structural += 1;

        // Exact finite elimination: every other legal completion is
        // nonterminal for A and leaves `remote` as a legal immediate six for
        // D.  This proves necessity without interpreting UNKNOWN as absence.
        let mut alternatives_eliminated = 0usize;
        let mut invalid_alternative = None;
        for &alternative in &legal {
            if alternative == remote {
                continue;
            }
            let mut child = root.clone();
            let attack_result = apply_placement(&mut child, Placement { coord: alternative })
                .expect("enumerated legal attacker completion");
            if attack_result.outcome.is_some() {
                invalid_alternative = Some((alternative, "attacker_completion"));
                break;
            }
            let defender_result = apply_placement(&mut child, Placement { coord: remote })
                .expect("remote remains a legal defender completion");
            if defender_result.outcome.map(|outcome| outcome.winner) != Some(defender) {
                invalid_alternative = Some((alternative, "no_immediate_defender_completion"));
                break;
            }
            alternatives_eliminated += 1;
        }
        if let Some((alternative, reason)) = invalid_alternative {
            eprintln!(
                "QL_ADV_CASE id={} structural=true exact_elimination=false reason={} alternative=[{},{}]",
                case.id, reason, alternative.q, alternative.r
            );
            continue;
        }
        assert_eq!(alternatives_eliminated + 1, legal.len());
        if case.id == FROZEN_ID {
            assert_eq!(alternatives_eliminated, 537);
        }

        let mut post = root.clone();
        let post_result = apply_placement(&mut post, Placement { coord: remote })
            .expect("forced remote attacker block");
        assert!(post_result.outcome.is_none());
        if engine_turn_forces_small_reply(&post, claimant) {
            eprintln!(
                "QL_ADV_CASE id={} structural=true quiet=false d_stone={} legal={} eliminated={}",
                case.id,
                d_stone,
                legal.len(),
                alternatives_eliminated
            );
            continue;
        }
        quiet += 1;

        let caps = SolveCaps {
            node_cap: cap,
            tt_bytes_cap: tt_bytes,
            semantic_horizon: root.placements_made().saturating_add(horizon_slack),
        };
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::round3_consume());
        let result = solver.solve_goal(&post, &caps, SolveGoal::Loss);
        eprintln!(
            "QL_ADV_CASE id={} structural=true quiet=true d_stone={} legal={} eliminated={} status={:?} nodes={} cert_nodes={}",
            case.id,
            d_stone,
            legal.len(),
            alternatives_eliminated,
            result.status,
            result.stats.nodes,
            result.cert.as_ref().map_or(0, |cert| cert.nodes.len()),
        );
        if case.id == FROZEN_ID {
            assert_eq!(
                result.status,
                ProofStatus::Loss,
                "frozen continuation drift"
            );
        }
        if result.status == ProofStatus::Unknown {
            solver_unknown += 1;
            continue;
        }
        if result.status != ProofStatus::Loss {
            solver_hard_nonloss += 1;
            continue;
        }

        let mut cert = result.cert.expect("hard result carries certificate");
        assert!(TssVerifier.verify(&post, &cert, ProofStatus::Loss));
        assert!(TssVerifier.verify_with_dispatch_oracle(&post, &cert, ProofStatus::Loss));
        assert_eq!(cert.claimant, claimant);
        let old_root = cert.root_node;
        let parent_root = u32::try_from(cert.nodes.len()).expect("certificate node id");
        cert.nodes.push(CertNode::Choice {
            mv: remote,
            child: old_root,
        });
        cert.root_node = parent_root;
        cert.root = RootBinding::from_state(&root);
        assert!(TssVerifier.verify(&root, &cert, ProofStatus::Win));
        assert!(TssVerifier.verify_with_dispatch_oracle(&root, &cert, ProofStatus::Win));
        if case.id == FROZEN_ID {
            assert_eq!(cert.semantic_horizon, 36u32.saturating_add(horizon_slack));

            // Close the claimed D6 family on this exact witness: replay, phase,
            // locality census, unique alternative elimination, quiet gate, and
            // the complete remapped certificate all survive every symmetry.
            for symmetry in 0..D6_SYMMETRY_COUNT {
                let transformed_replay = d6_moves(&replay, symmetry).expect("D6 replay mapping");
                let transformed_root =
                    replay_moves(&transformed_replay).expect("D6 witness replay");
                let transformed_remote =
                    d6_transform_coord(remote, symmetry).expect("D6 remote mapping");
                let transformed_first = d6_transform_coord(HexCoord::new(6, 0), symmetry)
                    .expect("D6 stored-first mapping");
                assert_eq!(
                    transformed_root.phase(),
                    TurnPhase::SecondStone {
                        first: transformed_first
                    }
                );
                let transformed_cert =
                    d6_remap_certificate(&cert, symmetry).expect("D6 certificate mapping");
                assert_eq!(
                    transformed_cert.root,
                    RootBinding::from_state(&transformed_root)
                );
                assert!(TssVerifier.verify(&transformed_root, &transformed_cert, ProofStatus::Win));

                let mut transformed_legal = Vec::new();
                transformed_root.write_legal_moves(&mut transformed_legal);
                assert_eq!(transformed_legal.len(), 538);
                assert!(transformed_legal.contains(&transformed_remote));
                let transformed_live = live_attacker_windows(&transformed_root, claimant);
                let transformed_families = build_families(&transformed_live);
                let transformed_stones = attacker_stones(&transformed_root, claimant);
                let transformed_distance = transformed_stones
                    .iter()
                    .map(|stone| i32::from(hex_distance(transformed_remote, *stone)))
                    .min()
                    .expect("transformed claimant stones");
                assert_eq!(transformed_distance, 6);
                let transformed_candidates = candidate_universes(
                    &transformed_legal,
                    transformed_remote,
                    &transformed_live,
                    &transformed_families,
                    &transformed_stones,
                );
                for (name, size) in [
                    ("join_live", 141usize),
                    ("join_adj2", 75),
                    ("join_adj1", 38),
                    ("adj_stone_k2", 93),
                    ("adj_stone_k1", 39),
                ] {
                    let (_, actual_size, actual_hit) = transformed_candidates
                        .iter()
                        .find(|(candidate, _, _)| *candidate == name)
                        .expect("transformed named candidate");
                    assert_eq!(*actual_size, size, "D6 {name} census drift");
                    assert!(!*actual_hit, "D6 remote entered {name}");
                }
                let (_, transformed_block_all, transformed_kernel) =
                    reply_kernel_parts(&transformed_root, claimant);
                assert_eq!(transformed_block_all, vec![transformed_remote]);
                assert_eq!(transformed_kernel, vec![transformed_remote]);

                for &alternative in &transformed_legal {
                    if alternative == transformed_remote {
                        continue;
                    }
                    let mut child = transformed_root.clone();
                    let attack = apply_placement(&mut child, Placement { coord: alternative })
                        .expect("D6 alternative legal");
                    assert!(attack.outcome.is_none());
                    let defense = apply_placement(
                        &mut child,
                        Placement {
                            coord: transformed_remote,
                        },
                    )
                    .expect("D6 remote defense legal");
                    assert_eq!(
                        defense.outcome.map(|outcome| outcome.winner),
                        Some(defender)
                    );
                }
                let mut transformed_post = transformed_root.clone();
                let result = apply_placement(
                    &mut transformed_post,
                    Placement {
                        coord: transformed_remote,
                    },
                )
                .expect("D6 remote attacker block");
                assert!(result.outcome.is_none());
                assert!(!engine_turn_forces_small_reply(&transformed_post, claimant));
            }
        }

        eprintln!(
            "QL_ADV_WITNESS schema=1 id={} claimant={} phase=SecondStone remote=[{},{}] d_stone={} quiet=true legal={} eliminated={} candidates={} horizon={} cert_nodes={} replay={}",
            case.id,
            claimant.index(),
            remote.q,
            remote.r,
            d_stone,
            legal.len(),
            alternatives_eliminated,
            cand_json(&candidates),
            cert.semantic_horizon,
            cert.nodes.len(),
            pairs_json(&replay),
        );
        witness = Some(case.id);
        break;
    }

    eprintln!(
        "QL_ADV_DONE generated={} structural={} quiet={} solver_unknown={} solver_hard_nonloss={} witnesses={}",
        cases.len(),
        structural,
        quiet,
        solver_unknown,
        solver_hard_nonloss,
        usize::from(witness.is_some()),
    );
    assert_eq!(witness, Some(FROZEN_ID), "frozen witness did not verify");
}
