//! R-CF1 deep-census diagnostics and shadow support.
//!
//! Everything in this module is test-only. The counters are inert unless
//! `TSS_CENSUS_DEEP_COUNTERS=1` exactly; the later candidate audit is inert
//! unless `TSS_CENSUS_DEEP_SHADOW=1` exactly. Census values always come from
//! a complete `WindowStore::entries()` scan, never from the threat index.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use hexo_engine::{HexCoord, HexoState, Placement, Player, TurnPhase};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Candidate {
    StageDtw,
    DefenderRestore4,
    DeadlineEs,
    DeadlineEsPreblock,
    DeadlineEsTriple,
    PairServiceC1,
    PairServiceC2,
    PairServiceC3,
    DefenderReplyLift,
    TwoCycleLift,
    CensusAttractor,
}

pub(crate) const CANDIDATE_COUNT: usize = 11;
pub(crate) const CANDIDATES: [Candidate; CANDIDATE_COUNT] = [
    Candidate::StageDtw,
    Candidate::DefenderRestore4,
    Candidate::DeadlineEs,
    Candidate::DeadlineEsPreblock,
    Candidate::DeadlineEsTriple,
    Candidate::PairServiceC1,
    Candidate::PairServiceC2,
    Candidate::PairServiceC3,
    Candidate::DefenderReplyLift,
    Candidate::TwoCycleLift,
    Candidate::CensusAttractor,
];

impl Candidate {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::StageDtw => 0,
            Self::DefenderRestore4 => 1,
            Self::DeadlineEs => 2,
            Self::DeadlineEsPreblock => 3,
            Self::DeadlineEsTriple => 4,
            Self::PairServiceC1 => 5,
            Self::PairServiceC2 => 6,
            Self::PairServiceC3 => 7,
            Self::DefenderReplyLift => 8,
            Self::TwoCycleLift => 9,
            Self::CensusAttractor => 10,
        }
    }

    pub(crate) fn is_bounded(self) -> bool {
        matches!(
            self,
            Self::StageDtw
                | Self::DefenderRestore4
                | Self::DeadlineEs
                | Self::DeadlineEsPreblock
                | Self::DeadlineEsTriple
        )
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::StageDtw => "STAGE_DTW",
            Self::DefenderRestore4 => "DEFENDER_RESTORE4",
            Self::DeadlineEs => "DEADLINE_ES",
            Self::DeadlineEsPreblock => "DEADLINE_ES_PREBLOCK",
            Self::DeadlineEsTriple => "DEADLINE_ES_TRIPLE",
            Self::PairServiceC1 => "PAIR_SERVICE_C1",
            Self::PairServiceC2 => "PAIR_SERVICE_C2",
            Self::PairServiceC3 => "PAIR_SERVICE_C3",
            Self::DefenderReplyLift => "DEFENDER_REPLY_LIFT",
            Self::TwoCycleLift => "TWO_CYCLE_LIFT",
            Self::CensusAttractor => "CENSUS_ATTRACTOR",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CandidateAudit {
    pub(crate) candidate: Option<Candidate>,
    pub(crate) fires: u64,
    pub(crate) would_prunes: u64,
    pub(crate) expansion_mass: u64,
    pub(crate) counterexamples: u64,
    pub(crate) search_refuted: u64,
    pub(crate) search_unknown: u64,
    pub(crate) late_wins: u64,
    pub(crate) unresolved_wins: u64,
    pub(crate) evaluations: u64,
    pub(crate) evaluation_nanos: u64,
    pub(crate) work_units: u64,
    pub(crate) capped_evaluations: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PairServiceProfile {
    pub(crate) census: u8,
    pub(crate) live_count2: u32,
    pub(crate) live_count3: u32,
    pub(crate) upgrade_cells: u32,
    pub(crate) families_checked: u64,
    pub(crate) all_families_hit_one: bool,
}

#[derive(Clone, Debug)]
struct UpgradeWindow {
    strength: u8,
    empties: Vec<HexCoord>,
}

fn family_needs_two_hits(
    windows: &[UpgradeWindow],
    first: HexCoord,
    second: Option<HexCoord>,
) -> bool {
    let mut common: Option<Vec<hexo_engine::HexCoord>> = None;
    for window in windows {
        let added = u8::from(window.empties.contains(&first))
            + u8::from(second.is_some_and(|cell| window.empties.contains(&cell)));
        if window.strength.saturating_add(added) < 4 {
            continue;
        }
        let residual = window
            .empties
            .iter()
            .copied()
            .filter(|&cell| cell != first && second != Some(cell))
            .collect::<Vec<_>>();
        if residual.is_empty() {
            return true;
        }
        match &mut common {
            None => common = Some(residual),
            Some(cells) => {
                cells.retain(|cell| residual.contains(cell));
                if cells.is_empty() {
                    return true;
                }
            }
        }
    }
    false
}

/// Exact one-turn census used by the R-CF1 shadow candidates.
///
/// At a post-tactical claimant `FirstStone` node the wide TSS grammar admits
/// a pair only if its complete post-pair threat family needs at least two
/// defender hits. Every such threat comes from a currently live count-2 or
/// count-3 window. We enumerate the union of their exact empty cells and ask
/// whether every singleton/pair-created family retains a common hit. A
/// placement outside this union cannot alter or create one of these families,
/// so singleton evaluation covers it as a filler. Defender threats can only
/// reject additional pairs and are deliberately omitted from this sufficient
/// deadness test.
pub(crate) fn pair_service_profile(state: &HexoState, claimant: Player) -> PairServiceProfile {
    let mut census = 0u8;
    let mut windows = Vec::new();
    let mut cells = Vec::new();
    let mut live_count2 = 0u32;
    let mut live_count3 = 0u32;
    for entry in state.board().windows().entries() {
        let cc = entry.count(claimant);
        let oc = entry.count(claimant.other());
        if cc == 0 || oc != 0 {
            continue;
        }
        census = census.max(cc);
        if !(2..=3).contains(&cc) {
            continue;
        }
        live_count2 = live_count2.saturating_add(u32::from(cc == 2));
        live_count3 = live_count3.saturating_add(u32::from(cc == 3));
        let empties = entry.empty_cells();
        cells.extend(empties.iter().copied());
        windows.push(UpgradeWindow {
            strength: cc,
            empties,
        });
    }
    cells.sort_unstable_by_key(|cell| (cell.q, cell.r));
    cells.dedup();
    let mut out = PairServiceProfile {
        census,
        live_count2,
        live_count3,
        upgrade_cells: u32::try_from(cells.len()).unwrap_or(u32::MAX),
        ..PairServiceProfile::default()
    };
    if census > 3 {
        return out;
    }
    for (left, &first) in cells.iter().enumerate() {
        out.families_checked = out.families_checked.saturating_add(1);
        if family_needs_two_hits(&windows, first, None) {
            return out;
        }
        for &second in &cells[(left + 1)..] {
            out.families_checked = out.families_checked.saturating_add(1);
            if family_needs_two_hits(&windows, first, Some(second)) {
                return out;
            }
        }
    }
    out.all_families_hit_one = true;
    out
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DefenderRestoreProfile {
    pub(crate) restored: bool,
    pub(crate) sequences_checked: u64,
    pub(crate) scan_nanos: u64,
}

fn exact_claimant_census(state: &HexoState, claimant: Player) -> u8 {
    state
        .board()
        .windows()
        .entries()
        .filter(|entry| entry.count(claimant.other()) == 0)
        .map(|entry| entry.count(claimant))
        .max()
        .unwrap_or(0)
}

fn restored_successor(state: &HexoState, claimant: Player) -> bool {
    state.current_player() == claimant
        && matches!(state.phase(), TurnPhase::FirstStone)
        && exact_claimant_census(state, claimant) <= 3
}

/// Exhibit an exact, sequentially legal remainder of the defender turn whose
/// successor restores the claimant's full-store census to at most three.
/// The caller separately checks the exact `tau=b` threat-analysis premise.
pub(crate) fn defender_restore4_profile(
    state: &HexoState,
    claimant: Player,
) -> DefenderRestoreProfile {
    let started = Instant::now();
    if state.current_player() == claimant || matches!(state.phase(), TurnPhase::Opening) {
        return DefenderRestoreProfile::default();
    }
    let mut support = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.count(claimant) < 4 || entry.count(claimant.other()) != 0 {
            continue;
        }
        support.extend(entry.empty_cells());
    }
    support.sort_unstable_by_key(|cell| (cell.q, cell.r));
    support.dedup();
    let mut out = DefenderRestoreProfile::default();
    match state.phase() {
        TurnPhase::SecondStone { .. } => {
            for cell in support {
                out.sequences_checked = out.sequences_checked.saturating_add(1);
                let mut work = state.clone();
                let Ok((result, _delta)) = work.apply_with_delta(Placement { coord: cell }) else {
                    continue;
                };
                if result
                    .outcome
                    .is_some_and(|outcome| outcome.winner != claimant)
                    || result.outcome.is_none() && restored_successor(&work, claimant)
                {
                    out.restored = true;
                    break;
                }
            }
        }
        TurnPhase::FirstStone => {
            'pairs: for left in 0..support.len() {
                for right in (left + 1)..support.len() {
                    for (first, second) in [
                        (support[left], support[right]),
                        (support[right], support[left]),
                    ] {
                        out.sequences_checked = out.sequences_checked.saturating_add(1);
                        let mut work = state.clone();
                        let Ok((first_result, _first_delta)) =
                            work.apply_with_delta(Placement { coord: first })
                        else {
                            continue;
                        };
                        if let Some(outcome) = first_result.outcome {
                            if outcome.winner != claimant {
                                out.restored = true;
                                break 'pairs;
                            }
                            continue;
                        }
                        let Ok((second_result, _second_delta)) =
                            work.apply_with_delta(Placement { coord: second })
                        else {
                            continue;
                        };
                        if second_result
                            .outcome
                            .is_some_and(|outcome| outcome.winner != claimant)
                            || second_result.outcome.is_none()
                                && restored_successor(&work, claimant)
                        {
                            out.restored = true;
                            break 'pairs;
                        }
                    }
                }
            }
        }
        TurnPhase::Opening => {}
    }
    out.scan_nanos = elapsed_nanos(started);
    out
}

#[derive(Clone, Debug)]
struct DeadlineWindow {
    count: u8,
    empties: Vec<HexCoord>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DeadlineProfile {
    pub(crate) applicable: bool,
    pub(crate) claimant_slots: u8,
    pub(crate) family_windows: u32,
    pub(crate) dtes: bool,
    pub(crate) dtes_preblock: bool,
    pub(crate) dtes_triple: bool,
    pub(crate) scan_nanos: u64,
    pub(crate) preblock_nanos: u64,
    pub(crate) triple_nanos: u64,
    pub(crate) preblock_steps: u64,
    pub(crate) triple_states: u64,
    pub(crate) triple_capped: bool,
}

fn elapsed_nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn claimant_slots_before_deadline(
    state: &HexoState,
    claimant: Player,
    remaining: usize,
) -> Option<u8> {
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let mut player = state.current_player();
    let mut second = matches!(state.phase(), TurnPhase::SecondStone { .. });
    let mut slots = 0u8;
    for _ in 0..remaining {
        slots = slots.saturating_add(u8::from(player == claimant));
        if second {
            player = player.other();
            second = false;
        } else {
            second = true;
        }
    }
    Some(slots)
}

fn potential_bins(windows: &[DeadlineWindow]) -> [u64; 6] {
    let mut bins = [0u64; 6];
    for window in windows {
        if let Some(bin) = bins.get_mut(usize::from(window.count.min(5))) {
            *bin = bin.saturating_add(1);
        }
    }
    bins
}

/// Exact integer comparison for
/// `sum_W (sqrt(3))^(-(6-count(W))) < thirds / 3`.
fn potential_lt_thirds(bins: &[u64; 6], thirds: u8) -> bool {
    let a = u128::from(bins[1])
        .saturating_add(u128::from(bins[3]).saturating_mul(3))
        .saturating_add(u128::from(bins[5]).saturating_mul(9));
    let b = u128::from(bins[2]).saturating_add(u128::from(bins[4]).saturating_mul(3));
    let target = u128::from(thirds).saturating_mul(3);
    if b >= target {
        return false;
    }
    let gap = target - b;
    a.saturating_mul(a) < 3u128.saturating_mul(gap).saturating_mul(gap)
}

fn deadline_threshold(state: &HexoState, claimant: Player) -> u8 {
    match (state.current_player() == claimant, state.phase()) {
        (false, TurnPhase::FirstStone) => 3,
        (false, TurnPhase::SecondStone { .. }) => 2,
        (true, TurnPhase::FirstStone | TurnPhase::SecondStone { .. }) => 1,
        (_, TurnPhase::Opening) => 0,
    }
}

fn preblock_extension(
    state: &HexoState,
    claimant: Player,
    windows: &[DeadlineWindow],
) -> (bool, u64) {
    if state.current_player() == claimant || matches!(state.phase(), TurnPhase::Opening) {
        return (false, 0);
    }
    let steps = match state.phase() {
        TurnPhase::FirstStone => 2,
        TurnPhase::SecondStone { .. } => 1,
        TurnPhase::Opening => 0,
    };
    let mut work = state.clone();
    let mut residual = windows.to_vec();
    let mut applied = 0u64;
    for _ in 0..steps {
        if residual.is_empty() {
            return (true, applied);
        }
        let mut scores: Vec<(HexCoord, u64)> = Vec::new();
        for window in &residual {
            for &cell in &window.empties {
                if let Some((_, score)) = scores.iter_mut().find(|(seen, _)| *seen == cell) {
                    *score = score.saturating_add(1);
                } else {
                    scores.push((cell, 1));
                }
            }
        }
        scores.sort_unstable_by_key(|(cell, score)| (std::cmp::Reverse(*score), cell.q, cell.r));
        let mut selected = None;
        for (cell, _) in scores {
            if let Ok((result, _delta)) = work.apply_with_delta(Placement { coord: cell }) {
                applied = applied.saturating_add(1);
                if let Some(outcome) = result.outcome {
                    return (outcome.winner != claimant, applied);
                }
                selected = Some(cell);
                break;
            }
        }
        let Some(cell) = selected else {
            return (false, applied);
        };
        residual.retain(|window| !window.empties.contains(&cell));
    }
    (potential_lt_thirds(&potential_bins(&residual), 1), applied)
}

fn triples_for(empties: &[HexCoord]) -> Vec<[HexCoord; 3]> {
    let mut out = Vec::new();
    for first in 0..empties.len() {
        for second in (first + 1)..empties.len() {
            for third in (second + 1)..empties.len() {
                let mut triple = [empties[first], empties[second], empties[third]];
                triple.sort_unstable_by_key(|cell| (cell.q, cell.r));
                if !out.contains(&triple) {
                    out.push(triple);
                }
            }
        }
    }
    out
}

fn triple_compatible(triple: &[HexCoord; 3], used: &[HexCoord]) -> bool {
    triple.iter().all(|cell| !used.contains(cell))
}

fn triple_cover_search(
    windows: &[DeadlineWindow],
    candidates: &[Vec<[HexCoord; 3]>],
    covered: &mut [bool],
    used: &mut Vec<HexCoord>,
    states: &mut u64,
    capped: &mut bool,
) -> bool {
    const STATE_CAP: u64 = 2_000;
    *states = (*states).saturating_add(1);
    if *states > STATE_CAP {
        *capped = true;
        return false;
    }
    let mut selected: Option<(usize, Vec<[HexCoord; 3]>)> = None;
    for (index, is_covered) in covered.iter().copied().enumerate() {
        if is_covered {
            continue;
        }
        let compatible = candidates[index]
            .iter()
            .copied()
            .filter(|triple| triple_compatible(triple, used))
            .collect::<Vec<_>>();
        if compatible.is_empty() {
            return false;
        }
        if selected
            .as_ref()
            .is_none_or(|(_, old)| compatible.len() < old.len())
        {
            selected = Some((index, compatible));
        }
    }
    let Some((_index, choices)) = selected else {
        return true;
    };
    for triple in choices {
        let used_len = used.len();
        used.extend(triple);
        let mut changed = Vec::new();
        for (index, window) in windows.iter().enumerate() {
            if !covered[index] && triple.iter().all(|cell| window.empties.contains(cell)) {
                covered[index] = true;
                changed.push(index);
            }
        }
        if triple_cover_search(windows, candidates, covered, used, states, capped) {
            return true;
        }
        for index in changed {
            covered[index] = false;
        }
        used.truncate(used_len);
        if *capped {
            return false;
        }
    }
    false
}

fn disjoint_triple_extension(windows: &[DeadlineWindow]) -> (bool, u64, bool) {
    if windows.is_empty() {
        return (true, 1, false);
    }
    let candidates = windows
        .iter()
        .map(|window| triples_for(&window.empties))
        .collect::<Vec<_>>();
    if candidates.iter().any(Vec::is_empty) {
        return (false, 1, false);
    }
    let mut covered = vec![false; windows.len()];
    let mut used = Vec::new();
    let mut states = 0;
    let mut capped = false;
    let found = triple_cover_search(
        windows,
        &candidates,
        &mut covered,
        &mut used,
        &mut states,
        &mut capped,
    );
    (found, states, capped)
}

/// Evaluate the finite, stage-complete family used by the ES and triple
/// conjectures. `remaining` is a hypothetical absolute placement deadline
/// relative to this node, not the unbounded semantic horizon.
pub(crate) fn deadline_profile(
    state: &HexoState,
    claimant: Player,
    remaining: usize,
) -> DeadlineProfile {
    let started = Instant::now();
    let Some(claimant_slots) = claimant_slots_before_deadline(state, claimant, remaining) else {
        return DeadlineProfile::default();
    };
    if claimant_slots > 5 {
        return DeadlineProfile {
            claimant_slots,
            scan_nanos: elapsed_nanos(started),
            ..DeadlineProfile::default()
        };
    }
    let mut windows = Vec::new();
    for entry in state.board().windows().entries() {
        let count = entry.count(claimant);
        if count == 0 || entry.count(claimant.other()) != 0 {
            continue;
        }
        let gap = 6u8.saturating_sub(count);
        if gap <= claimant_slots {
            windows.push(DeadlineWindow {
                count,
                empties: entry.empty_cells(),
            });
        }
    }
    let family_windows = u32::try_from(windows.len()).unwrap_or(u32::MAX);
    let dtes = potential_lt_thirds(
        &potential_bins(&windows),
        deadline_threshold(state, claimant),
    );
    let scan_nanos = elapsed_nanos(started);

    let preblock_started = Instant::now();
    let (preblock, preblock_steps) = if dtes {
        (true, 0)
    } else {
        preblock_extension(state, claimant, &windows)
    };
    let preblock_nanos = elapsed_nanos(preblock_started);

    let triple_started = Instant::now();
    let (triple, triple_states, triple_capped) = if dtes {
        (true, 0, false)
    } else {
        disjoint_triple_extension(&windows)
    };
    let triple_nanos = elapsed_nanos(triple_started);
    DeadlineProfile {
        applicable: true,
        claimant_slots,
        family_windows,
        dtes,
        dtes_preblock: preblock,
        dtes_triple: triple,
        scan_nanos,
        preblock_nanos,
        triple_nanos,
        preblock_steps,
        triple_states,
        triple_capped,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Backend {
    Wide,
    Narrow,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Wide => "wide",
            Self::Narrow => "narrow",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WindowProfile {
    claimant_census: u8,
    opponent_census: u8,
    claimant_live: [u64; 6],
    opponent_live: [u64; 6],
    touched_windows: u64,
    mixed_windows: u64,
    invariant_ok: bool,
    claimant_phi_lt_one: bool,
    opponent_phi_lt_one: bool,
}

fn phi_lt_one(bins: &[u64; 6]) -> bool {
    // ES Corollary 2, in integer form, for attacker-touched live windows:
    // phi = b/9 + a/(9*sqrt(3)), where
    // a=n1+3n3+9n5 and b=n2+3n4.
    let a = u128::from(bins[1])
        .saturating_add(u128::from(bins[3]).saturating_mul(3))
        .saturating_add(u128::from(bins[5]).saturating_mul(9));
    let b = u128::from(bins[2]).saturating_add(u128::from(bins[4]).saturating_mul(3));
    if b > 8 {
        return false;
    }
    let gap = 9u128 - b;
    a.saturating_mul(a) < 3u128.saturating_mul(gap).saturating_mul(gap)
}

fn scan_windows(state: &HexoState, claimant: Player) -> WindowProfile {
    let mut out = WindowProfile {
        invariant_ok: true,
        ..WindowProfile::default()
    };
    for entry in state.board().windows().entries() {
        let cc = entry.count(claimant);
        let oc = entry.count(claimant.other());
        out.touched_windows = out.touched_windows.saturating_add(1);
        if cc > 5 || oc > 5 {
            out.invariant_ok = false;
        }
        if cc > 0 && oc == 0 {
            out.claimant_census = out.claimant_census.max(cc);
            if let Some(bin) = out.claimant_live.get_mut(usize::from(cc.min(5))) {
                *bin = bin.saturating_add(1);
            }
        }
        if oc > 0 && cc == 0 {
            out.opponent_census = out.opponent_census.max(oc);
            if let Some(bin) = out.opponent_live.get_mut(usize::from(oc.min(5))) {
                *bin = bin.saturating_add(1);
            }
        }
        if cc > 0 && oc > 0 {
            out.mixed_windows = out.mixed_windows.saturating_add(1);
        }
    }
    out.claimant_phi_lt_one = phi_lt_one(&out.claimant_live);
    out.opponent_phi_lt_one = phi_lt_one(&out.opponent_live);
    out
}

fn phase_code(phase: TurnPhase) -> u8 {
    match phase {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { .. } => 2,
    }
}

fn phase_name(code: u8) -> &'static str {
    match code {
        0 => "opening",
        1 => "first",
        2 => "second",
        _ => "unknown",
    }
}

fn lb_plies(phase: TurnPhase, census: u8) -> Option<u8> {
    if census > 5 {
        return None;
    }
    let m = match phase {
        TurnPhase::FirstStone if census >= 4 => 6 - census,
        TurnPhase::FirstStone => (7 - census).min(6),
        TurnPhase::SecondStone { .. } if census >= 3 => 6 - census,
        TurnPhase::SecondStone { .. } => (7 - census).min(6),
        TurnPhase::Opening => return None,
    };
    let index = usize::from(m.saturating_sub(1));
    match phase {
        TurnPhase::FirstStone => [1, 2, 5, 6, 9, 10].get(index).copied(),
        TurnPhase::SecondStone { .. } => [1, 4, 5, 8, 9, 12].get(index).copied(),
        TurnPhase::Opening => None,
    }
}

pub(crate) fn stage_dtw_evaluation(
    state: &HexoState,
    claimant: Player,
    remaining: usize,
) -> (bool, u64) {
    let started = Instant::now();
    let profile = scan_windows(state, claimant);
    let fires = remaining <= 8
        && profile.invariant_ok
        && coordinate_safe(state, i64::try_from(remaining).unwrap_or(i64::MAX))
        && state.current_player() == claimant
        && lb_plies(state.phase(), profile.claimant_census)
            .is_some_and(|bound| usize::from(bound) > remaining);
    (fires, elapsed_nanos(started))
}

fn coordinate_safe(state: &HexoState, h_rem: i64) -> bool {
    const SAFE: i64 = 16_383;
    if h_rem < 0 {
        return false;
    }
    let Some(radius) = h_rem.checked_add(1).and_then(|x| x.checked_mul(8)) else {
        return false;
    };
    let Some(limit) = SAFE.checked_sub(radius) else {
        return false;
    };
    state.board().occupied_cells().iter().all(|coord| {
        let q = i64::from(coord.q);
        let r = i64::from(coord.r);
        q.checked_add(r)
            .and_then(|sum| sum.checked_neg())
            .and_then(|s| Some((q.checked_abs()?, r.checked_abs()?, s.checked_abs()?)))
            .is_some_and(|(qa, ra, sa)| qa <= limit && ra <= limit && sa <= limit)
    })
}

fn horizon_bucket(h_rem: Option<i64>) -> &'static str {
    match h_rem {
        None => "underflow",
        Some(h) if h < 0 => "underflow",
        Some(0..=8) => "0_8",
        Some(9..=12) => "9_12",
        Some(13..=16) => "13_16",
        Some(17..=32) => "17_32",
        Some(33..=64) => "33_64",
        Some(65..=128) => "65_128",
        Some(129..=256) => "129_256",
        Some(_) => "257_plus",
    }
}

fn excess_bucket(h_rem: Option<i64>, lb: Option<u8>) -> &'static str {
    let (Some(h), Some(lb)) = (h_rem, lb) else {
        return "unavailable";
    };
    let delta = h.saturating_sub(i64::from(lb));
    match delta {
        i64::MIN..=-1 => "fires",
        0 => "strict_boundary",
        1..=4 => "miss_1_4",
        5..=8 => "miss_5_8",
        9..=16 => "miss_9_16",
        17..=64 => "miss_17_64",
        65..=256 => "miss_65_256",
        _ => "miss_257_plus",
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DepthStats {
    points: u64,
    claimant_owned: u64,
    census_hist: [u64; 6],
    opponent_census_hist: [u64; 6],
    phi_lt_one: u64,
    opponent_phi_lt_one: u64,
    stage_dtw_fires: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidateTotals {
    fires: u64,
    would_prunes: u64,
    expansion_mass: u64,
    counterexamples: u64,
    search_refuted: u64,
    search_unknown: u64,
    late_wins: u64,
    unresolved_wins: u64,
    evaluations: u64,
    evaluation_nanos: u64,
    work_units: u64,
    capped_evaluations: u64,
}

#[derive(Clone, Debug, Default)]
struct Report {
    counters: bool,
    shadow: bool,
    context_id: String,
    context_cap: u64,
    points: u64,
    scans: u64,
    scan_nanos: u64,
    backends: BTreeMap<Backend, u64>,
    first_failure: BTreeMap<&'static str, u64>,
    orthogonal: BTreeMap<&'static str, u64>,
    horizons: BTreeMap<&'static str, u64>,
    lb_relation: BTreeMap<&'static str, u64>,
    depth: BTreeMap<(Backend, u8, u16), DepthStats>,
    live_profile: BTreeMap<(u8, u8, u8), u64>,
    stage_remaining: BTreeMap<u16, u64>,
    current_evaluations: u64,
    current_dismissals: u64,
    invariant_failures: u64,
    coordinate_unsafe_h8: u64,
    min_h_rem: Option<i64>,
    max_h_rem: Option<i64>,
    shadow_audits: u64,
    shadow_audit_nanos: u64,
    shadow_pair_scans: u64,
    shadow_pair_nanos: u64,
    shadow_families_checked: u64,
    shadow_traversal_errors: u64,
    candidates: BTreeMap<Candidate, CandidateTotals>,
    counterexamples: Vec<String>,
}

thread_local! {
    static REPORT: RefCell<Report> = RefCell::new(Report::default());
}

pub(crate) fn counters_enabled() -> bool {
    std::env::var("TSS_CENSUS_DEEP_COUNTERS").ok().as_deref() == Some("1")
}

pub(crate) fn shadow_enabled() -> bool {
    std::env::var("TSS_CENSUS_DEEP_SHADOW").ok().as_deref() == Some("1")
}

pub(crate) fn reset() {
    let counters = counters_enabled();
    let shadow = shadow_enabled();
    REPORT.with(|slot| {
        *slot.borrow_mut() = Report {
            counters,
            shadow,
            ..Report::default()
        };
    });
}

pub(crate) fn set_context(id: &str, cap: u64) {
    REPORT.with(|slot| {
        let mut report = slot.borrow_mut();
        report.context_id.clear();
        report.context_id.push_str(id);
        report.context_cap = cap;
    });
}

pub(crate) fn record_shadow_audit(
    audits: &[CandidateAudit],
    audit_nanos: u64,
    pair_scans: u64,
    pair_nanos: u64,
    families_checked: u64,
    traversal_errors: u64,
) {
    REPORT.with(|slot| {
        let mut report = slot.borrow_mut();
        if !report.shadow {
            return;
        }
        report.shadow_audits = report.shadow_audits.saturating_add(1);
        report.shadow_audit_nanos = report.shadow_audit_nanos.saturating_add(audit_nanos);
        report.shadow_pair_scans = report.shadow_pair_scans.saturating_add(pair_scans);
        report.shadow_pair_nanos = report.shadow_pair_nanos.saturating_add(pair_nanos);
        report.shadow_families_checked = report
            .shadow_families_checked
            .saturating_add(families_checked);
        report.shadow_traversal_errors = report
            .shadow_traversal_errors
            .saturating_add(traversal_errors);
        for audit in audits {
            let Some(candidate) = audit.candidate else {
                continue;
            };
            let totals = report.candidates.entry(candidate).or_default();
            totals.fires = totals.fires.saturating_add(audit.fires);
            totals.would_prunes = totals.would_prunes.saturating_add(audit.would_prunes);
            totals.expansion_mass = totals.expansion_mass.saturating_add(audit.expansion_mass);
            totals.counterexamples = totals.counterexamples.saturating_add(audit.counterexamples);
            totals.search_refuted = totals.search_refuted.saturating_add(audit.search_refuted);
            totals.search_unknown = totals.search_unknown.saturating_add(audit.search_unknown);
            totals.late_wins = totals.late_wins.saturating_add(audit.late_wins);
            totals.unresolved_wins = totals.unresolved_wins.saturating_add(audit.unresolved_wins);
            totals.evaluations = totals.evaluations.saturating_add(audit.evaluations);
            totals.evaluation_nanos = totals
                .evaluation_nanos
                .saturating_add(audit.evaluation_nanos);
            totals.work_units = totals.work_units.saturating_add(audit.work_units);
            totals.capped_evaluations = totals
                .capped_evaluations
                .saturating_add(audit.capped_evaluations);
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_counterexample(
    candidate: Candidate,
    state: &HexoState,
    claimant: Player,
    entry: usize,
    depth: usize,
    expansion_events: u32,
    pair: Option<PairServiceProfile>,
    proof_resolution: Option<u32>,
    deadline: Option<u32>,
    path_from_root: &[HexCoord],
) {
    REPORT.with(|slot| {
        let mut report = slot.borrow_mut();
        if !report.shadow {
            return;
        }
        let mut stones = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                let owner = state.board().get(coord).expect("occupied cell has owner");
                (coord.q, coord.r, owner.index())
            })
            .collect::<Vec<_>>();
        stones.sort_unstable();
        let context_id = report.context_id.clone();
        let context_cap = report.context_cap;
        let path = path_from_root
            .iter()
            .map(|cell| (cell.q, cell.r))
            .collect::<Vec<_>>();
        report.counterexamples.push(format!(
            "CENSUS_DEEP_COUNTEREXAMPLE candidate={} id={} cap={} entry={} depth={} expansion_events={} claimant={} placements={} player={} phase={:?} proof_resolution={proof_resolution:?} deadline={deadline:?} pair={pair:?} path_from_root={path:?} stones={stones:?}",
            candidate.name(),
            context_id,
            context_cap,
            entry,
            depth,
            expansion_events,
            claimant.index(),
            state.placements_made(),
            state.current_player().index(),
            state.phase(),
        ));
    });
}

fn increment(map: &mut BTreeMap<&'static str, u64>, key: &'static str) {
    let count = map.entry(key).or_default();
    *count = count.saturating_add(1);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn observe_gate_point(
    state: &HexoState,
    claimant: Player,
    root_ply: u32,
    semantic_horizon: u32,
    depth: usize,
    stage_remaining: Option<usize>,
    backend: Backend,
) {
    let active = REPORT.with(|slot| slot.borrow().counters);
    if !active {
        return;
    }

    let started = Instant::now();
    // Scan every supported, post-opening point even if ownership or horizon
    // already makes the landed member ineligible. This is counterfactual
    // telemetry only and preserves the mandatory complete census recipe.
    let supported = matches!(
        state.phase(),
        TurnPhase::FirstStone | TurnPhase::SecondStone { .. }
    );
    let interior = state.placements_made() > root_ply;
    let owner = state.current_player() == claimant;
    let profile = (supported && interior).then(|| scan_windows(state, claimant));
    let scan_nanos = profile
        .is_some()
        .then(|| started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0);
    let h_rem = i64::from(semantic_horizon).checked_sub(i64::from(state.placements_made()));

    REPORT.with(|slot| {
        let mut report = slot.borrow_mut();
        report.points = report.points.saturating_add(1);
        let backend_count = report.backends.entry(backend).or_default();
        *backend_count = backend_count.saturating_add(1);
        increment(&mut report.horizons, horizon_bucket(h_rem));
        if let Some(h) = h_rem {
            report.min_h_rem = Some(report.min_h_rem.map_or(h, |old| old.min(h)));
            report.max_h_rem = Some(report.max_h_rem.map_or(h, |old| old.max(h)));
        }
        if owner {
            increment(&mut report.orthogonal, "claimant_owned");
        } else {
            increment(&mut report.orthogonal, "defender_owned");
        }
        if interior {
            increment(&mut report.orthogonal, "interior");
        } else {
            increment(&mut report.orthogonal, "root");
        }
        if supported {
            increment(&mut report.orthogonal, "phase_supported");
        } else {
            increment(&mut report.orthogonal, "opening");
        }

        let first_failure = if state.is_terminal() {
            "terminal"
        } else if !owner {
            "gate_shape_owner"
        } else if !interior {
            "gate_shape_root"
        } else if !supported {
            "gate_shape_opening"
        } else if h_rem.is_none_or(|h| h < 0) {
            "horizon_underflow"
        } else if h_rem.is_some_and(|h| h > 8) {
            "horizon_gt8"
        } else if !h_rem.is_some_and(|h| coordinate_safe(state, h)) {
            "coordinate_unsafe"
        } else if profile.is_some_and(|p| !p.invariant_ok) {
            "census_invariant"
        } else {
            "eligible"
        };
        increment(&mut report.first_failure, first_failure);

        let Some(profile) = profile else {
            return;
        };
        report.scans = report.scans.saturating_add(1);
        report.scan_nanos = report.scan_nanos.saturating_add(scan_nanos);
        report.invariant_failures = report
            .invariant_failures
            .saturating_add(u64::from(!profile.invariant_ok));
        report.coordinate_unsafe_h8 = report
            .coordinate_unsafe_h8
            .saturating_add(u64::from(!coordinate_safe(state, 8)));

        let phase = phase_code(state.phase());
        let depth_key = u16::try_from(depth).unwrap_or(u16::MAX);
        let lb = owner
            .then(|| lb_plies(state.phase(), profile.claimant_census))
            .flatten();
        let stage_fire = owner
            && stage_remaining
                .zip(lb)
                .is_some_and(|(remaining, bound)| usize::from(bound) > remaining);
        {
            let stats = report.depth.entry((backend, phase, depth_key)).or_default();
            stats.points = stats.points.saturating_add(1);
            stats.claimant_owned = stats.claimant_owned.saturating_add(u64::from(owner));
            if let Some(bin) = stats
                .census_hist
                .get_mut(usize::from(profile.claimant_census.min(5)))
            {
                *bin = bin.saturating_add(1);
            }
            if let Some(bin) = stats
                .opponent_census_hist
                .get_mut(usize::from(profile.opponent_census.min(5)))
            {
                *bin = bin.saturating_add(1);
            }
            stats.phi_lt_one = stats
                .phi_lt_one
                .saturating_add(u64::from(profile.claimant_phi_lt_one));
            stats.opponent_phi_lt_one = stats
                .opponent_phi_lt_one
                .saturating_add(u64::from(profile.opponent_phi_lt_one));
            stats.stage_dtw_fires = stats.stage_dtw_fires.saturating_add(u64::from(stage_fire));
        }

        let live_count = report
            .live_profile
            .entry((phase, profile.claimant_census, profile.opponent_census))
            .or_default();
        *live_count = live_count.saturating_add(1);

        if !owner {
            return;
        }
        increment(&mut report.lb_relation, excess_bucket(h_rem, lb));
        if h_rem.is_some_and(|h| (0..=8).contains(&h))
            && coordinate_safe(state, h_rem.unwrap_or_default())
            && profile.invariant_ok
        {
            report.current_evaluations = report.current_evaluations.saturating_add(1);
            report.current_dismissals = report.current_dismissals.saturating_add(u64::from(
                h_rem.is_some_and(|h| lb.is_some_and(|bound| i64::from(bound) > h)),
            ));
        }
        if let Some(remaining) = stage_remaining {
            let remaining_key = u16::try_from(remaining).unwrap_or(u16::MAX);
            let count = report.stage_remaining.entry(remaining_key).or_default();
            *count = count.saturating_add(1);
        }
    });
}

pub(crate) fn print_report() {
    REPORT.with(|slot| {
        let report = slot.borrow();
        if !report.counters && !report.shadow {
            return;
        }
        let mean_us = if report.scans == 0 {
            0.0
        } else {
            report.scan_nanos as f64 / report.scans as f64 / 1_000.0
        };
        println!(
            "CENSUS_DEEP_SUMMARY counters={} shadow={} points={} scans={} scan_ms={:.3} scan_mean_us={mean_us:.3} current_evals={} current_dismissals={} invariant_failures={} coord_unsafe_h8={} min_h_rem={:?} max_h_rem={:?}",
            report.counters,
            report.shadow,
            report.points,
            report.scans,
            report.scan_nanos as f64 / 1e6,
            report.current_evaluations,
            report.current_dismissals,
            report.invariant_failures,
            report.coordinate_unsafe_h8,
            report.min_h_rem,
            report.max_h_rem,
        );
        for (&backend, &count) in &report.backends {
            println!("CENSUS_DEEP_BACKEND backend={} points={count}", backend.name());
        }
        for (&reason, &count) in &report.first_failure {
            println!("CENSUS_DEEP_FAILURE reason={reason} count={count}");
        }
        for (&predicate, &count) in &report.orthogonal {
            println!("CENSUS_DEEP_SCOPE predicate={predicate} count={count}");
        }
        for (&bucket, &count) in &report.horizons {
            println!("CENSUS_DEEP_HORIZON bucket={bucket} count={count}");
        }
        for (&relation, &count) in &report.lb_relation {
            println!("CENSUS_DEEP_NEAR_MISS relation={relation} count={count}");
        }
        for (&remaining, &count) in &report.stage_remaining {
            println!("CENSUS_DEEP_STAGE_REMAINING plies={remaining} count={count}");
        }
        for (&(phase, claimant_c, opponent_c), &count) in &report.live_profile {
            println!(
                "CENSUS_DEEP_CENSUS phase={} claimant_c={claimant_c} opponent_c={opponent_c} count={count}",
                phase_name(phase),
            );
        }
        for (&(backend, phase, depth), stats) in &report.depth {
            println!(
                "CENSUS_DEEP_DEPTH backend={} phase={} depth={depth} points={} claimant_owned={} claimant_c_hist={:?} opponent_c_hist={:?} claimant_phi_lt1={} opponent_phi_lt1={} stage_dtw_fires={}",
                backend.name(),
                phase_name(phase),
                stats.points,
                stats.claimant_owned,
                stats.census_hist,
                stats.opponent_census_hist,
                stats.phi_lt_one,
                stats.opponent_phi_lt_one,
                stats.stage_dtw_fires,
            );
        }
        if report.shadow {
            println!(
                "CENSUS_DEEP_SHADOW_COST audits={} audit_ms={:.3} pair_scans={} pair_ms={:.3} families_checked={} traversal_errors={} counterexamples={}",
                report.shadow_audits,
                report.shadow_audit_nanos as f64 / 1e6,
                report.shadow_pair_scans,
                report.shadow_pair_nanos as f64 / 1e6,
                report.shadow_families_checked,
                report.shadow_traversal_errors,
                report.counterexamples.len(),
            );
            for candidate in CANDIDATES {
                let totals = report.candidates.get(&candidate).copied().unwrap_or_default();
                println!(
                    "CENSUS_DEEP_CANDIDATE name={} contract={} fires={} would_prunes={} expansion_mass={} counterexamples={} search_refuted={} search_unknown={} late_wins={} unresolved_wins={} evaluations={} eval_ms={:.3} work_units={} capped_evaluations={} verdict={}",
                    candidate.name(),
                    if candidate.is_bounded() { "stage" } else { "forcing_grammar" },
                    totals.fires,
                    totals.would_prunes,
                    totals.expansion_mass,
                    totals.counterexamples,
                    totals.search_refuted,
                    totals.search_unknown,
                    totals.late_wins,
                    totals.unresolved_wins,
                    totals.evaluations,
                    totals.evaluation_nanos as f64 / 1e6,
                    totals.work_units,
                    totals.capped_evaluations,
                    if totals.counterexamples != 0 {
                        "REFUTED"
                    } else if totals.unresolved_wins != 0 {
                        "INCONCLUSIVE"
                    } else if totals.fires == 0 {
                        "DRY"
                    } else {
                        "SURVIVES"
                    },
                );
            }
            for specimen in &report.counterexamples {
                println!("{specimen}");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{apply_placement, HexCoord, Placement};

    #[test]
    fn phi_integer_screen_matches_boundary_examples() {
        let mut bins = [0; 6];
        assert!(phi_lt_one(&bins));
        bins[4] = 3;
        assert!(!phi_lt_one(&bins), "three count-4 windows have phi=1");
        bins = [0; 6];
        bins[5] = 2;
        assert!(!phi_lt_one(&bins));
    }

    #[test]
    fn deadline_potential_uses_strict_exact_thresholds() {
        let mut bins = [0; 6];
        bins[4] = 3;
        assert!(!potential_lt_thirds(&bins, 3), "potential exactly one");
        bins[4] = 2;
        assert!(potential_lt_thirds(&bins, 3));
        assert!(!potential_lt_thirds(&bins, 2), "potential exactly 2/3");
    }

    #[test]
    fn pair_service_detects_disjoint_residuals() {
        let first = HexCoord::new(0, 0);
        let windows = vec![
            UpgradeWindow {
                strength: 3,
                empties: vec![first, HexCoord::new(1, 0)],
            },
            UpgradeWindow {
                strength: 3,
                empties: vec![first, HexCoord::new(0, 1)],
            },
        ];
        assert!(family_needs_two_hits(&windows, first, None));
        assert!(!family_needs_two_hits(&windows[..1], first, None));
    }

    #[test]
    fn disjoint_triple_cover_respects_cell_conflicts() {
        let a = HexCoord::new(0, 0);
        let b = HexCoord::new(1, 0);
        let c = HexCoord::new(2, 0);
        let disjoint = DeadlineWindow {
            count: 3,
            empties: vec![
                HexCoord::new(0, 2),
                HexCoord::new(1, 2),
                HexCoord::new(2, 2),
            ],
        };
        let first = DeadlineWindow {
            count: 3,
            empties: vec![a, b, c],
        };
        assert!(disjoint_triple_extension(&[first.clone(), disjoint]).0);
        let conflicting = DeadlineWindow {
            count: 3,
            empties: vec![a, HexCoord::new(0, 1), HexCoord::new(1, 1)],
        };
        assert!(!disjoint_triple_extension(&[first, conflicting]).0);
    }

    #[test]
    fn complete_scan_populates_census_without_a_horizon_gate() {
        let mut state = HexoState::new();
        for coord in [
            HexCoord::new(0, 0),
            HexCoord::new(1, 0),
            HexCoord::new(0, 1),
        ] {
            apply_placement(&mut state, Placement { coord }).expect("legal replay");
        }
        let claimant = state.current_player();
        let profile = scan_windows(&state, claimant);
        assert!(profile.touched_windows > 0);
        assert!(profile.invariant_ok);
        assert!(profile.claimant_census <= 5);
    }

    #[test]
    fn exact_flag_only() {
        let prior = std::env::var_os("TSS_CENSUS_DEEP_COUNTERS");
        std::env::set_var("TSS_CENSUS_DEEP_COUNTERS", "true");
        assert!(!counters_enabled());
        std::env::set_var("TSS_CENSUS_DEEP_COUNTERS", "1");
        assert!(counters_enabled());
        match prior {
            Some(value) => std::env::set_var("TSS_CENSUS_DEEP_COUNTERS", value),
            None => std::env::remove_var("TSS_CENSUS_DEEP_COUNTERS"),
        }
    }
}
