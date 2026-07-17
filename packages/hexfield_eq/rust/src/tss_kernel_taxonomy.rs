//! R-KT1 test-only forced-reply taxonomy shadow.
//!
//! The proven Q8 `K_reply` hook is a claimant Choice/OR filter.  This module
//! deliberately does not reuse that seam: it observes the forced Universal/AND
//! nodes built by `WidePnSearch`.  It is compiled only through lib.rs's
//! `#[cfg(test)]` module declaration and is inert unless
//! `TSS_KERNEL_TAXONOMY_SHADOW=1`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

use hexo_engine::coord::coords_within_radius;
use hexo_engine::{hex_distance, Axis, HexCoord, HexoState, Player, WindowKey};

use crate::tss_verify::RootBinding;

const MAX_SMALL_COVER: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplyVerdict {
    /// The claimant has a hard proof below this defender reply.
    DefenderFails,
    /// The restricted claimant TSS is genuinely exhausted below this reply.
    /// This excludes staged depth cutoffs but is not a certified opponent win.
    DefenderRefutes,
    /// Neither side is hard at the measured cap/horizon.
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UrgencyClass {
    Count4Only,
    Count5Only,
    Mixed,
    Empty,
}

impl UrgencyClass {
    fn suffix(self) -> &'static str {
        match self {
            Self::Count4Only => "C4",
            Self::Count5Only => "C5",
            Self::Mixed => "MIXED",
            Self::Empty => "EMPTY",
        }
    }
}

#[derive(Clone, Debug)]
struct ReplyStatus {
    coord: HexCoord,
    verdict: ReplyVerdict,
}

#[derive(Clone, Debug)]
struct CounterexampleSpecimen {
    class: &'static str,
    corpus_id: String,
    cap: u64,
    out_reply: HexCoord,
    kernel: Vec<HexCoord>,
    replies: Vec<ReplyStatus>,
    pair_edges: Vec<(HexCoord, HexCoord, ReplyVerdict)>,
    position: RootBinding,
}

#[derive(Clone, Debug, Default)]
struct ClassStats {
    conjectured: bool,
    fires: u64,
    safe: u64,
    counterexamples: u64,
    inconclusive: u64,
    legal_width_sum: u64,
    reply_width_sum: u64,
    child_width_sum: u64,
    kernel_width_sum: u64,
    work_units: u64,
    legal_width_min: usize,
    legal_width_max: usize,
    reply_width_min: usize,
    reply_width_max: usize,
    child_width_min: usize,
    child_width_max: usize,
    kernel_width_min: usize,
    kernel_width_max: usize,
    width_tuples: BTreeMap<(usize, usize, usize, usize), u64>,
}

impl ClassStats {
    fn observe_widths(
        &mut self,
        conjectured: bool,
        legal_width: usize,
        reply_width: usize,
        child_width: usize,
        kernel_width: usize,
        work_units: usize,
    ) {
        self.conjectured |= conjectured;
        self.fires = self.fires.saturating_add(1);
        self.legal_width_sum = self.legal_width_sum.saturating_add(legal_width as u64);
        self.reply_width_sum = self.reply_width_sum.saturating_add(reply_width as u64);
        self.child_width_sum = self.child_width_sum.saturating_add(child_width as u64);
        self.kernel_width_sum = self.kernel_width_sum.saturating_add(kernel_width as u64);
        self.work_units = self.work_units.saturating_add(work_units as u64);
        if self.fires == 1 {
            self.legal_width_min = legal_width;
            self.reply_width_min = reply_width;
            self.child_width_min = child_width;
            self.kernel_width_min = kernel_width;
        } else {
            self.legal_width_min = self.legal_width_min.min(legal_width);
            self.reply_width_min = self.reply_width_min.min(reply_width);
            self.child_width_min = self.child_width_min.min(child_width);
            self.kernel_width_min = self.kernel_width_min.min(kernel_width);
        }
        self.legal_width_max = self.legal_width_max.max(legal_width);
        self.reply_width_max = self.reply_width_max.max(reply_width);
        self.child_width_max = self.child_width_max.max(child_width);
        self.kernel_width_max = self.kernel_width_max.max(kernel_width);
        *self
            .width_tuples
            .entry((legal_width, reply_width, child_width, kernel_width))
            .or_default() += 1;
    }
}

#[derive(Clone, Debug, Default)]
struct ShadowReport {
    classes: BTreeMap<&'static str, ClassStats>,
    specimens: Vec<CounterexampleSpecimen>,
    and_nodes: u64,
    conjecture_fires: u64,
    skipped_fires: u64,
    total_child_width: u64,
    total_work_units: u64,
    audits: u64,
    audit_nanos: u64,
    traversal_errors: u64,
}

#[derive(Clone, Debug, Default)]
struct ShadowContext {
    corpus_id: String,
    cap: u64,
}

thread_local! {
    static REPORT: RefCell<ShadowReport> = RefCell::new(ShadowReport::default());
    static CONTEXT: RefCell<ShadowContext> = RefCell::new(ShadowContext::default());
}

pub(crate) fn enabled() -> bool {
    std::env::var("TSS_KERNEL_TAXONOMY_SHADOW").as_deref() == Ok("1")
}

pub(crate) fn reset() {
    REPORT.with(|slot| *slot.borrow_mut() = ShadowReport::default());
    CONTEXT.with(|slot| *slot.borrow_mut() = ShadowContext::default());
}

pub(crate) fn set_context(corpus_id: &str, cap: u64) {
    if !enabled() {
        return;
    }
    CONTEXT.with(|slot| {
        *slot.borrow_mut() = ShadowContext {
            corpus_id: corpus_id.to_owned(),
            cap,
        };
    });
}

pub(crate) fn record_audit(nanos: u64, traversal_errors: u64) {
    REPORT.with(|slot| {
        let mut report = slot.borrow_mut();
        report.audits = report.audits.saturating_add(1);
        report.audit_nanos = report.audit_nanos.saturating_add(nanos);
        report.traversal_errors = report.traversal_errors.saturating_add(traversal_errors);
    });
}

fn urgency_class(state: &HexoState, claimant: Player) -> UrgencyClass {
    let mut count4 = false;
    let mut count5 = false;
    for (owner, entry) in state.board().windows().live_threat_entries() {
        if owner != claimant || entry.active_player() != Some(claimant) {
            continue;
        }
        match entry.count(claimant) {
            4 => count4 = true,
            5 => count5 = true,
            _ => {}
        }
    }
    match (count4, count5) {
        (true, false) => UrgencyClass::Count4Only,
        (false, true) => UrgencyClass::Count5Only,
        (true, true) => UrgencyClass::Mixed,
        (false, false) => UrgencyClass::Empty,
    }
}

fn class_name(prefix: &str, urgency: UrgencyClass) -> &'static str {
    match (prefix, urgency) {
        ("F2_COVER1", UrgencyClass::Count4Only) => "F2_COVER1_C4",
        ("F2_COVER1", UrgencyClass::Count5Only) => "F2_COVER1_C5",
        ("F2_COVER1", UrgencyClass::Mixed) => "F2_COVER1_MIXED",
        ("F2_COVER1", UrgencyClass::Empty) => "F2_COVER1_EMPTY",
        ("F2_COVER2", UrgencyClass::Count4Only) => "F2_COVER2_C4",
        ("F2_COVER2", UrgencyClass::Count5Only) => "F2_COVER2_C5",
        ("F2_COVER2", UrgencyClass::Mixed) => "F2_COVER2_MIXED",
        ("F2_COVER2", UrgencyClass::Empty) => "F2_COVER2_EMPTY",
        ("F2_COVER3", UrgencyClass::Count4Only) => "F2_COVER3_C4",
        ("F2_COVER3", UrgencyClass::Count5Only) => "F2_COVER3_C5",
        ("F2_COVER3", UrgencyClass::Mixed) => "F2_COVER3_MIXED",
        ("F2_COVER3", UrgencyClass::Empty) => "F2_COVER3_EMPTY",
        ("F2_COVER4", UrgencyClass::Count4Only) => "F2_COVER4_C4",
        ("F2_COVER4", UrgencyClass::Count5Only) => "F2_COVER4_C5",
        ("F2_COVER4", UrgencyClass::Mixed) => "F2_COVER4_MIXED",
        ("F2_COVER4", UrgencyClass::Empty) => "F2_COVER4_EMPTY",
        ("F2_NO_SMALL_COVER", UrgencyClass::Count4Only) => "F2_NO_SMALL_COVER_C4",
        ("F2_NO_SMALL_COVER", UrgencyClass::Count5Only) => "F2_NO_SMALL_COVER_C5",
        ("F2_NO_SMALL_COVER", UrgencyClass::Mixed) => "F2_NO_SMALL_COVER_MIXED",
        ("F2_NO_SMALL_COVER", UrgencyClass::Empty) => "F2_NO_SMALL_COVER_EMPTY",
        ("F2_UNCOMPRESSED", UrgencyClass::Count4Only) => "F2_UNCOMPRESSED_C4",
        ("F2_UNCOMPRESSED", UrgencyClass::Count5Only) => "F2_UNCOMPRESSED_C5",
        ("F2_UNCOMPRESSED", UrgencyClass::Mixed) => "F2_UNCOMPRESSED_MIXED",
        ("F2_UNCOMPRESSED", UrgencyClass::Empty) => "F2_UNCOMPRESSED_EMPTY",
        ("S1_SINGLETON", UrgencyClass::Count4Only) => "S1_SINGLETON_C4",
        ("S1_SINGLETON", UrgencyClass::Count5Only) => "S1_SINGLETON_C5",
        ("S1_SINGLETON", UrgencyClass::Mixed) => "S1_SINGLETON_MIXED",
        ("S1_SINGLETON", UrgencyClass::Empty) => "S1_SINGLETON_EMPTY",
        ("S1_NO_CONJECTURE", UrgencyClass::Count4Only) => "S1_NO_CONJECTURE_C4",
        ("S1_NO_CONJECTURE", UrgencyClass::Count5Only) => "S1_NO_CONJECTURE_C5",
        ("S1_NO_CONJECTURE", UrgencyClass::Mixed) => "S1_NO_CONJECTURE_MIXED",
        ("S1_NO_CONJECTURE", UrgencyClass::Empty) => "S1_NO_CONJECTURE_EMPTY",
        ("S1_DEAD_SPOKE", _) => "S1_DEAD_SPOKE_C4",
        _ => unreachable!("unregistered taxonomy class {prefix}/{}", urgency.suffix()),
    }
}

fn coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

fn cover_recursive(
    edges: &[(usize, usize)],
    uncovered: &[usize],
    remaining: usize,
    chosen: &mut Vec<usize>,
) -> bool {
    let Some(&edge_index) = uncovered.first() else {
        return true;
    };
    if remaining == 0 {
        return false;
    }
    let (left, right) = edges[edge_index];
    for vertex in [left, right] {
        chosen.push(vertex);
        let next = uncovered
            .iter()
            .copied()
            .filter(|&index| {
                let (a, b) = edges[index];
                a != vertex && b != vertex
            })
            .collect::<Vec<_>>();
        if cover_recursive(edges, &next, remaining - 1, chosen) {
            return true;
        }
        chosen.pop();
    }
    false
}

/// Deterministic fixed-parameter vertex cover search.  `k<=4` makes this at
/// most 2^4 edge filters; it never scans combinations of all board cells.
fn smallest_cover(edges: &[(usize, usize)]) -> Option<Vec<usize>> {
    let uncovered = (0..edges.len()).collect::<Vec<_>>();
    for size in 1..=MAX_SMALL_COVER {
        let mut chosen = Vec::with_capacity(size);
        if cover_recursive(edges, &uncovered, size, &mut chosen) {
            chosen.sort_unstable();
            chosen.dedup();
            return Some(chosen);
        }
    }
    None
}

fn incident_window_dead(state: &HexoState, key: WindowKey) -> bool {
    state
        .board()
        .windows()
        .entries()
        .find(|entry| entry.key() == key)
        .is_some_and(|entry| entry.count(Player::Player0) > 0 && entry.count(Player::Player1) > 0)
}

fn old_supports(state: &HexoState, cell: HexCoord) -> bool {
    state.board().get(cell).is_some() || state.board().legal_moves().contains(cell)
}

/// Exact P2 dead-spoke/frontier-equivalence predicate from DOMINATION.md.
fn dead_spoke_interchangeable(
    state: &HexoState,
    claimant: Player,
    x: HexCoord,
    y: HexCoord,
) -> bool {
    let witness = state.board().windows().entries().find_map(|entry| {
        let mut empties = entry.empty_cells();
        empties.sort_unstable_by_key(|coord| coord_key(*coord));
        (entry.active_player() == Some(claimant)
            && entry.count(claimant) == 4
            && empties.as_slice() == [x, y])
        .then_some(entry.key())
    });
    let Some(witness) = witness else {
        return false;
    };

    for cell in [x, y] {
        for axis in Axis::ALL {
            for offset in 0..6i16 {
                let key = WindowKey {
                    start: cell - axis.vector().scale(offset),
                    axis,
                };
                if key != witness && !incident_window_dead(state, key) {
                    return false;
                }
            }
        }
    }

    let mut support_probe = coords_within_radius(x, 8)
        .chain(coords_within_radius(y, 8))
        .collect::<Vec<_>>();
    support_probe.sort_unstable_by_key(|coord| coord_key(*coord));
    support_probe.dedup();
    support_probe.into_iter().all(|cell| {
        let old = old_supports(state, cell);
        (old || hex_distance(x, cell) <= 8) == (old || hex_distance(y, cell) <= 8)
    })
}

fn evaluate(
    class: &'static str,
    conjectured: bool,
    state: &HexoState,
    legal_width: usize,
    reply_width: usize,
    child_width: usize,
    kernel: &[HexCoord],
    replies: &[ReplyStatus],
    pair_edges: &[(HexCoord, HexCoord, ReplyVerdict)],
    work_units: usize,
) {
    let kernel_set = kernel.iter().copied().collect::<HashSet<_>>();
    let all_kernel_fail = !kernel.is_empty()
        && kernel.iter().all(|cell| {
            replies
                .iter()
                .any(|reply| reply.coord == *cell && reply.verdict == ReplyVerdict::DefenderFails)
        });
    let kernel_refutes = replies.iter().any(|reply| {
        kernel_set.contains(&reply.coord) && reply.verdict == ReplyVerdict::DefenderRefutes
    });
    let all_full_fail = !replies.is_empty()
        && replies
            .iter()
            .all(|reply| reply.verdict == ReplyVerdict::DefenderFails);
    let out_refuters = replies
        .iter()
        .filter(|reply| {
            !kernel_set.contains(&reply.coord) && reply.verdict == ReplyVerdict::DefenderRefutes
        })
        .collect::<Vec<_>>();
    let counterexample = conjectured && all_kernel_fail && !out_refuters.is_empty();
    let safe = conjectured && !counterexample && (kernel_refutes || all_full_fail);

    REPORT.with(|slot| {
        let mut report = slot.borrow_mut();
        report.and_nodes = report.and_nodes.saturating_add(1);
        report.total_child_width = report.total_child_width.saturating_add(child_width as u64);
        report.total_work_units = report.total_work_units.saturating_add(work_units as u64);
        if conjectured {
            report.conjecture_fires = report.conjecture_fires.saturating_add(1);
        } else {
            report.skipped_fires = report.skipped_fires.saturating_add(1);
        }
        let stats = report.classes.entry(class).or_default();
        stats.observe_widths(
            conjectured,
            legal_width,
            reply_width,
            child_width,
            kernel.len(),
            work_units,
        );
        if counterexample {
            stats.counterexamples = stats
                .counterexamples
                .saturating_add(out_refuters.len() as u64);
        } else if safe {
            stats.safe = stats.safe.saturating_add(1);
        } else {
            stats.inconclusive = stats.inconclusive.saturating_add(1);
        }

        if counterexample {
            let context = CONTEXT.with(|context| context.borrow().clone());
            let position = RootBinding::from_state(state);
            for out_reply in out_refuters {
                report.specimens.push(CounterexampleSpecimen {
                    class,
                    corpus_id: context.corpus_id.clone(),
                    cap: context.cap,
                    out_reply: out_reply.coord,
                    kernel: kernel.to_vec(),
                    replies: replies.to_vec(),
                    pair_edges: pair_edges.to_vec(),
                    position: position.clone(),
                });
            }
        }
    });
}

pub(crate) fn observe_secondstone(
    state: &HexoState,
    claimant: Player,
    mut replies: Vec<(HexCoord, ReplyVerdict)>,
) {
    let legal_width = state.legal_move_count();
    replies.sort_unstable_by_key(|(coord, _)| coord_key(*coord));
    replies.dedup_by_key(|(coord, _)| *coord);
    let urgency = urgency_class(state, claimant);
    let statuses = replies
        .iter()
        .map(|&(coord, verdict)| ReplyStatus { coord, verdict })
        .collect::<Vec<_>>();
    match statuses.as_slice() {
        [only] => {
            let class = class_name("S1_SINGLETON", urgency);
            evaluate(
                class,
                true,
                state,
                legal_width,
                1,
                1,
                &[only.coord],
                &statuses,
                &[],
                1,
            );
        }
        [left, right] if dead_spoke_interchangeable(state, claimant, left.coord, right.coord) => {
            let class = class_name("S1_DEAD_SPOKE", urgency);
            evaluate(
                class,
                true,
                state,
                legal_width,
                2,
                2,
                &[left.coord],
                &statuses,
                &[],
                2,
            );
        }
        _ => {
            let class = class_name("S1_NO_CONJECTURE", urgency);
            evaluate(
                class,
                false,
                state,
                legal_width,
                statuses.len(),
                statuses.len(),
                &[],
                &statuses,
                &[],
                statuses.len(),
            );
        }
    }
}

pub(crate) fn observe_firststone_pairs(
    state: &HexoState,
    claimant: Player,
    pair_edges: Vec<(HexCoord, HexCoord, ReplyVerdict)>,
) {
    let legal_width = state.legal_move_count();
    let urgency = urgency_class(state, claimant);
    let mut vertices = pair_edges
        .iter()
        .flat_map(|(left, right, _)| [*left, *right])
        .collect::<Vec<_>>();
    vertices.sort_unstable_by_key(|coord| coord_key(*coord));
    vertices.dedup();
    let index = vertices
        .iter()
        .copied()
        .enumerate()
        .map(|(index, coord)| (coord, index))
        .collect::<HashMap<_, _>>();
    let graph = pair_edges
        .iter()
        .map(|(left, right, _)| (index[left], index[right]))
        .collect::<Vec<_>>();
    let Some(cover) = smallest_cover(&graph) else {
        let class = class_name("F2_NO_SMALL_COVER", urgency);
        let statuses = vertices
            .iter()
            .copied()
            .map(|coord| ReplyStatus {
                coord,
                verdict: ReplyVerdict::Unknown,
            })
            .collect::<Vec<_>>();
        evaluate(
            class,
            false,
            state,
            legal_width,
            vertices.len(),
            pair_edges.len(),
            &[],
            &statuses,
            &pair_edges,
            vertices
                .len()
                .saturating_add(pair_edges.len().saturating_mul(2)),
        );
        return;
    };
    let kernel = cover
        .iter()
        .map(|&vertex| vertices[vertex])
        .collect::<Vec<_>>();
    let prefix = match kernel.len() {
        1 => "F2_COVER1",
        2 => "F2_COVER2",
        3 => "F2_COVER3",
        4 => "F2_COVER4",
        _ => unreachable!("small-cover search returned oversized cover"),
    };
    let class = class_name(prefix, urgency);
    let statuses = vertices
        .iter()
        .copied()
        .map(|coord| {
            let incident = pair_edges
                .iter()
                .filter(|(left, right, _)| *left == coord || *right == coord)
                .map(|(_, _, verdict)| *verdict)
                .collect::<Vec<_>>();
            let verdict = if incident
                .iter()
                .any(|verdict| *verdict == ReplyVerdict::DefenderRefutes)
            {
                ReplyVerdict::DefenderRefutes
            } else if !incident.is_empty()
                && incident
                    .iter()
                    .all(|verdict| *verdict == ReplyVerdict::DefenderFails)
            {
                ReplyVerdict::DefenderFails
            } else {
                ReplyVerdict::Unknown
            };
            ReplyStatus { coord, verdict }
        })
        .collect::<Vec<_>>();
    evaluate(
        class,
        true,
        state,
        legal_width,
        vertices.len(),
        pair_edges.len(),
        &kernel,
        &statuses,
        &pair_edges,
        vertices
            .len()
            .saturating_add(pair_edges.len().saturating_mul(2)),
    );
}

pub(crate) fn observe_firststone_uncompressed(
    state: &HexoState,
    claimant: Player,
    mut replies: Vec<(HexCoord, ReplyVerdict)>,
) {
    let legal_width = state.legal_move_count();
    replies.sort_unstable_by_key(|(coord, _)| coord_key(*coord));
    replies.dedup_by_key(|(coord, _)| *coord);
    let urgency = urgency_class(state, claimant);
    let class = class_name("F2_UNCOMPRESSED", urgency);
    let statuses = replies
        .into_iter()
        .map(|(coord, verdict)| ReplyStatus { coord, verdict })
        .collect::<Vec<_>>();
    evaluate(
        class,
        false,
        state,
        legal_width,
        statuses.len(),
        statuses.len(),
        &[],
        &statuses,
        &[],
        statuses.len(),
    );
}

pub(crate) fn observe_unsupported(
    state: &HexoState,
    _claimant: Player,
    class: &'static str,
    child_width: usize,
) {
    evaluate(
        class,
        false,
        state,
        state.legal_move_count(),
        child_width,
        child_width,
        &[],
        &[],
        &[],
        child_width,
    );
}

pub(crate) fn print_report() {
    if !enabled() {
        return;
    }
    REPORT.with(|slot| {
        let report = slot.borrow();
        let counterexamples = report
            .classes
            .values()
            .map(|stats| stats.counterexamples)
            .sum::<u64>();
        println!(
            "KERNEL_TAXONOMY_SUMMARY audits={} and_nodes={} conjecture_fires={} skipped_fires={} total_child_width={} total_work_proxy_units={} counterexamples={} traversal_errors={} audit_ms={:.3}",
            report.audits,
            report.and_nodes,
            report.conjecture_fires,
            report.skipped_fires,
            report.total_child_width,
            report.total_work_units,
            counterexamples,
            report.traversal_errors,
            report.audit_nanos as f64 / 1e6,
        );
        for (&class, stats) in &report.classes {
            let retention = if stats.reply_width_sum == 0 {
                0.0
            } else {
                stats.kernel_width_sum as f64 / stats.reply_width_sum as f64
            };
            let work_share = if report.total_work_units == 0 {
                0.0
            } else {
                stats.work_units as f64 / report.total_work_units as f64
            };
            let child_share = if report.total_child_width == 0 {
                0.0
            } else {
                stats.child_width_sum as f64 / report.total_child_width as f64
            };
            println!(
                "KERNEL_TAXONOMY_CLASS class={class} conjecture={} fires={} safe={} counterexamples={} inconclusive={} legal_width_sum={} reply_width_sum={} child_width_sum={} child_share={child_share:.9} kernel_width_sum={} reply_retention={retention:.9} legal_width_min={} legal_width_max={} reply_width_min={} reply_width_max={} child_width_min={} child_width_max={} kernel_width_min={} kernel_width_max={} work_proxy_units={} work_proxy_share={work_share:.9}",
                u8::from(stats.conjectured),
                stats.fires,
                stats.safe,
                stats.counterexamples,
                stats.inconclusive,
                stats.legal_width_sum,
                stats.reply_width_sum,
                stats.child_width_sum,
                stats.kernel_width_sum,
                stats.legal_width_min,
                stats.legal_width_max,
                stats.reply_width_min,
                stats.reply_width_max,
                stats.child_width_min,
                stats.child_width_max,
                stats.kernel_width_min,
                stats.kernel_width_max,
                stats.work_units,
            );
            println!(
                "KERNEL_TAXONOMY_WIDTHS class={class} tuples_legal_reply_child_kernel={:?}",
                stats.width_tuples
            );
        }
        for specimen in &report.specimens {
            println!(
                "KERNEL_TAXONOMY_COUNTEREXAMPLE class={} corpus_id={} cap={} out_reply={:?} kernel={:?} replies={:?} pair_edges={:?} position={:?}",
                specimen.class,
                specimen.corpus_id,
                specimen.cap,
                specimen.out_reply,
                specimen.kernel,
                specimen.replies,
                specimen.pair_edges,
                specimen.position,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_taxonomy_small_cover_finds_star_and_cycle() {
        let star = vec![(0, 1), (0, 2), (0, 3)];
        assert_eq!(smallest_cover(&star), Some(vec![0]));

        let cycle = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let cover = smallest_cover(&cycle).expect("C4 has a size-two cover");
        assert_eq!(cover.len(), 2);
        assert!(cycle
            .iter()
            .all(|(left, right)| cover.contains(left) || cover.contains(right)));
    }

    #[test]
    fn kernel_taxonomy_small_cover_rejects_matching_five() {
        let matching = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
        assert_eq!(smallest_cover(&matching), None);
    }

    #[test]
    fn kernel_taxonomy_load_bearing_requires_resolved_kernel_failure() {
        reset();
        let state = HexoState::new();
        let inside = HexCoord::new(0, 0);
        let outside = HexCoord::new(1, 0);
        let replies = vec![
            ReplyStatus {
                coord: inside,
                verdict: ReplyVerdict::DefenderFails,
            },
            ReplyStatus {
                coord: outside,
                verdict: ReplyVerdict::DefenderRefutes,
            },
        ];
        evaluate(
            "TEST_LOAD_BEARING",
            true,
            &state,
            2,
            2,
            2,
            &[inside],
            &replies,
            &[],
            2,
        );
        REPORT.with(|slot| {
            let report = slot.borrow();
            assert_eq!(report.classes["TEST_LOAD_BEARING"].counterexamples, 1);
            assert_eq!(report.specimens.len(), 1);
            assert_eq!(report.specimens[0].out_reply, outside);
        });

        reset();
        let unresolved = vec![
            ReplyStatus {
                coord: inside,
                verdict: ReplyVerdict::Unknown,
            },
            ReplyStatus {
                coord: outside,
                verdict: ReplyVerdict::DefenderRefutes,
            },
        ];
        evaluate(
            "TEST_UNKNOWN",
            true,
            &state,
            2,
            2,
            2,
            &[inside],
            &unresolved,
            &[],
            2,
        );
        REPORT.with(|slot| {
            let report = slot.borrow();
            assert_eq!(report.classes["TEST_UNKNOWN"].counterexamples, 0);
            assert_eq!(report.classes["TEST_UNKNOWN"].inconclusive, 1);
            assert!(report.specimens.is_empty());
        });
    }
}
