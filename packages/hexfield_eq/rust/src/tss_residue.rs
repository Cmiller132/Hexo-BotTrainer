//! Default-off, observation-only wall-time residue accounting.
//!
//! One thread-local stack clock owns each measurement job. `OtherMeasured` is
//! the root leaf and is timed directly. Named children pause their parents;
//! temporary Choice-edge leaves retain elapsed time until the completed node
//! can label them winner, ordering miss, or unresolved.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::OnceLock;
#[cfg(not(windows))]
use std::time::Instant;

pub(crate) const RESIDUE_SCHEMA_VERSION: u32 = 1;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn QueryPerformanceCounter(value: *mut i64) -> i32;
    fn QueryPerformanceFrequency(value: *mut i64) -> i32;
}

#[cfg(windows)]
fn wall_tick() -> u64 {
    let mut value = 0i64;
    let ok = unsafe { QueryPerformanceCounter(&mut value) };
    assert_ne!(ok, 0, "QueryPerformanceCounter failed");
    u64::try_from(value).expect("negative performance counter")
}

#[cfg(windows)]
fn wall_tick_frequency() -> u64 {
    static FREQUENCY: OnceLock<u64> = OnceLock::new();
    *FREQUENCY.get_or_init(|| {
        let mut value = 0i64;
        let ok = unsafe { QueryPerformanceFrequency(&mut value) };
        assert_ne!(ok, 0, "QueryPerformanceFrequency failed");
        u64::try_from(value).expect("nonpositive performance-counter frequency")
    })
}

#[cfg(windows)]
fn wall_ticks_to_ns(ticks: u64) -> u64 {
    u64::try_from(
        u128::from(ticks).saturating_mul(1_000_000_000) / u128::from(wall_tick_frequency()),
    )
    .unwrap_or(u64::MAX)
}

#[cfg(not(windows))]
fn wall_tick() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    nanos(ORIGIN.get_or_init(Instant::now).elapsed())
}

#[cfg(not(windows))]
fn wall_ticks_to_ns(ticks: u64) -> u64 {
    ticks
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ResidueCategory {
    DForcedGen = 0,
    DUnforcedFhwEligibleGen,
    DUnforcedNonfhwGen,
    DUnforcedUnclassifiedGen,
    AOrGen,
    AOrWinnerPath,
    AOrOrderingMiss,
    AOrUnresolved,
    TtProbe,
    TtStore,
    CensusGate,
    SearchBookkeeping,
    CertBuild,
    CertVerify,
    HorizonLadderOverhead,
    CapResumeOverhead,
    OtherMeasured,
}

impl ResidueCategory {
    pub(crate) const COUNT: usize = 17;
    pub(crate) const ALL: [Self; Self::COUNT] = [
        Self::DForcedGen,
        Self::DUnforcedFhwEligibleGen,
        Self::DUnforcedNonfhwGen,
        Self::DUnforcedUnclassifiedGen,
        Self::AOrGen,
        Self::AOrWinnerPath,
        Self::AOrOrderingMiss,
        Self::AOrUnresolved,
        Self::TtProbe,
        Self::TtStore,
        Self::CensusGate,
        Self::SearchBookkeeping,
        Self::CertBuild,
        Self::CertVerify,
        Self::HorizonLadderOverhead,
        Self::CapResumeOverhead,
        Self::OtherMeasured,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::DForcedGen => "D_FORCED_GEN",
            Self::DUnforcedFhwEligibleGen => "D_UNFORCED_FHW_ELIGIBLE_GEN",
            Self::DUnforcedNonfhwGen => "D_UNFORCED_NONFHW_GEN",
            Self::DUnforcedUnclassifiedGen => "D_UNFORCED_UNCLASSIFIED_GEN",
            Self::AOrGen => "A_OR_GEN",
            Self::AOrWinnerPath => "A_OR_WINNER_PATH",
            Self::AOrOrderingMiss => "A_OR_ORDERING_MISS",
            Self::AOrUnresolved => "A_OR_UNRESOLVED",
            Self::TtProbe => "TT_PROBE",
            Self::TtStore => "TT_STORE",
            Self::CensusGate => "CENSUS_GATE",
            Self::SearchBookkeeping => "SEARCH_BOOKKEEPING",
            Self::CertBuild => "CERT_BUILD",
            Self::CertVerify => "CERT_VERIFY",
            Self::HorizonLadderOverhead => "HORIZON_LADDER_OVERHEAD",
            Self::CapResumeOverhead => "CAP_RESUME_OVERHEAD",
            Self::OtherMeasured => "OTHER_MEASURED",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OrEdgeKey {
    pub(crate) choice_node: u64,
    pub(crate) child: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidueLeaf {
    Category(ResidueCategory),
    OrEdge(OrEdgeKey),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidueFrame {
    leaf: ResidueLeaf,
    elapsed_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidueJobKey {
    pub(crate) profile: String,
    pub(crate) row: String,
    pub(crate) cap_rung: u64,
    pub(crate) horizon_rung: String,
    pub(crate) horizon: u32,
    pub(crate) resume: bool,
    pub(crate) repetition: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UnforcedCounts {
    pub(crate) eligible: u64,
    pub(crate) noneligible: u64,
    pub(crate) unclassified: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidueJobReport {
    pub(crate) key: ResidueJobKey,
    pub(crate) status: String,
    pub(crate) nodes: u64,
    pub(crate) expansions: u64,
    pub(crate) tt_hits: u64,
    pub(crate) tt_entries: u64,
    pub(crate) peak_tt_bytes: u64,
    pub(crate) cert_nodes: u64,
    pub(crate) cert_edges: u64,
    pub(crate) strict_verify_result: Option<bool>,
    pub(crate) job_wall_ns: u64,
    pub(crate) category_ns: [u64; ResidueCategory::COUNT],
    pub(crate) other_measured_ns: u64,
    pub(crate) crosscheck_residual_ns: i128,
    pub(crate) crosscheck_abs_ns: u64,
    pub(crate) instrumentation_events: u64,
    pub(crate) horizon_cut: u64,
    pub(crate) horizon_cut_tall: u64,
    pub(crate) deep_kb_death: u64,
    pub(crate) cap_resume_advances: u64,
    pub(crate) cap_resume_reentries: u64,
    pub(crate) unforced_nodes: UnforcedCounts,
    pub(crate) overflow: bool,
    pub(crate) stack_imbalance: bool,
    pub(crate) missing_job_end_flush: bool,
    pub(crate) unmapped_or_edges: u64,
    pub(crate) valid: bool,
    pub(crate) invalid_reason: Option<String>,
}

impl ResidueJobReport {
    pub(crate) fn accounted_ns(&self) -> u64 {
        self.category_ns
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    pub(crate) fn allowed_residual_ns(&self) -> u64 {
        let half_percent = self.job_wall_ns.saturating_add(199) / 200;
        1_000_000u64.max(half_percent)
    }

    pub(crate) fn to_jsonl(&self) -> String {
        let mut out = String::with_capacity(2048);
        let key = &self.key;
        let _ = write!(
            out,
            "{{\"schema_version\":{},\"profile\":\"{}\",\"row\":\"{}\",\"rep\":{},\"cap_rung\":{},\"horizon_rung\":\"{}\",\"horizon\":{},\"resume\":{},\"status\":\"{}\",\"strict_verify_result\":{},\"nodes\":{},\"expansions\":{},\"tt_hits\":{},\"tt_entries\":{},\"peak_tt_bytes\":{},\"cert_nodes\":{},\"cert_edges\":{},\"job_wall_ns\":{},\"categories\":{{",
            RESIDUE_SCHEMA_VERSION,
            json_escape(&key.profile),
            json_escape(&key.row),
            key.repetition,
            key.cap_rung,
            json_escape(&key.horizon_rung),
            key.horizon,
            key.resume,
            json_escape(&self.status),
            self.strict_verify_result.map_or("null", |v| if v { "true" } else { "false" }),
            self.nodes,
            self.expansions,
            self.tt_hits,
            self.tt_entries,
            self.peak_tt_bytes,
            self.cert_nodes,
            self.cert_edges,
            self.job_wall_ns,
        );
        for (index, category) in ResidueCategory::ALL.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            let ns = self.category_ns[*category as usize];
            let _ = write!(out, "\"{}\":{}", category.name(), ns);
        }
        let _ = write!(
            out,
            "}},\"other_measured_ns\":{},\"crosscheck_residual_ns\":{},\"crosscheck_abs_ns\":{},\"instrumentation_events\":{},\"horizon_cut\":{},\"horizon_cut_tall\":{},\"deep_kb_death\":{},\"cap_resume_advances\":{},\"cap_resume_reentries\":{},\"unforced_nodes\":{{\"eligible\":{},\"noneligible\":{},\"unclassified\":{}}},\"overflow\":{},\"stack_imbalance\":{},\"missing_job_end_flush\":{},\"unmapped_or_edges\":{},\"valid\":{},\"invalid_reason\":{}}}",
            self.other_measured_ns,
            self.crosscheck_residual_ns,
            self.crosscheck_abs_ns,
            self.instrumentation_events,
            self.horizon_cut,
            self.horizon_cut_tall,
            self.deep_kb_death,
            self.cap_resume_advances,
            self.cap_resume_reentries,
            self.unforced_nodes.eligible,
            self.unforced_nodes.noneligible,
            self.unforced_nodes.unclassified,
            self.overflow,
            self.stack_imbalance,
            self.missing_job_end_flush,
            self.unmapped_or_edges,
            self.valid,
            self.invalid_reason.as_ref().map_or_else(|| "null".to_owned(), |reason| format!("\"{}\"", json_escape(reason))),
        );
        out
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedResidueRow {
    pub(crate) job_wall_ns: u64,
    pub(crate) category_ns: [u64; ResidueCategory::COUNT],
    pub(crate) valid: bool,
}

fn json_scalar<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let tail = line.split_once(&needle)?.1.trim_start();
    let end = tail
        .find(|ch: char| ch == ',' || ch == '}' || ch.is_whitespace())
        .unwrap_or(tail.len());
    Some(&tail[..end])
}

pub(crate) fn parse_jsonl_row(line: &str) -> Result<ParsedResidueRow, String> {
    if json_scalar(line, "schema_version") != Some("1") {
        return Err("unsupported or missing schema_version".to_owned());
    }
    let job_wall_ns = json_scalar(line, "job_wall_ns")
        .ok_or_else(|| "missing job_wall_ns".to_owned())?
        .parse::<u64>()
        .map_err(|_| "invalid job_wall_ns".to_owned())?;
    let mut category_ns = [0u64; ResidueCategory::COUNT];
    for category in ResidueCategory::ALL {
        category_ns[category as usize] = json_scalar(line, category.name())
            .ok_or_else(|| format!("missing {}", category.name()))?
            .parse::<u64>()
            .map_err(|_| format!("invalid {}", category.name()))?;
    }
    let valid = match json_scalar(line, "valid") {
        Some("true") => true,
        Some("false") => false,
        _ => return Err("missing or invalid valid flag".to_owned()),
    };
    Ok(ParsedResidueRow {
        job_wall_ns,
        category_ns,
        valid,
    })
}

pub(crate) fn render_aggregate_table(reports: &[ResidueJobReport]) -> String {
    let mut out = String::from(
        "| category | median sum ms | wall % | p95 row % | max row/id | disposition | measured value estimate | estimate method | eliminability upper bound |\n|---|---:|---:|---:|---|---|---:|---|---:|\n",
    );
    if reports.is_empty() {
        return out;
    }
    let mut repetitions = reports
        .iter()
        .map(|report| report.key.repetition)
        .collect::<Vec<_>>();
    repetitions.sort_unstable();
    repetitions.dedup();
    let median_index = (repetitions.len() - 1) / 2;
    let mut wall_sums = repetitions
        .iter()
        .map(|rep| {
            reports
                .iter()
                .filter(|row| row.key.repetition == *rep)
                .map(|row| row.job_wall_ns)
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    wall_sums.sort_unstable();
    let median_wall = wall_sums[median_index].max(1);
    for category in ResidueCategory::ALL {
        let mut sums = repetitions
            .iter()
            .map(|rep| {
                reports
                    .iter()
                    .filter(|row| row.key.repetition == *rep)
                    .map(|row| row.category_ns[category as usize])
                    .sum::<u64>()
            })
            .collect::<Vec<_>>();
        sums.sort_unstable();
        let median_sum = sums[median_index];
        let mut shares = reports
            .iter()
            .map(|row| {
                let share = row.category_ns[category as usize] as f64 * 100.0
                    / row.job_wall_ns.max(1) as f64;
                (share, row.key.row.as_str())
            })
            .collect::<Vec<_>>();
        shares.sort_by(|left, right| left.0.total_cmp(&right.0));
        let p95_index = ((shares.len() * 95).saturating_add(99) / 100)
            .saturating_sub(1)
            .min(shares.len() - 1);
        let p95 = shares[p95_index].0;
        let (maximum, maximum_row) = shares.last().copied().unwrap();
        let _ = writeln!(
            out,
            "| {} | {:.3} | {:.6} | {:.6} | {:.6}/{} | OPEN | NOT_MEASURED | ceiling only | {:.6} |",
            category.name(),
            median_sum as f64 / 1e6,
            median_sum as f64 * 100.0 / median_wall as f64,
            p95,
            maximum,
            maximum_row,
            median_sum as f64 * 100.0 / median_wall as f64,
        );
    }
    out
}

#[derive(Debug)]
pub(crate) struct ResidueClock {
    totals: [u64; ResidueCategory::COUNT],
    temporary_or: HashMap<OrEdgeKey, u64>,
    stack: Vec<ResidueFrame>,
    active_start_tick: u64,
    events: u64,
    overflow: bool,
    stack_imbalance: bool,
    ended: bool,
    unforced_nodes: UnforcedCounts,
}

impl ResidueClock {
    pub(crate) fn new_at(now_tick: u64) -> Self {
        Self {
            totals: [0; ResidueCategory::COUNT],
            temporary_or: HashMap::new(),
            stack: vec![ResidueFrame {
                leaf: ResidueLeaf::Category(ResidueCategory::OtherMeasured),
                elapsed_ns: 0,
            }],
            active_start_tick: now_tick,
            events: 0,
            overflow: false,
            stack_imbalance: false,
            ended: false,
            unforced_nodes: UnforcedCounts::default(),
        }
    }

    fn add_saturating(slot: &mut u64, value: u64, overflow: &mut bool) {
        let (sum, did_overflow) = slot.overflowing_add(value);
        if did_overflow {
            *slot = u64::MAX;
            *overflow = true;
        } else {
            *slot = sum;
        }
    }

    fn commit_frame(&mut self, frame: ResidueFrame) {
        match frame.leaf {
            ResidueLeaf::Category(category) => Self::add_saturating(
                &mut self.totals[category as usize],
                frame.elapsed_ns,
                &mut self.overflow,
            ),
            ResidueLeaf::OrEdge(key) => Self::add_saturating(
                self.temporary_or.entry(key).or_default(),
                frame.elapsed_ns,
                &mut self.overflow,
            ),
        }
    }

    fn flush_at(&mut self, now_tick: u64) {
        if self.stack.is_empty() {
            self.stack_imbalance = true;
            return;
        }
        let elapsed = match now_tick.checked_sub(self.active_start_tick) {
            Some(value) => value,
            None => {
                self.overflow = true;
                0
            }
        };
        let frame = self.stack.last_mut().expect("checked nonempty timer stack");
        Self::add_saturating(&mut frame.elapsed_ns, elapsed, &mut self.overflow);
        if self.stack.len() == 1 {
            let root = self.stack[0];
            self.stack[0].elapsed_ns = 0;
            self.commit_frame(root);
        }
        self.active_start_tick = now_tick;
    }

    fn enter_at(&mut self, leaf: ResidueLeaf, now_tick: u64) -> Option<usize> {
        if self.ended || self.stack.last().is_some_and(|frame| frame.leaf == leaf) {
            return None;
        }
        self.flush_at(now_tick);
        self.stack.push(ResidueFrame {
            leaf,
            elapsed_ns: 0,
        });
        Self::add_saturating(&mut self.events, 1, &mut self.overflow);
        Some(self.stack.len())
    }

    fn exit_at(&mut self, leaf: ResidueLeaf, depth: usize, now_tick: u64) {
        if self.ended
            || self.stack.len() != depth
            || !self.stack.last().is_some_and(|frame| frame.leaf == leaf)
        {
            self.stack_imbalance = true;
            return;
        }
        self.flush_at(now_tick);
        let frame = self.stack.pop().expect("checked nonempty timer stack");
        self.commit_frame(frame);
        self.active_start_tick = now_tick;
        Self::add_saturating(&mut self.events, 1, &mut self.overflow);
    }

    pub(crate) fn classify_or_edge(&mut self, key: OrEdgeKey, category: ResidueCategory) {
        debug_assert!(matches!(
            category,
            ResidueCategory::AOrWinnerPath
                | ResidueCategory::AOrOrderingMiss
                | ResidueCategory::AOrUnresolved
        ));
        if let Some(ns) = self.temporary_or.remove(&key) {
            Self::add_saturating(&mut self.totals[category as usize], ns, &mut self.overflow);
        }
    }

    fn count_unforced(&mut self, category: ResidueCategory) {
        let slot = match category {
            ResidueCategory::DUnforcedFhwEligibleGen => &mut self.unforced_nodes.eligible,
            ResidueCategory::DUnforcedNonfhwGen => &mut self.unforced_nodes.noneligible,
            ResidueCategory::DUnforcedUnclassifiedGen => &mut self.unforced_nodes.unclassified,
            _ => return,
        };
        Self::add_saturating(slot, 1, &mut self.overflow);
    }

    fn force_overflow_for_test(&mut self) {
        self.overflow = true;
    }
}

struct ActiveResidueJob {
    key: ResidueJobKey,
    origin_tick: u64,
    clock: ResidueClock,
    next_choice_key: u64,
}

thread_local! {
    static ACTIVE_JOB: RefCell<Option<Box<ActiveResidueJob>>> = const { RefCell::new(None) };
    static ACTIVE_JOB_PTR: Cell<*mut ActiveResidueJob> = const { Cell::new(std::ptr::null_mut()) };
}

pub(crate) struct ResidueScopeGuard {
    job: *mut ActiveResidueJob,
    leaf: ResidueLeaf,
    depth: Option<usize>,
}

impl ResidueScopeGuard {
    fn inactive(leaf: ResidueLeaf) -> Self {
        Self {
            job: std::ptr::null_mut(),
            leaf,
            depth: None,
        }
    }
}

impl Drop for ResidueScopeGuard {
    fn drop(&mut self) {
        let Some(depth) = self.depth.take() else {
            return;
        };
        let still_active = ACTIVE_JOB_PTR.with(|active| active.get() == self.job);
        if !still_active {
            return;
        }
        // The active job is boxed and therefore stays at a stable address.
        // Scope guards never outlive `end_job`, which requires a balanced
        // stack before taking the box. The raw pointer avoids a TLS/RefCell
        // lookup on every hot-path scope exit without changing attribution.
        if let Some(job) = unsafe { self.job.as_mut() } {
            job.clock.exit_at(self.leaf, depth, wall_tick());
        }
    }
}

pub(crate) fn scope(category: ResidueCategory) -> ResidueScopeGuard {
    scope_leaf(ResidueLeaf::Category(category))
}

pub(crate) fn or_edge_scope(choice_node: u64, child: u32) -> ResidueScopeGuard {
    scope_leaf(ResidueLeaf::OrEdge(OrEdgeKey { choice_node, child }))
}

fn scope_leaf(leaf: ResidueLeaf) -> ResidueScopeGuard {
    ACTIVE_JOB_PTR.with(|active| {
        let job_ptr = active.get();
        let Some(job) = (unsafe { job_ptr.as_mut() }) else {
            return ResidueScopeGuard::inactive(leaf);
        };
        ResidueScopeGuard {
            job: job_ptr,
            leaf,
            depth: job.clock.enter_at(leaf, wall_tick()),
        }
    })
}

pub(crate) fn begin_job(key: ResidueJobKey) {
    ACTIVE_JOB.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none(), "residue job already active");
        let origin_tick = wall_tick();
        let mut active = Box::new(ActiveResidueJob {
            key,
            origin_tick,
            clock: ResidueClock::new_at(origin_tick),
            next_choice_key: 1u64 << 63,
        });
        let active_ptr = active.as_mut() as *mut ActiveResidueJob;
        *slot = Some(active);
        ACTIVE_JOB_PTR.with(|pointer| pointer.set(active_ptr));
    });
}

pub(crate) struct OrChoiceTracker {
    choice_node: Option<u64>,
    keys: Vec<OrEdgeKey>,
}

impl OrChoiceTracker {
    pub(crate) fn new() -> Self {
        let choice_node = ACTIVE_JOB_PTR.with(|active| {
            let job = unsafe { active.get().as_mut() }?;
            let key = job.next_choice_key;
            job.next_choice_key = job.next_choice_key.saturating_add(1);
            Some(key)
        });
        Self {
            choice_node,
            keys: Vec::new(),
        }
    }

    pub(crate) fn next_edge(&mut self) -> Option<OrEdgeKey> {
        let choice_node = self.choice_node?;
        let key = OrEdgeKey {
            choice_node,
            child: u32::try_from(self.keys.len()).unwrap_or(u32::MAX),
        };
        self.keys.push(key);
        Some(key)
    }

    pub(crate) fn finish_winner(&mut self, winner: OrEdgeKey) {
        for key in self.keys.drain(..) {
            classify_or_edge(
                key,
                if key == winner {
                    ResidueCategory::AOrWinnerPath
                } else {
                    ResidueCategory::AOrOrderingMiss
                },
            );
        }
        self.choice_node = None;
    }
}

impl Drop for OrChoiceTracker {
    fn drop(&mut self) {
        for key in self.keys.drain(..) {
            classify_or_edge(key, ResidueCategory::AOrUnresolved);
        }
    }
}

pub(crate) fn job_active() -> bool {
    ACTIVE_JOB_PTR.with(|active| !active.get().is_null())
}

pub(crate) fn classify_or_edge(key: OrEdgeKey, category: ResidueCategory) {
    ACTIVE_JOB_PTR.with(|active| {
        if let Some(job) = unsafe { active.get().as_mut() } {
            job.clock.classify_or_edge(key, category);
        }
    });
}

pub(crate) fn count_unforced(category: ResidueCategory) {
    ACTIVE_JOB_PTR.with(|active| {
        if let Some(job) = unsafe { active.get().as_mut() } {
            job.clock.count_unforced(category);
        }
    });
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ResidueJobOutcome {
    pub(crate) status: String,
    pub(crate) nodes: u64,
    pub(crate) expansions: u64,
    pub(crate) tt_hits: u64,
    pub(crate) tt_entries: u64,
    pub(crate) peak_tt_bytes: u64,
    pub(crate) cert_nodes: u64,
    pub(crate) cert_edges: u64,
    pub(crate) strict_verify_result: Option<bool>,
    pub(crate) horizon_cut: u64,
    pub(crate) horizon_cut_tall: u64,
    pub(crate) deep_kb_death: u64,
    pub(crate) cap_resume_advances: u64,
    pub(crate) cap_resume_reentries: u64,
}

pub(crate) fn end_job(outcome: ResidueJobOutcome) -> ResidueJobReport {
    ACTIVE_JOB_PTR.with(|pointer| pointer.set(std::ptr::null_mut()));
    ACTIVE_JOB.with(|slot| {
        let mut active = *slot.borrow_mut().take().expect("no residue job active");
        let job_end_tick = wall_tick();
        let job_wall_ticks = job_end_tick
            .checked_sub(active.origin_tick)
            .unwrap_or_else(|| {
                active.clock.overflow = true;
                0
            });
        let job_wall_ns = wall_ticks_to_ns(job_wall_ticks);
        let stack_imbalance = active.clock.stack.len() != 1
            || !active.clock.stack.last().is_some_and(|frame| {
                frame.leaf == ResidueLeaf::Category(ResidueCategory::OtherMeasured)
            });
        active.clock.stack_imbalance |= stack_imbalance;
        active.clock.flush_at(job_end_tick);
        active.clock.ended = true;
        for total in &mut active.clock.totals {
            *total = wall_ticks_to_ns(*total);
        }
        let accounted = active
            .clock
            .totals
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add);
        let signed = i128::from(job_wall_ns) - i128::from(accounted);
        let abs = u64::try_from(signed.unsigned_abs()).unwrap_or(u64::MAX);
        let allowed = 1_000_000u64.max(job_wall_ns.saturating_add(199) / 200);
        let unmapped_or_edges = active.clock.temporary_or.len() as u64;
        let mut reasons = Vec::new();
        if signed < 0 {
            reasons.push("negative signed cross-check");
        }
        if abs > allowed {
            reasons.push("cross-check tolerance failure");
        }
        if active.clock.stack_imbalance {
            reasons.push("timer stack imbalance");
        }
        if unmapped_or_edges != 0 {
            reasons.push("unmapped temporary OR edge");
        }
        if active.clock.overflow {
            reasons.push("counter overflow");
        }
        let valid = reasons.is_empty();
        ResidueJobReport {
            key: active.key,
            status: outcome.status,
            nodes: outcome.nodes,
            expansions: outcome.expansions,
            tt_hits: outcome.tt_hits,
            tt_entries: outcome.tt_entries,
            peak_tt_bytes: outcome.peak_tt_bytes,
            cert_nodes: outcome.cert_nodes,
            cert_edges: outcome.cert_edges,
            strict_verify_result: outcome.strict_verify_result,
            job_wall_ns,
            category_ns: active.clock.totals,
            other_measured_ns: active.clock.totals[ResidueCategory::OtherMeasured as usize],
            crosscheck_residual_ns: signed,
            crosscheck_abs_ns: abs,
            instrumentation_events: active.clock.events,
            horizon_cut: outcome.horizon_cut,
            horizon_cut_tall: outcome.horizon_cut_tall,
            deep_kb_death: outcome.deep_kb_death,
            cap_resume_advances: outcome.cap_resume_advances,
            cap_resume_reentries: outcome.cap_resume_reentries,
            unforced_nodes: active.clock.unforced_nodes,
            overflow: active.clock.overflow,
            stack_imbalance: active.clock.stack_imbalance,
            missing_job_end_flush: false,
            unmapped_or_edges,
            valid,
            invalid_reason: (!valid).then(|| reasons.join("; ")),
        }
    })
}

fn nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub(crate) fn closure_paragraph_allowed(
    reports: &[ResidueJobReport],
    overhead_fraction: f64,
    open_categories_measured: bool,
) -> Result<(), Vec<String>> {
    let mut residuals = Vec::new();
    if reports.is_empty() {
        residuals.push("no residue jobs".to_owned());
        return Err(residuals);
    }
    if reports.iter().any(|report| !report.valid) {
        residuals.push("partition/cross-check gate failed".to_owned());
    }
    if overhead_fraction > 0.02 {
        residuals.push(format!(
            "instrumentation overhead {:.6}% exceeds 2%",
            overhead_fraction * 100.0
        ));
    }
    let wall = reports.iter().map(|report| report.job_wall_ns).sum::<u64>();
    let other = reports
        .iter()
        .map(|report| report.other_measured_ns)
        .sum::<u64>();
    if other.saturating_mul(100) > wall {
        residuals.push(format!(
            "OTHER_MEASURED {:.6}% exceeds 1%",
            other as f64 * 100.0 / wall as f64
        ));
    }
    if reports
        .iter()
        .any(|report| report.other_measured_ns.saturating_mul(50) > report.job_wall_ns)
    {
        residuals.push("at least one row has OTHER_MEASURED above 2%".to_owned());
    }
    if reports
        .iter()
        .any(|report| report.unforced_nodes.unclassified != 0)
    {
        residuals.push("D_UNFORCED_UNCLASSIFIED_GEN is nonzero".to_owned());
    }
    if !open_categories_measured {
        residuals.push("an OPEN category lacks a measured central estimate".to_owned());
    }
    if residuals.is_empty() {
        Ok(())
    } else {
        Err(residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn key(row: &str) -> ResidueJobKey {
        ResidueJobKey {
            profile: "unit".to_owned(),
            row: row.to_owned(),
            cap_rung: 10_000,
            horizon_rung: "base".to_owned(),
            horizon: 10,
            resume: false,
            repetition: 0,
        }
    }

    #[test]
    fn residue_clock_nested_scopes_partition_exactly() {
        let mut clock = ResidueClock::new_at(0);
        let a = ResidueLeaf::Category(ResidueCategory::SearchBookkeeping);
        let b = ResidueLeaf::Category(ResidueCategory::TtProbe);
        let da = clock.enter_at(a, 10).unwrap();
        let db = clock.enter_at(b, 30).unwrap();
        clock.exit_at(b, db, 50);
        clock.exit_at(a, da, 80);
        clock.flush_at(100);
        assert_eq!(clock.totals[ResidueCategory::OtherMeasured as usize], 30);
        assert_eq!(
            clock.totals[ResidueCategory::SearchBookkeeping as usize],
            50
        );
        assert_eq!(clock.totals[ResidueCategory::TtProbe as usize], 20);
        assert_eq!(clock.totals.iter().sum::<u64>(), 100);
    }

    #[test]
    fn residue_other_is_direct_not_subtracted() {
        let mut clock = ResidueClock::new_at(0);
        let leaf = ResidueLeaf::Category(ResidueCategory::CertBuild);
        let depth = clock.enter_at(leaf, 40).unwrap();
        clock.exit_at(leaf, depth, 70);
        clock.flush_at(100);
        assert_eq!(clock.totals[ResidueCategory::OtherMeasured as usize], 70);
        assert_eq!(clock.totals.iter().sum::<u64>(), 100);
    }

    #[test]
    fn residue_or_edges_finalize_winner_miss_unresolved_without_changing_sum() {
        let mut clock = ResidueClock::new_at(0);
        for (key, start, end, category) in [
            (
                OrEdgeKey {
                    choice_node: 1,
                    child: 0,
                },
                0,
                11,
                ResidueCategory::AOrOrderingMiss,
            ),
            (
                OrEdgeKey {
                    choice_node: 1,
                    child: 1,
                },
                11,
                30,
                ResidueCategory::AOrWinnerPath,
            ),
            (
                OrEdgeKey {
                    choice_node: 2,
                    child: 0,
                },
                30,
                37,
                ResidueCategory::AOrUnresolved,
            ),
        ] {
            clock.active_start_tick = start;
            let depth = clock.enter_at(ResidueLeaf::OrEdge(key), start).unwrap();
            clock.exit_at(ResidueLeaf::OrEdge(key), depth, end);
            clock.classify_or_edge(key, category);
        }
        assert_eq!(clock.totals[ResidueCategory::AOrOrderingMiss as usize], 11);
        assert_eq!(clock.totals[ResidueCategory::AOrWinnerPath as usize], 19);
        assert_eq!(clock.totals[ResidueCategory::AOrUnresolved as usize], 7);
        assert_eq!(clock.totals.iter().sum::<u64>(), 37);
        assert!(clock.temporary_or.is_empty());
    }

    #[test]
    fn residue_unforced_missing_or_bad_fhw_class_is_unclassified_not_ineligible() {
        let mut clock = ResidueClock::new_at(0);
        clock.count_unforced(ResidueCategory::DUnforcedUnclassifiedGen);
        clock.count_unforced(ResidueCategory::DUnforcedUnclassifiedGen);
        assert_eq!(clock.unforced_nodes.unclassified, 2);
        assert_eq!(clock.unforced_nodes.noneligible, 0);
        assert_eq!(clock.unforced_nodes.eligible, 0);
    }

    #[test]
    fn residue_forced_pair_fallback_is_counted_once() {
        let mut clock = ResidueClock::new_at(0);
        let forced = ResidueLeaf::Category(ResidueCategory::DForcedGen);
        let outer = clock.enter_at(forced, 0).unwrap();
        assert!(clock.enter_at(forced, 2).is_none());
        clock.exit_at(forced, outer, 9);
        assert_eq!(clock.totals[ResidueCategory::DForcedGen as usize], 9);
    }

    #[test]
    fn residue_tt_probe_store_nested_calls_do_not_double_count() {
        let mut clock = ResidueClock::new_at(0);
        let probe = ResidueLeaf::Category(ResidueCategory::TtProbe);
        let store = ResidueLeaf::Category(ResidueCategory::TtStore);
        let p = clock.enter_at(probe, 0).unwrap();
        assert!(clock.enter_at(probe, 1).is_none());
        clock.exit_at(probe, p, 5);
        let s = clock.enter_at(store, 5).unwrap();
        assert!(clock.enter_at(store, 6).is_none());
        clock.exit_at(store, s, 12);
        assert_eq!(clock.totals[ResidueCategory::TtProbe as usize], 5);
        assert_eq!(clock.totals[ResidueCategory::TtStore as usize], 7);
    }

    #[test]
    fn residue_horizon_tags_partition_base_tall_exact_retry() {
        let reports = ["base", "tall", "exact_retry"].map(|tag| {
            begin_job(ResidueJobKey {
                horizon_rung: tag.to_owned(),
                ..key(tag)
            });
            std::hint::black_box(1usize);
            end_job(ResidueJobOutcome {
                status: "UNKNOWN".to_owned(),
                ..Default::default()
            })
        });
        for report in &reports {
            assert_eq!(report.accounted_ns(), report.job_wall_ns);
            assert!(report.valid);
        }
    }

    #[test]
    fn residue_accounting_rejects_stack_leak_overflow_and_tolerance_failure() {
        let mut leak = ResidueClock::new_at(0);
        leak.enter_at(ResidueLeaf::Category(ResidueCategory::CertBuild), 1);
        assert_ne!(leak.stack.len(), 1);
        leak.force_overflow_for_test();
        assert!(leak.overflow);
        let wall = 10_000_000u64;
        let accounted = 8_000_000u64;
        let residual = wall.abs_diff(accounted);
        assert!(residual > 1_000_000u64.max((wall + 199) / 200));
    }

    #[test]
    fn residue_jsonl_snapshot_and_closure_refusal_parser() {
        begin_job(key("snapshot"));
        let report = end_job(ResidueJobOutcome {
            status: "UNKNOWN".to_owned(),
            ..Default::default()
        });
        let json = report.to_jsonl();
        assert!(json.starts_with("{\"schema_version\":1,\"profile\":\"unit\""));
        assert!(json.contains("\"D_FORCED_GEN\":"));
        assert!(json.contains("\"other_measured_ns\":"));
        let parsed = parse_jsonl_row(&json).expect("parse generated JSONL");
        assert_eq!(parsed.job_wall_ns, report.job_wall_ns);
        assert_eq!(parsed.category_ns, report.category_ns);
        assert!(parsed.valid);
        let table = render_aggregate_table(std::slice::from_ref(&report));
        assert!(table.starts_with("| category | median sum ms | wall % | p95 row % |"));
        assert!(table.contains("| OTHER_MEASURED |"));
        let refused =
            closure_paragraph_allowed(std::slice::from_ref(&report), 0.0, false).unwrap_err();
        assert!(refused
            .iter()
            .any(|reason| reason.contains("OPEN category")));
        assert!(refused
            .iter()
            .any(|reason| reason.contains("OTHER_MEASURED")));

        let mut unclassified = report.clone();
        unclassified.unforced_nodes.unclassified = 1;
        let refused = closure_paragraph_allowed(&[unclassified], 0.0, true).unwrap_err();
        assert!(refused
            .iter()
            .any(|reason| reason.contains("D_UNFORCED_UNCLASSIFIED_GEN")));

        let mut bad_crosscheck = report.clone();
        bad_crosscheck.valid = false;
        bad_crosscheck.crosscheck_abs_ns = bad_crosscheck.allowed_residual_ns() + 1;
        bad_crosscheck.invalid_reason = Some("cross-check residual exceeds tolerance".to_owned());
        let refused = closure_paragraph_allowed(&[bad_crosscheck], 0.0, true).unwrap_err();
        assert!(refused
            .iter()
            .any(|reason| reason.contains("partition/cross-check")));

        assert!(parse_jsonl_row("{\"schema_version\":1}").is_err());
    }
}

#[cfg(test)]
mod campaign {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use hexo_engine::{apply_placement, HexCoord, HexoState, Placement};

    use crate::tss_core::{CertVerify, DeepSolve, ProofStatus, SolveCaps};
    use crate::tss_solver::{TssSolver, WidthOptions};
    use crate::tss_verify::{CertNode, TssCertificate, TssVerifier};

    const HUMAN_POSITIONS: &str = "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/group2-zones/.codex-group2-shadow/human_positions.txt";

    fn repetition() -> u32 {
        std::env::var("TSS_RESIDUE_REP")
            .ok()
            .map(|value| value.parse().expect("numeric TSS_RESIDUE_REP"))
            .unwrap_or(0)
    }

    fn output_path(profile: &str) -> PathBuf {
        if let Ok(path) = std::env::var("TSS_RESIDUE_OUTPUT") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".codex-residue")
            .join(format!("{profile}-rep{}.jsonl", repetition()))
    }

    fn initialize_output(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create residue output directory");
        }
        fs::write(path, []).expect("initialize residue JSONL");
    }

    fn append_report(path: &Path, report: &ResidueJobReport) {
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open residue JSONL");
        writeln!(file, "{}", report.to_jsonl()).expect("append residue JSONL");
    }

    fn append_overhead_observation(
        path: &Path,
        profile: &str,
        repetition: Option<usize>,
        disabled_ns: u64,
        enabled_ns: u64,
        events: u64,
    ) {
        let overhead = enabled_ns as f64 / disabled_ns.max(1) as f64 - 1.0;
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open overhead JSONL");
        writeln!(
            file,
            "{{\"schema_version\":1,\"profile\":\"{}\",\"repetition\":{},\"disabled_ns\":{},\"enabled_ns\":{},\"overhead_pct\":{:.9},\"events\":{}}}",
            profile,
            repetition
                .map(|value| value.to_string())
                .unwrap_or_else(|| "null".to_owned()),
            disabled_ns,
            enabled_ns,
            overhead * 100.0,
            events,
        )
        .expect("append overhead JSONL");
        file.flush().expect("flush overhead JSONL");
    }

    fn overhead_tail(rows: &mut HashMap<String, (Vec<u64>, Vec<u64>)>) -> (u64, u64, f64, u64) {
        let mut disabled_medians = Vec::with_capacity(rows.len());
        let mut enabled_medians = Vec::with_capacity(rows.len());
        for (disabled, enabled) in rows.values_mut() {
            disabled.sort_unstable();
            enabled.sort_unstable();
            disabled_medians.push(disabled[(disabled.len() - 1) / 2]);
            enabled_medians.push(enabled[(enabled.len() - 1) / 2]);
        }
        disabled_medians.sort_unstable();
        enabled_medians.sort_unstable();
        let p95_index = (disabled_medians.len() * 95)
            .saturating_add(99)
            .checked_div(100)
            .unwrap_or(0)
            .saturating_sub(1)
            .min(disabled_medians.len().saturating_sub(1));
        let disabled_p95 = disabled_medians[p95_index];
        let enabled_p95 = enabled_medians[p95_index];
        (
            disabled_p95,
            enabled_p95,
            enabled_p95 as f64 / disabled_p95.max(1) as f64 - 1.0,
            enabled_p95.abs_diff(disabled_p95),
        )
    }

    fn status_name(status: ProofStatus) -> &'static str {
        match status {
            ProofStatus::Win => "WIN",
            ProofStatus::Loss => "LOSS",
            ProofStatus::Unknown => "UNKNOWN",
        }
    }

    fn cert_shape(cert: Option<&TssCertificate>) -> (u64, u64) {
        let Some(cert) = cert else {
            return (0, 0);
        };
        let edges = cert.nodes.iter().fold(0u64, |total, node| {
            let add = match node {
                CertNode::Choice { .. } => 1,
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => u64::try_from(edges.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(
                        u64::try_from(commutations.len())
                            .unwrap_or(u64::MAX)
                            .saturating_mul(2),
                    ),
                _ => 0,
            };
            total.saturating_add(add)
        });
        (u64::try_from(cert.nodes.len()).unwrap_or(u64::MAX), edges)
    }

    fn measure_job(
        profile: &str,
        row: &str,
        state: &HexoState,
        caps: SolveCaps,
        horizon_rung: &str,
        width: WidthOptions,
    ) -> (
        ResidueJobReport,
        crate::tss_core::DeepResult<TssCertificate>,
    ) {
        begin_job(ResidueJobKey {
            profile: profile.to_owned(),
            row: row.to_owned(),
            cap_rung: caps.node_cap,
            horizon_rung: horizon_rung.to_owned(),
            horizon: caps.semantic_horizon,
            resume: false,
            repetition: repetition(),
        });
        let mut solver = TssSolver::default();
        solver.set_width_options(width);
        let result = solver.solve(state, &caps);
        let strict = result
            .cert
            .as_ref()
            .map(|cert| TssVerifier.verify(state, cert, result.status));
        let (cert_nodes, cert_edges) = cert_shape(result.cert.as_ref());
        let report = end_job(ResidueJobOutcome {
            status: status_name(result.status).to_owned(),
            nodes: result.stats.nodes,
            expansions: result.stats.expansions,
            tt_hits: result.stats.tt_hits,
            tt_entries: result.stats.tt_entries,
            peak_tt_bytes: result.stats.peak_tt_bytes,
            cert_nodes,
            cert_edges,
            strict_verify_result: strict,
            ..Default::default()
        });
        assert!(
            report.valid,
            "{}: {}",
            row,
            report
                .invalid_reason
                .as_deref()
                .unwrap_or("invalid residue row")
        );
        assert!(strict != Some(false), "{row}: strict verifier rejection");
        (report, result)
    }

    fn solve_without_residue(
        state: &HexoState,
        caps: &SolveCaps,
        width: WidthOptions,
    ) -> (
        u64,
        crate::tss_core::DeepResult<TssCertificate>,
        Option<bool>,
    ) {
        assert!(!job_active());
        let mut solver = TssSolver::default();
        solver.set_width_options(width);
        let started = Instant::now();
        let result = solver.solve(state, caps);
        let strict = result
            .cert
            .as_ref()
            .map(|cert| TssVerifier.verify(state, cert, result.status));
        (nanos(started.elapsed()), result, strict)
    }

    fn assert_semantic_identity(
        label: &str,
        left: &crate::tss_core::DeepResult<TssCertificate>,
        left_verify: Option<bool>,
        right: &crate::tss_core::DeepResult<TssCertificate>,
        right_verify: Option<bool>,
    ) {
        assert_eq!(left.status, right.status, "{label}: status drift");
        assert_eq!(left.stats.nodes, right.stats.nodes, "{label}: node drift");
        assert_eq!(
            left.stats.expansions, right.stats.expansions,
            "{label}: expansion drift"
        );
        assert_eq!(
            left.stats.tt_hits, right.stats.tt_hits,
            "{label}: TT-hit drift"
        );
        assert_eq!(
            left.stats.tt_entries, right.stats.tt_entries,
            "{label}: TT-entry drift"
        );
        assert_eq!(
            left.cert, right.cert,
            "{label}: certificate-byte/shape drift"
        );
        assert_eq!(left_verify, right_verify, "{label}: verifier drift");
    }

    #[test]
    fn residue_verification_is_inside_job_wall() {
        let state = HexoState::new();
        begin_job(ResidueJobKey {
            row: "accepted".to_owned(),
            ..super::tests::key("accepted")
        });
        {
            let _guard = scope(ResidueCategory::CertVerify);
            std::hint::black_box(1usize);
        }
        let accepted = end_job(ResidueJobOutcome {
            status: "WIN".to_owned(),
            strict_verify_result: Some(true),
            ..Default::default()
        });
        assert!(accepted.category_ns[ResidueCategory::CertVerify as usize] <= accepted.job_wall_ns);

        for (name, strict) in [("rejected", Some(false)), ("unknown", None)] {
            begin_job(ResidueJobKey {
                row: name.to_owned(),
                ..super::tests::key(name)
            });
            if strict.is_some() {
                let _guard = scope(ResidueCategory::CertVerify);
                std::hint::black_box(&state);
            }
            let report = end_job(ResidueJobOutcome {
                status: name.to_uppercase(),
                strict_verify_result: strict,
                ..Default::default()
            });
            assert_eq!(report.accounted_ns(), report.job_wall_ns);
        }
    }

    #[test]
    fn residue_cap_resume_overhead_excludes_continued_search() {
        let mut clock = ResidueClock::new_at(0);
        let cap = ResidueLeaf::Category(ResidueCategory::CapResumeOverhead);
        let search = ResidueLeaf::Category(ResidueCategory::SearchBookkeeping);
        let dc = clock.enter_at(cap, 0).unwrap();
        let ds = clock.enter_at(search, 4).unwrap();
        clock.exit_at(search, ds, 19);
        clock.exit_at(cap, dc, 25);
        assert_eq!(
            clock.totals[ResidueCategory::CapResumeOverhead as usize],
            10
        );
        assert_eq!(
            clock.totals[ResidueCategory::SearchBookkeeping as usize],
            15
        );
    }

    #[test]
    #[ignore = "official F19 residue campaign; run in release, one process per repetition"]
    fn tss_residue_f19_gate() {
        let path = output_path("f19");
        initialize_output(&path);
        let tt_bytes_cap = 2usize << 30;
        let selected = std::env::var("TSS_RESIDUE_ID").ok();
        let max_cap = std::env::var("TSS_RESIDUE_MAX_CAP")
            .ok()
            .map(|value| value.parse::<u64>().expect("numeric TSS_RESIDUE_MAX_CAP"))
            .unwrap_or(20_000_000);
        for position in crate::tss_corpus::load_corpus() {
            if selected
                .as_ref()
                .is_some_and(|ids| !ids.split(',').any(|id| id == position.id))
            {
                continue;
            }
            let mut final_status = ProofStatus::Unknown;
            for cap in [10_000, 100_000, 1_000_000, 20_000_000]
                .into_iter()
                .filter(|cap| *cap <= max_cap)
            {
                if !position.expect_win && cap > 1_000_000 {
                    break;
                }
                let (report, result) = measure_job(
                    "f19",
                    &position.id,
                    &position.state,
                    SolveCaps {
                        node_cap: cap,
                        tt_bytes_cap,
                        semantic_horizon: u32::MAX,
                    },
                    "unbounded",
                    WidthOptions::vcf_pair_complete(),
                );
                append_report(&path, &report);
                final_status = result.status;
                if result.status != ProofStatus::Unknown {
                    break;
                }
            }
            if max_cap == 20_000_000 && position.expect_win {
                assert_eq!(
                    final_status,
                    ProofStatus::Win,
                    "{}: expected WIN",
                    position.id
                );
            } else if !position.expect_win {
                assert_ne!(final_status, ProofStatus::Win, "{}: false WIN", position.id);
            }
        }
        println!("RESIDUE_OUTPUT {}", path.display());
    }

    #[test]
    #[ignore = "canonical S2 residue campaign; run in release, one process per repetition"]
    fn tss_residue_spare_gate() {
        let path = output_path("s2");
        initialize_output(&path);
        for position in crate::tss_spare_corpus::load_spare_corpus() {
            let horizon = position
                .state
                .placements_made()
                .saturating_add(position.reference_plies);
            let (report, _) = measure_job(
                "s2",
                &position.id,
                &position.state,
                SolveCaps {
                    node_cap: 1_000_000,
                    tt_bytes_cap: 512 << 20,
                    semantic_horizon: horizon,
                },
                "base",
                WidthOptions::vcf_pair_complete(),
            );
            append_report(&path, &report);
        }
        println!("RESIDUE_OUTPUT {}", path.display());
    }

    fn load_human_positions() -> (Vec<(String, HexoState)>, Vec<String>) {
        let path = std::env::var("TSS_RESIDUE_HUMAN_POSITIONS")
            .unwrap_or_else(|_| HUMAN_POSITIONS.to_owned());
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read human positions {path}: {error}"));
        let mut positions = Vec::new();
        let mut skipped = Vec::new();
        for (line_index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((id, encoded)) = line.split_once(';') else {
                skipped.push(format!("line {}: missing semicolon", line_index + 1));
                continue;
            };
            let mut state = HexoState::new();
            let mut legal = true;
            for token in encoded.split_whitespace() {
                let Some((q, r)) = token.split_once(',') else {
                    legal = false;
                    break;
                };
                let Ok(q) = q.parse::<i16>() else {
                    legal = false;
                    break;
                };
                let Ok(r) = r.parse::<i16>() else {
                    legal = false;
                    break;
                };
                if apply_placement(
                    &mut state,
                    Placement {
                        coord: HexCoord::new(q, r),
                    },
                )
                .is_err()
                {
                    legal = false;
                    break;
                }
            }
            if legal && !state.is_terminal() {
                positions.push((id.to_owned(), state));
            } else {
                skipped.push(format!(
                    "line {} id={}: illegal or terminal replay",
                    line_index + 1,
                    id
                ));
            }
        }
        (positions, skipped)
    }

    #[test]
    #[ignore = "frozen 160-position human residue campaign"]
    fn tss_residue_human_gate() {
        let path = output_path("human160");
        initialize_output(&path);
        let (positions, skipped) = load_human_positions();
        assert_eq!(
            positions.len() + skipped.len(),
            160,
            "human cohort line-count drift"
        );
        let selected = std::env::var("TSS_RESIDUE_ID").ok();
        for reason in &skipped {
            eprintln!("RESIDUE_HUMAN_SKIP {reason}");
        }
        for (id, state) in positions {
            if selected
                .as_ref()
                .is_some_and(|ids| !ids.split(',').any(|wanted| wanted == id))
            {
                continue;
            }
            let horizon = state.placements_made().saturating_add(10);
            let (report, _) = measure_job(
                "human160",
                &id,
                &state,
                SolveCaps {
                    node_cap: 50_000,
                    tt_bytes_cap: 256 << 20,
                    semantic_horizon: horizon,
                },
                "base_rel10",
                WidthOptions::round3_consume(),
            );
            append_report(&path, &report);
        }
        println!(
            "RESIDUE_OUTPUT {} skipped={}",
            path.display(),
            skipped.len()
        );
    }

    #[test]
    #[ignore = "matched disabled/enabled A/A gate; full frozen profiles are intentionally expensive"]
    fn tss_residue_overhead_gate() {
        let overhead_path = output_path("overhead");
        initialize_output(&overhead_path);
        let corpus = crate::tss_corpus::load_corpus();
        let selected = std::env::var("TSS_RESIDUE_ID").ok();
        let overhead_repetitions = std::env::var("TSS_RESIDUE_OVERHEAD_REPS")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .expect("numeric overhead repetitions")
            })
            .unwrap_or(7);
        let max_cap = std::env::var("TSS_RESIDUE_OVERHEAD_MAX_CAP")
            .ok()
            .map(|value| value.parse::<u64>().expect("numeric overhead cap"))
            .unwrap_or(20_000_000);
        let mut disabled_totals = Vec::new();
        let mut enabled_totals = Vec::new();
        let mut event_totals = Vec::new();
        let mut f19_rows: HashMap<String, (Vec<u64>, Vec<u64>)> = HashMap::new();
        for repetition in 0..overhead_repetitions {
            let mut disabled = 0u64;
            let mut enabled = 0u64;
            let mut events = 0u64;
            for position in &corpus {
                if selected
                    .as_ref()
                    .is_some_and(|ids| !ids.split(',').any(|id| id == position.id))
                {
                    continue;
                }
                for cap in [10_000, 100_000, 1_000_000, 20_000_000]
                    .into_iter()
                    .filter(|cap| *cap <= max_cap)
                {
                    if !position.expect_win && cap > 1_000_000 {
                        break;
                    }
                    let caps = SolveCaps {
                        node_cap: cap,
                        tt_bytes_cap: 2usize << 30,
                        semantic_horizon: u32::MAX,
                    };
                    let (wall, plain, plain_verify, report, timed) = if repetition % 2 == 0 {
                        let (wall, plain, plain_verify) = solve_without_residue(
                            &position.state,
                            &caps,
                            WidthOptions::vcf_pair_complete(),
                        );
                        let (report, timed) = measure_job(
                            "overhead_f19",
                            &position.id,
                            &position.state,
                            caps,
                            "unbounded",
                            WidthOptions::vcf_pair_complete(),
                        );
                        (wall, plain, plain_verify, report, timed)
                    } else {
                        let (report, timed) = measure_job(
                            "overhead_f19",
                            &position.id,
                            &position.state,
                            caps,
                            "unbounded",
                            WidthOptions::vcf_pair_complete(),
                        );
                        let (wall, plain, plain_verify) = solve_without_residue(
                            &position.state,
                            &caps,
                            WidthOptions::vcf_pair_complete(),
                        );
                        (wall, plain, plain_verify, report, timed)
                    };
                    disabled = disabled.saturating_add(wall);
                    enabled = enabled.saturating_add(report.job_wall_ns);
                    events = events.saturating_add(report.instrumentation_events);
                    let timed_verify = timed
                        .cert
                        .as_ref()
                        .map(|cert| TssVerifier.verify(&position.state, cert, timed.status));
                    assert_semantic_identity(
                        &format!("{}@{cap}", position.id),
                        &plain,
                        plain_verify,
                        &timed,
                        timed_verify,
                    );
                    let row_wall = f19_rows
                        .entry(format!("{}@{cap}", position.id))
                        .or_default();
                    row_wall.0.push(wall);
                    row_wall.1.push(report.job_wall_ns);
                    if timed.status != ProofStatus::Unknown {
                        break;
                    }
                }
            }
            disabled_totals.push(disabled);
            enabled_totals.push(enabled);
            event_totals.push(events);
            append_overhead_observation(
                &overhead_path,
                "f19_sample",
                Some(repetition),
                disabled,
                enabled,
                events,
            );
        }
        disabled_totals.sort_unstable();
        enabled_totals.sort_unstable();
        let median = (overhead_repetitions - 1) / 2;
        let overhead = enabled_totals[median] as f64 / disabled_totals[median] as f64 - 1.0;
        event_totals.sort_unstable();
        let ns_per_event = enabled_totals[median].saturating_sub(disabled_totals[median]) as f64
            / event_totals[median].max(1) as f64;
        let (disabled_p95, enabled_p95, p95_regression, p95_abs_diff) =
            overhead_tail(&mut f19_rows);
        append_overhead_observation(
            &overhead_path,
            "f19_median",
            None,
            disabled_totals[median],
            enabled_totals[median],
            event_totals[median],
        );
        append_overhead_observation(
            &overhead_path,
            "f19_p95_row_median",
            None,
            disabled_p95,
            enabled_p95,
            0,
        );
        println!("RESIDUE_OVERHEAD profile=f19 disabled_ns={} enabled_ns={} overhead_pct={:.6} events={} ns_per_event={:.3}", disabled_totals[median], enabled_totals[median], overhead * 100.0, event_totals[median], ns_per_event);
        assert!(
            overhead <= 0.02,
            "F19 residue overhead {:.6}% exceeds hard 2%",
            overhead * 100.0
        );
        assert!(
            p95_regression <= 0.05 || (p95_abs_diff < 1_000_000 && overhead <= 0.02),
            "F19 residue p95 row regression {:.6}% (absolute {} ns) exceeds gate",
            p95_regression * 100.0,
            p95_abs_diff,
        );

        if selected.is_none() && max_cap == 20_000_000 && overhead_repetitions >= 7 {
            let spare = crate::tss_spare_corpus::load_spare_corpus();
            let mut s2_disabled = Vec::new();
            let mut s2_enabled = Vec::new();
            let mut s2_events = Vec::new();
            let mut s2_rows: HashMap<String, (Vec<u64>, Vec<u64>)> = HashMap::new();
            for repetition in 0..3 {
                let mut disabled = 0u64;
                let mut enabled = 0u64;
                let mut events = 0u64;
                for position in &spare {
                    let caps = SolveCaps {
                        node_cap: 1_000_000,
                        tt_bytes_cap: 512 << 20,
                        semantic_horizon: position
                            .state
                            .placements_made()
                            .saturating_add(position.reference_plies),
                    };
                    let (wall, plain, plain_verify, report, timed) = if repetition % 2 == 0 {
                        let (wall, plain, plain_verify) = solve_without_residue(
                            &position.state,
                            &caps,
                            WidthOptions::vcf_pair_complete(),
                        );
                        let (report, timed) = measure_job(
                            "overhead_s2",
                            &position.id,
                            &position.state,
                            caps,
                            "base",
                            WidthOptions::vcf_pair_complete(),
                        );
                        (wall, plain, plain_verify, report, timed)
                    } else {
                        let (report, timed) = measure_job(
                            "overhead_s2",
                            &position.id,
                            &position.state,
                            caps,
                            "base",
                            WidthOptions::vcf_pair_complete(),
                        );
                        let (wall, plain, plain_verify) = solve_without_residue(
                            &position.state,
                            &caps,
                            WidthOptions::vcf_pair_complete(),
                        );
                        (wall, plain, plain_verify, report, timed)
                    };
                    disabled = disabled.saturating_add(wall);
                    enabled = enabled.saturating_add(report.job_wall_ns);
                    events = events.saturating_add(report.instrumentation_events);
                    let timed_verify = timed
                        .cert
                        .as_ref()
                        .map(|cert| TssVerifier.verify(&position.state, cert, timed.status));
                    assert_semantic_identity(
                        &position.id,
                        &plain,
                        plain_verify,
                        &timed,
                        timed_verify,
                    );
                    let row_wall = s2_rows.entry(position.id.clone()).or_default();
                    row_wall.0.push(wall);
                    row_wall.1.push(report.job_wall_ns);
                }
                s2_disabled.push(disabled);
                s2_enabled.push(enabled);
                s2_events.push(events);
                append_overhead_observation(
                    &overhead_path,
                    "s2_sample",
                    Some(repetition),
                    disabled,
                    enabled,
                    events,
                );
            }
            s2_disabled.sort_unstable();
            s2_enabled.sort_unstable();
            s2_events.sort_unstable();
            let s2_overhead = s2_enabled[1] as f64 / s2_disabled[1] as f64 - 1.0;
            let s2_ns_per_event =
                s2_enabled[1].saturating_sub(s2_disabled[1]) as f64 / s2_events[1].max(1) as f64;
            let (s2_disabled_p95, s2_enabled_p95, s2_p95_regression, s2_p95_abs_diff) =
                overhead_tail(&mut s2_rows);
            append_overhead_observation(
                &overhead_path,
                "s2_median",
                None,
                s2_disabled[1],
                s2_enabled[1],
                s2_events[1],
            );
            append_overhead_observation(
                &overhead_path,
                "s2_p95_row_median",
                None,
                s2_disabled_p95,
                s2_enabled_p95,
                0,
            );
            println!("RESIDUE_OVERHEAD profile=s2 disabled_ns={} enabled_ns={} overhead_pct={:.6} events={} ns_per_event={:.3}", s2_disabled[1], s2_enabled[1], s2_overhead * 100.0, s2_events[1], s2_ns_per_event);
            assert!(
                s2_overhead <= 0.02,
                "S2 residue overhead {:.6}% exceeds hard 2%",
                s2_overhead * 100.0
            );
            assert!(
                s2_p95_regression <= 0.05 || (s2_p95_abs_diff < 1_000_000 && s2_overhead <= 0.02),
                "S2 residue p95 row regression {:.6}% (absolute {} ns) exceeds gate",
                s2_p95_regression * 100.0,
                s2_p95_abs_diff,
            );
        }
    }
}
