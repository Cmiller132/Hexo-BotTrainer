//! Exact, dependency-free native kernel for the Horizon R3 h13/h14 endpoint.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

pub type CellId = u16;
const NONE: CellId = CellId::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    First,
    Second,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Action {
    pub a: CellId,
    pub b: CellId,
}

impl Action {
    pub const EMPTY: Self = Self { a: NONE, b: NONE };

    pub fn one(a: CellId) -> Self {
        Self { a, b: NONE }
    }

    pub fn pair(a: CellId, b: CellId) -> Self {
        debug_assert_ne!(a, b);
        if a < b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }

    pub fn len(self) -> usize {
        usize::from(self.a != NONE) + usize::from(self.b != NONE)
    }

    pub fn code(self) -> u32 {
        (u32::from(self.a) << 16) | u32::from(self.b)
    }

    fn each(self, mut f: impl FnMut(CellId)) {
        if self.a != NONE {
            f(self.a);
        }
        if self.b != NONE {
            f(self.b);
        }
    }

    pub fn as_vec(self) -> Vec<CellId> {
        let mut out = Vec::with_capacity(self.len());
        self.each(|x| out.push(x));
        out
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub q: i32,
    pub r: i32,
    pub anchored: bool,
    pub root_legal: bool,
}

#[derive(Clone, Debug)]
pub struct Edge {
    cells: [CellId; 6],
    len: u8,
}

impl Edge {
    pub fn new(raw: &[CellId]) -> Result<Self, String> {
        if raw.is_empty() || raw.len() > 6 {
            return Err(format!("edge length {} is outside 1..=6", raw.len()));
        }
        let mut sorted = raw.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err("edge contains a duplicate cell".to_string());
        }
        let mut cells = [NONE; 6];
        cells[..sorted.len()].copy_from_slice(&sorted);
        Ok(Self { cells, len: sorted.len() as u8 })
    }

    pub fn cells(&self) -> &[CellId] {
        &self.cells[..usize::from(self.len)]
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub id: String,
    pub horizon: u8,
    pub phase: Phase,
    pub timeout_ms: Option<u64>,
    pub cells: Vec<Cell>,
    pub target_anchored: Vec<Edge>,
    pub opponent_anchored: Vec<Edge>,
    pub near: Vec<Edge>,
    pub preferred: Option<Action>,
    pub preferred_required: Option<CellId>,
}

impl Model {
    pub fn validate(&self) -> Result<(), String> {
        if self.cells.len() >= usize::from(NONE) {
            return Err("universe does not fit in u16 cell IDs".to_string());
        }
        match (self.phase, self.horizon) {
            (Phase::First, 13 | 14) | (Phase::Second, 13) => {}
            _ => return Err("supported clocks are fresh h13/h14 and SecondStone h13".to_string()),
        }
        for edge in self.target_anchored.iter().chain(&self.opponent_anchored).chain(&self.near) {
            if edge.cells().iter().any(|&x| usize::from(x) >= self.cells.len()) {
                return Err("edge cell index is outside the universe".to_string());
            }
        }
        if let Some(action) = self.preferred {
            let mut valid = true;
            action.each(|x| valid &= usize::from(x) < self.cells.len());
            if !valid {
                return Err("preferred action cell is outside the universe".to_string());
            }
            if action.a != NONE && action.a == action.b {
                return Err("preferred action contains a duplicate cell".to_string());
            }
        }
        if self.preferred_required.is_some_and(|cell| usize::from(cell) >= self.cells.len()) {
            return Err("preferred required cell is outside the universe".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn new(n: usize) -> Self {
        Self { words: vec![0; n.div_ceil(64)] }
    }

    #[inline]
    fn contains(&self, cell: CellId) -> bool {
        let i = usize::from(cell);
        self.words[i >> 6] & (1_u64 << (i & 63)) != 0
    }

    #[inline]
    fn insert(&mut self, cell: CellId) {
        let i = usize::from(cell);
        self.words[i >> 6] |= 1_u64 << (i & 63);
    }

    #[inline]
    fn remove(&mut self, cell: CellId) {
        let i = usize::from(cell);
        self.words[i >> 6] &= !(1_u64 << (i & 63));
    }
}

#[derive(Clone, Debug)]
struct Position {
    a: BitSet,
    d: BitSet,
    a_cells: Vec<CellId>,
    d_cells: Vec<CellId>,
}

#[derive(Clone, Copy, Debug)]
enum Player {
    Attacker,
    Defender,
}

impl Position {
    fn new(n: usize) -> Self {
        Self { a: BitSet::new(n), d: BitSet::new(n), a_cells: Vec::with_capacity(6), d_cells: Vec::with_capacity(4) }
    }

    fn play(&mut self, player: Player, action: Action) -> usize {
        let before = match player { Player::Attacker => self.a_cells.len(), Player::Defender => self.d_cells.len() };
        action.each(|cell| match player {
            Player::Attacker => { debug_assert!(!self.a.contains(cell) && !self.d.contains(cell)); self.a.insert(cell); self.a_cells.push(cell); }
            Player::Defender => { debug_assert!(!self.a.contains(cell) && !self.d.contains(cell)); self.d.insert(cell); self.d_cells.push(cell); }
        });
        before
    }

    fn undo(&mut self, player: Player, before: usize) {
        match player {
            Player::Attacker => while self.a_cells.len() > before { let x = self.a_cells.pop().unwrap(); self.a.remove(x); },
            Player::Defender => while self.d_cells.len() > before { let x = self.d_cells.pop().unwrap(); self.d.remove(x); },
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Residual {
    cells: [CellId; 6],
    len: u8,
}

impl Residual {
    fn cells(&self) -> &[CellId] {
        &self.cells[..usize::from(self.len)]
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub nodes: u64,
    pub first_actions: u64,
    pub defender1_actions: u64,
    pub attacker2_actions: u64,
    pub defender2_actions: u64,
    pub attacker3_actions: u64,
    pub cache_hits_a2: u64,
    pub cache_hits_d2: u64,
    pub cache_hits_a3: u64,
    pub first_action_total: usize,
    pub quotient_classes_built: u64,
    pub quotient_actions_skipped: u64,
    pub d1_illegal_pairs_pruned: u64,
    pub d1_singleton_actions: u64,
    pub residual_edges_pruned: u64,
    pub shortcut_d1_defender_completion: u64,
    pub shortcut_d1_attacker_fork: u64,
    pub shortcut_d2_defender_completion: u64,
    pub shortcut_d2_attacker_fork: u64,
    pub shortcut_d1_noncover_replies: u64,
    pub shortcut_d2_noncover_replies: u64,
    pub shortcut_a2_immediate_completion: u64,
    pub shortcut_a2_noncover_actions: u64,
    pub shortcut_a2_defender_uncoverable: u64,
    pub a2_cover_actions_generated: u64,
    pub d2_cover_actions_generated: u64,
}

#[derive(Clone, Debug)]
pub enum Status {
    Win { witness: Action },
    Negative,
    Timeout,
    Error(String),
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub id: String,
    pub horizon: u8,
    pub phase: Phase,
    pub status: Status,
    pub stats: Stats,
    pub universe: usize,
    pub target_windows: usize,
    pub opponent_windows: usize,
    pub cache_entries: usize,
    pub wall: Duration,
}

impl Decision {
    pub fn json_line(&self) -> String {
        fn esc(s: &str) -> String {
            let mut out = String::with_capacity(s.len() + 2);
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c.is_control() => out.push('?'),
                    c => out.push(c),
                }
            }
            out
        }
        let (status, witness, error) = match &self.status {
            Status::Win { witness } => ("win", format!("[{}]", witness.as_vec().iter().map(ToString::to_string).collect::<Vec<_>>().join(",")), "null".to_string()),
            Status::Negative => ("negative", "null".to_string(), "null".to_string()),
            Status::Timeout => ("timeout", "null".to_string(), "null".to_string()),
            Status::Error(message) => ("error", "null".to_string(), format!("\"{}\"", esc(message))),
        };
        let phase = match self.phase { Phase::First => "first", Phase::Second => "second" };
        format!(
            "{{\"id\":\"{}\",\"horizon\":{},\"phase\":\"{}\",\"status\":\"{}\",\"witness_first_action\":{},\"error\":{},\"wall_ns\":{},\"universe\":{},\"target_windows\":{},\"opponent_windows\":{},\"first_action_total\":{},\"nodes\":{},\"first_actions\":{},\"defender1_actions\":{},\"attacker2_actions\":{},\"defender2_actions\":{},\"attacker3_actions\":{},\"cache_hits_a2\":{},\"cache_hits_d2\":{},\"cache_hits_a3\":{},\"cache_entries\":{},\"quotient_classes_built\":{},\"quotient_actions_skipped\":{},\"d1_illegal_pairs_pruned\":{},\"d1_singleton_actions\":{},\"residual_edges_pruned\":{},\"shortcut_d1_defender_completion\":{},\"shortcut_d1_attacker_fork\":{},\"shortcut_d2_defender_completion\":{},\"shortcut_d2_attacker_fork\":{},\"shortcut_d1_noncover_replies\":{},\"shortcut_d2_noncover_replies\":{},\"shortcut_a2_immediate_completion\":{},\"shortcut_a2_noncover_actions\":{},\"shortcut_a2_defender_uncoverable\":{},\"a2_cover_actions_generated\":{},\"d2_cover_actions_generated\":{}}}",
            esc(&self.id), self.horizon, phase, status, witness, error, self.wall.as_nanos(), self.universe,
            self.target_windows, self.opponent_windows, self.stats.first_action_total, self.stats.nodes,
            self.stats.first_actions, self.stats.defender1_actions, self.stats.attacker2_actions,
            self.stats.defender2_actions, self.stats.attacker3_actions, self.stats.cache_hits_a2,
            self.stats.cache_hits_d2, self.stats.cache_hits_a3, self.cache_entries,
            self.stats.quotient_classes_built, self.stats.quotient_actions_skipped,
            self.stats.d1_illegal_pairs_pruned, self.stats.d1_singleton_actions,
            self.stats.residual_edges_pruned,
            self.stats.shortcut_d1_defender_completion, self.stats.shortcut_d1_attacker_fork,
            self.stats.shortcut_d2_defender_completion, self.stats.shortcut_d2_attacker_fork,
            self.stats.shortcut_d1_noncover_replies, self.stats.shortcut_d2_noncover_replies,
            self.stats.shortcut_a2_immediate_completion, self.stats.shortcut_a2_noncover_actions,
            self.stats.shortcut_a2_defender_uncoverable, self.stats.a2_cover_actions_generated,
            self.stats.d2_cover_actions_generated,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub default_timeout_ms: Option<u64>,
    pub max_cache_entries: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self { default_timeout_ms: None, max_cache_entries: 2_000_000 }
    }
}

#[derive(Debug)]
struct TimedOut;

impl fmt::Display for TimedOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "deadline reached") }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    First,
    Defender1,
    Attacker2,
    Defender2,
    Attacker3,
}

#[derive(Clone, Debug)]
struct ClassRep {
    first: CellId,
    second: CellId,
    score: u32,
}

/// Exact node-local incidence partition.  Multiplicity is represented by the
/// optional second member.  Only those two representatives can occur in a
/// normalized pair.
#[derive(Clone, Debug)]
struct ClassPartition {
    classes: Vec<ClassRep>,
    owner: Vec<usize>,
    physical: usize,
    ranked: Vec<usize>,
}

impl ClassPartition {
    fn build(n: usize, own: &[Residual], other: &[Residual]) -> Self {
        // A signature is the complete tagged residual-window incidence.  Tags
        // for the two players occupy disjoint integer ranges.
        let mut signatures: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        let mut scores = vec![0_u32; n];
        for (tag, edge) in own.iter().enumerate() {
            for &cell in edge.cells() {
                signatures[usize::from(cell)].push(tag as u32);
                scores[usize::from(cell)] += 1;
            }
        }
        for (tag, edge) in other.iter().enumerate() {
            for &cell in edge.cells() {
                signatures[usize::from(cell)].push(0x8000_0000 | tag as u32);
                scores[usize::from(cell)] += 2;
            }
        }
        let mut grouped: HashMap<Vec<u32>, Vec<CellId>> = HashMap::new();
        for i in 0..n {
            if signatures[i].is_empty() {
                continue;
            }
            grouped.entry(std::mem::take(&mut signatures[i])).or_default().push(i as CellId);
        }
        let mut groups: Vec<Vec<CellId>> = grouped.into_values().collect();
        groups.sort_unstable_by_key(|cells| cells[0]);
        let mut owner = vec![usize::MAX; n];
        let mut classes = Vec::with_capacity(groups.len());
        let mut physical = 0;
        for cells in groups {
            let ci = classes.len();
            for &cell in &cells {
                owner[usize::from(cell)] = ci;
            }
            physical += cells.len();
            classes.push(ClassRep {
                first: cells[0],
                second: cells.get(1).copied().unwrap_or(NONE),
                score: scores[usize::from(cells[0])],
            });
        }
        let mut ranked: Vec<usize> = (0..classes.len()).collect();
        ranked.sort_unstable_by(|&a, &b| classes[b].score.cmp(&classes[a].score).then_with(|| classes[a].first.cmp(&classes[b].first)));
        Self { classes, owner, physical, ranked }
    }

    fn normalize_required(&self, cells: &[CellId]) -> Option<Action> {
        match cells.len() {
            0 => None,
            1 => {
                let ci = self.owner[usize::from(cells[0])];
                if ci == usize::MAX { return None; }
                let first = self.classes[ci].first;
                if self.physical == 1 { return Some(Action::one(first)); }
                for &mate_ci in &self.ranked {
                    if mate_ci != ci {
                        return Some(Action::pair(first, self.classes[mate_ci].first));
                    }
                    if self.classes[ci].second != NONE {
                        return Some(Action::pair(first, self.classes[ci].second));
                    }
                }
                None
            }
            2 => {
                let ca = self.owner[usize::from(cells[0])];
                let cb = self.owner[usize::from(cells[1])];
                if ca == usize::MAX || cb == usize::MAX { return None; }
                if ca == cb {
                    (self.classes[ca].second != NONE).then(|| Action::pair(self.classes[ca].first, self.classes[ca].second))
                } else {
                    Some(Action::pair(self.classes[ca].first, self.classes[cb].first))
                }
            }
            _ => None,
        }
    }

    fn normalize_action(&self, action: Action) -> Option<Action> {
        self.normalize_required(&action.as_vec())
    }
}

/// Tactical prefix plus an exhaustive stream of unordered incidence-class
/// pairs.  Prefix duplicates are removed; the fallback guarantees that every
/// quotient action is eventually returned.
struct PairIter {
    partition: ClassPartition,
    prefix: Vec<Action>,
    prefix_at: usize,
    seen: HashSet<u32>,
    i: usize,
    j: usize,
    special_done: bool,
    skipped: u64,
}

impl PairIter {
    fn new(n: usize, own: &[Residual], other: &[Residual], after_pair_quota: Option<usize>) -> Self {
        let partition = ClassPartition::build(n, own, other);
        let mut prefix = Vec::new();
        let mut seen = HashSet::new();
        let mut push = |action: Action| {
            if seen.insert(action.code()) { prefix.push(action); }
        };
        if partition.physical == 0 {
            push(Action::EMPTY);
        } else if partition.physical == 1 {
            push(Action::one(partition.classes[0].first));
        } else {
            // Current-player completions are exact ordering hints.
            for edge in own {
                if edge.cells().len() <= 2 {
                    if let Some(action) = partition.normalize_required(edge.cells()) { push(action); }
                }
            }
            // Threshold synergy: for r=k+2, neither selected cell alone leaves
            // a live own residual, while selecting both does.  Try every exact
            // quotient pair within such an edge before cellwise scoring.  This
            // changes only order; the exhaustive fallback is untouched.
            if let Some(k) = after_pair_quota {
                for edge in own.iter().filter(|edge| edge.cells().len() == k + 2) {
                    for i in 0..edge.cells().len() {
                        for j in i + 1..edge.cells().len() {
                            if let Some(action) = partition.normalize_required(&[edge.cells()[i], edge.cells()[j]]) {
                                push(action);
                            }
                        }
                    }
                }
            }
            // Same-class and cross-class combinations among the twelve most
            // incident classes reproduce the exactness-neutral R3 prefix.
            let top = &partition.ranked[..partition.ranked.len().min(12)];
            for (at, &ci) in top.iter().enumerate() {
                let class = &partition.classes[ci];
                if class.second != NONE { push(Action::pair(class.first, class.second)); }
                for &cj in &top[at + 1..] {
                    push(Action::pair(class.first, partition.classes[cj].first));
                }
            }
        }
        Self { partition, prefix, prefix_at: 0, seen, i: 0, j: 0, special_done: false, skipped: 0 }
    }
}

impl Iterator for PairIter {
    type Item = Action;

    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_at < self.prefix.len() {
            let out = self.prefix[self.prefix_at];
            self.prefix_at += 1;
            return Some(out);
        }
        if self.partition.physical <= 1 {
            return None;
        }
        while self.i < self.partition.classes.len() {
            let ci = self.i;
            if !self.special_done {
                self.special_done = true;
                let class = &self.partition.classes[ci];
                if class.second != NONE {
                    let action = Action::pair(class.first, class.second);
                    if self.seen.insert(action.code()) { return Some(action); }
                    self.skipped += 1;
                }
                self.j = ci + 1;
            }
            while self.j < self.partition.classes.len() {
                let cj = self.j;
                self.j += 1;
                let action = Action::pair(self.partition.classes[ci].first, self.partition.classes[cj].first);
                if self.seen.insert(action.code()) { return Some(action); }
                self.skipped += 1;
            }
            self.i += 1;
            self.j = self.i + 1;
            self.special_done = false;
        }
        None
    }
}

#[derive(Clone, Debug)]
struct PhysicalCell {
    id: CellId,
    q: i32,
    r: i32,
    current_legal: bool,
    score: u32,
}

/// D1 is the only post-root turn whose retained interaction carrier can still
/// contain radius-eight fringe cells (root distance nine or ten).  Its action
/// quotient therefore remains physical: geometric legality is a pair relation
/// and is not implied by window incidence.
struct D1PairIter {
    cells: Vec<PhysicalCell>,
    owner: Vec<usize>,
    class_owner: Vec<usize>,
    ranked: Vec<usize>,
    prefix: Vec<Action>,
    prefix_at: usize,
    seen: HashSet<u32>,
    seen_pair_classes: HashSet<u64>,
    seen_single_classes: HashSet<usize>,
    i: usize,
    j: usize,
    singleton_at: usize,
    pairs_done: bool,
    empty_done: bool,
    illegal_pairs: u64,
    singleton_actions: u64,
    quotient_skipped: u64,
}

impl D1PairIter {
    fn axial_distance(a: &PhysicalCell, b: &PhysicalCell) -> i32 {
        let dq = a.q - b.q;
        let dr = a.r - b.r;
        dq.abs().max(dr.abs()).max((dq + dr).abs())
    }

    fn legal_pair_cells(a: &PhysicalCell, b: &PhysicalCell) -> bool {
        (a.current_legal && (b.current_legal || Self::axial_distance(a, b) <= 8))
            || (b.current_legal && (a.current_legal || Self::axial_distance(a, b) <= 8))
    }

    fn legal_pair_at(&self, a: usize, b: usize) -> bool {
        Self::legal_pair_cells(&self.cells[a], &self.cells[b])
    }

    fn pair_class_key(&self, action: Action) -> u64 {
        let a = self.class_owner[usize::from(action.a)];
        let b = self.class_owner[usize::from(action.b)];
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        ((lo as u64) << 32) | hi as u64
    }

    fn push_prefix(&mut self, action: Action) {
        if action.b != NONE {
            let ai = self.owner[usize::from(action.a)];
            let bi = self.owner[usize::from(action.b)];
            debug_assert!(ai != usize::MAX && bi != usize::MAX && self.legal_pair_at(ai, bi));
            let key = self.pair_class_key(action);
            if !self.seen_pair_classes.insert(key) {
                self.quotient_skipped += 1;
                return;
            }
        } else if action.a != NONE {
            let class = self.class_owner[usize::from(action.a)];
            if !self.seen_single_classes.insert(class) {
                self.quotient_skipped += 1;
                return;
            }
        }
        if self.seen.insert(action.code()) { self.prefix.push(action); }
    }

    fn new(model_cells: &[Cell], a1: Action, own: &[Residual], other: &[Residual]) -> Self {
        let n = model_cells.len();
        let mut active = vec![false; n];
        let mut scores = vec![0_u32; n];
        for edge in own {
            for &cell in edge.cells() {
                active[usize::from(cell)] = true;
                scores[usize::from(cell)] += 1;
            }
        }
        for edge in other {
            for &cell in edge.cells() {
                active[usize::from(cell)] = true;
                scores[usize::from(cell)] += 2;
            }
        }
        let a1_cells = a1.as_vec();
        let distance = |x: &Cell, y: &Cell| {
            let dq = x.q - y.q;
            let dr = x.r - y.r;
            dq.abs().max(dr.abs()).max((dq + dr).abs())
        };
        let mut cells = Vec::new();
        let mut owner = vec![usize::MAX; n];
        for i in 0..n {
            if !active[i] { continue; }
            let current_legal = model_cells[i].root_legal || a1_cells.iter().any(|&placed| distance(&model_cells[i], &model_cells[usize::from(placed)]) <= 8);
            owner[i] = cells.len();
            cells.push(PhysicalCell { id: i as CellId, q: model_cells[i].q, r: model_cells[i].r, current_legal, score: scores[i] });
        }
        let mut ranked: Vec<usize> = (0..cells.len()).collect();
        ranked.sort_unstable_by(|&a, &b| cells[b].score.cmp(&cells[a].score).then_with(|| cells[a].id.cmp(&cells[b].id)));
        let partition = ClassPartition::build(n, own, other);
        let mut out = Self {
            pairs_done: cells.len() < 2, cells, owner, class_owner: partition.owner, ranked,
            prefix: vec![], prefix_at: 0, seen: HashSet::new(),
            seen_pair_classes: HashSet::new(), seen_single_classes: HashSet::new(), i: 0, j: 1,
            singleton_at: 0, empty_done: false, illegal_pairs: 0, singleton_actions: 0,
            quotient_skipped: 0,
        };
        if out.cells.is_empty() {
            out.push_prefix(Action::EMPTY);
            out.empty_done = true;
            return out;
        }
        if out.cells.len() == 1 {
            if out.cells[0].current_legal { out.push_prefix(Action::one(out.cells[0].id)); }
            out.push_prefix(Action::EMPTY);
            out.empty_done = true;
            return out;
        }

        // Exact first-placement terminals.  A root/post-A1-legal singleton
        // wins before a filler is selected.  A non-current-legal singleton is
        // paired with a legal active cell when possible.  It has no standalone
        // projected effect: such a fringe cell occurs only in a pristine near
        // defender window, whose residual five is quota-pruned after D1.
        for edge in own {
            if edge.cells().len() == 1 {
                let x = edge.cells()[0];
                let xi = out.owner[usize::from(x)];
                if out.cells[xi].current_legal {
                    out.push_prefix(Action::one(x));
                } else {
                    let legal_mate = out.ranked.iter().copied().find(|&yi| yi != xi && out.legal_pair_at(xi, yi));
                    if let Some(yi) = legal_mate {
                        out.push_prefix(Action::pair(x, out.cells[yi].id));
                    }
                }
            } else if edge.cells().len() == 2 {
                let x = edge.cells()[0];
                let y = edge.cells()[1];
                let xi = out.owner[usize::from(x)];
                let yi = out.owner[usize::from(y)];
                if out.legal_pair_at(xi, yi) { out.push_prefix(Action::pair(x, y)); }
            }
        }
        let top: Vec<usize> = out.ranked.iter().copied().take(12).collect();
        for i in 0..top.len() {
            for j in i + 1..top.len() {
                if out.legal_pair_at(top[i], top[j]) {
                    out.push_prefix(Action::pair(out.cells[top[i]].id, out.cells[top[j]].id));
                }
            }
        }
        out
    }
}

impl Iterator for D1PairIter {
    type Item = Action;

    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_at < self.prefix.len() {
            let action = self.prefix[self.prefix_at];
            self.prefix_at += 1;
            if action.len() == 1 { self.singleton_actions += 1; }
            return Some(action);
        }
        if !self.pairs_done {
            while self.i < self.cells.len() {
                while self.j < self.cells.len() {
                    let ai = self.i;
                    let bi = self.j;
                    self.j += 1;
                    if !self.legal_pair_at(ai, bi) {
                        self.illegal_pairs += 1;
                        continue;
                    }
                    let action = Action::pair(self.cells[ai].id, self.cells[bi].id);
                    let key = self.pair_class_key(action);
                    if !self.seen_pair_classes.insert(key) {
                        self.quotient_skipped += 1;
                        continue;
                    }
                    if self.seen.insert(action.code()) { return Some(action); }
                }
                self.i += 1;
                self.j = self.i + 1;
            }
            self.pairs_done = true;
        }
        // One relevant placement plus an inert filler is a distinct projected
        // action.  Only L1-legal cells have such an effect.  (An initially
        // illegal fringe singleton is quota-equivalent to EMPTY; see above.)
        while self.singleton_at < self.cells.len() {
            let at = self.singleton_at;
            self.singleton_at += 1;
            if !self.cells[at].current_legal { continue; }
            let action = Action::one(self.cells[at].id);
            let class = self.class_owner[usize::from(action.a)];
            if !self.seen_single_classes.insert(class) {
                self.quotient_skipped += 1;
                continue;
            }
            if self.seen.insert(action.code()) {
                self.singleton_actions += 1;
                return Some(action);
            }
        }
        if !self.empty_done {
            self.empty_done = true;
            if self.seen.insert(Action::EMPTY.code()) { return Some(Action::EMPTY); }
        }
        None
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StateKey {
    a1: u32,
    a: [CellId; 4],
    d: [CellId; 4],
    an: u8,
    dn: u8,
}

impl StateKey {
    fn new(a1: Action, pos: &Position) -> Self {
        let mut a = [NONE; 4];
        let mut d = [NONE; 4];
        let mut av = pos.a_cells.clone();
        let mut dv = pos.d_cells.clone();
        av.sort_unstable();
        dv.sort_unstable();
        debug_assert!(av.len() <= a.len() && dv.len() <= d.len());
        a[..av.len()].copy_from_slice(&av);
        d[..dv.len()].copy_from_slice(&dv);
        Self { a1: a1.code(), a, d, an: av.len() as u8, dn: dv.len() as u8 }
    }
}

struct Kernel {
    model: Model,
    target: Vec<Edge>,
    opponent: Vec<Edge>,
    target_anchored_len: usize,
    deadline: Option<Instant>,
    max_cache_entries: usize,
    stats: Stats,
    cache_a2: HashMap<StateKey, bool>,
    cache_d2: HashMap<StateKey, bool>,
    cache_a3: HashMap<StateKey, bool>,
    after_a1: usize,
    after_a2: usize,
    final_capacity: usize,
}

impl Kernel {
    fn new(model: Model, config: Config, started: Instant) -> Self {
        let timeout = model.timeout_ms.or(config.default_timeout_ms);
        let deadline = timeout.map(|ms| started + Duration::from_millis(ms));
        let target_anchored_len = model.target_anchored.len();
        let mut target = model.target_anchored.clone();
        target.extend(model.near.iter().cloned());
        let mut opponent = model.opponent_anchored.clone();
        opponent.extend(model.near.iter().cloned());
        let fresh_h13 = model.phase == Phase::First && model.horizon == 13;
        Self {
            model, target, opponent, target_anchored_len, deadline,
            max_cache_entries: config.max_cache_entries, stats: Stats::default(),
            cache_a2: HashMap::new(), cache_d2: HashMap::new(), cache_a3: HashMap::new(),
            after_a1: if fresh_h13 { 5 } else { 6 },
            after_a2: if fresh_h13 { 3 } else { 4 },
            final_capacity: if fresh_h13 { 1 } else { 2 },
        }
    }

    fn cache_entries(&self) -> usize { self.cache_a2.len() + self.cache_d2.len() + self.cache_a3.len() }

    #[inline]
    fn check_deadline(&self) -> Result<(), TimedOut> {
        if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) { Err(TimedOut) } else { Ok(()) }
    }

    #[inline]
    fn tick(&mut self, stage: Stage) -> Result<(), TimedOut> {
        self.stats.nodes += 1;
        match stage {
            Stage::First => self.stats.first_actions += 1,
            Stage::Defender1 => self.stats.defender1_actions += 1,
            Stage::Attacker2 => self.stats.attacker2_actions += 1,
            Stage::Defender2 => self.stats.defender2_actions += 1,
            Stage::Attacker3 => self.stats.attacker3_actions += 1,
        }
        if self.stats.nodes == 1 || self.stats.nodes & 1023 == 0 { self.check_deadline()?; }
        Ok(())
    }

    fn target_edge(&self, i: usize) -> &Edge { &self.target[i] }

    fn residual(edge: &Edge, own: &BitSet, blocked_by: &BitSet, max_len: Option<usize>) -> Option<Residual> {
        if edge.cells().iter().any(|&cell| blocked_by.contains(cell)) { return None; }
        let mut cells = [NONE; 6];
        let mut len = 0;
        for &cell in edge.cells() {
            if !own.contains(cell) {
                cells[len] = cell;
                len += 1;
            }
        }
        if max_len.is_some_and(|limit| len > limit) { return None; }
        Some(Residual { cells, len: len as u8 })
    }

    /// Residuals form a monotone terminal/blocking family.  If E is a strict
    /// subset of F for the same owner, completing F already completes E and
    /// every opponent placement killing E also kills F.  F can never be the
    /// unique first terminal and is exactly dominated.  Equal residuals are
    /// likewise interchangeable.
    fn residual_antichain(family: Vec<Residual>) -> Vec<Residual> {
        let mut seen = HashSet::new();
        let mut unique = Vec::with_capacity(family.len());
        for edge in family {
            if seen.insert(edge.clone()) { unique.push(edge); }
        }
        if unique.len() <= 1 { return unique; }
        let keys: HashSet<Residual> = unique.iter().cloned().collect();
        unique.retain(|edge| {
            let n = edge.cells().len();
            if n == 0 { return true; }
            // Six-cell windows make exhaustive proper-subset lookup tiny
            // (at most 62 keys), avoiding a quadratic family comparison.
            let full = (1_usize << n) - 1;
            for choice in 1..full {
                let mut cells = [NONE; 6];
                let mut len = 0;
                for (i, &cell) in edge.cells().iter().enumerate() {
                    if choice & (1 << i) != 0 {
                        cells[len] = cell;
                        len += 1;
                    }
                }
                if keys.contains(&Residual { cells, len: len as u8 }) { return false; }
            }
            true
        });
        unique
    }

    fn target_family(&mut self, ids: &[usize], pos: &Position, max_len: Option<usize>) -> Vec<Residual> {
        let raw: Vec<Residual> = ids.iter().filter_map(|&i| Self::residual(self.target_edge(i), &pos.a, &pos.d, max_len)).filter(|x| x.len != 0).collect();
        let before = raw.len();
        let out = Self::residual_antichain(raw);
        self.stats.residual_edges_pruned += (before - out.len()) as u64;
        out
    }

    fn opponent_family(&mut self, pos: &Position, max_len: Option<usize>) -> Vec<Residual> {
        let raw: Vec<Residual> = self.opponent.iter().filter_map(|edge| Self::residual(edge, &pos.d, &pos.a, max_len)).filter(|x| x.len != 0).collect();
        let before = raw.len();
        let out = Self::residual_antichain(raw);
        self.stats.residual_edges_pruned += (before - out.len()) as u64;
        out
    }

    fn any_target_complete(&self, ids: &[usize], pos: &Position) -> bool {
        ids.iter().any(|&i| Self::residual(self.target_edge(i), &pos.a, &pos.d, None).is_some_and(|x| x.len == 0))
    }

    fn any_opponent_complete(&self, pos: &Position) -> bool {
        self.opponent.iter().any(|edge| Self::residual(edge, &pos.d, &pos.a, None).is_some_and(|x| x.len == 0))
    }

    fn pair_iter(&mut self, own: &[Residual], other: &[Residual], after_pair_quota: Option<usize>) -> PairIter {
        let iter = PairIter::new(self.model.cells.len(), own, other, after_pair_quota);
        self.stats.quotient_classes_built += iter.partition.classes.len() as u64;
        iter
    }

    fn hex_distance(&self, a: CellId, b: CellId) -> i32 {
        let a = &self.model.cells[usize::from(a)];
        let b = &self.model.cells[usize::from(b)];
        let dq = a.q - b.q;
        let dr = a.r - b.r;
        dq.abs().max(dr.abs()).max((dq + dr).abs())
    }

    fn generate_first_actions(&mut self) -> Result<Vec<Action>, TimedOut> {
        self.check_deadline()?;
        let n = self.model.cells.len();
        let mut seen = HashSet::new();
        match self.model.phase {
            Phase::Second => {
                for i in 0..n {
                    if self.model.cells[i].root_legal { seen.insert(Action::one(i as CellId).code()); }
                }
            }
            Phase::First => {
                for ai in 0..n {
                    if !self.model.cells[ai].anchored { continue; }
                    for bi in 0..n {
                        if ai == bi { continue; }
                        if self.model.cells[bi].root_legal || self.hex_distance(ai as CellId, bi as CellId) <= 8 {
                            seen.insert(Action::pair(ai as CellId, bi as CellId).code());
                        }
                    }
                    if ai & 31 == 0 { self.check_deadline()?; }
                }
                for edge in &self.model.near {
                    let cells = edge.cells();
                    for i in 0..cells.len() {
                        for j in i + 1..cells.len() {
                            if self.model.cells[usize::from(cells[i])].root_legal || self.model.cells[usize::from(cells[j])].root_legal {
                                seen.insert(Action::pair(cells[i], cells[j]).code());
                            }
                        }
                    }
                }
            }
        }
        if seen.is_empty() { seen.insert(Action::EMPTY.code()); }
        let mut actions: Vec<Action> = seen.into_iter().map(|code| {
            let a = (code >> 16) as CellId;
            let b = (code & 0xffff) as CellId;
            Action { a, b }
        }).collect();
        let mut score = vec![0_u32; n];
        for edge in &self.target { for &cell in edge.cells() { score[usize::from(cell)] += 2; } }
        for edge in &self.opponent { for &cell in edge.cells() { score[usize::from(cell)] += 1; } }
        let action_score = |action: Action| {
            let mut value = 0;
            action.each(|cell| { if cell != NONE { value += score[usize::from(cell)]; } });
            value
        };
        actions.sort_unstable_by(|&a, &b| action_score(b).cmp(&action_score(a)).then_with(|| a.code().cmp(&b.code())));
        if let Some(required) = self.model.preferred_required {
            // A verifier-accepted root Choice certifies only this physical
            // first placement.  Move the complete block of containing pairs
            // forward while retaining exhaustive score order within/beyond it.
            let mut without = Vec::with_capacity(actions.len());
            let mut with = Vec::new();
            for action in actions {
                if action.a == required || action.b == required { with.push(action); } else { without.push(action); }
            }
            with.extend(without);
            actions = with;
        }
        if let Some(preferred) = self.model.preferred {
            if let Some(at) = actions.iter().position(|&action| action == preferred) {
                let action = actions.remove(at);
                if self.model.preferred_required.map_or(true, |required| action.a == required || action.b == required) {
                    actions.insert(0, action);
                } else {
                    // A non-certified full-pair hint must not jump ahead of a
                    // verifier-certified required-cell block.
                    actions.push(action);
                }
            }
        }
        self.stats.first_action_total = actions.len();
        Ok(actions)
    }

    fn eligible_after_a1(&self, a1: Action, pos: &Position) -> Vec<usize> {
        let mut out = Vec::new();
        for i in 0..self.target.len() {
            let Some(residual) = Self::residual(&self.target[i], &pos.a, &pos.d, Some(self.after_a1)) else { continue; };
            if residual.len == 0 { continue; }
            if i < self.target_anchored_len
                || self.target[i].cells().iter().any(|&cell| cell == a1.a || cell == a1.b)
            {
                out.push(i);
            }
        }
        out
    }

    fn insert_cache(map: &mut HashMap<StateKey, bool>, key: StateKey, value: bool, current_total: usize, limit: usize) {
        if current_total < limit { map.insert(key, value); }
    }

    fn solve_root(&mut self) -> Result<Status, TimedOut> {
        self.check_deadline()?;
        let first_actions = self.generate_first_actions()?;
        self.check_deadline()?;
        let all_target: Vec<usize> = (0..self.target.len()).collect();
        let mut pos = Position::new(self.model.cells.len());
        for a1 in first_actions {
            self.tick(Stage::First)?;
            let before = pos.play(Player::Attacker, a1);
            if self.any_target_complete(&all_target, &pos) {
                pos.undo(Player::Attacker, before);
                self.check_deadline()?;
                return Ok(Status::Win { witness: a1 });
            }
            let eligible = self.eligible_after_a1(a1, &pos);
            let wins = self.solve_d1(a1, &eligible, &mut pos)?;
            pos.undo(Player::Attacker, before);
            if wins {
                self.check_deadline()?;
                return Ok(Status::Win { witness: a1 });
            }
        }
        self.check_deadline()?;
        Ok(Status::Negative)
    }

    /// After A1: every normalized D1 pair must admit an A2 answer.
    fn solve_d1(&mut self, a1: Action, target_ids: &[usize], pos: &mut Position) -> Result<bool, TimedOut> {
        let target = self.target_family(target_ids, pos, Some(self.after_a1));
        let defender = self.opponent_family(pos, None);
        // D completes a one/two-cell residual on this pair.  Otherwise, if
        // A's immediate residuals have no two-cover, every possible D pair
        // leaves an A2 completion.  These are quantified node identities, not
        // ordering heuristics; D1 fringe legality can only remove covers.
        if defender.iter().any(|edge| edge.cells().len() <= 2) {
            self.stats.shortcut_d1_defender_completion += 1;
            return Ok(false);
        }
        let immediate_a: Vec<Residual> = target.iter().filter(|edge| edge.cells().len() <= 2).cloned().collect();
        if !immediate_a.is_empty() && !Self::has_two_cover(&immediate_a) {
            self.stats.shortcut_d1_attacker_fork += 1;
            return Ok(true);
        }
        let mut actions = D1PairIter::new(&self.model.cells, a1, &defender, &target);
        while let Some(d1) = actions.next() {
            self.tick(Stage::Defender1)?;
            if !immediate_a.is_empty() && immediate_a.iter().any(|edge| !Self::action_hits(d1, edge)) {
                // This D1 pair leaves an A residual fillable immediately on
                // A2.  Since D has no current completion (checked above), the
                // universal branch is already answered without building A2.
                self.stats.shortcut_d1_noncover_replies += 1;
                continue;
            }
            let before = pos.play(Player::Defender, d1);
            let answered = if self.any_opponent_complete(pos) { false } else { self.solve_a2(a1, target_ids, pos)? };
            pos.undo(Player::Defender, before);
            if !answered {
                self.stats.d1_illegal_pairs_pruned += actions.illegal_pairs;
                self.stats.d1_singleton_actions += actions.singleton_actions;
                self.stats.quotient_actions_skipped += actions.quotient_skipped;
                return Ok(false);
            }
        }
        self.stats.d1_illegal_pairs_pruned += actions.illegal_pairs;
        self.stats.d1_singleton_actions += actions.singleton_actions;
        self.stats.quotient_actions_skipped += actions.quotient_skipped;
        Ok(true)
    }

    fn a2_action_wins(&mut self, a1: Action, target_ids: &[usize], pos: &mut Position, a2: Action) -> Result<bool, TimedOut> {
        self.tick(Stage::Attacker2)?;
        let before = pos.play(Player::Attacker, a2);
        let wins = self.any_target_complete(target_ids, pos) || self.solve_d2(a1, target_ids, pos)?;
        pos.undo(Player::Attacker, before);
        Ok(wins)
    }

    /// After D1: some normalized A2 pair must survive every D2 reply.
    fn solve_a2(&mut self, a1: Action, target_ids: &[usize], pos: &mut Position) -> Result<bool, TimedOut> {
        let key = StateKey::new(a1, pos);
        if let Some(value) = self.cache_a2.get(&key).copied() {
            self.stats.cache_hits_a2 += 1;
            return Ok(value);
        }
        let target = self.target_family(target_ids, pos, Some(self.after_a1));
        let defender = self.opponent_family(pos, Some(4));
        if target.iter().any(|edge| edge.cells().len() <= 2) {
            // A2 fills this residual now, before D2 can use any threat.
            self.stats.shortcut_a2_immediate_completion += 1;
            let total = self.cache_entries();
            Self::insert_cache(&mut self.cache_a2, key, true, total, self.max_cache_entries);
            return Ok(true);
        }
        let immediate_d: Vec<Residual> = defender.iter().filter(|edge| edge.cells().len() <= 2).cloned().collect();
        let mut result = false;
        if !immediate_d.is_empty() {
            if !Self::has_two_cover(&immediate_d) {
                self.stats.shortcut_a2_defender_uncoverable += 1;
            } else {
                let partition = ClassPartition::build(self.model.cells.len(), &target, &defender);
                self.stats.quotient_classes_built += partition.classes.len() as u64;
                let mut actions = self.covering_actions(&immediate_d, &partition);
                let score = |action: Action| {
                    target.iter().map(|edge| edge.cells().iter().filter(|&&x| x == action.a || x == action.b).count()).sum::<usize>()
                        + 2 * defender.iter().filter(|edge| Self::action_hits(action, edge)).count()
                };
                actions.sort_unstable_by(|&a, &b| score(b).cmp(&score(a)).then_with(|| a.code().cmp(&b.code())));
                self.stats.a2_cover_actions_generated += actions.len() as u64;
                for a2 in actions {
                    if self.a2_action_wins(a1, target_ids, pos, a2)? { result = true; break; }
                }
            }
        } else {
            let mut actions = self.pair_iter(&target, &defender, Some(self.after_a2));
            while let Some(a2) = actions.next() {
                if self.a2_action_wins(a1, target_ids, pos, a2)? { result = true; break; }
            }
            self.stats.quotient_actions_skipped += actions.skipped;
        }
        let total = self.cache_entries();
        Self::insert_cache(&mut self.cache_a2, key, result, total, self.max_cache_entries);
        Ok(result)
    }

    fn d2_action_survives(&mut self, a1: Action, target_ids: &[usize], pos: &mut Position, d2: Action) -> Result<bool, TimedOut> {
        self.tick(Stage::Defender2)?;
        let before = pos.play(Player::Defender, d2);
        let survives = if self.any_opponent_complete(pos) { false } else { self.solve_a3(a1, target_ids, pos)? };
        pos.undo(Player::Defender, before);
        Ok(survives)
    }

    /// After A2: every normalized D2 pair must admit a final A3 fork.
    fn solve_d2(&mut self, a1: Action, target_ids: &[usize], pos: &mut Position) -> Result<bool, TimedOut> {
        let key = StateKey::new(a1, pos);
        if let Some(value) = self.cache_d2.get(&key).copied() {
            self.stats.cache_hits_d2 += 1;
            return Ok(value);
        }
        let target = self.target_family(target_ids, pos, Some(self.after_a2));
        let defender = self.opponent_family(pos, Some(4));
        if defender.iter().any(|edge| edge.cells().len() <= 2) {
            self.stats.shortcut_d2_defender_completion += 1;
            let total = self.cache_entries();
            Self::insert_cache(&mut self.cache_d2, key, false, total, self.max_cache_entries);
            return Ok(false);
        }
        let immediate_a: Vec<Residual> = target.iter().filter(|edge| edge.cells().len() <= 2).cloned().collect();
        if !immediate_a.is_empty() && !Self::has_two_cover(&immediate_a) {
            self.stats.shortcut_d2_attacker_fork += 1;
            let total = self.cache_entries();
            Self::insert_cache(&mut self.cache_d2, key, true, total, self.max_cache_entries);
            return Ok(true);
        }
        let mut result = true;
        if !immediate_a.is_empty() {
            let partition = ClassPartition::build(self.model.cells.len(), &defender, &target);
            self.stats.quotient_classes_built += partition.classes.len() as u64;
            let mut actions = self.covering_actions(&immediate_a, &partition);
            let score = |action: Action| {
                defender.iter().map(|edge| edge.cells().iter().filter(|&&x| x == action.a || x == action.b).count()).sum::<usize>()
                    + 2 * target.iter().filter(|edge| Self::action_hits(action, edge)).count()
            };
            actions.sort_unstable_by(|&a, &b| score(b).cmp(&score(a)).then_with(|| a.code().cmp(&b.code())));
            self.stats.d2_cover_actions_generated += actions.len() as u64;
            for d2 in actions {
                if !self.d2_action_survives(a1, target_ids, pos, d2)? { result = false; break; }
            }
        } else {
            let mut actions = self.pair_iter(&defender, &target, Some(2));
            while let Some(d2) = actions.next() {
                if !self.d2_action_survives(a1, target_ids, pos, d2)? { result = false; break; }
            }
            self.stats.quotient_actions_skipped += actions.skipped;
        }
        let total = self.cache_entries();
        Self::insert_cache(&mut self.cache_d2, key, result, total, self.max_cache_entries);
        Ok(result)
    }

    fn action_hits(action: Action, edge: &Residual) -> bool {
        edge.cells().iter().any(|&cell| cell == action.a || cell == action.b)
    }

    fn residual_after_action(edge: &Residual, action: Action) -> Residual {
        let mut cells = [NONE; 6];
        let mut len = 0;
        for &cell in edge.cells() {
            if cell != action.a && cell != action.b {
                cells[len] = cell;
                len += 1;
            }
        }
        Residual { cells, len: len as u8 }
    }

    fn pair_from_cells(a: CellId, b: CellId, physical: usize) -> Option<Action> {
        if a != b { Some(Action::pair(a, b)) } else if physical == 1 { Some(Action::one(a)) } else { None }
    }

    /// Enumerate every pair hitting all one/two-cell defender threats.  The
    /// common-cell case intentionally varies the mate because that placement
    /// can create a distinct attacker fork.
    fn covering_actions(&self, threats: &[Residual], partition: &ClassPartition) -> Vec<Action> {
        let mut out = HashSet::new();
        if threats.is_empty() { return Vec::new(); }
        let first = &threats[0];
        for &x in first.cells() {
            let rest: Vec<&Residual> = threats.iter().filter(|edge| !edge.cells().contains(&x)).collect();
            if rest.is_empty() {
                for class in &partition.classes {
                    if class.first != x {
                        if let Some(action) = Self::pair_from_cells(x, class.first, partition.physical) {
                            if let Some(norm) = partition.normalize_action(action) { out.insert(norm.code()); }
                        }
                    }
                    if class.second != NONE && class.second != x {
                        if let Some(norm) = partition.normalize_action(Action::pair(x, class.second)) { out.insert(norm.code()); }
                    }
                }
                if partition.physical == 1 {
                    if let Some(norm) = partition.normalize_action(Action::one(x)) { out.insert(norm.code()); }
                }
                continue;
            }
            let mut common: Vec<CellId> = rest[0].cells().to_vec();
            common.retain(|cell| rest[1..].iter().all(|edge| edge.cells().contains(cell)));
            for y in common {
                if let Some(action) = Self::pair_from_cells(x, y, partition.physical) {
                    if let Some(norm) = partition.normalize_action(action) { out.insert(norm.code()); }
                }
            }
        }
        out.into_iter().map(|code| Action { a: (code >> 16) as CellId, b: (code & 0xffff) as CellId }).collect()
    }

    fn threat_making_actions(&self, target: &[Residual], partition: &ClassPartition) -> Vec<Action> {
        let mut out = HashSet::new();
        if self.final_capacity == 1 {
            // With no immediate completion, only a size-three residual can be
            // reduced to a singleton, and both selected cells must lie in it.
            for edge in target.iter().filter(|edge| edge.cells().len() == 3) {
                for i in 0..3 {
                    for j in i + 1..3 {
                        if let Some(norm) = partition.normalize_action(Action::pair(edge.cells()[i], edge.cells()[j])) { out.insert(norm.code()); }
                    }
                }
            }
        } else {
            // A size-three residual needs either selected cell; its mate can
            // affect a second line.  A size-four residual needs both cells.
            let low: HashSet<CellId> = target.iter().filter(|edge| edge.cells().len() == 3).flat_map(|edge| edge.cells().iter().copied()).collect();
            for x in low {
                for class in &partition.classes {
                    if class.first != x {
                        if let Some(norm) = partition.normalize_action(Action::pair(x, class.first)) { out.insert(norm.code()); }
                    }
                    if class.second != NONE && class.second != x {
                        if let Some(norm) = partition.normalize_action(Action::pair(x, class.second)) { out.insert(norm.code()); }
                    }
                }
            }
            for edge in target.iter().filter(|edge| edge.cells().len() == 4) {
                for i in 0..4 {
                    for j in i + 1..4 {
                        if let Some(norm) = partition.normalize_action(Action::pair(edge.cells()[i], edge.cells()[j])) { out.insert(norm.code()); }
                    }
                }
            }
        }
        out.into_iter().map(|code| Action { a: (code >> 16) as CellId, b: (code & 0xffff) as CellId }).collect()
    }

    fn has_two_cover(family: &[Residual]) -> bool {
        if family.is_empty() { return true; }
        let mut common = family[0].cells().to_vec();
        common.retain(|cell| family[1..].iter().all(|edge| edge.cells().contains(cell)));
        if !common.is_empty() { return true; }
        for &x in family[0].cells() {
            let rest: Vec<&Residual> = family.iter().filter(|edge| !edge.cells().contains(&x)).collect();
            if rest.is_empty() { return true; }
            if rest[0].cells().iter().any(|cell| rest[1..].iter().all(|edge| edge.cells().contains(cell))) { return true; }
        }
        false
    }

    /// A3 is the third attacker pair.  It either wins immediately or leaves a
    /// one/two-cell threat family that the intervening defender pair cannot
    /// cover.  Immediate completion is checked before defender-cover pruning;
    /// this preserves first-placement termination exactly.
    fn solve_a3(&mut self, a1: Action, target_ids: &[usize], pos: &mut Position) -> Result<bool, TimedOut> {
        let key = StateKey::new(a1, pos);
        if let Some(value) = self.cache_a3.get(&key).copied() {
            self.stats.cache_hits_a3 += 1;
            return Ok(value);
        }
        let target = self.target_family(target_ids, pos, Some(self.after_a2));
        // Any residual of one or two cells is filled by A3 now.  The game ends
        // before a defender threat matters.
        if target.iter().any(|edge| edge.cells().len() <= 2) {
            let total = self.cache_entries();
            Self::insert_cache(&mut self.cache_a3, key, true, total, self.max_cache_entries);
            return Ok(true);
        }
        let defender_threats = self.opponent_family(pos, Some(2));
        let partition = ClassPartition::build(self.model.cells.len(), &target, &defender_threats);
        self.stats.quotient_classes_built += partition.classes.len() as u64;
        let mut actions = if defender_threats.is_empty() {
            self.threat_making_actions(&target, &partition)
        } else {
            self.covering_actions(&defender_threats, &partition)
        };
        let score = |action: Action| target.iter().map(|edge| edge.cells().iter().filter(|&&x| x == action.a || x == action.b).count()).sum::<usize>();
        actions.sort_unstable_by(|&a, &b| score(b).cmp(&score(a)).then_with(|| a.code().cmp(&b.code())));
        let mut result = false;
        for a3 in actions {
            self.tick(Stage::Attacker3)?;
            if defender_threats.iter().any(|edge| !Self::action_hits(a3, edge)) {
                continue;
            }
            // `target` is already the exact live antichain at this node.  Own
            // placement only subtracts cells, so every previously dominated
            // superset remains dominated after A3.  Derive the endpoint
            // family directly instead of rescanning all root windows for each
            // candidate pair.
            let threats: Vec<Residual> = target.iter()
                .map(|edge| Self::residual_after_action(edge, a3))
                .filter(|edge| edge.cells().len() <= self.final_capacity)
                .collect();
            if !threats.is_empty() {
                if self.final_capacity == 1 {
                    let distinct: HashSet<CellId> = threats.iter().filter(|edge| edge.cells().len() == 1).map(|edge| edge.cells()[0]).collect();
                    result = distinct.len() > 2;
                } else {
                    result = !Self::has_two_cover(&threats);
                }
            }
            if result { break; }
        }
        let total = self.cache_entries();
        Self::insert_cache(&mut self.cache_a3, key, result, total, self.max_cache_entries);
        Ok(result)
    }
}

pub fn decide(model: Model, config: Config) -> Decision {
    let started = Instant::now();
    let id = model.id.clone();
    let horizon = model.horizon;
    let phase = model.phase;
    let universe = model.cells.len();
    let target_windows = model.target_anchored.len() + model.near.len();
    let opponent_windows = model.opponent_anchored.len() + model.near.len();
    if let Err(message) = model.validate() {
        return Decision { id, horizon, phase, status: Status::Error(message), stats: Stats::default(), universe, target_windows, opponent_windows, cache_entries: 0, wall: started.elapsed() };
    }
    let mut kernel = Kernel::new(model, config, started);
    let status = match kernel.solve_root() {
        Ok(status) => status,
        Err(_) => Status::Timeout,
    };
    Decision {
        id, horizon, phase, status, stats: kernel.stats.clone(), universe,
        target_windows, opponent_windows, cache_entries: kernel.cache_entries(),
        wall: started.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(cells: &[CellId]) -> Edge { Edge::new(cells).unwrap() }

    fn residual(cells: &[CellId]) -> Residual {
        let mut packed = [NONE; 6];
        packed[..cells.len()].copy_from_slice(cells);
        Residual { cells: packed, len: cells.len() as u8 }
    }

    fn cell() -> Cell { Cell { q: 0, r: 0, anchored: true, root_legal: true } }

    fn model(phase: Phase, horizon: u8, n: usize, target: Vec<Edge>, opponent: Vec<Edge>) -> Model {
        Model {
            id: "test".to_string(), horizon, phase, timeout_ms: None,
            cells: (0..n).map(|_| cell()).collect(), target_anchored: target,
            opponent_anchored: opponent, near: vec![], preferred: None,
            preferred_required: None,
        }
    }

    #[test]
    fn incidence_class_retains_two_member_multiplicity() {
        // All three cells have the same complete incidence.  Exactly one
        // normalized same-class pair remains, and it uses two representatives.
        let own = vec![residual(&[0, 1, 2])];
        let actions: Vec<Action> = PairIter::new(3, &own, &[], None).collect();
        assert_eq!(actions, vec![Action::pair(0, 1)]);
        assert_eq!(actions[0].len(), 2);
    }

    #[test]
    fn immediate_a3_completion_precedes_defender_cover() {
        // A can fill {0,1} now even though its pair cannot also cover the two
        // disjoint defender singleton threats.  First-terminal semantics says
        // this is a win, with no endpoint pair enumeration required.
        let m = model(Phase::First, 14, 4, vec![edge(&[0, 1])], vec![edge(&[2]), edge(&[3])]);
        let mut kernel = Kernel::new(m, Config::default(), Instant::now());
        let mut pos = Position::new(4);
        assert!(kernel.solve_a3(Action::pair(0, 1), &[0], &mut pos).unwrap());
        assert_eq!(kernel.stats.attacker3_actions, 0);
    }

    #[test]
    fn fresh_h13_size_three_requires_two_selected_cells() {
        let m = model(Phase::First, 13, 3, vec![edge(&[0, 1, 2])], vec![]);
        let kernel = Kernel::new(m, Config::default(), Instant::now());
        assert_eq!(kernel.final_capacity, 1);
        let target = vec![residual(&[0, 1, 2])];
        let partition = ClassPartition::build(3, &target, &[]);
        let actions = kernel.threat_making_actions(&target, &partition);
        assert!(!actions.is_empty());
        assert!(actions.iter().all(|action| action.len() == 2));
        assert!(actions.iter().all(|action| action.as_vec().iter().all(|cell| [0, 1, 2].contains(cell))));
    }

    #[test]
    fn timeout_is_never_reported_as_negative() {
        let m = model(Phase::Second, 13, 1, vec![edge(&[0])], vec![]);
        let decision = decide(m, Config { default_timeout_ms: Some(0), max_cache_entries: 16 });
        assert!(matches!(decision.status, Status::Timeout));
    }

    #[test]
    fn duplicate_preferred_cell_is_rejected() {
        let mut m = model(Phase::First, 14, 2, vec![edge(&[0, 1])], vec![]);
        m.preferred = Some(Action { a: 0, b: 0 });
        assert!(m.validate().unwrap_err().contains("duplicate"));
    }

    #[test]
    fn d1_filters_geometry_and_retains_projected_actions() {
        let cells = vec![
            Cell { q: 0, r: 0, anchored: true, root_legal: true }, // A1
            Cell { q: 10, r: 0, anchored: false, root_legal: false },
            Cell { q: 11, r: 0, anchored: false, root_legal: false },
            Cell { q: 2, r: 0, anchored: true, root_legal: true },
        ];
        let own = vec![residual(&[1, 2, 3])];
        let actions: Vec<Action> = D1PairIter::new(&cells, Action::one(0), &own, &[]).collect();
        // Two non-L1 cells cannot start the pair, even when close together.
        assert!(!actions.contains(&Action::pair(1, 2)));
        // Cell 1 is reached second from root-legal cell 3 at distance eight.
        assert!(actions.contains(&Action::pair(1, 3)));
        // Cell 2 is distance nine from the only legal active cell.
        assert!(!actions.contains(&Action::pair(2, 3)));
        // Exact projection retains a legal singleton plus the all-inert move.
        assert!(actions.contains(&Action::one(3)));
        assert!(actions.contains(&Action::EMPTY));
        assert!(!actions.contains(&Action::one(1)));
        assert!(!actions.contains(&Action::one(2)));
    }

    #[test]
    fn residual_antichain_deduplicates_and_drops_strict_supersets() {
        let family = vec![
            residual(&[1]),
            residual(&[1, 2]),       // dominated by {1}
            residual(&[1, 2, 3]),    // dominated by both preceding edges
            residual(&[2, 3]),
            residual(&[1]),          // duplicate
            residual(&[2, 3, 4]),    // dominated by {2,3}
            residual(&[4, 5]),
        ];
        let reduced = Kernel::residual_antichain(family);
        assert_eq!(reduced, vec![residual(&[1]), residual(&[2, 3]), residual(&[4, 5])]);

        // Reminder of the endpoint consequence: deleting those supersets does
        // not change whether a two-cell defender cover exists.
        let original = vec![residual(&[1]), residual(&[1, 2]), residual(&[2, 3]), residual(&[2, 3, 4]), residual(&[4, 5])];
        assert_eq!(Kernel::has_two_cover(&original), Kernel::has_two_cover(&reduced));
    }

    #[test]
    fn quantified_pair_node_shortcuts_are_exact() {
        let mut d1_fork = Kernel::new(
            model(Phase::First, 14, 6, vec![edge(&[0]), edge(&[1]), edge(&[2])], vec![edge(&[3, 4, 5])]),
            Config::default(), Instant::now(),
        );
        let mut pos = Position::new(6);
        assert!(d1_fork.solve_d1(Action::EMPTY, &[0, 1, 2], &mut pos).unwrap());
        assert_eq!(d1_fork.stats.shortcut_d1_attacker_fork, 1);
        assert_eq!(d1_fork.stats.defender1_actions, 0);

        let mut d1_completion = Kernel::new(
            model(Phase::First, 14, 6, vec![edge(&[0, 1, 2])], vec![edge(&[3, 4])]),
            Config::default(), Instant::now(),
        );
        assert!(!d1_completion.solve_d1(Action::EMPTY, &[0], &mut pos).unwrap());
        assert_eq!(d1_completion.stats.shortcut_d1_defender_completion, 1);

        let mut d2_fork = Kernel::new(
            model(Phase::First, 14, 6, vec![edge(&[0]), edge(&[1]), edge(&[2])], vec![edge(&[3, 4, 5])]),
            Config::default(), Instant::now(),
        );
        assert!(d2_fork.solve_d2(Action::EMPTY, &[0, 1, 2], &mut pos).unwrap());
        assert_eq!(d2_fork.stats.shortcut_d2_attacker_fork, 1);
        assert_eq!(d2_fork.stats.defender2_actions, 0);

        let mut d2_completion = Kernel::new(
            model(Phase::First, 14, 6, vec![edge(&[0, 1, 2])], vec![edge(&[3, 4])]),
            Config::default(), Instant::now(),
        );
        assert!(!d2_completion.solve_d2(Action::EMPTY, &[0], &mut pos).unwrap());
        assert_eq!(d2_completion.stats.shortcut_d2_defender_completion, 1);
    }

    #[test]
    fn direct_h13_a3_endpoint_builds_three_uncoverable_singletons() {
        let m = model(
            Phase::First,
            13,
            5,
            vec![edge(&[0, 1, 2]), edge(&[0, 1, 3]), edge(&[0, 1, 4])],
            vec![],
        );
        let mut kernel = Kernel::new(m, Config::default(), Instant::now());
        let mut pos = Position::new(5);
        assert!(kernel.solve_a3(Action::EMPTY, &[0, 1, 2], &mut pos).unwrap());
        assert_eq!(kernel.stats.attacker3_actions, 1);
        assert!(pos.a_cells.is_empty() && pos.d_cells.is_empty());
    }

    #[test]
    fn a2_immediate_and_uncoverable_defender_shortcuts() {
        let mut immediate = Kernel::new(
            model(Phase::First, 14, 4, vec![edge(&[0, 1])], vec![edge(&[2, 3, 0])]),
            Config::default(), Instant::now(),
        );
        let mut pos = Position::new(4);
        assert!(immediate.solve_a2(Action::EMPTY, &[0], &mut pos).unwrap());
        assert_eq!(immediate.stats.shortcut_a2_immediate_completion, 1);

        let mut uncovered = Kernel::new(
            model(
                Phase::First,
                14,
                7,
                vec![edge(&[0, 1, 2])],
                vec![edge(&[3]), edge(&[4]), edge(&[5]), edge(&[6, 0, 1])],
            ),
            Config::default(), Instant::now(),
        );
        pos = Position::new(7);
        assert!(!uncovered.solve_a2(Action::EMPTY, &[0], &mut pos).unwrap());
        assert_eq!(uncovered.stats.shortcut_a2_defender_uncoverable, 1);
        assert_eq!(uncovered.stats.attacker2_actions, 0);
    }

    #[test]
    fn required_cell_hint_orders_a_complete_block_without_pruning() {
        let mut m = model(Phase::First, 14, 4, vec![], vec![]);
        for (i, cell) in m.cells.iter_mut().enumerate() {
            cell.q = i as i32;
            cell.anchored = true;
            cell.root_legal = true;
        }
        m.preferred_required = Some(2);
        let mut kernel = Kernel::new(m, Config::default(), Instant::now());
        let actions = kernel.generate_first_actions().unwrap();
        assert_eq!(actions.len(), 6);
        assert!(actions[..3].iter().all(|action| action.a == 2 || action.b == 2));
        assert!(actions[3..].iter().all(|action| action.a != 2 && action.b != 2));
    }
}
