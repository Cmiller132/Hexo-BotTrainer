//! Default-off producer for the leaf-only `RefuteLeafExact/V1` artifact.
//!
//! This is an explicit post-search API. It is not called by the production
//! solve path and cannot change ordinary solver status or certificate bytes.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use hexo_engine::{HexoState, Player, TurnPhase};

use crate::tss_refute_leaf_cert::{
    encode_artifact, root_header_from_engine, root_semantic_sha256, LeafArtifactV1, LeafCountsV1,
    ReachableRootV1,
};
use crate::tss_refute_verify::{
    verify_refute_leaf_exact_v1, OfflinePolicyV1, RefuteWorkV1, VerifiedClassRefutationV1,
    VerifyRejectionV1,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RefuteLeafModeV1 {
    #[default]
    Off,
    Emit,
}

impl RefuteLeafModeV1 {
    /// Read-once environment selector for an offline caller. Unknown values
    /// fail closed to Off.
    pub fn from_env() -> Self {
        match std::env::var("TSS_REFUTE_CERT_V1").ok().as_deref() {
            Some("emit") => Self::Emit,
            _ => Self::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchProfileV1 {
    LeafNaturalWidthExhaustExactV1 {
        ordinary_search_completed: bool,
        natural_width_exhaustion: bool,
        staged_reopens_complete: bool,
        bottom_up_refresh_complete: bool,
        semantic_horizon: u32,
        exact_v1_width_options: bool,
        header_profile: u16,
    },
    Other,
}

impl SearchProfileV1 {
    pub fn completed_natural_width_exhaust() -> Self {
        Self::LeafNaturalWidthExhaustExactV1 {
            ordinary_search_completed: true,
            natural_width_exhaustion: true,
            staged_reopens_complete: true,
            bottom_up_refresh_complete: true,
            semantic_horizon: u32::MAX,
            exact_v1_width_options: true,
            header_profile: 1,
        }
    }
    fn exact(self) -> bool {
        matches!(
            self,
            Self::LeafNaturalWidthExhaustExactV1 {
                ordinary_search_completed: true,
                natural_width_exhaustion: true,
                staged_reopens_complete: true,
                bottom_up_refresh_complete: true,
                semantic_horizon: u32::MAX,
                exact_v1_width_options: true,
                header_profile: 1
            }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeafIneligibilityV1 {
    IneligibleLeafProfile,
    IneligibleNodeCap,
    IneligibleRoot(&'static str),
    UnsupportedPolicyBudget(&'static str),
    NotRefuteLeafExactSemantic(&'static str),
    SelfVerificationRejected(VerifyRejectionV1),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProducedRefuteLeafV1 {
    pub bytes: Vec<u8>,
    pub artifact: LeafArtifactV1,
    pub verified: VerifiedClassRefutationV1,
    pub producer_work: RefuteWorkV1,
    pub producer_wall: Duration,
    pub verifier_wall: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProduceResultV1 {
    Disabled,
    Ineligible(LeafIneligibilityV1),
    Emitted(ProducedRefuteLeafV1),
}

/// Explicit, post-ordinary-search entry point. Cheap profile/root/cap gates run
/// before any semantic scan. No bytes are returned unless the complete named
/// conjunction and the public independent verifier both succeed.
pub fn produce_refute_leaf_exact_v1_after_search(
    mode: RefuteLeafModeV1,
    state: &HexoState,
    reachable: &ReachableRootV1,
    policy: OfflinePolicyV1,
    profile: SearchProfileV1,
    expansions: u64,
    node_cap: u64,
) -> ProduceResultV1 {
    if mode == RefuteLeafModeV1::Off {
        return ProduceResultV1::Disabled;
    }
    if !profile.exact() {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleLeafProfile);
    }
    if expansions >= node_cap {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleNodeCap);
    }
    if !policy.within_compiled_ceilings() {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::UnsupportedPolicyBudget(
            "policy ceiling",
        ));
    }
    if state.phase() != TurnPhase::FirstStone {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleRoot("phase"));
    }
    if state.terminal().is_some() {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::NotRefuteLeafExactSemantic(
            "ClaimantTerminal",
        ));
    }
    if state.placements_made() == 0 {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleRoot("post-opening"));
    }
    if state.placements_made().checked_add(1).is_none() {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleRoot("clock overflow"));
    }
    let Some(root) = root_header_from_engine(state) else {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleRoot("root"));
    };
    if root.stones.len() as u64 > policy.root_stones {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::UnsupportedPolicyBudget(
            "root stones",
        ));
    }
    let root_bytes = (root.stones.len() as u64).saturating_mul(24);
    if root_bytes > policy.state_bytes || root_bytes.saturating_mul(3) > policy.heap_bytes {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::UnsupportedPolicyBudget(
            "root allocation",
        ));
    }
    let digest = root_semantic_sha256(&root);
    if !reachable.matches(state, &digest) {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleRoot("reachability"));
    }
    if root.stones.iter().any(|s| {
        !root_safe(P {
            i: s.q as i32,
            j: s.r as i32,
        })
    }) {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::IneligibleRoot("D6 root domain"));
    }

    let producer_started = Instant::now();
    let mut meter = ProdMeter::new(policy);
    let semantic = match regenerate_producer(state, &mut meter) {
        Ok(v) => v,
        Err(ProdError::Budget(n)) => {
            return ProduceResultV1::Ineligible(LeafIneligibilityV1::UnsupportedPolicyBudget(n))
        }
        Err(ProdError::Semantic(n)) => {
            return ProduceResultV1::Ineligible(LeafIneligibilityV1::NotRefuteLeafExactSemantic(n))
        }
    };
    if semantic.completion != 0 || semantic.tactical != 0 || semantic.tight != 0 {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::NotRefuteLeafExactSemantic(
            "positive or tight disposition",
        ));
    }
    let artifact = LeafArtifactV1 {
        root,
        root_semantic_sha256: digest,
        counts: semantic.counts,
    };
    let bytes = encode_artifact(&artifact);
    if bytes.len() as u64 > policy.wire_bytes {
        return ProduceResultV1::Ineligible(LeafIneligibilityV1::UnsupportedPolicyBudget(
            "wire bytes",
        ));
    }
    let producer_wall = producer_started.elapsed();
    let verifier_started = Instant::now();
    let verified = match verify_refute_leaf_exact_v1(state, reachable, &bytes, policy) {
        Ok(v) => v,
        Err(e) => {
            return ProduceResultV1::Ineligible(LeafIneligibilityV1::SelfVerificationRejected(e))
        }
    };
    ProduceResultV1::Emitted(ProducedRefuteLeafV1 {
        bytes,
        artifact,
        verified: verified.verified,
        producer_work: meter.work,
        producer_wall,
        verifier_wall: verifier_started.elapsed(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct P {
    i: i32,
    j: i32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    axis: u8,
    start: P,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PPhase {
    First,
    Second(P),
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct PState {
    map: BTreeMap<P, u8>,
    mover: u8,
    phase: PPhase,
    clock: u32,
    terminal: Option<u8>,
}
#[derive(Debug)]
enum ProdError {
    Budget(&'static str),
    Semantic(&'static str),
}
struct ProdMeter {
    policy: OfflinePolicyV1,
    start: Instant,
    work: RefuteWorkV1,
}
impl ProdMeter {
    fn new(policy: OfflinePolicyV1) -> Self {
        Self {
            policy,
            start: Instant::now(),
            work: RefuteWorkV1::default(),
        }
    }
    fn bump(v: &mut u64, limit: u64, n: &'static str) -> Result<(), ProdError> {
        let x = v.checked_add(1).ok_or(ProdError::Budget(n))?;
        if x > limit {
            return Err(ProdError::Budget(n));
        }
        *v = x;
        Ok(())
    }
    fn check(&self) -> Result<(), ProdError> {
        let ms = self.start.elapsed().as_millis() as u64;
        if ms > self.policy.wall_ms {
            return Err(ProdError::Budget("wall"));
        }
        if ms > self.policy.cpu_ms {
            return Err(ProdError::Budget("cpu"));
        }
        Ok(())
    }
    fn w(&mut self) -> Result<(), ProdError> {
        Self::bump(&mut self.work.windows, self.policy.windows, "windows")?;
        self.check()
    }
    fn q(&mut self) -> Result<(), ProdError> {
        Self::bump(&mut self.work.q, self.policy.q_count, "Q")?;
        self.check()
    }
    fn m(&mut self) -> Result<(), ProdError> {
        Self::bump(
            &mut self.work.threat_memberships,
            self.policy.threat_memberships,
            "threat memberships",
        )?;
        self.check()
    }
    fn op(&mut self) -> Result<(), ProdError> {
        Self::bump(&mut self.work.pair_ops, self.policy.pair_ops, "pair ops")?;
        self.check()
    }
    fn tr(&mut self) -> Result<(), ProdError> {
        Self::bump(
            &mut self.work.transversal_ops,
            self.policy.transversal_ops,
            "transversal ops",
        )?;
        self.check()
    }
}
fn pl(p: Player) -> u8 {
    match p {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}
fn oth(p: u8) -> u8 {
    p ^ 1
}
fn safe(c: P) -> bool {
    let Some(s) = c.i.checked_add(c.j).and_then(|x| x.checked_neg()) else {
        return false;
    };
    [c.i, c.j, s].iter().all(|&x| i16::try_from(x).is_ok())
}
fn root_safe(c: P) -> bool {
    let Some(s) = c.i.checked_add(c.j).and_then(|x| x.checked_neg()) else {
        return false;
    };
    c.i.abs() <= 31_480 && c.j.abs() <= 31_480 && s.abs() <= 31_480 && safe(c)
}
fn metric(a: P, b: P) -> i32 {
    let x = a.i - b.i;
    let y = a.j - b.j;
    x.abs().max(y.abs()).max((-x - y).abs())
}
fn is_legal(s: &PState, c: P) -> bool {
    !s.map.contains_key(&c) && s.map.keys().any(|&x| metric(x, c) <= 8)
}
fn six(k: Key) -> [P; 6] {
    let d = match k.axis {
        0 => P { i: 1, j: 0 },
        1 => P { i: 0, j: 1 },
        _ => P { i: 1, j: -1 },
    };
    std::array::from_fn(|n| P {
        i: k.start.i + d.i * n as i32,
        j: k.start.j + d.j * n as i32,
    })
}
fn all_windows(s: &PState, m: &mut ProdMeter) -> Result<Vec<Key>, ProdError> {
    let dirs = [P { i: 1, j: 0 }, P { i: 0, j: 1 }, P { i: 1, j: -1 }];
    let mut keys = BTreeSet::new();
    for &c in s.map.keys() {
        for (axis, d) in dirs.iter().enumerate() {
            for n in 0..6 {
                let k = Key {
                    axis: axis as u8,
                    start: P {
                        i: c.i - d.i * n,
                        j: c.j - d.j * n,
                    },
                };
                if six(k).iter().any(|&x| !safe(x)) {
                    return Err(ProdError::Semantic("D6 discovered coordinate"));
                }
                if !keys.contains(&k) {
                    m.w()?;
                    keys.insert(k);
                }
            }
        }
    }
    Ok(keys.into_iter().collect())
}
fn nown(s: &PState, k: Key, p: u8) -> usize {
    six(k).iter().filter(|x| s.map.get(x) == Some(&p)).count()
}
fn gaps(s: &PState, k: Key) -> Vec<P> {
    six(k)
        .into_iter()
        .filter(|x| !s.map.contains_key(x))
        .collect()
}
fn is_live(s: &PState, k: Key, p: u8) -> bool {
    nown(s, k, p) > 0 && nown(s, k, oth(p)) == 0 && !gaps(s, k).is_empty()
}
fn won(s: &PState, c: P) -> Option<u8> {
    let p = *s.map.get(&c)?;
    for axis in 0..3 {
        let d = match axis {
            0 => P { i: 1, j: 0 },
            1 => P { i: 0, j: 1 },
            _ => P { i: 1, j: -1 },
        };
        for n in 0..6 {
            if six(Key {
                axis,
                start: P {
                    i: c.i - d.i * n,
                    j: c.j - d.j * n,
                },
            })
            .iter()
            .all(|x| s.map.get(x) == Some(&p))
            {
                return Some(p);
            }
        }
    }
    None
}
fn place(s: &PState, c: P) -> Result<PState, ProdError> {
    if s.terminal.is_some() || !is_legal(s, c) || !safe(c) {
        return Err(ProdError::Semantic("illegal replay"));
    }
    let mut x = s.clone();
    x.map.insert(c, s.mover);
    x.clock = x.clock.checked_add(1).ok_or(ProdError::Semantic("clock"))?;
    x.terminal = won(&x, c);
    if x.terminal.is_none() {
        match s.phase {
            PPhase::First => x.phase = PPhase::Second(c),
            PPhase::Second(_) => {
                x.phase = PPhase::First;
                x.mover = oth(s.mover);
            }
        }
    }
    Ok(x)
}
fn initial(e: &HexoState) -> PState {
    PState {
        map: e
            .board()
            .occupied_cells()
            .iter()
            .map(|&c| {
                (
                    P {
                        i: c.q as i32,
                        j: c.r as i32,
                    },
                    pl(e.board().get(c).unwrap()),
                )
            })
            .collect(),
        mover: pl(e.current_player()),
        phase: PPhase::First,
        clock: e.placements_made(),
        terminal: e.terminal().map(|o| pl(o.winner)),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hit {
    Z,
    O,
    T,
    M,
}
fn hit(f: &[Vec<P>], m: &mut ProdMeter) -> Result<Hit, ProdError> {
    if f.is_empty() {
        return Ok(Hit::Z);
    }
    if f.iter().any(Vec::is_empty) {
        return Ok(Hit::M);
    }
    for &x in &f[0] {
        let mut all = true;
        for set in f {
            m.tr()?;
            if !set.contains(&x) {
                all = false;
                break;
            }
        }
        if all {
            return Ok(Hit::O);
        }
    }
    let mut u = BTreeSet::new();
    for set in f {
        for &x in set {
            m.m()?;
            u.insert(x);
        }
    }
    let v = u.into_iter().collect::<Vec<_>>();
    for i in 0..v.len() {
        for j in i + 1..v.len() {
            let mut yes = true;
            for set in f {
                m.tr()?;
                if !set.contains(&v[i]) && !set.contains(&v[j]) {
                    yes = false;
                    break;
                }
            }
            if yes {
                return Ok(Hit::T);
            }
        }
    }
    Ok(Hit::M)
}
fn family(
    s: &PState,
    ws: &[Key],
    p: u8,
    min: usize,
    m: &mut ProdMeter,
) -> Result<Vec<Vec<P>>, ProdError> {
    let mut f = Vec::new();
    for &k in ws {
        if is_live(s, k, p) && nown(s, k, p) >= min {
            let g = gaps(s, k);
            for _ in &g {
                m.m()?
            }
            f.push(g)
        }
    }
    Ok(f)
}
fn win_now(s: &PState, ws: &[Key], p: u8, b: u8, m: &mut ProdMeter) -> Result<bool, ProdError> {
    for &k in ws {
        m.op()?;
        if is_live(s, k, p) {
            let g = gaps(s, k);
            if !g.is_empty() && g.len() <= b as usize && g.iter().all(|&x| is_legal(s, x)) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Why {
    N,
    D,
    L0,
    L1,
    C,
    X,
    T,
}
#[derive(Clone)]
struct Eval {
    why: Why,
    prefix_terminal: bool,
    full: Option<PState>,
}
#[derive(Default)]
struct ProdSemantic {
    counts: LeafCountsV1,
    completion: u64,
    tactical: u64,
    tight: u64,
}
fn regenerate_producer(engine: &HexoState, m: &mut ProdMeter) -> Result<ProdSemantic, ProdError> {
    regenerate_producer_inner(engine, m, true)
}
fn regenerate_producer_inner(
    engine: &HexoState,
    m: &mut ProdMeter,
    enforce_earlier_constructors: bool,
) -> Result<ProdSemantic, ProdError> {
    let s = initial(engine);
    let a = s.mover;
    let ws = all_windows(&s, m)?;
    if enforce_earlier_constructors && win_now(&s, &ws, a, 2, m)? {
        return Err(ProdError::Semantic("OwnWinNow_A"));
    }
    if enforce_earlier_constructors && matches!(hit(&family(&s, &ws, oth(a), 4, m)?, m)?, Hit::M) {
        return Err(ProdError::Semantic("ForcedLoss_A"));
    }
    let mut t = BTreeSet::new();
    for &k in &ws {
        if (is_live(&s, k, a) && nown(&s, k, a) >= 2)
            || (is_live(&s, k, oth(a)) && nown(&s, k, oth(a)) >= 4)
        {
            for c in gaps(&s, k) {
                m.op()?;
                if is_legal(&s, c) && !t.contains(&c) {
                    if t.len() as u64 >= m.policy.t_count {
                        return Err(ProdError::Budget("T"));
                    }
                    t.insert(c);
                }
            }
        }
    }
    let mut out = ProdSemantic::default();
    out.counts.t_count = t.len() as u64;
    for &x in &t {
        let mut ss = t
            .iter()
            .copied()
            .filter(|&y| y != x)
            .collect::<BTreeSet<_>>();
        if ss.len() as u64 > m.policy.s_count {
            return Err(ProdError::Budget("S"));
        }
        for &k in &ws {
            if is_live(&s, k, a) && nown(&s, k, a) >= 1 {
                let g = gaps(&s, k);
                if g.contains(&x) {
                    for y in g {
                        m.op()?;
                        if y != x && is_legal(&s, y) && !ss.contains(&y) {
                            if ss.len() as u64 >= m.policy.s_count {
                                return Err(ProdError::Budget("S"));
                            }
                            ss.insert(y);
                        }
                    }
                }
            }
        }
        for y in ss {
            m.q()?;
            out.counts.q_count += 1;
            if t.contains(&y) {
                let first = eval_pair(&s, &ws, a, x, y, m)?;
                let reverse = eval_pair(&s, &ws, a, y, x, m)?;
                let commute = !first.prefix_terminal
                    && !reverse.prefix_terminal
                    && first.full.is_some()
                    && first.full == reverse.full;
                if commute {
                    if first.why != reverse.why {
                        return Err(ProdError::Semantic("quotient disagreement"));
                    }
                    if x < y {
                        add_prod(&mut out, first.why, 2);
                        out.counts.quotient_class_count += 1
                    }
                    continue;
                }
                add_prod(&mut out, first.why, 1);
                out.counts.quotient_class_count += 1
            } else {
                let first = eval_pair(&s, &ws, a, x, y, m)?;
                add_prod(&mut out, first.why, 1);
                out.counts.quotient_class_count += 1
            }
        }
    }
    Ok(out)
}
fn add_prod(out: &mut ProdSemantic, w: Why, n: u64) {
    match w {
        Why::N => out.counts.fail_no_new += n,
        Why::D => out.counts.fail_defender_first += n,
        Why::L0 => out.counts.fail_loose_0 += n,
        Why::L1 => out.counts.fail_loose_1 += n,
        Why::C => out.completion += n,
        Why::X => out.tactical += n,
        Why::T => out.tight += n,
    }
}
fn eval_pair(
    s: &PState,
    ws: &[Key],
    a: u8,
    x: P,
    y: P,
    m: &mut ProdMeter,
) -> Result<Eval, ProdError> {
    let sx = place(s, x)?;
    if sx.terminal == Some(a) {
        return Ok(Eval {
            why: Why::C,
            prefix_terminal: true,
            full: None,
        });
    }
    let xy = place(&sx, y)?;
    if xy.terminal == Some(a) {
        return Ok(Eval {
            why: Why::C,
            prefix_terminal: false,
            full: Some(xy),
        });
    }
    if xy.terminal.is_some() {
        return Err(ProdError::Semantic("nonclaimant terminal"));
    }
    let mut f = Vec::new();
    for &k in ws {
        m.op()?;
        if is_live(s, k, a) && nown(s, k, a) >= 2 {
            let cs = six(k);
            if cs.contains(&x) || cs.contains(&y) {
                let g = gaps(s, k);
                if nown(s, k, a) + usize::from(g.contains(&x)) + usize::from(g.contains(&y)) >= 4 {
                    let after = gaps(&xy, k);
                    for _ in &after {
                        m.m()?
                    }
                    f.push(after)
                }
            }
        }
    }
    let why = if f.is_empty() {
        Why::N
    } else {
        let mut dw = false;
        for &k in ws {
            m.op()?;
            if is_live(s, k, oth(a)) && nown(s, k, oth(a)) >= 4 {
                let cs = six(k);
                if !cs.contains(&x) && !cs.contains(&y) {
                    dw = true;
                    break;
                }
            }
        }
        if dw {
            Why::D
        } else {
            match hit(&f, m)? {
                Hit::Z => Why::L0,
                Hit::O => Why::L1,
                Hit::T => Why::T,
                Hit::M => Why::X,
            }
        }
    };
    Ok(Eval {
        why,
        prefix_terminal: false,
        full: Some(xy),
    })
}

#[cfg(test)]
pub(crate) fn producer_counts_for_test(
    state: &HexoState,
    policy: OfflinePolicyV1,
) -> Result<(LeafCountsV1, u64, u64, u64), String> {
    let mut m = ProdMeter::new(policy);
    regenerate_producer(state, &mut m)
        .map(|x| (x.counts, x.completion, x.tactical, x.tight))
        .map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
pub(crate) fn producer_counts_without_earlier_for_test(
    state: &HexoState,
    policy: OfflinePolicyV1,
) -> Result<(LeafCountsV1, u64, u64, u64), String> {
    let mut m = ProdMeter::new(policy);
    regenerate_producer_inner(state, &mut m, false)
        .map(|x| (x.counts, x.completion, x.tactical, x.tight))
        .map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flag_parser_is_default_off() {
        let old = std::env::var_os("TSS_REFUTE_CERT_V1");
        std::env::remove_var("TSS_REFUTE_CERT_V1");
        assert_eq!(RefuteLeafModeV1::from_env(), RefuteLeafModeV1::Off);
        if let Some(v) = old {
            std::env::set_var("TSS_REFUTE_CERT_V1", v)
        }
    }
}
