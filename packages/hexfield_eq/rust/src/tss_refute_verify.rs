//! Independent, fail-closed verifier for the leaf-only `RefuteLeafExact/V1`.
//!
//! This module intentionally does not import the solver, positive verifier,
//! threat analyzer, window store, or any producer semantic helper.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_refute_leaf_cert::{
    sha256, LeafArtifactV1, LeafCountsV1, ReachableRootV1, RootHeaderV1, WireStoneV1, CLASS_V1,
    COORDINATE_V1, FORMAT_V1, MAGIC, MAX_CPU_MS, MAX_HEAP_BYTES, MAX_PAIR_OPS, MAX_Q,
    MAX_ROOT_STONES, MAX_S, MAX_STATE_BYTES, MAX_T, MAX_THREAT_MEMBERSHIPS, MAX_TRANSVERSAL_OPS,
    MAX_WALL_MS, MAX_WINDOWS, MAX_WIRE_BYTES, PROFILE_V1, ROOT_DOMAIN, RULESET_V1,
    TAG_NO_ADMISSIBLE_FIRST_TURN,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OfflinePolicyV1 {
    pub wire_bytes: u64,
    pub root_stones: u64,
    pub windows: u64,
    pub t_count: u64,
    pub s_count: u64,
    pub q_count: u64,
    pub threat_memberships: u64,
    pub pair_ops: u64,
    pub transversal_ops: u64,
    pub state_bytes: u64,
    pub heap_bytes: u64,
    pub cpu_ms: u64,
    pub wall_ms: u64,
}

impl Default for OfflinePolicyV1 {
    fn default() -> Self {
        Self {
            wire_bytes: MAX_WIRE_BYTES as u64,
            root_stones: MAX_ROOT_STONES,
            windows: MAX_WINDOWS,
            t_count: MAX_T,
            s_count: MAX_S,
            q_count: MAX_Q,
            threat_memberships: MAX_THREAT_MEMBERSHIPS,
            pair_ops: MAX_PAIR_OPS,
            transversal_ops: MAX_TRANSVERSAL_OPS,
            state_bytes: MAX_STATE_BYTES,
            heap_bytes: MAX_HEAP_BYTES,
            cpu_ms: MAX_CPU_MS,
            wall_ms: MAX_WALL_MS,
        }
    }
}

impl OfflinePolicyV1 {
    pub fn within_compiled_ceilings(self) -> bool {
        self.wire_bytes <= MAX_WIRE_BYTES as u64
            && self.root_stones <= MAX_ROOT_STONES
            && self.windows <= MAX_WINDOWS
            && self.t_count <= MAX_T
            && self.s_count <= MAX_S
            && self.q_count <= MAX_Q
            && self.threat_memberships <= MAX_THREAT_MEMBERSHIPS
            && self.pair_ops <= MAX_PAIR_OPS
            && self.transversal_ops <= MAX_TRANSVERSAL_OPS
            && self.state_bytes <= MAX_STATE_BYTES
            && self.heap_bytes <= MAX_HEAP_BYTES
            && self.cpu_ms <= MAX_CPU_MS
            && self.wall_ms <= MAX_WALL_MS
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyRejectionV1 {
    Malformed(&'static str),
    Version(&'static str),
    RootBinding(&'static str),
    Semantic(&'static str),
    UnsupportedPolicyBudget(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedClassRefutationV1 {
    pub ruleset: u16,
    pub coordinate_version: u16,
    pub class_version: u16,
    pub wire_version: u16,
    pub root_semantic_sha256: [u8; 32],
    pub claimant: u8,
    pub reachable_root_token: ReachableRootV1,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefuteWorkV1 {
    pub windows: u64,
    pub q: u64,
    pub threat_memberships: u64,
    pub pair_ops: u64,
    pub transversal_ops: u64,
    pub retained_state_bytes: u64,
    pub estimated_heap_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifySuccessV1 {
    pub verified: VerifiedClassRefutationV1,
    pub artifact: LeafArtifactV1,
    pub work: RefuteWorkV1,
}

pub fn verify_refute_leaf_exact_v1(
    state: &HexoState,
    reachable: &ReachableRootV1,
    bytes: &[u8],
    policy: OfflinePolicyV1,
) -> Result<VerifySuccessV1, VerifyRejectionV1> {
    if !policy.within_compiled_ceilings() {
        return Err(VerifyRejectionV1::UnsupportedPolicyBudget("policy ceiling"));
    }
    if bytes.len() as u64 > policy.wire_bytes {
        return Err(VerifyRejectionV1::UnsupportedPolicyBudget("wire bytes"));
    }
    let artifact = decode_strict(bytes, policy)?;
    let mut budget = Budget::new(policy);
    bind_root(state, reachable, &artifact, &mut budget)?;
    let regenerated = regenerate(state, artifact.root.claimant, &mut budget)?;
    if regenerated.completion != 0 || regenerated.tactical != 0 || regenerated.tight != 0 {
        return Err(VerifyRejectionV1::Semantic("positive or tight disposition"));
    }
    if regenerated.counts != artifact.counts {
        return Err(VerifyRejectionV1::Semantic("telemetry mismatch"));
    }
    let fail_sum = artifact
        .counts
        .fail_no_new
        .checked_add(artifact.counts.fail_defender_first)
        .and_then(|v| v.checked_add(artifact.counts.fail_loose_0))
        .and_then(|v| v.checked_add(artifact.counts.fail_loose_1))
        .ok_or(VerifyRejectionV1::Malformed("counter overflow"))?;
    if fail_sum != artifact.counts.q_count
        || regenerated.failure_classes != artifact.counts.quotient_class_count
    {
        return Err(VerifyRejectionV1::Semantic("sum identity"));
    }
    let verified = VerifiedClassRefutationV1 {
        ruleset: RULESET_V1,
        coordinate_version: COORDINATE_V1,
        class_version: CLASS_V1,
        wire_version: FORMAT_V1,
        root_semantic_sha256: artifact.root_semantic_sha256,
        claimant: artifact.root.claimant,
        reachable_root_token: reachable.clone(),
    };
    Ok(VerifySuccessV1 {
        verified,
        artifact,
        work: budget.work,
    })
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Parser<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], VerifyRejectionV1> {
        let end = self
            .at
            .checked_add(n)
            .ok_or(VerifyRejectionV1::Malformed("length overflow"))?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or(VerifyRejectionV1::Malformed("truncated"))?;
        self.at = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, VerifyRejectionV1> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, VerifyRejectionV1> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32, VerifyRejectionV1> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i16(&mut self) -> Result<i16, VerifyRejectionV1> {
        let b = self.take(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }
    fn uvar(&mut self) -> Result<u64, VerifyRejectionV1> {
        let start = self.at;
        let mut value = 0u64;
        for shift in (0..=63).step_by(7) {
            let byte = self.u8()?;
            if shift == 63 && byte > 1 {
                return Err(VerifyRejectionV1::Malformed("uvar overflow"));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                let shortest = if value == 0 {
                    1
                } else {
                    ((64 - value.leading_zeros() as usize) + 6) / 7
                };
                if self.at - start != shortest {
                    return Err(VerifyRejectionV1::Malformed("noncanonical uvar"));
                }
                return Ok(value);
            }
        }
        Err(VerifyRejectionV1::Malformed("uvar overflow"))
    }
}

fn decode_strict(
    bytes: &[u8],
    policy: OfflinePolicyV1,
) -> Result<LeafArtifactV1, VerifyRejectionV1> {
    let mut p = Parser { bytes, at: 0 };
    if p.take(8)? != MAGIC {
        return Err(VerifyRejectionV1::Malformed("magic"));
    }
    for (got, want, name) in [
        (p.u16()?, FORMAT_V1, "format"),
        (p.u16()?, RULESET_V1, "ruleset"),
        (p.u16()?, COORDINATE_V1, "coordinate"),
        (p.u16()?, CLASS_V1, "class"),
        (p.u16()?, PROFILE_V1, "profile"),
    ] {
        if got != want {
            return Err(VerifyRejectionV1::Version(name));
        }
    }
    let count = p.uvar()?;
    if count > policy.root_stones {
        return Err(VerifyRejectionV1::UnsupportedPolicyBudget("root stones"));
    }
    let capacity =
        usize::try_from(count).map_err(|_| VerifyRejectionV1::Malformed("stone count"))?;
    let root_bytes = count
        .checked_mul(24)
        .ok_or(VerifyRejectionV1::UnsupportedPolicyBudget("state bytes"))?;
    if root_bytes > policy.state_bytes || root_bytes.saturating_mul(3) > policy.heap_bytes {
        return Err(VerifyRejectionV1::UnsupportedPolicyBudget(
            "root allocation",
        ));
    }
    let mut stones = Vec::with_capacity(capacity);
    let mut previous = None;
    for _ in 0..count {
        let stone = WireStoneV1 {
            q: p.i16()?,
            r: p.i16()?,
            owner: p.u8()?,
        };
        if stone.owner > 1 {
            return Err(VerifyRejectionV1::Malformed("owner"));
        }
        if previous.is_some_and(|x: (i16, i16)| x >= (stone.q, stone.r)) {
            return Err(VerifyRejectionV1::Malformed("stone order"));
        }
        previous = Some((stone.q, stone.r));
        stones.push(stone);
    }
    let mover = p.u8()?;
    if mover > 1 {
        return Err(VerifyRejectionV1::Malformed("mover"));
    }
    let phase = p.u8()?;
    if phase != 1 {
        return Err(VerifyRejectionV1::Version("phase"));
    }
    let placements_made = p.u32()?;
    let terminal = p.u8()?;
    if terminal != 0 {
        return Err(VerifyRejectionV1::Version("terminal"));
    }
    let claimant = p.u8()?;
    if claimant != mover {
        return Err(VerifyRejectionV1::Malformed("claimant"));
    }
    let digest_slice = p.take(32)?;
    let mut digest = [0u8; 32];
    digest.copy_from_slice(digest_slice);
    let payload_len = p.uvar()?;
    let payload_len =
        usize::try_from(payload_len).map_err(|_| VerifyRejectionV1::Malformed("payload length"))?;
    let payload = p.take(payload_len)?;
    let trailer = p.take(32)?;
    if p.at != bytes.len() {
        return Err(VerifyRejectionV1::Malformed("trailing bytes"));
    }
    if sha256(payload).as_slice() != trailer {
        return Err(VerifyRejectionV1::Malformed("payload checksum"));
    }
    let mut q = Parser {
        bytes: payload,
        at: 0,
    };
    if q.u8()? != TAG_NO_ADMISSIBLE_FIRST_TURN {
        return Err(VerifyRejectionV1::Version("tag"));
    }
    let counts = LeafCountsV1 {
        t_count: q.uvar()?,
        q_count: q.uvar()?,
        quotient_class_count: q.uvar()?,
        fail_no_new: q.uvar()?,
        fail_defender_first: q.uvar()?,
        fail_loose_0: q.uvar()?,
        fail_loose_1: q.uvar()?,
    };
    if q.at != payload.len() {
        return Err(VerifyRejectionV1::Malformed("payload trailing"));
    }
    Ok(LeafArtifactV1 {
        root: RootHeaderV1 {
            stones,
            mover,
            phase,
            phase_first: None,
            placements_made,
            terminal,
            claimant,
        },
        root_semantic_sha256: digest,
        counts,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct C {
    i: i32,
    j: i32,
}
impl C {
    fn from_hex(c: HexCoord) -> Self {
        Self {
            i: c.q as i32,
            j: c.r as i32,
        }
    }
    fn hex(self) -> Result<HexCoord, VerifyRejectionV1> {
        Ok(HexCoord {
            q: i16::try_from(self.i)
                .map_err(|_| VerifyRejectionV1::Semantic("unsafe coordinate"))?,
            r: i16::try_from(self.j)
                .map_err(|_| VerifyRejectionV1::Semantic("unsafe coordinate"))?,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct W {
    axis: u8,
    start: C,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    First,
    Second(C),
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Direct {
    stones: BTreeMap<C, u8>,
    mover: u8,
    phase: Phase,
    placements: u32,
    terminal: Option<u8>,
}

struct Budget {
    policy: OfflinePolicyV1,
    start: Instant,
    work: RefuteWorkV1,
}
impl Budget {
    fn new(policy: OfflinePolicyV1) -> Self {
        Self {
            policy,
            start: Instant::now(),
            work: RefuteWorkV1::default(),
        }
    }
    fn checkpoint(&self) -> Result<(), VerifyRejectionV1> {
        let ms = self.start.elapsed().as_millis() as u64;
        if ms > self.policy.wall_ms {
            return Err(VerifyRejectionV1::UnsupportedPolicyBudget("wall"));
        }
        if ms > self.policy.cpu_ms {
            return Err(VerifyRejectionV1::UnsupportedPolicyBudget("cpu"));
        }
        Ok(())
    }
    fn charge(
        slot: &mut u64,
        add: u64,
        limit: u64,
        name: &'static str,
    ) -> Result<(), VerifyRejectionV1> {
        let next = slot
            .checked_add(add)
            .ok_or(VerifyRejectionV1::UnsupportedPolicyBudget(name))?;
        if next > limit {
            return Err(VerifyRejectionV1::UnsupportedPolicyBudget(name));
        }
        *slot = next;
        Ok(())
    }
    fn window(&mut self) -> Result<(), VerifyRejectionV1> {
        Self::charge(&mut self.work.windows, 1, self.policy.windows, "windows")?;
        self.checkpoint()
    }
    fn q(&mut self) -> Result<(), VerifyRejectionV1> {
        Self::charge(&mut self.work.q, 1, self.policy.q_count, "Q")?;
        self.checkpoint()
    }
    fn threat(&mut self) -> Result<(), VerifyRejectionV1> {
        Self::charge(
            &mut self.work.threat_memberships,
            1,
            self.policy.threat_memberships,
            "threat memberships",
        )?;
        self.checkpoint()
    }
    fn pair(&mut self) -> Result<(), VerifyRejectionV1> {
        Self::charge(&mut self.work.pair_ops, 1, self.policy.pair_ops, "pair ops")?;
        self.checkpoint()
    }
    fn transversal(&mut self) -> Result<(), VerifyRejectionV1> {
        Self::charge(
            &mut self.work.transversal_ops,
            1,
            self.policy.transversal_ops,
            "transversal ops",
        )?;
        self.checkpoint()
    }
    fn retain(&mut self, bytes: u64) -> Result<(), VerifyRejectionV1> {
        if bytes > self.policy.state_bytes {
            return Err(VerifyRejectionV1::UnsupportedPolicyBudget("state bytes"));
        }
        self.work.retained_state_bytes = self.work.retained_state_bytes.max(bytes);
        let heap = bytes.saturating_mul(3);
        if heap > self.policy.heap_bytes {
            return Err(VerifyRejectionV1::UnsupportedPolicyBudget("heap"));
        }
        self.work.estimated_heap_bytes = self.work.estimated_heap_bytes.max(heap);
        Ok(())
    }
}

fn bind_root(
    state: &HexoState,
    token: &ReachableRootV1,
    a: &LeafArtifactV1,
    b: &mut Budget,
) -> Result<(), VerifyRejectionV1> {
    if state.terminal().is_some() {
        return Err(VerifyRejectionV1::Semantic("ClaimantTerminal"));
    }
    if state.phase() != TurnPhase::FirstStone {
        return Err(VerifyRejectionV1::RootBinding("phase"));
    }
    if state.placements_made().checked_add(1).is_none() {
        return Err(VerifyRejectionV1::RootBinding("clock overflow"));
    }
    if a.root.placements_made != state.placements_made()
        || a.root.stones.len() != state.board().len()
        || a.root.claimant != player(state.current_player())
    {
        return Err(VerifyRejectionV1::RootBinding("header"));
    }
    let mut actual = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&c| (c.q, c.r, player(state.board().get(c).unwrap())))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let claimed = a
        .root
        .stones
        .iter()
        .map(|s| (s.q, s.r, s.owner))
        .collect::<Vec<_>>();
    if actual != claimed {
        return Err(VerifyRejectionV1::RootBinding("stones"));
    }
    for s in &a.root.stones {
        if !root_domain(C {
            i: s.q as i32,
            j: s.r as i32,
        }) {
            return Err(VerifyRejectionV1::RootBinding("D6 root domain"));
        }
    }
    let preimage = preimage_private(&a.root)?;
    if sha256(&preimage) != a.root_semantic_sha256 {
        return Err(VerifyRejectionV1::RootBinding("semantic digest"));
    }
    if !token.matches(state, &a.root_semantic_sha256) {
        return Err(VerifyRejectionV1::RootBinding("reachability token"));
    }
    b.retain((a.root.stones.len() as u64).saturating_mul(24))?;
    Ok(())
}

fn preimage_private(root: &RootHeaderV1) -> Result<Vec<u8>, VerifyRejectionV1> {
    let mut out = Vec::new();
    out.extend_from_slice(ROOT_DOMAIN);
    for x in [RULESET_V1, COORDINATE_V1, CLASS_V1, FORMAT_V1, PROFILE_V1] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    put_uvar_private(&mut out, root.stones.len() as u64);
    for s in &root.stones {
        out.extend_from_slice(&s.q.to_le_bytes());
        out.extend_from_slice(&s.r.to_le_bytes());
        out.push(s.owner);
    }
    out.push(root.mover);
    out.push(root.phase);
    if root.phase == 2 {
        let (q, r) = root
            .phase_first
            .ok_or(VerifyRejectionV1::Malformed("phase payload"))?;
        out.extend_from_slice(&q.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out.extend_from_slice(&root.placements_made.to_le_bytes());
    out.push(root.terminal);
    out.push(root.claimant);
    Ok(out)
}
fn put_uvar_private(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let x = (v & 127) as u8;
        v >>= 7;
        out.push(if v == 0 { x } else { x | 128 });
        if v == 0 {
            return;
        }
    }
}
fn player(p: Player) -> u8 {
    match p {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}
fn other(p: u8) -> u8 {
    p ^ 1
}
fn d6(c: C) -> bool {
    let s = match c.i.checked_add(c.j).and_then(|x| x.checked_neg()) {
        Some(x) => x,
        None => return false,
    };
    [c.i, c.j, s].iter().all(|&x| i16::try_from(x).is_ok())
}
fn root_domain(c: C) -> bool {
    let Some(s) = c.i.checked_add(c.j).and_then(|x| x.checked_neg()) else {
        return false;
    };
    c.i.abs() <= 31_480 && c.j.abs() <= 31_480 && s.abs() <= 31_480 && d6(c)
}
fn dist(a: C, c: C) -> i32 {
    let di = a.i - c.i;
    let dj = a.j - c.j;
    let ds = -di - dj;
    di.abs().max(dj.abs()).max(ds.abs())
}
fn legal(p: &Direct, c: C) -> bool {
    !p.stones.contains_key(&c) && p.stones.keys().any(|&s| dist(s, c) <= 8)
}
fn cells(w: W) -> [C; 6] {
    let axis = match w.axis {
        0 => C { i: 1, j: 0 },
        1 => C { i: 0, j: 1 },
        _ => C { i: 1, j: -1 },
    };
    std::array::from_fn(|n| C {
        i: w.start.i + axis.i * n as i32,
        j: w.start.j + axis.j * n as i32,
    })
}
fn windows(p: &Direct, b: &mut Budget) -> Result<Vec<W>, VerifyRejectionV1> {
    let axes = [C { i: 1, j: 0 }, C { i: 0, j: 1 }, C { i: 1, j: -1 }];
    let mut set = BTreeSet::new();
    for &c in p.stones.keys() {
        for (axis, d) in axes.iter().enumerate() {
            for off in 0..6 {
                let start = C {
                    i: c.i - d.i * off,
                    j: c.j - d.j * off,
                };
                for x in cells(W {
                    axis: axis as u8,
                    start,
                }) {
                    if !d6(x) {
                        return Err(VerifyRejectionV1::Semantic("D6 discovered coordinate"));
                    }
                }
                let key = W {
                    axis: axis as u8,
                    start,
                };
                if !set.contains(&key) {
                    b.window()?;
                    set.insert(key);
                }
            }
        }
    }
    Ok(set.into_iter().collect())
}
fn count(p: &Direct, w: W, owner: u8) -> usize {
    cells(w)
        .iter()
        .filter(|c| p.stones.get(c) == Some(&owner))
        .count()
}
fn empties(p: &Direct, w: W) -> Vec<C> {
    cells(w)
        .into_iter()
        .filter(|c| !p.stones.contains_key(c))
        .collect()
}
fn live(p: &Direct, w: W, x: u8) -> bool {
    count(p, w, x) > 0 && count(p, w, other(x)) == 0 && !empties(p, w).is_empty()
}
fn terminal_after(p: &Direct, placed: C) -> Option<u8> {
    let owner = *p.stones.get(&placed)?;
    for axis in 0..3 {
        let d = match axis {
            0 => C { i: 1, j: 0 },
            1 => C { i: 0, j: 1 },
            _ => C { i: 1, j: -1 },
        };
        for off in 0..6 {
            let w = W {
                axis,
                start: C {
                    i: placed.i - d.i * off,
                    j: placed.j - d.j * off,
                },
            };
            if cells(w).iter().all(|c| p.stones.get(c) == Some(&owner)) {
                return Some(owner);
            }
        }
    }
    None
}
fn apply_direct(p: &Direct, c: C) -> Result<Direct, VerifyRejectionV1> {
    if p.terminal.is_some() || !legal(p, c) || !d6(c) {
        return Err(VerifyRejectionV1::Semantic("illegal replay"));
    }
    let mut n = p.clone();
    n.stones.insert(c, p.mover);
    n.placements = n
        .placements
        .checked_add(1)
        .ok_or(VerifyRejectionV1::Semantic("clock"))?;
    n.terminal = terminal_after(&n, c);
    if n.terminal.is_none() {
        match p.phase {
            Phase::First => n.phase = Phase::Second(c),
            Phase::Second(_) => {
                n.phase = Phase::First;
                n.mover = other(p.mover);
            }
        }
    }
    Ok(n)
}
fn from_engine(s: &HexoState) -> Direct {
    let stones = s
        .board()
        .occupied_cells()
        .iter()
        .map(|&c| (C::from_hex(c), player(s.board().get(c).unwrap())))
        .collect();
    let phase = match s.phase() {
        TurnPhase::FirstStone => Phase::First,
        TurnPhase::SecondStone { first } => Phase::Second(C::from_hex(first)),
        TurnPhase::Opening => Phase::First,
    };
    Direct {
        stones,
        mover: player(s.current_player()),
        phase,
        placements: s.placements_made(),
        terminal: s.terminal().map(|o| player(o.winner)),
    }
}
fn agrees(d: &Direct, e: &HexoState) -> bool {
    *d == from_engine(e)
}
fn apply_both(d: &Direct, e: &HexoState, c: C) -> Result<(Direct, HexoState), VerifyRejectionV1> {
    let direct = apply_direct(d, c)?;
    let mut engine = e.clone();
    let result = apply_placement(&mut engine, Placement { coord: c.hex()? })
        .map_err(|_| VerifyRejectionV1::Semantic("engine replay"))?;
    if result.outcome.map(|o| player(o.winner)) != direct.terminal || !agrees(&direct, &engine) {
        return Err(VerifyRejectionV1::Semantic("direct/engine transition"));
    }
    Ok((direct, engine))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tau {
    Zero,
    One,
    Two,
    Over,
}
fn tau(f: &[Vec<C>], b: &mut Budget) -> Result<Tau, VerifyRejectionV1> {
    if f.is_empty() {
        return Ok(Tau::Zero);
    }
    if f.iter().any(Vec::is_empty) {
        return Ok(Tau::Over);
    }
    for &x in &f[0] {
        let mut all = true;
        for set in f {
            b.transversal()?;
            if !set.contains(&x) {
                all = false;
                break;
            }
        }
        if all {
            return Ok(Tau::One);
        }
    }
    let mut u = BTreeSet::new();
    for set in f {
        for &x in set {
            b.threat()?;
            u.insert(x);
        }
    }
    let v = u.into_iter().collect::<Vec<_>>();
    for i in 0..v.len() {
        for j in i + 1..v.len() {
            let mut all = true;
            for set in f {
                b.transversal()?;
                if !set.contains(&v[i]) && !set.contains(&v[j]) {
                    all = false;
                    break;
                }
            }
            if all {
                return Ok(Tau::Two);
            }
        }
    }
    Ok(Tau::Over)
}
fn threat_family(
    p: &Direct,
    ws: &[W],
    x: u8,
    min: usize,
    b: &mut Budget,
) -> Result<Vec<Vec<C>>, VerifyRejectionV1> {
    let mut out = Vec::new();
    for &w in ws {
        if live(p, w, x) && count(p, w, x) >= min {
            let e = empties(p, w);
            for _ in &e {
                b.threat()?;
            }
            out.push(e);
        }
    }
    Ok(out)
}
fn own_win(
    p: &Direct,
    ws: &[W],
    x: u8,
    budget: u8,
    b: &mut Budget,
) -> Result<bool, VerifyRejectionV1> {
    for &w in ws {
        b.pair()?;
        if live(p, w, x) {
            let e = empties(p, w);
            if !e.is_empty() && e.len() <= budget as usize && e.iter().all(|&c| legal(p, c)) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Disp {
    NoNew,
    DefenderFirst,
    Loose0,
    Loose1,
    Completion,
    Tactical,
    Tight,
}
#[derive(Clone)]
struct Occ {
    disp: Disp,
    prefix_terminal: bool,
    full: Option<Direct>,
}
#[derive(Default)]
struct Regen {
    counts: LeafCountsV1,
    completion: u64,
    tactical: u64,
    tight: u64,
    failure_classes: u64,
}

fn regenerate(
    engine: &HexoState,
    claimant: u8,
    b: &mut Budget,
) -> Result<Regen, VerifyRejectionV1> {
    let root = from_engine(engine);
    if root.terminal.is_some() {
        return Err(VerifyRejectionV1::Semantic("ClaimantTerminal"));
    }
    if root.phase != Phase::First || root.mover != claimant {
        return Err(VerifyRejectionV1::Semantic("root polarity"));
    }
    let ws = windows(&root, b)?;
    if own_win(&root, &ws, claimant, 2, b)? {
        return Err(VerifyRejectionV1::Semantic("OwnWinNow_A"));
    }
    let hd = threat_family(&root, &ws, other(claimant), 4, b)?;
    if matches!(tau(&hd, b)?, Tau::Over) {
        return Err(VerifyRejectionV1::Semantic("ForcedLoss_A"));
    }
    let mut t = BTreeSet::new();
    for &w in &ws {
        let add = (live(&root, w, claimant) && count(&root, w, claimant) >= 2)
            || (live(&root, w, other(claimant)) && count(&root, w, other(claimant)) >= 4);
        if add {
            for c in empties(&root, w) {
                b.pair()?;
                if legal(&root, c) && !t.contains(&c) {
                    if t.len() as u64 >= b.policy.t_count {
                        return Err(VerifyRejectionV1::UnsupportedPolicyBudget("T"));
                    }
                    t.insert(c);
                }
            }
        }
    }
    // Stream pair classes. No U table or occurrence map is retained. A reverse
    // occurrence exists exactly when the second coordinate is itself in T,
    // because S(P,b) always contains T-{b}.
    let mut out = Regen::default();
    out.counts.t_count = t.len() as u64;
    for &a in &t {
        let mut s = t
            .iter()
            .copied()
            .filter(|&c| c != a)
            .collect::<BTreeSet<_>>();
        if s.len() as u64 > b.policy.s_count {
            return Err(VerifyRejectionV1::UnsupportedPolicyBudget("S"));
        }
        for &w in &ws {
            if live(&root, w, claimant) && count(&root, w, claimant) >= 1 {
                let e = empties(&root, w);
                if e.contains(&a) {
                    for c in e {
                        b.pair()?;
                        if c != a && legal(&root, c) && !s.contains(&c) {
                            if s.len() as u64 >= b.policy.s_count {
                                return Err(VerifyRejectionV1::UnsupportedPolicyBudget("S"));
                            }
                            s.insert(c);
                        }
                    }
                }
            }
        }
        for c in s {
            b.q()?;
            out.counts.q_count = out
                .counts
                .q_count
                .checked_add(1)
                .ok_or(VerifyRejectionV1::UnsupportedPolicyBudget("Q"))?;
            if t.contains(&c) {
                let first = classify(&root, engine, &ws, claimant, a, c, b)?;
                let reverse = classify(&root, engine, &ws, claimant, c, a, b)?;
                let commuting = !first.prefix_terminal
                    && !reverse.prefix_terminal
                    && first.full.is_some()
                    && first.full == reverse.full;
                if commuting {
                    if first.disp != reverse.disp {
                        return Err(VerifyRejectionV1::Semantic("quotient disagreement"));
                    }
                    if a < c {
                        add_occurrence(&mut out, first.disp, 2);
                        out.counts.quotient_class_count += 1;
                        if failure(first.disp) {
                            out.failure_classes += 1;
                        }
                    }
                    continue;
                }
                add_occurrence(&mut out, first.disp, 1);
                out.counts.quotient_class_count += 1;
                if failure(first.disp) {
                    out.failure_classes += 1;
                }
            } else {
                let first = classify(&root, engine, &ws, claimant, a, c, b)?;
                add_occurrence(&mut out, first.disp, 1);
                out.counts.quotient_class_count += 1;
                if failure(first.disp) {
                    out.failure_classes += 1;
                }
            }
        }
    }
    Ok(out)
}

fn failure(d: Disp) -> bool {
    matches!(
        d,
        Disp::NoNew | Disp::DefenderFirst | Disp::Loose0 | Disp::Loose1
    )
}
fn add_occurrence(out: &mut Regen, d: Disp, n: u64) {
    match d {
        Disp::NoNew => out.counts.fail_no_new += n,
        Disp::DefenderFirst => out.counts.fail_defender_first += n,
        Disp::Loose0 => out.counts.fail_loose_0 += n,
        Disp::Loose1 => out.counts.fail_loose_1 += n,
        Disp::Completion => out.completion += n,
        Disp::Tactical => out.tactical += n,
        Disp::Tight => out.tight += n,
    }
}

fn classify(
    root: &Direct,
    engine: &HexoState,
    ws: &[W],
    a_player: u8,
    a: C,
    c: C,
    b: &mut Budget,
) -> Result<Occ, VerifyRejectionV1> {
    let (pa, ea) = apply_both(root, engine, a)?;
    if pa.terminal == Some(a_player) {
        return Ok(Occ {
            disp: Disp::Completion,
            prefix_terminal: true,
            full: None,
        });
    }
    let (pab, _) = apply_both(&pa, &ea, c)?;
    if pab.terminal == Some(a_player) {
        return Ok(Occ {
            disp: Disp::Completion,
            prefix_terminal: false,
            full: Some(pab),
        });
    }
    if pab.terminal.is_some() {
        return Err(VerifyRejectionV1::Semantic("nonclaimant terminal prefix"));
    }
    let mut family = Vec::new();
    for &w in ws {
        b.pair()?;
        if live(root, w, a_player) && count(root, w, a_player) >= 2 {
            let cs = cells(w);
            if cs.contains(&a) || cs.contains(&c) {
                let er = empties(root, w);
                let projected = count(root, w, a_player)
                    + usize::from(er.contains(&a))
                    + usize::from(er.contains(&c));
                if projected >= 4 {
                    let e = empties(&pab, w);
                    for _ in &e {
                        b.threat()?;
                    }
                    family.push(e);
                }
            }
        }
    }
    let disp = if family.is_empty() {
        Disp::NoNew
    } else {
        let mut defender = false;
        for &w in ws {
            b.pair()?;
            if live(root, w, other(a_player)) && count(root, w, other(a_player)) >= 4 {
                let cs = cells(w);
                if !cs.contains(&a) && !cs.contains(&c) {
                    defender = true;
                    break;
                }
            }
        }
        if defender {
            Disp::DefenderFirst
        } else {
            match tau(&family, b)? {
                Tau::Zero => Disp::Loose0,
                Tau::One => Disp::Loose1,
                Tau::Two => Disp::Tight,
                Tau::Over => Disp::Tactical,
            }
        }
    };
    Ok(Occ {
        disp,
        prefix_terminal: false,
        full: Some(pab),
    })
}

#[cfg(test)]
pub(crate) fn regenerate_for_test(
    state: &HexoState,
    policy: OfflinePolicyV1,
) -> Result<(LeafCountsV1, u64, u64, u64), VerifyRejectionV1> {
    let mut b = Budget::new(policy);
    let r = regenerate(state, player(state.current_player()), &mut b)?;
    Ok((r.counts, r.completion, r.tactical, r.tight))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_default_is_at_ceiling() {
        assert!(OfflinePolicyV1::default().within_compiled_ceilings());
        let mut p = OfflinePolicyV1::default();
        p.q_count += 1;
        assert!(!p.within_compiled_ceilings());
    }
    #[test]
    fn d6_extremes_fail_closed() {
        assert!(!root_domain(C { i: -32768, j: 0 }));
        assert!(root_domain(C {
            i: 31480,
            j: -31480
        }));
    }
    #[test]
    fn every_tau_case_is_literal() {
        let p = OfflinePolicyV1::default();
        let mut b = Budget::new(p);
        assert_eq!(tau(&[], &mut b).unwrap(), Tau::Zero);
        assert_eq!(
            tau(
                &[
                    vec![C { i: 0, j: 0 }],
                    vec![C { i: 0, j: 0 }, C { i: 1, j: 0 }]
                ],
                &mut b
            )
            .unwrap(),
            Tau::One
        );
        assert_eq!(
            tau(&[vec![C { i: 0, j: 0 }], vec![C { i: 1, j: 0 }]], &mut b).unwrap(),
            Tau::Two
        );
        assert_eq!(tau(&[vec![]], &mut b).unwrap(), Tau::Over);
        assert_eq!(
            tau(
                &[
                    vec![C { i: 0, j: 0 }],
                    vec![C { i: 1, j: 0 }],
                    vec![C { i: 2, j: 0 }]
                ],
                &mut b
            )
            .unwrap(),
            Tau::Over
        );
    }
    #[test]
    fn second_stone_first_and_transition_faults_do_not_agree() {
        let mut engine = HexoState::new();
        apply_placement(
            &mut engine,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .unwrap();
        let direct = from_engine(&engine);
        let c = C { i: 1, j: 0 };
        let (placed, engine_placed) = apply_both(&direct, &engine, c).unwrap();
        assert!(agrees(&placed, &engine_placed));
        let mut bad = placed.clone();
        bad.phase = Phase::Second(C { i: 2, j: 0 });
        assert!(!agrees(&bad, &engine_placed));
        let mut bad = placed.clone();
        bad.mover = other(bad.mover);
        assert!(!agrees(&bad, &engine_placed));
        let mut bad = placed.clone();
        bad.terminal = Some(bad.mover);
        assert!(!agrees(&bad, &engine_placed));
    }
}
