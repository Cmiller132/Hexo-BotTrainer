//! Extension verifier for the v1 Group-2 certificate class
//! (.codex-g2-resolve/DESIGN_G2_CERT_EXTENSION.md + DESIGN_AMENDMENT_R1_R2.md).
//!
//! IMPLEMENTED SUB-CLASS (documented narrowing, see .codex-z/G2_IMPL_NOTES.md):
//! gate-free Group-2 certificates — trees containing `UniversalGroup2V1`
//! ordinary reduced-AND nodes but NO `FhwGateV1` node. Any certificate with a
//! gate node REJECTS. On gate-free trees the §3.2 cut clocks coincide with the
//! full clocks (the gate clauses are the only divergence), so the exact
//! per-role/per-window derivation below is the complete FHW obligation for
//! this sub-class.
//!
//! Design principles enforced throughout:
//! - conservative: every unspecified, ambiguous, or arithmetically-overflowing
//!   situation returns reject, never accept (checked arithmetic on all new
//!   paths; work caps reject);
//! - no acceptance oracle: stored scalars/digests are compared against fresh
//!   derivations from the replayed positions; `threats_shared::analyze` is
//!   used only as an ADDITIONAL rejector, never to accept;
//! - this module never imports `tss_solver`.

use std::collections::HashMap;

use hexo_engine::{
    hex_distance, Axis, HexCoord, HexoState as RustHexoState, Placement, Player, TurnPhase,
    WindowKey,
};

use crate::threats_shared;
use crate::tss_core::ProofStatus;
use crate::tss_verify::{
    certificate_metadata_for_group2, d6_transform_coord, validate_arena_for_group2, CertNode,
    CertNodeId, FhwEdgeClassV1, FhwKappaRowV1, FhwRoleRowV1, GuardResultV1, Group2AuthorityV1,
    RoleKeyV1, RootBinding, TssCertificate, D6_SYMMETRY_COUNT, MAX_CERT_DEPTH,
};
use std::collections::HashSet;

/// Hard fail-closed work/memory limits (design §3.5). Reaching any limit is
/// rejection of the new certificate, never partial acceptance.
pub(crate) const MAX_G2_ROLES: usize = 1_000_000;
pub(crate) const MAX_G2_WORK_ITEMS: u64 = 10_000_000;

// ---------------------------------------------------------------------------
// SHA-256 (self-contained; no new crate dependency enters the verifier TCB).
// FIPS 180-4. Golden vectors are pinned in the test module below.
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

pub(crate) struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    total_len: u64,
}

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            total_len: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    pub(crate) fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buffered != 56 {
            self.update(&[0x00]);
        }
        // Manual length append: update() would recount these bytes.
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

pub(crate) fn sha256(domain: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(payload);
    hash.finalize()
}

// ---------------------------------------------------------------------------
// Canonical scalar encoders (design §2.4 scalar grammar).
// ---------------------------------------------------------------------------

fn enc_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn enc_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn enc_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn enc_coord(out: &mut Vec<u8>, coord: HexCoord) {
    out.extend_from_slice(&coord.q.to_le_bytes());
    out.extend_from_slice(&coord.r.to_le_bytes());
}

fn axis_tag(axis: Axis) -> u8 {
    match axis {
        Axis::Q => 0,
        Axis::R => 1,
        Axis::QR => 2,
    }
}

/// Unoriented axis tag, then the numerically lexicographically smaller
/// endpoint (§2.4 window encoding).
fn enc_window(out: &mut Vec<u8>, key: WindowKey) {
    out.push(axis_tag(key.axis));
    let a = key.coord_at(0);
    let b = key.coord_at(5);
    let smaller = if (a.q, a.r) <= (b.q, b.r) { a } else { b };
    enc_coord(out, smaller);
}

fn window_sort_key(key: WindowKey) -> (u8, i16, i16) {
    let a = key.coord_at(0);
    let b = key.coord_at(5);
    let smaller = if (a.q, a.r) <= (b.q, b.r) { a } else { b };
    (axis_tag(key.axis), smaller.q, smaller.r)
}

fn player_tag(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

/// Total canonical order over a stored `RoleKeyV1` (for the stored-cert
/// sorted-unique preflight and the finder's emission order). The Merkle digest
/// re-sorts by encoded bytes under the `g*` frame independently.
fn role_key_order(role: &RoleKeyV1) -> (u8, u32, u8, i16, i16, i16, i16) {
    match role {
        RoleKeyV1::ChoiceMove { node, cell } => (0, *node, 0, 0, 0, cell.q, cell.r),
        RoleKeyV1::OrCompletionMove { node, cell } => (1, *node, 0, 0, 0, cell.q, cell.r),
        RoleKeyV1::LeafEmpty {
            node,
            witness,
            cell,
        } => {
            let w = window_sort_key(*witness);
            (2, *node, w.0, w.1, w.2, cell.q, cell.r)
        }
        RoleKeyV1::Checkpoint {
            gate,
            threat,
            cell,
        } => {
            let w = window_sort_key(*threat);
            (3, *gate, w.0, w.1, w.2, cell.q, cell.r)
        }
    }
}

fn enc_authority(out: &mut Vec<u8>, authority: &Group2AuthorityV1) {
    out.extend_from_slice(&authority.defender_commit);
    enc_u64(out, authority.defender_path.len() as u64);
    out.extend_from_slice(authority.defender_path.as_bytes());
    out.extend_from_slice(&authority.defender_sha256);
    out.extend_from_slice(&authority.fhw_commit);
    enc_u64(out, authority.fhw_path.len() as u64);
    out.extend_from_slice(authority.fhw_path.as_bytes());
    out.extend_from_slice(&authority.fhw_sha256);
}

// ---------------------------------------------------------------------------
// Internal typed view of the (already structurally validated) certificate.
// ---------------------------------------------------------------------------

type CoordKey = (i16, i16);
type WinId = (u8, i16, i16);

fn coord_key(coord: HexCoord) -> CoordKey {
    (coord.q, coord.r)
}

fn win_id(key: WindowKey) -> WinId {
    (axis_tag(key.axis), key.start.q, key.start.r)
}

/// Identity of a live role: the discharge node plus the carried cell (and the
/// named witness for leaf-empty roles). Matches `RoleKeyV1` minus checkpoint
/// roles, which only exist at gates (excluded from this sub-class).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum RoleId {
    ChoiceMove { node: CertNodeId, cell: CoordKey },
    OrCompletionMove { node: CertNodeId, cell: CoordKey },
    LeafEmpty { node: CertNodeId, witness: WinId, cell: CoordKey },
    /// Checkpoint role introduced at an FHW gate: an empty of a named gate
    /// threat, live in every strict ancestor and discharged at the gate
    /// (clock 0 there). Matches `RoleKeyV1::Checkpoint`.
    Checkpoint { gate: CertNodeId, threat: WinId, cell: CoordKey },
}

impl RoleId {
    fn carrier(&self) -> CoordKey {
        match self {
            RoleId::ChoiceMove { cell, .. }
            | RoleId::OrCompletionMove { cell, .. }
            | RoleId::LeafEmpty { cell, .. }
            | RoleId::Checkpoint { cell, .. } => *cell,
        }
    }
}

/// Immutable origin bits for demanded windows (design §2.4).
const SOURCE_TOUCHED: u8 = 0x01;
const SOURCE_VIRGIN: u8 = 0x02;
const SOURCE_DIRECT18: u8 = 0x04;
// SOURCE_GATE_LOCAL_WC (0x08) is not used on this accept path: it only arises
// for genuine NonFrontierCovered gate pairs, which are rejected before demand
// seeding (see `reconstruct_gate`). Kept documented for the §2.4 mask grammar.

/// Reconstructed FHW-T3-R forcing-gate data (design §3.3), produced during
/// replay and consumed by the budget/clock/row/digest passes. Every field is
/// verifier-derived from the replayed gate position `P_Q`; certificate scalars
/// are only ever compared against these, never trusted.
///
/// ACCEPT-PATH NARROWING (documented, fail-closed): this verifier accepts only
/// gates whose every real reply is `Exact` or `FrontierCovered`. Any
/// `NonFrontierCovered` edge is rejected at reconstruction, because a sound
/// end-to-end non-FC gate additionally requires the gate-local WC demand
/// enumeration tied to a proven `B(C_s) >= 6` representative subtree, which
/// cannot be positively fixture-tested in this lane. The RC/WC and charged-role
/// classifiers are nonetheless implemented and unit-tested verifier-side (they
/// are what a later non-FC extension recomputes, and they demonstrate the two
/// documented design defects as verifier-side rejections).
struct GateInfo {
    /// Defender budget b in {1, 2}.
    b: u8,
    /// Validated named threat family H_Q, canonical order.
    threats: Vec<WindowKey>,
    /// Kernel K, sorted-unique by coord key.
    kernel: Vec<HexCoord>,
    /// Representatives R, sorted-unique by coord key.
    reps: Vec<HexCoord>,
    /// Retraction phi: real reply d -> representative s.
    phi: HashMap<CoordKey, HexCoord>,
    /// Representative child node per representative s.
    rep_child: HashMap<CoordKey, CertNodeId>,
    /// Edge class per real reply d (Exact or FrontierCovered on this path).
    edge_class: HashMap<CoordKey, FhwEdgeClassV1>,
    /// escape_resolution_ply = p(Q) + b + 2 (R1).
    escape_ply: u32,
}

struct G2Context<'a> {
    cert: &'a TssCertificate,
    claimant: Player,
    /// Exact replayed state per arena node (tree: exactly one occurrence).
    states: Vec<RustHexoState>,
    /// Outgoing (move, child) pairs in stored order.
    children: Vec<Vec<(HexCoord, CertNodeId)>>,
    /// Postorder over the reachable tree (children before parents).
    postorder: Vec<CertNodeId>,
    /// D14 full scalar local budget per node.
    b_local: Vec<u32>,
    /// Subtree maximum exact leaf resolution per node.
    t_sub: Vec<u32>,
    /// Live-role clock map per node: `(r_full, f_cut)` per role. On gate-free
    /// nodes the two coincide; only the FHW gate clauses of §3.2 diverge
    /// (`r_full` takes the full `1+child` charge; `f_cut` takes
    /// `child + epsilon`). Stored together so `f_cut <= r_full` is checked.
    roles: Vec<HashMap<RoleId, (u32, u32)>>,
    /// Demanded windows with OR'd source bits per node (fixed point of the
    /// ordinary propagation rules; gate arms additionally seed the 18
    /// direct-through-`d` windows, tagged `SOURCE_DIRECT18`).
    demands: Vec<Vec<(WindowKey, u8)>>,
    /// The PAIR `(Q_cut, E_full)` memo per (node, window); design §3.2/§3.3.
    /// On gate-free nodes the two coincide by construction; only the FHW gate
    /// clauses diverge (`Q_cut = max{b, max_d(kappa+Q_cut(C_phi(d)))}` vs
    /// `E_full = max{b, max_d(1+E_full(C_phi(d)))}`). Stored together so the
    /// mandatory `Q_cut <= E_full <= B` inequality is checked on every pair.
    window_clock: HashMap<(CertNodeId, WinId), (u32, u32)>,
    /// Derived k (0 or 1) at each Group-2 node; absent elsewhere.
    derived_k: HashMap<CertNodeId, u8>,
    /// Derived Required_FHW per Group-2 node (§3.4).
    required: HashMap<CertNodeId, Vec<HexCoord>>,
    /// Reconstructed FHW gate data per gate node (§3.3), keyed by node id.
    gates: HashMap<CertNodeId, GateInfo>,
    /// Fail-closed work counter.
    work: u64,
}

impl<'a> G2Context<'a> {
    fn charge(&mut self, items: u64) -> Option<()> {
        self.work = self.work.checked_add(items)?;
        (self.work <= MAX_G2_WORK_ITEMS).then_some(())
    }
}

fn placements_remaining(state: &RustHexoState) -> u8 {
    match state.phase() {
        TurnPhase::Opening => 1,
        TurnPhase::FirstStone => 2,
        TurnPhase::SecondStone { .. } => 1,
    }
}

/// Direct window-mask reconstruction of "the mover could complete a window
/// this turn": some window holds `count(mover) >= 6 - remaining` stones with
/// zero opponent stones. This deliberately ignores empty-cell legality, so it
/// over-approximates true win-now positions: using it as a REJECTOR is sound
/// and conservative (notes deviation 2).
fn direct_own_win_now_upper(state: &RustHexoState) -> bool {
    let mover = state.current_player();
    let other = mover.other();
    let remaining = placements_remaining(state);
    state.board().windows().entries().any(|entry| {
        entry.count(other) == 0 && entry.count(mover).saturating_add(remaining) >= 6
    })
}

/// Z4 coupling-stability anchor (§3.4): an attacker stone or a root-binding
/// stone within hex distance eight of the placement.
fn anchored(state: &RustHexoState, claimant: Player, root_stones: &[HexCoord], mv: HexCoord) -> bool {
    state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|stone| state.board().get(*stone) == Some(claimant))
        .chain(root_stones.iter().copied())
        .any(|anchor| hex_distance(anchor, mv) <= 8)
}

fn window_has_claimant_stone(state: &RustHexoState, claimant: Player, key: WindowKey) -> bool {
    key.cells()
        .iter()
        .any(|cell| state.board().get(*cell) == Some(claimant))
}

fn window_defender_count(state: &RustHexoState, defender: Player, key: WindowKey) -> u32 {
    key.cells()
        .iter()
        .filter(|cell| state.board().get(**cell) == Some(defender))
        .count() as u32
}

fn window_is_all_empty(state: &RustHexoState, key: WindowKey) -> bool {
    key.cells().iter().all(|cell| state.board().get(*cell).is_none())
}

fn window_empty_cells(state: &RustHexoState, key: WindowKey) -> Vec<HexCoord> {
    key.cells()
        .iter()
        .copied()
        .filter(|cell| state.board().get(*cell).is_none())
        .collect()
}

/// Distance from a cell to a window: minimum over the six window cells.
fn window_distance(cell: HexCoord, key: WindowKey) -> u32 {
    key.cells()
        .iter()
        .map(|w| i32::from(hex_distance(cell, *w)) as u32)
        .min()
        .unwrap_or(u32::MAX)
}

fn sorted_legal_moves(state: &RustHexoState) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_key(*coord));
    legal
}

fn set_contains(sorted: &[HexCoord], coord: HexCoord) -> bool {
    sorted
        .binary_search_by_key(&coord_key(coord), |candidate| coord_key(*candidate))
        .is_ok()
}

// ---------------------------------------------------------------------------
// FHW gate geometry, classifiers, and reconstruction (design §3.2/§3.3),
// reimplemented verifier-side per the independent-derivation rule (this module
// never shares the finder/test-module code). The RC/WC and charged-role
// branches are implemented and unit-tested here; the ACCEPT path rejects any
// NonFrontierCovered gate edge before those branches can grant acceptance (see
// GateInfo), so they never license an accept in this lane.
// ---------------------------------------------------------------------------

/// Iterate the (in-range) cells of an inclusive radius-`radius` axial ball
/// around `center`. Out-of-range offsets are dropped for RC/WC (they only
/// shrink an intersection); FC re-derives directly and treats an out-of-range
/// ball cell as uncovered (conservative — see `frontier_covered`).
fn ball(center: HexCoord, radius: i32) -> Vec<HexCoord> {
    let mut out = Vec::new();
    for dq in -radius..=radius {
        let r_min = (-radius).max(-dq - radius);
        let r_max = radius.min(-dq + radius);
        for dr in r_min..=r_max {
            let q = i32::from(center.q) + dq;
            let r = i32::from(center.r) + dr;
            if let (Ok(q), Ok(r)) = (i16::try_from(q), i16::try_from(r)) {
                out.push(HexCoord { q, r });
            }
        }
    }
    out
}

/// Verifier-side ghost `G = P_Q + s`: the gate position with one representative
/// reply applied, plus the materialized `Lambda(G) = union B_8(x)` over
/// occupied `x`.
struct VGhost {
    state: RustHexoState,
    lambda: HashSet<CoordKey>,
}

impl VGhost {
    fn new(gate: &RustHexoState, s: HexCoord) -> Option<Self> {
        let mut state = gate.clone();
        state.apply_with_delta(Placement { coord: s }).ok()?;
        let mut lambda: HashSet<CoordKey> = HashSet::new();
        for &occ in state.board().occupied_cells() {
            for cell in ball(occ, 8) {
                lambda.insert(coord_key(cell));
            }
        }
        Some(Self { state, lambda })
    }

    fn in_lambda(&self, cell: HexCoord) -> bool {
        self.lambda.contains(&coord_key(cell))
    }

    /// `GI(G)(z)`: neither occupied nor legal in `G` (design §3.2).
    fn is_ghost_illegal(&self, z: HexCoord) -> bool {
        self.state.board().get(z).is_none() && !self.state.board().legal_moves().contains(z)
    }
}

/// FC predicate (design §3.3): `d == s` or every one of the 217 cells of
/// `B_8(d)` lies in `Lambda(G)`. Conservative: any `B_8(d)` cell out of
/// coordinate range (hence not in `Lambda`) fails FC.
fn frontier_covered(d: HexCoord, s: HexCoord, ghost: &VGhost) -> bool {
    if d == s {
        return true;
    }
    for dq in -8i32..=8 {
        let r_min = (-8i32).max(-dq - 8);
        let r_max = 8i32.min(-dq + 8);
        for dr in r_min..=r_max {
            let q = i32::from(d.q) + dq;
            let r = i32::from(d.r) + dr;
            match (i16::try_from(q), i16::try_from(r)) {
                (Ok(q), Ok(r)) if ghost.in_lambda(HexCoord { q, r }) => {}
                _ => return false,
            }
        }
    }
    true
}

/// RC predicate (design §3.3): `GI(G) ∩ B_8(d) ∩ B_{8(k-1)}(y)` is empty (the
/// inner ball is empty when `k == 0`).
fn rc_pass(d: HexCoord, y: HexCoord, k: u32, ghost: &VGhost) -> Option<bool> {
    if k == 0 {
        return Some(true);
    }
    let inner_radius = 8i32.checked_mul(i32::try_from(k.checked_sub(1)?).ok()?)?;
    let d_ball: HashSet<CoordKey> = ball(d, 8).into_iter().map(coord_key).collect();
    let empty = !ball(y, inner_radius)
        .into_iter()
        .any(|z| d_ball.contains(&coord_key(z)) && ghost.is_ghost_illegal(z));
    Some(empty)
}

/// WC predicate (design §3.3): `GI(G) ∩ B_8(d) ∩ B_{8(q-6)}(W)` is empty.
fn wc_pass(d: HexCoord, window: WindowKey, q: u32, ghost: &VGhost) -> Option<bool> {
    let radius = 8i32.checked_mul(i32::try_from(q.saturating_sub(6)).ok()?)?;
    let d_ball: HashSet<CoordKey> = ball(d, 8).into_iter().map(coord_key).collect();
    for w in window.cells() {
        for z in ball(w, radius) {
            if d_ball.contains(&coord_key(z)) && ghost.is_ghost_illegal(z) {
                return Some(false);
            }
        }
    }
    Some(true)
}

/// Classify a role row for pair `(d, s)` and a live role carried by `y` with
/// `k = f_cut(C_s, rho)` (design §3.3). Returns `(row, epsilon)` or `None` when
/// a mandatory condition fails (carrier not avoided, or the D22-N radius fails
/// on a charged ghost-illegal role) — fail-closed. `ghost` is required only on
/// the NonFrontierCovered branch; the accept path passes `None` because every
/// accepted edge is Exact/FC.
fn classify_role(
    edge_class: FhwEdgeClassV1,
    d: HexCoord,
    y: HexCoord,
    k: u32,
    ghost: Option<&VGhost>,
) -> Option<(FhwRoleRowV1, u8)> {
    if d == y {
        return None; // every row requires d to avoid the carrier
    }
    match edge_class {
        FhwEdgeClassV1::Exact | FhwEdgeClassV1::FrontierCovered => {
            Some((FhwRoleRowV1::ExactOrFcZero, 0))
        }
        FhwEdgeClassV1::NonFrontierCovered => {
            let ghost = ghost?;
            if ghost.is_ghost_illegal(y) {
                if rc_pass(d, y, k, ghost)? {
                    Some((FhwRoleRowV1::NonFcRcZero, 0))
                } else {
                    // epsilon = 1; mandatory D22-N radius dist(d,y) > 8k.
                    let radius = 8u32.checked_mul(k)?;
                    if u32::from(hex_distance(d, y).unsigned_abs()) > radius {
                        Some((FhwRoleRowV1::NonFcCharged, 1))
                    } else {
                        None
                    }
                }
            } else {
                Some((FhwRoleRowV1::NonFcCharged, 1))
            }
        }
    }
}

/// Window geometry at the gate position `P_Q`.
struct WindowGeomV {
    d_alive: bool,
    touched: bool,
    all_empty: bool,
    cnt_d: u32,
}

fn window_geom(gate: &RustHexoState, claimant: Player, window: WindowKey) -> WindowGeomV {
    let defender = claimant.other();
    let mut claimant_ct = 0u32;
    let mut defender_ct = 0u32;
    for cell in window.cells() {
        match gate.board().get(cell) {
            Some(p) if p == claimant => claimant_ct += 1,
            Some(p) if p == defender => defender_ct += 1,
            _ => {}
        }
    }
    let d_alive = claimant_ct == 0;
    WindowGeomV {
        d_alive,
        touched: d_alive && defender_ct >= 1,
        all_empty: claimant_ct == 0 && defender_ct == 0,
        cnt_d: defender_ct,
    }
}

fn edge_class_tag(class: FhwEdgeClassV1) -> u8 {
    match class {
        FhwEdgeClassV1::Exact => 0,
        FhwEdgeClassV1::FrontierCovered => 1,
        FhwEdgeClassV1::NonFrontierCovered => 2,
    }
}

fn role_row_tag(row: FhwRoleRowV1) -> u8 {
    match row {
        FhwRoleRowV1::ExactOrFcZero => 0,
        FhwRoleRowV1::NonFcRcZero => 1,
        FhwRoleRowV1::NonFcCharged => 2,
    }
}

fn kappa_row_tag(row: FhwKappaRowV1) -> u8 {
    match row {
        FhwKappaRowV1::NonDAlive => 0,
        FhwKappaRowV1::ExactOrFcNonIncident => 1,
        FhwKappaRowV1::ExactOrFcDirect => 2,
        FhwKappaRowV1::NonFcTouchedNonIncident => 3,
        FhwKappaRowV1::NonFcTouchedDirect => 4,
        FhwKappaRowV1::NonFcEmptyDirect => 5,
        FhwKappaRowV1::NonFcEmptyNonIncidentQlt6 => 6,
        FhwKappaRowV1::NonFcEmptyNonIncidentWcPass => 7,
        FhwKappaRowV1::NonFcEmptyNonIncidentWcFail => 8,
    }
}

fn guard_tag(guard: GuardResultV1) -> u8 {
    match guard {
        GuardResultV1::NotApplicable => 0,
        GuardResultV1::Pass => 1,
    }
}

/// Classify a `(d, s, W)` window row with `q = Q_cut(C_s, W)` (design §3.3's
/// ordered, mutually exclusive table). Returns `(row, kappa, guard)` or `None`
/// when a mandatory retained guard fails (a failed guard rejects even if the
/// finder wrote `Pass`). `ghost` is required only on the NonFC WC branch.
fn classify_window(
    edge_class: FhwEdgeClassV1,
    d: HexCoord,
    window: WindowKey,
    q: u32,
    geom: &WindowGeomV,
    ghost: Option<&VGhost>,
) -> Option<(FhwKappaRowV1, u8, GuardResultV1)> {
    use FhwKappaRowV1 as Row;
    use GuardResultV1 as Guard;

    if !geom.d_alive {
        return Some((Row::NonDAlive, 0, Guard::NotApplicable));
    }
    let d_in = window.contains(d);
    let exact_or_fc = matches!(
        edge_class,
        FhwEdgeClassV1::Exact | FhwEdgeClassV1::FrontierCovered
    );
    if exact_or_fc {
        if !d_in {
            return Some((Row::ExactOrFcNonIncident, 0, Guard::NotApplicable));
        }
        let guard_ok = if geom.touched {
            geom.cnt_d.checked_add(1)?.checked_add(q)? < 6
        } else {
            1u32.checked_add(q)? < 6
        };
        return guard_ok.then_some((Row::ExactOrFcDirect, 1, Guard::Pass));
    }
    // non-FC
    if geom.touched {
        if !d_in {
            return Some((Row::NonFcTouchedNonIncident, 0, Guard::NotApplicable));
        }
        let guard_ok = geom.cnt_d.checked_add(1)?.checked_add(q)? < 6;
        return guard_ok.then_some((Row::NonFcTouchedDirect, 1, Guard::Pass));
    }
    // non-FC, all-empty
    if d_in {
        let guard_ok = 1u32.checked_add(q)? < 6;
        return guard_ok.then_some((Row::NonFcEmptyDirect, 1, Guard::Pass));
    }
    if q < 6 {
        return Some((Row::NonFcEmptyNonIncidentQlt6, 0, Guard::NotApplicable));
    }
    let ghost = ghost?;
    if wc_pass(d, window, q, ghost)? {
        return Some((Row::NonFcEmptyNonIncidentWcPass, 0, Guard::Pass));
    }
    let virgin_radius = 8u32.checked_mul(1u32.checked_add(q)?.checked_sub(6)?)?;
    (window_distance(d, window) > virgin_radius)
        .then_some((Row::NonFcEmptyNonIncidentWcFail, 1, Guard::Pass))
}

/// Exact bounded transversal of `family` (each set = a threat's empties),
/// capped at `cap`. Returns the least `k <= cap` hitting every set, or
/// `cap + 1` above. Empty family -> 0; a set with no empties -> `cap + 1`
/// (unhittable). `F_Q \ d` (the sub-family not hit by `d`) is formed by the
/// caller. Since gates use `b <= 2`, `cap = 2` is exact for every use.
fn transversal_exact(family: &[Vec<HexCoord>], cap: u8) -> u8 {
    if family.is_empty() {
        return 0;
    }
    if family.iter().any(Vec::is_empty) {
        return cap.saturating_add(1);
    }
    let mut universe: Vec<HexCoord> = Vec::new();
    for set in family {
        for &c in set {
            if !universe.contains(&c) {
                universe.push(c);
            }
        }
    }
    if cap >= 1 && universe.iter().any(|c| family.iter().all(|s| s.contains(c))) {
        return 1;
    }
    if cap >= 2 {
        for i in 0..universe.len() {
            for j in (i + 1)..universe.len() {
                let (a, b) = (universe[i], universe[j]);
                if family.iter().all(|s| s.contains(&a) || s.contains(&b)) {
                    return 2;
                }
            }
        }
    }
    cap.saturating_add(1)
}

/// Reconstruct the FHW gate at `state` from the certificate node `gate`,
/// entirely from the replayed position (design §3.3). Every certificate field
/// is a claim compared against a fresh derivation. Returns the derived
/// `GateInfo` plus the ordered representative `(move, child)` edges to recurse
/// into. Rejects (returns `None`) on any mismatch, and — per the documented
/// accept-path narrowing — on any NonFrontierCovered edge.
fn reconstruct_gate(
    state: &RustHexoState,
    claimant: Player,
    gate: &crate::tss_verify::FhwGateNodeV1,
) -> Option<(GateInfo, Vec<(HexCoord, CertNodeId)>)> {
    // Eligibility (§3.3 preamble + R2).
    if state.current_player() == claimant
        || state.is_terminal()
        || matches!(state.phase(), TurnPhase::Opening)
        || direct_own_win_now_upper(state)
        || threats_shared::analyze(state).own_win_now
    {
        return None;
    }
    if gate.proof.schema_version != 1 || !gate.proof.authority.matches_compiled() {
        return None;
    }
    let b = placements_remaining(state);
    if !(1..=2).contains(&b) {
        return None;
    }

    // H_Q: canonical sorted/unique; count bound; each a real A-threat.
    let threats = &gate.proof.threats;
    if threats.is_empty() {
        return None;
    }
    let count_ok = match b {
        1 => threats.len() == 1,
        2 => (1..=3).contains(&threats.len()),
        _ => false,
    };
    if !count_ok {
        return None;
    }
    let mut previous: Option<WinId> = None;
    let mut family: Vec<Vec<HexCoord>> = Vec::with_capacity(threats.len());
    for &key in threats {
        let id = win_id(key);
        if previous.is_some_and(|prev| prev >= id) {
            return None; // noncanonical or duplicate
        }
        previous = Some(id);
        let cells = key.cells();
        let claimant_ct = cells
            .iter()
            .filter(|c| state.board().get(**c) == Some(claimant))
            .count();
        let defender_ct = cells
            .iter()
            .filter(|c| state.board().get(**c) == Some(claimant.other()))
            .count();
        let empties = window_empty_cells(state, key);
        if claimant_ct < 4 || defender_ct != 0 || empties.is_empty() {
            return None;
        }
        family.push(empties);
    }
    // Exact transversal == b.
    if transversal_exact(&family, 2) != b {
        return None;
    }

    // K = { d in Legal : transversal(F_Q \ d) <= b-1 }.
    let legal = sorted_legal_moves(state);
    let mut kernel: Vec<HexCoord> = Vec::new();
    for &d in &legal {
        let residual: Vec<Vec<HexCoord>> = family
            .iter()
            .filter(|set| !set.contains(&d))
            .cloned()
            .collect();
        if transversal_exact(&residual, 2) <= b.saturating_sub(1) {
            kernel.push(d);
        }
    }
    if kernel.is_empty() {
        return None;
    }
    // Every kernel reply applied must be nonterminal.
    for &d in &kernel {
        let mut probe = state.clone();
        let result = probe.apply_with_delta(Placement { coord: d }).ok()?.0;
        if result.outcome.is_some() {
            return None;
        }
    }
    let kernel_set: HashSet<CoordKey> = kernel.iter().map(|c| coord_key(*c)).collect();

    // R = representative moves (sorted-unique, subset K, one exact nonterminal
    // child each, phi(s)=s).
    let mut rep_child: HashMap<CoordKey, CertNodeId> = HashMap::new();
    let mut reps: Vec<HexCoord> = Vec::new();
    let mut previous: Option<CoordKey> = None;
    let mut rep_edges: Vec<(HexCoord, CertNodeId)> = Vec::new();
    for edge in &gate.representatives {
        let key = coord_key(edge.mv);
        if previous.is_some_and(|prev| prev >= key) {
            return None; // noncanonical or duplicate representative
        }
        previous = Some(key);
        if !kernel_set.contains(&key) {
            return None; // R must be a subset of K
        }
        let mut probe = state.clone();
        let result = probe.apply_with_delta(Placement { coord: edge.mv }).ok()?.0;
        if result.outcome.is_some() {
            return None; // representative child must be nonterminal
        }
        rep_child.insert(key, edge.child);
        reps.push(edge.mv);
        rep_edges.push((edge.mv, edge.child));
    }
    if reps.is_empty() {
        return None;
    }
    let rep_set: HashSet<CoordKey> = reps.iter().map(|c| coord_key(*c)).collect();

    // Map domain must equal K exactly; phi(d) in R; edge classes recomputed.
    // ACCEPT NARROWING: reject any NonFrontierCovered edge.
    if gate.proof.map.len() != kernel.len() {
        return None;
    }
    let mut phi: HashMap<CoordKey, HexCoord> = HashMap::new();
    let mut edge_class: HashMap<CoordKey, FhwEdgeClassV1> = HashMap::new();
    let mut ghosts: HashMap<CoordKey, VGhost> = HashMap::new();
    let mut previous: Option<CoordKey> = None;
    for entry in &gate.proof.map {
        let d_key = coord_key(entry.real_reply);
        if previous.is_some_and(|prev| prev >= d_key) {
            return None; // noncanonical or duplicate map order
        }
        previous = Some(d_key);
        if !kernel_set.contains(&d_key) {
            return None; // map domain must equal K
        }
        let s = entry.representative;
        if !rep_set.contains(&coord_key(s)) {
            return None; // phi(d) must be a representative
        }
        // Recompute edge class from geometry.
        let recomputed = if entry.real_reply == s {
            FhwEdgeClassV1::Exact
        } else {
            let ghost = ghosts
                .entry(coord_key(s))
                .or_insert(VGhost::new(state, s)?);
            if frontier_covered(entry.real_reply, s, ghost) {
                FhwEdgeClassV1::FrontierCovered
            } else {
                FhwEdgeClassV1::NonFrontierCovered
            }
        };
        if recomputed != entry.edge_class {
            return None; // stored edge class must match the derivation
        }
        if recomputed == FhwEdgeClassV1::NonFrontierCovered {
            return None; // accept-path narrowing (fail-closed)
        }
        phi.insert(d_key, s);
        edge_class.insert(d_key, recomputed);
    }
    // The map domain has exactly |K| canonical distinct d's == K.
    if phi.len() != kernel.len() {
        return None;
    }
    // phi(s) = s for every representative, realized as an Exact self-edge.
    for &s in &reps {
        match phi.get(&coord_key(s)) {
            Some(mapped) if *mapped == s => {}
            _ => return None,
        }
        if edge_class.get(&coord_key(s)) != Some(&FhwEdgeClassV1::Exact) {
            return None;
        }
    }

    // escape_resolution_ply = p(Q) + b + 2 (R1) and byte-equal to the claim.
    let escape_ply = state
        .placements_made()
        .checked_add(u32::from(b))?
        .checked_add(2)?;
    if escape_ply != gate.proof.escape_resolution_ply {
        return None;
    }

    let info = GateInfo {
        b,
        threats: threats.clone(),
        kernel,
        reps,
        phi,
        rep_child,
        edge_class,
        escape_ply,
    };
    Some((info, rep_edges))
}

/// Derive the window row for a gate pair `(d, W)` with `q = Q_cut(C_s, W)`
/// (the representative-child Q_cut). Used by the gate window clock, the check
/// pass, and the derived digest. `s`/edge class come from the reconstructed
/// `GateInfo`; the ghost is not required because every accepted edge is
/// Exact/FC.
fn derive_gate_window_row(
    ctx: &G2Context<'_>,
    gate_id: CertNodeId,
    d: HexCoord,
    key: WindowKey,
    child_q: u32,
) -> Option<(FhwKappaRowV1, u8, GuardResultV1)> {
    let info = ctx.gates.get(&gate_id)?;
    let edge_class = *info.edge_class.get(&coord_key(d))?;
    let state = &ctx.states[gate_id as usize];
    let geom = window_geom(state, ctx.claimant, key);
    classify_window(edge_class, d, key, child_q, &geom, None)
}

/// Derive the role row for a gate pair `(d, s)` and a live role carried by `y`
/// with `k = f_cut(C_s, rho)` (the representative-child f_cut). Ghost not
/// required on the Exact/FC accept path.
fn derive_gate_role_row(
    ctx: &G2Context<'_>,
    gate_id: CertNodeId,
    d: HexCoord,
    y: HexCoord,
    child_f: u32,
) -> Option<(FhwRoleRowV1, u8)> {
    let info = ctx.gates.get(&gate_id)?;
    let edge_class = *info.edge_class.get(&coord_key(d))?;
    classify_role(edge_class, d, y, child_f, None)
}

// ---------------------------------------------------------------------------
// Top-level verification (design §3.1, narrowed as documented).
// ---------------------------------------------------------------------------

pub(crate) fn verify_group2_certificate(
    state: &RustHexoState,
    cert: &TssCertificate,
    claimed: ProofStatus,
) -> bool {
    verify_group2_impl(state, cert, claimed).is_some()
}

fn verify_group2_impl(
    state: &RustHexoState,
    cert: &TssCertificate,
    claimed: ProofStatus,
) -> Option<()> {
    // Unchanged root/status/claimant/arena/horizon checks (§3.1 step 1).
    if claimed == ProofStatus::Unknown || cert.root != RootBinding::from_state(state) {
        return None;
    }
    let expected_claimant = match claimed {
        ProofStatus::Win => state.current_player(),
        ProofStatus::Loss => state.current_player().other(),
        ProofStatus::Unknown => return None,
    };
    if cert.claimant != expected_claimant || !validate_arena_for_group2(cert) {
        return None;
    }
    let meta = certificate_metadata_for_group2(cert)?;
    // R1: derived T includes every gate escape deadline (folded into
    // `certificate_metadata`) and must fit the declared semantic horizon.
    if meta.derived_t > cert.semantic_horizon {
        return None;
    }
    // R1 second clause: every gate's escape deadline must individually fit
    // the declared horizon.
    for node in &cert.nodes {
        if let CertNode::FhwGateV1(gate) = node {
            if gate.proof.escape_resolution_ply > cert.semantic_horizon {
                return None;
            }
        }
    }
    // R2: the root position of any certificate containing a new-class node is
    // post-opening — explicit structural rule, not the accidental Z4 vacuity.
    if matches!(cert.root.phase, TurnPhase::Opening) {
        return None;
    }

    // Narrow-v1 structural preflight (§2.3).
    preflight_structure(cert)?;

    // Bind one exact state to every node, run the per-node direct checks, and
    // reconstruct every FHW gate (§3.3).
    let mut ctx = build_context(state, cert)?;

    // Postorder derivations: D14 B, subtree resolution T, live-role clocks
    // (incl. gate paired f_cut + checkpoint roles).
    derive_budgets_and_roles(&mut ctx)?;

    // Window demand fixed point + Q_cut/E_full evaluation (incl. gate
    // direct-18 seeds and the paired gate clock).
    derive_window_demands(&mut ctx)?;

    // Per-Group-2-node class rules, zone coverage, and stored-scalar equality.
    check_group2_nodes(&mut ctx)?;

    // Per-gate role/window row equality + Cartesian demand completeness (§3.3).
    check_gate_nodes(&mut ctx)?;

    // Digest recomputation and comparison (§2.4). Never an acceptance oracle
    // on its own — everything semantic above has already been re-derived —
    // but a mismatch rejects.
    check_digests(&mut ctx)?;

    Some(())
}

/// §2.3 structural rules that need no replay: exact tree shape, no mixing
/// with legacy zone/dispatch/commutation machinery, schema and authority
/// bytes, canonical edge order.
fn preflight_structure(cert: &TssCertificate) -> Option<()> {
    let nodes = cert.nodes.len();
    let mut indegree = vec![0u32; nodes];
    for node in &cert.nodes {
        match node {
            CertNode::Choice { child, .. } => {
                indegree[*child as usize] = indegree[*child as usize].checked_add(1)?;
            }
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                // Class rule 2/3: a legacy Universal is admissible only as a
                // plain full-enumeration node (the full-set check itself needs
                // the replayed position and happens in build_context).
                if *implicit_dispatch || zone.is_some() || !commutations.is_empty() {
                    return None;
                }
                let mut previous: Option<CoordKey> = None;
                for edge in edges {
                    let key = coord_key(edge.mv);
                    if previous.is_some_and(|prev| prev >= key) {
                        return None;
                    }
                    previous = Some(key);
                    indegree[edge.child as usize] = indegree[edge.child as usize].checked_add(1)?;
                }
            }
            CertNode::UniversalGroup2V1(g2) => {
                if g2.proof.schema_version != 1 || !g2.proof.authority.matches_compiled() {
                    return None;
                }
                if g2.edges.is_empty() {
                    return None;
                }
                let mut previous: Option<CoordKey> = None;
                for edge in &g2.edges {
                    let key = coord_key(edge.mv);
                    if previous.is_some_and(|prev| prev >= key) {
                        return None;
                    }
                    previous = Some(key);
                    indegree[edge.child as usize] = indegree[edge.child as usize].checked_add(1)?;
                }
            }
            CertNode::FhwGateV1(gate) => {
                if gate.proof.schema_version != 1 || !gate.proof.authority.matches_compiled() {
                    return None;
                }
                if gate.representatives.is_empty() {
                    return None;
                }
                // Representatives canonical sorted-unique by move.
                let mut previous: Option<CoordKey> = None;
                for edge in &gate.representatives {
                    let key = coord_key(edge.mv);
                    if previous.is_some_and(|prev| prev >= key) {
                        return None;
                    }
                    previous = Some(key);
                    indegree[edge.child as usize] = indegree[edge.child as usize].checked_add(1)?;
                }
                // Threats canonical sorted-unique by window key.
                let mut previous: Option<(u8, i16, i16)> = None;
                for key in &gate.proof.threats {
                    let sort_key = window_sort_key(*key);
                    if previous.is_some_and(|prev| prev >= sort_key) {
                        return None;
                    }
                    previous = Some(sort_key);
                }
                // Map canonical sorted-unique by real reply; each role list
                // canonical by role key; each window list canonical by window
                // key. (Full semantic checks happen after replay.)
                let mut previous: Option<CoordKey> = None;
                for entry in &gate.proof.map {
                    let key = coord_key(entry.real_reply);
                    if previous.is_some_and(|prev| prev >= key) {
                        return None;
                    }
                    previous = Some(key);
                    let mut prev_role: Option<RoleKeyV1> = None;
                    for claim in &entry.roles {
                        if let Some(prev) = &prev_role {
                            if role_key_order(prev) >= role_key_order(&claim.role) {
                                return None;
                            }
                        }
                        prev_role = Some(claim.role.clone());
                    }
                    let mut prev_win: Option<(u8, i16, i16)> = None;
                    for claim in &entry.windows {
                        let sort_key = window_sort_key(claim.window);
                        if prev_win.is_some_and(|prev| prev >= sort_key) {
                            return None;
                        }
                        prev_win = Some(sort_key);
                    }
                }
            }
            CertNode::Loss { witnesses, .. } => {
                // Canonical sorted-unique witness order inside the new class.
                let mut previous: Option<(u8, i16, i16)> = None;
                for key in witnesses {
                    let sort_key = window_sort_key(*key);
                    if previous.is_some_and(|prev| prev >= sort_key) {
                        return None;
                    }
                    previous = Some(sort_key);
                }
            }
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => {}
        }
    }
    // Exact reachable tree: root indegree zero, every other node exactly one.
    // (Acyclicity and full reachability were established by validate_arena.)
    for (index, count) in indegree.iter().enumerate() {
        let expected = u32::from(index != cert.root_node as usize);
        if *count != expected {
            return None;
        }
    }
    Some(())
}

/// Replay the tree, binding one exact state per node, running every per-node
/// direct D9 check (§3.2) as the class requires.
fn build_context<'a>(root: &RustHexoState, cert: &'a TssCertificate) -> Option<G2Context<'a>> {
    let claimant = cert.claimant;
    let node_count = cert.nodes.len();
    let mut ctx = G2Context {
        cert,
        claimant,
        states: Vec::new(),
        children: vec![Vec::new(); node_count],
        postorder: Vec::with_capacity(node_count),
        b_local: vec![0; node_count],
        t_sub: vec![0; node_count],
        roles: Vec::new(),
        demands: vec![Vec::new(); node_count],
        window_clock: HashMap::new(),
        derived_k: HashMap::new(),
        required: HashMap::new(),
        gates: HashMap::new(),
        work: 0,
    };
    // Tree replay: recursion with explicit depth cap. Each node is visited
    // exactly once (indegree checks), so `states` can be dense.
    let mut states: Vec<Option<RustHexoState>> = vec![None; node_count];
    let root_stones = cert.root.occupancy.clone();
    replay_node(
        cert,
        cert.root_node,
        root.clone(),
        claimant,
        &root_stones,
        0,
        &mut states,
        &mut ctx,
    )?;
    let states = states.into_iter().collect::<Option<Vec<_>>>()?;
    ctx.states = states;
    // Postorder (children before parents) over the tree.
    let mut order = Vec::with_capacity(node_count);
    let mut stack = vec![(cert.root_node, false)];
    while let Some((id, exiting)) = stack.pop() {
        if exiting {
            order.push(id);
            continue;
        }
        stack.push((id, true));
        for (_, child) in &ctx.children[id as usize] {
            stack.push((*child, false));
        }
        if stack.len() > node_count.checked_mul(2)?.checked_add(4)? {
            return None;
        }
    }
    if order.len() != node_count {
        return None;
    }
    ctx.postorder = order;
    Some(ctx)
}

#[allow(clippy::too_many_arguments)]
fn replay_node(
    cert: &TssCertificate,
    id: CertNodeId,
    state: RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    depth: usize,
    states: &mut [Option<RustHexoState>],
    ctx: &mut G2Context<'_>,
) -> Option<()> {
    if depth > MAX_CERT_DEPTH {
        return None;
    }
    ctx.charge(1)?;
    if states.get(id as usize)?.is_some() {
        return None; // tree property violated
    }
    match cert.nodes.get(id as usize)? {
        CertNode::OrCompletion {
            mv,
            witness,
            completion_ply,
        } => {
            check_or_completion(&state, claimant, root_stones, *mv, *witness, *completion_ply)?;
            ctx.t_sub[id as usize] = *completion_ply;
        }
        CertNode::Win {
            witness,
            count,
            budget,
            resolution_ply,
        } => {
            check_win_leaf(
                &state,
                claimant,
                root_stones,
                *witness,
                *count,
                *budget,
                *resolution_ply,
            )?;
            ctx.t_sub[id as usize] = *resolution_ply;
        }
        CertNode::Loss {
            witnesses,
            resolution_ply,
        } => {
            let b = check_loss_leaf(&state, claimant, root_stones, witnesses, *resolution_ply)?;
            ctx.b_local[id as usize] = u32::from(b);
            ctx.t_sub[id as usize] = *resolution_ply;
        }
        CertNode::Choice { mv, child } => {
            if state.current_player() != claimant
                || state.is_terminal()
                || !anchored(&state, claimant, root_stones, *mv)
            {
                return None;
            }
            let mut next = state.clone();
            let result = next.apply_with_delta(Placement { coord: *mv }).ok()?.0;
            if result.outcome.is_some() {
                return None;
            }
            ctx.children[id as usize].push((*mv, *child));
            replay_node(cert, *child, next, claimant, root_stones, depth + 1, states, ctx)?;
        }
        CertNode::Universal { edges, .. } => {
            // Legacy full-enumeration AND inside the new class: defender to
            // move, nonterminal, no win-now, and the edge set must be exactly
            // the full sorted legal set.
            if state.current_player() == claimant
                || state.is_terminal()
                || matches!(state.phase(), TurnPhase::Opening)
                || direct_own_win_now_upper(&state)
                || threats_shared::analyze(&state).own_win_now
            {
                return None;
            }
            let legal = sorted_legal_moves(&state);
            ctx.charge(legal.len() as u64)?;
            let moves: Vec<HexCoord> = edges.iter().map(|edge| edge.mv).collect();
            if moves != legal {
                return None;
            }
            for edge in edges {
                let mut next = state.clone();
                let result = next.apply_with_delta(Placement { coord: edge.mv }).ok()?.0;
                if result.outcome.is_some() {
                    return None;
                }
                ctx.children[id as usize].push((edge.mv, edge.child));
                replay_node(
                    cert,
                    edge.child,
                    next,
                    claimant,
                    root_stones,
                    depth + 1,
                    states,
                    ctx,
                )?;
            }
        }
        CertNode::UniversalGroup2V1(g2) => {
            // Class rule 4 (§2.3): post-opening, defender-to-move,
            // nonterminal, not own_win_now, and an exactly reconstructed
            // current k < b.
            if state.current_player() == claimant
                || state.is_terminal()
                || matches!(state.phase(), TurnPhase::Opening)
                || direct_own_win_now_upper(&state)
                || threats_shared::analyze(&state).own_win_now
            {
                return None;
            }
            let b = placements_remaining(&state);
            if !(1..=2).contains(&b) {
                return None;
            }
            let k = derive_exact_k(&state, claimant, ctx)?;
            if u32::from(k) >= u32::from(b) {
                return None;
            }
            ctx.derived_k.insert(id, k);
            let legal = sorted_legal_moves(&state);
            for edge in &g2.edges {
                if !set_contains(&legal, edge.mv) {
                    return None;
                }
                let mut next = state.clone();
                let result = next.apply_with_delta(Placement { coord: edge.mv }).ok()?.0;
                if result.outcome.is_some() {
                    return None;
                }
                ctx.children[id as usize].push((edge.mv, edge.child));
                replay_node(
                    cert,
                    edge.child,
                    next,
                    claimant,
                    root_stones,
                    depth + 1,
                    states,
                    ctx,
                )?;
            }
        }
        CertNode::FhwGateV1(gate) => {
            // §3.3 gate reconstruction, entirely from the replayed position.
            let (info, rep_edges) = reconstruct_gate(&state, claimant, gate)?;
            ctx.charge(rep_edges.len() as u64)?;
            ctx.gates.insert(id, info);
            for (mv, child) in rep_edges {
                let mut next = state.clone();
                let result = next.apply_with_delta(Placement { coord: mv }).ok()?.0;
                if result.outcome.is_some() {
                    return None;
                }
                ctx.children[id as usize].push((mv, child));
                replay_node(cert, child, next, claimant, root_stones, depth + 1, states, ctx)?;
            }
        }
    }
    states[id as usize] = Some(state);
    Some(())
}

/// §3.2: enumerate the COMPLETE current claimant-threat family from the
/// replayed board and derive `k = tau` on the only accepted side of the
/// threshold: 0 iff empty, 1 iff every member shares a common cell, else >=2
/// (which rejects at both accepted budgets). Never trusts a capped
/// `min_hitting_set`.
fn derive_exact_k(state: &RustHexoState, claimant: Player, ctx: &mut G2Context<'_>) -> Option<u8> {
    let defender = claimant.other();
    let mut family: Vec<Vec<HexCoord>> = Vec::new();
    for entry in state.board().windows().entries() {
        ctx.charge(1)?;
        if entry.count(defender) == 0 && entry.count(claimant) >= 4 {
            let empties = entry.empty_cells();
            if empties.is_empty() {
                // A filled window is terminal; the node was required
                // nonterminal, so this is corrupt state.
                return None;
            }
            family.push(empties);
        }
    }
    if family.is_empty() {
        return Some(0);
    }
    let mut common = family[0].clone();
    for member in &family[1..] {
        ctx.charge(member.len() as u64)?;
        common.retain(|cell| member.contains(cell));
        if common.is_empty() {
            break;
        }
    }
    Some(if common.is_empty() { 2 } else { 1 })
}

fn check_or_completion(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    mv: HexCoord,
    witness: WindowKey,
    completion_ply: u32,
) -> Option<()> {
    if state.current_player() != claimant
        || state.is_terminal()
        || !witness.contains(mv)
        || !anchored(state, claimant, root_stones, mv)
        || completion_ply != state.placements_made().checked_add(1)?
    {
        return None;
    }
    let mut next = state.clone();
    let result = next.apply_with_delta(Placement { coord: mv }).ok()?.0;
    let outcome = result.outcome?;
    if outcome.winner != claimant {
        return None;
    }
    // Direct D9 mask check: the named window is completely claimant-filled.
    let filled = witness.cells().iter().all(|cell| {
        next.board().get(*cell) == Some(claimant)
    });
    filled.then_some(())
}

fn check_win_leaf(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    witness: WindowKey,
    count: u8,
    budget: u8,
    resolution_ply: u32,
) -> Option<()> {
    if state.is_terminal() || state.current_player() != claimant {
        return None;
    }
    let actual_budget = placements_remaining(state);
    let cells = witness.cells();
    let claimant_count = cells
        .iter()
        .filter(|cell| state.board().get(**cell) == Some(claimant))
        .count() as u8;
    let defender_count = cells
        .iter()
        .filter(|cell| state.board().get(**cell) == Some(claimant.other()))
        .count() as u8;
    if defender_count != 0 || claimant_count != count || budget != actual_budget {
        return None;
    }
    let empties: Vec<HexCoord> = cells
        .iter()
        .copied()
        .filter(|cell| state.board().get(*cell).is_none())
        .collect();
    let expected_resolution = match count {
        5 => state.placements_made().checked_add(1)?,
        4 if actual_budget == 2 => state.placements_made().checked_add(2)?,
        _ => return None,
    };
    if resolution_ply != expected_resolution {
        return None;
    }
    if !empties
        .iter()
        .all(|mv| anchored(state, claimant, root_stones, *mv))
    {
        return None;
    }
    // Replay the one/two legal continuations directly (§3.2).
    let mut replay = state.clone();
    let mut final_outcome = None;
    for mv in &empties {
        if final_outcome.is_some() {
            return None;
        }
        let result = replay.apply_with_delta(Placement { coord: *mv }).ok()?.0;
        final_outcome = result.outcome;
    }
    (final_outcome?.winner == claimant).then_some(())
}

/// Direct LOSS-leaf check. Returns the exact defender budget `b`.
fn check_loss_leaf(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    witnesses: &[WindowKey],
    resolution_ply: u32,
) -> Option<u8> {
    if state.is_terminal() || state.current_player() == claimant || witnesses.is_empty() {
        return None;
    }
    if direct_own_win_now_upper(state) || threats_shared::analyze(state).own_win_now {
        return None;
    }
    let b = placements_remaining(state);
    let mut empties = Vec::with_capacity(witnesses.len());
    for &key in witnesses {
        let cells = key.cells();
        let claimant_count = cells
            .iter()
            .filter(|cell| state.board().get(**cell) == Some(claimant))
            .count();
        let defender_count = cells
            .iter()
            .filter(|cell| state.board().get(**cell) == Some(claimant.other()))
            .count();
        if defender_count != 0 || claimant_count < 4 {
            return None;
        }
        let empty: Vec<HexCoord> = cells
            .iter()
            .copied()
            .filter(|cell| state.board().get(*cell).is_none())
            .collect();
        if empty.is_empty()
            || !empty
                .iter()
                .all(|mv| anchored(state, claimant, root_stones, *mv))
        {
            return None;
        }
        empties.push(empty);
    }
    // Exact tau > b at b in {1, 2}: no single cell (b=1) or no pair of cells
    // (b=2) hits every named member.
    if !family_hitting_exceeds(&empties, b) {
        return None;
    }
    let expected = state
        .placements_made()
        .checked_add(u32::from(b))?
        .checked_add(2)?;
    (resolution_ply == expected).then_some(b)
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

/// Postorder pass: D14 full scalar B, subtree resolution T, and the live-role
/// clock map (f_cut == r_full on gate-free trees, §3.2).
fn derive_budgets_and_roles(ctx: &mut G2Context<'_>) -> Option<()> {
    let mut roles: Vec<HashMap<RoleId, (u32, u32)>> = vec![HashMap::new(); ctx.cert.nodes.len()];
    let mut total_roles = 0usize;
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let mut map: HashMap<RoleId, (u32, u32)> = HashMap::new();
        match &ctx.cert.nodes[index] {
            CertNode::OrCompletion { mv, .. } => {
                map.insert(
                    RoleId::OrCompletionMove {
                        node: id,
                        cell: coord_key(*mv),
                    },
                    (0, 0),
                );
                ctx.b_local[index] = 0;
            }
            CertNode::Win { witness, .. } => {
                for cell in window_empty_cells(&ctx.states[index], *witness) {
                    map.insert(
                        RoleId::LeafEmpty {
                            node: id,
                            witness: win_id(*witness),
                            cell: coord_key(cell),
                        },
                        (0, 0),
                    );
                }
                ctx.b_local[index] = 0;
            }
            CertNode::Loss { witnesses, .. } => {
                for key in witnesses {
                    for cell in window_empty_cells(&ctx.states[index], *key) {
                        map.insert(
                            RoleId::LeafEmpty {
                                node: id,
                                witness: win_id(*key),
                                cell: coord_key(cell),
                            },
                            (0, 0),
                        );
                    }
                }
                // b_local was set during replay (LOSS retains the remainder).
            }
            CertNode::Choice { mv, child } => {
                map.insert(
                    RoleId::ChoiceMove {
                        node: id,
                        cell: coord_key(*mv),
                    },
                    (0, 0),
                );
                for (role, clocks) in &roles[*child as usize] {
                    // Ordinary OR while live: pass-through (both clocks).
                    if map.insert(*role, *clocks).is_some() {
                        return None;
                    }
                }
                ctx.b_local[index] = ctx.b_local[*child as usize];
                ctx.t_sub[index] = ctx.t_sub[*child as usize];
            }
            CertNode::Universal { .. } | CertNode::UniversalGroup2V1(_) => {
                let mut maximum_b = 0u32;
                let mut maximum_t = 0u32;
                for (_, child) in &ctx.children[index] {
                    let child_index = *child as usize;
                    maximum_b = maximum_b.max(ctx.b_local[child_index]);
                    maximum_t = maximum_t.max(ctx.t_sub[child_index]);
                    for (role, (r_full, f_cut)) in &roles[child_index] {
                        // Ordinary AND: 1 + child clock for BOTH. In a tree each
                        // role is reachable below exactly one child, so a repeat
                        // key is a structural corruption.
                        let bumped = (r_full.checked_add(1)?, f_cut.checked_add(1)?);
                        if map.insert(*role, bumped).is_some() {
                            return None;
                        }
                    }
                }
                ctx.b_local[index] = maximum_b.checked_add(1)?;
                ctx.t_sub[index] = maximum_t;
            }
            CertNode::FhwGateV1(gate) => {
                // Clone the small data we need, then release the cert borrow.
                let threats: Vec<WindowKey> = gate.proof.threats.clone();
                let info = ctx.gates.get(&id)?;
                let b = info.b;
                let escape_ply = info.escape_ply;
                let kernel = info.kernel.clone();
                let phi = info.phi.clone();
                let children = ctx.children[index].clone(); // (s_move, child_id)
                if children.is_empty() {
                    return None;
                }
                // D14 B(Q) = 1 + max_{s in R} B(C_s); require B(Q) >= b.
                let mut maximum_b = 0u32;
                let mut maximum_t = escape_ply;
                for (_, child) in &children {
                    let ci = *child as usize;
                    maximum_b = maximum_b.max(ctx.b_local[ci]);
                    maximum_t = maximum_t.max(ctx.t_sub[ci]);
                }
                let b_q = maximum_b.checked_add(1)?;
                if b_q < u32::from(b) {
                    return None;
                }
                ctx.b_local[index] = b_q;
                ctx.t_sub[index] = maximum_t;
                // Checkpoint roles: an empty of every named threat, clock 0,
                // discharged at the gate.
                for &w in &threats {
                    for cell in window_empty_cells(&ctx.states[index], w) {
                        let role = RoleId::Checkpoint {
                            gate: id,
                            threat: win_id(w),
                            cell: coord_key(cell),
                        };
                        if map.insert(role, (0, 0)).is_some() {
                            return None;
                        }
                    }
                }
                // Paired f_cut over the roles below each representative child:
                // f_cut(Q,rho) = f_cut(C_s,rho) + max_{d: phi(d)=s} eps(d,rho).
                for (s_move, child_id) in &children {
                    let mapped_ds: Vec<HexCoord> = kernel
                        .iter()
                        .copied()
                        .filter(|d| phi.get(&coord_key(*d)) == Some(s_move))
                        .collect();
                    if mapped_ds.is_empty() {
                        return None; // every representative maps at least itself
                    }
                    for (role, (child_r, child_f)) in &roles[*child_id as usize] {
                        let carrier = role.carrier();
                        let y = HexCoord {
                            q: carrier.0,
                            r: carrier.1,
                        };
                        let mut max_eps = 0u32;
                        for &d in &mapped_ds {
                            let (_row, eps) = derive_gate_role_row(ctx, id, d, y, *child_f)?;
                            max_eps = max_eps.max(u32::from(eps));
                        }
                        // r_full takes the full 1+child charge; f_cut takes
                        // child + max epsilon. f_cut <= r_full holds.
                        let r_q = child_r.checked_add(1)?;
                        let f_q = child_f.checked_add(max_eps)?;
                        if f_q > r_q {
                            return None;
                        }
                        if map.insert(*role, (r_q, f_q)).is_some() {
                            return None;
                        }
                    }
                }
            }
        }
        ctx.charge(map.len() as u64)?;
        total_roles = total_roles.checked_add(map.len())?;
        if total_roles > MAX_G2_ROLES {
            return None;
        }
        roles[index] = map;
    }
    ctx.roles = roles;
    Some(())
}

/// Compute the PAIR `(Q_cut, E_full)` at `(node, key)` with memoization
/// (design §3.2/§3.3). On gate-free nodes the two coincide; only the FHW gate
/// clauses diverge. Enforces `Q_cut <= E_full <= B` on every evaluated pair.
fn window_clock(ctx: &mut G2Context<'_>, id: CertNodeId, key: WindowKey) -> Option<(u32, u32)> {
    if let Some(value) = ctx.window_clock.get(&(id, win_id(key))) {
        return Some(*value);
    }
    ctx.charge(1)?;
    let index = id as usize;
    let (q_cut, e_full) = if window_has_claimant_stone(&ctx.states[index], ctx.claimant, key) {
        // Non-D-alive: permanence stop (first clause, has precedence).
        (0, 0)
    } else {
        match &ctx.cert.nodes[index] {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => (0, 0),
            CertNode::Loss { .. } => (ctx.b_local[index], ctx.b_local[index]),
            CertNode::Choice { mv, child } => {
                if key.contains(*mv) {
                    // OR placement entering W.
                    (0, 0)
                } else {
                    window_clock(ctx, *child, key)?
                }
            }
            CertNode::Universal { .. } | CertNode::UniversalGroup2V1(_) => {
                let children = ctx.children[index].clone();
                if children.is_empty() {
                    return None;
                }
                let mut max_q = 0u32;
                let mut max_e = 0u32;
                for (_, child) in children {
                    let (q, e) = window_clock(ctx, child, key)?;
                    max_q = max_q.max(q);
                    max_e = max_e.max(e);
                }
                (max_q.checked_add(1)?, max_e.checked_add(1)?)
            }
            CertNode::FhwGateV1(_) => gate_window_clock(ctx, id, key)?,
        }
    };
    // Q_cut <= E_full <= B on every evaluated pair (design §3.2).
    if q_cut > e_full || e_full > ctx.b_local[index] {
        return None;
    }
    ctx.window_clock.insert((id, win_id(key)), (q_cut, e_full));
    Some((q_cut, e_full))
}

/// Gate clause of the paired window clock (design §3.3):
///   E_full(Q,W) = max{ b, max_{d in K} (1 + E_full(C_phi(d), W)) }
///   Q_cut(Q,W)  = max{ b, max_{d in K} (kappa(d,W) + Q_cut(C_phi(d), W)) }
/// The non-D-alive permanence stop was already applied by the caller.
fn gate_window_clock(
    ctx: &mut G2Context<'_>,
    id: CertNodeId,
    key: WindowKey,
) -> Option<(u32, u32)> {
    let b = u32::from(ctx.gates.get(&id)?.b);
    // Child clocks first (mutable recursion), collected with the real reply.
    let kernel = ctx.gates.get(&id)?.kernel.clone();
    let mut child_pairs: Vec<(HexCoord, u32, u32)> = Vec::with_capacity(kernel.len());
    for d in kernel {
        let s = *ctx.gates.get(&id)?.phi.get(&coord_key(d))?;
        let child = *ctx.gates.get(&id)?.rep_child.get(&coord_key(s))?;
        let (q_child, e_child) = window_clock(ctx, child, key)?;
        child_pairs.push((d, q_child, e_child));
    }
    // Now derive kappa per d (pure, immutable borrows).
    let mut max_q = b;
    let mut max_e = b;
    for (d, q_child, e_child) in child_pairs {
        let (_row, kappa, _guard) = derive_gate_window_row(ctx, id, d, key, q_child)?;
        max_q = max_q.max(u32::from(kappa).checked_add(q_child)?);
        max_e = max_e.max(1u32.checked_add(e_child)?);
    }
    Some((max_q, max_e))
}

/// Seed and propagate the ordinary window demands (§3.2): all D-alive touched
/// windows at every Group-2 node plus the finite all-empty superset when
/// B >= 6; propagation stops below an OR entering W, at typed leaves, and at
/// the first non-D-alive node (recorded there with clock zero).
fn derive_window_demands(ctx: &mut G2Context<'_>) -> Option<()> {
    // Seeds per Group-2 node.
    let mut seeds: Vec<(CertNodeId, WindowKey, u8)> = Vec::new();
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        if matches!(ctx.cert.nodes[index], CertNode::FhwGateV1(_)) {
            // Gate demand seeding: the 18 length-six windows through every
            // real reply d in K (§3.2). (Incoming ordinary keys arrive by
            // downward propagation; no gate-local WC keys on the Exact/FC
            // accept path.)
            let kernel = ctx.gates.get(&id)?.kernel.clone();
            for d in kernel {
                ctx.charge(19)?;
                for axis in Axis::ALL {
                    let vector = axis.vector();
                    for offset in 0..6i16 {
                        let start = HexCoord {
                            q: d.q.checked_sub(vector.q.checked_mul(offset)?)?,
                            r: d.r.checked_sub(vector.r.checked_mul(offset)?)?,
                        };
                        seeds.push((id, WindowKey { start, axis }, SOURCE_DIRECT18));
                    }
                }
            }
            continue;
        }
        if !matches!(ctx.cert.nodes[index], CertNode::UniversalGroup2V1(_)) {
            continue;
        }
        let state = ctx.states[index].clone();
        let defender = ctx.claimant.other();
        for entry in state.board().windows().entries() {
            ctx.charge(1)?;
            if entry.count(ctx.claimant) == 0 && entry.count(defender) >= 1 {
                seeds.push((id, entry.key(), SOURCE_TOUCHED));
            }
        }
        let b = ctx.b_local[index];
        if b >= 6 {
            let radius = 8u32.checked_mul(b.checked_sub(6)?)?;
            let radius = i32::try_from(radius).ok()?;
            let legal = sorted_legal_moves(&state);
            let mut candidate_windows: Vec<WindowKey> = Vec::new();
            for c in &legal {
                // Cells x with dist(x, c) <= radius, then the 18 windows
                // through each x. d(c, W) <= radius iff W holds such a cell.
                // Charge the square-box enumeration before running it so an
                // adversarially large derived budget rejects instead of
                // spinning.
                let side = u64::try_from(radius.checked_mul(2)?.checked_add(1)?).ok()?;
                ctx.charge(side.checked_mul(side)?.checked_mul(19)?)?;
                let mut x_cells = Vec::new();
                for dq in -radius..=radius {
                    for dr in -radius..=radius {
                        let q = i32::from(c.q).checked_add(dq)?;
                        let r = i32::from(c.r).checked_add(dr)?;
                        let cell = HexCoord {
                            q: i16::try_from(q).ok()?,
                            r: i16::try_from(r).ok()?,
                        };
                        if i32::from(hex_distance(*c, cell)) <= radius {
                            x_cells.push(cell);
                        }
                    }
                }
                ctx.charge(x_cells.len() as u64)?;
                for x in x_cells {
                    for axis in Axis::ALL {
                        for offset in 0..6i16 {
                            let start = HexCoord {
                                q: x.q.checked_sub(axis.vector().q.checked_mul(offset)?)?,
                                r: x.r.checked_sub(axis.vector().r.checked_mul(offset)?)?,
                            };
                            candidate_windows.push(WindowKey { start, axis });
                        }
                    }
                }
            }
            candidate_windows.sort_by_key(|key| win_id(*key));
            candidate_windows.dedup_by_key(|key| win_id(*key));
            ctx.charge(candidate_windows.len() as u64)?;
            for key in candidate_windows {
                if window_is_all_empty(&state, key) {
                    seeds.push((id, key, SOURCE_VIRGIN));
                }
            }
        }
    }
    // Top-down propagation. Preorder = reverse postorder.
    let mut incoming: Vec<HashMap<WinId, (WindowKey, u8)>> =
        vec![HashMap::new(); ctx.cert.nodes.len()];
    for (id, key, bits) in seeds {
        let entry = incoming[id as usize].entry(win_id(key)).or_insert((key, 0));
        entry.1 |= bits;
    }
    let preorder: Vec<CertNodeId> = ctx.postorder.iter().rev().copied().collect();
    for &id in &preorder {
        let index = id as usize;
        let rows: Vec<(WindowKey, u8)> = {
            let mut rows: Vec<_> = incoming[index]
                .values()
                .map(|(key, bits)| (*key, *bits))
                .collect();
            rows.sort_by_key(|(key, _)| window_sort_key(*key));
            rows
        };
        ctx.charge(rows.len() as u64)?;
        for (key, bits) in &rows {
            // Evaluate the clock (also enforces the <= B containment).
            let _ = window_clock(ctx, id, *key)?;
            // Propagate downward unless a stop rule applies at this node.
            let non_d_alive = window_has_claimant_stone(&ctx.states[index], ctx.claimant, *key);
            if non_d_alive {
                continue;
            }
            match &ctx.cert.nodes[index] {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
                CertNode::Choice { mv, child } => {
                    if !key.contains(*mv) {
                        let entry = incoming[*child as usize]
                            .entry(win_id(*key))
                            .or_insert((*key, 0));
                        entry.1 |= bits;
                    }
                }
                CertNode::Universal { .. }
                | CertNode::UniversalGroup2V1(_)
                | CertNode::FhwGateV1(_) => {
                    for (_, child) in ctx.children[index].clone() {
                        let entry = incoming[child as usize]
                            .entry(win_id(*key))
                            .or_insert((*key, 0));
                        entry.1 |= bits;
                    }
                }
            }
        }
        ctx.demands[index] = rows;
    }
    Some(())
}

/// §3.4 zone construction plus the stored-scalar equality checks for every
/// Group-2 node.
fn check_group2_nodes(ctx: &mut G2Context<'_>) -> Option<()> {
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let CertNode::UniversalGroup2V1(g2) = &ctx.cert.nodes[index] else {
            continue;
        };
        let g2 = g2.clone();
        let state = ctx.states[index].clone();
        // Stored scalars are evidence only; equality with the derivation is
        // mandatory.
        if g2.proof.claimed_d14_budget != ctx.b_local[index]
            || g2.proof.build_horizon != ctx.cert.semantic_horizon
        {
            return None;
        }
        let legal = sorted_legal_moves(&state);
        let stones = state.board().occupied_cells();
        let explicit: Vec<HexCoord> = g2.edges.iter().map(|edge| edge.mv).collect();
        if explicit.is_empty() {
            return None;
        }

        // Z_dir and Z_seed from the live-role f_cut clocks.
        let mut required: Vec<HexCoord> = Vec::new();
        let mut carrier_f: HashMap<CoordKey, u32> = HashMap::new();
        for (role, (_r_full, f_cut)) in &ctx.roles[index] {
            let carrier = role.carrier();
            let slot = carrier_f.entry(carrier).or_insert(0);
            *slot = (*slot).max(*f_cut);
        }
        ctx.charge(carrier_f.len() as u64)?;
        for (carrier, f) in &carrier_f {
            let cell = HexCoord {
                q: carrier.0,
                r: carrier.1,
            };
            if set_contains(&legal, cell) {
                required.push(cell); // Z_dir
            } else if !stones.contains(&cell) && *f >= 1 {
                // Z_seed: Legal within B_{8(f-1)}(carrier).
                let radius = 8u32.checked_mul(f.checked_sub(1)?)?;
                ctx.charge(legal.len() as u64)?;
                for c in &legal {
                    if i32::from(hex_distance(*c, cell)) as u32 <= radius {
                        required.push(*c);
                    }
                }
            }
        }

        // Z_touch and Z_virgin from the demanded windows at this node.
        let demands = ctx.demands[index].clone();
        for (key, _bits) in &demands {
            let q = window_clock(ctx, id, *key)?.0;
            let defender_count = window_defender_count(&state, ctx.claimant.other(), *key);
            let claimant_blocked = window_has_claimant_stone(&state, ctx.claimant, *key);
            if !claimant_blocked
                && defender_count >= 1
                && defender_count.checked_add(q)? >= 6
            {
                required.extend(window_empty_cells(&state, *key)); // Z_touch
            }
            if !claimant_blocked && defender_count == 0 && window_is_all_empty(&state, *key) && q >= 6 {
                let radius = 8u32.checked_mul(q.checked_sub(6)?)?;
                ctx.charge(legal.len() as u64)?;
                for c in &legal {
                    if window_distance(*c, *key) <= radius {
                        required.push(*c); // Z_virgin
                    }
                }
            }
        }

        required.sort_by_key(|coord| coord_key(*coord));
        required.dedup();
        // Every required coordinate must be legal and covered by an explicit
        // edge. Supersets are valid; empty Required still needs the (already
        // enforced) nonempty explicit set.
        for cell in &required {
            if !set_contains(&legal, *cell) {
                return None;
            }
            if explicit
                .binary_search_by_key(&coord_key(*cell), |c| coord_key(*c))
                .is_err()
            {
                return None;
            }
        }
        ctx.required.insert(id, required);
    }
    Some(())
}

/// Convert a stored `RoleKeyV1` into the internal `RoleId` (arena node ids are
/// shared between the two representations).
fn role_key_to_id(role: &RoleKeyV1) -> RoleId {
    match role {
        RoleKeyV1::ChoiceMove { node, cell } => RoleId::ChoiceMove {
            node: *node,
            cell: coord_key(*cell),
        },
        RoleKeyV1::OrCompletionMove { node, cell } => RoleId::OrCompletionMove {
            node: *node,
            cell: coord_key(*cell),
        },
        RoleKeyV1::LeafEmpty {
            node,
            witness,
            cell,
        } => RoleId::LeafEmpty {
            node: *node,
            witness: win_id(*witness),
            cell: coord_key(*cell),
        },
        RoleKeyV1::Checkpoint {
            gate,
            threat,
            cell,
        } => RoleId::Checkpoint {
            gate: *gate,
            threat: win_id(*threat),
            cell: coord_key(*cell),
        },
    }
}

/// §3.3 per-gate acceptance: every stored role row and window row is recomputed
/// from the replayed position and the representative-child clocks and compared
/// byte-for-byte; the role domain must equal the live roles below `C_s` and the
/// window domain must equal the exact `demands(Q)` (Cartesian `K x demands`).
fn check_gate_nodes(ctx: &mut G2Context<'_>) -> Option<()> {
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let CertNode::FhwGateV1(gate) = &ctx.cert.nodes[index] else {
            continue;
        };
        let gate = gate.clone();
        let phi = ctx.gates.get(&id)?.phi.clone();
        let rep_child = ctx.gates.get(&id)?.rep_child.clone();
        let demands = ctx.demands[index].clone();
        let demand_ids: HashSet<WinId> = demands.iter().map(|(k, _)| win_id(*k)).collect();
        let state = ctx.states[index].clone();
        let claimant = ctx.claimant;

        for entry in &gate.proof.map {
            let d = entry.real_reply;
            let s = *phi.get(&coord_key(d))?;
            if s != entry.representative {
                return None;
            }
            let child = *rep_child.get(&coord_key(s))?;

            // ---- Role rows: domain == live roles below C_s ----
            let child_roles = ctx.roles[child as usize].clone();
            ctx.charge(child_roles.len() as u64)?;
            if entry.roles.len() != child_roles.len() {
                return None;
            }
            let mut seen_roles: HashSet<RoleId> = HashSet::new();
            for claim in &entry.roles {
                let rid = role_key_to_id(&claim.role);
                if !seen_roles.insert(rid) {
                    return None; // duplicate
                }
                let (_child_r, child_f) = *child_roles.get(&rid)?; // real live role
                if claim.child_f != child_f {
                    return None;
                }
                let carrier = rid.carrier();
                let y = HexCoord {
                    q: carrier.0,
                    r: carrier.1,
                };
                let (row, epsilon) = derive_gate_role_row(ctx, id, d, y, child_f)?;
                if claim.row != row || claim.epsilon != epsilon {
                    return None;
                }
            }

            // ---- Window rows: domain == demands(Q) exactly (Cartesian) ----
            ctx.charge(demands.len() as u64)?;
            if entry.windows.len() != demand_ids.len() {
                return None;
            }
            let mut seen_windows: HashSet<WinId> = HashSet::new();
            for claim in &entry.windows {
                let wid = win_id(claim.window);
                if !demand_ids.contains(&wid) || !seen_windows.insert(wid) {
                    return None; // unrequested, or duplicate, window row
                }
                let child_q = window_clock(ctx, child, claim.window)?.0;
                if claim.child_q != child_q {
                    return None;
                }
                let d_in = claim.window.contains(d);
                let s_in = claim.window.contains(s);
                if claim.d_in_window != d_in || claim.s_in_window != s_in {
                    return None;
                }
                let (row, kappa, guard) =
                    derive_gate_window_row(ctx, id, d, claim.window, child_q)?;
                if claim.row != row || claim.kappa != kappa || claim.retained_guard != guard {
                    return None;
                }
                // Sanity: window geometry recomputed from the same board.
                let _ = window_geom(&state, claimant, claim.window);
            }
        }
    }
    Some(())
}

// ---------------------------------------------------------------------------
// §2.4 digest recomputation.
// ---------------------------------------------------------------------------

struct TransformTables {
    /// Per transform: preorder IDs after transformed-move edge sorting.
    pre_ids: Vec<Vec<u32>>,
    /// Per transform, per node: sorted outgoing (transformed move, child).
    sorted_children: Vec<Vec<Vec<(HexCoord, CertNodeId)>>>,
}

fn build_transform_tables(ctx: &G2Context<'_>) -> Option<TransformTables> {
    let node_count = ctx.cert.nodes.len();
    let mut pre_ids = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
    let mut sorted_children = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let mut per_node = Vec::with_capacity(node_count);
        for children in &ctx.children {
            let mut transformed = children
                .iter()
                .map(|(mv, child)| Some((d6_transform_coord(*mv, symmetry)?, *child)))
                .collect::<Option<Vec<_>>>()?;
            transformed.sort_by_key(|(mv, _)| coord_key(*mv));
            if transformed
                .windows(2)
                .any(|pair| coord_key(pair[0].0) == coord_key(pair[1].0))
            {
                return None;
            }
            per_node.push(transformed);
        }
        // Depth-first preorder from the root, following sorted edges.
        let mut ids = vec![u32::MAX; node_count];
        let mut next_id = 0u32;
        let mut stack = vec![ctx.cert.root_node];
        while let Some(id) = stack.pop() {
            if ids[id as usize] != u32::MAX {
                return None;
            }
            ids[id as usize] = next_id;
            next_id = next_id.checked_add(1)?;
            for (_, child) in per_node[id as usize].iter().rev() {
                stack.push(*child);
            }
        }
        if ids.iter().any(|assigned| *assigned == u32::MAX) {
            return None;
        }
        pre_ids.push(ids);
        sorted_children.push(per_node);
    }
    Some(TransformTables {
        pre_ids,
        sorted_children,
    })
}

fn enc_state_record(out: &mut Vec<u8>, state: &RustHexoState, symmetry: u8) -> Option<()> {
    let mut stones: Vec<(HexCoord, Player)> = state
        .board()
        .occupied_cells()
        .iter()
        .map(|coord| {
            let owner = state.board().get(*coord)?;
            Some((d6_transform_coord(*coord, symmetry)?, owner))
        })
        .collect::<Option<Vec<_>>>()?;
    stones.sort_by_key(|(coord, owner)| (coord.q, coord.r, player_tag(*owner)));
    enc_u64(out, stones.len() as u64);
    for (coord, owner) in &stones {
        enc_coord(out, *coord);
        out.push(player_tag(*owner));
    }
    out.push(player_tag(state.current_player()));
    match state.phase() {
        TurnPhase::Opening => out.push(0),
        TurnPhase::FirstStone => out.push(1),
        TurnPhase::SecondStone { first } => {
            out.push(2);
            enc_coord(out, d6_transform_coord(first, symmetry)?);
        }
    }
    enc_u32(out, state.placements_made());
    match state.terminal() {
        None => out.push(0),
        Some(outcome) => {
            out.push(1);
            out.push(player_tag(outcome.winner));
            enc_u32(out, outcome.placements);
        }
    }
    Some(())
}

fn node_tag(node: &CertNode) -> u8 {
    match node {
        CertNode::OrCompletion { .. } => 0,
        CertNode::Win { .. } => 1,
        CertNode::Loss { .. } => 2,
        CertNode::Choice { .. } => 3,
        CertNode::Universal { .. } => 4,
        CertNode::UniversalGroup2V1(_) => 5,
        CertNode::FhwGateV1(_) => 6,
    }
}

fn transform_window(key: WindowKey, symmetry: u8) -> Option<WindowKey> {
    // Reuse the verifier's D6 window transform through the public coord
    // transform: map both endpoints and rebuild the canonical key.
    let first = d6_transform_coord(key.coord_at(0), symmetry)?;
    let second = d6_transform_coord(key.coord_at(1), symmetry)?;
    let dq = i32::from(second.q) - i32::from(first.q);
    let dr = i32::from(second.r) - i32::from(first.r);
    match (dq, dr) {
        (1, 0) => Some(WindowKey {
            start: first,
            axis: Axis::Q,
        }),
        (0, 1) => Some(WindowKey {
            start: first,
            axis: Axis::R,
        }),
        (1, -1) => Some(WindowKey {
            start: first,
            axis: Axis::QR,
        }),
        (-1, 0) => Some(WindowKey {
            start: d6_transform_coord(key.coord_at(5), symmetry)?,
            axis: Axis::Q,
        }),
        (0, -1) => Some(WindowKey {
            start: d6_transform_coord(key.coord_at(5), symmetry)?,
            axis: Axis::R,
        }),
        (-1, 1) => Some(WindowKey {
            start: d6_transform_coord(key.coord_at(5), symmetry)?,
            axis: Axis::QR,
        }),
        _ => None,
    }
}

/// Local semantic payload (§2.4): the node payload with outgoing edges and
/// child IDs removed; stored digest fields omitted. `pre_ids` (the `g*`-frame
/// preorder ids) remap gate role-key node references.
fn enc_semantic_local(
    out: &mut Vec<u8>,
    node: &CertNode,
    symmetry: u8,
    pre_ids: &[u32],
) -> Option<()> {
    out.push(node_tag(node));
    match node {
        CertNode::OrCompletion {
            mv,
            witness,
            completion_ply,
        } => {
            enc_coord(out, d6_transform_coord(*mv, symmetry)?);
            enc_window(out, transform_window(*witness, symmetry)?);
            enc_u32(out, *completion_ply);
        }
        CertNode::Win {
            witness,
            count,
            budget,
            resolution_ply,
        } => {
            enc_window(out, transform_window(*witness, symmetry)?);
            out.push(*count);
            out.push(*budget);
            enc_u32(out, *resolution_ply);
        }
        CertNode::Loss {
            witnesses,
            resolution_ply,
        } => {
            let mut keys = witnesses
                .iter()
                .map(|key| transform_window(*key, symmetry))
                .collect::<Option<Vec<_>>>()?;
            keys.sort_by_key(|key| window_sort_key(*key));
            enc_u64(out, keys.len() as u64);
            for key in keys {
                enc_window(out, key);
            }
            enc_u32(out, *resolution_ply);
        }
        CertNode::Choice { .. } => {}
        CertNode::Universal {
            implicit_dispatch,
            zone,
            commutations,
            ..
        } => {
            out.push(u8::from(*implicit_dispatch));
            match zone {
                None => out.push(0),
                Some(zone) => {
                    out.push(1);
                    enc_u32(out, zone.d);
                    enc_u32(out, zone.build_horizon);
                }
            }
            enc_u64(out, commutations.len() as u64);
            // The narrow class rejects commutations before hashing; encode
            // the count anyway so the grammar stays total.
        }
        CertNode::UniversalGroup2V1(g2) => {
            enc_u16(out, g2.proof.schema_version);
            enc_authority(out, &g2.proof.authority);
            enc_u32(out, g2.proof.claimed_d14_budget);
            enc_u32(out, g2.proof.build_horizon);
        }
        CertNode::FhwGateV1(gate) => {
            // Complete stored gate payload (edges/child IDs removed), fully
            // transformed and re-sorted into canonical order under this
            // symmetry so all 12 D6 images produce identical bytes.
            enc_u16(out, gate.proof.schema_version);
            enc_authority(out, &gate.proof.authority);
            let mut threats = gate
                .proof
                .threats
                .iter()
                .map(|k| transform_window(*k, symmetry))
                .collect::<Option<Vec<_>>>()?;
            threats.sort_by_key(|k| window_sort_key(*k));
            enc_u64(out, threats.len() as u64);
            for k in threats {
                enc_window(out, k);
            }
            enc_u32(out, gate.proof.escape_resolution_ply);
            let mut map_rows: Vec<(CoordKey, Vec<u8>)> = Vec::with_capacity(gate.proof.map.len());
            for entry in &gate.proof.map {
                let d = d6_transform_coord(entry.real_reply, symmetry)?;
                let s = d6_transform_coord(entry.representative, symmetry)?;
                let mut row = Vec::new();
                enc_coord(&mut row, d);
                enc_coord(&mut row, s);
                row.push(edge_class_tag(entry.edge_class));
                // Roles, transformed and re-sorted by encoded (canonical) key.
                let mut role_rows: Vec<Vec<u8>> = Vec::with_capacity(entry.roles.len());
                for claim in &entry.roles {
                    let rid = role_key_to_id(&claim.role);
                    let mut r = Vec::new();
                    enc_role_key(&mut r, &rid, symmetry, pre_ids)?;
                    enc_u32(&mut r, claim.child_f);
                    r.push(role_row_tag(claim.row));
                    r.push(claim.epsilon);
                    role_rows.push(r);
                }
                role_rows.sort();
                enc_u64(&mut row, role_rows.len() as u64);
                for r in role_rows {
                    row.extend_from_slice(&r);
                }
                // Windows, transformed and re-sorted by encoded (canonical) key.
                let mut win_rows: Vec<Vec<u8>> = Vec::with_capacity(entry.windows.len());
                for claim in &entry.windows {
                    let w = transform_window(claim.window, symmetry)?;
                    let mut r = Vec::new();
                    enc_window(&mut r, w);
                    enc_u32(&mut r, claim.child_q);
                    r.push(u8::from(claim.d_in_window));
                    r.push(u8::from(claim.s_in_window));
                    r.push(kappa_row_tag(claim.row));
                    r.push(claim.kappa);
                    r.push(guard_tag(claim.retained_guard));
                    win_rows.push(r);
                }
                win_rows.sort();
                enc_u64(&mut row, win_rows.len() as u64);
                for r in win_rows {
                    row.extend_from_slice(&r);
                }
                map_rows.push((coord_key(d), row));
            }
            map_rows.sort_by_key(|(k, _)| *k);
            enc_u64(out, map_rows.len() as u64);
            for (_, row) in map_rows {
                out.extend_from_slice(&row);
            }
        }
    }
    Some(())
}

/// The FhwGate derived-record class payload (§2.4). All coords/windows are
/// transformed and every set re-sorted into canonical order under `symmetry`
/// so the 12 D6 images produce identical derived hashes. Uses the RECOMPUTED
/// gate values; on the Exact/FC accept path the `*_evaluated`/RC/WC bits are
/// all false (those predicates are inapplicable to Exact/FC edges).
fn gate_derived_class_payload(
    ctx: &mut G2Context<'_>,
    gate_id: CertNodeId,
    symmetry: u8,
    pre_ids: &[u32],
) -> Option<Vec<u8>> {
    let info = ctx.gates.get(&gate_id)?;
    let b = info.b;
    let escape = info.escape_ply;
    let kernel = info.kernel.clone();
    let reps = info.reps.clone();
    let threats = info.threats.clone();
    let phi = info.phi.clone();
    let rep_child = info.rep_child.clone();
    let demands = ctx.demands[gate_id as usize].clone();
    let claimant = ctx.claimant;

    // Precompute every needed child Q_cut (mutable recursion) before the
    // immutable derivations below.
    let mut child_qs: HashMap<(CertNodeId, WinId), u32> = HashMap::new();
    for &d in &kernel {
        let s = *phi.get(&coord_key(d))?;
        let child = *rep_child.get(&coord_key(s))?;
        for (w, _) in &demands {
            let key = (child, win_id(*w));
            if !child_qs.contains_key(&key) {
                let q = window_clock(ctx, child, *w)?.0;
                child_qs.insert(key, q);
            }
        }
    }

    let mut out = Vec::new();
    out.push(2); // FhwGate
    out.push(b);
    // H (threats), transformed + re-sorted.
    let mut ht = threats
        .iter()
        .map(|k| transform_window(*k, symmetry))
        .collect::<Option<Vec<_>>>()?;
    ht.sort_by_key(|k| window_sort_key(*k));
    enc_u64(&mut out, ht.len() as u64);
    for k in ht {
        enc_window(&mut out, k);
    }
    // K, transformed + re-sorted.
    let mut kt = kernel
        .iter()
        .map(|c| d6_transform_coord(*c, symmetry))
        .collect::<Option<Vec<_>>>()?;
    kt.sort_by_key(|c| coord_key(*c));
    enc_u64(&mut out, kt.len() as u64);
    for c in kt {
        enc_coord(&mut out, c);
    }
    // R, transformed + re-sorted.
    let mut rt = reps
        .iter()
        .map(|c| d6_transform_coord(*c, symmetry))
        .collect::<Option<Vec<_>>>()?;
    rt.sort_by_key(|c| coord_key(*c));
    enc_u64(&mut out, rt.len() as u64);
    for c in rt {
        enc_coord(&mut out, c);
    }
    enc_u32(&mut out, escape);

    let state = &ctx.states[gate_id as usize];
    let mut map_rows: Vec<(CoordKey, Vec<u8>)> = Vec::with_capacity(kernel.len());
    for &d in &kernel {
        let s = *phi.get(&coord_key(d))?;
        let child = *rep_child.get(&coord_key(s))?;
        let dt = d6_transform_coord(d, symmetry)?;
        let st = d6_transform_coord(s, symmetry)?;
        let mut row = Vec::new();
        enc_coord(&mut row, dt);
        enc_coord(&mut row, st);
        let edge_class = *ctx.gates.get(&gate_id)?.edge_class.get(&coord_key(d))?;
        row.push(edge_class_tag(edge_class));
        // Derived role rows over the live roles below C_s.
        let child_roles = &ctx.roles[child as usize];
        let mut role_rows: Vec<Vec<u8>> = Vec::with_capacity(child_roles.len());
        for (role, (_r_full, child_f)) in child_roles {
            let carrier = role.carrier();
            let y = HexCoord {
                q: carrier.0,
                r: carrier.1,
            };
            let (drow, eps) = derive_gate_role_row(ctx, gate_id, d, y, *child_f)?;
            let mut r = Vec::new();
            enc_role_key(&mut r, role, symmetry, pre_ids)?;
            enc_coord(&mut r, d6_transform_coord(y, symmetry)?);
            r.push(1); // child_reachable
            enc_u32(&mut r, *child_f);
            r.push(0); // carrier_ghost_legal (inapplicable on Exact/FC)
            r.push(0); // rc_evaluated
            r.push(0); // rc_pass
            r.push(0); // d22n_pass
            r.push(role_row_tag(drow));
            r.push(eps);
            role_rows.push(r);
        }
        role_rows.sort();
        enc_u64(&mut row, role_rows.len() as u64);
        for r in role_rows {
            row.extend_from_slice(&r);
        }
        // Derived window rows over demands(Q).
        let mut win_rows: Vec<Vec<u8>> = Vec::with_capacity(demands.len());
        for (w, _) in &demands {
            let child_q = *child_qs.get(&(child, win_id(*w)))?;
            let geom = window_geom(state, claimant, *w);
            let (krow, kappa, guard) = derive_gate_window_row(ctx, gate_id, d, *w, child_q)?;
            let mut r = Vec::new();
            enc_window(&mut r, transform_window(*w, symmetry)?);
            enc_u32(&mut r, child_q);
            r.push(u8::from(geom.d_alive));
            r.push(u8::from(geom.all_empty));
            r.push(u8::from(w.contains(d)));
            r.push(u8::from(w.contains(s)));
            r.push(0); // wc_evaluated (inapplicable on Exact/FC)
            r.push(0); // wc_pass
            r.push(kappa_row_tag(krow));
            r.push(kappa);
            r.push(guard_tag(guard));
            win_rows.push(r);
        }
        win_rows.sort();
        enc_u64(&mut row, win_rows.len() as u64);
        for r in win_rows {
            row.extend_from_slice(&r);
        }
        map_rows.push((coord_key(dt), row));
    }
    map_rows.sort_by_key(|(k, _)| *k);
    enc_u64(&mut out, map_rows.len() as u64);
    for (_, row) in map_rows {
        out.extend_from_slice(&row);
    }
    Some(out)
}

struct DigestTables {
    /// semantic_hash[g][node]
    semantic: Vec<Vec<[u8; 32]>>,
    /// derived_hash[g][node]
    derived: Vec<Vec<[u8; 32]>>,
    transforms: TransformTables,
}

fn build_digest_tables(ctx: &mut G2Context<'_>) -> Option<DigestTables> {
    let transforms = build_transform_tables(ctx)?;
    let node_count = ctx.cert.nodes.len();
    let mut semantic = vec![vec![[0u8; 32]; node_count]; D6_SYMMETRY_COUNT as usize];
    let mut derived = vec![vec![[0u8; 32]; node_count]; D6_SYMMETRY_COUNT as usize];
    let postorder = ctx.postorder.clone();
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let g = symmetry as usize;
        for &id in &postorder {
            let index = id as usize;
            ctx.charge(4)?;
            // Semantic Merkle value.
            let mut payload = Vec::new();
            enc_semantic_local(
                &mut payload,
                &ctx.cert.nodes[index],
                symmetry,
                &transforms.pre_ids[g],
            )?;
            let children = &transforms.sorted_children[g][index];
            enc_u64(&mut payload, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut payload, *mv);
                payload.extend_from_slice(&semantic[g][*child as usize]);
            }
            semantic[g][index] = sha256(b"hexo-g2-semantic-node-v1\0", &payload);

            // Derived Merkle value.
            let mut record = Vec::new();
            enc_state_record(&mut record, &ctx.states[index], symmetry)?;
            record.push(node_tag(&ctx.cert.nodes[index]));
            enc_u32(&mut record, ctx.b_local[index]);
            enc_u32(&mut record, ctx.t_sub[index]);
            enc_u32(&mut record, ctx.cert.semantic_horizon);
            // Role rows.
            let mut role_rows: Vec<Vec<u8>> = Vec::with_capacity(ctx.roles[index].len());
            for (role, (r_full, f_cut)) in &ctx.roles[index] {
                let mut row = Vec::new();
                enc_role_key(&mut row, role, symmetry, &transforms.pre_ids[g])?;
                let carrier = role.carrier();
                enc_coord(
                    &mut row,
                    d6_transform_coord(
                        HexCoord {
                            q: carrier.0,
                            r: carrier.1,
                        },
                        symmetry,
                    )?,
                );
                enc_u32(&mut row, *r_full); // r_full
                enc_u32(&mut row, *f_cut); // f_cut (== r_full off gates)
                role_rows.push(row);
            }
            role_rows.sort();
            enc_u64(&mut record, role_rows.len() as u64);
            for row in role_rows {
                record.extend_from_slice(&row);
            }
            // Demand rows.
            let mut demand_rows: Vec<Vec<u8>> = Vec::with_capacity(ctx.demands[index].len());
            for (key, bits) in ctx.demands[index].clone() {
                let (q_cut, e_full) = window_clock(ctx, id, key)?;
                let mut row = Vec::new();
                enc_window(&mut row, transform_window(key, symmetry)?);
                row.push(bits);
                enc_u32(&mut row, e_full); // E_full
                enc_u32(&mut row, q_cut); // Q_cut (== E_full off gates)
                demand_rows.push(row);
            }
            demand_rows.sort();
            enc_u64(&mut record, demand_rows.len() as u64);
            for row in demand_rows {
                record.extend_from_slice(&row);
            }
            // Derived class payload. The gate payload needs the mutable window
            // clock, so it is precomputed before the immutable match below.
            let gate_payload: Option<Vec<u8>> =
                if matches!(ctx.cert.nodes[index], CertNode::FhwGateV1(_)) {
                    Some(gate_derived_class_payload(
                        ctx,
                        id,
                        symmetry,
                        &transforms.pre_ids[g],
                    )?)
                } else {
                    None
                };
            match &ctx.cert.nodes[index] {
                CertNode::UniversalGroup2V1(_) => {
                    record.push(1); // OrdinaryGroup2
                    record.push(*ctx.derived_k.get(&id)?);
                    let mut cells = ctx
                        .required
                        .get(&id)?
                        .iter()
                        .map(|cell| d6_transform_coord(*cell, symmetry))
                        .collect::<Option<Vec<_>>>()?;
                    cells.sort_by_key(|coord| coord_key(*coord));
                    enc_u64(&mut record, cells.len() as u64);
                    for cell in cells {
                        enc_coord(&mut record, cell);
                    }
                }
                CertNode::FhwGateV1(_) => {
                    record.extend_from_slice(gate_payload.as_ref()?);
                }
                _ => record.push(0), // Other
            }
            let mut payload = record;
            enc_u64(&mut payload, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut payload, *mv);
                payload.extend_from_slice(&derived[g][*child as usize]);
            }
            derived[g][index] = sha256(b"hexo-g2-derived-node-v1\0", &payload);
        }
    }
    Some(DigestTables {
        semantic,
        derived,
        transforms,
    })
}

fn enc_role_key(
    out: &mut Vec<u8>,
    role: &RoleId,
    symmetry: u8,
    pre_ids: &[u32],
) -> Option<()> {
    match role {
        RoleId::ChoiceMove { node, cell } => {
            out.push(0);
            enc_u32(out, *pre_ids.get(*node as usize)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
        RoleId::OrCompletionMove { node, cell } => {
            out.push(1);
            enc_u32(out, *pre_ids.get(*node as usize)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
        RoleId::LeafEmpty {
            node,
            witness,
            cell,
        } => {
            out.push(2);
            enc_u32(out, *pre_ids.get(*node as usize)?);
            let key = win_id_to_window(*witness)?;
            enc_window(out, transform_window(key, symmetry)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
        RoleId::Checkpoint {
            gate,
            threat,
            cell,
        } => {
            out.push(3);
            enc_u32(out, *pre_ids.get(*gate as usize)?);
            let key = win_id_to_window(*threat)?;
            enc_window(out, transform_window(key, symmetry)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
    }
    Some(())
}

/// Rebuild a `WindowKey` from its stored `WinId` tuple `(axis_tag, q, r)`.
fn win_id_to_window(id: WinId) -> Option<WindowKey> {
    Some(WindowKey {
        start: HexCoord { q: id.1, r: id.2 },
        axis: match id.0 {
            0 => Axis::Q,
            1 => Axis::R,
            2 => Axis::QR,
            _ => return None,
        },
    })
}

fn lexicographic_min(candidates: Vec<Vec<u8>>) -> Option<Vec<u8>> {
    candidates.into_iter().min()
}

/// Recompute and compare `child_plan_sha256` and `finder_summary_sha256` for
/// every Group-2 node.
fn check_digests(ctx: &mut G2Context<'_>) -> Option<()> {
    let tables = build_digest_tables(ctx)?;
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let CertNode::UniversalGroup2V1(g2) = &ctx.cert.nodes[index] else {
            continue;
        };
        // child_plan_sha256.
        let mut plan_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            let children = &tables.transforms.sorted_children[g][index];
            enc_u64(&mut preimage, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut preimage, *mv);
                preimage.extend_from_slice(&tables.semantic[g][*child as usize]);
            }
            plan_preimages.push(preimage);
        }
        let child_plan = sha256(
            b"hexo-g2-child-plan-v1\0",
            &lexicographic_min(plan_preimages)?,
        );
        if child_plan != g2.proof.child_plan_sha256 {
            return None;
        }
        // finder_summary_sha256.
        let mut summary_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_authority(&mut preimage, &g2.proof.authority);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            preimage.extend_from_slice(&child_plan);
            preimage.extend_from_slice(&tables.derived[g][index]);
            summary_preimages.push(preimage);
        }
        let summary = sha256(
            b"hexo-g2-summary-v1\0",
            &lexicographic_min(summary_preimages)?,
        );
        if summary != g2.proof.finder_summary_sha256 {
            return None;
        }
    }
    Some(())
}

// ---------------------------------------------------------------------------
// Finder-facing helpers (called from tss_solver; this module never imports
// the solver). Deviation 3 in the notes: finder and verifier share these
// derivations, so the digest comparison detects drift/tampering, not
// correlated implementation bugs.
// ---------------------------------------------------------------------------

/// Exact §3.4 Required_FHW for a candidate Group-2 node under construction:
/// the node's state plus its proven `(move, child)` edges over `arena`.
/// Returns None when the subtree leaves the narrow class (gates, dispatch,
/// zoned or commuted nodes, DAG sharing is fine here — clocks are
/// state-determined) or any derivation fails; the caller must then fall back
/// to the legacy uniform path.
pub(crate) fn finder_required_fhw(
    state: &RustHexoState,
    claimant: Player,
    edges: &[(HexCoord, CertNodeId)],
    arena: &[CertNode],
) -> Option<Vec<HexCoord>> {
    // Build a temporary single-node certificate view: a synthetic Group-2
    // node above the given children. We reuse the verification derivations by
    // materializing the subtree as its own certificate.
    let mut nodes: Vec<CertNode> = Vec::new();
    let mut remap: HashMap<CertNodeId, CertNodeId> = HashMap::new();
    let mut new_edges = Vec::with_capacity(edges.len());
    for (mv, child) in edges {
        let child = copy_subtree(arena, *child, &mut nodes, &mut remap)?;
        new_edges.push(crate::tss_verify::CertEdge { mv: *mv, child });
    }
    new_edges.sort_by_key(|edge| coord_key(edge.mv));
    if new_edges
        .windows(2)
        .any(|pair| coord_key(pair[0].mv) == coord_key(pair[1].mv))
    {
        return None;
    }
    let synthetic = CertNode::UniversalGroup2V1(Box::new(
        crate::tss_verify::UniversalGroup2NodeV1 {
            edges: new_edges,
            proof: crate::tss_verify::Group2ZoneV1 {
                schema_version: 1,
                authority: Group2AuthorityV1::compiled(),
                claimed_d14_budget: 0,
                build_horizon: 0,
                child_plan_sha256: [0u8; 32],
                finder_summary_sha256: [0u8; 32],
            },
        },
    ));
    let root_id = nodes.len() as CertNodeId;
    nodes.push(synthetic);
    let cert = TssCertificate {
        root: RootBinding::from_state(state),
        claimant,
        root_node: root_id,
        nodes,
        semantic_horizon: u32::MAX,
    };
    let mut ctx = build_context(state, &cert)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;
    // check_group2_nodes enforces required ⊆ explicit, which is exactly the
    // closure question; recompute required directly instead.
    compute_required_only(&mut ctx, root_id)
}

/// The §3.4 union for one node without the coverage requirement.
fn compute_required_only(ctx: &mut G2Context<'_>, id: CertNodeId) -> Option<Vec<HexCoord>> {
    let index = id as usize;
    let state = ctx.states[index].clone();
    let legal = sorted_legal_moves(&state);
    let stones = state.board().occupied_cells();
    let mut required: Vec<HexCoord> = Vec::new();
    let mut carrier_f: HashMap<CoordKey, u32> = HashMap::new();
    for (role, (_r_full, f_cut)) in &ctx.roles[index] {
        let carrier = role.carrier();
        let slot = carrier_f.entry(carrier).or_insert(0);
        *slot = (*slot).max(*f_cut);
    }
    for (carrier, f) in &carrier_f {
        let cell = HexCoord {
            q: carrier.0,
            r: carrier.1,
        };
        if set_contains(&legal, cell) {
            required.push(cell);
        } else if !stones.contains(&cell) && *f >= 1 {
            let radius = 8u32.checked_mul(f.checked_sub(1)?)?;
            for c in &legal {
                if i32::from(hex_distance(*c, cell)) as u32 <= radius {
                    required.push(*c);
                }
            }
        }
    }
    let demands = ctx.demands[index].clone();
    for (key, _bits) in &demands {
        let q = window_clock(ctx, id, *key)?.0;
        let defender_count = window_defender_count(&state, ctx.claimant.other(), *key);
        let claimant_blocked = window_has_claimant_stone(&state, ctx.claimant, *key);
        if !claimant_blocked && defender_count >= 1 && defender_count.checked_add(q)? >= 6 {
            required.extend(window_empty_cells(&state, *key));
        }
        if !claimant_blocked && defender_count == 0 && window_is_all_empty(&state, *key) && q >= 6 {
            let radius = 8u32.checked_mul(q.checked_sub(6)?)?;
            for c in &legal {
                if window_distance(*c, *key) <= radius {
                    required.push(*c);
                }
            }
        }
    }
    required.sort_by_key(|coord| coord_key(*coord));
    required.dedup();
    Some(required)
}

/// Copy the subtree below `root` from a solver arena into `nodes`, unfolding
/// DAG sharing into a tree (each occurrence copied). Rejects when the copy
/// leaves the narrow class or exceeds arena caps.
fn copy_subtree(
    arena: &[CertNode],
    root: CertNodeId,
    nodes: &mut Vec<CertNode>,
    _remap: &mut HashMap<CertNodeId, CertNodeId>,
) -> Option<CertNodeId> {
    fn copy(
        arena: &[CertNode],
        id: CertNodeId,
        nodes: &mut Vec<CertNode>,
        depth: usize,
    ) -> Option<CertNodeId> {
        if depth > MAX_CERT_DEPTH || nodes.len() >= crate::tss_verify::MAX_CERT_NODES {
            return None;
        }
        let node = arena.get(id as usize)?;
        let copied = match node {
            CertNode::OrCompletion { .. }
            | CertNode::Win { .. }
            | CertNode::Loss { .. } => node.clone(),
            CertNode::Choice { mv, child } => CertNode::Choice {
                mv: *mv,
                child: copy(arena, *child, nodes, depth + 1)?,
            },
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                if *implicit_dispatch || zone.is_some() || !commutations.is_empty() {
                    return None;
                }
                let mut new_edges = Vec::with_capacity(edges.len());
                for edge in edges {
                    new_edges.push(crate::tss_verify::CertEdge {
                        mv: edge.mv,
                        child: copy(arena, edge.child, nodes, depth + 1)?,
                    });
                }
                new_edges.sort_by_key(|edge| coord_key(edge.mv));
                CertNode::Universal {
                    edges: new_edges,
                    implicit_dispatch: false,
                    zone: None,
                    commutations: Vec::new(),
                }
            }
            CertNode::UniversalGroup2V1(g2) => {
                let mut new_edges = Vec::with_capacity(g2.edges.len());
                for edge in &g2.edges {
                    new_edges.push(crate::tss_verify::CertEdge {
                        mv: edge.mv,
                        child: copy(arena, edge.child, nodes, depth + 1)?,
                    });
                }
                new_edges.sort_by_key(|edge| coord_key(edge.mv));
                CertNode::UniversalGroup2V1(Box::new(crate::tss_verify::UniversalGroup2NodeV1 {
                    edges: new_edges,
                    proof: g2.proof.clone(),
                }))
            }
            CertNode::FhwGateV1(gate) => {
                // Unfold the representative subtrees; canonicalize the
                // representative edges. The proof (threats/map/escape) is
                // preserved; the finalizer fills the map rows.
                let mut new_reps = Vec::with_capacity(gate.representatives.len());
                for edge in &gate.representatives {
                    new_reps.push(crate::tss_verify::CertEdge {
                        mv: edge.mv,
                        child: copy(arena, edge.child, nodes, depth + 1)?,
                    });
                }
                new_reps.sort_by_key(|edge| coord_key(edge.mv));
                CertNode::FhwGateV1(Box::new(crate::tss_verify::FhwGateNodeV1 {
                    representatives: new_reps,
                    proof: gate.proof.clone(),
                }))
            }
        };
        let new_id = u32::try_from(nodes.len()).ok()?;
        nodes.push(copied);
        Some(new_id)
    }
    copy(arena, root, nodes, 0)
}

/// Post-compaction pass used by the finder: given an assembled certificate
/// whose Group-2 nodes carry placeholder scalars/digests, (1) sort every edge
/// and Loss-witness list into canonical order, (2) unfold DAG sharing into a
/// strict tree, (3) fill `claimed_d14_budget`, `build_horizon`, and both
/// digests from the same derivations the verifier replays. Returns None when
/// the certificate cannot be brought into the narrow class.
pub(crate) fn finder_finalize_group2(
    state: &RustHexoState,
    cert: &TssCertificate,
) -> Option<TssCertificate> {
    // Unfold to a strict tree rooted at root_node.
    let mut nodes: Vec<CertNode> = Vec::new();
    let mut remap = HashMap::new();
    let root = copy_subtree(&cert.nodes, cert.root_node, &mut nodes, &mut remap)?;
    // Canonicalize Loss witness order.
    for node in &mut nodes {
        if let CertNode::Loss { witnesses, .. } = node {
            witnesses.sort_by_key(|key| window_sort_key(*key));
            if witnesses
                .windows(2)
                .any(|pair| window_sort_key(pair[0]) == window_sort_key(pair[1]))
            {
                return None;
            }
        }
    }
    let mut out = TssCertificate {
        root: cert.root.clone(),
        claimant: cert.claimant,
        root_node: root,
        nodes,
        semantic_horizon: cert.semantic_horizon,
    };
    // Derive scalars on the unfolded tree.
    let mut ctx = build_context(state, &out)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;
    let b_local = ctx.b_local.clone();
    // Fill claimed scalars first (they enter the semantic hashes).
    for (index, node) in out.nodes.iter_mut().enumerate() {
        if let CertNode::UniversalGroup2V1(g2) = node {
            g2.proof.schema_version = 1;
            g2.proof.authority = Group2AuthorityV1::compiled();
            g2.proof.claimed_d14_budget = b_local[index];
            g2.proof.build_horizon = out.semantic_horizon;
        }
    }
    // Re-derive on the finalized scalar values and fill digests. The derived
    // k / required tables come from the checking pass.
    let mut ctx = build_context(state, &out)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;
    check_group2_nodes(&mut ctx)?;
    let tables = build_digest_tables(&mut ctx)?;
    let mut plans: HashMap<usize, ([u8; 32], [u8; 32])> = HashMap::new();
    for (index, node) in out.nodes.iter().enumerate() {
        let CertNode::UniversalGroup2V1(g2) = node else {
            continue;
        };
        let mut plan_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            let children = &tables.transforms.sorted_children[g][index];
            enc_u64(&mut preimage, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut preimage, *mv);
                preimage.extend_from_slice(&tables.semantic[g][*child as usize]);
            }
            plan_preimages.push(preimage);
        }
        let child_plan = sha256(
            b"hexo-g2-child-plan-v1\0",
            &lexicographic_min(plan_preimages)?,
        );
        let mut summary_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_authority(&mut preimage, &g2.proof.authority);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            preimage.extend_from_slice(&child_plan);
            preimage.extend_from_slice(&tables.derived[g][index]);
            summary_preimages.push(preimage);
        }
        let summary = sha256(
            b"hexo-g2-summary-v1\0",
            &lexicographic_min(summary_preimages)?,
        );
        plans.insert(index, (child_plan, summary));
    }
    for (index, node) in out.nodes.iter_mut().enumerate() {
        if let CertNode::UniversalGroup2V1(g2) = node {
            let (plan, summary) = plans.get(&index)?;
            g2.proof.child_plan_sha256 = *plan;
            g2.proof.finder_summary_sha256 = *summary;
        }
    }
    Some(out)
}

/// Reverse of `role_key_to_id`: rebuild a stored `RoleKeyV1` from the internal
/// `RoleId` (used by the gate-row filler that produces positive fixtures).
fn id_to_role_key(role: &RoleId) -> Option<RoleKeyV1> {
    Some(match role {
        RoleId::ChoiceMove { node, cell } => RoleKeyV1::ChoiceMove {
            node: *node,
            cell: HexCoord { q: cell.0, r: cell.1 },
        },
        RoleId::OrCompletionMove { node, cell } => RoleKeyV1::OrCompletionMove {
            node: *node,
            cell: HexCoord { q: cell.0, r: cell.1 },
        },
        RoleId::LeafEmpty {
            node,
            witness,
            cell,
        } => RoleKeyV1::LeafEmpty {
            node: *node,
            witness: win_id_to_window(*witness)?,
            cell: HexCoord { q: cell.0, r: cell.1 },
        },
        RoleId::Checkpoint {
            gate,
            threat,
            cell,
        } => RoleKeyV1::Checkpoint {
            gate: *gate,
            threat: win_id_to_window(*threat)?,
            cell: HexCoord { q: cell.0, r: cell.1 },
        },
    })
}

/// Finder helper: given a certificate whose `FhwGateV1` nodes carry a complete
/// map SKELETON (correct `real_reply`/`representative`/`edge_class` per K, but
/// empty role/window lists), fill every gate map's role and window rows from the
/// verifier's own derivation. Used to construct positive fixtures for the
/// accept path (the rows are redundant claims recomputed and byte-compared at
/// verification; the finalizer fills them from the shared derivation, exactly
/// as the finder would). Returns `None` if the skeleton does not reconstruct.
pub(crate) fn finder_fill_gate_rows(
    state: &RustHexoState,
    cert: &TssCertificate,
) -> Option<TssCertificate> {
    let mut ctx = build_context(state, cert)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;

    // Collect the derived rows for every gate node, then rebuild the cert.
    let mut new_maps: HashMap<usize, Vec<crate::tss_verify::FhwMapV1>> = HashMap::new();
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        if !matches!(ctx.cert.nodes[index], CertNode::FhwGateV1(_)) {
            continue;
        }
        let phi = ctx.gates.get(&id)?.phi.clone();
        let rep_child = ctx.gates.get(&id)?.rep_child.clone();
        let edge_class = ctx.gates.get(&id)?.edge_class.clone();
        let kernel = ctx.gates.get(&id)?.kernel.clone();
        let demands = ctx.demands[index].clone();
        let mut maps: Vec<crate::tss_verify::FhwMapV1> = Vec::with_capacity(kernel.len());
        for d in kernel {
            let s = *phi.get(&coord_key(d))?;
            let child = *rep_child.get(&coord_key(s))?;
            let cls = *edge_class.get(&coord_key(d))?;
            // Role rows over the live roles below C_s.
            let child_roles = ctx.roles[child as usize].clone();
            let mut roles: Vec<crate::tss_verify::FhwRoleClaimV1> = Vec::new();
            for (role, (_r, child_f)) in &child_roles {
                let carrier = role.carrier();
                let y = HexCoord {
                    q: carrier.0,
                    r: carrier.1,
                };
                let (row, epsilon) = derive_gate_role_row(&ctx, id, d, y, *child_f)?;
                roles.push(crate::tss_verify::FhwRoleClaimV1 {
                    role: id_to_role_key(role)?,
                    child_f: *child_f,
                    row,
                    epsilon,
                });
            }
            roles.sort_by(|a, b| role_key_order(&a.role).cmp(&role_key_order(&b.role)));
            // Window rows over demands(Q).
            let mut windows: Vec<crate::tss_verify::FhwWindowClaimV1> = Vec::new();
            for (w, _) in &demands {
                let child_q = window_clock(&mut ctx, child, *w)?.0;
                let (row, kappa, guard) = derive_gate_window_row(&ctx, id, d, *w, child_q)?;
                windows.push(crate::tss_verify::FhwWindowClaimV1 {
                    window: *w,
                    child_q,
                    d_in_window: w.contains(d),
                    s_in_window: w.contains(s),
                    row,
                    kappa,
                    retained_guard: guard,
                });
            }
            windows.sort_by(|a, b| window_sort_key(a.window).cmp(&window_sort_key(b.window)));
            maps.push(crate::tss_verify::FhwMapV1 {
                real_reply: d,
                representative: s,
                edge_class: cls,
                roles,
                windows,
            });
        }
        maps.sort_by(|a, b| coord_key(a.real_reply).cmp(&coord_key(b.real_reply)));
        new_maps.insert(index, maps);
    }

    let mut out = cert.clone();
    for (index, node) in out.nodes.iter_mut().enumerate() {
        if let CertNode::FhwGateV1(gate) = node {
            gate.proof.map = new_maps.remove(&index)?;
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tss_core::{CertVerify, ProofStatus, SolveCaps, SolveGoal, ZoneSearchCaps};
    use crate::tss_solver::TssSolver;
    use crate::tss_verify::{
        d6_remap_certificate, CertCommutation, CertEdge, FhwGateNodeV1, FhwGateProofV1,
        Group2Verifier, Group2ZoneV1, TssVerifier, UniversalGroup2NodeV1, ZoneInfo,
        D6_SYMMETRY_COUNT,
    };
    use hexo_engine::apply_placement;

    // ----- SHA-256 golden vectors (FIPS 180-4) -----

    #[test]
    fn sha256_matches_fips_golden_vectors() {
        fn hex(bytes: &[u8; 32]) -> String {
            bytes.iter().map(|b| format!("{b:02x}")).collect()
        }
        let mut empty = Sha256::new();
        empty.update(b"");
        assert_eq!(
            hex(&empty.finalize()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut abc = Sha256::new();
        abc.update(b"abc");
        assert_eq!(
            hex(&abc.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut long = Sha256::new();
        long.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&long.finalize()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Chunked update must match one-shot.
        let mut chunked = Sha256::new();
        for chunk in b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".chunks(7) {
            chunked.update(chunk);
        }
        assert_eq!(
            hex(&chunked.finalize()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    // ----- Layout freeze (design 2.2 / 2.5) -----

    #[test]
    fn cert_node_layout_unchanged_by_extension_variants() {
        use hexo_engine::{HexCoord, WindowKey};
        // Mirror of the five legacy variants exactly as they were before the
        // two boxed extension variants were appended.
        #[allow(dead_code)]
        enum LegacyCertNodeMirror {
            OrCompletion {
                mv: HexCoord,
                witness: WindowKey,
                completion_ply: u32,
            },
            Win {
                witness: WindowKey,
                count: u8,
                budget: u8,
                resolution_ply: u32,
            },
            Loss {
                witnesses: Vec<WindowKey>,
                resolution_ply: u32,
            },
            Choice {
                mv: HexCoord,
                child: crate::tss_verify::CertNodeId,
            },
            Universal {
                edges: Vec<CertEdge>,
                implicit_dispatch: bool,
                zone: Option<ZoneInfo>,
                commutations: Vec<CertCommutation>,
            },
        }
        assert_eq!(
            std::mem::size_of::<CertNode>(),
            std::mem::size_of::<LegacyCertNodeMirror>(),
            "boxed extension variants must not change CertNode size"
        );
        assert_eq!(
            std::mem::align_of::<CertNode>(),
            std::mem::align_of::<LegacyCertNodeMirror>(),
            "boxed extension variants must not change CertNode alignment"
        );
    }

    // ----- Fixture construction -----

    const FIXTURE_MOVES: [(i16, i16); 20] = [
        (0, 0),   // P0 opening
        (0, 3),   // P1 group A
        (1, 3),   // P1 group A
        (-5, 1),  // P0 scatter
        (5, 5),   // P0 scatter
        (2, 3),   // P1 group A
        (6, 0),   // P1 group B
        (-3, -6), // P0 scatter
        (9, -4),  // P0 scatter
        (6, -1),  // P1 group B
        (6, -2),  // P1 group B
        (-7, 6),  // P0 scatter
        (11, 1),  // P0 scatter
        (0, -5),  // P1 group C
        (1, -5),  // P1 group C
        (3, 9),   // P0 scatter
        (-8, -2), // P0 scatter
        (2, -5),  // P1 group C
        (-4, 0),  // P1 inert
        (2, 12),  // P0 first stone of the root turn
    ];

    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap_or_else(|err| panic!("move ({q},{r}) must be legal: {err:?}"));
        }
        state
    }

    /// Defender-to-move (P0, SecondStone, b=1) position with k=0 where the
    /// claimant (P1) holds three separated three-in-a-row groups: after any
    /// defender reply, extending two untouched groups to four creates a
    /// tau > 2 family, i.e. a LOSS leaf. The proof tree is gate-free, so the
    /// Group-2 selector preconditions hold at the root.
    fn group2_fixture_root() -> RustHexoState {
        replay(&FIXTURE_MOVES)
    }

    fn solve_fixture(group2: bool) -> (RustHexoState, crate::tss_core::DeepResult<TssCertificate>) {
        let state = group2_fixture_root();
        assert_eq!(state.current_player(), Player::Player0);
        assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }));
        let mut solver = TssSolver::default();
        solver.set_zone_options(ZoneSearchCaps {
            enabled: true,
            stale_area_filter: false,
            count2_threshold: false,
            pair_commutation: false,
        });
        solver.set_group2(group2);
        let caps = SolveCaps {
            node_cap: 200_000,
            tt_bytes_cap: 1 << 20,
            semantic_horizon: state.placements_made() + 16,
        };
        let result = solver.solve_goal(&state, &caps, SolveGoal::Loss);
        (state, result)
    }

    fn accepted_group2_cert() -> (RustHexoState, TssCertificate) {
        let (state, result) = solve_fixture(true);
        assert_eq!(result.status, ProofStatus::Loss, "fixture must be decided");
        let cert = result.cert.expect("decided result carries a certificate");
        assert!(
            cert.nodes.iter().any(CertNode::is_group2_extension),
            "the Group-2 selector must fire on the constructed fixture"
        );
        (state, cert)
    }

    // ----- Phase B: selector fires, reduces fanout, and verifies -----

    #[test]
    fn selector_emits_reduced_fanout_certificate_that_verifies() {
        let (state, cert) = accepted_group2_cert();
        // Strict acceptance under the extension policy; strict rejection
        // under the default legacy-only policy.
        assert!(Group2Verifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!Group2Verifier.verify(&state, &cert, ProofStatus::Win));
        // Reduced fanout: the root Group-2 node explicit set must be far
        // below the full legal set.
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        let CertNode::UniversalGroup2V1(root) = &cert.nodes[cert.root_node as usize] else {
            panic!("fixture root must be the reduced Group-2 AND");
        };
        assert!(!root.edges.is_empty());
        eprintln!(
            "G2_SELECTOR_FIXTURE root_edges={} full_legal={}",
            root.edges.len(),
            legal.len()
        );
        assert!(
            root.edges.len() * 2 < legal.len(),
            "expected a genuine fanout reduction: {} edges vs {} legal",
            root.edges.len(),
            legal.len()
        );
        // Scalars were derived, not guessed.
        assert_eq!(root.proof.schema_version, 1);
        assert!(root.proof.authority.matches_compiled());
        assert_eq!(root.proof.build_horizon, cert.semantic_horizon);
    }

    #[test]
    fn flag_on_and_off_agree_on_fixture_verdicts() {
        let (_, on) = solve_fixture(true);
        let (_, off) = solve_fixture(false);
        assert_eq!(on.status, off.status);
        assert_eq!(on.status, ProofStatus::Loss);
        // Flag-off must not contain any extension node.
        assert!(!off
            .cert
            .expect("flag-off decides the fixture too")
            .nodes
            .iter()
            .any(CertNode::is_group2_extension));
    }

    #[test]
    fn flag_off_solver_is_deterministically_identical_across_runs() {
        let (_, first) = solve_fixture(false);
        let (_, second) = solve_fixture(false);
        assert_eq!(first.status, second.status);
        assert_eq!(first.cert, second.cert);
        assert_eq!(first.stats.nodes, second.stats.nodes);
        assert_eq!(first.stats.expansions, second.stats.expansions);
    }

    // ----- Phase A: mutation battery (every mutation must reject) -----

    fn root_proof_mut(cert: &mut TssCertificate) -> &mut Group2ZoneV1 {
        let root_node = cert.root_node as usize;
        let CertNode::UniversalGroup2V1(node) = &mut cert.nodes[root_node] else {
            panic!("fixture root must be Group-2");
        };
        &mut node.proof
    }

    #[test]
    fn group2_mutations_reject() {
        let (state, cert) = accepted_group2_cert();
        let reject = |label: &str, mutated: &TssCertificate| {
            assert!(
                !Group2Verifier.verify(&state, mutated, ProofStatus::Loss),
                "mutation {label} must reject"
            );
        };

        let mut schema = cert.clone();
        root_proof_mut(&mut schema).schema_version = 2;
        reject("schema_version", &schema);

        let mut commit = cert.clone();
        root_proof_mut(&mut commit).authority.defender_commit[0] ^= 0x01;
        reject("authority_commit_byte", &commit);

        let mut path = cert.clone();
        root_proof_mut(&mut path).authority.fhw_path = "PROOF_TSS_ZONES_FHW.mD".into();
        reject("authority_path", &path);

        let mut sha = cert.clone();
        root_proof_mut(&mut sha).authority.defender_sha256[31] ^= 0x80;
        reject("authority_sha_byte", &sha);

        let mut budget = cert.clone();
        root_proof_mut(&mut budget).claimed_d14_budget += 1;
        reject("claimed_d14_budget", &budget);

        let mut horizon = cert.clone();
        root_proof_mut(&mut horizon).build_horizon += 1;
        reject("build_horizon", &horizon);

        let mut plan = cert.clone();
        root_proof_mut(&mut plan).child_plan_sha256[0] ^= 0x01;
        reject("child_plan_sha256", &plan);

        let mut summary = cert.clone();
        root_proof_mut(&mut summary).finder_summary_sha256[0] ^= 0x01;
        reject("finder_summary_sha256", &summary);

        // FHW-O1-style omitted required reply (hostile review C2, gate-free
        // analogue): deleting one explicit edge (and pruning its orphaned
        // subtree, so only the coverage rule can be blamed) must reject.
        {
            let root_node = cert.root_node as usize;
            let edge_total = {
                let CertNode::UniversalGroup2V1(root) = &cert.nodes[root_node] else {
                    unreachable!()
                };
                root.edges.len()
            };
            for drop_index in 0..edge_total {
                let mut omitted = cert.clone();
                {
                    let CertNode::UniversalGroup2V1(node) = &mut omitted.nodes[root_node] else {
                        unreachable!()
                    };
                    node.edges.remove(drop_index);
                }
                let (nodes, new_root) = prune_reachable(&omitted);
                omitted.nodes = nodes;
                omitted.root_node = new_root;
                reject("omitted_required_reply", &omitted);
            }
        }

        // Canonical order violation: swap the first two root edges.
        {
            let mut swapped = cert.clone();
            let root_node = swapped.root_node as usize;
            let CertNode::UniversalGroup2V1(node) = &mut swapped.nodes[root_node] else {
                unreachable!()
            };
            if node.edges.len() >= 2 {
                node.edges.swap(0, 1);
                reject("noncanonical_edge_order", &swapped);
            }
        }

        // Horizon understatement below the derived resolution.
        let mut short = cert.clone();
        short.semantic_horizon -= 1;
        reject("semantic_horizon_below_derived_t", &short);

        // Root binding tamper.
        let mut binding = cert.clone();
        binding.root.placements_made -= 1;
        reject("root_binding", &binding);

        // Claimant flip.
        let mut claimant = cert.clone();
        claimant.claimant = claimant.claimant.other();
        reject("claimant_flip", &claimant);

        // No-mixing rule (class rules 2/3): a certificate containing BOTH an
        // extension node and a legacy zoned Universal must reject, regardless
        // of anything else. Splice a zoned Universal between the root and its
        // first subtree.
        {
            let mut mixed = cert.clone();
            let root_node = mixed.root_node as usize;
            let (first_mv, first_child) = {
                let CertNode::UniversalGroup2V1(node) = &mixed.nodes[root_node] else {
                    unreachable!()
                };
                (node.edges[0].mv, node.edges[0].child)
            };
            let spliced = mixed.nodes.len() as u32;
            mixed.nodes.push(CertNode::Universal {
                edges: vec![CertEdge {
                    mv: first_mv,
                    child: first_child,
                }],
                implicit_dispatch: false,
                zone: Some(ZoneInfo {
                    d: 3,
                    build_horizon: mixed.semantic_horizon,
                }),
                commutations: Vec::new(),
            });
            let CertNode::UniversalGroup2V1(node) = &mut mixed.nodes[root_node] else {
                unreachable!()
            };
            node.edges[0].child = spliced;
            reject("legacy_zone_mixing", &mixed);
        }

        // Implicit-dispatch mixing rejects the same way.
        {
            let mut mixed = cert.clone();
            let root_node = mixed.root_node as usize;
            let (first_mv, first_child) = {
                let CertNode::UniversalGroup2V1(node) = &mixed.nodes[root_node] else {
                    unreachable!()
                };
                (node.edges[0].mv, node.edges[0].child)
            };
            let spliced = mixed.nodes.len() as u32;
            mixed.nodes.push(CertNode::Universal {
                edges: vec![CertEdge {
                    mv: first_mv,
                    child: first_child,
                }],
                implicit_dispatch: true,
                zone: None,
                commutations: Vec::new(),
            });
            let CertNode::UniversalGroup2V1(node) = &mut mixed.nodes[root_node] else {
                unreachable!()
            };
            node.edges[0].child = spliced;
            reject("implicit_dispatch_mixing", &mixed);
        }
    }

    /// Rebuild `cert.nodes` keeping only nodes reachable from the root,
    /// remapping children densely (used to make single-edge deletions
    /// arena-valid so coverage is the only possible rejection reason).
    fn prune_reachable(cert: &TssCertificate) -> (Vec<CertNode>, u32) {
        fn copy(nodes: &[CertNode], id: u32, out: &mut Vec<CertNode>) -> u32 {
            let node = nodes[id as usize].clone();
            let mapped = match node {
                CertNode::Choice { mv, child } => CertNode::Choice {
                    mv,
                    child: copy(nodes, child, out),
                },
                CertNode::Universal {
                    edges,
                    implicit_dispatch,
                    zone,
                    commutations,
                } => CertNode::Universal {
                    edges: edges
                        .iter()
                        .map(|edge| CertEdge {
                            mv: edge.mv,
                            child: copy(nodes, edge.child, out),
                        })
                        .collect(),
                    implicit_dispatch,
                    zone,
                    commutations,
                },
                CertNode::UniversalGroup2V1(node) => {
                    CertNode::UniversalGroup2V1(Box::new(UniversalGroup2NodeV1 {
                        edges: node
                            .edges
                            .iter()
                            .map(|edge| CertEdge {
                                mv: edge.mv,
                                child: copy(nodes, edge.child, out),
                            })
                            .collect(),
                        proof: node.proof.clone(),
                    }))
                }
                other => other,
            };
            let new_id = out.len() as u32;
            out.push(mapped);
            new_id
        }
        let mut out = Vec::new();
        let root = copy(&cert.nodes, cert.root_node, &mut out);
        (out, root)
    }

    // ----- R1 / R2 amendment tests -----

    #[test]
    fn r1_gate_escape_deadline_above_horizon_rejects() {
        // Mandatory R1 test: semantic_horizon equals the leaf-derived
        // maximum but is strictly below one gate p(Q)+b+2. In the current
        // narrowed class ANY gate-bearing certificate rejects (a fortiori);
        // this pins the behavior, and `certificate_metadata` additionally
        // folds the escape deadline into derived T (R1 rule 1).
        let (state, cert) = accepted_group2_cert();
        let mut gated = cert.clone();
        // Splice a gate between the root and its first subtree: replace the
        // first root edge child with a gate whose representative points at
        // the original child.
        let root_node = gated.root_node as usize;
        let (first_mv, first_child) = {
            let CertNode::UniversalGroup2V1(node) = &gated.nodes[root_node] else {
                panic!("fixture root must be Group-2")
            };
            (node.edges[0].mv, node.edges[0].child)
        };
        let gate_id = gated.nodes.len() as u32;
        gated
            .nodes
            .push(CertNode::FhwGateV1(Box::new(FhwGateNodeV1 {
                representatives: vec![CertEdge {
                    mv: first_mv,
                    child: first_child,
                }],
                proof: FhwGateProofV1 {
                    schema_version: 1,
                    authority: crate::tss_verify::Group2AuthorityV1::compiled(),
                    threats: Vec::new(),
                    // Strictly above the leaf-derived max == semantic_horizon.
                    escape_resolution_ply: gated.semantic_horizon + 1,
                    map: Vec::new(),
                },
            })));
        {
            let CertNode::UniversalGroup2V1(node) = &mut gated.nodes[root_node] else {
                unreachable!()
            };
            node.edges[0].child = gate_id;
        }
        assert!(!Group2Verifier.verify(&state, &gated, ProofStatus::Loss));
        // R1 rule 1: the derived resolution now includes the escape deadline.
        let (derived_t, _) = crate::tss_verify::certificate_horizon_preflight(&gated)
            .expect("metadata stays computable");
        assert!(derived_t > gated.semantic_horizon);

        // Same construction with the deadline INSIDE the horizon still
        // rejects (gates are wholesale-narrowed out of the accepted class).
        let mut inside = gated.clone();
        let CertNode::FhwGateV1(gate) = &mut inside.nodes[gate_id as usize] else {
            unreachable!()
        };
        gate.proof.escape_resolution_ply = inside.semantic_horizon;
        assert!(!Group2Verifier.verify(&state, &inside, ProofStatus::Loss));
    }

    #[test]
    fn r2_opening_root_with_extension_node_rejects() {
        // Mandatory R2 test: a certificate whose ROOT is the Opening
        // placement and which contains a new-class node anywhere below must
        // reject by the explicit post-opening-root rule. The rule fires in
        // `verify_group2_impl` before replay/Z4 ever runs (source order), so
        // the rejection cannot be the accidental empty-board anchor failure.
        let opening = RustHexoState::new();
        let cert = TssCertificate {
            root: RootBinding::from_state(&opening),
            claimant: Player::Player1,
            root_node: 1,
            nodes: vec![
                CertNode::UniversalGroup2V1(Box::new(UniversalGroup2NodeV1 {
                    edges: vec![CertEdge {
                        mv: HexCoord::new(1, 0),
                        child: 2,
                    }],
                    proof: Group2ZoneV1 {
                        schema_version: 1,
                        authority: Group2AuthorityV1::compiled(),
                        claimed_d14_budget: 0,
                        build_horizon: 64,
                        child_plan_sha256: [0u8; 32],
                        finder_summary_sha256: [0u8; 32],
                    },
                })),
                CertNode::Choice {
                    mv: HexCoord::ZERO,
                    child: 0,
                },
                CertNode::Loss {
                    witnesses: vec![WindowKey {
                        start: HexCoord::ZERO,
                        axis: Axis::Q,
                    }],
                    resolution_ply: 8,
                },
            ],
            semantic_horizon: 64,
        };
        assert!(!Group2Verifier.verify(&opening, &cert, ProofStatus::Loss));
        assert!(!Group2Verifier.verify(&opening, &cert, ProofStatus::Win));
        assert!(!TssVerifier.verify(&opening, &cert, ProofStatus::Loss));
    }

    // ----- Legacy interop -----

    #[test]
    fn legacy_certificates_verify_identically_under_both_policies() {
        let (state, result) = solve_fixture(false);
        let cert = result.cert.expect("legacy solve decides the fixture");
        assert!(TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(Group2Verifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Win));
        assert!(!Group2Verifier.verify(&state, &cert, ProofStatus::Win));
    }

    #[test]
    fn legacy_policy_rejects_extension_certificates() {
        let (state, cert) = accepted_group2_cert();
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
    }

    // ----- D6 invariance of the extension class -----

    #[test]
    fn all_d6_images_of_group2_certificate_verify() {
        let (_, cert) = accepted_group2_cert();
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let transformed_state = {
                let mut state = RustHexoState::new();
                for &(q, r) in &FIXTURE_MOVES {
                    let coord =
                        d6_transform_coord(HexCoord::new(q, r), symmetry).expect("in range");
                    apply_placement(&mut state, Placement { coord }).expect("legal");
                }
                state
            };
            let transformed_cert =
                d6_remap_certificate(&cert, symmetry).expect("remap stays in range");
            assert!(
                Group2Verifier.verify(&transformed_state, &transformed_cert, ProofStatus::Loss),
                "symmetry {symmetry} image must verify (stored digests are D6-invariant)"
            );
        }
    }

    // ----- Verifier-side FHW gate machinery (§3.2/§3.3) unit tests -----
    // These mirror the finder's classifier tests but exercise the INDEPENDENT
    // verifier-side reimplementation (transversal_exact, VGhost, FC/GI/RC/WC,
    // classify_role, classify_window), including the two documented design
    // defects realized as verifier-side rejections.

    fn cluster_state() -> RustHexoState {
        let mut state = RustHexoState::new();
        let moves = [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (0, 2),
        ];
        for (q, r) in moves {
            apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) })
                .unwrap_or_else(|e| panic!("cluster move ({q},{r}) illegal: {e:?}"));
        }
        state
    }

    fn cluster_ghost() -> VGhost {
        VGhost::new(&cluster_state(), HexCoord::new(4, 0)).expect("ghost")
    }

    fn win(start: (i16, i16), axis: Axis) -> WindowKey {
        WindowKey {
            start: HexCoord::new(start.0, start.1),
            axis,
        }
    }

    fn geom(d_alive: bool, touched: bool, all_empty: bool, cnt_d: u32) -> WindowGeomV {
        WindowGeomV {
            d_alive,
            touched,
            all_empty,
            cnt_d,
        }
    }

    #[test]
    fn transversal_exact_small_families() {
        let empty: Vec<Vec<HexCoord>> = vec![];
        assert_eq!(transversal_exact(&empty, 3), 0);
        let one = vec![vec![HexCoord::new(0, 0), HexCoord::new(1, 0)]];
        assert_eq!(transversal_exact(&one, 3), 1);
        let common = vec![
            vec![HexCoord::new(0, 0), HexCoord::new(1, 0)],
            vec![HexCoord::new(0, 0), HexCoord::new(5, 5)],
        ];
        assert_eq!(transversal_exact(&common, 3), 1);
        let disjoint = vec![
            vec![HexCoord::new(0, 0), HexCoord::new(1, 0)],
            vec![HexCoord::new(5, 5), HexCoord::new(6, 5)],
        ];
        assert_eq!(transversal_exact(&disjoint, 3), 2);
        let three = vec![
            vec![HexCoord::new(0, 0)],
            vec![HexCoord::new(5, 5)],
            vec![HexCoord::new(9, 0)],
        ];
        assert_eq!(transversal_exact(&three, 2), 3); // "> 2"
    }

    #[test]
    fn fc_and_gi_predicates() {
        let ghost = cluster_ghost();
        let d = HexCoord::new(1, 0);
        assert!(frontier_covered(d, d, &ghost)); // d == s is Exact/FC
        assert!(!frontier_covered(HexCoord::new(40, 0), HexCoord::new(4, 0), &ghost));
        assert!(ghost.is_ghost_illegal(HexCoord::new(40, 0)));
        assert!(!ghost.is_ghost_illegal(HexCoord::new(4, 1)));
        assert!(!ghost.is_ghost_illegal(HexCoord::new(0, 0))); // occupied
    }

    #[test]
    fn role_rows_exact_fc_and_rc_zero() {
        let ghost = cluster_ghost();
        let d = HexCoord::new(1, 0);
        let y = HexCoord::new(5, 0);
        assert_eq!(
            classify_role(FhwEdgeClassV1::Exact, d, y, 3, None),
            Some((FhwRoleRowV1::ExactOrFcZero, 0))
        );
        assert_eq!(
            classify_role(FhwEdgeClassV1::FrontierCovered, d, y, 3, None),
            Some((FhwRoleRowV1::ExactOrFcZero, 0))
        );
        // Carrier avoidance: d == y rejects.
        assert_eq!(classify_role(FhwEdgeClassV1::Exact, d, d, 3, None), None);
        // Non-FC ghost-illegal carrier, k=0 => RC passes => NonFcRcZero.
        let yi = HexCoord::new(40, 0);
        assert!(ghost.is_ghost_illegal(yi));
        assert_eq!(
            classify_role(FhwEdgeClassV1::NonFrontierCovered, d, yi, 0, Some(&ghost)),
            Some((FhwRoleRowV1::NonFcRcZero, 0))
        );
        // Non-FC ghost-legal carrier => conservative charged, eps 1.
        let yl = HexCoord::new(4, 1);
        assert!(!ghost.is_ghost_illegal(yl));
        assert_eq!(
            classify_role(FhwEdgeClassV1::NonFrontierCovered, d, yl, 2, Some(&ghost)),
            Some((FhwRoleRowV1::NonFcCharged, 1))
        );
    }

    #[test]
    fn defect_charged_via_ghost_illegal_rc_fail_is_unrealizable() {
        // DEFECT 1 (from the closure report): RC fails => dist(d,y)<=8k, which
        // contradicts the mandatory D22-N guard dist(d,y)>8k. So a charged row
        // via a ghost-illegal RC-fail carrier is never accepted — it REJECTS.
        let ghost = cluster_ghost();
        let y = HexCoord::new(40, 0); // ghost-illegal
        let d = HexCoord::new(35, 0); // within 8 of y
        let out = classify_role(FhwEdgeClassV1::NonFrontierCovered, d, y, 1, Some(&ghost));
        assert_eq!(out, None, "unrealizable charged-via-illegal claim must reject");
    }

    #[test]
    fn window_rows_exact_fc_paths() {
        // Exact/FC window rows never consult the ghost (None is passed).
        let w = win((10, 0), Axis::Q); // cells (10,0)..(15,0)
        // NonDAlive.
        let g = geom(false, false, false, 0);
        assert_eq!(
            classify_window(FhwEdgeClassV1::Exact, HexCoord::new(10, 0), w, 3, &g, None),
            Some((FhwKappaRowV1::NonDAlive, 0, GuardResultV1::NotApplicable))
        );
        // ExactOrFcNonIncident.
        let g = geom(true, false, true, 0);
        assert_eq!(
            classify_window(FhwEdgeClassV1::FrontierCovered, HexCoord::new(0, 0), w, 4, &g, None),
            Some((FhwKappaRowV1::ExactOrFcNonIncident, 0, GuardResultV1::NotApplicable))
        );
        // ExactOrFcDirect touched: guard cnt_d+1+q<6 passes at q=2, fails at q=3.
        let touched = geom(true, true, false, 2);
        let d = HexCoord::new(12, 0);
        assert_eq!(
            classify_window(FhwEdgeClassV1::Exact, d, w, 2, &touched, None),
            Some((FhwKappaRowV1::ExactOrFcDirect, 1, GuardResultV1::Pass))
        );
        assert_eq!(
            classify_window(FhwEdgeClassV1::Exact, d, w, 3, &touched, None),
            None
        );
        // ExactOrFcDirect all-empty: 1+q<6 passes at q=4, fails at q=5.
        let empty = geom(true, false, true, 0);
        assert_eq!(
            classify_window(FhwEdgeClassV1::FrontierCovered, d, w, 4, &empty, None),
            Some((FhwKappaRowV1::ExactOrFcDirect, 1, GuardResultV1::Pass))
        );
        assert_eq!(
            classify_window(FhwEdgeClassV1::FrontierCovered, d, w, 5, &empty, None),
            None
        );
    }

    #[test]
    fn window_rows_non_fc_and_wc() {
        let state = cluster_state();
        let ghost = VGhost::new(&state, HexCoord::new(4, 0)).unwrap();
        // NonFcEmptyNonIncidentQlt6.
        let w = win((10, 0), Axis::Q);
        let empty = geom(true, false, true, 0);
        assert_eq!(
            classify_window(
                FhwEdgeClassV1::NonFrontierCovered,
                HexCoord::new(0, 0),
                w,
                5,
                &empty,
                Some(&ghost)
            ),
            Some((FhwKappaRowV1::NonFcEmptyNonIncidentQlt6, 0, GuardResultV1::NotApplicable))
        );
        // WcPass at q=6 with a window whose cells are legal (near cluster).
        let w2 = win((4, 2), Axis::Q);
        for c in w2.cells() {
            assert!(!ghost.is_ghost_illegal(c));
        }
        let d2 = HexCoord::new(2, 2);
        assert!(!w2.contains(d2));
        assert_eq!(
            classify_window(FhwEdgeClassV1::NonFrontierCovered, d2, w2, 6, &empty, Some(&ghost)),
            Some((FhwKappaRowV1::NonFcEmptyNonIncidentWcPass, 0, GuardResultV1::Pass))
        );
    }

    #[test]
    fn defect_wc_fail_leaf_is_unrealizable_with_passing_guard() {
        // DEFECT 2: on the non-FC/all-empty/nonincident/q>=6 branch, WC fail =>
        // dist(d,W)<=8(q-5), contradicting the N-virgin guard dist(d,W)>8(q-5).
        // So the WcFail leaf is never accepted with a passing guard: the result
        // is WcPass or a rejection, never a passing WcFail row.
        let state = cluster_state();
        let ghost = VGhost::new(&state, HexCoord::new(4, 0)).unwrap();
        let d = HexCoord::new(10, 0);
        let w = win((11, 0), Axis::Q); // dist(d,W)=1
        let empty = geom(true, false, true, 0);
        let out = classify_window(FhwEdgeClassV1::NonFrontierCovered, d, w, 7, &empty, Some(&ghost));
        assert!(
            matches!(
                out,
                None | Some((FhwKappaRowV1::NonFcEmptyNonIncidentWcPass, _, _))
            ),
            "WcFail must not be accepted with a passing guard; got {out:?}"
        );
    }

    // ----- End-to-end positive gate fixture + mutation battery -----
    // A hand-built double-threat gate: defender P0 to move (SecondStone, b=1);
    // claimant P1 holds two disjoint count-4 windows U (the named gate threat)
    // and V. K = E(U) = {(4,1),(5,1)}; each K reply blocks U, after which P1
    // wins via V (a Win leaf, count 4 / budget 2). An all-Exact gate (R = K)
    // that exercises reconstruction, checkpoint roles, paired clocks, the
    // Cartesian window demand, role/window rows, and escape horizon.

    /// The 16-move prefix reaching P0-to-move SecondStone with the two threats.
    const GATE_MOVES: [(i16, i16); 16] = [
        (0, 0),   // 1  P0 opening
        (0, 1),   // 2  P1 U
        (1, 1),   // 3  P1 U
        (-1, 0),  // 4  P0
        (-1, -1), // 5  P0
        (2, 1),   // 6  P1 U
        (3, 1),   // 7  P1 U  (U = (0,1)..(3,1), empties (4,1),(5,1))
        (0, -1),  // 8  P0
        (-2, 0),  // 9  P0
        (0, 3),   // 10 P1 V
        (1, 3),   // 11 P1 V
        (-2, -1), // 12 P0
        (1, -2),  // 13 P0
        (2, 3),   // 14 P1 V
        (3, 3),   // 15 P1 V  (V = (0,3)..(3,3), empties (4,3),(5,3))
        (2, -2),  // 16 P0
    ];

    fn gate_position() -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in &GATE_MOVES {
            apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) })
                .unwrap_or_else(|e| panic!("gate move ({q},{r}) illegal: {e:?}"));
        }
        state
    }

    /// Build the accepting gate certificate (skeleton + filled rows).
    fn accepted_gate_cert() -> (RustHexoState, TssCertificate) {
        let state = gate_position();
        assert_eq!(state.current_player(), Player::Player0, "defender to move");
        assert!(matches!(state.phase(), TurnPhase::SecondStone { .. }), "b=1");
        let u = win((0, 1), Axis::Q);
        let v = win((0, 3), Axis::Q);
        // Sanity: U/V are P1 count-4 windows with 0 P0 stones.
        let p1_count = |w: WindowKey| {
            w.cells()
                .iter()
                .filter(|c| state.board().get(**c) == Some(Player::Player1))
                .count()
        };
        assert_eq!(p1_count(u), 4, "U must be a P1 count-4 window");
        assert_eq!(p1_count(v), 4, "V must be a P1 count-4 window");

        let placements = state.placements_made();
        let escape = placements + 1 + 2; // p(Q) + b + 2
        let win_resolution = placements + 1 + 2; // after s: placements+1, +2 to win
        let horizon = escape.max(win_resolution);

        // Win leaves proving P1 wins via V after each K reply.
        let win_leaf = || CertNode::Win {
            witness: v,
            count: 4,
            budget: 2,
            resolution_ply: win_resolution,
        };
        let u1 = HexCoord::new(4, 1);
        let u2 = HexCoord::new(5, 1);
        let skeleton_map = |d: HexCoord| crate::tss_verify::FhwMapV1 {
            real_reply: d,
            representative: d,
            edge_class: FhwEdgeClassV1::Exact,
            roles: Vec::new(),
            windows: Vec::new(),
        };
        let gate = CertNode::FhwGateV1(Box::new(crate::tss_verify::FhwGateNodeV1 {
            representatives: vec![
                CertEdge { mv: u1, child: 1 },
                CertEdge { mv: u2, child: 2 },
            ],
            proof: FhwGateProofV1 {
                schema_version: 1,
                authority: Group2AuthorityV1::compiled(),
                threats: vec![u],
                escape_resolution_ply: escape,
                map: vec![skeleton_map(u1), skeleton_map(u2)],
            },
        }));
        let cert = TssCertificate {
            root: RootBinding::from_state(&state),
            claimant: Player::Player1,
            root_node: 0,
            nodes: vec![gate, win_leaf(), win_leaf()],
            semantic_horizon: horizon,
        };
        let filled =
            finder_fill_gate_rows(&state, &cert).expect("gate skeleton must reconstruct + fill");
        (state, filled)
    }

    #[test]
    fn gate_certificate_reconstructs_and_verifies() {
        let (state, cert) = accepted_gate_cert();
        // The reconstruction found exactly K = {(4,1),(5,1)} with two Exact
        // self-edges.
        let CertNode::FhwGateV1(gate) = &cert.nodes[0] else {
            panic!("root must be the gate");
        };
        assert_eq!(gate.proof.map.len(), 2, "|K| == 2");
        assert!(gate
            .proof
            .map
            .iter()
            .all(|m| m.edge_class == FhwEdgeClassV1::Exact));
        // Every map entry carries the Cartesian window domain (all equal).
        let domain: std::collections::HashSet<_> = gate.proof.map[0]
            .windows
            .iter()
            .map(|w| window_sort_key(w.window))
            .collect();
        assert!(!domain.is_empty(), "direct-18 demands present");
        for m in &gate.proof.map {
            let d: std::collections::HashSet<_> =
                m.windows.iter().map(|w| window_sort_key(w.window)).collect();
            assert_eq!(d, domain, "Cartesian K x demands: identical window domain");
        }
        // Strict acceptance under the extension policy; rejection under legacy.
        assert!(
            Group2Verifier.verify(&state, &cert, ProofStatus::Loss),
            "the constructed gate certificate must verify"
        );
        assert!(!TssVerifier.verify(&state, &cert, ProofStatus::Loss));
        assert!(!Group2Verifier.verify(&state, &cert, ProofStatus::Win));
    }

    #[test]
    fn gate_mutation_battery_rejects() {
        let (state, cert) = accepted_gate_cert();
        let reject = |label: &str, mutated: &TssCertificate| {
            assert!(
                !Group2Verifier.verify(&state, mutated, ProofStatus::Loss),
                "gate mutation {label} must reject"
            );
        };
        macro_rules! with_gate {
            ($m:ident, $g:ident, $body:block) => {{
                if let CertNode::FhwGateV1($g) = &mut $m.nodes[0] {
                    $body
                }
            }};
        }

        // schema version.
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.proof.schema_version = 2;
        });
        reject("schema_version", &m);

        // authority byte.
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.proof.authority.fhw_sha256[0] ^= 0x01;
        });
        reject("authority_sha", &m);

        // escape horizon: below p(Q)+b+2.
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.proof.escape_resolution_ply -= 1;
        });
        reject("escape_ply", &m);

        // threat window tamper (H_Q no longer a real A-threat / tau != b).
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.proof.threats[0] = win((9, 9), Axis::Q);
        });
        reject("threat_window", &m);

        // K-domain: drop one map entry (map domain != K).
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.proof.map.pop();
        });
        reject("map_domain_short", &m);

        // representative move tamper (R no longer subset of K).
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.representatives[0].mv = HexCoord::new(9, 9);
        });
        reject("representative_move", &m);

        // edge class flip (Exact -> FrontierCovered contradicts geometry).
        let mut m = cert.clone();
        with_gate!(m, g, {
            g.proof.map[0].edge_class = FhwEdgeClassV1::FrontierCovered;
        });
        reject("edge_class", &m);

        // role row tamper.
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            if let Some(r) = g.proof.map[0].roles.first_mut() {
                r.epsilon ^= 1;
            }
        }
        reject("role_epsilon", &m);

        // role child_f tamper.
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            if let Some(r) = g.proof.map[0].roles.first_mut() {
                r.child_f += 1;
            }
        }
        reject("role_child_f", &m);

        // window kappa tamper.
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            if let Some(w) = g.proof.map[0].windows.first_mut() {
                w.kappa ^= 1;
            }
        }
        reject("window_kappa", &m);

        // window child_q tamper.
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            if let Some(w) = g.proof.map[0].windows.first_mut() {
                w.child_q += 1;
            }
        }
        reject("window_child_q", &m);

        // drop a window row (Cartesian incomplete).
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            g.proof.map[0].windows.pop();
        }
        reject("window_domain_short", &m);

        // representative child tamper: point at the wrong leaf (loss instead).
        let mut m = cert.clone();
        m.nodes[1] = CertNode::Loss {
            witnesses: vec![win((0, 3), Axis::Q)],
            resolution_ply: cert.semantic_horizon,
        };
        reject("representative_child_swapped_to_loss", &m);

        // horizon below derived T.
        let mut m = cert.clone();
        m.semantic_horizon -= 1;
        reject("semantic_horizon", &m);

        // claimant flip.
        let mut m = cert.clone();
        m.claimant = m.claimant.other();
        reject("claimant_flip", &m);

        // duplicate representative / map entry.
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            g.proof.map[1] = g.proof.map[0].clone();
        }
        reject("duplicate_map_entry", &m);
    }

    #[test]
    fn gate_certificate_is_d6_invariant() {
        let (_, cert) = accepted_gate_cert();
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let transformed_state = {
                let mut s = RustHexoState::new();
                for &(q, r) in &GATE_MOVES {
                    let coord = d6_transform_coord(HexCoord::new(q, r), symmetry).expect("range");
                    apply_placement(&mut s, Placement { coord }).expect("legal");
                }
                s
            };
            let transformed_cert = d6_remap_certificate(&cert, symmetry).expect("remap");
            assert!(
                Group2Verifier.verify(&transformed_state, &transformed_cert, ProofStatus::Loss),
                "gate symmetry {symmetry} image must verify"
            );
        }
    }

    #[test]
    fn gate_with_nonfrontiercovered_edge_rejects() {
        // Accept-path narrowing: if a map entry claims NonFrontierCovered, the
        // reconstruction rejects (the non-FC end-to-end path is fail-closed).
        let (state, cert) = accepted_gate_cert();
        let mut m = cert.clone();
        if let CertNode::FhwGateV1(g) = &mut m.nodes[0] {
            g.proof.map[0].edge_class = FhwEdgeClassV1::NonFrontierCovered;
        }
        assert!(!Group2Verifier.verify(&state, &m, ProofStatus::Loss));
    }

    #[test]
    fn window_geom_is_exact() {
        // Both incidence bits and touched/all-empty derive from the board.
        let state = cluster_state();
        // (0,0) is P0-occupied, (1,0)/(2,0)/(3,0) are P1. Window (0,0)-(5,0).
        let w = win((0, 0), Axis::Q);
        // claimant = P0: window has P0 stone at (0,0) => non-D-alive.
        let g = window_geom(&state, Player::Player0, w);
        assert!(!g.d_alive);
        // claimant = P1: (0,0) is P0 => defender(P0) present, no P1 => but P1 is
        // claimant; window has P1 at (1,0),(2,0),(3,0) => claimant present =>
        // non-D-alive for P1 too. Use a fully empty far window for all-empty.
        let far = win((20, 0), Axis::Q);
        let g2 = window_geom(&state, Player::Player1, far);
        assert!(g2.d_alive && g2.all_empty && !g2.touched);
    }
}
