//! Data-only `RefuteLeafExact/V1` wire definitions and reviewed SHA-256.
//!
//! Semantic generation and semantic decoding deliberately do not live here:
//! the producer and verifier each own those operations.

use hexo_engine::{HexoState, Player, TurnPhase};

pub const MAGIC: &[u8; 8] = b"HXRFLV1\0";
pub const ROOT_DOMAIN: &[u8; 25] = b"HXRFLV1:ROOT-SEMANTIC:V1\0";
pub const FORMAT_V1: u16 = 1;
pub const RULESET_V1: u16 = 1;
pub const COORDINATE_V1: u16 = 1;
pub const CLASS_V1: u16 = 1;
pub const PROFILE_V1: u16 = 1;
pub const TAG_NO_ADMISSIBLE_FIRST_TURN: u8 = 0x20;

pub const MAX_WIRE_BYTES: usize = 8 << 20;
pub const MAX_ROOT_STONES: u64 = 4_096;
pub const MAX_WINDOWS: u64 = 80_000;
pub const MAX_T: u64 = 4_096;
pub const MAX_S: u64 = 4_096;
pub const MAX_Q: u64 = 2_000_000;
pub const MAX_THREAT_MEMBERSHIPS: u64 = 8_000_000;
pub const MAX_PAIR_OPS: u64 = 16_000_000;
pub const MAX_TRANSVERSAL_OPS: u64 = 8_000_000;
pub const MAX_STATE_BYTES: u64 = 64 << 20;
pub const MAX_HEAP_BYTES: u64 = 256 << 20;
pub const MAX_CPU_MS: u64 = 30_000;
pub const MAX_WALL_MS: u64 = 60_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireStoneV1 {
    pub q: i16,
    pub r: i16,
    pub owner: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootHeaderV1 {
    pub stones: Vec<WireStoneV1>,
    pub mover: u8,
    pub phase: u8,
    pub phase_first: Option<(i16, i16)>,
    pub placements_made: u32,
    pub terminal: u8,
    pub claimant: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeafCountsV1 {
    pub t_count: u64,
    pub q_count: u64,
    pub quotient_class_count: u64,
    pub fail_no_new: u64,
    pub fail_defender_first: u64,
    pub fail_loose_0: u64,
    pub fail_loose_1: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafArtifactV1 {
    pub root: RootHeaderV1,
    pub root_semantic_sha256: [u8; 32],
    pub counts: LeafCountsV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReachableRootV1 {
    ruleset: u16,
    root_semantic_sha256: [u8; 32],
    history_sha256: [u8; 32],
}

impl ReachableRootV1 {
    /// Mint the reachability premise from the trusted engine history API.
    /// Replaying the snapshot prevents a hand-mutated state from being blessed.
    pub fn from_trusted_engine_state(state: &HexoState) -> Option<Self> {
        let replayed = hexo_engine::load_state(&state.snapshot()).ok()?;
        if !same_engine_state(state, &replayed) {
            return None;
        }
        let root = root_header_from_engine(state)?;
        let root_semantic_sha256 = root_semantic_sha256(&root);
        let mut history = Vec::with_capacity(state.placement_history().len() * 10 + 8);
        history.extend_from_slice(b"HXRFLV1:HISTORY\0");
        for record in state.placement_history() {
            history.extend_from_slice(&record.coord.q.to_le_bytes());
            history.extend_from_slice(&record.coord.r.to_le_bytes());
            history.push(player_u8(record.player));
            history.push(phase_u8(record.phase));
            if let TurnPhase::SecondStone { first } = record.phase {
                history.extend_from_slice(&first.q.to_le_bytes());
                history.extend_from_slice(&first.r.to_le_bytes());
            }
            history.extend_from_slice(&record.placement_index.to_le_bytes());
        }
        Some(Self {
            ruleset: RULESET_V1,
            root_semantic_sha256,
            history_sha256: sha256(&history),
        })
    }

    pub(crate) fn matches(&self, state: &HexoState, digest: &[u8; 32]) -> bool {
        if self.ruleset != RULESET_V1 || &self.root_semantic_sha256 != digest {
            return false;
        }
        let Some(reminted) = Self::from_trusted_engine_state(state) else {
            return false;
        };
        reminted == *self
    }
}

fn same_engine_state(a: &HexoState, b: &HexoState) -> bool {
    if a.current_player() != b.current_player()
        || a.phase() != b.phase()
        || a.placements_made() != b.placements_made()
        || a.terminal() != b.terminal()
        || a.placement_history() != b.placement_history()
        || a.board().len() != b.board().len()
    {
        return false;
    }
    a.board()
        .occupied_cells()
        .iter()
        .all(|&c| a.board().get(c) == b.board().get(c))
}

pub(crate) fn player_u8(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

pub(crate) fn phase_u8(phase: TurnPhase) -> u8 {
    match phase {
        TurnPhase::Opening => 0,
        TurnPhase::FirstStone => 1,
        TurnPhase::SecondStone { .. } => 2,
    }
}

pub(crate) fn root_header_from_engine(state: &HexoState) -> Option<RootHeaderV1> {
    let mut stones = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&coord| {
            Some(WireStoneV1 {
                q: coord.q,
                r: coord.r,
                owner: player_u8(state.board().get(coord)?),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    stones.sort_by_key(|s| (s.q, s.r));
    let (phase, phase_first) = match state.phase() {
        TurnPhase::Opening => (0, None),
        TurnPhase::FirstStone => (1, None),
        TurnPhase::SecondStone { first } => (2, Some((first.q, first.r))),
    };
    Some(RootHeaderV1 {
        stones,
        mover: player_u8(state.current_player()),
        phase,
        phase_first,
        placements_made: state.placements_made(),
        terminal: u8::from(state.terminal().is_some()),
        claimant: player_u8(state.current_player()),
    })
}

pub fn root_semantic_preimage_v1(root: &RootHeaderV1) -> Vec<u8> {
    let mut out = Vec::with_capacity(25 + 10 + 16 + root.stones.len() * 5);
    out.extend_from_slice(ROOT_DOMAIN);
    for value in [RULESET_V1, COORDINATE_V1, CLASS_V1, FORMAT_V1, PROFILE_V1] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    put_uvar(&mut out, root.stones.len() as u64);
    for stone in &root.stones {
        out.extend_from_slice(&stone.q.to_le_bytes());
        out.extend_from_slice(&stone.r.to_le_bytes());
        out.push(stone.owner);
    }
    out.push(root.mover);
    out.push(root.phase);
    if root.phase == 2 {
        if let Some((q, r)) = root.phase_first {
            out.extend_from_slice(&q.to_le_bytes());
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    out.extend_from_slice(&root.placements_made.to_le_bytes());
    out.push(root.terminal);
    out.push(root.claimant);
    out
}

pub fn root_semantic_sha256(root: &RootHeaderV1) -> [u8; 32] {
    sha256(&root_semantic_preimage_v1(root))
}

pub(crate) fn put_uvar(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        out.push(if value == 0 { low } else { low | 0x80 });
        if value == 0 {
            return;
        }
    }
}

pub(crate) fn encode_artifact(artifact: &LeafArtifactV1) -> Vec<u8> {
    let root = &artifact.root;
    let mut out = Vec::with_capacity(128 + root.stones.len() * 5);
    out.extend_from_slice(MAGIC);
    for value in [FORMAT_V1, RULESET_V1, COORDINATE_V1, CLASS_V1, PROFILE_V1] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    put_uvar(&mut out, root.stones.len() as u64);
    for stone in &root.stones {
        out.extend_from_slice(&stone.q.to_le_bytes());
        out.extend_from_slice(&stone.r.to_le_bytes());
        out.push(stone.owner);
    }
    out.push(root.mover);
    out.push(root.phase);
    out.extend_from_slice(&root.placements_made.to_le_bytes());
    out.push(root.terminal);
    out.push(root.claimant);
    out.extend_from_slice(&artifact.root_semantic_sha256);
    let mut payload = Vec::with_capacity(64);
    payload.push(TAG_NO_ADMISSIBLE_FIRST_TURN);
    for value in [
        artifact.counts.t_count,
        artifact.counts.q_count,
        artifact.counts.quotient_class_count,
        artifact.counts.fail_no_new,
        artifact.counts.fail_defender_first,
        artifact.counts.fail_loose_0,
        artifact.counts.fail_loose_1,
    ] {
        put_uvar(&mut payload, value);
    }
    put_uvar(&mut out, payload.len() as u64);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&sha256(&payload));
    out
}

/// Compact competent baseline: the identical literal root/identity and a
/// one-byte empty-set payload protected by the same payload digest.
pub fn root_plus_empty_set_baseline_bytes(root: &RootHeaderV1) -> usize {
    let artifact = LeafArtifactV1 {
        root: root.clone(),
        root_semantic_sha256: root_semantic_sha256(root),
        counts: LeafCountsV1::default(),
    };
    let actual = encode_artifact(&artifact).len();
    // The v1 payload has one tag plus seven one-byte zero counters. The
    // competent empty-set baseline has one tag, so it is exactly 7 bytes less
    // (six payload bytes plus the payload-length shrink where applicable).
    actual.saturating_sub(7)
}

// Small, dependency-free SHA-256. This implementation is covered by FIPS
// vectors and the independent Python oracle goldens.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
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
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((input.len() + 72) & !63);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut out = [0u8; 32];
    for (chunk, value) in out.chunks_exact_mut(4).zip(h) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_fips_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn root_domain_is_literal_25_bytes() {
        assert_eq!(ROOT_DOMAIN.len(), 25);
        assert_eq!(
            hex(ROOT_DOMAIN),
            "485852464c56313a524f4f542d53454d414e5449433a563100"
        );
    }

    #[test]
    fn uvar_is_shortest_form() {
        let mut out = Vec::new();
        put_uvar(&mut out, 0);
        put_uvar(&mut out, 127);
        put_uvar(&mut out, 128);
        assert_eq!(out, [0, 127, 0x80, 1]);
    }
}
