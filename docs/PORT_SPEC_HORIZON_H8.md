# Rust port specification: exact horizon-H8 leaf verdicts

## 1. Scope and contract

**HYPOTHESIS -- proof-ready.** Port only the proven finite `h<=8` tier.  At a
nonterminal fresh `FirstStone` leaf with mover `A`, it returns one of:

```rust
enum Horizon8Verdict {
    ExactWin { first: HexCoord, second: Option<HexCoord> },
    ExactForcedLoss,
    NoVerdict,
}
```

`ExactWin` means `A` has a full-game win within six physical placements (and
therefore within eight); the pair is the winning first turn.  `second=None`
means the first placement terminates the game.  `ExactForcedLoss` means the
opponent forces a full-game win within eight placements.  `NoVerdict` includes
both `NoWinWithin8` and positions whose loss dual is false; it is never a draw,
game loss, or permission to prune a legal move.

Opening and `SecondStone` leaves initially return `NoVerdict`.  The measured
Python partial-turn algebra is useful, but is outside this first port.

## 2. Engine seam and invariants

**CODE-FACT.** `Board::windows()` exposes the incremental `WindowStore`.
`WindowStore::entries()` yields every touched `WindowEntry`; `key()`,
`count(player)`, `mask(player)`, `empty_mask()`, and `cells()` supply everything
needed.  A placement touches exactly 18 windows.  No board-radius scan is
needed.

At h6 the schedule is `A,A,D,D,A,A`, with quotas `(k_A,k_D)=(4,2)`.  At h8
the loss dual uses `(4,4)`.  Since every quota is below six, every completable
pure window contains a root stone and is already in `WindowStore`.  Build only
entries satisfying:

```text
entry.mask(other) == 0
entry.empty_mask().count_ones() <= quota[player]
```

The input state must be nonterminal, `phase()==FirstStone`, and
`current_player()==A`.  Assert all generated coordinates are empty.  Preserve
the rule that a win after the first placement terminates without a second.

## 3. Exact data layout

Use a per-call arena; no global cache is required for correctness.

```rust
type CellId = u16; // checked conversion; fall back to u32 if U exceeds u16::MAX

struct CellUniverse {
    coords: Vec<HexCoord>,                 // sorted (q,r), stable certificate order
    index: AHashMap<HexCoord, CellId>,
    words: usize,                          // ceil(coords.len()/64)
}

struct Bits {
    words: SmallVec<[u64; 5]>,             // dynamic exact fallback beyond 320 cells
}

struct RelevantWindow {
    key: WindowKey,
    empty: Bits,                           // root-empty cells, indexed in U
}

struct HorizonModel {
    universe: CellUniverse,
    attacker: Vec<RelevantWindow>,
    defender: Vec<RelevantWindow>,
    active_union: Bits,
}

#[derive(Clone, Copy)]
struct PairMask {
    first: CellId,
    second: Option<CellId>,                // None is terminal-prefix/inert fill
}
```

**MEASURED.** R2's largest observed U8 was 260 cells, so five inline `u64`
words cover every measured root.  The representation must nevertheless grow
dynamically: 260 is evidence, not a semantic bound.  For the hot measured
case, `Bits` operations are five wordwise instructions plus loop overhead.

Build U in two passes.  First collect qualifying `WindowKey` plus its six-bit
`empty_mask`; then insert each empty coordinate in a sorted/deduplicated vector,
build `index`, and translate each six-bit local mask into `Bits`.  Sorting makes
the model and future certificate digest independent of `AHashMap` iteration.

## 4. Pair enumeration

Enumerate unordered distinct relevant cells.  If U is empty, emit one inert
pair.  If one relevant cell remains, emit `(cell,None)`.  With at least two
relevant cells, enumerate every `i<j`; a relevant stone monotonically dominates
an inert placement.  Separately test every one-cell terminal completion so the
second coordinate is not required after a win.

Ordering is non-semantic: immediate completions first, then descending
`own_window_incidence + 2*opponent_window_incidence`, then `(CellId,CellId)`.
The fallback must still visit every pair.

## 5. H6 attacker pair-fork

Build the h6 model with quotas `(4,2)`.  For every attacker pair `a`:

1. If either legal prefix completes an attacker window, return `ExactWin{a}`.
2. Remove attacker windows hit by no attacker stone only by residual masking;
   discard defender windows blocked by `a`.
3. If a live defender window has at most two root-empty cells, D can win on the
   intervening pair; reject `a`.
4. Form `F_a = { W\a | W attacker-pure and 1 <= |W\a| <= 2 }`.
5. If `F_a` is nonempty and has no hitting set of size at most two, return the
   exact winning pair.

The size-two cover test is allocation-free: intersect all edges for a
one-cover; otherwise choose each bit `x` of the first edge, intersect the edges
not hit by `x`, and accept if that intersection is nonempty.  If all pairs
fail, the attacker result is `NoWinWithin8` because the mover has no placements
at physical plies seven and eight.

## 6. H8 forced-loss dual

Build the h8 model with quotas `(4,4)`.  For every normalized initial attacker
pair `a`:

1. If `a` completes A, return `NoVerdict`; this refutes a universal D strategy.
2. Enumerate every normalized D pair `d` over the live union.
3. `d` succeeds immediately if it completes D.
4. Otherwise reject `d` if A has a live residual of size at most two after
   `a+d`; A would win on physical plies five--six.
5. Form D's residual family with sizes one or two.  `d` succeeds when that
   family has no two-cover: every A pair leaves a D completion for plies
   seven--eight.
6. If no `d` succeeds for this `a`, return `NoVerdict`.  If every `a` has a
   successful `d`, return `ExactForcedLoss`.

This is the exact `forall a, exists d` interaction, including D pairs split
between blocking A and advancing D.  Do not substitute the one-ply standing
threat predicate.

## 7. Leaf integration

Add a pure module beside `threats_shared.rs`, for example
`horizon8_shared.rs`, and path-include the same file in dense_cnn, hexgt, and
hexfield_eq just as the shared threat implementation is included today.

At a newly selected nonterminal leaf, before network evaluation or node
creation:

```text
terminal outcome
existing transposition node
cheap one-ply threats::analyze verdict
fresh-FirstStone horizon8 verdict
network evaluation / node creation
```

The one-ply tier is a cheap strict subset and should remain first.  Back up
`+1.0/-1.0` from the leaf mover's perspective through the existing
`backup_virtual` path.  For `ExactWin`, optionally attach the witnessing pair to
diagnostics/move ordering; do not inject a synthetic two-placement tree edge.
For `NoVerdict`, continue unchanged.  Gate the tier independently from TSS so
an A/B run can measure its cost at visit cap 500.

## 8. Cost model and rollout gate

**MEASURED -- CPython, not Rust latency.** Fresh-root pair counts from R2 were:

| cohort | H6 attacker nodes p50 / p90 / max | mean wall | H8 loss nodes p50 / p90 / max | mean / max wall |
|---|---:|---:|---:|---:|
| self-play | 171 / 1,035 / 2,926 | 3.83 ms | 596 / 4,372 / 1,345,240 | 37.35 ms / 8.57 s |
| human | 210 / 990 / 9,730 | 4.45 ms | 904 / 18,590 / 6,042,901 | 113.54 ms / 32.63 s |
| puzzle | 190 / 903 / 2,926 | 2.86 ms | 1,100 / 7,377 / 1,140,310 | 55.20 ms / 4.50 s |
| grinds | 703 / 1,653 / 2,926 | 6.67 ms | 2,146 / 5,357 / 39,162 | 27.00 ms / 0.15 s |

**HYPOTHESIS.** The H6 bitset check is suitable for every fresh leaf after the
one-ply tier.  The H8 loss dual needs a rollout gate: begin with root-only or a
small universe/pair-count threshold, instrument calls/hits/p50/p99/max, and do
not enable it at every cap-500 leaf until compiled tails are measured.  There
were no h8 hits on 248 grinds, so grind relief must not be assumed.

## 9. Compact `NoWinWithin8` leaf

The future refutation grammar should use a canonical, checkable record:

```rust
struct NoWinWithin8LeafV1 {
    version: u8,                 // 1
    position_hash: u64,
    mover: Player,
    phase: TurnPhaseTag,         // must be FirstStone
    schedule: ScheduleTag,       // Fresh8
    coords_delta_zigzag: Vec<u8>,
    attacker_windows: Vec<CompactWindow>,
    defender_windows: Vec<CompactWindow>,
    universe_sha256: [u8; 32],
    normalized_pair_count: u32,
    refutations: Vec<PairRefutation>,
}

struct CompactWindow { key: WindowKey, empty_ids: SmallVec<[CellId; 4]> }
struct PairRefutation {
    attacker_pair: PairMask,
    reason: Refutation,
}
enum Refutation {
    DefenderCompletion { empty_ids: SmallVec<[CellId; 2]> },
    TwoCover { first: CellId, second: Option<CellId> },
}
```

Coordinates and windows are lexicographically sorted; pairs are in canonical
`i<j` order.  The verifier rebuilds `WindowStore` from the root, recomputes U
and its digest, checks exhaustive pair coverage, then checks each local
defender-completion or two-cover witness.  The leaf proves only bounded
`NoWinWithin8`; a parent grammar must never reinterpret it as full-game LOSS.

## 10. Required port tests

- Reproduce every R2 fresh-root H6/H8 ID set and every witness pair legality.
- Assert H2 win implies H6 win and H6/H8 attacker ID equality.
- Reproduce all 76 depth-at-most-eight certified wins with zero misses.
- Match Python on U size, window count, pair count, verdict, and first witness
  for a stable fixture set including max-U and max-node roots.
- Round-trip and independently verify `NoWinWithin8LeafV1`; mutate every field
  class and require rejection.
- Benchmark threat-free, median, p90, and recorded tail roots at cap 500 before
  enabling the loss dual below the root.
