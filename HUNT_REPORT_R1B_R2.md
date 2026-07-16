# Hunt report — zone-radius sharpness R1b and R2

Empirical fixture-or-shrink attack on the two OPEN numerical frontiers of
`docs/PROOF_TSS_DEFENDER_ZONES.md` (§12 items 5–6; §12a rows R1b, R2).
This is DATA and candidate fixtures, not proofs.

- Worktree: `.claude/worktrees/hunt-r1b-r2`, branch `hunt/r1b-r2`.
- Base commit: `dba6111d` (round-9b certificate-grade engine).
- Harness: `packages/hexfield_eq/rust/src/tss_hunt.rs` (test-gated, `#[ignore]`).
- Regenerate (from `packages/hexfield_eq/rust`, `CARGO_TARGET_DIR=../../../.target-hunt`):
  ```
  cargo test -p hexfield_eq --lib hunt_ -- --include-ignored --nocapture --test-threads=1
  ```
  (`--include-ignored` runs the two cross-checks in §1 plus the four `#[ignore]`
  datasets.) All randomized data uses fixed xorshift seeds embedded in the tests;
  the full run is deterministic and completes in ~50s (debug).

## 0. What the shipped code actually computes (established by reading)

The verifier's zone-coverage clause (`tss_verify.rs::verify_zone_node`,
~L1031–1071) and the solver's generator (`tss_solver.rs::zone_certificate_extras`,
~L4981) both implement the UNIFORM wrapper, not the fine per-role/per-window
forms:

- **Completion guard** (touched windows): a window with
  `active_player == defender && cnt_D(W) + d >= 6` contributes all its empty
  cells. `d` is `remaining_defender_placements(state, claimant, derived_t)` —
  the defender budget from the node to the certificate's GLOBAL horizon `T`.
- **Seed band (Z5)**: for each ghost-illegal, non-stone protected target `y`,
  every legal cell within radius **`8*d`** is searched.
- `d >= 6` short-circuits to the full legal set.

Two consequences that frame both problems:

1. **The seed radius is `8*d`, not the proof's `8(B-1)`.** The proof's uniform
   wrapper (D11 Z5′ with `r_N(y) := B(N)`) is `8(B-1)`. Because `d` counts to
   the global `T`, `d >= B(N)`, so the shipped radius is `8*d >= 8*B >=
   8(B-1)+8`: **at least one full relay (8 cells) above the proven-sufficient
   uniform radius, possibly more.**
2. **`Z_virgin` is ABSORBED.** An all-empty (virgin) window has
   `active_player == None`, so the completion guard never fires on it; and at
   `d < 6` no virgin obligation is created, while `d >= 6` takes the full legal
   set. The `8(E^D-6)` virgin radius form is **never computed by the shipped
   verifier**. R2 is therefore a question about the finer DEFINITIONAL zone
   (D16), not the shipped code path.

## 1. Harness faithfulness (cross-checks, both PASS)

- `hunt_legality_matches_engine`: the harness recomputes legality (empty ∧
  within-8-of-a-stone) directly from ownership maps; it equals the engine's
  `write_legal_moves` on 234 comparisons across 6 seeded legal games. **OK.**
- `hunt_seed_band_matches_production`: the harness's radius-parameterized seed
  band at multiplier `m == d` reproduces the production `zone_certificate_extras`
  output byte-for-byte on a real state with a hand-built arena, for `d = 1..5`.
  **OK.**

So harness zone numbers are the production numbers; only the radius multiplier
is varied.

## 2. R1b — uniform live-role seed band `8(B-1)`

### 2.1 Method

The seed band exists to enforce L9′ (first protected occupation): a dismissed
ghost-legal seed is dangerous iff, starting from it, the defender can build a
legal distance-8 relay chain that OCCUPIES a ghost-illegal protected cell `y`.
The harness measures `reach_seed_distance(y, B)` = the largest hex distance from
a currently-legal chain-start seed to `y`, over defender chains of length `<= B`
that occupy `y` (single BFS on the "within-8" graph of empty cells). The minimal
SOUND uniform seed radius for target `y` is exactly this reach; the proof bounds
it by `8(B-1)` (each relay `<= 8`, at most `B-1` relays).

### 2.2 Synthetic sharpness fixture (`hunt_r1b_chain_sharpness`) — VERDICT: SHARP at 8(B-1)

Family (docs `_TIGHTNESS_FRONTIER_REPORT.md` §2.1, realized as a board): attacker
anchor at `(0,0)`; protected target `y = (8B, 0)` (initially illegal); the only
legal cell within `8(B-1)` of `y` is the chain start `seed = (8,0)`. The defender
walks `seed -> (16,0) -> ... -> y` in exactly `B` stones.

```
  B | reach | min_pl | 8(B-1) | seed@8B | seed@8(B-1) | seed@8(B-2) | |req|@8B/8(B-1)/8(B-2)
  2 |    8  |   2    |    8   |  true   |    true     |    false    | 80/1/0
  3 |   16  |   3    |   16   |  true   |    true     |    false    | 80/1/0
  4 |   24  |   4    |   24   |  true   |    true     |    false    | 80/1/0
  5 |   32  |   5    |   32   |  true   |    true     |    false    | 80/1/0
```

Reading: for every budget `B`, the reach ATTAINS `8(B-1)` exactly; `y` needs
exactly `B` defender placements. The seed band keeps the binding seed at the
implementation radius `8B` (with a whole relay of slack — `|req|=80`) and at the
proof wrapper `8(B-1)` (where `|req|=1`, the seed alone), but SHEDS it at
`8(B-2)` (`|req|=0`). Shedding the seed leaves the defender's first protected
occupation of `y` unguarded — the L9′ violation. **A uniform seed radius below
`8(B-1)` is therefore unsound: candidate coverage-level PIN of R1b.**

Residual gap (honest): this is a coverage/reachability pin realized on a board,
not a full D9 WIN-certificate declaring a false win. It exhibits the exact game
mechanism L9′ prevents; embedding it in a complete solved certificate (the §2.2
synchronization concern — whether `B = r` survives all terminal/Z4/LOSS labels)
is the remaining step and needs a solver run (see §4).

### 2.3 Reach envelope over diverse positions (`hunt_r1b_reach_envelope`)

364 positions: 360 uniform-random legal games (seeds `1..=90 × plies {12,24,40,60}`,
fixed xorshift) + the 4 chain fixtures. For each (position, budget) the minimal
sound uniform seed radius = `max_y reach(y,B)` over ghost-illegal frontier
targets `y` (up to 80 per position, deterministic stride).

```
  B | positions_with_target | max_reach | at_8(B-1) | frac_ge_half | ever_gt_8(B-1) | ever_ge_8B
  1 |            0          |      0    |      0    |        0     |     false      |    false
  2 |          364          |      8    |    364    |      364     |     false      |    false
```

Reading:
- **B=1:** no ghost-illegal target is occupiable in 1 placement, so the seed band
  is never needed — matches the proof's empty `r=1` band (D15) and exposes the
  shipped `8*d` at `d=1` as pure over-search (proof wrapper `8(1-1)=0`).
- **B=2** (the standard Hexo defender turn): reach `= 8 = 8(B-1)` in **all 364**
  positions — the proof wrapper minimum is generically tight, not just in
  adversarial geometry. `reach` never reaches `8B = 16` (structural: at most
  `B-1` relays × 8), so the shipped radius always carries exactly one removable
  relay. `ever_gt_8(B-1)` and `ever_ge_8B` are both false everywhere (asserted).

### 2.4 R1b headline

- **The wrapper radius `8(B-1)` is SHARP** (§2.2): a smaller uniform radius sheds
  a seed whose dismissal enables a real-only occupation of a protected obligation
  (the L9′ violation), realized as a board fixture for every `B`. Moreover the
  envelope (§2.3) shows `8(B-1)` is generically tight — at `B=2` it is attained
  in EVERY one of 364 diverse positions, not only in adversarial chains.
- **The shipped verifier uses `8*d >= 8*B`, at least one full removable relay
  (8 cells) above the sharp minimum `8(B-1)`** (§0), and `reach` never exceeds
  `8(B-1) < 8B` in any position (structural + empirical). This is an
  unconditional IMPROVEMENT candidate for the shipped code, independent of R1b's
  theoretical status. Verdict: **SHRINK-EVIDENCE for the implementation
  (`8*d -> 8(B-1)`), plus a SHARP coverage-level fixture pinning `8(B-1)`.**
  The one gap to an absolute pin (a complete false-WIN certificate) is the
  deferred solver step in §4.

## 3. R2 — virgin-window completion radius `8(E^D-6)`

### 3.1 Absorption (`hunt_r2_virgin_absorption`) — structural, PASS

Confirmed on the §3.1 family: no window overlapping the all-empty `W` is
defender-active, so the completion guard (`active_player == defender && cnt_D +
d >= 6`) yields ZERO protected cells from virgin windows at every `d = 1..5`.
The `8(E^D-6)` virgin radius form is never computed by the shipped verifier —
it relies on the touched-window guard plus the `d >= 6 => full legal set`
fallback. **R2's numeric frontier is not on the shipped code path.**

### 3.2 Completion reach (`hunt_r2_completion_reach`)

The L12 analog: the defender COMPLETES an all-empty window `W` by relaying to it
then filling its 6 cells; filling `W` in `E^D` placements costs `E^D-6` relays +
6 fills, so a seed at distance up to `8(E^D-6)` must be searched. Family:
docs §3.1, direction `v=(8,-4)`.

```
  E | k=E-6 | reach | 8(E-6) | min_complete | seed==p0
  6 |   0   |    0  |    0   |     6        |  true
  7 |   1   |    8  |    8   |     7        |  true
  8 |   2   |   16  |   16   |     8        |  true
  9 |   3   |   24  |   24   |     9        |  true
 10 |   4   |   32  |   32   |    10        |  true
 11 |   5   |   40  |   40   |    11        |  true
```

For every `E >= 6` the completion reach ATTAINS `8(E-6)`, with the binding seed
`p_0` at exactly that distance and `W` completable in exactly `E` placements
(and NOT in `E-1`, asserted). This reproduces the fixed-window sharpness the
docs already record as "attained", now as a game-reachability trace with full
coordinates.

### 3.3 Full-union probe (`hunt_r2_full_union`) — VERDICT: INCONCLUSIVE (static)

```
  E | incident_all_empty_windows@p0 | axis Q/R/QR
  7 |              18               |   6/6/6
  8 |              18               |   6/6/6
  9 |              18               |   6/6/6
 10 |              18               |   6/6/6
 11 |              18               |   6/6/6
```

The binding seed `p_0` lies in 18 incident all-empty windows (6 per axis). Each
is a POTENTIAL self-cover of `p_0` at distance 0 — but only if the certificate
gives that window exposure `E^D >= 6`. Exposure is a D16 recurrence quantity
(defender placements before the attacker enters that window), fixed by the proof
tree, NOT a static board property. So static geometry cannot decide whether the
union self-covers `p_0`, and the full-union sharpness cannot be settled at this
level. **BLOCKER: per-window exposure labels require a certificate.** The
construction's isolated-seed geometry (all 18 incident windows all-empty) is
exactly the configuration §3.1 needs — but whether every incident window can be
forced to exposure `< 6` (or non-D-alive) simultaneously is the open question,
untouched by static probing.

### 3.4 R2 headline

- The fixed-window virgin radius `8(E^D-6)` is ATTAINED (reachability trace,
  §3.2) — confirms the docs' "fixed-window arithmetic attained".
- The full-union sharpness remains **OPEN / INCONCLUSIVE from static data**: it
  hinges on per-window exposure labels that only a certificate supplies (§3.3).
- **Structural finding:** the shipped verifier ABSORBS `Z_virgin` (§3.1), so R2's
  radius is moot for the shipped code path — it is a property of the finer D16
  definitional zone, not of `zone_certificate_extras`.

## 4. Deferred / what would strengthen this

- **Solver verdict-level confirmation of the R1b fixture.** Embed the §2.2 chain
  in a complete attacker-win position, solve at small caps (TT ≤ 512 MiB),
  confirm the produced certificate verifies at radius `8(B-1)` and that a
  radius-`8(B-2)` variant admits a real defender refutation (false WIN). This
  upgrades the R1b pin from coverage-level to absolute. Requires a release build
  of the solver (heavy; deferred here for host-capacity discipline).
- **R2 at the definitional level** needs per-window exposure labels `E^D(W)`
  (the D16 recurrence over a real certificate); static geometry cannot assign
  them, which is precisely the blocker for the full-union question.

## 5. Files

- Harness: `packages/hexfield_eq/rust/src/tss_hunt.rs`.
- Visibility-only edit (to reuse the production seed-band generator for the
  faithfulness cross-check): `tss_solver.rs` — `zone_certificate_extras` made
  `pub(crate)`. Module wiring: `lib.rs` — `#[cfg(test)] mod tss_hunt;`.
  No production logic changed.

## Absolute-pin run (follow-up)

**Outcome: `BLOCKED`.** The single ignored test
`hunt_r1b_absolute_pin` executes three deterministic, legal-replay position
families under finder/verifier relay deltas 0, 1, and 2. It did not mint a
weakened false-WIN certificate, so this follow-up does **not** upgrade R1b from
relative to absolute. The test retains the exact negative controls and prints
one machine-readable line per attempt plus the final `outcome=BLOCKED` line.

The exact blocker for the literal far-resource family is the finder width gate
`tss_solver.rs::threat_creating_moves` (currently line 3914), called by
`prove_choice` (currently line 3350): the shipped narrow finder admits only an
empty of an already claimant-active count-3-or-stronger length-six window. On
the real B=4 linear frontier, the claimant has only the anchor, so no downstream
`Choice` can be generated, `arena_core` never acquires the ghost-illegal target,
and every delta returns `UNKNOWN` without a certificate. Adding the count-three
scaffold at the eventual witness collapses the construction into attempt 2:
every witness empty is within five cells of a claimant stone, hence legal, so
`verify_zone_node`'s `pending` filter (currently lines 1055-1059) is empty and
Z5 never fires. Turning that scaffold into a forcing chain collapses it into
attempt 1: `prove_universal`'s `implicit_dispatch` predicate (currently lines
3459-3462) is true at the defender nodes, and the zone attachment gate
(currently lines 3475-3477) attaches no zone. These are the three position
families permitted by the construction budget; coordinate tweaks inside the
families reproduced the same trichotomy (leaf/no pending, forced/no zone, or
quiet/no generated attack).

Attempt details (all coordinates are axial `(q,r)`):

1. **Deep forcing / forced-dispatch control.** Replay
   `[(0,0),(-1,0),(0,-1),(-2,-3),(-1,-3),(-2,1),(-3,1),
   (0,-3),(1,-3),(-4,2),(2,-4),(1,4),(2,4),(-5,2),(2,-5),
   (3,4),(4,1),(-6,3),(3,-6),(4,2),(4,3),(-7,3),(3,-7),
   (1,7),(2,6),(-1,2),(2,-1),(3,5),(2,-3)]`. The root is
   Player1/FirstStone at ply 29; solve `Loss` to absolute T=37. All three
   deltas produce the same accepted 10-node certificate with Universals but
   `zones=0`; every Universal is `implicit_dispatch=true`.
2. **One-turn witness / empty-pending control.** Replay
   `[(0,0),(0,8),(2,7),(1,0),(2,0),(4,6),(6,5),(3,0),(4,0),
   (8,4),(10,3)]`; solve `Win` to T=13. All three deltas produce the same typed
   Win leaf for witness `WindowKey { start: (-1,0), axis: Q }`; its independently
   recomputed ghost-illegal pending set is empty.
3. **Literal sharp B=4 chain.** Root replay `[(0,0)]`, Player1/FirstStone,
   `s=(8,0)`, `y=(32,0)`, defender relay
   `[(8,0),(16,0),(24,0),(32,0)]`, with legal intervening attacker fillers
   `[(0,-1),(0,-2)]`; solve `Loss` to T=9. The engine accepts the full real line
   and ends with Player1 owning `y`. At the root, `s` is the only legal cell
   within the sharp radius 24 of `y`; shipped/sharp/weakened radii are
   32/24/16. Nevertheless all three finder runs are `UNKNOWN` because the
   count-three attacker-width precondition blocks creation of the far core.

No RNG is used. Every run uses `node_cap=100_000`, `tt_bytes_cap=64 MiB`
(well below the 512 MiB cap), zone search enabled with other zone options at
default, and the fixed exact horizons above. Repro from
`packages/hexfield_eq/rust`:

```
CARGO_TARGET_DIR=../../../.target-hunt cargo test -p hexfield_eq --lib hunt_r1b_absolute_pin -- --include-ignored --nocapture --test-threads=1
```

Production-default confirmation (delta defaults to zero) remains green:

```
CARGO_TARGET_DIR=../../../.target-hunt cargo test -p hexfield_eq --lib hunt_ -- --nocapture --test-threads=1
```

Both unchanged cross-checks pass, including
`hunt_seed_band_matches_production`; a non-test `cargo check -p hexfield_eq`
also passes with the test-only override absent.
