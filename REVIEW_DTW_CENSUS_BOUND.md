# Hostile review: DTW census bound

Review date: 2026-07-16. Review target: worktree commit `ffdd414a`.

## Overall verdict

**ACCEPT-WITH-EDITS.** I found no counterexample to either phase-specific
census theorem or to the correctly implemented `h=8` gate. All 16 `PROVEN`
ledger rows are **CONFIRMED**; none is broken.

This is not unconditional production sign-off. Contract 8.1 needs an exact
`WindowStore` recipe because the most tempting existing index produces a
silent false gate on explicit forced wins. Contract 8.2 also overstates the
coordinate-safe endpoint carried by its artifact. Separately, the empirical
record's “exact DTW” and “386 checked wins” interpretations are invalid for a
restricted positive-only solver. Those defects do not enter the deductive
proof of T1/T2, but they must be repaired before this document is used as the
production implementation specification.

Final tally: **16 CONFIRMED, 0 BROKEN-with-counterexample**. Machine premises
A and B are also confirmed against fresh exhaustive runs.

## Per-claim verdicts

Every row labeled `PROVEN` in `PROOF_DTW_CENSUS_BOUND.md` is tracked here.

| ID | Proof claim | Review verdict | Evidence / counterexample |
|---|---|---|---|
| R1 | Exact Hexo rules and quantifiers used by the theorem | **CONFIRMED** | `is_legal_placement`, `LEGAL_RADIUS=8`, immediate terminal detection, persistent stones, and the phase machine match. The negated forced-WIN quantifier is the one the defensive policy proves. |
| L1 | Persistence, root-window census, and the single-window floor | **CONFIRMED** | Any completed window is root-live. A completion on attacker placement `6-c` forces root count exactly `c` and every earlier future attacker placement into that one window. Untouched windows contribute exactly zero. |
| L2 | Turn maps from attacker-placement number to ply | **CONFIRMED** | Engine replay and `ply_of_mth` give FS `[1,2,5,6,9,10]` and SS `[1,4,5,8,9,12]`. `placements_made` increments before the terminal outcome is recorded. |
| L3 | Two distinct cells determine at most one winning line | **CONFIRMED** | They determine at most one infinite axis line; a non-collinear pair or separation greater than five has an empty target family. The ledger should say “axis line,” not the ambiguous “winning line.” |
| L4 | Defender-insertion/subfamily monotonicity over the ternary universe | **CONFIRMED** | For fixed attackers, adding defenders only deletes intervals; on survivors the empty set remains `I \ S`. Actual inter-turn configurations are a subset of the checked ternary universe. |
| L5 | Every requested service placement is legal when made | **CONFIRMED** | Every service cell is currently empty and within five of a just-placed attacker stone, hence in the radius-8 store. The second is distinct; extremal radius-1 fillers exist. |
| F3 | FirstStone case `c = 3` | **CONFIRMED** | Claim A kills every post-`a1,a2` count-5 window; `a3@5` can reach at most five. The explicit non-collinear split-pair attack creates no target count-5 family. |
| F2 | FirstStone case `c = 2` | **CONFIRMED** | Claim B hits every post-`a1,a2` count-4 window. A floor-equality `a4` win would have belonged to that family. |
| F01 | FirstStone cases `c <= 1`, including an exhaustive line-hit split | **CONFIRMED** | Before `a3,a4`, the live cap is three even after inter-turn defenders. Claim A applies to the actual ternary line; the in-window/on-line/off-line split is exhaustive. `c=0` is the six-placement floor. |
| T1 | FirstStone census two-gap theorem | **CONFIRMED** | The six census cases compose correctly from L1--L5 and A/B. |
| S3 | SecondStone `c = 3` analysis | **CONFIRMED** | The 18-placement witness is reachable, has root `c=3`, and forces a win on ply 5 through three disjoint two-cell gaps. A fresh solver certificate at relative 5 independently verified. |
| S2 | SecondStone `c = 2` analysis | **CONFIRMED** | After singleton `a1`, the next pair is governed by cap-3/count-5 Claim A; service on plies 6--7 precedes `a4@8`. |
| S1 | SecondStone `c = 1` analysis | **CONFIRMED** | After singleton `a1`, the next pair is governed by cap-2/count-4 Claim B; service precedes the `a4,a5` pair. |
| S0 | SecondStone `c = 0` analysis | **CONFIRMED** | Six attacker placements first occur at ply 12, so the floor excludes through 11. |
| T2 | Sound phase-specific SecondStone theorem | **CONFIRMED** | The `c<=2` increment follows from S2/S1; S3 proves that extending it to `c=3` is false. |
| C | Production leaf-gate consumer contract | **CONFIRMED** | The mathematical WIN-only gate, phase split, absolute-to-relative clock, and strict comparison are sound. Production sign-off is withheld pending the exact-census and artifact-scope repairs below. |

## Machine-premise audit

| ID | Machine premise | Review status | Predicate-to-prose audit |
|---|---|---|---|
| A | Cap-3/count-5 empty-union bound | **CONFIRMED** | Ternary cells; pair cells pre-empty; every defender-free local interval capped at 3; target intervals contain both pair cells and have exact post-count 5; union is over their current empty cells. Fresh run: 6,182,998 admissible, 214,439 nonempty families, worst union 2. |
| B | Cap-2/count-4 two-cell hitting-set bound | **CONFIRMED** | Same universe with cap 2/exact post-count 4. All singleton hits are tested before all distinct pairs, so the reported minimum is genuine. Fresh run: 4,953,489 admissible, 261,705 nonempty families, worst minimum 2. |
| V | Recorded machine checks and frozen results | **CONFIRMED, NARROW SCOPE** | The asserted tallies and schedule/witness regression reproduce. The injection assertion is intentionally one spot check; universality is deductive L4. The 245-row portion is schedule arithmetic, not a game oracle. Its source rows are not proven “exact DTW”; see High finding 2. |

## Findings by severity

### Critical

1. **Production census API trap.** Contract 8.1's mathematical
   scan must be implemented over `WindowStore::entries()`, retaining entries
   with attacker count greater than zero and defender count zero. Neither
   `threat_entries()` nor `live_threat_entries()` is equivalent: both discard
   live counts 1--3. The reachable FirstStone position in Section B has true `c=3`,
   zero threat entries, and a verified solver WIN at relative ply 6. A
   threat-only scan returns `c=0`, computes `LB_plies=10`, and silently skips
   the production `h=8` solve. The shorter SecondStone S3 witness is worse:
   threat-only also returns zero and gates `h=8`, despite its forced ply-5 win.

### High

1. **Contract 8.2 overclaims the artifact endpoint.** Section 5 correctly says
   the full theorem endpoint requires
   `coordinate_safe(state,LB_plies-1)`. Contract 8.2 checks only
   `coordinate_safe(state,h)` but tells the artifact to record no forced win
   through `LB_plies-1`. Safety through requested `h` is enough to skip the
   `h`-bounded solve, but not to mint that stronger engine artifact when
   `h < LB_plies-1`. At production `h=8`, the mismatch occurs for FS `c<=1`
   (claimed endpoint 9) and SS `c<=1` (claimed endpoint 11).

2. **The empirical “exact DTW” oracle is not exact.** `dtw_oracle` sets
   `exact=true` merely when the prior `SolveGoal::Win` result used fewer than
   the node cap. It forces `WidthOptions::vcf_pair_complete()`, while
   `tss_solver.rs` explicitly states that exhausting this restricted attacker
   set means only “no proof found,” not a disproof. Therefore:

   - a positive WIN at `h*` is a sound upper bound and can refute `LB > h*`;
   - prior-rung `Unknown` does not establish `dtw == h*`;
   - `None` does not establish `dtw > cap_h` or make a row “auto-safe”; and
   - `{2,6,10}` is a distribution of first restricted-proof horizons, not
     certified exact game DTW.

   Thus “0 violations on 386 checked wins,” exact slack, and all 245
   `exact=true` labels overstate the empirical evidence. The 162 positive
   triple-heuristic refutations remain valid. T1/T2 do not depend on any of
   these rows. The hunt should also independently verify every positive
   certificate before calling the solver production ground truth.

### Medium

1. **The original V suite did not bind the harness census to the production
   store.** `alive_windows` is correct, but no cited test compared it with
   `WindowStore::entries()` or froze the threat-index counterexample. The new
   ignored review test does both; this regression should accompany the
   eventual production implementation as a normal unit/property test.

2. **Insertion-route servicing should name `H \ D'`.** A hitting set chosen
   before restoring injected defenders may contain an already occupied
   defender cell. Such a cell has already killed every interval containing it;
   `H \ D'` still hits every surviving defender-free interval. L4 is sound,
   but spelling out this operational set removes an implementation ambiguity.

3. **The report's WindowStore cost claim is unsupported.** The current store
   has no max-census index. A correct implementation scans `entries()` or
   performs the deduplicated 18-key lookup set for every attacker stone; it is
   not presently “a few lookups.” This affects performance expectations, not
   soundness.

### Low / editorial

1. Ledger L3 says two cells determine at most one “winning line.” Adjacent
   cells such as `(0,0),(1,0)` belong to five different six-cell windows; what
   is unique is their infinite **axis line**. The lemma body is correct; rename
   the ledger row.

2. The proof header records repository commit `bd3c842...`, while this review
   target is `ffdd414a...`. A production theorem/ruleset version should bind
   the reviewed source and the new census regression, not the stale header.

## Mandatory attack-surface log

### A. Ply accounting end to end — CONFIRMED

`HexoState::apply_with_delta` increments `placements_made` once for each
single placement before recording a terminal outcome. The normal phase
transitions give exactly:

```text
FirstStone:  attacker placements a1..a6 at relative plies 1,2,5,6,9,10
SecondStone: attacker placements a1..a6 at relative plies 1,4,5,8,9,12
```

Every relevant production solver path records the winning placement as an
absolute `completion_ply` and admits it with the inclusive condition
`completion_ply <= semantic_horizon`. Thus, with
`base=state.placements_made()`, a relative placement `p` has absolute clock
`base+p`, and `h=semantic_horizon-base` is exact. A lower bound at placement
`LB_plies` excludes precisely through `LB_plies-1`; the gate is
`LB_plies > h`, not `>=`.

The following one legal, nonterminal replay supplies a concrete root for every
distinct phase-table boundary value (and every FS census; SS `c=0` shares the
same boundary as the displayed SS `c=1` row):

```text
[(0,0),(0,2),(1,2),(1,0),(2,0),(2,2),
 (3,2),(3,0),(4,0),(4,2),(0,4)]
```

The new test replayed every prefix and compared the harness census directly
with `WindowStore::entries()`:

| base | phase | current player | `c` | `LB_plies` |
|---:|---|---|---:|---:|
| 1 | FirstStone | P1 | 0 | 10 |
| 2 | SecondStone | P1 | 1 | 12 |
| 3 | FirstStone | P0 | 1 | 10 |
| 4 | SecondStone | P0 | 2 | 9 |
| 5 | FirstStone | P1 | 2 | 9 |
| 6 | SecondStone | P1 | 3 | 5 |
| 7 | FirstStone | P0 | 3 | 6 |
| 8 | SecondStone | P0 | 4 | 4 |
| 9 | FirstStone | P1 | 4 | 2 |
| 10 | SecondStone | P1 | 5 | 1 |
| 11 | FirstStone | P0 | 5 | 1 |

Positive ladder checks pin the dangerous equality side. Every WIN certificate
in this table was passed through `TssVerifier`:

| position | `base` | `LB` | horizon `LB-1` | horizon `LB` |
|---|---:|---:|---|---|
| 37-ply FS `c=3` prefix from Section B | 37 | 6 | `H=42`: Unknown, 720 nodes | `H=43`: **Win**, 2 nodes |
| 18-ply SS `c=3` S3 witness | 18 | 5 | `H=22`: Unknown, 28 nodes | `H=23`: **Win**, 2 nodes |
| 9-ply FS `c=4` prefix above | 9 | 2 | `H=10`: Unknown, 1 node | `H=11`: **Win**, 1 node |
| 10-ply SS `c=5` prefix above | 10 | 1 | `H=10`: Unknown, 1 node | `H=11`: **Win**, 1 node |
| 11-ply FS `c=5` prefix above | 11 | 1 | `H=11`: Unknown, 1 node | `H=12`: **Win**, 1 node |

The pre-boundary `Unknown` values are deliberately **not** used as no-win
evidence; the deductive theorem supplies that half. The positive boundary
certificates prove that allowing equality in the gate is unsound. At actual
`h=8`, arithmetic in both phases yields exactly `c<=2`; the two verified
`c=3` wins at plies 6 and 5 show the cutoff cannot be relaxed.

### B. Census definition and production store — THEOREM CONFIRMED, SPEC EDIT REQUIRED

The proof census and harness agree. For every attacker stone, `alive_windows`
enumerates 3 axes × 6 offsets, deduplicates `(axis,start)`, rejects a window
containing a defender, and counts attacker stones. Every omitted window has
attacker count zero, so returning zero when the touched live family is empty
is equivalent to the infinite-window definition.

The correct current production-store implementation is:

```rust
let mut c = 0u8;
for entry in state.board().windows().entries() {
    let ac = entry.count(attacker);
    let dc = entry.count(attacker.other());
    if ac > 0 && dc == 0 {
        c = c.max(ac);
    }
}
```

After only opening `(0,0)`, P1 is at FirstStone with no attacker-touched live
window; both implementations return `c=0`. Virgin defender-free windows exist
and also have count zero, so the `m(c)=6` no-window case is sound.

The explicit counterexample to a threat-only production scan is the legal
37-placement prefix `4c716bfed1924aaf@37`:

```text
[(0,0),(2,-2),(2,0),(-2,2),(-2,0),(0,-2),(0,2),(-1,1),
 (-3,3),(-4,4),(4,-2),(-2,1),(-2,-1),(-2,4),(-2,-2),
 (-3,1),(-4,1),(-5,1),(1,1),(-1,3),(5,-3),(1,-2),
 (1,-1),(-1,-2),(3,-2),(1,-4),(-5,2),(-1,-1),(-1,0),
 (-1,2),(-1,-3),(-4,0),(-3,0),(1,0),(-5,0),(1,-3),(1,2)]
```

It is P1 FirstStone, has four live count-3 windows, zero live count-4-or-more
windows, and a verified WIN certificate at relative ply 6. The correct census
is 3 and does not gate `h=8`; `threat_entries()` returns no rows, falsely gives
`c=0`, and gates. Contract 8.1 must explicitly prohibit `threat_entries()`,
`live_threat_entries()`, `has_threats()`, and use of only the last
`WindowUpdate`.

### C. Pair concentration and the one-line reduction — CONFIRMED

The attempted non-collinear split is reachable. Replay:

```text
[(0,0),(1,1),(2,1),(5,5),(8,8),(3,1),(10,11),
 (6,7),(7,7),(10,12),(10,13),(6,8),(7,8)]
```

At the resulting P1 FirstStone root, `c=3`. The legal next pair
`x=(0,1), y=(10,10)` has difference `(10,9)`, so it is not collinear on a win
axis. `x` advances the `r=1` cluster and `y` advances the `q=10` cluster, but
no length-6 window contains both; therefore no post-pair count-5 target exists.
“The first starts one window and the second starts another” cannot produce a
floor-equality win.

The equality reduction was checked case by case:

| `c` | Forced concentration at the claimed floor |
|---:|---|
| 5 | `a1` alone completes the root-count-5 window; no pair lemma is used. |
| 4 | `a1,a2` both lie in the root-count-4 window; no increment is claimed. |
| 3 | FS: `a1,a2,a3` lie in one window, so after the first pair it is a Claim-A count-5 member. SS deliberately takes only the floor. |
| 2 | FS: after `a1,a2`, the equality window is a Claim-B count-4 member. SS: after singleton `a1`, pair `a2,a3` makes it a Claim-A count-5 member before `a4`. |
| 1 | FS: later pair `a3,a4` makes the equality window a Claim-A count-5 member. SS: pair `a2,a3` makes it a Claim-B count-4 member before `a4,a5`. |
| 0 | Any completion on `a6` spends all six future placements in one virgin root-live window; no seventh-placement increment is claimed. |

If a target `k+2` window exists, both pair cells lie in it. Two distinct cells
then select one infinite axis line and have separation 1--5. Every such target
interval lies inside the enumerated finite span. There is no missed 2D family.

### D. Defender injection and ternary coverage — CONFIRMED

The exhaustive A/B loop ranges over every ternary assignment in the relevant
line span and constrains only defender-free intervals. Hence every reachable
configuration with defender stones between attacker pairs is in the direct
universe. The deductive fixed-attacker monotonicity lemma, not the one recorded
spot check, proves universality.

Concrete stress cases, with the later pair at `{0,1}`:

- **Claim A is tight:** pre-pair `A={2,3,4}`, `D=∅`. Post-pair count-5
  windows starting at `-1` and `0` have completion union `{-1,5}`. Both
  services can be necessary.
- **Claim B is tight:** pre-pair `A={2,3}`, `D=∅`. The gap pairs are
  `{-2,-1}`, `{-1,4}`, `{4,5}`. No singleton hits all; `{-1,4}` does.
- **Inter-turn injection / `c<=1`:** root attacker `{4}`, earlier attacker
  pair `{2,3}`, injected defenders `{-1,6}`, later pair `{0,1}`. Omitting the
  injected pair gives starts `{-1,0}` and union `{-1,5}`; restoring it leaves
  start `{0}` and union `{5}`.
- **Ternary necessity:** pre-pair `A={-5,-4,-3,-2}`, `D={-1}` contains a
  dead four-attacker window while the live cap can remain 3. Erasing the
  defender revives the high-count window and invalidates the cap.

Defenders already lying inside the alleged winning window are the proof's dead
branch. Defenders elsewhere on its line merely delete family members. If an
insertion-route hitting set contains a restored defender, that cell has already
performed its service; operationally place only `H \ D'`.

### E. SecondStone `c=3` and the sound T2 threshold — CONFIRMED

The proof's 18-placement replay reaches P1 SecondStone with census 3. After
`a1=(1,1)`, the three live count-4 windows have pairwise-disjoint gaps:

```text
{(5,1),(6,1)}
{(1,5),(1,6)}
{(5,-3),(6,-4)}
```

Different-axis windows meet only at attacker-occupied `(1,1)`, so one legal
defender stone hits at most one gap pair. Two defender placements cannot cover
three. The defender's root live count is at most 3, so those two placements
cannot win first. P1 chooses an untouched pair and wins on `a3@5`. The new
ladder check found and independently verified that WIN at
`semantic_horizon=base+5`, while no certificate was found at `base+4`.

For SS `c=2`, singleton `a1` leaves cap at most 3 before pair `a2,a3`; Claim A
is serviced on plies 6--7, one attacker placement before `a4@8`. For SS `c=1`,
the corresponding cap is 2 and Claim B is serviced before `a4,a5`. These are
the genuine phase-correct increments. Extending the threshold to `c=3` is
explicitly false.

### F. Touched-window sufficiency and coordinate safety — CONFIRMED WITH CONTRACT REPAIR

Scanning only attacker-touched windows is sufficient for the reason in B, but
only if all touched counts 1--5 are retained. The exact store recipe must be
part of Contract 8.1.

The `i16` predicate is conservative and sufficient if its implementation uses
wide arithmetic throughout. Let
`M=16383-8(h+1)`. A legal future placement at relative `t<=h` follows a chain
of at most `t` radius-8 links, so each cube component is at most
`M+8t<=16375`. Post-placement legal-store generation reaches at most 16383;
window construction needs only ±5, reaching at most 16380. Any two guarded
components differ by at most 32766, avoiding `i16::MIN`, subtraction, and
`abs` overflow.

The implementation must cast `q,r` before computing `s=-q-r`, compute
`8*(h+1)` wide, and run this guard before doing any unsafe per-stone window-key
arithmetic. This covers root census keys, service cells, and extremal fillers.
It does not repair the Contract 8.2 artifact-scope mismatch in High finding 1.

### G. Service and filler legality in degenerate openings — CONFIRMED

After the sole opening `(0,0)`, the nonterminal legal store already has exactly
216 cells: the radius-8 ball minus the occupied origin. It cannot be exhausted
by a finite validated state on the unbounded board.

The minimal legal replay

```text
(0,0), (1,0), (2,0), (-3,0), (6,0)
```

places the attacker pair `(1,0),(2,0)` and then defender services at the two
distance-5 endpoints. Both endpoints were inserted while empty; the first
service removes only itself, and the distinct second remains legal on
SecondStone. If a service set has fewer than two cells, choose a q-extremal
occupied cell and step outward one cell; switch direction if that cell is the
sole required service. Repeating gives a distinct legal filler. Coordinate
safety keeps the construction representable.

## Numbered repair list

1. **Make Contract 8.1 executable:** include the `WindowStore::entries()`
   pseudocode from B, require attacker count `>0` and defender count `==0`,
   fallback to zero, and explicitly forbid threat-only indexes and local
   `WindowUpdate` scans.
2. **Fix Contract 8.2's artifact scope:** either record only “no forced WIN
   through requested `h`,” or require
   `coordinate_safe(state,LB_plies-1)` before minting the stronger endpoint.
   Keep the cheaper `coordinate_safe(state,h)` route for an `h`-scoped gate.
3. **Specify wide safety arithmetic:** calculate `R`, `q`, `r`, `s`, absolute
   values, and horizon subtraction in checked `i32`/`i64` (or wider), before
   any unsafe coordinate reconstruction.
4. **Retract the empirical exactness claims:** rename `dtw_h` to a restricted
   proof horizon/upper bound, treat all `Unknown`/`None` rows as unresolved,
   recompute the claimed validation/slack text, and independently verify every
   positive solver certificate.
5. **Promote the new census parity regression:** keep the post-opening `c=0`,
   all-c phase-prefix sweep, both `c=3` threat-index counterexamples, and
   `LB-1/LB` certificate checks with the production implementation tests.
6. **Clarify insertion servicing:** say that restored defender cells already
   service their intervals and only `H \ D'` needs new placements.
7. **Tighten artifact/version language:** rename L3's “winning line” to “axis
   line” and bind the theorem/ruleset record to the reviewed commit and exact
   census implementation.

## Contract 8.1 / 8.2 production sign-off

**Contract 8.1 mathematical result: SIGNED OFF. Contract 8.1 as the production
implementation spec: OBJECT pending repairs 1 and 5.** A correct implementation
over `WindowStore::entries()` gives the proof census. A threat-index
implementation demonstrably skips forced wins inside `h=8`.

**Contract 8.2 WIN-gate arithmetic: SIGNED OFF. Contract 8.2 full-endpoint
artifact: OBJECT pending repairs 2 and 3.** The exact production predicate at
an actually computed relative horizon `h=8` is `c<=2` in either supported
phase, with `LB_plies > h`. It authorizes skipping only the current player's
bounded `SolveGoal::Win` attempt. It does not authorize skipping an unsplit
`SolveGoal::Both`, manufacturing a global loss, or using the SS `c=3`
FirstStone formula.

No sign-off is given for Contract 8.3 integration, which the proof correctly
leaves `OPEN`.

## Test and artifact log

- Review skeleton was created before claim adjudication.
- Required inputs were read in the mandated order at worktree commit
  `ffdd414ad5197444eef44af4f28da376a5d95507`.
- Added ignored test `dtw_hostile_ply_boundaries` in
  `packages/hexfield_eq/rust/src/dtw_bounds_hunt.rs`. It checks the all-c
  phase/LB sweep, harness-to-`WindowStore::entries()` census parity,
  post-opening `c=0`/216 legal cells, both threat-index counterexamples, five
  strict solver boundaries, and independent verification of every WIN
  certificate.
- Final boundary run: 13.082 GB free; 1 passed, 0 failed; test 0.06 s,
  total 6.5 s.
- `dtw_line_lemma`: 12.877 GB free; 1 passed, 0 failed; 1.18 s. Frozen A/B
  tallies reproduced exactly.
- `dtw_secondstone_regression`: 12.915 GB free; 1 passed, 0 failed; 0.00 s.
  Frozen row arithmetic and S3 replay/gaps reproduced.
- `rustfmt --edition 2021 --check` and `git diff --check` passed.
- No proof document was edited, no corpus/artifact output was rewritten, and no
  commit was made.

Files intentionally changed by this review:

- `REVIEW_DTW_CENSUS_BOUND.md`
- `packages/hexfield_eq/rust/src/dtw_bounds_hunt.rs` (ignored adversarial test
  only)
