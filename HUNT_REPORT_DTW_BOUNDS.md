# Machine hunt report — CERTIFIED DISTANCE-TO-WIN LOWER BOUNDS (NQ1)

**Purpose.** Find a cheap, position-computable LB(P) with the eventual theorem
"the attacker cannot win within LB(P) placements against any defense," measured
in PLIES from the position (the unit of the production leaf ladder's +8/+12
deadlines). If LB(P) > deadline, the whole leaf solve is skippable for free and
the skip doubles as a proof-backed bounded-no-win artifact. This hunt produced
CANDIDATES with empirical soundness validation plus one exhaustively-verified
geometric core lemma; a later round proves the survivor.

**Ground-truth oracle (exact).** The production WIN solver strictly respects
`semantic_horizon`: any node beyond it is Refuted (tss_solver.rs:2435) and any
completion must satisfy `completion_ply <= semantic_horizon`
(tss_solver.rs:2644/2655). Laddering `semantic_horizon = base+h` for h=1,2,...
yields the minimal h* at which a forced win is provable — an upper bound on the
true distance-to-win dtw (dtw <= h*), and exactly dtw when the previous rung
exhausted below the node cap (it did, on **all 245** resolved samples). Any
candidate with LB(P) > h* on a solved win is REFUTED.

**Units and the structural cap (frames everything).** Every candidate here
maps "attacker's own placements needed, m" through the deterministic turn
structure (2 placements/turn) to a ply count `ply_of_mth(m)` — for a
FirstStone attacker: m=1..6 → ply 1, 2, 5, 6, 9, 10; SecondStone: 1, 4, 5, 8,
9, 12. A single window is filled by at most 6 own placements, so m <= 6 and
**no bound of this census family can exceed ply 10 (FirstStone) / 12
(SecondStone)**. Corollary: fire rate at h=12 is structurally ZERO for the
entire family — proved by arithmetic, confirmed by the sweep.

---

## HEADLINE

- **`lb_census_block1` — the sharpest validated candidate: 0 violations on 386
  solved wins (245 with exact dtw), EXACT (slack 0) on 162/245 of them, and
  its core geometric step is now EXHAUSTIVELY VERIFIED (11.1M line
  configurations). Fire rate at the h=8 deadline: 49.0% of 4,000 leaf-shaped
  nodes (90.97% in the opening band), at ~17 µs/eval (unoptimized).**
- **At the h=12 deadline the fire rate is 0.0% — structurally.** No
  single-window census bound can gate +12. A candidate designed to go deeper
  (`lb_triple_heur`) was **REFUTED** (162 violations, overshoot up to 8 plies):
  census counts alone, without servicing analysis, overestimate the distance.
- Verdict: **leaf gating pays at h=8, cannot pay at h=12 with this family.**

---

## Candidate catalog (exact definitions)

Let `me` = side to move at the leaf (the attacker being gated). Census: alive
windows = length-6 segments on the 3 axes with >= 1 `me` stone and no opponent
stone; `maxcnt` = max `me`-stone count over alive windows (0 if none).

### C1. `lb0_single_window` [PROVEN shape] — the floor

    m = max(1, 6 - maxcnt)   (no alive window: m = 6)
    LB0(P) = ply_of_mth(m)

Proof (single-window arithmetic + turn structure): any attacker win completes
a window with no defender stone in it; a window containing a defender stone at
P is dead forever, and a virgin window needs 6 own placements >= any alive
window's need. Completing a window with pre-count c requires filling its 6-c
empties, each with one of `me`'s own future placements (the defender only adds
stones — no defense accelerates the attacker). The m-th own placement lands at
`ply_of_mth(m)` by strict alternation. No win strictly before that ply,
against ANY defense — this direction needs no servicing argument at all.

### C2. `lb_census_block1` (`lb_block1_heur` in the harness) [VALIDATED + core verified]

    m = 6 - maxcnt           if maxcnt >= 4
    m = min(6-maxcnt+1, 6)   if maxcnt <= 3
    LB1(P) = ply_of_mth(m)

One extra own placement whenever the fastest window is slow enough
(maxcnt <= 3) that the defender gets a full turn before it can complete. See
the lemma section for the exact claim and the verified proof skeleton.

### C3. `lb_triple_heur` [REFUTED]

Greedy census estimate of own-placements to build 3 distinct-empty-set alive
windows to count-5, +1 to complete (fresh windows cost 5), m capped at 6.
Motivated by "the pre-win defender node needs a family two placements cannot
service." **REFUTED — see below.** The motivation ignores that a count-4 fork
(two 2-empty windows) also beats two defender placements, and that overlapping
clusters reach the pileup far faster than the distinct-window greedy estimate.

---

## Soundness results

Sample = 386 attacker-to-move solved-WIN positions:
14 forcing-corpus WINs + 122 leaf-width `wide_only_win` prefixes + 250 fresh
human-corpus FirstStone nodes screened by a 30k-node WIN solve at horizon
base+14 (deterministic seed; the screen deliberately harvests SHALLOW wins —
the tight tests). Oracle ladder: node cap 1.5M, TT cap 256 MiB, per-sample
rungs h = 1..min(max candidate LB, 14); a sample with no WIN by that rung has
dtw > every candidate LB and is safe by construction (counted as checked).
245/386 resolved to an exact dtw (all with the exhausted-previous-rung
certificate); the rest have dtw > 14 > every LB.

| candidate | checked | violations | min slack | verdict |
|---|--:|--:|--:|:--|
| `lb0_single_window` | 386 | **0** | 0 | **VALIDATED-ON-386-WINS** |
| `lb_census_block1`  | 386 | **0** | 0 | **VALIDATED-ON-386-WINS** |
| `lb_triple_heur`    | 386 | **162** | **-8** | **REFUTED** |

Refutation example (`DTW_SOUNDNESS.jsonl` row 1): leaf-width node
`4c716bfed1924aaf@37` — `lb_triple` claims no win before ply 10; the engine
proves a win in **6** plies (exact). The position's threats overlap: the
greedy distinct-empty-set estimate charges ~9 fills where the real cluster
needs 4 placements. 162/245 resolved wins refute it; worst overshoot 8 plies.

**Exact-dtw distribution (245 resolved).** dtw in {2, 6, 10} ONLY — every
solved win completes on the SECOND placement of an attacker turn (pair
completion; odd-ply wins, which require a count-5 surviving a full defender
turn, never occur in this corpus):

| dtw | n | lb0 value (n) | block1 value (n) |
|--:|--:|:--|:--|
| 2  | 78 | 2 (78) — exact | 2 (78) — exact |
| 6  | 84 | 5 (84) — slack 1 | **6 (84) — exact** |
| 10 | 83 | 5 (65), 6 (18) | 6 (65), **9 (18) — slack 1** |

`lb_census_block1` is EXACT on 162/245 and within 1 ply on 18 more; the
remaining 65 (maxcnt=3 leaves whose real distance is 10) show the intrinsic
limit of a one-number census: it cannot see how much servicing the defender
still has in hand.

---

## The core lemma, exhaustively verified

**One-line two-gap lemma (computational core of C2).** All windows that reach
count-5 (resp. count-4) from one attacker pair placed into a maxcnt<=3 (resp.
maxcnt<=2) position must contain BOTH placed stones (pre-count <= maxcnt, and
a pair adds 2), and two distinct cells are collinear on at most one axis — so
the whole family lies on ONE lattice line, and the claim is one-dimensional:

- **Claim A.** Line cells ternary (empty/attacker/defender), every
  defender-free window <= 3 attacker stones, pair placed at distance d<=5 on
  empty cells: the defender-free count-5 windows after the pair have **<= 2
  distinct empty cells**. Verified exhaustively: d = 1..5, all 3^(d+9)
  admissible configurations = **6,182,998 checked / 214,439 with non-empty
  families / worst distinct-empties = 2.**
- **Claim B.** Same with cap 2 and count-4 windows: **min hitting set of the
  empty-pairs <= 2.** Verified exhaustively: **4,953,489 / 261,705 / worst
  hitting set = 2.**

(The ternary universe matters: dead windows may legally carry >= 4 attacker
stones; a binary-stone check would miss those configurations. Both claims
survive the ternary universe.)

### Lemma statement (ready for the proof round)

> **Lemma (census two-gap distance bound).** Let P be a non-terminal Hexo
> position with the attacker to move (FirstStone), no six on the board, and
> let c be the maximum attacker-stone count over defender-free length-6
> windows (c = 0 if none). Set
> m(c) = 6-c for c >= 4, and m(c) = min(6-c+1, 6) for c <= 3.
> Then the attacker has no strategy that completes six-in-a-row within
> ply_of_mth(m(c)) - 1 plies against every defense, where ply_of_mth is the
> turn-structure map (FirstStone: 1,2,5,6,9,10).
>
> *Proof skeleton.* (i) Any completed window has pre-count <= c and receives
> only attacker placements (LB0 arithmetic — proven above). A win with
> exactly 6-c placements forces ALL of them into one window W with
> pre-count exactly c. (ii) c=3: placements 1,2 lie in W, hence collinear;
> every post-pair count-5 window contains both, hence lies on their unique
> common line; **Claim A** gives <= 2 distinct completion cells; the defender
> occupies them (legal: within distance 5 of stones), killing every count-5;
> placement 3 (ply 5) then completes nothing (all other windows are <=
> count-4 + 1 placement = 5 < 6). (iii) c=2: same collinearity puts every
> post-pair count-4 window on one line; **Claim B** gives a 2-cell hitting
> set; the defender occupies it; a ply-6 win would need a window containing
> all 4 placements — every such window is dead. (iv) c<=1: a ply-9 win puts
> placements 1..4 in W (collinear); at the attacker's second turn the line's
> defender-free windows hold <= 1+2 = 3 stones, so **Claim A applied at the
> second defender turn** (pair = placements 3,4; defender stones only shrink
> the family, and a sub-family has a subset of empties) again yields <= 2
> completion cells; the defender services them, excluding m=5; m=6 is the
> LB0 floor. (v) The defender placements above are unconditional (Hexo
> defenders place freely; the serviced cells are legal and empty). ∎-shape
>
> Open obligations for the proof round: (a) formalize the sub-family
> monotonicity in (iv) (defender stones between the turns); (b) the c<=1
> case's bookkeeping when the defender's turn-1 stones already hit the line;
> (c) lift the harness convention (attacker = side to move at a FirstStone
> node) to the SecondStone variant if the consumer needs it.

Empirical status: 0 violations on 386 solved wins; exact on 162/245.
The two computational claims are exhaustive facts, not samples.

---

## Fire rate x cost (the money table)

4,000 leaf-shaped nodes (FirstStone, non-terminal, mover = attacker),
deterministic sample from the 6,902-game human corpus, bands by placement
count: opening (<=12): 831, middle (13-40): 1,523, late (>40): 1,646.
Cost measured warm, 64 reps/node, BTreeSet-based reference implementation.

**maxcnt histogram** (what the census sees at leaves):
maxcnt 0: 124, 1: 375, 2: 1460, 3: 1854, 4: 183, 5: 4.

| candidate | µs/eval (med) | h=2 | h=4 | h=6 | **h=8** | **h=12** | h=16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `lb0_single_window` (proven) | 17.3 | 95.3% | 95.3% | 12.5% | **12.5%** | **0.0%** | 0.0% |
| `lb_census_block1` (validated) | 16.5 | 95.3% | 95.3% | 49.0% | **49.0%** | **0.0%** | 0.0% |
| `lb_triple_heur` (REFUTED) | 17.3 | — | — | — | — | — | — |

Per-band fire at h=8: `lb0` 56.2% / 2.0% / 0.1%; `block1` **91.0% / 51.6% /
25.3%** (opening / middle / late). The h2=h4 and h6=h8 equalities are
structural (no m maps to plies 3-4 or 7-8 at FirstStone).

Cost context (sibling leaf-width hunt, same corpus, 4 MiB TT profile,
`LEAF_WIDTH_TIMING_RAW_4MiB.txt`): production narrow leaf solve at cap 500 =
604-684 µs median, 12-25 ms p95; WIN-goal-only narrow = 78 µs median. The
17 µs reference screen is 4-40x cheaper than the median solve and ~10^3x
cheaper than p95 — and it is a deliberately naive BTreeSet re-scan; a
production implementation reading the engine's incrementally-maintained
`WindowStore` would be O(few lookups), plausibly ~1 µs.

---

## Verdict on leaf gating

- **h=8 ladder rung: PAYS.** The validated `lb_census_block1` skips **49%**
  of all leaf solves (91% in the opening band, where the +8 rung is most
  often invoked) for ~3% of the median solve cost — before any
  implementation tuning. Even restricting to the already-proven-shape `lb0`,
  12.5% of solves (56% in the opening band) are skippable today, each skip
  carrying a proof-backed "no win within 8 plies" artifact.
- **h=12 ladder rung: DOES NOT PAY — and cannot.** The single-window census
  family is arithmetically capped at ply 10 (FirstStone). Zero fire rate is
  not a tuning problem; gating +12 requires a bound that reasons about
  defender servicing across >= 2 windows, and the one candidate of that
  shape tested here (`lb_triple_heur`) is REFUTED by 162 real positions.
  Building a sound >12-ply bound needs the servicing/potential machinery
  (ES Theorem 2 / birth-ledger ceilings) as hypotheses, not just counts —
  a proof-round-first problem, not a hunt-first one.
- Bonus consumer note: the same census evaluated for the OPPONENT (whose
  m-th own placement from a mover-FirstStone node lands at plies 3,4,7,8,
  11,12) gates the LOSS half of `SolveGoal::Both` symmetrically; its m=6
  value is ply 12, so the +12 rung's loss half is reachable by census only
  at exactly h=12 (LB > 12 still impossible). Not measured here.

---

## Files

- `HUNT_REPORT_DTW_BOUNDS.md` — this report.
- `DTW_SOUNDNESS.jsonl` — 162 refutation rows for `lb_triple_heur`
  (candidate, source, tag, lb_plies, oracle_h, replay prefix). No rows for
  the two validated candidates.
- `DTW_RESOLVED.jsonl` — 245 resolved samples (source, tag, exact dtw_h,
  lb0, side to move).
- `DTW_FIRERATE.json` — machine-readable fire-rate table.
- `packages/hexfield_eq/rust/src/dtw_bounds_hunt.rs` — the harness
  (test-gated, registered in lib.rs under `#[cfg(test)]`; uncommitted).

## Reproduction

Worktree `hunt-dtw-bounds` (branch `hunt/dtw-bounds`, HEAD 5536b2bb + the
uncommitted harness). Deterministic: fixed xorshift seeds, no RNG on any
solved path, fixed node/TT caps; `--test-threads=1` throughout.

```
cd E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-dtw-bounds
# pilot (ply-map asserts + forcing-corpus oracle sanity, ~1 min):
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  dtw_bounds_hunt::dtw_pilot -- --ignored --nocapture --test-threads=1
# soundness (seed 0x51EDC0DE20260716 = 5903586746807224086, hmax 14,
# fresh_n 250, fresh_cap 30000, oracle cap 1.5M; ~10 min):
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  dtw_bounds_hunt::dtw_soundness -- --ignored --nocapture --test-threads=1
# fire-rate (seed 0x9E3779B97F4A7C15, n 4000; ~40 s):
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  dtw_bounds_hunt::dtw_firerate -- --ignored --nocapture --test-threads=1
# exhaustive one-line two-gap lemma (ternary, ~1 s):
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  dtw_bounds_hunt::dtw_line_lemma -- --ignored --nocapture --test-threads=1
```
