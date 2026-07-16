# Machine hunt report — GAP-RAW adaptive-witness search

**Question (continues `HUNT_REPORT_GAP_RAW.md`).** The prior hunt proved neither
fixed greedy is a GAP-RAW witness: dynamic touched-window greedy loses the ES
cohort-target line but foils fresh births; fixed-cohort / target-lock foils the
ES line but loses fresh births — so any witness must be **adaptive**. This hunt
asks: **is a SIMPLE adaptive rule that witness?**

**Potential (unchanged).** `λ = √3`; `Φ(P) = Σ` over attacker-touched alive
windows (`cnt_A ≥ 1, cnt_D = 0`) of `λ^{−#empties}`; all-empty windows excluded.
Role convention (proven in the prior report): engine `Player0` = **defender**,
`Player1` = **attacker**; a Defender-`FirstStone` position is `Player0` to move
in `TurnPhase::FirstStone`, turn cycle `D D A A …` (two placements per turn).

---

## VERDICT

**No simple fixed rule surveyed is a complete witness — but one rule,
`R1b` (completion-first touched-window greedy, threshold τ=2), comes decisively
close and its residual failures are exactly the known birth-divergence
obstruction, not a new refutation.**

- **Mechanism of the old dilemma, pinned.** Dynamic greedy loses the ES line
  because it is **completion-blind**: when the target window `W` reaches count-5
  (one empty), the attacker's own 4-in-a-row spawns a *cluster of count-4 birth
  windows* whose shared empties out-score `W`'s single completion cell in the
  danger ranking. The defender blocks the denser cluster and leaves the actual
  completion open. (Traced exactly — see "The failure mechanism" below.)

- **The fix is completion-awareness, and it must be GLOBAL (all windows, not a
  fixed family).** Patch dynamic greedy so that whenever some window is within
  one attacker turn (≤ τ empties) of completion, the defender **aims a block at
  the most-imminent cluster** instead of at the max aggregate-danger cell;
  otherwise it plays dynamic greedy. This single rule (`R1b`, τ=2):
  - **foils every one of the 12 adversarial scripts on every `Φ<1` root**
    (the ES line and its D6 images, fresh births, birth danger-magnets, a
    self-anchored "birth-ES", double births, interleaved lines, and three
    hand-built fork attempts);
  - has **no** best-play-attacker forced win at any exhaustively reachable
    horizon (plies 4 and 6, sound Maker-Breaker attacker vs the fixed policy);
  - holds **98.6 – 99.9 %** of thousands of randomized full-depth episodes
    (per-root break rates 41, 15, 6, 2 per 3000 — see the matrix), against a
    strong completion-seeking, cluster-building, fork-biased attacker.

- **τ is non-monotonic; τ=2 is the sweet spot.** τ=1 reacts too late (misses a
  one-move double / "cross fork"); τ=3 over-reacts (chases the numerous count-3
  windows of a danger magnet and spreads thin). On es_core deep (3000 eps):
  **τ=1 → 806 breaks, τ=2 → 41, τ=3 → 825.** "Block exactly when a window is one
  attacker turn from completion" is the right trigger.

- **The residual R1b leaks are the birth-divergence, not a refutation.** Every
  R1b break is a **long grind-out** (win at placement 48–60, ~25+ attacker
  stones). Traced: at the losing defender turn the board carries **≥ 3–4
  simultaneous count-4 (2-empty) windows**; two aimed blocks per turn cannot
  clear them, so one matures. This is precisely the unbounded-birth pileup that
  `ES_GLOBAL_BOUNDARY.md` already names ("the cumulative birth sum diverges; no
  strategy can prove the raw claim by maintaining `Φ<1`"). It is a loss for the
  *fixed* R1b policy, **not** a proof that the position is an attacker win — a
  different defender may hold it. GAP-RAW is **not** refuted.

**Bottom line for the proof lane:** the witness is `completion-first + dynamic
suppression`; the invariant it *targets* is a joint, all-windows property
(§"Candidate invariant"). Whether that invariant is *maintainable against every
attacker* reduces to bounding how many count-4 windows the attacker can force to
mature in one turn — i.e. to resolving the open birth-ledger gaps
(`GAP-GLOBAL-RENEWAL` / `GAP-AMORTIZED-ABANDONMENT`).

---

## The candidate rules (all memoryless, pure functions of the position)

Every rule is a pure function of the current board plus the **frozen root
cohort** `F` (the windows attacker-alive at the root — fixed context, not
per-node state). Purity is required so the rule is legal inside a tree search
*and* statable as a positional invariant. Shared vocabulary: a window is
*attacker-alive* if it holds ≥1 attacker and 0 defender stones; its *empties*
are its unfilled cells; *danger* of a cell = `Σ λ^{−#empties}` over the alive
windows the cell is an empty of; the *aimed block* on tier `m` is the empty
shared by the most alive windows having exactly `m` empties (ties: danger, then
lexicographic).

| id | name | one-paragraph rule |
|---|---|---|
| **R1 / R1b / R1c** | **Completion-First Greedy (τ = 1 / 2 / 3)** | Let `m` = min empties over all alive windows. If `m ≤ τ`, play the aimed block on tier `m`; else play the dynamic touched-window max-danger cell. |
| **R2** | **Cohort-Priority Greedy (boost 3, τ=1)** | Completion override at `m ≤ 1`; else dynamic greedy but with cohort-`F` windows weighted ×3 (bias toward initial targets without hard commitment). |
| **R3** | **Starved-Target-Lock (k=3)** | If some alive **cohort** window has ≤ k empties **and** is *starved* (the plain dynamic-greedy cell is not one of its empties), lock the most urgent such target on its top-danger empty; else dynamic greedy. (The task's lexicographic hybrid — protects only the frozen cohort's completions.) |
| **R4 / R4b** | **Guarded F-greedy (τ = 1 / 2)** | Proof-shaped (Thm 3 form): completion override at `m ≤ τ`; else **fixed-cohort** F-greedy. Differs from R1 only in the fallback (cohort-F-greedy, which ignores non-imminent births, vs dynamic greedy which suppresses them). |
| — | dynamic_greedy, cohort_greedy | the two prior baselines, for contrast. |

---

## The failure mechanism (why the old dilemma exists), pinned exactly

Trace of dynamic greedy vs the ES script on `es_core` (`A={(0,0)}, D={(1,0)}`,
`Φ=0.834`), target `W = {(-5,0)..(0,0)}` (`trace_dynamic_vs_es`):

```
ply 12: D plays (-6,0) danger=(A21,B4)=21+4√3 | W count=5, empty (-1,0)
ply 13: D plays (-3,1) danger=(A3,B9)=3+9√3  | W count=5, empty (-1,0)
ply 14: A plays (-1,0) -> ATTACKER SIX
```

At ply 12 `W` is count-5, so its empty `(-1,0)` carries danger `9+9√3 ≈ 24.6`.
But the attacker's contiguous four `(-5,-4,-3,-2)` has spawned two count-4
*birth* windows (`{(-6,0)..(-1,0)}`, `{(-7,0)..(-2,0)}`), and their shared empty
`(-6,0)` carries `21+4√3 ≈ 27.9 > 24.6`. Dynamic greedy blocks the **cluster**,
not the **completion**. The count-4 cluster is a *danger magnet*; the lone
count-5 completion is *starved*. This is the exact obstruction a witness must
overcome, and it says the fix is to prioritise **imminence** over **aggregate
danger** — globally, since the magnet windows are births, not cohort members.

---

## Stress matrix

Roots (all Defender-`FirstStone`, `Φ<1`): `es_core` (0.834), `blocker_1_-1`
(0.834), `blocker_2_0` (0.898), `blocker_3_0` (0.962), `near2_-3_-3__-3_0`
(0.962). Attacks: **12 fixed scripts** (full depth, deterministic), **randomized
attacker** (completion-seeking + threat-gain cluster builder + fork/birth bias,
four birth-biases × N seeds, 60-placement horizon), **bounded exhaustive**
best-play attacker vs the fixed policy (sound `AttackerWin`; plies 4, 6).

### Scripts — which rules an adversarial script BREAKS (attacker completes six)

| rule | scripts it loses to |
|---|---|
| dynamic_greedy | `es` (ply 15); `birth_cross_fork` (ply 24) |
| cohort_greedy | every birth: `fresh_birth`, `birth_magnet`, `birth_es_far`, `double_birth`, `fork_attempt`, `L_double_four` |
| R4 (τ=1) | `fresh_birth`, `birth_magnet`, `birth_es_far`, `double_birth` |
| R1 (τ=1) | `birth_cross_fork` (ply 24) |
| R2, R3 | `birth_cross_fork` (ply 24) |
| **R1b (τ=2)** | **none** — foils all 12 on all roots |
| **R1c (τ=3)** | **none** — foils all 12 (`birth_cross_fork` foiled ply 23) |
| **R4b (τ=2)** | **none** (but leaks heavily to randomized — see below) |

`birth_cross_fork` (the line the randomized attacker used to beat R1/R2/R3):
a Q-line four `(11..14,0)` crossed by an R-column `(12,*)`, forcing two count-5
completions on one turn. τ=1 blocks only count-5 (too late once both mature); τ=2
pre-empts it at count-4. This is the clean "τ=1 insufficient, τ=2 necessary"
witness.

### Randomized attacker — break count per **480** episodes (broad sweep)

| rule | es_core | blocker_1_-1 | blocker_2_0 | blocker_3_0 | near2 |
|---|--:|--:|--:|--:|--:|
| dynamic_greedy | 169 | 149 | 175 | 175 | 173 |
| cohort_greedy | 475 | 479 | 480 | 476 | 479 |
| R1 (τ=1) | 133 | 121 | 32 | 38 | 23 |
| **R1b (τ=2)** | **7** | **6** | **1** | **0** | **0** |
| R4 (τ=1) | 431 | 432 | 441 | 432 | 433 |
| R4b (τ=2) | 289 | 269 | 273 | 283 | 268 |
| R2 (boost3,τ1) | 132 | 162 | 165 | 36 | 168 |
| R3 (lock k3) | 146 | 169 | 176 | 175 | 173 |

### Randomized attacker — break count per **3000** episodes (deep, τ-ladder)

| rule | es_core | blocker_2_0 | blocker_3_0 | es_core_reflected |
|---|--:|--:|--:|--:|
| R1 (τ=1) | 806 | 176 | 228 | 758 |
| **R1b (τ=2)** | **41** | **6** | **2** | **15** |
| R1c (τ=3) | 825 | 248 | 252 | 627 |

### Bounded exhaustive best-play attacker (sound `AttackerWin`)

Every rule, every root, plies 4 and 6: **`Unknown`, completed, `break=false`**
(no attacker forced win). Plies 8 caps out (`completed=false`) — the same
~230-wide branching depth wall documented in the prior hunt; the exhaustive arm
certifies only short horizons and cannot decide the long-horizon regime.

---

## Per-rule breaking analysis

- **dynamic_greedy** — completion-blind (mechanism above). Loses the ES line and
  the cross-fork; the randomized attacker beats it ~35 % of episodes.
- **cohort_greedy** — never scores births; loses every birth script and ~99 % of
  randomized episodes. Confirms the prior report.
- **R4 / R4b (guarded F-greedy)** — the **cohort-greedy fallback is the
  weakness**: because it ignores non-imminent births, birth clusters build up to
  count-4 in several places at once and overwhelm the τ-override (R4b still
  leaks 268–289/480). Direct evidence that the **dynamic-greedy base in R1b is
  load-bearing**: it suppresses *all* windows' counts, so the override rarely
  faces simultaneous count-4s.
- **R2 / R3 (cohort-priority / starved-lock)** — both are τ=1-class and both lose
  `birth_cross_fork` and 130–176/480 randomized episodes; R3's cohort-only
  protection is blind to births' completions (a birth-replayed ES beats its
  dynamic fallback).
- **R1 (τ=1)** — foils all *structured* scripts except the one-move cross-fork,
  but the randomized attacker beats it 23–133/480 (broad) / 176–825/3000 (deep)
  via short (win ≤ 28) cross-fork lines. τ=1 is too late.
- **R1c (τ=3)** — foils all scripts (including the τ=2 break lines, which it
  foils at ply 23) yet is **globally the worst** completion-first variant
  (627–825/3000): reacting at count-3 forces the defender to chase the many
  count-3 windows of a magnet, neglecting suppression. Over-reaction.
- **R1b (τ=2)** — the survivor. Failures characterised next.

### R1b residual leaks — the sharpened frontier

All R1b break lines are **long grind-outs** (win placement 48–60; ~25+ attacker
stones building a sprawling 2-D cluster). Replaying one against R1b with endgame
threat instrumentation (`trace_r1b_breaks`, root `blocker_2_0`):

```
ply 56: D(τ2) plays (7,-4) | imminent: 1-empty cells={(7,-10),(7,-4)}  2-empty windows=3
ply 57: D(τ2) plays (7,-10)| imminent: 1-empty cells={(7,-10)}         2-empty windows=2
ply 58: A plays (11,-8)
ply 59: A plays (10,-8)      -> six at ply 60
```

At the last defender turn the board holds **3–4 simultaneous count-4 windows**;
two aimed blocks per turn cannot kill them all, so one matures. This is a
genuine **≥ 3-fold count-4 fork** — structural, not a tie-break artifact (no aim
over 2 stones clears ≥3 disjoint count-4 windows). Whether R1b's *earlier* play
could have prevented the fork from assembling is the open crux; τ=3 foils *these*
lines but assembles its own forks elsewhere, so raising τ is not the fix.

Note the counter-intuitive gradient: R1b leaks *more* at lower Φ (es_core 41 vs
blocker_3_0 2 per 3000) — fewer initial blockers means more open board for the
attacker to erect a far pileup before suppression bites.

---

## Candidate invariant for the proof lane

**Rule (R1b), precisely.** *At each defender placement, let `m` be the minimum
number of empties over all attacker-alive windows. If `m ≤ 2`, place on the
empty cell contained in the greatest number of alive windows that have exactly
`m` empties (ties broken by exact danger `Σ λ^{−#empties}` over all alive
windows, then lexicographically). Otherwise place on the dynamic touched-window
maximum-danger cell (argmax over empties of `Σ` over all currently alive windows
of `λ^{−#empties}`).* Two placements per turn apply the rule twice, recomputing
`m` between them.

**The quantity it maintains (that neither greedy maintains).** A **joint,
global** account at every attacker-turn boundary:

- **(I1) imminence floor:** every attacker-alive window has **≥ 3 empties** — no
  window is completable within the attacker's next turn (2 placements). Enforced
  by the `m ≤ 2` override, which spends the turn killing the most-imminent
  cluster.
- **(I2) global danger suppression:** between overrides the defender minimises
  the *all-windows* touched-window potential `Φ_all` (dynamic greedy), which
  keeps count-3 windows from co-arising in numbers that would let the attacker
  lift ≥ 3 of them to count-4 in a single turn.

Dynamic greedy maintains **(I2)** but not **(I1)** (completion-blind).
Fixed-cohort greedy maintains **(I1)** only over the frozen family `F`
(`ES_POTENTIAL` Thm 1) and nothing for births. R1b's novelty is enforcing
**(I1) over *all* windows** while retaining **(I2)** as the base — and the
empirical result is that (I1)+(I2) together defeat every structured attack and
all but a long-horizon pileup.

**Where the invariant is not maintainable (honest frontier).** R1b does **not**
keep (I1) against every attacker: the residual leaks are exactly the turns where
the attacker has forced **≥ 3 disjoint count-4 windows** to face one defender
turn, which two aimed blocks cannot clear. A *proof* that a completion-first
strategy maintains (I1)+(I2) forever therefore requires a **bound on the number
of count-4 windows the attacker can force to mature simultaneously** — i.e. the
birth-ledger bound that `ES_GLOBAL_BOUNDARY` leaves open
(`GAP-GLOBAL-RENEWAL`, `GAP-AMORTIZED-ABANDONMENT`). This matches the base-`√3`
design note in the proof: three separated count-4 targets give `Σ = 1.0` (not
`<1`), and the defender kills only two per turn — so the whole question is
whether the attacker can *renew* a 3-fold count-4 fork from births after `Φ`
leaves `<1`.

**Consequence for GAP-RAW.** The evidence is consistent with GAP-RAW being a
theorem (no `Φ<1` attacker win was ever forced; every rule's failures are
losses of a *fixed* policy, not proofs of an attacker win). It shows the witness
is completion-first + suppression, and it localises the remaining proof
obligation to the birth-ledger bound. It does **not** prove GAP-RAW.

---

## Honest scope — what was and was not covered

- **Covered.** 12 hand-built adversarial scripts (ES + D6 images, fresh/birth
  danger-magnet / birth-ES / double-birth / interleaved / three fork geometries)
  at full depth on 5 `Φ<1` roots; a strong randomized attacker (completion +
  threat-gain cluster building + fork/birth bias) for ~3–13 k episodes per rule
  per root at a 60-placement horizon; sound best-play-attacker exhaustion to
  plies 4 and 6; perturbations (D6 reflection, blocker translation, near-
  threshold two-blocker roots).
- **Not covered.** (1) The randomized attacker is a strong *heuristic*, not an
  optimal one — a "0 breaks" cell is **evidence, not proof** that no forced win
  exists; only the plies-4/6 exhaustion is a certificate, and it is shallow.
  (2) Horizons beyond 60 placements were not swept; R1b's long-grind leaks show
  the interesting regime is long, and the exhaustive arm cannot reach it (the
  ~230-wide depth wall). (3) The R1b leak positions were **not** re-examined
  under a *different* defender — so "R1b loses here" does not decide whether the
  position is an attacker win. (4) Two-attacker-stone `Φ<1` roots are sparse;
  the battery is dominated by single-attacker near-threshold shapes (the shapes
  the prior report showed dominate the threshold).

---

## Reproduction

Harness commit baseline `5fc2244b`; the adaptive rules, attacker, and reports
live in `packages/hexfield_eq/rust/src/gap_raw_hunt.rs` (test-gated;
uncommitted — the orchestrator gates/commits). Deterministic: no RNG on any
scored defender path; the randomized *attacker* uses fixed seeds
(`seed·0x9E3779B97F4A7C15 + bias·0xD1B54A32D192ED03 + 1`, biases `{0,15,40,70}`),
so every break line is regenerable. Set `CARGO_TARGET_DIR=.target-hunt`.

Validation (existing 13 tests + Φ/doc reproduction, unchanged, green):
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt -- --test-threads=1
```

Failure-mechanism trace (dynamic greedy vs ES, ply-by-ply danger + W status):
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::trace_dynamic_vs_es -- --ignored --nocapture --test-threads=1
```

Broad sweep (8 rules × 5 roots; scripts + 480 random + exhaustive 4,6):
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::adaptive_broad_sweep -- --ignored --nocapture --test-threads=1
```

Deep survivor (τ=1/2/3 × 4 roots; scripts + 3000 random + exhaustive; dumps
break lines):
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::adaptive_survivor_deep -- --ignored --nocapture --test-threads=1
```

R1b break-line classification (fork vs mis-aim; τ=2 vs τ=3 on the break lines):
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::trace_r1b_breaks -- --ignored --nocapture --test-threads=1
```

Node caps: exhaustive scan 2–4 M (broad/deep); all exhaustive rows at plies 4,6
complete under cap. Broad ≈ 4 min, deep ≈ 8 min on the shared host.
