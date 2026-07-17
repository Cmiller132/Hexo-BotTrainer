# R-CF1 census conjecture ladder

Status: **conjectures only**. Nothing in this file authorizes a production
refutation, cache entry, verifier change, or subtree deletion. A positive
shadow result means only “observed without a counterexample on the stated
profile.” Every candidate still needs a theorem, Lean proof, and a separate
consumption review.

## Contracts used by the shadow

The queue deliberately separates two contracts which must never be conflated.

1. A **stage-bounded semantic certificate** proves that claimant `A` cannot
   complete a six-window by an absolute placement deadline `T`. Its only
   conceivable consumer is `DepthCutoff` plus later reopen. A proof found at
   resolution `> T` is a `late_win`, not a counterexample. A proof with exact
   materialized resolution `<= T` is a counterexample.
2. A **forcing-grammar certificate** proves that the current
   `vcf_pair_complete` WideTurnGate/AND-OR search grammar has no positive WIN
   derivation. It is horizon-free only relative to that generator. Any PN=0
   subtree is a counterexample. It is **not** a game-deadness theorem and is
   not a strict-verifier theorem: verifier `Choice` nodes accept arbitrary
   legal placements and do not re-check WideTurnGate.

All window predicates below scan every `WindowStore::entries()` label, retain
the zero fallback, require exact opponent-free aliveness and exact residual
empties, and use checked wide arithmetic. No candidate uses the threat index
as a census surrogate. This is the hard repair boundary imposed by R-IG1 and
the reachable SecondStone `c=3` refutation.

## Ordered semantic proof queue

The catalogue below is organized by mechanism. The measured, dependency-aware
Lean order is frozen in “Shadow results and final proof order” below.

### S0 — `STAGE_DTW` (control; existing theorem, new scheduler contract)

Let `P` be a nonterminal interior supported-phase position with `A` to move,
`h = T - placements(P)` and `0 <= h <= 8`. Let

`c_A(P) = max({ count_A(W) | W is an exact labelled A-alive window } ∪ {0})`.

Under the landed coordinate guard, if the exact R-IG1 phase-table lower bound
`LB_phase(c_A(P)) > h`, then `A` has no WIN by `T`.

- Proof reuse: the landed DTW census theorem and all seven production repairs.
- Intended action after proof review: `DepthCutoff`, never permanent
  `Refuted`.
- Cost: one complete window scan.
- Role: measured control for all stronger bounded candidates.

### S1 — `DEFENDER_RESTORE4`

Let defender `D` be to move at a post-tactical supported phase and let `b` be
the remaining placements in `D`’s turn. Require a nonempty exact family of
current `A` count-at-least-four windows and exact hitting number `tau = b`.
An exact legal minimum hit set gives a successor `P'` with
`c_A(P') <= 3`: defender stones only kill A-alive windows and cannot birth an
A-stone. Consequently:

- from defender FirstStone (`b=2`), no A WIN occurs in fewer than
  `2 + LB_FirstStone(3) = 8` placements;
- from defender SecondStone (`b=1`), no A WIN occurs in fewer than
  `1 + LB_FirstStone(3) = 7` placements.

The shadow fires when this shifted lower bound is strictly greater than the
captured stage remainder.

- Proof reuse: exact minimum-hit-set service/hitting lemmas, DTW service
  restoration, R-IG1 FirstStone `c<=3` row, L7 completion counting.
- New Lean obligation: phase-exact sequential realization of the exhibited
  hit set and the `c_A(P')<=3` full-store lemma.
- Cost: shared threat analysis plus constant arithmetic.
- Intended action: bounded `DepthCutoff` only.

### S2 — `DEADLINE_ES`

For deadline `T`, let `a(P,T)` be the exact number of claimant placements in
the remaining phase clock. Require `a<=5` and define the complete finite target
family

`F(P,T) = { labelled A-alive W | 6-count_A(W) <= a(P,T) }`.

Every omitted touched window needs too many claimant placements (L7), every
mixed window is permanently blocked, and an initially all-empty/virgin window
needs six claimant placements, so `a<=5` makes `F` complete for wins by `T`.
For bins `n_i` over `F`, set

`alpha=n_1+3n_3+9n_5`, `beta=n_2+3n_4`.

The exact test for `Psi_F < t/3` is

`beta < 3t  AND  alpha^2 < 3(3t-beta)^2`.

Use `t=3` at defender FirstStone, `t=2` at defender SecondStone, and the
conservative `t=1` at either claimant phase. The conjectured conclusion is no
A WIN by `T`.

- Proof reuse: ES fixed-family Theorem 1, its round growth inequality,
  L7 completion counting.
- New Lean obligation: the finite-family completeness lemma and the two
  phase-handoff inequalities. Births outside `F` are irrelevant only because
  of the proved deadline completeness wrapper.
- Cost: one complete store scan and exact integer arithmetic.
- Intended action: bounded `DepthCutoff` only.

### S3 — `DEADLINE_ES_PREBLOCK`

This is `DEADLINE_ES` **or** the following exhibited-witness extension at a
defender-owned phase. Build the same complete `F(P,T)`. Sequentially validate
an exact ordered set `K` of the defender’s remaining one or two placements,
delete every `F` label hit by `K`, and require residual `Psi < 1/3`. The next
claimant pair raises fixed-family potential by less than a factor of three;
the position then reaches defender FirstStone with `Psi<1`, where ES Theorem 1
blocks `F` forever. Completeness of `F` turns that into no WIN by `T`.

- Proof reuse: S2 plus ES positive-danger legality and Theorem 1.
- New obligation: replay the ordered witness, including terminal prefixes and
  filler existence. No unordered-pair shortcut is allowed without P3.
- Cost: one scan, a deterministic danger cover, and at most two validated
  placements.
- Intended action: bounded `DepthCutoff` only.

### S4 — `DEADLINE_ES_TRIPLE`

This is `DEADLINE_ES` **or** an exact disjoint-triple certificate over the
same complete `F(P,T)`. Exhibit pairwise cell-disjoint triples `Q_j`, each
contained in the residual empties of a labelled A-alive window, such that
every `W in F` contains some whole `Q_j` in its residual empty set. After each
claimant pair, defender occupies one remaining cell in every triple touched by
that pair. Disjointness means at most two responses are required; a triple
cannot be filled by one claimant pair; and a response shares a labelled alive
window with an existing claimant stone, supplying legality. Thus every `F`
window remains blocked through `T`.

- Proof reuse: L7 completeness wrapper and the positive-response legality
  lemma; T8 is only a warning that global *pairing* is impossible, not a proof
  of this triple strategy.
- New Lean obligation: the phase-exact disjoint-triple invariant, including
  mid-turn terminal checks.
- Discovery cost: capped exact cover/backtracking; verification cost is linear
  in the certificate.
- Intended action: bounded `DepthCutoff` only.

### S5 — `FF_GAP` (designed, not implemented in this round’s first shadow)

On the same complete `F(P,T)`, solve the exact finite residual-hypergraph
Maker/Breaker game with the true 2:2 phase cadence and mid-placement wins.
Claimant may play any union cell plus a no-op filler (a supergame); defender
may use a no-op filler because a mandatory real placement cannot revive a
fixed blocked label. A Breaker certificate implies no real WIN by `T`.

This is the exact-occupancy fallback for c=3 overlap cases. It explicitly
avoids the false naive-SS move: labels, aliveness, residual gaps, overlap,
occupancy, phase, and terminal checks are all part of the state. Proofability
is medium; evaluation is exponential and must remain capped/Unknown.

### S6 — exposure-closed deadline family (designed, lower queue)

For `a>=6`, enlarge `F(P,T)` with every initially virgin window whose exact
D15/D16 activation exposure is within the remaining claimant budget, then
apply S2–S5. This is the finite horizon-parametric route for arbitrary finite
`T`, but current exact exposure labels are not serialized and several D15/D16
rows remain stated rather than proved. Queue only after S2–S5.

## Forcing-grammar auxiliary queue

These are useful search-taxonomy conjectures, not semantic deadness gates.

### G0 — `PAIR_SERVICE_C1/C2/C3`

At a post-tactical claimant FirstStone node, let `A_2,A_3` be the exact live
count-2/count-3 windows. For every singleton and unordered pair over the union
of their exact empties, construct the exact post-pair count-at-least-four
family. Require every nonempty family to retain a common residual hit. A pair
with one endpoint outside the union has the singleton’s effect; two outside
endpoints have no effect. Therefore no pair has post-pair hitting number two
or greater, so WideTurnGate emits no forcing pair. Defender threats can only
reject more pairs and are safely ignored. The three candidates add the
respective exact `c<=1`, `c<=2`, and `c<=3` screens.

Claim: **no WideTurnGate pair child**, not “no game WIN” and not “no strict
certificate.” Cost is quadratic in the upgrade-cell support and can duplicate
pair-generation work; its prize is primarily skipped generator wall, not
descendant expansion mass.

### G1 — `DEFENDER_REPLY_LIFT`, `TWO_CYCLE_LIFT`, `CENSUS_ATTRACTOR`

On the fully materialized resolved PN DAG, define a finite ranked predicate:

- rank 0: G0 pair service;
- Universal: dead if any resolved child is dead;
- Choice: dead only if the child list is nonempty, every edge is resolved,
  and every child is dead.

`DEFENDER_REPLY_LIFT` is the first Universal lift; `TWO_CYCLE_LIFT` is the
next complete Choice lift; `CENSUS_ATTRACTOR` is the least fixed point. By
AND/OR induction, a ranked node has no positive derivation in the exact
forcing grammar. Tactical/completion and unresolved edges block a Choice
classification. The shadow uses exact apply-with-delta/undo DAG replay and a
visited entry bitmap; no PN/DN outcome participates in predicate formation.

Proofability is high for the grammar theorem, but a separate argument would be
needed before this could justify any production behavior beyond the exact
generator.

## Explicitly discarded or parked

- **Raw higher c.** It cannot fix the unbounded-horizon mismatch; the maximum
  single-window DTW lower bound remains finite. SecondStone `c=3` is already a
  reachable counterexample to the naive strengthening.
- **Raw global ES `Phi<1`.** It fired zero times in Phase A and births make the
  unrestricted forever claim false/open. Path-observed birth mass is not a
  uniform strategy bound.
- **Independent regional potential sums.** Separate regions compete for the
  same two defender placements. They do not compose without a global budget
  invariant.
- **Threat-index census.** It omits live low-count labels and repeats a landed
  production repair.
- **Small defender zones as attacker confinement.** T3/T4 restrict defender
  replies; they do not confine remote quiet claimant turns.
- **LOSS-side raw color swap.** The official deep lane is a positive WIN
  search. Color symmetry supplies features, not a no-WIN theorem. S1 is the
  useful defender-side exposure statement.
- **Dead moat / finite basin forever claim.** A finite moat does not provide an
  infinite supply of frontier-inert defender fillers, so forever confinement
  is unproved. At most a separately budgeted finite-stage version is viable.

## Shadow results and final proof order

The official 18-position, 1 GiB lazy+gate profile completed 31 rows with zero
corpus failures. The post-solve audit had zero traversal errors, zero
counterexamples, and zero PN=0 deadline relations it could not resolve. For a
bounded candidate, a PN=0 proof with minimum exact resolution later than the
captured deadline is reported as a `late_win`, not a counterexample to its
finite claim.

### Ordered Lean proof queue

1. **`DEADLINE_ES` shared core.** Prove the `a<=5` complete-family wrapper and
   the three phase-handoff thresholds first. It fired 168,400 times
   (4.960789% of all interior nodes), had 0 counterexamples, and cost
   1,359.464 ms aggregate (about 0.400 us/evaluation). This theorem is the
   dependency for candidates 2 and 4 and is the cleanest cost/proof anchor.
2. **`DEADLINE_ES_PREBLOCK`.** Prove exact ordered witness replay plus the
   mandatory-filler lemma. It had the largest semantic surface: 256,386 fires
   (7.552714%), 411 ancestor-dominated roots, 0 counterexamples, and
   5,786.590 ms aggregate evaluation. It adds 87,986 fires beyond base ES.
3. **`DEFENDER_RESTORE4`.** Prove sequential hit-set realization and the
   full-store successor restoration lemma. The shadow did not assume it: it
   checked 717,197 exact legal sequences. Result: 224,761 fires (7.722814% of
   defender nodes), 411 dominated roots, 0 counterexamples, and 5,445.713 ms
   aggregate evaluation. This is independent of ES and leans most directly on
   the landed DTW service theorem.
4. **`DEADLINE_ES_TRIPLE`.** Prove the disjoint-triple response invariant only
   after the common deadline wrapper. It fired 171,408 times, just 3,008 more
   than base ES, with 0 counterexamples, 2,981.007 ms evaluation, and 82 capped
   discovery attempts. It survives but has materially less incremental value
   than pre-block.
5. **`FF_GAP`.** Implement as the exact capped oracle if the proof round needs
   more coverage of the positive c=3 overlap tail. It was designed but not
   shadowed here.
6. **D16 exposure closure.** Only after exact exposure labels and their stated
   rows are proved/available; it is the route beyond five claimant placements.

`STAGE_DTW` is a control rather than a new deadness theorem. It fired 42,914
times with 0 deadline counterexamples. What remains to prove is scheduler
glue: the captured expansion stage is not itself a certificate-resolution
deadline because atomic tactical leaves may resolve beyond it. Any consumer
must be an explicit absolute-deadline defer/reopen contract.

The enormous semantic expansion unions (3.357M–3.363M events) are **upper
bounds only**. Early finite deadlines dominate descendants that a later stage
would reopen, so these numbers are not predicted savings. The proof queue is
ranked by fire surface, incremental coverage, cost, proof dependencies, and
theorem proximity—not by pretending those unions are realizable deletions.

### Forcing-grammar verdicts

| candidate | fires | dominated roots | expansion mass | counterexamples | aggregate eval | verdict |
|---|---:|---:|---:|---:|---:|---|
| `PAIR_SERVICE_C1` | 42 | 42 | 42 | 0 | 14,506.756 ms shared | survives; negligible |
| `PAIR_SERVICE_C2` | 12,165 | 12,165 | 12,190 | 0 | 14,506.756 ms shared | survives; seed-only |
| `PAIR_SERVICE_C3` | 20,304 | 20,304 | 20,347 | 0 | 14,506.756 ms shared | survives; seed-only |
| `DEFENDER_REPLY_LIFT` | 36,336 | 36,217 | 57,806 | 0 | 81.726 ms shared postorder | survives grammar |
| `TWO_CYCLE_LIFT` | 9,332 | 9,332 | 34,626 | 0 | 81.726 ms shared postorder | survives grammar |
| `CENSUS_ATTRACTOR` | 86,331 | 33,990 | 138,303 | 0 | 81.726 ms shared postorder | survives grammar |

Do not put these into the semantic Lean queue. Pair service is exact but costs
14.507 s and almost every fire is already a leaf with one expansion. The
attractor covers a larger descendant union, but its predicate is computed
from those already solved descendants; it is a ranked proof/cache closure,
not demonstrated online savings. If pursued, prove G0 first and then the
ranked AND/OR induction, followed by a separate prospective evaluator study.

### Final consumption verdict

All positive entries remain **CONJECTURE / SHADOW-SURVIVOR**. There is no
permanent unbounded semantic deadness candidate ready for Lean, no production
refutation, and no verifier change. The semantic queue proves finite absolute
deadlines only; the grammar queue proves only the exact forcing generator.
Raw evidence is retained in `CENSUS_DEEP_PHASE_C_RAW.log`; it contains no
`CENSUS_DEEP_COUNTEREXAMPLE` specimen because none was observed.
