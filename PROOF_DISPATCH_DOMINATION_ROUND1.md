# Dispatch domination, round 1

> **Final dispatch.** Authored against worktree base `6b853c0e` (the U11
> hunt landing); this document itself is committed as `7e240388`, the
> commit reviewed by `REVIEW_DISPATCH_DOMINATION_ROUND1.md`
> (ACCEPT-WITH-EDITS: all five theorem regimes CONFIRMED; its 7-item
> repair list is binding on the b=2 experiment round — repairs 2–6 amend
> §7's protocol before any grind, and its analytic baseline
> `O_0=O_1=O_2=Unknown` for covered b=2 candidates makes the d=1,2 core
> a smoke test, not discriminatory evidence). This file was written in
> statement-first order; the controlling statements, full proofs, evidence
> audit, future experiment, and attack surface now appear below.
> The imported rule model and domination notation are from
> `docs/proof_parts/DOMINATION.md` in the `hexfield-eq-main-review-65bd3a`
> worktree.  Only the window, budget, and radius-8 conventions D1--D6 are
> looked up in `docs/PROOF_TSS_DEFENDER_ZONES.md`.

## 1. Status contract

`PROVEN` means that an all-continuation pencil argument is complete.
`VERIFIED` is reserved for an exhaustive machine fact, not a sampled hunt.
`CONJECTURED` means that some proof obligation remains.  The hunt counts are
reported as sampled measurements and are not silently promoted to
`VERIFIED`.

| Target | Status | Controlling claim |
|---|---|---|
| L-DISPATCH-B1 | **PROVEN** | A b=1 reply leaving a current attacker count-4/5 window unhit is dominated by **every** legal full-coverer at every finite horizon.  From horizon 2 onward the non-coverer is worst-valued.  No coverer-to-coverer equivalence is asserted. |
| L-DRQ | **PROVEN** | At a post-opening nonterminal node, any two distinct empty dead cells give mutually outcome-dominating replies at every finite horizon.  Frontier-inertness is a consequence of deadness, not an extra hypothesis. |
| b=2 spare-stone sub-algebra | **CONJECTURED / experiment only** | No nontrivial spare-stone domination theorem is claimed.  Section 7 proves only the elementary uncovered-pair deadline as a calibration and specifies the exact-oracle falsification experiment. |

## 2. Adopted notation and exact node scope

Use Definitions 1--5 and Lemmas 1--7 of `DOMINATION.md`.  In particular,
`n` counts placements **after** the compared reply, and

\[
 b\preceq_n a\quad\Longleftrightarrow\quad
 V_D^n(P+a)\ge V_D^n(P+b)
 \tag{D-L3}
\]

by its Lemma 3.  A window has six cells.  It is alive for player `A` when it
contains no `D` stone, and is dead when it contains both colours.  Stones,
deadness, and legal support are permanent in the senses of Lemma 2.  A legal
post-opening placement lies within radius 8 of an old stone.

For a nonterminal position `P` with defender `D` to place, put

\[
 \mathcal T_A(P)=\{W:W\text{ is alive for }A,
                         \operatorname{cnt}_A(W,P)\ge4\}.
\]

For `W` in this family let `E_P(W)` be its one- or two-cell empty set.  A
legal reply `z` *hits* `W` iff `z in E_P(W)`, and is a *full-coverer* iff it
hits every member of `T_A(P)`.  Thus `mhs(P)=1` means that at least one such
cell exists.  It does **not** say that different full-coverers have equal
continuation values.

For L-DISPATCH-B1, `P` is specifically a defender `SecondStone{first}`
state.  The defender's first placement of this turn is already part of
`sigma_P`; exactly one defender placement remains.  If that placement is
nonterminal, phase advancement gives the attacker `FirstStone`, hence two
consecutive placements before the defender can act again.  This is the
entire meaning of `b=1` here.  Both compared branches have the identical
parent `P`: the stored `first` coordinate, complete occupancy and labelled
window masks, legal support, current player, and absolute placement counter
are fixed.  This theorem does not compare two different first placements.

## 3. L-DISPATCH-B1 -- controlling statement

### 3.1 Named hypotheses

- **DB1-RULE:** `P` is a reachable, finite, nonterminal, post-opening rule
  position in the unbounded-lattice model of `DOMINATION.md`.
- **DB1-PHASE:** `D` is to place at `SecondStone{first}` (remaining defender
  budget one).
- **DB1-NOWIN:** `own_win_now(P)` is false.  At budget one this excludes a
  `D`-alive count-5 window and therefore excludes an immediate defender win
  by any legal reply.
- **DB1-THREATS:** `T_A(P)` is nonempty and consists of all current live
  attacker count-4/5 windows.
- **DB1-MHS1:** `mhs(P)=1`; equivalently here, at least one legal
  full-coverer exists.
- **DB1-C:** `c` is a legal reply but not a full-coverer, so some
  `W in T_A(P)` has `c notin E_P(W)`.
- **DB1-A:** `a` is an arbitrary legal full-coverer.  The quantifier is
  `for every a`; no equivalence between two choices of `a` is asserted.

### 3.2 Theorem (sharpened horizon and quantifier)

**L-DISPATCH-B1 [PROVEN].** Under DB1-RULE--DB1-A,

\[
       c\preceq_n a\qquad\text{for every finite }n\ge0
       \text{ and every legal full-coverer }a.
\]

From horizon 2 onward there is a stronger pairwise conclusion: `c` is
dominated by **every legal defender reply**, whether or not that reply is a
full-coverer.  DB1-MHS1 and DB1-A are availability/interface conditions that
make the result useful to the dispatch arm.  The hunt's stated `n>=3` result
is therefore a strict corollary.

### 3.3 Proof skeleton

1. DB1-NOWIN makes `P+c` nonterminal.
2. Choose the unhit `W`.  Since `P` is nonterminal, its attacker count is
   four or five, so `E_P(W)` has two or one cells.  The reply `c` is not in
   `W` and changes none of those cells or `W`'s masks.
3. Every empty of `W` is already legal: it is at distance at most 5 from an
   old attacker stone in the same window, below the legality radius 8.
4. The attacker receives two consecutive placements.  It fills the one or
   two empties and wins no later than the second further placement.  No
   defender structure, killed count-at-most-3 attacker window, or new legal
   frontier created by `c` can move before that terminal placement.
5. Hence `V_D^n(P+c)=-1` for every `n>=2`.  Since `-1` is the minimum value,
   `V_D^n(P+a)>=V_D^n(P+c)` for every legal `a`; invoke `DOMINATION.md`
   Lemma 3.
6. At `n=0`, both reply successors are nonterminal.  At `n=1`, a
   full-coverer has blocked every pre-existing attacker count-5 window, so
   the attacker cannot win in one placement after `a`; after `c` its value
   is either `?` or an attacker win.  These two short horizons have the same
   domination direction.

Section 6.1 explicitly audits early terminality, all three causal channels,
the `any a` quantifier, and the distinction between the theorem and the
narrower hunt comparison.

### 3.4 Hypothesis-boundary ledger

| Hypothesis | Role in the theorem | Boundary evidence / disposition |
|---|---|---|
| DB1-RULE | Fixes the imported formal-game domain and excludes a pre-existing terminal window. | This is a rule-domain condition, not a tactical guard.  Reachability is **possibly droppable** for a generalized raw-mask theorem if nonterminality, phase consistency, and the legal-support formula are separately assumed; no such generalization is needed here. |
| DB1-PHASE | Supplies two uninterrupted attacker placements after the compared reply. | **Proof-load-bearing** for the two-placement value-floor argument; theorem-level necessity beyond b=1 is **possibly droppable / undecided**.  The existing reachable `spare_tempo_fixture` is b=2, `mhs=1`, and `!own_win_now`; its non-covering first stone `(4,4)` followed by the sole guard `(1,-3)` makes the attacker lambda-one forced-loss.  Thus a non-covering **first** stone is not itself lost when a spare remains, but this does not by itself prove a value reversal against every full-coverer.  Completed-turn b=2 dominance is assigned to Section 7. |
| DB1-NOWIN | Prevents `c` itself from ending the game for `D`. | Load-bearing.  The reachable row-0/row-5 counterexample below has all other B1 premises, but a non-coverer wins immediately and is not dominated by a full-coverer.  The sharp pairwise replacement is merely “`P+c` is nonterminal”; global `!own_win_now` is the cheap gate that guarantees it for every omitted reply. |
| DB1-THREATS | Uses the exact family of all current count-4/5 attacker windows. | Completeness is load-bearing only for the added `n=1` strengthening and is **possibly droppable** if the theorem is stated for `n>=2`.  The existence of a missed member with count at least four is load-bearing for the two-placement loss.  Extension to count 3 is **CONJECTURED / not claimed**: three empties cannot be filled in the next two-stone turn, so this proof does not apply. |
| DB1-MHS1 | Ensures a full-coverer exists; not used to prove `c` loses. | **Possibly droppable** from the conditional pairwise lemma, retained for the engine dispatch interface. |
| DB1-C | Supplies an unhit count-4/5 window. | Load-bearing for the universal pruning rule.  Without it the canonical `d7e1b56c925b7f32:20` pair consists of two full-coverers with opposite proven outcomes, so one cannot be deleted merely because the other exists. |
| DB1-A | Names a retained reply in the common intersection. | No purity, dead-spoke, support-equality, or counterfork guard is needed.  “Any `a`” is proved from the value floor, not from interchangeability. |
| Any structural/frontier purity guard on `c` | None is assumed. | Deliberately absent: the forced win occurs before that value can be exercised. |

The promised DB1-NOWIN counterexample is the following legal replay, with
`A=Player0`, `D=Player1`; semicolons separate turns and the last listed stone
is D's first stone of its current turn:

```text
(0,0);
(0,5),(1,5);
(1,0),(4,0);
(2,5),(3,5);
(5,0),(-5,0);
(4,5)
```

At the resulting D-`SecondStone` node, A's row-0 stones are at
`q={-5,0,1,4,5}`.  Its sole count-4 window is the segment `q=0..5`, with
empties `(2,0),(3,0)`, so `mhs=1`.  D has row-5 stones at
`q=0,1,2,3,4`.  The non-coverer `c=(5,5)`
wins for D immediately.  The full-coverer `a=(2,0)` is nonterminal; A can
then occupy `(-1,5)` and `(5,5)`, killing both possible row-5 completions.
After `a`, every other D window contains at most the off-row D stone `(2,0)`
and one of the row-5 stones, so D has no alternative one-placement
completion at further placement 3.
Through horizon 3 after the reply, `P+c` has value `+1`, whereas the latter A
strategy keeps `P+a` at value at most `0`.  Hence `c not preceq_3 a` once
DB1-NOWIN is removed.

## 4. L-DRQ -- controlling statement

### 4.1 Named hypotheses

- **DRQ-RULE:** `P` is a reachable, finite, nonterminal, post-opening rule
  position.
- **DRQ-MOVER:** the current mover is `X` in either engine-player identity and
  at either normal phase (`FirstStone` or `SecondStone`).  For applying P1,
  rename its strategic role `D` to `X` and `A` to the opponent.
- **DRQ-CELLS:** `x != y` are empty cells and every window in
  `Omega(x) union Omega(y)` is already dead in `P`.

Legality and frontier-inertness need not be independent hypotheses:
`DOMINATION.md` Lemma 7 gives, separately for `z=x,y`,

\[
 B_8(z)\subseteq\Lambda(P),
\]

and in particular says that the empty dead cell is legal and adds no support.

### 4.2 Theorem

**L-DRQ [PROVEN].** Under DRQ-RULE--DRQ-CELLS, for every finite `n>=0`,

\[
 x\preceq_n y\quad\text{and}\quad y\preceq_n x,
 \qquad\text{hence}\qquad
 V_X^n(P+x)=V_X^n(P+y).
\]

Thus all currently empty dead cells at one node may be represented by one
member when comparing rule outcomes.  “Pass-into-dead-region” is only a
macromove name: the placement still consumes its stone and advances the
ordinary phase machine; this theorem does not equate it with a literal pass.

### 4.3 Proof skeleton

1. Apply Lemma 7 to both cells.  Their successor supports are both exactly
   `Lambda(P)`.
2. Apply Pattern P1 with searched reply `a=x` and discarded reply `b=y`.
   P1-M holds because `y` is dead; P1-LF holds because `x` is frontier-inert.
   Obtain `y preceq_n x` for every `n`.
3. Swap `x,y` and apply P1 again.  Obtain `x preceq_n y`.
4. Lemma 3 turns the two inequalities into value equality.  The body of P1
   is the required future-relevance argument: Lemma 2 prevents mask rebirth,
   support equality handles legalization, and its `x<->y` transposition maps
   a later attempt to occupy the branch-exclusive cell.

### 4.4 Hypothesis-boundary ledger

| Hypothesis | Role in the theorem | Boundary evidence / disposition |
|---|---|---|
| DRQ-RULE | Places the claim in the exact domain of Lemma 7 and P1. | This is an imported-domain condition.  Raw unreachable assignments are **possibly droppable** only after restating the upstream lemmas for that domain; Opening is outside P1 and in any event has no dead empty cells. |
| DRQ-MOVER | Names the current mover as P1's strategic D role at either normal phase. | Engine-player identity is freely droppable by colour symmetry.  A normal phase is an upstream P1 domain condition, not a new spatial hypothesis. |
| Distinct empty cells | Makes both coordinates legal candidate replies and the collapse nontrivial. | Emptiness is definitional for a placement.  Distinctness is **possibly droppable** but yields only the tautology that a reply equals itself. |
| Both cells dead | Supplies P1-M in both directions and, through Lemma 7, both frontier facts. | Load-bearing for this P1 corollary, although not necessary for every possible equivalence theorem (P2 is another sufficient regime).  In the canonical `d7e1b56c925b7f32:20` witness a direct replay support calculation gives \(\lvert B_8(-2,3)\setminus\Lambda(P)\rvert=\lvert B_8(-1,2)\setminus\Lambda(P)\rvert=0\), but the cells lie in a live count-4 window and have opposite proven outcomes.  Thus frontier-inertness alone cannot replace dead-mask inertness. |
| Explicit frontier-inertness | None beyond deadness. | **Removed from the hunt statement as an independent hypothesis**: Lemma 7 proves it. |
| Same phase/mover | Automatic because the replies share one parent. | Definitional, not an extra spatial guard. |

## 5. Machine evidence and exact scope

The hunt is evidence, not a proof and not an exhaustive verification.

- DISPATCH: 3,412 of 4,001 sampled defensive `SecondStone` nodes had
  `mhs=1` (85.3%).  The code compared `full.first()`--one deterministic
  full-coverer--against legal non-full-covering **defender counter-threat
  cells**, for 20,495 comparisons, and found zero referee refutations.  It
  did not enumerate every non-coverer against every full-coverer.  The
  theorem's stronger quantifier comes from the forced-loss proof, not that
  sampled loop.
- DRQ: the detector found 5,133 unordered dead/frontier-inert pairs in the
  4,001-node sample.  It adjudicated at most `de.first()` per node: 288
  narrow-referee comparisons, with zero categorical mismatches.  In
  particular, `UNKNOWN/UNKNOWN` counted as agreement and DRQ had no wide
  confirmation stage.  These nodes were defensive `SecondStone` nodes; the
  all-mover/all-phase extension comes from P1, not the scan.
- COVERER boundary: `dom_hunt_counterexamples.jsonl` records four
  doubly-proven failures of unrestricted full-coverer interchange.  The
  canonical `d7e1b56c925b7f32:20` replies `(-2,3)` and `(-1,2)` have opposite
  proven outcomes.  L-DISPATCH-B1 compares a losing **non-coverer** with each
  coverer and makes no coverer-to-coverer comparison.

No new machine grind is presently required for either proof: the alleged
DRQ legalization gap is discharged analytically by Lemma 7, and dispatch is
a two-placement terminal argument.  Section 8 will retain regeneration
commands for the existing evidence and any check added during review.

## 6. Proofs

### 6.1 Proof of L-DISPATCH-B1

Choose `W in T_A(P)` missed by `c`.  Since `W` is alive for A and `P` is
nonterminal, its A-count is either four or five and its empty set has size
two or one.  The legal reply `c` is empty.  If it lay in `W`, it would be one
of precisely those empties and would hit `W`; therefore `c notin W`.
Placing it changes neither labelled mask of `W`.

First discharge terminal order.  If `P+c` were an immediate D win, the
completed window had five D stones, no A stone, and unique empty `c` in `P`.
At remaining budget one this is exactly a D count-5 `own_win_now` witness,
contrary to DB1-NOWIN.  A D placement cannot complete an A window.  Thus
`P+c` is nonterminal.  Because the old phase was `SecondStone{first}`, the
ordinary phase transition makes A the mover at `FirstStone`, with two
placements before D can act.

Every `e in E_P(W)` was legal already in `P`.  Pick any old A stone `s` in
`W`.  Fact F1 in the shared conventions (equivalently Definition 1's
length-six geometry) gives `d(e,s)<=5<8`; hence `e in Lambda(P)`.  The reply
`c` occupies none of these empties, and legal support is monotone by
`DOMINATION.md` Lemma 2.  If there are two empties, placing the first also
cannot make the second illegal.

Define one A strategy after `c` from this fixed witness `W`.

- If `cnt_A(W,P)=5`, play its unique empty and win at further placement 1.
- If `cnt_A(W,P)=4`, play either empty.  If that placement completes some A
  window, the desired terminal result has occurred even earlier.  Otherwise
  A remains the mover at `SecondStone`; play the other empty and complete
  `W` at further placement 2.

The strategy is legal and is never interrupted by D.  Consequently, for
every D continuation strategy and every `n>=2`, A forces `A@t` for some
`t<=2`, and

\[
V_D^n(P+c)=-1. \tag{1}
\]

This identity disposes of all alleged “other value” of `c`.  It may kill any
number of A count-at-most-3 windows, add a D stone to several live D windows,
create a counterfork, or add an unmatched radius-8 ball.  Those are genuine
mask, occupancy, and frontier effects under `DOMINATION.md` Lemma 1, but D
gets no placement on which to exploit them and A's winning cells were legal
from old stones, independently of the new ball.

For every legal reply `r`, `V_D^n(P+r)>=-1`.  Combining this bound with (1)
and applying `DOMINATION.md` Lemma 3 gives

\[
c\preceq_n r\qquad(n\ge2). \tag{2}
\]

In particular (2) holds separately for every full-coverer `a`.  This is why
the universal quantifier is sound even though distinct coverers can have
opposite values: the proof never compares their structures or asserts that
their values agree.

It remains only to include the two shorter stopped horizons in the stated
theorem.  At `n=0`, DB1-NOWIN makes both `P+c` and `P+a` nonterminal, so both
values are `0`.  At `n=1`, every A window that could be completed in one
placement would already have been A-alive with count five in `P`.  It belongs
to `T_A(P)` and is hit--therefore killed--by the full-coverer `a`.  A D stone
creates no new A count-5 window, so A cannot win in one placement after `a`;
`V_D^1(P+a)=0`.  After `c`, A either has an unhit count-5 window and value
`-1`, or no one-placement win and value `0`.  Thus
`V_D^1(P+a)>=V_D^1(P+c)`.  Lemma 3 proves `c preceq_n a` also for `n=0,1`.
This completes all finite horizons.  QED.

For readers who prefer Definition 5's transfer quantifiers, the `n>=2`
certificate is even simpler than a branch simulation.  Send every discarded-
branch D policy to any legal searched-branch D policy; send every searched-
branch A challenge to the fixed `W`-completion strategy after `c`.  The
right-hand utility is always `-1`, while the left-hand utility cannot be
smaller.  No future move map between the two legality frontiers is required.

**Engine corollary.** At b=1, the `vcf_pair_complete` implementation's
`extendable_hit_kernel` is exactly the intersection of all threat-empty sets,
and the verifier independently derives the same kernel.  The theorem
therefore certifies the b=1 arm of shipped `implicit_dispatch`: the
complement may be omitted while every full-coverer remains explicit.  The
production boolean also has a distinct b=2 `mhs=b=2` arm; this round does not
claim to prove that arm.  This b=1 complement dismissal is the sound
sub-hitting dispatch kernel that U11 may import; it is not a certificate for
U11's unresolved spare-stone cases.

### 6.2 Proof of L-DRQ

Rename the current mover X as the strategic role D and the other player as A.
The rule model is colour-symmetric, so Pattern P1 applies with this naming at
either normal turn phase.

Apply `DOMINATION.md` Lemma 7 separately to the empty dead cells `x` and `y`:

\[
B_8(x)\subseteq\Lambda(P),\qquad
B_8(y)\subseteq\Lambda(P). \tag{3}
\]

The same lemma says both cells are legal.  They add no support, and hence

\[
\Lambda(P)\cup B_8(x)=\Lambda(P)
=\Lambda(P)\cup B_8(y). \tag{4}
\]

Neither reply can be an immediate X win: every window whose X mask it can
change lies in its `Omega` set and already contains an opponent stone.  For
the first direction, instantiate Pattern P1 with searched reply `a=x` and
discarded reply `b=y`.  P1-M is DRQ-CELLS at `y`, and P1-LF is (4).
Therefore

\[
y\preceq_n x\qquad\text{for every finite }n. \tag{5}
\]

Swap the two cells.  P1-M now holds at `x` and the same equation (4) supplies
P1-LF, giving

\[
x\preceq_n y\qquad\text{for every finite }n. \tag{6}
\]

By `DOMINATION.md` Lemma 3, (5)--(6) force equality of the stopped-horizon
minimax values.  This is the claimed mutual outcome-equivalence.  QED.

There is no residual “might become relevant later” premise hidden in this
corollary.  `DOMINATION.md` Lemma 2 keeps all windows through either cell dead
forever.  P1's occupancy channel explicitly transposes a later attempt to
play the branch-exclusive counterpart.  Its frontier channel starts from
(4), observes that common outside moves add identical radius-8 balls, and
also tracks the transposed `SecondStone{first}` payload when the compared
reply was a first stone.  Those are precisely mask rebirth, occupancy, future
legalization, and phase drift.  Requiring a new persistence hypothesis would
duplicate the imported proof.

The corollary equates formal rule outcomes, not byte-level states, ordered
histories, model features, or the dead placement with a literal pass.  An
implementation may keep one currently legal representative of the class,
but must recompute the class at each node and still apply the representative
through the ordinary placement/phase transition.  Removing the detector's
explicit `frontier_inert` conjunct has zero mathematical fire-rate cost on
reachable positions: Lemma 7 proves it for every cell that passes
`cell_dead`.  This is the rule-outcome quotient U24 may use for its
dead-region macromove class; it does not justify merging the representation
channels excluded in the preceding sentence.

## 7. b=2 sub-algebra: the exact-oracle experiment

**Status: CONJECTURED / DESIGN ONLY.** No oracle grind was run in this round.
The primary subject is the spare-stone boundary `b=2, mhs=1`; `mhs=2` is
included as a secondary control because it exercises two-cell hitting-set
identity.  The only b=2 fact proved here is the elementary uncovered-pair
deadline B2-COVER, included to calibrate the future harness; no nontrivial
spare-stone domination is promoted.  A clean finite experiment can refute a general algebra and can
verify a frozen finite matrix, but cannot promote an all-position theorem to
PROVEN.  Any surviving general claim returns to a pencil-proof round.

### 7.1 Compare completed turns, not first cells

Let `P` be a reachable nonterminal defender-`FirstStone` position, let D be
the mover and A the opponent, and assume `!own_win_now(P)`.  Let `T` be the
complete nonempty family of current A-alive count-4/5 windows.  A defender
macromove is an ordered pair

\[
M=(u,v)
\]

such that `u in L(P)`, `P+u` is nonterminal, and `v in L(P+u)`.  Apply both
placements with the real phase machine.  If the second placement is also
nonterminal, write `C_M=P+u+v`; A is then at `FirstStone`.  A terminal D win
would be recorded as the best defender value and never passed to the oracle,
but under this section's predicate it cannot occur: a D completion in at
most two stones requires a D-alive count-5 or count-4 window already in P,
which `own_win_now` detects at b=2.  A D placement cannot complete an A
window.  The harness nevertheless asserts nonterminality after each
placement as an implementation check.  Say that `M` *covers T* when every
`W in T` contains at least one of `u,v`.

At `mhs=1`, put

\[
H(P)=\bigcap_{W\in T}E_P(W),
\]

the possibly multi-cell set of single-stone full-coverers.  The experiment
also defines

\[
U(P)=\bigcup_{W\in T}E_P(W).
\]

Every member of `U(P)` is already legal in P because it shares a length-six
window with an old A stone at distance at most 5.  This gives an exact,
non-overlapping coverage classification after P3 quotienting:

1. At **K1** (`H(P)` nonempty), every covered completed successor is exactly
   one of:
   - an old/old pair `{h,s}` with `h in H(P)` and
     `s in L(P)\{h}`;
   - an old/old **split cover** `{p,q} subseteq U(P)\H(P)` that covers T; or
   - a directed frontier pair `h;v` with `h in H(P)` and
     `v in L(P+h)\L(P)`.
2. At **K2** (`H(P)` empty and `mhs=2`), every covered successor is an
   old/old pair `{p,q} subseteq U(P)` that covers T.
3. An **uncovered pair** is a mandatory losing calibration, not a candidate
   defense.

The reason the directed case must start with `h` is useful: a newly legalized
`v` is not in `U(P)` and hits no initial T-window, so the first stone must
cover T by itself.  Coverage type (H-containing versus split versus
uncovered) and order type (P3-quotiented old/old versus directed-new) are
separate tags; “directed” is not a fourth coverage class.

Restricting the study to “one full-coverer plus a spare” would assume away
the main sub-hitting question: even when a one-cell transversal exists, two
partial hitters can also cover the family and may carry different structure.

Pattern P3 may canonicalize `u;v` and `v;u` only when both cells were legal in
P and both singleton successors were nonterminal.  The old/old cases above
satisfy those conditions under `!own_win_now`, which the harness still
asserts.  A cell in `L(P+h)\L(P)` remains a directed hit-first spare and is
never silently sorted.  Enumeration is ordered first; quotienting happens
only after the P3 predicate succeeds.

### 7.2 Exact stopped-horizon value

For each nonterminal completed turn and depth `d`, call the player-identity
API explicitly:

```rust
tss_reference_fast::solve_for_player(&C_M, attacker_a, d, config)
```

Define defender rank from the returned status (which is relative to A):

\[
\rho_D(\texttt{Loss})=2,\qquad
\rho_D(\texttt{Unknown})=1,\qquad
\rho_D(\texttt{Win})=0. \tag{7}
\]

Thus retained macromove `M` dominates omitted macromove `N` at depth `d`
exactly when `rho_D(O_d(M)) >= rho_D(O_d(N))`; equivalence requires equality.
Oracle depth starts **after** both D placements.  It is a total placement
budget of `d+2` from P, or `d+1` further placements after the first compared
reply in Definition 3's convention.

Completed pairs are the atomic data, but a claim that prunes a **first**
stone needs one further minimax aggregation.  There is no intervening A move,
so for each legal first stone `u` define

\[
F_d(u)=\max_{v\in L(P+u)}\rho_D(O_d(u,v)), \tag{8}
\]

including an immediate terminal D win as rank 2.  A sampled set of second
stones cannot adjudicate `F_d`: every legal `v`, including newly legalized
ones, must be included.  The sole shortcut is the analytically PROVEN
B2-COVER lemma below: uncovered pairs have rank 0 for `d>=2`; its oracle
instances are harness controls, not the justification.  A first-stone dismissal
`c preceq h` at this
bounded horizon requires `F_d(h)>=F_d(c)`.  This aggregation is what turns a
macromove experiment into an answer about the b=2 dispatch arm.

When an old/old pair is P3-quotiented, its manifest row retains **both** first-
action aliases.  For example, the same successor `{c,h}` contributes to
`F_d(c)` through `c;h` and to `F_d(h)` through `h;c`; canonicalizing the final
state must not erase either membership.

`Unknown` from a fully completed depth-limited recurrence is an exact stopped-
horizon value: neither player can force a terminal result by that depth.  A
node/time/work cutoff is different and must be serialized as `INCOMPLETE`,
never as `Unknown`.  No comparison containing `INCOMPLETE` adjudicates a
claim.

For fixed-hit comparisons define `S_h=L(P+h)`, the exact second-stone legal
set.  Deadness, G3 degree, purity, frontier delta, DRQ, and P2 predicates are
all evaluated in the state `P+h`.  A “fixed spare” comparison across several
hits uses the same ordered role `(h,s)` and only hits for which
`s in S_h`; it never compares a second stone in one branch with a first stone
in another.

### 7.3 Pre-registered subclaims and decisive patterns

| ID | Proposed statement | Exact result that refutes it | Meaning of a clean bounded run |
|---|---|---|---|
| **B2-COVER [PROVEN]; oracle control** | A completed turn that leaves an initial T-window uncovered loses at the exact remaining-empty deadline. | Put `r(M)=min{|E_P(W)|: W in T is uncovered by M}`, which is 1 or 2.  The same legality/tempo argument as L-DISPATCH gives `O_r(M)=Win`; any oracle disagreement attacks the harness/oracle. | Analytically, `O_0=Unknown`; at `d=1`, status is `Win` iff `r=1` and otherwise `Unknown`; for `d>=r`, status is `Win`.  These exact values may shortcut uncovered pairs in `F_d`. |
| **B2-P3-ORDER** | Where P3's hypotheses hold, `u;v` and `v;u` agree. | Any exact status mismatch at a matched depth. | Metamorphic positive control for phase and move-order handling. |
| **B2-K1-FIRST-DISMISS-INDEXED** | At each depth, every legal first stone `c notin H(P)` is no better than at least one full-coverer first stone (the `h` may depend on depth). | After exhaustive second-stone aggregation, `F_d(c) > max_{h in H(P)} F_d(h)` at any depth. | If all `v` are enumerated, verifies only the finite node/depth first-move restriction.  The stronger “every h dominates c” is separate and is not required for retaining the H-first set. |
| **B2-K1-CONTAIN-INDEXED** | At each depth, every covering split pair `N` is dominated by at least one H-containing pair (which may depend on depth). | `rho_D(N)` exceeds the maximum rank of the **complete** H-containing set at any depth. | Verifies only `for every tested d,N there exists M` on a completed frozen matrix. |
| **B2-K1-CONTAIN-UNIFORM** | For each split pair `N`, one H-containing `M` dominates it at every tested depth. | The intersection over tested depths of `{M: rho_D(O_d(M)) >= rho_D(O_d(N))}` is empty, even if the indexed claim survives. | Verifies only the frozen `for every N there exists M for every tested d`; the general all-horizon claim remains CONJECTURED. |
| **B2-K1-SPARE-ANY-EQ** | For fixed `h`, all ordered completions `(h,s)` with `s in S_h` are equivalent. | Any two such spares with unequal exact status at one depth. | Equality only on the completed finite matrix; zero mismatches is not a general proof. |
| **B2-K1-HIT-ANY-EQ** | With a fixed second-stone `s` legal after every compared hit, the ordered completions `(h,s)` are equivalent over those `h in H(P)`. | Any unequal exact status. | Expected negative control: the b=1 coverer corpus already warns that hit identity can be decisive. |
| **B2-K1-DRQ-LIFT** | After a fixed first hit `h`, two dead second-stone cells are equivalent. | Any exact mismatch. | Positive control for applying L-DRQ at `P+h`; a mismatch attacks the harness/imported machinery, not a new conjecture. |
| **B2-K2-HSET-ANY-EQ** | All minimum two-cell hitting sets at `mhs=2` are equivalent. | Any two sets with unequal exact status at one depth. | Bounded evidence only and another expected negative boundary. |
| **B2-K2-P2-LIFT** | If, after a common first stone `z`, second cells `x,y` satisfy P2-M and P2-LF, then ordered completions `(z,x)` and `(z,y)` are equivalent. | Any exact mismatch. | Positive control for a valid P2 lift. |

For an equivalence conjecture, one unequal completed value refutes the
all-horizon statement.  For one-way dominance, equality is consistent and
only the reversed inequality refutes it.  Any future directional guard must
define its state-relative predicate and predicted `M,N` direction **before**
enumeration; this round intentionally pre-registers no vague “counterfork is
better” claim.  Representative sampling can find a
counterexample but cannot validate words such as “any” or “all”; Section
7.4 therefore includes a small full-cross-product audit.

### 7.4 Frozen position families

The source universe is the same decisive-game human corpus schema used by the
hunt (`TSS_DOM_CORPUS`, rows with winner `+1/-1`), enumerating every legal
replay prefix before any solve.  The selection seed is the main-sweep seed
`7766554433221100`.  The driver computes and records the corpus SHA-256.

Selection is result-blind and reproducible.  For each panel, bucket eligible
parents by the listed stratum-bit tuple.  Where a panel defines candidate
signatures below, this parent tuple is the presence/absence bit vector of
those signatures in their printed Boolean order.  Within a bucket, sort by the SHA-256
digest of the UTF-8 string
`seed|panel-id|game-hash|prefix`, then by `(game_hash,prefix)`; round-robin
over lexicographically sorted nonempty bucket keys until the panel quota is
met.  A genuine shortfall is recorded rather than borrowed from another
panel.  Within a parent, coordinate ties use
`(u.q,u.r,v.q,v.r)`; overlapping candidate classes use the order printed
below and are de-duplicated while retaining every class flag and both P3
first-action aliases.  "First" and "last" below always refer to this
coordinate order; an extremal metric uses it as its tie-breaker.  A two-cell
set is internally coordinate-sorted, and the key for a tuple of sets/triples
is the concatenation of those canonical coordinate keys in the role order
specified below.

Before any oracle call, write the complete parent/candidate manifest,
including the source SHA, seed, selection code commit, and explicit directed-
fixture replays; then compute a manifest SHA-256.  Result rows must bind that
manifest SHA.  No oracle outcome may change or refill the sealed manifest.

1. **K1 fixed-hit/spare corpus panel (64 prefixes).** Defender
   `FirstStone`, `mhs=1`, at least one `h in H(P)`, and at least two spares.
   For the lexicographically first `h`, take the coordinate-first spare
   from each nonempty class as evaluated at the second-stone state `P+h`:
   dead; pure
   quiet (touches no D-alive window); G3
   counter-threat (creates D count at least four); minimum new-support;
   maximum new-support; and newly legalized after `h`.  The minimum/maximum
   classes first optimize support delta and then use the coordinate tie-break.
2. **K1 split-cover panel (64 prefixes).** Require at least one covering
   pair with neither member in `H(P)`.  Enumerate every comparison tuple
   `(N,M)`, where `N` is a P3-quotiented split cover and `M` is a
   P3-quotiented H-containing cover.  Its four-bit signature records whether
   the cells of `N` share an initial T-window, whether `N` creates a G3
   counter-threat, whether `N` is pure quiet, and whether
   `Lambda(C_N)=Lambda(C_M)`.  (G3 and purity are independent bits, not an
   assumed dichotomy.)  Retain the coordinate-first tuple from every
   nonempty signature and add both linked candidate rows to the manifest.
   These sampled
   comparisons do **not** adjudicate B2-K1-CONTAIN-INDEXED or UNIFORM; only
   the full audit in item 7 can do that.
3. **K1 multi-full-coverer/counterfork panel (32 prefixes).** Require at
   least two members of `H(P)` and a G3 or frontier-active spare common to two
   hits.  Enumerate triples `(h_i,h_j,s)` with `h_i<h_j`,
   `s in (S_{h_i} intersection S_{h_j})\{h_i,h_j}`, and with `(h_i,s)` or
   `(h_j,s)` G3-producing or frontier-active.  The four-bit signature is G3
   after `h_i`, G3 after `h_j`, positive support delta after `h_i`, and
   positive support delta after `h_j`.  Retain the coordinate-first and, when
   distinct, coordinate-last triple in every nonempty signature; for each
   retained triple add the linked rows `(h_i,s)` and `(h_j,s)`.  This attacks
   both hit identity and spare identity without assuming the coverers are
   interchangeable.
4. **K2 multi-hitting-set panel (64 prefixes).** Require `mhs=2` and at
   least two distinct minimum hitting sets.  Enumerate every unordered
   comparison `(M,N)` of two such sets.  Its three-bit signature records
   whether the sets share one cell (otherwise they are disjoint), whether
   their sorted multisets of per-cell labelled T-window coverage sets agree,
   and whether a shared-cell substitution satisfies P2 at the corresponding
   second-stone state.  Retain the coordinate-first comparison in every
   nonempty signature and add both linked hitting-set rows.  Thus
   shared/disjoint, equal/unequal coverage profile, and P2-protected/
   unprotected cases have a deterministic manifest interpretation.
5. **Uncovered deadline controls (32 parents).** From the result-blind union
   of panels 1--4, first de-duplicate parents, then score each eligible parent
   with `SHA256(seed|uncovered-control|game-hash|prefix)` and take the lowest
   32 having an uncovered pair.  Per parent select, in priority order with coordinate
   de-duplication: the first old/old pure-quiet uncovered pair; the old/old
   uncovered pair with maximum G3 counter-threat degree; and the first
   directed-new uncovered pair.  Keep up to one of each, record `r(M)=1/2`,
   and check the exact B2-COVER deadline at `d=r(M)`.
6. **Directed adversaries and controls.**
   - Lift each of the four doubly-proven b=1 coverer witnesses to its preceding
     `FirstStone` parent: replay prefix `p-1`, use the actual `p`th corpus
     placement as the common first stone, and compare the two recorded cells
     as second stones.  Retain only parents satisfying this section's b=2
     predicate.
   - Include the existing reachable `spare_tempo_fixture`: at its b=2,
     `mhs=1` root, `(4,4)` followed by `(1,-3)` is the frozen G3 tempo line.
   - Include the b=2 defender-`FirstStone` child of
     `compact_urgent_spare` after attacker `(3,0)`, and the analogous
     `double_fork_compact` child after attacker `(4,0)`, only as shallow
     stress cases.  The latter's 478-cell wall makes it unsuitable for the
     bulk deep panel.
   - Private sibling-test helpers are not an experiment interface.  Copy the
     explicit legal replay coordinates and expected child action into the
     sealed manifest so the domination harness reconstructs each fixture
     independently.
   - For four named bases--the spare-tempo line, canonical lifted coverer
     witness #1, the first K1 audit node, and the first K2 audit node--replay
     all 12 D6 images at `d=1,2` with `d6_canonical=false`.  These are
     metamorphic controls, not independent samples.  Also solve both P3
     orders where applicable.
7. **Universal-quantifier audit.** From the sealed panel union, take the first
   four K1 and first four K2 parents sorted by
   `(legal_move_count,game_hash,prefix)`, then enumerate every legal completed covering
   macromove, add the analytically known B2-COVER value for every uncovered
   first/second alias, and compare the entire status cross-product at
   `d=1,2`.  A
   deeper depth joins the UNIFORM tested set only if **every** candidate at
   that node completes there.  Give this audit its own two-CPU-hour aggregate
   budget; if a node/depth does not complete, it supports no universal
   wording.  Only completed audit nodes may support a bounded result worded
   “all spares” or “any hitting set.”  The K1 audit computes every `F_d` and
   the complete H-containing maximum/intersection, thereby adjudicating the
   frozen FIRST-DISMISS-INDEXED and both CONTAIN variants.

Each JSONL row records the manifest SHA, oracle/source commit, replay
identity, absolute placement count, exact parent phase/player, both
placements and their legality-at-turn-start bits, initial threat keys and
masks, coverage set, deadness, frontier inertness, successor-support delta,
all incident D/A masks, counter-threat degree, P2 protection, exact depth,
P3 canonical successor plus all ordered first-action aliases, oracle
configuration, completeness, status, nodes, TT statistics, child legal
width, and wall time.  That payload is what lets a mismatch sharpen the next hypothesis
instead of becoming an anecdote.

### 7.5 Depth ladder and stopping rules

- **`d=1` bulk:** immediate A-completion distinctions after the completed
  defense.  This is cheap and prevents a shallow mismatch from disappearing
  at `d=2`.
- **`d=2` bulk:** exactly one A turn after the completed defense.  Run every
  selected child and every B2-COVER oracle-control case.
- **`d=3` tactical:** all G3/counterfork cases.  This reaches D's first reply
  after A's turn and can turn an unanswerable counterfork into exact attacker
  `Loss`.
- **`d=4` selective:** cases that completed `d=3` cheaply, plus every frozen
  mismatch and every surviving conjecture; this reaches the end of D's next
  turn.
- **`d=5,6` targeted:** only unresolved conjecture survivors and frozen
  counterexamples.  Depth 5 includes A's first placement of its following
  turn; depth 6 reaches the end of that turn.

Depth zero is trivial for the nonterminal completed children (all are exact
`Unknown`) and terminal turns were already recorded.  A result described as
“through depth 6” requires the contiguous `d=1..6` matrix; a shard that skips
a depth may report only the explicit set it ran.

Stop deeper work for a universal equivalence or direction immediately after
one exact refutation.  Rerun every mismatch with D6 canonicalization disabled
and with a second TT size.  A completed exact status is configuration-
independent; only cost/completeness may change.

### 7.6 Oracle qualification and execution discipline

The existing fast-reference differential is 209/209, but its status coverage
was 42 `Win`, 167 `Unknown`, and **zero `Loss`**.  Because G3 adjudication may
depend on attacker-`Loss`, qualification phase Q0 scans the sealed corpus in
`(player_index, phase, game_hash, prefix, d)` order at `d=1..4` with the stock
reference and freezes the first four tractable `Loss` rows (stock nodes at
most one million) in each player-identity x `FirstStone`/`SecondStone`
bucket.  The resulting 16-row qualification manifest stores the full replay,
exact `d`, stock status/nodes, corpus SHA, and its own SHA.  A shortfall blocks
the experiment.  Q0 also uses the nine-minute incomplete result described
below, one row per invocation; an incomplete stock solve is skipped but
recorded and cannot become a qualification row.  Only after that manifest is sealed does fast reference run
the same 16 rows; all must return `Loss`.  This intentionally uses stock
outcomes to build an oracle-qualification set, but neither fast results nor
b=2 candidate outcomes influence the frozen rows.  No b=2 result using
`Loss` is admitted before this gate is green.

The current reference implementations have no work-limit result suitable for
this campaign.  The future test-only driver therefore implements the same
propagating recurrence independently around both the stock reference used by
Q0 and `tss_reference_fast` used by Q0/candidate solves, with result type
`Complete(ProofStatus) | Incomplete`.  At a maximizing node:
return complete `Win` if any completed child is `Win`; otherwise return
`Incomplete` if any child is incomplete; otherwise perform the ordinary
`Unknown/Loss` backup.  At a minimizing node, dually return complete `Loss`
if any completed child is `Loss`, then `Incomplete` if any child is
incomplete, otherwise the ordinary `Unknown/Win` backup.  Never cache
`Incomplete`.  A nine-minute cooperative deadline may produce only
`INCOMPLETE`; it must never manufacture `Unknown`.  Flush one bound result
row after each case so interruption retains prior exact work.

Use release mode, one cargo process, `--test-threads=1`, a 512 MiB TT per
solve, primary `d6_canonical=true`, `ordering_hint=None`, and the free-RAM
gate above 8 GiB.  The driver must be an in-crate `#[cfg(test)]` unit test:
the fast-reference API is `pub(crate)` and test-only.  It explicitly parses
`TSS_REFERENCE_FAST_TT_BYTES` and `TSS_REFERENCE_FAST_D6` into
`FastReferenceConfig`; the oracle itself does not read those variables.
Pre-build once, then run exactly one manifest case and one depth per
invocation with the internal 540-second deadline.

Prospective command shape (the named test does **not** exist at this HEAD):

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 8) { throw "Need >8 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_MANIFEST='E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-domination/dom_b2_exact_manifest.jsonl'
$env:TSS_DOM_B2_RESULTS='E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-domination/dom_b2_exact_results.jsonl'
$env:TSS_DOM_B2_CASE_ID='k1-fixed-0000'
$env:TSS_DOM_B2_DEPTH='2'
$env:TSS_DOM_B2_DEADLINE_MS='540000'
$env:TSS_REFERENCE_FAST_TT_BYTES='536870912'
$env:TSS_REFERENCE_FAST_D6='1'
cargo test -p hexfield_eq --release dom_hunt_b2_exact -- --ignored --nocapture --test-threads=1
```

Run `cargo test -p hexfield_eq --release --no-run` as a separate pre-build.
The case driver itself enforces the deadline and appends either a complete
status or an `INCOMPLETE` row bound to the manifest; no `0/N` shard is allowed
to hide an over-ten-minute case.

### 7.7 Estimated cost

The sealed manifest first performs a no-solve dry pass and reports the exact
selected-child count plus `min/median/p95/max` of both
`L_0(M)=|L(C_M)|` and the nonterminal first-child widths
`L_1(M,a)=|L(C_M+a)|`.  This matters because a directed outward spare or A's
first reply can add much of a 217-cell radius-8 ball, so either width can
exceed the historical 300--600 band.  Price the order-independent potential
direct-extension work at `d=2` by

\[
Q_2=\sum_M
    \sum_{\substack{a\in L(C_M)\\ C_M+a\ \text{ nonterminal}}}
       |L(C_M+a)|,
\]

not by parent width or by `L_0(M)^2`.  The dry pass obtains this quantity by
locally applying and undoing each first reply; no oracle result is consulted.
Immediate first-placement wins and search short-circuiting can only reduce
the executed scan from this potential count.  If both observed width
distributions happen to be 300--600, that is roughly `9e4`--`3.6e5`
direct-extension probes per completed-turn child.
`FastReferenceResult.nodes` does **not** count those
leaf probes: because its `plies_left==1` case scans moves without recursing,
the reported fast-node count is only about `1+L`.  The existing
324-legal-cell stock-reference control reported 129,455 recursively counted
nodes at depth 2; it is a work-scale comparison, not a fast-node prediction.
The provisional representative-panel envelope is `Q_2=1e8`--`4e8` leaf
probes, roughly 2--8 serialized CPU-hours after compilation, plus up to two
hours for the full-quantifier audit.  The dry manifest must revise that
estimate before solving if its candidate count or width distribution falls
outside the envelope; no outcome-based down-selection is allowed.  Tactical
`d=3` cases often short-circuit near the same order when a counterfork
converts, but quiet cases can approach cubic work;
budget 2--4 more CPU-hours and mark the rest incomplete.  Cap the aggregate
`d=4..6` targeted stage at eight CPU-hours and reserve up to two hours for Q0
qualification.  Thus 16--24 serialized CPU-hours
is a **campaign ceiling**, not a promised completion estimate; every case is
still a sub-ten-minute invocation and incomplete rows are expected.

This estimate deliberately treats `double_fork_compact` as a warning: one
of its depth-7 478-cell branches was manually stopped after about nine
minutes, and the full depth-9 wall did not close.  It is a shallow stress
fixture, not evidence that the bulk deep ladder is affordable.

## 8. Regeneration commands

No Rust test or harness was added or changed in this round, and no cargo grind
was run.  The two proofs are analytic.  The commands below regenerate only
pre-existing evidence cited here; check RAM first and keep one process.
The pre-existing untracked `.codex-proof/` and `.target-hunt/` directories
were left untouched and are not deliverables of this round.

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 8) { throw "Need >8 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
cargo test -p hexfield_eq --release dom_hunt_selftest -- --ignored --nocapture --test-threads=1
```

Canonical doubly-proven coverer boundary:

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 8) { throw "Need >8 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_VERIFY_HASH='d7e1b56c925b7f32'
$env:TSS_DOM_VERIFY_PREFIX='20'
$env:TSS_DOM_VERIFY_A='-2,3'
$env:TSS_DOM_VERIFY_B='-1,2'
cargo test -p hexfield_eq --release dom_hunt_verify -- --ignored --nocapture --test-threads=1
```

Existing b=2 tempo-boundary regression:

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 8) { throw "Need >8 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
cargo test -p hexfield_eq --release spare_stone_counter_threat_cannot_be_implicitly_dispatched -- --nocapture --test-threads=1
```

The historical main sweep command is already recorded in
`HUNT_REPORT_DOMINATION.md` Section 5.2.  Its measured wall time was 902.9 s,
so it was not rerun under this round's ten-minute-per-run rule.

## 9. Attack surface

1. **Imported machinery is load-bearing.** L-DRQ is a two-line corollary only
   because `DOMINATION.md` Lemma 7 proves the full radius-8 inclusion and P1
   proves the all-depth occupancy/mask/frontier transposition.  This document
   intentionally does not re-prove them.  A reviewer disputing either import
   attacks the upstream proven file, not a hidden DRQ persistence assumption.
2. **The low-horizon dispatch strengthening is the most dispensable step.**
   The operational core is equation (1) for `n>=2`.  The `n=0,1` extension
   additionally uses that `T_A(P)` is the *complete* count-4/5 family and that
   a full-coverer blocks every pre-existing count-5.  Dropping those two
   paragraphs still proves the hunt's requested `n>=3` theorem.
3. **Global no-win is stronger than logically necessary.** The sharp
   per-pair premise is `P+c` nonterminal.  `!own_win_now(P)` is retained
   because it certifies that premise simultaneously for the entire omitted
   complement and matches the shipped gate.  The explicit reachable
   counterexample in Section 3.4 shows some terminal guard is necessary.
4. **No coverer collapse follows.** The “any full-coverer” quantifier is a
   value-floor comparison against the non-coverer.  It supplies no inequality
   between two full-coverers.  The four doubly-proven corpus witnesses remain
   direct blockers to such a use.
5. **The empirical headline was broader than its loop.** The 20,495 hunt
   comparisons used one canonical full-coverer and only G3 counter-threat
   non-coverers; DRQ used one narrow-referee pair per node.  The status table
   relies on pencil proofs, not on treating either sample as exhaustive
   `VERIFIED` evidence.
6. **Shipped-behavior scope is b=1 only — enforceable fence.** `implicit_dispatch`
   also has a b=2 `mhs=b=2` path.  L-DISPATCH-B1 certifies the b=1
   intersection kernel, not the entire production predicate.  Section 7
   primarily studies the different b=2 spare case `mhs=1<b=2`.  A consumer
   MUST branch on the computed `mhs`/budget pair and take the certified
   prune only on the exact b=1 branch; wiring the prune upstream of that
   branch (where a b=2 case could reach it) is outside every theorem in
   this document.
7. **DRQ is rule equivalence, not representation equivalence.** Ordered
   history, learned features, cache keys, and certificate serialization may
   distinguish the two successors.  A consumer may prune a formal-game move
   only at a layer where P1's strategy transfer is an accepted certificate;
   it may not merge arbitrary byte states.
8. **Coordinate carrier qualification — enforceable at the boundary.** The
   pencil rules use `Z^2`.  The Rust corollaries have the ordinary
   no-`i16`-overflow precondition already stated in `DOMINATION.md`'s
   source concordance.  A production consumer enforces it with the same
   cheap check the DTW consumer contract names (all candidate cells and
   their six-window spans within `i16` range of the occupied bounding
   box); reject-not-certify on failure.
9. **The future oracle's weakest assurance is `Loss`.** Its 209-case
   differential had no `Loss` coverage.  Section 7 therefore gates all
   attacker-`Loss` conclusions on new stock-reference differentials and
   separates exact `Unknown` from interrupted work.

Subject to these named boundaries, no proof obligation remains open for
either round-1 target.
