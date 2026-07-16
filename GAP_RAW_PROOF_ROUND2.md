# GAP-RAW Proof Round 2 — Repair Round

**Worktree:** `hunt-gap-raw` at input commit `159c75f4`  
**Date:** 2026-07-16  
**Disposition:** REPAIR ROUND COMPLETE — GAP-RAW remains **OPEN**.  
**Prime directive:** statement fidelity. GAP-RAW remains **OPEN** unless every quantifier in the normative blanket-game claim is discharged.

## 0. Executive result and target

The round-1 reduction to `W3′+O1′` is **WITHDRAWN**. O1′ is false on the exact
three-gadget history in §4.2; L8.3.2's two-placement price is false on both the
same-axis and mixed cross-axis configurations in §§4.3–4.4; K3 never reduced
to pairwise overlap; and the old Theorem D did not bind one strategy's choices.

What survives or strengthens:

- The normative blanket domain and the actual-action quantifier are repaired.
  Theorems A₂/A₂′ are exact **PROVEN** reformulations.
- L5b and K1 survive. K1 strengthens to *every* two-cell cover being unripe.
- A complete one-axis lemma proves the former `0.4147` witness-floor
  conjecture, and the mass/inventory classification closes K2 completely.
- Exact path/cycle critical-core attacks are handled or excluded. The general
  matching-number-two K3 family—including pairwise, branched/nonclique, and
  chorded cases—remains **OPEN**, narrowed to at most eight labels with the
  first abstract obstruction needing at least five cores.
- A new complete normalized machine universe verifies the global seven-stone
  fork floor L2, including count-five singleton residuals. Global L3 maxima do
  **not** survive: only edge-connected and bounded-region scopes are VERIFIED.

The coherent successor is one strategy-reachable obligation J (**OPEN**) that
binds the actual service/cover/spare pair, the unripe handoff, and renewal of a
precise canonical graded account `B₂=Θ₂`, whose exact updates include
`n₁(c)/9`. Theorem D₂ **PROVES** `J⇒GAP-RAW`; no strategy witnessing J is
supplied. This is honest shrinkage: the conditional statement is sound, the
finite kernel is larger, and the infinite-horizon residue is sharper, but
GAP-RAW itself is not proved.

## 1. Normative domain, notation, and legal play

### 1.1 Blanket-game state space

The board is the infinite hexagonal lattice in axial coordinates. Its three
unoriented window axes have unit vectors `(1,0)`, `(0,1)`, and `(1,-1)`. A
*window* is six consecutive cells on one axis. Hex distance is

`d((q,r),(q′,r′)) = max(|q-q′|, |r-r′|, |(q+r)-(q′+r′)|)`.

A game position is a quadruple `P=(A,D,s,p)`, where `A` and `D` are **finite,
disjoint** sets of Attacker and Defender stones, `A∪D` is **nonempty**, `s` is
the side to move, and `p∈{FirstStone,SecondStone}` is the placement phase of
that side's two-placement turn. A placement is legal exactly when its cell is
empty and is within distance eight of some already occupied cell. After a
`FirstStone` placement the same side moves at `SecondStone`; after a
`SecondStone` placement the other side moves at `FirstStone`.

The actual engine's exceptional singleton empty-board opening is outside this
ordinary nonempty-root cycle. It is used only to replay the reachability history
in §4.2; every GAP-RAW root and every two-placement service theorem below uses
the explicit phase just defined.

This is the conservative blanket Maker–Breaker game. Only Attacker completion
of a window is terminal. A Defender six is ignored: Defender stones block but
do not create a second winning condition. A position is *nonterminal* exactly
when Attacker has no completed window. Thus every theorem below concerns the
same blanket semantics as the normative claim, not the actual-engine
Maker–Maker rule.

A *Defender epoch* is a finite, nonempty, nonterminal
Defender-`FirstStone` position. An *Attacker handoff* is the nonterminal
Attacker-`FirstStone` position after Defender's ordered two-cell reply. A pure
Defender strategy maps every finite nonterminal history whose current position
has Defender to place—at either `FirstStone` or `SecondStone`—to a legal cell.
Its *actual ordered reply* at an epoch is the two cells it chooses sequentially
on that history. “Reached” always means reached by a finite history consistent
with that one strategy and with some legal Attacker continuation.

### 1.2 Alive windows, residuals, and the two accounts

At a nonterminal position, a window W is *Defender-free* when `W∩D=∅` and
*Attacker-alive* when it is Defender-free and `W∩A≠∅`. Below, “alive” is
shorthand for Attacker-alive; a remote virgin Defender-free window is not
called alive. Put

`count_P(W)=|W∩A|`, `E_P(W)=W∖(A∪D)`, and `e_P(W)=|E_P(W)|=6-count_P(W)`

for Defender-free windows. Nonterminality gives `1≤e_P(W)≤5` for every alive
window. With `λ=√3`, define the normative potential

`Φ(P)=Σ_W λ^{-e_P(W)}`,

where the sum ranges over Attacker-alive windows. It is finite because
each Attacker stone lies in eighteen windows. Define also the round-1 graded
subtotal

`Θ₂(P)=Σ_{W: Attacker-alive, count_P(W)≥2} λ^{-e_P(W)}`.

The imminent family is deliberately defined only on nonterminal positions:

`I(P)={W: W is Attacker-alive and count_P(W)∈{4,5}}`.

For a finite family `F` of Attacker-alive windows with nonempty residuals, its hitting
number is

`τ_P(F)=min{|H|: H⊆⋃_{W∈F}E_P(W) and H∩E_P(W)≠∅ for every W∈F}`,

with `τ_P(∅)=0`. Write `τ(P)=τ_P(I(P))`. Completed windows are not members of
`I`, and no empty-residual convention is hidden in `τ`.

For later account bookkeeping, let `n₁(P,c)` be the number of Attacker-alive count-one
windows whose empty set contains a legal Attacker cell `c`, and let `S₂(P,c)`
be the `Θ₂`-mass of alive count-at-least-two windows through `c`. On a
nonterminal Attacker placement that itself does not complete six,

`ΔΘ₂ ≤ (λ-1)S₂(P,c)+n₁(P,c)/9`.                                      (1)

The second term is the latent count-one promotion capacity omitted by the
round-1 renewal claim. It is retained explicitly in every successor account
obligation below.

### 1.3 Normative target [OPEN]

**GAP-RAW.** For every finite, nonempty, nonterminal blanket-game position
`P₀` with Defender at `FirstStone` and `Φ(P₀)<1`, there exists one Defender
strategy that, against every legal Attacker continuation, prevents Attacker
from ever completing a window.

The quantifier order is `∀P₀ ∃S ∀ attacker continuations`. This document does
not prove that statement; it gives an exact strategy reformulation, retained
local lemmas, adopted counterexamples, and one honest conditional residue.

## 2. Exact strategy reformulation

### 2.1 Completion and service

**Lemma L1.1 (Attacker-`FirstStone` completion criterion) [PROVEN].** At a
finite, nonterminal Attacker-`FirstStone` position `Q`, Attacker has a legal
continuation completing a window during that two-placement turn if and only if
`I(Q)≠∅`.

*Proof.* If `W∈I(Q)`, it has one or two empty cells. Fill them. Each is at
distance at most five from an Attacker stone already in the same six-segment,
so every required placement is legal. Conversely, a window completed using at
most two new Attacker stones had at least four Attacker stones and no Defender
stone before the turn, hence was in `I(Q)`. The `Attacker-FirstStone` and
nonterminal premises are essential: at `SecondStone` only one placement
remains, and an already completed window has no residual cell. ∎

Call a Defender ordered reply `(x₁,x₂)` at epoch `P` *servicing* when the two
initially empty cells form a hitting set for `I(P)`; equivalently, every
`W∈I(P)` receives `D@x₁` or `D@x₂` during that actual reply.

**Lemma L1.2 (Defender-epoch service criterion) [PROVEN].** At a Defender epoch
`P`, Defender has a legal servicing ordered reply if and only if `τ(P)≤2`.
Every servicing reply restores `I=∅` at the Attacker handoff. If `τ(P)≥3`,
every legal Defender reply misses some member of `I(P)`, and Attacker can
complete a window on the following turn.

*Proof.* A servicing pair is itself a hitting set, proving necessity. If
`τ(P)≤2`, choose a minimum hitting set. Every selected cell lies in an imminent
window containing at least four Attacker stones, hence is legal by the same
distance-five argument as L1.1. Play those cells first. If fewer than two were
needed, legal spare cells exist: in the current finite nonempty occupied set,
take a cell of maximum `q`; its outward `+(1,0)` neighbour is empty and legal,
and the argument repeats after that placement. Defender placements only kill
alive windows and never increase Attacker counts, so hitting all of `I(P)`
leaves `I=∅`. If `τ(P)≥3`, an actual two-cell reply is not a hitting set. The
missed imminent window remains alive with at most two empties, which Attacker
legally fills by L1.1. ∎

This is an existence statement about servicing replies. It says nothing about
whether a strategy's chosen reply is one of them; that missing action
quantifier was the fatal defect in round 1.

### 2.2 The actual-action theorem

For a root `P₀` and Defender strategy `S`, write `Service(S,P₀)` for the
following single-strategy property:

> Against every legal Attacker continuation, at every Defender epoch `P`
> reached from `P₀` by a history consistent with `S`, the **actual ordered pair**
> selected by `S` is legal and services `I(P)`.

**Theorem A₂ (exact strategy reformulation) [PROVEN].** GAP-RAW holds if and
only if, for every root `P₀` in §1.3, there exists one Defender strategy `S`
such that `Service(S,P₀)`.

*Proof.* Suppose `Service(S,P₀)`. At the root and inductively at every later
Defender epoch, S's actual pair restores `I=∅`; L1.1 then prevents completion
during the entire following Attacker turn. Thus every finite play prefix is
safe, so S blocks forever.

Conversely, let S block forever against every Attacker continuation. If its
actual pair at some reached epoch missed `W∈I(P)`, extend that same history by
having Attacker fill the one or two residual cells of W. L1.1 makes this a
legal winning continuation against S, a contradiction. Hence every actual
reply services. The proof fixes one strategy throughout and uses only a finite
bad prefix; neither determinacy nor a König compactness step is involved. ∎

### 2.3 Ripeness form, proved with the same strategy

Let `Q` be a nonterminal Attacker-`FirstStone` position with `I(Q)=∅`. Call Q
*ripe* if some legal ordered Attacker pair `(c₁,c₂)` leads to the next Defender
epoch `P=Q+A@c₁+A@c₂` with `τ(P)≥3`; otherwise Q is *unripe*. Because Q starts
with every alive window at count at most three, the pair cannot itself complete
six, so the next epoch in this definition really exists.

**Theorem A₂′ (perpetual-unripeness form) [PROVEN].** GAP-RAW holds if and only
if, for every root `P₀` in §1.3, there is one Defender strategy S such that,
against every Attacker continuation and at every reached Defender epoch:

1. S's actual legal ordered pair services `I(P)`; and
2. the resulting Attacker handoff is unripe.

*Proof.* For the forward direction, take the forever-blocking S from GAP-RAW.
Theorem A₂ gives item 1. If one of its reached handoffs were ripe, choose the
witnessing Attacker pair. The resulting reached epoch has `τ≥3`, so L1.2 says
no possible actual reply of S services it and Attacker wins next turn. Both
conclusions contradict the same S's forever property; hence item 2.

For the reverse direction, fix the one S asserted. Its servicing reply makes
the immediate next turn safe by L1.1. Unripeness says every legal Attacker pair
returns an epoch with `τ≤2`, where the hypothesis again requires S's actual
reply—not merely some available reply—to service. Induction over reached
epochs prevents every completion. ∎

The repaired A₂/A₂′ statements preserve the normative quantifier order
`∀P₀∃S∀continuations`; they do not exchange an available cover for S's actual
action.

## 3. Local ledger facts retained or repaired

### 3.1 Named-root maturation and the straight-four cover

**Lemma L4a (one-stone pencil bound) [PROVEN].** From any position with at most
one Attacker stone, the next two Attacker placements cannot produce a
count-four window: afterward there are at most three Attacker stones total. ∎

The inherited one-turn enumeration has a narrower scope than round 1 claimed.
It covers exactly these six report roots:

`es_core`, `blocker_2_0`, `blocker_3_0`, `dense_01_10`,
`dense_01_20`, and `dense_01_1m1`.

**Lemma L4b (named-root maturation) [VERIFIED].** On the three one-Attacker
roots in that list, the raw one-turn count-four-or-higher maximum is zero. On
the adjacent two-Attacker roots `dense_01_10` and `dense_01_1m1`, it is three,
and every maximizer is a straight four. On the gapped root `dense_01_20`, it is
zero. After the report's specified first R1b Defender turn it is zero on all
six roots. This is per-root exhaustive evidence, not a classification of all
`Φ<1` roots and not a theorem about arbitrary two-Attacker positions.

The service step for the straight four is now explicit. Normalize its four
Attacker cells to axis coordinates `{0,1,2,3}`. The only three length-six
windows containing all four have residuals

`{-2,-1}`, `{-1,4}`, and `{4,5}`.

The two cells `{-1,4}` hit all three residuals, so `τ≤2`. This is an actual
two-cell cover; no inference from matching number is used. The finite
regression is recorded in §8.

**Lemma L5a (named-root `ΔΦ` ceilings) [VERIFIED].** The same per-root
enumeration gives maximum one-turn `ΔΦ=4/√3≈2.309` on `es_core`,
`blocker_2_0`, `blocker_3_0`, and `dense_01_20`, and approximately `2.591` on
the two tested adjacent roots `dense_01_10` and `dense_01_1m1`. The first
number is not a universal per-turn ceiling.

### 3.2 General potential growth

**Lemma L5b (general `ΔΦ` bound) [PROVEN].** For one Attacker placement at `c`,
let `S(P,c)` be the current `Φ`-mass of touched alive windows through c. With a
completed label assigned terminal weight one only for this inequality,

`Φ(P+A@c) ≤ Φ(P)+(λ-1)S(P,c)+18λ^{-5} ≤ λΦ(P)+2/√3`.       (2)

Consequently, over a two-placement Attacker turn,

`Φ_after ≤ λ²Φ_before+(λ+1)2/√3`

and hence

`ΔΦ ≤ 2Φ_before+2(1+√3)/√3`.                               (3)

*Proof.* Every previously touched alive window through c is promoted by one
count, multiplying its weight by λ. At most eighteen previously virgin
windows through c enter at weight `λ^{-5}`, for total at most `2/√3`. Windows
not through c are unchanged. Since `S(P,c)≤Φ(P)`, (2) follows; apply it twice
and use `λ²=3` for (3). ∎

This surviving bound shows why raw Φ is not a service budget: a safe turn can
inject more than two units without creating an imminent window.

### 3.3 Legality, service choices, and kill multiplicity

**Lemma L6₂ [PROVEN].** Every empty cell of an Attacker-alive window
is legal for either side: it lies at distance at most five from an Attacker
stone in that window. A Defender placement at c kills every alive window
through c, up to eighteen at once.

The qualifier is necessary. In the round-1 literal counterexample
`A={(0,0)}`, `D={(1,0)}`, the virgin Defender-free window
`{(100,0),…,(105,0)}` has no legal cell. Accordingly, the unqualified L6 is
**WITHDRAWN**.

At an epoch with `τ≤2`, the mandatory service *number* is τ, but the Defender's
discretion is not limited to cells informally called “spares.” Minimum covers
may be nonunique, a nonminimum two-cell servicing pair may be strategically
useful, and order may matter to the policy. The joint obligation in §6
therefore quantifies the actual ordered pair, not “a forced cover plus an
independent spare.”

### 3.4 Ripe-witness structure

Fix an Attacker handoff Q with `I(Q)=∅`, a legal Attacker pair `(c₁,c₂)`, the
next epoch `P=Q+A@c₁+A@c₂`, and `F=I(P)`. Partition

`F₁={W∈F:c₁∈W}` and `F₂={W∈F:c₂∈W and c₁∉W}`.

**Lemma L7.1 (membership) [PROVEN].** Every `W∈F` was alive at Q with count at
most three and contains c₁ or c₂. If W is count five at P, it contains both
placements and was count three at Q. A count-four W containing exactly one
placement was count three at Q; one containing both was count two. ∎

**Lemma L7.2 (decomposition) [PROVEN].**

`τ_P(F)≤τ_P(F₁)+τ_P(F₂)`.                                   (4)

Thus a ripe witness has, after possibly exchanging the pair, either a cluster
with hitting number at least three, or one cluster with hitting number two and
a nonempty other cluster. The latter cluster may itself have hitting number
two or more; it is not licensed to become a “single-window remainder.”

*Proof.* The union of separate hitting sets hits the union. If both cluster
hitting numbers were at most one, (4) would give `τ(F)≤2`.

For the stated “after exchanging” form, if the original `τ(F₁)≤1`, then
`τ(F₂)≥2`. After exchanging the trigger order, the new first cluster contains
the old F₂ and is therefore heavy. If that new first cluster has hitting
number exactly two, its new second cluster is nonempty—otherwise it would be
all of F and would give `τ(F)=2`. This proves the displayed dichotomy for the
asymmetric partition.
∎

Call `(cᵢ,Fᵢ)` a *heavy trigger cluster* when `τ_P(Fᵢ)≥2`.

**Lemma L7.3₂ (per-cluster defusal only) [PROVEN].** A pre-emptive Defender
placement `D@cᵢ` at Q, before the hypothetical Attacker pair, kills every
window of the single cluster Fᵢ because every such window contains cᵢ. No
stronger conclusion follows: it need not kill F₍₃₋ᵢ₎, annihilate the whole
witness, or make the remainder serviceable.

In particular one witness may have two independent heavy halves. The round-1
whole-witness wording is **WITHDRAWN**. Globally, let `𝒞(Q)` contain every
heavy trigger cluster arising from every legal Attacker pair at Q. A Defender
cell x *covers* a member `(c,G)` of `𝒞(Q)` only when killing all windows of G
(or an explicitly proved sufficient subfamily) makes that cluster nonheavy.
Choosing at most the available service/spare cells to cover enough members of
`𝒞(Q)` to destroy every ripe pair is a set-cover problem. Occupying its trigger
c is always one valid way to cover `(c,G)`, but shared window labels do not by
themselves prove a shared covering cell. Section 5 states the unresolved global
problem without pretending L7.3₂ solves it.

**Lemma L7.4 (witness mass floors) [PROVEN].** Give every `W∈F` its pre-pair
weight at Q and sum distinct window labels. Every such window had count at
least two, hence weight at least `1/9`. Therefore every ripe witness has
pre-mass at least `3/9=1/3`. In the branch with a hitting-number-two first
cluster and a nonempty second cluster, at least two first-cluster labels have
weight at least `1/9`, while every second-cluster label contained only c₂ and
was count three, of weight `1/(3√3)`. That branch has pre-mass at least

`2/9+1/(3√3)≈0.4147`.                                       (5)

*Proof.* A family of hitting number k has at least k members. Apply L7.1 and
count distinct labels in the disjoint partition `F=F₁⊎F₂`. ∎

**Refinement R7.4 [PROVEN].** A pure all-count-two collinear cluster of hitting
number three is impossible, so (5), rather than `1/3`, is a stronger universal
ripe floor. No optimality claim is made. Section 5.2 supplies the complete
one-axis proof; §8 supplies an independent hard-asserting finite regression.

### 3.5 Complete fork floor and corrected horizon

**Lemma L2₂ (global fork floor) [VERIFIED].** If a family F of alive
count-four/count-five windows has `τ(F)≥3`, then at least seven Attacker stones
lie inside `⋃F`.

*Complete reduction.* Suppose instead that at most six relevant Attacker stones
lie in `⋃F`, and choose `W∈F`. W contains four or five of them. Every other
`U∈F` contains at least four, so inclusion–exclusion makes U and W share at
least two Attacker cells. Two distinct cells determine one lattice axis;
therefore every member of F lies on W's axis. Normalize W to `[0,5]`. Every U
is then contained in `[-5,10]`; stones off that line occur in no member of F
and can be omitted. Section 8's hard enumeration tries every four-/five-subset
of W plus every choice of the remaining at-most-two cells in that interval,
discards terminal count-six sets, includes both count-four residual pairs and
count-five residual singletons, and checks the full imminent family. All 902
normalized configurations have hitting number at most two, with maximum two.
Defender stones can only delete labels, and a subfamily cannot have larger
hitting number than a family already covered by two cells. This exhausts the
contradiction universe. ∎

This is not the invalid round-1 connected-polyhex bridge; edge-disconnected
co-window configurations are present in the normalized line universe.

**Lemma L7.5 [VERIFIED].** A ripe witness has at least five pre-pair Attacker
stones in its footprint. After adding c₁,c₂ its imminent family has `τ≥3`, so
L2₂ gives at least seven footprint stones; removing the two triggers leaves at
least five earlier stones. ∎

**Theorem C₂ (horizon statement) [VERIFIED].**
Let `P₀` be a finite, nonempty, nonterminal Defender-`FirstStone` position with
`a₀` Attacker stones, and let Defender use a strategy whose actual reply
services every reached epoch whenever `τ≤2`. Put

`t*=min{t≥0:a₀+2t≥7}`.

Attacker completion **can be no earlier than** Attacker turn `t*+1`. In
particular, for `a₀≤2`, completion can be no earlier than future
Attacker placement 7 (possibly placement 8 of that turn). This is a lower
bound only; it does not assert that the bound is attainable.

*Proof.* Before the end of t* Attacker turns, every Defender epoch has fewer
than seven Attacker stones. L2₂ forces `τ≤2`, so the stated
strategy actually services it and L1.1 blocks the next turn. The first epoch
at which L2₂ no longer gives that conclusion is after t* turns, making turn
`t*+1` the first not certified. ∎

The normative boundary's independent five-placement Theorem 2 remains
**PROVEN** at its own scope. Its sharpness example applies only to the specified
fixed-initial-cohort strategy **with the specified filler**; it is not a
sharpness theorem for the whole fixed-cohort strategy class.

## 4. Adopted refutations and graded-account boundary

### 4.1 What the count-two grade does and does not remove

**Lemma L8.1₂ (direct clean-escape calculation) [PROVEN].** The clean-escape
pair in `ES_GLOBAL_BOUNDARY.md` creates thirty-six distinct count-one windows
and therefore injects exactly zero into Θ₂ while injecting `4/√3` into Φ. ∎

This neutralizes only Corollary 2's *direct* source term at the instant of
birth. It does not “neutralize clean escape for graded accounts”: those labels
remain latent and can later cross the grade. The broader round-1 wording is
**WITHDRAWN**.

**Lemma L8.2₂ (local injection and adjacent-pair benchmark) [PROVEN].** For a
nonterminal Attacker placement c, inequality (1) holds. An isolated
axis-adjacent Attacker pair creates exactly five count-two windows, contributing
`5/9`; one outward axis-neighbour Defender chase kills four and leaves `1/9`,
while two such chase cells kill all five.

The `5/9` value is a benchmark for a newly isolated adjacent pair, not a
universal remote-minting ceiling. Stored count-one windows contribute the
separate `n₁(P,c)/9` term and can inject more than `5/9` in one placement.

### 4.2 Three-gadget refutation of O1′ [WITHDRAWN]

The round-1 obligation O1′ asserted renewal from **every** `Θ₂<1` position.
Adopt the hostile review's exact counterexample. Let

`c₀=(6,-3)`, `c₁=(18,-3)`, and `c₂=(30,-3)`.

For each `cᵢ=(q,r)`, put Attacker supports at `(q+1,r)` and `(q,r-1)`. The
position occurs after this legal, nonterminal history (the first Defender
placement is the engine's singleton opening; later turns use pairs):

```text
D (0,0)
A (7,-3), (6,-4)
D (12,0), (13,1)
A (19,-3), (18,-4)
D (24,0), (25,1)
A (31,-3), (30,-4)
```

At the resulting Defender epoch, no two Attacker stones share an axis window.
The exact alive profile is 106 count-one windows and no window of count at
least two. Hence

`Θ₂=0` and `I=∅`.                                           (6)

Its raw potential is `Φ=106λ^{-5}=106/(9√3)>1`. Thus this position lies in
O1′'s universal `Θ₂<1` domain but not in GAP-RAW's original `Φ<1` root domain.

At a center `cᵢ`, five Q-axis windows contain both cᵢ and `(q+1,r)`, and five
R-axis windows contain both cᵢ and `(q,r-1)`. These ten labels are distinct.
The unions of the ten focal windows for the three centers are pairwise
disjoint. Any legal two-cell Defender reply therefore meets the focal union of
at most two gadgets. Choose an untouched center and play `A@cᵢ`, which is legal
because it is adjacent to both supports. All ten surviving focal labels move
from count one to count two, so immediately after that first Attacker placement

`Θ₂≥10·(1/9)=10/9>1`.                                      (7)

There is no completion; a legal second Attacker placement exists and cannot
decrease Θ₂, so the violating next Defender epoch exists. The regression in §8
replays the history and quotient-exhausts every legal two-cell defense for the
focal objective: cells inside the three focal unions are enumerated literally,
while an outside cell kills no focal label. It then chooses a best untouched
response and hard-asserts (6)–(7).

Thus no strategy can maintain `Θ₂<1` from every `Θ₂<1` start. The failure is
exactly the latent entry charge `n₁(cᵢ)/9=10/9`. O1′ and every theorem using it
at that domain are **WITHDRAWN**.

The minimum plausible successor is strategy-reachable: begin at an original
root satisfying `Φ<1`, fix one named Defender strategy S, and quantify only
histories reachable under S against arbitrary Attacker continuations. The
three-gadget state may then be excluded only because S previously prevented
its stock, not because Θ₂ forgot the stock. No such renewal theorem is proved
this round; its precise joint form is obligation J in §6.

### 4.3 Same-axis one-placement heavy formation [VERIFIED]

Use one Q-axis with coordinates written as integers. Before a past placement
p, put local Attacker stones at `{-2,-1}`; then play `p=1`. Let the prospective
future trigger be `c=0`. The three windows starting at `-4,-3,-2` were count
two before p, are count three after p, and after the future `A@c` have residual
pairs

`{-4,-3}`, `{-3,2}`, and `{2,3}`.                          (8)

Their hitting number is two. Add a remote count-three Q-window with Attacker
stock `(100,20),(102,20),(105,20)` and take the second future trigger
`d=(101,20)`. Before `(c,d)`, every alive window has count at most three. After
the pair, the remote window supplies one residual pair disjoint from (8), so
the four-window family has hitting number three. Removing the single past
placement p lowers the three local labels from count three to count two. The
remote count-three label is pre-existing; it embeds the one-placement local
heavy formation in a full ripe T2 witness.

One past promotion therefore creates a same-axis heavy cluster inside a full
ripe witness. The round-1 assertion that the collinear T1 analysis excluded
such formation confused “heavy” (`τ≥2`) with the narrower T1 condition
(`τ≥3`). The exact finite regression is in §8.

### 4.4 Mixed cross-axis one-placement formation [VERIFIED]

Let the pre-pair Attacker set be

`Q.A={(-4,0),(-3,0),(0,1),(0,2),(0,3)}`

and use future triggers `c=(0,0)`, `d=(1,0)`. Q satisfies `I=∅`. The horizontal
window `(-4,0)…(1,0)` is count two and contains both future triggers. Three
vertical windows through c are count three. After `(c,d)`, the four residual
pairs have hitting number three.

Now remove the single past placement `p=(0,3)`. Every one of those four labels
is then count at most two. Thus one past placement creates this cross-axis
witness: the horizontal label stays on the mixed pre-count-two branch and
jumps by two, while the vertical labels are pre-count-three. L7.1 expressly
allows that branch; L8.3.2's proof omitted it. The exact finite regression is
in §8.

The omitted branch must be kept separate from the restricted
pre-count-three/pre-count-three case. If two windows on different axes through
one still-empty future trigger c are both count three at Q, but each was count
at most two at a named earlier baseline, then their necessary promotions since
that baseline occurred at two distinct past cells: different-axis windows meet
only at c, which has not yet been played. That narrow observation is valid. It
does not price the mixed branch above, where the horizontal pre-count-two label
contains both future triggers and jumps directly to count four.

### 4.5 What remains of reactivation pricing

**L8.3.1 (two-trigger locality) [PROVEN].** Every imminent label created by a
witness pair passes through at least one of the two trigger cells and lies
within its length-six window footprint. This is L7.1, not a one-center bounded-
radius theorem: the two legal triggers may be arbitrarily far apart because
each can be legal near old stock.

**L8.3.3 (per-cell killing) [PROVEN].** A pre-emptive Defender stone at a
prospective trigger, before the hypothetical Attacker pair, kills all labels
in that trigger's one cluster, exactly as L7.3₂ states.

**L8.3.2 (two-past-placement price) [WITHDRAWN].** Sections 4.3–4.4 refute both
the same-axis exclusion and the useful cross-axis claim. No positive universal
two-placement reactivation price is retained.

Any replacement history charge must name its interval. For a proposed cluster
of persistent window labels G at an Attacker handoff Q, a *pre-trigger charge
interval* must begin at a specified earlier history index `t₀` where each
charged label is alive and its stated baseline count is measured, and must end
at Q, **before** the hypothetical future witness pair. Placements of that
future pair cannot pay for a Defender action that must pre-empt the pair. The
two regressions show that, even when all charged labels were count at most two
at `t₀`, this interval may contain only one relevant Attacker placement. Which
labels persist, how Defender kills earn refunds, and whether reused stones can
be charged again are all part of a still-**OPEN** attempt to weaken the
conservative `B₂=Θ₂` renewal demanded by J; no such credit system is used here.

## 5. Budget kernel and witness-suppression geometry

Throughout this section P is a Defender epoch with `Θ₂(P)<1`. Every window
label discussed in a future ripe witness is already alive at P with count two
or three; Defender placements may delete such labels but do not change their
counts.

### 5.1 Fully forced service: K1 survives and strengthens

**K1 (`τ(P)=2`) [PROVEN].** Every legal two-cell cover of `I(P)` hands Attacker
an unripe position.

*Proof.* Hitting number two implies at least two distinct imminent labels, each
of weight at least `1/3`, so `mass(I(P))≥2/3`. Fix any two-cell cover and let Q
be its handoff. If Q had a ripe witness F, its distinct pre-pair labels were
alive at P with count at most three and hence label-disjoint from `I(P)`. L7.4
gives them mass at least `1/3`. Thus `Θ₂(P)≥2/3+1/3=1`, contradiction. The
argument did not select a special cover. ∎

K1 establishes unripeness, not account compatibility: J still requires the
same cover to satisfy its renewal transition.

### 5.2 A complete one-axis lemma

**Lemma R7.4₂ (no all-count-two ripe core) [PROVEN].**
At a handoff with `I=∅`, no one- or two-cell trigger set—even when legality is
dropped for a finite over-approximation—can turn a family of pre-count-two
labels alone into an imminent residual family of hitting number three. One
trigger cannot promote count two to count four. For two triggers at separation
`d=1,2,3,4,5` on their common axis, the exact maximum hitting numbers are
respectively

`2, 2, 2, 1, 1`.                                           (10)

*Proof.* Every pre-count-two label that becomes count at least four must
contain both trigger cells. Two distinct cells sharing a window determine one
axis; normalize them to `0,d`, `1≤d≤5`. If any cell strictly between the
triggers is empty, it remains in the residual of every qualifying window, so
the hitting number is at most one. Suppose instead all internal cells are
occupied. For `d≥4`, there are already at least three old stones, so no window
is pre-count two. For `d=3`, the residuals are a subfamily of

`{-2,-1}`, `{-1,4}`, `{4,5}`,

covered by `{-1,4}`.

For `d=2`, remove the one occupied internal cell. In their natural left-to-
right flank order, the four possible residual pairs are the two zero positions
in those consecutive length-three factors of a six-bit word that have exactly
two zeros. For `d=1` the analogous objects come from the consecutive
length-four factors with exactly two zeros in an eight-bit flank word. After
duplicate pairs are removed, the residual-pair graph is a forest: at the leftmost
vertex of a cycle, its two right neighbours would both lie in the interval
witnessing the farther edge, creating a forbidden third zero in that factor.
It also has no matching of size three. For length three, three pairwise
disjoint witnessing factors would require three starts pairwise at least two
apart inside `{0,1,2,3}`, impossible. For length four the only candidate starts
are `{0,2,4}`, but then the middle factor is all ones. By König's theorem for a
forest, the residual graph has a vertex cover of size at most two. Direct
examples attain the first three entries of (10); deletion by blockers cannot
raise a hitting number. ∎

Consequently the former R7.4 strengthening is no longer conjectural: every ripe
witness contains a pre-count-three label, and a universal pre-mass floor is

`2/9+1/(3√3)≈0.4147`.                                      (11)

The weaker `1/3` floor remains sufficient for K1 and K3's matching bound.

### 5.3 One forced cover plus one spare: K2 closes

**K2 (`τ(P)=1`) [PROVEN].** For every chosen one-cell cover x of `I(P)`, there
is a legal second Defender cell y such that the actual reply `(x,y)` hands an
unripe position.

*Proof.* Let `R=P+D@x`, and let V be the finite union of the empty cells of all
R-alive count-two/count-three labels. Before selecting y, enumerate every
trigger set `T⊆V` with `1≤|T|≤2`, without imposing a legality condition. Put

`G_T={W:count_R(W)∈{2,3} and count_R(W)+|T∩W|≥4}`,

with residuals `E_R(W)∖T`. A *critical core* is an inclusion-minimal
`C⊆G_T` whose residual family has hitting number at least three. Because this
universe deliberately includes illegal trigger sets, killing at least one
label in every critical core is a sufficient suppression condition, not a
necessary one. It becomes equivalent after restricting to trigger sets legal
after y. The sufficient direction used here is that any actual surviving ripe
witness contains a surviving minimal core.

This finite pre-y universe over-approximates every attack pair legal after any
choice of y. Discard from an actual pair any placement irrelevant to its
imminent labels; its remaining one- or two-cell trigger set is in V. Thus
color-blind legality expansion by D@y cannot create an omitted core. Defender
y can only occupy a trigger or delete labels.

Let U be the union of the distinct labels in all critical cores, and write h
for its count-three labels and l for its count-two labels. U is label-disjoint
from `I(P)`. Since `mass(I(P))≥1/3`,

`h/(3√3)+l/9 < 2/3`, or equivalently `h√3+l<6`.             (12)

Lemma R7.4₂ says every core has at least three labels and contains a
count-three label. If there is no critical core, choose any legal second filler
by the finite max-q construction in L1.2; the over-approximation already
certifies unripeness. Hence assume a core exists. Then `h≥1`, and the only
integer inventories allowed by (12) are

- `h=1,l≤4`;
- `h=2,l≤2`; and
- `h=3,l=0`.

(The arithmetically possible `h=0,l≤5` inventory has no core and belongs to
the filler case.)

With one high label, it lies in every core. With `(h,l)=(3,0)`, or with two
highs and at most one low, every at-least-three-label core has a common label.
Place y in an empty of that label; minimality says every hit core drops below
hitting number three.

The only exceptional inventory is `h=l=2`, with labels
`H₁,H₂,L₁,L₂`. If all cores share a label, do the same. Otherwise some core
omits, say, H₁. Having at least three labels, it must be
`{H₂,L₁,L₂}`. Under that core's witnessing pair, both pre-count-two windows
L₁,L₂ contain both empty triggers. Choose one trigger for y. It kills both low
labels. Every at-least-three-label subset of this four-label universe contains
a low label, so y hits every critical core.

In every case y is empty, belongs to a touched alive window, is legal by L6₂,
and differs from x because its core labels survived D@x. Count-three labels
were allowed to contain one or both triggers; the inventory argument made no
contrary assumption. Hence `(x,y)` is a servicing, unripe handoff. ∎

This proves more than the review-requested pairwise-intersecting K2 case; no
pairwise-intersection premise is needed.

### 5.4 Free pair and matching-number-two residue: K3 remains open

At `τ(P)=0`, `I(P)=∅` and both Defender cells are available. Let `𝒦(P)` be all
critical cores obtained by repeating §5.3 with `R:=P` before either free
Defender placement: V is the union of the empty cells of all P-alive
count-two/count-three labels, and T ranges over every one-/two-cell subset of
V. By R7.4₂, each core has pre-mass at least the value β in (11). Therefore:

**K3 mass statement [PROVEN].** `𝒦(P)` has no three mutually label-disjoint
members; its matching number is at most two. Indeed
`3β=2/3+1/√3>1`. ∎

Nothing stronger follows from this mass argument. In particular:

- matching number at most two does not make `𝒦(P)` pairwise intersecting;
- a shared label does give a legal action suppressing those two incident
  minimal cores, but matching number at most two does not supply two labels
  hitting the entire core family or a shared trigger action; and
- one witness may have two far-separated heavy halves, so “choose one heavy
  cell per witness” is false even before comparing different witnesses.

There is nevertheless a useful finite residue. If h and l count the distinct
count-three/count-two labels in the union of all cores, then

`h√3+l<9`,

so the only possibilities are `h=0,l≤8`; `h=1,l≤7`; `h=2,l≤5`;
`h=3,l≤3`; `h=4,l≤2`; or `h=5,l=0`. The first has no core by R7.4₂, and in all
other cases the union has at most eight labels.

**Named path/cycle intersection attacks [PROVEN].** Work
with minimal critical cores, and let edges mean label intersection. If the
entire graph is the exact path `A—B—C`, choose one label from `A∩B` and one from
`B∩C`; a Defender cell in each label kills all three cores. For the exact
four-vertex path, use labels on its first and last edges. A path on at least
five vertices has three independent vertices and contradicts the matching-
number-two bound.

Triangles are hit by labels from any two edges; a four-cycle is hit by labels
from two opposite edges. An induced five-cycle is impossible under the mass
budget: a label occurs in at most two cycle cores (otherwise it creates a
chord), so the sum of the five core masses is at most twice the union mass,
hence below two because the union mass is strictly below one. But it is at least

`5β=(10+5√3)/9>2` (`5√3>8`).

Cycles of length at least six have three independent vertices. Thus the
review's pure path/cycle attack is not silently excluded: exact paths and
cycles are either covered by two cells or contradicted by the proved mass
bound. Graphs with chords or additional cores remain part of the general
residue.

Each selected label is alive with count two or three, so L6₂ makes an empty
cell in it a legal pre-emptive Defender action. If two selected labels admit
the same chosen cell, that one placement suppresses both and the second move is
an arbitrary legal filler from L1.2; otherwise play the two cells sequentially.

**Narrow pairwise-intersecting K3 suppression [OPEN].**
The K2 classification does not extend at the larger `<1` inventory. An
abstract obstruction already fits every proved count and mass constraint: five
count-three labels have total mass `5/(3√3)<1`, while the family of all their
three-label subsets is pairwise intersecting and has label-transversal number
three. It is unknown whether length-six/trigger geometry can realize those
subsets as critical cores, or whether two physical cells would kill several
labels at once. Any pairwise family of at most four cores is two-cell
suppressible by pairing cores and choosing one shared label per pair; hence the
first unresolved pairwise case has at least five cores.

A one-center radius-eight enumeration is unsound because the two trigger
neighbourhoods of one legal pair can be arbitrarily far apart near separate
old stock. This round found no geometric counterexample, but supplied no
complete two-neighbourhood classification, so the honest label is OPEN.

The matching-number-two nonclique cases beyond the exact path/cycle families
are also **OPEN**. The minimal schematic attack is

```text
A — B — C,       A∩C=∅,
```

where spending cells on arbitrary actions that suppress the disjoint endpoints
does not imply suppression of B. The edge-label construction above does handle
the *exact* three-core path; the warning remains against the round-1
representative-cell argument and against larger chorded families not reducible
to an exact path/cycle component.

The round-1 count-three-pool sentence, two-representative defusal, and W3′ are
therefore **WITHDRAWN**. The finite kernel now consists of K1 and K2; a finite
sufficient over-approximation to K3 is the suppression-set-cover problem on
`𝒦(P)` with matching number at most two, including both its pairwise and
nonclique branches. Illegal trigger sets may add cores, so this is not claimed
to be a necessary or exact residue.

## 6. One joint strategy obligation J

Round 1 separately quantified a statewise defusal action and an adaptive
renewal rule. That had the invalid form `∃a W(a)` and `∃S O(S)` when the proof
needed S to choose a. The successor below has one strategy witness and only its
own reachable histories.

### 6.1 A precise latent-complete graded account

For a fixed root `P₀` and strategy S, let `Hist(S,P₀)` be all finite,
nonterminal histories consistent with S and with arbitrary legal Attacker
choices. On this history set use the canonical account

`B₂(h)=Θ₂(P_h)`.

This is a fully defined function, not a placeholder for undocumented credits.
For a Defender placement at x, let `κ₂(P,x)` be the `Θ₂`-mass of all alive
count-at-least-two windows through x. The exact nonterminal transitions are

`B₂(h+D@x)-B₂(h) = -κ₂(P_h,x)`,

`B₂(h+A@c)-B₂(h) = (λ-1)S₂(P_h,c)+n₁(P_h,c)/9`.          (9)

The second equality is precisely the latent count-one promotion charge. It
cannot be replaced by the isolated-pair `5/9` benchmark. At a normative root,
`B₂(h₀)=Θ₂(P₀)≤Φ(P₀)<1`.

A more permissive history-sensitive credit system could be future work, but it
is not used as a theorem hypothesis until its ledger variables, earning rule,
consumption rule, and transition inequality are formal. J therefore uses B₂
itself. This is a stronger but precise successor: the open content is renewal
along one strategy's reachable histories, not the existence of a syntactically
described “account.”

### 6.2 The joint obligation

**Obligation J (strategy-reachable joint service, suppression, and renewal)
[OPEN].** For every normative root `P₀` with `Φ(P₀)<1`, exhibit **one** pure
Defender strategy S such that, against every legal Attacker continuation, at
every reached Defender epoch h with position P:

1. **Renewal:** the canonical account satisfies `B₂(h)<1`.
2. **Actual service choice:** S's actual sequential two-cell reply is legal and
   hits every member of `I(P)`. All discretion among alternative covers,
   nonminimum servicing pairs, order, and fillers is resolved inside S's one
   actual sequential pair; no separate existential action is conjoined.
3. **Unripeness handoff:** for the actual handoff Q produced by that same pair,
   every legal ordered Attacker pair returns a Defender epoch with `τ≤2`.
4. **Account transition under the same choice:** Defender's two exact kill
   transitions and every possible Attacker response are evaluated by (9), including
   `n₁/9`, and establish `B₂(h′)<1` at the resulting next Defender epoch h′.

The renewal domain is exactly `Hist(S,P₀)`: it does not include the arbitrary
`Θ₂=0` start in §4.2 unless S itself allows that history from P₀. Conversely,
reachability is not permission to forget dormant labels; equation (9) charges
them when they promote.

J is intentionally stronger than a bare statewise cover lemma and is not
claimed equivalent to GAP-RAW. The account B₂ is explicit, but no strategy S
satisfying all four clauses is constructed here. Its open parts contain both
normative residues:

- `GAP-GLOBAL-RENEWAL`: establish the next `B₂<1` state without demanding
  renewal of raw Φ; and
- `GAP-AMORTIZED-ABANDONMENT`: choose the same service/suppression actions
  while dormant count-one stock is charged on promotion.

The finite suppression work in §5 can be a construction lemma for item 3, but
cannot be conjoined afterward with a different renewal strategy. It must
certify the actual pair chosen by S in items 2 and 4.

## 7. Conditional assembly theorem

**Theorem D₂ (J alone suffices) [PROVEN].** If obligation J holds, then GAP-RAW
holds.

*Proof.* Fix an arbitrary normative root `P₀` with `Φ(P₀)<1` and take the one S
supplied by J. Clause 2 says that, against every Attacker continuation, this
same S's actual ordered pair services every reached Defender epoch. This is
exactly `Service(S,P₀)`, so Theorem A₂ proves that S blocks forever. Since the
root was arbitrary, GAP-RAW follows. ∎

Clauses 1, 3, and 4 are not logically needed once clause 2 is quantified over
all reached epochs; they bind the renewal and suppression invariants that a
construction of that same strategy must certify. This redundancy is explicit:
J is a stronger joint construction obligation, not a claimed smaller
equivalent reformulation.

There is no corollary conjoining W3′ with O1′. Both are **WITHDRAWN** at their
round-1 formulations: W3′ did not cover K3's matching-number-two residue, O1′
is false on §4.2, and their witnesses were not one strategy. D₂ is a sound
conditional theorem with a single **OPEN** hypothesis J, not a proof that J is
true.

## 8. Machine verification and exact scope

### 8.1 Coordinated run

One serial Cargo invocation ran all round-2 ignored checks. Immediately before
launch, the embedded preflight found zero existing Cargo invocations and
`13.796 GiB` free physical RAM. Result:

```text
running 7 tests
test result: ok. 7 passed; 0 failed
test time: 170.19 s
wall time including release compilation: 178.662 s
```

This is below the ten-minute per-run cap. The exact command is in §13.

### 8.2 Mandatory regressions [VERIFIED]

| Test | Hard assertions |
|---|---|
| `round2_o1_prime_three_gadget_refutation` | Engine-legal exact history; Defender-`FirstStone`; profile `(n₁,…,n₆)=(106,0,0,0,0,0)`; `Θ₂=0`; three pairwise-disjoint ten-label focal unions; quotient-complete best two-spare defense; an untouched center gives exact `27Θ₂=30`, i.e. `10/9`, on the first Attacker reply and still `Θ₂≥1` at the next epoch. |
| `round2_l832_same_axis_counterexample` | Exact pre-counts; residuals (8); local hitting number two; focal and full actual imminent families have hitting number at least three after the remote trigger. |
| `round2_l832_cross_axis_counterexample` | Exact mixed pre-count-two/pre-count-three branch; removing `(0,3)` leaves no count-three-or-higher label; four exact residual pairs; focal and full hitting number at least three. |
| `round2_straight_four_explicit_cover` | Exact three residual pairs, explicit `{-1,4}` cover, and `τ=2`; a straight-five companion check includes both count-five singleton residuals. |
| `round2_r74_collinear_all_count2_max_tau` | Every local attacker subset for separations 1–5—respectively 256, 128, 64, 32, and 16 configurations; no connectivity premise; exact maximum vector `[2,2,2,1,1]`. |

The O1′ defense loop is quotient-complete, not a sample: a cell outside all
three focal unions kills no focal label, while a cell in a union is enumerated
literally; pairwise disjointness means two cells meet at most two gadgets.

### 8.3 Complete L2 universe [VERIFIED]

`round2_birth_ledger_geometry_complete_and_scoped` implements the normalization
proved in §3.5. It enumerated 902 nonterminal anchored configurations with at
most six Attacker stones, count-four pairs and count-five singletons, and
hard-asserted `τ≤2` in every case. The maximum was exactly two. This verifies
global L2₂; it does not use edge connectivity or the old superadditivity
bridge.

The regression separately includes the review's edge-disconnected but
co-window-interacting set `{(0,0),(2,0),(3,0),(4,0)}` and asserts that it really
has a count-four window.

### 8.4 L3: largest sound sub-universes only [VERIFIED]

All four numerical columns were hard-asserted over every edge-connected free
polyhex for `n=4..12`:

| n | max count-4+ | max count-3+ | max count-5+ | max pairwise residual-disjoint exact count-4 |
|---:|---:|---:|---:|---:|
| 4 | 3 | 5 | 0 | 2 |
| 5 | 4 | 8 | 2 | 2 |
| 6 | 5 | 12 | 3 | 2 |
| 7 | 6 | 16 | 4 | 4 |
| 8 | 7 | 24 | 5 | 4 |
| 9 | 9 | 28 | 6 | 6 |
| 10 | 10 | 33 | 7 | 6 |
| 11 | 12 | 38 | 8 | 8 |
| 12 | 18 | 41 | 9 | 12 |

These are bare attacker-set incidence statistics: “count-k+” includes any
count-six labels in a terminal set. They are not being substituted for a
nonterminal global theorem.

The generator counts were hard-equal to A000228 through `n=12`. The test also
hard-asserted all four columns and the variable-residual fork predicate over
**every** subset of the unrestricted `[0,6]²` rhombus for `n=4,5,6`—211,876;
1,906,884; and 13,983,816 configurations. Those bounded maxima match the first
three table rows and contain no variable-residual fork.

No global inference is made from either scope. Arbitrary edge-disconnected
configurations outside `[0,6]²`, especially for `n=7..12`, remain unenumerated;
absolute global L3 maxima therefore remain **OPEN**. The old table survives
only with the edge-connected qualifier, and the bounded cross-check survives
only with its explicit region and n-range.

### 8.5 R1b break-epoch audit [VERIFIED]

`round2_trace_r1b_breaks` computed the exact minimum hitting set over the full
variable-size imminent residual family at **every** Defender-`FirstStone` epoch
of each stored replay. It gated the implication `Θ₂≥1` behind a measured
`τ≥3` hard assertion.

For the stored τ=2-policy losses on `es_core`, `blocker_1_-1`, and
`blocker_2_0`, the exact certified break is ply 56 with `τ=3`. In exact
`27Θ₂=A+B√3` notation, the values are `(78,15)`, `(78,15)`, and `(57,39)`.
Earlier rows with three or four imminent windows sometimes had `τ=1` or `2`;
their window count was not misreported as a pileup. The τ=3 policy foiled these
same scripts, which is trace evidence only, not a universal strategy theorem.

The repaired claim is therefore narrow: these three stored R1b losses contain
an actually measured `τ=3` epoch. Loss of an arbitrary fixed policy, or number
of imminent windows alone, never licenses the inference.

### 8.6 Inherited evidence, not rerun here

- L4b/L5a use the six exact maturation roots in §3.1. The prior full maturation
  run was 608.24 seconds, above this round's per-run cap, so it was not rerun.
- The inherited sound pileup minimax through plies 2, 4, and 6 covers exactly
  `es_core`, `blocker_1_-1`, `blocker_2_0`, `blocker_3_0`, `dense_01_10`, and
  `dense_01_20`. It found no forced pileup or six in that completed horizon.
  Eight-ply minimax and ten-ply fixed-R1b runs hit their node caps and support
  no stronger statement.
- The normative boundary's five-placement Theorem 2 and its specified-filler
  sharpness example were not rerun; their proofs, not this harness, establish
  their stated scopes.

## 9. Withdrawn claims ledger

| Round-1 claim | Round-2 label | Reason / adopted boundary |
|---|---|---|
| D0.2's `(A,D)`-only domain and remote-virgin “alive” use | **WITHDRAWN** | Omitted finiteness, nonemptiness, terminal status, blanket semantics, and phase; remote virgin cells need not be legal. Replaced in §1. |
| L1.1 without nonterminal Attacker-`FirstStone` premises | **WITHDRAWN** | False at `SecondStone` for a two-empty window and ill-defined on a completed label. Replaced by L1.1 in §2. |
| L1.2's unquantified “survives iff” and infinite-board spare proof | **WITHDRAWN** | Existence of a servicing reply must be distinguished from the actual reply; finiteness/nonemptiness supply fillers. Replaced by L1.2. |
| Theorem A as originally written | **WITHDRAWN** | Its RHS constrained epoch debt but not S's actual pair. Replaced by A₂/A₂′ in §2. |
| Global L2 verification by edge-connected polyhexes and superadditivity | **WITHDRAWN** | Edge-disconnected co-window interactions invalidate that bridge. The repaired complete `≤6` anchor reduction is separately reported in §8. |
| L3 table as absolute global maxima | **WITHDRAWN** | No complete arbitrary-configuration run for `n=7..12`; only the exact restricted universes in §8 remain VERIFIED. |
| L4 “densest-root” classification and matching-number service inference | **WITHDRAWN** | The report covered six named roots, and matching number is not hitting number. §3 gives the exact scope and explicit `{-1,4}` cover. |
| Universal L5a scope | **WITHDRAWN** | The `2.309/2.591` values are per six named maturation roots only. L5b survives globally. |
| L6 “any empty of any alive window” and “only spares are discretionary” | **WITHDRAWN** | The remote virgin counterexample refutes legality; alternative covers are also choices. Repaired in §3.3. |
| L7.3 whole-witness annihilation / T2 single-window remainder | **WITHDRAWN** | `D@cᵢ` kills only its cluster; a witness can have two heavy halves. §3.4 retains the per-cluster fact. |
| L7.5 and Theorem C as unconditional on the old L2 evidence | **WITHDRAWN** | Their repaired final labels follow the complete L2 check in §8; Theorem C never asserts attainability. |
| “Corollary 2 neutralized for graded accounts” | **WITHDRAWN** | Only the direct clean-escape birth turn has zero Θ₂; latent count-one stock later promotes. |
| Universal remote minting `≤5/9` | **WITHDRAWN** | `5/9` is an isolated adjacent-pair benchmark; §4.2 injects `10/9` from stored count-one labels. |
| L8.3.2 two-past-placement reactivation price | **WITHDRAWN** | One past placement creates the same-axis local heavy cluster and the complete mixed cross-axis witness in §§4.3–4.4. |
| O2′ sharpening based on that price | **WITHDRAWN** | Any weaker history-charge successor must specify its pre-trigger interval and safe refund rule; J conservatively uses `B₂=Θ₂` without such credits. |
| K3 pairwise-overlap reduction, representative defusal, and count-three pool | **WITHDRAWN** | Matching number at most two does not imply a clique, and shared labels do not imply shared trigger actions. §5 states a sufficient over-approximate critical-core problem. |
| W3′ as displayed | **WITHDRAWN** | Its universe and locality were undefined and it did not cover K3. The pairwise critical-core case remains OPEN. |
| O1′ from every `Θ₂<1` start | **WITHDRAWN** | The three separated promotion gadgets force `Θ₂≥10/9` against every Defender pair. |
| R1b loss alone implies `τ≥3` and hence `Θ₂≥1` | **WITHDRAWN** | Policy loss can occur after a missed payable cover. Only exact audited epochs with measured `τ≥3` support that inference (§8). |
| Corollary B′ / old Theorem D from `W3′+O1′` | **WITHDRAWN** | K3 was not closed and the two existential witnesses were not one strategy. Replaced by J and D₂. |
| Pileup/six evidence for all `Φ<1` roots or beyond completed caps | **WITHDRAWN** | The inherited exhaustive claim is only six named roots through six plies; eight/ten-ply runs capped. |
| Theorem 2 sharp for a whole fixed-cohort class | **WITHDRAWN** | Its example is sharp only for the specified fixed-family strategy with its specified filler. |
| Theorem C wording suggesting the lower bound is attained | **WITHDRAWN** | C₂ says completion “can be no earlier than,” with `t*` starting at zero. |

## 10. Authoritative round-2 status table

| Claim | Label | Evidence / exact scope |
|---|---|---|
| GAP-RAW | **OPEN** | Normative target, §§0–1; J is unproved. |
| L1.1 Attacker-`FirstStone` completion criterion | **PROVEN** | §2.1 |
| L1.2 Defender-epoch servicing criterion | **PROVEN** | §2.1 |
| Theorems A₂/A₂′ actual-action reformulations | **PROVEN** | §§2.2–2.3 |
| L4a one-Attacker pencil bound | **PROVEN** | §3.1 |
| L4b one-turn maturation | **VERIFIED** | Six exact maturation roots only, §3.1/§8.6. |
| L5a per-root `ΔΦ` ceilings | **VERIFIED** | Same six exact roots, §3.1/§8.6. |
| L5b general `ΔΦ` bound | **PROVEN** | §3.2 |
| L6₂ touched-window legality and kill multiplicity | **PROVEN** | §3.3 |
| L7.1 membership and L7.2 decomposition | **PROVEN** | §3.4 |
| L7.3₂ per-cluster defusal | **PROVEN** | §3.4; whole-witness version withdrawn. |
| L7.4 original mass floors | **PROVEN** | §3.4 |
| R7.4₂ `β≈0.4147` floor | **PROVEN** | Pencil proof §5.2; regression §8.2; no optimality claim. |
| L2₂ global seven-stone fork floor | **VERIFIED** | Complete 902-configuration reduction, §§3.5/8.3. |
| L3 four columns, edge-connected | **VERIFIED** | Free polyhexes, `n=4..12`, §8.4. |
| L3 four columns, unrestricted bounded region | **VERIFIED** | Every `[0,6]²` subset, `n=4..6`, §8.4. |
| L3 absolute global maxima | **OPEN** | Arbitrary configurations outside those scopes remain. |
| L7.5 five-prestone witness floor | **VERIFIED** | Deduction from L2₂, §3.5. |
| Theorem C₂ lower horizon | **VERIFIED** | Deduction from machine-VERIFIED L2₂; no attainability claim, §3.5. |
| Normative-boundary Theorem 2 | **PROVEN** | Five-placement scope only; sharpness only for its specified strategy/filler, §3.5/§8.6. |
| L8.1₂ direct clean-escape zero injection | **PROVEN** | §4.1 |
| L8.2₂ local `Θ₂` update and adjacent benchmark | **PROVEN** | §4.1 |
| O1′ every-`Θ₂<1` renewal | **WITHDRAWN** | Three-gadget `10/9` refutation, §§4.2/8.2. |
| Same-axis L8.3.2 counterexample | **VERIFIED** | §§4.3/8.2 |
| Mixed cross-axis L8.3.2 counterexample | **VERIFIED** | §§4.4/8.2 |
| L8.3.1 two-trigger locality and L8.3.3 per-cell killing | **PROVEN** | §4.5 |
| L8.3.2 price and price-based O2′ sharpening | **WITHDRAWN** | Both one-placement counterexamples, §4.5. |
| Strategy-reachable `B₂=Θ₂` renewal | **OPEN** | Exact `n₁/9` transition; no witnessing strategy, §§4.5/6.1. |
| K1, every two-cell cover unripe | **PROVEN** | §5.1 |
| K2, any one-cell cover has a good spare | **PROVEN** | Critical-core inventory proof, §5.3. |
| K3 matching-number-at-most-two mass statement | **PROVEN** | §5.4 |
| Exact path/cycle critical-core cases | **PROVEN** | Covered or mass-excluded, §5.4. |
| General matching-number-at-most-two K3 suppression | **OPEN** | Pairwise, branched/nonclique, and chorded cases beyond exact paths/cycles; at most eight labels, §5.4. |
| W3′ as displayed | **WITHDRAWN** | Undefined/insufficient universe and locality, §§5.4/9. |
| Joint obligation J | **OPEN** | One strategy using canonical `B₂=Θ₂`, §6. |
| Theorem D₂, `J⇒GAP-RAW` | **PROVEN** | §7 |
| Three stored R1b break-line `τ` audits | **VERIFIED** | Exact `τ=3` at ply 56, §8.5. |
| Six-root no-forced-pileup/six through six plies | **VERIFIED** | Inherited completed minimax scope, §8.6. |
| Old Corollary B′ / Theorem D assembly | **WITHDRAWN** | K3 and common-strategy failures, §§7/9. |

**Inventory tally:** 19 **PROVEN**, 11 **VERIFIED**, 5 **OPEN**, 4
**WITHDRAWN**, 0 **CONJECTURED** rows. The larger §9 ledger lists every
withdrawn round-1 overstatement rather than hiding it inside grouped rows.

## 11. Repair-list compliance table

| # | Required repair (review §10, verbatim) | Round-2 action | Where |
|---:|---|---|---|
| 1 | **Restore the normative domain verbatim.** Define finite, nonempty, nonterminal blanket Maker–Breaker positions with explicit `FirstStone` phase; exclude completed windows before defining `I` and `τ`. | Done. The state tuple includes finite disjoint supports, nonemptiness, side, phase, radius-eight legality, blanket terminal semantics, and nonterminal `I={count 4,5}` only. | §§1–2 |
| 2 | **Replace Theorem A's right-hand side.** Quantify one strategy whose actual legal two-cell reply services `I(P)` at every reached defender epoch against every attacker continuation. Prove A′ directly from that statement. | Done. `Service(S,P₀)` names the actual ordered reply; A₂ and A₂′ are proved by finite bad-prefix arguments with the same S. | §2 |
| 3 | **Withdraw O1′ as stated and add the three-gadget regression.** Replace "every `Θ₂<1` start" with histories reachable from an original `Φ<1` root under the same named strategy; do not claim renewal until count-one latent promotion capacity is included. | Done. O1′ is withdrawn; the exact engine history and best-defense `10/9` regression pass; J uses only S-reachable histories and mandates `n₁/9`. Renewal stays OPEN. | §§4.2, 6, 8.2 |
| 4 | **Replace W3′ + O1′ by one joint obligation J.** J must select the actual cover/spare pair, hand an unripe position, and renew the account on every J-reachable history. Do not conjoin separate existential witnesses. | Done. J quantifies one S and uses the explicit account `B₂=Θ₂`; D₂ derives GAP-RAW from J alone. | §§6–7 |
| 5 | **Delete L8.3.2's pricing claim.** Add both explicit counterexamples above; case-split pre-count-3/pre-count-3 from the mixed pre-count-2/pre-count-3 branch; define exactly which past interval any reactivation charge covers. | Done. Both counterexamples are adopted and machine-verified; the two-high restricted observation is separated; the charge interval ends before the witness pair. | §§4.3–4.5, 8.2 |
| 6 | **Rebuild K3 before naming its residue.** Either prove a two-cell defusal theorem for every witness collection of matching number at most two, or prove a geometry lemma reducing path/cycle intersection graphs to the pairwise case. Remove the unsupported count-three-pool sentence. | Partial / OPEN. Critical cores replace representatives; exact path/cycle graphs are covered or mass-excluded and the count-three pool is withdrawn, but neither requested general disjunct is proved. Pairwise and other matching-number-two families remain OPEN. K2 closes independently. | §5 |
| 7 | **Rewrite L7.3 as per-cluster only.** Explicitly allow two heavy halves of one witness and formulate suppression as a hitting problem on all heavy trigger cells. | Done. L7.3₂ is per-cluster; two heavy halves are explicit; the global object is the set cover of all minimal critical cores. | §§3.4, 5 |
| 8 | **Reverify L2/L3 over a complete universe.** Include edge-disconnected but co-window-interacting configurations, count-5 singleton residuals, all four L3 columns, and hard assertions for every advertised number. Until this is done, label L2/L3/L7.5/Theorem C as dependent on an unverified lemma. | L2 is now complete and VERIFIED by the 902-case anchor reduction with singleton residuals. All four L3 columns are hard-gated only for edge-connected `n≤12` and unrestricted `[0,6]²`, `n≤6`; global L3 remains OPEN with the missing universe named. | §§3.5, 8.3–8.4 |
| 9 | **Repair the straight-four cover step.** Replace matching-number language by the explicit two-cell cover and restrict L4/L5a to the exact named roots. | Done. Residuals and cover `{-1,4}` are proved and tested; the six maturation roots are listed exactly. | §§3.1, 8.2, 8.6 |
| 10 | **Instrument every R1b break epoch.** Print and assert the variable-size minimum hitting set, then infer `Θ₂≥1` only on epochs where `τ≥3` is actually established. | Done for all epochs of the three stored replay lines. Each certified loss has exact `τ=3` at ply 56; earlier window counts with `τ≤2` make no inference. | §§8.2, 8.5 |
| 11 | **Narrow all evidence prose.** Say "six named roots," narrow Theorem 2 sharpness to its specified strategy/filler, narrow Cor-2 neutralization to its direct clean-escape source, and update the HEAD/provenance labels. | Done. The two different six-root scopes are explicit, sharpness and clean escape are narrowed, and provenance is `159c75f46…`. | §§3.1, 3.5, 4.1, 8.6, 13 |
| 12 | **Correct Theorem C's statement.** Add nonterminality and use "can be no earlier than" throughout; do not claim attainability from the L2 lower bound. | Done. C₂ includes the full domain, defines `t*` from zero, and is only a lower bound. | §3.5 |

## 12. Attack surface for round-2 hostile review

Strike these points first:

1. **J is sufficient but unproved.** Its account is the explicit conservative
   choice `B₂=Θ₂`, not an admissibility schema. Attack whether one strategy can
   renew `B₂<1` while making the same service and suppression choices. D₂ uses
   the universally quantified Service clause logically; the renewal and
   unripeness clauses bind the intended construction but make J stronger, not
   equivalent or smaller.
2. **K3 is still load-bearing.** Exact paths and cycles are handled, but the
   general matching-number-two critical-core family is OPEN. The first abstract
   obstruction has five count-three labels and all three-label subsets as
   cores. A successful attack should realize that incidence pattern below
   `Θ₂<1` in actual length-six geometry, or prove that two physical Defender
   cells cannot suppress it.
3. **Two-neighbourhood locality is not proved.** A witness's triggers may be
   arbitrarily far apart near separate stock. Any proposed bounded enumeration
   must factor two local neighbourhoods plus their label interaction; a single
   radius-eight ball is not complete.
4. **K2's new proof has two sharp joints.** Attack the sliding-word proof of
   the all-count-two lemma, and the sufficient implication from hitting every
   inclusion-minimal over-approximate critical core to unripeness. The
   pre-spare universe uses
   every one-/two-cell subset of the finite relevant empty set, deliberately
   over-approximating legality; no post-spare trigger may fall outside that
   argument.
5. **The L2 normalization is mathematical, not just a passing test.** Challenge
   the claim that with at most six relevant Attacker stones, any two imminent
   windows share at least two Attacker cells and therefore lie on one axis.
   The complete machine sweep is valid only if that reduction and the
   nonterminal/count-five handling are sound.
6. **L3 is not global.** Passing edge-connected `n≤12` and bounded unrestricted
   `[0,6]²`, `n≤6` tests does not establish arbitrary-configuration maxima for
   `n=7..12`. Any prose or downstream lemma using those rows as absolute is a
   round-2 defect.
7. **The O1′ test is a regression, not a GAP refutation.** Its final state has
   `Φ>1`; it refutes the every-`Θ₂<1` domain only. The strategy-reachable repair
   may exclude it, but must do so through the same S's earlier actions while
   still booking `n₁/9`.
8. **R1b loss is not synonymous with an unblockable epoch.** Only rows whose
   exact variable-residual minimum is at least three license
   `Θ₂≥1`. A loss after a bad policy choice with `τ≤2` proves no pileup theorem.
9. **Evidence roots remain named.** The maturation table's six roots and the
   pileup table's six (different) roots are not the set of all `Φ<1` positions.
   Theorem 2 sharpness remains tied to its stated filler.
10. **Strict thresholds and phases matter.** `Φ<1` and `B₂<1` must not become
    `≤1`; completion/service criteria apply at `FirstStone`; completed windows
    never enter `I` or an empty-residual hitting-set convention.

## 13. Regeneration commands and provenance

**Input worktree commit:**
`159c75f46db0090aca87ab8306dc0e3001541a50`. No commit was created in this
repair round.

**Edited harness:** `packages/hexfield_eq/rust/src/gap_raw_hunt.rs` (test-gated;
production rules untouched).

**Evidence read:** `GAP_RAW_REVIEW_ROUND1.md`, `GAP_RAW_PROOF_ROUND1.md`,
`HUNT_REPORT_GAP_RAW.md`, `HUNT_REPORT_ADAPTIVE.md`, the read-only sibling
`hunt-birth-ledger/HUNT_REPORT_BIRTH_LEDGER.md`, and the read-only normative
`hexfield-eq-main-review-65bd3a/docs/proof_parts/ES_GLOBAL_BOUNDARY.md`.

From the worktree root, the complete round-2 machine gate is:

```powershell
$existing = @(Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^cargo(\.exe)?$' })
if ($existing.Count -ne 0) { throw "another Cargo invocation is active" }

$free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($free -le 9) { throw "free physical RAM must be >9 GiB" }

$env:CARGO_TARGET_DIR = '.target-hunt'
cargo test -p hexfield_eq --lib --release `
  'gap_raw_hunt::tests::round2_' -- `
  --ignored --nocapture --test-threads=1
```

Observed result on 2026-07-16: seven passed, zero failed; 178.662 seconds wall
time; preflight 13.796 GiB free.

The inherited maturation and pileup report commands are, for provenance only:

```powershell
$env:CARGO_TARGET_DIR = '.target-hunt'
cargo test -p hexfield_eq --lib --release `
  'gap_raw_hunt::tests::birth_ledger_maturation' -- `
  --ignored --nocapture --test-threads=1

cargo test -p hexfield_eq --lib --release `
  'gap_raw_hunt::tests::birth_ledger_pileup' -- `
  --ignored --nocapture --test-threads=1
```

They were not run in this round: the prior maturation runtime exceeded the
per-run limit, and the pileup result is cited only at its already completed
six-root/six-ply scope.
