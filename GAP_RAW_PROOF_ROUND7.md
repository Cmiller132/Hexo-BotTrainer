# GAP-RAW Proof Round 7 — the `P_stock` decision

**Worktree:** `hunt/gap-raw` at input HEAD
`8ac6caaec8668e77e7c4097c12336e0154c73841`  
**Date:** 2026-07-18  
**Disposition:** **NOT-REPAIRABLE at `P_stock`.** Every hub-pre-empting
servicing pair at every round-6 plateau epoch has a legal Attacker response
returning `M>=3`. At the final `P_stock` epoch the stronger statement holds:
**every** legal Defender pair has such a response. Thus `P_stock` is already a
local forced-loss state, and no tie-breaking refinement of `S_T` can repair it
after arrival. This is a Q1-reached-state/Q3-action theorem; Q2 reachability
against every Defender strategy remains **OPEN**.

This document continues the definitions, theorem numbering, and equation
numbering of `GAP_RAW_PROOF_ROUND6.md`, including the binding folded Section
58 errata and the review's exact post-hub strengthening `M=4`, `tau=4` on the
round-6 ray line. In particular, `I`, `tau`, `TEMPO`, `M`, `Serv`, `V_T`,
`S_T`, and the diagnostic `R_1` retain their inherited meanings. The
four-pencil shield is an exact value plateau: hub pre-emption ties immediate
`TEMPO=2`; it is not charged a larger immediate value.

No Cargo command, Lean build, harness, game/search program, or machine
enumeration was run. Every new result below is a hand proof. There are no new
`VERIFIED` or `[UNRUN]` claims, and no git commit is created by this pass.

## 59. Executive verdict and quantifier contract

### 59.1 The sharp decision

Let `P_i^pl`, `i=0,...,5`, denote the six exact round-6 Defender plateau
epochs immediately after, respectively,

```text
the diamond shield, U^- stock, V^- stock, W stock, V^+ stock, U^+ stock.
```

Thus `P_5^pl=P_stock`. By L12.2 and L12.4, every `P_i^pl` has `I=empty`,
every alive label has count at most two, all four shield pencils remain
intact, and every legal Defender pair has immediate handoff value exactly
two.

**Theorem R7.1 (hub-pre-emption does not repair a plateau) [Q1/Q3,
PROVEN].** For every `i=0,...,5` and every legal servicing pair `a` at
`P_i^pl` which contains the hub `h=(10,0)`, Attacker has a legal nonterminal
response pair `b` such that

`M(P_i^pl+D@a_1+D@a_2+A@b_1+A@b_2)>=3`.                 (72)

Consequently no hub-pre-empting pair at any named plateau epoch satisfies
`max_b M(P_{a,b})<=2`: `P_stock` is **not repairable by hub pre-emption**.
The full universal **NOT-REPAIRABLE** verdict — over EVERY legal Defender
pair, hub-containing or not — follows from R7.2 below, whose adaptive
all-pair fork independently supplies the missing non-hub cases (review
round-7 Finding 9).

Expanded through the forced completion, the negative-gadget chronology is

```text
for every i, for every legal hub-containing a,
there exists sigma in {-,+} such that, after b_sigma,
for every next Defender pair d there exists a legal Attacker pair e
returning tau>=3, and after every following Defender pair f
there exists a legal Attacker completion.
```

**Theorem R7.2 (`P_stock` is a universal local stop state) [Q1/Q3,
PROVEN].** At the exact final `P_stock`, for every legal Defender pair `a`
there is a legal nonterminal Attacker response `b` with

`M(P_stock+D@a_1+D@a_2+A@b_1+A@b_2)>=3`.                (73)

Equivalently, using round 6 equation (71),

`min_{a in Serv(P_stock)} R_1(P_stock,a)>=3`.             (74)

Every pair is in `Serv(P_stock)` and is immediate-`TEMPO` minimizing, so
(74) excludes every possible tie refinement at that state, including a
one-ply Bellman tie refinement. One-ply risk correctly detects the stop
state; it does not supply an action which repairs it.

**Corollary R7.2.1 (shorter fixed-`S_T` refutation) [Q1, PROVEN].** The
actual first plateau ray reply `(-24,8),(-32,8)` already admits the triangle
response in Section 60 and returns `M>=3`. Hence the five stock turns are not
needed to refute fixed `S_T`. They remain valid and useful for the named
hub/`P_stock` diagnosis proved in round 6.

### 59.2 What the result does not say

**Q2 root-forcing status [Q2, OPEN].** Theorems R7.1--R7.2 prove a universal
Defender loss *from the reached state* `P_stock`; they do not prove that a
strict root forces arrival at that state against every Defender strategy.
The Q2 order

`exists P_0, for every S, exists an S-consistent Attacker continuation`

therefore remains open.

**Q3 policy status [Q3, OPEN].** No `S_T'` is defined in this round. The
positive branch required for such a repair is false at `P_stock`. A viable
Q3 policy must intervene before the three-threat packing of Section 61 is
complete and must also control every other inherited repair class.

### 59.3 Secondary initialization result

**Lemma L13.6 (count-three-transversal initialization) [Q3-initialization,
PROVEN].** At a finite nonterminal Defender epoch with `tau=0`, if the alive
count-three residual family has hitting number at most two, then `M<=2`.
In particular, arbitrary lower-count stock together with at most two alive
count-three labels is initialized. This extends L12.6 through the first
count-three label, and in fact through two; it is not a renewal theorem.

## 60. The two triangle fans hidden in the shield

### 60.1 A consecutive-triple demand lemma

On any one lattice axis, parameterize cells by integers. Suppose the three
consecutive cells at parameters `0,1,2` are occupied by Attacker, the four
length-six windows with starts `-3,-2,-1,0` are Defender-free, and those
windows contain no other Attacker stone. Call these four count-three windows
a *consecutive-triple family*.

**Lemma L13.1 (one-trigger demand of a consecutive triple) [Q3-structural,
PROVEN].** A consecutive-triple family has these properties.

1. All four labels have count three.
2. No empty Defender cell kills all four labels.
3. If the family is untouched by Defender, one future Attacker trigger at
   parameter `-1` creates a residual subfamily

   `{-3,-2}, {-2,3}, {3,4}`,                              (75)

   of hitting number exactly two.
4. If one Defender cell has touched the family, at least one count-three
   label survives, and one future trigger in that label creates demand at
   least one.

*Proof.* The no-other-A premise gives count exactly three in all four
windows. Their physical intersection is the occupied triple `{0,1,2}`, so no
empty Defender cell kills all four. The trigger `-1` matures the first three
windows and leaves exactly (75). Its first and third grounds are disjoint,
while `{-2,3}` hits all three, so the hitting number is two. After one
Defender contact, killing all four would again require a cell in their
occupied common intersection. A surviving count-three label has three empty
residual cells; triggering any one leaves a nonempty count-four residual and
hence demand one. ∎

The translated or reflected version of L13.1 will be used without changing
its status.

### 60.2 Exact south and north responses

Retain the round-6 diamond Attacker stones

`(0,0), (0,1), (1,0), (1,-1)`.

Define two ordered Attacker pairs

```text
b_- = ((0,-1),(2,-1)),
b_+ = ((1,1),(-1,1)).                                    (76)
```

The *south fan* uses the three lines

`q=0`, `q+r=1`, and `r=-1`,                              (77)

and the *north fan* uses

`q=1`, `q+r=0`, and `r=1`.                               (78)

The south response creates these three consecutive triples:

```text
q=0:     (0,-1),(0,0),(0,1),
q+r=1:   (0,1),(1,0),(2,-1),
r=-1:    (0,-1),(1,-1),(2,-1).
```

The north response is their image under the inherited half-turn

`rho(q,r)=(1-q,-r)`.

Let `G_-` and `G_+` be the physical unions of the twelve length-six windows
in the respective three consecutive-triple families.

**Lemma L13.2 (exact fan inventory and separation) [Q1/Q3, PROVEN].** At
every `P_i^pl`, if no newly chosen Defender cell lies in `G_sigma`, then
`b_sigma` is a legal ordered nonterminal response and creates twelve focal
count-three labels, four on each line of that fan. It creates no count-four
label anywhere. Moreover:

1. within one fan, its three lines meet pairwise only at occupied Attacker
   cells after `b_sigma`;
2. `G_- intersect G_+` consists only of the four old occupied diamond cells;
3. `h=(10,0)` lies in neither fan union; and
4. every pre-existing round-6 Defender cell and every stock cell lies outside
   both fan unions.

*Proof.* In the south fan, the first trigger lies in the intact `q=0` shield
pencil and the second lies in the intact `q+r=1` shield pencil. Each is an
empty cell of an alive count-two window, so both were already legal; they are
also at distance two from one another. The bridge line `r=-1` contained the
old Attacker stone `(1,-1)` and no second old Attacker stone. Thus (76)--(77)
give exactly the three displayed consecutive triples. The north case follows
by `rho`.

Before either response, L12.4 gives global count at most two. A label
containing only one new trigger therefore reaches at most count three. A label
containing both south triggers lies on `r=-1`, which had only `(1,-1)` before
the response, and likewise the north bridge `r=1` had only `(0,1)`. Such a
label also reaches at most count three. Hence `I=empty` after the response.

Within the south fan the three line intersections are `(0,1)`, `(0,-1)`,
and `(2,-1)`, all Attacker-occupied after the response. The north statement
is its image. Across the two fans, equal-axis pairs are parallel; all other
line intersections are among

`(0,0), (0,1), (1,0), (1,-1)`.

Each of these four old diamond cells belongs to both finite unions, so this
is the exact intersection. Thus no empty cell lies in both fan unions. The
hub has `q=10`, `r=0`, and `q+r=10`, so it lies on none of the six fan lines.

For the finite-support check, the three south segments are

```text
q=0, -4<=r<=4;   q+r=1, -3<=q<=5;   r=-1, -3<=q<=5,
```

and the three north segments are

```text
q=1, -4<=r<=4;   q+r=0, -4<=q<=4;   r=1, -4<=q<=4.
```

The anchors and Defender ray are outside those finite segments;
`(-4,0),(2,0)` lie on none of the six lines. The stock census in L12.4 puts
every stock stone on `r=0`, `q=10`, `r=-3`, or levels
`6,7,10,14,15`, outside the displayed fan supports. ∎

### 60.3 The forced rebuild after an untouched fan

**Lemma L13.3 (triangle-fan cascade) [Q1/Q3, PROVEN].** Suppose one of
`G_-`,`G_+` is Defender-free at a plateau handoff and Attacker plays its pair
`b_sigma`. At the returned Defender epoch `P`,

`I(P)=empty` and `M(P)>=3`.                                (79)

Consequently Attacker has a forced win from `P` after every next Defender
pair.

*Proof.* Lemma L13.2 gives `I(P)=empty`, so every legal Defender pair `d`
services. An empty Defender cell lies on high labels from at most one of the
three fan lines, because their pairwise intersections are occupied Attacker
cells. Two Defender cells therefore leave at least one line family untouched.

If `d` touches at most one line family, two consecutive-triple families are
untouched. Apply the flank trigger from L13.1 independently on those two
lines. Each creates demand two. Their residual grounds are disjoint: the two
physical lines meet only at an Attacker-occupied cell, which belongs to no
residual. The combined future Attacker pair therefore creates demand four.

If `d` touches two line families, it uses one cell on each, while the third
family is untouched. The untouched family supplies demand two by L13.1. A
touched family retains a count-three label, again by L13.1; choose one of its
empty residual cells as the other future trigger, giving demand at least one.
The two residual grounds are disjoint for the same occupied-intersection
reason. This future pair therefore creates demand at least three.

Thus every actual pair `d` hands over `TEMPO>=3`. Definition (21) gives
(79). Equivalently, after `d` Attacker chooses the displayed two-trigger
witness, returning `tau>=3`; L1.2 then gives a missed imminent label and a
completion on the following Attacker turn. Every future trigger used above is
an empty cell of a surviving alive window, hence legal by L6_2, and the
ordinary ordered 2:2 cadence is respected. ∎

### 60.4 Proof of the plateau theorem

*Proof of Theorem R7.1.* Fix a plateau `P_i^pl` and a servicing pair `a`
containing `h`. By L13.2, `h` touches neither fan union. Its other cell is
empty and hence cannot lie in `G_- intersect G_+`, whose only members are
occupied diamond stones. It can therefore touch at most one fan. Choose the
other sign `sigma`. That entire `G_sigma` remains Defender-free, so L13.3
applies to the legal response `b_sigma` and yields (72). ∎

*Proof of Corollary R7.2.1.* The first actual plateau reply
`(-24,8),(-32,8)` lies outside both finite fan unions. Attacker may play
either pair in (76), after which L13.3 gives `M>=3`. The deterministic `S_T`
reply then hands over a `TEMPO>=3` state, and the L13.3 continuation forces
completion. No post-fan `S_T` coordinates are asserted or needed: L13.3
quantifies over every legal next reply, including whichever exact minimizer
and tie-break `S_T` selects. ∎

## 61. The final `P_stock` state is already lost

### 61.1 Three pairwise action-disjoint threat regions

At final `P_stock`, let

`H=U^- union U^+ union V^- union V^+ union W`              (80)

be the physical union of the five focal windows from round 6 equation (64).
All five are intact count-two labels with common empty hub `h`.

**Lemma L13.4 (hub/fan region separation) [Q1/Q3, PROVEN].** The three
regions `H,G_-,G_+` are pairwise disjoint for legal Defender actions:
`H` is disjoint from both fan unions, while `G_- intersect G_+` contains
only occupied Attacker cells. Hence one empty Defender cell can affect labels
in at most one of the three regions.

*Proof.* The fan/fan statement is L13.2. The two horizontal focal windows in
`H` lie on `r=0` with `6<=q<=15`; every intersection there with a fan line has
`q=0` or `1`, outside the focal intervals. The two vertical focal windows lie
on `q=10`, `-4<=r<=5`; their intersections with fan rows `r=+-1` have
`q=10`, while those row-family unions end by `q=5`. Their intersections with
fan diagonals similarly have varying coordinate outside the fan's
`[-4,5]` range. Finally, `W` lies on `q+r=10` with `8<=q<=13`; its
intersections with fan rows occur at `q=9` or `11`, again outside those finite
row-family unions, and its intersections with `q=0,1` lie outside `W`.
Parallel cases contribute no intersection. Thus `H` meets neither fan
union. ∎

### 61.2 The adaptive fork

*Proof of Theorem R7.2.* Let `a` be any legal Defender pair at `P_stock`.
Because `I=empty`, it is servicing.

If neither cell of `a` lies in `H`, all five focal labels remain intact. The
hub pair

`(h,h+u)=((10,0),(11,0))`

is still legal. The two horizontal labels become current with the disjoint
residuals from round 6 equation (66). Every servicing pair must use one cell
in each residual, consuming both Defender placements on nonhub `r=0` cells.
Those cells lie in none of `V^-`, `V^+`, or `W`. The future pair
`(h+v,h+w)` therefore leaves the three pairwise-disjoint focal residual
grounds from equation (69), so every servicing reply has `TEMPO>=3` and the
returned epoch has `M>=3`. This focal lower bound is independent of which
extra, nonfocal labels `a` deleted.

If at least one cell of `a` lies in `H`, that cell lies in neither fan by
L13.4. At most one other Defender cell remains, and it can meet at most one
fan. Thus one `G_sigma` is untouched. Attacker uses `b_sigma`, and L13.3 gives
the returned `M>=3` state.

These two branches cover every `a` and prove (73)--(74). In either branch,
if the next Defender pair fails to service a current imminent, the missed
label is completed directly; if it services, the `M>=3` argument supplies
the ripe handoff and L1.2 completion. Thus `P_stock` is a local forced-loss
state, not only a failure of one evaluation rule. ∎

The full local quantifier order is therefore

```text
for every legal a at P_stock, there exists b in {(h,h+u),b_-,b_+}
such that for every next Defender reply d, either d misses a current imminent
and Attacker completes directly, or d services and there exists a legal pair
e returning tau>=3, after which every following Defender pair permits
completion.
```

### 61.3 Exact scope of the stop theorem

**Local stop conclusion [Q1/Q3, PROVEN].** Once `P_stock` is presented as a
Defender epoch, Attacker wins against every subsequent Defender action by
choosing the hub or triangle branch after seeing the actual pair. This is the
local forcing lemma

`for every Defender pair a, there exists a legal response b and continuation`.

**Root reachability conclusion [Q2, OPEN].** The theorem supplies no
strategy-independent route from a strict root to `P_stock`. The round-6 route
is `S_T`-consistent only. Therefore neither Q2 nor GAP-RAW is refuted.

## 62. What a Q3 policy must see, and when

### 62.1 One-ply risk detects the failure but cannot repair the stop state

At a Defender epoch `P`, put

```text
A_0(P)={a in Serv(P): TEMPO(Q_a)=M(P)},
B_1(P)=min_{a in A_0(P)} R_1(P,a),                         (81)
```

with `R_1` exactly as in round 6 equation (71).

**Theorem R7.3 (one-ply stop-criterion assembly) [Q3, PROVEN].** Fix one
deterministic Defender policy which, at every
reached epoch `P`, uses one actual pair `a in A_0(P)` satisfying
`R_1(P,a)<=2`. If `M(P_0)<=2` at its root and such a choice remains available
after every legal response on every reached history, then that same policy
blocks forever.

*Proof.* The chosen actual pair services and has `TEMPO=M(P)<=2`, so its
handoff is unripe by R4.1. Its `R_1<=2` bound says that every legal Attacker
response returns an epoch with `M<=2`. Repeat with the same deterministic
policy. Every reached epoch is serviced, and A2 gives perpetual survival. ∎

This is a conditional assembly theorem, not a new `S_T'`: the availability
premise is precisely the missing closure claim.

**Exact `P_stock` diagnosis [Q3, PROVEN].** Equation (74) says
`B_1(P_stock)>=3`. One-ply risk therefore recognizes that every action at
`P_stock` fails the stop criterion. It cannot choose a safe action there.

**First-plateau diagnosis [Q1/Q3, PROVEN].** The actual `S_T` ray pair at
`P_0^pl` has `R_1>=3`, witnessed by either untouched triangle response.
Thus one-ply risk exposes the fixed-policy error at the first plateau, five
turns before the named hub cascade.

**First-plateau decision [Q3, OPEN].** It is not proved whether
`B_1(P_0^pl)<=2`. A pair can spend one cell in each fan region, but no
all-response `M<=2` theorem for such a pair is supplied here. Therefore this
round proves neither that one-ply minimization is sufficient nor that deeper
Bellman recursion is logically necessary. It proves the sharper bounded
statement: one ply detects the ray's failure and the final stop state, while
the existence and closure of a safe one-ply action remain open.

### 62.2 The state information omitted by immediate `TEMPO`

**Threat-packing requirement [Q3, PROVEN for the exact line].** A repair
proof which allows the round-6 line to continue must distinguish three
pairwise action-disjoint response certificates at `P_stock`: the focal hub
region `H` and the two bridge fans `G_-`,`G_+`. Two Defender cells cannot
touch all three. Equivalently, a policy must prevent this packing before it
is complete or prove that its histories never reach it.

The bridge fans are not visible as a second common count-two hub. Each uses
two old count-two shield pencils and one count-one bridge through a diamond
stone; the Attacker pair promotes that bridge into the third high pencil.
Thus a Q3 invariant tailored to this obstruction must retain enough state to
recognize count-one bridge incidence and cross-hull line intersections, not
only current `tau`, immediate `TEMPO`, or the five-label focal hub.

This is a necessary distinction for excluding the displayed continuation,
not a claim that a policy must literally store `G_+` and `G_-`. A structural
rule, history credit, or deeper recursive value could encode the same
distinction differently.

### 62.3 Early exact `R_1` audit

**Lemma L13.5 (root and raw-epoch one-ply values) [Q1/Q3, PROVEN].** On the
round-6 line:

1. at the `Phi=0` three-anchor root, every initial Defender pair has
   `R_1=0`; and
2. at the isolated raw adjacent-pair epoch, every immediate-`TEMPO`-zero
   Q-row cover has exact `R_1=2`.

Consequently adding `R_1` only as a tie layer would retain the exact initial
lexicographic pair and the exact R5.3 axial cleanup. The diamond plateau is
the first epoch which requires a new `R_1` comparison and at which that layer
could distinguish the actual ray; whether it ultimately selects another
action remains open.

*Proof.* After any initial root pair, an Attacker response leaves at most two
Attacker stones. If they share no window, play two sequential L1.2 fillers.
Otherwise they lie on one axis. For nonadjacent stones, any internal gap hits
every common window; if such a gap is already Defender-occupied, those common
windows are already dead. Play one empty gap when needed and append enough
sequential fillers to complete the actual pair. For adjacent stones, the two
exterior flank cells cover the five common windows; omit any flank which is
already Defender-occupied or redundant, and append fillers. Every used
nonfiller is within five of unchanged Attacker stock. Thus an actual legal
ordered pair deletes every count-two label and hands over `L_23=empty`,
proving `M=0` for every response.

At `P_raw`, the round-6 review enumerates exactly twenty ordered value-zero
Q-row covers. Each kills every old count-two Q-label and touches none of the
four transverse pencil lines away from their occupied endpoints. The handoff
therefore has no `L_23` label. After an arbitrary Attacker pair, a count-three
label can only arise from a pre-count-one label containing both triggers; all
such labels lie on the unique axis through the two triggers. The rank-triple
transversal argument of L11.4 supplies a physical cover of size at most two
for that entire one-axis high family. If a cover point is already occupied by
Defender, every label through it is already dead; discard that point, discard
any newly redundant point, and append L1.2 fillers. The remaining cover cells
are empty and legal. Since the pre-response handoff has no `L_23` label, no
count-four label can arise, so this actual pair services. Its surviving
graded tier is pure count two, and L10.4 gives `M<=2` after every response.
The shield pair `(0,1),(1,-1)` returns a four-pencil state satisfying L12.2,
where every Defender pair has handoff `TEMPO=2`; hence its returned epoch has
`M=2`. The worst response is therefore exactly two for every value-zero
cover. The inherited lex order then selects the same pairs as in round 6. ∎

### 62.4 Stress against inherited classes

**Tie-only repair at the R5.4 shared hub [Q3, REFUTED at the R5.4
domain].** At `P^dagger`, the two singleton count-five residuals force the
servicing occupancy `{z_1,z_2}`. There is no alternative occupancy or
handoff effect for a tie-break to select (the two orders remain), and the
legal hub response returns `M>=3`. Hence no
refinement which only chooses among immediate-`TEMPO`-minimizing servicing
pairs repairs the unrestricted R5.4 geometry. That position has `Phi>1`;
this remains a statewise negative control, not a strict-root Q3 refutation.

**Axial-cleanup exact-line stress [Q1, PROVEN].** The exact round-6
axial-cleanup handoff still permits the shield response, and the actual first
plateau ray then loses to L13.3. This proves that bridge-aware repair is
required on that line.

**General axial-cleanup repair [Q3, OPEN].** The complete response universe
from the axial-cleanup class, including whether a different first-plateau pair
has `R_1<=2`, remains unclassified.

**Separated sealed class [Q3, OPEN beyond the banked one cycle].** R5.2 still
proves that every first response from its fully one-cycle-separated sealed
class returns `M<=2`. It does not prove `R_1<=2` at the derivative epoch and
does not iterate the nested derivative (31). The exact sealed profile is
protected for the banked first cycle by R5.1, while R5.2's full-separation
premise prevents cross-hull coupling. No `S_T'` survival theorem follows.

## 63. Initialization ladder: the first count-three rung

### 63.1 A residual-transversal condition

For a finite nonterminal Defender epoch `P`, define

```text
I_3(P)={W: W is alive at P and count_P(W)=3},
tau_3(P)=the hitting number of {E_P(W): W in I_3(P)}.       (82)
```

The empty family has value zero, as usual. This is an initialization-only
notation; it is not added to the proposed perpetual state invariant.

*Proof of Lemma L13.6.* Since `tau(P)=0` and every imminent residual is
nonempty at a nonterminal epoch, `I(P)=empty`. Choose a transversal `X` of
the alive count-three residuals with `|X|<=2`.

Every `x in X` is an empty cell of an Attacker-alive window and hence is
legal by L6_2. If `|X|=2`, play its cells in axial lexicographic order. The
second cell remains legal even if the first kills a label which formerly
contained it, because it was already within distance five of an unchanged
Attacker stone. If `|X|<2`, append sequential legal fillers using L1.2's
finite max-`q` construction after the cells of `X` have been played.

The resulting ordered pair services the empty current imminent family and
kills every old count-three label. Defender placements create no new alive
label and change no Attacker count. Thus every surviving member of `L_23` at
the actual handoff has count exactly two. L10.4 gives `TEMPO<=2` for this same
actual pair, and definition (21) gives `M(P)<=2`. ∎

**Corollary L13.6.1 (one- and two-label rung) [Q3-initialization, PROVEN].**
If `tau(P)=0` and `|I_3(P)|<=2`, then `M(P)<=2`, with arbitrary additional
alive count-one/count-two stock. In particular, the smallest
class beyond L12.6 requested in this round—arbitrary low stock plus exactly
one count-three label—is initialized.

*Proof.* Pick one residual representative from each count-three label. The
resulting set has size at most two and witnesses `tau_3<=2`; apply L13.6. ∎

No `Phi<1` hypothesis is used in L13.6 or its corollary. Intersecting the
state class with normative strict roots gives the strict-root result. The
potential bound only limits how much count-three stock a strict root can
carry; it is not needed for this transversal construction.

### 63.2 Strict nonvacuity example

Let `u=(1,0)`,

```text
W={t u:0<=t<=5},
A={0,u,2u}.
```

Let `T` be the finite family of all length-six windows meeting `A`, and put

`D=union_{Y in T, Y!=W}(Y\W)`.                             (83)

Let `P=(A,D,Defender,FirstStone)`.

**Lemma L13.7 (isolated one-count-three strict root) [Q3-initialization,
PROVEN].** Equation (83) with the displayed phase defines a finite
nonterminal Defender root whose
exact alive family is `{W}` at count three. It has

`Phi=1/(3 sqrt(3))<1`, `tau=0`, and `M=0`.

*Proof.* A same-axis length-six window distinct from `W` leaves the physical
interval `W`. An off-axis window meets the central axis in at most one cell,
so it also has a cell outside `W`. Hence every set `Y\W` in (83) is nonempty.
The blocker union is finite, avoids `W`, and kills every Attacker-touched
window other than `W`. Since `A subset W`, it also satisfies
`D intersect A=empty`. Thus the exact profile is one count-three label and
the displayed potential follows.

The empty cell `3u` is legal and kills `W`; append a sequential legal max-`q`
filler from L1.2. The handoff has `L_23=empty`, so its `TEMPO` is zero and
`M=0`. ∎

### 63.3 Remaining initialization boundary

**General count-three initialization [Q3-initialization, OPEN].** L13.6 does
not settle roots whose count-three residual family has hitting number at
least three. Strict potential permits several count-three labels, and their
window-intersection geometry is not proved here to reduce to two residual
representatives. The inherited general `tau=0`/K3 problem therefore remains
open.

**Renewal from the new rung [Q3-repair, OPEN].** The Attacker's next pair may
create a new high family or a bridge fan. L13.6 proves initialization of the
displayed state class only; it supplies no all-response renewal theorem.

## 64. Authoritative round-7 status ledger

| Claim / named gap | Quantifier tag | Status | Exact basis / remaining scope |
|---|---|---|---|
| GAP-RAW | Q2 counterroute / Q3 target | **OPEN** | `P_stock` is locally lost, but it is not forced from a strict root against every Defender strategy; no universal Defender policy is proved |
| R7.1 hub-pre-emption at all six plateau epochs | Q1/Q3 | **PROVEN** | For every hub-containing pair, one of `G_-`,`G_+` is untouched and L13.3 returns `M>=3` |
| Named `P_stock` decision | Q1/Q3 | **PROVEN: NOT-REPAIRABLE** | R7.2's adaptive all-pair fork: EVERY legal Defender pair at final `P_stock` has one-ply worst-response value at least three (R7.1 supplies the hub-containing cases at all six plateaus) |
| R7.2 universal local `P_stock` stop theorem | Q1/Q3 | **PROVEN** | Adaptive fork among `H,G_-,G_+`; every Defender pair returns `M>=3` under one legal response; Q2 root reachability remains open |
| `min_a R_1(P_stock,a)>=3` | Q3 diagnostic | **PROVEN** | Equation (74); all legal pairs are immediate-`TEMPO` minimizers on the plateau |
| Strategy-independent reachability of `P_stock` | Q2 | **OPEN** | Round 6 reaches it only against fixed `S_T` |
| L13.1 consecutive-triple demand | Q3 structural | **PROVEN** | Four high intervals, occupied common intersection, exact demand-two flank residuals |
| L13.2 two exact triangle fans | Q1/Q3 | **PROVEN** | Exact lines, counts, legal responses, nonterminality, and fan separation |
| L13.3 triangle cascade | Q1/Q3 | **PROVEN** | Every next Defender pair leaves demand `2+1` or `2+2` on residual-disjoint lines |
| R7.2.1 shorter fixed-`S_T` refutation | Q1 | **PROVEN** | The actual first plateau ray misses both fans; stock assembly is unnecessary for policy refutation |
| Fixed `S_T` as universal repair policy | Q1 consequence | **REFUTED** | Inherited R6 refutation, strengthened by R7.2.1 |
| A tie-only `S_T'` repair at `P_stock` | Q3 | **REFUTED** | R7.2 ranges over every legal pair, not only the lexicographic ray |
| R7.3 one-ply stop-criterion assembly | Q3 | **PROVEN** | Conditional theorem: the same actual pair, `TEMPO<=2`, and `R_1<=2` give induction when availability closes |
| `B_1(P_0^pl)<=2` at the first plateau | Q3 | **OPEN** | The actual ray has risk at least three; actions touching both fan regions are not classified against every response |
| One-ply Bellman minimization as a complete repair | Q3 | **OPEN** | It detects the known failure; safe-action availability and closure are unproved |
| Threat-packing/count-one-bridge diagnosis | Q3 exact-line structural result | **PROVEN** | `H,G_-,G_+` are pairwise action-disjoint; fans use a promoted count-one bridge |
| L13.5 root/raw `R_1` audit | Q1/Q3 | **PROVEN** | Root worst response zero; every raw value-zero Q-row cover has worst response exactly two |
| R5.4 shared-hub tie repair | Q3 negative control | **REFUTED at the R5.4 statewise domain** | Mandatory `{z_1,z_2}` service leaves no tie choice; position has `Phi>1` |
| R5.2 separated sealed first cycle | Q3 | **PROVEN at inherited scope** | Round-5 one-cycle `M<=2` theorem unchanged |
| Iterated sealed/nested-derivative repair | Q3 | **OPEN** | No `R_1<=2` or next-cycle closure theorem |
| Complete axial-cleanup response class | Q3 | **OPEN** | Triangle response refutes the actual ray; other first-plateau actions remain unclassified |
| L13.6 `tau_3<=2` initialization | Q3-initialization | **PROVEN** | One actual pre-emptive pair kills every count-three label; L10.4 handles arbitrary count-two stock |
| At most two count-three labels plus lower stock | Q3-initialization | **PROVEN** | Corollary L13.6.1 |
| L13.7 strict one-label example | Q3-initialization | **PROVEN** | Exact isolator, strict potential, and `M=0` |
| General count-three `tau=0` initialization | Q3-initialization | **OPEN** | Residual hitting number at least three remains outside L13.6 |
| `GAP-TEMPO-REPAIR` for one named strategy | Q3 | **OPEN** | Neither one-ply availability nor another inductive invariant is proved |
| `GAP-TEMPO-INITIALIZATION` | Q3 | **OPEN** | Low-only and `tau_3<=2` slices close; remaining count-three geometry does not |
| Strategy-independent `GAP-HUB-FANOUT-REACHABILITY` | Q2 | **OPEN** | Fixed-`S_T` hub/`P_stock` reachability is proved; no strict root forces it against every strategy |
| `GAP-CASCADE-REACHABILITY` | Q2 | **OPEN** | Stronger local cascades do not supply strict-root `for every S` reachability |
| Other shared/nonshared `M>2` fanouts | Q3 | **OPEN** | Hub and triangle fans do not exhaust the possible high-family geometries |
| General cross-hull interaction closure | Q3 | **OPEN** | L13.3 resolves one exact bridge fan; R5.2's separation remains load-bearing elsewhere |
| Alternative forced-service transverse-seal entrance | Q3 | **OPEN** | R5.3 excluded only the natural entrance; round 7 supplies no alternative classification |
| `GAP-REPLACEMENT-INVARIANT` / amortized-credit route | Q3 | **OPEN** | No initialized and renewed replacement invariant or formal credit rule is supplied |
| Minimum separation for the R5.2 value theorem | ancillary | **OPEN** | Radius 21 remains envelope-sharp only |
| New machine verification | all | **none** | Hand proofs only; no prohibited run or generated enumeration |

No inherited round-2 through round-6 `PROVEN` or `VERIFIED` theorem is
downgraded. The exact round-6 post-hub strengthening `M=4`, `tau=4` remains
binding on its stated ray history. Round 7 adds a different earlier cascade
and a stronger final-state action quantifier.

## 65. Hostile-review attack surface

1. **Plateau wording.** Every legal pair still has exact immediate
   `TEMPO=2`. The triangle response attacks the next value `M`; it does not
   turn hub pre-emption into a worse immediate candidate.
2. **Bridge source.** The third south pencil is the pre-count-one row through
   `(1,-1)`; the third north pencil is its `rho` image through `(0,1)`.
   Omitting count-one labels would erase the load-bearing fan.
3. **No current imminent after a triangle pair.** Before the response all
   counts are at most two. The two triggers share a row containing only one
   old Attacker stone, so even a label receiving both reaches only count
   three.
4. **Empty-cell intersection rule.** Physical high-window families on
   different fan lines do intersect, but only at Attacker-occupied vertices.
   Their residual grounds are disjoint, and an empty Defender cell cannot
   service two line families at once.
5. **One-contact lower bound.** One Defender cell cannot kill all four
   consecutive-triple windows because their total intersection is the
   occupied triple. A surviving label always has a legal residual trigger.
6. **Future-pair legality.** Each trigger lies in a surviving alive
   count-three label before the future pair. Distinct-line triggers are
   distinct because the line intersection is occupied.
7. **Fan/fan separation.** `G_- intersect G_+` is not literally empty; it is
   the four occupied diamond cells. The conclusion needed is that no *legal
   empty Defender cell* lies in both.
8. **Hub/fan separation.** Infinite central lines intersect, but the finite
   twelve-window fan unions end near the diamond. The coordinate ranges in
   L13.4 are load-bearing.
9. **Universal `P_stock` fork.** In the hub branch the arbitrary first
   Defender pair must miss the entire physical union `H`, not merely the hub
   cell. This preserves the service residuals and future triggers used by the
   focal lower bound.
10. **Local versus reachable loss.** `P_stock` is losing against every action
    once presented. Only fixed-`S_T` reachability is banked; Q2 root forcing
    is not inferred.
11. **`R_1` status.** Equation (74) is a stop certificate. R7.3 is
    conditional and does not prove that minimizing `R_1` supplies a global
    policy.
12. **No silent `S_T'`.** Since every `P_stock` action fails, no repaired
    tie-break is named and no unenumerated reply table is claimed.
13. **R5.4 scope.** Its forced service is a valid negative control but starts
    above the root threshold. It is not promoted to a strict-root theorem.
14. **Initialization fillers.** Count-three transversal cells are played
    first. L1.2 supplies only the unused placements, and Defender
    augmentation cannot recreate a killed high label.
15. **Initialization versus renewal.** `tau_3<=2` is sufficient at one
    Defender epoch. The triangle fan itself demonstrates why later bridge
    promotion still requires a repair theorem.
16. **Evidence status.** Every count, residual, legal support, and hitting
    number is a hand proof labeled `PROVEN`, never machine `VERIFIED`.

## 66. Exact resume point and provenance

### 66.1 Sharpest next question

**FIRST-PLATEAU SAFE-ACTION DECISION [Q3, OPEN].** At the exact diamond
plateau `P_0^pl`, does some legal Defender pair `a` satisfy

`max_b M(P_0^pl+D@a_1+D@a_2+A@b_1+A@b_2)<=2`?           (84)

The actual ray fails. Unlike final `P_stock`, two Defender cells can touch
both known fan regions, so the three-region pigeonhole proof does not decide
(84). A positive answer would give the first exact one-ply rerouting point;
a negative answer would show that every immediate-`TEMPO` tie-break acts too
late and that prevention must occur before the shield, potentially by
sacrificing the raw epoch's strict immediate value advantage.

After (84), the next Q3 obligations are all-response closure of the selected
action, the nested derivative, other shared/nonshared fanouts, general
cross-hull interactions, alternative forced-service entrances to the
transverse seal, and the remaining count-three initialization family. The Q2
problem remains the independent strict-root forcing question. Ancillarily,
the minimum separation for the R5.2 value theorem remains open.

### 66.2 Provenance and no-run record

**Input commit:** `8ac6caaec8668e77e7c4097c12336e0154c73841` on branch
`hunt/gap-raw`. This authoring pass creates no commit (sentence scoped to
the authoring session). **Reviewed/output artifact:**
`fbae2f7ba13fcf8446e134d3d8cdfb7063688510` (blob
`12e91ef709dd7d27037ac87f6cd3641fa7b2f067`, SHA-256
`1758de37f0988b0dd332a692e73dadeb74e0e881842672c03c605a98a03601bb`),
added post-review per round-7 Finding 14.

During authoring, an unrelated concurrent job advanced the observed shared
branch HEAD to `7c09dee43842bdb73cd3fdfc9e144d51b3b9b62f`. A final read-only
name comparison from the input commit showed changes only in unrelated
strategy-stealing artifacts and their prompt records; none of the required
GAP-RAW corpus changed. No concurrent file was opened as evidence or used in
any proof above.

**Required corpus read first, in order, and in full:**

1. `GAP_RAW_PROOF_ROUND4.md` with folded errata, then
   `GAP_RAW_REVIEW_ROUND4.md`;
2. `GAP_RAW_PROOF_ROUND5.md` with binding Section 47 errata, then
   `GAP_RAW_REVIEW_ROUND5.md`;
3. `GAP_RAW_PROOF_ROUND6.md` with binding Section 58 errata, then
   `GAP_RAW_REVIEW_ROUND6.md`.

The needed K3, filler, legality, and service definitions were then consulted
in rounds 2--3. No `STRATEGY_STEALING_*` file was read as evidence.

**File authored:** `GAP_RAW_PROOF_ROUND7.md`.

The test-gated harness, production rules, strict verifier, and Lean sources
were not modified. No Cargo command, Lean build, harness, game/search
executable, generated enumeration, or git commit was run.

## 67. Errata and strengthenings folded from the round-7 hostile review

`GAP_RAW_REVIEW_ROUND7.md` (ultra, reviewed artifact `fbae2f7b`) returned
**SOUND-WITH-MINOR-ERRATA**: no REFUTED or MAJOR finding; R7.1, R7.2,
R7.2.1, and L13.6 all CONFIRMED at their stated scopes. Folds and
confirmations of record:

1. **(Finding 9, MINOR — folded)** The universal `P_stock`
   NOT-REPAIRABLE verdict is now attributed to R7.2's adaptive
   all-pair fork rather than to R7.1's hub-only consequence; R7.1's
   closing sentence is qualified to "not repairable by hub
   pre-emption," and the §64 ledger row cites the all-pair basis.
   The verdict itself was always mathematically correct.
2. **(Finding 14, MINOR — folded)** §66.2 now records the landed
   reviewed/output artifact `fbae2f7b` (blob + SHA-256), alongside
   the authoring-scoped no-commit sentence.
3. **(Review completion of record)** The final `P_stock` label census
   was independently completed: exact `(n_1,n_2)=(127,55)`, no count
   at least three — the plateau premises hold with an explicit census
   rather than an inherited one.
4. **(Confirmations of record)** L13.1's demand census is axis- and
   D6-complete; the south/north fan responses are legal, nonterminal,
   and census-exact; L13.3 exhausts every legal next Defender pair
   and the completion chronology; L13.4's three action regions are
   pairwise disjoint in the required physical sense; R7.2's adaptive
   fork covers cross-region and outside pairs; R7.2.1 is independent
   of all five later stock turns; L13.6 handles arbitrary lower stock
   at initialization-only scope; Q1/Q2/Q3 quantifier discipline and
   the local-stop scope statement are honest.

The review's closing unresolved-obstacle list (Q2 root forcing,
pre-`P_stock` Q3 intervention, renewal, amortized invariants,
separation sharpness) is the authoritative open state for round 8.
