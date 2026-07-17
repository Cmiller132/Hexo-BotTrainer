# GAP-RAW Proof Round 4 — replacement-invariant research

**Worktree:** `hunt/gap-raw` at input HEAD
`0f7e9405088e7a4a43005e2a271cb17a3d8fa6c3`  
**Date:** 2026-07-17  
**Disposition:** a non-dominating promotion-tempo account is **PROVEN** as an
exact one-turn service-demand factorization; its global repair/renewal is
**OPEN**. A broad remote-local invariant class and the zero-grade-contact
strategy class are ruled out by exact **PROVEN** countertheorems. GAP-RAW and
GAP-REPLACEMENT-INVARIANT remain **OPEN**.

This document resumes the numbering, definitions, and status discipline of
`GAP_RAW_PROOF_ROUND3.md`. In particular, every position below uses the finite
blanket Maker–Breaker state space of round 2; `I`, `tau`, `Phi`, `Theta_2`,
`B_2`, `kappa_2`, service, handoff, and ripeness retain their earlier meanings.
The round-3 root `P_*` and its exact transitions are not modified.

No Cargo command, Lean build, or other machine proof/search was run. Every new
result below is a hand proof. No new harness case was added, so there is no new
`[UNRUN]` case and no new `VERIFIED` claim.

## 24. Executive verdict and route

### 24.1 Route taken

The main route is **Route A**, with a sharp Route-B/impossibility boundary.
Raw mass is replaced by an interaction- and deadline-sensitive quantity:

- `tau(P)` is the deadline-zero demand that must be serviced now;
- after the same actual service pair, `TEMPO(Q)` is the exact maximum service
  demand that the next two Attacker placements can create; and
- `M(P)` is the minimum `TEMPO` obtainable by an actual ordered servicing
  pair at `P`.

`TEMPO` is not additive and does not dominate `Theta_2`. It assigns no global
penalty to arbitrarily many separated pure count-two components. Instead, it
keeps a two-trigger tier inside each interaction component and adds only the
two largest one-trigger component demands, exactly matching the Attacker's
two-placement turn.

Under the user's ranked outcomes, this round achieves mode **(c)** through
R4.3 and R4.7.1, and a narrowly bounded mode-**(b)** partial through R4.8.
It does **not** achieve mode (a): initialization and perpetual renewal of the
tempo account remain open.

### 24.2 New proved results

**Theorem R4.1 (exact promotion-tempo factorization) [PROVEN].** At every
nonterminal Attacker-`FirstStone` handoff `Q` with `I(Q)=empty`,

`TEMPO(Q) = max_b tau(Q+A@b_1+A@b_2)`,

where the maximum is over all legal ordered Attacker pairs. Consequently,

`TEMPO(Q)<=2` if and only if `Q` is unripe.

The definition and proof are in §25. This gives a correct formal expression
of why the round-3 successor of `P_*` is fine even when `Theta_2>1`.

**Theorem R4.2 (same-pair defender form) [PROVEN].** For a Defender epoch `P`,
`M(P)<=2` if and only if one legal ordered reply both services `I(P)` and
hands over an unripe position. The minimizing pair is the same actual pair in
both clauses; no statewise action is conjoined with a different strategy.

**Theorem R4.3 (remote-third-component necessity) [PROVEN].** No state
predicate can simultaneously cover every `Phi<1` root, imply `tau<=2`, and
be remote-2-saturating—that is, unable to distinguish two sufficiently
separated copies of a component from three. This excludes max-local,
fixed-radius universal/conjunctive, and top-two-only *current-demand*
invariants at their exact defined scopes, whether or not they dominate
`Theta_2`.

**Theorem R4.4 (zero-grade-contact strategies lose) [PROVEN].** There is an
exact `Phi=1/sqrt(3)<1` root with three count-three labels and `I=empty` such
that every initial Defender pair killing zero graded mass permits one legal
Attacker pair to create `tau=3`. Thus a forever strategy must sometimes
pre-empt non-imminent stock. The needed statistic is shared-trigger
congestion, not raw count-three mass: one Defender cell at the common trigger
kills all three labels.

**Theorem R4.5 (a high-stock promotion cascade is an Attacker win) [PROVEN].**
An exact three-gadget position with `I=empty` and `tau=0` is a forced Attacker
win against every Defender pair. Its potential is
`1/3+sqrt(3)>1`, so this is not a GAP-RAW refutation. It isolates the Route-C
remainder: force such congestion from a `Phi<1` root despite causal
pre-emption.

### 24.3 Exact boundary

**GAP-TEMPO-INITIALIZATION [OPEN].** Prove `M(P_0)<=2` for every normative
`Phi(P_0)<1` root. K1 and K2 settle the `tau=2` and `tau=1` slices because
`Theta_2(P_0)<1`; the `tau=0` slice retains the general K3 geometry.

**GAP-TEMPO-REPAIR [OPEN].** Construct one strategy whose same actual
servicing choice keeps `M<=2` after every legal Attacker response, including
epochs reached after safe `Theta_2>=1` excursions.

Those two named gaps are the exact missing renewal steps. Therefore
GAP-REPLACEMENT-INVARIANT remains **OPEN**, and GAP-RAW remains **OPEN**. No
Attacker win from a `Phi<1` root against *every* Defender strategy is claimed.

## 25. The exact interaction/deadline account

### 25.1 Label components at an Attacker handoff

Fix a finite nonterminal Attacker-`FirstStone` position `Q` with
`I(Q)=empty`. Let

`L_23(Q)={W: W is alive at Q and count_Q(W) in {2,3}}`.

Give `L_23(Q)` the intersection graph in which two labels are adjacent exactly
when their physical six-cell windows intersect. Let `C` range over its
connected components and put

`E(C)=union_{W in C} E_Q(W)`.

Distinct components have disjoint physical cell unions. Indeed, a shared cell
would itself be an intersection edge. In particular, their residual ground
sets are disjoint.

For a component `C` and a set `T` of one or two distinct cells from `E(C)`,
define

`F_C(T)={W in C: count_Q(W)+|T intersect E_Q(W)| in {4,5}}`.

Give each `W in F_C(T)` the post-trigger residual `E_Q(W)\T`, and let
`r_C(T)` be that residual family's hitting number, with the empty family
having value zero. No residual is empty: `Q` has count at most three and at
most two new Attacker stones are inserted, so a listed label has final count
four or five, never six.

Every cell of `E(C)` is legal for Attacker at `Q` by round 2's L6_2. Hence the
following maxima are over actual legal trigger effects, not over remote virgin
cells:

`h_C=max_{T subset E(C), |T|=1} r_C(T)`,

`g_C=max_{T subset E(C), |T|=2} r_C(T)`.

The finite maxima exist. Put `max_C g_C=0` when there is no component, pad the
`h_C` list by zeros, let `h_(1)>=h_(2)` be its two largest values, and define

`TEMPO(Q)=max(max_C g_C, h_(1)+h_(2))`.                    (20)

This is a state quantity, but it is path-compatible: an account may retain
ages or credits in addition to (20). Equation (20) deliberately records only
the exact next-turn promotion demand.

### 25.2 Exact decomposition

**Lemma L10.1 (two-trigger component decomposition) [PROVEN].** For every
legal ordered Attacker pair `b=(c_1,c_2)` from `Q`, all newly imminent labels
lie in at most two components of `L_23(Q)`.

If both triggers affect one component `C`, their service demand is at most
`g_C`. If they affect two distinct components `C,D`, only pre-count-three
labels mature in either component and the total demand is the sum of the two
one-trigger demands.

*Proof.* A virgin or pre-count-one label has count at most three after two
placements and cannot enter `I`. Every new imminent label was therefore in
`L_23(Q)` and contains at least one trigger.

One trigger cannot affect labels in two components: those labels would share
the trigger cell and hence be adjacent. If a pre-count-two label contains both
triggers, it intersects every affected label at the corresponding trigger, so
all those labels lie in one component. Thus, when two distinct components are
affected, each receives exactly one trigger and only its pre-count-three
labels can mature. Distinct component residual sets are disjoint, so hitting
numbers add. ∎

**Lemma L10.2 (`g_C` includes a one-trigger effect) [PROVEN].** For every
component, `g_C>=h_C`.

*Proof.* If `h_C=0` there is nothing to prove. Otherwise take a maximizing
trigger `c` and one matured count-three label `W` through it. Choose
`d in E_Q(W)\{c}`. Such a distinct empty cell exists, and the pair is legal.
Every label matured by `c` remains matured after `d`; its residual is unchanged
or shrinks, and additional labels may mature. A hitting set for the enlarged,
shrunk family also hits the former family, so neither operation can lower the
hitting number. Hence some two-trigger set has demand at least `h_C`. ∎

**Proof of Theorem R4.1.** Lemma L10.1 gives the upper bound in (20): one
affected component costs at most its `g_C`; two cost at most the sum of the
two largest `h_C` values. A pair with only one relevant trigger is covered by
Lemma L10.2.

For the reverse bound, take a two-cell set attaining some `g_C`. Both cells
are already legal at `Q`, and no other component contains either cell, so
playing them realizes that component demand. If two actual components attain
the two largest `h_C` values, their maximizing singleton triggers are distinct,
already legal, and may be played as one ordered pair; disjoint residual grounds
make their hitting numbers add. If there is only one actual component, the
synthetic `h_(2)=0` gives `h_(1)+h_(2)=h_(1)<=max_C g_C` by L10.2, so the
already attained `g` arm covers it. Thus both terms in (20) are attainable, and

`TEMPO(Q)=max_b tau(Q+A@b_1+A@b_2)`.

Round 2's definition of ripe is exactly the existence of a pair with final
`tau>=3`. The displayed equality therefore gives
`TEMPO(Q)<=2` exactly when `Q` is unripe. ∎

### 25.3 Defender monotonicity

**Lemma L10.3 (Defender augmentation does not increase TEMPO) [PROVEN].** Let
`Q=(A,D,Attacker,FirstStone)` and
`Q'=(A,D',Attacker,FirstStone)` be finite nonterminal positions with
`D subset D'` and `I(Q)=I(Q')=empty`. Then

`TEMPO(Q')<=TEMPO(Q)`.

*Proof.* Added Defender stones only delete alive labels; they do not change the
Attacker count or residual of a surviving label. Consider a legal Attacker pair
at `Q'`. If both triggers are relevant—each contributes to some newly
imminent label—each lies in a surviving count-two/count-three window and was
already legal at `Q` by L6_2; the same pair at `Q` creates a superfamily, so
its demand is no smaller.

If exactly one trigger is relevant, its one-trigger demand at `Q'` is at most
the corresponding one-trigger demand in the surviving-label superfamily at
`Q`. Lemma L10.2 supplies a fully legal two-trigger pair at `Q` with at least
that demand. If neither trigger is relevant, the demand is zero. Thus every
`Q'` demand is bounded by some demand already counted by `TEMPO(Q)`. Taking
the maximum in R4.1 proves the claim. ∎

## 26. The same-pair Defender invariant

### 26.1 Current and next-clock demands

At a Defender epoch `P`, let `Serv(P)` be the finite set of legal ordered
two-cell replies that service `I(P)`. The set includes nonminimum covers and
all sequentially legal fillers. For `a=(x_1,x_2) in Serv(P)`, write

`Q_a=P+D@x_1+D@x_2`.

Every such `Q_a` has `I(Q_a)=empty`. Define

`M(P)=min_{a in Serv(P)} TEMPO(Q_a)`,                       (21)

and set `M(P)=infinity` when `Serv(P)=empty`.

**Proof of Theorem R4.2.** By Theorem R4.1,
`TEMPO(Q_a)<=2` exactly when the actual handoff of `a` is unripe. Minimizing
over the same ordered pairs that service `I(P)` proves that `M(P)<=2` exactly
when one pair performs both tasks. ∎

The candidate Defender-epoch invariant is therefore the two-clock condition

`V_T(P): tau(P)<=2 and M(P)<=2`.                            (22)

The first coordinate is current global service demand. The second is the
least next-turn promotion demand after the same actual service. The top-two
operation in (20) is used only after current imminents have been serviced; it
does not forget a third current urgent component.

### 26.2 Exact deterministic policy and conditional assembly

Define `S_T` on every finite nonterminal Defender history as follows.

1. At Defender `FirstStone`, if `Serv(P)` is nonempty, compute a pair
   minimizing the finite tuple `(TEMPO(Q_a),x_1,x_2)`, where cells use
   lexicographic axial order. Otherwise compute the lexicographically first
   sequentially legal pair. Output its first cell `x_1`.
2. At the successor history obtained by that prescribed `D@x_1`, output the
   precomputed `x_2`.
3. At any other Defender-`SecondStone` history, output the lexicographically
   first legal cell. This off-policy fallback makes `S_T` a total pure
   strategy but carries no survival claim.

The fillers and order are part of this one rule; there is no later existential
choice.

**Theorem R4.6 (tempo conditional assembly) [PROVEN].** Fix a normative root
`P_0` with `Phi(P_0)<1`. If `M(P_0)<=2` and, against every legal Attacker continuation, every
later Defender epoch `P'` reached under `S_T` again satisfies `M(P')<=2`, then
`S_T` blocks forever from `P_0`.

*Proof.* The finite value `M(P_0)<=2` already implies that a servicing pair
exists. The minimizing actual pair has `TEMPO<=2`, so its
handoff is unripe by R4.1. Every Attacker response therefore returns an epoch
with `tau<=2`. The assumed `M<=2` closure lets the same argument repeat.
Thus `S_T` actually services every reached epoch. Theorem A2 proves perpetual
survival. ∎

This theorem is conditional. Neither `GAP-TEMPO-INITIALIZATION` nor
`GAP-TEMPO-REPAIR` is silently included in its conclusion.

## 27. Mandatory regressions

### 27.1 `P_*` and repeated fresh launches

**Lemma L10.4 (pure count-two ocean) [PROVEN].** If every label in
`L_23(Q)` has count two, then `TEMPO(Q)<=2`, independently of the number of
labels, components, or total `Theta_2` mass.

*Proof.* Every `h_C` is zero because one trigger cannot promote count two to
count four. Round 2's R7.4_2 says that no family of pre-count-two labels can
acquire hitting number three from two triggers, even with legality dropped.
Thus every `g_C<=2`, and (20) gives `TEMPO<=2`. ∎

This immediately passes round 3's counterexample. The handoff after any
Defender pair at `P_*` has only surviving count-two target labels in pairwise
disjoint components, hence `TEMPO<=2`. The untouched fresh adjacent launch
returns a Defender epoch with five more count-two labels and count-one pencil
labels, but nothing of count three. At that epoch `I=empty` and `M<=2`, because
every subsequent Defender augmentation hands over a pure-count-two graded
family covered by L10.3–L10.4. This remains true when the post-launch
`Theta_2` is at least `11/9`.

The repeated-launch audit has two distinct meanings which should not be
conflated.

1. **Unbounded gross injection.** With arbitrarily many separated anchored
   sites, Attacker can inject the exact `5/9` fresh-pencil charge on
   arbitrarily many turns. L10.4 remains insensitive to the cumulative total.
2. **Live-stock response.** For a normalized adjacent pair at Q-coordinates
   `0,1`, Defender cells at `-1` and `2` hit all five common Q-windows (indeed
   all seven Q-windows touched by either endpoint). They leave exactly the
   twenty-four off-axis count-one windows. Thus pure repeated `P_*` launches
   do not force live count-two accumulation against that chase; they do force
   arbitrarily long repeated `5/9` excursions and can leave unbounded latent
   count-one stock. If Defender declines the chase and count-two pencils do
   accumulate, L10.4 still accepts them.

The second point corrects a possible overreading of “unbounded
accumulation”: the unavoidable quantity on this exact chase is latent
count-one stock, not necessarily live `Theta_2` mass.

### 27.2 The local fresh-pair trilemma

Let the fresh adjacent endpoints be `a=(0,0)`, `b=(1,0)`. Let `U_Q` be the
union of their five common Q-windows, and put

`p=(0,1)`, `p'=(1,-1)`.

The ten count-one labels promoted by `A@p` are the R-pencil through `a,p`
and the QR-pencil through `b,p`. The analogous ten labels for `p'` are the
R-pencil through `b,p'` and the QR-pencil through `a,p'`.

**Lemma L10.5 (fresh-pair channel separation) [PROVEN].** After deleting the
occupied endpoints, the three physical unions `U_Q`, `U_p`, and `U_p'` are
pairwise disjoint. Consequently every two-cell Defender reply has this exact
dichotomy:

- if it hits any one of the five count-two Q-labels, at most one transverse
  union is touched, so the other transverse center promotes all ten of its
  focal count-one labels; or
- if it touches both transverse unions, it hits none of the five Q-labels.

*Proof.* The two transverse unions lie on the four lines `q=0`, `q=1`,
`q+r=0`, and `q+r=1`. Between `U_p` and `U_p'`, the only cross-intersections
are `a` and `b`; the within-union intersections are respectively `p` and
`p'`. Each transverse union intersects the Q-row `U_Q` only at `a,b`. Those
cells are already occupied by Attacker, so the empty response channels are
disjoint. Two Defender cells can meet at most two of the three channels. The
stated promotion conclusion follows from the five-window adjacent-pair pencil
count. ∎

This does not prove an Attacker win. It identifies the scheduling choice a
replacement strategy must make: erase the visible count-two pencil and leave
a ten-label latent promotion, or seal both transverse promotions while
tolerating the `5/9` pencil. `TEMPO` calls both immediate handoffs unripe and
charges the stock only when it enters a one- or two-trigger tier.

### 27.3 The round-2 O1-prime position and genuine unbounded count-two stock

At the round-2 O1-prime Defender epoch, `I=empty` and `M=0`: every Defender
reply hands over an `L_23`-empty position with `TEMPO=0`. Playing an untouched
center promotes ten focal labels to count two but creates no imminent label.
The ten labels split into two adjacent-pair pencils, one on each of two axes.
Within one pencil the five four-cell residuals are

`{-4,-3,-2,-1}`, `{-3,-2,-1,2}`, `{-2,-1,2,3}`,

`{-1,2,3,4}`, `{2,3,4,5}`.

Their hitting number is two: the extreme residuals are disjoint, while
`{-1,2}` hits all five. The two axis pencils meet only at the occupied center,
so one empty Defender cell hits labels on only one axis and at most four of
its five labels. Hence any immediate two-cell cleanup leaves at least two
focal count-two labels; one maximum-hit cell per axis attains that bound.

The following quantified iteration is useful because, unlike the exact
`P_*` chase, it forces unbounded live count-two stock.

For exact coordinates in the iteration, for `i=0,...,N-1` put
`b_i=(30i,0)`, take supports `b_i+(1,0)` and `b_i+(0,-1)`, and let `U_i` be
the union of the ten focal Q/R windows through the empty center `b_i`. Put
`f_i=b_i+(2,1)` and `R_i=U_i union {f_i}`. The regions `R_i` are pairwise
disjoint. The filler `f_i` is legal, lies on neither focal axis, and shares no
window with the center or either support.

The exact preloaded position is

`P_N^O1=(union_{i=0}^{N-1} {b_i+(1,0),b_i+(0,-1)}, empty, Defender,
FirstStone)`.

It is finite and nonterminal. Its `Phi` is above one; the construction is not
a normative-root claim.

**Lemma L10.6 (iterated preloaded O1 stock) [PROVEN at the O1 domain].** For
every `t>=1`, preload `N>=3t` such gadgets with no Defender cell in any `R_i`.
Against every sequence of Defender pairs, Attacker can activate `t` untouched
centers on successive turns, playing `b_i` then `f_i`, and leave at least

`10t-8(t-1)=2t+8`

focal count-two labels alive at the epoch after the `t`-th activation.

*Proof.* Before Attacker turn `s`, there have been `2s` Defender placements
and `s-1` consumed gadget regions. Because the private regions are disjoint,
fewer than `3s` regions are excluded; with `N>=3t>=3s`, an untouched,
unconsumed region exists. Play its center `b_i` and its filler `f_i`.

Each activation creates ten focal labels. Only the `t-1` intervening Defender
turns can clean already activated gadgets. A Defender cell belongs to at most
one of the two empty axis pencils and kills at most four of its five labels,
so those turns kill at most `8(t-1)` focal labels in total. The displayed
lower bound follows. ∎

The O1 preload has `Phi>1`; this lemma is a mandatory hostile regression, not
a GAP-RAW root theorem. In this exact history, the only collinear Attacker
pairs are a center with one of its two supports: fillers share no window with
them, and different gadgets are farther apart than a window diameter. Thus
every `L_23` label at each resulting Defender epoch has count two, so
L10.3–L10.4 give `M<=2` no matter how large the lower bound becomes.

### 27.4 Both round-2 promotion cascades

The account rejects, rather than averages away, both L8.3.2 counterexamples.

- In the same-axis construction, the local component has a one-trigger
  demand `h=2` at `c`, and the separated remote count-three label has `h=1`
  at `d`. Thus `h_(1)+h_(2)>=3` and `TEMPO>=3`.
- In the mixed cross-axis construction, the horizontal pre-count-two label
  and the three vertical pre-count-three labels all intersect at `c` and lie
  in one component. The displayed pair `(c,d)` has component demand `g>=3`,
  so again `TEMPO>=3`.

These are exact evaluations of the already proved residual families. No
superadditivity or pairwise-overlap inference is used.

## 28. Promotion cascade: exact positive and negative boundaries

### 28.1 A `Phi<1` root that forces proactive grade contact

Let the three axis vectors be

`V={(1,0),(0,1),(1,-1)}`.

For `v in V`, set

`W_v={t v:-3<=t<=2}`,

and put `X=union_{v in V} W_v` and

`A={t v: v in V and t in {-3,-2,-1}}`.

Define the exact finite blocker set

`D={a+t v: a in A, v in V, 1<=|t|<=5}\X`,                 (23)

and let `P_cas=(A,D,Defender,FirstStone)`.

**Lemma L10.7 (three-axis count-three isolator) [PROVEN].** The exact alive
family of `P_cas` is `{W_v:v in V}`. Each label has count three. Therefore

`Phi(P_cas)=Theta_2(P_cas)=3 lambda^{-3}=1/sqrt(3)<1`,

and `I(P_cas)=empty`.

*Proof.* The set (23) is finite and disjoint from `X`, hence from `A` and all
three desired windows. Consider any other window `U` meeting `A`. If its axis
line is one of the three central lines, `X` contains exactly the displayed
six-cell interval on that line, so a distinct six-interval has a cell outside
`X`. On a noncentral axis line, the line is parallel to one central line and
meets each of the other two in at most one cell, so it contains at most two
cells of `X`. Thus `U\X` is nonempty. Choosing `a in U intersect A`, every
cell of `U\X` has the form included in (23), so `U` contains a blocker.
Exactly the three desired labels survive, with their three stated stones. ∎

For an ordered Defender pair `a=(x_1,x_2)`, retain round 3's exact sequential
kill notation

`K(P,a)=kappa_2(P,x_1)+kappa_2(P+D@x_1,x_2)`.

**Theorem R4.4 (zero-grade-contact strategies lose) [PROVEN].** If an initial
Defender pair at `P_cas` has `K(P_cas,a)=0`, the displayed Attacker pair
creates `tau=3`; after the intervening Defender reply, Attacker forces a six
on the following turn.

*Proof.* The zero kill says all three focal labels survive. Attacker plays

`c=(0,0)`, then `y=(1,0)`.

The first cell is in all three alive windows and is legal; the second is
adjacent and also lies in the Q-window. No six is completed. At the next
Defender epoch the focal counts are respectively five on the Q-axis and four
on the other two axes, with residuals

`{(2,0)}`,

`{(0,1),(0,2)}`,

`{(1,-1),(2,-2)}`.                                         (24)

They are pairwise disjoint, so their hitting number is three. Every other
window meeting the old Attacker set contains a blocker, and a window using
only the two new cells has count at most two; hence no omitted label can
invalidate nonterminality or lower the focal hitting number. L1.2 says every
Defender pair misses one member of (24), which Attacker completes on the
following turn. ∎

The construction is label-minimal because `tau(F)<=|F|`: any `tau=3` family
has at least three labels, and (24) uses exactly three. It also gives the exact
necessary initial pre-emption bound

`K(P_cas,a)>=lambda^{-3}=1/(3sqrt(3))`

for every forever-blocking strategy. In contrast, `D@c` kills all three
labels at once; after any legal filler the old `TEMPO` is zero. In account
language the three labels form one component with
`r_C({c,y})=3`, hence `g_C>=3`; meanwhile `M(P_cas)=0`, while every
zero-contact reply has handoff
`TEMPO>=3`.

Thus “ignore all non-imminent mass” is false even at a strict `Phi<1` root.
The proved conclusion is only about the zero-contact strategy class; it is
not an Attacker win against the pre-emptive reply `D@c`.

### 28.2 An exact universal high-stock cascade

The next construction follows the promotion cascade to a real forced win, but
starts outside the normative threshold.

For `i=0,1,2`, put `b_i=(30i,0)` and

`A_i=b_i+{(-4,0),(-3,0),(0,1),(0,2),(0,3)}`.

Let `F_i` contain the horizontal window

`H_i=b_i+{(t,0):-4<=t<=1}`

and the three vertical windows

`V_i^s=b_i+{(0,t):s<=t<=s+5}`, for `s=-2,-1,0`.

Put `Z_i=union F_i`. Let `T_i` be all length-six windows meeting `A_i`, and
define the exact finite blockers

`D_i=union_{W in T_i\F_i}(W\Z_i)`.                         (25)

As in L10.7, every undesired window has a cell outside the displayed cross:
on a central Q/R line only the listed six-subsegments are contained in
`Z_i`, and a QR line meets the cross in at most two cells. Thus (25) kills
every undesired label and none of `F_i`.

Let `P_hi` be the union of the three translated gadgets with Defender at
`FirstStone`. Its exact profile is

`n_2=3, n_3=9`, with all other `n_k=0`.

Hence

`Phi(P_hi)=Theta_2(P_hi)=1/3+sqrt(3)>1`, and `I(P_hi)=empty`.

**Theorem R4.5 (universal high-stock cascade) [PROVEN].** Attacker wins from
`P_hi` against every legal ordered Defender pair.

*Proof.* The three `Z_i` are pairwise disjoint, so two Defender cells leave
one `Z_i` untouched. At that gadget Attacker plays

`c=b_i`, `d=b_i+(1,0)`.

The first cell is adjacent to the vertical stock; the second is adjacent to
the first. The resulting focal residuals are

`b_i+{(-2,0),(-1,0)}`

from `H_i`, and

`b_i+{(0,-2),(0,-1)}`,

`b_i+{(0,-1),(0,4)}`,

`b_i+{(0,4),(0,5)}`

from the three vertical labels. The vertical residual path has hitting
number two, while the horizontal pair is disjoint from its ground set.
Therefore the focal family has hitting number three. The pair is legal and
nonterminal, and L1.2 supplies the forced completion after every reply. ∎

**GAP-CASCADE-REACHABILITY [OPEN].** Force `P_hi`, or a lower-mass shared-label
analogue, from some `Phi<1` root against proactive Defender play. The theorem
above proves the game-theoretic danger of accumulated promotion stock; the
strict root threshold and causal pre-emption remain the exact missing steps
for Route C.

## 29. A sharp remote-local impossibility

### 29.1 Exact count-four isolator

Put

`A_4={(t,0):0<=t<=3}`

and

`D_4={(-1,0),(6,0),(-1,1),(4,-1)}`

`    union {(t,-1),(t,1):0<=t<=3}`.

Let `W_4={(t,0):0<=t<=5}` and let `P_4` be this occupancy at a Defender
epoch.

**Lemma L10.8 (count-four isolator) [PROVEN].** The sole alive label of `P_4`
is `W_4`, at count four.

*Proof.* The Q-axis windows meeting `A_4` have starts `-5,...,3`. The blocker
`(-1,0)` kills starts `-5,...,-1`, and `(6,0)` kills starts `1,2,3`, leaving
only start zero.

On each R-pencil through `(t,0)`, the cells `(t,-1),(t,1)` hit all six
windows. On the QR-pencil through `(t,0)`, the analogous two cells are
`(t-1,1)` and `(t+1,-1)`. For interior `t` they are already among the listed
R-blockers; the two boundary cases are `(-1,1)` and `(4,-1)`. Thus every
off-axis window is killed, while no listed blocker lies in `W_4`. ∎

### 29.2 Remote-2-saturating predicates

For a finite position template `P` and a sufficiently large integer `L`, let
`P_L^[m]` be the union of `m` translates by `j(L,0)`, `j=0,...,m-1`, with the
same side and phase. Call a state predicate `V` **remote-2-saturating** when

`V(P_L^[2]) => V(P_L^[3])`                                 (26)

for every template and all sufficiently large separations.

Call `V` **root-covering** when it holds at every normative `Phi<1` root, and
**service-sufficient** when `V(P)` implies `tau(P)<=2` at every finite
nonterminal Defender epoch.

**Theorem R4.3 (remote-third-component necessity) [PROVEN].** No state
predicate is simultaneously root-covering, service-sufficient, and
remote-2-saturating.

*Proof.* Let `L_0(P_4)` be the separation threshold supplied by remote-2
saturation, and choose `L>=max(30,L_0(P_4))`. The copies have no cross-window
interactions. By L10.8 the two-copy position has exact profile `n_4=2`, so

`Phi=2/3<1`.

This is a normative root; root coverage gives `V((P_4)_L^[2])`. Property (26)
then gives `V((P_4)_L^[3])`. The three-copy position has exactly three
count-four labels with pairwise disjoint residual pairs

`{(jL+4,0),(jL+5,0)}`, for `j=0,1,2`.

Hence `tau=3`, contradicting service sufficiency. ∎

For exact scope, define these three state-predicate classes.

1. An **R-local universal predicate** has the form
   `V(P)=for every cell z, F(local_R(P,z))`, where `F` is
   translation-invariant and no global count or absolute coordinate is an
   additional input.
2. A **component-max predicate** forms the physical-window intersection
   components of the full alive-label family of `P`, assigns each component a
   translation-invariant component-intrinsic score `s(C)`,
   unchanged by adding sufficiently separated components, and applies an
   arbitrary fixed predicate only to `max_C s(C)` (plus the common side and
   phase), with no component-count input. The empty maximum has a fixed null
   value.
3. A **component-top-two predicate** is analogous but applies its fixed
   predicate only to the ordered two largest component scores, padding a
   missing score by a fixed null value, again with no other global input.

**Corollary R4.3.1 [PROVEN at these defined classes].** No predicate in any of
the three classes is both root-covering and service-sufficient.

*Proof.* Once the gap between occupied template supports exceeds twice the
fixed radius, adding a third identical copy creates no new local neighborhood
type. A maximum is unchanged, and the ordered top two scores among three
copies equal those among two.
Each class is therefore remote-2-saturating, so R4.3 applies. ∎

This is broader than R3.3 along a different axis: domination of `Theta_2` is
not assumed. It does not exclude globally additive accounts, signed/path
credits, or a predicate with an explicit third-component overflow test.

There is no conflict with (20)–(22). The counterexample has three **current**
imminent components, and the deadline-zero coordinate `tau(P)` rejects it.
`TEMPO` uses two one-trigger order statistics only after service has produced
`I=empty`, where the Attacker physically has only two placements.

## 30. Fixed-positive additive dormant-stock accounts cannot work

### 30.1 Exact cleanup number of a fresh adjacent launch

Normalize a completely Defender-free adjacent pair to

`A={a=(0,0),b=(1,0)}`.

It has thirty-one alive windows: seven Q-axis labels, and four six-window
off-axis pencils (R and QR through each endpoint). Five of the Q labels have
count two; the other twenty-six labels have count one.

**Lemma L10.9 (the count-one stock needs eight cells) [PROVEN].** The hitting
number of the twenty-six count-one residuals is exactly eight.

*Proof.* The two endpoint-exclusive Q-windows have starts `-5` and `1`; their
residuals lie on opposite sides of the pair and are disjoint. Thus they need
two Q-row cells, attained by `(-1,0),(4,0)`.

Each of the four off-axis pencils needs two line incidences: its two extreme
windows through the endpoint have disjoint residuals. The four pencil lines
have only two empty double-incidence intersections,

`p=(0,1)` and `p'=(1,-1)`.

The same-end intersections are the occupied cells `a,b`; the other line pairs
are parallel. Consequently five empty cells supply at most seven of the eight
required line incidences, so at least six off-axis cells are needed. Six
attain the bound:

`(0,1),(0,-4),(2,-1),(1,-1),(-3,3),(1,3)`.

The empty Q-row channel and all off-axis channels intersect only at occupied
endpoints, so their lower bounds add. The displayed eight cells hit all
twenty-six labels. ∎

The whole thirty-one-label family also has hitting number eight: the same two
Q cells hit all seven Q labels, while the six off-axis cells are unchanged.

### 30.2 Universal repeated-launch construction from `Phi=0`

For `j=0,...,N-1`, put

`R_j=100+30j`, `a_j=(0,R_j+8)`,

`c_j=(0,R_j)`, `d_j=(1,R_j)`.

Let the root have `A=empty`, `D={a_j:0<=j<N}`, and Defender at
`FirstStone`. This is a finite, nonempty, nonterminal normative root with
`Phi=0`. Let `V_j` be the union of all thirty-one prospective windows through
`c_j` or `d_j`. The `V_j` are pairwise disjoint and contain no anchor.

**Theorem R4.7 (unbounded dormant components) [PROVEN].** Put
`T=floor(N/3)`. Against every pure Defender strategy, Attacker can make `T`
legal fresh adjacent launches on distinct sites such that, at the epoch after
the last launch, at least

`T-floor(T/4)>=3T/4`                                       (27)

sites still contain an alive count-one label. Throughout this construction
`I=empty` and `tau=0`.

*Proof.* Before Attacker launch `t`, exactly `2t` Defender placements have
occurred and `t-1` sites have been consumed. A Defender cell meets at most one
of the disjoint `V_j`, so fewer than `3t` sites are unavailable. For
`t<=T`, `N>=3t`; choose an unconsumed `V_j` containing no Defender cell. The
anchor makes `c_j` legal at distance eight, and adjacency makes `d_j` legal.
The fresh component contains only two Attacker stones and is nonterminal.

By L10.9, erasing all twenty-six count-one labels from one launched site costs
at least eight later Defender cells in that site's private footprint. There
have been only `2T` Defender placements in total, including placements made
before their eventual sites were chosen. Hence at most
`floor(2T/8)=floor(T/4)` launched sites can be completely erased. This proves
(27). All surviving labels have count one or two, so `I` stays empty. ∎

For the following corollary, give the alive count-one/count-two labels their
physical-window intersection graph. A **dormant component** is a connected
component of this graph that contains a count-one label; write `N_dorm(P)`
for their number. Distinct surviving launch sites in R4.7 lie in disjoint
`V_j`, so each contributes at least one distinct dormant component.

**Corollary R4.7.1 (positive dormant-component charge is impossible)
[PROVEN].** Fix a threshold `B<infinity` and `epsilon>0`. No root-uniform
nonnegative account satisfying

`C(P)>=epsilon * N_dorm(P)`                                                  (28)

can remain below `B` from every `Phi<1` root against every Attacker
continuation.

*Proof.* Choose `N` so that the lower bound in (27) exceeds `B/epsilon` and
apply R4.7 against the account's proposed strategy. Equation (28) forces
`C>B`. ∎

This covers every component-additive, translation-invariant deadline ledger
which gives a fixed positive charge even to a remote count-one component. It
is a genuine normative-root impossibility and is independent of R3.3's
pointwise domination hypothesis.

More generally, if

`C(Q)=sum_C f(type(C))`

is nonnegative and translation invariant, admitting arbitrarily many copies
of a clonable dormant type forces `f(type)=0`; otherwise `N f(type)` exceeds
every fixed bound. A viable account must use zero/asymptotically vanishing
dormant charge, order statistics such as `TEMPO`, signed/path credits, or a
proved reachability cap.

### 30.3 Simply moving the canonical cutoff to count three also fails

Define the naive next tier

`B_3(P)=sum_{W alive, count_P(W)>=3} lambda^{-e_P(W)}`.

Take four remote adjacent pairs

`A_i={(0,30i),(1,30i)}`, for `i=0,1,2,3`,

with no Defender stones, and let

`U_i={(q,30i):-4<=q<=5}`.

Precisely, set

`P_B3=(union_{i=0}^3 A_i, empty, Defender, FirstStone)`.

Initially `B_3=0` and `I=empty`. Any two Defender cells meet at most two of
the disjoint `U_i`. Extend two untouched pairs by playing `(2,30i)` at each.
Each extension creates exactly four count-three Q-windows, with starts
`-3,-2,-1,0`. Thus the next Defender epoch is nonterminal and has

`B_3=8/(3sqrt(3))=8sqrt(3)/9>1`, while `I=empty` and `tau=0`.

**Corollary R4.7.2 (no grade-three subunit renewal) [PROVEN on the stated
accumulated-stock domain].** There is no Defender rule renewing `B_3<1` from
every `I=empty, B_3<1` position against every Attacker pair.

The displayed start has very large `Phi` and is not a normative root. This
corollary refutes only the tempting “drop count two and reuse J” account. It
does not refute GAP-RAW or a strategy-reachable path credit.

## 31. A bounded direct-strategy partial

The fresh-pair trilemma suggests sealing transverse promotion centers while
tolerating the collinear count-two pencil. The following exact bounded check
shows that this is not immediately defeated by two remote axial returns.

### 31.1 Sealed pencil

Normalize one fresh pair to `A={(0,0),(1,0)}` and play

`D@(0,1), D@(1,-1)`.

These are the two transverse centers from L10.5. They kill five labels in
each of the four off-axis pencils and no Q-axis label. The exact surviving
graded profile in the isolated launch footprint is therefore

`n_1=6, n_2=5`,

with no higher count: two endpoint-exclusive Q count-one labels, one extreme
count-one label in each off-axis pencil, and all five common-Q count-two
labels.

Suppose Attacker later plays the positive axial extension `(2,0)`, producing
the consecutive triple `{0,1,2}`. Its four count-three Q-windows have starts
`-3,-2,-1,0`.

**Lemma L10.10 (one-cell axial stabilization) [PROVEN].** The Defender cell
`(-2,0)` kills the first two count-three windows. The surviving relevant
Q-labels have residuals

`{-1,3,4}` and `{3,4,5}` at count three,

and `{3,4,5,6}` at count two. For that component, `h<=1` and `g<=2`.

*Proof.* A single future trigger either matures at most one count-three label,
or is `3` or `4` and matures both. At trigger `3` their residual pairs share
`4`; at trigger `4` they share `3`. Thus `h<=1`.

For two triggers, the only way to remove both shared central cells is
`{3,4}`. It leaves residuals `{-1}`, `{5}`, and `{5,6}`, hit by
`{-1,5}`. Every other pair leaves a common `3` or `4` for the two high labels,
or matures at most one of them; adding the count-two label requires both
triggers and still gives a two-cell cover. Hence `g<=2`. ∎

The negative axial extension at `-1` is the reflection: `D@(3,0)` gives the
same conclusion.

**Theorem R4.8 (two remote sealed-pencil returns) [PROVEN for the stated
bounded history class].** Let `Q` be an Attacker-`FirstStone` handoff with
`I(Q)=empty` whose entire alive-label family is exactly the disjoint union of
finitely many translated or reflected normalized sealed-pencil profiles from
§31.1. Require the enlarged physical footprints containing every alive label
after either allowed adjacent axial extension and its stabilizer to be
pairwise disjoint.

Suppose the legal Attacker pair consists exactly of one adjacent axial
extension in each of two distinct pencils. Then Defender can play the
corresponding one-cell stabilizer in each activated component and hand over a
position with `TEMPO<=2`.

*Proof.* The exact-alive-family premise ensures that an extension promotes no
unaudited external count-one label into the activated footprint. L10.10 gives
`h<=1,g<=2` for its complete post-stabilizer `L_23` component. Untouched sealed
pencils are pure count two in their graded tier, so L10.4 gives `h=0` and
`g<=2`. The two largest one-trigger demands sum to at most two, and every
within-component demand is at most two. Equation (20) proves the claim. ∎

This is a direct-construction partial, not a perpetual strategy. The exact
unproved extensions are:

- nonadjacent or two-stone returns within one sealed pencil;
- returns through the six surviving count-one extremes;
- one placement coupling several promotion centers;
- cross-axis interactions between formerly separate components; and
- combining a mandatory current service with transverse sealing or axial
  stabilization in the same two-cell Defender turn.

Those cases, rather than the already controlled two-remote-axial-return
history, are part of `GAP-TEMPO-REPAIR`.

## 32. Initialization, repair, and conditional completion

### 32.1 What the root threshold already proves

At every normative root, `Theta_2<=Phi<1`; L9.6 gives `tau<=2`.

- If `tau=2`, K1 says every legal two-cell cover hands over an unripe
  position. Hence `M<=2`.
- If `tau=1`, K2 says every chosen one-cell cover has a legal second cell
  producing an unripe handoff. Hence `M<=2`.
- If `tau=0`, both cells are pre-emptive. Round 2's general K3 suppression is
  still open. The root `P_cas` is one solved instance—`D@(0,0)` kills all
  three labels—but no universal `Phi<1` theorem follows from that example.

Thus `GAP-TEMPO-INITIALIZATION` is exactly concentrated in the free-pair
`tau=0` geometry; it is not a renamed demand to renew `Theta_2<1`.

### 32.2 What renewal must prove

At a later epoch, `Theta_2` may be arbitrarily large while `M<=2`; L10.4 and
the exact accumulated-stock family L10.6 show why this is necessary. R4.7
separately forces arbitrarily many dormant count-one components. Conversely,
`P_hi` has `I=empty` but `M>=3`, so the invariant does not declare arbitrary
non-imminent stock safe.

The exact repair obligation is strategy-bound:

> At every reached epoch `P` with `tau(P)<=2` and `M(P)<=2`, the one actual
> servicing/pre-emptive pair selected by the strategy must have
> `TEMPO(Q)<=2` and must ensure `M(P')<=2` for every legal Attacker response
> `P'`.

Theorem R4.6 then assembles perpetual service. Proving only that some pair
services, some possibly different pair minimizes `TEMPO`, or a third pair
repairs each successor would repeat the existential-witness defect repaired
in round 2.

### 32.3 Why this is a genuine replacement for canonical J

The new account changes all three features that killed J:

| Feature | Canonical `B_2=Theta_2` | Two-clock tempo account |
|---|---|---|
| Remote count-two copies | summed at `1/9` each | zero one-trigger demand; two-trigger demand capped componentwise |
| Two separated promotions | raw masses summed | only the two attainable one-trigger demands are added |
| Three current urgent components | counted indirectly by mass | rejected exactly by current `tau` |
| Shared-trigger congestion | overcounted or hidden in subtotal | retained inside `g_C` and exact hitting geometry |
| Safety implication | `B_2<1 => tau<=2` | `tau<=2` now, `TEMPO<=2` next turn |

R3.3 does not apply because `TEMPO` can be zero below arbitrarily large
`Theta_2`. R4.3 does not apply because (22) retains global current `tau` and
does not saturate after a third imminent component. R4.7 does not apply
because dormant components receive no fixed positive charge.

None of those escapes proves the two open closure gaps.

## 33. Authoritative round-4 status ledger

| Claim | Status | Exact basis / scope |
|---|---|---|
| GAP-RAW | **OPEN** | No perpetual Defender strategy and no `Phi<1` universal Attacker win |
| GAP-REPLACEMENT-INVARIANT | **OPEN** | Exact account supplied; initialization and renewal not closed |
| R4.1 exact `TEMPO` factorization | **PROVEN** | L10.1–L10.2, §25 |
| L10.3 Defender monotonicity | **PROVEN** | Surviving-label subfamilies plus relevant-trigger legality, §25.3 |
| R4.2 `M<=2` same-pair characterization | **PROVEN** | Definition (21) and R4.1, §26.1 |
| Candidate two-clock invariant `V_T` | **OPEN** | Root/repair closure not proved |
| R4.6 tempo conditional assembly | **PROVEN** | Same `S_T`, induction, Theorem A2, §26.2 |
| L10.4 pure count-two ocean | **PROVEN** | Round-2 R7.4_2 plus (20), §27.1 |
| P* / repeated-launch regression | **PROVEN** | Exact pencil arithmetic; no new machine evidence, §27.1–§27.2 |
| L10.6 iterated O1 stock | **PROVEN** | O1 domain only; disjoint-region pigeonhole and `4+4` cleanup ceiling, §27.3 |
| Same-axis and mixed cascade detection | **PROVEN** | Exact round-2 residuals evaluated in (20), §27.4 |
| R4.4 zero-grade-contact strategy obstruction | **PROVEN** | Exact `Phi=1/sqrt(3)` root and residuals (24), §28.1 |
| R4.5 universal high-stock cascade | **PROVEN** | Exact three-gadget forced win outside `Phi<1`, §28.2 |
| GAP-CASCADE-REACHABILITY | **OPEN** | Strict root threshold plus causal pre-emption missing |
| R4.3 remote-third-component necessity | **PROVEN** | Two-copy `Phi=2/3` isolator versus three-copy `tau=3`, §29 |
| L10.9 fresh count-one cleanup number eight | **PROVEN** | Exact line-incidence lower bound and cover, §30.1 |
| R4.7 unbounded dormant components | **PROVEN** | `Phi=0` roots, every Defender strategy, §30.2 |
| Positive dormant-component account impossibility | **PROVEN** | R4.7.1 at exact inequality (28) |
| No grade-three subunit renewal on every accumulated-stock start | **PROVEN** | Four-pair universal reply, §30.3 |
| R4.8 sealed-pencil/two-return partial | **PROVEN** | Stated bounded histories only; L10.10 and component formula, §31 |
| GAP-TEMPO-INITIALIZATION | **OPEN** | `tau=1,2` closed by K1/K2; free-pair `tau=0` remains |
| GAP-TEMPO-REPAIR | **OPEN** | Same-pair all-response renewal of `M<=2` |
| New machine verification | **none** | No Cargo/Lean; no `[UNRUN]` test added |

For literal inventory, the R4.1 row consolidates L10.1–L10.2; the P*/launch
row consolidates L10.5; the R4.4 row consolidates L10.7; the R4.3 row
consolidates L10.8 and R4.3.1; the R4.7/account rows consolidate R4.7.1; the
grade-three row is R4.7.2; and the R4.8 row consolidates L10.10. Every named
round-4 proposition is therefore represented.

No round-2 or round-3 `PROVEN`/`VERIFIED` result is downgraded. Canonical J,
canonical renewal, and pointwise-dominating subunit accounts remain refuted at
their round-3 scopes.

## 34. Hostile-review attack surface

1. **Component relation.** It is intersection of physical six-cell windows,
   not residual intersection and not distance between arbitrary stones.
   Distinct components therefore have genuinely disjoint residual grounds.
2. **Legality of maximizing triggers.** Every trigger used in `h_C` or `g_C`
   lies in an alive window and is legal by L6_2. A newly legal virgin frontier
   cell cannot create an imminent window in the same two-placement turn.
3. **One relevant trigger.** L10.2 is needed; without it the `max g_C` term
   would not automatically cover a pair with one irrelevant filler.
4. **Top two versus top three.** Top two is exact only for the next Attacker
   pair after `I=empty`. Current service remains the full global `tau`; R4.3
   catches any invariant that drops that coordinate.
5. **Fresh-launch accumulation.** The axial chase erases the five count-two
   labels but leaves twenty-four count-one labels. R4.7 proves unbounded
   dormant components from `Phi=0`; no prose should claim that pure P*
   launching forces live `Theta_2` growth against every response.
6. **O1 scope.** L10.6 starts in the round-2 preloaded O1 domain with
   `Phi>1`. It is a stress test, not a normative-root theorem.
7. **Exact blocker comprehensions.** In (23) and (25), every undesired touched
   window is proved to contain a cell outside the protected union. Merely
   defining blockers by a set difference would be circular without that
   geometric proof.
8. **Cascade status.** `P_cas` defeats only zero-grade-contact replies; the
   common-trigger reply survives. `P_hi` defeats every reply but has
   `Phi>1`. Neither is a GAP-RAW refutation.
9. **Empty-A roots.** R4.7 uses `A=empty,D!=empty`. Round 2's normative domain
   requires `A union D` nonempty, not `A` itself nonempty; `Phi=0` is valid.
10. **Strict/common strategy quantifiers.** All thresholds remain strict at
    roots. R4.6 fixes one deterministic strategy and its actual sequential
    pair; the two open gaps cannot be discharged with separate witnesses.
11. **No machine status.** Hand enumeration is `PROVEN`, not `VERIFIED`.
    Historical round-2/3 regressions are cited only at their banked scopes.

## 35. Exact resume point

The next round should begin with `GAP-TEMPO-INITIALIZATION`'s `tau=0` slice,
then attack the one-cycle Bellman closure in `GAP-TEMPO-REPAIR`.

The smallest concrete positive program is:

1. start from the transverse seal in §31 rather than the raw count-two chase;
2. classify every return through its six surviving count-one extremes and
   every nonconsecutive axial return;
3. prove that one placement cannot create more than two stabilizer demands
   after mandatory current service, or exhibit the exact violating cascade;
4. if a violating cascade is found, determine whether it can be forced from a
   strict `Phi<1` root, which is `GAP-CASCADE-REACHABILITY`.

The exact state quantity to preserve is (22), not `Theta_2<1`, `B_3<1`, a
positive sum over dormant components, or a top-two-only current score.

## 36. Provenance and no-run record

**Input commit:** `0f7e9405088e7a4a43005e2a271cb17a3d8fa6c3` on branch
`hunt/gap-raw`. This authoring pass created no commit.

**Required corpus read first, in order, and in full:**

1. `GAP_RAW_PROOF_ROUND2.md`;
2. `GAP_RAW_PROOF_ROUND3.md`;
3. `GAP_RAW_REVIEW_ROUND3.md`;
4. `HUNT_REPORT_GAP_RAW.md`.

Conceptual deadline adjacency was then checked against the read-only sibling
`../hunt-census-deep/CENSUS_CANDIDATES.md`. The already proved global-pairing
impossibility in `docs/PROOF_TSS_DEFENDER_ZONES.md` §8 was read so that no
pairing claim was resurrected.

**File authored:** `GAP_RAW_PROOF_ROUND4.md`.

The test-gated harness, production rules, strict verifier, and `tss-lean` were
not modified. No Cargo command, Lean build, test binary, search, or generated
machine enumeration was run. All coordinate/profile/hitting calculations in
this document are hand proofs, and none is cited as `VERIFIED` evidence.
