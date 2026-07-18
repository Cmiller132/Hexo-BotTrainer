# GAP-RAW Proof Round 8 — the first plateau and the Q2 forcing front

**Worktree:** `hunt/gap-raw` at input HEAD
`a8a0b92d641b690b63d43f049d2b4c2fa0d4e9c1`  
**Date:** 2026-07-18  
**Disposition:** **REPAIRABLE-AT-P0.**  At the exact first diamond plateau,
the legal Defender pair

`a^dagger=((0,-1),(1,1))`

has exact one-successor risk two.  The same action has exact risk two at the
next named plateau after the `U^-` stock turn.  Thus the diamond shield alone
is not a local stop state, and one-ply intervention is still available after
one stock turn.  On the independent Q2 front, every first raw-epoch Defender
deviation except the double-corner seal is rerouted to a five-pencil lozenge
plateau with exact `tau=0,M=2`; the double-corner seal is the exact earliest
unresolved branch.  Q2 root forcing, perpetual Q3 renewal, and GAP-RAW remain
**OPEN**.

This document continues the definitions, theorem numbering, and equation
numbering of `GAP_RAW_PROOF_ROUND7.md`, including binding Section 67.  In
particular, the universal final-`P_stock` verdict is attributed to R7.2's
all-pair fork, not to R7.1; the reviewed final census is
`(n_1,n_2)=(127,55)`; and the exact round-6 post-hub values remain `M=4` and
next `tau=4` on their stated history.  The meanings of `I`, `tau`, `TEMPO`,
`M`, `Serv`, `R_1`, and the Q1/Q2/Q3 tags are unchanged.

No Cargo command, Lean build, harness, game/search program, machine
enumeration, or git commit was run.  Every new result below is a hand proof;
there are no new `VERIFIED` or `[UNRUN]` claims.  No
`STRATEGY_STEALING_*` file was read as evidence.

## 68. Executive verdict and quantifier contract

### 68.1 First-plateau answer

Let `P_i^pl`, `i=0,...,5`, retain the six exact plateau meanings from round
7.  Thus `P_0^pl` is the Defender epoch immediately after the diamond shield
and before any hub stock, while `P_1^pl` is the epoch after the first stock
pair `(6,0),(7,0)`.

**Theorem R8.1 (the first plateau is one-ply repairable) [Q3, PROVEN].**  At
`P_0^pl`, the non-hub ordered pair

`a^dagger=((0,-1),(1,1))`                                  (85)

is legal and satisfies

`R_1(P_0^pl,a^dagger)=max_b M(P_{a^dagger,b})=2`.          (86)

The maximum ranges over every legal nonterminal ordered Attacker response,
exactly as in round-6 equation (71).  In particular, equation (84) has a
positive answer.  L14.1 has no count above two, so adding one Attacker pair
cannot complete a six; every legal response in this maximum is automatically
nonterminal.  Moreover,

`B_1(P_0^pl)=2`.                                           (87)

The intervention principle is **dual near-trigger capping**: occupy the two
near triangle triggers exchanged by `rho`, thereby truncating the two
vertical shield pencils and deleting all but one of each bridge-row pencil.
Every immediate Attacker response then has high stock on at most two physical
axes.  In a distinct-axis response, a one-trigger hard consecutive-triple
pencil is reduced to its isolated nested outer tail and every easier high
pencil is deleted; a same-axis family is covered outright with at most two
cells.

**Theorem R8.2 (the cap survives one stock turn) [Q3, PROVEN].**  The same
ordered action (85) is legal at the exact `P_1^pl` and has

`R_1(P_1^pl,a^dagger)=2`.                                  (88)

Thus the intervention principle remains available after `U^-` has been
installed.  This is a two-epoch local theorem, not an assertion that the same
cells can be replayed on one history and not a perpetual renewal theorem.

### 68.2 Q2 outcome

**Theorem R8.3 (complete earliest Q2 branch) [Q2, PROVEN at one-level
scope].**  From the exact round-6 strict `Phi=0` root, Attacker can choose an
untouched launch against every initial Defender pair and reach an exact raw
adjacent-pair epoch.  At that first genuine branch:

1. if Defender's occupancy is not the double-corner set
   `{(0,1),(1,-1)}`, Attacker can form a four-stone lozenge having five
   adjacent-pair pencils, at least three of them intact; the returned epoch
   has exact `tau=0,M=2`;
2. the two orders of `{(0,1),(1,-1)}` give exactly the transverse sealed
   handoff of R5.1, with local profile `n_1=6,n_2=5` and exact
   `TEMPO=2`.

The second class is the exact earliest unresolved Q2 obstruction.  R5.1 says
that every immediate response from it returns `M<=2`, so no inherited local
stop theorem closes that branch.  The lozenge plateaux in the first class are
also not proved local stop states.  Hence strategy-independent root forcing
and GAP-RAW remain **OPEN**.

### 68.3 Renewal outcome

**Corollary R8.2.1 (one honest renewal step) [Q3-repair, PROVEN at the exact
class].**  The exact first two round-6 plateau occupancies each admit an
immediate-value-minimizing action with all-response successor bound `M<=2`.
This supplies one finite renewal step beyond initialization.  It does not
show that the successor class can be reused from an arbitrary member, and it
does not close R7.3's availability hypothesis on arbitrary reached histories.

## 69. The dual near-trigger cap at `P_0^pl`

### 69.1 Exact capped inventory

Write

`s=q+r`,

and retain the four diamond stones

`(0,0),(0,1),(1,0),(1,-1)`.

The four shield lines are `q=0`, `q=1`, `s=0`, and `s=1`.  The existing
local Defender cells are `(-4,0),(2,0)`; all anchors and ray cells lie outside
the finite families used below.

**Lemma L14.1 (exact cap inventory) [Q3, PROVEN].**  After (85), the local
alive profile is exactly

`n_1=8, n_2=12`, with no higher count.                      (89)

The count-two family consists of

```text
s=0: five common windows, q-starts -4,-3,-2,-1,0;
s=1: five common windows, q-starts -4,-3,-2,-1,0;
q=0: the single window 0<=r<=5;
q=1: the single window -5<=r<=0.
```

The count-one family consists of the two endpoint-exclusive extremes on each
diagonal line, the single outer extreme on each capped vertical line, and the
single surviving window on each bridge row:

```text
q=0:  1<=r<=6;                 q=1: -6<=r<=-1;
r=1: -5<=q<=0;                 r=-1: 1<=q<=6,
```

together with the four diagonal extremes.

*Proof.*  Before (85), L12.1 gives five count-two windows on each shield
line and two count-one extremes on each, plus six count-one windows on each
of `r=1` and `r=-1`.  The cell `(0,-1)` kills four of the five `q=0`
common windows, one vertical extreme, and five of the six row-`-1` windows.
It lies on neither diagonal shield line.  Its `rho` image `(1,1)` makes the
identical deletions on `q=1` and `r=1`.  Defender augmentation creates no
label and changes no Attacker count.  The displayed survivors give
`12` count-two and `8` count-one labels exactly.  Both cap cells lie in an
alive count-two window before they are played, so the ordered pair is legal;
the unchanged adjacent Attacker stock keeps the second placement legal after
the first.  ∎

The graded tier after (85) is pure count two.  L10.4 gives
`TEMPO<=2`; either intact diagonal pencil and its two standard flank triggers
give demand two, so the handoff value is exactly `TEMPO=2`.  Since every
action at `P_0^pl` has value two by L12.2, (85) is an immediate-value
minimizer.

### 69.2 Exact one-axis stabilization table

Parameterize either intact diagonal `s=j`, `j in {0,1}`, by its `q`
coordinate.  Its old adjacent Attacker pair is at parameters `0,1`, and its
five common count-two windows have starts `-4,...,0`.

**Lemma L14.2 (one-trigger cap table) [Q3, PROVEN].**  If one response cell
at parameter `t` promotes this pencil, one Defender cell has the following
complete effects:

| trigger `t` | promoted count-three starts | Defender cell | surviving high part |
|---:|---|---:|---|
| `-4,-3,-2` | respectively `{-4}`, `{-4,-3}`, `{-4,-3,-2}` | `-1` | none |
| `-1` | `-4,-3,-2,-1` | `2` | start `-4`, residual `{-4,-3,-2}` |
| `2` | `-3,-2,-1,0` | `-1` | start `0`, residual `{3,4,5}` |
| `3,4,5` | respectively `{-2,-1,0}`, `{-1,0}`, `{0}` | `2` | none |

In either adjacent case the surviving count-three label and the sole
surviving adjacent count-two extreme form the exact nested derivative

```text
negative tail: high {-4,-3,-2} inside low {-5,-4,-3,-2};
positive tail: high { 3, 4, 5} inside low { 3, 4, 5, 6}.
```                                                         (90)

The derivative has exact local `h=g=1`.  A promoted singleton on `q=0` or
`q=1`, and the sole count-three label born from either surviving bridge-row
window, is deleted by one empty residual cell.

*Proof.*  A common window `W_s` contains `t` exactly when
`s<=t<=s+5`; intersecting this condition with `-4<=s<=0` gives the table.
For `t=-1`, the cell `2` kills starts `-3,-2,-1` and leaves only `W_-4`;
the count-two extreme at start `-5` has the larger residual displayed in
(90).  The `t=2` case is its reflection.  At the other depths the displayed
cell lies in every promoted window.  The residual inclusion proves
`h=g=1` exactly as in L11.3.  Every chosen cell is empty, belongs to a current
alive high label, and is therefore legal.  The singleton assertions are
immediate.  ∎

**Lemma L14.2.1 (complete same-axis cover) [Q3, PROVEN].**  Suppose both
response cells lie on one of the four shield central axes.  Whether or not one
old count-two window contains both response cells, at most two empty cells
meet every returned count-at-least-three window, service every current
imminent on that axis, and delete the complete high family.

*Proof.*  Sort the old adjacent pair and the two new axial stones.  Every high
window on the axis contains at least three of those four ranks.  Conversely,
two response cells on that central axis cannot jointly create a high label on
a second physical axis: the line through the two cells is unique, and a
one-trigger promotion on a different shield line would require an empty
intersection of two shield central lines, whereas all such intersections are
old Attacker stones.  Thus the rank-triple transversal of L11.4 is complete,
not merely local to one old window.  It uses at most two empty cells and meets
every feasible rank-triple interval.  This includes separated effects such as
parameters `-4,5`, whose two high windows are hit at `-1,2`.

If a nominal transversal point is an existing Defender cell, every window it
would meet is already dead and that point is simply omitted.  Every remaining
transversal point is empty by the L11.4 rank argument, so the resulting
at-most-two-cell action is legal.

If a common old count-two window contains both response cells, a capped
vertical family has exact `tau=1`.  An intact diagonal family has exact
`tau=2` when the four axial Attacker cells are consecutive and exact `tau=1`
otherwise: in the nonconsecutive case an empty internal gap belongs to every
interval containing all four stones.  If there is no common old count-two
window, no count-four label exists and exact `tau=0`.  In every case the
transversal services `I` and deletes the complete high family; the handoff is
pure count two and has `TEMPO<=2` by L10.4.  ∎

### 69.3 Exhaustive response-axis census

Call a label *high* at the returned epoch when its count is at least three.

**Lemma L14.3 (at most two high axes) [Q3, PROVEN].**  For every legal
Attacker response `b` after (85), either both cells lie on one shield central
axis, as handled by L14.2.1, or the returned epoch has exact `tau=0` and all
high labels lie on at most two physical axes.

*Proof.*  Every old count-two label lies on a shield central axis.  Hence, if
the response cells do not lie on one common shield central axis, no old
count-two label contains both and no such label reaches count four.  An old
count-one label reaches at most three and a virgin label at most two, so
`I=empty` and `tau=0` exactly.

An empty response cell lies on at most one of the four shield central lines,
because every nonparallel intersection of those lines is one of the four old
Attacker stones.  Thus one-trigger promotion supplies at most two shield
axes.  A third high axis could only be an old count-one window containing
both response cells.

The complete count-one inventory is L14.1's eight labels.  A diagonal or
vertical extreme lies on the same central line as its adjacent-pair pencil.
On `r=1`, both triggers must lie in `-5<=q<=-1`; among those cells only
`(-1,1)` lies in the residual union of an intact count-two pencil, namely
`s=0`.  On `r=-1`, only `(2,-1)` similarly lies in the `s=1` pencil.
Therefore a bridge row can coexist with at most one promoted shield pencil.
Equivalently, the only uncapped three-axis pairs in the original diamond are
the south and north pairs

`((0,-1),(2,-1))` and `((1,1),(-1,1))`,

and (85) already occupies the first cell of each.  No legal response creates
three high axes.  ∎

### 69.4 Outer-tail isolation

In physical coordinates the four possible nested high residuals are

```text
R_0^-={(-k,k):       k=2,3,4},   R_0^+={(k,-k):   k=3,4,5},
R_1^-={(-k,1+k):     k=2,3,4},   R_1^+={(k,1-k): k=3,4,5}.
```                                                         (91)

**Lemma L14.4 (no secondary count-two bridge) [Q3, PROVEN].**  After the
L14.2 reply, a future trigger in one of the tails (91) belongs to no
pre-count-two label outside its displayed nested diagonal pair.  If two
tails survive, no pre-count-two label contains one future trigger from each.
Consequently the resulting Attacker handoff has `TEMPO<=2`.

*Proof.*  For `z in R_0^- union R_0^+`, neither off-diagonal diamond stone
`(0,1)` nor `(1,0)` is axis-collinear with `z`; for
`z in R_1^- union R_1^+`, neither `(0,0)` nor `(1,-1)` is axis-collinear
with `z`.  The old adjacent pair and the response trigger on `z`'s diagonal
already account for that local nested pair.  Only one other response stone
exists, so every transverse line through `z` contains at most one
post-response Attacker stone and cannot support a count-two label.

For two surviving tails, the only possibilities are the parallel diagonals
`s=0,1`.  The four side combinations have this complete connector census:

| tails | possible common connector | old Attacker count there |
|---|---|---:|
| `R_0^-,R_1^-` | equal `q` in `[-4,-2]`, or equal `r` in `{3,4}` | `0` |
| `R_0^+,R_1^+` | equal `q` in `[3,5]`, or equal `r` in `{-4,-3}` | `0` |
| `R_0^-,R_1^+` | none | `0` |
| `R_0^+,R_1^-` | none | `0` |

The response stones on the same side have parameter `-1` or `2`, outside
the residual parameter ranges.  Hence no connector is pre-count-two.

Now take an arbitrary future Attacker pair.  If it avoids all tails, every
new imminent comes from pre-count-two labels alone and R7.4_2 gives demand at
most two.  If it uses one tail, the nested pair contributes demand exactly
one; a transverse pre-count-two label cannot also use that trigger.  If it
uses two tails, each contributes one and the connector table excludes an
additional count-two demand.  These cases exhaust the future pair and prove
`TEMPO<=2`.  ∎

### 69.5 Proof and exact risk value

*Proof of Theorem R8.1.*  Fix a legal response `b`.

If both response cells lie on one shield central axis, use L14.2.1's
at-most-two-cell rank-triple cover, after first discarding any redundant cell
and then appending a legal filler if necessary.  This includes the case in
which the two one-trigger effects are too far apart to share an old
count-two window.  The same pair services the exact current family and kills
all high labels, so L10.4 gives `M(P_{a^dagger,b})<=2`.

Otherwise L14.3 gives `tau=0` and at most two high axes.  On each axis use one
cell from L14.2: delete an easy or singleton family, and stabilize an adjacent
diagonal family into its nested outer tail.  Every promoted old count-one
bridge outside a diagonal extreme is one of L14.1's two singleton row labels,
so one residual cell deletes it outright.  Fill an unused placement only
after the effective cells, with the SAFE-FILLER restriction (review round-8
Finding 5): choose the standard max-`q` filler outside every surviving
derivative support (or, for an easy second-axis deletion, a residual cell
away from the at-most-one low-only outer cell of a retained derivative);
if a filler nevertheless kills a derivative's high label, remove that
derivative from the named list. Arbitrary deletion is monotone for the
scalar `TEMPO` bound but does not preserve the exact nested-pair clause of
`C_cap`; the restriction repairs exact class membership without changing
R8.1, R8.2, or the `TEMPO<=2` conclusion.
L14.4 then gives `TEMPO<=2` for the actual serviced handoff.  Thus

`M(P_{a^dagger,b})<=2` for every `b`.                       (92)

For equality, use

`b^dagger=((-1,1),(-1,2))`.                               (93)

It creates four consecutive-triple count-three windows on each of the
parallel lines `s=0,1`, and no count-four label, so the returned epoch has
exact `tau=0`.  If a next Defender pair concentrates on one line, the other
untouched family has demand two by L13.1.  If it uses one cell on each line,
one cell cannot kill all four windows on either line; a surviving
count-three label on each line gives two residual-disjoint singleton demands.
Hence every reply has `TEMPO>=2`.  The pair

`((2,-2),(2,-1))`

leaves the two negative nested derivatives of (90), and L14.4 gives exact
`TEMPO=2`.  Therefore

`M(P_{a^dagger,b^dagger})=2`,                              (94)

which combines with (92) to prove (86).

Finally, every action at `P_0^pl` leaves at least two of the four shield
pencils untouched.  Attacker may play one adjacent exterior trigger in each.
The two resulting consecutive-triple families have exact current `tau=0`.
Every next Defender pair either leaves one family untouched, giving demand
two, or touches each once and leaves one count-three label in each, again
giving two residual-disjoint singleton demands.  Thus every first-plateau
action has `R_1>=2`.  Together with (86), this proves the exact minimum (87),
not merely `B_1<=2`.  ∎

## 70. The one-stock-turn test

### 70.1 Exact inventory after the same cap

At `P_1^pl` the additional Attacker stones are `(6,0),(7,0)`.  The existing
row-zero blockers are still `(-4,0),(2,0)`.

**Lemma L14.5 (exact one-stock capped inventory) [Q3, PROVEN].**  After (85)
at `P_1^pl`, the complete alive profile is

`n_1=33, n_2=16`, with no higher count.                    (95)

Its axis census is

| axis family | count two | count one |
|---|---:|---:|
| `q=0,q=1` | `1+1=2` | `1+1=2` |
| `q=6,q=7` | `0` | `6+6=12` |
| `s=0,s=1` | `5+5=10` | `2+2=4` |
| `s=6,s=7` | `0` | `6+6=12` |
| `r=0` | starts `3,4,5,6`: `4` | start `7`: `1` |
| `r=1,r=-1` | `0` | `1+1=2` |

*Proof.*  L14.1 supplies the diamond contribution.  On `r=0`, the adjacent
stock pair belongs to the five common starts `2,...,6`; `D@(2,0)` kills start
`2` and the left count-one extreme at start `1`.  Starts `3,...,6` and the
right extreme at start `7` survive.  Each stock stone is alone on its fixed
`q` and fixed `s` lines, contributing six count-one labels on each of the four
lines `q=6,q=7,s=6,s=7`.  No cell of (85), no old blocker, and no ray cell
lies in those twenty-four windows.  Summing gives (95).  ∎

The graded tier in (95) is pure count two.  The untouched diagonal shield
pencils give the lower bound two and L10.4 gives the upper bound two, so the
handoff after (85) has exact `TEMPO=2`.  The same untouched-pencil argument
applies after every action at `P_1^pl`; hence (85) is again an
immediate-value minimizer.

The four stock-row count-two residuals are

```text
start 3: {3,4,5,8};       start 4: {4,5,8,9};
start 5: {5,8,9,10};      start 6: {8,9,10,11}.
```                                                         (96)

For a same-row response, the current hitting number contributed by (96) is
exactly two for the coordinate pairs `{5,8}` and `{8,9}`, exactly one for
every other pair co-contained in a displayed residual, and zero otherwise.
The first and last residuals in the two exceptional families are disjoint;
in every other co-contained case a direct intersection of the matured
residuals gives a one-cell cover.

### 70.2 Stock-row stabilizer and bridge audit

**Lemma L14.6 (one-stock stabilization table) [Q3, PROVEN].**  A single
trigger on the live stock-row pencil has this exact table:

| trigger `t` | high starts | Defender cell | surviving high part |
|---:|---|---:|---|
| `3` | `3` | `8` | none |
| `4` | `3,4` | `8` | none |
| `5` | `3,4,5` | `8` | none |
| `8` | `3,4,5,6` | `5` | start `6`, residual `{9,10,11}` |
| `9` | `4,5,6` | `8` | none |
| `10` | `5,6` | `8` | none |
| `11` | `6` | `8` | none |

In the sole hard case, the surviving count-two extreme has residual
`{9,10,11,12}`, so the result is another exact nested derivative with local
`h=g=1`.

*Proof.*  Read containment directly from the four intervals with starts
`3,...,6`.  For `t=8`, the cell `5` kills starts `3,4,5` and leaves start
`6`; the right extreme at start `7` supplies the displayed nested low label.
Every other displayed cell meets every promoted window.  ∎

The central count-two lines are now

`q=0,q=1,s=0,s=1,r=0`.                                  (97)

The P0 bridge census remains complete after adding the following cases.

- For one trigger on `r=0` and one on `q=i`, an old-stone bridge would have
  fixed level `s` in `{0,1,6,7}`.  The row trigger at `(s,0)` is then one of
  the four already occupied diamond/stock cells.
- For one trigger on `r=0` and one on `s=j`, the analogous bridge has fixed
  `q` in `{0,1,6,7}`, again making the row trigger occupied.
- A bridge between parallel `q` or parallel `s` pencils through stock level
  `6` or `7` meets the old shield pencils outside their live activation
  ranges.

Thus two distinct count-two lines in (97) cannot be coupled through a third
old count-one bridge by one legal response pair.

The twenty-four new count-one windows require one further check.  If two
response stones form a consecutive triple with the stock stone on one of
`q=6,q=7,s=6,s=7`, they can create four count-three windows on that singleton
pencil.  On `q=k`, `k in {6,7}`, the new cells have
`r in {-2,-1,1,2}`; on `s=k` they have the corresponding two exterior
parameters.  None lies on another live line in (97).  More generally, the
union of a six-window pencil through `q=6` reaches `s=1` only at parameter
`q=6`, outside the diagonal count-two unions `-4<=q<=5`, and the other three
stock singleton pencils are still farther away.  Hence such a hard
count-one family is the sole hard axis and its at-most-two-cell rank-triple
cover deletes it completely.

It follows exactly as in L14.3 that every response falls into one of two
exhaustive classes.

1. All high labels lie on one physical axis.  The axial rank-triple
   transversal uses at most two cells and deletes that complete family.  On
   one of the five old count-two central axes this is L14.2.1 with the
   corresponding adjacent shield or stock ranks, whether or not the two
   effects share an old window.  On a hard stock-singleton pencil it is the
   same three-rank interval cover.
2. High labels lie on two physical axes.  The returned epoch has exact
   `tau=0`; each axis is then a one-trigger L14.2/L14.6 family or a singleton
   old-count-one family.  The hard stock-singleton audit above excludes that
   two-cell family from this class, so one effective cell per axis suffices.

### 70.3 Extended tail isolation and proof

The stock tail is

`R_U^+={(9,0),(10,0),(11,0)}`.                            (98)

No transverse fixed-`q` or fixed-`s` line through (98) contains an old
Attacker stone: old `q` coordinates and levels are only `0,1,6,7`.  The
other response stone can add at most one.  Therefore no transverse
pre-count-two label contains a future stock-tail trigger.

The diagonal tails (91) remain isolated after the stock addition.  Their
fixed-`q` ranges are `[-4,-2]` or `[3,5]`, their noncentral rows lie outside
`r=0`, and their levels are `0,1`; the stock stones at `q=6,7`, `r=0`, levels
`6,7` supply no second transverse support.  Finally, (98) is axis-collinear
with no cell of a diagonal tail: its `q` range is `9,...,11`, its row is zero,
and its level is `9,...,11`, while a diagonal tail has `q` between `-4` and
`5`, nonzero row, and level zero or one.

Accordingly the future-pair decomposition in L14.4 applies verbatim to any
one or two surviving tails from (91) and (98).  A pair avoiding tails is
pure-count-two and costs at most two; one tail contributes one; two tails
contribute `1+1`, with no count-two connector.

*Proof of Theorem R8.2.*  Use the two exhaustive classes just stated.  In the
one-axis class, the at-most-two-cell axial cover services every current
imminent and deletes the whole high family.  In particular, this covers two
widely separated response cells on one central axis even when no old
count-two window contains both.  In the two-axis class, exact `tau=0` makes
service vacuous and L14.2/L14.6 or the singleton deletion supplies one
stabilizer per axis.  The extended isolation audit gives `TEMPO<=2` for the
resulting actual handoff.  Hence every response returns `M<=2`.

The same response (93) again produces the two full parallel diagonal triple
families.  The lower-bound contact census from R8.1 is unchanged by extra
stock, while `((2,-2),(2,-1))` leaves the same two isolated derivatives.
Thus the returned value is exactly `M=2`, proving (88).  As at `P_0^pl`, two
of the four shield pencils remain untouched after every initial action and
give the universal risk floor two.  Therefore the exact one-ply minimum at
this epoch is also

`B_1(P_1^pl)=2`.                                           (99)

This completes the required one-stock-turn test.  No later plateau is
classified here.  ∎

## 71. Q2: the earliest strategy-independent branch

### 71.1 Absorbing the arbitrary root reply

Normalize one launch to

```text
c=(0,0), d=(1,0), p=(0,1), p'=(1,-1),
x_0=(0,-1), x_2=(2,-1), y_-1=(-1,1), y_1=(1,1),
```

and put

`K={c,d,p,p',x_0,x_2,y_-1,y_1}`.                         (100)

For launch `j`, translate (100) by `(0,30j)` and let `B_j` be the union of
every length-six window meeting that translate.

**Lemma L14.7 (enlarged untouched launch) [Q2, PROVEN].**  The three sets
`B_0,B_1,B_2` are pairwise disjoint and exclude their round-6 anchors.
Consequently every initial Defender pair leaves one `B_j` untouched, and
Attacker can legally reach an exact raw adjacent-pair epoch there.

*Proof.*  Every point of translated `K` has `r` coordinate in
`{30j-1,30j,30j+1}`.  A cell in a six-window through it differs in `r` by at
most five, so

`B_j subset {30j-6<=r<=30j+6}`.

The bands are disjoint at spacing thirty.  The anchor `(0,30j+8)` is outside
its band.  Two Defender cells meet at most two of the three disjoint sets;
choose the untouched one.  The first endpoint is distance eight from its
anchor and the second is adjacent, exactly as in round 6.  Every window
through the adjacent pair belongs to `B_j`, so the local returned family is
the exact raw inventory: five common row windows at count two, twenty-six
other windows at count one, and no higher label.  Thus the returned raw epoch
has exact `tau=0`.  ∎

This proves that the root reply itself is not the first Q2 obstruction.  The
first genuine arbitrary-strategy branch is Defender's pair at the raw epoch.

### 71.2 Complete raw immediate-value census

Write

`W_s={(q,0):s<=q<=s+5}`, `-4<=s<=0`,

and for a legal raw reply `a` let

`S(a)={s:W_s survives a}`.

**Lemma L14.8 (raw effect quotient) [Q2, PROVEN].**  The set `S(a)` is one
consecutive block.  With `k=|S(a)|`, the exact handoff value is

| `k` | number of possible survivor blocks | exact `TEMPO(Q_a)` |
|---:|---:|---:|
| `0` | `1` | `0` |
| `1` | `5` | `1` |
| `2` | `4` | `1` |
| `3` | `3` | `2` |
| `4` | `2` | `2` |
| `5` | `1` | `2` |

In particular, the raw epoch has exact

`tau(P_raw)=0, M(P_raw)=0`.                               (101)

*Proof.*  A Defender cell off `r=0` meets no `W_s`.  A row cell at
`-4,-3,-2,-1` deletes respectively a prefix of `1,2,3,4` starts; a cell at
`2,3,4,5` deletes respectively a suffix of `4,3,2,1`; every other empty row
cell deletes none.  Intersecting a prefix and a suffix leaves one consecutive
block.

One surviving count-two label has exact future demand one.  For two adjacent
starts, any pair maturing both labels occupies two of the three empty cells
in their five-cell physical intersection, leaving the third as a common
residual hit; their exact value is again one.  A block of at least three
contains starts differing by two.  According to its position, one of

`{-2,-1}`, `{-1,2}`, `{2,3}`

matures such a pair and leaves two disjoint residual grounds, giving demand
two.  L10.4 supplies every reverse upper bound.

The value-zero replies are exactly the ten unordered covers

```text
(-4,2);
(-3,2),(-3,3);
(-2,2),(-2,3),(-2,4);
(-1,2),(-1,3),(-1,4),(-1,5),
```

with both orders, hence twenty ordered minimizers.  This proves (101) and
exhausts the infinite legal-pair universe by its finite survivor-block
effect.  ∎

Every value-zero cover kills both endpoint-exclusive row labels as well as
the five common labels, and it meets none of the four transverse central
lines away from the occupied endpoints.  Therefore `A@p,A@p'` after any of
the twenty minimizers gives the exact local L12.1 census

`(n_1,n_2)=(20,20)`, with exact `tau=0,M=2`.               (102)

Thus every raw immediate-value minimizer, not only the lexicographically
first R5.3 pair, reaches a demand-equivalent exact `P_0^pl`.  R8.1 now shows
that this forced minimizing branch is repairable for one ply.

### 71.3 Adaptive lozenge rerouting

**Lemma L14.9 (all non-seal raw replies) [Q2, PROVEN].**  If the raw
Defender occupancy is not `{p,p'}`, Attacker has a legal nonterminal pair
forming a four-stone lozenge with five adjacent-pair pencils, at least three
of them intact.  The returned epoch `P_a^lozenge` has exact

`tau(P_a^lozenge)=0, M(P_a^lozenge)=2`,                  (103)

and every legal next Defender pair has exact handoff `TEMPO=2`.

*Proof.*  Use this exhaustive response rule.

- If neither corner is occupied, play `(p,p')`.
- If only `p` is occupied, play `p'` and whichever of `x_0,x_2` is not the
  other Defender cell.
- If only `p'` is occupied, play `p` and whichever of `y_-1,y_1` is not the
  other Defender cell.

Legality splits by template (review round-8 Finding 3): in each one-corner
template the first placement is adjacent to old Attacker stock and the wing
cell is adjacent to the first response; in the no-corner template
`(p,p')=((0,1),(1,-1))` the two cells are at distance two, and each is
INDEPENDENTLY adjacent to unchanged old stock (`p` to `c=(0,0)`; `p'` to
`c` and `d=(1,0)`), so the ordered pair is legal either way. The five
central lines are exactly

| completion | five adjacent-pair central lines | occupied corner's line |
|---|---|---|
| `p,p'` | `r=0,q=0,q=1,s=0,s=1` | none |
| `p',x_0` | `r=0,r=-1,q=0,q=1,s=0` | `p` on `q=0` |
| `p',x_2` | `r=0,r=-1,q=1,s=0,s=1` | `p` on `s=1` |
| `p,y_-1` | `r=0,r=1,q=0,s=0,s=1` | `p'` on `s=0` |
| `p,y_1` | `r=0,r=1,q=0,q=1,s=1` | `p'` on `q=1` |

In each row every nonparallel intersection of two listed lines is one of the
four lozenge Attacker stones.  Hence an empty Defender cell lies on at most
one central line.  In the no-corner branch the two raw cells damage at most
two pencils.  In a one-corner branch the corner damages the one displayed
pencil and the other cell damages at most one more.  At least three pencils
are intact.

The four Attacker stones contain no collinear triple, so every alive count is
at most two and current `tau=0` exactly.  Any next Defender pair touches at
most two of the five central lines and leaves one of the three intact pencils
untouched.  Its standard two flank triggers give the exact demand-two
residual family

`{-3,-2}, {-2,3}, {3,4}`.

Thus every candidate handoff has `TEMPO>=2`; the pure-count-two bound L10.4
gives `TEMPO<=2`.  This proves all assertions in (103).  ∎

### 71.4 The unique double-corner obstruction

**Lemma L14.10 (exact sealed branch) [Q2, PROVEN at one-level scope].**  The
only raw Defender occupancy not covered by L14.9 is

`{p,p'}={(0,1),(1,-1)}`.                                  (104)

Its two orders give the exact R5 sealed handoff with

`n_1=6,n_2=5,tau=0,TEMPO=2`.                              (105)

Every legal Attacker response from (105) returns a nonterminal Defender epoch
with `M<=2` by R5.1.

*Proof.*  If at most one corner is occupied, the case rule in L14.9 has two
candidate wing cells and at most one other Defender cell, so one wing remains
available.  Thus only (104) prevents all five displayed one-turn lozenge
templates.  This is also complete geometrically: any unit lozenge containing
the adjacent edge `cd` contains a triangle on that edge, and its third vertex
must be one of the two common neighbors `p,p'`.  Occupying both therefore
excludes every local one-turn lozenge completion, not merely the displayed
case rule.

The inventory and exact `TEMPO` are L11.1 and R5.3.  For the all-response
bound, the no-external-support premise of R5.1 causes no proof gap here, even
though remote Defender support can enlarge the legal response set.
Every root Defender cell is outside the chosen `B_j`; Defender stones create
no alive label and can only delete labels.  A response cell outside `B_j`
lies in no six-window meeting a sealed Attacker cell, by the definition of
`B_j`, and is therefore virgin relative to the sealed family.  If both
response cells are local, L11.2--L11.4 classify them; if one is local and one
is remote, only the local cell can promote a sealed label; if both are remote,
they create at most count-two labels.  These are exactly the two-Q, one-Q,
and virgin/split branches of the R5.1 proof.  Its servicing construction thus
still gives `M<=2` in the full root occupancy.  ∎

For an exact current-demand check inside this obstruction, let a sealed
response use two Q-row parameters `x<y`.  It has `tau=0` when the four
coordinates `x,y,0,1` have span above five; otherwise it has `tau=2` exactly
for

`{x,y} in {{-2,-1},{-1,2},{2,3}}`,                         (106)

and `tau=1` in every other case.  In the three cases (106), the maximal
possible graded-Q start ranges are

| response parameters | all possible graded-Q starts | service occupancy |
|---|---|---|
| `{-2,-1}` | `-6,-5,-4,-3,-2,-1,0` | `{-3,2}` |
| `{-1,2}` | `-5,-4,-3,-2,-1,0,1` | `{-2,3}` |
| `{2,3}` | `-4,-3,-2,-1,0,1,2` | `{-1,4}` |

The actual alive Q set can be a subset of the displayed maximal range because
a remote root Defender cell may already delete a newly born extreme.  Every
displayed service nevertheless meets the whole maximal range, while every
non-Q label has count at most one.  It therefore deletes the exact actual
graded family and gives exact `M=0`.  If a sealed
response is not two Q-row cells, it creates no current imminent and has exact
`tau=0`.  These statements enumerate current `tau`; no unproved exact value
of `M` is assigned to the remaining sealed responses.

*Proof of Theorem R8.3.*  L14.7 absorbs every initial strategy branch.
L14.8 classifies the exact raw action values, L14.9 handles every non-seal
occupancy, and L14.10 handles the two remaining orders.  This is a complete
one-level tree.  It does not iterate either successor family, so it does not
prove arrival at a local stop state and does not close Q2.  ∎

The exact remaining Q2 deviation classes are therefore:

1. renewal or forcing after the double-corner sealed handoff and its banked
   safe first cycle;
2. arbitrary next-Defender-action trees at generalized lozenge plateaux; and
3. after a minimizing raw reply, continuation beyond R8.1's exact one-ply
   repair of `P_0^pl`.

## 72. The finite renewal certificate

Define `C_cap` to be the following class of Attacker handoffs:

1. `I=empty`;
2. every graded label not in a named derivative has count two;
3. there are at most two exact nested derivatives, each with a count-three
   residual contained in its adjacent count-two residual as in (90) or the
   stock analogue after L14.6; and
4. no transverse pre-count-two label meets a derivative tail, while no
   pre-count-two connector contains cells from two different tails.

**Theorem R8.4 (one-cycle capped landing certificate) [Q3-repair, PROVEN at
the exact P0/P1 domains].**  From the handoff after (85) at either
`P_0^pl` or `P_1^pl`, every legal Attacker response returns an epoch with a
legal servicing pair whose actual handoff belongs to `C_cap`.  Every member
of `C_cap` reached by this construction has `TEMPO<=2`.

*Proof.*  The complete same-axis rank cover in R8.1/R8.2, including two
separated one-trigger effects with no common old window, deletes all high
labels and lands in the zero-derivative subcase.  The sole hard
stock-singleton family is covered the same way.  In the distinct-axis branch,
L14.3 and the P1 extension leave at most two high axes; L14.2 and L14.6 delete
easy axes and produce exactly the named nested derivatives on hard axes.
L14.4 and the extended P1 tail audit establish clause 4.  Their complete
future-pair split proves `TEMPO<=2`.  The chosen pair services every current
imminent in the same-axis branch, while current service is vacuous in the
exact-`tau=0` branch.  ∎

R8.4 is the promised honest renewal step: after one completed cap/response/
service cycle, the proof restores an explicit unripe handoff class rather
than only quoting a scalar maximum.  It does **not** prove that every response
from an arbitrary `C_cap` member can be serviced back into `C_cap`; the
isolation ranges and stock incidence were proved only for the two exact input
epochs.  Consequently `GAP-TEMPO-REPAIR`, `GAP-REPLACEMENT-INVARIANT`, and
the amortized-credit route remain **OPEN**.

## 73. Authoritative round-8 status ledger

| Claim / named gap | Quantifier tag | Status | Exact basis / remaining scope |
|---|---|---|---|
| GAP-RAW | Q2 counterroute / Q3 target | **OPEN** | Neither full root forcing nor a perpetual Defender policy is proved |
| R8.1 first-plateau decision | Q3 | **PROVEN: REPAIRABLE-AT-P0** | Dual near-trigger cap (85); exhaustive response-axis census and tail isolation |
| `R_1(P_0^pl,a^dagger)=2` | Q3 diagnostic | **PROVEN EXACT** | All-response upper (92), exact witness (93)--(94) |
| `B_1(P_0^pl)=2` | Q3 diagnostic | **PROVEN EXACT** | Every action leaves two shield pencils for the risk floor; (85) attains two |
| Diamond shield alone is a one-ply local stop | Q3 negative alternative | **REFUTED** | R8.1 exhibits a safe action |
| R8.2 one-stock test | Q3 | **PROVEN at exact `P_1^pl`** | Exact census `(33,16)`, stock stabilizer, extended bridge/tail audit |
| `R_1(P_1^pl,a^dagger)=B_1(P_1^pl)=2` | Q3 diagnostic | **PROVEN EXACT** | Same exact witness and shield risk floor |
| Dual near-trigger cap at `P_2^pl,P_3^pl,P_4^pl` | Q3 | **OPEN** | No `V^-`, `W`, or `V^+` incidence audit is claimed |
| Dual near-trigger cap at `P_5^pl=P_stock` | Q1/Q3 | **UNSAFE, `R_1>=3`** | By binding R7.2 (§67): every legal Defender action at final `P_stock` has a response with `M>=3`; the cap action is one such `a`. The earliest loss index in `{2,3,4,5}` remains unknown |
| R8.3 complete earliest Q2 branch | Q2 | **PROVEN at one-level scope** | Enlarged untouched launch plus exhaustive raw action dichotomy |
| Arbitrary initial root reply | Q2 | **ABSORBED / PROVEN** | One of three disjoint enlarged launch footprints is untouched |
| Raw action quotient | Q2 | **PROVEN EXACT** | Survivor-block table; exact `tau=0,M=0`; twenty ordered minimizers |
| Raw minimizers reach exact local `P_0^pl` | Q2 | **PROVEN** | All seven Q labels deleted; transverse diamond census untouched |
| Non-seal raw deviations | Q2 | **PROVEN one-level reroute** | Generalized five-pencil lozenge with exact `tau=0,M=2` |
| Double-corner raw branch | Q2 | **OPEN after banked first cycle** | Exact sealed profile; every immediate response has `M<=2` by R5.1 |
| Full strategy-independent root forcing | Q2 | **OPEN** | Neither successor family is iterated to a proven stop state |
| R8.4 capped landing certificate | Q3-repair | **PROVEN at exact P0/P1 domains** | Every first response has a service landing in `C_cap` with `TEMPO<=2` |
| General renewal of `C_cap` | Q3-repair | **OPEN** | No arbitrary-member closure theorem |
| R7.2 final `P_stock` stop theorem | Q1/Q3 inherited | **PROVEN, unchanged** | Universal verdict remains attributed to R7.2; exact census `(127,55)` |
| General count-three initialization | Q3-initialization | **OPEN** | L13.6 still covers only residual transversal at most two |
| Other shared/nonshared fanouts and cross-hull closure | Q3 | **OPEN** | The cap audit covers only the exact diamond/first-stock geometries |
| Nested derivative beyond the capped landing audit | Q3 | **OPEN** | No perpetual iteration of (90) or (98) |
| Alternative forced-service transverse-seal entrance | Q2/Q3 boundary | **OPEN** | The seal occurs only when Defender selects the double-corner deviation; no forced entrance or renewal theorem follows |
| `GAP-REPLACEMENT-INVARIANT` / amortized account | Q3 | **OPEN** | R8.4 is finite local regeneration, not global closure or a credit rule |
| Minimum separation for R5.2 | ancillary | **OPEN** | Radius 21 remains envelope-sharp only |
| New machine verification | all | **none** | Hand proofs only; no prohibited run or generated enumeration |

No inherited round-2 through round-7 `PROVEN` or `VERIFIED` theorem is
downgraded.  R8.1 refutes only the previously open negative alternative at
the first plateau.  It does not change R7.2's universal final-state theorem.

## 74. Hostile-review attack surface

1. **Candidate legality.**  The two cells in (85) are Defender stones placed
   into different alive count-two pencils; they are not Attacker responses.
   The second remains radius-eight legal after the first.
2. **Exact capped census.**  Each cap kills four vertical count-two labels and
   five bridge-row count-one labels, not an entire diagonal pencil.  The P0
   totals are exactly `(8,12)`.
3. **Current versus future demand.**  When both response cells enter one old
   count-two label, current `tau` is one or two and the rank cover must service
   it.  Two cells on the same central axis may instead be too far apart to
   share such a label; then `tau=0`, but one stabilizer need not meet both high
   subfamilies.  L14.2.1 deliberately spends the full two-cell rank cover on
   that separated same-axis class.
4. **High-axis exhaustion.**  A third high axis requires an old count-one
   bridge containing both triggers.  The complete P0 bridge check reduces to
   the two round-7 fan pairs, each disabled by one cap cell.
5. **Stabilization is not deletion.**  Adjacent trigger parameters `-1` and
   `2` leave the exact nested tails (90).  The proof carries those
   count-three labels forward rather than silently declaring a pure tier.
6. **New count-two bridge leak.**  It is not enough that high-axis residuals
   are disjoint.  L14.4 checks that no lower count-two label uses a tail
   trigger, including a label newly born from the response.
7. **Two-tail connectors.**  Same-side parallel tails can be collinear on a
   fixed `q` or `r`; the connector table proves those lines have old Attacker
   count zero.  Cross-side tails are not collinear.
8. **Fillers and order.**  Effective service/stabilizer cells are played
   first.  A filler is appended only afterward and can only delete labels.
   Every effective cell was already within five of unchanged Attacker stock.
9. **Exact risk witness.**  The pair (93) creates two four-window
   consecutive-triple families and no imminent.  Concentrated and split
   Defender replies give the complete lower-bound census; the displayed
   stabilizers attain two.
10. **Risk minimum.**  The equality `B_1=2` also needs a lower bound for every
    action.  Two of four shield pencils are untouched after any pair, and one
    exterior trigger in each creates the two-family lower-bound witness.
11. **P1 row blockers.**  `D@(2,0)` kills stock start `2` and the left
    extreme.  Exactly starts `3,...,6` survive at count two; omitting this
    blocker would give the wrong `(33,16)` census and stabilizer table.
12. **Stock singleton pencils.**  A consecutive triple on `q=6,q=7,s=6`, or
    `s=7` may need two cleanup cells.  The coordinate audit proves that this
    hard family cannot coexist with another activated central high axis.
13. **Stock tail isolation.**  Tail levels `9,10,11` are distinct from all
    old levels `0,1,6,7`.  This is the load-bearing exclusion of a future
    count-two bridge at `P_1^pl`.
14. **Initial Q2 branch.**  The old launch union is enlarged to `B_j`; without
    this enlargement an initial Defender cell could pre-occupy an adaptive
    lozenge wing without touching the original adjacent-pair windows.
15. **Lozenge intersections.**  A legal empty cell touches at most one of the
    five central lines because every nonparallel intersection is one of the
    four Attacker vertices.  The occupied corner itself accounts for one
    damaged pencil in the one-corner branches.
16. **Double-corner scope.**  The seal is the unique obstruction to one-turn
    lozenge assembly, not a proved perpetual Defender escape.  R5.1 banks only
    its first all-response `M<=2` cycle.
17. **Local versus root forcing.**  Generalized lozenges have exact `M=2`,
    not a proved stop property.  R8.3 is a complete one-level tree and no
    more.
18. **Renewal boundary.**  `C_cap` is reached from two exact inputs.  No claim
    quantifies over every abstract member or every later stock prefix.
19. **Quantifier discipline.**  R8.1/R8.2 are Q3 local-action theorems;
    R8.3 is Q2 one-level forcing; none is promoted to GAP-RAW.
20. **Evidence status.**  Every table, count, residual, and hitting number is
    a hand proof labeled `PROVEN`, never machine `VERIFIED`.

## 75. Provenance and no-run record

**Input commit:** `a8a0b92d641b690b63d43f049d2b4c2fa0d4e9c1` on branch
`hunt/gap-raw`.  This authoring pass creates no commit.  **Reviewed/output
artifact:** `GAP_RAW_PROOF_ROUND8.md` is the working-tree deliverable; no
landed output commit exists during this no-commit pass, so no commit identity
is fabricated.  Once the artifact is landed and reviewed, its landed identity
must be added in the round-8 review/folded errata, following Sections 47, 58,
and 67. **Landed/reviewed artifact (added post-review per Finding 11):**
`c57da44286f75feb236e6da6c55cdd53e5ec2e68` (blob
`6c460713444ff94758b5416debdfa5b27fa878ef`), byte-identical to the reviewed
working-tree copy.

The required corpus was read first, in this order and in full:

1. `GAP_RAW_PROOF_ROUND5.md`, including binding Section 47, then
   `GAP_RAW_REVIEW_ROUND5.md`;
2. `GAP_RAW_PROOF_ROUND6.md`, including binding Section 58, then
   `GAP_RAW_REVIEW_ROUND6.md`;
3. `GAP_RAW_PROOF_ROUND7.md`, including binding Section 67, then
   `GAP_RAW_REVIEW_ROUND7.md`.

Rounds 2--4 were consulted only for the inherited blanket rules, legality and
service lemmas, equation (20), pure-count-two theorem, and raw-launch
geometry.  No `STRATEGY_STEALING_*` file was opened as evidence.

**File authored:** `GAP_RAW_PROOF_ROUND8.md`.

This pass did not modify the test-gated harness, production rules, strict
verifier, Lean sources, or any unrelated working-tree file.  No Cargo command,
Lean build, harness, game/search executable, generated enumeration, or git
commit was run.

## 76. Errata and strengthenings folded from the round-8 hostile review

`GAP_RAW_REVIEW_ROUND8.md` (ultra, reviewed artifact `c57da442`) returned
**SOUND-WITH-MINOR-ERRATA**: no REFUTED or MAJOR finding. R8.1 CONFIRMED
with the crux held — the review independently verified the response
enumeration is EXHAUSTIVE (remote, split, bridge, and separated same-axis
pairs all covered; no missed response with `M>=3`; the cap's exact
`(n_1,n_2)=(8,12)` inventory recomputed). §70 CONFIRMED (exact `P_1^pl`
risk two, stock-assisted classes included). §71 CONFIRMED with one wording
repair. R8.4 CONFIRMED with one bookkeeping repair. Folds:

1. **(Finding 3, MINOR)** L14.9's legality sentence split by template:
   the no-corner pair `(p,p')` is at distance two with each cell
   independently adjacent to unchanged old stock; the one-corner
   templates keep the adjacent-wing phrasing. No value changes.
2. **(Finding 5, MINOR)** R8.4 gains the safe-filler clause: fillers are
   chosen outside surviving derivative supports (max-`q` standard, or
   away from the low-only outer cell for easy second-axis deletions),
   else the killed derivative leaves the named list. Repairs exact
   `C_cap` membership; R8.1/R8.2/`TEMPO<=2` unchanged.
3. **(Finding 10, MINOR)** The later-plateau ledger row split:
   `P_2..P_4^pl` OPEN; `P_5^pl=P_stock` UNSAFE with `R_1>=3` by binding
   R7.2 — the earliest loss index in `{2,3,4,5}` is the open datum.
4. **(Finding 11, MINOR)** §75 records the landed/reviewed artifact
   identity `c57da442` (blob `6c460713`).

**Review confirmations of record.** The cap is legal, servicing, and
non-hub (Finding 1); §71 absorbs the root reply over the complete raw
action quotient (Finding 2); every non-seal lozenge endpoint has exact
`tau=0,M=2` and the double-corner seal is honestly unresolved (Finding 4);
tail isolation, the exact witness, and the universal risk floor recompute
(Finding 8); the `U^-` stock census preserves exact risk two (Finding 9);
Q1/Q2/Q3 boundaries and the meaning of REPAIRABLE-AT-P0 are honest
(Finding 12). The review's nine-item unresolved-obstacle list is the
authoritative open state for round 9.
