# GAP-RAW Proof Round 10 — exact `B_1(P_3^pl)`, seal successors, and proactive renewal

**Worktree:** `hunt/gap-raw` at input HEAD
`f5349d3eb985cdb9ee719ec75272f3a73772604d`  
**Date:** 2026-07-18  
**Method:** hand proofs and read-only corpus inspection only. No Cargo command,
Lean build, harness, game/search program, solver, generated enumeration, or
git commit was run.

This document continues the definitions, theorem numbering, and equation
numbering of `GAP_RAW_PROOF_ROUND9.md`, with binding Section 86. In
particular, the corrected transition is `k*=3`; the cap is exact-risk two at
`P_2^pl`; every action is unsafe at `P_3^pl`; the cap is exact-risk three at
`P_4^pl`; the seal quotient is `435/870` with `41/45` root-robust row
successors; and the six full-delete actions at `P_x` all have risk at least
three. The Q1/Q2/Q3 contract, Section 76 safe-filler clause, and Section 58
flat-shield contract remain binding.

## 87. Executive disposition and quantifier contract

### 87.1 Round-10 results

1. **Exact first unsafe plateau value [Q1/Q3, PROVEN].** The cap
   `a^dagger=((0,-1),(1,1))` has exact one-successor risk three at
   `P_3^pl`. Since binding (R9-REV-4) gives the same lower bound for every
   action, the last unknown plateau value closes as

   `B_1(P_3^pl)=3`.                                      (137)

2. **Seal successor classes [Q2, PARTIAL].** Each of the four canonical
   surviving-exterior singleton landings has an all-next-response returned
   bound `M<=2`. Every response consisting of two genuinely virgin returns
   has exact returned value zero or one under a complete born-window
   criterion, including a born carrier crossing the old Q row at an empty
   cell. One virgin plus one local return is exact when no response-pair
   bridge survives. Noncanonical exterior minimizers, the exact returned
   strata after the canonical exterior landing, every Q/arm value-one
   successor, and one-virgin born-bridge values remain open.

3. **Proactive renewal minimizers [Q3-repair, PARTIAL with a complete
   action characterization].** A legal action at `P_x` is an immediate
   minimizer exactly when it meets one explicit eight-cell set `C`. This
   characterizes every proactive flank truncation, not only the six
   full-delete orders. Every such minimizer except `48` unordered (`96`
   ordered) occupancies is proved to have risk at least three. The remaining
   finite boundary is explicit; no action in it is proved safe and no
   universal `B_1(P_x)>=3` conclusion is claimed.

### 87.2 Scope

The exact value (137) is a reached-state Q1/Q3 theorem. It does not prove that
every strict-root Defender strategy reaches `P_3^pl`. The seal theorems are
Q2 subtree classifications only at the stated returned classes. The `P_x`
result is a Q3-repair action and risk classification at one exact successor;
it is neither arbitrary-member `C_cap` closure nor a perpetual policy.

## 88. Exact `B_1(P_3^pl)`

Write `s=q+r` and retain

`a^dagger=((0,-1),(1,1))`.                              (138)

After (138), binding L15.2 gives the exact profile

`(n_1,n_2)=(69,34), n_j=0 for j>=3`.                    (139)

Thus every legal Attacker response is nonterminal.

### 88.1 Enumeration architecture

**Lemma L16.1 (complete post-cap response index) [Q3, PROVEN].** Every
returned high label after a response `b=(x,y)` has exactly one of two sources:

1. an old count-two label hit by `x` or `y`; or
2. an old count-one label containing both `x,y`, necessarily on their unique
   common lattice axis.

The first source is indexed by the following ten finite count-two carriers;
the multiplicities sum to the `34` in (139).

| carrier | `q=0` | `q=1` | `s=0` | `s=1` | `r=0` | `q=10` | `s=6` | `s=7` | `s=10` | `r=-3` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count-two labels | 1 | 1 | 5 | 5 | 4 | 5 | 2 | 3 | 5 | 3 |

The only empty cells incident with more than one of these finite carriers are

`h=(10,0) in r=0,q=10,s=10`,
`g=(9,-3) in s=6,r=-3`.                                 (140)

Count-one extremes on a carrier in the table fold into its same-axis class.
The remaining singleton carriers and all of their empty incidences with the
ten-carrier family are exactly:

| singleton carrier | empty finite-carrier incidences |
|---|---|
| capped `r=1` | `s=0` at `(-1,1)` |
| capped `r=-1` | `s=1` at `(2,-1)` |
| `q=6` | `s=7` at `(6,1)` |
| `q=7` | `s=6` at `(7,-1)` |
| `r=-4` | `s=1` at `(5,-4)`; `s=7` at `(11,-4)`; `s=10` at `(14,-4)` |
| `q=12` | `s=7` at `(12,-5)`; `r=-3` at `(12,-3)` |
| `q=13` | none |
| `r=-2` | `s=6` at `(8,-2)`; `s=7` at `(9,-2)`; `q=10` at `(10,-2)` |

Consequently the complete stock-assisted class in which two distinct
finite-carrier incidences lie in one alive singleton window consists of
exactly five unordered occupancies:

```text
{(11,-4),(14,-4)},
{(12,-5),(12,-3)},
{(8,-2),(9,-2)},
{(8,-2),(10,-2)},
{(9,-2),(10,-2)}.                                      (141)
```

*Proof.* Equation (139) excludes every other source of a high label: a
pre-count-zero label reaches at most count two, and a pre-count-one label
reaches count three only if it receives both new stones. Distinct stones
share at most one lattice axis. The ten-carrier list is (115)--(116).
Intersecting their finite six-window unions gives only (140); intersections
at old Attacker stones are not legal response incidences. For the second
source, take the three carriers through each old Attacker stone and delete
the carriers already in the ten-carrier list. Direct substitution gives the
singleton table. On `r=-4`, the new binding boundary incidence `(5,-4)` is
included. Its distances from the other two incidences are six and nine, so
it forms no alive common singleton window with either. The `q=12` row gives
one pair and the three `r=-2` incidences give three pairs. Together with the
one `r=-4` pair these are exactly (141). Remote/remote and local/remote
responses are, respectively, the zero- and one-old-effect rows of the same
index. This exhausts effects, not sampled board coordinates. ∎

### 88.2 Complete returned-value upper quotient

Let `K_3` be the union of every alive positive-count window after the cap at
`P_3^pl`. It is finite. From the exact P3 Attacker coordinates,

`K_3 subseteq {-5<=q<=18, -9<=r<=6}`.

The future `V^+` stones `(10,4),(10,5)` lie outside `K_3`: on `q=10` the
`V^-` touched-window union ends at `r=2`, while rows `4,5` and levels `14,15`
contain no P3 Attacker stone. The intervening P4 ray cells
`(-72,8),(-80,8)` lie outside every window meeting `K_3`.

**Lemma L16.1.1 (P4-to-P3 graded transfer) [Q3, PROVEN].** If both response
cells lie in `K_3`, then

`M(P_3^pl+a^dagger+b)<=3`.

*Proof.* The same ordered `b` is empty and legal after the cap at `P_4^pl`.
Relative to every P3 returned label, that P4 state has the two additional
`V^+` Attacker stones and may have extra graded labels; its two additional
ray Defender cells are remote from every window meeting the P3 support.
Binding L15.1.1, confirmed in
Section 86.2, supplies a legal P4 service `d^+` whose handoff has
`TEMPO<=3`.

Construct a P3 service `d^-` by retaining, in order, every cell of `d^+`
which lies in an alive P3 label of count at least two in the returned state
before either service is played. Every retained cell is
legal from unchanged P3 Attacker stock. Every P3 current imminent is also a
P4 current imminent, so the retained cells still service the complete P3
current family. If fewer than two cells remain, append a Section 76 safe
filler outside every surviving graded support.

Every graded P3 label surviving `d^-` now corresponds to a surviving P4
label with at least the same Attacker count. Indeed, if a discarded `d^+`
cell had killed such a P3 label, that cell would have lain in a P3
count-at-least-two label and would have been retained.

Take an arbitrary legal future P3 pair. A cell not occupied in the P4
handoff is copied unchanged. If a future cell is `(10,4)` or `(10,5)`, its
effect is already present in P4; moreover, a copied mate licensed by that
cell in P3 is independently licensed in P4 by the already-present `V^+`
stone. If a future cell is an extra or discarded P4 Defender cell, it lies in
no P3 graded label and cannot contribute to an imminent after one pair. In
that second conflict case, any contributing mate triggers a surviving
count-three label and is independently legal from its unchanged Attacker
stock, so play it first even if it was second in the original order. Joint
maturation of a pre-count-two label cannot use a Defender-conflict cell,
because that cell would lie in the graded label and would have been retained.

Omit every conflict cell and complete the P4 pair with standard empty
max-coordinate legal fillers supported by a copied first cell or by unchanged
P4 stock. If no copied cell contributes, the relevant P3 demand is either
already represented by `V^+` in P4 or is zero, and any legal P4 filler pair
suffices. Extra Attacker effects from fillers only add P4 constraints and
therefore do not weaken the comparison.

The resulting legal P4 pair has a returned imminent family at least as hard
as the P3 family: corresponding residuals are subsets because of the extra
Attacker stock, and extra labels can only add constraints. Thus its hitting
number is at least the P3 hitting number. The confirmed P4 handoff bounds
every such pair by three, so the P3 handoff does as well. This proves the
graded-transfer claim.
∎

**Lemma L16.2 (all responses to the cap return `M<=3`) [Q3, PROVEN].** For
every legal ordered response `b` after (138),

`M(P_3^pl+a^dagger+b)<=3`.                              (142)

*Proof.* Use the exhaustive quotient `j=|{x,y} intersect K_3|`.

1. **`j=0`.** No old alive label receives a response stone. Every old label
   stays at count at most two and every newly born label has count at most
   two. The low-only theorem L12.6 gives returned `M<=2`.
2. **`j=1`.** The exterior cell lies in no alive old label, so it creates no
   old high effect and no old count-one bridge with the local cell. If the
   local cell meets no count-two carrier, the state is again low-only. On one
   ordinary count-two carrier, the at-most-two-cell axial cover deletes the
   complete one-trigger high family. At `g`, one common residual cell deletes
   each of its two easy families. At `h`, use `(11,-1)` to delete the
   `s=10` family and one common residual cell to delete either the `r=0` or
   `q=10` family. The remaining incident family is the atomic `U^-` or
   `V^-` two-start block and has one-trigger demand at most one. If a future
   pair is concentrated on that carrier, its complete mixed axial family
   costs at most two. If it is split, at most one cell triggers the atomic
   block; any additional imminent must come from a pre-count-two label
   containing both future cells, hence from their unique common axis and
   costs at most two. Thus this last row has handoff `TEMPO<=3`, and every
   other one-local row has at most two.
3. **`j=2`.** Lemma L16.1.1 applies uniformly and gives returned `M<=3`.
   L16.1's ten central carriers, `h,g`, singleton table, and exactly five
   occupancies (141) are the complete finite effect refinement of this row.
   In particular, same-axis, distinct-axis, capped bridge, ordinary hard
   bridge, all five stock-assisted double incidences, and the repaired
   `(5,-4)` boundary are included; none is imported from the refuted P3
   value-two synthesis.

The three quotient rows are disjoint and exhaustive, and every ordered pair
has been bounded. This proves (142). ∎

### 88.3 The attaining response

Put

`b_*=((9,-2),(10,-2))`,
`d_0={(11,-2),(8,-1)}`.                                 (143)

Binding Section 86.1 proves that every Defender action after `b_*` hands
over `TEMPO>=3`. The only occupancy which deletes both common-blocker
families is `d_0`; every other action leaves either two untouched demand-two
families or one untouched demand-two family plus a surviving demand-one
family.

**Lemma L16.3 (exact returned value of `b_*`) [Q3, PROVEN].** The response
in (143) has

`M(P_3^pl+a^dagger+b_*)=3`.                             (144)

*Proof.* After `d_0`, the only count-at-least-three labels are the four
`q=10` count-three windows with starts `-7,-6,-5,-4`; every other alive label
has count at most two. Index an arbitrary future Attacker pair by its number
of `q=10` cells.

- With two `q=10` cells, the one-axis five-rank interval cover has hitting
  number at most two.
- With no `q=10` cell, the old high family is not promoted. Every imminent
  comes from pre-count-two labels containing both future stones, hence lies
  on their unique common axis and has demand at most two.
- With exactly one cell `(10,t)`, the vertical family has demand two only at
  `t=-5` or `t=-1`; it has demand at most one for
  `t=-7,-6,0,1`, and zero outside the residual union. At `t=-5`, the
  transverse carriers `r=-5,s=5` have no pre-future-pair Attacker support.
  At `t=-1`, `r=-1` has only `(1,-1)`, at axial distance nine, and `s=9`
  has none. Thus the two demand-two depths cannot also mature a transverse
  pre-count-two label. At every other depth the vertical contribution is at
  most one and the unique lower-tier axis contributes at most two.

This is the complete future-pair quotient, so `d_0` hands over
`TEMPO<=3`. The pair

`e_*=((10,0),(11,-1))`                                  (145)

is empty after `d_0` and independently legal from the unchanged `q=10` and
`s=10` stock, in either order. It attains three. On `q=10` its residuals are
`{-5,-1},{-1,1}`, of hitting number one. On the intact `s=10` pencil its
residuals are `{8,9},{9,14},{14,15}`, of hitting number two. The axes meet
only at the newly occupied hub, so their residual grounds are disjoint.
Together with the binding lower bound for every service, this proves (144).
∎

**Theorem R10.1 (exact first-loss plateau value) [Q1/Q3, PROVEN].** At the
exact `P_3^pl`,

`R_1(P_3^pl,a^dagger)=B_1(P_3^pl)=3`.                   (146)

*Proof.* Lemma L16.2 gives the cap upper and L16.3 gives an attaining
response, so its risk is exactly three. Binding (R9-REV-4) gives risk at
least three for every legal action. By the Section 58 shield contract every
legal action is an immediate-value minimizer of exact value two, so the
minimum defining `B_1` ranges over that same complete action set. The cap
attains its universal floor. ∎

**Other `P_3^pl` action risks [Q3, OPEN individually].** The theorem needs no
upper bound for an `H_3`-touching action: the cap already attains the universal
lower floor. Such actions retain their binding risk-at-least-three status;
their individual exact risks are not assigned here.

## 89. The seal's open successor classes

Retain the normalized sealed handoff

```text
A@(0,0), A@(1,0);  D@(0,1), D@(1,-1),
W_s={(q,0):s<=q<=s+5}.
```

The five robust count-two row labels are `W_-4,...,W_0`. A *virgin return*
below means a response cell lying in no alive label of this sealed family.
Remote root support may make such a cell legal and may prune a newly born
window; it does not add an Attacker stone or an alive old label.

### 89.1 Canonical exterior singleton landings

The four surviving-exterior cases of L15.8 have these canonical landings.

| first response | canonical service | sole count-two row label | possible outer count-one row label |
|---|---|---|---|
| `{-5,-4}` | `{-3,2}` | `W_-9` | `W_-10` |
| `{-4,-3}` | `{-2,2}` | `W_-8` | `W_-9` |
| `{4,5}` | `{-1,3}` | `W_4` | `W_5` |
| `{5,6}` | `{-1,4}` | `W_5` | `W_6` |

The last two rows are the `rho` images of the first two. The count-one row
label may already be root-deleted; the displayed count-two label is assumed
alive, which is exactly the value-one side of the root-dependent dichotomy.

**Lemma L16.4 (canonical exterior successor safety) [Q2, PROVEN at the four
displayed landings].** From each displayed canonical value-one handoff,
every legal next Attacker pair returns an epoch with `M<=2`. Its exact current
demand is

```text
tau=1  iff both next returns lie in the four-cell residual of the sole
            count-two row label;
tau=0  otherwise.                                      (147)
```

*Enumeration architecture and proof.* Index the complete legal response set
by the number `j in {0,1,2}` of returns on the old Q row.

1. If `j=2`, every high label is either the sole count-two row label or its
   sole possible outer count-one neighbor. There are at most two such labels,
   on one axis. One empty residual representative from each deletes the
   complete high family and services the count-four label when it exists. If
   the representatives coincide, play the common effective cell once and
   append a Section 76 safe filler.
2. If `j=1`, the only possible high label is the sole count-two row label. A
   nonrow carrier through both response stones meets Q at the newly occupied
   row-return cell; that cell was empty before the response, so the carrier
   contains no old Attacker stone and cannot promote a pre-count-one bridge
   to count three.
3. If `j=0`, the count-two row label is unchanged. If a high label exists,
   both returns lie in old count-one windows on their unique common non-Q
   axis. That axis meets the old Q row once and therefore contains at most one
   old Attacker stone. The complete high family is a deletion-subfamily of
   the intervals through one fixed rank triple. The consecutive/nonconsecutive
   rank-triple cover of L11.4 deletes it with at most two cells.

These rows include local/local, local/remote, remote/remote, aligned,
nonaligned, and root-pruned responses. Effective cells lie in alive high
labels; play them first and use a Section 76 safe filler when needed. The
resulting graded tier is pure count two, so L10.4 gives `TEMPO<=2` and hence
the claimed returned bound. Only the sole pre-count-two label can reach count
four, which proves (147). ∎

**Exterior-successor disposition [Q2, PARTIAL-WITH-BOUNDARY].** Lemma L16.4
closes the all-response safety bound for each canonical singleton landing.
It does not enumerate every noncanonical immediate minimizer of the four
first-response epochs, and it does not divide the returned responses into
exact `M=0,1,2` strata. If root contact already deleted the exterior window,
the inherited value-zero theorem R9.5 applies instead.

### 89.2 Every response with two virgin returns

Let `v,w` be a legal pair of genuinely virgin returns. Define `B(v,w)` to be
the actual alive family of newly born length-six windows containing both.
Let

```text
C_Q={
 {-4,2},
 {-3,2},{-3,3},
 {-2,2},{-2,3},{-2,4},
 {-1,2},{-1,3},{-1,4},{-1,5}
}                                                        (148)
```

be the ten unordered two-cell covers of the five robust Q labels.

**Lemma L16.5 (complete two-virgin effect index) [Q2, PROVEN].** The returned
graded tier consists exactly of `W_-4,...,W_0` together with `B(v,w)`, all at
count two, and current `tau=0`. Either `B` is empty, or `v,w` have one unique
common carrier `L`, separation `d in {1,...,5}`, and—after normalizing their
carrier ranks to `0,d`—an actual born-start subset

`S subseteq {d-5,...,0}`.                               (149)

This index is exhaustive, with all exterior root pruning represented by
`S`.

*Proof.* Virginity means neither return raises an alive old label. A new
label containing one return has count one; a new label containing both has
count two. Distinct cells share at most one lattice axis, and a length-six
window contains both exactly when their axial separation is at most five.
Defender pruning can only delete members of the maximal start interval. ∎

**Theorem R10.2 (exact two-virgin seal quotient) [Q2, PROVEN EXACT].** For
every legal two-virgin response,

```text
M=0  iff some C in C_Q meets every member of B(v,w);
M=1  otherwise.                                         (150)
```

In particular, no two-virgin response has returned value two.

*Proof.* If `B` is empty, any member of (148) deletes the complete graded
tier and gives value zero. Suppose `B` is nonempty. One Q cap leaves at most
one robust Q label. On `L`, with response ranks `0<d`, carrier rank `-1`
deletes every maximal born start except `0`. It is distinct from both response
stones. If it is empty, it lies next to unchanged Attacker stock and is legal.
If it is an old Defender cell, every born start through it is already dead.
If it is an old Attacker cell, an alive window through it would contradict
the virginity of the two returns; such starts are again absent. Thus, whenever
the cell cannot be played, the actual born family is already a singleton. One
effective cell per family leaves at most one count-two label on each carrier.
If the Q cap and carrier cap coincide, play that common effective cell once
and append a safe filler; otherwise each effective cell is independently
radius-supported by the unchanged Attacker stock on its carrier.

If `L` differs from Q, parallel carriers cannot be matured together and
nonparallel carriers meet in only one cell; two distinct future stones
therefore cannot mature both singleton labels. If `L=Q`, both virgin returns
are on one exterior side. On the left, write `v<w<=-6`; the service cells
`w+1` and `2` leave only born start `w-5` and robust `W_-4`, which are
physically disjoint. If `w+1` is already Defender-occupied, every other born
start is already dead and only the same leftmost start can survive. If it is
Attacker-occupied, any alive born window through it would contradict
virginity, so the same conclusion holds. In either case omit that effective
cell and use a safe filler after the Q cap. The reflected pair
handles the right side. Hence an actual service always hands over
`TEMPO<=1`.

A handoff here has `TEMPO=0` exactly when every count-two label is deleted:
any survivor has two legal residual triggers which return demand at least
one.
Deleting all five robust Q labels already uses both placements and forces an
occupancy in (148). Such an action deletes `B` exactly when it meets every
member. This proves both directions of (150). ∎

**Corollary R10.2.1 (empty Q-row crossing) [Q2, PROVEN EXACT].** Suppose the
born carrier is nonparallel to Q and crosses it at the empty row cell `c`.
Then

```text
M=0 iff c lies in every alive born window and
           q(c) in {-4,-3,-2,-1,2,3,4,5};
M=1 otherwise.                                          (151)
```

*Proof.* A Q-row cover can meet a born window on the non-Q carrier only at
`c`. The displayed eight ranks are exactly the row coordinates occurring in
at least one cover in (148). Apply (150). A parallel non-Q carrier with
nonempty `B` therefore always has exact value one. ∎

### 89.3 One virgin return

Use the exact local return sets `X,N` from (136). A *surviving born bridge*
means precisely that the actual newly born family containing both the local
and virgin return is nonempty after inherited Defender pruning. First
distinguish whether such a bridge exists.

**Lemma L16.6 (one-virgin nonbridge quotient) [Q2, PROVEN EXACT].** If no
born bridge survives, the returned state has `tau=0` and

```text
local return in X: M=0;
local return in N: M=1.                                 (152)
```

*Proof.* For a positive Q return `1+k`, the two row cells `{-1,3}` at depth
one and `{-1,2}` at depths two through five delete the complete graded row
support; reflect on the negative side. With no born bridge the virgin cell
adds no graded label, giving exact value zero.

For an arm return, the graded tier is the five robust Q labels plus its one
promoted arm label, all at count two. One Q cap leaves at most one Q label,
while one arm residual cell deletes the arm label completely. The arm is
therefore dead, and the handoff has at most the sole Q carrier, so
`TEMPO<=1`. Value zero is
impossible: deleting all five Q labels consumes two Q-row placements, while
the arm label meets Q only at that occupied endpoint. ∎

**One-virgin bridge boundary [Q2, PARTIAL/OPEN].** If an alive bridge is
born, current `tau` remains zero. An arm-plus-virgin response is low-only and
has `M<=2`; for Q plus virgin, the full row cleanup leaves only pure
count-two born stock and again gives `M<=2`. The exact `0/1/2` quotient is
open. Its finite index is the local return type and depth, the actual born
start subset, and its crossings with Q or the promoted arm.

### 89.4 Off-row value-one successors

**Q/arm successor continuation [Q2, OPEN].** The `200+190` first-response
values in L15.9 remain exact, but their minimizing landing sets and complete
next-response quotients are not classified here. After an off-row landing, a
trigger in a retained component can lie on another old-A carrier, so the
one-row-intersection reduction used in L16.4 does not apply. A complete
tail/connector audit is still required.

## 90. Proactive renewal minimizers at `P_x`

Retain

`b_x=((-2,2),(0,2))`, `P_x=Q_1^cap+b_x`.                (153)

Round 9 proved `tau(P_x)=0,M(P_x)=2` and enumerated the six ordered actions
which delete every high label. This section enumerates the whole immediate
minimizer set before studying risk.

### 90.1 Exact graded census and the minimizer index

**Lemma L16.7 (complete graded census at `P_x`) [Q3-repair, PROVEN].** The
complete count-two/count-three family is:

| axis and parameter | count-three starts | count-two starts |
|---|---|---|
| `s=0`, parameter `q` | `-4,-3,-2` | `-5,-1,0` |
| `q=0`, parameter `r` | `0` | `1` |
| `q=1`, parameter `r` | none | `-5` |
| `r=0`, parameter `q` | none | `3,4,5,6` |
| `r=2`, parameter `q` | none | `-5,-4,-3,-2` |
| `s=1`, parameter `q` | none | `-4,-3,-2,-1,0` |

No other axis has count at least two.

*Proof.* The eight local Attacker stones are

```text
(0,0),(0,1),(1,0),(1,-1),(6,0),(7,0),(-2,2),(0,2).
```

Enumerate the fixed-`q`, fixed-`r`, and fixed-`s` lines containing at least
two of them and delete windows hit by
`(-4,0),(2,0),(0,-1),(1,1)`. Direct interval containment gives the table.
Every omitted line has at most one Attacker stone. ∎

Define the empty high-union cells

```text
C_S={(-4,4),(-3,3),(-1,1),(2,-2),(3,-3)},
C_Q={(0,3),(0,4),(0,5)},
C=C_S union C_Q.                                        (154)
```

The first set is the union of the empty cells in the three `s=0`
count-three residuals;
the second is the residual of the single `q=0` count-three label.

**Theorem R10.3 (complete immediate-minimizer characterization) [Q3-repair,
PROVEN EXACT].** A legal ordered action `d` at `P_x` is an
immediate-`TEMPO` minimizer if and only if its occupancy meets `C`.

*Proof, necessity.* If `d` misses `C`, the legal response

`{(-1,1),(0,3)}`

creates on `s=0` the residual path

`{-4,-3}, {-3,2}, {2,3}`,                               (155)

of hitting number two, and on `q=0` the residual `{4,5}`, of hitting number
one. The carriers meet at occupied `(0,0)`, so the grounds are disjoint and
the returned demand is three. Such an action has handoff `TEMPO>=3` and
cannot attain `M(P_x)=2`.

*Proof, sufficiency.* First record the complete concentrated-axis bounds.
On `s=0`, starts `-5,...,0` have counts `2,3,3,3,2,2`. If a future pair
does not use parameter `-1`, the empty cell `-1` hits every activated
count-three label and every matured low start except possibly start `0`,
which needs at most one further hit. If the pair is `{-1,t}`, the following
two-cell covers exhaust the nontrivial parameter effects:

| `t` | cover |
|---|---|
| `-5,-4` | `{-3,2}` |
| `-3` | `{-4,2}` |
| `2` | `{-3,3}` |
| `3,4,5`, or outside the graded support | `{-3,2}` |

Thus the full mixed `s=0` family has two-response demand at most two. The
`q=0` count-three start `0` is nested inside count-two start `1`: whenever
both mature, the former residual is contained in the latter, so their total
demand is at most one.

A Defender contact at the five cells of `C_S`, in displayed order, leaves
the `s=0` high-start sets

`{-3,-2}, {-2}, empty, {-4}, {-4,-3}`.                  (156)

Either adjacent two-start family retains a common empty hit after any one
trigger, so every family in (156) has one-trigger demand at most one. A
contact in `C_Q` deletes the complete nested `q=0` pair.

It remains to exclude a stock-assisted extra demand. The finite graded-axis
intersections are exhaustive:

```text
s=0 with q=0,q=1,r=0,r=2 meets at
    (0,0),(1,-1),(0,0),(-2,2), and is parallel to s=1;
q=0 with s=0,s=1,r=0,r=2 meets at
    (0,0),(0,1),(0,0),(0,2), and is parallel to q=1.
```

Every listed intersection is Attacker-occupied. Hence a split high trigger
cannot simultaneously mature an off-axis pre-count-two label. A
pre-count-one response bridge reaches only count three and a virgin label
only count two. If `d` meets `C_Q`, only the full `s=0` bound remains. If it
meets `C_S`, a split response costs at most one on each high axis, while a
concentrated response costs at most two. If a future pair avoids every
surviving count-three residual, any imminent can only arise from a
pre-count-two label containing both future cells; those labels lie on the
pair's unique common axis and have demand at most two. Therefore every
action meeting `C` hands over
`TEMPO<=2`; the inherited universal floor two makes it an exact minimizer.
∎

**Corollary R10.3.1 (all proactive actions characterized) [Q3-repair,
PROVEN EXACT].** The full-delete minimizers are exactly

`{(-1,1),(0,j)}, j in {3,4,5}`,                          (157)

with both orders. Every other legal action meeting `C` is a non-full-delete
immediate minimizer. Thus proactive minimizers are no longer an unindexed
completeness caveat.

### 90.2 Risk-three subfamilies

The common R9.4 response

`e_x={(-1,2),(8,0)}`                                    (158)

creates consecutive-triple families on `r=2,s=1,r=0`. Their empty finite
supports at the time an action at `P_x` is chosen are

```text
F_2={(q,2): q=-5,-4,-3,-1,1,2,3},
F_1={(q,1-q): q=-4,-3,-2,-1,2,3,4},
F_U={(q,0): q=3,4,5,8,9,10,11}.
```

Put `F=F_2 union F_1 union F_U`. The first two sets meet only at
`p=(-1,2)`, so `|F|=20`, and direct substitution gives `C intersect F=empty`.

**Lemma L16.8 (cofinite proactive risk obstruction) [Q3-repair, PROVEN].**
Every immediate minimizer outside the following `48` unordered occupancies
has `R_1>=3`:

1. the three occupancies `{(-1,2),(0,j)}`, `j=3,4,5`; and
2. a cell of `C_S` together with one of the nine cells in

```text
r=2: q in {-4,-3,-1,1,2};
s=1: q in {-3,-2,-1,2,3},                               (159)
```

where the shared cell `p=(-1,2)` is counted once.

*Enumeration architecture and proof.* By R10.3, every minimizer meets `C`.
If both its cells lie in `C`, or its unique non-`C` cell misses `F`, (158)
leaves all three focal families intact and the binding R9.4 `2+1` argument
applies. The same argument survives a contact at any of the six outer
endpoints

```text
r=2: q=-5,3;  s=1: q=-4,4;  r=0: q=3,11.
```

Such a contact leaves three consecutive starts and hence a demand-two flank.

Now suppose the mandatory minimizer cell lies in `C_Q`. The old `s=0`
family remains intact. For every proactive cell in `F` except `p` and
`(8,0)`, response (158) is legal and leaves two of its three focal families
untouched. Together with old `s=0`, these are three action-disjoint hard
axes. If the proactive cell is `(8,0)`, use
`{(-1,2),(9,0)}` instead; `D@(8,0)` killed the `U^-` block, so `(9,0)` adds
no high family, while `p` creates the intact `r=2,s=1` families. The old
`s=0` family is the third hard axis. Thus only the three `C_Q` actions paired
with `p` remain.

Finally suppose the mandatory cell lies in `C_S` and the proactive cell is
one of

`(4,0),(5,0),(8,0),(9,0),(10,0)`.                       (160)

Use response `{(0,3),(-1,2)}`. The `q=0` high label becomes the sole current
imminent, forcing one service cell on its residual. The other service cell
can touch at most one of the intact `r=2,s=1` consecutive-triple families,
so the returned handoff has demand at least `2+1`. Removing the five cells
in (160) leaves exactly the nine-cell union (159). Counts are therefore
`3+5*9=48` unordered occupancies. Every case excluded from this list has the
stated lower bound. ∎

**Renewal-risk disposition [Q3-repair, PARTIAL-WITH-EXACT-BOUNDARY].** The
remaining boundary has `48` unordered and `96` ordered actions; both orders
are independently legal. Indeed, every `C` cell lies in an alive high
residual, every displayed partner lies in an alive `r=2` or `s=1` count-two
residual, and `C intersect F=empty`, so the two distinct cells are supported
by unchanged Attacker stock independently of order. No member is proved to
have risk at most two, and no member is proved unsafe here. In particular, neither
`B_1(P_x)<=2` nor `B_1(P_x)>=3` follows.

**Three-action micro-quotient [Q3-repair, OPEN].** The sharpest unresolved
subfamily is

`{(-1,2),(0,j)}, j=3,4,5`.                               (161)

These actions delete every `q=0` and `r=2` graded window and truncate `s=1`
to its sole count-two start `0`. Their unclassified responses can still form
a hard born count-one bridge together with central promotions; that complete
tail/connector sweep is the next local obligation.

## 91. Authoritative round-10 status ledger

| Claim / obstacle | Quantifier tag | Status | Exact basis / remaining scope |
|---|---|---|---|
| `GAP-RAW` | Q2 counterroute / Q3 target | **OPEN** | No strict-root all-strategy force and no initialized all-history Defender policy |
| Corrected plateau transition | Q1/Q3 | **PROVEN: `k*=3`**, inherited | Safe cap through `P_2^pl`; every action unsafe from `P_3^pl` onward |
| Exact cap risk at `P_3^pl` | Q3 | **PROVEN EXACT: 3** | L16.1--L16.3; all response effects indexed, `b_*` attains three |
| Exact `B_1(P_3^pl)` | Q1/Q3 | **PROVEN EXACT: 3** | R10.1 combines cap attainment with binding universal lower bound |
| P3 stock-assisted sweep | Q3 | **PROVEN COMPLETE** | Ten finite central carriers, two empty multi-carrier cells, eight singleton-carrier types, exactly five double incidences |
| Four canonical exterior row successors | Q2 | **PROVEN all-response `M<=2`; exact strata PARTIAL** | Complete `j=0,1,2` Q-row-response-count quotient; noncanonical first minimizers and exact returned values unclassified |
| Two genuinely virgin first returns | Q2 | **PROVEN EXACT** | R10.2: exact `M=0/1` criterion through actual born-window family |
| Virgin bridge crossing empty Q cell | Q2 | **PROVEN EXACT for two-virgin class** | Corollary R10.2.1; common-crossing and eight-cover-rank criterion |
| One virgin, no born bridge | Q2 | **PROVEN EXACT** | Q return gives `M=0`; arm return gives `M=1` |
| One virgin with born bridge | Q2 | **PARTIAL / OPEN exact value** | `tau=0,M<=2`; exact `0/1/2` incidence quotient remains |
| Q/arm value-one successor continuation | Q2 | **OPEN** | First landing values remain exact; next tail/connector quotient unproved |
| Complete immediate minimizers at `P_x` | Q3-repair | **PROVEN EXACT** | R10.3: exactly the legal actions meeting eight-cell set `C` |
| Full-delete minimizers at `P_x` | Q3-repair | **PROVEN EXACT**, inherited and recovered | Exactly the three occupancies (157), both orders |
| Non-full-delete proactive minimizers | Q3-repair | **PROVEN COMPLETE CHARACTERIZATION** | Every minimizing action meeting `C` but not in (157) |
| Proactive risk-three classification | Q3-repair | **PROVEN outside 48 occupancies** | L16.8; full legal action set reduced to `48/96` exact boundary |
| `B_1(P_x)` | Q3-repair | **OPEN** | No safe action exhibited; the `48/96` boundary prevents a universal lower bound |
| Arbitrary-member `C_cap` renewal | Q3-repair | **OPEN** | Exact action classification at one successor is not class closure |
| Q2 strategy-independent forcing | Q2 | **OPEN** | Reached-state P3 loss is not a forced strict-root route |
| General count-three initialization | Q3 | **OPEN beyond L13.6** | Residual hitting number at least three remains outside the theorem |
| R5.2 separation sharpness | ancillary | **OPEN** | Radius 21 remains envelope-sharp only |
| New machine verification | methodological | **NONE** | Every result is a hand proof |

## 92. Updated seven-item caveat ledger

This is the required carry-forward of Section 86.7, in the same order.

1. **Q2 root forcing [Q2, OPEN].** The exact local transition `k*=3` and
   exact value `B_1(P_3^pl)=3` do not force arrival at that plateau from every
   strict-root strategy.
2. **Exact `B_1(P_3^pl)` magnitude [Q1/Q3, CLOSED / PROVEN].** Round 10
   closes the former lower-bound-only item with exact value three.
3. **Seal exceptional/value-one/virgin successors [Q2, PARTIAL].** The four
   canonical surviving-exterior landings now have all-response `M<=2`; all
   two-virgin responses are exact `M=0/1`, including empty-Q crossings; and
   one-virgin nonbridges are exact. Noncanonical exterior minimizers, exact
   exterior returned strata, Q/arm value-one successors, and one-virgin
   born-bridge values remain open.
4. **Proactive renewal and arbitrary `C_cap` renewal [Q3-repair, PARTIAL].**
   Immediate minimizers at `P_x` are completely characterized. All but
   `48/96` have risk at least three. That exact boundary and arbitrary-member
   closure remain open, so no universal negative or positive renewal theorem
   follows.
5. **Alternative repair geometries and forced-seal entrances [Q2/Q3,
   OPEN].** No result here classifies generalized-lozenge continuation,
   other shared/nonshared fanouts, cross-hull closure, later derivatives, an
   alternative forced entrance to the transverse seal, or any amortized
   credit / `GAP-REPLACEMENT-INVARIANT` argument.
6. **General count-three initialization [Q3, OPEN].** L13.6 still covers
   only residual hitting number at most two. Nothing in the exact `P_x`
   census generalizes that initialization theorem.
7. **R5.2 separation sharpness [ancillary, OPEN].** Radius 21 remains
   envelope-sharp only; no smaller exact value separation is proved.

## 93. Proof-audit checklist

1. **P3 enumeration precedes values.** L16.1 indexes all high-label sources
   before assigning a service. The response space is factored by finite
   carrier effects, not sampled coordinates.
2. **Stock-assisted completeness.** The `(5,-4)` boundary is present. The
   five and only five singleton double incidences are displayed in (141),
   and `b_*` is not folded into an easy bridge row.
3. **`b_*` upper as well as lower.** Binding Section 86.1 supplied only the
   lower bound. L16.3 separately enumerates every future pair after `d_0`
   and proves the missing upper three.
4. **Exact `B_1`, not a causal synthesis.** Equation (146) follows from one
   attained action and the binding universal lower bound. The carrier map is
   an exhaustive proof index; it is not presented as an explanation of why
   the stock phase must occur on arbitrary histories.
5. **Exterior scope.** L16.4 concerns the displayed canonical singleton
   landing. It does not quantify over unenumerated noncanonical minimizers or
   claim exact returned `M` values.
6. **Virgin definition.** A virgin cell lies in no *alive* old sealed label.
   The actual born family `B(v,w)` retains root pruning as the start subset
   `S`; it is not silently replaced by a clean maximal carrier.
7. **Empty Q-row crossing.** The born carrier can cross Q at an empty cell.
   Corollary R10.2.1 keeps that cell and tests both common-window membership
   and occurrence in a complete Q cover.
8. **One-virgin bridge boundary.** The nonbridge exact values are not
   extended to a surviving born bridge. Only the proved upper two is carried.
9. **Complete `P_x` action universe.** R10.3 starts from the full graded
   axis census. Pre-count-one bridges cannot become imminent in the one
   future pair used by `TEMPO`; every pre-count-two crossing with the two
   high axes is explicitly occupied.
10. **Proactive risk boundary.** L16.8 classifies an infinite action family
    down to exactly `48` unordered occupancies. Those cases remain genuinely
    open; no `B_1(P_x)>=3` synthesis is made.
11. **Ordered legality and fillers.** Effective cells are played before any
    filler. Both orders are claimed only when each cell is independently
    supported by unchanged Attacker stock. Every spare placement obeys the
    Section 76 safe-filler clause.
12. **Quantifiers.** P3 is Q1/Q3 reached-state, the seal is a Q2 subtree,
    and `P_x` is Q3-repair. None is promoted to Q2 root forcing, perpetual
    Q3 renewal, or GAP-RAW.
13. **Evidence.** No executable, solver, generated table, or formal build is
    evidence for any claim.

## 94. Provenance, artifact identity, and no-run record

- **Input branch:** `hunt/gap-raw`.
- **Input HEAD:** `f5349d3eb985cdb9ee719ec75272f3a73772604d`
  (`f5349d3e`), confirmed by a read-only identity check before authoring.
- **Shared-branch drift:** a final read-only check observed HEAD
  `88bca52d2a52dbcda5da60db81f00f69ad6cfcd7`. A name-only comparison from
  the required input showed changes solely in excluded strategy-stealing
  review/output records; all six required GAP-RAW proof/review files were
  byte-identical across the two commits. No reset or checkout was performed,
  and no drifted file was read as evidence.
- **Required corpus, read first in this exact order and in full:**
  1. `GAP_RAW_PROOF_ROUND7.md`, including binding Section 67, then
     `GAP_RAW_REVIEW_ROUND7.md`;
  2. `GAP_RAW_PROOF_ROUND8.md`, including binding Section 76, then
     `GAP_RAW_REVIEW_ROUND8.md`;
  3. `GAP_RAW_PROOF_ROUND9.md`, including binding Section 86, then
     `GAP_RAW_REVIEW_ROUND9.md`.
- **Supplementary inherited corpus consulted afterward:** Round-5 Sections
  38--39 for the normalized seal and rank covers; Round-6 Sections 49--51 for
  exact plateau coordinates and stock cadence; and placement-legality
  passages in Sections 52--53.
- **Excluded corpus:** no `STRATEGY_STEALING_*` file was opened or used as
  evidence.
- **Artifact authored:** `GAP_RAW_PROOF_ROUND10.md`.
- **Landed artifact hash:** `LANDED_ARTIFACT_HASH: <TO-BE-FOLDED-POST-REVIEW>`.
- **Commit status:** no commit was created; committing was prohibited.
- **Execution record:** no Cargo command, Lean command, harness,
  game/search executable, solver, generated enumeration, or test was run.
  Read-only file and Git inspection were used only for corpus and provenance;
  all mathematical case classifications above are hand proofs.

## Review erratum (R-G8-REV, folded)

**Reviewed artifact:** this document, landed unmodified at `8424e9a4`
(supplies the §94 `LANDED_ARTIFACT_HASH` placeholder). **Review:**
`GAP_RAW_REVIEW_ROUND10.md` (committed `5787961a`). **Verdict:**
SOUND-WITH-MINOR-ERRATA.

- Target 1 — `B_1(P_3^pl)=3` **CONFIRMED EXACT** under hostile recomputation:
  no legal nonterminal response to the cap `a†` reaches `M>=4`.
- Target 2 — **CONFIRMED at stated partial scope**; the named seal classes
  (Q/arm successors, born-bridge refinements) remain honestly OPEN.
- Target 3 — **CONFIRMED at stated partial scope**; immediate `P_x`
  minimizers and the exact `48/96` renewal residual verified.
- Only finding = the MINOR provenance omission above, now recorded.
