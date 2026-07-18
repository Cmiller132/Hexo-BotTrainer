# GAP-RAW Proof Round 9 — the earliest loss index and the double-corner seal

**Worktree:** `hunt/gap-raw` at input HEAD
`c019400ad14e06fa6f600c5462113a74795e3270`  
**Date:** 2026-07-18  
**Method:** hand proofs only. No Cargo command, Lean build, harness, game/search
program, machine enumeration, or git commit was run.

This document continues the definitions, theorem numbering, and equation
numbering of `GAP_RAW_PROOF_ROUND8.md`, including binding Section 76. In
particular, the safe-filler clause, the split `P_2^pl`–`P_4^pl` versus
`P_5^pl=P_stock` ledger, and the no-corner legality wording are binding. The
meanings of `I`, `tau`, `TEMPO`, `M`, `Serv`, `R_1`, `B_1`, `C_cap`, and the
Q1/Q2/Q3 tags are unchanged.

## 77. Executive verdict and quantifier contract

### 77.1 The three fronts remain separate

1. **Stock-phase repairability (Q3).** The exact capped plateau epochs are
   statewise action questions. A safe cap at one index is not the replay of
   already occupied cells on one history.
2. **Double-corner continuation (Q2).** The sealed handoff is reached only on
   the corresponding arbitrary raw-reply branch. A local continuation from
   that handoff does not by itself close strategy-independent root forcing.
3. **Capped renewal (Q3-repair).** A landing action with low scalar value is
   not automatically a second-cycle action. The same actual action must be
   checked against every next Attacker response.

Every result below uses ordered `FirstStone`/`SecondStone` placements and the
inclusive radius-eight legality rule. Remote, split, bridge, same-axis, and
stock-assisted response effects are included whenever an all-response claim
is made.

### 77.2 Round-9 disposition

1. **Earliest loss:** `k*=4` is **PROVEN**. The cap has exact risk two at
   `P_2^pl,P_3^pl`; every action is unsafe at `P_4^pl`, where the cap has
   exact risk three.
2. **Double-corner seal:** **PARTIAL**. The complete finite residual-return
   first-response quotient is exact, and forty-one of forty-five row returns
   are root-robust for another all-response `M<=2` safety bound. Four exterior
   singleton successors, Q/arm value-one successors, and virgin returns remain
   open.
3. **Renewal:** **OPEN with a PROVEN obstruction**. At one exact capped
   successor, all six full-delete immediate minimizers have risk at least
   three; unclassified proactive minimizers prevent a universal negative
   conclusion.

Q2 root forcing, a perpetual Q3 policy, general count-three initialization,
and GAP-RAW are not inferred from any local theorem in this round.

## 78. The post-`V^+` three-demand obstruction

Retain the exact cap

`a^dagger=((0,-1),(1,1))`.                                (107)

At `P_4^pl`, let

```text
U = [6,11] on r=0,          with A at q=6,7,
V^- = [-4,1] on q=10,       with A at r=-4,-3,
V^+ = [0,5] on q=10,        with A at r=4,5,
W_0 = [8,13] on q+r=10,     with A at q=12,13,
W_sat = [8,15] on q+r=10.
```

(108)

Call `H_4=U ∪ V^- ∪ V^+ ∪ W_sat`. The last segment is the full
physical union of the three prospective `s=10` count-four windows, including
their outer residual ranks `14,15`. This enlarged finite union is disjoint
from both triangle-fan unions `G_-`,`G_+` of L13.2: its additional two cells
meet the fan carrier lines only outside their finite supports. As before,
`G_- intersect G_+` contains only occupied diamond cells. Hence one empty
Defender cell affects labels in at most one of `H_4,G_-,G_+`.

**Lemma L15.1 (exact focal value-three witness) [Q1/Q3, PROVEN].** Suppose a
legal Defender action at `P_4^pl` misses `H_4`. Then the legal Attacker pair

`b_4=((10,0),(11,-1))`                                    (109)

returns a Defender epoch whose complete imminent family is the three
`q+r=10` windows with `q`-starts `8,9,10`. Their residuals are

`{8,9}`, `{9,14}`, `{14,15}`,                             (110)

in line parameters, so `tau=2`. The unordered servicing occupancies are
exactly

`{8,14}`, `{9,14}`, `{9,15}`                              (111)

on `q+r=10`, with both orders in every case. After every one of these six
ordered services, the handoff has exact `TEMPO=3`.

*Proof.* Before (109), every alive count is at most two. The two new cells are
independently legal: `(10,0)` is within range of the old `q=10` stock and
`(11,-1)` is adjacent to the old `W_0` pair; missing `H_4` makes both empty.
The two new cells are
consecutive on `q+r=10`; together with the old consecutive pair at parameters
`12,13` they make four consecutive Attacker stones. Exactly the starts
`8,9,10` reach count four. A label on another axis receives at most one new
cell, so no other imminent exists. This proves the complete current family,
(110), and the three two-cell covers (111). Every cover is sequentially legal
in the still-alive current windows.

Every cell in (111) has physical coordinate `(q,10-q)` with
`q in {8,9,14,15}`. It lies on none of `U,V^-,V^+`, so the following pair is
empty and legal after every service:

`e_4=((10,1),(8,0))`.                                    (112)

Its first cell is independently within range of `V^+`, and its second is
adjacent to `U`; neither relies on the other for legality. It creates the
following focal imminent residuals:

```text
V^-: {(10,-2),(10,-1)},
V^+: {(10, 2),(10, 3)},
U, start 6 (mandatory): {(9,0),(11,0)},
U, start 5 (when alive): {(5,0),(9,0)}.
```

(113)

The two vertical grounds are disjoint. The mandatory row ground is nonempty;
when the optional row ground survives, the two share `(9,0)`. Row and
vertical grounds are mutually disjoint, so the complete current family has
hitting number exactly three in either case; hence (112) has exact `tau=3`.

Before (112), the complete count-three family is a deletion-subfamily of the
five maximal labels `U` starts `5,6`, `V^-` starts `-5,-4`, and `V^+` start
`0`. The protected segments force `U` start `6`, `V^-` start `-4`, and
`V^+` start `0` to survive every action missing `H_4`; cells outside `H_4`
may delete `U` start `5` or `V^-` start `-5`. The latter does not mature under
(112), and no label outside this five-label family becomes imminent.

For the upper bound, first note that the maximal pretrigger residual triples
on `q=10` are

`{-5,-2,-1}`, `{-2,-1,1}`, `{1,2,3}`.

A vertical trigger has one of the four nonzero membership types
`{-5}`, `{-2,-1}`, `{1}`, or `{2,3}` in this three-triple chain. The direct
two-trigger audit is:

| first / second type | `{-5}` | `{-2,-1}` | `{1}` | `{2,3}` |
|---|---:|---:|---:|---:|
| `{-5}` | unavailable | `1` | `2` | `2` |
| `{-2,-1}` |  | `2` | `2` | `2` |
| `{1}` |  |  | unavailable | `2` |
| `{2,3}` |  |  |  | `1` |

The entries are maximum resulting hitting numbers over the coordinates in
the two types; a repeated singleton type is unavailable because a cell cannot
be replayed. A zero-membership trigger adds no high effect. Each nontrivial
entry leaves either one outer component, the two separated outer components,
or a three-residual path whose middle residual shares a hit with one outer
residual. Deletion of an optional triple is monotone. Thus the two
separated vertical blocks together have demand at most two.

Now split an arbitrary future pair by high-family effect. If
both cells use `r=0`, the complete row family has demand at most two. If both
use `q=10`, the two separated vertical blocks together have demand at most
two by the preceding audit. If one cell uses each axis, the row
contributes at most one and the two vertical blocks together at most two.
Their only physical row/vertical intersection is the already occupied stone
`h`; the other finite connector levels `5,8,9,11` contain no transverse
pre-count-two label. If exactly one cell uses a high axis and the other uses
neither, the finite-axis check shows that the high residual cell lies in no
transverse pre-count-two label; the off-axis cell alone raises lower stock by
only one, so only the at-most-two high-axis demand remains. A pair with no
high-family trigger is pure count two and has demand at most two. These cases
give `TEMPO<=3`, while (112) gives the
reverse bound. The value is exact. ∎

**Theorem R9.1 (universal local stop at `P_4^pl`) [Q1/Q3, PROVEN].** Every
legal Defender pair `a` at `P_4^pl` has a legal nonterminal response returning
`M>=3`.

*Proof.* If `a` misses `H_4`, use L15.1; all six services have exact value
three, so the returned epoch has exact `M=3`. If `a` touches `H_4`, that cell
touches neither triangle fan and the other cell touches at most one fan. One
complete `G_sigma` is untouched, and L13.3 supplies a legal response returning
`M>=3`. These cases exhaust all legal actions, including pairs outside every
region, pairs split across the fans, and pairs with one or two `H_4` cells. ∎

**Lemma L15.1.1 (exact cap risk at `P_4^pl`) [Q3, PROVEN].** The cap has

`R_1(P_4^pl,a^dagger)=3`, and `B_1(P_4^pl)=3`.             (114)

*Proof.* L15.1 supplies a response of exact returned value three. For the
reverse bound, repeat the `P_3` effect quotient of Section 79 with one change:
`V^+` adds a second six-window-separated adjacent-pair block on the already
present physical axis `q=10`.

First consider two responses on `q=10`. Within either old adjacent block,
apply the exact L14.2/L11.4 interval cover. If both responses make a hard
family in one block, the other block is unaffected or has only an easy short
effect. If one hard trigger lies in each separated block, use one outward
stabilizer per block. The `V^-` tails are the negative tails of Section 79.4;
for `V^+`, trigger `r=3` is stabilized at `r=2` to tail `6,7,8`, and trigger
`r=6` is stabilized at `r=3` to tail `7,8,9`. The two retained vertical tails
are atomic and separated by more than one six-window, so their aggregate
future weight is at most two. The overlap triggers `r=0,1` have only easy
effects on both blocks; when both are played, the exact current grounds are
the two disjoint pairs `{-2,-1}` and `{2,3}`. Thus every all-vertical response
has an actual current service leaving vertical weight at most two.

For completeness, if one response is `h=(10,0)`, write the other as
`(10,t)`. At `t=1`, the exact current family is the two vertical starts `-4`
and `0`, with grounds `{-2,-1}` and `{2,3}` and exact `tau=2`. Service at
`(10,-1),(10,2)`; this deletes every vertical count-at-least-three label.
The `h`-born `U` block has future weight one and the `W` block weight two;
their intersection `h` is occupied. The legal future pair
`{(8,0),(11,-1)}` attains the disjoint sum `1+2`, while concentration in
either block and every other split pair give at most three. This service has
exact handoff `TEMPO=3`. If `t` is not `1`, the current vertical demand is at
most one. Play `D@(11,-1)`,
the common empty gap of the three
`h`-born `W` starts, and use the second cell on `q=10` to service and reduce
the complete vertical family to total atomic weight at most two. The `U`
block remains with weight one and intersects the vertical axis only at
occupied `h`, giving handoff bound at most three.

On every other same-axis response use the Section-79 axial or connector
service. For a split response, stabilize each nonvertical hard one-trigger
family to future weight one. The unique vertical effect class has weight at
most two by the preceding paragraph, so total future weight is at most three.
The only cross-axis finite connector remains `h`: if empty, `D@h` truncates
the retained high-low incidence; if response-occupied, its incident effects
are short and easy. A `V^+` hard tail is oriented away from `h` as above.

The new `r=4,r=5,s=14,s=15` count-one carriers meet no other live count-two
union in an empty cell. Existing bridge responses use the oriented one-cell
stabilization and the hard `r=-2` exception of Section 79.4; a new isolated
carrier is deleted outright. Remote/remote, local/remote, two-local split,
same-axis, bridge, and stock-assisted responses are therefore exhaustive by
effect. In each case the assigned service hands over `TEMPO<=3`. Effective
cells are played first and any unused placement is a Section-76 safe filler.

Hence every response to the cap returns `M<=3`, while L15.1 attains three.
R9.1 gives the universal lower bound three for every initial action, so the
minimum `B_1` is also exactly three. ∎

The theorem is a reached-state result. It does not assert that an arbitrary
strict-root Defender strategy must reach `P_4^pl`.

## 79. The cap through `P_2^pl` and `P_3^pl`

### 79.1 Exact stock-prefix inventories

For two Attacker stones at distance `d<=5` on one unblocked axis, their common
count-two windows number `6-d` and their exclusive count-one windows number
`2d`. An adjacent pair therefore contributes five and two, respectively.

**Lemma L15.2 (complete capped prefix census) [Q3, PROVEN].** After (107),
the exact alive inventories at the six named plateau epochs are

| index | last installed stock | `n_1` | `n_2` | `n_j`, `j>=3` |
|---:|---|---:|---:|---:|
| `0` | diamond | `8` | `12` | `0` |
| `1` | `U^-` | `33` | `16` | `0` |
| `2` | `V^-` | `49` | `26` | `0` |
| `3` | `W` | `69` | `34` | `0` |
| `4` | `V^+` | `95` | `39` | `0` |
| `5` | `U^+` | `115` | `47` | `0` |

At `P_2^pl` the complete count-two central-axis census is

```text
q=0:1, q=1:1, s=0:5, s=1:5,
r=0:4, q=10:5, s=6:2, s=7:3.             (115)
```

At `P_3^pl` it additionally has

`s=10:5, r=-3:3`.                                      (116)

*Proof.* The first two rows are L14.1 and L14.5. Adding `V^-` gives an
adjacent pair on `q=10` (`+2,+5`), two singleton rows (`+12,0`), a
distance-four pair on `s=6` in place of one singleton (`+2,+2`), and a
distance-three pair on `s=7` in place of one singleton (`+0,+3`). Thus the
increment is `(16,10)`.

Adding `W` gives an adjacent pair on `s=10` (`+2,+5`), singleton columns
`q=12,13` (`+12,0`), one new singleton row `r=-2` (`+6,0`), and converts the
old `r=-3` singleton into a distance-three pair (`+0,+3`). Its increment is
`(20,8)`. Adding `V^+` supplies a second, six-window-separated adjacent pair
on `q=10` (`+2,+5`) and four singleton row/level pencils (`+24,0`), for
`(26,5)`. Finally `U^+` gives a separated adjacent row pair (`+2,+5`), two
singleton columns (`+12,0`), and converts the `s=14,15` singletons into
distance-four and distance-five pairs (`+2,+2` and `+4,+1`), for `(20,8)`.
No listed addition makes a collinear triple in a six-window; Defender
augmentation only deletes labels. This proves the table and (115)--(116). ∎

### 79.2 The finite intersection quotient

Among the finite residual unions in (115), the only empty intersection of
two distinct count-two central axes is

`h=(10,0) in (r=0) intersect (q=10)`.                    (117)

At `P_3^pl`, `h` also lies in the `s=10` pencil, and there is one further
empty intersection

`g=(9,-3) in (s=6) intersect (r=-3)`.                    (118)

Every other nonparallel intersection is an old Attacker stone or lies outside
at least one finite live residual union. Parallel pairs have no intersection.
For reference, the potentially deceptive infinite-line intersections with
the capped shield, the `s=6,7` bridges, and the `r=-3` pair fail by these
finite ranges:

| axes compared | possible old-stock connector | disposition |
|---|---|---|
| `r=0` with `q=0,1,s=0,1,s=6,s=7` | levels `0,1,6,7` | the intersection is an old Attacker stone |
| `q=10` with `s=6,s=7,r=-3` | rows `-4,-3` | the intersection is an old Attacker stone |
| `q=0,1` with `s=6,s=7` | rows near `5,6,7` or `-3,-4` | outside the capped vertical interval, or the stock endpoint is occupied |
| `s=0,1` with `s=6,7,10` | parallel | no intersection |
| `r=-3` with shield lines | columns `0,1` or levels `0,1` | outside the three live row intervals |
| `r=-3` with `s=7,s=10` | `(10,-3),(13,-3)` | occupied stock endpoints |

This table is about physical finite window unions, not merely central lines.
It is the stock-prefix replacement for the occupied-intersection premise of
L14.3.

### 79.3 Exact current-demand quotient

Parameterize a response pair on each central axis by that axis's displayed
integer coordinate. If both response cells share a live count-two axis, all
current imminents lie on that unique physical axis. Exact `tau=2` occurs only
for the following unordered response pairs:

| axis | old ranks | exact `tau=2` response pairs |
|---|---|---|
| `s=0,s=1` | `0,1` | `{-2,-1}`, `{-1,2}`, `{2,3}` |
| `r=0` | `6,7` | `{5,8}`, `{8,9}` |
| `q=10` at `P_2/P_3` | `-4,-3` | `{-6,-5}`, `{-5,-2}`, `{-2,-1}` |
| `s=7` | `7,10` | `{8,9}` |
| `s=10` at `P_3` | `12,13` | `{10,11}`, `{11,14}`, `{14,15}` |
| `r=-3` at `P_3` | `10,13` | `{11,12}` |

These are exactly the cases in which four consecutive axial ranks occur and
all three corresponding six-windows survive. A co-contained pair in every
other row of the effect table has exact `tau=1`; a pair contained in no old
count-two label has exact `tau=0`. The capped `q=0,q=1` singletons and the
distance-four `s=6` family never contribute more than one current demand.

For a same-axis response avoiding `h,g`, the axial rank-triple cover of L11.4
services and deletes the complete main axis, including separated one-trigger
effects. A secondary high family can then only be a one-trigger family on an
incident axis or the unique old count-one carrier containing both response
cells. The intersection quotient (117)--(118) makes each such secondary
family atomic: one residual cell deletes it, or its exact future demand is
one.

The connector-containing same-axis cases require a different service. At
`h`, first distinguish the main axis. On `q=10` or `r=0`, play
`D@(11,-1)` to delete the `h`-born `s=10` block at `P_3` (omit that cell's
effect at `P_2`), and use the second cell on the main axis: it deletes the
short `h`-side starts and leaves at most the far one-trigger atomic block.
The untouched incident `U^-` or `V^-` block is itself atomic, so the two
retained components have total future demand at most two. On `s=10`, the
exact axial cover deletes the main family; its sole exact-`tau=2`
`h`-containing pair is `{10,11}`, and only the atomic `U^-` and `V^-` blocks
remain. Any other `s=10` pair has main-axis demand at most one, so one main
reduction and one secondary deletion suffice. At `g`, every containing
same-axis pair has main-axis `tau<=1`; again use one main reduction and one
secondary deletion. Thus every same-axis response, including the two finite
connectors, has an explicitly assigned service with handoff `TEMPO<=2`.

### 79.4 Distinct-axis, bridge, and stock-assisted responses

The only central one-trigger families which are not deleted outright are the
following adjacent-pair exterior cases. The stabilizers are chosen on the
stock-free side; this choice, rather than an arbitrary valid stabilizer, is
load-bearing for the later connector audit.

| central pencil | hard trigger | stabilizer | surviving tail |
|---|---:|---:|---|
| `s=0,s=1` | `-1` | `2` | parameters `-4,-3,-2` |
| `s=0,s=1` | `2` | `3` | parameters `-3,-2,-1` |
| `r=0` (`U^-`) | `8` | `9` | `q=3,4,5` |
| `q=10` (`V^-`) | `-5` | `-2` | `r=-8,-7,-6` |
| `q=10` (`V^-`) | `-2` | `-1` | `r=-7,-6,-5` |
| `s=10` (`W`, at `P_3`) | `11` | `10` | `q=14,15,16` |
| `s=10` (`W`, at `P_3`) | `14` | `11` | `q=15,16,17` |

Every one-trigger family on `q=0,q=1,s=6,s=7,r=-3` is easy: one empty common
residual deletes its complete high part. A singleton carrier hit by only one
response remains at count two; carriers containing both response cells are
classified explicitly below.

If neither response cell is `h` or `g`, each promotes at most one old
count-two axis. A third high axis would have to be the old count-one window
containing both response cells. Count-one extremes on a central axis are
already in the same-axis class. The remaining singleton-carrier incidence is
exhausted by this table; omitted infinite-line intersections lie outside the
six-window carrier.

| count-one carrier | empty intersections with live count-two axes |
|---|---|
| `q=6`, through `(6,0)` | `s=7` at `(6,1)` |
| `q=7`, through `(7,0)` | `s=6` at `(7,-1)` |
| `r=-4`, through `(10,-4)`, at `P_2` | `s=7` at `(11,-4)` |
| the same `r=-4` carrier at `P_3` | additionally `s=10` at `(14,-4)` |
| `r=-3`, through `(10,-3)`, at `P_2` only | `s=6` at `(9,-3)` |
| `q=12`, through `(12,-2)`, at `P_3` | `s=7` at `(12,-5)`; `r=-3` at `(12,-3)` |
| `q=13`, through `(13,-3)`, at `P_3` | none |
| `r=-2`, through `(12,-2)`, at `P_3` | `s=6` at `(8,-2)`; `s=7` at `(9,-2)`; `q=10` at `(10,-2)` |

For a hard response on one of these singleton carriers, normalize the old
Attacker rank to zero. The only consecutive response patterns are
`{-2,-1}`, `{-1,1}`, and `{1,2}`. In that order, choose the stock-free
orientation shown below; entries are the three physical coordinates of the
retained consecutive tail, not merely an infinite-line direction.

| carrier (parameter) | retained tails for the three patterns | complete tail envelope |
|---|---|---|
| `q=6` (`r`) | `[-5,-3]`, `[-4,-2]`, `[-3,-1]` | `[-5,-1]` |
| `q=7` (`r`) | `[1,3]`, `[2,4]`, `[3,5]` | `[1,5]` |
| `r=-4` through `q=10` (`q`) | `[5,7]`, `[6,8]`, `[7,9]` | `[5,9]` |
| `r=-3` through `q=10`, at `P_2` (`q`) | `[11,13]`, `[12,14]`, `[13,15]` | `[11,15]` |
| `q=12` through `r=-2`, at `P_3` (`r`) | `[-1,1]`, `[0,2]`, `[1,3]` | `[-1,3]` |
| `q=13` through `r=-3`, at `P_3` (`r`) | `[-2,0]`, `[-1,1]`, `[0,2]` | `[-2,2]` |
| `r=-2` through `q=12`, at `P_3` (`q`) | `[13,15]`, `[14,16]`, `[15,17]` | `[13,17]` |

Each displayed envelope misses every finite live transverse count-two union.
In particular, the tempting opposite orientation at `q=6` would meet `s=7`
at `(6,1)`, and the tempting negative orientation at `q=12` would meet
`s=7` at `(12,-5)`; neither is used.

There are exactly five nonconsecutive stock-bridge double-incidence patterns
at `P_3`: on `r=-4`, the pair of its `s=7` and `s=10` intersections; on
`q=12`, the pair of its `s=7` and `r=-3` intersections; and on `r=-2`, the
three pairs among its `s=6`, `s=7`, and `q=10` intersections. The carrier's
empty internal-gap deletion, followed by deletion of one easy central family
or stabilization of the unique hard one, leaves only an easy one-trigger
family in every case. The sole further transverse
leak is `g=(9,-3)`; its companion on `r=-3` is never one of that axis's exact
`tau=2` pairs. This is the complete nonconsecutive bridge ledger at `P_2`
and `P_3` (with absent `P_3` axes simply deleted at `P_2`).

Each capped bridge row `r=1,-1` has only one surviving count-one label. Its
one empty diagonal intersection is respectively `(-1,1)` on `s=0` and
`(2,-1)` on `s=1`. Thus even two returns promote only that singleton bridge
label, which one residual cell deletes; at most the one intersected central
axis remains, and the second cell stabilizes it.

A hard stock count-one bridge is a consecutive triple on its carrier. It
consumes both response cells and coexists with at most one central high axis;
their physical intersection is a response stone. Stabilize the bridge with
one cell using the oriented tail table above, and use the second cell to
delete the central family. There is one hard/hard exception at `P_3`:

`b={(10,-2),(11,-2)}` on the `r=-2` carrier.

Here `D@(9,-2)` leaves the bridge tail `q=13,14,15`, while `D@(10,-1)`
leaves the `V^-` tail `r=-7,-6,-5`. The two tails have no axial connector,
so their combined future demand is at most two. Every other coexisting
central family is easy and is deleted by the second cell. A nonconsecutive
bridge has an empty internal-gap blocker; the five double-incidence cases
were exhausted immediately above.

Thus a distinct-axis nonconnector response has at most two hard axes. Use one
stabilizer per hard axis, delete every easy axis first, and use a SAFE FILLER
outside every surviving tail support when fewer than two effective cells are
needed. If such a filler deletes a high label, remove that derivative from the
named list, exactly as required by binding Section 76.

For connector effects, distinguish an occupied connector from an empty one.
A response at `h` produces only the short one-trigger blocks of the exact
current-demand table; at `P_3`, delete the sole weight-two `s=10` block at its
gap `(11,-1)` unless the second response already lies on `s=10`, in which case
the same-axis cover applies. Stabilize the other response's hard family with
the second cell. A response at `g` produces only easy distance-three/four
families. This includes the legal double-connector response `(h,g)`.

If `h` remains empty between two activated incident axes, `D@h` simultaneously
cuts their hard families: the `r=0` family is reduced to starts `3,4`, either
hard `q=10` family acquires a common residual hit, and either hard `s=10`
family is reduced to one or two labels with a common hit. The second Defender
cell deletes the only remaining hard or bridge family. The axes incident with
an empty `g` are never hard, so `D@g` and the second effective cell give the
same conclusion.

With the displayed stock-free choices, the central tail envelopes are

```text
diamond: s=0 or 1, q in [-4,-1];
U^-:     r=0, q in {3,4,5};
V^-:     q=10, r in [-8,-5];
W:       s=10, q in [14,17].
```

No tail cell lies in a transverse pre-count-two label. Equal-level `U^-/V^-`
tail pairs occur at `s=3,4,5`, with axial spans `7,6,5`; hence only `s=5`
can lie in a common six-window. Equal-row `V^-/W` pairs occur at
`r=-7,-6,-5`, again with spans `7,6,5`; hence only `r=-5` is feasible.
Each feasible stock connector contains no old Attacker stone and at most one
response stone. The inherited parallel-diamond connectors have the same
pre-count-at-most-one property. Every other tail pair has no six-window
connector. Thus a future cell activates at most one retained atomic component,
a concentrated pair has demand at most two, and a pair avoiding every
retained high residual is in the pure-count-two class by (117)--(118) and the
exact table of Section 79.3. Every future pair has demand at most two.

Remote/remote responses are virgin pure-count-two stock. A local/remote split
is the corresponding one-trigger row of the table. The preceding cases also
cover two local split axes, count-one bridges, a same-axis pair with separated
effects, and every stock-assisted connector. They exhaust the infinite legal
response set by effect, not by sampled coordinates.

**Theorem R9.2 (post-`V^-` and post-`W` cap values) [Q3, PROVEN].** The exact
one-successor values are

`R_1(P_2^pl,a^dagger)=R_1(P_3^pl,a^dagger)=2`.           (119)

*Proof.* Sections 79.3--79.4 construct, after every legal response, an actual
servicing pair whose handoff has `TEMPO<=2`. The response

`b^dagger=((-1,1),(-1,2))`

and the service `((2,-2),(2,-1))` remain inside the original diamond support.
They create and then stabilize the same two parallel negative derivatives as
in R8.1. All `V^-` and `W` stock is outside their tail and connector ranges,
so the returned value remains exactly two. This supplies the reverse bound.
Every action at either plateau leaves at least two of the four original shield
pencils untouched, giving the universal risk floor two exactly as in R8.1.
Therefore `B_1(P_2^pl)=B_1(P_3^pl)=2` as well. ∎

**Corollary R9.2.1 (the earliest loss index) [Q3, PROVEN].** Define `k*` as
the least named plateau index at which no legal Defender action has
one-successor risk at most two. Then

`k*=4`.                                                   (120)

Indeed R8.1, R8.2, and R9.2 give safe actions at indices `0,1,2,3`, while
R9.1 proves that every action at index `4` has a response returning `M>=3`.
The inherited `P_5^pl=P_stock` stop theorem remains true but is no longer the
earliest known loss.

The transition mechanism is the first three-region packing. Through `P_3`,
the capped incidence quotient can retain at most two atomic demand regions.
Installing `V^+` supplies a second separated vertical block: an action missing
`H_4` permits the exact row/negative-vertical/positive-vertical witness of
L15.1, while an action touching `H_4` necessarily leaves one triangle fan
uncapped. This is why the loss appears at `4`, not merely because the scalar
inventory has grown.

## 80. The double-corner sealed continuation

### 80.1 The axial count-four quotient

Use the exact sealed handoff notation of L11.1:

```text
A on r=0 at q=0,1;    D@(0,1), D@(1,-1).
```

The three two-Q response occupancies with exact current demand two are

`{-2,-1}`, `{-1,2}`, `{2,3}`.                             (121)

The two orders of each occupancy are legal. All other first responses retain
the R5.1 bound `M<=2`, but are not folded into the exact second-cycle theorem
below.

**Lemma L15.3 (exact axial burst service) [Q2, PROVEN].** For every response
occupancy in (121), the returned epoch has exact `tau=2` and exact `M=0`.
The following three canonical unordered row covers for each response, with
both orders, are value-zero servicing actions:

| Attacker response | three canonical servicing occupancies |
|---|---|
| `{-2,-1}` | `{-4,2}`, `{-3,2}`, `{-3,3}` |
| `{-1,2}` | `{-3,3}`, `{-2,3}`, `{-2,4}` |
| `{2,3}` | `{-1,5}`, `{-1,4}`, `{-2,4}` |

At the middle response `{-1,2}`, these six orders are the complete
value-zero minimizing set on the R8.3 reached history.

*Proof.* In each row the four Attacker ranks are consecutive. Writing those
ranks as `{x-1,x,x+1,x+2}`, the three count-four windows have the standard
residual path

`{x-3,x-2}`, `{x-2,x+3}`, `{x+3,x+4}`,                  (122)

so current demand is exactly two. Direct interval containment shows that
each displayed cover meets every row window of count at least two, not only
the current family. The handoff therefore has `L_23=empty` and exact
`TEMPO=0`, proving `M=0`.

For `{-1,2}`, the complete graded start range is `-5,...,1`; its only
two-cell covers are exactly the middle row of the table. Every one of those
windows meets the original adjacent pair and hence lies in the untouched
launch footprint `B_j`, so remote root support cannot enlarge the minimizing
set. In the two exterior responses a newly born outer count-two window can
lie outside `B_j`; prior Defender contact may delete it and create additional
value-zero covers. The three canonical covers remain legal and sufficient,
but completeness is not claimed there. The last row follows from the first
by `rho(q,r)=(1-q,-r)`. ∎

**Theorem R9.3 (one further safe cycle for the axial-demand-two class) [Q2,
PROVEN at the stated response class].** After every value-zero minimizing
service in L15.3, every legal next Attacker pair returns an epoch with exact
`tau=0` and `M<=2`.

*Proof.* A value-zero handoff has `TEMPO=0`, which forces `L_23=empty`: any
surviving count-three label has a legal residual trigger, and any surviving
count-two label has two legal residual triggers in the same six-window, so
either would give a future response with `tau>=1`. This argument is
service-coordinate independent and therefore includes the extra exterior
minimizers created by root pruning.

A next Attacker pair can now raise a pre-count-one label only to count three
and a virgin label only to count two, so exact current demand is zero. Any
count-three label must be an old count-one label containing both new triggers.
Two distinct triggers determine at most one physical axis, so the complete
high family is one-axis. Its at-most-two-cell rank-triple cover deletes every
high label and leaves a pure count-two tier. L10.4 gives the claimed returned
bound. This class includes remote/remote, local/remote, split, bridge,
nonconsecutive, and same-axis second responses: the unique-axis argument is an
effect quotient over the whole legal pair set. ∎

This is one additional all-response safety layer beyond R5.1, but only after the
three exact first responses (121). It is not a theorem over every first
response from the sealed handoff.

### 80.2 The central second lozenge contracts to value one

Take the middle response

`b_C=((-1,0),(2,0))`.                                    (123)

After any of its six exact value-zero services, both outer row edges remain
unsealed. On the left edge play

`e_C=((-1,1),(0,-1))`.                                   (124)

The reflected right-edge choice is equivalent.

**Lemma L15.4 (complete central-lozenge action table) [Q2, PROVEN].** The
epoch after (124) has exact `tau=0,M=1`. Its row pencil is dead. Of its four
transverse adjacent-pair pencils, two are intact five-start blocks and the
two facing the original seals are singleton count-two blocks. The complete
immediate-minimizer family consists of one cap-depth cell on each intact
pencil, with relative axis parameter independently in

`{-2,-1,2,3}`,                                            (125)

in either order: exactly `4*4*2=32` ordered actions. Every one has exact
handoff `TEMPO=1`; every other legal action has exact handoff `TEMPO=2`.
For the displayed left edge, the two physical four-cell choice sets are

```text
q=-1: {(-1,-2),(-1,-1),(-1,2),(-1,3)},
s=-1: {(-3,2),(-2,1),(1,-2),(2,-3)}.
```

*Proof.* The two new cells are the empty common corners of the outer adjacent
edge and are independently legal from unchanged Attacker stock. The old row
service touches no transverse pencil away from an occupied lozenge vertex.
Each old seal truncates the inward pencil which contains it to its single far
common window; the outward two pencils are intact. The value-zero row service
killed every `r=0` label of count at least two, while off that row the complete
six-stone Attacker set has only the four transverse adjacent-pair pencils, of
alive count two. Thus current `tau=0` exactly.

An empty Defender cell lies on at most one of the four pencil lines because
all their nonparallel intersections are Attacker vertices. A cap cell at one
of the four depths (125) leaves a survivor block of size at most two; every
other depth leaves at least three starts. Thus a minimizing pair must and can
cap both intact pencils at the displayed depths. Afterward every count-two
component has one-turn demand one: two Attacker placements must be spent in
one component to make a current imminent, so different components do not add.
At least one inward singleton window survives every such cap pair, and two
legal residual placements in that window attain demand one. This gives exact
value one. If an intact pencil is not capped at one of those
depths, its three-start subblock has the standard demand-two witness, giving
exact value two; L10.4 supplies the reverse bound. ∎

The most natural second lozenge therefore contracts rather than producing a
stop state.

### 80.3 The exterior second lozenge reaches value two, not three

For the left exterior response and its second lozenge use

```text
b_L=((-2,0),(-1,0)),
e_L=((-2,1),(-1,-1)).                                   (126)
```

Choose any canonical row service from the first row of L15.3.
The cells of `e_L` are distance two from one another. Cell `(-2,1)` is
adjacent to unchanged Attacker stones `(-2,0),(-1,0)`, as is `(-1,-1)`;
both lie in untouched `B_j`. Thus either order is legal, neither cell was
occupied by the initial root action, and the first new stone is not used to
justify the second.

**Lemma L15.5 (exact exterior-lozenge plateau) [Q2, PROVEN on the R8.3
history].** The epoch after (126) has exact `tau=0,M=2`, and every legal
Defender action has exact handoff `TEMPO=2`.

*Proof.* The second pair makes four transverse adjacent-pair pencils; the
serviced row pencil is dead. The original two seals are two columns farther
right and meet none of the four new pencil lines. Every nonparallel
intersection of the four pencils is an occupied lozenge vertex, so an empty
Defender cell touches at most one pencil. A remote initial-root cell
could nevertheless prune a later extreme outside `B_j`. If both root cells
meet the same pencil, at most that pencil loses demand two. If they meet
distinct pencils, a five-start pencil drops below demand two from its single
contact only when that contact is at a cap depth. All cap-depth cells of the
four pencils lie in the untouched `B_j` except

`u=(-2,-2)` and `v=(-4,2)`.                              (127)

Relative to the selected anchor `(0,8)`, their distances are respectively
`12` and `10`; neither can be the first initial-root placement. Hence at most
one distinct pencil can fall below demand two, and at least three of the four
pencils retain future demand two. Every later Defender pair touches at most
two pencil lines, so one demand-two pencil remains untouched. This gives
`TEMPO>=2` after every action; the pure-count-two theorem gives
`TEMPO<=2`. The canonical row service killed every `r=0` label of count at
least two. Off `r=0`, the complete six-stone Attacker set has only the four
adjacent-pair pencils, of alive count at most two. Hence current `tau=0`
exactly. ∎

**Lemma L15.6 (delayed exterior cap) [Q2/Q3, PROVEN upper].** Translate the
outer edge by `(2,0)`, so the clean Attacker set is

`{(0,0),(1,0),(0,1),(1,-1),(2,0),(3,0)}`.               (128)

The translated cap `{(0,-1),(1,1)}` has all-response returned bound `M<=2`.
Its cells are distance two from one another, but each is independently
adjacent to the translated lozenge stock and lies in the untouched launch
footprint. Thus either order is legal without using the first cap stone to
license the second.

*Proof.* The old seals are `(2,1),(3,-1)`, and the preceding row service is
one of `{-2,4}`, `{-1,4}`, `{-1,5}`. In the clean inventory the cap leaves
`(n_1,n_2)=(10,12)`. A pair on one existing shield axis uses the rank cover.
Distinct local or split axes have exact `tau=0` and at most two high axes.
Local/local and local/remote separated-same-axis effects use the same
one-trigger reductions; their born response-pair bridge has only count two.
A one-local/one-remote response uses one-trigger stabilization, and a
remote/remote response is virgin pure-count-two stock. The only
stock-assisted three-axis response is the pair displayed next.

That sole response is

`{(2,-1),(2,-2)}`.                                      (129)

It has exact `tau=0`: its complete high family consists of count-three
labels on `s=1`, `s=0`, and `q=2`, with no count-four label.
Stabilize its `s=1` and `s=0` families at `(-1,2)` and `(-1,1)`. Together
with the truncation already supplied by old seal `(2,1)`, the three retained
nested tails are

```text
s=1: {(3,-2),(4,-3),(5,-4)},
s=0: {(3,-3),(4,-4),(5,-5)},
q=2: {(2,-3),(2,-4),(2,-5)}.
```

No transverse old count-two label meets the `q=2` tail; same-row connectors
to the diagonal tails contain no old Attacker stone; the only new `q=3`
connector is killed by old seal `(3,-1)`. A future pair therefore activates
at most two isolated value-one derivatives, while a pair avoiding every
retained high residual is pure count two. These classes exhaust the response
effects and prove the upper bound. If exterior root contact already occupies
a named cleanup cell, the family which required that cell is already deleted;
play the remaining effective cell first and use the standard legal maximum-`q`
safe filler outside every surviving derivative support. Thus root contact can
only prune the clean inventory without invalidating the Section-76 landing
bookkeeping. ∎

In the clean inventory (129) returns exact `M=2`. Before service, the `s=0`
and `s=1` components are intact consecutive-triple families of future demand
two. The old seal `(2,1)` deletes three of the four `q=2` starts, leaving only
start `-5` with residual `{-5,-4,-3}`, of future weight one. The three axes
meet only at the two response stones, so an empty Defender cell touches at
most one family. If a Defender pair does not touch both diagonal families,
one intact diagonal supplies demand two. If it touches both, it uses one cell
on each; the `q=2` label remains, and either once-touched diagonal retains a
count-three label by L13.1. One legal residual trigger in that diagonal and
one in the `q=2` label give two residual-disjoint singleton demands. Thus
every service hands over `TEMPO>=2`, while L15.6 supplies the reverse bound.
On the arbitrary R8.3 root history, outside pruning can lower that clean
value; only the all-response upper is proved, so no exact `R_1` value is
assigned.

### 80.4 Axial-subtree disposition

**Axial seal status [Q2, PROVEN at the stated class].** No `M>=3`
continuation occurs in the
complete value-zero-service axial-demand-two subtree. Its middle second
lozenge has exact value one, its exterior second lozenge has exact value two,
and R9.3 covers every second response after every value-zero burst cleanup.
Section 82 subsequently classifies the other forty-two row occupancies and
every off-row response drawn from the current graded residual support. The
axial theorem itself makes
no claim about virgin returns or value-one successor epochs.

## 81. An exact capped-renewal successor obstruction

### 81.1 An exact successor and all of its full-delete minimizers

Let `Q_1^cap=P_1^pl+a^dagger` be the exact Attacker handoff after the cap at
the one-stock plateau. Choose the legal response

`b_x=((-2,2),(0,2))`, and write `P_x=Q_1^cap+b_x`.       (130)

Both cells are independently legal from the unchanged stock, so both orders
reach the same occupancy.

**Lemma L15.7 (exact renewal-successor census) [Q3-repair, PROVEN].** At
`P_x` there is no count-four label and the complete count-at-least-three
family is:

| axis | count-three starts | exact residuals |
|---|---|---|
| `s=0`, parameter `q` | `-4,-3,-2` | `{-4,-3,-1}`, `{-3,-1,2}`, `{-1,2,3}` |
| `q=0`, parameter `r` | the interval start `0` only | `{3,4,5}` |

Thus `tau(P_x)=0`. The complete unordered actions which delete every high
label are

`d_j={(-1,1),(0,j)},  j in {3,4,5}`,                    (131)

with both orders, for six ordered actions. Every `d_j` is sequentially legal,
lands in the zero-derivative subcase of `C_cap`, and has exact handoff
`TEMPO=2`. Moreover,

`M(P_x)=2`,                                              (132)

so all six `d_j` are immediate-value minimizers. Completeness is asserted
only for the full-delete minimizers, not for the entire minimizing set.

*Proof.* The new cell `(-2,2)` promotes to count three exactly the three
displayed `s=0` windows. The new cell `(0,2)` promotes to count three only
the one surviving capped `q=0` window. Their common row `r=2` contained no
old Attacker stone, so the four labels receiving both cells finish at count
two; no pre-response count-one or count-two label receives both. This proves
the high census and exact `tau=0`.

The three `s=0` residuals have the unique common empty cell `(-1,1)`. The
vertical residual has exactly the three physical cells `(0,j)`,
`j=3,4,5`. Hence (131) is exactly the full-delete family. After such an
action every alive graded label has count two; no filler is used, so the
Section-76 safe-filler condition is automatic. The intact `s=1` pencil has
future demand two, while L10.4 gives the reverse bound, proving the exact
handoff value.

For the lower bound in (132), three action-disjoint demand-two regions remain
at `P_x`:

```text
s=0: trigger q=-1 leaves {-4,-3}, {-3,2}, {2,3};
s=1: triggers q=-1,2 leave {-3,-2}, {-2,3}, {3,4};
r=0: triggers q=5,8 leave {3,4}, {4,9}, {9,10}.
```

Each displayed residual path has hitting number two. The first two axes are
parallel; `r=0` meets them only at old Attacker cells `(0,0),(1,0)`. Thus an
empty Defender cell touches at most one region, and a Defender pair leaves at
least one region untouched. Every candidate action therefore hands over
`TEMPO>=2`; the six actions (131) attain two. ∎

### 81.2 Every full-delete minimizer fails at the next response

After any order of any `d_j`, Attacker can play, in either order,

`e_x=((-1,2),(8,0))`.                                   (133)

Cell `(-1,2)` is adjacent to the unchanged `r=2` pair, and `(8,0)` is
adjacent to stock stones `(6,0),(7,0)`. Neither belongs to any `d_j`, so each
is independently legal before the other is played.

**Theorem R9.4 (full-delete renewal obstruction) [Q3-repair, PROVEN].** For
every one of the six ordered full-delete minimizers in (131), the epoch after
(133) has exact `tau=0` and satisfies

`M(P_{x,d_j,e_x})>=3`; hence `R_1(P_x,d_j)>=3`.          (134)

*Proof.* The complete post-response high family consists of exactly twelve
count-three labels and no count-four label:

| axis and Attacker triple | starts | exact residuals, in axis parameter |
|---|---|---|
| `r=2`, `q=-2,-1,0` | `-5,-4,-3,-2` | `{-5,-4,-3}`, `{-4,-3,1}`, `{-3,1,2}`, `{1,2,3}` |
| `s=1`, `q=-1,0,1` | `-4,-3,-2,-1` | `{-4,-3,-2}`, `{-3,-2,2}`, `{-2,2,3}`, `{2,3,4}` |
| `r=0`, `q=6,7,8` | `3,4,5,6` | `{3,4,5}`, `{4,5,9}`, `{5,9,10}`, `{9,10,11}` |

Neither cell of any `d_j` lies in these finite unions. The three physical
axes meet pairwise only at Attacker cells: `r=2` and `s=1` meet at new stone
`(-1,2)`, `r=0` and `s=1` at old stone `(1,0)`, and the two rows are
parallel. Hence an empty Defender cell touches at most one family.

Each untouched family has an exact one-trigger demand-two flank:

```text
r=2, trigger q=-3: {-5,-4}, {-4,1}, {1,2};
s=1, trigger q=-2: {-4,-3}, {-3,2}, {2,3};
r=0, trigger q=5:  {3,4}, {4,9}, {9,10}.
```

If a candidate Defender pair touches zero or one family, two untouched
families give the lower bound `2+2`. If it touches two distinct families,
the third remains untouched with demand two, and either once-touched family
retains a count-three label with a legal one-trigger demand-one response.
The two residual grounds are on distinct axes and intersect only at occupied
Attacker stones, so the combined hitting number is at least `2+1=3`. These
cases exhaust every Defender pair and prove (134). No upper value is assigned:
lower stock can add demand outside this focal family. ∎

**Renewal status [Q3-repair, OPEN with a proved obstruction].** R9.4 refutes
the naive iteration of every full-delete, zero-derivative service chosen by
R8.4 at the exact successor (130). It does not prove `B_1(P_x)>=3` because
other immediate minimizers may proactively truncate the `r=2` or `s=1`
future flank. Whether some immediate minimizer `d` at `P_x` satisfies
`R_1(P_x,d)<=2` remains open; therefore two-cycle renewal is not claimed and
the obstruction is not promoted to a universal unrepairability theorem.

## 82. The finite `X ∪ N` residual-return seal quotient

### 82.1 All forty-two lower-demand two-Q occupancies

Let

`X_0={-5,-4,-3,-2,-1,2,3,4,5,6}`, and `W_s=[s,s+5]` on `r=0`.  (135)

The sealed state already occupies ranks `0,1`. Thus the `45` unordered
two-Q occupancies choose two ranks from `X_0`; L15.3 handled the three in
(121). Every off-row label has count at most one after any such response, so
the complete graded tier is the surviving row family.

**Lemma L15.8 (complete lower-demand two-Q quotient) [Q2, PROVEN].** Among
the other forty-two occupancies, thirty-eight have exact `M=0` on every
arbitrary R8.3 root history. The remaining four have this exact
root-pruning dichotomy:

| response ranks | canonical unordered service | sole possible survivor | exact `M` |
|---|---|---|---:|
| `{-5,-4}` | `{-3,2}` | `W_-9`, residual `{-9,-8,-7,-6}` | `1` if alive, else `0` |
| `{-4,-3}` | `{-2,2}` | `W_-8`, residual `{-8,-7,-6,-5}` | `1` if alive, else `0` |
| `{4,5}` | `{-1,3}` | `W_4`, residual `{6,7,8,9}` | `1` if alive, else `0` |
| `{5,6}` | `{-1,4}` | `W_5`, residual `{7,8,9,10}` | `1` if alive, else `0` |

All displayed services are legal in both orders. For the full forty-two-case
current-demand census, excluding the three exact-`tau=2` pairs (121),
`tau=1` exactly in the following cases and `tau=0` otherwise:

1. same-left ranks `x<y<0` with `x>=-4`, excluding `{-2,-1}`;
2. same-right ranks `1<u<v` with `v<=5`, excluding `{2,3}`; and
3. cross-side ranks `{-a,1+b}`, `1<=a,b<=5`, with `a+b<=4`, excluding
   `(a,b)=(1,1)`.

*Proof.* Four sorted Attacker ranks have a count-at-least-two window only
through a consecutive pair in their sorted order. Consequently the complete
pre-service start support `F_2` is

| response form | complete start interval `F_2` |
|---|---|
| `x<y<0` | `[y-5,0]` |
| `1<u<v` | `[-4,u]` |
| `{-a,1+b}` | `[-5,1]` |

A reached root history may delete exterior members of these intervals; the
actual support is the corresponding subset. All services below cover the
maximal interval, so the upper bounds are monotone under that pruning.

A row placement at rank `d` deletes exactly the start interval `[d-5,d]`.
For a same-left response choose

`l in [-4,y-1] minus {x}`, and service `{l,2}`.

Such an `l` exists except for `{-5,-4}` and `{-4,-3}`. For a same-right
response use the reflected choice

`r in [u+1,5] minus {v}`, and service `{-1,r}`.

It fails only for `{4,5}` and `{5,6}`. For a cross-side response use

```text
l=-1, except l=-2 when a=1;
r= 2, except r= 3 when b=1.
```

The two deletion intervals cover the corresponding complete `F_2` in every
listed nonexceptional case. Hence `L_23=empty` and the handoff has exact
`TEMPO=0`, proving `M=0`. Each effective cell is empty and independently
legal from unchanged row stock under the radius-eight rule, so both orders
are legal. Every displayed cleanup cell lies in untouched `B_j` and is
therefore empty on the reached history. Exterior root pruning only deletes
additional labels, so the same two-cell service remains valid.

The count is `7` nonexceptional same-left, `7` same-right, and `24`
cross-side occupancies after removing the three cases (121), for `38` total.

The current-demand formula follows from the same interval list. Four ranks
fit in one six-window precisely under the three displayed inequalities. The
only cases whose count-four residual path needs two hits are the four
consecutive ranks in (121); every other nonempty current family has a common
empty hit. Every current window contains the old pair `0,1`, so it is one of
the five robust `B_j` row windows and cannot be lost to exterior root pruning.

In an exceptional row the canonical service deletes every graded label except
the displayed exterior window. If that window was already root-deleted, the
same canonical pair deletes every remaining graded label and gives exact
`M=0`. Suppose it survives. For `{-5,-4}`, an empty hit on `W_-9` has rank
at most `-6` and
hits none of the robust block `W_-4,...,W_0`; that block's common intersection
is `{0,1}`, both occupied. For `{-4,-3}`, an empty hit on `W_-8` lies in
`[-8,-5]`; at best `-5` also hits `W_-5`, while the remaining block
`W_-4,...,W_0` has common intersection `{0,1}`, both occupied. Thus no empty
two-cell transversal deletes the complete graded family. Reflection proves
the right-side cases. The canonical service leaves one count-two label, whose
two residual triggers attain future demand one, proving exact `M=1`. ∎

**Corollary R9.5 (root-robust second cycle for forty-one of forty-five row
returns) [Q2, PROVEN at the stated class].** The three responses (121) and the
thirty-eight robust value-zero responses of L15.8 have the R9.3 successor property: after
every value-zero minimizing service, every next Attacker pair has exact
`tau=0` and returns `M<=2`.

On a root history which has already deleted an exceptional exterior window,
that response also has `M=0` and every value-zero minimizer inherits the same
R9.3 property. Thus `41` is the number of root-robust occupancy classes, not
an upper bound on how many classes can be safe on one particular history.

After the displayed canonical service, when its exceptional exterior window
survives, a next Attacker pair has exact `tau=1` iff both cells lie in that
four-cell residual, and exact `tau=0` otherwise. Its complete all-response
`M` quotient is not proved.

### 82.2 Every Q/arm off-row occupancy

In physical coordinates define the ten Q returns and twenty local arm
returns by

```text
X={(-k,0),(1+k,0): 1<=k<=5},
N={(0,-l),(-l,l),(1,l),(1+l,-l): 1<=l<=5}.
```

(136)

**Lemma L15.9 (complete Q/arm off-row seal quotient) [Q2, PROVEN].** The
following exact values hold on every R8.3 reached sealed history.

1. For every `x in X,y in N`, both response orders are independently legal,
   the returned epoch has exact `tau=0,M=1`. This is `200` occupancies and
   `400` ordered responses.
2. For every two distinct `y,z in N`, both orders are independently legal,
   the returned epoch has exact `tau=0,M=1`. This is `190` occupancies and
   `380` ordered responses: `40` same-arm and `150` split-arm.

Every cell of `X ∪ N` is an empty residual cell of an alive sealed-state
label before the response. Thus each member of a chosen pair is independently
legal from unchanged Attacker stock; neither order relies on the first new
stone to license the second.

*Proof, one Q plus one arm.* A positive Q return has maximal graded row
starts `-4,...,1`; exterior root pruning may delete start `1`, but the five
robust starts `-4,...,0` remain and have only occupied common cells `0,1`.
Thus deleting the row tier needs two row cells. The arm
return promotes its unique alive extreme to count two; its carrier meets the
row only at an occupied original endpoint. Hence a value-zero action would
need two row cells and a third off-row cell, proving `M>=1`.

There are exactly twenty born response-pair bridges. For
`x=(1+k,0)`, they occur at `y=(1,k)` and `y=(1+k,-k)`, `1<=k<=5`, together
with their `rho` images. In a nonbridge case delete the entire positive-side
row tier by `{-1,3}` for `k=1` and by `{-1,2}` for `k>=2`; use the reflected
actions on the negative side. The sole promoted arm label remains and has
future demand one.

In a bridge case use the one-cell row stabilizer at row rank `-1` for depths
`k=1,5` and at rank `2` for `k=2,3,4`, reflecting on the negative side. For
bridge endpoints normalized to ranks `0,k`, the start set is `k-5,...,0`.
When `k<=3`, its empty rank `-1` deletes every start except `0`; at `k=4,5`
the bridge already has future value one. The physical rank-`-1` cap is
`(0,k+1)` for `y=(1,k)` and `(1+k,-k-1)` for
`y=(1+k,-k)`. Every named row or bridge cleanup cell lies in untouched `B_j`
and is empty on the reached history. Exterior root pruning can only delete
additional born bridge windows. When no effective bridge cap remains, play
the row stabilizer first and use the standard legal maximum-`q` filler outside
every surviving component; if the filler deletes a named component, remove
that component from the handoff ledger.

Every nonfiller service cell above is independently within distance at most
five of unchanged Attacker stock: row cells lie in a surviving row window,
and bridge caps lie in or one step beyond the response carrier. Thus both
effective service orders are legal.

The retained row component is the only possible count-three component. Every
other surviving component is count two with future value at most one, and
distinct carriers meet only at occupied original or response stones. A
future pair therefore cannot mature two carriers simultaneously. The displayed handoff has exact
`TEMPO=1`, proving the upper bound and equality.

*Proof, two arms.* Distinct arms promote only count-two labels. Two returns
on one arm promote its unique extreme to count three but never count four, so
`tau=0` in all `190` cases. Cap the unchanged five-window row block at
`(-1,0)`, leaving only `W_0`.

For same-arm depths `k<l`, normalize the arm to ranks `0,...,5`. Its graded
starts are start `0` at count three and starts `1,...,k` at count two. Use
rank `5` if `l<5`, rank `4` if `l=5,k<4`, and rank `3` for `(k,l)=(4,5)`.
This deletes the count-three label and leaves at most one count-two start.
Each cleanup rank lies in an alive graded window, is distinct from both
response cells, and is independently legal with the row cap; both service
orders are therefore legal.

For split arms, each promoted extreme is a singleton count-two component. The
response pair creates a bridge exactly in these eight occupancies:

```text
{(0,-k),(1+k,-k)} or {(-k,k),(1,k)},  k=1,2,3,4.
```

The other `142` split-arm pairs are nonbridges. Normalize a bridge's endpoints
to `0,d`, where `d=k+1` lies in `{2,3,4,5}`. Rank `-1` deletes every bridge
start except `0` for `k<=3` (`d<=4`), while `k=4` has `d=5` and is already a
singleton. At `k=5` the span is six, so no six-window bridge exists. Named
caps lie in untouched `B_j`; exterior pruning only helps. If no effective
bridge cap is required, play the row cap first and use the Section-76 safe
filler outside every surviving support. Every effective bridge cap lies in an
alive bridge window or one step beyond its response carrier and is
independently within distance five of unchanged stock, so it is legal in
either order with the row cap.

All surviving carriers have future value at most one and no count-three
derivative. Two nonparallel carriers share at most one empty cell, whereas
maturation from count two needs both future triggers, so different carriers
cannot mature together. The handoff has `TEMPO<=1`. Value zero is impossible:
the five row windows require two row cells, while at least one promoted
off-row label survives those cells because its row intersection is an
occupied original endpoint. Therefore `M=1` exactly. ∎

### 82.3 Exact sealed boundary after the finite quotient

**Seal-cycle status [Q2, PARTIAL].** The entire finite residual-return
quotient on `X ∪ N` is now exact: all `45` two-Q occupancies, all `200`
Q-plus-arm occupancies, and all `190` two-arm occupancies, with both orders.
This is `435` unordered and `870` ordered first responses. Forty-one of the
forty-five row occupancies are root-robust for a complete next all-response
`M<=2` safety bound.

The remaining obstructions are narrower but genuine:

1. continuation after every minimizing landing for the four surviving
   exceptional responses of L15.8; only the displayed canonical singleton
   landing has the stated next-`tau` test;
2. continuation after the exact value-one Q/arm off-row landings of L15.9;
3. one or two genuinely virgin returns, including local/remote and
   remote/remote pairs; a virgin response-pair bridge can cross the old Q row
   at an empty cell, so its surviving bridge-start subset and crossing rank
   cannot be discarded; and
4. generalized-lozenge continuation outside the sealed branch.

No universal two-cycle seal theorem and no Q2 root-forcing conclusion is
claimed.

## 83. Authoritative round-9 status ledger

This ledger supersedes the round-8 nine-obstacle exit state only where a row
below records a new proof. An inequality is not silently promoted to an exact
value, and a reached-state theorem is not promoted to a root theorem.

| Claim / obstacle | Quantifier tag | Round-9 status | Exact basis / remaining scope |
|---|---|---|---|
| `GAP-RAW` | Q2 counterroute / Q3 target | **OPEN** | Neither root forcing nor an all-history Defender policy is proved |
| Earliest stock loss index | Q3 | **PROVEN: `k*=4`** | Exact cap risks two at `P_0^pl`--`P_3^pl`; every action is unsafe at `P_4^pl` |
| Capped prefix inventories | Q3 diagnostic | **PROVEN EXACT** | `(8,12),(33,16),(49,26),(69,34),(95,39),(115,47)`, with no count at least three |
| `P_0^pl,P_1^pl` cap values | Q3 | **PROVEN EXACT, inherited** | `R_1=B_1=2` by R8.1--R8.2 |
| `P_2^pl,P_3^pl` cap values | Q3 | **PROVEN EXACT** | `R_1=B_1=2` by R9.2; all response effects enumerated |
| `P_4^pl` phase transition | Q1/Q3 | **PROVEN UNSAFE** | Every action has a response with returned `M>=3`; for the cap, `R_1=B_1=3` exactly |
| `P_5^pl=P_stock` | Q1/Q3 | **UNSAFE, inherited** | Binding R7.2: every action has a response with returned `M>=3` |
| Double-corner seal, whole branch | Q2 | **PARTIAL / OPEN** | Entire finite residual-return quotient on `X ∪ N` is exact; value-one successors and virgin returns remain |
| Axial seal continuation | Q2 | **PROVEN at exact class** | L15.3 has exact `tau=2,M=0`; after every value-zero service, R9.3 gives exact next `tau=0` and an all-response `M<=2` bound |
| Other forty-two two-Q returns | Q2 | **PROVEN EXACT** | Thirty-eight have robust `M=0`; four have the exact root-dependent `M=0/1` exterior-window dichotomy |
| Forty-one root-robust row-return successors | Q2 | **PROVEN safety bound at exact class** | R9.3/R9.5 give exact next `tau=0` and returned `M<=2` after every value-zero service; pruned exceptional rows may add classes |
| Q/arm off-row first responses | Q2 | **PROVEN EXACT** | `400` ordered Q-plus-arm and `380` ordered two-arm responses all have exact `tau=0,M=1` |
| Four exterior singleton successors | Q2 | **OPEN** | Next `tau` is exact, but the all-response returned-`M` quotient is not classified |
| Q/arm off-row value-one successors | Q2 | **OPEN** | L15.9 classifies the landing, not its next arbitrary response tree |
| Central second lozenge | Q2 | **PROVEN EXACT** | `tau=0,M=1`; all 32 ordered immediate minimizers enumerated |
| Exterior second lozenge | Q2 | **PROVEN EXACT at the reached history** | `tau=0,M=2`; every next action has exact handoff `TEMPO=2` |
| Delayed exterior cap | Q2/Q3 | **PROVEN UPPER** | All-response `M<=2`; clean stock-assisted witness has exact `M=2`, arbitrary-root exact risk unassigned |
| Generalized-lozenge continuation | Q2 | **OPEN** | Arbitrary next actions at all non-seal R8.3 plateaux remain unclassified |
| Continuation after the minimizing raw branch | Q2/Q3 | **OPEN** | Exact `P_0^pl` repair does not initialize a perpetual policy or Q2 force |
| R8.4 first capped landing | Q3-repair | **PROVEN, inherited** | Exact `P_0/P_1` inputs land in `C_cap` under the binding safe-filler clause |
| Exact renewal successor `P_x` | Q3-repair | **PROVEN EXACT locally** | `tau=0,M=2`; all six full-delete minimizers enumerated |
| Reuse of a full-delete minimizer at `P_x` | Q3-repair | **REFUTED** | Every one has `R_1>=3` by the same exact legal response (133) |
| Some alternative minimizer at `P_x` has risk at most two | Q3-repair | **OPEN** | Proactive truncation actions are not classified |
| Arbitrary-member `C_cap` renewal | Q3-repair | **OPEN** | R9.4 is an obstruction at one reached member, not a closure or impossibility theorem |
| Other repair geometries and forced seal entrances | Q2/Q3 boundary | **OPEN** | Shared/nonshared fanouts, cross-hull interactions, later derivatives, and alternative entrances remain |
| General count-three initialization | Q3 | **OPEN beyond L13.6** | Strict `tau=0` families with hitting number at least three remain excluded |
| Root-level Q2 / positive Q3 conclusions | Q2/Q3 | **OPEN** | No all-strategy loss force and no initialized all-history repair policy |
| R5.2 separation sharpness | ancillary | **OPEN** | Radius 21 remains envelope-sharp only |
| New machine verification | methodological | **NONE** | Every round-9 result is a hand proof |

The round-8 obstacle list therefore changes as follows: obstacle 1 is closed
with `k*=4`; obstacle 3 now has a complete finite residual-return quotient on
`X ∪ N` and forty-one row-return safety bounds, but remains open at the
value-one and virgin successors; obstacle 2 gains the full-delete obstruction R9.4 but remains open;
obstacles 4--9 retain their prior open status.

## 84. Round-9 attack surface

The following points are deliberately exposed for review.

1. **Plateau identity and reachability.** `P_i^pl` are six different exact
   statewise tests. The safe cap is not replayed on one history, and the
   `P_4` stop is not asserted to be reached by every root strategy.
2. **Inventory arithmetic.** Recheck both `n_1,n_2` and the finite-axis
   census (115)--(116), especially singleton-to-pair conversions at `W` and
   the later stock increments.
3. **Finite intersections.** The proof uses finite six-window unions, not
   infinite carrier lines. Its only empty central connectors are `h` and
   `g`; an omitted connector would invalidate the split quotient.
4. **Exact same-axis table.** Section 79.3 lists every `tau=2` pair. The
   `h`-containing same-axis cases intentionally do not use the naive full
   rank cover; `D@(11,-1)` and a one-cell main reduction are load-bearing.
5. **Singleton-carrier orientation.** The seven oriented tail envelopes in
   Section 79.4 are physical intervals. Reversing the `q=6` or `q=12`
   orientation creates a live transverse leak.
6. **Bridge exhaustiveness.** The capped `r=1,-1` rows are singleton
   families, the five nonconsecutive stock double-incidences are explicit,
   and `b={(10,-2),(11,-2)}` is the sole hard/hard exception. Deleting that
   bridge with two cells is unsafe; the displayed two one-cell stabilizers
   are required.
7. **Tail connectors.** Only `s=5` for `U^-/V^-` and `r=-5` for `V^-/W`
   fit in a six-window. Each has no old Attacker stone and at most one response
   stone. Equal-line pairs at spans six and seven are not connectors.
8. **Safe fillers.** Effective cells are ordered first. A spare cell must
   satisfy Section 76; if it deletes a named derivative, that derivative is
   removed from the landing ledger. R9.4's `d_j` use no filler.
9. **`P_4` dichotomy.** The universal lower bound separates actions missing
   `H_4` from actions touching it. Its `W_sat=[8,15]` segment includes both
   outer service ranks `14,15`; using only `[8,13]` would leave a boundary
   deletion gap. Only the focal miss branch and the cap risk receive exact
   value three; the inherited fan branch is used only as a lower bound.
10. **`P_4` upper quotient.** The separated `V^- / V^+` blocks share the
    one physical `q=10` effect class. The `h`-containing subclass separates
    `t=1` from `t` not equal to `1`; the former service has exact
    `TEMPO=3` from `U` weight one plus `W` weight two. Its explicit services
    are load-bearing. The cap upper is three, not a claim that every response
    produces three.
11. **Two-Q seal quotient.** The three `tau=2` pairs plus L15.8 exhaust all
    `45` occupancies. The thirty-eight value-zero cases are root-robust; the
    four exterior cases retain an explicit root-dependent singleton window,
    so their `M=0/1` dichotomy must not be collapsed.
12. **Local off-row enumeration.** L15.9 covers exactly `200` Q-plus-arm and
    `190` two-arm occupancies. Its twenty born Q/arm bridges, forty same-arm
    pairs, 150 split-arm pairs, and exactly eight split-arm bridges are
    separate service classes. Named caps lie in untouched `B_j`; virgin cells
    are not silently included in `N`.
13. **Seal successor scope.** R9.3/R9.5 cover every value-zero service for
    forty-one row occupancies. They do not cover every minimizing landing of
    the four surviving exceptional responses, the value-one off-row landings,
    or virgin return bridges; only the canonical singleton landing has the
    displayed next-`tau` test.
14. **No-corner legality.** The two second-lozenge corner cells and the two
    delayed-cap cells are each distance two from their partner. Legality is
    justified because each cell is independently within range of unchanged
    old Attacker stock, not because the first newly placed stone licenses the
    second.
15. **Root pruning in the seal.** The middle burst has a complete six-order
    minimizer set. Exterior root contact can add minimizers, so only three
    canonical unordered covers are claimed there. The exceptional cells
    `u,v` in L15.5 cannot both be initial-root placements.
16. **Clean versus arbitrary delayed cap.** Equation (129) gives exact
    `M=2` only in the clean inventory. Monotone exterior pruning proves an
    upper on the actual root history; no exact arbitrary-history `R_1` is
    assigned.
17. **Renewal minimizer scope.** Equation (131) is the complete family that
    deletes every high label, and all six are minimizers. It is not claimed
    to be the complete immediate-minimizer family.
18. **Renewal value scope.** After (133), `tau=0` and the twelve count-three
    labels are exact; only `M>=3` is proved. No unaudited upper is folded into
    R9.4.
19. **Infinite responses.** All all-response claims close remote/remote,
    local/remote, split, bridge, same-axis, and stock-assisted effects by
    class. Coordinate lists are finite representatives of effects, not a
    sampled board search.
20. **Proof method.** No executable result, solver output, or machine count is
    evidence for any statement in this artifact.

## 85. Provenance, artifact identity, and no-run record

- **Input branch:** `hunt/gap-raw`.
- **Input commit:** `c019400ad14e06fa6f600c5462113a74795e3270`
  (`c019400a`).
- **Shared-branch drift:** a final read-only check found that unrelated work
  had advanced the shared branch to `ab0fd965f2c8be07d373c0426e7a457c21f4700a`
  after this pass began. No reset or checkout was performed. Every theorem in
  this artifact remains anchored to the required `c019400a` input corpus.
- **Artifact:** `GAP_RAW_PROOF_ROUND9.md`.
- **Landed artifact hash:** `LANDED_ARTIFACT_HASH: <TO-BE-FOLDED-POST-REVIEW>`.
  This is intentionally a placeholder for the post-review landed identity.
- **Commit status:** no commit was created; the user explicitly prohibited
  committing.
- **Required corpus order:** round-6 proof through Section 58, round-6 review,
  round-7 proof through Section 67, round-7 review, round-8 proof through
  Section 76, then round-8 review, each read in full and in that order.
  Round-5 Sections 38--39 were consulted only for the inherited sealed-state
  coordinates and service lemmas.
- **Excluded corpus:** no `STRATEGY_STEALING_*` file was read.
- **Execution record:** no Cargo command, Lean command, harness, game/search
  program, machine enumeration, solver, or test was run. File inspection and
  independent hand-audit passes supplied the proof review; all mathematical
  derivations in this artifact are paper arguments.
