# R-G8-REV — Round-10 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`,
`8424e9a474f160e4d30ec73cf6b655e1da741554`

**Reviewed document:** `GAP_RAW_PROOF_ROUND10.md`

**Artifact's named authoring input:**
`f5349d3eb985cdb9ee719ec75272f3a73772604d`

**Artifact identity:** Git blob
`c3557d1e63241c1585ef52efb16158313219d0ff`; SHA-256
`309077856d7df8ff2a82b48c913c0c456b4ce37eae3039c45c06851bb6d4959c`.

**Method.** First-principles hostile proof audit, with the positive universal
bounds presumed false until their response quotients closed. I read the
required corpus first, in the stipulated order and in full:
`GAP_RAW_PROOF_ROUND7.md` through binding §67, then
`GAP_RAW_REVIEW_ROUND7.md`; `GAP_RAW_PROOF_ROUND8.md` through binding §76,
then `GAP_RAW_REVIEW_ROUND8.md`; `GAP_RAW_PROOF_ROUND9.md` through binding
§86, then `GAP_RAW_REVIEW_ROUND9.md`; and finally the reviewed round-10
artifact. I afterward consulted only the needed `TEMPO`/`M` definitions in
round 4 and the exact plateau/stock passages in round 6. No
`STRATEGY_STEALING_*` file was read or used as evidence.

Every inventory, finite carrier incidence, residual, service, legality
chain, `tau`, `TEMPO`, and `M` comparison below was recomputed by hand. No
Cargo command, Lean build, harness, game/search program, solver, generated
enumeration, or test was run. Read-only Git commands were used only to check
identity. No Git commit was created. A later read-only check found that the
shared worktree had advanced to
`1bdaf02a9ea631f399fd99d9d8049c973de5898c`, but all seven required GAP-RAW
proof/review files were unchanged and the working round-10 file retained the
blob above.

**Overall verdict:** **SOUND-WITH-MINOR-ERRATA.** The hostile audit found no
legal response to the cap with returned `M>=4`, no missed `M>=3` response in
any seal class claimed bounded by two, and no error in the exact `48/96`
proactive-renewal boundary. Target 1's headline is confirmed:

`B_1(P_3^pl)=3`.

The only defect is documentary: §94 retains a landed-artifact placeholder
instead of recording reviewed commit `8424e9a4...`. The input identity is
present and correct.

## Numbered findings

### 1. NOTE — the cap inventory and response-enumeration architecture are complete

**Quoted claims:**

> “After (138), binding L15.2 gives the exact profile
> `(n_1,n_2)=(69,34), n_j=0 for j>=3`.” (§88)

> “Every returned high label after a response `b=(x,y)` has exactly one of
> two sources” — an old count-two label hit by a response cell, or an old
> count-one label containing both cells. (L16.1)

**Independent recomputation.** The cap

`a^dagger=((0,-1),(1,1))`

is sequentially legal at exact `P_3^pl`. Starting from its exact diamond
inventory `(8,12)`, the three installed stock increments are

| addition | `Delta n_1` | `Delta n_2` |
|---|---:|---:|
| `U^-` | `25` | `4` |
| `V^-` | `16` | `10` |
| `W` | `20` | `8` |

so the cap leaves `(n_1,n_2)=(69,34)` and no higher label. The count-two
carrier multiplicities recompute as

```text
q=0:1, q=1:1, s=0:5, s=1:5, r=0:4,
q=10:5, s=6:2, s=7:3, s=10:5, r=-3:3,
```

which sum to `34`. Count-one extremes on those ten carriers contribute `31`;
the remaining singleton carriers

```text
r=1, r=-1, q=6, q=7, r=-4, q=12, q=13, r=-2
```

contribute `38`, giving `n_1=69`.

Because the pre-response maximum count is two, L16.1's two sources are
exhaustive. Direct intersection of the ten finite carrier unions gives only

```text
h=(10,0) in r=0,q=10,s=10,
g=(9,-3) in s=6,r=-3
```

as empty multi-carrier cells. The singleton-incidence table also recomputes.
In particular, the repaired boundary point `(5,-4)` is present, but its
distances six and nine from the other `r=-4` incidences prevent a common
length-six window. The only singleton-carrier pairs using two distinct
finite-carrier incidences are exactly

```text
{(11,-4),(14,-4)},
{(12,-5),(12,-3)},
{(8,-2),(9,-2)},
{(8,-2),(10,-2)},
{(9,-2),(10,-2)}.
```

Thus the source states the demanded architecture before assigning a value.
The quotient by `j=|{x,y} intersect K_3|` covers remote/remote (`j=0`),
local/remote (`j=1`), and every same-axis, split, bridge, `h/g`, and
stock-assisted local pair (`j=2`). It is an effect partition of the infinite
legal response set, not a finite sample.

**Proposed repair:** none.

### 2. NOTE — Target 1's exact minimum and its Q1/Q3 scope follow

**Quoted claim:**

> “At the exact `P_3^pl`,
> `R_1(P_3^pl,a^dagger)=B_1(P_3^pl)=3`.” (R10.1)

**Independent recomputation.** Findings 1 and 6--8 prove that the cap has
all-response upper three and that `b_*` attains three. Binding (R9-REV-4)
gives `R_1(P_3^pl,a)>=3` for every legal initial action. Binding §58 is also
used correctly: every action at this low-only shield plateau is servicing and
has exact immediate handoff `TEMPO=2`, so the minimizing set in the definition
of `B_1` is the complete legal action set. The cap therefore attains the
universal floor and the exact minimum is three.

The theorem is explicitly Q1/Q3 and reached-state only. No forcing route from
an arbitrary strict root, exact risk for every other P3 action, or Q2 result is
inferred.

**Proposed repair:** none.

### 3. NOTE — the canonical exterior-successor `M<=2` bound is universal at its stated class

**Quoted claim:**

> “From each displayed canonical value-one handoff, every legal next
> Attacker pair returns an epoch with `M<=2`.” (L16.4)

**Independent recomputation.** Consider the leftmost representative. After
response `{-5,-4}` and canonical service `{-3,2}`, the sole alive count-two
row label is `W_-9`, with residual `{-9,-8,-7,-6}`; the only possible outer
count-one neighbor is `W_-10`. The second left row has the same one-axis
interval calculation shifted at its exterior component; the two right rows
are the `rho`-images of the corresponding left rows.

The quotient by the number `j` of next returns on the old Q row is exhaustive.

1. For `j=2`, only the sole count-two row label and its optional count-one
   neighbor can become high. One empty residual representative from each
   services any count-four label and deletes the complete high family.
2. For `j=1`, only the sole count-two row label can be high. The non-Q carrier
   through the two response cells meets Q at the newly occupied row-return
   cell, which was empty before the response; it therefore had no old
   Attacker support and cannot be a promoted count-one bridge.
3. For `j=0`, any high labels form a deletion-subfamily of the intervals
   through the unique rank triple consisting of the response pair and at
   most one old row stone. The rank-triple cover uses at most two cells.

Each constructed action is legal, services the current family, and leaves a
pure count-two graded tier, so L10.4 gives `TEMPO<=2`. This includes aligned,
nonaligned, local/remote, remote/remote, and root-pruned responses. Only the
sole pre-count-two row label can reach count four, so the exact current
demand condition (147) also holds.

The class boundary is honest. If the displayed exterior count-two window was
already root-deleted, R9.5's value-zero branch applies. L16.4 does not claim
to enumerate noncanonical first services or the exact `M=0,1,2` strata after
the canonical landing.

**Proposed repair:** none.

### 4. NOTE — the two-virgin quotient is exact `M in {0,1}`

**Quoted claim:**

> “For every legal two-virgin response,
> `M=0` iff some `C in C_Q` meets every member of `B(v,w)`;
> `M=1` otherwise.” (R10.2)

**Independent recomputation.** Virginity prevents either return from
promoting an old alive sealed label. Hence the complete returned graded tier
is exactly the five robust labels `W_-4,...,W_0` plus the actual born family
`B(v,w)`, all at count two; current `tau=0`. If the returns have an alive
common carrier `L`, it is unique. At normalized ranks `0<d<=5`, root pruning
is exactly a subset of starts `{d-5,...,0}`.

One Q cap leaves at most one robust Q label. On `L`, carrier rank `-1` kills
every maximal born start except start `0`. If that cell is already Defender,
all starts through it are already dead. If it is old Attacker, any alive born
window through it would contradict virginity. Thus one effective carrier
cell still leaves at most one born label.

For `L` nonparallel to Q, the two surviving singleton windows intersect in at
most one physical cell, so a two-stone future pair cannot mature both. For a
parallel carrier they do not intersect. If `L=Q`, virgin returns lie beyond
one exterior side. On the left, write `v<w<=-6`; service `{w+1,2}` leaves only
born start `w-5` and robust `W_-4`, whose physical windows are disjoint. The
right side is reflected. Therefore an actual service always has
`TEMPO<=1`.

A handoff has `TEMPO=0` exactly when every count-two label is deleted. Two
placements delete all five robust Q labels only in one of the ten occupancies
`C_Q`; such an occupancy also deletes the born family exactly when it meets
every member of `B(v,w)`. This proves both directions of (150), so there is no
missed value-two or value-three virgin pair.

For a non-Q born carrier crossing Q at an empty cell, a Q cover can meet a
born window only at that crossing. The eight ranks in (151) are exactly the
coordinates appearing in some member of `C_Q`, confirming the corollary.

**Proposed repair:** none.

### 5. NOTE — one-virgin nonbridges are exact and the bridge/Q-arm boundary stays OPEN

**Quoted claims:**

> “If no born bridge survives ... local return in `X: M=0`; local return in
> `N: M=1`.” (L16.6)

> “If an alive bridge is born ... the exact `0/1/2` quotient is open.”
> (§89.3)

**Independent recomputation.** With a local Q return in `X` and no surviving
response-pair bridge, the only graded stock is axial. The displayed full row
cleanups `{-1,3}` at positive depth one and `{-1,2}` at depths two through
five, with reflection on the left, delete it completely. Thus `M=0`.

With a local arm return in `N`, the graded tier is the five robust Q labels
plus one promoted arm label, all at count two. One Q cap leaves at most one Q
label and one arm residual cell deletes the arm label, giving `M<=1`. Value
zero is impossible: deleting all five Q labels consumes two row placements,
whereas the arm carrier meets Q only at its occupied original endpoint. Thus
`M=1` exactly. Current `tau=0` in both cases.

The source does not silently apply these values when a response-pair bridge
survives. The arm-plus-virgin bridge case is only bounded by the low-only
theorem, and full Q cleanup leaves pure count-two born stock in the
Q-plus-virgin case, giving only `M<=2`. Their exact values remain explicitly
OPEN. Section 89.4 likewise leaves all `200+190` Q/arm value-one successor
quotients OPEN. Noncanonical exterior services and exact exterior successor
strata also remain outside the positive theorem.

**Proposed repair:** none.

### 6. NOTE — the `P_4^pl`-to-`P_3^pl` transfer closes every two-local response

**Quoted claim:**

> “If both response cells lie in `K_3`, then
> `M(P_3^pl+a^dagger+b)<=3`.” (L16.1.1)

**Independent recomputation.** The two later `V^+` cells `(10,4),(10,5)`
lie outside `K_3`: the `q=10` P3 positive-window union ends at `r=2`, and
their row and level carriers contain no P3 Attacker stone. The intervening
P4 ray pair is outside every window meeting `K_3`. Hence a pair
`b subset K_3` is the same empty, legal, nonterminal response at the capped
P3 and P4 states.

Take the confirmed P4 service `d^+` with handoff `TEMPO<=3`. Retain, in
order, precisely its cells that meet a P3 returned label of count at least
two. A retained cell is independently legal from unchanged P3 Attacker stock.
Every P3 current imminent is also a P4 current imminent, so those retained
cells still service the whole P3 current family. A discarded service cell
cannot have killed a P3 graded label; otherwise it would have met that label
and been retained. Safe fillers outside surviving graded supports complete
the P3 action when necessary.

The future-pair comparison also survives newly born labels. If a future P3
cell is already one of the P4 `V^+` stones, its effect is already present at
P4. If it conflicts with an extra or discarded P4 Defender cell, that cell
lies in no P3 graded label and cannot be load-bearing in an imminent created
by one pair. Any remaining mate which triggers a pre-count-three label is
independently legal and can be copied first. Corresponding P4 residuals are
subsets of the P3 residuals; extra P4 labels and replacement Attacker filler
stones only add P4 constraints. The Section-76 Defender filler in `d^-` was
chosen outside surviving P3 graded supports. Therefore every P3 future demand
is bounded by a legal P4 future demand, hence by three.

This transfer is the load-bearing upper-bound proof. It does not import the
refuted round-9 P3 value-two synthesis.

**Proposed repair:** none.

### 7. NOTE — hard unnamed stock-assisted responses do not seed an `M>=4` fan

**Quoted claim:**

> “Same-axis, distinct-axis, capped bridge, ordinary hard bridge, all five
> stock-assisted double incidences, and the repaired `(5,-4)` boundary are
> included.” (L16.2)

**Independent recomputation.** I separately stressed responses other than
the attaining `b_*`.

- For `{(8,-2),(10,-2)}`, the `r=-2` and `s=6` families have effective
  blockers at `(9,-2)` and `(7,-1)`. The remaining consecutive `q=10`
  family has weight two; its demand-two trigger depths have no simultaneous
  transverse pre-count-two support.
- For `{(11,-4),(14,-4)}`, blockers `(12,-4)` and `(8,-1)` delete the
  `r=-4` and `s=7` effects. Only the `s=10` consecutive-triple family
  remains hard.
- For `{(12,-5),(12,-3)}`, the three born high axes have common-blocker
  reductions; two can be deleted outright and the third contributes at most
  the remaining axial demand.
- The hard/hard bridge `{(10,-2),(11,-2)}` is stabilized by
  `(9,-2),(10,-1)` into two atomic tails with no axial connector.
- A response at `h` together with a remote mate promotes `U^-`, `V^-`, and
  `W`, but `(11,-1)` deletes the W block and one common residual deletes U
  or V. The surviving U/V block has one-trigger weight one. A future pair
  concentrated on it costs at most two; in the split case it contributes at
  most one while the unique lower-axis family costs at most two. Local mates
  are covered by the two-local transfer.

None produces four irreducible one-trigger demands or a one-component demand
four. These checks agree with, but do not replace, the exhaustive transfer in
Finding 6.

**Proposed repair:** none.

### 8. NOTE — the stock-assisted attaining response has exact returned value three

**Quoted claim:**

> “The response in (143) has
> `M(P_3^pl+a^dagger+b_*)=3`.” (L16.3)

**Independent recomputation.** For

```text
b_*=((9,-2),(10,-2)),
d_0={(11,-2),(8,-1)},
```

the binding §86.1 audit gives the lower bound three after every Defender
service. The only occupancy deleting both common-blocker families is `d_0`.
After it, the exact high family is the four `q=10` count-three windows with
starts `-7,-6,-5,-4`.

The future-pair quotient is complete:

- two `q=10` cells have a one-axis rank cover of size at most two;
- zero `q=10` cells can mature only pre-count-two labels on their unique
  common axis, of demand at most two;
- one cell `(10,t)` gives vertical demand two only at `t=-5,-1`. At those
  depths the transverse carriers have no pre-count-two support. At every
  other depth the vertical contribution is at most one and the unique
  lower-axis contribution at most two.

Thus `d_0` attains a handoff upper of three. The legal pair

`e_*=((10,0),(11,-1))`

attains it: the `q=10` residuals `{-5,-1},{-1,1}` have hitting number one,
while the intact `s=10` residuals `{8,9},{9,14},{14,15}` have hitting number
two. Their carrier intersection is the newly occupied hub, so their empty
grounds are disjoint. Therefore the returned value is exactly three, not
merely at least three.

**Proposed repair:** none.

### 9. NOTE — the `P_x` immediate-minimizer criterion is exact

**Quoted claim:**

> “A legal ordered action `d` at `P_x` is an immediate-`TEMPO` minimizer if
> and only if its occupancy meets `C`.” (R10.3)

**Independent recomputation.** With

```text
A={(0,0),(0,1),(1,0),(1,-1),(6,0),(7,0),(-2,2),(0,2)}
```

and local Defender cells `(-4,0),(2,0),(0,-1),(1,1)`, the complete graded
axis census is exactly:

| axis | count-three starts | count-two starts |
|---|---|---|
| `s=0` | `-4,-3,-2` | `-5,-1,0` |
| `q=0` | `0` | `1` |
| `q=1` | none | `-5` |
| `r=0` | none | `3,4,5,6` |
| `r=2` | none | `-5,-4,-3,-2` |
| `s=1` | none | `-4,-3,-2,-1,0` |

All other axes contain at most one Attacker stone. The eight-cell set `C` is
exactly the union of the empty residual grounds of the current count-three
family.

Necessity is sharp. If an action misses `C`, both cells of the legal response
`{(-1,1),(0,3)}` remain empty. On `s=0` it creates the residual path

`{-4,-3}, {-3,2}, {2,3}`,

of demand two, and on `q=0` the disjoint residual `{4,5}`, of demand one.
The carriers meet at occupied `(0,0)`, so the returned demand is three and
the action cannot attain `M(P_x)=2`.

For sufficiency, a contact in `C_Q` deletes the complete nested `q=0`
count-three/count-two pair, leaving only the mixed `s=0` family, whose full
two-trigger axial quotient has demand at most two. A contact at the five
successive `C_S` cells leaves high-start sets

`{-3,-2}, {-2}, empty, {-4}, {-4,-3}`,

each of one-trigger weight at most one. The `q=0` nested component also has
weight one. Every intersection of either high carrier with another graded
carrier is an occupied Attacker cell, so a split pair cannot add a matured
off-axis pre-count-two label; a concentrated pair remains within the axial
bound two. The action's other cell only deletes more stock. Hence every
legal action meeting `C` has handoff `TEMPO<=2`; the inherited universal
floor makes it an exact minimizer.

**Proposed repair:** none.

### 10. NOTE — the `48/96` proactive residual is an exact boundary of the proved obstruction

**Quoted claim:**

> “Every immediate minimizer outside the following `48` unordered
> occupancies has `R_1>=3`.” (L16.8)

**Independent recomputation.** The three focal empty supports have sizes
seven, seven, and seven. The `r=2` and `s=1` supports meet only at
`p=(-1,2)`; the row support is disjoint from both. Thus `|F|=20`, and direct
substitution gives `C intersect F=empty`.

The common response `e_x={p,(8,0)}` remains legal and preserves the round-9
three-family `2+1` lower bound whenever both action cells lie in `C`, the
unique non-`C` cell misses `F`, or that cell is one of the six outer
endpoints. These cases are therefore not residual cases.

For an action using `C_Q`, every remaining `F` contact except `p` and
`(8,0)` leaves two focal families untouched together with the intact old
`s=0` hard family. The `(8,0)` conflict is defeated by the substitute
response `{p,(9,0)}`. Only the three occupancies pairing `p` with a `C_Q`
cell remain.

For an action using `C_S`, the five interior U-row partners

`(4,0),(5,0),(8,0),(9,0),(10,0)`

are defeated by `{(0,3),p}`. The `q=0` current imminent consumes one service
cell; the other can touch at most one of the intact `r=2,s=1` families, so
the handoff still has demand at least `2+1`. Removing those five cells and
the already handled outer endpoints leaves the nine-cell union in (159):
five cells on `r=2`, five on `s=1`, with `p` counted once.

The exact unclassified count is therefore

`3 + |C_S|*9 = 3 + 5*9 = 48`

unordered occupancies. Each `C` cell and each displayed partner is supported
by unchanged Attacker stock, so both orders are independently legal, giving
`96` ordered actions. This is not an artifact of stopping before an
unindexed infinite family: it is exactly the complement of the proved
risk-three cases inside the complete minimizing set. It is also not a
safe/unsafe classification of those 48 actions; the artifact correctly
leaves every one OPEN and draws no conclusion about `B_1(P_x)`.

**Proposed repair:** none. An optional wording clarification could say
“exact residual boundary of this obstruction proof,” to prevent “exact
boundary” from being misread as an exact value classification.

### 11. NOTE — quantifiers and binding errata are carried consistently

**Quoted claim:**

> “P3 is Q1/Q3 reached-state, the seal is a Q2 subtree, and `P_x` is
> Q3-repair. None is promoted to Q2 root forcing, perpetual Q3 renewal, or
> GAP-RAW.” (§93)

**Independent recomputation.** The tags match the proved orders.

- R10.1 is a statewise Q1/Q3 minimum. It uses §86's corrected `k*=3` lower
  bound and does not infer strategy-independent arrival.
- The seal results classify only specified successors on the already reached
  double-corner Q2 branch. The inherited finite quotient remains exactly
  `435` unordered / `870` ordered, with `41/45` root-robust row successors.
- R10.3 and L16.8 are Q3-repair statements at one exact `P_x`; neither is
  arbitrary-member `C_cap` closure.
- Binding §58's flat shield is used only to identify the complete P3
  immediate-minimizer set. Binding §67's universal final stop remains
  attributed to R7.2. Binding §76's safe-filler restriction is invoked when
  spare cells are needed. Binding §86's repaired P2 equality and corrected
  P3 stop are not contradicted.

No OPEN Q/arm, born-bridge, noncanonical exterior, or proactive action class
is silently folded into a positive theorem.

**Proposed repair:** none.

### 12. MINOR — §94 omits the landed reviewed/output identity

**Quoted claim:**

> “Landed artifact hash:
> `LANDED_ARTIFACT_HASH: <TO-BE-FOLDED-POST-REVIEW>`.” (§94)

**Independent recomputation.** The authoring input
`f5349d3eb985cdb9ee719ec75272f3a73772604d` is present and correct. The proof
was landed unmodified at

`8424e9a474f160e4d30ec73cf6b655e1da741554`,

with Git blob `c3557d1e63241c1585ef52efb16158313219d0ff` and SHA-256
`309077856d7df8ff2a82b48c913c0c456b4ce37eae3039c45c06851bb6d4959c`.
The working copy reviewed here matches that blob. The authoring-scoped
no-commit sentence is compatible with this later landing, but the placeholder
does not satisfy the requested post-landing provenance record.

**Proposed repair:** replace the placeholder with the landed commit and blob
above, optionally retaining the SHA-256 as the independent byte identity.

## Per-target verdicts

| Target | Source claim | Review verdict | Disposition |
|---|---|---|---|
| **1 — exact `B_1(P_3^pl)`** | Cap risk and minimum are exactly `3` | **CONFIRMED EXACT** | Capped inventory `(69,34)` is exact; the response architecture covers remote, split, bridge, same-axis, `h/g`, and all five stock-assisted cases; every response has returned `M<=3`; `b_*` has exact returned `M=3`; §86 gives the universal floor |
| **2 — seal successor classes** | Canonical exterior successors have all-response `M<=2`; two-virgin and one-virgin-nonbridge values are exact; named refinements OPEN | **CONFIRMED AT STATED PARTIAL SCOPE** | Exterior `j=0,1,2` quotient is exhaustive; two-virgin values are exactly `0/1`; one-virgin nonbridges are exactly `0` for Q and `1` for arms; noncanonical exterior services, exact exterior strata, Q/arm successors, and surviving one-virgin bridges remain honestly OPEN |
| **3 — proactive renewal** | Complete minimizers; all but `48/96` have risk at least three | **CONFIRMED AT STATED PARTIAL SCOPE** | A legal action minimizes iff it meets `C`; every risk witness is legal and recomputes; `3+5*9=48` unordered and both orders give `96`; every residual action remains OPEN, so no `B_1(P_x)` claim follows |

## Overall verdict

**SOUND-WITH-MINOR-ERRATA.** The default hostile posture did not produce a
mathematical refutation. Target 1 is confirmed: Round 10 states an explicit
exhaustive response architecture, its transfer argument validly bounds the
entire two-local class by the confirmed P4 cap value, and the previously
missed stock-assisted response `b_*` is now handled on both sides with exact
returned value three. Together with the binding universal lower bound, this
proves `B_1(P_3^pl)=3` exactly.

Target 2 is confirmed only at the partial scope it claims. Each universal
`M<=2` class survives its full response quotient, while Q/arm successors,
one-virgin born bridges, noncanonical exterior minimizers, and exact exterior
strata stay OPEN. Target 3's minimizer criterion and `48/96` residual count
are exact; the residual is an explicit open boundary, not a disguised safety
claim.

No REFUTED or MAJOR finding was found. The sole correction is MINOR
provenance: §94 must replace its placeholder with landed artifact
`8424e9a4...`.

## Exact unresolved obstacles

1. **Q2 strategy-independent forcing.** Exact local loss at `P_3^pl` does
   not show that every strict-root Defender strategy reaches that plateau or
   another losing state.
2. **Other P3 action magnitudes.** The minimum is exact, but individual exact
   risks of non-cap `H_3`-touching actions remain unclassified above their
   inherited lower bound three.
3. **Exterior seal refinements.** Noncanonical minimizers of the four
   exterior first-response epochs and the exact `M=0,1,2` strata after the
   canonical landing remain open.
4. **Value-one and born-bridge seal successors.** The `200+190` Q/arm
   value-one successor quotients and the exact one-virgin surviving-bridge
   values are not classified.
5. **Proactive renewal boundary.** The `48` unordered / `96` ordered actions
   are explicit but unresolved. No member is proved safe, no member is proved
   unsafe, and `B_1(P_x)` remains open.
6. **Arbitrary-member capped renewal.** Exact action classification at one
   `P_x` does not prove closure from every `C_cap` member or define a
   perpetual all-history Q3 policy.
7. **Other continuation geometries.** Generalized-lozenge continuation,
   other shared/nonshared fanouts, cross-hull closure, later derivatives,
   alternative forced-seal entrances, and an amortized replacement invariant
   remain open.
8. **General count-three initialization.** L13.6 still excludes residual
   families of hitting number at least three.
9. **Ancillary separation sharpness.** The exact minimum separation for R5.2
   remains unknown; radius 21 is envelope-sharp only.

None of these open items weakens the exact reached-state headline proved in
Target 1.
