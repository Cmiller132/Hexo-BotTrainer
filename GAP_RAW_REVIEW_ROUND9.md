# R-G7-REV — Round-9 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`,
`9e57ea060462788841d1f8f761db894981b482e9`

**Reviewed document:** `GAP_RAW_PROOF_ROUND9.md`

**Artifact's named authoring input:**
`c019400ad14e06fa6f600c5462113a74795e3270`

**Artifact identity:** Git blob
`a54d98fd84eafac724dd563b259c4becdbf57a03`; SHA-256
`1ba2efd0ea48faf84d29e6da168d804650ddfae9468eaa6b7f1d394ec62635fb`.

**Theorem-name crosswalk.** The review request calls the positive `P_2^pl` /
`P_3^pl` cap assertion “R9.1” and the universal `P_4^pl` stop assertion
“R9.2.” The artifact numbers them in the opposite order: artifact **R9.1** is
the `P_4^pl` theorem in §78, and artifact **R9.2** is the `P_2^pl` / `P_3^pl`
theorem in §79. This review uses the artifact's numbering and always states
the plateau as well.

**Method.** First-principles hostile proof audit with the requested default
posture against both directions of the transition. I read, in the stipulated
order and in full, `GAP_RAW_PROOF_ROUND7.md` including binding §67,
`GAP_RAW_REVIEW_ROUND7.md`, `GAP_RAW_PROOF_ROUND8.md` including binding §76,
`GAP_RAW_REVIEW_ROUND8.md`, and then the reviewed round-9 artifact. I then
consulted round 6 including binding §58 for the exact plateau coordinates,
stock sequence, and value-plateau contract. No `STRATEGY_STEALING_*` file was
read as evidence. Every inventory, finite support, response effect, residual,
transversal, legality chain, `tau`, `TEMPO`, and `M` bound below was recomputed
by hand. No Cargo command, Lean build, harness, game/search program, solver,
or machine enumeration was run. Read-only Git identity checks were used only
for provenance. No Git commit was created.

**Overall verdict:** **REFUTED.** The central positive assertion at
`P_3^pl` is false. After the cap, the legal stock-assisted response
`((9,-2),(10,-2))` returns a state with `M>=3`. More strongly, adjoining its
finite support to the two inherited triangle-fan regions proves that **every**
Defender action at `P_3^pl` is unsafe. The exact named-plateau transition is
therefore

`k*=3`,

not four. The `P_2^pl` cap remains exact-risk two after one localized incidence
repair. Artifact R9.1's universal `P_4^pl` stop theorem, the 435-class seal
quotient, and the six-action full-delete renewal obstruction survive. Three
additional localized defects do not change those surviving values: one omitted
boundary incidence in §79, one false connector-uniqueness sentence in §78,
and an overcount of live split-arm bridges in §82. The landed provenance is
also absent from §85.

## Numbered findings

### 1. REFUTED — §79 misses a legal `W`-assisted `P_3^pl` response with returned `M>=3`

**Quoted claims:**

> “There are exactly five nonconsecutive stock-bridge double-incidence
> patterns at `P_3` ... This is the complete nonconsecutive bridge ledger.”
> (§79.4)

> “Sections 79.3--79.4 construct, after every legal response, an actual
> servicing pair whose handoff has `TEMPO<=2`,” and hence
> `R_1(P_3^pl,a^dagger)=2`. (artifact R9.2, equation (119))

**Independent recomputation.** Work after the cap

`a^dagger=((0,-1),(1,1))`

at exact `P_3^pl`. Play the legal response

`b_*=((9,-2),(10,-2))`.                                   (R9-REV-1)

Both cells are empty and independently within one step of unchanged
`A@(10,-3)`; neither order relies on the first new stone. The complete high
family after (R9-REV-1) is:

| axis | Attacker ranks | count-three starts | exact residuals |
|---|---|---|---|
| `r=-2`, parameter `q` | `9,10,12` | `7,8,9` | `{7,8,11}`, `{8,11,13}`, `{11,13,14}` |
| `s=7`, parameter `q` | `7,9,10` | `5,6,7` | `{5,6,8}`, `{6,8,11}`, `{8,11,12}` |
| `q=10`, parameter `r` | `-4,-3,-2` | `-7,-6,-5,-4` | `{-7,-6,-5}`, `{-6,-5,-1}`, `{-5,-1,0}`, `{-1,0,1}` |

No old count-two window receives both response cells. The old `r=-2` carrier
had count one, while the other two axes receive one response cell each.
Therefore every displayed label has count exactly three, there is no
count-four label, and current `tau=0`.

The three carrier intersections are precisely the Attacker stones

```text
r=-2 ∩ s=7  = (9,-2),
r=-2 ∩ q=10 = (10,-2),
s=7  ∩ q=10 = (10,-3).
```

An empty Defender cell consequently touches high labels on at most one of
the three axes. The first two families have unique empty common blockers
`(11,-2)` and `(8,-1)`, respectively. The four-window consecutive triple on
`q=10` has no empty common blocker. If a next Defender pair touches zero or
one family, two untouched families each supply a one-trigger demand-two path.
If it touches two families and either touched family survives, the untouched
family supplies demand two and a surviving touched count-three label supplies
demand one. Residual grounds on different axes are disjoint because their
only carrier intersections are occupied.

Thus the sole remaining effect occupancy is

`d_0={(11,-2),(8,-1)}`,                                   (R9-REV-2)

with its two legal orders: it kills the `r=-2` and `s=7` families and leaves
the `q=10` family. After either order, Attacker plays the independently legal,
nonterminal pair

`e_*=((10,0),(11,-1))`.                                   (R9-REV-3)

On `q=10`, the four Attacker ranks `-4,-3,-2,0` give the count-four residuals

`{-5,-1}`, `{-1,1}`,

of hitting number one. On the intact `W` pencil `s=10`, ranks `10,11,12,13`
give

`{8,9}`, `{9,14}`, `{14,15}`,

of hitting number two. The axes meet at the now-Attacker-occupied hub, so the
physical residual grounds are disjoint and the combined demand is exactly
three. Therefore every Defender pair after (R9-REV-1) hands over
`TEMPO>=3`, including (R9-REV-2), and

`M(P_3^pl+a^dagger+b_*)>=3`.

In particular,

`R_1(P_3^pl,a^dagger)>=3`,

directly contradicting equation (119). This is not a remote or virgin edge
case: it is exactly the `r=-2` pair of the `s=7` and `q=10` intersections,
one of §79.4's five purportedly completed nonconsecutive double-incidences.
The proposed carrier-gap/vertical-stabilizer service is also unsafe: it leaves
the `s=7` gap trigger with demand two and the vertical tail with demand one.

The miss strengthens to all initial `P_3^pl` actions. Let `H_3` be the union
of the four finite segments

```text
R:  r=-2, q=7..14;
S:  s=7,  q=5..12;
V:  q=10, r=-7..1;
W*: s=10, q=8..15.
```

All older Defender cells miss `H_3`. If an initial pair misses `H_3`, every
window, response cell, unique gap, and future residual used above is protected,
so (R9-REV-1)--(R9-REV-3) apply verbatim. If it touches `H_3`, that cell lies
in neither finite triangle-fan union: direct coordinate substitution places
every possible fan intersection beyond the displayed segment and fan ranges.
The other Defender cell can touch at most one of `G_-`,`G_+`, because their
only common cells are occupied diamond stones. Hence one complete fan is
untouched and L13.3 returns `M>=3`.

This proves the stronger corrected statement

`for every legal a at P_3^pl, R_1(P_3^pl,a)>=3`.           (R9-REV-4)

The inherited safe actions at indices zero and one and Finding 2's repaired
safe cap at index two combine with (R9-REV-4) to give the exact correction

`k*=3`.                                                    (R9-REV-5)

This remains a Q1/Q3 reached-state result. It gives no Q2 route forcing
arrival at `P_3^pl` from every strict-root strategy.

**Proposed repair:** split artifact R9.2. Retain only the exact `P_2^pl`
equality. Replace its `P_3^pl` equality, the claim `B_1(P_3^pl)=2`, and
Corollary R9.2.1 by the `H_3/G_-/G_+` universal stop theorem above. Change the
headline, mechanism paragraph, attack surface, and authoritative ledger from
`k*=4` to `k*=3`.

### 2. NOTE — the capped stock inventories are exact, and the `P_2^pl` half survives

**Quoted claim:**

> “After (107), the exact alive inventories” are `(8,12)`, `(33,16)`,
> `(49,26)`, `(69,34)`, `(95,39)`, and `(115,47)`, with no count at least
> three. (L15.2)

**Independent recomputation.** Starting from the cap's exact `(n_1,n_2)`
value `(8,12)`, the five stock increments are

| stock addition | `Delta n_1` | `Delta n_2` | hand basis |
|---|---:|---:|---|
| `U^-` | `25` | `4` | four surviving row starts, one row extreme, four singleton transverse pencils |
| `V^-` | `16` | `10` | adjacent `q=10`; two singleton rows; distance-four `s=6`; distance-three `s=7` |
| `W` | `20` | `8` | adjacent `s=10`; singleton `q=12,13`, singleton `r=-2`, and distance-three `r=-3` conversion |
| `V^+` | `26` | `5` | separated adjacent `q=10` block plus four singleton carriers |
| `U^+` | `20` | `8` | separated adjacent row block, two singleton columns, and `s=14,15` conversions |

This gives L15.2's six rows exactly. At `P_2^pl`, the count-two total is

```text
q=0:1, q=1:1, s=0:5, s=1:5,
r=0:4, q=10:5, s=6:2, s=7:3,
```

which sums to `26`; the corresponding count-one census sums to `49`. Adding
`W` contributes `(20,8)`, producing exact `(69,34)` at `P_3^pl`. The false
`P_3` value is therefore not caused by an inventory error: it is a missed
incidence of those correctly counted labels.

After the boundary repair in Finding 3, the `P_2^pl` response quotient has no
`W` block and no analogue of (R9-REV-3). Its only empty count-two/count-two
connector is `h`; same-axis responses use the complete axial cover; a hard
count-one carrier can coexist with at most one easy central family; and the
hub effects reduce to atomic `U^-` and `V^-` components. Remote/remote,
local/remote, split, bridge, separated same-axis, and stock-assisted cases then
all admit an actual handoff of `TEMPO<=2`. The inherited response
`b^dagger=((-1,1),(-1,2))` still has returned value exactly two. Thus

`R_1(P_2^pl,a^dagger)=B_1(P_2^pl)=2`

is **CONFIRMED-WITH-MINOR-REPAIR**.

**Proposed repair:** retain L15.2 and the `P_2^pl` half of artifact R9.2;
separate them from the refuted `P_3^pl` half.

### 3. MINOR — §79 omits the boundary incidence `(5,-4)`

**Quoted claims:**

> The `r=-4` count-one carrier has only the displayed intersections with
> `s=7` and, at `P_3`, `s=10`. (§79.4 incidence table)

> “Each displayed envelope misses every finite live transverse count-two
> union.” (§79.4)

**Independent recomputation.** The singleton carrier `r=-4` through
`A@(10,-4)` has physical six-window union `q=5..15`. The live capped shield
pencil `s=1` has union `q=-4..5`. Their empty boundary intersection is

`z=(5,-4)`.

It is absent from the incidence table. The displayed retained `r=-4` envelope
`[5,9]` contains `z`, so the blanket tail-isolation sentence is also literally
false. This does not create an additional nonconsecutive double-incidence
occupancy with the listed `q=11,14` intersections: their separations from
`q=5` are six and nine, too large for one length-six carrier window.

The affected hard response `{(8,-4),(9,-4)}` is safely repaired by the full
axial deletion

`D@(7,-4), D@(11,-4)`

in either legal order. The first displayed order deletes starts `5,6,7` and
leaves start `8` alive to support the second; afterward every high label is
gone and L10.4 gives `TEMPO<=2`. If a response actually uses `z`, its
`r=-4` family is nonconsecutive with a one-cell internal-gap deletion, while
`z` promotes only the singleton `s=1` start `0`; the second effective cell
deletes that label. No theorem value changes after this finite repair.

**Proposed repair:** add `s=1 at (5,-4)` to the `r=-4` carrier row; qualify
the envelope-isolation sentence; use the full axial deletion for the sole
retained tail reaching `z`.

### 4. NOTE — artifact R9.1's universal `P_4^pl` stop theorem is confirmed

**Quoted claim:**

> “Every legal Defender pair `a` at `P_4^pl` has a legal nonterminal response
> returning `M>=3`.” (artifact R9.1)

**Independent recomputation.** Before the cap, exact `P_4^pl` has
`(n_1,n_2)=(107,47)` and no higher count. The cap deletes twelve count-one
and eight count-two diamond labels, yielding L15.2's correct `(95,39)`.

The finite region `H_4` is literally disjoint from both fan unions. If an
initial action touches `H_4`, that cell touches neither fan and the other cell
touches at most one; L13.3 applies to the untouched fan. If the action misses
`H_4`, the response

`b_4=((10,0),(11,-1))`

creates exactly the three count-four `s=10` starts `8,9,10`, with residuals

`{8,9}`, `{9,14}`, `{14,15}`.

Their only unordered services are `{8,14}`, `{9,14}`, and `{9,15}`, with both
orders. Every service also deletes the two outer `W` count-three windows.
After every service,

`e_4=((10,1),(8,0))`

is empty, independently legal, and nonterminal. It creates the disjoint
vertical grounds `{-2,-1}` and `{2,3}`, plus the mandatory row ground
`{9,11}` and optional `{5,9}`. The row subfamily costs one hit and the two
vertical subfamilies one each, so exact `tau=3`. This proves the focal
`M=3` branch, while the fan branch supplies the required lower bound for all
remaining actions.

The source's *particular* `b_4/e_4` mechanism does lose one demand when
`V^+` is removed: at `P_3^pl` it gives only `V^-+U`, of demand two. That fact
does not make index four the transition, because Finding 1 supplies the
different `r=-2/s=7/q=10/W` obstruction already at index three.

**Proposed repair:** none to artifact R9.1 or L15.1. Revise only the claim
that this is the earliest stock-phase obstruction.

### 5. MINOR — §78's connector-uniqueness sentence forgets inherited `g`

**Quoted claim:**

> “The only cross-axis finite connector remains `h`.” (L15.1.1 proof)

**Independent recomputation.** At `P_4^pl`, the empty point

`g=(9,-3) in (s=6) intersect (r=-3)`

still lies in both finite live count-two unions. Installing `V^+` neither
occupies `g` nor deletes either incident family. Equation (118), introduced in
the following section for `P_3^pl`, therefore remains an inherited connector
at `P_4^pl`. The literal uniqueness assertion is false.

This does not invalidate the cap upper bound. The preceding §78 paragraph
imports the §79 axial/connector service, whose `g` case has only easy
distance-three/four effects. Direct checks of a `g`/vertical split and of the
double-connector response `(h,g)` still leave total future weight at most
three. The genuinely new connector involving the enlarged separated
`q=10` class is `h`; `g` is simply not new.

Finding 1's newly exposed response also does not exceed the `P_4^pl` upper
target. At `P_4^pl`, after `b_*`, service `D@(11,-2)` kills the `r=-2`
family and `D@(10,-1)` stabilizes the consecutive negative-vertical family
to one atomic tail. The untouched `s=7` family has future weight two and the
vertical tail weight one; the added `V^+` block is six-window-separated from
that tail. A future pair therefore has total demand at most three. Thus the
same class which refutes the `P_3` upper of two is consistent with, and
attains no contradiction to, the `P_4` upper of three.

**Proposed repair:** replace the quoted sentence by: “The only connector
involving the enlarged `q=10` vertical class is `h`; inherited `g` remains and
is handled by the §79 connector service,” and explicitly reclassify `b_*` at
the relaxed upper target three. With those localized repairs,
L15.1.1's `R_1(P_4^pl,a^dagger)=B_1(P_4^pl)=3` remains confirmed.

### 6. NOTE — the 435-class finite seal quotient and its exact values are confirmed

**Quoted claim:**

> “The entire finite residual-return quotient on `X union N` is now exact ...
> This is `435` unordered and `870` ordered first responses.” (§82.3)

**Independent recomputation.** The local residual set has `|X|=10` row cells
and `|N|=20` arm cells. Distinct unordered response occupancies therefore
number

`C(30,2)=C(10,2)+10*20+C(20,2)=45+200+190=435`,

and both orders give `870`. The three exact axial-demand-two row occupancies
have `M=0` after their displayed complete graded covers. Of the remaining
forty-two row occupancies, the count is

```text
7 nonexceptional same-left
+ 7 nonexceptional same-right
+ 24 nonexceptional cross-side
= 38 root-robust M=0 classes,
```

leaving four exterior cases. For example, response `{-5,-4}` with service
`{-3,2}` leaves only `W_-9`, while `{-4,-3}` with `{-2,2}` leaves only
`W_-8`; reflection gives the two right cases. When that exterior window is
alive, no two empty row cells cover it together with the robust central block,
so exact `M=1`; when root contact already killed it, the same service gives
exact `M=0`. Thus the stated dichotomy is exact, not merely an upper bound.

For one Q return plus one arm return, value zero would require two row cells
for the robust row tier and a third off-row cell for the promoted arm, whose
carrier meets the row only at an occupied endpoint. The displayed row/bridge
caps attain `TEMPO=1`, giving exact `M=1` for all `200` occupancies. The same
three-hit obstruction and the one-row-cap/one-arm-or-bridge cleanup give exact
`M=1` for all `190` two-arm occupancies. Finding 7 corrects one live-bridge
subcount but does not alter this value argument.

The root-robust safe-successor count is exactly `3+38=41`. The source does
not extend the result to the four surviving exterior value-one successors,
the Q/arm value-one successors, or a response containing one or two virgin
returns. Those classes are explicitly **OPEN**, including the possible virgin
bridge crossing the old Q row at an empty cell. No universal seal-cycle or Q2
root-forcing conclusion is smuggled into the finite classification.

**Proposed repair:** none to the `435/870` count, exact value classes,
`41/45` successor count, or OPEN boundary; apply only Finding 7's bridge
subcount correction.

### 7. MINOR — L15.9 counts eight live split-arm bridges, but only six survive the seals

**Quoted claims:**

> “The response pair creates a bridge exactly in these eight occupancies ...
> `k=1,2,3,4`.” (L15.9)

> The attack surface lists “exactly eight split-arm bridges.” (§84 item 12)

**Independent recomputation.** For `k=1`, one listed pair is

`{(0,-1),(2,-1)}`.

Its carrier is `r=-1`, but the pre-existing seal `D@(1,-1)` lies between the
two response cells. Every length-six row window containing both responses
also contains that Defender stone and is already dead. Under `rho`, the pair
`{(-1,1),(1,1)}` is likewise killed by `D@(0,1)`. Thus the two `k=1`
occupancies are geometrically collinear but do not create alive bridges.

For each of `k=2,3,4`, the two reflected orientations do create an alive
bridge; `k=5` has span six and no common length-six window. The exact live
count is therefore

`2 orientations * 3 depths = 6`,

not eight. This is a monotone overcount in the upper-bound construction. In
the dead `k=1` cases no bridge cap is needed; the proof's own no-effective-cap
and safe-filler branch applies. The independent row/arm lower bound still
forbids value zero, so every affected landing retains exact `M=1`.

**Proposed repair:** say “eight geometrically collinear templates, of which
the two `k=1` carriers are pre-killed by the seals; six alive bridges at
`k=2,3,4`,” and change the §84 live-bridge subcount from eight to six.

### 8. NOTE — the six full-delete renewal minimizers and their common failure are confirmed

**Quoted claims:**

> “The complete unordered actions which delete every high label are
> `d_j={(-1,1),(0,j)}`, `j in {3,4,5}`, with both orders,” and every one has
> `R_1(P_x,d_j)>=3`. (L15.7, R9.4)

**Independent recomputation.** At `P_x`, response
`b_x=((-2,2),(0,2))` creates exactly three `s=0` count-three residuals

`{-4,-3,-1}`, `{-3,-1,2}`, `{-1,2,3}`,

whose unique common empty hit is physical `(-1,1)`, and the one `q=0`
residual `{(0,3),(0,4),(0,5)}`. No count-four label exists and `tau=0`.
The two grounds are carrier-disjoint, so every full deletion must choose
`(-1,1)` and exactly one of the three vertical residual cells. This gives
three unordered and six ordered actions, all sequentially legal. Each leaves
pure count-two stock and exact `TEMPO=2`; the intact `s=1` pencil supplies the
lower bound. The three action-disjoint `s=0`, `s=1`, and `r=0` demand-two
regions give the matching state value `M(P_x)=2`, so all six actions really
are immediate minimizers.

After any one of them, the common response

`e_x=((-1,2),(8,0))`

creates four count-three windows on each of `r=2`, `s=1`, and `r=0`, exactly
twelve in total and no count-four label. Every `d_j` misses all three finite
unions; the axes meet only at Attacker stones. A next Defender pair therefore
leaves either two untouched demand-two families or one untouched demand-two
family plus a once-touched surviving demand-one family. Hence returned
`M>=3` and `R_1(P_x,d_j)>=3` for all six orders.

The completeness statement is correctly limited to **full-delete**
minimizers. A proactive immediate minimizer might truncate a future flank
without deleting every present high label. Such actions are unclassified, so
the source honestly does not infer `B_1(P_x)>=3` or arbitrary-member
`C_cap` impossibility.

**Proposed repair:** none.

### 9. NOTE — quantifier scopes and binding errata remain honest, but the plateau ledger must change at `P_3^pl`

**Quoted claims:**

> The round-9 ledger records exact safe cap values at `P_2^pl,P_3^pl`, a
> universal unsafe `P_4^pl`, and inherited unsafe `P_5^pl=P_stock`. (§83)

> “Q2 root forcing, a perpetual Q3 policy ... and GAP-RAW are not inferred.”
> (§77.2)

**Independent recomputation.** Binding §58's flat immediate-value-two shield
contract is respected: none of the one-successor findings changes the fact
that every initial action at these low-only plateau epochs has immediate
handoff `TEMPO=2`. Binding §67 still attributes the final `P_stock` universal
stop to R7.2, and the pre-cap `(127,55)` final census is consistent with the
post-cap `(115,47)` census after the cap's `(12,8)` deletions. Binding §76's
safe-filler clause is invoked whenever exact derivative membership matters;
the renewal actions `d_j` use no filler.

Section 58's separate exact post-hub ray-history strengthening
`M=4`, next `tau=4` is also untouched. The new `>=3` local-stop witnesses
occur at different response histories and do not replace or weaken that exact
ray value.

The mathematical correction from Findings 1--3 changes the authoritative
later-plateau line to:

| plateau | corrected one-successor status |
|---|---|
| `P_0^pl,P_1^pl` | cap exact-risk two, inherited |
| `P_2^pl` | cap exact-risk two, confirmed after finite incidence repair |
| `P_3^pl` | **every action unsafe**, returned lower bound at least three |
| `P_4^pl` | every action unsafe; cap exact risk three |
| `P_5^pl=P_stock` | every action unsafe, inherited |

Thus §76's former OPEN `P_2^pl`--`P_4^pl` row is legitimately updated, but
the round-9 update chose the wrong status at `P_3^pl`. The corrected transition
is `k*=3`.

The Q tags themselves remain disciplined. The new `P_3^pl` stop is Q1/Q3 and
statewise; it does not force that plateau from a strict root. The seal results
remain Q2 subtree statements at explicitly reached classes. R9.4 remains a
Q3-repair obstruction, not a universal no-policy theorem. Q2 root forcing,
perpetual positive Q3 repair, and GAP-RAW remain OPEN.

**Proposed repair:** replace the two `P_3^pl` safe rows and `k*=4` row in §83,
and the corresponding §84 attack-surface items, by (R9-REV-4)--(R9-REV-5).
No Q2 or global-policy promotion should accompany the correction.

### 10. MINOR — §85 leaves the landed artifact identity as a placeholder

**Quoted claim:**

> “Landed artifact hash: `LANDED_ARTIFACT_HASH:
> <TO-BE-FOLDED-POST-REVIEW>`.” (§85)

**Independent recomputation.** The named authoring input
`c019400ad14e06fa6f600c5462113a74795e3270` is present and correct. At review
time the artifact is landed unmodified at

`9e57ea060462788841d1f8f761db894981b482e9`,

with Git blob `a54d98fd84eafac724dd563b259c4becdbf57a03` and SHA-256
`1ba2efd0ea48faf84d29e6da168d804650ddfae9468eaa6b7f1d394ec62635fb`.
The authoring-scoped no-commit sentence is compatible with this later landed
identity, but the placeholder is not an adequate provenance record after
landing.

During finalization, an unrelated concurrent job advanced the shared branch
from `9e57ea06` to `e19e97f7`. A read-only name comparison showed changes only
in excluded strategy-stealing artifacts and prompt records. The reviewed
GAP-RAW proof retained the blob and SHA-256 above; no concurrent file was read
as proof evidence, and no reset or checkout was performed.

**Proposed repair:** fold the landed commit, blob, and optionally SHA-256
above into §85 while retaining the authoring-scoped no-commit statement.

## Per-theorem verdicts

| Claim | Source status | Review verdict | Disposition |
|---|---|---|---|
| L15.1 exact focal `P_4^pl` witness | Q1/Q3, PROVEN | **CONFIRMED** | `b_4` has exactly three current W services; every service permits exact future `tau=3` |
| Artifact R9.1 universal local stop at `P_4^pl` | Q1/Q3, PROVEN | **CONFIRMED** | `H_4`-miss focal branch plus `H_4`-touch untouched-fan branch exhaust every action |
| L15.1.1 cap value at `P_4^pl` | Q3, PROVEN exact | **CONFIRMED-WITH-MINOR-WORDING** | Exact `R_1=B_1=3`; inherited connector `g` contradicts one uniqueness sentence but is covered by the imported service |
| L15.2 capped prefix inventories | Q3, PROVEN exact | **CONFIRMED** | All six `(n_1,n_2)` rows and no-higher-count assertions recompute |
| §79 finite response quotient at `P_2^pl` | Q3, PROVEN | **CONFIRMED-WITH-MINOR-REPAIR** | Add `(5,-4)` incidence and use the response-specific full deletion; all effect classes then have returned `M<=2` |
| Artifact R9.2 at `P_2^pl` | Q3, PROVEN exact | **CONFIRMED-WITH-MINOR-REPAIR** | `R_1(P_2^pl,a^dagger)=B_1(P_2^pl)=2` after Finding 3's finite incidence repair |
| §79 “complete” response quotient at `P_3^pl` | Q3, PROVEN | **REFUTED** | Legal response `((9,-2),(10,-2))` has returned `M>=3` |
| Artifact R9.2 at `P_3^pl` | Q3, PROVEN exact two | **REFUTED** | Cap risk is at least three, not two; the claimed `B_1=2` witness is false |
| Universal `P_3^pl` action status | not claimed | **PROVEN BY REVIEW: UNSAFE** | `H_3`-miss uses the new stock response; `H_3`-touch leaves one inherited fan untouched |
| R9.2.1 / headline `k*=4` | Q3, PROVEN | **REFUTED** | Safe indices are `0,1,2`; every action is unsafe already at index `3`, so exact `k*=3` |
| L15.3 and R9.3 axial seal cycle | Q2, PROVEN at stated class | **CONFIRMED** | Exact `tau=2,M=0` burst services; value-zero handoff makes the next high family one-axis and returns `M<=2` |
| L15.4 central second lozenge | Q2, PROVEN exact | **CONFIRMED** | Exact `tau=0,M=1`; 32 ordered minimizing caps |
| L15.5 exterior second lozenge | Q2, PROVEN exact at history | **CONFIRMED** | Root pruning leaves enough demand-two pencils; every next action has exact handoff two |
| L15.6 delayed exterior cap | Q2/Q3, PROVEN upper | **CONFIRMED AT UPPER-BOUND SCOPE** | Clean witness has exact two; arbitrary-root pruning is assigned only `M<=2` |
| L15.8 forty-two lower-demand row returns | Q2, PROVEN exact | **CONFIRMED** | Thirty-eight robust `M=0`; four exact root-dependent `M=0/1` cases |
| R9.5 forty-one root-robust row successors | Q2, PROVEN at class | **CONFIRMED** | Count is `3+38=41`; four surviving value-one successors remain open |
| L15.9 Q/arm and two-arm values | Q2, PROVEN exact | **CONFIRMED-WITH-MINOR-SUBCOUNT** | All `200+190` occupancies have exact `tau=0,M=1`; six, not eight, split-arm bridges are alive |
| Entire finite `X union N` quotient | Q2, exact / branch partial | **CONFIRMED** | `435` unordered and `870` ordered responses; exceptional and virgin successors honestly OPEN |
| L15.7 exact renewal successor | Q3-repair, PROVEN | **CONFIRMED** | Exactly six ordered full-delete minimizers, each with immediate value two |
| R9.4 full-delete renewal obstruction | Q3-repair, PROVEN lower | **CONFIRMED** | Common response gives exact twelve-label fan and returned `M>=3` for all six; proactive minimizers remain open |
| Q2 root forcing / perpetual Q3 policy / GAP-RAW | OPEN | **CONFIRMED OPEN** | All stop results remain reached-state; neither all-strategy forcing nor all-history defense is proved |
| Provenance | record | **MINOR ERRATUM** | Input is correct; landed/output `9e57ea06...` remains a placeholder in §85 |

## Overall verdict

**REFUTED.** The round's headline and its positive `P_3^pl` maximum claim do
not survive hostile enumeration. The response `((9,-2),(10,-2))` simultaneously
uses the `r=-2` W carrier, the `s=7` stock pair, and the `q=10` V pair. Its
three exact high families force demand at least three after every Defender
reply; the sole pair which kills both gapped families is defeated by the legal
hub/W response `((10,0),(11,-1))`. This directly refutes artifact R9.2 at
`P_3^pl`.

The same certificate, protected by the finite region `H_3`, combines with the
two inherited triangle fans to cover every initial `P_3^pl` action. Since the
cap remains exact-risk two at `P_2^pl`, the corrected exact transition is
`k*=3`. This correction is local/Q3 and does not settle Q2 reachability.

Artifact R9.1 at `P_4^pl` is independently **CONFIRMED**. Its focal response,
six services, future exact demand three, and fan-region complement are all
sound. The finite seal quotient is also **CONFIRMED**: its `435=45+200+190`
unordered count and exact values hold, with the four exterior, value-one
off-row, and virgin successor classes honestly OPEN. R9.4's six full-delete
minimizers and their common `R_1>=3` response are **CONFIRMED**, while proactive
minimizers remain unclassified.

The omitted `(5,-4)` incidence, forgotten `g` connector, six-versus-eight live
split-arm bridge count, and landed-hash placeholder are localized
**MINOR** defects. None of them rescues the refuted `P_3^pl` equality or changes
the surviving P2/P4/seal/renewal values after the repairs stated above.

## Exact unresolved obstacles after correction

1. **Q2 strategy-independent forcing.** `P_3^pl` is now a proved local stop,
   but no strict root is shown to force arrival there, at `P_4^pl`, or at
   another losing state against every Defender strategy.
2. **Exact risk magnitude at `P_3^pl`.** Every action has one-successor risk at
   least three, which is enough for `k*=3`; exact `B_1(P_3^pl)` and exact risk
   of the cap above that lower bound are not classified.
3. **Earlier all-history Q3 intervention.** The cap is safe through `P_2^pl`
   and too late at `P_3^pl`, but no single initialized policy is proved to
   choose safe actions and renew after every response on every reached history.
4. **Four exterior sealed successors.** When their exterior singleton survives,
   only the canonical landing and its next-`tau` test are known; the complete
   all-response returned-`M` quotient remains open.
5. **Q/arm value-one and virgin seal successors.** L15.9 classifies the first
   landing only. One-/two-virgin returns, including an empty Q-row crossing by
   a born bridge, remain outside the finite `X union N` successor theorem.
6. **Other Q2 continuations.** Generalized-lozenge plateaux and continuation
   after the minimizing raw branch are not iterated to either a forced stop or
   a perpetual defense.
7. **Proactive capped renewal.** R9.4 excludes all six full-delete minimizers
   at one exact successor, but it does not classify non-full-delete immediate
   minimizers, prove `B_1(P_x)>=3`, or decide arbitrary-member `C_cap` closure.
8. **Remaining structural classes.** General count-three initialization with
   residual hitting number at least three, other shared/nonshared fanouts,
   cross-hull interactions, later derivatives, alternative forced-seal
   entrances, and any amortized replacement invariant remain open.
9. **Ancillary separation sharpness.** The minimum separation for the R5.2
   theorem remains unknown; radius 21 is still only envelope-sharp.

None of these open questions weakens the corrected local transition
`k*=3`; they are the barriers to lifting it to Q2 or to a perpetual positive
Q3/GAP-RAW theorem.
