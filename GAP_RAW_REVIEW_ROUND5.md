# R-G3-REV — Round-5 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`, `d93d5768`

**Document:** `GAP_RAW_PROOF_ROUND5.md`

**Method:** first-principles hostile proof audit.  The prerequisite corpus was
read in the required order and in full:
`GAP_RAW_PROOF_ROUND2.md`, `GAP_RAW_PROOF_ROUND3.md`,
`GAP_RAW_REVIEW_ROUND3.md`, `GAP_RAW_PROOF_ROUND4.md` with its folded
post-review errata, `GAP_RAW_REVIEW_ROUND4.md`, and then
`GAP_RAW_PROOF_ROUND5.md`.  Every new coordinate family, residual, hitting
set, potential, radius-eight placement, and ordered 2:2 transition was
recomputed by hand.  No Cargo command, Lean build, harness, search program, or
machine enumeration was run.  No `STRATEGY_STEALING_*` file was read as
evidence or modified.

**Overall verdict:** **SOUND-WITH-ERRATA.**  The sealed-pencil classification,
the two one-cycle value bounds, the strict `S_T` entrance rejection, the
shared-hub cascade, and the unrestricted equation-(22) counterexample all
survive.  No round-5 theorem is refuted.  The most severe defect is a
**MAJOR** overstatement in §46: `GAP-HUB-FANOUT-REACHABILITY` is a useful
obstruction-specific subproblem, but it is not a quantifier-complete or sole
gate for the still-open initialization and repair obligations.

## Numbered findings

### 1. NOTE — inherited definitions and theorem-dead routes do not drift

**Quoted claim:**

> “This document continues the definitions, equation numbering, and status
> discipline of `GAP_RAW_PROOF_ROUND4.md`, including its post-review errata.”

**Confirmation.**  Round 5 retains the exact inherited meanings and phases:

- `I(P)` is the family of Defender-free, Attacker-alive count-four/count-five
  windows at a finite nonterminal position;
- `tau(P)` is the variable-residual hitting number of `I(P)`;
- `TEMPO(Q)` is equation (20)'s exact maximum next-pair demand at an
  Attacker-`FirstStone` handoff with `I(Q)=empty`;
- `M(P)` is equation (21)'s minimum `TEMPO` over the same actual ordered pairs
  that service `I(P)`; and
- equation (22) is the two-coordinate state condition
  `tau(P)<=2 and M(P)<=2`, not a statement that either coordinate is already
  strategy-reachable or inductive.

The document does not revive canonical J, pointwise `Theta_2<1` renewal,
`B_3<1`, the overbroad version of uniform dormant-component charges, the
withdrawn separate-witness conjunction, a global pairing, or NQ2-style
locality pruning.  “Dormant stock” is used only as physical-label terminology.  The
round-4 R4.7.1 boundary is respected in its narrowed, post-errata form: no
claim is made against selective-type, root-dependent, or vanishing charges.

**Proposed repair:** none.

### 2. MINOR — R5.4's “complete gadget support” separation should be formalized

**Quoted claim:**

> “Choose two translations `b_1,b_2` far enough that every cell of one
> complete gadget support is at hex distance at least six from every cell of
> another gadget or of the hub.”

**Audit.**  The intended construction is sound because every region involved
is finite and the translations can be chosen arbitrarily far away.  However,
“complete gadget support” is not defined here as `O_i` was in §39.2.  The
later demand-equivalence assertion also uses the service cells `z_i`, which
are initially empty cells of `Y_i`, and needs them not to lie in or kill a hub
window.  The occupied cells at `z_i-u` and `z_i+u` help enforce this under the
intended support reading, but that auxiliary inference is not stated.

This is not a counterexample to R5.4: choosing the translations with the
whole finite protected regions separated makes every subsequent claim
literal and leaves the calculation in Finding 11 unchanged.

**Proposed repair:** define, for example,
`G_i=A_i union D_i union Y_i` and
`H=A_H union D_H union X`, and require pairwise set distance at least six
between `G_1,G_2,H`.  Then `z_i` and every protected window cell are expressly
included.  R5.4 is **CONFIRMED-WITH-ERRATA** on this wording point.

### 3. NOTE — the strict hub is defusable, and the initialization boundary stays honest

**Quoted claims:**

> “The position `P_H` is a normative strict root in the `tau=0` slice and
> satisfies `M(P_H)=0`.”

> “The general `tau=0` strict-root slice is OPEN.”

**Confirmation.**  All nine hub labels contain the empty cell `0`, so
`D@0` is legal and kills them simultaneously.  A sequential legal filler
then leaves `L_23=empty`, giving `TEMPO=0` and hence `M(P_H)=0`.  The exact
potential from Finding 9 is strict.  Conversely, attaching one independent
count-four service label to the intact hub would add `1/3`, and promoting one
of its count-two labels to count four would add `2/9`; the displayed
inequalities (49)–(50) correctly put both ready-made embeddings over one.

The sealed Defender-root instance also has the claimed

`Phi=5/9+2/(3sqrt(3))<1`.

Its graded tier is pure count two, so Defender monotonicity and L10.4 give
`M<=2`.  These are two instances, not a universal `tau=0` initialization
theorem.  The first exact sealed return is exhausted by L11.2–L11.4 and cannot
source the three-axis fanout, but iterative derivatives and other strict roots
remain unclassified.  The OPEN labels in §§42–44 are therefore correct.

**Proposed repair:** none.

### 4. MAJOR — `GAP-HUB-FANOUT-REACHABILITY` does not gate every remaining obligation

**Quoted claims:**

> “Starting from a strict `Phi<1`, `tau=0` root under one named Defender
> strategy, prove one of the following ...”

> “It contains both remaining exact questions.”

**Counter-derivation.**  The proposed label conflates three different
quantifier problems.

1. Reaching the hub against one fixed policy such as `S_T` proves only that
   this policy fails.  A GAP-RAW counterexample route would require the much
   stronger form `exists P_0, for every Defender strategy S, there exists an`
   `S-consistent Attacker continuation reaching a forced-loss state`, i.e.
   `∃P_0 ∀S ∃α` with the stated reachability property.

2. Showing that one named strategy avoids this exact hub from one strict root
   does not prove the normative positive order
   `∀P_0 ∃S ∀α`.  It does not even settle
   all strict-root `tau=0` initialization instances.
3. Avoiding this hub does not classify other `M>2` fanouts, cross-hull
   interactions, the next turn from nested derivatives, or all responses from
   the axial-cleanup handoff reached in R5.3.1.  Those independent OPEN classes
   are acknowledged in §§37.2, 39.2, 40.2, 43, and the status ledger itself.

Thus hub reachability is a sharp obstruction-specific research target, and it
may decide `S_T`, but it is neither necessary nor sufficient for all of
`GAP-TEMPO-INITIALIZATION`, `GAP-TEMPO-REPAIR`, or GAP-RAW.  The mathematical
hub theorems remain intact; the “exact/sole resume point” characterization is
**DOWNGRADED**.

**Proposed repair:** retain `GAP-HUB-FANOUT-REACHABILITY` as a named subproblem
of the broader `GAP-CASCADE-REACHABILITY`.  State separately:

- fixed-`S_T` hub reachability, which can refute that policy;
- strategy-independent hub forcing, with the quantifiers needed for a
  GAP-RAW counterroute; and
- the positive universal initialization/repair obligation for one strategy,
  which must also exclude every other escape class.

### 5. MINOR — the authoritative ledger omits the new named gap

**Quoted claim:**

> “## 44. Authoritative round-5 status ledger”

**Counter-check.**  The table includes `GAP-CASCADE-REACHABILITY`, but
`GAP-HUB-FANOUT-REACHABILITY` is introduced later in §46 and receives no row.
Nor does §46 state whether the new label replaces, refines, or merely supplies
one attack on the older gap.  An “authoritative” inventory is therefore not
literal.

**Proposed repair:** add an OPEN row for
`GAP-HUB-FANOUT-REACHABILITY` and state that it is the narrower
obstruction-specific subproblem described in Finding 4, not a replacement
for the broader cascade, initialization, or repair obligations.

### 6. MINOR — provenance omits the reviewed/output artifact

**Quoted claim:**

> “Input commit: `12980bc8` on branch `hunt/gap-raw`.  This authoring pass
> created no commit.”

**Counter-check.**  The qualified statement about the authoring pass can
remain, but this review examines the round-5 artifact at `d93d5768`.  The
round-3 and round-4 reviews already required input/base and reviewed/output
identifiers to be distinguished; round 5 repeats the omission.

**Proposed repair:** add “Reviewed/output artifact: `d93d5768`,” while keeping
the no-commit sentence explicitly limited to the authoring session.

### 7. NOTE — R5.2's radius-21 separation is sufficient and envelope-sharp, not value-sharp

**Quoted claim:**

> “Every six-window touched by those stones, and every local
> service/stabilizer cell used below, lies in the radius-21 neighborhood.”

> “Pairwise support distance at least 43 is a simple sufficient condition.”

**Independent derivation.**  A first placement can be eight from a pencil's
occupied support, a second placement can chain another eight, and a six-window
through that second stone can extend another five.  Thus the closed radius is
exactly

`8+8+5=21`.

This radius is attained for the stated *all touched alive-window envelope*:
from a support cell `o`, play at `o+8u` and then `o+16u`; a count-one window
through the second stone can extend to `o+21u`.  Disjoint closed radius-21
neighborhoods therefore remove every possible cross-window contact in the
proof, and support distance at least `21+21+1=43` is sufficient.

Under that premise the R5.2 value derivation is exact.  A same-hull pair is
handled by R5.1 and leaves every hull pure count two.  A split pair contributes
at most one return per hull.  L11.5 leaves each activated hull either pure
count two (`h=0,g<=2`) or as the nested derivative (`h=g=1`).  At most two
nested derivatives exist, so equation (20) gives

`TEMPO<=max(2,1+1)=2`.

No split pair creates a current imminent, and the constructed stabilizers are
legal; a filler only deletes labels and cannot increase `TEMPO`.

The constants are not proved necessary for the `M<=2` conclusion.  The
radius-21 extremal tail above is only count one because the two chained stones
are eight apart, and is invisible to `L_23` and immediate `TEMPO`.  At support
distance 42 the closed envelopes can touch, so the written proof no longer
applies, but the two-placement cadence cannot realize two radius-21 extremes
simultaneously.  No `M>2` leakage just below 43 is exhibited, and the minimum
separation for the value theorem remains **UNRESOLVED**.  This is consistent
with the source's express words “sufficient” and “simple sufficient”; it is
not a theorem defect.

**Proposed repair:** add one sentence saying that radius 21 is tight for the
conservative all-alive-window envelope but is not claimed sharp for the
one-cycle `M` bound.

### 8. NOTE — R5.3's `S_T` rejection is exact and independent of its tie-break

**Quoted claim:**

> “At `P_raw`, the actual `S_T` pair is `D@(-4,0), D@(2,0)`, and its handoff
> has `TEMPO=0`.  The transverse pair ... has handoff `TEMPO=2`.”

**Independent derivation.**  `D@(-4,0)` kills the common Q-window at start
`-4`; `D@(2,0)` kills the four starts `-3,-2,-1,0`.  No count-two/count-three
label survives, so `TEMPO=0`.

The transverse pair hits none of those five labels.  Future triggers
`{-1,2}` make starts `-3,-2,-1` count four with residuals

`{-3,-2}`, `{-2,3}`, `{3,4}`.

The first and third residuals are disjoint, while `{-2,3}` hits all three;
their hitting number is exactly two.  L10.4 supplies the reverse upper bound,
so the transverse-sealed handoff has exact `TEMPO=2`.

Any value-zero reply must delete all five count-two labels.  If even one
survives, two of its four empty residual cells are legal future triggers and
give positive `g`.  A first cell lexicographically earlier than `(-4,0)` hits
none of the five Q-labels, after which no one cell covers them because their
total intersection is the occupied pair `{(0,0),(1,0)}`.  After
`D@(-4,0)`, `(2,0)` is the unique empty common cell of the remaining four.
Thus (33) is the exact `S_T` pair.  The rejection is the strict objective gap
`0<2`; changing only the lexicographic tie-break cannot select the seal.

R5.3.1 is also exact.  From the three-anchor `Phi=0` root, two initial
Defender cells meet at most two disjoint prospective-window unions.  Attacker
uses the untouched launch, and the returned epoch has precisely `P_raw`'s
five graded labels.  This proves failure of the natural fresh-pair entrance
on an `S_T`-consistent history, not nonreachability of every alternative
sealed entrance.

**Proposed repair:** none.

### 9. NOTE — L11.6's shared-hub profile and strict potential are exact

**Quoted claim:**

> “The exact alive family is `F_H`: the five windows in (37) have count two,
> the four in (38) have count one, and there is no other alive label.”

**Independent derivation.**  On each of the central u- and v-lines, the
protected union is the ten-cell interval `[-4,5]`.  Its five length-six starts
`-4,-3,-2,-1,0` have Attacker counts

`2,1,0,1,2`.

Thus each line contributes two count-two labels and two count-one labels; the
count-zero middle interval is not alive.  On the w-line, the only six-window
contained in `X` is `[-2,3]_w`, at count two.  A noncentral line is parallel
to one central axis and meets each of the other two in at most one cell, so it
contains at most two cells of `X` and cannot contain a six-window wholly in
`X`.  Every other Attacker-touched window therefore has a cell outside `X`
and receives a blocker from (39), while no blocker enters `X`.

The exact profile is consequently `n_2=5,n_1=4`, giving

`Phi_H=5/9+4/(9sqrt(3))<1`, with `I=empty`.

Its `L_23` tier is pure count two, so L10.4 gives `TEMPO<=2`.  The pair in
L11.7 creates demand two, establishing exact `TEMPO=2`.

**Proposed repair:** none.

### 10. NOTE — L11.7's `M: 2 -> >=3` cascade has no uncounted Defender dodge

**Quoted claim:**

> “Every servicing Defender pair at `P_H'` hands over a position `R` with
> `TEMPO(R)>=3`.  Consequently `M(P_H')>=3`.”

**Independent derivation.**  The ordered pair `c=0,d=u` is empty and legal.
After it, `U^-` and `U^+` are the only imminent labels, each at count four,
with residuals

`{-2u,-u}` and `{2u,3u}`.

The residuals are disjoint, so `tau(P_H')=2`.  Every servicing pair must spend
one cell in each; there is no spare, nonminimum cover, or off-axis alternative.
All four possible service cells are nonzero u-axis cells.  The central axes
meet only at the now-occupied hub, so none of those cells lies in `V^-`,
`V^+`, or `W`.  Those three count-three labels survive every mandatory
service.  The service cells also remain sequentially legal because each lies
in its own still-alive imminent window before it is played.

At the resulting handoff, the already legal prospective triggers `v,w`
promote the two v-labels and the w-label to count four with residuals

`{-2v,-v}`, `{2v,3v}`, `{-2w,-w}`.

Their physical grounds are pairwise disjoint, so the future demand is three.
R4.1 gives `TEMPO>=3` after every servicing pair, and definition (21) gives
`M(P_H')>=3`.

The chronology matters: `A@c` fans five low labels across three axes;
`A@d` makes the two u-labels the saturated current `tau=2` service; only after
that mandatory service does the prospective pair `(v,w)` witness demand
three.  The source uses this chronology correctly.  There is no hidden claim
that the first placement alone immediately creates `tau=3`.

**Proposed repair:** none.

### 11. NOTE — R5.4 satisfies every side condition of unrestricted equation-(22) closure

**Quoted claim:**

> “The position `P^dagger` satisfies `tau(P^dagger)=2` and
> `M(P^dagger)=2` ... [yet] the unrestricted statewise implication (47) is
> false.”

**Independent derivation.**  Each remote gadget has exactly one alive
count-five label `Y_i`, with singleton residual `{z_i}`.  Consequently the
initial imminent family is exactly `{Y_1,Y_2}`, its residuals are distinct,
`tau=2`, and every servicing pair has occupancy `{z_1,z_2}` in one of the two
orders.  Both cells are legal and remain sequentially legal.

After that service the remote alive family is dead and the hub handoff has the
same alive and `L_23` data as `Q_H`, hence exact `TEMPO=2`.  Therefore
`M(P^dagger)=2`, equation (22) holds, and `S_T` is forced to the same
servicing occupancy.  The legal nonterminal response `(c,d)` returns the
epoch audited in Finding 10.  Its current `tau` is two; every next service is
again forced onto the two u-residuals; and the legal future pair `(v,w)` proves
that every serviced handoff has `TEMPO>=3`.  Thus the returned epoch has
`M>=3`.

All relevant side conditions hold: the initial and returned states are finite,
nonempty, nonterminal Defender-`FirstStone` epochs; the initial actual pair
services; its handoff has `TEMPO<=2`; the response is an ordered legal pair;
and the successor `M` is defined over its nonempty servicing set.  The fact
that `Phi(P^dagger)>1` is not a side condition of the expressly unrestricted
statewise implication.  It is instead the reason this counterexample does not
refute R4.6's strategy-reachable hypothesis or GAP-RAW.

**Proposed repair:** none to the theorem or its boundary.

### 12. NOTE — L11.1's sealed inventory is exact

**Quoted claim:**

> “The five count-two labels are `W_s` for `s=-4,-3,-2,-1,0`.  The six
> count-one residual arms are [equation (29)].  The six residual sets in (29)
> are pairwise disjoint.”

**Independent derivation.**  The adjacent pair belongs to seven Q-windows and
four transverse six-window pencils, for

`7+4*6=31`

distinct alive labels before sealing.  `D@(0,1)` kills five labels in the
R-pencil through `(0,0)` and five in the QR-pencil through `(1,0)`.
`D@(1,-1)` kills five in each of the other two transverse pencils.  Those
twenty labels are distinct, and neither seal lies on the Q-row.  The survivors
are therefore:

- Q starts `-4,-3,-2,-1,0`, each containing both Attacker endpoints;
- Q starts `-5` and `1`, each containing one endpoint; and
- one extreme count-one window in each transverse pencil.

Their six five-cell residuals are exactly

`{(-k,0)}`, `{(0,-k)}`, `{(-k,k)}`,
`{(1+k,0)}`, `{(1,k)}`, `{(1+k,-k)}`, for `1<=k<=5`.

Direct comparison of `q`, `r`, and `q+r` shows that these six sets are
pairwise disjoint.  Thus the exact profile is `n_1=6,n_2=5`, all other
`n_k=0`; the involution `rho(q,r)=(1-q,-r)` has the claimed action.

**Proposed repair:** none.

### 13. NOTE — L11.2 really exhausts killed-pencil, arm, and virgin returns

**Quoted claim:**

> “After any legal ordered Attacker pair from one exact sealed handoff, every
> high label is of exactly one of the following forms [an old common-Q label,
> or the unique pre-count-one extreme containing both triggers].”

**Independent derivation.**  A Defender-free label containing an old endpoint
was already alive at the sealed handoff and hence is one of the eleven labels
in Finding 12; every other old endpoint window already contains a seal and
stays dead.  A label containing neither old endpoint begins virgin and can
reach only count two.  Consequently:

- an old common-Q count-two label reaches count three or four according as it
  receives one or two Q-row triggers;
- a surviving count-one extreme reaches count two or three according as it
  receives one or two triggers; and
- pairwise-disjoint extreme residuals allow at most one such extreme to
  receive both triggers.

Thus two non-Q returns in one arm create exactly one count-three label; returns
in different non-Q arms create none; a killed-pencil or virgin return creates
no omitted high label; and every newly imminent label is Q-axial and contains
both triggers.  This closes the potentially missing return class, rather than
merely checking the six displayed focal lines.

**Proposed repair:** none.

### 14. NOTE — L11.3's depth table, hitting cells, and nested derivative are exact

**Quoted claim:**

> “On the positive side write it as `x=1+k`, `1<=k<=5`.  It promotes exactly
> `5-k` of the five common Q-labels to count three.”

**Independent derivation.**  For positive returns `x=2,3,4,5,6`, the exact
high-window starts are respectively

`{-3,-2,-1,0}`, `{-2,-1,0}`, `{-1,0}`, `{0}`, `empty`.

At `x=2` their residuals are

`{-3,-2,-1}`, `{-2,-1,3}`, `{-1,3,4}`, `{3,4,5}`,

all hit by `{-1,3}`.  Playing only `D@(-1,0)` leaves precisely `W_0` at
count three with residual `{3,4,5}` and `W_1` at count two with residual
`{3,4,5,6}`.  The first residual is nested in the second after every
two-trigger promotion, so this component has exact `h=g=1`.

At `x=3,4,5`, the cell `(2,0)` hits every high window; `x=6` creates none.
Reflection supplies the negative cases.  In the “exactly one Q return” branch,
the other return cannot already occupy any displayed cleanup cell, because
those cells themselves lie in a Q residual arm.  Every cleanup cell is empty,
belongs to a current alive high label, and is therefore radius-eight legal.

**Proposed repair:** none.

### 15. NOTE — L11.4 covers all 45 axial pairs, including endpoint extremes and gaps

**Quoted claim:**

> “There is a set of at most two empty Q-cells meeting every post-return
> Q-window of count at least three.”

**Independent derivation.**  Sorting the four Attacker coordinates gives two
rank triples.  Every interval containing at least three of the four contains
the left or right rank triple.  A feasible nonconsecutive triple has an empty
internal integer gap; the fourth stone is outside that rank interval.  A
consecutive triple `{u,u+1,u+2}` has its four possible length-six windows hit
by `{u-1,u+3}`.  When all four stones are consecutive,
`{s_1-1,s_4+1}` hits the five possible high windows.

An independent pair taxonomy gives the same cover for all
`C(10,2)=45` unordered pairs of Q-arm cells:

- on one positive side, `{2,3}` is hit by `{-1,4}`;
  `{2,y}` with `y>=4` by `{-1,3}`; and
  `3<=x<y<=6` by `{2}`;
- the negative side is the `rho` image; and
- for opposite returns `-i` and `1+j`, the four cases
  `(i,j)=(1,1)`, `i=1<j`, `j=1<i`, and `i,j>=2` are hit by
  `{-2,3}`, `{-2,2}`, `{-1,3}`, and `{-1,2}`, respectively, with an
  unused point replaced by a filler when a rank triple is infeasible.

Every required cover point is an empty Q-cell in an alive high window.  The
transversal therefore deletes the full high family, including a promoted
endpoint-exclusive Q extreme, not only the five old common labels.

**Proposed repair:** optional clarity only: state explicitly that coincident
gap choices coalesce and that the zero-feasible-triple case uses two fillers.
The current proof already implies both facts.

### 16. NOTE — R5.1's isolated one-cycle value bound is complete and cadence-correct

**Quoted claim:**

> “For every legal ordered Attacker pair `b`, the returned nonterminal
> Defender epoch `P_b` satisfies `M(P_b)<=2`.”

**Confirmation.**  The exhaustive partition is by zero, one, or two Q-arm
triggers.  With two, Finding 15 deletes every high label.  With one, Finding 14
does so; the non-Q or virgin trigger adds no high label.  With zero, either the
graded tier is pure count two or two same-arm returns create one count-three
extreme, killed by one residual cell.  The constructed reply always services
all current imminents and leaves a pure-count-two tier, so L10.4 gives
`TEMPO<=2` and definition (21) gives `M<=2`.

The start has only two Attacker stones, so the ordered pair cannot complete a
six.  Every nonfiller Defender cell is within five of unchanged Attacker stock
in an alive high window.  A second cleanup remains legal even after the first
kills its own labels; when fewer than two cleanup cells are needed, L1.2's
finite max-q construction supplies sequential legal fillers.  The proof uses
the actual Attacker pair followed by the actual Defender pair and therefore
respects the inherited 2:2 cadence.

**Proposed repair:** none.

## Per-theorem verdicts

| Claim | Source status | Review verdict | Disposition |
|---|---|---|---|
| L11.1 exact six-extreme inventory | PROVEN | **CONFIRMED** | Exact 31-label audit; profile `n_1=6,n_2=5`; six residual arms disjoint |
| L11.2 exhaustive high-label reduction | PROVEN | **CONFIRMED** | Killed endpoint windows stay dead; virgin labels reach at most count two; the 0/1/2 Q-trigger partition is complete |
| L11.3 depth table and nested derivative | PROVEN | **CONFIRMED** | All starts, residuals, cleanup cells, and exact `h=g=1` derivative recompute |
| L11.4 every two-Q/nonconsecutive return | PROVEN | **CONFIRMED** | All 45 axial pairs have the claimed at-most-two-cell transversal |
| R5.1 isolated sealed one-cycle bound | PROVEN | **CONFIRMED** | Complete legal-pair case split; servicing, legality, nonterminality, and cadence all close |
| L11.5 split stabilizers | PROVEN | **CONFIRMED** | Direct consequence of the exact depth table; one legal action per activated pencil |
| R5.2 separated sealed one-cycle bound | PROVEN at stated class | **CONFIRMED** | Radius-21 hull separation is sufficient; exact sharp separation for the `M` conclusion remains unresolved and is not claimed |
| Cross-hull and iterative sealed closure | OPEN | **CONFIRMED** | Neither radius-overlap returns nor the next derivative turn is classified |
| R5.3 natural `S_T` entrance exclusion | PROVEN | **CONFIRMED** | Strict value comparison `0<2`; rejection is not a lexicographic tie artifact |
| R5.3.1 strict-root `S_T` history | PROVEN | **CONFIRMED** | Three disjoint launch unions give an untouched exact `P_raw` on an ordinary 2:2 history |
| Alternative forced-service seal entrance | OPEN | **CONFIRMED** | R5.3 excludes only the natural fresh-pair response |
| L11.6 strict shared-hub profile | PROVEN | **CONFIRMED** | Exact `n_2=5,n_1=4`, blocker comprehension, strict potential, and `TEMPO=2` |
| L11.7 shared-hub cascade | PROVEN | **CONFIRMED** | Both service cells are mandatory on u; the surviving v/v/w demands have three disjoint residual grounds |
| R5.4 unrestricted `V_T` closure counterexample | PROVEN | **CONFIRMED-WITH-ERRATA** | Every equation-(22) side condition holds; formally define the separated protected supports as in Finding 2 |
| L11.8 immediate strict-hub defusal | PROVEN | **CONFIRMED** | `D@0` kills all nine hub labels and a filler gives `M=0` |
| L11.9 two strict `tau=0` instances | PROVEN | **CONFIRMED** | Hub and sealed-root values recompute; neither is universal initialization |
| First isolated sealed return cannot source the hub escape | PROVEN | **CONFIRMED** | Exhaustive L11.2–L11.4 classification leaves at most one high axis |
| Failure of unrestricted statewise equation-(22) closure | REFUTED closure claim | **CONFIRMED** | `P^dagger` has `V_T`, its forced actual service has `TEMPO=2`, and legal response `(c,d)` returns `M>=3` |
| `GAP-HUB-FANOUT-REACHABILITY` as an obstruction-specific problem | OPEN | **CONFIRMED** | Strict hub is defusable and forced embedding has high potential; causal reachability is genuinely unresolved |
| `GAP-HUB-FANOUT-REACHABILITY` as the exact/sole gate | claimed resume characterization | **DOWNGRADE** | Fixed-policy reachability, strategy-independent forcing, and positive universal repair have different quantifiers; other escape classes remain |
| `GAP-CASCADE-REACHABILITY` | OPEN | **CONFIRMED** | No strict-root universal Attacker route and no universal causal pre-emption invariant is proved |
| `GAP-TEMPO-INITIALIZATION` | OPEN | **CONFIRMED** | General strict-root `tau=0` geometry remains unproved |
| `GAP-TEMPO-REPAIR` | OPEN | **CONFIRMED** | One-cycle value bounds do not yield same-strategy all-response induction |
| GAP-RAW | OPEN | **CONFIRMED** | No perpetual Defender strategy and no universal Attacker win from a strict root |
| New machine verification | none | **CONFIRMED** | No Cargo, Lean, harness, search, `[UNRUN]`, or `VERIFIED` evidence is asserted |

## Overall verdict

**SOUND-WITH-ERRATA.**  The complete first-return sealed-pencil
classification is **CONFIRMED**.  The isolated and fully radius-21-separated
one-cycle `M<=2` value theorems are **CONFIRMED** at their exact scopes.  The
natural `S_T` seal rejection is **CONFIRMED** and is robust to its defined
tie-break.  The shared-hub `M:2 -> >=3` cascade is **CONFIRMED**, with both
mandatory service cells and all three later demands accounted for.  R5.4's
refutation of unrestricted statewise equation-(22) closure is
**CONFIRMED-WITH-ERRATA** only because its finite separation region should be
named explicitly; its mathematics and side-condition audit survive.

The most severe finding is Finding 4 (**MAJOR**): hub-fanout reachability does
not “gate everything.”  Against one fixed strategy it can refute only that
strategy; a GAP-RAW counterroute needs forcing against every Defender
strategy; and positive hub avoidance does not by itself close general
`tau=0` initialization, other fanouts, cross-hull derivatives, or all-response
repair.

The exact unresolved obstacles are therefore:

1. universal strict-root `tau=0` initialization of `M<=2`;
2. one named Defender strategy preserving `M<=2` after every response on all
   of its reached histories;
3. the quantifier-correct causal reachability or exclusion of the shared hub,
   together with other still-unclassified escape classes; and
4. if the separated theorem is to be sharpened, the minimum support separation
   needed for its `M<=2` conclusion—radius 21 is sharp only for the
   conservative all-touched-window envelope.

These are **UNRESOLVED**.  None may be replaced by the stronger false claim
that the hub subproblem alone decides GAP-RAW.
