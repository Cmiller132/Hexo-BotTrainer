# R-G2-REV — Round-4 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`, `8261c177`

**Document:** `GAP_RAW_PROOF_ROUND4.md`

**Method:** first-principles hostile proof audit. The prerequisite corpus was
read in the required order and in full:
`GAP_RAW_PROOF_ROUND2.md`, `GAP_RAW_PROOF_ROUND3.md`,
`GAP_RAW_REVIEW_ROUND3.md`, then `GAP_RAW_PROOF_ROUND4.md`.
`docs/PROOF_TSS_DEFENDER_ZONES.md` §8 and the frozen NQ2 quiet-locality proof
at `5e06c29c:PROOF_QUIET_LOCALITY.md` were then checked for non-resurrection.
Every displayed coordinate gadget, residual family, potential, hitting number,
radius-eight placement, and 2:2 phase transition was recomputed by hand.
The historical harness was not run or treated as evidence. **No Cargo command,
Lean build, harness, or machine enumeration was run.**

**Overall verdict:** **SOUND-WITH-ERRATA.** Every formal theorem R4.1–R4.8
survives at its stated mathematical scope. The most severe defect is a
**MAJOR** overstatement immediately after R4.7.1: inequality (28) excludes a
uniform positive floor on every dormant component, not every additive ledger
that charges any one dormant type. Section 24.1 also needs to make its
one-cycle mode-(b) partial explicit, and the provenance should identify the
reviewed artifact.

## Numbered findings

### 1. NOTE — inherited definitions are used without drift

**Quoted claim:**

> “`I`, `tau`, `Phi`, `Theta_2`, `B_2`, `kappa_2`, service, handoff, and
> ripeness retain their earlier meanings.” (§24, lines 12–16)

**Confirmation.** Round 4 consistently uses the round-2/3 meanings:

- positions are finite, nonempty (`A∪D≠∅`), blanket Maker–Breaker positions;
- a normal turn has sequential `FirstStone`, `SecondStone` placements and the
  legality radius is the inclusive value eight around either color's occupied
  set;
- alive means Defender-free and Attacker-touched; virgin windows are not alive;
- `I(P)` contains exactly alive count-four/count-five windows at a nonterminal
  position, and `tau(P)` is the hitting number of their nonempty residuals;
- `Phi` sums all alive labels, `Theta_2` sums alive count-at-least-two labels,
  and `B_2=Theta_2`; and
- service is performed by the strategy's actual ordered pair, while ripeness is
  defined at a nonterminal Attacker-`FirstStone` handoff with `I=∅`.

The new quantities `TEMPO` and `M` do not redefine any inherited account.
They deliberately permit `Theta_2>1`, so the round-3 root `P_*` remains a
counterexample to canonical J rather than being silently reclassified.

**Proposed repair:** none.

### 2. NOTE — R4.7.2's grade-three countertheorem is confirmed at its accumulated-stock scope

**Quoted claim:**

> “There is no Defender rule renewing `B_3<1` from every
> `I=empty, B_3<1` position against every Attacker pair.” (R4.7.2,
> lines 831–837)

**Independent derivation.** Initially the four separated adjacent pairs have
`B_3=0` and `I=∅`. Their common-Q unions `U_i=[-4,5]×{30i}` are disjoint, so
two Defender cells leave at least two untouched. Attacker uses the two
placements of its turn at `(2,30i)` on those two sites, one placement per site. Each is empty
and adjacent to an old endpoint, so both were already legal and the ordered
pair respects the 2:2 cadence. A consecutive triple `{0,1,2}` belongs to
exactly four Q-windows, with starts `-3,-2,-1,0`. Thus the two extensions
create exactly eight count-three labels of weight `1/(3sqrt(3))`, giving

`B_3=8/(3sqrt(3))=8sqrt(3)/9>1`.

No count-four label or six is created, so the returned epoch is nonterminal
with `I=∅` and `tau=0`. The initial raw `Phi` is large; the corollary is
correctly confined to the every-accumulated-stock-start renewal proposal and
does not touch GAP-RAW roots.

**Proposed repair:** define the quantified starts in the corollary explicitly
as finite nonterminal Defender-`FirstStone` positions. This is a scope
clarification, not a mathematical repair.

### 3. NOTE — R4.8's two sealed-pencil return lemma is confirmed

**Quoted claim:**

> “Defender can play the corresponding one-cell stabilizer in each activated
> component and hand over a position with `TEMPO<=2`.” (R4.8, lines 886–904)

**Independent sealed-profile audit.** For
`A={(0,0),(1,0)}` and `D={(0,1),(1,-1)}`, each sealing cell lies at the empty
intersection of two off-axis pencils and kills five of the six labels in each.
No Q-axis label is hit. The exact surviving profile is therefore:

- two endpoint-exclusive Q labels at count one;
- one extreme count-one label in each of four off-axis pencils; and
- all five common-Q labels at count two.

Hence `(n_1,n_2)=(6,5)`, `I=∅`, `tau=0`,
`Theta_2=5/9`, and `Phi=5/9+2/(3sqrt(3))`.

After the legal positive extension `(2,0)`, the four count-three Q-windows
start at `-3,-2,-1,0`. The stabilizer `(-2,0)` is empty and legal—it lies in
an alive triple window—and kills the first two. It also kills the old left
count-two label. The complete surviving `L_23` component has residuals

- count three, start `-1`: `{-1,3,4}`;
- count three, start `0`: `{3,4,5}`; and
- count two, start `1`: `{3,4,5,6}`.

A singleton trigger `3` or `4` matures both high labels while leaving common
cell `4` or `3`; other singleton triggers mature at most one. Thus `h=1`.
For the pair `{3,4}`, the three residuals become `{-1}`, `{5}`, and `{5,6}`,
with hitting number two. Every other pair leaves a common `3` or `4` for the
two high labels or matures at most one of them; including the low label still
needs no more than two hits. Hence `g=2`. The negative extension is its
reflection, stabilized by `(3,0)`.

With two activated components, the two stabilizers are individually already
legal, distinct, and may be played sequentially. The enlarged-footprint
premise prevents cross-label interactions. Activated components have
`h≤1,g≤2`; untouched sealed pencils have pure count-two graded tiers and hence
`h=0,g≤2`. Equation (20) gives top-two demand at most `1+1=2` and every
within-component demand at most two. R4.8 is correct for precisely its stated
one-attack bounded class.

**Proposed repair:** none to R4.8 itself.

### 4. MINOR — §24.1's mode-(b) partial needs an explicit `S_T`-bound corollary

**Quoted claim:**

> “this round achieves mode (c) through R4.3 and R4.7.1, and a narrowly bounded
> mode-(b) partial through R4.8.” (lines 41–44)

**Status audit.** Mode (c) is achieved: R4.3 and the formal uniform-floor part
of R4.7.1 are genuine, sharply scoped impossibility theorems extending the
round-3 boundary.

The original mode-(b) criterion required an exact strategy `S` and an invariant
proved on a nontrivial bounded history class. R4.8 is phrased only as an
existential response: “Defender can play” two stabilizers. It does not state
that `S_T` creates the transverse seals, that the sealed class is reached under
`S_T`, or that it is closed under `S_T`. Indeed, §31 expressly leaves the next
return classes open.

There is a valid one-cycle `S_T` corollary: after the prescribed two extensions
the stabilizer pair witnesses `M≤2`, so `S_T`'s minimizing actual pair also
hands over `TEMPO≤2`. Thus “narrowly bounded mode-(b) partial” is defensible as
an ingredient; it is not a completed mode-(b) strategy result. The source
should state that corollary and boundary rather than leave the strategy binding
implicit.

**Proposed repair:** add the explicit `S_T`-bound one-cycle corollary and say
that mode (b) itself remains unachieved because neither reachability nor closure
of sealed profiles is proved. This finding does not downgrade R4.8's theorem.

### 5. NOTE — canonical J, global pairing, and NQ2 remain refuted at their prior scopes

**Quoted claim:**

> “Canonical J, canonical renewal, and pointwise-dominating subunit accounts
> remain refuted at their round-3 scopes.” (lines 1009–1011)

**Confirmation.** Nothing in round 4 revives the three prohibited routes.

- **Canonical J.** `TEMPO` can be zero or at most two with arbitrarily large
  `Theta_2`; in particular it accepts the safe `P_*` successors at
  `Theta_2≥11/9`. No `B_2<1` renewal or pointwise domination is claimed.
  R4.6 uses A₂'s actual-service conclusion, not the false J antecedent.
- **Global pairing.** Theorem T8 in `docs/PROOF_TSS_DEFENDER_ZONES.md` §8
  excludes a position-independent partial matching whose pairs cover every
  window. Round 4 uses state-specific hitting sets, minimization over actual
  legal pairs, and history-specific seals/stabilizers. It proposes no global
  matching.
- **NQ2.** The frozen NQ2 witness refutes pruning a quiet SecondStone consume
  universe to `join_live` or adjacency: its unique winning move is a remote
  block of the opponent's five. R4.1 maximizes over **all** legal ordered
  Attacker pairs in the blanket Maker–Breaker game and does not prune a search
  universe. Its observation that a virgin cell cannot create an imminent
  Attacker label within the current two placements is a demand calculation,
  not a claim that the cell is strategically dispensable in Maker–Maker play.
  R4.8's restricted move is an explicit premise of a bounded lemma, not a
  consumption rule.

**Proposed repair:** none.

### 6. NOTE — no `[UNRUN]` or `VERIFIED` status was introduced

**Quoted claim:**

> “No new harness case was added, so there is no new `[UNRUN]` case and no new
> `VERIFIED` claim.” (lines 18–20)

**Confirmation.** The round-4 document labels its hand arguments `PROVEN` and
its missing closure steps `OPEN`. The only occurrences of `VERIFIED` and
`[UNRUN]` say that no such new evidence exists. The diff from input
`0f7e9405...` to reviewed artifact `8261c177` adds only the round prompt and
`GAP_RAW_PROOF_ROUND4.md`; it changes no harness, production, verifier, Cargo,
or Lean source. Historical round-2/3 machine results are cited only as banked
facts at their inherited scopes.

**Proposed repair:** none.

### 7. MINOR — provenance repeats round 3's stale-output omission

**Quoted claim:**

> “Input commit: `0f7e9405...` ... This authoring pass created no commit.”
> (§36, lines 1067–1068)

**Counter-check.** The input identifier is correct and the qualified
authoring-pass sentence may remain, but the reviewed document is committed at
`8261c177`. Round 3's hostile review already required provenance to distinguish
the input/base commit from the later reviewed/output artifact. Round 4 repeats
that omission.

**Proposed repair:** add “Reviewed/output artifact:
`8261c177`.” Keep the existing sentence only as a statement about the authoring
session itself.

### 8. NOTE — R4.4's `Phi=1/sqrt(3)` zero-contact obstruction is exact

**Quoted claim:**

> “If an initial Defender pair at `P_cas` has `K(P_cas,a)=0`, the displayed
> Attacker pair creates `tau=3`.” (R4.4, lines 487–490)

**Independent coordinate audit.** The old Attacker set has nine distinct
cells: parameters `-3,-2,-1` on each of the three central axes. Each protected
window `W_v={tv:-3≤t≤2}` contains exactly the three stones on its own axis.
The blocker set (23) is finite, avoids the protected union `X`, and kills every
other Attacker-touched window. On a central line, a distinct length-six
interval must leave the one protected six-interval. On a noncentral line of
one axis, the line is parallel to one central line and meets each of the other
two in at most one cell, so at most two of its cells lie in `X`. If
`a∈U∩A`, every cell of `U\X` is `a+tv` for the axis of `U` and
`1≤|t|≤5`, hence is included in (23). Thus the exact alive profile is
`n_3=3` and all other `n_k=0`.

Each label has `e=3` and weight
`lambda^-3=1/(3sqrt(3))`, so

`Phi(P_cas)=Theta_2(P_cas)=3/(3sqrt(3))=1/sqrt(3)<1`,

with `I=∅` and `tau=0`. The sequential kill quantity is a sum of nonnegative
`kappa_2` terms. Since these are the only graded labels, `K=0` means neither
Defender cell lies in any of the three protected windows; in particular
`c=(0,0)` and `y=(1,0)` remain empty. The first cell is legal from the old
axis stock and the second is adjacent after `c`, so the ordered pair respects
the radius-eight rule and the Defender-pair/Attacker-pair cadence.

The resulting focal residuals are exactly

- Q-axis count five: `{(2,0)}`;
- R-axis count four: `{(0,1),(0,2)}`; and
- QR-axis count four: `{(1,-1),(2,-2)}`.

Their grounds are pairwise disjoint, hence their hitting number is exactly
three. Every nonfocal window meeting old `A` retains its blocker; a label
using only the two new cells has count at most two. No six is formed, and
adding any further imminent labels cannot lower the focal hitting number.
L1.2 therefore forces a missed label and completion on the following Attacker
turn. Conversely, `D@c` is legal and kills all three old labels at once, so the
result excludes only zero-contact replies, exactly as stated.

**Proposed repair:** none.

### 9. NOTE — R4.5's universal high-stock cascade is confirmed outside the root threshold

**Quoted claim:**

> “Attacker wins from `P_hi` against every legal ordered Defender pair.”
> (R4.5, lines 567–591)

**Independent coordinate audit.** In one gadget the protected horizontal
window contains `(-4,0),(-3,0)` and has count two. The protected vertical
windows with starts `-2,-1,0` each contain `(0,1),(0,2),(0,3)` and have
count three. Definition (25) really kills every other touched label: on the
central Q-line the sole protected length-six subsegment is `H`; on the
central R-line the protected interval has length eight and exactly the three
listed length-six subsegments; a noncentral parallel line meets the cross in
at most one or two cells. Hence every undesired six has a cell outside `Z_i`,
and no blocker lies in a protected window.

Translation by 30 prevents cross-gadget windows. The exact union profile is
therefore `n_2=3,n_3=9`, so

`Phi(P_hi)=Theta_2(P_hi)=3/9+9/(3sqrt(3))=1/3+sqrt(3)>1`,

with `I=∅`. Two Defender cells meet at most two disjoint `Z_i`, leaving one
gadget untouched. There `c=b_i` is adjacent to vertical stock and
`d=b_i+(1,0)` is adjacent to `c`. The Attacker pair is legal and returns a
nonterminal Defender epoch with residuals

- horizontal: `{(-2,0),(-1,0)}`;
- vertical: `{-2,-1}`, `{-1,4}`, and `{4,5}` on `q=0`.

The vertical path has hitting number two, attained by `{-1,4}` and forced by
its disjoint extreme edges. Its ground is disjoint from the horizontal pair,
so the focal family has hitting number three. L1.2 gives a forced completion
after every intervening reply. The theorem is a genuine high-stock win and,
because its displayed `Phi` exceeds one, is not a GAP-RAW refutation.

**Proposed repair:** none.

### 10. NOTE — R4.3 is correct at its stated remote-2-saturating scope

**Quoted claim:**

> “No state predicate is simultaneously root-covering, service-sufficient,
> and remote-2-saturating.” (R4.3, lines 643–659)

**Independent isolator audit.** For `A_4={(0,0),(1,0),(2,0),(3,0)}`, Q-axis
starts meeting `A_4` are `-5,...,3`. The blocker `(-1,0)` kills starts
`-5,...,-1`, and `(6,0)` kills `1,2,3`, leaving only start zero. On each
R-pencil through `(t,0)`, `(t,-1),(t,1)` hit all six windows. On each QR-pencil,
`(t-1,1),(t+1,-1)` do the same; the listed row blockers handle interior `t`
and `(-1,1),(4,-1)` handle the boundaries. Thus the sole alive label is
`W_4={(t,0):0≤t≤5}`, count four with residual `{(4,0),(5,0)}` and weight
`1/3`.

At separation at least 30, copies have no cross-window interaction. Two copies
form a valid nonterminal Defender-`FirstStone` root with exact `Phi=2/3<1`,
so root coverage gives `V(P^[2])`. Remote-2 saturation gives `V(P^[3])`.
The three-copy position has three pairwise-disjoint residual pairs and hence
`tau=3`, contradicting global service sufficiency. The three subclasses in
R4.3.1 really are remote-2-saturating: after sufficiently large separation a
third copy creates no new fixed-radius local type, does not change a component
maximum, and does not change the two largest scores already duplicated by two
copies.

The quantifier scope matters. This theorem does **not** exclude, and cannot be
quoted against, a root-indexed/history-indexed predicate, a condition asserted
only on one strategy's reachable histories, an additive or signed account, an
explicit third-component overflow test, or the current global coordinate
`tau≤2`. Such classes could still be used in a GAP-RAW strategy proof. The
source already acknowledges additive, signed/path, and third-overflow escapes;
adding the root/history/strategy-indexed escapes would make the boundary fully
explicit.

**Proposed repair:** optional clarification after R4.3.1: “Root-dependent,
history-dependent, and strategy-reachable predicates are also outside this
statewise remote-saturation theorem.”

### 11. NOTE — L10.9 and R4.7's unbounded dormant-component construction are exact

**Quoted claims:**

> “The hitting number of the twenty-six count-one residuals is exactly eight.”
> (L10.9, lines 708–730)

> “at least `T-floor(T/4)` sites still contain an alive count-one label.”
> (R4.7, lines 748–770)

**Independent hitting-set audit.** A Defender-free adjacent pair has 31 alive
labels: seven Q-axis labels, of which five are common count-two labels and two
are endpoint-exclusive count-one labels, plus four off-axis six-window
count-one pencils. The two exclusive-Q residuals lie on opposite sides and
need two Q-row cells, attained by `(-1,0),(4,0)`.

Each off-axis pencil needs two line incidences because its two extreme-window
residuals are disjoint. The four lines are `q=0`, `q=1`, `q+r=0`, and `q+r=1`.
Their only empty double-incidence intersections are `(0,1)` and `(1,-1)`;
the same-end intersections `(0,0),(1,0)` are occupied, and the remaining line
pairs are parallel. Five empty cells therefore supply at most seven of the
eight required incidences, so at least six off-axis cells are needed. The six
displayed cells

`(0,1),(0,-4),(2,-1),(1,-1),(-3,3),(1,3)`

hit every off-axis pencil. The empty Q channel intersects those four lines
only at the occupied endpoints, so the lower bounds add to eight; the displayed
eight-cell union attains them. It also hits all five common Q labels.

**Independent temporal audit.** R4.7's root has `A=∅` but is nonempty because
of the Defender anchors, so it is in the inherited normative domain and has
`Phi=0`. The private prospective-window sets `V_j` are mutually disjoint and
exclude their distance-eight anchors. Before launch `t`, `2t` Defender
placements and `t-1` already consumed sites exclude at most `3t-1` sites;
`N≥3T≥3t` leaves an untouched one. The anchor makes the first endpoint legal
at distance exactly eight and adjacency makes the second endpoint legal.

No Defender stone preceded the launch inside its private footprint. Erasing
that site's original 26 count-one labels later costs eight distinct Defender
cells by L10.9. Across all sites, the total `2T` placements can therefore erase
at most `floor(T/4)` sites. Attacker never revisits an old site, so a surviving
count-one label is not promoted away. Different private footprints yield
different dormant components. All alive counts remain at most two, hence
`I=∅` and `tau=0` throughout.

**Proposed repair:** none.

### 12. MAJOR — R4.7.1's formal uniform-floor theorem is sound, but its prose overclaims the excluded account class

**Quoted claims:**

> “No root-uniform nonnegative account satisfying
> `C(P)>=epsilon * N_dorm(P)` can remain below `B` ...” (R4.7.1, lines 778–789)

> “This covers every component-additive, translation-invariant deadline ledger
> which gives a fixed positive charge even to a remote count-one component.”
> (lines 791–794)

**Confirmation and scope counterexample.** The formal implication from (28) is
correct. More explicitly, fix finite `B`, `epsilon>0`, and one state function
`C` satisfying `C(P)≥epsilon N_dorm(P)` at every finite nonterminal Defender
epoch. R4.7 proves

`not [for every normative P_0, there exists a pure S such that, for every legal S-consistent Attacker continuation and every reached Defender epoch P, C(P)<B]`.

Indeed, choose `N` so the constructed `P_N` forces
`N_dorm>B/epsilon`. For that one normative root, every pure Defender strategy
has a legal Attacker continuation reaching an epoch with `C>B`. This makes
“root-uniform” mean that `C`, `B`, and `epsilon` are fixed across roots.

The following prose is strictly broader than (28). Translation invariance and
component additivity do not imply a uniform `epsilon` floor over **every**
dormant component type. For example, let

`C_select(P)=number of dormant components having the exact pristine fresh-pair type`.

This is nonnegative, component-additive, translation-invariant, and gives a
fixed positive charge to one remote count-one component type. Yet on the
Defender turn following a launch at `c_j=(0,R_j)`, Defender can play the legal
cell `c_j+(-5,0)`. It lies in the left endpoint-exclusive count-one Q-label,
kills that label, and changes the exact 31-label pristine type. One cell cannot
erase all count-one stock because L10.9's hitting number is eight. Repeating
this mutation after each nonfinal launch leaves at most the just-created site
pristine; at the final audited epoch only the last launch need be pristine,
even though `N_dorm` is unbounded. Thus R4.7 does not force `C_select` to grow.

Likewise, a fixed intrinsic ledger may assign zero to at least one
strategy-reachable damaged dormant type and positive charge only to selected
types. History-, age-, or rank-dependent charges may tend to zero, and a bound
`B(P_0)` may depend on the root's initial anchor stock. (There are only finitely
many residual types inside one fixed launch footprint, so if a fixed
component-additive `f(type)` is positive on **all** of them, their positive
minimum restores inequality (28).) Such coordinates can still be combined
with `tau`, `TEMPO`, or a strategy-reachable invariant; this review does not
claim that any such combination closes GAP-RAW, only that R4.7.1 has not
excluded it.

The sentence at lines 796–804 is valid only conditionally: a positive
`f(type)` is fatal when the quantified play can force arbitrarily many live
copies of that **same** type, not merely arbitrarily many dormant components
of unspecified residual types.

**Proposed repair.** Retitle the corollary “uniform positive dormant-component
charge is impossible.” Replace “a remote count-one component” with “every
dormant component, uniformly by one fixed `epsilon`,” define `root-uniform`,
and state the strategy quantifiers explicitly. Qualify the later clone sentence
as “admitting arbitrarily many strategy-forced copies of that same type.” The
formal R4.7.1 verdict remains **CONFIRMED-WITH-ERRATA**.

### 13. NOTE — R4.1's exact `TEMPO` factorization is confirmed

**Quoted claim:**

> “`TEMPO(Q) = max_b tau(Q+A@b_1+A@b_2)`” and “`TEMPO(Q)<=2` if and only if
> `Q` is unripe.” (R4.1, lines 48–55)

**Independent derivation.** At an `I(Q)=∅` handoff every alive label has count
at most three. A virgin or count-one label reaches at most count two or three
after the two placements, so every newly imminent label was already in
`L_23(Q)`. Labels affected by one trigger all contain that cell and are in one
physical-window intersection component. If a pre-count-two label contains
both triggers, it intersects every label affected at either trigger; hence a
pair affecting two components contributes exactly one trigger to each and can
promote only pre-count-three labels there.

Different components have disjoint physical windows and therefore disjoint
residual grounds, so their hitting numbers add. A one-component attack costs at
most `g_C`; a two-component attack costs at most the sum of the corresponding
`h_C` values; and the two largest such values give the upper bound in (20).
For a pair with only one relevant trigger, L10.2 is valid: a maximizing trigger
`c` matures a count-three window `W`, and a second cell
`d∈E_Q(W)\{c}` is distinct and already legal. Shrinking old residuals and
adding labels cannot reduce their hitting number, so `g_C≥h_C`.

The reverse inequality is also exact. Every cell used by an `h_C` or `g_C`
maximizer is empty in an alive window, hence was already legal at `Q` by L6₂.
Two such cells are sequentially legal in either order and cannot complete six,
because the largest pre-count is three. A `g_C` maximizer realizes its arm of
(20); maximizing singleton triggers in two distinct components are distinct,
legal, and realize the additive arm. With only one component, the padded second
`h` value is zero and L10.2 makes the `g` arm dominate. Thus (20) equals the
maximum over the full legal ordered-pair universe, not a locality-restricted
sample. Round 2's definition of ripeness then gives the threshold equivalence.

**Proposed repair:** none.

### 14. NOTE — L10.3 Defender monotonicity is confirmed

**Quoted claim:**

> “Defender augmentation does not increase `TEMPO`.” (L10.3, lines 205–224)

**Independent derivation.** Adding Defender stones deletes alive labels and
does not alter the count or residual of a surviving label. If both triggers of
a `Q'` pair are relevant, both lie in surviving old count-two/count-three
labels and were already legal at `Q`; the `Q` attack produces a residual
superfamily with demand no smaller. If only one trigger is relevant, its
one-trigger demand is bounded by the corresponding surviving-label demand at
`Q`, and L10.2 embeds that demand in a fully legal two-trigger attack at `Q`.
A second cell made newly legal only by the additional Defender stones is
irrelevant to imminent creation and creates no missing case. Taking maxima
proves `TEMPO(Q')≤TEMPO(Q)`.

**Proposed repair:** none.

### 15. NOTE — R4.2 and R4.6 bind the same actual Defender pair

**Quoted claims:**

> “`M(P)<=2` if and only if one legal ordered reply both services `I(P)` and
> hands over an unripe position.” (R4.2, lines 60–63)

> “If `M(P_0)<=2` and ... every later Defender epoch ... again satisfies
> `M(P')<=2`, then `S_T` blocks forever.” (R4.6, lines 273–276)

**Independent derivation.** A finite occupied set has a finite nonempty normal
legal frontier, so the legal ordered reply set and `Serv(P)` are finite. For
each `a∈Serv(P)`, its actual handoff `Q_a` has `I=∅`, and R4.1 says
`TEMPO(Q_a)≤2` exactly when that same handoff is unripe. The minimum in (21)
therefore characterizes one pair doing both jobs; if `Serv(P)=∅`, the declared
value `∞` correctly makes the predicate false.

The deterministic policy `S_T` selects the same minimizing ordered pair,
including its order and filler. Under R4.6's explicit closure hypothesis, that
pair services and produces an unripe handoff; every legal Attacker response is
nonterminal and returns with `tau≤2`; the next assumed finite `M≤2` supplies
the next actual service pair. Induction gives `Service(S_T,P_0)`, and A₂—not
canonical J—gives survival. No separately existential action is conjoined.

**Proposed repair:** none.

### 16. NOTE — the pure-count-two, fresh-channel, and O1 stress regressions survive

**Quoted claims:**

> “If every label in `L_23(Q)` has count two, then `TEMPO(Q)<=2`.” (L10.4,
> lines 292–299)

> “the three physical unions `U_Q`, `U_p`, and `U_p'` are pairwise disjoint.”
> (L10.5, lines 340–357)

> “leave at least `10t-8(t-1)=2t+8` focal count-two labels alive.” (L10.6,
> lines 401–419)

**Independent derivation.** With only pre-count-two graded labels, a singleton
trigger has demand zero and round-2 R7.4₂ gives every two-trigger residual
family hitting number at most two. Hence every `h_C=0`, every `g_C≤2`, and
L10.4 follows regardless of total `Theta_2`. This correctly accepts the
`P_*` successor at `Theta_2≥11/9`.

For the normalized fresh pair `a=(0,0), b=(1,0)`, the two transverse channels
lie on `q=0`, `q=1`, `q+r=0`, and `q+r=1`. Their only cross-intersections are
the already occupied endpoints; their intersections with the common-Q channel
are also only those endpoints. After deleting occupied cells the three response
channels are disjoint, so two Defender cells cannot touch all three.

For L10.6, the focal Q/R regions translated by 30 are disjoint, and the filler
`(2,1)` shares no lattice axis with the center or either support. Before
Attacker turn `s`, at most `2s` Defender-touched regions and `s-1` consumed
regions are excluded, strictly fewer than `3s`; `N≥3t` supplies an untouched
center. Each activation produces ten count-two labels. A legal empty Defender
cell lies in at most one of the two pencils and in at most four of its five
labels, because the sole common cell of all five is the occupied center.
Only the `t-1` intervening Defender turns can clean activated sites, for at most
`8(t-1)` kills. The lower bound and the stated high-`Phi` O1-only scope follow.

The §27.4 cascade detections also recompute correctly. In the same-axis
round-2 gadget, the local component's three residual pairs have hitting number
two after trigger `c`, while the separated count-three window contributes
one-trigger demand one at `d`; disjoint components give
`h_(1)+h_(2)≥3`. In the mixed gadget, the pre-count-two horizontal label and
three pre-count-three vertical labels intersect at `c`, and the already audited
pair `(c,d)` produces one-component hitting number three, hence `g_C≥3`.
Neither calculation uses the withdrawn global-pairing or superadditivity
inferences.

**Proposed repair:** none.

## Per-theorem verdicts

| Claim | Source status | Review verdict | Disposition |
|---|---|---|---|
| R4.1 exact `TEMPO` factorization | PROVEN | **CONFIRMED** | Full legal ordered-pair maximum; component decomposition and attainability both close |
| L10.1 two-trigger decomposition | PROVEN | **CONFIRMED** | Newly imminent labels come only from `L_23`; one trigger cannot affect two components |
| L10.2 `g_C≥h_C` | PROVEN | **CONFIRMED** | A second empty in the matured count-three label is legal and cannot lower hitting number |
| L10.3 Defender monotonicity | PROVEN | **CONFIRMED** | Deleted-label subfamilies and relevant-trigger legality cover all cases |
| R4.2 same-pair Defender form | PROVEN | **CONFIRMED** | The minimum is over the same actual ordered servicing pair |
| R4.6 conditional assembly | PROVEN | **CONFIRMED** | One deterministic `S_T`; its stated all-response `M` closure remains a hypothesis |
| L10.4 pure count-two ocean | PROVEN | **CONFIRMED** | `h=0`, `g≤2`, independent of raw mass |
| L10.5 fresh-channel separation | PROVEN | **CONFIRMED** | Three empty channels intersect only at occupied endpoints before deletion |
| `P_*` / repeated-launch regression | PROVEN | **CONFIRMED** | Safe at `Theta_2≥11/9`; axial chase leaves 24 count-one labels and no forced live-mass growth |
| L10.6 iterated O1 stock | PROVEN at O1 domain | **CONFIRMED** | Exact `2t+8` lower bound; correctly outside `Phi<1` |
| Same-axis/mixed cascade detection | PROVEN | **CONFIRMED** | Exact round-2 residual families give respectively `h_(1)+h_(2)≥3` and `g_C≥3` |
| L10.7 three-axis isolator | PROVEN | **CONFIRMED** | Exact profile `n_3=3`, `Phi=1/sqrt(3)` |
| R4.4 zero-grade-contact obstruction | PROVEN | **CONFIRMED** | Legal pair yields three disjoint residual grounds and `tau=3`; common-trigger defense remains available |
| R4.5 universal high-stock cascade | PROVEN | **CONFIRMED** | Exact `n_2=3,n_3=9`; forced win only at `Phi>1` |
| L10.8 count-four isolator | PROVEN | **CONFIRMED** | Sole alive count-four label with the stated residual pair |
| R4.3 remote-third-component necessity | PROVEN | **CONFIRMED** | Exact contradiction for root-covering, globally service-sufficient, remote-2-saturating state predicates |
| R4.3.1 three predicate classes | PROVEN at defined classes | **CONFIRMED** | Each defined class is remote-2-saturating; root/history/strategy-indexed classes escape |
| L10.9 count-one cleanup number eight | PROVEN | **CONFIRMED** | Independent line-incidence lower bound and explicit eight-cell cover |
| R4.7 unbounded dormant components | PROVEN | **CONFIRMED** | Exact `Phi=0` roots, adaptive untouched-site construction, and cleanup count |
| R4.7.1 uniform positive dormant floor | PROVEN | **CONFIRMED-WITH-ERRATA** | Inequality (28) is sound; title/follow-on prose must not cover selective-type or root-dependent bounds |
| R4.7.2 no grade-three subunit renewal | PROVEN at stated domain | **CONFIRMED** | Two legal remote extensions create exact `B_3=8sqrt(3)/9` outside the normative root domain |
| L10.10 axial stabilization | PROVEN | **CONFIRMED** | Exact post-stabilizer component has `h=1,g=2` |
| R4.8 two sealed-pencil returns | PROVEN at bounded class | **CONFIRMED** | Correct one-cycle existential response under the exact alive-family/disjoint-footprint premises |
| §24.1 mode-(c) ranking | achieved | **CONFIRMED-WITH-ERRATA** | R4.3 and the formal R4.7.1 result qualify; use the narrowed uniform-floor scope |
| §24.1 mode-(b) ranking | “narrowly bounded partial” | **CONFIRMED-WITH-ERRATA** | The inferred `S_T` one-cycle partial is valid; mode (b) itself remains unachieved and should be said explicitly |
| Candidate two-clock invariant `V_T` | OPEN | **CONFIRMED** | Exact candidate only; initialization and repair are not proved |
| `GAP-CASCADE-REACHABILITY` | OPEN | **CONFIRMED** | `P_hi` has `Phi>1`; no forced route from a strict root is supplied |
| `GAP-TEMPO-INITIALIZATION` | OPEN | **CONFIRMED** | `tau=0` free-pair K3 geometry remains unresolved |
| `GAP-TEMPO-REPAIR` | OPEN | **CONFIRMED** | No same-strategy all-response closure of `M≤2` is proved |
| GAP-REPLACEMENT-INVARIANT | OPEN | **CONFIRMED** | Candidate state quantities exist, but initialization and repair are both missing |
| GAP-RAW | OPEN | **CONFIRMED** | Neither a perpetual Defender strategy nor a universal Attacker win from a strict root is supplied |
| New machine verification | none | **CONFIRMED** | No new harness/Lean case and no new `[UNRUN]` or `VERIFIED` evidence |

## Overall verdict

**SOUND-WITH-ERRATA.** No formal R4 theorem is refuted or needs a mathematical
downgrade. R4.7.1 is **CONFIRMED-WITH-ERRATA** because the displayed theorem is
sound but the surrounding account-class claim is too broad. The §24.1
mode-(b) partial is **CONFIRMED-WITH-ERRATA**: its inferred `S_T` one-cycle
corollary is valid, but mode (b) itself remains unachieved and should be stated
as such. All other R4.1–R4.8 theorems are **CONFIRMED**.

The exact remaining obstacle is unchanged and **UNRESOLVED**: no proof shows
`M(P_0)≤2` for every strict root in the `tau=0` slice, and no one strategy is
shown to preserve `M≤2` after every legal response at all reached epochs.
Neither the selective-type/root-dependent account escapes identified in
Finding 12 nor the sealed-pencil lemma closes those obligations.
