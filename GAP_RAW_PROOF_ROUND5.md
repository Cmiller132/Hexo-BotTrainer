# GAP-RAW Proof Round 5 — sealed-return and Bellman research

**Worktree:** `hunt/gap-raw` at input HEAD
`12980bc8`  
**Date:** 2026-07-17  
**Disposition:** the six sealed-pencil extremes and every isolated axial return
are classified by hand.  The separated sealed-pencil class satisfies a
one-cycle `M<=2` bound, but the full proposed Bellman route does not close:
`S_T` does not create the transverse seal at its natural fresh-pair entrance,
and an exact shared-hub cascade refutes unrestricted statewise closure of
equation (22).  `GAP-TEMPO-REPAIR`, `GAP-TEMPO-INITIALIZATION`,
`GAP-CASCADE-REACHABILITY`, and GAP-RAW remain **OPEN**.

This document continues the definitions, equation numbering, and status
discipline of `GAP_RAW_PROOF_ROUND4.md`, including its post-review errata.  In
particular, the quantity under study is

`V_T(P): tau(P)<=2 and M(P)<=2`,

with `TEMPO` and `M` exactly as in round 4 equations (20)--(22).  No claim below
uses `Theta_2<1`, `B_3<1`, a dormant-stock sum, or a top-two-only *current*
demand.  Canonical J, global pairing, and NQ2 join consumption are not used.

No Cargo command, Lean build, harness, search program, or machine enumeration
was run.  Every new result is a hand proof.  Accordingly there are no new
`VERIFIED` or `[UNRUN]` claims.

## 37. Executive verdict

### 37.1 Answers to the four round questions

1. **Sealed returns [PROVEN].**  For one exact transverse-sealed adjacent
   pencil, the six surviving count-one residual arms are classified in
   Lemmas L11.1--L11.3.  Distinct transverse arms create no count-three label;
   two returns in one transverse arm create exactly one; a Q-axial return at
   depth `k=1,...,5` creates respectively `4,3,2,1,0` count-three Q-labels.
   Lemma L11.4 gives a two-cell transversal for every pair of Q-axial returns,
   consecutive or nonconsecutive.
2. **Local bound and global counterexample [PROVEN at their stated
   scopes].**  In an isolated pencil, both return placements can be cleaned
   with the same two Defender cells.  When the placements split between
   one-cycle-separated pencils, each placement creates at most one stabilizer
   action and the two actions leave `TEMPO<=2`.  Section 41 gives an exact
   shared-hub counterexample outside that scope: one hub placement fans out
   across three axes, mandatory current service consumes both Defender cells,
   and the surviving handoff has `TEMPO>=3`.
3. **Cascade reachability [OPEN].**  The hub stock itself has strict
   `Phi<1`, but at a Defender root the legal proactive move at the empty hub
   kills every focal label.  The forced-service embedding proving the Bellman
   failure has `Phi>1`.  Thus the cascade is real, but no strategy-independent
   route to it from a strict root is proved.
4. **Bellman boundary [PROVEN at the separated scope; strategy-wide repair
   OPEN].**  The one-cycle *value* bound from an isolated or fully
   one-cycle-separated sealed handoff survives:
   every Attacker pair returns an epoch with `M<=2`.  Literal sealed-profile
   closure does not follow, because the stabilized derivatives are not sealed
   pencils.  More decisively, the natural sealed profile is not reached by
   `S_T`: at the raw adjacent-pair epoch `S_T` strictly prefers a Q-axial
   cleanup of value zero to the transverse seal of value two.  Finally, the
   shared-hub construction supplies a legal response to `S_T`'s forced pair
   for which `V_T(P)` holds but `M(P')>2`.  None of these negative results
   refutes strategy-reachable repair from every strict root.

### 37.2 Exact round boundary

**GAP-TEMPO-INITIALIZATION [OPEN].**  The round-4 reduction to the `tau=0`
strict-root slice is unchanged.  The new strict hub and sealed-profile roots
are individually serviceable, but no uniform free-pair construction and no
exact counterexample root is supplied.

**GAP-TEMPO-REPAIR [OPEN].**  The separated sealed class satisfies one
value-step bound, not an inductive class closure; `S_T` does not enter it by the intended seal; and
equation (22) alone is not a global inductive invariant.  It remains possible
that the histories reached from strict roots avoid the hub escape under a
better or more deeply Bellman-aware servicing rule.

**GAP-CASCADE-REACHABILITY [OPEN].**  Determine whether forced service can
assemble the shared-hub handoff before proactive Defender occupation of its
hub.  Section 42 proves only the immediate strict-root exclusions.

## 38. Exact sealed-pencil return classification

### 38.1 Normal form and the six extremes

Put

`a=(0,0)`, `b=(1,0)`, `p=(0,1)`, and `p'=(1,-1)`.

The normalized sealed handoff has `A={a,b}`, `D={p,p'}`, Attacker at
`FirstStone`, and no external occupied support or alive label.  For an integer
start `s`, write

`W_s={(q,0): s<=q<=s+5}`.

**Lemma L11.1 (exact sealed inventory) [PROVEN].**  The five count-two labels
are `W_s` for `s=-4,-3,-2,-1,0`.  The six count-one residual arms are

```text
(-k,0),       (0,-k),       (-k,k),
(1+k,0),      (1,k),        (1+k,-k),       1<=k<=5.         (29)
```

The six residual sets in (29) are pairwise disjoint.  The involution

`rho(q,r)=(1-q,-r)`                                           (30)

exchanges the two endpoints, the two seals, and the left/right return cases.

*Proof.*  Before sealing, the adjacent pair has seven Q-axis labels and four
six-window transverse pencils, hence thirty-one alive labels.  `p` is the
empty intersection of the R-pencil through `a` and the QR-pencil through `b`;
it kills five labels in each.  Likewise `p'` kills five labels in each of the
other two transverse pencils.  Neither seal lies on the Q-axis.

The two endpoint-exclusive Q-windows survive at starts `-5` and `1`; their
residuals are the first and fourth arms of (29).  In each transverse pencil,
the seal is the cell immediately opposite the displayed arm, so it kills all
but the extreme six-window extending five steps along that arm.  This gives
the other four arms.  The coordinate display proves their pairwise
disjointness directly.  The remaining Q starts `-4,...,0` contain both
endpoints and are the five count-two labels.  There is no omitted alive label.
The formula (30) is immediate.  ∎

### 38.2 Which returns can make a high label

Call a post-return alive label *high* when its Attacker count is at least
three.  This is bookkeeping terminology local to this section, not a new
account.

**Lemma L11.2 (high-label reduction) [PROVEN].**  After any legal ordered
Attacker pair from one exact sealed handoff, every high label is of exactly one
of the following forms:

1. a Q-axis label which was one of the five pre-count-two `W_s` and received
   at least one Q-axial trigger; or
2. the unique pre-count-one extreme whose residual contains both triggers.

In particular, every newly imminent label is Q-axial and contains both
triggers.  Two non-Q returns in distinct arms create no high label, while two
returns in the same non-Q arm create exactly its one count-three extreme.

*Proof.*  A virgin or pre-count-one label has final count at most two or three,
respectively.  A pre-count-two label can become high after one trigger and
imminent only after both.  The only pre-count-two labels are the five Q-labels
of L11.1, so a trigger affecting one of them is Q-axial.  The residual arms of
all pre-count-one extremes are pairwise disjoint, so two triggers lie in at
most one such residual.  In a non-Q pencil, the seal on the opposite side of
the endpoint kills every endpoint window except the displayed extreme; hence
two same-arm returns create exactly that one count-three label.  ∎

This also covers a placement which is virgin relative to the sealed family.
Such a placement can make a new count-one label, and together the pair can make
new count-two labels, but neither operation adds a high label beyond L11.2.
Below, a *Q-axial return* means a cell in the first or fourth residual arm of
(29).  A Q-row cell outside those arms is virgin relative to the sealed family
and belongs to the preceding case.

### 38.3 Zero or one Q-axial return

**Lemma L11.3 (single-axis depth table) [PROVEN].**  Suppose exactly one return
is Q-axial.  On the positive side write it as `x=1+k`, `1<=k<=5`.  It promotes
exactly `5-k` of the five common Q-labels to count three.  The complete table is

| return `x` | count-three starts | full cleanup | one-cell split stabilizer |
|---:|---|---|---|
| `2` | `-3,-2,-1,0` | `{-1,3}` | `-1` |
| `3` | `-2,-1,0` | `{2}` | `2` |
| `4` | `-1,0` | `{2}` | `2` |
| `5` | `0` | `{2}` | `2` |
| `6` | none | none | none |

The negative table is the image under `rho`.  Every listed cleanup cell is
empty and legal.  A full cleanup kills every count-three label, so the handoff
has a pure count-two graded tier and `TEMPO<=2` by L10.4.

For the adjacent split return `x=2`, the one-cell stabilizer `D@(-1,0)` leaves
exactly the following non-pure part of its Q-component:

```text
count three:  W_0,  residual {3,4,5};
count two:    W_1,  residual {3,4,5,6}.                      (31)
```

This nested component has exact `h=g=1`.  Its reflection gives the adjacent
negative case.  For depths `k=2,3,4`, the one-cell entry in the last column
kills every high label; depth five creates none.

*Proof.*  A common label `W_s`, `-4<=s<=0`, contains `1+k` exactly when
`s>=k-4`, giving `5-k` starts.  At `x=2`, `-1` hits starts `-3,-2,-1` and `3`
hits starts `-2,-1,0`.  At `x=3,4,5`, the cell `2` lies in every high interval.
The stated cells lie in alive high windows and are within distance at most five
of an Attacker stone, so L6_2 gives legality; they remain sequentially legal in
either order.  Reflection proves the other side.

After `D@(-1,0)` in the adjacent case, all other old graded Q-labels through
that cell are killed, leaving (31).  A singleton trigger can mature only the
high label, for demand one.  If two triggers also mature the low label, the
post-trigger residual of `W_0` is a nonempty subset of that of `W_1`; one cell
hits both.  Conversely a trigger in `{3,4,5}` realizes demand one.  Hence
`h=g=1`.  ∎

If the second Attacker placement is non-Q, L11.2 says it adds no further high
label unless both placements were in the same non-Q arm, which is a different
case.  Therefore the full-cleanup column proves every one-Q plus non-Q or
virgin return safe for one value step.  With no Q return, either the graded
tier is pure count two or one same-arm count-three label is killed by any one
of its three residual cells; a legal filler supplies the second Defender move.

### 38.4 Every two-Q return, including nonconsecutive returns

**Lemma L11.4 (rank-triple transversal) [PROVEN].**  Let the two Attacker
returns both be Q-axial, and sort the four Attacker coordinates after the pair
as

`s_1<s_2<s_3<s_4`.

There is a set of at most two empty Q-cells meeting every post-return Q-window
of count at least three.  Every used cell is legal.  Consequently Defender can
service every current imminent label, delete every count-three label, use a
legal filler if fewer than two cells are needed, and hand over a pure-count-two
graded family with `TEMPO<=2`.

*Proof.*  A length-six interval containing at least three of the four sorted
stones contains the left rank triple

`T_L={s_1,s_2,s_3}`

or the right rank triple

`T_R={s_2,s_3,s_4}`.

Ignore either triple when its span exceeds five, since then no length-six
interval contains it.  For a feasible nonconsecutive triple `u<v<w`, any empty
integer gap strictly between `u` and `w` belongs to every interval containing
the triple.  For a consecutive triple `{u,u+1,u+2}`, the two cells
`{u-1,u+3}` hit its four possible length-six intervals.

These covers combine without a third cell.  If neither feasible rank triple is
consecutive, choose one internal gap for each feasible triple and use a filler
if fewer than two cells result.  If `T_L` is consecutive and
`s_4>=s_3+2`, use `{s_1-1,s_3+1}`: the first and second cells cover the left
triple, while `s_3+1` is an internal gap of the right triple.  If all four
stones are consecutive, use `{s_1-1,s_4+1}`: the five possible high-window
starts `s_1-3,...,s_1+1` are hit by the first cell on the first three starts
and by the second on the last two.  The case in which only `T_R` is
consecutive is the reflection.  Each selected point is outside the four
Attacker coordinates.  Any selected point actually needed by the surviving
family lies in an alive high window and is therefore legal by L6_2; an unused
point is replaced by the standard legal filler.

Every high label is Q-axial by L11.2, so this transversal deletes the complete
high family, not merely a focal subfamily.  It also hits every count-four
member of `I`.  L10.4 applies to the surviving pure-count-two tier.  ∎

Lemmas L11.1--L11.4 are the requested exhaustive classification of all six
count-one extremes and all nonconsecutive axial returns for one exact sealed
pencil.  No first-return multi-axis cascade survives this classification.

## 39. One-cycle Bellman bound on the sealed class

### 39.1 One isolated pencil

**Theorem R5.1 (isolated sealed one-cycle value bound) [PROVEN].**  Let `Q`
be one exact normalized sealed handoff from Section 38, with no other alive
label.  For every legal ordered Attacker pair `b`, the returned nonterminal
Defender epoch

`P_b=Q+A@b_1+A@b_2`

satisfies `M(P_b)<=2`.  Therefore `S_T`'s actual minimizing servicing pair at
`P_b` also hands over `TEMPO<=2`.

*Proof.*  The start has only two Attacker stones, so the pair cannot complete a
six and `P_b` exists.  If both relevant returns are Q-axial, L11.4 supplies a
legal pair which hits every member of `I(P_b)`, deletes every high label, and
leaves `TEMPO<=2`.  If exactly one is Q-axial, use the full-cleanup column of
L11.3; the other return adds no high label by L11.2.  If neither is Q-axial,
L11.2 leaves either a pure-count-two tier or one count-three extreme, killed by
one legal residual cell plus a legal filler.

Thus in every case the displayed Defender pair belongs to `Serv(P_b)` and its
actual handoff has `TEMPO<=2`.  Definition (21) gives `M(P_b)<=2`.  Since `S_T`
minimizes the same quantity over the same servicing-pair set, its actual pair
has no larger value.  Every nonfiller cell of a constructed reply is legal by
L6_2 because it lies in an Attacker-alive high window.  Any filler is legal by
L1.2's finite max-q construction.  After the first nonfiller is placed, the
second remains within distance five of the same Attacker stock.  Hence the
proof respects the ordered 2:2 cadence and the radius-eight growth rule.  ∎

This is a Bellman *value* statement for one Attacker-Defender cycle.  It does
not say that the resulting alive family is another exact two-stone sealed
pencil, so it is not yet an inductive shape theorem.

### 39.2 Split returns between separated pencils

For a concrete sufficient separation condition, let `O_i` be the complete
occupied support of a translated or reflected exact sealed pencil, and define
support distance by

`d(O_i,O_j)=min{d(x,y):x in O_i, y in O_j}`.

During one
Attacker turn, a placement supported only from `O_i` lies in the closed
radius-16 neighborhood of `O_i`: the first placement reaches radius eight and
the second may extend another eight.  Every six-window touched by those stones,
and every local service/stabilizer cell used below, lies in the radius-21
neighborhood.  Call a union of pencils **fully one-cycle-separated** when it
has no other occupied support and these radius-21 neighborhoods are pairwise
disjoint.  Pairwise support distance at least 43 is a simple sufficient
condition.  (Post-review erratum, R-G3-REV Finding 7: radius 21 is tight for
this conservative all-touched-alive-window envelope — from a support cell `o`,
play `o+8u` then `o+16u`, and a count-one window through the second stone
extends to `o+21u` — but it is NOT claimed sharp for the `M<=2` value
conclusion itself; the minimum support separation needed for the value theorem
is unresolved.)

This deliberately strengthens R4.8's enlarged-footprint premise, which was
tailored only to the two adjacent extensions audited there.  It excludes a
new count-two bridge or physical-window intersection between return hulls.

**Lemma L11.5 (one-cell split stabilizers) [PROVEN].**  Suppose a legal
Attacker pair places at most one relevant trigger in a given sealed pencil.
A non-Q trigger creates no high label.  A Q trigger at positive depth
`k=2,3,4` has every high label killed by `D@(2,0)`; depth five creates none.
At depth one, `D@(-1,0)` leaves only the nested component (31), with `h=g=1`.
The negative cases are their images under `rho`.

*Proof.*  This is precisely the last column and the nested-residual conclusion
of L11.3.  Each action uses one legal cell in the activated pencil.  ∎

**Theorem R5.2 (separated sealed one-cycle value bound) [PROVEN for the stated
history class].**  Let `Q` be an Attacker handoff whose entire alive family is
an exact fully one-cycle-separated union of finitely many sealed profiles.
Then every legal ordered Attacker pair returns an epoch `P_b` with
`M(P_b)<=2`.  Consequently the actual `S_T` reply at `P_b` hands over
`TEMPO<=2`.

*Proof.*  Because there is no outside occupied support and the radius-21
neighborhoods are disjoint, the first placement is assigned to the unique
`O_i` supplying its radius-eight legality; a second placement supported by the
first follows the same legality chain, while one supported by another old
component is assigned there.  Separation makes these assignments unique even
for a placement which is virgin relative to its hull's old alive family.  If
both placements lie in one hull, use R5.1 there; the cleanup
leaves a pure-count-two graded tier.  If they lie in different hulls, use at
most one L11.5 stabilizer in each affected pencil.  An
activated component is then either pure count two, with `h=0,g<=2` by L10.4,
or the adjacent nested derivative (31), with `h=g=1`.  Untouched pencils are
pure count two.  A virgin pair or an irrelevant trigger adds only pure
count-two stock.

Full one-cycle separation prevents labels in different hulls from joining one
component.  Hence every component has `g<=2`, and the only positive singleton
demands are contributed by at most the two adjacent derivatives, each of value
one.  Equation (20) gives

`TEMPO<=max(2,1+1)=2`.                                      (32)

No current imminent can be created by a split pair, since each pre-count-two
label receives at most one trigger; in the one-pencil branch R5.1 explicitly
services all current imminents.  Thus the same constructed pair is in
`Serv(P_b)` and witnesses `M(P_b)<=2`.  The final `S_T` conclusion again uses
its minimization over that exact set.  Every nonfiller response cell is legal
by L6_2; when fewer than two stabilizers are needed, use the global legal
filler construction of L1.2.  A filler can lie outside a named return hull, but
it only deletes labels, so L10.3 shows that it cannot increase `TEMPO` or join
two components.  The nonfillers were already legal before the reply, so their
displayed order is sequentially legal.  ∎

R5.2 broadens R4.8's return classification from the two adjacent axial
extensions to all six extreme arms and all axial depths, but assumes stronger
full-return-hull separation.  The two theorem scopes are therefore not nested.
Cross-hull coupling and the next turn from the nested derivatives remain
**OPEN**.

## 40. The transverse sealed class is not entered by `S_T`

### 40.1 Strict value comparison at the raw adjacent-pair epoch

Let `P_raw` be the isolated Defender epoch immediately after Attacker has made
the fresh adjacent pair `A={(0,0),(1,0)}`, before either transverse seal is
played.  Its graded family consists of the five common Q-windows at starts
`-4,...,0`, all at count two.

**Theorem R5.3 (natural `S_T` sealed-entry exclusion) [PROVEN].**  At `P_raw`,
the actual `S_T` pair is

`D@(-4,0), D@(2,0)`,                                       (33)

and its handoff has `TEMPO=0`.  The transverse pair

`D@(0,1), D@(1,-1)`                                        (34)

has handoff `TEMPO=2`.  Therefore `S_T` strictly rejects (34); changing only
its lexicographic tie-breaker cannot make the round-4 sealed profile its
natural fresh-pair response.

*Proof.*  The first cell of (33) kills the common window at start `-4`; the
second lies in all four starts `-3,-2,-1,0`.  Thus no count-two/count-three
label survives, `L_23` is empty, and equation (20) gives `TEMPO=0`.

Conversely, (34) meets no Q-label, so all five count-two labels survive.  The
future triggers `{-1,2}` make the starts `-3,-2,-1` imminent with residuals

`{-3,-2}`, `{-2,3}`, and `{3,4}`.                          (35)

The extreme residuals are disjoint, while `{-2,3}` hits the family, so (35)
has hitting number two.  Hence the sealed handoff has `TEMPO>=2`; L10.4 gives
the reverse bound and therefore exact value two.

Any value-zero reply must delete all five old count-two labels, because one
surviving count-two label can be made count four by choosing two of its four
residual cells and hence gives positive `g`.  Both deleting cells must lie on
the Q-row.  Axial lexicographic order is by `q` and then `r`.  A first cell
earlier than `(-4,0)` lies in none of the five Q-labels; one remaining cell
cannot cover all five because their total intersection is the occupied set
`{(0,0),(1,0)}`.  Thus the earliest possible first cell is `(-4,0)`.  After it,
the four remaining common windows have only the empty common cell `(2,0)`.
This proves the exact ordered pair (33) under the `S_T` definition.  All four
displayed placements are legal in Attacker-alive windows.  ∎

### 40.2 The exclusion occurs on a strict-root `S_T` history

**Corollary R5.3.1 [PROVEN].**  The comparison in R5.3 occurs on an
`S_T`-consistent history from a normative root with `Phi=0`.

*Proof.*  Use the round-4 R4.7 anchor root with three pairwise-disjoint launch
unions `V_0,V_1,V_2`, no Attacker stones, and Defender at `FirstStone`.  It is
finite and nonempty because of its distance-eight Defender anchors, and its
potential is zero.  `S_T`'s initial two placements meet at most two private
launch unions.  Attacker chooses the untouched third and legally plays its
anchored adjacent pair: the first endpoint is at distance eight from its
anchor, and the second is adjacent after the first placement.

At the returned Defender epoch, no Defender cell lies in any of the thirty-one
windows through that pair.  The only graded labels are therefore the five
common Q-labels of `P_raw`.  R5.3 applies verbatim at the translated site, so
the actual `S_T` response is the axial value-zero cleanup, not the transverse
seal.  The history is a Defender pair followed by an Attacker pair and thus
uses the ordinary 2:2 cadence.  ∎

R5.3.1 refutes the intended *natural entrance* into the sealed strategy class,
not the possibility that some different forced-service history happens to
produce the same alive profile.  Such an alternative entrance is **OPEN**.

## 41. The exact escaping return class

The sealed classification proves that the first isolated return has only one
axis of high stock.  It does not prove the proposed global statement that one
placement can never fan out across several already coupled promotion centers.
The following strict-potential handoff gives the exact obstruction.

### 41.1 A five-low shared-hub isolator

Let the three axis vectors be

`u=(1,0)`, `v=(0,1)`, and `w=(1,-1)`.

For an axis vector `z`, write `[i,j]_z={t z:i<=t<=j}`.  Put

```text
A_H={-4u,-3u,4u,5u,-4v,-3v,4v,5v,2w,3w}.                 (36)
```

Protect the five count-two windows

```text
U^-=[-4,1]_u,   U^+=[0,5]_u,
V^-=[-4,1]_v,   V^+=[0,5]_v,   W=[-2,3]_w,                (37)
```

and let `X` be their physical union.  Four additional Attacker-touched windows
are unavoidably contained in `X`:

```text
U_L=[-3,2]_u, U_R=[-1,4]_u,
V_L=[-3,2]_v, V_R=[-1,4]_v.                               (38)
```

Each member of (38) has count one.  Let `T_H` be the finite family of all
length-six windows meeting `A_H`, let `F_H` be the nine windows in (37)--(38),
and define

`D_H=union_{Y in T_H\F_H} (Y\X)`.                          (39)

**Lemma L11.6 (strict shared-hub profile) [PROVEN].**  Every set `Y\X` used in
(39) is nonempty.  At the position with occupancy `(A_H,D_H)`, the exact alive
family is `F_H`: the five windows in (37) have count two, the four in (38) have
count one, and there is no other alive label.  Thus

`Phi_H=5/9+4/(9 sqrt(3))<1`, `I=empty`.                    (40)

At the Attacker-`FirstStone` handoff `Q_H` with this occupancy,
`TEMPO(Q_H)=2`.

*Proof.*  On the central u-line, `X` is the ten-cell interval `[-4,5]_u`.
The length-six subintervals contained in it start at `-4,-3,-2,-1,0`.
Against the u-axis stones in (36), their counts are respectively `2,1,0,1,2`.
The v-line is identical.  On the central w-line, the only six-cell subinterval
of `X` is `W`, at count two.

A noncentral line is parallel to one central axis and meets each of the other
two central axes in at most one cell.  Hence it contains at most two cells of
`X` and cannot contain a length-six window wholly inside `X`.  Therefore every
Attacker-touched window outside `F_H` has a cell outside `X`, proving the first assertion.  Definition
(39) blocks all such windows and places no Defender stone in `X`; (37)--(38)
are exactly the alive family.  Their displayed counts give (40), and strictness
is equivalent to `4<4 sqrt(3)`.

The `L_23` tier of `Q_H` consists only of the five count-two labels (37), so
L10.4 gives `TEMPO<=2`.  The pair analyzed in L11.7 below creates current
hitting number two, giving the reverse inequality by R4.1.  ∎

### 41.2 One hub placement survives mandatory current service

Put `c=0` and `d=u`.

**Lemma L11.7 (hub-fanout cascade) [PROVEN].**  The ordered Attacker pair
`(c,d)` is legal from `Q_H` and returns a nonterminal Defender epoch `P_H'`
with `tau(P_H')=2`.  Every servicing Defender pair at `P_H'` hands over a
position `R` with

`TEMPO(R)>=3`.                                               (41)

Consequently `M(P_H')>=3`.

*Proof.*  The cell `c` lies in every focal window and is empty by (39), so it
is legal within distance five of old Attacker stock.  The cell `d` is empty,
adjacent to `c`, and lies in both u-windows.  After the pair, `U^-` and `U^+`
have count four with residuals

`{-2u,-u}` and `{2u,3u}`.                                  (42)

They are disjoint.  The two u-axis count-one intermediates in (38) have count
three; `V^-`, `V^+`, and `W` also have count three; the v-axis intermediates
have count two.  A previously blocked label retains its Defender stone, and a
label using only the two new cells has count at most two.  Thus the only
imminent labels are the two in (42), and their hitting number is exactly two.
No six is formed.

Every servicing pair must use one cell of each residual in (42).  Both cells
therefore lie on the nonzero u-axis.  They also kill the two u-axis
intermediates, but a nonzero u-cell lies in none of `V^-`, `V^+`, or `W`, whose
central axes meet the u-axis only at the occupied hub `c`.  Those three
count-three labels survive every service.

At the resulting handoff `R`, play the legal prospective triggers `v` and
`w`.  The two v-labels and the w-label become count four with residuals

`{-2v,-v}`, `{2v,3v}`, and `{-2w,-w}`.                       (43)

The three grounds are pairwise disjoint.  Hence this future pair creates
hitting number three.  R4.1 gives (41), and definition (21) gives
`M(P_H')>=3` for every possible servicing choice.  The proof evaluates one
complete Attacker pair, the mandatory Defender pair, and the prospective next
Attacker pair in the ordinary 2:2 cadence; every used cell is an empty of an
alive window and therefore radius-eight legal.  ∎

The load-bearing event is the *first* placement `A@c`: it promotes all five
count-two hub labels across three axes.  The second placement `A@d` makes the
two u-labels current service obligations.  Mandatory service consumes both
Defender cells on that axis, leaving the v/v/w fanout in (43).  Thus the
proposed global statement that one placement creates at most two stabilizer
demands after current service is false.  The exact escaping class is the
**shared-hub fanout after saturated two-cell service**.

### 41.3 A forced-service Bellman embedding

Define the complete protected supports (post-review erratum, R-G3-REV
Finding 2): `G_i = A_i union D_i union Y_i` for each gadget and
`H = A_H union D_H union X` for the hub.  Choose two translations `b_1,b_2`
such that the three sets `G_1`, `G_2`, `H` have pairwise set distance at least
six.  This expressly includes the service cells `z_i` (empty cells of `Y_i`)
and every protected window cell, so no service cell can lie in or kill a hub
window.  For `i=1,2`, put

```text
Y_i=b_i+[0,5]_u,
A_i=b_i+{0,u,2u,3u,4u},
z_i=b_i+5u.                                                 (44)
```

Let `T_i` be all windows meeting `A_i` and set

`D_i=union_{Z in T_i\{Y_i}} (Z\Y_i)`.                       (45)

Every set in (45) is nonempty: a different interval on the central u-line
leaves `Y_i`, and an off-axis line meets `Y_i` in at most one cell.  Thus
`Y_i` is the sole alive label of this translated gadget, at count five with
singleton residual `{z_i}`.

Let `P^dagger` be the union of `(A_H,D_H)` and the two gadgets (44)--(45), with
Defender at `FirstStone`.

**Theorem R5.4 (unrestricted `V_T` Bellman closure fails) [PROVEN].**  The
position `P^dagger` satisfies

`tau(P^dagger)=2` and `M(P^dagger)=2`.                      (46)

Every servicing strategy, including `S_T`, is forced to occupy `{z_1,z_2}`.
The legal Attacker response `(c,d)` then returns a demand-equivalent union
whose hub restriction is the epoch `P_H'` of L11.7 and whose `M` is at least
three.  Therefore the unrestricted statewise implication

```text
V_T(P)  =>  every response to the actual S_T pair has M(P')<=2              (47)
```

is false.

*Proof.*  The exact imminent family at `P^dagger` is `{Y_1,Y_2}` with the two
distinct singleton residuals in (44).  Hence its hitting number is two and
every servicing pair has occupancy `{z_1,z_2}`, in one of the two orders.
After that pair, both remote labels are dead.  Their old stones and blockers
remain in the full occupancy, but separation leaves no remote alive label;
the complete alive-family and `L_23` data are exactly those of `Q_H`.
Lemma L11.6 therefore gives `TEMPO=2`, establishing (46) and forcing this same
demand-equivalent handoff under `S_T`.

L11.7 applies on the hub restriction; the dead remote supports cannot add or
delete a hub label.  The successor's imminent family is still exactly the two
hub labels (42), so every global servicing pair consumes both cells there, and
the local future pair `(v,w)` remains legal.  Hence the full successor has
`M>=3`.  Thus
`V_T(P^dagger)` holds while its prescribed one-cycle successor violates the
second coordinate of (22).  ∎

The root potential of `P^dagger` is

`Phi_H+2/sqrt(3)>1`.                                       (48)

Accordingly R5.4 refutes a *global statewise* induction on equation (22), not
`GAP-TEMPO-REPAIR` on histories reachable from strict roots.  R4.6 never
claimed that every abstract `V_T` state was reachable under `S_T`.

## 42. Cascade reachability from a strict root

### 42.1 Immediate proactive defusal

Let `P_H` have the same occupancy `(A_H,D_H)` as L11.6, but put Defender at
`FirstStone`.

**Lemma L11.8 (the strict hub is immediately defusable) [PROVEN].**  The
position `P_H` is a normative strict root in the `tau=0` slice and satisfies
`M(P_H)=0`.  In particular, the L11.7 cascade is not forced immediately from
this root against every Defender reply.

*Proof.*  Finiteness, nonemptiness, nonterminality, and strict `Phi<1` are in
L11.6.  Every one of the nine exact alive labels (37)--(38) contains the empty
hub `c=0`.  The move `D@c` is legal by L6_2 and kills all of them at once.  A
legal filler completes the Defender turn.  The handoff then has `L_23=empty`
and `TEMPO=0`, so definition (21) gives `M(P_H)=0`.  ∎

The strict potential budget independently blocks the simplest saturated-
service embedding.  While all nine hub labels remain alive,

`Phi_H+1/3>1`.                                             (49)

Thus no *additional* imminent label can coexist with the exact hub family at a
strict root.  Promoting one of the five hub count-two labels to count four
while retaining the other eight is also impossible under the threshold,
because replacing its `1/9` weight by at least `1/3` gives

`Phi_H+2/9>1`.                                             (50)

Equations (49)--(50) do not exclude a history which first deletes, damages, or
rebuilds hub arms.  They only show why attaching a ready-made mandatory service
gadget to the exact strict hub cannot prove reachability.

### 42.2 What is and is not decided

**First-exact-sealed-return exclusion [PROVEN].**  L11.2 shows that current
service created from an exact sealed profile is
Q-axial.  If no current service is created, two non-Q placements can make at
most one count-three extreme.  Hence the three-axis fanout of L11.7 cannot be
created during the first isolated sealed return.  R5.1 exhausts that complete
first-return universe and proves its one-cycle value bound.

**Forced high-potential route [PROVEN].**  R5.4 forces the cascade from a
`V_T` state, but (48) puts that state outside the normative root threshold.
This establishes the Bellman obstruction and nothing stronger.

**GAP-CASCADE-REACHABILITY [OPEN].**  No proof in this round shows that a
strict-root Attacker can assemble the shared hub on a strategy-consistent
handoff while both Defender cells are causally committed elsewhere.  No proof
shows that Defender can pre-empt every such assembly either.  Therefore the
hub cascade is not a GAP-RAW counterexample.

## 43. Secondary initialization audit

The primary work did not resolve early enough to justify a broad new K3 claim.
The exact initialization boundary remains the round-4 one:

- `tau=2` strict roots have `M<=2` by K1;
- `tau=1` strict roots have `M<=2` by K2; and
- the general `tau=0` strict-root slice is **OPEN**.

**Lemma L11.9 (two strict `tau=0` instances) [PROVEN].**  The strict shared hub
`P_H` has `M=0`.  The same sealed occupancy as Section 38, viewed instead as a
Defender-`FirstStone` root, has

`Phi=5/9+2/(3 sqrt(3))<1` and `M<=2`.                       (51)

*Proof.*  The hub statement is L11.8.  L11.1 gives the sealed profile
`n_1=6,n_2=5`, hence (51).  Its `L_23` tier is pure count two, so L10.4 gives
`TEMPO<=2` before any additional Defender placement.  Every legal Defender
augmentation preserves that upper bound by L10.3; service is vacuous because
`I=empty`.  Therefore the minimum in (21) is at most two.  ∎

The raw fresh adjacent-pair epoch separately has `M=0` by R5.3, even though
that minimizing action is strategically hostile to the intended transverse-
seal shape.  Its raw potential is above one, so this reached accumulated-stock
state is not a third initialization result.

These examples emphasize that initialization of the scalar value and selection
of a useful repair class are different obligations.  They provide neither a
uniform servicing-pair construction nor an exact `Phi<1` counterexample root.
Thus `GAP-TEMPO-INITIALIZATION` remains **OPEN**.

## 44. Authoritative round-5 status ledger

| Claim | Status | Exact basis / scope |
|---|---|---|
| GAP-RAW | **OPEN** | No perpetual Defender strategy and no universal strict-root Attacker win |
| L11.1 exact six-extreme inventory | **PROVEN** | Complete 31-label sealed-pencil audit, Section 38.1 |
| L11.2 exhaustive high-label reduction | **PROVEN** | Pre-count classification and disjoint extreme residuals, Section 38.2 |
| L11.3 single-axis depth table and nested derivative | **PROVEN** | Exact Q-start containment and residual nesting, Section 38.3 |
| L11.4 every two-Q/nonconsecutive return | **PROVEN** | Rank-triple transversal, Section 38.4 |
| R5.1 isolated sealed one-cycle value bound | **PROVEN** | Exhaustive case split using L11.2--L11.4 |
| L11.5 split stabilizers | **PROVEN** | One placement per pencil, exact depth table |
| R5.2 separated sealed one-cycle value bound | **PROVEN at the stated history class** | No outside support; disjoint closed radius-21 neighborhoods of complete pencil supports; equation (20) |
| Cross-hull and iterative sealed closure | **OPEN** | The R5.2 separation premise is load-bearing; derivative next turns unclassified |
| R5.3/R5.3.1 natural `S_T` sealed-entry exclusion | **PROVEN** | Strict value `0<2`, including an `S_T`-reached `Phi=0` history |
| Alternative forced-service entrance to the transverse seal | **OPEN** | R5.3 excludes only the natural fresh-pair response |
| L11.6 strict five-low shared-hub profile | **PROVEN** | Exact blocker comprehension and `Phi_H<1` |
| L11.7 shared-hub fanout cascade | **PROVEN** | Two disjoint current residuals force u-service; future v/w pair gives three disjoint residuals |
| R5.4 failure of unrestricted statewise `V_T` closure | **PROVEN** | Two unique count-five service demands force `TEMPO=2 -> M>=3` |
| L11.8 immediate strict-hub defusal | **PROVEN** | `D@0` kills all nine labels and gives `M=0` |
| L11.9 strict hub/sealed initialization instances | **PROVEN** | Direct hub pre-emption; sealed `n_1=6,n_2=5` plus L10.3--L10.4 |
| First isolated sealed return cannot source the hub escape | **PROVEN** | L11.2 and R5.1 |
| GAP-CASCADE-REACHABILITY | **OPEN** | Forced embedding has `Phi>1`; exact strict hub is proactively defusable |
| GAP-HUB-FANOUT-REACHABILITY | **OPEN** | Post-review addition (R-G3-REV Finding 5): the narrower obstruction-specific subproblem of GAP-CASCADE-REACHABILITY defined in Section 46 — NOT a replacement for the broader cascade, initialization, or repair obligations |
| GAP-TEMPO-INITIALIZATION | **OPEN** | General `tau=0` free-pair geometry remains |
| GAP-TEMPO-REPAIR | **OPEN** | Local one-cycle success does not give strategy-reachable induction |
| New machine verification | **none** | No Cargo, Lean, harness, search, `[UNRUN]`, or `VERIFIED` claim |

No round-2, round-3, or review-confirmed round-4 theorem is downgraded.
R5.4 supplies a counterexample to a stronger global statewise closure premise,
not to R4.6's explicitly strategy-reachable conditional theorem.

## 45. Hostile-review attack surface

1. **Exact sealed inventory.**  The four transverse seals kill five labels per
   pencil incidence, not five labels total.  The six survivors in (29) are
   count one and their residual sets, rather than merely their axes, must be
   pairwise disjoint.
2. **High-label exhaustion.**  A virgin label reaches count at most two, and a
   pre-count-one label reaches at most three.  This is why all current
   imminents in the first sealed return are Q-axial.
3. **Rank-triple cover.**  A chosen internal gap must be empty; sorting the four
   Attacker coordinates ensures that the fourth stone is outside the relevant
   rank-triple interval.  Infeasible triples of span above five are omitted.
4. **Legality and cadence.**  Every nonfiller Defender cell lies in a current
   alive high window.  Every displayed prospective Attacker trigger lies in a
   surviving alive label.  The proofs use ordered Attacker and Defender pairs,
   not simultaneous placements.
5. **Separation scope.**  R5.2 uses full radius-21 one-cycle hulls, stronger than
   R4.8's adjacent-return footprint.  It says nothing about cross-hull bridges.
6. **`S_T` binding.**  The entrance failure is strict, `0<2`; it is not a bad
   lexicographic tie.  The exact lex pair (33) is proved only after minimizing
   `TEMPO` first, exactly as `S_T` is defined.
7. **Hub blocker comprehension.**  Definition (39) is sound only because every
   undesired touched window is proved to leave `X`.  The central-line count
   sequence `2,1,0,1,2` is the complete source of the four count-one labels.
8. **Mandatory service.**  The residuals (42) are disjoint and contain no cell
   on either the v- or w-axis.  Hence both service cells are genuinely consumed on
   the u-axis and cannot pre-empt (43).
9. **Claim boundary.**  `P_H` has strict potential but is defusable; the forced
   `P^dagger` has potential above one.  Combining those facts into a strict-root
   Attacker win would be invalid.
10. **No dead-account substitution.**  The hub has only five count-two labels
    and four count-one labels; its danger comes from their common trigger, not
    from a forbidden `Theta_2`, `B_3`, or dormant-component threshold.
11. **No machine status.**  Every coordinate, count, residual, and hitting set
    above is a hand proof and is labeled `PROVEN`, never `VERIFIED`.

## 46. Exact resume point and provenance

**GAP-HUB-FANOUT-REACHABILITY [OPEN].**  Starting from a strict `Phi<1`,
`tau=0` root under one named Defender strategy, prove one of the following:

1. repeated forced service can assemble the five count-two shared-hub stock of
   L11.6 on an Attacker handoff before Defender occupies its hub; or
2. a causal pre-emption invariant prevents every such assembly while retaining
   `M<=2` after all responses.

For the actual round-4 policy `S_T`, the first concrete state to analyze is no
longer the transverse seal.  It is the axial-cleanup handoff produced by (33),
with its twenty-four surviving off-axis count-one labels.  A successful repair
may instead replace `S_T`'s one-step `TEMPO` objective by a genuinely
Bellman-aware rule, but that rule and its common-strategy closure must be stated
and proved rather than inferred from the sealed existential response.

**Scope correction (post-review erratum, R-G3-REV Finding 4, MAJOR).**  The
original text called this "the named round-5 resume point ... contain[ing]
both remaining exact questions."  That characterization is DOWNGRADED:
hub-fanout reachability is a sharp obstruction-specific subproblem of
`GAP-CASCADE-REACHABILITY`, but it is neither necessary nor sufficient for all
of `GAP-TEMPO-INITIALIZATION`, `GAP-TEMPO-REPAIR`, or GAP-RAW, because it
conflates three differently quantified problems that must be stated
separately:

1. **Fixed-`S_T` hub reachability** — reaching the hub against the one fixed
   policy `S_T` proves only that this policy fails; it can refute `S_T` but
   nothing more.
2. **Strategy-independent hub forcing** — a GAP-RAW counterexample route
   requires the much stronger `exists P_0, for every Defender strategy S,
   there exists an S-consistent Attacker continuation reaching a forced-loss
   state` (`∃P_0 ∀S ∃α`).
3. **The positive universal initialization/repair obligation** — one named
   strategy preserving `M<=2` after every response (`∀P_0 ∃S ∀α`) must also
   exclude every OTHER escape class: other `M>2` fanouts, cross-hull
   interactions, next turns from nested derivatives, and all responses from
   the R5.3.1 axial-cleanup handoff (acknowledged OPEN in Sections 37.2, 39.2,
   40.2, and 43).

Hub reachability remains the most concrete next attack; it may decide `S_T`.
The two questions in the original phrasing (can the axial-chase dormant stock
build the hub escape; if not, what one-strategy invariant proves exclusion)
are real, but they are entries in the broader obligation set above, not its
entirety.

**Input commit:** `12980bc8` on branch `hunt/gap-raw`.  This authoring pass
created no commit.  **Reviewed/output artifact:** `d93d5768` (the commit the
R-G3-REV hostile review examined; errata from that review are folded in this
file).

An unrelated concurrent strategy-stealing job advanced the shared branch HEAD
during authoring.  A final read-only diff from `12980bc8` to the observed HEAD
showed no change to the required GAP-RAW corpus or the GAP-RAW harness.  No
concurrent file is used as evidence here.

**Required corpus read first, in order, and in full:**

1. `GAP_RAW_PROOF_ROUND2.md`;
2. `GAP_RAW_PROOF_ROUND3.md`;
3. `GAP_RAW_REVIEW_ROUND3.md`;
4. `GAP_RAW_PROOF_ROUND4.md`, including its folded post-review errata; and
5. `GAP_RAW_REVIEW_ROUND4.md`.

**File authored:** `GAP_RAW_PROOF_ROUND5.md`.

The test-gated harness, production rules, strict verifier, and Lean sources
were not modified.  No Cargo command, Lean build, harness, search executable,
or generated enumeration was run.

## 47. Post-review errata (R-G3-REV, folded from GAP_RAW_REVIEW_ROUND5.md)

Hostile review of artifact `d93d5768` returned **SOUND-WITH-ERRATA**: no
round-5 theorem refuted; L11.1–L11.9, R5.1–R5.3.1, the shared-hub cascade, and
the unrestricted equation-(22) closure refutation all CONFIRMED.  The
following repairs are folded in place above:

1. **Finding 4 (MAJOR, folded in Section 46):** the "exact/sole resume point"
   characterization of `GAP-HUB-FANOUT-REACHABILITY` is downgraded — it is an
   obstruction-specific subproblem of `GAP-CASCADE-REACHABILITY`, with the
   three differently quantified problems now stated separately.
2. **Finding 2 (MINOR, folded in Section 41.3):** R5.4's separation region is
   now formal — `G_i = A_i ∪ D_i ∪ Y_i`, `H = A_H ∪ D_H ∪ X`, pairwise set
   distance at least six — so the service cells `z_i` and all protected window
   cells are expressly covered.  R5.4 verdict: CONFIRMED-WITH-ERRATA.
3. **Finding 5 (MINOR, folded in Section 44):** the authoritative ledger now
   carries an OPEN row for `GAP-HUB-FANOUT-REACHABILITY` with the
   narrower-subproblem characterization.
4. **Finding 6 (MINOR, folded in Section 46):** provenance now distinguishes
   the input commit (`12980bc8`) from the reviewed/output artifact
   (`d93d5768`).
5. **Finding 7 (NOTE, folded in Section 39.x separation paragraph):** radius
   21 is tight for the conservative all-touched-alive-window envelope but not
   claimed sharp for the `M<=2` value conclusion; minimum separation for the
   value theorem is unresolved.

The review's closing list of exact unresolved obstacles is adopted verbatim
as the round-5 exit state: (1) universal strict-root `tau=0` initialization of
`M<=2`; (2) one named Defender strategy preserving `M<=2` after every response
on all reached histories; (3) quantifier-correct causal reachability or
exclusion of the shared hub, plus the other unclassified escape classes; and
(4) optionally, the minimum support separation for the separated value
theorem.
