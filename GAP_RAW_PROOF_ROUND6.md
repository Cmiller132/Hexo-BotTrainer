# GAP-RAW Proof Round 6 - fixed-policy hub reachability

**Worktree:** `hunt/gap-raw` at input HEAD `aed0fecb`  
**Date:** 2026-07-17  
**Disposition:** fixed-`S_T` hub reachability is **PROVEN** and the policy
`S_T` is **REFUTED** as a universal repair policy. Strategy-independent hub
forcing and positive universal initialization/repair remain **OPEN**.

This document continues the definitions, theorem numbering, and equation
numbering of `GAP_RAW_PROOF_ROUND5.md`, including its binding folded Section
47 errata. In particular, `I`, `tau`, `TEMPO`, `M`, `Serv`, `V_T`, and `S_T`
have exactly their inherited meanings. `S_T` first minimizes `TEMPO` over the
same actual ordered pairs which service `I`, and only then minimizes the two
cells lexicographically, by axial coordinate `(q,r)`.

No Cargo command, Lean build, harness, search program, or machine enumeration
was run. Every new result is a hand proof. There are no new `VERIFIED` or
`[UNRUN]` claims.

## 48. Executive verdict and quantifier contract

### 48.1 The three problems remain separate

Every round-6 claim below carries one of these tags.

1. **(Q1) Fixed-`S_T` hub reachability.** The root and continuation in
   Sections 49--52 are consistent with the one already defined policy `S_T`.
   They reach the five count-two shared-hub labels and then force an Attacker
   win. This refutes `S_T` only.
2. **(Q2) Strategy-independent hub forcing.** The needed order is
   `exists P_0, for every Defender strategy S, there exists an S-consistent
   Attacker continuation alpha`. Nothing below supplies that universal
   Defender-strategy quantifier. **Q2 remains OPEN.**
3. **(Q3) Positive universal initialization/repair.** The needed order is
   `for every P_0, there exists one S, for every alpha`. Refuting `S_T` does
   not refute that statement. A different policy might occupy the hub. It
   would also have to control every other fanout, cross-hull interaction,
   nested-derivative next turn, and response from the axial-cleanup class.
   **Q3 remains OPEN.**

### 48.2 Main result

**Theorem R6.1 (fixed-`S_T` shared-hub assembly) [Q1, PROVEN].** There is a
finite, nonempty, nonterminal Defender-`FirstStone` root `P_0` with
`Phi(P_0)=0<1` and a legal `S_T`-consistent Attacker continuation which,
after the R5.3.1 axial-cleanup handoff, reaches an Attacker-`FirstStone`
handoff at which:

- the hub `h=(10,0)` is empty and Defender-free;
- the five focal windows `U^-`, `U^+`, `V^-`, `V^+`, and `W` from the L11.6
  geometry are all Defender-free and have Attacker count exactly two; and
- every intervening `S_T` pair through this required assembly handoff is the
  exact pair derived from its actual minimize-`TEMPO`-then-lexicographic
  definition.

The constructed handoff has additional alive labels; it is not claimed to
have the exact isolated `F_H` profile of L11.6. The five load-bearing focal
labels are nevertheless intact, and Theorem R6.2 proves that the extra labels
do not spoil the L11.7 cascade.

**Theorem R6.2 (`S_T` loses from the constructed strict root) [Q1,
PROVEN].** From the R6.1 handoff, Attacker plays the hub pair. Every servicing
reply, hence the actual reply selected by `S_T`, leaves three disjoint next
demands. Attacker then reaches `tau>=3` and wins after the following Defender
turn. Thus `S_T` is **REFUTED** as a strategy witnessing GAP-RAW or
`GAP-TEMPO-REPAIR` from every strict root.

Theorems R6.1--R6.2 do not decide Q2 or GAP-RAW.

### 48.3 Secondary initialization result

**Lemma L12.6 (low-only initialization slice) [Q3-initialization, PROVEN].**
At every finite nonterminal Defender epoch whose alive labels all have count
at most two, `tau=0` and `M<=2`. If every alive label has count at most one,
then `M=0`. Consequently every strict `Phi<1` root in this entire class is an
initialized `M<=2` root, beyond the two individual round-5 instances.

This closes only the low-only slice. Strict roots containing count-three
labels remain in the general `tau=0` initialization problem.

## 49. The exact strict root and the axial-cleanup handoff

### 49.1 Normalize the R4.7 three-anchor root

Translate the three-site root used in R5.3.1 so that its lowest launch is at
row zero. Put

```text
c_j=(0,30j),   d_j=(1,30j),   a_j=(0,30j+8),   j=0,1,2,
```

and let `V_j` be the union of all thirty-one prospective windows through
`c_j` or `d_j`. Define

`P_0=(A=empty, D={a_0,a_1,a_2}, Defender, FirstStone)`.      (52)

This is finite, nonempty, nonterminal, and has no alive window. Hence
`Phi(P_0)=0`, `I(P_0)=empty`, and every legal Defender pair services. Every
resulting handoff also has `L_23=empty`, so every candidate pair has
`TEMPO=0`.

The lexicographically first legal cell is `(-8,8)`: no legal cell can have
`q<-8`, and among legal cells at `q=-8` the anchor `a_0=(0,8)` permits the
least second coordinate `r=8`. After that placement, the newly legal minimum
is `(-16,8)`.
Therefore the exact initial `S_T` pair is

`D@(-8,8), D@(-16,8)`.                                    (53)

Both cells lie outside all three `V_j`; in particular `V_0` is untouched.
Attacker legally plays

`A@(0,0), A@(1,0)`.                                       (54)

The first cell is at distance eight from `a_0`, and the second is adjacent.
All pre-existing Defender cells are outside the thirty-one-window footprint,
so the returned epoch is exactly the translated `P_raw` of R5.3 as far as its
alive family is concerned.

### 49.2 Derive the actual axial cleanup

R5.3 applies with no external-label qualification: the unique
lexicographically first value-zero servicing pair is

`D@(-4,0), D@(2,0)`.                                      (55)

The additional remote supports do not create another value-zero reply. As in
R5.3, value zero requires deletion of all five common count-two labels, so
both effective cells must occur in their Q-row union; cells made legal only
by a remote support cannot replace either member of the cover.

The first cell kills the Q-windows with starts `-5,-4`; the second kills the
starts `-3,-2,-1,0,1`. Thus all seven Q-row labels through the adjacent pair
are dead. Neither cell lies on one of the four transverse pencil lines.

Let `Q_ax` be the resulting Attacker handoff. Its complete local alive family
is the following twenty-four count-one windows:

- six R-axis windows through `(0,0)`;
- six QR-axis windows through `(0,0)`;
- six R-axis windows through `(1,0)`; and
- six QR-axis windows through `(1,0)`.

There is no local count-two label, so this is exactly the R5.3.1
axial-cleanup handoff class at the selected untouched site. All later
coordinates are in this normalization.

## 50. The four-pencil tempo shield

### 50.1 One exact Attacker pair

From `Q_ax`, Attacker plays

`p=(0,1)`, then `p'=(1,-1)`.                               (56)

Both placements are empty and adjacent to old Attacker stones, so the ordered
pair is legal. The first placement promotes five R-windows through `(0,0)`
and five QR-windows through `(1,0)`. The second promotes the complementary
five R-windows through `(1,0)` and five QR-windows through `(0,0)`. The two
new cells are not collinear, so no window contains both of them without also
being one of those old pencils.

**Lemma L12.1 (exact diamond profile) [Q1, PROVEN].** At the returned
Defender epoch the exact local profile is

`n_2=20, n_1=20`, with no label of higher count.             (57)

The twenty count-two labels form four five-window adjacent-pair pencils on

`q=0`, `q=1`, `q+r=0`, and `q+r=1`.                        (58)

*Proof.* Equation (56) promotes twenty of the twenty-four old count-one
labels. The four unpromoted old extremes stay at count one. Each new stone
lies in eighteen windows, ten of which are the promoted focal labels. Its
other eight windows are new count-one labels. The two new stones share no
axis, so the two sets of eight are distinct. None of the external Defender
stones lies within a six-window of a new stone. Thus
`n_1=4+8+8=20`, and (57) follows. The four promoted adjacent pairs are
respectively

```text
(0,0),(0,1);   (1,-1),(1,0);
(0,0),(1,-1);  (0,1),(1,0),
```

which gives exactly the four lines in (58).  ∎

The two fixed-`q` lines are parallel, as are the two fixed-`q+r` lines. The
four remaining intersections are

```text
(0,0), (0,1), (1,-1), (1,0),
```

and all four are occupied by Attacker. Therefore an empty Defender cell lies
on at most one line in (58).

### 50.2 Every immediate value is exactly two

For two adjacent Attacker cells at axis parameters `0,1`, call the five
common windows the *raw pencil*. Call it *intact* when all five labels are
Defender-free at count two and the witness cells at parameters `-1,2` are
empty. In an intact pencil, those future triggers make the three windows with
starts `-3,-2,-1` imminent, with residuals

`{-3,-2}`, `{-2,3}`, and `{3,4}`.                           (59)

Their hitting number is two: the first and third grounds are disjoint, while
`{-2,3}` hits all three.

**Lemma L12.2 (four-pencil plateau) [Q1, PROVEN].** Suppose a Defender epoch
has `I=empty`, every `L_23` label has count two, and it contains the four
intact raw pencils in (58). Suppose also that no empty cell lies on two of
their four central lines. Then every legal ordered Defender pair hands over
exactly

`TEMPO=2`.                                                 (60)

*Proof.* Defender augmentation preserves the property that every surviving
`L_23` label has count two, so L10.4 gives `TEMPO<=2` after every pair. One
Defender cell can meet labels of at most one of the four raw pencils. Two
cells therefore leave at least two pencils untouched. In particular one
untouched pencil retains the legal triggers and residual family (59), so its
component has `g>=2`. Equation (20) gives `TEMPO>=2`.  ∎

At the epoch of L12.1, `I=empty`, all graded labels are count two, and the
intersection condition was just proved. The earlier Defender cells do not
damage a shield pencil: `(0,8)` lies beyond the union on `q=0`, `(-8,8)` lies
beyond the union on `q+r=0`, and all other old Defender cells lie on none of
the four finite pencil unions. Hence every candidate `S_T` reply has the same
value two.

### 50.3 Exact lexicographic ray

**Lemma L12.3 (left-ray tie breaker) [Q1, PROVEN].** Suppose `I=empty` at a
Defender epoch, every legal ordered pair has the same `TEMPO`, and the
occupied set has a unique minimum-`q` cell `(ell,8)`. Then the exact `S_T`
pair is

`D@(ell-8,8), D@(ell-16,8)`.                               (61)

Afterward `(ell-16,8)` is the unique minimum-`q` occupied cell.

*Proof.* Radius-eight legality implies that no first cell can have
`q<ell-8`. The cell `(ell-8,8)` is legal from `(ell,8)`. Because the old
minimum-`q` cell is unique, any legal cell with `q=ell-8` must be supported
from `(ell,8)`; the axial distance inequalities make `r=8` the least possible
second coordinate. After that placement the same argument gives
`(ell-16,8)` as the lexicographically first second cell. Since `I=empty`,
every legal pair services; since all such pairs tie in the primary objective,
`S_T` uses these two lexicographic minima.  ∎

Immediately before the first plateau reply, `(-16,8)` is the unique
minimum-`q` occupied cell. Lemmas L12.2--L12.3 therefore derive the actual
reply

`D@(-24,8), D@(-32,8)`.                                   (62)

This is not an asserted filler choice: it follows from the complete
`TEMPO=2` tie and the actual definition of `S_T`.

## 51. Exact hub assembly behind the shield

### 51.1 The five Attacker stock turns

Set

`h=(10,0)`, `u=(1,0)`, `v=(0,1)`, and `w=(1,-1)`.

On five successive Attacker turns, play

```text
U- stock:  (6,0),   (7,0)
V- stock:  (10,-4), (10,-3)
W  stock:  (12,-2), (13,-3)
V+ stock:  (10,4),  (10,5)
U+ stock:  (14,0),  (15,0).                               (63)
```

Every ordered pair is legal on the ordinary 2:2 cadence. The first placements
have the following already occupied supports and distances; every second
placement is adjacent to its first:

| First placement | Earlier support | Hex distance |
|---|---|---:|
| `(6,0)` | `(1,0)` | 5 |
| `(10,-4)` | `(7,0)` | 4 |
| `(12,-2)` | `(10,-3)` | 3 |
| `(10,4)` | `(10,-3)` | 7 |
| `(14,0)` | `(13,-3)` | 4 |

Thus all ten placements satisfy the inclusive radius-eight rule. They are
also disjoint from all earlier Defender cells and from the hub `h`.

### 51.2 No hidden high label

**Lemma L12.4 (global count-two audit) [Q1, PROVEN].** After all placements
in (63), every length-six window contains at most two Attacker stones. The
same is true after every initial segment of (63).

*Proof.* It is enough to inspect the final Attacker set, since intermediate
sets are subsets. On fixed-`q` lines, the only groups of size above one are

```text
q=0:  r={0,1};
q=1:  r={-1,0};
q=10: r={-4,-3,4,5}.
```

The two pairs on `q=10` have span seven between `-3` and `4`, so a six-cell
interval meets at most one pair. On fixed-`r` lines, the only larger groups
are

```text
r=0:  q={0,1,6,7,14,15};
r=-3: q={10,13}.
```

Every six-consecutive factor of the first set contains at most two listed
coordinates. Finally, on fixed-`q+r` lines the non-singleton groups have
levels

```text
0, 1, 6, 7, 10, 14, 15,
```

and each level contains exactly two Attacker stones. These three line
families are all lattice axes, proving the claim.  ∎

In particular every intermediate Defender epoch has `I=empty`, and every
surviving `L_23` label has count two. None of the stock cells in (63) lies on
one of the shield lines `q=0`, `q=1`, `q+r=0`, or `q+r=1`, so the four raw
pencils remain intact. Every candidate Defender pair is therefore servicing,
and L12.2 makes its handoff value exactly two.

The unique leftmost occupied cell remains on the row `r=8`; each actual
reply extends it by equation (61). Consequently the complete sequence of
actual `S_T` replies is

| Attacker pair just played | Exact `S_T` reply |
|---|---|
| shield pair `(0,1),(1,-1)` | `(-24,8),(-32,8)` |
| `(6,0),(7,0)` | `(-40,8),(-48,8)` |
| `(10,-4),(10,-3)` | `(-56,8),(-64,8)` |
| `(12,-2),(13,-3)` | `(-72,8),(-80,8)` |
| `(10,4),(10,5)` | `(-88,8),(-96,8)` |
| `(14,0),(15,0)` | `(-104,8),(-112,8)` |

Each first Defender cell is at distance exactly eight from the preceding
leftmost cell, and each second is at distance exactly eight from the first.
Thus the table derives both legality and the ordered `FirstStone`/
`SecondStone` choices; no simultaneous-pair convention is used.

### 51.3 The attained handoff

Let `Q_H^*` be the Attacker handoff after the last reply in the table. Define

```text
U^- = {h+t u : -4<=t<=1},
U^+ = {h+t u :  0<=t<=5},
V^- = {h+t v : -4<=t<=1},
V^+ = {h+t v :  0<=t<=5},
W   = {h+t w : -2<=t<=3}.                                 (64)
```

These are the five focal windows of L11.6, translated to `h`. Their exact
Attacker supports are

```text
U^-: (6,0),(7,0);       U^+: (14,0),(15,0);
V^-: (10,-4),(10,-3);   V^+: (10,4),(10,5);
W:   (12,-2),(13,-3).
```

No displayed `S_T` cell belongs to a window in (64). The old anchors and the
two axial-cleanup cells also lie outside them. Hence all five windows are
Defender-free and have count exactly two. Their common hub `h=(10,0)` is
empty. This proves Theorem R6.1.

The construction deliberately does not install the isolating blockers
`D_H` from L11.6. Extra count-one and count-two labels remain alive. Section
52 audits the complete imminent family at the cascade turn, so the focal
conclusion does not rely on pretending those labels are absent.

## 52. The focal cascade survives the extra labels

### 52.1 Exact current service

From `Q_H^*`, Attacker plays

`A@h=(10,0)`, then `A@(h+u)=(11,0)`.                       (65)

The first cell lies in all five alive focal windows and is legal by L6_2;
the second is adjacent. L12.4 says every window had count at most two before
this pair, so no six is completed.

**Lemma L12.5 (complete imminent-family audit) [Q1, PROVEN].** At the
returned Defender epoch `P_1`, the imminent family is exactly
`{U^-,U^+}`. Its residuals are

`E(U^-)={(8,0),(9,0)}`, `E(U^+)={(12,0),(13,0)}`,           (66)

so `tau(P_1)=2`.

*Proof.* A pre-count-two label can become count four only if it contains both
new triggers. Two distinct u-adjacent triggers share windows only on their
u-line. The five u-windows containing both have starts `6,7,8,9,10`. Before
(65), their respective old Attacker counts were

`2,1,0,1,2`.

After (65), their counts are `4,3,2,3,4`. Since there was no pre-count-three
label, no window receiving only one trigger becomes imminent. This proves
completeness, and (66) is immediate from (64). The two residual grounds are
disjoint.  ∎

Every servicing ordered pair selected at `P_1` therefore has one occupancy
cell

`ell in {(8,0),(9,0)}`

and the other

`r in {(12,0),(13,0)}`,                                    (67)

in one of the two orders. There is no spare and no non-u-axis servicing
alternative. Because `S_T` minimizes only over `Serv(P_1)`, its actual pair
is necessarily one of the pairs in (67), regardless of how the extra labels
affect its primary-value comparison.

### 52.2 Three forced surviving demands

The service cells in (67) are nonhub u-axis cells. They lie in none of
`V^-`, `V^+`, or `W`: the first two are on `q=10`, the third is on
`q+r=10`, and the four possible service cells have neither coordinate
property. After (65), those three focal labels each have count three, and
they therefore survive every servicing reply.

At the resulting Attacker handoff, play

`A@(h+v)=(10,1)`, then `A@(h+w)=(11,-1)`.                  (68)

Both cells are empty residual cells of surviving alive labels, hence legal.
The handoff before (68) has `I=empty` because the actual pair serviced the
complete family in L12.5; L1.1 therefore also proves that (68) is
nonterminal. At the next Defender epoch, the three focal residuals are

```text
V^-: {(10,-2),(10,-1)},
V^+: {(10, 2),(10, 3)},
W:   {(8,2),(9,1)}.                                       (69)
```

They are pairwise disjoint. Thus the full imminent family has hitting number
at least three, whether or not additional labels also became imminent.
Equivalently, R4.1 gives `TEMPO>=3` after every servicing pair in (67), so
definition (21) gives `M(P_1)>=3`.
L1.2 says every Defender pair misses a member, which Attacker completes on
the following turn. This proves Theorem R6.2.

### 52.3 Exact Q1 conclusion

The complete chronology is

```text
strict Phi=0 root
  -> exact initial S_T lex pair
  -> fresh adjacent launch
  -> exact R5.3 axial cleanup
  -> four-pencil shield
  -> five exact stock turns behind six exact S_T lex pairs
  -> Attacker handoff with empty hub and five count-two focal labels
  -> hub pair, mandatory saturated service, three-demand fanout, win.
```

Every arrow is an ordered pair on the ordinary 2:2 cadence. All Attacker
placements have an explicit distance-at-most-eight support; all `S_T` ray
placements have distance exactly eight support; and every service or future
trigger cell lies in an alive window. The constructed history therefore
decides Q1 positively: the hub is reachable against fixed `S_T`, and `S_T`
loses.

## 53. Quantifier-correct consequences for repair

### 53.1 Q1 does not lift to Q2

**Q2 status [OPEN].** The proof fixed `S_T` before choosing the continuation.
Its key plateau uses the fact that this policy resolves equal immediate values
by chasing the lexicographic left frontier. A different strategy may occupy
`h` as soon as focal stock identifies it. In particular, at the final
Defender epoch immediately before the last reply in the Section 51 table,
`D@h` is legal and kills all five focal labels at once. This defeats the
displayed hub continuation, although no all-response survival claim is made
for that alternative action.

Therefore R6.1 has the fixed-policy form needed to refute one strategy. It
does not have the Q2 order

`exists P_0, for every S, exists alpha`.                    (70)

Strategy-independent hub forcing, and the broader
`GAP-CASCADE-REACHABILITY`, remain **OPEN**.

### 53.2 The exact `S_T` decision failure

Let `P_stock` be the Defender epoch after Attacker has played the final pair
`(14,0),(15,0)` but before `S_T` replies. At this state all five focal labels
are present, the four shield pencils are intact, `I=empty`, and every legal
Defender pair has exact post-handoff `TEMPO=2` by L12.2. Two particular kinds
of pair therefore tie in `S_T`'s entire primary objective:

- a pair containing `D@h`, which deletes the five focal labels; and
- the lexicographic ray pair `(-104,8),(-112,8)`, which leaves them all alive.

For the first kind, the concrete ordered pair
`D@(-104,8), D@h` is legal: its first cell is at distance eight from the old
leftmost cell `(-96,8)`, while `h` lies in the focal alive windows.

The actual definition chooses the ray pair solely by the ordered
lexicographic tie breaker: after the common first cell `(-104,8)`, its second
cell `(-112,8)` precedes `h`. Its resulting handoff admits (65), and the
successor `P_1` has `M(P_1)>=3` by Sections 52.1--52.2.

For diagnosis only, define the one-successor Bellman risk of a servicing pair
`a` by

`R_1(P,a)=max_b M(P+D@a_1+D@a_2+A@b_1+A@b_2)`,             (71)

where `b` ranges over legal nonterminal ordered Attacker responses; assign
infinite risk to an immediate winning response. This is not asserted as a
closed invariant or as a complete policy. At `P_stock`, the actual `S_T`
pair has `R_1>=3`, witnessed by (65), even though its immediate `TEMPO` is
only two.

**Q3 structural conclusion [PROVEN state distinction; repair OPEN].**
Immediate `TEMPO` alone cannot distinguish the losing ray from focal-hub
pre-emption at this state. A repair proof which excludes this displayed
cascade must distinguish actions that tie at `TEMPO=2` using some additional
information. Two natural representations of the missing information are:

1. the worst next-epoch value represented by (71); or
2. a latent certificate consisting of five low labels with one
   common empty hub and the saturated-service continuation of L11.7.

These are design targets, not an exhaustive theorem about how a policy must
be encoded; a hard-coded structural tie rule could distinguish the actions
without literally storing either representation.

The earliest causal warning occurs at the epoch after the second stock turn:
once `U^-` and `V^-` have both been built, their unique intersection is the
still-empty cell `h`, and `D@h` is already legal. Immediate `TEMPO` is
nevertheless flat at two because of the shield. At `P_stock`, equation (71)
exposes the losing ray's one-successor risk. Recognizing the earlier warning
requires a deeper or explicitly history-sensitive objective.

No claim is made that a hub-first pair has `R_1<=2` against every response.
That all-response comparison is part of the still-open repair theorem.

### 53.3 What Q3 still has to cover

The R6.1 continuation proves that minimizing equation (20) alone is not a
positive repair policy. It does not close any of these independent Q3
classes:

- other shared-cell or non-shared `M>2` fanouts;
- cross-hull interactions below the R5.2 separation premise;
- next turns from the nested derivative (31);
- alternative forced-service entrances to a transverse seal;
- every other legal response from the R5.3.1 axial-cleanup handoff; and
- the general strict-root `tau=0` initialization geometry containing
  count-three labels.

Accordingly `GAP-TEMPO-REPAIR`, `GAP-TEMPO-INITIALIZATION`, and the positive
universal order `for every P_0, exists S, for every alpha` all remain
**OPEN**. The only repaired-policy candidate actually refuted here is `S_T`.

## 54. A broader strict-root initialization slice

**Proof of Lemma L12.6.** Let `P` be a finite nonterminal Defender epoch in
which every alive label has count at most two. Then `I(P)=empty` and
`tau(P)=0`, so every legal ordered Defender pair services. Such a pair exists
by the finite nonempty filler construction in L1.2.

Defender placements only delete labels. At every resulting Attacker handoff,
all labels in `L_23` therefore have count exactly two. L10.4 gives
`TEMPO<=2`; minimizing over `Serv(P)` gives `M(P)<=2`. If no count-two label
exists at `P`, the handoff has `L_23=empty`, so every candidate value is zero
and `M(P)=0`.  ∎

The strict-root corollary follows by intersecting this state class with
`Phi(P)<1`. The proof itself does not need the potential bound. This is a
complete initialization class, not a renewal theorem: one later Attacker
turn may create count-three labels, and L12.6 gives no next-cycle repair.

## 55. Authoritative round-6 status ledger

| Claim / named gap | Quantifier tag | Status | Exact basis / remaining scope |
|---|---|---|---|
| GAP-RAW | Q3 target / Q2 counterroute | **OPEN** | `S_T` loses, but neither another universal Defender strategy nor a strategy-independent Attacker win is proved |
| R6.1 fixed-`S_T` hub assembly | Q1 | **PROVEN** | Exact strict root, actual replies, legal cadence, and five intact focal labels, Sections 49--51 |
| R6.2 fixed-`S_T` forced loss | Q1 | **PROVEN** | Complete imminent audit, saturated service, and three disjoint next residuals, Section 52 |
| `S_T` as a universal GAP-RAW / tempo-repair policy | Q1 consequence | **REFUTED** | The strict `Phi=0` root and legal continuation of R6.1--R6.2 |
| `GAP-HUB-FANOUT-REACHABILITY`, fixed-`S_T` branch | Q1 | **PROVEN** | Positive reachability decision; the hub is assembled before `S_T` occupies it |
| `GAP-HUB-FANOUT-REACHABILITY`, strategy-independent branch | Q2 | **OPEN** | No `for every Defender strategy` forcing argument |
| `GAP-CASCADE-REACHABILITY` | Q2 / broader | **OPEN** | The fixed-policy cascade does not force this or another cascade against every strategy |
| Positive universal initialization/repair | Q3 | **OPEN** | Required order remains `for every P_0, exists S, for every alpha` |
| `GAP-TEMPO-INITIALIZATION` | Q3 | **OPEN** | L12.6 closes the low-only slice; count-three `tau=0` roots remain |
| L12.6 low-only initialization slice | Q3-initialization | **PROVEN** | Every alive count at most two implies `M<=2`; at most one implies `M=0` |
| `GAP-TEMPO-REPAIR` for some one named strategy | Q3 | **OPEN** | Refuting `S_T` does not exclude a different Bellman-aware strategy |
| `S_T` as an immediate-`TEMPO` universal repair candidate | Q1/Q3 boundary | **REFUTED** | Exact plateau at `P_stock` selects a reply with Bellman risk at least three |
| `GAP-REPLACEMENT-INVARIANT` | Q3 | **OPEN** | No alternative invariant/policy is initialized and renewed on every reached history |
| `GAP-AMORTIZED-ABANDONMENT` / non-dominating credit route | Q3 | **OPEN** | Round 6 supplies no formal credit, refund, or closure rule |
| Canonical `GAP-GLOBAL-RENEWAL` and canonical J | inherited boundary | **REFUTED** | R3.1 remains binding; no pointwise `Theta_2<1` route is revived |
| General standalone K3 suppression | Q3-initialization | **OPEN** | L12.6 avoids count-three labels and does not solve the remaining free-pair geometry |
| L12.1 diamond shield profile | Q1 | **PROVEN** | Exact `n_2=20,n_1=20` hand count |
| L12.2 four-pencil plateau | Q1 | **PROVEN** | Pure-count-two upper bound plus an untouched `g=2` raw pencil after every pair |
| L12.3 lexicographic left ray | Q1 | **PROVEN** | Exact radius-eight and axial lexicographic argument |
| L12.4 no-hidden-high-label audit | Q1 | **PROVEN** | Complete fixed-`q`, fixed-`r`, fixed-`q+r` classification |
| L12.5 complete hub-pair imminent audit | Q1 | **PROVEN** | Only `U^-`,`U^+` are current; residuals are disjoint |
| Bellman-risk diagnosis at `P_stock` | Q3 structural partial | **PROVEN** | Actual `S_T` pair has a legal response returning `M>=3` |
| A hub-first pair has robust Bellman risk at most two | Q3 | **OPEN** | Only deletion of the focal stock is proved, not all-response closure |
| Complete axial-cleanup response classification | Q3 | **OPEN** | One explicit losing continuation suffices for Q1 but does not classify every response |
| Other `M>2` fanouts | Q3 | **OPEN** | Shared hub is only one escape class |
| Cross-hull interaction closure | Q3 | **OPEN** | R5.2's separation remains load-bearing |
| Nested-derivative next turns | Q3 | **OPEN** | Equation (31) is not iterated here |
| Alternative forced-service transverse-seal entrance | Q3 | **OPEN** | R5.3 excluded only the natural entrance; round 6 does not decide another |
| Minimum separation for the R5.2 value theorem | ancillary | **OPEN** | Radius 21 remains only envelope-sharp, as in folded round-5 errata |
| New machine verification | all | **none** | Pure hand-proof authoring; no prohibited run or generated enumeration |

No inherited round-2 through round-5 `PROVEN` or `VERIFIED` theorem is
downgraded. R6.2 changes the status only of the proposed fixed policy `S_T`:
the policy is now refuted on a strict root. The positive universal theorem
and all differently quantified gaps retain their OPEN status.

## 56. Hostile-review attack surface

1. **Quantifier substitution.** R6.1 is Q1. Replacing fixed `S_T` by
   `for every S`, or treating its failure as a GAP-RAW refutation, is invalid.
2. **Initial lex pair.** At the three-anchor root all candidate values are
   zero. Check the closed radius-eight inequalities at `q=-8` and then the
   within-turn frontier expansion to `q=-16`.
3. **Use of R5.3.** The extra anchors and first lex pair are outside `V_0`.
   Hence the alive family at the raw adjacent-pair epoch is exactly the local
   family used by R5.3; otherwise (55) would not automatically follow.
4. **Diamond profile.** Verify all twenty promotions and all sixteen new
   count-one births. The two transverse centers are not collinear, and no
   old Defender cell silently kills one of their new windows.
5. **Physical-line intersections.** A Defender cell kills a shield label only
   by lying on that label's central line. The four nonparallel line
   intersections are precisely the four occupied Attacker cells, so no legal
   empty cell touches two shield pencils.
6. **Plateau lower bound.** L10.4 gives only the upper bound. The load-bearing
   lower bound is the untouched raw pencil and its exact residual family
   (59). Its trigger cells must remain empty.
7. **Lex-ray induction.** At each plateau epoch, every legal pair, not merely
   the displayed pair, has value two. Only then is it valid to invoke the
   lexicographic tie breaker. The unique minimum-`q` premise must persist.
8. **Global count audit.** Extra collinearities between the diamond and hub
   stock exist, notably on `r=0` and levels `q+r=6,7,14,15`. L12.4 includes
   them and proves that none yields a count-three window.
9. **Exact claim at the hub.** `Q_H^*` does not have the isolated nine-label
   family `F_H`. R6.1 claims only the five focal count-two labels. Any prose
   silently restoring the L11.6 blockers is false.
10. **Extra-label cascade audit.** L12.5 proves the complete current imminent
    family after (65), not just a focal subfamily. After service, the three
    focal residuals (69) are a subfamily with hitting number three; extra
    labels cannot lower that number.
11. **Actual service versus exact order.** At `P_1`, `S_T` chooses some member
    of the actual servicing set (67). The proof does not guess its order; it
    proves every member loses. This is sufficient and avoids a false
    tie-break claim in the presence of extra labels.
12. **Bellman boundary.** Equation (71) is a diagnostic. No theorem says that
    the hub-first action has low worst-response risk or that minimizing
    `R_1` closes repair globally.
13. **Initialization scope.** L12.6 is statewise and complete for low-only
    labels. It says nothing about the remaining count-three K3 geometry or
    about renewal after one Attacker turn.
14. **Legality and phase.** All radius bounds are inclusive. The proof uses
    sequential `FirstStone`/`SecondStone` updates for both colors and never a
    simultaneous pair.
15. **Evidence status.** Coordinate and hitting-set calculations are hand
    proofs labeled `PROVEN`; none is machine-verified.

## 57. Provenance and no-run record

**Input commit:** `aed0fecb` on branch `hunt/gap-raw`. This authoring pass
creates no commit; the orchestrator commits the artifact.

**Required corpus read first, in order, and in full:**

1. `GAP_RAW_PROOF_ROUND2.md`;
2. `GAP_RAW_PROOF_ROUND3.md` and `GAP_RAW_REVIEW_ROUND3.md`;
3. `GAP_RAW_PROOF_ROUND4.md` with its folded errata, then
   `GAP_RAW_REVIEW_ROUND4.md`; and
4. `GAP_RAW_PROOF_ROUND5.md` with its binding folded Section-47 errata, then
   `GAP_RAW_REVIEW_ROUND5.md`.

**File authored:** `GAP_RAW_PROOF_ROUND6.md`.

The test-gated harness, production rules, strict verifier, and Lean sources
were not modified. No Cargo command, Lean build, harness, search executable,
generated enumeration, or git commit was run.
