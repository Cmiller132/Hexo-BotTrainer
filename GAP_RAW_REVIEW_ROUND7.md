# R-G5-REV — Round-7 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`,
`fbae2f7ba13fcf8446e134d3d8cdfb7063688510`

**Reviewed document:** `GAP_RAW_PROOF_ROUND7.md`

**Named review input:** `R-G5-REV — HOSTILE REVIEW (ultra):
GAP_RAW_PROOF_ROUND7.md`

**Artifact's named authoring input:**
`8ac6caaec8668e77e7c4097c12336e0154c73841`

**Artifact identity:** SHA-256
`1758de37f0988b0dd332a692e73dadeb74e0e881842672c03c605a98a03601bb`;
Git blob `12e91ef709dd7d27037ac87f6cd3641fa7b2f067` at the reviewed HEAD.

**Method.** First-principles hostile proof audit. I read the required corpus in
the stipulated order and in full: `GAP_RAW_PROOF_ROUND4.md` with its folded
errata, `GAP_RAW_REVIEW_ROUND4.md`, `GAP_RAW_PROOF_ROUND5.md` including binding
§47, `GAP_RAW_REVIEW_ROUND5.md`, `GAP_RAW_PROOF_ROUND6.md` including binding
§58, `GAP_RAW_REVIEW_ROUND6.md`, and then the reviewed round-7 artifact. I
consulted rounds 2–3 only for the inherited blanket-game, legality, service,
completion, and ripeness definitions. I did not read any
`STRATEGY_STEALING_*` file as evidence and used no GAP-RAW-external evidence.
Every window census, count, residual, transversal, legality chain, phase
transition, and `M`/`tau` value below was recomputed by hand. No Cargo command,
Lean build, harness, game/search program, or machine enumeration was run.

**Overall verdict:** **SOUND-WITH-MINOR-ERRATA.** The hostile presumption does
not produce a mathematical refutation. R7.1, R7.2, R7.2.1, L13.1–L13.7, and
the conditional R7.3 all survive at their stated scopes. In particular,
`P_stock` is a universal local stop state, while Q2 root forcing remains open.
The two defects are documentary: the R7.1 paragraph and one ledger row attach
the universal `P_stock` verdict to the hub-only theorem before R7.2 supplies
the missing all-pair quantifier, and §66.2 omits the landed reviewed/output
artifact.

## Numbered findings

### 1. NOTE — the six plateau premises and exact value-two baseline are confirmed

**Quoted claim:**

> “Every `P_i^pl` has `I=empty`, every alive label has count at most two, all
> four shield pencils remain intact, and every legal Defender pair has
> immediate handoff value exactly two.” (§59.1)

**Independent recomputation.** At the diamond epoch `P_0^pl`, the local
Attacker set is

```text
(0,0), (0,1), (1,0), (1,-1).
```

The round-6 census is exact: four five-window adjacent-pair pencils give
`n_2=20`; four unpromoted old extremes plus eight new count-one windows through
each of `(0,1)` and `(1,-1)` give `n_1=20`; no label has higher count. The four
central lines are `q=0`, `q=1`, `q+r=0`, and `q+r=1`. Their nonparallel
intersections are precisely the four occupied diamond cells, so an empty
Defender cell can meet labels in at most one shield pencil.

The full stock list does not create a hidden count-three window. On the three
axis families, every nontrivial final line has the following parameter set:

```text
fixed q:     q=0:{0,1}, q=1:{-1,0}, q=10:{-4,-3,4,5}
fixed r:     r=0:{0,1,6,7,14,15}, r=-3:{10,13}
fixed q+r:   levels 0,1,6,7,10,14,15, two stones on each level
```

On `q=10`, the gap from `-3` to `4` prevents a length-six interval from
meeting both adjacent pairs. On `r=0`, every six consecutive parameters meet
at most two of `{0,1,6,7,14,15}`. Every other listed line has only two stones.
Every stock prefix is a subset of this final set, so the same upper bound holds
at all six plateau epochs. None of the ten stock cells lies on a shield line,
and the round-6 Defender ray and cleanup cells do not enter a shield-window
union. Defender augmentation only deletes labels.

For completeness, the full final `P_stock` alive-label census can also be
finished without an isolator assumption. Counting length-six intervals on each
occupied axis line, with `D@(-4,0),D@(2,0)` and the `r=8` ray included, gives

| Axis family | `n_1` | `n_2` |
|---|---:|---:|
| fixed `r` | 45 | 12 |
| fixed `q` | 44 | 20 |
| fixed `q+r` | 38 | 23 |
| **Total** | **127** | **55** |

The fixed-`r` count-two labels are nine on `r=0` and three on `r=-3`.
The fixed-`q` labels are five each on `q=0,1` and ten on `q=10`. The diagonal
count-two counts on levels `0,1,6,7,10,14,15` are respectively
`5,5,2,3,5,2,1`. For two stones at axis distance `d<=5` on an unblocked line,
these entries use `6-d` common count-two windows and `2d` exclusive count-one
windows. On `r=0`, the blockers at `-4,2` kill all windows through the old
pair and leave four count-two windows plus one count-one window through
`6,7`, while `14,15` contribute five and two. The remaining singleton lines
supply the rest of `n_1`. This accounts for every touched window and confirms
that `n_k=0` for all `k>=3`.

For any legal Defender pair, L10.4 gives `TEMPO<=2` because every surviving
`L_23` label is count two. The pair touches at most two shield pencils, leaving
an intact one. Its triggers at axis parameters `-1,2` create the residual
family

```text
{-3,-2}, {-2,3}, {3,4},
```

of exact hitting number two. Hence every candidate has `TEMPO>=2`, and the
complete candidate universe has exact value two. This confirms the plateau
wording required by binding §58: the shield is a flat value plateau, not a
strict immediate penalty on hub pre-emption.

**Proposed repair:** none.

### 2. NOTE — L13.1 is axis-complete, including every translated/reflected image

**Quoted claim:**

> “A consecutive-triple family” has no empty common blocker, and the flank
> trigger produces the residual family (75) of exact hitting number two.
> (§60.1, L13.1)

**Independent recomputation.** Normalize the Attacker triple to parameters
`0,1,2`. The four relevant six-intervals are

```text
[-3,2], [-2,3], [-1,4], [0,5].
```

Their common intersection is exactly `{0,1,2}`, all Attacker-occupied. Thus no
empty Defender cell lies in all four. Playing the empty flank `-1` matures the
first three windows and leaves

```text
{-3,-2}, {-2,3}, {3,4}.
```

The first and third residuals are disjoint, forcing at least two hits, while
`{-2,3}` hits all three, so the demand is exactly two. After one Defender
contact, killing all four labels would still require a cell in the occupied
common triple. At least one count-three label survives; any empty cell in its
three-cell residual is L6₂-legal and produces a nonempty count-four residual,
so the surviving demand is at least one.

There is no omitted axis or dihedral case. The three unoriented lattice axes
are isometric copies of this integer interval calculation, and every D6 image
preserves window length, occupancy, physical intersection, residual size, and
hitting number. Section 60 uses one family on each of the three axis
orientations, while `rho(q,r)=(1-q,-r)` supplies the displayed opposite fan.

**Proposed repair:** none.

### 3. NOTE — the south/north responses are legal, nonterminal, and have the exact fan census

**Quoted claim:**

> “`b_sigma` is a legal ordered nonterminal response and creates twelve focal
> count-three labels, four on each line of that fan. It creates no count-four
> label anywhere.” (§60.2, L13.2)

**Independent recomputation.** For

```text
b_- = ((0,-1),(2,-1)),
```

the first cell lies in the intact `q=0` shield pencil and is adjacent to old
Attacker stones. The second lies in the intact `q+r=1` pencil and is distance
two from the first (and already legal in an alive window). The resulting
triples are exactly

```text
q=0:     (0,-1),(0,0),(0,1)
q+r=1:   (0,1),(1,0),(2,-1)
r=-1:    (0,-1),(1,-1),(2,-1).
```

On the first two lines, the new trigger belongs to exactly four of the five
old count-two shield windows. On `r=-1`, the sole old Attacker stone was
`(1,-1)`, so exactly the four six-windows containing the new consecutive
triple rise from count one to count three. The other two axes through each new
cell contain no old pair. Thus the complete new count-three census is
`4+4+4=12`. Before the response every global count was at most two; a window
receiving one trigger reaches at most three, while the unique line receiving
both triggers had old count one. No label reaches count four, much less six.
The response is therefore nonterminal and returns `I=empty`. The north pair

```text
b_+ = ((1,1),(-1,1))
```

has the identical census under `rho`.

The finite fan supports recompute to

```text
south: q=0, r=-4..4; q+r=1, q=-3..5; r=-1, q=-3..5
north: q=1, r=-4..4; q+r=0, q=-4..4; r=1, q=-4..4.
```

Within either fan, the three pairwise line intersections are its three
Attacker-occupied triangle vertices. Across the fans, the only intersections
are the old occupied diamond cells `(0,0),(0,1),(1,0),(1,-1)`. The hub has
none of `q,r,q+r` in the required finite ranges. The anchors, all ray cells,
and cleanup cells `(-4,0),(2,0)` lie outside both supports. The stock cells
have row/column/level data `r=0`, `q=10`, `r=-3`, or levels
`6,7,10,14,15`, and direct substitution places every one outside both finite
fan unions. These checks establish legality and exact counts at every stock
prefix, not just at `P_stock`.

**Proposed repair:** none.

### 4. NOTE — L13.3 exhausts every legal next Defender pair and the full completion chronology

**Quoted claim:**

> “An empty Defender cell lies on high labels from at most one of the three fan
> lines ... Two Defender cells therefore leave at least one line family
> untouched.” (§60.3, L13.3)

**Independent recomputation.** After `b_sigma`, `I=empty`, so every legal
ordered Defender pair `d` is in `Serv`. For a pair `d`, let `C(d)` be the set
of the fan's three line families whose twelve focal windows contain at least
one of its cells. A cell in two families would have to be their central-line
intersection, but every such intersection is Attacker-occupied. Consequently
`|C(d)|` is exactly one of `0,1,2`; no legal pair has a missing fourth case.
This effect classification covers all legal fillers and all orders, rather
than sampling selected service coordinates.

| `|C(d)|` | Exhaustive effect | Future witness | Demand lower bound |
|---:|---|---|---:|
| 0 or 1 | At least two triple families are untouched, even if both cells concentrate on the one touched family | Use the L13.1 flank trigger on two untouched lines | `2+2=4` |
| 2 | One cell touches each of two families; the third is untouched | Untouched flank gives 2; either touched family retains a count-three label and an empty residual trigger gives at least 1 | `2+1=3` |

For reference, the three untouched flank cells are

```text
south: (0,-2), (-1,2), (-1,-1)
north: (1,-2), (-2,2), (-2,1).
```

An untouched flank cannot already contain Defender because it lies in that
Defender-free fan union. In the two-contact case, the auxiliary trigger is
chosen only after `d`, from the residual of a surviving alive count-three
label, so it too is empty and legal. Triggers on distinct lines are distinct:
the only line intersection is an occupied Attacker vertex. Their post-trigger
residual grounds are disjoint for the same reason, so the hitting numbers add.

Defender's stones only delete fan labels; they cannot create a new label or
change an Attacker count. Before the future pair every surviving alive count is
at most three, so that pair cannot complete six. It returns a nonterminal epoch
with `tau>=3`. Any following Defender pair misses an imminent label, and L1.2
then gives a legal one- or two-cell Attacker completion. Thus the expanded
chronology in §59.1 is proved for every `d`, not only for a selected response.
Taking the minimum over the complete servicing-pair set gives `M>=3`.

**Proposed repair:** none.

### 5. NOTE — R7.1 is confirmed at all six epochs and only for hub-containing pairs

**Quoted claim:**

> “For every `i=0,...,5` and every legal servicing pair `a` at `P_i^pl` which
> contains the hub `h=(10,0)`, Attacker has a legal nonterminal response pair
> `b`” returning `M>=3`. (R7.1, equation (72))

**Independent recomputation.** Every plateau has `I=empty`, so every legal
pair is servicing. The hub is outside both `G_-` and `G_+`. The other cell of
`a` is legally empty. It cannot lie in both fans because their exact
intersection consists only of four occupied Attacker cells. Hence at least one
complete `G_sigma` receives no new Defender stone. All older Defender cells
are already outside it by Finding 3. The corresponding `b_sigma` is legal and
nonterminal, and Finding 4 gives `M>=3` followed by the stated forced
completion chronology.

The quantifier does not extend to non-hub pairs at `P_0^pl,...,P_4^pl`.
Section 61's all-pair theorem applies only at `P_5^pl=P_stock`. The OPEN
first-plateau safe-action question is therefore compatible with R7.1.

**Proposed repair:** none to R7.1.

### 6. NOTE — R7.2.1 is independent of all five later stock turns

**Quoted claim:**

> “The actual first plateau ray reply `(-24,8),(-32,8)` already admits the
> triangle response in Section 60 and returns `M>=3`.” (R7.2.1)

**Independent recomputation.** Immediately after that first ray reply, the
Defender set relevant to the round-6 line is

```text
anchors:       (0,8), (0,38), (0,68)
initial ray:   (-8,8), (-16,8)
raw cleanup:   (-4,0), (2,0)
plateau ray:   (-24,8), (-32,8).
```

Every cell is outside both finite fan unions in Finding 3. The Attacker stock
at this point is only the four diamond stones; none of the ten cells from the
later `U^-`, `V^-`, `W`, `V^+`, or `U^+` turns exists yet. Nevertheless the
four shield pencils are already intact, the bridge rows contain exactly their
one old diamond stone, and both pairs in (76) are legal. Either response gives
the twelve-label triangle fan and the exhaustive `M>=3` conclusion of Finding
4. No later focal-hub label, stock support, or stock-era Defender ray cell is
load-bearing. This is a complete shorter fixed-`S_T` refutation.

**Proposed repair:** none.

### 7. NOTE — L13.4's three action regions are pairwise disjoint in the required physical sense

**Quoted claim:**

> “The three regions `H,G_-,G_+` are pairwise disjoint for legal Defender
> actions ... Hence one empty Defender cell can affect labels in at most one
> of the three regions.” (§61.1, L13.4)

**Independent recomputation.** The focal region is the union of three finite
axis segments:

```text
horizontal: r=0,  q=6..15       (U^- union U^+)
vertical:   q=10, r=-4..5       (V^- union V^+)
diagonal:   q+r=10, q=8..13     (W).
```

Compare these with the six fan segments in Finding 3.

- The horizontal segment meets fan columns `q=0,1` and fan diagonals
  `q+r=0,1` only at `q=0,1`, outside `q=6..15`; the fan rows are parallel.
- The vertical segment meets the fan rows at `(10,-1)` and `(10,1)`, while
  those fan-row segments end at `q=5` and `q=4`. Its intersections with fan
  diagonals would have `r=-9` or `-10`, outside `-4..5`; fan columns are
  parallel.
- The diagonal segment meets fan rows at `(11,-1)` and `(9,1)`, beyond their
  respective `q` ranges, and meets fan columns only at `q=0,1`, outside
  `q=8..13`; fan diagonals are parallel.

Thus `H` is literally disjoint from both finite fan unions. The two fan unions
meet only at occupied Attacker cells, as already recomputed. An empty legal
Defender action therefore belongs to at most one of `H,G_-,G_+`. This is
disjointness of the complete service/action cell sets, not merely disjointness
of labels or of their names.

**Proposed repair:** none.

### 8. NOTE — R7.2's adaptive fork covers every Defender pair, including cross-region and outside pairs

**Quoted claim:**

> “These two branches cover every `a` and prove (73)–(74).” (§61.2, R7.2)

**Independent recomputation.** Partition an arbitrary legal ordered pair `a`
only by the number of its cells in `H`.

| Cells of `a` in `H` | Remaining placements | Attacker response | Why it works |
|---:|---|---|---|
| 0 | Both may split across `G_-`,`G_+`, concentrate in one fan, or lie outside all three regions | Hub pair `(h,h+u)` | All five focal windows and all focal residual/trigger cells remain intact |
| 1 | The other cell may lie in one fan or outside every region | The untouched `b_sigma` | The `H` cell touches no fan and the other cell touches at most one |
| 2 | No placement remains for either fan | Either triangle pair | Both fans are untouched |

This table includes pairs split across the two fans, pairs wholly inside one
region, `H`/fan pairs, `H`/outside pairs, and pairs outside all regions. Order
does not create another effect class because both cells are classified after
their legal sequential insertion.

In the zero-`H` branch, `(h,h+u)=((10,0),(11,0))` is empty and legal. Before
it, arbitrary `a` could delete only nonfocal labels. The complete current focal
family is therefore still `U^-`,`U^+`, with residuals

```text
L={(8,0),(9,0)},    R={(12,0),(13,0)}.
```

They are disjoint. The servicing universe has exactly eight orders: choose
one of two cells in `L`, one of two in `R`, and either order,

```text
(8,0),(12,0)  (8,0),(13,0)  (9,0),(12,0)  (9,0),(13,0)
(12,0),(8,0)  (13,0),(8,0)  (12,0),(9,0)  (13,0),(9,0).
```

Every first cell is legal in one still-alive imminent window; the other window
remains alive to support the second. All four possible cells are nonhub
`r=0` cells and lie in none of `V^-`,`V^+`,`W`. The future pair
`(h+v,h+w)=((10,1),(11,-1))` is also untouched because both trigger cells lie
in `H`. It leaves the focal residuals

```text
V^-: {(10,-2),(10,-1)}
V^+: {(10, 2),(10, 3)}
W:   {(8,2),(9,1)},
```

whose grounds are pairwise disjoint. Hence every one of the eight services
hands over `TEMPO>=3`, regardless of which extra labels `a` killed. A
nonservicing reply misses one of the two current horizontal labels and loses
directly.

In the positive-`H` branch, Finding 7 leaves one entire fan untouched, and
Findings 3–4 give a legal nonterminal `b_sigma` returning `M>=3`. Thus every
legal `a` has a response satisfying (73), and every continuation after that
response is covered. The theorem proves a game-theoretic local loss, not only
a failure of the scalar evaluation rule.

At `P_stock`, `I=empty`, so `Serv(P_stock)` is the whole legal ordered-pair
set. Finding 1 makes every such pair an immediate-`TEMPO` minimizer of exact
value two. Since no Attacker pair can complete six from a pre-count-at-most-two
handoff, equation (71)'s maximum is over the same nonterminal responses used in
(73). Therefore

```text
min_a R_1(P_stock,a) >= 3
```

is valid here, and it excludes every tie refinement, including a one-ply
Bellman selection. The risk detects that the state has no safe action; it does
not produce one.

**Proposed repair:** none.

### 9. MINOR — the universal `P_stock` verdict is attributed one theorem too early

**Quoted claims:**

> “Consequently no hub-pre-empting pair at any named plateau epoch satisfies
> ... The requested `P_stock` answer is therefore **NOT-REPAIRABLE**.”
> (immediately after R7.1)

> “Named `P_stock` decision ... **PROVEN: NOT-REPAIRABLE** ... No
> hub-pre-empting pair has one-ply worst-response value at most two.” (§64
> ledger)

**Independent recomputation.** R7.1 quantifies only over pairs containing
`h`. By itself it leaves every non-hub pair unclassified, at `P_stock` as well
as at earlier plateaus. Therefore the universal phrase “NOT-REPAIRABLE” does
not follow from the immediately preceding hub-only consequence. The verdict
is nevertheless mathematically correct because R7.2 independently ranges over
every legal pair and proves the missing all-action statement. The adjacent
R7.2 ledger row also has the correct universal basis. No theorem status changes.

**Proposed repair:** move the universal `P_stock` verdict sentence to the end
of R7.2, or qualify the R7.1 sentence as “not repairable by hub pre-emption.”
In the named-decision ledger row, cite the adaptive all-pair fork of R7.2 rather
than only the hub-pre-emption result.

### 10. NOTE — the local-stop scope and Q1/Q2/Q3 quantifiers remain honest

**Quoted claim:**

> “Once `P_stock` is presented as a Defender epoch, Attacker wins against
> every subsequent Defender action ... [but] the theorem supplies no
> strategy-independent route from a strict root to `P_stock`.” (§61.3)

**Independent recomputation.** The proved local order is

```text
for every legal a at P_stock,
  there exists b,
    for every next Defender pair d,
      either d misses a current imminent,
      or there exists e returning tau>=3,
        after which every Defender pair permits completion.
```

That is enough to make the presented state losing. The only banked route into
the state is the fixed-`S_T` history of round 6. It does not quantify over an
arbitrary Defender strategy before choosing the continuation and therefore
does not establish the Q2 order

```text
exists P_0, for every S, exists an S-consistent Attacker continuation.
```

Likewise, failure of every action after arrival says that a positive Q3 policy
must intervene earlier; it does not refute the existence of some earlier
policy from every strict root. R7.1 never expands to non-hub actions at the
first five plateaus, the first-plateau safe-action problem remains open, and
the source does not infer a root-forcing result or GAP-RAW refutation.

**Proposed repair:** none.

### 11. NOTE — R7.3 and L13.5 correctly bind the same actual pair and exact early `R_1` values

**Quoted claims:**

> “Fix one deterministic Defender policy which ... uses one actual pair
> `a in A_0(P)` satisfying `R_1(P,a)<=2` ... then that same policy blocks
> forever.” (R7.3)

> “At the `Phi=0` three-anchor root, every initial Defender pair has `R_1=0`;
> ... every immediate-`TEMPO`-zero Q-row cover has exact `R_1=2`.” (L13.5)

**Independent recomputation.** For R7.3, membership in `A_0(P)` means that the
same actual ordered pair services and hands over `TEMPO=M(P)`. The root bound
and induction give `M(P)<=2`, so this handoff is unripe by R4.1. Its
`R_1<=2` bound says every legal response returns `M<=2`; the convention that an
immediate winning response has infinite risk rules out that missing case.
Repeating the same deterministic selection services every reached epoch, and
A2 applies. The availability hypothesis is exactly the still-open closure
obligation; it is not asserted as a constructed policy.

At the three-anchor root, any Attacker response has at most two stones. If
they share no window, there is no count-two label. If they are nonadjacent on
one axis, any empty internal gap lies in every common window; if such a gap is
already Defender-occupied, those windows are already dead. If they are
adjacent, the two exterior flanks hit the five common windows, with any
already-Defender-occupied or redundant flank omitted. Playing the needed
cells first and then fillers deletes every count-two label, so the returned
epoch has a candidate handoff with `L_23=empty` and exact `M=0`. This holds for
every response and gives exact root `R_1=0` for every initial action.

At `P_raw`, the twenty ordered value-zero Q-row covers enumerated in the
binding round-6 review delete every old count-two Q-label and meet none of the
four transverse pencil lines away from their occupied endpoints. An arbitrary
next Attacker pair can make a count-three label only from an old count-one
label containing both triggers. Two distinct triggers define a unique axis,
so the entire count-three family is a one-axis rank-triple family with an
at-most-two-cell transversal. Playing that cover leaves only count-two graded
stock, and L10.4 gives `M<=2`. Conversely, the legal shield response creates
the exact four-pencil plateau, where every Defender candidate has value two.
Thus every raw value-zero cover has worst response exactly `R_1=2`, not merely
an upper bound. Adding `R_1` as a tie layer preserves the round-6 initial and
raw-cleanup actions; the first new comparison is the diamond plateau.

**Proposed repair:** none.

### 12. NOTE — L13.6 handles arbitrary lower stock without claiming renewal

**Quoted claim:**

> “At a finite nonterminal Defender epoch with `tau=0`, if the alive
> count-three residual family has hitting number at most two, then `M<=2`.”
> (L13.6)

**Independent recomputation.** At a nonterminal epoch every imminent residual
is nonempty. Therefore `tau=0` forces `I=empty`; current service is vacuous.
Let `X` be a transversal of all alive count-three residuals with `|X|<=2`.
Every member of `X` is empty and belongs to an alive count-three window, so it
is L6₂-legal. If `|X|=2`, both cells are already legal before the first is
played; after the first kills some labels, the unchanged Attacker stone in a
window that originally contained the second still supplies distance-at-most-
five support. If fewer cells are needed, L1.2 fillers are appended only after
the transversal cells.

This same actual ordered pair kills every old count-three label. Defender
placements neither create alive labels nor raise Attacker counts, so every
surviving member of `L_23` has count exactly two. This conclusion remains true
when count-one/count-two labels intersect the killed count-three labels or the
transversal cells: such interactions can only delete additional lower stock.
L10.4 then gives `TEMPO<=2` for the same servicing pair, and definition (21)
gives `M<=2`.

For `|I_3|<=2`, choosing one residual representative per label yields such a
transversal; shared representatives only make it smaller. Hence L13.6.1 is
valid with arbitrary additional lower stock and no potential hypothesis.

The strict example is also exact. With

```text
W={0,u,2u,3u,4u,5u},    A={0,u,2u},
D=union_{Y meeting A, Y!=W}(Y\W),
```

every other same-axis length-six interval leaves `W`, and an off-axis interval
meets the central axis in at most one cell, so each `Y\W` is nonempty. The
finite blocker union avoids `W` and `A`, kills every other Attacker-touched
window, and leaves exactly `W` alive at count three. Thus
`Phi=1/(3 sqrt(3))<1` and `tau=0`. Playing `D@3u` plus a legal filler leaves
`L_23=empty`, proving exact `M=0`.

Section 63.3 correctly leaves both `tau_3>=3` initialization and every
all-response renewal question open. The next Attacker pair may create a new
high family or a count-one bridge fan; no part of L13.6 says that its
transversal condition regenerates.

**Proposed repair:** none.

### 13. NOTE — binding round-5/6 errata, numbering, and theorem statuses remain consistent

**Quoted claim:**

> “The exact round-6 post-hub strengthening `M=4`, `tau=4` remains binding on
> its stated ray history. Round 7 adds a different earlier cascade and a
> stronger final-state action quantifier.” (§64)

**Independent recomputation.** Section numbering resumes at §59 after binding
§58, and equations resume at (72) after the round-6 diagnostic (71). L12.2 is
used only where the four intact shield pencils and pure-count-two graded tier
are present; L12.4 supplies the all-prefix count-at-most-two premise. Neither is
extended into an unaudited renewal statement.

The round-6 exact result concerns its particular post-hub ray history: all
eight service orders there have `TEMPO=4`, the actual order is
`(8,0),(12,0)`, and the next exact demand is `tau=4`. Round 7's `>=3` bounds
are weaker lower bounds that cover more Defender actions and, for R7.2.1, a
different earlier triangle response. They do not revise or contradict the
exact round-6 values.

The folded round-5 Finding-4 contract is also respected. Q1 fixed-policy
reachability is banked; Q2 strategy-independent forcing and Q3 positive
initialization/repair retain their distinct quantifiers. R7.1 is hub-only at
all six plateaus, R7.2 is all-pair only at final `P_stock`, and R7.3 is
conditional. The OPEN rows for first-plateau safe action, general
count-three initialization, nested derivatives, cross-hull interactions,
other fanouts, and one-policy closure remain OPEN. No inherited theorem is
silently downgraded, and no hand proof is mislabeled `VERIFIED`.

**Proposed repair:** none.

### 14. MINOR — §66.2 omits the landed reviewed/output artifact

**Quoted claim:**

> “Input commit: `8ac6caaec8668e77e7c4097c12336e0154c73841` on branch
> `hunt/gap-raw`. This authoring pass creates no commit.” (§66.2)

**Independent recomputation.** The named authoring input is correct. The
reviewed file is present unmodified at HEAD
`fbae2f7ba13fcf8446e134d3d8cdfb7063688510`, with Git blob
`12e91ef709dd7d27037ac87f6cd3641fa7b2f067` and SHA-256
`1758de37f0988b0dd332a692e73dadeb74e0e881842672c03c605a98a03601bb`.
Section 66.2 records an authoring-time observed HEAD but not this landed
reviewed/output identity. That repeats the provenance omission repaired after
rounds 5 and 6. It does not affect any proof.

**Proposed repair:** add “Reviewed/output artifact:
`fbae2f7ba13fcf8446e134d3d8cdfb7063688510`” (optionally with the blob or
SHA-256 above), while keeping the existing sentence explicitly scoped to the
authoring session.
## Per-theorem verdicts

| Claim | Source status | Review verdict | Disposition |
|---|---|---|---|
| Six `P_i^pl` plateau premises | PROVEN / inherited | **CONFIRMED** | Every prefix has global count at most two; four shield pencils stay intact; every legal pair has exact immediate `TEMPO=2` |
| Final `P_stock` label census | implicit premise | **CONFIRMED AND COMPLETED** | Exact `(n_1,n_2)=(127,55)`; no count at least three |
| L13.1 consecutive-triple demand | PROVEN | **CONFIRMED** | All three axes and D6 images reduce to the same four-interval calculation; exact flank demand two; one contact leaves demand at least one |
| L13.2 exact south/north fans | PROVEN | **CONFIRMED** | Both ordered responses are empty, legal, and nonterminal; exactly twelve new count-three labels; no count-four label; finite supports and intersections are exact |
| L13.3 triangle-fan cascade | PROVEN | **CONFIRMED** | The complete next-pair contact census is `0/1/2` line families, giving lower demand `4/4/3`; triggers remain empty and legal |
| R7.1 all six hub-pre-emption failures | Q1/Q3, PROVEN | **CONFIRMED** | Hub misses both fans; the other legal empty cell misses at least one; no conclusion is extended to non-hub pairs at plateaus `i<5` |
| L13.4 hub/fan region separation | Q1/Q3, PROVEN | **CONFIRMED** | `H` is physically disjoint from both finite fan unions; the fan/fan intersection contains only occupied Attacker cells |
| R7.2 universal local `P_stock` stop | Q1/Q3, PROVEN | **CONFIRMED** | Exhaustive `H`-contact partition covers outside, concentrated, cross-fan, and split-region pairs; hub branch has all eight services; fan branch has the full `0/1/2` census |
| Equation (74), `min_a R_1>=3` | PROVEN | **CONFIRMED** | Every action is servicing and an exact immediate-value-two minimizer; each has a legal nonterminal response returning `M>=3` |
| R7.2.1 shorter fixed-`S_T` refutation | Q1, PROVEN | **CONFIRMED** | The first ray reply misses both complete fans before any stock turn; no later label is load-bearing |
| `P_stock` universal verdict as an immediate R7.1 consequence | wording / ledger basis | **MINOR ERRATUM** | R7.1 is hub-only; R7.2 supplies the valid all-action result |
| R7.3 one-ply stop-criterion assembly | Q3, PROVEN conditionally | **CONFIRMED AT ITS CONDITIONAL SCOPE** | One actual minimizing servicing pair carries both `TEMPO<=2` and `R_1<=2`; availability/closure remains a hypothesis |
| First plateau actual ray risk | PROVEN | **CONFIRMED** | Either untouched fan witnesses `R_1>=3` |
| Existence of a safe first-plateau one-ply action | OPEN | **CONFIRMED OPEN** | A pair can touch both known fans, but its entire response universe is not classified |
| One-ply Bellman minimization as a complete repair | OPEN | **CONFIRMED OPEN** | It detects the ray and final stop; safe-action availability and inductive closure are unproved |
| Threat-packing / count-one-bridge diagnosis | PROVEN on exact line | **CONFIRMED** | `H,G_-,G_+` are action-disjoint; each fan's third line is promoted count-one bridge stock |
| L13.5 root and raw `R_1` audit | PROVEN | **CONFIRMED** | Every root action has exact `R_1=0`; every raw value-zero Q-row cover has exact `R_1=2` |
| R5.4 tie-only negative control | REFUTED at statewise high-`Phi` domain | **CONFIRMED AT INHERITED SCOPE** | Forced service has no alternative occupancy; no strict-root conclusion is added |
| L13.6 `tau_3<=2` initialization | PROVEN | **CONFIRMED** | The same legal pre-emptive pair kills all count-three labels; arbitrary interacting lower stock can only be further deleted; L10.4 then applies |
| L13.6.1 at most two count-three labels | PROVEN | **CONFIRMED** | One residual representative per label gives a transversal of size at most two |
| L13.7 strict one-label example | PROVEN | **CONFIRMED** | Blocker comprehension is nonempty and disjoint; exact profile is one count-three label; exact `M=0` |
| General `tau_3>=3` initialization | OPEN | **CONFIRMED OPEN** | Strict roots can contain several count-three labels; no universal two-cell transversal is proved |
| Renewal from L13.6's rung | OPEN | **CONFIRMED OPEN** | The lemma is initialization-only and supplies no all-response regeneration |
| Q2 strategy-independent reachability of `P_stock` or a losing cascade | OPEN | **CONFIRMED OPEN** | Only fixed-`S_T` reachability is banked |
| Positive Q3 one-policy repair | OPEN | **CONFIRMED OPEN** | Local loss requires earlier intervention but does not rule out another earlier strategy |
| GAP-RAW | OPEN | **CONFIRMED OPEN** | No `for every Defender strategy` Attacker route and no universal surviving Defender strategy is proved |
| Binding round-6 exact `M=4`, next `tau=4` | PROVEN | **CONFIRMED UNCHANGED** | Round 7's broader `>=3` bounds do not replace the exact ray-history result |
| Provenance | record | **MINOR ERRATUM** | Authoring input is correct; landed reviewed/output artifact `fbae2f7b` is omitted |
| New machine verification | none | **CONFIRMED NONE** | No Cargo, Lean, harness, search, generated enumeration, or `VERIFIED` claim |

## Overall verdict

**SOUND-WITH-MINOR-ERRATA.** R7.1 is **CONFIRMED** for every hub-containing
pair at each of the six named plateau epochs. The two triangle fans have the
claimed exact census and separation; the complete legal next-pair universe is
captured by the `0/1/2` touched-line taxonomy; every case regenerates demand at
least three and then forces completion.

R7.2 is **CONFIRMED**. The physical action regions `H,G_-,G_+` are pairwise
disjoint for legal empty cells. If an initial action misses `H`, the intact hub
branch has exactly two current focal residuals, all eight servicing orders
remain on `r=0`, and the vertical/diagonal future pair creates three disjoint
demands. If the action touches `H`, one whole fan is untouched and L13.3
applies. This is exhaustive even for pairs split across fans or outside all
regions. Consequently `P_stock` is genuinely **NOT-REPAIRABLE once reached**,
and equation (74) excludes every one-ply tie refinement there.

R7.2.1 is **CONFIRMED**: the first actual plateau ray already loses to the
triangle fan before any stock is placed. L13.6 is **CONFIRMED** at exactly its
initialization scope, including arbitrary lower-stock interactions, its
two-label corollary, and the strict one-label example. R7.3 and L13.5 are also
confirmed.

No formal theorem is refuted or downgraded. The most serious issues are only
**MINOR**: the universal `P_stock` verdict should be attributed to R7.2 rather
than to R7.1's hub-only consequence, and §66.2 should record reviewed/output
artifact `fbae2f7b`. Q2, positive Q3 repair, and GAP-RAW remain open exactly as
the source says.

## Exact unresolved obstacles

1. **Q2 strategy-independent forcing.** No strict root is shown to force
   `P_stock`, the shared hub, the triangle packing, or another losing cascade
   against every Defender strategy. The round-6/7 arrival line is fixed-
   `S_T` only.
2. **First-plateau safe-action decision.** Equation (84) remains unanswered.
   The actual ray has `R_1>=3`, but a pair touching both fan regions has not
   been checked against every legal response; neither `B_1(P_0^pl)<=2` nor its
   negation is proved.
3. **One-policy all-response repair.** No named earlier policy has a proved
   safe action at every reached epoch and closes `M<=2`/`R_1<=2` after every
   legal response. One-ply risk is a diagnostic and conditional assembly
   input, not an initialized and renewed invariant.
4. **Other repair geometries.** Other shared and nonshared `M>2` fanouts,
   the complete axial-cleanup response universe, cross-hull interactions below
   R5.2 separation, nested-derivative next turns, and alternative forced-
   service entrances to the transverse seal remain unclassified.
5. **Remaining count-three initialization.** L13.6 does not cover a strict
   `tau=0` root whose alive count-three residual family has hitting number at
   least three. No universal K3/transversal reduction is supplied.
6. **Renewal and replacement accounts.** The new initialization rung has no
   regeneration theorem, and no amortized credit or other non-dominating
   replacement invariant is both initialized and closed on all histories.
7. **Ancillary separation sharpness.** The minimum support separation for the
   R5.2 value theorem remains unknown; radius 21 is sharp only for its
   conservative all-touched-window envelope.

None of these open obligations weakens the local `P_stock` stop theorem or the
shorter fixed-`S_T` refutation. They are the exact barriers to lifting the
reached-state result to Q2 or completing a positive Q3/GAP-RAW proof.
