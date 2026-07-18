# R-G4-REV — Round-6 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`,
`110042e5f8b563aa7c5587543182c15ffcd0afa9`

**Reviewed document:** `GAP_RAW_PROOF_ROUND6.md`

**Named review input:** `R-G4-REV — HOSTILE REVIEW (ultra):
GAP_RAW_PROOF_ROUND6.md — the S_T refutation`

**Artifact's named authoring input:** `aed0fecb`

**Method.** First-principles hostile proof audit. I read, in the required order
and in full, `GAP_RAW_PROOF_ROUND4.md` with its folded errata,
`GAP_RAW_REVIEW_ROUND4.md`, `GAP_RAW_PROOF_ROUND5.md` with binding §47
errata, `GAP_RAW_REVIEW_ROUND5.md`, and then the reviewed round-6 artifact.
I consulted only the needed inherited definitions and completion/service
lemmas in rounds 2–3. I did not read any `STRATEGY_STEALING_*` file as
evidence. Every coordinate, window count, residual, hitting number, legal
support, phase transition, and `S_T` objective comparison below was recomputed
by hand. No Cargo command, Lean build, harness, game/search program, or machine
enumeration was run.

During review authoring, an unrelated concurrent job advanced the shared branch
from `110042e5` to `d0af2ef4`. A final read-only comparison showed no change to
the required round-4/5 corpus or `GAP_RAW_PROOF_ROUND6.md`; this review remains
pinned to artifact `110042e5`, and no concurrent artifact is used as evidence.

**Overall verdict:** **SOUND-WITH-MINOR-ERRATA.** R6.1, R6.2, L12.1–L12.6,
and the Q1/Q2/Q3 status boundary survive. In particular, no lower-`TEMPO`
Defender pair pre-empts the constructed continuation. The complete post-hub
audit strengthens the source: all eight servicing orders have exact
`TEMPO=4`, the actual `S_T` service is `(8,0),(12,0)`, and the next epoch has
exact `tau=4`. The two defects found are non-load-bearing: §53.2 overstates
when the earliest causal warning appears, and §57 omits the reviewed/output
commit.

## Numbered findings

### 1. NOTE — the strict root, R4.7 normalization, and initial `S_T` pair are exact

**Quoted claim:**

> “Define `P_0=(A=empty, D={a_0,a_1,a_2}, Defender, FirstStone)` ...
> `Phi(P_0)=0` ... Therefore the exact initial `S_T` pair is
> `D@(-8,8), D@(-16,8)`.” (§49.1)

**Independent recomputation.** Translating R4.7's rows
`100,130,160` by `(0,-100)` gives exactly

```text
c_j=(0,30j), d_j=(1,30j), a_j=(0,30j+8), j=0,1,2.
```

The root is finite and nonempty because it has three Defender anchors. It is
nonterminal because `A=empty`. Since alive means Defender-free and
Attacker-touched, its alive family is empty and therefore `Phi=0`, `I=empty`,
and `tau=0` exactly.

For every legal ordered Defender pair, `A` remains empty at the handoff, so
`L_23=empty` and the candidate value is exactly `TEMPO=0`. Thus the full
primary-objective candidate set is one value class: all legal ordered pairs.

Using

`d((q,r),(q',r'))=max(|dq|,|dr|,|dq+dr|)`,

no initially legal cell has `q<-8`. At `q=-8`, support from an anchor
`(0,R)` requires `R<=r<=R+8`; the least possibility is supplied by
`a_0=(0,8)` and is `r=8`. Hence the first lexicographic cell is `(-8,8)`.
After it is inserted, `q=-16` can be supported only from that new unique
minimum-`q` cell, and the same inequalities make `(-16,8)` the least second
cell. Both are outside every prospective launch union `V_j`, whose windows
extend only five cells along an axis from a launch endpoint.

Hub pre-emption is already possible as a *two-step* root action: `(5,4)` is
distance five from `a_0`, and `h=(10,0)` is distance five from `(5,4)`. That
ordered pair also has value zero, like every root candidate, but its first
cell is lexicographically later than `(-8,8)`. Thus there is no missed
lower-value root pre-emption; the exact policy rejects the tied action by lex.

The first Attacker endpoint `(0,0)` is at distance eight from `a_0`; after it
is inserted, `(1,0)` is adjacent. Neither placement completes six. This is an
ordinary Defender pair followed by an Attacker pair, with the first placement
inserted before the second is tested.

**Proposed repair:** none.

### 2. NOTE — the stock-turn census has no hidden count-three label

**Quoted claim:**

> “After all placements in (63), every length-six window contains at most two
> Attacker stones. The same is true after every initial segment of (63).”
> (L12.4)

**Independent recomputation.** It is enough to list every nontrivial line
after each complete stock turn; an individual-stone prefix is a subset of the
corresponding final set.

| Stock through | Fixed `q` lines | Fixed `r` lines | Fixed `q+r` lines |
|---|---|---|---|
| diamond only | `q=0:{0,1}`, `q=1:{-1,0}` | `r=0:{0,1}` | levels `0,1`, two each |
| `U^-` | unchanged pairs | `r=0:{0,1,6,7}` | levels `0,1` have two; `6,7` one |
| `V^-` | add `q=10:{-4,-3}` | unchanged | levels `6:{6,10}`, `7:{7,10}` |
| `W` | unchanged | add `r=-3:{10,13}` | add level `10:{12,13}` |
| `V^+` | `q=10:{-4,-3,4,5}` | unchanged | add singleton levels `14,15` |
| `U^+` | unchanged | `r=0:{0,1,6,7,14,15}` | levels `14:{10,14}`, `15:{10,15}` |

On `q=10`, the middle gap from `-3` to `4` is seven, so a length-six interval
cannot meet both pairs. On `r=0`, every six consecutive parameters contain at
most one of the separated adjacent pairs `{0,1}`, `{6,7}`, `{14,15}` (the
boundary intervals such as `1,...,6` still contain only two stones). Every
other listed line has exactly two stones total. These are all three axis
families, so every physical six-window has count at most two at every stock
prefix.

Consequently every stock Defender epoch has `I=empty`; every surviving
`L_23` label is count two; and no forced service can change the plateau
optimization. None of the ten stock cells has `q=0,1` or `q+r=0,1`, so no
shield label or witness trigger is occupied. Defender augmentation can only
delete labels.

The legality table in §51.1 also recomputes exactly. The first-stock support
distances are respectively

`5,4,3,7,4`,

and every second placement is adjacent to its first. All cells are distinct
from the Defender ray and the empty hub.

**Proposed repair:** none.

### 3. NOTE — the attained handoff has all five exact focal labels and no hidden Defender contact

**Quoted claim:**

> “All five windows are Defender-free and have count exactly two. Their common
> hub `h=(10,0)` is empty.” (§51.3)

**Independent recomputation.** The final focal supports are

```text
U^- on r=0, q=6..11:      A at q=6,7
U^+ on r=0, q=10..15:    A at q=14,15
V^- on q=10, r=-4..1:    A at r=-4,-3
V^+ on q=10, r=0..5:     A at r=4,5
W on q+r=10, q=8..13:    A at q=12,13.
```

Their only common cell is `h=(10,0)`, and it is absent from the pre-hub
Attacker set. The old axial-cleanup cells `(-4,0),(2,0)` lie outside both
horizontal focal intervals. The anchors have `q=0`, and the ray has `r=8`
with nonpositive `q`; none belongs to a focal window. The six plateau replies
are even farther left. Thus each focal window is Defender-free with exactly
the two displayed Attacker stones.

The result does not need, and the position does not have, L11.6's isolated
nine-label profile. The additional live labels are retained in Findings 4–5
rather than silently deleted.

**Proposed repair:** none.

### 4. NOTE — the hub pair has exactly two current demands and eight servicing orders

**Quoted claim:**

> “At the returned Defender epoch `P_1`, the imminent family is exactly
> `{U^-,U^+}` ... so `tau(P_1)=2`.” (L12.5)

**Independent recomputation.** Before `(h,h+u)`, no window has count above
two. The two adjacent row-zero triggers are contained together in exactly the
five row-zero windows with starts `6,7,8,9,10`. Their old counts are

`2,1,0,1,2`,

so afterward their counts are

`4,3,2,3,4`.

No off-row window contains both triggers, and no pre-count-three label exists.
Thus a label receiving only one trigger reaches at most count three and the
complete imminent family really is the two endpoint windows. Their residuals
are

```text
E(U^-)={(8,0),(9,0)}
E(U^+)={(12,0),(13,0)}.
```

They are disjoint, proving exact `tau=2`. Therefore `Serv(P_1)` has exactly
the following eight ordered members:

```text
(8,0),(12,0)   (8,0),(13,0)
(9,0),(12,0)   (9,0),(13,0)
(12,0),(8,0)   (13,0),(8,0)
(12,0),(9,0)   (13,0),(9,0).
```

Each is sequentially legal: the first cell lies in its live imminent window,
and after that window is killed the other imminent window remains alive to
support the second. There is no spare, no pair using two cells on one side,
and no off-row servicing alternative.

All four possible service cells have `q` in `{8,9,12,13}`, `r=0`, and
`q+r` in the same set. They lie in none of the vertical focal windows
`V^-,V^+` (`q=10`) or diagonal focal window `W` (`q+r=10`). They also do not
occupy either future trigger `(10,1),(11,-1)`. Hence no servicing cell has
the double duty that could kill a `v/w` focal demand.

**Proposed repair:** none.

### 5. NOTE — full post-hub optimization strengthens R6.2 to exact `TEMPO=4` and `tau=4`

**Quoted claims:**

> “Every servicing ordered pair ... is necessarily one of the pairs in (67),
> regardless of how the extra labels affect its primary-value comparison.”
> (§52.1)

> “The full imminent family has hitting number at least three ... Thus
> `M(P_1)>=3`.” (§52.2)

**Independent recomputation.** The source correctly avoids guessing the
post-hub `S_T` order because every service loses. The campaign-standard
every-turn audit can nevertheless finish the exact optimization.

Immediately after the hub pair, the complete count-three family is:

```text
r=0:       starts 5,7,9,11
q=10:      r-starts -5,-4,0
q+r=10:    q-starts 8,9,10.
```

Every service choice from Finding 4 kills all four row-zero count-three
windows: the first two contain either `8` or `9`, and the last two contain
either `12` or `13`. No such service cell lies on `q=10` or `q+r=10`, so
exactly six count-three labels survive every service.

Their pre-trigger residuals, in their one-dimensional axis parameters, are

```text
q=10:
  start -5: {-5,-2,-1}
  start -4: {-2,-1,1}
  start  0: {1,2,3}

q+r=10:
  start  8: {8,9,11}
  start  9: {9,11,14}
  start 10: {11,14,15}.
```

A direct one-axis check bounds the demand generated in either high pencil by
two under any one or two triggers. On the vertical pencil, the only split
singleton trigger is parameter `1`, and all two-trigger cases have a two-cell
cover. On the diagonal pencil, trigger `11` matures all three and leaves the
path `{8,9}`, `{9,14}`, `{14,15}`, of hitting number two; the other cases are
no worse.

The remaining pre-count-two labels contribute at most two under any trigger
pair by the confirmed pure-count-two theorem. If a trigger pair hits only one
high pencil, subadditivity gives at most `2+2=4`. If it hits both high pencils,
a pre-count-two bridge would have to contain the two triggers on their third,
fixed-`r` axis. The only empty same-`r` pairs capable of hitting both high
pencils occur at

```text
r=-5: q=10,15
r=-1: q=10,11
r= 1: q=10,9
r= 2: q=10,8.
```

No length-six row window through any such pair had two old Attacker stones:
the only old row stones at `r=-1` and `r=1` are respectively at `q=1` and
`q=0`, too far away, and the other two rows had none. Thus no count-two bridge
matures in the two-high-pencil case. The global upper bound is four.

The prospective pair

`v=(10,1)`, `w=(11,-1)`

attains that bound. `v` matures the vertical starts `-4,0`, leaving residuals

`{-2,-1}` and `{2,3}`,

while `w` matures all three diagonal starts and leaves

`{8,9}`, `{9,14}`, `{14,15}`.

The vertical and diagonal residual grounds are disjoint because the two
physical axes meet only at the occupied hub. Their hitting numbers add as
`1+1+2=4`. Therefore **every one of the eight service orders has exact
`TEMPO=4`**, not merely a lower bound of three.

The lexicographically least first service cell is `(8,0)` and, after it, the
least allowed second cell is `(12,0)`. Hence the actual pair selected by
`S_T` is exactly

`D@(8,0), D@(12,0)`.

After `(v,w)`, no count-two label can become imminent because the two triggers
are not collinear. The third vertical start `-5` does not contain `v`.
Consequently the five labels displayed above are the **complete** imminent
family and the returned epoch has exact

`tau=4`.

This strictly strengthens the claimed `M(P_1)>=3`, `tau>=3` route. It does not
expose a defect: the three focal residual grounds in (69) are indeed pairwise
disjoint, and the two extra diagonal labels only increase the demand.

**Proposed repair:** optional strengthening: record all six surviving
count-three labels, all eight exact service values, the actual service pair,
and exact `M(P_1)=4`/next `tau=4`. The weaker R6.2 proof is already valid.

### 6. NOTE — the complete raw-epoch minimization gives the inherited axial cleanup

**Quoted claim:**

> “R5.3 applies ... the unique lexicographically first value-zero servicing
> pair is `D@(-4,0), D@(2,0)`.” (§49.2)

**Independent recomputation.** The remote anchors and the first ray pair lie
outside the 31 windows through `(0,0),(1,0)`, so the returned alive family is
exactly `P_raw`'s family. Its `L_23` tier consists of the five count-two
Q-windows with starts `-4,-3,-2,-1,0`. Because `I=empty`, every legal ordered
pair services.

A value-zero pair must hit all five count-two labels. Both effective cells
must lie on `r=0`. Writing only their `q` coordinates, the complete unordered
two-cell cover census is

```text
(-4,2);
(-3,2), (-3,3);
(-2,2), (-2,3), (-2,4);
(-1,2), (-1,3), (-1,4), (-1,5).
```

Including both orders gives exactly twenty ordered value-zero minimizers.
All listed cells are already legal in alive windows. This also clarifies the
source's phrase that remote supports create no “another” value-zero reply:
there are many local minimizers, but no new remote-supported effect class and
no lexicographically earlier minimizer.

For completeness, every non-cover candidate has an exact value determined by
its surviving start set `S`. The five old residuals are

```text
R_-4={-4,-3,-2,-1}   R_-3={-3,-2,-1,2}
R_-2={-2,-1,2,3}     R_-1={-1,2,3,4}
R_0 ={2,3,4,5}.
```

If `S` is empty, `TEMPO=0`. If `S` contains two starts differing by two,
the trigger pairs `{-2,-1}`, `{-1,2}`, or `{2,3}` leave two disjoint residual
grounds, so `TEMPO=2`. In every other nonempty `S`, a trigger pair matures at
most one label or only adjacent-start labels whose post-trigger residuals
share a cell; hence `TEMPO=1`. This exhausts every legal candidate by its
effect on the only `L_23` component.

The earliest first coordinate among the twenty minimizers is `-4`, and then
the only compatible second coordinate is `2`. Thus the actual ordered pair is
`(-4,0),(2,0)`. It kills all seven Q-row labels with starts `-5,...,1` and
lies on none of the four transverse pencil lines, leaving exactly the 24
count-one labels asserted for `Q_ax`.

A hub-containing reply cannot tie here. For example `(2,0)` followed by
`h=(10,0)` is sequentially legal but leaves the start-`-4` count-two label,
so it has value one. More generally, `h` lies in none of the five common
Q-windows and one other empty cell cannot cover all five. Thus every legal
hub-pre-emption pair is strictly worse than the value-zero minimum at this
epoch.

**Proposed repair:** optional wording only: replace “do not create another
value-zero reply” with “do not create an additional value-zero effect outside
the local Q-row covers.” No theorem repair is needed.

### 7. NOTE — the diamond census and four-pencil plateau are exact

**Quoted claims:**

> “At the returned Defender epoch the exact local profile is
> `n_2=20, n_1=20`.” (L12.1)

> “Every legal ordered Defender pair hands over exactly `TEMPO=2`.” (L12.2)

**Independent recomputation.** From the 24 axial-cleanup count-one labels,
`p=(0,1)` promotes five labels on `q=0` and five on `q+r=1`;
`p'=(1,-1)` promotes the complementary five on `q=1` and five on
`q+r=0`. The four old extreme labels remain count one. Each new stone touches
18 windows, ten of which are those promoted labels, leaving eight new
count-one windows. The two new stones share no axis, so the two eight-window
sets are distinct. None contains an old Defender cell. Therefore

`n_2=20` and `n_1=4+8+8=20`,

with no higher label.

The four central shield lines are

`q=0`, `q=1`, `q+r=0`, and `q+r=1`.

The two fixed-`q` lines are parallel, as are the two fixed-`q+r` lines. Their
other intersections are precisely

`(0,0),(0,1),(1,-1),(1,0)`,

all occupied by Attacker. Hence an empty Defender cell lies on at most one
shield line and can kill labels in at most one raw pencil.

For one untouched raw pencil, future axis parameters `-1,2` mature the three
starts `-3,-2,-1` and leave residuals

`{-3,-2}`, `{-2,3}`, `{3,4}`.

The extreme grounds are disjoint and `{-2,3}` hits all three, so this family
has exact hitting number two. Any two Defender cells leave at least two of the
four pencils untouched, proving `TEMPO>=2`; the pure-count-two theorem L10.4
gives `TEMPO<=2`. Thus every legal ordered pair—not just the displayed ray
pair—has exact value two.

The hostile “strictly worse pre-emption” gloss is not the actual comparison.
Once the shield exists, a legal pair containing `D@h` also has value exactly
two. The shield creates a flat value-two plateau that prevents hub pre-emption
from being *better*; `S_T` then rejects it through the defined lexicographic
tie-break. This is exactly the mechanism stated later in §53.2.

**Proposed repair:** none to the artifact. Any campaign summary should call
the shield a value plateau, not a strict primary-objective penalty.

### 8. NOTE — L12.3 derives every plateau reply, including all hub-pre-emption comparisons

**Quoted claim:**

> “The complete sequence of actual `S_T` replies is [the six-row table].”
> (§51.2)

**Independent recomputation.** At a plateau epoch, `I=empty`, so every legal
pair services; Finding 7 puts every candidate in the single exact value class
`TEMPO=2`. If the unique minimum-`q` occupied cell is `(ell,8)`, no legal first
cell has `q<ell-8`. At `q=ell-8`, only `(ell,8)` can supply radius-eight
support, and the distance inequalities make `r=8` least. After inserting that
cell, the same calculation forces `(ell-16,8)` as the second cell. The new
second cell is the next unique minimum-`q` cell.

Starting from `ell=-16`, this gives exactly

| Defender epoch after | Candidate values | Exact `S_T` pair |
|---|---:|---|
| shield pair `(0,1),(1,-1)` | every legal pair: `2` | `(-24,8),(-32,8)` |
| `U^-` stock `(6,0),(7,0)` | every legal pair: `2` | `(-40,8),(-48,8)` |
| `V^-` stock `(10,-4),(10,-3)` | every legal pair: `2` | `(-56,8),(-64,8)` |
| `W` stock `(12,-2),(13,-3)` | every legal pair: `2` | `(-72,8),(-80,8)` |
| `V^+` stock `(10,4),(10,5)` | every legal pair: `2` | `(-88,8),(-96,8)` |
| `U^+` stock `(14,0),(15,0)` | every legal pair: `2` | `(-104,8),(-112,8)` |

Every first ray cell is exactly eight from the preceding leftmost stone, and
every second is exactly eight from the just-inserted first cell.

The hub is already directly legal from the axial-cleanup stone `(2,0)` once
the shield turn is reached, and it stays legal thereafter. At every row in the
table, a hub-containing pair nevertheless has the same value two, not a lower
value. If it shares the actual first ray cell, the next left-ray cell has a
smaller `q` than `h`; otherwise its first cell is itself lexicographically
later. At `P_stock`, for example, both
`(-104,8),(-112,8)` and `(-104,8),h` have value two, and `(-112,8)<h`.
There is therefore no missed lower-`TEMPO` pre-emption at any shield stage.

**Proposed repair:** none.

### 9. NOTE — the complete cadence reaches only the intended final Attacker six

**Quoted claim:**

> “Every arrow is an ordered pair on the ordinary 2:2 cadence ... `S_T`
> loses.” (§52.3)

**Independent recomputation.** Combining the preceding objective audits gives
the complete Defender-turn ledger, including the two turns not explicitly
optimized to an exact pair in the source:

| Defender epoch | `Serv` / candidate census | Exact candidate values | Actual `S_T` action |
|---|---|---|---|
| root `P_0` | every legal ordered pair | all `0` | `(-8,8),(-16,8)` |
| raw adjacent-pair epoch | every legal ordered pair | `0`, `1`, or `2` by Finding 6; exactly 20 ordered value-zero covers | `(-4,0),(2,0)` |
| diamond epoch | every legal ordered pair | all `2` | `(-24,8),(-32,8)` |
| after `U^-` | every legal ordered pair | all `2` | `(-40,8),(-48,8)` |
| after `V^-` | every legal ordered pair | all `2` | `(-56,8),(-64,8)` |
| after `W` | every legal ordered pair | all `2` | `(-72,8),(-80,8)` |
| after `V^+` | every legal ordered pair | all `2` | `(-88,8),(-96,8)` |
| `P_stock` after `U^+` | every legal ordered pair | all `2` | `(-104,8),(-112,8)` |
| `P_1` after hub pair | eight orders in Finding 4 | all `4` | `(8,0),(12,0)` |
| epoch after `(v,w)` | `Serv=empty`, exact `tau=4` | no servicing value | fallback `(-120,8),(-128,8)` |

At the final epoch the unique minimum-`q` stone is still `(-112,8)`, so the
fallback cells are the lexicographically first sequentially legal pair and
are each supported at exact distance eight. They miss all five imminent
labels. Attacker may then play `(10,-2)` and `(10,-1)`, the two untouched
residual cells of `V^-`, completing it on the second placement.

There is no earlier Attacker completion. Before the hub pair every window has
count at most two. The hub pair raises the maximum only to four. Service does
not change Attacker counts, and the noncollinear fanout pair raises the six
surviving count-three labels only to count four. The final first completion
cell produces count five, and only the second produces count six.

Although a Defender six is ignored by the inherited blanket semantics, none
is accidentally formed either. The long ray on `r=8` is spaced by eight; the
three `q=0` anchors are spaced by thirty; and the final row-zero Defender set
is `{-4,2,8,12}`, with at most two stones in a length-six interval. Cross-axis
levels contain at most two of these displayed stones. The ordinary
`FirstStone`/`SecondStone` update is respected on every turn.

**Proposed repair:** optional strengthening only: add the last two exact
`S_T` actions to the source chronology. R6.2 already follows from the
all-service and `tau>=3` arguments.

### 10. NOTE — L12.6 proves exactly the low-only initialization slice claimed

**Quoted claim:**

> “Every alive label has count at most two [implies] `tau=0` and `M<=2`. If
> every alive label has count at most one, then `M=0`.” (L12.6)

**Independent recomputation.** At a finite nonterminal Defender epoch with
alive counts at most two, `I=empty` and hence `tau=0`. A legal ordered pair
exists by the finite nonempty filler construction, and every legal pair
services the empty imminent family. Defender placements only delete labels
and do not change Attacker counts. Thus every candidate handoff has `L_23`
consisting only of count-two labels, so L10.4 gives every candidate
`TEMPO<=2`; minimizing gives `M<=2`.

If every input label has count at most one, Defender augmentation leaves
`L_23=empty` at every candidate handoff. Every candidate value is then zero,
so `M=0` exactly. No potential assumption was used; intersecting this
statewise class with `Phi<1` gives the advertised strict-root corollary.

The source correctly stops at initialization. One later Attacker pair can
create count-three labels, so the lemma proves neither renewal nor the
remaining count-three `tau=0` slice.

**Proposed repair:** none.

### 11. MINOR — §53.2 overstates the “earliest causal warning”

**Quoted claim:**

> “The earliest causal warning occurs at the epoch after the second stock
> turn: once `U^-` and `V^-` have both been built, their unique intersection is
> the still-empty cell `h`.” (§53.2)

**Independent recomputation.** What first appears after the second stock turn
is the specific *two-axis common-hub certificate*: the completed count-two
windows `U^-` and `V^-` have unique intersection `h`.

The unqualified “earliest causal warning” is too strong. Immediately after
the first stock turn `(6,0),(7,0)`, `U^-` already exists, `h=(10,0)` lies in
it, and `h` is empty and legal—indeed it is only distance three from `(7,0)`.
A Defender strategy with knowledge of the displayed continuation can already
occupy `h` and destroy that focal label. More aggressively, after the axial
cleanup `h` is directly legal from `D@(2,0)` even before focal stock is
installed, although at that point there is not yet a structural hub
certificate.

This wording has no effect on R6.1 or R6.2. The proof needs only the actual
fixed-`S_T` replies, and those remain the exact plateau ray replies.

**Proposed repair:** replace the quoted sentence with: “The first epoch at
which the two-axis common-hub certificate is present occurs after the second
stock turn: `U^-` and `V^-` then have the unique empty intersection `h`.”

### 12. NOTE — §53.2's Bellman diagnosis is exact and appropriately limited

**Quoted claims:**

> “Two particular kinds of pair therefore tie in `S_T`'s entire primary
> objective.” (§53.2)

> “No claim is made that a hub-first pair has `R_1<=2` against every response.”

**Independent recomputation.** At `P_stock`, every legal ordered pair has
exact post-handoff `TEMPO=2`, so the ray reply and every hub-containing reply
really do tie in the entire primary objective. The concrete pair
`(-104,8),h` is sequentially legal. After their common possible first cell
`(-104,8)`, the ray cell `(-112,8)` precedes `h`, so the actual policy selects
the losing ray solely through its stated lexicographic rule.

The response `(h,h+u)` returns `M=4` by Finding 5, strengthening the source's
diagnostic `R_1>=3` to `R_1>=4` for the actual ray action. This proves the
state distinction: immediate `TEMPO` cannot distinguish the two actions at
`P_stock`.

The source does not infer that hub pre-emption survives all responses, that
minimizing `R_1` is sufficient, or that either proposed representation is
necessary. Calling the Bellman-risk and latent-certificate ideas design
targets rather than a closed invariant is quantifier-honest.

**Proposed repair:** besides the exact-value strengthening, only the wording
repair in Finding 11.

### 13. NOTE — Q1 is not leaked into Q2, Q3, or GAP-RAW

**Quoted claim:**

> “Theorems R6.1–R6.2 do not decide Q2 or GAP-RAW.” (§48.2)

**Independent recomputation.** The constructed root is fixed, `S_T` is fixed
before Attacker chooses the continuation, and every Defender action on that
history is derived from this one policy. The result therefore has exactly the
Q1 content “this policy loses from this strict root.” It does not supply the
Q2 counterroute

`exists P_0, for every Defender strategy S, exists an S-consistent alpha`,

because a different strategy can legally occupy `h`—in particular at
`P_stock`, and even earlier as Finding 11 notes. Nor does refuting one policy
negate the positive GAP-RAW/Q3 order

`for every P_0, exists S, for every alpha`.

The executive section, §53.1, §53.3, and the ledger consistently keep Q2,
`GAP-CASCADE-REACHABILITY`, positive universal repair, and GAP-RAW open. The
round-5 erratum that hub reachability is not the sole gate is honored by the
explicit list of other fanouts, cross-hull interactions, nested derivatives,
alternative seal entrances, axial-cleanup responses, and count-three
initialization states. L12.6 is correctly tagged Q3-initialization only.

**Proposed repair:** none.

### 14. MINOR — provenance omits the reviewed/output artifact

**Quoted claim:**

> “Input commit: `aed0fecb` on branch `hunt/gap-raw`. This authoring pass
> creates no commit; the orchestrator commits the artifact.” (§57)

**Independent recomputation.** The named authoring input can remain, but this
review examines the committed output at
`110042e5f8b563aa7c5587543182c15ffcd0afa9`. Round-4 and round-5 review errata
already established the convention that input/base and reviewed/output
identifiers should both be recorded. Section 57 repeats the old omission.

**Proposed repair:** add: “Reviewed/output artifact: `110042e5`.”

## Per-theorem and shield-lemma verdicts

| Claim | Source status | Review verdict | Disposition |
|---|---|---|---|
| R6.1 fixed-`S_T` shared-hub assembly | Q1, PROVEN | **CONFIRMED** | Exact `Phi=0` root; exact initial and raw-cleanup minimizers; every plateau candidate value is two; all five focal labels arrive count two and Defender-free with empty hub |
| R6.2 fixed-`S_T` forced loss | Q1, PROVEN | **CONFIRMED-AND-STRENGTHENED** | Exact current `tau=2`; all eight services have `TEMPO=4`; actual service is `(8,0),(12,0)`; next exact `tau=4`; fallback misses a completable label |
| Fixed `S_T` as a universal GAP-RAW/tempo-repair witness | REFUTED | **CONFIRMED REFUTED** | One strict root and one legal `S_T`-consistent continuation suffice to refute this policy candidate, but not GAP-RAW |
| L12.1 exact diamond profile | Q1, PROVEN | **CONFIRMED** | Twenty old promotions, four old extremes, and sixteen distinct new count-one labels give exact `(n_1,n_2)=(20,20)` |
| L12.2 four-pencil plateau | Q1, PROVEN | **CONFIRMED** | Every legal pair has exact value two; hub pre-emption ties rather than improving the objective |
| L12.3 left-ray tie-break | Q1, PROVEN | **CONFIRMED** | Unique minimum-`q`, radius-eight inequalities, and sequential insertion force all six displayed ray pairs |
| L12.4 no-hidden-high audit | Q1, PROVEN | **CONFIRMED** | Complete fixed-`q`, fixed-`r`, fixed-`q+r` census at every stock prefix; maximum window count two |
| L12.5 complete hub-pair imminent audit | Q1, PROVEN | **CONFIRMED** | Only `U^-`,`U^+` are current; exact residuals `{8,9}` and `{12,13}` on `r=0` give `tau=2` |
| L12.6 low-only initialization | Q3-init, PROVEN | **CONFIRMED** | Counts at most two imply `tau=0,M<=2`; counts at most one imply `M=0`; no renewal conclusion |
| §53.2 immediate-`TEMPO` decision failure | Q3 structural partial, PROVEN | **CONFIRMED-WITH-MINOR-WORDING** | Exact value-two tie and losing lex choice are correct; “earliest causal warning” needs narrowing |
| Q1/Q2/Q3 quantifier contract | stated boundary | **CONFIRMED** | Fixed-policy failure is never promoted to strategy-independent forcing or a GAP-RAW refutation |
| Provenance | record | **MINOR ERRATUM** | Input is named; reviewed/output artifact `110042e5` is omitted |

## Audit of the authoritative §55 ledger

| Ledger row | Source status | Review verdict / exact scope |
|---|---|---|
| GAP-RAW | OPEN | **CONFIRMED OPEN.** Neither a surviving Defender strategy nor a `for every S` Attacker counterroute is proved |
| R6.1 fixed-`S_T` hub assembly | PROVEN | **CONFIRMED** by Findings 1–8 |
| R6.2 fixed-`S_T` forced loss | PROVEN | **CONFIRMED-AND-STRENGTHENED** by Findings 4–5 and 9 |
| `S_T` as universal GAP-RAW / tempo-repair policy | REFUTED | **CONFIRMED REFUTED** at Q1 only |
| Fixed-`S_T` branch of `GAP-HUB-FANOUT-REACHABILITY` | PROVEN | **CONFIRMED**; five focal labels are causally assembled before actual `S_T` occupation of `h` |
| Strategy-independent branch of `GAP-HUB-FANOUT-REACHABILITY` | OPEN | **CONFIRMED OPEN**; another policy may occupy `h` |
| `GAP-CASCADE-REACHABILITY` | OPEN | **CONFIRMED OPEN** at the broader Q2 scope |
| Positive universal initialization/repair | OPEN | **CONFIRMED OPEN** with quantifier order `for every P_0, exists S, for every alpha` |
| `GAP-TEMPO-INITIALIZATION` | OPEN | **CONFIRMED OPEN** beyond L12.6's low-only slice |
| L12.6 low-only slice | PROVEN | **CONFIRMED** |
| `GAP-TEMPO-REPAIR` for some one named strategy | OPEN | **CONFIRMED OPEN**; refuting `S_T` does not refute another policy |
| `S_T` as immediate-`TEMPO` universal candidate | REFUTED | **CONFIRMED REFUTED**; `P_stock` has a losing exact value-two tie |
| `GAP-REPLACEMENT-INVARIANT` | OPEN | **CONFIRMED OPEN** |
| `GAP-AMORTIZED-ABANDONMENT` / credit route | OPEN | **CONFIRMED OPEN**; no credit rule is supplied |
| Canonical global renewal and canonical J | inherited REFUTED | **CONFIRMED inherited boundary**; no `Theta_2<1` route is revived |
| General standalone K3 suppression | OPEN | **CONFIRMED OPEN** for count-three `tau=0` roots |
| L12.1 diamond shield | PROVEN | **CONFIRMED** |
| L12.2 plateau | PROVEN | **CONFIRMED** |
| L12.3 lex ray | PROVEN | **CONFIRMED** |
| L12.4 no hidden high label | PROVEN | **CONFIRMED** |
| L12.5 complete current imminent family | PROVEN | **CONFIRMED** |
| Bellman-risk diagnosis at `P_stock` | PROVEN | **CONFIRMED-AND-STRENGTHENED** from `R_1>=3` to `R_1>=4` for the actual ray action |
| Hub-first robust Bellman risk at most two | OPEN | **CONFIRMED OPEN**; deletion of focal stock is not an all-response theorem |
| Complete axial-cleanup response classification | OPEN | **CONFIRMED OPEN**; one losing continuation is enough for Q1 only |
| Other `M>2` fanouts | OPEN | **CONFIRMED OPEN** |
| Cross-hull interaction closure | OPEN | **CONFIRMED OPEN**; R5.2's separation remains load-bearing |
| Nested-derivative next turns | OPEN | **CONFIRMED OPEN** |
| Alternative forced-service transverse-seal entrance | OPEN | **CONFIRMED OPEN** |
| Minimum R5.2 separation | OPEN | **CONFIRMED OPEN** as an ancillary sharpening question |
| New machine verification | none | **CONFIRMED NONE**; all new evidence is hand recomputation |

The ledger is substantively complete and quantifier-correct. Its only review
qualification is the non-ledger provenance erratum and the §53.2 “earliest”
wording; neither changes a row's status.

## Overall verdict

**SOUND-WITH-MINOR-ERRATA.** The default hostile presumption did not produce a
refutation. R6.1 is **CONFIRMED**: the normalized `Phi=0` root, exact initial
reply, R5.3.1 axial-cleanup handoff, shield plateau, five stock turns, six
actual ray pairs, radius-eight supports, label census, and empty
Defender-free focal hub all recompute.

R6.2 is **CONFIRMED-AND-STRENGTHENED**. The extra labels do not add a current
imminent at the hub-pair epoch and do not give any service cell double duty on
the vertical/diagonal stock. After every service, they increase rather than
decrease the fanout: the actual `S_T` service is `(8,0),(12,0)`, its handoff
has exact `TEMPO=4`, the next epoch has exact `tau=4`, and the exact fallback
ray pair cannot stop completion.

L12.6 is **CONFIRMED** at precisely its statewise initialization scope. The
executive, consequences, and ledger do not leak Q1 into Q2, Q3, or GAP-RAW.

The most severe defects are only **MINOR**: “earliest causal warning” should
be narrowed to the first two-axis common-hub certificate, and provenance must
name reviewed/output artifact `110042e5`.

## Exact unresolved obstacles

1. **Q2 strategy-independent forcing.** No strict root is shown to force this
   hub, or any losing cascade, against every Defender strategy. A different
   policy can occupy `h` on the displayed history.
2. **Count-three initialization.** L12.6 does not settle strict `tau=0` roots
   containing alive count-three labels, so general
   `GAP-TEMPO-INITIALIZATION` remains open.
3. **One-policy all-response repair.** No named replacement strategy is proved
   to preserve a sufficient invariant after every legal response on every
   reached strict-root history.
4. **Other repair classes.** Other shared/nonshared fanouts, cross-hull
   interactions, nested-derivative next turns, alternative forced-service
   sealed entrances, and the complete response universe from the R5.3.1
   axial-cleanup handoff remain unclassified.
5. **Bellman-aware action comparison.** Hub pre-emption deletes this focal
   stock, but its worst-response risk is not bounded; equation (71) is only a
   diagnostic, not a closed policy or invariant.
6. **Ancillary separation sharpness.** The minimum support separation for the
   R5.2 value theorem remains unknown; radius 21 is only envelope-sharp.

None of these unresolved obstacles weakens the fixed-policy refutation proved
in R6.1–R6.2. They are the exact barriers to lifting Q1 to Q2 or completing a
positive Q3/GAP-RAW proof.
