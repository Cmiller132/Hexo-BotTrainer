# R-Z11 hostile review: repaired `FHW-T3-R`

## Verdict

**Overall: SOUND-WITH-ERRATA.** I did not find a counterexample to
`FHW-T3-R` on its stated D22/RC/WC annotated class. The repaired
`kappa_cut^*` is a single-valued, disjoint, exhaustive charge function on that
class. The R-Z10 `1+5=6` trace is rejected, the strict boundary is correct,
the RC/WC offsets survive inclusive-boundary attacks, and the first-real-only-
fill induction closes the nested-mask case without assuming mask equality.

There are three errata, none of which changes the recurrence:

1. In the proof of `FHW-T3-R`, the two all-empty bullets must explicitly be
   nested under the preceding "no earlier real-only fill" branch. Read
   literally as independent bullets, "the current placement is the first real
   fill" is false on a nested path with an earlier real-only fill.
2. Section 2.2b enumerates the nine **charge** leaves, but calls them the exact
   verifier leaves without splitting the mandatory touched/all-empty guards
   into pass and fail outcomes. The charge partition is exhaustive; the full
   acceptance partition needs those Boolean guard splits.
3. The proof and design still depend on a pinned 2,011-line authority in the
   `consolidate-main` worktree. That is an explicit conditional dependency,
   not a mathematical gap in the repair, but it does not resolve R-Z10-REV's
   request to land or vendor the controlling D14--D22/T3/T11 source in this
   branch. `BUILD-READY` is therefore reproducible only where that exact
   external authority is available.

No reviewed document was edited. This review is doc-only; no Cargo command,
build, solver, source edit, commit, push, other worktree access, or spawned
session was used.

Reviewed identity:

| artifact | identity |
|---|---|
| branch / commit | `claude/tss-vcf-width` / `4729d70832d4fc33bcce9445dbe75c4d3538805e` |
| `PROOF_TSS_ZONES_FHW.md` | 1,203 lines; SHA-256 `E90C576C610839481A1030555997263EBBDC67676EB60FB38279B122E5D294E4` |
| `DESIGN_GROUP2_NEXT.md` | 1,206 lines; SHA-256 `BFB1E0F182E54B837E75B37C742894299BCDD36B682B71161D3CFD68CED77922` |
| R-Z10 review used as starting arsenal | `PROOF_TSS_ZONES_FHW_REVIEW.md`; SHA-256 `6D6C5EAC109C223B643E8C1DFB4ECCDB6A4D5119A580C7CF8DDEF16BC0582C72` |

## 1. Independent reconstruction before section 2.2b

I reconstructed the partition from the R-Z10 repair certificate and the
definitions of D22, RC, WC, and `kappa_cut^*`, before reading section 2.2b.
For one fixed mapped edge `(d,s)` and one fixed length-six target `W`, let

```text
q = Q^cut_{C_phi(d)}(W).
```

The independent decision order is forced by the proof obligations:

1. Stop if `W` is non-D-alive. An opponent stone is permanent, so D can never
   complete `W`; incidence and cuts are irrelevant.
2. Otherwise split exact/global-FC from genuine non-FC. Exact is included in
   the first side even though exactness also makes FC automatic.
3. Split real incidence `d in W` from `d notin W`. This must precede every
   target-local zero: a real incident placement changes the real D-count by
   one in all edge classes.
4. For the required verifier check, split the D-alive ghost-parent window into
   touched (`cnt_D>0`) and all-empty (`cnt_D=0`). D-aliveness excludes an
   attacker stone, so these are exhaustive.
5. Only for genuine non-FC, nonincident, all-empty targets, split the integer
   child clock into `q<6` and `q>=6`.
6. Only on the latter side query WC; split pass/fail.
7. Independently cross the role axis with RC pass/fail and the ghost incidence
   `s in W`/`s notin W`. Neither may overwrite the real-incidence decision.

This gives the following charge partition. The check column is part of
annotation acceptance, not another value of `kappa_cut^*`.

| domain leaf | charge | acceptance check |
|---|---:|---|
| non-D-alive | `0` | permanent stop |
| D-alive exact/FC, `d notin W` | `0` | no direct window guard |
| D-alive exact/FC, `d in W`, touched | `1` | retained touched guard |
| D-alive exact/FC, `d in W`, all-empty | `1` | `1+q<6` |
| D-alive non-FC, touched, `d notin W` | `0` | none beyond the child clock |
| D-alive non-FC, touched, `d in W` | `1` | N-touch |
| D-alive non-FC, all-empty, `d in W` | `1` | `1+q<6` |
| D-alive non-FC, all-empty, `d notin W`, `q<6` | `0` | arithmetic stop |
| same with `q>=6`, WC pass | `0` | finite cut certificate |
| same with `q>=6`, WC fail | `1` | N-virgin |

The table is disjoint. It is exhaustive because the successive predicates are
Boolean except `q`, whose nonnegative integer domain is partitioned by
`q<6`/`q>=6`. The final `otherwise` in the displayed definition is exactly
the last row, not a hidden additional case.

The four real/ghost incidence pairs recompute as follows, with
`c=cnt_D(W,P_Q)`:

| `(1[d in W],1[s in W])` | real post-count | ghost post-count | inequality |
|---|---:|---:|---|
| `(0,0)` | `c` | `c` | equality |
| `(0,1)` | `c` | `c+1` | real is lower by one |
| `(1,0)` | `c+1` | `c` | direct unit is necessary |
| `(1,1)` | `c+1` | `c+1` | equality |

Thus `cnt_real<=cnt_ghost+1[d in W]` in all four cases. Exact edges can only
take the diagonal pairs; genuine FC/non-FC substitutions can take all four.

### Comparison with section 2.2b

Section 2.2b has the same predicate order and the same nine coarsened charge
rows (it combines the two exact/FC direct guard rows in its display). It also
records the RC and `s`-incidence axes independently. There is no missed charge
case and no overlap analogous to R-Z10.

The sole enumeration erratum is terminological: the table is complete for
**charges**, but not literally the complete verifier-outcome tree. Each row
with N-touch or `1+q<6`, and the WC-fail row with N-virgin, has an accepting
and rejecting outcome. Section 1.2 itself asks that those outcomes be
enumerated. Adding the pass/fail leaves makes the acceptance tree explicit;
it does not alter `kappa_cut^*`.

**Verdict on the decision tree: SOUND-WITH-ERRATA.** The function and its
charge completeness are sound; only the self-review's claim about the exact
acceptance leaves is overstated.

## 2. Adversarial arithmetic

Hex distance below is
`dist((q,r),(q',r'))=max(|dq|,|dr|,|dq+dr|)`. Unless a full replay is shown,
an instance is a verifier-level local row test: omitted setup stones can be
placed on a support chain away from the named danger cell, and the stated
child clock can be realized by ordinary AND opportunities. Such a local test
can falsify an accounting row, as R-Z10 did, but is not represented as a full
D9 certificate.

### A0. R-Z10's original `1+5=6` certificate

The reachable prefix is:

```text
D (0,0)
A [(5,0),(6,0)]       D [(0,2),(2,0)]
A [(7,0),(8,0)]       D [(-2,2),(1,2)]
A [(5,1),(6,1)]       D [(2,-2),(-1,-1)]
A [(7,1),(8,1)]
```

At the resulting defender FirstStone node, define

```text
U_i = {(q,i):5<=q<=10},
E(U_i) = {(9,i),(10,i)},              i=0,1.
```

The two empty pairs are disjoint. Any transversal needs one cell from each,
so `tau=2=b`; both `s=(9,0)` and `d=(10,0)` are kernel-eligible replies. Take

```text
W={(10,r):0<=r<=5},  d=(10,0),  s=(9,0).
```

The parent `W` is D-alive and all-empty, and `d in W`. FC fails at
`z=(18,0)`:

```text
dist(z,d)=8,
dist(z,s)=9,
dist(z,(8,0))=10,
```

and the other displayed prefix stones are farther from `z`; hence
`z in B_8(d)` but `z notin Lambda(P_Q+s)`.

The child continuation has five W-hazards:

```text
exact child hit:                    (10,1)       -> 1
first later real pair:              (10,2),(10,3) -> 2
second later real pair:             (10,4),(10,5) -> 2
q = 1+2+2 = 5.
```

The real W-count is

```text
parent 0
+ current d                 1
+ child exact hit           1   => 2
+ first pair                2   => 4
+ second pair               2   => 6.
```

The repaired row gives

```text
kappa_cut^*=1,
kappa_cut^*+q=1+5=6,
required guard: 1+5<6, which is false,
Q_Q^cut(W)>=max{b,1+q}=max{2,6}=6.
```

The edge is rejected. The old zero reading reported only five; that reading
is no longer expressible.

**Verdict: SOUND.** The published R-Z10 counterexample is completely blocked.

### Recompute the repair's three named overlap probes

The three section 2.2a probes also check independently.

For R11-A,

```text
W_A={(20,r):0<=r<=5}, d=s=(20,0), q=5.
```

This is exact, all-empty, and direct. Therefore

```text
kappa_cut^*=1[d in W_A]=1,
edge expression=1+5=6,
guard 1+5<6 is false,
actual possible count=0+1+5=6.
```

For R11-B, `d=(30,0)`, `s=(29,0)`, and the shared stone `t=(31,0)` flank
the center. For a relative `(a,b) in B_8(0)`, simultaneous failure of the
balls about `(-1,0)` and `(1,0)` would require opposite axial boundary signs;
the remaining cross-pairings force `|b|=16`. Hence
`B_8(d) subseteq B_8(s) union B_8(t)`, so the edge is genuine FC. With
`d in W_B` and `q=5`, it has the same arithmetic:

```text
kappa_cut^*=1,
1+q=6,
guard 6<6 is false.
```

For R11-C,

```text
d=(40,0), s=(39,0), support=(40,-1), z=(48,0), q=4.
```

The distances are

```text
dist(z,d)=8,
dist(z,s)=9,
dist(z,support)=max(8,1,9)=9,
```

so the edge is genuine non-FC. It is all-empty and direct, giving

```text
kappa_cut^*=1,
1+q=1+4=5<6,
maximum displayed W-count=0+1+4=5.
```

**Verdict on R11-A/B/C: SOUND.** They test exact, genuine FC, and genuine
non-FC direct incidence at the unsafe/safe integer boundary without reviving
the old overlap.

### A1. Fresh safe all-empty direct boundary (`q=4`)

Let

```text
W={(0,r):0<=r<=5}, d=(0,0), s=(-1,0), shared support t=(0,-1).
```

Choose no other shared stone in `B_8(z)` for `z=(8,0)`. Then

```text
dist(z,d)=8,
dist(z,s)=9,
dist(z,t)=max(8,1,9)=9,
```

so the edge is genuine non-FC. Let the child fill `(0,1)..(0,4)`, giving
`q=4`. Arithmetic:

```text
kappa_cut^*=1,
1+q=1+4=5<6  (accept),
real W-count: 0+1+4=5.
```

Cell `(0,5)` remains empty. This confirms that the repair did not move the
safe integer boundary below four.

**Verdict: SOUND.**

### A2. Fresh touched direct rejection

Let `W={(10,r):0<=r<=5}` have shared D-stones at `(10,0)..(10,3)`, so
`c=4`. Take `d=(10,4)` and `s=(9,4)`, with no other support covering
`z=(18,4)`. The nearest shared W-stone is `(10,3)` and

```text
dist(z,d)=8,
dist(z,s)=9,
dist(z,(10,3))=max(8,1,9)=9,
```

so FC fails. Let the child clock be the one remaining fill `(10,5)`, `q=1`.
Then

```text
kappa_cut^*=1,
edge/child hazard=1+1=2,
N-touch: c+1+q=4+1+1=6<6  (false),
actual count: 4 -> 5 at d -> 6 in child.
```

**Verdict: SOUND.** Direct incidence is charged and the touched guard rejects
the completion.

### A3. Fresh touched nonincident zero at its protection boundary

Let `W={(20,r):0<=r<=5}` contain two shared D-stones `(20,0),(20,1)`, so
`c=2`. Take a genuine non-FC transition `d=(40,0),s=(39,0)`, supported from
the left, and let the child use four ordinary opportunities on
`(20,2)..(20,5)`, so `q=4`.

Because `d notin W` and W is touched,

```text
kappa_cut^*=0,
Q_Q^cut(W)>=max{b,0+4}=4,
c+Q_Q^cut(W)=2+4=6.
```

The zero does not declare the window harmless: D21's touched threshold fires
and protects the four empties. Charging the remote transition would produce
five, but is unnecessary for this target because the child already accounts
for every remaining fill.

**Verdict: SOUND.** Direct fills and nonincident transitions do not overlap.

### A4. Fresh all-empty nonincident `q=5`

Let

```text
W={(50,r):0<=r<=5}, shared support=(51,0),
d=(70,0), s=(69,0),
```

and make the transition non-FC with an uncovered cell at `(78,0)`. Let the
child fill exactly `(50,0)..(50,4)`, so `q=5`. Then

```text
d notin W,
kappa_cut^*=0 because q=5<6,
Q_Q^cut(W)=max{2,0+5}=5,
real W-count=0+5=5.
```

There is no sixth defender hazard. This is the boundary that the withdrawn
theorem confused with the **direct** `1+5` case.

**Verdict: SOUND.** The two `q=5` rows differ precisely by incidence.

### A5. Fresh WC pass at `q=6`

Let

```text
W={(80,r):0<=r<=5}, shared W-support=(81,0),
d=(100,0), s=(99,0), transition support=(92,0).
```

An uncovered `z=(108,0)` makes FC fail:
`dist(z,d)=8`, `dist(z,s)=9`, and `dist(z,(92,0))=16`. Give the child six W
fills, so `q=6`. Since `q-6=0`, WC is

```text
GI(G) intersect B_8(d) intersect B_0(W)=empty.
```

Here `dist(d,W)=20`, so `B_8(d)` is disjoint from `W`; WC passes. Arithmetic:

```text
kappa_cut^*=0,
Q_Q^cut(W)=max{2,0+6}=6,
virgin radius=8(Q-6)=0.
```

The six child fills can complete W, but the clock is six and D21 protects W
itself. WC removes only the unrelated transition unit; it does not erase the
six child hazards.

**Verdict: SOUND.**

### A6. Fresh WC/N-virgin inclusive-boundary rejection

Let

```text
d=(110,0), s=(109,0), shared support t=(102,0),
W={(118,r):0<=r<=5}, z=(118,0).
```

The edge is legal and non-FC at the exact radius-eight boundary:

```text
dist(d,t)=8,
dist(s,t)=7,
dist(z,d)=8,
dist(z,s)=9,
dist(z,t)=16.
```

Thus `z in GI(P_Q+s) intersect B_8(d) intersect W`. With `q=6`,
`B_{8(q-6)}(W)=B_0(W)`, so WC fails. The fallback computes

```text
kappa_cut^*=1,
kappa_cut^*+q=1+6=7,
N-virgin radius=8(1+q-6)=8,
dist(d,W)=8,
required 8>8  (false).
```

This strict failure is correct: `d` makes `z` legal at inclusive distance
eight, after which `z` and the other five W cells consume the six child
hazards and complete W.

**Verdict: SOUND.** Both the `q=6` zero-radius WC boundary and the inclusive
N-virgin boundary have the correct off-by-one.

### A7. Fresh nested first-real-only-fill attack

Let `W={(140,r):0<=r<=5}`. At an ancestor mapped edge use

```text
d_0=(140,0), s_0=(139,0), shared support=(140,-1).
```

At a descendant mapped edge use

```text
d_1=(140,1), s_1=(139,1),
```

and let its child fill `(140,2)..(140,5)`, so `q_1=4`. The two ghost replies
are outside W, so the ghost parent still calls W all-empty at the descendant;
the real path already contains `d_0`. Both transitions can be made non-FC,
for example at `(148,0)` and `(148,1)` respectively: each is at distance
eight from its real reply and at least nine from the corresponding ghost
reply and the named support.

The descendant's local row alone is safe relative to its ghost parent:

```text
kappa_1+q_1=1+4=5,
descendant guard 5<6  (pass).
```

But its paired clock is `Q_1=max{2,5}=5`. The earliest-fill ancestor sees

```text
kappa_0+Q_1=1+5=6,
ancestor guard 1+5<6  (false).
```

The actual real count agrees:

```text
d_0  1
d_1  1
child 4
total 6.
```

This is the best attack on the new induction. It fails because the proof does
not reset at the descendant's ghost count; the ancestor envelope contains the
descendant's paired direct charge and child clock.

**Verdict: SOUND-WITH-ERRATA.** The invariant is sound. The prose should make
the later all-empty bullets syntactically subordinate to the no-earlier-fill
branch, because `d_1` is not literally the first real fill here.

### A8. Fresh overlapping RC/WC outcomes on one edge

Reuse A6's transition and window, and add a role carrier `y=(200,0)` with
child role clock `k=1`. RC asks for

```text
GI(G) intersect B_8(d) intersect B_{8(k-1)}(y)
= GI(G) intersect B_8(d) intersect B_0(y).
```

Since `dist(d,y)=90`, the intersection is empty and RC passes, so
`epsilon_cut=0` for that role. Simultaneously A6 has WC fail and
`kappa_cut^*=1` for W. The edge therefore has

```text
role expression:   0 + f_child(rho),
window expression: 1 + Q_child(W)=1+6=7.
```

**Verdict: SOUND.** RC and WC are target-specific independent axes; a zero on
one cannot overwrite the charge on the other.

### A9. Fresh genuine-FC direct boundary

Let

```text
W={(220,r):0<=r<=5}, d=(220,0), s=(219,0),
shared support t=(221,0), q=5.
```

For any relative axial cell `(a,b)` in `B_8(d)`, at least one of its distances
to `s=d+(-1,0)` and `t=d+(1,0)` is at most eight. Failure on both sides would
require incompatible opposite boundary signs (or `|b|=16`). Hence
`B_8(d) subseteq B_8(s) union B_8(t) subseteq Lambda(P_Q+s)`, so the genuine
substitution is FC. Nevertheless `d in W`, and

```text
kappa_cut^*=1,
1+q=1+5=6,
all-empty direct guard 6<6  (false).
```

**Verdict: SOUND.** Global FC removes frontier divergence, not a real direct
window fill.

### A10. Non-D-alive and degenerate-boundary check

Let `W={(240,r):0<=r<=5}` already contain an attacker stone at `(240,0)`, and
let a mapped defender reply be `d=(240,1)`. The direct incidence is real, but
W is not D-alive and can never be D-completed because stones are permanent.
The first row returns zero. This does not contradict the repaired headline,
which correctly says every **D-alive** direct fill costs one.

There is no additional geometric degeneracy: a game window is six distinct
contiguous cells, so any existing defender stone in a touched W is within
distance at most five of every remaining W cell. That is exactly the
monotonic legality fact used by the touched row and by the earliest-fill
induction.

**Verdict: SOUND.**

## 3. Induction and monotonicity audit

The first-real-only-fill proof uses four monotonic facts. All are valid on the
stated game/certificate class:

1. **Stone permanence.** Once the real play has a D-stone in W, that stone is
   never removed. A later A2 filler can make the ghost catch up, but it does
   not remove the real stone.
2. **Length-six geometry.** Every other cell of W is at distance at most five
   from that stone, hence is real-legal thereafter under the inclusive
   radius-eight legality rule. A later nonincident substitution cannot be the
   necessary first causal source for a W fill.
3. **Clock containment.** The earliest ancestor's edge charge is paired with
   its own child before taking the maximum. Ordinary opportunities, later
   incident mapped edges, LOSS remainders, and off-kernel escape floors remain
   inside that child clock. A7 demonstrates the exact nested arithmetic.
4. **Permanent death.** Once an attacker stone enters W, W is non-D-alive on
   every descendant. The zero stop cannot revive.

The proof does **not** need arbitrary equality of real and ghost masks. Before
the first real-only W fill it needs only
`cnt_real<=cnt_ghost`; a ghost-only `s in W` strengthens that inequality. At
the first real-only fill the applicable direct/ordinary envelope opens. After
it, the proof retains that ancestor rather than trying to rebase on a later
ghost parent.

The touched nonincident zero also survives a ghost-only touched stone. Even if
the real copy lacks that particular Y-stone, each ghost-empty W cell is
already ghost-legal and therefore receives the normal fresh searched/
dismissed check; if it is ghost-occupied, A2 consumes a defender opportunity.
The current remote `d` need not receive a W-specific transition charge.

The induction would fail outside these properties: with capture/removal, a
non-contiguous target of diameter greater than eight, a recurrence that took
independent maxima, or a history transition not covered by D22/D17's coupling
cases. None belongs to the theorem's domain.

**Verdict on the induction: SOUND-WITH-ERRATA.** No hidden false monotonicity
assumption was found. The bullet nesting should be repaired as stated above.

## 4. Scope audit

The following hypotheses are load-bearing and are used where required:

| hypothesis | load-bearing use | audit result |
|---|---|---|
| finite D18 unfolding / fixed T10 labels | preserves path-local earliest ancestors while keeping one node clock | held; no cheap-edge/other-child splice |
| D22 annotation on every mapped edge | supplies retained roles, masks, LOSS, horizons, A2/A3, and either FC or transition tests | held in theorem and design class tag |
| exact/global FC | removes only the C3 transition seed; direct incidence remains | held; A9 rejects `1+5` |
| RC | removes one role-transition unit only for its named role | held; A8 shows independence from WC |
| WC | removes one nonincident all-empty window-transition unit only at `q>=6` | held; A5/A6 check pass/fail boundary |
| N-touch / direct all-empty guard / N-virgin | rejects an unsafe mapped real reply outside the representative proof | held in every applicable row |
| unchanged scalar `B`, LOSS bases, escape floor `b`, horizons | covers branches not represented by a continuing mapped child | explicitly retained |
| D-alive target | makes touched/all-empty exhaustive and non-D-alive death permanent | held; A10 checks boundary |

No proof step uses arbitrary-history reasoning. Pure annotated nested D22
paths are covered by the chronological induction. Generic D17 substitutions,
unannotated transitions, and the Phase-2 `SR` recurrence across arbitrary
D17/D22 mixtures remain outside the claim. The proof also does not debit
scalar `B`, claim total-zone shrink, or identify `Q^cut` with the exact global
`max(F+H_W)` trace value.

The remaining authority issue is documentary. Both reviewed files explicitly
make the external SHA-256
`39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B`
load-bearing for D14--D22/T3/T11, while the local checked-in proof ends before
those definitions. This review obeyed the no-other-worktrees constraint and
therefore did not independently re-audit that external file. R-Z10-REV's
mathematical audit of the substitution framework is useful history, but the
new hash is not vendored here. The proper disposition is:

- FHW-T3-R itself: sound conditional on the stated pinned authority;
- claim of a self-contained branch proof: not made and not available;
- portable `BUILD-READY` implementation contract: conditional until the
  authority is landed or vendored.

**Verdict on scope: SOUND-WITH-ERRATA.** The theorem's logical scope is held;
the branch-local reproducibility defect remains.

## 5. Design disposition audit

| R-Z11 design bar | verdict | reason |
|---|---|---|
| FHW-T3 selector — BUILD-READY | **SOUND-WITH-ERRATA** | `FhwExactOrD22` requires a frozen class verdict, stores the decision-tree row, rejects every all-empty direct `1+q>=6`, and falls back when ineligible. It is conditional on the external authority dependency. |
| finite matched FHW clock ratio — BUILD-READY | **SOUND** | `I_FHW` starts from a finite B-bounded uniform target set, propagates identical queries, canonically names/deduplicates windows, freezes eligibility before values, and uses the same keys for old/new sums. |
| matched net-zone `J` — BUILD-READY | **SOUND** | `J_zone` binds certificate, node, position, phase/budget, child plan, roles, horizon, and summary; unmatched keys hard-fail. `J_zone^FHW` is frozen before sizes. |
| H1152 population prevalence — DEAD | **SOUND** | The prior invalid population interpretation is explicitly prohibited. |
| H1152-B benchmark bars — BUILD-READY | **SOUND** | The lexicographic set is now correctly treated only as a fixed-key benchmark; the 100-node rule is a materiality denominator, not a sampling cure. |
| radius-nine constant substitution — DEAD | **SOUND** | No radius-substituted theorem is inferred from telemetry. |
| radius-nine exhaustive replacement — SPEC-FOR-CARGO | **SOUND** | It enumerates every radius-nine reply to the fixed horizon, makes truncation/resource limits inconclusive, and requires zero unchecked branches for PASS. |
| identities/native measurements — SPEC-FOR-CARGO | **SOUND** | Manifest, hashes, horizons, caps, commands, repetitions, Off identity, and failure actions are frozen before results. |
| Exact/FHW/SR materiality/economics — SPEC-FOR-CARGO | **SOUND** | Semantic set/clock reductions and matched node/wall/peak promotion gates are distinct and use frozen cohorts. |

No BUILD-READY bar rests on a broken FHW-T3-R claim. The only qualification is
the already disclosed external authority. All empirical outcomes remain
deferred; the design makes no Cargo-result claim.

## 6. Per-claim verdicts

| claim | verdict | certificate |
|---|---|---|
| `kappa_cut^*` is a function | **SOUND** | Predicate order terminates direct incidence before touched/empty cuts; no row overlaps. |
| `kappa_cut^*` is exhaustive on the stated window domain | **SOUND** | Non-alive/alive, exact-or-FC/non-FC, incidence, touched/empty, integer threshold, and WC are exhaustive. |
| Section 2.2b is the exact full verifier enumeration | **SOUND-WITH-ERRATA** | Exact for charge rows; mandatory guard pass/fail outcomes should be added. |
| every D-alive direct fill costs one | **SOUND** | All exact, FC, and non-FC direct rows charge one; A0/A1/A2/A9 cover boundaries. |
| an all-empty direct edge requires `1+q<6` | **SOUND** | `q=4` passes with total five; `q=5` fails with total six. |
| nonincident all-empty `q<6` may cost zero | **SOUND** | A4 has at most five child fills and no current fill. |
| WC zero at `q>=6` is sound | **SOUND** | A5 preserves all six child hazards; A6 catches the first transition-enabled illegal seed at equality. |
| RC zero is role-local | **SOUND** | A8 has RC pass and WC fail on one edge without cross-overwrite. |
| first-real-only-fill induction | **SOUND-WITH-ERRATA** | A7's local pass is rejected by the earliest ancestor; prose nesting should be explicit. |
| original R-Z10 `1+5=6` counterexample is rejected by construction | **SOUND** | A0 lands uniquely in the non-FC/all-empty/direct row and fails the strict guard. |
| R11-A/B/C overlap probes | **SOUND** | Their exact/FC/non-FC classifications and `1+5`, `1+5`, `1+4` arithmetic check. They are correctly described as verifier-level probes, not full alternate-branch D9 certificates. |
| FHW-T3-R on the D22/RC/WC annotated class | **SOUND-WITH-ERRATA** | No counterexample found; conditional authority and two exposition fixes above. |
| extension to arbitrary D17/D22 histories or mixed-history SR remains OPEN | **SOUND** | Neither proof nor design silently claims the extension. |
| scalar-B debit, logical maximality, and total-zone shrink remain OPEN/unclaimed | **SOUND** | Full scalar and branch floors remain; target-local `Q^cut` is not relabelled exact global capacity. |
| companion per-bar dispositions | **SOUND-WITH-ERRATA** | Prior metric/sampling/R=9 defects are repaired; selector portability remains authority-conditional. |

## 7. Explicit errata list

1. **Proof bullet nesting (`PROOF_TSS_ZONES_FHW.md`, FHW-T3-R C2 proof).**
   Replace the four peer bullets with a top-level split:

   ```text
   If an earlier real-only W fill exists: retain its earliest envelope.
   Otherwise:
     - touched ghost W ...
     - all-empty and d in W: current d is the first real fill ...
     - all-empty and d notin W: all six real fills are in the child ...
   ```

   This matches the actual induction and prevents the false literal reading
   exposed by A7.

2. **Acceptance enumeration (`PROOF_TSS_ZONES_FHW.md`, section 2.2b).** Call
   the displayed table the complete **charge partition**, or add pass/fail
   subleaves for the retained touched guard, N-touch, direct `1+q<6`, WC, and
   N-virgin checks before claiming the exact verifier leaves.

3. **Authority portability (both reviewed files).** Land or vendor the pinned
   2,011-line authority under a repository-relative immutable path and bind
   its digest there. Until then, label the FHW selector
   `BUILD-READY-CONDITIONAL-ON-AUTHORITY-SHA-391974...64B` when the artifact is
   consumed outside the authoring machine.

## Final disposition

**FHW-T3-R: SOUND-WITH-ERRATA on the stated D22/RC/WC annotated class.** The
repair closes the exact R-Z10 hole. None of ten fresh adversarial row tests,
including the nested hidden-fill attack and the inclusive WC boundary,
produces an undercount or an admitted defender completion. The companion's
FHW BUILD-READY bar does not rest on a broken theorem, but remains conditional
on its external normative authority. No conclusion extends to arbitrary
D17/D22 histories, mixed-history SR, scalar-B debit, logical maximality, or
total-zone shrink.
