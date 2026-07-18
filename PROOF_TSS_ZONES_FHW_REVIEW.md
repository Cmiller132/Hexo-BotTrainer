# R-Z10 proof audit: `PROOF_TSS_ZONES_FHW.md` and `DESIGN_GROUP2_NEXT.md`

## Method and identity

This is a first-principles proof audit of the two files landed at
`ded361c1e4ae8950ffe5cc7e302db49c92487eef` from input
`7c2706c86a0362f8e9ddff35ddb1e3185fa0670c`. The landing is a direct child of
the input and adds only the two reviewed files. I read the required corpus in
the prescribed order and in full:

1. `docs/PROOF_TSS_DEFENDER_ZONES.md` (899 lines, including its checked-in
   review log);
2. `docs/PLAN_TSS_SOLVER_UPGRADES.md` (831 lines), followed by the cited
   Group-2 round-1/2/3 records and `hunt/r1b-r2` report;
3. `PROOF_TSS_ZONES_FHW.md` (950 lines); and
4. `DESIGN_GROUP2_NEXT.md` (981 lines).

Because both reviewed files make later definitions load-bearing, I also read
the complete 2,006-line rounds-5--7 overlay at external SHA-256
`48D3B0887519681EFF338A6861D81E1E8D4169E86853463EAEDA21DF361118F6`,
the complete `_OPEN_FHW_REPORT.md`, and its binding review repairs. I
consulted the permitted engine rule sources read-only for distance, legality,
cadence, and per-placement termination. This review used hand recomputation
only: no Cargo, Lean, proof harness, or game solver was run, and no commit was
made.

Reviewed identities:

| artifact | SHA-256 |
|---|---|
| `PROOF_TSS_ZONES_FHW.md` | `CFA52D689387D4011061FC3BE50A8EF1661CB488B6B44F7BBF1DDDF1B474A755` |
| `DESIGN_GROUP2_NEXT.md` | `80567469AABD4E2D69DAB75175D5B6CC2E998DC97E55BBD000E64D429B0E2442` |
| checked-in zone proof | `6A9C10ACD67DE242E10B2E60B2AA79ABF5280711EFD9849D7D2876F3BC7CABBC` |
| solver-upgrade plan | `C599102C233BD6286FDE93BD6737A7918058D11CA29FCACA69044D2C7D376CE2` |

## Findings

### 1. MAJOR — the later theorem authority is external to the reviewed branch

> “The checked-in proof corpus at HEAD ends with hostile Round 4 and
> definitions D1–D13/T1–T8,” while a 2,006-line file outside this branch is
> used as a “tightening overlay.” (`PROOF_TSS_ZONES_FHW.md`, §0)

This is an honest disclosure, but it leaves every use of D14–D21, T9–T11,
the revised T3/T4 zone, and checkpoint roles outside the required checked-in
normative corpus. The external copies named by the artifact do hash-match each
other, and their D19–D21/T11 text supports the baseline FHW citations. Their
review log records Rounds 5–7, however, while the living plan describes a
rounds-5–8 authority. Consequently the new proofs are mathematically auditable
conditional on external SHA `48D3…118F6`, but the landed `PROVEN-ON-CLASS`
labels are not self-contained or reproducible from this branch alone.

This is not a counterexample to FHW-T1, FR-T1, or G2-Z1. It is a binding-source
failure: the phrase “unchanged T3/T4 class hypotheses” does not identify a
landed theorem in the required corpus.

**Repair.** Land the controlling later proof and every binding erratum under a
repo-relative immutable identity, reconcile the plan’s Round-8 reference, and
then cite that landed source. Until then, spell the affected statuses as
`PROVEN-CONDITIONAL-ON-EXTERNAL-SHA-48D3…118F6`.

### 2. MINOR — both landed artifacts retain stale pre-landing provenance

> “Landed-hash placeholder: `UNLANDED`.” (both artifact preambles)

The input identity `7c2706c8` is present and correct, but the reviewed files
are now landed unmodified at `ded361c1`. “No commit was made” is accurate only
as history of the authoring campaign, not as current artifact identity.

**Repair.** Record both input `7c2706c8` and landing `ded361c1`, plus the two
file hashes above. Preserve the no-Cargo/no-author-commit statement explicitly
as campaign history.

### 3. NOTE — FHW-O1 is a legal, reachable, nonterminal refutation of the flat debit

> A flat rule reads `3 + 1 + 0 < 6`, while the real continuation attains
> `3 + 1 + 2 = 6`. (`PROOF_TSS_ZONES_FHW.md`, FHW-O1)

The replay checks by hand. Cadence is `D; A,A; D,D; …`. Every coordinate is
new, and the supplied legality witnesses have exact hex distances
`4,1,1,1,1,1,1,4,1,6,8,5,1,1,8`; the split replies have distances `1` and
`8`, and the two shared attacker moves have distance `1`. No setup prefix has
a six: the only long attacker runs are the two five-stone rows, and the real
defender’s longest run is four cells of `W`.

At the gate the ghost defender stones have no D-alive length-six window above
count three, so `own_win_now` is false at `b=2`. In the real branch only
`x=(0,3)` differs, raising `cnt_D(W)` from three to four without making the
position terminal. The complete attacker threat-empty family is

```text
{{a}, {a,(6,-4)}, {b}, {b,(-6,8)}}.
```

The singleton members force every transversal to contain `a` and `b`, hence
`tau=2`. Removing `a` or `b` leaves transversal number one; removing either
other empty leaves both singletons, so the exact extendable kernel is
`K={a,b}`, disjoint from `W`. Both kernel cells are legal. The real defender
may instead play `u=(0,4)` and `v=(0,5)`, both legal already at turn start;
the counts are five and six, and only `v` terminates. Thus pressure does not
make a kernel reply compulsory, and the off-kernel `b=2` branch must remain in
the same branch maximum. This refutes the proposed weakening, not landed T3.

The separate LOSS floor is also necessary, not merely an attained coefficient.
The following rooted replay reaches an ordinary defender FirstStone node `N`
with `b=2` (bracketed pairs are complete turns):

```text
A (0,0)
D [(8,0),(8,8)]       A [(1,0),(2,0)]
D [(8,16),(8,24)]     A [(0,17),(0,18)]
D [(8,32),(0,40)]     A [(0,19),(-1,21)]
D [(1,40),(2,40)]     A [(-2,22),(-3,23)]
D [(-1,0),(-8,0)]     A [(3,0),(4,0)]
D [(0,-8),(8,-8)]     A [(0,-7),(-1,20)]
D [(16,-8),(16,0)]    A [(-2,20),(-3,20)]
```

Every support hop is at most eight (the long defender chains use equality),
all cells are distinct, and the largest same-owner line after the replay is
five for A and three for D. Hence the displayed node and every prefix are
legal, reachable, and nonterminal. Split D's first placement into ghost
`s=(-8,8)` and real `x=(3,40)`. Both are legal and nonterminal; in
`W={(q,40):0<=q<=5}` they leave counts three and four respectively. Both
copies then play the singleton gate hit `h=(5,0)`, after which A plays
`(3,20),(0,20)` and the existing three-pair obstruction reaches a `LOSS(2)`
leaf. Those shared moves do not touch `W`. In the real remainder D may play
`(4,40),(5,40)`, taking `W` from four to five to six and winning on the second
placement.

The correct target clock is therefore
`1 + max{1, 0+2} = 3`: the leading one is the ordinary `x`, the gate retains
its off-kernel floor one, and LOSS contributes two. From the initial count
three, `3+3=6` searches `x`. If only the LOSS base is deleted while the
off-kernel floor is retained, the clock becomes
`1 + max{1, 0+0} = 2`; `3+2=5` omits `x`, although the displayed real line
uses exactly `x` plus the two LOSS placements to complete `W`. Thus both
floors are independently mandatory. This also rules out explaining either
floor as an artifact of the other debit. In this annotated position `x` is
neither a live role nor a checkpoint cell; the future role and `h` are already
legal at `N`, and every other incident target through `x` has count at most
three. The weakened clock therefore does not recover `x` through a different
zone component.

The four local ratios also recompute exactly:

| row | old | new | old/new |
|---|---:|---:|---:|
| strict-debit `W` | `1+2=3` | `max{1,0+2}=2` | `3/2 = 1.50x` |
| dual-purpose `W'` | `3` | `max{1,1+2}=3` | `1.00x` |
| disjoint-hit escape | `2` | escape floor `2` | `1.00x` |
| FHW-O1 | `1+2=3` | `1+2=3` | `1.00x` |

For the first row, `cnt_D(W)=3` changes the touched test from `3+3=6` to
`3+2=5`, so that component drops the three window empties. This is only a
local component calculation. The artifact correctly makes no total-zone
shrink theorem and does not reuse the unrelated `62/478` or `18/479` figures
as FHW evidence.

**Repair.** Preserve the branch maximum, the off-kernel `b` floor, and the
`LOSS(b)` base. Add the isolated LOSS replay above to the normative worked
examples; the currently displayed FHW-O1 proves the off-kernel floor directly
but does not by itself isolate deletion of the LOSS base.

### 4. NOTE — FHW-T2 and the RC/WC geometry close C1, C2, and C3 on their stated class

> “A frontier-covered genuine substitution pays zero role-transition units
> and only the direct incidence `1[d in W]`; a non-frontier-covered
> substitution pays D22-N’s full transition unit.” (FHW-T2)

No unconditional substitution debit is present. For an FC edge,
`B_8(d) subseteq Lambda(P_Q+s)` makes every ghost-empty cell newly reachable
through real `d` already ghost-legal; a ghost-occupied real-empty cell is the
A2 cancellation case. This supplies C3. Direct avoidance of every reachable
role supplies C1. The 18 windows through `d`, the real/ghost incidence
inequality

```text
cnt_real(W) <= cnt_ghost(W) + 1[d in W],
```

and the touched/virgin inequalities supply C2. On FC failure, D22-N restores
the transition-inclusive radius and window unit rather than borrowing the
landed D17 theorem under weaker hypotheses.

The target-local cuts have the right off-by-one arithmetic. If `k` child
opportunities remain, the first child seed `z` consumes one, leaving at most
`k-1` radius-eight links to a role; this is exactly RC’s
`B_{8(k-1)}(y)`. If `q` target-window hazards remain and six must fill `W`,
at most `q-6` can be approach relays; this is WC’s `B_{8(q-6)}(W)`. A touched
window needs only direct incidence because all of its empties are already
ghost-legal. Each zero is target-local, while the same edge may still pay one
for another role or window. Escape and LOSS floors, full D14 `B`, absolute
horizons, and checkpoint roles remain unchanged.

The verifier index is finite: `K(Q)`, the descendant role set, the 18 direct
windows, and `B_8(d)` are finite; every further cut intersects that 217-cell
ball and a bounded clock ball. I found no hidden use of zone monotonicity or a
path-dependent folded-DAG label. FHW-T2 remains only
`PROVEN-ON-ANNOTATED-CLASS`; logical maximality and generic zero-cost D17
substitution remain open. RC and WC themselves are sound sufficient cuts, but
FHW-T3 does not turn them into a well-defined charge, as Finding 5 shows.

**Repair.** After the binding source is landed, make the verifier's D22 class
tag and exact pre-checkpoint role union explicit in the normative grammar, and
apply Finding 5's correction before restoring FHW-T3's PROVEN label.

### 5. REFUTED — FHW-T3 is ill-defined and its stated verifier alternative is unsound

> “all-empty W: `q<6` or (WC) permits `kappa_cut=0`.” (FHW-T3 verifier
> alternative)

The displayed definition immediately above contains both

```text
1,  if d in W on a non-FC edge;
0,  if W is all-empty and q<6.
```

These conditions overlap when an all-empty target receives the current
non-FC `d` and the child clock is below six. No first-match convention is
stated, and a mathematical case definition must be single-valued. Worse, the
quoted theorem/verifier summary resolves the overlap to zero and thereby
loses the current direct fill.

Here is a reachable local trace for the unsafe reading. Let D open at `(0,0)`,
then play complete turns

```text
A [(5,0),(6,0)]       D [(0,2),(2,0)]
A [(7,0),(8,0)]       D [(-2,2),(1,2)]
A [(5,1),(6,1)]       D [(2,-2),(-1,-1)]
A [(7,1),(8,1)]
```

Every hop is at most eight and no player has six. At the resulting defender
FirstStone state, let

```text
U_i = {(q,i):5<=q<=10}, with empties {(9,i),(10,i)}, i=0,1.
```

The pairs are disjoint, so `tau=2=b`. Map real `d=(10,0)` to ghost
`s=(9,0)`; both are legal kernel cells. For
`W={(10,r):0<=r<=5}`, the parent target is all-empty. FC fails: for example
`(18,0)` lies in `B_8(d)` but outside the ghost legal frontier after `s`.
At the `b=1` child, both copies play `(10,1)` exactly. After shared attacker
fillers `(3,2),(4,2)`, pair real `(10,2),(10,3)` with ghost
`(9,-1),(11,-1)`; after shared `(5,2),(6,2)`, pair real
`(10,4),(10,5)` with ghost `(8,-2),(12,-2)`. All placements are legal and no
prefix is terminal until the real `(10,5)` completes `W`. The ghost remains
nonterminal, and its certificate line can then complete the attacker row with
`(7,2),(8,2)`.

Thus the child target clock is `q=5`: one exact child hit plus four ordinary
opportunities. The current real `d` is the sixth fill missing from the unsafe
summary. The trace does not by itself annotate every alternate kernel branch
of a complete D9 certificate; none is needed to show that the displayed
piecewise definition is not a function. It independently confirms that the
summary's selected zero has the wrong local capacity.

The sound calculation is `1+q=6`: charge `kappa_cut=1` and apply N-virgin.
The quoted `q<6` row charges zero and reads only five. Its proof sentence “If
`q<6`, six fills are impossible” is true here only after assuming
`d notin W`. The independent `b=2` escape floor gives only
`max{2,0+5}=5`, so it does not recover the missing unit. Thus the theorem is
not merely build-ambiguous: one of its stated
verifier rules admits the displayed completion. FHW-T3 is REFUTED as written.
The repaired, disjoint definition has no counterhistory in this audit.

The short FHW-T2 proof also writes one common pre-count `c`. In a nested
real/ghost coupling the self-contained argument must instead split at the
first earlier real-only `W` fill. If one exists, its ancestor already charges
it; if none exists, mask identity supplies the common pre-count and direct
incidence handles the current edge. That induction repairs the exposition
without changing the formal clock.

**Repair.** Replace the quoted row by:

```text
all-empty W, d in W:      kappa_cut=1; require 1+q<6,
all-empty W, d notin W:  q<6 or (WC) permits kappa_cut=0;
                          otherwise use (N-virgin) and kappa_cut=1.
```

Add `d notin W` to both all-empty zero clauses, add the explicit `d in W`
row, and make every case mutually exclusive. Add the first-real-only-fill
induction to FHW-T2's C2 proof as well. Only then may FHW-T3 recover a
`PROVEN-ON-ANNOTATED-CLASS` label.

### 6. NOTE — FR-T1’s enumeration covers the ordinary/exact-copy class exhaustively

> “The scalar seed band may be replaced by a finite, branch-aware
> support-reach set at ordinary global-zone nodes, including certificates
> with protected exact-copy gates.” (Phase-2 disposition)

The required enumeration architecture is present before the theorem. Its
index is `(role, carrying path, ordinary-node subsequence, support chain)`.
There are finitely many roles and paths after finite DAG unfolding, finitely
many ordinary opportunities on each path, and every chain of at most `h`
links lies in `B_{8h}(y)`, containing `1+3(8h)(8h+1)` axial cells.

The reverse recurrence has every class case:

- deadline and dead-role bases;
- ordinary OR propagation;
- union over exact-copy kernel children; and
- at an ordinary AND, persistence plus one radius-eight predecessor expansion
  only from a support that is ghost-illegal at that node.

`Empty_N` removes a support consumed by a shared filler or attacker move.
There is no cross-child splice: each child set represents a complete carrying
path, and the union is only existential choice among paths. Legality itself
has a single-stone witness, so a first bad occupation always yields one
backward causal chain even if other `X` stones exist. OR and copied-gate moves
add no `X`; an off-kernel escape abandons the role. Thus every real first-bad
history on this class places its first ghost-legal dismissed seed in the
selected child’s `SR` set. The scalar containment adds at most eight per
ordinary opportunity and correctly uses the current-node `-1`.

The strict fragment also checks: `x=(7,0)` and filler `(0,2)` are legal at the
displayed D SecondStone node; shared `(8,0)` is legal at the inclusive
distance-eight boundary; shared `(8,0)` makes `y=(15,0)` legal before the next
defender opportunity. Three defender placements give scalar radius
`8(3-1)=16`, which includes `x`, while backward SR remains `{y}` and
`Legal(P_N) intersect {y}` is empty. This proves strict containment, not a
standalone WIN.

The theorem does not enumerate D17 or D22 substitution transitions, nor a
fresh T6 kernel handoff. The artifact says so: arbitrary D17/D22 mixed
histories remain OPEN. That OPEN label is honest and load-bearing.

**Repair.** Add T6 explicitly to the out-of-class sentence unless a separate
equal-position handoff recurrence is supplied. Do not generalize FR-T1 to
mixed histories without a new product/transition recurrence.

### 7. NOTE — the quiet-turn language and its commutation quotient are complete

> “The complete quiet attacker-turn universe has a finite exact index,
> including newly legal second stones.” (G2-Q1)

At Opening and SecondStone, one placement from `L(P)` is the complete
remainder. At FirstStone, immediate winning `x` gives `[x]`; every nonwinning
`x` has exactly one second placement `y in L(P+x)`. Immediate termination
makes these cases exhaustive. For a finite position, `L(P)` lies in a finite
union of radius-eight balls, each of size `217`.

The dynamic fixture is decisive: after Opening `(0,0)`, `(8,0)` is legal,
`(16,0)` is initially illegal, and it becomes legal at exact distance eight
from `(8,0)`. No prefix can win with at most two stones. Hence a turn-start
pair universe would be incomplete.

The quotient uses commutation only when both distinct endpoints were legal at
turn start and neither singleton wins. Then both orders are legal, produce the
same owner-labelled board, next mover, FirstStone cadence, placement count,
and game outcome. Newly legal second cells remain ordered `FrontierPair`s;
singleton-terminal cases remain ordered `TerminalSecond`s. Order-specific
`last_turn` data is explicitly outside the native TSS search signature, and
materialization still replays one actual legal orientation. This matches the
binding P3 side conditions.

**Repair.** None to G2-Q1/Q2. Retain both orientations only at a cache seam
whose exact ordered covariance is part of its key, as the design already
requires.

### 8. NOTE — G2-Z1 is finite and does not assume zone monotonicity

> `S_{i+1} = S_i union Zone(P,Sigma(S_i))` and a successful fixed point
> licenses the final Universal certificate. (G2-Z1)

For fixed non-opening `P`, `L=Legal(P)` is finite. Every strict generation
adds at least one previously absent member and removes none, so there are at
most `|L|-|S_0|` strict generations. Each frozen D9 child proof is finite.
The uniform selector ranges over finite legal/touched data; FHW queries are
bounded by finite clocks; and each SR set is bounded by a finite role ball.
Thus both the outer inflation and each selected zone computation are finite.

Each intermediate step merely adds exact legal child obligations and freezes
their proofs; it makes no dismissal claim. At termination,
`Zone(P,Sigma(S_*)) subseteq S_*` holds by the empty set difference, and the
same frozen proofs that produced `Sigma(S_*)` are materialized. This supplies
the selected T3/T4 coverage premise. Raw `Zone` may grow, shrink, or lose its
deterministic fallback. The proof needs only inflation of `S_i`, not
monotonicity of `Zone`; old edges remain through union.

Failure to prove a newly required child or a resource exit returns Unknown.
The theorem correctly does not promise success or completeness over alternate
child-certificate choices. Its soundness is conditional on a valid selected
zone class, exact summaries, and final independent verification.

**Repair.** Keep `sound-on-success` and the alternative-proof OPEN caveat.
Land the controlling T3/T4 class source per Finding 1.

### 9. MINOR — the closure summary is ambiguous at a child-root forcing gate

> `Sigma(S_i)` contains `union_{d in S_i} Prot(C_d)`.

The later gate grammar distinguishes incoming `Prot^-(Q)`, which still holds
checkpoint roles through the entry mask check, from post-check `Prot^+(Q)`.
If `C_d` begins at a D19/D22 gate and `Prot(C_d)` is implemented as the latter,
a parent dismissal could occupy a checkpoint carrier and invalidate the
real/ghost threat-mask equality. Design §3.1 separately requires every
checkpoint role, so the intended theorem is safe; the displayed summary is
nevertheless build-ambiguous.

**Repair.** Define the summary directly as the union of all roles live
anywhere in each frozen child proof, expressly using a child-root gate’s
pre-check `Prot^-`.

### 10. NOTE — the PN debts have the claimed one-sided algebra on the stated state machine

> Hidden Choice adds non-selectable `(INF,1)`; Open Universal adds
> non-selectable `(1,INF)`. (G2-PN-OR/G2-PN-AND)

For Hidden Choice, adding non-selectable `(INF,1)` leaves `pn` equal to the
minimum concrete proof number. A concrete forcing proof can still give
`pn=0`; if all represented forcing children refute, concrete `dn=0` becomes
`dn=1`, preventing a false disproof before reveal. The empty vector is
`(INF,1)`.

For Open Universal, adding non-selectable `(1,INF)` leaves `dn` equal to the
minimum concrete disproof number. A genuine refuting child still gives
`dn=0`; if every current child proves, concrete `pn=0` becomes `pn=1`,
preventing a false proof before closure. The empty vector is `(1,INF)`.
Sentinel-clamped addition is necessary because the proof-number infinity is
below the machine integer maximum.

The algebra is not sufficient by itself: reveal/closure events must precede
threshold exits and selection; DepthCutoff’s numeric collision with Refuted
must be distinguished by genuine status; closing plus storing the frozen
summary must be atomic; and no debt may enter a certificate or cache. All of
those hypotheses are stated. Therefore G2-PN-OR, G2-PN-AND, and the one-way
native soundness argument pass only on the declared design class. They are
not implementation evidence.

**Repair.** Retain every listed hard-verdict-site check and the final strict
verifier. Narrow G2-NATIVE’s sentence “every native-PN `pn=0` materializes a
valid certificate”: an internal logical proof may still encounter allocation
or assembly failure. State instead that every externally returned hard WIN
has a successfully materialized, verifier-accepted certificate, and every
materialization failure returns Unknown.

### 11. NOTE — the campaign does not revive the unsound radius-two trim or erase prior OPEN scopes

> FR-T1 is limited to “ordinary global-zone edges and protected exact-copy
> gates,” and “arbitrary D17/D22 mixed histories remain OPEN.”

The binding G1/G3 examples refute the old `r=2` scalar trim because a needed
legality seed can lie beyond that band. Nothing here reinstates it. FHW's
`f` label counts certified ordinary opportunities under the full T3/T11
envelope; FR-T1 replaces the scalar band only after an explicit path/support
recurrence on its smaller class. Its strict example even uses rank three and
proves only that one finite SR set is smaller than the radius-16 scalar set.

The exact-copy recurrence preserves T6's exact extendable kernel and
off-kernel escape. The mixed theorem retains the D17/D22 substitution envelope,
checkpoint roles, absolute D14 horizon, D18 fixed-label DAG unfolding, and
LOSS terms. The artifact also leaves generic zero-cost substitution, arbitrary
mixed-history SR, logical maximality, and total-zone shrink OPEN. Thus there is
no mathematical contradiction with the landed kernel, substitution-envelope,
DAG, or G1/G3 results. The authority defect in Finding 1 remains: the later
versions of several of those controls are not checked into this branch.

**Repair.** Add an explicit dependency/scope table naming T6, D17, D18, D22,
T3/T11, and G1/G3. State both “no radius-two trim” and “no mixed-history SR”
beside FR-T1's theorem label.

### 12. MAJOR — the FHW promotion ratio has no finite matched index

> `1 - sum_{(gate,W)} Q_new(gate,W) / sum_{(gate,W)} E_old(gate,W) >= 0.10`
> with at least 30 eligible protected gates. (design §6.5)

Neither `(gate,W)` nor its deduplication/matching rule is defined. This is
not cosmetic on the unbounded board. A finite position has infinitely many
all-empty length-six windows. At a protected gate with remaining budget
`b>0`, both the old exposure and the new recurrence retain a positive escape
floor for every target. Under a literal all-window index, both sums diverge;
under an implementation-selected query index, the ratio can change according
to which zero-gain or high-gain windows are emitted. “At least 30 gates” does
not make the window index finite.

**Repair.** Before any execution, define a finite canonical index such as
`(certificate_digest, gate_node_id, owner, window_direction, window_origin,
component)`, populated from the uniform verifier's complete eligible target
set. Freeze union/intersection, deduplication, ineligible handling, and the
same-index requirement for old/new values. Report the index cardinality and a
positive finite denominator. Until then the 10% FHW bar is not evaluable.

### 13. MAJOR — H1152's lexicographic prefix is a benchmark, not a prevalence sample

> Canonicalize, sort by full canonical bytes, and “take the first 384” in each
> stratum; H1152 measures “prevalence and zone distributions.” (design §6.1)

The selection is reproducible and outcome-blind, but it is not
population-representative. Conditional on the frozen corpus, the 384
lexicographically earliest keys have inclusion probability one and every
later key has probability zero. Board-byte order can therefore correlate
with geometry and zone size. Requiring 100 accepted nodes per populated
stratum increases a biased sample's size but does not remove the selection
bias. Equal allocation across three strata also cannot yield an aggregate
prevalence without the strata's population weights.

This does not invalidate H1152 as a fixed regression benchmark or its
preregistered denominators. It invalidates population-language and any
inference that the observed distribution estimates human-corpus prevalence.

**Repair.** Either relabel H1152 as a deterministic benchmark, or select the
lowest 384 values of a domain-separated, fixed-seed hash of the full canonical
key using a named algorithm frozen before results. Publish qualifying stratum
sizes and use population weights for any aggregate prevalence; full
enumeration is the strongest alternative.

### 14. MAJOR — the radius-nine adapter is not an independent robustness proof as specified

> The verifier replays the same certificate under `Legal_9` and changes every
> legality-propagation constant from eight to nine. (design §6.3)

The checked proofs establish the production radius-eight game. They do not
state a theorem parameterized by a general legality radius, nor a separate
R=9 theorem for D14, T3/T11, D17/D22, FC/WC, and FR-T1. Replacing every visible
`8` with `9` in a verifier repeats the proposed generalization; agreement with
that verifier is not independent evidence that no radius-specific proof step
was missed. In particular, a zone certificate under the enlarged legal set
must cover every newly legal defender edge, not only the edges selected by an
unproved radius-nine copy of the same formulas.

The design correctly says a failure blocks only the capstone robustness claim,
not production soundness or native promotion. A PASS, however, licenses at
most stress-test telemetry under the present specification.

**Repair.** Prove the relevant zone/coupling lemmas for symbolic `R` (then
instantiate `R=9`), prove them separately for nine, or have the robustness
checker exhaustively enumerate every `Legal_9` reply and verify the resulting
obligations without relying on a radius-substituted zone theorem. Otherwise
rename PASS to a non-proof stress result.

### 15. MINOR — several otherwise coherent bars need their comparison identities frozen

> Exact-zone consumption uses “one fixed final certificate and node index
> `J`”; component gains use an “explicitly named eligible index.” (design
> §6.5)

`J` is not yet tied to a canonical matched node identity, and
`S_uniform(j)`/`S_variant(j)` are not explicitly identified as mandatory
searched sets derived from the same frozen proof summary. If a scheduler or
variant chooses a different child proof, unmatched nodes can change
`G_total=1-X/U`. Similarly, P1/P2/C2/F19 rely partly on historical or plan
profiles without freezing one exact future baseline command, horizon, TT
configuration, and flags-off artifact for every comparison. Recording these
afterward is not the same as preregistering them. The exact/FHW/SR materiality
bars also decide zone consumption without a separate runtime/memory economics
gate; that is coherent only if the decision is expressly semantic/materiality,
not an efficiency promotion.

The displayed arithmetic itself is sound: ratios of sums avoid averaging
small-node percentages; `U>0` protects division; the native P1/P2 2.00x cap,
F19 1.10x aggregate caps, positive-witness rungs, and control treatment do not
contradict each other.

**Repair.** Match `J` by frozen certificate digest, proof-node ID, role set,
and summary digest; define both sets at that same node and publish unmatched
counts as failure. Freeze exact baseline/variant commands and semantic
horizons before implementation results. State whether refined-zone promotion
is a materiality-only decision or add explicit performance/memory bars.

### 16. NOTE — witnesses, mutations, kill criteria, and empirical status are honestly specified

> “executions `0`; observations from this campaign `0`,” and every build or
> measurement item is `DEFERRED-NEEDS-CARGO`. (design §6)

The positive/control witness handling is coherent. P1 is a current strict
witness; P2 is explicitly provisional and must be replayed and strictly
reverified before becoming a bar; C2 UNKNOWN cases are not mislabeled LOSS
oracles. The dynamic second-stone fixture, two-generation closure fixture,
debt fixtures, and frozen-plan/summary witnesses address the new state seams.

The mutation and kill gates cover missed dynamic actions, illegal
commutation, early debt clearing, debt materialization, plan swaps, closure
edge removal/repetition, class/role/horizon/radius disagreement, D6
covariance, resource-to-hard-verdict errors, old mutation regression, and
finder/verifier common mode. The native overhead and zone subset gates have
defined failure actions (`Consume` back to `Shadow`), and radius-nine failure
is scoped correctly. Subject to Findings 12--15 and the external authority in
Finding 1, these are coherent preregistrations.

No sentence reports a new Cargo result. Historical node counts and ratios are
identified as prior evidence used to choose bars; design executions and
observations are explicitly zero. Accordingly the implementation and every
performance/materiality outcome remain DEFERRED, exactly as claimed.

**Repair.** Preserve the zero-execution statement and historical/current
separation. Replace the final unqualified `BUILD-READY` label with
`BUILD-READY AFTER FINDINGS 1, 5, AND 12--15` (or simply `DESIGN-READY;
IMPLEMENTATION DEFERRED`) until the controlling source and measurement indices
are fixed.

## Per-claim verdicts

| claim | disposition | independent basis and exact scope |
|---|---|---|
| Phase-1 flat-debit refutation | **UPHELD — proposed weakening is FALSE** | FHW-O1 is legal/reachable/nonterminal; exact `tau=2`, `K={a,b}`, and the off-kernel `4->5->6` line check. The separate rooted trace gives `3+[1+max{1,2}]=6` versus `3+[1+max{1,0}]=5`, isolating the LOSS floor. Ratios are exactly `1.50x/1.00x/1.00x/1.00x`; no total-zone shrink follows. |
| Phase-1 target-local refinement | **MIXED: FHW-T2 UPHELD ON CLASS; FHW-T3 REFUTED AS WRITTEN** | FC/D22 and the RC/WC geometry cover C1/C2/C3, but `kappa_cut` has overlapping `d in W`/all-empty cases and the theorem's `q<6` verifier row chooses the unsound zero. The repaired disjoint rule is plausible but is not the landed statement. |
| FR-T1 frontier bands | **UPHELD ON ORDINARY/EXACT-COPY CLASS** | Finite path/role/opportunity/support-chain index; exhaustive recurrence; strict rank-three example checks. No conclusion for arbitrary D17/D22 mixtures or a fresh unmodelled T6 handoff. |
| G2-Z1 closure | **UPHELD SOUND-ON-SUCCESS ON STATED CLASS** | Exact quiet children, frozen finite proofs, at most `card(Legal minus S_0)` strict additions, and final containment prove soundness without `Zone` monotonicity. A child-root gate must use pre-check roles, and an invalid FHW-T3 selector is excluded until repaired. |
| Native lambda-squared soundness | **UPHELD ON DESIGN CLASS ONLY** | Quiet enumeration/commutation and one-sided PN debt algebra check. Event ordering, genuine status, atomic closure, no debt in certificates/cache, successful materialization, and strict verification are all load-bearing; no implementation result exists. |
| Design bars and build status | **MAJOR REPAIR; EMPIRICAL CLAIMS HONESTLY DEFERRED** | Witnesses, mutations, core kill gates, and zero-execution wording pass. The FHW metric lacks a finite index, H1152 cannot support prevalence as sampled, radius nine lacks an independent theorem/check, matched `J` and several profiles remain underfixed, and N4 depends on an external normative source. |

## Overall verdict

**REFUTED IN PART / MAJOR REPAIR REQUIRED.** The central flat-debit
refutation, FR-T1 on its explicitly narrow class, G2-Z1 sound-on-success, and
the conditional native state-machine theorem survive first-principles audit.
FHW-T3's claimed verifier theorem does not: its charge cases overlap and its
own prose rule admits a reachable direct-fill plus five-child-fill completion.
The reviewed branch also lacks its controlling later theorem source, and the
design is not fully build-ready because one promotion metric is undefined and
two claimed robustness/population interpretations are unsupported.

No `PROVEN-ON-CLASS` conclusion extends off-class. The radius-two trim remains
unsound; arbitrary mixed D17/D22 SR and total-zone shrink remain OPEN. No
measured evidence was produced or implied by this campaign.

## Exact unresolved obstacles

1. Make FHW-T3's `kappa_cut` cases disjoint (`d in W` must pay one), repair its
   verifier table, and add the first-real-only-window-fill induction before
   restoring the PROVEN label.
2. Land or vendor external SHA
   `48D3B0887519681EFF338A6861D81E1E8D4169E86853463EAEDA21DF361118F6`,
   including every binding erratum, and reconcile the plan's Round-8
   reference. Until then later T3/T11/D19--D22 dependencies are conditional.
3. Supply a branch-product/transition recurrence for arbitrary mixed D17/D22
   support reach. FR-T1 supplies none; mixed histories remain OPEN.
4. Decide whether a fresh T6 kernel handoff is represented as an exact-copy
   recurrence case or explicitly excluded from FR-T1.
5. Prove completeness or a canonical least-fixed-point result across
   alternative child certificates if progress, rather than sound-on-selected-
   proofs, is desired from G2-Z1. The present theorem intentionally does not.
6. Keep generic zero-cost D17 substitution, logical maximality of the danger
   cuts, and total searched-zone shrink OPEN unless separately proved.
7. Define the finite matched FHW `(gate,W)` index, canonical matched net-zone
   node index, exact comparison profiles, and refined-zone economics policy.
8. Replace H1152's lexicographic prefix for prevalence inference or relabel it
   as a fixed benchmark; publish stratum populations for any aggregate claim.
9. Prove a symbolic/R=9 zone theorem or exhaustively verify every newly legal
   radius-nine reply; constant substitution alone is not an independent proof.
10. Clarify child-root `Prot^-`, hard-WIN materialization failure, and exact
    baseline identities before implementation.
11. Implement, mutate, replay P2, and run the preregistered measurements only
    when Cargo execution is authorized. Every such result remains
    DEFERRED-NEEDS-CARGO.
