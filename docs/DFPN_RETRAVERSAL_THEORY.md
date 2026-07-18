# df-pn re-traversal theory: the formal counterpart of R-TS1

**CITED (local repository state).** Research round R-T1, 2026-07-17.  Branch
`claude/tss-vcf-width`, commit `d5a2b5fd94ef5ff75ecbb4b08e80da53e8636832`.

This note separates four statuses rigorously:

- **PROVEN** means proved in this note for the explicitly stated idealized
  model.  It does **not** silently claim that every production policy in
  `WidePnSearch` satisfies the model.
- **SKETCH** means that a construction or argument is given, but at least one
  reduction to the production engine is not discharged.
- **CONJECTURE** means a falsifiable prediction, not a result.
- **CITED** means either a statement supported by a named source or a recorded
  local measurement.  `CITED-FROM-MEMORY` is used when the primary source was
  not verified in this round.

**PROVEN (summary; model scope below).** Even on a depth-two finite tree,
fixed prior scale $H=2$ and fixed increment $\delta=2$ can force
$\Theta(N)$ extra expansions that unit increments avoid.  A second depth-two
family shows that a fixed non-admissible overestimate ratio 2 can likewise
cause $\Theta(N)$ starvation overhead.  A stronger, deeper family gets the
delta result with $H=1$.  Thus neither $\delta$ nor $H$, nor an overestimate
ratio alone, controls total work.  One explicit witness of the missing
structure is the **frontier mass inside the widened score band**.

## 1. Result ledger

| ID | Status | Result | Scope |
|---|---|---|---|
| T1 | **PROVEN** | Persistent progress-certified df-pn makes at most $(d+1)E$ recursive activations; with exact selected-cutoff deepening, $E\le 2N-1$, hence revisits are $O(N(d+1))$ (and $O(Nd)$ for $d\ge1$). | Finite acyclic arena; no capacity stall; definitions in Section 4. |
| F1 | **PROVEN** | A unary staged-deepening family has exactly $N=d+1$, $E=2d+1$, and $d(d+3)/2=\Theta(Nd)$ repeat activations. | Near-matches T1; independent of $\delta,H$. |
| T2 | **PROVEN** | For every integer $q\ge2$ and $M\ge1$, an explicit depth-two Choice tree of prior scale $H=q$ costs 2 post-root expansions with $\delta=1$, but $M+2$ with $\delta=q$. | Exact second-best thresholds and unit child-progress floor; no TT effect needed. |
| T2b | **PROVEN** | The same $\Theta(N)$ coarse-window overhead occurs with **all priors equal to 1** ($H=1$); a long refinement ladder replaces the width-$H$ plateau. | Finite tree; no TT; depth grows with $N$. |
| C2 | **PROVEN** | Consequently, at fixed $\delta=H=2$, the extra expansions can be $N-O(1)$; no universal $o(N)$ overhead bound can depend only on $\delta,H,d$. | Immediate from T2. |
| T3 | **PROVEN** | A true-cost-one winning child whose prior is overestimated from 1 to 2 can be starved behind $M+1$ useless expansions.  The overhead is $\Theta(N)$ at fixed ratio $\rho=2$, $H=2$, $\delta=1$, and depth two. | Explicit tree; deterministic tie order stated. |
| T4 | **PROVEN** | Under unit-calibrated score response, starvation before first selecting a winning child is at most $\sum_{i\ne w}(p_w+\delta-p_i)_+\le(b-1)(H+\delta-1)$.  If $P\le p'_w\le\rho P$, total heuristic-run starvation is at most $(b-1)(\rho P+\delta-1)$; only the analytical envelope, not paired-run work, has increment at most $(b-1)(\rho-1)P$. | Local Choice frontier; calibration is essential; matching cases and the paired-run obstruction are explicit. |
| T5 | **PROVEN** | Competitive-barrier returns are at most $PV/\delta$, where $PV$ is positive variation of the active child score; if scores are monotone in $[1,U]$, this is at most $\sum_i\lfloor(U-p_i)/\delta\rfloor$, near-matched by unit-response arms. | Local call-count theorem, not a total-expansion theorem. |
| T6 | **PROVEN** | A scheduling change with positive extra non-revisit cost can win only if saved revisit cost exceeds that extra cost; revisit-attributed baseline wall is an absolute saving ceiling. | Component accounting identity. |
| D1 | **PROVEN** | After admission saturation, $k$ parents of $M$ unindexed shared DAG children can create $(k-1)M$ duplicate expansions relative to an unlimited exact index. | Abstract admission-only TT model matching the relevant data-structure behavior. |
| E1 | **CONJECTURE** | R-TS1 delta 2 crossed one or more high-mass score bands; the observed expansion growth reached admission saturation, which may then have amplified work through loss of eligibility for indexed transposition reuse. | Consistent with counters, not identified causally by the retained aggregate logs. |

**CITED (novelty boundary; not an exhaustive nonexistence claim).** The checked
literature proves expansion-selection
equivalence, finite-DAG completeness, and a logarithmic *local recursive-call*
bound for multiplicative $1+\epsilon$.  It reports empirical tradeoffs for
additive increments and heuristic initialization.  No source in the checked
corpus gives a total-work theorem jointly parameterized by additive $\delta$,
prior scale $H$, depth, and bounded-TT behavior; Section 3 gives the source map
and its limitations.

## 2. The exactly measured instance

### 2.1 What R-TS1 measured

**CITED (local measurement).** The retained +1 counter run took 499.85 s,
versus an uncontaminated 495.94 s baseline.  Exclusive descent/state traversal
outside expansion was 13.93% of baseline wall.  Proportional attribution by
the visit/revisit ratio assigned 34.80 s, or 7.02% of wall, to revisits
(conventional rounding; the source report's 34.79 s / 7.01% is truncation).  This
was an attribution, not a separately timed revisit bucket
(`HUNT_REPORT_THRESHOLD_SCALE.md:23-34`).  The raw totals were 9,080,708
recursive visits, 4,574,016 revisits, 8,464,552 threshold-cross returns,
8,056,474 reselections, and 6,188,156 sibling switches
(`THRESHOLD_COUNTER_FULL_RAW.log:192`).

**CITED (local measurement).** Delta 2 increased official wall from 499.85 s
to 927.59 s, +85.6% (`HUNT_REPORT_THRESHOLD_SCALE.md:36-44`).  On the hardest
row, `0l4291i_live`, expansions rose from 1,879,611 to 6,054,588 (3.2212x),
wall from 199.0280 s to 627.7455 s, and peak indexed-TT bytes from 549,161,606
to 1,073,741,810 (`THRESHOLD_COUNTER_FULL_RAW.log:40`,
`THRESHOLD_DELTA2_FULL_RAW.log:41`).

**CITED (local implementation status).** The campaign closed null: +1 remains
the production schedule and no production delta flag ships.  Default-off
`cfg(test)` counters and the experimental delta selector remain
(`HUNT_REPORT_THRESHOLD_SCALE.md:3-6`;
`packages/hexfield_eq/rust/src/tss_solver.rs:1979-2000`, `:3308-3318`,
`:3561-3572`).

**CITED (derived exactly from retained counters).** On that row, delta 2 did
reduce *revisit intensity*: revisits per expansion fell from
$1{,}538{,}944/1{,}879{,}611=0.819$ to
$3{,}653{,}962/6{,}054{,}588=0.604$, a 26.3% reduction.  It nevertheless
raised absolute revisits 2.374x and visits 2.840x because expansions rose
3.221x (`THRESHOLD_COUNTER_FULL_RAW.log:42`,
`THRESHOLD_DELTA2_FULL_RAW.log:43`).  This is direct evidence for the
distinction formalized below: coarser windows can reduce re-entry per expansion
while causing much more expansion work.

**CITED (local measurement).** At the small-cap leaf surface, h8 was identical
in verdict count and expansions for all four arms (16 verdicts, 1,852
expansions).  At h16, +1 produced
39 verdicts/6,649 expansions, delta 2 produced 39/6,726, delta 4 produced
38/7,106, and mean-prior delta produced 38/7,135.  Delta 4 and mean each lost
one hard verdict.  The eight raw rows contain $4(16)+(39+39+38+38)=218$
verdicts; all 218 were strict-verified and there were zero contradictions
(`THRESHOLD_LEAF_AB_RAW.log:58-65`).  The “228” at
`HUNT_REPORT_THRESHOLD_SCALE.md:62-63` is therefore an arithmetic typo, not an
additional ten verdicts.

**CITED (local algorithm-class context).** Candidate 2 was posed precisely as
the mismatch between fork PN priors spanning 1..37, tau-derived DN priors, and
`second+1`, with fixed/mean delta to precede any epsilon grid
(`IDEATION_FINAL.md:210-226`).  The broader
sweep already classified standard df-pn as landed and heuristic initialization
as live.  Pure 1+epsilon was not independently nominated and was to be tested
only if local seesaw counters justified it; the 1 GiB index fit made a broad
epsilon sweep unwarranted.  DAG-aware PN was dry on the then-low transposition
evidence (`IDEATION_FINAL.md:360-374`).  R-TS1 is
therefore an exact local test of the literature’s proposed remedy, not a claim
that standard df-pn itself was missing.

### 2.2 Exact engine discipline used by the model

**CITED (code).** `WidePnSearch` uses the saturated standard recurrences:
at Choice, $pn=\min_i pn_i$ and $dn=\sum_i dn_i$; at Universal,
$pn=\sum_i pn_i$ and $dn=\min_i dn_i$
(`packages/hexfield_eq/rust/src/tss_solver.rs:4943-4976`).  Its finite sentinel
is `PN_INFINITY = 1_000_000_000` (`tss_solver.rs:1977`).

**CITED (code).** Ignoring root-only ordering and Universal commitment for the
moment, a selected Choice child $j$ receives

\[
 P_j=\max\{pn_j+1,\min(P,\operatorname{secondmin}_{i\ne j}pn_i+\delta)\},
\qquad
 D_j=\max\{dn_j+1,D\mathbin{\dotminus}(dn\mathbin{\dotminus}dn_j)\}.
\tag{1}
\]

The Universal formula is the dual: second-best-plus-$\delta$ acts on $dn$,
while the conjunctive $pn$ budget is subtracted, where $x\dotminus y$ denotes
saturating subtraction.  The implementation and its own summary are at
`tss_solver.rs:4008-4019` and `:4139-4203`.  A call returns when either live
number reaches its supplied threshold (`:4069-4085`).

**CITED (code); ERRATUM (review Finding 6).** The selected delta value
enters the scheduling equations only through the competitive second-best
term — but the official off-versus-2 A/B is exact only below the
sentinel: any `Some(delta)`, unlike `None`, also clamps inherited PN/DN
thresholds, second-best additions, and the unit progress floors to
`PN_INFINITY`.  Retained `THRESHOLD_FULL_D1_RAW.log` shows `Some(1)`
reproduced the off run's integer structural totals exactly (9,080,708 /
4,574,016 / 8,464,552 / 8,056,474 / 6,188,156), making the clamp
observationally inert in the +1 counters; the retained delta-2 aggregate
has no sentinel-hit counter, so attributing every delta-2 schedule
difference solely to second-best widening requires an identically clamped
control or a sentinel-hit assertion.  This does not change the empirical
fact that shipped +1 beat the tested delta-2 implementation.  The
test-arm plumbing separately
clamps inherited-threshold and increment arithmetic to `PN_INFINITY`
(`tss_solver.rs:3999-4005`, `:4033-4042`).  All test arms retain a unit
child-progress floor, $\min(\text{current}+1,\texttt{PN_INFINITY})$, equal to
current-number-plus-one below the sentinel (`:4154-4169`, `:4173-4203`).  Mean
delta is the integer mean of
immutable nonselected sibling PN priors at Choice or DN priors at Universal
(`:3965-3995`).

**CITED (code).** Fork initialization is
$pn=37-\min(\text{fork degree},36)$, hence 1..37; tau initialization is exactly
`tau.map(u32::from).unwrap_or(1).max(1)`.  See
`tss_solver.rs:2060-2075`, claimant/opponent position
initialization at `:3664-3690`, and completed-turn initialization at
`:3719-3732`.  Unexpanded entries retain immutable priors
(`:3606-3637`, `:4943-4949`).

**CITED (code; scope caveat).** Production selection is not pure numerical
df-pn everywhere.  Root sequential/width-tier rules and Universal commitment
can override the minimum-number child (`tss_solver.rs:4095-4119`,
`:4690-4751`, `:4754-4913`).  The proofs below therefore concern the numerical
core unless a theorem explicitly says otherwise.

**CITED (code; terminology correction).** A recursive `work` entry is a visit;
later entries to the same arena node are counter “revisits.”  Expansion occurs
only while an entry is `Unexpanded`; staged deepening can separately reopen a
`DepthCutoff` (`tss_solver.rs:3827-3893`, `:3921-3939`, `:4053-4068`).  Thus
the measured phenomenon is chiefly **re-traversal/re-entry**, not repeatedly
generating every interior node.

**CITED (code; TT correction).** The wide TT does not evict indexed entries at
the byte cap.  It keeps the arena entry but declines to index a new key, while
already indexed keys still hit (`tss_solver.rs:3606-3612`, `:3621-3651`).
Accordingly this note says **admission saturation and loss of eligibility for
future indexed reuse**, not
“eviction,” for the measured engine.

## 3. Known-results map and the exact gap

| Source | Status and result actually used | What it does **not** establish |
|---|---|---|
| Nagai & Imai, [“Proof for the Equivalence between Some Best-First Algorithms and Depth-First Algorithms for AND/OR Trees”](https://repository.dl.itc.u-tokyo.ac.jp/records/48966) (2002) | **CITED.** Shows that df-pn expands a PNS most-proving node; with retained values and consistent tie breaking, this yields the PNS expansion order in its tree model. | Traversal count, additive widening, heuristic-prior scale, or a saturated bounded TT. |
| Kishimoto & Müller, [“About the Completeness of Depth-First Proof-Number Search”](https://doi.org/10.1007/978-3-540-87608-3_14) (2008) | **CITED.** Proves completeness on finite DAGs with unlimited time and memory and gives a finite cyclic counterexample. | A quantitative completion or re-entry bound. |
| Pawlewicz & Lew, [“Improving Depth-First PN-Search: 1+epsilon Trick”](https://www.mimuw.edu.pl/~lew/files/epsilon_trick.pdf) (CG 2006/volume 2007) | **CITED.** With multiplicative threshold $\lceil(1+\epsilon)s\rceil$, within one retained parent invocation and fixed cap, each nonfinal child call either fires the opposing budget or grows the competitive score by at least $1+\epsilon$; there are therefore $O(\log_{1+\epsilon}P)$ such calls, plus at most one cap-ending call.  The paper explicitly gives up PNS expansion order and reports over-exploration for larger $\epsilon$. | Later parent re-entries or state loss; total expansions/wall; additive $\delta$; non-admissible $H$; or this engine’s admission-only TT. |
| Kishimoto & Müller, [“Search versus Knowledge for Solving Life and Death Problems in Go”](https://webdocs.cs.ualberta.ca/~mmueller/ps/aaai05-tsumego.pdf) (AAAI 2005) | **CITED.** Identifies priors above 1 plus unit increments as a re-expansion problem; uses the mean heuristic initialization as \(\delta\); reports reexpanded/total 45% to 33% and about 21% fewer nodes on harder problems.  It explicitly leaves re-expansion ratio versus execution time for future work. | A bound, monotonic benefit from larger \(\delta\), or parameter transfer across engines. |
| Kishimoto, Winands, Müller & Saito, [“Game-Tree Search Using Proof Numbers: The First Twenty Years”](https://dke.maastrichtuniversity.nl/m.winands/documents/ICGA2012PNS.pdf) (2012) | **CITED.** Surveys standard second-best+1, constant increments, mean-initialization increments, and the multiplicative trick as distinct remedies. | A joint \(\delta,H\) analysis. |
| Zhang, Iida & van den Herik, [“Deep df-pn and Its Efficient Implementations”](https://dspace.jaist.ac.jp/dspace/bitstream/10119/15854/1/23404.pdf) (2017) | **CITED.** Uses depth-shaped initialization $E^{D-\mathrm{depth}}$.  Under its local conditions $E'\le E$ and $\mathrm{depth}+1<D$, an expansion does not increase the relevant aggregate prior and avoids one immediate switch; the reported global reductions are empirical and parameter-sensitive. | A global starvation or total-work guarantee for arbitrary heuristic priors. |
| Winands & Schadd, [“Evaluation-Function Based Proof-Number Search”](https://dke.maastrichtuniversity.nl/m.winands/documents/CG2010pneval.pdf) (2011) | **CITED (PNS initialization context, not df-pn).** Provides empirical evaluation-derived initialization and warns, by design/tuning discussion, that heuristic PN values need not retain the traditional lower-bound interpretation. | An overestimate-ratio bound. |
| Kishimoto & Marinescu, [“Recursive Best-First AND/OR Search for Optimization in Graphical Models”](https://www.auai.org/uai2014/proceedings/individuals/110.pdf) (UAI 2014) | **CITED (adjacent, not df-pn).** Its second-best-threshold AND/OR search empirically shows the same qualitative non-monotonicity: a larger additive allowance reduces re-expansion but can increase CPU by exploring unpromising regions. | A theorem about proof numbers or R-TS1. |

**CITED (literature-audit conclusion, not a claim of exhaustive
nonexistence).** The sources above supply correctness/equivalence results, a
local logarithmic call bound for multiplicative widening, and empirical
mitigations.  This audit found no primary-source theorem bounding total
re-traversal or total expansion as a joint function of additive $\delta$,
heuristic scale $H$, and finite-tree/DAG structure.  The 2005 paper’s explicit
future-work sentence is especially close to the open question answered
partially here.

**CITED-FROM-MEMORY (context only; verify before external publication).** The
familiar \(b^{d/2}\)-flavored scale belongs to minimal proof-tree/minimax-tree
size under alternating uniform branching, not to a theorem about df-pn
re-traversal overhead.  It is therefore not used in any proof below.  Likewise,
the often repeated “df-pn is $O(Nd)$-ish” statement is treated here as a path
accounting idea, not attributed to the PNS literature; T1 states and proves the
precise assumptions under which it is valid.

## 4. Formal model

### 4.1 Idealized numerical core below saturation

**ERRATUM (review Finding 1).** Model M uses a formal $\infty$; the engine
clamps at `PN_INFINITY = 10^9` (`tss_solver.rs:1977`, `:3999-4005`,
`:4033-4042`, `:4959-4969`), so at the sentinel the progress floor is not
strict.  Saturation is therefore an explicit engine exclusion of this
model, alongside the policy exclusions listed below.  Finite-sentinel
realization ranges: T2 transfers to the engine for $2\le q<I$ and
$1\le M<I-1$ (a literal current-engine prior also has $q\le37$); C2 is a
valid formal asymptotic, not an unbounded fixed-sentinel engine trace.

**DEFINITION (model M).** Fix integers $H\ge1$ and $\delta\ge1$.  A search
arena is a finite rooted acyclic AND/OR
graph.  An arena entry is either unresolved with a positive integer prior
$(pn,dn)\in[1,H]^2$, expanded to a branch, or terminal with the conventional
$(0,\infty)$ / $(\infty,0)$ values.  Branch values use the min/sum recurrences
from Section 2.  Expanded structure and backed-up numbers persist.  At a
Choice node the scheduler selects a minimum-$pn$ child and uses (1); at a
Universal node it uses the dual formula.  The minimum of an empty sibling set
is $\infty$.  Arithmetic is exact below a formal $\infty$.  Deterministic tie
order is part of an instance.

**DEFINITION (events).** An **expansion event** is an expansion attempt that
changes an unresolved entry to a terminal or branch or marks it as a too-deep
cutoff.  Reopening cutoff markers between stages is not an expansion event;
the later expansion attempt is.  An **activation** is one recursive entry into
a node.  Its first activation is a visit and every later activation is a **repeat
activation/revisit**.  Let $N$ be the number of nodes in the finite underlying
tree, or distinct semantic nodes in the coherent-index DAG; $E$ is the number
of expansion events, $V$ activations, $R$ repeat activations, and $d$ the
maximum active path length in edges.  Section 9 explicitly separates semantic
nodes from duplicated arena copies after admission saturation.

**DEFINITION (progress-certified run).** A run is progress-certified if every
recursive activation that occurs before the root is solved contains at least
one expansion event in its dynamic extent before returning.  There is no
node/soft cap, no policy-induced stall, and no immediate return from stale DAG
information.  The current+1 floors make this the natural idealization, but it
is an assumption, not a theorem about every production path.

**DEFINITION (exact selected-cutoff deepening).** At depth cap $k$, first
touching an unresolved node deeper than $k$ consumes one expansion event and
marks that entry `DepthCutoff`.  The next stage advances to the exact depth of
the selected cutoff.  Such an entry may be reopened and expanded once more,
but no arena entry is cutoff more than once.

**PROVEN (duality).** Every Choice-side construction and proof below has a
Universal dual obtained by exchanging $pn\leftrightarrow dn$,
Choice$\leftrightarrow$Universal, and proof$\leftrightarrow$disproof.  This
follows by direct substitution in the min/sum recurrences and threshold
formulas.

### 4.2 Parameters that a valid total-work theorem must expose

**DEFINITION (frontier-band mass).** For a selected child $x$, sibling score
$s$, and increment $1<\delta$, define

\[
B_x(s,\delta)=\text{expansions performed after }x\text{ first reaches }s+1,
\text{ through the event that first reaches }s+\delta\text{ or resolves}.
\tag{2}
\]

The count is conditional on the same inherited opposing budget and on no
intervening parent reselection.  It may be zero, or it may contain an entire
wide plateau.

**PROVEN (why $\delta,H$ are insufficient parameters).** T2 below has fixed
$\delta=H=2$, fixed depth two, and $B_D(1,2)=M$.  Hence band mass is not
bounded by any function of $\delta,H,d$ alone.  A total-work theorem must
include $N$, a band-mass/score-response condition, or an equivalent
structural statistic.

## 5. Structural re-traversal bound

### T1. Path-charge upper bound

**PROVEN (T1).** In a progress-certified run of model M,

\[
V\le (d+1)E. \tag{3}
\]

Without staged reopening, $E\le N$, so $V\le(d+1)N$.  With exact
selected-cutoff deepening, $E\le2N-1$, so
$V\le(d+1)(2N-1)$.  In either case $R\le V=O(N(d+1))$, which is
$O(Nd)$ for every nontrivial depth $d\ge1$.

**PROVEN (proof).** Charge each activation to the first expansion event in its
dynamic extent.  Progress certification makes this event exist.  At the instant
of one expansion, at most one activation at each level of the active recursion
stack can have that event as its first expansion; the stack contains at most
$d+1$ nodes.  An activation has only one first expansion, so charges do not
move or duplicate later.  This proves (3).  With persistent expansion, an
entry expands once.  Under exact deepening, every nonroot entry may additionally
be expanded once as a too-deep cutoff and once after reopening; the depth-zero
root cannot be a too-deep cutoff.  Hence $E\le2N-1$.  Finally, repeat
activations are a subset of activations, so $R\le V$.  No step uses
$\delta$ or $H$.

**CITED/PROVEN (scope against the engine).** The engine’s unit progress floors,
`current+1` below the sentinel (`tss_solver.rs:4159-4169`, `:4175-4203`), make the charge argument
plausible, but node caps, depth-cutoff bubbling, Universal yields, and a DAG
entry changed through another parent can produce a no-new-expansion return.
Those cases are deliberately excluded by “progress-certified.”  T1 must not be
quoted as an unconditional production bound.

### F1. Near-matching unary deepening family

**DEFINITION (family $U_d$).** $U_d$ is a unary AND/OR chain
$v_0\to v_1\to\cdots\to v_d$, with every unresolved prior equal to
$(1,1)$.  Nodes $v_0,\ldots,v_{d-1}$ expand to their one child and $v_d$
expands to a proving terminal.  The driver starts at cap 0 and performs exact
selected-cutoff deepening.  Unary node types may alternate; min and sum agree
on one child.

**PROVEN (F1 exact count).** The $d+1$ stages have caps $0,1,\ldots,d$.
At cap $k<d$, the run activates $v_0,\ldots,v_{k+1}$, expands the current
frontier $v_k$ (reopened when $k>0$), and marks $v_{k+1}$ as the selected cutoff: $k+2$
activations and two expansion events.  At cap $d$, it activates
$v_0,\ldots,v_d$ and expands the terminal: $d+1$ activations and one event.
Therefore

\[
N=d+1,\qquad E=2d+1,\qquad
V=\sum_{k=0}^{d-1}(k+2)+(d+1)=\frac{d^2+5d+2}{2},
\]

and, because all $N$ nodes are activated,

\[
R=V-N=\frac{d(d+3)}2=\Theta(Nd). \tag{4}
\]

This near-matches T1 up to a constant factor.  The construction has $H=1$
and no competitive sibling, so $\delta$ is irrelevant: an $O(Nd)$ term can
come from exact deepening alone.

**PROVEN (structural conclusion).** T1 and F1 justify the folklore-shaped
$O(Nd)$ ceiling for $d\ge1$ only for persistent, progress-certified traversal.
They also
show why that ceiling cannot answer R-TS1: it has no favorable $\delta$ or
$H$ dependence, and two schedules can differ by $\Theta(N)$ expansions well
below the ceiling, as T2 shows next.

## 6. Coarse thresholds can force arbitrarily bad frontier overshoot

### 6.1 The local barrier fact

**PROVEN (competitive-window lemma).** Consider a Choice node with
nonbinding inherited PN threshold and opposing DN budget.  Let selected child
$a$ have $pn(a)\le s$, where $s$ is the second-smallest sibling PN.
With increment $\delta$, the recursive call cannot return through the
competitive PN barrier before $pn(a)\ge s+\delta$; with unit increment it may
return as soon as $pn(a)\ge s+1$.  Therefore the larger window irrevocably
grants all work in the refinement band $[s+1,s+\delta)$, unless the child
resolves or the opposing threshold fires first.

**PROVEN (proof).** Substitute the hypotheses into (1).  The parent cap does
not bind, and $s+\delta\ge pn(a)+1$, so the selected PN threshold is exactly
$s+\delta$.  The recursive stopping test is `current >= threshold`.  The
unit case is identical with $\delta=1$.  These are the only facts used.

### 6.2 T2: a depth-two $H=q$ frontier cliff

**DEFINITION (family $C_{q,M}$).** For integers $q\ge2,M\ge1$, expand a
Choice root $R$ into ordered children $D,W$, each with prior $(1,1)$, with
$D$ first in a tie.  Expanding $W$ proves it.  Expanding $D$ makes it a
Choice branch with $M$ children $x_1,\ldots,x_M$, each prior $(q,1)$;
expanding any $x_i$ disproves it.  The full finite tree is

```text
R : Choice
|- D : prior (1,1), then Choice
|  |- x1 : prior (q,1), false terminal on expansion
|  |- ...
|  `- xM : prior (q,1), false terminal on expansion
`- W : prior (1,1), true terminal on expansion
```

It has $N=M+3$, depth two, and prior scale $H=q$.  Root thresholds and the
opposing DN budget are infinite.

**PROVEN (T2, unit run).** With $\delta=1$, expanding $R$ costs one event.
It selects $D$ with PN threshold $pn(W)+1=2$.  Expanding $D$ changes its
PN from 1 to $\min_i pn(x_i)=q\ge2$, so it returns immediately without
expanding an $x_i$.  Now $W$, at PN 1, is the unique minimum and proves the
root in one event.  Thus

\[
E_1=3 \quad\text{including the root, or 2 post-root}. \tag{5}
\]

**PROVEN (T2, coarse run).** With $\delta=q$, $D$'s threshold is $q+1$.
After expanding, $pn(D)=q<q+1$.  While any $x_i$ remains, it is a minimum-PN
child at $q$; its current+1 floor and inherited cap both give threshold
$q+1$, so its false terminal must be expanded.  The Choice minimum stays $q$
until the last $x_i$ is disproved; then $D$ is disproved and $W$ proves.
The DN budget cannot interrupt this run because it was inherited as infinity.
Consequently

\[
E_q=M+3=N \quad\text{including the root, or }M+2\text{ post-root}. \tag{6}
\]

The coarse schedule performs exactly $M=N-3$ extra expansions.

**PROVEN (C2).** Setting $q=2$ fixes $\delta=H=2$ and depth at two while
$M$ grows.  Hence the worst-case additive overhead is $\Theta(N)$, matching
the trivial persistent-tree upper bound $E_q-E_1\le N$ to an additive
constant.  In particular, no $o(N)$ upper bound can be a function only of
$\delta,H,d$.  A TT is not necessary for a delta-2 catastrophe.

**PROVEN (scope clarification).** Model M permits consecutive nodes of the
same kind, and T2 uses a Choice-to-Choice edge to keep depth fixed.  T2b uses
the alternating Choice/Universal core (with alternating unary wrappers as
needed) and establishes the same $\Theta(N)$ gap with $H=1$, at growing
depth.  No production claim relies on silently treating T2 as a literal game
position.

### 6.3 T2b: the same cliff with $H=1$

**DEFINITION (unit-prior refinement ladder $L_{q,\ell}$).** Fix integers
$q\ge2$ and even $\ell\ge2$.  Expand Choice
root $R$ into ordered unit-prior children $A,B$, with $A$ first; $B$
proves on expansion.  Expanding $A$ makes it Universal with unit-prior
ordered children $X,Y$, with $X$ first, so $(pn(A),dn(A))=(2,1)$.  $Y$
remains unresolved.  $X$
is a chain of $\ell$ unit-prior nodes: each nonbottom expansion reveals one
unit-prior child, and the bottom expansion reveals a Universal node’s $q$
unit-prior children.  Thus $\ell$ expansions change $X$ from $(1,1)$ to
$(q,1)$ without expanding those $q$ children.  Starting $X$ as Choice, even
$\ell$ makes the bottom Universal, so every edge in this core alternates;
unary min and sum preserve the carried pair.

**PROVEN (T2b).** Unit delta gives $A$ root threshold 2.  Its own expansion
makes $pn(A)=2$, so it returns and $B$ proves: $E_1=3$, including $R$.
For $\delta=q$, $A$'s root threshold is $q+1$.  At $A$, conjunctive
budget subtraction gives selected $X$ PN threshold
$(q+1)-(2-1)=q$; its competitive DN threshold is $1+q$.  The ladder keeps
DN at 1 and needs exactly $\ell$ expansions to reach PN $q$.  Then
$pn(A)=pn(X)+pn(Y)=q+1$, $A$ returns, and $B$ proves.  Therefore

\[
E_q=\ell+3,\qquad N=\ell+q+4.
\tag{7}
\]

For fixed $q=2$, the overhead $\ell=\Theta(N)$ occurs with $H=1$, on a
finite tree, without transpositions.  This proves that coarse-window overshoot
is not merely a mismatch between \(+1\) and nonunit priors.

**PROVEN (interpretation limited to the mechanism class).** T2/T2b do not
prove that `0l4291i_live` contains either literal gadget.  They prove the
sufficient mechanism in the disjoint persistent-tree construction: if the
interval newly admitted by delta 2 contains a high-mass refinement plateau,
the unit schedule’s alternative would resolve first, and no transposition or
policy side effect differs, then increasing delta increases total work by the
entire plateau mass.

## 7. Heuristic initialization and starvation

### 7.1 T3: fixed ratio two, linear worst-case overhead

**DEFINITION (true proof cost for this section).** $P(x)$ is the minimum
number of still-pending leaf expansions needed to expose a proof below $x$
in the fully specified finite tree.  A positive prior $p(x)\le P(x)$ is
called admissible.  This is a work lower bound before expansion; it is not the
terminal PN, which becomes zero after a proof.

**DEFINITION (family $S_M$).** A Choice root $R$ has ordered children
$W,D$, with $W$ first in a tie.  $W$ proves on its one expansion, so
$P(W)=1$.  $D$, prior 1, expands to a Choice node with $M$ unit-prior
children, each of which disproves on expansion.  Compare only two
initializations of $W$: admissible $p(W)=1$, and heuristic $p(W)=2$.
The maximum prior in the latter run is $H=2$, and $\delta=1$.

**PROVEN (T3).** With admissible initialization, $W,D$ tie at 1 and the fixed
order expands $W$ first; including $R$, the cost is 2.  With heuristic
initialization, $D$ is the unique minimum.  Its root-supplied PN threshold is
$p(W)+1=3$.  After $D$ expands, its Choice PN remains 1 while any false
child is pending, so all $M$ false children must be expanded before $D$
becomes disproved; only then does $W$ prove.  The cost is $M+3=N$.
Therefore a fixed overestimate ratio

\[
\rho=p(W)/P(W)=2
\]

causes $M+1=N-2=\Theta(N)$ extra expansions at fixed $H=2$, $\delta=1$,
and depth two.  No overhead bound $o(N)$ can depend only on $\rho,H,d$.

**PROVEN (bounded-branching variant).** The tie dependence can be removed at
the cost of $H=\rho=3$.  Give $D$ prior 2 and $W$ either admissible prior
1 or heuristic prior 3.  On expansion, $D$ becomes a unary Universal wrapper
around false Choice tree $F_k$.  Let $F_0$ be one losing leaf and let
$F_{k+1}$ be a binary Choice node whose two children are unary Universal
wrappers around independent copies of $F_k$, all newly exposed priors 1.
Its node count satisfies

\[
n_0=1,\qquad n_{k+1}=3+2n_k,qquad n_k=4\cdot2^k-3.
\]

Once expanded, $D$ and its selected $F_k$ have PN 1 until the entire false
tree is refuted.  The heuristic run therefore expands all
\(n_k=\Theta(2^k)\) false
nodes before $W$; the admissible run selects $W$ strictly first.  The depth
is $2k+O(1)$, giving $\Theta(2^{d/2})$ starvation in this explicit binary
strictly alternating family.  This is a result of the construction, not an
invocation of a folklore PNS bound.

**PROVEN (what admissibility does and does not buy).** On $S_M$, certified
admissibility forces the cost-one winner’s integer prior to 1 and prevents the
strict priority inversion; with the stated tie order it removes all starvation.
Admissibility alone is not a global search guarantee: if an admissible false
child and the admissible winner both have prior 1 and the false child wins the
tie, the same false subtree can still be exhausted.  Certified floors need a
calibrated response/tie theorem, not merely $p\le P$.

### 7.2 T4: the bound available under score/work calibration

**DEFINITION (unit-calibrated frontier).** A pre-expanded Choice node has a
not-yet-selected winning child $w$, fixed at PN $1\le p_w\le H$, and $b-1$
distractors with current PNs $p_i\ge1$.  Until $w$ is first selected, each
expansion in a selected distractor makes its PN increase by at least one and
never decrease, or refutes it.  Opposing and inherited thresholds do not bind.
Ties may put $w$ last.

**PROVEN (T4 starvation bound).** While $w$ is unselected, the second-smallest
PN seen by a selected distractor is at most $p_w$, so its competitive
threshold is at most $p_w+\delta$.  After at most

\[
S_i=(p_w+\delta-p_i)_+
\]

of its expansions, it is either refuted or strictly above $w$.  Summing,

\[
S\le\sum_{i\ne w}(p_w+\delta-p_i)_+
 \le(b-1)(H+\delta-1). \tag{8}
\]

The last inequality uses $p_w\le H$ and $p_i\ge1$.  The bound is exact already
for $b=2$: one unit-response distractor starting at 1 is driven to
$p_w+\delta$ before the winner.  Disjoint copies give additive equality.  For
$\delta=1$, the expression specializes to the exact \((b-1)H\) worst case.

**PROVEN (overestimate-ratio total bound).** Use the maximally informative
admissible calibrated value $p_w=P=P(w)$ as reference, and let a heuristic
replace it by $P\le p'_w\le\rho P$.  Applying (8) to the heuristic run gives

\[
S(p'_w)\le\sum_{i\ne w}(p'_w+\delta-p_i)_+
          \le(b-1)(\rho P+\delta-1). \tag{9}
\]

The analytical envelope
$F(x)=\sum_{i\ne w}(x+\delta-p_i)_+$ is $(b-1)$-Lipschitz, so
$F(p'_w)-F(P)\le(b-1)(\rho-1)P$.  This is a comparison of envelopes, not a
bound on the difference between two actual schedules.

**PROVEN (paired-run obstruction).** The stronger claim
$S(p'_w)-S(P)\le(b-1)(p'_w-P)$ is false even under unit response.  Let
$b=3$, $\delta=2$, put two distractors before $w$ in tie order, start both at
1, and increase the selected score by exactly one per expansion.  With
$P=2$, the distractors run $1\to3$ and $1\to4$, so $S(2)=2+3=5$.  With
$p'_w=3$, they run $1\to3$, $1\to5$, and then the first tie-winning
distractor runs $3\to5$, so $S(3)=2+4+2=8$.  The paired excess is 3, greater
than $(b-1)(3-2)=2$.  Thus (9), rather than a paired-difference formula, is
the valid ratio-dependent guarantee.

**PROVEN (calibration is necessary).** T3 violates unit calibration: a false
Choice subtree can consume $M$ expansions while its PN stays on one plateau.
It has fixed $\rho=H=2$ but unbounded $M$.  Thus (8)-(9) cannot be extended
to arbitrary heuristic df-pn trees by dropping the response assumption.

## 8. The favorable regime and its exact crossover

### 8.1 T5: what additive delta really bounds

**DEFINITION (competitive-barrier return and $PV$).** At a Choice parent, let
$s$ be the second-best PN at child-activation entry.  A child activation is a
competitive-barrier return if it ends by reaching the fixed threshold
$s+\delta$ computed at entry, rather than by a parent cap, DN budget, terminal,
or external stall.  Use the Universal/DN dual there.  For each parent-child
edge, $PV$ is the sum of positive changes of that competitive child number
over the disjoint barrier-return intervals; sum over edges for global $PV$.

**PROVEN (T5 variation bound).** At the start of a barrier interval, selected
score $x$ is no greater than second-best $s$.  At its end, score
$y\ge s+\delta$.  Even if the score decreases inside the call, its positive
variation over the interval is at least $y-x\ge\delta$.  Intervals at one
parent are disjoint.  Therefore

\[
B_{\mathrm{barrier}}\le \frac{PV}{\delta}. \tag{10}
\]

**PROVEN (monotone bounded-score corollary).** If child $i$'s competitive
score is monotone from $p_i$ through an exclusive stopping boundary $U$,
its positive variation before resolution is at most $U-p_i$.  Consequently

\[
B_{\mathrm{barrier}}
 \le\sum_i\left\lfloor\frac{U-p_i}{\delta}\right\rfloor.
\tag{11}
\]

Resolution activations are excluded from $B_{\mathrm{barrier}}$; adding all
of them contributes at most one per resolved child.  For $b$ equal-score
unit-response arms with $U=p+\delta$, the first $b-1$ arms each make one
barrier return, while (11) gives $b$; the ratio $(b-1)/b$ approaches one.
Independent one-return parent gadgets attain (10) exactly with
$PV=\delta$ apiece.  Thus the local $1/\delta$ dependence is order-tight,
although ordinary second-best scheduling is not a fixed-size round-robin
block scheduler.

**PROVEN (why this is not an $H/\delta$ total-work theorem).** $H$ bounds
only immutable priors.  After expansion, a Universal PN or Choice DN is a sum
and may greatly exceed $H$; heuristic expansion may also make a score
decrease before it increases.  Hence $H$ does not bound $U$ or $PV$.
Moreover, (10) charges recursive returns, not expansions inside an interval:
T2 makes one coarse interval contain $M$ expansions.  At fixed or independently
bounded $PV$, larger delta gives a smaller local call ceiling; (10) is not a
cross-schedule monotonicity theorem because $PV$ may itself change with delta.

### 8.2 T6: component-level revisit-fraction threshold

**DEFINITION (cost decomposition).** Write unit-schedule wall as
$W_1=C_{\rm other}+C_{\rm rev}>0$, where $C_{\rm rev}$ is cost attributable
to revisit traversal and $C_{\rm other}>0$ is everything else.  Let
$f=C_{\rm rev}/W_1$, so $0\le f<1$.  Suppose a coarser schedule saves a
fraction $\sigma\in[0,1]$ of $C_{\rm rev}$ but inflates $C_{\rm other}$ by a
fraction $\gamma\ge0$:

\[
W_\delta=(1+\gamma)C_{\rm other}+(1-\sigma)C_{\rm rev}. \tag{12}
\]

**PROVEN (T6 crossover identity).** If $\gamma+\sigma>0$, direct subtraction
and division by $W_1$ give

\[
W_\delta<W_1
\iff \gamma(1-f)<\sigma f
\iff f>\frac{\gamma}{\gamma+\sigma}. \tag{13}
\]

If $\gamma=\sigma=0$, then $W_\delta=W_1$ and no strict win occurs, so the
undivided middle inequality is the complete statement for that degenerate
case.

Since $\sigma\le1$, a necessary condition for *any* win is

\[
\gamma<\frac{f}{1-f}. \tag{14}
\]

This is an accounting theorem; it does not assume constant event costs because
$C_{\rm other}$ and $C_{\rm rev}$ are aggregate components.

**CITED/PROVEN (application to R-TS1’s attribution).** The 7.01% figure is a
proportional attribution from aggregate descent time, not a directly timed
causal revisit component.  If it is taken as $f=0.0701$ in (12), even perfect
elimination of all
revisit-attributed work tolerates less than

\[
\gamma<0.0701/0.9299=0.0754,
\]

or 7.54%, inflation of the non-revisit component.  To achieve the campaign’s
5% promotion target with zero inflation would require
$\sigma\ge0.05/0.0701=71.3\%$, matching the recorded ceiling argument
(`HUNT_REPORT_THRESHOLD_SCALE.md:23-34`).  Delta 2 instead increased official
wall, expansions, and absolute revisits on the hard row; empirically it did not
approach a win.

**CITED/PROVEN (engine-useful conditional rule).** Under decomposition (12),
revisit share alone never licenses widening.  A candidate also needs a bound or
small measurement for score-band mass/extra expansion cost.  If R-TS1’s 7.01%
proportional attribution is used as $f$, any parameter that raises non-revisit
work by 7.54% or more cannot win even under the impossible best case of deleting
every revisit.

## 9. DAG admission saturation: amplifier, not prerequisite

### D1. Exact admission-only duplication lemma

**DEFINITION (capped exact-index model).** An exact transposition index holds
$C$ semantic keys.  Before capacity, the first encounter of a key creates an
arena entry and every later parent links that entry.  After capacity, an
unseen key still creates a persistent edge-local arena entry, but is not put in
the index; a later, different parent therefore cannot discover that copy and
creates another.  Assume the schedule forces every stated edge to expand.

**PROVEN (D1).** Fill the index with $C$ unrelated keys.  Then create $k$
parents, each with edges to the same $M$ semantic children
$z_1,\ldots,z_M$, none among the indexed keys.  An unlimited index expands
the $M$ children under the first parent and reuses them under the other
$k-1$, for $M$ child expansions.  The saturated index creates and expands
one private copy per parent edge, for $kM$ child expansions.  The exact
amplification is

\[
kM-M=(k-1)M. \tag{15}
\]

This is a finite acyclic semantic DAG.  It proves that admission-only saturation
can amplify already-misallocated frontier work without evicting anything.

**CITED (mapping of the data structure only).** The model’s admission behavior
matches `insert_position`: an indexed key hits, while an over-cap key still
gets an arena entry but is not inserted into `by_position`
(`tss_solver.rs:3606-3651`).  It does not model lazy future keys, commitment,
or which transposed edges the engine actually visits.

**SKETCH (composition with T2).** Replace T2’s cheap winning leaf $W$ by a
winning subproblem containing the $k$-by-$M$ transposition layer.  Choose
the bad child’s widened score band to create enough distinct filler keys to
fill the index.  Unit delta returns before those fillers, enters $W$ with
index space, and reuses its shared children; coarse delta consumes the filler
band first, enters the same $W$ after saturation, and pays (15).  This is an
explicit construction in the capped-index abstraction.  A line-by-line
realization through lazy `WidePnChild` admission and all production selection
overrides remains unproved.

**CITED (measured consistency, not causation).** The hard delta-2 row reached
1,073,741,810 indexed bytes and had 6,054,588 expansions but 3,586,248 indexed
entries; +1 stopped at 549,161,606 bytes (523.7 MiB) with 1,879,611 expansions
(`THRESHOLD_DELTA2_FULL_RAW.log:41`, `THRESHOLD_COUNTER_FULL_RAW.log:40`).
Those aggregates establish admission saturation and much larger expansion and
indexed-entry totals.  They do not identify how many over-cap entries were
duplicate semantic keys, so they do not prove D1 occurred materially in this
instance.

**CONJECTURE (E1: measured catastrophe mechanism).** Delta 2 first crossed a
high-mass score band of the T2/T2b kind, causing the observed 3.221x expansion
count.  Admission saturation then amplified later work through loss of
eligibility for cross-parent indexed reuse.  The first clause is consistent
with worse global best-first arbitration; the second is plausible from the
data structure and peak/admission totals.  Neither clause is causally isolated
by the retained aggregate counters.

## 10. Implications for this engine

**CITED (measured-profile recommendation; single most useful implication).**
Retain `second+1` for this engine’s present profile: it won the official R-TS1
comparison, while delta 2 added 85.6% wall and delta 4 and mean each lost one
hard verdict (`HUNT_REPORT_THRESHOLD_SCALE.md:36-67`).

**PROVEN/CITED (conditional accounting consequence).** If R-TS1’s 7.01%
proportional attribution is treated as $f$ in (12), a coarser schedule cannot
win if it inflates all non-revisit work by 7.54%, even while deleting every
revisit.  T2 and T2b prove that neither $\delta$ nor the 1..37 PN-prior range
alone supplies a sublinear safety guarantee: one widened score band can contain
$\Theta(N)$ expansion work.

**CONJECTURE (engineering implication from T2/T2b and D1).** A future widening
proposal should require a measured **band-mass ceiling and TT-headroom gate**,
not merely a high revisit count.

**PROVEN (what the initialization results support).** On $S_M$, replacing the
overestimate by certified admissible initialization prevents strict starvation
of the cost-one proof.  T4 quantifies the benefit when
live score is calibrated to work.  The same results also rule out a stronger
claim: admissibility by itself does not cap global starvation, because ties and
score plateaus remain.

**CONJECTURE (testable counter for a future non-Cargo round design).** In a
delta arm, count expansions after a selected child has crossed the threshold
that the +1 arm would have used, but before the actual coarse call returns.
Call this `competitive_band_expansions`.  T2 predicts it equals the harmful
$M$; T5 separately measures saved barrier calls.  Together those counters
separate coarse-band work from re-entry savings.  A useful paired report would
include:

1. competitive-band expansions by depth, node kind, and prior bucket;
2. calls saved relative to the shadow +1 barrier;
3. first index-admission rejection and post-rejection arena inserts;
4. repeated semantic keys among unindexed inserts, using a bounded shadow
   fingerprint set; and
5. expansion/wall cost on each side of Equation (13).

**CONJECTURE (promotion gate).** Widening should be rejected before a full run
unless a leaf/counter pass shows both

\[
\text{saved revisit cost}>
\text{competitive-band expansion cost}
\]

and enough projected index headroom to avoid crossing the admission cap.  This
gate is sufficient only in a restricted additive model in which those two
quantities exhaust every schedule-dependent cost difference; it is not proved
sufficient for the full engine with heterogeneous expansion costs.

## 11. What remains open

**PROVEN (negative boundary established here).** No theorem depending only on
$\delta,H,d$, or only on a heuristic overestimate ratio, can give a
sublinear-in-$N$ bound on extra expansions: T2 and T3 are explicit
counterexamples.  Any sublinear or delta-favorable theorem must assume bounded
frontier-band mass, monotone/calibrated score response, or a related structural
quantity.

**SKETCH (strongest plausible next positive theorem).** Let $E_0$ count
expansion events outside coarse-only competitive bands; let every entered unit
score interval $[z,z+1)$ contain at most $L(z)$ expansion events; let parent
caps cause at most $Q$ otherwise-unaccounted returns; and let every semantic
DAG key retain exact indexing.  Count the “entered bands” sum with multiplicity
over parent-child activation band-entry occurrences.  If every remaining
activation with no expansion in its dynamic extent is a competitive-barrier
return, then T1 and (10) suggest

\[
E\le E_0+\sum_{\text{entered bands}}L(z),
\qquad
V\le(d+1)E+PV/\delta+Q,
\]

and, if $c_a,c_e$ bound per-activation overhead and per-expansion cost,

\[
\text{wall}(\delta)\le
c_a\bigl((d+1)E+PV/\delta+Q\bigr)+c_eE.
\]

F1, the independent-arm T5 gadgets, and T2/T2b separately match the path,
barrier-return, and band-mass terms.  The displayed combination remains a
sketch because the activation classes can overlap, ordinary PN/DN values do not
bound $L(z)$, and production policy overrides make $Q$ path-dependent.

**SKETCH (DAG obstacle).** With a coherent unlimited index, path charging can
be extended by charging propagation-only activations to a finite number of
edge-value changes.  With admission saturation, D1 shows that “$N$” must be
declared as either semantic keys or materialized arena copies; the two can
differ by a transposition multiplicity factor.  A tight theorem needs both
quantities and the admission order.

**CONJECTURE (actual admissible floors).** Some Connect6 geometry facts may
provide certified lower bounds on true proof/disproof leaf cost, but the live
fork prior $37-\min(\text{fork degree},36)$ and tau-derived prior are not shown
admissible by this note.  Proving such bounds, then measuring their
calibration/ordering loss, is the precise theorem-backed continuation for
agenda 1.3.

### Named resume point

**SKETCH (R-T1.1 — frontier-band response census).** Resume by specifying a
test-only `competitive_band_expansions` counter against the exact formulas at
`tss_solver.rs:4139-4203`, plus post-admission duplicate-key telemetry.  The
first analytical target is the conditional theorem in Section 11 with an
empirically bounded $L(z)$; the first engine target is deciding whether the
hard row’s 4,174,977 extra expansions lie chiefly inside widened competitive
bands or after admission saturation.  No new delta A/B should precede that
census.

## 12. Bottom line

**PROVEN.** The persistent idealized class has an $O(N(d+1))$ traversal ceiling
($O(Nd)$ for $d\ge1$), near-matched by unary exact deepening.  More importantly,
increasing additive delta can add $\Theta(N)$ expansions at fixed
$\delta=H=2$ and even at
$H=1$; a fixed non-admissible ratio can add $\Theta(N)$ starvation work.
Under T5’s explicit variation/monotonicity assumptions, a conditional
$1/\delta$ call ceiling is recoverable; T6 gives the exact cost crossover.

**CITED.** R-TS1 sits on the unfavorable side: 7.01% of wall was
proportionally revisit-attributed; delta 2 lowered revisits per expansion but
increased the hard row to 3.221x expansions and saturated index admission;
delta 4 and mean each lost one hard verdict.

**CONJECTURE.** The measurements are consistent with the proven mechanism
class—frontier accuracy can be worth much more than the re-traversal it causes—but
the aggregate log does not identify a literal worst-case gadget.  Frontier-band
mass, not prior scale alone, is the right next predictor.  R-T1.1 is the named
point at which to test it.

## 13. Errata folded from the hostile review (R-T1-REV)

The review (`docs/DFPN_RETRAVERSAL_REVIEW.md`) confirmed all ten formal
results (no refutation, no downgrade to SKETCH) with the following
scoped corrections, which are hereby part of this document:

1. **Model scope (MAJOR, folded in §4.1):** the numerical core is
   idealized below saturation; `PN_INFINITY` clamping is an explicit
   engine exclusion.  T2's engine realization range is $2\le q<I$,
   $1\le M<I-1$.
2. **Official A/B scope (MAJOR, folded in §2.2):** the off-versus-2
   comparison also toggles sentinel clamping; exactness holds below the
   sentinel, with the `Some(1)` structural-equality control cited and
   the missing delta-2 sentinel-hit counter noted.
3. **T3 hypotheses (MINOR):** the family $S_M$ additionally assumes all
   DN priors are 1 and root thresholds are infinite (the run analysis
   uses both).  The overestimate parameter is the winner-prior
   overestimate ratio **in model M**, not a claim for every global
   heuristic-ratio definition.
4. **T5 engine transfer (MINOR):** inequality (10) is a formal-model
   call ceiling; for direct engine use it requires numerical minimum
   selection and $s\le I-\delta$.  The transferable weighted form
   charges each return by its actual margin
   (`effective_threshold - entry_score`); the sum of those margins is
   at most $PV$.  (10) is not a counter ceiling for every production
   threshold-cross return.
5. **D1 statement (MINOR):** the duplication family quantifies positive
   integers $k,M$, and the counted quantity is direct z-entry
   expansions (one terminal-on-expansion entry per z).  The D1-to-T2
   production composition remains SKETCH.
6. **F1 prose (MINOR, folded in §5):** at cap 0 the frontier vertex is
   fresh, not reopened.
7. **Attribution rounding (MINOR, folded in §2.1):** 34.80 s / 7.02%
   under conventional rounding.
