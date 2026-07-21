# Research divergence 1: forcing width, semantic clocks, and reply structure

Date: 2026-07-21  
Branch / base inspected: `claude/research-div` / `6ba0c9615321`  
Scope: research report only; no engine or verifier changes

## Executive result

**MEASURED.** The three atlas-certified wins that naturally exhaust the
production width do not need a quiet attacker turn.  Each starts with a stone
that already buys the whole defender turn by creating three overlapping
count-four threats with hitting number two.  The missing move is the attacker's
otherwise free second stone: it seeds two new axes but lies outside the exact
second-stone universe `S(P,a)`.  This is a narrower failure than the phrase
"non-forcing width" suggests.

**HYPOTHESIS — top build candidate.** Add a default-off `J2near` free-tempo
tier only after the first stone has already forced the defender.  It captures
all three exact certified lifts.  At the witness roots it changes accepted
attacker children `19 -> 39`, `19 -> 39`, and `8 -> 12`.  Across 2,720 human
roots its mean accepted-child multiplier is 1.039 on the 100 eligible roots;
across 3,255 self-play roots only four are eligible; none of the 248 grind
roots is eligible.  These are candidate-count measurements, not a runtime or
coverage claim.  A matched-cap solver A/B remains mandatory.

**HYPOTHESIS — proof-ready efficiency candidate.** At a pure
`vcf_pair_complete` defender boundary, the live threat sets have rank at most
two.  A budget-two, hitting-number-two family has at most four unordered
minimum-cover pairs.  The current solver nevertheless applies each kernel
first stone, reruns threat analysis, derives mates, and checks reverse-order
equality.  The same reply plan can be constructed statelessly.  There were
zero violations over 229 real reply pairs and an exhaustive 33,861-family
rank-two model check.

**CONJECTURE — safe high-upside experiment.** Guessed reply equivalence remains
false: four real positions have two minimum coverers with opposite proven
outcomes.  A sound replacement is sibling-certificate transplantation guarded
by the strict verifier.  Prove one child, rebind its standalone proof skeleton
to a sibling, and reuse it only when independent strict verification accepts.
The experiment makes no equivalence inference.

**HYPOTHESIS — theorem result, implementation closed for the target.** A finite
forcing-region invariant yields a genuine semantic deadline `e+2`, where `e`
is the number of initially empty cells in the region.  This is a real bridge
from an unbounded contract search to the existing census theorem.  However,
an open-carrier-ray theorem proves that this pointwise finite region cannot
exist whenever a live count-two carrier has a defender-free continuation ray.
That obstruction occurs on all 248 grind roots.  The theorem should be kept;
this concrete region abstraction should not be built for grind acceleration.

**MEASURED.** The cheap necessary-condition leaves are useful as a future
certified-refutation base case, not as a wall-time cure.  `NoJointCarrier`
fires on 29.94% of fresh-turn self-play roots and 13.77% of fresh-turn human
roots, but on zero of 193 fresh-turn grind roots; every observed hit already
exhausted in one expansion.

### Recommended order

1. **HYPOTHESIS.** Shadow and then A/B `J2near`; require all three atlas wins
   to verify before considering broader coverage.
2. **HYPOTHESIS.** Replace dynamic K2 reply-plan reconstruction with direct
   rank-two cover enumeration; carry plans across lazy edges only as a second
   memory-gated rung.
3. **CONJECTURE.** Shadow strict sibling-certificate transplantation on
   zone-free forced universals.
4. **HYPOTHESIS.** Formalize the finite-region deadline and the open-ray
   impossibility together, as a boundary result.  Do not implement the
   pointwise closure for production.
5. **HYPOTHESIS.** Add `NoJointCarrier` and exact
   `NoAdmissibleFirstTurn` only when the restricted-strategy refutation grammar
   is ready to consume them.

## 1. Scope, notation, and evidence discipline

**CODE-FACT.** The production leaf profile selects
`vcf_pair_complete`, lazy frontier, and the interior census gate in
[`tss_solver.rs`](../packages/hexfield_eq/rust/src/tss_solver.rs#L997).
The census evaluator refuses remaining horizons above eight
([lines 252–295](../packages/hexfield_eq/rust/src/tss_solver.rs#L252)), so
the gate is inert under the production `u32::MAX` semantic horizon.

**CODE-FACT.** At a fresh attacker turn, the current first-stone universe is
the union of empties in claimant-pure count-at-least-two windows and empties of
live defender count-at-least-four windows.  For a chosen first stone `a`,
[`WideTurnGate::second_candidates`](../packages/hexfield_eq/rust/src/tss_solver.rs#L9393)
constructs

`S(P,a) = (T(P) \ {a}) ∪ G1(P,a)`.

Here `G1(P,a)` contains empties of claimant-pure count-one windows promoted by
`a`.  [`evaluate_pair`](../packages/hexfield_eq/rust/src/tss_solver.rs#L9458)
retains the complete turn only if it creates a nonempty claimant threat family,
answers every live defender win-now threat, and leaves the defender with
hitting number two or with no size-two cover.

**CODE-FACT.** Let `F_A(P)` be the residual-empty-set family of live claimant
count-four/count-five windows.  This report writes `tau(F)` for the minimum
hitting-set size, and `tau(F)=∞` when no set of size at most two hits `F`.
For production pair-complete pending children, `tau` is therefore `2` or `∞`.

**CODE-FACT.** A failed restricted search is not a game-theoretic loss.  In the
wide PN engine an empty attacker child set becomes `Refuted`, and a defender
node outside exact implicit dispatch also becomes `Refuted`
([lines 6453–6497](../packages/hexfield_eq/rust/src/tss_solver.rs#L6453)).
Every negative theorem below is therefore stated as `NoContractWin
VcfPairComplete`, never as a global Connect6 loss.

**MEASURED.** Evidence was reconstructed from the 47,902-row atlas and its
verifier-accepted maxsolve raw, the 468-row `puzzle_v3`, 2,720-row `human_v1`,
3,255-row `selfplay_v1`, the 248 labeled grind roots, 14 checked-in forcing
certificate lines, and the frozen human-160 residue cohort.  Scratch analyzers
and hashes are listed in Appendix A.

**MEASURED.** The checked-in `raws/lanec_labels.jsonl` snapshot divides the 191
unknown grind rows into 97 below-50k natural exits and 94 exact-50k cap exits.
`SOLVER_NOTES.md` records the earlier prose split as 96/95.  This report labels
the live raw snapshot explicitly and does not mix the two partitions.  All
claims over "248 grinds" are invariant to that one-row drift.

**CODE-FACT.** The following labels are used literally throughout:

- **MEASURED** means emitted by a named artifact or probe on real positions.
- **CODE-FACT** means directly visible in the checked-out implementation or
  verifier contract.
- **HYPOTHESIS** means a precise, falsifiable claim with a proof sketch or
  proposed theorem, not yet accepted in Lean and production.
- **CONJECTURE** means the mechanism is plausible but lacks either a complete
  proof shape or direct target-cohort evidence.

## 2. Ranked finding 1 — the missing width is a free-tempo seed

### 2.1 Precise mechanism

**MEASURED.** In every cheap atlas miss, one attacker placement already creates
exactly three overlapping count-four windows with `tau=2`.  It also answers
every live defender count-four threat.  The second attacker placement creates
no count-four threat.  Its job is to turn count-one support into count-two
support on two axes while the defender is compelled to spend both replies on
the first stone's fork.

**CODE-FACT.** The current normalization deletes that tempo.  In the two
fresh-turn roots, the forcing stone belongs to `T(P)` but the seed does not
belong to `S(P,a)`; reversing the order is impossible because the seed is not
in `T(P)`.  In the partial-turn root, the first stone is already fixed and the
seed is outside the regenerated count-at-least-two `T` universe.

**MEASURED.** This is not a principal-variation guess.  Each missing move lands
on an exact certified LOSS child for the same claimant.  The opening-atlas
campaign re-solved that child, prepended the legal claimant Choice placement(s),
rebuilt the parent certificate, and passed the normative `TssVerifier`
([maxsolve report, lines 41–44](../../opening-atlas/OPENING_ATLAS_MAXSOLVE_REPORT.md#L41)).

| Parent certified WIN | Current wide result | Missing root turn | First-alone family | Seed support | Exact decisive child |
|---|---:|---|---|---:|---|
| `oa-0153903c5a863630` | UNKNOWN, 42 nodes / 41 expansions | `(0,-1), (-1,2)` | 3 count-four, `tau=2` | `5,5` axes | `oa-5166cc20e3ecc7b7`, certified LOSS, 521 cert nodes |
| `oa-773ca1a59e95f4e1` | UNKNOWN, 42 / 41 | `(3,-3), (3,-2)` | 3 count-four, `tau=2` | `5,5` | `oa-3fa9037cf3f1144b`, certified LOSS, 521 cert nodes |
| `oa-6fda812864c6d19a` | UNKNOWN, 20 / 19; root is SecondStone after `(-2,0)` | seed `(0,-2)` | 3 count-four, `tau=2` | `4,4` | D6-equivalent `oa-9e524a9bf4fab453`, certified LOSS, 521 cert nodes |

**MEASURED.** Metric names in this table are deliberate: 20/42 are reproduced
wide-search nodes, while 521 is the atlas `cert_nodes` field.  The separate
atlas solve-work counts are 1,402–2,023 for the decisive children and are
retained in the scratch ledger.

**MEASURED.** The parent certificates contain 523, 523, and 600 nodes and have
terminal winning lines of 22, 22, and 21 placements.  All three are
`certified=1`; their D6 verification count is at least two.  The exact roots,
lines, child records, and replay classifications are recorded in
`.scratch/width_witness_results.md`.

**MEASURED.** A direct `opening_atlas_pass1` reproduction at cap 100,000,
unbounded semantic horizon, and 256 MiB TT naturally exhausts at the node
counts in the table.  The brief's "21–43" convention includes the outer/root
accounting; the engine raw reports 20/42.  `quiet_turn_consume` is identical on
these roots.  A much broader `round3_consume` attempt failed to finish its
100,000-node target after about 140 seconds and was stopped.

### 2.2 The proposed minimal tested extension

**HYPOTHESIS.** Call a normal first stone `a` *turn-buying* when, in `Q=P+a`,
there is no live defender count-at-least-four window and
`tau(F_A(Q)) >= 2`.

**HYPOTHESIS.** For an empty `b` and axis `alpha`, define

`m_alpha(Q,b) = |{ W : W is A-pure in Q, count_A(W)=1,
                         axis(W)=alpha, b in empty_Q(W) }|`.

Define the near-cross spare tier

`J2near(P,a) = { b in Legal(Q) \ (S(P,a) ∪ {a}) :
                 at least two axes alpha have m_alpha(Q,b) >= 4 }`.

The `2` names the two supported axes; four is the per-axis support threshold.
Use `S(P,a) ∪ J2near(P,a)` only for a turn-buying `a`.  At a SecondStone root
`Q`, use the analogous `J2near(Q)` outside the ordinary `T(Q)` only when the
already-played first stone has bought the turn.

**MEASURED.** This is the tightest uniform member of the tested support
hierarchy that covers all three rows.  Requiring five weak windows per axis
drops `oa-6fda...`, whose support is `4,4`; requiring two supported axes is
load-bearing.  Broad all-count-one width adds 110, 110, and 56 root candidates.
Loose two-axis `J2` is materially more expensive on real cohorts.  `J2near` adds
20, 20, and 4 accepted root children.

**HYPOTHESIS.** `J2near` is "minimal" only in this preregistered family of
axis-count predicates.  No completeness claim is made for all strategically
useful spare stones, and no omitted count-one cell is claimed losing.

### 2.3 Real-position branching estimate

**MEASURED.** The following scan exactly replays geometry and the forcing gate,
but counts accepted root children rather than running PN search.  `Eligible`
means the first/current partial turn is already forcing.  Multiplier statistics
are over all eligible roots, including roots on which `J2near` is empty.

| Cohort | Rows / usable | Eligible | `J2near` nonempty | Added children | `(current+J2near)/current` |
|---|---:|---:|---:|---|---|
| three witnesses | 3 / 3 | 3 | 3 | `20,20,4` | `2.053,2.053,1.500` |
| `puzzle_v3` | 468 / 463 | 21 (4.54%) | 15 (3.21% of all) | mean 4.43, p50 2, p90 14, max 20 | mean 1.191, p50 1.067, p90 1.519, max 2.053 |
| `human_v1` | 2,720 / 2,701 | 100 (3.70%) | 38 (1.40% of all) | mean 1.22, p50 0, p90 4, max 13 | mean 1.039, p50 1.000, p90 1.057, max 2.250 |
| `selfplay_v1` | 3,255 / 3,124 | 4 (0.13%) | 1 | 1 total | mean 1.010, max 1.040 |
| 248 grinds | 248 / 248 | 0 | 0 | 0 | 1.000 at roots |

**MEASURED.** Loose `J2` has human mean multiplier 1.423 and maximum 5.167;
on puzzles its mean is 1.519 and p90 is 2.074.  The near threshold therefore
does real work beyond merely renaming the broad count-one tier.

**HYPOTHESIS.** Root dormancy does not imply internal-node dormancy.  PN
allocation is budget-sensitive, so OR-set inclusion also does not guarantee
matched-cap decision monotonicity.  Runtime, memory, node allocation, and
decisive-to-UNKNOWN regressions are explicit A/B gates below.

### 2.4 Soundness and exact proof obligation

**HYPOTHESIS — proof-ready.** `FreeTempoPreservesForcing`:

> Let `Q` be a legal partial attacker turn with no live defender
> count-at-least-four window and `tau(F_A(Q)) >= 2`.  For any legal attacker
> placement `b`, either `b` completes six for the claimant, or `Q+b` has no
> live defender count-at-least-four window and `tau(F_A(Q+b)) >= 2`.

**HYPOTHESIS — proof sketch.** An attacker stone cannot create a defender-pure
window.  If an old singleton residual is filled, the claimant has completed
six.  Otherwise every old claimant residual set remains nonempty and is
preserved or loses `b`; a hitting set for all new residuals would therefore
hit the old family.  New claimant threats only add constraints.  Hence the
minimum hitting number cannot fall on the nonterminal branch.

**CODE-FACT.** The extension is existential attacker widening, not pruning.
Every retained pair still passes the ordinary forcing classifier, every
defender reply remains universal, and the strict verifier replays legal Choice
edges.  Adding legal attacker options can expose a true WIN; it cannot mint a
false WIN.  There is therefore **no Lean pruning theorem required** for `J2near`.
`FreeTempoPreservesForcing` licenses routing the added pair through the existing
forced dispatcher; it does not say that any omitted spare is irrelevant.

### 2.5 Engine design and kill criteria

**HYPOTHESIS.** Add `free_tempo_j2near: bool` to `WidthOptions` near
[`tss_solver.rs:662`](../packages/hexfield_eq/rust/src/tss_solver.rs#L662),
with a named default-off profile `vcf_pair_j2near` and environment flag
`TSS_VCF_J2NEAR`.

**HYPOTHESIS.** At
[`WideTurnGate`](../packages/hexfield_eq/rust/src/tss_solver.rs#L9182):

1. derive the first-alone threat family from count-three windows through `a`;
2. require `tau>=2` and that `a` answers every live defender threat;
3. count post-first count-one window membership by axis;
4. append cells whose second-largest axis multiplicity is at least four;
5. keep the existing unordered deduplication, exact pair evaluation, lazy
   frontier, and certificate materialization unchanged.

**HYPOTHESIS.** The post-first weak scan can be derived without a general legal
scan: retain turn-start count-one windows not promoted by `a`, plus the at most
18 formerly empty windows through `a`.  A simpler first prototype may
apply/undo only the rare turn-buying firsts.  The final implementation must
measure whether that loses the stateless P7 generation gain.

**HYPOTHESIS.** At
[`attack_single_children`](../packages/hexfield_eq/rust/src/tss_solver.rs#L6922),
perform the same count-one axis scan for a SecondStone root whose existing
first has already bought the turn.  Assign ordinary support ordering only;
do not route through quiet-turn or ranked-zone flags.

**HYPOTHESIS — prediction.** At cap 100,000, unbounded horizon, 256 MiB TT,
the flag should produce three canonical-verifier WINs with terminal lines and
zero verifier failures.  The exact root moves above should appear in the
candidate trace.

**HYPOTHESIS — kill criteria.** Kill the mechanism if any witness remains
UNKNOWN after its candidate is present, if any verifier or D6 check rejects,
or if no verified win beyond the seeded three appears on a broader certified
miss sample.  Block default-on adoption if eligible-root p90 child multiplier
exceeds 2, if decision-identical cohort median wall exceeds 1.05x or p95 wall
exceeds 1.20x, or if any existing decisive row becomes UNKNOWN at matched cap.
Flag-off candidate digests must remain byte-identical.

## 3. Ranked finding 2 — direct rank-two defender cover plans

### 3.1 Precise claim

**CODE-FACT.** `WideTurnGate::evaluate_pair` already constructs the exact
post-pair claimant threat family and computes its hitting number.  It returns
only `(result, prior)`.  At the following defender node,
[`forced_defender_pair_plan`](../packages/hexfield_eq/rust/src/tss_solver.rs#L3394)
reconstructs the same response structure by deriving a kernel, applying every
kernel first stone, rerunning `threats::analyze`, deriving second replies, and
checking reverse-order final-key equality.  The redundant apply/analyze loop is
at lines 3434–3462.

**HYPOTHESIS — proof-ready.** Let `F` be the nonempty residual family of live
claimant count-four/count-five windows at a nonterminal defender node.  Every
edge has one or two cells.

- If the defender has budget one and `tau(F)=1`, the exact explicit reply set
  is `intersection(F)`, so it has at most two cells.
- If the defender has budget two and `tau(F)=2`, define
  `M(F)={{x,y}:x!=y and every E in F contains x or y}`.  Then `|M(F)|<=4`,
  and at most four cells occur in all pairs in `M(F)`.

**HYPOTHESIS — proof sketch.** If `F` has two disjoint edges, every two-cover
chooses one endpoint from each, giving at most `2*2=4` pairs and four kernel
cells.  Otherwise `F` is pairwise intersecting.  Since `tau=2`, its total
intersection is empty.  A rank-two pairwise-intersecting family with empty
total intersection is the triangle case, which has three minimum covers and
three kernel cells.  Singleton edges reduce to the forced-endpoint subcase.

**HYPOTHESIS — game bridge.** At a defender FirstStone node with `tau=2` and
no defender own-win-now:

1. every coordinate occurring in `M(F)` is an empty of a live claimant threat
   and was legal at turn start;
2. neither a singleton prefix nor the complete defender pair can make six,
   because no defender-pure count-four window existed before the turn;
3. after the first endpoint `x`, the residual threat family has `tau=1`, and
   its legal forced seconds are exactly the mates of `x` in `M(F)`;
4. both orders are legal and reach the identical completed-turn state by P3.

**HYPOTHESIS.** The current dynamic validation is therefore theorem-redundant.
Pure VCF defender width is universally tiny; the optimization target is the
generation machinery, not an unrecognized large reply class.

### 3.2 Evidence

**MEASURED.** Fourteen checked-in certified forcing lines produced 91 attacker
turns and 63 `tau=2` defender boundaries.  Those boundaries had mean threat
universe 3.98 cells, mean kernel 3.84 cells, and mean 3.05 unordered minimum
cover pairs.  The pair-count histogram was:

| minimum pairs | states |
|---:|---:|
| 1 | 3 |
| 2 | 4 |
| 3 | 43 |
| 4 | 13 |

**MEASURED.** All 192 line-derived reply pairs satisfied the residual-`tau=1`,
mate-legality, no-own-win, and pair-commutation invariants.  Twelve eligible
frozen human-160 roots contributed another 37 pairs with zero violations.

**MEASURED.** An independent exhaustive model checked all 33,861 nonempty
rank-one/rank-two families on universes of up to five cells.  It included
4,073 `tau=2` families and found zero bound or factorization violations; the
maximum minimum-pair count was four.  Five cells covers the maximum threat
universe observed in the real probe.  The executable evidence is
`.scratch/hypergraph_k2_exhaustive.py` and its JSON output.

**MEASURED.** The production-shaped P7 residue attributes 20.4% of solver wall
to forced defender generation, while the deeper F19 residue attributes 36.0%
([`SOLVER_NOTES.md`, lines 208–223](SOLVER_NOTES.md#L208) and lines 447–457).
The cardinality data explains why this can coexist with at most four actual
children: most of the cost is reconstructing and validating a tiny plan.

### 3.3 Engine design

**HYPOTHESIS — rung 1.** Add
`rank2_cover_plan(F) -> {kernel:[Cell;<=4], pairs:[Pair;<=4]}` beside the shared
hitting-kernel helpers.  At `defender_pair_children`, scan the live threat
family once, enumerate `M(F)` directly, and construct each completed-turn key
from the root state plus `[x,y]`.  Retain the old apply/analyze loop behind a
debug equality oracle.  Flag: `TSS_DIRECT_K2_COVER_PLAN`.

**HYPOTHESIS — rung 2.** `evaluate_pair` already has `F`.  Return a compact
cover plan on the lazy pair edge and consume it when that defender node is
selected.  Flag this separately as `TSS_CARRY_K2_COVER_PLAN`; 63–67% of lazy
frontier entries are never expanded, so retained-byte cost can erase the
rescan saving.

**HYPOTHESIS.** Do not allocate a `Vec` per edge.  Four kernel coordinates and
a six-bit unordered-pair mask fit a small fixed representation.  Canonical
coordinate order remains presentation/order only; membership comes solely
from the exact cover predicate.

### 3.4 Exact licensing theorem

**HYPOTHESIS — Lean target.** The pruning-shaped operation is omission of
non-covering turns and reverse duplicate orders:

> `forcedB2PairQuotient`: Let `P` be a post-opening, nonterminal defender
> FirstStone position, with no defender own-win-now, live claimant threat
> family `F`, rank `F<=2`, and `tau(F)=2`.  The ordered legal defender traces
> that do not concede an immediate claimant completion are exactly the two
> orders of members of `M(F)`.  Both orders are legal, nonterminal, and reach
> the same completed-turn position.  Moreover `|M(F)|<=4`.  Therefore one
> Universal child per unordered member of `M(F)` preserves claimant WIN
> exactly.

**HYPOTHESIS.** This theorem composes the rank-two cover lemma, threat
permanence/T6, and P3 from
[`PROOF_TSS_DEFENDER_ZONES.md`](PROOF_TSS_DEFENDER_ZONES.md).  The debug oracle
must remain until the theorem and independent verifier path agree on every
supported phase and terminal guard.

### 3.5 Prediction and kill criteria

**HYPOTHESIS — prediction.** Direct mode should preserve statuses, nodes,
certificates, final position keys, and proof hashes while reducing
`D_FORCED_GEN` wall by 15–25%.

**HYPOTHESIS — gates.** Require byte identity on the existing 6,443-position
battery and zero mismatch between direct and dynamic plans on all P7/F19,
human, puzzle, self-play, and forcing-line nodes.  Advance direct mode only if
the measured defender-generation block falls at least 10% with no total-wall
regression.  Kill carry mode if retained frontier bytes grow more than 5% or
its gain does not exceed direct recomputation.

## 4. Ranked finding 3 — verifier-mediated sibling proof transplantation

### 4.1 Why equivalence must be checked, not guessed

**MEASURED.** Naive minimum-coverer interchange is false.  The prior
domination hunt found 27 sibling mismatches, 22 outcome-grade, including four
positions where both opposite outcomes were proved.  In the canonical example,
two empties of the same count-four threat are both minimum hits, yet one permits
an attacker WIN and the other completes a defender counterfork and refutes it.
See `git show 6b853c0e:HUNT_REPORT_DOMINATION.md`.

**CODE-FACT.** Count profile, shared threat membership, proximity, and
frontier-inertness do not encode all other incident windows.  P2's dead-spoke
hypotheses are load-bearing, as already explained in
[`PROOF_TSS_DEFENDER_ZONES.md`](PROOF_TSS_DEFENDER_ZONES.md).  No signature in
this report is authority for sibling equivalence.

### 4.2 Precise safe experiment

**CONJECTURE.** At a zone-free forced Universal with sibling successor states
`Q_1,...,Q_k`:

1. solve one child `Q_i` and materialize its proof subtree as a standalone WIN
   certificate;
2. for an unsolved sibling `Q_j`, rebind/rekey that proof skeleton to the exact
   sibling root;
3. run the ordinary strict verifier on `Q_j`;
4. only if it accepts, install the verified object as a solve-local exact
   `ProvenFragment` for `Q_j`; otherwise make no inference and search normally.

**CODE-FACT.** This is not support hashing, a local-frame theorem, or
"prove one, delete all."  Every consumed sibling receives its own strict
verification.  A coverage signature may order attempts; it may not license
reuse.

**MEASURED.** The target population is large: 3,013 of 3,412 forced budget-one
nodes in the domination hunt had multiple coverers.  The active-sibling
acceptance rate has not been measured.

**MEASURED.** A different but relevant support hunt rebound the root and
translated clocks after adding 4–32 remote stones.  The unchanged proof body
then passed later strict-verifier checks in 77.8–96.1% of trials.  Strict
unchanged transfer was 0 because complete `RootBinding` equality rejected
first.  These rates motivate trying a rebased skeleton; they are **not** an
estimate for active threat siblings.  See
`git show 3cd224fe:HUNT_REPORT_CERT_SUPPORT.md`.

### 4.3 Soundness, implementation, and gates

**CODE-FACT.** No new equivalence or pruning theorem is needed.  The sole
licensing implication is the existing verifier soundness boundary:

> `Verify(Q, RebindRoot(C,Q), WIN) = true -> AttackerWins(Q)`.

The ordinary Universal composition theorem then licenses the parent.  A
rejection means nothing.

**CONJECTURE.** Start in shadow mode only on legacy, zone-free
`implicit_dispatch` Universals.  Use the existing materializer near
[`WideProofMaterializer`](../packages/hexfield_eq/rust/src/tss_solver.rs#L7180)
and exact fragment path near
[`WidePnSearch::expand`](../packages/hexfield_eq/rust/src/tss_solver.rs#L6364).
Eliminate or rederive shared replay keys when making the child subtree
standalone.  Flags:
`TSS_SIBLING_CERT_TRANSPLANT=off|shadow|consume`.

**CONJECTURE — prediction.** If active sibling proofs share enough structure,
at least 15% of attempted siblings will strictly verify and fixed-cap search
nodes will fall at least 5% on multi-coverer rows.

**CONJECTURE — kill criteria.** Kill if shadow acceptance is below 5%, if
zero-accept rows add more than 2% wall, or if verified coverage does not
improve.  Any disagreement between the transplanted-child result and the final
whole-root verifier is an immediate stop.  Do not redesign certificate support
or weaken root binding for this experiment.

## 5. Ranked finding 4 — a sound local deadline, and an open-ray impossibility

### 5.1 Finite forcing-region theorem

**HYPOTHESIS — proof-ready.** Let `P` be post-opening, `A0` its initial claimant
stones, and `D0` its initial defender stones.  A finite set `R` is a
`VcfForcingRegion(P,R)` when:

1. `A0 subset R` and `R` is disjoint from `D0`;
2. **A-closure:** for every length-six window `W`, if `W` contains no `D0`
   stone and `|W intersect R|>=2`, then `W \ D0 subset R`;
3. **D-threat closure:** for every `W`, if `W` contains no `A0` stone and
   `|(W intersect (D0 union R))|>=4`, then `W \ D0 subset R`.

Let `e=|R \ Stones(P)|`, the initially empty cells of `R`.

**HYPOTHESIS — induction.** Before the first implicit-dispatch escape:

- every normal claimant `T` cell lies in a live claimant window with at least
  two actual claimant stones in `R`, so A-closure contains it;
- every `G1` second cell lies in a window containing the prior claimant stone
  and the just-played first, so A-closure contains it;
- every forced defender hit lies in a claimant threat window and is in `R`;
- every claimant block of a possible defender count-four window is in `R` by
  D-threat closure, because future pre-escape defender stones are themselves
  already in `R`.

Initial masks are permanent.  Treating every `R` cell as possibly either
future color is an over-approximation, so the induction does not assume branch
compatibility.

**HYPOTHESIS — semantic clock.** Any `vcf_pair_complete` contract WIN can be
normalized to an actual claimant win within `e+2` future placements.  The `+2`
is not bookkeeping slack: a legal defender reply omitted by implicit dispatch
is not an atomic terminal event.  T6/lambda replay may consume the remaining
one or two physical defender placements before the claimant's surviving threat
completes; the completion cells are in `R`.

**HYPOTHESIS.** In inclusive Lean clock notation, if `nextPly` is the next
placement, the deadline is `nextPly+e+1`.  In the Rust absolute placement
clock it is `placements_made+e+2`.

### 5.2 Exact Lean licensing statements

**HYPOTHESIS — Lean definitions and theorem targets.** These are restricted
strategy statements, not `AttackerWinsFrom` for the full game:

```lean
def regionEmptyCount (P : Position) (R : Finset Cell) : Nat :=
  (R.filter fun c => c ∉ P.stones).card

def VcfForcingRegion (P : Position) (R : Finset Cell) : Prop :=
  attackerCells P ⊆ R ∧ Disjoint R (defenderCells P) ∧
  (∀ W, count .defender W P = 0 →
     2 ≤ (W.cells ∩ R).card → W.cells \ defenderCells P ⊆ R) ∧
  (∀ W, count .attacker W P = 0 →
     4 ≤ (W.cells ∩ (defenderCells P ∪ R)).card →
     W.cells \ defenderCells P ⊆ R)

theorem vcfPairCompleteWin_bounded_of_forcingRegion
    (hpost : P.phase ≠ .opening)
    (hR : VcfForcingRegion P R)
    (hwin : VcfContractWin P nextPly) :
    ∃ σ, AttackerWinsBy P nextPly
      (nextPly + regionEmptyCount P R + 1) σ

theorem noContractWin_of_forcingRegion_noWinBy
    (hpost : P.phase ≠ .opening)
    (hR : VcfForcingRegion P R)
    (hno : ∀ σ, ¬ AttackerWinsBy P nextPly
      (nextPly + regionEmptyCount P R + 1) σ) :
    NoContractWin VcfPairComplete P nextPly

theorem noContractWin_of_forcingRegion_census
    (hpost : P.phase ≠ .opening)
    (hR : VcfForcingRegion P R)
    (hlb : regionEmptyCount P R + 2 <
      censusLowerBound P.phase (attackerCensus P)) :
    NoContractWin VcfPairComplete P nextPly
```

**CODE-FACT.** The current census lower-bound table is, for claimant census
`c=0..5`, `FirstStone=[10,10,9,6,2,1]` and
`SecondStone=[12,12,9,5,4,1]`.  The existing evaluator also requires checked
coordinate safety and `h_rem<=8`.  Thus only `e<=6` is implementation-useful.

### 5.3 The pointwise closure is structurally impossible on the target

**HYPOTHESIS — proof-ready open-ray theorem.** If `P` has a claimant-alive
window `W` with at least two claimant stones and one axial continuation ray of
shifted length-six windows contains no initial defender stone, then no finite
`VcfForcingRegion(P,R)` exists.

**HYPOTHESIS — proof sketch.** `A0 subset R` and A-closure first force all of
`W` into `R`.  Shift `W` one cell along the defender-free ray.  The shifted
window overlaps `W` in five `R` cells, so A-closure forces its new endpoint.
Induct over shifts.  Infinitely many distinct endpoints enter `R`, contradicting
finiteness.

**HYPOTHESIS — exact Lean target.** With a signed shift direction chosen so
consecutive windows share five cells:

```lean
theorem no_finite_forcingRegion_of_openCarrierRay
    (halive : count .defender W P = 0)
    (hcarrier : 2 ≤ count .attacker W P)
    (hray : ∀ n : Nat,
      count .defender (W.shift (n+1)) P = 0) :
    ¬ ∃ R : Finset Cell, VcfForcingRegion P R
```

**MEASURED.** A direct blocker scan found the open-ray obstruction on:

| cohort | obstructed roots |
|---|---:|
| `selfplay_v1` | 2,567 / 3,255 |
| `human_v1` | 2,651 / 2,720 |
| `puzzle_v3` | 464 / 468 |
| all grinds | 248 / 248 |
| grind deep WIN / raw below-50k / cap50k | 57/57, 97/97, 94/94 |

**MEASURED.** An independent monotone saturation, aborting only after 100
initially empty region cells, found fixed points on 673 self-play roots and 58
human roots, every one with `e=0`; it found none on puzzles or grinds.  The
census inequality refuted 625 supported self-play roots and all 58 human roots.
These are trivial/sparse roots, not hidden grind savings.  No positive-`e`
fixed region occurred.

**HYPOTHESIS — disposition.** The semantic theorem is useful and the open-ray
lemma explains exactly why the tested abstraction fails.  The pointwise set
closure is **closed for the grind objective**; this is stronger than an
empirical cap failure.  It should be formalized as a boundary result, not built
into the hot path.

### 5.4 The only plausible reopen direction

**CONJECTURE.** A future deadline must track compatibility, not just possible
cells.  A branch-indexed or relational invariant could avoid placing every
mutually exclusive future claimant stone into one `R`, and attach a
well-founded budget/rank to threat-lineage states.  The open-ray theorem does
not refute such a relation; it refutes any pointwise set satisfying the two
closure clauses above.

**CONJECTURE — test and kill.** Before Lean work, require a checker that emits
an independently replayable relational witness with deadline at most eight.
Run a preregistered 20-row sample from the raw cap50k and width-exhaust strata.
Kill the abstraction if it yields no witness, if its state count exceeds the
search it is intended to replace, or if any transition cannot be checked from
local window facts.

**HYPOTHESIS — dormant engine seam.** If a different abstraction ever passes
that gate, place `forcing_region_short_deadline` beside
[`evaluate_interior_census_gate`](../packages/hexfield_eq/rust/src/tss_solver.rs#L252),
call it before the global-horizon census block at line 6434, and use local
`h=e+2` without mutating `semantic_horizon`.  A consume mode needs a
verifier-recomputed `ForcingRegionCensus` leaf and the Lean theorem.  Shadow
flag: `TSS_LOCAL_FORCING_DEADLINE=off|shadow|consume`.

## 6. Ranked finding 5 — an exact completed-turn quotient for unforced defense

### 6.1 Scope and construction

**CODE-FACT.** This finding is outside pure `vcf_pair_complete`, whose pending
attacker turns always leave `tau=b`.  It is an exact reference grammar for a
future Group-2/unforced width, not a proposal to merge or replace the existing
Group-2 certificate lane.

**HYPOTHESIS — proof-ready.** Let `P` be a nonterminal defender FirstStone
position with budget two, no defender own-win-now, nonempty old claimant threat
family `F`, and `tau(F)=1`.  Define:

- `L0=Legal(P)`, the old-legal cells;
- `P_x=P+x` for `x in L0`;
- `N_x=Legal(P_x) \ L0`, cells legalized specifically by `x`;
- `Cov_old={{x,y} subset L0 : x!=y and {x,y} hits every E in F}`;
- `Cov_new={(x,y) : x in L0, y in N_x, and {x,y} hits every E in F}`.

**HYPOTHESIS.** Every empty of an old threat is already in `L0`.  Therefore a
newly legal `y in N_x` cannot hit an old threat, and every member of `Cov_new`
must put its old-legal first `x` in `intersection(F)`.

**HYPOTHESIS.** Every completed defender turn is exactly one of:

1. an old/old cover, whose two legal orders reach the same completed state and
   are represented once by P3;
2. a directed old/new cover in `Cov_new`;
3. an uncovered turn, which leaves a claimant count-four/count-five window
   alive so that the claimant fills its at-most-two empties next turn.

This quotients complete turns.  It does **not** assert the parked, false-shaped
claim that a non-hitting first placement is statewise dominated by a hitter
before the second placement is known.

### 6.2 Evidence and economics

**MEASURED.** Five eligible frozen human-160 roots had 346–581 old-legal cells.
Across them:

- raw ordered old-legal two-stone turns: 1,047,938;
- unordered old-legal covering pairs: 3,501;
- directed newly-legal covering pairs: 88;
- sequential cover-tree lower bound
  `sum |L0| + 2|Cov_old| + |Cov_new| = 9,347`;
- atomic completed-turn children
  `|Cov_old|+|Cov_new| = 3,589`, or 38.4% of that lower bound.

**MEASURED.** The structural reduction is about 2.6x on these eligible nodes,
but the resulting average fanout is still hundreds of turns.  Eligibility was
5/160 roots.  This is a capability/reference result, not evidence that it
beats the existing ranked zone on wall time or verified coverage.

### 6.3 Exact licensing theorem

**HYPOTHESIS — Lean target.** The verifier may omit raw traces only after this
completed-turn theorem is proved:

> `unforcedB2CompletedTurnQuotient`: Under the hypotheses above, every legal
> defender trace either (a) is one of the two orders of a unique
> `Cov_old` representative, (b) is a directed member of `Cov_new`, or (c)
> leaves an old threat unhit, from which the claimant wins within two further
> placements.  Therefore a Universal certificate containing one proof per
> member of `Cov_old union Cov_new`, plus theorem-closed uncovered turns,
> proves the original node.

**HYPOTHESIS.** The proof must include terminal prefixes, newly legalized
second cells, and the old-threat legality lemma.  It must not compose with a
ranked-zone omission until a separate pair-zone theorem exists.

### 6.4 Engine sketch and kill criteria

**HYPOTHESIS.** Add `WidePnMove::UnforcedDefenderPair` and a verifier arm that
independently rederives `L0`, every `N_x`, `F`, `Cov_old`, and `Cov_new`.
Shadow/consume flag:
`TSS_UNFORCED_TURN_QUOTIENT=off|shadow|consume`.  First compare it with a
full-legal unforced reference, isolated from Group-2 ranked zones.

**HYPOTHESIS — prediction.** Atomic representation should reduce sequential
unforced turn nodes by 2x–3x on `tau=1` nodes.

**HYPOTHESIS — kill criteria.** Kill it as a production route if the existing
ranked per-placement zone is faster or finds more verified wins at fixed cap,
if atomic fanout remains cap-prohibitive, or if eligible prevalence stays below
1% at the intended operating cap.  It may remain valuable as an exact oracle
grammar even after an economic kill.

## 7. Ranked finding 6 — cheap `NoContractWin` base leaves

### 7.1 `NoJointCarrier`

**HYPOTHESIS — proof-ready.** At a nonterminal claimant FirstStone node with no
own win-now, a post-pair threat can arise only by:

- hitting a current claimant-pure count-three window at least once; or
- hitting both cells of a current claimant-pure count-two carrier window.

Every admitted pair has `tau>=2`, so it must create at least two distinct
threat windows.  A sufficient linear-time refutation predicate is therefore:

1. fewer than two count-three windows;
2. no count-three/count-two pair has intersecting empty sets; and
3. no two distinct count-two windows share an unordered empty-cell pair.

The last condition is checked by hashing the six unordered pairs from each
four-empty count-two window.

**HYPOTHESIS — exact Lean theorem.** This theorem licenses a restricted-search
refutation leaf:

```lean
theorem noContractWin_of_noJointCarrier
    (hphase : P.phase = .firstStone)
    (hnt : Nonterminal P)
    (hnow : ¬ ownWinNow P .attacker)
    (h : NoJointCarrier P) :
    NoContractWin VcfPairComplete P nextPly
```

**MEASURED.** Root prevalence among supported FirstStone rows was:

| cohort | hits / eligible FirstStone roots |
|---|---:|
| `selfplay_v1` | 482 / 1,610 = 29.94% |
| `human_v1` | 182 / 1,322 = 13.77% |
| `puzzle_v3` | 8 / 331 = 2.42% |
| 248 grinds | 0 / 193 = 0% |

**MEASURED.** Every corresponding main4 hit was already UNKNOWN after exactly
one expansion / two reported nodes, with median tens of microseconds.  In the
deep-labeled human slice, 24/106 fresh-turn rows hit; all were two-node
Unknowns and none was a certified WIN or LOSS.  This certificate saves proof
representation, not target search wall.

**HYPOTHESIS.** Fold the predicate into `WideTurnGate::build` immediately
before first-candidate generation at
[`attack_pair_children`](../packages/hexfield_eq/rust/src/tss_solver.rs#L6582).
Shadow must assert `NoJointCarrier -> attack_pair_children.is_empty`; one
mismatch kills the predicate.  Do not pay a second full window scan.

### 7.2 Exact `NoAdmissibleFirstTurn`

**HYPOTHESIS — proof-ready.** A stronger base leaf independently enumerates
exact `T(P)` and exact `S(P,a)`, and checks that every pair fails at least one
of: new claimant threat, defender-win-first answer, or `tau>=2`.

```lean
theorem noContractWin_of_noAdmissibleFirstTurn
    (hphase : P.phase = .firstStone)
    (hnt : Nonterminal P)
    (hnow : ¬ ownWinNow P .attacker)
    (h : ∀ a ∈ T P, ∀ b ∈ S P a,
      ¬ AdmissibleForcingPair P a b) :
    NoContractWin VcfPairComplete P nextPly
```

**CODE-FACT.** Search already computes this fact when a fresh attacker node
has no children.  The new value is an independently checkable refutation leaf,
not fewer search nodes.

**MEASURED.** One-expansion Unknown proxies were 85/168 self-play, 78/160
human, and 13/34 puzzle fresh-turn screens.  Fifty-eight of 106 deep-labeled
human fresh-turn Unknowns have two-node win passes.  No grind acceleration is
predicted.

**HYPOTHESIS — implementation boundary.** Add these only to the planned
restricted-strategy refutation DAG.  The verifier must rederive window facts
independently rather than share the finder helper.  Never translate either leaf
to full-game LOSS.  A useful adoption gate is compact-leaf bytes and verifier
time versus storing/replaying the empty expansion, not coverage.

### 7.3 Experiment contract

**HYPOTHESIS — implementation and flags.** Put the cheap predicate at the
`WideTurnGate::build` seam named above and emit the exact-enumeration leaf only
after ordinary pair generation returns no child.  Gate both with
`TSS_NO_CONTRACT_BASE_LEAVES=off|shadow|consume`; `consume` must remain
unavailable until the restricted-refutation DAG and its independent verifier
arms exist.  The shadow path records the proposed leaf and compares it with
the ordinary exact child set without changing a node value.

**HYPOTHESIS — prediction.** On the frozen cohorts, `NoJointCarrier` should
reproduce the prevalence table and imply an empty exact attacker-child list
with zero exceptions.  `NoAdmissibleFirstTurn` should exactly match an empty
enumeration.  Shadow mode should change neither root status nor proof digest
and should add less than 1% cohort wall.  Once consumed, either leaf should be
smaller and faster to verify than a replayable empty expansion; no grind-node
reduction is predicted.

**HYPOTHESIS — kill criteria.** Any predicate false positive, verifier
disagreement, flag-off digest change, or full-game LOSS exposure kills the
design immediately.  Remove the predicates from the hot path if shadow cost
exceeds 1%.  Do not adopt them as compact leaves if serialized bytes or
verification wall fail to improve over the exact empty-expansion alternative.

## 8. Incubator result — reply freedom as an ordering feature

**HYPOTHESIS.** For a pending attacker pair, augment the static prior with

`Theta(F) = (tau(F), mu(F), componentProfile(F))`,

where `mu(F)=|M(F)|`, the exact number of minimum defender cover pairs.  All
pending pure-VCF pairs have the same finite `tau=2` class, so `dn_from_tau`
cannot distinguish one forced reply from four.  Threat count also does not
encode defender freedom.

**MEASURED.** On the 63 certified-line `tau=2` boundaries, `mu` realizes every
value one through four with histogram `1:3, 2:4, 3:43, 4:13`.  Eighteen states
(28.6%) split into two disjoint threat-hypergraph components, each of hitting
number one; their exact reply pairs factor as
`intersection(F1) × intersection(F2)`.

**MEASURED.** Only 5/18 split states remained split at the next attacker turn,
although 14/18 remained `tau=2`.  Component factorization is a useful node-local
description, not evidence for persistent independent board lanes.

**CONJECTURE.** Instrument `mu`, kernel size, and component profile, then test
their correlation with child proof cost.  Only if preregistered Spearman
correlation is stable should smaller `mu` become a tie-break under the existing
proof-number classes.  Flag: `TSS_REPLY_FREEDOM_ORDER`.

**CONJECTURE — kill criteria.** Kill if correlation is negligible or changes
sign across F19/human cohorts, or if one fixed-cap A/B loses any decided row.
Generic policy/proximity ordering has already regressed, so this gets one
instrument-first round, not an ordering campaign.  This is ordering only and
requires no pruning theorem.

## 9. What was examined and rejected

### 9.1 Broad attacker width

**MEASURED — rejected.** Full count-one second-stone widening adds 110, 110,
and 56 candidates on the three witnesses.  Loose two-axis `J2` captures them
but has human mean eligible-root multiplier 1.423 and maximum 5.167, versus
1.039 and 2.250 for `J2near`.  Use the near-cross tier first.

**MEASURED — rejected.** Raising the near support threshold from four to five
misses the real `oa-6fda...` certificate.  It is not a harmless cost trim.

**MEASURED — rejected for this mechanism.** Relaxing the post-pair gate to
`tau=1` does not explain any of the three witnesses.  Their missing turns are
already `tau=2`; the omission is solely the spare universe.  A `tau=1`
extension introduces an unforced defender spare and the large turn quotient of
Section 6.

**CODE-FACT — rejected as a substitute.** `quiet_turn_consume` is structurally
inert in the Wide PN route selected by production.  `round3_consume` is much
broader, admits unforced defender zones, and was already prohibitively slow on
the exact witnesses.  `J2near` should be a separate atomic-pair flag.

### 9.2 Scalar and pointwise deadline ideas

**HYPOTHESIS — rejected as unsound.** Stone deficit and the census table are
lower bounds on completion time.  They are not upper deadlines on every
possible forcing win.  A node cap, wall clock, or proof-number threshold is
also not a game-semantic deadline.

**HYPOTHESIS — rejected.** Filling the finite Rust `i16` carrier is both
astronomically nonlocal and incompatible with the Lean game semantics on
`Z^2`.  Implementation overflow guards cannot become a game theorem.

**MEASURED — rejected for prevalence.** The correct-phase proxy
"opponent-as-attacker `Phi<1` at defender FirstStone" fired zero times across
3,255 self-play, 2,720 human, 468 puzzle, and 248 grind roots.  A raw potential
also lacks a deadline because new live windows are born dynamically.

**CODE-FACT — ledger-closed.** Dynamic touched-window ES greedy defense is
refuted by IL-03, and global position-independent pairing is impossible for
six-in-a-row by IL-02 in
[`IMPOSSIBILITY_LEDGER.md`](../../tss-vcf-width/IMPOSSIBILITY_LEDGER.md).
Neither is revived here.

**HYPOTHESIS — rejected for the target.** The finite pointwise forcing region
is sound, but the open-ray theorem proves it cannot exist on any of the 248
grind roots.  A larger saturation cap cannot fix this structural obstruction.

### 9.3 Defender dominance and equivalence

**MEASURED — rejected as unsound.** "All minimum coverers are equivalent" has
four doubly proven real counterexamples and 22 outcome-grade mismatches.  No
count/profile signature may prune siblings.

**CODE-FACT — rejected for pure VCF.** A forced reply is an empty of a live
claimant threat window, so at least one of its 18 incident windows is live.  It
cannot be a dead cell.  Dead-cell quotienting therefore cannot fire inside the
pure forced kernel.

**MEASURED — rejected for prevalence.** The frozen human-160 scan found only
14 dead cells among 74,524 root-legal cells (0.0188%), across 9/160 positions.
Five `b=2,tau=1` roots produced eight post-hit positions and 3,590 legal spare
cells; zero were dead.  P2 fired only 2/4,001 nodes in the earlier domination
hunt.

**MEASURED — insufficient for soundness.** Frontier-inert but nondead cells
were 3,708/74,524 (4.98%).  P1/P2 require exact mask/support hypotheses;
frontier inertness alone does not survive the counterfork examples.

**MEASURED — rejected for prevalence.** The full affine hex-lattice group
(translation composed with all 12 D6 maps), preserving colored occupancy and
the `SecondStone.first` payload, had zero nontrivial stabilizers on 160 frozen
human roots.  Beyond-D6 stabilizer orbits do not justify a hot-path build.

### 9.4 Board decomposition

**MEASURED — rejected as a persistent product.** Threat hypergraphs split
locally in 18/63 certified `tau=2` states, but only 5 remain split one attacker
turn later.  Prior turn-quotient measurements found independent consecutive
turn interiors on only 0.039–0.162% of forcing cohorts and 0.155% of 100 human
roots, with threat/response coupling above 99.8%.

**HYPOTHESIS — rejected.** A product certificate over node-local threat
components would be unsound without a theorem excluding future cross-component
windows and legality-frontier bridges.  The rank-two cover plan already obtains
the safe local factorization at a maximum fanout of four.

### 9.5 Cheap certificates as a grind optimization

**MEASURED — rejected economically.** `NoJointCarrier` fires broadly on sparse
general roots but zero times on the entire grind target.  Exact no-admissible
turn leaves merely certify an expansion the current engine finishes in tens of
microseconds.  Keep them for refutation provenance, not fail-fast marketing.

### 9.6 Previously buried routes

**CODE-FACT.** The following were not retested as novel proposals because the
ledger or normative plan already closes them:

- blanket radius-two defender trimming (IL-01), while the scoped
  single-window lemma remains intact;
- global six-window matching defense (IL-02);
- dynamic ES greedy defense (IL-03);
- treating radius nine as an arbitrary-substitution constant (IL-04);
- extrapolating the H1152 fixture as population prevalence (IL-05);
- stealing/tempo arguments beyond the exact safe plateau (IL-06 through
  IL-12);
- D6 search-TT folding, which measured zero useful duplicates;
- generic locality for quiet spares, refuted by a unique required move at
  claimant-stone distance six;
- broad horizon laddering and strict support hashing, already economically
  closed in the normative upgrade plan;
- Strix/df-pn probe seeding, epsilon thresholds, TT work replacement, and
  Unknown summaries, which belong to the existing import lanes rather than
  this report.

## 10. Experiment ledger and decision thresholds

| Rank | Proposal | Effect type | Evidence state | First experiment | Advance threshold | Hard stop |
|---:|---|---|---|---|---|---|
| 1 | `J2near` free-tempo seed | capability widening | 3 exact certified lifts; four real cohorts counted | default-off solver A/B | 3/3 witness WINs verify; acceptable cohort wall | verifier failure, witness miss, or matched-cap regression beyond gates |
| 2 | direct K2 cover plan | exact generation reduction | proof-ready; 229 real pairs + exhaustive abstract check | dynamic-oracle identity A/B | `D_FORCED_GEN` at least 10% lower | any child/key/cert mismatch or total-wall regression |
| 3 | sibling proof transplant | verifier-guarded reuse | large target population; active-sibling rate unknown | zone-free shadow verification | at least 15% accept and at least 5% node reduction | below 5% accept, over 2% miss overhead, any final-verifier mismatch |
| 4 | pointwise forcing deadline | restricted refutation | theorem-ready, open-ray impossible on 248/248 grinds | Lean boundary theorem only | reopen only for relational witness on target rows | no short relational witness in preregistered pilot |
| 5 | unforced turn quotient | exact capability/reference grammar | 5 real roots; 2.6x structural reduction | isolated full-legal oracle | 2x–3x node reduction and coverage parity | ranked zone wins economics or atomic fanout remains prohibitive |
| 6 | no-contract base leaves | certified restricted refutation | high sparse-root prevalence, zero grind hits | verifier leaf size/time | compact trustworthy provenance | shared finder/verifier logic or any nonempty-child contradiction |

**HYPOTHESIS.** Only ranks 1 and 2 merit immediate engine prototypes.  Rank 3
merits a cheap shadow harness.  Rank 4 merits formalization specifically
because it supplies both a sound theorem and a sharp impossibility boundary.
Ranks 5 and 6 should wait for the corresponding verifier/refutation consumers.

## 11. Soundness boundary summary

**CODE-FACT.** `J2near` adds legal claimant choices.  It prunes nothing and requires
no theorem that omitted choices are losing.

**HYPOTHESIS.** Direct K2 planning omits defender traces and therefore requires
`forcedB2PairQuotient`, including legality, terminal, residual-hit, and P3
claims.

**CODE-FACT.** Sibling transplantation consumes only independently
strict-verifier-accepted certificates.  Verification is the authority; no
equivalence theorem is assumed.

**HYPOTHESIS.** A local census dismissal under an unbounded outer search
requires both `vcfPairCompleteWin_bounded_of_forcingRegion` and the census
corollary.  The current pointwise region is target-impossible by the open-ray
theorem.

**HYPOTHESIS.** The unforced completed-turn quotient requires
`unforcedB2CompletedTurnQuotient`; P3 alone is insufficient for newly legalized
directed seconds or uncovered-turn dismissal.

**HYPOTHESIS.** `NoJointCarrier` and `NoAdmissibleFirstTurn` prove only
`NoContractWin VcfPairComplete`.  Exposing either as full-game LOSS would be a
soundness bug.

## Appendix A — reproducibility and artifact ledger

**MEASURED.** The research ran at repository commit `6ba0c9615321`.  No engine,
verifier, Lean, test, or other tracked source was edited.  Scratch artifacts
are intentionally untracked under `.scratch/`.

**MEASURED.** Primary scratch artifacts:

- `.scratch/analyze_width_witnesses.py`
- `.scratch/measure_j2.py`
- `.scratch/width_witness_results.md`
- `.scratch/deadline_cert_probe.py`
- `.scratch/reply_structure_probe.py`
- `.scratch/reply_structure_probe.json`
- `.scratch/reply_structure_summary.json`
- `.scratch/hypergraph_k2_exhaustive.py`
- `.scratch/hypergraph_k2_exhaustive.json`
- `.scratch/analyze_atlas_width.py` and
  `.scratch/measure_spare_extension.py` as an independent reconstruction of
  the width mechanism

**MEASURED.** Important input SHA-256 values:

| artifact | SHA-256 |
|---|---|
| `puzzle_v3.jsonl` | `12B79C6EA132B8D0CAA3C2A9108D5830039CD407B2E774670B59A144EA3495E7` |
| `selfplay_v1.jsonl` | `D8B4256408DFDABF71A90D3653962160BCC05EC66BBA580DD6379149D998B708` |
| `human_v1.jsonl` | `5784DEFE2531DB55360E9860DDDDC9B89B148547B16A0C970FF7D83F407C66B6` |
| `raws/lanec_labels.jsonl` | `48BD13AB76D477FEFFD3067FD18BCA41F0E9E30707A505BDC437C9DAFC6ECB95` |
| opening-atlas `atlas.json` | `797D0AA3F829F016E6E195181F4930970E3D314A6803924A27A0CAC26674EC7E` |
| `OPENING_ATLAS_MAXSOLVE_RAW.txt` | `82CBAAD8B62EB2B2E797F91EB81F70C7E66979C68F7C5367275F7370B18FFDA6` |
| reply summary | `9AFB565BC6BBE0659E04A9DE971EF4DFABA5A7F3A19873AE062173E9006D4C59` |
| abstract K2 output | `7AF6B6D9A25156F2B08A53B4534299BE1386766718233AA6355AFDBD26079466` |

**MEASURED.** Core reproduction commands:

```powershell
python .scratch\analyze_width_witnesses.py
python .scratch\measure_j2.py
python .scratch\deadline_cert_probe.py --cap 100 `
  --sets selfplay_v1 human_v1 puzzle_v3
python .scratch\deadline_cert_probe.py --skip-closure `
  --sets selfplay_v1 human_v1 puzzle_v3
python .scratch\reply_structure_probe.py
python .scratch\hypergraph_k2_exhaustive.py
```

**MEASURED.** The width baseline reproduction used the existing
`opening_atlas_pass1` harness at cap 100,000, unbounded horizon, and 256 MiB TT.
The exact command, emitted rows, certified child joins, terminal lines, and
candidate-count output are preserved in `.scratch/width_witness_results.md`.

**CODE-FACT.** Ground sources read before forming hypotheses were
[`SOLVER_NOTES.md`](SOLVER_NOTES.md),
[`STRIX_SOLVER_COMPARISON.md`](STRIX_SOLVER_COMPARISON.md),
[`INVESTIGATION_PDSPN_IMPORTS.md`](INVESTIGATION_PDSPN_IMPORTS.md),
the sibling [`IMPOSSIBILITY_LEDGER.md`](../../tss-vcf-width/IMPOSSIBILITY_LEDGER.md),
[`PROOF_TSS_DEFENDER_ZONES.md`](PROOF_TSS_DEFENDER_ZONES.md), both the local
historical and sibling normative `PLAN_TSS_SOLVER_UPGRADES.md`, the opening
atlas campaign report/raw, and the relevant wide-PN, candidate, pair-gate,
defender-plan, census, materializer, and verifier seams.

## Appendix B — interpretation limits

**CODE-FACT.** Atlas `win_line` is a principal variation, but the three parent
claims here do not rely on PV terminality alone: each exact post-move child has
a certified LOSS proof for the same claimant, and each lifted parent was
rebuilt and strictly verified.

**HYPOTHESIS.** Candidate-count multipliers are a branching proxy.  They do not
predict df-pn selection order, TT admission, certificate size, wall time, or
matched-cap coverage.  The report intentionally separates them from the
solver A/B prediction.

**HYPOTHESIS.** The finite-region deadline is a theorem about the exact
restricted forcing contract plus implicit-dispatch resolution.  It is not an
upper bound on arbitrary full-game play.

**CODE-FACT.** No result here weakens strict verification, reclassifies Unknown
as full-game Loss, revives a ledgered dead route, or duplicates the existing
Strix/df-pn import lanes.
