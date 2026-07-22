# Horizon R4 report: compiled h14 closure and the deep-rung theory wall

Starting tracked HEAD: `43cbdffb77d412b8b6800a239c2af9a67006623c`  
Branch: `claude/deadline-ladder`  
Campaign target: h24 or higher

## 1. Executive result

**MEASURED — PHASE1_CLOSED.** The standalone Rust endpoint closed the R3
h13/h14 runtime wall: one canonical single-threaded pass returned WIN for all
155 required registry IDs (149 fresh h14 and 6 `SecondStone` h13), with zero
timeouts, completed-negative mismatches, or errors.  The pass took 43.606 s in
the kernel and 64.182 s including Python model construction and transport.

**HYPOTHESIS — scope of that closure.** The compiled search is exact for the
finite interaction model defined in R3 §5.2--5.3 and corrects two inherited
Python rule/clock bugs.  Transfer from that model to every true-game remote
branch still has R3's explicitly unformalized mixed anchored/remote
normalization hypothesis.  “Phase 1 closed” means the requested compiled
endpoint and 155-ID validation wall are closed; it does not silently promote
that hypothesis to a theorem.

**MEASURED — PHASE2_STATUS.** Static two-cover domination genuinely breaks
after six attacker stones.  A complete seven-stone two-axis census has 304 of
1,600 translation classes with cover number greater than two.  The sharp h18
local obstruction is a six-stone precursor whose 48 dangerous activation pairs
need a minimum three-cell defender precover.  The unresolved theorem is the
dynamic coupling of that local tax to defense of the anchored interaction,
not another finite-board or Python-runtime issue.

**CODE-FACT / HYPOTHESIS — LADDER_TOP=14.** The all-phase exact decider ladder
ends at h14.  A static endpoint sublemma reaches fresh h17, and the schedule
identity `fresh Win22 = Win23 = Win24` is proved at definition level, but no
h17/h18 true-game normalization was established and therefore no deeper
registry verdict was claimed.  The stop is a theory wall; 907 eligible d17/18
and 1,049 eligible d21/22 registry IDs show that it is not certificate
exhaustion.

## 2. Claim discipline and scope

**CODE-FACT.** Labels in this report mean:

| label | meaning |
|---|---|
| `MEASURED` | reproduced by a checked executable enumeration or recorded run |
| `CODE-FACT` | follows directly from the inspected schedule, implementation, or artifact structure |
| `PROOF-SKETCH` | mathematical reduction supplied here, not newly Lean-checked |
| `HYPOTHESIS` | required transfer/coupling statement that remains unproved |

**CODE-FACT.** True placement legality is exactly the engine-confirmed rule:
after the opening, an empty cell is playable iff it is within hex distance 8
of an existing stone.  Every reachable state's legal carrier is finite.  R4
does not use an infinite-board approximation.

**CODE-FACT.** The existing formal file
`E:\tss-lean\TssZones\HorizonRound.lean` was read only.  It backs the
two-cover machinery, the positive radius-8 bridge, and the h6 decider; it does
not contain the R4 excursion coupling theorem or the new clock-collapse
theorem.

**CODE-FACT.** R4 changed only scratch Python, a standalone scratch Rust
crate, JSON/JSONL evidence, this report, and the successor state.  It made no
engine, verifier, package, configuration, or Lean edit and made no Git commit.

## 3. Phase 1: the corrected finite endpoint

### 3.1 Quantified semantics

**CODE-FACT.** At a fresh h14 root the kernel decides the R3 endpoint

```text
exists A1, forall D1, exists A2, forall D2, exists A3:
  A wins immediately
  or (D has no completion pair and tau(A final residual family) > 2).
```

**CODE-FACT.** Every displayed action is an unordered two-placement action.
Fresh h13 has the same action prefix but only one final attacker placement;
its endpoint requires more than two distinct singleton completions after D2.
`SecondStone` h13 starts with one attacker placement and ends with an attacker
pair, so it uses the two-cover endpoint.

**HYPOTHESIS.** The search universe contains root-pure completable windows and
root-empty windows incident to that interaction, as in R3.  Exhaustion of this
finite universe is exact conditional on R3's remote/interaction normalization.

### 3.2 Two inherited Python boundary bugs

**MEASURED / CODE-FACT — D1 legality.** R3's Python endpoint applies the true
radius-8 check to A1 but sends D1 through an incidence-pair iterator without a
physical legality check.  Every one of the 155 d13/14 registry IDs contains
such a fringe: 11,973 fringe cells total, with min/p50/p90/max
`56/75/85/128` per root.

**MEASURED.** In `atlas_oa-c515cddcef6134b3`, after
`A1={(2,0),(2,1)}`, the old D1 stream's rank-67 pair is
`{(-19,3),(-19,4)}`.  Both cells have minimum distance 10 from the occupied
post-A1 board, so neither can be the first placement and their mutual adjacency
cannot legalize the pair.  The first 66 stream actions are legal; this is a
concrete boundary, not a hypothetical corner case.

**PROOF-SKETCH — native D1 correction.** Let `L1(x)` mean root-legal or within
distance 8 of an A1 stone.  An unordered D1 pair `{x,y}` is legal exactly when

```text
(L1(x) and (L1(y) or dist(x,y) <= 8))
or
(L1(y) and (L1(x) or dist(x,y) <= 8)).
```

The native iterator tests this relation on physical representatives before
incidence-class deduplication.  It also retains the normalized EMPTY action
and a singleton only when that cell is currently legal.  An initially illegal
fringe singleton has no physical effect and projects to EMPTY.

**MEASURED.** The legality audit checked 136 archived atlas principal
variations through their available five post-root actions; all 136/136 native
witness prefixes were generated and true-rule legal.  After D1, every retained
active residual cell was within distance 5 of an occupied stone, so later
node-local action carriers were radius-8 saturated in this audited endpoint.

**MEASURED / CODE-FACT — `SecondStone` clock.** R3's Python
`build_next_model` first inserts legal singleton A1 masks for `SecondStone`,
then unconditionally inserts two-cell pairs from near windows.  All six d13
`SecondStone` roots contain this bug.  Native first-action counts equal the
correct legal singleton counts 6/6; Python adds 37,723 phase-illegal pair masks
in total, with min/p50/p90/max `3,723/5,874/7,557/8,606`.

**MEASURED.** Fresh first-action counts match Python 149/149.  Across all 155
IDs, the audit reconstructs 154 exact ordered move histories because
`atlas_oa-c515...` and `atlas_full_oa-c515...` are aliases.  These results are
in `.scratch/horizon_r4_python_boundary.json`.

### 3.3 Native design and exact reductions

**CODE-FACT.** `.scratch/horizon_native/` is a dependency-free Rust 2021 crate
with an empty local workspace declaration.  The Python driver constructs an
R3 `NextModel` and streams physical cells plus target/opponent/near residual
families to a persistent single-threaded process.  Cell indices are `u16`;
residual masks, compact bitsets, and sparse antichains avoid engine
dependencies.

**PROOF-SKETCH.** The following reductions preserve the displayed quantifiers:

- A same-owner residual that strictly contains another live residual is
  dominated: completing or being forced to answer the subset happens no later
  than the superset.  Deduplication and minimal-antichain reduction therefore
  preserve the verdict.
- At an ordinary pair node, cells with identical incidence in every live
  family are twins.  Keeping two representatives per class preserves every
  unordered singleton/pair action, including a same-class pair.
- Pair placements commute because terminal checks are performed after the
  first placement when needed and D1 chronology is handled by the physical
  legality relation above.  Later saturated carriers admit unordered pairs.
- If a universal defender node has a live defender residual of size at most
  two, D can complete it; if A's live size-one/two family has no two-cover,
  every defender pair leaves an A completion.  The dual A2 cover identities
  follow from the same hitting-set definition.
- When immediate threats exist, enumerating only their exact cover actions is
  exhaustive: every omitted action loses immediately by definition.
- A stage/placement transposition key is sound for the fixed h13/h14 nesting;
  cache insertion never converts a deadline into a Boolean verdict.

**CODE-FACT.** Certificate choices, frozen continuations, pair synergy, and
cell scores affect ordering only.  Every remaining quotient action stays in an
exhaustive tail.  A deadline returns `timeout`; it never returns `negative`.

**CODE-FACT.** The current `StateKey` stores only the h13/h14 placement prefix.
It must be widened or replaced by a semantic residual-family key before a
deeper-rung implementation; simply accepting a larger horizon in the parser
would be wrong.

### 3.4 Verification

**MEASURED.** The final serialized Cargo run passed 11/11 unit tests, including
D1 geometry/projection, incidence multiplicity, residual antichains,
quantified shortcuts, h13 endpoint cardinality, ordering-only required-cell
hints, and timeout propagation.  A release build then completed from the same
source.  Cargo processes were serialized host-wide.

**MEASURED.** The final Python/native comparison contains four cases:

| case | phase/h | native | native nodes | Python nodes | role |
|---|---:|---:|---:|---:|---|
| `synthetic_fresh_five` | fresh/13 | WIN | 1 | 1 | immediate fresh singleton tail |
| `synthetic_fresh_five_h14` | fresh/14 | WIN | 1 | 1 | immediate fresh pair tail |
| `synthetic_second_five` | SecondStone/13 | WIN | 1 | 1 | immediate shifted-clock action |
| `synthetic_fresh_cross7_negative` | fresh/13 | NEGATIVE | 29,027 | 58,054 | exhausts all A1 actions; detects a legal D1 completion for each |

**PROOF-SKETCH.** In the negative cross case, each refuting D1 action fills a
two-cell completion whose cells lie within distance 5 of an opponent root
stone.  Thus a legal D1 terminal reply exists for every A1, and Python's
illegal-fringe overgeneration cannot change that negative.  The immediate
`SecondStone` match does not validate Python's erroneous exhaustive pair
space; the separate six-root clock audit supplies that boundary.

**CODE-FACT.** Native records those 29,027 replies through the exact
`shortcut_d1_defender_completion` identity (`defender1_actions=0`); the Python
oracle explicitly visits one D1 response per A1.  This suite therefore checks
a universal D1 verdict but contains no completed A2-or-deeper Python oracle
case.

### 3.5 Complete registry pass

**MEASURED.** The inherited R3 Python endpoint attempted the same 155 d13/14
registry IDs at 250 ms per root and returned 155 timeouts, with no completed
verdict.  That frozen boundary is the baseline this phase was required to
close.

**MEASURED.** The canonical run used a 10,000 ms per-root kernel deadline, a
500,000-entry cache limit, one thread, and the final release binary:

| metric | result |
|---|---:|
| eligible/caught/missed | 155 / 155 / 0 |
| completed negatives / timeouts / errors | 0 / 0 / 0 |
| phase/depth split | 149 fresh h14; 6 `SecondStone` h13 |
| kernel total / mean | 43.605881 s / 281.328 ms |
| kernel p50 / p90 / max | 27.118 ms / 304.705 ms / 6.826 s |
| model+transport total | 20.575689 s |
| end-to-end total | 64.181570 s |
| nodes total | 20,978,264 |
| nodes p50 / p90 / max | 75,402 / 145,281 / 1,577,155 |
| universe p50 / p90 / max | 418 / 564 / 898 |
| normalized A1 actions p50 / p90 / max | 37,494 / 64,043 / 212,691 |

**MEASURED.** Every root completed on its first ordered A1 action.  Hint sources
were 137 atlas continuations, 11 frozen continuations, and 7 verifier-accepted
root/child choices.  This is a search-order observation only: the exhaustive
tail remains present and no certificate choice is treated as a proof rule.

**MEASURED.** The preserved initial development pass returned 137 WIN and 18
timeouts; subsequent exact pruning and ordering closed the tail.  Its apparent
p50/p90 improvement relative to the final pass is not a controlled benchmark:
the cache limit and evolving source/binary snapshot differ.  Only the final
single-pass numbers above are authoritative performance measurements.

## 4. Frozen-cohort bite under stated budgets

**MEASURED.** Human and self-play roots used 10 ms/50,000 cache entries per
root.  Puzzle, grind, and forcing roots used 25 ms/100,000 entries.  Deadline
checks occur at solver safe points, so wall time can exceed the nominal budget;
every such row remains a timeout rather than a loss.

| cohort | frozen rows | supported attempted | budget | WIN | timeout | completed negative | raw firing on attempted |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 2,720 | 2,720 | 10 ms | 6 | 2,714 | 0 | 0.220588% |
| puzzle | 468 | 468 | 25 ms | 7 | 461 | 0 | 1.495726% |
| grinds | 248 | 248 | 25 ms | 0 | 248 | 0 | 0% |
| forcing-19 | 19 | 19 | 25 ms | 0 | 19 | 0 | 0% |
| self-play | 3,255 | 3,207 | 10 ms | 0 | 3,207 | 0 | 0% |
| membership total | 6,710 | 6,662 | mixed | 13 | 6,649 | 0 | 0.195137% |

**CODE-FACT.** The 48 unsupported self-play rows are opening roots, whose first
A placement is the forced origin and whose R4 endpoint was intentionally not
implemented.  Within each cohort every supported `(cohort,id)` membership was
attempted exactly once.  Cross-cohort overlap leaves 6,246 distinct attempted
IDs and 416 additional memberships; the 13 WIN memberships represent 12 IDs.

**MEASURED.** The cohort sweep used 949.633 s of native kernel time and
1,257.302 s of model/transport time, or 2,206.935 s end to end.  It produced no
completed cohort negative and no error.  These are deliberately low-budget
bite floors, not a classification of the timeout population.

### 4.1 Literal IDs versus equal positions

**MEASURED / CODE-FACT.** The inherited registry contains 2,941 unique IDs but
only 2,788 exact ordered move histories: 153 duplicate-ID alias pairs, with no
non-null depth conflict.  Proofs transfer across byte-identical move histories,
so literal-ID intersection undercounts some frozen-cohort floors.

**MEASURED.** The honest lower-bound progression is:

| cohort | exact h8 | literal h8∪cert≤10 | move-normalized h8∪cert≤10 | plus native | literal h8∪cert≤14 | move-normalized h8∪cert≤14 | plus native |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 157 | 163 | 163 | 164 | 176 | 176 | 176 |
| puzzle | 20 | 21 | 23 | 27 | 23 | 31 | 31 |
| self-play | 101 | 102 | 102 | 102 | 107 | 107 | 107 |
| grinds | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| forcing-19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| membership sum | 278 | 286 | 288 | 293 | 306 | 314 | 314 |

**MEASURED.** Relative to the move-normalized h10 floor, the native run adds
five cohort memberships (four distinct move histories): one human membership,
four puzzle memberships, with one ID shared between those cohorts.  Relative
to the complete move-normalized ≤14 certificate floor it adds zero new
positions.  Four apparent additions over R3's literal h14 puzzle count are
`atlas_...` aliases of already-certified `atlas_full_...` roots and are not
reported as novel proofs.

## 5. The exact registry validation ladder

**CODE-FACT.** `cert_depth` is the maximum exact-leaf resolution ply in a
verified certificate minus the root placements.  It is a sufficient deadline,
not a claim of minimal winning depth.  All requested sets below are exact
registry-ID sets; no opening certificate is present.

**MEASURED.** Cumulative eligibility is:

| deadline | eligible IDs | distinct move histories | FirstStone IDs | SecondStone IDs |
|---:|---:|---:|---:|---:|
| ≤14 | 278 | 275 | 239 | 39 |
| ≤18 | 1,185 | 1,177 | 964 | 221 |
| ≤22 | 2,234 | 2,226 | 1,712 | 522 |
| ≤24 | 2,234 | 2,226 | 1,712 | 522 |
| ≤26 | 2,566 | 2,558 | 1,950 | 616 |

**MEASURED.** The genuine rung increments and compact sorted-ID hashes are:

| exact depths | IDs | FirstStone | SecondStone | SHA-256 of compact sorted ID array |
|---:|---:|---:|---:|---|
| 13/14 | 155 | 149 | 6 | `EC4DE6586A39A1382232F529D42923B52D8E9318CD2D7A89D22F925C2B5911DB` |
| 17/18 | 907 | 725 | 182 | `68A6D26CAA96DA0D6C3DED50DBC2EEA6362362D7EF3A2817450E6405343A1A2D` |
| 21/22 | 1,049 | 748 | 301 | `C9BF9C64499B03E12EEA56692CE556456EDFBA34B0FE452A418321192865824F` |
| 25/26 | 332 | 238 | 94 | `452115F2E85CF37E70B9DB20CC70BEFA3025B4B1A955448CD980E3A42D79C5` |

**MEASURED.** Of 2,676 depth-stamped IDs, 2,553 (95.404%) have literal depth
greater than 12; that is 86.807% of all 2,941 registry IDs.  Therefore the
inherited “about 40% exceed depth 12” premise is not true under its literal
reading for this registry.  Two nearby but different statistics are 1,062 IDs
at depths 13--18 (39.686% of stamped IDs) and 1,185 IDs eligible at ≤18
(40.292% of all IDs, 44.283% of stamped IDs).  There are 265 undated IDs; no
denominator is silently substituted.

## 6. Phase 2: k-stone cover and excursion obstruction

### 6.1 Static cover growth

**PROOF-SKETCH.** R3's six-stone lemma is necessarily single-axis: two
distinct geometric lines with four attacker stones each need at least seven
stones because lines intersect in at most one cell.  At seven stones that
counting barrier disappears, so no proof can continue by merely changing the
constant six to ten or twelve.

**MEASURED.** Exhaustive translation-normalized enumeration of seven-stone
unions of two four-stone axis runs produced 1,600 classes:

| cover number `tau` | classes |
|---:|---:|
| 2 | 1,296 |
| 3 | 288 |
| 4 | 16 |

**MEASURED.** Thus 304 classes defeat a single defender pair.  The smallest
named cross has seven stones, six residual windows, cover number 4, and
radius-8 maximum construction hop 1 conditional on a legal seed.

**MEASURED.** Exact named nonterminal obstructions show continued growth:

| attacker set | stones | residual cover number | radius-8 max hop |
|---|---:|---:|---:|
| two-axis cross | 7 | 4 | 1 |
| two same-axis separated four-runs | 8 | 4 | 7 |
| three-line triangle | 9 | 6 | 1 |
| three-axis star | 10 | 6 | 1 |
| six-line weave | 12 | 12 | 1 |

**MEASURED — bounded one-axis theorem.** For normalized subsets of the carrier
`[-5,10]` meeting base interval `[0,5]` in at least four stones, 6,570 shapes
through k=12 were enumerated; the R3 k≤6 prefix has 259.  The maximum
nonterminal residual cover numbers for k=4..12 are
`2,2,2,2,3,4,4,4,5`.

**HYPOTHESIS — scope limit.** That 6,570-class result is exact only for the
displayed one-axis carrier.  It does not normalize arbitrary separated
components or multi-axis excursions, and therefore is not a complete k≤12
remote theorem.

### 6.2 The minimal h18 coupling obstruction

**MEASURED.** The six-stone nonterminal precursor

```text
X6 = {(-2,0), (-1,2), (0,-1), (0,0), (0,1), (1,0)}
```

has no current four-stone residual family and is radius-8 chainable with
maximum hop 2, conditional on a legal seed.  The exact local suffix is
`forall D0, exists A_activation, forall D_cover, exists A_final`.  The
activation-pair carrier has 24 cells and 48 dangerous pairs.  Activated
families have cover number 3 for 40 pairs and 4 for 8 pairs.

**MEASURED.** A dominance-complete defender precover carrier has 17 cells.  Its
minimum transversal size is 3; one witness is
`{(-4,0),(-3,4),(0,-3)}`.  Therefore “reserve the last defender pair for the
remote family” is false at this local state.

**PROOF-SKETCH.** The obstruction is clock-relevant to fresh h18: after six
attacker stones have formed the precursor, the next of two remaining attacker
pairs can activate a family that the intervening defender pair cannot cover,
leaving the final attacker pair.  The earlier `D0` pair can still pay part of
the required three-cell precover, so this is not by itself a forcing line or a
reachable counterexample.

**HYPOTHESIS — precise open theorem.** A successful excursion-tempo proof must
show that paying those three local preblocks necessarily surrenders an
incompatible obligation in the anchored interaction (or in a mirrored
excursion), or else exhibit a chronological defender strategy that pays both.
Static cover number, radius-8 chainability, and “one reply per excursion pair”
do not establish that coupling.

**MEASURED / HYPOTHESIS.** A bounded census of 6,228 unions of two
consecutive-cross precursors, with relative centers in hex radius 1, found up
to three counted dangerous pivots.  This is exact for that schema only.  An
audit found activations outside the predicate, so the artifact explicitly does
not claim exhaustive latent-pivot normalization.

## 7. Phase 3: clocks, collapse, and where the ladder stops

### 7.1 All root-phase clocks

**CODE-FACT.** Attacker/defender placement quotas at the genuine fresh rungs
are:

| root phase | h13 | h14 | h17 | h18 | h21 | h22 | h24 | h25 | h26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh `FirstStone` | 7/6 | 8/6 | 9/8 | 10/8 | 11/10 | 12/10 | 12/12 | 13/12 | 14/12 |
| `SecondStone` | 7/6 | 7/7 | 9/8 | 9/9 | 11/10 | 11/11 | 12/12 | 13/12 | 13/13 |
| opening | 7/6 | 7/7 | 9/8 | 9/9 | 11/10 | 11/11 | 12/12 | 13/12 | 13/13 |

**CODE-FACT.** Opening has the same quota clock as `SecondStone` but its first
A placement is the forced origin.  Consequently fresh h24 adds only defender
placements after h22, while `SecondStone` and opening h24 add a genuine
attacker placement and need distinct endpoints.

### 7.2 Collapse proof

**PROOF-SKETCH.** Suppose two schedules share the complete prefix through the
last attacker placement and differ only by one or two trailing defender
placements.  If the common prefix already reaches an A terminal, both bounded
predicates are true.  At an ongoing leaf, a D placement cannot create an A
terminal; recursively, the remaining defender-only suffix reaches the
zero-fuel false case.  Substitute these equal false leaves and induct backward
through the shared `exists/forall` prefix.  Legal moves remain nonempty around
a finite nonempty board, so the universal layer is not vacuous.

**PROOF-SKETCH / CODE-FACT.** Applying that argument to the generated schedules
gives

```text
fresh:       Win18 = Win19 = Win20;  Win22 = Win23 = Win24
SecondStone: Win17 = Win18 = Win19;  Win21 = Win22 = Win23
opening:     Win17 = Win18 = Win19;  Win21 = Win22 = Win23
```

No new Lean theorem was added; this remains a definition-level paper proof.

### 7.3 Static endpoints and the stop

**MEASURED / PROOF-SKETCH — fresh h17 static endpoint.** A singleton completion
uses five attacker stones.  With at most eight attacker stones before the
final singleton, two supporting lines would require at least nine stones, so
all such windows lie on one line.  Normalizing a support to `[0,5]` gives the
finite `[-5,10]` carrier.  Exhaustive enumeration checked 944 nonterminal raw
normalized candidates (k=5..8), found at most two singleton completion cells,
and found zero failures of a two-cell defender cover.

**HYPOTHESIS.** This closes only the static final-singleton cover needed by
fresh h17.  It does not prove that all earlier remote placements can be moved
into the interaction while preserving the defender's anchored obligations.

**MEASURED — h18 obstruction.** `X6` shows that the following attacker pair can
raise the local cover requirement to 3 or 4.  The dynamic three-preblock tax is
the first unresolved rung theorem.

**MEASURED — h21/h22 obstruction.** A nonterminal nine-stone cross consisting
of two length-five arms has four distinct singleton completion cells
`{(-1,0),(0,-1),(0,5),(5,0)}`.  A tenth attacker stone preserves them.  Hence a
simple reserved defender pair already fails statically before the h21/h22
dynamic problem is considered.

**CODE-FACT.** No h17/h18 or h21/h22 compiled decider was run, because doing so
would require silently assuming the missing normalization.  The deepest
globally claimed exact normalized decider is h14; the deepest static sublemma
is h17; the deepest schedule equality proved is fresh h24.

**CODE-FACT.** The limiting resource is therefore:

| frontier | status | limiting reason |
|---|---|---|
| h13/h14 | closed | compiled runtime wall removed |
| fresh h17 static tail | closed | finite singleton-cover theorem |
| h17/h18 dynamic | open | excursion precover/anchored-defense coupling |
| h21/h22 | open | stronger four-singleton obstruction plus same coupling |
| fresh h23/h24 | schedule-equivalent to h22 | no h22 verdict exists to transport |
| eligible registry | ample | 907 then 1,049 rung IDs; not exhaustion |

## 8. Reproduction, host discipline, and artifacts

**CODE-FACT.** Core artifacts are:

- `.scratch/horizon_native/`: crate, driver, README, local target;
- `.scratch/horizon_r4_phase1.json`: consolidated registry, parity, cohort,
  identity, runtime, and source-hash evidence;
- `.scratch/horizon_r4_d1_legality.{py,json}` and
  `.scratch/horizon_r4_python_boundary.{py,json}`: rule/clock audits;
- `.scratch/horizon_r4_registry.{py,json}`: exact ID validation ladder;
- `.scratch/horizon_r4_remote.py` and `.scratch/horizon_r4_phase2.json`:
  cover/excursion enumeration;
- `.scratch/horizon_r4_ladder.py` and `.scratch/horizon_r4_phase3.json`:
  all-phase clocks, collapse data, static h17 and h21 results;
- `.scratch/horizon_r4_cert_hints.{py,json}`: seven independently verified
  ordering hints, never pruning rules;
- `.scratch/HORIZON_R4_STATE.md`: lossless successor handoff;
- `.scratch/horizon_r4_hashes.json`: final byte-level SHA-256 manifest.

**MEASURED.** The finite generators reproduced their evidence before report
freeze: registry `2,941/2,676/265`, D1 audit `155/155`, phase-2 output SHA-256
`6E0FF9A660ED7C6F42440E229A22DF1AE8559C31A96178DA202CE0401784911B`,
and phase-3 output SHA-256
`41FEAA6C0B4D38D94DC485BBEF707BDD1843D8CEA0F695F0DCE66DBD0D76397A`.

**CODE-FACT.** The final manifest hashes this report, the state file, every R4
Python/JSON/JSONL artifact, crate source and metadata, and the release binary.
The manifest excludes itself to avoid a recursive hash.  Phase-1's JSON also
embeds hashes for every authoritative input shard.

**MEASURED.** All native search was single-threaded.  Cargo test/build commands
were preceded by a host-wide `cargo`/`rustc` process check and run one at a
time, using only `.scratch/horizon_native/.target`.  No dependency download or
engine build was performed, and the live trainer was not modified.

## 9. Gate summary

**MEASURED — PHASE1_CLOSED.** 155/155 required h13/h14 registry IDs completed
WIN in the canonical native pass; 4/4 Python comparison verdicts matched.

**HYPOTHESIS — PHASE2_STATUS.** Static cover failure is characterized through
k=12 examples and finite scoped censuses; the exact remaining wall is the
dynamic excursion-precover versus anchored-defense tax theorem.

**CODE-FACT — LADDER_TOP=14.** Fresh h24 collapse is established, but the
all-phase verdict ladder cannot cross the h17/h18 theory gate.

HORIZON_R4_PARTIAL
