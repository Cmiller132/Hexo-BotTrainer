# HOSTILE REVIEW: REFUTE-CERT V1

**Review date:** 2026-07-21  
**Target:** `docs/DESIGN_REFUTE_CERT_V1.md` at
`270de2360820ac2f5af3b34e35b2e829157e2a3e`  
**Target SHA-256:**
`D6B07007709ABB2609C3A830FAB09CDA527FA6022C53F7C18FC49FFC42630E2A`  
**Review mode:** hostile, doc-only

## VERDICT: SOUND-WITH-REQUIRED-CHANGES

The central idea is defensible. A phase-aware polarity dual of the positive
`vcf_pair_complete` tree can establish a class-relative non-win without
establishing full-game Loss. At an attacker Choice every admitted turn must be
refuted; at a defender Universal one independently checked member of the
positive reply set is enough. The document also correctly refuses the most
dangerous conversions: the proposed result is typed separately, the ordinary
solve remains `Unknown`, `HardValue(-1)` is forbidden, boundary leaves are
forbidden, raw D6 equality is retained, and a standalone verifier replays the
root and all semantic fields.

That does **not** earn the section 7 GO. The exact proposition and its positive
grammar are not present in the target tree, `T/S` exactness and the pair
quotient are elsewhere recorded as unformalized, and the design gives
`NoPositiveConstructor` authority before publishing a phase-indexed
constructor table. A producer and verifier can agree perfectly on the wrong
finite universe. Differential agreement is not a completeness theorem.

Two further defects are concrete rather than merely missing proof work:

1. Section 3.2 tells the producer to use the current shallow
   `child_is_genuinely_refuted` helper when selecting a Universal
   counterexample. That helper excludes only a child whose **immediate** entry
   is `DepthCutoff`; it accepts an entry with `dn == 0` whose descendant support
   contains a cutoff. The lexicographically first such child can make
   materialization fail even when a later structural counterexample is already
   available. The specified algorithm has no backtracking or recursive
   provenance fixed point.
2. Complement encoding bounds stored `K`, but neither the parser nor the
   verifier bounds regenerated `Q`. Because `S(P,a)` contains `T(P)-{a}`, a
   tiny `NoAdmissibleFirstTurn` artifact can require quadratic pair replay.
   The advertised one-million-stone root limit permits a practically
   unbounded verifier job behind a one-node certificate.

The economic gate can also approve an artifact mode that makes eligible solves
materially slower: verifier replay is bounded, but producer emission wall has
no pass/fail threshold. The 17,957-node `mvp2lvc` size premise has no pinned
measurement in the target tree and conflicts with the available committed
20,000-node cap row, which still reports Unknown/cap saturation. Finally, the
universal twelve-way D6 requirement is false over the accepted `i16` coordinate
domain unless the format restricts roots to a symmetry-closed subset.

No implementation or owner GO is authorized by this review. Apply every
R-item, freeze the literal semantics and wire grammar, and hostile-review the
amended design before source work. The exact-enumeration leaf-only fallback is
the only currently reviewable sub-scope.

## Required changes

### R1 -- Publish the literal phase-indexed proposition before naming acceptance theorems

**Amends:** sections 1.1, 1.3, 2.1, 4.2, 5, and 7.

Define, in one normative block, the positive `VcfPairComplete` judgment that
the negative DAG is meant to refute. It must be mutually phase-indexed over at
least:

- claimant `FirstStone` Choice;
- nonclaimant `FirstStone` and `SecondStone { first }` Universal states;
- claimant completion and tactical constructors;
- opponent terminal/tactical failures; and
- the exact equality-only boundary at which a Universal constructor exists.

Then define `NoContractWin` as the absence of that literal judgment, with all
arguments visible: root state, claimant, phase, placement clock, resolution
clock/horizon, class version, and reachability premise. State and prove the
monotonicity or structural lemma that turns one boundary-free negative DAG into
`forall finite h >= nextPly, not ContractWin ... h`. Do not call that statement
"equivalent" before the clock convention and constructor monotonicity are
fixed.

Separate four theorem layers that section 1.3 currently blends:

1. a model decoder/checker accepts a model byte list implies the model
   `NoContractWin` judgment;
2. compact leaves imply the model negative judgment;
3. the executed Rust decoder/checker is extensionally equal to the model
   checker on the literal bytes; and
4. producer natural support materializes bytes accepted by that checker.

`refuteCertV1_accepts_implies_noContractWin` may name layer 1, not executed
Rust acceptance, until layer 3 exists. `widePnNaturalExhaust_materializes...`
is a producer-correspondence theorem and must not be presented as part of the
negative semantic capstone. The public Rust result must either require a
trusted reachable root input or carry the reachability precondition; phase and
stone-count consistency alone do not prove reachability.

This is not clerical. No Lean file exists in the target tree, and
`STRIX_SOLVER_COMPARISON.md` lines 208-210 explicitly records the `S_exact`
identity and P3 pair quotient as unformalized. Without the literal positive
judgment, the statements "not a tight dispatcher" and "V1 does not accept
`mhs < b`" can be read either as equality-only syntax or as T6's full-legal-set
case. Those readings produce different negative grammars.

**Why load-bearing:** an independent checker can faithfully reject every tree
in the wrong grammar. Only a literal shared theorem boundary establishes that
its exhaustive set is the set quantified by `NoContractWin`.

### R2 -- Freeze `T`, `G1`, `S`, the ordered universe, and the wire expansion as versioned mathematics

**Amends:** sections 1.1, 2.1, 2.2, 2.4, 4.2, 4.3, and 6.

Replace "executable shape" and prose regeneration with exact equations. At a
minimum, define:

- live/owner-pure windows and raw coordinate order;
- `T(P)` as the precise union of claimant count-at-least-two empties and live
  defender count-at-least-four empties;
- `G1(P,a)`, including the count-one-through-`a` promotion and whether the
  count-at-least-two portion is definitionally redundant with `T`;
- `S(P,a)` from the **turn-start** state, not from a post-`a` window cache;
- the ordered occurrence universe
  `U(P) = {(a,b) | a in T(P), b in S(P,a)}`;
- the canonical quotient relation, including the unique orientation when only
  one occurrence exists and the hidden-terminal-prefix rule;
- exact post-pair threat-family construction, defender-win-first precedence,
  and the `tau = 0/1/2/>2` disposition; and
- the complete mapping from every occurrence in `U(P)` to one stored
  disposition or one uniquely regenerated complement leaf.

Publish the actual wire grammar. `NoAdmissiblePair(a)` appears in section 2.2
but not in the grammar in section 2.1; `NoPositiveConstructor(reason)` has no
reason enum; and it is unclear whether passing dispositions are keyed by a row,
an ordered occurrence, or an unordered coordinate pair. These cannot be left
to the encoder implementation.

The positive grammar, producer, Rust verifier, and later model checker must all
be pinned to this same **specification**, while retaining independent
implementations. A semantic change requires a class/format version bump.
Require exhaustive bounded-state comparison to a simple oracle, adversarial
fixtures for every `G1` and stale-defender-block edge case, and exact telemetry
for `|T|`, every `|S(P,a)|`, ordered `Q`, quotient classes, failing reasons,
and stored `K`. Producer/verifier equality remains a useful implementation
gate; it is not the proof of the specification itself.

**Why load-bearing:** one missed ordered pair is a false universal refutation.
The current prose is close to the code but is not yet a definition against
which either implementation can be proved complete.

### R3 -- Select a recursively materializable support, not a shallow zero

**Amends:** sections 1.2, 3.2, 3.3, and 6.

Replace the section 3.2 selection rule with a bottom-up, memoized support
classification over exact arena entries and direct edges:

```text
Support = Structural(negative_plan) | Unresolved(cause_set)
```

The causes must distinguish at least depth cutoff, horizon refusal, census,
node cap, stalled/lazy frontier, unsupported defender boundary, claimant
positive leaf, opponent structural leaf, and unexplained current-source
`Refuted`. At a claimant Choice, `Structural` requires a structural plan for
every independently regenerated passing pair. At a defender Universal, it
requires at least one structural member of exact `K_b`; choose the
lexicographically least **successful plan**, not the least child with a shallow
zero. Try later members when an earlier member is unresolved. Atomic
`DefenderPair` edges must be classified only after both unfolded placements and
their child plan succeed.

The existing helper at `tss_solver.rs` lines 5908-5919 is insufficient: it
accepts any child entry with `dn == 0` unless that entry itself is
`DepthCutoff`. A descendant cutoff still produces such a zero. Section 3.2's
claim that the helper "already distinguishes that case" is therefore only true
for an immediate cutoff.

Derive `NaturalExhaust` from this completed support fixed point after all stage
reopens and bottom-up refreshes. Do not let the termination enum classify the
support before materialization. Add NCE-01 as a mandatory fixture, plus variants
with lazy thunks, a horizon `Refuted`, the current both-winners-to-`Refuted`
terminal arm at lines 6393-6399, and the reverse child order.

**Why load-bearing:** without recursive provenance and alternative selection,
the eligibility gate is neither complete nor deterministic with respect to an
available structural counterline. The independent verifier prevents false
acceptance, but the prescribed producer can silently fail the two named
witnesses and any cohort root with the same shape.

### R4 -- Remove catch-all and unproved compact leaves from the v1 cut

**Amends:** sections 1.3, 2.1, 2.4, 4.2, 4.3, and 7.

For v1, replace `NoPositiveConstructor(reason)` with a closed, typed reason
enum and a state/polarity acceptance matrix. Each reason must name the absent
positive constructor and its exact directly rederived premises. In particular,
distinguish:

- claimant fresh-turn empty admitted-pair set;
- nonclaimant equality-dispatch failure because `tau < b` or the threat family
  is empty;
- opponent own-win-now/terminal; and
- opponent forced tactical loss of the claimant.

`tau > b` is a claimant-positive tactical constructor at a nonclaimant state,
not generic "not tight". Unknown future positive constructors must make an old
verifier reject through the class/version boundary, not fall through a
negative catch-all.

Remove `NoJointCarrier` from full-tree v1 and from the fallback. The referenced
`RESEARCH_DIVERGENCE_1.md` is absent from the target commit; its historical
section 7.1 labels the result "HYPOTHESIS -- proof-ready," reports zero hits on
the grind roots, and predicts no search-wall reduction. The three-condition
argument appears combinatorially plausible, but a different theorem and a
different verifier path are unjustified scope in the first certified negative
format. Re-admit it under a new version only after its model theorem exists, a
pinned source is present, exhaustive bounded-state testing finds no
contradiction, and it beats exact `NoAdmissibleFirstTurn` replay on the frozen
cohort.

Retain `NoAdmissibleFirstTurn` as the sole compact fresh-turn leaf because it is
definitionally an exact expansion with an empty passing set. It still depends
on R1/R2 and R6.

**Why load-bearing:** a negative catch-all silently becomes unsound when the
positive grammar grows. The optional carrier leaf adds theorem and firewall
surface without helping either named full-tree witness.

### R5 -- Turn the verifier firewall into an enforceable trust-base contract

**Amends:** sections 4.1, 4.2, 4.3, and 6.

Publish a transitive source/call-graph allowlist and denylist and enforce it in
CI. The current named-function denylist is too easy to evade by moving logic or
calling a lower shared layer. The independent arm must not call
`Board::windows`, `threats::analyze`, producer ordering/canonicalization,
`WindowStore` summaries, or a neutral helper that computes live windows,
`T/S`, threat families, transversals, `K_b`, quotient classes, or semantic
successors. It should enumerate `(axis,start)` windows from canonical stones
and recount cell ownership itself.

State the deliberately shared trust base precisely. If engine placement and
terminal primitives are shared, independence is about theorem analysis, not
the entire game transition. Require an independent direct-state successor and
terminal cross-check before using an engine-applied child. Bind the ruleset,
coordinate semantics, win length, opening rule, legality radius, phase
schedule, and semantic class version in the verifier policy/header. Data-only
wire structs, tag constants, and raw checksum primitives may be shared, but no
semantic decoder normalization may be shared with the producer unless the
literal decoder is separately modeled and golden-tested.

Tests must include one-sided defect injection, not only seeded differential
agreement: omit one weak promotion, retain one stale defender window, flip one
`tau` case, corrupt one `SecondStone.first`, and make only one implementation
wrong. Pin independent golden vectors from a third simple oracle and audit the
compiled call graph, not only imports.

**Why load-bearing:** two copies written from the same ambiguous prose or both
reading the same incremental window cache are correlated, not independent.
The entire completeness claim rests on preventing exactly that correlated
omission.

### R6 -- Bound regenerated semantic work, not only bytes and stored records

**Amends:** sections 2.5, 2.6, 3.3, 4.3, 4.4, and 6.

Add checked, externally selected verifier limits for all regenerated work:

- root stones and deduplicated direct windows;
- `|T|`, every `|S(P,a)|`, and total ordered `Q`;
- total threat-family memberships and pair-gate operations;
- total `K_b` membership/transversal operations;
- exact-state bytes retained for DAG propositions; and
- absolute wall/CPU and peak heap under the supported offline API.

Preflight counts before allocating quadratic tables. Stream pair evaluation
where possible. Exceeding a work limit returns rejection/unsupported and never
becomes a semantic leaf. The limit is external policy, not selected by the
artifact. Replace the one-million-stone default with a measured supported
maximum or prove a safe memory/work bound for it. Add NCE-02 as an adversarial
verifier test.

The replay gate must report absolute `Q` and work, not just "each pair at most
once." Once can still mean hundreds of billions of times. Include malformed
and valid-but-hostile roots in the robustness battery, with deterministic
termination requirements.

**Why load-bearing:** complement encoding makes byte limits almost orthogonal
to verification work. A verifier that accepts only after unbounded computation
is not fail-closed in any operationally meaningful sense.

### R7 -- Gate end-to-end artifact economics and replace the node-count size proxy

**Amends:** sections 2.6, 3.3, 4.4, 6, and 7.

Add a hard producer-emission wall/CPU threshold and an end-to-end
search-plus-emission comparator. For the artifact-only prototype, predeclare
the utility being purchased: archival size, offline replay amortization count,
or a specific owner workflow. A no-consumer side artifact that makes every
eligible solve 70% slower is not economically successful because a later
standalone replay is 30% faster than rerunning search.

Replace bytes per "reported exhausted search node" with denominators that
actually generate bytes and replay work: `A`, ordered `Q`, stored `K`, edge
occurrences, `U`, `L`, unique propositions, and exact-state key bytes. Report
per-root median/p95/max and aggregate weighted totals. Transpositions can make
edge/disposition count diverge from expanded/unique node count, so the
10-16-bytes-per-node `mvp2lvc` estimate is not evidence.

Pin the exact witness command, flags, binary, termination reason, arena nodes,
expansions, edges, `Q/K/U/L`, wall, and raw output before using either witness
as a size model. The target tree contains no provenance for 17,957 and the
available committed row at `raws/lanec_labels.jsonl` line 261 reports
`forcing_mvp2lvc` at 20,000 win-pass nodes, Unknown, and TT-saturation suspect;
the contemporaneous investigation also says it remained Unknown at one
million nodes. A newer measurement may exist, but it is not reviewable until
pinned.

Clarify that every required-result column in section 6 is a logical AND. The
later statement that economics kills full-tree v1 only if artifacts fail
**both** stored-search size advantage and replay speed contradicts "must pass
every gate." Compare against a compact, competently implemented stored-search
baseline and the exact-enumeration leaf fallback, not an intentionally verbose
`StoredSearchV0`. Add producer p95/max, verifier p95/max, peak-memory tails,
and a held-out measurement cohort.

**Why load-bearing:** the current campaign can pass every written hard stop
while artifact emission makes the only enabled workflow substantially slower.

### R8 -- Close the coordinate/D6 domain and bind semantic versions in the root identity

**Amends:** sections 2.1, 2.5, 4.1, 4.3, and 6.

The engine calls the board unlimited but represents `q,r` as `i16`
(`coord.rs` lines 9-15). Existing D6 transforms correctly return `None` when a
transformed coordinate leaves `i16` (`tss_verify.rs` lines 1836-1863). The v1
format nevertheless accepts arbitrary `i16` root coordinates and requires all
twelve images to verify.

Choose one rule:

1. restrict accepted roots to a documented coordinate domain closed under all
   twelve transforms and reject the rest before proof work;
2. use a wider mathematical wire/verifier coordinate type and prove checked
   conversion to the engine; or
3. scope the D6 gate only to representable images and stop claiming universal
   twelve-image behavior.

Also bind a semantic ruleset/version identifier, not merely codec version and
class text, into canonical root bytes and the typed result. A legality-radius,
opening, window, coordinate, or phase-rule change must reject old policy or use
an explicitly proved compatibility path. Add extreme-coordinate roots and
each failed transform as mutation/robustness cases.

**Why load-bearing:** the current parser admits roots for which a mandatory
gate cannot even construct all twelve transformed roots. Silent ruleset drift
also makes `VcfPairComplete/V1` an unstable proposition across verifier builds.

## Complete section-by-section audit

"Contained" means the attack is fail-closed relative to the still-missing
literal semantics. "PARTIAL" means the section needs an R-item. No section is
credited for a claim that depends on an undefined positive constructor family.

| Design section | Hostile disposition |
|---|---|
| Status/evidence discipline | **PARTIAL -> R1/R2.** The CODE-FACT/HYPOTHESIS labels are honest, but section 7 converts hypotheses into GO before the mathematical class, wire grammar, or evidence pins exist. |
| 1.1 Exact claim | **PARTIAL -> R1/R2/R8.** Correctly class-relative and explicitly not Loss. `NoContractWin`, `VcfPairComplete`, reachability, ruleset identity, and the clock quantifiers are not literal definitions in the target. |
| 1.2 Clock and natural exhaustion | **PARTIAL -> R3.** Correct that `dn==0` is insufficient and resource leaves are forbidden. The specified selector does not recursively establish a resource-free support and can miss an available structural sibling. |
| 1.3 Lean statement family | **BLOCKED -> R1/R4.** Correct theorem direction in spirit, but model acceptance, executed Rust acceptance, all-horizon lifting, producer completeness, and optional-leaf theorems are conflated. No target-tree Lean object supports the named family. |
| 2.1 Logical tree and wire form | **PARTIAL -> R1/R2/R4/R8.** Backward IDs, exact state reuse, orphans, and typed class header are good. The wire grammar omits later shorthand variants and leaves the `NoPositiveConstructor` reason/polarity table open. Ruleset identity is absent. |
| 2.2 Exact attacker coverage | **PARTIAL -> R2.** The intended `T/S`, turn-start stale-block behavior, quotient guard, gate precedence, and `tau>2` rejection are sensible. "Independent set equality" proves only implementation agreement until the universe is a versioned definition and theorem. |
| 2.3 Defender Universal | **CONTAINED relative to R1.** The negative polarity is correct: all attacker turns, one defender counterexample. Exact `tau=b`, `not own_win_now`, full `K_b`, membership, and successor replay defeat omitted-reply attacks. The T6 source supports equality-boundary kernel pruning. |
| 2.4 Optional compact leaves | **PARTIAL -> R4.** Exact `NoAdmissibleFirstTurn` is a legitimate compression once R2 is fixed. `NoJointCarrier` is optional theorem surface with a missing target source, no grind hits, and no demonstrated advantage. |
| 2.5 D6 and graph bounds | **PARTIAL -> R6/R8.** Raw-state identity, no D6 proof quotient, backward DAGs, and checked allocation are correct. Stored bounds do not bound regenerated `Q`, and all-twelve D6 is not closed over arbitrary `i16` roots. |
| 2.6 Size model | **PARTIAL -> R6/R7.** `O(Q+...)` replay versus `O(K+...)` bytes is honestly stated, but the planning estimate then substitutes exhausted nodes for `Q/K/U/L` without evidence. The large-witness count is unpinned. |
| 3.1 Emission seam | **CONTAINED.** A sibling side artifact, private type, self-verification, and unchanged `Unknown` result are the right isolation. The read-once off path and no positive-certificate reuse are strong. |
| 3.2 Eligibility/materialization | **PARTIAL -> R3.** Exact replay, all attacker children, one defender counterline, terminal reclassification, exact memo equality, and final verification are sound goals. Shallow genuine-refutation selection and no alternative search make the actual algorithm incomplete. |
| 3.3 Cost/isolation | **PARTIAL -> R6/R7.** No new PN search and zero non-exhaust hot-path sidecar are good scope controls. The 32-byte memo target is unsubstantiated, regenerated work is unbounded, and emission wall is report-only. |
| 4.1 Module/API boundary | **PARTIAL -> R5/R8.** A separate typed result and no `ProofStatus`/`HardValue` conversion strongly contain class leakage. The trust base, semantic version, transitive call graph, and direct-state transition cross-check need enforcement. |
| 4.2 Re-derivation | **PARTIAL -> R1/R2/R4/R5/R6.** The obligation list is unusually strong and catches hidden terminals, false compact leaves, wrong `K_b`, and state aliasing. It still depends on an undefined positive constructor set, permits an unbounded complement scan, and does not itself prove the optional carrier implication. |
| 4.3 Fail-closed rules | **PARTIAL -> R4/R5/R6/R8.** The malformed-graph and semantic mismatch list is good. "Never panic" is not enough when valid input can induce quadratic work; unknown constructor reasons and semantic-version drift need explicit rejection. |
| 4.4 Replay-cost bar | **PARTIAL -> R6/R7.** Fresh-state, quiet, batched, median/p95 comparisons and no search calls are useful. Relative replay-to-search speed omits producer cost, absolute latency, hostile inputs, and the artifact's actual use/amortization. |
| 5 Consumption roadmap | **CONTAINED.** Every proposed consumer remains post-v1 and categorical. The text correctly forbids backup as `-1`, full-game NO/Loss merging, and search trust. Keep it informational. |
| 6 Gates/kill criteria | **PARTIAL -> R2/R3/R5/R6/R7/R8.** Acceptance, mutation, scope, class-boundary, off-identity, D6, and firewall stops are directionally strong. Eligibility relies on the broken selector; work is unbounded; size evidence is unpinned; producer wall is ungated; and the all-gates rule contradicts the later "fail both" economics sentence. |
| 7 Manageability verdict | **NO-GO pending R1-R8.** The forbidden scope-creep list is correct and the exact-enumeration leaf fallback is sensible. LOC estimates do not establish semantic or economic manageability, and `NoJointCarrier` should not be in the fallback. |

## New counterexample constructions

These are explicit constructions or implementation schedules. A construction
marked `DESIGN-REJECTS` is still a mandatory test. A construction marked
`DESIGN-VULNERABLE` passes or defeats the rules as currently written.

### NCE-01 -- descendant cutoff wins the shallow lexicographic selector

**Arena/state shape.** Let a claimant Choice have one gate-passing pair whose
child is defender Universal `U` with exact `K_2 = {d0,d1}`, in raw order
`d0 < d1`. Generate both replies. The `d1` edge is directly structural: it is
an opponent-terminal move or reaches a state whose admitted attacker set is
exactly empty. The `d0` edge reaches a Branch `B`; below `B`, PN arithmetic has
propagated `(infinity,0)` from a staged `DepthCutoff`. Thus `B.dn == 0`, but
`B.node` is not itself `DepthCutoff`.

The current helper returns true for both `d0` and `d1`. Section 3.2 selects
lexicographically least `d0`. Recursive materialization eventually encounters
the cutoff and returns `None`. The algorithm does not retry `d1`, even though
`d1` is a complete structural counterexample and therefore witnesses a natural
negative support.

**DESIGN-VULNERABLE -> R3.** Final verifier replay prevents a false artifact,
but eligibility/emission completeness and the named witness gate fail. Reverse
the order to make the artifact appear, proving selection-order dependence.

### NCE-02 -- one-node certificate with a quadratic omitted complement

**Root family.** Build `n` separated, reachable motifs. Each motif contains a
claimant-pure count-two length-six window and separators/bridges arranged to
avoid any claimant count-four and any terminal line. Each motif contributes at
least four legal cells to `T(P)`. Extra bridge-induced candidates only
strengthen the attack.

For every `a`, `S(P,a)` contains `T(P)-{a}`. Cross-motif pairs raise at most
one stone in each count-two motif and create no count-four threat, so they fail
`NoNewClaimantThreat`. Same-motif pairs create at most one count-four family
and fail `LooseReply`. Arrange no defender win-now. The exact passing set is
empty, so the wire can be a single `NoAdmissibleFirstTurn` node, while

```text
|T| >= 4n
Q >= |T|(|T|-1) = Omega(n^2).
```

With hundreds of thousands of candidates, the current one-million-root-stone
and 8 MiB wire limits still permit tens or hundreds of billions of regenerated
pair checks. Node count, stored dispositions, byte size, and DAG depth remain
tiny.

**DESIGN-VULNERABLE -> R6.** There is no `Q` or semantic-work preflight. The
artifact can tie up the verifier or exhaust transient memory without violating
any parser count.

### NCE-03 -- the count-one `G1` pair that a post-first universe loses

**Position shape.** At claimant `FirstStone`, choose legal empty `a` lying in
two distinct claimant-pure count-three windows. Arrange their post-`a` empty
sets to have no common hitting cell. Also place `a` in a claimant-pure
count-one window `W`, and choose another empty `b` of `W` that is in no
turn-start claimant count-at-least-two or defender count-at-least-four window.
Thus `a in T(P)`, `b notin T(P)`, but `b in G1(P,a)`.

Neither singleton wins. After `(a,b)`, the two count-three windows through
`a` are distinct claimant threats with `tau=2`; arrange no defender win-now.
The ordered pair is gate-passing even though `b` exists only through the weak
same-turn promotion. A verifier that regenerates candidates after applying
`a`, or defines `G1` as only the stronger window tier, omits it.

**DESIGN-REJECTS in prose; R2/R5 mandatory.** Section 2.2 mentions count-one
promotions and turn-start legality, so a literal implementation should catch
this. The fixture is required because `G1` has no normative equation and a
correlated producer/verifier omission would pass present differential gates.

### NCE-04 -- identical ownership board, different `SecondStone.first`

**State shape.** Use two converging legal histories that place the same colour
stones in different turns and reach the same sorted `(q,r,owner)` board,
current player, placement count, and `SecondStone` phase, but with different
last first placements `x` and `y`. A board-only or phase-tag-only memo key
aliases the states. The next placement has a different same-turn pair payload,
which can change turn-created-threat classification, pair witnesses, and the
semantic proposition attached to the shared node.

**DESIGN-REJECTS.** Sections 2.1 and 4.2 correctly require the full
`SecondStone.first` payload at every reuse. Current `PositionKey::from_state`
also records it (`tss_solver.rs` lines 10433-10450). This remains a mandatory
forced-collision and mutation test for the new producer/verifier rather than a
reason to weaken the key.

### NCE-05 -- every written economics gate passes while emit is net-negative

**Campaign.** On a frozen natural-exhaust cohort, let cold rerun search take
100 seconds aggregate. Let independent verification take 70 seconds, with
every per-witness median and p95 below rerun and no search call. Let artifacts
meet all density and stored-search comparisons, memory stay below 16 MiB, and
all correctness/firewall gates pass. Let post-search producer regeneration,
support planning, encoding, and self-verification take another 70 seconds.

All section 4.4 gates pass because 70% is below 75%. Section 3.3 merely reports
emission wall. Enabled eligible solves now take 170 seconds instead of 100,
and v1 has no consumer. Nothing in section 6 rejects the 70% regression.

**DESIGN-VULNERABLE -> R7.** The artifact can be smaller than an inflated
`StoredSearchV0` and cheaper to replay than search while the only implemented
workflow is slower and economically useless.

### NCE-06 -- accepted raw root has no twelve-image D6 orbit

**Root.** Under the document's declared unlimited-board game, extend a legal
chain from the opening to an `i16` boundary and form a nonterminal
`FirstStone` root containing, for example, a coordinate component `-32768`.
Keep the total root well below one million stones. A D6 reflection/rotation
requires `32768` or another out-of-range axial component. Existing checked D6
code returns `None`. If the executable engine cannot construct that otherwise
mathematically reachable root because its own `i16` arithmetic fails first,
that is the same missing-domain defect: the design must state and enforce the
smaller executable root domain rather than call the board unlimited.

The root satisfies the stated wire coordinate type and parser count. Section
2.5 nevertheless says every accepted artifact transformed under all twelve
actions verifies, and section 6 makes failure a hard stop.

**DESIGN-VULNERABLE -> R8.** The semantic negative at the original root may be
fine, but the mandatory D6 contract is impossible for this admitted root.

### NCE-07 -- equality-only grammar and full-T6 grammar disagree at `tau < b`

**State.** Reach a nonclaimant `FirstStone` state with no own win-now, a
nonempty claimant threat family, `tau=1`, and budget `b=2`; or reach the
analogous `tau=0`, `b=1` residual after a selected first cover.

Under the current wide equality dispatcher, the state has no positive
Universal constructor and `NoPositiveConstructor` ends the negative path.
Under the literal T6 kernel statement, `mhs < b` makes `K_b` the full legal set
with no pruning, so a positive Universal constructor remains available if the
contract language includes full T6 rather than only tight forcing boundaries.

**DESIGN-VULNERABLE AS A SPECIFICATION FORK -> R1/R4.** The design gestures at
both readings: section 2.1 says non-tight means no constructor; sections 2.3
and 4.2 appeal to baseline T6 and say v1 "does not accept" `mhs < b`. A literal
phase-indexed grammar must choose. Implementations are not allowed to settle a
soundness proposition by convention.

### NCE-08 -- current terminal conflation reaches negative support

**Arena/state shape.** Force or directly unit-construct an arena entry whose
replayed state is terminal for the claimant and arrange an ancestor negative
number to consume it. Current expansion assigns `WidePnNode::Refuted` for both
terminal winners (`tss_solver.rs` lines 6393-6399), so the numeric support alone
does not encode the winner.

**DESIGN-REJECTS; R3 test.** Section 3.2 requires terminal replay and accepts
only `OpponentTerminal`; section 4 independently checks the winner. A claimant
terminal must therefore make negative materialization fail. The construction
proves why root `dn==0`, a generic `Refuted` tag, or the current terminal arm
can never be treated as provenance.

## Attack log

This log is complete for attacks mounted in this review. NCE-01 through NCE-08
are incorporated by reference.

| ID | Attack | Outcome and defeating text / finding |
|---|---|---|
| A01 | Review a moving or wrong target. | **FAILED.** HEAD equals the pinned full commit, the target blob matches the working file, and the SHA-256 above is recorded. |
| A02 | Convert class refutation to full-game Loss/HardValue. | **FAILED normatively.** Sections 1.1, 3.1, 4.1, 5, and 6 explicitly keep `Unknown`, use a separate type, and make any game-value exposure KILL. |
| A03 | Infer claimant from stone parity or recursion depth. | **FAILED.** Claimant equals the root mover, player identity controls polarity, and replay checks current player and phase per placement. |
| A04 | Treat `u32::MAX` as an infinity proof. | **FAILED in prose; PARTIAL -> R1.** Section 1.2 calls it only a producer marker and bans horizon leaves. The all-finite-horizons lifting theorem is still undefined/unproved. |
| A05 | Treat root `dn==0` as natural exhaustion. | **FAILED as an explicit claim.** Section 1.2 says necessary, not sufficient. The actual support selector is still vulnerable (A06). |
| A06 | Hide a descendant cutoff below a non-cutoff zero and win Universal tie order. | **SUCCEEDS; NCE-01 -> R3.** The cited helper is shallow and the algorithm has no alternative structural-child search. |
| A07 | Use last-cap exhaustion as natural. | **FAILED.** The strict `expansions < node_cap` trigger rejects equality even if the fact might later be replayable. |
| A08 | Smuggle horizon, census, zone, Group-2, or depth evidence as a leaf. | **FAILED normatively.** All such tags reject, and recursive replay must classify every terminal `Refuted`. R3 is needed to make producer provenance complete. |
| A09 | Use current both-winners-to-`Refuted` terminal code as an opponent fact. | **SUCCEEDS against current numeric source; DESIGN-REJECTS via NCE-08.** Winner replay and standalone verification must reject claimant terminal. |
| A10 | Recompute `T` after the first stone and drop a stale turn-start defender-block candidate. | **FAILED in prose; R2 test required.** `S(P,a)` is turn-start-based and the design says no producer-generation omission is trusted. |
| A11 | Omit the count-one `G1` promotion. | **FAILED in prose; NCE-03.** The design mentions it, but no literal equation/model theorem prevents correlated omission. |
| A12 | Deduplicate ordered pairs before checking a hidden terminal prefix. | **FAILED.** Quotienting requires both prefixes nonwinning and identical full successor states. Negative fresh-turn nodes also reject any claimant local positive constructor. |
| A13 | Use one unordered key for an asymmetric sole occurrence without orientation. | **PARTIAL -> R2.** The verifier can derive the unique valid orientation, but the wire mapping does not say that normatively. |
| A14 | Trust a producer gate-failure reason or omit one passing pair. | **FAILED relative to correct R2 semantics.** Reasons are rederived and the full passing set must equal stored keys. A wrong shared universe remains R1/R2/R5. |
| A15 | Label `tau>2` as `LooseReply`/negative. | **FAILED.** It is explicitly a positive tactical constructor and causes verifier rejection. |
| A16 | Reverse defender Universal polarity and require/refute all replies. | **FAILED.** Section 2.3 correctly takes one counterexample to refute a positive Universal and still reconstructs full `K_b`. |
| A17 | Select a reply outside `K_b`. | **FAILED.** Exact `tau=b`, full kernel regeneration, membership, legality, and successor replay are mandatory. |
| A18 | Let defender win first while treating its reply as a normal cover. | **FAILED.** Pair gating excludes an unhit defender win-now; Universal premises require `not own_win_now`; opponent terminal is typed. |
| A19 | Demand all legal defender cells outside the equality kernel. | **FAILED relative to T6.** The reviewed T6 proof licenses off-kernel dismissal at equality and the selected negative branch is inside the complete kernel. |
| A20 | Apply the T6 equality pruning at `mhs<b`. | **FAILED for counter nodes, but specification fork SUCCEEDS -> R1/R4.** Section 4.2 rejects loose counter nodes while generic `NoPositiveConstructor` leaves the positive grammar ambiguous (NCE-07). |
| A21 | Reuse atomic `DefenderPair` commutation as negative evidence. | **FAILED.** The wire unfolds two ordinary Universal replies and independently checks each intermediate state/kernel. |
| A22 | Make `NoJointCarrier` conditions hold while a two-threat pair exists. | **FAILED combinatorially for the stated class.** The c3/c3, c3/c2, and c2/c2 cases are covered. Formal licensing/source and economic value are still absent -> R4. |
| A23 | Use `NoAdmissibleFirstTurn` when one pair passes. | **FAILED relative to R2.** It is accepted only after complete exact enumeration yields an empty passing set. |
| A24 | Let `NoPositiveConstructor` swallow a future positive leaf. | **SUCCEEDS under class evolution/current underspecification -> R1/R4.** The reason enum and constructor matrix are open. |
| A25 | Reuse a node at a hash collision or D6 image. | **FAILED.** Full raw semantic equality is mandatory; hash and canonical D6 image never authorize value reuse. |
| A26 | Alias `SecondStone` states that differ only in `first`. | **FAILED normatively and in the current exact key; NCE-04.** New code still needs forced-collision tests. |
| A27 | Repair only the checksum after semantic mutation. | **FAILED.** Exact decoded fields, regeneration, root binding, and replay are authoritative; checksum is corruption detection only. |
| A28 | Share `T`, threat, transversal, or kernel helpers through a neutral module. | **FAILED normatively; PARTIAL -> R5.** The prose bans this, but no transitive allowlist/call-graph enforcement is specified. |
| A29 | Read both "independent" implementations from the same incremental WindowStore. | **DESIGN-REJECTS in intent; enforcement gap -> R5.** Direct geometry must be literal and CI-enforced. |
| A30 | Hide enormous work behind complement encoding. | **SUCCEEDS; NCE-02 -> R6.** Stored count/byte/depth limits do not bound regenerated `Q`. |
| A31 | Exhaust memory with the one-million-stone parser allowance. | **SUCCEEDS as an operational risk -> R6.** Allocation preflight cannot make the subsequent direct-window maps, `T/S`, or quadratic replay cheap. |
| A32 | Replay every logical pair "only once" but still run for impractical time. | **SUCCEEDS -> R6.** An at-most-once multiplicity statement is not an absolute work cap. |
| A33 | Make dispositions/edges much larger than unique expanded nodes via transpositions. | **SUCCEEDS against the planning model -> R7.** The codec stores occurrences while the estimate scales from an undefined reported-node count. |
| A34 | Pass replay gates while producer emission regresses enabled solves. | **SUCCEEDS; NCE-05 -> R7.** Emission wall is report-only. |
| A35 | Beat an intentionally verbose stored-search serialization while losing to a competent compact baseline. | **SUCCEEDS -> R7.** `StoredSearchV0` is predeclared but has no competitive-size bar or leaf-fallback comparison. |
| A36 | Use an unpinned `mvp2lvc` node count as implementation evidence. | **SUCCEEDS -> R7.** No target-tree command/artifact supports 17,957; committed evidence reports cap-bound Unknown under the available row. |
| A37 | Pass every gate but fail one of size or replay because section 6 later says "fail both." | **SUCCEEDS as a decision ambiguity -> R7.** "Must pass every gate" and the conjunctive kill sentence conflict. |
| A38 | Cross-accept original bytes against another representable D6 image. | **FAILED.** Exact raw root fields and digest must differ; transformed bytes must be rebuilt and recanonicalized. |
| A39 | Require all twelve images for an extreme `i16` root. | **SUCCEEDS; NCE-06 -> R8.** The accepted coordinate domain is not D6-closed. |
| A40 | Change legality/window/phase semantics without changing class identity. | **PARTIAL -> R8.** Format version exists, but no normative ruleset binding/version-bump rule is part of root identity. |
| A41 | Forge a disconnected/unreachable but phase-consistent external root. | **PARTIAL -> R1.** The mathematical claim assumes reachability; the API/header checks do not say how that premise is established. |
| A42 | Panic on malformed IDs/counts/cycles/trailing bytes. | **FAILED normatively.** Checked preflight, backward IDs, orphan/cycle rejection, canonical varints, and no trailing bytes are explicit. Work exhaustion remains A30-A32. |
| A43 | Let verifier rejection change the ordinary result or search. | **FAILED.** Rejection drops the side artifact and leaves `Unknown`; emit runs after ordinary search terminates. |
| A44 | Let flag-off alter positive bytes, TT, stats, or outputs. | **FAILED by a strong gate.** Bit identity covers all named observables; landing still needs same-binary tests. |
| A45 | Use a v1 artifact to prune search, fill a cache, steer trainer backup, or import atlas truth. | **FAILED by scope.** Section 5 makes every consumer post-v1 and section 6 kills any game-value exposure. |
| A46 | Ship a partial negative tree as the fallback. | **FAILED.** The fallback still requires an exact empty first-turn universe and explicitly disclaims the two full-tree witnesses. Remove `NoJointCarrier` per R4. |

## Claim-boundary disposition

The document is unusually clear that the result is **not** Loss. That boundary
survived every direct leakage attack in this review. The result type, artifact
class identifier, root claimant check, unchanged scalar status, and post-v1
consumer roadmap all point in the correct direction.

The remaining claim risk is earlier: the class being negated is not literal.
`NoContractWin VcfPairComplete P nextPly` is a name plus English, not a
reviewable inductive family in this target. In particular, equality-only
defender dispatch, the treatment of a loose residual boundary, the local
positive constructor priority, and the all-horizon clock lifting need one
normative source. Until R1/R2 land, "verifier accepts" can only mean that one
Rust implementation accepts its own interpretation of a proposed class.

## Coverage and verifier-firewall disposition

The intended coverage algorithm is strong: derive windows directly, rebuild
turn-start `T/S`, replay both placements, compare the entire passing set,
rebuild exact `K_b`, and reject any local positive constructor. If implemented
against a frozen mathematical definition, it is the right shape.

The firewall is not yet structural. Named forbidden helpers do not prevent a
neutral semantic module, a shared incremental window store, a generated
decision table, or two copies produced from the same ambiguous template. The
landing gate must audit transitive calls and inject one-sided faults. Parser
and checksum sharing are not themselves semantic defects, but the literal
decoder-to-model bridge must be separate before an executed-byte theorem is
claimed.

## Economics and evidence disposition

The design correctly recognizes both storage and replay, but its bars measure
the wrong lifecycle. Producer materialization repeats the expensive semantic
work and is ungated; v1 has no consumer; relative replay speed does not pay the
creation cost. Complement encoding also makes a certificate's bytes a poor
proxy for verification work.

The two witness bars are hypotheses, not evidence. `l9mxn59` has a committed
226-node row. `mvp2lvc` does not have a pinned 17,957-node natural-exhaust row
in this target; available repository evidence says Unknown at the named older
caps. This does not prove the newer number false, but it makes the size and
manageability claim unauditable. Freeze exact commands and raw outputs before
using it to authorize a 2.5k-LOC build.

## What I would attack next

I would first build a tiny executable specification of the phase-indexed
positive grammar and exhaustively enumerate small direct-board states. The
highest-value check is not producer-versus-verifier agreement; it is that the
literal `T/S` universe, quotient, equality-only Universal rule, compact leaf,
and negative induction are exact complements of the positive constructors.

In parallel I would model-check the support selector over synthetic PN arenas
containing every mixture of direct Refuted edges, lazy thunks, nested cutoffs,
horizon/census refusals, terminal winners, transpositions, and alternative
Universal replies. The invariant is: `Structural(plan)` iff replay of `plan`
contains no unresolved cause, and selection order cannot change whether an
artifact exists.

Finally I would generate adversarial large sparse roots to measure `Q`, direct
window counts, memory, emit wall, and verifier wall before choosing parser
limits. The next likely failure is a tiny artifact whose omitted complement is
too expensive to replay, followed by correlated producer/verifier drift at a
weak-promotion or loose-boundary edge—not a reversal of the already-correct
Universal polarity.
