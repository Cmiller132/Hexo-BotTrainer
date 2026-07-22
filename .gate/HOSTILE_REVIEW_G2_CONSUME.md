# HOSTILE REVIEW: G2-CONSUME

**Review date:** 2026-07-21  
**Target:** `docs/DESIGN_G2_CONSUME.md` at `fcea3c6919e08d91de663332f7f67cb9eb793090`  
**Target SHA-256:** `69C79A42DF25B53CCBAB1C8452EAB5DF9CD37B77C1EB46DE50DFA2099AA172D6`  
**Review mode:** hostile, doc-only

## VERDICT: SOUND-WITH-REQUIRED-CHANGES

I did not find a path by which the specified Consume state machine can mint a
false hard Win or Loss past the independently implemented, standalone strict
verifier. The strongest search-composition attacks--Consume-to-Full TT
leakage, claimant reversal in the LOSS leg, cutoff/reopen aliasing, and a
mixed gate/compact shared parent--are explicitly blocked by Open debt,
occurrence-local regimes, exact whole-tree context stabilization, unfolded
materialization, and sealed re-verification.

That does **not** make the design acceptable as written. Its owner decision
protocol can (1) promote a solver that is drastically worse than deployed
`Off`, (2) tune and evaluate on the same development roots, (3) pass a
supposedly decisive step-zero screen after the screen has already proved the
10% target impossible, and (4) kill a real prize on a correlated false
negative from a pre-independent shadow classifier. The plan-complete screen
has no pass rule at all. These are not cosmetic reporting defects: they
change BUILD/KILL/PROMOTE decisions.

The semantic verdict is therefore conditional. No source implementation is
authorized by this review, and promotion remains blocked until every R-item
below is incorporated and hostile-reviewed. Current source is also
nonconforming in the exact ways the design admits: native wide search has no
Full unforced arm or Open/Closed state, its mutable keys are insufficient,
and producer `finder_*` paths still call verifier-module semantic helpers.

## Required changes

### R1 -- Add the deployed `Off` and production-profile adoption boundary

**Amends:** sections 3.1, 6.2, 6.3, 6.5, and 9; GAPs C01, C09, and C20.

The causal A/B is `Verify/FullControl` versus `Consume`; `Off` is explicitly
only a parity lane. That permits this passing campaign:

| arm | verified decisions | all-root nodes | end-to-end wall |
|---|---:|---:|---:|
| deployed Off | 100 | 100,000 | 100 s |
| Verify/FullControl | 100 | 1,000,000 | 1,000 s |
| Consume | 100 | 800,000 | 900 s |

Consume has identical coverage, a 20% node reduction and 10% wall reduction
against Verify, and can make every confidence, memory, occurrence, residue,
and verifier gate pass. It is nevertheless 8x the nodes and 9x the wall of
the deployed solver. The fixed `D0` denominator does not repair a missing
deployment comparator.

Before default promotion, require one of these two mutually exclusive facts:

1. FullControl has already passed its own adoption/Pareto bar and is the live
   baseline for the exact profile; or
2. Consume also passes an outer, all-work Consume-versus-Off deployment gate.

The outer gate must include a **Production-500** cell using the actual
persistent batched trainer path, or the document must explicitly limit the
feature to an offline 2k/50k profile and prohibit cap-500/default enablement.
The repository's current production configuration is cap 500, and its comment
records that 500/2,000/8,000 decided the same release-A/B positions while the
extra 1,500 nodes burned CPU (`configs/hexfield_eq_main_3.toml`, lines 46-52).
The owner harness separately requires persistent, production-shaped batching
(`docs/PLAN_TSS_HARNESS.md`, sections 1-2); fresh per-root caches remain useful
as a causal diagnostic, not as the sole deployment gate.

This amendment must also state that historical structural
"never-decides-less" is retired for Consume at finite cap. Section 6.4's
pointwise set inclusion is a strong **frozen-campaign** replacement, not a
universal production theorem. If zero loss on unseen production is still an
owner invariant, require canary/rollback, dual-run, or another explicit
online guard; otherwise prohibit the universal claim.

**Why load-bearing:** without this change, the prescribed bar can promote a
known net-negative replacement and can apply an offline result to a production
profile it never tested.

### R2 -- Separate discovery from adoption and use the correct sampling unit

**Amends:** sections 6.1, 6.2, 6.3, 7.2, and 9; GAPs C02, C19, and C23.

Both shadows run on the exact Labeling-2k, Atlas-50k, and Human160 roots before
the independent producer and Consume scheduler are authorized. The completed
implementation is then promoted on those same roots. The 6,462 roots are
described as three **development** splits plus F19, while the owner harness
requires a held-out split consumed only at adoption and labels visible-only
gains `OVERFIT` (`docs/PLAN_TSS_HARNESS.md`, lines 61-69). Freezing a manifest
before shadow results does not keep the final test blind after those results
have guided scheduling, fallback, limits, or data structures.

Require:

- visible discovery/shadow manifests separate from an untouched adoption
  holdout, or escrowed holdout results that remain unavailable until the
  implementation, scheduler, limits, and analysis commit are frozen;
- grouping and resampling by independent source game/sequence or atlas family,
  not by individual root pairs;
- predeclared production weights and inclusion probabilities for any
  stratified Atlas sample, with weighted estimands and confidence intervals;
- a complete Atlas as `MUST` when the production claim is about the complete
  atlas; otherwise an owner-attested sampling frame and scope-limited claim;
  and
- retention of every failed/abandoned campaign.

The clustering issue is material, not theoretical. Read-only grouping of the
pinned IDs gives 3,255 selfplay roots from only 48 game-prefix groups (as many
as 87 roots in one group) and 2,720 human roots from 340 groups. Section 6.3's
root-pair bootstrap pseudo-replicates correlated positions and can produce a
spuriously tight 5% lower bound. Root totals may remain the production
estimand, but uncertainty must be clustered at the process that generated the
roots and any sample must carry its production weights.

**Why load-bearing:** the current protocol permits deliberate or accidental
overfit and invalid confidence, so a nominal promotion pass is not independent
evidence of deployable gain.

### R3 -- Align the step-zero node ceiling with the actual promotion target

**Amends:** section 7.2 and GAP C02.

Section 6.5 requires at least 10% node reduction at each primary operating
point. Section 7.2 stops only when exact `U_nodes < 5%`. If
`U_nodes = 7%` at Labeling-2k or Atlas-50k, the "eligible roots become free"
argument proves that no implementation can reach 10%, yet the screen passes
and section 9 authorizes the next build stage.

Use `U_nodes < 10%` as the decisive stop at Labeling-2k and Atlas-50k. Retain
the 5% threshold for Human160, where 5% is the stated independent node floor.
Spell out equality and confidence-bound handling against every downstream
target.

**Why load-bearing:** the current decisive screen gives the wrong BUILD
decision in a range where its own bound has already proved promotion
impossible.

### R4 -- Make a shadow negative a certified conservative classification

**Amends:** section 7.2 and GAPs C02 and C06.

The root-support equation is safe only if `E` is a no-false-negative superset
of every future Consume divergence root. The shadow precedes the independent
producer and may use shared semantic helpers. Runtime incompleteness,
overflow, and work failure become `indeterminate`, which is good; a
confidently wrong "ineligible" result from a narrow or correlated classifier
does not. Such a root is omitted from `E`, and because the build is killed,
the later independent producer that would expose the mistake is never built.

Define a kill-grade negative contract tied to the exact future hook and
classifier version. Require an audited conservative over-approximation,
exhaustive/golden boundary tests, one-sided defect injection, and an
independently reviewed implementation or checker for negative classification.
Any classifier uncertainty is indeterminate. If a later producer ever finds
an eligible occurrence in an `E`-complement root, invalidate the earlier
screen and all decisions derived from it, then rerun.

**Why load-bearing:** without a certified no-false-negative property, the
claimed mathematical upper bound is merely telemetry and can incorrectly kill
the only profitable class.

### R5 -- Give FullControlShadow-PC a preregistered pass/fail rule

**Amends:** sections 7.2 and 9; GAPs C02 and C19.

The initial support bound may be very large while every reached site fails
plan closure, context stabilization, admissibility, memory, or remaining-cap
fallback. FullControlShadow-PC records the relevant realization data and
treats unresolved sites maximally optimistically, but the design defines no
equation, threshold, confidence rule, or disposition. Section 9 nevertheless
says only if "those economics screens pass" may the independent producer and
Consume state machine be authorized. Any PC output can presently be called a
pass.

Pre-register a maximally optimistic plan-complete upper bound on realizable
node and wall savings, including dominated-interval union, fallback, closure,
verification, and memory costs. Give it kill thresholds aligned with sections
6.5 and R6. It remains a one-sided kill screen and cannot promote.

**Why load-bearing:** this screen is the stated authorization boundary for
the expensive duplicated theorem implementation, but the boundary currently
has no decision function.

### R6 -- Gate end-to-end labeling economics, not only solver-node accounting

**Amends:** sections 6.3 and 6.5; GAP C20.

At Labeling-2k, a 10% node reduction plus any positive wall point estimate can
pass. Derivation, strict verification, scheduling, and materialization are
correctly excluded from fabricated solver nodes and included in wall/CPU, but
CPU is report-only and Labeling wall has no material floor or lower-confidence
requirement. Three noisy repetitions can therefore pass with a 0.001% wall
gain, higher CPU, worse tails, and reduced offline throughput.

Add a preregistered, confidence-bounded Labeling throughput/CPU and wall floor
appropriate to the owner's batch service, plus p90/p99/max slowdown guards or
an explicit statement that only aggregate batch throughput is operationally
relevant. Define how peak memory is aggregated across roots and persistent
batches. Use the game/family cluster unit required by R2.

**Why load-bearing:** the design can otherwise pass its economic gate while
increasing the resource that production comments identify as the binding
capacity constraint.

## Complete section 8 composition audit

"Contained" below means fail-closed for hard truth. `PARTIAL` identifies a
decision/economics composition defect even where the strict verifier still
prevents false truth.

| GAP | Hostile composition and disposition |
|---|---|
| C01 | **PARTIAL with C20 -> R1.** Semantically safe: no A/B before exact lazy nested FullControl and pair tests (sections 3.2.1, 8/C01). Economically unsafe: beating that new expensive control does not show improvement over deployed Off. |
| C02 | **PARTIAL with C06/C19 -> R3-R5.** Missing deep prevalence blocks promotion, but its prescribed shadow can use a wrong negative, uses the final roots for discovery, has a 5%-versus-10% threshold error, and gives PC no pass rule. |
| C03 | **Contained.** Occurrence-scoped G2/Full overlays, exact context fields, ancestor-minimal permanent Full fencing, exact epoch passes, and unfolded certs prohibit context-unsafe DAG sharing (sections 3.3.1, 3.8). |
| C04 | **Contained/open implementation blocker.** Open is never Proven, cached, or materialized; cutoff/unresolved ends in remaining-budget Full or Unknown (sections 3.3-3.6, 8/C04). |
| C05 | **Contained.** One worker plus canonical orders remains mandatory until schedule-independent byte identity is proved (sections 3.9, 8/C05). |
| C06 | **Contained for truth; PARTIAL with C02 for kill decisions -> R4.** Verify/Consume cannot use current shared semantic helpers, and the transitive source firewall rejects neutral relocation (section 4.3). The preliminary kill classifier is nevertheless allowed shared helpers without a negative-certification bar. |
| C07 | **Contained by scope.** NonFC `FhwGateV1` rejects; only the separately admitted exact compact-T6 seam may run, else Full/Unknown (sections 3.2, 4.2, 8/C07). |
| C08 | **Contained.** The first complete plan is immutable; inability to close opens Full, and reselection to manufacture acceptance is forbidden (sections 3.3, 4.1, 8/C08). |
| C09 | **Contained for truth/coverage campaign.** Fresh claimant legs, one combined ledger, no refunds or outside-cap rescue, dedicated Loss tests, and any lost FullControl decision is KILL (sections 3.7, 6.4). It does not cure R1's missing deployment comparator. |
| C10 | **Contained/open implementation blocker.** Fresh per-leg arenas, complete mode/policy clearing, explicit resume binding, and no persistent G2 fragment close the default (sections 3.7-3.8). |
| C11 | **Contained by an explicit stop.** CE23 absence blocks promotion; it is not presumed solved (sections 5/CE23, 8/C11). |
| C12 | **Contained.** Closure/resource/memory failure opens Full when fully publishable or returns Unknown; truncation never licenses omission (sections 3.5, 3.9, 8/C12). |
| C13 | **Contained only relative to an explicit trusted base.** Authority is hash-pinned and theorem objects are independently reconstructed; exhaustive small-board/D6 tests remain. Engine/formal correspondence stays an external assumption and must not disappear from the landing review (sections 4.2-4.3, 8/C13). |
| C14 | **Contained by nonclaim.** Full scalar `B`, Loss remainder, horizon, escape floors, and paired edge/child clocks are preserved; no generic `k<b` debit is claimed (sections 3.2, 4.2, 8/C14). |
| C15 | **Contained by language restriction.** D17, SR, commutation, legacy zones, relabeling, and arbitrary mixing reject; only the reviewed compact exact seam composes (sections 1.3, 4.2, 8/C15). |
| C16 | **Contained with C03.** No G2 closure sharing, exact no-change stabilization, tree unfolding, and per-occurrence independent rederivation defeat DAG completeness attacks (sections 3.3.1, 3.8). |
| C17 | **Contained.** Digests establish byte identity only; the verifier independently derives each semantic preimage before comparison (sections 3.8, 4.3, 8/C17). |
| C18 | **Contained/open build blocker.** In-memory only, immutable run-wide v3 before search, unknown tags reject, and search state is not serialized (sections 3.8, 8/C18). |
| C19 | **PARTIAL with C02/C23 -> R2, R5.** The text correctly bars gain forecasts, but it reuses discovery roots for final promotion and permits an unweighted Atlas sample; PC has no economics pass rule. |
| C20 | **PARTIAL with C01 -> R1, R6.** Same-cap all-work accounting and coverage KILL are sound against attrition. They still allow a massive Off regression and noise-sized Labeling wall gain. |
| C21 | **Contained/open implementation-evidence blocker.** Closed PN is only a candidate; standalone root+bytes rejection discards it and yields Unknown/KILL, with sealed re-verification (sections 4.1, 8/C21). |
| C22 | **Contained.** Stable old categories, absolute block reconciliation, and unexplained drift as economics failure defeat denominator relabeling (sections 6.6, 8/C22). |
| C23 | **PARTIAL with C19 -> R2.** Missing Atlas blocks both arms and Human160 cannot substitute. The optional stratified sample still needs production weights, a mandatory frame, and a scope-limited claim. |

No pair of defaults produced a false Win/Loss after strict replay. The two
load-bearing default interactions are instead decision-unsound:

- **C01 + C20:** exact FullControl may be much more expensive than Off;
  Consume can beat it while remaining net-negative, yet the design promotes.
- **C02 + C06:** the kill-grade shadow may reuse correlated helpers before
  the independent producer exists; a wrong complete negative shrinks `E` and
  kills the producer that would have discovered the error.

## New counterexample constructions

These are new schedules/constructions, not claims that the current tests
already instantiate every future state. A construction marked
`DESIGN-REJECTS` remains a required implementation test where cited.

### NCE-01 -- Consume plan leaks into a recursive-Full revisit

**Position/state.** Replay this exact post-opening sequence:

```text
(0,0), (0,3), (1,3), (-5,1), (5,5), (2,3), (6,0),
(-3,-6), (9,-4), (6,-1), (6,-2), (-7,6), (11,1),
(0,-5), (1,-5), (3,9), (-8,-2), (2,-5), (-4,0), (2,12)
```

The resulting `N` is P0/SecondStone, `b=1`; relative to claimant P1 it has
`k=0` and is an eligible unforced Group-2 occurrence. The frozen fixture has
a 19-edge reduced cert against 886 legal Full replies. Reach `N` twice inside
one claimant leg: occurrence A remains Consume and closes with `S*`; a
stronger incoming context at occurrence B makes the ancestor-minimal mismatch
fence B recursively Full. Force the two occurrence identities into the same
StateCore/hash bucket and materialize A first. The bug returns A's
`CandidateProven`, plan, or `CertNodeId` to B, omitting 867 Full replies; the
reverse order can relabel or bloat A.

**DESIGN-REJECTS.** Only immutable legal moves/outcomes share through
StateCore; every mutable Full/G2 overlay is occurrence- and regime-local,
with one owner and checked epoch (section 3.8). G2 and Full lookups cannot
answer one another; materialization unfolds each occurrence from its frozen
local plan and strict replay checks the full legal set (sections 3.5, 4.1;
CE14, CE15, CE18, CE21, CE23).

### NCE-02 -- consumed WIN-side state crosses into LOSS/dual

**Position/state.** Use the same exact `N`. Root the Both solve immediately
after opening `(0,0)`, which is post-opening. The two legal P1 placements
`(0,3)` then `(1,3)`, or `(1,3)` then `(0,3)`, are nonterminal and converge
to the same exact post-pair state; continue with the remaining replay to `N`.
The root player and primal claimant are P1. At `N`, P0 is therefore the
defender and `N` is the eligible G2 occurrence above. Force the primal attempt
to end only after warming every cache category with `S*`/Closed state. In the
dual LOSS leg claimant is P0; the identical `N` is claimant-to-move and must
follow a Choice path. Attack in both warm orders: import primal reduced state
into dual, then import dual node kind/negative into primal. Let the combined
ledger carry its legitimate spent count so the only illicit carrier is
semantic state.

**DESIGN-REJECTS.** Each leg is a fresh positive claimant proof with a fresh
closure arena and TT namespace; no plan, negative, cutoff, or summary flows
between legs (section 3.7). Claimant, primal/dual leg, policy, context, unique
occurrence, and regime are equality fields, and every cache category clears
at the boundary (section 3.8). The final claimed player is independently
verified (section 4.1; CE16).

### NCE-03 -- pre-cutoff Full result closes a reopened Consume occurrence

**Position/state.** Prewarm a complete FullFence overlay for exact `N` under
claimant P1. Start a fresh staged Consume solve of `N`. At retained reply `s`,
set the stage depth so a descendant is `DepthCutoff(d)`. The G2 parent is
`Open(g,S_g)` and has no complete plan for `s`. On the next stage, reopen that
exact descendant; its completed plan adds role/window demand whose first
legal missing coordinate is
`x = min(Required(P,plan_g) - S_g)`. The attack uses the old board-key Full
`pn=0`, or current-style `(INF,0)` treatment of a cutoff, to publish the
parent before appending and searching `x`.

**DESIGN-REJECTS.** Cutoff is checked before eligibility at the cut node; it
has no closure state, is explicit unresolved provenance, and prevents plan
freeze, closure, caching, or materialization (sections 3.4, 3.6). Deepening
reopens and charges it, refreshes ancestors deepest-first, recomputes every
dependent summary, and appends `x`. Full and G2 overlays cannot alias
(section 3.8; CE05, CE06, CE15, CE19, CE24). Deleting `x` also fails strict
replay.

### NCE-04 -- mixed gate/compact shared parent after consumption changes children

**Position/state.** Replay the current production mixed root:

```text
(0,0), (-8,0), (-8,-1), (1,0), (-8,1), (-9,0), (-7,-1),
(-10,1), (-9,1), (-7,0), (-6,0), (-10,0), (-5,0),
(-6,-1), (-5,-1), (-4,-1), (-10,-1)
```

The existing mixed battery finds a verified Both/cap-500 certificate with at
least one `FhwGateV1` and one compact `Universal(implicit_dispatch=true)`.
In the future native fixture, place both below common ordinary-G2 parent `u`.
Freeze the gate child first and hold the compact sibling below a cutoff. On
reopen, the compact plan raises `child_f/child_q` or incoming demand and adds
ordinary edge `x` at `u`. Force a StateCore collision between a gate
representative descendant and compact-edge descendant, then materialize in
both orders. The intended bugs are: one `CertNodeId` discharges both contexts,
stale `u` stays Closed, the cheaper sibling's row is spliced, or a selector
rerun changes which consumed children appear.

**DESIGN-REJECTS.** Frozen child plans are immutable; whole-tree
stabilization waits for complete Full/legacy plans, compares exact context
fields, and fences every ancestor-minimal mismatch before global closure
(sections 3.3-3.3.1, 3.6). Occurrences unfold and rederive separately, and
the verifier recomputes mixed roles, clocks, rows, and ordinary required
edges (sections 3.8, 4.1; CE12, CE14, CE18, CE21, CE23). CE23's current
absence is itself a promotion blocker, not permission to assume success.

### NCE-05 -- passing but net-negative production campaign

**Workload/state.** Use 100 roots decided identically by all arms, with the
table in R1. Give Verify and Consume the same strict grammar/policy, no
verifier failures, 30 consumed occurrences over 10 roots in each cell,
memory ratio 1.0, and tight repetitions. Duplicate the 20% nodes/10% wall
delta at both primary cells and Human160. Every written correctness and
economics gate passes against Verify. Consume remains 8x/9x worse than Off.

**DESIGN-VULNERABLE -> R1.** No section 6 formula makes Off an economic
promotion comparator.

### NCE-06 -- step zero passes after proving the target impossible

**Workload/state.** At Labeling-2k, let exact `E` contain roots accounting for
7% of shadow-off FullControl nodes. Even making them free gives
`U_nodes=7%`; all other roots have exact negative hooks. Repeat at Atlas-50k.

**DESIGN-VULNERABLE -> R3.** Section 7.2 stops only below 5%, so both screens
pass, while section 6.5 later requires an impossible 10% reduction.

### NCE-07 -- false-negative shadow kills a real deep prize

**Workload/state.** One deep root `r*` accounts for 20% of FullControl nodes
and reaches exact `N` above. A shared shadow helper reconstructs an incomplete
threat family and confidently labels the hook forced/ineligible rather than
indeterminate. Other eligible/indeterminate roots account for 4%. The report
therefore has `U_nodes=4%` and kills the build. An independent producer would
classify `N` as `b=1,k=0` and could remove most of the 20% subtree, but it is
never authorized.

**DESIGN-VULNERABLE -> R4.** "Exact" is asserted, but no kill-grade
no-false-negative certification rule protects a complete negative produced by
the explicitly shared pre-independent classifier.

### NCE-08 -- development-root tuning and pseudo-replicated confidence

**Workload/state.** Run both shadows on all final Labeling/Atlas roots. Tune
closure priority and fallback limits to the revealed high-cost eligible roots,
freeze the finished binary, and evaluate it on the same roots. In one source
game, include dozens of near-adjacent high-saving positions; other independent
games regress slightly. Root-pair resampling treats the adjacent positions as
independent and tightens the lower bound until it passes. A game/family-level
held-out evaluation does not reproduce the gain.

**DESIGN-VULNERABLE -> R2.** Section 6 freezes post-result edits but never
separates the section 7 discovery roots from adoption, and section 6.3 names
root pairs rather than source games/families as bootstrap clusters.

## Attack log

This log is complete for attacks mounted in this review. New constructions
NCE-01 through NCE-08 above are incorporated by reference rather than
duplicated.

| ID | Attack | Outcome and defeating text / finding |
|---|---|---|
| A01 | Tamper with the frozen target or authority identity. | **FAILED.** Target hash matches the brief. The controlling defender proof and FHW companion hashes match section 4.2's pins. |
| A02 | Reopen the prior R1 escape-deadline hole. | **FAILED.** Sections 3.3 and 4.2 retain every exact `p(Q)+b+2` deadline and scalar horizon; CE17 must reject deletion. The R1/R2 amendment is incorporated. |
| A03 | Reopen the prior R2 post-opening descendant loophole. | **FAILED.** Section 3.2 makes post-opening attempt-global and disables all new-class search/emission for an Opening-root solve; CE01 covers forged descendants. |
| A04 | Extend FHW-T3-R beyond its reviewed class. | **FAILED.** Sections 1.1, 1.3, 3.2, and 4.2 admit only pinned Exact/FC and retain direct incidence, paired clocks, `b`, Loss, scalar, horizon, and escape floors. NonFC/D17/SR/commutation remain rejected. |
| A05 | Use authority or digest equality as theorem truth. | **FAILED.** Sections 3.8 and 4.3 require independent semantic reconstruction before digest comparison; C13/C17 remain explicit assumptions/limits. |
| A06 | Leak Consume mutable state into recursive Full at the same board. | **FAILED; NCE-01.** Sections 3.5 and 3.8 isolate occurrence overlays/regimes and share only immutable FullCore facts. |
| A07 | Reverse the warm order, Full then G2 then Full, under a forced collision. | **FAILED.** Section 3.8 forbids cross-regime hits, checks owner epoch, clears mode state, and mandates A/B/A collision tests (CE15). |
| A08 | Leak primal WIN-side state into dedicated Loss/Both dual. | **FAILED; NCE-02.** Sections 3.7-3.8 use a fresh claimant arena/namespace and claimant+leg keys; CE16. |
| A09 | Treat primal SearchDead as opponent truth. | **FAILED.** Sections 3.4 and 3.7 permit LOSS only from a fresh positive opponent cert; SearchDead never persists or crosses claimant boundaries. |
| A10 | Let a cutoff or `(INF,0)` arithmetic close/refute an ancestor. | **FAILED; NCE-03.** Open debt plus explicit `Unresolved` provenance binds every number consumer (section 3.4), and section 3.6 recomputes on reopen; CE05/19/24. |
| A11 | Change a sibling plan/kind after a context pass without bumping an ancestor append epoch. | **FAILED.** Stabilization runs only with complete Full/legacy plans (section 3.3.1); pre-global candidates cannot materialize, newly completed children recompute summaries, and changing a legitimately frozen plan is KILL (section 3.6; CE06/23). |
| A12 | Share a G2 DAG plan across stronger incoming demands. | **FAILED.** Exact incoming fields plus unique occurrence are equality fields; G2 closure is not shared and certs unfold/rederive (sections 3.3.1, 3.8; CE14). |
| A13 | Splice gate `child_f/child_q` or a cheap compact sibling at a mixed parent. | **FAILED; NCE-04.** Immutable plans, exact whole-tree context, mixed strict replay, CE12/23. |
| A14 | Rerun a selector during materialization and emit a different valid plan. | **FAILED.** Section 4.1 serializes the exact immutable Closed plan and forbids selector rerun; CE18. A separately valid cert does not establish conformance to the search claim. |
| A15 | Delete a later-generation ordinary edge and its orphan while retaining summaries/digests. | **FAILED.** Independent required-set reconstruction rejects it (section 4.1; CE21). |
| A16 | Exhaust closure memory and publish the partial reduced domain. | **FAILED.** Sections 3.5/3.9 atomically publish complete Full or return Unknown; C12/CE07. |
| A17 | Flatten a two-placement turn into a stale Cartesian pair domain. | **FAILED.** Section 3.2.1 uses nested exact FirstStone/SecondStone legal reconstruction and immediate terminal handling; CE25. |
| A18 | Use a NonFC FHW edge because WC rows appear to cover it. | **FAILED.** Section 4.2 rejects NonFC before rows can admit it; exact compact seam or Full/Unknown only (CE08/10). |
| A19 | Share a neutral theorem helper, semantic encoder, generator, or copied decision table. | **FAILED normatively.** Section 4.3 bans semantic sharing and neutral relocation and requires each side audited against the theorem. Landing review must treat common code generation/template provenance as sharing, not rely only on runtime calls. |
| A20 | Use today's shared finder/verifier implementation as promotion evidence. | **SUCCEEDS against current source; DESIGN-REJECTS.** Current `tss_solver.rs` calls verifier-module `finder_required_fhw`, `finder_build_fhw_gate`, `finder_fill_gate_rows`, and `finder_finalize_group2`. Section 4.3 and C06 explicitly block Verify/Consume until two implementations and the source firewall exist. |
| A21 | Let raw `ProofStatus` reach the trainer without sealed verification. | **FAILED for the specified consumed path.** Section 2 invariant 10 and section 4.1 require sealed concrete verification; section 6.3 defines decisions accordingly. Current production `tree.rs` converts deep results through concrete hard mint and verifier rejection to Unknown. Raw solve uses found are diagnostics/tests. |
| A22 | Select verifier policy from certificate bytes. | **FAILED.** Sections 3.1 and 4.1 require external policy before search; mismatch/unsupported version returns Unknown. |
| A23 | Pass initial self-verify, then mutate/substitute the cert before mint. | **FAILED.** Section 4.1 re-verifies at the sealed concrete mint and binds exact root, status, bytes, policy, authority, and version. |
| A24 | Smuggle an extension-free legacy-zone cert through current Group2Verifier delegation. | **Current-source conformance risk; DESIGN-REJECTS.** Section 4.2's Consume grammar admits legacy leaves/Choice/full Universal, not legacy zones; C15 rejects broader mixing. Landing must make this unreachable or apply strict grammar even without an extension node. |
| A25 | Exploit shared minimal game primitives to correlate theorem errors. | **FAILED within declared trust base.** Section 4.3 permits only coordinate/board/legal transition/phase/terminal primitives; theorem analyzers and encoders cannot be shared. C13 explicitly retains engine/formal correspondence as an assumption. |
| A26 | Game the decision denominator by making treatment lose decisions. | **FAILED.** `D0` is fixed from control, every manifest root remains in the numerator, and any lost control decision is KILL (sections 6.3-6.4). |
| A27 | Trade one lost decision for several new ones. | **FAILED.** Section 6.4 uses pointwise `Decided_Off subseteq Decided_FullControl subseteq Decided_Consume`; aggregate compensation is forbidden. |
| A28 | Change thresholds, roots, parser output, or machine after results. | **FAILED as written.** Sections 6.1-6.2 freeze the campaign and canonical state bytes and retain failed runs. A29 still breaks independence by exposing final roots before implementation. |
| A29 | Tune on the final evaluation roots and pseudo-replicate correlated positions. | **SUCCEEDS; NCE-08 -> R2.** Sections 6-7 reuse development roots and bootstrap root pairs; owner held-out law is not met. |
| A30 | Pass against Verify while massively regressing Off. | **SUCCEEDS; NCE-05 -> R1.** Off is parity-only, not an economic comparator. |
| A31 | Preserve the historical universal never-decides-less claim. | **FAILED: it is retired.** Sections 1.2 and 3.7 explicitly admit finite-cap order losses; section 6.4 replaces the structural property only on frozen cells. R1 requires honest scope/production mitigation. |
| A32 | Pass Labeling economics with a noise-sized wall gain and CPU/tail regression. | **SUCCEEDS -> R6.** Section 6.5 gives Labeling wall only a positive point requirement and no CPU/tail gate. |
| A33 | Call a pre-FullControl narrow trace a decisive upper bound. | **FAILED.** Section 7.2 explicitly labels it heuristic and forbids killing the build from it. |
| A34 | Let a large root-support bound promote Consume. | **FAILED.** Section 7.2 says the bound cannot promote and passing only avoids an early kill. The missing PC disposition remains R5. |
| A35 | Let `U_nodes=7%` pass despite a 10% target. | **SUCCEEDS; NCE-06 -> R3.** The thresholds contradict each other. |
| A36 | Miss a valuable eligible root with a confidently wrong shadow negative. | **SUCCEEDS; NCE-07 -> R4.** Only operational incompleteness is forced indeterminate; negative certification is unspecified. |
| A37 | Substitute Human160 or 40.46% residue for an Atlas/deployment estimate. | **FAILED.** Sections 1.1, 6.2, 6.5, 6.6, and 7 distinguish forced cap-500 firing, the 19/886 fixture, Human160, Atlas, direct wall residue, and node savings. C23 blocks absent Atlas. |
| A38 | Quote 36-106/6,462 forced gate roots as unforced deep prevalence. | **FAILED.** Sections 1.1 and 6.5 explicitly prohibit it; the 142 cross-check observations may overlap and current search consumed no gates. |
| A39 | Quote 19/886 local fanout as population economics. | **FAILED.** Section 1.1 calls it a motivating local fixture only; sections 6-7 require deep cost-weighted evidence. |
| A40 | Turn the 40.46% human residue into expected gain or node share. | **FAILED.** Sections 1.1, 6.5-6.6, and 7.2 call it a direct-block wall ceiling, require absolute reconciliation, and preserve Human160 as diagnostic rather than Atlas. |
| A41 | Hide category growth so the residue percentage falls. | **FAILED.** Section 6.6 requires the absolute unforced block to shrink consistently with nodes; section 8/C22 makes unexplained category drift an economics failure. |

## Verifier-firewall and source-seam disposition

The required independence is real **in the design**, not in current source.
Section 4.3 permits only data records/tag constants, authority constants, raw
hash primitives, and minimal game primitives. It bans theorem tables,
threat/transversal analyzers, roles/clocks, window logic, closure, row builders,
summaries, and semantic encoders; moving them to a neutral module still fails.
The call-graph/source allowlist, one-sided `threats_shared` rule, named
`finder_*` bans, independent golden vectors, and separate theorem audits make
this a stronger boundary than the earlier hand audit.

Every consumed hard verdict is specified to pass three barriers: globally
Closed no-change state, standalone unfolded strict replay without search
state, and sealed concrete re-verification under externally selected policy
(sections 4.1-4.3). Verifier rejection gives no hard result and is campaign
KILL. The reviewed current trainer seam already consumes a private
certificate-verified hard value, but that is not evidence that future Consume
conforms; the source firewall and all raw-result consumers must be re-audited
at landing. The preexisting immediate leaf/lambda-one hard path is outside the
Consume hook and should not be confused with a consumed Group-2 verdict.

## Corpus-semantics disposition

The design correctly refuses the obvious fixture-to-deployment fraud:

- cap-500 forced gate emission was only 36-106 of 6,462 roots across two
  nondeterministic materialization runs; 142/142 observations cross-verified
  but may overlap, and nodes were identical because search consumed nothing;
- the 19-versus-886 ordinary unforced fixture is a real 46.6x local fanout
  reduction, not prevalence or end-to-end economics;
- the finder observations are fixture/surface evidence, not a population
  estimate; and
- 40.46% on Human160-50k is the directly timed unforced-generation wall
  block, not node share, expected saving, or Atlas. The residue artifact has
  no selector-based Consume estimate.

The remaining corpus failure is subtler and covered by R1/R2: the document
calls the dev-root cap-2k cell a promotion/production cell even though the live
profile is cap 500 and persistent-batch, then exposes those same roots during
shadow development. Atlas absence is honestly blocking, but optional sampling
without production weights does not establish complete-atlas economics.

## What I would attack next

With more time I would build a small exhaustive state-machine model that
interleaves append generations, staged reopen, ancestor context changes,
Full fencing, primal/dual transitions, forced hash collisions, and
materialization, then model-check the invariant that no hard state is
reachable without a complete independently replayable tree. In parallel I
would mutate the eventual two implementations with deliberately correlated
one-sided theorem defects and audit every raw `ProofStatus`/certificate sink
through the trainer FFI. Empirically, I would demand an owner-escrowed,
game-clustered Production-500 persistent-batch holdout before believing any
2k/50k win; the highest remaining risk is scheduler/corpus overfit and
finite-cap coverage outside frozen manifests, not the already narrow
Exact/FC theorem row.
