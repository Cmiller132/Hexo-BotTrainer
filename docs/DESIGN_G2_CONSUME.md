# G2 consume mode: search-domain reduction under independently verified closure

Status: **design only; implementation not authorized**

Target: native proof-number search at eligible unforced defender AND nodes

Evidence snapshot: <code>claude/g2-cert</code> at
<code>af3a9a88904c46c5e757850fa95e75ea3641fef6</code>

Required next gates: fresh hostile review of this document, owner authorization
for the minimal FullControl/shadow baseline, step-zero kill screen, then a
separate owner decision on any Consume implementation

This document specifies the first Group-2 mode in which search work, rather
than only certificate materialization, may depend on FHW-T3-R. It does not
authorize source changes. The feature remains default-off even after an
implementation lands until every correctness and economics gate in section 6
passes.

The words MUST, MUST NOT, SHOULD, and MAY are normative.

## 1. Decision and evidence boundary

### 1.1 What is already established

The following are inputs to this design, not claims made by it.

| Evidence | Established fact | Consequence here |
|---|---|---|
| FHW proof lineage, theorem FHW-T3-R | Omitted-reply coverage is proved on its annotated class. Review 2 found the proof sound with stated errata and scope limits. | Consume may omit replies only through the exact admitted theorem class. It does not generalize D17, SR, commutation, scalar-budget debit, or arbitrary mixed histories. |
| Current Group-2 verifier v1+v2 | The verifier is fail-closed for the admitted Exact/FrontierCovered FHW class. NonFrontierCovered FHW edges remain rejected. | Search-side labels never enlarge the language. A forced NonFC site may use the separately reviewed exact compact-T6 seam; if that also fails, use Full/Unknown. It never enters as an FHW gate. |
| Owner-authorized mixed amendment | An ordinary unforced <code>UniversalGroup2V1</code> may contain forced Exact/FC <code>FhwGateV1</code> descendants and forced compact legacy <code>Universal(implicit_dispatch=true)</code> descendants. | Consume v1 admits exactly these three channels. Legacy zones, D17, SR, and commutation remain outside the language. |
| Wide-PN production report | Group-2 emission currently occurs after a group2-agnostic search. Across 6,462 cap-500 roots, 36 to 106 roots emitted gates; 142/142 gate-decision observations across two runs cross-verified at 50k and may overlap by root. | Current soundness evidence is useful, but the byte-identical ON/OFF nodes are expected: no gate is consumed by search today. |
| Ordinary selector fixture | One verified unforced <code>UniversalGroup2V1</code> uses 19 explicit edges where the full legal defender domain has 886 moves, about 46.6 times less local fanout. | This is the motivating local reduction. It is not a measured population-wide saving and it is not an <code>FhwGateV1</code> representative count. |
| Residue instrument at <code>e11c393d</code> | On human-160 at cap 50k, the directly timed unforced-defender generation block accounts for 40.46% of solve wall. | 40.46% bounds direct removal of that measured block, not total causal subtree savings, expected gain, or node count. |

Two reductions must remain distinct:

1. At an **unforced** defender node, <code>UniversalGroup2V1</code> closes an
   ordinary Group-2 edge set, called <code>S*</code> below, against the full
   legal set. The 19-versus-886 fixture is of this kind.
2. At a **forced** defender node with exact <code>tau=b</code>,
   <code>FhwGateV1</code> maps a forced kernel <code>K</code> to an Exact/FC
   representative set <code>R</code>.

Calling both of these a “gate” in telemetry must not cause their sets,
eligibility predicates, or prevalence measurements to be combined.

### 1.2 The decision in this design

Consume v1 is sound enough to build only if all of the following hold:

- the unforced search node has an explicit Open/Closed closure state;
- Open closure debt prevents a proof, cache promotion, or materialization;
- every failure to justify an omitted reply transitions atomically to the
  exact full legal defender domain, or ultimately returns <code>Unknown</code>;
- a completed search produces a standalone certificate that the strict
  verifier reconstructs without search state;
- the producer and verifier independently implement every theorem-specific
  derivation; and
- mode, claimant, horizon, closure context, authority, and search profile
  isolate all partial search state.

There are two different failure directions:

- **Search-domain fail-open:** when omission cannot be justified, open the
  domain to every legal defender reply. This preserves the uncapped semantic
  domain relative to full search; finite-cap coverage remains a measured KILL
  gate because speculative work is not refunded.
- **Trust fail-closed:** when a certificate or derivation cannot be validated,
  mint no hard result. Full search may still decide; otherwise return
  <code>Unknown</code>.

“Fail-open” in this document always means the first rule. It never means
accepting weaker evidence.

### 1.3 Non-goals

Consume v1 does not:

- admit NonFrontierCovered FHW edges;
- make digests into proof;
- allow a certificate to select verifier policy;
- persist Group-2 proof fragments across solver calls;
- turn failure of a restricted claimant search into proof for the opponent;
- prove completeness of alternative child-plan choices;
- add DAG certificates; certificates remain unfolded trees;
- claim a cap-500 gain; or
- change the scalar <code>B</code>, LOSS remainder, semantic horizon, or
  escape-deadline accounting proved by the existing theory.

## 2. Terms and invariants

For one positive proof attempt:

- <code>A</code> is the attempt claimant.
- <code>D</code> is the other player.
- An AND node is a state in which <code>D</code> moves and every relevant
  defender reply must be discharged.
- <code>L(P)</code> is the complete, canonical, duplicate-free list of legal
  defender moves at position <code>P</code>.
- <code>Full(P)</code> is the semantic obligation to prove a child for every
  move in <code>L(P)</code>.
- <code>Seed(P)</code> is the legal current hitting universe reconstructed
  from the complete claimant threat family, in canonical coordinate order. If
  it is empty, D9 supplies the canonical least legal move.
- <code>Required_FHW(P, plans)</code> is the independently derived
  <code>Z_dir union Z_seed union Z_touch union Z_virgin</code> set for the
  immutable child plans supplied at that occurrence.
- <code>S_g</code> is the append-only explicit edge set in closure generation
  <code>g</code>.
- <code>S*</code> is a nonempty fixed point for which
  <code>Required_FHW(P, frozen_plans) subseteq S*</code>.
- <code>SearchDeadForAttempt</code> means the restricted positive
  <code>A</code>-proof cannot continue through that child. It is not proof
  that <code>D</code> wins.
- <code>VerifiedCounterexample</code> is reserved for an exact terminal
  opponent outcome or a sealed, independently verified opponent certificate.
- A cutoff, unsupported forcing state, resource failure, missing derivation,
  ordinary positive-proof failure, or numeric disproof number of zero is
  neither opponent truth nor permission to mint LOSS.

The following invariants hold at all times:

1. Every edge in <code>S_g</code> is a distinct member of <code>L(P)</code>.
2. <code>S_g</code> is nonempty and append-only. No generation deletes,
   replaces, or reorders an earlier edge.
3. A child plan is frozen only after it is complete, cutoff-free, and
   immutable. It is merely a local candidate until its exact ancestor context
   is globally stable.
4. An Open or LocallyClosed node cannot be a proof even if every currently
   instantiated child is proved.
5. Only globally Closed or Full nodes can materialize an AND certificate.
6. A globally Closed node contains no <code>DepthCutoff</code>, unresolved
   child, provisional ancestor context, mutable selector, or reference to
   search-only state.
7. An omitted move is justified only by a strict verifier's reconstruction
   of the final certificate, never by producer telemetry.
8. Search may scan <code>L(P)</code> to establish legality, canonical order,
   the complement, and resource bounds. In Consume it MUST instantiate,
   apply, schedule, and recursively search only moves in the evolving
   <code>S_g</code>, unless it transitions to Full.
9. All counts include speculative closure and fallback work. No node or wall
   time is refunded because a derivation later fails.
10. A hard API result exists only after sealed concrete verification of the
    standalone certificate.

## 3. Consumption semantics

### 3.1 Modes and policy

Replace the current boolean concept with an explicit search mode:

| Mode | Search domain | Certificate behavior |
|---|---|---|
| <code>Off</code> | Existing frozen production search | Legacy verifier policy; historical search/status and canonical-legacy-certificate identity control |
| <code>Shadow</code> | Exactly the selected control's children and order | Derive telemetry only; cannot alter scheduling, certificates, caches, or status |
| <code>Verify</code> | Future <code>FullControl</code>: exact full unforced domain specified below | May materialize Group-2 certificates and strictly verify them; search does not consume them |
| <code>Consume</code> | May use the closed <code>S*</code> domain below | Must use externally selected <code>Group2V1</code> verifier policy |

Search mode and verifier policy are separate inputs. Certificate bytes MUST NOT
enable <code>Group2V1</code>. <code>Consume</code> with
<code>LegacyOnly</code>, an authority mismatch, or an unsupported certificate
version is a configuration error that returns <code>Unknown</code> before
search.

Today's reported group2-enabled, post-search-only wide behavior is called
<code>EmitLegacy</code> in this design; it is a regression lane, not the
future FullControl. The consumption A/B is invalid until FullControl exists. Its
control is <code>Verify</code>, with the same certificate grammar and verifier
policy as treatment; only omission during search differs. A separate
<code>Off</code>/<code>EmitLegacy</code> lane must reproduce its own historical
search status, node ledger, every legacy result field, and canonical legacy
<code>TssCertificate</code> bytes. This identity claim does not cover the
whole exported telemetry blob: only explicitly predeclared build/mode stamps
may differ. In particular, the current run-wide mint-engine version contract
requires the Consume-capable binary's predeclared v3 stamp even for Off; that
stamp does not alter canonical legacy certificate bytes. Before the Consume
comparison, FullControl must also
reproduce every Off decision at the same hard cap and status; it may
legitimately add coverage, but a lost Off decision blocks the campaign.

### 3.2 Exact eligibility

R2 is attempt-global: the exact solve root itself MUST be post-opening before
any new-class Group-2 search, emission, or materialization is enabled anywhere
in its descendant tree. A solve rooted in Opening uses the legacy/full
certificate path with Consume and promotion-Verify emission disabled (or
returns Unknown under a profile that cannot do so). Merely waiting until a
post-opening descendant would produce a whole certificate the verifier
correctly rejects.

The independent producer runs the unforced eligibility check only after the
ordinary depth, horizon, terminal, and immediate-winner checks. It MUST
reconstruct, rather than trust cached analyzer labels, all of these premises:

1. the state is post-opening and nonterminal;
2. the mover is defender <code>D</code> relative to this attempt's claimant;
3. placements remaining <code>b</code> is exactly 1 or 2;
4. the mover has no win now, under both exact direct-window reconstruction and
   the complete game outcome rules;
5. the node is unforced: from the complete claimant-alive threat family, the
   exact minimum transversal class has <code>k &lt; b</code>;
6. the complete legal domain is representable, nonempty, and canonical;
7. the configured authority, schema, arithmetic policy, and work limits are
   valid for starting a Group-2 closure.

These are a guarded dispatch, not a claim that every failed predicate is an
AND node. Opening, terminal, immediate-winner, and claimant-to-move states
bypass this hook and follow their exact existing opening/leaf/Choice paths.
Only exact <code>k=b</code>/<code>tau=b</code> may follow the separately
reviewed forced FHW or compact-T6 path. A <code>k>b</code> state follows an
independently established terminal/tactical claimant path when one exists;
otherwise the frozen profile produces SearchDeadForAttempt/Unknown or its
specified Full-fence path. It MUST NOT be relabeled <code>tau=b</code>. Only
an otherwise unforced defender occurrence whose omission premises are
unavailable, unsupported, or resource-failed enters <code>Full(P)</code>. No
failed predicate may accidentally select the reduced unforced arm.

Consume v1 changes child instantiation and scheduling **only** at this
ordinary unforced <code>UniversalGroup2V1</code> <code>S*</code> hook. Forced
Exact/FC FHW and compact-T6 nodes have identical search domains, child order,
and cap accounting in FullControl and Consume; any forced representative
reduction remains post-search certificate materialization. Therefore a root
with no eligible or indeterminate unforced hook has no other treatment
divergence point.

### 3.2.1 Exact FullControl and pair turns

This design resolves the semantic domain rather than leaving it to an
implementation choice. At every unforced defender AND occurrence,
<code>Full(P)</code> binds **every legal single placement in the exact current
engine phase**.

For a two-placement defender turn, the FirstStone state has <code>b=2</code>.
FullControl branches over every legal first placement. A nonterminal child
keeps the same defender at SecondStone with <code>b=1</code>, where another
Full Universal branches over that child's freshly reconstructed legal
placements. A terminal first placement is handled immediately and has no
invented second placement. Thus dynamic second-move legality and same-player
phase semantics come from exact state transition; the solver never
precomputes an unsound Cartesian pair domain.

The future native <code>Uniform</code> arm implements this lazy nested
single-placement domain. Consume replaces only an eligible occurrence's
instantiated Full child obligations with <code>S_g</code>; it does not change
turn phase or skip a required SecondStone state. Current wide PN lacks this
Uniform arm, so FullControl must land and pass exact exhaustive/legacy
regressions before a Consume A/B is meaningful.

### 3.3 Derivation timing and append-only closure

The hook is in native AND expansion, after exact state checks and before the
current implementation either generates defender children or declares the
unforced position outside its forcing class.

The normative state machine is:

~~~text
Unexpanded
  |
  | exact eligibility succeeds
  v
Open(g=0, S=Seed, closure debt)
  |\
  | \-- retained child is SearchDeadForAttempt --> A-attempt fails
  |
  |-- cutoff or unresolved child ----------------> remain Open
  |
  |-- all S_g children have frozen proofs
  |      derive Required_FHW from those immutable plans
  |      |
  |      |-- derivation fails/unsupported --------> Uniform Full(P)
  |      |-- missing set nonempty ----------------> append; Open(g+1)
  |      `-- no missing edge ---------------------> LocallyClosed candidate
  |
  `-- resource expires before safe transition ----> Unknown

whole-tree context pass
  |-- any candidate context changed --------------> highest affected node Full
  `-- all exact contexts stable ------------------> globally Closed

globally Closed -- standalone materialization -- strict verify --> sealed candidate
globally Closed -- verify failure ------------------------------> no hard value
~~~

More precisely:

1. Construct canonical <code>L(P)</code> and <code>Seed(P)</code>. This legal
   scan is not child enumeration.
2. Set <code>S_0 = Seed(P)</code>, create an Open closure record, and
   instantiate only those child obligations.
3. Drive each current edge to a complete positive proof, an explicit
   unresolved/cutoff state, or <code>SearchDeadForAttempt</code>.
4. If a required edge is <code>SearchDeadForAttempt</code>, this positive
   certificate attempt cannot close and returns no <code>A</code>-proof.
   Expanding the parent to Full cannot repair that fact because Full contains
   the same required child. Every permitted alternative proof and any
   recursive ungated Full-fence retry must occur **inside that child** before
   it receives SearchDeadForAttempt. It never establishes opponent truth. An
   exact terminal opponent outcome or sealed opponent certificate is stronger
   evidence, but even that mints LOSS only through the separately verified
   opponent-claimant leg.
5. If any edge is unresolved, do not derive a final summary and do not close.
6. Once all current edges are complete, freeze their exact materialization
   plans in a closure-owned immutable arena. The producer independently
   derives the full child summaries, roles, clocks, window demands, FHW rows,
   and <code>Required_FHW</code>.
7. Before using that derivation, require every descendant to be in the
   admitted legacy, compact-amended, or Exact/FC class, and require the
   semantic horizon to cover every derived leaf resolution and every exact
   <code>p(Q)+b+2</code> escape deadline. These are closure checks because the
   plans and deadlines do not exist at initial eligibility.
8. Canonically sort
   <code>M_g = (Required_FHW intersect L(P)) minus S_g</code>. A required
   coordinate outside the reconstructed legal set is a derivation failure,
   not something to drop.
9. If <code>M_g</code> is nonempty, append all of it, increment the generation,
   and repeat. The previously frozen plans remain immutable; no alternative
   selector is substituted.
10. If <code>M_g</code> is empty, store the child-plan digest,
    producer-summary digest, authority, provisional exact-context fields and
    their digest, and <code>S*</code>. This is a LocallyClosed candidate, not
    a proof or materializable node.

### 3.3.1 Whole-tree context stabilization

Verifier roles, clocks, and window demands are derived over the whole unfolded
tree and flow from ancestors into descendants. Appending an ancestor sibling
can therefore change a descendant's required set even when the descendant
board and child bytes did not change. Occurrence-scoped TT keys alone do not
solve this dependency.

Every ancestor append increments a whole-tree context epoch. Once the current
tree has local candidates and complete Full/legacy plans, a coordinator:

1. unfolds the exact occurrence tree for the current epoch;
2. independently derives complete budgets, roles, clocks, and incoming
   window-demand context top-down and child summaries bottom-up;
3. compares **exact canonical context fields**, not only a digest, at every
   Group-2 candidate; and
4. refuses root closure while any context, required edge, or descendant plan
   is provisional.

V1 chooses the simple terminating policy: collect every
**ancestor-minimal mismatch**—a mismatched Group-2 occurrence with no
mismatched Group-2 ancestor—in canonical occurrence-path order. Atomically
transition that disjoint set to Full, invalidate each member's
occurrence-local descendants and stale parent epochs, and recompute. This
removes ambiguity when several incomparable mismatches are equally high.
Add every transitioned canonical path to a solve-local
<code>forced_full</code> set keyed by root binding, claimant, horizon, rule
profile, authority, and move/phase path. The marks survive ancestor rewrites
for the rest of the solve: if Full expansion later exposes or recreates such a
path, it cannot attempt G2 again. Do not delete old explicit moves and do not
try a different reduced plan.

Only a no-change whole-tree pass atomically marks the surviving candidates
globally Closed. A descendant that verifies as a standalone root with empty
incoming demands is not thereby Closed in an embedding. A future
context-parametric or monotone-reopen optimization requires a new theorem and
hostile review.

Termination follows from append-only growth inside finite <code>L(P)</code>.
For whole-tree stabilization, consider the finite fully unfolded placement
tree under the exact board and semantic horizon. Every mismatch adds a
previously unmarked canonical path from that finite universe to the monotone
<code>forced_full</code> set. Newly exposed subtrees may contain new G2
candidates, but they were already paths in that finite universe, and a marked
path can never be recreated as gated. Thus stabilization reaches a no-change
mix or exhausts its explicit resource cap and returns <code>Unknown</code>.
Failure to store or compare the path exactly is itself Unknown/KILL.
Termination alone is not soundness; strict reconstruction of the resulting
certificate is.

### 3.4 Proof-state provenance and PN/DN while Open

Closure is a semantic obligation, not telemetry. Open has the **numeric
equivalent** of an extra AND debt with <code>(pn,dn)=(1,INF)</code>, stored in
the Open node state rather than in the board-move vector. Selectors can never
apply it as a move. When every concrete child is proved, the work dispatcher
runs the closure event before any threshold, root-completion, materialization,
or cache-promotion check. Appending edges retains the debt; global closure
discharges it.

Numbers alone cannot carry truth provenance. Every entry and every aggregate
stores an explicit state such as:

~~~text
Pending
OpenDebt
CandidateProven
SearchDeadForAttempt { reason }
Unresolved { all_causes, selected_path, selected_cutoff_depth_if_any }
VerifiedCounterexample { terminal | sealed_opponent_cert }
~~~

The normative aggregation algebra is:

| Parent | Decisive state | Otherwise |
|---|---|---|
| Choice for claimant A | <code>CandidateProven</code> if any selectable child is CandidateProven | SearchDeadForAttempt only after every selectable child has exhausted permitted alternatives and is SearchDead/VerifiedCounterexample. If none is Proven and any selected/live child is Unresolved, propagate that selected unresolved cause; otherwise Pending. |
| Universal for claimant A | SearchDeadForAttempt if any required child is SearchDead/VerifiedCounterexample | CandidateProven only if every required child is CandidateProven **and** global closure debt is discharged. Otherwise propagate the scheduler-selected Unresolved cause if one exists; otherwise OpenDebt or Pending. |

<code>VerifiedCounterexample</code> has the SearchDead effect inside this
positive A-attempt, but it does not mint opponent truth here. The separate
opponent-claimant leg must still produce the LOSS certificate.

The scheduler preserves every unresolved provenance needed to choose work and
records the exact unresolved path it selected. Stage advancement uses that
selected path's cutoff depth; it does not replace it with an unrelated global
minimum. A decisive Proven Choice sibling or SearchDead Universal sibling may
override other unresolved children exactly as the table states. A deeper
cutoff may never become SearchDead merely because intermediate PN/DN min/sum
arithmetic produces <code>dn=0</code>. OpenDebt prevents CandidateProven.
SearchDead never mints LOSS, persists as opponent truth, or crosses
claimant/profile/cache boundaries.

Current exits map as follows:

| Cause | Provenance/action |
|---|---|
| Depth cap | <code>Unresolved{cutoff_depth}</code> |
| Node/work/memory exhaustion | <code>Unresolved{resource}</code> |
| Semantic-horizon or live-leaf resolution refusal | <code>Unresolved{semantic_horizon}</code> |
| Unsupported forcing class after all required Full-fence alternatives | <code>SearchDeadForAttempt{unsupported_class}</code> |
| Exact terminal claimant win | <code>CandidateProven</code> leaf |
| Exact terminal opponent win | <code>VerifiedCounterexample{terminal}</code> |
| Sealed opponent certificate | <code>VerifiedCounterexample{sealed_cert}</code> |
| Producer derivation/admissibility failure | Atomic transition to Full; if resources prevent publication, <code>Unresolved{resource}</code> |
| Strict rejection of a globally Closed candidate | No hard state; <code>Unknown</code> and campaign KILL |

Only exact terminal game logic or a sealed opponent certificate constructs
<code>VerifiedCounterexample</code>. Today's wide <code>Refuted</code>
constructors for horizon refusal, unsupported/unforced state, forcing-search
failure, and resource limits MUST map to
<code>SearchDeadForAttempt</code> or <code>Unresolved</code>, never opponent
truth. LOSS still requires a fresh positive opponent-claimant certificate.

Every number consumer—root stop, threshold stop, work dispatch, sibling
selection, transposed-parent refresh, fragment promotion, materialization, and
solve-goal mapping—MUST consult provenance. A direct check that the immediate
child node is not <code>DepthCutoff</code> is insufficient.

### 3.5 Full-domain fallback

On an otherwise-unforced omission-precondition or derivation failure, the semantic fallback is
<code>Uniform Full(P)</code>: bind every move in canonical
<code>L(P)</code>, not a zone subset, a forced kernel, or today's immediate
wide-PN attempt refusal. In v1 this publishes a **recursive Full fence**:
ordinary unforced descendants of that occurrence also use Full and cannot
Consume. They may still emit independently verified Group-2 certificate
evidence after the full search, but no descendant omission affects search.
This makes a fallback subtree context-free and prevents repeated
G2-to-Full-to-G2 cycling.

An Open-to-Full transition is atomic:

1. stop using the closure debt, summary, missing-set queue, and all
   mode-dependent PN/DN values;
2. construct the separately keyed occurrence-local Full overlay off to the
   side, optionally sharing only its read-only state core;
3. bind the complete legal domain before publishing it or reporting Proven;
4. add the exact occurrence path to the solve-local
   <code>forced_full</code> set so every later recreation remains fenced;
5. atomically retarget the occurrence's sole owner handle, which is exactly
   <code>RootAttemptSlot</code> or <code>ParentEdge</code>. For ParentEdge,
   refresh its ancestor chain deepest-first. For RootAttemptSlot, recompute and
   publish the root overlay numbers directly; and
6. retain only immutable legacy/context-free positive child fragments.
   Partial numbers, cutoffs, negative sentinels, closure summaries, and
   unverified plans are discarded.

V1 defaults to discarding even positive child fragments during the transition.
No <code>UniversalGroup2V1</code>, <code>FhwGateV1</code>, path-dependent
plan, or Group-2 summary crosses from the occurrence overlay into Full:
standalone validity under empty incoming demand does not prove safe embedding
under a stronger context. Reuse is limited to the exact same
<code>FullFenceOverlayKey</code> with a matching current mutable epoch; or to
the immutable read-only FullCore/separately sealed legacy cache. It is never
hard opponent truth. Any broader reuse requires exact
destination-context unfolding, rederivation, strict reverification, and a
separate reviewed optimization.

If the remaining resource budget cannot establish the full binding, return
<code>Unknown</code>. Resource exhaustion never becomes
SearchDeadForAttempt or opponent truth.

The current native wide search assigns its <code>Refuted</code> attempt tag to
non-implicit unforced defender nodes; it therefore has no
<code>Uniform Full(P)</code> arm today.
The narrow path's optional legacy zone is also not Full. Consume cannot claim
FullControl equivalence until the exact nested single-placement baseline in
section 3.2.1 exists and is tested. This is GAP C01, not an implementation
detail.

### 3.6 DepthCutoff and staged reopen

Depth is checked before eligibility at a newly reached node. A node cut off
there has no closure state.

For a cutoff below an Open node:

- the cutoff edge remains an explicit member of <code>S_g</code>;
- the parent remains Open and bubbles the exact encountered cutoff depth;
- the cutoff is not a failed derivation, a frozen plan, or a genuine
  counterreply;
- no final <code>Required_FHW</code> is derived while any current child is
  cutoff or unresolved; and
- the parent cannot materialize or enter a persistent cache.

Stages advance only to the selected encountered cutoff. At the next stage,
every cutoff entry whose intrinsic depth is at most the new stage becomes
Unexpanded; each re-expansion is charged again. Then a global deepest-first
refresh clears stale provenance and PN/DN values from every ancestor.
Complete, immutable, cutoff-free child plans MAY remain frozen within the same
whole-tree context epoch. A newly completed child may add roles, clocks,
window demands, or required edges, so every dependent summary is recomputed.
Newly required moves append; old moves are never removed.

A globally Closed node is stable only under its frozen exact ancestor context,
semantic horizon, engine version, and authority. Staged depth growth does not
change those semantic inputs. An ancestor append before global closure bumps
the context epoch and invokes section 3.3.1. A horizon, engine, authority, or
resume-binding change is a fresh solve that discards the occurrence arena; it
does not mutate a Closed node across contexts. Any unexpected replacement of
a legitimately frozen plan is an invariant violation and correctness KILL,
with fail-closed runtime fallback to Full/Unknown.

Required tests compare:

- staged search with one-shot search at the final depth and a sufficient cap;
- Consume with FullControl at the same final depth and a sufficient cap;
- the final strict certificate after an omitted move becomes required only
  after a descendant reopens; and
- deletion of that newly appended move, which must reject.

An Open root left at the final stage is <code>Unknown</code>. It is never a
SearchDeadForAttempt or opponent proof merely because its selected child was a
cutoff.

### 3.7 LOSS goal and dual pass

Every solver leg is a positive proof for a fixed claimant:

- <code>Win</code>: claimant is the root player;
- <code>Loss</code>: claimant is the root player's opponent;
- <code>Both</code>: the primal leg uses the root player; if needed, the dual
  leg independently uses the opponent.

Eligibility, threat ownership, roles, clocks, gates, closure context, and the
final certificate are all relative to that leg's claimant. They are never
relative to the spelling of <code>SolveGoal</code>. A dedicated Loss query and
the dual leg of Both therefore run the same Consume algorithm with the
opponent claimant.

Each leg MUST have a fresh closure arena and a fresh mode-local TT namespace.
No Open, Closed, cutoff, negative result, child plan, or producer summary flows
from primal to dual. A verified positive proof may be logically reusable, but
v1 deliberately does not cross this boundary.

The primary Consume query preserves the existing combined node ledger:

- the initial reserve and allocation policy are fixed by the manifest;
- speculative closure, fallback, and verification-triggered search work count
  against the leg that incurred them;
- with reserve zero, primal initially receives all post-root work and the
  actual dual allowance is the hard cap minus all accumulated query nodes,
  including root, speculative closure, cutoff re-expansion, and local
  fallback; there is no refund for failed closure; and
- an internal SearchDeadForAttempt state does not prove the opponent. LOSS is
  minted only from an independently constructed and strictly verified
  opponent proof.

Because a Both query can spend its entire allowance in primal, the correctness
suite also runs dedicated <code>goal=Loss</code> fixtures and corpus strata;
observing Both alone is not dual-pass coverage.

There is one hard combined cap. Local Open-to-Full fallback spends only the
nodes remaining in that claimant leg and query. There is no uncounted refund
and no second query outside the cap. Consequently, a changed order can still
lose a control decision at a finite cap even though the semantic domain opens
correctly. The promotion rule treats any such loss as KILL. This keeps the
2k/50k experiment genuinely paired and distinguishes semantic completeness
from finite-budget coverage.

### 3.8 TT, transposition, resume, and proof-cache isolation

The current wide position key encodes board state and phase, but not claimant,
horizon, mode, or closure context. That is insufficient for Consume.

There are distinct key types; one overloaded position key is forbidden:

~~~text
StateCoreKey =
  exact engine position and phase
  + engine/rule version

OccurrenceBase =
  StateCoreKey + claimant + semantic horizon
  + width/search-profile digest + primal-or-dual leg
  + immutable mint-engine/build version + verifier policy
  + Group-2 authority/schema version
  + exact canonical incoming-context fields + unique occurrence identity

FullFenceOverlayKey = OccurrenceBase + FullFenceV1
G2OverlayKey        = OccurrenceBase + G2ConsumeV1
~~~

Only StateCoreKey transposition-shares, and only the immutable data enumerated
below. Both mutable overlay variants are occurrence-scoped. A sealed
legacy-only/context-free proof cache is a third, separately policy-keyed
facility; it is not a Full overlay lookup.

Open/LocallyClosed/Closed, generation, and context epoch are mutable entry
state under the immutable <code>G2ConsumeV1</code> regime; they are not
different semantic lookup keys. Stage depth is likewise session state, not TT
equality, so complete plans can survive ordinary staged deepening. The whole
resume binding stores final depth policy, current stage, all tuple fields, and
the arena epoch; any mismatch discards the session.

Exact canonical incoming role/window context and unique occurrence identity
participate in equality. Their digest may choose a bucket and detect drift,
but a digest alone never authorizes a hit. Closed additionally stores frozen
plan and summary digests for materialization checks. These establish identity
only; the verifier reconstructs meaning.

V1 uses the safest transposition policy:

- A read-only <code>FullCore</code> may be shared by exact state for canonical
  legal moves and exact engine outcomes only. It
  stores no PN/DN aggregate, provenance, chosen child, frozen proof plan,
  CandidateProven status, Group-2 child overlay, or materialization evidence.
- Every mutable proof-plan overlay—Full or Group-2—is occurrence-scoped in
  FullControl and Consume. Equal boards under different incoming edges may
  share the FullCore but not PN truth or an outgoing plan, because a nested G2
  descendant can receive different top-down demands.
- A Full TT/core hit is therefore only a state-construction hint. It cannot
  provide a frozen/materializable plan. Legacy-only, context-free verified
  proof reuse remains governed by the stricter cache rule below; any selected
  proof containing a Group-2 descendant is always occurrence-local.
- Open-to-LocallyClosed and global Closed transitions bump the occurrence
  epoch atomically. Every owning parent reference checks occurrence, regime,
  and epoch before reading provenance, PN/DN, or a plan; old aliases are
  invalidated before publication.
- A Group-2 occurrence-local overlay has exactly one owner handle:
  <code>RootAttemptSlot</code> or <code>ParentEdge</code>. Open-to-Full builds
  a complete occurrence-local Full overlay off to the side, optionally sharing
  only the read-only FullCore, atomically retargets that handle, invalidates
  the G2 alias, and either republishes root numbers or refreshes the owning
  ancestor chain. Partial Full child binding is never visible.
- A G2-regime entry can never answer a Full lookup, and a Full entry can never
  answer a G2 lookup.
- Search DAGs are unfolded during materialization. Every certificate
  occurrence gets independently derived evidence and tree indegree one.
- <code>DepthCutoff</code>, Open, LocallyClosed, Unknown,
  SearchDeadForAttempt, negative sentinels, and partially materialized entries
  never enter a persistent proof cache.
- Existing persistent rejection of <code>UniversalGroup2V1</code> and
  <code>FhwGateV1</code> fragments remains in v1. Cross-call reuse requires a
  separate reviewed design.
- At the primal/dual and mode boundaries, fresh or disabled state includes
  local position maps, shared proof caches, fragment stores, finder/selector
  memoization, closure/digest memoization, prior/negative tables, and resume
  handles—not just the object called TT.
- Toggling mode or verifier policy clears all of that state.
- Resume state binds the exact monotone <code>forced_full</code> path set and
  its epoch; loss, mismatch, or partial restoration discards the session.
- Current source defines <code>TSS_CERT_VERSION</code> as a compile-time,
  run-wide mint-engine/schema stamp and requires a bump for any mint-engine or
  grammar change. A Consume-capable binary therefore selects immutable v3
  before search and includes it in every TT and resume identity, including
  Off. It never derives that key from the eventual materialized result. If the
  original extension's dynamic v2/v3 reporting is retained, it must become a
  separately named post-materialization emitted-certificate-class field after
  an API review; it cannot replace or key on the run-wide engine stamp.

Forced hash collisions, A/B/A mode sequences, G2-to-Full and Full-to-G2
lookups, stale Open/Closed/Full epoch aliases, primal-to-dual prewarming of
every cache category, and diamond transpositions are mandatory tests against
cold Full results. One diamond specifically shares a FullCore whose two
occurrence overlays contain a G2 descendant under different incoming demands;
neither overlay may observe the other's plan or CandidateProven state.

### 3.9 Determinism and resource limits

Parallel materialization currently changes how many post-search gates are
selected. That was harmless while search was identical. It is unacceptable
when selection changes capped search.

Consume v1 fixes:

- canonical coordinate order;
- canonical threat, window, role, demand, missing-set, and row order;
- deterministic first frozen plan per occurrence;
- deterministic closure-event priority before ordinary PN threshold checks;
- deterministic collision resolution; and
- one worker for search, closure, materialization, finalization, and every
  relevant hash-map traversal in v1, with canonical iteration at each seam,
  unless schedule-independent canonical outcome has been separately proved
  and hostile-reviewed.

Repeated runs from empty caches MUST have identical status, node count,
closure generations, consumed occurrence set, edge counts, certificate bytes,
and verifier result. Nondeterminism is a correctness-gate failure, not
statistical noise.

All checked arithmetic, legal-set size, closure generations, frozen-plan
bytes, row counts, recursion, and verification work have explicit caps. A cap
failure transitions Open to Full when possible; otherwise it yields
<code>Unknown</code>. It never closes with truncated evidence.

## 4. Soundness architecture

### 4.1 Full trust chain

~~~text
Pinned FHW-T3-R theorem and owner amendment
                  |
                  v
Independent search producer selects S* and frozen child plans
                  |
        Open debt / append-only local fixed point
                  |
       exact whole-tree context epoch reaches no-change pass
                  |
                  v
Standalone mixed tree certificate, unfolded from any search DAG
                  |
                  v
Independent Group2Verifier replays root and every edge,
reconstructs every premise, and checks complete omitted-reply coverage
                  |
                  v
Sealed concrete mint re-verifies under externally selected policy
                  |
                  v
Hard Win or Loss
~~~

The search may instantiate a provisional reduced domain, but LocallyClosed is
not CandidateProven and cannot materialize. Only a globally Closed no-change
epoch can become a search proof candidate, and even that candidate is not an
API verdict. A hard result requires a certificate verifiable from:

- the exact root state;
- claimed status and therefore claimed player;
- certificate bytes;
- externally selected verifier policy and work limits; and
- compiled authority/schema/version constants.

The verifier receives no PN/DN values, TT entries, closure epochs, search
telemetry, finder caches, hidden child plans, or omitted-state oracle. It
replays every represented move, enforces an unfolded tree, derives exact
metadata and resolution, checks R2/post-opening and horizon rules, reconstructs
the complete threats and legal domains, derives roles and clocks, validates
ordinary Group-2 coverage, validates every Exact/FC FHW row and map, checks
mixed compact-T6 nodes, and recomputes all canonical digests.

Materialization serializes the exact immutable plans and edge sets from the
globally Closed epoch. It MUST NOT rerun a selector, choose a cheaper
alternative child proof, or derive a different closure after search has
claimed completion.

If initial in-process verification fails, materialization returns no
certificate and the hard candidate is discarded; remaining budget may enter
Full, otherwise the result is <code>Unknown</code>. Even after it passes, the
sealed result-mint boundary runs the concrete verifier again. Any rejected
Closed candidate or other producer/verifier discrepancy is a correctness KILL
in evaluation, even though fail-closed runtime behavior prevented a hard
result.

### 4.2 Authority and admitted language

The manifest binds the controlling defender authority and FHW companion by
commit and content digest:

- controlling <code>docs/PROOF_TSS_DEFENDER_ZONES.md</code> at
  <code>6dc08d7a89d422524f6d92dadf662073d25b1963</code>, SHA-256
  <code>39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B</code>;
- companion <code>PROOF_TSS_ZONES_FHW.md</code> at
  <code>9945c21bf177055aa4de0bbd3aad15b9cf245e51</code>, SHA-256
  <code>16F7D684B5D763E8B673EC3A03B5110B9ABF5BB7E80FCA063E62C81A113F9EA0</code>.

The verifier accepts only the exact schema and dual authority chosen before
search. The obsolete authority candidate and all unknown tags or fields
reject.

For Consume v1, the recursive certificate language is limited to:

1. admitted legacy Choice and leaf forms, plus a full-legal ordinary
   Universal where the mixed verifier permits it;
2. unforced <code>UniversalGroup2V1</code> with exact required-set coverage;
3. forced <code>FhwGateV1</code> with Exact or FrontierCovered edges only;
4. forced compact legacy <code>Universal(implicit_dispatch=true)</code> with
   the verifier-reconstructed exact <code>tau=b</code> kernel; and
5. the owner-amended combinations of those nodes.

A compact forced node does not seed new gate demands, but it propagates
incoming demands and participates in ordinary full <code>1+max</code>
recurrences. A gate parent uses its reviewed paired cut-clock rules. No node
kind may be relabeled to borrow the other's omission rule.

### 4.3 Independent producer and verifier

The current code does not meet the required boundary: production finder paths
call semantic helpers housed in <code>tss_verify_group2.rs</code>, including
required-set derivation, gate construction, row filling, finalization, and
digest inputs. A hand audit of shared helpers was enough for the current
post-search Exact/FC lane; it is not enough once search omits work.

Before Consume exists, create two independent implementations:

**Search producer**

- owned by a production search/finder module;
- derives eligibility, complete threat families, exact <code>k</code>,
  <code>b</code>, and <code>tau</code>;
- derives <code>H</code>, <code>F</code>, <code>K</code>, <code>R</code>,
  <code>phi</code>, Exact/FC classification, and escape sets;
- derives budgets, roles, <code>Q/E</code> clocks, window demands, zones,
  ordinary required-set closure, Cartesian rows, and stored scalars;
- owns Open/Closed fixed-point selection; and
- independently encodes every canonical digest preimage.

**Strict verifier**

- remains in the verifier boundary;
- independently replays and derives the same mathematical objects;
- never imports or calls the producer, solver, PN state, or finder caches;
- never trusts producer eligibility, summaries, rows, classifications, or
  digests; and
- uses its own canonical traversal and digest-preimage encoder.

They MAY share only:

- immutable certificate record types and numeric tag constants;
- compiled authority constants;
- raw cryptographic hash primitives, not semantic digest encoders; and
- minimal game primitives: coordinate representation, exact board ownership,
  legal state transition, phase, and terminal/winner rules.

They MUST NOT share theorem decision tables, threat/transversal analyzers,
role or clock derivation, window-demand logic, zone construction, gate
classification, fixed-point closure, row builders, plan summaries, or
canonical semantic encoders. If raw winning-window enumeration is shared as
an engine primitive, each side still independently reconstructs all
theorem-specific families from it; the hostile review must approve that exact
boundary.

Enforcement is part of the landing gate:

- a transitive call-graph/source allowlist fails if producer and verifier share
  anything beyond the named data-only records/constants, raw hash primitive,
  and minimal engine primitives above; moving a semantic helper into a neutral
  third module still fails;
- shared record types expose no theorem method, derived accessor, semantic
  encoder, or digest builder;
- the current <code>threats_shared::analyze</code> may remain verifier-only as
  an additional rejector that cannot establish any acceptance premise.
  Producer omission logic cannot call it, and verifier acceptance must survive
  removing its positive result;
- the Consume call graph has no call to the current
  <code>finder_required_fhw</code>, <code>finder_build_fhw_gate</code>,
  <code>finder_fill_gate_rows</code>, or
  <code>finder_finalize_group2</code> helpers in the verifier module;
- independent golden vectors and differential corpora require agreement, but
  agreement is supplemental and does not substitute for code independence;
  and
- the fresh hostile reviewer audits both implementations against the theorem,
  rather than reviewing one through the other.

Digests detect drift and tampering. They do not make correlated logic sound.

## 5. Counterexample and test obligations

Every test below is either:

- **must-reject:** a mutated or forged standalone certificate is rejected by
  the strict verifier under <code>Group2V1</code>; or
- **must-equal:** at a sufficient/exhaustive test cap, Consume and Full produce
  the same hard truth on the exact root; at any finite campaign cap, every
  root decided by both has identical status and the separate coverage gate
  permits no lost FullControl decision. Every certificate independently
  verifies. Different capped order is not hidden as a truth mismatch.

Tests build on the existing Group-2 mutation suites, D6 fixtures,
R-Z11/FHW-O1 hostile cases, Exact/FC adoption fixture, every-edge deletion
battery, escape-horizon amendment battery, and mixed seam battery. Passing old
tests is necessary but not sufficient because none exercises native
Open/Closed PN state.

| ID | Attack | Required oracle |
|---|---|---|
| CE01 | Attempt reduced closure or promotion-Verify Group-2 emission with an Opening solve root (even at a post-opening descendant), at an Opening node, terminal state, claimant-to-move state, or after mover win-now. | Attempt-global R2 disables every new-class search/emission path for the whole Opening-root solve. Other cases follow ordinary opening, leaf, Choice, or winner paths. Forged reduced cert must-reject. Cover direct-window and analyzer win-now paths. |
| CE02 | Lie about <code>b</code>, omit a live threat from the complete family, or classify exact <code>k>=b</code> as unforced <code>k&lt;b</code>. | Exact k=b may follow reviewed forced FHW/compact; k>b must never be relabeled tau=b and follows only an independently proved tactical/terminal, Full-fence, SearchDead, or Unknown path from the frozen profile. Mutation must-reject. Boundaries are b=1: unforced k=0, forced k=1, above-bound k>1; b=2: unforced k=0/1, forced k=2, above-bound k>2. |
| CE03 | Supply an empty, duplicate, illegal, or incomplete required edge set, or make producer order noncanonical. | Producer failure opens Full; illegal/duplicate/empty explicit certs must-reject. D9 makes the producer choose one canonical legal seed when Required is empty, but the verifier must accept any nonempty legal superset that independently covers Required; producer determinism is not a theorem premise. |
| CE04 | Omit FHW-O1's quiet reply <code>x</code> by dropping the <code>max{b,...}</code> floor in <code>Q_cut</code>. | Existing static mutation must-reject; new staged must-equal fixture reaches <code>x</code> only after reopen, appends it, and verifies. Deleting <code>x</code> and its orphan subtree must-reject. |
| CE05 | Let Choice → Universal → Open → <code>DepthCutoff</code> propagate numeric <code>dn=0</code> through several ancestors and treat it as refutation. | Provenance remains Unresolved with the exact cutoff depth. Must-equal staged versus one-shot at sufficient final cap and cold Full. Open cannot materialize; final unresolved root is Unknown. |
| CE06 | Reopen a formerly unresolved child and thereby add roles, <code>Q</code>, window demand, or a required zone; or let ancestor appends create several incomparable context mismatches. | Force generation/epoch growth; old candidates cannot close. V1 canonically sorts and atomically fences every ancestor-minimal mismatch. Stale-plan/context/row variants must-reject. A legitimately frozen plan changing is KILL, not normal reopen. |
| CE07 | Trigger producer error, arithmetic overflow, closure work cap, allocation cap, or a required coordinate not in legal. | Must-equal FullControl at sufficient cap; with insufficient remaining budget, Unknown. Never SearchDeadForAttempt, LocallyClosed, or globally Closed. Fault-inject each exit. |
| CE08 | Present genuine NonFrontierCovered geometry, spoof it as FC, omit a geometrically non-FC kernel reply, or map outside <code>K</code>. | A forced NonFC site deterministically uses the exact reviewed compact-T6 seam when its premises hold; otherwise remaining-budget Full/Unknown. It never emits FhwGateV1. FC/kernel spoof certificates must-reject; preserve the reductive FC positive fixture and all D6 images. |
| CE09 | R-Z11 boundary: all-empty incident direct fill with <code>q=5</code> is accepted by weakening strict <code>1+q&lt;6</code>; relabel it nonincident. | <code>q=5</code> must-reject, <code>q=4</code> passing neighbor must verify, and incident-bit/row mutation must-reject. |
| CE10 | Forge a NonFC root gate, delete a purported WC row, or use an ancestor demand to excuse it. | V1 rejects the NonFC gate before WC rows can grant acceptance. Keep row-deletion vectors as future NonFC-extension negatives; this design claims no positive NonFC WC support. |
| CE11 | Splice a cheaper role or clock from sibling <code>C_s'</code> into <code>C_s</code>, or pair after taking maxima. | Must-reject cross-branch row/role mutation. Unrequested rows also reject. |
| CE12 | At one real mixed parent, swap a gate child with a compact child, copy gate <code>child_f/child_q</code> from the cheaper compact sibling, or remove the ordinary G2 edge whose demand flows through that sibling. | Positive mixed fixture verifies. Every splice/removal must-reject. Consume and Full must-equal in both canonical construction orders. |
| CE13 | Relabel compact forced T6 as full legacy, duplicate/drop a forced-kernel edge, relabel compact as bogus gate, or smuggle zone/commutation fields. | Preserve all existing mixed-seam must-reject tests. No broader amendment is inferred. |
| CE14 | DAG-share a board reached through a gate representative and a compact edge, or let two parents share a FullCore whose selected proof contains a G2 descendant under different incoming demands. | Only read-only canonical legal moves and exact engine outcomes may be shared. Priors, PN/provenance, and plans remain in two occurrence overlays; materialization emits two tree occurrences. Plan sharing, child-ID swap, and duplicate-discharge mutations must-reject. |
| CE15 | Visit the same position G2 then Full, Full then G2, and G2/Full/G2 under forced hash collisions and stale phase epochs. | Every warm result must-equal a cold Full result; regimes never hit each other and parent epoch checks reject stale Open/Closed/Full aliases. Repeat across cache eviction and mode toggles. |
| CE16 | Reuse a primal claimant plan, TT entry, or cert in dedicated Loss or the Both dual leg, including a G2 occurrence at the attempt root. | Prewarm each cache category in both orders. Exercise RootAttemptSlot Open→Full and stale epochs in dedicated Loss and Both-dual. Win/Loss/Both must-equal FullControl; claimed-player verification rejects cross-claimant certs. |
| CE17 | Exclude a gate escape deadline from <code>T</code>, or use a horizon sufficient for leaves but below <code>p(Q)+b+2</code>. | Existing amendment horizon mutation must-reject. Search opens Full or returns Unknown before closure. |
| CE18 | Mutate a frozen plan after closure, splice stale bytes, change canonical child order, or accept digest equality as proof. | Every stale/spliced form must-reject and search never swaps its first frozen plan. A separately recomputed alternative valid plan is allowed to verify; the incompleteness rule controls producer selection, not verifier truth. |
| CE19 | Let Open reach proof number zero, persist an Open/negative/cutoff fragment, or materialize before the closure event. | State-machine assertions and fault-injected cache tests must-equal cold Full; no certificate bytes may be produced from Open. |
| CE20 | Let parallel scheduling choose a different closure, edge set, cap outcome, or certificate. | Repeated empty-cache runs must be byte- and count-identical. Any difference is KILL before economics are read. |
| CE21 | Remove each explicit ordinary G2 edge in turn, prune its orphan, and retain all summary/digest fields. | Every mutation must-reject, including the edge added only by a later closure generation. |
| CE22 | Verify producer and verifier agreement only because they call the same direct or neutral-third-module helper. | Transitive dependency/firewall test must fail the build. Use theorem-derived golden expected results and separately seeded one-sided defects whose expected reject/fallback is fixed in advance; two implementations merely disagreeing does not identify an oracle. |
| CE23 | Combine the dangerous seams: one common parent has forced gate and compact siblings; a deeper cutoff reopens, raises <code>child_f/child_q</code> or incoming demand, adds a formerly omitted ordinary-G2 edge, and the same board is offered through a forced TT collision. | Run both claimants, Win/Loss/Both, sufficient-cap staged/one-shot FullControl, and all D6 images. Drop the propagated compact-seam demand, splice the cheaper sibling row, reuse the stale epoch, and delete the new edge plus orphan; every mutation must-reject and every unmutated verdict must-equal. |
| CE24 | Mis-tag each current wide <code>Refuted</code> constructor—semantic-horizon refusal, unsupported unforced state, forcing-class failure, or resource exit—as opponent truth. | Provenance tests require SearchDeadForAttempt or Unresolved as specified; no persistent negative, LOSS mint, or cross-claimant hit. Only terminal opponent truth or a sealed opponent cert constructs VerifiedCounterexample. |
| CE25 | Flatten a two-placement defender turn into a stale Cartesian pair set, demand a second move after a terminal first placement, or reuse second-move legality from a different first stone. | Exhaustive pair-phase must-equal tests require nested exact FirstStone/SecondStone Universals, dynamic legal reconstruction, immediate terminal handling, and independently verified FullControl certificates. |

In addition, exhaustive small-board oracles MUST compare Full and Consume
where feasible, including every eligibility boundary, staged depth, both
claimants, D6 transforms, mixed node kind, and injected producer failure.

## 6. Pre-registered promotion bar

### 6.1 Freeze rule

Thresholds may be adjusted only before the evaluation manifest is frozen.
After any arm's result is observed, a threshold, corpus, root parser, flag,
counter definition, exclusion, or machine setting change creates a new
campaign identifier and requires both arms to be rerun. Failed campaigns stay
reported.

The manifest binds:

- implementation commit and clean-tree status;
- theorem, amendment, authority, schema, certificate, engine, producer, and
  verifier content digests;
- canonical parsed root bytes and source file digests;
- binary, compiler, target, optimization, CPU, OS, thread count, affinity,
  scheduler, and environment;
- exact goal, claimant mapping, <code>wide</code>, <code>dual_pass</code>,
  loss reserve, zone setting, horizon resolver, hard combined cap, TT bytes,
  work/memory caps, deterministic ordering, and verifier policy;
- fresh-cache construction, three same-binary repetitions, and balanced AB/BA
  wall-run order;
- definitions of node, verified decision, local fallback, closure occurrence, and
  verifier failure; and
- scripts and formulas that produce every table.

### 6.2 Fixed operating points

The mandatory promotion cells are:

1. **Labeling-2k:** the existing 6,462-root production-configuration
   evaluation corpus (three dev splits plus F19) at hard combined cap 2,000,
   consisting of 3,255 <code>selfplay_v1</code>, 2,720
   <code>human_v1</code>, 468 <code>puzzle_v3</code>, and 19 forcing roots.
2. **Atlas-50k:** a genuine production-atlas root manifest at hard combined
   cap 50,000, supplied by the owner and frozen as canonical state bytes before
   either result is observed.

No current cited artifact identifies the production Atlas-50k root list; the
old A5 source gate explicitly says its frozen opening-atlas family is absent.
Human-160 MUST NOT be silently renamed “atlas.” Until the owner supplies and
freezes Atlas-50k, default promotion is blocked by GAP C23.

Atlas-50k should be the complete pre-existing production atlas. If cost
requires a sample, the owner must pre-register the source snapshot,
stratification fields, sample size, and random seed without consulting
Group-2 eligibility or either arm's result. The manifest publishes included
and excluded canonical root digests; no post-result replacement is allowed.

The residue instrument's **Human160-50k** roots run at cap 50,000 as a
mandatory deep diagnostic, and at cap 2,000 as a bridge. They carry the
40.46% comparison but cannot substitute for Atlas-50k. Running all 6,462
roots at 50k is encouraged and may be pre-registered as a secondary cell, but
it cannot replace either promotion cell after freeze.

Current source-file pins to carry into the canonical manifest are:

| Source | Roots | SHA-256 |
|---|---:|---|
| <code>scripts/tss_harness/sets/selfplay_v1.jsonl</code> | 3,255 | <code>d8b4256408dfdabf71a90d3653962160bcc05ec66bba580dd6379149d998b708</code> |
| <code>scripts/tss_harness/sets/human_v1.jsonl</code> | 2,720 | <code>5784defe2531db55360e9860ddddc9b89b148547b16a0c970ff7d83f407c66b6</code> |
| <code>scripts/tss_harness/sets/puzzle_v3.jsonl</code> | 468 | <code>12b79c6ea132b8d0caa3c2a9108d5830039cd407b2e774670b59a144ea3495e7</code> |
| <code>packages/hexfield_eq/rust/corpus/forcing_corpus_moves.txt</code> | 19 | <code>89f16724483756ec8e41ba4a03009747ebb4760473a1f4bda75121e1c261f047</code> |
| sibling residue <code>human_positions.txt</code> | 160 | <code>d6b629b99084575d83e898ed90b7df3f4e779acaad9c0e88d5275a6eee57b046</code> |

File digests are not enough: freeze canonical parsed state bytes so parser
drift cannot silently change a root.

Both primary arms use <code>goal=both</code>, unbounded semantic horizon unless the
frozen production resolver says otherwise, <code>wide=true</code>,
<code>dual_pass=true</code>, <code>loss_reserve=0</code>, and
<code>zone=false</code>, matching the current production comparison except
for the hard cap. Dedicated Loss strata use the same frozen engine settings.
Any different deep production profile must be named
and frozen before evaluation, not selected after seeing results.

### 6.3 Arms and accounting

Run paired roots from fresh caches:

- **Control:** <code>Verify</code>, exact Full search domain, Group2V1
  materialization and strict verifier enabled.
- **Treatment:** <code>Consume</code>, including all speculative closure,
  local Full fallback, and strict verification inside the same hard cap.
- **Legacy control:** <code>Off</code>, used for search/status, node-ledger,
  legacy-field, and canonical legacy-certificate parity (subject only to the
  declared build/mode stamp exception above), not to hide materializer cost
  from the primary comparison.

The economic node counter is exact <code>SolveStats.nodes</code>, the same
ledger enforced by the hard cap. It includes the one examined root plus every
charged primal, dual, speculative, cutoff-reopen, and fallback expansion. A
cutoff expansion is charged before the depth test and again if reopened and
re-expanded. Report raw <code>search.expansions</code> separately for
diagnosis; do not substitute it for the cap/NPD numerator. Derivation,
verification, memory, and scheduling work are not fabricated as solver nodes;
they have separate work counters and are fully included in CPU and wall
economics.

A decision is a Win or Loss whose certificate passes the sealed strict
verifier. Unknown, verifier-rejected candidates, and unmaterialized PN states
are not decisions.

Let <code>D0</code> be the frozen set of roots that Control strictly verifies
as Win or Loss. The correctness gate first requires Treatment to decide every
member of <code>D0</code> with identical status. The primary denominator is
therefore fixed before treatment economics are interpreted:

~~~text
NPD_X_fixed = sum(SolveStats.nodes over every manifest root in arm X)
              / |D0|

reduction = 1 - NPD_Consume_fixed / NPD_Verify_fixed
~~~

<code>|D0|</code> MUST be nonzero; a zero-decision cell is economically
unevaluable and remains flagged off. Treatment-only decisions cannot improve
the denominator, but their real early-stop or extra-work effect remains in the
all-root numerator. Thus they may affect compute honestly, while being
reported as coverage rather than denominator credit. Also report conventional
arm-specific NPD and nodes restricted to <code>D0</code> as secondary
diagnostics. Do not average per-cohort or per-root ratios. Report fixed-root
totals, each named cohort, Win/Loss/Unknown classes, newly decided roots, local
fallback roots, and the consumed-site distribution. Use paired cluster
bootstrap over root pairs, with campaign-fixed seed and method, to compute the
95% lower confidence bound for the fixed-denominator aggregate ratio. Raw
deterministic counts remain authoritative.

### 6.4 Correctness gate: KILL

Coverage is pointwise, not aggregate:
<code>Decided_Off subseteq Decided_FullControl subseteq Decided_Consume</code>,
and status is equal on every inherited member. One added decision cannot
compensate for one lost decision at either seam. FullControl qualification is
read before Consume economics.

Any one of the following kills Consume promotion and blocks further economics
interpretation:

- an Off verified root is lost or changes status in FullControl at the same
  hard cap;
- a FullControl/Verify verified root is Unknown under treatment at the same
  hard cap;
- treatment and control produce different Win/Loss status for a root;
- any strict verifier false accept, any globally Closed producer candidate
  rejected by the strict verifier, panic, timeout misclassified as truth, or
  nonzero verifier failure;
- any gated decision fails cold Full cross-verification at 50k or the
  pre-registered exhaustive/adjudication ceiling;
- a certificate depends on search state, a shared semantic helper, an
  unpinned authority, or certificate-selected policy;
- a cutoff, resource failure, derivation failure, overflow, or cache entry
  creates a hard result;
- any must-reject or must-equal test in section 5 fails;
- any repeated-run determinism requirement fails; or
- actual work accounting exceeds a cap without being recorded.

Every treatment-only decision and every decision containing a consumed
ordinary G2 node is cold-cross-verified FullControl at 50k, or at a higher
pre-registered ceiling when 50k remains Unknown. A still-unresolved
treatment-only decision blocks promotion; it is not counted as corroborated.

Correctness KILL means disable the feature and investigate. It cannot be
converted into an economics miss or waived by aggregate parity.

### 6.5 Economics gate: target and floor

The pre-registered nodes/decision target is:

- **target:** at least 10% reduction in each primary operating point; and
- **hard floor:** the paired 95% lower confidence bound is at least 5% in
  each primary operating point.

Both conditions are required. Missing either condition is
<code>KEEP-FLAGGED-OFF</code>, not a correctness failure.

The rationale is deliberately conservative:

- 10% matches the original design's semantic-materiality threshold and is a
  meaningful return for a new search state machine plus a duplicated theorem
  implementation.
- 5% is the smallest deep-point saving accepted as robust enough to pay for
  maintenance and tail risk. A smaller point gain is too easy to erase with
  verifier, derivation, memory, or fallback overhead.
- The 40.46% number is a wall-time ceiling for the entire unforced-defender
  generation block, not a node ceiling. It does not mathematically imply any
  node reduction.
- The observed 0.56% to 1.64% cap-500 gate-bearing-root rate concerns current
  forced certificate gates and cannot be substituted for deep unforced
  consume prevalence. It does, however, demand cost-weighted deep evidence
  rather than extrapolation from the 19-versus-886 fixture.

Secondary economics requirements are:

- total wall divided by the same fixed <code>|D0|</code> denominator has a
  positive paired reduction at both primary cells and at least 5% at
  Atlas-50k;
- peak memory is no worse than 1.10 times control;
- local fallback rate, derivation wall, verifier wall, and closure-plan bytes are all
  reported and included; and
- in **each** of Labeling-2k, Atlas-50k, and mandatory Human160-50k, at least
  30 distinct <code>(root_digest, occurrence_path)</code> consumed unforced
  occurrences across at least 10 distinct roots survive strict verification,
  so one operating-point claim cannot be carried by one repeated fixture.

Wall gates use three clean repetitions of the same binary and manifest in
balanced AB/BA order. Publish every run and the predeclared median/paired
summary; root bootstrap does not estimate machine noise. Until
schedule-independent canonical behavior is proved, promotion and production
both run one worker across search, closure, materialization, finalization, and
relevant hash traversal. Merely pinning an arbitrary parallel scheduler is not
determinism.

Human160-50k must independently show at least 5% wall reduction and at least
5% fixed-denominator node reduction so the residue claim and the actual atlas
claim cannot mask one another.

Missing a secondary economics requirement is also
<code>KEEP-FLAGGED-OFF</code>. The flag stays available only for further
measurement; it does not become the default.

### 6.6 Post-landing residue obligation

After implementation lands but before promotion, rerun the
<code>e11c393d</code> residue instrument, rebased without changing its
category definitions, on the exact human-160/50k manifest and the paired
control.

The report MUST show:

- absolute wall and share for unforced-defender generation;
- eligible, Open, LocallyClosed, globally Closed, Full-fallback, and cutoff
  subcategories;
- full legal versus consumed edge counts, cost-weighted rather than only
  event-weighted;
- producer derivation, verifier, materialization, TT, and closure-memory wall;
- unioned dominated expansion intervals so nested sites are not double
  counted; and
- reconciliation from category totals to end-to-end wall and node changes.

Promotion requires the absolute unforced-defender block to shrink consistently
with the observed node saving. A lower percentage caused only by growth
elsewhere is failure. A node win with no deep wall win is economics failure
and remains flagged off.

## 7. Honest kill analysis and step zero

### 7.1 Strongest case against building

The case for stopping now is substantial:

1. The 19-versus-886 fixture proves a possible local fanout ratio, not
   frequency, ancestry dominance, or avoided work. Most of those 886 replies
   may never be expensive under PN ordering.
2. Current cap-500 firing is about one percent of roots and measures forced
   post-search FHW gates, not unforced consume closure. It offers no evidence
   that deep eligible sites dominate nodes.
3. The 40.46% residue category contains all directly timed
   unforced-defender generation. Eligibility, plan-complete closure, and
   Exact/FC support may cover only a small fraction. It is not evidence that
   descendant search work is avoidable.
4. The current native wide engine does not enumerate the proposed full
   unforced AND baseline at all. Adding both Uniform and gated arms expands
   scope before demonstrating that the target route can exploit them.
5. Unknown roots can spend scarce cap on a closure that later falls back.
   Existing narrow clean-rerun observations show near-doubling examples and
   warn against adding any hidden outside-cap rescue.
6. Consumption turns harmless gate-selection nondeterminism, TT context, and
   reopen bookkeeping into cap and coverage hazards.
7. Removing the shared-helper risk requires maintaining two implementations
   of complicated role, clock, window, zone, and row logic. The larger trusted
   review surface may not be justified by a single-digit gain.
8. A bug in the producer should be caught by the verifier, but then costs
   coverage and fallback; a correlated verifier bug threatens soundness. The
   cost of ongoing hostile review is real.

Prevalence killed an earlier adoption case; this is not the hostile review's
later FC-positive advisory, which was discharged. The same economic outcome
is plausible here even at 50k: a large category-level wall share can coexist
with very few plan-complete, theorem-admitted, cost-dominant sites.

### 7.2 Step zero: FullControl root-support upper bound

There is no rigorous pre-FullControl counterfactual in the current native-wide
trace: it refuses the unforced node before producing Full children, and a
different round3 trace does not contain every state a future FullControl
scheduler may reach. Instrumenting those existing hooks for local eligibility,
<code>|L|</code>, and <code>|Seed|</code> is useful heuristic triage only. It
MUST NOT be called a mathematical upper bound or used alone to kill the build.

The cheapest **decisive** shadow comes immediately after the independently
useful FullControl baseline exists and before Open/Closed Consume scheduling
or a second theorem implementation is built. This preliminary
<code>FullControlShadow</code> is search-only/non-minting: shared Group-2
helpers may classify telemetry but cannot emit acceptance evidence or a hard
Group-2 certificate. It is not promotion-grade <code>Verify</code>. On exact Labeling-2k,
Human160-50k, and owner-frozen Atlas-50k, a behavior-preserving FullControl
shadow records only:

- exact node-local eligibility and first failure at the actual future Consume
  hook;
- whether each root reaches at least one eligible **or indeterminate**
  occurrence; telemetry work/memory/overflow or any incomplete classification
  is indeterminate, never a negative;
- <code>|L|</code>, <code>|Seed|</code>, and root/occurrence identities; and
- the root's total <code>SolveStats.nodes</code> and end-to-end wall.

It leaves children, order, node ledger, status, every legacy result field, and
canonical legacy certificate bytes identical to shadow-off; declared
build/mode stamps are compared separately, and no <code>S*</code> is derived.
Let <code>E</code> be the roots
that reach at least one eligible or indeterminate occurrence. With
deterministic identical prefixes, a root outside <code>E</code> has only exact
negative classifications and no point at which Consume can diverge. Even
granting zero cost to every root in <code>E</code>, the maximum possible
reductions are:

~~~text
U_nodes = sum_{r in E} FullControl SolveStats.nodes(r)
          / sum_all_roots FullControl SolveStats.nodes(r)

U_wall  = sum_{r in E} matched shadow-off FullControl wall(r)
          / sum_all_roots matched shadow-off FullControl wall(r)
~~~

This deliberately absurd “eligible roots become free” bound includes all
possible downstream effects and is therefore safe for killing. Node identity
must be exact and <code>U_nodes</code> uses shadow-off
<code>SolveStats.nodes</code>. For wall, derive <code>E</code> from shadow-on
but take per-root times from three matched, balanced-order shadow-off runs so
instrument overhead on non-E roots cannot shrink the ratio. Freeze a
one-sided 95% upper-confidence procedure and any conservative A/A overhead
correction before running. If exact <code>U_nodes &lt; 5%</code> at
Labeling-2k, Atlas-50k, or mandatory Human160-50k, stop. For wall, stop only if
the one-sided 95% **upper** bound on <code>U_wall</code> is below 5% at
Atlas-50k or Human160-50k. The bound cannot promote the feature.

Only if that screen passes, run a **plan-complete
FullControlShadow-PC** before Consume scheduling. It may estimate provisional
<code>S*</code>, descendant admissibility, exact whole-tree contexts, closure
generations, fallback causes, plan bytes, and omitted-reply intervals from
complete Full plans without changing FullControl search. Shared helpers may
be used only for this non-minting economic telemetry. Promotion-grade
<code>Verify</code> still waits for the independent producer. This shadow is
an estimate, not a proof of savings; unresolved sites receive maximally
optimistic treatment.

The 40.46% residue remains a separate direct-block check: saving 5% total wall
solely by deleting that measured generation block would require eliminating
at least 12.36% of it at zero overhead
(<code>5 / 40.46 = 12.36%</code>). Missing that share kills the
direct-generation explanation, but not the whole Consume proposal if the
root-support bound leaves room for downstream savings.

Passing step zero does not authorize Consume. It only shows that a rigorous
economics ceiling has not already killed it.

## 8. Numbered GAPS and fail-closed defaults

The fresh hostile review attacks this list first. A gap is closed only by
theorem, independent implementation evidence, or a frozen test/report—not by
removing it from prose.

This list carries forward all thirteen gaps from the original extension
design: imported lambda-one/authority (C13), wire format (C18), economics
(C02/C19), broader mixed histories (C15), scalar clocks and slack (C14),
compact-T6 scope (C15), DAG completeness (C03/C16), resource progress (C12),
engine/formal correspondence (C13), digest limits (C17), alternative frozen
plans (C08), and stale empirical pins (C19/C23). The consume-specific gaps are
added rather than replacing them.

1. **GAP C01 — target full-unforced route.** Native wide PN currently
   immediately refutes non-implicit unforced defender nodes, while the narrow
   fallback may use a legacy zone. Neither is the exact <code>Full(P)</code>
   specified here. Section 3.2.1 normatively chooses lazy nested
   single-placement Universals, including terminal FirstStone and dynamic
   SecondStone legality. **Default:** no Consume A/B until that Uniform
   FullControl exists and passes exhaustive pair-phase tests.

2. **GAP C02 — deep eligible prevalence.** No current report measures
   plan-complete unforced <code>S*</code> sites at cap 2k or 50k. Forced-gate
   firing cannot answer it. **Default:** run step zero; if its optimistic
   upper bound misses the floor, kill the build.

3. **GAP C03 — context-sufficient closure identity.** The exact minimal set
   of incoming role/window obligations needed to share a gated transposition
   is not proved, and ancestor growth can change context within one occurrence.
   **Default:** occurrence-scope every G2 entry, run the exact whole-tree epoch
   pass, permanently mark the canonical ancestor-minimal mismatched paths Full
   for that solve, unfold every cert occurrence, and allow no proof-plan DAG
   sharing.

4. **GAP C04 — native cutoff/reopen implementation.** Current wide nodes have
   no Open/Closed closure state, and existing reopen tests do not cover an
   omitted reply becoming required. **Default:** Open is never Proven,
   cached, or materialized; unresolved final closure uses remaining-budget
   Full or is Unknown.

5. **GAP C05 — deterministic consumed selection.** Current post-search gate
   counts vary with parallel materialization/hash order. **Default:** Consume
   uses one thread in evaluation and production until schedule-independent
   canonical selection is demonstrated byte-identical and hostile-reviewed.

6. **GAP C06 — independent producer.** Current production finder and verifier
   share semantic helpers. **Default:** shared code may run only Off, Shadow,
   or the historical EmitLegacy regression lane. Promotion Verify/FullControl
   and Consume both use the independent producer; shared semantic code cannot
   control an edge or produce their acceptance evidence.

7. **GAP C07 — NonFrontierCovered support.** Theory material exists beyond
   the presently admitted Exact/FC verifier class, but no enabled
   implementation/review admits a NonFC FHW edge. **Default:** at a forced
   site, use the independently reconstructed compact-T6 seam when eligible,
   otherwise Full/Unknown; verifier rejects every NonFC <code>FhwGateV1</code>.

8. **GAP C08 — alternative frozen plans.** Closure completeness under a
   different valid child proof is not established. **Default:** deterministic
   first complete plan is immutable; if it cannot close, use Full. Never
   reselect to force acceptance.

9. **GAP C09 — dual and finite-cap coverage.** Consume ordering and fallback
   can spend a hard cap differently from FullControl, and current dual
   allowance is dynamic leftover rather than a fixed second budget.
   **Default:** one combined ledger, remaining-budget local fallback, no
   outside-cap rescue, dedicated Loss tests, and any lost FullControl decision
   is KILL. No work is hidden or refunded.

10. **GAP C10 — cache and resume persistence.** Current wide local keys and
    resume bindings omit Consume semantics; persistent caches intentionally
    reject extension nodes. **Default:** fresh per-leg arenas, explicit keys,
    mode-toggle clears, and no persistent Group-2 fragments.

11. **GAP C11 — combined mixed/reopen/TT hostile fixture.** Existing tests do
    not combine a real gate-plus-compact parent, descendant reopen, stronger
    incoming demand, appended ordinary-G2 edge, and forced TT collision.
    **Default:** only the exact reviewed seam is admitted, occurrence-local;
    CE23 is mandatory and its absence blocks promotion.

12. **GAP C12 — resource and memory progress.** No bound is measured for
    closure-owned frozen plans, full legal thunks, independent derivation, or
    unfolded mixed certs at deep caps. **Default:** cap failure opens Full or
    yields Unknown; it never permits omission.

13. **GAP C13 — imported lambda-one and engine/formal correspondence.** The
    original design's authority and direct-mask checks reduce reliance on an
    imported lambda-one claim, but correspondence between engine primitives
    and the paper model remains an external assumption. **Default:** pin
    authority, reconstruct directly in the verifier, reject drift, and retain
    exhaustive small-board/D6 tests.

14. **GAP C14 — scalar clocks and slack pressure.** The existing proof does
    not debit scalar <code>B</code>, LOSS remainder, horizon, or escape floors,
    and <code>k&lt;b</code> supplies no generic FHW debit. **Default:** preserve
    every full scalar recurrence and floor exactly; no extra pruning claim.

15. **GAP C15 — broader mixed histories.** The amendment discharges only the
    compact exact <code>tau=b</code> T6 seam. D17, SR, commutation, legacy
    zones, and arbitrary mixing remain unproved. **Default:** reject those
    fields and transitions.

16. **GAP C16 — DAG and whole-tree completeness.** Search transpositions can
    merge a state, while Group-2 certificates require a tree and evidence can
    be occurrence- and ancestor-dependent. **Default:** no G2 closure sharing;
    exact epoch stabilization, then unfold and independently rederive every
    occurrence.

17. **GAP C17 — digest meaning.** A matching child-plan or summary digest
    establishes byte identity, not theorem truth. **Default:** verifier derives
    every semantic value before comparing its independently encoded digest.

18. **GAP C18 — wire format and unknown versions.** No public untrusted wire
   parser for the new runtime state is designed here. **Default:** in-memory
   certificate use only; bump the immutable run-wide
   <code>TSS_CERT_VERSION</code> to v3 before search; unknown schema/tag/field
   rejects; search state is never serialized. Any dynamic emitted-certificate
   class is a distinct post-materialization telemetry field and never a TT or
   resume key.

19. **GAP C19 — empirical pins and causality.** The 6,462 cap-500 results are
   current, but deep manifests are not yet frozen. The 40.46% wall share is
   not a node share, and the roughly one-percent forced-gate root rate is not
   unforced consume prevalence. **Default:** make no gain forecast and no
   promotion decision until static shadow, plan-complete shadow, frozen deep
   A/B, and residue reconciliation.

20. **GAP C20 — fallback coverage versus economics.** Local Full fallback can
    change capped order and consume the remaining allowance after speculative
    work. **Default:** use the same hard cap in both arms, count all work, KILL
    on any lost FullControl decision, and otherwise keep the feature flagged
    off unless it clears every economic floor.

21. **GAP C21 — materialization independence from search state.** Current
    post-search materialization can reconstruct a proof, but no consumed native
    closure yet demonstrates that its frozen plan unfolds into a standalone
    mixed certificate. **Default:** a Closed PN root is only a candidate; if
    root-plus-bytes strict verification cannot reproduce it, discard it and
    use remaining-budget Full or return Unknown; the campaign is KILL.

22. **GAP C22 — post-landing residue category stability.** Rebased
    instrumentation could accidentally move work between categories and make
    the 40.46% comparison meaningless. **Default:** retain old category
    definitions, publish reconciliation, and treat unexplained drift as an
    economics failure.

23. **GAP C23 — production Atlas-50k manifest.** Human-160 is a residue
    diagnostic, not an identified production atlas, and the earlier A5 source
    gate records the opening-atlas manifest as absent. **Default:** the owner
    must supply and freeze canonical Atlas-50k roots before either arm runs;
    Human160 cannot substitute, and default promotion remains blocked.

## 9. Owner gate and required next action

The next action is a **fresh hostile review of this design**, beginning with
section 8 and the counterexamples in section 5. No source implementation is
authorized before that review and the owner's disposition of every
fail-closed default.

If the review passes, the owner may authorize only the independently useful
FullControl baseline plus non-minting FullControlShadow instrumentation. The
root-support kill screen then runs. A passing screen permits the
plan-complete non-minting shadow; neither shadow is Consume or
promotion-grade Verify.

Only if those economics screens pass may the owner separately authorize the
independent producer and Consume state machine. That implementation receives
its own hostile review and remains flagged off until the full correctness,
promotion, and post-landing residue gates pass.

## 10. Evidence read for this design

- <code>CODEX_BRIEF.md</code>
- <code>.gate/G2V2_ACCEPT_REPORT.md</code>
- <code>.gate/G2V2_ADOPTION_REPORT.md</code>
- <code>.gate/G2V2_HOSTILE_REVIEW.md</code>
- <code>.gate/G2V2_WIDEPN_REPORT.md</code>
- <code>.gate/G2V2_MIXED_REPORT.md</code>
- <code>.gate/G2_FINDER_CLOSURE_REPORT.md</code>
- <code>.gate/G2V2_REPORT.md</code>
- sibling <code>group2-zones/.codex-g2-resolve/DESIGN_G2_CERT_EXTENSION.md</code>
- sibling <code>group2-zones/.codex-g2-resolve/DESIGN_AMENDMENT_R1_R2.md</code>
- sibling <code>group2-zones/.codex-g2-resolve/HOSTILE_REVIEW_1.md</code>
- sibling <code>group2-zones/PROOF_TSS_ZONES_FHW.md</code> and review 2
- <code>packages/hexfield_eq/rust/src/tss_solver.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_verify.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_verify_group2.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_core.rs</code>
- residue commit <code>e11c393d</code> and sibling human-160 artifacts
