# Proof-carrying forced-tree solver: soundness argument

This document is the proof obligation for the Stage-3 deliverable specified in
`docs/TSS_SOLVER_SPEC.md` and its round-2 optimization specification in
`docs/TSS_SOLVER_OPT_SPEC.md`.  Code mappings prefer function names because the
optimization work moves hot-path line numbers.  `SolveCaps` remains
source-compatible.  The only seam addition is the `SolveGoal` request enum;
requested-side specialization is exposed by the additive
`TssSolver::solve_goal` companion API.  Making the goal a new required
`SolveCaps` field would force a change to the existing literal in forbidden
`tree.rs`.

## 1. Scope, notation, and result meaning

Fix a reachable engine state `s` and a player identity `C`, called the
**claimant**.  A proof node is existential when
`s.current_player() == C` and universal otherwise.  This identity rule, rather
than placement-depth parity, is essential because `FirstStone -> SecondStone`
does not change the player (`hexo_engine/rust/src/state.rs:324`), while
`SecondStone -> FirstStone` does (`state.rs:330`).

`Win` means that the root side to move is the claimant.  `Loss` means that the
claimant is the other player.  `Unknown` carries no game value.  The claimant
mapping is implemented by `TssSolver::solve_goal`, `prove_for`, and
`status_for_claimant`; `TssVerifier::verify` derives the mapping independently.

The shared lambda-one facts are:

- `B = 2` at `FirstStone`, and `B = 1` at `Opening`/`SecondStone`
  (`threats_shared.rs:49`);
- a count-5 is win-now for either budget, and a count-4 is win-now only at
  `B = 2` (`threats_shared.rs:140`);
- `forced_loss` means no own win-now and no hitting set of size at most `B`
  (`threats_shared.rs:75`);
- the hitting-set search is exhaustive for the only possible budgets, one and
  two (`threats_shared.rs:98`).

`prove_for` clones the supplied root once.  Every hot-path descendant is entered
with `apply_with_delta` and left with `undo` in `prove_choice` and
`prove_universal`; engine support is in `HexoState::apply_with_delta` and
`HexoState::undo`.

## 2. Search scheme

The implementation is a deterministic, proof-number-ordered depth-first
AND/OR proof constructor (`SearchContext::prove`).  It is an argued-equivalent
best-first scheme rather than a numerical-threshold df-pn implementation:

1. Every eligible OR child receives a static proof-cost tuple derived from the
   current `WindowStore`: immediate proof, fully forced reply, same-turn build,
   half-forced reply, or quiet reply.  `ordered_threat_creating_moves` obtains
   these features without applying and re-analyzing every candidate.
2. D6-canonical coordinates break remaining tactical ties
   (`canonical_frame`/`canonical_coord_key`).  Universal moves retain this
   canonical tie order even when the hitting partition is empty, preserving
   D6-covariant cap behavior.
3. The frontier order is the lexicographic sequence of those costs, with every
   descendant of a lower-cost OR child ordered before the next OR sibling.
   Depth-first traversal therefore always expands the least frontier element
   under this fixed best-first order.
4. An AND node has no selective proof: all non-dispatched children must prove.
   `prove_universal` materializes them lazily, with hitting-universe moves
   first when that is a real tactical partition.  Early termination can only
   abandon the universal proof; a returned universal certificate contains all
   required explicit children.

This scheme and df-pn recognize the same completed proof trees over the same
restricted OR/full universal graph when resources are unbounded.  Under a cap,
the order can change `Unknown` into a discovered proof or vice versa, but it
cannot change the validity of a completed proof.  No heuristic number is ever
converted to `Win` or `Loss`.

The solver first checks a terminal/lambda-one root (`immediate_winner`).  For a
deeper result, the default `DeepSolve::solve` requests both sides and gives
deterministic disjoint portions of the node budget to a primal proof and an
independent dual proof.  The additive requested-side entry point instead gives
the remaining budget to the sole requested claimant.  A failed primal search
is never reused as a dual proof.  A bounded cache owned by `TssSolver` may reuse
only completed positive proof fragments from earlier solves; the local
per-attempt TT remains separate.

## 3. Certificate format and independent verification

The wire format is defined in the verifier module, so the verifier imports no
solver code:

- `RootBinding` (`tss_verify.rs:38`) stores sorted occupancy, aligned owners,
  current player, exact phase including the `SecondStone.first` witness,
  placement count, and terminal fact;
- `Terminal` and `Lambda1` are factual leaves;
- `Choice { move, child }` is one claimant continuation;
- `Universal { edges, implicit_dispatch }` lists searched opponent moves and,
  only at the L1 boundary, represents the non-hitting complement implicitly
  (`tss_verify.rs:85`);
- `TssCertificate` binds a claimant and arena root to the exact root position
  (`tss_verify.rs:103`).

The fixed acceptance bounds are 100,000 nodes, 1,000,000 explicit universal
edges, depth 256, and 1,000,000 root stones (`tss_verify.rs:17`-`23`).  Thus a
certificate is `O(root stones + nodes + edges)` with fixed maxima.  The
verifier's shared-DAG memo has its own 64 MiB hard ceiling
(`tss_verify.rs:27`, `tss_verify.rs:200`); ordinary tree nodes are not memoized.

Before replay, the verifier rejects a wrong root, wrong claimant/status,
`Unknown`, invalid IDs, duplicate moves, excess bounds, cycles in any arena
component, and unreachable nodes (`tss_verify.rs:114`, `tss_verify.rs:461`,
`tss_verify.rs:523`).  Replay then uses only engine application/legal
enumeration plus `threats_shared::analyze` (`tss_verify.rs:272`).

### Certificate soundness theorem

**Theorem.** If `TssVerifier::verify(s, cert, status)` returns true, the player
identified by `status` has a winning strategy from `s`.

**Proof by induction over the accepted acyclic certificate graph.**

- `Terminal`: the engine terminal winner is the claimant.
- `Lambda1`: the independent checker rejects Opening and terminal misuse, runs
  lambda-one analysis, maps its sign by player identity, and accepts only when
  that winner is the claimant (`tss_verify.rs:429`).
- `Choice`: it is the claimant's turn; the certified legal move is replayed;
  the induction hypothesis supplies a claimant win in the child
  (`tss_verify.rs:272`).
- `Universal`: it is the opponent's turn.  Every legal move is enumerated.
  Every listed move has an inductively winning child.  Every unlisted move is
  accepted only under L1 and is applied and independently lambda-one-refuted.
  Hence all opponent choices lead to a claimant win (`tss_verify.rs:325`).

The graph is acyclic, so the induction is well founded.  Root binding and
claimant mapping then yield exactly the claimed side-to-move `Win` or `Loss`.

## 4. Required lemmas

### L1 — instant dispatch

Suppose a defender node is post-opening, has live claimant threats, has no own
win-now, and its minimum hitting-set size is `k = B`.  “Non-hitting” here means
outside the **union** of all live claimant-window empties; it does not mean a
cell that hits one window but fails to cover the whole family.

For `B = 2`, the node is `FirstStone`.  A non-hitting placement leaves every
threat window untouched and leaves the same defender at `SecondStone`, now with
budget one.  The unchanged family still needs two hits, so the child is a
lambda-one forced loss.  The placement cannot secretly win: absence of parent
own-win-now excludes a defender count-4 or count-5; adding one stone can at most
turn count-3 into count-4, which is not win-now at `SecondStone`.

For `B = 1` at `SecondStone`, a non-hitting placement ends the turn without
touching the claimant threat.  The claimant begins `FirstStone` with budget two
and completes the live count-4/count-5 window.  The child lambda-one verdict is
therefore a claimant win.  The other `B = 1` phase, `Opening`, cannot satisfy
the premise in a reachable state because the board is empty; both solver and
verifier explicitly reject Opening dispatch.

`prove_universal` checks the parent boundary from the node's single immutable
`ThreatAnalysis`, and `hitting_universe` collects the full union.  The producer
does not enumerate or apply the non-hitting complement: the preceding argument
proves that whole class at once.  This is a producer-side proof compression,
not a verifier shortcut.  `dispatch_boundary` independently reconstructs the
boundary/universe, and `verify_universal` enumerates, applies, and lambda-one
checks every omitted legal move.  Thus every accepted certificate remains
stapled to its complete complement even though the optimized producer no
longer pays to staple-check it while searching.

Tests: `deep_win_contains_verified_universal_coverage`,
`spare_stone_counter_threat_cannot_be_implicitly_dispatched`, and
`certificate_mutations_are_rejected`.

### L2 — OR restriction safety

At a claimant node the solver considers only empty cells extending an active
claimant length-six window with at least three stones
(`threat_creating_moves`).
The selected placement therefore creates a >=4 threat (or wins).  A `Choice`
certificate needs only one legal winning continuation.  Omitting other claimant
moves can hide a different win, but cannot invalidate the replayed selected
win.  Exhaustion of this restricted set returns no proof, never `Loss`
(`prove_choice`).  The verifier does not trust the generator; `verify_node`
replays the chosen legal move.

Tests: `deep_win_contains_verified_universal_coverage`, the seeded
differential, and `d6_status_and_certificate_replay_all_twelve_symmetries`.

### L3 — AND completeness

At a universal node there are exactly two cases (`prove_universal`):

- under L1, every hitting-universe move is explicit and every complement move
  is independently dispatchable;
- otherwise, including every `k < B` spare-stone node, the explicit list is the
  full engine legal list.

Under L1 the producer need not enumerate the complement because L1 proves the
whole class.  Outside L1, including every `k < B` spare-stone node, the engine's
entire deterministic legal list is consumed.  Hitting moves may be ordered
first, but children are materialized one at a time and every explicit child
must return a certificate before `alloc_node` creates the universal node.  A
capped, failed, or absent child returns `None` and poisons the universal
attempt; early exit therefore cannot produce a partial universal certificate.
`verify_universal` enumerates the legal set again, rejects duplicate/illegal
edges, rejects implicit coverage outside the L1 boundary, requires every
hitting move explicitly, and checks each complement move.  Therefore
`searched(node) union dispatch(node) = legal(node)` in every accepted proof.

Tests: `deep_win_contains_verified_universal_coverage`, the dropped-child
mutation in `certificate_mutations_are_rejected`,
`spare_stone_counter_threat_cannot_be_implicitly_dispatched`, and
`node_cap_only_moves_results_toward_unknown`.

### L4 — dual LOSS and identity perspective

A `Loss` certificate is not failure of the root player's attack.  It is a
separate winning certificate with claimant `root.current_player().other()`.
Recursive node kind is selected by `state.current_player() == claimant` in
`SearchContext::prove`, so
`FirstStone -> SecondStone` retains node kind and a completed turn changes it.
`TssVerifier::verify` independently requires the opponent claimant for `Loss`.

Tests: `root_lambda_loss_has_dual_certificate`,
`differential_forced_loss_after_first_defender_stone`, and the independent
`player_identity_is_fixed_across_two_stone_turn` test.

### L5 — Unknown monotonicity

The only certificate-producing local return is a successfully allocated
factual, choice, or complete-universal node.  Node/depth or certificate bounds
set the limit flag and return `None`; an OR may try another non-capped child,
while an AND immediately returns `None`.  The outer solve emits a hard status
only when the corresponding requested claimant attempt returns a complete
certificate.  Local TT entries contain proved IDs in that attempt's arena;
persistent entries contain self-contained completed positive-proof fragments.
Neither table stores failure, absence, `Unknown`, disproof numbers, or a
heuristic value.  A rejected or over-budget shared import is a miss, never a
verdict.  `SolveStats` is never read by a verdict path.

With a fixed initial cache state, missing work still moves an attempt only
toward `Unknown`.  A warmer cache can discover a valid proof under a cap that a
cold cache misses; round 2 explicitly permits that cache-history dependence.
It does not violate this lemma because the warm result comes from an earlier
complete positive proof, not from converting the cold failure into a value.

Tests: `zero_node_cap_is_unknown_and_certless`,
`node_cap_only_moves_results_toward_unknown`, local/shared TT cap modes, the
shared warm-versus-cold tests, and the typed seam's verifier gate in
`tss_core.rs`.

### L6 — cache identity

`PositionKey::from_state` contains sorted `(q,r,owner)` occupancy, current
player, exact phase with `SecondStone.first`, placement count, and terminal
winner/count.  The claimant is compared alongside the key in both the local
and persistent tables.  `PositionKey::stable_hash` selects a bucket only; every
value-bearing lookup additionally requires full-key and claimant equality.
`hexo_utils::StateHash` is never imported.

The omitted engine history and `last_turn` fields do not affect terminal
status, legal placement generation, window analysis, or any transition fact
used by a certificate.  Consequently equal `PositionKey`s are future-equivalent
for this solver and verifier.  The persistent cache is in-process only; no
proof survives a rules or binary change without reconstruction.

The solver has no D6 value cache.  D6 is used only for covariant tie ordering.
Certificates transformed for replay map every root/phase/move coordinate and
are checked against the transformed full root (`tss_verify.rs:582`,
`tss_verify.rs:608`).  Shared certificate arena nodes are reused by the
verifier only when its independently built full replay position is equal
(`tss_verify.rs:200`).

Tests: `full_key_rejects_forced_hash_collision`, the persistent forced-collision
and cross-solve contamination tests, TT off/tiny/large modes, exact
root/witness mutations in `certificate_mutations_are_rejected`, and
`d6_status_and_certificate_replay_all_twelve_symmetries`.

## 5. Optimization-by-optimization arguments

No epsilon, randomized choice, wall clock, neural score, history hash, or
unverified symmetry cache is present.

### O1. Root terminal/lambda-one fast path

`immediate_winner` emits exactly the same factual leaf that recursive search
would emit.  It avoids duplicating the root between primal and dual budgets;
the verifier recomputes the fact.  Cost changes, verdict semantics do not.

### O2. Threat-creating OR restriction

This is L2.  It deletes only existential alternatives.  Deletion can change a
would-be proof to `Unknown`, never create a proof.

### O3. Instant-dispatch certificate compression

This is L1/L3.  The complement is not stored cell-by-cell, but the verifier
reconstructs the full legal complement and checks every child.  Compression
does not weaken coverage.

### O4. Static proof-number initialization and move ordering

The tuple produced by `ordered_threat_creating_moves` changes only traversal
order.  Every OR child in the restricted generator remains present; every
required AND child remains present.  A cap can make a proof easier or harder to
find, but a returned tree has the same verifier obligations.  No priority or
derived feature enters a certificate fact.

### O5. D6-canonical tie ordering

`canonical_frame` selects the least full-position D6 image and compares tied
moves in that frame.  D6 maps legal placements, windows, owners, and the phase
witness bijectively.  Therefore it permutes equal-cost branches without adding
or removing one.  It improves cap covariance; certificate remapping is still
verified independently.

### O6. Make/unmake instead of descendant clones

All returns after a successful apply execute the matching LIFO undo.  This is
observationally identical to cloning the parent then applying the move because
the engine delta restores occupancy, legal/window stores, player, phase,
terminal state, last turn, and history.  Exact round trips on successful proof
and forced cap exit are tested by
`make_unmake_round_trips_on_proof_and_cap_exit`; the independent reference
repeats the check in `recursive_make_unmake_restores_exact_public_state`.

### O7. Direct-mapped bounded TT and replacement

The per-attempt local table reserves a fixed inline slot vector within its
allocated portion of `tt_bytes_cap`; cap zero or an undersized portion disables
it (`BoundedTt::new`).  A position key's actual `Vec` capacity plus a
conservative allocation charge is accounted.  `BoundedTt::insert` first
subtracts the replaced key, adds the new key, and skips the insertion if its
charged total would exceed the local allocation.  Direct replacement can turn
a future hit into a miss only.  A miss recomputes; it cannot authorize a
verdict.  A local hit is a previously proved node in the same arena and
requires L6 full equality.

The accounting formula and every tested cap are recomputed independently in
`tt_allocation_never_exceeds_cap`.

### O8. Stable hash bucket selection

FNV is not a proof identity.  Masking every hash to zero still cannot cross-hit
because full equality is mandatory.  Thus hash choice affects collision rate
and replacement cost only (`full_key_rejects_forced_hash_collision`).

### O9. Certificate arena, TT DAG reuse, and compaction

The arena permits an exact-key TT hit to reference an existing proved node.
Before emission, `compact_certificate` copies only nodes reachable from the
chosen proof and remaps IDs.  It detects a construction
cycle and respects fixed node/edge bounds.  The verifier rejects cycles and
orphans and, for a genuinely shared node, compares a full replay key.  These
operations remove garbage or share identical facts; they do not change a
node's logical obligation.

### O10. Deterministic default primal/dual budget split

When both sides are requested, the deterministic split prevents a failed
primal attack from starving all dual work while keeping total expansions within
`node_cap`.  Either half may miss a proof and return `Unknown`.  A hard result
still requires a complete certificate, so the split changes completeness under
cost only.  O17 specializes this allocation when the caller requests only one
side.

### O11. Independent reference alpha-beta exits and direct-line order

The oracle enumerates its own complete legal set (`tss_reference.rs:51`) and
uses ordinary three-valued minimax (`tss_reference.rs:137`).  At an existential
node, one proven win makes remaining children irrelevant; at an opponent node,
one proven loss does likewise.  Ordering direct line extensions first
(`tss_reference.rs:204`) changes only when such a decisive child is found.  If
no decisive child exists, all legal moves are still visited.

### O12. Producer-side L1 dispatch enumeration elimination

At the L1 boundary, `prove_universal` now materializes only
`hitting_universe(state, claimant)`.  It no longer enumerates every other legal
placement, applies it, invokes lambda-one, undoes it, and rediscovers the same
fact one move at a time.  This deletion is justified by L1 itself: after the
single parent analysis establishes post-opening, a live claimant threat, no
defender win-now, and `min_hitting_set == B`, every legal placement outside the
union leaves all claimant threat windows unchanged and is lambda-one-refuted.
The producer therefore replaces repeated instances of one proved implication
with one invocation of the implication's premise.

This optimization cannot turn an incomplete universal search into an accepted
universal proof.  The certificate records `implicit_dispatch = true`, and the
independent `verify_universal` still enumerates the complete engine legal set,
reconstructs the boundary with `dispatch_boundary`, requires every
hitting-universe move explicitly, and applies and lambda-one-checks every
omitted move.  If the producer misclassifies a boundary, the certificate is
rejected.  Caps can prevent an explicit hitting child from proving, in which
case `prove_universal` returns `None`; they cannot validate an omitted class.

### O13. WindowStore-derived candidate features replace OR probes

`threat_creating_moves` already obtains the exact restricted OR candidate set
from active claimant windows with count at least three.  Its `CandidateBatch`
now records, during that same window scan, existing claimant and defender
threat-window empties and the count-three windows each candidate would turn
into threats.  `WindowStore` iteration order is not assumed: aggregation is
order-invariant, minimum-hitting-set existence is set-valued, and the final
candidate coordinate-key sort is deterministic.  A placement changes only
windows containing its coordinate:
a claimant window gains that bit, while a defender window containing the bit
becomes blocked.  Thus `post_turn_reply_priority` can remove the placed cell
from the affected empty sets, detect an unblocked defender win-now, and invoke
`min_hitting_set_at_most_two` on exactly the child claimant-threat family.
The latter exhausts the only possible reply budgets, one and two, just as the
lambda-one analyzer does.

For reachable unresolved nodes that enter `prove_choice`, these set deltas
reconstruct the former nonterminal child priority tuple.  A candidate that
itself terminates is the deliberate edge exception: its actual recursive apply
proves a `Terminal` leaf, while secondary post-child ordering fields need not
model an unresolved child that does not exist.  Such a candidate remains in the
immediate-proof class.  The derived class and threat count replace the former
apply/analyze/undo ordering probe; they do not replace application of the
selected search move.  More importantly, even an ordering-feature defect could
only permute candidates:
`ordered_threat_creating_moves` neither adds a non-candidate nor removes a
candidate based on the derived priority.  Search still applies the chosen move,
recursive factual checks run on the actual child state, and the verifier
replays it.  Therefore these features can change which proof is found under a
cap, but cannot make any returned proof valid or invalid.

### O14. Lazy hitting-first universal materialization

Outside L1, `prove_universal` retains the full engine legal set required by L3
but constructs child proofs only as the loop reaches them.  When live claimant
threats provide a nonempty hitting universe, those moves are ordered first.
When that tactical partition is empty, no hitting-membership priority work is
semantically added: the Boolean priority component is identical for every move
and the sort reduces to D6-canonical coordinate order, preserving covariance
across transformed solves.  At L1, O12 handles the complement and only hitting
moves enter the explicit loop.

The evaluated variant that skipped `canonical_frame` for an empty hitting
partition was not retained: although branch order alone cannot falsify a
certificate, it changed cap-sensitive behavior across D6-equivalent roots.
The shipped optimization skips only substantive threat-priority work and keeps
the canonical tie order, as exercised by
`quiet_no_hitting_universal_is_d6_cap_stable`.

An early failed, capped, or opposing child returns `None` before a universal
arena node exists.  A successful `Universal` node is allocated only after all
members of the required explicit list have produced edges.  Hitting-first
ordering and lazy child construction therefore change cost and cap-sensitive
discovery only.  They cannot remove an edge from a returned complete
certificate, and `verify_universal` remains insensitive to producer order.

### O15. One threat analysis per nonterminal node

After exact local/shared proof lookup, `SearchContext::prove` handles the
terminal fact, then calls `threats::analyze` once for a newly examined
nonterminal state.  `winner_from_analysis` consumes that result for the
lambda-one leaf check, and `prove_universal` borrows the same immutable
analysis for its dispatch-boundary decision.  The state is not mutated between
creation of the analysis and either use.  Descendant mutation begins only after
the current node kind and move list have been selected, so no analysis is
carried across a state transition.  Opening still cannot mint a lambda-one
leaf.  Reusing a pure analysis result at the same state is observationally
identical to recomputing it and changes no certificate fact.

This claim is per newly expanded search node.  The solve-level
`immediate_winner` root fast path is an independent preclassification, and the
primal and dual attempts are intentionally independent searches; neither is a
second analysis inside one `SearchContext::prove` expansion.

### O16. Bounded persistent positive-proof fragments and atomic import

The persistent cache invariant is expressed as a proposition.  A resident
entry `(K, C, P)` means that self-contained fragment `P` is a finite winning
strategy for claimant identity `C` from the exact semantic position `K`.
`PositionKey::from_state` supplies `K`: sorted coordinates with owners, current
player, exact phase including the `SecondStone.first` witness, placement count,
and terminal winner/count.  The cache compares `C` separately.  Neither caps,
requested side, discovery order, nor the 64-bit bucket hash is part of the
proposition.

Every `SharedTtEntry` owns a `CachedProof`: a compact certificate fragment and
its root.  Its IDs refer only to nodes inside that fragment; it never retains a
`SearchContext` arena ID, a borrowed node, or a pointer into a previous solve.
Only a completed positive proof may be inserted.  Concretely, local search
first constructs factual leaves, claimant choices, and complete universal
nodes under L1--L5.  `SearchContext::remember_proof` is reached only after such
a node succeeds.  It promotes non-leaf choice/universal structure (standalone
factual leaves are deliberately cheaper to recompute), and
`CachedProof::from_arena_limited` then copies the promoted node's entire
reachable acyclic sub-DAG, including its leaves, remaps all IDs, and records the
fragment bounds.  The successful solve root is offered separately after final
compaction.  No failed or `Unknown` proof value, partial parent, or heuristic
cost has a persistent representation.  A completed positive descendant
discovered before its enclosing root later fails or hits a cap may persist: its
proposition is already complete by induction and does not depend on the
unfinished parent.  This establishes the invariant for a new entry from the
same induction used by the certificate soundness theorem, and
`SharedProofCache::insert` admits only that owned value.  Final result emission
independently uses `compact_certificate` to retain only the selected root proof
and offers that complete root through `CachedProof::from_compact`.  Internal
promotion's fixed node/edge limits affect only which descendant fragments are
retained; declining a larger internal fragment is a cache miss, not a result.

The insertion invariant is inductive over the exact point at which
`remember_proof` runs:

- a `Terminal` or `Lambda1` node is inserted only after its winner has been
  mapped to `C`;
- a `Choice` is inserted only after one applied legal move returned an existing
  proof for `C`;
- a `Universal` is inserted only after every required explicit child returned
  a proof and L1 justified precisely the implicit complement;
- a local-TT child already denotes a proved node in the same arena, while a
  shared-TT child first passed the full-key lookup and atomic import obligations
  below.

There is therefore no insertion call on the negative return paths of
`SearchContext::prove`.  Re-caching an imported fragment preserves, rather than
assumes, the same invariant.

`SharedProofCache::lookup_cloned` first uses `PositionKey::stable_hash` to
select a deterministic bucket, then requires equality of the stored hash, the
full `PositionKey`, and the claimant.  A forced 64-bit collision is therefore a
miss.  Equal keys are future-equivalent for every proof operation: board
ownership determines
windows and legal cells, current player and exact phase determine turn budget
and transitions, and terminal/count fields determine factual leaves.  The
history-bearing neural `StateHash` is neither necessary nor allowed.

`SearchContext::lookup_shared` repeats the exact key/claimant lookup before a
descendant fragment can become a local proof.
`SearchContext::import_cached_proof` preflights its root ID and stored acyclic
bounds, maximum path depth against
`MAX_SEARCH_DEPTH - current_depth`, reachable nodes against the remaining
certificate-node allowance, explicit edges against the remaining edge
allowance, and every checked `u32` ID remap.  Import builds a complete remapped
fragment before committing it to the local arena.  If any bounds or remap check
fails, no imported root becomes visible and lookup degrades to a miss or
`None`.  Thus a proof originally valid at cache-root depth zero cannot be
spliced into an over-depth or oversized certificate, and partial import cannot
authorize a parent.

Replacement preserves the invariant.  An entry owns its proof allocation, so
replacing it cannot leave a surviving dangling node.  Eviction, an undersized
cap, or refusal to admit an oversized fragment can only turn a later hit into a
miss.  Direct-bucket collisions likewise affect retention, never identity.
Warm cache history may change traversal, certificate shape, node telemetry, or
whether a proof is discovered before `node_cap`; it cannot change the truth of
any resident proposition.  A positive proof found under a larger earlier
budget remains a proof under a smaller current search budget.  A zero-node
request is still rejected before proof lookup.

The byte invariant covers the persistent slot storage, full-key capacities,
fragment-node capacity, every nested universal-edge capacity, and conservative
allocation charges.  `split_tt_cap` gives the local TT and persistent cache
disjoint portions of the one `tt_bytes_cap`; their simultaneous
charged resident total, and therefore combined `peak_tt_bytes`, never exceeds
that cap.  The cache's `current_bytes` charge and
`SearchContext::observe_tt_bytes` report the combined charge.  Admission
computes replacement cost before commit.  `SharedProofCache::reconfigure`
handles sustained reuse and cap shrinkage by deterministically evicting or
disabling entries rather than retaining memory charged under an earlier larger
cap.

This accounting ceiling is the retained TT/cache ceiling named by
`tt_bytes_cap`.  `lookup_cloned`'s owned import candidate and the local
certificate arena are transient proof-construction memory, separately bounded
by the fixed certificate node/edge limits; they are not a second retained cache
and are not reported as resident TT bytes.  Once imported, only the local arena
owns that clone, and it is dropped with the attempt.

Finally, cache construction is not part of the verifier's trust base.  Every
returned hard result contains a newly root-bound complete certificate, and
`TssVerifier::verify` independently replays cached and newly searched nodes
alike before `hard_value_from_verified` can mint a hard value.  The round-2
determinism contract is consequently conditional: for a fixed state, caps,
requested side, and initial cache state, traversal, replacement, result, and
resulting cache state are deterministic.  Different prior cache histories may
change discovery but never verdict validity.  The mutable handle is used
exclusively; no concurrent replacement or wall clock participates in a solve.

### O17. Requested-side budget specialization

The additive `TssSolver::solve_goal` companion API accepts
`SolveGoal::{Win, Loss, Both}`.  `DeepSolve::solve` remains the
source-compatible `Both` entry point.  In `Both`, O10's deterministic
primal/dual split is unchanged.  In `Win` or `Loss`, the sole requested
claimant receives the entire remaining node budget and the unrequested
independent attempt is skipped.  Root terminal/lambda-one facts are filtered by
the same request, so the fast path cannot return an unrequested status.

Skipping an attempt deletes a possible output; it does not reinterpret that
attempt's absence.  A requested hard result still requires a complete
certificate for the corresponding claimant, whose identity is also part of
every TT lookup.  Therefore specialization can turn a `Both`-mode `Unknown`
into a discovered requested proof by avoiding the half split, but cannot turn a
failed attack into `Loss` or cross-reuse a proof for the other claimant.

This is deliberately a companion API rather than a required field added to
`SolveCaps`: the existing caller in `tree.rs` uses a struct literal, and the
round-2 scope forbids editing that file.  Existing callers retain `Both`;
future caller-owned solver handles may opt into the specialized request without
changing the frozen consumption path.

### O18. Allocation-reused canonical-frame construction

`canonical_frame` still evaluates all twelve D6 images and chooses the first
image with the lexicographically least `(phase, sorted owned stones)` pair.
The implementation now allocates two stone buffers once, clears and refills the
candidate buffer for each image, and swaps it with the best buffer only when the
same comparison selects a new minimum.  Buffer capacity and allocation history
are not compared, and equal images retain the earlier symmetry exactly as
before.  Thus the selected frame and every downstream move-order key are
unchanged; only repeated temporary allocation is removed.  The all-twelve D6
certificate replay and quiet cap-stability tests exercise the preserved
selection contract.

## 6. Determinism and memory bounds

- There is no wall-clock read in solver or verifier code.  `Instant` appears
  only in the ignored `tss_bench_report` harness.
- `WindowStore` iteration order is never trusted.  Candidate and threat-feature
  aggregation is order-invariant and `threat_creating_moves` applies a final
  deterministic coordinate-key sort; `hitting_universe` likewise deduplicates
  and sorts.  Universal ties retain D6-canonical ordering.
- Every attempt allocates a fresh `SearchContext`, local TT, and certificate
  arena.  The caller-owned `TssSolver::shared_tt` persists bounded positive
  proofs across solves.  Consequently prior solves may affect discovery and
  telemetry.  For fixed state, caps, `SolveGoal`, and initial shared-cache
  state, the solve and resulting cache state are deterministic.
- Node count is at most `SolveCaps.node_cap`.  Search/certificate depth is at
  most 256.  Certificate nodes/edges/root and verifier memo have the fixed
  bounds in section 3.
- `split_tt_cap` bounds the simultaneously resident local and shared TT charge.
  `peak_tt_bytes` observes their sum, which never exceeds `tt_bytes_cap`;
  replacement and cap reconfiguration are deterministic.  Retained
  `PositionKey` and `CachedProof` allocations are charged, while temporary
  search keys remain ordinary search temporaries.

Cold configuration equality remains exercised by
`solver_configurations_are_deterministic_on_hard_leaf`.  The persistent-cache
tests compare solves begun from equivalent cache states and separately assert
that every hard warm result verifies; they do not incorrectly require a cold
and cache-mutating warm call to have identical telemetry or certificates.

## 7. Independent reference and harness map

The reference solver does not call engine legal enumeration, window storage,
threat analysis, or the solver generator.  It builds the radius-eight union in
a `BTreeSet` (`tss_reference.rs:51`), scans 6+ directly along the three axes
(`tss_reference.rs:97`), and holds the root player identity fixed through
minimax (`tss_reference.rs:137`).  Tests cross-check independent legal sets on
seeded states (`tss_reference.rs:283`), identity across phases
(`tss_reference.rs:332`), all axes (`tss_reference.rs:354`), and undo integrity
(`tss_reference.rs:390`).

Spec section 6 maps as follows:

| Requirement | Tests |
|---|---|
| Curated + seeded differential | `curated_differential_and_every_hard_certificate_verifies`; `differential_forced_loss_after_first_defender_stone`; `seeded_random_differential_covers_all_phases_and_dense_positions` |
| Every certificate + mutations | `root_lambda_loss_has_dual_certificate`; `deep_win_contains_verified_universal_coverage`; `certificate_mutations_are_rejected`; verifier graph/bound tests |
| Twelve D6 replays | `d6_status_and_certificate_replay_all_twelve_symmetries`; `quiet_no_hitting_universal_is_d6_cap_stable`; verifier terminal-remap tests |
| Local TT off/tiny/large/collision | `full_key_rejects_forced_hash_collision`; `tt_allocation_never_exceeds_cap`; `tt_disabled_tiny_large_and_interleaved_solves_match` |
| Shared warm/cold validity and lifetime | `shared_tt_warm_and_cold_verdicts_verify`; `shared_tt_survives_origin_arena_drop`; `shared_tt_parent_proof_reuses_descendant` |
| Shared collision and claimant isolation | `shared_tt_forced_collision_a_b_a_stays_valid`; `shared_tt_claimant_isolation` |
| Shared cap/accounting under reuse | `shared_tt_sustained_churn_respects_cap`; `shared_tt_large_tiny_zero_reconfiguration` |
| Atomic import and positive-only invariant | `shared_tt_import_preflight_is_atomic`; `shared_tt_never_caches_unknown` |
| Requested-side specialization | `solve_goal_filters_root_facts`; `solve_goal_one_sided_gets_full_budget` |
| Make/unmake integrity | `make_unmake_round_trips_on_proof_and_cap_exit`; `recursive_make_unmake_restores_exact_public_state` |
| Caps/accounting | `zero_node_cap_is_unknown_and_certless`; `node_cap_only_moves_results_toward_unknown`; local/shared TT accounting; oversized-certificate rejection |
| Conditional determinism | `solver_configurations_are_deterministic_on_hard_leaf`; `shared_tt_conditional_determinism` |
| Extended benchmark corpus | `curated_bench_corpus_matches_shadow_fixture_families`; ignored `tss_bench_report` |

The four Python shadow fixture histories are replayed in Rust.  Under the
agreement rule, the intentionally zero-capped run of the very wide exact
FirstStone forced-loss fixture is `Unknown`; its hard lambda dual is separately
verified, and its reachable `SecondStone` continuation is exhaustively compared
to the reference at sufficient depth.  This keeps default debug CI bounded
without treating an unsearched reference frontier as a verdict.  The
spare-tempo fixture proves that `(4,4)` is a real non-hitting counter-fork when
`k=1<B=2`; a malicious implicit-dispatch certificate is rejected.

## 8. Benchmark snapshot

Command:

```text
cargo test --release -p hexfield_eq tss_bench_report -- --ignored --nocapture
```

Configuration: `node_cap=100`, `tt_bytes_cap=65536`, fourteen synthetic
stones-on-board buckets with five positions each, and four D6 images of each
curated `FORCED_DEFENSE`, `DEEP_WIN`, and `FORCED_LOSS` family (82 positions
total).  A reusable solver handle is retained within each timed pass.  These
2026-07-13 host measurements are small samples, not a hardware SLA; the gate
column is the specified 20,000 nodes/s threshold.

| Bucket | Stones | Positions | Before nodes/s | Final nodes/s | Gate |
|---|---:|---:|---:|---:|:---:|
| synthetic | 3 | 5 | 87,504.4 | 84,146.8 | PASS |
| synthetic | 4 | 5 | 159,109.0 | 162,866.4 | PASS |
| synthetic | 7 | 5 | 705.6 | 103,272.7 | PASS |
| synthetic | 8 | 5 | 510.9 | 106,339.9 | PASS |
| synthetic | 11 | 5 | 48,287.0 | 43,960.6 | PASS |
| synthetic | 12 | 5 | 384.7 | 63,051.7 | PASS |
| synthetic | 15 | 5 | 218.1 | 78,784.1 | PASS |
| synthetic | 16 | 5 | 265.1 | 67,227.4 | PASS |
| synthetic | 19 | 5 | 45,180.7 | 44,881.4 | PASS |
| synthetic | 20 | 5 | 68,691.3 | 105,673.0 | PASS |
| synthetic | 23 | 5 | 29,093.9 | 25,961.2 | PASS |
| synthetic | 24 | 5 | 136.0 | 24,909.1 | PASS |
| synthetic | 27 | 5 | 22,309.6 | 37,695.2 | PASS |
| synthetic | 28 | 5 | 23,491.6 | 55,296.9 | PASS |
| `FORCED_DEFENSE` | 9 | 4 | 565.7 | 241,935.5 | PASS |
| `DEEP_WIN` | 15 | 4 | 15,081.5 | 141,592.9 | PASS |
| `FORCED_LOSS` | 17 | 4 | 233,918.1 | 222,222.2 | PASS |

The cap-2000 pass changed from 12,510 nodes in 158.978 seconds (78.7
nodes/s) to 9,296 nodes in 0.322549 seconds (28,820.5 nodes/s).  Median solve
latency changed from 0.1585 ms to 0.07845 ms; final p95 was 28.8729 ms and
maximum was 51.7928 ms.  The required median-at-most-10-ms gate passes, and the
slowest bucket is synthetic stones-24 at 24,909.1 nodes/s.

## 9. Limitations that preserve soundness

- The solver is deliberately incomplete: claimant moves that do not immediately
  create a >=4 threat are omitted.  Such wins become `Unknown`.
- In `SolveGoal::Both`, the primal/dual half-budget split and fixed
  depth/certificate bounds can turn a proof into `Unknown`.  A single-side goal
  removes the split but not the fixed bounds.
- The persistent TT stores completed positive proof fragments only.  Bounded
  replacement or an import that does not fit the remaining depth/node/edge
  allowance becomes a miss, so cache reuse remains incomplete.  There is no D6
  value cache.
- Warm cache history may change whether a proof is discovered under a cap and
  may change certificate shape and telemetry.  Validity remains independent of
  that history, and fixed initial cache state restores deterministic output.
- The board is unbounded, but every engine legal set at a finite state is
  finite.  An infinite forcing line is stopped by deterministic caps and is
  `Unknown`.
- The benchmark sample is small and cap-specific.  It measures this harness,
  not a production SLA.

None of these limitations supplies a path from missing work to a hard verdict.
