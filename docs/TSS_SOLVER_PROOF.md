# Proof-carrying forced-tree solver: soundness argument

This document is the proof obligation for the Stage-3 deliverable specified in
`docs/TSS_SOLVER_SPEC.md`.  Line references below name the formatted source in
this worktree.  The frozen seam in `tss_core.rs` was not changed.

## 1. Scope, notation, and result meaning

Fix a reachable engine state `s` and a player identity `C`, called the
**claimant**.  A proof node is existential when
`s.current_player() == C` and universal otherwise.  This identity rule, rather
than placement-depth parity, is essential because `FirstStone -> SecondStone`
does not change the player (`hexo_engine/rust/src/state.rs:324`), while
`SecondStone -> FirstStone` does (`state.rs:330`).

`Win` means that the root side to move is the claimant.  `Loss` means that the
claimant is the other player.  `Unknown` carries no game value.  The claimant
mapping is implemented at `tss_solver.rs:79`, `tss_solver.rs:109`, and
`tss_solver.rs:197`; the verifier derives the mapping independently at
`tss_verify.rs:114`.

The shared lambda-one facts are:

- `B = 2` at `FirstStone`, and `B = 1` at `Opening`/`SecondStone`
  (`threats_shared.rs:49`);
- a count-5 is win-now for either budget, and a count-4 is win-now only at
  `B = 2` (`threats_shared.rs:140`);
- `forced_loss` means no own win-now and no hitting set of size at most `B`
  (`threats_shared.rs:75`);
- the hitting-set search is exhaustive for the only possible budgets, one and
  two (`threats_shared.rs:98`).

The solver clones the supplied root once at `tss_solver.rs:142`.  Every hot-path
descendant is entered with `apply_with_delta` and left with `undo` in
`tss_solver.rs:287` and `tss_solver.rs:323`; engine support is at
`state.rs:289`/`state.rs:361`.

## 2. Search scheme

The implementation is a deterministic, proof-number-ordered depth-first
AND/OR proof constructor (`tss_solver.rs:249`).  It is an argued-equivalent
best-first scheme rather than a numerical-threshold df-pn implementation:

1. Every eligible OR child receives a static proof-cost tuple: immediate proof,
   fully forced reply, same-turn build, half-forced reply, or quiet reply;
   within a class, more live threats and stronger pre-placement windows have
   lower cost (`tss_solver.rs:460`).
2. D6-canonical coordinates break remaining ties (`tss_solver.rs:528`).
3. The frontier order is the lexicographic sequence of those costs, with every
   descendant of a lower-cost OR child ordered before the next OR sibling.
   Depth-first traversal therefore always expands the least frontier element
   under this fixed best-first order.
4. An AND node has no selective proof: all non-dispatched children must prove,
   so its traversal order cannot alter the logical result.

This scheme and df-pn recognize the same completed proof trees over the same
restricted OR/full universal graph when resources are unbounded.  Under a cap,
the order can change `Unknown` into a discovered proof or vice versa, but it
cannot change the validity of a completed proof.  No heuristic number is ever
converted to `Win` or `Loss`.

The solver first checks a terminal/lambda-one root (`tss_solver.rs:209`).  For a
deeper result it gives deterministic disjoint portions of the node budget to a
primal proof and an independent dual proof (`tss_solver.rs:109` and
`tss_solver.rs:125`).  A failed primal search is never reused as a dual proof.

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

The solver collects the full union at `tss_solver.rs:511`, checks the parent
boundary and staple-checks every proposed omission at `tss_solver.rs:323`, and
falls back to an all-explicit node if any child check fails.  The verifier
independently reconstructs the boundary/universe at `tss_verify.rs:403` and
applies every omitted move at `tss_verify.rs:325`.  Thus the proof is stapled to
the complement rather than assumed.

Tests: `deep_win_contains_verified_universal_coverage`
(`tss_solver.rs:1165`),
`spare_stone_counter_threat_cannot_be_implicitly_dispatched`
(`tss_solver.rs:1319`), and `certificate_mutations_are_rejected`
(`tss_solver.rs:1362`).

### L2 — OR restriction safety

At a claimant node the solver considers only empty cells extending an active
claimant length-six window with at least three stones (`tss_solver.rs:429`).
The selected placement therefore creates a >=4 threat (or wins).  A `Choice`
certificate needs only one legal winning continuation.  Omitting other claimant
moves can hide a different win, but cannot invalidate the replayed selected
win.  Exhaustion of this restricted set returns no proof, never `Loss`
(`tss_solver.rs:287`).  The verifier does not trust the generator; it replays
the chosen legal move (`tss_verify.rs:272`).

Tests: `deep_win_contains_verified_universal_coverage`
(`tss_solver.rs:1165`), seeded differential (`tss_solver.rs:1245`), and all-D6
replay (`tss_solver.rs:1406`).

### L3 — AND completeness

At a universal node there are exactly two cases (`tss_solver.rs:323`):

- under L1, every hitting-universe move is explicit and every complement move
  is independently dispatchable;
- otherwise, including every `k < B` spare-stone node, the explicit list is the
  full engine legal list.

Every explicit child must return a certificate before the universal arena node
is allocated (`tss_solver.rs:402`).  A capped, failed, or absent child returns
`None` and poisons the universal attempt.  The verifier enumerates the legal set
again, rejects duplicate/illegal edges, rejects implicit coverage outside the
L1 boundary, requires every hitting move explicitly, and checks each complement
move (`tss_verify.rs:325`).  Therefore
`searched(node) union dispatch(node) = legal(node)` in every accepted proof.

Tests: real universal proof (`tss_solver.rs:1165`), dropped-child mutation
(`tss_solver.rs:1362`), spare-tempo counterexample (`tss_solver.rs:1319`), and
cap poisoning (`tss_solver.rs:1472`).

### L4 — dual LOSS and identity perspective

A `Loss` certificate is not failure of the root player's attack.  It is a
separate winning certificate with claimant `root.current_player().other()`
(`tss_solver.rs:79`, `tss_solver.rs:125`).  Recursive node kind is selected by
`state.current_player() == claimant` at `tss_solver.rs:249`, so
`FirstStone -> SecondStone` retains node kind and a completed turn changes it.
The verifier independently requires the opponent claimant for `Loss`
(`tss_verify.rs:114`).

Tests: `root_lambda_loss_has_dual_certificate` (`tss_solver.rs:1065`), the
exhaustive forced-loss differential (`tss_solver.rs:1220`), and the independent
two-stone identity test (`tss_reference.rs:332`).

### L5 — Unknown monotonicity

The only certificate-producing return is a successfully allocated factual,
choice, or complete-universal node (`tss_solver.rs:249`-`402`).  Node/depth or
certificate bounds set the limit flag and return `None`; an OR may try another
non-capped child, while an AND immediately returns `None`.  The outer solve
emits a hard status only when the corresponding independent attempt returns a
certificate (`tss_solver.rs:79`).  TT entries contain proved arena node IDs
only; there are no negative/heuristic value entries.  `SolveStats` is never
read by a verdict path.

Tests: zero cap (`tss_solver.rs:1134`), one-below-proof cap
(`tss_solver.rs:1472`), TT modes (`tss_solver.rs:1429`), and the typed seam's
existing verifier gate in `tss_core.rs`.

### L6 — cache identity

`PositionKey` contains sorted `(q,r,owner)` occupancy, current player, exact
phase with `SecondStone.first`, placement count, and terminal winner/count
(`tss_solver.rs:599`, `tss_solver.rs:607`).  The claimant is compared alongside
the key.  The 64-bit FNV value selects one direct-mapped bucket only
(`tss_solver.rs:641`); a hit additionally requires full-key and claimant
equality (`tss_solver.rs:743`).  `hexo_utils::StateHash` is never imported.

The solver has no D6 value cache.  D6 is used only for covariant tie ordering.
Certificates transformed for replay map every root/phase/move coordinate and
are checked against the transformed full root (`tss_verify.rs:582`,
`tss_verify.rs:608`).  Shared certificate arena nodes are reused by the
verifier only when its independently built full replay position is equal
(`tss_verify.rs:200`).

Tests: forced constant-hash collision (`tss_solver.rs:1079`), TT off/tiny/large
and interleaving (`tss_solver.rs:1429`), exact root/witness mutations
(`tss_solver.rs:1362`), and D6 replay (`tss_solver.rs:1406`).

## 5. Optimization-by-optimization arguments

No epsilon, randomized choice, wall clock, neural score, history hash, or
unverified symmetry cache is present.

### O1. Root terminal/lambda-one fast path

`tss_solver.rs:209` emits exactly the same factual leaf that recursive search
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

The tuple at `tss_solver.rs:460` changes only traversal order.  Every OR child
in the restricted generator remains present; every required AND child remains
present.  A cap can make a proof easier or harder to find, but a returned tree
has the same verifier obligations.  No priority enters a certificate fact.

### O5. D6-canonical tie ordering

`tss_solver.rs:528` selects the least full-position D6 image and compares tied
moves in that frame.  D6 maps legal placements, windows, owners, and the phase
witness bijectively.  Therefore it permutes equal-cost branches without adding
or removing one.  It improves cap covariance; certificate remapping is still
verified independently.

### O6. Make/unmake instead of descendant clones

All returns after a successful apply execute the matching LIFO undo.  This is
observationally identical to cloning the parent then applying the move because
the engine delta restores occupancy, legal/window stores, player, phase,
terminal state, last turn, and history.  Exact round trips on successful proof
and forced cap exit are tested at `tss_solver.rs:1460`; the independent
reference repeats the check at `tss_reference.rs:390`.

### O7. Direct-mapped bounded TT and replacement

The table reserves a fixed inline slot vector within `tt_bytes_cap`; cap zero or
an undersized cap disables it (`tss_solver.rs:707`).  A position key's actual
`Vec` capacity plus a conservative allocation charge is accounted.  An insert
first subtracts the replaced key, adds the new key, and is skipped if the total
would exceed the cap (`tss_solver.rs:754`).  Peak telemetry is the maximum of
that exact charged total.  Direct replacement can turn a future hit into a miss
only.  A miss recomputes; it cannot authorize a verdict.  A hit is a previously
proved arena node and requires L6 full equality.

The accounting formula and every tested cap are recomputed independently in
`tt_allocation_never_exceeds_cap` (`tss_solver.rs:1096`).

### O8. Stable hash bucket selection

FNV is not a proof identity.  Masking every hash to zero still cannot cross-hit
because full equality is mandatory.  Thus hash choice affects collision rate
and replacement cost only (`tss_solver.rs:1079`).

### O9. Certificate arena, TT DAG reuse, and compaction

The arena permits an exact-key TT hit to reference an existing proved node.
Before emission, `compact_certificate` copies only nodes reachable from the
chosen proof and remaps IDs (`tss_solver.rs:786`).  It detects a construction
cycle and respects fixed node/edge bounds.  The verifier rejects cycles and
orphans and, for a genuinely shared node, compares a full replay key.  These
operations remove garbage or share identical facts; they do not change a
node's logical obligation.

### O10. Deterministic primal/dual budget split

The split at `tss_solver.rs:109` prevents a failed primal attack from starving
all dual work while keeping total expansions within `node_cap`.  Either half
may miss a proof and return `Unknown`.  A hard result still requires a complete
certificate, so the split changes completeness under cost only.

### O11. Independent reference alpha-beta exits and direct-line order

The oracle enumerates its own complete legal set (`tss_reference.rs:51`) and
uses ordinary three-valued minimax (`tss_reference.rs:137`).  At an existential
node, one proven win makes remaining children irrelevant; at an opponent node,
one proven loss does likewise.  Ordering direct line extensions first
(`tss_reference.rs:204`) changes only when such a decisive child is found.  If
no decisive child exists, all legal moves are still visited.

## 6. Determinism and memory bounds

- There is no wall-clock read in solver or verifier code.  `Instant` appears
  only in the ignored benchmark (`tss_bench.rs:156`).
- Window-store/hash iteration is collected, deduplicated, and explicitly
  ordered before it affects traversal (`tss_solver.rs:429`,
  `tss_solver.rs:511`).
- Every solve allocates fresh context/TT/arena state (`tss_solver.rs:142`), so
  prior and interleaved solves cannot affect it.
- Node count is at most `SolveCaps.node_cap`.  Search/certificate depth is at
  most 256.  Certificate nodes/edges/root and verifier memo have the fixed
  bounds in section 3.
- TT charged peak never exceeds `tt_bytes_cap`; replacement is deterministic.
  Temporary position keys are stack/search temporaries, not retained caches.

Repeated and interleaved equality is exercised at `tss_solver.rs:1149` and
`tss_solver.rs:1429`.

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
| Curated + seeded differential | `curated_differential_and_every_hard_certificate_verifies` (`tss_solver.rs:1184`); `differential_forced_loss_after_first_defender_stone` (`:1220`); `seeded_random_differential_covers_all_phases_and_dense_positions` (`:1245`) |
| Every certificate + mutations | `root_lambda_loss_has_dual_certificate` (`:1065`); `deep_win_contains_verified_universal_coverage` (`:1165`); `certificate_mutations_are_rejected` (`:1362`); verifier graph/bound tests (`tss_verify.rs:725`, `:738`, `:760`) |
| Twelve D6 replays | `d6_status_and_certificate_replay_all_twelve_symmetries` (`tss_solver.rs:1406`); verifier terminal remaps (`tss_verify.rs:777`) |
| TT off/tiny/large/collision | `full_key_rejects_forced_hash_collision` (`tss_solver.rs:1079`); `tt_allocation_never_exceeds_cap` (`:1096`); `tt_disabled_tiny_large_and_interleaved_solves_match` (`:1429`) |
| Make/unmake integrity | `make_unmake_round_trips_on_proof_and_cap_exit` (`tss_solver.rs:1460`); reference round trip (`tss_reference.rs:390`) |
| Caps/accounting | `zero_node_cap_is_unknown_and_certless` (`tss_solver.rs:1134`); `node_cap_only_moves_results_toward_unknown` (`:1472`); TT accounting (`:1096`); oversized cert rejection (`tss_verify.rs:760`) |
| Determinism/interleaving | `solver_configurations_are_deterministic_on_hard_leaf` (`tss_solver.rs:1149`); interleaved forced-collision solve (`:1429`) |

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

Configuration: `node_cap=100`, `tt_bytes_cap=65536`, two uniform plus two
line-biased positions per bucket, plus one deterministic line-biased prefix.
Observed on the 2026-07-13 worktree host (small samples; throughput is
indicative, not a stable hardware promise):

| stones | positions | threatful | nodes/s | Unknown-rate |
|---:|---:|---:|---:|---:|
| 3 | 5 | 0 | 88,464.3 | 1.00 |
| 4 | 5 | 0 | 148,038.5 | 1.00 |
| 7 | 5 | 0 | 730.0 | 1.00 |
| 8 | 5 | 0 | 501.2 | 1.00 |
| 11 | 5 | 1 | 42,700.3 | 0.80 |
| 12 | 5 | 1 | 394.1 | 1.00 |
| 15 | 5 | 1 | 221.4 | 0.80 |
| 16 | 5 | 1 | 265.8 | 0.80 |
| 19 | 5 | 1 | 39,765.2 | 0.80 |
| 20 | 5 | 1 | 64,016.2 | 0.80 |
| 23 | 5 | 1 | 26,622.7 | 0.80 |
| 24 | 5 | 3 | 162.9 | 0.80 |
| 27 | 5 | 1 | 31,765.2 | 0.80 |
| 28 | 5 | 1 | 36,793.2 | 0.80 |

The large variation is real: a counted universal expansion may enumerate
hundreds of legal moves, while a root lambda-one fact costs one node.  Production
cap sizing should therefore stratify by phase/threat state as well as raw node
count.

## 9. Limitations that preserve soundness

- The solver is deliberately incomplete: claimant moves that do not immediately
  create a >=4 threat are omitted.  Such wins become `Unknown`.
- The primal/dual half-budget split and fixed depth/certificate bounds can turn
  a proof into `Unknown`.
- The TT is per solve; there is no persistent or D6 value cache.
- The board is unbounded, but every engine legal set at a finite state is
  finite.  An infinite forcing line is stopped by deterministic caps and is
  `Unknown`.
- The benchmark sample is small and cap-specific.  It measures this harness,
  not a production SLA.

None of these limitations supplies a path from missing work to a hard verdict.

