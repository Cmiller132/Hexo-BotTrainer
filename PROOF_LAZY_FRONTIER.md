# R-LF1: Lazy-Frontier Refinement Lemma

## Scope and code model

This proof is about the wide proof-number engine `WidePnSearch` in
`packages/hexfield_eq/rust/src/tss_solver.rs`. It does not change the narrow
compatibility engine.

The wide arena is `WidePnSearch.entries`, the exact-position TT is
`WidePnSearch.by_position`, and a parent edge is `WidePnChild` (lines 976-993,
1961-1993). A pending edge carries its move, result, immutable PN/DN prior, and
an optional arena ID. `insert_position` (line 2142) probes the exact key, counts
a TT hit and returns the existing ID if present; otherwise it creates an
`Unexpanded` arena record and indexes it if the byte ceiling permits. Before
R-LF1 the attacker generators already left pending children unlinked, while
`defender_children` and `defender_pair_children` eagerly called
`insert_position` for every pending reply (now lines 3469 and 3537).

R-LF1 stores exact future-key material in `WideFutureKey` (lines 996-1023) and
uses two visibility modes. A defender `Virtual` key represents the eager entry
to every pre-selection read. An attacker `OnSelection` key preserves the old
attacker behavior: it is invisible to PN/DN reads and is used only if that
historically unlinked edge is selected. `deferred_by_position` records the
first eager-equivalent defender depth/prior for each not-yet-materialized key,
without allocating an arena record or admitting it to `by_position`
(lines 1026-1029, 1988-1993, and `defer_position` at line 2191).

On first selection, after replaying the move into the working state, `work`
(line 2382) takes the saved exact key, calls `insert_position`, installs the
returned arena ID, and only then recurses. `insert_position` consumes a matching
deferred record so the realized entry uses the same first-admitted depth/prior
as eager. A pre-existing materialized transposition is joined before the
selected state can be expanded.

The statement below is for an uncapped exact-position index, or for an
execution prefix before either implementation refuses an index insertion. The
cap-aware corollary states the deliberately weaker result after that point.

## Lazy-Frontier Refinement Lemma

Let `E` be a reachable eager wide-search configuration and let `L` be obtained
from it by replacing any set of never-selected, pending, unexpanded defender
child entries with thunks. For each replaced edge, retain:

1. the same `WidePnMove` and `WidePnChildResult::Pending`;
2. the same `WidePnPrior`, urgency bit, width tier, and generator rank; and
3. the exact `WidePositionKey` of the state obtained by replaying the edge move.

Identify each eager unexpanded defender entry with the lazy deferred record and
all `Virtual` edge thunks having its key. Identify an eager attacker lazy edge
with the corresponding `OnSelection` edge. Identify an eager expanded entry
with the lazy arena entry indexed by the same exact key. Then, until an
index-cap refusal:

- every corresponding child has the same `child_numbers`;
- every parent recomputation produces the same PN/DN pair;
- every selection policy and df-pn threshold computation chooses the same
  generator-ranked edge and computes the same thresholds;
- realizing a selected thunk produces the identical state and exact position
  key, joins the same existing transposition (if any) before expansion, and
  otherwise creates an entry with identical depth, prior, node tag, and live
  PN/DN numbers; and
- eager and lazy executions therefore have the same reachable PN fixed points
  and the same materializable certificates.

### Proof

There are three pending-edge cases in `child_numbers` (line 2710). First, an
eager defender entry that remains unexpanded exposes the immutable prior of
the first eager insertion. Its lazy `Virtual` counterpart reads that same
first prior from `deferred_by_position`. This matters when two generated edges
have the same exact key but different locally computed priors. Second, if that
key has since been materialized through any transposed parent, both eager and
lazy read the live arena entry found through `by_position` via
`resolved_child_entry` (line 2735). Third, historical attacker edges are
unlinked before selection in eager; `OnSelection` is deliberately invisible to
both lookup paths, so lazy reads the same edge-local prior. Terminal results
are unchanged and retain `(0, infinity)` or `(infinity, 0)`.

The Choice recurrence is minimum PN and saturated-sum DN. The Universal
recurrence is saturated-sum PN and minimum DN. Since corresponding child
numbers are equal in generator order, `recompute` returns equal parent numbers.
The same pointwise equality preserves ordinary lowest-number selection,
sequential-root probing, urgent-block ordering, root width-tier ordering,
genuine-proven/refuted filtering, second-best sibling values, and both df-pn
budget-subtraction thresholds.

Universal commitment has one additional identity obligation. The eager policy
counts distinct linked defender arena IDs and treats yielded children sharing
an ID as one transposed obligation. `has_commitment_fanout` and
`child_obligation_identity` (lines 2977 and 2996) derive the same relation from
`Virtual` keys: a materialized key denotes its indexed ID; otherwise equal keys
denote one deferred first admission. `OnSelection` keys never enter a
Universal defender vector. With an uncapped index, this is exactly eager,
generator-ordered `insert_position` identity. The four-obligation fanout latch,
retained obligation, and yielded-alias suppression (`universal_obligation_index`,
line 3040) therefore make the same choices.

Suppose the common policy first selects a transformed edge. Replaying its
stored `One`, `Pair`, or `DefenderPair` move uses the same state transition as
generation and eager descent. `WidePositionKey::from_state` is exact: it encodes
current player, placement count, phase (including a pending first stone), and
the sorted owned stones. `after_completed_pair` constructs the same encoding
directly for the stateless attacker pair generator; the equivalence harness can
recompute and assert every saved key at realization. The lazy path calls
`insert_position` before recursive `work`. If the key is indexed, it obtains
the corresponding live entry. If deferred, it consumes the stored first
depth/prior and creates the same logical `Unexpanded` record. Numeric arena IDs
need not match; no recurrence orders by ID. Installing the ID establishes the
configuration relation with one fewer thunk, so recursion follows by induction.

If a defender edge is never selected, its eager record may remain unexpanded
or may become live through a transposed parent. The deferred/virtual lookup
cases above expose the same numbers in either event. Induction over selections,
expansions, and staged reopenings gives equal selected edges, equal expansion
states, and equal PN/DN values on corresponding logical entries. Both
executions reach a zero-PN proof or zero-DN restricted-search refutation
together in the uncapped-index model, or retain corresponding unresolved
frontiers.

Certificate construction follows only zero-PN edges. Every pending prior is
strictly positive (`pn_from_fork_degree` is in `1..=37` and `dn_from_tau` is at
least one), so a purely deferred thunk cannot itself be proof evidence. A
zero-PN virtual edge resolves through `by_position` to a realized proof entry.
The materializer uses `resolved_child_entry` in `build_choice`,
`build_universal`, and `build_defender_pair_universal` (lines 3636, 3797, and
3845), replays identical moves in generator order, and memoizes exact
positions. It emits the same certificate graph and canonical bytes. Later
compaction and zone-distance rebase consume that certificate, not frontier
thunks.

## Exhaustive pre-selection read audit and design constraints

The following is every wide-engine place that can observe a child before that
child is selected. Line references are to the final R-LF1
`tss_solver.rs`.

1. **Generation and MHS/prior computation.** `attack_pair_children`,
   `attack_single_children`, `defender_children`, and
   `defender_pair_children` (lines 3314, 3388, 3469, and 3537) compute the
   result/prior while the child state (or stateless pair gate) is available.
   `position_prior` and
   `completed_turn_prior` may run threat analysis/minimum-hitting-set work, but
   neither reads an arena child. R-LF1 saves the already-computed prior; it does
   not recompute a heuristic at realization. Defender edges register `Virtual`
   keys and the first deferred prior/depth. Attacker edges save
   `OnSelection` keys solely so selection can join a deferred defender state;
   letting attacker keys affect pre-selection numbers would change the
   historical attacker-lazy policy.

2. **Initial and bottom-up PN/DN recomputation.** `child_numbers`, `recompute`,
   `refresh`, and `refresh_all_bottom_up` (notably lines 2710 and 3160) read
   child numbers before selection. Constraint: a `Virtual` edge observes a
   materialized transposition or the deferred first prior; an `OnSelection`
   edge observes only its edge prior until linked.

3. **Ordinary selection and thresholds.** `choice_order_pn`,
   `select_child_index_with_tier`, `select_step_child_index_with_commitment`,
   and `work` read child numbers, terminal status, urgency, width tier,
   generator rank, second-best sibling values, and parent sums. Constraint:
   preserve all edge fields and iteration order; realization happens only
   after the edge and thresholds have been selected.

4. **Semantic terminal filters.** `child_is_genuinely_refuted` and
   `child_is_genuinely_proven` inspect a linked entry's node/numbers. A thunk is
   neither genuinely refuted nor proven unless its `Virtual` key resolves to a
   materialized entry with that verdict. A deferred positive prior is
   unresolved.

5. **Universal commitment and transposed sibling identity.**
   `has_commitment_fanout`, `universal_commitment_active`, and
   `universal_obligation_index` (lines 2977-3058) inspect entry identity before
   selection. Constraint: use `Virtual` exact-key identity, resolve it through
   `by_position` when materialized, and recognize equal deferred keys when
   suppressing a yielded alias. Merely treating all thunks as distinct or
   postponing commitment changes visit order.

6. **Staged reopening.** `reopen_depth_cutoffs` scans arena entries and
   `refresh_all_bottom_up` propagates reopened numbers. An unselected eager
   entry is `Unexpanded`, never `DepthCutoff`, so omitting its arena record
   changes nothing. A selected thunk is admitted before `expand` can label it
   `DepthCutoff`; it therefore participates in every later reopening exactly
   as eager does.

7. **Tracing and root diagnostics (test-only observation).**
   `format_trace_child`, `trace_selected_path`, and the `TSS_TRACE_PN` root
   dump inspect entry linkage. Their diagnostic text may intentionally say
   thunk/unlinked instead of displaying an eager unexpanded ID. They must not
   feed search decisions.

8. **NQ4 telemetry (test-only observation).** `observe_insert`,
   `observe_expand`, `observe_stage`, `sound_verdicts`, and `finish` walk
   retained/indexed entries. Lazy thunks are intentionally absent from the
   arena/index denominators until realization; the deferred registry is also
   excluded because it is frontier metadata, not an arena/TT admission.
   Expanded-position accounting remains keyed by the admitted entry.
   Horizon shadow counts, indexed/retained counts, D6 denominators, TT hits,
   and insertion timing measure actual lazy admission rather than reconstructing
   the eager counterfactual.

9. **TT-hit accounting.** `insert_position` is the only wide TT-hit increment.
   A selected thunk calls it before expansion. Hits against actual
   `by_position` entries are counted; matches against deferred frontier records
   are not TT hits. Eager generation hits for never-selected edges disappear.
   This is intended and must be reported rather than forced to match.

10. **Certificate assembly.** `WideProofMaterializer::build`, `build_choice`,
    `build_universal`, and `build_defender_pair_universal` read child numbers,
    results, moves, and arena IDs. Positive pending priors prevent a thunk from
    satisfying their zero-PN preconditions. A zero-PN virtual proof edge must
    resolve to an arena entry even if that particular parent edge never stored
    the ID. Defender-pair assembly independently rebuilds the canonical plan
    and checks the final exact key.

11. **Zone rebase and certificate compaction.** `compact_certificate` and
    `rebase_zone_distances` run only after materialization and read certificate
    nodes, not search children. They impose no extra thunk representation, but
    identical certificate input is required by the lemma.

There is no MHS, zone-generator, commutation-generator, or shared-proof-cache
path that directly reads a wide child arena entry before selection. Those
computations either produce the saved prior/move during generation or consume
the completed certificate afterward.

## Cap-aware corollary

Proof validity is invariant under lazy admission, including under caps: the
solver still returns a certificate only after the unchanged materializer has
followed realized zero-PN children, and the independent verifier checks that
certificate against the root. A lazy run cannot turn an unresolved thunk into
proof evidence.

The following quantities may differ, honestly, under resource ceilings:

- retained arena records and indexed-entry count (the intended reduction);
- `current_bytes`/`peak_tt_bytes`, index-rejection counts, and which later keys
  fit the TT, because admission is delayed and follows selection order rather
  than generation order;
- after an eager index refusal, eager creates separate unindexed arena records
  for later equal keys, while lazy may still have one deferred exact-key
  identity; this is the precise point at which the uncapped simulation relation
  is no longer required;
- `tt_hits`, because never-selected prospective hits are no longer counted and
  a different capped index population can change later hits;
- wall-clock and allocation/hash work;
- staged telemetry denominators and shadow horizon/D6 counts, because thunks
  are not arena entries;
- after a TT-cap refusal, traversal order and expanded-node count, because the
  exact transposition identities available to Universal commitment and later
  selection can differ; and
- consequently, at a node cap, the exact point at which an unresolved search
  returns `UNKNOWN` may differ. A wall-clock cap, if introduced by a caller,
  would have the analogous timing dependence.

The node cap counts expansions, not admissions, so delayed allocation alone
does not consume or refund nodes. Before index-cap divergence, the lemma gives
identical expansion states and counts even at a node cap. After index-cap
divergence, a capped eager run and capped lazy run need not close on the same
expansion; `UNKNOWN` timing and node count are therefore not semantic
equivalence requirements. A `WIN` or `LOSS` remains acceptable only with its
valid materialized certificate. No cap-derived `UNKNOWN`, staged
`DepthCutoff`, unexpanded entry, or thunk is promoted to a proof or refutation.

Thus the uncapped reachable PN fixed points and certificates are equal. Caps
may select different finite prefixes of that sound search, but they do not
change the meaning or validity of any returned proof.

`peak_tt_bytes` continues to mean the production exact-position TT charge.
Like child vectors and eager arena records, edge-owned future keys and the
deferred frontier registry are not included in that counter. R-LF1 therefore
proves and measures a large arena/TT-admission reduction, but it does **not**
claim to remove all key construction or all frontier memory: exact equivalence
requires key-bearing attacker edges and a lightweight deferred-key map.
