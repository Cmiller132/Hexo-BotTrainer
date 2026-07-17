# G2R9 shared proven fragments: design and soundness contract

Date: 2026-07-17  
Design base: `hunt/turn-quotient` at `86a6418c`; completion checkpoint
`8615726d`
Rollout flag: `TSS_SHARED_FRAGMENTS=1` (default off; read once when a
`TssSolver` is constructed)

## Design verdict

The build is licensed, with one deliberately narrow interpretation. Lean T10
at `E:\tss-lean`, commit
`69adffc7dd3cb1b33d56242c5d219b1a7d969224`, proves that a strictly valid
certificate DAG unfolds to a sound tree certificate over the original base and
D17 compilers and the same absolute horizon, including both exact dismissal
corollaries (`TssZones/DAGUnfoldingSoundness.lean:24-47,49-103,105-191,193-241`).
T10 validates the *finished DAG*; it does not state that an arbitrary
cache-payload union preserves validity. Consequently this build shares one
self-contained positive proof payload at an exact semantic node, recomputes
dominant labels over the assembled reachable DAG, and submits the entire final
certificate to the unchanged strict verifier. It does not splice incompatible
node forms or treat cached acceptance as a mint.

The master plan's T10 status text is stale, but its contract is consistent with
the proved theorem. U18 requires shared-node budget/rank/exposure labels to
dominate every path copy, reachable obligations to union, and coupling history
to remain path-local
(`docs/PLAN_TSS_SOLVER_UPGRADES.md:307-334,314-323`). U22/A4 requires exact-key
and build-horizon binding, byte-accounted admission, and positive fragments
that survive solve boundaries (`docs/PLAN_TSS_SOLVER_UPGRADES.md:545-564,646`).
No discrepancy requires a stop.

## Stored fragment

A `ProvenFragment` is one owned, self-contained, acyclic certificate sub-DAG.
Its arena IDs are local to the fragment and its root ID names the proposition
"`claimant` wins from this exact position." References in the wide PN frontier
share the owned payload rather than clone it. Import remaps all IDs atomically;
failure of any structural, depth, node, edge, horizon, or byte check is a cache
miss.

The key is:

- the complete, equality-checked `PositionKey`: sorted coordinates and owners,
  current player, exact turn phase (including a pending first stone), absolute
  `placements_made`, and terminal record;
- the fixed claimant;
- the solver proof profile. The store is solver-owned and is cleared rather
  than reused across a width/zone profile change.

The 64-bit hash selects a bucket only. Full key and claimant equality authorize
a hit. There is no D6, support-local, related-position, or hash-only lookup.

The payload carries the compact certificate nodes/edges and resource metadata:
root ID, node/edge counts, height, maximum exact leaf/completion resolution
`resolution_t`, and the minimum zone build deadline `zone_build_t`. The last
two are compatibility stamps, not substitutes for verifier checks. The store
also accounts for the key, slot, arena, nested edge/commutation/witness
allocations, and the single shared payload ownership under the caller's TT byte
cap. Retained fragments are capped at one eighth of that cap. Direct-table
slots are allocated lazily only after a verified promotion; each later search
receives the caller cap minus bytes actually retained. Thus a cold/empty store
leaves the historical wide-search TT cap intact rather than reserving memory
for hypothetical hits.

The store is immutably borrowed for the entire `WidePnSearch`. Every live
frontier reference owns an `Arc`, so a proof obligation in use is pinned and
cannot be replaced underneath the search. Verified admissions and replacements
happen only after the search is dropped. A changed caller TT ceiling, hash
profile, width profile, or zone profile clears/reconfigures the retained store
instead of silently carrying entries across resource/profile regimes.

Only a complete positive proof is a fragment. `UNKNOWN`, an unexpanded or
partial branch, node/TT/certificate-cap exit, staged `DepthCutoff`, and its
inherited `dn=0` are never admitted. The current wide VCF restriction does not
produce a complete refutation, so this round stores no negative fragments.

## Exact merge rule

There are two different classes of labels and they must not be confused.
Positions, node form, edge moves, exact successors, absolute `nextPly`, leaf
resolution, and the structural topological rank are exact. They are compared,
replayed, or required to decrease; they are never max-merged.

For a structurally compatible node representing one exact semantic position,
the U18/T10 proof labels are merged as follows over every incoming path copy
`p`:

```text
B_shared(n)       = max_p B_p(n)
R_shared(n, role) = max_p R_p(n, role)
E_shared(n, W)    = max_p E_p(n, W)
O_shared(n)       = union_p O_p(n)
```

Here `B` is the D14 local defender budget, `R` is the D15 live-role/cell rank,
`E` is the D16 per-window exposure, and `O` is the reachable protected/core/zone
obligation set. Missing roles/windows contribute no obligation, not an invented
equality. Incoming commutation/coupling permission is *not* unioned into the
node key: it remains path-local. The current strict verifier binds
`allowed_commuted` into its shared-node replay key, so all occurrences of one
arena node must have the same context and that context must verify; otherwise
the assembled DAG is rejected.

This is the plan's max-dominant rule, not “labels happen to equal on the first
copy.” Lean makes the distinction concrete. Canonical DAG recurrences take
child suprema and satisfy admissible inequality interfaces
(`TssZones/Zones.lean@69adffc:768-818,900-1022`). D14 budget and D16 exposure
are equal after path unfolding, but a path copy's D15 role rank is only bounded
above by the shared DAG rank because the DAG can see additional reconverging
copies (`TssZones/DAGUnfolding.lean@69adffc:1474-1623,1627-1772`). Mandatory
path-copy zones are subsets of the projected shared-node zone
(`DAGUnfolding.lean@69adffc:1797-1851`).

The Rust certificate does not serialize independent D15/D16 tables. Therefore
the implementation realizes the rule by reconstructing the final reachable
sub-DAG: Universal child budgets combine with `max`, protected/reachable
obligations combine by set union, and zone `d` is relabelled from that dominant
result. The strict verifier independently performs the same max/union
reconstruction (`tss_verify.rs:1024-1087`) and requires stored `zone.d` and the
build horizon to match (`tss_verify.rs:1234-1250`). A shared arena node must
also replay to the same complete state and path-local commutation context
(`tss_verify.rs:322-375,391-459,478-590`).

If two candidates at the same position have incompatible node forms or
outgoing proof payloads, they are alternatives, not merge operands. One
complete payload may replace another cache entry according to bounded policy,
but they are never edge-unioned into a purported proof. This restriction avoids
claiming a constructive merge theorem that T10 does not provide.

## Lookup and horizon rules

A lookup may settle only a positive proof obligation for the identical full
position and claimant. It is checked when a wide node is selected, before that
node is expanded; a root lookup is the same operation at depth zero. Multiple
parents may then reference the one imported fragment root. A miss leaves the
historical PN path unchanged.

Consumption is deliberately stricter than key equality. A cached fragment
whose root is `Universal` is consumed only at solve depth zero, where the
path-local `allowed_commuted` context is known to be empty. Other exact-key
fragment roots may be tried below the root, but exact position is not the whole
verifier context: for example, attacker placement well-formedness also depends
on final-certificate metadata such as `root_stones`. Such an embedding is only
a provisional PN hit. If its outer context is incompatible, final strict replay
rejects the assembled certificate and the solve returns `UNKNOWN`; the cached
fragment can never mint a verdict by key equality alone.

Every consumer embeds the NQ4 transfer rules
(`HUNT_REPORT_TURN_QUOTIENT.md:125-149,233-238`):

- a WIN resolving by `h` may answer only a query with `h' >= h`;
- a complete restricted-search refutation at allowance `h'` could answer only
  `h <= h'`, but no such fragments are admitted this round;
- a typed terminal fact must retain its exact resolution label and that label
  must be within the queried semantic horizon;
- a zone-bearing import additionally requires
  `resolution_t <= query_horizon <= zone_build_t`, the existing build-budget
  direction, zone rebase, certificate-horizon preflight, and final replay;
- `UNKNOWN`, any cap exit, an unexpanded/partial node, and staged
  `DepthCutoff` transfer nowhere. In particular, `dn=0` alone is not a
  refutation.

The store never attempts the NQ3-refuted transfer to a different but allegedly
irrelevant position. Strict `RootBinding` and shared-node `ReplayKey` bind the
complete occupancy, so a different root can hit only an actually identical
descendant position, never a C_rel-style support projection.

## Strict verifier remains the single mint

Cache metadata validation establishes only safe import shape and compatibility.
It never establishes a hard value. Before admission, a candidate subcertificate
is made standalone at its exact root and accepted by `TssVerifier`. After
consumption, IDs are remapped, max/union labels are recomputed, zones are
rebased, and horizon preflight runs on the assembled certificate. Every returned
hard certificate must then be accepted by the unchanged strict verifier for the
caller's exact root. Production can mint `HardValue` only through
`tss_core::hard_value_from_verified` (`tss_core.rs:176-210`). A rejection is
`UNKNOWN`/no mint; cache warmth can improve discovery but cannot bypass replay.

## Scope

In scope this round:

- within-process reuse owned by one `TssSolver`;
- exact-state sharing between transposed parents;
- fragments proven below an ultimately UNKNOWN root;
- repeat/warm solves and a later exact descendant/root visit;
- default-off composition with `TSS_LAZY_FRONTIER` both disabled and enabled;
- bounded, byte-accounted volatile storage and test-only hit/size telemetry.

Out of scope:

- on-disk serialization or persistence across processes;
- cross-position/support-local/C_rel transfer. NQ3 refuted that for today's
  strict certificate format (`HUNT_REPORT_CERT_SUPPORT.md` at commit
  `3cd224fe`; strict unchanged transfer was 0/180 at every tested radius);
- D6-canonical proof lookup;
- negative/refutation fragments until a consumer can prove and tag search
  completeness separately from restricted VCF failure;
- changing the certificate grammar, verifier axioms, or hard-value mint.

## Stop conditions and gates

Implementation stops rather than guessing if an exact-state shared node reaches
different replay states, if a path-local commutation permission would be erased,
or if dominant labels cannot be reconstructed.

The orchestrator's amended verdict contract distinguishes resource warmth from
cold behavior. Every flag-off solve must remain identical to its cold baseline,
and a fragments-on cold solve must have the same verdict as fragments-off cold.
A warm fragments-on verdict may differ from the corresponding flag-off warm
verdict only as `UNKNOWN -> WIN` or `UNKNOWN -> LOSS`, and only when the new
hard verdict carries a certificate accepted by the unchanged strict verifier.
Loss of a hard verdict, `WIN <-> LOSS`, or an unverified new hard verdict is a
mandatory stop. Forced-NO rows must never return `WIN` in any mode. A warm solve
of a different root after seeding remains a cold-contract comparison: its
verdict must exactly match fresh fragments-on and fresh fragments-off solves.

This amendment does not weaken the proof contract. `UNKNOWN` is a resource
verdict, not a game-theoretic result. An imported fragment supplies only an
independently verified sub-proof; T10 licenses its final DAG composition, and
the strict verifier re-accepts the assembled certificate before the single mint
can return a hard value. Thus warm `UNKNOWN ->` verified-hard is capacity
recovered at a fixed budget, equivalent in kind to raising the node cap, rather
than a contradictory verdict. Every such improvement is reported with its
root, ladder rung, before/after verdicts, and expansions saved.

The campaign additionally requires every hard certificate to verify, the
different-root mutation control, and both fragment-only and
fragment-plus-lazy official corpus gates. Cold and warm work, reduced-budget
closure, store bytes/entries, and independent fragment lookup/hit rates are
measurements, not soundness inputs.
