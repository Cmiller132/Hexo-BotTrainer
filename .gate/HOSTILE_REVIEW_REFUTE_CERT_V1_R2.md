# HOSTILE REVIEW: REFUTE-CERT V1, ROUND 2

**Review date:** 2026-07-21  
**Target:** `docs/DESIGN_REFUTE_CERT_V1.md` at
`305eeeb850b59d43da4ef4400ae22116f812e33b`  
**Target blob:** `1bed13531b1e891c16cdf52a614cfc7fa0a35086`  
**Prior review:** `.gate/HOSTILE_REVIEW_REFUTE_CERT_V1.md` at
`d4af5aef96ccad325502d35ffc01f372ae6c7d6f`  
**Review mode:** hostile confirmation, doc-only

## VERDICT: CLEARED-WITH-REQUIRED-CHANGES

The amended design fixes the substantive soundness defects found in round 1.
The positive class is now literal and equality-only; the negative grammar is
closed; `T/G1/S/U`, ordered replay, quotienting, pair dispositions, work caps,
the verifier firewall, lifecycle economics, coordinate closure, and semantic
version boundaries are all stated strongly enough to support the leaf cut. All
eight named counterexamples are neutralized against the text as amended.

The leaf-only proof idea also survives a fresh attack. If the verifier has
independently established every premise of `NoAdmissibleFirstTurn`, complete
regeneration really does eliminate every root `ClaimantChoice` constructor.
The separate result type, root binding, no-consumer rule, and unchanged
`Unknown` status prevent that fact from becoming game Loss or a stronger
search/training claim. The external `Q`/operation/time/heap caps and the
conjunctive lifecycle gates are decisive even for a one-leaf artifact.

Two specification defects remain before source work is authorized:

1. Sections 3.2 and 6 do not use the exact leaf predicate published in section
   2.4. They make an empty admitted set appear sufficient, and the section 6
   gate also omits the producer-profile and `expansions < node_cap` conditions.
   That conflicts both with the verifier and with the no-false-scope gate.
2. The claimed literal wire still does not define the byte preimage of
   `root_semantic_sha256` or the unit/invariants of the four failing counters.
   Independent implementations therefore do not yet have one canonical root
   identity and one canonical redundant-count interpretation.

These are bounded document corrections, not a redesign. They are nevertheless
pre-build requirements because one creates contradictory eligibility promises
and the other leaves proof identity implementation-defined. This verdict does
**not** authorize the implementation lane until R2-1 and R2-2 below are applied
and confirmed. It never authorizes the full-tree format.

## Amendment fidelity

| item | verdict | stricter-reading result |
|---|---|---|
| R1 | **DISCHARGED** | Section 1.1 publishes one phase-indexed positive constructor table with claimant, full phase payload, placement clock, finite deadline, versions, and reachability visible. It defines the pointwise and all-finite-horizon negatives, states boundary-free soundness and monotonicity separately, and section 1.3 separates model-byte soundness, leaf soundness, Rust extensionality, and producer materialization. The text no longer calls executed Rust or producer correspondence proved. |
| R2 | **PARTIAL** | Sections 2.2-2.4 freeze the direct windows, `T`, count-one `G1`, turn-start `S`, ordered `U/Q`, guarded reverse quotient, pair family, precedence, total occurrence expansion, closed leaves, tests, and telemetry. The remaining literal-wire gap is that SHA-256's typed “canonical sequence” is not serialized to specified bytes and the four failure counters do not say whether they count quotient classes or ordered occurrences. See R2-2. |
| R3 | **DISCHARGED** | Leaf v1 consumes no PN arena or selector. For any future full tree, section 3.2 mandates a bottom-up memoized `Structural | Unresolved` fixed point, recursive provenance, later-successful-`K_b` selection, atomic-pair unfolding, exact memo identity, cycle refusal, and post-refresh natural-exhaust derivation. A shallow zero has no authority. |
| R4 | **PARTIAL** | The catch-all and `NoJointCarrier` are removed, unknown constructors reject, `tau<b`, `tau=b`, and `tau>b` have distinct polarity, and `NoAdmissibleFirstTurn` is the sole accepted wire tag. However, the producer and gate later weaken the exact section 2.4 leaf premises to “empty admitted set”; the closed matrix is not used consistently. See R2-1. |
| R5 | **DISCHARGED** | Section 4 states a deliberate shared trust base, a transitive semantic denylist, private direct geometry/state/transition checks, engine/direct agreement, source and compiled-call-graph enforcement, third-oracle vectors, and one-sided mutations. Renaming or indirectly reaching a forbidden semantic helper remains forbidden. |
| R6 | **DISCHARGED** | Externally selected ceilings now cover root/window/set sizes, ordered `Q`, primitive semantic work, retained state, heap, CPU, and wall time. Counters are charged before work, pair evaluation streams, producer counts never bound verifier loops, and budget exhaustion is typed non-evidence. This kills the omitted-complement work bomb. |
| R7 | **DISCHARGED** | Node-count proxies and unpinned witness estimates are withdrawn. The design gates causal denominators, competent baselines, held-out tails, absolute producer cost, enabled workflow cost, replay cost, and the declared three-audit amortization. Every applicable result is an AND, so size cannot compensate for failed replay or emission economics. |
| R8 | **PARTIAL** | The fixed cube-component root domain rejects the extreme-coordinate construction; every encountered semantic coordinate is checked; versions occur in the header, abstract root identity, and typed result; and semantic drift requires a new version. The root hash does not yet bind those values through a specified byte preimage, so the identity portion is not fully frozen. See R2-2. |

No amendment is **EVADED**. R2 and R8 have the same serialization defect; R4
has the separate eligibility-consistency defect.

## Counterexample neutralization

| NCE | result of revival attempt against the amended text |
|---|---|
| NCE-01 | **NEUTRALIZED.** No recursive selector exists in the accepted leaf wire. The future-only algorithm recursively classifies support, continues past an earlier unresolved defender reply, selects the least successful plan, and makes cycles/unexplained `Refuted` unresolved. Reversing child order no longer changes existence. |
| NCE-02 | **NEUTRALIZED.** The leaf may still be one node while `Q` is quadratic, but `|T|`, each `|S|`, total `Q`, primitive memberships, heap, CPU, and wall are externally capped and charged before work. Evaluation must stream. A larger construction becomes `UnsupportedPolicyBudget`, not an accepted negative. |
| NCE-03 | **NEUTRALIZED.** `G1(P,a)` literally includes the count-one-through-`a` promotion, `S` is evaluated on turn-start `P`, and the exact ordered occurrence must be classified. The third-oracle fixture and one-sided omission mutation attack correlated producer/verifier drift. |
| NCE-04 | **NEUTRALIZED.** Full phase payload, including `SecondStone.first`, participates in semantic state, direct transition, exact memo/reuse identity, mutation tests, and replay. Leaf v1 has no recursive `SecondStone` node to alias, but its direct transition cross-check still preserves the rule. |
| NCE-05 | **NEUTRALIZED.** The proposed `E = 0.70*S` schedule fails `E < 0.25*S`, enabled-solve tail limits, and likely the three-audit inequality. Replay speed alone cannot pass the producer/end-to-end gate. |
| NCE-06 | **NEUTRALIZED.** The fixed root preflight excludes `-32768`; checked `i32` cube/D6 operations precede conversion; every subsequently encountered leaf-enumeration coordinate must be D6-safe. Original bytes remain bound to the raw root and cannot verify against a distinct image. |
| NCE-07 | **NEUTRALIZED.** The positive grammar has a Universal only at `tau=b`. Nonempty `tau<b` is the typed negative `LooseDefenderBoundary`; `tau>b` is claimant-positive. Generic full-legal-set T6 behavior is explicitly not imported. |
| NCE-08 | **NEUTRALIZED.** Direct terminal replay distinguishes the winner. A claimant terminal/completion prevents a negative leaf; an opponent terminal has a typed mathematical leaf; current numeric `Refuted` and PN zero have no proof authority. |

The two new findings below do not revive an NCE. They are consistency and
canonical-wire failures introduced or left exposed by the leaf cut itself.

## Required changes

### R2-1 -- Use one exact `RefuteLeafExact/V1` eligibility predicate everywhere

**Amends:** sections 3.2 and 6; section 2.4 remains authoritative.

Define one named leaf predicate and use that identical conjunction for producer
eligibility, self-verification expectations, and the leaf-eligibility gate. It
must include:

- every root, reachability, phase, claimant, D6, policy, profile, and
  `expansions < node_cap` condition required by the cut;
- absence of the earlier claimant terminal and `OwnWinNow_A` constructors;
- `not ForcedLoss_A(P,2)`, as required by the section 2.4
  `NoAdmissibleFirstTurn` row; and
- complete regeneration of `U` and zero completion, claimant-tactical, and
  `TightPair` dispositions.

“The admitted set is empty” must not stand alone as a synonym for this
predicate. A reachable claimant root can have no admitted positive pair while
`ForcedLoss_A(P,2)` holds: arrange three independently hittable defender
count-four families, no claimant own-win, and no claimant post-pair threat.
Every pair fails, but section 2.4 correctly requires the different
`OpponentForcedTactical` negative constructor, which the leaf wire does not
carry. The verifier rejects the sole leaf while sections 3.2/6 currently promise
eligibility.

Likewise, section 6's “every” promise must include the declared profile and
strict-cap predicates, or explicitly say that otherwise-semantic leaf roots
return a typed profile/cap ineligibility. As written, the leaf-eligibility gate
requires emission for an exact-empty root at the cap while the no-false-scope
gate forbids it.

Add mandatory fixtures for (a) empty admitted set plus opponent forced tactical,
(b) empty admitted set plus any earlier claimant-positive constructor that can
be realized, and (c) semantic leaf premises with profile/cap ineligibility. Do
not solve this by enabling another leaf tag in wire v1.

### R2-2 -- Finish the canonical byte identity and redundant-count contract

**Amends:** sections 2.1, 4.2, 4.3, and 6.

Publish the exact byte string hashed by `root_semantic_sha256`: include a fixed
domain separator and specify the order and exact encoding/length treatment of
every ruleset, coordinate, class, wire/profile, stone, owner, mover, full phase
payload, placement-clock, and terminal field. State whether it is a specified
subset of the literal header bytes or a separate canonical encoding. An
abstract tuple called a “canonical sequence” is not a SHA-256 preimage.

Also state whether each `fail_*` field counts quotient classes or ordered
occurrences, and give the applicable sum identities against
`quotient_class_count` and/or `Q`. Require independent golden vectors that pin
the digest and both sole-orientation and two-member-quotient counter values.
These fields may remain redundant, but their interpretation cannot be selected
by either implementation.

## Fresh attack log: `RefuteLeafExact/V1`

| ID | attack | outcome |
|---|---|---|
| L01 | Find an open or catch-all positive/negative branch in the fixed equality-only class. | **FAILED.** The priority table is closed, loose/equal/tactical boundaries are separated, and unknown constructors require a new class/wire. |
| L02 | Accept the leaf solely because all pair dispositions fail even though an earlier root constructor controls. | **SUCCEEDS against producer/gate wording -> R2-1.** Section 2.4 and verifier logic reject correctly; sections 3.2/6 overpromise emission. |
| L03 | Make a boundary-free leaf prove only one `u32::MAX` horizon. | **FAILED normatively.** The leaf induction and constructor monotonicity target all finite deadlines; `u32::MAX` is only a producer-profile marker. |
| L04 | Omit a weak `G1` candidate, stale defender block, reverse orientation, or hidden terminal prefix. | **FAILED.** Literal turn-start equations, total ordered occurrence expansion, guarded quotienting, direct replay, third-oracle vectors, and one-sided mutations cover each omission. |
| L05 | Let quotienting erase an asymmetric occurrence or count one member without checking the other. | **FAILED semantically.** Only guarded reverse occurrences commute, both members are classified, and every ordered occurrence maps exactly once. Counter units still need R2-2. |
| L06 | Use payload counts to truncate regeneration or make producer telemetry authoritative. | **FAILED.** The verifier's regenerated loops are policy-bound, not count-bound, and all counts must equal independently derived values. |
| L07 | Make two independent implementations choose different proof-identity bytes while both claim the literal wire. | **SUCCEEDS as a specification/interoperability attack -> R2-2.** The SHA-256 preimage is a typed tuple, not specified bytes. Exact root comparison still prevents a direct false cross-root acceptance. |
| L08 | Hide NCE-02-scale work behind the one-byte leaf tag. | **FAILED.** Absolute `Q`, primitive-work, memory, CPU, and wall ceilings fail closed before unbounded work; producer and verifier share the external ceilings, not semantic code. |
| L09 | Reuse an artifact across a raw root, D6 image, claimant, phase, ruleset, or reachability change. | **FAILED subject to R2-2's canonical encoding fix.** Literal root fields, token, semantic comparisons, and transformed rehashing are authoritative; a digest alone never authorizes reuse. |
| L10 | Mint Loss, `HardValue(-1)`, a trainer label, cache entry, pruning fact, or opponent strategy from leaf acceptance. | **FAILED.** The public type carries only the class refutation, ordinary status stays `Unknown`, v1 has no consumer, and every stronger conversion is a kill condition. |
| L11 | Make the one-leaf codec economically pass by being small while emit or replay is net-negative. | **FAILED.** Size, replay, producer, enabled workflow, and three-audit amortization are independent conjuncts; the compact leaf baseline applies on every root and in aggregate. |
| L12 | Quietly require the removed full-tree support fixed point, DAG, Universal nodes, or arena provenance to verify the leaf. | **FAILED.** Leaf verification is direct root enumeration. Search profile/cap values are producer eligibility metadata, not logical evidence, but their exact eligibility role must be made consistent by R2-1. |

## Leaf-cut disposition

| question | disposition |
|---|---|
| Is the equality-only class fixed, well-defined, and closed? | **YES.** The published constructor priority and negative matrix are a closed class version. `tau<b` is negative, `tau=b` is the only Universal boundary, and `tau>b` is positive. Unknown growth cannot fall through. |
| Is complete `T/G1/S/U` regeneration independently verifiable? | **YES.** Direct window enumeration, turn-start equations, ordered replay, guarded quotienting, disposition precedence, private transitions, exact telemetry, oracle fixtures, call-graph enforcement, and mutations form an independently checkable obligation. R2-2 is needed only to finish byte/count canonicality. |
| Can an accepted path mint more than the literal fact? | **NO.** Acceptance yields only `NoContractWinV1` for the bound root/class/reachability arguments via `NoAdmissibleFirstTurn`. It yields neither full-game loss nor opponent strategy nor absence of out-of-class wins. |
| Are economics decisive at leaf scale? | **YES.** A leaf must pass the compact leaf-byte comparator plus absolute and relative replay, producer, enabled-workflow, tail, held-out, and three-audit gates. No alternate conjunct or “fail both” escape remains. |
| Does the cut depend on removed full-tree machinery? | **NO logical dependency.** It reads no arena zero, support plan, DAG, Universal reply, or recursive node. Existing search completion/profile metadata controls when the optional producer runs; R2-1 must make that policy dependency explicit and consistent. |

## Scope-boundary integrity

Full-tree v1 remains **NO-GO**. The retained full phase-indexed mathematics is
needed to state what the root leaf refutes; it does not make recursive tags part
of the accepted wire.

| surface | scope finding |
|---|---|
| Wire/codec | `RefuteLeafExact/V1` accepts only tag `0x20`. DAG nodes, Universal counterexamples, and every other leaf require a new wire and hostile review. |
| Producer | V1 performs producer-side exact root enumeration after ordinary search. The recursive support fixed point and arena compaction are explicitly future-only. |
| Verifier | V1 rederives only root semantics and pair dispositions. `K_b`, recursive replies, and DAG reuse are introduced only by the “before a future full-tree version” obligation. |
| Limits/metrics | The document records future DAG/depth/state metrics as ceilings and planning constraints, but no leaf acceptance rule consumes a future node. Their presence grants no format tag or implementation authority. |
| Gates/evidence | Historical full-tree witnesses are evidence rows only and are removed from v1 emission gates. A future format must pin its own evidence and return to design review. |
| Consumers | There is no v1 consumer. Full-tree and leaf facts alike remain incapable of changing the game value without a separate owner decision and design. |

After R2-1 and R2-2 are incorporated literally, no remaining round-1
counterexample or fresh leaf attack blocks launching the
`RefuteLeafExact/V1` implementation lane under the design's preregistered
gates. The final adoption decision remains the owner's. Full-tree remains
outside that authorization.
