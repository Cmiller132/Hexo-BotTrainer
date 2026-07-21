# Certified `vcf_pair_complete` refutations: v1 design

Status: design only, 2026-07-21. This document proposes a new artifact and a
separate verifier arm. It does not authorize changes to search policy, the
trainer, the positive certificate grammar, or `tss_verify.rs`.

**HYPOTHESIS — evidence discipline.** In this document, **CODE-FACT** means a
fact directly visible in the current Rust implementation and is accompanied by
a current line reference. Every proposed rule, semantic bridge, estimate, gate,
and Lean statement is labelled **HYPOTHESIS**. In particular, naming a Lean
target below is not a claim that the target has been implemented, proved, or
connected to the executed Rust bytes.

## 1. Claim semantics

### 1.1 The exact claim

**HYPOTHESIS.** Fix a reachable, nonterminal, post-opening engine state `P` in
`FirstStone`, let `A = P.current_player()`, and let
`nextPly = P.placements_made() + 1`. The v1 artifact certifies exactly:

```text
NoContractWin VcfPairComplete P nextPly

meaning: there is no finite positive contract tree, at any finite
resolution clock, whose attacker turns obey VcfPairComplete and whose
defender coverage is licensed by the baseline forced-hit rules.
```

**HYPOTHESIS.** Equivalently, in a horizon-indexed presentation, the accepted artifact proves
the negative proposition for every finite semantic horizon at or after
`nextPly`; it does not merely refute one selected horizon. `u32::MAX` is the
producer-profile marker, not a mathematical infinity witness.

**HYPOTHESIS.** The v1 root hypotheses are deliberately narrower than every
state the wide engine can search:

- the root is nonterminal and its exact phase is `FirstStone`;
- the claimant is the current player, not the other player and not a colour
  inferred from placement parity;
- the root is post-opening and its phase/placement clock is internally
  consistent;
- the width profile is exactly `vcf_pair_complete` with its proven-exact
  `T(P)`/`S(P,a)` characterization;
- the semantic profile is unbounded, Group-2 and certificate-relative zones
  are absent, and no census dismissal participates; and
- every leaf in the negative support is structural. No cap, staged-depth,
  semantic-horizon, or verifier-depth refusal is admissible evidence.

**HYPOTHESIS.** This root-phase refinement is the v1 manageability cut. It covers the supplied
`l9mxn59` and `mvp2lvc` witnesses and the ordinary fresh-turn corpus rows. A
root in `Opening` or claimant `SecondStone` remains `Unknown` to this artifact
arm in v1. Defender `SecondStone` states reached while replaying a selected
counterturn are supported.

**CODE-FACT.** Player identity, not depth parity, must control polarity:
`FirstStone -> SecondStone` keeps the mover, whereas completing `SecondStone`
changes it
([`state.rs` lines 317-334](../packages/hexo_engine/rust/src/state.rs#L317)).
The hard-result seam also defines `Loss` as a real opponent winning strategy
and leaves exhausted or unproven work as `Unknown`
([`tss_core.rs` lines 24-43](../packages/hexfield_eq/rust/src/tss_core.rs#L24)).

**HYPOTHESIS.** Therefore this artifact is never a `ProofStatus::Loss`, never a
`HardValue(-1)`, and never evidence that the other player wins. A position may
satisfy `NoContractWin VcfPairComplete` while the same player wins through a
quiet, free-tempo, or otherwise out-of-class strategy. It may also be a draw or
a full-game win for either side. Any conversion of this class fact to a
full-game `Loss` is a soundness bug and kills v1.

### 1.2 Clock and “natural exhaustion”

**CODE-FACT.** The wide engine derives its final search depth from the semantic
horizon but clamps it to `MAX_SEARCH_DEPTH`
([`tss_solver.rs` lines 2555-2561](../packages/hexfield_eq/rust/src/tss_solver.rs#L2555)).
A depth cutoff and a structural refutation currently both receive disproof
numbers `(infinity, 0)`
([lines 5935-5967](../packages/hexfield_eq/rust/src/tss_solver.rs#L5935)).
The expansion code also maps semantic-horizon refusal, opponent tactical facts,
unsupported defender boundaries, census dismissal, and empty child lists into
the same `WidePnNode::Refuted` representation
([lines 6341-6497](../packages/hexfield_eq/rust/src/tss_solver.rs#L6341)).

**HYPOTHESIS.** Consequently `root.dn == 0` is necessary but not sufficient for
v1 emission. A **natural width exhaust** is a root with `dn == 0` for which a
complete negative support can be selected without traversing any
`DepthCutoff`, horizon refusal, census dismissal, cap boundary, stalled edge,
or other resource-derived condition. The producer additionally requires
`semantic_horizon == u32::MAX`, `expansions < node_cap`, and the exact v1
profile. Reaching `dn == 0` on the last permitted expansion is rejected by the
strict v1 trigger even if a later independent replay might establish the same
fact; this conservative loss of coverage keeps “natural” mechanically clear.

**HYPOTHESIS.** The artifact itself contains no `StructuralBoundary` leaf. An
accepted negative DAG is therefore horizon-parametric: increasing a semantic
horizon cannot invalidate a forcing-gate failure, an exact empty attacker
universe, a legal defender counterline, or an opponent terminal fact. A
verifier recursion/byte cap is only a reason to reject an oversized artifact;
it is never a proposition inside an accepted artifact.

### 1.3 Lean statement family

**HYPOTHESIS — Lean targets only.** The design is intended to specialize the
existing `ContractWin` / `NoContractWin` statement family, not the full-game
`AttackerWinsBy` proposition. The target theorem names and direction are:

1. `refuteCertV1_accepts_implies_noContractWin`: acceptance of the exact
   replayed v1 bytes implies `NoContractWin VcfPairComplete P nextPly`.
2. `refuteCertV1_accepts_implies_noContractWin_at`: for every exact finite
   horizon query for the same root, acceptance implies the query-bound
   `NoContractWin` proposition.
3. `noContractWin_of_noJointCarrier` and
   `noContractWin_of_noAdmissibleFirstTurn`: the optional compact leaves imply
   the same class-relative negative proposition under the side conditions in
   section 2.4.
4. `widePnNaturalExhaust_materializes_acceptedRefuteCertV1`: a later Rust/Lean
   correspondence target connecting the producer's eligible `dn == 0` support
   to accepted bytes.

**HYPOTHESIS.** The generic negative-tree capstone name
`checkNoDagBytes_implies_noContractWin` is the intended model-side precedent.
V1 still needs a specialization to the literal `VcfPairComplete`
regeneration, a proof that the wire compression expands to the negative
grammar, and a structural-boundary-free all-horizons corollary. None of those
bridges is claimed by this design, and no Lean source belongs in the v1 Rust
build round.

## 2. Certificate grammar

### 2.1 Logical tree and compact wire form

**HYPOTHESIS.** The logical certificate is the polarity dual of a positive
strategy tree. Its uncompressed grammar is:

```text
RefuteNode :=
    ChoiceExhausted(FirstRow*)
  | UniversalCounterexample(reply, EdgeNo)
  | NoPositiveConstructor(reason)
  | OpponentTerminal(winner)
  | OpponentTactical
  | NoJointCarrier
  | NoAdmissibleFirstTurn

FirstRow := FirstStone(a, PairDisposition for every b in S(P,a))

PairDisposition :=
    FailsForcingGate(reason)
  | DefenderCoverSurvives(UniversalCounterexample)

EdgeNo := ChildNo(RefuteNode) | OpponentTerminal(winner)
```

**HYPOTHESIS.** `ChoiceExhausted` is valid only when the claimant moves. It reconstructs the
exact first-stone universe `T(P)`, reconstructs `S(P,a)` for every `a`, and
gives one negative disposition for every resulting ordered pair. A passing
unordered pair appears only once after the verifier proves the same-turn
quotient side conditions. Every other ordered occurrence maps either to that
same exact final position or to a `FailsForcingGate` leaf.

**HYPOTHESIS.** The wire form is a versioned, backward-referenced DAG encoding
of this logical tree. It stores only forcing-gate-passing canonical pairs and
their negative dispositions. The verifier regenerates all of `T`, every `S`,
and every pair classification; the omitted complement expands uniquely to
`FailsForcingGate` leaves. A node stores the regenerated counts as redundant
assertions, and the verifier requires both exact counts and exact set equality.
This complement encoding avoids one coordinate record per rejected pair while
retaining the full logical coverage claim.

**HYPOTHESIS.** The file header contains:

- magic, format version, and exact class identifier
  `NoContractWin/VcfPairComplete/V1`;
- the complete root binding: sorted `(q,r,owner)` stones, current player,
  `FirstStone`, placements made, nonterminal marker, and a digest of those
  canonical bytes;
- claimant identity, which must equal the root mover;
- the fixed profile bits `pair_complete`, `baseline_t6_kernel`, `unbounded`,
  `no_group2`, `no_zone`, and `no_census`;
- hard counts for nodes, negative dispositions, and root ID; and
- a checksum over the payload for corruption detection. The checksum is not a
  proof identity; exact decoded fields and replay are authoritative.

**HYPOTHESIS.** Node IDs are backward-only. Shared nodes are legal only when every incoming
path independently reconstructs the same full semantic state, claimant,
phase—including a `SecondStone.first` witness—and placement clock. Orphans,
forward references, duplicate propositions, and cycles are rejected. The
logical object remains a finite tree; DAG sharing is serialization only.

**HYPOTHESIS.** `NoPositiveConstructor` is a narrow structural leaf, not a
catch-all `Refuted` tag. It is accepted only when direct replay finds no
claimant terminal/tactical state leaf and the current nonclaimant node is not a
tight defender dispatcher, so the positive `VcfPairComplete` grammar has no
Universal constructor there. A producer resource refusal can never use this
tag.

### 2.2 Exact attacker coverage and pair leaves

**CODE-FACT.** In wide mode the current first-candidate generator takes empties
from claimant-pure count-at-least-two windows and adds empties of live defender
count-at-least-four windows
([`tss_solver.rs` lines 8961-9046](../packages/hexfield_eq/rust/src/tss_solver.rs#L8961),
[`lines 9063-9076`](../packages/hexfield_eq/rust/src/tss_solver.rs#L9063)).
For a first stone, `second_candidates` constructs the turn-start candidates,
the stronger same-window promotions, and the count-one promotions
([lines 9384-9444](../packages/hexfield_eq/rust/src/tss_solver.rs#L9384)).
This is the executable shape of
`S(P,a) = (T(P) - {a}) union G1(P,a)`.

**HYPOTHESIS.** The independent verifier defines `T` and `S` from direct
six-cell window scans, not by importing either generator. It canonicalizes the
finite ordered pair universe as follows:

1. enumerate every `a` in `T(P)` in raw `(q,r)` order;
2. enumerate every `b` in `S(P,a)` in raw order and check both placements are
   legal at the turn start;
3. if both singleton prefixes are nonwinning and both orders reach the same
   full state, map `(a,b)` and `(b,a)` to the lexicographically smaller
   unordered key; otherwise keep the ordered edge distinct; and
4. require the stored passing-pair keys to equal exactly the keys whose
   independently rederived forcing gate passes.

**HYPOTHESIS.** No candidate is omitted because the producer did not happen to generate it;
producer/verifier set equality is the coverage proof.

**CODE-FACT.** The current gate admits a pair only after building its post-pair
claimant threat family, excluding an unblocked defender win-now, and finding
minimum hitting number two or no two-cell cover
([`tss_solver.rs` lines 9446-9558](../packages/hexfield_eq/rust/src/tss_solver.rs#L9446)).
The producer deduplicates the two legal coordinate orders only after successful
classification
([lines 6772-6805](../packages/hexfield_eq/rust/src/tss_solver.rs#L6772)).

**HYPOTHESIS.** The logical `FailsForcingGate` leaf has one of three canonical
reasons, chosen in this order so the wire has one representation:

1. `NoNewClaimantThreat`: the pair creates no post-pair live claimant
   count-four-or-better family;
2. `DefenderWinsFirst`: at least one live defender count-four/count-five
   window remains unhit, so the defender has own win-now; or
3. `LooseReply`: the new claimant threat family has a hitting set of size zero
   or one, so the pair does not consume the defender's two-stone turn.

**HYPOTHESIS.** The verifier derives the reason. It does not trust a producer reason tag. A
pair with no hitting set of size at most two is a positive tactical constructor
under this contract, not a negative leaf. Encountering such a pair in a
purported root refutation makes the verifier reject because the claimant has an
admitted local win constructor.

**HYPOTHESIS.** A `NoAdmissiblePair(a)` wire run is the compact form for a whole
`S(P,a)` row in which every pair fails one of those gates. A
`NoAdmissibleFirstTurn` node compresses all rows only when the verifier obtains
an empty set of gate-passing canonical pairs for the entire root. These are
compressions of exact regeneration, not trusted producer summaries.

### 2.3 Defender Universal nodes and surviving covers

**CODE-FACT.** The current PN recurrence already has the required negative
polarity. At an attacker `Choice`, `dn` is the sum of every child disproof
number; at a defender `Universal`, `dn` is the minimum child disproof number
([`tss_solver.rs` lines 5944-5967](../packages/hexfield_eq/rust/src/tss_solver.rs#L5944)).
Thus an attacker Choice is refuted only when all admitted attacker turns are
refuted, whereas one defender counterexample refutes a positive Universal.

**HYPOTHESIS.** A v1 `UniversalCounterexample` therefore stores exactly one
legal defender reply plus its negative child. Requiring every defender reply's
subtree to be refuted would be a strictly stronger claim, would reverse the
dual recurrence, and would inflate the artifact for no soundness benefit. The
verifier nevertheless reconstructs the complete positive Universal reply set
to prove that the selected reply belongs to it.

**CODE-FACT.** At a forced boundary the current wide generator uses the
extendable-hit kernel: a cell belongs to `K_b` exactly when the residual threat
family can be hit with at most `b-1` further cells
([`tss_solver.rs` lines 9723-9789](../packages/hexfield_eq/rust/src/tss_solver.rs#L9723)).
The two-stone optimization is all-or-nothing and falls back unless every first
reply remains at an exact forced boundary and both legal orders reach the same
final position
([lines 3388-3421](../packages/hexfield_eq/rust/src/tss_solver.rs#L3388)).

**HYPOTHESIS.** The verifier accepts a defender counter node only after it
independently establishes all of the following:

- defender to move, nonterminal, post-opening, and no defender own win-now;
- the claimant threat family is nonempty and its exact hitting number equals
  the phase budget `b`;
- the complete positive reply set is the independently enumerated `K_b`;
- the selected reply is legal and is a member of `K_b`; and
- replaying the selected reply yields exactly the state claimed by its child.

**HYPOTHESIS.** At defender `FirstStone`, the usual selected path is a two-step surviving
cover: choose `d1 in K_2`, then at defender `SecondStone` choose
`d2 in K_1` for the residual family, then recursively refute the claimant's
next `FirstStone` node. The wire uses two ordinary Universal-counterexample
nodes. It deliberately unfolds the search's atomic `DefenderPair`; v1 needs no
commutation witness or zone annotation on the refutation side. An opponent
terminal reached by a selected reply is a typed `OpponentTerminal` leaf.

**HYPOTHESIS.** T6 is treated as the baseline forced-hit theorem, not as a
certificate-relative zone. The proof document describes `K_b` as all legal
replies when the hitting number is below budget and as a possible strict
refinement at equality (`PROOF_TSS_DEFENDER_ZONES.md` section 6, T6). V1 accepts
only the equality boundary used by the pair-complete forcing gate. Group-2,
ranked zones, FHW gates, substitute replies, and unforced-turn quotients are
format errors, not ignored extensions.

### 2.4 Optional compact base leaves

**HYPOTHESIS — `NoJointCarrier`.** This optional leaf is valid only at a
nonterminal claimant `FirstStone` node with no claimant own win-now. The
verifier directly scans the current claimant-pure windows and establishes all
three side conditions from `RESEARCH_DIVERGENCE_1.md` section 7.1:

1. fewer than two claimant count-three windows exist;
2. no count-three/count-two pair has intersecting empty sets; and
3. no two distinct count-two windows share an unordered pair of empty cells.

**HYPOTHESIS.** Under those conditions no pair can create the two distinct post-pair threats
required by an admitted `vcf_pair_complete` turn. The leaf proves only
`NoContractWin VcfPairComplete`. It is optional because its verifier logic is
different from ordinary exact enumeration and because it did not hit the
measured grind roots in the cited research.

**HYPOTHESIS — `NoAdmissibleFirstTurn`.** This optional leaf is valid only at a
nonterminal claimant `FirstStone` node with no claimant own win-now. The
verifier independently enumerates exact `T(P)` and every exact `S(P,a)` and
checks that every pair fails at least one of: creation of a new claimant threat,
answering defender win-now, or hitting number at least two. It is equivalent to
an empty gate-passing pair list and may be used at any recursive fresh-turn
node, not just the artifact root.

**HYPOTHESIS.** Neither compact leaf is licensed by producer detection alone.
The producer may request the tag, but the verifier reruns the entire predicate.
Any mismatch rejects the file. If either predicate ever holds while ordinary
independent regeneration finds a passing pair, the leaf and its producer fast
path are removed.

### 2.5 D6 handling and graph bounds

**CODE-FACT.** The current search uses the lexicographically least of twelve
D6 images only for tie ordering; its TT retains exact raw-position equality
([`tss_solver.rs` lines 10318-10370](../packages/hexfield_eq/rust/src/tss_solver.rs#L10318)).

**HYPOTHESIS.** V1 likewise performs no symmetry quotient in the proof
identity. Root stones, pair coordinates, and selected defender replies are raw
axial coordinates. A certificate transformed by any of the twelve D6 actions
must transform every coordinate, rebuild canonical list order and counts, bind
to the transformed exact root, and verify. The original bytes must fail against
the transformed root. No value-bearing cache or shared DAG node crosses a D6
image merely because a hash or canonical image matches.

**HYPOTHESIS.** Initial hard parser limits are 100,000 DAG nodes, 1,000,000
negative dispositions, depth 256, 1,000,000 root stones, and 8 MiB of bytes.
All count arithmetic is checked before allocation. These are rejection limits,
not semantic refutation leaves. The hostile build round may lower them after
measuring the frozen cohorts; raising them requires another memory review.

### 2.6 Size model and the two witnesses

**HYPOTHESIS.** Let:

- `R` be root stones;
- `A` be fresh-turn attacker nodes in the negative support;
- `Q` be the number of logical ordered `(a,b)` pairs independently checked;
- `K` be gate-passing canonical pair dispositions stored in the file;
- `U` be stored defender counterexample nodes; and
- `L` be terminal or compact base leaves.

**HYPOTHESIS.** With fixed-width root coordinates and varint IDs/counts, the planning model is

```text
bytes = 64 + 5R
      + sum_AttackNode(3 + 10..12 * passing_pairs)
      + sum_UniversalNode(6..8)
      + sum_Leaf(1..5)
      + checksum/padding.
```

**HYPOTHESIS.** The logical size is `O(Q + U + L)`. Complement encoding makes serialized size
`O(A + K + U + L)`, while replay time remains `O(Q + defender-family work)`.
DAG sharing is by full replay state and can only reduce the tree size.

**HYPOTHESIS — estimate, not measurement.** On naturally exhausted trees,
each stored passing pair or selected defender reply normally corresponds to an
expanded arena obligation. Using a planning band of 10-16 serialized bytes per
reported exhausted node, plus the root binding, gives:

| witness | reported exhausted nodes | root stones | planning estimate |
|---|---:|---:|---:|
| `l9mxn59` | 226 | 17 | 2.4-3.7 KiB |
| `mvp2lvc` | 17,957 | 45 | 176-281 KiB |

**HYPOTHESIS.** These figures are not allowed to become implementation folklore. The producer
must report `A`, `Q`, `K`, `U`, `L`, unique DAG nodes, encoded bytes, and the
corresponding minimal stored-search snapshot bytes. Section 6 makes the
measured ratios gates.

## 3. Producer

### 3.1 Existing arena facts and the emission seam

**CODE-FACT.** `WidePnEntry` retains a node, proof/disproof numbers, depth, and
child vectors; each child retains its move, result, optional entry/future key,
and prior
([`tss_solver.rs` lines 2782-2815](../packages/hexfield_eq/rust/src/tss_solver.rs#L2782),
[`lines 3026-3069`](../packages/hexfield_eq/rust/src/tss_solver.rs#L3026)).
The production wide call inserts the root, runs the search, and still has the
whole arena available before materialization
([lines 1472-1503](../packages/hexfield_eq/rust/src/tss_solver.rs#L1472)).
The current materializer immediately refuses unless the root has `pn == 0`
([lines 7147-7168](../packages/hexfield_eq/rust/src/tss_solver.rs#L7147)), so a
`dn == 0` arena produces no artifact.

**HYPOTHESIS.** The owner-gated build should add a sibling
`materialize_refutation_v1` call after `search.run` and before the arena is
dropped. It must not route through the positive `TssCertificate` or
`AttemptResult.cert` field. The result is an optional side artifact with its
own type and verifier result; the ordinary deep result remains
`ProofStatus::Unknown`.

**HYPOTHESIS.** The call is gated by a solve-local, read-once
`TSS_REFUTE_CERT_V1=off|emit` flag. `off` takes the historical path without a
new arena allocation, child reorder, stat mutation, digest field, cache entry,
or output byte. `emit` does no extra work until the ordinary search terminates
and the cheap eligibility predicate succeeds. Thus non-exhaust solves do not
pay for a shadow provenance sidecar.

### 3.2 Eligibility and materialization algorithm

**CODE-FACT.** The search loop stops when the root has either zero proof or
zero disproof number, when the node cap binds, or when staged deepening cannot
continue
([`tss_solver.rs` lines 4529-4568](../packages/hexfield_eq/rust/src/tss_solver.rs#L4529),
[`lines 4621-4661`](../packages/hexfield_eq/rust/src/tss_solver.rs#L4621)).
Production currently exposes root proof/disproof numbers only in test-oriented
paths and telemetry, while the wide positive result is built later from
`search.materialize`
([lines 1490-1503](../packages/hexfield_eq/rust/src/tss_solver.rs#L1490),
[`lines 1565-1594`](../packages/hexfield_eq/rust/src/tss_solver.rs#L1565)).

**HYPOTHESIS.** Retain one local termination enum until the post-search seam:
`Proven`, `NaturalExhaust`, `NodeCap`, `DepthBoundary`, `HorizonBoundary`, or
`Stalled`. This value is producer telemetry, never verifier evidence.
`NaturalExhaust` requires all section 1.2 conditions and an independently
selected support free of forbidden boundaries.

**HYPOTHESIS.** Materialization is a deterministic replay from the exact root:

1. At a claimant `Choice` with `dn == 0`, regenerate the producer's complete
   pair list, key it by the artifact's canonical pair key, and require every
   gate-passing pair to have a genuinely refuted child. Recursively materialize
   all such children. Recompute gate-failing rows only to choose the compact
   wire form.
2. At a defender `Universal` with `dn == 0`, select the lexicographically least
   genuinely refuted reply, replay it, and recursively materialize that one
   child. A child whose only zero came from `DepthCutoff` is not genuine; the
   current helper already distinguishes that case
   ([`tss_solver.rs` lines 5908-5921](../packages/hexfield_eq/rust/src/tss_solver.rs#L5908)).
3. If the search child is an atomic `DefenderPair`, replay its two placements
   and emit two ordinary Universal-counter nodes. Do not copy the positive
   commutation representation.
4. Reclassify every terminal `Refuted` arena state by replay. Accept only a
   structural gate failure, exact empty attacker set, opponent terminal, or
   independently materializable recursive refutation. A horizon/census/depth
   cause makes materialization return `None`.
5. Memoize by the full exact position key plus claimant. A repeated state may
   reuse an already built backward node; a hash alone never authorizes reuse.
6. Sort records canonically, encode them, call the independent v1 verifier, and
   emit bytes only if that verifier accepts. Verifier rejection drops the side
   artifact and increments diagnostic telemetry; it cannot alter the ordinary
   `Unknown` result.

**HYPOTHESIS.** Facts currently discarded but needed at the seam are only the
root numbers/termination classification and the still-live arena itself. The
producer does **not** need to retain every gate failure or add a refutation
cause to every hot-path node: exact pairs and causes can be regenerated after
natural exhaustion. Direct defender terminal edges already retain their move;
replay recovers the outcome. This choice is the main reason v1 can have zero
non-exhaust hot-path memory cost.

### 3.3 Cost and isolation budget

**HYPOTHESIS.** Let the logical exhausted support include all virtual
gate-failure leaves. Producer emission is one deterministic pass over that
support: `O(Q + K + U + L)` time and `O(K + U + L)` additional memory. It may
not invoke PN search, deepen a cutoff, expand a new recursive position, consult
a persistent proof/refutation cache, or change a child result. Regeneration for
encoding is allowed; new search is not.

**HYPOTHESIS.** Initial resource targets are:

- peak additional producer heap at most 32 bytes per unique emitted DAG node
  plus the final byte buffer, and below 16 MiB for `mvp2lvc`;
- emission wall excluded from search-node telemetry and separately reported;
- flag-off status, positive-certificate bytes, stats, node counts, TT
  signatures, and corpus output are bit-identical; and
- with emission enabled, every non-natural result has identical search
  behavior and no materialization scan.

**HYPOTHESIS.** Any need to retain window snapshots, gate-failure vectors, or refutation
subtrees during ordinary search is a scope alarm. The build returns to hostile
design review before accepting such a regression.

## 4. Independent verifier arm

### 4.1 Module and API boundary

**HYPOTHESIS.** V1 lives in a new module such as
`tss_refute_verify.rs`, with its wire types either in that module or a small
`tss_refute_cert.rs`. It must not modify or add variants to `tss_verify.rs`.
Its public result is a typed class fact, for example
`VerifiedClassRefutation { class: VcfPairComplete, root_digest }`, not
`ProofStatus`, `HardValue`, or `TssCertificate`.

**HYPOTHESIS.** The firewall is stricter than ordinary code reuse:

- forbidden imports/calls include `WidePnSearch`, `WidthOptions`,
  `ordered_threat_creating_moves_with_width`, `WideTurnGate`,
  `wide_family_min_hitting_set`, `forced_defender_replies`,
  `extendable_hit_kernel`, `forced_defender_pair_plan`, producer canonical
  ordering, and any producer-only window summary;
- allowed dependencies are engine state/placement primitives, coordinate and
  player types, a local checksum implementation/library, and separately
  audited direct geometry; and
- the verifier implements its own legal-move check, direct six-cell window
  enumeration, count classification, hitting-set-at-most-two search, pair
  quotient check, and D6 transforms.

**HYPOTHESIS.** Even apparently harmless shared `T(P)` or hitting-set helpers violate this v1
firewall. If duplication becomes unmanageable, the answer is to shrink the
scope, not to merge producer and verifier truth sources.

### 4.2 Re-derivation obligations

**HYPOTHESIS.** After strict decoding and exact root binding, the verifier
performs these obligations from scratch:

1. Enumerate all potentially live length-six windows by taking the 18 windows
   through every occupied cell, deduplicate their `(axis,start)` keys, and
   recount ownership directly from the board.
2. Reconstruct `T(P)` from claimant count-at-least-two windows and live
   defender count-at-least-four windows.
3. For every first stone reconstruct `G1(P,a)` and exact `S(P,a)`; check
   turn-start legality and the same-turn quotient rather than trusting the
   stored pair key.
4. Apply both attacker placements through the engine and independently check
   the new claimant threat family, defender own win-now, and exact hitting
   number. Compare the complete passing-pair set with the stored dispositions.
5. At every defender counter node directly derive the claimant threat family,
   budget, `not own_win_now`, exact `tau == b`, and complete `K_b`; check the
   selected reply and replay its successor.
6. Recompute every optional base predicate from direct window facts.
7. Before accepting any negative node, prove that no claimant terminal,
   lambda-one, or other admitted local positive constructor exists at that
   exact state and clock. An opponent tactical fact must be independently
   established before accepting `OpponentTactical`.
8. Reconstruct the full state reached at every DAG node and require equality on
   every reuse. Verify phase and placement-clock transitions per placement.
9. Expand the accepted DAG conceptually into the logical polarity-dual tree and
   conclude the class-relative negative proposition by well-founded induction.

**HYPOTHESIS.** The verifier does not have to enumerate every legal defender
cell outside `K_b`: the baseline T6 premise proves that an off-kernel reply
cannot be a required positive Universal edge at the equality boundary. It does
have to derive the complete finite `K_b` and the premise independently. V1 does
not accept the `mhs < b` case, where T6 says the kernel becomes the full legal
set.

### 4.3 Fail-closed rules

**HYPOTHESIS.** Verification returns false on the first of:

- bad magic/version/class/profile, unknown tag or noncanonical varint;
- trailing bytes, checksum mismatch, count overflow, bound excess, or failed
  allocation preflight;
- root mismatch in any stone owner, mover, phase, placement count, terminal
  record, claimant, or digest;
- an Opening/SecondStone root, terminal root, wrong claimant, or inconsistent
  phase schedule;
- duplicate/unsorted pair records, duplicate replies, bad node IDs, forward
  references, cycles, or orphans;
- any regenerated-set/count mismatch;
- an illegal placement, hidden terminal prefix, wrong full successor state, or
  unequal pair quotient;
- a claimed gate failure that passes, a missing passing pair, an extra pair, or
  a `tau > 2` positive tactical pair;
- any exact state with a claimant terminal/tactical positive constructor, or a
  false `NoPositiveConstructor`/`OpponentTactical` classification;
- a defender counter outside exact `K_b`, wrong budget, own win-now, or a loose
  `tau < b` boundary;
- a false compact base predicate;
- a depth/horizon/cap/census/zone/Group-2/quotient boundary tag; or
- a DAG reuse at a merely hashed, D6-related, or otherwise nonidentical state.

**HYPOTHESIS.** Malformed artifacts never panic, allocate from unchecked counts, partially
accept, or degrade to a weaker class claim.

### 4.4 Replay-cost bar

**HYPOTHESIS.** Replay must beat rerunning the search on the same frozen
natural-exhaust cohort. The preregistered measurement is a quiet, single-thread,
release binary with warmed code pages, fresh solver state for each search, and
at least 30 batched repetitions per small witness. The hard gates are:

- total verifier wall over all emitted cohort artifacts is less than 75% of
  total cold rerun search wall;
- for each of `l9mxn59` and `mvp2lvc`, median verifier wall is strictly below
  median cold rerun wall, with a target at or below 50%;
- verifier p95 is below the matching rerun p95 on the cohort; and
- each DAG node is replayed at most once per exact proposition, while every
  logical `(a,b)` gate is evaluated at most once.

**HYPOTHESIS.** The economic design is killed if aggregate replay is not strictly faster than
rerun, even if individual noisy sub-millisecond rows fluctuate. A verifier that
secretly invokes search automatically fails this gate.

## 5. Consumption roadmap (post-v1, informational)

**HYPOTHESIS.** V1 has no consumer. After artifact and verifier gates pass, the
following are separate owner decisions:

1. **Trainer width-exhaust backup.** Keep the ordinary scalar result
   `Unknown`; attach a separate categorical
   `NoContractWin(VcfPairComplete)` fact or auxiliary target. It must not enter
   `backup_virtual` as `-1`, force an opponent move, or rewrite game value.
2. **Atlas NO-side.** Display and query a distinct
   `CERTIFIED-NO(VcfPairComplete)` state with artifact hash and verifier
   version. Do not merge it with full-game `NO`/`Loss`.
3. **Corpus NO rows.** Add a certification column to NO-labelled rows and
   preserve the original reference label. This can distinguish a class theorem
   from an uncertified heuristic `No` without claiming equivalence.
4. **Harness disproof coverage.** Populate the existing width-exhaust/disproof
   metric with `eligible`, `emitted`, `accepted`, bytes, emit wall, and replay
   wall. Keep cap-bound Unknowns in a separate denominator.
5. **Lean/Rust correspondence.** Only after the byte grammar is frozen and the
   Rust arm survives hostile review, connect the literal decoder/checker to the
   target theorem family in section 1.3.

**HYPOTHESIS.** Cache reuse, search pruning, atlas-to-solver imports, and trainer consumption
are explicitly post-v1. A verified refutation may be stored as an artifact, but
the v1 search never trusts one to close a node.

## 6. Gates and kill criteria

**HYPOTHESIS.** Before implementation measurements, freeze manifests and hashes
for the 248 grind roots, the forcing/puzzle NO rows, the human-160 residue, and
the two named witnesses. Record the exact solver flags, binary revision,
node/TT caps, and which rows are natural exhaust versus cap-bound. Moving a row
between cohorts after results are visible invalidates the gate.

**HYPOTHESIS.** The build must pass every gate below:

| gate | required result | hard stop |
|---|---|---|
| Eligibility | 100% of roots classified as v1 natural exhaust before the build either emit an artifact or have a pre-registered unsupported root-phase reason; `l9mxn59` and `mvp2lvc` both emit. | Any eligible FirstStone natural exhaust silently falls back because materialization cannot explain a `Refuted` node. |
| Acceptance | The independent verifier accepts 100% of emitted artifacts on every frozen cohort. Producer self-verification uses the same public entry point as offline replay. | Any emitted artifact rejection. |
| No false scope | No cap-bound, finite-horizon, depth-cutoff, census, Group-2, zone, Opening-root, or claimant-SecondStone-root attempt emits. | One such emission. |
| Class boundary | The ordinary solve status stays `Unknown`; no `HardValue`, `Loss`, trainer backup, or full-game label is minted. | Any exposure as game value. |
| Flag-off identity | Statuses, positive certificate bytes/digests, stats, node counts, TT behavior signatures, logs, and corpus outputs are byte-identical with the feature compiled and `off`. | One unexplained byte or behavior difference. |
| Enabled non-exhaust identity | With `emit`, non-natural searches have identical nodes, selected paths, and positive outputs and perform no post-search tree scan. | Any search regression or result change. |
| Mutation rejection | 100% rejection for dropped/extra pair, changed root owner/phase/claimant, changed reply, bad count/ID/order, false compact leaf, injected boundary/zone tag, cycle/orphan, checksum-only repair, and D6 root mismatch. | One accepted semantic mutation. |
| D6 | Each accepted artifact transformed and re-canonicalized under all twelve symmetries verifies against the transformed root; original bytes fail on a different image. | Any asymmetric semantic result or cross-root acceptance. |
| Size density | Cohort median at most 16 bytes and p95 at most 24 bytes per exhausted search node; `l9mxn59 <= 4 KiB`, `mvp2lvc <= 320 KiB`. | Either witness exceeds twice its bar, or cohort p95 exceeds 32 bytes/node. |
| Stored-search comparison | Artifact bytes are at most 50% of a predeclared `StoredSearchV0` serialization of the exact negative support (full replay keys, moves, kinds, and child links) on aggregate and on both named witnesses. | Artifact is no smaller than the stored-search alternative on aggregate. |
| Producer memory | Additional peak heap follows section 3.3 and remains below 16 MiB on `mvp2lvc`. | Unbounded growth, unchecked allocation, or a hot-path retention requirement. |
| Replay cost | All section 4.4 hard bars pass. | Aggregate verifier wall is at least rerun wall, either named witness median does not improve, or search is called by verification. |
| Firewall | A source audit finds no forbidden solver/generator import and direct tests compare independent `T`, `S`, gate, and `K_b` results with the producer on seeded positions. | Shared truth helper, shared window summary, or unexplained differential. |

**HYPOTHESIS.** Soundness stops are absolute: a false acceptance, set-coverage
mismatch, structural-boundary acceptance, full-game `Loss` exposure, or
producer/verifier shared truth helper kills the design rather than merely the
current implementation. Economic stops kill **full-tree v1** if artifacts fail
both the stored-search size advantage and replay-speed advantage after one
codec-only optimization round. Do not keep a larger, slower artifact merely
because it is interesting.

## 7. Manageability verdict

**HYPOTHESIS.** Under the strict scope in section 1, the expected owner-gated
implementation size is:

| component | estimated LOC |
|---|---:|
| producer trigger, negative materializer, canonical compaction, telemetry | 400-550 |
| v1 wire types and strict codec | 250-350 |
| separate independent verifier arm | 900-1,200 |
| unit, mutation, D6, differential, corpus, and cost tests | 700-950 |
| total Rust/test change | 2,250-3,050 |

**HYPOTHESIS.** No estimate includes a trainer consumer, cache, Group-2/zone rule, arbitrary
root-phase support, `tss_verify.rs` edit, or Lean proof.

**HYPOTHESIS.** Three review rounds are required after this design:

1. hostile semantics/grammar review, especially the all-horizons claim,
   attacker-pair quotient, and Universal polarity;
2. implementation/firewall review with mutation and direct-regeneration
   differentials; and
3. owner gate on frozen-cohort acceptance, flag identity, size, memory, and
   replay economics.

**HYPOTHESIS.** The later Lean/Rust correspondence is a fourth, separately owned proof round,
not a condition for starting the artifact-only Rust prototype.

**HYPOTHESIS — verdict: GO, conditional on the v1 cut.** A roughly 2.5k LOC
artifact/verifier addition is manageable because it replays an already closed
arena, uses the polarity-dual recurrence already present in PN, unfolds
defender-pair optimization, forbids all zones and boundary leaves, and has no
v1 consumer. The design becomes **TOO BIG** if the first build is asked to
support arbitrary phases, bounded-horizon negatives, Group-2/FHW/ranked zones,
unforced defender quotients, refutation caches, or trainer backup; those change
the proposition and trusted surface rather than merely the codec.

**HYPOTHESIS — fallback.** If full-tree v1 is killed by verifier complexity or
economics, the smallest sub-scope that still pays is leaf-only certification
for one-expansion fresh-turn exhausts:

- `NoAdmissibleFirstTurn` as the mandatory exact leaf; and
- `NoJointCarrier` only if it is smaller/faster than exact enumeration and has
  zero shadow contradictions.

**HYPOTHESIS.** That fallback will not certify `l9mxn59` or `mvp2lvc`, but it preserves the
class-relative theorem, independent-verifier firewall, and future wire namespace
without pretending a partial negative tree is a complete refutation.
