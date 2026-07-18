# NoTssCertificate v1: exact-root exhaustion artifact and checker

**[DOCUMENT STATUS]** AUTHORITATIVE CANDIDATE-C DESIGN, NOT AN IMPLEMENTATION OR PROOF

**[DOCUMENT STATUS]** Engine snapshot: `hunt/completeness` at `ae034425fe0d7fd506b58bd182ccd215a2362461`.

**[LEAN FACT]** Spine snapshot inspected read-only: `E:\tss-lean` at committed HEAD `ad82f2226b2c09a546647686dc780cdf3e43083a`, plus the live S9S-38 edits to `LEDGER.md` and `TssZones/ForcedHit.lean`; the live ledger still classifies L17, T11, T11.1, and the full T6 capstone as `UNSTATED` (`E:\tss-lean\LEDGER.md:85-86,114-116`).

**[DOCUMENT STATUS]** Date: 2026-07-18. This document specializes Candidate C and Sections 3.3, 3.9, and 5.3 of `COMPLETENESS_SPEC.md`. The parent document's claim-label discipline and CP-O numbering are incorporated unchanged.

## 0. Decision and security boundary

**[DESIGN]** `NoTssCertificate` v1 proves only `NoContractWin(Q, root)` for the exact query `Q` and frozen grammar `CP1-a49e8abd-v1`. It is not a game-theoretic no-win certificate and is not portable to a different root, claimant, horizon, structural cap, flag profile, or grammar version.

**[DESIGN]** The certificate is a compact negative-DAG skeleton. It serializes the negative induction choices and nothing that the checker can safely regenerate: no non-root positions, move coordinates, legal sets, threat sets, terminal results, typed-leaf tags, PN/DN values, solver node tags, transposition hashes, generator fingerprints, priors, or trusted `exhausted=true` bit.

**[DESIGN]** `checkNo` globally regenerates legal moves, the applicable CP1 attack or defense generator, remote threats and goals, terminal outcomes after every placement, typed leaf facts, phase/clock facts, and exact structural costs. This is the binding O4 ruling: full-root binding combined with local-only rescanning is rejected.

**[CODE FACT]** The positive verifier already establishes the intended independence shape: `tss_verify.rs` does not depend on `tss_solver`, replays represented moves through `HexoState`, and uses shared one-turn analysis rather than solver arena facts (`packages/hexfield_eq/rust/src/tss_verify.rs:1-15,171-220`).

**[DESIGN]** A producer may be arbitrarily buggy. Its only authority is the byte string accepted by the independently specified checker. Parse failure, resource refusal, regeneration mismatch, unsupported version, or checker rejection produces `UNKNOWN(Incomplete)` and no semantic result.

## 1. Logical object and minimal regenerable skeleton

### 1.1 Query and node proposition

**[DESIGN]** The bound query is

```text
Q = (grammar_id, exact_root, claimant, H, S, C)
```

**[DESIGN]** For v1, `grammar_id = "CP1-a49e8abd-v1"`, `claimant = exact_root.current_player`, `H` is the absolute semantic horizon, `C = 256`, and `S = min(H - exact_root.placements_made, C)` after checked arithmetic.

**[DESIGN]** A negative node denotes the proposition

```text
NoAt(Q, state, state_depth, cert_depth) :=
  no CP1 positive derivation rooted at state can be expanded
  within H, S, C when entered at those exact depths.
```

**[DESIGN]** `state_depth` is the number of placements since the exact root. `cert_depth` is the recursive strict-certificate depth already consumed. Both belong to the proposition: two equal board states reached with different structural depth are not shareable.

### 1.2 Negative node forms

**[DESIGN]** The only v1 node forms are:

```text
BaseNoConstructor
ChoiceExhausted(regenerated edge dispositions)
UniversalCounterexample(regenerated chosen edge disposition)
```

**[DESIGN]** `BaseNoConstructor` is accepted only when global replay finds no in-contract typed positive leaf and the owner/phase/tight-dispatch rules expose no applicable positive internal constructor. A claimant node with zero attack edges is encoded as `ChoiceExhausted` with zero dispositions, so claimant generator exhaustion remains explicit.

**[DESIGN]** `ChoiceExhausted` carries one disposition for every edge of the checker's complete canonical `AttackEdges_CP1` result. The record's `EXHAUSTED` event is a checked assertion, not authority: acceptance occurs only after the checker itself finishes generation and consumes exactly the regenerated edge count.

**[DESIGN]** `UniversalCounterexample` carries one canonical edge ordinal and one negative disposition. The checker nevertheless regenerates the complete canonical `DefendEdges_CP1` list before accepting membership, because owner, tight-dispatch, remote own-win, exact hitting number, and the named obligation are global facts.

**[DESIGN]** An edge disposition is exactly one of:

```text
LocalNo
ChildNo(forward node reference)
StructuralBoundary(observed child state depth, observed child cert depth)
```

**[DESIGN]** `LocalNo` is accepted only when independent edge replay classifies the regenerated edge as locally nonpositive with no recursive CP1 obligation. `ChildNo` is accepted only for an in-contract recursive child and checks the referenced negative node at the exact replayed state and depths. `StructuralBoundary` is accepted only for an otherwise recursive edge whose replayed child crosses `S` or whose exact positive expansion crosses `C`, after excluding every edge-local claimant completion, tactical leaf, or other positive form within `H` and `C`.

### 1.3 Canonical edge order

**[DESIGN]** The byte format identifies generated edges by ordinal, so v1 freezes a checker-owned canonical order independent of Rust `HashMap`/`HashSet` iteration and solver ranking.

**[DESIGN]** Each regenerated sequential edge has a canonical identity

```text
(edge_kind, placement_count, q1, r1, q2, r2, quotient_kind)
```

where unused second coordinates are zero and `quotient_kind` distinguishes an ordinary sequential edge from a checked defender-pair quotient representative.

**[DESIGN]** Coordinates compare as signed integers. Edges sort lexicographically by the tuple above. Exact duplicate identities are rejected by the checker generator rather than silently deduplicated. For attacker pairs, both legal orders are examined before the grammar's proved unordered/final-state quotient is applied. For defender pairs, the logical generator is sequential; a batch atomic pair is admitted only through the proved quotient and commutation relation.

**[CODE FACT]** The current engine move representation distinguishes one placement, an attacker pair, and a canonicalized defender pair (`packages/hexfield_eq/rust/src/tss_solver.rs:2117-2125`). Its materializer expands an ordinary pending edge by one strict `Choice`, an attacker pending pair by two `Choice` nodes, an ordinary defender edge by one `Universal` edge, and a defender pair by nested Universals plus checked commutations (`packages/hexfield_eq/rust/src/tss_solver.rs:6523-6580,6588-6670,6684-6729,6732-6865`).

**[DESIGN]** The exact structural increments used by v1 are therefore:

| Regenerated recursive edge | Increment to child `state_depth` | Increment to child `cert_depth` |
|---|---:|---:|
| **[DESIGN]** ordinary one-placement Choice | 1 | 1 |
| **[DESIGN]** pending attacker complete-turn pair | 2 | 2 |
| **[DESIGN]** ordinary one-placement Universal | 1 | 1 |
| **[DESIGN]** checked defender complete-turn pair quotient | 2 | 2 |

**[DESIGN]** Edge-local positive shapes use their actual strict expansion: a direct one-placement `OrCompletion` is at the current certificate node; a second-placement pair completion expands as first `Choice` then `OrCompletion`; and a tactical pair expands through two `Choice` nodes. The checker tests these positive shapes before considering a boundary.

### 1.4 DAG compactness and exact sharing

**[DESIGN]** Node `0` is the root. Every `ChildNo` reference is a strictly positive delta from the referring node ID, so children have larger IDs and the byte-level graph is acyclic without serialized ranks.

**[DESIGN]** Every node except `0` has a first incoming reference. That reference defines its primary replay path. Later references may share it only when replay from the exact root along the stored primary-parent/edge-ordinal chain produces an exactly equal full state and equal `(state_depth, cert_depth)`.

**[DESIGN]** The checker stores only the primary parent, primary edge ordinal, depth pair, validation bit, and reachability bit for each node. On a repeated reference it reconstructs the primary state from the root and compares full states. A hash may accelerate a negative comparison but never establishes equality.

**[DESIGN]** Every node must be reachable, every proposition must have a unique node ID, and a second node reached at an already represented exact proposition is rejected. These canonicality rules force useful DAG sharing and prohibit padding the artifact with an unvisited search arena.

## 2. Version-1 byte format

### 2.1 Scalar conventions

**[DESIGN]** All fixed-width integers are unsigned little-endian except coordinates, which are two's-complement little-endian `i16`. All reserved bytes and bits must be zero.

**[DESIGN]** `ULEB32` is unsigned LEB128 restricted to `u32`, at most five bytes, and must use the shortest encoding. Overflow, a continuation bit in byte five, or a nonminimal encoding is a parse failure.

**[CODE FACT]** Engine coordinates are axial `(q,r)` pairs of `i16`; players have exactly `Player0` and `Player1`; phases are `Opening`, `FirstStone`, and `SecondStone { first }`; terminal outcomes store winner plus absolute placement count (`packages/hexo_engine/rust/src/coord.rs:9-25`; `packages/hexo_engine/rust/src/state.rs:21-71`).

### 2.2 Fixed prefix and exact root

**[DESIGN]** The v1 file begins with this exact 86-byte prefix:

| Offset | Bytes | Field | Required v1 value/meaning |
|---:|---:|---|---|
| 0 | 8 | **[DESIGN]** magic | ASCII `NTSSCP1\0` = `4e 54 53 53 43 50 31 00` |
| 8 | 2 | **[DESIGN]** format major | `1` |
| 10 | 2 | **[DESIGN]** format minor | `0` |
| 12 | 4 | **[DESIGN]** header flags | `0` |
| 16 | 8 | **[DESIGN]** total byte length | exact file length, including prefix/root/nodes |
| 24 | 1 | **[DESIGN]** grammar length | `15` |
| 25 | 15 | **[DESIGN]** grammar bytes | ASCII `CP1-a49e8abd-v1` |
| 40 | 4 | **[DESIGN]** semantic horizon `H` | absolute placement index |
| 44 | 2 | **[DESIGN]** state-depth cap `S` | checked value `min(H-p0,256)` |
| 46 | 2 | **[DESIGN]** certificate-depth cap `C` | `256` |
| 48 | 1 | **[DESIGN]** claimant | `0=Player0`, `1=Player1` |
| 49 | 1 | **[DESIGN]** root current player | same encoding; must equal claimant |
| 50 | 1 | **[DESIGN]** root phase | `0=Opening`, `1=FirstStone`, `2=SecondStone` |
| 51 | 1 | **[DESIGN]** terminal-present tag | `0=None`, `1=Some` |
| 52 | 4 | **[DESIGN]** root placements made `p0` | exact absolute clock |
| 56 | 2 | **[DESIGN]** `SecondStone.first.q` | exact when phase=2; zero otherwise |
| 58 | 2 | **[DESIGN]** `SecondStone.first.r` | exact when phase=2; zero otherwise |
| 60 | 1 | **[DESIGN]** terminal winner | player encoding when present; `0xff` otherwise |
| 61 | 1 | **[DESIGN]** reserved | `0` |
| 62 | 4 | **[DESIGN]** terminal placement count | exact when present; zero otherwise |
| 66 | 4 | **[DESIGN]** root stone count `R` | number of following root entries |
| 70 | 4 | **[DESIGN]** node count `N` | `1 <= N <= MAX_NO_NODES` |
| 74 | 8 | **[DESIGN]** disposition count `D` | exact count in all node records |
| 82 | 4 | **[DESIGN]** root node ID | `0` |

**[DESIGN]** Exactly `R` root entries follow. Each is five bytes: `q:i16`, `r:i16`, `owner:u8`. Entries are strictly increasing by signed `(q,r)`, contain no duplicate coordinate, and owner is `0` or `1`.

**[DESIGN]** `WellFormedCP1` reconstructs a full engine/model state from those entries and all root scalar fields; it does not merely compare occupancy. In particular it checks the phase schedule, current player, placement clock, terminal consistency, and the `SecondStone.first` anchor. V1 then requires the root to be post-opening and nonterminal.

**[CODE FACT]** The existing positive `RootBinding` contains sorted occupancy and parallel owners, current player, exact phase including the stored first coordinate, `placements_made`, and terminal outcome (`packages/hexfield_eq/rust/src/tss_verify.rs:39-77`).

### 2.3 Node stream

**[DESIGN]** The node stream begins immediately after the `5R` root bytes and contains exactly `N` self-delimiting records in node-ID order. No trailing bytes are permitted.

**[DESIGN]** Every node begins with this five-byte prefix:

```text
tag:u8 | state_depth:u16 | cert_depth:u16
```

**[DESIGN]** The tags and payloads are:

| Tag | Form | Payload after five-byte prefix |
|---:|---|---|
| `0x00` | **[DESIGN]** `BaseNoConstructor` | none |
| `0x01` | **[DESIGN]** `ChoiceExhausted` | `0x01` generator event, `generated_count:ULEB32`, then exactly that many dispositions |
| `0x02` | **[DESIGN]** `UniversalCounterexample` | `0x01` generator event, `generated_count:ULEB32`, `chosen_ordinal:ULEB32`, then one disposition |

**[DESIGN]** `0x01` is the only v1 generator event and means `EXHAUSTED`. `UNSTARTED`, a produced prefix, or an unknown event is never serializable as negative evidence. The checker requires `generated_count` to equal the length of its completed canonical regeneration. At a Universal it also requires `generated_count > 0` and `chosen_ordinal < generated_count`.

**[DESIGN]** Dispositions are encoded as:

| Tag | Form | Payload |
|---:|---|---|
| `0x00` | **[DESIGN]** `LocalNo` | none |
| `0x01` | **[DESIGN]** `ChildNo` | `forward_delta:ULEB32`, required `>=1`; target ID is current ID plus delta with checked addition |
| `0x02` | **[DESIGN]** `StructuralBoundary` | `observed_child_state_depth:u16`, `observed_child_cert_depth:u16` |

**[DESIGN]** The header's `D` equals the number of disposition tags: all Choice dispositions plus one for each Universal. A mismatch is a parse failure.

**[DESIGN]** The observed boundary depths are provenance, not trusted cutoffs. They must equal the checker's independently replayed depths. No stage-local or eligible intermediate cutoff has an encoding.

### 2.4 Limits and version behavior

**[DESIGN]** Initial executed-checker limits are:

```text
MAX_NO_CERT_BYTES    = 134_217_728   // 128 MiB
MAX_NO_NODES         = 4_000_000
MAX_NO_DISPOSITIONS  = 16_000_000
MAX_NO_ROOT_STONES   = 1_000_000
MAX_NO_DEPTH         = 256
```

**[DESIGN]** The parser checks `total byte length` and `MAX_NO_CERT_BYTES` before allocation, then checks all counts by checked arithmetic. Limits are deployment refusal bounds, not semantic clauses. A true negative whose artifact exceeds them remains `UNKNOWN(Incomplete)`.

**[DESIGN]** Major version mismatch is rejected. Minor versions are not prefix-compatible in v1: only `(1,0)` is accepted. A new grammar, flag matrix, edge order, or semantic rule requires a new `grammar_id`; a byte-layout change requires a new format version. Old checkers never reinterpret new bytes.

## 3. `checkNo`: total bounded checker

### 3.1 Top-level pseudocode

**[DESIGN]** `checkNo` has no solver-state input and cannot accept a solver arena pointer.

**[DESIGN]** `WellFormedCP1(Q,P)` is the conjunction of these executable checks:

1. **[DESIGN]** `P`'s canonical full `RootBinding` is byte-for-field equal to the certificate root; sorted occupancy is unique; every owner/player/tag is valid; `R = placements_made`; and rebuilding the board from the entries returns the same occupancy/owners.
2. **[DESIGN]** The root is post-opening and nonterminal; recomputed terminal detection finds no completed window; current player and `FirstStone`/`SecondStone` phase agree with the placement schedule; and a `SecondStone.first` is occupied by the current player and satisfies the legal-store anchor.
3. **[DESIGN]** `claimant = P.current_player`; grammar bytes are exactly `CP1-a49e8abd-v1`; and the frozen CP1 flags/profile are pair-complete base-wide search with quiet/ranked/census/shared-fragment/experimental consumers off and canonical batch defense.
4. **[DESIGN]** `H >= p0`; checked subtraction gives `S = min(H-p0,256)`; `C=256`; root/node/disposition/byte limits hold; and every coordinate, placement clock, resolution, depth, and structural-cost conversion required by regeneration is checked for overflow before use.
5. **[DESIGN]** Rebuilding all global derived primitives—windows, complete/threat families, legal cells queried by the finite generators, own-win-now, phase budget, and terminal status—succeeds and agrees with the exact root fields. No solver-maintained incremental cache is an input to this rebuild.

```text
checkNo(external_root, bytes) -> bool:
  cert := parseStrictV1(bytes)                         // CP-O18, CP-O29
  reject unless cert.root == canonical(external_root) // CP-O1, CP-O27
  Q := queryFromHeader(cert)
  reject unless WellFormedCP1(Q, external_root)       // CP-O1, CP-O2, CP-O3
  reject unless cert.N >= 1 and node[0].depths=(0,0)

  meta := arrays[N] of
    { primary_parent?, primary_edge_ordinal?,
      state_depth, cert_depth, reached=false, validated=false }
  seenPropositions := exact-state comparison service, not a hash oracle
  dispositionCounter := 0

  ok := verifyNoNode(0, clone(external_root), 0, 0)
  reject unless ok
  reject unless every node is reached and validated
  reject unless dispositionCounter == cert.D
  reject unless parser consumed exactly cert.total_length bytes
  accept
```

### 3.2 Exact replay and sharing

```text
enterNode(id, state, sd, cd, incoming?):
  reject unless id < N
  reject unless node[id].state_depth == sd
  reject unless node[id].cert_depth == cd
  reject unless sd <= S and cd <= C

  if not meta[id].reached:
    set primary incoming edge (root has none)
    mark reached
  else:
    primaryState := replayPrimaryPathFromExactRoot(id)
    reject unless primaryState == state exactly
    reject unless primary replay derives the same sd and cd

  reject if a different node ID already denotes this exact
    (state, sd, cd, Q) proposition
  if meta[id].validated: return true
  mark validation-in-progress
  result := verifyNoNodeBody(id, state, sd, cd)
  reject unless result
  mark validated
  return true
```

**[DESIGN]** Forward-only references make `validation-in-progress` re-entry impossible; observing it is an invariant failure. Primary-path replay performs at most `C` checked placements/expanded steps and stops immediately on terminal state or replay disagreement.

### 3.3 Global node regeneration

```text
verifyNoNodeBody(id, P, sd, cd):
  leaf := regenerateTypedPositiveLeafGlobally(Q, P, cd)
  owner := classifyOwnerPhaseAndTightDispatchGlobally(Q, P)

  match node[id]:
    BaseNoConstructor:
      reject if leaf is positive within H,C
      accept iff owner exposes no applicable positive internal constructor

    ChoiceExhausted(event, recordedCount, dispositions):
      reject unless P.current_player == claimant
      reject if leaf is positive within H,C
      edges := regenerateCanonicalAttackEdgesGlobally(Q, P)
      reject unless event == EXHAUSTED
      reject unless recordedCount == len(edges)
      reject unless len(dispositions) == len(edges)
      for ordinal in 0 .. len(edges)-1:
        reject unless checkEdgeNo(id, P, sd, cd,
                                  ordinal, edges[ordinal], dispositions[ordinal])
      accept

    UniversalCounterexample(event, recordedCount, chosen, disposition):
      reject if leaf is positive within H,C
      reject unless owner is the exact applicable nonempty tight dispatcher
      edges := regenerateCanonicalDefendEdgesGlobally(Q, P)
      reject unless event == EXHAUSTED
      reject unless recordedCount == len(edges) and len(edges) > 0
      reject unless chosen < len(edges)
      accept iff checkEdgeNo(id, P, sd, cd,
                             chosen, edges[chosen], disposition)
```

**[DESIGN]** “Globally” means scanning/reconstructing all facts named by the CP1 grammar from the full replay state, including defender remote win-now windows. The checker does not accept a solver-supplied `WideTurnGate`, candidate vector, terminal bit, hitting number, or leaf classification.

### 3.4 Edge checking and structural boundary

```text
checkEdgeNo(parentId, P, sd, cd, ordinal, edge, disposition):
  replay := apply edge placements one at a time with checked legality
  classify terminal outcome after every placement
  enumerate every CP1 positive edge-local shape and its exact
    resolution and strict-certificate expansion

  if any positive edge-local shape fits H and C:
    reject                         // no negative disposition can cover it

  class := independently classify edge as
    LocalNonpositive
    | Recursive(childState, childSd, childCd)
    | Boundary(childState, childSd, childCd, reason)

  match disposition:
    LocalNo:
      accept iff class == LocalNonpositive

    ChildNo(delta):
      reject unless class == Recursive(...)
      childId := checkedAdd(parentId, delta)
      reject unless childId > parentId and childId < N
      register/compare primary incoming (parentId, ordinal)
      accept iff enterNode(childId, childState, childSd, childCd, incoming)

    StructuralBoundary(observedSd, observedCd):
      reject unless class == Boundary(...)
      reject unless observedSd == childSd and observedCd == childCd
      accept iff (childSd > S or childCd > C)
        and no edge-local positive shape fits H,C
```

**[DESIGN]** `H` is checked separately from `S` and `C`. A child state beyond `S` does not erase a completion or tactical success generated from an admitted parent. An atomic overshoot is therefore not boundary evidence until the checker has expanded and rejected all positive shapes for that exact edge.

**[DESIGN]** A horizon-exceeding typed leaf is simply unavailable to this query. Arithmetic overflow is never interpreted as horizon exclusion: it rejects `WellFormedCP1` or the edge replay.

### 3.5 Exact acceptance condition

**[DESIGN]** `checkNo(Q, root, bytes) = true` if and only if all of the following operational checks succeed:

1. **[DESIGN]** The byte string is the unique strict v1 parse, is within every limit, contains no trailing data, and uses no unknown/reserved value.
2. **[DESIGN]** The complete exact root and every query field match the external root and `WellFormedCP1`.
3. **[DESIGN]** Node 0 checks as `NoAt(Q,root,0,0)`.
4. **[DESIGN]** Every Choice generator independently reaches `EXHAUSTED`, its canonical regenerated count equals the record, and every regenerated edge has checked negative evidence.
5. **[DESIGN]** Every Universal counterexample is selected from a completely regenerated nonempty defender relation and has checked negative evidence.
6. **[DESIGN]** Every leaf, legal, threat, terminal, owner, phase, hitting, resolution, transition, and structural-cost fact is rederived globally.
7. **[DESIGN]** Every DAG reference is forward, reachable, proposition-exact, and depth-exact; every record is referenced and validated exactly once as a proposition.
8. **[DESIGN]** Every serialized boundary is a final structural boundary, never an intermediate cutoff, and no in-contract positive edge-local form was hidden by it.

**[DESIGN]** Lean's `checkNo_sound` proof follows the successful checker recursion: `BaseNoConstructor` discharges the no-constructor base; Choice uses regenerated set equality plus all negative dispositions; Universal uses the regenerated member plus one negative disposition; a boundary uses the definition of the bounded positive grammar; exact repeated propositions use the already proved induction result.

### 3.6 Totality and resource argument

**[DESIGN]** Parsing is total by the decreasing unread-byte count and fixed maxima. Graph recursion is total because every child ID is strictly greater and every recursive replay increases the checked state/structural depth; accepted paths are bounded by `C=256`.

**[DESIGN]** CP-O2 must provide a computable finite bound `G(Q,P)` for each attack/defense regeneration. Let `N` be nodes, `D` dispositions, `R` repeated references, and `Gmax` the maximum regeneration bound over reached states. The checker performs at most:

```text
N node-body checks
D edge classifications
R * C primary-path replay steps
N * Gmax generator candidates
```

with all counters bounded by the parsed deployment maxima. It has no unbounded retry, scheduler, cache-retention, or proof-number loop.

**[DESIGN]** The executed checker may stream node bytes after building an offset table. Its required per-node metadata is bounded and fixed-size; full states are held only for the active replay and reconstructed primary path. This avoids storing `N` full boards while preserving exact equality.

### 3.7 Obligation discharge map

| Checker step | Obligations directly served |
|---|---|
| **[DESIGN]** strict header/root/WellFormed validation | CP-O1, CP-O2, CP-O3, CP-O27, CP-O29 |
| **[DESIGN]** global typed-leaf/terminal replay | CP-O3, CP-O16, CP-O27 |
| **[DESIGN]** canonical attack regeneration and all-edge loop | CP-O14, CP-O16, CP-O18 |
| **[DESIGN]** canonical defense regeneration and selected member | CP-O15, CP-O18, CP-O25 |
| **[DESIGN]** finite dual recursion | CP-O2, CP-O17, CP-O18 |
| **[DESIGN]** exact forward DAG sharing and primary replay | CP-O19, CP-O26 |
| **[DESIGN]** structural-boundary expansion | CP-O16, CP-O23 |
| **[DESIGN]** strict parser, limits, termination, executed bridge | CP-O27, CP-O29 |
| **[DESIGN]** checker rejection before sealed mint | CP-O28, CP-O29 |

**[NON-CLAIM]** Candidate C does not discharge CP-O20 through CP-O24 or CP-O26 as scheduler/lazy/cache completeness theorems. The rows above show where the checker avoids trusting those mechanisms, not proofs that the mechanisms are live or complete.

## 4. Size model and hard-row estimate

**[DESIGN]** V1's exact byte count is

```text
86 + 5R
+ sum(Base records: 5)
+ sum(Choice records: 6 + uleb(generated_count) + disposition bytes)
+ sum(Universal records:
       6 + uleb(generated_count) + uleb(chosen_ordinal) + disposition bytes)
```

where a disposition is one byte for `LocalNo`, `1 + uleb(delta)` for `ChildNo`, or five bytes for `StructuralBoundary`.

**[DESIGN]** With only header totals known, a conservative per-record envelope is `16N + 6D` bytes for the node stream: 16 bytes covers the largest Universal fixed/varint overhead and six bytes covers the largest `ChildNo` disposition. The exact root cost is `86 + 5R` bytes.

**[PRIOR-ROUND FACT]** The current hardest forcing row is `0l4291i_live`; its exact corpus root has 63 stones (`packages/hexfield_eq/rust/corpus/forcing_corpus_moves.txt:24`). In the retained +1 flags-off run it used 1,879,612 arena nodes, while the attacker closure counter reported 2,788,989 retained generated children (`CLOSURE_COUNTER_FULL_OFF_RAW.log:22-23`). The row is a verified WIN, so these are stress-sizing proxies, not a negative certificate measurement.

**[ESTIMATE]** Substituting `R=63`, `N=1,879,612`, and proxy `D=2,788,989` gives an exact conservative format envelope of:

```text
401 + 16*1,879,612 + 6*2,788,989
= 46,808,127 bytes
= 44.640 MiB.
```

**[ESTIMATE]** A planning case of 11 bytes per node plus three bytes per disposition is exactly 29,043,100 bytes = 27.698 MiB for the same proxy counts. The node-prefix floor alone is exactly 9,398,461 bytes = 8.963 MiB. Neither planning number is an observed certificate size; the first emitter campaign must replace them with measured `(R,N,D,tag/varint histogram)` values.

**[DESIGN]** The artifact is therefore not the 549 MiB mutable hard-row search arena and not its hundreds of millions of evaluated pair candidates. Only the canonical negative proof DAG and its regenerated-edge dispositions survive. If representative genuine negatives do not remain under the predeclared 128 MiB artifact and 256 MiB extra-checker-peak gates, hot-path Candidate C fails closed and the route-(b) kill criterion is triggered.

## 5. Required hostile tests before authority

**[DESIGN]** The minimum mutation suite must reject: one omitted remote in-contract attacker edge; one extra edge; a swapped edge ordinal; a forged `EXHAUSTED` count; a truncated disposition stream; a positive completion labeled `LocalNo`; a hidden positive atomic overshoot labeled boundary; an eligible cutoff labeled boundary; a wrong `SecondStone.first`; a remote defender win-now omission; a terminal winner/clock mismatch; a backward/cyclic node reference; a shared node reached at a different exact state or cert depth; an unreachable padding node; a nonminimal ULEB; trailing bytes; and every count/length overflow.

**[DESIGN]** NQ2 remains an out-of-contract boundary fixture for frozen CP1 because quiet-turn consumption is off. A distinct exact-root fixture whose omitted remote winning edge is proved to belong to `AttackEdges_CP1` is mandatory for the in-contract omission mutation.

**[DESIGN]** No production `NO_CONTRACT_WIN` mint is reachable until the strict v1 codec, proved `checkNo_sound`, executed-checker correspondence, exact-root binding, and fail-closed stop taxonomy all pass their respective gates.
