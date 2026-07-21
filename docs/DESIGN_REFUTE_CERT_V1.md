# Certified `vcf_pair_complete` refutations: amended v1 design

Status: design only, amended 2026-07-21 after hostile review. This document
does not authorize source, test, Lean, search-policy, trainer, positive-certificate,
or `tss_verify.rs` changes. A new hostile review of this amended target is
required before implementation.

**HYPOTHESIS — evidence discipline.** In this document, **CODE-FACT** means a
fact directly visible in the current Rust implementation and is accompanied by
a current line reference. Every proposed rule, semantic bridge, estimate, gate,
and Lean statement is labelled **HYPOTHESIS**. Naming a theorem is not a claim
that it exists or that executed Rust has been connected to it. Normative
`MUST`/`MUST NOT` language freezes the proposed contract; it is not implementation
evidence.

## Amendment ledger

| item | sections touched | amendment |
|---|---|---|
| R1 | 1.1, 1.3, 2.1, 4.2, 5, 7 | Publishes the literal phase-indexed positive judgment, clock lifting, reachability premise, and four separate theorem layers. |
| R2 | 1.1, 2.1, 2.2, 2.4, 4.2, 4.3, 6 | Freezes direct-window sets, `T/G1/S/U`, quotient orientation, pair classification, expansion, telemetry, oracle fixtures, and version-bump rules. |
| R3 | 1.2, 3.2, 3.3, 6 | Replaces shallow-zero selection with a memoized recursive `Structural | Unresolved` support fixed point and alternative defender selection. |
| R4 | 1.3, 2.1, 2.4, 4.2, 4.3, 7 | Deletes `NoPositiveConstructor` and `NoJointCarrier`, closes the leaf/polarity matrix, and retains only exact empty enumeration in leaf v1. |
| R5 | 4.1, 4.2, 4.3, 6 | Makes independence a transitive source/call-graph contract with a stated shared trust base, direct transition cross-checks, and one-sided fault tests. |
| R6 | 2.5, 2.6, 3.3, 4.3, 4.4, 6 | Adds externally selected hard limits for regenerated semantic work, state retention, CPU/wall, and heap; the one-million-stone allowance is removed. |
| R7 | 2.6, 3.3, 4.4, 6, 7 | Gates producer and end-to-end cost, uses causal size/work denominators, requires competitive baselines and held-out tails, and withdraws the unpinned node estimate. |
| R8 | 2.1, 2.5, 4.1, 4.3, 6 | Restricts acceptance to a checked D6-safe coordinate closure and binds ruleset, coordinate, and semantic versions into root identity and the typed result. |
| R2-1 | 1.3, 2.4, 2.6, 3.1, 3.2, 4.2, 6 | Defines one authoritative `RefuteLeafExactEligibleV1` conjunction for production, self-verification, and gating, including all root, profile, strict-cap, constructor, and exact-expansion premises. |
| R2-2 | 2.1, 4.2, 4.3, 6 | Freezes the byte preimage of `root_semantic_sha256`, defines failure counters as ordered-occurrence counts, states their sum identities, and requires independent digest/counter goldens. |

## 1. Claim semantics

### 1.1 Literal phase-indexed proposition — amended per R1, R2, R8

**HYPOTHESIS — normative semantic block.** The following definitions, together
with the equations in section 2.2, are the single v1 meaning of the class. The
producer, independent Rust verifier, simple oracle, and later model checker
MUST implement this specification independently. Agreement between two
implementations is not a definition and is not a completeness proof.

The frozen identifiers are:

```text
ruleset       = HexoConnect6/Opening1-Then2/Win6/LegalRadius8/I16/V1
coordinate    = AxialQR/D6SafeClosure/V1
class         = VcfPairComplete/EqualityDispatch/V1
wire          = RefuteLeafExact/V1
```

A semantic change to legality radius, opening rule, win length, coordinate
meaning, phase schedule, direct-window equations, constructor priority,
`T/G1/S/U`, quotienting, pair disposition, or equality dispatch MUST use a new
ruleset/class identifier and a new wire format. An old verifier MUST reject the
new identifier unless an explicit compatibility theorem and policy entry exist.

Let `P` be a finite engine state, `A` the fixed claimant, `D = other(A)`,
`phi = P.phase()`, and `n = P.placements_made()`. `h` is an absolute finite
resolution deadline measured in placements. A placement constructor is
available only when its resulting placement clock is at most `h`. The phase
budget is

```text
B(Opening) = 1;  B(FirstStone) = 2;  B(SecondStone { first }) = 1.
```

The state arguments always include the complete phase payload, especially
`SecondStone { first }`; writing only `phi` below is notation, not permission to
drop `first`. `H_X(P)`, `tau`, `K_b`, `T`, `G1`, `S`, the ordered occurrence
universe `U`, and `Disposition` are the exact functions in sections 2.2-2.3.
`OwnWinNow_X(P,b)` means that player `X`, who is the mover, has a direct sequence
of at most `b` legal placements in the current turn completing a live length-six
`X` window. `ForcedLoss_X(P,b)` means `not OwnWinNow_X(P,b)` and
`tau(H_other(X)(P)) > b`. Both predicates are recomputed from literal windows.

`ContractWinV1(P,A,phi,n,h,ruleset,class,reachable)` is the least finite,
mutually phase-indexed judgment generated by exactly the following closed
constructor table, in the listed priority order:

| exact state | positive constructor | exact premises |
|---|---|---|
| any supported phase | `ClaimantTerminal` | `P.terminal().winner = A` and `n <= h`. |
| claimant `FirstStone` | `ClaimantWinNow` | nonterminal; `P.current_player() = A`; `OwnWinNow_A(P,2)` has a direct completing sequence whose last clock is `<= h`. |
| claimant `FirstStone` | `ClaimantChoice` | no prior constructor; not `ForcedLoss_A(P,2)`; there exists `(a,b) in U(P)` whose ordered replay is legal. A claimant-terminal prefix or full successor yields `ClaimantCompletion`; otherwise `Disposition(P,a,b)` is either `ClaimantTactical` (`tau > 2`) or `TightPair` (`tau = 2`) and, for `TightPair`, the resulting nonclaimant `FirstStone` state satisfies the mutually recursive judgment. |
| nonclaimant `FirstStone` | `ClaimantTactical` | nonterminal; mover is `D`; `not OwnWinNow_D(P,2)`; `H_A(P)` is nonempty; `tau(H_A(P)) > 2`; its fixed resolution clock is `n + 2 <= h`. |
| nonclaimant `FirstStone` | `DefenderUniversal2` | nonterminal; mover is `D`; `not OwnWinNow_D(P,2)`; `H_A(P)` is nonempty; `tau(H_A(P)) = 2`; for every `d in K_2(P,A)`, applying `d` is legal and the resulting `SecondStone { first = d }` state satisfies the mutually recursive judgment. |
| nonclaimant `SecondStone { first }` | `ClaimantTactical` | nonterminal; mover is `D`; `not OwnWinNow_D(P,1)`; `H_A(P)` is nonempty; `tau(H_A(P)) > 1`; its fixed resolution clock is `n + 1 <= h`. |
| nonclaimant `SecondStone { first }` | `DefenderUniversal1` | nonterminal; mover is `D`; `not OwnWinNow_D(P,1)`; `H_A(P)` is nonempty; `tau(H_A(P)) = 1`; for every `d in K_1(P,A)`, applying `d` is legal and the resulting claimant `FirstStone` state satisfies the mutually recursive judgment. |

There are no other constructors. In particular:

- a terminal winner `D`, `OwnWinNow_D`, or `ForcedLoss_A` is an opponent
  structural failure of the positive judgment;
- at a nonclaimant state, an empty `H_A(P)` or `tau(H_A(P)) < b` has no
  Universal constructor; this class is equality-only and does **not** import
  the full-legal-set `tau < b` branch of generic T6;
- `tau(H_A(P)) > b` is claimant-positive tactical success, never a negative
  “not tight” leaf;
- a claimant `Opening` or claimant `SecondStone` state is outside the class;
  and
- an unknown future constructor is a new class version, never fall-through
  authority for an old negative verifier.

**HYPOTHESIS — explicit negative.** A v1 call takes a
`ReachableRootV1(P, ruleset)` premise minted either by replaying a canonical
legal history from the opening or by a separately trusted engine API. Sorted
stones, phase, and clock consistency do not establish reachability. The public
result MUST retain this premise or require the trusted token; bare external
bytes are insufficient.

For all arguments visible, define:

```text
NoContractWinAtV1(P,A,phi,n,h,ruleset,class,reachable)
  := not ContractWinV1(P,A,phi,n,h,ruleset,class,reachable)

NoContractWinV1(P,A,FirstStone,n,nextPly,ruleset,class,reachable)
  := forall finite h, h >= nextPly ->
       NoContractWinAtV1(P,A,FirstStone,n,h,ruleset,class,reachable)
```

The artifact root MUST be reachable, nonterminal, post-opening,
`FirstStone`, D6-safe under section 2.5, and have `A = P.current_player()` and
`nextPly = n + 1`. The artifact certifies exactly the second proposition. It
does not certify full-game loss, an opponent strategy, or even that `A` lacks a
quiet or otherwise out-of-class win.

**HYPOTHESIS — clock lifting obligation.** Define `BoundaryFreeNo(P,A,phi,n)`
by structural recursion over the closed negative grammar of section 2.1, with
no horizon/cap/depth premise. The model MUST prove:

```text
boundaryFreeNo_sound_at:
  BoundaryFreeNo(P,A,phi,n) ->
  forall finite h >= n, not ContractWinV1(P,A,phi,n,h,...)

contractWin_monotone:
  ContractWinV1(P,A,phi,n,h,...) -> h <= h' ->
  ContractWinV1(P,A,phi,n,h',...)
```

The first theorem is proved by induction on the negative tree against the
literal constructor table; monotonicity is proved by induction on the positive
derivation because enlarging an absolute deadline preserves every placement
side condition. Only then may a boundary-free accepted object yield all finite
horizons. `u32::MAX` remains a producer-profile marker and is not infinity.

**CODE-FACT.** Player identity, not depth parity, controls polarity:
`FirstStone -> SecondStone` keeps the mover, while completing `SecondStone`
changes it
([`state.rs` lines 317-334](../packages/hexo_engine/rust/src/state.rs#L317)).
The hard-result seam defines `Loss` as a real opponent winning strategy and
leaves exhausted work as `Unknown`
([`tss_core.rs` lines 24-43](../packages/hexfield_eq/rust/src/tss_core.rs#L24)).

**HYPOTHESIS.** A verified v1 fact is never `ProofStatus::Loss`,
`HardValue(-1)`, or a full-game `NO`. Any such conversion kills the design.

### 1.2 Clock and natural exhaustion — amended per R3

**CODE-FACT.** Current PN arithmetic gives both depth cutoffs and structural
refutations zero disproof number, and several unrelated refusals collapse into
`WidePnNode::Refuted`
([`tss_solver.rs` lines 5935-5967](../packages/hexfield_eq/rust/src/tss_solver.rs#L5935),
[`lines 6341-6497`](../packages/hexfield_eq/rust/src/tss_solver.rs#L6341)).

**HYPOTHESIS.** `root.dn == 0`, a termination enum, and the current
`child_is_genuinely_refuted` helper are not provenance. A full-tree candidate is
a **natural width exhaust** only after all staged reopens and bottom-up PN
refreshes and only if the recursive fixed point in section 3.2 returns
`Structural(plan)` at the root. `semantic_horizon == u32::MAX`, exact v1 width
options, and `expansions < node_cap` remain necessary producer conditions, but
none can substitute for the completed support plan.

The plan MUST contain no depth/horizon/census/cap/stalled/lazy/zone/Group-2
boundary. A producer or verifier resource limit causes `Unsupported`/rejection;
it is never a logical leaf. The currently authorized leaf-only v1 cut does not
consume the PN arena and therefore cannot confuse a shallow zero with evidence.

### 1.3 Theorem layers — amended per R1, R4, R2-1

**HYPOTHESIS — Lean targets only.** The following layers are distinct and MUST
not be stated as one theorem:

1. `modelCheckLeafV1Bytes_sound`: the model decoder/checker accepts a model byte
   list and a model reachable root, therefore the literal
   `NoContractWinV1` proposition holds. A later full-tree format needs its own
   `modelCheckNoDagBytes_sound` theorem.
2. `noContractWin_of_noAdmissibleFirstTurn`: exact `U` expansion has no
   admitted occurrence, therefore `BoundaryFreeNo` and the literal negative
   judgment hold. There is no v1 `NoJointCarrier` theorem.
3. `rustLeafV1_extensional`: the executed Rust decoder/checker returns the same
   answer as the model checker on the same literal bytes, policy, root, and
   reachability token. Until this exists, `refuteCertV1_accepts...` may name
   model acceptance only, not executed Rust acceptance.
4. `producerLeafV1_materializes_accepted`: a producer satisfying the complete
   `RefuteLeafExactEligibleV1` predicate of section 2.4 emits bytes accepted by
   the checker. A future
   `widePnStructuralSupport_materializes...` is a separate producer
   completeness theorem, not part of the semantic capstone.

The boundary-free soundness and horizon monotonicity lemmas in section 1.1 are
prerequisites to layer 1. No Lean file exists in the target and no Lean work is
authorized by this design.

## 2. Certificate grammar

### 2.1 Logical tree and literal wire form — amended per R1, R2, R4, R8, R2-2

**HYPOTHESIS — closed logical grammar.** The polarity dual of the constructor
table is:

```text
NoA(P : claimant FirstStone) :=
    OpponentTerminal(winner = D)
  | OpponentForcedTactical(b = 2)
  | NoAdmissibleFirstTurn
  | ChoiceExhausted(for every admitted PairClass, NoD(child))

NoD(P : nonclaimant FirstStone | SecondStone { first }) :=
    OpponentTerminal(winner = D)
  | OpponentOwnWinNow(b)
  | EmptyClaimantThreatFamily(b)
  | LooseDefenderBoundary(b, tau)          // H_A nonempty and tau < b
  | UniversalCounterexample(b, reply, No(child))

PairComplement :=
    NoNewClaimantThreat
  | DefenderWinsFirst
  | LooseReply(tau = 0 | 1)
```

`OpponentForcedTactical` is accepted only at a claimant state with no claimant
own-win-now and `tau(H_D(P)) > 2`. `OpponentOwnWinNow` is accepted only at a
nonclaimant state with the direct completing sequence. The empty and loose
leaves have the exact premises shown. There is no `NoPositiveConstructor`, no
open reason value, no `NoJointCarrier`, and no producer-selected gate-failure
tag. `PairComplement` is derived virtual expansion, not stored authority.

**HYPOTHESIS — v1 scope cut.** The accepted `RefuteLeafExact/V1` wire contains
exactly one `NoAdmissibleFirstTurn` leaf. Every full-tree node form above is a
mathematical specification and future design constraint, not an accepted v1
tag. Enabling `ChoiceExhausted`, `UniversalCounterexample`, or any other leaf
requires a new wire identifier and another hostile review. This smaller cut is
the manageability decision in section 7.

**HYPOTHESIS — literal leaf wire.** Multibyte integers are little-endian;
unsigned counts use shortest-form unsigned LEB128; coordinates are signed
little-endian `i16`; players and phases use the closed one-byte enums below.
The exact byte sequence is:

```text
Header :=
  magic[8] = "HXRFLV1\0"
  format_u16 = 1
  ruleset_u16 = 1
  coordinate_u16 = 1
  class_u16 = 1
  profile_u16 = 1                    // exact-enumeration leaf
  root_stone_count_uvar
  root_stones[root_stone_count]      // (q_i16, r_i16, owner_u8), raw sorted
  mover_u8                            // 0 or 1
  phase_u8 = 1                        // FirstStone only
  placements_made_u32
  terminal_u8 = 0                     // nonterminal only
  claimant_u8                         // MUST equal mover
  root_semantic_sha256[32]
  payload_len_uvar

Payload :=
  tag_u8 = 0x20                       // NoAdmissibleFirstTurn
  t_count_uvar
  q_count_uvar
  quotient_class_count_uvar
  fail_no_new_uvar
  fail_defender_first_uvar
  fail_loose_0_uvar
  fail_loose_1_uvar

Trailer := payload_sha256[32]
```

No bytes may follow the trailer. Owner/player encodings are
`0 = Player0`, `1 = Player1`; these are wire values, not Rust discriminants.
Each `fail_*` payload field counts **ordered occurrences in `U(P)`**, not
quotient classes. If a guarded two-member commuting quotient fails for reason
`x`, its two ordered members contribute two to `fail_x`; a sole-orientation
class contributes one. Let `classes_x` be the independently regenerated number
of failing quotient classes of reason `x`, and define analogous occurrence and
class counts for `ClaimantCompletion`, `ClaimantTactical`, and `TightPair`.
For every fully regenerated root:

```text
Q = fail_no_new + fail_defender_first + fail_loose_0 + fail_loose_1
    + completion_occurrences + claimant_tactical_occurrences
    + tight_pair_occurrences

quotient_class_count = classes_no_new + classes_defender_first
    + classes_loose_0 + classes_loose_1 + classes_completion
    + classes_claimant_tactical + classes_tight_pair

Q = sum over regenerated quotient classes C of |C|,
quotient_class_count = sum over those classes C of 1, and |C| is 1 or 2.
```

For an eligible leaf the three positive/tight occurrence counts are zero, so
the four serialized `fail_*` values sum exactly to `q_count`; the four derived
failing-class counts, which are not serialized, sum exactly to
`quotient_class_count`. No payload count controls enumeration.

**HYPOTHESIS — exact root-digest preimage.** `root_semantic_sha256` is
`SHA-256(RootSemanticPreimageV1)` over the following concatenation, with no
padding, alignment, implicit string terminator, or omitted length other than
the bytes explicitly shown:

```text
RootSemanticPreimageV1 :=
  domain[25] = ASCII "HXRFLV1:ROOT-SEMANTIC:V1\0"
  ruleset_u16_le
  coordinate_u16_le
  class_u16_le
  wire_u16_le                         // equals Header.format_u16
  profile_u16_le
  root_stone_count_uvar               // shortest-form unsigned LEB128
  root_stones[root_stone_count]       // each q_i16_le, r_i16_le, owner_u8
  mover_u8
  phase_tag_u8
  phase_payload
  placements_made_u32_le
  terminal_u8
  claimant_u8

phase_payload :=
  empty                               // phase_tag 0 = Opening
  empty                               // phase_tag 1 = FirstStone
  first_q_i16_le, first_r_i16_le      // phase_tag 2 = SecondStone { first }
```

The 25 domain bytes are exactly hexadecimal
`48 58 52 46 4c 56 31 3a 52 4f 4f 54 2d 53 45 4d 41 4e 54 49 43 3a 56 31 00`.
Every `_u16_le` is exactly two bytes and `_u32_le` exactly four bytes. Every
`_i16_le` is the exact two-byte little-endian two's-complement representation.
The unsigned LEB128 count emits low-order seven-bit groups first, sets the high
bit on every nonfinal byte, clears it on the final byte, and forbids redundant
zero groups. It counts stones, not bytes; it is followed immediately by exactly
`root_stone_count` five-byte stone records and has no separate array byte-length
field. All one-byte enums are exactly one byte. `phase_payload` has no length
prefix: its length is zero, zero, or four bytes as determined solely by the
preceding phase tag.

The v1 leaf admits only `ruleset=1`, `coordinate=1`, `class=1`, `wire=1`,
`profile=1`, `phase_tag=1` with its zero-byte payload, `terminal=0`, and
`claimant=mover`; the other phase encodings above freeze what “full phase
payload” means and do not enable those phases in this wire. Stone order and all
numeric/player encodings are exactly the literal-header encodings. The digest
preimage is a **separate canonical encoding assembled from strictly decoded
header values**, not a literal contiguous or noncontiguous subset of the header:
it prepends the domain, spells out the wire value represented by `format_u16`,
omits magic, the digest itself, and `payload_len`, and includes no payload or
trailer bytes.

The semantic digest is part of proof identity. The payload hash detects
corruption only. Exact decoded fields, root binding, policy, reachability token,
and regeneration are authoritative. Unknown enum values, tags, versions, or
noncanonical encodings MUST be rejected.

**HYPOTHESIS — future full-tree wire constraint.** A later format MUST use
backward-only node IDs, canonical raw ordering, the typed logical tags above,
and the `PairKey` definition in section 2.2. It MUST publish numeric tags and
every field before implementation. Reuse requires equality of the complete
state, claimant, clock, and `SecondStone.first`; hashes and D6 images never
authorize reuse. This paragraph does not reserve silently accepted v1 tags.

### 2.2 Versioned direct mathematics — amended per R2

**HYPOTHESIS — coordinates and windows.** Let raw coordinate order be signed
numeric lexicographic `(q,r)`. The three positive window axes, in order, are
`(1,0)`, `(0,1)`, and `(1,-1)`. A `WindowKey` is `(axis_index,start_q,start_r)`;
its six cells are `start + i*axis` for `i=0..5`. Keys are ordered by the tuple
shown. Direct enumeration takes the 18 length-six keys through each occupied
cell, deduplicates keys, sorts them, and recounts all six cells from canonical
stone ownership. Incremental window-store membership is not part of the
definition.

For player `X`, write `c_X(P,W)` for its stone count and
`E(P,W)` for the raw-sorted empty cells. A window is `live_X(P,W)` exactly when
`c_X(P,W) > 0`, `c_other(X)(P,W) = 0`, and `E(P,W)` is nonempty. `Legal_P(c)`
is the frozen ruleset's turn-start placement predicate, including emptiness and
radius. All set outputs below are raw-sorted and duplicate-free.

**HYPOTHESIS — exact attacker universe.** At a nonterminal claimant
`FirstStone` turn start:

```text
T(P) =
  { c | Legal_P(c) and exists W,
        live_A(P,W) and c_A(P,W) >= 2 and c in E(P,W) }
  union
  { c | Legal_P(c) and exists W,
        live_D(P,W) and c_D(P,W) >= 4 and c in E(P,W) }.

G1(P,a) =
  { c | c != a and Legal_P(c) and exists W,
        live_A(P,W) and c_A(P,W) >= 1 and
        a in E(P,W) and c in E(P,W) }.

S(P,a) = (T(P) - {a}) union G1(P,a).

U(P) = { (a,b) | a in T(P), b in S(P,a), a != b }.
Q(P) = |U(P)| = sum_{a in T(P)} |S(P,a)|.
```

All terms are evaluated on the same turn-start `P`; `T` MUST NOT be regenerated
after applying `a`. The `c_A >= 2` part of `G1` is definitionally redundant
with `T(P)-{a}` but remains in the equation to make the promotion rule explicit.
The nonredundant `c_A = 1` part is the weak same-turn promotion. A stale
defender-block candidate remains in `S` even if applying `a` would kill its
turn-start defender-window status.

**HYPOTHESIS — ordered replay and pair family.** For `(a,b) in U(P)`, replay
`a` and then `b` through both the engine transition and the independent direct
transition cross-check. A terminal claimant prefix is `ClaimantCompletion` and
MUST remain orientation-specific. Otherwise let `Pab` be the full successor
and define the distinct, `WindowKey`-keyed post-pair family

```text
Hpair_A(P,a,b) =
  { E(Pab,W) |
      live_A(P,W), c_A(P,W) >= 2,
      (a in W or b in W),
      c_A(P,W) + [a in E(P,W)] + [b in E(P,W)] >= 4 }.
```

`DefenderWinsFirst(P,a,b)` holds when a turn-start live defender window with
count at least four contains neither placement. `tau(F)` is the least cardinality
of a coordinate transversal of every set in finite family `F`, with
`tau(empty)=0` and `tau(F)=infinity` if no finite transversal exists. V1 only
needs the exact cases `0`, `1`, `2`, and `>2`; an empty member gives `>2` for
the defender budget.

Disposition uses this total precedence:

1. illegal replay or a nonclaimant terminal prefix is not a disposition and
   makes the purported root/specification invalid;
2. claimant terminal after `a` or `b` is `ClaimantCompletion`;
3. empty `Hpair_A` is `Fail(NoNewClaimantThreat)`;
4. otherwise `DefenderWinsFirst` is `Fail(DefenderWinsFirst)`;
5. otherwise `tau = 0` or `1` is `Fail(LooseReply(tau))`;
6. otherwise `tau = 2` is `TightPair(Pab)`; and
7. otherwise `tau > 2` is `ClaimantTactical`.

Completion and claimant tactical dispositions are positive constructors. A
negative artifact encountering either MUST reject; they can never be encoded
as a failing complement.

**HYPOTHESIS — exact quotient and expansion.** Define a relation only between
reverse ordered occurrences. `(a,b) ~ (b,a)` exactly when both occurrences are
in `U(P)`, both singleton prefixes are nonterminal, and both ordered replays
produce the identical full semantic state including owner map, mover, phase
payload, and placement clock. Its two-member class has
`PairKey = Commuting(min_raw(a,b),max_raw(a,b))`. Every other occurrence,
including a sole reverse orientation, has
`PairKey = Ordered(first=a,second=b)`; orientation MUST NOT be inferred from an
unordered key.

Each class is classified from every member; members MUST agree after the guarded
commutation check. Every occurrence in `U(P)` maps to exactly one class. Every
failing class expands uniquely to its derived `PairComplement`; every tight
class maps to exactly one stored future full-tree disposition; any completion
or tactical class prevents a negative node. Thus the conceptual expansion is a
total function from all `Q` ordered occurrences, not merely from stored `K`.

**HYPOTHESIS — required evidence.** Tests MUST exhaustively compare bounded
small direct-board states to a third, simple specification oracle. Mandatory
fixtures include the weak count-one `G1` promotion (NCE-03), a stale
turn-start defender block, sole-orientation keys, both quotient orientations,
hidden terminal prefixes, and every disposition priority. Telemetry MUST report
`|T|`, every `|S(P,a)|`, `Q`, quotient class count, all four failing counts,
positive completion/tactical counts, and stored `K`; producer/verifier equality
is only an implementation gate.

### 2.3 Equality-only defender Universal nodes

**HYPOTHESIS.** At a nonclaimant state let `b = B(P.phase())` and let

```text
H_A(P) = { E(P,W) | live_A(P,W) and c_A(P,W) >= 4 }.

K_b(P,A) =
  { d in union(H_A(P)) |
      Legal_P(d) and tau({ H - {d} | H in H_A(P), d notin H }) <= b-1 }.
```

The residual notation drops threats hit by `d`; equivalently, `d` belongs to a
size-`b` transversal. `K_b` is complete, raw-sorted, and duplicate-free. At
`tau=b`, generic T6 licenses omission of replies outside this kernel. V1's
positive grammar has a Universal constructor **only** at `tau=b`; it does not
import generic T6's full-legal-set behavior at `tau<b`.

The negative polarity is exact: a claimant Choice is refuted only if every
admitted pair class is refuted, while one independently checked member of exact
`K_b` refutes a positive Universal. At defender `FirstStone`, an eventual
full-tree wire unfolds `d1 in K_2` and then `d2 in K_1` as two ordinary nodes;
an atomic search `DefenderPair` is never proof evidence.

Group-2, ranked zones, FHW gates, substitute replies, certificate-relative
zones, and unforced-turn quotients are outside the class and MUST reject.

### 2.4 Closed negative leaves — amended per R2, R4, R2-1

**HYPOTHESIS.** The state/polarity acceptance matrix is exhaustive:

| leaf | allowed state | directly rederived premises |
|---|---|---|
| `NoAdmissibleFirstTurn` | claimant nonterminal `FirstStone` | no claimant terminal or own-win-now; not opponent-forced tactical; exact expansion of `U(P)` contains no completion, claimant tactical, or `TightPair` class. |
| `EmptyClaimantThreatFamily(b)` | nonclaimant phase with matching `b` | nonterminal; no opponent own-win-now; `H_A(P)` is empty. |
| `LooseDefenderBoundary(b,tau)` | nonclaimant phase with matching `b` | nonterminal; no opponent own-win-now; `H_A(P)` nonempty; exact `tau < b`. |
| `OpponentTerminal(D)` | any supported recursive state | direct terminal replay names `D`; claimant terminal rejects. |
| `OpponentOwnWinNow(b)` | nonclaimant phase | direct `OwnWinNow_D(P,b)` and no prior terminal. |
| `OpponentForcedTactical(2)` | claimant `FirstStone` | no claimant own-win-now and exact `tau(H_D(P)) > 2`. |

Unknown tags or reasons reject. `tau>b` at a nonclaimant state is claimant
tactical and is absent from this negative table. `NoAdmissibleFirstTurn` is the
sole compact v1 leaf because its exact premises are the closed row above and
the authoritative conjunction below. The verifier reruns the complete expansion
under the work limits.

**HYPOTHESIS — one authoritative leaf eligibility predicate.** The following
named conjunction is the only meaning of producer eligibility, producer
self-verification eligibility, and the section 6 leaf-eligibility promise. The
arguments `policy`, `profile`, `expansions`, and `node_cap` are trusted
solve-local inputs, not artifact-selected limits:

```text
RefuteLeafExactEligibleV1(
    P, A, reachable, policy, profile, expansions, node_cap) :=
  ruleset = HexoConnect6/Opening1-Then2/Win6/LegalRadius8/I16/V1
  and coordinate = AxialQR/D6SafeClosure/V1
  and class = VcfPairComplete/EqualityDispatch/V1
  and wire = RefuteLeafExact/V1
  and ReachableRootV1(P, ruleset)
  and reachable is the trusted token for that exact P and ruleset
  and P is post-opening and nonterminal
  and P's stones are raw-lexicographically sorted, duplicate-free, valid i16
      coordinates paired with closed owner values, and bind exactly to the
      literal root header
  and P.phase() = FirstStone with the complete zero-byte phase payload
  and A = P.current_player() = P.mover()
  and nextPly = P.placements_made() + 1 without overflow
  and D6ClosedV1 holds for the root and every coordinate encountered by the
      complete regeneration below
  and policy is a caller-selected OfflinePolicyV1 at or below every section
      2.5 compiled ceiling, and regeneration stays within every selected
      count, allocation, state-byte, heap, CPU, and wall limit
  and profile = LeafNaturalWidthExhaustExactV1, meaning ordinary search reports
      natural width exhaustion after all staged reopens and bottom-up PN
      refreshes, semantic_horizon = u32::MAX, the exact v1 width options were
      used, and Header.profile_u16 = 1; this is eligibility metadata and does
      not assert a Structural plan or supply logical leaf evidence
  and node_cap is the externally selected solve cap
  and expansions < node_cap
  and the earlier ClaimantTerminal constructor is absent
  and not OwnWinNow_A(P,2)
  and not ForcedLoss_A(P,2)
  and producer-side direct regeneration completely constructs T(P), every
      G1(P,a), every S(P,a), all Q ordered occurrences of U(P), the guarded
      quotient classes, both ordered members of every two-member class, every
      prefix/full replay, and every disposition under section 2.2
  and completion_occurrences = 0
  and claimant_tactical_occurrences = 0
  and tight_pair_occurrences = 0.
```

The last three zeroes, after complete regeneration, imply that every ordered
occurrence has one of the four failing dispositions and hence that the admitted
set is empty. “The admitted set is empty” is only a derived description after
all conjuncts above have been established; it MUST NOT stand alone as a synonym
for `RefuteLeafExactEligibleV1`. In particular, an empty admitted set does not
override `ForcedLoss_A(P,2)`, an earlier claimant-positive constructor, a
profile mismatch, `expansions >= node_cap`, or a policy/D6 failure. None of
those cases enables a different v1 leaf tag.

`NoJointCarrier` is removed from full-tree v1 and the fallback. It may be
reconsidered only in a new class/wire version after a present model theorem,
pinned source, exhaustive bounded-state testing, and measured advantage over
exact enumeration.

### 2.5 Coordinate closure and graph/work bounds — amended per R6, R8

**CODE-FACT.** The engine stores axial components as `i16`; current checked D6
transforms can fail when an image leaves that type. Search uses D6 only for tie
ordering and retains raw exact equality
([`tss_solver.rs` lines 10318-10370](../packages/hexfield_eq/rust/src/tss_solver.rs#L10318)).

**HYPOTHESIS — D6-safe closure.** For axial `(q,r)`, let its cube triple be
`(q,r,-q-r)`, computed in checked `i32`. The accepted root domain requires the
absolute value of every cube component of every root stone to be at most
31,480. This fixed hexagonal domain is closed under all twelve D6 actions and
is checked before semantic proof work. The margin covers at least five cube
units per placement through the future depth-256 ceiling plus a final
length-six-window scan while remaining inside positive `i16::MAX`.

In addition, `D6Safe(c)` means all twelve axial images of `c`, computed first
in checked `i32`, have both components representable as `i16`. An input is
`D6ClosedV1` only when the fixed root-domain check passes and `D6Safe` holds for
every coordinate subsequently encountered while directly enumerating windows,
`T`, each `S`, pair successors, threat families, `K_b`, and decoded replies at
every accepted node. The check happens before a discovered coordinate is
converted or used. If semantic enumeration discovers an unsafe coordinate,
the root is unsupported and no artifact is accepted.

This predicate is closed under all twelve actions. Every accepted artifact can
therefore be transformed, fully re-sorted and rehashed, and verified against
all twelve transformed roots. Original bytes MUST fail against a distinct
image. No D6 image or hash permits proof-node reuse. The `-32768` construction
in NCE-06 is rejected by the root preflight.

**HYPOTHESIS — external policy.** Limits are selected by the trusted offline
caller and are not read from artifact bytes. A caller may lower them. The
compiled `OfflinePolicyV1` MUST NOT permit values above these design ceilings
without a new memory/work review:

| resource | hard ceiling |
|---|---:|
| wire bytes | 8 MiB |
| root stones | 4,096 |
| deduplicated direct windows | 80,000 (covers `18 * (root_stones + depth)` with checked headroom) |
| `|T|` | 4,096 |
| any `|S(P,a)|` | 4,096 |
| total ordered `Q` | 2,000,000 |
| threat-family memberships visited | 8,000,000 |
| pair-gate primitive membership tests | 16,000,000 |
| `K_b`/transversal primitive membership tests | 8,000,000 |
| future full-tree DAG nodes / depth | 100,000 / 256 |
| retained exact-state bytes | 64 MiB |
| verifier peak heap | 256 MiB |
| verifier CPU / wall | 30 s / 60 s |

The 4,096-stone bound replaces the unsupported one-million-stone value. The
direct-window bound covers the root plus the future depth ceiling; `Q` and operation
caps bound the omitted complement. Checked counters MUST be charged before the
operation and checked arithmetic MUST precede allocation. Pair evaluation MUST
stream and MUST NOT allocate a `T x T` table. Exact-state retention includes
all keys, owner maps, phase payloads, and memo overhead, not only serialized
coordinates.

CPU/wall cancellation is checked at deterministic work checkpoints by the
supported offline API. Crossing any count, CPU, wall, heap, or allocation
budget returns `UnsupportedPolicyBudget`; malformed structure returns
`Rejected`. Neither result is a semantic negative leaf. Valid-but-hostile and
malformed inputs MUST terminate deterministically within the selected budget.

### 2.6 Size and work model — amended per R6, R7, R2-1

**HYPOTHESIS.** Required causal metrics are:

- `R`: root stones; `W`: deduplicated direct windows;
- `A`: fresh-turn attacker propositions; ordered `Q`; quotient classes; and
  stored tight `K` dispositions;
- stored edge occurrences, `U_d`: defender counter nodes, typed leaves `L`,
  unique propositions, and retained exact-state-key bytes;
- threat-family memberships, pair-gate operations, and kernel/transversal
  operations; and
- encoded bytes, producer CPU/wall/heap, verifier CPU/wall/heap, and cold-search
  CPU/wall.

Logical full-tree replay is `O(Q + defender-family work + edge occurrences)`;
complement encoding can make bytes `O(A + K + U_d + L)` while leaving replay
quadratic in `|T|` until the absolute `Q` cap intervenes. No byte or node-count
density is a proxy for regenerated work.

The earlier 10-16 bytes per “reported exhausted node” estimate and the
17,957-node `mvp2lvc` premise are withdrawn. The committed 226-node
`l9mxn59` row does not supply `Q/K/U/L`, edge, state-byte, or emission evidence;
the available `mvp2lvc` row is cap-bound `Unknown`, not a pinned natural
exhaust. Neither witness may authorize size, LOC, or performance claims until
the manifest records the exact command, flags, binary hash, termination reason,
arena nodes, expansions, edge occurrences, all metrics above, wall/CPU/heap,
and raw output.

Reports MUST give per-root median/p95/max and aggregate weighted totals. Leaf
v1 MUST be compared with a predeclared competent `RootPlusEmptySetV1` archival
baseline containing the same canonical root and semantic identity; its bytes
MUST be no more than 110% of that baseline on every root and in aggregate. A
future full-tree format MUST be smaller than a compact, competently implemented
stored-support representation with the same root and semantic keys both in
aggregate and on every named gate witness. On a common exact-empty root it MUST
emit the leaf-v1 representation byte-for-byte instead of a larger tree; here
“exact-empty” means that every semantic conjunct of
`RefuteLeafExactEligibleV1`, not merely an empty admitted set, has been
rederived. An intentionally verbose baseline is inadmissible.

## 3. Producer

### 3.1 Emission seam

**CODE-FACT.** The wide arena is live after `search.run` and before current
positive materialization
([`tss_solver.rs` lines 1472-1503](../packages/hexfield_eq/rust/src/tss_solver.rs#L1472)).

**HYPOTHESIS.** Leaf v1 uses a solve-local, read-once
`TSS_REFUTE_CERT_V1=off|emit` flag. `off` takes the historical path with no new
allocation, ordering, statistic, cache entry, digest, or output byte. `emit`
does no work until ordinary search terminates and cheap root/profile/policy
eligibility succeeds. It then performs independent producer-side exact
enumeration. The side artifact has its own type; ordinary status remains
`Unknown`. Only a root satisfying `RefuteLeafExactEligibleV1` enters producer
self-verification through the public independent entry point, and rejection
only drops the artifact and records isolated telemetry.

### 3.2 Eligibility and future full-tree support — amended per R3, R2-1

**HYPOTHESIS — leaf v1.** A producer is eligible if and only if the complete
`RefuteLeafExactEligibleV1(P,A,reachable,policy,profile,expansions,node_cap)`
conjunction in section 2.4 holds. An empty admitted set alone is insufficient.
The predicate includes the literal root/reachability/phase/claimant/D6/policy
premises, the declared natural-exhaust exact profile, strict
`expansions < node_cap`, absence of all earlier root constructors, and complete
`U` regeneration with zero completion, claimant-tactical, and `TightPair`
occurrences. It does not rely on an arena zero.

Exact occurrence and quotient counts are encoded. Producer self-verification
MUST first re-establish that same named conjunction from its solve-local
eligibility context and then invoke the public independent verifier; this
two-part result, not verifier acceptance in isolation, is “self-verifies” in
the producer and section 6 promises. A false semantic conjunct returns a typed
`NotRefuteLeafExactSemantic(reason)`; a profile mismatch returns
`IneligibleLeafProfile`; `expansions >= node_cap` returns
`IneligibleNodeCap`; and a selected-policy overrun returns
`UnsupportedPolicyBudget`. Bytes are emitted only when the named conjunction
and public verification both succeed. These typed results are non-evidence and
do not authorize another leaf tag.

**HYPOTHESIS — mandatory algorithm before any full-tree version.** A full-tree
producer MUST first compute a bottom-up memoized classification keyed by exact
arena entry/direct-edge proposition:

```text
Support = Structural(negative_plan) | Unresolved(cause_set)

Cause = DepthCutoff | HorizonRefusal | CensusDismissal | NodeCap
      | StalledOrLazyFrontier | UnsupportedDefenderBoundary
      | ClaimantPositiveLeaf(kind) | OpponentStructuralLeaf(kind)
      | UnexplainedCurrentRefuted | PolicyBudget
```

`OpponentStructuralLeaf(kind)` records the typed replay cause and produces a
`Structural` leaf; it is retained in provenance telemetry rather than confused
with a generic zero. Claimant terminal/completion/tactical produces
`Unresolved({ClaimantPositiveLeaf(kind)})` for a negative plan. A generic
current-source `Refuted` is never structural without direct reclassification.

The fixed point MUST obey:

1. Reconstruct the exact semantic state and apply the closed constructor
   priority before consulting PN numbers.
2. At claimant Choice, independently regenerate every admitted quotient class.
   `Structural` requires a structural child plan for every tight pair; one
   unresolved child makes the Choice unresolved, and a completion/tactical
   pair makes it claimant-positive.
3. At defender Universal with exact `tau=b`, recursively classify every member
   of exact `K_b`. The result is structural if at least one member has a
   structural plan. Choose the raw-lexicographically least **successful plan**;
   continue past earlier unresolved members. If none succeeds, union their
   cause sets.
4. An atomic `DefenderPair` edge is structural only after both placements are
   unfolded, both intermediate transitions and kernels are checked, and the
   final child plan succeeds.
5. Memo reuse requires complete state/claimant/phase-clock equality, including
   `SecondStone.first`; recursion through an in-progress key is a cycle and
   unresolved. Only a completed plan may become a backward wire reference.
6. Derive `NaturalExhaust` from the completed root fixed point after all stage
   reopens and refreshes. The search termination enum is telemetry only.

The current shallow helper at lines 5908-5919 is explicitly insufficient. The
mandatory NCE-01 family includes a lexicographically earlier descendant cutoff,
a later structural reply, reversed order, lazy thunks, horizon-refuted nodes,
and the current both-winners-to-`Refuted` terminal arm. NCE-08 requires claimant
terminal replay to fail negative materialization and opponent terminal replay
to become the typed structural leaf.

### 3.3 Cost, isolation, and utility — amended per R3, R6, R7

**HYPOTHESIS.** Producer enumeration and any future support fixed point are
subject to the same externally supplied semantic-work ceilings as verification.
They may not invoke PN search, deepen, expand a new recursive state, consult a
persistent proof cache, or mutate a child result. The future classifier is one
memoized pass over available arena propositions plus bounded semantic
regeneration; alternative defender plans are classified once, not searched.

The prototype purchases one declared utility: a portable archival class fact
expected to receive `N = 3` independent offline audits. Let `S` be cold search
CPU/wall, `E` all post-search emission cost including regeneration, support
planning if any, encoding, and self-verification, and `V` one standalone
verification. All measurements use the same frozen root and binary.

Hard producer and end-to-end gates are:

- absolute per-root producer CPU/wall is at most 30 s/60 s and peak additional
  heap at most 256 MiB, with lower caller limits honored;
- aggregate `E < 0.25*S`; producer p95 is below `0.35*S` and max below
  `0.50*S` on matched roots;
- aggregate enabled solve `S+E < 1.25*S`; per-root p95 and max are below
  `1.40*S` and `1.50*S`, respectively; and
- amortized archival cost satisfies `E + 3*V < 3*S` both in aggregate and for
  every preregistered named gate witness.

Sub-millisecond roots may be batched for ratios, but absolute totals and maxima
remain reported. If three audits are not a real owner workflow, the utility
premise fails and the artifact-only prototype is economically rejected. The
NCE-05 schedule (`E=0.70*S`) fails both producer and end-to-end gates even if
verification alone is faster.

Flag-off status, positive bytes/digests, stats, nodes, TT signatures, logs, and
corpus output MUST be byte-identical. With emission enabled, a noneligible root
performs only bounded cheap preflight and no semantic enumeration or arena scan.
Any hot-path window snapshot, failure vector, or refutation-tree retention is a
scope alarm requiring hostile review.

## 4. Independent verifier arm

### 4.1 Trust-base and API contract — amended per R5, R8

**HYPOTHESIS.** V1 lives in a new isolated module such as
`tss_refute_verify.rs`; data-only wire constants may live in
`tss_refute_leaf_cert.rs`. It MUST NOT modify `tss_verify.rs`. Its public result
is, for example:

```text
VerifiedClassRefutation {
  ruleset, coordinate_version, class_version, wire_version,
  root_semantic_sha256, claimant, reachable_root_token
}
```

It is not convertible to `ProofStatus`, `HardValue`, or `TssCertificate`.

**HYPOTHESIS — deliberately shared trust base.** Runtime dependencies are
transitively allowlisted to:

- Rust core/std checked integer, slice, allocation, and time primitives;
- raw `HexCoord` and `Player` value types and the frozen phase value type;
- read-only canonical stone iteration plus engine placement/terminal primitives,
  each used only alongside the independent direct-state cross-check;
- a reviewed SHA-256 implementation; and
- verifier-private codec, policy counters, direct board map, direct geometry,
  transition, terminal, window, transversal, and semantic functions reachable
  only from the refutation verifier.

Sharing engine transitions means independence concerns theorem analysis rather
than the entire game engine. Before an engine-applied child contributes to
acceptance, verifier-private direct state must independently check legality,
owner insertion, mover/phase/`SecondStone.first` transition, placement clock,
and terminal result and require exact agreement.

The transitive denylist includes all `tss_solver` and positive-verifier modules,
`WidePnSearch`, `WidthOptions`, every producer generator/order/canonicalizer,
`Board::windows`, `WindowStore`, `threats::analyze`/`threats_shared`, and any
shared or “neutral” helper computing live windows, `T/G1/S/U`, threat families,
transversals, `K_b`, quotient classes, gate outcomes, normalized semantic
decoding, or semantic successors. Moving forbidden logic behind another name
does not make it allowed.

Data-only tag constants, plain wire structs, and SHA-256 primitives may be
shared. Producer semantic normalization/decoding may not be shared unless the
literal decoder is separately modeled, golden-tested, and approved in the
call-graph review.

### 4.2 Re-derivation obligations — amended per R1, R2, R4, R5, R2-1, R2-2

**HYPOTHESIS.** After strict decoding, version/policy checks, reachable-token
validation, D6 preflight, and exact root binding, the verifier MUST:

1. Rebuild `RootSemanticPreimageV1` byte-for-byte under section 2.1, hash it,
   and require exact equality with `root_semantic_sha256`; a typed tuple hash,
   native struct serialization, or hashing a convenient header slice is invalid.
2. Build its private direct board from canonical stones and enumerate/deduplicate
   literal `(axis,start)` windows; it MUST not read an incremental window store.
3. Reconstruct exact `T`, every `G1/S`, ordered `U/Q`, quotient class, pair
   family, terminal prefix, defender precedence, `tau` case, and total
   disposition under section 2.2 while charging work before execution.
4. Prove that there is no claimant terminal constructor, no `OwnWinNow_A(P,2)`,
   and `not ForcedLoss_A(P,2)` before accepting `NoAdmissibleFirstTurn`.
5. Require complete classification of all `Q` ordered occurrences and zero
   completion, claimant-tactical, and `TightPair` occurrences. “Empty admitted
   set” alone is not an acceptance test.
6. Require exact equality of `T/Q/quotient-class/disposition` telemetry and all
   redundant payload counts. Each `fail_*` is checked as an ordered-occurrence
   count; its two sum identities and the distinct derived class-count identity
   in section 2.1 MUST hold. A producer count never limits the verifier loop.
7. Cross-check every engine placement, phase transition, semantic successor,
   and terminal result against the private direct state.
8. Conclude `BoundaryFreeNo` using the closed matrix, then use the model soundness
   and clock-lifting layers; it MUST NOT appeal to search exhaustion.

Before a future full-tree version, the same arm must additionally derive exact
`K_b`, validate one selected equality-boundary reply, replay every DAG reuse,
and reject claimant-positive constructors at every state. The future work does
not broaden leaf-v1 acceptance.

Tests MUST include independent golden vectors from a third simple oracle. Each
root-identity vector pins the literal canonical-preimage bytes in hexadecimal
and the resulting 32 digest bytes. Counter vectors MUST separately pin (1) a
sole-orientation failing class, where `Q=1`, `quotient_class_count=1`, and the
selected `fail_*` value is `1`, and (2) a two-member commuting failing class,
where `Q=2`, `quotient_class_count=1`, and the selected `fail_*` value is `2`.
All other failing fields are zero in those focused vectors. At least one vector
MUST combine digest and counter expectations in the same literal artifact.

One-sided defect injection MUST omit a weak promotion in only one
implementation, retain one stale defender window, flip each `tau` case, corrupt
`SecondStone.first`, change only one transition/terminal result, alter one
domain/preimage byte, and count a two-member quotient as one occurrence. Seeded
producer/verifier agreement alone is insufficient.

### 4.3 Enforced fail-closed rules — amended per R2, R4, R5, R6, R8, R2-2

**HYPOTHESIS.** Verification returns `Rejected` on malformed or semantically
false input and `UnsupportedPolicyBudget` on an externally imposed resource
limit. It MUST never partially accept, panic, or turn either result into a
weaker claim. It fails on the first of:

- bad magic, noncanonical integer, trailing byte, checksum, count, length,
  allocation preflight, or unknown field/tag;
- any ruleset/coordinate/class/wire/profile mismatch or an unproved semantic
  compatibility path;
- root/reachable-token mismatch in stone, owner, mover, full phase payload,
  clock, claimant, terminal status, semantic digest, or D6-safe closure;
- any nonexact `RootSemanticPreimageV1` domain, field order, numeric encoding,
  length treatment, phase payload, or digest result;
- unsorted/duplicate stones or pair facts, invalid owner/coordinate, illegal
  move, hidden terminal mismatch, wrong quotient orientation, or unequal direct
  and engine successor;
- any regenerated `T/S/U/Q`, class, reason, count, or disposition mismatch,
  including treating a `fail_*` field as a quotient-class count or violating
  either occurrence/class sum identity in section 2.1;
- any claimant completion/terminal/tactical constructor or a false closed leaf;
- any depth/horizon/cap/census/zone/Group-2/quotient resource tag;
- any semantic-work, exact-state-byte, CPU, wall, or heap budget excess; or
- for a future DAG, bad/backward IDs, cycle/orphan, nonidentical reuse,
  `SecondStone.first` alias, bad `K_b`, loose Universal, or unexplained
  current-source `Refuted`.

CI MUST enforce the firewall in three ways: a source/import denylist; a compiled
call-graph reachability audit from the public verifier entry point against
forbidden symbols/modules; and one-sided mutation builds plus third-oracle
golden tests. A waiver, indirect call, generated table, or newly reachable
semantic helper fails the build until hostile review. CI records the audited
compiler, feature set, symbol map, and allowlist hash.

### 4.4 Replay and lifecycle cost bar — amended per R6, R7

**HYPOTHESIS.** Measurements use a quiet single-thread release binary, warmed
code pages, fresh solver/verifier state, at least 30 batched repetitions for
small roots, a preregistered training cohort, and an untouched held-out cohort.
Reports include median/p95/max and weighted totals for all causal metrics in
section 2.6, including absolute `Q`, operation counters, CPU, wall, and heap.

Hard replay requirements are logical ANDs:

- total verifier CPU and wall are each below 75% of matched cold rerun search;
- each preregistered named witness median is below rerun, with a target at or
  below 50%, and verifier p95 **and max** are below matched rerun tails;
- every artifact remains within the absolute work/time/heap policy; and
- verification invokes no search and performs no semantic operation without
  charging the appropriate absolute counter.

“At most once” is reported for memo multiplicity but is not a work bound. The
producer/end-to-end/amortization gates in section 3.3 are equally mandatory.
Malformed and valid-but-hostile NCE-02 roots are in the robustness battery and
must terminate with the same deterministic budget classification.

## 5. Consumption roadmap (post-v1, informational) — amended per R1

**HYPOTHESIS.** V1 has no live consumer. An archival record carries the typed
literal proposition arguments, semantic versions, root identity, claimant,
and reachability token. Any later trainer, atlas, corpus, or harness consumer
is a separate owner decision and MUST preserve `Unknown` as ordinary game
status.

Potential future uses are a categorical
`NoContractWin(VcfPairComplete/EqualityDispatch/V1)` auxiliary target, a
distinct atlas `CERTIFIED-NO(class)` label, a corpus certification column, and
disproof-coverage telemetry. None may become `-1`, full-game `Loss`, a forced
opponent move, search pruning, a proof cache, or imported atlas truth. The
executed-byte/model correspondence is a separate later proof round.

## 6. Gates, NCE disposition, and kill criteria — amended per R2, R3, R5, R6, R7, R8, R2-1, R2-2

**HYPOTHESIS.** Before measurements, freeze hashes/manifests for training and
held-out cohorts, exact commands, solver flags, binary/compiler/features,
policy limits, caps, natural/cap termination reasons, and raw outputs. Cohort
movement after results are visible invalidates the campaign. The two historical
full-tree witnesses are evidence rows only until independently pinned; leaf v1
does not promise they emit.

Every required-result cell below is a logical AND. Failure of any applicable
gate stops the artifact cut; there is no “fail both” exception.

For this section, “eligible” means the complete, identically named
`RefuteLeafExactEligibleV1(P,A,reachable,policy,profile,expansions,node_cap)`
conjunction in section 2.4 and nothing weaker. In particular, “empty admitted
set” never abbreviates away the root, reachability, phase, claimant, D6,
selected-policy, exact-profile, strict-cap, earlier-constructor, forced-loss, or
complete-regeneration conjuncts.

| gate | required result | hard stop |
|---|---|---|
| Leaf eligibility | Every root satisfying `RefuteLeafExactEligibleV1` emits and self-verifies in the exact two-part sense of section 3.2; every other root emits no bytes and returns its typed semantic, D6, policy, `IneligibleLeafProfile`, or `IneligibleNodeCap` reason. Thus a semantic leaf at a wrong profile or with `expansions >= node_cap` is explicitly ineligible. | Silent fallback, use of PN zero as evidence, emission without every named-predicate conjunct, or failure to emit after every conjunct and public verification succeed. |
| Acceptance | Independent verifier accepts 100% of emitted training and held-out artifacts through the public entry point. | One emitted rejection. |
| No false scope | No cap/depth/horizon/census/Group-2/zone/opening/claimant-SecondStone/root-policy failure emits. | One such emission. |
| Class boundary | Ordinary status remains `Unknown`; no hard value, Loss, trainer backup, or full-game label is minted. | Any game-value exposure. |
| Flag isolation | Flag-off named observables are byte-identical; enabled noneligible roots perform no semantic scan and preserve search/output identity. | Any unexplained difference or search regression. |
| Specification | Exhaustive bounded-state oracle comparison and every R2 adversarial fixture pass; exact telemetry is recorded; independent goldens pin the exact root-digest preimage/digest and the sole-orientation and two-member-quotient occurrence counters required by sections 2.1 and 4.2. | One oracle contradiction, digest/preimage mismatch, counter-unit/sum mismatch, omitted occurrence, or ambiguous orientation. |
| Mutation | 100% rejection of root/version/policy/count/owner/phase/claimant/reply/order/checksum-only/terminal/leaf mutations and unknown tags. | One semantic mutation accepted. |
| D6/domain | Preflight rejects unsafe roots; all twelve rebuilt images of every accepted artifact verify; original bytes fail on a distinct image. | Unsafe acceptance, failed accepted-root image, or cross-root acceptance. |
| Semantic work | All section 2.5 counters, memory, CPU, and wall ceilings hold on valid, malformed, and hostile roots including NCE-02. | Uncharged/unbounded work, budget-selected artifact, or nondeterministic overrun. |
| Firewall | Source and compiled transitive call graphs pass; third-oracle goldens and one-sided fault injections fail the altered implementation. | Shared semantic truth, forbidden reachable symbol, or correlated fault acceptance. |
| Size/baselines | Causal denominators and exact bytes are reported median/p95/max/aggregate; no node proxy is used; leaf bytes are `<=110%` of the competent compact leaf baseline. Any future full tree beats the compact stored-support baseline in aggregate and on every named gate witness and uses leaf bytes on common roots satisfying `RefuteLeafExactEligibleV1`. | Any size conjunct fails, an unpinned witness claim is used, tail/held-out data is missing, or the baseline is intentionally weak. |
| Replay | Every section 4.4 conjunct passes. | Any replay conjunct fails. |
| Producer/end-to-end | Every section 3.3 absolute, relative, tail, and three-audit conjunct passes. | Any producer, enabled-workflow, or amortization conjunct fails. |

The R2-1 fixture set is mandatory and does not authorize another wire tag:

1. A reachable, nonterminal claimant `FirstStone` root has an empty admitted
   set, no claimant own-win, and an opponent forced tactical construction with
   `ForcedLoss_A(P,2)` (for example, three independently hittable defender
   count-four families). It MUST fail `RefuteLeafExactEligibleV1`, emit no
   bytes, and a forged tag `0x20` artifact MUST be rejected.
2. A realizable nonterminal claimant `FirstStone` root has an empty admitted set
   but has the earlier claimant-positive `OwnWinNow_A(P,2)` constructor. It MUST
   fail the named predicate, emit no bytes, and a forged tag `0x20` artifact
   MUST be rejected. Separate terminal-root coverage MUST likewise show that
   the earlier `ClaimantTerminal` constructor cannot enter the leaf cut.
3. A root satisfies every semantic leaf premise, including complete
   regeneration and all three zero disposition counts, but is run (a) with a
   profile other than `LeafNaturalWidthExhaustExactV1` and (b) with
   `expansions == node_cap`. The cases MUST return `IneligibleLeafProfile` and
   `IneligibleNodeCap`, respectively, emit no bytes, and perform no public
   self-verification. The strict-cap case MUST NOT be treated as natural
   exhaustion.

The R2-2 golden set is also a gate artifact, not an implementation-generated
snapshot: a third simple oracle independently supplies the exact preimage hex,
digest bytes, `Q`, quotient-class count, and four `fail_*` values. It includes
both counter shapes fixed in section 4.2 and is checked unchanged by producer,
verifier, and model-codec tests.

**HYPOTHESIS — explicit counterexample closure.** None of the eight review
counterexamples is accepted residual risk:

| NCE | disposition |
|---|---|
| NCE-01 | Neutralized by R3: full-tree promotion requires recursive fixed-point classification, tries later structural `K_b` members, and is order-invariant. Leaf v1 has no recursive selector. The full variant remains a mandatory promotion test. |
| NCE-02 | Neutralized by R6: `R/W/T/S/Q`, membership-operation, state-byte, heap, CPU, and wall caps are external and charged before work; pair evaluation streams. |
| NCE-03 | Neutralized by R2/R5: literal `G1` includes count-one-through-`a`, all sets use turn-start `P`, and a one-sided omission plus third-oracle vector is mandatory. |
| NCE-04 | Neutralized by exact identity and R5: `SecondStone.first` participates in state, direct transition, memo equality, mutation tests, and root/node replay. |
| NCE-05 | Neutralized by R7: `E < 25% S`, enabled solve tails, absolute producer cost, and `E+3V < 3S` are hard gates; its 70% emission regression fails. |
| NCE-06 | Neutralized by R8: every semantic coordinate must satisfy checked `D6Safe`; the extreme `i16` root is unsupported before proof work. |
| NCE-07 | Neutralized by R1/R4: the published grammar is equality-only; nonempty `tau<b` is the typed `LooseDefenderBoundary`, while `tau>b` is claimant-positive. Generic full-T6 syntax is not imported. |
| NCE-08 | Neutralized by R3/R4/R5: direct terminal replay distinguishes the winner; claimant terminal blocks negative materialization, opponent terminal has a typed leaf, and generic `Refuted` has no authority. |

Soundness stops—false acceptance, incomplete ordered coverage, boundary evidence,
version drift, unsafe D6 admission, shared semantic truth, or full-game exposure—kill
the design, not merely one implementation. Economic gates are also independent
hard stops after one codec-only optimization round; a size win cannot excuse a
replay or producer loss, and vice versa.

## 7. Re-audited manageability verdict — amended per R1, R4, R7

**HYPOTHESIS.** R1's literal semantics, R3's fixed point, R5's enforceable
firewall, R6's operational budgets, and R7's lifecycle campaign add real work;
removing `NoJointCarrier` does not offset it. The amended full-tree estimate is:

| full-tree component | estimated Rust/test LOC |
|---|---:|
| producer trigger, recursive support fixed point, canonical compaction, telemetry | 650-900 |
| strict full-tree codec and exact-state DAG machinery | 300-450 |
| independent direct-state verifier and policy accounting | 1,100-1,500 |
| oracle, call-graph, mutation, D6, hostile-work, corpus, and lifecycle tests | 1,200-1,600 |
| benchmark/baseline tooling | 300-450 |
| total | 3,550-4,900 |

**HYPOTHESIS — verdict: FULL-TREE V1 IS TOO BIG / NO-GO.** This exceeds the
owner's roughly 2.5k-LOC manageable envelope, and its large-witness economics
are unpinned. The design MUST NOT quietly absorb that growth. `l9mxn59` and
`mvp2lvc` are therefore removed as v1 emission gates; they remain future
full-tree evidence targets after pinned natural-exhaust measurements.

**HYPOTHESIS — amended v1 cut: exact leaf only, still review-gated.** The only
implementation-sized cut is `RefuteLeafExact/V1`:

- reachable, nonterminal, claimant `FirstStone`, fixed equality-only class;
- one literal `NoAdmissibleFirstTurn` payload after complete `T/G1/S/U`
  regeneration;
- no DAG, Universal node, recursive support, catch-all, compact theorem leaf,
  consumer, cache, or game-value conversion; and
- the full R5 firewall, R6 work budgets, R7 lifecycle economics, semantic
  version binding, and D6-safe closure remain mandatory.

Its planning range is 1,600-2,300 Rust/test/tooling LOC: producer/codec
`350-500`, verifier/direct geometry `550-800`, and oracle/firewall/mutation/D6/
resource/economic tests `700-1,000`. This is within but near the original
envelope. The verdict is **NO-GO pending a hostile review of this amended
document**; after that review, it may become a conditional owner GO only if all
section 6 gates are preregistered.

The full phase-indexed mathematics is retained because even the exact leaf must
prove absence of every root positive constructor. Any later recursive artifact
must use a new wire version, implement the R3 algorithm, pin its evidence, and
return to hostile design review. Arbitrary phases, bounded-horizon negatives,
Group-2/FHW/ranked zones, unforced defender quotients, refutation caches,
trainer backup, and `tss_verify.rs` changes remain outside the cut.
