# C-REL round 1: relative certificate design and obligations

Date: 2026-07-17

Design target: consolidated engine at `6ef67cfe49dfe4f016cab866d267ea07ff58d1ef`

NQ3 evidence base: `hunt-cert-support` at `3cd224fe3ed5b06084b526e8532e9de53c8d620c`

Round: design only; no build, solve, or experiment was run

## Verdict up front

**GO, but only for an additive strict-discharge project. NO-GO for a new
support-only verifier.**

`C_rel` v1 should be a versioned, rootless, relative-clock proof template plus
a finite candidate-selection interface. It is not itself proof authority. For
a target position it deterministically materializes an ordinary
`TssCertificate` whose root is the target's complete `RootBinding`; the
**unchanged** `TssVerifier` then has the final word. A rejected template yields
no fact and cold search remains available. This preserves the NQ3 refutation:
unchanged strict certificates still have unbounded support, transferred
`0/180` at every measured K, and have a measured reuse multiplier of exactly
`1.000x` [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L18-L24,
L109-L124, L138-L149].

This is not merely a defensive retreat. The shadow operation closest to this
materializer already made the unchanged proof body strict-verify on
`169/180`, `173/180`, `150/180`, and `140/180` target mutations at K=1,2,4,8
respectively [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L109-L124]. The consolidated
deep profile spends `392.065 s` of `495.940 s`, or `79.055%`, in its two
generation paths, while verifier plus harness overhead is bounded above by
`0.348 s`, or `0.070%` [IDEATION_FINAL.md@6ef67cfe:L72-L98]. Those facts earn a
shadow-first economics hunt. They do not establish a production win.

The single largest refutation risk is the **remote-dependency/selectivity
squeeze**: a cheap finite interface may match many bodies that strict replay
rejects because of a remote threat, legal-store change, or premature outcome;
adding enough global facts to prevent those false-positive probes may collapse
the key back toward the complete position or cost nearly as much as re-search.
The cheapest sharp attack is a second, disjoint remote defender count-five
whose completion is not answered by the reused body.

Citation notation in this document is `file@commit:lines`. All empirical
numbers are tied to a retained file and commit. Proposed v1 tags and bounds are
design constants, not measurements.

## 1. Non-negotiable boundary

### 1.1 What remains refuted

The current strict certificate is exactly root-bound. `RootBinding` contains
sorted complete occupancy and owners, current player, exact phase (including
the stored first coordinate in `SecondStone`), absolute placement count, and
terminal result [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L39-L78].
Strict verification rejects before arena replay unless that value equals
`RootBinding::from_state(target)` [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L179-L209].
Shared arena nodes are independently memoized under a full replay position and
commutation context [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L322-L460].

Therefore v1 makes none of these claims:

- it does not reinterpret current strict support as finite;
- it does not accept a proof from a support hash or hash collision;
- it does not weaken, wrap with an alternate implementation of, or bypass
  `TssVerifier`;
- it does not infer a hard result from interface match or successful
  materialization; and
- it does not promise that a nearby position will accept a body.

### 1.2 Two predicates whose names must not be conflated

For relative artifact `R=(I_hint,B)`, target state `P`, D6 action `g`, and
query `Q=(claimed_status, absolute_semantic_horizon)`, define:

```text
HintMatch(R, P, g)
    := the cheap finite declaration I_hint matches P under g.

Materialize(R, P, g, Q)
    : Option<TssCertificate>
    := the deterministic partial function in Section 3.

CRelAccept(R, P, g, Q)
    := HintMatch(R, P, g)
       AND Materialize(R, P, g, Q) = Some(C)
       AND C.semantic_horizon = Q.absolute_semantic_horizon
       AND derived_resolution(C) <= Q.absolute_semantic_horizon
       AND TssVerifier.verify(P, C, Q.claimed_status).
```

`match`, `candidate`, and `materialized` are pre-verification words.
`accepted`, `verified`, and `hard` are reserved for the last line. A production
API should expose `try_materialize` and `try_strict_discharge`, not
`verify_interface`.

The precise relative-certificate statement is:

> Let interface `I` be the pair `(I_hint, StrictReplayV1)`. **If interface `I`
> holds at position `P`**, where `holds` means `CRelAccept(I,B,P,g,Q)`, **then
> body `B`'s verdict is sound at `P` within query `Q`'s semantic horizon** to
> exactly the same extent as an
> ordinary `TssVerifier`-accepted strict certificate at `P`.

The cheap projection alone is deliberately not the antecedent. The argument
is one reduction: `CRelAccept` supplies an ordinary strict certificate accepted
against the complete target state, then the existing strict-verifier
soundness boundary applies. In particular v1 inherits, and does not claim to
strengthen, production instant dispatch's independently rederived kernel
theorem; the per-omitted-move lambda-1 oracle remains test-only
[packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L827-L963].

## 2. Exact v1 format

There is no production strict-certificate wire format today: the normative
types are in-memory Rust values. `C_rel` therefore needs its own canonical,
versioned codec and materializes an in-memory `TssCertificate`; it must not
claim wire compatibility with strict certificates
[packages/hexfield_eq/rust/src/lib.rs@6ef67cfe:L23-L31;
packages/hexfield_eq/rust/src/tss_solver.rs@6ef67cfe:L11967-L12096]. The current strict grammar
is `TssCertificate { root, claimant, root_node, nodes, semantic_horizon }` with
five arena-node variants [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L81-L153].

The schema below is normative for the design. Rust spelling is expository;
the field set and invariants are the contract.

```rust
type RelNodeId = u32;
type ClockOffset = u32;

enum RelDeadline {
    AfterRoot(ClockOffset),
    MaxU32,                         // materializes as exact numeric u32::MAX
}

struct CRelV1 {
    header: RelHeaderV1,
    interface: RelInterfaceV1,
    body: RelBodyV1,
}

struct RelHeaderV1 {
    magic: [u8; 4],                 // exactly b"HXCR"
    format_version: u16,            // exactly 1
    ruleset_contract: [u8; 32],     // exact engine/rules semantic ID
    strict_contract: [u8; 32],      // exact strict schema/verifier contract ID
}

struct RelInterfaceV1 {
    current_player: Player,
    phase: RelPhaseBinding,
    terminal: MustBeNonterminal,
    root_projection: Vec<CellRequirement>,
    zone_hints: Vec<ZoneHint>,
    wf_plan: Vec<WfWitness>,
}

enum RelPhaseBinding {
    Opening,
    FirstStone,
    SecondStone { first: HexCoord },
}

struct CellRequirement {
    coord: HexCoord,
    root_value: RootCellValue,
}

enum RootCellValue { Empty, Player0, Player1 }

struct ZoneHint {
    node: RelNodeId,
    source_required_cells: Vec<HexCoord>,
}

struct WfWitness {
    node: RelNodeId,
    subject: HexCoord,
    anchor: WfAnchor,
}

enum WfAnchor {
    RootOccupied(HexCoord),
    PriorClaimantPlacement(HexCoord),
}

struct RelBodyV1 {
    claimed: RelVerdict,            // Win or Loss; never Unknown
    claimant: Player,
    root_node: RelNodeId,
    nodes: Vec<RelNode>,
    derived_resolution_offset: ClockOffset,
    semantic_deadline: RelDeadline,
}

enum RelVerdict { Win, Loss }

enum RelNode {
    OrCompletion {
        mv: HexCoord,
        witness: WindowKey,
        completion_offset: ClockOffset,
    },
    Win {
        witness: WindowKey,
        count: u8,
        budget: u8,
        resolution_offset: ClockOffset,
    },
    Loss {
        witnesses: Vec<WindowKey>,
        resolution_offset: ClockOffset,
    },
    Choice {
        mv: HexCoord,
        child: RelNodeId,
    },
    Universal {
        edges: Vec<RelEdge>,
        implicit_dispatch: bool,
        zone: Option<RelZoneInfo>,
        commutations: Vec<RelCommutation>,
    },
}

struct RelEdge { mv: HexCoord, child: RelNodeId }

struct RelZoneInfo {
    d: u32,
    build_deadline: RelDeadline,
}

struct RelCommutation {
    first: HexCoord,
    omitted_second: HexCoord,
    first_child: RelNodeId,
    mirror_child: RelNodeId,
}
```

### 2.1 What the interface declares

`root_projection` is a sorted, duplicate-free vector of exact root occupancy
predicates. Its cell set is the union of:

1. every move, edge, and commutation coordinate in the body;
2. all six cells of every named `WindowKey`;
3. the stored `SecondStone.first`, if present;
4. every root coordinate named by the WF witness plan; and
5. every cell in every source-rederived `ZoneHint`.

The vector stores whether each such cell was empty, Player-0-owned, or
Player-1-owned at the admitted source root. Named windows are therefore
declared as their six exact cell predicates rather than as count-only
summaries. This prevents two different arrangements with the same count from
passing the cheap projection.

`zone_hints` bind each zone node to the required zone cells rederived at the
source replay state. They are candidate-routing and diagnostics data. They do
**not** assert that the target has no additional exposure or legal cells. The
target verifier reconstructs the proof core, local defender clock, legal set,
and uniform zone again [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L1018-L1150,
L1216-L1303].

`wf_plan` gives an explicit existential anchor for every WF query made by an
`OrCompletion`, `Choice`, `Win`, or `Loss` occurrence. A root anchor must be
occupied in `root_projection`; a prior-placement anchor must be established by
an earlier claimant placement on every path to that exact replay node. Both
must be within engine distance 8 of `subject`. This is an audit/prefilter
witness, not a replacement for the strict global WF scan.

The retained `38/54/68` and `22/42/53` body-footprint distributions did not
contain this new selected-root-anchor plan; they measured body coordinates,
named windows, and rederived zone cells. V1's actual projection can therefore
be larger. Those measurements are compactness evidence and a lower-bound input
to the interface sizing hunt, not a forecast of v1 bytes or cells
[HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L63-L107].

The exact phase is part of the declaration. `SecondStone` does not mean merely
the enum variant: its stored first coordinate is transformed and compared.
That precision is important at the NQ2 root, whose phase is
`SecondStone { first: (6,0) }` [PROOF_QUIET_LOCALITY.md@5e06c29c:L127-L146].

### 2.2 What the interface intentionally does not declare

V1 carries no trusted assertion of any of the following:

- absence of stones outside `root_projection`;
- the complete target legal-move store or its complement;
- the complete target live-threat/window family;
- `own_win_now`, hitting-set, dispatch-kernel, or terminal invariance;
- absence of new target zone exposure cells;
- equality of projected shared-DAG replay states; or
- commutation outcomes at the target.

Those are precisely the nonlocal facts that made the original support theorem
dangerous. Recording them completely would recreate a target replay or a full
position binding. Omitting them from `I_hint` is sound because `I_hint` has no
verdict authority; it may increase failed strict probes.

### 2.3 What the body binds

The body binds the claimed root-relative verdict, absolute player identity of
the claimant, full arena topology, concrete coordinate choices, witness
window identities, leaf counts and budgets, complete explicit Universal edge
lists, dispatch mode, zone `d`, commutation records, and all event/horizon
times as offsets from the root placement count. A semantic or zone-build
deadline is either `AfterRoot(delta)` or the explicit `MaxU32` sentinel;
`MaxU32` means the exact numeric cap `u32::MAX`, not mathematical infinity and
not a saturated finite event. `u32::MAX` is never converted into an
overflowing finite offset. The stored
derived-resolution offset must equal the maximum leaf/completion offset. The
body does not bind source root occupancy or source absolute placement count.

No target branch is invented or deleted in v1. D6 mapping and clock
materialization are the only body transformations. In particular:

- a target full-Universal node with one extra legal move rejects;
- a target zone with one extra required cell rejects unless that cell was
  already an explicit source edge;
- a changed dispatch kernel rejects if an element is unrepresented;
- a shared node reached under a different full target state/context rejects;
  and
- an early terminal outcome rejects before the advertised child.

This intentionally matches the NQ3 shadow operation--replace the root and
translate the leaf, completion, semantic-horizon, and zone-build clocks--whose
failure modes were already observed [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L109-L136].

### 2.4 Canonicalization, bounds, and cache identity

The codec is canonical little-endian: fixed enum tags, `u32` vector lengths,
lexicographically sorted/deduplicated projection and hint vectors, and no
ignored trailing bytes. The cache artifact ID is SHA-256 of the canonical
payload. The digest selects bytes; it never authorizes a result. Full bytes are
decoded, validated, materialized, and strict-verified.

The decoder rejects a contract-ID mismatch. An engine/rules or strict-contract
change therefore invalidates old artifacts rather than attempting a semantic
upgrade. The body adopts the strict verifier's current ceilings: 100,000
nodes; 1,000,000 edges, witnesses, and commutations; depth 256; and a 64 MiB
materialized replay memo. V1 additionally caps the canonical artifact at 64
MiB and the combined projection/hint/WF record count at 1,000,000. The first
set mirrors the normative verifier limits [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L17-L34];
the latter two are proposed fail-closed v1 admission limits.

## 3. Admission, rebinding, and strict discharge

### 3.1 Source admission

Only an already strict-accepted source certificate is admitted to the warm
library. Admission is deterministic:

1. Strict-verify the exact `(source state, certificate value, status)` tuple
   being converted. A bare `HardValue` or telemetry marker is not provenance
   for later-supplied bytes. Strict acceptance already rejects orphans, so
   retain that accepted arena unchanged; if any compaction is performed, the
   post-compaction certificate must be strict-verified again before conversion.
2. Replay every leaf occurrence and derive its logical event delta from the
   normative node rule. Require both the source replay-state addition and
   `n0+root_relative_delta` to succeed with `checked_add` and to equal the
   stored event field. Reject any certificate whose strict equality succeeded
   only because `saturating_add` reached `u32::MAX`.
3. Let `n0=source.placements_made`. Replace every admitted finite completion
   and resolution field `x` by checked `x-n0`; reject underflow. Encode a
   semantic or zone-build value of `u32::MAX` as `MaxU32`; otherwise encode
   checked `AfterRoot(x-n0)`. Store the independently derived maximum leaf
   offset and require it to agree with the relative arena.
4. Extract `root_projection`, source zone hints, and WF witnesses by replaying
   the accepted source arena. Reject an incomplete closure, ambiguous shared
   occurrence, invalid anchor, noncanonical vector, or limit overflow.
5. Store the canonical artifact under its untrusted content digest and a
   workload-scoped routing record.

Strict source admission is a quality/provenance rule rather than the final
soundness argument: an arbitrary body still cannot mint a target result unless
the target's unchanged verifier accepts it.

The consolidated solver already contains a useful implementation precedent:
its internal `CachedProof` is a rootless compact body, but today it is admitted
and looked up under complete `PositionKey` equality; imported bodies are
assembled under a fresh exact root and strict-verified. C_rel should reuse that
seam, not modify `tss_verify.rs`
[packages/hexfield_eq/rust/src/tss_solver.rs@6ef67cfe:L7470-L7565,
L7695-L7824, L8007-L8015].

### 3.2 Candidate routing

V1 is a scoped warm library, not an unbounded global scan. A caller supplies a
bounded cohort of bodies from a meaningful relationship: recent trainer
queries from one game/session, an atlas neighborhood, or explicitly named
sibling roots. The router first filters on the two contract IDs, claimed
status, current player, exact phase binding, and nonterminal target; it then
tries the 12 D6 actions and evaluates `root_projection`.

There is no translation or player swap in v1. D6 is the only coordinate
action already represented by the strict certificate machinery
[packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L1580-L1700]. Candidate
fanout remains an experimental parameter until the routing hunt measures the
precision/cost curve; Section 7 gives its kill condition.

### 3.3 Deterministic materialization

Given `HintMatch(R,P,g)`, query `Q`, and target placement count `n`,
`Materialize` does exactly this:

1. Require `Q.claimed_status == R.body.claimed`, reject `Unknown`, and require
   the body claimant to equal the claimant derived from
   `(P.current_player,Q.claimed_status)` by the strict convention.
2. Apply D6 action `g` to every coordinate, window, phase-first coordinate,
   projection cell, zone-hint cell, and WF coordinate. Reject invalid or
   colliding images.
3. Set `candidate.root = RootBinding::from_state(P)`--never a projected or
   reconstructed root.
4. Copy claimant, root node, topology, counts, budgets, edge modes, zone `d`,
   and commutation IDs from the transformed relative body.
5. For each event offset `delta`, set the candidate field to checked
   `n+delta`. For each deadline, map `AfterRoot(delta)` to checked `n+delta`
   and `MaxU32` to the exact value `u32::MAX`. Reject event overflow or a
   mismatch between the stored and recomputed maximum leaf offset; an event
   clock never uses `MaxU32` as a saturation escape.
6. Require the materialized `candidate.semantic_horizon` to equal
   `Q.absolute_semantic_horizon`. This deliberately conservative v1 rule keeps
   every zone's preserved build deadline in the same query contract; reuse
   across a different semantic horizon is left to a later proved format
   version. Materialize the derived resolution by checked
   `n+derived_resolution_offset` and require it not to exceed
   `Q.absolute_semantic_horizon`. Search node/TT/width flags need no binding;
   they do not change the self-contained strict verdict. The claimed status
   and semantic horizon do.

Materialization is a partial syntax/translation function, not a proof check.
It performs bounded decoding, canonicality/D6 conversion, and checked clock
conversion only. It may return a syntactically constructed candidate that the
single strict-discharge call rejects; it neither calls the private arena
validator nor invokes `TssVerifier` itself.

### 3.4 The only hard-result path

Wrap the candidate in the ordinary `DeepResult` and submit it to
`hard_value_from_verified(&TssVerifier, target, result)`. That mint takes the
concrete strict verifier, and a missing or rejected certificate yields no
`HardValue` [packages/hexfield_eq/rust/src/tss_core.rs@6ef67cfe:L187-L230].
This is the sole `TssVerifier` invocation for a probe.

The unchanged strict verifier is normative. The surrounding consolidated
producer may retry some zoned searches until derived T equals its requested
horizon, but that producer policy is not an additional C_rel acceptance rule:
v1 materializes the exact requested semantic horizon and accepts any candidate
the unchanged strict verifier accepts, including a zoned candidate whose
derived resolution is earlier [packages/hexfield_eq/rust/src/tree.rs@6ef67cfe:L625-L650].

The state-update rule is monotone:

```text
UNKNOWN + strict-accepted same-query C_rel verdict -> that hard verdict
UNKNOWN + no match/materialization failure/strict rejection -> cold solve
existing hard + same hard verdict                 -> unchanged
existing hard + opposite alleged verdict          -> fatal alarm, no update
```

Warm evidence may only upgrade `UNKNOWN`; it never changes an established hard
value. An opposite strict-accepted status for the same exact root contradicts
the assumed strict theorem: disable the warm lane for that query/session and
raise a fatal invariant alarm.

Never route or cache a bare `HardValue`. The verified result envelope carries
the exact `RootBinding`, query, status, materializer contract, and candidate
strict certificate used by the mint. Immediately before asynchronous delivery
or atomic installation, recompute the current state's full binding and require
equality; mismatch is a miss/`UNKNOWN`. Existing tree memo consumption uses
the same complete-binding pattern [packages/hexfield_eq/rust/src/tree.rs@6ef67cfe:L815-L831,
L1239-L1253]. A strict-accepted candidate may then be retained as an ordinary
exact-root strict certificate.

Pre-strict failures may be cached only under the exact tuple
`(full_payload_identity, RootBinding, g, claimed_status,
absolute_semantic_horizon, materializer_contract)`. Full payload equality
disambiguates a digest collision. The entry means "do not retry this exact
probe," never a game fact.

## 4. The NQ2 remote witness is a mandatory adversary

The frozen root has 538 legal SecondStone completions. The unique winning
completion is `r=(6,-6)`; it is distance 6 from the nearest **attacker** stone
and lies in no live **attacker** window. Every other one of the 537 legal moves
leaves `r` to the defender for an immediate win, while the `r` continuation
has a verifier-accepted general-branching certificate
[PROOF_QUIET_LOCALITY.md@5e06c29c:L127-L220]. All 12 D6 images preserve the
construction and verification [PROOF_QUIET_LOCALITY.md@5e06c29c:L222-L240].

Two precision points matter. The 537 are all alternatives, not 537 local
alternatives. Also, `r` is remote from attacker structure but sits in the
defender's urgent channel; "no live window" must not be read as "in no window
of either player."

This witness kills an interface consisting only of attacker-local windows or
distance bands. It does **not** by itself kill positive body rebinding: a body
that explicitly plays `r` may remain winning when the original reason for
playing `r` disappears. The soundness-shaped mutation is to add a *second
disjoint* defender count-five, with completion `s` outside the declared body,
such that the reused body's first move does not occupy `s`. A support-only
checker could miss the defender's immediate win. V1 instead replays the full
target: global `analyze`, Loss, dispatch, legal-store, zone, and terminal
checks either reject the candidate or the candidate earns ordinary strict
acceptance. Locality is therefore unnecessary for soundness and remains
crucial only to hit rate and cost.

The retained NQ3 far-defender construction is already one member of this
attack class. Its root-rebound/clock-shifted body was rejected, with no
soundness finding [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L126-L136].

## 5. Obligation scoreboard

Each obligation has exactly one disposition. O1-O8 are the recorded NQ3 list
[HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L161-L172]. O9-O13 are new obligations
created by this format and integration boundary.

| ID | Obligation | Exclusive disposition |
|---|---|---|
| O1 | Full binding / complement absence | **DISSOLVED** |
| O2 | Absolute clocks | **PROOF-SKETCHED** |
| O3 | Legal-store growth | **DISSOLVED** |
| O4 | Remote threats and goals | **DISSOLVED** |
| O5 | No new six / premature terminal | **DISSOLVED** |
| O6 | WF anchors | **PROOF-SKETCHED** |
| O7 | Shared DAG identity and commutation | **DISSOLVED** |
| O8 | D6 image closure | **PROOF-SKETCHED** |
| O9 | Strict-discharge reduction, exact-snapshot delivery, and monotone integration (new) | **PROOF-SKETCHED** |
| O10 | Codec/version/resource/cache boundary (new) | **PROOF-SKETCHED** |
| O11 | Cold-fallback and resource-state isolation (new) | **PROOF-SKETCHED** |
| O12 | End-to-end amortization (new) | **REFUTATION-RISK** |
| O13 | Interface selectivity and candidate routing (new) | **REFUTATION-RISK** |

Score: **6 PROOF-SKETCHED / 2 REFUTATION-RISK / 5 DISSOLVED.**

### O1. Full binding / complement absence -- DISSOLVED

**Design change.** The relative artifact does not replace strict full binding.
Materialization installs `RootBinding::from_state(P)` from the complete target,
and strict root equality remains the first semantic check. The finite
projection is only a candidate filter.

**Why the old obligation is moot.** No theorem must show that outside cells are
absent or irrelevant: all target cells are in the materialized root binding,
and all replay keys remain complete.

**Trade-off.** Every hit becomes a target-specific strict artifact; an
interface match may still reject, and v1 cannot answer without replaying the
strict body. It is not a standalone relative verifier.

### O2. Absolute clocks -- PROOF-SKETCHED

**Full-strength lemma (relative clock translation).** Let a relative body be
admitted from a strict-accepted source with root placement count `n0`, with the
additional admission fact that every logical leaf/completion addition is
nonsaturating and equals its stored event time. For any valid nonterminal
target `P` with placement count `n` satisfying `HintMatch(R,P,g)`, and for any
body replay path whose moves remain legal and nonterminal
until its advertised leaf, require every corresponding target event addition
to succeed with `checked_add`; replace source event time `x` by offset `x-n0`
and materialize it as checked `n+(x-n0)`. Then:

1. every replay state at path length `k` has placement count `n+k` instead of
   `n0+k`;
2. every `completion_ply` and `resolution_ply` equality is preserved;
3. maximum-derived leaf horizon and every finite zone-build/semantic deadline
   commute with the same translation, while the `MaxU32` deadline remains
   the exact numeric value `u32::MAX`;
4. every copied zone `d` remains equal to the verifier's proof-tree recurrence:
   matching player/phase schedules preserve the `Loss`-leaf placement budget,
   `Choice` passes through its child's value, and `Universal` preserves one
   plus the maximum child value; protected and target-zone cell sets need not
   be invariant and are rederived by strict replay;
5. the translated semantic deadline equals the target query's absolute
   semantic horizon, every translated zone-build deadline retains its strict
   equality to that semantic deadline, and the materialized derived resolution
   is within it; and
6. underflow or overflow makes conversion/materialization undefined and hence
   cannot accept.

**Argument outline.** Induct on replay length. `apply_with_delta` increments
the placement clock once, while the phase/player transition is determined by
the matching starting phase and the same move count. Addition by a fixed
translation commutes with successor, finite max, finite min, and the strict
`<=` horizon comparisons. Source strict acceptance establishes zone-build to
semantic-deadline equality; applying the same translation preserves it. Treat
`MaxU32` as a separate exact numeric case rather than ordinary addition. The
extra nonsaturation premise is necessary because strict leaf equations use
`saturating_add`, which otherwise loses the logical delta near `u32::MAX`
[packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L601-L739]. Prove the
zone recurrence structurally over `OrCompletion`/`Win`, `Loss`, `Choice`, and
`Universal` [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L1018-L1088].
The `SecondStone.first` coordinate is bound exactly even though clock
arithmetic needs only the phase schedule; this prevents a weaker phase matcher
from acquiring unintended semantics.

**Proof home.** The field inventory and engine assumptions belong in the paper
spec, `C_REL_FORMAT_AND_SOUNDNESS.md`. The natural-number translation,
phase-schedule induction, max/min lemmas, and checked-arithmetic fail-closed
corollary belong in a Lean spine module `CRel/ClockTranslation.lean`. Rust
mutation tests remain executable correspondence evidence, not the proof.

### O3. Legal-store growth -- DISSOLVED

**Design change.** V1 preserves the source Universal edge family only as a
candidate. The strict verifier rederives the target legal set. Full mode
requires exact equality; dispatch mode rederives its kernel; zone mode
rederives target coverage [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L827-L963,
L1018-L1150, L1216-L1303].

**Why the old obligation is moot.** C_rel soundness does not assume legal-store
invariance. One new remote legal move can simply make strict discharge fail.

**Trade-off.** Target global legal generation remains in verification cost,
and benign legal-frontier growth may reduce transfer. V1 does not splice in a
new child or warm-start a failed body below the full-cert level.

### O4. Remote threats and goals -- DISSOLVED

**Design change.** No threat digest is authoritative. Target strict replay
reruns `analyze`, leaf witness/hitting checks, dispatch derivation, and zone
exposure on the complete target. The NQ2 witness and disjoint-count-five mutant
are mandatory regressions, not premises of a locality theorem.

**Why the old obligation is moot.** An undeclared remote defensive dependency
cannot authorize a result. It either changes no strict obligation, in which
case acceptance is an ordinary strict proof, or changes one and causes
rejection.

**Trade-off.** There is no bounded-radius soundness claim. Global threat/window
work remains in every relevant target replay, and this is the sharpest source
of interface false positives. If a future design calls `HintMatch` sufficient,
the disjoint remote count-five immediately moves this item back from dissolved
to a fatal proof obligation.

### O5. No new six / premature terminal -- DISSOLVED

**Design change.** The interface requires a nonterminal root, but terminal
soundness comes from strict target replay: each represented move goes through
`HexoState`, internal nodes reject terminal states, and advertised completion
leaves check the actual winner and six-cell witness
[packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L463-L669].

**Why the old obligation is moot.** V1 never asserts that added stones cannot
create an earlier six. An earlier outcome invalidates the candidate at the
exact replay edge where it occurs.

**Trade-off.** A projection match cannot avoid replay. Target mutations that
are tactically harmless at the root can still reject deep in the body.

### O6. WF anchors -- PROOF-SKETCHED

**Full-strength lemma (WF-plan totality and sufficiency).** Successful source
admission constructs a `WfWitness` for every strict WF query at every reachable
source occurrence, including every data-dependent `Win`/`Loss` empty. It
rejects a shared node if its accepted occurrences do not induce one identical
WF-query set and valid anchor plan. If `HintMatch(R,P,g)` and target replay
reaches the corresponding node without an earlier strict failure, the exact
six-cell projection of every named window plus the identical prior body moves
preserves that WF-query set under `g`. For every mapped query
`(node,subject)`, its witness names an anchor within distance 8 and either:

- `RootOccupied(a)`, where target root projection proves `a` occupied; or
- `PriorClaimantPlacement(a)`, where every path to that exact replay node has
  already legally placed claimant stone `a` and has not undone it.

Then the strict `attacker_placement_wf` existential succeeds at every query.
This covers the explicit attacker moves and every empty witness cell tested by
`Win` and `Loss`, over all reachable arena occurrences. Extra target anchors
are permitted and need not be declared.

**Argument outline.** Enumerate WF call sites by node variant during source
replay. Exact projection fixes each named window's root contents; induction on
the identical body prefix therefore fixes its target empty-cell query set.
Compare every shared occurrence during extraction and reject disagreement.
Root occupancy persists because game replay only adds stones. For a
prior-placement witness, induct over the path and use exact shared-node
occurrence identity to show the claimant-owned stone remains present. D6
preserves hex distance. Either case provides a member of the strict
existential scan. The proof deliberately matches the normative scan,
which checks current claimant stones and the recorded target-root stone vector
[packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L590-L615].

**Proof home.** The annotation extraction and node-variant coverage table
belong in `C_REL_FORMAT_AND_SOUNDNESS.md`; persistence, path induction, and D6
distance preservation belong in `CRel/WfAnchors.lean`. Differential tests must
include the NQ2 distance-6 move so an implementation cannot silently tighten
the engine's distance-8 rule to adjacency.

### O7. Shared DAG identity and commutation -- DISSOLVED

**Design change.** The candidate retains the source DAG and commutation records,
but the unchanged strict verifier constructs target `ReplayKey` values from
the complete replay state plus allowed-commuted context and rechecks both move
orders [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L322-L483,
L741-L824].

**Why the old obligation is moot.** No projected key discharges a target
occurrence. If two occurrences no longer denote the same complete state or
context, strict memo equality rejects.

**Trade-off.** V1 loses the entire candidate rather than unfolding or repairing
one target-specific shared occurrence. A later unfolding materializer would
be a separate format version and obligation set.

### O8. D6 image closure -- PROOF-SKETCHED

**Full-strength lemma (interface, routed materialization, and strict
equivariance).** Define routed action `g` as mapping artifact coordinates from
their stored source frame into the target frame. For every one of the 12 D6
actions `g`, every decodable relative body/interface `R`, query `Q`, and target
`P` for which the relevant sides are defined:

```text
HintMatch(R, P, g) <-> HintMatch(g.R, P, identity)
HintMatch(R, g.P, g) <-> HintMatch(R, P, identity)

Materialize(R, P, g, Q)
    = Materialize(g.R, P, identity, Q)

Materialize(R, g.P, g, Q)
    = Option.and_then(Materialize(R, P, identity, Q),
                      |C| d6_remap_certificate(C, g))

TssVerifier.verify(P, C, s)
    <-> TssVerifier.verify(g.P, g.C, s)

CRelAccept(R, P, identity, Q)
    <-> CRelAccept(R, g.P, g, Q)
```

The action covers complete root occupancy/owners, `SecondStone.first`, every
body and interface coordinate, every window, WF/zone hint, edge, and
commutation coordinate. Player identities, counts, IDs, relative offsets, and
statuses are invariant. The router enumerates canonical pairs
`(artifact_id,g)` with `g=0..11` and deduplicates identical materialized
candidates before strict discharge.

**Argument outline.** Prove that the coordinate action is bijective and maps
each length-six axis window to its normalized image; it preserves occupancy
ownership, axial distance, legal application, phase transitions, and terminal
outcomes. Prove the two `HintMatch` laws field-by-field, the routed materializer
laws including `None`/overflow cases, and strict-verifier equivariance by
induction over replay. The existing strict remapper covers every current coordinate-bearing
certificate field [packages/hexfield_eq/rust/src/tss_verify.rs@6ef67cfe:L1502-L1700],
and the NQ2 report supplies all-image executable evidence
[PROOF_QUIET_LOCALITY.md@5e06c29c:L222-L240].

**Proof home.** Group action, coordinate/window normalization, and materializer
commutation belong in `CRel/D6.lean`; the exhaustive schema map and engine
equivariance assumptions belong in `C_REL_FORMAT_AND_SOUNDNESS.md`. All-12
round-trip and hostile-mutant regressions are required but are not substitutes
for the lemma.

### O9. Strict-discharge reduction, exact-snapshot delivery, and monotone integration -- PROOF-SKETCHED

**Full-strength lemma (sealed reduction and exact-snapshot delivery).** Assume the existing strict
soundness statement `TssVerifier.verify(P,C,s) -> GameVerdict(P,s)`. For all
bytes, cache states, target states, D6 actions, claimed statuses, malformed
artifacts, query horizons, arithmetic failures, interface mismatches, and
strict rejections,
the v1 wrapper can return a new hard result envelope only if it has constructed `C` and
the concrete `TssVerifier` accepted `(P,C,s)` and `C`'s derived resolution is
within the requested query horizon. The envelope binds the exact
`RootBinding`, full query, status, materializer contract, and certificate; its
binding is rechecked against the current target immediately before delivery or
installation. Its state transition is:

- `Unknown -> s` only on that acceptance;
- `Unknown -> Unknown` on every other path;
- `s -> s` for a matching pre-existing hard value; and
- no transition plus a fatal warm-lane alarm for an opposite pre-existing hard
  value.

Therefore every v1-produced hard value reduces to the existing strict theorem,
and warm reuse cannot overturn a hard result.

**Argument outline.** Case-split over decode, hint, materialization, and strict
result. Only one branch calls the sealed hard-value mint; all others return
`None` or cold fallback. Pair the returned value immediately with the exact
snapshot and prove that the second full-binding comparison prevents stale
asynchronous delivery. Then prove the four update cases. Concurrency must
perform the same transition atomically; an opposite result is fatal, not
last-writer-wins.

**Proof home.** The semantic theorem and result-state transition belong in
`CRel/StrictDischarge.lean`. The paper spec pins the exact Rust call graph to
`hard_value_from_verified(&TssVerifier,...)`, whose concrete-verifier sealing is
already explicit [packages/hexfield_eq/rust/src/tss_core.rs@6ef67cfe:L187-L230].
An implementation review must reject any new `CertVerify` implementation or
hard-value constructor for C_rel.

### O10. Codec/version/resource/cache boundary -- PROOF-SKETCHED

**Full-strength lemma (bounded canonical decode and non-authority).** For every
byte string and cache state, v1 decoding either:

1. rejects before any arithmetic or allocation exceeds the declared byte,
   per-vector, or checked aggregate caps; or
2. returns exactly one canonical artifact whose complete byte consumption,
   every nested vector length, and aggregate nodes/edges/witnesses/
   commutations/projection/zone/WF counts are within those caps, with no
   trailing bytes.

All length multiplication/addition is checked before allocation, and contract
IDs are compared before body allocation. A stale contract, malformed length,
truncation, noncanonical order, duplicate, trailing byte, overflow, oversized
record, digest collision, or cache replacement can yield only rejection,
eviction, or a fully decoded payload that still crosses O9's exact
materialization and strict-discharge boundary. Cache lookup never returns a
bare hard value.

**Argument outline.** Model the decoder as a remaining-byte cursor with a
monotonically decreasing input measure and an allocation ledger. Prove each
primitive read preserves cursor/ledger bounds; lift the invariant through
nested vectors using checked aggregate counters. Prove encode/decode
canonical round-trip and rejection of alternate encodings. Finally case-split
digest/full-payload equality, version mismatch, positive result envelope, and
negative-probe entry, reducing every semantic result to O9.

**Proof home.** The byte grammar, allocation order, aggregate formulas,
contract invalidation rule, and proposed 64 MiB artifact ceiling belong in
`C_REL_FORMAT_AND_SOUNDNESS.md`. A pure decoder/ledger model and
round-trip/bound/non-authority lemmas belong in `CRel/Codec.lean`. Rust fuzz,
truncation, maximum-length, collision-injection, and allocation-counter tests
are mandatory correspondence evidence. Engine/verifier contract changes
invalidate the library, and large otherwise-valid bodies may be refused; that
is the explicit availability trade-off.

### O11. Cold-fallback and resource-state isolation -- PROOF-SKETCHED

**Full-strength lemma (failed-probe transparency).** Fix target `P`, full query
`Q`, solver configuration `K`, residual solver/cache budget `B`, exact-cache
snapshot `X`, and deterministic cold-solver seed. If every C_rel probe misses,
fails decoding/materialization, or fails strict discharge, then invoking the
unchanged cold solver afterward is observationally equivalent to invoking it
directly from `(P,Q,K,B,X)`, except for explicitly charged wall/telemetry and
the separate C_rel lookup/negative-cache state. In particular the probe does
not mutate `P`, consume the cold node cap or semantic horizon, perturb the
solver TT/frontier/exact-fragment store or RNG, retain a verifier replay clone,
or turn a baseline hard result into `UNKNOWN` at the same residual budget.

This lemma is relative to the configured residual budget `B`. A fair fixed-RAM
economics arm may deliberately reserve bytes for C_rel and thus give the cold
solver a smaller `B` than a no-C_rel baseline; that allocation difference is
not hidden by the lemma and must be charged in O12.

**Argument outline.** Make artifact parsing, projection, materialization, and
strict replay operate on immutable target references and owned temporary
values. Put C_rel routing and negative entries in a disjoint store. On every
failure edge, drop temporaries and call the existing solver entry point with
the untouched tuple `(P,Q,K,B,X,seed)`. Prove the warm-wrapper state-machine
projection onto solver-visible state is identity, then couple direct and
post-miss cold executions step for step.

**Proof home.** The state partition and API/cap call graph belong in
`C_REL_FORMAT_AND_SOUNDNESS.md`; a small state-monad noninterference/coupling
lemma belongs in `CRel/FallbackIsolation.lean`. Rust forced-miss differential
tests must snapshot state, caps, exact stores, TT/frontier initialization, RNG,
status, certificate strict acceptance, and accounted bytes before/after.

### O12. End-to-end amortization -- REFUTATION-RISK

**Counterexample shape that kills the project.** Source bodies are expensive to
obtain or rare; targets are cheap or mostly reject; global interface/rebind/
strict replay is always paid. Even a sound cache then increases wall time. The
current shallow trainer profile is a concrete danger: one retained h8/cap-500
configuration totals only `76.513 ms` across 300 solves
[HUNT_REPORT_LEAF_SURFACE.md@5172d42d:L78-L101]. Source scarcity is also real:
only `34/200` human roots produced a WIN certificate at the retained
30k-node, 64 MiB, root-plus-50-horizon screen; that is not a general human-root
availability rate [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L89-L107].

**Cheapest surfacing experiment.** Freeze the complete retained acquisition
manifest before running consolidated code: every retained official/human
source position, its original cap ladder/horizon, and the NQ3 seed/trial rule.
Do not select only consolidated successes. Report acquisition failures, Loss
fixtures, and non-`FirstStone` skips separately. Let `m` be the resulting
eligible strict-admitted `FirstStone` sources; construct all `8m` K=1/K=2
targets (four trials per K) and set incremental `G=0` because every source
solve is already demanded. The old engine happened to give `m=45` and 360
such targets, but consolidated HEAD is allowed to change `m`
[HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L63-L124].

In frozen `(source-kind, source-id, K, trial)` order, measure source admission
`E`, lookup `L`, every hint check `I`, materialization `M`, strict verification
`V`, matched baseline/fallback solves `S0/SR`, and accepted indicator `A`. Run baseline exact
fragments and C_rel with identical semantic caps and the fixed-total-memory
accounting in Section 7; return no warm hard result in the shadow lane. Then
repeat only the empty-library overhead path on the retained 300-query shallow
trainer cohort as a negative control. Apply Section 7's paired, source-clustered
decision rule. Kill production work if the lower confidence bound on net
end-to-end saving is at most the program's 5% bar, or if the workload's fixed
RAM budget is exceeded. The shallow cohort is retained at
HUNT_REPORT_LEAF_SURFACE.md@5172d42d:L8-L28,L78-L101. No broader corpus run is
needed until this test satisfies the same algebra.

### O13. Interface selectivity and candidate routing -- REFUTATION-RISK

**Counterexample shape that kills the project.** A weak projection has a large
candidate bucket: many roots agree on the measured tens-of-cells body
footprints but differ in a
remote opponent threat, legal frontier, or zone exposure, so most candidates
fail only after strict replay. Strengthening the interface with complete
threat/legal complements either makes interface construction search-like or
recovers full-position equivalence and the already measured `1.000x` reuse.
This is the remote-dependency/selectivity squeeze
[HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L63-L107, L138-L149].

**Cheapest surfacing experiment.** Use exactly O12's predeclared consolidated
source manifest and all `8m` deterministic K=1/K=2 targets. Attempt to preseed
all `m` bodies before target replay using Section 7's deterministic
reservation/admission rule; report every capacity refusal, and do not route a
target directly to its known parent body. For every admitted
body-target-D6 triple, compute the following shadow matrix:

```text
unconditional shadow strict-acceptability
HintMatch
HintMatch AND strict acceptance
HintMatch AND strict rejection
not HintMatch AND bypassed strict acceptance
candidate bucket size and rank of first accepted body
projection bytes; L/I/M/V time; complete-root-key equality
```

Order candidates by `(status, phase, projection_cell_count, artifact_id, g)`
and sweep maximum fanout `{1,2,4,8,16,32}`. Replay targets in
`(source-kind,source-id,K,trial)` order. Apply the paired/source-clustered timing
rule in Section 7. As the canonical hostile row, also use the retained exact
NQ3 far-defender window and 12-placement addition sequence; it already makes a
shifted/rebound body reject [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L126-L136].
The NQ2 canonical root and its 12 D6 images remain state/interface regressions,
but no unfrozen "second NQ2" coordinates are counted as evidence.

Kill the finite-interface project if (a) no new acceptable cross-root pair
exists beyond exact root equality, (b) matching loses more acceptable bodies
than its saved strict probes are worth under the O12 equation, or (c) every
bounded-fanout cell fails O12's measured net-saving inequality.
A
`HintMatch && strict-reject` is not a soundness failure in this design; calling
it accepted would be. A later real trainer or atlas cohort counts toward the
project verdict only after a manifest freezes its root IDs, neighborhood
construction, caps/horizons, source-library build, query order, and fallback
policy.

## 6. Economics

### 6.1 Measured inputs and non-inputs

Retained evidence supplies:

- Strict unchanged transfer: `0/180` at each K in `{1,2,4,8}`; shifted/rebound
  shadow transfer: `93.89%`, `96.11%`, `83.33%`, and `77.78%`
  [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L109-L124]. These rates cover 45
  transferable `FirstStone` roots; the one `SecondStone` root was skipped, so
  they provide no measured support for v1's exact `SecondStone.first` handling.
- On 12 solved official rows, body-footprint cells have median/p90/max
  `38/54/68`; on 34 human-corpus certificates they are `22/42/53`, versus root
  populations `31/81/149` [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L63-L87,
  L89-L107]. These are cell counts, **not serialized body bytes**; no retained
  report measures C_rel wire size.
- The consolidated official deep run has 34 solves, `495.592 s` summed solve
  wall and `495.940 s` test wall. Its attacker and defender generation paths
  consume `216.228 s` and `175.837 s`, together `79.055%` of test wall; strict
  verifier plus harness is at most `0.070%`
  [IDEATION_FINAL.md@6ef67cfe:L72-L112].
- The same official ladder throws away `151.988 s`, or `30.67%` of solve wall,
  in 15 non-final lower rungs because each solve starts fresh
  [IDEATION_FINAL.md@6ef67cfe:L114-L149]. This is evidence for cold-start cost,
  but an UNKNOWN lower rung has no certificate; exact frontier continuation,
  not C_rel, is the direct same-root remedy.
- The existing exact shared-fragment baseline is nontrivial. Across its warm
  139-root campaign it saved `20.457%` of expansions and `16.322-16.476%` wall,
  despite only `199/424,391` lookup hits and 39 imports; cold overhead was
  `0.809-0.989%` [HUNT_REPORT_SHARED_FRAGMENTS.md@b45b9bf0:L242-L259]. C_rel must
  beat this baseline incrementally where both apply.

### 6.2 Cost model

For target query `j`, let:

```text
S0_j  strongest no-C_rel baseline wall under the fixed total budget
SR_j  cold-fallback wall with C_rel's residual solver budget
L_j   cohort/index lookup cost
n_j   candidate bodies actually probed
I_ji  finite interface evaluation cost for candidate i
M_ji  deterministic materialization cost
V_ji  unchanged strict-verification cost
A_j   1 iff some candidate is strict-accepted, else 0
E     one-time extraction/serialization/index build cost
G     seed solve cost, but only if the source was not already a demanded query
```

Then the baseline and complete warm path with mandatory fallback are:

```text
T_base = sum_j S0_j

T_Crel = G + E
       + sum_j [ L_j
                 + sum_attempted_i (I_ji + M_ji + V_ji)
                 + (1-A_j) SR_j ]

net saving = T_base - T_Crel
           = sum_j [A_j*S0_j + (1-A_j)*(S0_j-SR_j)]
             - G - E
             - sum_j [L_j + sum_attempted_i(I_ji+M_ji+V_ji)]
```

If the two arms give the solver the same residual resources so
`S0_j=SR_j=S`, homogeneous targets with strict-accept probability `p`,
per-query warm overhead `H`, and `N` targets reduce to:

```text
break-even: p*S > H + (G+E)/N
5% gate:    p*S - H - (G+E)/N >= 0.05*S
```

Set `G=0` when the source query was needed anyway. Count every failed probe in
`H`; a miss is more expensive than baseline because cold search follows it.
Use wall time as the decision metric, with expansion/generation counters only
for diagnosis. `S0_j` and `SR_j` must be measured under a fair
fixed-total-cache envelope:
bytes reserved for C_rel are subtracted from the solver/fragment allocation in
that arm, rather than added free of charge. Report accounted TT, exact-fragment,
C_rel artifact/index, and verifier-temporary bytes separately, plus process
peak RSS. Any loss caused by the smaller residual solver budget is already in
the measured `SR_j`; memory-budget excess is a separate kill.

As a cross-report planning calculation, multiplying the old strict-accepted
shadow rates by the consolidated generation share puts generation time equal
to roughly 61-76% of a similar workload's **total** wall in reach--equivalently,
the accepted fraction is roughly 78-96% of the generation component--before
any interface, materialization, failed-probe, or seed costs. This is an
extrapolation from two different retained campaigns, not a measured
consolidated speedup
[HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L109-L124;
IDEATION_FINAL.md@6ef67cfe:L72-L98].

### 6.3 Profiles

**Plausible payers:**

- already-needed deep trainer solves followed by nearby positions in the same
  session, where the source-body marginal `G` is zero;
- a preseeded opening/forcing atlas queried in neighborhoods with repeated
  player/phase and coordinate frame;
- deep sibling positions that differ outside compact proof bodies; and
- any cohort where C_rel creates strict-accepted cross-position hits beyond
  the complete-key shared-fragment store.

**Non-payers or wrong tool:**

- one-shot arbitrary/UNKNOWN-heavy roots, where source coverage and
  amortization are poor;
- the retained sub-millisecond h8 trainer profile, unless a near-zero-cost
  router proves otherwise;
- same-root node-cap ladders before proof closure--resume the exact unfinished
  frontier instead;
- distant mutations whose global legal/threat facts change repeatedly;
- any interface whose equivalence classes equal complete `RootBinding`; and
- a workload that must generate special seed proofs solely to populate C_rel
  but cannot amortize `G`.

For an asynchronous trainer trial, aggregate solve wall is necessary but not
sufficient operational evidence. The live A/B must also report completed-query
throughput, queue/park tail, and p95 request latency; a cache that improves
summed solver time but worsens trainer backpressure does not pay.

## 7. Hunt ladder for later rounds

Round 1 ran none of these. Every future Cargo stage must obey the lane/RAM
rules in its own authorization; the classes below are planning envelopes, not
permission to run.

### 7.1 Frozen measurement protocol

Stages 1-5 use the complete retained NQ3 acquisition manifest and recipes, not
only the positions that happen to solve on consolidated HEAD. Official sources
retain their recorded 10k/100k ladder and unlimited semantic horizon; the 200
deterministic human roots retain the recorded 30k cap, 64 MiB TT, and
root-plus-50 horizon. Acquisition outcome and conditional rebind outcome are
reported separately [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L63-L124]. Target and
candidate order is lexicographic as specified in O12/O13.

Every wall-decision cell uses three paired repetitions in order `AB`, `BA`,
`AB`; each letter is a separate Cargo/process invocation, so one statistical
cell has six serialized invocations. Target order and warmed-code conditions
are identical inside each pair. For every source cluster, average its three
paired wall deltas first. The estimator is the sum of those per-cluster means.
Its 95% interval is a 10,000-resample percentile cluster bootstrap with fixed seed
`0xC0DE_C0DE_5EED_0001`: cluster by source root for the K cohort and by the 50
six-state game batches for the trainer cohort. The Stage-6 official cold
control clusters every rung/attempt under its official root ID. Individual
mutations or rungs from one root are never treated as independent. Raw
aggregate, every paired repeat, and the interval are all reported; no per-root
parameter choice is allowed.

All A/Bs fix a total accounted cache budget. C_rel artifact/index bytes are
subtracted from the solver/fragment allocation in the C_rel arm. Report each
accounted component and process peak RSS; the latter is not inferred from TT
telemetry and is judged separately by relative regression, not against the
cache-byte ceiling. Peak RSS has a deliberately conservative process-level
rule instead of the source-cluster bootstrap: compute the relative RSS
regression inside each of the three paired invocations and require the maximum
observed paired regression to meet the stage's bound. For a seeded reservation, sort bodies by
`(source-kind,source-id,artifact_id)`, charge canonical artifact plus index
bytes, admit only when the whole next record fits, refuse otherwise, and do no
eviction during that cell. A runtime limit below is per Cargo/process
invocation; a statistical cell consists of the six invocations above, and a
multi-cell campaign is split accordingly.

### 7.2 Fixed hostile manifest

Stage 3 has six named cases rather than an open-ended mutation search:

1. `H1_NQ2_REMOTE`: the exact 36-placement NQ2 replay, `SecondStone.first`,
   unique `r`, and all 12 D6 images from the retained proof; regenerate the
   strict certificate at horizon 66 and a 10k node cap
   [PROOF_QUIET_LOCALITY.md@5e06c29c:L127-L240].
2. `H2_NQ3_FAR5`: the retained `0hz3hty` body and exact 12-cell balanced
   addition sequence forming the remote defender window; unchanged and
   rebound candidates must reject [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L126-L136].
3. `H3_CLOCK_SATURATION`: unit-admit a leaf whose replay base is
   `u32::MAX-1` and whose logical event delta is two; the saturated source
   encoding must reject rather than become relative offset one.
4. `H4_NEGATIVE_HORIZON_KEY`: probe one artifact/root first at absolute
   horizon 109 (forced mismatch), then 110 (match); the first negative entry
   must not suppress the second.
5. `H5_STALE_DELIVERY`: materialize and strict-discharge the retained
   `0hz3hty` source body at its exact source `P`, apply the lexicographically
   first legal nonterminal outside-projection move to obtain `P'`, then attempt
   delivery; complete-binding mismatch must return `UNKNOWN` (absence of such
   a move is a fixture-construction failure, not a skipped pass).
6. `H6_FORCED_MISS_ISOLATION`: force a contract-ID mismatch immediately
   before the retained `xsnfyll` 10k-closing query, then compare the post-miss
   cold execution with direct cold execution under the exact O11 snapshot
   tuple [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L63-L87].

The exact retained far-five fixture, not an unfrozen proposed "second NQ2"
coordinate set, is the canonical disjoint-remote-threat test in round 2.

| Stage | Shadow-only experiment | Kill criterion | RAM/runtime class |
|---|---|---|---|
| 1. Consolidated shadow reproduction | Add a cfg(test)-only relative-clock converter/materializer against the consolidated engine. Run the frozen acquisition manifest and four deterministic balanced outside-footprint mutations per K, plus strict-accepted hand Loss fixtures. Preserve unchanged-strict transfer as the negative control; submit every rebound candidate to the unchanged strict verifier. Freeze every acquisition/rejection reason and `M/V` timers. Compare conditional rates with the retained `77.78-96.11%` shadow range [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L109-L124]. | Halt on any hard result emitted without strict acceptance. Kill the full-certificate path if it produces no cross-root strict acceptance beyond exact root equality; otherwise defer economics until matched `S0/SR` is measured in Stage 4. Never replace consolidated acquisition failures to restore the old denominator. | **M:** 64 MiB TT, serial, prior campaign about `61.8 s` after compilation; under 10 min, launch above 9 GiB free [HUNT_REPORT_CERT_SUPPORT.md@3cd224fe:L176-L199]. |
| 2. Interface recorder and confusion matrix | Extract the exact v1 projection/zone/WF declaration. Evaluate it, but bypass it in a parallel shadow arm so unconditional strict-acceptable bodies are known. Measure true/false probe classes, serialized bytes, candidate count, and lookup/extract/match/materialize/verify time separately. Return no production hard values. | Kill this interface if its saved strict probes are worth less than the acceptable bodies it filters out under the O12 equation, or if projected classes create no cross-root matches beyond exact keys. `HintMatch && strict-reject` is an economic datum, not a safety failure. | **S/M:** fixture replay through the Stage-1 64 MiB cohort; seconds to low minutes, under 10 min. |
| 3. Hostile remote/global suite | Run only H1-H6 from the fixed hostile manifest. H1/H2 exercise remote defensive value and D6; H3/H4 attack clock/cache completeness; H5 attacks exact-snapshot delivery; H6 attacks cold-fallback isolation. No corpus profile. | Fatal contract kill if any case returns hard without the one strict mint, H2 fails to strict-reject, H3/H4 accepts/suppresses incorrectly, H5 installs stale evidence, or H6 differs in solver-visible state/status/certificate from direct cold execution. Ordinary H1 source acceptance and hostile target rejection are expected. | **S:** hand fixtures plus the named 10k NQ2 regeneration; 64 MiB maximum test TT, seconds to low minutes. |
| 4. Scoped library/routing economics | Use O13's exact all-body/all-K-target shadow matrix and the deterministic admission/refusal rule above. Compare current exact fragments with C_rel plus exact fragments plus cold fallback at total accounted cache budget 512 MiB; sweep C_rel reservations `{1,8,32,64}` MiB and fanout `{1,2,4,8,16,32}`, subtracting each reservation from residual solver bytes. Include `G/E/L/I/M/V/S0/SR`, admitted/refused bodies, artifact/index bytes, accounted peak, process RSS, and first-accepted rank. Return no production warm hard values. | Kill if the source-clustered 95% lower bound of net gain is at most 5% over the strongest exact baseline in every budget/fanout cell, if no bounded fanout pays, if accounted cache exceeds 512 MiB, or if the maximum observed paired process-RSS regression exceeds 5%. RSS is not compared directly with 512 MiB. | **M:** each serialized invocation under 10 min; each statistical cell is six invocations and the total campaign is split across cells. The retained 139-root exact-fragment comparator is HUNT_REPORT_SHARED_FRAGMENTS.md@b45b9bf0:L242-L259. |
| 5. Workload A/B | First, run retained configuration-D 50-by-6 h8/cap-500 trainer workload with an empty C_rel library to measure wrapper overhead only: artifact reservation is zero, while actual empty router/index bytes are measured and subtracted from residual solver bytes. Second, select the Stage-4 cell with the greatest lower 95% net-gain bound (ties: smaller C_rel reservation, then smaller fanout) and run live C_rel+fallback on O12's complete deep K=1/K=2 target cohort, same source library, query order, acquisition caps/horizons, exact-cache settings, and fixed total budget. Strict-verify and retain the full envelope for every hard result. | Empty-library control: upper 95% bound on wall regression must be at most 1%, else disable the wrapper on shallow profiles. Deep live cell: lower 95% bound on net wall gain must exceed 5%; no hard-to-UNKNOWN, opposite verdict, stale binding, or p95 latency regression above 5%. For trainer deployment, upper 95% throughput regression must be at most 1% and queue/park p95 regression at most 5%. | **S** for h8 (retained 300-solve total `76.513 ms` [HUNT_REPORT_LEAF_SURFACE.md@5172d42d:L8-L28,L78-L101]); **M/L** for the deep cell, serial and under 10 min per Cargo/process invocation. |
| 6. Additive gate candidate | Only after O2/O6/O8/O9/O10/O11 proof artifacts and Stages 1-5 close: implement the default-off new module/materializer and sealed-mint call. Run the selected warm cell plus the consolidated official profile with an empty cohort as a cold overhead/semantic control. Keep a fixed total 1 GiB accounted cache envelope and do not edit `tss_verify.rs`. | Require all warm hard values to be ordinary strict-accepted and snapshot-bound; warm lower 95% gain must exceed 5% over exact fragments. Cold official upper 95% wall regression must be at most 1%, with no status/certificate change. Accounted cache must stay within 1 GiB, and the maximum observed paired process-RSS regression must be at most 5%. Otherwise keep shadow-only or remove. | **L:** retained official class `495.940 s` test wall. Its hardest row's roughly 549 MiB figure is accounted search/TT peak, not process RSS [IDEATION_FINAL.md@6ef67cfe:L72-L112]. Each invocation runs alone with more than 11 GiB free at launch [IDEATION_FINAL.md@6ef67cfe:L494-L523]. |

## 8. Final assessment

The ambitious conjecture--a finite local interface whose cheap match alone
licenses a verdict--still dies at O4. The NQ2 defensive-tempo witness explains
why attacker locality is the wrong boundary, and a disjoint remote count-five
is the direct support-only counterexample shape.

The narrower project does not collapse under that obligation. It changes the
question from "did we prove all outside cells irrelevant?" to "can this
rootless body be cheaply routed to targets on which the existing strict proof
still replays?" That reduction is sound, aligns with the consolidated sealed
mint and rootless-body precedent, and is supported by high shadow acceptance
and a generation-dominated deep profile. Its open difficulty is economics,
not a new game theorem.

Accordingly, C_rel is a **real, bounded experimental project as a
strict-discharge warm template cache**. It is not yet a production design, and
it should die without regret if O13's routing experiment shows the predicted
remote-dependency/selectivity squeeze or if O12 fails the matched 5% gate.
