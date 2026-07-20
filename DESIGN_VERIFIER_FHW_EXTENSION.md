# R-G2-EXT-DESIGN — strict-verifier contract extension for FHW-T3-R

Status: **DOC-ONLY DESIGN; NOT AUTHORIZATION TO IMPLEMENT.** This document
proposes a v3 certificate contract. No verifier code may be written until the
owner accepts this design and the pre-registered shadow promotion bar in
section 9 clears without amendment after results are known.

The strict verifier remains the trusted acceptance boundary. A finder result,
shadow verdict, class tag, cached summary, stored set, or digest is never a
proof by itself. Every semantic set, clock, predicate, and guard used for v3
acceptance is reconstructed by the verifier from the exact replayed position,
the certificate arena, and independently validated annotations.

## 1. Decision and scope

The extension adds one new way for a v3 `Universal` node to discharge its
coverage obligation:

```text
explicit edges cover verifier-computed FHW zone
+ every applicable D22/RC/WC annotation and guard verifies
+ the frozen-child fixed-point check succeeds
-----------------------------------------------------------
the omitted legal complement is discharged by FHW-T3-R
```

This is not a relaxation of any existing branch. The current
`tss_verify.rs` behavior is frozen as contract v2. Version 1, wherever an
existing legacy adapter recognizes it, also stays on its old verifier path.
Only v3 may request `FhwExactOrD22V1` coverage.

The extension is deliberately narrow:

- It applies only at a post-opening, nonterminal, unforced defender
  `Universal` whose current threat transversal has `k < b_current`.
- Its descendant substitution gates must be exact or carry a valid D22
  annotation in the FHW-T3-R annotated class.
- It keeps the full scalar `B`, LOSS remainders, off-kernel escape floor,
  completion channels, resolution indices, and absolute horizons.
- It does not authorize generic D17 substitution, mixed-history `SR`, a
  scalar-`B` debit, a target-independent debit, or a claim that `Q^cut` is the
  exact global `max(F+H_W)` value.
- FHW coverage, legacy `ZoneInfo`, `implicit_dispatch`, and P3 commutation are
  mutually exclusive at the same `Universal`. An annotated D22 gate also may
  not carry P3 commutations. A future composition requires a new theorem and
  a new contract version.

The obstruction in the read-only R-G2-IMPL record is therefore resolved only
in design: v2 has no place for the required evidence and still requires the
uniform set; v3 supplies that place and a new independently checked branch.
Until v3 exists and passes its gates, a materially narrowed FHW certificate
remains unrepresentable and must not mint a value.

## 2. Normative basis and exact citations

The implementation review must use these statements, not the withdrawn
FHW-T3 rule:

1. `PROOF_TSS_ZONES_FHW.md`, **FHW-T3-R (repaired target-specific
   danger-cut extension)**, section 2.2, lines 490–525, with proof at lines
   527–579. It states soundness on the D22/RC/WC annotated class, the paired
   recurrence
   `max{b, max_d(kappa_cut^*(d,W)+Q_child^cut(W))}`, unchanged scalar `B`,
   LOSS bases and escape horizons, and the mutually exclusive edge/window
   table.
2. The same file, section 2.2b, lines 644–678, is the complete **charge
   partition**. Acceptance additionally splits the retained guard families
   into pass/fail leaves: touched guard, N-touch, direct `1+q<6`, WC, and
   N-virgin. The review-required wording matters: the charge table alone is
   not the full verifier acceptance tree.
3. The same file, section 2.2a, lines 591–603, requires rejection of the
   R-Z10 trace `d=(10,0)`, `s=(9,0)`, all-empty
   `W={(10,r):0<=r<=5}`, `q=5`: the direct charge is one and `1+5<6` is
   false.
4. `PROOF_TSS_ZONES_FHW_REVIEW2.md`, Verdict and sections 1, 4, and 6
   (especially lines 5–27, 70–120, 584–621, and 641–659), finds FHW-T3-R
   sound-with-errata on exactly that annotated class. It confirms that all
   five guard families are load-bearing, RC and WC are target-local and
   independent, inclusive distance boundaries matter, and no conclusion
   extends to arbitrary D17/D22 histories.
5. `DESIGN_GROUP2_NEXT.md`, **G2-Z1 (finite inflationary closure)**,
   section 3.2, lines 269–316. For finite `L=Legal(P)`, frozen exact child
   proofs, and
   `S_(i+1)=S_i union Zone(P,Sigma(S_i))`, success is licensed only at a fixed
   point where `Zone(P,Sigma(S_*)) subseteq S_*`. A failed child or resource
   limit fails closed; termination is not a completeness promise.
6. `DESIGN_GROUP2_NEXT.md`, section 3.1, lines 224–267, retains the D9/D10/D14
   obligations and says refined classes affect condition 6 only. Rank orders;
   it never caps. The unforced parent remains an ordinary full-cost defender
   opportunity.

The authority bundle proposed for contract identifier
`hexo-fhw-t3r-rz11-v1` is:

```text
zone authority:
  docs/PROOF_TSS_DEFENDER_ZONES.md at commit 6dc08d7a
  SHA-256 39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B
FHW proof at design input 9f589c80:
  PROOF_TSS_ZONES_FHW.md
  SHA-256 16F7D684B5D763E8B673EC3A03B5110B9ABF5BB7E80FCA063E62C81A113F9EA0
hostile review at design input 9f589c80:
  PROOF_TSS_ZONES_FHW_REVIEW2.md
  SHA-256 2591071BDF2FA77A88E2A7E7EC57B60A7F9FD871B252EA908ECAEA58D72A677F
```

The first blob is still an external-branch dependency in the reviewed proof.
It must be landed or vendored at a repository-relative immutable path before
verifier implementation. A runtime hash does not prove a theorem; the bundle
is a review/provenance gate that prevents silently implementing a different
definition.

## 3. Exact v3 grammar

### 3.1 Version envelope

The current type has no version field and the repository explicitly says that
the certificate has no public wire codec. The implementation must not pretend
otherwise. Introduce an explicit contract discriminator and freeze the
current in-memory grammar and behavior as v2:

```rust
pub const TSS_CERT_V1: u16 = 1;
pub const TSS_CERT_V2: u16 = 2;
pub const TSS_CERT_V3: u16 = 3;

pub struct TssCertificate {
    pub contract_version: u16,             // NEW; exactly 1, 2, or 3
    pub root: RootBinding,
    pub claimant: Player,
    pub root_node: CertNodeId,
    pub nodes: Vec<CertNode>,
    pub semantic_horizon: u32,
    pub fhw_evidence: Option<FhwCertificateEvidence>, // NEW; v3 only
}
```

Every existing Rust constructor is migrated mechanically to
`contract_version: TSS_CERT_V2, fhw_evidence: None`. If a legacy wire adapter
exists outside this crate, absence of a version may map to v2 only in that
explicit legacy adapter; the strict decoder never guesses a version. Unknown
versions reject.

Version rules are exact:

| version | `fhw_evidence` | `Universal.fhw_zone` / `d22_gate` | verifier path |
|---|---|---|---|
| v1 | `None` | `None` / `None` | frozen v1 adapter/path only |
| v2 | `None` | `None` / `None` | byte-for-byte-equivalent current verifier semantics |
| v3 | `Some` if either v3 field occurs; otherwise `None` | optional / optional, never both | v2 rules plus the explicitly selected v3 branches |

A v1/v2 certificate carrying any v3 field rejects. A v3 certificate with
either v3 node kind but no top-level evidence rejects. No v1/v2 certificate is
implicitly upgraded.

### 3.2 Universal-node field

Add two optional fields without changing the meaning of the four existing
fields:

```rust
CertNode::Universal {
    edges: Vec<CertEdge>,
    implicit_dispatch: bool,
    zone: Option<ZoneInfo>,
    fhw_zone: Option<FhwZoneInfo>,          // NEW
    d22_gate: Option<D22GateRef>,           // NEW
    commutations: Vec<CertCommutation>,
}

pub struct D22GateRef {
    pub annotation_index: u32,
    pub annotation_digest: [u8; 32],
}

pub struct FhwZoneInfo {
    pub selector: FhwSelector,              // must be FhwExactOrD22V1
    pub local_budget_hint: u32,             // evidence only; recomputed D14 B
    pub build_horizon: u32,                 // must equal certificate horizon
    pub closure_generations_hint: u32,      // telemetry only; never licenses
    pub scope_gate_nodes_hint: Vec<CertNodeId>,
    pub child_plan_digest: [u8; 32],
    pub finder_summary_digest: [u8; 32],
    pub claimed_zone_digest: [u8; 32],
    pub annotation_digest: [u8; 32],
    pub node_evidence_digest: [u8; 32],
}

#[repr(u16)]
pub enum FhwSelector {
    FhwExactOrD22V1 = 1,
}
```

The replayed node position, node ID, explicit edges, frozen descendant arena,
semantic horizon, and validated gate annotations are the recomputation seed.
The zone is not accepted from a finder-supplied coordinate list. The
`claimed_zone_digest` is a check against the verifier's sorted, deduplicated
reconstruction, not a substitute for it.

At an outer FHW-zone node, `implicit_dispatch` must be false, legacy `zone`
and `d22_gate` must be `None`, `commutations` must be empty, and the inherited
`allowed_commuted` context must be empty. Violating any of these conditions
disables the FHW branch.

At a descendant D22 protected tight gate, `d22_gate` selects the annotation
by index and repeats its digest. That node must have `implicit_dispatch=false`,
`zone=None`, `fhw_zone=None`, empty commutations, and no inherited commuted
reply. Its explicit edges are exactly the representatives `R`, not all of
`K(Q)`. The v3 D22 gate verifier independently derives the ordinary dispatch
boundary and kernel, validates the total mapping `K(Q)->R`, and uses the
unchanged off-kernel escape contract. This separate field is necessary: an
`FhwZoneInfo` describes an unforced outer zone, while a `D22GateRef` describes
a protected tight descendant substitution gate.

### 3.3 Certificate-wide D22 evidence

```rust
pub struct FhwCertificateEvidence {
    pub contract_id: FhwContractId,         // one known numeric discriminant
    pub authority: FhwAuthorityBinding,
    pub gates: Vec<D22GateAnnotation>,      // sorted unique by gate_node
}

#[repr(u16)]
pub enum FhwContractId {
    HexoFhwT3rRz11V1 = 1,
}

pub struct FhwAuthorityBinding {
    pub zone_authority_sha256: [u8; 32],
    pub fhw_proof_sha256: [u8; 32],
    pub hostile_review_sha256: [u8; 32],
}

pub struct D22GateAnnotation {
    pub gate_node: CertNodeId,
    pub named_threats_hint: Vec<WindowKey>,
    pub checkpoint_roles_hint: Vec<FhwRoleKey>,
    pub representatives: Vec<FhwRepresentative>,
    pub mapping: Vec<D22MapEntry>,
    pub annotation_digest: [u8; 32],
}

pub struct FhwRepresentative {
    pub mv: HexCoord,
    pub child: CertNodeId,
}

pub struct D22MapEntry {
    pub real_reply: HexCoord,               // d
    pub representative: HexCoord,           // s = phi(d)
    pub representative_child: CertNodeId,   // C_s
    pub claimed_class: D22EdgeClassHint,
    pub role_rows_hint: Vec<FhwRoleRowHint>,
    pub window_rows_hint: Vec<FhwWindowRowHint>,
    pub row_digest: [u8; 32],
}

#[repr(u8)]
pub enum D22EdgeClassHint {
    Exact = 0,
    FrontierCovered = 1,
    NonFrontierCovered = 2,
}
```

`representatives` must equal the gate's explicit `(move,child)` edges after
canonical sorting. The verifier independently derives the tight-gate threat
family, `b`, kernel `K(Q)`, and replayed representative children. `mapping`
must contain exactly one row for every `d in K(Q)`; `R` must be nonempty;
every `s` must be in `R`; `phi(s)=s`; and exact rows must have `d=s`. No
finder-supplied kernel, representative, or mapping is trusted before these
checks.

Every `D22GateRef` must point to exactly one annotation whose `gate_node` is
the referring node and whose digest matches. Every annotation must be
referenced by a reachable D22 gate in at least one FHW node's frozen summary;
unused, duplicate, or multiply indexed annotations reject. For each outer FHW
node, `scope_gate_nodes_hint` must equal the verifier-derived sorted unique set
of contributing D22 gate IDs.

Role and window rows are explicit audit evidence and diagnostic hints:

```rust
pub struct FhwRoleKey {
    pub carrier: HexCoord,
    pub source_node: CertNodeId,
    pub source: FhwRoleSource,
    pub witness: Option<WindowKey>,
}

#[repr(u8)]
pub enum FhwRoleSource {
    ChoicePlacement = 0,
    OrCompletionPlacement = 1,
    WinWitnessEmpty = 2,
    LossWitnessEmpty = 3,
    GateCheckpointEmpty = 4,
}

pub struct FhwRoleRowHint {
    pub role: FhwRoleKey,
    pub claimed_f_cut: u32,
    pub claimed_rc_pass: bool,
    pub claimed_transition_charge: u8,      // 0 or 1
}

pub struct FhwWindowRowHint {
    pub window: WindowKey,
    pub claimed_q_cut: u32,
    pub claimed_row: FhwWindowRow,
    pub claimed_charge: u8,                 // 0 or 1
    pub claimed_guard_pass: bool,
}

#[repr(u8)]
pub enum FhwWindowRow {
    NonDAlive = 0,
    ExactOrFcNonincident = 1,
    ExactOrFcDirectTouched = 2,
    ExactOrFcDirectEmpty = 3,
    NonFcTouchedNonincident = 4,
    NonFcTouchedDirect = 5,
    NonFcEmptyDirect = 6,
    NonFcEmptyNonincidentQlt6 = 7,
    NonFcEmptyNonincidentWcPass = 8,
    NonFcEmptyNonincidentWcFail = 9,
}
```

`witness` is `None` for placement roles and `Some` for witness/checkpoint
roles. The verifier checks that `source_node` has the named node kind, the
window exists at the replayed source position, and `carrier` is exactly the
named move or a required empty of that window. Any other combination rejects.

The verifier derives the complete reachable role set and finite window query
set from replay, including attacker placements, leaf-witness empties, named
threat/checkpoint roles, both completion channels, and every reachable LOSS
remainder. It then recomputes every `f^cut`, `Q^cut`, RC/WC result, row,
charge, and applicable guard. Hint rows must form a canonical bijection with
the derived rows and match them exactly. A missing, duplicate, extra, stale,
or false hint rejects the FHW branch. Thus the fields make annotations
auditable without making finder-supplied sets authoritative.

### 3.4 Canonical digests and caps

Use SHA-256 over a specified canonical binary encoding, never Rust
`Hash`, debug text, JSON map order, pointer identity, or platform-sized
integers. Integers are fixed-width little-endian; vectors have `u32` length
prefixes; coordinates are `(i16 q,i16 r)`; enum discriminants are the values
above; lists are sorted by their documented semantic key and duplicates are
rejected. Every hash begins with a NUL-terminated ASCII domain separator.

Required domains are:

```text
hexo-tss-v3/certificate-payload\0
hexo-tss-v3/child-plan\0
hexo-tss-v3/finder-summary\0
hexo-tss-v3/zone\0
hexo-tss-v3/d22-row\0
hexo-tss-v3/d22-annotation\0
hexo-tss-v3/node-evidence\0
```

`certificate-payload` covers all non-digest certificate fields and preserves
arena IDs and shared-DAG edges. `child-plan` covers the FHW node ID, its
replayed position binding, sorted explicit `(move,child)` pairs, and the
complete reachable child subarenas. `finder-summary` covers the claimed full
`Sigma`: scalar `B`, roles, LOSS/horizon data, and all role/window labels.
`zone` covers the verifier's sorted deduplicated coordinates. Each row and
annotation digest excludes its own digest field. `node-evidence` covers the
certificate-payload digest, node ID, replayed position binding, selector,
authority bundle, horizon, recomputed budget, the other four node digests,
and the sorted gate IDs in scope. This prevents accidental or adversarial
transplant between nodes, roots, horizons, child plans, or authority epochs.

Digests are consistency checks, not semantic oracles. Even a digest match
does not skip replay or recomputation.

Add hard limits checked with `checked_add` before allocation:

```text
FHW zone nodes             <= MAX_CERT_NODES
D22 gate annotations       <= MAX_CERT_NODES
D22 mapping rows           <= MAX_CERT_EDGES
role-row hints              <= 1,000,000 total
window-row hints            <= 1,000,000 total
encoded v3 evidence bytes   <= 64 MiB
verifier FHW work items     <= 20,000,000
```

Crossing a limit rejects the FHW branch; it never truncates a set or converts
an absent row to a zero charge. The exact caps may be lowered by the owner
before implementation, but never raised adaptively for a certificate under
verification.

## 4. Precise acceptance rule

Let `N` be a v3 `Universal` with `fhw_zone=Some`, replayed position `P`, legal
set `L`, explicit edge moves `S`, claimant `A`, defender `D`, and frozen exact
child arena below every move in `S`. Sort and deduplicate all sets by
`(q,r)`; duplicate explicit moves reject as today.

The verifier performs the following steps in order.

1. **Baseline node checks.** Apply every current `verify_universal` check:
   exact root/state replay, defender to move, nonterminal, no defender
   `own_win_now`, nonempty represented set, legal explicit moves, no
   defender-terminal edge, exact child replay, arena bounds, acyclicity,
   reachability, depth, memo identity, and leaf/horizon checks.
2. **Mode check.** Require v3, the exact authority bundle and selector,
   `implicit_dispatch=false`, `zone=None`, `d22_gate=None`, no commutations
   or inherited commuted replies, post-opening phase, and `k<b_current`. Any
   unproved composition disables this branch.
3. **Frozen summary.** From the actual child subarenas selected by `S`,
   reconstruct D10 roles, checkpoint masks, D14 local `B`, LOSS remainders,
   exact resolutions, completions, and the complete finite role/window query
   index. The stored budget and summary are hints. Require
   `local_budget_hint=B` and `build_horizon=cert.semantic_horizon`; retain the
   existing global derived-horizon inequalities.
4. **Gate annotations.** Traverse the frozen summary's reachable protected
   gates. For each, independently derive the named threat family,
   `tau(F_Q)=b`, `not own_win_now(P_Q)`, `K(Q)`, `R`, total retraction `phi`,
   exact representative children, and full D22 conditions: unchanged
   `Bhat=1+B(C_s)` and D14/D15/D16 inequalities, every descendant/checkpoint
   role, every LOSS/nesting/completion clause, WF legality, A2/A3 inheritance,
   nonempty `R`, and direct avoidance of every obligation carrier. A shared
   DAG node must have one annotation and one clock label; otherwise it must
   have been split before folding.
5. **Class and clocks.** Recompute exact/global-FC using inclusive
   `B_8(d) subseteq Lambda(P_Q+s)`. For genuine non-FC rows, reconstruct
   `GI(P_Q+s)`, then calculate branch-paired `f^cut` and `Q^cut` in reverse
   topological order. Form each edge charge plus its own child's clock before
   taking the maximum. Never form independent marginal maxima. Keep the
   off-kernel `b` floor and full LOSS base in the same branch maximum.
6. **All five guard families.** For every derived edge/target row, evaluate
   every guard applicable to that row. “All five” means the complete families
   below are implemented and every applicable instance is checked; WC-pass
   and WC-fail/N-virgin are mutually exclusive, so they are not simultaneous
   requirements on one row. In every bullet,
   `q=Q^cut_(C_phi(d))(W)` is verifier-derived.

   - **Touched guard (exact or global-FC direct):** for D-alive touched `W`
     with `d in W`, require
     `cnt_D(W,P_Q) + 1 + q < 6`.
   - **N-touch (genuine non-FC direct):** for D-alive touched `W` with
     `d in W`, require the same strict inequality using the derived cut-child
     `q=Q^cut_(C_phi(d))(W)`.
   - **Direct all-empty guard:** for every D-alive all-empty `W` with
     `d in W`, in every exact/FC/non-FC class, charge one and require
     `1+q<6`. Direct incidence is terminal in the decision tree; `q<6` or WC
     can never overwrite it.
   - **WC:** only for genuine non-FC, D-alive all-empty, nonincident
     `d notin W`, `q>=6`, evaluate
     `GI(G) intersect B_8(d) intersect B_(8(q-6))(W) = empty`, where
     `G=P_Q+s`. A pass permits window charge zero.
   - **N-virgin:** on the same row when WC fails, charge one and require
     `dist(d,W) > 8(1+q-6)`. Equality fails.

   Independently, every role gets the RC decision
   `GI(G) intersect B_8(d) intersect B_(8(k-1))(y)=empty`, with the last ball
   empty for `k=0`. RC can remove only the named role's transition charge;
   it cannot change a window row or guard. If RC fails, require the D22-N
   radius `dist(d,y)>8*f^cut_child(rho)` and charge one.
7. **Recompute the zone.** Using the pinned D21 formulas, the replayed
   position, and the verified `f^cut/Q^cut` summary, reconstruct
   `Z_dir union Z_seed union Z_touch union Z_virgin`, intersect it with `L`,
   apply the deterministic nonempty fallback, and call the result `Z_FHW`.
   No finder list participates. Check every digest only after the semantic
   reconstruction.
8. **Fixed point and coverage.** Require every explicit edge to have the same
   frozen child used in the summary and require
   `Z_FHW subseteq S subseteq L`, with `S` nonempty. This is the verifier form
   of `Zone(P,Sigma(S_*)) subseteq S_*`; the numeric closure-generation hint
   is irrelevant.
9. **Accept.** Verify every explicit child recursively. Only after all prior
   steps succeed may the verifier treat the omitted `L\S` as discharged by
   FHW-T3-R and accept the `Universal` as logically full-covered.

For a `Universal` with `d22_gate=Some`, the corresponding protected-gate
subrule is also exact. The verifier applies the baseline node checks, derives
the ordinary dispatch boundary and full kernel `K(Q)`, validates the reference
and its total D22 annotation, and requires the explicit edges to equal the
nonempty representative set `R`. It recursively verifies each representative
child from the real replay `P_Q+s`. It evaluates the D22/RC/WC rows, clocks,
and all applicable guards over the complete union of role/window queries from
every enclosing FHW scope that reaches this shared gate; the annotation and
clock label must be identical in every scope. The existing T6 boundary
discharges off-kernel replies, and the validated FHW-T3-R mapping discharges
each kernel reply `d` through its verified `C_phi(d)`. Only then may this gate
be accepted without an explicit child at every `d`. A D22 gate not reachable
from a verified outer FHW scope, or reached with incompatible query/label
requirements, rejects.

Fail-closed fallback is exact: on any FHW failure, discard all FHW-derived
facts. If the represented moves independently equal the full sorted legal set,
the node may be checked by the ordinary explicit-full branch; otherwise reject
the certificate. A producer should normally remint that case with
`fhw_zone=None`. It is forbidden to fall back to the legacy uniform zone after
an annotation failure, because the edges may cover only the smaller FHW set.
Resource exhaustion, unknown enum values, arithmetic overflow, hash mismatch,
or an unavailable authority has the same result.

## 5. Soundness reduction and trust ledger

Assume the v3 branch accepts at `N`. Steps 1–6 establish that the frozen arena
is in the exact D22/RC/WC annotated class and that every charge and guard is the
one required by FHW-T3-R. Therefore, by FHW-T3-R as cited in section 2, D21's
zone computed with the branch-paired target-local cut clocks safely protects
every omitted defender continuation while retaining all ordinary, terminal,
LOSS, escape, role, window, legality-frontier, and horizon channels.

Steps 3, 7, and 8 use exactly the child proofs that generated `Sigma(S)` and
establish `Zone(P,Sigma(S)) subseteq S`. By G2-Z1 as cited in section 2, that
successful frozen fixed point licenses the final restricted `Universal`.
Recursive verification proves every child in `S`; FHW-T3-R discharges
`L\S`. Hence all legal defender replies are covered logically even though
not all are represented as edges.

New trust assumptions introduced by v3 are exhaustively:

1. the pinned D14–D22/T3/T11 authority and the repaired FHW-T3-R theorem are
   sound on their stated class;
2. the implementation is a faithful, reviewed translation of the D22,
   RC/WC, `f^cut/Q^cut`, charge-tree, five-guard, D21-zone, and G2-Z1
   fixed-point definitions;
3. the independent verifier's role/window enumeration includes every
   reachable placement, witness empty, checkpoint role, completion channel,
   LOSS remainder, and nested path required by the theorem;
4. fixed-width checked arithmetic, inclusive hex distance/balls, canonical
   window identity, and finite-DAG traversal are implemented correctly;
5. version decoding is unambiguous and v3-only fields cannot reach a v1/v2
   verifier branch; and
6. the new verifier work/memory limits reject rather than truncate.

Existing inherited assumptions remain the engine's legality, apply/undo,
window store, threat analysis, root binding, typed leaves, and the present
arena verifier. SHA-256 collision resistance is useful for artifact identity
and transplant detection but is not a game-soundness assumption: no digest
ever replaces semantic recomputation.

The extension cannot weaken v1/v2 verification. Dispatch is a top-level
version match: v1 calls only the frozen v1 adapter, v2 calls a mechanically
preserved `verify_v2`, and v3 alone can call `verify_fhw_zone_v3`. No v1/v2
predicate consults, defaults, or deserializes FHW evidence. A differential
golden test must prove the refactor returns the same verdict for every existing
v1/v2 fixture and mutation. Unknown versions reject. Thus the accepted set for
v1 and v2 is identical before and after this extension.

## 6. Threat model and mandatory rejection tests

The finder and stored certificate are adversarial inputs. The following
attacks and defenses are mandatory:

| attack | fail-closed defense |
|---|---|
| Claim v3/FHW with no evidence, wrong authority, or unknown selector | Version/mode/authority check rejects the narrowed branch. |
| Supply a smaller fake kernel or omit a mapping | Verifier derives `K(Q)` and requires a total one-row-per-`d` retraction. |
| Use empty `R`, map outside `R`, fail `phi(s)=s`, or point at a different child | Exact representative/edge/child bijection rejects. |
| Map an illegal or terminal reply | Engine replay and nonterminal-child checks reject. |
| Lie that a genuine edge is exact or FC | Verifier recomputes equality and inclusive frontier coverage. |
| Omit a role, checkpoint, LOSS path, or target window | Verifier derives the complete finite index; hint bijection and summary digest fail. |
| Forge `f`, `q`, `B`, a row, or a zero charge | Reverse-topological recomputation and exact hint comparison fail. |
| Splice the cheapest charge from one child with another child's clock | Edge-plus-own-child expressions are formed before every maximum. |
| Apply RC zero to another role or use RC to erase a window charge | Role keys are exact; RC and window axes are independently recomputed. |
| Query WC on a direct, touched, or `q<6` row | Decision-tree ordering makes that row inexpressible; hint mismatch rejects. |
| Hide a direct fill behind the old overlapping `q<6` rule | Direct incidence is terminal, costs one, and the strict direct guard is checked. |
| Exploit distance-eight equality or integer overflow | Inclusive balls plus checked radius multiplication/addition reject equality/overflow. |
| Reset a nested window after an earlier real-only fill | Path-local earliest-fill state is retained on finite unfolding; a path-dependent folded label rejects. |
| Swap a child proof after closure or transplant evidence to another node/root/horizon | Frozen subarena identity and node/child/annotation digests mismatch. |
| Provide only one closure pass or omit a newly required edge | Verifier recomputes the final zone and detects `Z_FHW` not contained in `S`. |
| Combine FHW with legacy zone, implicit dispatch, commutation, SR, or generic D17 | Mode check rejects the unproved composition. |
| Exhaust memory/work to induce partial acceptance | Caps return rejection/Unknown; no prefix or truncated set is accepted. |
| Downgrade v3 bytes to v2 | Explicit version plus forbidden-field checks reject; no permissive auto-detection. |

The mutation suite starts from at least one strictly narrowed, accepted v3
certificate, so full-legal fallback cannot mask a broken annotation. Every
mutation below must reject:

1. delete one FHW-required explicit edge;
2. delete, duplicate, or add one kernel mapping; set `R` empty; break
   `phi(s)=s`; change a representative child ID;
3. flip each edge class (`Exact`, FC, non-FC) independently;
4. delete/add one role and one window hint; alter a source node, carrier,
   witness, `f`, `q`, `B`, LOSS remainder, build horizon, or completion index;
5. flip each of the five guard outcomes independently and exercise both sides
   of every strict boundary;
6. mutate RC at `k=0` and at inclusive distance eight; mutate WC at `q=6`
   (`B_0(W)`) and inclusive distance eight; try to transfer an RC pass into a
   WC pass on the same edge;
7. change each digest, transplant intact evidence between two nodes with equal
   set cardinality, reorder rows, and introduce a duplicate canonical key;
8. mutate a shared DAG so one incoming path needs a different annotation;
9. force a second closure generation, then remove the newly added edge; test a
   disappearing fallback and prove that old edges remain retained;
10. add a terminal defender edge; inject an `own_win_now` state; use Opening;
    combine FHW with commutation, legacy zone, implicit dispatch, or SR;
11. cross every resource cap and every checked arithmetic boundary; and
12. D6-remap a valid v3 certificate, recomputing all coordinates and digests;
    then mutate one unmapped role/window coordinate and require rejection.

### Mandatory R-Z10 rejection trace

The suite must encode the complete reachable prefix from
`PROOF_TSS_ZONES_FHW_REVIEW2.md` A0:

```text
D (0,0)
A [(5,0),(6,0)]       D [(0,2),(2,0)]
A [(7,0),(8,0)]       D [(-2,2),(1,2)]
A [(5,1),(6,1)]       D [(2,-2),(-1,-1)]
A [(7,1),(8,1)]
```

At the resulting defender FirstStone node, use
`U_i={(q,i):5<=q<=10}` for `i=0,1`, whose disjoint empty pairs are
`{(9,i),(10,i)}`, so `tau=b=2`. Map real `d=(10,0)` to
`s=(9,0)` and target `W={(10,r):0<=r<=5}`. `W` is D-alive,
all-empty, and incident; FC fails at `z=(18,0)`. The representative child has
five W-hazards `(10,1)`, `(10,2),(10,3)`, and `(10,4),(10,5)`, so
`q=5`. The only valid row is direct non-FC/all-empty:

```text
kappa_cut^* = 1
edge value  = 1 + 5 = 6
guard       = 1 + 5 < 6  // false
```

The narrowed certificate must be rejected. A mutant implementing the
withdrawn first-match/zero interpretation is a required test-harness
self-test and must be caught.

## 7. Re-verification and mixed stores

The P0–P3 precedent is binding. `docs/PLAN_TSS_SOLVER_UPGRADES.md` lines
753–758 says stored old-engine zone certificates were not promoted across a
new verifier contract without re-verification. The same plan records that the
G2R3 radius shrink re-verified all prior certificates (line 112) and that any
previously verifying certificate failure required stop/revert/write-up (U20,
line 728).

Use this protocol:

1. Freeze the old verifier build, engine/rules identity, corpus manifest,
   roots, claimed status, horizon, and canonical certificate bytes or
   structural digest. Because this repository has no public certificate wire
   codec, an external record containing only `tss_cert_id` or a digest is not
   replayable evidence and must be quarantined rather than “verified.”
2. Before enabling v3 minting, replay every retained v1/v2 certificate under
   its original version path and exact root. Record old verdict, new verdict,
   engine identity, verifier build, horizon, and failure reason. Required
   result: exact verdict agreement and acceptance of every previously accepted
   still-supported certificate. Any mismatch stops promotion.
3. Do not rewrite a v1/v2 certificate in place. To obtain v3, rerun the finder,
   freeze a new child plan and annotations, mint a new digest/ID, and verify it
   from the root. Version is part of identity.
4. Flush process-local `CachedProof`/`ProvenFragmentStore` state across the
   upgrade. Any future persistent fragment store keys by
   `(contract_version, root binding, claimant, horizon, certificate digest,
   authority contract_id)` and verifies on import before reuse.
5. A mixed store may contain accepted v1, v2, and v3 records. The reader
   dispatches by exact version; deduplication never aliases different versions.
   A hard value is usable only after that record's version-specific verifier
   accepts it in the current engine. v2 does not need FHW evidence and v3 does
   not retroactively change v2's coverage rule.
6. Publish counts by version: discovered, replayable, accepted, rejected,
   quarantined-no-payload, and reminted. Zero unexplained verdict drift is a
   release gate.

This is consistent: each accepted record is justified by exactly one frozen
contract. Coexistence does not combine premises across versions, and the v3
extension changes only the set accepted by the v3 dispatcher.

## 8. Implementation plan and estimated scope

This section is planning only.

1. `packages/hexfield_eq/rust/src/tss_verify.rs`: add the version dispatcher,
   v3 structs/caps, universal mode selection, full-legal fallback, digest
   entry points, D6 remapping hooks, and preserve current logic as the v2 path.
2. `packages/hexfield_eq/rust/src/tss_verify/fhw_v3.rs` (new): pure verifier
   reconstruction for roles/windows, D22 mapping validation, FC/RC/WC,
   branch-paired clocks, guard tree, D21 zone, and canonical encoding. It may
   depend on the engine and `threats_shared`, never on `tss_solver`.
3. `packages/hexfield_eq/rust/src/tss_solver.rs`: only after promotion,
   materialize the already-shadowed frozen evidence and v3 nodes. Finder
   helpers remain separate from verifier helpers; no shared set-construction
   function is allowed.
4. `packages/hexfield_eq/rust/src/lib.rs`: declare the verifier submodule and
   test harness module as needed.
5. `packages/hexfield_eq/rust/Cargo.toml` and the workspace lockfile: add one
   pinned SHA-256 implementation unless an audited workspace primitive is
   introduced first. Digest code is not allowed to replace semantic checks.
6. A new verifier-focused Rust test module/file plus the existing solver
   corpus tests: grammar/version tests, v1/v2 differential goldens, all
   mutations in section 6, R-Z10 A0, R11-A/B/C, review2 A1–A10 boundaries,
   multi-generation closure, shared DAGs, D6, caps, and mixed-store import.
7. Re-verification tooling/records under the existing Group-2 campaign
   convention: immutable manifest, per-certificate results, mutation report,
   and final gate disposition. These are implementation artifacts, not part
   of this doc-only round.

Estimated change: 900–1,400 verifier-TCB lines, 500–900 solver/materializer
lines, 500–800 test lines, and 150–300 schema/digest/re-verification lines;
roughly 2,050–3,400 lines total. The review burden is dominated by role/window
completeness, branch-paired clock recursion, the five guard families, and
resource bounds—not by the struct additions.

The shadow specification already provides directly:

- the `FhwExactOrD22` class verdict and authority digests;
- the frozen certificate/node/position key, child-plan digest, role and
  summary identity, horizon, explicit coordinates, and closure generations;
- per-edge/per-window FHW-T3-R decision-tree rows, including rejection of an
  all-empty direct edge with `1+q>=6`;
- uniform/exact/FHW sizes and eligibility; and
- finder-summary, verifier-summary, strict-verdict, solve wall, node, TT, and
  memory telemetry fields specified in `DESIGN_GROUP2_NEXT.md` sections
  5.1 and 6.5.

Those records can populate v3 hints and golden fixtures. They do not provide
runtime proof: R-G2-IMPL stopped before implementation, and its obstruction
correctly notes that no current certificate can serialize these fields or
pass a materially narrowed set through the unchanged verifier. Every shadow
field must therefore be independently reconstructed again by v3.

## 9. PRE-REGISTERED PROMOTION BAR (proposed; owner-adjustable only before runs)

These gates must be frozen in a signed/hashed manifest before inspecting any
new shadow result. They are intentionally stronger than “the formula is
sound”: adding roughly a thousand permanent trusted-base lines plus a full
re-verification obligation needs broad reach and measurable end-to-end value.

### 9.1 Preconditions

- The owner has accepted this design in writing.
- The 2,011-line authority is landed/vendored and its byte digest matches the
  authority bundle. Any identity mismatch is `INELIGIBLE`, never repaired
  after seeing results.
- No verifier source has changed for the measurement. FHW remains shadow-only
  and cannot mint a hard value; all reported hard statuses come from current
  strict v2 verification.
- The cohort, official profiles, roots, horizons, caps, feature flags,
  repetitions, compiler, machine, and formulas below are manifest-frozen.
  Use three clean matched repetitions and medians, with no warm TT sharing.
- The R-Z10 A0 trace and every available guard-boundary shadow mutation are
  rejected before materiality is evaluated.

### 9.2 Required thresholds

Let `J` be all replayed unforced `Universal` occurrences with
`k<b_current` in strict-accepted official-profile certificates. Let `e_j` be
baseline exclusive expansion time at occurrence `j` (child time is charged to
the child occurrence, so nesting is not double-counted), and let `E` be the
subset independently classified FHW-eligible before any size/time result is
read.

1. **Breadth:** at least 30 distinct eligible D22/exact gates and at least 20
   eligible unforced nodes from at least 10 distinct certificate digests and
   two official profile families. No single certificate may contribute more
   than 50% of eligible exclusive wall.
2. **Wall-weighted coverage:** require

   ```text
   C_wall = sum_{j in E} e_j / sum_{j in J} e_j >= 0.25,
   ```

   with a positive denominator. This prevents a large count of trivial nodes
   from justifying a permanent verifier branch.
3. **Semantic materiality:** on the exact frozen indices from
   `DESIGN_GROUP2_NEXT.md` section 6.5, require both the reviewed FHW clock
   reduction and matched net-zone reduction to be at least 10%:

   ```text
   1 - sum Q_new / sum E_old >= 0.10
   1 - sum |S_FHW| / sum |S_uniform| >= 0.10.
   ```

   Require `S_FHW subseteq S_uniform` at every node and zero unmatched keys.
4. **Counterfactual official-wall floor:** run a shadow consume simulation
   that may schedule the narrowed set but cannot export its verdict. For each
   matched repetition include finder annotation time, evidence encoding, the
   independent shadow-checker time used as a conservative proxy for v3
   verification, and any fallback/reconstruction time. Define

   ```text
   S_wall = 1 - sum_jobs wall_shadow_fhw / sum_jobs wall_uniform.
   ```

   Require median-repetition `S_wall >= 0.075` (7.5% of total
   official-profile solve wall), positive savings in at least two profile
   families, and no hard-status disagreement after the shadow job is rerun or
   rematerialized with uniform v2 coverage for current strict verification.
5. **Permanent-cost bounds:** independent evidence/checking overhead alone is
   at most 3% of matched uniform solve wall; median encoded evidence growth is
   at most 15% and p95 at most 30% of the corresponding v2 certificate bytes;
   no valid sample hits a proposed v3 work/memory cap.
6. **Correctness:** zero accepted broken mutations, zero missing/extra derived
   role or window rows, zero digest/summary disagreements, zero v1/v2 verdict
   drift in the full re-verification corpus, and zero use of SR, generic D17,
   commutation, implicit dispatch, or legacy zone semantics inside an FHW
   node.

### 9.3 Kill and defer criteria

Any correctness failure, R-Z10 acceptance, authority mismatch, version drift,
unmatched comparison key, non-subset FHW set, or truncation is an immediate
**KILL** for verifier implementation pending a new reviewed theorem/design.

Failure of breadth, `C_wall`, either 10% semantic threshold, the 7.5% wall
floor, or the permanent-cost bounds is **KEEP SHADOW / DO NOT WRITE VERIFIER
CODE**. Do not repair a failed preregistration by deleting profiles, changing
horizons/caps, weakening denominators, reclassifying ineligible nodes as zero
savings, or tuning thresholds after results. The owner may set different
numbers only before the manifest is frozen; any later change is a new named
round with a new baseline.

The proposed 25% coverage and 7.5% net-wall floors are higher than a typical
finder-only experiment because the cost here includes a versioned grammar,
roughly 900–1,400 permanent trusted-base lines, adversarial resource handling,
digest/schema maintenance, and re-verification of every retained certificate.
The existing 10% clock/zone bars establish local semantic value; the added
coverage and wall bars require that value to be common and large enough to
pay for the enduring contract surface.

## 10. Owner ruling preserved

This design does not authorize implementation or consumption. The strict
verifier is never weakened. Verifier work begins only after both independent
conditions hold: the owner explicitly accepts this document, and a frozen
shadow campaign clears every pre-registered bar in section 9. Until then,
FHW evidence is telemetry only, all certificates continue to use the existing
v1/v2 rules, and any verifier failure yields rejection/Unknown rather than a
hard value.
