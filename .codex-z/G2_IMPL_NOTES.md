# G2 certificate extension — implementation running notes

Branch `claude/g2-cert`. Spec: `.codex-g2-resolve/DESIGN_G2_CERT_EXTENSION.md`
(+ `DESIGN_AMENDMENT_R1_R2.md`, `HOSTILE_REVIEW_1.md` §4).

## Scope decision (recorded before writing code)

The full §3 algorithm (ordinary Group-2 nodes + FHW-T3-R gates with
RC/WC/FC/kappa rows and the gate-local WC demand fixed point) is estimated by
the design itself at 2,800-4,400 production lines. Per the task's explicit
narrowing clause, this implementation validates a **strict, fully-covered
subset** and rejects everything else:

- **Accepted v1 sub-class: gate-free Group-2 certificates.** Certificates
  containing `UniversalGroup2V1` nodes but **no `FhwGateV1` node**. Any
  certificate containing an `FhwGateV1` node REJECTS (documented narrowing —
  §3.3 gate reconstruction, RC/WC/FC predicates, kappa/epsilon tables and the
  gate-local WC enumeration are NOT implemented in this session).
- On gate-free trees the cut clocks coincide with the full clocks
  (`f_cut == r_full`, `Q_cut == E_full`: the gate clauses of the §3.2
  recurrences are the only place they diverge). Both are still computed as
  written and the `f_cut<=r_full`, `Q_cut<=E_full<=B` inequalities checked.
- Everything else from the design that applies to gate-free certificates is
  implemented as specified: narrow-v1 tree/no-mixing preflight, dual authority
  binding, direct D9 typed-leaf checks, exact `k<b` reconstruction, full D14
  `B`, §3.4 `Required_FHW` (Z_dir/Z_seed/Z_touch/Z_virgin from exact per-role
  and per-window clocks), Z4 anchor recheck, canonical order/duplicate
  rejection, checked arithmetic (overflow => reject), work caps, R1, R2, and
  the child-plan/summary digest recomputation of §2.4 (gate-free payloads).

Consequences for the mandated tests are recorded per test below; hostile §4
constructions that require gates (C1 q=5 replay, C3 root-gate WC row
deletion, C4 role splice) are not constructible inside the accepted class —
certificates containing gates reject wholesale — and are documented instead,
with gate-free analogues where one exists (C2's omitted-required-reply
analogue is constructible and tested).

## Deviations / narrowings (honest list, kept current)

1. **FhwGateV1 wholesale-rejected** (above). R1's per-gate
   `escape_resolution_ply <= semantic_horizon` requirement is enforced a
   fortiori (no gate can be accepted at all); the R1 metadata rule
   ("escape deadlines participate in derived T") IS implemented in
   `certificate_metadata` so the derived-T maximum includes
   `escape_resolution_ply` for any gate node encountered before the class
   rejection fires.
2. **`not own_win_now` reconstruction is a conservative over-approximation**:
   reject when any mover window has `count(mover) + placements_remaining >= 6`
   and zero opponent stones (ignores empty-cell legality, so it rejects a
   superset of true win-now positions — sound, slightly incomplete), AND
   additionally reject when `threats_shared::analyze(..).own_win_now` fires.
3. **Finder and verifier share the derivation helpers** (roles/clocks/zone and
   the digest encoders live in `tss_verify_group2.rs`; the solver calls them
   at emission/post-processing). The design wants an independent finder-side
   derivation; verifier soundness against adversarial certificates is
   unaffected (the verifier re-derives from replay at verification time), but
   the correlated-bug redundancy the design asks for is NOT provided in this
   session. Flagged as the main reviewer-scrutiny item.
4. Demand-row propagation micro-rules I had to fix (design leaves them at
   pseudocode granularity, both sides use the same definition): a demanded
   window stops propagating below an OR whose move enters W, at typed leaves,
   and at the first node where W is non-D-alive (its row is recorded there
   with Q=E=0); source bits OR on coincidence.
5. Loss-witness lists inside new-class certificates must be sorted-unique by
   canonical window key (the solver post-pass sorts them); legacy certificates
   keep the old unordered acceptance.

## What was built (file map)

- `packages/hexfield_eq/rust/src/tss_verify.rs`
  - Exact §2.2 record types (`Group2AuthorityV1`, `Group2ZoneV1`, the full
    FHW gate record family incl. `RoleKeyV1`/kappa rows — carried in the
    grammar even though gates are narrowed out of acceptance) and the two
    appended boxed `CertNode` variants (`UniversalGroup2V1`, `FhwGateV1`).
  - Compiled dual authority binding constants (§1.1 six fields, byte-exact).
  - LegacyOnly policy: `verify_certificate` rejects any certificate carrying
    an extension node BEFORE any other work; every legacy certificate takes
    the byte-identical old path.
  - New `Group2Verifier` (policy `Group2V1`): legacy certs → unchanged legacy
    path; extension certs → `tss_verify_group2`.
  - Shared structural helpers extended with explicit (never wildcard) arms:
    metadata (R1: gate escape deadlines fold into derived T), arena
    validation (bounds/caps/duplicate-edge for new variants), reachability,
    `d6_remap_certificate` (full extension-node remap; canonical re-sort of
    edge/witness lists ONLY when the cert contains an extension node —
    legacy remap output is byte-identical).
- `packages/hexfield_eq/rust/src/tss_verify_group2.rs` (new, verifier-only —
  no `tss_solver` import): self-contained SHA-256 (FIPS golden vectors
  pinned), §2.4 scalar encoders, narrow-v1 preflight (strict tree, no
  mixing, schema/authority, canonical order, R2), replay with per-node
  direct D9 checks (typed leaves incl. replayed continuations, exact-`k`
  reconstruction, conservative own-win-now), D14 `B`, role clocks, window
  demand fixed point with source bits, `Q_cut`/`E_full` (+`<=B` check),
  §3.4 `Required_FHW` + coverage, §2.4 child-plan/summary Merkle digests
  over all 12 D6 transforms (lexicographic min), work caps (checked
  arithmetic throughout; every failure is a reject).
- `packages/hexfield_eq/rust/src/tss_solver.rs`
  - `group2` solver option (`set_group2` with cache-clear-on-change,
    `group2_enabled`, `EffectiveSolveConfig.group2` — same pattern as
    `dual_pass`).
  - Selector in `NarrowCompatSearch::prove_universal`: flag on + class
    preconditions (post-opening, unforced defender node, b∈{1,2}, exact
    k<b, conservative no-win-now) + no dirty node emitted yet → G2-Z1
    append-only closure seeded from the hitting universe, exact
    `Required_FHW` recomputed against frozen children each round; emits a
    placeholder `UniversalGroup2V1`. Any failure falls through to the
    unchanged legacy paths.
  - `prove_narrow_compat`: post-compaction finalization for extension certs
    (`finder_finalize_group2`: canonical sort, DAG→tree unfolding, derived
    scalars, digests) + strict self-verification under `Group2Verifier`;
    ANY group2-enabled attempt that fails to produce a verified certificate
    re-solves cleanly with the selector off (costs summed) — this is what
    makes "flag-on never decides fewer positions" structural, not
    statistical.
  - Cache discipline: `CachedProof::from_compact` refuses extension nodes
    (nothing new-class ever enters the shared TT / fragment store);
    compaction/remap handle `UniversalGroup2V1` and refuse gates; complete
    boxed-v3 heap accounting for both new variants alongside the untouched
    legacy formula.
- Mint: `tss_core::hard_value_from_verified_group2` (sealed, concrete
  `Group2Verifier` parameter, mirroring the legacy sealed mint); `tree.rs`
  selects the policy from the SOLVER flag (trainer configuration), never
  from certificate contents.
- Plumbing end-to-end (config.py `tss_solver_group2` +
  `build_divergence_overrides`; tree.rs `Divergences` + root/inline/async
  request wiring; tss_async `SolveRequest.group2` + worker set; search.rs
  divergence whitelist/extract, root-guard solver set, batch API kwarg
  `group2=false` and manifest kwarg (trailing, source-compatible), manifest
  echoes `group2`, batch rows report `group2_nodes`).
- Harness: `tss_batch.py` DEFAULTS `"group2": False`, passed through
  manifest+batch, `declared_features` claims `"group2"`; `canaries.py`
  registered `group2` canary (manifest truthfulness both ways + identical
  verified verdicts on the `wide_win`+`loss_pos` fixture battery on/off +
  zero verify-failures).

## Selector-firing fixture

FOUND/CONSTRUCTED (Rust unit test
`selector_emits_reduced_fanout_certificate_that_verifies`): a
defender-to-move (SecondStone, b=1, k=0) position where the claimant holds
three separated three-in-a-row groups; after any defender reply, extending
two untouched groups to four yields a tau>2 LOSS leaf, so the entire proof
is gate-free. The selector fires at the root, emits a reduced
`UniversalGroup2V1` (measured: 19 explicit root edges vs 886 full legal
moves, a ~46x local fanout reduction), the certificate
strictly verifies under `Group2Verifier`, and strictly REJECTS under the
default legacy policy. All 12 D6 images verify (stored digests are
D6-invariant by the lexicographic-min construction).

On the harness fixture battery (`wide_win`/`loss_pos`, wide profile) the
selector does NOT fire — the wide narrow-compat path refuses unforced
nodes, and real TSS proofs lean on implicit dispatch, which the v1 class
excludes (gates are the class's replacement for dispatch, and gates are
narrowed out this session). The canary therefore checks verdict-identity
and manifest truthfulness, not firing; firing evidence is the constructed
Rust fixture. This is stated plainly per the task instruction.

## Phase log

- Phase A (checker): DONE. New node parsing + narrowed §3 validation + R1 +
  R2 + must-reject battery. Full existing suite after the change:
  **207 passed / 1 failed / 37 ignored** — the single failure is the
  documented pre-existing parallel-run flake
  (`cap_resume_discards_on_binding_or_cap_mismatch` vs env-mutating warmth
  tests); it passes in isolation (1/1) and in the single-threaded rerun
  (below). Zero regressions attributable to this change.
- Phase B (solver emission): DONE. Selector + closure + finalize +
  self-verify + clean-rerun fail-safe. Property "flag-on never decides
  fewer positions" is enforced structurally (rerun on any non-verified
  group2 attempt) and tested on the fixture
  (`flag_on_and_off_agree_on_fixture_verdicts`,
  `flag_off_solver_is_deterministically_identical_across_runs`).
- Phase C (plumbing + harness): DONE (see file map). Harness smoke test:
  canary registered, adapter accepts/propagates `group2`, defaults off.
- Phase D (full battery): DONE — parallel 208/1(known flake)/37, isolated
  flake pass, single-threaded 209/0/37 (see Test results).

## Test results

- `cargo test --features python tss_verify_group2`: 11/11 passed
  (SHA-256 golden vectors; CertNode size/align frozen vs a legacy mirror
  enum; selector fires + reduced fanout + strict verify + legacy-policy
  reject; flag-on/off verdict agreement; flag-off run-to-run determinism;
  mutation battery — schema, authority commit/path/sha bytes, claimed B,
  build horizon, both digests, omitted-required-reply (per edge, with
  orphan pruning so coverage is the only rejection reason), noncanonical
  edge order, horizon understatement, root-binding tamper, claimant flip,
  legacy-zone mixing splice, implicit-dispatch mixing splice; R1 gate
  fixture; R2 opening-root fixture; legacy dual-policy equivalence; 12
  D6-image acceptance).
- FINAL full battery (`cargo test --features python`, MSVC, Git Bash,
  CARGO_TARGET_DIR=E:/cargo-targets/g2-cert):
  - parallel: 208 passed / 1 failed / 37 ignored — the one failure is the
    documented pre-existing parallel-run flake
    (`cap_resume_discards_on_binding_or_cap_mismatch`); it passes 1/1 in
    isolation;
  - single-threaded rerun (`-- --test-threads=1`): **209 passed / 0
    failed / 37 ignored** (209 includes the group2 manifest-echo test
    added after the first parallel pass). Both results reported per the
    task's flake protocol.
- Measured selector effect on the constructed fixture: 19 explicit root
  edges vs 886 full legal moves.

## Hostile-review §4 constructions — status

- C2 (FHW-O1 omission): gate-free analogue ENCODED as a must-reject test —
  deleting any explicit edge of the accepted reduced node (with orphan
  pruning) rejects on `required ⊆ explicit`.
- C1 (q=5 replay), C3 (root-gate WC row deletion), C4 (cross-branch role
  splice): NOT constructible at test scale inside the accepted class — all
  three require an accepted `FhwGateV1`, and this implementation rejects
  every gate-bearing certificate wholesale (narrowing #1). They are
  represented indirectly: any such certificate rejects (pinned by the R1
  test's two gate variants), which is conservative but does not exercise
  the specific WC/kappa/splice logic (unimplemented).

## Honest list of anything incomplete

1. `FhwGateV1` validation (§3.3: H/R/phi reconstruction, FC/RC/WC, the
   nine-row kappa table, gate-local WC demand enumeration, checkpoint
   roles, D22-N) — NOT implemented; gates reject wholesale. The grammar,
   D6 remap, heap accounting, and R1 metadata handling for gates ARE in.
2. Because gates reject, `f_cut==r_full` and `Q_cut==E_full` on the
   accepted class; both sides of the mandated inequalities are computed
   from one derivation (documented in-module) rather than two.
3. Finder/verifier derivation sharing (deviation 3 above) — the digest
   comparison detects drift/tampering, not correlated implementation bugs.
   THE thing the reviewer should scrutinize hardest: the derivation
   helpers in `tss_verify_group2.rs` are load-bearing for both sides.
4. `own_win_now` is a conservative count-based over-approximation (+ the
   shared analyzer as a second rejector), not the design's exact
   legality-aware reconstruction: slightly narrows acceptance, never
   widens.
5. Demand-row propagation micro-rules (deviation 4) are this
   implementation's formalization of pseudocode-granularity spec text.
6. The v3 full-certificate digest / `J_zone`/`I_FHW` measurement keys,
   golden tag vectors, and the §6 promotion battery (P1/P2/C2/F19/H1152-B/
   T300) are measurement/promotion machinery, out of scope here and not
   built.
7. The design's §2.4 "assign preorder IDs and remap RoleKeyV1 IDs" is
   implemented per transform for the derived Merkle records; with gates
   absent no RoleKey ever appears in CERTIFICATE data, so the remap is
   exercised only inside derived-record hashing.
8. No wire codec (per design §2.5 there is none to be compatible with);
   `TSS_CERT_VERSION` telemetry was left at 2 — no v1-class certificate
   ever reaches the trainer telemetry path in this build (default-off
   flag), and the design says only a materialized new-class certificate
   reports v3; wiring that dynamic report is left undone.
9. Group-2 emission on the WIDE profile (native PN closure) — explicitly
   out of scope (design §5: wide PN has no Open/Closed closure state);
   selector lives on the narrow zone path only.
10. The rerun fail-safe means flag-on can cost up to ~2x nodes when a
    group2 attempt fails late; acceptable for a default-off v1, noted for
    the perf lane.
