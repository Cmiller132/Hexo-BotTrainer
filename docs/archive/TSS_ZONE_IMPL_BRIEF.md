# Implementation brief: TSS zone-upgrade catalog (branch claude/tss-zone-upgrades)

You are implementing `docs/PLAN_TSS_SOLVER_UPGRADES.md` (FINAL, three review
rounds — it is the **normative spec**; this brief only adds execution
mechanics and the async-integration requirement). Formal backing:
`docs/PROOF_TSS_DEFENDER_ZONES.md` (theorem tags), survey
`docs/PLAN_TSS_MOVESET_ZONES.md`, `docs/proof_parts/{DOMINATION,ES_POTENTIAL}.md`.
Existing build docs: `docs/TSS_SOLVER_SPEC.md`, `docs/TSS_SOLVER_OPT_SPEC.md`,
`docs/TSS_SOLVER_PROOF.md`, `docs/TSS_RUNBOOK.md`, `docs/PLAN_TSS_DEEPENING.md`.

## Base state — READ FIRST

This branch is cut from `claude/tss-v2-build` at 05ac7162, which includes
the **fresh async deep-solve pool**. Before writing any code, read these
three commits and the files they touch:

- 179a99a0 `feat(hexfield_eq): async deep-solve pool (tss_solver_async) —
  100% leaf coverage off the critical path`
- a83e314d `docs(tss): runbook — async rung flags, watch items, memory notes`
- 05ac7162 `fix(hexfield_eq): async pool hardening — Codex review round 1`

Solver modules: `packages/hexfield_eq/rust/src/{tss_core,tss_solver,tss_verify,tss_reference}.rs`;
integration `tree.rs`/`search.rs` same crate; λ¹ in
`packages/hexo_models/rust/src/threats_shared.rs`.

## Async-integration requirements (mandatory, cross-cutting)

Every upgrade must work through BOTH the synchronous paths (root guard,
any residual sync leaf path) and the async pool:

1. **Thread-safety**: whatever sharing model the async pool uses for
   solver instances / `SharedProofCache`, your changes must preserve it.
   If the pool has per-worker solvers with isolated caches, keep zone
   state per-worker too; if anything is shared, your additions must be
   `Send`/`Sync`-correct with no new locks on the search critical path.
2. **Determinism**: the solver itself must stay deterministic per
   (position, caps). If the async pool introduces scheduling
   nondeterminism in *which* solves complete, that is the pool's existing
   contract — do not add more. No wall-clock anywhere in solver/verifier.
3. **Result plumbing**: new counters (`horizon_retry`,
   zone-verify-rejection reasons, preflight outcomes) must flow through
   the async pool's result/telemetry channel the same way
   `deep_verify_failed` does today, and appear in whatever stats surface
   the pool exposes.
4. **The single mint is unchanged**: async results must still pass
   `hard_value_from_verified` (`tss_core.rs`). If the pool caches or
   defers results, verification happens before the value is handed to the
   tree, exactly as the hardening commit left it.
5. If the async pool's 100%-leaf-coverage design makes parts of U8
   (subsample-based triggering) obsolete, implement U8's λ¹-informed
   *prioritization* as pool queue ordering / budget weighting instead of
   sampling gates, and say so in the status doc.

## Scope and phase discipline

Implement in the plan's phase order. **One commit per phase minimum**,
conventional messages, each commit leaves the tree green (build + tests).
All new flags default-off (the U6 default-flip is an ops decision — DO NOT
flip it; implement the mask lever + shadow counters only).

- **P0**: U4 ply-clock plumbing (typed-leaf resolution semantics land with
  U2, not here — P0 is the clock threading + `SolveCaps` semantic-horizon
  field + `horizon_retry` counter scaffolding); U10 fixture scaffolding
  (test harness for certificate mutations, G1/G3 position builders ported
  from `scripts/_tss_moveset_zone_experiments.py` — the python file is the
  geometry reference).
- **P1**: U3 staple-by-theorem in `tss_verify.rs` (keep the per-move
  staple behind a debug flag as paired oracle; add the paired-oracle
  differential test).
- **P2**: U2 typed certificate schema + full 12-obligation verifier
  (plan §2 U2 — the obligation list is normative and complete; implement
  every item), then U1 zone generator (monotone closure loop; stale-area
  filter and count≥2 threshold as separate flags, both [H]; D≥6 full-legal
  fallback; Opening exclusion). U1 omissions non-consumable until U2's
  verifier passes the full mutation suite.
- **P3**: U4 cache two-stamp rule with composite aggregation
  (resolution_T = max, zone_build_T = min, final-T recheck in the
  preflight; zone-fragment promotion stays disabled until this passes its
  test); U5 P3 pair canonicalization (search restriction + verifier
  commutation arm with ALL the state-binding conditions in plan §2 U5 +
  its six fixtures).
- **P4** (only if budget remains): U8 as pool prioritization (see async
  note above); U9 ES futility behind the semantic-horizon API. If you
  stop before P4, that is acceptable — stop at a phase boundary.

## Definition of done

- Phases P0–P3 implemented; each phase's gate from plan §3 satisfied where
  it is a test/build gate (perf gates: record numbers in the status doc,
  do not block on them).
- `cargo test` green for the touched crates; the existing python shadow
  suite (`tests/test_hexfield_eq_tss_shadow.py`, CPU env per repo docs)
  green; the new U10 mutation suite green — every listed malformed
  certificate REJECTED, every valid golden certificate ACCEPTED.
- The one-sided reference differential (optimized hard claims vs
  `tss_reference.rs` at matched semantic horizon) run on a random corpus:
  0 one-sided divergences.
- No public API breakage for the Python bindings unless additive.
- Write `docs/TSS_ZONE_IMPL_STATUS.md`: per-upgrade status, flag names +
  defaults, test/bench evidence, any deviations from the plan with
  justification, and open items. Deviations from the *soundness-relevant*
  parts of the plan (U2 obligations, U3 lemma premises, U5 conditions)
  are NOT permitted — if something appears impossible as specified, stop
  and record it as a blocker instead of improvising.

## Hard constraints (restated from the plan; violations are rejections)

1. Values enter only via `hard_value_from_verified` — no second mint.
2. No count-based truncation of defender candidate lists, anywhere.
3. Any verify failure ⇒ Unknown + `deep_verify_failed`; horizon-preflight
   failures are solver-side (`horizon_retry`), never the fatal counter.
4. Opening phase: no zone omissions, no dispatch, no futility.
5. Solver/verifier stay deterministic and wall-clock-free.
