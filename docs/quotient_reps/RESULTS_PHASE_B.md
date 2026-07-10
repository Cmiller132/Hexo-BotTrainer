# Phase B results — production quotient representations

- Date: 2026-07-09
- Integration branch: `quotient-phase-b`
- Base: `codex/quotient-reps-phase-a` at `e5ea1c6a`
- Phase-F merge: `claude/raytap-phase-f-975286` via `355ef35f`
- Production implementation: `07ed5658`
- Phase-B gates: `aa1eb470`

## 1. Scope and status

Phase B's production representation plumbing, typed modules, regression gates,
and CPU verification are implemented. The default remains the old pure-regular
architecture when `HEXFIELD_EQ_TYPE_SIG` is unset. Feature versions 1 and 2 are
both supported by the typed stem (`25 = 13*triv + 4*axis` and
`46 = 16*triv + 10*axis`, respectively).

No training arm, launch configuration, deployment action, service restart, or
live-environment edit was performed. Sections 8 and 9 remain behind the owner
stop-gates. The GPU benchmark and CUDA-only serve gate were not run because the
only GPU could not be established as safely idle; details are in §5.

## 2. As-built production surface

- `HEXFIELD_EQ_TYPE_SIG` is parsed in canonical order
  `reg,mirror,point,axis,triv`. An explicit mixed signature computes the
  residual width; an explicit disagreeing `HEXFIELD_EQ_CHANNELS` is rejected.
- `HEXFIELD_EQ_ATTN_ORBIT` selects the regular attention interior. It defaults
  to the regular multiplicity for pure-regular builds and is mandatory for a
  mixed residual signature.
- `TypedLinear`, typed `HexNodeConv`, per-instance norm/LayerScale, attention
  boundary projections, typed MLPs, per-instance tokens, register refresh/read,
  and typed invariant head pooling implement locked decisions D1–D7.
- The pure-regular specializations retain the old parameter names, shapes,
  initialization order, generators, serve-fold behavior, and no-grad caches.
  This is exercised by D8 rather than inferred from source similarity.
- `arch_meta()` records canonical `type_sig` and integer `attn_orbit`;
  meta-first inference reconstructs them. A mixed checkpoint without metadata
  is rejected, partial quotient metadata is rejected, while old pure-regular
  checkpoints retain shape fallback. Foreign pure-regular reconstruction uses
  the signature-parameterized stem unless it is the exact process-global D8
  specialization.
- Phase-A representation tables are the production source for mixed maps.
  Basis labels and channel-instance indices have per-device caches;
  `head_perm6(K)` is parameterized by the regular attention multiplicity.
- `inference.py` and the Rust crates have zero Phase-B changes.

The three nominated random-init builds have the expected stored parameter
counts:

| Arm | Residual signature | C | K_attn | Stored parameters |
|---|---|---:|---:|---:|
| B1 | `reg:8,mirror:8,axis:4,triv:4` | 160 | 8 | 586,522 |
| B2 | `reg:4,mirror:6,point:2,axis:8,triv:8` | 128 | 16 | 595,298 |
| B3 | `reg:4,mirror:6,point:2,axis:8,triv:8` | 128 | 8 | 537,618 |

## 3. D8 — pure-regular compatibility gate

The checkpoint was selected as the newest `epoch_*.pt` at execution time:

- Path: `E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\checkpoints\epoch_000020.pt`
- Size: 8,910,678 bytes
- Last write: 2026-07-10 01:24:36 UTC
- SHA-256: `f930b8f3fbc9ffb7dbdb0fbbe35407bad81a52cac291bff3aae72578f8143f2e`
- Phase-A reference tree: `E:\Hexo-BotTrainer-hexgt` at `e5ea1c6a`

The gate scrubbed inherited `HEXFIELD*` variables, then built the checkpoint
architecture with feature version 1, `HEXFIELD_EQ_RAYTAP=0`, and both
`HEXFIELD_EQ_TYPE_SIG` and `HEXFIELD_EQ_ATTN_ORBIT` genuinely unset. CUDA was
hidden. Results:

| Evidence | Result |
|---|---|
| Arm-4 pure-regular state manifest | 321 tensors; exact match |
| Canonical manifest SHA-256 | `1ce1cd95e41eaba486974ec4d8c69758cd7117307aeec6dd1ee4dcb17be8be28` |
| Named-parameter order | 320 entries; exact Phase-A match |
| Policy logits | max absolute Phase-A/Phase-B error `0.0` |
| Value logits | max absolute Phase-A/Phase-B error `0.0` |
| Moves-left logits | max absolute Phase-A/Phase-B error `0.0` |
| Required tolerance | `atol=1e-5`, `rtol=0` |

The passthrough manifest is also frozen and matched: 179 tensors, canonical
SHA-256 `570627a31644c61b769eb1b0de37c46f19e7ac2b3265394188af489c308004d9`.

Command (PowerShell, from the integration worktree):

```powershell
$env:PYTHONPATH='packages/hexfield_eq/python;packages/hexo_engine/python;packages/hexo_utils/python'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HEXFIELD_EQ_D8_CHECKPOINT='E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\checkpoints\epoch_000020.pt'
$env:HEXFIELD_EQ_D8_REFERENCE_ROOT='E:\Hexo-BotTrainer-hexgt'
python -B -m pytest -p no:cacheprovider tests/test_hexfield_eq_typed_regression.py -q
```

## 4. Test matrix

All subprocess tests scrub architecture variables before import. CPU runs hide
CUDA explicitly. The four new suites together, with the live D8 variables set,
produced after the final foreign-stem/config fixes:

```text
19 passed, 1 skipped in 45.75s
```

The one skip is the CUDA-only half/kernels/graphs test. The CPU mixed serve test
does cover eager versus no-grad folded weights, magnitude-scaled fold tolerance,
cache invalidation for every mixed coefficient tensor, and evaluator parity.

Final aggregate counts:

| Environment | Result | Notes |
|---|---|---|
| Default (`FEATURE_VERSION=1`, new arch vars unset) | 101 passed, 40 skipped, 1 failed in 191.19s | 18 runnable files; sole failure is the pre-existing Windows Flex/Dynamo artifact; smoke cannot collect before its skip without Linux Rust |
| Arm-4 deploy env | 86 passed, 46 skipped, 16 failed | 18 runnable files: 86/40/16 in 83.09s, plus 6 smoke skips; failures reproduce integration-base test/env assumptions described below; live D8 is 4/4 passed |
| Phase-A five `test_hexfield_eq_reps_*.py` suites | 19 passed | Re-run after production changes |
| Combined Phase-A + Phase-F WSL baseline | 35 passed, 2 failed | Both failures are the named pre-existing radius/off-legal raylen fixture artifact |

The aggregate used this exact file list:

```powershell
$matrix = @(
  'tests/test_hexfield_eq_typed_model.py',
  'tests/test_hexfield_eq_typed_regression.py',
  'tests/test_hexfield_eq_typed_serve.py',
  'tests/test_hexfield_eq_typed_checkpoint_meta.py',
  'tests/test_hexfield_eq_equivariance.py',
  'tests/test_hexfield_eq_perm_fold.py',
  'tests/test_hexfield_eq_serve.py',
  'tests/test_hexfield_eq_triton_ray.py',
  'tests/test_hexfield_eq_ray_block.py',
  'tests/test_hexfield_eq_register_lane.py',
  'tests/test_hexfield_eq_checkpoint_meta.py',
  'tests/test_hexfield_eq_rust_parity.py',
  'tests/test_hexfield_eq_derivation.py',
  'tests/test_hexfield_eq_reps_group.py',
  'tests/test_hexfield_eq_reps_homdims.py',
  'tests/test_hexfield_eq_reps_parity.py',
  'tests/test_hexfield_eq_reps_typed_layers.py',
  'tests/test_hexfield_eq_reps_toynet.py'
)
python -B -m pytest -p no:cacheprovider $matrix -q -ra --tb=short
```

Before the default command, all inherited `HEXFIELD*` variables were removed.
Before the arm-4 command, every non-comment `KEY=VALUE` line from
`scripts/prefit_env/hexfield_eq_arm4_raylayout.env` was loaded into the process;
the D8 checkpoint/reference variables were then set. Both commands set
`CUDA_VISIBLE_DEVICES=-1`, disabled optional backend gates, and used:

```powershell
$env:PYTHONPATH='packages/hexfield_eq/python;packages/hexo_engine/python;packages/hexo_utils/python;packages/hexo_train/python;packages/hexo_runner/python'
```

Focused production checks additionally established:

- all-12 full-network fp32 policy equivariance and value invariance for B1 and
  B2 on transformed oracle facts;
- the B3 finite forward smoke;
- feature-v2 all-12 mixed full-network equivariance;
- exact pure-regular v1/v2 input-action, stem, and head-permutation parity;
- mixed signature metadata round-trip for all three nominees under both feature
  versions, strict mismatch rejection, legacy pure-regular fallback, and
  mixed-no-meta rejection;
- arm-4 mixed construction including L blocks and the register lane.
- Phase-F's feature-v2 suite on Windows: 4 passed, 1 skipped, and 3 corpus
  cases blocked by the unavailable Windows `hexo_engine` Rust bridge, exactly
  as at integration baseline; the corresponding WSL baseline was green.

A post-implementation WSL aggregate was not launched: the concurrent Phase-R
prefit was still running at ~117% CPU with ten data workers and owned the only
GPU. Starting another WSL Torch process would violate the resource-isolation
constraint. The Rust/feature WSL evidence is therefore the clean integration
baseline plus the mandatory zero-Rust-diff audit; all Phase-B Python/model
changes were exercised by the final Windows CPU runs above.

Two repository test-command facts required code-grounded handling:

1. The documented three-root `PYTHONPATH` is insufficient for
   `test_hexfield_eq_smoke.py`: `plugin.py:9` imports `hexo_train`, and
   `selfplay.py:31` imports `hexo_runner`. Windows aggregation therefore adds
   `packages/hexo_train/python` and `packages/hexo_runner/python`.
2. Windows cannot import the Linux Rust extensions. Rust-backed tests use WSL;
   CPU-only Python/model suites use Windows as allowed by `CONTEXT.md` §9. For
   the arm-4 count, smoke was collected with inert import-only Rust sentinels;
   all six tests then took their `GROUP_ORDER=12` skip before any codec call.

The default matrix's only failure is
`test_hexfield_eq_ray_block.py::test_materialized_matches_flex`: PyTorch
2.10/Windows Python 3.14 cannot specialize the data-dependent Flex score-mod
index. The identical node fails in the untouched Phase-A reference at
`model.py:379`; it is not introduced by Phase B.

The arm-4 environment exposes contradictions between the unmodified regression
tests and the production code they target:

- ten tests in `test_hexfield_eq_equivariance.py` / `test_hexfield_eq_register_lane.py`
  inherit the arm-4 L layout with blockers but do not pass `raylen`; production
  correctly rejects that call at `model.py:2304-2308`;
- one register-lane test assumes the import-global lane default is off, although
  arm-4 sets it on; another C=192 counting-linearity tolerance also fails in the
  untouched Phase-A tree;
- `test_hexfield_eq_register_lane.py:60` defines lane prefixes without
  `registers_l`/`tok_reads_l`, while unchanged production classification at
  `trainer.py:300-301` correctly includes those L-block lane parameters;
- `test_hexfield_eq_reps_toynet.py:400` hard-codes the default C=96/K=8 build;
- the named C=192 L-block absolute-tolerance artifact and the same Windows Flex
  artifact account for the remaining two failures.

Representative arm-4 failures (toggle-off identity, counting linearity, and
lane grad-group classification) were rerun against the untouched Phase-A tree
and failed identically. Existing regression files were not modified, per §6.
Thus the requested literal “unmodified and green under arm-4” matrix is not
satisfiable by the current code/test contracts; the production-specific Phase-B
suites and D8 are green under scrubbed, explicit subprocess environments.

The two named artifacts were not chased: the C=192 ray-block absolute-tolerance
edge and the radius-4/off-legal raylen fixtures. The separately demonstrated
Windows Flex and arm-4 test-assumption artifacts were likewise not “fixed” in
production code.

## 5. Serve benchmark and CUDA gate

### 5.1 Tile-efficiency microbenchmark

The prepared operator sequence and paste-ready benchmark commands are in
[`GPU_WINDOW_RUNBOOK.md`](GPU_WINDOW_RUNBOOK.md). The entries below remain
pending until that safety-gated idle window is run.

Pending. This host exposes one NVIDIA GeForce RTX 4070 Ti. At the final safety
check it reported 100% utilization and 8,918 MiB / 12,282 MiB allocated. The
concurrent Phase-R prefit and its worker pool were active. No GEMM, fused-conv,
attention, or other Phase-B GPU workload was launched.

| Dense width | fp16 GEMM TFLOP/s | Effective bandwidth | Efficiency vs C=192 |
|---:|---:|---:|---:|
| 192 | pending | pending | baseline pending |
| 176 | pending | pending | pending |
| 160 | pending | pending | pending |
| 128 | pending | pending | pending |
| 112 | pending | pending | pending |
| 96 | pending | pending | pending |

The fused conv shapes and attention widths 192/96 are pending for the same
reason. Consequently the C=160 tile-risk decision remains open; no arm was
re-nominated and no training spend was authorized.

### 5.2 Full-network serve benchmark versus G8

Run the prepared full-stack evaluator benchmark and CUDA typed-serve gate via
[`GPU_WINDOW_RUNBOOK.md`](GPU_WINDOW_RUNBOOK.md); record their outputs here.

| Arm | C | K_attn | G8 nominal-alpha projection | G8 alpha=4/7 projection | Measured speedup | Assessment |
|---|---:|---:|---:|---:|---:|---|
| B1 | 160 | 8 | 1.546–1.620x | 1.615x | pending | idle GPU unavailable |
| B2 | 128 | 16 | 1.541–1.612x | 1.466x | pending | idle GPU unavailable |
| B3 | 128 | 8 | 2.065–2.133x | 2.068x | pending | idle GPU unavailable |

The ±20% comparison cannot be called until the orchestrator provides a verified
idle window. The CUDA portion of `test_hexfield_eq_typed_serve.py`—SERVE_HALF,
all fused kernels, and CUDA graphs—is pending with the benchmark rather than
risking contention with the live soak.

## 6. Rust and inference audits

The Phase-B diff under `packages/hexfield_eq/rust/` is empty. The mandatory grep
found no `CHANNELS`, `C_ORBIT`, or fiber-width assumption. Feature width is
versioned in Rust via `NUM_FEATURES_V1`, `NUM_FEATURES_V2`, and
`num_features(feature_version)` across constants/features/lib/payload/replay/
serve. This is the D-Δ1 replacement for the obsolete single-`NUM_FEATURES`
check in the Phase-B spec.

`inference.py` contains no residual-fiber-width assumption; its feature use is
through the active `NUM_FEATURES`. It has no Phase-B diff.

## 7. Ray-tap Phase-R merge note

No ray-tap mechanism was implemented and the worktree was not rebased onto the
Phase-R branch. `HexNodeConv` keeps a narrow materialization/cache seam so the
later merge can generalize ray-tap alpha from pure regular `(5, C_ORBIT)` to
`(5, n_instances)` and expand with `instance_of_channel`.

All nominated residual signatures have even multiplicity for every participating
type, so Phase R can split each type's instances evenly between own/opp sides:

| Signature | n_instances | Per-side split by type | Future alpha shape |
|---|---:|---|---|
| B1 | 24 | reg 4/4, mirror 4/4, axis 2/2, triv 2/2 | `(5, 24)` |
| B2/B3 | 28 | reg 2/2, mirror 3/3, point 1/1, axis 4/4, triv 4/4 | `(5, 28)` |
| Pure regular live build | 16 | regular 8/8 | `(5, 16)` |

Integration with Phase R is a later orchestrator-owned step. Expected textual
conflict risk is concentrated in `HexNodeConv` and `constants.py`; semantic risk
also includes preserving the typed feature-v1/v2 stem and the per-instance alpha
expansion. `inference.py` is intentionally untouched here.

## 8. Spec/code discrepancies and locked-decision audit

The implementation follows the code where documents predate landed work:

- Phase-B §0.1 names `E:\Hexo-BotTrainer-gumbel` as the protected live tree,
  while the running soak imports the main `E:\Hexo-BotTrainer-hexgt` checkout.
  The work order's live-tree statement was followed: the main checkout stayed
  on `codex/quotient-reps-phase-a` with its pre-existing dirty files untouched.
- Phase-B §1/§4.10 and `CONTEXT.md` described a fixed 25-plane input. Landed
  Phase-F code in `constants.py:38-131` makes the feature map import-time
  versioned (25 or 46). The typed stem therefore supports both versions.
- `SPEC_RAYTAP_CONV.md` says Phase F records `feature_version` in checkpoint
  metadata, but the merged Phase-F `model.py` did not do so. Phase B preserves
  its existing `feature_width` contract and adds only `type_sig`/`attn_orbit`;
  the Phase-R-owned metadata integration was not anticipated here.
- Phase-B names the stem helper `gen_typed_stem_weight`; the accepted Phase-A
  API is `reps.typed_stem_weight`, which production uses.
- Phase-B says “loaders” assert metadata as they do support radius. The actual
  support-radius checks are in `checkpoints.py` and `prefit.py`, so quotient
  checks were added at those same two load boundaries; `eval_arena.py` has no
  parallel support-radius assertion to generalize.
- The documents do not specify conflicts among explicit `TYPE_SIG` and legacy
  `CHANNELS`/`C_ORBIT`. Production treats the signature as authoritative:
  disagreeing explicit `CHANNELS` is rejected; `C_ORBIT` remains a compatibility
  value for mixed signatures, while a conflicting pure-regular `C_ORBIT` is
  rejected because the historical stem generator is parameterized by it.
- Phase-B §10 asks to fix the stale comment in
  `scripts/prefit_env/hexfield_eq_arm4_raylayout.env`, while the work order
  explicitly forbids editing existing files in `scripts/prefit_env/`. The file
  is deliberately unchanged.
- Phase-B §7.2 refers to a standing full-network hexfield_eq benchmark flow,
  but the repository has no such full-net benchmark harness. This did not
  trigger a new harness because the only GPU was actively owned; the later
  orchestrator-run benchmark must either nominate an existing external driver
  or add one in that resource window.
- The test command's transitive `hexo_train`/`hexo_runner` path requirements and
  Windows Rust-extension limitation are recorded in §4.
- Phase-B §6's literal arm-4 requirement disagrees with unmodified tests that
  inherit an L/blocker layout but omit required `raylen`, assume the lane env is
  off, or hard-code the default C=96/K=8 architecture. The failures and
  Phase-A reproductions are recorded in §4; those regression files remain
  unmodified as required.

No locked decision D1–D8 was intentionally changed. D-Δ4 is honored: L-block
machinery is retained. The unexecuted GPU gates are acceptance items pending a
safe resource window, not silent architectural deviations.

## 9. Training arms — deferred

**Deferred pending owner E3 call and ray-tap wave-1 read-out.**

No §8 arm launch config was created and no training GPU-hours were consumed.
The later arms must be built on the orchestrator-selected ray-tap architecture.

## 10. Deployment — deferred

No §9 deployment, service operation, merge to `main_9-fastrow-strip`/`main`, or
live-run environment change was performed.

## 11. Post-review fix pass (2026-07-10)

- **Dashboard metadata seeding:** `debug_infer.py` now seeds canonical quotient
  signature, attention orbit, and feature version before the first eq import;
  it validates the import-frozen signature/orbit/feature width while retaining
  the exact legacy pure-regular checks. The new
  `test_hexo_frontend_eq_meta_env.py` covers B1, feature v2, restart rejection,
  and the legacy env-key set.
- **Implicit pure-regular rollback:** `constants.py` rejects a stray non-default
  `HEXFIELD_EQ_ATTN_ORBIT` unless `HEXFIELD_EQ_TYPE_SIG` is explicit. The typed
  model suite covers rejection and the harmless default-value no-op.
- **Fresh initialization parity:** `test_hexfield_eq_typed_regression.py` now
  constructs the scrubbed default net under both the Phase-A reference and this
  tree at `manual_seed(0)`, then checks state-dict key order and every tensor
  bitwise.
- **Uniform quotient metadata contract:**
  `infer_net_kwargs_from_state_dict` rejects either partial metadata form before
  foreign reconstruction; the typed checkpoint suite covers both missing-key
  directions. `eval_arena.py` calls this function with checkpoint metadata at
  line 241 before constructing and strict-loading the net, so its foreign-load
  path is covered without a redundant arena assertion.
- **Visible warm-start drops:** `checkpoints.py` emits one summary warning with
  missing/unexpected/shape-mismatch counts and examples, plus a signature hint
  when typed weights drop. The typed checkpoint suite proves mismatch warning
  and exact-match silence.
- **B3 equivariance:** the literal B3 v1 signature with K=8 now runs the same
  all-12 full-network policy-equivariance/value-invariance gate as B1/B2 in
  `test_hexfield_eq_typed_model.py`.
- **Moves-left serve tolerance:** `test_hexfield_eq_typed_serve.py` uses a
  near-exact tolerance grounded in identical f16-rounded inputs instead of the
  former decode-scale `atol=1.0`. The tightened `atol=1e-4` was confirmed by
  the executed CPU run below.
- **GPU-window preparation:** `bench_quotient_tile.py` provides the dense,
  production fused-conv, and fused-attention median CUDA-event sweep;
  `bench_quotient_serve.py` reuses `HexfieldEvaluator` with the standing
  half/Rust-pack/copy-stream/kernel/CUDA-graph profile for baseline and B1/B2/B3.
  Both require explicit `--allow-gpu`, provide `--cpu-smoke`, and are sequenced
  by [`GPU_WINDOW_RUNBOOK.md`](GPU_WINDOW_RUNBOOK.md).
- **CPU verification (executed 2026-07-10, reviewer run):** the implementing
  agent's sandbox could not see the user-site packages
  (`%APPDATA%\Python\Python314\site-packages`, where torch/pytest live), so
  its own runtime verification was limited to dependency-free syntax
  compilation and AST smokes. The reviewer then executed the full verification
  unsandboxed. One test-harness defect was found and fixed in review: the new
  frontend child asserted that the worker namespace import pulls in no Triton
  modules, but the dashboard namespace deliberately imports
  `hexfield_eq.inference`, which loads Triton whenever it is installed — the
  over-strict assertion was removed. Final executed evidence, all CPU
  (`CUDA_VISIBLE_DEVICES=-1`), scrubbed env, live D8 variables set: four typed
  suites + `test_hexo_frontend_eq_meta_env.py` = **26 passed, 1 skipped in
  61.25s** (sole skip is the CUDA-only typed serve gate; baseline was 19/1 —
  seven new tests, zero regressions); regression spot-check (`equivariance`,
  `perm_fold`, `checkpoint_meta`) = **20 passed, 3 CUDA-only skips**,
  identical to baseline; both benchmark `--cpu-smoke` runs exited 0 (the serve
  smoke constructs baseline and B1/B2/B3 end to end in tiny CPU eager mode).
