# PHASE B SPEC — quotient representations: production implementation

Status: **READY FOR EXECUTION** (updated 2026-07-09). Phase A is complete on
`codex/quotient-reps-phase-a` @ `e5ea1c6a` and its `RESULTS_PHASE_A.md`
(qualified GO) was owner-accepted on 2026-07-09 after an independent deep
review (machinery, tests, audit script, and cost model each adversarially
re-verified; zero correctness defects; the review's caveats are folded into
this spec and marked "[review]"). Prerequisite reading: `CONTEXT.md`,
`DERIVATION_QUOTIENT_REPS.md`, `RESULTS_PHASE_A.md`, and
`docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md`.

## 0. Entry conditions

- E1: **SATISFIED.** Phase A acceptance checklist fully green; `reps.py` is
  in-tree on the base branch and its 19-test suite passes on Windows and WSL
  (independently re-run 2026-07-09).
- E2: **SATISFIED.** Owner-approved GO with the three nominated arms in §8
  (two signatures, three (sig, K_attn) configurations).
- E3: **OPEN — gates deployment only, not implementation.** The current
  hexfield_eq_main_1 soak has not reached its read-out decision point.
  Implementation happens on a feature branch (§0.1) and touches nothing the
  live run imports until deploy per §9, so coding, CPU tests, and idle-GPU
  benches may all start now. Launching training arms (§8) and any deploy
  remain blocked on the owner's soak read-out call.

### 0.1 Execution snapshot (facts pinned 2026-07-09)

- **Base branch**: start the Phase-B feature branch from
  `codex/quotient-reps-phase-a` @ `e5ea1c6a` (contains `reps.py`, the five
  reps test suites, and this doc bundle). Never work on the live tree
  (`E:\Hexo-BotTrainer-gumbel`).
- **Live checkpoints** live at
  `E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\checkpoints\epoch_*.pt`
  (WSL `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1/checkpoints/`), NOT
  under the prefit path. The G7-audited immutable reference is
  `epoch_000015.pt`, SHA-256
  `c02c9f460d0adb12ecd0684dc48ab94ca4726f12e9c7a0139f19103f3a61728e`. For the
  D8 logit gate pick the newest `epoch_*.pt` at execution time and record its
  path + SHA-256 in RESULTS_PHASE_B.
- **Arch env** for the live net:
  `scripts/prefit_env/hexfield_eq_arm4_raylayout.env`. Its header comment
  ("ray-layout support blocked") is stale — the values are verified correct
  against the live checkpoint (RESULTS_PHASE_A G7). Fix the comment in
  passing.
- **Feature version**: Phase B is scoped to the **25-plane v1 feature map**;
  `reps.py` hardcodes the v1 input rep (`INPUT_FEATURES=25`,
  `AXIS_PLANE_BASE=11`) and the live checkpoint/shards are v1. [review] The
  concurrent raytap project's feature-v2 (46-plane) work lives on branch
  `claude/raytap-phase-f-975286`; if it merges before Phase B lands, the typed
  stem must be generalized (`reps.input_rep_action`/`typed_stem_weight` gain a
  feature-version gate) — that sequencing decision belongs to the owner and
  must be recorded in RESULTS_PHASE_B. Until then, typed builds assert the
  v1 plane map.
- **CPU test contexts**: production `model.py` transitively imports
  `torch.nn.attention.flex_attention` (which loads Triton on the Windows
  torch build). Use the sentinel-guard pattern from
  `tests/test_hexfield_eq_reps_toynet.py` / RESULTS_PHASE_A D4 (disable every
  backend gate, place a `None` sentinel for the optional module, restore in
  `finally`, assert no `triton` in `sys.modules`).
- **Reference test commands** (both must stay green throughout Phase B):
  Windows —
  `$env:PYTHONPATH='packages/hexfield_eq/python;packages/hexo_engine/python;packages/hexo_utils/python'; python -B -m pytest -p no:cacheprovider tests/test_hexfield_eq_reps_*.py -q`;
  WSL — same file list via
  `source /root/.venvs/hexgt-build/bin/activate` from `/mnt/e/Hexo-BotTrainer-hexgt`.
- **WSL memory discipline**: while the soak is live, at most ONE extra torch
  CPU process in the VM at a time (OOM incident 2026-07-09); prefer running
  CPU tests on the Windows side.

### 0.2 Recommended execution order [review]

1. **D8 gate first** (§2-D8, §6): TypedLinear/TypedConv/state-dict parity +
   live-checkpoint logit reproduction. Cheapest, highest-value protection;
   everything else builds on it.
2. **Tile-efficiency microbench** (§7.1) on the idle GPU — before any
   training spend; may re-shape the C=160 arm.
3. Modules (§4) + new tests and full regression matrix (§6).
4. Serve suite + full-net idle-GPU bench vs the G8 projection (§7.2).
5. Arms (§8) — blocked on E3.

## 1. Objective and non-goals

**Objective**: make the trunk's residual-stream fiber a **mixed permutation
signature** (Phase A's type system) instead of pure regular rep, with
attention internals kept in pure regular rep at a configurable orbit width,
so that C (and hence ~all serve FLOPs and activation bytes) shrinks while
exact full-D6 equivariance is preserved. Target: the projected ≥1.5–2.1×
serve speedup from `RESULTS_PHASE_A.md` at acceptable strength — with
expectations judged at the cost-consistent `alpha = 4/7` calibration, not the
nominal 0.67 (§7, §8). [review]

**Non-goals**: no irrep/Fourier compute; no C6; no change to features,
support, search, shards, or the Rust engine (in particular no feature-v2
integration — see §0.1); no fp8 (separate project); no change to the
passthrough (GROUP_ORDER=1) build's behavior.

## 2. Locked design decisions

These were rehearsed by the Phase-A toy net (derivation §7). Deviations
require an explicit note in the results doc and owner sign-off.

- **D0 — Phase-A frozen conventions (normative; do NOT re-derive from
  `PHASE_A_CPU_PROOF_SPEC.md` §2, which contains a known erroneous
  layout sentence — see RESULTS_PHASE_A D1/D2).**
  Type order `reg, mirror, point, axis, triv`; type blocks contiguous in that
  order. **Within a type block the layout is slot-major, instance-minor:**
  `channel(T, slot s, instance i) = offset_T + s·m_T + i`
  (production-compatible; frozen by
  `test_regular_signature_layout_is_production_slot_major`). Distinguished
  elements: `sigma = g7` (the K-reflection fixing directed `(1,0)`),
  `rot180 = g3` (orientation-preserving, reverses both axial basis vectors).
  Canonical coset lists are DERIVATION §1.2 (`axis` ≡ production's Q/R/QR
  partition). All tables come from `reps.py` — never hand-derived.
- **D1 — types live on the residual stream only.** All attention internals
  (A-block `RelPosAttention`, L-block `RayAttention`, lane
  `RegisterRefresh`) operate in **pure regular rep** at internal width
  `W_attn = 12 · K_ATTN`. Boundaries: q/k/v are `TypedLinear(sig → reg:K_ATTN)`,
  out is `TypedLinear(reg:K_ATTN → sig)`. Everything already derived for
  coset heads transfers verbatim with `C_ORBIT → K_ATTN`:
  `head_dim_A = 4·K_ATTN` (3 heads), `head_dim_L = 2·K_ATTN` (6 heads,
  own/opp on the orbit index ⇒ **K_ATTN even**), `head_perm`/`head_perm6`
  built at width K_ATTN, bias tables and `joint_of_row_head` unchanged.
- **D2 — MLPs are typed.** `fc1: TypedLinear(sig → MLP_RATIO×sig)`
  (per-type multiplicity scaling), GELU pointwise (legal on permutation
  reps), `fc2` back. No rep conversion inside the MLP.
- **D3 — tokens are per-instance invariant content.** Token param shape
  `(NUM_TOKENS, n_instances(sig))`, expanded slot-constant within each type
  instance (generalizes today's `(NUM_TOKENS, C_ORBIT)` tiled ×12). fp32
  token-stream carry (D-S27) unchanged.
- **D4 — norms and LayerScale tie per instance.** Norm statistics over the
  full mixed fiber (permutation-invariant), affine `gamma/beta
  (n_instances,)` expanded to `(C,)` by a precomputed instance-of-channel
  index. The expanded `(C,)` `.weight`/`.bias` property views MUST remain so
  the fused Triton conv+LN kernel consumes them unchanged. LayerScale:
  `(n_instances,)` expanded the same way.
- **D5 — convs are typed** via the Phase-A conv basis (orbits on
  `(tap, out_slot, in_slot)`); the stem is the typed Reynolds lift from the
  25-plane input rep. Materialized dense weights keep the exact
  `(7, C_in, C_out)` / `(C_out, C_in)` layouts, so every Triton kernel and
  the serve weight cache remain blind to the tie.
- **D6 — heads.** `group_pool` generalizes to per-instance slot-mean
  (`typed_group_pool`), output width `n_instances`. Invariant reads become
  `INV_READ_EXPAND × sig` expansions pooled to
  `INV_READ_EXPAND·n_instances`; policy expansions likewise with
  `POLICY_READ_EXPAND`. Reduction/head `nn.Linear` widths change accordingly
  — this is a new-arch state dict, back-compat with mixed sigs is NOT
  required (see D8 for the pure-reg exception).
- **D7 — serve perm fold is unchanged in kind.** The folds act on the
  regular side of the boundary projections only (out_perm on q/k/v outputs,
  in_perm on out_proj inputs), exactly where they act today.
  `set_serve_perms` moves to `TypedLinear` with identical semantics and the
  identical `not torch.is_grad_enabled()` gate contract.
- **D8 — the pure-regular signature is a bit-compat regression gate.** With
  `sig = reg:16` and `K_ATTN = 16`, the typed build must produce (a) the
  IDENTICAL state-dict key set and parameter shapes as today's build
  (`wb (12, o, i)`; `w_base (7, 12, o, i)`; the same 84 conv blocks — Phase A
  G3 proved the parameterizations coincide, and
  `reps.production_linear_coefficients`/`production_conv_coefficients` are
  the ready-made conversion bridges), and (b) **load the live main_1
  checkpoint (§0.1 path) and reproduce its `forward_policy_value` outputs**
  (fp32 CPU, atol ≤ 1e-5 elementwise; document any residual diff). This gate
  is the single most important protection in Phase B. Run it FIRST (§0.2).

## 3. Configuration plumbing

- New env `HEXFIELD_EQ_TYPE_SIG`, format `"reg:8,mirror:8,axis:8,triv:8"`
  (canonical type order enforced; omitted types = 0; `point` allowed).
  Default: **`reg:<C_ORBIT>`** — i.e. unset reproduces today's build
  exactly (rollback = unset the env).
- New env `HEXFIELD_EQ_ATTN_ORBIT` (K_ATTN), default = regular multiplicity
  of the sig if pure-reg, else required explicit. Validation at import
  (`constants.py`): K_ATTN even when 'L' in layout; `4·K_ATTN ∈
  {16,32,64,128}` for the A fast path (96 → loud warning, slow path);
  computed `C = Σ mult·slots`; warn loudly if `C % 16 != 0` (Triton conv
  fast path falls off — allowed for experiments, never for deploy arms).
- `arch_meta()` gains `type_sig` (canonical string) and `attn_orbit`;
  `infer_net_kwargs_from_state_dict` learns to read them (meta-first;
  fallback inference from parameter shapes for pure-reg only). Loaders
  assert sig match exactly as they assert `support_radius` today.
- GROUP_ORDER must be 12 for any non-trivial sig; GROUP_ORDER=1 passthrough
  ignores the sig env entirely (assert it's unset or pure-reg).
- Typed builds assert the v1 feature map (25 planes) until the stem is
  feature-version-gated (§0.1). [review]

## 4. Module-by-module change list

All in `packages/hexfield_eq/python/hexfield_eq/` unless noted. `reps.py`
(Phase A) is promoted to the production source of tie tables; all tables it
generates are cached at import (`functools.lru_cache`, mirroring
`equivariant.py`).

1. **`reps.py`**: add the K_ATTN-parameterized `head_perm(K)` /
   `head_perm6(K)` builders (`head_perm(K)`/`head_perm_inv(K)` already exist
   — add `head_perm6`, and assert equality with `equivariant.head_perm*` at
   K=C_ORBIT in tests). `instance_of_channel(sig)` and
   `expand_per_instance(vec, sig)` already exist. [review] Add a
   **per-device cache** for the basis label tensors
   (`typed_linear_weight`/`typed_conv_weight` currently call `.to(device)`
   on every materialization — a host→device copy per forward on GPU), and
   dedupe the identical neighbor-gather blocks in
   `TypedConv.forward`/`TypedStem.forward`.
2. **`model.py` — `TypedLinear`** (new class alongside `EquivLinear`):
   free params stored per type-pair basis exactly as Phase A defined them,
   BUT with the D8 constraint: for pure-reg sigs the parameter tensor names
   and shapes must equal `EquivLinear`'s (`wb`, `bias_base`). NOTE: Phase A's
   `reps.TypedLinear` does NOT satisfy this (ParameterDict keyed
   `"reg__from__reg"`, no serve cache) — it is the proof-of-math, not the
   production class. Recommended implementation: production `TypedLinear`
   subsumes `EquivLinear` and `EquivLinear` becomes `TypedLinear` fixed at
   pure-reg (keeping the class name as an alias so external references and
   checkpoints are untouched). `_version`-keyed no-grad dense cache and
   `set_serve_perms` carried over unchanged. Bias: per-instance
   `bias_base (n_instances,)` expanded slot-constant (pure-reg case:
   `(corb_out,)` repeated — identical to today).
3. **`model.py` — `HexNodeConv`**: typed `regular` kind (basis-coefficient
   params, D8 shape-compat for pure-reg) and typed `stem` kind
   (`gen_typed_stem_weight`). Same cache structure.
4. **`model.py` — `GroupAffineNorm`, `LayerScale`, `_make_norm`**:
   per-instance affine per D4. Pure-reg: shapes `(C_ORBIT,)` — unchanged.
5. **`model.py` — `RelPosAttention`, `RayAttention`**: q/k/v/out become
   boundary `TypedLinear`s at width `W_attn`; `self.head_dim` derives from
   `W_attn // heads`; perms from `reps.head_perm(K_ATTN)`. Attention math,
   bias carriers, flex/Triton dispatch: UNCHANGED (they see (B,H,S,d)
   tensors and dense weights only). Residual/LayerScale/masking at sig
   width.
6. **`model.py` — `AttnBlock`, `RayAttnBlock`, `ConvBlock`**: widths only.
7. **`register.py` — `RegisterRefresh`, `TokenRead`**: boundary projections
   per D1; token/gate logic unchanged; `TokenRead` reads
   `TypedLinear(sig → sig)` per token.
8. **`model.py` — `HexfieldNet`**: token param per D3; head plumbing per D6
   (`_inv_read`, `_policy_logits`, `_cell_q_logits`, reduction widths);
   trunk walk unchanged; `arch_meta` per §3. `tokens.repeat(1, GROUP_ORDER)`
   becomes the per-instance expansion.
9. **`inference.py`**: no logic changes expected (it consumes the net's
   public surface). Verify: SERVE_HALF deepcopy, CUDA-graph statics, and the
   evaluator's fp32 value-top handling are width-agnostic. Any place that
   hardcodes `C_ORBIT`/`CHANNELS` semantics must be found by grep and
   generalized.
10. **Rust**: expected zero changes. Mandatory verification step: grep the
    Rust crates for `CHANNELS`, fiber, or C_ORBIT assumptions (the wire
    format is features/raylen/coords only — NUM_FEATURES is unchanged).
    Record the grep in the results doc.

## 5. What must NOT change (regression surface)

- Feature planes, support construction, shard/wire formats, search code.
- The passthrough (GROUP_ORDER=1) build: bit-identical behavior and state
  dicts.
- The pure-reg equivariant build: D8 gate (key set, shapes, live-checkpoint
  logit reproduction).
- All Triton kernels and their dispatch conditions; the sync-free ray index
  build; CUDA-graph capture keys; the serve weight cache and perm-fold gate
  contract; the D-S27 fp32 token carry.
- `state_dict` key-set discipline: sig changes MAY change the key set (new
  arch), but toggles that don't (norms, LayerScale) must keep today's keys
  in the pure-reg case.
- The Phase-A reps test suite (19 tests) stays green untouched.

## 6. Tests (new + regression matrix)

New:
- `tests/test_hexfield_eq_typed_model.py` — mixed-sig net construction,
  fp32 full-net equivariance (pattern of `test_hexfield_eq_equivariance.py`)
  at the two nominated sigs (§8); policy permutes, value invariant.
- `tests/test_hexfield_eq_typed_regression.py` — the D8 gate: pure-reg key
  set + shapes vs a recorded manifest, and live-checkpoint logit
  reproduction (checkpoint path via env/skip-if-absent so CI without the
  runs drive skips cleanly).
- `tests/test_hexfield_eq_typed_serve.py` — serve-path parity at a mixed
  sig: eager fp32 vs SERVE_HALF vs Triton-kernels-on vs CUDA-graphs-on
  (tolerances follow `test_hexfield_eq_serve.py`'s existing model), and the
  perm-fold gate re-run (pattern of `test_hexfield_eq_perm_fold.py`,
  including its magnitude-scaled tolerance model — read its comments).
- `tests/test_hexfield_eq_typed_checkpoint_meta.py` — arch_meta/infer
  round-trip for mixed sigs.

Regression (must pass unmodified, in the arm-4 deploy env AND the default
env): `test_hexfield_eq_equivariance.py`, `test_hexfield_eq_perm_fold.py`,
`test_hexfield_eq_serve.py`, `test_hexfield_eq_triton_ray.py`,
`test_hexfield_eq_ray_block.py`, `test_hexfield_eq_register_lane.py`,
`test_hexfield_eq_smoke.py`, `test_hexfield_eq_checkpoint_meta.py`,
`test_hexfield_eq_rust_parity.py`, `test_hexfield_eq_derivation.py`, plus
the five `test_hexfield_eq_reps_*.py` suites.
Known pre-existing artifacts (do not chase): ray_block equivariance ATOL at
C=192; raylen_parity fixtures at radius 4 (see
`docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md` §5 addendum).

## 7. Serve benchmark protocol

### 7.1 Tile-efficiency microbench — MANDATORY, before any training spend [review]

The G8 model counts ideal FLOPs/bytes and only gates `C % 16`; real
GEMM/Triton kernels quantize to 32/64-wide tiles, and **C=160 (2.5×64) is
the suspected offender** (a 64-wide K-tile pads it to 192, eroding its
projected 1.6×). On the idle GPU (never the soak's), bench:
- raw fp16 dense GEMMs at deploy shape `M ≈ 24k` for
  `K = N ∈ {192, 176, 160, 128, 112, 96}`;
- the fused conv path and attention shapes at `W_attn ∈ {192, 96}`
  (K_attn 16 vs 8).
Record achieved TFLOP/s and effective bandwidth per width; flag any width
whose efficiency drops >10% vs C=192. If C=160 degrades badly, either
re-nominate C=176 (`reg:8,mirror:8,axis:8,triv:8`, nominal 1.355–1.427×) or
accept documented reduced expectations for the C160 arm. Table goes in
RESULTS_PHASE_B.

### 7.2 Full-net bench

Build a mixed-sig net at each nominated sig, random-init, run the serve
evaluator's standing bench flow at deploy shapes (B·Npad ≈ 24k, fp16, full
kernel stack + CUDA graphs); record fwd ms and derived pos/s projection vs
the pure-reg baseline; compare to the G8 model's prediction — **agreement
judged against the `alpha = 4/7` column** (the cost-consistent calibration;
the nominal 0.67 column is throughput-interpolated and optimistic,
especially for the C128/K16 arm). Within ±20% or explain. Record in results.

## 8. Training arms and promotion criteria

Arms (from RESULTS_PHASE_A's decision section; control = pure-reg C=192,
the live bot):

| Arm | Residual signature | C | K_attn | Params eff/stored | Projected speedup nominal / α=4/7 | Role and risk [review] |
|---|---|---:|---:|---:|---:|---|
| B1 | `reg:8,mirror:8,axis:4,triv:4` | 160 | 8 | 0.561M/0.587M | 1.546–1.620× / 1.615× | Conservative: tests whether an 8-instance regular reserve protects the chiral policy signal. Tile risk at C=160 — §7.1 gates this arm's expectations. |
| B2 | `reg:4,mirror:6,point:2,axis:8,triv:8` | 128 | 16 | 0.575M/0.595M | 1.541–1.612× / **1.466×** | Attention-width control (pairs with B3). Softest projection: bandwidth-limited and α-sensitive; likely first closure if the bench under-delivers. |
| B3 | `reg:4,mirror:6,point:2,axis:8,triv:8` | 128 | 8 | 0.518M/0.538M | 2.065–2.133× / 2.068× | The value arm; its projection is the most robust (α-flat). |

Mirror-only and ultra-narrow extremes are REJECTED (causal G7 evidence:
ablating the ~3–5% mirror-odd energy flips 61% of top-1 moves at depth 0).
Every arm keeps a regular reserve; `point:2` in B2/B3 is a deliberate hedge
(E_point ≥ E_mirror through much of the live trunk at identical fixed-C
cost). C156-class sigs are rejected for deployment misalignment.

- Each arm: BC-prefit with the existing `prefit.py` protocol on the same
  shard set → short ladder soak under `eq_ladder_runner`-style supervision.
  Launch is blocked on E3.
- **Eval**: the centralized pentanomial eval driver vs the standing anchors
  (Strix, SealBot, eq self-anchors). Promotion requires: strength within
  the owner-set Elo tolerance of the control at matched wall-clock (NOT
  matched steps — the point is strength per GPU-hour), and **measured**
  serve speedup ≥ 1.4× or the arm is closed. An arm whose measured speedup
  misses its α=4/7 projection by >20% must be explained (tile efficiency,
  occupancy, launch overhead) before further training spend. [review]
- Every arm's config + env is a new launch toml + env file under
  `scripts/prefit_env/`; never edit the live main_1 files.

## 9. Deployment and rollback

- All work on a feature branch; the live soak's tree/env untouched until an
  arm wins and E3 is satisfied.
- Rollback is trivial by construction: `HEXFIELD_EQ_TYPE_SIG` unset ⇒
  pure-reg ⇒ D8-verified identical behavior. No migration of existing
  checkpoints is needed (pure-reg loads them as-is).
- Deploy of a winning arm follows the standing checklist discipline
  (`docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md`): stop → gates in the deploy
  env → idle bench → relaunch → live marginal-rate verification → checklist
  addendum + memory note.

## 10. Deliverables

- Code per §4, tests per §6.
- `docs/quotient_reps/RESULTS_PHASE_B.md`: D8 gate evidence (checkpoint
  path + SHA-256), §7.1 microbench table, full test matrix results, §7.2
  bench table vs both G8 alpha columns, arm configs, the feature-version
  sequencing decision (§0.1), and (after training) the strength/throughput
  read-out with a promote/close decision per arm.
- Updates: `docs/DERIVATION_QUOTIENT_REPS.md` §7 finalized to as-built;
  CONTEXT.md appended with the typed-fiber section (as-built); the stale
  header comment in `scripts/prefit_env/hexfield_eq_arm4_raylayout.env`
  fixed.

## 11. Acceptance checklist

- [ ] D8 gate: pure-reg key set/shape manifest match + live-checkpoint logit
      reproduction (checkpoint path + SHA-256 recorded).
- [ ] §7.1 tile-efficiency microbench recorded; C=160 arm expectations
      confirmed or re-shaped before training spend.
- [ ] Full-net fp32 equivariance at the 2 nominated sigs (all 12 g, real
      features).
- [ ] Entire regression matrix green in both envs, including the 19 Phase-A
      reps tests; typed serve suite green (half, kernels, graphs, perm fold)
      at a mixed sig.
- [ ] Rust-assumption grep recorded; zero Rust changes (or escalated).
- [ ] Idle-GPU bench vs G8 prediction within ±20% **at α=4/7** (or
      explained).
- [ ] Passthrough build untouched (its tests + key-set check).
- [ ] Feature-version scope asserted (v1) or the stem version-gate landed and
      tested; sequencing decision recorded.
- [ ] Arms launched per §8 with their own configs (E3 satisfied first); live
      main_1 files untouched.
- [ ] RESULTS_PHASE_B.md complete with promote/close recommendation.
