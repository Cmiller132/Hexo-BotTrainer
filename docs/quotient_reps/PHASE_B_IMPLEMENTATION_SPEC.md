# PHASE B SPEC — quotient representations: production implementation

Status: BLOCKED ON PHASE A ACCEPTANCE. Do not start until the owner has
signed off on `RESULTS_PHASE_A.md` (GO decision + nominated signatures).
Prerequisite reading: `CONTEXT.md`, `DERIVATION_QUOTIENT_REPS.md` (Phase A
output), `RESULTS_PHASE_A.md`, and `docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md`.

## 0. Entry conditions (all must hold)

- E1: Phase A acceptance checklist fully green; `reps.py` is in-tree and its
  test suites pass.
- E2: Owner-approved GO with 2–3 nominated signatures (from the G7 audit ×
  G8 ranking). These become the training arms in §8.
- E3: The current hexfield_eq_main_1 soak has reached its read-out decision
  point (owner's call). Phase B code changes touch files the live run
  imports, so implementation happens on a branch and deploys only per §9.

## 1. Objective and non-goals

**Objective**: make the trunk's residual-stream fiber a **mixed permutation
signature** (Phase A's type system) instead of pure regular rep, with
attention internals kept in pure regular rep at a configurable orbit width,
so that C (and hence ~all serve FLOPs and activation bytes) shrinks while
exact full-D6 equivariance is preserved. Target: the projected ≥1.5–2×
serve speedup from `RESULTS_PHASE_A.md` at acceptable strength.

**Non-goals**: no irrep/Fourier compute; no C6; no change to features,
support, search, shards, or the Rust engine; no fp8 (separate project); no
change to the passthrough (GROUP_ORDER=1) build's behavior.

## 2. Locked design decisions

These were rehearsed by the Phase-A toy net (derivation §7). Deviations
require an explicit note in the results doc and owner sign-off.

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
  G3 proved the parameterizations coincide), and (b) **load the live main_1
  checkpoint and reproduce its `forward_policy_value` outputs** (fp32 CPU,
  atol ≤ 1e-5 elementwise; document any residual diff). This gate is the
  single most important protection in Phase B.

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

## 4. Module-by-module change list

All in `packages/hexfield_eq/python/hexfield_eq/` unless noted. `reps.py`
(Phase A) is promoted to the production source of tie tables; all tables it
generates are cached at import (`functools.lru_cache`, mirroring
`equivariant.py`).

1. **`reps.py`**: add the K_ATTN-parameterized `head_perm(K)` /
   `head_perm6(K)` builders (generalizing `equivariant.head_perm*`, which
   are the K=C_ORBIT case — assert equality there in tests). Add
   `instance_of_channel(sig)`, `expand_per_instance(vec, sig)` helpers.
2. **`model.py` — `TypedLinear`** (new class alongside `EquivLinear`):
   free params stored per type-pair basis exactly as Phase A defined them,
   BUT with the D8 constraint: for pure-reg sigs the parameter tensor names
   and shapes must equal `EquivLinear`'s (`wb`, `bias_base`). Recommended
   implementation: `TypedLinear` subsumes `EquivLinear` and `EquivLinear`
   becomes `TypedLinear` fixed at pure-reg (keeping the class name as an
   alias so external references and checkpoints are untouched). `_version`-
   keyed no-grad dense cache and `set_serve_perms` carried over unchanged.
   Bias: per-instance `bias_base (n_instances,)` expanded slot-constant
   (pure-reg case: `(corb_out,)` repeated — identical to today).
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

## 6. Tests (new + regression matrix)

New:
- `tests/test_hexfield_eq_typed_model.py` — mixed-sig net construction,
  fp32 full-net equivariance (pattern of `test_hexfield_eq_equivariance.py`)
  at 2 nominated sigs; policy permutes, value invariant.
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
`test_hexfield_eq_rust_parity.py`, `test_hexfield_eq_derivation.py`.
Known pre-existing artifacts (do not chase): ray_block equivariance ATOL at
C=192; raylen_parity fixtures at radius 4 (see
`docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md` §5 addendum).

## 7. Serve benchmark protocol

On the idle GPU (never against the live soak): build a mixed-sig net at each
nominated sig, random-init, run the serve evaluator's standing bench flow at
deploy shapes (B·Npad ≈ 24k, fp16, full kernel stack + CUDA graphs); record
fwd ms and derived pos/s projection vs the pure-reg baseline; compare to the
G8 model's prediction (agreement within ±20% or explain). Record in results.

## 8. Training arms and promotion criteria

- **Arms**: pure-reg C=192 (the live bot, control) + the 2–3 nominated sigs.
  Each arm: BC-prefit with the existing `prefit.py` protocol on the same
  shard set → short ladder soak under `eq_ladder_runner`-style supervision.
- **Eval**: the centralized pentanomial eval driver vs the standing anchors
  (Strix, SealBot, eq self-anchors). Promotion requires: strength within
  the owner-set Elo tolerance of the control at matched wall-clock (NOT
  matched steps — the point is strength per GPU-hour), and measured serve
  speedup ≥ 1.4× or the arm is closed.
- Every arm's config + env is a new launch toml + env file under
  `scripts/prefit_env/`; never edit the live main_1 files.

## 9. Deployment and rollback

- All work on a feature branch; the live soak's tree/env untouched until an
  arm wins.
- Rollback is trivial by construction: `HEXFIELD_EQ_TYPE_SIG` unset ⇒
  pure-reg ⇒ D8-verified identical behavior. No migration of existing
  checkpoints is needed (pure-reg loads them as-is).
- Deploy of a winning arm follows the standing checklist discipline
  (`docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md`): stop → gates in the deploy
  env → idle bench → relaunch → live marginal-rate verification → checklist
  addendum + memory note.

## 10. Deliverables

- Code per §4, tests per §6.
- `docs/quotient_reps/RESULTS_PHASE_B.md`: D8 gate evidence, full test
  matrix results, bench table vs G8 predictions, arm configs, and (after
  training) the strength/throughput read-out with a promote/close decision
  per arm.
- Updates: `docs/DERIVATION_QUOTIENT_REPS.md` §7 finalized to as-built;
  CONTEXT.md appended with the typed-fiber section (as-built).

## 11. Acceptance checklist

- [ ] D8 gate: pure-reg key set/shape manifest match + live-checkpoint logit
      reproduction.
- [ ] Full-net fp32 equivariance at 2 mixed sigs (all 12 g, real features).
- [ ] Entire regression matrix green in both envs; typed serve suite green
      (half, kernels, graphs, perm fold) at a mixed sig.
- [ ] Rust-assumption grep recorded; zero Rust changes (or escalated).
- [ ] Idle-GPU bench vs G8 prediction within ±20% (or explained).
- [ ] Passthrough build untouched (its tests + key-set check).
- [ ] Arms launched per §8 with their own configs; live main_1 files
      untouched.
- [ ] RESULTS_PHASE_B.md complete with promote/close recommendation.
