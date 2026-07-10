# Plan: dense 31-tap conv (Design A) — CPU investigation

Date: 2026-07-10. Status: PLAN for an investigative, CPU-only build-out on
branch `raytap-31tap-cpu`. Companion to `docs/RAY_CONV_DESIGN_SPACE.md` §3
(Design A), which this plan implements with one correction (§3 below).
Nothing here touches GPU kernels, serve integration, configs, or the live
`hexfield_eq_main_2` run. The deliverable is a working reference
implementation + evidence, not a production path.

## 1. Decisions locked (from the design review)

- **Full 31-tap** (center + 6 directions × 5 distances), per-(direction,
  distance) tied weight blocks. No intermediate variants (shared per-distance
  mixers, reduced reach): compute is dominated by touching 30 fibers with
  full matrices regardless of tying, and parameters are not the constraint.
- **α is removed** in the dense31 conv, not kept as a parallel path — a
  per-distance diagonal is a strict special case of the full blocks. Trained
  α enters only via warm-start surgery (§3).
- **Hard visibility mask retained bit-identically**: per-(side, direction)
  reach, own-side visibility on the first orbit-channel half, opponent-side
  on the second half — exactly the semantics of `_raytap.py` /
  `SPEC_RAYTAP_CONV.md` §2.6. No learned gating of any kind (Designs B/C are
  explicitly out of scope; their gates can be grafted later since they are
  exact-identity at init).
- **CPU only, investigative.** Reference path + a recompute-in-backward
  autograd Function. Triton/K1-style kernels, fp16 serve folding, CUDA graph
  integration: all out of scope.

## 2. Mechanism specification

### 2.1 Tap set and ordering

Tap index space 0..30, **distance-shell-major**:

    t = 0                 : center (self)
    t = 1 + 6*(k-1) + d   : direction d (0..5, the DIRECTIONS order), distance k (1..5)

Shell k=1 (taps 1..6) coincides exactly with today's 7-tap direction taps —
this is what makes the surgery in §3 a per-shell block copy.

### 2.2 Group action and weight tying

The D6 action permutes directions and fixes distance. With `tapp[g]` the
existing 7-tap permutation (`equivariant.build_group()`), the 31-tap
permutation is blockwise per shell:

    tapp31[g][0] = 0
    tapp31[g][1 + 6*(k-1) + d] = 1 + 6*(k-1) + (tapp[g][1 + d] - 1)

New `conv_gather_index31()` in `equivariant.py`, same formula as
`conv_gather_index()` with `tapp31`:

    idx[t, a, b] = tapp31[inv[a]][t] * 12 + mult[inv[a]][b]     # (31, 12, 12)

Free parameter: `w_base (31, 12, C_ORBIT_out, C_ORBIT_in)` = 95,232 per conv
at C=192. `gen_conv_weight()` generalizes by replacing the literal `7` with
`w_base.shape[0]` (shape-generic; existing callers unaffected).

Equivariance argument (for the docstring): the tap set is closed under the
per-shell direction permutation and the distance index is group-invariant, so
the 7-tap derivation (docs/DERIVATION (GEN)) applies verbatim with the
extended tap permutation. The visibility mask transports covariantly exactly
as for ray-tap (the raylen wire transport is already pinned by the T3
harness in `tests/test_hexfield_eq_raytap.py`).

### 2.3 Forward semantics (reference path)

For an equipped conv, per cell i:

    x̃_{d,k} = vis_{d,k} ⊙ x_{i + k·d}          # gathered ray fiber, hard-masked
    out_i   = GEMM_31([x_i, x̃ in shell-major tap order]) + bias   # (31·C → C)

- The gather indices and reach already exist per forward: `_RayTapCtx.idx_taps
  (B, Npad, 6, 5)` and `.reach (B, Npad, 2, 6)` built once by `trunk()`. Reuse
  them unchanged. The (dir, k) axes must be permuted to shell-major (k-major)
  order when flattening to match §2.1.
- `vis_{d,k}` per channel c: `k <= reach[side(c), d]`, with side(c) = own for
  orbit index < C_ORBIT/2, opp otherwise — **do not re-derive this; factor or
  mirror the masking in `_raytap.py` so the semantics cannot drift**, and pin
  it with the equivalence tests (§5).
- Center tap, bias, LN placement, block structure: unchanged from
  `HexNodeConv` / `ConvBlock`.
- Missing cells (index sentinel Npad) hit the zero row exactly as today.

### 2.4 Mode plumbing

New raytap mode string **`dense31`** alongside `0|conv2|both`:

- `HEXFIELD_EQ_RAYTAP=dense31` ⇒ both convs of every trunk ConvBlock equipped
  with the dense31 conv (stem always baseline, as today).
- `HexNodeConv` grows a mode for it (bool→mode migration or a separate kwarg —
  implementer's choice, but the existing `raytap=True/False` constructor
  surface and state-dict key set for `0/conv2/both` must remain byte-for-byte
  unchanged; T6 tests pin this).
- `arch_meta` records the mode; `infer_net_kwargs_from_state_dict` recovers it
  meta-first with a key-set/shape fallback (`w_base.shape[0] == 31`, no
  `alpha` key).
- Fresh (non-surgery) init: center + shell-1 blocks use the standard 7-tap
  uniform init (fan-in on the orbit basis, as `HexNodeConv.__init__`), shells
  k≥2 zero. This makes a fresh dense31 net function-identical to a fresh
  7-tap baseline net given the same RNG stream — the T4 analogue — and the
  zero blocks sit directly in the GEMM gradient path (no gate in front), so
  they train from step 1. Consume no extra RNG for the zero blocks (mirror the
  alpha convention: the shared-param stream stays aligned with the baseline
  build).

### 2.5 Training-path autograd Function (K2 analogue)

A `_Dense31ConvFn(x, idx_taps, reach, weight, bias, mask)` custom Function
that computes gather → GEMM without saving the `(B, N, 31·C)` gathered tensor
for backward:

- forward: build the masked gather, GEMM, save only `x, idx_taps, reach,
  weight` (+ shapes).
- backward: recompute the gather for `grad_weight = gatheredᵀ @ grad_out`;
  `grad_x` = scatter-add of `(grad_out @ weightᵀ)` through the same masked
  index map (the existing `_RayTapTaps.backward` scatter pattern is the
  template).
- The dense `weight` argument is the *generated* tied weight — the tying
  gather upstream (`gen_conv_weight`) stays in normal autograd, as today.

On CPU shapes this saves little; it exists to de-risk the later GPU training
path (the design doc estimates the naive intermediate at 3–4 GB fp32 at
current batch shape) and must be oracle-tested now (§5, T8 analogue). The
reference (naive) path is kept and remains the oracle.

## 3. Warm-start surgery (corrects the design doc)

`docs/RAY_CONV_DESIGN_SPACE.md` §3 claims k≥2-zero init after folding only
α_{k=1} is output-equivalent to a trained ray-tap checkpoint. **That is wrong
once α has trained off its (1,0,0,0,0) init.** The exact fold is over every
distance:

    B_center        = W_center                       (copy)
    B_{d,k}[slot s] = W_d[slot s] @ diag(α[k-1, :])  for all d, k
                      (α column-scales orbit_in;  α is slot-constant, so the
                       same diag applies to every slot's 16×16 block)

Deliverable: `scripts/dense31_surgery.py` — takes a ray-tap state dict
(`both` mode), emits the dense31 state dict: per equipped conv, expand
`w_base (7,12,·,·) → (31,12,·,·)` by the fold above, drop `alpha`, update
`arch_meta`. Must round-trip through `infer_net_kwargs_from_state_dict`.

This is exact real-arithmetic equivalence; fp32 reassociation gives ~1e-6
relative differences. Test tolerance accordingly (calibrate, target ≤1e-5 on
outputs).

## 4. Deliverables

- **D1** — `equivariant.py`: `conv_gather_index31()`, shape-generic
  `gen_conv_weight`.
- **D2** — `model.py`: dense31 mode in `HexNodeConv`/`ConvBlock`/env/meta
  plumbing (§2.3–2.4), reference forward.
- **D3** — `_Dense31ConvFn` recompute-in-backward Function (§2.5), used by the
  dense31 forward when grad is enabled; reference path selectable for tests.
- **D4** — `scripts/dense31_surgery.py` (§3).
- **D5** — tests: new `tests/test_hexfield_eq_dense31.py` (§5).
- **D6** — expressivity probe: `scripts/_dense31_expressivity_probe.py` (§6).
- **D7** — CPU micro-bench: `scripts/_dense31_cpu_bench.py` (§7).
- **D8** — read-out: `docs/DENSE31_CPU_READOUT.md` — what was built, test
  results, probe curves/tables, bench table, deviations from this plan, and
  the explicit list of GPU-side items deliberately not done.

## 5. Tests (new file, following the raytap T-numbering conventions)

1. **Gather-index sanity**: `conv_gather_index31` shells permute exactly as
   `conv_gather_index` taps 1..6 (blockwise equality per shell); center row
   matches.
2. **T3 analogue — full-net D6 equivariance** in dense31 mode, all 12 group
   elements, randomized params, real positions, raylen transported — reuse
   the existing T3 harness machinery from `tests/test_hexfield_eq_raytap.py`.
3. **T4 analogue — fresh-init equivalence**: fresh dense31 net (shells k≥2
   zero, matched RNG) equals the fresh baseline 7-tap net on real positions,
   both sides to move; plus liveness (perturbing a k≥2 block changes output).
4. **Surgery equivalence (the headline)**: build a ray-tap `both` net,
   randomize all params *including α* (T5's trained-alpha convention), run
   the surgery, build the dense31 net from the result: outputs equal on real
   positions to the §3 tolerance. This test failing under only-k=1 folding
   (i.e., a regression to the design doc's claim) is the exact bug the plan
   corrects — add a negative control asserting only-k=1 folding does NOT
   match once α is off-init.
5. **T8 analogue — Function vs naive oracle**: outputs bitwise-equal, grads
   ≤1e-5 rel on small random shapes (mask + sentinel rows exercised), plus a
   float64 `gradcheck` on a tiny shape, plus an assertion the Function saves
   no `(·, 31·C)`-sized tensor (mirror
   `test_t8_k2_saves_no_gathered_intermediate`).
6. **T6 analogue — state-dict discipline**: key sets per mode unchanged for
   existing modes; dense31 arch_meta round-trip; `infer_net_kwargs...`
   fallback disambiguation; invalid mode strings rejected at env validation.
7. **Optimizer classification**: dense31 `w_base` lands in the same
   AdamW/grad group as the 7-tap `w_base` (mirror
   `test_alpha_lands_no_decay_and_trunk_conv`).
8. **Regression**: the entire existing `tests/test_hexfield_eq_raytap.py`
   stays green (CUDA-marked tests self-skip on CPU).

## 6. Expressivity probe (the investigative core)

Question: does per-(direction, distance) weighting buy single-layer pattern
identity that the α-form ray-tap cannot express, and does depth rescue the
7-tap (the design doc §5 "two convs per block" argument)?

Setup: synthetic boards via the pure-Python featurizer path
(`features.build_position` + `batching.collate_rows`, as the T4 tests do):
random legal-ish positions, ~8–24 stones per side within a radius-6 disk.
Per-legal-cell binary label, computed by an independent in-test ray walk:
**own stones at exactly k ∈ {1, 2, 4} and empty at k = 3 along at least one
of the 12 signed directions, under own-side visibility** (a broken-pattern
template — the kind of thing per-distance diagonal α provably compresses
away per channel).

Arms (identical data, steps, optimizer, LR schedule; report both param
counts — the point is expressivity per layer, not matched capacity):

- **P1**: stem → 1 ConvBlock (ray-tap `both`) → tied 1×1 → per-cell logit.
- **P2**: stem → 1 ConvBlock (dense31) → tied 1×1 → per-cell logit.
- **P3/P4**: same with 2 ConvBlocks (the depth-rescue control).

Metric: val BCE + AUC curves to convergence or a fixed step budget
(whichever first; keep each arm ≤ ~10 min CPU). Emit CSV + a markdown table
into the read-out. Expected (to be tested, not assumed): P2 ≈ perfect,
P1 clearly below, P3 narrowing the gap, P4 ≈ P2.

If the P1-vs-P2 gap fails to appear, that is a *material negative finding*
for Design A's core rationale — report it prominently, don't tune it away.

## 7. CPU micro-bench

Fwd and fwd+bwd wall time per equipped conv, ray-tap vs dense31, at
(B=2, Npad=448, C=192) and (B=8, Npad=448, C=192), reference vs Function
paths, `torch.set_num_threads(8)`. CPU ratios do not predict GPU — record
them anyway with that caveat stated in the read-out.

## 8. Environment and hard constraints

- **Interpreter**: Windows `C:\Python314\python.exe` (torch 2.10.0+cu126,
  pytest 9.0.3 — verified working on the existing raytap suite, 2026-07-10).
- **`CUDA_VISIBLE_DEVICES=` (empty) on every python invocation.** The GPU
  belongs to the live `hexfield_eq_main_2` training run. No CUDA allocation
  of any kind, including "just to check".
- `PYTHONPATH=packages/hexfield_eq/python` (repo root as cwd) for tests and
  scripts.
- **No WSL commands.** Everything runs on the Windows side.
- One torch process at a time; `torch.set_num_threads(8)` in probe/bench
  scripts; keep peak RSS under ~6 GB.
- Work only inside this worktree; commits on `raytap-31tap-cpu` only; no
  pushes; never touch `configs/`, `scripts/_hexfield_eq_*soak*`, the
  `_triton_*.py` kernels (imports of them must keep guarding on
  availability), or anything under other worktrees.
- The Rust `_rust` extension is not built here; `needs_rust` tests self-skip.
  Do not attempt to build it.

## 9. Definition of done

    CUDA_VISIBLE_DEVICES= PYTHONPATH=packages/hexfield_eq/python \
      C:/Python314/python.exe -m pytest \
      tests/test_hexfield_eq_dense31.py tests/test_hexfield_eq_raytap.py -q

green on Windows CPU; probe and bench executed with results committed in
`docs/DENSE31_CPU_READOUT.md`; logical commits on `raytap-31tap-cpu`.

## 10. Explicitly out of scope

Triton/K1/split-serve kernel variants for 31 taps; fp16 serve fold + CUDA
graphs; `inference.py` fast-profile integration; Designs B/C gates; any
training on real corpora; any config or launch-script changes; anything that
touches the live run, the WSL side, or the GPU.
