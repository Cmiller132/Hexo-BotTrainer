# PLAN: 31-tap dense conv (tap31 / Design A), then modified 31-tap (mod31 / Design C)

Date: 2026-07-10. Status: PLAN, not started. Follows the line-reading design
note reviewed 2026-07-10 (Designs A and C therein; Design B is NOT in this
plan — recorded as an available cheap probe, §8). Builds on
SPEC_RAYTAP_CONV.md (repo root) and the shipped main_2 / arch-A5 net
(configs/hexfield_eq_main_2.toml, scripts/prefit_env/hexfield_eq_raytap_a5.env).

Working names: **tap31** = Design A (per-(direction, distance) full weight
blocks, no pre-GEMM aggregation). **mod31** = Design C (tap31 + transmittance
+ reader gate; the β/E gated message is omitted, §6.1).

Two review corrections are baked into this plan and are NOT in the design
note:

1. **The design note's §3 warm-start recipe is wrong.** Folding only the α k=1
   diagonal and zero-initializing k≥2 blocks is equivalent to a *plain 7-tap*
   conv, not to the trained ray-tap checkpoint (trained α ≠ (1,0,0,0,0)).
   The correct surgery folds trained α[k] into **every** distance-k block
   (§3.2). This plan's init-equivalence gate (G1) tests against the trained
   checkpoint's forward, which the note's recipe would fail.
2. **Strength evals for tap31 arms run at fixed wall-clock, not fixed
   visits** (§5.3). Post-ATTN2 the conv GEMMs dominate the serve forward;
   tap31 raises conv GEMM FLOPs ~4.4× and will give back a large share of the
   +61% pos/s. An eval that ignores this can prefer a net that is a net loss
   in deployed MCTS terms.

---

## 1. Decision record

- tap31 first: the game's tactical alphabet is stone/gap arrangements along
  axis lines; per-(direction, distance) blocks put exactly that in the
  weights, and the two-convs-per-block layout gives detect-then-count.
  Wave-1's own evidence points at the resolution axis (content-blind
  distance-resolved ray-tap: +0.117; removing the content-conditioned L
  blocks: no measured cost).
- tap31's new weights sit additively in the GEMM gradient path — none of
  mod31's zero-init multiplicative-gate risk. mod31 is staged strictly after
  tap31 so its two gates (transmittance, reader gate) are attributable on
  top of a trained tap31 anchor.
- α is dropped under tap31, not kept as a parallel path: with trained α
  folded into the init (§3.2) the full blocks strictly subsume it.
- Input featurization is NOT reduced in this plan (§7): v2/46-plane is
  load-bearing for the exact warm start, and coupling a feature cut to the
  conv change destroys attribution. A post-tap31 feature ablation is Phase 3.

## 2. tap31 mechanism

### 2.1 Tap set and ordering

Center + 6 directions × 5 distances = 31 taps. Convention (direction-major,
matching the `_triton_ray` packing and `idx_taps (B, N, 6, 5)`):

    t = 0                      center
    t(d, k) = 1 + d*5 + (k-1)  d in 0..5 (constants.DIRECTIONS order), k in 1..5

The hard visibility mask is retained unchanged: gathered ray fibers are
masked per-(side, tap) with the own/opp split riding the orbit halves,
exactly `_raytap._masked_gather` (first C_ORBIT/2 orbit channels own-side,
second half opp-side; pad rows carry raylen 0). α and the pre-GEMM sum over
k are removed; the masked fibers enter the GEMM directly:

    out = GEMM_31([x_i, vis ⊙ x_{i+k·d} for all (d,k)]) ,  (31·C → C)

Wire format is UNCHANGED: tap31 consumes the existing `_RayTapCtx`
(`idx_taps`, `reach`, `ray_idx`, `raylen`). No featurizer, shard, or Rust
change; existing corpora and replay buffers stay valid for every arm.

### 2.2 Group action and tying

A group element permutes the 6 directions and fixes the distance index
(distance along a ray is D6-invariant). With `tapp` the existing 7-tap
permutation (`equivariant.build_group()`):

    tapp31[g][0] = 0
    tapp31[g][t(d, k)] = t(d', k)   where 1 + d' = tapp[g][1 + d]

`conv_gather_index` generalizes (7, 12, 12) → (31, 12, 12) with the same
formula `flat[t,a,b] = tapp31[inv[a]][t]*12 + mult[inv[a]][b]`.
`gen_conv_weight` becomes tap-count-generic (read `w_base.shape[0]` instead
of the hardcoded 7). Free weights: `w_base (31, 12, C_ORBIT, C_ORBIT)`.

Parameters per equipped conv: 31·12·16·16 = **95,232** (baseline 21,504 + α
80). Ten equipped convs: +736,480 net. Model total 627,343 → **≈1,363,800**
(≈2.17×).

### 2.3 Mode plumbing

Extend the ray-tap mode enum: `HEXFIELD_EQ_RAYTAP ∈ {0, conv2, both,
dense31}`. `dense31` equips the same set as `both` (all 10 trunk convs; the
stem stays baseline 7-tap, spec §2.3). Checkpoint meta (`arch_meta`) is
authoritative as today (model.py ~1463); the state-dict fallback sniff is
`w_base.shape[0] == 31` on equipped conv keys (α keys absent under dense31).
The strict loader's bidirectional key equality must hold for the dense31
build against surgered checkpoints (§3.2).

## 3. tap31 initialization

### 3.1 From scratch

Uniform with fan-in 31·C_in: `bound = 1/sqrt(31 * in_channels)` on the orbit
basis (the existing pattern with 7 → 31).

### 3.2 Warm start from main_2 (weight surgery — the corrected fold)

For each equipped conv, with `w7 (7, 12, 16, 16)` and trained
`alpha (5, 16)` from the source checkpoint:

    w31[0]                = w7[0]                                   (center)
    w31[t(d,k), s, o, i]  = w7[1+d, s, o, i] * alpha[k-1, i]        (all d, k)

α multiplies the **input-orbit** index `i`; it is slot-constant, so the fold
stays inside the tied space, and visibility is applied to gathered fibers
identically in both forms — the surgered conv equals the trained ray-tap
conv **in exact arithmetic** (the baseline's Σ_k α·vis·x then GEMM
re-associates into the folded GEMM). Float parity is therefore
tolerance-level, not bit-for-bit: gate at ≤ 1e-5 rel on fp32 forwards
(T8-style), not T4-style bitwise.

Shim: `scripts/_tap31_surgery.py` — load a main_2 `epoch_*.pt`, rewrite the
10 equipped `w_base` keys 7→31 with the fold, drop the 10 `alpha` keys,
rewrite meta (`raytap: dense31`), save. Optimizer state is NOT carried
(shape change; fresh optimizer, warmup per §5.2).

## 4. tap31 kernels and memory

- **Reference path** (numerics oracle, CPU/CUDA): reuse
  `_raytap._masked_gather` per direction, stack to (B, N, 30, C), cat center,
  one (31·C → C) GEMM. Plain autograd on this path saves the gathered
  intermediate: ~717 MB fp32 per conv at B=48, S=648 (the number recorded in
  _raytap.py), ~+7.2 GB across 10 convs — the same OOM profile K2 was built
  for. Therefore:
- **K2-31 (training, BLOCKING — no training arm runs without it):** custom
  autograd Function saving only `x`, `idx_taps`, `reach`; backward recomputes
  the masked gather per direction; `grad_w31[t(d,k)]` accumulates from the
  recomputed gather × grad_out; `grad_x` scatter-adds `vis ⊙ (grad_out @
  W_{d,k}^T)` into source rows. Same structure as `_RayTapTaps`, minus α.
  This is also the mitigation for the open PAIR_BUDGET=4.0e7 training crash
  (root cause unknown): tap31 must not add resident training memory beyond
  the weight/optimizer growth. Record train-step memory before/after
  (`scripts/_hexfield_train_step_bench.py`).
- **K1-31 (serve), phased:**
  1. *Split path first* (the RAYTAP7/ATTN2 lesson: kernel + cuBLAS beats
     fused for big GEMMs): a masked-gather kernel emitting (B, N, 31·C) fp16,
     then cuBLAS (31·C → C) + LN. K-dim grows 1344 → 5952 (tensor-core
     utilization improves).
  2. *Graph-pool caution:* the (B, N, 31·C) fp16 intermediate is ~0.5 GB at
     (B=96, Npad=448) and ~1.6 GB at Npad=1396. Per the ATTN2 commit's
     PAIR_CEILING finding, large per-key intermediates pin CUDA-graph memory
     pools and can regress end-to-end throughput. If graph capture degrades,
     the fallback is a fused gather+GEMM (extend `_hex_conv_fused`, which
     already avoids materializing 7·C) — build only if the split path's
     measured pools hurt.
- **Serve env:** new import-time gate `HEXFIELD_TRITON_TAP31` added to
  `serve_env.IMPORT_TIME_FLAGS` **in the same change that lands the kernel**
  (the RAYTAP7 omission cost a sprint of mismeasured evals — see the
  serve_env docstring and the ATTN2 commit message).

## 5. tap31 measurement and evaluation

### 5.1 Pre-training gates (G0, all recorded before any arm trains)

- Serve: `scripts/raytap_serve_throughput.py` + `scripts/_bench_forward_profile.py`
  extended to dense31 — ms/fwd, pos/s at the A5 soak shapes, CUDA-graph
  capture health. Expectation to beat: the design note's "end-to-end < 3×
  FLOPs factor"; record the actual pos/s ratio vs the ATTN2 baseline (7.12
  pos/s on the RTX 4070 Ti reference).
- Training: step time + peak memory at the main_2 batch shape with K2-31.

### 5.2 Arms (wave-1 discipline: one variable per arm)

Envs in `scripts/prefit_env/` (clone `hexfield_eq_raytap_a5.env`, change one
line each):

| arm | env delta vs A5 | init | tests what |
|---|---|---|---|
| T31-W | `HEXFIELD_EQ_RAYTAP=dense31` | surgery from main_2 (§3.2) | tap31 on top of trained reading |
| T31-S | `HEXFIELD_EQ_RAYTAP=dense31` | scratch | tap31 without warm-start confound |
| W288 | `HEXFIELD_EQ_CHANNELS=288`, `HEXFIELD_EQ_C_ORBIT=24` | scratch | **the Q7 control**: param-matched width (~1.41M ≈ tap31's 1.36M) at only 2.25× conv FLOPs |
| A5-C | none | main_2 continued | anchor: same extra compute, no arch change |

W288 notes: C must stay a regular fiber (÷12) with even C_ORBIT (own/opp
halves) — 288 = 12·24 satisfies both; HEAD_DIM becomes 96. If W288 matches
T31 strength at equal params and fewer FLOPs, width wins the spend and tap31
does not ship — that is a recordable outcome, not a failure of the plan.

### 5.3 Protocol corrections (both mandatory for this wave)

1. **Longer prefits.** 600-step prefits are structurally biased against a
   2.2× model (larger nets are worse early, better late). This wave runs
   2,400-step prefits minimum; if the ladder budget forces 600 steps, treat
   results as inconclusive for T31-S/W288 and decisive only for T31-W (which
   starts at trained function).
2. **Fixed wall-clock evals.** SealBot/Strix evals at a fixed time budget per
   move (the eval kit's time-budget path — `test_hexfield_eq_time_budget.py`
   pins the machinery), alongside the usual fixed-visit numbers. The
   fixed-clock number is the shipping criterion; fixed-visit is diagnostic
   (did the eval get better) only.

### 5.4 Gates

- **G1 (surgery):** surgered-dense31 forward parity vs the trained main_2
  checkpoint ≤ 1e-5 rel fp32 on real batches. A failure here means the fold
  is wrong — do not proceed to training on a warm start that silently
  discards the trained ray reading (the design-note §3 trap).
- **G2 (ship/stop):** T31-W beats A5-C at matched additional compute on
  fixed-wall-clock evals by ≥ 2σ, AND T31-* is not dominated by W288.
  Pass → tap31 becomes the new anchor arch; proceed to Phase 2 (mod31) and
  Phase 3 (feature ablation). Fail → stop; mod31 is not attempted on top of
  a losing base (its gates are additions to tap31, not to A5).

### 5.5 MacBook micro-probe (2026-07-10, run — scripts/_tap31_probe.py)

Pre-ladder representation probe, CPU/MPS reference math only. Tiny untied
nets received ONLY occupancy planes (own/opp/empty/legal) and regressed the
v2 featurizer's 30 graded window planes per node (ground truth free from the
featurizer; inputs withhold it). Both mechanisms shared identical ray
gather, own/opp visibility halves, and training protocol (500 steps, batch 8,
Adam 1e-3 cosine, ~2200 line-biased random positions, held-out R² per plane).
d1 = one linear conv (function-class gap); d2 = two pre-LN residual blocks.
Seed 0 throughout; the d2 param-matched pair was re-run at seed 1 and the
split ordering in finding 2 held on both seeds.

| arm | params | line | live | live3 | live4 | live5 |
|---|---|---|---|---|---|---|
| d1-ray7-C64 | 31k | 0.371 | 0.585 | 0.059 | −0.69 | −7.0 |
| d1-tap31-C64 | 129k | **0.724** | **0.810** | 0.068 | −1.03 | −10.1 |
| d2-ray7-C64 | 119k | 0.504 | 0.763 | −0.03 | −1.62 | −15.8 |
| d2-tap31-C32 | 129k | 0.470 | 0.825 | −0.52 | −3.14 | −40.4 |
| d2-tap31-C64 | 511k | 0.665 | 0.873 | −0.13 | −1.50 | −15.5 |
| d2-ray7-C128 | 467k | 0.637 | 0.829 | **0.169** | **−0.60** | **−8.0** |

Findings (micro-scale; single seed except where noted):

1. **The linear expressivity gap is real and large** (d1): tap31 nearly
   doubles ray7's R² on line/live with no nonlinearity to compensate. The
   mechanism's premise holds.
2. **At depth 2 and matched params: a stable split, both seeds.** tap31-C64
   beat ray7-C128 on the two high-signal families (line +0.03/+0.13, live
   +0.04/+0.08 across seeds); the width arm beat tap31 on the rare families
   (live3 the only meaningful one: 0.17/0.12 vs −0.13/+0.07; live4/5 are
   negative for every arm — "less bad", not "learned"). tap31 also showed
   larger seed-to-seed variance. Param-matching tap31 by narrowing (C32)
   was worse across the board: tap-locked params cannot buy back fiber
   width.
3. **No arm reconstructed live4/live5** (negative R² everywhere at this
   scale) — the hand-computed planes carry structure tiny trunks do not
   rebuild; supports §7 (keep v2 features).

Caveats: 2 blocks vs the real 5C+3A trunk; untied; occupancy-only inputs;
reconstruction targets, not strength; 500 steps. Standing implication for
§5.4: the probe neither confirms nor kills tap31 — the mechanism's edge on
common line structure survives depth and matched params, but width holds
the rare patterns. The W288 control arm is mandatory and genuinely
undecided; G2 stands as written.

## 6. Phase 2: mod31 (Design C)

Runs only after G2 passes. Warm-starts from the best tap31 checkpoint.

### 6.1 Mechanism

Per equipped conv, walking k = 1..5 near-to-far per direction:

    opacity        t_j  = 1 − γ_t ⊙ σ(u ⊙ x_j + c)         (elementwise)
    transmittance  T_k  = vis_k ⊙ Π_{j<k} t_j              (T_1 = vis_1)
    attenuated     x̃_k  = T_k ⊙ x_{i+k·d}
    reader gate    g_i  = 1 + γ_g ⊙ tanh(W_g · x_i)

    out = GEMM_center(x_i) + g_i ⊙ GEMM_ray([x̃_{d,k} all d,k])

Implementation note: the 31-tap GEMM splits into center (C → C) + ray
(30·C → C) so the gate multiplies only the ray portion; at γ_g = 0 this is
exactly tap31's single GEMM re-associated.

`u, c, γ_t, γ_g (C_ORBIT,)` slot-constant tiled; `W_g` one tied 1×1
(EquivLinear-form, 12 free 16×16 blocks) per conv, shared across the 6
directions. All new ops are elementwise / same-rep products / one tied 1×1 —
no new tying derivations. Transmittance is attenuate-only on top of the hard
mask (cannot pass hard-blocked cells); the β/E gated message of Design B is
omitted per the design note's two-convs-per-block argument (recorded caveat:
conv2 aggregates direction-mixed conv1 outputs, so pre-sum per-direction
rectification is not fully replicated — accepted for attribution's sake).

Params: +48 (opacity) + 3,072 (W_g) + 16 (γ_g) = +3,136 per conv, +31,360
model-wide (≈1.395M total). Compute: +one tied C×C GEMM per conv +
elementwise on the gathered fibers; est. ≤ +10% forward over tap31.

### 6.2 Init, gates-health, ablation

- Zero-init γ_t and γ_g → bit-for-bit tap31 at load (multiplicative
  identities; no re-association this time). Direct strict-load after adding
  the new keys (surgery not needed; a key-adding tolerant warm start, the
  checkpoints.py warm path).
- **Dead-gate risk is the known failure mode** (design note §4; precedent
  spec D-S28). Mandatory diagnostics on the training dashboard from step 0:
  per-conv ‖γ_t‖, ‖γ_g‖, grad-norms on both, and realized transmittance
  stats (mean/min T_5 over a fixed probe batch). Decision rule: if all γ
  grad-norms sit at noise floor through the first eval checkpoint, record
  the arm as gate-dead rather than "mod31 adds nothing" — these are
  different findings. Zero-init is retained despite the risk (exact warm
  start is worth more, and the failure is detectable); small-noise init is
  the recorded fallback, costing init-equivalence.
- Post-hoc ablation on the trained checkpoint: zero γ_t / zero γ_g
  independently, re-run fixed-clock evals → per-component attribution.

### 6.3 Kernels and tests

- K1: the cumulative product is a running register value in the ordered
  k-loop of the split taps kernel (the structure RAYTAP7 already has); W_g
  rides the existing EquivLinear serve materialization/cache.
- K2: all new ops recompute from `x` (elementwise + one 1×1) — extend the
  K2-31 Function; T8-style small-shape gradient oracle against plain
  autograd on the reference path.
- Tests: init-equivalence (γ=0 ≡ tap31, bitwise), equivariance harness pass,
  kernel parity, zero-γ post-hoc ablation parity.

## 7. Input featurization: decision — do NOT reduce now

Question raised: with tap31 able to express line patterns in-weights, should
the v2 46-plane featurization (graded window planes: own/opp line, live,
live3/4/5 per axis) be cut back accordingly, now?

**No.** Four reasons, in order:

1. **It forfeits the warm start the plan is built on.** The stem is a
   typed-lift Reynolds projection whose free param is `w0 (7, C,
   NUM_FEATURES)` and whose typing sets (`_AXIS_PLANES`, `_SCALAR_PLANES`,
   `N_AXIS_QUANTITIES`) are derived from the plane map at import
   (equivariant.py). Any feature cut changes the stem shape and the input
   rep — no surgery preserves function across it, and the repo's own rule
   applies (main_2 config header: "NO cross-feature-width permanent
   anchors").
2. **It destroys attribution.** The wave-1 discipline is one variable per
   arm; coupling a feature cut to the conv change makes T31 results
   uninterpretable. This is the same error class as skipping A0 was, doubled.
3. **The planes' contribution is unmeasured.** The v1-vs-v2 ablation (arm
   A0) was never run. The liveK planes hand the trunk, at layer 0, exact
   clean-window counts over full length-6 windows *through* each cell —
   something a tap31 conv can only rebuild by composing detect + count
   across blocks after training. Cutting them on the theory that tap31
   re-derives them is a bet stacked on an untrained bet.
4. **Keeping them is nearly free where it matters.** Feature width touches
   only the stem lift and the CPU featurizer; trunk GPU cost is independent
   of NUM_FEATURES. Reduction buys ~nothing on the serve profile tap31
   actually stresses.

**Phase 3 (after G2, orderable alongside Phase 2):** run the deferred
feature ablation at the tap31 arch, where redundancy is most plausible —
scratch prefits, matched protocol (§5.3): `tap31 + v2` vs `tap31 + v1`
(`HEXFIELD_EQ_FEATURE_VERSION=1`, 25 planes). If v1 matches v2 under tap31,
*then* cut the liveK planes in a follow-up wave (featurizer CPU savings and
a smaller stem are the prize); if it doesn't, the planes are load-bearing
and stay. Either way the answer becomes a measurement, not a guess.

## 8. Out of scope (recorded)

- **Design B as a standalone arm** (+5% params, +8–10% FLOPs, direct
  warm start, per-component γ ablation). Not in this plan per the A-then-C
  decision; it remains the cheap independent probe of the
  content/receiver-conditioning axis if tap31 disappoints or if mod31's
  gates come up dead.
- GPU-side raylen derivation from occupancy planes (orthogonal, noted in the
  design note).
- Bounded CUDA-graph key ladder for PAIR_CEILING (the ATTN2 commit's
  follow-up); interacts with K1-31 pool sizing but is its own work item.

## 9. Work items and sequencing

| # | item | blocks |
|---|---|---|
| W1 | `equivariant.py`: `tap31_gather_index()` (31,12,12); `gen_conv_weight` tap-count-generic; T7-style generated-geometry assertion | — |
| W2 | `HexNodeConv` dense31 mode (construction, reference forward via `_masked_gather`, no α); `ConvBlock` dispatch; mode enum + `arch_meta` + state-dict sniff | W1 |
| W3 | K2-31 training Function + T8-style gradient oracle | W2 |
| W4 | `scripts/_tap31_surgery.py` + strict-load round-trip + **G1 parity test vs trained main_2** | W2 |
| W5 | equivariance + smoke + checkpoint-meta tests green under `HEXFIELD_EQ_RAYTAP=dense31` | W2 |
| W6 | K1-31 split serve path + `HEXFIELD_TRITON_TAP31` in `serve_env.IMPORT_TIME_FLAGS` (same change); parity tests | W2 |
| W7 | G0 benches (serve pos/s, forward profile, train step/memory) recorded | W3, W6 |
| W8 | arm envs + prefit configs (T31-W, T31-S, W288, A5-C); ladder run; G2 decision | W4, W5, W7 |
| W9 | Phase 2: mod31 params/reference/K1/K2, diagnostics, tests; arm + ablation | G2 |
| W10 | Phase 3: feature ablation arms (tap31+v1 vs tap31+v2) | G2 |

Sequencing: W1→W2→{W3, W4, W5, W6}→W7→W8 (G2) → {W9, W10} in either order
or parallel.
