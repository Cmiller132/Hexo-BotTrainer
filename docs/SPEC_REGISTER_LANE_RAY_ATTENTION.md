# SPEC — register lane + ray attention: implementation contract

Status: IMPLEMENTATION SPEC. Date: 2026-07-08. Companion to
`docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md` (decisions R1–R8, L1–L7, T1–T3 are
LOCKED there; this doc pins every implementation detail the plan leaves open).
Target package: `packages/hexfield_eq`. Group-theory grounding:
`docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md` §4 (coset heads), §5 (joint bias
tie, `S_o = D6` head-constancy), §6 (token rows).

Notation: `C = GROUP_ORDER * C_ORBIT` (192 = 12·16 at the D5 target), `T =
NUM_TOKENS = 6`, `N = Npad`, heads/cosets indexed `[Q, R, QR]` in the
`equivariant.build_group()["cosets"]` order.

---

## 0. Spec decisions beyond the plan (each with rationale)

- **D-S1 — env names are `HEXFIELD_EQ_REG_LANE` / `HEXFIELD_EQ_REG_TOK_READ`**
  (not the plan's provisional `HEXFIELD_REG_*`). Plan §4 says new knobs follow
  whatever resolution the package adopts for the env-collision issue; the
  package has since adopted the `HEXFIELD_EQ_*` namespace for every arch knob
  (`constants.py:100-106`, enforced by
  `tests/test_hexfield_eq_checkpoint_meta.py::test_hexfield_legacy_env_names_are_ignored`).
  `REG_LANE` is an arch knob (it changes the state-dict key set), so it takes
  the arch namespace. Phase-L knobs follow: `HEXFIELD_EQ_RAY_BLOCKERS`.
- **D-S2 — the toggles are also constructor kwargs** `reg_lane: bool | None`,
  `reg_tok_read: bool | None` on `HexfieldNet` (None → the env constant),
  recorded in `arch_meta()` and read meta-first by
  `infer_net_kwargs_from_state_dict` with a state-dict-key fallback
  (`registers.` / `tok_reads.` prefix presence ⇒ True, absence ⇒ False).
  Rationale: plan §4 makes checkpoint meta load-bearing for exactly these
  fields, and the package's existing idiom (channels / attention_heads /
  trunk_layout) is env-default + explicit-kwarg override for foreign-arch
  loaders. This also lets one test process build both arms.
- **D-S3 — register modules are constructed AFTER `_init_weights()`** in
  `HexfieldNet.__init__`. Rationale: `_init_weights` re-initializes every
  `nn.Linear` (it would clobber the zero-init `out_proj` in the passthrough
  build), and constructing the lane last keeps the shared-parameter RNG stream
  identical to the toggle-off build under the same seed (the zero-init identity
  gate compares the two).
- **D-S4 — `register.py` is imported lazily inside `HexfieldNet.__init__`**
  (only when the lane is on). Rationale: `register.py` reuses `EquivLinear` /
  `_make_norm` / `EQUIVARIANT` from `model.py`; a top-level import in both
  directions is a cycle. The lazy import runs after `model.py` is fully
  initialized, and costs nothing when the lane is off.
- **D-S5 — `TokenRead` is a `ModuleList` of `NUM_TOKENS` per-token 1×1s**
  (`EquivLinear` in the equivariant build), each with its own bias. The
  per-token biases are redundant (they sum into one effective bias) but
  zero-init makes them a no-op and `EquivLinear` has no bias-less mode; a
  flattened `Linear(T*C, C)` is NOT usable in the equivariant build (its input
  would not be slot-major regular fibers, breaking the `EquivLinear` tie).
- **D-S6 — `RegisterRefresh(channels, heads=None)`** mirrors
  `RelPosAttention`'s optional-heads signature so foreign-arch loaders (the
  dashboard debug worker) can rebuild at an explicit head count.
- **D-S7 — grad-norm group name is `trunk_reg`**, covering `registers.*` and
  `tok_reads.*`. Rationale: plan §6 wants the lane's grad-norm watched during
  prefit as its own signal, not folded into `trunk_attn` (whose scale it would
  distort) — `trainer._group_grad_norms` logs each group generically so a new
  key rides for free.
- **D-S8 — `gate_bias` joins the NAMED no-decay predicate** in `plugin.py` and
  `prefit.py` (it is 1-D so already no-decay by ndim; the name-match guards a
  future 2-D reshaping exactly as the `bias_theta` predicate comment does).
- **D-S9 — `ln_kv` is evaluated once** and shared by `k_proj`/`v_proj` (the
  plan sketch writes `self.ln_kv(x)` twice; one evaluation, same math).
- **D-S10 — the pad-cell key mask is a multiplicative fp32 zero on the gates**
  (`gates * mask`), not an additive −3e4 on the scores. Rationale: R1's
  aggregation is an unnormalized sum, so exact zeros (not σ(−3e4) ≈ 0) are the
  correct "cell absent" semantics and keep the count interpretation exact.
- **D-S11 (Phase L) — the L bias parameter is named `bias_theta_l`**. The
  substring `bias_theta` keeps it inside the existing AdamW no-decay predicate
  (`plugin.py`) and the `trunk_attn` grad-norm predicate
  (`trainer.py:startswith("bias_theta")`) with no predicate edits.
- **D-S12 (Phase L) — raylen is measured from the ROW'S OWN cell regardless of
  occupancy** (stones and empties both get rays; the walk truncation depends
  only on the cells beyond `x`, matching L1's rule which never inspects `x`
  itself). Every support cell gets 12 values; halo cells included (they are
  attention rows like any other).
- **D-S13 (Phase L) — `raylen` rides the serve wire as one `u8` buffer of
  `total_nodes * 12`**, padded in batching to `(B, Npad, 12)` with pad rows 0
  (a 0-length ray masks everything but the diagonal, which is the correct
  no-op for pad rows since their outputs are re-zeroed anyway).
- **D-S14 (L0, as built) — `ABI_VERSION` stays 1.** The payload is a
  self-describing dict, the `raylen` key is additive, and no consumer requires
  it yet (`inference.py` threads it into `build_serve_groups` but the model
  takes no raylen input until L1). The version bump belongs with the L1 model
  consumer, where a missing key must fail loudly.
- **D-S15 (L0→L1, CLOSED) — train-side `ExpandedRow` threading is landed.**
  `samples.ExpandedRow.raylen` (default empty for legacy constructors, packed
  as zeros), the serial expand computes it via the oracle
  (`features.build_ray_lengths`), `expand_backends._reassemble_rust_rows`
  slices the kernel buffer by `node_off` (tolerant of an older .so),
  `collate_training` always emits `batch["raylen"]`, and the trainer/prefit
  forwards pass `raylen=batch.get("raylen")` (C/A layouts ignore it).
- **D-S16 (L1, as built) — `RAY_BLOCKERS` is a MODEL-SIDE mask-build variant**
  (`HEXFIELD_EQ_RAY_BLOCKERS` env default "1", `ray_blockers` constructor
  kwarg, recorded in meta and read meta-first), NOT a Rust walk variant: the
  wire always carries the true game raylen; blockers-off simply ignores it
  (constant `RAY_REACH` reach — plan L6's geometric control). One shard /
  payload therefore serves both A/B arms, and blockers-off accepts
  `raylen=None`; blockers-on without raylen fails loudly in `trunk()`.
- **D-S17 (L1, as built) — L blocks use the plain `_FlexRayBias` carrier under
  BOTH flex modes** (fp32 master table on the grad path, fp16 cast on serve);
  the flex-pair precomputed-index variant and the bespoke Triton attention
  kernel never apply to L blocks in v1 (Phase L3 owns any gathered-kernel
  work). With both flex flags off, the materialized reference bias runs.
- **D-S18 (L1, as built) — serve-EVALUATOR threading for L layouts rides
  Phase L3.** The model-level serve path is complete
  (`forward_policy_value(..., raylen=...)`, flex/materialized under no-grad),
  but `inference.py`'s group assembly / CUDA-graph static buffers do not build
  a raylen tensor yet — that wiring lands with the L3 serve-perf pass
  alongside the graphs/kernel decisions. **CLOSED 2026-07-08 by D-S34
  (§0.2): the evaluator threading is landed and gated.**
- **D-S19 (L1, as built) — the L materialized grad path uses a plain
  differentiable `table[rows]` gather** (no `_BiasGather` histogram backward —
  an optimization owned by L3 if it measures hot), and the passthrough
  (GROUP_ORDER=1) L bias is an orbit-tied free `(BIAS_FREE_ROWS, RAY_HEADS)`
  table per block named `ray_bias_free_tables` (plain 6-head masked attention,
  no coset permutation), mirroring the A-block passthrough tie. Both L bias
  names keep the existing AdamW predicates' substrings (`bias_theta` /
  `bias_free_table`), so no optimizer predicate edit was needed; the trainer's
  `trunk_attn` grad-norm predicate gained `ray_blocks*` /
  `ray_bias_free_tables*`, and the L-block register lane
  (`registers_l*` / `tok_reads_l*`) lands in `trunk_reg`.

### 0.1 Pre-prefit fix bundle (2026-07-08 adversarial reviews; all landed)

- **D-S20 — widened invariant reads.** The scalar heads' fiber-invariant reads
  were a `C_ORBIT`(=16)-dim bottleneck. Now: one SHARED `inv_read =
  EquivLinear(C, INV_READ_EXPAND*C)` (`INV_READ_EXPAND = 4`) expansion before
  the group-pool feeds every value/aux/ml read block (block width
  `4*C_ORBIT = 64`); each per-cell policy head gets its own
  `POLICY_READ_EXPAND = 2` expansion (`{policy,opp,soft,cell_q}_expand`,
  pooled width 32); `red_out = 4*C_ORBIT = 64`. The VALUE head reads ALL
  `NUM_TOKENS` tokens (`value_reduction` in = `(NUM_TOKENS+2)*64 = 512`);
  aux/ml keep their pairs (in = 256). `equivariant.group_pool` generalized to
  any `k*C` regular fiber (reshape `(GROUP, width // GROUP)`). One shared
  expansion keeps params low; block diversity comes from the reductions.
- **D-S21 — pre-ln_final token read.** `trunk()` now returns the RAW token
  stream before `ln_final` (`pre_tokens`) and every scalar-head input gains
  one read block `inv_read(pre_tokens.mean(dim=1))` — the register lane's
  count magnitudes reach the heads unerased (ln_final would normalize them
  away). Gate: `test_value_input_counts_duplicated_patterns_end_to_end`.
- **D-S22 — out_proj init zero → trunc_normal std 3e-3.** Strict zero made the
  q/k/v/gate_bias/sum_scale grads provably exactly zero (every path chains
  through W_out). Still a numerical no-op at step 0 (measured max head delta
  1.9e-4); the R0-c gate is atol 5e-3 instead of bitwise; tok_reads stay
  exactly zero (their grads are live at zero).
- **D-S23 — gate_bias init −1.0 → −2.5** (background gate 0.076): kills the
  board-size-integrator common mode every token had at init.
- **D-S24 — learnable per-refresh `sum_scale`** (0-dim, init `REG_SUM_SCALE`,
  no-decay by ndim, `trunk_reg` group) multiplying the summed update, so
  blocks rescale their counts as boards grow. `REG_SUM_SCALE` stays as the
  init constant.
- **D-S25 — prefit regime.** Cosine LR decay `LR → LR/10` over the run after
  the 500-step warmup (`trainer.scheduled_lr`, epoch-granular decay);
  `GRAD_CLIP` 1.0 → 3.0 (the ×12-tied global grad-norm runs hotter;
  persistent clipping silently rescales LR non-uniformly); EMA twin (decay
  0.9995, updated every step, saved under `"ema_model"`, restored on resume,
  `evaluate()` runs on raw AND EMA → `ema_*` metrics); instrumentation:
  per-group grad norms incl. `trunk_reg`, `token_stream_max` (pre-ln probe),
  `train_val_{policy,value}_ce_gap`, ply-bucketed `value_mae_ply_*`,
  threat-dense subset metrics (`threat_rows/threat_top1/threat_value_mae` at
  fork-plane ≥ 0.6), and a `--soft-policy-weight` CLI (default = the losses
  constant 4.0) threaded to `hexfield_loss` in both call sites.
- **D-S26 — support radius pinned everywhere.** `arch_meta()["support_radius"]`;
  the serve payload's `support_radius` item, asserted by
  `inference.submit_payload`; `checkpoints.load_into` and
  `prefit.load_checkpoint` assert it at load; the three env readers
  (support.py, `expand_backends._resolve_support_radius`, Rust
  `support_radius()`) all clamp to `[1, HALO_DIST]`.
- **D-S27 — fp32 token-lane carry.** With the lane on, the loop-carried token
  stream stays fp32 between A blocks (re-upcast after each A split); the
  refresh projections run in the cells' compute dtype and only the residual
  add runs in the stream's dtype — half-precision serve cannot ulp-round late
  count writes away. No-ops on the fp32 train path.
- **D-S28 — `bias_theta_l` live-class note.** Only ~6 disk joint classes × 2
  sides are reachable by ray offsets; every other class sits behind the −3e4
  mask with underflowed grads. Documented at the parameter so future init/reg
  changes don't trip on it.
- **D-S29 — serial raylen oracle gated on layout.** `samples._EXPAND_RAYLEN =
  "L" in TRUNK_LAYOUT`: C/A prefit workers skip the per-row Python walk; the
  Rust expand kernel keeps emitting raylen regardless; the parity tests
  monkeypatch the gate on.
- **D-S30 — layout-aware `PAIR_BUDGET`.** Default drops 2.0e7 → 8.0e6 when
  the layout contains `L` (the materialized 6-head ray bias + live-mask
  transient runs ~2–3× the A-block bias); `HEXFIELD_EQ_PAIR_BUDGET` overrides.
- **D-S31 — all-zero-raylen guards.** `expand_backends` RAISES instead of
  fabricating zeros when the kernel omits the buffer under an L layout; a
  one-time latch (`trainer._check_raylen_once`, shared by prefit) asserts the
  first train batch's raylen is not identically zero over live cells when
  L + blockers are on.
- **D-S32 — prefit resume meta asserts.** `prefit.load_checkpoint` asserts
  `trunk_layout` / `reg_lane` / `reg_tok_read` / `ray_blockers` /
  `support_radius` against the BUILT model (a `ray_blockers` flip on an
  arm-4c resume was a silent mask-semantics change).
- **D-S33 — width contingency.** `head_dim = 96` (C=288, C_ORBIT=24) is
  permitted with a loud import warning: the bespoke Triton attention fast
  path will not engage (sdpa/flex serve only). Pre-authorized underfit
  contingency (docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md).
- **Test-tolerance note (equivariance with the lane on):** the lane's
  unnormalized sums make fp32 round-off scale with token magnitude; the
  randomized-param equivariance gates re-bound `gate_bias`/`sum_scale` to
  design-scale random values (`_bound_lane_scales`) so the 1e-4 atol keeps
  measuring STRUCTURE, not magnitude-amplified round-off.

### 0.2 Phase L3 serve pass (2026-07-08; closes the D-S18 deferral)

- **D-S34 — serve-evaluator raylen threading (landed).** `HexfieldEvaluator`
  reads the trunk layout FROM THE MODEL (`model._trunk_layout`, meta-first
  semantics — foreign-arch checkpoints serve correctly regardless of env) and
  sets `_needs_raylen = "L" in layout`. When set, every serve path builds a
  `(B, Npad, RAYLEN_SLOTS)` **uint8** raylen tensor (never halved, never
  widened — the model gathers from it directly) from the payload's CSR
  `raylen` buffer, pad rows 0 (D-S13), and passes it POSITIONALLY as the 5th
  forward arg; when unset, NO raylen object is created and the forward call
  carries the exact pre-L argument list — the C/A serve is **byte-identical**
  (sha256-verified against the pre-change tree over a fixed payload battery).
  Coverage: the CSR/Python pack (`_forward_group`), the Rust-pack consumption
  (`grp["raylen"]`, all three sub-paths: plain frombuffer, copy-stream pinned
  staging — `_PinnedRing.KEYS` gained `"raylen"` — and the fused
  graphs+copy-stream direct-into-static copy), CUDA-graph statics (a fixed
  -shape uint8 buffer per `(B_bucket, Npad, ml)` key, filled/zero-padded per
  replay exactly like `nbr`), the compiled path (`mark_dynamic` dims 0/1 on
  raylen alongside the other inputs), and the batch-1→2 duplication. The
  fp32 token-lane carry (D-S27) is model-side and unaffected. Guards: a
  payload MISSING the `raylen` key under an L net raises (the D-S14 "fail
  loudly at the model consumer" obligation — a pre-L0 featurizer .so cannot
  serve an L net; ABI stays 1, the key check is the loud failure); a byte
  -count mismatch raises; a ONE-TIME per-evaluator latch (the serve twin of
  `trainer._check_raylen_once`, D-S31) rejects an all-zero first buffer when
  blockers are on. Blockers-off L nets still get raylen threaded (uniform
  shapes for graphs/compile across the 4/4c arms) but skip the latch —
  geometric rays never read the values (D-S16). Gates:
  `tests/test_hexfield_eq_serve.py` (CPU: L-net serve ≡ train forward at the
  3e-3 serve tolerance with the wire f16 rounding mirrored into the
  reference; raylen-liveness A/B; C/A spy asserting the pre-L call shape;
  the three guard tests; CUDA-gated: eager fp16-autocast L serve parity +
  Rust-pack ≡ CSR bitwise).
- **D-S35 — equal-TIME eval leg (ray plan §3 L2/L3, review F7).** The Rust
  `session.search` has no wall-clock budget; deep surgery was declined. The
  sanctioned approximation: **per-arm visit calibration**
  (`eval_driver.calibrate_time_budget_visits`): one warmup + one timed
  single-root probe search per probe position (depths 0/8/16, seeded lines,
  probe keys ≥ 1.9e9 outside the game-key space, trees discarded), median ms
  → `visits_for_time_budget` (linear visits↔time model, clamps: floor 16,
  cap 8× base). `play_eval_match` calibrates net A after its
  session/evaluator exist; `HexfieldCheckpointAdapter.start` calibrates net B
  with ITS OWN evaluator — each arm pays its own architecture's latency,
  which is the point (arm 4's L blocks run the slow flex path; a fixed-visit
  arena hides that cost). Knob: `multi_stage_eval.eval_time_budget_ms`
  (config.py, default 0.0 = off = byte-identical fixed-visit path) or the
  explicit `time_ms_per_move` kwarg on `play_checkpoint_match` /
  `play_eval_match` (explicit wins; explicit 0.0 forces off). Calibration
  records ride `MatchTelemetry.time_calibration` / adapter
  `.time_calibration` into `meta.time_calibration_{a,b}`. Strix (fixed sims,
  the pinned anchor) and SealBot (already time-limited) ignore the knob.
  Known approximation limits (accepted, documented): multi-root batching
  amortizes forwards so the REALIZED per-move wall time sits below the budget
  for both arms alike — the A/B fairness rides on the ratio of measured nps,
  not the absolute per-move time; probes measure single-root latency at the
  base visit count. Gates: `tests/test_hexfield_eq_time_budget.py`.
- **D-S36 — gathered local-attention Triton kernel for L blocks: IMPLEMENTED
  2026-07-08 (as built: D-S37 below; the sketch is kept for the record).**
  D-S17's default stands — the kernel is env-gated OFF (`HEXFIELD_EQ_TRITON_RAY`)
  and L blocks serve on flex/materialized until it is benchmarked in an idle
  window (`scripts/bench_eq_ray_kernel.py`). Original sketch, measured against
  the flex path FIRST on real serve shapes before any code: (i) key set per
  query row `i` is `self ∪ {i + k·a⃗ : a ∈ {Q,R,QR}, k ∈ ±1..±5}` — ≤ 31
  distinct CELLS (the 6 heads share the 30 geometric neighbours; per head
  2 axes are dead, and the two side-heads of one coset differ only in
  raylen gating, ≤ 61 live (head, key) pairs); build a `(B, Npad, 31)` int32
  gather index ONCE per forward from coords (reuse the `_build_pair_u8`
  support-lookup machinery), value 0..Npad with Npad = the zero pad row.
  (ii) Kernel: one program per (batch, query-tile of 16/32 rows); load the
  31-key index tile, gather K/V fp16 tiles for ALL 6 heads at once
  (head_dim 2·C_ORBIT = 32 at C=192 → K tile 31×32 fp16 per head, fits
  registers/SMEM comfortably), compute the 6 per-head 31-wide score rows,
  add `bias_table_L[lut_row, h]` gathered by the same relative-offset LUT,
  mask by `|k| ≤ raylen[i, side(h), axis(h), dir(k)]` read from the u8 buffer
  (raylen tile 16×12 u8 per query tile), softmax over ≤ 32 keys (self + 31 →
  pad one), accumulate V. No N² object anywhere; arithmetic intensity is low,
  so the win over flex is bandwidth-shaped — measure `flex vs materialized vs
  this` at B×Npad of the live serve mix before committing. (iii) Env gate
  `HEXFIELD_EQ_TRITON_RAY` default "0"; parity gate 3e-3 vs the materialized
  path, CUDA-idle-gated test; fp16 in/out, fp32 softmax accumulator, follow
  `_triton_attn.py` idioms (tile-skip via seq_lens, the pad-row zeroing
  convention). Nothing in v1 serve depends on it (D-S30's PAIR_BUDGET drop
  keeps the materialized transient bounded).
- **D-S37 — gathered ray kernel, as built (2026-07-08;
  `_triton_ray.py` + the model.py wiring). Deviations from the D-S36 sketch
  are flagged.**
  - **Slot-indexed gather, 32 slots.** Slot 0 = self; slot
    `1 + axis*10 + dir*5 + (k-1)` for axis ∈ {Q, R, QR}, dir ∈ {+, −},
    k ∈ 1..RAY_REACH (packing asserts RAY_REACH == 5 at import); slot 31 is a
    permanent sentinel pad. The (B, Npad, 32) int32 index is block-independent,
    built ONCE per forward, value = key row or the sentinel `Npad` = absent
    (there is no appended zero row — the kernel masks absent lanes instead of
    reading a pad row; numerically identical).
  - **Index build = dense coordinate-grid lookup, NOT the `_build_pair_u8`
    pairwise-delta machinery** (deviation): scatter live rows into a
    (qmax−qmin+1)×(rmax−rmin+1) grid keyed off the LIVE coords' bounding box,
    then read grid[coords + slot_offset] — O(B·N·32), which keeps the sketch's
    "no N² object anywhere" literal. Pad rows go to a dump slot, so a pad cell
    can never be gathered even if its garbage coords alias a live cell; slot 0
    is forced to self on every row (no empty softmax row). The extent
    computation syncs (`.item()`), so the build lives in the opaque
    `hexfield_eq::ray_gather_index` custom op (torch.compile-safe) but is
    **NOT CUDA-graph capturable** — the knob must stay off under the graphs
    serve path (graphs integration deferred with the rest of L3 perf).
  - **Per-slot bias, no in-kernel LUT** (simplification): each slot's relative
    offset is fixed, so the expanded (BIAS_ROWS, 6) table collapses to a
    (32, 6) fp16 "slot bias" per block (`slot_bias_rows()` = rel_bias_index of
    each slot offset), resolved per block in `trunk()`'s `ray_bias()`.
  - **Kernel shape: one program per (batch, head, query-tile of BM rows)**
    (deviation from "all 6 heads per tile": heads share only the tiny
    idx/raylen tiles, not K/V bytes, so heads ride the grid axis and register
    pressure stays low). Per program: (BM, 32) idx tile, per-head liveness
    (slot 0 always; else axis == coset ∧, blockers on, k ≤
    `raylen[i, side*6 + axis*2 + dir]` read from the u8 wire tile; blockers
    off = constant reach, raylen never dereferenced — an empty dummy rides the
    op), fp16 K/V (BM, 32, D) gathers, SINGLE-PASS fp32 softmax over the 32
    slots (the whole key set is one tile — no online rescan), fp32 V
    accumulate, fp16 store. Dead slots get the additive −3e4: exp-underflow
    to exactly 0.0 in fp32, bit-identical to the materialized N² softmax's
    masked keys. seq_lens tile-skip + pad-row zero-store per `_triton_attn.py`.
    Tile knobs `HEXFIELD_RAY_BM` (16) / `HEXFIELD_RAY_WARPS` (4).
  - **Wiring.** `HEXFIELD_EQ_TRITON_RAY=1` (default 0) imports the ops in
    model.py (the `_TRITON_ATTN` idiom); `trunk()` builds a `_RayGatherBias`
    carrier only under no-grad ∧ CUDA ∧ (fp16 stream OR cuda-fp16 autocast —
    both evaluator serve modes) ∧ head_dim ∈ {16, 32, 64, 128}
    (C=288/head_dim 48 falls through per D-S33); `RayAttention.forward` routes
    the carrier to the `hexfield_eq::ray_attn` custom op. Every miss falls
    through to the flex/materialized paths unchanged; inside the op, a launch/
    compile failure is memoized per head_dim (`_triton_attn.py` idiom) and
    served from `_ray_ref`, a gathered pure-torch twin (also the CPU oracle).
    Knob-off serve is byte-identical (all additions are behind the gate).
  - **Gates** (`tests/test_hexfield_eq_triton_ray.py`): CPU — slot-table
    structure; gather-index implied live set ≡ `_ray_live_mask` EXACTLY
    (blockers on/off; lone-stone/disk/padded batches incl. an aliasing pad
    row); op ≡ materialized full-softmax at 3e-3 (and op ≡ `_ray_ref`
    bitwise); default-off. CUDA (tiny-shape, busy-GPU-polite with OOM
    retry+skip) — kernel through the RayAttention module vs the fp32
    materialized module path at 3e-3 over {lone, r3, dense r8, padded} ×
    blockers on/off, plus full-net serve wiring parity (5e-3, fp16 both arms)
    with a routing spy. All ran green 2026-07-08 (kernel verified compiled —
    empty `_RAY_FAILED`, ≤ 1e-3 vs the gathered reference).
  - **Perf: UNBENCHMARKED** (GPU owned by the prefit ladder at build time).
    `scripts/bench_eq_ray_kernel.py` (idle-gated: refuses at > 10% GPU util)
    sweeps kernel vs flex vs materialized over Npad ∈ {128, 256, 512, 768} ×
    batch × blocker arms, reporting the once-per-forward index build
    separately. Expected shape of the win: per-query work drops from
    Npad keys (flex/materialized softmax width, plus the (B, 6, N, N) bias/
    mask materialization traffic) to 32 keys — the kernel reads
    O(B·N·(32·D·2 heads' bytes)) instead of O(B·N²) score-side traffic, so
    the advantage grows with Npad and is bandwidth-shaped; measure before
    flipping the knob (D-S17 remains the default).

---

## 1. Phase R0 — register lane

### 1.1 `constants.py` additions

Follow the read-once-at-import + validate idiom (`constants.py:96-213`):

```python
# --- register lane (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase R) -----------
# HEXFIELD_EQ_REG_LANE ("0"/"1", default "0"): attach a RegisterRefresh (one-way
# sigmoid-gated SUM cross-attention, tokens <- cells) at the exit of every C
# block. HEXFIELD_EQ_REG_TOK_READ ("0"/"1", default "0"): the cells <- tokens
# broadcast read at C-block entry; only meaningful with the lane on.
_REG_LANE_ENV = os.environ.get("HEXFIELD_EQ_REG_LANE", "0")
if _REG_LANE_ENV not in ("0", "1"):
    raise ValueError(f"HEXFIELD_EQ_REG_LANE={_REG_LANE_ENV!r} must be '0' or '1'")
REG_LANE = _REG_LANE_ENV == "1"
_REG_TOK_READ_ENV = os.environ.get("HEXFIELD_EQ_REG_TOK_READ", "0")
if _REG_TOK_READ_ENV not in ("0", "1"):
    raise ValueError(
        f"HEXFIELD_EQ_REG_TOK_READ={_REG_TOK_READ_ENV!r} must be '0' or '1'"
    )
REG_TOK_READ = _REG_TOK_READ_ENV == "1"
if REG_TOK_READ and not REG_LANE:
    raise ValueError(
        "HEXFIELD_EQ_REG_TOK_READ=1 requires HEXFIELD_EQ_REG_LANE=1 (the read is "
        "an arm of the register lane, not a standalone mechanism)"
    )
# Fixed scale on the sigmoid-gated SUM aggregation (plan R1); matched-set sizes
# are tens of cells, so updates land O(1)-O(10). A constant, not an env knob.
REG_SUM_SCALE = 1.0 / 32.0
```

### 1.2 `register.py` (NEW file) — exact module code

```python
"""Register lane (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase R): a one-way
sigmoid-gated SUM cross-attention refreshing the summary tokens from the cells
at every C-block exit, plus the optional cells <- tokens broadcast read at
C-block entry. Imported lazily by model.HexfieldNet only when the lane is on."""

from __future__ import annotations

import math

import torch
from torch import nn

from .constants import ATTENTION_HEADS, GROUP_ORDER, NUM_TOKENS, REG_SUM_SCALE
from .model import EQUIVARIANT, EquivLinear, _make_norm

if EQUIVARIANT:
    from . import equivariant as _eq


class RegisterRefresh(nn.Module):
    # (docstring per package idiom)

    def __init__(self, channels: int, heads: int | None = None) -> None:
        super().__init__()
        self.heads = ATTENTION_HEADS if heads is None else int(heads)
        self.head_dim = channels // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.equivariant = EQUIVARIANT
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.ln_q = _make_norm(channels)     # pre-norm on tokens (q input)
        self.ln_kv = _make_norm(channels)    # pre-norm on cells (k/v input)
        self.q_proj = linear(channels, channels)
        self.k_proj = linear(channels, channels)
        self.v_proj = linear(channels, channels)
        self.out_proj = linear(channels, channels)
        # Per-token gate threshold, broadcast over heads (head-constant: tokens
        # carry no position, S_o = D6 forces head-constancy — plan R2).
        self.gate_bias = nn.Parameter(torch.full((NUM_TOKENS,), -1.0))
        if EQUIVARIANT:
            self.register_buffer("_head_perm", _eq.head_perm(), persistent=False)
            self.register_buffer(
                "_head_perm_inv", _eq.head_perm_inv(), persistent=False
            )
        self._init_projections()

    def _init_projections(self) -> None:
        # q/k/v: the trunk Linear init (trunc_normal 0.02, zero bias); out_proj
        # ZERO (weight+bias / base params) so the lane is a no-op at step 0 (R3).
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            if self.equivariant:
                nn.init.trunc_normal_(proj.wb, std=0.02)
                nn.init.zeros_(proj.bias_base)
            else:
                nn.init.trunc_normal_(proj.weight, std=0.02)
                nn.init.zeros_(proj.bias)
        if self.equivariant:
            nn.init.zeros_(self.out_proj.wb)
            nn.init.zeros_(self.out_proj.bias_base)
        else:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens, x, mask):      # (B,T,C), (B,N,C), (B,N) bool
        b, t, c = tokens.shape
        n = x.shape[1]
        h, d = self.heads, self.head_dim
        kv = self.ln_kv(x)
        q = self.q_proj(self.ln_q(tokens))
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        if self.equivariant:
            hp = self._head_perm
            q = q[..., hp]
            k = k[..., hp]
            v = v[..., hp]
        q = q.reshape(b, t, h, d).transpose(1, 2)   # (B, h, T, d)
        k = k.reshape(b, n, h, d).transpose(1, 2)   # (B, h, N, d)
        v = v.reshape(b, n, h, d).transpose(1, 2)
        scores = (q @ k.mT) * self.scale + self.gate_bias.view(1, 1, t, 1)
        # fp32 SUM aggregation (R1): sigmoid gates, pad keys exactly zeroed
        # (multiplicative — the sum must see 0, not sigma(-3e4)), no softmax.
        gates = torch.sigmoid(scores.float()) * mask[:, None, None, :]
        upd = (gates @ v.float()) * REG_SUM_SCALE   # (B, h, T, d)
        upd = upd.transpose(1, 2).reshape(b, t, c)
        if self.equivariant:
            upd = upd[..., self._head_perm_inv]
        # Raw residual add — no norm on the update (R5): the count magnitudes
        # ARE the signal; the next A block's pre-norm re-normalizes the stream.
        return tokens + self.out_proj(upd.to(tokens.dtype))


class TokenRead(nn.Module):
    # cells <- tokens broadcast read (plan R4): per-token tied 1x1s, summed,
    # ZERO-INIT, added at C-block entry.

    def __init__(self, channels: int) -> None:
        super().__init__()
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.reads = nn.ModuleList(
            [linear(channels, channels) for _ in range(NUM_TOKENS)]
        )
        for read in self.reads:
            if EQUIVARIANT:
                nn.init.zeros_(read.wb)
                nn.init.zeros_(read.bias_base)
            else:
                nn.init.zeros_(read.weight)
                nn.init.zeros_(read.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B,T,C) -> (B,1,C)
        upd = self.reads[0](tokens[:, 0])
        for ti in range(1, NUM_TOKENS):
            upd = upd + self.reads[ti](tokens[:, ti])
        return upd.unsqueeze(1)
```

Equivariance argument (why this composes with the derivation): tokens start
slot-constant (invariant, §6). `q_proj`/`ln_q` commute with `M(g)`; after
`head_perm` the token query satisfies `q^h(T_g f) = P_K(g)·q^{g^{-1}h}(f)`
(trivially at block 0 where `q` is `M(g)`-fixed, inductively thereafter), and
the cell keys/values satisfy the derivation's (H). Hence
`scores^h_{t,i}(T_g f) = scores^{g^{-1}h}_{t,g^{-1}i}(f)`; `gate_bias` is
head-constant so it survives the head shift (the `S_o = D6` case of §5.2);
sigmoid is pointwise; the SUM over cells is permutation-invariant. So
`upd^h(T_g f) = P_K(g)·upd^{g^{-1}h}(f)`, i.e. the merged update transforms by
`M(g)` — the token stream becomes a *covariant* regular fiber after the first
refresh, which is fine: every downstream consumer (A-block rows, whose token
bias rows are head-constant; the value/aux/ml heads, which `group_pool` the
tokens) is already equivariant for covariant token fibers.

### 1.3 `model.py` threading (exact edits)

`HexfieldNet.__init__` signature gains `reg_lane: bool | None = None,
reg_tok_read: bool | None = None`; resolved against the env constants with the
`REG_TOK_READ ⇒ REG_LANE` validation mirrored for explicit kwargs; stored as
`self._reg_lane` / `self._reg_tok_read`. Immediately after the existing
`self._init_weights()` call (D-S3):

```python
        if self._reg_lane:
            from .register import RegisterRefresh, TokenRead

            n_conv = layout.count("C")
            self.registers = nn.ModuleList(
                [RegisterRefresh(c, heads) for _ in range(n_conv)]
            )
            if self._reg_tok_read:
                self.tok_reads = nn.ModuleList(
                    [TokenRead(c) for _ in range(n_conv)]
                )
```

Trunk walk — current code (`model.py:1270-1274`):

```python
        for pos, kind in enumerate(layout):
            if kind == "C":
                x = self.conv_blocks[ci](x, gather_idx, mask)
                ci += 1
```

Replacement (the A-block arm is untouched):

```python
        for pos, kind in enumerate(layout):
            if kind == "C":
                if self._reg_tok_read:
                    x = x + self.tok_reads[ci](tokens) * mask.unsqueeze(-1)
                x = self.conv_blocks[ci](x, gather_idx, mask)
                if self._reg_lane:
                    tokens = self.registers[ci](tokens, x, mask)
                ci += 1
```

`tokens` is already live at every layout position (bound before the walk from
the learned slot-constant init, reassigned at every non-final A split), so the
loop-carry needs no other change. Read at block ENTRY, write at block EXIT
(reading the block's output), per plan §2.1.

`arch_meta()` gains two additive keys: `meta["reg_lane"] = self._reg_lane`,
`meta["reg_tok_read"] = self._reg_tok_read`.
`infer_net_kwargs_from_state_dict` reads them meta-first, with the key-set
fallback of D-S2. `KNOWN_TRUNK_LAYOUTS` is NOT extended (plan §4).

### 1.4 Param classification table (the lockstep contract)

Every new parameter name → AdamW group (`plugin.py`, `prefit.py`) and
grad-norm group (`trainer.py`). Passthrough / equivariant leaf names both
listed. The AdamW predicate is the existing
`ndim >= 2 and not named-no-decay and name != "tokens"`; only the named set
changes (D-S8).

| parameter name pattern                                   | ndim | AdamW    | grad-norm   |
|----------------------------------------------------------|------|----------|-------------|
| `registers.{i}.{q,k,v,out}_proj.weight` / `.wb`          | 2/3  | decay    | `trunk_reg` |
| `registers.{i}.{q,k,v,out}_proj.bias` / `.bias_base`     | 1    | no-decay | `trunk_reg` |
| `registers.{i}.ln_q.{weight,bias}` / `.{gamma,beta}`     | 1    | no-decay | `trunk_reg` |
| `registers.{i}.ln_kv.{weight,bias}` / `.{gamma,beta}`    | 1    | no-decay | `trunk_reg` |
| `registers.{i}.gate_bias`                                | 1    | no-decay (named, D-S8) | `trunk_reg` |
| `tok_reads.{i}.reads.{t}.weight` / `.wb`                 | 2/3  | decay    | `trunk_reg` |
| `tok_reads.{i}.reads.{t}.bias` / `.bias_base`            | 1    | no-decay | `trunk_reg` |
| *(L1)* `ray_blocks.{i}.*` proj / MLP `.wb`/`.weight`     | 2/3  | decay    | `trunk_attn` |
| *(L1)* `ray_blocks.{i}.*` norms / biases / LayerScale    | 1    | no-decay | `trunk_attn` |
| *(L1)* `bias_theta_l.{i}`                                | 2    | no-decay (named `bias_theta`, D-S11) | `trunk_attn` |
| *(L1)* `ray_bias_free_tables.{i}` (passthrough)          | 2    | no-decay (named `bias_free_table`, D-S19) | `trunk_attn` |
| *(L1)* `registers_l.{i}.*` / `tok_reads_l.{i}.*`         | —    | as `registers`/`tok_reads` rows above | `trunk_reg` |

`trainer._build_grad_norm_groups` gains the `trunk_reg` bucket keyed on
`name.startswith("registers") or name.startswith("tok_reads")` (checked before
the `heads` else-branch; `registers_l`/`tok_reads_l` match by prefix), and the
`trunk_attn` startswith set gains `ray_blocks` / `ray_bias_free_tables`.

### 1.5 Cost / serve notes

k/v/out are full-width GEMMs ≈ 3·N·C² (~15–20% of a conv block, R8 defers
thinning); score/aggregation is `T·N·C`. No data-dependent shapes: CUDA-graph
and `torch.compile(dynamic=True)` safe. The fp32 upcast covers fp16 serve
(`SERVE_HALF`): sums of ≤ N terms of O(1) magnitudes stay well inside fp32.

---

## 2. Phase L0 — ray data (Rust + wire + oracle)

### 2.1 The walk (plan L1, normative restatement)

For support cell `x`, side `s ∈ {own=0, opp=1}` (own = the side to move),
axis `a ∈ {Q=0, R=1, QR=2}` (vectors `(1,0), (0,1), (1,-1)`), direction
`dir ∈ {+=0, −=1}`: walk `j = 1..5`, `y = x + j·(±a.vector())`:

1. `y` not in the support node set → stop (len stays `j−1`);
2. `y` holds an anti-`s` stone (opp stone for `s=own`, own stone for `s=opp`)
   → len = `j` (terminal blocker INCLUDED), stop;
3. else (empty or `s`-side stone) → len = `j`, continue.

`raylen ∈ 0..5`. The occupancy of `x` itself is never consulted (D-S12).

### 2.2 Wire ABI layout

Per cell, `u8[12]`, flat index `side*6 + axis*2 + dir`
(`[side own,opp][axis Q,R,QR][dir +,−]`), values 0–5:

```
idx:  0        1        2        3        4        5        6..11
      own,Q,+  own,Q,−  own,R,+  own,R,−  own,QR,+ own,QR,− opp,(same)
```

- **Serve** (`payload.rs`): new payload key `"raylen"` = `bytes` of
  `Vec<u8>`, length `total_nodes * 12`, node-major in support row order,
  concatenated across rows exactly like `nbr`. `Row` gains
  `raylen: Vec<u8>`. ABI_VERSION bumps 1 → 2.
- **serve_pack.rs**: `plane_buffer!` u8 buffer (the existing `U8Buf`), padded
  per group to `g * pad_to * 12` with pad fill 0; `pack_groups` grows a
  `raylen_bytes: &[u8]` argument (length-validated `total_nodes * 12`) and the
  group dict gains `"raylen"`.
- **batching.py** (`collate_rows`): `raylen = torch.zeros(b, npad, 12,
  dtype=torch.uint8)`; filled per row; pad rows stay 0 (D-S13). Keyed
  `"raylen"` in the batch dict. Threaded to the model only under an L layout.
- **Train** (`replay_expand.rs`): recomputed from the transformed placements —
  derived data, **no shard schema change** (same rationale as the graded
  planes).

### 2.3 Rust signatures (as built)

```rust
// features.rs — the shared walk, one cell. Generic over i32-coord closures so
// one core serves both modules' Support/board types: `on_support` is support
// membership, `owner` the stone owner (player index 0/1), `me` the side to
// move's player index.
pub(crate) fn ray_length_row<S, O>(
    on_support: &S, owner: &O, xq: i32, xr: i32, me: u8,
) -> [u8; RAYLEN_SLOTS]
where S: Fn(i32, i32) -> bool, O: Fn(i32, i32) -> Option<u8>;

// features.rs — serve entry (mirrors build_features): (N * 12) node-major.
pub fn build_ray_lengths(state: &RustHexoState, sup: &Support) -> Vec<u8>;

// replay_expand.rs — train entry: the same core over the reconstructed
// owner map (records = the D6-transformed placement facts).
fn build_ray_lengths(
    sup: &Support,
    records: &[(i32, i32, u8, u32)],
    current_player: i32,
) -> Vec<u8>;
```

Both delegate to `ray_length_row` so serve/train cannot drift (the
`window_feature_row` precedent). The serve closures wrap
`state.board().get(HexCoord)`; the replay closures read a
`HashMap<(i32,i32), u8>` built from `records`. Constants `RAYLEN_SLOTS = 12`,
`RAY_REACH = WINDOW_LEN - 1 = 5` live in both `constants.rs` and
`constants.py`.

### 2.4 Python oracle (`features.py`)

```python
def ray_lengths_for_cell(
    owner_at: dict[tuple[int, int], int],
    support: set[tuple[int, int]],
    xq: int, xr: int,
    me: int, other: int,
) -> list[int]:
    """u8[12] ray lengths for one cell, index order [side own/opp][axis
    Q,R,QR][dir +,-]; a literal transcription of the Rust walk (L1)."""
```

Axis vectors from the existing `_AXES`; the 3-way parity test compares this
against both Rust paths elementwise-exactly (u8 — no tolerance).

### 2.5 D6 covariance of the data (the parity test's oracle relation)

Rays are recomputed from the transformed board, so for every `g`:

```
raylen[g·x, s, σ_g(a), dir_g(a, dir)] == raylen[x, s, a, dir]
```

where `σ_g` is the axis permutation (`cosp[g]`) and `dir_g` accounts for the
direction flip: `g` maps the ray direction `±a.vector()` to
`±' σ_g(a).vector()` with `±'` read off `apply_d6(g, a.vector())` (rotations
by 60°·k flip the sign for k ∈ {2,3,4} depending on axis; reflections
likewise — the test computes `dir_g` directly by transforming the direction
vector and matching it against `±σ_g(a).vector()`, no hand table). The `side`
index is g-invariant.

---

## 3. Phase L1 — `head_perm6`, `RayAttnBlock`, `bias_theta_l`

### 3.1 `head_perm6` (equivariant.py) — construction and the L4 trap

Channel order: **coset-major, then orbit-half, then K-slot, then orbit
channel**. With `HALF = C_ORBIT // 2` (C_ORBIT must be even — 16 ✓):

```python
@functools.lru_cache(maxsize=1)
def head_perm6() -> torch.Tensor:
    G = build_group()
    cosets = G["cosets"]
    order = [
        slot * C_ORBIT + half * (C_ORBIT // 2) + o
        for h in range(N_AXES)             # coset (win axis)
        for half in range(2)               # orbit-half: 0=own, 1=opp
        for slot in cosets[h]              # the 4 K-slots of the coset
        for o in range(C_ORBIT // 2)       # orbit channel within the half
    ]
    return torch.tensor(order, dtype=torch.long)
```

so `reshape(..., heads=6, head_dim=2*C_ORBIT)` lands head `h6 = 2*coset +
half` on the channels `{slot*C_ORBIT + half*HALF + o : slot ∈ coset_h, o <
HALF}`. `head_perm6_inv` by the same scatter as `head_perm_inv`.

**Why the split must be along the orbit index and NEVER the K-slots.**
`M(g) = ρ_reg(g) ⊗ I_{C_ORBIT}` acts on a channel `(slot, orbit)` by
left-multiplying the slot and **fixing the orbit index**. Restricted to one
coset's block, the within-block action is `P_K(g)` — a permutation of the 4
K-slots (derivation (H)). Any partition of the *orbit* index therefore
commutes with the action ( it is untouched), so each orbit-half sub-head is a
`D6`-block: head `2c+s` maps to head `2·cosp[g](c)+s` with an internal
`P_K(g) ⊗ I_HALF` permutation, and per-head dot products are preserved. A
K-slot split instead partitions `{k0,k1,k2,k3}` into two pairs; K acting on
itself by left multiplication is **simply transitive** (regular), so for any
pair `{a, b}` there exists `k ∈ K` with `k·a ∉ {a·, b·}`-side of the
partition — concretely for `K = {e, r³, g7, g10}` every non-identity element
moves `e` to a different slot, so *every* 2+2 partition is broken by some
`P_K(g)`: channels cross the sub-head boundary, the head partition is not a
block system, and equivariance breaks silently (no shape error). This is the
plan's risk-register item 1; the structure unit test (§4, L1-t2) asserts the
conjugated action `head_perm6 ∘ M(g) ∘ head_perm6⁻¹` is block-structured over
the 6 heads with the predicted head permutation `2c+s → 2·cosp[g](c)+s`.

The `constants.py:191-197` `ATTENTION_HEADS == 3` import check is relaxed to
per-block-type: A blocks stay 3 (structural); L blocks use `6` internally
(derived, not env-configurable — a new `RAY_HEADS = 6` constant, plus a
`C_ORBIT % 2 == 0` validation when the layout contains `L`).

### 3.2 `bias_theta_l`

Per L block: `nn.Parameter(torch.zeros(n_joint_classes, 2))`, expanded to the
`(BIAS_ROWS, 6)` table every consumer sees by

```python
bias_table_L[row, h6] = bias_theta_l[block][joint_of_row_head[row, h6 // 2], h6 % 2]
```

— the joint (row, coset) LUT is REUSED unchanged (all ray offsets have
hex-dist ≤ 5 < 8, inside the exact disk), and the side index `h6 % 2` is
group-invariant (L5), so the diagonal-action invariance (B) of §5 holds
per side slice. Token rows never appear in the L sequence (cells-only) but
their LUT rows are simply unused. Masked (non-ray) pairs get additive
`PAD_KEY_MASK_VALUE = -3e4` (fp16-finite, `model.py:64` convention), applied
together with the pad-KEY fill.

### 3.3 `RayAttnBlock` (cells-only)

Mirrors `AttnBlock` (pre-norm, LayerScale, MLP `MLP_RATIO*C`) minus the token
rows; sequence length `Npad`. Two paths:

- **Materialized reference:** extend the `_build_pair` machinery — the pair
  `(dq, dr)` tensors are already computed; the live-ray test per (i, j) for
  head `h6 = 2c + s` is

  ```
  aligned_c(dq, dr, kk):  c=Q: dr==0, kk=dq | c=R: dq==0, kk=dr | c=QR: dr==-dq, kk=dq
  live(i,j,h6) = (i == j) or (aligned_c and 1 <= |kk| <= 5
                              and |kk| <= raylen[b, i, s, c, 0 if kk>0 else 1])
  ```

  bias = `bias_table_L[row(i,j), h6]` where live, else `PAD_KEY_MASK_VALUE`;
  produced as a `(B, 6, N, N)` additive tensor consumed by the same
  sdpa/materialized impls.
- **Flex score_mod:** closure over `(coords, raylen, table_L)` exactly
  parallel to `_FlexBias` (no token clamping — cells only):

  ```python
  def score_mod(score, b, h, q_idx, kv_idx):
      dq = coords[b, kv_idx, 0] - coords[b, q_idx, 0]
      dr = coords[b, kv_idx, 1] - coords[b, q_idx, 1]
      side = h % 2
      coset = h // 2
      kk = torch.where(coset == 1, dr, dq)          # R reads dr; Q/QR read dq
      aligned = torch.where(
          coset == 0, dr == 0, torch.where(coset == 1, dq == 0, dr == -dq)
      )
      fwd = kk > 0
      reach = raylen[b, q_idx, side * 6 + coset * 2 + torch.where(fwd, 0, 1)]
      live = (q_idx == kv_idx) | (
          aligned & (kk != 0) & (kk.abs() <= 5) & (kk.abs() <= reach)
      )
      biased = score + table_l[lut_row(dq, dr), h].to(score.dtype)
      dead = ~live | is_pad_key(b, kv_idx)
      return torch.where(dead, biased + PAD_KEY_MASK_VALUE, biased)
  ```

  (`lut_row` is the existing `_cell_bias_lut` gather; `RAY_BLOCKERS=0` (L6)
  swaps `reach` for a constant 5 — geometric rays, computable from coords
  alone.) The bespoke Triton pair kernel is NOT extended in v1 (plan §2.2);
  L blocks run flex/materialized until the L3 perf pass.

Empty-row safety: the diagonal is always live and every live query row has
its own cell as a key, so the masked softmax never sees an all-masked row;
pad query rows carry garbage as everywhere else and are re-zeroed by the
post-block mask multiply.

### 3.4 Trunk integration

Layout grammar gains `L` (`constants.py` validation set `{"C","A","L"}`,
still must end `A`); `HexfieldNet` builds `self.ray_blocks` in layout order;
the walk's L arm mirrors the C arm's register attachment (L is a non-A
block — R6):

```python
            elif kind == "L":
                if self._reg_tok_read:
                    x = x + self.tok_reads_l[li](tokens) * mask.unsqueeze(-1)
                x = self.ray_blocks[li](x, ray_bias(li), mask)
                if self._reg_lane:
                    tokens = self.registers_l[li](tokens, x, mask)
                li += 1
```

(L-block register/tok-read ModuleLists are separate (`registers_l` /
`tok_reads_l`) so C-block indices stay dense; both are Phase-L work, not R0.)
Meta gains `trunk_layout` (already present), `ray_blockers`, and the L-head
config (`ray_heads: 6`).

---

## 4. Test list and acceptance criteria

Phase R0 — `tests/test_hexfield_eq_register_lane.py` (NEW; equivariant
default build, self-skips under GROUP_ORDER != 12):

| id | test | acceptance |
|----|------|------------|
| R0-a | full-net equivariance, `reg_lane=True` (and `+reg_tok_read=True`), every param randomized N(0, 0.3), all 12 g | covariant heads permute / invariant heads fixed to atol 1e-4 fp32 (the existing gate's tolerance) |
| R0-b | toggle-off identity | `HexfieldNet()` state-dict key set contains no `registers.`/`tok_reads.` keys and equals the explicit `reg_lane=False` build's key set; same-seed forward outputs bit-identical (`torch.equal`) |
| R0-c | zero-init identity | lane-on net with the vanilla net's shared params copied in produces `torch.equal` outputs for every head at init (out_proj/tok_reads zero ⇒ exact no-op) |
| R0-d | grads reach the lane | randomized params: every `registers.*`/`tok_reads.*` param gets a nonzero grad after one backward; PLUS at zero-init the `out_proj` base params still get nonzero grad (grow-in is trainable) |
| R0-e | counting probe | isolated `RegisterRefresh`, N fixed, k ∈ {1,2,4,8} duplicated pattern cells: ‖upd(k) − upd(0)‖ ≈ k·‖upd(1) − upd(0)‖, rtol 0.1 (sum-shape sanity; linear in k because the aggregation is an unnormalized sum) |
| R0-f | predicate classification | plugin AdamW + prefit AdamW: every table-§1.4 name in its intended decay/no-decay group; `trainer._build_grad_norm_groups` puts every lane param in `trunk_reg` and nothing else there |

Existing gates that must stay green: `test_hexfield_eq_equivariance.py` (all),
`test_hexfield_eq_derivation.py`, `test_hexfield_eq_orbit_bias.py`
(GROUP_ORDER=1), `test_hexfield_eq_checkpoint_meta.py`,
`test_hexfield_eq_rust_parity.py`, `test_hexfield_eq_smoke.py`.

Phase L0 — extend `tests/test_hexfield_eq_rust_parity.py` idioms (new file
`tests/test_hexfield_eq_raylen_parity.py`):

| id | test | acceptance |
|----|------|------------|
| L0-t1 | 3-way parity | serve Rust (`_rust` featurize path) ≡ train Rust (replay_expand) ≡ Python oracle, elementwise EXACT (u8) over the sampled decision states + the crafted line game |
| L0-t2 | all-12 D6 | `raylen[g·x, s, σ_g(a), dir_g] == raylen[x, s, a, dir]` for all g, both backends |
| L0-t3 | wire round-trip | payload `raylen` bytes → batching pad → per-row slices equal the per-state Rust output; pad rows all-zero |
| L0-t4 | blocker semantics | crafted board: an anti-side stone at distance j caps the ray at exactly j (included); an own stone does not; off-support truncates |

Phase L1:

| id | test | acceptance |
|----|------|------------|
| L1-t1 | full-net equivariance under an L layout (all 12 g, randomized params) | atol 1e-4 (the test class that catches a wrong head_perm6) |
| L1-t2 | head_perm6 structure | conjugated `M(g)` is block-diagonal over the 6 heads; head permutation equals `2c+s → 2·cosp[g](c)+s`; a deliberate K-slot-split permutation FAILS the same assertion (negative control) |
| L1-t3 | materialized ≡ flex | L-block outputs agree to 2e-4 fp32 |
| L1-t4 | empty-row safety | lone-stone board: no NaN, diagonal live |
| L1-t5 | grads reach `bias_theta_l` | both columns (sides) of every class touched by a coverage board get nonzero grad |

---

## 5. Build / run facts

- R0 is pure-Python — NO Rust rebuild.
- L0 rebuild: WSL, `hexfield-dev` venv, `maturin develop --release -m
  packages/hexfield_eq/Cargo.toml` (the `scripts/_rebuild_hexfield.sh` pattern,
  pointed at hexfield_eq). The editable install writes the built
  `_rust.cpython-312-x86_64-linux-gnu.so` straight into
  `packages/hexfield_eq/python/hexfield_eq/` (python-source layout), so no
  separate mirror `cp` is needed — verify the in-tree .so mtime after the
  build.
- Tests: WSL `hexgt-build` venv,
  `PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:packages/hexo_utils/python:packages/hexo_train/python`,
  `CUDA_VISIBLE_DEVICES=` (CPU gates), `python -m pytest tests/<file> -q`.
  The passthrough suites (`orbit_bias`, `smoke`) additionally set
  `HEXFIELD_EQ_GROUP_ORDER=1`.
- Gathered ray kernel (D-S36/D-S37): pure Python/Triton, no Rust rebuild.
  `tests/test_hexfield_eq_triton_ray.py` — CPU section under the battery env
  above; the CUDA section needs the GPU visible (tiny shapes, safe next to a
  running prefit; it self-skips after an OOM retry). Bench ONLY in an idle
  window: `python scripts/bench_eq_ray_kernel.py` (refuses at > 10% GPU util);
  serve knob `HEXFIELD_EQ_TRITON_RAY=1` (default 0 — do NOT combine with the
  CUDA-graphs serve path, the index build syncs), tile knobs
  `HEXFIELD_RAY_BM`/`HEXFIELD_RAY_WARPS`.
