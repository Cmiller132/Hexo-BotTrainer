# main_6 serve speedup — triton hex-conv + flex-pair score_mod (2026-07-02)

**Tree:** dev repo (`E:\Hexo-BotTrainer-hexgt`), NOT yet deployed to the live
gumbel worktree. **GPU:** RTX 4070 Ti. **torch:** 2.12.0+cu130.
All changes are env-flag-gated, default OFF; flags-off outputs are bit-identical
(parity self-check 0.0).

## Why: profile at main_6 shapes

The 2026-06-16 flex handoff profiled c=96 / S≈1900. The live main_6 regime is
c=128, radius-4, support N mean ≈ 362 (p90 514, max ~673), serve groups of
~100–250 rows. Compiled serve forward at N=384, B=247 (41.3 ms, 167 µs/state):

| component | ms/fwd | share |
|---|---|---|
| conv gathers (17 HexNodeConvs materializing (B,N,7C)) | 12.45 | 30% |
| flex attention kernel (score_mod does 2 int64 coord gathers + int64 LUT gather + 3 wheres per score) | 10.26 | 25% |
| GEMMs (conv 8.0 @ ~46 TFLOPS, near roofline) | 10.09 | 24% |
| LayerNorm (fp32 stream) | 4.53 | 11% |
| fused pointwise | 4.26 | 10% |

`scripts/_hexfield_main6_profile.py` reproduces this (realistic blob supports,
serve-style dynamic compile).

## What shipped (dev repo, flag-gated)

1. **`HEXFIELD_FLEX_PAIR=1`** (`model.py`): precompute the per-pair bias-table
   row index ONCE per forward as (B,S,S) uint8 (int16/int32 intermediates, no
   int64 materialization), shared by all 3 attention blocks; fold the pad-KEY
   fill into an appended 238th table row. score_mod = one 1-byte gather + tiny
   table read. Flex kernel 10.26 → 5.55 ms.
2. **`HEXFIELD_TRITON_CONV=1`** (`model.py`, `_triton_conv.py`): fused
   gather+GEMM Triton kernel for HexNodeConv as the `hexfield::hex_conv`
   custom op (in-graph under the dynamic serve compile, no graph breaks).
   The (B,N,7C) gathered tensor is never built; missing-neighbour handling is
   a masked load; the row mask is the epilogue. Serve-only (no backward):
   no-grad + CUDA + 16-aligned channels (stem C_in=15 keeps the reference
   path). gather 12.45 → ~0 ms; all 17 convs run in 8.98 ms fp32-in /
   5.94 ms fp16-in (~62 TFLOPS effective). BM=32 tile variant for small
   flushes (<32k rows).
3. **`HEXFIELD_SERVE_HALF=1`** (`inference.py`): fp16 COPY of the net (master
   weights untouched; evaluator is rebuilt each epoch so it tracks training),
   autocast disabled so the residual stream stays fp16 (halves LN/pointwise
   traffic). Value/moves-left tops kept fp32 (head-boundary dtype cast in
   `forward_policy_value`). **Above the shipped parity gate — see below.**
   Requires f16 feats (default).

Also: `scripts/_hexfield_serve_ref.py` now honors `HEXFIELD_REF_RUN`.

## Forward benchmarks (compiled, main_6 shapes)

ms/fwd (µs/state):

| N (B) | baseline | +flex_pair | +triton | both | both+half |
|---|---|---|---|---|---|
| 256 (256) | 26.3 (103) | 22.4 (88) | 18.0 (70) | **16.0 (62)** | **10.9 (42)** |
| 384 (247) | 41.3 (167) | 35.2 (143) | 28.6 (116) | **26.0 (105)** | **18.5 (75)** |
| 512 (140) | 32.2 (230) | 27.1 (194) | 23.1 (165) | **20.1 (144)** | **14.5 (104)** |
| 640 (90)  | 26.5 (295) | 22.3 (248) | 19.1 (212) | **16.7 (185)** | **12.7 (141)** |

**both = 1.59–1.65× forward; both+half = 2.09–2.42×.**

(A plain `.half()` WITHOUT the triton conv is a REGRESSION — inductor's fp16
gather codegen is worse than its fp32→fp16 fused-cast gather: 45.5 ms vs 41.3.
half only pays after the gathers are gone.)

## Parity (real main_6 states + epoch_000071 ckpt, `_main6_parity.sh`/`2.sh`)

| config | max\|dvalue\| | max\|dprior\| | gate 3e-3 |
|---|---|---|---|
| flags off vs saved baseline | 0.0 | 0.0 | bit-identical |
| flex_pair | 1.95e-3 | 9.6e-4 | PASS |
| triton_conv (+flex_pair) | 2.58e-3 | 1.0e-3 | PASS |
| + serve_half | 4.7e-3 | 2.3e-3 | **FAIL (value)** |

serve_half's value drift is accumulated fp16 trunk-stream rounding (fp32 value
tops did NOT recover it) — no surgical fix; shipping it is a run-quality
judgment call (priors pass; 4.7e-3 on [-1,1] is far below search noise, but it
exceeds the gate the flex ship established).

## End-to-end (real continuous scheduler, `_hexfield_selfplay_throughput.py`)

Shallow (96 games, 30 plies, vbs=4): OFF 23.2 pos/s → triton+flex_pair 27.0
(+16%) → +half 29.8 (+29%). Serve ms/state 0.171 → 0.139 → 0.115.
GPU idle grows 29% → 39% as the forward shrinks (host/scheduler becomes the
limiter — that's the SH-batching lever, not a kernel lever).

Deep STARVED regime (48 games, vbs=4, mean flush ~110): the flags do NOT pay
(triton+flex_pair 12.9 vs OFF 14.2 pos/s) — small groups underutilize both the
kernel and the GPU, and the run is scheduler-bound (idle ~50%). The BM=32 tile
variant was added after this measurement. The live regime (vbs=16,
active_limit=192, mean flush ~1088) is NOT this regime.

**Live-like (192 games, 60 plies, vbs=16, flush 1024, mean flush ~1670 — the
regime that matches the live supervisor):**

| config | pos/s | serve ms/state | GPU busy |
|---|---|---|---|
| OFF | 28.67 | 0.163 | 82% |
| triton + flex_pair | **38.61 (+35%)** | 0.111 | 76% |
| + half + rust_pack | **46.62 (+63%)** | 0.085 | 70% |

(pos/s here are bench-absolute, not live pos/s — the bench skips per-decision
record/sample work — but the ratios are the honest serve-lever effect at live
batching. Applied to the live epoch: self-play 1433 s → ~1060 s (safe) /
~880 s (full stack); epoch wall ≈ −20% / −30%.)

## Deployment record (2026-07-02, late)

**DEPLOYED — full stack.** The operator accepted serve_half's 4.7e-3 value
parity ("reasonable to assume it is okay"); the every-5-epoch SealBot
multistage eval guards strength.

- Code edits hand-applied to the live worktree (`E:\Hexo-BotTrainer-gumbel`
  `model.py` / `inference.py`, `_triton_conv.py` copied) — the worktree carries
  original (non-neutralized) comments, so a wholesale file copy was NOT used.
  Worktree full-stack parity re-verified from the worktree itself:
  4.73e-3 / 2.28e-3 — identical to the dev tree (faithful port).
- `hexfield-supervisor-6.service` gained `HEXFIELD_TRITON_CONV=1`,
  `HEXFIELD_FLEX_PAIR=1`, `HEXFIELD_SERVE_HALF=1`, `HEXFIELD_RUST_PACK=1`
  (with a provenance comment). Partial epoch 72 wiped per procedure; supervisor
  restarted clean: flags confirmed in the trainer env, 0 CantSplit /
  0 InductorError.

Training path (grad) is untouched by all four flags; eval arena inherits the
env and serves with the same parity class. The dev-repo copies of the same
changes are the canonical source (uncommitted at time of writing).

**Live verdict (epoch 72, first round-1 epoch):** 18.64 pos/s vs 13.75
baseline (+36%), 6.1k evals/s (+39%), clean health stats.

---

# ROUND 2 (2026-07-03): allocator + copy-stream + train-compile + vbs

Deep profile of the post-round-1 serve (phase timing added to the Rust
scheduler — `select/submit/finish/backup/complete_seconds` now in the
scheduler stats): **submit_payload was ~95% of scheduler wall**, decomposed by
py-spy --native + strace into (a) glibc allocator churn (per-flush
mmap/munmap + heap trim ≈ 40% of host samples), (b) pageable-H2D stream
serialization (submit's duration tracked the flush's device time exactly),
and (c) python dispatch pacing.

Shipped (live-like bench, epoch-72 ckpt, vbs baseline 47.8 pos/s):
- `virtual_batch_size` 16→48 (toml): +6%. SH round quotas/halving unchanged
  (barriers check backed-up visits); only within-round staleness grows.
- glibc malloc tunables (`MALLOC_TRIM_THRESHOLD_`/`MMAP_THRESHOLD_`
  =512MB, `TOP_PAD_`=128MB, unit env): +12%.
- `HEXFIELD_COPY_STREAM=1`: pinned staging ring + dedicated copy stream +
  per-group events (`_PinnedRing`, inference.py); submit becomes a true
  enqueue. +3%, parity bit-identical. (First cut hit the classic
  cross-stream allocator UAF → NaN; fixed with `record_stream`.)
- `HEXFIELD_TRAIN_COMPILE=1` (trainer.py): compiled training forward,
  `maybe_mark_dynamic(B)` + `mark_static(Npad)` → one graph per PAD_QUANTUM
  multiple. 1.30× on the production shape mix; warmup ~105s amortizes in the
  long-lived trainer. (Deploy note: a function-level `import torch._dynamo`
  initially shadowed `torch` → startup crash-loop → breaker halt; fixed,
  module-level import.)

Cumulative bench: 47.8 → 58.0 pos/s (+21% over round 1; ≈ +64% over the
pre-round-1 baseline).

Measured and REJECTED (documented so nobody re-chases them):
- depth-2 pipeline + complete-overlap: −5%/−9% (wider staleness, smaller
  flushes).
- `HEXFIELD_PAIR_CEILING` raise: no-op (WASTE_FRACTION binds grouping first).
- `HEXFIELD_WASTE_FRACTION` raise: −24%/−43% (pad keys cost full S²
  attention with block_mask=None).
- mimalloc / jemalloc LD_PRELOAD: both lose to tuned glibc (53.1/48.5 vs 56.5).
- OMP/rayon thread knobs: null (the futex/yield census was idle-pool noise).
- `HEXFIELD_TRAIN_PAIR_BUDGET` raise (8e7): 4.7× WORSE per row (flex backward
  memory pressure).
- shape-keyed flex compile instances: perf-neutral (kept — bounded guard
  chains).
- `HEXFIELD_GATE_COMPLETE` (skip idle complete scans): decision-DIVERGED
  (complete also activates games); default off, do not enable.
- **CUDA graphs** (`HEXFIELD_CUDA_GRAPHS`, `_GraphCache`): fully built,
  parity bit-identical, perf-neutral on the bench (python dispatch paces the
  GPU either way; captures amortize only over epochs). Kept default-OFF.

Remaining ceiling: python dispatch ≈ GPU pace at ~58 pos/s bench (~80% GPU
busy). The next frontier is moving the submit loop (pack→H2D→forward
dispatch) out of Python entirely (Rust/C++ + CUDA graphs), est. +15-20%.

New instrumentation that persists: scheduler phase timing (`*_seconds` in
selfplay scheduler stats), `_hexfield_selfplay_throughput.py` grew
vbs/eval-stats/phase output, `_main6_*probe*.sh` bench harnesses,
`_pyspy_agg.py`.

**Live verdict (epoch 73, first round-2 epoch):** self-play **24.15 pos/s**
(vs 18.64 round-1, 13.75 baseline = **+76% cumulative**), self-play wall
1433 → 797s, 7,985 evals/s, mean flush 1,973, health stats clean
(entropy/losses/audit all normal). Deploy note: the first
HEXFIELD_TRAIN_COMPILE trainer had a function-level `import torch._dynamo`
that shadowed `torch` → startup crash-loop → breaker halt; fixed
(module-level import), ~10 min lost. Training was 443s on epoch 73
(one-time compile warmup for the shape set); epoch 74+ is the steady-state
number. Early-stop counters shifted under vbs=48 (fewer stop
opportunities with more leaves in flight) but evals/decision stayed ~330 —
the accounting moved, not the work.
