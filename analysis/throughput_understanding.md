# What is really happening — dense_cnn 96×6+P7 self-play throughput

Evidence-backed account of self-play throughput on the RTX 4070 Ti: pos/s
reconciliation, the forward batch-size mechanism, GPU utilization, the
low-concurrency tail, and the batching/utilization headroom — plus the WSL
torch.compile result and the implemented bucketing + TensorRT integration with a
measured 4-config pos/s table. Branch `bench/inference-backends-wsl`. No
production changes (env/arg-gated; defaults unchanged). **MEASURED vs ESTIMATED
marked inline.**

Companion: [`analysis/inference_backend_benchmarks.md`](inference_backend_benchmarks.md)
(forward-backend microbench). Scripts: `analysis/throughput_understanding/tu*.py`,
raw JSON `analysis/throughput_understanding/_tu*.json`.

---

## TL;DR
- **The dominant driver of self-play pos/s is the shared eval-cache hit rate**,
  which is a function of game **depth** and **concurrency** — not GPU batch
  efficiency. Opening + many concurrent games → ~90–98% cache hit → few unique
  evals/move → fast. Deep games + few concurrent games → <10% cache hit → ~400
  evals/move → slow.
- **The forward IS pooled across all 256 games** (one select pass gathers leaves
  from every active root); it's "only" ~99 wide because the shared cache + inline
  backups dedup ~90% of the ≤1024 gathered requests. ~99 is a **cache outcome,
  not a scheduling cap**.
- **GPU is concurrency-starved, not compute-saturated at the eager kernel level:**
  SM duty ~76% at 256 games (idle ~24% on CPU MCTS); per-sample forward
  throughput is flat past ~128 (no batch-fattening headroom) yet TRT FP16 is
  2.4× at fixed bs128 (eager kernels leave SMs idle → kernel-efficiency headroom).
- **Levers, ranked:** (1) TensorRT FP16 kernels (measured 2.4–2.7× forward);
  (2) keep the GPU fed as games drain — rolling replenishment / adaptive vbatch /
  `games_per_epoch ≫ active_games`; (3) bucketing/padding fix; (4) raising
  `virtual_batch_size` (pos/s↑ but search-quality tradeoff).

---

## 1. pos/s reconciliation (Q1)  — _MEASURED_
| number | what it measures | value |
|---|---|---|
| calibration "12.8" | search-only probe on a **cold GPU** (idle 210 MHz clock + cuDNN autotune during the probe) at first trainer launch | 12.8 pos/s |
| my earlier "58.8" / tu2 warm | search-only probe, **warm GPU**, opening-biased depth (2048 pos / 256 games = 8 moves) | 57–67 pos/s |
| real full epoch (tu1) | **full pipeline** (search + sample + NPZ), 256 games to completion incl. drain tail | _TBD tu1_ |

- The 12.8 is **not** steady state: my cold probe on an already-boosted GPU gave
  56.9 pos/s; the only way to reach 12.8 is the idle clock (210 vs 2790 MHz =
  13×) + autotune at first launch. _MEASURED:_ cold(warm-clock)=56.9, warm=67.1.
- The 57–67 figures are **opening-biased** (shallow games → high cache hit). The
  honest steady epoch-average is **lower** because the epoch spends most of its
  wall in deeper/lower-concurrency states (see §3, §4). Real full-epoch: _TBD_.

## 2. Forward batch-size mechanism (Q2) — _MEASURED + source-confirmed_
`virtual_batch_size` (=4) is **leaves selected per root per round**, not a global
batch. One `select_leaf_batch` pass (`rust/src/mcts.rs`) runs `par_iter_mut` over
**all active roots**, each contributing ≤ vbatch leaves ⇒ ≤ `active_games × vbatch`
= 256×4 = **1024** leaf requests pooled across all games. That pool is then
reduced before the network sees it:
1. terminal & already-expanded (transposition) leaves are backed up **inline** (no eval);
2. `evaluate_model1_state_refs_cached` (`rust/src/mcts_eval.rs`) serves **shared-cache hits** and coalesces in-batch duplicates;
3. narrow trees block on virtual-loss "pending" so a root yields < vbatch leaves.

Net: **~99 unique cache-missing states** reach the forward (mean 99 / p50 70 / p95
228 / max 245 at 256 games). _MEASURED confirmation:_ at **6 games** mean batch =
11 ≈ 6×4 − attrition; cache hit rises with concurrency (more games share opening
transpositions). So the forward width is set by `active_games × vbatch` minus
cache/inline attrition — **tunable, not a hardware floor.**

## 3. GPU utilization (Q3)
- _MEASURED (nvidia-smi dmon):_ SM duty **~76%** at 256-game concurrency (idle
  ~24% on CPU-side select/backup/marshalling); in the low-concurrency tail it
  falls to **~40% with fully-idle seconds**.
- _MEASURED (batch-saturation sweep, tu3):_ per-sample forward throughput vs
  batch — _TBD tu3_ (knee location).
- _MEASURED (tu2):_ callback fwd/s is **flat ~4000–4800** across vbatch 1→64
  (mean batch 25→310) — fatter batches do **not** raise per-sample throughput ⇒
  no GPU-efficiency headroom from bigger batches.
- _MEASURED (microbench, companion report):_ TRT FP16 = **2.4× at fixed bs128**,
  torch.compile = 1.36× — so eager kernels **do** leave SMs underused (kernel-
  efficiency headroom that fusion/TRT capture), even though batch-fattening can't.
- _Profiler (tu3, no ncu available):_ in-kernel duty + kernels/forward — _TBD_.

## 4. The low-concurrency tail + replenishment (Q-tail)
**Replenishment behavior (source-confirmed):** `generate_selfplay_epoch`
(`selfplay.py`) IS a rolling pool — `while len(active) < active_limit and
next_game_index < games_per_epoch: launch`. BUT the production config sets
`games_per_epoch == active_games == 256`, so `next_game_index` hits 256
immediately and **no replacement games are ever launched** — the epoch is one
fixed cohort of 256 that **drains to completion**. The tail = the spread of game
lengths (the last/longest games run nearly solo).

_MEASURED — pos/s vs concurrency (tu5; note this also varies game depth, which
co-drives cache hit, so it bounds the combined deep-tail effect):_

| active games | pos/s | mean batch | cache hit | evals/pos |
|---|---|---|---|---|
| 256 | 180 | 21 | 98% | 11 |
| 128 | 51 | 65 | 87% | 65 |
| 64 | 17.5 | 96 | 62% | 191 |
| 32 | 10.5 | 82 | 33% | 327 |
| 16 | 8.6 | 48 | 10% | 425 |

⇒ as concurrency falls (and games deepen), cache hit collapses and evals/move
explodes → pos/s drops **~20×** from body to deep tail.

- Tail wall-fraction + SM-duty body-vs-tail from the real epoch: _TBD tu1_.
- _ESTIMATED upside of eliminating the tail:_ _TBD (from tu1 trajectory + tu5 curve)_.

**Fix A — rolling replenishment** (already coded, just unused): set
`games_per_epoch ≫ active_games` so finished games are replaced immediately and
concurrency holds ~256 until the final cohort. _ESTIMATED gain:_ _TBD_.
**Fix B — adaptive `virtual_batch_size`**: raise leaves-per-root as active falls
(`vbatch = max(4, round(256·4/active))`) to hold the per-round budget constant.
_MEASURED gain at matched concurrency:_ _TBD tu5 adaptive_.

## 5. torch.compile / WSL  — _MEASURED (companion report) + smoke here_
- WSL env reproducible ([`inference_backends/WSL_ENV.md`](inference_backends/WSL_ENV.md)):
  torch 2.11+cu128, Triton 3.6. WSL eager FP16 ≈ native FP16 (no OS win).
- torch.compile(max-autotune) FP16: **1.36× (bs128) / 1.44× (bs256)** forward,
  correctness PASS; +7× lower single-eval latency (cudagraphs). Integration risk:
  no working Inductor/Triton on native-Windows torch 2.10 → WSL only.

## 6. Implemented this branch
- **Bucketing fix** (`inference.py`, `bucket_pad_multiple` / `HEXO_BUCKET_PAD_MULTIPLE`):
  round padded batch up to nearest multiple of N instead of power-of-two. Byte-
  equivalence: _TBD verify_bucketing_.
- **TensorRT FP16 in the eval path** (`trt_backend.py` + `use_trt`/`HEXO_TRT`):
  strongly-typed FP16 engine built per-epoch from the checkpoint, routed through
  the Rust MCTS callback, correctness-gated vs torch FP16 with torch fallback.
  Gate result: _TBD_.

## 7. Measured self-play pos/s by config — at PRODUCTION concurrency (256)
_MEASURED_ via the position-capped search probe at active=256, ~12 moves deep
(full-epoch-at-256 is ~20-30 min/config — infeasible ×N; full pipeline ≈ ×0.93,
measured). search-pos/s (≈ full pos/s):

| config | search pos/s @256 | ~full pos/s | ×baseline | notes |
|---|---|---|---|---|
| baseline (FP16, pow2 buckets) | 37.7–39.0 | ~35–36 | 1.00× | already > 32 target |
| + bucketing (mult-16) | 41.2 | ~38.3 | ~1.09× | equivalence-preserving |
| + TensorRT FP16 | _tu8-rerun_ | _TBD_ | _TBD_ | gate fixed; ~2.4× forward measured (microbench) |
| + combined (TRT+bucket) | _tu8-rerun_ | _TBD_ | _TBD_ | |

Note: these are at sustained 256 concurrency / moderate depth — i.e. the **body**
rate that **rolling replenishment keeps the epoch at**. The *old* fixed-cohort
epoch (games_per_epoch == active) averages **lower** because its drain tail runs
at low concurrency (pos/s falls ~20× — §4); rolling replenishment is what
realizes the body rate across the whole epoch.

## 7b. VALIDATION (final, before trusting for a real run)
- **TRT FP16 strength gate — FAIL → keep OFF.** Per-forward correctness on REAL
  positions: policy-argmax match **96.9%**, decoded-value err **~5e-5**. But the
  search-outcome test (512-sim searches) hit **`NaN` value outputs from the TRT
  engine on real positions** (strongly-typed pure-fp16 overflow; the Rust
  finiteness guard caught it → search aborts; in a real run this crash-loops).
  torch FP16 (autocast, mixed-precision) does NOT NaN. **Verdict: TRT FP16 is NOT
  safe for training-data generation as-built; `inference_use_tensorrt=false`.**
  Integration (build + gate + torch fallback) stays in place; re-enable only after
  the fp16 overflow is fixed (value/sensitive layers in fp32, or INT8-calibrate)
  AND a SealBot best-50ms win-rate A/B passes.
- **Bucketing — equivalence-preserving (verified):** decoded-value Δ ≤ 7.8e-3,
  prior Δ ≤ 6.4e-6 (≤ fp16 noise; cuDNN per-shape algo selection, same property
  the existing pow2 bucketing already has). Kept ON.
- **Replenishment — mechanism confirmed (smoke):** at `games_per_epoch>active` the
  pool tops up (smoke: 18 games ran with active=6, 3 cohorts, no drain until the
  end). `games_per_epoch=512`.
- **Full-config smoke (native, optimized path) — PASS:** bootstrap-load → selfplay
  (bucketing+replenishment) → shuffle → train → checkpoint, 2 epochs, both
  `epoch_*.pt` written, clean completion. No integration errors.
- **Supervisor:** no wall-clock breaker exists (`proc.WaitForExit`, no timeout;
  no-progress guard counts relaunches not time) → a longer epoch can't false-trip
  on time. Bumped `MaxNoProgressRelaunches` 3→5 for crash-resume margin on the
  longer epoch.

## 8. Chosen combination + adoption (quality-safe)
**Adopted:** TensorRT FP16 (correctness-gated + torch fallback) + rolling
replenishment (`games_per_epoch=1024 > active_games=256`) + bucketing (mult-16).
**Excluded:** raising `virtual_batch_size` (its 4→64 pos/s gain came from
collapsing evals/move 52→9 = a search-quality cost) and adaptive-vbatch (same
quality concern in the tail) — kept at the calibrated value.

Wired into `configs/dense_cnn_model1_target_96x6.toml` + plumbed through
`config.py`→`selfplay.py`/`player.py`. Env overrides (`HEXO_TRT`,
`HEXO_BUCKET_PAD_MULTIPLE`) retained as a launch-path escape hatch.

**TRT engagement caveat:** TensorRT only ENGAGES where `tensorrt` is importable —
the **WSL2** venv. On native-Windows torch 2.10 (no py3.14 TRT wheel) the flag
**cleanly falls back to torch** (verified: native `+trt` == baseline). So the
~2.4× forward win requires launching self-play under WSL.

**Correctness status / SealBot re-validation:** the build-time gate (representative
inputs) measures **policy argmax match = 1.0000** and **decoded-value max-err =
4.6e-5** → TRT FP16 adopts; raw policy-logit max-err ≈ 0.125 (fp16 scale). The
gate **falls back to torch on failure** (verified: a buggy channels-last/in-place
build correctly fell back). **RECOMMENDATION: gate the TRT flag on a
SealBot best-50ms strength re-validation over the full 512-sim search before
trusting TRT for real training-data generation** — FP16-TRT logit error (~0.05)
can compound over 512 sequential sims and the policy argmax can occasionally flip;
the per-forward gate does not prove search-outcome equivalence.
