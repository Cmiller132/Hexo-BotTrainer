# Performance profiling — dense_cnn Model 1 (scratch_64) epoch cycle

**Date:** 2026-05-29  **Run:** `runs/dense_cnn_model1_scratch_64`
**GPU:** RTX 4070 Ti (12 GB), torch 2.10.0+cu126, AMP/channels_last, cudnn.benchmark
**Author:** profiling deep-dive. The live supervised run was **stopped by design** for the
optimization stage, so this pass runs **real GPU micro-benchmarks on a free GPU** against the
exact production network, evaluator, loss, and data path.

> **Method & honesty note.** Every number is tagged:
> **[M-log]** the run's own instrumentation; **[M-bench]** measured now on the free GPU against
> production code; **[M-model]** a measured rate (e.g. zlib MB/s) projected over a known data
> volume; **[Est]** reasoned estimate. Scripts + raw JSON are committed beside this file:
> `parse_epoch_timings.py`, `parse_selfplay_diag.py`, `reconstruct_epoch_timeline.py`,
> `gpu_microbench.py`, `evaluator_microbench.py`, `train_microbench.py`,
> `train_step_reconcile.py`, `train_step_pipeline.py`, `train_step_components.py`,
> `shuffle_mem_probe.py` (+ their `*_summary.json`).
>
> **This pass measured repeatedly and corrected two of its own earlier conclusions** — see §0.
> Repeated measurement matters: a single bench gave a training step of 260 ms that **four**
> later measurements showed was a transient artifact (true value ~465 ms).

---

## 0. Corrections (incl. to this report's own earlier drafts)

| Claim | Final measurement | Status |
|---|---|---|
| (orig draft) "Training step is a few ms of GPU; ~1 s/step is 20–100× off compute." | **~465 ms/step** of real GPU compute (forward+backward of the hex-conv trunk). | Corrected. |
| (rev 1) "Training GPU step = 260 ms." | 260 ms was a **non-reproducible transient** (boost-clock state right after freeing the GPU). Re-runs + 3 other methods = **457 / 466 / 472 / 486 ms**. | Corrected to ~465 ms. |
| (rev 1) "~Half the evaluator wall-time is Python marshaling/gather." | The evaluate path is **92–95% GPU forward**; legal-prior gather+softmax+D2H is **~6%**, value-decode ~1%. | Corrected: evaluator is GPU-forward-bound. |
| (orig) "Per-step `.cpu().item()` sync kills overlap; removing it is a win." | Per-step-synced **latency 466 ms** ≈ pipelined **throughput 461 ms** (5 ms gap). Sync is a non-factor **because the step is compute-bound**, not because overlap is hidden. | Confirmed non-factor (clean reason). |
| (orig) "Oversized FC policy head is the dominant GPU cost; shrinking it speeds things up." | Inference: head adds **0.26 ms/batch (~2%)**. Training: full model (12.2 M params) vs tiny conv-head variant (0.97 M) = **472 vs 473 ms** — identical. | Corrected: head is **quality-only**, not a speed lever. |
| (orig) "Cold 11 pos/s = cuDNN autotune *or* clock ramp (couldn't tell)." | **Proven cuDNN autotune:** first forward on a never-seen batch shape = **~925 ms** (vs 32 ms steady) under `cudnn.benchmark=True`; **0 ms** penalty with `benchmark=False`. ~925 ms × ~900 distinct shapes ≈ **830 s** ≈ the observed **842 s** cold-epoch penalty. | Resolved. |

The NPZ re-decompression bug (§4) and the per-phase budget (§2) from earlier drafts hold up and
are re-confirmed with fresh measurements.

---

## 1. TL;DR

1. **Per-phase budget of a steady epoch (epoch 21, reconciled to 0.2%):** training **41.9%**,
   self-play **32.6%**, SealBot eval **15.2%**, shuffle **10.2%**. Epoch ≈ **1143 s (19 min)**. [M-log]
2. **Self-play is CPU/Python-bound, not GPU-bound.** Per searched position (28.4 pos/s ⇒ 35.2 ms/pos):
   **orchestration 12.2 ms (35%), NN-evaluator 10.2 ms (29%), Rust MCTS tree 8.1 ms (23%), Rust
   encode 4.7 ms (13%)**. The GPU is busy ~29% of self-play, and the **raw GPU forward is only
   ~16%**. The evaluator itself is 92–95% GPU forward. [M-log + M-bench]
3. **The "11 pos/s" in NOTES is the cold-process number, fully explained: cuDNN `benchmark=True`
   autotuning** re-tunes on every new batch shape (~925 ms each). The evaluator sees ~900 distinct
   shapes; the first epoch of each process pays ~830 s of autotune. Setting `cudnn.benchmark=False`
   removes it at **identical** steady speed. Steady self-play is **28–34 pos/s**. [M-bench]
4. **Training is GPU-compute-bound at a ~465 ms/step floor** (391 steps ⇒ ~182 s/epoch
   irreducible for this net), dominated by the **trunk forward+backward**, not heads, optimizer,
   sync, or H2D. On top of it, a **data-loader bug wastes ~244 ms/step (~95 s/epoch)** of
   redundant zlib — a ~5-line, correctness-neutral fix. [M-bench]
5. **Shuffle (~116 s) is CPU-zlib-compress-bound** (recompress 538 MB/s vs decompress 2018 MB/s);
   peak RAM **0.87 GB** per 8000-row group over a **32.8 GB** window — well bounded. [M-bench/model]
6. **Highest-value fixes:** (P0) NPZ load-once **−~95 s/epoch, measured, free**; (PB)
   `cudnn.benchmark=False` **−~830 s on every cold/relaunch epoch, measured**; (P3) uncompressed
   shuffle scratch **−~40 s/epoch**. The FC-head rework is **quality-only**. Raising MCTS sims is
   the quality lever and roughly **doubles** the epoch (linear in sims). [M-bench/model]

---

## 2. Per-phase wall-clock budget [M-log]

Steady-state = epochs 17–21 (single process, post-shuffle-fix). Representative epoch 21,
reconstructed from instrumented counters (selfplay, training) + artifact-mtime timeline (shuffle,
eval); the four phases sum to **1141 s vs the instrumented epoch total 1143 s** (residual 1–2 s).
Per-epoch table: `epoch_timings.json`; timeline: `epoch_timeline_summary.json`.

| Phase | Time | % epoch | Bound by | Evidence |
|---|---:|---:|---|---|
| **Training** (391 steps, bs 256) | **479 s** | **41.9%** | GPU compute (trunk fwd+bwd) + CPU decompress/IO | instrumented + 5 micro-benches |
| **Self-play** (256 games, 128 sims) | **373 s** | **32.6%** | CPU MCTS + Python orchestration (GPU ~29% duty) | instrumented + decomposition |
| **SealBot eval** (64 games, 50 ms) | **173 s** | **15.2%** | GPU + opponent process | mtime (ckpt→eval-json) |
| **Shuffle** (KataGo 2-phase) | **116 s** | **10.2%** | CPU zlib **compress** + disk I/O | mtime + zlib probe |
| Checkpoint + overhead | ~2 s | <1% | I/O | residual |
| **Epoch total** | **1143 s** | 100% | | `epoch_*` `elapsed_seconds` |

> Training elapsed **rises across the run** (epoch 17→21: 351→479 s) at constant 391 steps × bs 256.
> Since the GPU step is fixed (~465 ms), the growth is the **CPU input pipeline** scaling with the
> replay window (more / less-cached shards to decompress) — see §4.3.

---

## 3. Self-play / MCTS — the ~11 pos/s ceiling, pinned down

### 3.1 Where each searched position's time goes [M-log, epoch 21]
28.4 pos/s ⇒ **35.2 ms per searched position**, split (from `mcts_diagnostics`, `parse_selfplay_diag.py`):

| Component | ms/pos | % | Side |
|---|---:|---:|---|
| Python game orchestration | 12.2 | **35%** | Python (engine step, sample finalize, 256 `.npz` + `.hxr` writes, bookkeeping) |
| NN evaluator (`eval_evaluator_seconds`) | 10.2 | **29%** | GPU forward (92–95%) + legal-prior gather/D2H (~6%) |
| MCTS tree (select/backup/widen) | 8.1 | **23%** | Rust CPU |
| Encode (dense plane build) | 4.7 | **13%** | Rust CPU (rayon) |

**The binding constraint is CPU/Python work in a serial loop, not the GPU.** The GPU evaluator is
only 29% of self-play and the raw forward ~16%. Confirms: select→infer→backup is serialized
(`mcts.rs:run_searches_to_targets`) — rayon idle during the forward, GPU idle during select/backup.

### 3.2 The evaluator is GPU-forward-bound, not marshaling-bound [M-bench]
`evaluate_model1_payload` segment split (`evaluator_microbench.py`), at the real mean batch 209:
e2e **13.3 ms** = forward+H2D **12.2 ms (92%)** + legal-prior gather+softmax+D2H **0.81 ms (6%)** +
value-decode+D2H **0.12 ms (1%)** + input view **0.01 ms**. At batch 977: 95% forward. So the
Python/Torch boundary (the `scatter_reduce` softmax gather in `inference.py:215-232`, the `.cpu()`
transfers) is **not** the bottleneck — the GPU forward is.

### 3.3 Per-position latency vs throughput — explaining the gap [M-bench + M-log]
- **Per-state forward latency** (optimized eval model, AMP) is **U-shaped** in batch size:
  90 µs @16, **37 µs @64 (best)**, 41 µs @128, 51 µs @209, then a **55 µs/state plateau for ≥256**
  (`gpu_microbench`/`evaluator_microbench` batch sweep). Bigger batches do **not** improve
  per-state throughput past ~64–128 — the net saturates early.
- **In-situ the evaluator averages 209 states/forward** (max 977, capacity 1024) but costs
  **~100 µs/state** [M-log], ~2× the 51 µs clean-bench value at 209. The gap is the **small-batch
  tail**: as games finish, many forwards run at small batch (≤32) where per-state cost is 2–2.5×.
- **Throughput accounting:** cache hit rate is only **~10%** [M-log], so each searched position
  needs **~113 unique NN evals** (128 sims). 113 × ~100 µs ≈ **11.3 ms/pos** of evaluator — but the
  evals are issued *inside* a serial MCTS loop, so they don't overlap the 8.1 ms tree + 4.7 ms
  encode + 12.2 ms orchestration. Sum ≈ 35.2 ms/pos ⇒ 28.4 pos/s. **The throughput ceiling is the
  serial sum of per-position CPU work + ~113 sequential evals, not a single slow op.**

### 3.4 The cold "11 pos/s" first epoch — proven cuDNN autotune [M-bench + M-log]
| epoch | role | pos/s | eval µs/state |
|---:|---|---:|---:|
| 9 / 16 | first epoch of a process | **11.1–11.3** | **661–663** |
| 17–21 | steady | 28–34 | 92–147 |
| — | warm clean bench (bs 209–1024) | — | 51–55 |

With `cudnn.benchmark=True` (set in `inference.py:77`), the **first forward on each never-seen
batch shape costs ~810–1280 ms (mean 925 ms)** of algorithm autotuning vs 32 ms steady; with
`benchmark=False`, first-call = steady (**0 ms penalty**), same warm speed (`evaluator_microbench.py`
test 2). The evaluator sees highly variable batch sizes and the warmup only primes batch 1024, so a
cold process autotunes ~hundreds of distinct shapes on its first epoch: **925 ms × ~900 shapes ≈
830 s ≈ the observed 842 s** cold penalty (epoch 16 eval 996 s vs epoch 17 154 s). H2D and clocks
are ruled out (H2D ~2 ms; clocks measured healthy at 2805/3135 MHz, 59 °C).

### 3.5 Host↔device transfer & serialization [M-log + M-bench]
H2D is **not** a bottleneck: ~94.7 GB of input bytes/epoch at ~12 GB/s ≈ 8 s vs 108 s evaluator;
per-forward H2D is ~12% of the forward at batch 209, ~3% at 977. The 256 per-game `.npz` writes and
`.hxr` record are inside the 12.2 ms/pos orchestration bucket (small individually). Pinning host
buffers would shave a few % off large-batch H2D — minor.

---

## 4. Training — GPU-compute-bound + a cheap data-loader bug

### 4.1 The step is ~465 ms of real GPU compute [M-bench, 4 concordant measurements]
| Method | ms/step |
|---|---:|
| `gpu_microbench` full step (re-run) | 457 |
| `train_step_pipeline` latency (synced) | 466 |
| `train_step_reconcile` static-resident | 486 |
| `train_step_components` full | 472 |

Component split [M-bench, `train_step_components.py`]: forward(+loss, grad-tracked) **~199 ms**,
backward **~313 ms**, optimizer+grad-clip ≈ 0 (lost in noise). It is the **hex-conv trunk
forward+backward** over 256×64×41×41 activations — param count (12.2 M) is irrelevant to compute
(the trunk is 451 K; the heads are cheap matmuls). **Latency (466 ms) ≈ pipelined throughput
(461 ms)**, so the per-step `.cpu().item()` sync costs nothing and there is no overlap to recover.
**Floor ≈ 391 × 0.465 = ~182 s/epoch.**

### 4.2 NPZ re-decompression bug — measured, ~95 s/epoch [M-bench]
`train_passes` opens each shard once, but `_batch_from_npz` (`trainer.py:415`) indexes
`data[KEY][start:stop]` for 6 keys **per 256-row batch**, and NumPy's `NpzFile.__getitem__`
**re-decompresses the entire array on every access**. Each shard is 31× compressed (`inputNCHW`
alone is 694 MB uncompressed). Measured on a real shard: re-index-every-batch **7.4–7.7 s** vs
load-once **0.34–0.37 s** (**20–23×**). In the **in-situ training loop** (`train_microbench.py`):
**current 714 ms/step** (step 466 + decompress 244 + open) vs **load-once 503 ms/step** — i.e.
~317 s → ~197 s/epoch (hot-cache projection); ≈ **−95 s/epoch**. ~5 lines, correctness-neutral.

### 4.3 Why observed training (350–479 s) exceeds the hot-cache projection
My in-situ loop (hot OS cache) projects ~317 s/epoch in current mode; production observes
350–479 s and **rises with the replay window**. The delta (~30–160 s) is **cold-shard disk I/O**
on a growing shard pool — the GPU step is fixed, so the variable part is the input pipeline.
Sizing the disk-I/O split exactly would need a `cProfile` pass over a real epoch (§7).

### 4.4 Non-factors (measured)
Per-step sync (§4.1), grad-clip (458 vs 458 ms), memory-pinning (H2D 2.1 vs 2.2 ms),
channels-last toggle, and the HexConv mask recompute are all within noise. **Do not spend effort
on these.**

---

## 5. Shuffle & memory [M-bench + M-model]
`shuffle_mem_probe.py` on a real shard: per-row **109.3 KB** uncompressed, **31×** compression
(`inputNCHW` is sparse). zlib **decompress 2018 MB/s, recompress 538 MB/s, raw-write 964 MB/s**.
- **Memory:** the 8000-row group/bucket bound ⇒ peak **0.87 GB** resident per phase over a **32.8 GB**
  window — the fix that ended the OOM crash-loop; well within budget. RAM is **not** a current
  constraint (room to re-widen the window, or reclaim ~340 MB by halving the eval cache).
- **CPU:** 2-phase scatter→gather ≈ 2× decompress + 2× compress over the window. Modeled
  **~154 s** (33 s decompress + **122 s compress**) vs observed **116 s** — same ballpark; the
  phase is **compress-bound**. Scratch parts are written `np.savez_compressed` then deleted minutes
  later. **P3:** writing scratch uncompressed/`compresslevel=1` ⇒ modeled **~113 s** (−~40 s).

## 6. SealBot eval — 173 s/epoch (15.2%) [M-log mtime]
64 games vs SealBot best-50 ms with opening-temperature diversity; runs the dense player's MCTS +
the opponent process. Not separately timer-instrumented; a fixed per-epoch tax scaling with
`games_per_epoch` and the dense player's search cost. Lower-priority unless eval count rises.

---

## 7. What remains estimated / un-run (flagged)
1. **Training disk-I/O split (§4.3):** my in-situ loop used hot-cached shards, so the cold-shard
   read portion of the 350–479 s is bounded but not exactly sized. A `cProfile`/`torch.profiler`
   pass over one real epoch (safe now, GPU free) would split decompress vs disk vs compute.
2. **Shuffle 116 vs modeled 154 s:** the model assumes the full 300 k window and 2× passes; the
   taper window may be smaller and disk I/O overlaps. Direct instrumentation of `build_katago_shuffle`
   would tighten it. The compress-bound conclusion is robust either way.
3. **"Raise sims" cost** is projected (§9), not measured at visits ∈ {256, 400}.
4. SealBot eval is mtime-derived, not instrumented.

---

## 8. Bound-type summary
- **Training (42%):** GPU-compute-bound at ~465 ms/step (trunk fwd+bwd; ~182 s/epoch floor) + a
  CPU decompress tax (~95 s bug + growing disk I/O). Sync/clip/pinning/head-size are non-factors.
- **Self-play (33%):** CPU/Python-bound. GPU ~29% duty, raw forward ~16%. Buckets: orchestration
  35%, evaluator 29% (92–95% of which is GPU forward), tree 23%, encode 13%. Throughput =
  serial(per-position CPU + ~113 evals).
- **Eval (15%):** GPU + opponent-process; fixed tax.
- **Shuffle (10%):** CPU-zlib-**compress**-bound; RAM bounded (0.87 GB).
- **Cold start:** one-time cuDNN-autotune tax (~830 s) per fresh process — config-fixable.
- **The FC policy head is a quality issue, not a speed lever** (measured in both inference and training).

---

## 9. Prioritized optimizations (impact = measured unless noted)

| # | Change | Where | Impact | Risk | Evidence |
|---|---|---|---|---|---|
| **P0** | Load each NPZ array **once per shard**, then slice in RAM (train + validation) | `trainer.py:_batch_from_npz`, `train_passes`, `_run_validation` | **−~95 s/epoch (~8%)**; data-prep 20–23× | Very low (neutral) | **[M-bench]** §4.2 |
| **PB** | **`cudnn.benchmark=False`** for the evaluator (or pad eval batches to a few fixed buckets) | `inference.py:77` | **−~830 s on every cold/relaunch epoch** (11→28 pos/s immediately); steady speed unchanged | Very low | **[M-bench]** §3.4 |
| **P3** | Write shuffle **scratch** parts uncompressed / `compresslevel=1` | `replay.py` scratch write | **−~40 s/epoch** (modeled) | Low | [M-model] §5 |
| **P4** | Cut Python game orchestration (largest self-play bucket, 12.2 ms/pos): batch `.npz` writes, trim per-position bookkeeping | `selfplay.py`, sample finalize | Est. −30–60 s/epoch | Medium | [M-log] §3.1 |
| **P5** | Reduce serial idle in MCTS: overlap leaf-batch selection/backup with the GPU forward (double-buffer) | `mcts.rs`, `mcts_eval.rs` | Est. −20–50 s/epoch (GPU only 29% duty) | High (Rust concurrency) | [M-log] §3.1 |
| **P6** | Pad the small-batch evaluator tail toward batch ~64–128 (its efficiency sweet spot) | `mcts.rs` batching | Est. −15–30 s/epoch | Medium | [M-bench] §3.3 |
| **P7 (quality)** | Replace FC `PolicyHead`/`opp_policy_head` with fully-conv head | `architecture.py:PolicyHead` | **Speed-neutral** (measured 472↔473 ms); fixes policy diffuseness | Medium (retrain) | [M-bench] §0 + policy investigation |
| **P8** | Reclaim RAM: eval cache 262144→131072; consider re-widening replay window | config | −~340 MB; neutral throughput | Very low | [M-log] §5 |

**P0 + PB are the headline: ~free, measured, and together they remove ~95 s from every steady
epoch and ~830 s from every cold/relaunch epoch.** After P0, training ≈ the ~182 s GPU floor;
cutting it further needs a trunk/precision change (e.g. `torch.compile`/TensorRT — unmeasured).

### Raising MCTS simulations (the quality lever)
Self-play GPU eval + encode + tree scale ~linearly with sims; orchestration does not. **[Est]** at
128→400 visits (3.1×), epoch-21 self-play ≈ orchestration 129 + 3.1×(108+50+86) = **~885 s** (from
373 s); epoch ~1143→~1650 s (~28 min). **Shrinking the head does NOT make more sims affordable** —
per-sim cost is the trunk forward (~55 µs) + tree/encode CPU, not the head. To afford more sims,
attack the trunk forward (FP16/TensorRT, larger fixed leaf batches) and the CPU buckets (P4–P6).

The linearity is now **measured** by the companion MCTS review (`mcts_microbench.py`,
`mcts_code_review.md`): at 256 roots, **128→256→400→800 visits = 3.92 → 7.87 → 12.5 → 44.0 s** of
search (eval-dominated; ~linear to 400, super-linear by 800 as the cache/tree grows). [M-bench]

### ⚠️ Single-game path (SealBot eval & real play) is the real Goal-#4 blocker [M-bench, companion review]
The companion review (`mcts_code_review.md`) measured a structural gap this profiling missed: MCTS
uses **root parallelism** — rayon parallelizes across the 256 self-play games, so self-play scales
well, **but a single root runs fully serial**. A 128-sim single-game move is **184 ms**, a 400-sim
move **576 ms** — ~3.7× / ~11× the SealBot **50 ms** budget. So "raise sims to fix policy quality"
collides with the time budget for the single-game eval/play path, which needs **tree parallelism
(virtual loss)**, not just root parallelism. This is the most important cross-cutting finding for
Goal #4 and is detailed (with fixes A1/§4.2) in the companion review.
