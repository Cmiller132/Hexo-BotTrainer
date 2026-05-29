# Performance profiling — dense_cnn Model 1 (scratch_64) epoch cycle

**Date:** 2026-05-29  **Branch:** rust-rebuild (work staged on a review branch)
**Run:** `runs/dense_cnn_model1_scratch_64`  **GPU:** RTX 4070 Ti (12 GB), torch 2.10.0+cu126
**Author:** profiling deep-dive. The live supervised run was **stopped by design** for the
optimization stage, so this pass adds **real GPU micro-benchmarks** on a free GPU (the earlier
draft of this file could only estimate them).

> **Method & honesty note.** Three evidence sources, all read-only w.r.t. run state:
> 1. **The run's own instrumentation** — per-epoch `diagnostics/epoch_*.json`
>    (`metadata.result.{selfplay,training}` + `mcts_diagnostics`), parsed by
>    [`parse_epoch_timings.py`](parse_epoch_timings.py) → [`epoch_timings.json`](epoch_timings.json),
>    and `calibrate_performance.json`.
> 2. **GPU micro-benchmarks run now, on the free GPU**, against the **exact production network,
>    loss, and data path** (installed editable `hexo_models`):
>    [`gpu_microbench.py`](gpu_microbench.py) → [`gpu_microbench_summary.json`](gpu_microbench_summary.json).
> 3. Static code analysis.
>
> Every number below is tagged **[M-log]** (from run logs), **[M-bench]** (measured now on the
> free GPU), or **[Est]** (reasoned projection). Items I could not fully pin down are called out
> in §8. **§0 lists the corrections this pass makes to the earlier estimate-only draft** — two of
> its central claims were wrong, which matters because they drove the optimization ranking.

---

## 0. Corrections to the earlier (estimate-only) draft

The prior draft of this file deliberately ran **no GPU probe** and reasoned from "the net is
tiny (trunk 451 K params), so a 256-batch step should be a few ms of GPU." Direct measurement
on the free GPU shows that reasoning was wrong on two counts:

| Prior claim (estimate) | Measurement [M-bench] | Verdict |
|---|---|---|
| "A 256-batch training step should be a few ms of GPU; ~1 s/step is 20–100× off compute." | **260 ms/step** of genuine GPU compute (fwd 92 ms + bwd/opt ~168 ms). | **Wrong.** Param count ≠ FLOPs: the hex-conv trunk over 256×64×41×41 activations is a real ~260 ms. Steady ~1 s/step is ~4× off compute, not 20–100×. |
| "Per-step `.cpu().item()` sync kills overlap; remove it for a big win (P1)." | Step with sync **258.7 ms** vs without **257.9 ms**; H2D pageable **2.1 ms** vs pinned **2.2 ms**. | **Refuted.** The sync, grad-clip, and pinning are all <1% of the step. Not a lever. |
| "The oversized FC policy head is the dominant GPU cost in inference and training; shrinking it (P2) is a speed win." | Policy head adds **0.26 ms** (bs256) / 0.83 ms (bs1024) to a 14.2 / 56.9 ms forward — **~2%**. | **Refuted on the speed axis.** The head is negligible for speed. P2 remains a **quality** fix (policy diffuseness), not a speed fix, and will **not** cut per-sim search cost. |

The NPZ re-decompression bug (§4.1) and the per-phase wall-clock budget (§2) from the prior
draft **hold up** and are re-confirmed here with fresh measurements.

---

## 1. TL;DR

1. **Two phases dominate a steady-state epoch: training (~40%) and self-play (~34%).** Shuffle
   ~11%, SealBot eval ~16%. [M-log]
2. **The "~11 pos/s self-play" figure in NOTES is a cold-process artifact, now fully explained.**
   The first epoch of every OS process (epoch 9 at launch, epoch 16 at relaunch) runs at
   **11.1–11.3 pos/s**; every later epoch in the same process runs **25–36 pos/s**. The whole
   gap is the GPU evaluator: **~661 µs/state cold vs ~100–147 µs/state warm** (≈4.5×). This is a
   one-time cuDNN-autotune + GPU-clock-ramp tax per fresh process, **not** a steady ceiling. [M-log]
3. **Training is GPU-compute-bound at a 260 ms/step floor (≈102 s/epoch irreducible for this net
   at bs256), plus a CPU NPZ-decompression tax that grows with the replay window.** The biggest
   *cheap* win is the data-loader bug in §4.1: **measured 19.9× slower** data prep than necessary,
   ≈ **90 s/epoch** of pure redundant zlib, fixable in ~5 lines at zero quality cost. [M-bench]
4. **Self-play is CPU/Python-bound, not GPU-bound.** Of ~373 s self-play (epoch 21): GPU evaluator
   108 s (29%) — and **only ~60 s of that is the raw GPU forward** (55 µs/state × 1.08 M); the
   other ~48 s is Python payload marshaling + the legal-prior softmax-gather. Encode 50 s, Rust
   MCTS tree 86 s, Python game orchestration 129 s. The **raw GPU forward is only ~16% of
   self-play.** [M-log + M-bench]
5. **The real second training lever is a prefetch/overlap loader** (decompress the next shard on a
   background thread — zlib releases the GIL — so the ~240 ms CPU decompress hides under the
   260 ms GPU step). Combined with the §4.1 fix this can take training from ~400 s → ~110–150 s.
   This replaces the prior draft's (refuted) sync/pinning P1. [Est, from M-bench parts]

---

## 2. Per-phase wall-clock budget of one epoch [M-log]

Steady-state = epochs **17–21** (single process pid 53664, post-shuffle-fix, no crashes). Cold
first-epochs (9, 16) are excluded as warm-up outliers (§7). Full per-epoch table:
[`epoch_timings.json`](epoch_timings.json).

Representative steady epoch (epoch 21, 1143 s total):

| Phase | Time | % epoch | Bound by | Source |
|---|---:|---:|---|---|
| **Training** (391 steps, bs 256) | **479 s** | **42%** | GPU compute floor + CPU decompress/IO | `training.elapsed_seconds` |
| **Self-play** (256 games, 128 sims) | **373 s** | **33%** | CPU MCTS + Python orchestration (GPU ~29% duty) | `selfplay.elapsed_seconds` |
| **Shuffle** (KataGo 2-phase) | **~116 s** | **10%** | CPU zlib + disk I/O | mtime-derived (selfplay-end → shuffle-out) |
| **SealBot eval** (64 games, 50 ms) | **~173 s** | **15%** | GPU + opponent process | mtime-derived (ckpt → eval json) |
| Checkpoint + overhead | ~2 s | <1% | I/O | residual |
| **Epoch total** | **~1143 s (19 min)** | 100% | | `epoch_*` stage `elapsed_seconds` |

> **Training elapsed rises across the run** (epoch 17→21: 351→479 s, samples/s 285→209) at
> **constant** 391 steps × bs 256. Since the GPU step is fixed at 260 ms (§4), the growth is
> entirely the **CPU input pipeline** scaling with the replay window (more / less-cached shards
> to decompress) — direct evidence that training is input-pipeline bound on top of its GPU floor.

### 2.1 Inside self-play (epoch 21) [M-log]
| Sub-step | Time | Note |
|---|---:|---|
| `mcts_search_elapsed_seconds` | 244 s | the search (rest is orchestration) |
| &nbsp;&nbsp;• GPU evaluator (`eval_evaluator_seconds`) | **108 s** | of which **~60 s raw forward** [M-bench], **~48 s Python marshal + legal-prior gather** |
| &nbsp;&nbsp;• encode (`eval_encoding_seconds`, Rust rayon) | 50 s | dense 13×41×41 plane build |
| &nbsp;&nbsp;• MCTS tree CPU (selection/backup/widen) | ~86 s | = search − eval − encode |
| Game orchestration (engine step, finalize, .npz/.hxr writes, Python) | ~129 s | = self-play − search |

**GPU is busy only ~29% of self-play, and the raw forward is only ~16%.** This is a serial
select→infer→backup loop on a cheap network — CPU/Python-bound, not GPU-bound.

### 2.2 Inside training [M-bench + M-log]
- 100 000 samples, bs 256 → **391 steps**; steady **351–479 s ⇒ 0.90–1.22 s/step**.
- **Measured GPU compute per step = 260 ms** (fwd 92 + bwd/opt 168). So the GPU floor is
  ≈ 391 × 0.260 = **~102 s/epoch**. The remaining 250–380 s is the CPU input pipeline
  (decompression + IO + the §4.1 bug), confirmed by the rising-with-window trend above.

---

## 3. Self-play / MCTS — detailed findings

**Bound type: CPU/Python, ~29% GPU duty, cheap forward.** Evidence + code:

- **The GPU forward is cheap and the FC head is negligible** [M-bench]. Optimized eval model,
  AMP, `forward_policy_value`: bs256 = 14.2 ms (**55.4 µs/state**), bs1024 = 56.9 ms (55.5 µs/
  state). Trunk-only = 13.7 / 55.3 ms; the **policy head adds only 0.26 / 0.83 ms (~2%)** and the
  value head ~0.2 ms. The trunk (4 `GatedResBlock`s of masked hex convs, `architecture.py:89`)
  is ~97% of the forward. So shrinking `PolicyHead` (`architecture.py:113`) does **not** speed up
  search.
- **In-situ evaluator is ~1.8× the raw forward** [M-log vs M-bench]: epoch 21 `eval_evaluator_seconds`
  = 108 s / 1.084 M states = **99.7 µs/state** vs **55 µs/state** raw. The ~45 µs/state delta is
  Python-side: `torch.frombuffer`/`reshape` of the payload (`inference.py:188`), chunked `.cpu()`
  transfers (`inference.py:194`), and the legal-prior **softmax-gather** (`scatter_reduce_` +
  `exp` + per-row normalize + `.cpu()`, `inference.py:215-232`). This — not the head — is the
  evaluator's reducible overhead.
- **Serial select→infer→backup, no GPU/CPU overlap** (`mcts.rs:run_searches_to_targets`):
  rayon selects a leaf batch, then a single Python evaluator call runs under the GIL
  (`mcts_eval.rs`), then backup runs. GPU idle during select/backup; rayon idle during the
  forward. Classic AlphaZero serial-batch shape → the ~29% duty cycle. Overlapping helps less
  than it sounds here (GPU is only 16%); the bigger self-play wins are cutting Python
  orchestration (129 s) and evaluator marshaling (~48 s).
- **Encoding is well-optimised:** rayon `par_iter`, thread-local scratch reuse, zero-copy
  `PyBytes` (`mcts_eval.rs`). ~50 s/epoch, scales with simulations.
- **Eval cache effective:** `eval_cache_inserts == eval_unique_states`, no recompute on eviction.
  The 262 144 cap is throughput-neutral; lowering to 131 072 just reclaims ~340 MB.

---

## 4. Training — detailed findings (highest cheap ROI)

### 4.1 NPZ re-decompression bug — **[M-bench], re-confirmed**
`train_passes` (`trainer.py:224`) opens each shard once with `np.load`, but `_batch_from_npz`
(`trainer.py:415`) indexes `data[KEY][start:stop]` for ~6 keys **per 256-row batch**. NumPy's
`NpzFile.__getitem__` **re-reads and re-decompresses the entire array from the zip on every
access** (no cache). Shards are `np.savez_compressed` (`replay.py`), `inputNCHW` alone is ~694 MB
uncompressed (13×41×41 f32 × 7936 rows), so each batch decompresses ~hundreds of MB and keeps 256
rows.

Micro-benchmark on a real shard (`…/epoch_000022/train/data00008.npz`, 7936 rows, 31 batches,
CPU only, free machine):

| Pattern | Time | vs optimal |
|---|---:|---|
| **current** (re-index every batch) | **7.42 s** | **19.9×** |
| load each array once, then slice in RAM | 0.37 s | 1× |

- **Per-epoch waste ≈ 90 s** (~13 shards × ~7 s; ~20% of training time), recovered by a ~5-line
  change. (The prior draft measured 33.8× / ~130–155 s under live-trainer CPU contention; the
  clean number here is 19.9× / ~90 s. Same bug, same fix, both large.) Validation has the same
  bug (`trainer.py:298`) but is disabled here (`validation_fraction=0`).
- **Memory:** the fix holds one shard's arrays (~0.8 GB) at a time — *less* churn than repeated
  transient decompressions. No RAM cost.

### 4.2 Per-step sync / grad-clip / pinning are NOT levers — **[M-bench], refutes prior P1**
Step with the production `.cpu().item()` sync = **258.7 ms**; accumulating loss on-GPU instead =
257.9 ms (−0.3%). Dropping grad-clip = 258.2 ms. H2D pageable = 2.1 ms, pinned = 2.2 ms. None of
these move the needle — the step is dominated by the 260 ms of conv compute, and there is no async
prefetch for the sync to "block," so removing it buys nothing. **Do not spend effort here.**

### 4.3 The real lever: overlap decompression with the GPU step — **[Est from M-bench parts]**
Today each step is serial: `_batch_from_npz` (~240 ms CPU decompress, §4.1) **then**
`_optimizer_step` (260 ms GPU). With (a) §4.1's load-once-per-shard (decompress ~once per shard,
not per batch) and (b) a background prefetch thread decompressing the next shard while the GPU
runs (zlib releases the GIL, so this genuinely overlaps), the per-step cost approaches
`max(amortized_decompress, 260 ms) ≈ 260 ms`. **Estimated training: ~400 s → ~110–150 s/epoch.**
*(Estimated — confirm with an in-situ profiler, §8.)*

### 4.4 Minor: train-time HexConv mask recompute
`HexConv2d.forward` recomputes `weight * hex_mask` every train forward (`architecture.py:73`). The
task's earlier toggle test found this ~irrelevant (~249 ms either way), consistent with the 260 ms
measured here — the mask multiply is a rounding error against the conv FLOPs. Low priority.

**Classification: training is GPU-compute-bound at a 260 ms/step floor, *plus* a CPU
input-pipeline tax (decompression + IO) that is large, cheap to cut, and grows with the window.**

---

## 5. Shuffle — ~116 s/epoch (10%) [M-log mtime]
Two-phase scatter→gather (`replay.py`), bounded to 8000-row groups/buckets (the OOM-crash fix;
peak RAM ~0.9 GB). Cost is **CPU zlib + disk I/O**: phase-1 writes ~21 compressed scratch parts
per bucket; phase-2 re-reads and re-compresses them. **Scratch parts are transient yet written
`np.savez_compressed`** — full zlib for data deleted minutes later.
- **Opportunity (P3):** write *scratch* parts uncompressed / `compresslevel=1`. `inputNCHW` is
  very sparse (694 MB → ~7 MB compressed), so this trades disk bandwidth for CPU. Medium priority
  (shuffle is only 10%); needs an A/B.

## 6. SealBot evaluation — ~173 s/epoch (15%) [M-log mtime]
64 games vs SealBot best-50 ms with opening-temperature diversity; runs the dense player's MCTS +
the SealBot opponent process. Not separately timer-instrumented (mtime-derived). Fixed per-epoch
tax scaling with `games_per_epoch` (64) and the dense player's own search cost. Revisit only if
eval count rises.

---

## 7. The cold-first-epoch 3× slowdown — **SOLVED [M-log]**
| epoch | role | pos/s | eval_evaluator_seconds | unique states | µs/state |
|---:|---|---:|---:|---:|---:|
| 9  | first after 23:18 launch | **11.3** | 930.7 | 1.40 M | **663** |
| 16 | first after 07:46 relaunch | **11.1** | 995.9 | 1.51 M | **661** |
| 17 | 2nd in process | 30.7 | 154.4 | 1.05 M | 147 |
| 21 | steady | 28.4 | 108.1 | 1.08 M | 100 |
| —  | warm micro-bench (bs256) | — | — | — | **55** [M-bench] |

The slowdown is **entirely** in the GPU evaluator (~4.5× per-state), is **per-process-first-epoch**
(so not the per-epoch `DenseCNNInference` rebuild or its `_warm_up_cuda`, which run every epoch),
and matches the signature of **cuDNN benchmark autotuning + GPU clock ramp on a cold process**. The
warm steady forward (55 µs/state bench, 100 µs/state in-situ) confirms there is no intrinsic
slowness once warm. **The "11 pos/s" in NOTES is the cold number; steady self-play is ~28–34 pos/s.**
*(I did not isolate autotune-vs-clock with a cold/`cudnn.benchmark=off` probe — see §8.2 — but the
12× cold-forward gap and the per-process-first-epoch pattern fit autotune+clock.)*

---

## 8. What I could NOT fully pin down (flagged)
1. **The ~250–380 s/epoch training residual** beyond the measured 260 ms GPU floor + ~90 s NPZ
   bug. It is CPU input-pipeline (decompress on cold shards + IO + Python loop) and it grows with
   the replay window, but my single-hot-shard micro-bench can't size the cold-shard IO portion. A
   `cProfile`/`torch.profiler` pass over **one real train epoch** would split decompress vs IO vs
   compute exactly. (Now that the GPU is free, this is safe to run.)
2. **Cold-start root cause (§7):** autotune vs clock-ramp not separated. Probe: throwaway process,
   time first 200 vs next 200 evaluator batches with `cudnn.benchmark` on/off while watching
   `nvidia-smi` clocks.
3. **"Raise sims" cost** is projected (§9), not measured at visits ∈ {256, 400}.

---

## 9. Prioritized optimizations (with expected impact)

| # | Change | Where | Impact | Risk | Evidence |
|---|---|---|---|---|---|
| **P0** | **Load each NPZ array once per shard, then slice in RAM** (train + validation) | `trainer.py:_batch_from_npz`, `train_passes`, `_run_validation` | **−~90 s/epoch (~8%)**; data prep 19.9× faster | Very low (correctness-neutral) | **[M-bench]** §4.1 |
| **P1** | **Background prefetch/double-buffer**: decompress next shard on a worker thread, overlapping the 260 ms GPU step (zlib drops the GIL) | `trainer.py:train_passes` | With P0, training **~400→~110–150 s/epoch** | Medium (threading) | [Est] §4.3 from M-bench |
| **P2** | Replace FC `PolicyHead`/`opp_policy_head` with fully-conv head (`3×3→ReLU→1×1→1 logit/cell`) | `architecture.py:PolicyHead` | **QUALITY only** (policy diffuseness). ~Neutral on speed (head is ~2%). | Medium (retrain) | [M-bench] §3 + policy-diffuseness investigation |
| **P3** | Write shuffle *scratch* parts uncompressed / `compresslevel=1` | `replay.py` (scratch write) | Est. −20–50 s/epoch shuffle CPU; +transient disk | Low | Static §5; needs A/B |
| **P4** | Cut evaluator Python overhead: keep the legal-prior gather + value decode on-GPU and transfer once; minimise `frombuffer`/`reshape` copies | `inference.py:188-232`, `mcts_eval.rs` | Est. −20–40 s/epoch self-play (the ~48 s marshal bucket) | Medium | [M-log vs M-bench] §3 |
| **P5** | Reduce Python game orchestration (largest self-play bucket, 129 s): batch `.npz` writes, trim per-position Python bookkeeping | `selfplay.py`, sample finalize | Est. −30–60 s/epoch self-play | Medium | [M-log] §2.1 |
| **P6** | Reclaim RAM: eval cache cap 262144→131072 (throughput-neutral) | `mcts_session_cache_max_states` | −~340 MB; room for a bigger replay window | Very low | [M-log] §3 |

**P0+P1 alone are estimated to cut the steady epoch from ~19 min to ~13–14 min (~25–30%) at zero
quality cost**, and P0 is essentially free.

### On raising MCTS simulations (the quality lever)
Self-play GPU eval + encode + tree-CPU scale ~linearly with sims; orchestration does not.
**[Est]** at 128→400 visits (3.1×), epoch 21 self-play ≈ orchestration 129 + 3.1×(108+50+86) =
**~885 s** (from 373 s, ~2.4×); epoch ~1143→~1650 s (~28 min). **Correction to the prior draft:**
shrinking the policy head (P2) will **not** make more sims affordable — per-sim cost is the trunk
forward (55 µs) + Python marshaling + Rust tree work, *not* the head (0.26 ms/batch). To afford
more sims, attack the **trunk forward** (FP16/TensorRT export, larger leaf batches per forward) and
the **evaluator/orchestration Python** (P4/P5), not the head. Confirm with the §8.3 probe.

---

## 10. Bound-type summary
- **Training:** GPU-compute-bound at a **260 ms/step floor** (~102 s/epoch) **+** CPU
  input-pipeline tax (decompression + IO, ~250–380 s/epoch, grows with window). Per-step sync /
  grad-clip / pinning are non-factors. *Cheap wins: P0 then P1.*
- **Self-play:** CPU/Python-bound. GPU ~29% duty; raw forward only ~16%. Buckets: orchestration
  35%, MCTS tree 23%, encode 13%, evaluator-marshal ~13%, raw forward ~16%. *Wins: P4/P5.*
- **Shuffle:** CPU-zlib + disk-I/O; RAM bounded (fixed). *P3.*
- **Eval:** GPU + opponent-process; fixed per-epoch tax.
- **Cold start:** one-time cuDNN-autotune/clock-ramp tax (~4.5× evaluator on first epoch/process).
- **The FC policy head is a quality problem, not a speed problem.**
