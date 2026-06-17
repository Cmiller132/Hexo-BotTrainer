# OPTION 2 — multiple self-play worker processes sharing the GPU

DESIGN AGENT NOTES. Goal: decide, with ZERO architectural code change, whether running
TWO concurrent self-play processes that time-slice the GPU raises **aggregate** self-play
pos/s vs one process — without touching any strength knob.

## Hypothesis
In the early/mid (host/launch-bound) regime the GPU is ~57% idle (submit ~17 ms host enqueue
while the GPU does ~11 ms; hundreds of tiny kernels per forward). A second self-play process
on its own CUDA stream/context can interleave its kernels into process A's idle gaps, raising
aggregate GPU utilization → more aggregate pos/s, and it uses the otherwise-idle CPU cores
(the whole pipeline — Rust search + Python driver + evaluator — is single-threaded, search.rs
`run_continuous` has no rayon/threads).

The late/deep regime is GPU-COMPUTE-bound (GPU ~97%), where a second process buys nothing
(they just serialize on the SMs). The realistic refill MIX is a blend, so the question is
strictly **empirical**: does the launch-bound headroom outweigh the compute-bound contention.

## Why this needs NO code change
`scripts/_hexfield_lategame_bench.py` is already a standalone, self-contained self-play driver:
it builds its own `HexfieldEvaluator`, its own `_rust.HexfieldMctsSession(max_states=CACHE)`,
its own game states, and writes its own `out.json`. Nothing is shared between two invocations —
each process gets an independent CUDA context, model copy, Rust session, and cache. So the test
is: launch TWO of them at `active_games=32` (combined concurrency ≈ the single-process
`active_games=64` baseline) for the SAME window, and sum their `pos_per_s`.

`refill=True` (the default — `argv[5] != "cohort"`) is the regime that mirrors a real epoch
(continuous refill MIX of early/mid/late), so the test reproduces production behavior.

## What can break parity — and why this test does NOT
- The two processes share NOTHING in-memory; each runs identical math to the baseline, just at
  `active_games=32` instead of 64. Self-play action math is unaffected by `active_games`
  (it only changes how many games are co-resident / the flush batch composition; search_visits,
  pcr, virtual_batch_size, c_puct, widening, dirichlet are all passed identically below).
- The PARITY question for THIS option is purely "did we change the math?" The answer is no
  (no source under `packages/` or `configs/` is touched), so the authoritative parity
  harnesses must still pass UNCHANGED. We run them as the gate before trusting the bench.
- The real-architecture concern (shared sample dir, key collisions, BT-pool merge) is a
  PRODUCTIVIZATION question, NOT a parity question for the throughput probe. See "If it wins".

## VRAM analysis (measured, not assumed) — fits 12 GB comfortably
- Model: 1.23 M params ≈ 6 MB. CUDA context after first alloc: ~0.5–1.0 GB (driver/runtime).
- Handoff records peak **serve** VRAM ≈ 2.3 GiB single-process after the bias rewrite, and
  notes VRAM is **flat across active_games** (≈3.4–4.8 GB live) because the per-group
  `PAIR_CEILING = 3.8e7` caps the (B, 4, S, S) fp16 bias transient regardless of how many games
  are resident. So halving `active_games` to 32 does NOT halve a process's peak — peak is set by
  the per-group ceiling + cache, not by concurrency.
- Therefore two processes ≈ 2 × (per-process fixed: context + cache + one capped forward
  transient). Conservative upper bound: 2 × ~2.5 GiB ≈ 5 GiB, plus 2 CUDA contexts → still well
  under the 11.3 GiB free measured by `nvidia-smi` (12.0 total, 0.65 used at rest). **VRAM is
  not the wall** for 2 processes. The bench records combined peak via each out.json's
  `peak_vram_gib`; the host-level guard is `nvidia-smi` sampled DURING the run (per-process
  `torch.cuda.max_memory_allocated` does NOT see the other process, so the GPU-wide check
  must come from nvidia-smi).
- Compute Mode is **Default** (not EXCLUSIVE_PROCESS), so two processes legally share the GPU
  by time-slicing. (No MPS daemon is configured; default time-slicing is exactly the regime we
  want to measure. Setting up MPS would be a separate, larger experiment — out of scope for the
  minimal decisive test.)

## The decisive minimal test

Baseline to beat: single process, `active_games=64`, refill MIX, same window.
Treatment: two processes, each `active_games=32`, refill MIX, same window, run concurrently.

Decision rule:
- `sum(treatment.pos_per_s)`  >>  `baseline.pos_per_s`  (say ≥ +20% to clear noise/overhead)
  → multi-proc HELPS; scope the real architecture (below).
- `sum(treatment.pos_per_s)`  ≲  `baseline.pos_per_s`  → the GPU is the wall; multi-proc does
  NOT help. Report and stop.
- Also require combined peak VRAM (nvidia-smi) < ~10 GiB to leave headroom.

### Exact launch recipe (run both treatment procs in background, wait, sum)

Single command per step; all paths absolute. CKPT and the bench script are fixed.

```bash
# ---- vars ----
REPO=/mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000031.pt
BENCH=$REPO/scripts/_hexfield_lategame_bench.py
SECS=180          # long enough to reach the late-game MIX, same for all 3 runs

# ---- 1) BASELINE: single process, active_games=64 ----
cd $REPO && PYTHONPATH=packages/hexfield/python $PY $BENCH $CKPT $SECS "64" /tmp/mp_base64.json refill

# ---- 2) TREATMENT: two processes, active_games=32 each, CONCURRENT ----
# launch both in background, sampling nvidia-smi for combined peak, then wait for both
cd $REPO && \
  ( PYTHONPATH=packages/hexfield/python $PY $BENCH $CKPT $SECS "32" /tmp/mp_a.json refill > /tmp/mp_a.log 2>&1 & echo $! > /tmp/mp_a.pid ) && \
  ( PYTHONPATH=packages/hexfield/python $PY $BENCH $CKPT $SECS "32" /tmp/mp_b.json refill > /tmp/mp_b.log 2>&1 & echo $! > /tmp/mp_b.pid ) && \
  ( for i in $(seq 1 $((SECS+20))); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done > /tmp/mp_vram.csv & echo $! > /tmp/mp_vram.pid ) && \
  wait $(cat /tmp/mp_a.pid) $(cat /tmp/mp_b.pid); kill $(cat /tmp/mp_vram.pid) 2>/dev/null; true

# ---- 3) SUM + COMPARE ----
$PY - <<'PYEOF'
import json
base = json.load(open("/tmp/mp_base64.json"))[0]["pos_per_s"]
a = json.load(open("/tmp/mp_a.json"))[0]["pos_per_s"]
b = json.load(open("/tmp/mp_b.json"))[0]["pos_per_s"]
peak = max(int(x) for x in open("/tmp/mp_vram.csv") if x.strip())
combo = a + b
print(f"baseline ag64        : {base:.2f} pos/s")
print(f"treatment ag32 x2 sum: {combo:.2f} pos/s   (A={a:.2f} B={b:.2f})")
print(f"speedup              : {combo/base:.2f}x")
print(f"combined peak VRAM   : {peak} MiB (must be < ~10000)")
print("VERDICT:", "MULTI-PROC HELPS" if combo >= 1.20*base and peak < 10000 else "GPU IS THE WALL / does not help")
PYEOF
```

Notes for the validate phase:
- Run **baseline first, alone**, so its GPU timing is uncontended (apples-to-apples vs each
  treatment proc being contended is the whole point — the SUM captures the contention cost).
- Identical `$SECS` for all three runs; each run has its own warm-up inside the window
  (torch.compile + cudnn autotune happen once at process start). 180 s is enough that warm-up
  amortizes and games reach the late-game support buckets (watch `support_hist_256` in each
  out.json to confirm the MIX got past early game; if it's all ≤512, lengthen the window).
- `base_seed=7` is identical in both treatment procs (the bench hardcodes it). That is FINE for
  a throughput probe — it does not affect kernel timing — but it means the two procs play
  identical games; for the throughput measurement that is irrelevant. (In the real architecture
  you would seed per-worker; see below.)
- Sanity: if treatment A and B are wildly asymmetric (one starved), the OS scheduler / CPU
  oversubscription is interfering — re-run pinning each proc to 8 cores
  (`taskset -c 0-7` / `taskset -c 8-15`) and note it.

### Parity gate (run BEFORE trusting the bench — proves math unchanged)
No source changed, so these must PASS exactly as on HEAD:
```bash
cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexfield_model.py tests/test_hexfield_continuous_parity.py -q
cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_compile_overlap_test.py   # expect RESULT: PASS
```

## If it wins — scope of the REAL architecture (describe only, do NOT build)
A productized 2-worker self-play would need:
1. **Shared sample dir, no key collisions.** `ContinuousDriver.next_key = epoch*1_000_000`
   (selfplay.py:75) and game shards are `out_dir/game_{key}.npz`. Two workers in the same
   `samples_dir/epoch_XXXXXX` would collide. Fix: partition the key space per worker
   (`next_key = epoch*1_000_000 + worker_id*100_000`) AND split `games_per_epoch` across
   workers (e.g. 96/96 of 192). Training already reads a rolling mtime window over all
   `epoch_*/game_*.npz`, so two workers writing into the same epoch dir merges for free.
2. **Per-worker .hxr record file** (selfplay.py:332 — `HexoRecordFile.create` clobbers): give
   each worker its own `epoch_XXXXXX_wN.hxr`; the dashboard scans the dir.
3. **Per-worker RNG seed** so the workers don't play identical games:
   `base_seed = run.seed*1_000_003 + epoch + worker_id` (currently selfplay.py:362 with no
   worker term).
4. **Per-worker evaluator + cache** (already independent — each builds its own
   `HexfieldEvaluator` + `HexfieldMctsSession`; the eval cache is ~inert at ~4% hit so NOT
   sharing it costs ~nothing). No merge needed.
5. **Orchestration**: `generate_selfplay_epoch` (selfplay.py:274) would `fork`/`spawn` N worker
   subprocesses each running a slice of the epoch, then join and aggregate the diag stats dicts.
   The trainer process itself stays single (only self-play is parallelized). VRAM budget per the
   measured numbers supports N=2 comfortably; N=3 would need a re-measure of combined peak.
6. **No strength change**: each worker uses the IDENTICAL search config; splitting games across
   workers is statistically identical to playing them serially (independent games), so playing
   strength / sample distribution is unchanged. This is the key invariant to assert.

## If it does NOT win
Then the refill MIX is dominated by compute-bound late-game positions that already saturate the
SMs, and the launch-bound early/mid idle is too small a fraction of epoch wall-clock to recover
by a second process. Conclusion: the GPU is the wall; pursue the launch-bound fix INSIDE one
process instead (CUDA graphs / fewer-kernel forward / batching across games), which Option(s)
targeting the kernel-launch count address. Report the SUM-vs-single numbers and stop.

## Expected gain (honest prior)
Plausible but UNCERTAIN. Upside is real: GPU ~57% idle in early/mid means up to ~1.7x headroom
THERE, and the CPU pipeline is single-threaded so a 2nd process genuinely adds a 2nd core of
host enqueue. But the epoch is a MIX, and the compute-bound late-game tail (GPU ~97%) gives a
second process nothing and adds context-switch / scheduler overhead. Net realistic expectation:
somewhere between **1.0x (wall) and ~1.5x** aggregate. Worth measuring because the test is cheap
(three bench runs, no code change) and the answer is decisive either way.
