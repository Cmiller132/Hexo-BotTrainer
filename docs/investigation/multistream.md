# OPTION 4 — Multi-CUDA-stream group forwards (design + prototype)

Author: design agent. Baseline = HEAD (build_attn_bias rewrite + active_games=64).
GPU free, run stopped. Do NOT run benches in this phase — validate phase only.

## 1. What the code does today

`HexfieldEvaluator.submit_payload` (packages/hexfield/python/hexfield/inference.py:126)
parses the flush, computes `plan_groups(sizes)`, and loops:

```python
for start, end, pad_to in plan_groups(sizes):
    self._forward_group(..., gpu_values, gpu_ml, gpu_priors)
return {... "values_gpu": torch.cat(gpu_values), "priors_gpu": torch.cat(gpu_priors) ...}
```

Each `_forward_group` (inference.py:209) does host-pack → pinned H2D → one
`forward_policy_value` (the trunk: 6 conv blocks + 3 attn blocks + heads) →
on-GPU decode/softmax, appending GPU tensors to the shared lists. EVERYTHING
runs on the **default CUDA stream**. The single D2H is deferred to `result()`
(inference.py:180). All groups in a flush are processed **sequentially**, so the
launches AND kernels of group g+1 only begin after group g's launches are issued.

`plan_groups` emits groups in **ascending row order** (rows arrive size-DESC; a
group is a contiguous run), and the per-group outputs are concatenated in that
order. That ordering is load-bearing: `priors_gpu` IS the row-major flat layout
the Rust parser walks (inference.py:155-156, 191-196). The prototype MUST keep the
per-group append order identical.

## 2. The regime facts that decide feasibility (established, not re-derived)

- EARLY/MID flush = **GPU-LAUNCH-bound**: `submit` ~17 ms host-enqueue while GPU
  busy only ~11 ms; GPU ~57% idle. Hundreds of tiny kernels per forward.
- LATE/DEEP flush = **GPU-COMPUTE-bound**: GPU ~97% busy (O(S²) attention on huge
  Npad). Large-S groups saturate the GPU alone.
- Group counts per flush (from `PAIR_CEILING=3.8e7`, `(g)(pad+8)² ≤ ceiling`):
  - early (Npad ~64–260): pad²≈70k → ~150 rows fit in **1 group** (maybe 2).
  - mid (256–768): pad≈776² → ~63 rows/group → **2–3 groups**.
  - late (768–1800): pad≈1808² → ~11 rows/group → **~14 groups**.
  - deep (1500–3300): → many groups, each large-S.

## 3. Why a measurable win is DOUBTFUL (the honest analysis)

Multi-stream helps **only** when the GPU sits idle due to a *false dependency*
between independent kernels that the scheduler serializes — i.e. the host has
ALREADY enqueued work that could run concurrently, but it is stuck behind an
unrelated kernel on the same stream. It does **not** help when:

1. **Host launch rate is the bottleneck** (the launch-bound regime). The ~57% GPU
   idle is because a single Python thread (holding the GIL) cannot dispatch the
   hundreds of tiny kernels fast enough — `submit` host time (17 ms) > GPU busy
   (11 ms). K streams do NOT add a second host dispatch thread: the same one
   thread still issues every `cudaLaunchKernel` serially, just tagging them with
   different stream handles. **Total host enqueue time is unchanged** (it may even
   rise slightly from per-group stream bookkeeping + the extra
   `record_event`/`wait_event` syncs). Since wall-clock for a launch-bound flush ≈
   host enqueue time, streams cannot shorten it. This is the regime the brief
   hoped to help, and it is precisely the regime where streams structurally cannot.

2. **In the launch-bound regime there are barely any groups** (early = 1 group;
   mid = 2–3). With 1 group there is nothing to overlap at all. With 2–3 groups
   the inter-group concurrency opportunity is tiny, and is dominated by (1).

3. **The compute-bound regime (late/deep) has many groups but the GPU is already
   ~97% busy.** Each large-S group's O(S²) attention saturates the SMs by itself;
   co-residency just time-slices the same SMs → no speedup (and possible L2/cache
   contention regression). This is the explicit risk flagged in the brief.

So the two regimes split exactly the wrong way: where there are many groups to
overlap, the GPU is saturated; where the GPU is idle, there is one group and the
idleness is launch-rate, not dependency.

### Counter-consideration (why it is still worth a *small* prototype)
There is a narrow band — **mid (2–3 groups), forward-heavy** — where the GPU is
not yet saturated AND there is >1 group. If a non-trivial fraction of each group's
~11 ms GPU time is itself made of small kernels with gaps (the conv blocks are
gather+GEMM, many small launches), then group g+1's kernels on stream 2 *could*
backfill the gaps left while group g's tail kernels drain. But the size of that
band is bounded by the host being able to get ahead — which the profiling says it
cannot in the launch-bound part. The realistic upside is therefore **small (single
-digit % on the mid mix at best)**, and a regression on deep is plausible. A
minimal prototype is cheap to write and gives a definitive measured answer, so it
is worth ONE measured pass — but expectations should be low. (This matches the
HANDOFF "realistic ceiling ~1.5–1.8× total, then matmul-bound" framing: scheduling
tricks are near their limit.)

## 4. Minimal prototype (Python-only, draft — DO NOT APPLY to packages/)

Self-contained: add a stream ring to the evaluator, run each group's
`_forward_group` on a round-robin stream, then sync every used stream onto the
default stream BEFORE the `torch.cat`/`result()` D2H so ordering + single-D2H hold.
Math is unchanged: identical ops, identical order of appends; only the stream tag
and explicit event syncs differ. Gate behind `HEXFIELD_NSTREAMS` (default 1 =
exact current behaviour) so it is a no-op until measured.

```python
# --- in HexfieldEvaluator.__init__, after compile setup ---
self._nstreams = int(os.environ.get("HEXFIELD_NSTREAMS", "1"))
self._streams = (
    [torch.cuda.Stream() for _ in range(self._nstreams)]
    if (self.device.type == "cuda" and self._nstreams > 1)
    else None
)

# --- submit_payload group loop replacement ---
groups = plan_groups(sizes)
if self._streams is None:
    for start, end, pad_to in groups:
        self._forward_group(feats, qr, nbr, offsets, sizes, legal_counts,
                            start, end, pad_to, request_ml,
                            gpu_values, gpu_ml, gpu_priors)
else:
    default = torch.cuda.current_stream()
    used = []
    for i, (start, end, pad_to) in enumerate(groups):
        s = self._streams[i % self._nstreams]
        # All H2D + forward + decode for this group go on stream s. The lists
        # are appended IN GROUP ORDER regardless of stream (Python is serial),
        # so torch.cat below preserves the exact row-major layout.
        with torch.cuda.stream(s):
            self._forward_group(feats, qr, nbr, offsets, sizes, legal_counts,
                                start, end, pad_to, request_ml,
                                gpu_values, gpu_ml, gpu_priors)
        used.append(s)
    # Make the default stream wait for every group's stream to finish so the
    # subsequent torch.cat (default stream) reads fully-produced tensors and
    # result()'s .cpu() is the single, correct D2H sync point.
    for s in set(used):
        default.wait_stream(s)
```

Important correctness notes for the prototype:
- **Pinned-buffer lifetime / non_blocking H2D:** `_forward_group` already does
  `t.pin_memory().to(device, non_blocking=True)`. The pinned host tensor is a
  fresh Python local that is dropped at the end of `_forward_group`. With a
  non-default stream the async copy could outlive the host buffer → torn read.
  To stay bit-safe the prototype MUST either (a) keep the host buffers alive until
  the stream syncs (return/stash them), or (b) drop `non_blocking` inside the
  multi-stream path. Simplest safe form: in the multi-stream branch, force a
  blocking H2D (set a flag the `_h2d` closure reads). The H2D is ~negligible
  per the profiling, so this costs nothing measurable and removes the hazard.
- **torch.cat ordering:** appends happen on the serial Python thread in group
  order; stream choice does not reorder the lists. Layout is preserved exactly.
- **compile interaction:** compiled buckets run fine under a stream context; the
  graph is replayed on whatever stream is current. No recompile (shapes unchanged).
- Keep `HEXFIELD_NSTREAMS=1` as the default so production is byte-for-byte today's
  path until the measurement says otherwise.

## 5. Parity strategy (math MUST stay bit/fp16-identical)

The prototype changes only stream assignment + sync, not arithmetic. Gate proves it:
1. `scripts/_hexfield_compile_overlap_test.py` — its ASYNC-PARITY block already
   compares `result(submit_payload(p))` vs `evaluate_payload(p)` and requires
   maxabsdiff **== 0.0** (exact). Run it once with `HEXFIELD_NSTREAMS=1` (sanity:
   unchanged) and once with `HEXFIELD_NSTREAMS=4` (the new path). Both must print
   `RESULT: PASS` with the async block at 0.0. (COMPILE-PARITY block must also stay
   within its 3e-3 fp16 tol.)
2. `tests/test_hexfield_model.py` — model-level oracle (sdpa==materialized, pad
   inertness). Streams do not touch the model, so this guards no regression slipped
   in. Must pass green.

Both run in the hexgt-build venv.

## 6. Validate recipe (commands; baseline number to beat)

Run from WSL, GPU free, run stopped. Step (a) is implement; (b) parity; (c) bench.

**(a) Apply the prototype** to `packages/hexfield/python/hexfield/inference.py`
(the §4 diff: stream ring in `__init__`, branch in `submit_payload`, blocking H2D
in the multi-stream path). Python-only — NO Rust rebuild needed.

**(b) Parity (must pass BEFORE any bench is trusted):**
```
wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && \
  HEXFIELD_NSTREAMS=1 PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
  scripts/_hexfield_compile_overlap_test.py'
wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && \
  HEXFIELD_NSTREAMS=4 PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
  scripts/_hexfield_compile_overlap_test.py'
wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && \
  PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
  /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexfield_model.py -q'
```
GATE: both overlap-test runs print `RESULT: PASS` with the ASYNC block maxabsdiff
exactly 0.0; pytest green. If async maxabsdiff != 0.0 → the stream sync is wrong
(or the pinned-buffer hazard fired) → fix before benching.

**(c) Benchmark — establish the baseline first, then the multi-stream number:**
```
# BASELINE (single stream) — record submit/result/evaluate ms per regime:
wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && \
  HEXFIELD_NSTREAMS=1 PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
  scripts/_hexfield_serve_profile.py'
# MULTI-STREAM (sweep K=2,4,8) — same harness, same mixes:
wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && \
  HEXFIELD_NSTREAMS=4 PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
  scripts/_hexfield_serve_profile.py'
```
Repeat with `HEXFIELD_NSTREAMS=2` and `8`. The number to beat is the BASELINE
`evaluate_payload <ms>` line PER REGIME (and the `submit <ms>` host-enqueue line),
from the NSTREAMS=1 run on THIS box — the harness prints them directly. Focus on
**mid(256-768)** and **late(768-1800)** (the multi-group regimes); early has 1
group (expect no change), deep is compute-bound (expect flat or a small
regression). A win = mid/late `evaluate_payload` ms drops by more than run-to-run
noise (±~3%) at some K, AND no regime regresses materially.

Optional end-to-end confirmation IF (c) shows a mid/late win (only then worth the
heavier bench): compare self-play pos/s with the env flag on vs off —
```
wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && HEXFIELD_NSTREAMS=4 \
  PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
  scripts/_hexfield_lategame_bench.py \
  /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000031.pt 60 "64" /tmp/ms.json'
```
vs the same with the flag unset. Baseline pos/s = the unset run on this box.

## 7. Expected gain (honest)
- early: 0% (1 group).
- mid: 0–single-digit % at best; plausibly 0 (host-launch-bound dominates).
- late: 0% to small regression (GPU already ~97% busy; co-residency time-slices).
- deep: 0% to small regression.
Net realistic: ~0–3% on the mid mix, likely a wash overall. Not the lever.

## 8. Risks
- Pinned-buffer / non_blocking lifetime hazard on non-default streams → torn reads
  → parity FAIL (mitigated by blocking H2D in the multi-stream branch).
- Per-group stream + event overhead can make the launch-bound regime *slower*.
- Cross-stream contention on large-S groups → late/deep regression.
- Stream sync must precede `torch.cat`/`result()` or the single-D2H discipline
  breaks (reads half-produced tensors) → parity FAIL.

## 9. Recommendation
worth_prototyping = TRUE but LOW priority: the prototype is ~20 lines, Python-only,
gated, and gives a definitive measured yes/no in one pause. But the structural
analysis says the launch-bound regime (the target) cannot be helped by streams
(single host dispatch thread is the long pole), and the multi-group regime is
GPU-saturated. Expect a wash. The real levers remain elsewhere (host-launch
reduction via CUDA graphs / fewer kernels, or the ruled-out strength tradeoffs).
