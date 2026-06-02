# HEXGT (Model 2) — Self-Play Performance Profile at 512 MCTS sims

Detailed performance profile of the `hexgt` dynamic-GNN self-play search at the
**512-simulation** config we are moving to (`configs/hexgt_model2.toml`
`[selfplay] search_visits = 512`). Profiling only — no model or run changes.

- **Worktree / branch:** `E:\Hexo-BotTrainer-hexgt` (`/mnt/e/...`), `hexgt-rewrite`
  @ `1eaef5d` ("strict dead-cell exclusion in the shared candidate builder").
- **Build:** `maturin develop --release` (rebuilt to match the committed Rust);
  full suite **196 passed**.
- **Hardware:** RTX 4070 Ti (12 GB), Ryzen 7950X (16c/32t), 28 GB RAM visible to
  WSL2; torch 2.12 + CUDA 13.0, FP16 autocast, `torch.compile(dynamic=True)`.
- **GPU coordination:** the live RL run was stopped first; nvidia-smi confirmed
  GPU idle (0 % util, 765 MB desktop-only) before profiling. No concurrent run.
- **Workload:** real 96x8 / RL self-play positions reconstructed from
  `runs/hexgt_rl_main/selfplay` shards; native `HexgtMctsSession` driven over
  64 (and 256) concurrent games × several moves at the production C1 search
  params (c_puct 1.5, Dirichlet α_total 10.83 / ε 0.25, root_policy_temp 1.1,
  fpu 0.20, nucleus widening mass 0.95 / max_children 96, forced_playout_k 2,
  candidate_radius n=3), vbatch=64, compile on. Tooling: Rust stage timers
  (six buckets) + cuda-event GPU split + `torch.profiler` + background
  `/proc/stat` (CPU), `nvidia-smi` (GPU), `/proc/self/status` (RAM) samplers.
  Script: `scripts/_perf_512.py`.

> **Note on variance.** FP16 + `torch.compile` make the forward numerically
> non-deterministic, so tree shape (and thus unique-forward count) varies run to
> run. Three 512-sim/active=64 measurements gave **9.59 / 7.61 / 9.31** searched
> pos/s; treat the central estimate as **≈8.5 pos/s (±15 %)**.

---

## 1. Executive summary

| | value (512 sims, active=64, vbatch=64, compile) |
|---|---|
| **Searched / self-play positions/sec** | **≈8.5** (range 7.6–9.6) |
| NN forwards per searched position | **~400–430** of 512 sims (cache hit ~16 %) |
| **GPU utilization** | mean **38–46 %**, peak 97–99 % → **starved ~55 %** of the time |
| **CPU utilization** | mean **~7 %** (~2 of 32 threads) → **31× idle headroom** |
| VRAM (alloc / reserved peak) | **1.6 / 1.8 GB** — trivial vs the 12 GB ceiling |
| System RAM (process peak) | **2.7 GB** (active=64) / 5.7 GB (active=256) |

**There is no single dominant cost. Search time splits almost evenly between the
GPU forward (~44 %) and Rust CPU featurization/encode (~35 %), with host↔device
transport + value/softmax decode (~13 %) and Rust prior post-processing (~7 %)
making up the rest.** Crucially, these stages run **sequentially on one
GIL-holding thread** (encode → forward → parse, per chunk), so **both the GPU
(~40 %) and the CPU (~7 %) sit idle for roughly half the wall-clock each** — they
never overlap. Neither VRAM nor RAM is anywhere near a limit.

**The single biggest opportunity is therefore structural, not per-kernel:
pipeline featurization against the GPU forward (the deferred Phase-7 async
batcher) and spread featurization across the 32 threads.** Either alone recovers
a large fraction of the ~50 % idle on its resource; together they could roughly
double self-play throughput at 512 sims with no model change.

---

## 2. Time breakdown (where the time goes)

Per-stage attribution from the Rust `Instant` timers summed over the timed moves,
as a fraction of wall-clock. Consolidated from the clean run (real wall) and the
cuda-event GPU-split run (which isolates pure GPU compute inside the evaluator).

**512 sims, active=64, vbatch=64, compile — consolidated:**

| stage | % of wall | what it is | where |
|---|---:|---|---|
| **NN forward (GPU compute)** | **~44 %** | GNN + transformer fp16 forward, chunked | GPU |
| **Rust featurize + graph-encode** | **~35 %** | `build_graph` (candidate∪window∪edge) + D6-invariant node featurize + disjoint collate (`features.rs`, `.par_iter()` + serial `collate`) | CPU |
| **Transport + softmax + D2H + serialize** | **~13 %** | `torch.frombuffer` → pageable HtoD copy, per-graph log-softmax, value decode, `.cpu().numpy()` D2H, `tobytes` back to Rust | CPU+GPU |
| **Rust parse (legality / sort / normalize)** | **~7 %** | intersect priors with engine legality, descending-sort, renormalize per leaf | CPU |
| **MCTS tree ops (select / expand / backup)** | **~1 %** | PUCT selection, nucleus expansion, virtual-loss backup, subtree reuse | CPU |
| **dedup (transposition) + cache insert** | **<0.1 %** | leaf dedup before eval, bounded-LRU insert | CPU |

Raw measured numbers feeding the consolidation:

| | clean active=64 | gpu-split active=64 | clean active=256 |
|---|---:|---:|---:|
| wall (s) | 33.65 | 20.61 | 99.48 |
| Rust featurize+encode | 32.9 % | 37.2 % | 38.8 % |
| Py evaluator (total) | 59.6 % | 54.6 % | 55.1 % |
| — pure GPU forward | — | **41.9 %** (77 % of evaluator) | — |
| — transport + post | — | 12.7 % (23 % of evaluator) | — |
| Rust parse | 6.6 % | 7.3 % | 5.4 % |
| dedup + cache insert | <0.1 % | <0.1 % | 0.1 % |
| MCTS tree ops (remainder) | 0.9 % | 0.9 % | 0.6 % |

The cost structure is **scale-invariant**: the per-stage percentages are
essentially identical at 128 sims (encode 35.6 %, evaluator 55.7 %, parse 7.4 %)
and at active=256 (encode 38.8 %, evaluator 55.1 %, parse 5.4 %). The split is a
property of one search step, not of the sim count or concurrency.

### GPU kernel attribution (`torch.profiler`, one 512-sim round, 30 forwards)

Self CUDA time total 1.473 s. Top kernels by CUDA time:

| kernel | CUDA time | % | interpretation |
|---|---:|---:|---|
| `triton_poi_fused_add_bmm_clamp_index_index_add…` | 232 ms | 15.8 % | fused GNN message-pass (`index_add` scatter) + attention `bmm` |
| `aten::addmm` | 194 ms | 13.2 % | linear layers (FFN + projections) |
| `aten::bmm` | 168 ms | 11.4 % | batched matmul (attention scores/context) |
| `aten::copy_` + `Memcpy HtoD (Pageable→Device)` | 158 + 157 ms | **10.7 % + 10.6 %** | **host→device transfer of the featurized batch (pageable, not pinned)** |
| `ampere_sgemm_128x128_nn` | 137 ms | **9.3 %** | **fp32 GEMM running under autocast (not fp16)** |
| `triton_poi_fused__to_copy_add_bmm…` | 64 ms | 4.3 % | more fused GNN/attention |
| `aten::mm` | 61 ms | 4.1 % | matmul |

Two non-compute items stand out on the GPU timeline: **~10.6 % is pageable HtoD
memcpy** (the zero-copy Rust buffers are still copied host→device each forward,
unpinned), and **~9.3 % is an fp32 sgemm** that autocast did not lower to fp16.
The genuine model FLOPs are dominated by the transformer/GNN `bmm` + `addmm`
path, consistent with the Phase-5b finding that the transformer (not the GNN
einsum) is the heaviest model component.

---

## 3. Throughput, GPU & CPU utilization

| config (512 sims, vbatch=64, compile) | pos/s | fwd/pos | cache hit | unique fwd/s | GPU util (mean/peak) | CPU util (mean) | cores busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **active=64** (task/run config) | **7.6–9.6** | 385–427 | 16–18 % | ~3250 | **38–46 % / 99 %** | **7 %** | ~1–2 / 32 |
| active=256 (config-file default) | **5.15** | 373 | 27 % | ~1920 | 40 % / 97 % | 7 % | ~1–2 / 32 |

- **Searched pos/s ≈ self-play pos/s.** Every searched root is one played move =
  one training sample, so the search rate is the self-play sample rate (the real
  loop adds only minor sample-finalize/shard-write/refill overhead on top).
- **GPU is starved**, not saturated: mean 38–46 % util with brief 97–99 % peaks
  during the actual forward chunks, idle in between while the CPU featurizes,
  transports and parses.
- **CPU is almost entirely idle**: ~7 % system-wide ≈ 2 of 32 logical cores. The
  `features.rs` `.par_iter()` featurizer is **not** translating into broad
  multicore use in the live loop — the per-chunk work + serial `collate()` +
  per-call overhead keep effective parallelism to ~2–3 cores. This directly
  contradicts the standing "maximize 32-thread CPU" directive and is pure
  headroom.
- **More concurrency does not help at 512 sims.** active=256 (the config-file
  default) is *slower* than active=64 (5.15 vs ~8.5 pos/s) and uses 2× RAM, even
  though its cache-hit rate is higher (27 % vs 16 %). Because the eval loop is
  serial-CPU-bound rather than GPU-bound, adding games just adds proportional
  per-chunk work (more 1024-capped chunks processed sequentially) without
  filling the idle GPU. The earlier "concurrent-game count is THE lever" result
  held only at the GPU-starved small-batch operating point (8→64 games); past
  ~64 it has saturated.

### Scaling 128 → 512 sims

| | 128 sims | 512 sims | ratio |
|---|---:|---:|---:|
| searched pos/s (active=64) | 37.3 | 7.6 | **4.9× slower** for 4× sims |
| fwd/pos | 108 / 128 | 427 / 512 | 3.94× (≈ linear in sims) |
| cache hit | 15.3 % | 16.0 % | ~flat |

Cost is **~linear in sims with a mild super-linear penalty** (4.9× for 4×): NN
forwards grow ~linearly (cache assist is sim-count-independent at ~16 %), and the
extra is deeper/larger trees (node count 8.5k→33.6k, chunks/move 29→29 but bigger
candidate sets). **There is no large fixed per-search overhead** — the MCTS tree
ops + dedup + cache are <2 % combined. So 512 sims simply costs ~4× the forwards
of 128 sims, plus a little tree growth.

---

## 4. Memory

### VRAM — not a constraint

| config | torch alloc peak | torch reserved peak | nvidia-smi peak |
|---|---:|---:|---:|
| 512 / active=64 | 1.62 GB | 1.82 GB | 2.74 GB |
| 512 / active=256 | 2.32 GB | 2.64 GB | 3.52 GB |
| 128 / active=64 | 1.65 GB | 1.97 GB | 2.88 GB |

The sorted-chunked forward (`forward_pad_budget=200k`, the Phase-10 VRAM
compression) caps the forward at `max_chunk_states ≈ 1024` graphs, so peak VRAM
is tiny — **~1.6–2.6 GB allocated against a 12 GB ceiling (>9 GB free)**.
nvidia-smi adds the ~0.8 GB CUDA/driver context. Drivers are the model + the
single chunk's padded candidate×context attention tensors; the MCTS tree itself
is negligible on-GPU. The notes warn that the `torch.compile` shape-variant cache
grows reserved VRAM toward ~8 GB over many epochs — not observed in a short probe
(reserved stayed ≤2.6 GB) but worth watching over a 60-epoch run; ample headroom
remains. **The chunk budget is now far too conservative for 512/active=64** — see
opportunity #5.

### System RAM — not a constraint

| config | process RSS peak | system used peak | drivers |
|---|---:|---:|---:|
| 512 / active=64 | 2.7 GB | 3.5 GB | torch+model ~1.5 GB, transposition cache (141k entries), states |
| 512 / active=256 | 5.7 GB | 6.2 GB | transposition cache (320k entries) + 256 trees (133k nodes, but only 5.3 MB edge bytes) |

RAM is dominated by the **bounded transposition cache** (it scales with the
unique states seen — 141k→320k entries between active=64 and 256) and the torch
runtime. The MCTS trees themselves are cheap (33k–133k nodes = 1.3–5.3 MB of
active-edge bytes). Peak system use ≤6.2 GB of 28 GB → **>21 GB free**. The
replay-window / shard pipeline is not exercised by the search probe (it lives in
training, not self-play search).

---

## 5. Bottleneck & ranked optimization opportunities

**Bottleneck:** the search loop is **co-bottlenecked and serialized**. The single
largest atomic cost is the GPU forward (~44 % of wall), but Rust CPU
featurization is nearly equal (~35 %), and because the per-chunk stages
(encode → transport → forward → parse) run sequentially on one GIL thread,
**neither the GPU (40 % util) nor the CPU (7 % util) is more than ~half busy.**
The highest-leverage fixes attack the *serialization and under-parallelism*, not
any one kernel.

Ranked by expected payoff:

1. **Pipeline featurization against the GPU forward (async / double-buffered
   leaf eval — the deferred Phase-7 batcher).** *Expected: up to ~1.5–2× pos/s
   (≈8.5 → 13–17).* Today encode (35 %) and forward (44 %) are sequential; the
   GPU is idle during encode/parse and the CPU is idle during the forward.
   Overlapping chunk N's GPU forward with chunk N+1's CPU featurization hides the
   smaller stage behind the larger. The HEXGT_DECISIONS Phase-7 doc deferred this
   because the sync path beat dense_cnn *at 128 sims*; the 512-sim data (both
   resources ~half-idle) is exactly the condition that justifies building it.
   Lower-risk partial version: a 2-stage producer/consumer (one thread featurizes
   ahead, one thread submits forwards) without the full multi-game async batcher.

2. **Actually parallelize featurization across the 32 threads.** *Expected:
   encode 35 % → ~7–10 % of wall (~1.3–1.4× pos/s standalone; compounds with #1).*
   `featurize_collate_states` uses `.par_iter()` yet only ~2–3 cores are active.
   Investigate: (a) `collate()` is **serial** (concatenation of all per-graph
   tensors) and may dominate the encode bucket — parallelize the copy or
   pre-size+scatter; (b) per-chunk state counts may be small/variable so rayon
   work-stealing under-fills — coalesce more leaves per featurize call; (c)
   confirm the rayon global pool sees 32 threads under WSL (`RAYON_NUM_THREADS`).
   This is the standing "maximize CPU multithreading" directive, currently unmet.

3. **Cut GPU forward cost (the ~44 % stage), in payoff order:**
   - **Pinned-memory / on-device HtoD** — ~10.6 % of GPU time is *pageable* HtoD
     memcpy of the featurized buffers each forward. Emit the Rust buffers into
     pinned (page-locked) memory, or stage to a reusable pinned staging tensor,
     to roughly halve that transfer. *~3–5 % wall.*
   - **fp16 the residual fp32 sgemm** — `ampere_sgemm_128x128_nn` is ~9.3 % of
     GPU time running fp32 under autocast. Find the op autocast left in fp32
     (likely an einsum / matmul not on the autocast allow-list) and cast it.
     *~2–4 % wall.*
   - **Shallower trunk** — Phase-5b showed `ctx_layers` is the throughput lever
     (ctx3→ctx2 = 1.31×, shallow-wide 192/g2/c2 = 1.67× at fewer params). A
     learnability bet, validated by BC/RL eval, but the single biggest model-side
     forward win if it holds. *up to ~1.3–1.7× the forward stage.*
   - Attention/GNN op fusion is largely already done (the dominant kernels are
     fused triton); further fusion is low-value vs the above.

4. **Raise `forward_pad_budget` / chunk size (cheap tuning).** *Expected: a few
   % from fewer kernel launches + better GPU occupancy.* VRAM peaks at 1.6 GB of
   12 GB at active=64, yet chunks are capped at ~1024 graphs. The Phase-10
   200k-slot budget was tuned for the OOM/spill regime at large active counts;
   for 512/active=64 it leaves >9 GB unused. Larger chunks mean fewer
   encode/transport/launch round-trips per round. (Keep the cap modest at
   active=256 where VRAM rises.)

5. **Do not adopt `active_games=256`; keep active≈64.** *Expected: avoid a ~35 %
   throughput regression + 2× RAM.* The config file currently sets
   `active_games=256`, but it measured **5.15 pos/s vs ~8.5 at active=64**, for no
   GPU-util gain and 2× RAM, because the loop is serial-CPU-bound. (After #1/#2
   make the GPU the bottleneck, higher concurrency becomes useful again — revisit
   then.) **This is a config change to make before the 512-sim run, separate from
   the perf work.**

6. **Reduce redundant forwards via cache/policy (model-dependent).** fwd/pos is
   ~400–430 of 512 sims at only ~16 % cache hit. A trained, sharper policy
   produces more transpositions and deeper subtree reuse (active=256 already
   showed 27 % hit from cross-tree sharing). Not a pure-perf lever, but every
   point of cache hit is a proportional forward saving; worth tracking as the
   model converges.

### What is *not* worth doing

- **VRAM compression / OOM mitigation** — VRAM is at ~15 % of capacity; the
  chunking is over-conservative here (see #4).
- **RAM-compaction of trees** — trees are MB-scale; the cache (bounded) and torch
  dominate RAM, and there is >21 GB free.
- **MCTS tree-op optimization** — selection/expansion/backup/dedup/cache are
  <2 % combined.

---

## Appendix — reproduce

```bash
# in the hexgt-rewrite worktree, hexgt-build venv, GPU free:
python -u scripts/_perf_512.py \
  --shards runs/hexgt_rl_main/selfplay \
  --visits 512 --active 64 --vbatch 64 --moves 4 \
  --compile --gpu-split --profiler --also-128
# active=256 variant: --active 256 --moves 2
```

Logs: `runs/_perf_512.log`, `runs/_perf_512_a256.log`.
