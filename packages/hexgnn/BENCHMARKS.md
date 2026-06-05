# hexgnn benchmarks + the 200 pos/s achievability verdict

Hardware: RTX 4070 Ti (12 GB), torch 2.12+cu130, compiled (dynamic) + fp16.
All numbers measured with the live run permanently halted (GPU free).

## Parameter counts (forward sweep, all <500k)

| config | params | fwd ms @b256 | fwd ms @b512 |
|---|--:|--:|--:|
| td168/gnn4 (original hexgnn) | 931,459 | 62.7 | — |
| **td128/gnn3** | **445,635** | 33.8 | 66.2 |
| td128/gnn2 | 345,411 | 24.6 | 45.9 |
| td112/gnn3 | 343,347 | 31.0 | 59.7 |
| td96/gnn3 | 254,371 | 27.5 | 52.7 |
| **td96/gnn2** | **197,635** | — | — |
| (hexgt model-3, for ref) | 2,584,774 | — | — |

Fewer GNN layers cuts latency more than narrowing dims (each layer is a
sync-heavy per-edge-type einsum + scatter).

## Forward speedup vs model-3 (GPU, compiled fp16)
hexgnn-931k vs model-3 at b128: 30.6 ms vs 68.0 ms = **2.2x** forward (CPU was 1.58x;
the transformer's padded attention costs relatively more on GPU). A <500k model is
faster still.

## Self-play pos/s at active=512 (the production batching)

Opening-region samples (max_actions 16-24, num_games=512); these OVERESTIMATE
full-game throughput (the rate decays with ply as midgame graphs grow — the
td128/gnn3 run below decayed 77 -> 20 pos/s over 24 plies). Full-game steady-state
is roughly 0.4-0.6x these.

| model | visits | PCR | mean v/move | pos/s (opening) |
|---|--:|---|--:|--:|
| td128/gnn3 (445k) | 512 | off | 512 | 19.6 |
| td96/gnn2 (197k) | 512 | p=0.5 fast=85 | 301 | 47.7 |
| td96/gnn2 (197k) | 128 | off | 128 | 135.3 |
| td96/gnn2 (197k) | 64 | off | 64 | **242.1** |

## Verdict: 200 pos/s @ 512 visits is NOT achievable
pos/s × avg_visits = NN leaf-evals/s, and that throughput is capped by a fixed
per-sim cost (Rust featurize + graph build + engine state clone + GNN forward).
GPU util was only ~49% at active=512/visits=512 — the loop is partly CPU-bound on
the MCTS, so a smaller model buys ~1.5x, not 10x. At 512 visits the realistic
ceiling is ~20 pos/s (td128/gnn3) to ~30-50 pos/s (td96/gnn2 + cheap PCR), i.e.
~3-7x the old model-3 (~6-8 pos/s) — a real win, but not 200.

## Detailed profiling — where the time goes (td128/gnn3, active=512, visits=512)

### (1) Wall-time attribution (instrumented self-play, ~ply-18 sample, pos/s=30.2)
| phase | %wall | notes |
|---|--:|---|
| NN forward (H2D + GNN compute) [GPU] | **59.9%** | the single biggest bucket; GROWS with ply |
| Rust MCTS select/expand/backup + featurize + serialize + py loop | **38.5%** | CPU |
| NN eval: build (frombuffer -> tensors) [CPU] | 0.1% | negligible (zero-copy buffers) |
| NN eval: post (softmax/decode + D2H + tobytes) | 1.4% | negligible |

So it is BOTH-bound but **forward-dominant (~60%) with a hard ~38% CPU/Rust floor**. The
synchronous MCTS design has no GPU↔CPU overlap (Rust must back up the eval result
before the next select), so the two phases are sequential — GPU sits idle (~0%, 8 W)
during the 38% Rust phase and bursts during the 60% forward phase (avg util ~50-60%).

### (2) Is the forward launch-/memory-/compute-bound? (torch profiler, eager op mix)
Matmuls (addmm + sgemm) are only ~13-26% of forward CUDA time. The rest is
**elementwise + scatter/gather over the EDGE-scale tensors**: aten::add ~29%,
index_add_ (the message scatter) ~13-16%, aten::index (gather) ~10%, clamp_min ~15%,
copy_ ~9%. At 445k params the matmul FLOPs are trivial — the cost is moving the
`(edges × token_dim)` message tensors and scattering them => **memory-bandwidth-bound**,
plus **kernel-launch pressure** in eager ("Command Buffer Full" 21-63%, hundreds of
small kernels/forward). torch.compile (production) already fuses much of it:
eager->compiled = 36.6->11.8 ms (opening, 3.1x) and 257->115 ms (ply-60, 2.2x).
=> A narrower/shallower model helps the GPU half (per-edge vector + #layers) but NOT
the edge COUNT or scatter-index overhead, so it buys ~1.5x, not 10x.

### (3) Why pos/s decays 77 -> 20 as games leave the opening
Per-leaf graph size grows ~7x by ply 60 (radius-3 candidate set + window-hub edges
scale with accumulated stones/active windows):
| | nodes/leaf | edges/leaf | cand/leaf | compiled fwd ms (512 leaves) |
|---|--:|--:|--:|--:|
| opening ~6 plies | 222 | 1,499 | 215 | 11.8 |
| midgame ~60 plies | 1,569 | 11,420 | 1,507 | 115.4 |
Forward is O(edges) memory-bound and featurize is O(nodes+edges), so each position
costs ~7-10x more by midgame. Rust featurizer alone: 12.3 ms (opening) -> 85.7 ms
(midgame) per 512 states. The cheap opening inflates the cumulative pos/s, which
decays toward the midgame steady-state.

### (4) Revised ceilings @512 visits IF the top fixes were applied
Both buckets must improve (Amdahl over 60% forward / 38% Rust):
- torch.compile: ALREADY ON (2.2-3x vs eager).
- CUDA graphs (cut launch overhead; needs shape bucketing): ~1.2-1.3x forward.
- Fused relational-message scatter kernel (torch_scatter segment-matmul / custom):
  ~1.5-2x forward (kills the index_add_/index/add memory passes).
- Rust-side graph-build batching + caching the static board adjacency across a
  position's sims: ~1.2-1.4x on the Rust 38%.
Stacking the plausible GPU fixes (~1.8x forward) + Rust (~1.3x): total ~1.55x ->
config (B) ~30 pos/s(early)/~15-20(full) becomes ~45 pos/s(early)/~25-35(full).
**Even fully fixed, config (B) @512 visits reaches ~30-50 pos/s, NOT 100+.** Hitting
100+ @512 visits needs a fundamentally SPARSER graph (fewer edges/candidates — a
representation change), because total work = positions x visits x O(edges) and the
midgame edge explosion dominates. The only reliable path to >=200 stays **lower
visits** (~64).

**What reaches >=200 pos/s:** drop the AVERAGE visits to ~64. td96/gnn2 @ visits=64
measured 242 pos/s (opening; ~120-180 full-game est). visits=128 -> 135 (opening).
PCR helps at HIGH visits (lowers the average) but its two-call split adds overhead
and gives little at low visits, so for a 200-target run prefer plain low visits.

### Two launch-ready configs
- **Throughput (hits ~200):** td96/gnn2 (197k), visits=64, PCR off.
  Caveat: 2 GNN layers = shallow threat propagation; 64 visits is a low search.
- **Balanced @512 (does NOT hit 200):** td128/gnn3 (445k), visits=512, PCR p=0.5
  fast=85 -> ~20-30 pos/s full-game but much stronger per-move search.
The owner picks the visits/strength-vs-throughput tradeoff (200 pos/s ⟺ ~visits 64).
