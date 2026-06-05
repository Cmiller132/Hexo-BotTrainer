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
