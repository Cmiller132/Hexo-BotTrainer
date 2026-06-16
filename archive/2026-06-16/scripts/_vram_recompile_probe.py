"""Does the compiled hexgt forward fall back to EAGER under self-play's shape
diversity, and does that explain the ~11.8 GB reserved peak?

Compiles `forward_policy_value(dynamic=True)` exactly like scripts/_rl_train.py,
then feeds it a stream of DISTINCT-shaped real batches (as self-play does, one
different leaf-batch shape per move). Tracks per step:
  * torch._dynamo recompile count + whether the cache_size_limit was hit
    (after which dynamo runs the function EAGER for the rest of the process),
  * reserved / live VRAM,
so we can watch reserved jump from the compiled (~3 GB) regime to the eager
(~11 GB) regime at the moment of fallback.

Usage: python scripts/_vram_recompile_probe.py --shards <dir> [--limit 8] [--steps 24]
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch._dynamo as dynamo

from hexo_models.hexgt.architecture import HexgtNetwork, precompute_attention_layout
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.inference import _plan_chunks, _subbatch

import hexo_engine.api as eng

MB = 1024 * 1024


def gather_states(n, shard_dir, seed=1):
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
    rng.shuffle(npzs)
    out = []
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        for row in rows:
            s = reconstruct_state(row.placement_history)
            if eng.terminal(s) is None:
                out.append(s)
            if len(out) >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--limit", type=int, default=8, help="torch._dynamo cache_size_limit (default 8)")
    args = ap.parse_args()
    dynamo.config.cache_size_limit = args.limit
    dynamo.config.accumulated_cache_size_limit = max(args.limit * 8, 64)

    device = torch.device("cuda")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')}  "
          f"cache_size_limit={dynamo.config.cache_size_limit}")
    model = HexgtNetwork(token_dim=168, gnn_layers=4, ctx_layers=3, attention_heads=4,
                         ffn_dim=336, short_term_value_horizons=(4, 12, 24), value_pma_seeds=2,
                         ).to(device).eval()
    model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)

    states = gather_states(4096, args.shards, seed=1)
    print(f"loaded {len(states)} states")

    # Build a series of DISTINCT-shaped sub-batches (vary graph count per step).
    sizes = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224,
             256, 320, 384, 448, 512, 640, 768, 1024]
    print(f"\n{'step':>4} {'graphs':>7} {'nodes':>7} {'cands':>7} {'recompiles':>11} "
          f"{'live_MB':>9} {'reserved_MB':>12} {'smi-ish':>8}")
    prev_recompiles = 0
    fell_back_at = None
    for i in range(min(args.steps, len(sizes))):
        ng = sizes[i]
        sub_states = states[:ng]
        if len(sub_states) < ng:
            break
        b = batch_from_states(sub_states, 3)
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        num_graphs = int(b["num_graphs"])
        b = {**b, **precompute_attention_layout(b["node_graph"], b["node_type"], num_graphs)}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model.forward_policy_value(b)
        torch.cuda.synchronize()
        recompiles = dynamo.utils.counters["frames"].get("ok", 0) + \
            dynamo.utils.counters["frames"].get("total", 0)
        # A cleaner recompile signal: number of guard failures / recompiles.
        rc = sum(v for k, v in dynamo.utils.counters.get("graph_break", {}).items())
        recompile_count = dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0)
        live = torch.cuda.max_memory_allocated() / MB
        reserved = torch.cuda.max_memory_reserved() / MB
        print(f"{i:>4} {num_graphs:>7} {int(b['node_feat'].shape[0]):>7} "
              f"{int(b['candidate_index'].shape[0]):>7} {recompile_count:>11} "
              f"{live:>9.1f} {reserved:>12.1f} {reserved:>8.0f}", flush=True)
        del b
        torch.cuda.empty_cache()

    print("\n=== dynamo counters (recompile / fallback evidence) ===")
    import json
    cnt = {k: dict(v) for k, v in dynamo.utils.counters.items()}
    print(json.dumps(cnt, indent=2, default=str)[:4000])
    print(f"\nfinal reserved peak = {torch.cuda.max_memory_reserved()/MB:.1f} MiB "
          f"(live peak = {torch.cuda.max_memory_allocated()/MB:.1f} MiB)")


if __name__ == "__main__":
    main()
