"""Per-phase throughput: the realistic self-play case is a batch of leaves from
ONE search tree -> nearly-uniform graph sizes -> minimal attention padding waste.
The mixed-position profile (_profile_hexgt_forward) understates this. Here we sort
the position pool by candidate count and measure UNIFORM contiguous-size batches
in opening/midgame/endgame tertiles, eager + compiled, with padding efficiency.

Usage: python scripts/_profile_hexgt_phases.py --shards <dir> [--batch 512]
"""

from __future__ import annotations

import argparse
import time

import torch

from hexo_models.hexgt.architecture import HexgtNetwork, build_attention_layout
from hexo_models.hexgt.constants import (
    DEFAULT_ATTENTION_HEADS, DEFAULT_CTX_LAYERS, DEFAULT_FFN_DIM, DEFAULT_GNN_LAYERS, DEFAULT_TOKEN_DIM,
    NODE_TYPE_CANDIDATE,
)
from hexo_models.hexgt.graph_build import batch_from_states
import hexo_engine.api as eng


def _states_from_shards(n_states, shard_dir, seed=1):
    from pathlib import Path
    import random
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
    epochs = sorted({p.name.split("_game_")[0] for p in npzs})
    if len(epochs) > 1:
        npzs = [p for p in npzs if p.name.split("_game_")[0] != epochs[-1]]
    rng.shuffle(npzs)
    states = []
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        for row in rows:
            states.append(reconstruct_state(row.placement_history))
            if len(states) >= n_states:
                return states
    return states


def _time(fn, iters, device):
    en = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=en):
        for _ in range(3):
            fn()
    if en:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=en):
        for _ in range(iters):
            fn()
    if en:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--shards", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = HexgtNetwork(
        token_dim=DEFAULT_TOKEN_DIM, gnn_layers=DEFAULT_GNN_LAYERS, ctx_layers=DEFAULT_CTX_LAYERS,
        attention_heads=DEFAULT_ATTENTION_HEADS, ffn_dim=DEFAULT_FFN_DIM,
        short_term_value_horizons=(),
    ).to(device).eval()
    print(f"params={sum(p.numel() for p in model.parameters()):,} device={device} batch={args.batch}")

    pool = _states_from_shards(args.pool, args.shards)
    # candidate count per state (proxy for board fullness / phase)
    sized = []
    for s in pool:
        nc = len(list(eng.legal_actions(s)))  # n=3 candidate count <= legal; use legal as cheap proxy ordering
        sized.append((nc, s))
    sized.sort(key=lambda t: t[0])
    states_sorted = [s for _, s in sized]
    N = len(states_sorted)
    print(f"pool={N} states, sorted by size")

    cfwd = torch.compile(model.forward_policy_value, dynamic=True)

    # tertiles -> take a contiguous (uniform-size) window of `batch` centered in each
    print(f"\n{'phase':>8} | {'cands/pos':>9} | {'ctx_eff':>7} | {'cand_eff':>8} | {'eager pos/s':>11} | {'compiled pos/s':>14}")
    tertiles = {"opening": 0.12, "midgame": 0.5, "endgame": 0.92}
    for name, frac in tertiles.items():
        center = int(frac * N)
        lo = max(0, min(center - args.batch // 2, N - args.batch))
        states = states_sorted[lo:lo + args.batch]
        batch = batch_from_states(states, args.n)
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        num_graphs = int(b["num_graphs"])
        cands = int(b["candidate_index"].shape[0])
        is_cand = b["node_type"] == NODE_TYPE_CANDIDATE
        layout = build_attention_layout(b["node_graph"], is_cand, num_graphs)
        ctx_eff = layout.ctx_valid.sum().item() / layout.ctx_index.numel()
        cand_eff = layout.cand_valid.sum().item() / max(1, layout.cand_index.numel())
        t_eager = _time(lambda: model.forward_policy_value(b), args.iters, device)
        t_comp = _time(lambda: cfwd(b), args.iters, device)
        print(f"{name:>8} | {cands/args.batch:>9.0f} | {ctx_eff:>6.1%} | {cand_eff:>7.1%} | "
              f"{args.batch/(t_eager/1000):>11.0f} | {args.batch/(t_comp/1000):>14.0f}")

    if device.type == "cuda":
        print(f"\npeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
