"""Breakdown profiler: where does the hexgt forward spend time, and how much
attention padding waste is there? Splits node_in / GNN / transformer / heads and
reports padding efficiency (real tokens / padded tokens). Also sweeps torch.compile
modes. Realistic n=3 96x8 positions from shards.

Usage: python scripts/_profile_hexgt_breakdown.py --shards <dir> [--batch 512]
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
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(3):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(iters):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--shards", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = HexgtNetwork(
        token_dim=DEFAULT_TOKEN_DIM, gnn_layers=DEFAULT_GNN_LAYERS, ctx_layers=DEFAULT_CTX_LAYERS,
        attention_heads=DEFAULT_ATTENTION_HEADS, ffn_dim=DEFAULT_FFN_DIM,
        short_term_value_horizons=(),
    ).to(device).eval()

    pool = _states_from_shards(args.batch, args.shards)
    batch = batch_from_states(pool[:args.batch], args.n)
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    num_graphs = int(b["num_graphs"])
    nodes = int(b["node_feat"].shape[0])
    cands = int(b["candidate_index"].shape[0])
    edges = int(b["edge_index"].shape[1])
    print(f"device={device} batch={args.batch} nodes={nodes} cands={cands} edges={edges}")

    # padding efficiency
    is_cand = b["node_type"] == NODE_TYPE_CANDIDATE
    layout = build_attention_layout(b["node_graph"], is_cand, num_graphs)
    ctx_real = int(layout.ctx_valid.sum().item())
    ctx_slots = layout.ctx_index.numel()
    cand_real = int(layout.cand_valid.sum().item())
    cand_slots = layout.cand_index.numel()
    print(f"ctx tokens: real={ctx_real} slots={ctx_slots} eff={ctx_real/ctx_slots:.1%} (max_ctx={layout.ctx_index.shape[1]})")
    print(f"cand tokens: real={cand_real} slots={cand_slots} eff={cand_real/cand_slots:.1%} (max_cand={layout.cand_index.shape[1]})")

    # component timing (eager)
    edge_index = b["edge_index"]; edge_type = b["edge_type"]; edge_attr = b["edge_attr"]

    def node_in():
        return model.node_in(b["node_feat"])

    h0 = model.node_in(b["node_feat"])

    def gnn_only():
        h = h0
        for layer in model.gnn:
            h = layer(h, edge_index, edge_type, edge_attr)
        return h

    h_after_gnn = gnn_only()

    def transformer_only():
        h = h_after_gnn
        for layer in model.transformer:
            h = layer(h, layout)
        return h

    def heads_only():
        return model._heads(b, h_after_gnn, with_aux=False)

    def full():
        return model.forward_policy_value(b)

    t_ni = _time(node_in, args.iters, device)
    t_gnn = _time(gnn_only, args.iters, device)
    t_tf = _time(transformer_only, args.iters, device)
    t_hd = _time(heads_only, args.iters, device)
    t_full = _time(full, args.iters, device)
    print(f"\n--- eager component ms (batch {args.batch}) ---")
    print(f"node_in     {t_ni:7.2f}")
    print(f"gnn (x{model.gnn_layers})    {t_gnn:7.2f}")
    print(f"transformer {t_tf:7.2f}  (x{model.ctx_layers})")
    print(f"heads       {t_hd:7.2f}")
    print(f"FULL        {t_full:7.2f}  -> {args.batch/(t_full/1000):.0f} pos/s")

    # compile modes
    print(f"\n--- compile modes (FULL forward, batch {args.batch}) ---")
    for mode_name, kw in [("dynamic", dict(dynamic=True)),
                          ("default-static", dict(dynamic=False)),
                          ("reduce-overhead", dict(mode="reduce-overhead", dynamic=False)),
                          ("max-autotune", dict(mode="max-autotune", dynamic=False))]:
        try:
            torch._dynamo.reset()
            cfwd = torch.compile(model.forward_policy_value, **kw)
            t = _time(lambda: cfwd(b), args.iters, device)
            print(f"{mode_name:18s} {t:7.2f} ms -> {args.batch/(t/1000):.0f} pos/s")
        except Exception as e:
            print(f"{mode_name:18s} FAILED: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
