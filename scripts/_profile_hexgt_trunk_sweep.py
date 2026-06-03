"""Trunk speed knob: sweep (token_dim, gnn_layers, ctx_layers) configs near the
~2.0M-param budget and measure midgame+endgame compiled throughput. Answers the
plan's 'characterize a shallower/narrower trunk speed knob (keep a ~2.07M-param
variant)'. Uses uniform same-phase batches (the realistic self-play case).

Usage: python scripts/_profile_hexgt_trunk_sweep.py --shards <dir> [--batch 512]
"""

from __future__ import annotations

import argparse
import time

import torch

from hexo_models.hexgt.architecture import HexgtNetwork
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
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--shards", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    pool = _states_from_shards(args.pool, args.shards)
    sized = sorted(pool, key=lambda s: len(list(eng.legal_actions(s))))
    N = len(sized)

    def phase_batch(frac):
        center = int(frac * N)
        lo = max(0, min(center - args.batch // 2, N - args.batch))
        b = batch_from_states(sized[lo:lo + args.batch], args.n)
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}

    mid = phase_batch(0.5)
    end = phase_batch(0.92)

    configs = [
        ("ref  168/g3/c3", dict(token_dim=168, gnn_layers=3, ctx_layers=3, ffn_dim=336)),
        ("ctx2 168/g3/c2", dict(token_dim=168, gnn_layers=3, ctx_layers=2, ffn_dim=336)),
        ("shal 192/g2/c2", dict(token_dim=192, gnn_layers=2, ctx_layers=2, ffn_dim=384)),
        ("wide 224/g2/c2", dict(token_dim=224, gnn_layers=2, ctx_layers=2, ffn_dim=448)),
        ("ndeep144/g3/c4", dict(token_dim=144, gnn_layers=3, ctx_layers=4, ffn_dim=288)),
        ("g2c3 176/g2/c3", dict(token_dim=176, gnn_layers=2, ctx_layers=3, ffn_dim=352)),
    ]
    print(f"device={device} batch={args.batch} (compiled dynamic, uniform same-phase)\n")
    print(f"{'config':>16} | {'params':>9} | {'mid pos/s':>10} | {'mid impl':>8} | {'end pos/s':>10} | {'end impl':>8}")
    for name, kw in configs:
        model = HexgtNetwork(attention_heads=4, short_term_value_horizons=(), **kw).to(device).eval()
        nparams = sum(p.numel() for p in model.parameters())
        cfwd = torch.compile(model.forward_policy_value, dynamic=True)
        t_mid = _time(lambda: cfwd(mid), args.iters, device)
        t_end = _time(lambda: cfwd(end), args.iters, device)
        mid_ps = args.batch / (t_mid / 1000)
        end_ps = args.batch / (t_end / 1000)
        print(f"{name:>16} | {nparams:>9,} | {mid_ps:>10.0f} | {mid_ps/512:>8.1f} | {end_ps:>10.0f} | {end_ps/512:>8.1f}")
        del model, cfwd
        torch._dynamo.reset()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
