"""GPU forward micro-benchmark: hexgnn vs hexgt model-3, compiled + fp16.

Times the search forward (`forward_policy_value`) on the GPU exactly as self-play
runs it: eager SIDE-rows/attention-layout precompute attached, fp16 autocast,
torch.compile(dynamic=True). Same collated batch feeds both (identical graph
layout). Reports params + per-forward latency + speedup at a few batch sizes.
"""
import random
import statistics
import sys
import time

import torch

import hexo_engine.api as eng
from hexgnn.architecture import HexgnnNetwork, precompute_side_rows
from hexgnn.graph_build import batch_from_states
from hexo_models.hexgt.architecture import HexgtNetwork, precompute_attention_layout

DEV = torch.device("cuda")


def make_states(k, seed=7):
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        s = eng.new_game(seed=rng.randint(0, 10**7))
        for _ in range(rng.randint(8, 60)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def to_dev(b):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def bench(fpv, batch, iters=50, warmup=12):
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=torch.float16):
                fpv(batch)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                fpv(batch)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts)


def main():
    torch.manual_seed(0)
    sizes = [int(x) for x in (sys.argv[1:] or ["64", "128", "256"])]

    gnn = HexgnnNetwork().to(DEV).eval()
    m3 = HexgtNetwork(token_dim=168, gnn_layers=4, ctx_layers=3, attention_heads=4, ffn_dim=336,
                      short_term_value_horizons=(4, 12, 24), value_pma_seeds=2,
                      value_head_use_side=True).to(DEV).eval()
    p_gnn = sum(p.numel() for p in gnn.parameters())
    p_m3 = sum(p.numel() for p in m3.parameters())
    gnn_fpv = torch.compile(gnn.forward_policy_value, dynamic=True)
    m3_fpv = torch.compile(m3.forward_policy_value, dynamic=True)

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"params: hexgt model-3 = {p_m3:,}   hexgnn = {p_gnn:,}   "
          f"ratio={p_gnn/p_m3:.3f} ({100*(1-p_gnn/p_m3):.1f}% fewer)\n")
    print(f"{'batch':>6}{'nodes':>8}{'cand':>8}{'m3 ms':>10}{'gnn ms':>10}{'speedup':>9}")
    for n in sizes:
        states = make_states(n, seed=7)
        base = to_dev(batch_from_states(states, n=3))
        ng = int(base["num_graphs"])
        gnn_b = {**base, **precompute_side_rows(base["node_type"], base["node_graph"])}
        m3_b = {**base, **precompute_attention_layout(base["node_graph"], base["node_type"], ng)}
        med_g, min_g = bench(gnn_fpv, gnn_b)
        med_m, min_m = bench(m3_fpv, m3_b)
        ntot = int(base["node_feat"].shape[0]); ctot = int(base["candidate_index"].shape[0])
        print(f"{ng:>6}{ntot:>8}{ctot:>8}{med_m:>10.2f}{med_g:>10.2f}{med_m/med_g:>8.2f}x")


if __name__ == "__main__":
    main()
