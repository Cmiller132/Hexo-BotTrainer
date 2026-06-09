"""CPU forward micro-benchmark: hexgnn (transformer-free) vs hexgt model-3.

Same batch of real packed graphs, same candidate radius, eval mode, fp32 CPU.
Times the SEARCH forward (`forward_policy_value`, the self-play hot path — hexgt
notes put the NN forward at ~78% of self-play wall) and reports params + speedup.
The graph layout is identical between the two lineages, so the SAME collated batch
feeds both (each gets its own eager precompute keys attached).
"""
import random
import statistics
import time

import torch

import hexo_engine.api as eng
from hexgnn.architecture import HexgnnNetwork
from hexgnn.architecture import precompute_side_rows
from hexgnn.graph_build import batch_from_states

from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.architecture import precompute_attention_layout


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


def bench(model, batch, iters=30, warmup=5):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_policy_value(batch)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model.forward_policy_value(batch)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    N, RADIUS = 64, 3
    states = make_states(N, seed=7)
    base = batch_from_states(states, n=RADIUS)
    ng = int(base["num_graphs"])
    ctot = int(base["candidate_index"].shape[0])
    ntot = int(base["node_feat"].shape[0])
    print(f"batch: {ng} states, {ntot} nodes, {ctot} candidates, radius={RADIUS}, "
          f"CPU threads={torch.get_num_threads()}")

    # hexgnn (defaults: token_dim 168, gnn_layers 4, PMA k=2, +SIDE; NO transformer/STV)
    gnn = HexgnnNetwork()
    gnn_batch = {**base, **precompute_side_rows(base["node_type"], base["node_graph"])}

    # hexgt model-3 (token_dim 168, 4 GNN + 3 ctx transformer, STV[4,12,24], PMA k=2)
    m3 = HexgtNetwork(
        token_dim=168, gnn_layers=4, ctx_layers=3, attention_heads=4, ffn_dim=336,
        short_term_value_horizons=(4, 12, 24), value_pma_seeds=2, value_head_use_side=True,
    )
    m3_batch = {**base, **precompute_attention_layout(base["node_graph"], base["node_type"], ng)}

    p_gnn = sum(p.numel() for p in gnn.parameters())
    p_m3 = sum(p.numel() for p in m3.parameters())

    med_gnn, min_gnn = bench(gnn, gnn_batch)
    med_m3, min_m3 = bench(m3, m3_batch)

    print()
    print(f"{'model':<16}{'params':>14}{'median ms':>14}{'min ms':>12}")
    print(f"{'hexgt model-3':<16}{p_m3:>14,}{med_m3:>14.2f}{min_m3:>12.2f}")
    print(f"{'hexgnn':<16}{p_gnn:>14,}{med_gnn:>14.2f}{min_gnn:>12.2f}")
    print()
    print(f"param ratio   hexgnn / model-3 = {p_gnn / p_m3:.3f}  "
          f"({100*(1 - p_gnn/p_m3):.1f}% fewer params)")
    print(f"forward speedup (median) model-3 / hexgnn = {med_m3 / med_gnn:.2f}x")
    print(f"forward speedup (min)    model-3 / hexgnn = {min_m3 / min_gnn:.2f}x")
    print()
    print("Self-play projection: hexgt notes put the NN forward at ~78% of self-play")
    print("wall. With forward sped up S=median ratio above, Amdahl on the 78% forward /")
    print("22% non-forward share gives an expected self-play speedup of:")
    s = med_m3 / med_gnn
    proj = 1.0 / (0.22 + 0.78 / s)
    print(f"   1 / (0.22 + 0.78/{s:.2f}) = {proj:.2f}x  (forward-only; TSS scans etc. unchanged)")


if __name__ == "__main__":
    main()
