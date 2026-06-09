"""Forward-latency sweep over <500k hexgnn configs (GPU, compiled, fp16).

Picks the model: for each (token_dim, gnn_layers) it reports param count and the
compiled fp16 forward latency at a realistic coalesced eval batch. Guards the
<500k budget. The chosen config then gets a real self-play pos/s measurement.
"""
import random
import statistics
import time

import torch

import hexo_engine.api as eng
from hexgnn.architecture import HexgnnNetwork, precompute_side_rows
from hexgnn.graph_build import batch_from_states

DEV = torch.device("cuda")
CONFIGS = [(128, 3), (128, 2), (112, 3), (96, 3), (96, 4)]
BATCHES = [256, 512]


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


def bench(fpv, batch, iters=40, warmup=12):
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
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    batches = {n: to_dev(batch_from_states(make_states(n, seed=7), n=3)) for n in BATCHES}
    for n, b in batches.items():
        b.update(precompute_side_rows(b["node_type"], b["node_graph"]))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"{'config':>14}{'params':>12}{'<500k':>7}" + "".join(f"{('b%d ms' % n):>10}" for n in BATCHES))
    for td, gl in CONFIGS:
        m = HexgnnNetwork(token_dim=td, gnn_layers=gl).to(DEV).eval()
        p = sum(q.numel() for q in m.parameters())
        fpv = torch.compile(m.forward_policy_value, dynamic=True)
        ms = [bench(fpv, batches[n]) for n in BATCHES]
        ok = "yes" if p < 500_000 else "NO"
        print(f"{f'td{td}/gnn{gl}':>14}{p:>12,}{ok:>7}" + "".join(f"{x:>10.2f}" for x in ms), flush=True)
        del m, fpv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
