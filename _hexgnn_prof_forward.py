"""Forward profiler: where does the GNN forward time go? Profiles the search
forward at production-shaped batches (active=512 leaves) for an OPENING-sized and
a MIDGAME-sized graph set, eager (clear op names) and compiled (real perf). Reports
tensor sizes, CUDA vs CPU/wall (launch-bound test), memcpy, and the top ops."""
import random, statistics, time
import torch
from torch.profiler import profile, ProfilerActivity

import hexo_engine.api as eng
from hexgnn.architecture import HexgnnNetwork, precompute_side_rows
from hexgnn.graph_build import batch_from_states

DEV = torch.device("cuda")
TD, GNN = 128, 3


def states(k, plies, seed):
    rng = random.Random(seed); out = []
    while len(out) < k:
        s = eng.new_game(seed=rng.randint(0, 10**7))
        for _ in range(plies):
            if eng.terminal(s) is not None: break
            acts = list(eng.legal_actions(s))
            if not acts: break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None: out.append(s)
    return out


def mkbatch(k, plies, seed):
    b = batch_from_states(states(k, plies, seed), n=3)
    b = {kk: (v.to(DEV) if torch.is_tensor(v) else v) for kk, v in b.items()}
    b.update(precompute_side_rows(b["node_type"], b["node_graph"]))
    return b


def wall(fn, batch, it=50, wu=12):
    with torch.no_grad():
        for _ in range(wu):
            with torch.autocast("cuda", dtype=torch.float16): fn(batch)
        torch.cuda.synchronize(); ts = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16): fn(batch)
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)


def prof_one(fn, batch, label):
    with torch.no_grad():
        for _ in range(8):
            with torch.autocast("cuda", dtype=torch.float16): fn(batch)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                with torch.autocast("cuda", dtype=torch.float16): fn(batch)
            torch.cuda.synchronize()
    ka = prof.key_averages()
    cuda_total = sum(getattr(k, "self_device_time_total", 0.0) for k in ka) / 1e3 / 20  # ms/iter
    cpu_total = sum(k.self_cpu_time_total for k in ka) / 1e3 / 20
    print(f"\n===== {label} =====")
    print(f"  CUDA self time/iter = {cuda_total:.2f} ms   CPU self time/iter = {cpu_total:.2f} ms")
    print(ka.table(sort_by="self_cuda_time_total", row_limit=14))


def main():
    torch.manual_seed(0)
    m = HexgnnNetwork(token_dim=TD, gnn_layers=GNN).to(DEV).eval()
    print(f"model td{TD}/gnn{GNN} params={sum(p.numel() for p in m.parameters()):,}")
    comp = torch.compile(m.forward_policy_value, dynamic=True)
    for label, plies, seed in [("OPENING ~6 plies", 6, 1), ("MIDGAME ~60 plies", 60, 2)]:
        b = mkbatch(512, plies, seed)
        print(f"\n#### {label}: leaves(num_graphs)={int(b['num_graphs'])} "
              f"nodes={int(b['node_feat'].shape[0])} edges={int(b['edge_index'].shape[1])} "
              f"candidates={int(b['candidate_index'].shape[0])} "
              f"nodes/leaf={b['node_feat'].shape[0]/int(b['num_graphs']):.0f} "
              f"cand/leaf={b['candidate_index'].shape[0]/int(b['num_graphs']):.0f}")
        w_e = wall(m.forward_policy_value, b); w_c = wall(comp, b)
        print(f"  wall/iter: eager={w_e:.2f} ms   compiled={w_c:.2f} ms")
        prof_one(m.forward_policy_value, b, f"{label} EAGER op breakdown")


if __name__ == "__main__":
    main()
