"""Forward-only A/B: is the hexgnn forward launch-bound (CUDA graphs help) or
bandwidth/compute-bound (they don't) at the REAL per-forward batch size?

Synthesizes a fixed-shape collated batch at opening density for G leaves, then
times model.forward_policy_value under eager / dynamic-compile (production) /
reduce-overhead (cudagraphs). min-over-iters defeats co-tenant GPU contention.
argv: G nodes_per_leaf edges_per_leaf cand_per_leaf
"""
import sys, time, torch
from hexgnn.architecture import HexgnnNetwork
from hexgnn.constants import NODE_FEATURE_DIM, EDGE_ATTR_DIM, NUM_EDGE_TYPES, NUM_NODE_TYPES

G   = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
NPL = int(sys.argv[2]) if len(sys.argv) > 2 else 222
EPL = int(sys.argv[3]) if len(sys.argv) > 3 else 1499
CPL = int(sys.argv[4]) if len(sys.argv) > 4 else 215
dev = "cuda"
torch.manual_seed(0)

def make_batch(G, NPL, EPL, CPL):
    N, E, C = G*NPL, G*EPL, G*CPL
    # graph-contiguous (sorted) node/candidate ids, matching the Rust collate.
    node_graph = torch.arange(G, device=dev).repeat_interleave(NPL)
    cand_graph = torch.arange(G, device=dev).repeat_interleave(CPL)
    # candidate rows index nodes within their own graph (offset by graph start).
    cand_in_graph = torch.randint(0, NPL, (C,), device=dev)
    candidate_index = (node_graph.new_tensor([], dtype=torch.long),)  # placeholder
    cand_index = (cand_graph * NPL + cand_in_graph).clamp_max(N-1)
    # edges: both endpoints within the same graph.
    e_graph = torch.arange(G, device=dev).repeat_interleave(EPL)
    src_in = torch.randint(0, NPL, (E,), device=dev)
    dst_in = torch.randint(0, NPL, (E,), device=dev)
    edge_index = torch.stack([e_graph*NPL + src_in, e_graph*NPL + dst_in]).clamp_max(N-1)
    edge_type  = torch.randint(0, NUM_EDGE_TYPES, (E,), device=dev)
    edge_dir   = torch.randint(-1, 6, (E,), device=dev)
    return {
        "node_feat": torch.randn(N, NODE_FEATURE_DIM, device=dev, dtype=torch.float16),
        "node_type": torch.randint(0, NUM_NODE_TYPES, (N,), device=dev),
        "node_graph": node_graph,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_attr": torch.randn(E, EDGE_ATTR_DIM, device=dev, dtype=torch.float16),
        "edge_dir": edge_dir,
        "candidate_index": cand_index,
        "candidate_graph": cand_graph,
        "num_graphs": G,
    }

def bench(fn, batch, iters=60, warmup=12):
    for _ in range(warmup):
        fn(batch); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(batch); torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ts.sort()
    return ts[0]*1e3, ts[len(ts)//2]*1e3   # min, median ms

def run(G, NPL, EPL, CPL):
    print(f"\n=== G={G} nodes/leaf={NPL} edges/leaf={EPL} cand/leaf={CPL} "
          f"(N={G*NPL} E={G*EPL} C={G*CPL}) fp16 ===", flush=True)
    batch = make_batch(G, NPL, EPL, CPL)
    m = HexgnnNetwork(token_dim=96, gnn_layers=2, steerable_channels=4).to(dev).half().eval()
    with torch.no_grad():
        eager = lambda b: m.forward_policy_value(b)
        e_min, e_med = bench(eager, batch)
        print(f"eager            : min {e_min:7.2f} ms  median {e_med:7.2f} ms", flush=True)

        dyn = torch.compile(m.forward_policy_value, dynamic=True)
        d_min, d_med = bench(dyn, batch)
        print(f"compile dynamic  : min {d_min:7.2f} ms  median {d_med:7.2f} ms  (PRODUCTION)", flush=True)

        try:
            ro = torch.compile(m.forward_policy_value, mode="reduce-overhead")
            r_min, r_med = bench(ro, batch)
            print(f"reduce-overhead  : min {r_min:7.2f} ms  median {r_med:7.2f} ms  (CUDA graphs)", flush=True)
            print(f">>> cudagraph speedup vs production dynamic: min {d_min/r_min:.2f}x  median {d_med/r_med:.2f}x", flush=True)
        except Exception as ex:
            print(f"reduce-overhead FAILED: {ex}", flush=True)

for (g, npl, epl, cpl) in [(G, NPL, EPL, CPL)]:
    try:
        run(g, npl, epl, cpl)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"OOM at G={g}; retry G={g//2}", flush=True)
        run(g//2, npl, epl, cpl)
