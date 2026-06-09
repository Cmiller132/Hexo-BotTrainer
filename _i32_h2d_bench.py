"""Measure the H2D (_to_device) transfer time for the hexgnn eval payload BEFORE
(int64 transport) vs AFTER (int32 transport) the narrowing, on a REAL large payload.

Builds the int32 wire payload (`hexgnn_featurize_states`) and an equivalent int64
host batch. Times `infr._to_device(batch)` (the H2D copy stage) for each, min-over-
iters to defeat co-tenant GPU contention (the live run shares the GPU). Reports the
per-forward H2D ms before/after and the index-byte volume moved.
"""
from __future__ import annotations

import importlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for pkg in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    p = ROOT / "packages" / pkg / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
_HEXGNN = ROOT / "packages" / "hexgnn" / "python"
if str(_HEXGNN) not in sys.path:
    sys.path.insert(0, str(_HEXGNN))

from hexgnn import rust_bridge
from hexgnn.architecture import HexgnnNetwork
from hexgnn.inference import HexgnnInference

NARROW_KEYS = ("node_type", "node_graph", "edge_index", "edge_type",
               "edge_dir", "candidate_index", "candidate_graph")


def _states(k, seed=11):
    eng = importlib.import_module("hexo_engine.api")
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        s = eng.new_game(seed=rng.randint(0, 10**7))
        for _ in range(rng.randint(0, 90)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def _host_batch(payload, dtype):
    """Build a HOST batch dict with the index buffers in `dtype` (int32 or int64),
    floats as views. Mirrors the buffers evaluate_featurized_batch builds before
    _to_device (the int32 case is the shipping wire format)."""
    tn = int(payload["total_nodes"]); te = int(payload["total_edges"])
    fd = int(payload["feat_dim"]); ad = int(payload["attr_dim"])
    npdt = np.int32 if dtype == torch.int32 else np.int64
    def idx(k):
        a = np.frombuffer(payload[k], dtype=np.int32)
        return torch.from_numpy(a.astype(npdt, copy=False) if npdt is np.int32 else a.astype(np.int64))
    b = {
        "node_feat": torch.frombuffer(payload["node_feat"], dtype=torch.float32).reshape(tn, fd),
        "node_type": idx("node_type"),
        "node_graph": idx("node_graph"),
        "candidate_index": idx("candidate_index"),
        "candidate_graph": idx("candidate_graph"),
        "num_graphs": int(payload["num_graphs"]),
    }
    b["edge_index"] = idx("edge_index").reshape(2, te)
    b["edge_type"] = idx("edge_type")
    b["edge_attr"] = torch.frombuffer(payload["edge_attr"], dtype=torch.float32).reshape(te, ad)
    b["edge_dir"] = idx("edge_dir")
    return b


def _bench_h2d(infr, batch, iters=80, warmup=20):
    for _ in range(warmup):
        infr._to_device(batch); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        infr._to_device(batch); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0] * 1e3, ts[len(ts) // 2] * 1e3


def _index_bytes(payload, elem):
    te = int(payload["total_edges"]); tn = int(payload["total_nodes"]); tc = int(payload["total_candidates"])
    # node_type+node_graph (tn each), edge_index(2*te), edge_type+edge_dir(te each),
    # candidate_index+candidate_graph(tc each)
    n = 2 * tn + 2 * te + 2 * te + 2 * tc
    return n * elem


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    states = _states(96, seed=11)
    payload = _capture = rust_bridge._hexgnn_module().hexgnn_featurize_states(tuple(states), 3)
    ng = int(payload["num_graphs"]); tn = int(payload["total_nodes"])
    te = int(payload["total_edges"]); tc = int(payload["total_candidates"])
    print(f"payload: graphs={ng} nodes={tn} edges={te} cands={tc}")
    print(f"index bytes int64={_index_bytes(payload,8)/1e6:.2f} MB  int32={_index_bytes(payload,4)/1e6:.2f} MB")

    model = HexgnnNetwork(token_dim=96, gnn_layers=2, steerable_channels=4).to(device).eval()
    infr = HexgnnInference(model, device=device, fp16=True)

    b64 = _host_batch(payload, torch.int64)
    b32 = _host_batch(payload, torch.int32)
    # pin caches differ by size; bench each independently, min-over-iters.
    m64, med64 = _bench_h2d(infr, b64)
    m32, med32 = _bench_h2d(infr, b32)
    print(f"H2D int64 (BEFORE): min {m64:.3f} ms  median {med64:.3f} ms")
    print(f"H2D int32 (AFTER) : min {m32:.3f} ms  median {med32:.3f} ms")
    print(f">>> H2D reduction: min {m64-m32:.3f} ms ({100*(m64-m32)/m64:.1f}%)  "
          f"median {med64-med32:.3f} ms ({100*(med64-med32)/med64:.1f}%)")


if __name__ == "__main__":
    main()
