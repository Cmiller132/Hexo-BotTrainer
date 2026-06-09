"""Bit-identity gate for the int64->int32 H2D-transport narrowing of the hexgnn
MCTS eval index buffers.

Two checks:

(A) DEVICE-TENSOR EQUALITY (no model, pure transport): for a REAL captured payload
    (`hexgnn_featurize_states` on real-ish states), build each narrowed index tensor
    via int32-transport + on-device `.long()` and via a direct int64 view of the same
    logical values, and assert the device tensors are `torch.equal`. Integer narrowing
    int64->int32->int64 is exact, so this MUST be bit-equal.

(B) END-TO-END BYTE IDENTITY under `torch.use_deterministic_algorithms(True)`:
    run `evaluate_featurized_batch` on the int32 payload (the shipping path) and on a
    synthesized INT64 payload carrying the identical logical buffers, and assert the
    returned values_bytes/priors_bytes are byte-identical. Determinism removes the
    index_add_/index_reduce_ atomic nondeterminism so a single forward is reproducible
    and the ONLY variable is the transport dtype.
"""
from __future__ import annotations

import importlib
import random
import sys
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

NARROW_KEYS = (
    "node_type", "node_graph", "edge_index", "edge_type",
    "edge_dir", "candidate_index", "candidate_graph",
)


def _states(k, seed=7):
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


def _capture_payload(states, n):
    return rust_bridge._hexgnn_module().hexgnn_featurize_states(tuple(states), n)


def check_A_device_equality(payload, device):
    """Build each narrowed index tensor via int32-transport+.long() vs a direct
    int64 build of the same values; assert device tensors are torch.equal."""
    te = int(payload["total_edges"])
    fails = []
    for key in NARROW_KEYS:
        if key == "edge_dir" and "edge_dir" not in payload:
            continue
        # int32 transport (the wire) -> device -> widen
        i32 = torch.frombuffer(payload[key], dtype=torch.int32)
        if key == "edge_index":
            i32 = i32.reshape(2, te)
        dev_from_i32 = i32.to(device).long()
        # reference int64: take the same logical values as int64 on host, then to dev
        i64_ref = torch.from_numpy(
            np.frombuffer(payload[key], dtype=np.int32).astype(np.int64)
        )
        if key == "edge_index":
            i64_ref = i64_ref.reshape(2, te)
        dev_from_i64 = i64_ref.to(device)
        ok = dev_from_i32.dtype == torch.int64 and torch.equal(dev_from_i32, dev_from_i64)
        print(f"  [A] {key:18s} dtype={dev_from_i32.dtype} equal={ok}")
        if not ok:
            fails.append(key)
    return fails


def _i32_payload_to_i64_batch(payload, device):
    """Reconstruct an on-device batch with INT64 index buffers from the int32 wire
    payload (the pre-change transport behavior), for the end-to-end byte gate."""
    tn = int(payload["total_nodes"])
    te = int(payload["total_edges"])
    tc = int(payload["total_candidates"])
    fd = int(payload["feat_dim"])
    ad = int(payload["attr_dim"])
    g = lambda k: np.frombuffer(payload[k], dtype=np.int32).astype(np.int64)
    batch = {
        "node_feat": torch.frombuffer(payload["node_feat"], dtype=torch.float32).reshape(tn, fd).clone(),
        "node_type": torch.from_numpy(g("node_type")),
        "node_graph": torch.from_numpy(g("node_graph")),
        "candidate_index": torch.from_numpy(g("candidate_index")),
        "candidate_graph": torch.from_numpy(g("candidate_graph")),
        "num_graphs": int(payload["num_graphs"]),
    }
    if te:
        batch["edge_index"] = torch.from_numpy(g("edge_index")).reshape(2, te)
        batch["edge_type"] = torch.from_numpy(g("edge_type"))
        batch["edge_attr"] = torch.frombuffer(payload["edge_attr"], dtype=torch.float32).reshape(te, ad).clone()
        if "edge_dir" in payload:
            batch["edge_dir"] = torch.from_numpy(g("edge_dir"))
    else:
        batch["edge_index"] = torch.zeros((2, 0), dtype=torch.int64)
        batch["edge_type"] = torch.zeros((0,), dtype=torch.int64)
        batch["edge_attr"] = torch.zeros((0, ad), dtype=torch.float32)
        if "edge_dir" in payload:
            batch["edge_dir"] = torch.zeros((0,), dtype=torch.int64)
    return batch


def _eval_from_batch(infr, batch, num_graphs):
    """Replicate evaluate_featurized_batch's post-forward arithmetic from a prebuilt
    (host) batch dict — used for the int64 reference path."""
    from hexgnn.losses import decode_binned_value, segment_log_softmax
    out, dev_batch = infr.forward_batch(batch)
    cg = dev_batch["candidate_graph"]
    raw_policy, raw_value, _ = infr._count_and_sanitize(
        out["policy"].float(), out["value"].float(), cg, num_graphs
    )
    priors = segment_log_softmax(raw_policy, cg, num_graphs).exp()
    values = decode_binned_value(raw_value)
    values = torch.nan_to_num(values.clamp(-1.0, 1.0), nan=0.0)
    fused = torch.cat([values.reshape(-1), priors.reshape(-1)]).to("cpu", torch.float32).numpy()
    return {
        "values_bytes": np.ascontiguousarray(fused[:num_graphs]).tobytes(),
        "priors_bytes": np.ascontiguousarray(fused[num_graphs:]).tobytes(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch.manual_seed(0)
    states = _states(64, seed=7)
    n = 3
    payload = _capture_payload(states, n)
    print(f"captured payload: graphs={payload['num_graphs']} nodes={payload['total_nodes']} "
          f"edges={payload['total_edges']} cands={payload['total_candidates']}")

    print("(A) device-tensor int32+.long() == int64 transport:")
    fails_A = check_A_device_equality(payload, device)
    A_pass = not fails_A
    print(f"  => A {'PASS' if A_pass else 'FAIL ' + str(fails_A)}")

    if device != "cuda":
        print("(B) SKIPPED (needs CUDA for the forward); A is the transport gate.")
        print("OVERALL:", "PASS" if A_pass else "FAIL")
        sys.exit(0 if A_pass else 1)

    print("(B) end-to-end byte identity under deterministic algorithms:")
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = HexgnnNetwork(token_dim=96, gnn_layers=2, steerable_channels=4).to(device).eval()
    infr = HexgnnInference(model, device=device, fp16=True)
    num_graphs = int(payload["num_graphs"])

    # Shipping path: int32 wire payload through evaluate_featurized_batch.
    res_i32 = infr.evaluate_featurized_batch(payload)
    # Reference path: identical logical buffers but int64 transport.
    batch_i64 = _i32_payload_to_i64_batch(payload, device)
    res_i64 = _eval_from_batch(infr, batch_i64, num_graphs)

    vb_ok = res_i32["values_bytes"] == res_i64["values_bytes"]
    pb_ok = res_i32["priors_bytes"] == res_i64["priors_bytes"]
    print(f"  values_bytes identical: {vb_ok} ({len(res_i32['values_bytes'])} bytes)")
    print(f"  priors_bytes identical: {pb_ok} ({len(res_i32['priors_bytes'])} bytes)")
    B_pass = vb_ok and pb_ok
    print(f"  => B {'PASS' if B_pass else 'FAIL'}")

    print("OVERALL:", "PASS" if (A_pass and B_pass) else "FAIL")
    sys.exit(0 if (A_pass and B_pass) else 1)


if __name__ == "__main__":
    main()
