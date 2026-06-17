"""Where does serve wall-clock go in a flush, host vs GPU? Splits the
HexfieldEvaluator serve path into host-pack (from_numpy), pin_memory, H2D copy,
and the forward, across flush mixes from early to deep late game. Identifies
parity-safe host-side overhead (e.g. per-group pin_memory on fresh buffers).

Run (GPU free):
    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_serve_profile.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import numpy as np
import torch
import hexfield.constants as C
from hexfield.inference import HexfieldEvaluator, plan_groups
from hexfield.model import HexfieldNet
from torch.profiler import profile, ProfilerActivity

NBR_SENTINEL = 0xFFFF
rng = np.random.default_rng(0)


def make_payload(sizes, request_ml=True):
    sizes = sorted((int(s) for s in sizes), reverse=True)
    total = sum(sizes)
    feats = rng.standard_normal((total, C.NUM_FEATURES)).astype(np.float16)
    qr = rng.integers(-20, 21, size=(total, 2), dtype=np.int16)
    nbr = np.empty((total, 6), dtype=np.uint16)
    legal_counts = np.empty(len(sizes), dtype=np.int32)
    offsets = [0]; pos = 0
    for i, n in enumerate(sizes):
        row = rng.integers(0, n, size=(n, 6)).astype(np.uint16)
        row[rng.random((n, 6)) < 0.2] = NBR_SENTINEL
        nbr[pos:pos + n] = row
        legal_counts[i] = max(1, min(n, int(rng.integers(1, n + 1))))
        pos += n; offsets.append(pos)
    return {"abi": 1, "shape": (len(sizes), total), "node_feats": feats.tobytes(),
            "node_qr": qr.tobytes(), "node_row_offsets": offsets, "nbr": nbr.tobytes(),
            "legal_counts": legal_counts.tobytes(), "request_moves_left": request_ml}


dev = torch.device("cuda")
model = HexfieldNet().eval()
ev = HexfieldEvaluator(model, device=dev)

# flush mixes: ~150 states (typical late-game flush count), spanning regimes.
REGIMES = {
    "early(20-260)":   list(rng.integers(20, 260, size=150)),
    "mid(256-768)":    list(rng.integers(256, 768, size=150)),
    "late(768-1800)":  list(rng.integers(768, 1800, size=150)),
    "deep(1500-3300)": list(rng.integers(1500, 3300, size=120)),
}


def timed(fn, *a, reps=15):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): r = fn(*a)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000, r


for name, sizes in REGIMES.items():
    p = make_payload(sizes)
    ng = len(plan_groups(sorted(sizes, reverse=True)))
    # full synchronous serve, and the submit (host-enqueue) vs result (D2H) split
    t_full, _ = timed(ev.evaluate_payload, p)
    t_sub, h = timed(ev.submit_payload, p)
    t_res, _ = timed(ev.result, h)
    print(f"\n=== {name}  states={len(sizes)} groups={ng} ===")
    print(f"  evaluate_payload {t_full:6.1f} ms | submit {t_sub:6.1f} ms (host enqueue) | result {t_res:6.1f} ms (D2H)")
    # CPU vs CUDA breakdown on the full path
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(8): ev.evaluate_payload(p)
        torch.cuda.synchronize()
    rows = prof.key_averages()
    def share(substr):
        return sum(e.self_cpu_time_total for e in rows if substr in e.key.lower()) / 8 / 1000
    pin = share("pin_memory") + share("pinned") + share("_pin")
    h2d = sum(e.self_cpu_time_total for e in rows if e.key.lower() in ("aten::to", "aten::copy_", "aten::_to_copy")) / 8 / 1000
    frm = share("from_numpy") + share("frombuffer") + share("ascontiguous") + share("numpy")
    def _cu(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    cuda_tot = sum(_cu(e) for e in rows) / 8 / 1000
    cpu_tot = sum(e.self_cpu_time_total for e in rows) / 8 / 1000
    print(f"  host self-CPU ~{cpu_tot:6.1f} ms  [pin~{pin:.1f} h2d/copy~{h2d:.1f} numpy~{frm:.1f}] | GPU self-CUDA ~{cuda_tot:6.1f} ms")
