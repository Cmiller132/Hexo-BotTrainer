"""Pause-window validation for the serve-forward compile fix + async overlap.

Run in the live torch venv WHILE THE RUN IS STOPPED (hexfield-dev has no torch):

    HEXFIELD_NO_COMPILE= /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_compile_overlap_test.py

Checks, on a CUDA HexfieldNet over ABI-valid synthetic flush payloads:
  1. COMPILE PARITY  — per-bucket compiled serve forward vs eager, fp16 tol.
  2. ASYNC PARITY    — result(submit_payload(p)) byte-identical to the eager
                       evaluate_payload(p) at the Python boundary (the Rust
                       submit/finish split reuses these, so this gates it).
  3. THROUGHPUT      — eager vs compiled forward wall time at production shapes,
                       confirming the compile fallback is actually gone.

Math is unchanged by both fixes; this is a guard, not a tuning sweep. It runs no
self-play and touches no run files.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

import hexfield.constants as C
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

NBR_SENTINEL = 0xFFFF
rng = np.random.default_rng(0)


def make_payload(sizes: list[int], *, request_ml: bool) -> dict:
    """An ABI-valid flush payload (rows sorted DESCENDING by support size, as
    the Rust packer emits). legal_counts in [1, n]; nbr local idx or sentinel."""
    sizes = sorted((int(s) for s in sizes), reverse=True)
    total = sum(sizes)
    feats = rng.standard_normal((total, C.NUM_FEATURES)).astype(np.float16)
    qr = rng.integers(-20, 21, size=(total, 2), dtype=np.int16)
    nbr = np.empty((total, 6), dtype=np.uint16)
    legal_counts = np.empty(len(sizes), dtype=np.int32)
    offsets = [0]
    pos = 0
    for i, n in enumerate(sizes):
        row_nbr = rng.integers(0, n, size=(n, 6)).astype(np.uint16)
        # sprinkle some missing-neighbour sentinels
        mask = rng.random((n, 6)) < 0.2
        row_nbr[mask] = NBR_SENTINEL
        nbr[pos : pos + n] = row_nbr
        legal_counts[i] = max(1, min(n, int(rng.integers(1, n + 1))))
        pos += n
        offsets.append(pos)
    return {
        "abi": 1,
        "shape": (len(sizes), total),
        "node_feats": feats.tobytes(),
        "node_qr": qr.tobytes(),
        "node_row_offsets": offsets,
        "nbr": nbr.tobytes(),
        "legal_counts": legal_counts.tobytes(),
        "request_moves_left": request_ml,
    }


def reply_arrays(reply: dict) -> dict:
    out = {
        "values": np.frombuffer(reply["values_bytes"], dtype=np.float32).copy(),
        "priors": np.frombuffer(reply["priors_bytes"], dtype=np.float32).copy(),
    }
    if "moves_left_bytes" in reply:
        out["moves_left"] = np.frombuffer(reply["moves_left_bytes"], dtype=np.float32).copy()
    return out


def max_abs_diff(a: dict, b: dict) -> dict:
    return {k: float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0 for k in a}


def build_evaluator(model, *, compile_on: bool) -> HexfieldEvaluator:
    prev = os.environ.get("HEXFIELD_NO_COMPILE")
    os.environ["HEXFIELD_NO_COMPILE"] = "0" if compile_on else "1"
    try:
        return HexfieldEvaluator(model, device="cuda")
    finally:
        if prev is None:
            os.environ.pop("HEXFIELD_NO_COMPILE", None)
        else:
            os.environ["HEXFIELD_NO_COMPILE"] = prev


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — this test must run on the GPU box.")
        return 2
    torch.manual_seed(0)
    model = HexfieldNet().eval()

    eager = build_evaluator(model, compile_on=False)
    compiled = build_evaluator(model, compile_on=True)

    # Representative flush mixes: small, medium, the ~144 production mean, and a
    # big skewed batch that spans several pad buckets.
    cases = {
        "single": [40],
        "uniform-64": [60] * 64,
        "prod-mean-144": list(rng.integers(20, 260, size=144)),
        "skewed-200": sorted(rng.integers(8, 300, size=200), reverse=True),
    }

    print("=== COMPILE PARITY (compiled vs eager, fp16 tol) ===")
    TOL = 3e-3
    ok = True
    for name, sizes in cases.items():
        p = make_payload(sizes, request_ml=True)
        ref = reply_arrays(eager.evaluate_payload(p))
        got = reply_arrays(compiled.evaluate_payload(p))
        diffs = max_abs_diff(ref, got)
        bad = any(v > TOL for v in diffs.values())
        ok = ok and not bad
        flag = "FAIL" if bad else "ok"
        print(f"  [{flag}] {name:16s} maxabsdiff={ {k: round(v,5) for k,v in diffs.items()} }")

    print("\n=== ASYNC PARITY (result(submit(p)) vs evaluate_payload(p), exact) ===")
    for name, sizes in cases.items():
        p = make_payload(sizes, request_ml=True)
        a = reply_arrays(eager.evaluate_payload(p))
        b = reply_arrays(eager.result(eager.submit_payload(p)))
        diffs = max_abs_diff(a, b)
        bad = any(v != 0.0 for v in diffs.values())
        ok = ok and not bad
        flag = "FAIL" if bad else "ok"
        print(f"  [{flag}] {name:16s} maxabsdiff={ {k: round(v,7) for k,v in diffs.items()} }")

    print("\n=== THROUGHPUT (eager vs compiled, prod-mean-144, request_ml) ===")
    p = make_payload(list(rng.integers(20, 260, size=144)), request_ml=True)

    def bench(ev: HexfieldEvaluator, reps: int = 30) -> float:
        for _ in range(5):  # warmup (compiles each bucket once)
            ev.evaluate_payload(p)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            ev.evaluate_payload(p)
        torch.cuda.synchronize()
        return (time.time() - t0) / reps * 1000.0

    e_ms = bench(eager)
    c_ms = bench(compiled)
    print(f"  eager    : {e_ms:6.2f} ms/flush")
    print(f"  compiled : {c_ms:6.2f} ms/flush   ({e_ms / c_ms:.2f}x)")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
