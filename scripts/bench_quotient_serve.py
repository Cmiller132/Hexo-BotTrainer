#!/usr/bin/env python
"""Phase-B full-network serve benchmark; GPU execution requires --allow-gpu."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EQ_PATH = ROOT / "packages" / "hexfield_eq" / "python"
sys.path.insert(0, str(EQ_PATH))

ARMS = (
    ("baseline", "reg:16", 16, 192, 1.0),
    ("B1", "reg:8,mirror:8,axis:4,triv:4", 8, 160, 1.615),
    ("B2", "reg:4,mirror:6,point:2,axis:8,triv:8", 16, 128, 1.466),
    ("B3", "reg:4,mirror:6,point:2,axis:8,triv:8", 8, 128, 2.068),
)


def _tiny_inputs(torch):
    batch, nodes, features = 1, 7, 25
    feats = torch.randn(batch, nodes, features)
    nbr = torch.full((batch, nodes, 6), nodes, dtype=torch.long)
    mask = torch.ones(batch, nodes, dtype=torch.bool)
    coords = torch.zeros(batch, nodes, 2, dtype=torch.long)
    raylen = torch.ones(batch, nodes, 12, dtype=torch.uint8)
    return feats, nbr, mask, coords, raylen


def _cpu_smoke(torch) -> int:
    sys.modules["torch.nn.attention.flex_attention"] = None
    from hexfield_eq.model import HexfieldNet

    torch.manual_seed(0)
    inputs = _tiny_inputs(torch)
    print("| Arm | C | K_attn | Tiny CPU eager |")
    print("|---|---:|---:|---|")
    with torch.no_grad():
        for name, signature, orbit, channels, _projection in ARMS:
            model = HexfieldNet(
                channels=channels,
                type_sig=signature,
                attn_orbit=orbit,
                trunk_layout="CLA",
                reg_lane=False,
            ).eval()
            output = model.forward_policy_value(*inputs)
            assert all(torch.isfinite(value).all() for value in output.values())
            print(f"| {name} | {channels} | {orbit} | ok |")
    return 0


def _child_env(signature: str, orbit: int, channels: int) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("HEXFIELD")}
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(EQ_PATH), env.get("PYTHONPATH", ""))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HEXFIELD_EQ_FEATURE_VERSION": "1",
            "HEXFIELD_EQ_GROUP_ORDER": "12",
            "HEXFIELD_EQ_TYPE_SIG": signature,
            "HEXFIELD_EQ_ATTN_ORBIT": str(orbit),
            "HEXFIELD_EQ_CHANNELS": str(channels),
            "HEXFIELD_EQ_ATTENTION_HEADS": "3",
            "HEXFIELD_EQ_TRUNK": "CCLACCLACLA",
            "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
            "HEXFIELD_EQ_REG_LANE": "1",
            "HEXFIELD_EQ_REG_TOK_READ": "0",
            "HEXFIELD_EQ_RAY_BLOCKERS": "1",
            "HEXFIELD_TRITON_CONV": "1",
            "HEXFIELD_TRITON_CONV_LN": "1",
            "HEXFIELD_TRITON_ATTN": "1",
            "HEXFIELD_EQ_TRITON_RAY": "1",
            "HEXFIELD_SERVE_FLEX": "1",
            "HEXFIELD_FLEX_PAIR": "1",
            "HEXFIELD_SERVE_HALF": "1",
            "HEXFIELD_RUST_PACK": "1",
            "HEXFIELD_COPY_STREAM": "1",
            "HEXFIELD_CUDA_GRAPHS": "1",
            "HEXFIELD_PERF_TRACE": "1",
            "HEXFIELD_NO_COMPILE": "1",
        }
    )
    return env


def _synthetic_payload(torch, batch: int, nodes: int) -> dict:
    import numpy as np
    from hexfield_eq.constants import NUM_FEATURES, RAYLEN_SLOTS

    total = batch * nodes
    feats = torch.linspace(-0.5, 0.5, total * NUM_FEATURES, dtype=torch.float32)
    feats = feats.reshape(total, NUM_FEATURES).half().numpy()
    coords = np.zeros((total, 2), dtype=np.int16)
    local = np.arange(nodes, dtype=np.uint16)
    nbr = np.repeat(local[:, None], 6, axis=1)
    nbr = np.tile(nbr[None, :, :], (batch, 1, 1)).reshape(total, 6)
    raylen = np.ones((total, RAYLEN_SLOTS), dtype=np.uint8)
    return {
        "abi": 1,
        "shape": (batch, total),
        "node_feats": feats.tobytes(),
        "node_qr": coords.tobytes(),
        "nbr": nbr.tobytes(),
        "raylen": raylen.tobytes(),
        "node_row_offsets": [index * nodes for index in range(batch + 1)],
        "legal_counts": np.full(batch, nodes, dtype=np.int32).tobytes(),
        "request_moves_left": False,
    }


def _gpu_child(name: str, batch: int, nodes: int, warmup: int, iterations: int) -> int:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA child launched without CUDA")
    from hexfield_eq.inference import HexfieldEvaluator
    from hexfield_eq.model import HexfieldNet

    torch.manual_seed(0)
    evaluator = HexfieldEvaluator(HexfieldNet().eval(), device="cuda")
    if not evaluator._serve_half or not evaluator._use_graphs:
        raise RuntimeError("serve-half and CUDA graphs must both be active")
    payload = _synthetic_payload(torch, batch, nodes)
    for _ in range(warmup):
        evaluator.evaluate_payload(dict(payload))
    torch.cuda.synchronize()
    evaluator._perf = type(evaluator._perf)()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        evaluator.evaluate_payload(dict(payload))
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    if evaluator._graph_cache is None or not evaluator._graph_cache._graphs:
        raise RuntimeError("serve benchmark did not capture a CUDA graph")
    roundtrip_ms = statistics.median(samples)
    trace = evaluator.perf_trace_report()
    if trace is None or trace.get("measured_flushes") != iterations:
        raise RuntimeError(f"incomplete CUDA-event forward trace: {trace}")
    print(
        json.dumps(
            {
                "arm": name,
                "fwd_ms": trace["median_forward_ms"],
                "roundtrip_ms": roundtrip_ms,
                "pos_s": batch * 1000.0 / roundtrip_ms,
            }
        )
    )
    return 0


def _gpu_controller(torch, args) -> int:
    results = []
    for name, signature, orbit, channels, projection in ARMS:
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--allow-gpu",
                "--_child",
                name,
                "--batch",
                str(args.batch),
                "--nodes",
                str(args.nodes),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
            ],
            cwd=ROOT,
            env=_child_env(signature, orbit, channels),
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{name} serve child failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        result.update({"channels": channels, "orbit": orbit, "projection": projection})
        results.append(result)

    baseline = results[0]["pos_s"]
    print("| Arm | C | K_attn | Forward ms | Positions/s | Speedup | G8 alpha=4/7 | +/-20% verdict |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        speedup = result["pos_s"] / baseline
        projection = result["projection"]
        within = True if result["arm"] == "baseline" else abs(speedup / projection - 1.0) <= 0.20
        verdict = "baseline" if result["arm"] == "baseline" else ("PASS" if within else "OUTSIDE")
        print(
            f"| {result['arm']} | {result['channels']} | {result['orbit']} | "
            f"{result['fwd_ms']:.3f} | {result['pos_s']:.2f} | {speedup:.3f}x | "
            f"{projection:.3f}x | {verdict} |"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-gpu", action="store_true")
    mode.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--_child", metavar="ARM", help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.batch * args.nodes < 23000 or args.batch * args.nodes > 26000:
        if not args.cpu_smoke:
            parser.error("GPU deploy shape must keep batch*nodes approximately 24k")

    if args.cpu_smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    if args.cpu_smoke:
        return _cpu_smoke(torch)
    if not args.allow_gpu or not torch.cuda.is_available():
        print("refusing GPU benchmark: pass --allow-gpu and provide CUDA", file=sys.stderr)
        return 2
    if args._child:
        return _gpu_child(args._child, args.batch, args.nodes, args.warmup, args.iterations)
    return _gpu_controller(torch, args)


if __name__ == "__main__":
    raise SystemExit(main())
