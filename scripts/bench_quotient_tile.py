#!/usr/bin/env python
"""Phase-B tile-efficiency benchmark; GPU execution requires --allow-gpu."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "hexfield_eq" / "python"))


def _median_cuda_ms(torch, function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _cpu_smoke(torch) -> int:
    torch.manual_seed(0)
    a = torch.randn(16, 8)
    b = torch.randn(8, 8)
    dense = a @ b

    batch, nodes, channels = 2, 5, 8
    x = torch.randn(batch, nodes, channels)
    gather = torch.arange(nodes).view(1, nodes, 1).expand(batch, nodes, 7)
    weight = torch.randn(7, channels, channels)
    gathered = x.gather(
        1, gather.reshape(batch, nodes * 7, 1).expand(-1, -1, channels)
    ).reshape(batch, nodes, 7 * channels)
    conv = gathered @ weight.reshape(7 * channels, channels)

    q = torch.randn(batch, 3, nodes, 4)
    attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    assert dense.shape == (16, 8)
    assert conv.shape == (batch, nodes, channels)
    assert attention.shape == q.shape
    assert all(torch.isfinite(value).all() for value in (dense, conv, attention))
    print("# quotient tile benchmark CPU smoke")
    print("| Path | Shape | Result |")
    print("|---|---|---|")
    print("| dense | M=16, K=N=8 | ok |")
    print("| fused-conv plumbing | B=2, N=5, C=8 | ok |")
    print("| attention plumbing | B=2, H=3, S=5, d=4 | ok |")
    return 0


def _gpu_bench(torch, warmup: int, iterations: int, rows: int, batch: int, nodes: int) -> int:
    from hexfield_eq._triton_attn import attn_pair
    from hexfield_eq._triton_conv import hex_conv, hex_conv_ln

    if hex_conv is None or hex_conv_ln is None or attn_pair is None:
        raise RuntimeError("required Triton fused kernels are unavailable")
    if batch * nodes != rows:
        raise ValueError("--batch * --nodes must equal --rows")

    torch.manual_seed(0)
    device = torch.device("cuda")
    widths = (192, 176, 160, 128, 112, 96)
    dense_rows = []
    for width in widths:
        a = torch.randn(rows, width, device=device, dtype=torch.float16)
        b = torch.randn(width, width, device=device, dtype=torch.float16)
        ms = _median_cuda_ms(torch, lambda: a @ b, warmup, iterations)
        flops = 2 * rows * width * width
        traffic = 2 * (rows * width + width * width + rows * width)
        dense_rows.append((width, ms, flops / (ms * 1.0e9), traffic / (ms * 1.0e6)))
    baseline = dense_rows[0][2]

    print("## Dense fp16 GEMM (M ~= 24k)")
    print("| C=K=N | Median ms | TFLOP/s | Effective GB/s | Efficiency vs C=192 |")
    print("|---:|---:|---:|---:|---:|")
    for width, ms, tflops, bandwidth in dense_rows:
        print(f"| {width} | {ms:.4f} | {tflops:.3f} | {bandwidth:.3f} | {tflops / baseline:.3f} |")

    print("\n## Production fused hex-conv kernels")
    print("| C | Kernel | Shape | Median ms | TFLOP/s | Effective GB/s |")
    print("|---:|---|---|---:|---:|---:|")
    for width in (192, 160, 128):
        x = torch.randn(batch, nodes, width, device=device, dtype=torch.float16)
        base = torch.arange(nodes, device=device, dtype=torch.int64)
        gather = base.view(1, nodes, 1).expand(batch, nodes, 7).contiguous()
        mask = torch.ones(batch, nodes, device=device, dtype=torch.bool)
        weight = torch.randn(7, width, width, device=device, dtype=torch.float16)
        bias = torch.randn(width, device=device)
        ln_weight = torch.ones(width, device=device)
        ln_bias = torch.zeros(width, device=device)
        calls = (
            ("hex_conv", lambda: hex_conv(x, gather, mask, weight, bias)),
            (
                "hex_conv_ln",
                lambda: hex_conv_ln(
                    x, gather, mask, weight, bias, ln_weight, ln_bias, 1.0e-5, True
                ),
            ),
        )
        for name, call in calls:
            ms = _median_cuda_ms(torch, call, warmup, iterations)
            flops = 2 * rows * 7 * width * width
            traffic = 2 * (rows * 7 * width + 7 * width * width + rows * width)
            print(
                f"| {width} | {name} | B={batch}, N={nodes} | {ms:.4f} | "
                f"{flops / (ms * 1.0e9):.3f} | {traffic / (ms * 1.0e6):.3f} |"
            )

    print("\n## Production fused relative-position attention")
    print("| W_attn | Heads x d | Shape | Median ms | TFLOP/s | Effective GB/s |")
    print("|---:|---:|---|---:|---:|---:|")
    for width in (192, 96):
        heads, dim = 3, width // 3
        q = torch.randn(batch, heads, nodes, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        pair = torch.zeros(batch, nodes, nodes, device=device, dtype=torch.int16)
        table = torch.zeros(1, heads, device=device, dtype=torch.float16)
        seq = torch.full((batch,), nodes, device=device, dtype=torch.int32)
        ms = _median_cuda_ms(
            torch, lambda: attn_pair(q, k, v, pair, table, seq), warmup, iterations
        )
        flops = 4 * batch * heads * nodes * nodes * dim
        traffic = 2 * (4 * batch * nodes * width) + pair.numel() * pair.element_size()
        print(
            f"| {width} | {heads} x {dim} | B={batch}, S={nodes} | {ms:.4f} | "
            f"{flops / (ms * 1.0e9):.3f} | {traffic / (ms * 1.0e6):.3f} |"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-gpu", action="store_true")
    mode.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rows", type=int, default=24576)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--nodes", type=int, default=512)
    args = parser.parse_args()

    if args.cpu_smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    if args.cpu_smoke:
        return _cpu_smoke(torch)
    if not args.allow_gpu or not torch.cuda.is_available():
        print("refusing GPU benchmark: pass --allow-gpu and provide CUDA", file=sys.stderr)
        return 2
    return _gpu_bench(torch, args.warmup, args.iterations, args.rows, args.batch, args.nodes)


if __name__ == "__main__":
    raise SystemExit(main())
