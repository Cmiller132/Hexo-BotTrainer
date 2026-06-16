"""torch.compile variant for the inference-backend cycle — RUN IN WSL2.

Builds on the shared `bench_harness` (same model load, folded inference clone,
representative inputs, timer, and FP32-correctness comparator), adding a
`torch.compile` forward variant. Per the user, torch.compile is exercised under
WSL2 (torch 2.11 + Triton); native-Windows torch 2.10 lacks a working Inductor/
Triton path.

Design notes:
  - The Variant.forward(model, inputs) contract is preserved, so the harness's
    `time_variant` and `compare_to_reference` work unchanged.
  - torch.compile is applied to a thin wrapper exposing `forward_policy_value`
    (the inference head path search uses). `mode="max-autotune"` enables Triton
    autotuning + CUDA graphs (the latter is what slashes single-eval latency).
  - `dynamic=False` so each bucket shape (1 / 128 / 256 / 1024) compiles to a
    static-shape graph — matching the evaluator's power-of-two bucket padding.
    The compiled callable is cached once; the FIRST call at a new shape pays the
    compile cost (reported separately by the runner), later calls are hot.

Smoke (LIGHT — proves it runs + rough parity; full timing is the verification
agent's job on a quiet GPU):
    wsl ... python -m analysis.inference_backends.compile_variant --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Mapping

import torch
from torch import nn

from .bench_harness import (
    Variant,
    build_inference_model,
    compare_to_reference,
    fp16_amp,
    fp32_reference,
    load_model,
    make_inputs,
    time_variant,
)


class _PolicyValueWrapper(nn.Module):
    """Expose forward_policy_value as forward so torch.compile traces that path."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        if hasattr(self.model, "forward_policy_value"):
            return self.model.forward_policy_value(x)
        return self.model(x)


# One compiled callable per process (keyed by the wrapped model id). torch.compile
# itself caches per input shape, so a single compiled object handles all buckets.
_COMPILED: dict[int, nn.Module] = {}


def _compiled_for(model: nn.Module) -> nn.Module:
    key = id(model)
    if key not in _COMPILED:
        wrapper = _PolicyValueWrapper(model).eval()
        _COMPILED[key] = torch.compile(wrapper, mode="max-autotune", dynamic=False)
    return _COMPILED[key]


def _compile_forward(model: nn.Module, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    compiled = _compiled_for(model)
    with torch.inference_mode():
        with torch.autocast(device_type=inputs.device.type, dtype=torch.float16):
            return compiled(inputs)


compile_fp16 = Variant(
    name="compile",
    forward=_compile_forward,
    description="torch.compile(max-autotune) + autocast float16 (WSL: torch 2.11 + Triton)",
)


def run_smoke(device: str = "cuda") -> None:
    print("=== torch.compile SMOKE (WSL; light, not a benchmark) ===", flush=True)
    print(f"[env] torch={torch.__version__} cuda_build={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    model, _ = load_model(device=device)
    inf_model = build_inference_model(model, device=device)

    # Small bucket-representative batches only (proving it runs).
    for batch in (1, 128):
        inputs = make_inputs(batch, device=device)
        # First call compiles for this shape — time it so the cost is visible.
        t0 = time.perf_counter()
        with torch.inference_mode():
            _compile_forward(inf_model, inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        parity = compare_to_reference(compile_fp16, inf_model, inputs)
        print(f"[smoke] bs={batch}: first-call(compile) {compile_s:.1f}s; {parity.render()}", flush=True)
        # A few hot iters just to confirm wiring (NOT a benchmark).
        timing = time_variant(compile_fp16, inf_model, inputs, warmup_iters=3, iters=8)
        print(f"        {timing.render()}", flush=True)

    print("[smoke] WSL-vs-native baseline plumbing: eager fp16 vs compile fp16 on this box.", flush=True)
    inputs = make_inputs(128, device=device)
    eager = time_variant(fp16_amp, inf_model, inputs, warmup_iters=3, iters=8)
    comp = time_variant(compile_fp16, inf_model, inputs, warmup_iters=3, iters=8)
    print(f"        eager : {eager.render()}", flush=True)
    print(f"        compile:{comp.render()}", flush=True)
    print("=== SMOKE DONE (full clean timing is the verification agent's job) ===", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", default=True)
    args = ap.parse_args(argv)
    run_smoke(args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
