"""Variant 3: TensorRT (hardest). Run in WSL2 (Linux TRT wheels are reliable).

TRT 11.0 on this box exposes only STRONGLY_TYPED networks (no FP16/BF16 builder
flags). So precision is baked into the ONNX: export the model cast to fp16/bf16,
build a strongly-typed engine that honors those dtypes. Optimization profile
covers bs 1..1024 (opt=128). Inference via the TRT v3 API using torch CUDA
tensors as device buffers (no pycuda). Correctness vs the native FP32 reference.

Requires (WSL venv): onnx, onnxscript, tensorrt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bc

DEVICE = "cuda"
HERE = Path(__file__).resolve().parent
REF_NPZ = HERE / "_ref_outputs.npz"
RESULT_JSON = HERE / "_results_trt.json"
BATCHES = [1, 128, 256, 1024]

TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


class PVTuple(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m.forward_policy_value(x)
        return o["policy"], o["value"]


def onnx_path(precision: str) -> Path:
    return HERE / f"_model1_{precision}.onnx"


def export_onnx(model, precision: str):
    dt = TORCH_DT[precision]
    wrap = PVTuple(model).eval().to(DEVICE).to(dt)
    dummy = torch.zeros(128, 13, 41, 41, device=DEVICE, dtype=dt)
    torch.onnx.export(
        wrap, (dummy,), str(onnx_path(precision)),
        input_names=["input"], output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"[onnx] exported {onnx_path(precision).name} ({precision})", flush=True)


def build_engine(precision: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path(precision), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i), flush=True)
            raise RuntimeError("ONNX parse failed")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 13, 41, 41), (128, 13, 41, 41), (1024, 13, 41, 41))
    cfg.add_optimization_profile(profile)
    print(f"[trt] building {precision} strongly-typed engine (TRT {trt.__version__}) ...", flush=True)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, cfg)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f"TRT engine build failed ({precision})")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    print(f"[trt] built {precision} in {build_s:.1f}s", flush=True)
    return engine, build_s


_TRT_TO_TORCH = None


def trt_runner(engine):
    import tensorrt as trt

    global _TRT_TO_TORCH
    if _TRT_TO_TORCH is None:
        _TRT_TO_TORCH = {
            trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
        }
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = "input"
    out_names = [n for n in names if n != in_name]
    in_dt = _TRT_TO_TORCH[engine.get_tensor_dtype(in_name)]

    def run(x):
        bs = x.shape[0]
        ctx.set_input_shape(in_name, (bs, 13, 41, 41))
        x = x.to(in_dt).contiguous()
        ctx.set_tensor_address(in_name, x.data_ptr())
        outs = {}
        for n in out_names:
            shp = tuple(ctx.get_tensor_shape(n))
            dt = _TRT_TO_TORCH[engine.get_tensor_dtype(n)]
            buf = torch.empty(shp, device=DEVICE, dtype=dt)
            outs[n] = buf
            ctx.set_tensor_address(n, buf.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outs["policy"], outs["value"]

    return run


def bench(fn, batch, label):
    bc.warmup(fn, seconds=5.0, min_iters=40)
    s = bc.time_iters(fn, iters=300)
    s.update({"label": label, "batch": batch,
              "forwards_per_s_mean": bc.fwd_per_s(s["mean_ms"], batch),
              "forwards_per_s_p95": bc.fwd_per_s(s["p95_ms"], batch)})
    print(f"[{label}] bs={batch} mean={s['mean_ms']:.4f}ms p50={s['p50_ms']:.4f} "
          f"p95={s['p95_ms']:.4f} -> {s['forwards_per_s_mean']:.0f} fwd/s", flush=True)
    return s


def main():
    import tensorrt as trt
    print(f"[trt] torch={torch.__version__} tensorrt={trt.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)

    model = bc.build_inference_model(DEVICE, channels_last=False)
    ref = np.load(REF_NPZ)
    arr = ref["inputs"]
    pol_ref = torch.from_numpy(ref["policy"])
    val_ref = torch.from_numpy(ref["value"])
    n_corr = arr.shape[0]

    results = {"tensorrt": trt.__version__, "torch": torch.__version__, "precisions": {}}

    for precision in ("fp16", "bf16"):
        try:
            if not onnx_path(precision).exists():
                export_onnx(model, precision)
            engine, build_s = build_engine(precision)
        except Exception as e:
            print(f"[trt] {precision} skipped: {e}", flush=True)
            results["precisions"][precision] = {"error": str(e)}
            continue
        run = trt_runner(engine)
        entry = {"build_seconds": build_s, "variants": []}

        xc = torch.from_numpy(np.ascontiguousarray(arr[:n_corr])).to(DEVICE)
        p, v = run(xc)
        torch.cuda.synchronize()
        entry["correctness"] = bc.correctness_report(pol_ref, val_ref, p.cpu(), v.cpu())
        print(f"[corr trt {precision}] {json.dumps(entry['correctness'])}", flush=True)

        for bs in BATCHES:
            idx = np.arange(bs) % arr.shape[0]
            x = torch.from_numpy(np.ascontiguousarray(arr[idx])).to(DEVICE)
            entry["variants"].append(bench(lambda: run(x), bs, f"trt_{precision}_bs{bs}"))
        results["precisions"][precision] = entry

    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
