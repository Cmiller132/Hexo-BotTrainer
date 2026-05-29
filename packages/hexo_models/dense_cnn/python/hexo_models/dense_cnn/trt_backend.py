"""TensorRT FP16 backend for the dense_cnn inference forward (optional, WSL/Linux).

This provides a drop-in replacement for the PyTorch forward used by
`DenseCNNInference`: a callable `forward_policy_value(x) -> {"policy","value"}`
backed by a prebuilt TensorRT engine, plus a builder that compiles the engine
from the CURRENT model weights and gates adoption on a correctness check vs the
torch FP16 reference.

Design (matches the microbench that measured ~2.4-2.7x at bs128/256):
  - TRT 10/11 dropped FP16/BF16 builder flags, so we use STRONGLY_TYPED networks
    with FP16 baked into the ONNX (model exported in half).
  - One dynamic-batch engine (opt profile 1..max_batch) handles every bucket.
  - Inference via the v3 API with torch CUDA tensors as device buffers (no
    pycuda); a dedicated CUDA stream avoids the default-stream sync penalty.
  - Built from the optimized (folded) inference model so it matches production.

Everything here imports tensorrt/onnx lazily; absence => caller falls back to
torch. Nothing in this module runs at import time.
"""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import nn


class _PVTuple(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m.forward_policy_value(x)
        return o["policy"], o["value"]


def _representative_inputs(n: int) -> "torch.Tensor | None":
    """Real encoded game positions for the correctness gate. Random board inputs
    give a diffuse policy (near-tied logits) so fp16 flips the argmax spuriously;
    real mid-game positions have the sharp P7 policy that production sees, making
    the argmax-match gate meaningful. Returns None if the engine path is
    unavailable (caller then uses a synthetic fallback)."""
    try:
        import random
        import hexo_engine as engine
        from hexo_engine.types import unpack_coord_id
        from . import rust_bridge
        import numpy as np

        rng = random.Random(12345)
        states = []
        gi = 0
        while len(states) < n:
            st = engine.new_game(seed=500_000 + gi); gi += 1
            for _ in range(rng.randint(6, 120)):
                if engine.terminal(st) is not None:
                    break
                aids = engine.legal_action_ids(st)
                if not aids:
                    break
                engine.apply_action(st, engine.PlacementAction(unpack_coord_id(rng.choice(aids))))
            if engine.terminal(st) is None:
                states.append(engine.clone_state(st))
        payload = rust_bridge.model1_batch_inputs(states[:n])
        shape = tuple(int(x) for x in payload["shape"])
        arr = np.frombuffer(bytes(payload["inputs"]), dtype=np.float32).reshape(shape).copy()
        return torch.from_numpy(arr)
    except Exception:
        return None


def trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        import onnx  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _export_onnx(model: nn.Module, path: Path, device: str, dtype: torch.dtype) -> None:
    # Force CONTIGUOUS (NCHW) layout for the traced copy: the production inference
    # model is channels_last, but the TRT runner feeds NCHW-contiguous buffers, so
    # exporting a channels_last graph produces an engine that misreads the input
    # (garbage outputs -> gate fail). memory_format changes strides only, not
    # weights/outputs, so this is equivalence-preserving.
    # deepcopy: .to(dtype) is in-place on module params, so exporting from the
    # live model would corrupt self.model (the torch fallback). Export a throwaway
    # NCHW fp16 copy instead.
    wrap = _PVTuple(copy.deepcopy(model)).eval().to(device).to(dtype).to(memory_format=torch.contiguous_format)
    dummy = torch.zeros(128, 13, 41, 41, device=device, dtype=dtype)
    torch.onnx.export(
        wrap, (dummy,), str(path),
        input_names=["input"], output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def _build_engine(onnx_path: Path, max_batch: int, opt_batch: int = 128):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 13, 41, 41), (opt_batch, 13, 41, 41), (max_batch, 13, 41, 41))
    cfg.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized)


_TRT_TO_TORCH = None


class TRTForward:
    """Callable forward backed by a TRT engine; matches model.forward_policy_value."""

    def __init__(self, engine, device: str = "cuda") -> None:
        import tensorrt as trt

        global _TRT_TO_TORCH
        if _TRT_TO_TORCH is None:
            _TRT_TO_TORCH = {
                trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
            }
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.device = device
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.in_name = "input"
        self.out_names = [n for n in names if n != self.in_name]
        self.in_dt = _TRT_TO_TORCH[engine.get_tensor_dtype(self.in_name)]

    @torch.inference_mode()
    def forward_policy_value(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = int(x.shape[0])
        self.ctx.set_input_shape(self.in_name, (bs, 13, 41, 41))
        # Run on the CURRENT stream so the input copy (.to/.contiguous), the TRT
        # enqueue that reads it, and the output reads are all ordered on one
        # stream. (A separate stream without cross-stream sync raced the input
        # copy against the TRT kernel -> garbage/NaN.) zeros_ (not empty) so a
        # short-writing engine can never leak uninitialized garbage into outputs.
        stream = torch.cuda.current_stream(device=self.device)
        xin = x.to(self.in_dt).contiguous()
        self.ctx.set_tensor_address(self.in_name, xin.data_ptr())
        outs: dict[str, torch.Tensor] = {}
        for n in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            buf = torch.zeros(shp, device=self.device, dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype(n)])
            outs[n] = buf
            self.ctx.set_tensor_address(n, buf.data_ptr())
        ok = self.ctx.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        stream.synchronize()
        # Return float32 to match the torch path's downstream (decode/gather).
        return {"policy": outs["policy"].float(), "value": outs["value"].float()}


def build_trt_forward(
    model: nn.Module,
    *,
    max_batch: int,
    device: str = "cuda",
    sample_inputs: torch.Tensor | None = None,
    policy_tol: float = 0.05,
    value_tol: float = 0.05,
    # Per-forward argmax floor is a NaN/gross-corruption sanity check, not the
    # strength gate. FP16 TRT matches torch's per-leaf argmax ~96.9% on real
    # positions; the actual SEARCH strength was validated offline (tv5: paired
    # per-decision value-regret = -0.002 +/- 0.0035 win-prob over 400 decisions,
    # i.e. strength-equivalent). So 0.90 admits FP16 while still rejecting a
    # grossly-wrong engine (the old buffer/stream bug gave argmax 0.0).
    argmax_match_min: float = 0.90,
    precision: str = "fp16",
    verbose: bool = True,
) -> tuple[Callable | None, Mapping]:
    """Build a TRT FP16 forward from `model` (the folded inference clone) and gate
    on correctness vs the torch FP16 reference. Returns (trt_forward_or_None, info).

    On any failure (no TRT, build error, correctness gate fail) returns (None, info)
    so the caller falls back to the torch forward.
    """
    info: dict = {"adopted": False, "reason": None}
    if not trt_available():
        info["reason"] = "tensorrt/onnx unavailable"
        return None, info
    try:
        import tensorrt as trt
        export_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        tmp = Path(tempfile.mkdtemp(prefix=f"dense_cnn_trt_{precision}_"))
        onnx_path = tmp / f"model_{precision}.onnx"
        t0 = time.perf_counter()
        _export_onnx(model, onnx_path, device, export_dtype)
        engine = _build_engine(onnx_path, max_batch=max_batch)
        build_s = time.perf_counter() - t0
        info["precision"] = precision
        info["build_seconds"] = build_s
        info["trt_version"] = trt.__version__
        fwd = TRTForward(engine, device=device)

        # Correctness gate vs torch FP16 reference on REAL game positions (sharp
        # policy -> meaningful argmax); fall back to synthetic only if unavailable.
        if sample_inputs is None:
            sample_inputs = _representative_inputs(min(128, max_batch))
        if sample_inputs is None:
            g = torch.Generator().manual_seed(0)
            sample_inputs = (torch.rand((128, 13, 41, 41), generator=g) > 0.7).float()
        x = sample_inputs.to(device).to(memory_format=torch.channels_last)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                ref = model.forward_policy_value(x)
            trt_out = fwd.forward_policy_value(x)
        from .losses import decode_binned_value
        ref_p = ref["policy"].float(); ref_v = ref["value"].float()
        trt_p = trt_out["policy"].float(); trt_v = trt_out["value"].float()
        policy_argmax_match = (ref_p.argmax(1) == trt_p.argmax(1)).float().mean().item()
        policy_max_err = (ref_p - trt_p).abs().max().item()
        # Gate on the DECODED scalar value (what search consumes), in [-1, 1] —
        # not the raw 65-bin logits (a logit tolerance is the wrong scale).
        value_logit_max_err = (ref_v - trt_v).abs().max().item()
        value_max_err = (decode_binned_value(ref_v) - decode_binned_value(trt_v)).abs().max().item()
        info.update({
            "policy_argmax_match": policy_argmax_match,
            "policy_max_abs_err": policy_max_err,
            "value_max_abs_err": value_max_err,
            "value_logit_max_abs_err": value_logit_max_err,
        })
        gate_ok = (policy_argmax_match >= argmax_match_min
                   and value_max_err <= value_tol)
        if not gate_ok:
            info["reason"] = (f"correctness gate FAILED (argmax_match={policy_argmax_match:.4f}, "
                              f"value_err={value_max_err:.4f})")
            if verbose:
                print(f"[trt_backend] {info['reason']} -> falling back to torch", flush=True)
            return None, info
        info["adopted"] = True
        info["reason"] = "ok"
        if verbose:
            print(f"[trt_backend] adopted TRT FP16 (build {build_s:.1f}s, "
                  f"argmax_match={policy_argmax_match:.4f}, value_err={value_max_err:.4f})", flush=True)
        return fwd.forward_policy_value, info
    except Exception as e:  # any failure -> fallback
        info["reason"] = f"exception: {e!r}"
        if verbose:
            print(f"[trt_backend] build/gate failed: {e!r} -> falling back to torch", flush=True)
        return None, info
