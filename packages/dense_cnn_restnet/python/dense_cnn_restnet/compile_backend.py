"""Optional torch.compile backend for dense_cnn_restnet inference.

The compiled callable is a drop-in replacement for ``forward_policy_value``.
Adoption is fail-loud by default and gated against the eager torch reference on
representative encoded positions, matching the TensorRT backend contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


class CompileAdoptError(RuntimeError):
    """Raised when torch.compile cannot be adopted and fallback is disabled."""


def bucket_sizes(max_batch: int) -> tuple[int, ...]:
    max_batch = max(1, int(max_batch))
    sizes: list[int] = []
    size = 1
    while size < max_batch:
        sizes.append(size)
        size <<= 1
    sizes.append(max_batch)
    return tuple(dict.fromkeys(sizes))


class CompiledForward:
    """Shape-bucketed compiled forward with persistent input buffers."""

    def __init__(self, compiled: Callable[[torch.Tensor], Mapping[str, torch.Tensor]], *, max_batch: int) -> None:
        self._compiled = compiled
        self._max_batch = int(max_batch)
        self._buffers: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}

    @torch.inference_mode()
    def __call__(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        key = (inputs.device, inputs.dtype, tuple(inputs.shape))
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.empty_like(inputs)
            self._buffers[key] = buf
        buf.copy_(inputs)
        output = self._compiled(buf)
        return {key: value.clone() for key, value in output.items()}

    @torch.inference_mode()
    def warm(self, *, channels: int, board_size: int, device: torch.device, dtype: torch.dtype) -> None:
        for batch in bucket_sizes(self._max_batch):
            x = torch.zeros((batch, channels, board_size, board_size), device=device, dtype=dtype)
            if x.ndim == 4:
                x = x.to(memory_format=torch.channels_last)
            _ = self(x)


def build_compiled_forward(
    model: nn.Module,
    *,
    max_batch: int,
    sample_inputs: torch.Tensor | None = None,
    value_tol: float = 0.05,
    argmax_match_min: float = 0.90,
    allow_torch_fallback: bool = False,
) -> tuple[Callable[[torch.Tensor], Mapping[str, torch.Tensor]] | None, Mapping[str, Any]]:
    info: dict[str, Any] = {"adopted": False, "reason": None}
    banner = "!" * 72

    def fail(reason: str, exc: BaseException | None = None):
        info["reason"] = reason
        if allow_torch_fallback:
            print(
                f"\n{banner}\n[compile_backend] torch.compile NOT ADOPTED: {reason}\n"
                f"[compile_backend] allow_torch_fallback=True -> running eager torch.\n{banner}",
                flush=True,
            )
            return None, info
        print(
            f"\n{banner}\n[compile_backend] FATAL: torch.compile NOT ADOPTED and fallback is DISABLED.\n"
            f"[compile_backend] reason: {reason}\n{banner}",
            flush=True,
        )
        raise CompileAdoptError(f"torch.compile NOT ADOPTED and fallback disabled: {reason}") from exc

    if not torch.cuda.is_available():
        return fail("CUDA unavailable")
    if not hasattr(model, "forward_policy_value"):
        return fail("model does not expose forward_policy_value")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda")
    if torch.device(device).type != "cuda":
        return fail(f"model is on {device}, expected cuda")
    dtype = next((p.dtype for p in model.parameters()), torch.float32)

    try:
        # dynamic=False compiles one static graph per bucket shape, so dynamo
        # needs a recompile-cache budget above the bucket count (default 8 <
        # ~log2(max_batch)+1 buckets, which hard-fails under fullgraph=True with
        # FailOnRecompileLimitHit). Size it to the bucket count with headroom.
        import torch._dynamo as _dynamo

        n_buckets = len(bucket_sizes(max_batch))
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, n_buckets * 2 + 8)
        _dynamo.config.accumulated_cache_size_limit = max(
            getattr(_dynamo.config, "accumulated_cache_size_limit", 0), n_buckets * 4 + 16
        )
        compiled_raw = torch.compile(
            model.forward_policy_value,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        forward = CompiledForward(compiled_raw, max_batch=max_batch)
        from .constants import BOARD_SIZE, INPUT_CHANNELS

        forward.warm(channels=INPUT_CHANNELS, board_size=BOARD_SIZE, device=torch.device(device), dtype=dtype)

        if sample_inputs is None:
            from . import trt_backend

            sample_inputs = trt_backend._representative_inputs(min(128, int(max_batch)))
        if sample_inputs is None:
            return fail("representative real-position inputs unavailable")

        x = sample_inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
        with torch.inference_mode():
            eager = model.forward_policy_value(x)
            compiled = forward(x)
        from .losses import decode_binned_value

        eager_policy = eager["policy"].float()
        compiled_policy = compiled["policy"].float()
        eager_value = eager["value"].float()
        compiled_value = compiled["value"].float()
        policy_argmax_match = (eager_policy.argmax(1) == compiled_policy.argmax(1)).float().mean().item()
        policy_max_err = (eager_policy - compiled_policy).abs().max().item()
        value_max_err = (decode_binned_value(eager_value) - decode_binned_value(compiled_value)).abs().max().item()
        info.update(
            {
                "policy_argmax_match": float(policy_argmax_match),
                "policy_max_abs_err": float(policy_max_err),
                "value_max_abs_err": float(value_max_err),
                "bucket_sizes": bucket_sizes(max_batch),
            }
        )
        if policy_argmax_match < argmax_match_min or value_max_err > value_tol:
            return fail(
                "correctness gate FAILED "
                f"(argmax_match={policy_argmax_match:.4f}, decoded_value_err={value_max_err:.4f})"
            )
        info["adopted"] = True
        info["reason"] = "ok"
        print(
            "[compile_backend] adopted torch.compile "
            f"(argmax_match={policy_argmax_match:.4f}, value_err={value_max_err:.4f})",
            flush=True,
        )
        return forward, info
    except CompileAdoptError:
        raise
    except Exception as exc:
        return fail(f"exception during build/gate: {exc!r}", exc=exc)
