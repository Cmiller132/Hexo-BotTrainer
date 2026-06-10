from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    _src = _ROOT / "packages" / _pkg / "python"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from dense_cnn_restnet.compile_backend import bucket_sizes  # noqa: E402
from dense_cnn_restnet.config import parse_model1_config  # noqa: E402
from dense_cnn_restnet.constants import BOARD_AREA, BOARD_SIZE, INPUT_CHANNELS, VALUE_BINS  # noqa: E402
from dense_cnn_restnet.inference import DenseCNNInference  # noqa: E402


def test_compile_bucket_sizes_are_power_of_two_until_cap() -> None:
    assert bucket_sizes(1) == (1,)
    assert bucket_sizes(7) == (1, 2, 4, 7)
    assert bucket_sizes(16) == (1, 2, 4, 8, 16)


def test_compile_and_tensorrt_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_model1_config(
            {
                "performance": {
                    "inference_use_tensorrt": True,
                    "inference_use_torch_compile": True,
                }
            }
        )


def test_runtime_compile_and_tensorrt_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        DenseCNNInference(
            torch.nn.Identity(),
            device="cpu",
            optimize_for_inference=False,
            use_torch_compile=True,
            use_trt=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile inference gate is CUDA-only")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_torch_compile_gate_adopts_and_matches_eager_cuda() -> None:
    class TinyPV(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            policy = torch.zeros((batch, BOARD_AREA), device=inputs.device, dtype=inputs.dtype)
            policy[:, 3] = self.scale.to(inputs.dtype) * 8.0
            value = torch.zeros((batch, VALUE_BINS), device=inputs.device, dtype=inputs.dtype)
            return {"policy": policy, "value": value}

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            return self.forward_policy_value(inputs)

    model = TinyPV()
    inference = DenseCNNInference(
        model,
        device="cuda",
        amp=True,
        max_batch_size=4,
        optimize_for_inference=False,
        use_torch_compile=True,
    )

    assert inference.compile_info["adopted"] is True
    assert inference.bucket_pad_multiple is None
    x = torch.zeros((3, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float16)
    out = inference._forward_inputs_device(x)
    assert out["policy"].argmax(1).tolist() == [3, 3, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile fallback test is CUDA-only")
def test_torch_compile_allowed_fallback_restores_configured_bucket_multiple_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dense_cnn_restnet import compile_backend

    class TinyPV(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            policy = torch.zeros((batch, BOARD_AREA), device=inputs.device, dtype=inputs.dtype)
            value = torch.zeros((batch, VALUE_BINS), device=inputs.device, dtype=inputs.dtype)
            return {"policy": policy, "value": value}

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            return self.forward_policy_value(inputs)

    monkeypatch.setattr(
        compile_backend,
        "build_compiled_forward",
        lambda *_args, **_kwargs: (None, {"adopted": False, "reason": "forced fallback"}),
    )
    inference = DenseCNNInference(
        TinyPV(),
        device="cuda",
        amp=True,
        max_batch_size=4,
        optimize_for_inference=False,
        bucket_pad_multiple=16,
        use_torch_compile=True,
        compile_allow_fallback=True,
    )

    assert inference.compile_info == {"adopted": False, "reason": "forced fallback"}
    assert inference.bucket_pad_multiple == 16
