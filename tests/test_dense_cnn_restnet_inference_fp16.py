from __future__ import annotations

import struct
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

from dense_cnn_restnet.config import parse_model1_config  # noqa: E402
from dense_cnn_restnet.constants import BOARD_AREA, BOARD_SIZE, INPUT_CHANNELS, VALUE_BINS  # noqa: E402
from dense_cnn_restnet.inference import DenseCNNInference  # noqa: E402


class _DeterministicPV(torch.nn.Module):
    def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        batch = int(inputs.shape[0])
        base = torch.linspace(-1.0, 1.0, BOARD_AREA, device=inputs.device, dtype=inputs.dtype)
        policy = base.unsqueeze(0).repeat(batch, 1)
        value = torch.zeros((batch, VALUE_BINS), device=inputs.device, dtype=inputs.dtype)
        return {"policy": policy, "value": value}

    def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.forward_policy_value(inputs)


def _payload(rows: int) -> dict[str, object]:
    inputs = torch.zeros((rows, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float16)
    flats: list[int] = []
    offsets = [0]
    for row in range(rows):
        flats.extend([row, BOARD_AREA - 1 - row])
        offsets.append(len(flats))
    return {
        "inputs": inputs.numpy().tobytes(),
        "shape": tuple(inputs.shape),
        "legal_flat_indices_bytes": torch.tensor(flats, dtype=torch.int64).numpy().tobytes(),
        "legal_row_offsets": tuple(offsets),
    }


def test_config_parses_fp16_inference_flags() -> None:
    cfg = parse_model1_config(
        {
            "performance": {
                "inference_fp16_model": True,
                "inference_fp16_allow_fallback": True,
            }
        }
    )
    assert cfg.performance.inference_fp16_model is True
    assert cfg.performance.inference_fp16_allow_fallback is True


def test_payload_chunked_matches_unchunked_bytes() -> None:
    payload = _payload(5)
    chunked = DenseCNNInference(_DeterministicPV(), device="cpu", amp=False, max_batch_size=2)
    unchunked = DenseCNNInference(_DeterministicPV(), device="cpu", amp=False, max_batch_size=99)

    a = chunked.evaluate_model1_payload(payload)
    b = unchunked.evaluate_model1_payload(payload)

    assert a["values_bytes"] == b["values_bytes"]
    assert a["priors_bytes"] == b["priors_bytes"]


def test_payload_legal_softmax_bytes_are_row_local() -> None:
    payload = _payload(2)
    inference = DenseCNNInference(_DeterministicPV(), device="cpu", amp=False, max_batch_size=1)

    output = inference.evaluate_model1_payload(payload)

    values = struct.unpack("2f", output["values_bytes"])
    priors = struct.unpack("4f", output["priors_bytes"])
    assert values == pytest.approx((0.0, 0.0), abs=1.0e-6)
    # Each row softmaxes only over its two legal flats.
    row0 = torch.softmax(torch.tensor([-1.0, 1.0]), dim=0).tolist()
    row1 = torch.softmax(torch.tensor([-1.0 + 2.0 / (BOARD_AREA - 1), 1.0 - 2.0 / (BOARD_AREA - 1)]), dim=0).tolist()
    assert priors[:2] == pytest.approx(row0, rel=1.0e-6, abs=1.0e-6)
    assert priors[2:] == pytest.approx(row1, rel=1.0e-6, abs=1.0e-6)


def test_payload_byte_layout_is_dense_f32() -> None:
    # Byte LAYOUT contract only: row-major f32, values then priors, lengths
    # exactly rows / selected legal counts. (Numeric values are covered with
    # proper tolerances above — decode_binned_value of a uniform head is ~1e-9,
    # not exactly 0.0, so exact byte comparison would be flaky by construction.)
    payload = _payload(2)
    inference = DenseCNNInference(_DeterministicPV(), device="cpu", amp=False, max_batch_size=1)

    output = inference.evaluate_model1_payload(payload)

    assert len(output["values_bytes"]) == 2 * 4
    assert len(output["priors_bytes"]) == 4 * 4
    values = struct.unpack("2f", output["values_bytes"])
    priors = struct.unpack("4f", output["priors_bytes"])
    assert all(abs(item) <= 1.0 for item in values)
    assert priors[0] + priors[1] == pytest.approx(1.0, abs=1.0e-6)
    assert priors[2] + priors[3] == pytest.approx(1.0, abs=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 model adoption is CUDA-only")
def test_fp16_model_gate_adopts_sharp_policy_cuda() -> None:
    class SharpModel(torch.nn.Module):
        def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            policy = torch.zeros((batch, BOARD_AREA), device=inputs.device, dtype=inputs.dtype)
            policy[:, 17] = 10.0
            value = torch.zeros((batch, VALUE_BINS), device=inputs.device, dtype=inputs.dtype)
            return {"policy": policy, "value": value}

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            return self.forward_policy_value(inputs)

    inference = DenseCNNInference(
        SharpModel(),
        device="cuda",
        amp=True,
        max_batch_size=4,
        optimize_for_inference=False,
        fp16_model=True,
    )

    assert inference.fp16_info["adopted"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 model adoption is CUDA-only")
def test_fp16_model_gate_requires_real_representative_inputs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from dense_cnn_restnet import trt_backend

    class SharpModel(torch.nn.Module):
        def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            policy = torch.zeros((batch, BOARD_AREA), device=inputs.device, dtype=inputs.dtype)
            value = torch.zeros((batch, VALUE_BINS), device=inputs.device, dtype=inputs.dtype)
            return {"policy": policy, "value": value}

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            return self.forward_policy_value(inputs)

    monkeypatch.setattr(trt_backend, "_representative_inputs", lambda _count: None)
    with pytest.raises(RuntimeError, match="representative real-position inputs unavailable"):
        DenseCNNInference(
            SharpModel(),
            device="cuda",
            amp=True,
            max_batch_size=4,
            optimize_for_inference=False,
            fp16_model=True,
        )
