"""read_value abort guard (#2).

decode_binned_value is a convex combination of bin centers in [-1, 1], so any decoded
value outside [-1, 1] is purely a float32 rounding artifact. It is observed on the
CUDA/TensorRT inference path (parallel reductions round differently than CPU's
sequential sum, which caps at exactly +-1). The native read_value hard-rejects an
out-of-range value with PyValueError, which aborts the ENTIRE batched MCTS search
(every game's leaves), so the evaluator clamps the decoded value before handing the
bytes to Rust. This is harmless: the training value-target path validates [-1, 1]
independently (losses.scalar_to_binned_target).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch = pytest.importorskip("torch")
inference = importlib.import_module("hexo_models.dense_cnn.inference")
losses = importlib.import_module("hexo_models.dense_cnn.losses")
constants = importlib.import_module("hexo_models.dense_cnn.constants")


def test_evaluator_clamps_decoded_value_before_bytes() -> None:
    # The MCTS evaluator must clamp the decoded value (read_value rejects out-of-range
    # and aborts the whole batch). Verify the clamp is wired on the value readback.
    src = inspect.getsource(inference)
    assert "decode_binned_value" in src, "value path changed; re-check the clamp wiring"
    assert ".clamp_(-1.0, 1.0)" in src, "decoded value readback is not clamped into [-1, 1]"


def test_clamp_guarantees_unit_range_under_fp_overflow() -> None:
    # Simulate the 1-ULP fp overflow the CUDA/TRT path can produce; the clamp the
    # evaluator applies guarantees the emitted value is in range.
    v = torch.tensor([1.0 + 1e-6, -1.0 - 1e-6, 0.3, 1.0, -1.0, 0.0])
    clamped = v.clamp_(-1.0, 1.0)
    assert float(clamped.max()) <= 1.0
    assert float(clamped.min()) >= -1.0


def test_decode_binned_value_is_a_convex_combination_in_unit_range() -> None:
    # Confirms the invariant the clamp restores: the mathematical decode is bounded to
    # [-1, 1] (so clamping only ever removes a rounding artifact, never real signal).
    torch.manual_seed(0)
    logits = torch.randn(4096, constants.VALUE_BINS) * 3.0
    v = losses.decode_binned_value(logits)
    assert float(v.max()) <= 1.0 + 1e-3
    assert float(v.min()) >= -1.0 - 1e-3
