"""Tests for A7 evaluator batch-shape bucketing (PB cuDNN cold-start fix).

The bucketing pads forward batches up to power-of-two sizes so cuDNN autotune
sees a handful of shapes instead of hundreds. Correctness rests on one property:
padded rows must never leak into the real rows. In eval mode every op in the
trunk/heads (conv, eval-BatchNorm with running stats, ReLU, FC) is per-sample,
so this holds — these tests pin both the bucket math and the no-leak property.
"""

from __future__ import annotations

import pytest
import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.constants import BOARD_SIZE, INPUT_CHANNELS
from hexo_models.dense_cnn.inference import DenseCNNInference, _bucket_batch_size


def test_bucket_batch_size_rounds_to_power_of_two_within_cap() -> None:
    cap = 1024
    assert _bucket_batch_size(1, cap) == 1
    assert _bucket_batch_size(2, cap) == 2
    assert _bucket_batch_size(3, cap) == 4
    assert _bucket_batch_size(5, cap) == 8
    assert _bucket_batch_size(33, cap) == 64
    assert _bucket_batch_size(64, cap) == 64
    assert _bucket_batch_size(129, cap) == 256
    # at/above the cap the batch is returned unchanged (already chunked upstream)
    assert _bucket_batch_size(1024, cap) == 1024
    assert _bucket_batch_size(1500, cap) == 1500
    assert _bucket_batch_size(0, cap) == 0


def test_bucketing_bounds_distinct_shapes() -> None:
    cap = 1024
    distinct = {_bucket_batch_size(n, cap) for n in range(1, cap + 1)}
    # power-of-two buckets up to the cap: 1,2,4,...,1024 => 11 shapes, vs the
    # ~900 distinct shapes the unbucketed evaluator was measured to autotune.
    assert distinct == {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
    assert len(distinct) == 11


def _small_net() -> Model1Network:
    torch.manual_seed(0)
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=16, blocks=2)
    # Give BatchNorm non-trivial running stats so a leak would actually show.
    for module in net.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 1.5)
    return net.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bucketing path is CUDA-only")
def test_padding_does_not_leak_into_real_rows() -> None:
    """Same real rows, different padding content -> identical real-row outputs."""

    device = torch.device("cuda")
    infer = DenseCNNInference(_small_net(), device=device, amp=False, optimize_for_inference=False)
    assert infer.pad_to_buckets is True

    rows = 37
    real = torch.randn(rows, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # Bucketing on: real rows get padded with zeros up to 64 internally.
    out_zeros = infer._forward_inputs_device(real)

    # Now force the padding to be garbage by manually padding to the same bucket
    # with random rows and running the model directly; the first `rows` must match.
    target = _bucket_batch_size(rows, infer.max_batch_size)
    assert target == 64
    garbage = torch.randn(target, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device).to(
        memory_format=torch.channels_last
    )
    garbage[:rows].copy_(real.to(device).to(memory_format=torch.channels_last))
    with torch.inference_mode():
        out_garbage = infer.model.forward_policy_value(garbage)

    for key in ("policy", "value"):
        a = out_zeros[key].detach().float().cpu()
        b = out_garbage[key].detach().float()[:rows].cpu()
        assert torch.equal(a, b), f"padding leaked into real rows for head {key!r}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bucketing path is CUDA-only")
def test_bucketing_on_off_matches() -> None:
    """Bucketing on vs off agree on the real rows (per-sample equivalence)."""

    device = torch.device("cuda")
    net = _small_net()
    infer = DenseCNNInference(net, device=device, amp=False, optimize_for_inference=False)

    rows = 37
    real = torch.randn(rows, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    infer.pad_to_buckets = True
    on = infer._forward_inputs_device(real)
    infer.pad_to_buckets = False
    off = infer._forward_inputs_device(real)

    for key in ("policy", "value"):
        a = on[key].detach().float().cpu()
        b = off[key].detach().float().cpu()
        # Different forward shapes may pick different cuDNN/tf32 algos, so allow a
        # small tolerance; a real leak would diverge by orders of magnitude.
        assert torch.allclose(a, b, atol=1e-2, rtol=1e-2), f"head {key!r} diverged"
