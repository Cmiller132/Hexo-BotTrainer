"""Unit proof of the config-driven cosine LR schedule used by main_4.

main_4 owner ruling (2026-07-20): cosine decay 2e-4 -> 2e-5 over 150 epochs,
then HOLD at the 2e-5 floor. The trainer already exposes this as the pure
function ``hexfield_eq.trainer.scheduled_lr`` (schedule="cosine", base_lr,
final_lr, decay_epochs, with a clamped p = epoch/decay_epochs so it holds
final_lr for epoch > decay_epochs). This test pins the exact main_4 curve:

  * lr(epoch=0)   == 2e-4  (starts at base; warmup disabled)
  * lr(epoch=150) == 2e-5  (reaches the floor exactly at decay_epochs)
  * lr(epoch>150) == 2e-5  (holds the floor)
  * monotone non-increasing across the whole run
  * the analytic cosine midpoint at epoch 75

Runs in the Windows CPU suite (CUDA_VISIBLE_DEVICES=-1); pure function, no CUDA.
"""

from __future__ import annotations

import math

import pytest

from hexfield_eq.trainer import scheduled_lr

# The main_4 schedule parameters (see configs/hexfield_eq_main_4.toml).
BASE_LR = 2e-4
FINAL_LR = 2e-5
DECAY_EPOCHS = 150
WARMUP_STEPS = 0
# A global step comfortably past any warmup; with WARMUP_STEPS=0 the warmup
# branch is skipped entirely, so the value is epoch-only.
STEP = 10_000


def _lr(epoch: int, global_step: int = STEP) -> float:
    return scheduled_lr(
        schedule="cosine",
        base_lr=BASE_LR,
        final_lr=FINAL_LR,
        warmup_steps=WARMUP_STEPS,
        decay_epochs=DECAY_EPOCHS,
        global_step=global_step,
        epoch=epoch,
    )


def test_lr_starts_at_base():
    # epoch 0 -> p=0 -> cos=1 -> base_lr, with warmup disabled even at step 0.
    assert math.isclose(_lr(0, global_step=0), BASE_LR, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(_lr(0), BASE_LR, rel_tol=0.0, abs_tol=1e-12)


def test_lr_reaches_floor_at_decay_epochs():
    # epoch == decay_epochs -> p=1 -> cos=0 -> exactly final_lr.
    assert math.isclose(_lr(DECAY_EPOCHS), FINAL_LR, rel_tol=0.0, abs_tol=1e-12)


def test_lr_holds_floor_after_decay():
    for epoch in (DECAY_EPOCHS + 1, 200, 500, 10_000):
        assert math.isclose(_lr(epoch), FINAL_LR, rel_tol=0.0, abs_tol=1e-12), epoch


def test_lr_midpoint_matches_analytic_cosine():
    # epoch = decay_epochs/2 -> p=0.5 -> cos=0.5 -> final + (base-final)*0.5.
    expected = FINAL_LR + (BASE_LR - FINAL_LR) * 0.5
    assert math.isclose(_lr(DECAY_EPOCHS // 2), expected, rel_tol=1e-9)


def test_lr_is_monotone_non_increasing():
    values = [_lr(epoch) for epoch in range(0, 201)]
    for earlier, later in zip(values, values[1:]):
        assert later <= earlier + 1e-15, (earlier, later)
    # And strictly decreasing across the decay window (no flat plateau before
    # the floor) — first vs last decayed epoch.
    assert values[0] > values[DECAY_EPOCHS]


def test_lr_never_below_floor_or_above_base():
    for epoch in range(0, 301):
        lr = _lr(epoch)
        assert FINAL_LR - 1e-15 <= lr <= BASE_LR + 1e-15, (epoch, lr)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
