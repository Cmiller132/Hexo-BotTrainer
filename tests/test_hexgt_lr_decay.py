"""Resume-safe post-warmup LR decay for the hexgt trainer.

The schedule is a pure function of `global_step` (the checkpointed train step), so
it carries across bounces with no scheduler state. Validates warmup -> flat ->
exponential halving -> floor, the disabled path, and step-purity (resume safety).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _trainer(**training_overrides):
    import torch

    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.config import HexgtConfig, HexgtTrainingConfig
    from hexo_models.hexgt.trainer import HexgtTrainer

    cfg = HexgtConfig(training=HexgtTrainingConfig(**training_overrides))
    model = HexgtNetwork(token_dim=32, gnn_layers=1, ctx_layers=1, ffn_dim=32).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=training_overrides.get("learning_rate", 2e-4))
    return HexgtTrainer(model=model, config=cfg, optimizer=opt)


def test_lr_decay_schedule_phases() -> None:
    pytest.importorskip("torch")
    base, start, hl, floor = 2.0e-4, 17429, 10240, 2.5e-5
    tr = _trainer(
        learning_rate=base, warmup_steps=200,
        lr_decay_enabled=True, lr_decay_start_step=start,
        lr_decay_halflife_steps=hl, lr_min=floor,
    )
    # Warmup: linear ramp.
    assert tr._lr_at_step(99) == pytest.approx(base * 100 / 200)
    assert tr._lr_at_step(199) == pytest.approx(base)
    # Flat between warmup end and decay start.
    assert tr._lr_at_step(5000) == pytest.approx(base)
    assert tr._lr_at_step(start - 1) == pytest.approx(base)
    # Decay start: exactly base (2**0).
    assert tr._lr_at_step(start) == pytest.approx(base)
    # One / two / three halflives.
    assert tr._lr_at_step(start + hl) == pytest.approx(base / 2)        # 1.0e-4
    assert tr._lr_at_step(start + 2 * hl) == pytest.approx(base / 4)    # 5.0e-5
    assert tr._lr_at_step(start + 3 * hl) == pytest.approx(floor)       # 2.5e-5 == floor
    # Beyond the floor crossing: clamped, never below floor.
    assert tr._lr_at_step(start + 6 * hl) == pytest.approx(floor)
    assert tr._lr_at_step(start + 100 * hl) == pytest.approx(floor)


def test_lr_decay_disabled_is_flat() -> None:
    pytest.importorskip("torch")
    tr = _trainer(learning_rate=2.0e-4, warmup_steps=200, lr_decay_enabled=False)
    for step in (200, 5000, 50_000, 500_000):
        assert tr._lr_at_step(step) == pytest.approx(2.0e-4)


def test_lr_decay_zero_halflife_is_flat() -> None:
    """A non-positive halflife disables decay even if the flag is on (guard)."""
    pytest.importorskip("torch")
    tr = _trainer(
        learning_rate=2.0e-4, warmup_steps=200,
        lr_decay_enabled=True, lr_decay_start_step=0, lr_decay_halflife_steps=0.0,
    )
    assert tr._lr_at_step(50_000) == pytest.approx(2.0e-4)


def test_lr_at_step_is_pure_function_of_step() -> None:
    """Resume safety: the LR depends ONLY on the passed step, not on any mutable
    trainer/optimizer state — so a bounce that restores global_step restores the LR
    exactly. Same step always yields the same LR; advancing train_state never shifts it."""
    pytest.importorskip("torch")
    tr = _trainer(
        learning_rate=2.0e-4, warmup_steps=200,
        lr_decay_enabled=True, lr_decay_start_step=17429,
        lr_decay_halflife_steps=10240, lr_min=2.5e-5,
    )
    probe = 17429 + 10240
    first = tr._lr_at_step(probe)
    tr.train_state.global_step = 999_999  # mutate state
    assert tr._lr_at_step(probe) == pytest.approx(first)
    # Monotonic non-increasing across the decay region.
    seq = [tr._lr_at_step(17429 + k * 1000) for k in range(0, 40)]
    assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))
