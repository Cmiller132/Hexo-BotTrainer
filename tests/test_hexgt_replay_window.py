"""Replay-buffer changes in the RL driver: recency-weighted sampling + cap-bounded
growing window.

Tests the pure, importable helpers in ``scripts/_rl_train.py``:
  - ``epoch_recency_weight``: geometric 0.9^age decay.
  - ``select_window_epochs``: window grows with epochs, caps at ~pool_cap positions
    (drops the oldest), honors the base-window floor, and queries epoch positions
    lazily/once (bounded metadata access — the RAM-safety contract).
  - recency-weighted group sampling (``random.Random.choices`` with the per-shard
    weights) over-represents recent epochs by the decay ratio.

These are pure (no GPU / no shard IO), so the change is exercised without touching
the live run. D6/parity are unaffected (this is replay SAMPLING only).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    p = ROOT / "packages" / package / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _mod():
    return pytest.importorskip("_rl_train", reason="RL driver import (needs built hexo_models)")


# --- recency weight -----------------------------------------------------------

def test_epoch_recency_weight() -> None:
    m = _mod()
    assert m.epoch_recency_weight(40, 40, 0.9) == pytest.approx(1.0)
    assert m.epoch_recency_weight(39, 40, 0.9) == pytest.approx(0.9)
    assert m.epoch_recency_weight(38, 40, 0.9) == pytest.approx(0.81)
    assert m.epoch_recency_weight(30, 40, 0.9) == pytest.approx(0.9 ** 10)
    # never weights the future above the present (clamped at 0 age)
    assert m.epoch_recency_weight(41, 40, 0.9) == pytest.approx(1.0)


# --- window selection: cap, floor, growth, bounded metadata access ------------

def test_select_window_caps_and_drops_oldest() -> None:
    m = _mod()
    epochs, total = m.select_window_epochs(
        40, lambda e: 18_000, base_window=8, pool_cap=500_000
    )
    # 18k/epoch under a 500k cap -> 27 epochs (27*18k=486k; a 28th would hit 504k)
    assert len(epochs) == 27
    assert epochs[-1] == 40 and epochs[0] == 14  # newest kept, oldest dropped
    assert total == 486_000
    assert epochs == sorted(epochs)


def test_select_window_floor_overrides_cap() -> None:
    m = _mod()
    # pathologically huge epochs: the base-window floor still keeps the newest 8
    epochs, total = m.select_window_epochs(
        20, lambda e: 1_000_000, base_window=8, pool_cap=500_000
    )
    assert len(epochs) == 8
    assert epochs == list(range(13, 21))


def test_select_window_grows_early() -> None:
    m = _mod()
    # only 4 epochs exist -> window is all of them (well under any cap)
    epochs, total = m.select_window_epochs(
        3, lambda e: 18_000, base_window=8, pool_cap=500_000
    )
    assert epochs == [0, 1, 2, 3]
    assert total == 72_000


def test_select_window_reads_metadata_lazily_and_bounded() -> None:
    """RAM-safety contract: positions are queried once per CANDIDATE epoch (cheap
    metadata), and the scan STOPS at the cap rather than walking the whole history
    — it never enumerates (let alone loads) the full pool."""
    m = _mod()
    seen: list[int] = []

    def pos(e):
        seen.append(e)
        return 18_000

    epochs, _ = m.select_window_epochs(40, pos, base_window=8, pool_cap=500_000)
    assert len(seen) == len(set(seen))          # each epoch queried at most once
    assert len(seen) == 28                       # 27 kept + 1 that tripped the cap
    assert min(epochs) > 0                        # stopped early; didn't scan to epoch 0


# --- recency-weighted group sampling (what the driver does) -------------------

def test_recency_weighted_sampling_overrepresents_recent() -> None:
    m = _mod()
    cur = 4
    # 3 equal-sized epochs; per-shard weight = epoch recency weight
    epochs = [2, 3, 4]
    n_per = 12
    shards, weights = [], []
    for e in epochs:
        w = m.epoch_recency_weight(e, cur, 0.9)
        for i in range(n_per):
            shards.append((e, i))
            weights.append(w)

    rng = random.Random(1234)
    counts = {e: 0 for e in epochs}
    draws = 90_000
    for s in rng.choices(shards, weights=weights, k=draws):
        counts[s[0]] += 1

    # expected fraction per epoch ∝ n_per * decay^age (equal n -> just the weights)
    wsum = sum(m.epoch_recency_weight(e, cur, 0.9) for e in epochs)
    for e in epochs:
        exp = m.epoch_recency_weight(e, cur, 0.9) / wsum
        got = counts[e] / draws
        assert got == pytest.approx(exp, abs=0.02), f"epoch {e}: got {got:.3f} exp {exp:.3f}"
    # strictly monotone: newest sampled most, oldest least
    assert counts[4] > counts[3] > counts[2]
