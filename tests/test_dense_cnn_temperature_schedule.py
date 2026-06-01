"""Move-selection temperature schedule: anchor interpolation + config parsing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

selfplay = importlib.import_module("hexo_models.dense_cnn.selfplay")
config = importlib.import_module("hexo_models.dense_cnn.config")

# The user's requested curve: hot opening, 0.5 by ply 30, 0.3 by ply 90, then keep
# decreasing along the final slope down to the floor.
SCHEDULE = ((0, 1.0), (30, 0.5), (90, 0.3))
FLOOR = 0.1


def _temp(move: int) -> float:
    return selfplay._move_temperature(
        move, initial=1.0, final=1.0, decay_moves=0, schedule=SCHEDULE, floor=FLOOR
    )


def test_schedule_hits_requested_anchors() -> None:
    assert _temp(0) == pytest.approx(1.0)
    assert _temp(30) == pytest.approx(0.5)
    assert _temp(90) == pytest.approx(0.3)


def test_schedule_interpolates_between_anchors() -> None:
    assert _temp(15) == pytest.approx(0.75)          # halfway 1.0 -> 0.5
    assert _temp(60) == pytest.approx(0.4)           # halfway 0.5 -> 0.3
    assert _temp(45) == pytest.approx(0.45)          # quarter into 30..90


def test_schedule_continues_past_last_anchor_then_floors() -> None:
    # Final slope is (0.3-0.5)/60 = -1/300 per ply; from (90,0.3) it reaches the
    # 0.1 floor at ply 150, then holds.
    assert _temp(120) == pytest.approx(0.2)
    assert _temp(150) == pytest.approx(0.1)
    assert _temp(300) == pytest.approx(0.1)


def test_schedule_is_monotonic_non_increasing() -> None:
    vals = [_temp(m) for m in range(0, 200)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))


def test_linear_fallback_unchanged_when_no_schedule() -> None:
    f = lambda m: selfplay._move_temperature(m, initial=1.0, final=0.2, decay_moves=30)
    assert f(0) == pytest.approx(1.0)
    assert f(15) == pytest.approx(0.6)
    assert f(30) == pytest.approx(0.2)
    assert f(60) == pytest.approx(0.2)   # held flat after decay_moves
    g = lambda m: selfplay._move_temperature(m, initial=1.0, final=0.2, decay_moves=0)
    assert g(0) == pytest.approx(1.0) and g(99) == pytest.approx(1.0)  # decay disabled


def test_parse_temperature_schedule_sorts_and_validates() -> None:
    parsed = config._parse_temperature_schedule([[90, 0.3], [0, 1.0], [30, 0.5]])
    assert parsed == ((0, 1.0), (30, 0.5), (90, 0.3))
    assert config._parse_temperature_schedule(()) == ()
    with pytest.raises(ValueError):
        config._parse_temperature_schedule([[0, 1.0], [0, 0.5]])    # duplicate move
    with pytest.raises(ValueError):
        config._parse_temperature_schedule([[-1, 1.0]])             # negative move
    with pytest.raises(ValueError):
        config._parse_temperature_schedule([[0, 0.0]])              # non-positive temp
    with pytest.raises(ValueError):
        config._parse_temperature_schedule([[0, 1.0, 2.0]])         # not a pair


def test_selfplay_config_defaults_keep_empty_schedule() -> None:
    cfg = config.Model1SelfPlayConfig()
    assert cfg.temperature_schedule == ()
    assert cfg.temperature_floor == pytest.approx(0.1)
