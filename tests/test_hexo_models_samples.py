"""Dense CNN D6 symmetry and compact-sample expansion tests."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch = pytest.importorskip("torch")
d6 = importlib.import_module("hexo_models.dense_cnn.d6")
samples = importlib.import_module("hexo_models.dense_cnn.samples")
geometry = importlib.import_module("hexo_models.dense_cnn.geometry")

BOARD_SIZE = 41
TRANSFORMS = tuple(d6.D6Symmetry(index) for index in range(d6.D6_SIZE))


def _orbit(coord: tuple[int, int]) -> set[tuple[int, int]]:
    q, r = coord
    return {
        (q, r), (-r, q + r), (-q - r, q), (-q, -r), (r, -q - r), (q + r, -q),
        (r, q), (-q - r, r), (q, -q - r), (-r, -q), (q + r, -r), (-q, q + r),
    }


def _transformed(coord: tuple[int, int], transform: Any) -> tuple[int, int]:
    moved = d6.transform_coord(coord, transform, center=d6.Axial(0, 0))
    return int(moved.q), int(moved.r)


def test_d6_has_twelve_unique_hex_transforms() -> None:
    probe = (2, 5)
    image = {_transformed(probe, transform) for transform in TRANSFORMS}
    assert len(TRANSFORMS) == 12
    assert image == _orbit(probe)


def test_d6_transforms_round_trip_through_inverse() -> None:
    domain = [(q, r) for q in range(-3, 4) for r in range(-3, 4)]
    for transform in TRANSFORMS:
        inverse = d6.D6Symmetry(d6.inverse_index(transform.index))
        image = [_transformed(coord, transform) for coord in domain]
        assert len(set(image)) == len(domain)
        for coord in domain:
            assert _transformed(_transformed(coord, transform), inverse) == coord


def _sample(policy: tuple[tuple[tuple[int, int], float], ...]) -> Any:
    packed = tuple((int(d6.pack_coord_id(d6.Axial(q, r))), float(w)) for (q, r), w in policy)
    return samples.Model1SampleData(
        game_id="s",
        turn_index=0,
        current_player="player0",
        phase="FirstStone",
        center=(0, 0),
        stones=(),
        legal_action_ids=tuple(action for action, _ in packed),
        policy=packed,
        root_prior_policy=packed,
        opp_policy=packed,
        value=-0.375,
        short_term_value=((1, 0.125), (4, -0.75)),
    )


def _flat(coord: tuple[int, int]) -> int:
    return int(geometry.coord_to_flat(d6.Axial(*coord), center=d6.Axial(0, 0)))


def test_policy_targets_move_with_d6_coordinates() -> None:
    raw = (((2, -3), 0.7), ((-4, 1), 0.3))
    sample = _sample(raw)
    transform = next(t for t in TRANSFORMS if _transformed((2, -3), t) != (2, -3))

    expanded = samples.expand_sample(sample, symmetry=transform)
    policy = expanded["policy"].reshape(-1)
    opp = expanded["opp_policy"].reshape(-1)

    assert policy.numel() == BOARD_SIZE * BOARD_SIZE
    for coord, weight in raw:
        assert float(policy[_flat(_transformed(coord, transform))]) == pytest.approx(weight, abs=1e-6)
        assert float(opp[_flat(_transformed(coord, transform))]) == pytest.approx(weight, abs=1e-6)
    assert math.isclose(float(policy.sum()), 1.0, abs_tol=1e-6)


def test_value_and_short_term_value_are_d6_invariant() -> None:
    sample = _sample((((2, -3), 0.7), ((-4, 1), 0.3)))
    baseline = samples.expand_sample(sample, symmetry=TRANSFORMS[0])
    for transform in TRANSFORMS:
        expanded = samples.expand_sample(sample, symmetry=transform)
        assert float(expanded["value"]) == pytest.approx(float(baseline["value"]))
        assert float(expanded["stvalue_1"]) == pytest.approx(float(baseline["stvalue_1"]))
        assert float(expanded["stvalue_4"]) == pytest.approx(float(baseline["stvalue_4"]))
