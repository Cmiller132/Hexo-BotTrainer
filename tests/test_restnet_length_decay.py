"""main_4 sample-pipeline levers for dense_cnn_restnet (2026-06-12).

Covers: C1 length-decay row weighting in materialize_policy_surprise_rows
(dormant by default, exact decay arithmetic, truncated-row moves_left fallback,
max_weight cap, Bernoulli-drop semantics), C2 truncation-poison quarantine
(_rows_to_write), and the new config keys (frozen_win_override,
drop_truncated_rows, length_decay_*) parsing from TOML with byte-identical
defaults.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

torch = pytest.importorskip("torch")
config_module = importlib.import_module("dense_cnn_restnet.config")
d6 = importlib.import_module("dense_cnn_restnet.d6")
replay = importlib.import_module("dense_cnn_restnet.replay")
samples = importlib.import_module("dense_cnn_restnet.samples")
selfplay = importlib.import_module("dense_cnn_restnet.selfplay")


def _aid(q: int, r: int) -> int:
    return int(d6.pack_coord_id(d6.Axial(q, r)))


def _row(
    turn: int,
    *,
    moves_left: float = -1.0,
    policy=None,
    prior=None,
    metadata=None,
) -> object:
    default = ((_aid(0, 0), 1.0),)
    return samples.Model1SampleData(
        game_id="g",
        turn_index=turn,
        current_player="player0" if turn % 2 == 0 else "player1",
        phase="Placement",
        center=(0, 0),
        stones=(),
        legal_action_ids=(_aid(0, 0), _aid(1, 0)),
        policy=tuple(policy or default),
        root_prior_policy=tuple(prior or default),
        value=0.0,
        moves_left=float(moves_left),
        metadata=dict(metadata or {}),
    )


def _weight_of_single(row, **kwargs) -> float:
    # One-row games have KL == 0 against themselves only if policy == prior; the
    # callers below always pass policy == prior so the base weight is exactly 1.0
    # and the returned frequency_weight_mean IS the decayed weight.
    _materialized, stats = replay.materialize_policy_surprise_rows([row], seed=3, **kwargs)
    return stats["frequency_weight_mean"]


# --------------------------- C1: length decay --------------------------- #


def test_length_decay_knee_zero_is_bit_identical_regression() -> None:
    rows = [
        _row(0, moves_left=200.0),
        _row(1, moves_left=199.0, policy=((_aid(1, 0), 1.0),)),  # KL > 0 row
        _row(2, moves_left=198.0),
        _row(3, moves_left=0.0),
    ]
    baseline_rows, baseline_stats = replay.materialize_policy_surprise_rows(rows, seed=123)
    decayed_rows, decayed_stats = replay.materialize_policy_surprise_rows(
        rows,
        seed=123,
        game_decisions=400,
        moves_left_knee=0.0,
        moves_left_halflife=40.0,
        game_length_knee=0.0,
        game_length_halflife=60.0,
    )
    assert baseline_stats == decayed_stats
    assert [r.frequency_weight for r in baseline_rows] == [r.frequency_weight for r in decayed_rows]
    assert [r.game_id for r in baseline_rows] == [r.game_id for r in decayed_rows]
    assert [r.turn_index for r in baseline_rows] == [r.turn_index for r in decayed_rows]


def test_length_decay_moves_left_exactness() -> None:
    knobs = dict(moves_left_knee=110.0, moves_left_halflife=40.0)
    # ml=150 -> 0.5 ** ((150-110)/40) = 0.5; ml=190 -> 0.25.
    assert _weight_of_single(_row(0, moves_left=150.0), **knobs) == pytest.approx(0.5)
    assert _weight_of_single(_row(0, moves_left=190.0), **knobs) == pytest.approx(0.25)
    # At or below the knee: untouched.
    assert _weight_of_single(_row(0, moves_left=110.0), **knobs) == pytest.approx(1.0)
    assert _weight_of_single(_row(0, moves_left=0.0), **knobs) == pytest.approx(1.0)


def test_length_decay_game_length_exactness_and_stacking() -> None:
    # game_decisions=170, knee=110, halflife=60 -> 0.5 (row's ml below its knee).
    assert _weight_of_single(
        _row(0, moves_left=10.0),
        game_decisions=170,
        moves_left_knee=110.0,
        moves_left_halflife=40.0,
        game_length_knee=110.0,
        game_length_halflife=60.0,
    ) == pytest.approx(0.5)
    # Both terms stack multiplicatively: ml=150 (x0.5) and gd=170 (x0.5) -> 0.25.
    assert _weight_of_single(
        _row(0, moves_left=150.0),
        game_decisions=170,
        moves_left_knee=110.0,
        moves_left_halflife=40.0,
        game_length_knee=110.0,
        game_length_halflife=60.0,
    ) == pytest.approx(0.25)
    # game_decisions absent or at the knee: no game-length decay.
    assert _weight_of_single(
        _row(0, moves_left=10.0), game_length_knee=110.0, game_length_halflife=60.0
    ) == pytest.approx(1.0)
    assert _weight_of_single(
        _row(0, moves_left=10.0),
        game_decisions=110,
        game_length_knee=110.0,
        game_length_halflife=60.0,
    ) == pytest.approx(1.0)


def test_length_decay_truncated_row_falls_back_to_game_decisions() -> None:
    # Truncated rows carry the moves_left = -1 mask; with game_decisions given,
    # ml = game_decisions - turn_index - 1 = 200 - 9 - 1 = 190 -> x0.25.
    assert _weight_of_single(
        _row(9, moves_left=-1.0),
        game_decisions=200,
        moves_left_knee=110.0,
        moves_left_halflife=40.0,
    ) == pytest.approx(0.25)
    # Without game_decisions the masked ml stays negative -> no moves-left decay.
    assert _weight_of_single(
        _row(9, moves_left=-1.0), moves_left_knee=110.0, moves_left_halflife=40.0
    ) == pytest.approx(1.0)


def test_length_decay_respects_max_weight_cap() -> None:
    surprising = _row(0, moves_left=150.0, policy=((_aid(1, 0), 1.0),))
    flat = _row(1, moves_left=150.0)
    materialized, stats = replay.materialize_policy_surprise_rows(
        [flat, surprising],
        seed=11,
        max_weight=1.5,
        moves_left_knee=110.0,
        moves_left_halflife=40.0,
    )
    # Base weights clamp to [0.5, 1.5] (the existing policy-surprise arithmetic),
    # decay halves both AFTER the clamp -> [0.25, 0.75]; the cap holds.
    assert stats["frequency_weight_mean"] == pytest.approx(0.5)
    assert all(row.frequency_weight <= 1.5 for row in materialized)
    assert {round(row.frequency_weight, 6) for row in materialized} <= {0.25, 0.75}


def test_length_decay_sub_unit_weights_drop_rows_probabilistically() -> None:
    # Weights < 1 act as Bernoulli keeps in the duplication loop: ~half of 200
    # rows decayed to weight 0.5 survive (deterministic for a fixed seed).
    rows = [_row(turn, moves_left=150.0) for turn in range(200)]
    materialized, stats = replay.materialize_policy_surprise_rows(
        rows, seed=42, moves_left_knee=110.0, moves_left_halflife=40.0
    )
    assert stats["raw_rows"] == 200.0
    assert stats["frequency_weight_mean"] == pytest.approx(0.5)
    assert 0 < len(materialized) < 200
    assert 60 <= len(materialized) <= 140
    assert stats["effective_rows"] == float(len(materialized))


# --------------------------- C2: truncation-poison quarantine --------------------------- #


def _finalized_game(n: int = 6, *, fast_turns: tuple[int, ...] = ()) -> list:
    return [
        _row(turn, metadata={"pcr_full": turn not in fast_turns})
        for turn in range(n)
    ]


def test_truncated_game_writes_zero_rows_with_flag_on() -> None:
    finalized = _finalized_game(6, fast_turns=(2,))
    to_write, fast_excluded, dropped = selfplay._rows_to_write(
        finalized, truncated=True, drop_truncated_rows=True
    )
    assert to_write == []
    assert fast_excluded == 1
    assert dropped == 5


def test_truncated_game_unchanged_with_flag_off() -> None:
    finalized = _finalized_game(6, fast_turns=(2,))
    to_write, fast_excluded, dropped = selfplay._rows_to_write(
        finalized, truncated=True, drop_truncated_rows=False
    )
    assert [row.turn_index for row in to_write] == [0, 1, 3, 4, 5]
    assert fast_excluded == 1
    assert dropped == 0


def test_completed_game_unaffected_by_quarantine_flag() -> None:
    finalized = _finalized_game(4)
    for flag in (False, True):
        to_write, fast_excluded, dropped = selfplay._rows_to_write(
            finalized, truncated=False, drop_truncated_rows=flag
        )
        assert [row.turn_index for row in to_write] == [0, 1, 2, 3]
        assert fast_excluded == 0
        assert dropped == 0


# --------------------------- config keys --------------------------- #


def test_config_parses_new_keys_from_toml() -> None:
    raw = tomllib.loads(
        "\n".join(
            [
                "[selfplay]",
                "frozen_win_override = true",
                "",
                "[samples]",
                "drop_truncated_rows = true",
                "length_decay_moves_left_knee = 110.0",
                "length_decay_moves_left_halflife = 35.0",
                "length_decay_game_length_knee = 120.0",
                "length_decay_game_length_halflife = 55.0",
            ]
        )
    )
    parsed = config_module.parse_model1_config(raw)
    assert parsed.selfplay.frozen_win_override is True
    assert parsed.samples.drop_truncated_rows is True
    assert parsed.samples.length_decay_moves_left_knee == pytest.approx(110.0)
    assert parsed.samples.length_decay_moves_left_halflife == pytest.approx(35.0)
    assert parsed.samples.length_decay_game_length_knee == pytest.approx(120.0)
    assert parsed.samples.length_decay_game_length_halflife == pytest.approx(55.0)


def test_config_defaults_keep_new_levers_dormant() -> None:
    parsed = config_module.parse_model1_config({})
    assert parsed.selfplay.frozen_win_override is False
    assert parsed.samples.drop_truncated_rows is False
    assert parsed.samples.length_decay_moves_left_knee == 0.0
    assert parsed.samples.length_decay_moves_left_halflife == pytest.approx(40.0)
    assert parsed.samples.length_decay_game_length_knee == 0.0
    assert parsed.samples.length_decay_game_length_halflife == pytest.approx(60.0)
    # Knee 0 disables the decay entirely (covered bit-exactly by
    # test_length_decay_knee_zero_is_bit_identical_regression).
