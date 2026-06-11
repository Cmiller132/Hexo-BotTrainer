"""main_2 self-play data levers for dense_cnn_restnet (2026-06-10).

Covers: KataGo Playout Cap Randomization (the per-move full/fast coin and the
write-time fast-row exclusion semantics), KataGo policy-initialized openings
(per-game ply draw + raw-prior sampling), the KataGo root-policy temperature
early ramp, the soft-Z value-target blend, and the new config keys.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
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
samples = importlib.import_module("dense_cnn_restnet.samples")
selfplay = importlib.import_module("dense_cnn_restnet.selfplay")


def _aid(q: int, r: int) -> int:
    return int(d6.pack_coord_id(d6.Axial(q, r)))


def _mk_sample(turn: int, *, metadata=None):
    legal = tuple((q, r) for q in range(-2, 3) for r in range(-2, 3) if abs(q + r) <= 2)
    return samples.Model1SampleData(
        game_id="g",
        turn_index=turn,
        current_player="player0" if turn % 2 == 0 else "player1",
        phase="Placement",
        center=(0, 0),
        stones=(),
        legal_action_ids=tuple(_aid(q, r) for q, r in legal),
        policy=tuple((_aid(q, r), 1.0) for q, r in legal[:2]),
        root_prior_policy=tuple((_aid(q, r), 1.0) for q, r in legal),
        value=1.0,
        short_term_value=(),
        metadata=dict(metadata or {}),
    )


# --- config keys ----------------------------------------------------------------


def test_config_parses_new_selfplay_and_samples_keys():
    parsed = config_module.parse_model1_config(
        {
            "selfplay": {
                "pcr_enabled": True,
                "pcr_full_proportion": 0.25,
                "pcr_fast_visits": 128,
                "policy_init_fraction": 0.25,
                "policy_init_avg_plies": 4.0,
                "policy_init_max_plies": 8,
                "policy_init_temperature": 1.4,
                "root_policy_temperature_early": 1.25,
                "root_policy_temperature_halflife": 16.0,
            },
            "samples": {"soft_z_lambda": 0.1},
        }
    )
    assert parsed.selfplay.pcr_enabled is True
    assert parsed.selfplay.pcr_full_proportion == pytest.approx(0.25)
    assert parsed.selfplay.pcr_fast_visits == 128
    assert parsed.selfplay.policy_init_fraction == pytest.approx(0.25)
    assert parsed.selfplay.policy_init_avg_plies == pytest.approx(4.0)
    assert parsed.selfplay.policy_init_max_plies == 8
    assert parsed.selfplay.policy_init_temperature == pytest.approx(1.4)
    assert parsed.selfplay.root_policy_temperature_early == pytest.approx(1.25)
    assert parsed.selfplay.root_policy_temperature_halflife == pytest.approx(16.0)
    assert parsed.samples.soft_z_lambda == pytest.approx(0.1)


def test_config_parses_adaptive_virtual_batch():
    parsed = config_module.parse_model1_config(
        {"selfplay": {"adaptive_virtual_batch": True}}
    )
    assert parsed.selfplay.adaptive_virtual_batch is True
    assert config_module.parse_model1_config({}).selfplay.adaptive_virtual_batch is False


def test_config_defaults_disable_new_levers():
    parsed = config_module.parse_model1_config({})
    assert parsed.selfplay.pcr_enabled is False
    assert parsed.selfplay.policy_init_fraction == 0.0
    assert parsed.selfplay.root_policy_temperature_early == 0.0
    assert parsed.samples.soft_z_lambda == 0.0


# --- PCR coin -------------------------------------------------------------------


def test_pcr_coin_bounds_and_determinism():
    assert selfplay._pcr_is_full(1, 2, 3, 4, 1.0) is True
    assert selfplay._pcr_is_full(1, 2, 3, 4, 0.0) is False
    draws = [selfplay._pcr_is_full(7, 11, 13, move, 0.25) for move in range(4096)]
    again = [selfplay._pcr_is_full(7, 11, 13, move, 0.25) for move in range(4096)]
    assert draws == again  # deterministic
    fraction = sum(draws) / len(draws)
    assert 0.20 < fraction < 0.30  # ~Bernoulli(0.25)
    # Distinct coordinates decorrelate the schedule.
    other_game = [selfplay._pcr_is_full(7, 11, 14, move, 0.25) for move in range(4096)]
    assert other_game != draws


# --- policy-initialized openings --------------------------------------------------


def test_policy_init_plies_disabled_and_capped():
    assert selfplay._policy_init_plies(1, 2, 3, fraction=0.0, avg_plies=4.0, max_plies=8) == 0
    assert selfplay._policy_init_plies(1, 2, 3, fraction=1.0, avg_plies=4.0, max_plies=0) == 0
    counts = [
        selfplay._policy_init_plies(9, 4, game, fraction=1.0, avg_plies=4.0, max_plies=8)
        for game in range(4096)
    ]
    assert all(0 <= count <= 8 for count in counts)
    # Truncated exponential with mean 4 capped at 8: mean lands well inside (1, 8).
    mean = sum(counts) / len(counts)
    assert 2.0 < mean < 5.0
    again = [
        selfplay._policy_init_plies(9, 4, game, fraction=1.0, avg_plies=4.0, max_plies=8)
        for game in range(4096)
    ]
    assert counts == again  # deterministic


def test_policy_init_fraction_selects_subset():
    selected = [
        selfplay._policy_init_plies(5, 6, game, fraction=0.25, avg_plies=64.0, max_plies=64) > 0
        for game in range(4096)
    ]
    fraction = sum(selected) / len(selected)
    # avg 64 >> 0 so a selected game almost surely draws >= 1 ply; the observed
    # fraction tracks the selection coin.
    assert 0.20 < fraction < 0.30


def test_policy_init_sampling_temperature_and_determinism():
    pairs = [(11, 0.90), (22, 0.09), (33, 0.01)]
    # Near-zero temperature sharpens to the argmax.
    sharp = [
        selfplay._sample_policy_init_action(
            pairs, 1.0e-6, base_seed=1, epoch=2, game_key=game, move_index=1
        )
        for game in range(64)
    ]
    assert set(sharp) == {11}
    # High temperature spreads mass onto the tail.
    wide = [
        selfplay._sample_policy_init_action(
            pairs, 1.0e6, base_seed=1, epoch=2, game_key=game, move_index=1
        )
        for game in range(512)
    ]
    assert {22, 33} & set(wide)
    one = selfplay._sample_policy_init_action(
        pairs, 1.4, base_seed=1, epoch=2, game_key=3, move_index=4
    )
    two = selfplay._sample_policy_init_action(
        pairs, 1.4, base_seed=1, epoch=2, game_key=3, move_index=4
    )
    assert one == two  # deterministic
    with pytest.raises(RuntimeError):
        selfplay._sample_policy_init_action(
            [], 1.0, base_seed=1, epoch=2, game_key=3, move_index=4
        )


# --- root-policy temperature early ramp -------------------------------------------


def test_root_policy_temperature_ramp():
    kwargs = dict(base=1.1, early=1.25, halflife_plies=16.0)
    assert selfplay._root_policy_temperature(0, **kwargs) == pytest.approx(1.25)
    assert selfplay._root_policy_temperature(16, **kwargs) == pytest.approx(1.1 + 0.15 * 0.5)
    assert selfplay._root_policy_temperature(160, **kwargs) == pytest.approx(1.1, abs=1e-3)
    # Disabled ramp returns the base everywhere.
    assert selfplay._root_policy_temperature(0, base=1.1, early=0.0, halflife_plies=16.0) == 1.1


# --- soft-Z blend + fast-row semantics ---------------------------------------------


def test_finalize_soft_z_blends_outcome_with_root_value():
    pending = [("player0", _mk_sample(0), 0.5), ("player1", _mk_sample(1), -0.25)]
    done = samples.finalize_game_samples(pending, winner="player0", horizons=(), soft_z_lambda=0.1)
    # value = 0.9 * z + 0.1 * root_value, root_value already side-to-move.
    assert done[0].value == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)
    assert done[1].value == pytest.approx(0.9 * -1.0 + 0.1 * -0.25)
    hard = samples.finalize_game_samples(pending, winner="player0", horizons=())
    assert hard[0].value == pytest.approx(1.0)


def test_finalize_masks_opp_policy_from_fast_rows():
    fast = _mk_sample(1, metadata={"pcr_full": False})
    pending = [
        ("player0", _mk_sample(0), 0.0),
        ("player1", fast, 0.0),
        ("player0", _mk_sample(2), 0.0),
    ]
    done = samples.finalize_game_samples(
        pending, winner="player0", horizons=(), mask_opp_from_fast=True
    )
    assert done[0].opp_policy == ()
    assert done[0].metadata["opp_policy_source"] == "fast_unrecorded_masked"
    # Without the mask the fast row's policy is still used (non-PCR behavior).
    plain = samples.finalize_game_samples(pending, winner="player0", horizons=())
    assert plain[0].opp_policy == tuple(fast.policy)


def test_fast_rows_are_excluded_by_pcr_full_filter():
    finalized = [
        replace(_mk_sample(0), metadata={"pcr_full": True}),
        replace(_mk_sample(1), metadata={"pcr_full": False}),
        replace(_mk_sample(2), metadata={"pcr_full": False, "policy_init": True}),
        _mk_sample(3),  # no flag -> treated as full (non-PCR back-compat)
    ]
    to_write = [s for s in finalized if s.metadata.get("pcr_full", True)]
    assert [s.turn_index for s in to_write] == [0, 3]
