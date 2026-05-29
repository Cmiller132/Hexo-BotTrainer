from __future__ import annotations

import importlib
import inspect
import struct
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _state_facts(game_id: str, turn_index: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "turn_index": turn_index,
        "current_player": "player0",
        "phase": "Opening",
        "center": (0, 0),
        "stones": (),
        "legal_action_ids": (10, 20),
        "placement_history": (),
        "first_stone": None,
        "own_hot": (),
        "opponent_hot": (),
        "opponent_last_turn": (),
        "metadata": metadata,
    }


def test_sample_from_state_attaches_search_targets_to_rust_facts(monkeypatch: Any) -> None:
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    calls: dict[str, Any] = {}
    state = object()

    class FakeDenseCnnRust:
        def model1_sample_from_state(
            self,
            live_state: object,
            game_id: str,
            turn_index: int,
            metadata: dict[str, Any],
        ) -> dict[str, Any]:
            # Rust receives only the live state + identity + metadata. It must not
            # be handed policy/value/opp_policy/lookahead any more.
            calls.update(
                {"state": live_state, "game_id": game_id, "turn_index": turn_index, "metadata": metadata}
            )
            return _state_facts(game_id, turn_index, metadata)

    monkeypatch.setattr(samples.rust_bridge, "_dense_cnn_module", lambda: FakeDenseCnnRust())

    sample = samples.sample_from_state(
        state,
        game_id="game",
        turn_index=3,
        policy={99: 4.0},
        root_prior_policy={99: 1.0},
        metadata={"sample_source": "mcts"},
    )

    assert calls["state"] is state
    assert calls["turn_index"] == 3
    assert calls["metadata"]["sample_source"] == "mcts"
    assert calls["metadata"]["target_schema_version"] == samples.CURRENT_TARGET_SCHEMA_VERSION
    # Python attaches the search policy and normalizes the root prior.
    assert sample.policy == ((99, 4.0),)
    assert sample.root_prior_policy == ((99, 1.0),)
    assert sample.value == 0.0  # outcome target is filled at finalization, not here


def test_sample_from_state_requires_root_prior(monkeypatch: Any) -> None:
    samples = importlib.import_module("hexo_models.dense_cnn.samples")

    class FakeDenseCnnRust:
        def model1_sample_from_state(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
            raise AssertionError("rust must not be called without root_prior_policy")

    monkeypatch.setattr(samples.rust_bridge, "_dense_cnn_module", lambda: FakeDenseCnnRust())

    with pytest.raises(ValueError, match="root_prior_policy"):
        samples.sample_from_state(object(), game_id="game", turn_index=0, policy={1: 1.0})


def test_mcts_session_search_forwards_slim_signature(monkeypatch: Any) -> None:
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    state = object()
    evaluator = object()
    calls: dict[str, Any] = {}

    class FakeSession:
        def search(
            self,
            game_keys: tuple[int, ...],
            states: tuple[object, ...],
            visits: int,
            c_puct: float,
            temperature: float,
            seed: int,
            callback: object,
            virtual_batch_size: int | None,
            active_root_limit: int | None,
            root_dirichlet_total_alpha: float | None,
            root_dirichlet_noise_fraction: float | None,
            root_policy_temperature: float | None,
            fpu_reduction: float | None,
            virtual_loss: float | None,
            widening_policy_mass: float | None,
            widening_max_children: int | None,
            widening_min_children: int | None,
        ) -> tuple[dict[str, Any], ...]:
            calls["args"] = {
                "game_keys": game_keys,
                "states": states,
                "visits": visits,
                "c_puct": c_puct,
                "temperature": temperature,
                "seed": seed,
                "callback": callback,
                "virtual_batch_size": virtual_batch_size,
                "active_root_limit": active_root_limit,
                "root_dirichlet_total_alpha": root_dirichlet_total_alpha,
                "root_dirichlet_noise_fraction": root_dirichlet_noise_fraction,
                "root_policy_temperature": root_policy_temperature,
                "fpu_reduction": fpu_reduction,
                "virtual_loss": virtual_loss,
                "widening_policy_mass": widening_policy_mass,
                "widening_max_children": widening_max_children,
                "widening_min_children": widening_min_children,
            }
            return (
                {
                    "action_id": 7,
                    "visit_policy_action_ids_bytes": struct.pack("I", 7),
                    "visit_policy_weights_bytes": struct.pack("f", 1.0),
                    "visit_policy_count": 1,
                    "root_prior_policy_action_ids_bytes": struct.pack("I", 7),
                    "root_prior_policy_weights_bytes": struct.pack("f", 1.0),
                    "root_prior_policy_count": 1,
                    "root_value": 0.0,
                    "visits": visits,
                },
            )

    class FakeDenseCnnRust:
        def Model1MctsSession(self, max_states: int | None) -> object:
            return FakeSession()

    monkeypatch.setattr(rust_bridge, "_dense_cnn_module", lambda: FakeDenseCnnRust())
    session = rust_bridge.model1_new_mcts_session(max_states=99)
    payloads = rust_bridge.model1_mcts_session_search(
        session,
        [123],
        [state],
        visits=5,
        c_puct=2.0,
        temperature=0.25,
        seed=13,
        evaluator=evaluator,
        virtual_batch_size=3,
        root_dirichlet_total_alpha=10.83,
        root_policy_temperature=1.1,
    )

    assert calls["args"] == {
        "game_keys": (123,),
        "states": (state,),
        "visits": 5,
        "c_puct": 2.0,
        "temperature": 0.25,
        "seed": 13,
        "callback": evaluator,
        "virtual_batch_size": 3,
        "active_root_limit": None,
        "root_dirichlet_total_alpha": 10.83,
        "root_dirichlet_noise_fraction": None,
        "root_policy_temperature": 1.1,
        "fpu_reduction": None,
        "virtual_loss": None,
        "widening_policy_mass": None,
        "widening_max_children": None,
        "widening_min_children": None,
    }
    assert payloads[0]["action_id"] == 7


def test_dense_cnn_boundary_has_no_legacy_api() -> None:
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    mcts = importlib.import_module("hexo_models.dense_cnn.mcts")

    assert not hasattr(samples, "CompressedSample")
    assert not hasattr(samples, "encode_compact_sample")
    assert not hasattr(rust_bridge, "model1_finalize_game_samples")
    assert not hasattr(mcts, "run_mcts")
    source = "\n".join(inspect.getsource(module) for module in (rust_bridge, samples, mcts))
    for banned in ("to_python_state", "progressive_widening", "hidden_prior_mass", "lookahead"):
        assert banned not in source, banned


def test_finalize_game_samples_assigns_python_outcome_targets() -> None:
    samples = importlib.import_module("hexo_models.dense_cnn.samples")

    def mk(player: str, turn: int, policy: tuple[tuple[int, float], ...]) -> Any:
        return samples.Model1SampleData(
            game_id="g",
            turn_index=turn,
            current_player=player,
            phase="Opening",
            center=(0, 0),
            stones=(),
            legal_action_ids=(1, 2),
            policy=policy,
        )

    pending = [
        ("player0", mk("player0", 0, ((1, 1.0),)), 0.2),
        ("player1", mk("player1", 1, ((2, 1.0),)), -0.3),
        ("player0", mk("player0", 2, ((1, 1.0),)), 0.5),
    ]
    finalized = samples.finalize_game_samples(pending, winner="player0", horizons=(1,))

    assert [f.value for f in finalized] == [1.0, -1.0, 1.0]
    # Opponent-policy target is the next opposing decision's policy.
    assert finalized[0].opp_policy == ((2, 1.0),)
    assert finalized[0].metadata["opp_policy_source"] == "future_opponent_mcts"
    assert finalized[2].opp_policy == ()
    assert finalized[2].metadata["opp_policy_source"] == "none"
    # Short-term value is a perspective-corrected EMA of future root values.
    # decision 0 (player0): future = [+0.3 (player1 root -0.3 flipped), +0.5]; m=1 decay 0.5.
    assert finalized[0].short_term_value[0][0] == 1
    assert finalized[0].short_term_value[0][1] == pytest.approx((0.3 + 0.5 * 0.5) / 1.5)
    assert finalized[2].short_term_value == ()


def test_finalize_truncated_game_is_a_draw() -> None:
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    sample = samples.Model1SampleData(
        game_id="g",
        turn_index=0,
        current_player="player0",
        phase="Opening",
        center=(0, 0),
        stones=(),
        legal_action_ids=(1,),
        policy=((1, 1.0),),
    )
    finalized = samples.finalize_game_samples([("player0", sample, 0.0)], winner=None, horizons=(1, 4), truncated=True)
    assert finalized[0].value == 0.0
    assert finalized[0].metadata["value_target_reason"] == "max_actions_draw"
    assert finalized[0].short_term_value == ()
