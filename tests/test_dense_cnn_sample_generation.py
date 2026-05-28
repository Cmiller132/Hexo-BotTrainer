from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_sample_from_state_delegates_live_facts_to_rust_without_python_mirror(monkeypatch: Any) -> None:
    engine = importlib.import_module("hexo_engine")
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    calls: dict[str, Any] = {}
    state = object()

    class FakeDenseCnnRust:
        def model1_sample_from_state(
            self,
            live_state: object,
            game_id: str,
            turn_index: int,
            policy: dict[int, float],
            value: float,
            opp_policy: tuple[tuple[int, float], ...],
            lookahead: tuple[tuple[int, float], ...],
            metadata: dict[str, Any],
        ) -> dict[str, Any]:
            calls.update(
                {
                    "state": live_state,
                    "game_id": game_id,
                    "turn_index": turn_index,
                    "policy": policy,
                    "value": value,
                    "metadata": metadata,
                }
            )
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
                "policy": tuple(policy.items()),
                "opp_policy": opp_policy,
                "value": value,
                "lookahead": lookahead,
                "metadata": metadata,
            }

    monkeypatch.setattr(samples.rust_bridge, "_dense_cnn_module", lambda: FakeDenseCnnRust())
    monkeypatch.setattr(
        engine,
        "to_python_state",
        lambda _state: (_ for _ in ()).throw(AssertionError("dense-cnn must pass live states directly")),
    )

    sample = samples.sample_from_state(
        state,
        game_id="game",
        turn_index=3,
        policy={99: 1.0},
        value=0.25,
        metadata={"sample_source": "mcts"},
    )

    assert calls["state"] is state
    assert calls["game_id"] == "game"
    assert calls["turn_index"] == 3
    assert sample.policy == ((99, 1.0),)
    assert sample.metadata["sample_source"] == "mcts"


def test_rust_bridge_forwards_live_states_without_history_conversion(monkeypatch: Any) -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    state_a = object()
    state_b = object()
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
            progressive_widening_initial_actions: int | None,
            progressive_widening_child_initial_actions: int | None,
            progressive_widening_growth_interval: float | None,
            progressive_widening_growth_base: float | None,
            progressive_widening_candidate_actions: int | None,
            active_root_limit: int | None,
            root_dirichlet_alpha: float | None,
            root_dirichlet_noise_fraction: float | None,
            hidden_prior_mass: float | None,
            fpu_reduction: float | None,
            virtual_loss: float | None,
        ) -> tuple[dict[str, Any], ...]:
            calls["mcts"] = {
                "session": self,
                "game_keys": game_keys,
                "states": states,
                "visits": visits,
                "c_puct": c_puct,
                "temperature": temperature,
                "seed": seed,
                "callback": callback,
                "virtual_batch_size": virtual_batch_size,
                "progressive_widening_initial_actions": progressive_widening_initial_actions,
                "progressive_widening_child_initial_actions": progressive_widening_child_initial_actions,
                "progressive_widening_candidate_actions": progressive_widening_candidate_actions,
                "progressive_widening_growth_interval": progressive_widening_growth_interval,
                "progressive_widening_growth_base": progressive_widening_growth_base,
                "active_root_limit": active_root_limit,
                "root_dirichlet_alpha": root_dirichlet_alpha,
                "root_dirichlet_noise_fraction": root_dirichlet_noise_fraction,
                "hidden_prior_mass": hidden_prior_mass,
                "fpu_reduction": fpu_reduction,
                "virtual_loss": virtual_loss,
            }
            return ({"action_id": 7, "visit_policy": ((7, 1.0),), "root_value": 0.0, "visits": visits},)

    class FakeDenseCnnRust:
        def model1_batch_inputs(self, states: tuple[object, ...]) -> dict[str, Any]:
            calls["batch_states"] = states
            return {"ok": True}

        def Model1MctsSession(self, max_states: int | None) -> object:
            calls["session_max_states"] = max_states
            session = FakeSession()
            calls["session"] = session
            return session

    monkeypatch.setattr(rust_bridge, "_dense_cnn_module", lambda: FakeDenseCnnRust())
    monkeypatch.setattr(
        engine,
        "to_python_state",
        lambda _state: (_ for _ in ()).throw(AssertionError("dense-cnn must not mirror states through Python")),
    )

    assert rust_bridge.model1_batch_inputs([state_a, state_b]) == {"ok": True}
    session = rust_bridge.model1_new_mcts_session(max_states=99)
    payloads = rust_bridge.model1_mcts_session_search(
        session,
        [123],
        [state_a],
        visits=5,
        c_puct=2.0,
        temperature=0.25,
        seed=13,
        evaluator=evaluator,
        virtual_batch_size=3,
    )

    assert calls["batch_states"] == (state_a, state_b)
    assert calls["session_max_states"] == 99
    assert calls["mcts"] == {
        "session": session,
        "game_keys": (123,),
        "states": (state_a,),
        "visits": 5,
        "c_puct": 2.0,
        "temperature": 0.25,
        "seed": 13,
        "callback": evaluator,
        "virtual_batch_size": 3,
        "progressive_widening_initial_actions": None,
        "progressive_widening_child_initial_actions": None,
        "progressive_widening_candidate_actions": None,
        "progressive_widening_growth_interval": None,
        "progressive_widening_growth_base": None,
        "active_root_limit": None,
        "root_dirichlet_alpha": None,
        "root_dirichlet_noise_fraction": None,
        "hidden_prior_mass": None,
        "fpu_reduction": None,
        "virtual_loss": None,
    }
    assert payloads[0]["action_id"] == 7


def test_dense_cnn_python_boundary_has_no_history_api() -> None:
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    mcts = importlib.import_module("hexo_models.dense_cnn.mcts")
    inference = importlib.import_module("hexo_models.dense_cnn.inference")
    selfplay_source = (
        ROOT
        / "packages"
        / "hexo_models"
        / "dense_cnn"
        / "python"
        / "hexo_models"
        / "dense_cnn"
        / "selfplay.py"
    ).read_text()

    assert not hasattr(rust_bridge, "history_rows_from_states")
    assert not hasattr(samples, "sample_from_history")
    source = "\n".join(
        inspect.getsource(module)
        for module in (rust_bridge, samples, mcts, inference)
    )
    source = f"{source}\n{selfplay_source}"
    assert "to_python_state" not in source
    assert "model1_sample_from_history" not in source
    assert "history_rows_from_states" not in source
    assert "sample_from_history" not in source


def test_finalize_game_samples_delegates_outcomes_to_rust(monkeypatch: Any) -> None:
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    captured: dict[str, Any] = {}
    pending_sample = samples.Model1SampleData(
        game_id="game",
        turn_index=0,
        current_player="player0",
        phase="Opening",
        center=(0, 0),
        stones=(),
        legal_action_ids=(1,),
        policy=((1, 1.0),),
        metadata={"sample_source": "mcts"},
    )

    class FakeDenseCnnRust:
        def model1_finalize_game_samples(
            self,
            pending: tuple[tuple[str, dict[str, Any], float], ...],
            winner: str | None,
            horizons: tuple[int, ...],
            truncated: bool,
        ) -> list[dict[str, Any]]:
            captured.update(
                {
                    "pending": pending,
                    "winner": winner,
                    "horizons": horizons,
                    "truncated": truncated,
                }
            )
            payload = dict(pending[0][1])
            payload.update(
                {
                    "value": 1.0,
                    "opp_policy": (),
                    "lookahead": ((1, 1.0),),
                    "metadata": {
                        **dict(payload["metadata"]),
                        "opp_policy_source": "none",
                        "truncated": True,
                    },
                }
            )
            return [payload]

    monkeypatch.setattr(samples.rust_bridge, "_dense_cnn_module", lambda: FakeDenseCnnRust())

    finalized = samples.finalize_game_samples(
        [("player0", pending_sample, 0.5)],
        winner="player0",
        horizons=(1,),
        truncated=True,
    )

    assert captured["pending"][0][0] == "player0"
    assert captured["pending"][0][1]["game_id"] == "game"
    assert captured["winner"] == "player0"
    assert captured["horizons"] == (1,)
    assert captured["truncated"] is True
    assert finalized[0].value == 1.0
    assert finalized[0].lookahead == ((1, 1.0),)
    assert finalized[0].metadata["truncated"] is True
