from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_sample_from_history_delegates_live_facts_to_rust(monkeypatch: Any) -> None:
    samples = importlib.import_module("hexo_models.dense_cnn.samples")
    calls: dict[str, Any] = {}

    class FakeDenseCnnRust:
        def model1_sample_from_history(
            self,
            history_row: tuple[int, ...],
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
                    "history_row": history_row,
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
                "legal_action_ids": history_row,
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

    sample = samples.sample_from_history(
        [10, 20],
        game_id="game",
        turn_index=3,
        policy={99: 1.0},
        value=0.25,
        metadata={"sample_source": "mcts"},
    )

    assert calls["history_row"] == (10, 20)
    assert calls["game_id"] == "game"
    assert calls["turn_index"] == 3
    assert sample.policy == ((99, 1.0),)
    assert sample.metadata["sample_source"] == "mcts"


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
                        "opp_policy_source": "uniform_legal_fallback",
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
