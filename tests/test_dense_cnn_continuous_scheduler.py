from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    _src = _ROOT / "packages" / _pkg / "python"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402

from dense_cnn_restnet.config import parse_model1_config  # noqa: E402
from dense_cnn_restnet.constants import BOARD_AREA, VALUE_BINS  # noqa: E402
from dense_cnn_restnet.inference import DenseCNNInference  # noqa: E402
from dense_cnn_restnet.mcts import _result_from_payload, new_mcts_session  # noqa: E402
from dense_cnn_restnet.performance import _summarize_mcts_diagnostic_batches  # noqa: E402
from dense_cnn_restnet import rust_bridge  # noqa: E402
from dense_cnn_restnet import selfplay as selfplay_module  # noqa: E402


def test_config_parses_continuous_scheduler_keys() -> None:
    cfg = parse_model1_config(
        {
            "selfplay": {
                "scheduler": "continuous",
                "scheduler_flush_target": 256,
            }
        }
    )
    assert cfg.selfplay.scheduler == "continuous"
    assert cfg.selfplay.scheduler_flush_target == 256


def test_config_rejects_unknown_scheduler() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        parse_model1_config({"selfplay": {"scheduler": "other"}})


def test_mcts_run_continuous_delegates_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    native_session = object()
    calls: list[Mapping[str, Any]] = []

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_run_continuous(
        session: object,
        game_keys: Sequence[int],
        states: Sequence[object],
        **kwargs: object,
    ) -> Mapping[str, Any]:
        calls.append({"session": session, "game_keys": tuple(game_keys), "states": tuple(states), **kwargs})
        return {
            "flush_count": 1,
            "flushed_states": 4,
            "mean_flush_states": 4.0,
            "no_progress_flushes": 0,
            "flush_size_histogram": {4: 1},
            "on_move_seconds": 0.0,
        }

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", lambda **_kwargs: native_session)
    monkeypatch.setattr(rust_bridge, "model1_mcts_session_run_continuous", fake_run_continuous)

    on_move = object()
    states = [object(), object()]
    session = new_mcts_session(max_states=10)
    summary = session.run_continuous(
        [7, 8],
        states,
        FakeInference(),
        on_move,
        visits=32,
        c_puct=1.7,
        base_seed=123,
        virtual_batch_size=4,
        flush_target=16,
        active_root_limit=64,
        temperature_by_ply=[1.0, 0.5],
        forced_playout_k=2.0,
    )

    assert summary["flush_count"] == 1
    assert calls == [
        {
            "session": native_session,
            "game_keys": (7, 8),
            "states": tuple(states),
            "evaluator": FakeInference.evaluate_model1_payload,
            "on_move": on_move,
            "visits": 32,
            "c_puct": 1.7,
            "base_seed": 123,
            "virtual_batch_size": 4,
            "flush_target": 16,
            "active_root_limit": 64,
            "temperature_by_ply": [1.0, 0.5],
            "root_dirichlet_total_alpha": None,
            "root_dirichlet_noise_fraction": None,
            "root_policy_temperature": None,
            "fpu_reduction": None,
            "virtual_loss": None,
            "widening_policy_mass": None,
            "widening_max_children": None,
            "widening_min_children": None,
            "forced_playout_k": 2.0,
        }
    ]


def test_selfplay_dispatches_to_continuous(monkeypatch: pytest.MonkeyPatch) -> None:
    class Trainer:
        config = parse_model1_config({"selfplay": {"scheduler": "continuous"}})

    class Components:
        class model:
            trainer = Trainer()

    called: list[str] = []

    def fake_continuous(**_kwargs: object) -> Mapping[str, Any]:
        called.append("continuous")
        return {"status": "completed", "scheduler": "continuous"}

    monkeypatch.setattr(selfplay_module, "_generate_selfplay_epoch_continuous", fake_continuous)

    result = selfplay_module.generate_selfplay_epoch(
        ctx=object(),
        components=Components(),
        epoch=1,
        games_per_epoch=0,
    )

    assert result["scheduler"] == "continuous"
    assert called == ["continuous"]


# =========================================================================== #
# Native continuous scheduler, end to end on CPU. These exercise the REAL Rust
# run_continuous against the real engine with a trivial deterministic network:
# uniform policy logits and a flat value head keep the forward sub-millisecond
# while the search machinery (selection, virtual loss, tree reuse, TSS guard,
# Dirichlet noise, move sampling) runs exactly as in production.
# =========================================================================== #
class _UniformPV(torch.nn.Module):
    def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        batch = int(inputs.shape[0])
        return {
            "policy": torch.zeros((batch, BOARD_AREA), dtype=torch.float32),
            "value": torch.zeros((batch, VALUE_BINS), dtype=torch.float32),
        }

    def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.forward_policy_value(inputs)


def _drive_continuous_epoch(
    *,
    games_total: int,
    slots: int,
    visits: int,
    moves_per_game: int,
    base_seed: int = 7,
    fail_at_move: int | None = None,
) -> tuple[dict[int, list[int]], list[int], Mapping[str, Any]]:
    """Run a miniature continuous epoch; returns (actions per game, visit counts, summary)."""

    session = new_mcts_session(max_states=4096)
    inference = DenseCNNInference(_UniformPV(), device="cpu", amp=False, max_batch_size=32)
    states: dict[int, Any] = {}
    actions: dict[int, list[int]] = {}
    visits_seen: list[int] = []
    next_key = 0

    def _new_game() -> int:
        nonlocal next_key
        key = next_key
        next_key += 1
        states[key] = engine.new_game(seed=1_000 + key)
        actions[key] = []
        return key

    initial = [_new_game() for _ in range(slots)]

    def _on_move(game_key: int, payload: Mapping[str, Any]) -> object:
        if fail_at_move is not None and len(visits_seen) >= fail_at_move:
            raise ValueError("synthetic on_move failure")
        result = _result_from_payload(payload)
        visits_seen.append(int(result.visits))
        state = states[game_key]
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(result.action_id))))
        actions[game_key].append(int(result.action_id))
        if engine.terminal(state) is None and len(actions[game_key]) < moves_per_game:
            return ("advance", state)
        if next_key < games_total:
            key = _new_game()
            return ("replace", key, states[key])
        return None

    summary = session.run_continuous(
        initial,
        [states[key] for key in initial],
        inference,
        _on_move,
        visits=visits,
        c_puct=1.5,
        base_seed=base_seed,
        virtual_batch_size=4,
        flush_target=8,
        active_root_limit=64,
        temperature_by_ply=[1.0, 1.0, 0.5, 0.25],
        root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=8,
        widening_min_children=2,
        forced_playout_k=2.0,
    )
    return actions, visits_seen, summary


def test_run_continuous_exact_visits_every_move() -> None:
    actions, visits_seen, summary = _drive_continuous_epoch(
        games_total=3, slots=3, visits=24, moves_per_game=4
    )
    assert len(visits_seen) == 3 * 4
    assert all(item == 24 for item in visits_seen)
    assert summary["moves_decided"] == len(visits_seen)
    assert all(len(moves) == 4 for moves in actions.values())


def test_run_continuous_is_deterministic_across_runs() -> None:
    first, _, _ = _drive_continuous_epoch(games_total=4, slots=2, visits=16, moves_per_game=3)
    second, _, _ = _drive_continuous_epoch(games_total=4, slots=2, visits=16, moves_per_game=3)
    assert first == second


def test_run_continuous_seed_changes_actions() -> None:
    first, _, _ = _drive_continuous_epoch(
        games_total=2, slots=2, visits=16, moves_per_game=3, base_seed=7
    )
    second, _, _ = _drive_continuous_epoch(
        games_total=2, slots=2, visits=16, moves_per_game=3, base_seed=8
    )
    assert first != second


def test_run_continuous_rolling_refill_completes_all_games() -> None:
    actions, visits_seen, summary = _drive_continuous_epoch(
        games_total=5, slots=2, visits=16, moves_per_game=3
    )
    assert sorted(actions) == [0, 1, 2, 3, 4]
    assert all(len(moves) == 3 for moves in actions.values())
    assert summary["moves_decided"] == 5 * 3
    assert len(visits_seen) == 5 * 3


def test_run_continuous_on_move_exception_propagates() -> None:
    with pytest.raises(ValueError, match="synthetic on_move failure"):
        _drive_continuous_epoch(
            games_total=2, slots=2, visits=16, moves_per_game=3, fail_at_move=2
        )


def test_run_continuous_summary_feeds_existing_summarizer() -> None:
    _, visits_seen, summary = _drive_continuous_epoch(
        games_total=2, slots=2, visits=16, moves_per_game=2
    )
    aggregated = _summarize_mcts_diagnostic_batches([dict(summary["mcts_batch_diagnostics"])])
    assert aggregated["batch_count"] == 1
    assert aggregated["eval_requested_states"] > 0
    # Tree reuse and shortcut backups mean unique evals < raw simulations, but
    # the scheduler must account for every flushed state exactly once.
    assert summary["flushed_states"] == aggregated["eval_unique_states"]
    histogram = summary["flush_size_histogram"]
    assert sum(histogram.values()) == summary["flush_count"]
    # Power-of-two bucketing keeps the histogram compact.
    assert all(bucket & (bucket - 1) == 0 for bucket in histogram)
    assert summary["on_move_seconds"] >= 0.0
    assert len(visits_seen) == 2 * 2
