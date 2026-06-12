"""CPU integration smoke for the dense_cnn_restnet main_4 self-play levers.

Runs REAL self-play epochs (the native Rust continuous scheduler and the
lockstep scheduler, real engine, trivial deterministic network) with
frozen_win_override enabled and the new samples knobs present-but-dormant.
Asserts the epoch completes, the new diagnostics counters are emitted, and rows
are written. A bogus-tracker injection drives the override verify-failure path
through the real per-move loop to prove it falls through and never raises.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine", "hexo_runner", "hexo_train", "hexo_utils"):
    path = ROOT / "packages" / package / "python"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

torch = pytest.importorskip("torch")
pytest.importorskip("hexo_utils._rust")

config_module = importlib.import_module("dense_cnn_restnet.config")
constants = importlib.import_module("dense_cnn_restnet.constants")
selfplay_module = importlib.import_module("dense_cnn_restnet.selfplay")


class _UniformPV(torch.nn.Module):
    """Uniform policy + flat value: sub-millisecond CPU forward, real search."""

    def forward_policy_value(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        batch = int(inputs.shape[0])
        return {
            "policy": torch.zeros((batch, constants.BOARD_AREA), dtype=torch.float32),
            "value": torch.zeros((batch, constants.VALUE_BINS), dtype=torch.float32),
        }

    def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.forward_policy_value(inputs)


class _Diagnostics:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        (self.root / name).write_text(json.dumps(payload, default=str), encoding="utf-8")


def _harness(tmp_path: Path, model_config: Mapping[str, Any]) -> tuple[Any, Any]:
    config = config_module.parse_model1_config(dict(model_config))
    trainer = SimpleNamespace(
        config=config,
        device="cpu",
        inference_batch_size=32,
        selfplay_batch_size=4,
        mcts_virtual_batch_size=4,
    )
    components = SimpleNamespace(model=SimpleNamespace(trainer=trainer, model=_UniformPV()))
    ctx = SimpleNamespace(
        output_dir=tmp_path / "run",
        diagnostics=_Diagnostics(tmp_path / "run" / "diagnostics"),
        config=SimpleNamespace(run=SimpleNamespace(seed=11)),
    )
    return ctx, components


def _model_config(scheduler: str, *, max_actions: int, visits: int) -> dict[str, Any]:
    return {
        "device": "cpu",
        "training": {"amp": False},
        "selfplay": {
            "scheduler": scheduler,
            "search_visits": visits,
            "max_actions": max_actions,
            "widening_max_children": 8,
            "frozen_win_override": True,
        },
        # The new samples knobs present but DORMANT (knee 0 / flag off).
        "samples": {
            "drop_truncated_rows": False,
            "length_decay_moves_left_knee": 0.0,
            "length_decay_moves_left_halflife": 40.0,
            "length_decay_game_length_knee": 0.0,
            "length_decay_game_length_halflife": 60.0,
        },
    }


def test_continuous_smoke_with_frozen_win_override_enabled(tmp_path: Path) -> None:
    ctx, components = _harness(
        tmp_path, _model_config("continuous", max_actions=32, visits=32)
    )
    result = selfplay_module.generate_selfplay_epoch(
        ctx=ctx, components=components, epoch=1, games_per_epoch=8
    )

    assert result["status"] == "completed"
    assert result["scheduler"] == "continuous"
    assert result["games_finished"] == 8
    # The new counters are emitted alongside truncated_games; 32-ply games are
    # far too short to grow an out-of-crop frozen win, so both stay 0.
    assert result["frozen_win_overrides"] == 0
    assert result["frozen_win_override_failures"] == 0
    assert result["truncated_rows_dropped"] == 0
    # Rows are written (dormant samples knobs leave the pipeline untouched).
    assert result["raw_samples"] > 0
    assert result["selfplay_npz_files"] > 0
    npz_files = sorted((ctx.output_dir / "selfplay").glob("*.npz"))
    assert len(npz_files) == result["selfplay_npz_files"]
    # on_move overhead (incl. the tracker feed) is accounted by the scheduler.
    assert result["scheduler_diagnostics"]["on_move_seconds"] >= 0.0


def test_lockstep_smoke_emits_counters_and_rows(tmp_path: Path) -> None:
    ctx, components = _harness(tmp_path, _model_config("lockstep", max_actions=8, visits=8))
    result = selfplay_module.generate_selfplay_epoch(
        ctx=ctx, components=components, epoch=1, games_per_epoch=2
    )

    assert result["status"] == "completed"
    assert result["games_finished"] == 2
    assert result["frozen_win_overrides"] == 0
    assert result["frozen_win_override_failures"] == 0
    assert result["truncated_rows_dropped"] == 0
    assert result["raw_samples"] > 0


def test_override_verify_failure_falls_through_in_real_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a tracker that claims a bogus far-out-of-crop standing win on
    every move past the opening: the override verification must fail (the cell
    is illegal at these positions), count the failure, fall through to the
    search move, and finish the epoch without raising."""

    class _BogusTracker:
        def __init__(self) -> None:
            self._stones = 0

        def add_stone(self, q: int, r: int, player: object) -> None:
            self._stones += 1

        def standing_win_cells(self, player: object) -> set[tuple[int, int]]:
            # Far outside any radius-20 crop AND outside the distance-8 legal
            # ring of an opening-stage blob -> verification must fail.
            return {(120, 120)} if self._stones >= 4 else set()

    monkeypatch.setattr(selfplay_module, "IncrementalWinTracker", _BogusTracker)
    ctx, components = _harness(
        tmp_path, _model_config("continuous", max_actions=8, visits=8)
    )
    result = selfplay_module.generate_selfplay_epoch(
        ctx=ctx, components=components, epoch=1, games_per_epoch=2
    )

    assert result["status"] == "completed"
    assert result["games_finished"] == 2
    assert result["frozen_win_overrides"] == 0
    assert result["frozen_win_override_failures"] > 0
    assert result["raw_samples"] > 0


def test_quarantine_drops_truncated_rows_in_real_loop(tmp_path: Path) -> None:
    """End-to-end C2: with drop_truncated_rows on and max_actions tiny, every
    game truncates and the epoch writes zero training rows while the .hxr
    record and game bookkeeping stay intact."""

    model_config = _model_config("continuous", max_actions=6, visits=8)
    model_config["samples"]["drop_truncated_rows"] = True
    ctx, components = _harness(tmp_path, model_config)
    result = selfplay_module.generate_selfplay_epoch(
        ctx=ctx, components=components, epoch=1, games_per_epoch=2
    )

    assert result["status"] == "completed"
    assert result["games_finished"] == 2
    assert result["truncated_games"] == 2
    assert result["truncated_rows_dropped"] == 2 * 6
    assert result["raw_samples"] == 0
    assert result["effective_samples"] == 0
    assert result["selfplay_npz_files"] == 0
    assert not list((ctx.output_dir / "selfplay").glob("*.npz"))
    assert (ctx.output_dir / "selfplay" / "epoch_000001.hxr").exists()
