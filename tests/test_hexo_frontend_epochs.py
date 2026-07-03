"""Unit tests for the per-epoch telemetry strip (/api/training/epochs).

The record-assembly function ``web._training_epochs`` is exercised two ways:

* against the REAL ``hexfield_main_7`` diagnostics (read-only) — this asserts
  graceful degradation on the legacy/pre-schema epochs that exist on disk
  (12 = full legacy self-play, 13 = the zeroed/annotated resumed sample,
  15 = the resumed + newer-select-schema epoch), and
* against a synthetic epoch carrying the FULL upgraded self-play schema
  (2026-07-03), since no real on-disk epoch has the new keys yet — this asserts
  the new keys (per-phase entropy/value, decided_fraction, wins_by_player,
  policy_surprise, unique_openings {10/16/20}, merged_approx, segments) are
  surfaced.

Run (WSL build venv, same PYTHONPATH as the dashboard unit file):
    wsl -d Ubuntu-24.04 bash -lc "cd /mnt/e/Hexo-BotTrainer-hexgt && \
      PYTHONPATH=packages/hexo_frontend/python:packages/hexo_engine/python:\
packages/hexo_utils/python:packages/hexo_train/python \
      /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexo_frontend_epochs.py -q"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_frontend", "hexo_runner", "hexo_engine", "hexo_utils"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# web.py pulls in hexo_runner -> hexo_utils._rust at import time; the Rust
# extension is only built in the WSL venv, so the whole module skips elsewhere.
web = pytest.importorskip("hexo_frontend.web", reason="needs hexo_runner/engine build")

# The live run mount (production dashboard cwd). Its diagnostics are read-only.
REAL_RUN_ROOT = Path("/mnt/e/Hexo-BotTrainer")
REAL_RUN = "hexfield_main_7"
_real_run_dir = REAL_RUN_ROOT / "runs" / REAL_RUN
_HAS_REAL_RUN = (_real_run_dir / "diagnostics").is_dir()
real_only = pytest.mark.skipif(not _HAS_REAL_RUN, reason=f"{_real_run_dir} not present")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(epochs: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    match = next((rec for rec in epochs if rec.get("epoch") == epoch), None)
    assert match is not None, f"epoch {epoch} missing from strip"
    return match


# --------------------------------------------------------------------------- #
# Real-data assertions (graceful degradation on the on-disk legacy epochs).
# --------------------------------------------------------------------------- #


@real_only
def test_real_run_strip_shape_and_ordering() -> None:
    payload = web._training_epochs(_real_run_dir)
    assert payload["run"] == REAL_RUN
    epochs = payload["epochs"]
    assert isinstance(epochs, list) and epochs, "expected non-empty epoch list"
    nums = [rec["epoch"] for rec in epochs]
    assert nums == sorted(nums), "records must be ascending by epoch"
    # Every record carries the four blocks, each a dict (never partial-crash).
    for rec in epochs:
        for block in ("selfplay", "select", "training"):
            assert isinstance(rec[block], dict)
        assert rec["eval"] is None or isinstance(rec["eval"], dict)


@real_only
def test_real_epoch12_full_legacy_selfplay_degrades_gracefully() -> None:
    epochs = web._training_epochs(_real_run_dir)["epochs"]
    rec = _record(epochs, 12)
    sp, sel, tr = rec["selfplay"], rec["select"], rec["training"]

    # Legacy self-play keys present -> populated; new keys absent -> None (never KeyError).
    assert sp["mean_game_length"] is not None
    assert sp["game_length_p90"] is not None            # from legacy p90_game_length
    assert sp["root_policy_entropy_mean"] is not None
    assert sp["unique_openings"] == {"10": sp["unique_openings"]["10"]}  # legacy scalar -> {"10": n}
    # New-schema fields are absent on disk -> degrade to None, not crash.
    assert sp["decided_fraction"] is None
    assert sp["root_policy_entropy_by_phase"] is None
    assert sp["root_value_by_phase"] is None
    assert sp["policy_surprise_mean"] is None
    assert sp["p0_win_share"] is None
    # The per-move rates are DERIVED from the scheduler counters when the
    # producer omits the *_rate keys, so they populate on legacy epochs.
    assert 0.0 <= sp["fast_fraction"] <= 1.0
    assert 0.0 <= sp["full_fraction"] <= 1.0
    assert 0.0 <= sp["gumbel_play_winner_rate"] <= 1.0
    # Select + training subsets populate; the newer select keys stay None here.
    assert sel["selected_rows"] is not None
    assert sel["reuse_ratio"] is not None
    assert sel["window_epoch_span"] is None
    assert tr["loss_policy"] is not None
    assert tr["loss_value"] is not None
    assert tr["loss_total"] is not None
    assert tr["steps"] is not None


@real_only
def test_real_epoch13_zeroed_resumed_sample() -> None:
    epochs = web._training_epochs(_real_run_dir)["epochs"]
    rec = _record(epochs, 13)
    sp = rec["selfplay"]
    # Epoch 13 is the annotated resumed sample: everything zeroed, resumed
    # provenance carried via resumed_existing_games (the legacy annotation).
    assert sp["resumed_existing_games"] == 256
    assert sp["rows_written"] == 0
    assert sp["root_policy_entropy_mean"] is None
    # Still degrades cleanly — the block is a full dict, not a crash.
    assert sp["decided_fraction"] is None


@real_only
def test_real_epoch15_resumed_and_newer_select_schema() -> None:
    epochs = web._training_epochs(_real_run_dir)["epochs"]
    rec = _record(epochs, 15)
    sp, sel, tr = rec["selfplay"], rec["select"], rec["training"]
    # Epoch 15 carries the resumed_skip flag (drives the "resumed" badge).
    assert sp["resumed_skip"] is True
    # And the newer select schema (window span + shards_skipped + select_seconds).
    assert sel["shards_skipped"] == 0
    assert isinstance(sel["window_epoch_span"], dict)
    assert sel["window_epoch_span"]["epochs"] is not None
    assert sel["select_seconds"] is not None
    assert tr["train_seconds"] is not None


@real_only
def test_real_eval_epoch_carries_headline_elo() -> None:
    epochs = web._training_epochs(_real_run_dir)["epochs"]
    # Multi-stage eval exists at epochs 5 and 10 for this run.
    rec = _record(epochs, 10)
    ev = rec["eval"]
    assert isinstance(ev, dict)
    assert ev["verdict_label"] is not None
    assert isinstance(ev["edges"], list) and ev["edges"], "expected headline edges"
    for edge in ev["edges"]:
        assert "opponent" in edge and "winrate" in edge and "elo_point" in edge
    # Epochs without an eval report leave the block None.
    assert _record(epochs, 12)["eval"] is None


# --------------------------------------------------------------------------- #
# Upgraded-schema assertion (synthetic epoch 16 — no on-disk epoch has the
# full new key set yet, so a fixture proves the new keys are surfaced).
# --------------------------------------------------------------------------- #


def _upgraded_selfplay(epoch: int) -> dict[str, Any]:
    return {
        "status": "completed",
        "epoch": epoch,
        "elapsed_seconds": 1900.0,
        "search_visits": 1024,
        "games_finished": 256,
        "games_started": 256,
        "truncated_games": 3,
        "rows_written": 21800,
        "total_decisions": 22900,
        "mean_game_length": 88.0,
        "game_length_p10": 40,
        "game_length_p50": 85,
        "game_length_p90": 150,
        "game_length_max": 210,
        "root_policy_entropy_mean": 2.65,
        "root_policy_entropy_by_phase": {
            "opening": {"mean": 3.1, "n": 2560},
            "mid": {"mean": 2.6, "n": 9000},
            "late": {"mean": 1.9, "n": 4000},
        },
        "root_value_mean": -0.08,
        "root_value_abs_mean": 0.42,
        "root_value_std": 0.31,
        "root_value_by_phase": {
            "opening": {"mean": 0.01, "n": 2560},
            "mid": {"mean": -0.05, "n": 9000},
            "late": {"mean": -0.2, "n": 4000},
        },
        "decided_fraction": 0.978,
        "wins_by_player": {"0": 130, "1": 120},
        "policy_surprise_mean": 0.34,
        "policy_surprise_p90": 0.71,
        "policy_surprise_max": 1.9,
        "unique_openings": {"10": 245, "16": 233, "20": 210},
        "gumbel_play_winner_rate": 0.67,
        "gumbel_play_winner_early_rate": 0.46,
        "lcb_override_rate": 0.28,
        "fast_fraction": 0.66,
        "full_fraction": 0.33,
        "init_fraction": 0.01,
        "resumed_skip": True,
        "segments": [{"start": 0, "end": 128}, {"start": 128, "end": 256}],
        "merged_approx": True,
        "scheduler": {
            "full_moves": 7500,
            "fast_moves": 15000,
            "init_moves": 200,
            "moves_decided": 22900,
            "gumbel_play_moves": 7300,
            "gumbel_play_winner_moves": 4880,
            "gumbel_play_moves_early": 1460,
            "gumbel_play_winner_early": 625,
            "lcb_overrides": 6180,
        },
    }


def test_upgraded_epoch_surfaces_new_keys(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    run_dir = tmp_path / "runs" / "hexfield_main_synthetic"
    diag = run_dir / "diagnostics"

    # A manifest with the hexfield lineage so _diag_prefix picks the hexfield.* files.
    _write_json(run_dir / "manifest.json", {"model": {"name": "hexo_models.hexfield"}})
    _write_json(diag / "hexfield.selfplay.epoch_000016.json", _upgraded_selfplay(16))
    _write_json(
        diag / "hexfield.select.epoch_000016.json",
        {
            "epoch": 16,
            "keep_prob": 0.9,
            "select_request": 115000,
            "selected_rows": 96000,
            "window_rows": 96000,
            "reuse_ratio": 4.4,
            "shards_skipped": 2,
            "skipped_paths": ["shards/e09/part-003.hxz", "shards/e10/part-011.hxz"],
            "window_epoch_span": {"min": 10, "max": 16, "epochs": 7},
            "select_seconds": 12.4,
        },
    )
    _write_json(
        diag / "hexfield.training.epoch_000016.json",
        {
            "epoch": 16,
            "loss_policy": 2.01,
            "loss_soft_policy": 2.38,
            "loss_value": 0.57,
            "loss_total": 6.98,
            "steps": 375,
            "trained_rows": 96000,
            "surprise_weight_mean": 1.12,
            "surprise_weight_max": 3.4,
            "select_seconds": 12.4,
            "train_seconds": 405.0,
        },
    )

    rec = _record(web._training_epochs(run_dir)["epochs"], 16)
    sp, sel, tr = rec["selfplay"], rec["select"], rec["training"]

    # New self-play schema surfaced verbatim.
    assert sp["decided_fraction"] == 0.978
    assert sp["root_policy_entropy_by_phase"]["opening"]["mean"] == 3.1
    assert sp["root_value_by_phase"]["late"]["n"] == 4000
    assert sp["policy_surprise_mean"] == 0.34
    assert sp["policy_surprise_p90"] == 0.71
    assert sp["unique_openings"] == {"10": 245, "16": 233, "20": 210}
    assert sp["game_length_p50"] == 85
    assert sp["wins_by_player"] == {"0": 130, "1": 120}
    # p0_win_share derived from wins_by_player: 130 / (130 + 120).
    assert abs(sp["p0_win_share"] - (130 / 250)) < 1e-9
    # Emitted *_rate/​*_fraction keys win over the derivation path.
    assert sp["gumbel_play_winner_rate"] == 0.67
    assert sp["fast_fraction"] == 0.66
    # Resume/merge provenance for the badges.
    assert sp["resumed_skip"] is True
    assert sp["merged_approx"] is True
    assert isinstance(sp["segments"], list) and len(sp["segments"]) == 2

    # Newer select schema surfaced, skipped_paths carried (capped list of str).
    assert sel["shards_skipped"] == 2
    assert sel["window_epoch_span"] == {"min": 10, "max": 16, "epochs": 7}
    assert sel["skipped_paths"] == ["shards/e09/part-003.hxz", "shards/e10/part-011.hxz"]
    assert sel["keep_prob"] == 0.9

    # Training subset incl. the new surprise-weight keys.
    assert tr["surprise_weight_mean"] == 1.12
    assert tr["surprise_weight_max"] == 3.4
    assert tr["train_seconds"] == 405.0


def test_unknown_run_and_empty_diagnostics(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    # Unknown run -> ValueError from the cached entry point (route maps to 400).
    with pytest.raises(ValueError):
        web._training_epochs_cached("no_such_run")

    # A known run with an empty diagnostics dir -> empty epoch list, no crash.
    run_dir = tmp_path / "runs" / "hexfield_main_empty"
    (run_dir / "diagnostics").mkdir(parents=True)
    _write_json(run_dir / "manifest.json", {"model": {"name": "hexo_models.hexfield"}})
    payload = web._training_epochs_cached("hexfield_main_empty")
    assert payload["run"] == "hexfield_main_empty"
    assert payload["epochs"] == []


def test_partial_epoch_missing_files_degrades(tmp_path: Path, monkeypatch: Any) -> None:
    """An epoch with only a self-play file (no select/training) still yields a
    full record with the missing blocks all-None."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    run_dir = tmp_path / "runs" / "hexfield_main_partial"
    diag = run_dir / "diagnostics"
    _write_json(run_dir / "manifest.json", {"model": {"name": "hexo_models.hexfield"}})
    _write_json(
        diag / "hexfield.selfplay.epoch_000007.json",
        {"epoch": 7, "status": "completed", "mean_game_length": 90.0, "rows_written": 20000},
    )

    rec = _record(web._training_epochs(run_dir)["epochs"], 7)
    assert rec["selfplay"]["mean_game_length"] == 90.0
    # No select/training file -> those blocks are dicts of all-None.
    assert rec["select"]["selected_rows"] is None
    assert rec["training"]["loss_policy"] is None
    assert rec["eval"] is None
