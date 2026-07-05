"""Blunder-seeded-game visibility in the dashboard (web.py joins).

Covers the three payload surfaces the seeding feature touches:

* epoch rows (``_epoch_history`` -> ``_selfplay_epoch_summary``): carry
  ``games_seeded``/``seed_ply_mean``/``unique_openings_seeded`` when the
  epoch's diagnostics emit them, and OMIT the keys entirely for pre-seeding
  epochs and foreign (dense_cnn) producers;
* game rows (``_hxr_base_rows``): selfplay rows gain ``seeded``/``seed_ply``
  joined from the per-game npz sidecar ``samples/epoch_NNNNNN/game_<key>.json``
  via the key embedded in the record game_id (``epoch-NNNNNN-game-<key>``),
  and unseeded rows keep their pre-feature shape exactly;
* the replay payload (``_training_history``): ``history`` and ``record_games``
  carry the same joined fields, guarded for missing/unseeded sidecars.

Run (WSL — the Rust engine/.hxr codec only exists in the build venv):
  /root/.venvs/hexgt-build/bin/python -m pytest tests/test_frontend_training_seeded.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in (
    "hexo_frontend",
    "hexo_runner",
    "hexo_engine",
    "hexo_utils",
):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# web.py pulls in hexo_runner -> hexo_utils._rust at import time; the Rust
# extension is only built in the WSL venv, so the whole module skips elsewhere.
web = pytest.importorskip("hexo_frontend.web", reason="needs hexo_runner/engine build")

import hexo_engine as engine  # noqa: E402  (import order: after the skip gate)
from hexo_engine.types import unpack_coord_id  # noqa: E402
from hexo_runner.records import HexoRecordFile, HexoRecordPlayer  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_selfplay_hxr(run_dir: Path, epoch: int, games: list[tuple[int, int]]) -> Path:
    """Write ``selfplay/epoch_NNNNNN.hxr`` with one completed record per
    ``(game_key, n_moves)``, using engine-legal move sequences (the replay
    endpoint re-applies every action through the real engine) and the exact
    game_id shape hexfield.selfplay._write_record emits."""

    record_dir = run_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"epoch_{epoch:06d}.hxr"
    players = (
        HexoRecordPlayer("hexfield-a", "player0", "Hexfield A"),
        HexoRecordPlayer("hexfield-b", "player1", "Hexfield B"),
    )
    with HexoRecordFile.create(path, engine.engine_metadata(), players) as record_file:
        for key, n_moves in games:
            writer = record_file.begin_game(f"epoch-{epoch:06d}-game-{key}", seed=key)
            state = engine.new_game(seed=key)
            for _ in range(n_moves):
                action_id = int(engine.legal_action_ids(state)[0])
                action = engine.PlacementAction(unpack_coord_id(action_id))
                engine.apply_action(state, action)
                writer.record_action(action)
            writer.finish_completed("player0", n_moves)
    return path


def _clear_caches() -> None:
    web._hxr_history_cache.clear()
    web._hxr_count_cache.clear()
    web._seed_sidecar_cache.clear()


# ---------------------------------------------------------------------------
# Epoch rows: games_seeded passthrough / graceful omission.
# ---------------------------------------------------------------------------


def _epoch_diag(epoch: int, selfplay_extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "completed",
        "metadata": {
            "result": {
                "epoch": epoch,
                "selfplay": {
                    "status": "completed",
                    "games_finished": 256,
                    "rows_written": 9000,
                    **selfplay_extra,
                },
            }
        },
    }


def test_epoch_row_carries_games_seeded_when_present(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_caches()
    run_dir = tmp_path / "runs" / "hexfield_seed_epochs"
    # Epoch 69 predates the seeding deploy (no keys); epoch 70 seeded 26 games.
    _write_json(run_dir / "diagnostics" / "epoch_000069.json", _epoch_diag(69, {}))
    _write_json(
        run_dir / "diagnostics" / "epoch_000070.json",
        _epoch_diag(70, {"games_seeded": 26, "seed_ply_mean": 31.2, "unique_openings_seeded": 9}),
    )

    rows = {row["epoch"]: row for row in web._epoch_history(run_dir)}

    sp70 = rows[70]["selfplay"]
    assert sp70["games_seeded"] == 26
    assert sp70["seed_ply_mean"] == pytest.approx(31.2)
    assert sp70["unique_openings_seeded"] == 9

    # Pre-seeding epoch: the keys are absent, not null — app.js gates on presence.
    sp69 = rows[69]["selfplay"]
    assert "games_seeded" not in sp69
    assert "seed_ply_mean" not in sp69
    assert "unique_openings_seeded" not in sp69


def test_selfplay_summary_passes_none_seed_ply_through(tmp_path: Path) -> None:
    # An epoch that seeded games but recorded no ply mean (games_seeded==0 edge
    # is producer-side None) must not fabricate values.
    summary = web._selfplay_epoch_summary(
        {"status": "completed", "games_seeded": 0, "seed_ply_mean": None, "unique_openings_seeded": 0}
    )
    assert summary["games_seeded"] == 0
    assert summary["seed_ply_mean"] is None
    assert summary["unique_openings_seeded"] == 0


# ---------------------------------------------------------------------------
# Game rows + replay payload: sidecar join.
# ---------------------------------------------------------------------------


def _seeded_run(tmp_path: Path) -> Path:
    """Run with three selfplay games in epoch 5: game A seeded at ply 3,
    game B with an ordinary (unseeded) sidecar, game C with NO sidecar."""

    run_dir = tmp_path / "runs" / "hexfield_seed_join"
    _write_selfplay_hxr(run_dir, 5, [(5000001, 8), (5000002, 8), (5000003, 8)])
    _write_json(
        run_dir / "samples" / "epoch_000005" / "game_5000001.json",
        {"epoch": 5, "game_key": 5000001, "winner": 0, "truncated": False, "seeded": True, "seed_ply": 3},
    )
    _write_json(
        run_dir / "samples" / "epoch_000005" / "game_5000002.json",
        {"epoch": 5, "game_key": 5000002, "winner": 0, "truncated": False},
    )
    return run_dir


def test_hxr_rows_join_seed_sidecar(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_caches()
    run_dir = _seeded_run(tmp_path)

    rows = web._hxr_base_rows(run_dir / "selfplay" / "epoch_000005.hxr", run_dir)

    assert len(rows) == 3
    by_id = {row["game_id"]: row for row in rows}
    seeded = by_id["epoch-000005-game-5000001"]
    assert seeded["seeded"] is True
    assert seeded["seed_ply"] == 3
    # Unseeded sidecar and missing sidecar: rows keep the pre-feature shape.
    for game_id in ("epoch-000005-game-5000002", "epoch-000005-game-5000003"):
        assert "seeded" not in by_id[game_id]
        assert "seed_ply" not in by_id[game_id]


def test_replay_payload_joins_seed_sidecar(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    _clear_caches()
    _seeded_run(tmp_path)

    payload = web._training_history("hexfield_seed_join", "selfplay/epoch_000005.hxr", 0)

    history = payload["history"]
    assert history["seeded"] is True
    assert history["seed_ply"] == 3
    games = {item["game_id"]: item for item in payload["record_games"]}
    assert games["epoch-000005-game-5000001"]["seeded"] is True
    assert games["epoch-000005-game-5000001"]["seed_ply"] == 3
    assert "seeded" not in games["epoch-000005-game-5000002"]
    assert "seeded" not in games["epoch-000005-game-5000003"]


def test_replay_payload_unseeded_game_unchanged(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    _clear_caches()
    _seeded_run(tmp_path)

    for record_index in (1, 2):  # unseeded sidecar / missing sidecar
        payload = web._training_history("hexfield_seed_join", "selfplay/epoch_000005.hxr", record_index)
        assert "seeded" not in payload["history"]
        assert "seed_ply" not in payload["history"]


def test_foreign_game_ids_and_missing_samples_dir_are_untouched(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """dense_cnn-style ids never match the selfplay pattern, and a selfplay id
    whose samples dir is absent (pre-seeding epochs, pruned samples) joins {}."""

    monkeypatch.chdir(tmp_path)
    _clear_caches()
    run_dir = tmp_path / "runs" / "foreign_run"
    run_dir.mkdir(parents=True)

    assert web._seed_provenance_for_game(run_dir, "ep65-cand_ep65-vs-ep60-g3-candP1") == {}
    assert web._seed_provenance_for_game(run_dir, "dense-game-7") == {}
    assert web._seed_provenance_for_game(run_dir, None) == {}
    # Selfplay-shaped id, but no samples/epoch_000004 directory exists.
    assert web._seed_provenance_for_game(run_dir, "epoch-000004-game-4000001") == {}
