from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


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


def test_training_artifact_browser_finds_config_relative_runs_and_hxr_files(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import hexo_frontend.web as web

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "configs" / "runs" / "dense_cnn_model1"
    diagnostics = run_dir / "diagnostics"
    selfplay = run_dir / "selfplay"
    diagnostics.mkdir(parents=True)
    selfplay.mkdir(parents=True)
    (diagnostics / "dense_cnn.selfplay.epoch_000001.json").write_text(
        '{"epoch":1,"searched_positions":4,"mcts_simulations":512}',
        encoding="utf-8",
    )
    (selfplay / "epoch_000001.hxr").write_bytes(b"hxr")

    runs = web._training_runs()
    assert any(item["name"] == "dense_cnn_model1" for item in runs["runs"])

    run = web._training_run("dense_cnn_model1")
    artifact_paths = {item["path"] for item in run["artifacts"]}
    assert "diagnostics/dense_cnn.selfplay.epoch_000001.json" in artifact_paths
    assert "selfplay/epoch_000001.hxr" in artifact_paths

    hxr_path = web._resolve_run_path("dense_cnn_model1", "selfplay/epoch_000001.hxr")
    assert hxr_path == selfplay / "epoch_000001.hxr"
    assert web._resolve_run_path("dense_cnn_model1", "../secret") is None


def test_training_history_endpoint_replays_hxr_into_dashboard_state(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import hexo_frontend.web as web
    from hexo_engine.types import AxialCoord, pack_coord_id
    from hexo_runner.records import HexoRecordFile, HexoRecordPlayer

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "dense_cnn_history"
    selfplay = run_dir / "selfplay"
    diagnostics = run_dir / "diagnostics"
    selfplay.mkdir(parents=True)
    diagnostics.mkdir(parents=True)
    hxr = selfplay / "epoch_000001.hxr"
    (diagnostics / "dense_cnn.evaluation.epoch_000001.json").write_text(
        '{"status":"completed","epoch":1,"games":2,"wins":1,"losses":1,"mean_turns":12.5}',
        encoding="utf-8",
    )
    actions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
        (1, 1),
        (1, 2),
        (3, 0),
        (4, 0),
        (2, 1),
        (2, 2),
        (5, 0),
    ]

    with HexoRecordFile.create(
        hxr,
        {"rules_version": 1, "backend": "test"},
        (
            HexoRecordPlayer("dense-cnn-eval", "player0", "Dense CNN"),
            HexoRecordPlayer("sealbot-best-50ms", "player1", "SealBot best"),
        ),
    ) as record_file:
        writer = record_file.begin_game("history-game", seed=11)
        for q, r in actions:
            writer.record_action(pack_coord_id(AxialCoord(q=q, r=r)))
        writer.finish_completed("player0", len(actions))

    run = web._training_run("dense_cnn_history")
    hxr_artifact = next(item for item in run["artifacts"] if item["path"] == "selfplay/epoch_000001.hxr")
    history = run["histories"][0]
    payload = web._training_history("dense_cnn_history", "selfplay/epoch_000001.hxr")

    assert hxr_artifact["loadable_history"] is True
    assert hxr_artifact["history_count"] == 1
    assert history["epoch"] == 1
    assert history["winner_label"] == "P0"
    assert history["length"] == len(actions)
    assert history["diagnostics"]["evaluation"]["summary"]["mean_turns"] == 12.5
    assert payload["mode"] == "history"
    assert payload["game_id"] == "dense_cnn_history:history-game"
    assert payload["winner"] == "player0"
    assert len(payload["placements"]) == len(actions)
    assert payload["players"]["player0"]["kind"] == "dense-cnn"
    assert payload["players"]["player1"]["kind"] == "sealbot-best"


def test_training_run_lists_multi_record_history_metadata_and_loads_selected_record(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import hexo_frontend.web as web
    from hexo_engine.types import AxialCoord, pack_coord_id
    from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "dense_cnn_history"
    evaluation = run_dir / "evaluation"
    diagnostics = run_dir / "diagnostics"
    evaluation.mkdir(parents=True)
    diagnostics.mkdir(parents=True)
    hxr = evaluation / "epoch_000002.hxr"
    (diagnostics / "dense_cnn.evaluation.epoch_000002.json").write_text(
        '{"status":"completed","epoch":2,"games":2,"wins":1,"losses":0,"mean_turns":10.0}',
        encoding="utf-8",
    )
    aborted_actions = [(0, 0)]
    completed_actions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
        (1, 1),
        (1, 2),
        (3, 0),
        (4, 0),
        (2, 1),
        (2, 2),
        (5, 0),
    ]

    with HexoRecordFile.create(
        hxr,
        {"rules_version": 1, "backend": "test"},
        (
            HexoRecordPlayer("dense-cnn-eval", "player0", "Dense CNN"),
            HexoRecordPlayer("sealbot-best-50ms", "player1", "SealBot best"),
        ),
    ) as record_file:
        writer = record_file.begin_game("aborted-game", seed=101)
        for q, r in aborted_actions:
            writer.record_action(pack_coord_id(AxialCoord(q=q, r=r)))
        writer.finish_aborted(
            AbortRecord(
                stage="runner.max_actions",
                exception_type="MaxActionsExceeded",
                message="hit max_actions=1",
            )
        )

        writer = record_file.begin_game("completed-game", seed=202)
        for q, r in completed_actions:
            writer.record_action(pack_coord_id(AxialCoord(q=q, r=r)))
        writer.finish_completed("player0", len(completed_actions))

    run = web._training_run("dense_cnn_history")
    hxr_artifact = next(item for item in run["artifacts"] if item["path"] == "evaluation/epoch_000002.hxr")
    histories = {
        item["record_index"]: item
        for item in run["histories"]
        if item["path"] == "evaluation/epoch_000002.hxr"
    }
    payload = web._training_history("dense_cnn_history", "evaluation/epoch_000002.hxr", record_index=1)

    assert hxr_artifact["loadable_history"] is True
    assert hxr_artifact["history_count"] == 2
    assert set(histories) == {0, 1}
    assert histories[0]["status"] == "aborted"
    assert histories[0]["winner_label"] == "None"
    assert histories[0]["length"] == len(aborted_actions)
    assert histories[0]["source"] == "evaluation"
    assert histories[0]["epoch"] == 2
    assert histories[0]["abort"]["stage"] == "runner.max_actions"
    assert histories[0]["diagnostics"]["evaluation"]["summary"]["games"] == 2
    assert histories[1]["status"] == "completed"
    assert histories[1]["winner_label"] == "P0"
    assert histories[1]["seed"] == 202
    assert histories[1]["players"]["player0"]["label"] == "Dense CNN"

    assert payload["mode"] == "history"
    assert payload["game_id"] == "dense_cnn_history:completed-game"
    assert payload["match"]["seed"] == 202
    assert payload["history"]["record_index"] == 1
    assert payload["history"]["record_count"] == 2
    assert payload["history"]["abort"] is None
    assert payload["record_games"] == [
        {
            "index": 0,
            "game_id": "aborted-game",
            "status": "aborted",
            "actions": len(aborted_actions),
            "winner": None,
        },
        {
            "index": 1,
            "game_id": "completed-game",
            "status": "completed",
            "actions": len(completed_actions),
            "winner": "player0",
        },
    ]


def test_training_run_skips_quarantine_and_does_not_expand_bootstrap_hxr(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import hexo_frontend.web as web
    from hexo_engine.types import AxialCoord, pack_coord_id
    from hexo_runner.records import HexoRecordFile, HexoRecordPlayer

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "dense_cnn_history"
    selfplay = run_dir / "selfplay"
    bootstrap = run_dir / "bootstrap" / "sealbot_050000"
    quarantine = run_dir / "quarantine" / "stale_restart"
    for path in (selfplay, bootstrap, quarantine):
        path.mkdir(parents=True)

    players = (
        HexoRecordPlayer("dense-cnn", "player0", "Dense CNN"),
        HexoRecordPlayer("sealbot-best", "player1", "SealBot best"),
    )

    for hxr, game_id in (
        (selfplay / "epoch_000001.hxr", "selfplay-game"),
        (bootstrap / "classical_sealbot_bootstrap.hxr", "bootstrap-game"),
        (quarantine / "epoch_000001.hxr", "quarantined-game"),
    ):
        with HexoRecordFile.create(hxr, {"rules_version": 1, "backend": "test"}, players) as record_file:
            writer = record_file.begin_game(game_id, seed=1)
            writer.record_action(pack_coord_id(AxialCoord(q=0, r=0)))
            writer.finish_completed("player0", 1)

    run = web._training_run("dense_cnn_history")
    artifact_paths = {item["path"] for item in run["artifacts"]}
    history_ids = {item["game_id"] for item in run["histories"]}

    assert "selfplay/epoch_000001.hxr" in artifact_paths
    assert "bootstrap/sealbot_050000/classical_sealbot_bootstrap.hxr" in artifact_paths
    assert "quarantine/stale_restart/epoch_000001.hxr" not in artifact_paths
    assert history_ids == {"selfplay-game"}
