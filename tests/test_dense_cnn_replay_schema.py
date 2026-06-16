from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models",):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _samples_module() -> Any:
    return importlib.import_module("hexo_models.dense_cnn.samples")


def _replay_module() -> Any:
    return importlib.import_module("hexo_models.dense_cnn.replay")


def _action(q: int, r: int) -> int:
    d6 = importlib.import_module("hexo_models.dense_cnn.d6")
    return int(d6.pack_coord_id(d6.Axial(q, r)))


def _sample(sample_id: str, *, policy: tuple[tuple[int, float], ...], prior: tuple[tuple[int, float], ...]) -> Any:
    samples = _samples_module()
    return samples.Model1SampleData(
        game_id=sample_id,
        turn_index=0,
        current_player="player0",
        phase="Opening",
        center=(0, 0),
        stones=(),
        legal_action_ids=tuple(action for action, _weight in policy),
        policy=policy,
        root_prior_policy=prior,
        opp_policy=policy,
        value=0.25,
        metadata={"sample_id": sample_id},
    )


def test_policy_surprise_materializes_frequency_weighted_rows_deterministically() -> None:
    replay = _replay_module()
    action_a = _action(0, 0)
    action_b = _action(1, 0)
    rows = (
        _sample("low", policy=((action_a, 1.0),), prior=((action_a, 1.0),)),
        _sample("high", policy=((action_b, 1.0),), prior=((action_a, 1.0),)),
    )

    first, stats = replay.materialize_policy_surprise_rows(
        rows,
        seed=123,
    )
    second, second_stats = replay.materialize_policy_surprise_rows(
        rows,
        seed=123,
    )

    assert [row.game_id for row in first] == [row.game_id for row in second]
    assert stats == second_stats
    assert stats["raw_rows"] == 2.0
    assert stats["frequency_weight_mean"] == pytest.approx(1.0)
    weights = {row.game_id: row.frequency_weight for row in first}
    assert weights["low"] == pytest.approx(0.5)
    assert weights["high"] == pytest.approx(1.5)
    assert all(row.policy_surprise >= 0.0 for row in first)


def test_selfplay_npz_writer_uses_compact_schema_and_sidecar(tmp_path: Path) -> None:
    replay = _replay_module()
    compact_io = importlib.import_module("hexo_models.dense_cnn.compact_io")
    rows = replay.materialize_policy_surprise_rows(
        (
            _sample("a", policy=((_action(0, 0), 1.0),), prior=((_action(0, 0), 1.0),)),
            _sample("b", policy=((_action(1, 0), 1.0),), prior=((_action(0, 0), 1.0),)),
        ),
        seed=9,
    )[0]

    result = replay.write_selfplay_npz(
        tmp_path / "selfplay" / "game.npz",
        rows,
        raw_rows=2,
        epoch=3,
        game_id="game",
        short_term_value_horizons=(1, 4),
    )

    assert result.path.exists()
    assert result.sidecar_path.exists()
    # Compact columnar schema: per-row scalar arrays + variable-length data + offsets.
    with np.load(result.path, allow_pickle=True) as data:
        keys = set(data.files)
        assert {
            "turn_index", "value", "stvalue", "stvalue_mask", "horizons",
            "stones_off", "legal_ids", "legal_off", "pol_act", "pol_w", "pol_off",
        } <= keys
        assert data["turn_index"].shape == (len(rows),)
        assert data["value"].shape == (len(rows),)
        assert data["stvalue"].shape == (len(rows), 2)
        assert tuple(int(h) for h in data["horizons"]) == (1, 4)
        assert data["legal_off"].shape == (len(rows) + 1,)
    assert compact_io.compact_row_count(result.path) == len(rows)
    assert len(compact_io.read_compact_shard(result.path)) == len(rows)
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["num_rows"] == len(rows)
    assert sidecar["raw_rows"] == 2
    assert sidecar["effective_rows"] == len(rows)
    assert sidecar["target_schema_version"] == _samples_module().CURRENT_TARGET_SCHEMA_VERSION


def test_npz_row_count_reads_header_when_sidecar_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _replay_module()
    rows = [_sample("a", policy=((_action(0, 0), 1.0),), prior=((_action(0, 0), 1.0),)) for _ in range(3)]
    path = replay.write_selfplay_npz(
        tmp_path / "selfplay" / "game.npz",
        rows,
        raw_rows=3,
        epoch=1,
        game_id="game",
        short_term_value_horizons=(),
    ).path
    replay.sidecar_for_npz(path).unlink()

    monkeypatch.setattr(
        replay.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("np.load should not be used for row counts")),
    )

    assert replay.npz_row_count(path) == 3


def test_katago_shuffle_builds_latest_batch_aligned_train_dir(tmp_path: Path) -> None:
    replay = _replay_module()
    selfplay_dir = tmp_path / "selfplay"
    old_rows = [_sample("old", policy=((_action(0, 0), 1.0),), prior=((_action(0, 0), 1.0),)) for _ in range(4)]
    new_rows = [_sample("new", policy=((_action(1, 0), 1.0),), prior=((_action(1, 0), 1.0),)) for _ in range(4)]
    old_path = replay.write_selfplay_npz(
        selfplay_dir / "old.npz",
        old_rows,
        raw_rows=4,
        epoch=1,
        game_id="old",
        short_term_value_horizons=(),
    ).path
    new_path = replay.write_selfplay_npz(
        selfplay_dir / "new.npz",
        new_rows,
        raw_rows=4,
        epoch=2,
        game_id="new",
        short_term_value_horizons=(),
    ).path
    old_time = 100.0
    new_time = 200.0
    for suffix_path in (old_path, replay.sidecar_for_npz(old_path)):
        suffix_path.touch()
        import os

        os.utime(suffix_path, (old_time, old_time))
    for suffix_path in (new_path, replay.sidecar_for_npz(new_path)):
        suffix_path.touch()
        import os

        os.utime(suffix_path, (new_time, new_time))

    result = replay.build_katago_shuffle(
        selfplay_dir=selfplay_dir,
        shuffled_root=tmp_path / "shuffleddata",
        scratch_dir=tmp_path / "scratch",
        epoch=5,
        seed=17,
        min_rows=4,
        keep_target_rows=8,
        taper_window_exponent=0.65,
        expand_window_per_row=0.0,
        taper_window_scale=50_000.0,
        approx_rows_per_out_file=4,
        batch_size=2,
        worker_group_size=4,
    )

    assert result.status == "completed"
    assert result.shuffle_dir is not None
    assert result.shuffle_dir.parent == tmp_path / "shuffleddata"
    assert result.shuffle_dir.name.endswith("-epoch_000005")
    assert result.output_rows == 4
    assert result.output_files
    assert result.validation_rows == 0
    assert not any((tmp_path / "scratch").glob("*"))
    latest = replay.latest_shuffle_dir(tmp_path / "shuffleddata")
    assert latest == result.shuffle_dir
    train_json = replay.load_train_json(result.shuffle_dir)
    assert train_json["num_rows"] == 4
    assert train_json["total_num_data_rows"] == 8
    assert train_json["worker_group_size"] == 4
    assert train_json["window_start_data_row_idx"] == 4
    assert all(replay.npz_row_count(path) % 2 == 0 for path in result.output_files)


def _write_game(replay: Any, selfplay_dir: Path, name: str, *, epoch: int, horizons: tuple[int, ...], n: int = 4) -> Path:
    rows = [_sample(name, policy=((_action(0, 0), 1.0),), prior=((_action(0, 0), 1.0),)) for _ in range(n)]
    return replay.write_selfplay_npz(
        selfplay_dir / f"{name}.npz", rows, raw_rows=n, epoch=epoch, game_id=name,
        short_term_value_horizons=horizons,
    ).path


def test_shuffle_rejects_mixed_horizon_window(tmp_path: Path) -> None:
    # #6: a window spanning shards with different short_term_value horizons would
    # silently mis-slot stval targets (the output is written with one horizon set).
    # The shuffle must fail loudly instead.
    replay = _replay_module()
    selfplay_dir = tmp_path / "selfplay"
    _write_game(replay, selfplay_dir, "a", epoch=1, horizons=(1, 4))
    _write_game(replay, selfplay_dir, "b", epoch=2, horizons=(1, 4, 8))  # different horizons
    # Full-window params (expand_per_row=1.0, min_rows=1) so BOTH shards are in the
    # window -> the mixed-horizon guard must trigger.
    with pytest.raises(ValueError, match="mixes short_term_value horizons"):
        replay.build_katago_shuffle(
            selfplay_dir=selfplay_dir, shuffled_root=tmp_path / "shuf", scratch_dir=tmp_path / "scr",
            epoch=5, seed=1, min_rows=1, keep_target_rows=64, taper_window_exponent=1.0,
            expand_window_per_row=1.0, taper_window_scale=None, approx_rows_per_out_file=4,
            batch_size=2, worker_group_size=4,
        )


def test_shuffle_trivial_md5_range_is_unchanged(tmp_path: Path) -> None:
    # #5: the md5 sub-range is now applied to the file SET first; the default [0, 1)
    # range must keep every file and leave window_start_data_row_idx correct.
    replay = _replay_module()
    selfplay_dir = tmp_path / "selfplay"
    _write_game(replay, selfplay_dir, "old", epoch=1, horizons=())
    _write_game(replay, selfplay_dir, "new", epoch=2, horizons=())
    result = replay.build_katago_shuffle(
        selfplay_dir=selfplay_dir, shuffled_root=tmp_path / "shuf", scratch_dir=tmp_path / "scr",
        epoch=5, seed=3, min_rows=4, keep_target_rows=64, taper_window_exponent=0.65,
        expand_window_per_row=0.0, taper_window_scale=50_000.0, approx_rows_per_out_file=4,
        batch_size=2, worker_group_size=4,
    )
    assert result.status == "completed"
    train_json = replay.load_train_json(result.shuffle_dir)
    # md5 default range kept every file, so the full stream length is unchanged and
    # window_start stays a valid offset into it (exact value depends on the window
    # formula / mtimes; the dedicated batch-aligned test pins it).
    assert train_json["total_num_data_rows"] == 8
    assert 0 <= train_json["window_start_data_row_idx"] <= 8


def test_checkpoint_loader_rejects_legacy_sample_buffer_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoints = importlib.import_module("hexo_models.dense_cnn.checkpoints")
    replay = _replay_module()

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        train_state=replay.DenseTrainState(train_bucket_level=5.0),
        load_train_state=lambda state: setattr(trainer, "train_state", replay.DenseTrainState.from_mapping(state)),
    )
    components = SimpleNamespace(
        model=SimpleNamespace(
            model=model,
            optimizer=optimizer,
            trainer=trainer,
        )
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_state": {
                "global_step_samples": 12,
                "train_bucket_level": 3.0,
                "train_bucket_level_at_row": 10,
            },
            "sample_buffer": {"legacy": True},
            "epoch": 12,
            "metadata": {"run": "unit"},
        },
        checkpoint_path,
    )

    result = checkpoints.DenseCNNCheckpointLoader().load(checkpoint_path, ctx=None, components=components)

    assert result["status"] == "initialized"
    assert result["reason"] == "legacy dense_cnn sample_buffer checkpoints are unsupported"
    assert trainer.train_state.global_step_samples == 0
    assert trainer.train_state.train_bucket_level == pytest.approx(5.0)
