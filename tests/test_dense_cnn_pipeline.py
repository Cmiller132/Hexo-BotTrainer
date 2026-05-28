from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in (
    "hexo_models",
    "hexo_train",
    "hexo_utils",
    "hexo_engine",
    "hexo_runner",
):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch() -> Any:
    return pytest.importorskip("torch")


def _dense_cnn() -> Any:
    return importlib.import_module("hexo_models.dense_cnn")


def _dense_cnn_plugin() -> Any:
    pytest.importorskip("hexo_utils._rust")
    return importlib.import_module("hexo_models.dense_cnn.plugin").get_plugin()


def _dense_cnn_selfplay_module() -> Any:
    pytest.importorskip("hexo_utils._rust")
    return importlib.import_module("hexo_models.dense_cnn.selfplay")


def _small_model_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "device": "cpu",
        "architecture": {
            "channels": 4,
            "residual_blocks": 1,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "amp": False,
            "max_grad_norm": 1.0,
        },
        "samples": {
            "train_sample_count": 2,
            "compression_level": 1,
        },
        "selfplay": {
            "samples_per_epoch": 2,
            "search_visits": 1,
            "max_actions": 4,
            "worker_count": 1,
        },
        "evaluation": {
            "games_per_epoch": 1,
            "sealbot_time_limit": 0.001,
            "max_actions": 4,
        },
        "debug": {
            "write_game_history": True,
            "write_policy_targets": True,
            "write_sample_previews": False,
            "preview_games": 1,
        },
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def _build_dense_components(tmp_path: Path, model_config: Mapping[str, Any] | None = None) -> tuple[Any, Any]:
    from hexo_train.components import TrainingComponents, build_model_components
    from hexo_train.config import normalize_training_config
    from hexo_train.context import RunContext
    from hexo_train.defaults import build_shared_components

    config = normalize_training_config(
        {
            "model": {
                "name": "hexo_models",
                "module": "hexo_models.dense_cnn.plugin",
                "config": dict(model_config or _small_model_config()),
            },
            "run": {
                "output_dir": tmp_path / "run",
                "seed": 7,
            },
            "samples": {
                "train_sample_count": 2,
            },
            "train": {
                "passes_per_epoch": 1,
            },
        },
        base_dir=tmp_path,
    )
    ctx = RunContext.from_config(config)
    shared = build_shared_components(ctx)
    model = build_model_components(
        plugin=_dense_cnn_plugin(),
        ctx=ctx,
        shared=shared,
    )
    return ctx, TrainingComponents(shared=shared, model=model)


def _torch_load(path: Path) -> Mapping[str, Any]:
    torch = _torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    assert isinstance(payload, Mapping), f"checkpoint payload must be a mapping, got {type(payload).__name__}"
    return payload


def _required_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    raise AssertionError(f"checkpoint payload must include one of {keys!r}; got keys {sorted(payload)}")


def _write_pipeline_config(tmp_path: Path) -> Path:
    output_dir = (tmp_path / "pipeline-run").as_posix()
    config_path = tmp_path / "train_dense_cnn.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'name = "hexo_models"',
                'module = "hexo_models.dense_cnn.plugin"',
                "",
                "[model.config]",
                'device = "cpu"',
                "",
                "[model.config.architecture]",
                "channels = 4",
                "residual_blocks = 1",
                "dropout = 0.0",
                "",
                "[model.config.training]",
                "batch_size = 2",
                "learning_rate = 0.001",
                "weight_decay = 0.0",
                "amp = false",
                "max_grad_norm = 1.0",
                "",
                "[model.config.samples]",
                "train_sample_count = 2",
                "compression_level = 1",
                "",
                "[model.config.selfplay]",
                "samples_per_epoch = 2",
                "search_visits = 1",
                "max_actions = 4",
                "temperature = 1.0",
                "worker_count = 1",
                "",
                "[model.config.evaluation]",
                "games_per_epoch = 1",
                'sealbot_variant = "best"',
                "sealbot_time_limit = 0.001",
                "max_actions = 4",
                "",
                "[model.config.performance]",
                "calibrate = true",
                "target_selfplay_positions_per_second = 128",
                "inference_batch_candidates = [1, 2]",
                "training_batch_candidates = [1, 2]",
                "selfplay_batch_candidates = [1, 2]",
                "mcts_virtual_batch_candidates = [1, 2]",
                "selfplay_probe_positions = 4",
                "probe_batches = 1",
                "",
                "[model.config.debug]",
                "write_game_history = true",
                "write_policy_targets = true",
                "write_sample_previews = false",
                "preview_games = 1",
                "",
                "[run]",
                f'output_dir = "{output_dir}"',
                "seed = 11",
                "",
                "[loop]",
                "epochs = 1",
                "",
                "[selfplay]",
                "games_per_epoch = 1",
                "update_checkpoint_pointer = true",
                f'checkpoint_pointer = "{(tmp_path / "dense_cnn_latest.txt").as_posix()}"',
                "",
                "[samples]",
                "train_sample_count = 2",
                "",
                "[train]",
                "passes_per_epoch = 1",
                "",
                "[checkpoint]",
                'save_name = "latest"',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_real_diagnostic(payload: Any, label: str) -> None:
    assert isinstance(payload, Mapping), f"{label} diagnostic must be a mapping"
    assert payload.get("status") != "skipped", f"{label} diagnostic was skipped: {payload}"

    text = json.dumps(payload, default=str).lower()
    assert "not implemented" not in text, f"{label} diagnostic is still a placeholder: {payload}"
    assert "not wired" not in text, f"{label} diagnostic is still unwired: {payload}"
    assert "placeholder" not in text, f"{label} diagnostic is still placeholder-backed: {payload}"


def _diagnostic_payloads(diagnostics_dir: Path) -> list[Mapping[str, Any]]:
    return [_read_json(path) for path in diagnostics_dir.glob("*.json")]


def _matching_path_values(
    payloads: Sequence[Any],
    *,
    key_tokens: tuple[str, ...],
    base_dir: Path,
) -> list[Path]:
    paths: list[Path] = []

    def visit(value: Any, key_path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                next_key_path = f"{key_path}.{key}".lower() if key_path else str(key).lower()
                if all(token in next_key_path for token in key_tokens):
                    paths.extend(_coerce_paths(item, base_dir=base_dir))
                visit(item, next_key_path)
        elif isinstance(value, list):
            for item in value:
                visit(item, key_path)

    for payload in payloads:
        visit(payload, "")
    return paths


def _coerce_paths(value: Any, *, base_dir: Path) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [_resolve_path(Path(value), base_dir=base_dir)]
    if isinstance(value, Mapping):
        paths: list[Path] = []
        for key, item in value.items():
            if "path" in str(key).lower():
                paths.extend(_coerce_paths(item, base_dir=base_dir))
        return paths
    if isinstance(value, list):
        paths: list[Path] = []
        for item in value:
            paths.extend(_coerce_paths(item, base_dir=base_dir))
        return paths
    return []


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def test_plugin_entry_point_builds_model1_network_defaults() -> None:
    _torch()
    dense_cnn = _dense_cnn()
    plugin = _dense_cnn_plugin()

    model = plugin.build_model({}, {})

    assert isinstance(model, dense_cnn.Model1Network)
    assert model.in_channels == 13
    assert model.board_size == 41
    assert model.channels == 96
    assert model.blocks_count == 6
    assert tuple(model.conv_in.weight.shape[:2]) == (96, 13)
    assert len(model.blocks) == 6


def test_dense_cnn_model1_config_writes_to_repo_level_run_and_checkpoint_paths() -> None:
    from hexo_train.config import load_training_config

    config = load_training_config(ROOT / "configs" / "dense_cnn_model1.toml")
    parsed = _dense_cnn().parse_model1_config(config.model.config)

    assert config.run.output_dir == ROOT / "runs" / "dense_cnn_model1"
    assert config.selfplay.checkpoint_pointer == ROOT / "data" / "checkpoints" / "dense_cnn_model1_latest.txt"
    assert config.checkpoint.resume_from == ROOT / "data" / "checkpoints" / "dense_cnn_model1_latest.txt"
    assert parsed.selfplay.search_visits == 128
    assert parsed.selfplay.samples_per_epoch == 4096
    assert parsed.selfplay.progressive_widening_initial_actions == 128
    assert parsed.selfplay.progressive_widening_child_initial_actions == 32
    assert parsed.selfplay.progressive_widening_candidate_actions == 192
    assert parsed.selfplay.progressive_widening_growth_interval == pytest.approx(40.0)
    assert parsed.selfplay.progressive_widening_growth_base == pytest.approx(1.3)
    assert parsed.selfplay.mcts_evaluation_cache_max_states == 131_072
    assert parsed.selfplay.mcts_active_root_limit == 512
    assert parsed.samples.train_sample_count == 4096
    assert parsed.samples.capacity >= 200_000
    assert parsed.evaluation.games_per_epoch == 64
    assert parsed.evaluation.sealbot_variant == "best"
    assert parsed.evaluation.sealbot_time_limit == pytest.approx(0.05)
    assert parsed.evaluation.require_sealbot is True
    assert parsed.performance.target_selfplay_positions_per_second == pytest.approx(128.0)
    assert parsed.performance.selfplay_probe_positions >= max(parsed.performance.selfplay_batch_candidates)
    assert parsed.performance.inference_batch_candidates[-1] <= 1024
    assert parsed.performance.selfplay_batch_candidates == (2048,)
    assert parsed.performance.mcts_virtual_batch_candidates == (4,)
    assert max(parsed.performance.training_batch_candidates) <= 256


def test_training_overrides_wire_dense_cnn_pipeline_components(tmp_path: Path) -> None:
    from hexo_train.components import ComponentOverrides

    ctx, components = _build_dense_components(tmp_path)
    overrides = _dense_cnn_plugin().training_component_overrides(
        defaults=components.shared.defaults,
        config=ctx.config.model.config,
        shared=components.shared,
        model=components.model.model,
    )

    assert isinstance(overrides, ComponentOverrides)
    assert overrides.trainer is not None
    assert hasattr(overrides.trainer, "train_passes")
    assert overrides.sample_finalizer is not None
    assert hasattr(overrides.sample_finalizer, "finalize")
    assert overrides.checkpoint_saver is not None
    assert hasattr(overrides.checkpoint_saver, "save")
    assert overrides.checkpoint_loader is not None
    assert hasattr(overrides.checkpoint_loader, "load")
    assert overrides.optimizer is not None


def test_dense_cnn_player_constructs_with_slots(tmp_path: Path) -> None:
    pytest.importorskip("hexo_utils._rust")
    player_module = importlib.import_module("hexo_models.dense_cnn.player")
    ctx, components = _build_dense_components(tmp_path)
    _ = ctx

    player = player_module.DenseCNNPlayer(
        identity_id="dense-test",
        model=components.model.model,
        trainer=components.model.trainer,
    )

    assert player.identity.player_id == "dense-test"
    assert player.inference is not None


def test_checkpoint_saver_writes_model_and_optimizer_state(tmp_path: Path) -> None:
    ctx, components = _build_dense_components(tmp_path)
    saver = components.model.checkpoint_saver
    assert saver is not None

    path = Path(saver.save(name="checkpoint_state_contract", ctx=ctx, components=components))

    assert path.exists()
    checkpoint = _torch_load(path)
    assert "note" not in checkpoint
    assert checkpoint.get("model") != ctx.config.model.name

    model_state = _required_mapping(checkpoint, "model_state_dict", "model_state")
    optimizer_state = _required_mapping(
        checkpoint,
        "optimizer_state_dict",
        "optimizer_state",
    )
    sample_buffer_state = _required_mapping(checkpoint, "sample_buffer")

    assert any(str(key).startswith("conv_in.") for key in model_state)
    assert "param_groups" in optimizer_state
    assert optimizer_state["param_groups"]
    assert sample_buffer_state["capacity"] >= 200_000
    assert "samples" in sample_buffer_state


def test_checkpoint_loader_resumes_model_and_sample_buffer_from_pointer(tmp_path: Path) -> None:
    torch = _torch()
    ctx, components = _build_dense_components(tmp_path / "source")
    trainer = components.model.trainer
    trainer.buffer.append(
        {
            "sample_id": "resume-sample",
            "turn_index": 3,
            "current_player": "player0",
            "phase": "Opening",
            "center": (0, 0),
            "stones": (),
            "policy": [((0, 0), 1.0)],
            "opp_policy": [((0, 0), 1.0)],
            "value": 0.5,
        }
    )

    with torch.no_grad():
        saved_param = next(components.model.model.parameters())
        saved_param.fill_(0.375)
        expected_param = saved_param.detach().clone()

    checkpoint_path = Path(components.model.checkpoint_saver.save(name="epoch_7", ctx=ctx, components=components))
    pointer_path = ctx.checkpoint_dir / "dense_cnn_latest.txt"
    pointer_path.write_text(checkpoint_path.name, encoding="utf-8")

    fresh_ctx, fresh_components = _build_dense_components(tmp_path / "fresh")
    result = fresh_components.model.checkpoint_loader.load(
        pointer_path,
        ctx=fresh_ctx,
        components=fresh_components,
    )

    restored_param = next(fresh_components.model.model.parameters()).detach()
    restored_buffer = fresh_components.model.trainer.buffer
    restored_sample = restored_buffer.entries[0].decode()

    assert result["status"] == "loaded"
    assert result["epoch"] == 7
    assert result["sample_count"] == 1
    assert torch.allclose(restored_param, expected_param)
    assert restored_buffer.sample_count == 1
    assert restored_sample.game_id == "resume-sample"
    assert restored_sample.value == pytest.approx(0.5)


def test_final_checkpoint_preserves_latest_epoch_for_future_resume(tmp_path: Path) -> None:
    ctx, components = _build_dense_components(tmp_path)
    components.shared.checkpoint_state = {"status": "loaded", "epoch": 11}

    latest_path = Path(components.model.checkpoint_saver.save(name="latest", ctx=ctx, components=components))
    payload = _torch_load(latest_path)

    assert payload["epoch"] == 11


def test_training_pipeline_run_records_dense_cnn_epoch_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("hexo_utils._rust")
    dense_cnn = _dense_cnn()
    assert dense_cnn.parse_model1_config({}).samples.train_sample_count == 4096

    from hexo_train.pipeline import TrainingPipeline

    ctx = TrainingPipeline().run(_write_pipeline_config(tmp_path))

    assert ctx.config.samples.train_sample_count == 2
    assert len(ctx.epoch_outputs) == 1

    epoch_diagnostic = _read_json(ctx.diagnostics_dir / "epoch_000001.json")
    assert epoch_diagnostic["status"] == "completed"
    epoch_result = epoch_diagnostic["metadata"]["result"]

    assert epoch_result["samples"]["selection"]["window_size"] == 2
    assert epoch_result["symmetries"]["metadata"]["mode"] == "random_per_training_expansion"
    for section in ("selfplay", "training", "checkpoint", "evaluation"):
        assert section in epoch_result
        _assert_real_diagnostic(epoch_result[section], section)
    assert epoch_result["checkpoint"]["pointer"]["status"] == "updated"
    assert Path(epoch_result["checkpoint"]["pointer"]["pointer_path"]).exists()

    payloads = _diagnostic_payloads(ctx.diagnostics_dir)
    policy_target_paths = _matching_path_values(
        payloads,
        key_tokens=("policy", "target"),
        base_dir=ctx.output_dir,
    )
    game_history_paths = _matching_path_values(
        payloads,
        key_tokens=("game", "history"),
        base_dir=ctx.output_dir,
    )
    if not game_history_paths:
        game_history_paths = [
            path
            for path in _matching_path_values(
                payloads,
                key_tokens=("record",),
                base_dir=ctx.output_dir,
            )
            if path.suffix == ".hxr"
        ]

    assert policy_target_paths, "dense CNN diagnostics must include policy target artifact paths"
    assert game_history_paths, "dense CNN diagnostics must include game history artifact paths"
    assert all(path.exists() for path in policy_target_paths)
    assert all(path.exists() for path in game_history_paths)


def test_finalize_samples_does_not_fake_missing_opponent_policy_with_current_policy() -> None:
    samples_module = importlib.import_module("hexo_models.dense_cnn.samples")
    selfplay_module = _dense_cnn_selfplay_module()
    first = samples_module.Model1SampleData(
        game_id="g",
        turn_index=0,
        current_player="player0",
        phase="Opening",
        center=(0, 0),
        stones=(),
        legal_action_ids=(1, 2),
        policy=((1, 1.0),),
    )
    last = samples_module.Model1SampleData(
        game_id="g",
        turn_index=1,
        current_player="player0",
        phase="FirstStone",
        center=(0, 0),
        stones=(),
        legal_action_ids=(1, 2),
        policy=((2, 1.0),),
    )

    finalized = selfplay_module._finalize_game_samples(
        [("player0", first, 0.25), ("player0", last, 0.5)],
        winner="player0",
        horizons=(),
    )

    assert finalized[0].opp_policy == ()
    assert finalized[0].metadata["opp_policy_source"] == "none"
    assert finalized[1].opp_policy == ()
    assert finalized[1].metadata["opp_policy_source"] == "none"


def test_selfplay_records_only_sample_budget_with_mcts_and_rolls_out_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_cnn = _dense_cnn()
    selfplay_module = _dense_cnn_selfplay_module()
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    samples_module = importlib.import_module("hexo_models.dense_cnn.samples")
    engine = importlib.import_module("hexo_engine")

    ctx, components = _build_dense_components(
        tmp_path,
        _small_model_config(
            {
                "selfplay": {
                    "samples_per_epoch": 2,
                    "search_visits": 1,
                    "active_games": 4,
                    "max_actions": 2,
                },
                "debug": {
                    "write_game_history": False,
                    "write_policy_targets": False,
                    "write_sample_previews": False,
                    "preview_games": 0,
                },
            }
        ),
    )
    mcts_batch_sizes: list[int] = []
    rollout_batch_sizes: list[int] = []
    sampled_turns: list[tuple[str, int, int]] = []

    def fake_run_batched_mcts(root_states: Sequence[object], inference: object, **kwargs: object) -> list[Any]:
        _ = inference
        mcts_batch_sizes.append(len(root_states))
        return [
            mcts_module.SearchResult(
                action_id=int(engine.legal_action_ids(state)[0]),
                visit_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                root_value=0.0,
                visits=int(kwargs["visits"]),
            )
            for state in root_states
        ]

    def fake_rollout(playable: list[dict[str, Any]], **_kwargs: object) -> list[int]:
        rollout_batch_sizes.append(len(playable))
        return [int(engine.legal_action_ids(game["state"])[0]) for game in playable]

    def fake_sample_from_state(
        state: object,
        *,
        game_id: str,
        turn_index: int,
        policy: Mapping[int, float],
        value: float,
        metadata: Mapping[str, Any],
    ) -> Any:
        legal_ids = tuple(int(item) for item in engine.legal_action_ids(state))
        action_id = next(iter(policy))
        assert int(action_id) in legal_ids, "sample must be captured before the selected action mutates the state"
        sampled_turns.append((game_id, turn_index, len(legal_ids)))
        return samples_module.Model1SampleData(
            game_id=game_id,
            turn_index=turn_index,
            current_player="player0",
            phase="Opening",
            center=(0, 0),
            stones=(),
            legal_action_ids=legal_ids,
            policy=tuple((int(key), float(weight)) for key, weight in policy.items()),
            value=float(value),
            metadata=dict(metadata),
        )

    monkeypatch.setattr(selfplay_module, "run_batched_mcts", fake_run_batched_mcts)
    monkeypatch.setattr(selfplay_module, "_policy_rollout_actions", fake_rollout)
    monkeypatch.setattr(selfplay_module, "sample_from_state", fake_sample_from_state)

    result = selfplay_module.generate_selfplay_epoch(
        ctx=ctx,
        components=components,
        epoch=1,
        games_per_epoch=4,
    )

    assert result["samples_added"] == 2
    assert result["searched_positions"] == 2
    assert result["mcts_simulations"] == 2
    assert result["mcts_search_elapsed_seconds"] >= 0.0
    assert result["positions_per_second"] == result["end_to_end_positions_per_second"]
    assert result["end_to_end_positions_per_second"] <= result["search_positions_per_second"]
    assert mcts_batch_sizes == [2]
    assert len(sampled_turns) == 2
    assert rollout_batch_sizes, "non-sample tail moves should use policy rollout instead of MCTS"


def test_selfplay_rejects_under_counted_mcts_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selfplay_module = _dense_cnn_selfplay_module()
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    engine = importlib.import_module("hexo_engine")
    ctx, components = _build_dense_components(
        tmp_path,
        _small_model_config(
            {
                "selfplay": {
                    "samples_per_epoch": 1,
                    "search_visits": 8,
                    "active_games": 1,
                    "max_actions": 2,
                }
            }
        ),
    )

    def fake_run_batched_mcts(root_states: Sequence[object], inference: object, **_kwargs: object) -> list[Any]:
        _ = inference
        return [
            mcts_module.SearchResult(
                action_id=int(engine.legal_action_ids(state)[0]),
                visit_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                root_value=0.0,
                visits=7,
            )
            for state in root_states
        ]

    monkeypatch.setattr(selfplay_module, "run_batched_mcts", fake_run_batched_mcts)

    with pytest.raises(RuntimeError, match="expected exactly 8"):
        selfplay_module.generate_selfplay_epoch(
            ctx=ctx,
            components=components,
            epoch=1,
            games_per_epoch=1,
        )


def test_required_sealbot_evaluation_fails_fast_when_adapter_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEALBOT_PATH", raising=False)
    ctx, components = _build_dense_components(
        tmp_path,
        _small_model_config(
            {
                "evaluation": {
                    "games_per_epoch": 1,
                    "sealbot_variant": "best",
                    "sealbot_time_limit": 0.05,
                    "require_sealbot": True,
                }
            }
        ),
    )

    with pytest.raises(RuntimeError, match="Required SealBot evaluation is unavailable"):
        _dense_cnn_plugin().evaluate_epoch(ctx=ctx, components=components, epoch=1)

    payload = _read_json(ctx.diagnostics_dir / "dense_cnn.evaluation.epoch_000001.json")
    assert payload["status"] == "unavailable"
    assert payload["required"] is True
