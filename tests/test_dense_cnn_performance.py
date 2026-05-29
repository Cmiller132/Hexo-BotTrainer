from __future__ import annotations

import importlib
import inspect
import json
import struct
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


def _payload_visit_policy(payload: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    if "visit_policy" in payload:
        return tuple((int(action), float(weight)) for action, weight in payload["visit_policy"])
    actions = payload["visit_policy_action_ids_bytes"]
    weights = payload["visit_policy_weights_bytes"]
    count = int(payload["visit_policy_count"])
    return tuple(
        (
            int(struct.unpack_from("I", actions, index * 4)[0]),
            float(struct.unpack_from("f", weights, index * 4)[0]),
        )
        for index in range(count)
    )


def _engine_with_rust() -> Any:
    engine = importlib.import_module("hexo_engine")
    try:
        engine.engine_metadata()
    except engine.EngineUnavailableError as exc:
        pytest.skip(f"hexo_engine Rust bridge is unavailable: {exc}")
    return engine


def _skip_without_direct_state_rust(rust_bridge: Any) -> None:
    try:
        capabilities = rust_bridge.capabilities()
    except RuntimeError as exc:
        pytest.skip(f"dense_cnn Rust accelerator is not available: {exc}")
    if capabilities.get("state_source") != "direct_engine_state":
        pytest.skip("dense_cnn direct-state Rust accelerator is not available in this checkout")


def _public_attr(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(
        f"{module.__name__} public API must expose one of {names!r}; "
        f"available public names: {_public_names(module)}"
    )


def _public_names(module: Any) -> list[str]:
    return sorted(name for name in dir(module) if not name.startswith("_"))


def _small_config() -> Any:
    dense_cnn = _dense_cnn()
    return dense_cnn.parse_model1_config(
        {
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
            "selfplay": {
                "samples_per_epoch": 2,
                "search_visits": 1,
                "max_actions": 2,
                "worker_count": 1,
            },
            "performance": {
                "calibrate": True,
                "target_selfplay_positions_per_second": 128,
                "inference_batch_candidates": (1, 2),
                "training_batch_candidates": (1, 2),
                "selfplay_batch_candidates": (1, 2),
                "mcts_virtual_batch_candidates": (1, 2),
                "selfplay_probe_positions": 4,
                "probe_batches": 1,
            },
        }
    )


def _small_model() -> Any:
    _torch()
    dense_cnn = _dense_cnn()
    model = dense_cnn.Model1Network(
        channels=4,
        blocks=1,
        dropout=0.0,
    )
    model.eval()
    return model


def _assert_batch_calibration_payload(
    payload: Mapping[str, Any],
    *,
    inference_candidates: Sequence[int],
    training_candidates: Sequence[int],
) -> None:
    assert payload.get("status") == "completed"
    assert payload["selected_inference_batch_size"] in set(inference_candidates)
    assert payload["selected_training_batch_size"] in set(training_candidates)
    assert _measured_batch_sizes(payload, "inference") == set(inference_candidates)
    assert _measured_batch_sizes(payload, "training") == set(training_candidates)

    for key in (
        "inference_positions_per_second",
        "training_samples_per_second",
    ):
        assert key in payload, f"calibration payload missing measured throughput key {key!r}"
        assert isinstance(payload[key], int | float), f"{key} must be numeric"
        assert payload[key] > 0.0, f"{key} must be measured as positive throughput"

    assert _completed_throughput(payload, "inference") > 0.0
    assert _completed_throughput(payload, "training") > 0.0
    assert "measured_selfplay_positions_per_second" in payload
    assert isinstance(payload["measured_selfplay_positions_per_second"], int | float)


def _measured_batch_sizes(payload: Mapping[str, Any], section: str) -> set[int]:
    measurements = payload.get(section)
    assert isinstance(measurements, Sequence), f"calibration payload missing {section!r} measurements"
    return {
        int(item["batch_size"])
        for item in measurements
        if isinstance(item, Mapping) and "batch_size" in item
    }


def _completed_throughput(payload: Mapping[str, Any], section: str) -> float:
    measurements = payload.get(section)
    assert isinstance(measurements, Sequence), f"calibration payload missing {section!r} measurements"
    values = [
        float(item["positions_per_second"])
        for item in measurements
        if isinstance(item, Mapping)
        and item.get("status") == "completed"
        and "positions_per_second" in item
    ]
    assert values, f"{section} measurements must include completed positions_per_second values"
    return max(values)


def _call_calibration_api(api: Any, *, model: Any, config: Any) -> Mapping[str, Any]:
    signature = inspect.signature(api)
    kwargs: dict[str, Any] = {
        "model": model,
        "config": config,
    }
    if "inference_batch_candidates" in signature.parameters:
        kwargs["inference_batch_candidates"] = (1, 2)
    if "training_batch_candidates" in signature.parameters:
        kwargs["training_batch_candidates"] = (1, 2)
    result = api(**kwargs)
    assert isinstance(result, Mapping)
    return result


def _write_pipeline_config(tmp_path: Path) -> Path:
    output_dir = (tmp_path / "pipeline-run").as_posix()
    config_path = tmp_path / "dense_cnn_performance.toml"
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
                "train_sample_count = 1",
                "compression_level = 1",
                "",
                "[model.config.selfplay]",
                "samples_per_epoch = 1",
                "search_visits = 1",
                "max_actions = 2",
                "temperature = 1.0",
                "worker_count = 1",
                "",
                "[model.config.evaluation]",
                "games_per_epoch = 0",
                'sealbot_variant = "best"',
                "sealbot_time_limit = 0.001",
                "max_actions = 2",
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
                "write_game_history = false",
                "write_policy_targets = false",
                "write_sample_previews = false",
                "preview_games = 0",
                "",
                "[run]",
                f'output_dir = "{output_dir}"',
                "seed = 13",
                "",
                "[loop]",
                "epochs = 1",
                "",
                "[selfplay]",
                "games_per_epoch = 1",
                "",
                "[samples]",
                "train_sample_count = 1",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping), f"{path} must contain a JSON object"
    return payload


def _compact_sample(index: int) -> Any:
    dense_cnn = _dense_cnn()
    encode_compact_sample = _public_attr(dense_cnn, "encode_compact_sample")
    q = index % 2
    return encode_compact_sample(
        {
            "sample_id": f"perf-sample-{index}",
            "turn_index": index,
            "current_player": "player0",
            "phase": "Opening",
            "center": (0, 0),
            "stones": (),
            "policy": [((q, 0), 1.0)],
            "opp_policy": [((q, 0), 1.0)],
            "value": 0.0,
        }
    )


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise AssertionError(f"{type(value).__name__} must expose one of {names!r}: {value!r}")


def test_calibration_api_selects_batches_and_reports_measured_throughput() -> None:
    _engine_with_rust()
    dense_cnn = _dense_cnn()
    calibrate_dense_cnn = _public_attr(dense_cnn, "calibrate_dense_cnn")

    result = _call_calibration_api(
        calibrate_dense_cnn,
        model=_small_model(),
        config=_small_config(),
    )

    _assert_batch_calibration_payload(
        result,
        inference_candidates=(1, 2),
        training_candidates=(1, 2),
    )


def test_training_pipeline_writes_calibration_json_artifact_when_enabled(tmp_path: Path) -> None:
    _engine_with_rust()
    pytest.importorskip("hexo_utils._rust")
    from hexo_train.pipeline import TrainingPipeline

    ctx = TrainingPipeline().run(_write_pipeline_config(tmp_path))

    calibration_paths = sorted(ctx.diagnostics_dir.glob("*calibration*.json"))
    assert calibration_paths, "enabled dense CNN calibration must write a JSON diagnostics artifact"

    payloads = [_read_json(path) for path in calibration_paths]
    calibration_payload = next(
        (
            payload
            for payload in payloads
            if "selected_inference_batch_size" in payload
            and "selected_training_batch_size" in payload
        ),
        None,
    )
    assert calibration_payload is not None, (
        "calibration artifact must include selected inference/training batch sizes; "
        f"found payloads: {payloads}"
    )
    _assert_batch_calibration_payload(
        calibration_payload,
        inference_candidates=(1, 2),
        training_candidates=(1, 2),
    )


def test_build_benchmark_report_targets_128_pps_and_does_not_fabricate_success() -> None:
    dense_cnn = _dense_cnn()
    build_benchmark_report = _public_attr(dense_cnn, "build_benchmark_report")

    report = build_benchmark_report(
        config=_small_config(),
        measurements={
            "selfplay_positions_per_second": 127.0,
            "inference_positions_per_second": 1_000_000.0,
            "training_samples_per_second": 1_000_000.0,
        },
    )

    assert isinstance(report, Mapping)
    assert report["target_selfplay_positions_per_second"] == pytest.approx(128.0)
    assert isinstance(report["meets_target"], bool)
    assert report["meets_target"] is False
    measured = report.get(
        "selfplay_positions_per_second",
        report.get("measured_selfplay_positions_per_second"),
    )
    assert measured == pytest.approx(127.0)


def test_calibration_success_requires_exact_configured_mcts_visits(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_cnn = _dense_cnn()
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    calibrate_dense_cnn = _public_attr(dense_cnn, "calibrate_dense_cnn")
    config = dense_cnn.parse_model1_config(
        {
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
            "selfplay": {
                "search_visits": 128,
            },
            "performance": {
                "calibrate": True,
                "target_selfplay_positions_per_second": 128,
                "inference_batch_candidates": (2,),
                "training_batch_candidates": (2,),
                "selfplay_batch_candidates": (4,),
                "mcts_virtual_batch_candidates": (8,),
                "probe_batches": 1,
                "selfplay_probe_positions": 4,
            },
        }
    )

    monkeypatch.setattr(
        performance_module,
        "_benchmark_inference",
        lambda *_args, **_kwargs: [
            {
                "status": "completed",
                "batch_size": 2,
                "positions_per_second": 1_000_000.0,
            }
        ],
    )
    monkeypatch.setattr(
        performance_module,
        "_benchmark_training",
        lambda *_args, **_kwargs: [
            {
                "status": "completed",
                "batch_size": 2,
                "positions_per_second": 1_000_000.0,
            }
        ],
    )
    monkeypatch.setattr(
        performance_module,
        "_benchmark_selfplay",
        lambda *_args, **_kwargs: [
            {
                "status": "completed",
                "visits": 128,
                "selfplay_batch_size": 4,
                "mcts_virtual_batch_size": 8,
                "positions": 4,
                "searched_positions": 4,
                "recorded_positions": 4,
                "mcts_simulations": 127 * 4,
                "exact_visit_results": 3,
                "all_searches_exact": False,
                "positions_per_second": 10_000.0,
            }
        ],
    )

    report = calibrate_dense_cnn(model=_small_model(), config=config)

    assert report["target_mcts_simulations_per_position"] == 128
    assert report["measured_selfplay_positions_per_second"] == pytest.approx(10_000.0)
    assert report["all_searches_exact"] is False
    assert report["meets_target"] is False


def test_calibration_restores_training_probe_before_selfplay(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    baseline = next(model.parameters()).detach().clone()
    seen: dict[str, bool] = {"selfplay": False}

    monkeypatch.setattr(
        performance_module,
        "_benchmark_inference",
        lambda *_args, **_kwargs: [
            {
                "status": "completed",
                "batch_size": 2,
                "positions_per_second": 1_000_000.0,
            }
        ],
    )

    def fake_training(probe_model: Any, *, optimizer: Any | None, **_kwargs: object) -> list[dict[str, Any]]:
        with torch.no_grad():
            next(probe_model.parameters()).add_(10.0)
        if optimizer is not None:
            optimizer.state[next(probe_model.parameters())]["sentinel"] = torch.tensor(1.0)
        return [
            {
                "status": "completed",
                "batch_size": 2,
                "positions_per_second": 1_000_000.0,
            }
        ]

    def fake_selfplay(probe_model: Any, **_kwargs: object) -> list[dict[str, Any]]:
        torch.testing.assert_close(next(probe_model.parameters()).detach(), baseline)
        seen["selfplay"] = True
        return [
            {
                "status": "completed",
                "visits": 1,
                "selfplay_batch_size": 2,
                "mcts_virtual_batch_size": 1,
                "positions": 4,
                "searched_positions": 4,
                "recorded_positions": 4,
                "mcts_simulations": 4,
                "exact_visit_results": 4,
                "all_searches_exact": True,
                "positions_per_second": 1_000_000.0,
            }
        ]

    monkeypatch.setattr(performance_module, "_benchmark_training", fake_training)
    monkeypatch.setattr(performance_module, "_benchmark_selfplay", fake_selfplay)

    result = performance_module.calibrate_dense_cnn(
        model=model,
        optimizer=optimizer,
        config=_small_config(),
    )

    assert seen["selfplay"] is True
    assert result["meets_target"] is True
    torch.testing.assert_close(next(model.parameters()).detach(), baseline)
    assert optimizer.state == {}


def test_selfplay_benchmark_counts_every_root_as_its_own_128_sim_search(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_cnn = _dense_cnn()
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    engine = _engine_with_rust()

    calls: list[int] = []

    def fake_run_batched_mcts(root_states: Sequence[object], inference: object, **kwargs: object) -> list[Any]:
        _ = inference
        visits = int(kwargs["visits"])
        calls.append(len(root_states))
        return [
            mcts_module.SearchResult(
                action_id=int(engine.legal_action_ids(state)[0]),
                visit_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                root_value=0.0,
                visits=visits,
            )
            for state in root_states
        ]

    monkeypatch.setattr(mcts_module, "run_batched_mcts", fake_run_batched_mcts)

    result = performance_module._benchmark_selfplay_setting(
        inference=object(),
        config=_small_config(),
        selfplay_batch_size=4,
        virtual_batch_size=8,
        visits=128,
        probe_positions=4,
    )

    assert calls == [4], "benchmark must not collapse duplicate roots before counting positions"
    assert result["searched_positions"] == 4
    assert result["recorded_positions"] == 4
    assert result["mcts_simulations"] == 4 * 128
    assert result["all_searches_exact"] is True


def test_selfplay_benchmark_caps_last_batch_to_probe_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    engine = _engine_with_rust()

    calls: list[int] = []

    def fake_run_batched_mcts(root_states: Sequence[object], inference: object, **kwargs: object) -> list[Any]:
        _ = inference
        visits = int(kwargs["visits"])
        calls.append(len(root_states))
        return [
            mcts_module.SearchResult(
                action_id=int(engine.legal_action_ids(state)[0]),
                visit_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                root_value=0.0,
                visits=visits,
            )
            for state in root_states
        ]

    monkeypatch.setattr(mcts_module, "run_batched_mcts", fake_run_batched_mcts)

    result = performance_module._benchmark_selfplay_setting(
        inference=object(),
        config=_small_config(),
        selfplay_batch_size=4,
        virtual_batch_size=8,
        visits=128,
        probe_positions=2,
    )

    assert calls == [2]
    assert result["searched_positions"] == 2
    assert result["recorded_positions"] == 2
    assert result["mcts_simulations"] == 2 * 128
    assert result["all_searches_exact"] is True


def test_dense_cnn_mcts_virtual_batch_four_survives_production_root_count() -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")

    resolved = mcts_module._resolve_virtual_batch_size(
        root_count=1024,
        visits=128,
        virtual_batch_size=4,
    )

    assert resolved == 4


def test_dense_cnn_input_mask_topk_matches_bruteforce_legal_topk() -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    constants = importlib.import_module("hexo_models.dense_cnn.constants")
    inference_module = importlib.import_module("hexo_models.dense_cnn.inference")

    row_count = 3
    candidates = 5
    generator = torch.Generator().manual_seed(1234)
    policy = torch.randn(row_count, dense_cnn.BOARD_AREA, generator=generator)
    inputs = torch.zeros(
        row_count,
        dense_cnn.INPUT_CHANNELS,
        dense_cnn.BOARD_SIZE,
        dense_cnn.BOARD_SIZE,
    )
    legal_mask = torch.ones(row_count, dense_cnn.BOARD_AREA, dtype=torch.bool)
    illegal_by_row = (
        (0, 3, 9),
        (1, 2, 4, 8, 16, 32, 64),
        (5, 7),
    )
    for row, illegal in enumerate(illegal_by_row):
        legal_mask[row, list(illegal)] = False
        policy[row, list(illegal)] = 100.0 - torch.arange(len(illegal), dtype=policy.dtype)
    inputs[:, constants.PLANE_LEGAL] = legal_mask.reshape(
        row_count,
        dense_cnn.BOARD_SIZE,
        dense_cnn.BOARD_SIZE,
    ).to(dtype=inputs.dtype)

    priors, flats, offsets = inference_module._topk_legal_priors_from_input_mask(
        policy_batch=policy,
        inputs=inputs,
        max_candidates=candidates,
    )

    expected_priors: list[torch.Tensor] = []
    expected_flats: list[torch.Tensor] = []
    expected_offsets = [0]
    for row in range(row_count):
        legal = torch.where(legal_mask[row])[0]
        values = policy[row, legal]
        top_values, top_ordinals = torch.topk(values, k=candidates, largest=True, sorted=True)
        expected_priors.append(torch.softmax(top_values, dim=0))
        expected_flats.append(legal[top_ordinals].to(dtype=torch.int64))
        expected_offsets.append(expected_offsets[-1] + candidates)

    torch.testing.assert_close(priors, torch.cat(expected_priors), rtol=1.0e-6, atol=1.0e-6)
    assert flats.tolist() == torch.cat(expected_flats).tolist()
    assert offsets == expected_offsets


def test_dense_cnn_input_mask_topk_uses_crop_legal_counts_without_padding() -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    constants = importlib.import_module("hexo_models.dense_cnn.constants")
    inference_module = importlib.import_module("hexo_models.dense_cnn.inference")

    row_count = 3
    policy = torch.arange(row_count * dense_cnn.BOARD_AREA, dtype=torch.float32).reshape(
        row_count,
        dense_cnn.BOARD_AREA,
    )
    inputs = torch.zeros(
        row_count,
        dense_cnn.INPUT_CHANNELS,
        dense_cnn.BOARD_SIZE,
        dense_cnn.BOARD_SIZE,
    )
    legal_by_row = ((1, 5, 9, 13, 17), (2, 4), ())
    for row, flats in enumerate(legal_by_row):
        for flat in flats:
            inputs[row, constants.PLANE_LEGAL, flat // dense_cnn.BOARD_SIZE, flat % dense_cnn.BOARD_SIZE] = 1.0

    priors, flats, offsets = inference_module._topk_legal_priors_from_input_mask(
        policy_batch=policy,
        inputs=inputs,
        max_candidates=4,
        crop_legal_counts=tuple(len(row) for row in legal_by_row),
    )

    assert offsets == [0, 4, 6, 6]
    assert flats.tolist() == [17, 13, 9, 5, 4, 2]
    assert float(priors[:4].sum()) == pytest.approx(1.0)
    assert float(priors[4:].sum()) == pytest.approx(1.0)


def test_dense_cnn_rust_batch_input_encoder_matches_python_sample_encoder() -> None:
    torch = _torch()
    engine = importlib.import_module("hexo_engine")
    engine_types = importlib.import_module("hexo_engine.types")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    samples_module = importlib.import_module("hexo_models.dense_cnn.samples")
    dense_cnn = _dense_cnn()
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    states = []
    for step in range(10):
        states.append(engine.clone_state(state))
        if engine.terminal(state) is not None:
            break
        legal = tuple(int(item) for item in engine.legal_action_ids(state))
        if not legal:
            break
        action_id = legal[(step * 7 + 3) % len(legal)]
        engine.apply_action(
            state,
            engine.PlacementAction(engine_types.unpack_coord_id(action_id)),
        )

    payload = rust_bridge.model1_batch_inputs(states)
    shape = tuple(int(item) for item in payload["shape"])
    rust_inputs = torch.frombuffer(bytearray(payload["inputs"]), dtype=torch.float32).reshape(shape)

    assert shape == (len(states), dense_cnn.INPUT_CHANNELS, dense_cnn.BOARD_SIZE, dense_cnn.BOARD_SIZE)
    for index, encoded_state in enumerate(states):
        sample = samples_module.sample_from_state(
            encoded_state,
            game_id=f"encoder-parity-{index}",
            turn_index=index,
        )
        expected = samples_module.expand_sample(sample)["input"]

        assert tuple(int(item) for item in payload["centers"][index]) == sample.center
        assert tuple(int(item) for item in payload["legal_action_ids"][index]) == sample.legal_action_ids
        torch.testing.assert_close(rust_inputs[index], expected, rtol=0.0, atol=0.0)


def test_dense_cnn_rust_capabilities_smoke() -> None:
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    rust_module = importlib.import_module("hexo_models._rust.dense_cnn")
    capabilities = rust_module.capabilities()

    assert capabilities["model_family"] == "dense_cnn"
    assert capabilities["state_source"] == "direct_engine_state"
    assert capabilities["model1_batch_inputs"] is True
    assert capabilities["model1_batched_mcts"] is True
    assert capabilities["model1_mcts_progressive_widening"] is True
    assert capabilities["model1_mcts_progressive_widening_reference"] == "Chaslot_2008_progressive_unpruning"
    assert capabilities["model1_mcts_evaluation_cache"] is True
    assert capabilities["model1_mcts_tree_reuse_session"] is True
    assert capabilities["model1_mcts_tree_reuse_reference"] == "KataGo_Search_makeMove_promote_child"
    assert capabilities["model1_mcts_lazy_staged_edges"] is True
    assert capabilities["model1_mcts_lazy_staged_edges_reference"] == "KataGo_SearchNode_children0_1_2"
    assert capabilities["model1_sample_from_state"] is True
    assert "model1_sample_from_history" not in capabilities
    assert capabilities["coordinate_encoding"] == "u32_i16_pair"


def test_dense_cnn_rust_mcts_uses_model_local_accelerator() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    expected_action = engine.action_id(engine.PlacementAction(engine.AxialCoord(0, 0)))

    def evaluator(payload: Mapping[str, Any]) -> Mapping[str, bytes]:
        rows = int(payload["shape"][0])
        offsets = tuple(int(item) for item in payload["legal_row_offsets"])
        priors = max(0, offsets[-1])
        return {
            "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
            "priors_bytes": struct.pack(f"{priors}f", *([1.0] * priors)),
        }

    first = rust_bridge.model1_batched_mcts(
        [state],
        visits=3,
        c_puct=1.5,
        temperature=0.0,
        seed=42,
        evaluator=evaluator,
        virtual_batch_size=2,
    )[0]
    second = rust_bridge.model1_batched_mcts(
        [state],
        visits=3,
        c_puct=1.5,
        temperature=0.0,
        seed=42,
        evaluator=evaluator,
        virtual_batch_size=2,
    )[0]

    assert int(first["action_id"]) == expected_action
    assert int(first["action_id"]) == int(second["action_id"])
    assert int(first["visits"]) == 3
    assert sum(weight for _action_id, weight in _payload_visit_policy(first)) == pytest.approx(1.0)


def test_dense_cnn_rust_mcts_initial_progressive_widening_uses_top_128_priors() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(0, 0)))
    batch_inputs = rust_bridge.model1_batch_inputs([state])
    legal_action_ids = tuple(batch_inputs["legal_action_ids"][0])
    legal_flat_indices = tuple(int(item) for item in batch_inputs["legal_flat_indices"][0])
    assert len(legal_action_ids) > 128
    selected_flats = legal_flat_indices[-128:]

    def evaluator(payload: Mapping[str, Any]) -> Mapping[str, object]:
        rows = int(payload["shape"][0])
        return {
            "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
            "priors_bytes": struct.pack(
                "128f",
                *(float(index + 1) for index in range(128)),
            ),
            "selected_flat_indices_bytes": struct.pack("128q", *selected_flats),
            "selected_row_offsets": (0, 128),
        }

    result = rust_bridge.model1_batched_mcts(
        [state],
        visits=1,
        c_puct=1.5,
        temperature=0.0,
        seed=11,
        evaluator=evaluator,
        virtual_batch_size=1,
        progressive_widening_initial_actions=128,
        progressive_widening_child_initial_actions=32,
        progressive_widening_candidate_actions=128,
        progressive_widening_growth_interval=40.0,
        progressive_widening_growth_base=1.3,
    )[0]

    returned_action_ids = {action_id for action_id, _weight in _payload_visit_policy(result)}
    assert returned_action_ids == {int(legal_action_ids[-1])}
    root_diagnostics = dict(result["diagnostics"]["root"])
    assert int(root_diagnostics["root_active_edges"]) == 1
    assert int(root_diagnostics["root_hidden_priors"]) == len(legal_action_ids) - 1
    assert int(result["visits"]) == 1
    assert sum(weight for _action_id, weight in _payload_visit_policy(result)) == pytest.approx(1.0)


def test_dense_cnn_rust_mcts_keeps_authoritative_legal_actions_outside_dense_crop() -> None:
    engine = importlib.import_module("hexo_engine")
    engine_types = importlib.import_module("hexo_engine.types")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    for _ in range(5):
        legal = tuple(int(item) for item in engine.legal_action_ids(state))
        action_id = max(
            legal,
            key=lambda item: (
                engine_types.unpack_coord_id(item).q,
                -abs(engine_types.unpack_coord_id(item).r),
            ),
        )
        engine.apply_action(
            state,
            engine.PlacementAction(engine_types.unpack_coord_id(action_id)),
        )

    payload = rust_bridge.model1_batch_inputs([state])
    in_crop_legal = {int(item) for item in payload["legal_action_ids"][0]}
    all_legal = {int(item) for item in engine.legal_action_ids(state)}
    out_of_crop_legal = all_legal - in_crop_legal
    assert out_of_crop_legal

    def evaluator(payload: Mapping[str, Any]) -> Mapping[str, object]:
        rows = int(payload["shape"][0])
        return {
            "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
            "priors_bytes": b"",
            "selected_flat_indices_bytes": b"",
            "selected_row_offsets": tuple(0 for _ in range(rows + 1)),
        }

    result = rust_bridge.model1_batched_mcts(
        [state],
        visits=1,
        c_puct=1.5,
        temperature=0.0,
        seed=17,
        evaluator=evaluator,
        virtual_batch_size=1,
        progressive_widening_initial_actions=4,
        progressive_widening_child_initial_actions=4,
        progressive_widening_candidate_actions=4,
        progressive_widening_growth_interval=40.0,
        progressive_widening_growth_base=1.3,
    )[0]

    assert int(result["action_id"]) in out_of_crop_legal
    root_diagnostics = dict(result["diagnostics"]["root"])
    assert int(root_diagnostics["root_active_edges"]) == 1
    assert int(root_diagnostics["root_hidden_priors"]) == len(all_legal) - 1
    assert int(result["visits"]) == 1


def test_dense_cnn_rust_mcts_keeps_out_of_crop_legal_actions_hidden_when_topk_is_full() -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    constants = importlib.import_module("hexo_models.dense_cnn.constants")
    engine = importlib.import_module("hexo_engine")
    engine_types = importlib.import_module("hexo_engine.types")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    for _ in range(5):
        legal = tuple(int(item) for item in engine.legal_action_ids(state))
        action_id = max(
            legal,
            key=lambda item: (
                engine_types.unpack_coord_id(item).q,
                -abs(engine_types.unpack_coord_id(item).r),
            ),
        )
        engine.apply_action(
            state,
            engine.PlacementAction(engine_types.unpack_coord_id(action_id)),
        )

    payload = rust_bridge.model1_batch_inputs([state])
    in_crop_legal = {int(item) for item in payload["legal_action_ids"][0]}
    legal_flat_indices = tuple(int(item) for item in payload["legal_flat_indices"][0])
    all_legal = {int(item) for item in engine.legal_action_ids(state)}
    assert all_legal - in_crop_legal
    assert len(legal_flat_indices) >= 4

    selected_flats = legal_flat_indices[:4]

    def evaluator(payload: Mapping[str, Any]) -> Mapping[str, object]:
        rows = int(payload["shape"][0])
        assert int(payload["max_prior_candidates"]) == 4
        assert bool(payload["legal_mask_from_inputs"]) is True
        assert type(payload["inputs"]).__name__ == "memoryview"
        encoded = torch.frombuffer(payload["inputs"], dtype=torch.float16).reshape(tuple(int(item) for item in payload["shape"]))
        legal_counts = encoded[:, constants.PLANE_LEGAL].reshape(rows, dense_cnn.BOARD_AREA).count_nonzero(dim=1)
        assert tuple(int(item) for item in payload["crop_legal_counts"]) == tuple(
            int(item) for item in legal_counts.tolist()
        )
        return {
            "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
            "priors_bytes": struct.pack("4f", 0.4, 0.3, 0.2, 0.1),
            "selected_flat_indices_bytes": struct.pack("4q", *selected_flats),
            "selected_row_offsets": (0, 4),
        }

    result = rust_bridge.model1_batched_mcts(
        [state],
        visits=1,
        c_puct=1.5,
        temperature=0.0,
        seed=19,
        evaluator=evaluator,
        virtual_batch_size=1,
        progressive_widening_initial_actions=4,
        progressive_widening_child_initial_actions=4,
        progressive_widening_candidate_actions=4,
        progressive_widening_growth_interval=40.0,
        progressive_widening_growth_base=1.3,
    )[0]

    root_diagnostics = dict(result["diagnostics"]["root"])
    active = int(root_diagnostics["root_active_edges"])
    hidden = int(root_diagnostics["root_hidden_priors"])
    assert active == 1
    assert hidden == len(all_legal) - active
    assert hidden > len(in_crop_legal) - active


def test_dense_cnn_rust_mcts_evaluation_cache_keys_candidate_limit() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    cache = rust_bridge.model1_new_mcts_evaluation_cache(max_states=100)
    calls: list[int] = []

    def evaluator(payload: Mapping[str, Any]) -> Mapping[str, object]:
        rows = int(payload["shape"][0])
        calls.append(int(payload["max_prior_candidates"]))
        return {
            "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
            "priors_bytes": b"",
            "selected_flat_indices_bytes": b"",
            "selected_row_offsets": tuple(0 for _ in range(rows + 1)),
        }

    for candidate_limit in (1, 1, 3, 3):
        rust_bridge.model1_batched_mcts(
            [state],
            visits=1,
            c_puct=1.5,
            temperature=0.0,
            seed=23,
            evaluator=evaluator,
            virtual_batch_size=1,
            progressive_widening_initial_actions=1,
            progressive_widening_child_initial_actions=1,
            progressive_widening_candidate_actions=candidate_limit,
            progressive_widening_growth_interval=40.0,
            progressive_widening_growth_base=1.3,
            evaluation_cache=cache,
        )

    assert calls[:2] == [1, 1]
    assert calls[2:4] == [3, 3]
    assert calls[4:] == []


def test_dense_cnn_mcts_session_reuses_tree_and_reports_exact_delta_visits() -> None:
    engine = importlib.import_module("hexo_engine")
    engine_types = importlib.import_module("hexo_engine.types")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    class FakeInference:
        def __init__(self) -> None:
            self.rows: list[int] = []

        def evaluate_model1_payload(self, payload: Mapping[str, Any]) -> Mapping[str, object]:
            rows = int(payload["shape"][0])
            self.rows.append(rows)
            return {
                "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
                "priors_bytes": b"",
                "selected_flat_indices_bytes": b"",
                "selected_row_offsets": tuple(0 for _ in range(rows + 1)),
            }

    state = engine.new_game()
    inference = FakeInference()
    session = mcts_module.new_mcts_session(max_states=100)

    first = session.run(
        [7],
        [state],
        inference,
        visits=4,
        temperature=0.0,
        seed=1,
        virtual_batch_size=1,
        progressive_widening_initial_actions=1,
        progressive_widening_child_initial_actions=1,
        progressive_widening_candidate_actions=1,
    )[0]
    engine.apply_action(
        state,
        engine.PlacementAction(engine_types.unpack_coord_id(first.action_id)),
    )
    second = session.run(
        [7],
        [state],
        inference,
        visits=4,
        temperature=0.0,
        seed=2,
        virtual_batch_size=1,
        progressive_widening_initial_actions=1,
        progressive_widening_child_initial_actions=1,
        progressive_widening_candidate_actions=1,
    )[0]

    assert first.visits == 4
    assert second.visits == 4
    assert len(session) == 1
    assert sum(weight for _action_id, weight in first.visit_policy) == pytest.approx(1.0)
    assert sum(weight for _action_id, weight in second.visit_policy) == pytest.approx(1.0)
    assert int(second.diagnostics["batch"]["tree"]["completed_visits"]) > second.visits


def test_dense_cnn_rust_mcts_rejects_short_compact_evaluator_payloads() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    def evaluator(_payload: Mapping[str, Any]) -> Mapping[str, object]:
        return {
            "values_bytes": b"",
            "priors_bytes": b"",
            "selected_flat_indices_bytes": b"",
            "selected_row_offsets": (0,),
        }

    with pytest.raises(ValueError, match="values_bytes"):
        rust_bridge.model1_batched_mcts(
            [engine.new_game()],
            visits=1,
            c_puct=1.5,
            temperature=0.0,
            seed=29,
            evaluator=evaluator,
            virtual_batch_size=1,
            progressive_widening_candidate_actions=4,
        )


def test_dense_cnn_rust_mcts_deduplicates_duplicate_roots_without_mutating_live_state() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    state = engine.new_game()
    before = engine.to_python_state(state)
    evaluated_rows: list[int] = []

    def evaluator(payload: Mapping[str, Any]) -> Mapping[str, bytes]:
        rows = int(payload["shape"][0])
        offsets = tuple(int(item) for item in payload["legal_row_offsets"])
        priors = max(0, offsets[-1])
        evaluated_rows.append(rows)
        return {
            "values_bytes": struct.pack(f"{rows}f", *([0.0] * rows)),
            "priors_bytes": struct.pack(f"{priors}f", *([1.0] * priors)),
        }

    results = rust_bridge.model1_batched_mcts(
        [state, state],
        visits=1,
        c_puct=1.5,
        temperature=0.0,
        seed=5,
        evaluator=evaluator,
        virtual_batch_size=1,
    )

    assert len(results) == 2
    assert [int(result["visits"]) for result in results] == [1, 1]
    assert evaluated_rows and all(rows == 1 for rows in evaluated_rows)
    assert sum(evaluated_rows) < 4, "duplicate roots should share exact StateHash evaluator cache"
    assert engine.to_python_state(state) == before


def test_dense_cnn_rust_mcts_rejects_batches_above_strict_active_root_limit() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    def evaluator(_payload: Mapping[str, Any]) -> Mapping[str, bytes]:
        raise AssertionError("active-root guard should run before evaluation")

    with pytest.raises(ValueError, match="above strict limit"):
        rust_bridge.model1_batched_mcts(
            [engine.new_game(), engine.new_game(), engine.new_game()],
            visits=1,
            c_puct=1.5,
            temperature=0.0,
            seed=3,
            evaluator=evaluator,
            virtual_batch_size=1,
            active_root_limit=2,
        )


def test_dense_cnn_batched_mcts_requires_rust_and_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("dense_cnn Rust accelerator is unavailable")

    monkeypatch.setattr(rust_bridge, "model1_batched_mcts", unavailable)

    class FakeInference:
        evaluate_model1_payload = object()

        def infer_states(self, _states: Sequence[object]) -> list[Any]:
            raise AssertionError("dense CNN MCTS must not fall back to Python inference")

    with pytest.raises(RuntimeError, match="Rust accelerator is unavailable"):
        mcts_module.run_batched_mcts(
            [object()],
            FakeInference(),
            visits=1,
            temperature=0.0,
            seed=7,
            virtual_batch_size=1,
        )


def test_dense_cnn_mcts_python_boundary_delegates_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = importlib.import_module("hexo_engine")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    calls: list[Mapping[str, Any]] = []

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_batched_mcts(
        states: Sequence[object],
        *,
        visits: int,
        c_puct: float,
        temperature: float,
        seed: int,
        evaluator: object,
        virtual_batch_size: int | None,
        progressive_widening_initial_actions: int | None,
        progressive_widening_child_initial_actions: int | None,
        progressive_widening_candidate_actions: int | None,
        progressive_widening_growth_interval: float | None,
        progressive_widening_growth_base: float | None,
        root_dirichlet_alpha: float | None,
        root_exploration_fraction: float | None,
        evaluation_cache: object | None,
        active_root_limit: int | None,
    ) -> tuple[Mapping[str, Any], ...]:
        calls.append(
            {
                "states": states,
                "visits": visits,
                "c_puct": c_puct,
                "temperature": temperature,
                "seed": seed,
                "evaluator": evaluator,
                "virtual_batch_size": virtual_batch_size,
                "progressive_widening_initial_actions": progressive_widening_initial_actions,
                "progressive_widening_child_initial_actions": progressive_widening_child_initial_actions,
                "progressive_widening_candidate_actions": progressive_widening_candidate_actions,
                "progressive_widening_growth_interval": progressive_widening_growth_interval,
                "progressive_widening_growth_base": progressive_widening_growth_base,
                "root_dirichlet_alpha": root_dirichlet_alpha,
                "root_exploration_fraction": root_exploration_fraction,
                "evaluation_cache": evaluation_cache,
                "active_root_limit": active_root_limit,
            }
        )
        return (
            {
                "action_id": 17,
                "visit_policy": ((17, 0.75), (23, 0.25)),
                "root_value": -0.125,
                "visits": visits,
            },
        )

    monkeypatch.setattr(rust_bridge, "model1_batched_mcts", fake_model1_batched_mcts)
    monkeypatch.setattr(
        engine,
        "to_python_state",
        lambda _state: (_ for _ in ()).throw(AssertionError("dense-cnn MCTS must pass live states directly")),
    )

    state = object()
    inference = FakeInference()
    result = mcts_module.run_mcts(
        state,
        inference,
        visits=3,
        c_puct=2.0,
        temperature=0.0,
        seed=None,
    )

    assert result == mcts_module.SearchResult(
        action_id=17,
        visit_policy=((17, 0.75), (23, 0.25)),
        root_value=-0.125,
        visits=3,
    )
    assert calls == [
        {
            "states": [state],
            "visits": 3,
            "c_puct": 2.0,
            "temperature": 0.0,
            "seed": 0,
            "evaluator": inference.evaluate_model1_payload,
            "virtual_batch_size": 3,
            "progressive_widening_initial_actions": 8,
            "progressive_widening_child_initial_actions": 4,
            "progressive_widening_candidate_actions": 128,
            "progressive_widening_growth_interval": 256.0,
            "progressive_widening_growth_base": 1.3,
            "root_dirichlet_alpha": None,
            "root_exploration_fraction": None,
            "evaluation_cache": None,
            "active_root_limit": 1024,
        }
    ]


def test_dense_cnn_mcts_clamps_explicit_virtual_batch_to_safe_leaf_budget() -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")

    assert mcts_module.DEFAULT_EVAL_CHUNK_STATES == 4096
    assert mcts_module._resolve_virtual_batch_size(
        root_count=1024,
        visits=128,
        virtual_batch_size=4,
    ) == 4
    assert mcts_module._resolve_virtual_batch_size(
        root_count=512,
        visits=128,
        virtual_batch_size=32,
    ) == 8
    assert mcts_module._resolve_virtual_batch_size(
        root_count=128,
        visits=128,
        virtual_batch_size=32,
    ) == 32


def test_dense_cnn_mcts_chunks_active_roots_before_rust_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    calls: list[tuple[int, int]] = []

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_batched_mcts(states: Sequence[object], **kwargs: object) -> tuple[Mapping[str, Any], ...]:
        calls.append((len(states), int(kwargs["seed"])))
        visits = int(kwargs["visits"])
        return tuple(
            {
                "action_id": index + 1,
                "visit_policy": ((index + 1, 1.0),),
                "root_value": 0.0,
                "visits": visits,
            }
            for index, _state in enumerate(states)
        )

    monkeypatch.setattr(rust_bridge, "model1_batched_mcts", fake_model1_batched_mcts)

    results = mcts_module.run_batched_mcts(
        [object() for _ in range(5)],
        FakeInference(),
        visits=7,
        seed=11,
        virtual_batch_size=4,
        active_root_limit=2,
    )

    assert [result.visits for result in results] == [7, 7, 7, 7, 7]
    assert calls == [(2, 11), (2, 13), (1, 15)]


def test_dense_cnn_mcts_defaults_to_production_active_root_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    calls: list[tuple[int, int | None]] = []

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_batched_mcts(states: Sequence[object], **kwargs: object) -> tuple[Mapping[str, Any], ...]:
        calls.append((len(states), kwargs.get("active_root_limit")))
        visits = int(kwargs["visits"])
        return tuple(
            {
                "action_id": 1,
                "visit_policy": ((1, 1.0),),
                "root_value": 0.0,
                "visits": visits,
            }
            for _state in states
        )

    monkeypatch.setattr(rust_bridge, "model1_batched_mcts", fake_model1_batched_mcts)

    results = mcts_module.run_batched_mcts(
        [object() for _ in range(1025)],
        FakeInference(),
        visits=1,
        seed=31,
        virtual_batch_size=1,
    )

    assert len(results) == 1025
    assert calls == [(1024, 1024), (1, 1024)]


def test_dense_cnn_payload_inference_respects_configured_max_batch_size() -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    inference_cls = _public_attr(dense_cnn, "DenseCNNInference")

    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            self.batch_sizes.append(batch)
            return {
                "policy": torch.zeros((batch, dense_cnn.BOARD_AREA), dtype=torch.float32, device=inputs.device),
                "value": torch.zeros((batch, dense_cnn.VALUE_BINS), dtype=torch.float32, device=inputs.device),
            }

    model = CountingModel()
    inference = inference_cls(model, device="cpu", amp=False, max_batch_size=2)
    rows = 5
    inputs = torch.zeros((rows, dense_cnn.INPUT_CHANNELS, dense_cnn.BOARD_SIZE, dense_cnn.BOARD_SIZE), dtype=torch.float32)
    payload = {
        "inputs": inputs.numpy().tobytes(),
        "shape": inputs.shape,
        "legal_flat_indices_bytes": b"",
        "legal_row_offsets": tuple(0 for _ in range(rows + 1)),
    }

    output = inference.evaluate_model1_payload(payload)

    assert set(output) == {"values_bytes", "priors_bytes"}
    assert model.batch_sizes == [2, 2, 1]


def test_dense_cnn_state_inference_respects_configured_max_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    inference_cls = _public_attr(dense_cnn, "DenseCNNInference")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")

    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            self.batch_sizes.append(batch)
            return {
                "policy": torch.zeros((batch, dense_cnn.BOARD_AREA), dtype=torch.float32, device=inputs.device),
                "value": torch.zeros((batch, dense_cnn.VALUE_BINS), dtype=torch.float32, device=inputs.device),
            }

    bridge_batches: list[int] = []

    def fake_model1_batch_inputs(states: Sequence[object]) -> Mapping[str, Any]:
        batch = len(states)
        bridge_batches.append(batch)
        inputs = torch.zeros(
            (batch, dense_cnn.INPUT_CHANNELS, dense_cnn.BOARD_SIZE, dense_cnn.BOARD_SIZE),
            dtype=torch.float32,
        )
        return {
            "inputs": inputs.numpy().tobytes(),
            "shape": inputs.shape,
            "legal_action_ids": tuple((0,) for _ in range(batch)),
            "legal_flat_indices": tuple((0,) for _ in range(batch)),
        }

    monkeypatch.setattr(rust_bridge, "model1_batch_inputs", fake_model1_batch_inputs)

    model = CountingModel()
    inference = inference_cls(model, device="cpu", amp=False, return_logits=False, max_batch_size=2)
    results = inference.infer_states([object() for _ in range(5)])

    assert len(results) == 5
    assert bridge_batches == [2, 2, 1]
    assert model.batch_sizes == [2, 2, 1]


def test_batch_inference_fast_path_runs_compact_samples_in_one_forward_pass() -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    inference_cls = _public_attr(dense_cnn, "DenseCNNInference")

    class CountingModel(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.batch_sizes: list[int] = []

        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            self.batch_sizes.append(int(inputs.shape[0]))
            return self.inner(inputs)

    model = CountingModel(_small_model())
    inference = inference_cls(model, device="cpu", amp=False)
    infer_many = _public_attr(inference, "infer_batch", "infer_samples")

    results = tuple(infer_many([_compact_sample(0), _compact_sample(1)]))

    assert len(results) == 2
    assert model.batch_sizes == [2], "batch inference must run the model once for the whole input batch"
    for result in results:
        policy_logits = _field(result, "policy_logits", "policy")
        value_logits = _field(result, "value_logits", "value")
        assert tuple(policy_logits.shape) == (dense_cnn.BOARD_AREA,)
        assert tuple(value_logits.shape) == (dense_cnn.VALUE_BINS,)
