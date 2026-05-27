from __future__ import annotations

import importlib
import inspect
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
    dense_cnn = _dense_cnn()
    calibrate_performance = _public_attr(
        dense_cnn,
        "calibrate_performance",
        "calibrate_model1_performance",
        "calibrate_dense_cnn_performance",
        "calibrate_dense_cnn",
    )

    result = _call_calibration_api(
        calibrate_performance,
        model=_small_model(),
        config=_small_config(),
    )

    _assert_batch_calibration_payload(
        result,
        inference_candidates=(1, 2),
        training_candidates=(1, 2),
    )


def test_training_pipeline_writes_calibration_json_artifact_when_enabled(tmp_path: Path) -> None:
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


def test_benchmark_report_targets_128_pps_and_does_not_fabricate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_cnn = _dense_cnn()
    build_benchmark_report = getattr(dense_cnn, "build_benchmark_report", None)
    if build_benchmark_report is None:
        build_benchmark_report = getattr(dense_cnn, "benchmark_report", None)
    if build_benchmark_report is None:
        build_benchmark_report = getattr(dense_cnn, "report_performance_benchmark", None)

    if build_benchmark_report is not None:
        report = build_benchmark_report(
            config=_small_config(),
            measurements={
                "selfplay_positions_per_second": 127.0,
                "inference_positions_per_second": 1_000_000.0,
                "training_samples_per_second": 1_000_000.0,
            },
        )
    else:
        calibrate_performance = _public_attr(
            dense_cnn,
            "calibrate_performance",
            "calibrate_model1_performance",
            "calibrate_dense_cnn_performance",
            "calibrate_dense_cnn",
        )
        performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
        monkeypatch.setattr(
            performance_module,
            "_benchmark_inference",
            lambda **_kwargs: [
                {
                    "status": "completed",
                    "batch_size": 2,
                    "positions_per_second": 127.0,
                }
            ],
        )
        monkeypatch.setattr(
            performance_module,
            "_benchmark_training",
            lambda **_kwargs: [
                {
                    "status": "completed",
                    "batch_size": 2,
                    "positions_per_second": 1_000_000.0,
                }
            ],
        )
        report = _call_calibration_api(
            calibrate_performance,
            model=_small_model(),
            config=_small_config(),
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
    calibrate_performance = _public_attr(
        dense_cnn,
        "calibrate_performance",
        "calibrate_model1_performance",
        "calibrate_dense_cnn_performance",
        "calibrate_dense_cnn",
    )
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

    report = calibrate_performance(model=_small_model(), config=config)

    assert report["target_mcts_simulations_per_position"] == 128
    assert report["measured_selfplay_positions_per_second"] == pytest.approx(10_000.0)
    assert report["all_searches_exact"] is False
    assert report["meets_target"] is False


def test_selfplay_benchmark_counts_every_root_as_its_own_128_sim_search(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_cnn = _dense_cnn()
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    engine = importlib.import_module("hexo_engine")

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


def test_selfplay_benchmark_reports_actual_searches_when_batch_overshoots_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    engine = importlib.import_module("hexo_engine")

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

    assert calls == [4]
    assert result["searched_positions"] == 4
    assert result["recorded_positions"] == 4
    assert result["mcts_simulations"] == 4 * 128
    assert result["all_searches_exact"] is True


def test_rust_model1_batch_input_encoder_matches_python_sample_encoder() -> None:
    torch = _torch()
    engine = importlib.import_module("hexo_engine")
    engine_types = importlib.import_module("hexo_engine.types")
    samples_module = importlib.import_module("hexo_models.dense_cnn.samples")
    dense_cnn = _dense_cnn()

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

    payload = engine.model1_batch_inputs(states)
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
