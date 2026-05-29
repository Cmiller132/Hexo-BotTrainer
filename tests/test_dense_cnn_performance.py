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


def _full_prior_payload(payload: Mapping[str, Any], *, value: float = 0.0) -> Mapping[str, object]:
    # The single evaluator mode: Rust sends the exact legal crop flats via
    # legal_row_offsets and expects one positional uniform prior per legal flat.
    rows = int(payload["shape"][0])
    offsets = tuple(int(item) for item in payload["legal_row_offsets"])
    prior_count = offsets[-1]
    return {
        "values_bytes": struct.pack(f"{rows}f", *([float(value)] * rows)),
        "priors_bytes": struct.pack(f"{prior_count}f", *([1.0] * prior_count)) if prior_count else b"",
    }


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


def _run_raw_session_mcts(
    rust_bridge: Any,
    states: Sequence[object],
    *,
    evaluator: object,
    session: object | None = None,
    game_keys: Sequence[int] | None = None,
    visits: int = 1,
    c_puct: float = 1.5,
    temperature: float = 0.0,
    seed: int = 0,
    virtual_batch_size: int | None = 1,
    **kwargs: object,
) -> tuple[Mapping[str, Any], ...]:
    native_session = session or rust_bridge.model1_new_mcts_session(max_states=100)
    keys = tuple(range(len(states))) if game_keys is None else tuple(int(key) for key in game_keys)
    return rust_bridge.model1_mcts_session_search(
        native_session,
        keys,
        tuple(states),
        visits=int(visits),
        c_puct=float(c_puct),
        temperature=float(temperature),
        seed=int(seed),
        evaluator=evaluator,
        virtual_batch_size=virtual_batch_size,
        **kwargs,
    )


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
                "search_visits": 1,
                "max_actions": 2,
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
                "train_samples_per_epoch = 2",
                "max_train_bucket_per_new_data = 8.0",
                "max_train_bucket_size = 16",
                "",
                "[model.config.samples]",
                "shuffle_min_rows = 1",
                "shuffle_keep_target_rows = 16",
                "approx_rows_per_out_file = 8",
                "",
                "[model.config.selfplay]",
                "search_visits = 1",
                "max_actions = 2",
                "temperature = 1.0",
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


def test_selfplay_benchmark_counts_every_root_as_its_own_128_sim_search(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_cnn = _dense_cnn()
    performance_module = importlib.import_module("hexo_models.dense_cnn.performance")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    engine = _engine_with_rust()

    calls: list[int] = []

    class FakeMctsSession:
        def run(
            self,
            _game_keys: Sequence[int],
            root_states: Sequence[object],
            inference: object,
            **kwargs: object,
        ) -> list[Any]:
            _ = inference
            visits = int(kwargs["visits"])
            calls.append(len(root_states))
            return [
                mcts_module.SearchResult(
                    action_id=int(engine.legal_action_ids(state)[0]),
                    visit_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                    root_value=0.0,
                    visits=visits,
                    root_prior_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                )
                for state in root_states
            ]

    monkeypatch.setattr(mcts_module, "new_mcts_session", lambda **_kwargs: FakeMctsSession())

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
    engine = _engine_with_rust()

    calls: list[int] = []

    class FakeMctsSession:
        def run(
            self,
            _game_keys: Sequence[int],
            root_states: Sequence[object],
            inference: object,
            **kwargs: object,
        ) -> list[Any]:
            _ = inference
            visits = int(kwargs["visits"])
            calls.append(len(root_states))
            return [
                mcts_module.SearchResult(
                    action_id=int(engine.legal_action_ids(state)[0]),
                    visit_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                    root_value=0.0,
                    visits=visits,
                    root_prior_policy={int(engine.legal_action_ids(state)[0]): 1.0},
                )
                for state in root_states
            ]

    monkeypatch.setattr(mcts_module, "new_mcts_session", lambda **_kwargs: FakeMctsSession())

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
        legal_ids = tuple(int(item) for item in payload["legal_action_ids"][index])
        if not legal_ids:
            continue
        sample = samples_module.sample_from_state(
            encoded_state,
            game_id=f"encoder-parity-{index}",
            turn_index=index,
            root_prior_policy={legal_ids[0]: 1.0},
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
    assert capabilities["model1_mcts_policy_nucleus_widening"] is True
    assert "model1_mcts_all_legal_candidates" not in capabilities
    assert capabilities["model1_mcts_tree_reuse_session"] is True
    assert capabilities["model1_mcts_session_search"] is True
    assert capabilities["model1_mcts_tree_reuse_reference"] == "KataGo_Search_makeMove_promote_child"
    assert capabilities["model1_mcts_root_dirichlet_noise"] is True
    assert capabilities["model1_mcts_root_policy_temperature"] is True
    assert capabilities["model1_mcts_first_play_urgency"] is True
    assert capabilities["model1_mcts_virtual_loss"] is True
    assert capabilities["model1_sample_from_state"] is True
    assert "model1_mcts_progressive_widening" not in capabilities
    assert "model1_mcts_hidden_prior_mass" not in capabilities
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

    first = _run_raw_session_mcts(
        rust_bridge,
        [state],
        visits=3,
        c_puct=1.5,
        temperature=0.0,
        seed=42,
        evaluator=evaluator,
        virtual_batch_size=2,
    )[0]
    second = _run_raw_session_mcts(
        rust_bridge,
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
            return _full_prior_payload(payload)

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
    )[0]

    assert first.visits == 4
    assert second.visits == 4
    assert len(session) == 1
    assert sum(weight for _action_id, weight in first.visit_policy) == pytest.approx(1.0)
    assert sum(weight for _action_id, weight in second.visit_policy) == pytest.approx(1.0)
    assert int(second.diagnostics["batch"]["tree"]["completed_visits"]) > second.visits


def test_dense_cnn_mcts_session_exports_full_root_prior_across_tree_reuse() -> None:
    # Regression guard for the shared-prior (Arc + cursor) refactor: the exported
    # root_prior_policy must remain the FULL in-crop legal distribution, both for a
    # freshly searched root and for a promoted (reused) interior root. Truncating the
    # shared prior list would silently shrink this distribution and corrupt the
    # policy-surprise weighting and the rootPolicy auxiliary target.
    engine = importlib.import_module("hexo_engine")
    engine_types = importlib.import_module("hexo_engine.types")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    class FakeInference:
        def evaluate_model1_payload(self, payload: Mapping[str, Any]) -> Mapping[str, object]:
            return _full_prior_payload(payload)

    def in_crop_legal_count(state: object) -> int:
        return len(rust_bridge.model1_batch_inputs([state])["legal_action_ids"][0])

    state = engine.new_game()
    inference = FakeInference()
    session = mcts_module.new_mcts_session(max_states=100)

    expected_first = in_crop_legal_count(state)
    first = session.run(
        [7], [state], inference, visits=8, temperature=0.0, seed=1, virtual_batch_size=1
    )[0]
    assert len(first.root_prior_policy) == expected_first

    engine.apply_action(
        state,
        engine.PlacementAction(engine_types.unpack_coord_id(first.action_id)),
    )

    expected_second = in_crop_legal_count(state)
    second = session.run(
        [7], [state], inference, visits=8, temperature=0.0, seed=2, virtual_batch_size=1
    )[0]
    # The second move reuses the tree (the root is a promoted interior node), so this
    # exercises the Shared->Owned export path, not a freshly evaluated owned root.
    assert int(second.diagnostics["batch"]["tree"]["completed_visits"]) > second.visits
    assert len(second.root_prior_policy) == expected_second


def test_dense_cnn_rust_mcts_rejects_short_compact_evaluator_payloads() -> None:
    engine = importlib.import_module("hexo_engine")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    _skip_without_direct_state_rust(rust_bridge)

    def evaluator(_payload: Mapping[str, Any]) -> Mapping[str, object]:
        return {
            "values_bytes": b"",
            "priors_bytes": b"",
        }

    with pytest.raises(ValueError, match="values_bytes"):
        _run_raw_session_mcts(
            rust_bridge,
            [engine.new_game()],
            visits=1,
            c_puct=1.5,
            temperature=0.0,
            seed=29,
            evaluator=evaluator,
            virtual_batch_size=1,
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

    results = _run_raw_session_mcts(
        rust_bridge,
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
        _run_raw_session_mcts(
            rust_bridge,
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

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", unavailable)

    with pytest.raises(RuntimeError, match="Rust accelerator is unavailable"):
        mcts_module.new_mcts_session(max_states=100)


def test_dense_cnn_mcts_python_boundary_delegates_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = importlib.import_module("hexo_engine")
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    calls: list[Mapping[str, Any]] = []

    class FakeInference:
        evaluate_model1_payload = object()

    native_session = object()

    def fake_model1_mcts_session_search(
        session: object,
        game_keys: Sequence[int],
        states: Sequence[object],
        *,
        visits: int,
        c_puct: float,
        temperature: float,
        seed: int,
        evaluator: object,
        virtual_batch_size: int | None,
        active_root_limit: int | None,
        root_dirichlet_total_alpha: float | None,
        root_dirichlet_noise_fraction: float | None,
        root_policy_temperature: float | None,
        fpu_reduction: float | None,
        virtual_loss: float | None,
        widening_policy_mass: float | None,
        widening_max_children: int | None,
        widening_min_children: int | None,
    ) -> tuple[Mapping[str, Any], ...]:
        calls.append(
            {
                "session": session,
                "game_keys": tuple(game_keys),
                "states": states,
                "visits": visits,
                "c_puct": c_puct,
                "temperature": temperature,
                "seed": seed,
                "evaluator": evaluator,
                "virtual_batch_size": virtual_batch_size,
                "active_root_limit": active_root_limit,
                "root_dirichlet_total_alpha": root_dirichlet_total_alpha,
                "root_dirichlet_noise_fraction": root_dirichlet_noise_fraction,
                "root_policy_temperature": root_policy_temperature,
                "fpu_reduction": fpu_reduction,
                "virtual_loss": virtual_loss,
                "widening_policy_mass": widening_policy_mass,
                "widening_max_children": widening_max_children,
                "widening_min_children": widening_min_children,
            }
        )
        return (
            {
                "action_id": 17,
                "visit_policy_action_ids_bytes": struct.pack("2I", 17, 23),
                "visit_policy_weights_bytes": struct.pack("2f", 0.75, 0.25),
                "visit_policy_count": 2,
                "root_prior_policy_action_ids_bytes": struct.pack("2I", 17, 23),
                "root_prior_policy_weights_bytes": struct.pack("2f", 0.60, 0.40),
                "root_prior_policy_count": 2,
                "root_value": -0.125,
                "visits": visits,
            },
        )

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", lambda **_kwargs: native_session)
    monkeypatch.setattr(rust_bridge, "model1_mcts_session_search", fake_model1_mcts_session_search)
    monkeypatch.setattr(
        engine,
        "to_python_state",
        lambda _state: (_ for _ in ()).throw(AssertionError("dense-cnn MCTS must pass live states directly")),
    )

    state = object()
    inference = FakeInference()
    session = mcts_module.new_mcts_session(max_states=100)
    result = session.run(
        [0],
        [state],
        inference,
        visits=3,
        c_puct=2.0,
        temperature=0.0,
        seed=None,
        virtual_batch_size=3,
    )[0]

    assert result.action_id == 17
    assert [action for action, _weight in result.visit_policy] == [17, 23]
    assert [weight for _action, weight in result.visit_policy] == pytest.approx([0.75, 0.25])
    assert [action for action, _weight in result.root_prior_policy] == [17, 23]
    assert [weight for _action, weight in result.root_prior_policy] == pytest.approx([0.60, 0.40])
    assert result.root_value == pytest.approx(-0.125)
    assert result.visits == 3
    assert not hasattr(result, "policy_surprise")
    assert not hasattr(result, "frequency_weight")
    assert calls == [
        {
            "session": native_session,
            "game_keys": (0,),
            "states": [state],
            "visits": 3,
            "c_puct": 2.0,
            "temperature": 0.0,
            "seed": 0,
            "evaluator": inference.evaluate_model1_payload,
            "virtual_batch_size": 3,
            "active_root_limit": None,
            "root_dirichlet_total_alpha": None,
            "root_dirichlet_noise_fraction": None,
            "root_policy_temperature": None,
            "fpu_reduction": None,
            "virtual_loss": None,
            "widening_policy_mass": None,
            "widening_max_children": None,
            "widening_min_children": None,
        }
    ]


def test_dense_cnn_mcts_rejects_invalid_virtual_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_mcts_session_search(*_args: object, **_kwargs: object) -> tuple[Mapping[str, Any], ...]:
        raise ValueError("virtual_batch_size must be > 0")

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", lambda **_kwargs: object())
    monkeypatch.setattr(rust_bridge, "model1_mcts_session_search", fake_model1_mcts_session_search)
    session = mcts_module.new_mcts_session(max_states=100)

    with pytest.raises(ValueError, match="virtual_batch_size"):
        session.run([0], [object()], FakeInference(), visits=1, virtual_batch_size=0)


def test_dense_cnn_mcts_delegates_active_root_limit_to_rust_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_mcts_session_search(
        _session: object,
        _game_keys: Sequence[int],
        states: Sequence[object],
        **kwargs: object,
    ) -> tuple[Mapping[str, Any], ...]:
        assert len(states) == 3
        assert kwargs["active_root_limit"] == 2
        raise ValueError("active_root_limit")

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", lambda **_kwargs: object())
    monkeypatch.setattr(rust_bridge, "model1_mcts_session_search", fake_model1_mcts_session_search)
    session = mcts_module.new_mcts_session(max_states=100)

    with pytest.raises(ValueError, match="active_root_limit"):
        session.run([0, 1, 2], [object(), object(), object()], FakeInference(), visits=7, active_root_limit=2)


def test_dense_cnn_mcts_requires_root_prior_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_mcts_session_search(
        _session: object,
        _game_keys: Sequence[int],
        states: Sequence[object],
        **kwargs: object,
    ) -> tuple[Mapping[str, Any], ...]:
        visits = int(kwargs["visits"])
        return tuple(
            {
                "action_id": 1,
                "visit_policy_action_ids_bytes": struct.pack("I", 1),
                "visit_policy_weights_bytes": struct.pack("f", 1.0),
                "visit_policy_count": 1,
                "root_value": 0.0,
                "visits": visits,
            }
            for _state in states
        )

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", lambda **_kwargs: object())
    monkeypatch.setattr(rust_bridge, "model1_mcts_session_search", fake_model1_mcts_session_search)

    session = mcts_module.new_mcts_session(max_states=100)
    with pytest.raises(ValueError, match="root_prior_policy"):
        session.run([0], [object()], FakeInference(), visits=1)


def test_dense_cnn_mcts_leaves_default_active_root_limit_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_module = importlib.import_module("hexo_models.dense_cnn.mcts")
    rust_bridge = importlib.import_module("hexo_models.dense_cnn.rust_bridge")
    calls: list[tuple[int, int | None]] = []

    class FakeInference:
        evaluate_model1_payload = object()

    def fake_model1_mcts_session_search(
        _session: object,
        _game_keys: Sequence[int],
        states: Sequence[object],
        **kwargs: object,
    ) -> tuple[Mapping[str, Any], ...]:
        calls.append((len(states), kwargs.get("active_root_limit")))
        visits = int(kwargs["visits"])
        return tuple(
            {
                "action_id": 1,
                "visit_policy_action_ids_bytes": struct.pack("I", 1),
                "visit_policy_weights_bytes": struct.pack("f", 1.0),
                "visit_policy_count": 1,
                "root_prior_policy_action_ids_bytes": struct.pack("I", 1),
                "root_prior_policy_weights_bytes": struct.pack("f", 1.0),
                "root_prior_policy_count": 1,
                "root_value": 0.0,
                "visits": visits,
            }
            for _state in states
        )

    monkeypatch.setattr(rust_bridge, "model1_new_mcts_session", lambda **_kwargs: object())
    monkeypatch.setattr(rust_bridge, "model1_mcts_session_search", fake_model1_mcts_session_search)

    session = mcts_module.new_mcts_session(max_states=100)
    results = session.run([0], [object()], FakeInference(), visits=1, seed=31, virtual_batch_size=1)

    assert len(results) == 1
    assert calls == [(1, None)]


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


def test_dense_cnn_payload_inference_rejects_legacy_legal_index_payload() -> None:
    torch = _torch()
    dense_cnn = _dense_cnn()
    inference_cls = _public_attr(dense_cnn, "DenseCNNInference")

    class Model(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
            batch = int(inputs.shape[0])
            return {
                "policy": torch.zeros((batch, dense_cnn.BOARD_AREA), dtype=torch.float32, device=inputs.device),
                "value": torch.zeros((batch, dense_cnn.VALUE_BINS), dtype=torch.float32, device=inputs.device),
            }

    inference = inference_cls(Model(), device="cpu", amp=False)
    inputs = torch.zeros((1, dense_cnn.INPUT_CHANNELS, dense_cnn.BOARD_SIZE, dense_cnn.BOARD_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="legal_flat_indices_bytes"):
        inference.evaluate_model1_payload(
            {
                "inputs": inputs.numpy().tobytes(),
                "shape": inputs.shape,
                "legal_flat_indices": ((0,),),
            }
        )


def test_dense_cnn_inference_exposes_only_production_state_and_payload_paths() -> None:
    dense_cnn = _dense_cnn()
    inference_cls = _public_attr(dense_cnn, "DenseCNNInference")

    inference = inference_cls(_small_model(), device="cpu", amp=False)

    assert hasattr(inference, "infer_state")
    assert hasattr(inference, "infer_states")
    assert hasattr(inference, "infer_inputs")
    assert hasattr(inference, "evaluate_model1_payload")
    assert not hasattr(inference, "infer_batch")
    assert not hasattr(inference, "infer_samples")
