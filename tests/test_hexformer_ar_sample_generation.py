from __future__ import annotations

import sys
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEXFORMER_PACKAGE = ROOT / "packages" / "hexo_models" / "hexformer_ar" / "python" / "hexo_models" / "hexformer_ar"


def test_build_sparse_input_consumes_rust_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    input_module = _load_hexformer_module(monkeypatch, "input")
    config = importlib.import_module("hexo_models.hexformer_ar.config")
    HexformerArchitectureConfig = config.HexformerArchitectureConfig
    HexformerCandidateConfig = config.HexformerCandidateConfig

    arch = HexformerArchitectureConfig(local_crop_size=3, max_candidates=2, max_stones=1, max_windows=1)
    candidate_cfg = HexformerCandidateConfig(max_candidates=2)
    fake = _FakeRust(_payload(arch, candidate_ids=(7, 8)))
    monkeypatch.setattr(input_module, "_MODELS_RUST", SimpleNamespace(hexformer_ar=fake))
    monkeypatch.setattr(input_module, "_RUST_IMPORT_ERROR", None)
    state = object()

    sparse = input_module.build_sparse_input(
        state,
        architecture=arch,
        candidates=candidate_cfg,
        policy={7: 3.0},
        value=1.0,
        metadata={"source": "unit"},
    )

    assert sparse.candidate_action_ids == (7, 8)
    assert sparse.policy_target is not None
    assert fake.calls[0]["state"] is state
    assert fake.calls[0]["policy"] == ((7, 3.0),)
    assert fake.calls[0]["metadata"] == {"source": "unit"}


def test_selfplay_finalization_uses_rust_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    selfplay = _load_selfplay_module(monkeypatch)
    config = importlib.import_module("hexo_models.hexformer_ar.config")
    HexformerArchitectureConfig = config.HexformerArchitectureConfig
    HexformerCandidateConfig = config.HexformerCandidateConfig

    arch = HexformerArchitectureConfig(local_crop_size=3, max_candidates=2, max_stones=1, max_windows=1)
    captured: dict[str, object] = {}

    def fake_build_selfplay_sample_payloads(**kwargs: object) -> tuple[dict[str, object], ...]:
        captured.update(kwargs)
        return (
            {
                "game_id": kwargs["game_id"],
                "turn_index": 0,
                "input_payload": _payload(arch, candidate_ids=(7, 8)),
                "metadata": {"selected_action_id": 7, "model_family": "hexformer_ar"},
            },
        )

    monkeypatch.setattr(selfplay, "build_selfplay_sample_payloads", fake_build_selfplay_sample_payloads)
    pending = [
        selfplay.PendingDecision(
            state=object(),
            player="player0",
            turn_index=0,
            search=selfplay.SearchResult(
                action_id=7,
                visit_policy={7: 0.75, 8: 0.25},
                root_value=0.5,
                visits=4,
            ),
        )
    ]

    samples = selfplay._finalize_pending(
        "game-a",
        pending,
        "player0",
        SimpleNamespace(architecture=arch, candidates=HexformerCandidateConfig(max_candidates=2)),
    )

    assert len(samples) == 1
    assert samples[0].input_payload["candidate_action_ids"] == [7, 8]
    assert captured["players"] == ("player0",)
    assert captured["selected_action_ids"] == (7,)


def test_hexformer_ar_rust_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    input_module = _load_hexformer_module(monkeypatch, "input")

    monkeypatch.setattr(input_module, "_MODELS_RUST", None)
    monkeypatch.setattr(input_module, "_RUST_IMPORT_ERROR", ImportError("missing test module"))

    with pytest.raises(RuntimeError, match="hexformer_ar Rust sample generator is unavailable"):
        input_module._hexformer_ar_rust()


class _FakeRust:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def sparse_input_payload(
        self,
        state: object,
        architecture: dict[str, object],
        candidates: dict[str, object],
        policy: tuple[tuple[int, float], ...],
        opp_policy: tuple[tuple[int, float], ...],
        value: float | None,
        distance: float | None,
        lookahead: tuple[tuple[int, float], ...],
        metadata: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(
            {
                "state": state,
                "architecture": architecture,
                "candidates": candidates,
                "policy": policy,
                "opp_policy": opp_policy,
                "value": value,
                "distance": distance,
                "lookahead": lookahead,
                "metadata": metadata,
            }
        )
        return self.payload


def _install_hexformer_package_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    root_pkg = ModuleType("hexo_models")
    root_pkg.__path__ = [str(ROOT / "packages" / "hexo_models" / "python" / "hexo_models")]
    sub_pkg = ModuleType("hexo_models.hexformer_ar")
    sub_pkg.__path__ = [str(HEXFORMER_PACKAGE)]
    monkeypatch.setitem(sys.modules, "hexo_models", root_pkg)
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar", sub_pkg)


def _load_hexformer_module(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    _install_hexformer_package_stub(monkeypatch)
    full_name = f"hexo_models.hexformer_ar.{name}"
    spec = importlib.util.spec_from_file_location(full_name, HEXFORMER_PACKAGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, full_name, module)
    spec.loader.exec_module(module)
    return module


def _load_selfplay_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _install_hexformer_package_stub(monkeypatch)

    fake_input = ModuleType("hexo_models.hexformer_ar.input")
    fake_input.build_selfplay_sample_payloads = lambda **_kwargs: ()
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.input", fake_input)

    fake_inference = ModuleType("hexo_models.hexformer_ar.inference")
    fake_inference.HexformerInference = object
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.inference", fake_inference)

    fake_mcts = ModuleType("hexo_models.hexformer_ar.mcts")

    @dataclass(frozen=True, slots=True)
    class SearchResult:
        action_id: int
        visit_policy: dict[int, float]
        root_value: float
        visits: int

    fake_mcts.SearchResult = SearchResult
    fake_mcts.run_mcts = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.mcts", fake_mcts)

    fake_samples = ModuleType("hexo_models.hexformer_ar.samples")

    @dataclass(frozen=True, slots=True)
    class HexformerSample:
        game_id: str
        turn_index: int
        input_payload: dict[str, object]
        metadata: dict[str, object]

    fake_samples.SAMPLE_NAMESPACE = "hexo_models.hexformer_ar"
    fake_samples.HexoformerSample = HexformerSample
    fake_samples.HexformerSample = HexformerSample
    fake_samples.training_record_from_sample = lambda sample: sample
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.samples", fake_samples)

    fake_engine = ModuleType("hexo_engine")
    fake_engine.engine_metadata = lambda: {}
    fake_engine.new_game = lambda **_kwargs: object()
    fake_engine.terminal = lambda _state: None
    fake_engine.clone_state = lambda state: state
    fake_engine.current_player = lambda _state: "player0"
    fake_engine.apply_action = lambda *_args, **_kwargs: None
    fake_engine.PlacementAction = lambda coord: coord
    monkeypatch.setitem(sys.modules, "hexo_engine", fake_engine)
    fake_engine_types = ModuleType("hexo_engine.types")
    fake_engine_types.unpack_coord_id = lambda action_id: action_id
    monkeypatch.setitem(sys.modules, "hexo_engine.types", fake_engine_types)

    fake_runner_records = ModuleType("hexo_runner.records")
    fake_runner_records.AbortRecord = object
    fake_runner_records.HexoRecordFile = object
    fake_runner_records.HexoRecordPlayer = object
    monkeypatch.setitem(sys.modules, "hexo_runner.records", fake_runner_records)
    monkeypatch.setitem(sys.modules, "hexo_runner", ModuleType("hexo_runner"))

    fake_utils_samples = ModuleType("hexo_utils.samples")
    fake_utils_samples.append_samples = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "hexo_utils.samples", fake_utils_samples)
    monkeypatch.setitem(sys.modules, "hexo_utils", ModuleType("hexo_utils"))

    return _load_hexformer_module(monkeypatch, "selfplay")


def _tensor(shape: tuple[int, ...], dtype: str = "float32", fill: float = 0.0) -> dict[str, object]:
    total = 1
    for dim in shape:
        total *= dim
    return {"shape": list(shape), "dtype": dtype, "data": [fill] * total}


def _payload(arch: object, *, candidate_ids: tuple[int, ...]) -> dict[str, object]:
    count = len(candidate_ids)
    local_shape = (1, arch.local_input_channels, arch.local_crop_size, arch.local_crop_size)
    return {
        "candidate_action_ids": list(candidate_ids),
        "candidate_features": _tensor((count, arch.candidate_feature_dim)),
        "candidate_coords": _tensor((count, 5)),
        "candidate_mask": _tensor((count,), "int8", 1),
        "stone_features": _tensor((0, arch.stone_feature_dim)),
        "stone_coords": _tensor((0, 5)),
        "stone_mask": _tensor((0,), "int8"),
        "window_features": _tensor((0, arch.window_feature_dim)),
        "window_coords": _tensor((0, 5)),
        "window_mask": _tensor((0,), "int8"),
        "local_input": _tensor(local_shape[1:]),
        "local_inputs": _tensor(local_shape),
        "local_window_coords": _tensor((1, 5)),
        "local_window_mask": _tensor((1,), "int8", 1),
        "rel_edge_index": _tensor((0, 2), "int64"),
        "rel_edge_features": _tensor((0, arch.rel_edge_feature_dim)),
        "rel_edge_mask": _tensor((0,), "int8"),
        "global_features": _tensor((arch.global_feature_dim,)),
        "policy_target": {"shape": [count], "dtype": "float32", "data": [1.0] + [0.0] * (count - 1)},
        "metadata": {"anchor": [0, 0], "candidate": {}, "tactical": {}},
    }
