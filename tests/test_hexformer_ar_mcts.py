from __future__ import annotations

import importlib.util
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MCTS_PATH = ROOT / "packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/mcts.py"
INFERENCE_PATH = ROOT / "packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/inference.py"
MISSING = object()
COORD_OFFSET = 1 << 15


def _pack_coord_id(coord: object) -> int:
    return ((int(coord.q) + COORD_OFFSET) << 16) | (int(coord.r) + COORD_OFFSET)


def _install_package_shell(monkeypatch: pytest.MonkeyPatch, *, rust: object = MISSING) -> None:
    hexo_models = types.ModuleType("hexo_models")
    if rust is not MISSING:
        hexo_models._rust = rust
    hexformer_pkg = types.ModuleType("hexo_models.hexformer_ar")
    hexformer_pkg.__path__ = []
    inference = types.ModuleType("hexo_models.hexformer_ar.inference")
    inference.HexformerInference = type("HexformerInference", (), {})
    input_mod = types.ModuleType("hexo_models.hexformer_ar.input")
    input_mod._config_mapping = lambda config: {"config": config}

    monkeypatch.setitem(sys.modules, "hexo_models", hexo_models)
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar", hexformer_pkg)
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.inference", inference)
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.input", input_mod)


def _load_mcts(monkeypatch: pytest.MonkeyPatch, *, rust: object = MISSING) -> types.ModuleType:
    _install_package_shell(monkeypatch, rust=rust)
    spec = importlib.util.spec_from_file_location("hexo_models.hexformer_ar.mcts", MCTS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.mcts", module)
    spec.loader.exec_module(module)
    return module


def _state(*coords: tuple[int, int]) -> SimpleNamespace:
    return SimpleNamespace(coords=coords)


def test_hexformer_ar_mcts_requires_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    mcts = _load_mcts(monkeypatch)

    with pytest.raises(RuntimeError, match="hexformer_ar Rust MCTS accelerator is unavailable"):
        mcts.run_batched_mcts((_state(),), SimpleNamespace(), visits=1)


def test_hexformer_ar_mcts_delegates_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    action_id = _pack_coord_id(SimpleNamespace(q=0, r=0))
    calls: list[object] = []

    class FakeInference:
        config = SimpleNamespace(architecture="arch", candidates="candidates")

        def evaluate_mcts_payload(self, payload: object) -> dict[str, object]:
            calls.append(payload)
            return {}

    class FakeHexformerRust:
        def hexformer_ar_batched_mcts(
            self,
            root_states: object,
            visits: int,
            c_puct: float,
            temperature: float,
            seed: int,
            evaluator: object,
            architecture: object,
            candidates: object,
            virtual_batch_size: int,
        ) -> tuple[dict[str, object], ...]:
            assert root_states == (root_state,)
            assert visits == 7
            assert c_puct == 2.0
            assert temperature == 0.25
            assert seed == 11
            assert architecture == {"config": "arch"}
            assert candidates == {"config": "candidates"}
            assert virtual_batch_size == 3
            evaluator({"sparse_payloads": ("payload",)})
            return (
                {
                    "action_id": action_id,
                    "visit_policy": ((action_id, 1.0),),
                    "root_value": 0.5,
                    "visits": 7,
                },
            )

    rust = SimpleNamespace(hexformer_ar=FakeHexformerRust())
    mcts = _load_mcts(monkeypatch, rust=rust)
    root_state = _state()

    result = mcts.run_batched_mcts(
        (root_state,),
        FakeInference(),
        visits=7,
        c_puct=2.0,
        temperature=0.25,
        seed=11,
        virtual_batch_size=3,
    )

    assert calls == [{"sparse_payloads": ("payload",)}]
    assert result[0].action_id == action_id
    assert result[0].visit_policy == {action_id: 1.0}
    assert result[0].root_value == 0.5
    assert result[0].visits == 7


def test_hexformer_ar_mcts_root_candidate_error_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHexformerRust:
        def hexformer_ar_batched_mcts(self, *_args: object) -> tuple[dict[str, object], ...]:
            raise ValueError("Hexformer MCTS root has no legal candidate actions")

    rust = SimpleNamespace(hexformer_ar=FakeHexformerRust())
    mcts = _load_mcts(monkeypatch, rust=rust)

    with pytest.raises(RuntimeError, match="Hexformer MCTS root has no legal candidate actions"):
        inference = SimpleNamespace(
            config=SimpleNamespace(architecture=object(), candidates=object()),
            evaluate_mcts_payload=lambda _payload: {},
        )
        mcts.run_mcts(_state(), inference, visits=1)


def _load_inference(monkeypatch: pytest.MonkeyPatch, action_id: int) -> types.ModuleType:
    hexo_models = types.ModuleType("hexo_models")
    hexformer_pkg = types.ModuleType("hexo_models.hexformer_ar")
    hexformer_pkg.__path__ = []
    config = types.ModuleType("hexo_models.hexformer_ar.config")
    config.HexformerConfig = type("HexformerConfig", (), {})

    input_mod = types.ModuleType("hexo_models.hexformer_ar.input")
    input_mod.SparseDecisionInput = type("SparseDecisionInput", (), {})
    input_mod.build_sparse_input = lambda _state, **_kwargs: pytest.fail("evaluate_mcts_payload rebuilt Python states")
    input_mod.sparse_input_from_payload = lambda payload: payload
    input_mod.collate_sparse_inputs = lambda _samples: {}
    input_mod.validate_candidate_frontier = lambda *_args, **_kwargs: None

    losses = types.ModuleType("hexo_models.hexformer_ar.losses")
    losses.wdl_value_from_logits = lambda logits: logits

    for name, module in {
        "hexo_models": hexo_models,
        "hexo_models.hexformer_ar": hexformer_pkg,
        "hexo_models.hexformer_ar.config": config,
        "hexo_models.hexformer_ar.input": input_mod,
        "hexo_models.hexformer_ar.losses": losses,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("hexo_models.hexformer_ar.inference", INFERENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hexo_models.hexformer_ar.inference", module)
    spec.loader.exec_module(module)
    return module


def test_hexformer_ar_mcts_payload_callback_returns_candidate_priors(monkeypatch: pytest.MonkeyPatch) -> None:
    action_id = _pack_coord_id(SimpleNamespace(q=0, r=0))
    inference_module = _load_inference(monkeypatch, action_id)

    def infer_sparse(sparse_inputs: object) -> tuple[object, ...]:
        return tuple(
            inference_module.HexformerInferenceResult(
                legal_action_ids=tuple(sample.candidate_action_ids),
                legal_priors={int(action_id): 1.0 for action_id in sample.candidate_action_ids},
                value=0.25,
                wdl=(0.25, 0.5, 0.25),
                distance=1.0,
            )
            for sample in sparse_inputs
        )

    inference = SimpleNamespace(
        config=SimpleNamespace(architecture=object(), candidates=object()),
        infer_sparse=infer_sparse,
    )

    payload = inference_module.HexformerInference.evaluate_mcts_payload(
        inference,
        {"sparse_payloads": (SimpleNamespace(candidate_action_ids=(action_id,)),)},
    )

    assert payload["candidate_action_ids"] == ((action_id,),)
    assert struct.unpack("f", payload["values_bytes"]) == pytest.approx((0.25,))
    assert struct.unpack("f", payload["priors_bytes"]) == pytest.approx((1.0,))
