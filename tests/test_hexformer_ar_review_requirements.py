from __future__ import annotations

import struct
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_utils", "hexo_train", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_coord_id_near_i16_boundary() -> None:
    from hexo_models.hexformer_ar.coordinates import Axial, pack_action_id, unpack_action_id

    for coord in (
        Axial(-(1 << 15), -(1 << 15)),
        Axial(-(1 << 15), (1 << 15) - 1),
        Axial((1 << 15) - 1, -(1 << 15)),
        Axial((1 << 15) - 1, (1 << 15) - 1),
    ):
        assert unpack_action_id(pack_action_id(coord)) == coord


@pytest.mark.parametrize("coord", [(1 << 15, 0), (0, 1 << 15), (-(1 << 15) - 1, 0), (0, -(1 << 15) - 1)])
def test_sparse_payload_does_not_silently_wrap_coord(coord: tuple[int, int]) -> None:
    from hexo_models.hexformer_ar.coordinates import Axial, pack_action_id

    with pytest.raises(ValueError, match="outside i16 range"):
        pack_action_id(Axial(*coord))


def test_hexformer_input_has_no_history_row_helpers() -> None:
    import hexo_models.hexformer_ar.input as input_module

    assert not hasattr(input_module, "history_row_from_state")
    assert not hasattr(input_module, "history_rows_from_states")


def test_sparse_input_payload_round_trip() -> None:
    from hexo_models.hexformer_ar.samples import sparse_input_from_payload, sparse_input_to_payload

    arch = _small_arch()
    sparse = _sparse_input(arch, candidate_count=3, metadata={"source": "round-trip"})
    payload = sparse_input_to_payload(sparse)
    restored = sparse_input_from_payload(payload)

    assert restored.candidate_action_ids == sparse.candidate_action_ids
    assert restored.metadata == sparse.metadata
    assert torch.equal(restored.candidate_features, sparse.candidate_features)
    assert torch.equal(restored.candidate_mask, sparse.candidate_mask)
    assert torch.equal(restored.local_inputs, sparse.local_inputs)
    assert torch.equal(restored.rel_edge_index, sparse.rel_edge_index)
    assert torch.equal(restored.policy_target, sparse.policy_target)
    assert torch.equal(restored.lookahead_targets[1], sparse.lookahead_targets[1])


def test_collate_sparse_inputs_shapes() -> None:
    from hexo_models.hexformer_ar import collate_sparse_inputs

    arch = _small_arch()
    first = _sparse_input(arch, candidate_count=2, stone_count=1, window_count=1, local_count=1, edge_count=1)
    second = _sparse_input(arch, candidate_count=4, stone_count=3, window_count=2, local_count=2, edge_count=3)

    batch = collate_sparse_inputs((first, second))

    assert batch["candidate_features"].shape == (2, 4, arch.candidate_feature_dim)
    assert batch["candidate_mask"].shape == (2, 4)
    assert batch["stone_features"].shape == (2, 3, arch.stone_feature_dim)
    assert batch["window_features"].shape == (2, 2, arch.window_feature_dim)
    assert batch["local_inputs"].shape == (2, 2, arch.local_input_channels, arch.local_crop_size, arch.local_crop_size)
    assert batch["rel_edge_index"].shape == (2, 3, 2)
    assert batch["policy_target"].shape == (2, 4)
    assert batch["lookahead_1_target"].shape == (2, 3)


def test_forward_tiny_batch() -> None:
    from hexo_models.hexformer_ar import HexformerAR, collate_sparse_inputs

    arch = _small_arch()
    model = HexformerAR(arch).eval()
    batch = collate_sparse_inputs(
        (
            _sparse_input(arch, candidate_count=2),
            _sparse_input(arch, candidate_count=3),
        )
    )

    with torch.no_grad():
        outputs = model(batch)

    assert outputs["policy_logits"].shape == (2, 3)
    assert outputs["opp_policy_logits"].shape == (2, 3)
    assert outputs["wdl_logits"].shape == (2, 3)
    assert outputs["distance"].shape == (2,)
    assert outputs["threat_logits"].shape == (2, 3, arch.threat_classes)


def test_policy_logits_masked() -> None:
    from hexo_models.hexformer_ar import HexformerAR, collate_sparse_inputs

    arch = _small_arch()
    sparse = replace(
        _sparse_input(arch, candidate_count=3),
        candidate_mask=torch.tensor([True, False, True], dtype=torch.bool),
    )
    model = HexformerAR(arch).eval()
    batch = collate_sparse_inputs((sparse,))

    with torch.no_grad():
        outputs = model(batch)

    floor = torch.finfo(outputs["policy_logits"].dtype).min
    assert outputs["policy_logits"][0, 1].item() == floor
    assert outputs["opp_policy_logits"][0, 1].item() == floor
    assert outputs["rz_logits"][0, 1].item() == floor
    assert outputs["policy_logits"][0, 0].item() > floor


def test_inference_maps_priors_to_action_ids() -> None:
    from hexo_models.hexformer_ar.inference import HexformerInference

    arch = _small_arch()
    config = _small_config(device="cpu", arch=arch)
    action_ids = _action_ids(3)
    sparse = replace(_sparse_input(arch, candidate_count=3), candidate_action_ids=action_ids)
    inference = HexformerInference(_FixedLogitModel(policy_logits=torch.tensor([[0.0, 2.0, -4.0]])), config=config)

    [result] = inference.infer_sparse((sparse,))

    assert tuple(result.legal_priors) == action_ids
    assert result.legal_priors[action_ids[1]] > result.legal_priors[action_ids[0]]
    assert result.legal_priors[action_ids[0]] > result.legal_priors[action_ids[2]]
    assert sum(result.legal_priors.values()) == pytest.approx(1.0)


def test_evaluate_mcts_payload_schema() -> None:
    from hexo_models.hexformer_ar.inference import HexformerInference, HexformerInferenceResult
    from hexo_models.hexformer_ar.samples import sparse_input_to_payload

    arch = _small_arch()
    first = _sparse_input(arch, candidate_count=2)
    second = _sparse_input(arch, candidate_count=3)

    def infer_sparse(sparse_inputs: object) -> tuple[HexformerInferenceResult, ...]:
        return tuple(
            HexformerInferenceResult(
                legal_action_ids=tuple(sample.candidate_action_ids),
                legal_priors={int(action_id): 1.0 / len(sample.candidate_action_ids) for action_id in sample.candidate_action_ids},
                value=0.25 + index,
                wdl=(0.2, 0.3, 0.5),
                distance=2.0,
            )
            for index, sample in enumerate(sparse_inputs)
        )

    inference = SimpleNamespace(infer_sparse=infer_sparse)
    payload = HexformerInference.evaluate_mcts_payload(
        inference,
        {"sparse_payloads": (sparse_input_to_payload(first), sparse_input_to_payload(second))},
    )

    assert set(payload) == {"values_bytes", "candidate_action_ids", "priors_bytes"}
    assert payload["candidate_action_ids"] == (first.candidate_action_ids, second.candidate_action_ids)
    assert struct.unpack("2f", payload["values_bytes"]) == pytest.approx((0.25, 1.25))
    assert len(struct.unpack("5f", payload["priors_bytes"])) == 5


def test_plugin_builds_model() -> None:
    from hexo_models.hexformer_ar.architecture import HexformerAR
    from hexo_models.hexformer_ar.plugin import get_plugin

    arch = _small_arch()
    model = get_plugin().build_model({}, _small_config_dict(arch))

    assert isinstance(model, HexformerAR)
    assert model.config.token_dim == arch.token_dim
    assert model.config.max_candidates == arch.max_candidates


def test_config_parse_matches_toml() -> None:
    from hexo_models.hexformer_ar.config import parse_hexformer_config

    raw = tomllib.loads((ROOT / "configs" / "hexformer_ar.toml").read_text(encoding="utf-8"))
    config_section = raw["model"]["config"]
    parsed = parse_hexformer_config(config_section)

    assert parsed.device == config_section["device"]
    assert parsed.architecture.token_dim == config_section["architecture"]["token_dim"]
    assert parsed.architecture.gps_layers == config_section["architecture"]["gps_layers"]
    assert parsed.architecture.max_rel_edges == config_section["architecture"]["max_rel_edges"]
    assert parsed.candidates.max_candidates == config_section["candidates"]["max_candidates"]
    assert parsed.selfplay.search_visits == config_section["selfplay"]["search_visits"]
    assert parsed.samples.train_sample_count == config_section["samples"]["train_sample_count"]


def test_hexformer_model_specific_logic_stays_outside_hexo_engine_static() -> None:
    engine_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in (ROOT / "packages" / "hexo_engine" / "rust" / "src", ROOT / "packages" / "hexo_engine" / "python" / "hexo_engine")
        for path in root.rglob("*")
        if path.suffix in {".rs", ".py"}
    ).lower()

    forbidden = (
        "hexformer",
        "hexformer_ar",
        "sparse_input_payload",
        "candidate_frontier",
        "build_tactical_summary",
        "policy_logits",
        "puct",
        "mcts",
    )
    assert not any(token in engine_text for token in forbidden)


def test_hexformer_uses_engine_state_clone_static() -> None:
    input_source = _read("packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/input.py")
    mcts_source = _read("packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/mcts.py")
    rust_mcts_source = _read("packages/hexo_models/hexformer_ar/rust/src/mcts.rs")
    rust_sample_source = _read("packages/hexo_models/hexformer_ar/rust/src/sample_gen.rs")

    forbidden = ("history_row", "history_rows", "states_from_history", "state_from_history", "to_python_state")
    assert not any(token in input_source for token in forbidden)
    assert not any(token in mcts_source for token in forbidden)
    assert not any(token in rust_mcts_source for token in forbidden)
    assert not any(token in rust_sample_source for token in forbidden)
    assert "clone_py_engine_state" in rust_sample_source
    assert "clone_py_engine_states" in rust_mcts_source
    assert "sparse_payloads" in _read("packages/hexo_models/hexformer_ar/rust/src/mcts_eval.rs")
    assert "state_source\", \"engine_state_clone" in _read("packages/hexo_models/hexformer_ar/rust/src/lib.rs")
    assert "u32_i16_pair" in _read("packages/hexo_models/hexformer_ar/rust/src/lib.rs")
    assert "bounded to i16 coordinate components" in _read("README.md")


def test_hexformer_frontier_and_tactical_logic_are_rust_owned_static() -> None:
    python_root = ROOT / "packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar"
    rust_sample = _read("packages/hexo_models/hexformer_ar/rust/src/sample_gen.rs")

    assert not (python_root / "candidates.py").exists()
    assert not (python_root / "windows.py").exists()
    assert "fn build_candidate_frontier" in rust_sample
    assert "fn build_tactical_summary" in rust_sample
    assert "fn build_local_windows" in rust_sample


def _small_arch():
    from hexo_models.hexformer_ar.config import HexformerArchitectureConfig

    return HexformerArchitectureConfig(
        local_channels=8,
        local_blocks=1,
        token_dim=16,
        gps_layers=1,
        attention_heads=4,
        dropout=0.0,
        local_crop_size=9,
        max_local_windows=2,
        max_candidates=4,
        max_stones=4,
        max_windows=4,
        max_rel_edges=8,
        lookahead_horizons=(1,),
    )


def _small_config(*, device: str, arch: object):
    from hexo_models.hexformer_ar.config import parse_hexformer_config

    return parse_hexformer_config({**_small_config_dict(arch), "device": device})


def _small_config_dict(arch: object) -> dict[str, object]:
    return {
        "architecture": {
            "local_channels": arch.local_channels,
            "local_blocks": arch.local_blocks,
            "token_dim": arch.token_dim,
            "gps_layers": arch.gps_layers,
            "attention_heads": arch.attention_heads,
            "dropout": arch.dropout,
            "local_crop_size": arch.local_crop_size,
            "max_local_windows": arch.max_local_windows,
            "max_candidates": arch.max_candidates,
            "max_stones": arch.max_stones,
            "max_windows": arch.max_windows,
            "max_rel_edges": arch.max_rel_edges,
            "lookahead_horizons": list(arch.lookahead_horizons),
        },
        "candidates": {"max_candidates": arch.max_candidates},
        "training": {"batch_size": 2, "amp": False},
        "samples": {"train_sample_count": 2},
        "selfplay": {"samples_per_epoch": 2, "games_per_epoch": 2, "search_visits": 1},
    }


def _action_ids(count: int) -> tuple[int, ...]:
    from hexo_models.hexformer_ar.coordinates import Axial, pack_action_id

    return tuple(pack_action_id(Axial(index, -index)) for index in range(count))


def _sparse_input(
    arch: object,
    *,
    candidate_count: int = 2,
    stone_count: int = 1,
    window_count: int = 1,
    local_count: int = 1,
    edge_count: int = 2,
    metadata: dict[str, object] | None = None,
):
    from hexo_models.hexformer_ar.input import SparseDecisionInput

    candidate_ids = _action_ids(candidate_count)
    candidate_features = torch.arange(
        candidate_count * arch.candidate_feature_dim,
        dtype=torch.float32,
    ).reshape(candidate_count, arch.candidate_feature_dim)
    policy = torch.ones((candidate_count,), dtype=torch.float32) / float(candidate_count)
    rel_edge_index = torch.tensor(
        [[index % (1 + local_count + candidate_count), (index + 1) % (1 + local_count + candidate_count)] for index in range(edge_count)],
        dtype=torch.long,
    )
    return SparseDecisionInput(
        candidate_action_ids=candidate_ids,
        candidate_features=candidate_features,
        candidate_coords=torch.stack(
            [
                torch.tensor([float(index), 0.0, float(-index), float(index), float(index)], dtype=torch.float32)
                for index in range(candidate_count)
            ],
        ),
        candidate_mask=torch.ones((candidate_count,), dtype=torch.bool),
        stone_features=torch.zeros((stone_count, arch.stone_feature_dim), dtype=torch.float32),
        stone_coords=torch.zeros((stone_count, 5), dtype=torch.float32),
        stone_mask=torch.ones((stone_count,), dtype=torch.bool),
        window_features=torch.zeros((window_count, arch.window_feature_dim), dtype=torch.float32),
        window_coords=torch.zeros((window_count, 5), dtype=torch.float32),
        window_mask=torch.ones((window_count,), dtype=torch.bool),
        local_input=torch.zeros((arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
        local_inputs=torch.zeros((local_count, arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
        local_window_coords=torch.zeros((local_count, 5), dtype=torch.float32),
        local_window_mask=torch.ones((local_count,), dtype=torch.bool),
        rel_edge_index=rel_edge_index,
        rel_edge_features=torch.zeros((edge_count, arch.rel_edge_feature_dim), dtype=torch.float32),
        rel_edge_mask=torch.ones((edge_count,), dtype=torch.bool),
        global_features=torch.zeros((arch.global_feature_dim,), dtype=torch.float32),
        policy_target=policy,
        opp_policy_target=policy.flip(0),
        wdl_target=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        distance_target=torch.tensor(0.1, dtype=torch.float32),
        threat_target=torch.zeros((candidate_count,), dtype=torch.long),
        relevance_target=torch.ones((candidate_count,), dtype=torch.float32),
        lookahead_targets={1: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)},
        metadata=metadata or {},
    )


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, *, policy_logits: torch.Tensor) -> None:
        super().__init__()
        self.policy_logits = policy_logits

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["candidate_features"].shape[0]
        candidate_count = batch["candidate_features"].shape[1]
        policy = self.policy_logits.to(batch["candidate_features"].device).expand(batch_size, candidate_count)
        return {
            "policy_logits": policy,
            "wdl_logits": torch.zeros((batch_size, 3), dtype=torch.float32, device=policy.device),
            "distance": torch.ones((batch_size,), dtype=torch.float32, device=policy.device),
        }


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")
