from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_utils", "hexo_train", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_hexformer_forward_shapes() -> None:
    from hexo_models.hexformer_ar import HexformerAR, collate_sparse_inputs

    arch = _small_arch()
    model = HexformerAR(arch).eval()
    batch = collate_sparse_inputs((_sparse_input(arch),))

    with torch.no_grad():
        outputs = model(batch)

    assert outputs["policy_logits"].shape == (1, 2)
    assert outputs["wdl_logits"].shape == (1, 3)
    assert outputs["distance"].shape == (1,)
    assert outputs["opp_policy_logits"].shape == (1, 2)
    assert outputs["threat_logits"].shape == (1, 2, arch.threat_classes)
    assert outputs["lookahead_1"].shape == (1, 3)
    assert batch["rel_edge_index"].shape[-1] == 2
    assert batch["local_inputs"].shape[1] >= 1


def test_hexformer_sample_store_round_trip(tmp_path: Path) -> None:
    from hexo_models.hexformer_ar.samples import (
        SAMPLE_NAMESPACE,
        collate_compressed_samples,
        compressed_sample_from_training_record,
        sample_from_sparse_input,
        training_record_from_sample,
    )
    from hexo_utils.samples import SampleRequest, append_samples, open_sample_store, refresh_sample_index, sample_training_samples

    arch = _small_arch()
    sparse = _sparse_input(arch)
    sample = sample_from_sparse_input(
        sparse,
        game_id="game-a",
        turn_index=3,
        metadata={"selected_action_id": sparse.candidate_action_ids[0]},
    )
    record = training_record_from_sample(sample)
    store = open_sample_store(tmp_path / "samples", metadata={"run": "unit"})

    append_samples(store, (record,), metadata={"extensions": {SAMPLE_NAMESPACE: 1}})
    index = refresh_sample_index(store)
    batch = sample_training_samples(index, SampleRequest(count=1, required_extensions=(SAMPLE_NAMESPACE,)))

    [loaded] = batch.records
    compressed = compressed_sample_from_training_record(loaded)
    tensors = collate_compressed_samples((compressed,), architecture=arch)

    assert loaded.legal_action_ids == tuple(sample.input_payload["candidate_action_ids"])
    assert tensors["candidate_features"].shape == (1, 2, arch.candidate_feature_dim)
    assert tensors["policy_target"].shape == (1, 2)
    assert int(tensors["candidate_mask"].sum().item()) == 2


def test_hexformer_plugin_builds_training_components() -> None:
    from hexo_models.hexformer_ar.plugin import get_plugin
    from hexo_train.components import ComponentOverrides

    config = _small_config()
    plugin = get_plugin()
    model = plugin.build_model({}, config)
    overrides = plugin.training_component_overrides(
        defaults=SimpleNamespace(),
        config=config,
        shared=SimpleNamespace(),
        model=model,
    )

    assert isinstance(overrides, ComponentOverrides)
    assert overrides.trainer is not None
    assert overrides.optimizer is not None
    assert overrides.extra["model_family"] == "hexformer_ar"


def test_hexformer_d6_sparse_transform_round_trips() -> None:
    from hexo_models.hexformer_ar.augmentation import transform_sparse_input
    from hexo_models.hexformer_ar.d6 import inverse_index

    arch = _small_arch()
    sparse = _sparse_input(arch)
    sparse.window_features[0, 0] = 1.0
    transformed = transform_sparse_input(sparse, 3)
    restored = transform_sparse_input(transformed, inverse_index(3))

    assert restored.candidate_action_ids == sparse.candidate_action_ids
    assert torch.equal(restored.candidate_coords, sparse.candidate_coords)
    assert torch.equal(restored.local_input, sparse.local_input)
    assert torch.equal(restored.window_features[:, 0:3], sparse.window_features[:, 0:3])


def test_hexformer_collation_rebases_relation_edges_after_local_padding() -> None:
    from hexo_models.hexformer_ar import collate_sparse_inputs

    arch = _small_arch()
    single_local = replace(
        _sparse_input(arch),
        rel_edge_index=torch.tensor([[2, 3], [3, 4], [4, 5]], dtype=torch.long),
        rel_edge_features=torch.zeros((3, arch.rel_edge_feature_dim), dtype=torch.float32),
        rel_edge_mask=torch.ones((3,), dtype=torch.bool),
    )
    two_locals = replace(
        _sparse_input(arch),
        local_inputs=torch.zeros((2, arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
        local_window_coords=torch.zeros((2, 5), dtype=torch.float32),
        local_window_mask=torch.ones((2,), dtype=torch.bool),
        candidate_features=torch.ones((3, arch.candidate_feature_dim), dtype=torch.float32),
        candidate_coords=torch.zeros((3, 5), dtype=torch.float32),
        candidate_mask=torch.ones((3,), dtype=torch.bool),
        candidate_action_ids=tuple(range(3)),
        stone_features=torch.zeros((2, arch.stone_feature_dim), dtype=torch.float32),
        stone_coords=torch.zeros((2, 5), dtype=torch.float32),
        stone_mask=torch.ones((2,), dtype=torch.bool),
        rel_edge_index=torch.tensor([[3, 6], [6, 8], [8, 3]], dtype=torch.long),
        rel_edge_features=torch.zeros((3, arch.rel_edge_feature_dim), dtype=torch.float32),
        rel_edge_mask=torch.ones((3,), dtype=torch.bool),
        policy_target=torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32),
        opp_policy_target=torch.tensor([0.25, 0.25, 0.5], dtype=torch.float32),
        threat_target=torch.tensor([1, 0, 0], dtype=torch.long),
        relevance_target=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    )

    batch = collate_sparse_inputs((single_local, two_locals))

    assert torch.equal(batch["rel_edge_index"][0, 0], torch.tensor([3, 4]))
    assert torch.equal(batch["rel_edge_index"][0, 1], torch.tensor([4, 6]))
    assert torch.equal(batch["rel_edge_index"][0, 2], torch.tensor([6, 8]))
    assert torch.equal(batch["rel_edge_index"][1, 0], torch.tensor([3, 6]))
    assert torch.equal(batch["rel_edge_index"][1, 1], torch.tensor([6, 8]))
    assert torch.equal(batch["rel_edge_index"][1, 2], torch.tensor([8, 3]))


def test_hexformer_candidate_frontier_keeps_forced_moves() -> None:
    from hexo_models.hexformer_ar.candidates import TAG_IMMEDIATE_WIN, TAG_MUST_BLOCK, build_candidate_frontier
    from hexo_models.hexformer_ar.config import HexformerCandidateConfig
    from hexo_models.hexformer_ar.coordinates import Axial, pack_action_id

    win_id = pack_action_id(Axial(1, 0))
    block_id = pack_action_id(Axial(0, 1))
    state = SimpleNamespace(
        placement_history=(),
        board=SimpleNamespace(occupied=(Axial(0, 0),)),
    )
    result = build_candidate_frontier(
        state,
        (win_id, block_id, pack_action_id(Axial(2, 0))),
        immediate_win_action_ids=(win_id,),
        must_block_action_ids=(block_id,),
        config=HexformerCandidateConfig(max_candidates=2, require_tactical_candidates=True),
    )

    tags = {candidate.action_id: candidate.tags for candidate in result.candidates}
    assert tags[win_id] & TAG_IMMEDIATE_WIN
    assert tags[block_id] & TAG_MUST_BLOCK


def test_hexformer_curriculum_records_are_sparse_training_samples() -> None:
    from hexo_models.hexformer_ar.config import HexformerCurriculumConfig
    from hexo_models.hexformer_ar.curriculum import generate_tactical_pretraining_records
    from hexo_models.hexformer_ar.samples import SAMPLE_NAMESPACE

    records = generate_tactical_pretraining_records(
        count=3,
        architecture=_small_arch(),
        curriculum=HexformerCurriculumConfig(synthetic_samples=3),
        seed=7,
    )

    assert len(records) == 3
    assert all(record.model_payloads[0].namespace == SAMPLE_NAMESPACE for record in records)
    assert all(record.metadata["source"] == "synthetic_tactical_pretraining" for record in records)


def test_hexformer_double_threat_curriculum_has_two_targets() -> None:
    from hexo_models.hexformer_ar.config import HexformerCurriculumConfig
    from hexo_models.hexformer_ar.curriculum import generate_tactical_pretraining_records

    [record] = generate_tactical_pretraining_records(
        count=1,
        architecture=_small_arch(),
        curriculum=HexformerCurriculumConfig(synthetic_samples=1, enabled_stages=("double_threat",)),
        seed=11,
    )

    payload = record.model_payloads[0].payload
    policy = payload["policy_target"]["data"]
    assert payload["metadata"]["synthetic_threat_count"] == 2
    assert payload["window_features"]["shape"][0] == 2
    assert sum(1 for value in policy if value > 0.0) == 2


def _small_config() -> dict[str, object]:
    return {
        "architecture": {
            "local_channels": 8,
            "local_blocks": 1,
            "token_dim": 16,
            "gps_layers": 1,
            "attention_heads": 4,
            "dropout": 0.0,
            "local_crop_size": 9,
            "max_candidates": 4,
            "max_stones": 4,
            "max_windows": 4,
            "max_rel_edges": 8,
            "lookahead_horizons": [1],
        },
        "training": {"batch_size": 2, "learning_rate": 0.001, "amp": False},
        "samples": {"train_sample_count": 2},
        "selfplay": {"samples_per_epoch": 2, "games_per_epoch": 2, "search_visits": 1},
    }


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
        max_candidates=4,
        max_stones=4,
        max_windows=4,
        lookahead_horizons=(1,),
    )


def _sparse_input(arch):
    from hexo_models.hexformer_ar.coordinates import Axial, pack_action_id
    from hexo_models.hexformer_ar.input import SparseDecisionInput

    candidate_ids = (pack_action_id(Axial(0, 0)), pack_action_id(Axial(1, 0)))
    return SparseDecisionInput(
        candidate_action_ids=candidate_ids,
        candidate_features=torch.ones((2, arch.candidate_feature_dim), dtype=torch.float32),
        candidate_coords=torch.tensor([[0, 0, 0, 0, 0], [1, 0, -1, 1, 1]], dtype=torch.float32),
        candidate_mask=torch.ones((2,), dtype=torch.bool),
        stone_features=torch.zeros((1, arch.stone_feature_dim), dtype=torch.float32),
        stone_coords=torch.zeros((1, 5), dtype=torch.float32),
        stone_mask=torch.ones((1,), dtype=torch.bool),
        window_features=torch.zeros((1, arch.window_feature_dim), dtype=torch.float32),
        window_coords=torch.zeros((1, 5), dtype=torch.float32),
        window_mask=torch.ones((1,), dtype=torch.bool),
        local_input=torch.zeros((arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
        local_inputs=torch.zeros((1, arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
        local_window_coords=torch.zeros((1, 5), dtype=torch.float32),
        local_window_mask=torch.ones((1,), dtype=torch.bool),
        rel_edge_index=torch.tensor([[1, 2], [2, 1]], dtype=torch.long),
        rel_edge_features=torch.zeros((2, arch.rel_edge_feature_dim), dtype=torch.float32),
        rel_edge_mask=torch.ones((2,), dtype=torch.bool),
        global_features=torch.zeros((arch.global_feature_dim,), dtype=torch.float32),
        policy_target=torch.tensor([0.75, 0.25], dtype=torch.float32),
        opp_policy_target=torch.tensor([0.25, 0.75], dtype=torch.float32),
        wdl_target=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        distance_target=torch.tensor(0.1, dtype=torch.float32),
        threat_target=torch.tensor([1, 0], dtype=torch.long),
        relevance_target=torch.tensor([1.0, 0.0], dtype=torch.float32),
        lookahead_targets={1: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)},
    )
