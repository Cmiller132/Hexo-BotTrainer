"""Synthetic tactical curriculum samples for Hexformer AR."""

from __future__ import annotations

from random import Random
from typing import Sequence

import torch

from .config import HexformerArchitectureConfig, HexformerCurriculumConfig
from .coordinates import Axial, pack_action_id, relative
from .input import SparseDecisionInput
from .samples import training_record_from_sparse_input


def generate_tactical_pretraining_records(
    *,
    count: int,
    architecture: HexformerArchitectureConfig,
    curriculum: HexformerCurriculumConfig,
    seed: int,
) -> tuple[object, ...]:
    """Build sparse records for win/block/double-threat warm starts."""

    rng = Random(int(seed) + int(curriculum.seed_offset))
    stages = tuple(curriculum.enabled_stages) or ("win_in_1",)
    records = []
    for index in range(max(0, int(count))):
        stage = stages[index % len(stages)]
        sparse = _synthetic_sparse(stage, architecture=architecture, max_span=curriculum.max_span, rng=rng)
        records.append(
            training_record_from_sparse_input(
                sparse,
                game_id=f"synthetic-{stage}-{index:08d}",
                turn_index=0,
                selected_action_id=sparse.candidate_action_ids[0],
                metadata={
                    "curriculum_stage": stage,
                    "source": "synthetic_tactical_pretraining",
                    "hard": True,
                },
            )
        )
    return tuple(records)


def _synthetic_sparse(
    stage: str,
    *,
    architecture: HexformerArchitectureConfig,
    max_span: int,
    rng: Random,
) -> SparseDecisionInput:
    axis = rng.choice((Axial(1, 0), Axial(0, 1), Axial(1, -1)))
    start = Axial(rng.randint(-max_span, max_span), rng.randint(-max_span, max_span))
    line = tuple(Axial(start.q + axis.q * offset, start.r + axis.r * offset) for offset in range(6))
    lines = [line]
    axes = [axis]
    starts = [start]
    targets = [line[5]]
    if stage == "double_threat":
        second_axis = rng.choice(tuple(item for item in (Axial(1, 0), Axial(0, 1), Axial(1, -1)) if item != axis))
        second_start = line[0]
        second_line = tuple(
            Axial(second_start.q + second_axis.q * offset, second_start.r + second_axis.r * offset)
            for offset in range(6)
        )
        lines.append(second_line)
        axes.append(second_axis)
        starts.append(second_start)
        targets.append(second_line[5])
    target = targets[0]
    alt = Axial(target.q + 1, target.r)
    candidates = _unique_coords((*targets, alt, Axial(start.q - axis.q, start.r - axis.r)))
    candidate_ids = tuple(pack_action_id(coord) for coord in candidates)
    own_count = 5 if stage in {"win_in_1", "double_threat"} else 0
    opp_count = 5 if stage == "block_in_1" else 0
    stones = _unique_coords(tuple(cell for candidate_line in lines for cell in candidate_line[:5]))
    anchor = line[2]
    candidate_coords = _coord_tensor(candidates, anchor)
    candidate_features = torch.zeros((len(candidates), architecture.candidate_feature_dim), dtype=torch.float32)
    candidate_features[:, 0] = 1.0
    for index, coord in enumerate(candidates):
        if coord in targets:
            candidate_features[index, 2] = 100.0 if coord == target else 95.0
            candidate_features[index, 3] = 1.0 if stage in {"win_in_1", "double_threat"} else 0.0
            candidate_features[index, 4] = 1.0 if stage == "block_in_1" else 0.0
    candidate_features[:, 5:10] = candidate_coords

    stone_features = torch.zeros((len(stones), architecture.stone_feature_dim), dtype=torch.float32)
    stone_coords = _coord_tensor(stones, anchor)
    for index in range(len(stones)):
        stone_features[index, 0 if stage != "block_in_1" else 1] = 1.0
        stone_features[index, 2:7] = stone_coords[index]

    window_features = torch.zeros((len(starts), architecture.window_feature_dim), dtype=torch.float32)
    window_coords = _coord_tensor(starts, anchor)
    for index, window_axis in enumerate(axes):
        axis_index = {(1, 0): 0, (0, 1): 1, (1, -1): 2}[(window_axis.q, window_axis.r)]
        window_features[index, axis_index] = 1.0
        window_features[index, 3] = float(own_count)
        window_features[index, 4] = float(opp_count)
        window_features[index, 5] = 1.0
        window_features[index, 6] = 1.0 if stage in {"win_in_1", "double_threat"} else 0.0
        window_features[index, 7] = 1.0 if stage == "block_in_1" else 0.0
        window_features[index, 8 if stage != "block_in_1" else 9] = 1.0
        window_features[index, 10:15] = window_coords[index]

    local_input = torch.zeros((architecture.local_input_channels, architecture.local_crop_size, architecture.local_crop_size), dtype=torch.float32)
    _write_local(local_input, stones, targets, anchor, opponent=(stage == "block_in_1"))
    rel_edge_index, rel_edge_features, rel_edge_mask = _synthetic_rel_edges(
        candidate_count=len(candidates),
        stone_count=len(stones),
        window_count=len(starts),
        target_count=len(targets),
        architecture=architecture,
    )
    policy = torch.zeros((len(candidates),), dtype=torch.float32)
    target_indices = tuple(index for index, coord in enumerate(candidates) if coord in targets)
    for index in target_indices:
        policy[index] = 1.0 / max(1, len(target_indices))
    threat = torch.zeros((len(candidates),), dtype=torch.long)
    for index in target_indices:
        threat[index] = 1 if stage != "block_in_1" else 2
    relevance = torch.zeros((len(candidates),), dtype=torch.float32)
    for index in target_indices:
        relevance[index] = 1.0
    return SparseDecisionInput(
        candidate_action_ids=candidate_ids,
        candidate_features=candidate_features,
        candidate_coords=candidate_coords,
        candidate_mask=torch.ones((len(candidates),), dtype=torch.bool),
        stone_features=stone_features,
        stone_coords=stone_coords,
        stone_mask=torch.ones((len(stones),), dtype=torch.bool),
        window_features=window_features,
        window_coords=window_coords,
        window_mask=torch.ones((len(starts),), dtype=torch.bool),
        local_input=local_input,
        global_features=torch.zeros((architecture.global_feature_dim,), dtype=torch.float32),
        local_inputs=local_input.unsqueeze(0),
        local_window_coords=torch.zeros((1, 5), dtype=torch.float32),
        local_window_mask=torch.ones((1,), dtype=torch.bool),
        rel_edge_index=rel_edge_index,
        rel_edge_features=rel_edge_features,
        rel_edge_mask=rel_edge_mask,
        policy_target=policy,
        opp_policy_target=None,
        wdl_target=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        distance_target=torch.tensor(1.0 / 128.0, dtype=torch.float32),
        threat_target=threat,
        relevance_target=relevance,
        metadata={
            "anchor": (anchor.q, anchor.r),
            "curriculum_stage": stage,
            "candidate": {"candidate_count": len(candidates), "legal_count": len(candidates)},
            "synthetic_threat_count": len(target_indices),
        },
    )


def _coord_tensor(coords: Sequence[Axial], anchor: Axial) -> torch.Tensor:
    out = torch.zeros((len(coords), 5), dtype=torch.float32)
    for index, coord in enumerate(coords):
        rel = relative(coord, anchor)
        out[index] = torch.tensor([rel.dq, rel.dr, rel.ds, rel.distance, rel.ring], dtype=torch.float32)
    return out


def _unique_coords(coords: Sequence[Axial]) -> tuple[Axial, ...]:
    out: list[Axial] = []
    for coord in coords:
        if coord not in out:
            out.append(coord)
    return tuple(out)


def _synthetic_rel_edges(
    *,
    candidate_count: int,
    stone_count: int,
    window_count: int,
    target_count: int,
    architecture: HexformerArchitectureConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_offset = 2
    window_offset = candidate_offset + int(candidate_count) + int(stone_count)
    edges: list[tuple[int, int]] = []
    for candidate_index in range(min(int(target_count), int(candidate_count))):
        for window_index in range(int(window_count)):
            if len(edges) + 2 > architecture.max_rel_edges:
                break
            candidate_token = candidate_offset + candidate_index
            window_token = window_offset + window_index
            edges.append((candidate_token, window_token))
            edges.append((window_token, candidate_token))
    edge_index = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 2), dtype=torch.long)
    edge_features = torch.zeros((len(edges), architecture.rel_edge_feature_dim), dtype=torch.float32)
    if edge_features.numel() > 0:
        edge_features[:, -1] = 1.0
        if architecture.rel_edge_feature_dim > 8:
            edge_features[:, 8] = 1.0
    return edge_index, edge_features, torch.ones((len(edges),), dtype=torch.bool)


def _write_local(local_input: torch.Tensor, stones: Sequence[Axial], targets: Sequence[Axial], anchor: Axial, *, opponent: bool) -> None:
    half = local_input.shape[-1] // 2
    local_input[2].fill_(1.0)
    for stone in stones:
        q = stone.q - anchor.q + half
        r = stone.r - anchor.r + half
        if 0 <= q < local_input.shape[-1] and 0 <= r < local_input.shape[-2]:
            local_input[1 if opponent else 0, r, q] = 1.0
            local_input[2, r, q] = 0.0
    for target in targets:
        q = target.q - anchor.q + half
        r = target.r - anchor.r + half
        if 0 <= q < local_input.shape[-1] and 0 <= r < local_input.shape[-2]:
            local_input[3, r, q] = 1.0
