"""Sparse state encoding and tensor collation for Hexformer AR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .candidates import CandidateSet, build_candidate_frontier
from .config import HexformerArchitectureConfig, HexformerCandidateConfig
from .coordinates import Axial, as_axial, choose_anchor, hex_distance, relative, unpack_action_id
from .windows import TacticalSummary, build_tactical_summary


@dataclass(frozen=True, slots=True)
class SparseDecisionInput:
    candidate_action_ids: tuple[int, ...]
    candidate_features: torch.Tensor
    candidate_coords: torch.Tensor
    candidate_mask: torch.Tensor
    stone_features: torch.Tensor
    stone_coords: torch.Tensor
    stone_mask: torch.Tensor
    window_features: torch.Tensor
    window_coords: torch.Tensor
    window_mask: torch.Tensor
    local_input: torch.Tensor
    global_features: torch.Tensor
    local_inputs: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0, 0, 0), dtype=torch.float32))
    local_window_coords: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 5), dtype=torch.float32))
    local_window_mask: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.bool))
    rel_edge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.long))
    rel_edge_features: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 12), dtype=torch.float32))
    rel_edge_mask: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.bool))
    policy_target: torch.Tensor | None = None
    opp_policy_target: torch.Tensor | None = None
    wdl_target: torch.Tensor | None = None
    distance_target: torch.Tensor | None = None
    threat_target: torch.Tensor | None = None
    relevance_target: torch.Tensor | None = None
    lookahead_targets: Mapping[int, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_sparse_input(
    state: object,
    *,
    architecture: HexformerArchitectureConfig | None = None,
    candidates: HexformerCandidateConfig | None = None,
    policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    opp_policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    value: float | None = None,
    distance: float | None = None,
    lookahead: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> SparseDecisionInput:
    import hexo_engine as engine

    arch = architecture or HexformerArchitectureConfig()
    candidate_cfg = candidates or HexformerCandidateConfig(max_candidates=arch.max_candidates)
    python_state = engine.to_python_state(state)
    legal_action_ids = tuple(int(action_id) for action_id in engine.legal_action_ids(state))
    tactical = build_tactical_summary(python_state, legal_action_ids)
    candidate_set = build_candidate_frontier(
        python_state,
        legal_action_ids,
        tactical_action_ids=tactical.tactical_action_ids,
        immediate_win_action_ids=tactical.immediate_win_action_ids,
        must_block_action_ids=tactical.must_block_action_ids,
        config=candidate_cfg,
    )
    opening = str(getattr(python_state.phase, "value", python_state.phase)) == "Opening"
    anchor = choose_anchor(getattr(python_state.board, "occupied", ()), opening=opening)
    return sparse_input_from_python_state(
        python_state,
        candidate_set,
        tactical,
        anchor=anchor,
        architecture=arch,
        policy=policy,
        opp_policy=opp_policy,
        value=value,
        distance=distance,
        lookahead=lookahead,
        metadata=metadata,
    )


def sparse_input_from_python_state(
    python_state: object,
    candidate_set: CandidateSet,
    tactical: TacticalSummary,
    *,
    anchor: Axial,
    architecture: HexformerArchitectureConfig,
    policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    opp_policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    value: float | None = None,
    distance: float | None = None,
    lookahead: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> SparseDecisionInput:
    candidates = candidate_set.candidates[: architecture.max_candidates]
    stones = tuple(getattr(python_state.board, "stones", ()))[: architecture.max_stones]
    windows = tactical.windows[: architecture.max_windows]
    current = _player_label(getattr(python_state, "current_player", "player0"))
    phase = str(getattr(getattr(python_state, "phase", ""), "value", getattr(python_state, "phase", "")))

    candidate_features = torch.zeros((len(candidates), architecture.candidate_feature_dim), dtype=torch.float32)
    candidate_coords = torch.zeros((len(candidates), 5), dtype=torch.float32)
    for index, candidate in enumerate(candidates):
        rel = relative(candidate.coord, anchor)
        candidate_coords[index] = torch.tensor([rel.dq, rel.dr, rel.ds, rel.distance, rel.ring], dtype=torch.float32)
        candidate_features[index, 0] = 1.0
        candidate_features[index, 1] = float(candidate.tags)
        candidate_features[index, 2] = float(candidate.priority)
        candidate_features[index, 3] = 1.0 if candidate.action_id in tactical.immediate_win_action_ids else 0.0
        candidate_features[index, 4] = 1.0 if candidate.action_id in tactical.must_block_action_ids else 0.0
        candidate_features[index, 5:10] = candidate_coords[index]

    stone_features = torch.zeros((len(stones), architecture.stone_feature_dim), dtype=torch.float32)
    stone_coords = torch.zeros((len(stones), 5), dtype=torch.float32)
    for index, item in enumerate(stones):
        coord = as_axial(item[0])
        player = _player_label(item[1])
        rel = relative(coord, anchor)
        stone_coords[index] = torch.tensor([rel.dq, rel.dr, rel.ds, rel.distance, rel.ring], dtype=torch.float32)
        stone_features[index, 0] = 1.0 if player == current else 0.0
        stone_features[index, 1] = 1.0 if player != current else 0.0
        stone_features[index, 2:7] = stone_coords[index]

    window_features = torch.zeros((len(windows), architecture.window_feature_dim), dtype=torch.float32)
    window_coords = torch.zeros((len(windows), 5), dtype=torch.float32)
    axis_to_index = {"Q": 0, "R": 1, "QR": 2}
    for index, window in enumerate(windows):
        rel = relative(window.start, anchor)
        window_coords[index] = torch.tensor([rel.dq, rel.dr, rel.ds, rel.distance, rel.ring], dtype=torch.float32)
        window_features[index, axis_to_index.get(window.axis, 0)] = 1.0
        window_features[index, 3] = float(window.counts[0])
        window_features[index, 4] = float(window.counts[1])
        window_features[index, 5] = float(len(window.empty_cells))
        window_features[index, 6] = 1.0 if window.threat_player == current else 0.0
        window_features[index, 7] = 1.0 if window.threat_player is not None and window.threat_player != current else 0.0
        window_features[index, 8] = float(len(window.immediate_win_action_ids))
        window_features[index, 9] = float(len(window.must_block_action_ids))
        window_features[index, 10:15] = window_coords[index]

    candidate_ids = tuple(candidate.action_id for candidate in candidates)
    local_inputs, local_window_coords, local_window_mask = _build_local_windows(
        python_state,
        candidate_ids,
        tactical,
        anchor,
        architecture,
    )
    local_input = local_inputs[0] if local_inputs.shape[0] else torch.zeros(
        (architecture.local_input_channels, architecture.local_crop_size, architecture.local_crop_size),
        dtype=torch.float32,
    )
    rel_edge_index, rel_edge_features, rel_edge_mask = _build_rel_edges(
        candidates,
        stones,
        windows,
        local_count=int(local_inputs.shape[0]),
        architecture=architecture,
    )
    global_features = _global_features(python_state, anchor, len(candidate_ids), phase, current, architecture.global_feature_dim)
    wdl = _wdl_target(value)
    candidate_mask = torch.ones((len(candidates),), dtype=torch.bool)
    return SparseDecisionInput(
        candidate_action_ids=candidate_ids,
        candidate_features=candidate_features,
        candidate_coords=candidate_coords,
        candidate_mask=candidate_mask,
        stone_features=stone_features,
        stone_coords=stone_coords,
        stone_mask=torch.ones((len(stones),), dtype=torch.bool),
        window_features=window_features,
        window_coords=window_coords,
        window_mask=torch.ones((len(windows),), dtype=torch.bool),
        local_input=local_input,
        local_inputs=local_inputs,
        local_window_coords=local_window_coords,
        local_window_mask=local_window_mask,
        rel_edge_index=rel_edge_index,
        rel_edge_features=rel_edge_features,
        rel_edge_mask=rel_edge_mask,
        global_features=global_features,
        policy_target=_policy_vector(candidate_ids, policy),
        opp_policy_target=_policy_vector(candidate_ids, opp_policy),
        wdl_target=wdl,
        distance_target=(torch.tensor(float(distance), dtype=torch.float32) if distance is not None else None),
        threat_target=_threat_targets(candidate_ids, tactical),
        relevance_target=_relevance_targets(candidate_ids, tactical),
        lookahead_targets={
            int(key): _wdl_target(float(item))
            for key, item in (lookahead.items() if isinstance(lookahead, Mapping) else lookahead)
        },
        metadata={
            **dict(metadata or {}),
            "anchor": (anchor.q, anchor.r),
            "candidate": dict(candidate_set.metadata),
            "tactical": dict(tactical.metadata),
        },
    )


def collate_sparse_inputs(samples: Sequence[SparseDecisionInput]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty sparse batch")
    max_candidates = max(sample.candidate_features.shape[0] for sample in samples)
    max_stones = max(sample.stone_features.shape[0] for sample in samples)
    max_windows = max(sample.window_features.shape[0] for sample in samples)
    batch: dict[str, torch.Tensor] = {
        "candidate_features": _pad_2d([s.candidate_features for s in samples], max_candidates),
        "candidate_coords": _pad_2d([s.candidate_coords for s in samples], max_candidates),
        "candidate_mask": _pad_1d([s.candidate_mask for s in samples], max_candidates, dtype=torch.bool),
        "stone_features": _pad_2d([s.stone_features for s in samples], max_stones),
        "stone_coords": _pad_2d([s.stone_coords for s in samples], max_stones),
        "stone_mask": _pad_1d([s.stone_mask for s in samples], max_stones, dtype=torch.bool),
        "window_features": _pad_2d([s.window_features for s in samples], max_windows),
        "window_coords": _pad_2d([s.window_coords for s in samples], max_windows),
        "window_mask": _pad_1d([s.window_mask for s in samples], max_windows, dtype=torch.bool),
        "local_input": torch.stack([s.local_input for s in samples], dim=0),
        "local_inputs": _pad_local_inputs(samples),
        "local_window_coords": _pad_2d(
            [_local_window_coords(s) for s in samples],
            max(_local_window_coords(s).shape[0] for s in samples),
        ),
        "local_window_mask": _pad_1d([_local_window_mask(s) for s in samples], max(_local_window_mask(s).shape[0] for s in samples), dtype=torch.bool),
        "rel_edge_index": _pad_edge_index_for_batch(samples),
        "rel_edge_features": _pad_2d([s.rel_edge_features for s in samples], max(s.rel_edge_features.shape[0] for s in samples)),
        "rel_edge_mask": _pad_1d([s.rel_edge_mask for s in samples], max(s.rel_edge_mask.shape[0] for s in samples), dtype=torch.bool),
        "global_features": torch.stack([s.global_features for s in samples], dim=0),
    }
    if all(sample.policy_target is not None for sample in samples):
        batch["policy_target"] = _pad_1d([s.policy_target for s in samples if s.policy_target is not None], max_candidates)
    if all(sample.opp_policy_target is not None for sample in samples):
        batch["opp_policy_target"] = _pad_1d([s.opp_policy_target for s in samples if s.opp_policy_target is not None], max_candidates)
    if all(sample.wdl_target is not None for sample in samples):
        batch["wdl_target"] = torch.stack([s.wdl_target for s in samples if s.wdl_target is not None], dim=0)
    if all(sample.distance_target is not None for sample in samples):
        batch["distance_target"] = torch.stack([s.distance_target for s in samples if s.distance_target is not None], dim=0)
    if all(sample.threat_target is not None for sample in samples):
        batch["threat_target"] = _pad_1d([s.threat_target for s in samples if s.threat_target is not None], max_candidates, dtype=torch.long)
    if all(sample.relevance_target is not None for sample in samples):
        batch["relevance_target"] = _pad_1d([s.relevance_target for s in samples if s.relevance_target is not None], max_candidates)
    horizons = sorted(set().union(*(sample.lookahead_targets.keys() for sample in samples)))
    for horizon in horizons:
        if all(horizon in sample.lookahead_targets for sample in samples):
            batch[f"lookahead_{horizon}_target"] = torch.stack(
                [sample.lookahead_targets[horizon] for sample in samples],
                dim=0,
            )
    return batch


def _fill_local_crop(local_input: torch.Tensor, python_state: object, candidate_ids: Sequence[int], anchor: Axial) -> None:
    half = local_input.shape[-1] // 2
    current = _player_label(getattr(python_state, "current_player", "player0"))
    local_input[2].fill_(1.0)
    for coord, player in getattr(python_state.board, "stones", ()):
        q = int(coord.q) - anchor.q + half
        r = int(coord.r) - anchor.r + half
        if 0 <= q < local_input.shape[-1] and 0 <= r < local_input.shape[-2]:
            plane = 0 if _player_label(player) == current else 1
            local_input[plane, r, q] = 1.0
            local_input[2, r, q] = 0.0
    for action_id in candidate_ids:
        coord = unpack_action_id(action_id)
        q = coord.q - anchor.q + half
        r = coord.r - anchor.r + half
        if 0 <= q < local_input.shape[-1] and 0 <= r < local_input.shape[-2]:
            local_input[3, r, q] = 1.0
    if str(getattr(getattr(python_state, "phase", ""), "value", getattr(python_state, "phase", ""))) == "SecondStone":
        local_input[4].fill_(1.0)


def _build_local_windows(
    python_state: object,
    candidate_ids: Sequence[int],
    tactical: TacticalSummary,
    anchor: Axial,
    architecture: HexformerArchitectureConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors: list[Axial] = [anchor]
    history = tuple(getattr(python_state, "placement_history", ()))
    if history:
        last_coord = getattr(history[-1], "coord", None)
        if last_coord is not None:
            anchors.append(as_axial(last_coord))
    threat_windows = [
        window for window in tactical.windows
        if window.immediate_win_action_ids or window.must_block_action_ids or window.threat_player is not None
    ]
    if threat_windows:
        anchors.append(threat_windows[0].start)

    unique: list[Axial] = []
    for item in anchors:
        if item not in unique:
            unique.append(item)
        if len(unique) >= architecture.max_local_windows:
            break
    if not unique:
        unique = [anchor]

    local_inputs = torch.zeros(
        (
            len(unique),
            architecture.local_input_channels,
            architecture.local_crop_size,
            architecture.local_crop_size,
        ),
        dtype=torch.float32,
    )
    local_coords = torch.zeros((len(unique), 5), dtype=torch.float32)
    for index, local_anchor in enumerate(unique):
        _fill_local_crop(local_inputs[index], python_state, candidate_ids, local_anchor)
        rel = relative(local_anchor, anchor)
        local_coords[index] = torch.tensor([rel.dq, rel.dr, rel.ds, rel.distance, rel.ring], dtype=torch.float32)
    return local_inputs, local_coords, torch.ones((len(unique),), dtype=torch.bool)


def _build_rel_edges(
    candidates: Sequence[object],
    stones: Sequence[object],
    windows: Sequence[object],
    *,
    local_count: int,
    architecture: HexformerArchitectureConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_offset = 1 + max(0, int(local_count))
    stone_offset = candidate_offset + len(candidates)
    window_offset = stone_offset + len(stones)
    edges: list[tuple[int, int, list[float]]] = []

    def add(src: int, dst: int, left: Axial, right: Axial, relation_type: int) -> None:
        if len(edges) >= architecture.max_rel_edges:
            return
        dq = right.q - left.q
        dr = right.r - left.r
        ds = -dq - dr
        dist = max(abs(dq), abs(dr), abs(ds))
        same_q = 1.0 if dq == 0 else 0.0
        same_r = 1.0 if dr == 0 else 0.0
        same_qr = 1.0 if ds == 0 else 0.0
        features = [
            float(dq),
            float(dr),
            float(ds),
            float(dist),
            same_q,
            same_r,
            same_qr,
            1.0 if relation_type == 0 else 0.0,
            1.0 if relation_type == 1 else 0.0,
            1.0 if relation_type == 2 else 0.0,
            1.0 if relation_type == 3 else 0.0,
            1.0,
        ]
        edges.append((src, dst, features[: architecture.rel_edge_feature_dim]))

    candidate_coords = [item.coord for item in candidates]
    stone_coords = [as_axial(item[0]) for item in stones]
    window_coords = [item.start for item in windows]
    for ci, ccoord in enumerate(candidate_coords):
        ctoken = candidate_offset + ci
        for si, scoord in enumerate(stone_coords):
            if hex_distance(ccoord, scoord) <= 4:
                stoken = stone_offset + si
                add(ctoken, stoken, ccoord, scoord, 0)
                add(stoken, ctoken, scoord, ccoord, 0)
        for wi, window in enumerate(windows):
            if ccoord in window.empty_cells or ccoord in window.cells or hex_distance(ccoord, window.start) <= 3:
                wtoken = window_offset + wi
                add(ctoken, wtoken, ccoord, window.start, 1)
                add(wtoken, ctoken, window.start, ccoord, 1)
    for wi, wcoord in enumerate(window_coords):
        wtoken = window_offset + wi
        for si, scoord in enumerate(stone_coords):
            if hex_distance(wcoord, scoord) <= 6:
                stoken = stone_offset + si
                add(wtoken, stoken, wcoord, scoord, 2)
                add(stoken, wtoken, scoord, wcoord, 2)
    for left in range(len(candidate_coords)):
        for right in range(left + 1, min(len(candidate_coords), left + 32)):
            if hex_distance(candidate_coords[left], candidate_coords[right]) <= 2:
                add(candidate_offset + left, candidate_offset + right, candidate_coords[left], candidate_coords[right], 3)
                add(candidate_offset + right, candidate_offset + left, candidate_coords[right], candidate_coords[left], 3)

    edge_index = torch.zeros((len(edges), 2), dtype=torch.long)
    edge_features = torch.zeros((len(edges), architecture.rel_edge_feature_dim), dtype=torch.float32)
    for index, (src, dst, features) in enumerate(edges):
        edge_index[index] = torch.tensor([src, dst], dtype=torch.long)
        edge_features[index, : len(features)] = torch.tensor(features, dtype=torch.float32)
    return edge_index, edge_features, torch.ones((len(edges),), dtype=torch.bool)


def _global_features(
    python_state: object,
    anchor: Axial,
    candidate_count: int,
    phase: str,
    current: str,
    feature_dim: int,
) -> torch.Tensor:
    out = torch.zeros((feature_dim,), dtype=torch.float32)
    phase_map = {"Opening": 0, "FirstStone": 1, "SecondStone": 2}
    out[phase_map.get(phase, 0)] = 1.0
    out[3] = 1.0 if current == "player0" else 0.0
    out[4] = float(getattr(python_state, "placements_made", 0))
    out[5] = float(candidate_count)
    occupied = tuple(getattr(getattr(python_state, "board", object()), "occupied", ()))
    out[6] = float(len(occupied))
    out[7] = float(max((relative(coord, anchor).distance for coord in occupied), default=0))
    first = getattr(python_state, "first_stone", None)
    if first is not None:
        rel = relative(first, anchor)
        out[8:13] = torch.tensor([rel.dq, rel.dr, rel.ds, rel.distance, rel.ring], dtype=torch.float32)
    return out


def _policy_vector(action_ids: Sequence[int], weights: Mapping[int, float] | Sequence[tuple[int, float]]) -> torch.Tensor | None:
    items = dict((int(action), float(weight)) for action, weight in (weights.items() if isinstance(weights, Mapping) else weights))
    if not items:
        return None
    target = torch.tensor([max(0.0, items.get(int(action_id), 0.0)) for action_id in action_ids], dtype=torch.float32)
    total = target.sum()
    if float(total) <= 0.0:
        return None
    return target / total


def _wdl_target(value: float | None) -> torch.Tensor | None:
    if value is None:
        return None
    v = max(-1.0, min(1.0, float(value)))
    win = max(0.0, v)
    loss = max(0.0, -v)
    draw = max(0.0, 1.0 - win - loss)
    target = torch.tensor([loss, draw, win], dtype=torch.float32)
    return target / target.sum().clamp_min(1.0e-8)


def _threat_targets(action_ids: Sequence[int], tactical: TacticalSummary) -> torch.Tensor:
    target = torch.zeros((len(action_ids),), dtype=torch.long)
    win_ids = set(tactical.immediate_win_action_ids)
    block_ids = set(tactical.must_block_action_ids)
    tactical_ids = set(tactical.tactical_action_ids)
    for index, action_id in enumerate(action_ids):
        if action_id in win_ids:
            target[index] = 1
        elif action_id in block_ids:
            target[index] = 2
        elif action_id in tactical_ids:
            target[index] = 3
    return target


def _relevance_targets(action_ids: Sequence[int], tactical: TacticalSummary) -> torch.Tensor:
    tactical_ids = set(tactical.tactical_action_ids)
    win_ids = set(tactical.immediate_win_action_ids)
    block_ids = set(tactical.must_block_action_ids)
    return torch.tensor(
        [1.0 if action_id in tactical_ids or action_id in win_ids or action_id in block_ids else 0.0 for action_id in action_ids],
        dtype=torch.float32,
    )


def _pad_2d(values: Sequence[torch.Tensor], length: int) -> torch.Tensor:
    dim = int(values[0].shape[-1])
    out = torch.zeros((len(values), length, dim), dtype=values[0].dtype)
    for index, value in enumerate(values):
        out[index, : value.shape[0], :] = value
    return out


def _pad_edge_index(values: Sequence[torch.Tensor], length: int) -> torch.Tensor:
    out = torch.zeros((len(values), length, 2), dtype=torch.long)
    for index, value in enumerate(values):
        if value.numel() > 0:
            out[index, : value.shape[0], :] = value.to(dtype=torch.long)
    return out


def _pad_edge_index_for_batch(samples: Sequence[SparseDecisionInput]) -> torch.Tensor:
    max_edges = max(sample.rel_edge_index.shape[0] for sample in samples)
    max_local_windows = max(_local_inputs(sample).shape[0] for sample in samples)
    max_candidates = max(sample.candidate_features.shape[0] for sample in samples)
    max_stones = max(sample.stone_features.shape[0] for sample in samples)
    values: list[torch.Tensor] = []
    for sample in samples:
        edges = sample.rel_edge_index.to(dtype=torch.long)
        if edges.numel() == 0:
            values.append(edges)
            continue
        local_count = _local_inputs(sample).shape[0]
        candidate_count = sample.candidate_features.shape[0]
        stone_count = sample.stone_features.shape[0]
        edges = edges.clone()
        for column in range(2):
            edges[:, column] = _rebase_token_indices(
                edges[:, column],
                local_count=local_count,
                candidate_count=candidate_count,
                stone_count=stone_count,
                max_local_windows=max_local_windows,
                max_candidates=max_candidates,
                max_stones=max_stones,
            )
        values.append(edges)
    return _pad_edge_index(values, max_edges)


def _rebase_token_indices(
    indices: torch.Tensor,
    *,
    local_count: int,
    candidate_count: int,
    stone_count: int,
    max_local_windows: int,
    max_candidates: int,
    max_stones: int,
) -> torch.Tensor:
    out = indices.clone()
    candidate_start = 1 + int(local_count)
    stone_start = candidate_start + int(candidate_count)
    window_start = stone_start + int(stone_count)
    local_delta = int(max_local_windows) - int(local_count)
    candidate_delta = int(max_candidates) - int(candidate_count)
    stone_delta = int(max_stones) - int(stone_count)
    candidate_mask = (out >= candidate_start) & (out < stone_start)
    stone_mask = (out >= stone_start) & (out < window_start)
    window_mask = out >= window_start
    out[candidate_mask] += local_delta
    out[stone_mask] += local_delta + candidate_delta
    out[window_mask] += local_delta + candidate_delta + stone_delta
    return out


def _pad_local_inputs(samples: Sequence[SparseDecisionInput]) -> torch.Tensor:
    local_values = [_local_inputs(sample) for sample in samples]
    length = max(value.shape[0] for value in local_values)
    channels, height, width = local_values[0].shape[1:]
    out = torch.zeros((len(samples), length, channels, height, width), dtype=local_values[0].dtype)
    for index, value in enumerate(local_values):
        out[index, : value.shape[0]] = value
    return out


def _local_inputs(sample: SparseDecisionInput) -> torch.Tensor:
    if sample.local_inputs.numel() > 0:
        return sample.local_inputs
    return sample.local_input.unsqueeze(0)


def _local_window_coords(sample: SparseDecisionInput) -> torch.Tensor:
    if sample.local_window_coords.numel() > 0:
        return sample.local_window_coords
    return torch.zeros((1, 5), dtype=sample.local_input.dtype)


def _local_window_mask(sample: SparseDecisionInput) -> torch.Tensor:
    if sample.local_window_mask.numel() > 0:
        return sample.local_window_mask
    return torch.ones((1,), dtype=torch.bool)


def _pad_1d(values: Sequence[torch.Tensor], length: int, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    resolved_dtype = dtype or values[0].dtype
    out = torch.zeros((len(values), length), dtype=resolved_dtype)
    for index, value in enumerate(values):
        out[index, : value.shape[0]] = value.to(dtype=resolved_dtype)
    return out


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
