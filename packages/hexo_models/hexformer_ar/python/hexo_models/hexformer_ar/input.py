"""Sparse state encoding and tensor collation for Hexformer AR."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, Sequence

import torch

from .config import HexformerArchitectureConfig, HexformerCandidateConfig

try:
    from hexo_models import _rust as _MODELS_RUST
except ImportError as exc:  # pragma: no cover - exercised only in broken installs.
    _MODELS_RUST = None
    _RUST_IMPORT_ERROR: BaseException | None = exc
else:
    _RUST_IMPORT_ERROR = None


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
    arch = architecture or HexformerArchitectureConfig()
    candidate_cfg = candidates or HexformerCandidateConfig(max_candidates=arch.max_candidates)
    payload = _hexformer_ar_rust().sparse_input_payload(
        state,
        _config_mapping(arch),
        _config_mapping(candidate_cfg),
        _policy_items(policy),
        _policy_items(opp_policy),
        None if value is None else float(value),
        None if distance is None else float(distance),
        _lookahead_items(lookahead),
        dict(metadata or {}),
    )
    return _sparse_input_from_payload(payload)


def build_sparse_inputs(
    states: Sequence[object],
    *,
    architecture: HexformerArchitectureConfig | None = None,
    candidates: HexformerCandidateConfig | None = None,
) -> tuple[SparseDecisionInput, ...]:
    """Build sparse inputs from live engine states through Rust."""

    arch = architecture or HexformerArchitectureConfig()
    candidate_cfg = candidates or HexformerCandidateConfig(max_candidates=arch.max_candidates)
    payloads = _hexformer_ar_rust().sparse_input_payloads(
        tuple(states),
        _config_mapping(arch),
        _config_mapping(candidate_cfg),
    )
    return tuple(_sparse_input_from_payload(payload) for payload in payloads)


def sparse_input_from_payload(payload: Mapping[str, Any]) -> SparseDecisionInput:
    """Convert a Rust-built sparse payload into tensor-backed Python input."""

    return _sparse_input_from_payload(payload)


def build_selfplay_sample_payloads(
    *,
    game_id: str,
    states: Sequence[object],
    players: Sequence[str],
    turn_indices: Sequence[int],
    visit_policies: Sequence[Mapping[int, float] | Sequence[tuple[int, float]]],
    root_values: Sequence[float],
    search_visits: Sequence[int],
    selected_action_ids: Sequence[int],
    winner: str | None,
    architecture: HexformerArchitectureConfig,
    candidates: HexformerCandidateConfig,
) -> tuple[Mapping[str, Any], ...]:
    """Finalize self-play sparse sample payloads through Rust."""

    return tuple(
        _hexformer_ar_rust().selfplay_sample_payloads(
            game_id,
            tuple(states),
            tuple(str(player) for player in players),
            tuple(int(turn_index) for turn_index in turn_indices),
            tuple(_policy_items(policy) for policy in visit_policies),
            tuple(float(value) for value in root_values),
            tuple(int(visits) for visits in search_visits),
            tuple(int(action_id) for action_id in selected_action_ids),
            None if winner is None else str(winner),
            _config_mapping(architecture),
            _config_mapping(candidates),
        )
    )


def _sparse_input_from_payload(payload: Mapping[str, Any]) -> SparseDecisionInput:
    return SparseDecisionInput(
        candidate_action_ids=tuple(int(item) for item in payload["candidate_action_ids"]),
        candidate_features=_tensor_from_payload(payload["candidate_features"]),
        candidate_coords=_tensor_from_payload(payload["candidate_coords"]),
        candidate_mask=_tensor_from_payload(payload["candidate_mask"]).bool(),
        stone_features=_tensor_from_payload(payload["stone_features"]),
        stone_coords=_tensor_from_payload(payload["stone_coords"]),
        stone_mask=_tensor_from_payload(payload["stone_mask"]).bool(),
        window_features=_tensor_from_payload(payload["window_features"]),
        window_coords=_tensor_from_payload(payload["window_coords"]),
        window_mask=_tensor_from_payload(payload["window_mask"]).bool(),
        local_input=_tensor_from_payload(payload["local_input"]),
        local_inputs=_tensor_from_payload(payload["local_inputs"]),
        local_window_coords=_tensor_from_payload(payload["local_window_coords"]),
        local_window_mask=_tensor_from_payload(payload["local_window_mask"]).bool(),
        rel_edge_index=_tensor_from_payload(payload["rel_edge_index"]).to(dtype=torch.long),
        rel_edge_features=_tensor_from_payload(payload["rel_edge_features"]),
        rel_edge_mask=_tensor_from_payload(payload["rel_edge_mask"]).bool(),
        global_features=_tensor_from_payload(payload["global_features"]),
        policy_target=_optional_tensor(payload, "policy_target"),
        opp_policy_target=_optional_tensor(payload, "opp_policy_target"),
        wdl_target=_optional_tensor(payload, "wdl_target"),
        distance_target=_optional_tensor(payload, "distance_target"),
        threat_target=_optional_tensor(payload, "threat_target", dtype=torch.long),
        relevance_target=_optional_tensor(payload, "relevance_target"),
        lookahead_targets={
            int(key): _tensor_from_payload(value)
            for key, value in dict(payload.get("lookahead_targets", {})).items()
        },
        metadata=dict(payload.get("metadata", {})),
    )


def _tensor_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
    dtype = _dtype(str(payload.get("dtype", "float32")))
    data = torch.tensor(payload.get("data", ()), dtype=dtype)
    shape = tuple(int(item) for item in payload.get("shape", (data.numel(),)))
    return data.reshape(shape)


def _optional_tensor(payload: Mapping[str, Any], key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if key not in payload:
        return None
    tensor = _tensor_from_payload(payload[key])
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _config_mapping(config: object) -> dict[str, Any]:
    if is_dataclass(config):
        return {field.name: getattr(config, field.name) for field in fields(config)}
    if isinstance(config, Mapping):
        return dict(config)
    return dict(getattr(config, "__dict__", {}))


def _policy_items(weights: Mapping[int, float] | Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    items = weights.items() if isinstance(weights, Mapping) else weights
    return tuple((int(action_id), float(weight)) for action_id, weight in items)


def _lookahead_items(weights: Mapping[int, float] | Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    items = weights.items() if isinstance(weights, Mapping) else weights
    return tuple((int(horizon), float(value)) for horizon, value in items)


def _hexformer_ar_rust() -> Any:
    module = getattr(_MODELS_RUST, "hexformer_ar", None) if _MODELS_RUST is not None else None
    if module is None:
        raise RuntimeError(f"hexformer_ar Rust sample generator is unavailable: {_RUST_IMPORT_ERROR}")
    return module


def _dtype(name: str) -> torch.dtype:
    return {
        "bool": torch.bool,
        "int8": torch.int8,
        "int64": torch.int64,
        "long": torch.long,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(name, torch.float32)


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
