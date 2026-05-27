"""Sparse sample serialization and replay buffer helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from random import Random
from typing import Any, Mapping, Sequence
import json
import zlib

import torch

from hexo_utils.samples import ModelSamplePayload, PolicyOutputRecord, TrainingSampleRecord

from .augmentation import transform_sparse_input
from .config import HexformerArchitectureConfig
from .input import SparseDecisionInput, collate_sparse_inputs


SAMPLE_NAMESPACE = "hexo_models.hexformer_ar"
SAMPLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class HexformerSample:
    game_id: str
    turn_index: int
    input_payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompressedHexformerSample:
    payload: bytes
    uncompressed_bytes: int
    compression: str = "zlib+json"
    compressed: bool = True

    @classmethod
    def from_sample(cls, sample: HexformerSample, *, compression_level: int = 6) -> "CompressedHexformerSample":
        raw = json.dumps(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "game_id": sample.game_id,
                "turn_index": sample.turn_index,
                "input_payload": sample.input_payload,
                "metadata": dict(sample.metadata),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return cls(payload=zlib.compress(raw, level=int(compression_level)), uncompressed_bytes=len(raw))

    def decode(self) -> HexformerSample:
        raw = json.loads(zlib.decompress(self.payload).decode("utf-8"))
        return HexformerSample(
            game_id=str(raw["game_id"]),
            turn_index=int(raw["turn_index"]),
            input_payload=dict(raw["input_payload"]),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class HexformerReplayBuffer:
    capacity: int = 200_000
    recency_halflife: float = 100_000.0
    compression_level: int = 6
    seed: int | None = None
    _samples: list[CompressedHexformerSample] = field(default_factory=list)
    _total_appended: int = 0
    _draw_count: int = 0

    def append(self, sample: HexformerSample | CompressedHexformerSample) -> None:
        compressed = (
            sample
            if isinstance(sample, CompressedHexformerSample)
            else CompressedHexformerSample.from_sample(sample, compression_level=self.compression_level)
        )
        self._samples.append(compressed)
        self._total_appended += 1
        overflow = len(self._samples) - max(1, int(self.capacity))
        if overflow > 0:
            del self._samples[:overflow]

    def extend(self, samples: Sequence[HexformerSample | CompressedHexformerSample]) -> None:
        for sample in samples:
            self.append(sample)

    def sample(self, count: int, *, seed: int | None = None) -> tuple[CompressedHexformerSample, ...]:
        if count <= 0 or not self._samples:
            return ()
        resolved_seed = self.seed if seed is None else seed
        rng = Random(None if resolved_seed is None else int(resolved_seed) + self._draw_count)
        self._draw_count += 1
        selected = _weighted_without_replacement(
            len(self._samples),
            min(int(count), len(self._samples)),
            lambda index: self._recency_weight(index),
            rng,
        )
        return tuple(self._samples[index] for index in selected)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def total_appended(self) -> int:
        return self._total_appended

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "recency_halflife": float(self.recency_halflife),
            "compression_level": int(self.compression_level),
            "total_appended": int(self._total_appended),
            "draw_count": int(self._draw_count),
            "samples": [
                {
                    "payload": sample.payload,
                    "uncompressed_bytes": int(sample.uncompressed_bytes),
                    "compression": sample.compression,
                    "compressed": bool(sample.compressed),
                }
                for sample in self._samples
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.recency_halflife = float(state.get("recency_halflife", self.recency_halflife))
        self.compression_level = int(state.get("compression_level", self.compression_level))
        self._total_appended = int(state.get("total_appended", 0))
        self._draw_count = int(state.get("draw_count", 0))
        self._samples = [
            CompressedHexformerSample(
                payload=bytes(item["payload"]),
                uncompressed_bytes=int(item["uncompressed_bytes"]),
                compression=str(item.get("compression", "zlib+json")),
                compressed=bool(item.get("compressed", True)),
            )
            for item in state.get("samples", ())
        ][-self.capacity :]

    def _recency_weight(self, index: int) -> float:
        age = len(self._samples) - 1 - index
        return exp(-log(2.0) * age / max(1.0e-6, self.recency_halflife))


@dataclass(frozen=True, slots=True)
class SparseSampleWindow:
    records: tuple[CompressedHexformerSample, ...]
    seed: int
    epoch: int
    index: object
    window_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def sample_from_sparse_input(
    sparse_input: SparseDecisionInput,
    *,
    game_id: str,
    turn_index: int,
    metadata: Mapping[str, Any] | None = None,
) -> HexformerSample:
    return HexformerSample(
        game_id=game_id,
        turn_index=int(turn_index),
        input_payload=sparse_input_to_payload(sparse_input),
        metadata={**dict(sparse_input.metadata), **dict(metadata or {})},
    )


def training_record_from_sample(
    sample: HexformerSample,
    *,
    model_id: str = "hexformer_ar",
    selected_action_id: int | None = None,
    source_record_ref: object | None = None,
) -> TrainingSampleRecord:
    if selected_action_id is None and sample.metadata.get("selected_action_id") is not None:
        selected_action_id = int(sample.metadata["selected_action_id"])
    policy_payload = sample.input_payload.get("policy_target")
    policy = None
    if isinstance(policy_payload, Mapping):
        policy = PolicyOutputRecord(
            game_id=sample.game_id,
            turn_index=sample.turn_index,
            model_id=model_id,
            selected_action_id=selected_action_id,
            logits=tuple(float(item) for item in policy_payload.get("data", ())),
            value=_sample_value(sample.input_payload),
            metadata={"target": "search_visit_policy"},
        )
    return TrainingSampleRecord(
        game_id=sample.game_id,
        turn_index=sample.turn_index,
        legal_action_ids=tuple(int(item) for item in sample.input_payload.get("candidate_action_ids", ())),
        source_record_ref=source_record_ref,
        policy=policy,
        model_payloads=(
            ModelSamplePayload(
                game_id=sample.game_id,
                turn_index=sample.turn_index,
                model_id=model_id,
                namespace=SAMPLE_NAMESPACE,
                schema_version=SAMPLE_SCHEMA_VERSION,
                payload=dict(sample.input_payload),
            ),
        ),
        metadata={
            **dict(sample.metadata),
            "model_family": "hexformer_ar",
            "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        },
    )


def training_record_from_sparse_input(
    sparse_input: SparseDecisionInput,
    *,
    game_id: str,
    turn_index: int,
    model_id: str = "hexformer_ar",
    selected_action_id: int | None = None,
    source_record_ref: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingSampleRecord:
    return training_record_from_sample(
        sample_from_sparse_input(sparse_input, game_id=game_id, turn_index=turn_index, metadata=metadata),
        model_id=model_id,
        selected_action_id=selected_action_id,
        source_record_ref=source_record_ref,
    )


def compressed_sample_from_training_record(record: TrainingSampleRecord) -> CompressedHexformerSample:
    payload = _hexformer_payload(record)
    sample = HexformerSample(
        game_id=record.game_id,
        turn_index=record.turn_index,
        input_payload=dict(payload.payload),
        metadata=dict(record.metadata),
    )
    return CompressedHexformerSample.from_sample(sample)


def sparse_input_from_training_record(record: TrainingSampleRecord) -> SparseDecisionInput:
    payload = _hexformer_payload(record)
    return sparse_input_from_payload(payload.payload)


def sparse_input_to_payload(sample: SparseDecisionInput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_action_ids": list(sample.candidate_action_ids),
        "candidate_features": _tensor_payload(sample.candidate_features),
        "candidate_coords": _tensor_payload(sample.candidate_coords),
        "candidate_mask": _tensor_payload(sample.candidate_mask.to(dtype=torch.int8)),
        "stone_features": _tensor_payload(sample.stone_features),
        "stone_coords": _tensor_payload(sample.stone_coords),
        "stone_mask": _tensor_payload(sample.stone_mask.to(dtype=torch.int8)),
        "window_features": _tensor_payload(sample.window_features),
        "window_coords": _tensor_payload(sample.window_coords),
        "window_mask": _tensor_payload(sample.window_mask.to(dtype=torch.int8)),
        "local_input": _tensor_payload(sample.local_input),
        "local_inputs": _tensor_payload(sample.local_inputs),
        "local_window_coords": _tensor_payload(sample.local_window_coords),
        "local_window_mask": _tensor_payload(sample.local_window_mask.to(dtype=torch.int8)),
        "rel_edge_index": _tensor_payload(sample.rel_edge_index),
        "rel_edge_features": _tensor_payload(sample.rel_edge_features),
        "rel_edge_mask": _tensor_payload(sample.rel_edge_mask.to(dtype=torch.int8)),
        "global_features": _tensor_payload(sample.global_features),
        "metadata": dict(sample.metadata),
    }
    optional = {
        "policy_target": sample.policy_target,
        "opp_policy_target": sample.opp_policy_target,
        "wdl_target": sample.wdl_target,
        "distance_target": sample.distance_target,
        "threat_target": sample.threat_target,
        "relevance_target": sample.relevance_target,
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = _tensor_payload(value)
    if sample.lookahead_targets:
        payload["lookahead_targets"] = {str(key): _tensor_payload(value) for key, value in sample.lookahead_targets.items()}
    return payload


def sparse_input_from_payload(payload: Mapping[str, Any]) -> SparseDecisionInput:
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
        local_inputs=(
            _tensor_from_payload(payload["local_inputs"])
            if "local_inputs" in payload
            else _tensor_from_payload(payload["local_input"]).unsqueeze(0)
        ),
        local_window_coords=(
            _tensor_from_payload(payload["local_window_coords"])
            if "local_window_coords" in payload
            else torch.zeros((1, 5), dtype=torch.float32)
        ),
        local_window_mask=(
            _tensor_from_payload(payload["local_window_mask"]).bool()
            if "local_window_mask" in payload
            else torch.ones((1,), dtype=torch.bool)
        ),
        rel_edge_index=(
            _tensor_from_payload(payload["rel_edge_index"]).to(dtype=torch.long)
            if "rel_edge_index" in payload
            else torch.zeros((0, 2), dtype=torch.long)
        ),
        rel_edge_features=(
            _tensor_from_payload(payload["rel_edge_features"])
            if "rel_edge_features" in payload
            else torch.zeros((0, 12), dtype=torch.float32)
        ),
        rel_edge_mask=(
            _tensor_from_payload(payload["rel_edge_mask"]).bool()
            if "rel_edge_mask" in payload
            else torch.zeros((0,), dtype=torch.bool)
        ),
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


def collate_compressed_samples(
    records: Sequence[CompressedHexformerSample],
    *,
    architecture: HexformerArchitectureConfig | None = None,
    symmetries: Sequence[object] = (),
) -> dict[str, torch.Tensor]:
    _ = architecture
    sparse_inputs = [sparse_input_from_payload(record.decode().input_payload) for record in records]
    if symmetries:
        sparse_inputs = [
            transform_sparse_input(sample, symmetries[index])
            if index < len(symmetries)
            else sample
            for index, sample in enumerate(sparse_inputs)
        ]
    return collate_sparse_inputs(sparse_inputs)


def _tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.reshape(-1).tolist(),
    }


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


def _hexformer_payload(record: TrainingSampleRecord) -> ModelSamplePayload:
    for payload in record.model_payloads:
        if payload.namespace == SAMPLE_NAMESPACE:
            return payload
    raise ValueError(f"training sample {record.game_id}:{record.turn_index} has no {SAMPLE_NAMESPACE} payload")


def _sample_value(payload: Mapping[str, Any]) -> float | None:
    target = payload.get("wdl_target")
    if not isinstance(target, Mapping):
        return None
    data = tuple(float(item) for item in target.get("data", ()))
    if len(data) != 3:
        return None
    loss, _draw, win = data
    return win - loss


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


def _weighted_without_replacement(
    population_size: int,
    count: int,
    weight_at: object,
    rng: Random,
) -> list[int]:
    keys: list[tuple[float, int]] = []
    for index in range(population_size):
        weight = max(1.0e-12, float(weight_at(index)))
        keys.append((-log(max(1.0e-12, rng.random())) / weight, index))
    keys.sort(key=lambda item: item[0])
    return [index for _key, index in keys[:count]]
