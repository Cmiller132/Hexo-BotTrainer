"""Compact compressed Model 1 samples and dense training expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import exp, log
from random import Random
from typing import Any, Mapping, Sequence
import json
import zlib

import torch

from . import rust_bridge
from .d6 import Axial, D6_SIZE, D6Symmetry, inverse_index, pack_coord_id, transform_coord
from .constants import BOARD_SIZE
from .input import build_input_planes, dense_policy_target, legal_mask_flat

D6_TRANSFORMS = tuple(D6Symmetry(index) for index in range(D6_SIZE))
CURRENT_TARGET_SCHEMA_VERSION = 2


def _sample_buffer_load_stats(
    *,
    total: int = 0,
    loaded: int = 0,
    filtered_schema: int = 0,
    filtered_decode_errors: int = 0,
    dropped_by_capacity: int = 0,
) -> dict[str, int]:
    filtered = int(filtered_schema) + int(filtered_decode_errors) + int(dropped_by_capacity)
    return {
        "target_schema_version": int(CURRENT_TARGET_SCHEMA_VERSION),
        "total": int(total),
        "loaded": int(loaded),
        "filtered": filtered,
        "filtered_schema": int(filtered_schema),
        "filtered_decode_errors": int(filtered_decode_errors),
        "dropped_by_capacity": int(dropped_by_capacity),
    }


@dataclass(frozen=True, slots=True)
class Model1SampleData:
    game_id: str
    turn_index: int
    current_player: str
    phase: str
    center: tuple[int, int]
    stones: tuple[tuple[int, int, str], ...]
    legal_action_ids: tuple[int, ...]
    placement_history: tuple[tuple[int, int, str, str, int, int | None, int | None], ...] = ()
    first_stone: tuple[int, int] | None = None
    own_hot: tuple[tuple[int, int], ...] = ()
    opponent_hot: tuple[tuple[int, int], ...] = ()
    opponent_last_turn: tuple[tuple[int, int], ...] = ()
    policy: tuple[tuple[int, float], ...] = ()
    opp_policy: tuple[tuple[int, float], ...] = ()
    value: float = 0.0
    lookahead: tuple[tuple[int, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompressedSample:
    """One zlib-compressed JSON sample held in RAM until training decode."""

    payload: bytes
    uncompressed_bytes: int
    compression: str = "zlib+json"
    compressed: bool = True

    @classmethod
    def from_data(cls, data: Model1SampleData, *, compression_level: int = 6) -> "CompressedSample":
        raw = json.dumps(asdict(data), separators=(",", ":"), sort_keys=True).encode("utf-8")
        return cls(
            payload=zlib.compress(raw, level=int(compression_level)),
            uncompressed_bytes=len(raw),
        )

    @property
    def compressed_bytes(self) -> int:
        return len(self.payload)

    def decode(self) -> Model1SampleData:
        raw = zlib.decompress(self.payload).decode("utf-8")
        data = json.loads(raw)
        return _sample_data_from_json(data)


@dataclass(slots=True)
class SampleBuffer:
    """Capacity-bounded in-RAM compressed replay buffer with recency sampling."""

    capacity: int = 200_000
    recency_halflife: float = 50_000.0
    recency_decay: float | None = None
    seed: int | None = None
    compression_level: int = 6
    _samples: list[CompressedSample] = field(default_factory=list)
    _total_appended: int = 0
    _draw_count: int = 0
    _last_load_stats: dict[str, int] = field(default_factory=_sample_buffer_load_stats)

    def __post_init__(self) -> None:
        if self.capacity < 200_000:
            self.capacity = 200_000
        if self.recency_decay is not None:
            decay = float(self.recency_decay)
            if not 0.0 < decay < 1.0:
                raise ValueError("recency_decay must be in (0, 1)")
            self.recency_halflife = max(1.0e-6, log(0.5) / log(decay))
        if self.recency_halflife <= 0:
            raise ValueError("recency_halflife must be positive")

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def total_appended(self) -> int:
        return self._total_appended

    @property
    def compressed_bytes(self) -> int:
        return sum(sample.compressed_bytes for sample in self._samples)

    @property
    def entries(self) -> tuple[CompressedSample, ...]:
        return tuple(self._samples)

    @property
    def compact_samples(self) -> tuple[CompressedSample, ...]:
        return tuple(self._samples)

    @property
    def last_load_stats(self) -> dict[str, int]:
        return dict(self._last_load_stats)

    @property
    def load_stats(self) -> dict[str, int]:
        return self.last_load_stats

    def add(self, sample: Model1SampleData | CompressedSample | Mapping[str, Any]) -> None:
        self.append(sample)

    def append(self, sample: Model1SampleData | CompressedSample | Mapping[str, Any]) -> None:
        if isinstance(sample, Mapping):
            sample = raw_mapping_to_sample_data(sample)
        elif isinstance(sample, Model1SampleData):
            sample = _with_current_target_schema(sample)
        compressed = (
            sample
            if isinstance(sample, CompressedSample)
            else CompressedSample.from_data(sample, compression_level=self.compression_level)
        )
        self._samples.append(compressed)
        self._total_appended += 1
        overflow = len(self._samples) - self.capacity
        if overflow > 0:
            del self._samples[:overflow]

    def extend(self, samples: Sequence[Model1SampleData | CompressedSample]) -> None:
        for sample in samples:
            self.append(sample)

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

    def load_state_dict(self, state: Mapping[str, Any]) -> dict[str, int]:
        self.capacity = max(200_000, int(state.get("capacity", self.capacity)))
        self.recency_halflife = float(state.get("recency_halflife", self.recency_halflife))
        self.compression_level = int(state.get("compression_level", self.compression_level))
        self._total_appended = int(state.get("total_appended", 0))
        self._draw_count = int(state.get("draw_count", 0))
        loaded_samples: list[CompressedSample] = []
        filtered_schema = 0
        filtered_decode_errors = 0
        raw_samples = tuple(state.get("samples") or ())
        for item in raw_samples:
            try:
                compressed = _compressed_sample_from_state_item(item)
                data = compressed.decode()
            except (KeyError, TypeError, ValueError, zlib.error, json.JSONDecodeError, UnicodeDecodeError):
                filtered_decode_errors += 1
                continue
            if _metadata_target_schema_version(data.metadata) != CURRENT_TARGET_SCHEMA_VERSION:
                filtered_schema += 1
                continue
            loaded_samples.append(compressed)

        dropped_by_capacity = max(0, len(loaded_samples) - self.capacity)
        self._samples = loaded_samples[-self.capacity :]
        self._last_load_stats = _sample_buffer_load_stats(
            total=len(raw_samples),
            loaded=len(self._samples),
            filtered_schema=filtered_schema,
            filtered_decode_errors=filtered_decode_errors,
            dropped_by_capacity=dropped_by_capacity,
        )
        return self.last_load_stats

    def sample(self, count: int, *, seed: int | None = None) -> tuple[CompressedSample, ...]:
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

    def _recency_weight(self, index: int) -> float:
        age = len(self._samples) - 1 - index
        return exp(-log(2.0) * age / self.recency_halflife)


def encode_compact_sample(sample: Mapping[str, Any] | Model1SampleData) -> CompressedSample:
    """Compress sample metadata without materializing dense tensors."""

    data = sample if isinstance(sample, Model1SampleData) else raw_mapping_to_sample_data(sample)
    return CompressedSample.from_data(data)


def decode_compact_sample(
    sample: CompressedSample | Model1SampleData | Mapping[str, Any],
    *,
    transform: D6Symmetry | int | None = None,
    symmetry: D6Symmetry | int | None = None,
    d6: D6Symmetry | int | None = None,
) -> dict[str, Any]:
    """Expand one compact sample into dense training targets and metadata."""

    selected = transform
    if selected is None:
        selected = symmetry
    if selected is None:
        selected = d6
    if selected is None:
        selected = 0

    compact = sample
    if isinstance(sample, Mapping):
        compact = encode_compact_sample(sample)
    expanded = expand_sample(compact, symmetry=selected)
    data = compact.decode() if isinstance(compact, CompressedSample) else compact
    lookahead = {
        int(key.removeprefix("lookahead_")): value
        for key, value in expanded.items()
        if key.startswith("lookahead_")
    }
    return {
        **expanded,
        "policy_target": expanded["policy"],
        "dense_policy": expanded["policy"],
        "opp_policy_target": expanded["opp_policy"],
        "dense_opp_policy": expanded["opp_policy"],
        "value_target": expanded["value"],
        "lookahead_targets": lookahead,
        "lookahead": lookahead,
        "sample_id": data.metadata.get("sample_id", data.game_id),
        "metadata": dict(data.metadata),
    }


def d6_transforms() -> tuple[D6Symmetry, ...]:
    return D6_TRANSFORMS


def transform_axial(coord: tuple[int, int], transform: D6Symmetry | int) -> Axial:
    return transform_coord(coord, transform)


def inverse_d6(transform: D6Symmetry | int) -> D6Symmetry:
    index = transform.index if isinstance(transform, D6Symmetry) else int(transform)
    return D6Symmetry(inverse_index(index))


def dense_index_for_coord(
    coord: tuple[int, int],
    *,
    center: tuple[int, int] = (0, 0),
    board_size: int = BOARD_SIZE,
    size: int | None = None,
    crop_center: tuple[int, int] | None = None,
    crop_size: int | None = None,
) -> int:
    resolved_size = crop_size or size or board_size
    resolved_center = crop_center or center
    q, r = int(coord[0]), int(coord[1])
    center_q, center_r = int(resolved_center[0]), int(resolved_center[1])
    half = int(resolved_size) // 2
    col = q - center_q + half
    row = r - center_r + half
    if not 0 <= row < resolved_size or not 0 <= col < resolved_size:
        raise ValueError(f"coordinate {coord!r} outside {resolved_size} crop centered on {resolved_center!r}")
    return row * resolved_size + col


def raw_mapping_to_sample_data(sample: Mapping[str, Any]) -> Model1SampleData:
    """Normalize test/debug raw sample maps into the production compact schema."""

    sample_id = str(sample.get("sample_id", sample.get("game_id", "sample")))
    sequence = int(sample.get("sequence", sample.get("turn_index", 0)))
    center = tuple(int(item) for item in sample.get("center", (0, 0)))
    policy = _raw_policy_to_action_pairs(sample.get("policy", ()))
    opp_policy = _raw_policy_to_action_pairs(sample.get("opp_policy", ()))
    legal_ids = tuple(dict(policy + opp_policy).keys())
    return Model1SampleData(
        game_id=sample_id,
        turn_index=sequence,
        current_player=str(sample.get("current_player", "player0")),
        phase=str(sample.get("phase", "FirstStone")),
        center=(center[0], center[1]),
        stones=tuple((int(q), int(r), str(player)) for q, r, player in sample.get("stones", ())),
        legal_action_ids=tuple(int(item) for item in sample.get("legal_action_ids", legal_ids)),
        policy=policy,
        opp_policy=opp_policy,
        value=float(sample.get("value", 0.0)),
        lookahead=tuple(
            (int(horizon), float(value))
            for horizon, value in (
                sample.get("lookahead", {}).items()
                if isinstance(sample.get("lookahead", {}), Mapping)
                else sample.get("lookahead", ())
            )
        ),
        metadata={
            "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
            **dict(sample.get("metadata", {})),
            "sample_id": sample_id,
            "sequence": sequence,
        },
    )


def sample_from_state(
    state: object,
    *,
    game_id: str,
    turn_index: int,
    policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    value: float = 0.0,
    opp_policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    lookahead: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Model1SampleData:
    """Create a compact sample from a `hexo_engine` state before a decision.

    Live-state facts are built by the dense-cnn Rust accelerator from a cloned
    authoritative engine state.
    """

    resolved_metadata = {
        **dict(metadata or {}),
        "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
    }
    payload = rust_bridge.model1_sample_from_state(
        state,
        game_id=game_id,
        turn_index=turn_index,
        policy=policy,
        value=value,
        opp_policy=opp_policy,
        lookahead=lookahead,
        metadata=resolved_metadata,
    )
    return _sample_data_from_json(payload)


def finalize_game_samples(
    pending: Sequence[tuple[str, Model1SampleData | CompressedSample | Mapping[str, Any], float]],
    winner: str | None,
    horizons: Sequence[int],
    *,
    truncated: bool = False,
) -> list[Model1SampleData]:
    """Finalize self-play samples through Rust-owned outcome logic."""

    rust_pending = tuple(
        (str(player), _sample_payload(sample), float(root_value))
        for player, sample, root_value in pending
    )
    payloads = rust_bridge.model1_finalize_game_samples(
        rust_pending,
        winner=winner,
        horizons=horizons,
        truncated=truncated,
    )
    return [_sample_data_from_json(payload) for payload in payloads]


def expand_sample(
    sample: Model1SampleData | CompressedSample,
    *,
    symmetry: D6Symmetry | int = 0,
) -> dict[str, torch.Tensor]:
    """Decode a compact sample into dense training tensors, applying D6 first."""

    data = sample.decode() if isinstance(sample, CompressedSample) else sample
    center = Axial(*data.center)
    tensors: dict[str, torch.Tensor] = {
        "input": build_input_planes(
            current_player=data.current_player,
            phase=data.phase,
            center=center,
            stones=data.stones,
            legal_action_ids=data.legal_action_ids,
            placement_history=data.placement_history,
            first_stone=data.first_stone,
            own_hot=data.own_hot,
            opponent_hot=data.opponent_hot,
            opponent_last_turn=data.opponent_last_turn,
            symmetry=symmetry,
        ),
        "policy": dense_policy_target(
            data.policy,
            center=center,
            symmetry=symmetry,
        ),
        "opp_policy": dense_policy_target(
            data.opp_policy,
            center=center,
            symmetry=symmetry,
        ),
        "legal_mask": legal_mask_flat(data.legal_action_ids, center=center, symmetry=symmetry),
        "value": torch.tensor(float(data.value), dtype=torch.float32),
    }
    for horizon, value in data.lookahead:
        tensors[f"lookahead_{int(horizon)}"] = torch.tensor(float(value), dtype=torch.float32)
    return tensors


def stack_expanded(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot stack an empty sample sequence")
    keys = set(samples[0])
    for sample in samples[1:]:
        keys &= set(sample)
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in sorted(keys)}


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


def _sample_data_from_json(data: Mapping[str, Any]) -> Model1SampleData:
    return Model1SampleData(
        game_id=str(data["game_id"]),
        turn_index=int(data["turn_index"]),
        current_player=str(data["current_player"]),
        phase=str(data["phase"]),
        center=tuple(int(item) for item in data["center"]),  # type: ignore[arg-type]
        stones=tuple((int(q), int(r), str(player)) for q, r, player in data["stones"]),
        legal_action_ids=tuple(int(item) for item in data["legal_action_ids"]),
        placement_history=tuple(
            (int(q), int(r), str(player), str(phase), int(index), _optional_int(first_q), _optional_int(first_r))
            for q, r, player, phase, index, first_q, first_r in data.get("placement_history", ())
        ),
        first_stone=(
            tuple(int(item) for item in data["first_stone"]) if data.get("first_stone") is not None else None
        ),  # type: ignore[arg-type]
        own_hot=tuple((int(q), int(r)) for q, r in data.get("own_hot", ())),
        opponent_hot=tuple((int(q), int(r)) for q, r in data.get("opponent_hot", ())),
        opponent_last_turn=tuple((int(q), int(r)) for q, r in data.get("opponent_last_turn", ())),
        policy=tuple((int(action), float(weight)) for action, weight in data.get("policy", ())),
        opp_policy=tuple((int(action), float(weight)) for action, weight in data.get("opp_policy", ())),
        value=float(data.get("value", 0.0)),
        lookahead=tuple((int(horizon), float(value)) for horizon, value in data.get("lookahead", ())),
        metadata=dict(data.get("metadata", {})),
    )


def _with_current_target_schema(sample: Model1SampleData) -> Model1SampleData:
    if _metadata_target_schema_version(sample.metadata) == CURRENT_TARGET_SCHEMA_VERSION:
        return sample
    return replace(
        sample,
        metadata={
            **dict(sample.metadata),
            "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
        },
    )


def _metadata_target_schema_version(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("target_schema_version")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compressed_sample_from_state_item(item: object) -> CompressedSample:
    if isinstance(item, CompressedSample):
        return item
    if isinstance(item, Mapping):
        if "payload" in item:
            return CompressedSample(
                payload=bytes(item["payload"]),
                uncompressed_bytes=int(item["uncompressed_bytes"]),
                compression=str(item.get("compression", "zlib+json")),
                compressed=bool(item.get("compressed", True)),
            )
        return CompressedSample.from_data(_sample_data_from_json(item))
    raise TypeError(f"expected compressed sample state mapping, got {type(item).__name__}")


def _sample_payload(sample: Model1SampleData | CompressedSample | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(sample, CompressedSample):
        return asdict(sample.decode())
    if isinstance(sample, Model1SampleData):
        return asdict(sample)
    if isinstance(sample, Mapping):
        return dict(sample)
    raise TypeError(f"expected Model1SampleData or mapping, got {type(sample).__name__}")


def _raw_policy_to_action_pairs(value: object) -> tuple[tuple[int, float], ...]:
    if isinstance(value, Mapping):
        items = value.items()
    else:
        items = value if value is not None else ()
    pairs: list[tuple[int, float]] = []
    for action_or_coord, weight in items:  # type: ignore[assignment]
        if isinstance(action_or_coord, int):
            action_id = int(action_or_coord)
        else:
            coord = tuple(action_or_coord)  # type: ignore[arg-type]
            action_id = pack_coord_id(Axial(int(coord[0]), int(coord[1])))
        pairs.append((action_id, float(weight)))
    return tuple(pairs)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
