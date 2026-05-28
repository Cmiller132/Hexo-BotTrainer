"""Compact Model 1 samples, replay buffer, and dense expansion.

Self-play stores compact game facts rather than prebuilt tensors. Rust creates
and finalizes those facts from live engine states; Python validates, compresses,
stores, samples, decodes, applies D6 symmetry, and expands them into tensors for
training.

The checkpoint schema stores only compressed current-schema samples. Loading an
older raw mapping format or an incompatible target schema raises immediately so
training cannot continue from mixed target semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import exp, log
from random import Random
from typing import Any, Mapping, Sequence
import json
import zlib

import torch

from . import rust_bridge
from .d6 import Axial, D6_SIZE, D6Symmetry
from .constants import BOARD_SIZE
from .input import build_input_planes, dense_policy_target, legal_mask_flat

D6_TRANSFORMS = tuple(D6Symmetry(index) for index in range(D6_SIZE))
CURRENT_TARGET_SCHEMA_VERSION = 2


def _sample_buffer_load_stats(
    *,
    total: int = 0,
    loaded: int = 0,
) -> dict[str, int]:
    return {
        "target_schema_version": int(CURRENT_TARGET_SCHEMA_VERSION),
        "total": int(total),
        "loaded": int(loaded),
    }


@dataclass(frozen=True, slots=True)
class Model1SampleData:
    """Compact, schema-versioned facts needed to rebuild one training row.

    The dataclass is intentionally free of tensors so samples can be compressed,
    checkpointed, transformed with D6 symmetry, and expanded lazily at training
    time. Rust emits the authoritative facts from live engine states; Python
    owns compression and tensor expansion.
    """

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
        """Serialize a current-schema sample with deterministic JSON layout."""

        raw = json.dumps(asdict(data), separators=(",", ":"), sort_keys=True).encode("utf-8")
        return cls(
            payload=zlib.compress(raw, level=int(compression_level)),
            uncompressed_bytes=len(raw),
        )

    @property
    def compressed_bytes(self) -> int:
        return len(self.payload)

    def decode(self) -> Model1SampleData:
        """Inflate and parse the compressed JSON payload into `Model1SampleData`."""

        raw = zlib.decompress(self.payload).decode("utf-8")
        data = json.loads(raw)
        return _sample_data_from_json(data)


@dataclass(slots=True)
class SampleBuffer:
    """Capacity-bounded in-RAM compressed replay buffer with recency sampling.

    The buffer stores compressed samples only. That keeps long self-play runs
    memory-bounded and ensures checkpointed replay entries all pass through the
    current `CompressedSample`/target-schema validation path.
    """

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
            raise ValueError("SampleBuffer capacity must be >= 200000")
        if self.recency_decay is not None:
            decay = float(self.recency_decay)
            if not 0.0 < decay < 1.0:
                raise ValueError("recency_decay must be in (0, 1)")
            self.recency_halflife = log(0.5) / log(decay)
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

    def add(self, sample: Model1SampleData | CompressedSample) -> None:
        self.append(sample)

    def append(self, sample: Model1SampleData | CompressedSample) -> None:
        """Add one current-schema sample and evict oldest overflow entries."""

        if isinstance(sample, Model1SampleData):
            sample = _with_current_target_schema(sample)
        elif not isinstance(sample, CompressedSample):
            raise TypeError(f"expected Model1SampleData or CompressedSample, got {type(sample).__name__}")
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
        """Return the checkpoint payload for compressed replay state."""

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
        """Load a replay-buffer checkpoint after strict schema validation."""

        capacity = int(state.get("capacity", self.capacity))
        if capacity < 200_000:
            raise ValueError("sample buffer checkpoint capacity must be >= 200000")
        self.capacity = capacity
        self.recency_halflife = float(state.get("recency_halflife", self.recency_halflife))
        self.compression_level = int(state.get("compression_level", self.compression_level))
        self._total_appended = int(state.get("total_appended", 0))
        self._draw_count = int(state.get("draw_count", 0))
        raw_samples = tuple(state.get("samples") or ())
        loaded_samples: list[CompressedSample] = []
        for index, item in enumerate(raw_samples):
            try:
                compressed = _compressed_sample_from_state_item(item)
                data = compressed.decode()
            except (KeyError, TypeError, ValueError, zlib.error, json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError(f"sample buffer checkpoint sample {index} is not a valid compressed sample") from None
            if _metadata_target_schema_version(data.metadata) != CURRENT_TARGET_SCHEMA_VERSION:
                raise ValueError(
                    f"sample buffer checkpoint sample {index} has incompatible target_schema_version"
                )
            loaded_samples.append(compressed)

        if len(loaded_samples) > self.capacity:
            raise ValueError(
                f"sample buffer checkpoint contains {len(loaded_samples)} samples, above capacity {self.capacity}"
            )
        self._samples = loaded_samples
        self._last_load_stats = _sample_buffer_load_stats(
            total=len(raw_samples),
            loaded=len(self._samples),
        )
        return self.last_load_stats

    def sample(self, count: int, *, seed: int | None = None) -> tuple[CompressedSample, ...]:
        """Draw a recency-weighted sample window without replacement."""

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
    pending: Sequence[tuple[str, Model1SampleData | CompressedSample, float]],
    winner: str | None,
    horizons: Sequence[int],
    *,
    truncated: bool = False,
) -> list[Model1SampleData]:
    """Finalize self-play samples through Rust-owned outcome logic.

    `pending` contains samples collected before MCTS decisions. Rust receives
    the whole game sequence so it can assign final values, future opponent
    policy targets, and lookahead values consistently.
    """

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
            allow_empty=True,
        ),
        "legal_mask": legal_mask_flat(data.legal_action_ids, center=center, symmetry=symmetry),
        "value": torch.tensor(float(data.value), dtype=torch.float32),
    }
    for horizon, value in data.lookahead:
        tensors[f"lookahead_{int(horizon)}"] = torch.tensor(float(value), dtype=torch.float32)
    return tensors


def stack_expanded(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack expanded rows by common tensor keys for model training/inference."""

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
    """Efraimidis-Spirakis weighted sampling without replacement."""

    keys: list[tuple[float, int]] = []
    for index in range(population_size):
        weight = float(weight_at(index))
        if not weight > 0.0:
            raise ValueError(f"sample weight at index {index} must be > 0")
        draw = rng.random()
        while draw <= 0.0:
            draw = rng.random()
        keys.append((-log(draw) / weight, index))
    keys.sort(key=lambda item: item[0])
    return [index for _key, index in keys[:count]]


def _sample_data_from_json(data: Mapping[str, Any]) -> Model1SampleData:
    """Parse Rust/Python JSON-compatible sample payloads into the dataclass."""

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
        if "payload" not in item:
            raise ValueError("sample buffer checkpoints must store compressed sample payloads")
        return CompressedSample(
            payload=bytes(item["payload"]),
            uncompressed_bytes=int(item["uncompressed_bytes"]),
            compression=str(item.get("compression", "zlib+json")),
            compressed=bool(item.get("compressed", True)),
        )
    raise TypeError(f"expected compressed sample state mapping, got {type(item).__name__}")


def _sample_payload(sample: Model1SampleData | CompressedSample) -> Mapping[str, Any]:
    if isinstance(sample, CompressedSample):
        return asdict(sample.decode())
    if isinstance(sample, Model1SampleData):
        return asdict(sample)
    raise TypeError(f"expected Model1SampleData or CompressedSample, got {type(sample).__name__}")


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
