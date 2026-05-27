"""Compact compressed Model 1 samples and dense training expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, log
from random import Random
from typing import Any, Mapping, Sequence
import json
import zlib

import torch

from .d6 import Axial, D6_SIZE, D6Symmetry, inverse_index, pack_coord_id, transform_coord, unpack_coord_id
from .constants import BOARD_SIZE
from .geometry import coord_to_flat, crop_center
from .input import build_input_planes, dense_policy_target, legal_mask_flat

D6_TRANSFORMS = tuple(D6Symmetry(index) for index in range(D6_SIZE))


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

    def add(self, sample: Model1SampleData | CompressedSample | Mapping[str, Any]) -> None:
        self.append(sample)

    def append(self, sample: Model1SampleData | CompressedSample | Mapping[str, Any]) -> None:
        if isinstance(sample, Mapping):
            sample = raw_mapping_to_sample_data(sample)
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

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacity = max(200_000, int(state.get("capacity", self.capacity)))
        self.recency_halflife = float(state.get("recency_halflife", self.recency_halflife))
        self.compression_level = int(state.get("compression_level", self.compression_level))
        self._total_appended = int(state.get("total_appended", 0))
        self._draw_count = int(state.get("draw_count", 0))
        self._samples = [
            CompressedSample(
                payload=bytes(item["payload"]),
                uncompressed_bytes=int(item["uncompressed_bytes"]),
                compression=str(item.get("compression", "zlib+json")),
                compressed=bool(item.get("compressed", True)),
            )
            for item in state.get("samples", ())
        ][-self.capacity :]

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
        metadata={**dict(sample.get("metadata", {})), "sample_id": sample_id, "sequence": sequence},
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
    """Create a compact sample from a `hexo_engine` state before a decision."""

    import hexo_engine as engine

    python_state = engine.to_python_state(state)
    current_player = _player_label(python_state.current_player)
    opponent = "player1" if current_player == "player0" else "player0"
    stones = tuple(
        (int(coord.q), int(coord.r), _player_label(player))
        for coord, player in python_state.board.stones
    )
    center = crop_center((Axial(q, r) for q, r, _player in stones))
    first_stone = (
        (python_state.first_stone.q, python_state.first_stone.r)
        if python_state.first_stone is not None
        else None
    )
    history = tuple(
        (
            int(record.coord.q),
            int(record.coord.r),
            _player_label(record.player),
            _phase_label(record.phase),
            int(record.placement_index),
            int(record.first_stone.q) if record.first_stone is not None else None,
            int(record.first_stone.r) if record.first_stone is not None else None,
        )
        for record in python_state.placement_history
    )
    own_hot, opponent_hot = _hot_cells(python_state, current_player)
    last_opponent_turn = _last_completed_turn(history, opponent)
    legal_action_ids = tuple(
        int(action_id)
        for action_id in engine.legal_action_ids(state)
        if coord_to_flat(unpack_coord_id(int(action_id)), center=center) is not None
    )

    return Model1SampleData(
        game_id=game_id,
        turn_index=int(turn_index),
        current_player=current_player,
        phase=_phase_label(python_state.phase),
        center=(center.q, center.r),
        stones=stones,
        legal_action_ids=legal_action_ids,
        placement_history=history,
        first_stone=first_stone,
        own_hot=own_hot,
        opponent_hot=opponent_hot,
        opponent_last_turn=last_opponent_turn,
        policy=_weights_to_pairs(policy),
        opp_policy=_weights_to_pairs(opp_policy),
        value=float(value),
        lookahead=tuple((int(k), float(v)) for k, v in (lookahead.items() if isinstance(lookahead, Mapping) else lookahead)),
        metadata=dict(metadata or {}),
    )


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
            fallback_legal_action_ids=data.legal_action_ids,
        ),
        "opp_policy": dense_policy_target(
            data.opp_policy,
            center=center,
            symmetry=symmetry,
            fallback_legal_action_ids=data.legal_action_ids,
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


def _hot_cells(python_state: object, current_player: str) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    own: set[tuple[int, int]] = set()
    opponent: set[tuple[int, int]] = set()
    occupied = {(coord.q, coord.r) for coord in python_state.board.occupied}
    for entry in python_state.board.windows.entries:
        masks = tuple(int(mask) for mask in entry.masks)
        p0_count = masks[0].bit_count()
        p1_count = masks[1].bit_count()
        if p0_count and p1_count:
            continue
        if p0_count < 4 and p1_count < 4:
            continue
        player = "player0" if p0_count >= 4 else "player1"
        target = own if player == current_player else opponent
        key = getattr(entry, "key", entry)
        for coord in _window_cells(key.start, key.axis):
            if coord not in occupied:
                target.add(coord)
    return tuple(sorted(own)), tuple(sorted(opponent))


def _window_cells(start: object, axis: str) -> tuple[tuple[int, int], ...]:
    vector = {
        "Q": (1, 0),
        "R": (0, 1),
        "QR": (1, -1),
    }[str(axis)]
    return tuple((int(start.q) + vector[0] * index, int(start.r) + vector[1] * index) for index in range(6))


def _last_completed_turn(
    history: Sequence[tuple[int, int, str, str, int, int | None, int | None]],
    player: str,
) -> tuple[tuple[int, int], ...]:
    for q, r, record_player, phase, _index, first_q, first_r in reversed(tuple(history)):
        if record_player != player:
            continue
        if phase == "SecondStone" and first_q is not None and first_r is not None:
            return ((int(first_q), int(first_r)), (int(q), int(r)))
        if phase == "Opening":
            return ((int(q), int(r)),)
    return ()


def _weights_to_pairs(weights: Mapping[int, float] | Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    items = weights.items() if isinstance(weights, Mapping) else weights
    return tuple((int(action_id), float(weight)) for action_id, weight in items)


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


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))


def _phase_label(value: object) -> str:
    return str(getattr(value, "value", value))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
