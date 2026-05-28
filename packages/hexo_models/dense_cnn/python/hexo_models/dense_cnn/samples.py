"""Compact Model 1 samples and dense expansion.

Self-play stores compact game facts rather than prebuilt tensors. Rust creates
and finalizes those facts from live engine states; Python validates, compresses,
decodes, applies D6 symmetry, and expands them into tensors when needed.

Persistent replay is intentionally not in this module. Dense CNN self-play now
writes KataGo-style NPZ rows through `replay.py`, and checkpoints store only
model/optimizer/train state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence
import json
import zlib

import torch

from . import rust_bridge
from .d6 import Axial, D6_SIZE, D6Symmetry
from .constants import BOARD_SIZE
from .input import build_input_planes, dense_policy_target, legal_mask_flat

D6_TRANSFORMS = tuple(D6Symmetry(index) for index in range(D6_SIZE))
CURRENT_TARGET_SCHEMA_VERSION = 3


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
    root_prior_policy: tuple[tuple[int, float], ...] = ()
    opp_policy: tuple[tuple[int, float], ...] = ()
    value: float = 0.0
    lookahead: tuple[tuple[int, float], ...] = ()
    policy_surprise: float = 0.0
    frequency_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompressedSample:
    """One zlib-compressed JSON sample used for transient pending game data."""

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


def sample_from_state(
    state: object,
    *,
    game_id: str,
    turn_index: int,
    policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    root_prior_policy: Mapping[int, float] | Sequence[tuple[int, float]] | None = None,
    value: float = 0.0,
    opp_policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    lookahead: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Model1SampleData:
    """Create a compact sample from a `hexo_engine` state before a decision.

    Live-state facts are built by the dense-cnn Rust accelerator from a cloned
    authoritative engine state.
    """

    if root_prior_policy is None:
        raise ValueError("dense_cnn sample creation requires root_prior_policy")
    normalized_root_prior = _normalized_pairs(root_prior_policy, allow_empty=False)
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
    sample = _sample_data_from_json(payload)
    return replace(
        sample,
        root_prior_policy=normalized_root_prior,
    )


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
        "root_policy": dense_policy_target(
            data.root_prior_policy,
            center=center,
            symmetry=symmetry,
            allow_empty=True,
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
        root_prior_policy=tuple((int(action), float(weight)) for action, weight in data.get("root_prior_policy", ())),
        opp_policy=tuple((int(action), float(weight)) for action, weight in data.get("opp_policy", ())),
        value=float(data.get("value", 0.0)),
        lookahead=tuple((int(horizon), float(value)) for horizon, value in data.get("lookahead", ())),
        policy_surprise=float(data.get("policy_surprise", 0.0)),
        frequency_weight=float(data.get("frequency_weight", 1.0)),
        metadata=dict(data.get("metadata", {})),
    )


def _sample_payload(sample: Model1SampleData | CompressedSample) -> Mapping[str, Any]:
    if isinstance(sample, CompressedSample):
        return asdict(sample.decode())
    if isinstance(sample, Model1SampleData):
        return asdict(sample)
    raise TypeError(f"expected Model1SampleData or CompressedSample, got {type(sample).__name__}")


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _normalized_pairs(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    allow_empty: bool = False,
) -> tuple[tuple[int, float], ...]:
    items = weights.items() if isinstance(weights, Mapping) else tuple(weights)
    pairs = tuple((int(action), float(weight)) for action, weight in items)
    total = sum(weight for _action, weight in pairs)
    if total <= 0.0:
        if allow_empty:
            return ()
        raise ValueError("policy weights must contain positive mass")
    return tuple((action, weight / total) for action, weight in pairs)
