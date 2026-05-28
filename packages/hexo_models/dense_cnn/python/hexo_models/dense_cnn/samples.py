"""Compact Model 1 samples, finalization, and dense expansion.

Self-play stores compact game facts rather than prebuilt tensors. Rust builds
the per-position state facts (stones, legal moves, hot cells, crop center) from a
live engine state; Python attaches search targets, finalizes game-outcome
targets, applies D6 symmetry, and expands facts into tensors when needed.

Finalization is pure arithmetic over the game's decision sequence, so it lives
here in Python rather than crossing the Rust boundary: value targets come from
the winner, opponent-policy targets from the next opponent decision, and
short-term value targets from an exponential moving average of future MCTS root
values (KataGo-style).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch

from . import rust_bridge
from .d6 import Axial, D6_SIZE, D6Symmetry
from .input import build_input_planes, dense_policy_target, legal_mask_flat

D6_TRANSFORMS = tuple(D6Symmetry(index) for index in range(D6_SIZE))
CURRENT_TARGET_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class Model1SampleData:
    """Compact, schema-versioned facts needed to rebuild one training row.

    The dataclass holds no tensors so samples stay cheap to carry through a game,
    transform with D6 symmetry, and expand lazily at write time. Rust authors the
    state-derived facts; Python attaches `policy`/`root_prior_policy` from search
    and `value`/`opp_policy`/`short_term_value` at game end.
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
    short_term_value: tuple[tuple[int, float], ...] = ()
    policy_surprise: float = 0.0
    frequency_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def sample_from_state(
    state: object,
    *,
    game_id: str,
    turn_index: int,
    policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    root_prior_policy: Mapping[int, float] | Sequence[tuple[int, float]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Model1SampleData:
    """Create a compact sample from a live `hexo_engine` state before a decision.

    Rust builds the state-derived facts from the cloned authoritative engine
    state; Python attaches the search policy and root prior.
    """

    if root_prior_policy is None:
        raise ValueError("dense_cnn sample creation requires root_prior_policy")
    facts = rust_bridge.model1_sample_from_state(
        state,
        game_id=game_id,
        turn_index=turn_index,
        metadata={**dict(metadata or {}), "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION},
    )
    return replace(
        _sample_data_from_facts(facts),
        policy=_pairs(policy),
        root_prior_policy=_pairs(root_prior_policy, normalize=True),
    )


def finalize_game_samples(
    pending: Sequence[tuple[str, Model1SampleData, float]],
    winner: str | None,
    horizons: Sequence[int],
    *,
    truncated: bool = False,
) -> list[Model1SampleData]:
    """Assign outcome targets to a finished game's pre-decision samples.

    `pending` is the ordered `(player, sample, root_value)` sequence collected
    during self-play. Value, opponent-policy, and short-term value targets are
    computed here from the sequence and winner.
    """

    decisions = list(pending)
    horizons = tuple(int(horizon) for horizon in horizons)
    finalized: list[Model1SampleData] = []
    for index, (player, sample, _root_value) in enumerate(decisions):
        opp_policy, opp_source = _future_opponent_policy(decisions, index, player)
        metadata = {
            **dict(sample.metadata),
            "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
            "opp_policy_source": opp_source,
            "truncated": bool(truncated),
        }
        if truncated:
            metadata["value_target_reason"] = "max_actions_draw"
        finalized.append(
            replace(
                sample,
                value=_winner_value(winner, player),
                opp_policy=opp_policy,
                short_term_value=_short_term_value_targets(decisions, index, player, horizons),
                metadata=metadata,
            )
        )
    return finalized


def expand_sample(
    sample: Model1SampleData,
    *,
    symmetry: D6Symmetry | int = 0,
) -> dict[str, torch.Tensor]:
    """Decode a compact sample into dense training tensors, applying D6 first."""

    center = Axial(*sample.center)
    tensors: dict[str, torch.Tensor] = {
        "input": build_input_planes(
            current_player=sample.current_player,
            phase=sample.phase,
            center=center,
            stones=sample.stones,
            legal_action_ids=sample.legal_action_ids,
            placement_history=sample.placement_history,
            first_stone=sample.first_stone,
            own_hot=sample.own_hot,
            opponent_hot=sample.opponent_hot,
            opponent_last_turn=sample.opponent_last_turn,
            symmetry=symmetry,
        ),
        "policy": dense_policy_target(sample.policy, center=center, symmetry=symmetry),
        "root_policy": dense_policy_target(sample.root_prior_policy, center=center, symmetry=symmetry, allow_empty=True),
        "opp_policy": dense_policy_target(sample.opp_policy, center=center, symmetry=symmetry, allow_empty=True),
        "legal_mask": legal_mask_flat(sample.legal_action_ids, center=center, symmetry=symmetry),
        "value": torch.tensor(float(sample.value), dtype=torch.float32),
    }
    for horizon, value in sample.short_term_value:
        tensors[f"stvalue_{int(horizon)}"] = torch.tensor(float(value), dtype=torch.float32)
    return tensors


def stack_expanded(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack expanded rows by common tensor keys for model training/inference."""

    if not samples:
        raise ValueError("cannot stack an empty sample sequence")
    keys = set(samples[0])
    for sample in samples[1:]:
        keys &= set(sample)
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in sorted(keys)}


def _winner_value(winner: str | None, player: str) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def _future_opponent_policy(
    decisions: Sequence[tuple[str, Model1SampleData, float]],
    index: int,
    player: str,
) -> tuple[tuple[tuple[int, float], ...], str]:
    """Return the next opponent decision's policy as the opponent-policy target."""

    for future_player, future_sample, _root_value in decisions[index + 1 :]:
        if future_player != player:
            return tuple(future_sample.policy), "future_opponent_mcts"
    return (), "none"


def _short_term_value_targets(
    decisions: Sequence[tuple[str, Model1SampleData, float]],
    index: int,
    player: str,
    horizons: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    """Exponential moving average of future root values, one per horizon.

    Horizon `m` sets the EMA decay `lambda = m / (m + 1)`, whose effective mean
    look-ahead distance is `m` moves. Future root values are taken from the side
    to move at each future decision, so they are sign-flipped to this decision's
    perspective. A horizon is omitted only when no future decision exists.
    """

    future = decisions[index + 1 :]
    if not future:
        return ()
    perspective = [
        root_value if future_player == player else -root_value
        for future_player, _sample, root_value in future
    ]
    targets: list[tuple[int, float]] = []
    for horizon in horizons:
        decay = horizon / (horizon + 1.0)
        weighted_sum = 0.0
        weight_total = 0.0
        weight = 1.0
        for value in perspective:
            weighted_sum += weight * value
            weight_total += weight
            weight *= decay
        targets.append((int(horizon), weighted_sum / weight_total))
    return tuple(targets)


def _sample_data_from_facts(facts: Mapping[str, Any]) -> Model1SampleData:
    """Parse the state-derived facts dict Rust produces into the dataclass."""

    return Model1SampleData(
        game_id=str(facts["game_id"]),
        turn_index=int(facts["turn_index"]),
        current_player=str(facts["current_player"]),
        phase=str(facts["phase"]),
        center=tuple(int(item) for item in facts["center"]),  # type: ignore[arg-type]
        stones=tuple((int(q), int(r), str(player)) for q, r, player in facts["stones"]),
        legal_action_ids=tuple(int(item) for item in facts["legal_action_ids"]),
        placement_history=tuple(
            (int(q), int(r), str(player), str(phase), int(idx), _optional_int(fq), _optional_int(fr))
            for q, r, player, phase, idx, fq, fr in facts.get("placement_history", ())
        ),
        first_stone=(
            tuple(int(item) for item in facts["first_stone"]) if facts.get("first_stone") is not None else None
        ),  # type: ignore[arg-type]
        own_hot=tuple((int(q), int(r)) for q, r in facts.get("own_hot", ())),
        opponent_hot=tuple((int(q), int(r)) for q, r in facts.get("opponent_hot", ())),
        opponent_last_turn=tuple((int(q), int(r)) for q, r in facts.get("opponent_last_turn", ())),
        metadata=dict(facts.get("metadata", {})),
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _pairs(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    normalize: bool = False,
) -> tuple[tuple[int, float], ...]:
    items = weights.items() if isinstance(weights, Mapping) else tuple(weights)
    pairs = tuple((int(action), float(weight)) for action, weight in items)
    if not normalize:
        return pairs
    total = sum(weight for _action, weight in pairs)
    if total <= 0.0:
        raise ValueError("policy weights must contain positive mass")
    return tuple((action, weight / total) for action, weight in pairs)
