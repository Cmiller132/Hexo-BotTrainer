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

import struct
from array import array
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch

from . import rust_bridge
from .constants import BOARD_SIZE, MOVES_LEFT_CAP
from .d6 import Axial, D6_SIZE, D6Symmetry, unpack_coord_pair
from .geometry import hex_distance
from .input import build_input_planes, dense_policy_target, legal_mask_flat
from .mcts import CompactVisitPolicy

D6_TRANSFORMS = tuple(D6Symmetry(index) for index in range(D6_SIZE))
CURRENT_TARGET_SCHEMA_VERSION = 4
_HALF = BOARD_SIZE // 2

# Categories reported by ``count_spill`` (facts unrepresentable in the radius-20
# hex disk). Observation-only telemetry; see Spec A spill telemetry.
SPILL_CATEGORIES = ("stones", "legal", "policy", "history")

# Stone/placement owners are one of the two engine players; store a 1-byte index
# instead of repeating the string per stone.
_PLAYERS = ("player0", "player1")
_PLAYER_INDEX = {"player0": 0, "player1": 1}


@dataclass(frozen=True, slots=True)
class _PackedStones:
    """Columnar board stones (q,r as int16, owner as 1 byte) that iterates to the
    same ``(q, r, player)`` tuples ``build_input_planes`` expects — ~30x smaller
    than a tuple of Python tuples for a full late-game board, decoded lazily."""

    coords: "array"  # int16, interleaved q, r
    owners: bytes  # one index into _PLAYERS per stone

    def __len__(self) -> int:
        return len(self.owners)

    def __getitem__(self, index: int) -> tuple[int, int, str]:
        return (int(self.coords[2 * index]), int(self.coords[2 * index + 1]), _PLAYERS[self.owners[index]])

    def __iter__(self):
        for index in range(len(self.owners)):
            yield self[index]


@dataclass(frozen=True, slots=True)
class _PackedHistory:
    """Columnar placement history. Stores only the four fields the encoder uses
    (q, r, owner, placement_index); the unused phase/first_q/first_r slots are
    yielded as None so ``build_input_planes`` unpacks the same 7-tuple shape."""

    coords: "array"  # int16, interleaved q, r
    owners: bytes
    indices: "array"  # int32 placement_index

    def __len__(self) -> int:
        return len(self.owners)

    def __getitem__(self, index: int) -> tuple[int, int, str, None, int, None, None]:
        return (
            int(self.coords[2 * index]),
            int(self.coords[2 * index + 1]),
            _PLAYERS[self.owners[index]],
            None,
            int(self.indices[index]),
            None,
            None,
        )

    def __iter__(self):
        for index in range(len(self.owners)):
            yield self[index]


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
    # Columnar-packed for memory (see _PackedStones); a plain tuple of (q,r,player)
    # also works (tests construct one), since both iterate to the same tuples.
    stones: Sequence[tuple[int, int, str]]
    # Stored compactly to keep pending self-play samples small: ~1471 legal moves
    # per late-game position as a tuple of Python ints would be ~57 KB; an
    # ``array("q")`` is ~8x smaller and still iterates to Python ints, which is all
    # the expansion consumers need.
    legal_action_ids: Sequence[int]
    # Columnar-packed (see _PackedHistory); yields the same 7-tuple shape the
    # encoder unpacks, with unused phase/first_q/first_r as None.
    placement_history: Sequence[tuple[int, int, str, str | None, int, int | None, int | None]] = ()
    first_stone: tuple[int, int] | None = None
    own_hot: tuple[tuple[int, int], ...] = ()
    opponent_hot: tuple[tuple[int, int], ...] = ()
    opponent_last_turn: tuple[tuple[int, int], ...] = ()
    # ``policy`` and ``root_prior_policy`` are byte-backed (CompactVisitPolicy):
    # ~8 bytes/entry vs ~117 bytes for a Python (int,float) tuple. root_prior_policy
    # carries every in-crop legal prior (hundreds-to-~1500 entries), so this is the
    # single biggest pending-memory saving. Both are still Sequence[(int,float)], so
    # dense_policy_target / _policy_kl consume them unchanged.
    policy: Sequence[tuple[int, float]] = ()
    root_prior_policy: Sequence[tuple[int, float]] = ()
    opp_policy: tuple[tuple[int, float], ...] = ()
    value: float = 0.0
    short_term_value: tuple[tuple[int, float], ...] = ()
    # Decisions remaining AFTER this decision (0 at the game's final decision).
    # Negative = absent/masked (pre-finalize samples, truncated games, old shards).
    moves_left: float = -1.0
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
    root_prior_compact = _compact_policy(root_prior_policy, normalize=True)
    policy_compact = _compact_policy(policy)
    if len(policy_compact) == 0:
        policy_compact = root_prior_compact
    return replace(
        _sample_data_from_facts(facts),
        policy=policy_compact,
        root_prior_policy=root_prior_compact,
    )


def finalize_game_samples(
    pending: Sequence[tuple[str, Model1SampleData, float]],
    winner: str | None,
    horizons: Sequence[int],
    *,
    truncated: bool = False,
    soft_z_lambda: float = 0.0,
    mask_opp_from_fast: bool = False,
) -> list[Model1SampleData]:
    """Assign outcome targets to a finished game's pre-decision samples.

    `pending` is the ordered `(player, sample, root_value)` sequence collected
    during self-play. Value, opponent-policy, and short-term value targets are
    computed here from the sequence and winner.

    `soft_z_lambda` mixes the hard game outcome with the MCTS root value at the
    position itself (the "soft-Z" target of Willemsen/Baier/Kaisers 2022):

        value = (1 - lambda) * z + lambda * root_value

    `z = _winner_value(winner, player)` is the hard +1/-1/0 outcome and
    `root_value` is `search.root_value` captured at the decision, which is
    already in the side-to-move's (i.e. `player`'s) perspective -- the same
    convention the short-term-value heads consume -- so it needs no sign flip.
    A convex blend of two values in [-1, 1] stays in [-1, 1], so it flows
    through the unchanged 65-bin `scalar_to_binned_target` mapping with no
    head/schema change. `lambda = 0` (default) reproduces the pure hard-outcome
    target; `lambda = 1` is the paper's pure soft-Z. For truncated/draw rows
    (`winner is None`, `z = 0`) the blended target carries `lambda * root_value`,
    a free recalibration of otherwise-zeroed draw labels.
    """

    decisions = list(pending)
    horizons = tuple(int(horizon) for horizon in horizons)
    lam = float(soft_z_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"soft_z_lambda must be in [0, 1], got {soft_z_lambda!r}")
    finalized: list[Model1SampleData] = []
    for index, (player, sample, root_value) in enumerate(decisions):
        opp_policy, opp_source = _future_opponent_policy(
            decisions, index, player, mask_from_fast=mask_opp_from_fast
        )
        metadata = {
            **dict(sample.metadata),
            "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
            "opp_policy_source": opp_source,
            "truncated": bool(truncated),
        }
        if truncated:
            metadata["value_target_reason"] = "max_actions_draw"
        hard_z = _winner_value(winner, player)
        if lam > 0.0:
            value_target = (1.0 - lam) * hard_z + lam * float(root_value)
        else:
            value_target = hard_z
        finalized.append(
            replace(
                sample,
                value=value_target,
                opp_policy=opp_policy,
                short_term_value=_short_term_value_targets(decisions, index, player, horizons),
                # Moves-left auxiliary target: decisions remaining after this one
                # (0 at the final decision). Masked (-1) for truncated games — a
                # max_actions cutoff is not a real game end.
                moves_left=float(len(decisions) - index - 1) if not truncated else -1.0,
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
    if float(sample.moves_left) >= 0.0:
        # Normalize [0, MOVES_LEFT_CAP] (clamped) onto the binned value support
        # [-1, 1] so the moves-left head reuses scalar_to_binned_target unchanged.
        normalized = 2.0 * min(1.0, float(sample.moves_left) / MOVES_LEFT_CAP) - 1.0
        tensors["moves_left"] = torch.tensor(normalized, dtype=torch.float32)
    return tensors


def count_spill(sample: Model1SampleData) -> dict[str, int]:
    """Count facts that fall outside the radius-20 hex disk (spill), per category.

    A fact at hex-distance > ``BOARD_SIZE // 2`` from the crop center is
    unrepresentable in the dense view and is silently dropped by the encoder under
    the disk contract. This is observation-only telemetry (Spec A): it reports how
    much signal the fixed crop cannot carry, per ``SPILL_CATEGORIES`` (stones,
    legal, policy, history). Hex distance is D6-invariant, so the count is the same
    under any augmentation symmetry; it is computed once at identity.

    Note: ``legal`` and ``policy`` are disk-restricted at the source (the native
    ``sample_gen`` / MCTS only emit in-disk legal actions and visit policy), so in
    self-play their spill is structurally ~0; the meaningful signal is ``stones``
    and ``history`` (the raw board facts the crop cannot represent).
    """

    center = (int(sample.center[0]), int(sample.center[1]))

    def beyond(q: int, r: int) -> bool:
        return hex_distance((int(q), int(r)), center) > _HALF

    stones = sum(1 for q, r, _player in sample.stones if beyond(q, r))
    legal = sum(1 for action_id in sample.legal_action_ids if beyond(*unpack_coord_pair(int(action_id))))
    policy = sum(1 for action_id, _weight in sample.policy if beyond(*unpack_coord_pair(int(action_id))))
    history = sum(1 for item in sample.placement_history if beyond(int(item[0]), int(item[1])))
    return {"stones": stones, "legal": legal, "policy": policy, "history": history}


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
    *,
    mask_from_fast: bool = False,
) -> tuple[tuple[tuple[int, float], ...], str]:
    """Return the next opponent decision's policy as the opponent-policy target.

    Under KataGo Playout Cap Randomization (`mask_from_fast=True`), the opponent
    target is MASKED (empty -> dropped from the segmented opp loss denominator)
    when the next opponent move was a FAST search (`pcr_full=False`): a fast move's
    170-visit, unpruned, noise-free distribution is a systematically different
    (sharper) target family, so we only train the opponent head on full->full
    transitions (owner decision; see HEXGT_PCR_KATAGO_MAPPING.md -- removed from
    docs/ in the documentation wipe, recoverable from git history). Default off
    keeps the dense_cnn (non-PCR) behavior byte-identical.
    """

    for future_player, future_sample, _root_value in decisions[index + 1 :]:
        if future_player != player:
            if mask_from_fast and not future_sample.metadata.get("pcr_full", True):
                return (), "fast_unrecorded_masked"
            return tuple(future_sample.policy), "future_opponent_mcts"
    return (), "none"


def _short_term_value_targets(
    decisions: Sequence[tuple[str, Model1SampleData, float]],
    index: int,
    player: str,
    horizons: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    """Per-horizon EMA of future root values, stepped over FULL TURNS.

    Every turn after the opening is TWO decisions by the same player
    (FirstStone -> SecondStone), so consecutive future decisions alternate
    within-turn phase and the side-to-move root value swings with that tempo
    (a mover about to place their second stone is systematically up). A
    per-decision EMA aliases that intra-turn swing into the target. Sampling
    only at even decision offsets (+2, +4, ...) keeps every sampled value at
    the SAME within-turn phase as this decision, removing the aliasing; the
    sign still flips by actual mover label (even offsets alternate players).

    The per-step decay `lambda = (m - 1) / (m + 1)` gives horizon `m` the same
    effective mean look-ahead (`m + 1` decisions) as the old per-decision
    scheme's `lambda = m / (m + 1)`, so horizon semantics are unchanged.
    Horizons must therefore be even, turn-aligned decision counts (enforced at
    the config boundary). A horizon is omitted (masked downstream) when no
    even-offset future decision exists.
    """

    future = decisions[index + 1 :]
    perspective = [
        root_value if future_player == player else -root_value
        for future_player, _sample, root_value in future
    ]
    # future[k] sits at decision offset k+1; even offsets are k = 1, 3, 5, ...
    stepped = perspective[1::2]
    if not stepped:
        return ()
    targets: list[tuple[int, float]] = []
    for horizon in horizons:
        decay = (horizon - 1.0) / (horizon + 1.0)
        weighted_sum = 0.0
        weight_total = 0.0
        weight = 1.0
        for value in stepped:
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
        stones=_pack_stones(facts["stones"]),
        # array("q") (signed 64-bit) holds the u32 action ids, iterates to Python
        # ints for the expansion consumers, and is ~8x smaller than a tuple of ints.
        legal_action_ids=array("q", (int(item) for item in facts["legal_action_ids"])),
        placement_history=_pack_history(facts.get("placement_history", ())),
        first_stone=(
            tuple(int(item) for item in facts["first_stone"]) if facts.get("first_stone") is not None else None
        ),  # type: ignore[arg-type]
        own_hot=tuple((int(q), int(r)) for q, r in facts.get("own_hot", ())),
        opponent_hot=tuple((int(q), int(r)) for q, r in facts.get("opponent_hot", ())),
        opponent_last_turn=tuple((int(q), int(r)) for q, r in facts.get("opponent_last_turn", ())),
        metadata=dict(facts.get("metadata", {})),
    )


def _pack_stones(stones_facts: Sequence[tuple[int, int, str]]) -> _PackedStones:
    coords = array("h")
    owners = bytearray()
    for q, r, player in stones_facts:
        coords.append(int(q))
        coords.append(int(r))
        owners.append(_PLAYER_INDEX[str(player)])
    return _PackedStones(coords=coords, owners=bytes(owners))


def _pack_history(history_facts: Sequence[tuple]) -> _PackedHistory:
    coords = array("h")
    owners = bytearray()
    indices = array("i")
    # Rust facts carry (q, r, player, phase, idx, first_q, first_r); the encoder
    # only uses q, r, player, idx, so the rest is intentionally dropped here.
    for q, r, player, _phase, idx, _first_q, _first_r in history_facts:
        coords.append(int(q))
        coords.append(int(r))
        owners.append(_PLAYER_INDEX[str(player)])
        indices.append(int(idx))
    return _PackedHistory(coords=coords, owners=bytes(owners), indices=indices)


# UNUSED(2026-06-12): no references found in packages/tests/scripts (excl.
# scripts/archive) -- repo-wide grep finds only this definition.
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


def _compact_policy(
    weights: Mapping[int, float] | Sequence[tuple[int, float]] | None,
    *,
    normalize: bool = False,
) -> CompactVisitPolicy:
    """Pack a sparse policy into a byte-backed CompactVisitPolicy (~8 B/entry).

    The self-play search already hands back a CompactVisitPolicy (normalized in
    Rust), so the common path returns it unchanged with zero re-packing. Mapping /
    list inputs (runner players, tests) are packed here, normalized if requested.
    Stored policies feed dense_policy_target (which re-normalizes) and _policy_kl,
    both of which only iterate (int, float) pairs.
    """

    if weights is None:
        weights = ()
    # Fast path: native search output is already a normalized byte-backed policy.
    if isinstance(weights, CompactVisitPolicy):
        return weights
    items = weights.items() if isinstance(weights, Mapping) else tuple(weights)
    actions: list[int] = []
    values: list[float] = []
    for action, weight in items:
        actions.append(int(action))
        values.append(float(weight))
    if normalize:
        total = sum(values)
        if total <= 0.0:
            raise ValueError("policy weights must contain positive mass")
        values = [value / total for value in values]
    count = len(actions)
    return CompactVisitPolicy(
        action_ids_bytes=struct.pack(f"{count}I", *actions),
        weights_bytes=struct.pack(f"{count}f", *values),
        count=count,
    )
