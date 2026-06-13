"""Sample facts, game finalization, and train-time row expansion — spec §3/§6.

`finalize_game_samples`, the STV even-offset EMA, the future-opponent-policy
rule, and the moves-left target are exact semantic ports of the verified
restnet constructions (restnet samples.py is the test oracle). Expansion maps
targets from packed action ids onto the row's legal-prefix slots; policy mass
off the legal set is a hard error for the self policy and a tracked
projection drop (`opp_coverage`) for the opponent policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import MOVES_LEFT_CAP
from .features import PositionFacts, build_position, transform_facts
from .geometry import apply_d6, unpack_action_id
from .support import Support

STV_HORIZONS = (2, 6, 16)


@dataclass(frozen=True)
class HexfieldSampleData:
    """One decision row's raw facts + targets (players are ints 0/1)."""

    game_id: str
    turn_index: int
    current_player: int
    phase: str
    records: tuple[tuple[int, int, int, int], ...]  # (q, r, owner, placement_index)
    first_stone: tuple[int, int] | None
    own_hot: tuple[tuple[int, int], ...]
    opp_hot: tuple[tuple[int, int], ...]
    own_win: tuple[tuple[int, int], ...]
    opp_win: tuple[tuple[int, int], ...]
    policy: tuple[tuple[int, float], ...]
    opp_policy: tuple[tuple[int, float], ...] = ()
    value: float = 0.0
    short_term_value: tuple[tuple[int, float], ...] = ()
    moves_left: float = -1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def facts(self) -> PositionFacts:
        return PositionFacts(
            records=self.records,
            current_player=self.current_player,
            phase=self.phase,
            first_stone=self.first_stone,
            own_hot=self.own_hot,
            opp_hot=self.opp_hot,
            own_win=self.own_win,
            opp_win=self.opp_win,
        )


def _winner_value(winner: int | None, player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def _future_opponent_policy(
    decisions: Sequence[tuple[int, "HexfieldSampleData", float]],
    index: int,
    player: int,
    *,
    mask_from_fast: bool = False,
) -> tuple[tuple[tuple[int, float], ...], str]:
    """The next opponent decision's visit policy (restnet rule, ported):
    masked when that decision was a PCR fast search (`pcr_full=False`)."""

    for future_player, future_sample, _root_value in decisions[index + 1 :]:
        if future_player != player:
            if mask_from_fast and not future_sample.metadata.get("pcr_full", True):
                return (), "fast_unrecorded_masked"
            return tuple(future_sample.policy), "future_opponent_mcts"
    return (), "none"


def _short_term_value_targets(
    decisions: Sequence[tuple[int, "HexfieldSampleData", float]],
    index: int,
    player: int,
    horizons: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    """Per-horizon EMA of future root values stepped over FULL TURNS (even
    decision offsets only), decay (m-1)/(m+1) — restnet semantics verbatim."""

    future = decisions[index + 1 :]
    perspective = [
        root_value if future_player == player else -root_value
        for future_player, _sample, root_value in future
    ]
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


def finalize_game_samples(
    pending: Sequence[tuple[int, HexfieldSampleData, float]],
    winner: int | None,
    horizons: Sequence[int] = STV_HORIZONS,
    *,
    truncated: bool = False,
    soft_z_lambda: float = 0.0,
    mask_opp_from_fast: bool = False,
) -> list[HexfieldSampleData]:
    """Assign outcome targets to a finished game's pre-decision samples.

    Hard z is the v1 value target (soft_z_lambda stays 0 in production; the
    parameter is ported for parity with the restnet oracle). Truncated games'
    rows are never written by the caller (`drop_truncated` is unconditional in
    hexfield, spec §5.1) — the moves_left -1 sentinel path survives almost
    solely for the Phase-B legacy adapter.
    """

    decisions = list(pending)
    horizons = tuple(int(h) for h in horizons)
    lam = float(soft_z_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"soft_z_lambda must be in [0, 1], got {soft_z_lambda!r}")
    finalized: list[HexfieldSampleData] = []
    for index, (player, sample, root_value) in enumerate(decisions):
        opp_policy, opp_source = _future_opponent_policy(
            decisions, index, player, mask_from_fast=mask_opp_from_fast
        )
        metadata = {
            **dict(sample.metadata),
            "opp_policy_source": opp_source,
            "truncated": bool(truncated),
        }
        hard_z = _winner_value(winner, player)
        value_target = (1.0 - lam) * hard_z + lam * float(root_value) if lam > 0.0 else hard_z
        finalized.append(
            replace(
                sample,
                value=value_target,
                opp_policy=opp_policy,
                short_term_value=_short_term_value_targets(decisions, index, player, horizons),
                moves_left=float(len(decisions) - index - 1) if not truncated else -1.0,
                metadata=metadata,
            )
        )
    return finalized


@dataclass(frozen=True)
class ExpandedRow:
    """One expanded training row (numpy; collated by batching.py)."""

    support: Support
    feats: np.ndarray  # (N, F) f32
    policy: np.ndarray  # (L,) f32 over the legal prefix
    opp_policy: np.ndarray  # (L,) f32; zero row when absent/masked/uncovered
    opp_coverage: float  # kept mass / total mass (1.0 when no target existed)
    value: float
    stvalue: np.ndarray  # (H,) f32
    stvalue_mask: np.ndarray  # (H,) f32
    moves_left: float  # normalized to [-1, 1]; 0 when masked
    moves_left_mask: float


def expand_sample(
    sample: HexfieldSampleData,
    *,
    symmetry: int = 0,
    horizons: Sequence[int] = STV_HORIZONS,
) -> ExpandedRow:
    """Facts -> (support, features, legal-prefix targets) under one D6 draw.

    The drawn symmetry is applied to all stored coordinate facts (including
    policy / opp-policy action ids); support, node order, features, and
    target slots are rebuilt from the transformed facts. Augmentation is
    exact for 100% of rows — no spill, no drops (spec §4).
    """

    facts = transform_facts(sample.facts(), symmetry)
    sup, feats = build_position(facts)
    legal_count = sup.legal_count

    policy = np.zeros(legal_count, dtype=np.float32)
    total = 0.0
    for action_id, weight in sample.policy:
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError("policy weights must be finite and nonnegative")
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is None:
            raise ValueError(
                f"policy target action {action_id} is off the legal set (hard error)"
            )
        policy[slot] += w
        total += w
    if total <= 0.0:
        raise ValueError("policy target must carry positive mass")

    opp = np.zeros(legal_count, dtype=np.float32)
    opp_total = 0.0
    opp_kept = 0.0
    for action_id, weight in sample.opp_policy:
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError("opp policy weights must be finite and nonnegative")
        opp_total += w
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is not None:
            opp[slot] += w  # projection onto THIS row's legal set
            opp_kept += w
    opp_coverage = (opp_kept / opp_total) if opp_total > 0.0 else 1.0

    horizons = tuple(int(h) for h in horizons)
    stvalue = np.zeros(len(horizons), dtype=np.float32)
    stvalue_mask = np.zeros(len(horizons), dtype=np.float32)
    horizon_index = {h: i for i, h in enumerate(horizons)}
    for h, v in sample.short_term_value:
        col = horizon_index.get(int(h))
        if col is not None:
            stvalue[col] = float(v)
            stvalue_mask[col] = 1.0

    if float(sample.moves_left) >= 0.0:
        moves_left = 2.0 * min(1.0, float(sample.moves_left) / MOVES_LEFT_CAP) - 1.0
        moves_left_mask = 1.0
    else:
        moves_left = 0.0
        moves_left_mask = 0.0

    return ExpandedRow(
        support=sup,
        feats=feats,
        policy=policy,
        opp_policy=opp,
        opp_coverage=opp_coverage,
        value=float(sample.value),
        stvalue=stvalue,
        stvalue_mask=stvalue_mask,
        moves_left=moves_left,
        moves_left_mask=moves_left_mask,
    )


def _legal_slot(sup: Support, symmetry: int, action_id: int) -> int | None:
    q, r = unpack_action_id(action_id)
    cell = apply_d6(symmetry, q, r)
    slot = sup.index.get(cell)
    if slot is None or slot >= sup.legal_count:
        return None
    return slot
