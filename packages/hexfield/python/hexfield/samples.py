"""Sample facts, game finalization, and train-time row expansion.

`finalize_game_samples`, the STV even-offset EMA, the future-opponent-policy
rule, and the moves-left target are exact semantic ports of the restnet
constructions (restnet samples.py is the test oracle). Expansion maps targets
from packed action ids onto the row's legal-prefix slots; policy mass off the
legal set is a hard error for the self policy and a tracked projection drop
(`opp_coverage`) for the opponent policy.
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
    q_policy: tuple[tuple[int, float], ...] = ()  # (action_id, child Q); parallel to policy
    value: float = 0.0
    short_term_value: tuple[tuple[int, float], ...] = ()
    moves_left: float = -1.0
    policy_surprise: float = 0.0  # KL(visit ‖ root prior); reweights the self policy CE
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


def _policy_surprise_kl(action_ids, weights, prior_ids, prior_weights, *, eps: float = 1e-8) -> float:
    """KL(visit ‖ prior); visit normalized to sum 1; prior already ~sum 1.

    Returns ``max(0, kl)``, or ``0.0`` if non-finite. Mirrors restnet
    ``replay._policy_kl`` (self-policy only). Imported by ``selfplay.py`` to
    record the per-row policy-surprise scalar at decision time; the in-loss
    weight is recomputed at collate from the recorded scalar.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0.0:
        return 0.0
    prior_map = {int(a): float(p) for a, p in zip(prior_ids, prior_weights)}
    kl = 0.0
    for a, tw in zip((int(x) for x in action_ids), (w / s).tolist()):
        if tw <= 0.0:
            continue
        pw = max(prior_map.get(a, 0.0), eps)
        kl += tw * float(np.log((tw + eps) / pw))
    return max(0.0, kl) if np.isfinite(kl) else 0.0


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

    Hard z is the value target (soft_z_lambda stays 0 in production; the
    parameter is ported for parity with the restnet oracle).

    Truncated games (``truncated=True``, ``winner=None``) ARE written (with their
    outcome-dependent heads masked downstream): ``metadata['truncated']`` is set,
    ``moves_left`` is the -1 sentinel (→ moves_left_mask=0 at expand), and the
    same flag zeroes the value/stvalue/cell_q masks in ``expand_sample``. The
    outcome-INDEPENDENT heads (policy, opp_policy) train normally on truncated
    rows. The hard-z value target stays 0.0 for truncated rows but is NEVER used
    because value_mask gates it to zero loss. Completed games (truncated=False)
    are untouched.
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
    value_mask: float  # 1.0 for completed games; 0.0 for truncated (no winner)
    policy_valid: float  # NEW: 1.0=full (train policy/opp/soft/cell_q); 0.0=fast (value-only)
    stvalue: np.ndarray  # (H,) f32
    stvalue_mask: np.ndarray  # (H,) f32
    moves_left: float  # normalized to [-1, 1]; 0 when masked
    moves_left_mask: float
    cell_q: np.ndarray  # (L,) f32 per-cell Q target over the legal prefix; 0 where absent
    cell_q_mask: np.ndarray  # (L,) f32 presence mask (Q=0.0 is a valid target)
    policy_surprise: float  # KL(visit ‖ prior); collate turns it into the self-CE weight


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
    exact for 100% of rows — no spill, no drops.
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
    policy_valid = 0.0 if sample.metadata.get("pcr_full", True) is False else 1.0
    if policy_valid != 0.0 and total <= 0.0:
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

    # Per-cell Q target: scalar child Q projected onto THIS row's legal set + a
    # presence mask (mirrors the opp_policy projection but as scalar value+mask
    # rather than a probability distribution). Off-legal Q is dropped, like opp.
    cell_q = np.zeros(legal_count, dtype=np.float32)
    cell_q_mask = np.zeros(legal_count, dtype=np.float32)
    for action_id, q in sample.q_policy:
        qv = float(q)
        if not np.isfinite(qv) or qv < -1.0 or qv > 1.0:
            raise ValueError("cell_q targets must be finite and in [-1, 1]")
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is not None:
            cell_q[slot] = qv  # SCALAR assign (one action -> one distinct cell)
            cell_q_mask[slot] = 1.0

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

    # Truncated games (max_game_plies hit, no engine winner) have no grounded
    # terminal outcome: the value head's hard-z target, the bootstrapped STV
    # targets, and the value/Q-derived cell_q target are all undefined. Mask the
    # whole outcome/terminal-dependent family to ZERO loss for these rows while
    # leaving policy / opp_policy (visit-count distributions, outcome-INDEPENDENT)
    # to train normally. moves_left is ALREADY masked via its -1 sentinel above.
    # Completed rows (truncated absent/False) keep value_mask=1.0 and the
    # presence masks exactly as built.
    truncated = bool(sample.metadata.get("truncated", False))
    if truncated:
        value_mask = 0.0
        stvalue_mask = np.zeros_like(stvalue_mask)
        cell_q_mask = np.zeros_like(cell_q_mask)
    else:
        value_mask = 1.0
    # Fast (value-only) rows: mask the search-distribution head cell_q. policy/
    # opp_policy/soft_policy are gated by policy_valid at the loss; cell_q is gated
    # by its presence mask, so zeroing it here is the operative cell_q gate.
    # (For fast rows q_policy=() already ⇒ cell_q_mask is all-zero; this is
    # defense-in-depth and the explicit, parity-checked path.)
    if policy_valid == 0.0:
        cell_q_mask = np.zeros_like(cell_q_mask)

    return ExpandedRow(
        support=sup,
        feats=feats,
        policy=policy,
        opp_policy=opp,
        opp_coverage=opp_coverage,
        value=float(sample.value),
        value_mask=value_mask,
        policy_valid=policy_valid,
        stvalue=stvalue,
        stvalue_mask=stvalue_mask,
        moves_left=moves_left,
        moves_left_mask=moves_left_mask,
        cell_q=cell_q,
        cell_q_mask=cell_q_mask,
        policy_surprise=float(sample.policy_surprise),
    )


def _legal_slot(sup: Support, symmetry: int, action_id: int) -> int | None:
    q, r = unpack_action_id(action_id)
    cell = apply_d6(symmetry, q, r)
    slot = sup.index.get(cell)
    if slot is None or slot >= sup.legal_count:
        return None
    return slot
