"""Support-set construction — the one geometric law.

Ground truth is the engine, never re-derived geometry:

1. ``stones`` = occupied cells; ``legal`` = empty ∧ hex-dist <= LEGAL_RADIUS of
   any stone (ply 0 / Opening => forced {(0, 0)}).
2. ``core = stones ∪ legal``; ``halo`` = cells hex-adjacent to core, not in
   core (carries features, never logits).
3. ``support = core ∪ halo``.

One multi-source BFS of depth LEGAL_RADIUS+1 from the stones yields the support,
the halo, and the dist_to_stone feature in one pass. Geometric identities — core
is the union of radius-LEGAL_RADIUS disks (always connected), halo is exactly the
distance-(LEGAL_RADIUS+1) shell — are property tests, not construction steps.

Node order (layout contract): segments ``[ legal | stones | halo ]``, each
ascending by packed action id (== ascending signed (q, r)). Legal-prefix
property: the legal nodes of a row are exactly slots [0, legal_count).
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .constants import DIRECTIONS, HALO_DIST, LEGAL_RADIUS

# Model-side legal-move radius (NOT the game engine's). HEXFIELD_SUPPORT_RADIUS
# restricts the support to legal cells within hex-dist <= R of a stone (default
# LEGAL_RADIUS == the engine's legality, i.e. unchanged). A smaller R makes the
# model + MCTS consider fewer candidate moves (smaller support -> cheaper
# O(support^2) forward) while the engine still allows LEGAL_RADIUS. The bias table
# (BIAS_DISK_RADIUS) and DIST_SCALE stay at LEGAL_RADIUS, so the network
# architecture and feature scaling are UNCHANGED (the checkpoint loads); only the
# support shrinks. Serve (Rust featurizer) reads the same env var, so train/serve
# stay consistent.
_SUPPORT_RADIUS = int(os.environ.get("HEXFIELD_SUPPORT_RADIUS", LEGAL_RADIUS))
_SUPPORT_HALO = _SUPPORT_RADIUS + 1


class SupportContractError(ValueError):
    """Raised when :func:`build_support` is fed a non-decision (e.g. terminal)
    state — i.e. the closed-form legal set disagrees with the engine's.

    The closed-form legality ``empty ∧ dist <= LEGAL_RADIUS`` only equals the
    engine's legal set on *decision* states. On a terminal state the engine
    returns an empty legal set (terminal states are never evaluated), yet the
    closed form still yields a non-empty legal prefix — a silent latent-parity
    divergence. This error is the opt-in tripwire for that contract violation.
    """


@dataclass(frozen=True)
class Support:
    """One position's support set in canonical node order.

    coords: (N, 2) int32 axial (q, r) per node, [legal | stones | halo].
    dist:   (N,)  int32 raw hex distance to the nearest stone (0 on ply 0).
    nbr:    (N, 6) int32 row-local neighbour index per DIRECTIONS, -1 missing.
    index:  coord -> row lookup for the whole support.
    """

    coords: np.ndarray
    legal_count: int
    stone_count: int
    halo_count: int
    dist: np.ndarray
    nbr: np.ndarray
    index: dict[tuple[int, int], int]

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    def legal_coords(self) -> np.ndarray:
        return self.coords[: self.legal_count]

    def segments(self) -> tuple[range, range, range]:
        """(legal, stones, halo) row ranges."""

        a = self.legal_count
        b = a + self.stone_count
        return range(0, a), range(a, b), range(b, self.num_nodes)


def build_support(
    stones: list[tuple[int, int]],
    *,
    expected_legal: Iterable[tuple[int, int]] | None = None,
) -> Support:
    """Build the support set from the stone list (empty list == ply 0).

    CONTRACT — decision states only. ``build_support`` re-derives legality in
    closed form (``empty ∧ dist <= LEGAL_RADIUS``), which equals the engine's
    legal set *only on decision states*. Terminal states are never evaluated (the
    tree backs up engine outcomes), so callers on the train/serve paths must pass
    decision states; on a terminal state the engine returns an empty legal set
    while the closed form still produces a non-empty legal prefix — a silent
    latent-parity divergence.

    This is a contract, not a happy-path change: by default (``expected_legal``
    is ``None``) numerics are byte-for-byte unchanged. Callers that have the
    engine's legal coords cheaply at hand may opt into validation by passing
    them via ``expected_legal`` (any iterable of ``(q, r)`` coords); the
    closed-form legal set is then required to equal that set exactly, raising
    :class:`SupportContractError` otherwise (this catches terminal states,
    whose engine legal set is empty). See :func:`assert_decision_support` for a
    standalone, support-free check.
    """

    support = _build_support(stones)
    if expected_legal is not None:
        _validate_legal(support, expected_legal)
    return support


def assert_decision_support(
    stones: list[tuple[int, int]],
    expected_legal: Iterable[tuple[int, int]],
) -> Support:
    """Build the support and assert it is a decision state (opt-in tripwire).

    Thin wrapper over ``build_support(stones, expected_legal=...)`` for the
    train/serve paths that want the closed-form legal set checked against the
    engine's ``expected_legal`` (e.g. ``api.legal_action_ids`` unpacked to
    coords). Raises :class:`SupportContractError` on any divergence, including
    terminal states (engine legal set empty). Returns the validated support.
    """

    return build_support(stones, expected_legal=expected_legal)


def _validate_legal(
    support: Support, expected_legal: Iterable[tuple[int, int]]
) -> None:
    expected = {(int(q), int(r)) for q, r in expected_legal}
    derived = {
        (int(q), int(r)) for q, r in support.coords[: support.legal_count].tolist()
    }
    if derived != expected:
        missing = sorted(expected - derived)
        extra = sorted(derived - expected)
        raise SupportContractError(
            "build_support legal set diverges from the engine "
            f"(closed-form {len(derived)} vs engine {len(expected)}); "
            f"in_engine_not_closed_form={missing[:8]} "
            f"in_closed_form_not_engine={extra[:8]} — "
            "build_support is decision-states-only; a non-empty closed-form "
            "legal set against an empty engine set indicates a TERMINAL state, "
            "which is never evaluated."
        )


def _build_support(stones: list[tuple[int, int]]) -> Support:
    if not stones:
        # Ply 0: support = origin + its 6 halo neighbours (7 nodes, 1 legal);
        # dist_to_stone := 0 everywhere on this one state.
        ordered = [(0, 0)] + sorted(
            (dq, dr) for dq, dr in DIRECTIONS
        )
        coords = np.asarray(ordered, dtype=np.int32)
        dist = np.zeros(len(ordered), dtype=np.int32)
        index = {tuple(c): i for i, c in enumerate(ordered)}
        return Support(
            coords=coords,
            legal_count=1,
            stone_count=0,
            halo_count=6,
            dist=dist,
            nbr=_neighbor_table(ordered, index),
            index=index,
        )

    stone_set = set(stones)
    dist: dict[tuple[int, int], int] = {coord: 0 for coord in stone_set}
    frontier: deque[tuple[int, int]] = deque(stone_set)
    while frontier:
        cell = frontier.popleft()
        d = dist[cell]
        if d == _SUPPORT_HALO:
            continue
        q, r = cell
        for dq, dr in DIRECTIONS:
            nxt = (q + dq, r + dr)
            if nxt not in dist:
                dist[nxt] = d + 1
                frontier.append(nxt)

    legal = sorted(c for c, d in dist.items() if d <= _SUPPORT_RADIUS and c not in stone_set)
    stones_sorted = sorted(stone_set)
    halo = sorted(c for c, d in dist.items() if d == _SUPPORT_HALO)

    ordered = legal + stones_sorted + halo
    index = {coord: i for i, coord in enumerate(ordered)}
    return Support(
        coords=np.asarray(ordered, dtype=np.int32),
        legal_count=len(legal),
        stone_count=len(stones_sorted),
        halo_count=len(halo),
        dist=np.asarray([dist[c] for c in ordered], dtype=np.int32),
        nbr=_neighbor_table(ordered, index),
        index=index,
    )


def _neighbor_table(
    ordered: list[tuple[int, int]], index: dict[tuple[int, int], int]
) -> np.ndarray:
    nbr = np.full((len(ordered), 6), -1, dtype=np.int32)
    for row, (q, r) in enumerate(ordered):
        for k, (dq, dr) in enumerate(DIRECTIONS):
            j = index.get((q + dq, r + dr))
            if j is not None:
                nbr[row, k] = j
    return nbr
