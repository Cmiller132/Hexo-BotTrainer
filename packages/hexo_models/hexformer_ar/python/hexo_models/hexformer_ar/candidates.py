"""Legal candidate frontier construction for sparse pointer policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .config import HexformerCandidateConfig
from .coordinates import as_axial, cells_within_radius, hex_distance, pack_action_id, unpack_action_id


TAG_LEGAL = 1 << 0
TAG_TACTICAL = 1 << 1
TAG_RECENT = 1 << 2
TAG_FRONTIER = 1 << 3
TAG_IMMEDIATE_WIN = 1 << 4
TAG_MUST_BLOCK = 1 << 5


@dataclass(frozen=True, slots=True)
class Candidate:
    action_id: int
    coord: Axial
    tags: int = TAG_LEGAL
    priority: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateSet:
    candidates: tuple[Candidate, ...]
    legal_action_ids: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def action_ids(self) -> tuple[int, ...]:
        return tuple(candidate.action_id for candidate in self.candidates)


def build_candidate_frontier(
    python_state: object,
    legal_action_ids: Sequence[int],
    *,
    tactical_action_ids: Sequence[int] = (),
    immediate_win_action_ids: Sequence[int] = (),
    must_block_action_ids: Sequence[int] = (),
    config: HexformerCandidateConfig | None = None,
) -> CandidateSet:
    cfg = config or HexformerCandidateConfig()
    legal_ids = tuple(int(item) for item in legal_action_ids)
    legal_set = set(legal_ids)
    by_id: dict[int, Candidate] = {}

    def add(action_id: int, tags: int, priority: float) -> None:
        action_id = int(action_id)
        if action_id not in legal_set:
            return
        coord = unpack_action_id(action_id)
        existing = by_id.get(action_id)
        if existing is None:
            by_id[action_id] = Candidate(action_id=action_id, coord=coord, tags=TAG_LEGAL | tags, priority=priority)
        else:
            by_id[action_id] = Candidate(
                action_id=action_id,
                coord=existing.coord,
                tags=existing.tags | tags,
                priority=max(existing.priority, priority),
            )

    for action_id in immediate_win_action_ids:
        add(int(action_id), TAG_TACTICAL | TAG_IMMEDIATE_WIN, 100.0)
    for action_id in must_block_action_ids:
        add(int(action_id), TAG_TACTICAL | TAG_MUST_BLOCK, 90.0)
    for action_id in tactical_action_ids:
        add(int(action_id), TAG_TACTICAL, 75.0)
    for action_id in tuple(immediate_win_action_ids) + tuple(must_block_action_ids) + tuple(tactical_action_ids):
        center = unpack_action_id(int(action_id))
        for coord in cells_within_radius(center, cfg.tactical_radius):
            distance = hex_distance(coord, center)
            add(pack_action_id(coord), TAG_TACTICAL, 70.0 - distance)

    history = tuple(getattr(python_state, "placement_history", ()))
    for record in history[-8:]:
        record_coord = getattr(record, "coord", None)
        if record_coord is None:
            continue
        for coord in cells_within_radius(record_coord, cfg.recent_radius):
            add(pack_action_id(coord), TAG_RECENT, 50.0 - hex_distance(coord, record_coord))

    occupied = tuple(getattr(getattr(python_state, "board", object()), "occupied", ()))
    for occupied_coord in occupied[-64:]:
        center = as_axial(occupied_coord)
        for coord in cells_within_radius(center, cfg.frontier_radius):
            distance = hex_distance(coord, center)
            add(pack_action_id(coord), TAG_FRONTIER, 30.0 - distance)

    if len(legal_ids) <= cfg.include_all_legal_below:
        for action_id in legal_ids:
            add(action_id, TAG_FRONTIER, 10.0)

    if cfg.require_tactical_candidates and (immediate_win_action_ids or must_block_action_ids):
        required = {int(item) for item in tuple(immediate_win_action_ids) + tuple(must_block_action_ids)}
        missing = required.difference(by_id)
        for action_id in missing:
            add(action_id, TAG_TACTICAL, 95.0)

    if not by_id:
        for action_id in legal_ids[: cfg.max_candidates]:
            add(action_id, TAG_FRONTIER, 1.0)

    ordered = sorted(
        by_id.values(),
        key=lambda item: (-item.priority, item.coord.q, item.coord.r, item.action_id),
    )
    limited = tuple(ordered[: cfg.max_candidates])
    if not limited and legal_ids:
        limited = tuple(Candidate(action_id=action_id, coord=unpack_action_id(action_id)) for action_id in legal_ids[:1])
    return CandidateSet(
        candidates=limited,
        legal_action_ids=legal_ids,
        metadata={
            "legal_count": len(legal_ids),
            "candidate_count": len(limited),
            "truncated": len(ordered) > len(limited),
            "immediate_win_count": len(tuple(immediate_win_action_ids)),
            "must_block_count": len(tuple(must_block_action_ids)),
            "tactical_radius": cfg.tactical_radius,
            "frontier_radius": cfg.frontier_radius,
            "require_tactical_candidates": cfg.require_tactical_candidates,
        },
    )
