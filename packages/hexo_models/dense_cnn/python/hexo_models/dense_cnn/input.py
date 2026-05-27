"""Model 1 input encoding from compact state facts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

import torch

from .constants import (
    BOARD_AREA,
    BOARD_SIZE,
    INPUT_CHANNELS,
    PLANE_CENTER_DISTANCE,
    PLANE_EMPTY,
    PLANE_FIRST_STONE,
    PLANE_LEGAL,
    PLANE_OPPONENT_HOT,
    PLANE_OPPONENT_LAST_TURN,
    PLANE_OPPONENT_RECENCY,
    PLANE_OPPONENT_STONES,
    PLANE_OWN_HOT,
    PLANE_OWN_RECENCY,
    PLANE_OWN_STONES,
    PLANE_PLAYER_COLOUR,
    PLANE_SECOND_PLACEMENT,
)
from .d6 import Axial, D6Symmetry, transform_action_id, transform_coord, unpack_coord_id, unpack_coord_pair
from .geometry import coord_to_flat, coord_to_row_col, hex_distance


def build_input_planes(
    *,
    current_player: str,
    phase: str,
    center: Axial,
    stones: Sequence[tuple[int, int, str]],
    legal_action_ids: Sequence[int],
    placement_history: Sequence[tuple[int, int, str, str, int, int | None, int | None]] = (),
    first_stone: tuple[int, int] | None = None,
    own_hot: Sequence[tuple[int, int]] = (),
    opponent_hot: Sequence[tuple[int, int]] = (),
    opponent_last_turn: Sequence[tuple[int, int]] = (),
    symmetry: D6Symmetry | int = 0,
) -> torch.Tensor:
    """Encode one compact sample into a dense model tensor."""

    planes = torch.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    planes[PLANE_EMPTY].fill_(1.0)
    planes[PLANE_CENTER_DISTANCE].copy_(_distance_plane(BOARD_SIZE))
    identity = _is_identity_symmetry(symmetry)
    transformed_stones = {
        ((int(q), int(r)) if identity else _coord_tuple(transform_coord((q, r), symmetry, center=center))): player
        for q, r, player in stones
    }

    for (q, r), player in transformed_stones.items():
        row_col = coord_to_row_col((q, r), center=center)
        if row_col is None:
            continue
        row, col = row_col
        plane = PLANE_OWN_STONES if player == current_player else PLANE_OPPONENT_STONES
        planes[plane, row, col] = 1.0
        planes[PLANE_EMPTY, row, col] = 0.0

    legal_plane = planes[PLANE_LEGAL].view(-1)
    for action_id in legal_action_ids:
        flat = (
            _flat_for_action_id(action_id, center=center)
            if identity
            else coord_to_flat(unpack_coord_id(transform_action_id(action_id, symmetry, center=center)), center=center)
        )
        if flat is not None:
            legal_plane[flat] = 1.0

    if phase == "SecondStone":
        planes[PLANE_SECOND_PLACEMENT].fill_(1.0)
    if first_stone is not None:
        _set_coord(
            planes,
            PLANE_FIRST_STONE,
            first_stone if identity else transform_coord(first_stone, symmetry, center=center),
            center,
        )
    if current_player == "player0":
        planes[PLANE_PLAYER_COLOUR].fill_(1.0)

    latest_index = max((item[4] for item in placement_history), default=0)
    for q, r, player, _phase, placement_index, _first_q, _first_r in reversed(tuple(placement_history)):
        coord = (int(q), int(r)) if identity else transform_coord((q, r), symmetry, center=center)
        row_col = coord_to_row_col(coord, center=center)
        if row_col is None:
            continue
        row, col = row_col
        weight = 1.0 / (1.0 + max(0, latest_index - int(placement_index)))
        plane = PLANE_OWN_RECENCY if player == current_player else PLANE_OPPONENT_RECENCY
        planes[plane, row, col] = max(float(planes[plane, row, col]), weight)

    for coord in own_hot:
        _set_coord(planes, PLANE_OWN_HOT, coord if identity else transform_coord(coord, symmetry, center=center), center)
    for coord in opponent_hot:
        _set_coord(planes, PLANE_OPPONENT_HOT, coord if identity else transform_coord(coord, symmetry, center=center), center)
    for coord in opponent_last_turn:
        _set_coord(
            planes,
            PLANE_OPPONENT_LAST_TURN,
            coord if identity else transform_coord(coord, symmetry, center=center),
            center,
        )

    return planes


def dense_policy_target(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    center: Axial,
    symmetry: D6Symmetry | int = 0,
) -> torch.Tensor:
    target = torch.zeros((BOARD_AREA,), dtype=torch.float32)
    items = weights.items() if isinstance(weights, Mapping) else tuple(weights)
    identity = _is_identity_symmetry(symmetry)
    for action_id, weight in items:
        flat = (
            _flat_for_action_id(action_id, center=center)
            if identity
            else coord_to_flat(unpack_coord_id(transform_action_id(int(action_id), symmetry, center=center)), center=center)
        )
        if flat is not None:
            target[flat] += max(0.0, float(weight))

    total = target.sum().clamp_min(1.0e-8)
    return target / total


def legal_mask_flat(
    legal_action_ids: Iterable[int],
    *,
    center: Axial,
    symmetry: D6Symmetry | int = 0,
) -> torch.Tensor:
    mask = torch.zeros((BOARD_AREA,), dtype=torch.bool)
    identity = _is_identity_symmetry(symmetry)
    for action_id in legal_action_ids:
        flat = (
            _flat_for_action_id(action_id, center=center)
            if identity
            else coord_to_flat(unpack_coord_id(transform_action_id(action_id, symmetry, center=center)), center=center)
        )
        if flat is not None:
            mask[flat] = True
    return mask


def _set_coord(planes: torch.Tensor, plane: int, coord: Axial, center: Axial) -> None:
    row_col = coord_to_row_col(coord, center=center)
    if row_col is None:
        return
    row, col = row_col
    planes[plane, row, col] = 1.0


def _coord_tuple(coord: Axial) -> tuple[int, int]:
    return int(coord.q), int(coord.r)


def _is_identity_symmetry(symmetry: D6Symmetry | int) -> bool:
    return int(getattr(symmetry, "index", symmetry)) == 0


def _flat_for_action_id(action_id: int, *, center: Axial) -> int | None:
    q, r = unpack_coord_pair(int(action_id))
    half = BOARD_SIZE // 2
    row = r - int(center.r) + half
    col = q - int(center.q) + half
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row * BOARD_SIZE + col
    return None


@lru_cache(maxsize=8)
def _distance_plane(size: int) -> torch.Tensor:
    half = size // 2
    rows = torch.arange(size, dtype=torch.float32).view(size, 1) - half
    cols = torch.arange(size, dtype=torch.float32).view(1, size) - half
    s = -rows - cols
    return torch.maximum(torch.maximum(rows.abs(), cols.abs()), s.abs()) / float(size - 1)
