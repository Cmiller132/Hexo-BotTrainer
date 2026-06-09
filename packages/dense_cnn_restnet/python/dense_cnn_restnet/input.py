"""Python expansion of compact samples into Model 1 tensors.

Rust generates compact sample facts from live engine states; this module turns
those facts into training tensors. It applies the requested D6 symmetry first,
then projects stones, legal actions, recency, hot cells, and target policies
into the same 41x41 crop contract used by native inference encoding.

Facts that fall outside the current crop are skipped because the model cannot
represent them in its fixed dense view. Invalid target weights still raise at
this boundary instead of being silently repaired.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from math import isfinite

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
    """Encode one compact sample into a dense model tensor.

    This is the Python training equivalent of Rust `encode_model1_state_inner`.
    It starts from compact sample facts, applies the requested D6 transform, and
    writes each fact into the fixed plane index defined in `constants.py`.
    """

    planes = torch.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    planes[PLANE_EMPTY].fill_(1.0)
    planes[PLANE_CENTER_DISTANCE].copy_(_distance_plane(BOARD_SIZE))
    identity = _is_identity_symmetry(symmetry)
    # Stones are transformed once into a coordinate->owner map, then projected
    # into the crop. Out-of-crop facts remain valid game facts but cannot be
    # represented in this fixed dense input.
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
        if flat is None:
            continue
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
    # Recency planes store the strongest recency weight per cell. A cell can
    # appear in history only once in legal play, but keeping max mirrors the Rust
    # encoder and makes duplicate bad data fail later through target validation.
    for q, r, player, _phase, placement_index, _first_q, _first_r in reversed(tuple(placement_history)):
        coord = (int(q), int(r)) if identity else transform_coord((q, r), symmetry, center=center)
        row_col = coord_to_row_col(coord, center=center)
        if row_col is None:
            continue
        row, col = row_col
        weight = 1.0 / (1.0 + latest_index - int(placement_index))
        plane = PLANE_OWN_RECENCY if player == current_player else PLANE_OPPONENT_RECENCY
        if weight > float(planes[plane, row, col]):
            planes[plane, row, col] = weight

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
    allow_empty: bool = False,
) -> torch.Tensor:
    """Project sparse action weights into a normalized dense crop target."""

    target = torch.zeros((BOARD_AREA,), dtype=torch.float32)
    items = weights.items() if isinstance(weights, Mapping) else tuple(weights)
    identity = _is_identity_symmetry(symmetry)
    for action_id, weight in items:
        weight = float(weight)
        if not isfinite(weight) or weight < 0.0:
            raise ValueError(f"policy target weight for action {int(action_id)} must be finite and >= 0")
        flat = (
            _flat_for_action_id(action_id, center=center)
            if identity
            else coord_to_flat(unpack_coord_id(transform_action_id(int(action_id), symmetry, center=center)), center=center)
        )
        if flat is None:
            continue
        target[flat] += weight

    total = target.sum()
    if float(total.item()) <= 0.0:
        if allow_empty:
            return target
        raise ValueError("policy target must contain positive probability mass")
    return target / total


def legal_mask_flat(
    legal_action_ids: Iterable[int],
    *,
    center: Axial,
    symmetry: D6Symmetry | int = 0,
) -> torch.Tensor:
    """Return a flat boolean mask for legal in-crop policy cells."""

    mask = torch.zeros((BOARD_AREA,), dtype=torch.bool)
    identity = _is_identity_symmetry(symmetry)
    for action_id in legal_action_ids:
        flat = (
            _flat_for_action_id(action_id, center=center)
            if identity
            else coord_to_flat(unpack_coord_id(transform_action_id(action_id, symmetry, center=center)), center=center)
        )
        if flat is None:
            continue
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
