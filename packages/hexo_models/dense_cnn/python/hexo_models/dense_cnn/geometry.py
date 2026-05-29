"""Crop geometry helpers for the dense axial input view.

Model 1 always sees a fixed 41x41 crop, but Hexo coordinates are unbounded
axial coordinates. These helpers translate between axial coordinates, crop
rows/columns, and flat policy indices for the Python sample-expansion path.

The Rust encoder implements the same projection in `rust/src/encoding.rs`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .constants import BOARD_SIZE
from .d6 import Axial


def crop_center(coords: Iterable[Axial | tuple[int, int]]) -> Axial:
    """Return the rounded mean axial coordinate used as the dense crop center."""

    points = [_pair(coord) for coord in coords]
    if not points:
        return Axial(0, 0)
    q = round(sum(point[0] for point in points) / len(points))
    r = round(sum(point[1] for point in points) / len(points))
    return Axial(q, r)


def coord_at(row: int, col: int, *, center: Axial, size: int = BOARD_SIZE) -> Axial:
    """Return the axial coordinate represented by one crop row/column."""

    half = size // 2
    return Axial(q=center.q + int(col) - half, r=center.r + int(row) - half)


def coord_to_row_col(coord: Axial | tuple[int, int], *, center: Axial, size: int = BOARD_SIZE) -> tuple[int, int] | None:
    """Project an axial coordinate into crop row/column coordinates."""

    q, r = _pair(coord)
    half = size // 2
    col = q - center.q + half
    row = r - center.r + half
    if 0 <= row < size and 0 <= col < size:
        return row, col
    return None


def coord_to_flat(coord: Axial | tuple[int, int], *, center: Axial, size: int = BOARD_SIZE) -> int | None:
    """Project an axial coordinate into a flat row-major policy index."""

    row_col = coord_to_row_col(coord, center=center, size=size)
    if row_col is None:
        return None
    row, col = row_col
    return row * size + col


def flat_to_coord(index: int, *, center: Axial, size: int = BOARD_SIZE) -> Axial:
    """Convert a flat row-major policy index back to an axial coordinate."""

    row, col = divmod(int(index), size)
    return coord_at(row, col, center=center, size=size)


def hex_distance(left: Axial | tuple[int, int], right: Axial | tuple[int, int]) -> int:
    """Return axial hex distance between two coordinates."""

    lq, lr = _pair(left)
    rq, rr = _pair(right)
    dq = lq - rq
    dr = lr - rr
    return max(abs(dq), abs(dr), abs(-dq - dr))


def normalize_dense_target(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize nonnegative dense weights, returning zeros for zero mass."""

    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0.0:
        return tuple(0.0 for _ in values)
    return tuple(max(0.0, float(value)) / total for value in values)


def _pair(coord: Axial | tuple[int, int]) -> tuple[int, int]:
    if isinstance(coord, Axial):
        return int(coord.q), int(coord.r)
    return int(coord[0]), int(coord[1])
