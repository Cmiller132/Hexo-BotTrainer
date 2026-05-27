"""True axial-coordinate D6 transforms for Hexo samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


D6_SIZE = 12
_COORD_OFFSET = 1 << 15


@dataclass(frozen=True, slots=True)
class Axial:
    q: int
    r: int


@dataclass(frozen=True, slots=True)
class D6Symmetry:
    index: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.index) < D6_SIZE:
            raise ValueError(f"D6 index must be in [0, {D6_SIZE}); got {self.index!r}")


def pack_coord_id(coord: Axial | tuple[int, int]) -> int:
    q, r = _coord_pair(coord)
    return ((q + _COORD_OFFSET) << 16) | (r + _COORD_OFFSET)


def unpack_coord_id(action_id: int) -> Axial:
    value = int(action_id)
    q = (value >> 16) - _COORD_OFFSET
    r = (value & 0xFFFF) - _COORD_OFFSET
    return Axial(q=q, r=r)


def unpack_coord_pair(action_id: int) -> tuple[int, int]:
    value = int(action_id)
    return (value >> 16) - _COORD_OFFSET, (value & 0xFFFF) - _COORD_OFFSET


def transform_action_id(action_id: int, symmetry: D6Symmetry | int, *, center: Axial | tuple[int, int] = (0, 0)) -> int:
    return pack_coord_id(transform_coord(unpack_coord_id(action_id), symmetry, center=center))


def transform_action_ids(
    action_ids: Iterable[int],
    symmetry: D6Symmetry | int,
    *,
    center: Axial | tuple[int, int] = (0, 0),
) -> tuple[int, ...]:
    return tuple(transform_action_id(action_id, symmetry, center=center) for action_id in action_ids)


def transform_coord(
    coord: Axial | tuple[int, int],
    symmetry: D6Symmetry | int,
    *,
    center: Axial | tuple[int, int] = (0, 0),
) -> Axial:
    """Transform `coord` by one of the twelve axial D6 symmetries around `center`."""

    index = int(getattr(symmetry, "index", symmetry))
    if not 0 <= index < D6_SIZE:
        raise ValueError(f"D6 index must be in [0, {D6_SIZE}); got {index!r}")
    q, r = _coord_pair(coord)
    cq, cr = _coord_pair(center)
    local = Axial(q - cq, r - cr)
    if index >= 6:
        local = _reflect(local)
        index -= 6
    for _ in range(index):
        local = _rotate60(local)
    return Axial(local.q + cq, local.r + cr)


def inverse_index(index: int) -> int:
    """Return the D6 index that reverses `index`."""

    resolved = int(getattr(index, "index", index))
    if not 0 <= resolved < D6_SIZE:
        raise ValueError(f"D6 index must be in [0, {D6_SIZE}); got {resolved!r}")
    basis = (Axial(1, 0), Axial(0, 1), Axial(1, -2))
    for candidate in range(D6_SIZE):
        if all(transform_coord(transform_coord(coord, resolved), candidate) == coord for coord in basis):
            return candidate
    raise RuntimeError(f"no inverse found for D6 index {index}")


def compose_indices(left: int, right: int) -> int:
    """Return the transform index equivalent to applying `left`, then `right`."""

    basis = (Axial(1, 0), Axial(0, 1), Axial(1, -2))
    transformed = tuple(transform_coord(transform_coord(coord, left), right) for coord in basis)
    for candidate in range(D6_SIZE):
        if tuple(transform_coord(coord, candidate) for coord in basis) == transformed:
            return candidate
    raise RuntimeError(f"no D6 composition found for {left}, {right}")


def _rotate60(coord: Axial) -> Axial:
    # Cube rotation (x, y, z) -> (-z, -x, -y), with axial q=x and r=z.
    return Axial(q=-coord.r, r=coord.q + coord.r)


def _reflect(coord: Axial) -> Axial:
    # Cube reflection (x, y, z) -> (x, z, y).
    return Axial(q=coord.q, r=-coord.q - coord.r)


def _coord_pair(coord: Axial | tuple[int, int]) -> tuple[int, int]:
    if isinstance(coord, Axial):
        return int(coord.q), int(coord.r)
    return int(coord[0]), int(coord[1])
