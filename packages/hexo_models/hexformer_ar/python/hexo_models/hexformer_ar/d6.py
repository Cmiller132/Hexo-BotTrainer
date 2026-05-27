"""D6 transforms over model-local sparse coordinates and action IDs."""

from __future__ import annotations

from dataclasses import dataclass

from .coordinates import Axial, as_axial, pack_action_id, unpack_action_id


D6_SIZE = 12


@dataclass(frozen=True, slots=True)
class D6Symmetry:
    index: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.index) < D6_SIZE:
            raise ValueError(f"D6 index must be in [0, {D6_SIZE}); got {self.index!r}")


def transform_coord(coord: object, symmetry: D6Symmetry | int, *, center: object = (0, 0)) -> Axial:
    index = int(getattr(symmetry, "index", symmetry))
    if not 0 <= index < D6_SIZE:
        raise ValueError(f"D6 index must be in [0, {D6_SIZE}); got {index!r}")
    c = as_axial(coord)
    anchor = as_axial(center)
    local = Axial(c.q - anchor.q, c.r - anchor.r)
    if index >= 6:
        local = _reflect(local)
        index -= 6
    for _ in range(index):
        local = _rotate60(local)
    return Axial(local.q + anchor.q, local.r + anchor.r)


def transform_action_id(action_id: int, symmetry: D6Symmetry | int, *, center: object = (0, 0)) -> int:
    return pack_action_id(transform_coord(unpack_action_id(int(action_id)), symmetry, center=center))


def inverse_index(index: int) -> int:
    resolved = int(getattr(index, "index", index))
    basis = (Axial(1, 0), Axial(0, 1), Axial(1, -2))
    for candidate in range(D6_SIZE):
        if all(transform_coord(transform_coord(coord, resolved), candidate) == coord for coord in basis):
            return candidate
    raise RuntimeError(f"no inverse found for D6 index {index}")


def _rotate60(coord: Axial) -> Axial:
    return Axial(q=-coord.r, r=coord.q + coord.r)


def _reflect(coord: Axial) -> Axial:
    return Axial(q=coord.q, r=-coord.q - coord.r)
