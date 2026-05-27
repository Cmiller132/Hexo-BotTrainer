"""Relative axial/cube coordinate helpers and action-ID adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Axial:
    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r


@dataclass(frozen=True, slots=True)
class RelativeCoord:
    dq: int
    dr: int
    ds: int
    distance: int
    ring: int


def as_axial(coord: object) -> Axial:
    if isinstance(coord, Axial):
        return coord
    if hasattr(coord, "q") and hasattr(coord, "r"):
        return Axial(int(getattr(coord, "q")), int(getattr(coord, "r")))
    if isinstance(coord, Sequence) and not isinstance(coord, (str, bytes, bytearray)):
        return Axial(int(coord[0]), int(coord[1]))
    raise TypeError(f"cannot interpret axial coordinate {coord!r}")


def pack_action_id(coord: object) -> int:
    """Pack through `hexo_engine.types` when available.

    Keeping this indirection local makes the model use the existing 16-bit
    engine-compatible coordinate IDs without pushing packing details into every
    sparse feature builder.
    """

    try:
        from hexo_engine.types import AxialCoord, pack_coord_id
    except Exception:
        axial = as_axial(coord)
        _check_i16(axial.q)
        _check_i16(axial.r)
        offset = 1 << 15
        return ((axial.q + offset) << 16) | (axial.r + offset)
    axial = as_axial(coord)
    return int(pack_coord_id(AxialCoord(axial.q, axial.r)))


def unpack_action_id(action_id: int) -> Axial:
    try:
        from hexo_engine.types import unpack_coord_id
    except Exception:
        value = int(action_id)
        offset = 1 << 15
        return Axial((value >> 16) - offset, (value & 0xFFFF) - offset)
    coord = unpack_coord_id(int(action_id))
    return Axial(int(coord.q), int(coord.r))


def choose_anchor(stones: Iterable[object], *, opening: bool = False) -> Axial:
    coords = tuple(as_axial(coord) for coord in stones)
    if opening or not coords:
        return Axial(0, 0)
    q_sum = sum(coord.q for coord in coords)
    r_sum = sum(coord.r for coord in coords)
    return Axial(_round_half_away(q_sum / len(coords)), _round_half_away(r_sum / len(coords)))


def relative(coord: object, anchor: object) -> RelativeCoord:
    c = as_axial(coord)
    a = as_axial(anchor)
    dq = c.q - a.q
    dr = c.r - a.r
    ds = -dq - dr
    distance = max(abs(dq), abs(dr), abs(ds))
    return RelativeCoord(dq=dq, dr=dr, ds=ds, distance=distance, ring=distance)


def hex_distance(left: object, right: object) -> int:
    l = as_axial(left)
    r = as_axial(right)
    dq = l.q - r.q
    dr = l.r - r.r
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))


def cells_within_radius(center: object, radius: int) -> tuple[Axial, ...]:
    c = as_axial(center)
    out: list[Axial] = []
    for dq in range(-int(radius), int(radius) + 1):
        r_min = max(-int(radius), -dq - int(radius))
        r_max = min(int(radius), -dq + int(radius))
        for dr in range(r_min, r_max + 1):
            out.append(Axial(c.q + dq, c.r + dr))
    return tuple(out)


def line_cells(start: object, axis: str, length: int = 6) -> tuple[Axial, ...]:
    s = as_axial(start)
    vector = {
        "Q": (1, 0),
        "R": (0, 1),
        "QR": (1, -1),
    }[str(axis)]
    return tuple(Axial(s.q + vector[0] * index, s.r + vector[1] * index) for index in range(length))


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


def _check_i16(value: int) -> None:
    if int(value) < -(1 << 15) or int(value) > (1 << 15) - 1:
        raise ValueError(f"coordinate component outside i16 range: {value}")
