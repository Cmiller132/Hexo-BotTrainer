"""Hex D6 symmetry group on axial coordinates (re-derived + tested).

The hex grid has a 12-element dihedral symmetry group D6 (6 rotations × a
reflection). We re-derive it here from cube coordinates rather than importing the
old hexformer_ar group, and unit-test the group laws (`tests/test_hexgt_d6.py`).

Cube coords for axial `(q, r)`: `x = q`, `z = r`, `y = -q - r` (so x+y+z = 0).
A 60° rotation cycles the cube axes with a sign flip; the base reflection swaps
two cube axes. Composing gives all 12 elements.

hexgt's model is D6-INVARIANT by construction (invariant node/edge features +
permutation-equivariant message passing/attention, see `features.py`), so D6 is
NOT applied as training-time augmentation. These transforms exist to (a) build
the genuinely-rotated states the equivariance test replays, and (b) remain
available if a future variant wants explicit augmentation.
"""

from __future__ import annotations

D6_SIZE = 12


def _to_cube(q: int, r: int) -> tuple[int, int, int]:
    return (q, -q - r, r)  # (x, y, z)


def _from_cube(x: int, y: int, z: int) -> tuple[int, int]:
    return (x, z)  # (q, r)


def _rot60(c: tuple[int, int, int]) -> tuple[int, int, int]:
    """One 60° rotation: (x, y, z) -> (-z, -x, -y)."""

    x, y, z = c
    return (-z, -x, -y)


def _reflect(c: tuple[int, int, int]) -> tuple[int, int, int]:
    """Base reflection: swap y and z, i.e. (x, y, z) -> (x, z, y)."""

    x, y, z = c
    return (x, z, y)


def _build_elements() -> list:
    """Return the 12 D6 transforms as callables on `(q, r)`.

    Order: rotations r^0..r^5, then reflection∘r^0..r^5. Index 0 is identity.
    """

    elements = []
    for k in range(6):
        for reflected in (False, True):
            def transform(q: int, r: int, k: int = k, reflected: bool = reflected) -> tuple[int, int]:
                c = _to_cube(q, r)
                for _ in range(k):
                    c = _rot60(c)
                if reflected:
                    c = _reflect(c)
                return _from_cube(*c)

            elements.append(transform)
    # Reorder so rotations come first (k=0..5, reflected=False), then reflected.
    rotations = [elements[2 * k] for k in range(6)]
    reflections = [elements[2 * k + 1] for k in range(6)]
    return rotations + reflections


_ELEMENTS = _build_elements()


def transform_coord(element: int, q: int, r: int) -> tuple[int, int]:
    """Apply D6 element `element` (0..11) to axial coordinate `(q, r)`."""

    if not 0 <= element < D6_SIZE:
        raise ValueError(f"D6 element must be in [0, {D6_SIZE}), got {element}")
    return _ELEMENTS[element](int(q), int(r))


def inverse_index(element: int) -> int:
    """Return the index of the inverse of `element` (found by composition)."""

    if not 0 <= element < D6_SIZE:
        raise ValueError(f"D6 element must be in [0, {D6_SIZE}), got {element}")
    # Probe a non-symmetric point and find g' with g'(g(p)) == p for all probes.
    probes = [(1, 0), (0, 1), (2, -1), (3, 2)]
    for candidate in range(D6_SIZE):
        if all(
            transform_coord(candidate, *transform_coord(element, q, r)) == (q, r)
            for q, r in probes
        ):
            return candidate
    raise RuntimeError(f"no inverse found for D6 element {element}")


def compose_index(a: int, b: int) -> int:
    """Return the index c with element_c(p) == element_a(element_b(p))."""

    probes = [(1, 0), (0, 1), (2, -1), (3, 2)]
    for candidate in range(D6_SIZE):
        if all(
            transform_coord(candidate, q, r) == transform_coord(a, *transform_coord(b, q, r))
            for q, r in probes
        ):
            return candidate
    raise RuntimeError(f"D6 not closed under composition of {a}, {b}")
