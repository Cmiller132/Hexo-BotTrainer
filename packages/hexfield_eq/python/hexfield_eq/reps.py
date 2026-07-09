"""Generated D6 quotient-permutation representations.

The tables in this module are derived from :func:`geometry.apply_d6`; no
production equivariance table is imported or copied.  Quotient slots are the
canonically ordered left cosets ``G/H`` and actions are left translation.

Phase-A derivation: ``docs/quotient_reps/DERIVATION_QUOTIENT_REPS.md``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Final

import torch

from .constants import DIRECTIONS
from .geometry import apply_d6

GROUP_ORDER: Final = 12
TYPE_ORDER: Final = ("reg", "mirror", "point", "axis", "triv")


def _action_signature(g: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Faithful signature of a lattice action on an axial basis."""

    return apply_d6(g, 1, 0), apply_d6(g, 0, 1)


@functools.lru_cache(maxsize=1)
def build_group() -> dict[str, list]:
    """Generate the production D6 tables from ``apply_d6`` alone.

    The returned fields intentionally match ``equivariant.build_group`` so G1
    can compare them exactly.  Composition uses ``mult[a][b] = a o b``.
    """

    elements = tuple(range(GROUP_ORDER))
    by_signature = {_action_signature(g): g for g in elements}
    if len(by_signature) != GROUP_ORDER:
        raise AssertionError("apply_d6 is not faithful on the axial basis")

    mult: list[list[int]] = []
    for a in elements:
        row: list[int] = []
        for b in elements:
            bq, br = apply_d6(b, 1, 0)
            aq, ar = apply_d6(a, bq, br)
            bq2, br2 = apply_d6(b, 0, 1)
            aq2, ar2 = apply_d6(a, bq2, br2)
            row.append(by_signature[((aq, ar), (aq2, ar2))])
        mult.append(row)

    identities = [
        e
        for e in elements
        if all(mult[e][g] == g and mult[g][e] == g for g in elements)
    ]
    if len(identities) != 1:
        raise AssertionError(f"expected one identity, got {identities}")
    identity = identities[0]

    inv: list[int] = []
    for g in elements:
        candidates = [
            h for h in elements if mult[g][h] == identity and mult[h][g] == identity
        ]
        if len(candidates) != 1:
            raise AssertionError(f"expected one inverse for g={g}, got {candidates}")
        inv.append(candidates[0])

    taps = ((0, 0), *DIRECTIONS)
    tap_index = {offset: i for i, offset in enumerate(taps)}
    tapp = [
        [tap_index[apply_d6(g, *offset)] for offset in taps] for g in elements
    ]
    regp = [[mult[inv[g]][k] for k in elements] for g in elements]

    axis_subgroup = _axis_subgroup()
    cosets = [list(coset) for coset in _left_cosets(axis_subgroup, mult)]
    cos_of: list[int | None] = [None] * GROUP_ORDER
    for coset_index, coset in enumerate(cosets):
        for element in coset:
            cos_of[element] = coset_index
    if any(coset is None for coset in cos_of):
        raise AssertionError("axis cosets do not partition D6")
    cos_of_int = [int(coset) for coset in cos_of]
    cosp = [
        [cos_of_int[mult[g][coset[0]]] for coset in cosets] for g in elements
    ]

    return {
        "mult": mult,
        "inv": inv,
        "tapp": tapp,
        "regp": regp,
        "cosets": cosets,
        "cos_of": cos_of_int,
        "cosp": cosp,
    }


def _orientation(g: int) -> int:
    """Determinant (+1 rotation, -1 reflection) of an axial action."""

    (aq, ar), (bq, br) = _action_signature(g)
    det = aq * br - bq * ar
    if det not in (-1, 1):
        raise AssertionError(f"non-unimodular D6 action g={g}: det={det}")
    return det


def _axis_subgroup() -> tuple[int, ...]:
    """Derive ``K = stab(Q-axis)`` without importing a production table."""

    subgroup = tuple(
        g for g in range(GROUP_ORDER) if apply_d6(g, 1, 0) in ((1, 0), (-1, 0))
    )
    if len(subgroup) != 4:
        raise AssertionError(f"expected order-4 Q-axis stabilizer, got {subgroup}")
    return subgroup


@functools.lru_cache(maxsize=1)
def distinguished_elements() -> tuple[int, int]:
    """Return ``(sigma, rot180)`` derived from lattice actions.

    ``sigma`` is the unique reflection in ``K`` fixing the directed Q vector.
    ``rot180`` is the unique non-identity rotation in ``K`` reversing both
    axial basis vectors.  The second condition intentionally repairs the
    one-vector ambiguity in the Phase-A specification (reflection ``g10`` also
    maps ``(1, 0)`` to ``(-1, 0)``).
    """

    k_group = _axis_subgroup()
    reflections = [g for g in k_group if _orientation(g) == -1]
    sigma_candidates = [g for g in reflections if apply_d6(g, 1, 0) == (1, 0)]
    if len(sigma_candidates) != 1:
        raise AssertionError(f"expected one direction-fixing reflection: {sigma_candidates}")

    rot180_candidates = [
        g
        for g in k_group
        if _orientation(g) == 1
        and apply_d6(g, 1, 0) == (-1, 0)
        and apply_d6(g, 0, 1) == (0, -1)
    ]
    if len(rot180_candidates) != 1:
        raise AssertionError(f"expected one 180-degree rotation: {rot180_candidates}")
    return sigma_candidates[0], rot180_candidates[0]


@functools.lru_cache(maxsize=None)
def subgroup(type_name: str) -> tuple[int, ...]:
    """Subgroup ``H`` defining the named quotient type ``G/H``."""

    if type_name not in TYPE_ORDER:
        raise ValueError(f"unknown quotient type {type_name!r}; expected one of {TYPE_ORDER}")
    sigma, rot180 = distinguished_elements()
    groups = {
        "reg": (0,),
        "mirror": tuple(sorted((0, sigma))),
        "point": tuple(sorted((0, rot180))),
        "axis": _axis_subgroup(),
        "triv": tuple(range(GROUP_ORDER)),
    }
    result = groups[type_name]
    mult = build_group()["mult"]
    if any(mult[a][b] not in result for a in result for b in result):
        raise AssertionError(f"{type_name} elements are not closed: {result}")
    return result


def _left_cosets(
    h_group: Sequence[int], mult: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """Canonical left cosets, internally sorted then ordered by minimum."""

    unique = {
        tuple(sorted(mult[g][h] for h in h_group)) for g in range(GROUP_ORDER)
    }
    cosets = tuple(sorted(unique, key=lambda coset: (min(coset), coset)))
    flat = [element for coset in cosets for element in coset]
    if sorted(flat) != list(range(GROUP_ORDER)):
        raise AssertionError(f"cosets do not partition D6: {cosets}")
    return cosets


@functools.lru_cache(maxsize=None)
def quotient_cosets(type_name: str) -> tuple[tuple[int, ...], ...]:
    """Canonical slots of the quotient permutation representation ``G/H``."""

    return _left_cosets(subgroup(type_name), build_group()["mult"])


def slots(type_name: str) -> int:
    """Number of slots in a named quotient representation."""

    return len(quotient_cosets(type_name))


@functools.lru_cache(maxsize=None)
def rep_action(type_name: str, g: int) -> tuple[int, ...]:
    """Forward slot permutation under left translation by ``g``.

    The tuple maps source slot to destination slot.  Consequently
    ``rep_action(g)[rep_action(h)[s]] == rep_action(g*h)[s]``.
    """

    if not 0 <= g < GROUP_ORDER:
        raise ValueError(f"D6 element out of range: {g}")
    cosets = quotient_cosets(type_name)
    slot_of = {element: i for i, coset in enumerate(cosets) for element in coset}
    mult = build_group()["mult"]
    return tuple(slot_of[mult[g][coset[0]]] for coset in cosets)


@functools.lru_cache(maxsize=None)
def rep_gather(type_name: str, g: int) -> tuple[int, ...]:
    """Gather permutation: output slot -> input slot for ``rho(g)``."""

    inv_g = build_group()["inv"][g]
    return rep_action(type_name, inv_g)


def _canonical_orbit_labels(
    shape: Sequence[int], transforms: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Label finite-set orbits in first-occurrence (row-major) order."""

    size = 1
    for extent in shape:
        size *= extent
    parent = list(range(size))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for transform in transforms:
        if len(transform) != size:
            raise ValueError("orbit transform has the wrong cardinality")
        for source, destination in enumerate(transform):
            union(source, destination)

    label_of_root: dict[int, int] = {}
    labels: list[int] = []
    for flat_index in range(size):
        root = find(flat_index)
        if root not in label_of_root:
            label_of_root[root] = len(label_of_root)
        labels.append(label_of_root[root])
    return torch.tensor(labels, dtype=torch.long).reshape(tuple(shape))


@functools.lru_cache(maxsize=None)
def linear_basis_index(in_type: str, out_type: str) -> torch.Tensor:
    """Orbit-basis label for every ``(out_slot, in_slot)`` matrix entry.

    Entries carrying the same integer label share one free coefficient.  Labels
    are assigned by the first row-major entry in each diagonal-action orbit.
    """

    nout, nin = slots(out_type), slots(in_type)
    transforms: list[list[int]] = []
    for g in range(GROUP_ORDER):
        out_action = rep_action(out_type, g)
        in_action = rep_action(in_type, g)
        transforms.append(
            [out_action[a] * nin + in_action[b] for a in range(nout) for b in range(nin)]
        )
    return _canonical_orbit_labels((nout, nin), transforms)


@functools.lru_cache(maxsize=None)
def conv_basis_index(in_type: str, out_type: str) -> torch.Tensor:
    """Orbit-basis labels on ``(tap, out_slot, in_slot)`` triples."""

    ntaps, nout, nin = 7, slots(out_type), slots(in_type)
    tapp = build_group()["tapp"]
    transforms: list[list[int]] = []
    for g in range(GROUP_ORDER):
        out_action = rep_action(out_type, g)
        in_action = rep_action(in_type, g)
        transforms.append(
            [
                (tapp[g][tap] * nout + out_action[a]) * nin + in_action[b]
                for tap in range(ntaps)
                for a in range(nout)
                for b in range(nin)
            ]
        )
    return _canonical_orbit_labels((ntaps, nout, nin), transforms)


def orbit_dimension(in_type: str, out_type: str, *, conv: bool = False) -> int:
    """Dimension from diagonal-action orbit count (derivation sections 2–3)."""

    labels = conv_basis_index(in_type, out_type) if conv else linear_basis_index(
        in_type, out_type
    )
    return int(labels.max().item()) + 1


@functools.lru_cache(maxsize=None)
def double_cosets(in_type: str, out_type: str) -> tuple[tuple[int, ...], ...]:
    """Enumerate ``H_out \\ G / H_in`` directly in canonical order."""

    h_in, h_out = subgroup(in_type), subgroup(out_type)
    mult = build_group()["mult"]
    unique = {
        tuple(
            sorted(
                {
                    mult[mult[left][g]][right]
                    for left in h_out
                    for right in h_in
                }
            )
        )
        for g in range(GROUP_ORDER)
    }
    result = tuple(sorted(unique, key=lambda coset: (min(coset), coset)))
    if sorted({element for coset in result for element in coset}) != list(
        range(GROUP_ORDER)
    ):
        raise AssertionError(f"double cosets do not cover D6: {result}")
    for a, left in enumerate(result):
        for right in result[a + 1 :]:
            if set(left).intersection(right):
                raise AssertionError(f"overlapping double cosets: {left}, {right}")
    return result


def reynolds_projector(
    in_type: str,
    out_type: str,
    *,
    conv: bool = False,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Dense Reynolds projector on linear or conv matrix-entry space.

    This construction averages explicit permutation matrices and is independent
    of the orbit-label and double-coset algorithms used by the other G2 proofs.
    """

    ntaps = 7 if conv else 1
    nout, nin = slots(out_type), slots(in_type)
    dimension = ntaps * nout * nin
    projector = torch.zeros((dimension, dimension), dtype=dtype)
    tapp = build_group()["tapp"]
    for g in range(GROUP_ORDER):
        out_action = rep_action(out_type, g)
        in_action = rep_action(in_type, g)
        for tap in range(ntaps):
            tap_g = tapp[g][tap] if conv else tap
            for out_slot in range(nout):
                for in_slot in range(nin):
                    source = (tap * nout + out_slot) * nin + in_slot
                    destination = (
                        (tap_g * nout + out_action[out_slot]) * nin
                        + in_action[in_slot]
                    )
                    projector[destination, source] += 1.0 / GROUP_ORDER
    return projector
