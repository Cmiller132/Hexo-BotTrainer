"""Phase-A G2: three independent Hom-space dimension calculations."""

from __future__ import annotations

import os

os.environ["HEXFIELD_EQ_GROUP_ORDER"] = "12"

import torch  # noqa: E402

from hexfield_eq.reps import (  # noqa: E402
    TYPE_ORDER,
    conv_basis_index,
    double_cosets,
    linear_basis_index,
    orbit_dimension,
    reynolds_projector,
    slots,
)


# Rows are input types and columns are output types in TYPE_ORDER.
LINEAR_DIMS = (
    (12, 6, 6, 3, 1),
    (6, 4, 3, 2, 1),
    (6, 3, 6, 3, 1),
    (3, 2, 3, 2, 1),
    (1, 1, 1, 1, 1),
)
CONV_DIMS = (
    (84, 42, 42, 21, 7),
    (42, 24, 21, 12, 5),
    (42, 21, 24, 12, 4),
    (21, 12, 12, 7, 3),
    (7, 5, 4, 3, 2),
)


def _projector_rank(projector: torch.Tensor) -> int:
    singular_values = torch.linalg.svdvals(projector)
    return int((singular_values > 1e-9).sum().item())


def _assert_orthogonal_projector(projector: torch.Tensor) -> None:
    torch.testing.assert_close(projector, projector.T, atol=1e-12, rtol=0)
    torch.testing.assert_close(projector @ projector, projector, atol=1e-12, rtol=0)


def test_linear_dimensions_agree_three_independent_ways() -> None:
    """Pair orbits, double cosets, and Reynolds ranks agree for all 25 pairs."""

    measured: list[list[int]] = []
    for in_type in TYPE_ORDER:
        row: list[int] = []
        for out_type in TYPE_ORDER:
            labels = linear_basis_index(in_type, out_type)
            by_orbits = int(labels.max().item()) + 1
            by_double_cosets = len(double_cosets(in_type, out_type))
            projector = reynolds_projector(in_type, out_type)
            _assert_orthogonal_projector(projector)
            by_rank = _projector_rank(projector)
            assert by_orbits == by_double_cosets == by_rank
            row.append(by_orbits)
        measured.append(row)
    assert tuple(tuple(row) for row in measured) == LINEAR_DIMS


def test_linear_dimension_anchors() -> None:
    assert orbit_dimension("reg", "reg") == 12
    for type_name in TYPE_ORDER:
        assert orbit_dimension("reg", type_name) == slots(type_name)
        assert orbit_dimension(type_name, "reg") == slots(type_name)
    assert orbit_dimension("triv", "triv") == 1


def test_conv_dimensions_agree_between_orbits_and_reynolds_rank() -> None:
    """The tap-extended basis and an independent projector agree on all pairs."""

    measured: list[list[int]] = []
    for in_type in TYPE_ORDER:
        row: list[int] = []
        for out_type in TYPE_ORDER:
            labels = conv_basis_index(in_type, out_type)
            by_orbits = int(labels.max().item()) + 1
            projector = reynolds_projector(in_type, out_type, conv=True)
            _assert_orthogonal_projector(projector)
            by_rank = _projector_rank(projector)
            assert by_orbits == by_rank
            row.append(by_orbits)
        measured.append(row)
    assert tuple(tuple(row) for row in measured) == CONV_DIMS
    assert orbit_dimension("reg", "reg", conv=True) == 84


def test_basis_labels_are_dense_and_deterministic() -> None:
    """Every basis label occurs and canonical first occurrences are increasing."""

    for in_type in TYPE_ORDER:
        for out_type in TYPE_ORDER:
            for labels in (
                linear_basis_index(in_type, out_type),
                conv_basis_index(in_type, out_type),
            ):
                flat = labels.flatten().tolist()
                assert sorted(set(flat)) == list(range(max(flat) + 1))
                first = [flat.index(label) for label in range(max(flat) + 1)]
                assert first == sorted(first)
