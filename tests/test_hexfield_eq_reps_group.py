"""Phase-A G1: generated D6 foundation and quotient-action parity."""

from __future__ import annotations

import os

os.environ["HEXFIELD_EQ_GROUP_ORDER"] = "12"

from hexfield_eq import equivariant as production  # noqa: E402
from hexfield_eq.geometry import apply_d6  # noqa: E402
from hexfield_eq.reps import (  # noqa: E402
    GROUP_ORDER,
    TYPE_ORDER,
    build_group,
    distinguished_elements,
    quotient_cosets,
    rep_action,
    rep_gather,
    subgroup,
)


def test_generated_group_tables_exactly_match_production() -> None:
    """Every production field is regenerated exactly from apply_d6."""

    generated = build_group()
    expected = production.build_group()
    assert generated.keys() == expected.keys()
    for field in expected:
        assert generated[field] == expected[field], field


def test_distinguished_subgroups_are_derived_from_geometry() -> None:
    """Sigma and rot180 satisfy the corrected, unambiguous definitions."""

    sigma, rot180 = distinguished_elements()
    assert sigma == 7
    assert rot180 == 3
    assert subgroup("mirror") == (0, 7)
    assert subgroup("point") == (0, 3)
    assert subgroup("axis") == (0, 3, 7, 10)
    assert apply_d6(sigma, 1, 0) == (1, 0)
    assert apply_d6(rot180, 1, 0) == (-1, 0)
    assert apply_d6(rot180, 0, 1) == (0, -1)

    # The spec's one-vector rot180 criterion is ambiguous: g10 also reverses Q.
    assert [
        g for g in subgroup("axis") if apply_d6(g, 1, 0) == (-1, 0)
    ] == [3, 10]


def test_canonical_coset_order() -> None:
    """Slots are sorted left cosets, ordered by their minimum element."""

    for type_name in TYPE_ORDER:
        cosets = quotient_cosets(type_name)
        assert all(tuple(sorted(coset)) == coset for coset in cosets)
        assert tuple(sorted(cosets, key=lambda c: (min(c), c))) == cosets
        assert sorted(x for coset in cosets for x in coset) == list(range(GROUP_ORDER))


def test_all_quotient_actions_are_permutation_homomorphisms() -> None:
    """Five types x all 144 products obey rho(g)rho(h)=rho(gh)."""

    mult = build_group()["mult"]
    inv = build_group()["inv"]
    for type_name in TYPE_ORDER:
        nslots = len(quotient_cosets(type_name))
        for g in range(GROUP_ORDER):
            action_g = rep_action(type_name, g)
            assert sorted(action_g) == list(range(nslots))
            assert rep_gather(type_name, g) == rep_action(type_name, inv[g])
            for h in range(GROUP_ORDER):
                action_h = rep_action(type_name, h)
                composed = tuple(action_g[action_h[s]] for s in range(nslots))
                assert composed == rep_action(type_name, mult[g][h]), (type_name, g, h)

