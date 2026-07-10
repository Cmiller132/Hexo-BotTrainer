"""Phase-A G3: exact specialization to the production regular machinery."""

from __future__ import annotations

import os

os.environ["HEXFIELD_EQ_GROUP_ORDER"] = "12"
os.environ["HEXFIELD_EQ_FEATURE_VERSION"] = "1"
os.environ["HEXFIELD_EQ_SUPPORT_RADIUS"] = "1"

import torch  # noqa: E402

from hexfield_eq import constants as C  # noqa: E402
from hexfield_eq import equivariant as production  # noqa: E402
from hexfield_eq.reps import (  # noqa: E402
    input_rep_matrix,
    production_conv_coefficients,
    production_linear_coefficients,
    signature_action,
    typed_conv_weight,
    typed_linear_weight,
    typed_stem_weight,
)


def test_regular_signature_layout_is_production_slot_major() -> None:
    """The resolved layout discrepancy is frozen by raw channel-action parity."""

    multiplicity = 3
    group = production.build_group()
    for g in range(12):
        expected = tuple(
            group["mult"][g][slot] * multiplicity + instance
            for slot in range(12)
            for instance in range(multiplicity)
        )
        assert signature_action((('reg', multiplicity),), g) == expected


def test_typed_linear_exactly_reproduces_production() -> None:
    torch.manual_seed(0)
    wb = torch.randn(12, 3, 2, dtype=torch.float64)
    coefficients = production_linear_coefficients(wb)
    typed = typed_linear_weight(
        {("reg", "reg"): coefficients},
        (("reg", 2),),
        (("reg", 3),),
    )
    expected = production.gen_linear_weight(wb, production.linear_gather_index())
    torch.testing.assert_close(typed, expected, atol=0, rtol=0)


def test_typed_conv_exactly_reproduces_production() -> None:
    torch.manual_seed(0)
    w_base = torch.randn(7, 12, 3, 2, dtype=torch.float64)
    coefficients = production_conv_coefficients(w_base)
    typed = typed_conv_weight(
        {("reg", "reg"): coefficients},
        (("reg", 2),),
        (("reg", 3),),
    )
    expected = production.gen_conv_weight(w_base, production.conv_gather_index())
    torch.testing.assert_close(typed, expected, atol=0, rtol=0)


def test_input_rep_and_typed_stem_reproduce_production() -> None:
    torch.manual_seed(0)
    assert C.NUM_FEATURES == 25
    for g in range(12):
        torch.testing.assert_close(
            input_rep_matrix(g), production._in_rep_matrix()[g].to(torch.float64),
            atol=0, rtol=0,
        )
    w0 = torch.randn(7, C.CHANNELS, C.NUM_FEATURES, dtype=torch.float64)
    typed = typed_stem_weight(w0, (("reg", C.C_ORBIT),))
    expected = production.gen_stem_weight(w0)
    torch.testing.assert_close(typed, expected, atol=1e-12, rtol=0)
