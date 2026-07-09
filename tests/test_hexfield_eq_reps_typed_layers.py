"""Phase-A G4/G5: typed layer properties and nonlinearity legality."""

from __future__ import annotations

import os
import random

os.environ["HEXFIELD_EQ_GROUP_ORDER"] = "12"
os.environ["HEXFIELD_EQ_FEATURE_VERSION"] = "1"
os.environ["HEXFIELD_EQ_SUPPORT_RADIUS"] = "1"

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from hexfield_eq.geometry import apply_d6  # noqa: E402
from hexfield_eq.reps import (  # noqa: E402
    GROUP_ORDER,
    TYPE_ORDER,
    TypedConv,
    TypedGroupAffineNorm,
    TypedLayerScale,
    TypedLinear,
    build_group,
    canonical_signature,
    expand_per_instance,
    signature_action,
    transform_channels,
    typed_group_pool,
)
from hexfield_eq.support import build_support  # noqa: E402


def _random_signature(rng: random.Random) -> tuple[tuple[str, int], ...]:
    while True:
        values = [rng.randint(0, 4) for _ in TYPE_ORDER]
        if any(values):
            return canonical_signature(tuple(zip(TYPE_ORDER, values, strict=True)))


def _fifty_signature_pairs() -> list[
    tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]
]:
    rng = random.Random(0)
    return [(_random_signature(rng), _random_signature(rng)) for _ in range(50)]


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-10
) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _support_pair(stones: list[tuple[int, int]], g: int):
    base = build_support(stones)
    transformed_stones = [apply_d6(g, q, r) for q, r in stones]
    transformed = build_support(transformed_stones)
    permutation = torch.tensor(
        [
            transformed.index[apply_d6(g, int(q), int(r))]
            for q, r in base.coords.tolist()
        ],
        dtype=torch.long,
    )
    assert base.num_nodes == transformed.num_nodes
    return transformed, permutation


def test_fifty_random_signatures_typed_linear_equivariance() -> None:
    """Generated dense weights and fp64 forwards intertwine all 12 actions."""

    torch.manual_seed(0)
    for in_signature, out_signature in _fifty_signature_pairs():
        layer = TypedLinear(in_signature, out_signature, dtype=torch.float64)
        with torch.no_grad():
            layer.bias_base.copy_(torch.randn_like(layer.bias_base))
        weight = layer.materialize_weight()
        x = torch.randn(3, weight.shape[1], dtype=torch.float64)
        base = layer(x)
        for g in range(GROUP_ORDER):
            in_action = torch.tensor(signature_action(in_signature, g))
            out_action = torch.tensor(signature_action(out_signature, g))
            torch.testing.assert_close(
                weight.index_select(0, out_action).index_select(1, in_action),
                weight,
                atol=0,
                rtol=0,
            )
            # The forward comparison below exercises its expansion; this
            # explicit check freezes exact invariance of the randomized bias.
            expanded_bias = expand_per_instance(layer.bias_base, out_signature)
            torch.testing.assert_close(
                expanded_bias.index_select(0, out_action),
                expanded_bias,
                atol=0,
                rtol=0,
            )
            _assert_close(
                layer(transform_channels(x, in_signature, g)),
                transform_channels(base, out_signature, g),
            )


def test_fifty_random_signatures_typed_conv_algebra_and_support() -> None:
    """Conv ties hold algebraically and on rebuilt transformed support graphs."""

    torch.manual_seed(0)
    stones = [(0, 0), (2, -1)]
    group = build_group()
    base_support = build_support(stones)
    base_nbr = torch.from_numpy(base_support.nbr.astype("int64")).unsqueeze(0)
    for in_signature, out_signature in _fifty_signature_pairs():
        layer = TypedConv(in_signature, out_signature, dtype=torch.float64)
        with torch.no_grad():
            layer.bias_base.copy_(torch.randn_like(layer.bias_base))
        weight = layer.materialize_weight()
        x = torch.randn(1, base_support.num_nodes, weight.shape[1], dtype=torch.float64)
        base_output = layer(x, base_nbr)
        for g in range(GROUP_ORDER):
            in_action = torch.tensor(signature_action(in_signature, g))
            out_action = torch.tensor(signature_action(out_signature, g))
            taps = torch.tensor(group["tapp"][g])
            torch.testing.assert_close(
                weight.index_select(0, taps)
                .index_select(1, in_action)
                .index_select(2, out_action),
                weight,
                atol=0,
                rtol=0,
            )
            expanded_bias = expand_per_instance(layer.bias_base, out_signature)
            torch.testing.assert_close(
                expanded_bias.index_select(0, out_action),
                expanded_bias,
                atol=0,
                rtol=0,
            )

            transformed_support, node_permutation = _support_pair(stones, g)
            x_g = torch.empty_like(x)
            x_g[:, node_permutation] = transform_channels(x, in_signature, g)
            nbr_g = torch.from_numpy(transformed_support.nbr.astype("int64")).unsqueeze(0)
            output_g = layer(x_g, nbr_g)
            _assert_close(
                output_g[:, node_permutation],
                transform_channels(base_output, out_signature, g),
            )


def test_fifty_random_signatures_norm_scale_pool_and_pointwise() -> None:
    """Typed affine operations, invariant reads, ReLU, and GELU all commute."""

    torch.manual_seed(0)
    signatures = [pair[0] for pair in _fifty_signature_pairs()]
    for signature in signatures:
        norm = TypedGroupAffineNorm(signature, dtype=torch.float64)
        scale = TypedLayerScale(signature, dtype=torch.float64)
        with torch.no_grad():
            norm.gamma.copy_(torch.randn_like(norm.gamma))
            norm.beta.copy_(torch.randn_like(norm.beta))
            scale.gamma.copy_(torch.randn_like(scale.gamma))
        width = norm.weight.numel()
        x = torch.randn(2, 3, width, dtype=torch.float64)
        base_norm = norm(x)
        base_scale = scale(x)
        base_pool = typed_group_pool(x, signature)
        for g in range(GROUP_ORDER):
            x_g = transform_channels(x, signature, g)
            _assert_close(
                norm(x_g), transform_channels(base_norm, signature, g), atol=1e-12
            )
            torch.testing.assert_close(
                scale(x_g), transform_channels(base_scale, signature, g), atol=0, rtol=0
            )
            _assert_close(typed_group_pool(x_g, signature), base_pool, atol=1e-12)
            torch.testing.assert_close(
                F.relu(x_g), transform_channels(F.relu(x), signature, g), atol=0, rtol=0
            )
            # PyTorch's vectorized CPU GELU can differ by one fp64 ulp after a
            # channel permutation even though the operation is pointwise.
            _assert_close(
                F.gelu(x_g), transform_channels(F.gelu(x), signature, g), atol=1e-12
            )


def test_gelu_commutes_with_every_quotient_permutation() -> None:
    """G5 positive control: every required single-type action is legal."""

    torch.manual_seed(0)
    for type_name in TYPE_ORDER:
        signature = ((type_name, 1),)
        x = torch.randn(5, len(signature_action(signature, 0)), dtype=torch.float64)
        for g in range(GROUP_ORDER):
            _assert_close(
                F.gelu(transform_channels(x, signature, g)),
                transform_channels(F.gelu(x), signature, g),
                atol=1e-12,
            )


def test_sign_rep_is_a_representation_but_gelu_breaks_it() -> None:
    """G5 negative control: reflection signs do not commute with GELU."""

    def orientation(g: int) -> int:
        aq, ar = apply_d6(g, 1, 0)
        bq, br = apply_d6(g, 0, 1)
        return aq * br - ar * bq

    sign = [orientation(g) for g in range(GROUP_ORDER)]
    mult = build_group()["mult"]
    for g in range(GROUP_ORDER):
        for h in range(GROUP_ORDER):
            assert sign[g] * sign[h] == sign[mult[g][h]]

    x = torch.tensor([-2.0, -0.75, 0.25, 1.5], dtype=torch.float64)
    reflection_violations = []
    for g in range(GROUP_ORDER):
        lhs = F.gelu(sign[g] * x)
        rhs = sign[g] * F.gelu(x)
        violation = float((lhs - rhs).abs().max())
        if sign[g] == 1:
            assert violation == 0.0
        else:
            reflection_violations.append(violation)
    assert len(reflection_violations) == 6
    assert min(reflection_violations) > 1.0
