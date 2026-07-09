#!/usr/bin/env python3
"""Closed-form Phase-A cost model for mixed D6 quotient signatures.

This script is intentionally standard-library-only and CPU-only.  It models
the Phase-B boundary design from ``docs/quotient_reps``:

* the residual stream has an arbitrary mixed permutation signature;
* A/L/register attention internals are a regular representation of width
  ``W = 12 * K_attn``;
* typed convolutions and MLPs still materialize dense weights, so their serve
  FLOPs depend on dense widths, while their free parameter counts depend on
  the quotient Hom-space dimensions.

FLOPs count a multiply and an add separately (two FLOPs per MAC).  Activation
bytes are a documented logical matmul-operand traffic proxy: activation inputs
are read and outputs are written, while weights, masks, indices, bias adds,
normalization, and pointwise operations are excluded.  Fused A attention does
not materialize its score matrix.  Ray attention is reported both with ideal
K/V cache reuse and with all gathered K/V operands counted logically.

The default invocation emits the G8 Markdown evidence at the live shape.  Use
``--format json`` for machine-readable output and ``--self-test`` for the
closed-form regression anchors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence


TYPE_ORDER = ("reg", "mirror", "point", "axis", "triv")
TYPE_SLOTS = {
    "reg": 12,
    "mirror": 6,
    "point": 6,
    "axis": 3,
    "triv": 1,
}

# Rows are output type and columns are input type, both in TYPE_ORDER.  These
# are the Phase-A G2 orbit/double-coset/projector dimensions.
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

# The input is 13 trivial planes plus four axis modules.  For one output-type
# instance the stem's equivariant dimension is therefore
# 13*CONV_DIMS[out,triv] + 4*CONV_DIMS[out,axis].
STEM_EFFECTIVE_DIMS = (175, 113, 100, 67, 38)

NUM_FEATURES = 25
NUM_TOKENS = 6
VALUE_BINS = 65
ATTENTION_HEADS = 3
RAY_HEADS = 6
RAY_KEYS_MAX = 31  # self plus 6 directions * RAY_REACH(5)
JOINT_BIAS_CLASSES = 81
FAST_HEAD_DIMS = frozenset((16, 32, 64, 128))
DEFAULT_LAYOUT = "CCLACCLACLA"
DEFAULT_BATCH = 96
DEFAULT_NPAD = 250
DEFAULT_ALPHA = 0.67
COST_CONSISTENT_ALPHA = 4.0 / 7.0
G7_MIRROR_DEPTH_MEAN = 0.9673


def _matrix_quadratic(matrix: Sequence[Sequence[int]], values: Sequence[int]) -> int:
    return sum(
        matrix[out_i][in_i] * values[out_i] * values[in_i]
        for out_i in range(len(values))
        for in_i in range(len(values))
    )


@dataclass(frozen=True, order=True)
class Signature:
    """Canonical multiplicities in ``reg,mirror,point,axis,triv`` order."""

    multiplicities: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.multiplicities) != len(TYPE_ORDER):
            raise ValueError("a signature must contain exactly five multiplicities")
        if any(value < 0 for value in self.multiplicities):
            raise ValueError("signature multiplicities must be non-negative")
        if not any(self.multiplicities):
            raise ValueError("a signature must have at least one nonzero multiplicity")

    @classmethod
    def parse(cls, text: str) -> "Signature":
        values = {name: 0 for name in TYPE_ORDER}
        seen: set[str] = set()
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            raise ValueError("empty signature")
        for part in parts:
            if ":" not in part:
                raise ValueError(f"signature item {part!r} must be TYPE:MULTIPLICITY")
            name, raw_value = (piece.strip() for piece in part.split(":", 1))
            if name not in values:
                raise ValueError(
                    f"unknown quotient type {name!r}; expected one of {TYPE_ORDER}"
                )
            if name in seen:
                raise ValueError(f"duplicate quotient type {name!r}")
            seen.add(name)
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"multiplicity for {name!r} is not an integer: {raw_value!r}"
                ) from exc
            if value < 0:
                raise ValueError(f"multiplicity for {name!r} must be non-negative")
            values[name] = value
        return cls(tuple(values[name] for name in TYPE_ORDER))  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, values: Mapping[str, int]) -> "Signature":
        unknown = set(values) - set(TYPE_ORDER)
        if unknown:
            raise ValueError(f"unknown quotient types: {sorted(unknown)}")
        return cls(tuple(int(values.get(name, 0)) for name in TYPE_ORDER))  # type: ignore[arg-type]

    @property
    def width(self) -> int:
        return sum(
            TYPE_SLOTS[name] * value
            for name, value in zip(TYPE_ORDER, self.multiplicities)
        )

    @property
    def instances(self) -> int:
        return sum(self.multiplicities)

    @property
    def linear_hom(self) -> int:
        return _matrix_quadratic(LINEAR_DIMS, self.multiplicities)

    @property
    def conv_hom(self) -> int:
        return _matrix_quadratic(CONV_DIMS, self.multiplicities)

    @property
    def stem_effective_weights(self) -> int:
        return sum(
            dim * value
            for dim, value in zip(STEM_EFFECTIVE_DIMS, self.multiplicities)
        )

    def multiplicity(self, name: str) -> int:
        return self.multiplicities[TYPE_ORDER.index(name)]

    def canonical(self) -> str:
        return ",".join(
            f"{name}:{value}"
            for name, value in zip(TYPE_ORDER, self.multiplicities)
            if value
        )


@dataclass(frozen=True)
class ModelShape:
    """Shape and accounting controls for one closed-form evaluation."""

    batch: int = DEFAULT_BATCH
    npad: int = DEFAULT_NPAD
    layout: str = DEFAULT_LAYOUT
    k_attn: int = 16
    ray_keys: int = RAY_KEYS_MAX
    activation_bytes: int = 2
    fp32_bytes: int = 4
    reg_lane: bool = True
    reg_token_read: bool = False
    request_moves_left: bool = False

    def __post_init__(self) -> None:
        if self.batch <= 0 or self.npad <= 0:
            raise ValueError("batch and npad must be positive")
        if not self.layout or set(self.layout) - {"C", "A", "L"}:
            raise ValueError("layout must be a non-empty string containing only C/A/L")
        if not self.layout.endswith("A"):
            raise ValueError("layout must end in A, matching HexfieldNet")
        if self.k_attn <= 0:
            raise ValueError("K_attn must be positive")
        if not (1 <= self.ray_keys <= RAY_KEYS_MAX):
            raise ValueError(f"ray_keys must be in [1,{RAY_KEYS_MAX}]")
        if self.activation_bytes <= 0 or self.fp32_bytes <= 0:
            raise ValueError("activation byte widths must be positive")
        if self.reg_token_read and not self.reg_lane:
            raise ValueError("reg_token_read requires reg_lane")

    @property
    def conv_blocks(self) -> int:
        return self.layout.count("C")

    @property
    def attn_blocks(self) -> int:
        return self.layout.count("A")

    @property
    def ray_blocks(self) -> int:
        return self.layout.count("L")

    @property
    def refreshes(self) -> int:
        return (self.conv_blocks + self.ray_blocks) if self.reg_lane else 0


@dataclass(frozen=True)
class Evaluation:
    signature: Signature
    shape: ModelShape
    params_effective: int
    params_stored: int
    macs: Mapping[str, int]
    bytes_ideal: Mapping[str, int]
    bytes_logical: Mapping[str, int]

    @property
    def width(self) -> int:
        return self.signature.width

    @property
    def instances(self) -> int:
        return self.signature.instances

    @property
    def total_macs(self) -> int:
        return sum(self.macs.values())

    @property
    def total_flops(self) -> int:
        return 2 * self.total_macs

    @property
    def total_bytes_ideal(self) -> int:
        return sum(self.bytes_ideal.values())

    @property
    def total_bytes_logical(self) -> int:
        return sum(self.bytes_logical.values())

    @property
    def aligned_16(self) -> bool:
        return self.width % 16 == 0

    @property
    def head_dim_a(self) -> int:
        return 4 * self.shape.k_attn

    @property
    def head_dim_l(self) -> int:
        return 2 * self.shape.k_attn

    @property
    def legal_a(self) -> bool:
        return not self.shape.attn_blocks or self.head_dim_a in FAST_HEAD_DIMS

    @property
    def legal_l(self) -> bool:
        if not self.shape.ray_blocks:
            return True
        return self.shape.k_attn % 2 == 0 and self.head_dim_l in FAST_HEAD_DIMS

    @property
    def deploy_legal(self) -> bool:
        return self.aligned_16 and self.legal_a and self.legal_l


def parameter_counts(signature: Signature, shape: ModelShape) -> tuple[int, int]:
    """Return ``(effective equivariant DOF, estimated stored parameters)``.

    The stored estimate assumes Phase B retains the current dense Reynolds
    source ``w0`` for the stem (``7*25*C`` scalars), while other typed weights
    use basis coefficients.  The effective count replaces that redundant stem
    source by the actual equivariant stem dimension.
    """

    c = signature.width
    i = signature.instances
    h = signature.linear_hom
    v = signature.conv_hom
    k = shape.k_attn

    stem = signature.stem_effective_weights + 3 * i  # conv bias + stem norm
    conv_blocks = shape.conv_blocks * (2 * v + 7 * i)
    attn_blocks = shape.attn_blocks * (
        4 * k * c + 4 * h + 3 * k + 10 * i + JOINT_BIAS_CLASSES
    )
    ray_blocks = shape.ray_blocks * (
        4 * k * c + 4 * h + 3 * k + 10 * i + 2 * JOINT_BIAS_CLASSES
    )
    refreshes = shape.refreshes * (4 * k * c + 3 * k + 5 * i + NUM_TOKENS + 1)
    token_reads = (
        NUM_TOKENS * (h + i) if shape.reg_lane and shape.reg_token_read else 0
    )
    tokens_and_final_norm = (NUM_TOKENS + 2) * i

    # All heads in the state dict, including train-only opp/soft/cell-Q/STV.
    heads = 4 * v + 12 * h + 256 * i * i + 1464 * i + 393

    effective = (
        stem
        + conv_blocks
        + attn_blocks
        + ray_blocks
        + refreshes
        + token_reads
        + tokens_and_final_norm
        + heads
    )
    stem_nullspace = 7 * NUM_FEATURES * c - signature.stem_effective_weights
    stored = effective + stem_nullspace
    return effective, stored


def _mac_breakdown(signature: Signature, shape: ModelShape) -> dict[str, int]:
    b = shape.batch
    n = shape.npad
    t = NUM_TOKENS
    c = signature.width
    i = signature.instances
    w = 12 * shape.k_attn
    x = b * n
    joint = n + t

    result = {
        "stem": 7 * NUM_FEATURES * x * c,
        "conv_blocks": shape.conv_blocks * 14 * x * c * c,
        "attn_blocks": shape.attn_blocks
        * (b * joint * (4 * c * w + 4 * c * c) + 2 * b * joint * joint * w),
        "ray_blocks": shape.ray_blocks
        * (x * (4 * c * w + 4 * c * c) + 2 * x * shape.ray_keys * w),
        "register_refresh": shape.refreshes
        * (2 * b * (n + t) * c * w + 2 * b * t * n * w),
        "token_read": (
            b * t * c * c if shape.reg_lane and shape.reg_token_read else 0
        ),
        "policy": x * (9 * c * c + 2 * i),
        "value": b
        * (
            4 * (t + 2) * c * c
            + 16 * (t + 2) * i * i
            + 4 * i * VALUE_BINS
        ),
    }
    if shape.request_moves_left:
        result["moves_left"] = b * (
            16 * c * c + 64 * i * i + 4 * i * VALUE_BINS
        )
    return result


def _byte_breakdown(
    signature: Signature, shape: ModelShape, *, ray_kv_reads: int
) -> dict[str, int]:
    """Logical matmul activation traffic under the documented byte model."""

    b = shape.batch
    n = shape.npad
    t = NUM_TOKENS
    c = signature.width
    i = signature.instances
    w = 12 * shape.k_attn
    x = b * n
    joint = n + t
    elem = shape.activation_bytes
    fp32 = shape.fp32_bytes

    # Register score q@k uses the trunk compute dtype.  sigmoid gates, v.float,
    # and gates@v use fp32 in register.py.  Conversion/pointwise traffic itself
    # is outside this matmul-operand definition.
    register_projection = elem * 2 * b * (n + t) * (c + w)
    register_score = elem * b * (t * w + n * w + ATTENTION_HEADS * t * n)
    register_update = fp32 * b * (
        ATTENTION_HEADS * t * n + n * w + t * w
    )

    result = {
        "stem": elem * x * (7 * NUM_FEATURES + c),
        "conv_blocks": elem * shape.conv_blocks * 16 * x * c,
        # q/k/v/out boundaries + typed MLP + fused Q/K/V/O attention traffic.
        "attn_blocks": elem
        * shape.attn_blocks
        * b
        * joint
        * (10 * c + 8 * w),
        # The ray kernel reads q and writes out once; K and V are read
        # ray_kv_reads times.  q/k/v/out projection and MLP traffic contributes
        # the leading 10C+4W, hence total W coefficient 6+2*ray_kv_reads.
        "ray_blocks": elem
        * shape.ray_blocks
        * x
        * (10 * c + (6 + 2 * ray_kv_reads) * w),
        "register_refresh": shape.refreshes
        * (register_projection + register_score + register_update),
        "token_read": (
            elem * 2 * b * t * c
            if shape.reg_lane and shape.reg_token_read
            else 0
        ),
        "policy": elem * x * (11 * c + 2 * i + 1),
        # inv_read remains in trunk dtype; value reduction/top are explicitly
        # kept fp32 by the serve evaluator.
        "value": elem * b * (t + 2) * 5 * c
        + fp32 * b * (4 * i * (t + 3) + 4 * i + VALUE_BINS),
    }
    if shape.request_moves_left:
        result["moves_left"] = (
            elem * b * 20 * c + fp32 * b * (24 * i + VALUE_BINS)
        )
    return result


def evaluate(signature: Signature, shape: ModelShape) -> Evaluation:
    params_effective, params_stored = parameter_counts(signature, shape)
    return Evaluation(
        signature=signature,
        shape=shape,
        params_effective=params_effective,
        params_stored=params_stored,
        macs=_mac_breakdown(signature, shape),
        bytes_ideal=_byte_breakdown(signature, shape, ray_kv_reads=1),
        bytes_logical=_byte_breakdown(
            signature, shape, ray_kv_reads=shape.ray_keys
        ),
    )


def _speedup(flops_ratio: float, bytes_ratio: float, alpha: float) -> float:
    return 1.0 / (alpha * flops_ratio + (1.0 - alpha) * bytes_ratio)


def _ratios(
    evaluation: Evaluation, reference: Evaluation, alpha: float
) -> dict[str, float]:
    flops_ratio = evaluation.total_flops / reference.total_flops
    bytes_ideal_ratio = (
        evaluation.total_bytes_ideal / reference.total_bytes_ideal
    )
    bytes_logical_ratio = (
        evaluation.total_bytes_logical / reference.total_bytes_logical
    )
    return {
        "flops_ratio": flops_ratio,
        "bytes_ideal_ratio": bytes_ideal_ratio,
        "bytes_logical_ratio": bytes_logical_ratio,
        "speedup_ideal": _speedup(flops_ratio, bytes_ideal_ratio, alpha),
        "speedup_logical": _speedup(flops_ratio, bytes_logical_ratio, alpha),
    }


def _shape_with_k(base: ModelShape, k_attn: int) -> ModelShape:
    return ModelShape(
        batch=base.batch,
        npad=base.npad,
        layout=base.layout,
        k_attn=k_attn,
        ray_keys=base.ray_keys,
        activation_bytes=base.activation_bytes,
        fp32_bytes=base.fp32_bytes,
        reg_lane=base.reg_lane,
        reg_token_read=base.reg_token_read,
        request_moves_left=base.request_moves_left,
    )


def _built_in_candidates(base_shape: ModelShape) -> list[Evaluation]:
    """Required G8 sweep plus aligned controls and point/extreme variants."""

    rows: tuple[tuple[str, int], ...] = (
        ("reg:16", 16),
        ("reg:8,mirror:16", 16),
        ("reg:8,mirror:16", 8),
        ("reg:8,mirror:8,axis:8,triv:8", 16),
        ("reg:8,mirror:8,axis:8,triv:8", 8),
        ("reg:4,mirror:12,axis:8,triv:12", 16),
        ("reg:4,mirror:12,axis:8,triv:12", 8),
        ("reg:4,mirror:8,axis:8,triv:8", 16),
        ("reg:4,mirror:8,axis:8,triv:8", 8),
        ("reg:8,mirror:8,axis:4,triv:4", 16),
        ("reg:8,mirror:8,axis:4,triv:4", 8),
        ("reg:4,mirror:12,axis:8,triv:16", 16),
        ("reg:4,mirror:12,axis:8,triv:16", 8),
        ("reg:4,mirror:6,point:2,axis:8,triv:8", 16),
        ("reg:4,mirror:6,point:2,axis:8,triv:8", 8),
        ("reg:4,mirror:8,axis:4,triv:4", 8),
        ("reg:4,mirror:4,axis:4,triv:12", 8),
        ("mirror:16", 8),
    )
    return [
        evaluate(Signature.parse(text), _shape_with_k(base_shape, k))
        for text, k in rows
    ]


def _custom_candidates(
    signatures: Sequence[str], k_values: Sequence[int], base_shape: ModelShape
) -> list[Evaluation]:
    parsed = [Signature.parse(text) for text in signatures]
    ks = tuple(k_values) if k_values else (8, 16)
    unique: dict[tuple[Signature, int], Evaluation] = {}
    for signature, k in product(parsed, ks):
        unique[(signature, k)] = evaluate(signature, _shape_with_k(base_shape, k))
    return [unique[key] for key in sorted(unique, key=lambda item: (item[1], item[0]))]


@dataclass(frozen=True)
class SearchConstraints:
    min_width: int = 96
    max_width: int = 192
    multiplicity_step: int = 4
    min_instances: int = 16
    max_instances: int = 40
    min_reg: int = 4
    min_mirror: int = 4
    min_axis: int = 4
    min_triv: int = 4
    min_stored_param_ratio: float = 0.50
    max_stored_param_ratio: float = 1.50


def _search_candidates(
    base_shape: ModelShape,
    reference: Evaluation,
    constraints: SearchConstraints,
    alpha: float,
) -> list[Evaluation]:
    """Enumerate the G7-informed, deployment-aligned search space.

    G7 found mean mirror energy 0.9673 across depth.  The search nevertheless
    preserves a regular reserve, requires explicit mirror/axis/trivial capacity,
    and enforces quotient balance.  It is then Pareto-filtered by speed and
    effective parameter capacity, rather than selecting the narrowest stream.
    """

    step = constraints.multiplicity_step
    if step <= 0:
        raise ValueError("search multiplicity step must be positive")
    if constraints.min_width <= 0 or constraints.max_width < constraints.min_width:
        raise ValueError("invalid search width bounds")

    raw: dict[tuple[Signature, int], Evaluation] = {}
    first_width = ((constraints.min_width + 15) // 16) * 16
    for target_width in range(first_width, constraints.max_width + 1, 16):
        max_reg = target_width // TYPE_SLOTS["reg"]
        max_mirror = target_width // TYPE_SLOTS["mirror"]
        max_point = target_width // TYPE_SLOTS["point"]
        max_axis = target_width // TYPE_SLOTS["axis"]
        for reg in range(constraints.min_reg, max_reg + 1, step):
            for mirror in range(constraints.min_mirror, max_mirror + 1, step):
                if mirror < reg:  # mirror-heavy, directly informed by G7
                    continue
                for point_count in range(0, max_point + 1, step):
                    if point_count > mirror:
                        continue
                    for axis in range(constraints.min_axis, max_axis + 1, step):
                        if axis > mirror:
                            continue
                        used = (
                            12 * reg + 6 * mirror + 6 * point_count + 3 * axis
                        )
                        triv = target_width - used
                        if triv < constraints.min_triv or triv % step:
                            continue
                        # Prevent raw scalar-head multiplicity from masquerading
                        # as quotient capacity, and keep at least 2x as many
                        # quotient instances as fully chiral instances.
                        if triv > mirror + point_count + axis:
                            continue
                        quotient_instances = mirror + point_count + axis + triv
                        if quotient_instances < 2 * reg:
                            continue
                        signature = Signature.from_mapping(
                            {
                                "reg": reg,
                                "mirror": mirror,
                                "point": point_count,
                                "axis": axis,
                                "triv": triv,
                            }
                        )
                        if not (
                            constraints.min_instances
                            <= signature.instances
                            <= constraints.max_instances
                        ):
                            continue
                        for k in (8, 16):
                            candidate = evaluate(signature, _shape_with_k(base_shape, k))
                            if not candidate.deploy_legal:
                                continue
                            param_ratio = candidate.params_stored / reference.params_stored
                            if not (
                                constraints.min_stored_param_ratio
                                <= param_ratio
                                <= constraints.max_stored_param_ratio
                            ):
                                continue
                            raw[(signature, k)] = candidate

    candidates = list(raw.values())

    def dominates(left: Evaluation, right: Evaluation) -> bool:
        left_speed = _ratios(left, reference, alpha)["speedup_logical"]
        right_speed = _ratios(right, reference, alpha)["speedup_logical"]
        no_worse = (
            left_speed >= right_speed
            and left.params_effective >= right.params_effective
        )
        strictly_better = (
            left_speed > right_speed
            or left.params_effective > right.params_effective
        )
        return no_worse and strictly_better

    frontier = [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate and dominates(other, candidate)
            for other in candidates
        )
    ]
    frontier.sort(
        key=lambda candidate: (
            -_ratios(candidate, reference, alpha)["speedup_logical"],
            -candidate.params_effective,
            candidate.width,
            candidate.shape.k_attn,
            candidate.signature.canonical(),
        )
    )
    return frontier


def _evaluation_dict(
    evaluation: Evaluation,
    reference: Evaluation,
    alpha: float,
    alpha_grid: Sequence[float],
) -> dict[str, object]:
    ratios = _ratios(evaluation, reference, alpha)
    sensitivity = {
        f"{grid_alpha:.6g}": {
            "ideal": _speedup(
                ratios["flops_ratio"], ratios["bytes_ideal_ratio"], grid_alpha
            ),
            "logical": _speedup(
                ratios["flops_ratio"], ratios["bytes_logical_ratio"], grid_alpha
            ),
        }
        for grid_alpha in alpha_grid
    }
    return {
        "signature": evaluation.signature.canonical(),
        "multiplicities": dict(zip(TYPE_ORDER, evaluation.signature.multiplicities)),
        "width": evaluation.width,
        "instances": evaluation.instances,
        "k_attn": evaluation.shape.k_attn,
        "internal_width": 12 * evaluation.shape.k_attn,
        "aligned_16": evaluation.aligned_16,
        "head_dim_a": evaluation.head_dim_a,
        "head_dim_l": evaluation.head_dim_l,
        "legal_a": evaluation.legal_a,
        "legal_l": evaluation.legal_l,
        "deploy_legal": evaluation.deploy_legal,
        "linear_hom": evaluation.signature.linear_hom,
        "conv_hom": evaluation.signature.conv_hom,
        "params_effective": evaluation.params_effective,
        "params_stored": evaluation.params_stored,
        "macs": dict(evaluation.macs),
        "flops": evaluation.total_flops,
        "activation_bytes_ideal": dict(evaluation.bytes_ideal),
        "activation_bytes_logical": dict(evaluation.bytes_logical),
        "activation_bytes_ideal_total": evaluation.total_bytes_ideal,
        "activation_bytes_logical_total": evaluation.total_bytes_logical,
        **ratios,
        "alpha_sensitivity": sensitivity,
    }


def _fmt_ratio_range(first: float, second: float, digits: int = 3) -> str:
    if math.isclose(first, second, rel_tol=0.0, abs_tol=0.5 * 10 ** (-digits)):
        return f"{first:.{digits}f}"
    return f"{first:.{digits}f}–{second:.{digits}f}"


def _markdown_report(
    base_shape: ModelShape,
    reference: Evaluation,
    candidates: Sequence[Evaluation],
    frontier: Sequence[Evaluation],
    alpha: float,
    alpha_grid: Sequence[float],
    search_constraints: SearchConstraints,
    search_limit: int,
) -> str:
    lines: list[str] = []
    lines.extend(
        (
            "# Quotient-representation G8 cost model",
            "",
            f"Shape: `B={base_shape.batch}`, `Npad={base_shape.npad}`, "
            f"layout `{base_shape.layout}`, register lane "
            f"`{'on' if base_shape.reg_lane else 'off'}`. Reference: "
            "`reg:16`, `K_attn=16` at the same shape.",
            "",
            f"Nominal mixed-bound alpha is `{alpha:.6g}`. The endpoint-consistent "
            f"cost interpolation is `4/7 = {COST_CONSISTENT_ALPHA:.6f}`: "
            "`1/21 = alpha/18 + (1-alpha)/27`. The specified `0.67` instead "
            "comes from arithmetic throughput interpolation "
            "`21 ≈ .67*18 + .33*27`; both are reported below.",
            "",
            "FLOPs count multiply and add separately. Activation bytes count "
            "logical matmul activation reads/writes, not weights or pointwise "
            "traffic. `ideal` assumes ray K/V cache reuse equivalent to one "
            "stream-width read per query; `logical` counts all "
            f"{base_shape.ray_keys} gathered K/V operands. Large activations use "
            f"{base_shape.activation_bytes} bytes/element; register aggregation "
            f"and scalar tops use {base_shape.fp32_bytes} bytes/element.",
            "",
            "## Reference component accounting",
            "",
            "| Component | GFLOPs | Ideal GiB | Logical GiB |",
            "|---|---:|---:|---:|",
        )
    )
    for component in reference.macs:
        lines.append(
            f"| {component} | {2 * reference.macs[component] / 1e9:.3f} | "
            f"{reference.bytes_ideal[component] / 2**30:.4f} | "
            f"{reference.bytes_logical[component] / 2**30:.4f} |"
        )
    lines.extend(
        (
            f"| **total** | **{reference.total_flops / 1e9:.3f}** | "
            f"**{reference.total_bytes_ideal / 2**30:.4f}** | "
            f"**{reference.total_bytes_logical / 2**30:.4f}** |",
            "",
            f"Reference parameters: **{reference.params_effective:,} effective "
            f"DOF**, **{reference.params_stored:,} stored**. The difference is "
            "the current dense Reynolds-source stem (`w0`) nullspace.",
            "",
            "## Candidate ranking",
            "",
            "Byte and speed ranges are `ideal–logical` ray traffic. `A/L` shows "
            "head dimensions; a cross marks a fast-path illegality.",
            "",
            "| Signature | K | C | I | C%16 | A/L | Params eff/stored (M) | "
            "F ratio | B ratio | Speed @ nominal | Speed @ 4/7 logical |",
            "|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|",
        )
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_ratios(item, reference, alpha)["speedup_logical"],
            item.width,
            item.shape.k_attn,
            item.signature.canonical(),
        ),
    )
    for candidate in ranked:
        ratios = _ratios(candidate, reference, alpha)
        cost_alpha_speed = _speedup(
            ratios["flops_ratio"],
            ratios["bytes_logical_ratio"],
            COST_CONSISTENT_ALPHA,
        )
        a_mark = "✓" if candidate.legal_a else "✗"
        l_mark = "✓" if candidate.legal_l else "✗"
        lines.append(
            f"| `{candidate.signature.canonical()}` | {candidate.shape.k_attn} | "
            f"{candidate.width} | {candidate.instances} | "
            f"{'yes' if candidate.aligned_16 else 'no'} | "
            f"{candidate.head_dim_a}{a_mark}/{candidate.head_dim_l}{l_mark} | "
            f"{candidate.params_effective / 1e6:.3f}/"
            f"{candidate.params_stored / 1e6:.3f} | "
            f"{ratios['flops_ratio']:.3f} | "
            f"{_fmt_ratio_range(ratios['bytes_ideal_ratio'], ratios['bytes_logical_ratio'])} | "
            f"{_fmt_ratio_range(ratios['speedup_ideal'], ratios['speedup_logical'])} | "
            f"{cost_alpha_speed:.3f} |"
        )

    lines.extend(
        (
            "",
            "At fixed `(C, K_attn)`, the dense trunk FLOPs and bytes do not "
            "depend on the type mix. Only negligible pooled-head terms depend "
            "on `I`; G7 fit and the `H`, `V`, and parameter-capacity columns must "
            "break such ties. In particular, `reg:8,mirror:16` at C=192/K=16 "
            "is not compute-cheaper than `reg:16` and has more free parameters.",
            "",
            "## Alpha sensitivity (logical ray traffic)",
            "",
        )
    )
    alpha_headers = " | ".join(f"speed @{value:.3g}" for value in alpha_grid)
    lines.extend(
        (
            f"| Signature | K | {alpha_headers} |",
            "|---|---:|" + "---:|" * len(alpha_grid),
        )
    )
    for candidate in ranked:
        ratios = _ratios(candidate, reference, alpha)
        speeds = " | ".join(
            f"{_speedup(ratios['flops_ratio'], ratios['bytes_logical_ratio'], value):.3f}"
            for value in alpha_grid
        )
        lines.append(
            f"| `{candidate.signature.canonical()}` | {candidate.shape.k_attn} | "
            f"{speeds} |"
        )

    if frontier:
        lines.extend(
            (
                "",
                "## G7-informed aligned Pareto search",
                "",
                f"G7 mirror energy averaged **{G7_MIRROR_DEPTH_MEAN:.4f}** and "
                "all 11 audited depths exceeded 0.954. Search constraints still "
                f"preserve `reg>={search_constraints.min_reg}`, require "
                f"`mirror>={search_constraints.min_mirror}`, "
                f"`axis>={search_constraints.min_axis}`, "
                f"`triv>={search_constraints.min_triv}`, enforce "
                "`mirror>=reg`, at least two quotient instances per regular "
                "instance, `mirror>=axis,point`, aligned C, balanced triv "
                "capacity, and a stored-param "
                f"window [{search_constraints.min_stored_param_ratio:.2f}, "
                f"{search_constraints.max_stored_param_ratio:.2f}]x baseline. "
                "The frontier maximizes both nominal logical-byte speed and "
                "effective parameter capacity, so it is not a speed-only search.",
                "",
                "| Signature | K | C | I | Params eff/stored (M) | F ratio | "
                "B logical | Speed |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for candidate in frontier[:search_limit]:
            ratios = _ratios(candidate, reference, alpha)
            lines.append(
                f"| `{candidate.signature.canonical()}` | {candidate.shape.k_attn} | "
                f"{candidate.width} | {candidate.instances} | "
                f"{candidate.params_effective / 1e6:.3f}/"
                f"{candidate.params_stored / 1e6:.3f} | "
                f"{ratios['flops_ratio']:.3f} | "
                f"{ratios['bytes_logical_ratio']:.3f} | "
                f"{ratios['speedup_logical']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def _parse_alpha_grid(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"invalid alpha grid {text!r}") from exc
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("alpha grid must contain values in [0,1]")
    return values


def run_self_tests() -> None:
    """Closed-form anchors; raises AssertionError on any regression."""

    assert LINEAR_DIMS[0][0] == 12
    assert CONV_DIMS[0][0] == 84
    assert CONV_DIMS[4][4] == 2
    assert STEM_EFFECTIVE_DIMS == (175, 113, 100, 67, 38)

    base_sig = Signature.parse("reg:16")
    assert base_sig.width == 192
    assert base_sig.instances == 16
    assert base_sig.linear_hom == 3072
    assert base_sig.conv_hom == 21504

    base_shape = ModelShape()
    baseline = evaluate(base_sig, base_shape)
    assert baseline.params_effective == 679_626
    assert baseline.params_stored == 710_426
    assert math.isclose(
        baseline.total_flops / 1e9,
        273.675534336,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert baseline.macs["conv_blocks"] == 5 * 96 * 250 * 14 * 192 * 192
    assert baseline.total_bytes_logical >= baseline.total_bytes_ideal

    required_widths = {
        "reg:8,mirror:8,axis:8,triv:8": 176,
        "reg:4,mirror:12,axis:8,triv:12": 156,
        "reg:8,mirror:16": 192,
        "reg:4,mirror:8,axis:8,triv:8": 128,
    }
    for text, expected_width in required_widths.items():
        assert Signature.parse(text).width == expected_width
    assert Signature.parse("reg:4,mirror:12,axis:8,triv:12").width % 16 != 0

    k8 = evaluate(
        Signature.parse("reg:4,mirror:8,axis:8,triv:8"),
        _shape_with_k(base_shape, 8),
    )
    assert (k8.head_dim_a, k8.head_dim_l) == (32, 16)
    assert k8.deploy_legal
    assert math.isclose(
        k8.total_flops / baseline.total_flops,
        0.42007679047720137,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    same_width = evaluate(Signature.parse("reg:8,mirror:16"), base_shape)
    # Only tiny I-dependent scalar-head terms differ at fixed C/K.
    assert abs(same_width.total_flops / baseline.total_flops - 1.0) < 5e-5
    assert same_width.params_effective > baseline.params_effective
    assert len(_built_in_candidates(base_shape)) >= 12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-form FLOP/activation-byte model for quotient signatures."
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--npad", type=int, default=DEFAULT_NPAD)
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument(
        "--signature",
        action="append",
        default=[],
        help="TYPE:MULT comma list; repeat for a custom sweep",
    )
    parser.add_argument(
        "--k-attn",
        type=int,
        action="append",
        default=[],
        help="attention orbit width; repeat to cross custom signatures",
    )
    parser.add_argument("--ray-keys", type=int, default=RAY_KEYS_MAX)
    parser.add_argument("--activation-bytes", type=int, default=2)
    parser.add_argument("--fp32-bytes", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--alpha-grid",
        default="0.50,0.5714285714285714,0.67,0.80",
    )
    parser.add_argument("--no-reg-lane", action="store_true")
    parser.add_argument("--reg-token-read", action="store_true")
    parser.add_argument("--request-moves-left", action="store_true")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--search-min-width", type=int, default=96)
    parser.add_argument("--search-max-width", type=int, default=192)
    parser.add_argument("--search-step", type=int, default=4)
    parser.add_argument("--search-min-instances", type=int, default=16)
    parser.add_argument("--search-max-instances", type=int, default=40)
    parser.add_argument("--search-min-reg", type=int, default=4)
    parser.add_argument("--search-min-mirror", type=int, default=4)
    parser.add_argument("--search-min-axis", type=int, default=4)
    parser.add_argument("--search-min-triv", type=int, default=4)
    parser.add_argument("--search-min-param-ratio", type=float, default=0.50)
    parser.add_argument("--search-max-param-ratio", type=float, default=1.50)
    parser.add_argument("--search-limit", type=int, default=12)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run closed-form anchors and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Windows Python may inherit a legacy console code page; the deterministic
    # Markdown contains mathematical/checkmark glyphs used by the results doc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    try:
        if not 0.0 <= args.alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        alpha_grid = _parse_alpha_grid(args.alpha_grid)
        run_self_tests()
        if args.self_test:
            print(
                "self-test: ok (baseline 273.675534336 GFLOP, "
                "679626 effective / 710426 stored params)"
            )
            return 0

        base_shape = ModelShape(
            batch=args.batch,
            npad=args.npad,
            layout=args.layout,
            k_attn=16,
            ray_keys=args.ray_keys,
            activation_bytes=args.activation_bytes,
            fp32_bytes=args.fp32_bytes,
            reg_lane=not args.no_reg_lane,
            reg_token_read=args.reg_token_read,
            request_moves_left=args.request_moves_left,
        )
        reference = evaluate(Signature.parse("reg:16"), base_shape)
        if args.signature:
            candidates = _custom_candidates(
                args.signature, args.k_attn, base_shape
            )
        elif args.k_attn:
            signatures = sorted(
                {candidate.signature.canonical() for candidate in _built_in_candidates(base_shape)}
            )
            candidates = _custom_candidates(signatures, args.k_attn, base_shape)
        else:
            candidates = _built_in_candidates(base_shape)

        search_constraints = SearchConstraints(
            min_width=args.search_min_width,
            max_width=args.search_max_width,
            multiplicity_step=args.search_step,
            min_instances=args.search_min_instances,
            max_instances=args.search_max_instances,
            min_reg=args.search_min_reg,
            min_mirror=args.search_min_mirror,
            min_axis=args.search_min_axis,
            min_triv=args.search_min_triv,
            min_stored_param_ratio=args.search_min_param_ratio,
            max_stored_param_ratio=args.search_max_param_ratio,
        )
        frontier = (
            []
            if args.no_search
            else _search_candidates(
                base_shape, reference, search_constraints, args.alpha
            )
        )
        if args.search_limit < 0:
            raise ValueError("search_limit must be non-negative")

        if args.format == "json":
            payload = {
                "assumptions": {
                    "batch": base_shape.batch,
                    "npad": base_shape.npad,
                    "layout": base_shape.layout,
                    "num_tokens": NUM_TOKENS,
                    "ray_keys": base_shape.ray_keys,
                    "activation_bytes": base_shape.activation_bytes,
                    "fp32_bytes": base_shape.fp32_bytes,
                    "reg_lane": base_shape.reg_lane,
                    "reg_token_read": base_shape.reg_token_read,
                    "request_moves_left": base_shape.request_moves_left,
                    "nominal_alpha": args.alpha,
                    "cost_consistent_alpha": COST_CONSISTENT_ALPHA,
                    "g7_mirror_depth_mean": G7_MIRROR_DEPTH_MEAN,
                    "flop_definition": "2 FLOPs per MAC",
                    "byte_definition": (
                        "logical matmul activation operand reads+writes; weights, "
                        "pointwise ops, masks, and indices excluded"
                    ),
                },
                "reference": _evaluation_dict(
                    reference, reference, args.alpha, alpha_grid
                ),
                "candidates": [
                    _evaluation_dict(candidate, reference, args.alpha, alpha_grid)
                    for candidate in candidates
                ],
                "pareto_search": [
                    _evaluation_dict(candidate, reference, args.alpha, alpha_grid)
                    for candidate in frontier[: args.search_limit]
                ],
            }
            json.dump(payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            sys.stdout.write(
                _markdown_report(
                    base_shape,
                    reference,
                    candidates,
                    frontier,
                    args.alpha,
                    alpha_grid,
                    search_constraints,
                    args.search_limit,
                )
            )
        return 0
    except (AssertionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
