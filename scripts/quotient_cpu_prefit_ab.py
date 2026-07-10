"""Bounded CPU behavior-cloning probe for Phase-A quotient representations.

This is deliberately not a production-model benchmark.  It uses only the
CPU typed layers in :mod:`hexfield_eq.reps`, real compact-v4 rows expanded to
the 25-plane feature map, and a small ``CCA`` network.  The three arms have
the same materialized widths and therefore the same dense matmul work:

* ``reg:8``;
* ``reg:4,mirror:4,axis:4,triv:12``;
* ``mirror:16``.

Every arm has C=96, two genuine two-convolution residual C blocks, and one A
block whose q/k/v interior is the regular representation ``reg:8``.  The A
block operates on the joint ``[6 learned tokens; cells]`` sequence.  Its
relative-position bias is intentionally zero: the purpose of this probe is
typed channel capacity, not a CPU reimplementation of the production joint
bias LUT.  The policy head is typed conv -> fixed-width typed expansion ->
typed group pool -> linear.

No production model module, Triton module, or CUDA API is imported or used.
The script pins the import-time feature/support environment before importing
``hexfield_eq`` so every accepted row is a 25-plane, support-radius-4 row.

Example (WSL)::

    PYTHONPATH=packages/hexfield_eq/python \
      python scripts/quotient_cpu_prefit_ab.py \
      --data /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1/samples --smoke

The source may instead be a standard ``train/`` + ``val/`` shard tree, an
explicit ``.buffer_manifest.json``, one or more compact shards/directories,
or glob patterns passed with ``--shards``.  Inputs are always read-only.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import hashlib
import json
import math
import os
import random
import shlex
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# These modules read shape/feature knobs at import time.  Refuse a conflicting
# parent environment rather than silently constructing a different experiment.
_PINNED_ENV: Final = {
    "HEXFIELD_EQ_FEATURE_VERSION": "1",
    "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
    "HEXFIELD_EQ_CHANNELS": "96",
    "HEXFIELD_EQ_GROUP_ORDER": "12",
    "HEXFIELD_EQ_C_ORBIT": "8",
    "HEXFIELD_EQ_ATTENTION_HEADS": "3",
    "HEXFIELD_EQ_TRUNK": "CCA",
}
for _name, _value in _PINNED_ENV.items():
    _present = os.environ.get(_name)
    if _present is not None and _present != _value:
        raise RuntimeError(
            f"{_name}={_present!r} conflicts with the CPU probe's pinned {_value!r}"
        )
    os.environ[_name] = _value

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hexfield_eq.constants import NUM_FEATURES
from hexfield_eq.reps import (
    Signature,
    SignatureLike,
    TypedConv,
    TypedGroupAffineNorm,
    TypedLayerScale,
    TypedLinear,
    TypedStem,
    canonical_signature,
    expand_per_instance,
    head_perm,
    head_perm_inv,
    orbit_dimension,
    scale_signature,
    signature_instances,
    signature_width,
    typed_group_pool,
)
from hexfield_eq.samples import ExpandedRow, expand_sample
from hexfield_eq.shards import read_compact_shard


SEED: Final = 0
WIDTH: Final = 96
TOKENS: Final = 6
ATTN_K: Final = 8
ATTN_SIGNATURE: Final[Signature] = (("reg", ATTN_K),)
HEADS: Final = 3
HEAD_DIM: Final = 32
SCHEMA: Final = "hexfield_compact_v1"
SCHEMA_VERSION: Final = 4

# A common output representation keeps *all* policy-head dense shapes equal,
# including the post-pool linear.  It has width 192 and 48 pooled invariants.
POLICY_EXPAND_SIGNATURE: Final[Signature] = (("mirror", 16), ("axis", 32))
POLICY_POOLED_WIDTH: Final = 48

ARM_SIGNATURES: Final[tuple[tuple[str, Signature], ...]] = (
    ("reg:8", (("reg", 8),)),
    (
        "reg:4,mirror:4,axis:4,triv:12",
        (("reg", 4), ("mirror", 4), ("axis", 4), ("triv", 12)),
    ),
    ("mirror:16", (("mirror", 16),)),
)


@dataclass(frozen=True)
class ShardRef:
    """Read-only compact-shard location and trusted manifest row count."""

    path: Path
    rows: int


@dataclass(frozen=True)
class Position:
    """One expanded, cap-eligible row and its normalized BC target."""

    row: ExpandedRow
    target: np.ndarray

    @property
    def nodes(self) -> int:
        return self.row.support.num_nodes


@dataclass(frozen=True)
class TensorBatch:
    """Minimal CPU batch needed by the typed policy-only network."""

    feats: torch.Tensor
    nbr: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    legal_counts: torch.Tensor


@dataclass(frozen=True)
class Metrics:
    """Row-mean held-out behavior-cloning metrics."""

    ce: float
    top1: float
    rows: int


@dataclass(frozen=True)
class ArmResult:
    """One arm's complete bounded-run result."""

    name: str
    signature: str
    stored_params: int
    effective_params: int
    initial: Metrics
    final: Metrics
    train_seconds: float
    steps: int
    common_init_digest: str


class CPUAdamW:
    """Small eager fp32 AdamW implementation that never enters GPU/compiler code.

    Recent PyTorch optimizer wrappers may lazily import TorchDynamo and an
    installed Triton package even for CPU tensors.  That is inappropriate for
    this proof, so the few lines of ordinary AdamW state evolution live here.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
    ) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.steps = 0
        self.first = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second = [torch.zeros_like(parameter) for parameter in self.parameters]

    def zero_grad(self) -> None:
        """Clear gradients without allocating replacement tensors."""

        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        """Apply one decoupled-weight-decay Adam update."""

        self.steps += 1
        correction1 = 1.0 - self.beta1**self.steps
        correction2 = 1.0 - self.beta2**self.steps
        step_size = self.lr * math.sqrt(correction2) / correction1
        for parameter, first, second in zip(
            self.parameters, self.first, self.second, strict=True
        ):
            gradient = parameter.grad
            if gradient is None:
                continue
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(
                gradient, gradient, value=1.0 - self.beta2
            )
            parameter.addcdiv_(
                first, second.sqrt().add_(self.eps * math.sqrt(correction2)),
                value=-step_size,
            )


def _read_sidecar(path: Path, *, require_v4: bool = True) -> dict[str, Any]:
    """Validate a compact shard's commit-marker sidecar."""

    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"compact shard lacks committed sidecar: {path}")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"unsupported shard schema in {sidecar}: {document.get('schema')!r}")
    version = int(document.get("schema_version", -1))
    if require_v4 and version != SCHEMA_VERSION:
        raise ValueError(
            f"{sidecar} is compact schema v{version}; this experiment requires v4"
        )
    rows = int(document.get("rows", -1))
    if rows <= 0:
        raise ValueError(f"invalid row count in {sidecar}: {rows}")
    return document


def _manifest_refs(manifest_path: Path) -> list[ShardRef]:
    """Resolve live-buffer manifest entries without modifying the manifest."""

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"manifest entries are not a list: {manifest_path}")
    root = manifest_path.parent
    refs: list[ShardRef] = []
    missing = 0
    for entry in entries:
        path = (root / str(entry["rel_path"])).resolve()
        sidecar = path.with_suffix(".json")
        # A live buffer may prune after atomically publishing its manifest.  We
        # take an in-memory read-only snapshot of entries still fully committed.
        if not path.is_file() or not sidecar.is_file():
            missing += 1
            continue
        rows = int(entry["rows"])
        if rows <= 0:
            raise ValueError(f"manifest has invalid row count for {path}: {rows}")
        refs.append(ShardRef(path, rows))
    if missing:
        print(f"manifest snapshot skipped {missing} vanished/uncommitted entries", flush=True)
    if not refs:
        raise ValueError(f"manifest has no available committed shards: {manifest_path}")
    return refs


def _scan_refs(path: Path) -> list[ShardRef]:
    """Discover committed v4 shards beneath a path."""

    paths = [path] if path.is_file() and path.suffix.lower() == ".npz" else []
    if path.is_dir():
        paths = sorted(path.rglob("*.npz"))
    refs: list[ShardRef] = []
    for shard in paths:
        if not shard.with_suffix(".json").is_file():
            continue
        sidecar = _read_sidecar(shard)
        refs.append(ShardRef(shard.resolve(), int(sidecar["rows"])))
    return refs


def _refs_for_path(path: Path) -> list[ShardRef]:
    """Interpret one source path as a manifest, manifest root, or shard tree."""

    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"data source does not exist: {path}")
    if path.is_file() and path.name == ".buffer_manifest.json":
        return _manifest_refs(path.resolve())
    if path.is_dir() and (path / ".buffer_manifest.json").is_file():
        return _manifest_refs((path / ".buffer_manifest.json").resolve())
    refs = _scan_refs(path.resolve())
    if not refs:
        raise ValueError(f"no committed compact-v4 shards under {path}")
    return refs


def _dedupe_refs(refs: Iterable[ShardRef]) -> list[ShardRef]:
    """Deduplicate paths and reject conflicting row metadata."""

    result: dict[Path, ShardRef] = {}
    for ref in refs:
        prior = result.get(ref.path)
        if prior is not None and prior.rows != ref.rows:
            raise ValueError(f"conflicting row counts for duplicate shard {ref.path}")
        result[ref.path] = ref
    return sorted(result.values(), key=lambda item: str(item.path))


def discover_sources(
    data: str | None, shard_arguments: Sequence[str] | None
) -> tuple[list[ShardRef], list[ShardRef] | None, str]:
    """Return train refs, optional pre-split validation refs, and a description."""

    if data is not None:
        root = Path(data).expanduser()
        train_dir, val_dir = root / "train", root / "val"
        if root.is_dir() and train_dir.is_dir() and val_dir.is_dir():
            train_refs = _dedupe_refs(_refs_for_path(train_dir))
            val_refs = _dedupe_refs(_refs_for_path(val_dir))
            overlap = {ref.path for ref in train_refs} & {ref.path for ref in val_refs}
            if overlap:
                raise ValueError(f"train/val shard overlap: {next(iter(overlap))}")
            return train_refs, val_refs, f"pre-split tree {root.resolve()}"
        refs = _dedupe_refs(_refs_for_path(root))
        return refs, None, f"deterministic row split of {root.resolve()}"

    assert shard_arguments is not None
    paths: list[Path] = []
    for argument in shard_arguments:
        if glob.has_magic(argument):
            matches = sorted(glob.glob(argument, recursive=True))
            if not matches:
                raise FileNotFoundError(f"shard glob matched nothing: {argument}")
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(argument))
    refs = _dedupe_refs(ref for path in paths for ref in _refs_for_path(path))
    return refs, None, "deterministic row split of explicit --shards"


def _sample_global_indices(
    refs: Sequence[ShardRef], count: int, rng: random.Random
) -> list[int]:
    """Uniformly sample distinct global rows from a manifest snapshot."""

    total = sum(ref.rows for ref in refs)
    if count > total:
        raise ValueError(f"requested {count} candidate rows from only {total}")
    return rng.sample(range(total), count)


def _load_selected_rows(
    refs: Sequence[ShardRef], indices: Sequence[int]
) -> list[Any]:
    """Decode only shards touched by a list of global row indices."""

    cumulative: list[int] = []
    running = 0
    for ref in refs:
        running += ref.rows
        cumulative.append(running)

    selected: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for output_index, global_index in enumerate(indices):
        shard_index = bisect.bisect_right(cumulative, global_index)
        previous = cumulative[shard_index - 1] if shard_index else 0
        selected[shard_index].append((output_index, global_index - previous))

    output: list[Any | None] = [None] * len(indices)
    for shard_index in sorted(selected):
        ref = refs[shard_index]
        sidecar = _read_sidecar(ref.path)
        if int(sidecar["rows"]) != ref.rows:
            raise ValueError(
                f"manifest/sidecar row mismatch for {ref.path}: "
                f"{ref.rows} != {sidecar['rows']}"
            )
        with np.load(ref.path) as archive:
            actual_version = int(archive["schema_version"])
            actual_rows = int(archive["num_rows"])
        if actual_version != SCHEMA_VERSION:
            raise ValueError(f"{ref.path} payload is schema v{actual_version}, expected v4")
        if actual_rows != ref.rows:
            raise ValueError(
                f"payload row count mismatch for {ref.path}: {actual_rows} != {ref.rows}"
            )
        rows = read_compact_shard(ref.path)
        if len(rows) != ref.rows:
            raise ValueError(f"decoder row count mismatch for {ref.path}")
        for output_index, local_index in selected[shard_index]:
            output[output_index] = rows[local_index]
    if any(row is None for row in output):
        raise AssertionError("selected-row loader left an empty slot")
    return [row for row in output if row is not None]


def _policy_target(row: ExpandedRow, target_kind: str) -> np.ndarray:
    """Choose and normalize a legal-prefix visit or improved-policy target."""

    target = row.policy
    if (
        target_kind == "gumbel"
        and float(getattr(row, "gumbel_policy_valid", 0.0)) > 0.0
        and getattr(row, "gumbel_policy", np.empty(0)).shape == row.policy.shape
        and float(row.gumbel_policy.sum()) > 0.0
    ):
        target = row.gumbel_policy
    result = np.asarray(target, dtype=np.float32).copy()
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("behavior-cloning target has no finite positive mass")
    result /= total
    return result


def _expand_eligible(
    samples: Sequence[Any],
    wanted: int,
    node_cap: int,
    target_kind: str,
) -> tuple[list[Position], dict[str, int]]:
    """Expand candidates without truncation and keep rows under the node cap."""

    positions: list[Position] = []
    over_cap = 0
    for sample in samples:
        row = expand_sample(sample, symmetry=0)
        if row.feats.shape != (row.support.num_nodes, 25):
            raise AssertionError(f"expanded feature shape is {row.feats.shape}, expected (N,25)")
        if row.support.num_nodes > node_cap:
            over_cap += 1
            continue
        if row.support.legal_count != row.policy.shape[0]:
            raise AssertionError("policy target does not match legal prefix")
        positions.append(Position(row=row, target=_policy_target(row, target_kind)))
        if len(positions) == wanted:
            break
    stats = {"candidates": len(samples), "over_node_cap": over_cap}
    if len(positions) < wanted:
        raise ValueError(
            f"only {len(positions)}/{wanted} candidates fit --node-cap={node_cap}; "
            "increase --candidate-factor or --node-cap"
        )
    return positions, stats


def load_positions(
    train_refs: Sequence[ShardRef],
    val_refs: Sequence[ShardRef] | None,
    *,
    train_rows: int,
    val_rows: int,
    candidate_factor: int,
    node_cap: int,
    target_kind: str,
    seed: int,
) -> tuple[list[Position], list[Position], dict[str, int]]:
    """Select deterministic disjoint candidates and expand held-out positions."""

    rng = random.Random(seed)
    n_train_candidates = min(
        sum(ref.rows for ref in train_refs), max(train_rows, train_rows * candidate_factor)
    )
    if val_refs is not None:
        n_val_candidates = min(
            sum(ref.rows for ref in val_refs), max(val_rows, val_rows * candidate_factor)
        )
        train_indices = _sample_global_indices(train_refs, n_train_candidates, rng)
        val_indices = _sample_global_indices(val_refs, n_val_candidates, rng)
        raw_train = _load_selected_rows(train_refs, train_indices)
        raw_val = _load_selected_rows(val_refs, val_indices)
    else:
        total = sum(ref.rows for ref in train_refs)
        n_val_candidates = min(total - n_train_candidates, val_rows * candidate_factor)
        if n_val_candidates < val_rows:
            raise ValueError("not enough rows for a disjoint held-out validation split")
        all_indices = _sample_global_indices(
            train_refs, n_train_candidates + n_val_candidates, rng
        )
        train_indices = all_indices[:n_train_candidates]
        val_indices = all_indices[n_train_candidates:]
        if set(train_indices) & set(val_indices):
            raise AssertionError("row-level train/validation split overlaps")
        raw_train = _load_selected_rows(train_refs, train_indices)
        raw_val = _load_selected_rows(train_refs, val_indices)

    train, train_stats = _expand_eligible(
        raw_train, train_rows, node_cap, target_kind
    )
    val, val_stats = _expand_eligible(raw_val, val_rows, node_cap, target_kind)
    stats = {
        "train_candidates": train_stats["candidates"],
        "train_over_node_cap": train_stats["over_node_cap"],
        "val_candidates": val_stats["candidates"],
        "val_over_node_cap": val_stats["over_node_cap"],
    }
    return train, val, stats


def make_batch_plan(
    positions: Sequence[Position],
    *,
    steps: int,
    batch_size: int,
    bucket_width: int,
    seed: int,
) -> list[list[int]]:
    """Create a reusable node-bucketed batch plan shared by every arm."""

    plan: list[list[int]] = []
    epoch = 0
    while len(plan) < steps:
        rng = random.Random(seed + 1009 * epoch)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, position in enumerate(positions):
            buckets[(position.nodes - 1) // bucket_width].append(index)
        epoch_batches: list[list[int]] = []
        leftovers: list[int] = []
        for key in sorted(buckets):
            indices = buckets[key]
            rng.shuffle(indices)
            full = len(indices) // batch_size
            for batch_index in range(full):
                start = batch_index * batch_size
                epoch_batches.append(indices[start : start + batch_size])
            leftovers.extend(indices[full * batch_size :])
        rng.shuffle(leftovers)
        for start in range(0, len(leftovers), batch_size):
            chunk = leftovers[start : start + batch_size]
            if chunk:
                epoch_batches.append(chunk)
        rng.shuffle(epoch_batches)
        if not epoch_batches:
            raise ValueError("no train batches could be formed")
        plan.extend(epoch_batches)
        epoch += 1
    return plan[:steps]


def _validation_plan(
    positions: Sequence[Position], batch_size: int, bucket_width: int
) -> list[list[int]]:
    """Stable complete validation plan with low padding waste."""

    order = sorted(range(len(positions)), key=lambda index: positions[index].nodes)
    result: list[list[int]] = []
    for bucket_start in range(0, len(order), batch_size * 8):
        bucket = order[bucket_start : bucket_start + batch_size * 8]
        for start in range(0, len(bucket), batch_size):
            result.append(bucket[start : start + batch_size])
    return result


def collate(
    positions: Sequence[Position], indices: Sequence[int], pad_quantum: int
) -> TensorBatch:
    """Pad one small CPU batch; no production batching/model code is used."""

    selected = [positions[index] for index in indices]
    npad = max(position.nodes for position in selected)
    npad = -(-npad // pad_quantum) * pad_quantum
    batch = len(selected)
    feats = torch.zeros((batch, npad, 25), dtype=torch.float32)
    nbr = torch.full((batch, npad, 6), -1, dtype=torch.long)
    mask = torch.zeros((batch, npad), dtype=torch.bool)
    target = torch.zeros((batch, npad), dtype=torch.float32)
    legal_counts = torch.zeros((batch,), dtype=torch.long)
    for row_index, position in enumerate(selected):
        row = position.row
        nodes = position.nodes
        legal = row.support.legal_count
        feats[row_index, :nodes] = torch.from_numpy(np.ascontiguousarray(row.feats))
        nbr[row_index, :nodes] = torch.from_numpy(
            np.ascontiguousarray(row.support.nbr, dtype=np.int64)
        )
        mask[row_index, :nodes] = True
        target[row_index, :legal] = torch.from_numpy(position.target)
        legal_counts[row_index] = legal
    return TensorBatch(feats, nbr, mask, target, legal_counts)


class ConvBlock(nn.Module):
    """Production-shaped post-activation typed residual block with two convs."""

    def __init__(self, signature: Signature) -> None:
        super().__init__()
        self.conv1 = TypedConv(signature, signature)
        self.norm1 = TypedGroupAffineNorm(signature)
        self.conv2 = TypedConv(signature, signature)
        self.norm2 = TypedGroupAffineNorm(signature)
        self.scale = TypedLayerScale(signature, init=1.0e-2)

    def forward(
        self, x: torch.Tensor, nbr: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        branch = F.relu(self.norm1(self.conv1(x, nbr)))
        branch = self.norm2(self.conv2(branch, nbr))
        return F.relu(x + self.scale(branch)) * mask.unsqueeze(-1)


class JointAttentionBlock(nn.Module):
    """Typed-stream A block with a fixed 96-wide regular attention interior."""

    def __init__(self, signature: Signature) -> None:
        super().__init__()
        hidden = scale_signature(signature, 2)
        self.signature = signature
        self.norm1 = TypedGroupAffineNorm(signature)
        self.q = TypedLinear(signature, ATTN_SIGNATURE)
        self.k = TypedLinear(signature, ATTN_SIGNATURE)
        self.v = TypedLinear(signature, ATTN_SIGNATURE)
        self.out = TypedLinear(ATTN_SIGNATURE, signature)
        self.scale1 = TypedLayerScale(signature, init=1.0e-2)
        self.norm2 = TypedGroupAffineNorm(signature)
        self.fc1 = TypedLinear(signature, hidden)
        self.fc2 = TypedLinear(hidden, signature)
        self.scale2 = TypedLayerScale(signature, init=1.0e-2)
        self.register_buffer("head_order", head_perm(ATTN_K), persistent=False)
        self.register_buffer("head_inverse", head_perm_inv(ATTN_K), persistent=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        if width != WIDTH:
            raise AssertionError(f"attention interior width {width} != {WIDTH}")
        return x.index_select(-1, self.head_order).reshape(
            batch, length, HEADS, HEAD_DIM
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        q = self._heads(self.q(normalized))
        k = self._heads(self.k(normalized))
        v = self._heads(self.v(normalized))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        # Zero joint relative-position bias is intentional and documented in
        # the module docstring: only channel-type capacity is under test here.
        scores = scores.masked_fill(~mask[:, None, None, :], -1.0e9)
        attended = torch.matmul(F.softmax(scores, dim=-1), v)
        attended = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], WIDTH)
        attended = attended.index_select(-1, self.head_inverse)
        x = x + self.scale1(self.out(attended)) * mask.unsqueeze(-1)
        branch = self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return (x + self.scale2(branch)) * mask.unsqueeze(-1)


class TypedMicroNet(nn.Module):
    """The fixed-width CCA policy network used for all three CPU arms."""

    def __init__(self, signature: SignatureLike) -> None:
        super().__init__()
        self.signature = canonical_signature(signature)
        if signature_width(self.signature) != WIDTH:
            raise ValueError(f"trunk signature width must be {WIDTH}")
        self.stem = TypedStem(self.signature)
        self.stem_norm = TypedGroupAffineNorm(self.signature)
        self.conv_blocks = nn.ModuleList([ConvBlock(self.signature) for _ in range(2)])
        self.token_base = nn.Parameter(
            torch.empty((TOKENS, signature_instances(self.signature)))
        )
        nn.init.normal_(self.token_base, mean=0.0, std=0.02)
        self.attn = JointAttentionBlock(self.signature)
        self.policy_conv = TypedConv(self.signature, self.signature)
        self.policy_expand = TypedLinear(self.signature, POLICY_EXPAND_SIGNATURE)
        self.policy_read = nn.Linear(POLICY_POOLED_WIDTH, 1)
        self.reset_readout()

    def reset_readout(self) -> None:
        """Use a small common readout initialization for stable initial CE."""

        nn.init.normal_(self.policy_read.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.policy_read.bias)

    def forward(
        self, feats: torch.Tensor, nbr: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if feats.device.type != "cpu" or nbr.device.type != "cpu" or mask.device.type != "cpu":
            raise RuntimeError("quotient CPU prefit received a non-CPU tensor")
        if feats.shape[-1] != 25:
            raise ValueError("micro-prefit requires the compact-v4 25-plane feature map")
        cells = F.relu(self.stem_norm(self.stem(feats, nbr))) * mask.unsqueeze(-1)
        for block in self.conv_blocks:
            cells = block(cells, nbr, mask)

        token_channels = expand_per_instance(self.token_base, self.signature)
        tokens = token_channels.unsqueeze(0).expand(cells.shape[0], -1, -1)
        joint = torch.cat((tokens, cells), dim=1)
        token_mask = torch.ones(
            (cells.shape[0], TOKENS), dtype=torch.bool, device=cells.device
        )
        joint_mask = torch.cat((token_mask, mask), dim=1)
        joint = self.attn(joint, joint_mask)
        cells = joint[:, TOKENS:]

        policy_features = F.relu(self.policy_conv(cells, nbr)) * mask.unsqueeze(-1)
        expanded = self.policy_expand(policy_features)
        pooled = typed_group_pool(expanded, POLICY_EXPAND_SIGNATURE)
        if pooled.shape[-1] != POLICY_POOLED_WIDTH:
            raise AssertionError("policy pooled width drifted")
        return self.policy_read(pooled).squeeze(-1) * mask


@torch.no_grad()
def reset_common_initialization(model: TypedMicroNet, seed: int) -> str:
    """Reset same-shaped boundary tensors identically and return their digest.

    Quotient coefficient tensors generally have different shapes and meanings,
    so equality is neither possible nor desirable there.  The raw stem lift
    and final invariant readout are shape-identical across arms; this function
    isolates them from differing constructor RNG consumption and proves their
    bitwise-identical initialization.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 7919)
    model.stem.w0.normal_(
        mean=0.0, std=(7 * 25) ** -0.5, generator=generator
    )
    model.policy_read.weight.normal_(mean=0.0, std=0.02, generator=generator)
    model.policy_read.bias.zero_()
    digest = hashlib.sha256()
    digest.update(model.stem.w0.detach().numpy().tobytes())
    digest.update(model.policy_read.weight.detach().numpy().tobytes())
    digest.update(model.policy_read.bias.detach().numpy().tobytes())
    return digest.hexdigest()[:16]


def _typed_stem_effective(signature: Signature) -> int:
    """Rank of the Reynolds-projected 25-plane stem, including typed bias."""

    total = 0
    for out_type, out_instances in signature:
        total += out_instances * (
            13 * orbit_dimension("triv", out_type, conv=True)
            + 4 * orbit_dimension("axis", out_type, conv=True)
        )
    return total + signature_instances(signature)


def parameter_counts(model: TypedMicroNet) -> tuple[int, int]:
    """Return stored tensors and symmetry-effective trainable degrees of freedom."""

    stored = sum(parameter.numel() for parameter in model.parameters())
    handled: set[int] = set()
    effective = 0
    for module in model.modules():
        if isinstance(module, TypedStem):
            effective += _typed_stem_effective(module.out_signature)
        elif isinstance(module, (TypedLinear, TypedConv)):
            effective += sum(parameter.numel() for parameter in module.parameters())
        elif isinstance(module, (TypedGroupAffineNorm, TypedLayerScale)):
            effective += sum(parameter.numel() for parameter in module.parameters())
        else:
            continue
        handled.update(id(parameter) for parameter in module.parameters())
    effective += sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in handled
    )
    if not 0 < effective <= stored:
        raise AssertionError(f"invalid parameter accounting: {effective=} {stored=}")
    return stored, effective


def _loss_and_hits(
    logits: torch.Tensor, batch: TensorBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-mean CE and top-1 hits over each row's legal prefix only."""

    columns = torch.arange(logits.shape[1]).unsqueeze(0)
    legal = columns < batch.legal_counts.unsqueeze(1)
    legal_logits = logits.masked_fill(~legal, -1.0e9)
    row_ce = -(batch.target * F.log_softmax(legal_logits, dim=1)).sum(dim=1)
    hits = legal_logits.argmax(dim=1) == batch.target.argmax(dim=1)
    return row_ce, hits


@torch.no_grad()
def evaluate(
    model: TypedMicroNet,
    positions: Sequence[Position],
    plan: Sequence[Sequence[int]],
    pad_quantum: int,
) -> Metrics:
    """Evaluate every frozen validation row once."""

    model.eval()
    ce_sum = 0.0
    hits = 0
    rows = 0
    for indices in plan:
        batch = collate(positions, indices, pad_quantum)
        row_ce, row_hits = _loss_and_hits(
            model(batch.feats, batch.nbr, batch.mask), batch
        )
        ce_sum += float(row_ce.sum())
        hits += int(row_hits.sum())
        rows += len(indices)
    return Metrics(ce=ce_sum / rows, top1=hits / rows, rows=rows)


@torch.no_grad()
def _clip_grad_norm(parameters: Iterable[nn.Parameter], maximum: float) -> float:
    """Clip a global L2 gradient norm using only eager CPU tensor operations."""

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    squared = sum(float(gradient.detach().square().sum()) for gradient in gradients)
    norm = math.sqrt(squared)
    if math.isfinite(norm) and norm > maximum:
        scale = maximum / (norm + 1.0e-12)
        for gradient in gradients:
            gradient.mul_(scale)
    return norm


def run_arm(
    name: str,
    signature: Signature,
    train: Sequence[Position],
    val: Sequence[Position],
    train_plan: Sequence[Sequence[int]],
    val_plan: Sequence[Sequence[int]],
    *,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    pad_quantum: int,
    seed: int,
) -> ArmResult:
    """Train and evaluate one arm under an already-frozen batch plan."""

    torch.manual_seed(seed)
    model = TypedMicroNet(signature)
    common_init_digest = reset_common_initialization(model, seed)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("model construction produced a non-CPU parameter")
    stored, effective = parameter_counts(model)
    optimizer = CPUAdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    initial = evaluate(model, val, val_plan, pad_quantum)

    model.train()
    started = time.perf_counter()
    report_every = max(1, len(train_plan) // 4)
    for step, indices in enumerate(train_plan, start=1):
        batch = collate(train, indices, pad_quantum)
        optimizer.zero_grad()
        logits = model(batch.feats, batch.nbr, batch.mask)
        row_ce, _ = _loss_and_hits(logits, batch)
        loss = row_ce.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} produced non-finite loss at step {step}")
        loss.backward()
        _clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == len(train_plan):
            print(
                f"  {name} step {step:3d}/{len(train_plan)} "
                f"train_ce={float(loss.detach()):.5f}",
                flush=True,
            )
    train_seconds = time.perf_counter() - started
    final = evaluate(model, val, val_plan, pad_quantum)
    return ArmResult(
        name=name,
        signature=name,
        stored_params=stored,
        effective_params=effective,
        initial=initial,
        final=final,
        train_seconds=train_seconds,
        steps=len(train_plan),
        common_init_digest=common_init_digest,
    )


def _assert_experiment_contract() -> None:
    """Pin the fixed-width/matched-compute and CPU/no-Triton contract."""

    if NUM_FEATURES != 25:
        raise RuntimeError(f"feature environment produced NUM_FEATURES={NUM_FEATURES}, not 25")
    if signature_width(ATTN_SIGNATURE) != WIDTH or ATTN_K != 8:
        raise AssertionError("attention interior must be reg:8 at width 96")
    if HEADS * HEAD_DIM != WIDTH:
        raise AssertionError("attention must use three 32-wide structural heads")
    if signature_width(POLICY_EXPAND_SIGNATURE) != 2 * WIDTH:
        raise AssertionError("policy expansion must have dense width 192")
    if signature_instances(POLICY_EXPAND_SIGNATURE) != POLICY_POOLED_WIDTH:
        raise AssertionError("policy pooled width mismatch")
    for name, signature in ARM_SIGNATURES:
        if signature_width(signature) != WIDTH:
            raise AssertionError(f"{name} width is not 96")
    imported_triton = sorted(
        module for module in sys.modules if module == "triton" or module.startswith("triton.")
    )
    if imported_triton:
        raise RuntimeError(f"Triton was imported unexpectedly: {imported_triton[:3]}")


def _assert_no_triton_imported() -> None:
    """Prove that the complete load/train/eval path never imported Triton."""

    imported = sorted(
        module for module in sys.modules if module == "triton" or module.startswith("triton.")
    )
    if imported:
        raise RuntimeError(f"CPU experiment imported Triton: {imported[:5]}")


def _node_summary(positions: Sequence[Position]) -> str:
    nodes = [position.nodes for position in positions]
    return (
        f"min={min(nodes)}, median={statistics.median(nodes):.1f}, "
        f"max={max(nodes)}, mean={statistics.fmean(nodes):.1f}"
    )


def _print_results(results: Sequence[ArmResult]) -> None:
    """Emit a markdown-friendly table plus machine-readable JSON."""

    print("\n| arm | stored params | effective params | initial val CE | final val CE | "
          "initial top1 | final top1 | train s |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for result in results:
        print(
            f"| `{result.name}` | {result.stored_params:,} | {result.effective_params:,} | "
            f"{result.initial.ce:.6f} | {result.final.ce:.6f} | "
            f"{result.initial.top1:.4f} | {result.final.top1:.4f} | "
            f"{result.train_seconds:.3f} |",
            flush=True,
        )
    payload = [
        {
            "arm": result.name,
            "stored_params": result.stored_params,
            "effective_params": result.effective_params,
            "initial_val_ce": result.initial.ce,
            "final_val_ce": result.final.ce,
            "initial_val_top1": result.initial.top1,
            "final_val_top1": result.final.top1,
            "train_seconds": result.train_seconds,
            "steps": result.steps,
            "common_init_digest": result.common_init_digest,
        }
        for result in results
    ]
    print("\nRESULTS_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data",
        help="train/val tree, samples dir with .buffer_manifest.json, manifest, or shard",
    )
    source.add_argument(
        "--shards",
        nargs="+",
        help="one or more compact-v4 shard files, directories, manifests, or globs",
    )
    parser.add_argument("--train-rows", type=int, default=256, help="default 256, max 1024")
    parser.add_argument("--val-rows", type=int, default=64, help="default 64, max 256")
    parser.add_argument("--steps", type=int, default=30, help="optimizer steps per arm; max 100")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--node-cap", type=int, default=384, help="filter, never truncate, larger rows")
    parser.add_argument("--bucket-width", type=int, default=32)
    parser.add_argument("--pad-quantum", type=int, default=8)
    parser.add_argument("--candidate-factor", type=int, default=4)
    parser.add_argument("--policy-target", choices=("visit", "gumbel"), default="gumbel")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="force <=16 train rows, <=8 val rows, <=2 steps, batch <=2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.train_rows = min(args.train_rows, 16)
        args.val_rows = min(args.val_rows, 8)
        args.steps = min(args.steps, 2)
        args.batch_size = min(args.batch_size, 2)
    if not 1 <= args.train_rows <= 1024:
        raise SystemExit("--train-rows must be in [1,1024]")
    if not 1 <= args.val_rows <= 256:
        raise SystemExit("--val-rows must be in [1,256]")
    if not 1 <= args.steps <= 100:
        raise SystemExit("--steps must be in [1,100]")
    for name in ("batch_size", "node_cap", "bucket_width", "pad_quantum", "candidate_factor", "threads"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    _assert_experiment_contract()

    train_refs, val_refs, source_description = discover_sources(args.data, args.shards)
    print("=== quotient CPU micro-prefit A/B ===", flush=True)
    print(
        "command           : "
        + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        flush=True,
    )
    print(f"source            : {source_description}", flush=True)
    print(
        f"source shards     : {len(train_refs)} train/source"
        + (f", {len(val_refs)} held-out" if val_refs is not None else " (row-level split)"),
        flush=True,
    )
    print("features/support  : compact-v4, 25 planes, support radius 4", flush=True)
    print(
        "architecture      : C=96, CCA, regular attention K=8, "
        "heads=3x32, 6 joint tokens",
        flush=True,
    )
    print("shape contract     : PASS (all 3 arms C=96; q/k/v reg:8; expand C=192)", flush=True)
    print("joint bias        : zero (intentional capacity-only probe)", flush=True)
    print("numerics/device   : fp32 CPU only; no production model/Triton/CUDA API", flush=True)
    print(
        f"rows/steps        : train={args.train_rows}, val={args.val_rows}, "
        f"steps={args.steps}, batch={args.batch_size}, node_cap={args.node_cap}",
        flush=True,
    )

    load_started = time.perf_counter()
    train, val, selection_stats = load_positions(
        train_refs,
        val_refs,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        candidate_factor=args.candidate_factor,
        node_cap=args.node_cap,
        target_kind=args.policy_target,
        seed=args.seed,
    )
    print(f"data load/expand  : {time.perf_counter() - load_started:.3f}s", flush=True)
    print(f"selection         : {selection_stats}", flush=True)
    print(
        "split integrity   : PASS ("
        + (
            "disjoint train/val shard paths"
            if val_refs is not None
            else "disjoint sampled global row indices"
        )
        + ")",
        flush=True,
    )
    print(f"train nodes       : {_node_summary(train)}", flush=True)
    print(f"val nodes         : {_node_summary(val)}", flush=True)

    train_plan = make_batch_plan(
        train,
        steps=args.steps,
        batch_size=args.batch_size,
        bucket_width=args.bucket_width,
        seed=args.seed,
    )
    val_plan = _validation_plan(val, args.batch_size, args.bucket_width)
    plan_digest = hashlib.sha256(
        json.dumps(train_plan, separators=(",", ":")).encode("ascii")
    ).hexdigest()[:16]
    print(f"shared batch plan : sha256={plan_digest}; exact same rows/order for all arms", flush=True)

    # Dense materialized shapes are independent of the quotient tie.  This
    # per-cell count excludes the likewise-identical joint qk/av quadratic term.
    dense_macs = (
        7 * 25 * WIDTH
        + 4 * 7 * WIDTH * WIDTH
        + 8 * WIDTH * WIDTH
        + 7 * WIDTH * WIDTH
        + WIDTH * (2 * WIDTH)
        + POLICY_POOLED_WIDTH
    )
    print(f"matched dense MACs: {dense_macs:,}/cell + identical joint qk/av", flush=True)

    results: list[ArmResult] = []
    for name, signature in ARM_SIGNATURES:
        print(f"\n--- arm {name} ---", flush=True)
        results.append(
            run_arm(
                name,
                signature,
                train,
                val,
                train_plan,
                val_plan,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                pad_quantum=args.pad_quantum,
                seed=args.seed,
            )
        )
    init_digests = {result.common_init_digest for result in results}
    if len(init_digests) != 1:
        raise AssertionError(f"same-shaped common initialization differs: {init_digests}")
    print(
        f"\ncommon init       : PASS sha256={next(iter(init_digests))} "
        "(raw stem + final readout; typed coefficients differ by construction)",
        flush=True,
    )
    _assert_no_triton_imported()
    print("CPU import audit  : PASS (all tensors CPU; no Triton module imported)", flush=True)
    _print_results(results)


if __name__ == "__main__":
    main()
