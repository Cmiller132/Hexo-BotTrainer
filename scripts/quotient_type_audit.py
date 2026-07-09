#!/usr/bin/env python3
"""CPU-only G7 quotient-type audit for a trained ``hexfield_eq`` checkpoint.

The residual stream of the production model is a slot-major regular D6 fiber,
``channel = slot * C_ORBIT + orbit_channel``.  For a subgroup ``H <= D6`` the
orthogonal projection onto the quotient subspace ``R[G/H]`` is right averaging,

    (P_H v)[g] = (1 / |H|) * sum_{h in H} v[g h].

This script measures ``||P_H v||^2 / ||v||^2`` at every trunk depth for
``H in {G, K, <sigma>, <rot180>}``, separately for cells and register tokens.
It reads live shards and checkpoints strictly read-only and writes only when an
explicit ``--output`` path is supplied.

Phase-A specification: ``docs/quotient_reps/PHASE_A_CPU_PROOF_SPEC.md`` G7.
Derivation: ``docs/quotient_reps/DERIVATION_QUOTIENT_REPS.md``.
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
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# The audit must not create __pycache__ files in the live source tree.
sys.dont_write_bytecode = True

import numpy as np
import torch


SEED = 0
GROUP_ORDER = 12
ORBIT_CHANNELS = 16
CHANNELS = GROUP_ORDER * ORBIT_CHANNELS
NUM_TOKENS = 6
DEFAULT_POSITIONS = 512
DEFAULT_BATCH_SIZE = 8
VALUE_BINS = 65

ARCH_ENV_RELATIVE = Path("scripts/prefit_env/hexfield_eq_arm4_raylayout.env")
ARM4_ENV_EXPECTED = {
    "HEXFIELD_EQ_CHANNELS": "192",
    "HEXFIELD_EQ_GROUP_ORDER": "12",
    "HEXFIELD_EQ_C_ORBIT": "16",
    "HEXFIELD_EQ_ATTENTION_HEADS": "3",
    "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
    "HEXFIELD_EQ_TRUNK": "CCLACCLACLA",
    "HEXFIELD_EQ_REG_LANE": "1",
    "HEXFIELD_EQ_REG_TOK_READ": "0",
}
CPU_DISABLED_ENVS = (
    "HEXFIELD_TRITON_CONV",
    "HEXFIELD_TRITON_CONV_LN",
    "HEXFIELD_TRITON_ATTN",
    "HEXFIELD_EQ_TRITON_RAY",
    "HEXFIELD_SERVE_FLEX",
    "HEXFIELD_TRAIN_FLEX",
    "HEXFIELD_FLEX_PAIR",
    "HEXFIELD_TRAIN_FLEX_PAIR",
)
FLEX_MODULE = "torch.nn.attention.flex_attention"
HISTOGRAM_EDGES = (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
ARCH_META_KEYS = (
    "group_order",
    "c_orbit",
    "channels",
    "in_channels",
    "attention_heads",
    "trunk_layout",
    "num_tokens",
    "feature_width",
    "equivariant",
    "reg_lane",
    "reg_tok_read",
    "support_radius",
    "bias_reduction",
    "bias_joint_classes",
    "ray_heads",
    "ray_blockers",
)


def _triton_module_names() -> list[str]:
    return sorted(
        name for name in sys.modules if name == "triton" or name.startswith("triton.")
    )


def _install_cpu_import_guard() -> None:
    """Prevent production-model import from pulling FlexAttention/Triton.

    ``model.py`` imports ``torch.nn.attention.flex_attention`` even when its
    runtime flex flags are disabled.  On GPU-enabled PyTorch wheels that import
    loads Triton and can probe for a CUDA installation.  Placing ``None`` in
    ``sys.modules`` makes the production module's guarded import raise
    ``ModuleNotFoundError``; its existing fallback sets ``_flex_attention=None``
    and uses the materialized CPU reference path.
    """

    if "hexfield_eq.model" in sys.modules:
        raise RuntimeError(
            "hexfield_eq.model was imported before the CPU guard; run this audit "
            "as a fresh process"
        )
    loaded = _triton_module_names()
    if loaded:
        raise RuntimeError(f"Triton was imported before the CPU guard: {loaded[:5]}")
    for name in CPU_DISABLED_ENVS:
        os.environ[name] = "0"
    sys.modules[FLEX_MODULE] = None
    loaded = _triton_module_names()
    if loaded:
        raise RuntimeError(f"CPU guard unexpectedly imported Triton: {loaded[:5]}")


# Install before any import that can transitively reach hexfield_eq.model.
_install_cpu_import_guard()


@dataclass(frozen=True)
class ShardInfo:
    """One committed compact shard and its row cardinality."""

    path: Path
    rows: int
    generation: int | None
    game_key: int | None
    source_root: Path


@dataclass
class AuditPosition:
    """One fully featurized decision state, held only in CPU memory."""

    support: Any
    feats: np.ndarray
    raylen: np.ndarray
    source: str
    source_row: int
    turn_index: int


@dataclass(frozen=True)
class RuntimeAPI:
    """Late imports made only after architecture env and CPU guards are set."""

    reps: Any
    HexfieldNet: Any
    collate_rows: Any
    build_position: Any
    build_ray_lengths: Any
    read_compact_shard: Any
    num_features: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_packages_to_path(root: Path) -> None:
    for relative in (
        Path("packages/hexfield_eq/python"),
        Path("packages/hexo_engine/python"),
        Path("packages/hexo_utils/python"),
    ):
        path = str((root / relative).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"architecture env not found: {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty env name")
        if value[:1] in ("'", '"') and value[-1:] == value[:1]:
            value = value[1:-1]
        values[key] = value
    return values


def _configure_arch_env(path: Path) -> dict[str, str]:
    values = _parse_env_file(path)
    missing = sorted(set(ARM4_ENV_EXPECTED) - set(values))
    if missing:
        raise ValueError(f"arm-4 env is missing required keys: {missing}")
    wrong_file = {
        key: (values[key], expected)
        for key, expected in ARM4_ENV_EXPECTED.items()
        if values[key] != expected
    }
    if wrong_file:
        raise ValueError(f"arm-4 env file has unexpected architecture values: {wrong_file}")
    for key, expected in ARM4_ENV_EXPECTED.items():
        existing = os.environ.get(key)
        if existing is not None and existing != expected:
            raise ValueError(
                f"process env {key}={existing!r} disagrees with arm-4 value {expected!r}"
            )
        os.environ[key] = expected

    # The audited checkpoint is the 25-plane v1 model.  This env is absent from
    # the historical arm-4 file because v1 was then the only feature map.
    feature_version = os.environ.get("HEXFIELD_EQ_FEATURE_VERSION", "1")
    if feature_version != "1":
        raise ValueError(
            "G7 checkpoint requires HEXFIELD_EQ_FEATURE_VERSION=1 (25 planes), "
            f"got {feature_version!r}"
        )
    os.environ["HEXFIELD_EQ_FEATURE_VERSION"] = "1"

    ray_blockers = os.environ.get("HEXFIELD_EQ_RAY_BLOCKERS", "1")
    if ray_blockers != "1":
        raise ValueError(
            f"arm-4 checkpoint requires HEXFIELD_EQ_RAY_BLOCKERS=1, got {ray_blockers!r}"
        )
    os.environ["HEXFIELD_EQ_RAY_BLOCKERS"] = "1"
    return {key: os.environ[key] for key in ARM4_ENV_EXPECTED}


def _import_runtime(root: Path) -> RuntimeAPI:
    _add_repo_packages_to_path(root)
    from hexfield_eq import reps
    from hexfield_eq.batching import collate_rows
    from hexfield_eq.constants import NUM_FEATURES
    from hexfield_eq.features import build_position, build_ray_lengths
    from hexfield_eq.model import HexfieldNet
    from hexfield_eq.shards import read_compact_shard

    if NUM_FEATURES != 25:
        raise RuntimeError(f"audit requires NUM_FEATURES=25, imported {NUM_FEATURES}")
    model_module = sys.modules[HexfieldNet.__module__]
    if getattr(model_module, "_flex_attention", "not-present") is not None:
        raise RuntimeError("CPU guard failed: production FlexAttention is active")
    loaded = _triton_module_names()
    if loaded:
        raise RuntimeError(f"CPU contract violated: Triton modules loaded: {loaded[:5]}")
    return RuntimeAPI(
        reps=reps,
        HexfieldNet=HexfieldNet,
        collate_rows=collate_rows,
        build_position=build_position,
        build_ray_lengths=build_ray_lengths,
        read_compact_shard=read_compact_shard,
        num_features=NUM_FEATURES,
    )


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def _resolve_checkpoint(raw: Path) -> Path:
    path = raw.expanduser()
    if path.is_file():
        return path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"checkpoint path is neither a file nor directory: {path}")
    search_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    candidates = [
        item
        for item in search_dir.glob("epoch_*.pt")
        if item.is_file() and _checkpoint_epoch(item) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"no epoch_*.pt checkpoints under {search_dir}")
    return max(candidates, key=lambda item: (_checkpoint_epoch(item), item.name)).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    for key in ("meta", "model"):
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"checkpoint has no dict payload[{key!r}]")
    return payload


def _build_model(runtime: RuntimeAPI, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    model = runtime.HexfieldNet()
    checkpoint_meta = payload["meta"]
    model_meta = model.arch_meta()
    missing_required = [key for key in ARCH_META_KEYS if key not in model_meta]
    if missing_required:
        raise ValueError(f"arm-4 model omitted required architecture metadata: {missing_required}")
    missing_meta = [key for key in model_meta if key not in checkpoint_meta]
    if missing_meta:
        raise ValueError(f"checkpoint is missing architecture metadata: {missing_meta}")
    # Compare every architecture field emitted by the current model, not merely
    # today's required subset.  This fails loudly if a future import-time knob is
    # added to arch_meta but absent/different in the audited checkpoint.
    expected_subset = dict(model_meta)
    actual_subset = {key: checkpoint_meta[key] for key in model_meta}
    if actual_subset != expected_subset:
        differences = {
            key: {"checkpoint": actual_subset[key], "arm4_build": expected_subset[key]}
            for key in model_meta
            if actual_subset[key] != expected_subset[key]
        }
        raise ValueError(f"checkpoint/arm-4 architecture mismatch: {differences}")
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(
            f"strict load reported missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    model.set_attention_impl("materialized")
    non_cpu = [name for name, tensor in model.state_dict().items() if tensor.device.type != "cpu"]
    if non_cpu:
        raise RuntimeError(f"model contains non-CPU tensors: {non_cpu[:5]}")
    return model, model_meta


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"compact shard has no committed JSON sidecar: {path}")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if data.get("schema") != "hexfield_compact_v1":
        raise ValueError(f"unsupported shard schema in {sidecar}: {data.get('schema')!r}")
    rows = int(data.get("rows", -1))
    if rows <= 0:
        raise ValueError(f"invalid row count in {sidecar}: {rows}")
    return data


def _manifest_shards(root: Path, checkpoint_epoch: int | None) -> list[ShardInfo]:
    manifest_path = root / ".buffer_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"manifest entries are not a list: {manifest_path}")
    result: list[ShardInfo] = []
    for entry in entries:
        generation = int(entry["generation"]) if entry.get("generation") is not None else None
        if checkpoint_epoch is not None and generation is not None and generation > checkpoint_epoch:
            continue
        path = (root / str(entry["rel_path"])).resolve()
        rows = int(entry["rows"])
        if rows <= 0:
            raise ValueError(f"manifest has invalid row count for {path}: {rows}")
        # The manifest is atomically written from committed (npz + sidecar)
        # entries.  Avoid reopening thousands of sidecars here; selected shards
        # are revalidated against both files immediately before decode.
        result.append(
            ShardInfo(
                path=path,
                rows=rows,
                generation=generation,
                game_key=int(entry["game_key"]) if entry.get("game_key") is not None else None,
                source_root=root.resolve(),
            )
        )
    return result


def _scan_shards(root: Path, checkpoint_epoch: int | None) -> list[ShardInfo]:
    result: list[ShardInfo] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.npz"))
    for path in paths:
        if path.suffix.lower() != ".npz" or not path.is_file():
            continue
        sidecar_path = path.with_suffix(".json")
        if not sidecar_path.is_file():
            if root.is_file():
                raise FileNotFoundError(f"explicit shard lacks sidecar: {path}")
            continue
        sidecar = _read_sidecar(path)
        generation = int(sidecar["epoch"]) if sidecar.get("epoch") is not None else None
        if checkpoint_epoch is not None and generation is not None and generation > checkpoint_epoch:
            continue
        result.append(
            ShardInfo(
                path=path.resolve(),
                rows=int(sidecar["rows"]),
                generation=generation,
                game_key=int(sidecar["game_key"]) if sidecar.get("game_key") is not None else None,
                source_root=(root.parent if root.is_file() else root).resolve(),
            )
        )
    return result


def _expand_shard_args(arguments: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        if glob.has_magic(argument):
            matches = sorted(glob.glob(argument, recursive=True))
            if not matches:
                raise FileNotFoundError(f"shard pattern matched nothing: {argument}")
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(argument).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"shard path does not exist: {path}")
            paths.append(path)
    return paths


def _discover_shards(arguments: Sequence[str], checkpoint_epoch: int | None) -> list[ShardInfo]:
    discovered: list[ShardInfo] = []
    for path in _expand_shard_args(arguments):
        if path.is_dir() and (path / ".buffer_manifest.json").is_file():
            discovered.extend(_manifest_shards(path, checkpoint_epoch))
        else:
            discovered.extend(_scan_shards(path, checkpoint_epoch))
    by_path: dict[Path, ShardInfo] = {}
    for info in discovered:
        prior = by_path.get(info.path)
        if prior is not None and prior.rows != info.rows:
            raise ValueError(f"duplicate shard has conflicting row counts: {info.path}")
        by_path[info.path] = info
    result = sorted(by_path.values(), key=lambda info: str(info.path))
    if not result:
        raise ValueError("no committed hexfield_eq compact shards were discovered")
    return result


def _sample_row_indices(shards: Sequence[ShardInfo], count: int) -> dict[int, list[int]]:
    total_rows = sum(info.rows for info in shards)
    if count <= 0:
        raise ValueError("--positions must be positive")
    if count > total_rows:
        raise ValueError(f"requested {count} positions from only {total_rows} shard rows")
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(range(total_rows), count))
    cumulative: list[int] = []
    running = 0
    for info in shards:
        running += info.rows
        cumulative.append(running)
    selections: dict[int, list[int]] = defaultdict(list)
    for global_row in chosen:
        shard_index = bisect.bisect_right(cumulative, global_row)
        previous = cumulative[shard_index - 1] if shard_index else 0
        selections[shard_index].append(global_row - previous)
    return dict(selections)


def _load_real_positions(
    runtime: RuntimeAPI,
    shards: Sequence[ShardInfo],
    count: int,
) -> list[AuditPosition]:
    selections = _sample_row_indices(shards, count)
    positions: list[AuditPosition] = []
    for shard_index in sorted(selections):
        info = shards[shard_index]
        if not info.path.is_file():
            raise FileNotFoundError(f"selected manifest shard is missing: {info.path}")
        sidecar = _read_sidecar(info.path)
        if int(sidecar["rows"]) != info.rows:
            raise ValueError(
                f"manifest/sidecar row mismatch for {info.path}: "
                f"{info.rows} != {sidecar['rows']}"
            )
        rows = runtime.read_compact_shard(info.path)
        if len(rows) != info.rows:
            raise ValueError(
                f"decoded row count differs from sidecar for {info.path}: "
                f"{len(rows)} != {info.rows}"
            )
        for row_index in selections[shard_index]:
            sample = rows[row_index]
            facts = sample.facts()
            support, feats = runtime.build_position(facts)
            raylen = runtime.build_ray_lengths(facts, support)
            if feats.shape != (support.num_nodes, runtime.num_features):
                raise AssertionError(f"bad feature shape for {info.path} row {row_index}")
            positions.append(
                AuditPosition(
                    support=support,
                    feats=feats,
                    raylen=raylen,
                    source=str(info.path),
                    source_row=row_index,
                    turn_index=int(sample.turn_index),
                )
            )
    if len(positions) != count:
        raise AssertionError(f"loaded {len(positions)} positions, expected {count}")
    return positions


def _load_random_positions(runtime: RuntimeAPI, count: int) -> list[AuditPosition]:
    """Generate deterministic uniform-random legal prefixes through the engine."""

    if count <= 0:
        raise ValueError("--random-prefixes must be positive")
    from hexfield_eq.engine_facts import facts_from_state
    from hexfield_eq.geometry import unpack_action_id
    from hexo_engine import api
    from hexo_engine.errors import EngineUnavailableError
    from hexo_engine.types import AxialCoord, PlacementAction

    rng = random.Random(SEED)
    positions: list[AuditPosition] = []
    for position_index in range(count):
        try:
            state = api.new_game()
        except EngineUnavailableError as exc:
            raise RuntimeError(
                "random-prefix fallback requires the hexo_engine Rust extension; "
                "use --shards or run under the documented WSL build venv"
            ) from exc
        target_plies = rng.randint(1, 96)
        for _ in range(target_plies):
            action_ids = api.legal_action_ids(state)
            if not action_ids:
                break
            before = api.clone_state(state)
            q, r = unpack_action_id(int(rng.choice(action_ids)))
            result = api.apply_action(
                state, PlacementAction(AxialCoord(q=int(q), r=int(r)))
            )
            if result.terminal:
                state = before
                break
        if api.terminal(state) is not None:
            raise AssertionError("random-prefix fallback retained a terminal state")
        facts = facts_from_state(state)
        support, feats = runtime.build_position(facts)
        raylen = runtime.build_ray_lengths(facts, support)
        positions.append(
            AuditPosition(
                support=support,
                feats=feats,
                raylen=raylen,
                source=f"random-prefix-{position_index:06d}",
                source_row=0,
                turn_index=int(facts.placements_made),
            )
        )
    return positions


class EnergyAccumulator:
    """Fp64 energy sums and per-channel distributions for regular fibers."""

    def __init__(self, name: str, cosets: dict[str, tuple[tuple[int, ...], ...]]) -> None:
        self.name = name
        self.cosets = cosets
        self.denominator = torch.zeros(ORBIT_CHANNELS, dtype=torch.float64)
        self.numerators = {
            key: torch.zeros(ORBIT_CHANNELS, dtype=torch.float64) for key in cosets
        }
        self.macro_sums = {key: 0.0 for key in cosets}
        self.macro_counts = {key: 0 for key in cosets}
        self.site_count = 0

    def update(self, activation: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if activation.device.type != "cpu":
            raise RuntimeError(f"{self.name}: activation is not on CPU")
        if activation.ndim != 3 or activation.shape[-1] != CHANNELS:
            raise ValueError(
                f"{self.name}: expected (B,S,{CHANNELS}), got {tuple(activation.shape)}"
            )
        flat = activation.detach()
        if mask is not None:
            if mask.device.type != "cpu" or tuple(mask.shape) != tuple(flat.shape[:2]):
                raise ValueError(f"{self.name}: mask/activation shape mismatch")
            flat = flat[mask]
        else:
            flat = flat.reshape(-1, CHANNELS)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError(f"{self.name}: non-finite activation")
        fibers = flat.reshape(-1, GROUP_ORDER, ORBIT_CHANNELS).to(torch.float64)
        self.site_count += int(fibers.shape[0])
        denominator_by_vector = fibers.square().sum(dim=1)
        self.denominator += denominator_by_vector.sum(dim=0)
        for key, blocks in self.cosets.items():
            numerator_by_vector = torch.zeros_like(denominator_by_vector)
            for block in blocks:
                mean = fibers[:, list(block), :].mean(dim=1)
                numerator_by_vector += len(block) * mean.square()
            self.numerators[key] += numerator_by_vector.sum(dim=0)
            valid = denominator_by_vector > 0
            if bool(valid.any()):
                ratios = numerator_by_vector[valid] / denominator_by_vector[valid]
                self.macro_sums[key] += float(ratios.sum().item())
                self.macro_counts[key] += int(ratios.numel())

    def finalize(self) -> dict[str, Any]:
        if self.site_count <= 0:
            raise AssertionError(f"{self.name}: no activation sites were observed")
        total_denominator = float(self.denominator.sum().item())
        if not math.isfinite(total_denominator) or total_denominator <= 0:
            raise AssertionError(f"{self.name}: non-positive activation energy")
        tolerance = 1e-10 * torch.clamp(self.denominator, min=1.0)

        def _assert_le(left: str, right: str) -> None:
            violation = self.numerators[left] - self.numerators[right] - tolerance
            if bool((violation > 0).any()):
                index = int(torch.argmax(violation).item())
                raise AssertionError(
                    f"{self.name}: nesting {left}<={right} failed at orbit channel {index}"
                )

        _assert_le("G", "K")
        _assert_le("K", "mirror")
        _assert_le("K", "point")
        for key, numerator in self.numerators.items():
            if bool((numerator - self.denominator - tolerance > 0).any()):
                raise AssertionError(f"{self.name}: E_{key} exceeds 1")

        nonzero = self.denominator > 0
        active_channels = int(nonzero.sum().item())
        overall: dict[str, float] = {}
        per_channel: dict[str, list[float]] = {}
        macro: dict[str, float] = {}
        for key, numerator in self.numerators.items():
            overall[key] = float(numerator.sum().item() / total_denominator)
            ratios = torch.full_like(self.denominator, float("nan"))
            ratios[nonzero] = numerator[nonzero] / self.denominator[nonzero]
            per_channel[key] = [float(value) for value in ratios.tolist()]
            count = self.macro_counts[key]
            macro[key] = self.macro_sums[key] / count if count else float("nan")
        return {
            "name": self.name,
            "sites": self.site_count,
            "fiber_vectors": self.site_count * ORBIT_CHANNELS,
            "active_channels": active_channels,
            "overall": overall,
            "macro": macro,
            "per_channel": per_channel,
            "nesting": "PASS",
        }


class ActivationAudit:
    """Forward-hook orchestration for cells, tokens, and pre/post-final streams."""

    def __init__(
        self,
        model: Any,
        layout: str,
        cosets: dict[str, tuple[tuple[int, ...], ...]],
    ) -> None:
        self.model = model
        self.layout = layout
        self.cosets = cosets
        self.current_mask: torch.Tensor | None = None
        self.cell_order = ["stem"] + [
            f"depth_{depth:02d}_{kind}" for depth, kind in enumerate(layout)
        ] + ["pre_ln_final", "post_ln_final"]
        self.token_order = [
            f"depth_{depth:02d}_{kind}" for depth, kind in enumerate(layout)
        ] + ["pre_ln_final", "post_ln_final"]
        self.cells = {
            name: EnergyAccumulator(f"cells/{name}", cosets) for name in self.cell_order
        }
        self.tokens = {
            name: EnergyAccumulator(f"tokens/{name}", cosets) for name in self.token_order
        }
        self.handles: list[Any] = []
        self._register_hooks()

    def _mask(self) -> torch.Tensor:
        if self.current_mask is None:
            raise RuntimeError("activation hook fired outside an audit batch")
        return self.current_mask

    def _register_hooks(self) -> None:
        first_block: Any | None = None
        conv_index = ray_index = attn_index = 0
        for depth, kind in enumerate(self.layout):
            stream = f"depth_{depth:02d}_{kind}"
            if kind == "C":
                block = self.model.conv_blocks[conv_index]
                register = self.model.registers[conv_index]
                conv_index += 1
                if first_block is None:
                    first_block = block
                self.handles.append(block.register_forward_hook(self._cell_hook(stream)))
                self.handles.append(register.register_forward_hook(self._token_hook(stream)))
            elif kind == "L":
                block = self.model.ray_blocks[ray_index]
                register = self.model.registers_l[ray_index]
                ray_index += 1
                if first_block is None:
                    first_block = block
                self.handles.append(block.register_forward_hook(self._cell_hook(stream)))
                self.handles.append(register.register_forward_hook(self._token_hook(stream)))
            elif kind == "A":
                block = self.model.attn_blocks[attn_index]
                attn_index += 1
                if first_block is None:
                    first_block = block
                self.handles.append(block.register_forward_hook(self._joint_hook(stream)))
            else:
                raise ValueError(f"unsupported trunk kind {kind!r}")
        if first_block is None:
            raise ValueError("empty trunk layout")
        self.handles.append(first_block.register_forward_pre_hook(self._stem_hook()))
        self.handles.append(
            self.model.ln_final.register_forward_pre_hook(self._pre_final_hook())
        )
        self.handles.append(self.model.ln_final.register_forward_hook(self._post_final_hook()))

    def _stem_hook(self):
        def hook(_module, args) -> None:
            self.cells["stem"].update(args[0], self._mask())

        return hook

    def _cell_hook(self, stream: str):
        def hook(_module, _args, output) -> None:
            self.cells[stream].update(output, self._mask())

        return hook

    def _token_hook(self, stream: str):
        def hook(_module, _args, output) -> None:
            self.tokens[stream].update(output)

        return hook

    def _joint_hook(self, stream: str):
        def hook(_module, _args, output) -> None:
            if output.shape[1] != self._mask().shape[1] + NUM_TOKENS:
                raise ValueError(f"{stream}: malformed joint-sequence length")
            self.tokens[stream].update(output[:, :NUM_TOKENS])
            self.cells[stream].update(output[:, NUM_TOKENS:], self._mask())

        return hook

    def _pre_final_hook(self):
        def hook(_module, args) -> None:
            sequence = args[0]
            self.tokens["pre_ln_final"].update(sequence[:, :NUM_TOKENS])
            self.cells["pre_ln_final"].update(
                sequence[:, NUM_TOKENS:], self._mask()
            )

        return hook

    def _post_final_hook(self):
        def hook(_module, _args, output) -> None:
            self.tokens["post_ln_final"].update(output[:, :NUM_TOKENS])
            self.cells["post_ln_final"].update(
                output[:, NUM_TOKENS:], self._mask()
            )

        return hook

    def run(
        self,
        positions: Sequence[AuditPosition],
        collate_rows: Any,
        batch_size: int,
        *,
        verbose: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        ordered = sorted(
            positions,
            key=lambda item: (
                item.support.num_nodes,
                item.turn_index,
                item.source,
                item.source_row,
            ),
        )
        try:
            with torch.inference_mode():
                for start in range(0, len(ordered), batch_size):
                    subset = ordered[start : start + batch_size]
                    batch = collate_rows(
                        [(item.support, item.feats) for item in subset],
                        raylen=[item.raylen for item in subset],
                    )
                    if any(tensor.device.type != "cpu" for tensor in batch.values()):
                        raise RuntimeError("collate_rows produced a non-CPU tensor")
                    self.current_mask = batch["mask"]
                    self.model.trunk(
                        batch["feats"],
                        batch["nbr"],
                        batch["mask"],
                        batch["coords"],
                        batch["raylen"],
                    )
                    self.current_mask = None
                    if verbose:
                        print(
                            f"audited {min(start + batch_size, len(ordered))}/{len(ordered)} positions",
                            file=sys.stderr,
                        )
        finally:
            self.current_mask = None
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    def finalize(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        cells = [self.cells[name].finalize() for name in self.cell_order]
        tokens = [self.tokens[name].finalize() for name in self.token_order]
        return cells, tokens


def _projection_cosets(runtime: RuntimeAPI) -> tuple[
    dict[str, tuple[tuple[int, ...], ...]], int, int
]:
    reps = runtime.reps
    sigma, rot180 = reps.distinguished_elements()
    if (sigma, rot180) != (7, 3):
        raise AssertionError(f"unexpected distinguished elements: {(sigma, rot180)}")
    names = {"G": "triv", "K": "axis", "mirror": "mirror", "point": "point"}
    cosets = {key: reps.quotient_cosets(type_name) for key, type_name in names.items()}
    expected = {
        "G": (tuple(range(12)),),
        "K": ((0, 3, 7, 10), (1, 4, 8, 11), (2, 5, 6, 9)),
        "mirror": ((0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 6)),
        "point": ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)),
    }
    if cosets != expected:
        raise AssertionError(f"right-coset pairing mismatch: {cosets}")
    return cosets, sigma, rot180


class RightSigmaProjector:
    """Fp32 right-``<sigma>`` averaging with independently checked semantics."""

    def __init__(
        self,
        runtime: RuntimeAPI,
        sigma: int,
        mirror_cosets: tuple[tuple[int, ...], ...],
    ) -> None:
        mult = runtime.reps.build_group()["mult"]
        gather = tuple(int(mult[g][sigma]) for g in range(GROUP_ORDER))
        if any(gather[gather[g]] != g for g in range(GROUP_ORDER)):
            raise AssertionError("right-sigma gather is not an involution")
        blocks = tuple(
            sorted(
                {tuple(sorted((g, gather[g]))) for g in range(GROUP_ORDER)},
                key=lambda block: (min(block), block),
            )
        )
        if blocks != mirror_cosets:
            raise AssertionError(
                f"right-sigma pairs disagree with mirror quotient cosets: {blocks}"
            )
        self.gather = torch.tensor(gather, dtype=torch.long)
        self.mirror_cosets = mirror_cosets
        self.runtime_idempotence_checked = False

        # This reference is deliberately block-based rather than expressed with
        # the gather formula used by _project.  It catches a left/right or slot-
        # ordering mistake before any checkpoint inference is attempted.
        probe = torch.arange(
            2 * GROUP_ORDER * ORBIT_CHANNELS, dtype=torch.float64
        ).reshape(2, GROUP_ORDER * ORBIT_CHANNELS)
        projected = self._project(probe)
        reference_fibers = probe.reshape(2, GROUP_ORDER, ORBIT_CHANNELS).clone()
        for block in mirror_cosets:
            mean = reference_fibers[:, list(block), :].mean(dim=1, keepdim=True)
            reference_fibers[:, list(block), :] = mean
        reference = reference_fibers.reshape_as(probe)
        torch.testing.assert_close(projected, reference, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            self._project(projected), projected, rtol=0.0, atol=0.0
        )

    def _project(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != CHANNELS:
            raise ValueError(
                f"right-sigma projection requires width {CHANNELS}, "
                f"got {activation.shape[-1]}"
            )
        gather = self.gather.to(activation.device)
        fibers = activation.reshape(*activation.shape[:-1], GROUP_ORDER, ORBIT_CHANNELS)
        projected = (fibers + fibers.index_select(-2, gather)) * 0.5
        return projected.reshape_as(activation)

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.device.type != "cpu":
            raise RuntimeError("causal mirror projection received a non-CPU activation")
        if activation.dtype != torch.float32:
            raise RuntimeError(
                f"causal mirror projection requires fp32 inference, got {activation.dtype}"
            )
        if not bool(torch.isfinite(activation).all()):
            raise ValueError("causal mirror projection received non-finite activations")
        projected = self._project(activation)
        if not self.runtime_idempotence_checked:
            torch.testing.assert_close(
                self._project(projected), projected, rtol=0.0, atol=0.0
            )
            self.runtime_idempotence_checked = True
        return projected


def _causal_depth_modules(model: Any, layout: str) -> list[tuple[str, str, Any]]:
    """Map every layout depth to the module whose output carries that stream."""

    result: list[tuple[str, str, Any]] = []
    conv_index = ray_index = attn_index = 0
    for depth, kind in enumerate(layout):
        if kind == "C":
            module = model.conv_blocks[conv_index]
            conv_index += 1
        elif kind == "L":
            module = model.ray_blocks[ray_index]
            ray_index += 1
        elif kind == "A":
            module = model.attn_blocks[attn_index]
            attn_index += 1
        else:
            raise ValueError(f"unsupported trunk kind {kind!r}")
        result.append((f"depth_{depth:02d}_{kind}", kind, module))
    if len(result) != 11:
        raise AssertionError(
            f"arm-4 causal audit requires all 11 trunk depths, got {len(result)}"
        )
    return result


def _distribution_drift(
    baseline_logits: torch.Tensor,
    projected_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return baseline/projected fp64 probabilities plus KL(base||new) and TV."""

    baseline_logp = torch.log_softmax(baseline_logits.to(torch.float64), dim=-1)
    projected_logp = torch.log_softmax(projected_logits.to(torch.float64), dim=-1)
    baseline_p = baseline_logp.exp()
    projected_p = projected_logp.exp()
    kl = (baseline_p * (baseline_logp - projected_logp)).sum()
    if float(kl.item()) < -1e-12:
        raise AssertionError(f"negative KL beyond fp64 tolerance: {float(kl.item())}")
    kl = kl.clamp_min(0.0)
    tv = 0.5 * (baseline_p - projected_p).abs().sum()
    return baseline_p, projected_p, kl, tv


class CausalMetricAccumulator:
    """Per-position causal drifts, computed and retained as fp64 scalars."""

    METRICS = (
        "policy_kl",
        "policy_tv",
        "top1_flip",
        "value_kl",
        "value_tv",
        "expected_value_abs_drift",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.values: dict[str, list[float]] = {key: [] for key in self.METRICS}

    def update(
        self,
        baseline: dict[str, torch.Tensor],
        projected: dict[str, torch.Tensor],
        legal_counts: torch.Tensor,
    ) -> None:
        for output_name in ("policy", "value"):
            for label, output in (("baseline", baseline), ("projected", projected)):
                tensor = output[output_name]
                if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise RuntimeError(
                        f"{self.name}: {label} {output_name} must be CPU fp32, "
                        f"got {tensor.device}/{tensor.dtype}"
                    )
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"{self.name}: non-finite {label} {output_name}")
        if baseline["policy"].shape != projected["policy"].shape:
            raise ValueError(f"{self.name}: policy output shape changed")
        if baseline["value"].shape != projected["value"].shape:
            raise ValueError(f"{self.name}: value output shape changed")
        if baseline["value"].ndim != 2 or baseline["value"].shape[1] != VALUE_BINS:
            raise ValueError(f"{self.name}: expected {VALUE_BINS} value bins")
        if baseline["policy"].shape[0] != legal_counts.numel():
            raise ValueError(f"{self.name}: legal-count batch mismatch")

        value_support = torch.linspace(-1.0, 1.0, VALUE_BINS, dtype=torch.float64)
        for row in range(int(legal_counts.numel())):
            legal_count = int(legal_counts[row].item())
            if not 0 < legal_count <= baseline["policy"].shape[1]:
                raise ValueError(f"{self.name}: invalid legal_count={legal_count}")
            base_policy_logits = baseline["policy"][row, :legal_count]
            new_policy_logits = projected["policy"][row, :legal_count]
            _, _, policy_kl, policy_tv = _distribution_drift(
                base_policy_logits, new_policy_logits
            )
            top1_flip = float(
                int(torch.argmax(base_policy_logits).item())
                != int(torch.argmax(new_policy_logits).item())
            )

            base_value, new_value, value_kl, value_tv = _distribution_drift(
                baseline["value"][row], projected["value"][row]
            )
            base_expected = torch.dot(base_value, value_support)
            new_expected = torch.dot(new_value, value_support)
            expected_drift = (base_expected - new_expected).abs()

            scalars = {
                "policy_kl": policy_kl,
                "policy_tv": policy_tv,
                "top1_flip": torch.tensor(top1_flip, dtype=torch.float64),
                "value_kl": value_kl,
                "value_tv": value_tv,
                "expected_value_abs_drift": expected_drift,
            }
            for key, value in scalars.items():
                if value.dtype != torch.float64 or not bool(torch.isfinite(value)):
                    raise AssertionError(f"{self.name}: invalid fp64 metric {key}={value}")
                self.values[key].append(float(value.item()))

    def finalize(self) -> dict[str, Any]:
        counts = {len(values) for values in self.values.values()}
        if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
            raise AssertionError(f"{self.name}: incomplete causal metric accumulation")
        count = next(iter(counts))
        result: dict[str, Any] = {"name": self.name, "positions": count}
        for key, values in self.values.items():
            result[key] = {
                "mean": math.fsum(values) / count,
                "median": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "max": max(values),
            }
        result["top1_flip_rate"] = result["top1_flip"]["mean"]
        return result


def _run_causal_mirror_audit(
    *,
    model: Any,
    runtime: RuntimeAPI,
    layout: str,
    positions: Sequence[AuditPosition],
    count: int,
    batch_size: int,
    sigma: int,
    mirror_cosets: tuple[tuple[int, ...], ...],
    verbose: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Run paired serve forwards with one right-mirror intervention at a time."""

    if count <= 0:
        raise ValueError("--causal-mirror-positions must be positive when enabled")
    if count > len(positions):
        raise ValueError(
            f"--causal-mirror-positions={count} exceeds the {len(positions)} "
            "loaded real positions; increase --positions"
        )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # Select before bucketing: these are precisely the first N positions in the
    # deterministic manifest-uniform real-row sample used by the required audit.
    selected = list(positions[:count])
    ordered = sorted(
        selected,
        key=lambda item: (
            item.support.num_nodes,
            item.turn_index,
            item.source,
            item.source_row,
        ),
    )
    depths = _causal_depth_modules(model, layout)
    projector = RightSigmaProjector(runtime, sigma, mirror_cosets)
    accumulators = {name: CausalMetricAccumulator(name) for name, _, _ in depths}
    started = time.perf_counter()

    def intervention_hook(_module, _args, output):
        if not torch.is_tensor(output):
            raise TypeError("causal trunk hook expected one tensor output")
        return projector(output)

    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            subset = ordered[start : start + batch_size]
            batch = runtime.collate_rows(
                [(item.support, item.feats) for item in subset],
                raylen=[item.raylen for item in subset],
            )
            if any(tensor.device.type != "cpu" for tensor in batch.values()):
                raise RuntimeError("causal collate_rows produced a non-CPU tensor")
            expected_legal = [int(item.support.legal_count) for item in subset]
            actual_legal = [int(value) for value in batch["legal_counts"].tolist()]
            if actual_legal != expected_legal:
                raise AssertionError(
                    f"causal legal-count ordering mismatch: {actual_legal} != {expected_legal}"
                )
            inputs = (
                batch["feats"],
                batch["nbr"],
                batch["mask"],
                batch["coords"],
                batch["raylen"],
            )
            baseline = model.forward_policy_value(*inputs)
            for name, _kind, module in depths:
                handle = module.register_forward_hook(intervention_hook)
                try:
                    projected = model.forward_policy_value(*inputs)
                finally:
                    handle.remove()
                accumulators[name].update(
                    baseline, projected, batch["legal_counts"]
                )
            if verbose:
                print(
                    f"causal mirror audited "
                    f"{min(start + batch_size, len(ordered))}/{len(ordered)} positions",
                    file=sys.stderr,
                )

    if not projector.runtime_idempotence_checked:
        raise AssertionError("causal projection hook never fired")
    summaries = [accumulators[name].finalize() for name, _, _ in depths]
    if any(summary["positions"] != count for summary in summaries):
        raise AssertionError("causal depths observed inconsistent position counts")
    return summaries, time.perf_counter() - started


def _run_weight_audit(
    model: Any,
    cosets: dict[str, tuple[tuple[int, ...], ...]],
) -> list[dict[str, Any]]:
    """Optional projection of each EquivLinear ``wb`` as functions on G."""

    summaries: list[dict[str, Any]] = []
    for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        wb = getattr(module, "wb", None)
        if wb is None or not torch.is_tensor(wb) or wb.ndim != 3 or wb.shape[0] != 12:
            continue
        fibers = wb.detach().to(torch.float64).permute(1, 2, 0).reshape(-1, 12)
        denominator = fibers.square().sum(dim=1)
        if not bool((denominator > 0).any()):
            continue
        result: dict[str, Any] = {"name": name, "vectors": int(fibers.shape[0])}
        ratios: dict[str, torch.Tensor] = {}
        numerators: dict[str, torch.Tensor] = {}
        for key, blocks in cosets.items():
            numerator = torch.zeros_like(denominator)
            for block in blocks:
                mean = fibers[:, list(block)].mean(dim=1)
                numerator += len(block) * mean.square()
            numerators[key] = numerator
            valid = denominator > 0
            ratios[key] = numerator[valid] / denominator[valid]
            result[key] = float(numerator.sum().item() / denominator.sum().item())
            result[f"{key}_per_vector"] = [float(value) for value in ratios[key].tolist()]
        tolerance = 1e-10 * torch.clamp(denominator, min=1.0)
        for left, right in (("G", "K"), ("K", "mirror"), ("K", "point")):
            if bool((numerators[left] - numerators[right] - tolerance > 0).any()):
                raise AssertionError(f"weight nesting failed for {name}: {left}<={right}")
        result["nesting"] = "PASS"
        summaries.append(result)
    if not summaries:
        raise AssertionError("--weight-audit found no EquivLinear.wb tensors")
    return summaries


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _histogram(values: Sequence[float]) -> str:
    counts = [0] * (len(HISTOGRAM_EDGES) - 1)
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            continue
        value = min(1.0, max(0.0, value))
        index = bisect.bisect_right(HISTOGRAM_EDGES, value) - 1
        index = min(max(index, 0), len(counts) - 1)
        counts[index] += 1
    labels: list[str] = []
    for index, count in enumerate(counts):
        left, right = HISTOGRAM_EDGES[index], HISTOGRAM_EDGES[index + 1]
        close = "]" if index == len(counts) - 1 else ")"
        labels.append(f"[{left:.2f},{right:.2f}{close}:{count}")
    return " ".join(labels)


def _format_float(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "nan"


def _position_stats(positions: Sequence[AuditPosition]) -> dict[str, Any]:
    nodes = [int(item.support.num_nodes) for item in positions]
    plies = [int(item.turn_index) for item in positions]
    return {
        "nodes": tuple(_quantile(nodes, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)),
        "plies": tuple(_quantile(plies, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)),
    }


def _energy_table(title: str, summaries: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Stream | Sites | Fiber vectors | E_G | E_K | E_mirror | E_point | "
        "macro E_mirror | Nesting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for summary in summaries:
        overall, macro = summary["overall"], summary["macro"]
        lines.append(
            f"| `{summary['name'].split('/', 1)[-1]}` | {summary['sites']} | "
            f"{summary['fiber_vectors']} | {_format_float(overall['G'])} | "
            f"{_format_float(overall['K'])} | {_format_float(overall['mirror'])} | "
            f"{_format_float(overall['point'])} | {_format_float(macro['mirror'])} | "
            f"{summary['nesting']} |"
        )
    lines.append("")
    return lines


def _distribution_table(title: str, summaries: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "Each distribution contains the 16 orbit-channel energy fractions. The "
        "histogram intervals are left-closed and right-open except the final bin.",
        "",
        "| Stream | H | Active | Min | Q25 | Median | Q75 | Max | >=0.70 | Histogram |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        stream = summary["name"].split("/", 1)[-1]
        for key in ("G", "K", "mirror", "point"):
            values = summary["per_channel"][key]
            quantiles = [_quantile(values, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
            finite = [value for value in values if math.isfinite(value)]
            above = sum(value >= 0.70 for value in finite)
            lines.append(
                f"| `{stream}` | {key} | {len(finite)}/{ORBIT_CHANNELS} | "
                + " | ".join(_format_float(value) for value in quantiles)
                + f" | {above}/{len(finite)} | `{_histogram(values)}` |"
            )
    lines.append("")
    return lines


def _weight_table(summaries: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "## Optional EquivLinear weight-space audit",
        "",
        "Each `wb[:, out_orbit, in_orbit]` is treated as a function on D6 and "
        "projected by the same right-H averages.",
        "",
        "| Module | Vectors | E_G | E_K | E_mirror | E_point | mirror Q25 | "
        "mirror median | mirror Q75 | Nesting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for summary in summaries:
        mirror = summary["mirror_per_vector"]
        lines.append(
            f"| `{summary['name']}` | {summary['vectors']} | "
            f"{_format_float(summary['G'])} | {_format_float(summary['K'])} | "
            f"{_format_float(summary['mirror'])} | {_format_float(summary['point'])} | "
            f"{_format_float(_quantile(mirror, 0.25))} | "
            f"{_format_float(_quantile(mirror, 0.50))} | "
            f"{_format_float(_quantile(mirror, 0.75))} | {summary['nesting']} |"
        )
    lines.append("")
    return lines


def _causal_stat_cell(statistics: dict[str, float]) -> str:
    return " / ".join(
        f"{statistics[key]:.3e}" for key in ("mean", "median", "p95", "max")
    )


def _causal_mirror_report(
    summaries: Sequence[dict[str, Any]],
    runtime_seconds: float,
) -> str:
    if len(summaries) != 11:
        raise AssertionError(f"expected 11 causal-depth summaries, got {len(summaries)}")
    policy_peak = max(summaries, key=lambda item: item["policy_tv"]["mean"])
    value_peak = max(summaries, key=lambda item: item["value_tv"]["mean"])
    flip_peak = max(summaries, key=lambda item: item["top1_flip_rate"])
    lines = [
        "## Causal right-mirror projection extension",
        "",
        "At each depth independently, one block output is replaced by "
        "`P_<sigma> = (I + R_sigma) / 2` and the complete "
        "`forward_policy_value` serve path finishes downstream. C/L hooks replace "
        "the cell output before register refresh; A hooks replace the whole joint "
        "token+cell output.",
        "",
        f"- Positions: {summaries[0]['positions']} (first rows of the fixed-seed "
        "deterministic real-shard sample, then node-count bucketed)",
        "- Intervention inference: CPU fp32; probability and metric arithmetic: fp64",
        "- Policy comparison domain: exactly `[0, support.legal_count)` for each row",
        "- KL direction: baseline distribution || mirror-projected distribution",
        "- Value support: 65 equally spaced bins on `[-1, 1]`",
        "- Right-action semantics, mirror-coset agreement, and projector "
        "idempotence: PASS",
        f"- Causal paired-forward runtime: {runtime_seconds:.3f} seconds",
        f"- Largest mean legal-policy TV: `{policy_peak['name']}` "
        f"({_format_float(policy_peak['policy_tv']['mean'])})",
        f"- Largest mean value-distribution TV: `{value_peak['name']}` "
        f"({_format_float(value_peak['value_tv']['mean'])})",
        f"- Largest policy top-1 flip rate: `{flip_peak['name']}` "
        f"({100.0 * flip_peak['top1_flip_rate']:.2f}%)",
        "",
        "Metric cells are `mean / median / p95 / max` over positions.",
        "",
        "| Depth | N | Legal-policy KL | Legal-policy TV | Top-1 flip | "
        "Value KL | Value TV | Expected-value absolute drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['name']}` | {summary['positions']} | "
            f"`{_causal_stat_cell(summary['policy_kl'])}` | "
            f"`{_causal_stat_cell(summary['policy_tv'])}` | "
            f"{100.0 * summary['top1_flip_rate']:.2f}% | "
            f"`{_causal_stat_cell(summary['value_kl'])}` | "
            f"`{_causal_stat_cell(summary['value_tv'])}` | "
            f"`{_causal_stat_cell(summary['expected_value_abs_drift'])}` |"
        )
    lines.extend(
        [
            "",
            "This intervention removes only the right-`<sigma>`-odd component at "
            "one trained regular-representation boundary. It is a causal "
            "sensitivity measurement, not an end-to-end quotient-net accuracy "
            "estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_report(
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    checkpoint_size: int,
    checkpoint_meta: dict[str, Any],
    model_meta: dict[str, Any],
    arch_env_path: Path,
    arch_env: dict[str, str],
    source_description: str,
    shard_count: int | None,
    available_rows: int | None,
    positions: Sequence[AuditPosition],
    cells: Sequence[dict[str, Any]],
    tokens: Sequence[dict[str, Any]],
    sigma: int,
    rot180: int,
    cosets: dict[str, tuple[tuple[int, ...], ...]],
    weight_summaries: Sequence[dict[str, Any]] | None,
) -> str:
    stats = _position_stats(positions)
    depth_cells = [summary for summary in cells if summary["name"].split("/", 1)[-1].startswith("depth_")]
    mirror_depth = [summary["overall"]["mirror"] for summary in depth_cells]
    strong_count = sum(value >= 0.70 for value in mirror_depth)
    strong_signal = strong_count > len(mirror_depth) / 2

    lines = [
        "# G7 quotient-type audit",
        "",
        "## Provenance and contract",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Checkpoint SHA-256: `{checkpoint_hash}`",
        f"- Checkpoint bytes: {checkpoint_size}",
        f"- Checkpoint epoch: {checkpoint_meta.get('epoch')}",
        f"- Architecture env: `{arch_env_path}`",
        f"- Position source: {source_description}",
        f"- Positions audited: {len(positions)}",
        f"- Fixed seed: {SEED}",
        "- Device: CPU only",
        f"- Triton modules imported: {len(_triton_module_names())}",
        "- Production FlexAttention import: blocked; materialized CPU attention used",
        "- Activation forward dtype: fp32 checkpoint inference",
        "- Projection/energy accumulation dtype: fp64",
        "- Architecture metadata strict match: PASS",
        "- State-dict strict load: PASS",
        "",
    ]
    if shard_count is not None and available_rows is not None:
        lines.extend(
            [
                f"- Eligible committed shards: {shard_count}",
                f"- Eligible manifest/sidecar rows: {available_rows}",
            ]
        )
        train_state = checkpoint_meta.get("train_state") or {}
        if "total_num_data_rows" in train_state:
            aligned = int(train_state["total_num_data_rows"]) == available_rows
            lines.append(
                "- Checkpoint train-state rows vs eligible source rows: "
                f"{train_state['total_num_data_rows']} vs {available_rows} "
                f"({'MATCH' if aligned else 'DIFFER'})"
            )
        lines.append("")

    lines.extend(
        [
            "Node-count quartiles `(min, Q25, median, Q75, max)`: "
            f"`{tuple(round(value, 3) for value in stats['nodes'])}`.",
            "Ply quartiles `(min, Q25, median, Q75, max)`: "
            f"`{tuple(round(value, 3) for value in stats['plies'])}`.",
            "",
            "### Arm-4 architecture metadata",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )
    for key in ARCH_META_KEYS:
        lines.append(f"| `{key}` | `{model_meta[key]}` |")
    lines.extend(["", "### Parsed architecture env", "", "| Variable | Value |", "|---|---|"])
    for key in ARM4_ENV_EXPECTED:
        lines.append(f"| `{key}` | `{arch_env[key]}` |")

    lines.extend(
        [
            "",
            "## Projection definition and internal checks",
            "",
            f"- `sigma = g{sigma}`; `rot180 = g{rot180}`.",
            "- Right averaging: `(P_H v)[g] = mean_{h in H} v[g h]`.",
            f"- `G` blocks: `{cosets['G']}`",
            f"- `K` blocks: `{cosets['K']}`",
            f"- `mirror` blocks: `{cosets['mirror']}`",
            f"- `point` blocks: `{cosets['point']}`",
            "- Strengthened nesting asserted per stream and per orbit channel: "
            "`E_G <= E_K <= E_mirror` and `E_G <= E_K <= E_point`.",
            "- Overall `E_H` is the energy-weighted ratio of sums. `macro E_H` is "
            "the unweighted mean over nonzero `(site, orbit_channel)` fiber vectors.",
            "",
        ]
    )
    lines.extend(_energy_table("Cell-stream energy fractions", cells))
    lines.extend(_energy_table("Token-stream energy fractions", tokens))
    lines.extend(_distribution_table("Per-channel distributions — cells", cells))
    lines.extend(_distribution_table("Per-channel distributions — tokens", tokens))
    if weight_summaries is not None:
        lines.extend(_weight_table(weight_summaries))

    lines.extend(
        [
            "## G7 interpretation",
            "",
            f"- Mirror-invariant energy is at least 70% in {strong_count}/{len(mirror_depth)} "
            "trunk-depth cell streams.",
            f"- Mean mirror-invariant energy across trunk depth: "
            f"{_format_float(sum(mirror_depth) / len(mirror_depth))}.",
            "- Owner rule (`E_mirror >= 70%` across most trunk depth): "
            f"**{'STRONG GO SIGNAL' if strong_signal else 'NOT MET'}**.",
            "- This is the G7 representation-evidence signal only; the Phase-A "
            "GO/NO-GO decision must also use G8 costs and all earlier gates.",
            "",
            "## Acceptance checks",
            "",
            f"- Every expected cell stream observed ({len(cells)}): PASS",
            f"- Every expected token stream observed ({len(tokens)}): PASS",
            "- All subgroup energy bounds and strengthened nesting relations: PASS",
            "- CPU/no-Triton runtime contract: PASS",
            "- Deterministic seed and manifest-uniform sampling: PASS",
            "",
        ]
    )
    return "\n".join(lines)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _write_output(path: Path, text: str, checkpoint: Path) -> None:
    destination = path.expanduser().resolve()
    if destination == Path(__file__).resolve():
        raise ValueError("refusing to overwrite the audit script")
    run_root = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    if _is_within(destination, run_root.resolve()):
        raise ValueError(f"refusing to write inside the read-only run directory: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"--output parent directory does not exist (not creating it): {destination.parent}"
        )
    destination.write_text(text + "\n", encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="CPU-only quotient-type activation audit for a real hexfield_eq checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="epoch_*.pt file, checkpoints directory, or run directory (directory resolves latest)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--shards",
        nargs="+",
        metavar="PATH_OR_GLOB",
        help="compact-v4 shard file(s), directory/directories, or glob(s)",
    )
    source.add_argument(
        "--random-prefixes",
        type=int,
        metavar="N",
        help="fallback: generate N uniform-random legal engine prefixes",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=DEFAULT_POSITIONS,
        help=f"manifest-uniform real rows to audit (default: {DEFAULT_POSITIONS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"CPU inference batch size after node-count bucketing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--arch-env",
        type=Path,
        default=root / ARCH_ENV_RELATIVE,
        help="arm-4 architecture env file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the deterministic markdown report here (never writes by default)",
    )
    parser.add_argument(
        "--weight-audit",
        action="store_true",
        help="also project every EquivLinear.wb coefficient function on D6",
    )
    parser.add_argument(
        "--causal-mirror-positions",
        type=int,
        default=0,
        metavar="N",
        help=(
            "on the first N deterministic real-shard positions, replace each "
            "trunk depth in turn by its right-<sigma> projection and measure "
            "policy/value drift (default: 0/off)"
        ),
    )
    parser.add_argument("--threads", type=int, help="set torch CPU worker threads")
    parser.add_argument("--verbose", action="store_true", help="batch progress to stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.causal_mirror_positions < 0:
        raise ValueError("--causal-mirror-positions must be non-negative")
    if args.causal_mirror_positions and args.shards is None:
        raise ValueError(
            "--causal-mirror-positions requires --shards; random prefixes are "
            "not real-position evidence"
        )
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)

    root = _repo_root()
    arch_env_path = args.arch_env.expanduser().resolve()
    arch_env = _configure_arch_env(arch_env_path)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    checkpoint_size = checkpoint.stat().st_size
    payload = _load_checkpoint(checkpoint)
    checkpoint_epoch = int(payload["meta"]["epoch"]) if "epoch" in payload["meta"] else _checkpoint_epoch(checkpoint)

    runtime = _import_runtime(root)
    model, model_meta = _build_model(runtime, payload)
    cosets, sigma, rot180 = _projection_cosets(runtime)

    shard_count: int | None = None
    available_rows: int | None = None
    if args.shards is not None:
        shards = _discover_shards(args.shards, checkpoint_epoch)
        shard_count = len(shards)
        available_rows = sum(info.rows for info in shards)
        positions = _load_real_positions(runtime, shards, args.positions)
        roots = sorted({str(info.source_root) for info in shards})
        source_description = (
            f"manifest/sidecar-uniform real rows from {', '.join(f'`{item}`' for item in roots)}"
        )
    else:
        positions = _load_random_positions(runtime, int(args.random_prefixes))
        source_description = (
            "deterministic uniform-random legal engine prefixes (fallback; "
            "distributionally weaker than real shards)"
        )

    audit = ActivationAudit(model, str(model_meta["trunk_layout"]), cosets)
    audit.run(
        positions,
        runtime.collate_rows,
        args.batch_size,
        verbose=bool(args.verbose),
    )
    cells, tokens = audit.finalize()
    weight_summaries = _run_weight_audit(model, cosets) if args.weight_audit else None
    causal_summaries: list[dict[str, Any]] | None = None
    causal_runtime: float | None = None
    if args.causal_mirror_positions:
        causal_summaries, causal_runtime = _run_causal_mirror_audit(
            model=model,
            runtime=runtime,
            layout=str(model_meta["trunk_layout"]),
            positions=positions,
            count=int(args.causal_mirror_positions),
            batch_size=args.batch_size,
            sigma=sigma,
            mirror_cosets=cosets["mirror"],
            verbose=bool(args.verbose),
        )
    loaded = _triton_module_names()
    if loaded:
        raise RuntimeError(f"CPU contract violated during audit: Triton modules loaded: {loaded[:5]}")

    report = _markdown_report(
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        checkpoint_size=checkpoint_size,
        checkpoint_meta=payload["meta"],
        model_meta=model_meta,
        arch_env_path=arch_env_path,
        arch_env=arch_env,
        source_description=source_description,
        shard_count=shard_count,
        available_rows=available_rows,
        positions=positions,
        cells=cells,
        tokens=tokens,
        sigma=sigma,
        rot180=rot180,
        cosets=cosets,
        weight_summaries=weight_summaries,
    )
    if causal_summaries is not None and causal_runtime is not None:
        report += "\n" + _causal_mirror_report(causal_summaries, causal_runtime)
    print(report)
    if args.output is not None:
        _write_output(args.output, report, checkpoint)
    if args.verbose:
        print(f"audit runtime: {time.perf_counter() - started:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
