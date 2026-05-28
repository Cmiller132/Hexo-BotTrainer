"""KataGo-style NPZ replay, shuffling, and train-bucket helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import floor, isfinite, log
from pathlib import Path
from random import Random
from time import time, time_ns
from typing import Any, Mapping, Sequence
import hashlib
import json
import shutil
import zipfile

import numpy as np
import torch
from numpy.lib import format as np_format

from .constants import BOARD_SIZE, INPUT_CHANNELS
from .samples import CURRENT_TARGET_SCHEMA_VERSION, Model1SampleData, expand_sample, stack_expanded

INPUT_KEY = "inputNCHW"
POLICY_KEY = "policyTargetsNCHW"
OPP_POLICY_KEY = "oppPolicyTargetsNCHW"
ROOT_POLICY_KEY = "rootPolicyNCHW"
LEGAL_MASK_KEY = "legalMaskNCHW"
VALUE_KEY = "valueTargetsN"
SHORT_TERM_VALUE_KEY = "shortTermValueTargetsNC"
SHORT_TERM_VALUE_MASK_KEY = "shortTermValueMasksNC"
METADATA_KEY = "metadataInputNC"
NPZ_KEYS = (
    INPUT_KEY,
    POLICY_KEY,
    OPP_POLICY_KEY,
    ROOT_POLICY_KEY,
    LEGAL_MASK_KEY,
    VALUE_KEY,
    SHORT_TERM_VALUE_KEY,
    SHORT_TERM_VALUE_MASK_KEY,
    METADATA_KEY,
)


@dataclass(frozen=True, slots=True)
class DenseSelfplayWriteResult:
    path: Path
    sidecar_path: Path
    game_id: str
    raw_rows: int
    effective_rows: int
    policy_surprise_mean: float
    frequency_weight_mean: float


@dataclass(frozen=True, slots=True)
class ShuffleFileInfo:
    path: Path
    mtime: float
    rows: int


@dataclass(frozen=True, slots=True)
class DenseShuffleResult:
    status: str
    shuffle_dir: Path | None
    train_dir: Path | None
    train_json_path: Path | None
    total_num_data_rows: int
    desired_rows: int
    used_rows: int
    output_rows: int
    output_files: tuple[Path, ...]
    validation_dir: Path | None = None
    validation_json_path: Path | None = None
    validation_rows: int = 0
    validation_files: tuple[Path, ...] = ()
    window_start_data_row_idx: int = 0
    reason: str | None = None


@dataclass(slots=True)
class DenseTrainState:
    global_step_samples: int = 0
    total_num_data_rows: int = 0
    window_start_data_row_idx: int = 0
    train_bucket_level: float = 0.0
    train_bucket_level_at_row: int = 0
    train_steps_since_last_reload: int = 0
    data_files_used: set[str] = field(default_factory=set)
    old_train_data_dirs: list[str] = field(default_factory=list)
    latest_shuffle_dir: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "DenseTrainState":
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            global_step_samples=int(raw.get("global_step_samples", 0)),
            total_num_data_rows=int(raw.get("total_num_data_rows", 0)),
            window_start_data_row_idx=int(raw.get("window_start_data_row_idx", 0)),
            train_bucket_level=float(raw.get("train_bucket_level", 0.0)),
            train_bucket_level_at_row=int(raw.get("train_bucket_level_at_row", 0)),
            train_steps_since_last_reload=int(raw.get("train_steps_since_last_reload", 0)),
            data_files_used=set(str(item) for item in raw.get("data_files_used", ())),
            old_train_data_dirs=[str(item) for item in raw.get("old_train_data_dirs", ())],
            latest_shuffle_dir=(
                str(raw["latest_shuffle_dir"]) if raw.get("latest_shuffle_dir") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step_samples": int(self.global_step_samples),
            "total_num_data_rows": int(self.total_num_data_rows),
            "window_start_data_row_idx": int(self.window_start_data_row_idx),
            "train_bucket_level": float(self.train_bucket_level),
            "train_bucket_level_at_row": int(self.train_bucket_level_at_row),
            "train_steps_since_last_reload": int(self.train_steps_since_last_reload),
            "data_files_used": sorted(self.data_files_used),
            "old_train_data_dirs": list(self.old_train_data_dirs),
            "latest_shuffle_dir": self.latest_shuffle_dir,
        }


@dataclass(frozen=True, slots=True)
class DenseNpzSampleWindow:
    files: tuple[Path, ...]
    seed: int
    epoch: int
    index: Any
    window_size: int
    target_rows: int
    shuffle_dir: Path | None
    validation_files: tuple[Path, ...] = ()
    validation_rows: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _SplitBuildResult:
    split: str
    output_dir: Path
    output_files: tuple[Path, ...]
    output_rows: int
    expected_rows: int
    scratch_parts: int
    input_files: tuple[Path, ...]
    input_rows: int


def materialize_policy_surprise_rows(
    samples: Sequence[Model1SampleData],
    *,
    seed: int,
    uniform_fraction: float = 0.5,
    max_weight: float = 8.0,
) -> tuple[list[Model1SampleData], dict[str, float]]:
    """Return samples repeated by KataGo policy-surprise frequency weights.

    Each sample's frequency weight mixes a uniform floor with a term proportional
    to its policy surprise `KL(target || prior)`, so surprising positions are seen
    more often. Weights sum to the game length before the `max_weight` clamp.
    """

    if not samples:
        return [], {
            "raw_rows": 0.0,
            "effective_rows": 0.0,
            "policy_surprise_mean": 0.0,
            "frequency_weight_mean": 0.0,
        }
    surprises = [_policy_kl(sample.policy, sample.root_prior_policy) for sample in samples]
    surprise_total = sum(surprises)
    if surprise_total > 0.0:
        n = float(len(samples))
        kl_fraction = 1.0 - uniform_fraction
        weights = [
            min(max_weight, uniform_fraction + kl_fraction * n * surprise / surprise_total)
            for surprise in surprises
        ]
    else:
        weights = [1.0 for _sample in samples]

    rng = Random(int(seed))
    materialized: list[Model1SampleData] = []
    for sample, surprise, weight in zip(samples, surprises, weights):
        copies = floor(weight)
        if rng.random() < weight - copies:
            copies += 1
        copies = max(0, int(copies))
        updated = replace(
            sample,
            policy_surprise=float(surprise),
            frequency_weight=float(weight),
            metadata={
                **dict(sample.metadata),
                "policy_surprise": float(surprise),
                "frequency_weight": float(weight),
                "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
            },
        )
        materialized.extend(updated for _ in range(copies))

    return materialized, {
        "raw_rows": float(len(samples)),
        "effective_rows": float(len(materialized)),
        "policy_surprise_mean": float(sum(surprises) / len(samples)),
        "frequency_weight_mean": float(sum(weights) / len(weights)),
    }


def write_selfplay_npz(
    path: Path,
    samples: Sequence[Model1SampleData],
    *,
    raw_rows: int,
    epoch: int,
    game_id: str,
    short_term_value_horizons: Sequence[int],
) -> DenseSelfplayWriteResult:
    """Write one self-play game shard as dense training rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_for_npz(path)
    arrays = _samples_to_arrays(samples, short_term_value_horizons=short_term_value_horizons)
    np.savez_compressed(path, **arrays)
    policy_surprises = [float(sample.policy_surprise) for sample in samples]
    frequency_weights = [float(sample.frequency_weight) for sample in samples]
    sidecar = {
        "num_rows": int(len(samples)),
        "raw_rows": int(raw_rows),
        "effective_rows": int(len(samples)),
        "epoch": int(epoch),
        "game_id": str(game_id),
        "target_schema_version": int(CURRENT_TARGET_SCHEMA_VERSION),
        "policy_surprise_mean": float(sum(policy_surprises) / len(policy_surprises)) if policy_surprises else 0.0,
        "frequency_weight_mean": float(sum(frequency_weights) / len(frequency_weights)) if frequency_weights else 0.0,
        "created_at": float(time()),
    }
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True, indent=2), encoding="utf-8")
    return DenseSelfplayWriteResult(
        path=path,
        sidecar_path=sidecar_path,
        game_id=game_id,
        raw_rows=int(raw_rows),
        effective_rows=int(len(samples)),
        policy_surprise_mean=float(sidecar["policy_surprise_mean"]),
        frequency_weight_mean=float(sidecar["frequency_weight_mean"]),
    )


def build_katago_shuffle(
    *,
    selfplay_dir: Path,
    shuffled_root: Path,
    scratch_dir: Path,
    epoch: int,
    seed: int,
    min_rows: int,
    keep_target_rows: int,
    taper_window_exponent: float,
    expand_window_per_row: float,
    taper_window_scale: float | None,
    approx_rows_per_out_file: int,
    batch_size: int,
    worker_group_size: int,
    validation_fraction: float = 0.0,
    md5_lbound: float = 0.0,
    md5_ubound: float = 1.0,
) -> DenseShuffleResult:
    """Build a KataGo-style shuffled NPZ train directory from self-play rows."""

    _validate_shuffle_args(
        min_rows=min_rows,
        keep_target_rows=keep_target_rows,
        approx_rows_per_out_file=approx_rows_per_out_file,
        batch_size=batch_size,
        worker_group_size=worker_group_size,
        validation_fraction=validation_fraction,
        md5_lbound=md5_lbound,
        md5_ubound=md5_ubound,
    )
    files = scan_selfplay_npz_files(selfplay_dir)
    total_rows = sum(item.rows for item in files)
    if total_rows < min_rows:
        return _skipped_shuffle(total_rows=total_rows, reason=f"not enough rows: {total_rows} < {min_rows}")

    desired_rows = compute_katago_window_rows(
        total_rows,
        min_rows=min_rows,
        expand_window_per_row=expand_window_per_row,
        taper_window_exponent=taper_window_exponent,
        taper_window_scale=taper_window_scale,
    )
    desired_rows = max(int(desired_rows), int(min_rows))
    selected, used_rows = _select_recent_window(files, desired_rows)
    selected = [
        info
        for info in selected
        if md5_lbound <= _md5_path_fraction(str(info.path)) < md5_ubound
    ]
    used_rows = sum(info.rows for info in selected)
    if not selected:
        return _skipped_shuffle(
            total_rows=total_rows,
            desired_rows=desired_rows,
            reason="no files selected after md5 range split",
        )
    keep_prob = min(float(keep_target_rows), float(used_rows)) / float(used_rows)
    train_infos, val_infos = _split_by_md5(selected, validation_fraction=validation_fraction)
    if not train_infos:
        return _skipped_shuffle(
            total_rows=total_rows,
            desired_rows=desired_rows,
            used_rows=used_rows,
            reason="no train files selected after md5 validation split",
        )

    generation = f"{time_ns():019d}-epoch_{int(epoch):06d}"
    shuffled_root.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = shuffled_root / f"{generation}.tmp"
    shuffle_dir = shuffled_root / generation
    scratch_root = scratch_dir / generation
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    scratch_root.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(int(seed))
    window_start = max(0, int(total_rows - used_rows))
    try:
        train = _build_split_outputs(
            split="train",
            infos=train_infos,
            output_root=tmp_dir,
            scratch_root=scratch_root,
            keep_prob=keep_prob,
            rng=rng,
            approx_rows_per_out_file=approx_rows_per_out_file,
            batch_size=batch_size,
            worker_group_size=worker_group_size,
        )
        val = _build_split_outputs(
            split="val",
            infos=val_infos,
            output_root=tmp_dir,
            scratch_root=scratch_root,
            keep_prob=keep_prob,
            rng=rng,
            approx_rows_per_out_file=approx_rows_per_out_file,
            batch_size=batch_size,
            worker_group_size=worker_group_size,
        ) if validation_fraction > 0.0 else None

        if train.output_rows <= 0:
            return _cleanup_skipped_shuffle(
                tmp_dir=tmp_dir,
                scratch_root=scratch_root,
                total_rows=total_rows,
                desired_rows=desired_rows,
                used_rows=used_rows,
                window_start=window_start,
                reason="selected train rows rounded below one batch",
            )

        train_json_path = tmp_dir / "train.json"
        train_json_path.write_text(
            json.dumps(
                _split_json(
                    split_result=train,
                    total_rows=total_rows,
                    desired_rows=desired_rows,
                    used_rows=used_rows,
                    keep_prob=keep_prob,
                    epoch=epoch,
                    generation=generation,
                    window_start=window_start,
                    validation_fraction=validation_fraction,
                    worker_group_size=worker_group_size,
                    approx_rows_per_out_file=approx_rows_per_out_file,
                    batch_size=batch_size,
                ),
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        val_json_path: Path | None = None
        if val is not None:
            val_json_path = tmp_dir / "val.json"
            val_json_path.write_text(
                json.dumps(
                    _split_json(
                        split_result=val,
                        total_rows=total_rows,
                        desired_rows=desired_rows,
                        used_rows=used_rows,
                        keep_prob=keep_prob,
                        epoch=epoch,
                        generation=generation,
                        window_start=window_start,
                        validation_fraction=validation_fraction,
                        worker_group_size=worker_group_size,
                        approx_rows_per_out_file=approx_rows_per_out_file,
                        batch_size=batch_size,
                    ),
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )

        summary = {
            "status": "completed",
            "epoch": int(epoch),
            "generation": generation,
            "total_num_data_rows": int(total_rows),
            "desired_rows": int(desired_rows),
            "used_rows": int(used_rows),
            "window_start_data_row_idx": int(window_start),
            "window_end_data_row_idx": int(total_rows),
            "keep_prob": float(keep_prob),
            "validation_fraction": float(validation_fraction),
            "train_rows": int(train.output_rows),
            "validation_rows": int(val.output_rows if val is not None else 0),
            "train_output_files": [str(path.relative_to(tmp_dir)) for path in train.output_files],
            "validation_output_files": (
                [str(path.relative_to(tmp_dir)) for path in val.output_files] if val is not None else []
            ),
        }
        (tmp_dir / "shuffle.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
        if shuffle_dir.exists():
            shutil.rmtree(shuffle_dir)
        tmp_dir.rename(shuffle_dir)
    finally:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)

    train_files = tuple(shuffle_dir / path.relative_to(tmp_dir) for path in train.output_files)
    val_files = (
        tuple(shuffle_dir / path.relative_to(tmp_dir) for path in val.output_files)
        if val is not None
        else ()
    )
    return DenseShuffleResult(
        status="completed",
        shuffle_dir=shuffle_dir,
        train_dir=shuffle_dir / "train",
        train_json_path=shuffle_dir / "train.json",
        total_num_data_rows=int(total_rows),
        desired_rows=int(desired_rows),
        used_rows=int(used_rows),
        output_rows=int(train.output_rows),
        output_files=train_files,
        validation_dir=shuffle_dir / "val" if val is not None else None,
        validation_json_path=shuffle_dir / "val.json" if val_json_path is not None else None,
        validation_rows=int(val.output_rows if val is not None else 0),
        validation_files=val_files,
        window_start_data_row_idx=int(window_start),
    )


def scan_selfplay_npz_files(root: Path) -> list[ShuffleFileInfo]:
    files: list[ShuffleFileInfo] = []
    if not root.exists():
        return files
    for path in root.rglob("*.npz"):
        if ".tmp" in path.parts:
            continue
        rows = npz_row_count(path)
        if rows > 0:
            files.append(ShuffleFileInfo(path=path, mtime=path.stat().st_mtime, rows=rows))
    files.sort(key=lambda item: item.mtime)
    return files


def latest_shuffle_dir(shuffled_root: Path) -> Path | None:
    if not shuffled_root.exists():
        return None
    candidates = [
        path
        for path in shuffled_root.iterdir()
        if path.is_dir()
        and not path.name.endswith(".tmp")
        and (path / "train.json").exists()
        and (path / "train").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def shuffle_train_files(shuffle_dir: Path) -> tuple[Path, ...]:
    return shuffle_split_files(shuffle_dir, split="train")


def shuffle_validation_files(shuffle_dir: Path) -> tuple[Path, ...]:
    return shuffle_split_files(shuffle_dir, split="val")


def shuffle_split_files(shuffle_dir: Path, *, split: str) -> tuple[Path, ...]:
    split_dir = shuffle_dir / split
    if not split_dir.exists():
        return ()
    return tuple(sorted(split_dir.glob("*.npz")))


def load_train_json(shuffle_dir: Path) -> dict[str, Any]:
    return load_split_json(shuffle_dir, split="train")


def load_split_json(shuffle_dir: Path, *, split: str) -> dict[str, Any]:
    path = shuffle_dir / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def npz_row_count(path: Path) -> int:
    sidecar = sidecar_for_npz(path)
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return int(data.get("num_rows", data.get("effective_rows", 0)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    shape = _npz_array_shape(path, INPUT_KEY)
    return int(shape[0])


def sidecar_for_npz(path: Path) -> Path:
    return path.with_suffix(".json")


def compute_katago_window_rows(
    usable_rows: int,
    *,
    min_rows: int,
    expand_window_per_row: float,
    taper_window_exponent: float,
    taper_window_scale: float | None,
) -> int:
    offset = float(taper_window_scale if taper_window_scale is not None else min_rows)
    power_law_x = float(usable_rows) - float(min_rows) + offset
    unscaled = power_law_x ** taper_window_exponent - offset ** taper_window_exponent
    scaled = unscaled / (taper_window_exponent * (offset ** (taper_window_exponent - 1.0)))
    return int(scaled * expand_window_per_row + float(min_rows))


def _samples_to_arrays(
    samples: Sequence[Model1SampleData],
    *,
    short_term_value_horizons: Sequence[int],
) -> dict[str, np.ndarray]:
    expanded = []
    horizons = tuple(int(horizon) for horizon in short_term_value_horizons)
    for sample in samples:
        row = expand_sample(sample)
        for horizon in horizons:
            key = f"stvalue_{horizon}"
            mask_key = f"{key}_mask"
            if key in row:
                row[mask_key] = torch.tensor(1.0, dtype=torch.float32)
            else:
                row[key] = torch.tensor(0.0, dtype=torch.float32)
                row[mask_key] = torch.tensor(0.0, dtype=torch.float32)
        expanded.append(row)
    if not expanded:
        return {
            INPUT_KEY: np.zeros((0, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            POLICY_KEY: np.zeros((0, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            OPP_POLICY_KEY: np.zeros((0, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            ROOT_POLICY_KEY: np.zeros((0, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            LEGAL_MASK_KEY: np.zeros((0, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.bool_),
            VALUE_KEY: np.zeros((0,), dtype=np.float32),
            SHORT_TERM_VALUE_KEY: np.zeros((0, len(horizons)), dtype=np.float32),
            SHORT_TERM_VALUE_MASK_KEY: np.zeros((0, len(horizons)), dtype=np.float32),
            METADATA_KEY: np.zeros((0, 4), dtype=np.float32),
        }
    batch = stack_expanded(expanded)
    short_term_value_targets = []
    short_term_value_masks = []
    for horizon in horizons:
        short_term_value_targets.append(batch[f"stvalue_{horizon}"].reshape(-1, 1))
        short_term_value_masks.append(batch[f"stvalue_{horizon}_mask"].reshape(-1, 1))
    metadata = np.asarray(
        [
            [
                float(sample.turn_index),
                float(sample.policy_surprise),
                float(sample.frequency_weight),
                float(sample.metadata.get("search_visits", 0)),
            ]
            for sample in samples
        ],
        dtype=np.float32,
    )
    return {
        INPUT_KEY: batch["input"].cpu().numpy().astype(np.float32, copy=False),
        POLICY_KEY: _flat_to_nchw(batch["policy"]),
        OPP_POLICY_KEY: _flat_to_nchw(batch["opp_policy"]),
        ROOT_POLICY_KEY: _flat_to_nchw(batch["root_policy"]),
        LEGAL_MASK_KEY: _flat_to_nchw(batch["legal_mask"].to(dtype=torch.float32)).astype(np.bool_),
        VALUE_KEY: batch["value"].cpu().numpy().astype(np.float32, copy=False),
        SHORT_TERM_VALUE_KEY: (
            torch.cat(short_term_value_targets, dim=1).cpu().numpy().astype(np.float32, copy=False)
            if short_term_value_targets
            else np.zeros((len(samples), 0), dtype=np.float32)
        ),
        SHORT_TERM_VALUE_MASK_KEY: (
            torch.cat(short_term_value_masks, dim=1).cpu().numpy().astype(np.float32, copy=False)
            if short_term_value_masks
            else np.zeros((len(samples), 0), dtype=np.float32)
        ),
        METADATA_KEY: metadata,
    }


def _flat_to_nchw(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.reshape(tensor.shape[0], 1, BOARD_SIZE, BOARD_SIZE)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _policy_kl(
    target: Sequence[tuple[int, float]],
    prior: Sequence[tuple[int, float]],
    *,
    eps: float = 1.0e-8,
) -> float:
    if target and not prior:
        raise ValueError("policy surprise weighting requires root_prior_policy")
    prior_map = {int(action): float(weight) for action, weight in prior}
    kl = 0.0
    for action, weight in target:
        target_weight = float(weight)
        if target_weight <= 0.0:
            continue
        prior_weight = max(float(prior_map.get(int(action), 0.0)), eps)
        kl += target_weight * log((target_weight + eps) / prior_weight)
    return max(0.0, float(kl)) if isfinite(kl) else 0.0


def _validate_shuffle_args(
    *,
    min_rows: int,
    keep_target_rows: int,
    approx_rows_per_out_file: int,
    batch_size: int,
    worker_group_size: int,
    validation_fraction: float,
    md5_lbound: float,
    md5_ubound: float,
) -> None:
    if min_rows <= 0:
        raise ValueError("shuffle_min_rows must be > 0")
    if keep_target_rows <= 0:
        raise ValueError("shuffle_keep_target_rows must be > 0")
    if approx_rows_per_out_file <= 0:
        raise ValueError("approx_rows_per_out_file must be > 0")
    if batch_size <= 0:
        raise ValueError("dense_cnn training batch size must be > 0")
    if worker_group_size <= 0:
        raise ValueError("shuffle_worker_group_size must be > 0")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    if not 0.0 <= md5_lbound < md5_ubound <= 1.0:
        raise ValueError("md5 bounds must satisfy 0.0 <= lower < upper <= 1.0")


def _select_recent_window(files: Sequence[ShuffleFileInfo], desired_rows: int) -> tuple[list[ShuffleFileInfo], int]:
    selected: list[ShuffleFileInfo] = []
    used_rows = 0
    for info in reversed(files):
        selected.append(info)
        used_rows += info.rows
        if used_rows >= desired_rows:
            break
    selected.reverse()
    return selected, used_rows


def _split_by_md5(
    selected: Sequence[ShuffleFileInfo],
    *,
    validation_fraction: float,
) -> tuple[list[ShuffleFileInfo], list[ShuffleFileInfo]]:
    if validation_fraction <= 0.0:
        return list(selected), []
    train_upper = 1.0 - float(validation_fraction)
    train_infos: list[ShuffleFileInfo] = []
    val_infos: list[ShuffleFileInfo] = []
    for info in selected:
        fraction = _md5_path_fraction(str(info.path))
        if fraction < train_upper:
            train_infos.append(info)
        else:
            val_infos.append(info)
    return train_infos, val_infos


def _build_split_outputs(
    *,
    split: str,
    infos: Sequence[ShuffleFileInfo],
    output_root: Path,
    scratch_root: Path,
    keep_prob: float,
    rng: np.random.Generator,
    approx_rows_per_out_file: int,
    batch_size: int,
    worker_group_size: int,
) -> _SplitBuildResult:
    split_dir = output_root / split
    split_scratch = scratch_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    split_scratch.mkdir(parents=True, exist_ok=True)
    input_rows = sum(info.rows for info in infos)
    expected_rows = int(round(float(input_rows) * float(keep_prob)))
    if input_rows <= 0 or expected_rows <= 0:
        return _SplitBuildResult(
            split=split,
            output_dir=split_dir,
            output_files=(),
            output_rows=0,
            expected_rows=0,
            scratch_parts=0,
            input_files=tuple(info.path for info in infos),
            input_rows=int(input_rows),
        )
    output_count = max(1, int(round(expected_rows / max(1, approx_rows_per_out_file))))
    scratch_parts = 0
    for group_index, group in enumerate(_worker_groups(infos, worker_group_size)):
        arrays = _load_group_kept_arrays(group, keep_prob=keep_prob, rng=rng)
        rows = int(arrays.get(INPUT_KEY, np.zeros((0,), dtype=np.float32)).shape[0])
        if rows <= 0:
            continue
        permutation = rng.permutation(rows)
        arrays = {key: value[permutation] for key, value in arrays.items()}
        buckets = rng.integers(0, output_count, size=rows, endpoint=False)
        for output_index in range(output_count):
            indices = np.nonzero(buckets == output_index)[0]
            if indices.size <= 0:
                continue
            shard_dir = split_scratch / f"data{output_index:05d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            part_path = shard_dir / f"part_{group_index:05d}.npz"
            np.savez_compressed(part_path, **{key: value[indices] for key, value in arrays.items()})
            scratch_parts += 1

    output_files: list[Path] = []
    output_rows = 0
    for output_index in range(output_count):
        shard_dir = split_scratch / f"data{output_index:05d}"
        if not shard_dir.exists():
            continue
        part_paths = list(sorted(shard_dir.glob("part_*.npz")))
        if not part_paths:
            continue
        rng.shuffle(part_paths)
        arrays = _load_part_arrays(part_paths)
        rows = int(arrays[INPUT_KEY].shape[0])
        if rows <= 0:
            continue
        permutation = rng.permutation(rows)
        aligned = (rows // batch_size) * batch_size
        if aligned <= 0:
            continue
        output_arrays = {key: value[permutation[:aligned]] for key, value in arrays.items()}
        out_path = split_dir / f"data{len(output_files):05d}.npz"
        np.savez_compressed(out_path, **output_arrays)
        sidecar_for_npz(out_path).write_text(
            json.dumps(
                {
                    "num_rows": int(aligned),
                    "num_batches": int(aligned // batch_size),
                    "target_schema_version": int(CURRENT_TARGET_SCHEMA_VERSION),
                    "split": split,
                    "source_parts": len(part_paths),
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        output_files.append(out_path)
        output_rows += aligned
    return _SplitBuildResult(
        split=split,
        output_dir=split_dir,
        output_files=tuple(output_files),
        output_rows=int(output_rows),
        expected_rows=int(expected_rows),
        scratch_parts=int(scratch_parts),
        input_files=tuple(info.path for info in infos),
        input_rows=int(input_rows),
    )


def _worker_groups(infos: Sequence[ShuffleFileInfo], worker_group_size: int) -> list[list[ShuffleFileInfo]]:
    groups: list[list[ShuffleFileInfo]] = []
    current: list[ShuffleFileInfo] = []
    rows = 0
    for info in infos:
        if current and rows + info.rows > worker_group_size:
            groups.append(current)
            current = []
            rows = 0
        current.append(info)
        rows += info.rows
    if current:
        groups.append(current)
    return groups


def _load_group_kept_arrays(
    group: Sequence[ShuffleFileInfo],
    *,
    keep_prob: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in NPZ_KEYS}
    for info in group:
        with np.load(info.path) as data:
            rows = int(data[INPUT_KEY].shape[0])
            if rows <= 0:
                continue
            if keep_prob >= 1.0:
                indices = np.arange(rows)
            else:
                keep = rng.random(rows) < keep_prob
                indices = np.nonzero(keep)[0]
            if indices.size <= 0:
                continue
            for key in NPZ_KEYS:
                chunks[key].append(data[key][indices])
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items() if values}


def _load_part_arrays(part_paths: Sequence[Path]) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in NPZ_KEYS}
    for part_path in part_paths:
        with np.load(part_path) as data:
            for key in NPZ_KEYS:
                chunks[key].append(data[key])
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items() if values}


def _split_json(
    *,
    split_result: _SplitBuildResult,
    total_rows: int,
    desired_rows: int,
    used_rows: int,
    keep_prob: float,
    epoch: int,
    generation: str,
    window_start: int,
    validation_fraction: float,
    worker_group_size: int,
    approx_rows_per_out_file: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "split": split_result.split,
        "num_rows": int(split_result.output_rows),
        "expected_rows": int(split_result.expected_rows),
        "input_rows": int(split_result.input_rows),
        "total_num_data_rows": int(total_rows),
        "desired_rows": int(desired_rows),
        "used_rows": int(used_rows),
        "window_start_data_row_idx": int(window_start),
        "window_end_data_row_idx": int(total_rows),
        "keep_prob": float(keep_prob),
        "epoch": int(epoch),
        "generation": generation,
        "validation_fraction": float(validation_fraction),
        "worker_group_size": int(worker_group_size),
        "approx_rows_per_out_file": int(approx_rows_per_out_file),
        "batch_size": int(batch_size),
        "scratch_parts": int(split_result.scratch_parts),
        "target_schema_version": int(CURRENT_TARGET_SCHEMA_VERSION),
        "input_files": [str(path) for path in split_result.input_files],
        "output_files": [str(path.relative_to(split_result.output_dir.parent)) for path in split_result.output_files],
    }


def _skipped_shuffle(
    *,
    total_rows: int,
    desired_rows: int = 0,
    used_rows: int = 0,
    window_start: int = 0,
    reason: str,
) -> DenseShuffleResult:
    return DenseShuffleResult(
        status="skipped",
        shuffle_dir=None,
        train_dir=None,
        train_json_path=None,
        total_num_data_rows=int(total_rows),
        desired_rows=int(desired_rows),
        used_rows=int(used_rows),
        output_rows=0,
        output_files=(),
        window_start_data_row_idx=int(window_start),
        reason=reason,
    )


def _cleanup_skipped_shuffle(
    *,
    tmp_dir: Path,
    scratch_root: Path,
    total_rows: int,
    desired_rows: int,
    used_rows: int,
    window_start: int,
    reason: str,
) -> DenseShuffleResult:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    return _skipped_shuffle(
        total_rows=total_rows,
        desired_rows=desired_rows,
        used_rows=used_rows,
        window_start=window_start,
        reason=reason,
    )


def _md5_path_fraction(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:13]
    return int("0x" + digest, 16) / float(2**52)


def _npz_array_shape(path: Path, key: str) -> tuple[int, ...]:
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            version = np_format.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, _dtype = np_format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _fortran, _dtype = np_format.read_array_header_2_0(handle)
            elif version == (3, 0):
                shape, _fortran, _dtype = np_format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported npy header version {version!r} in {path}")
    return tuple(int(item) for item in shape)
