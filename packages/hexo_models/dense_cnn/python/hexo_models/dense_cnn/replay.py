"""KataGo-style NPZ replay, shuffling, and train-bucket helpers.

This module owns the on-disk replay pipeline between self-play and training:

- `materialize_policy_surprise_rows` bakes KataGo frequency weighting into row
  duplication before write, so the training loss stays unweighted.
- `write_selfplay_npz` writes one game as a compact shard (`compact_io.py`)
  plus a JSON sidecar with row counts and surprise/weight summaries.
- `build_katago_shuffle` selects the recent mtime-ordered window over the
  self-play shard tree, applies the KataGo window taper, optionally splits
  train/val by file-path md5, and writes a shuffled generation directory
  (`<ns>-epoch_NNNNNN/{train,val}/data*.npz` + `train.json`/`shuffle.json`).
- `DenseTrainState` is the KataGo train-bucket bookkeeping persisted inside
  checkpoints (`checkpoints.py`).

Callers: `selfplay.py` writes shards, `trainer.py` builds/consumes shuffles and
owns the train state, `plugin.py` threads `DenseNpzSampleWindow` through the
generic `hexo_train` epoch loop. The compact shard format itself lives in
`compact_io.py` and is also read cross-package by the
repo-root dashboard bridges; `NPZ_KEYS` names the dense-expanded schema that
`samples.expand_sample` produces at train read time.

The active `packages/dense_cnn_restnet` lineage carries a forked copy of this
module; window/shard semantics must stay byte-compatible across the fork.
"""

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

import numpy as np

from . import compact_io
from .samples import CURRENT_TARGET_SCHEMA_VERSION, Model1SampleData

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


# --- Result/record dataclasses -------------------------------------------------


@dataclass(frozen=True, slots=True)
class DenseSelfplayWriteResult:
    """Summary of one written self-play shard (paths + row/weight telemetry)."""

    path: Path
    sidecar_path: Path
    game_id: str
    raw_rows: int
    effective_rows: int
    policy_surprise_mean: float
    frequency_weight_mean: float


@dataclass(frozen=True, slots=True)
class ShuffleFileInfo:
    """One candidate shard for the shuffle window (mtime orders the stream)."""

    path: Path
    mtime: float
    rows: int


@dataclass(frozen=True, slots=True)
class DenseShuffleResult:
    """Outcome of `build_katago_shuffle`: `status` is "completed" or "skipped".

    On skip, `reason` explains why and all path fields are None; row counters
    still report what was scanned so callers can log window progress.
    """

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
    """KataGo train-bucket bookkeeping persisted inside dense_cnn checkpoints.

    `trainer.py` mutates this across epochs (rows seen, bucket level, files
    already trained on, retired shuffle dirs); `checkpoints.py` round-trips it
    via `to_dict`/`from_mapping`. `from_mapping(None)` yields fresh state.
    """

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
        """Rebuild state from a checkpoint dict; non-mapping input means fresh state."""
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
        """JSON/torch-save-friendly snapshot (sets become sorted lists)."""
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
    """Selected training window handed to the generic `hexo_train` epoch loop.

    Built by `trainer.select_training_samples`; `files` are the shuffled compact
    shards for this epoch and `index` is trainer-owned bookkeeping. The generic
    pipeline treats this as opaque and passes it back to `trainer.train_passes`.
    """

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


# --- Self-play write path (surprise weighting + shard write) --------------------


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
    """Write one self-play game shard as compact rows (expanded at train read)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_for_npz(path)
    compact_io.write_compact_shard(path, samples, short_term_value_horizons=short_term_value_horizons)
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


# --- Shuffle build (window taper + md5 split + shuffled generation dir) ---------


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
    # Apply any md5 sub-range to the file SET up front, so total_rows, the recent-window
    # selection, and window_start_data_row_idx are all consistent over the sub-ranged
    # stream. Filtering AFTER window selection (the old behavior) left window_start
    # meaningless and shrank the realized window below desired_rows. Default [0, 1) is a
    # no-op that keeps every file.
    if md5_lbound > 0.0 or md5_ubound < 1.0:
        files = [info for info in files if md5_lbound <= _md5_path_fraction(str(info.path)) < md5_ubound]
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
    if not selected:
        return _skipped_shuffle(
            total_rows=total_rows,
            desired_rows=desired_rows,
            reason="no files selected for the window",
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
    tmp_dir = shuffled_root / f"{generation}.tmp"
    shuffle_dir = shuffled_root / generation
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(int(seed))
    window_start = max(0, int(total_rows - used_rows))
    try:
        train = _build_compact_split(
            split="train",
            infos=train_infos,
            output_root=tmp_dir,
            keep_prob=keep_prob,
            rng=rng,
            approx_rows_per_out_file=approx_rows_per_out_file,
            batch_size=batch_size,
        )
        val = _build_compact_split(
            split="val",
            infos=val_infos,
            output_root=tmp_dir,
            keep_prob=keep_prob,
            rng=rng,
            approx_rows_per_out_file=approx_rows_per_out_file,
            batch_size=batch_size,
        ) if validation_fraction > 0.0 else None

        if train.output_rows <= 0:
            return _cleanup_skipped_shuffle(
                tmp_dir=tmp_dir,
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
        # Success and failure cleanup share this branch: after a successful
        # rename, tmp_dir no longer exists, so this only deletes the staging
        # dir on the error/early-return paths.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

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


# --- Discovery and read helpers (shared by trainer + selfplay) ------------------


def scan_selfplay_npz_files(root: Path) -> list[ShuffleFileInfo]:
    """Collect non-empty shards under ``root`` (recursively), oldest-mtime first.

    Files inside ``*.tmp`` shuffle staging directories are skipped. mtime order
    is the replay stream order, which is why window seeding scripts must copy
    shards with timestamps preserved.
    """
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
    """Return the newest completed shuffle generation dir, or None if none exist."""
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
    """Row count from the JSON sidecar when readable, else from the shard itself."""
    sidecar = sidecar_for_npz(path)
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return int(data.get("num_rows", data.get("effective_rows", 0)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return compact_io.compact_row_count(path)


def sidecar_for_npz(path: Path) -> Path:
    """Path of the JSON sidecar written next to a shard (same stem, .json)."""
    return path.with_suffix(".json")


def compute_katago_window_rows(
    usable_rows: int,
    *,
    min_rows: int,
    expand_window_per_row: float,
    taper_window_exponent: float,
    taper_window_scale: float | None,
) -> int:
    """KataGo replay-window size: sublinear growth in total rows generated.

    Implements KataGo's tapered power-law window (`taper_window_exponent` < 1
    shrinks the marginal window growth as data accumulates); the result is the
    desired number of most-recent rows, never below `min_rows` at the caller.
    """
    offset = float(taper_window_scale if taper_window_scale is not None else min_rows)
    power_law_x = float(usable_rows) - float(min_rows) + offset
    unscaled = power_law_x ** taper_window_exponent - offset ** taper_window_exponent
    scaled = unscaled / (taper_window_exponent * (offset ** (taper_window_exponent - 1.0)))
    return int(scaled * expand_window_per_row + float(min_rows))


# --- Internal helpers ------------------------------------------------------------


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


def _build_compact_split(
    *,
    split: str,
    infos: Sequence[ShuffleFileInfo],
    output_root: Path,
    keep_prob: float,
    rng: np.random.Generator,
    approx_rows_per_out_file: int,
    batch_size: int,
) -> _SplitBuildResult:
    """Shuffle compact rows from the selected shards in RAM, write compact output shards.

    Compact rows are small (no dense planes), so the whole split fits in memory
    and the dense two-phase on-disk shuffle is unnecessary: load -> per-row
    keep_prob -> permute -> batch-align -> write fixed-size compact output shards
    (each a whole number of batches). Output horizons are inherited from the input
    shards so the schema stays self-describing.
    """

    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    input_rows = sum(info.rows for info in infos)
    expected_rows = int(round(float(input_rows) * float(keep_prob)))
    empty = _SplitBuildResult(
        split=split,
        output_dir=split_dir,
        output_files=(),
        output_rows=0,
        expected_rows=int(max(0, expected_rows)),
        scratch_parts=0,
        input_files=tuple(info.path for info in infos),
        input_rows=int(input_rows),
    )
    if input_rows <= 0 or expected_rows <= 0:
        return empty

    horizons = compact_io.read_shard_horizons(infos[0].path)
    # All shards in a window must share the same short_term_value horizons: the output
    # is written with one `horizons` set, so rows from a shard with a different set would
    # be silently mis-slotted / dropped (poisoning the short-term-value head). This only
    # happens if the horizon config changed mid-run; fail loudly rather than corrupt.
    for info in infos[1:]:
        other = compact_io.read_shard_horizons(info.path)
        if tuple(other) != tuple(horizons):
            raise ValueError(
                f"dense_cnn shuffle window mixes short_term_value horizons: "
                f"{tuple(horizons)} (from {infos[0].path.name}) vs {tuple(other)} "
                f"(from {info.path.name}); refusing to silently mis-slot stval targets"
            )
    rows: list[Model1SampleData] = []
    for info in infos:
        shard = compact_io.read_compact_shard(info.path)
        if keep_prob >= 1.0:
            rows.extend(shard)
        else:
            keep = rng.random(len(shard)) < keep_prob
            rows.extend(sample for sample, k in zip(shard, keep) if k)
    if not rows:
        return empty

    permutation = rng.permutation(len(rows))
    rows = [rows[i] for i in permutation]
    aligned = (len(rows) // batch_size) * batch_size
    if aligned <= 0:
        return empty
    rows = rows[:aligned]

    chunk = max(batch_size, (approx_rows_per_out_file // batch_size) * batch_size)
    output_files: list[Path] = []
    output_rows = 0
    start = 0
    while start < aligned:
        stop = min(start + chunk, aligned)
        out_rows = rows[start:stop]
        out_path = split_dir / f"data{len(output_files):05d}.npz"
        compact_io.write_compact_shard(out_path, out_rows, short_term_value_horizons=horizons)
        sidecar_for_npz(out_path).write_text(
            json.dumps(
                {
                    "num_rows": int(len(out_rows)),
                    "num_batches": int(len(out_rows) // batch_size),
                    "target_schema_version": int(CURRENT_TARGET_SCHEMA_VERSION),
                    "split": split,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        output_files.append(out_path)
        output_rows += len(out_rows)
        start = stop
    return _SplitBuildResult(
        split=split,
        output_dir=split_dir,
        output_files=tuple(output_files),
        output_rows=int(output_rows),
        expected_rows=int(expected_rows),
        scratch_parts=0,
        input_files=tuple(info.path for info in infos),
        input_rows=int(input_rows),
    )


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
    total_rows: int,
    desired_rows: int,
    used_rows: int,
    window_start: int,
    reason: str,
) -> DenseShuffleResult:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
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
