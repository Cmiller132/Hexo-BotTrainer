"""Optimizer-backed training over KataGo-style shuffled compact CNN rows.

Shards store compact sample facts; each row is expanded to dense tensors at read
time under a fresh per-epoch D6 symmetry (``samples.expand_sample``), so board
symmetry augmentation is exact and re-randomized every epoch.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from . import compact_io
from .config import Model1Config
from .d6 import D6_SIZE
from .losses import model1_loss
from .replay import (
    DenseNpzSampleWindow,
    DenseTrainState,
    build_katago_shuffle,
    load_split_json,
    load_train_json,
    latest_shuffle_dir,
    npz_row_count,
    shuffle_train_files,
    shuffle_validation_files,
)


class DenseCNNTrainer:
    """Dense CNN model owner used by the generic training pipeline."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: Model1Config,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_state = DenseTrainState()
        requested = torch.device(config.device)
        self.device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model.to(device=self.device, memory_format=torch.channels_last)
        else:
            self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp and self.device.type == "cuda")
        # Calibration tunes only these four batch sizes; every other search and
        # training setting is read directly from `config`.
        self.inference_batch_size = 1
        self.selfplay_batch_size = config.selfplay.active_games
        self.mcts_virtual_batch_size: int | None = None
        self.training_batch_size = config.training.batch_size
        # Train-read sample expansion (build_input_planes + D6) is CPU-bound; a
        # persistent spawn process-pool parallelizes whole-shard expansion across
        # cores so the GPU is fed instead of starved. Created lazily, reused across
        # epochs, and skipped for small workloads (tests) and validation.
        self._expand_pool: Any | None = None
        self._expand_workers = _resolve_expand_workers()

    @property
    def sample_count(self) -> int:
        return int(self.train_state.total_num_data_rows)

    def _get_expand_pool(self) -> Any | None:
        """Lazily build the persistent spawn pool (None when parallelism is off)."""

        if self._expand_workers <= 1:
            return None
        if self._expand_pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._expand_pool = ProcessPoolExecutor(
                max_workers=self._expand_workers, mp_context=mp.get_context("spawn")
            )
        return self._expand_pool

    def close(self) -> None:
        """Shut down the expansion pool, if any.

        Called by the pipeline's run-end teardown (and the prefit script) so the
        spawn workers -- each holding ~250-400 MB of torch RSS -- are reclaimed when
        the run finishes instead of lingering until interpreter exit.
        """

        if self._expand_pool is not None:
            self._expand_pool.shutdown(wait=False, cancel_futures=True)
            self._expand_pool = None

    def __del__(self) -> None:
        # Best-effort backstop if the trainer is GC'd without an explicit close().
        # The reliable teardown is the pipeline finally / the prefit close(); a
        # finalizer cannot rely on ProcessPoolExecutor.shutdown at interpreter exit.
        try:
            self.close()
        except Exception:
            pass

    def select_training_samples(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        """Build/read the latest shuffled NPZ window and reserve one training epoch."""

        selfplay_dir = ctx.output_dir / "selfplay"
        shuffled_root = self._shuffled_root(ctx)
        shuffle_result = build_katago_shuffle(
            selfplay_dir=selfplay_dir,
            shuffled_root=shuffled_root,
            scratch_dir=ctx.output_dir / "shufflescratch",
            epoch=epoch,
            seed=(ctx.config.run.seed or 0) + epoch,
            min_rows=self.config.samples.shuffle_min_rows,
            keep_target_rows=self.config.samples.shuffle_keep_target_rows,
            taper_window_exponent=self.config.samples.shuffle_taper_window_exponent,
            expand_window_per_row=self.config.samples.shuffle_expand_window_per_row,
            taper_window_scale=self.config.samples.shuffle_taper_window_scale,
            approx_rows_per_out_file=self.config.samples.approx_rows_per_out_file,
            batch_size=self.training_batch_size,
            worker_group_size=self.config.samples.shuffle_worker_group_size,
            validation_fraction=self.config.samples.validation_fraction,
        )
        shuffle_dir = shuffle_result.shuffle_dir or latest_shuffle_dir(shuffled_root)
        if shuffle_dir is None:
            window = DenseNpzSampleWindow(
                files=(),
                seed=int(ctx.config.run.seed or 0),
                epoch=epoch,
                index=SimpleNamespace(sample_count=0, store=None),
                window_size=0,
                target_rows=0,
                shuffle_dir=None,
                metadata={"shuffle": _shuffle_result_dict(shuffle_result)},
            )
            components.shared.sample_index = window.index
            components.shared.sample_window = window
            return {
                "status": "skipped",
                "epoch": epoch,
                "reason": shuffle_result.reason or "no shuffled data",
                "metadata": dict(window.metadata),
            }

        train_json = load_train_json(shuffle_dir)
        total_rows = int(train_json.get("total_num_data_rows", train_json.get("num_rows", 0)))
        window_start = int(train_json.get("window_start_data_row_idx", 0))
        self._record_shuffle_dir(shuffle_dir)
        self._update_train_bucket(total_rows=total_rows, window_start=window_start)

        requested_rows = int(self.config.training.train_samples_per_epoch)
        if self.train_state.train_bucket_level + 1.0e-9 < requested_rows:
            window = self._empty_window(ctx, epoch, shuffle_dir, total_rows, shuffle_result)
            components.shared.sample_index = window.index
            components.shared.sample_window = window
            return {
                "status": "train_bucket_limited",
                "epoch": epoch,
                "reason": "train_bucket_limited",
                "train_bucket_level": self.train_state.train_bucket_level,
                "requested": requested_rows,
                "metadata": dict(window.metadata),
            }

        train_files = list(shuffle_train_files(shuffle_dir))
        if self.config.training.no_repeat_files:
            train_files = [
                path
                for path in train_files
                if str(path.resolve()) not in self.train_state.data_files_used
            ]
        rng = np.random.default_rng(int(ctx.config.run.seed or 0) + epoch * 65_537)
        selected, selected_rows = _select_files_for_rows(train_files, requested_rows, rng=rng)
        if selected_rows < requested_rows:
            window = self._empty_window(ctx, epoch, shuffle_dir, total_rows, shuffle_result)
            components.shared.sample_index = window.index
            components.shared.sample_window = window
            return {
                "status": "skipped",
                "epoch": epoch,
                "reason": "no new training files",
                "available_rows": selected_rows,
                "requested": requested_rows,
                "metadata": dict(window.metadata),
            }

        validation_files = shuffle_validation_files(shuffle_dir) if self.config.samples.validation_fraction > 0.0 else ()
        validation_rows = _validation_row_count(shuffle_dir) if validation_files else 0
        bucket_before = float(self.train_state.train_bucket_level)
        self.train_state.train_bucket_level = max(0.0, self.train_state.train_bucket_level - requested_rows)
        self.train_state.train_steps_since_last_reload += 1
        window = DenseNpzSampleWindow(
            files=tuple(selected),
            seed=int(ctx.config.run.seed or 0),
            epoch=epoch,
            index=SimpleNamespace(sample_count=total_rows, store=shuffle_dir),
            window_size=requested_rows,
            target_rows=requested_rows,
            shuffle_dir=shuffle_dir,
            validation_files=tuple(validation_files),
            validation_rows=validation_rows,
            metadata={
                "shuffle": _shuffle_result_dict(shuffle_result),
                "shuffle_dir": str(shuffle_dir),
                "total_num_data_rows": total_rows,
                "window_start_data_row_idx": window_start,
                "train_bucket_level_before_consume": bucket_before,
                "train_bucket_level": self.train_state.train_bucket_level,
                "requested": requested_rows,
                "selected_rows": selected_rows,
                "selected_files": [str(path) for path in selected],
                "validation_files": [str(path) for path in validation_files],
                "validation_rows": validation_rows,
            },
        )
        components.shared.sample_index = window.index
        components.shared.sample_window = window
        return {
            "status": "completed",
            "epoch": epoch,
            "sample_count": total_rows,
            "window_size": requested_rows,
            "requested": requested_rows,
            "metadata": dict(window.metadata),
        }

    def train_passes(
        self,
        *,
        passes: int,
        sample_window: DenseNpzSampleWindow,
        sample_symmetries: object,
        ctx: Any,
        components: Any,
        epoch: int,
    ) -> Mapping[str, Any]:
        """Train over shuffled compact rows, expanding each under a fresh D6 symmetry."""

        _ = (passes, ctx, components)
        if sample_window is None or not sample_window.files:
            return {
                "status": "skipped",
                "epoch": epoch,
                "reason": "no dense_cnn shuffled NPZ rows available",
                "train_state": self.train_state.to_dict(),
            }

        self.model.train()
        batch_size = int(self.training_batch_size)
        if batch_size <= 0:
            raise ValueError("dense_cnn training batch size must be > 0")
        horizons = tuple(int(h) for h in self.config.architecture.short_term_value_horizons)
        # Per-(run, epoch) D6 draw: each row is expanded under an independently
        # sampled symmetry, re-randomized every epoch (exact, train-time augmentation).
        sym_rng = np.random.default_rng(_aug_seed(sample_symmetries, sample_window, epoch))
        total_loss = 0.0
        component_sums: dict[str, float] = {}
        steps = 0
        trained_rows = 0
        target_rows = int(sample_window.target_rows)
        started = perf_counter()

        # Draw a whole-shard symmetry vector per file (deterministic file order,
        # re-randomized each epoch), then expand shards into dense arrays — across
        # worker processes for large epochs, serially for small ones.
        shard_syms: list[tuple[Path, np.ndarray]] = []
        total_available = 0
        for file_path in sample_window.files:
            n = int(compact_io.compact_row_count(file_path))
            total_available += n
            shard_syms.append((file_path, sym_rng.integers(0, D6_SIZE, size=n)))

        def _train_arrays(arrays: Mapping[str, np.ndarray]) -> None:
            nonlocal trained_rows, steps, total_loss
            remaining = target_rows - trained_rows
            if remaining <= 0:
                return
            for batch in _batches_from_arrays(arrays, batch_size, remaining):
                trained_rows += int(batch["value"].shape[0])
                loss_components = self._optimizer_step(batch)
                total_loss += loss_components["total"]
                for key, value in loss_components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + value
                steps += 1

        # Record only shards that actually contribute >=1 trained row. Marking every
        # selected file as used (below) would, with no_repeat_files=True, permanently
        # exclude shards that were submitted-but-cancelled or never reached before the
        # row target was met -- shrinking the eligible pool for no benefit.
        consumed: set[Path] = set()
        pool = self._get_expand_pool() if total_available >= _PARALLEL_MIN_ROWS else None
        if pool is None:
            for file_path, syms in shard_syms:
                if trained_rows >= target_rows:
                    break
                before = trained_rows
                _train_arrays(compact_io.expand_shard_to_arrays(file_path, syms, horizons))
                if trained_rows > before:
                    consumed.add(file_path)
        else:
            pending_iter = iter(shard_syms)
            inflight: dict[Any, Path] = {}

            def _submit_one() -> bool:
                for fpath, fsyms in pending_iter:
                    inflight[pool.submit(compact_io.expand_shard_to_arrays, fpath, fsyms, horizons)] = fpath
                    return True
                return False

            for _ in range(self._expand_workers + 2):
                if not _submit_one():
                    break
            while inflight and trained_rows < target_rows:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    fpath = inflight.pop(fut, None)
                    if trained_rows >= target_rows:
                        continue
                    before = trained_rows
                    _train_arrays(fut.result())
                    if trained_rows > before and fpath is not None:
                        consumed.add(fpath)
                    _submit_one()
            for fut in inflight:  # abandon any surplus shards once the row target is met
                fut.cancel()

        if steps <= 0:
            return {
                "status": "skipped",
                "epoch": epoch,
                "reason": "no optimizer steps",
                "train_state": self.train_state.to_dict(),
            }
        self.train_state.global_step_samples += trained_rows
        self.train_state.latest_shuffle_dir = str(sample_window.shuffle_dir) if sample_window.shuffle_dir else None
        for file_path in consumed:
            self.train_state.data_files_used.add(str(file_path.resolve()))

        validation = self._run_validation(sample_window)
        elapsed = perf_counter() - started
        return {
            "status": "completed",
            "epoch": epoch,
            "passes": 1,
            "generic_passes_requested": passes,
            "steps": steps,
            "samples": trained_rows,
            "batch_size": batch_size,
            "loss": total_loss / steps,
            # Per-component epoch means (unweighted CE per head + the weighted total).
            # Read these instead of the bare "loss": across epochs the policy-CE floor
            # drifts with the self-play target entropy, so the scalar loss is not
            # comparable epoch-to-epoch but the component split (esp. value) is legible.
            "loss_components": {key: value / steps for key, value in component_sums.items()},
            "validation": validation,
            "elapsed_seconds": elapsed,
            "samples_per_second": trained_rows / max(elapsed, 1.0e-9),
            "train_state": self.train_state.to_dict(),
        }

    def load_train_state(self, state: Mapping[str, Any] | None) -> None:
        self.train_state = DenseTrainState.from_mapping(state)

    def _optimizer_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        inputs = batch.pop("input")
        if self.device.type == "cuda" and inputs.ndim == 4:
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        else:
            inputs = inputs.to(self.device, non_blocking=True)
        batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.config.training.amp and self.device.type == "cuda"):
            outputs = self.model(inputs)
            loss, components_map = model1_loss(
                outputs,
                batch,
                policy_weight=self.config.training.policy_weight,
                value_weight=self.config.training.value_weight,
                opp_policy_weight=self.config.training.opp_policy_weight,
                short_term_value_weight=self.config.training.short_term_value_weight,
                moves_left_weight=self.config.training.moves_left_weight,
            )
        self.scaler.scale(loss).backward()
        if self.config.training.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Return the UNWEIGHTED per-component losses (policy/value/opp_policy/stvalue_*
        # and the weighted total) so the epoch can report a component breakdown. The
        # bare scalar loss alone is misleading across epochs because the self-play
        # targets drift (e.g. forced-playout widening raises the policy-CE floor).
        return {key: float(tensor.detach().cpu().item()) for key, tensor in components_map.items()}

    @torch.no_grad()
    def _run_validation(self, sample_window: DenseNpzSampleWindow) -> dict[str, Any]:
        if self.config.samples.validation_fraction <= 0.0 or not sample_window.validation_files:
            return {"status": "skipped", "reason": "validation disabled"}
        max_rows = int(self.config.training.max_validation_samples)
        if max_rows <= 0:
            return {"status": "skipped", "reason": "max_validation_samples <= 0"}
        was_training = self.model.training
        self.model.eval()
        batch_size = int(self.training_batch_size)
        steps = 0
        rows_seen = 0
        total_loss = 0.0
        component_sums: dict[str, float] = {}
        horizons = tuple(int(h) for h in self.config.architecture.short_term_value_horizons)
        for file_path in sample_window.validation_files:
            if rows_seen >= max_rows:
                break
            # Identity expansion (no augmentation): validation loss stays deterministic.
            n = int(compact_io.compact_row_count(file_path))
            arrays = compact_io.expand_shard_to_arrays(file_path, np.zeros(n, dtype=np.int64), horizons)
            for batch in _batches_from_arrays(arrays, batch_size, max_rows - rows_seen):
                rows_seen += int(batch["value"].shape[0])
                inputs = batch.pop("input")
                if self.device.type == "cuda" and inputs.ndim == 4:
                    inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(self.device, non_blocking=True)
                batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type=self.device.type, enabled=self.config.training.amp and self.device.type == "cuda"):
                    outputs = self.model(inputs)
                    loss, components_map = model1_loss(
                        outputs,
                        batch,
                        policy_weight=self.config.training.policy_weight,
                        value_weight=self.config.training.value_weight,
                        opp_policy_weight=self.config.training.opp_policy_weight,
                        short_term_value_weight=self.config.training.short_term_value_weight,
                        moves_left_weight=self.config.training.moves_left_weight,
                    )
                total_loss += float(loss.detach().cpu().item())
                for key, tensor in components_map.items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(tensor.detach().cpu().item())
                steps += 1
        if was_training:
            self.model.train()
        if steps <= 0:
            return {"status": "skipped", "reason": "no validation rows"}
        return {
            "status": "completed",
            "samples": int(rows_seen),
            "steps": int(steps),
            "loss": total_loss / steps,
            "loss_components": {key: value / steps for key, value in component_sums.items()},
        }

    def _update_train_bucket(self, *, total_rows: int, window_start: int) -> None:
        cap = max(
            float(self.config.training.max_train_bucket_size),
            float(self.config.training.train_samples_per_epoch),
        )
        if total_rows > self.train_state.train_bucket_level_at_row:
            new_rows = total_rows - self.train_state.train_bucket_level_at_row
            self.train_state.train_bucket_level = min(
                cap,
                self.train_state.train_bucket_level
                + new_rows * self.config.training.max_train_bucket_per_new_data,
            )
            self.train_state.train_bucket_level_at_row = int(total_rows)
        elif total_rows < self.train_state.train_bucket_level_at_row:
            self.train_state.train_bucket_level_at_row = int(total_rows)
            self.train_state.train_steps_since_last_reload = 0
            self.train_state.train_bucket_level = min(self.train_state.train_bucket_level, cap)
        self.train_state.total_num_data_rows = int(total_rows)
        self.train_state.window_start_data_row_idx = int(window_start)

    def _record_shuffle_dir(self, shuffle_dir: Path) -> None:
        train_dir = str((shuffle_dir / "train").resolve())
        dirs = [path for path in self.train_state.old_train_data_dirs if path != train_dir]
        dirs.append(train_dir)
        expired = dirs[:-20]
        kept = dirs[-20:]
        if expired:
            kept_dirs = set(kept)
            self.train_state.data_files_used = {
                item
                for item in self.train_state.data_files_used
                if str(Path(item).resolve().parent) in kept_dirs
            }
        self.train_state.old_train_data_dirs = kept
        self.train_state.latest_shuffle_dir = str(shuffle_dir)

    def _empty_window(
        self,
        ctx: Any,
        epoch: int,
        shuffle_dir: Any,
        total_rows: int,
        shuffle_result: Any,
    ) -> DenseNpzSampleWindow:
        return DenseNpzSampleWindow(
            files=(),
            seed=int(ctx.config.run.seed or 0),
            epoch=epoch,
            index=SimpleNamespace(sample_count=total_rows, store=shuffle_dir),
            window_size=0,
            target_rows=0,
            shuffle_dir=shuffle_dir,
            metadata={
                "shuffle": _shuffle_result_dict(shuffle_result),
                "shuffle_dir": str(shuffle_dir),
                "total_num_data_rows": total_rows,
                "train_bucket_level": self.train_state.train_bucket_level,
            },
        )

    def _shuffled_root(self, ctx: Any) -> Any:
        return ctx.output_dir / "shuffleddata"


def _aug_seed(sample_symmetries: Any, sample_window: DenseNpzSampleWindow, epoch: int) -> int:
    """Deterministic per-(run, epoch) seed for the D6 augmentation draw.

    Prefers the run seed carried by the generic pipeline's `sample_symmetries`
    selection, falling back to the sample window's seed (prefit passes
    `sample_symmetries=None`). Folding `epoch` in re-randomizes each row's
    orientation every epoch while staying reproducible.
    """

    base = None
    if sample_symmetries is not None:
        base = getattr(sample_symmetries, "seed", None)
    if base is None:
        base = getattr(sample_window, "seed", 0)
    return (int(base) * 1_000_003 + int(epoch) * 9_176 + 1) & 0x7FFFFFFF


def _resolve_expand_workers() -> int:
    """Worker count for parallel train-read expansion.

    Overridable via ``DENSE_CNN_EXPAND_WORKERS`` (set to 1 to force serial).
    Defaults to ~half the logical CPUs, capped at 16, leaving headroom for the
    main process feeding the GPU.
    """

    override = os.environ.get("DENSE_CNN_EXPAND_WORKERS")
    if override is not None:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    # Each spawn worker imports torch (~250-400 MB RSS), so default conservatively
    # to bound idle-pool RAM against the WSL host budget / RAM watchdog; bump via
    # DENSE_CNN_EXPAND_WORKERS when there is headroom.
    return min(8, max(1, (os.cpu_count() or 4) // 4))


# Expand below this many rows-per-epoch serially (tests / tiny runs) to skip the
# spawn-pool startup cost; large real epochs use the pool.
_PARALLEL_MIN_ROWS = 2048


def _batches_from_arrays(
    arrays: Mapping[str, "np.ndarray"], batch_size: int, max_rows: int
) -> "list[dict[str, torch.Tensor]]":
    """Slice expanded shard arrays into ``<= max_rows`` worth of training batches."""

    rows = min(int(arrays["value"].shape[0]), int(max_rows))
    batches: list[dict[str, torch.Tensor]] = []
    offset = 0
    while offset < rows:
        take = min(batch_size, rows - offset)
        batches.append(
            {key: torch.from_numpy(np.ascontiguousarray(value[offset:offset + take])) for key, value in arrays.items()}
        )
        offset += take
    return batches


def _select_files_for_rows(
    files: Sequence[Path],
    requested_rows: int,
    *,
    rng: np.random.Generator,
) -> tuple[list[Path], int]:
    candidates = [(path, npz_row_count(path)) for path in files]
    rng.shuffle(candidates)
    selected: list[Path] = []
    deferred: list[tuple[Path, int]] = []
    rows = 0
    for path, row_count in candidates:
        if rows > 0 and rows + row_count > requested_rows:
            overshoot = rows + row_count - requested_rows
            skip_prob = min(1.0, max(0.0, overshoot / max(1, row_count)))
            if rng.random() < skip_prob:
                deferred.append((path, row_count))
                continue
        selected.append(path)
        rows += row_count
        if rows >= requested_rows:
            return selected, rows
    for path, row_count in deferred:
        selected.append(path)
        rows += row_count
        if rows >= requested_rows:
            break
    return selected, rows


def _validation_row_count(shuffle_dir: Path) -> int:
    try:
        val_json = load_split_json(shuffle_dir, split="val")
    except FileNotFoundError:
        return 0
    return int(val_json.get("num_rows", 0))


def _shuffle_result_dict(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "shuffle_dir": str(result.shuffle_dir) if result.shuffle_dir is not None else None,
        "total_num_data_rows": result.total_num_data_rows,
        "desired_rows": result.desired_rows,
        "used_rows": result.used_rows,
        "output_rows": result.output_rows,
        "output_files": [str(path) for path in result.output_files],
        "validation_rows": getattr(result, "validation_rows", 0),
        "validation_files": [str(path) for path in getattr(result, "validation_files", ())],
        "window_start_data_row_idx": getattr(result, "window_start_data_row_idx", 0),
        "reason": result.reason,
    }
