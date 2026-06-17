"""hexo_train trainer: per-epoch KataGo / dense_cnn_restnet replay window.

Window selection (``select_training_samples``) is the KataGo / dense port
(PLAN §3.1-3.8, §5, §6): a mtime-free ``(generation, game_key)`` manifest →
power-law taper → recent-window cut → md5 split → keep_prob subsample →
overshoot-skip file selection → train-bucket reuse governor → an in-RAM packed
columnar :class:`~hexfield.window.PackedWindow`.

The consumer (``train_passes``, PLAN §3.4/§4.4/§4.5/§6 — Phase 5/6) drains that
PackedWindow with a SINGLE pass and no within-epoch repeat:

1. **Pre-draw** (on the main thread, before any expansion) a per-row D6 vector
   ``d6: int[n]`` from ``np.random.default_rng(_aug_seed(run_seed, epoch))`` and
   a survivor permutation from ``np.random.default_rng(_perm_seed(run_seed,
   epoch))`` — replacing the v1 in-loop ``random.Random.shuffle`` /
   ``rng.randrange(12)`` (the two MUST-FIX determinism violations, PLAN §4.4).
2. **Expand** all rows through ``expand_backends.expand_rows`` under the
   configured backend (Phase 6: ``serial`` | ``pool`` — the spawn ProcessPool —
   with a ``rust`` Phase-7 hook). Each backend maps ``window.row_view(i)`` → a
   ``PackedRowView`` → ``HexfieldSampleData`` shim → the unchanged
   ``expand_sample``, returning a per-row validity mask (off-legal rows flagged
   invalid, NOT dropped in-worker, PLAN §4.5 step 1). ``pool == serial``
   element-wise because all randomness is pre-drawn and results are reassembled
   in original row order (PLAN §4.4).
3. **Filter** survivors (validity mask), **permute** the survivor index,
   **truncate** to ``effective_rows`` (the load-bearing fidelity point — §3.4/M3).
4. **Micro-bucket** (``pair_budget_microbuckets``) — UNCHANGED.
5. loss / optimizer / AMP / grad-clip block — UNCHANGED.

The backend is ``config.training.expand_backend`` (env ``HEXFIELD_EXPAND``
overrides) and the pool worker count is ``config.training.expand_workers`` (env
``HEXFIELD_EXPAND_WORKERS`` overrides; ``0`` ⇒ auto ``min(8, cpu//4)``). The
persistent spawn pool is owned by the trainer (``_get_expand_pool``) and torn
down by ``close()`` at run end.

``effective_rows`` is threaded from ``select_training_samples`` via
``self._last_select`` (the trainer instance backs both dispatch calls); a direct
``train_passes`` call without a prior selection recomputes it from the window +
config.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

from .batching import (
    PAD_QUANTUM,
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from .buffer_manifest import scan_or_update_manifest
from .config import HexfieldConfig
from .expand_backends import (
    _row_view_to_sample,  # re-exported for back-compat (tests import it here)
    expand_rows,
    resolve_expand_workers,
)
from .losses import hexfield_loss
from .samples import STV_HORIZONS
from .train_state import HexfieldTrainState
from .window import (
    PackedWindow,
    _select_files_for_rows,
    _split_by_md5,
    build_window_split,
    compute_katago_window_rows,
    keep_prob as _keep_prob,
    select_recent_window,
)

# D6 augmentation cardinality (geometry.apply_d6 accepts 0-11). Verified 12 in
# PLAN §4.2, matching the v1 ``rng.randrange(12)`` this phase replaces.
D6_SIZE = 12


def _aug_seed(run_seed: int, epoch: int) -> int:
    """Deterministic per-(run, epoch) seed for the D6 augmentation draw.

    Mirrors ``dense_cnn_restnet.trainer._aug_seed`` (``trainer.py:545-559``): a
    stable fold of ``(run_seed, epoch)`` so each row's orientation re-randomizes
    every epoch while staying reproducible. ALL D6 randomness is drawn from this
    seed on the main thread BEFORE expansion (PLAN §3.8/§4.4) — no ``rng`` call
    ever happens per-row inside the loop.
    """
    return (int(run_seed) * 1_000_003 + int(epoch) * 9_176 + 1) & 0x7FFFFFFF


def _perm_seed(run_seed: int, epoch: int) -> int:
    """Deterministic per-(run, epoch) seed for the SURVIVOR permutation (PLAN
    §4.5 step 3). A distinct fold from :func:`_aug_seed` so the permutation
    stream is independent of the D6 stream; both are pure functions of
    ``(run_seed, epoch)`` so the same seed yields an identical survivor index and
    D6 vector (the Phase-5 determinism gate)."""
    return (int(run_seed) * 2_654_435_761 + int(epoch) * 40_503 + 7) & 0x7FFFFFFF


class HexfieldTrainer:
    def __init__(self, *, model, config: HexfieldConfig, optimizer):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = torch.device(config.device)
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.global_step = 0
        # Persisted KataGo-style train-bucket governor + window bookkeeping
        # (PLAN §6/M1). Serialized into the checkpoint meta by the saver and
        # restored by the loader on the RESUME branch only. Starts fresh here.
        self.train_state = HexfieldTrainState()
        # Per-epoch selection bookkeeping stashed by select_training_samples and
        # read back by train_passes (the same trainer instance backs both
        # dispatch calls). Threads effective_rows / window_start / reuse_ratio /
        # train_bucket_level into the consumer without touching the slotted
        # SharedComponents (PLAN §6: effective_rows comes from selection).
        self._last_select: dict[int, dict[str, Any]] = {}
        # Persistent spawn process-pool for the parallel ("pool") expand backend
        # (PLAN §4.3 rung 2). Created lazily on first pool-eligible epoch, reused
        # across epochs, and torn down by close() (the pipeline run-end teardown,
        # pipeline.py:104). None until needed; stays None for the serial backend.
        self._expand_pool: Any | None = None

    def _get_expand_pool(self):
        """Lazily build the persistent spawn pool for ``expand_backend="pool"``.

        Mirrors ``dense_cnn_restnet.trainer._get_expand_pool``: one
        ``ProcessPoolExecutor(mp_context="spawn")`` reused across epochs. Returns
        ``None`` when only one worker is resolved (no benefit to a pool — the
        backend then runs the serial in-process path). ``spawn`` is mandatory on
        WSL/Windows (no fork); the pool inherits the parent environment, so each
        worker re-reads the same ``HEXFIELD_SUPPORT_RADIUS`` at ``support`` import
        time and stays consistent with the main thread.
        """
        n_workers = resolve_expand_workers(self.config.training.expand_workers)
        if n_workers <= 1:
            return None
        if self._expand_pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._expand_pool = ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp.get_context("spawn")
            )
        return self._expand_pool

    def close(self) -> None:
        """Shut down the expansion pool, if any (PLAN §4.3).

        Called by the generic pipeline's run-end teardown (``pipeline.py:104``,
        best-effort ``getattr(trainer, "close")``) so the spawn workers are
        reclaimed when the run finishes instead of lingering until interpreter
        exit. Safe to call when no pool was ever created (serial backend).
        """
        if self._expand_pool is not None:
            self._expand_pool.shutdown(wait=False, cancel_futures=True)
            self._expand_pool = None

    def __del__(self) -> None:
        # Best-effort backstop if the trainer is GC'd without an explicit close().
        # The reliable teardown is the pipeline finally; a finalizer cannot rely
        # on ProcessPoolExecutor.shutdown at interpreter exit.
        try:
            self.close()
        except Exception:  # noqa: BLE001 - finalizer must never raise
            pass

    def _update_train_bucket(self, total_rows: int, window_start: int) -> None:
        """Accrue / clamp the train-bucket reuse governor (PLAN §3.5).

        Ported from ``dense_cnn_restnet.trainer.DenseCNNTrainer._update_train_bucket``
        (``trainer.py:481-499``) onto :class:`HexfieldTrainState`. ``total_rows`` is
        the **monotone** ``cumulative_rows_ever`` from the manifest (PLAN §3.5/M2 —
        NEVER the live total, which can shrink when shards are pruned and would
        spuriously trip the ``elif`` reload branch).

        * ``cap = max(max_train_bucket_size, train_samples_per_epoch)``.
        * each fresh row credits the bucket by ``max_train_bucket_per_new_data``,
          clamped at ``cap``, advancing ``level_at_row`` to ``total_rows``;
        * a *decrease* in ``total_rows`` (window regenerated/shrank) re-bases the
          watermark, zeroes ``steps_since_last_reload``, and re-clamps the level.
        """
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

    def select_training_samples(self, *, ctx, components, epoch: int) -> dict[str, Any]:
        """KataGo / dense_cnn_restnet window selection (PLAN §3.1-3.8, §5, §6).

        Mirrors ``DenseCNNTrainer.select_training_samples`` (``trainer.py:124-254``)
        with the hexfield divergences (no disk re-shard; in-RAM ``PackedWindow``):

        1. ``scan_or_update_manifest`` -> the mtime-free ``(generation,game_key)``
           ordered shard manifest (live ``total_rows`` for window selection; the
           monotone ``cumulative_rows_ever`` for the governor — PLAN §3.5/M2).
        2. ``compute_katago_window_rows`` power-law taper, clamped ``max(_,
           min_rows)``; ``select_recent_window`` newest->oldest whole-shard cut.
        3. ``_update_train_bucket(cumulative_rows_ever, window_start)`` — accrual is
           driven by the monotone counter; ``window_start = max(0, total_rows -
           used)`` is the live-window bookkeeping (computed first so the recorded
           ``window_start`` is the final one, exactly as dense reads it back from
           ``train.json`` — ``trainer.py:168,170``).
        4. ``_split_by_md5`` per-file train/val partition (real, PLAN §3.6/M5).
        5. ``_select_files_for_rows`` overshoot-skip selection capped at
           ``train_samples_per_epoch`` (``no_repeat_files`` honored first — default
           OFF for hexfield, PLAN §8/M2).
        6. ``effective_rows = min(requested, selected)``; the bucket throttle
           (``train_bucket_limited``) or debit-by-``effective_rows`` (PLAN §3.5).
        7. ``build_window_split`` keep_prob-subsamples + concats the survivors into
           one ``PackedWindow`` -> ``components.shared.sample_window``.

        Determinism (PLAN §3.8): a single ``np.random.default_rng((seed)+epoch)``
        drives keep_prob; a separate ``np.random.default_rng((seed)+epoch*65537)``
        drives the file selection. ``seed = (ctx.config.run.seed or 0)``.
        """
        cfg = self.config.training
        seed = int(ctx.config.run.seed or 0)

        # (1) manifest: live total drives window selection; the monotone counter
        # drives the governor (PLAN §3.5/M2).
        manifest = scan_or_update_manifest(ctx.samples_dir)
        entries = manifest.entries
        total_rows = int(manifest.total_rows)
        cumulative_rows_ever = int(manifest.cumulative_rows_ever)
        # New rows credited to the governor since the last accrual (for the
        # diagnostic reuse_ratio); captured BEFORE _update_train_bucket mutates it.
        prev_level_at_row = int(self.train_state.train_bucket_level_at_row)
        new_rows_this_epoch = max(0, cumulative_rows_ever - prev_level_at_row)

        # (2) taper window + recent-window cut.
        desired = compute_katago_window_rows(
            total_rows,
            min_rows=cfg.shuffle_min_rows,
            expand_window_per_row=cfg.shuffle_expand_window_per_row,
            taper_window_exponent=cfg.shuffle_taper_window_exponent,
            taper_window_scale=cfg.shuffle_taper_window_scale,
        )
        desired = max(int(desired), int(cfg.shuffle_min_rows))
        selected_window, used = select_recent_window(entries, desired)
        window_start = max(0, total_rows - used)

        # (3) governor accrual on the monotone counter (record the live window_start).
        self._update_train_bucket(cumulative_rows_ever, window_start)

        def _skip(status: str, reason: str, **extra) -> dict[str, Any]:
            components.shared.sample_window = PackedWindow.empty()
            base = {
                "status": status,
                "epoch": epoch,
                "reason": reason,
                "total_rows": cumulative_rows_ever,
                "live_total_rows": total_rows,
                "desired_rows": int(desired),
                "used_rows": int(used),
                "window_start": window_start,
                "train_bucket_level": float(self.train_state.train_bucket_level),
            }
            base.update(extra)
            # Stash for the consumer: an empty/limited selection trains nothing,
            # so effective_rows is 0 and the reuse_ratio is carried through.
            self._last_select[epoch] = {
                "effective_rows": int(base.get("effective_rows", 0) or 0),
                "window_start": int(window_start),
                "reuse_ratio": float(base.get("reuse_ratio", 0.0) or 0.0),
                "train_bucket_level": float(self.train_state.train_bucket_level),
            }
            self._write_select_diag(ctx, epoch, base)
            return base

        if not selected_window:
            return _skip("skipped", "no files selected for the window",
                         keep_prob=1.0, effective_rows=0, window_rows=0, reuse_ratio=0.0)

        kp = _keep_prob(used, int(cfg.shuffle_keep_target_rows))

        # (4) md5 train/val split (default validation_fraction=0.0 -> all-train).
        train_entries, _val_entries = _split_by_md5(
            selected_window, validation_fraction=float(cfg.validation_fraction)
        )
        if not train_entries:
            return _skip("skipped", "no train files after md5 validation split",
                         keep_prob=kp, effective_rows=0, window_rows=0, reuse_ratio=0.0)

        # (5) overshoot-skip selection, capped at train_samples_per_epoch.
        # no_repeat_files (default OFF for hexfield single-game shards) filters first.
        candidate_entries = train_entries
        if cfg.no_repeat_files:
            candidate_entries = [
                e for e in train_entries if str(e.rel_path) not in self.train_state.data_files_used
            ]
        requested_rows = int(cfg.train_samples_per_epoch)
        sel_rng = np.random.default_rng(seed + epoch * 65_537)
        selected_files, selected_rows = _select_files_for_rows(
            candidate_entries, requested_rows, sel_rng
        )
        if selected_rows <= 0:
            return _skip("skipped", "no new training files",
                         keep_prob=kp, effective_rows=0, window_rows=0, reuse_ratio=0.0,
                         requested=requested_rows, selected_rows=selected_rows)

        # (6) effective_rows = min(requested, selected); the bucket throttle / debit.
        effective_rows = min(requested_rows, selected_rows)
        if self.train_state.train_bucket_level + 1.0e-9 < effective_rows:
            return _skip("train_bucket_limited", "train_bucket_limited",
                         keep_prob=kp, effective_rows=int(effective_rows), window_rows=0,
                         reuse_ratio=effective_rows / max(1, new_rows_this_epoch),
                         requested=requested_rows, selected_rows=selected_rows)
        # Debit at SELECTION time (dense semantics — a later short pass does not
        # refund, trainer.py:217). steps_since_last_reload++ tracks reuse.
        self.train_state.train_bucket_level = max(
            0.0, self.train_state.train_bucket_level - effective_rows
        )
        self.train_state.train_steps_since_last_reload += 1

        # (7) build the packed in-RAM window: per-row keep_prob subsample + concat,
        # consumed via the SINGLE shared per-epoch rng in (generation, game_key)
        # order (PLAN §3.3/§3.8).
        keep_rng = np.random.default_rng(seed + epoch)
        window = build_window_split(
            selected_files, keep_prob=kp, rng=keep_rng, samples_dir=ctx.samples_dir
        )
        components.shared.sample_window = window

        result = {
            "status": "completed",
            "epoch": epoch,
            "total_rows": cumulative_rows_ever,
            "live_total_rows": total_rows,
            "desired_rows": int(desired),
            "used_rows": int(used),
            "keep_prob": float(kp),
            "effective_rows": int(effective_rows),
            "window_rows": int(window.n),
            "window_start": window_start,
            "train_bucket_level": float(self.train_state.train_bucket_level),
            "reuse_ratio": effective_rows / max(1, new_rows_this_epoch),
            "selected_files": len(selected_files),
            "selected_rows": int(selected_rows),
        }
        # Stash effective_rows / window_start / reuse_ratio / train_bucket_level
        # for train_passes (threaded via the trainer instance — the slotted
        # SharedComponents only carries the opaque PackedWindow, PLAN §6).
        self._last_select[epoch] = {
            "effective_rows": int(effective_rows),
            "window_start": int(window_start),
            "reuse_ratio": float(result["reuse_ratio"]),
            "train_bucket_level": float(self.train_state.train_bucket_level),
        }
        self._write_select_diag(ctx, epoch, result)
        return result

    def _write_select_diag(self, ctx, epoch: int, result: dict[str, Any]) -> None:
        """Persist the per-epoch selection summary next to the training diag
        (best-effort; never raises into the dispatch path)."""
        try:
            diag_dir = getattr(ctx, "diagnostics_dir", None)
            if diag_dir is None:
                return
            path = diag_dir / f"hexfield.select.epoch_{epoch:06d}.json"
            path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        except OSError:
            pass

    def _effective_rows_for(self, window: PackedWindow, epoch: int) -> int:
        """Row cap for this epoch's single pass (PLAN §3.4/M3).

        Prefers the ``effective_rows`` ``select_training_samples`` stashed for
        ``epoch`` (the bucket-debited count the governor honored). When
        ``train_passes`` is called WITHOUT a prior selection (a direct test call),
        recompute the faithful equivalent ``min(window.n,
        train_samples_per_epoch)`` so the truncation contract still holds.
        """
        stashed = self._last_select.get(epoch)
        if stashed is not None:
            return int(stashed["effective_rows"])
        return min(int(window.n), int(self.config.training.train_samples_per_epoch))

    def train_passes(self, *, passes, sample_window, sample_symmetries, ctx, components, epoch) -> dict[str, Any]:
        # The opaque PackedWindow is self-drawn for D6 (PLAN §6 opaque-window
        # guard); the framework's sample_symmetries selection is intentionally
        # ignored (the v1 ``_ = sample_symmetries`` contract is preserved).
        _ = sample_symmetries
        window = sample_window if isinstance(sample_window, PackedWindow) else None
        if window is None or window.n <= 0:
            return {"status": "skipped", "epoch": epoch, "reason": "empty sample window"}

        seed = int(ctx.config.run.seed or 0)
        # --- PRE-DRAW all randomness on the main thread (PLAN §3.8/§4.4) -------
        # (a) a per-row D6 vector in WINDOW row order, and (b) the survivor
        # permutation; both pure functions of (seed, epoch). This replaces the v1
        # in-loop ``random.Random.shuffle`` + ``rng.randrange(12)`` — the two
        # MUST-FIX determinism violations. Drawn here, consumed positionally.
        d6 = np.random.default_rng(_aug_seed(seed, epoch)).integers(
            0, D6_SIZE, size=int(window.n), dtype=np.int64
        )

        self.model.train().to(self.device)
        # RESUME SAFETY (the effective fix for the epoch-11 crash loop): the
        # checkpoint is loaded with map_location="cpu" while the model is still
        # on CPU at load time, so AdamW's per-parameter state (exp_avg/
        # exp_avg_sq) lands on CPU. The model is moved to self.device only on the
        # line above — so by the FIRST optimizer.step() the model params are on
        # cuda but the optimizer state is still on CPU, giving "Expected all
        # tensors on the same device, cuda:0 and cpu" (_foreach_lerp_). Move the
        # optimizer state to the model's device now (no-op after epoch 1).
        for _st in self.optimizer.state.values():
            for _k, _v in _st.items():
                if isinstance(_v, torch.Tensor) and _v.device != self.device:
                    _st[_k] = _v.to(self.device)
        batch_rows = self.config.training.batch_rows
        comp_totals: dict[str, float] = {}
        grad_norms: list[float] = []
        steps = 0
        started = time.time()
        # Radius transition: tolerate (skip) replay-buffer samples whose policy
        # targets are off the now-smaller legal set (the off-legal validity mask
        # below). The hard-error wire stays armed at the default radius.
        tolerate_off_legal = int(os.environ.get("HEXFIELD_SUPPORT_RADIUS", "8")) < 8

        # (1) EXPAND ALL window rows under their pre-drawn D6 via the configured
        # backend (PLAN §4.3 ladder: serial | pool | rust). The backend returns a
        # per-row ExpandedRow list aligned to range(window.n) plus a `valid` mask;
        # an off-legal row is flagged invalid (NOT dropped in-worker, PLAN §4.5
        # step 1). The PackedRowView -> shim -> expand_sample path keeps the
        # expansion math frozen (samples.py), so pool == serial element-wise.
        backend = str(os.environ.get("HEXFIELD_EXPAND", self.config.training.expand_backend))
        expand_pool = None
        if backend == "pool":
            # Reuse the trainer's persistent spawn pool. When the resolved worker
            # count is <= 1 (degenerate / 1-2 CPU host, or HEXFIELD_EXPAND_WORKERS=1)
            # _get_expand_pool returns None and there is no benefit to a pool, so we
            # fall back to the in-process serial path (matches dense, trainer.py:321).
            expand_pool = self._get_expand_pool()
            if expand_pool is None:
                backend = "serial"
        expanded_rows, valid = expand_rows(
            window,
            None,  # expand ALL rows; the survivor filter + truncation happen below
            d6,
            STV_HORIZONS,
            tolerate_off_legal=tolerate_off_legal,
            backend=backend,
            workers=self.config.training.expand_workers,
            pool=expand_pool,
        )

        # (2) FILTER survivors on the main thread using the validity mask (PLAN
        # §4.5 step 2). This compacted list is element-identical to the v1 inline
        # loop's survivors (same expansion, same off-legal drops, same order),
        # independent of which backend produced it.
        survivors: list = [row for row, ok in zip(expanded_rows, valid) if ok]
        rows_skipped_off_legal = int((~np.asarray(valid, dtype=bool)).sum())

        # (3) PERMUTE the SURVIVOR index (drawn over the POST-skip set, PLAN §4.5
        # step 3), then (4) TRUNCATE to effective_rows (PLAN §3.4/M3 — the
        # load-bearing fidelity point: single pass, no within-epoch repeat, capped
        # at exactly the bucket-debited effective_rows).
        n_surv = len(survivors)
        perm = np.random.default_rng(_perm_seed(seed, epoch)).permutation(n_surv)
        effective_rows = self._effective_rows_for(window, epoch)
        keep = perm[: max(0, int(effective_rows))]
        ordered_rows = [survivors[int(j)] for j in keep]

        # (5) MICRO-BUCKET (pair_budget_microbuckets) — UNCHANGED. One optimizer
        # step per nominal batch of ``batch_rows`` survivors; the VRAM split +
        # (6) loss/optimizer/AMP/grad-clip block below are byte-for-byte the v1
        # path (PLAN §6: do not change the loss block or the micro-bucketing).
        for start in range(0, len(ordered_rows), batch_rows):
            expanded = ordered_rows[start : start + batch_rows]
            if not expanded:
                continue
            denoms = step_global_denominators(expanded, STV_HORIZONS)
            self.optimizer.zero_grad(set_to_none=True)
            for bucket in pair_budget_microbuckets(expanded, quantize=PAD_QUANTUM):
                # Same quantum the budget split assumed (PAD_QUANTUM), so the
                # live (B,4,S,S) transient honours PAIR_BUDGET — see §6.3.
                pad_to = -(-max(r.support.num_nodes for r in bucket) // PAD_QUANTUM) * PAD_QUANTUM
                batch = split_stvalue_columns(
                    collate_training(bucket, pad_to=pad_to), STV_HORIZONS
                )
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    out = self.model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
                loss, comps = hexfield_loss(out, batch, denominators=denoms)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at epoch {epoch} step {steps}: "
                        f"{ {k: float(v) for k, v in comps.items()} }"
                    )
                self.scaler.scale(loss).backward()
                for key, val in comps.items():
                    comp_totals[key] = comp_totals.get(key, 0.0) + float(val.detach())
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip
            )
            if torch.isfinite(norm):
                grad_norms.append(float(norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            steps += 1
            self.global_step += 1

        trained_rows = len(ordered_rows)
        self.train_state.global_step_samples += trained_rows

        if steps <= 0:
            return {
                "status": "skipped",
                "epoch": epoch,
                "reason": "no optimizer steps (all rows skipped off-legal or empty)",
                "window_rows": int(window.n),
                "rows_skipped_off_legal": int(rows_skipped_off_legal),
            }

        stashed = self._last_select.get(epoch, {})
        grads = np.asarray(grad_norms or [0.0])
        result = {
            "status": "completed",
            "epoch": epoch,
            # Single pass, no within-epoch repeat (KataGo / dense semantics,
            # PLAN §2/§3.4); the generic ``passes`` request is reported but not
            # multiplied (mirrors dense trainer.py:371-374).
            "passes": 1,
            "generic_passes_requested": passes,
            "window_rows": int(window.n),
            "trained_rows": int(trained_rows),
            "steps": steps,
            "seconds": round(time.time() - started, 1),
            **{f"loss_{k}": v / max(steps, 1) for k, v in comp_totals.items()},
            "grad_norm_mean": float(grads.mean()),
            "grad_norm_p95": float(np.percentile(grads, 95)),
            "clip_fraction": float((grads > self.config.training.grad_clip).mean()),
            "amp_scale": float(self.scaler.get_scale()) if self.device.type == "cuda" else None,
            # New Phase-5 diagnostics (PLAN §6/§7/§9/S2).
            "reuse_ratio": float(stashed.get("reuse_ratio", 0.0)),
            "train_bucket_level": float(
                stashed.get("train_bucket_level", self.train_state.train_bucket_level)
            ),
            "train_steps_since_last_reload": int(self.train_state.train_steps_since_last_reload),
            "rows_skipped_off_legal": int(rows_skipped_off_legal),
        }
        diag_path = ctx.diagnostics_dir / f"hexfield.training.epoch_{epoch:06d}.json"
        diag_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result
