"""hexo_train trainer: per-epoch passes over the recent-shard window.

Window v1: mtime-ordered most-recent shards up to
`shuffle_keep_target_rows` rows (PCR filtering already happened at the
source — only Full rows are written; truncated games never written). The
1:1 restnet policy-surprise/taper port is the scheduled M9 upgrade and is
tracked there.
"""

from __future__ import annotations

import json
import os
import random
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
from .losses import hexfield_loss
from .samples import STV_HORIZONS, expand_sample
from .shards import read_compact_shard
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
        # restored by the loader on the RESUME branch only. Starts fresh here;
        # the window/governor mechanism that drives it lands in a later phase.
        self.train_state = HexfieldTrainState()

    def _window_paths(self, ctx) -> list:
        return sorted(
            ctx.samples_dir.glob("epoch_*/game_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

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

    def _window(self, ctx, paths=None) -> list:
        shard_paths = paths if paths is not None else self._window_paths(ctx)
        rows = []
        target = self.config.training.shuffle_keep_target_rows
        for path in shard_paths:
            rows.extend(read_compact_shard(path))
            if len(rows) >= target:
                break
        return rows

    def train_passes(self, *, passes, sample_window, sample_symmetries, ctx, components, epoch) -> dict[str, Any]:
        _ = sample_symmetries
        # sample_window is the path list from select_training_samples (or None
        # if called directly, e.g. in tests).
        paths = sample_window if isinstance(sample_window, list) else None
        rows = self._window(ctx, paths)
        if not rows:
            return {"status": "skipped", "epoch": epoch, "reason": "empty sample window"}
        rng = random.Random(epoch * 7919 + 13)
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
        # targets are off the now-smaller legal set (see the per-chunk skip below).
        tolerate_off_legal = int(os.environ.get("HEXFIELD_SUPPORT_RADIUS", "8")) < 8
        for _pass in range(max(int(passes), 1)):
            order = list(range(len(rows)))
            rng.shuffle(order)
            for start in range(0, len(order), batch_rows):
                chunk = [rows[i] for i in order[start : start + batch_rows]]
                # When HEXFIELD_SUPPORT_RADIUS<8 the replay window still holds
                # radius-8 samples whose policy targets fall outside the radius-4
                # legal set; skip those (they age out of the window) rather than
                # tripping the hard-error wire, which stays armed at the default.
                expanded = []
                for s in chunk:
                    try:
                        expanded.append(expand_sample(s, symmetry=rng.randrange(12)))
                    except ValueError as e:
                        if tolerate_off_legal and "off the legal set" in str(e):
                            continue
                        raise
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

        grads = np.asarray(grad_norms or [0.0])
        result = {
            "status": "completed",
            "epoch": epoch,
            "passes": passes,
            "window_rows": len(rows),
            "steps": steps,
            "seconds": round(time.time() - started, 1),
            **{f"loss_{k}": v / max(steps, 1) for k, v in comp_totals.items()},
            "grad_norm_mean": float(grads.mean()),
            "grad_norm_p95": float(np.percentile(grads, 95)),
            "clip_fraction": float((grads > self.config.training.grad_clip).mean()),
            "amp_scale": float(self.scaler.get_scale()) if self.device.type == "cuda" else None,
        }
        diag_path = ctx.diagnostics_dir / f"hexfield.training.epoch_{epoch:06d}.json"
        diag_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result
