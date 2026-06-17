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
from .config import HexfieldConfig
from .losses import hexfield_loss
from .samples import STV_HORIZONS, expand_sample
from .shards import read_compact_shard


class HexfieldTrainer:
    def __init__(self, *, model, config: HexfieldConfig, optimizer):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = torch.device(config.device)
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.global_step = 0

    def _window_paths(self, ctx) -> list:
        return sorted(
            ctx.samples_dir.glob("epoch_*/game_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def select_training_samples(self, *, ctx, components, epoch: int) -> dict[str, Any]:
        """hexo_train dispatch hook: own the window selection so the framework's
        shared-store path is bypassed (we set uses_shared_sample_store=False).
        The window is the mtime-ordered recent shards up to the keep target;
        the heavy row read happens in train_passes (PCR/truncation filtering
        already done at the source)."""

        paths = self._window_paths(ctx)
        components.shared.sample_window = paths
        return {
            "epoch": epoch,
            "shard_count": len(paths),
            "keep_target_rows": self.config.training.shuffle_keep_target_rows,
        }

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
