"""HexgtTrainer — training loop with dense_cnn discipline for the graph model.

Mirrors `DenseCNNTrainer`'s discipline (AdamW, AMP autocast, grad-clip, the
65-bin value loss, per-component loss reporting, KataGo-style train-state that
round-trips through the checkpoint) but consumes packed-graph batches and uses
the dynamic per-graph policy CE (`losses.hexgt_loss`). Inputs are recomputed at
expand time from the compact raw-fact shards (`expand.py`).

This Phase-4 trainer owns the optimizer step, the warmup LR schedule, and a
shard-driven training pass sufficient for the CPU gate ("a CPU training pass
decreases loss"). The full KataGo replay-window / train-bucket integration
(reusing dense_cnn's model-agnostic replay/shuffle) was deferred to the GPU
phases and never landed here — the production RL driver (scripts/_rl_train.py)
implements its own replay-window selection around `train_on_shards`, and the
lineage is now halted. Beware: `train_on_shards` loads every shard's rows into
one Python list, fine for driver-sized selections but not a full replay window.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import HexgtConfig
from .expand import build_training_batch
from .losses import hexgt_loss


# Transient CUDA-driver errors we RETRY (a WSL2/consumer-GPU TDR-style reset
# surfaces as "device not ready" at the next sync — observed crashing the training
# forward AND backward). Real, non-recoverable errors are excluded so they still
# raise immediately and are never masked.
_TRANSIENT_CUDA_MARKERS = ("device not ready", "cudaerrornotready")
_NONTRANSIENT_CUDA_MARKERS = (
    "out of memory", "illegal memory access", "device-side assert",
    "misaligned address", "cublas", "cudnn",
)


def _is_transient_cuda_error(exc: BaseException) -> bool:
    """True only for transient, retryable CUDA driver hiccups. OOM, illegal-access,
    device-side asserts, shape/library errors are NEVER matched (they re-raise)."""
    msg = str(exc).lower()
    if any(m in msg for m in _NONTRANSIENT_CUDA_MARKERS):
        return False
    return any(m in msg for m in _TRANSIENT_CUDA_MARKERS)


@dataclass
class HexgtTrainState:
    """KataGo-style train-bucket state persisted in the checkpoint."""

    global_step: int = 0
    global_step_samples: int = 0
    total_num_data_rows: int = 0
    data_files_used: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "global_step_samples": int(self.global_step_samples),
            "total_num_data_rows": int(self.total_num_data_rows),
            "data_files_used": sorted(self.data_files_used),
        }

    @classmethod
    def from_mapping(cls, state: Mapping[str, Any] | None) -> "HexgtTrainState":
        if not state:
            return cls()
        return cls(
            global_step=int(state.get("global_step", 0)),
            global_step_samples=int(state.get("global_step_samples", 0)),
            total_num_data_rows=int(state.get("total_num_data_rows", 0)),
            data_files_used=set(str(p) for p in state.get("data_files_used", ())),
        )


class HexgtTrainer:
    def __init__(self, *, model: torch.nn.Module, config: HexgtConfig, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_state = HexgtTrainState()
        self.training_batch_size = int(config.training.batch_size)
        self.inference_batch_size = 1
        self.selfplay_batch_size = 1
        self.mcts_virtual_batch_size: int | None = None
        self._base_lr = float(config.training.learning_rate)
        self._warmup_steps = max(0, int(config.training.warmup_steps))
        # Optional resume-safe exponential LR decay keyed to global_step (see config).
        self._lr_decay_enabled = bool(config.training.lr_decay_enabled)
        self._lr_decay_start_step = int(config.training.lr_decay_start_step)
        self._lr_decay_halflife = float(config.training.lr_decay_halflife_steps)
        self._lr_min = float(config.training.lr_min)
        self._max_grad_norm = float(config.training.max_grad_norm)
        self._amp = bool(config.training.amp)
        # AMP loss scaling. `optimizer_step` runs the forward under `torch.autocast`
        # with NO explicit dtype, so on CUDA the autocast dtype is fp16 (the torch
        # default), whose ~6e-5 smallest-normal underflows small gradients to zero
        # without loss scaling. A `GradScaler` scales the loss up before backward and
        # unscales the grads before the clip/step, recovering those gradients (bf16
        # would not need this -- it keeps fp32's exponent range -- but this path is
        # fp16, matching the fp16 inference evaluator). The scaler is a no-op when AMP
        # is off or on CPU. Its scale factor is intentionally NOT checkpointed (it
        # re-warms from the default within a few hundred steps on resume), matching
        # dense_cnn's trainer.
        scaler_device = next(model.parameters()).device
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self._amp and scaler_device.type == "cuda")
        )
        self._horizons = tuple(int(h) for h in config.architecture.short_term_value_horizons)
        self._candidate_radius = int(config.architecture.candidate_radius)
        self._prune_max_dropped_mass = float(config.samples.bc_prune_max_dropped_mass)
        self.pruned_rows = 0
        self.kept_rows = 0
        self._weights = dict(
            policy_weight=config.training.policy_weight,
            value_weight=config.training.value_weight,
            opp_policy_weight=config.training.opp_policy_weight,
            short_term_value_weight=config.training.short_term_value_weight,
        )
        # Retry guard for transient CUDA driver hiccups (see _is_transient_cuda_error).
        # A step is retried up to this many times (synchronize + backoff) before it
        # falls through to the crash path. cuda_retry_log: optional callback(msg) the
        # driver sets to surface retries in its log (defaults to stderr).
        self.cuda_retry_attempts = 3
        self.cuda_retry_backoff = 0.5
        self.cuda_retry_log = None

    @property
    def sample_count(self) -> int:
        return int(self.train_state.total_num_data_rows)

    @property
    def prune_rate(self) -> float:
        """Fraction of expanded rows pruned for out-of-candidate policy mass."""

        seen = self.pruned_rows + self.kept_rows
        return self.pruned_rows / seen if seen else 0.0

    def load_train_state(self, state: Mapping[str, Any] | None) -> None:
        self.train_state = HexgtTrainState.from_mapping(state)

    def _lr_at_step(self, step: int) -> float:
        if self._warmup_steps > 0 and step < self._warmup_steps:
            return self._base_lr * float(step + 1) / float(self._warmup_steps)
        # Post-warmup exponential decay (resume-safe: pure function of `step`, which
        # is the checkpointed global_step). Flat at base until lr_decay_start_step,
        # then halve every lr_decay_halflife_steps, clamped at lr_min. Disabled or a
        # non-positive halflife -> flat base_lr (the prior behavior).
        if (
            self._lr_decay_enabled
            and self._lr_decay_halflife > 0.0
            and step >= self._lr_decay_start_step
        ):
            decayed = self._base_lr * 2.0 ** (
                -float(step - self._lr_decay_start_step) / self._lr_decay_halflife
            )
            return max(self._lr_min, decayed)
        return self._base_lr

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def optimizer_step(self, batch: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, float]:
        """One optimizer step on a packed-graph batch; returns loss components."""

        self.model.train()
        device = next(self.model.parameters()).device
        batch = _to_device(batch, device)
        targets = _to_device(targets, device)

        lr = self._lr_at_step(int(self.train_state.global_step))
        self._set_lr(lr)
        use_amp = self._amp and device.type == "cuda"

        # Run the step under a retry guard: a transient CUDA driver hiccup (e.g. a
        # WSL2/consumer-GPU TDR-style "device not ready", which has crashed both the
        # forward SDPA and loss.backward()) is recovered by synchronize + brief
        # backoff + retrying the WHOLE step. The optimizer hasn't stepped on a failed
        # attempt, so a retry is safe. Real errors (OOM/illegal-access/shape) are not
        # matched and re-raise immediately; exhausting retries re-raises too (the
        # supervisor then relaunches and the epoch is re-trained).
        attempts = max(1, int(self.cuda_retry_attempts))
        components: dict[str, torch.Tensor] = {}
        for attempt in range(attempts):
            try:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = self.model(batch)
                    loss, components = hexgt_loss(outputs, targets, **self._weights)
                # Scale the loss before backward so fp16 gradients do not underflow;
                # unscale before the grad-norm clip so the clip threshold applies to the
                # true (unscaled) gradients; `scaler.step` skips the optimizer update on
                # an inf/NaN gradient and `scaler.update` adapts the scale. The
                # transient-CUDA retries documented below occur in the forward/backward
                # (SDPA / loss.backward); `scaler.scale()` only multiplies the loss and
                # registers no per-optimizer state, so re-running the whole step on a
                # retry is safe (no double `unscale_`).
                self.scaler.scale(loss).backward()
                if self._max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                break
            except RuntimeError as exc:
                if attempt >= attempts - 1 or not _is_transient_cuda_error(exc):
                    raise
                msg = (f"[trainer] transient CUDA error on step {self.train_state.global_step} "
                       f"(attempt {attempt + 1}/{attempts}): {str(exc).splitlines()[0]} "
                       f"-> synchronize + retry")
                (self.cuda_retry_log or (lambda m: print(m, file=sys.stderr, flush=True)))(msg)
                if device.type == "cuda":
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                time.sleep(self.cuda_retry_backoff * (attempt + 1))

        self.train_state.global_step += 1
        self.train_state.global_step_samples += int(batch["num_graphs"])
        report = {k: float(v.detach()) for k, v in components.items()}
        report["lr"] = lr
        report["dropped_policy_mass"] = float(targets.get("_dropped_policy_mass", 0.0))
        return report

    def train_on_rows(
        self,
        rows: Sequence[Any],
        *,
        batch_size: int | None = None,
        max_steps: int | None = None,
        epochs: int = 1,
    ) -> list[dict[str, float]]:
        """Expand + train over compact rows (CPU gate path). Returns loss history."""

        bs = int(batch_size or self.training_batch_size)
        history: list[dict[str, float]] = []
        step = 0
        for _ in range(epochs):
            for start in range(0, len(rows), bs):
                chunk = rows[start : start + bs]
                if not chunk:
                    continue
                batch, targets = build_training_batch(
                    chunk,
                    n=self._candidate_radius,
                    horizons=self._horizons,
                    prune_max_dropped_mass=self._prune_max_dropped_mass,
                )
                if batch is None:  # every row in this chunk was pruned
                    self.pruned_rows += len(chunk)
                    continue
                self.pruned_rows += int(targets.get("_pruned", 0))
                self.kept_rows += int(targets.get("_kept", 0))
                history.append(self.optimizer_step(batch, targets))
                self.train_state.total_num_data_rows += int(targets.get("_kept", len(chunk)))
                step += 1
                if max_steps is not None and step >= max_steps:
                    return history
        return history

    def train_on_shards(self, shard_paths: Sequence[Path], **kwargs: Any) -> list[dict[str, float]]:
        """Read compact shards and train over their rows (reuses dense_cnn IO)."""

        from hexo_models.dense_cnn.compact_io import read_compact_shard

        rows: list[Any] = []
        for path in shard_paths:
            rows.extend(read_compact_shard(Path(path)))
            self.train_state.data_files_used.add(str(Path(path).resolve()))
        return self.train_on_rows(rows, **kwargs)

    def close(self) -> None:
        pass


def _to_device(d: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out
