"""Optimizer-backed training for Hexformer AR sparse samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from math import cos, pi
from random import Random
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from hexo_utils.samples import SampleRequest, append_samples, refresh_sample_index, sample_training_samples

from .config import HexformerConfig
from .curriculum import generate_tactical_pretraining_records
from .losses import hexformer_loss, policy_symmetry_consistency_loss
from .samples import (
    SAMPLE_NAMESPACE,
    HexformerReplayBuffer,
    SparseSampleWindow,
    collate_compressed_samples,
    compressed_sample_from_training_record,
)


@dataclass(slots=True)
class HexformerTrainer:
    model: torch.nn.Module
    config: HexformerConfig
    buffer: HexformerReplayBuffer
    optimizer: torch.optim.Optimizer
    curriculum_seeded_epochs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        requested = torch.device(self.config.device)
        self.device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.training.amp and self.device.type == "cuda")
        self.training_batch_size = self.config.training.batch_size
        self.search_visits = self.config.selfplay.search_visits
        self.optimizer_step = 0
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.ema_state = (
            {name: value.detach().clone() for name, value in self.model.state_dict().items() if torch.is_floating_point(value)}
            if self.config.training.ema_decay > 0
            else {}
        )

    @property
    def sample_count(self) -> int:
        return self.buffer.sample_count

    def seed_curriculum_if_needed(self, *, store: Any, seed: int, epoch: int) -> bool:
        sample_index = refresh_sample_index(store)
        seeded = self._seed_curriculum_if_empty(store=store, sample_index=sample_index, seed=seed, epoch=epoch)
        if seeded:
            self.curriculum_seeded_epochs.add(int(epoch))
        return seeded

    def select_training_samples(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        requested = int(ctx.config.samples.train_sample_count or self.config.samples.train_sample_count)
        store = getattr(components.shared, "sample_store", None)
        seed = int(ctx.config.run.seed or 0) + int(epoch)
        if store is not None:
            sample_index = refresh_sample_index(store)
            seeded = self._seed_curriculum_if_empty(store=store, sample_index=sample_index, seed=seed, epoch=epoch)
            if seeded:
                self.curriculum_seeded_epochs.add(int(epoch))
                sample_index = refresh_sample_index(store)
            checkpoint_target = self._checkpoint_replay_target(requested)
            shared_requested = max(0, requested - checkpoint_target)
            selected_records, selection_metadata = self._select_mixed_shared_records(
                sample_index=sample_index,
                requested=shared_requested,
                seed=seed,
            )
            checkpoint_records = self.buffer.sample(
                min(
                    self.buffer.sample_count,
                    max(checkpoint_target, requested - len(selected_records)),
                ),
                seed=seed + 17,
            )
            records = tuple(compressed_sample_from_training_record(record) for record in selected_records) + tuple(checkpoint_records)
            if records:
                window = SparseSampleWindow(
                    records=records,
                    seed=seed,
                    epoch=int(epoch),
                    index=sample_index,
                    window_size=len(records),
                    metadata={
                        **selection_metadata,
                        "selected_count": len(records),
                        "indexed_sample_count": sample_index.sample_count,
                        "store": str(sample_index.store.path),
                        "model_family": "hexformer_ar",
                        "source": (
                            "shared_sample_store+checkpoint_replay"
                            if checkpoint_records
                            else "shared_sample_store"
                        ),
                        "curriculum_seeded": seeded or int(epoch) in self.curriculum_seeded_epochs,
                        "shared_selected": len(selected_records),
                        "checkpoint_replay_selected": len(checkpoint_records),
                    },
                )
                components.shared.sample_index = sample_index
                components.shared.sample_window = window
                return {
                    "epoch": epoch,
                    "sample_count": sample_index.sample_count,
                    "window_size": len(records),
                    "requested": requested,
                    "metadata": dict(window.metadata),
                }

        records = self.buffer.sample(requested, seed=(ctx.config.run.seed or 0) + epoch)
        window = SparseSampleWindow(
            records=records,
            seed=seed,
            epoch=int(epoch),
            index=SimpleNamespace(sample_count=self.buffer.sample_count, store=store),
            window_size=len(records),
            metadata={
                "buffer_count": self.buffer.sample_count,
                "requested": requested,
                "recency_halflife": self.buffer.recency_halflife,
                "model_family": "hexformer_ar",
                "source": "checkpoint_replay_buffer",
            },
        )
        components.shared.sample_index = window.index
        components.shared.sample_window = window
        return {
            "epoch": epoch,
            "sample_count": self.buffer.sample_count,
            "window_size": len(records),
            "requested": requested,
            "metadata": dict(window.metadata),
        }

    def _seed_curriculum_if_empty(self, *, store: Any, sample_index: Any, seed: int, epoch: int) -> bool:
        curriculum_count = int(self.config.curriculum.synthetic_samples)
        if curriculum_count <= 0:
            return False
        existing = sample_training_samples(
            sample_index,
            SampleRequest(
                count=int(getattr(sample_index, "sample_count", 0)),
                required_extensions=(SAMPLE_NAMESPACE,),
            ),
        )
        if existing.records:
            return False
        records = generate_tactical_pretraining_records(
            count=curriculum_count,
            architecture=self.config.architecture,
            curriculum=self.config.curriculum,
            seed=seed,
        )
        if not records:
            return False
        append_samples(
            store,
            records,
            metadata={
                "epoch": epoch,
                "model_family": "hexformer_ar",
                "curriculum": "synthetic_tactical_pretraining",
                "compression": self.config.samples.compression,
                "extensions": {SAMPLE_NAMESPACE: 1},
            },
        )
        return True

    def _checkpoint_replay_target(self, requested: int) -> int:
        if requested <= 0 or self.buffer.sample_count <= 0:
            return 0
        return min(self.buffer.sample_count, max(1, int(round(requested * 0.25))))

    def _select_mixed_shared_records(self, *, sample_index: Any, requested: int, seed: int) -> tuple[tuple[object, ...], dict[str, Any]]:
        batch = sample_training_samples(
            sample_index,
            SampleRequest(
                count=int(getattr(sample_index, "sample_count", 0)),
                required_extensions=(SAMPLE_NAMESPACE,),
            ),
        )
        records = tuple(batch.records)
        if len(records) <= requested:
            return records, {
                **dict(batch.metadata),
                "selection_mode": "all_available",
                "hard_selected": sum(1 for record in records if _is_hard_record(record)),
                "recent_selected": len(records),
            }
        rng = Random(seed)
        requested = max(0, int(requested))
        hard_pool = [record for record in records if _is_hard_record(record)]
        recent_pool = list(records[-max(requested * 4, requested):])
        hard_target = min(len(hard_pool), int(round(requested * self.config.samples.hard_sample_fraction)))
        recent_target = min(len(recent_pool), int(round(requested * self.config.samples.recent_sample_fraction)))
        selected: list[object] = []
        selected_ids: set[tuple[str, int]] = set()

        def add_many(pool: list[object], target: int) -> int:
            if target <= 0:
                return 0
            rng.shuffle(pool)
            added = 0
            for record in pool:
                key = _record_key(record)
                if key in selected_ids:
                    continue
                selected.append(record)
                selected_ids.add(key)
                added += 1
                if added >= target or len(selected) >= requested:
                    break
            return added

        hard_selected = add_many(hard_pool, hard_target)
        recent_selected = add_many(recent_pool, recent_target)
        add_many(list(records), requested - len(selected))
        selected.sort(key=lambda record: _record_key(record))
        return tuple(selected[:requested]), {
            **dict(batch.metadata),
            "selection_mode": "mixed_recent_hard_archive",
            "available_records": len(records),
            "hard_pool": len(hard_pool),
            "hard_selected": hard_selected,
            "recent_selected": recent_selected,
            "hard_sample_fraction": self.config.samples.hard_sample_fraction,
            "recent_sample_fraction": self.config.samples.recent_sample_fraction,
        }

    def train_passes(
        self,
        *,
        passes: int,
        sample_window: SparseSampleWindow,
        sample_symmetries: object,
        ctx: Any,
        components: Any,
        epoch: int,
    ) -> Mapping[str, Any]:
        _ = components
        if sample_window is None or not sample_window.records:
            return {
                "status": "skipped",
                "epoch": epoch,
                "passes": passes,
                "reason": "no hexformer_ar samples available",
                "buffer_count": self.buffer.sample_count,
            }
        self.model.train()
        batch_size = max(1, int(self.training_batch_size))
        total_loss = 0.0
        steps = 0
        started = perf_counter()
        records = tuple(sample_window.records)
        symmetries = tuple(getattr(sample_symmetries, "symmetries", ()))
        for _pass_index in range(max(1, int(passes))):
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                chunk_symmetries = symmetries[start : start + len(chunk)] if symmetries else ()
                batch = collate_compressed_samples(
                    chunk,
                    architecture=self.config.architecture,
                    symmetries=chunk_symmetries,
                )
                batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                self.optimizer.zero_grad(set_to_none=True)
                self._update_learning_rate()
                with torch.autocast(device_type=self.device.type, enabled=self.config.training.amp and self.device.type == "cuda"):
                    outputs = self.model(batch)
                    loss, components_map = hexformer_loss(
                        outputs,
                        batch,
                        policy_weight=self.config.training.policy_weight,
                        wdl_weight=self.config.training.wdl_weight,
                        distance_weight=self.config.training.distance_weight,
                        opponent_policy_weight=self.config.training.opponent_policy_weight,
                        lookahead_weight=self.config.training.lookahead_weight,
                        threat_weight=self.config.training.threat_weight,
                        relevance_weight=self.config.training.relevance_weight,
                    )
                    if self.config.training.symmetry_weight > 0 and _has_non_identity_symmetry(chunk_symmetries):
                        reference_batch = collate_compressed_samples(chunk, architecture=self.config.architecture)
                        reference_batch = {key: value.to(self.device, non_blocking=True) for key, value in reference_batch.items()}
                        reference_outputs = self.model(reference_batch)
                        symmetry_loss = policy_symmetry_consistency_loss(
                            outputs["policy_logits"],
                            reference_outputs["policy_logits"],
                            batch["candidate_mask"].to(self.device),
                        )
                        components_map["symmetry"] = symmetry_loss
                        loss = loss + self.config.training.symmetry_weight * symmetry_loss
                        components_map["total"] = loss
                self.scaler.scale(loss).backward()
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer_step += 1
                self._update_ema()
                total_loss += float(loss.detach().cpu().item())
                steps += 1
        elapsed = perf_counter() - started
        return {
            "status": "completed",
            "epoch": epoch,
            "passes": passes,
            "steps": steps,
            "samples": len(records),
            "batch_size": batch_size,
            "loss": total_loss / max(1, steps),
            "elapsed_seconds": elapsed,
            "samples_per_second": (len(records) * max(1, int(passes))) / max(elapsed, 1.0e-9),
            "optimizer_step": self.optimizer_step,
            "ema_decay": self.config.training.ema_decay,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer_step": int(self.optimizer_step),
            "ema_state": {key: value.detach().cpu() for key, value in self.ema_state.items()},
            "curriculum_seeded_epochs": tuple(sorted(self.curriculum_seeded_epochs)),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.optimizer_step = int(state.get("optimizer_step", 0))
        ema_state = state.get("ema_state", {})
        if isinstance(ema_state, Mapping):
            self.ema_state = {
                str(key): value.detach().clone().to(self.device)
                for key, value in ema_state.items()
                if isinstance(value, torch.Tensor)
            }
        self.curriculum_seeded_epochs = {
            int(item)
            for item in state.get("curriculum_seeded_epochs", ())
        }

    def _update_learning_rate(self) -> None:
        step = max(1, self.optimizer_step + 1)
        warmup = max(0, int(self.config.training.warmup_steps))
        decay_steps = max(1, int(self.config.training.cosine_decay_steps))
        min_lr = float(self.config.training.min_learning_rate)
        if warmup > 0 and step <= warmup:
            scale = step / warmup
            lrs = [base_lr * scale for base_lr in self.base_lrs]
        else:
            progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
            cosine = 0.5 * (1.0 + cos(pi * progress))
            lrs = [min_lr + (base_lr - min_lr) * cosine for base_lr in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def _update_ema(self) -> None:
        decay = float(self.config.training.ema_decay)
        if decay <= 0:
            return
        state = self.model.state_dict()
        for name, value in state.items():
            if not torch.is_floating_point(value):
                continue
            detached = value.detach()
            if name not in self.ema_state:
                self.ema_state[name] = detached.clone()
            else:
                self.ema_state[name].mul_(decay).add_(detached, alpha=1.0 - decay)

    @contextmanager
    def ema_weights(self):
        if not self.ema_state:
            yield
            return
        state = self.model.state_dict()
        backup = {name: state[name].detach().clone() for name in self.ema_state if name in state}
        try:
            patched = dict(state)
            for name, value in self.ema_state.items():
                if name in patched:
                    patched[name] = value.to(device=patched[name].device, dtype=patched[name].dtype)
            self.model.load_state_dict(patched, strict=False)
            yield
        finally:
            restored = dict(self.model.state_dict())
            for name, value in backup.items():
                restored[name] = value.to(device=restored[name].device, dtype=restored[name].dtype)
            self.model.load_state_dict(restored, strict=False)


def _has_non_identity_symmetry(symmetries: tuple[object, ...]) -> bool:
    return any(int(getattr(symmetry, "index", symmetry)) != 0 for symmetry in symmetries)


def _record_key(record: object) -> tuple[str, int]:
    return str(getattr(record, "game_id", "")), int(getattr(record, "turn_index", 0))


def _is_hard_record(record: object) -> bool:
    metadata = getattr(record, "metadata", {})
    if isinstance(metadata, dict) and metadata.get("hard"):
        return True
    root_value = metadata.get("root_value") if isinstance(metadata, dict) else None
    try:
        if root_value is not None and abs(float(root_value)) < 0.35:
            return True
    except (TypeError, ValueError):
        pass
    policy = getattr(record, "policy", None)
    logits = getattr(policy, "logits", None)
    if logits:
        values = [max(0.0, float(item)) for item in logits]
        total = sum(values)
        if total > 0:
            top = max(values) / total
            return top < 0.40
    return False
