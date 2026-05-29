"""Optimizer-backed training for dense CNN samples."""

from __future__ import annotations

from collections import Counter
import ctypes
import gc
import os
from dataclasses import dataclass, field
from math import ceil, exp, log
from random import Random
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from .config import Model1Config
from .d6 import D6Symmetry
from .losses import decode_binned_value, model1_loss
from .samples import CompressedSample, SampleBuffer, expand_sample, stack_expanded


@dataclass(slots=True)
class DenseSampleWindow:
    records: tuple[CompressedSample, ...]
    seed: int
    epoch: int
    index: Any
    window_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DenseCNNTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: Model1Config,
        buffer: SampleBuffer,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.buffer = buffer
        self.optimizer = optimizer
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
        self.inference_batch_size = max(1, max(config.performance.inference_batch_candidates or (1,)))
        self.selfplay_batch_size = config.selfplay.active_games
        virtual_candidates = tuple(int(item) for item in config.performance.mcts_virtual_batch_candidates)
        self.mcts_virtual_batch_size: int | None = max(1, virtual_candidates[0]) if virtual_candidates else None
        self.mcts_progressive_widening_initial_actions = config.selfplay.progressive_widening_initial_actions
        self.mcts_progressive_widening_child_initial_actions = config.selfplay.progressive_widening_child_initial_actions
        self.mcts_progressive_widening_candidate_actions = config.selfplay.progressive_widening_candidate_actions
        self.mcts_progressive_widening_growth_interval = config.selfplay.progressive_widening_growth_interval
        self.mcts_progressive_widening_growth_base = config.selfplay.progressive_widening_growth_base
        self.mcts_active_root_limit = config.selfplay.mcts_active_root_limit
        self.training_batch_size = max(1, max(config.performance.training_batch_candidates or (config.training.batch_size,)))
        self.search_visits = config.selfplay.search_visits

    @property
    def sample_count(self) -> int:
        return self.buffer.sample_count

    def select_training_samples(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        requested = ctx.config.samples.train_sample_count or self.config.samples.train_sample_count
        records = _select_training_records(
            self.buffer,
            requested,
            seed=(ctx.config.run.seed or 0) + epoch,
            classical_replay_min_fraction=self.config.samples.classical_replay_min_fraction,
        )
        source_summary = _summarize_sample_records(records)
        window = DenseSampleWindow(
            records=records,
            seed=int(ctx.config.run.seed or 0),
            epoch=epoch,
            index=SimpleNamespace(sample_count=self.buffer.sample_count, store=getattr(components.shared, "sample_store", None)),
            window_size=len(records),
            metadata={
                "buffer_count": self.buffer.sample_count,
                "requested": requested,
                "recency_halflife": self.buffer.recency_halflife,
                "classical_replay_min_fraction": self.config.samples.classical_replay_min_fraction,
                **source_summary,
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

    def train_passes(
        self,
        *,
        passes: int,
        sample_window: DenseSampleWindow,
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
                "reason": "no dense_cnn samples available",
                "buffer_count": self.buffer.sample_count,
            }

        self.model.train()
        batch_size = max(1, int(self.training_batch_size))
        symmetries = tuple(getattr(sample_symmetries, "symmetries", ()))
        total_loss = 0.0
        component_loss_totals: dict[str, float] = {}
        component_loss_counts: dict[str, int] = {}
        steps = 0
        started = perf_counter()
        last_progress_write = 0.0
        records = tuple(sample_window.records)
        source_summary = _summarize_sample_records(records)
        policy_preview: list[dict[str, Any]] = []
        total_steps = max(1, int(passes)) * max(1, ceil(len(records) / batch_size))

        rng = Random((int(ctx.config.run.seed or 0) + 1) * 1_000_003 + int(epoch))
        for pass_index in range(max(1, int(passes))):
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                expanded = []
                for offset, sample in enumerate(chunk):
                    absolute_index = start + offset
                    _ = symmetries
                    symmetry = D6Symmetry(rng.randrange(12))
                    data = sample.decode() if isinstance(sample, CompressedSample) else sample
                    decoded = expand_sample(data, symmetry=symmetry)
                    for horizon in self.config.architecture.lookahead_horizons:
                        key = f"lookahead_{int(horizon)}"
                        mask_key = f"{key}_mask"
                        if key in decoded:
                            decoded[mask_key] = torch.tensor(1.0, dtype=torch.float32)
                        else:
                            decoded[key] = torch.tensor(0.0, dtype=torch.float32)
                            decoded[mask_key] = torch.tensor(0.0, dtype=torch.float32)
                    if len(policy_preview) < 8:
                        nonzero = torch.nonzero(decoded["policy"] > 0, as_tuple=False).flatten().tolist()
                        opp_nonzero = torch.nonzero(decoded["opp_policy"] > 0, as_tuple=False).flatten().tolist()
                        policy_preview.append(
                            {
                                "sample_index": absolute_index,
                                "pass_index": pass_index,
                                "game_id": data.game_id,
                                "turn_index": data.turn_index,
                                "current_player": data.current_player,
                                "phase": data.phase,
                                "value": float(data.value),
                                "lookahead": {int(horizon): float(value) for horizon, value in data.lookahead},
                                "symmetry": int(getattr(symmetry, "index", symmetry)),
                                "nonzero_policy_cells": [int(item) for item in nonzero[:16]],
                                "nonzero_opp_policy_cells": [int(item) for item in opp_nonzero[:16]],
                                "policy": _policy_preview(data.policy),
                                "opp_policy": _policy_preview(data.opp_policy),
                                "metadata": dict(data.metadata),
                            }
                        )
                    expanded.append(decoded)
                batch = stack_expanded(expanded)
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
                        lookahead_weight=self.config.training.lookahead_weight,
                    )
                self.scaler.scale(loss).backward()
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += float(loss.detach().cpu().item())
                for key, value in components_map.items():
                    component_loss_totals[key] = component_loss_totals.get(key, 0.0) + float(value.detach().cpu().item())
                    component_loss_counts[key] = component_loss_counts.get(key, 0) + 1
                steps += 1
                now = perf_counter()
                if steps == 1 or now - last_progress_write >= 30.0:
                    _write_training_progress(
                        ctx,
                        epoch=epoch,
                        status="running",
                        pass_index=pass_index,
                        passes=max(1, int(passes)),
                        steps=steps,
                        total_steps=total_steps,
                        samples=len(records),
                        batch_size=batch_size,
                        started=started,
                        loss=total_loss / max(1, steps),
                        loss_components=_average_components(component_loss_totals, component_loss_counts),
                        source_summary=source_summary,
                    )
                    last_progress_write = now

        elapsed = perf_counter() - started
        loss_components = _average_components(component_loss_totals, component_loss_counts)
        _write_training_progress(
            ctx,
            epoch=epoch,
            status="completed",
            pass_index=max(0, max(1, int(passes)) - 1),
            passes=max(1, int(passes)),
            steps=steps,
            total_steps=total_steps,
            samples=len(records),
            batch_size=batch_size,
            started=started,
            loss=total_loss / max(1, steps),
            loss_components=loss_components,
            source_summary=source_summary,
        )
        policy_imitation = (
            self._policy_imitation_metrics(records, batch_size=batch_size)
            if self.config.debug.write_policy_targets
            else None
        )
        policy_target_path = None
        if self.config.debug.write_policy_targets:
            policy_target_path = ctx.diagnostics.write_json(
                f"dense_cnn.policy_targets.epoch_{epoch:06d}.json",
                {
                    "epoch": epoch,
                    "sample_count": len(records),
                    "d6": {
                        "mode": "random_per_training_expansion",
                        "preview_count": len(policy_preview),
                    },
                    "source_summary": source_summary,
                    "loss_components": loss_components,
                    "policy_imitation": policy_imitation,
                    "preview": policy_preview,
                },
            )
        return {
            "status": "completed",
            "epoch": epoch,
            "passes": passes,
            "steps": steps,
            "samples": len(records),
            "batch_size": batch_size,
            "loss": total_loss / max(1, steps),
            "loss_components": loss_components,
            "source_summary": source_summary,
            "policy_imitation": policy_imitation,
            "policy_target_path": str(policy_target_path) if policy_target_path is not None else None,
            "elapsed_seconds": elapsed,
            "samples_per_second": (len(records) * max(1, int(passes))) / max(elapsed, 1.0e-9),
        }

    def _policy_imitation_metrics(
        self,
        records: Sequence[CompressedSample],
        *,
        batch_size: int,
    ) -> dict[str, Any]:
        if not records:
            return {"status": "skipped", "reason": "no records"}

        was_training = self.model.training
        metrics: dict[str, dict[str, float]] = {}
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                chunk = tuple(records[start : start + batch_size])
                decoded = [sample.decode() if isinstance(sample, CompressedSample) else sample for sample in chunk]
                expanded = [expand_sample(data, symmetry=0) for data in decoded]
                batch = stack_expanded(expanded)
                inputs = batch["input"]
                if self.device.type == "cuda" and inputs.ndim == 4:
                    inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(self.device, non_blocking=True)
                policy = batch["policy"].to(self.device, non_blocking=True)
                legal_mask = batch["legal_mask"].to(self.device, non_blocking=True).bool()
                values = batch["value"].to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.training.amp and self.device.type == "cuda",
                ):
                    outputs = self.model(inputs)

                target = policy / policy.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                masked_logits = outputs["policy"].float().masked_fill(~legal_mask, -1.0e9)
                probs = torch.softmax(masked_logits, dim=-1)
                top_k = min(3, probs.shape[-1])
                top_indices = torch.topk(probs, k=top_k, dim=-1).indices
                target_indices = target.argmax(dim=-1)
                target_mass = (probs * target).sum(dim=-1)
                value_pred = decode_binned_value(outputs["value"].float())
                value_mae = (value_pred - values.float()).abs()
                value_sign_match = (value_pred >= 0.0) == (values.float() >= 0.0)

                top1 = (top_indices[:, 0] == target_indices).detach().cpu().tolist()
                top3 = (top_indices == target_indices.unsqueeze(-1)).any(dim=-1).detach().cpu().tolist()
                mass = target_mass.detach().cpu().tolist()
                mae = value_mae.detach().cpu().tolist()
                sign = value_sign_match.detach().cpu().tolist()

                for index, data in enumerate(decoded):
                    label = _sample_source(data)
                    _record_metric(metrics, "overall", top1[index], top3[index], mass[index], sign[index], mae[index])
                    _record_metric(metrics, label, top1[index], top3[index], mass[index], sign[index], mae[index])

        if was_training:
            self.model.train()
        return {
            "status": "completed",
            "samples": len(records),
            "overall": _finalize_metric_bucket(metrics.get("overall", {})),
            "by_source": {
                key: _finalize_metric_bucket(value)
                for key, value in sorted(metrics.items())
                if key != "overall"
            },
        }

    def release_epoch_resources(self, *, epoch: int) -> dict[str, Any]:
        """Release resident memory from large transient MCTS/inference epochs."""

        cuda_before: dict[str, float] = {}
        cuda_after: dict[str, float] = {}
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_before = {
                "allocated_gb": float(torch.cuda.memory_allocated(self.device) / 1024**3),
                "reserved_gb": float(torch.cuda.memory_reserved(self.device) / 1024**3),
            }
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
            cuda_after = {
                "allocated_gb": float(torch.cuda.memory_allocated(self.device) / 1024**3),
                "reserved_gb": float(torch.cuda.memory_reserved(self.device) / 1024**3),
            }

        collected = gc.collect()
        working_set_trimmed = False
        trim_error = None
        if os.name == "nt":
            try:
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                working_set_trimmed = bool(ctypes.windll.psapi.EmptyWorkingSet(handle))
            except (AttributeError, OSError) as exc:
                trim_error = repr(exc)

        return {
            "status": "completed",
            "epoch": int(epoch),
            "gc_collected": int(collected),
            "cuda_before": cuda_before,
            "cuda_after": cuda_after,
            "working_set_trimmed": working_set_trimmed,
            "trim_error": trim_error,
        }


def _policy_preview(policy: Sequence[tuple[int, float]], limit: int = 16) -> list[dict[str, float | int]]:
    return [
        {"action_id": int(action_id), "weight": float(weight)}
        for action_id, weight in tuple(policy)[:limit]
    ]


def _average_components(
    component_loss_totals: Mapping[str, float],
    component_loss_counts: Mapping[str, int],
) -> dict[str, float]:
    return {
        key: float(component_loss_totals[key]) / max(1, int(component_loss_counts.get(key, 0)))
        for key in sorted(component_loss_totals)
    }


def _write_training_progress(
    ctx: Any,
    *,
    epoch: int,
    status: str,
    pass_index: int,
    passes: int,
    steps: int,
    total_steps: int,
    samples: int,
    batch_size: int,
    started: float,
    loss: float,
    loss_components: Mapping[str, float],
    source_summary: Mapping[str, Any],
) -> None:
    elapsed = perf_counter() - started
    samples_seen = min(int(samples) * int(passes), int(steps) * int(batch_size))
    ctx.diagnostics.write_json(
        f"dense_cnn.training_progress.epoch_{epoch:06d}.json",
        {
            "epoch": int(epoch),
            "status": status,
            "pass_index": int(pass_index),
            "passes": int(passes),
            "steps": int(steps),
            "total_steps": int(total_steps),
            "samples": int(samples),
            "batch_size": int(batch_size),
            "samples_seen": int(samples_seen),
            "progress": min(1.0, int(steps) / max(1, int(total_steps))),
            "loss": float(loss),
            "loss_components": dict(loss_components),
            "source_summary": dict(source_summary),
            "elapsed_seconds": elapsed,
            "samples_per_second": samples_seen / max(elapsed, 1.0e-9),
        },
    )


def _select_training_records(
    buffer: SampleBuffer,
    requested: int,
    *,
    seed: int | None,
    classical_replay_min_fraction: float,
) -> tuple[CompressedSample, ...]:
    requested = max(0, int(requested))
    if requested <= 0:
        return ()
    replay_fraction = min(1.0, max(0.0, float(classical_replay_min_fraction)))
    if replay_fraction <= 0.0:
        return buffer.sample(requested, seed=seed)

    entries = tuple(buffer.entries)
    if not entries:
        return ()
    target_count = min(requested, len(entries))
    rng = Random(seed)
    classical_indices: list[int] = []
    for index, sample in enumerate(entries):
        data = sample.decode() if isinstance(sample, CompressedSample) else sample
        if _is_classical_source(_sample_source(data)):
            classical_indices.append(index)

    required_classical = min(len(classical_indices), ceil(target_count * replay_fraction))
    selected = _weighted_indices(
        classical_indices,
        required_classical,
        population_size=len(entries),
        recency_halflife=buffer.recency_halflife,
        rng=rng,
    )
    selected_set = set(selected)
    remaining_indices = [index for index in range(len(entries)) if index not in selected_set]
    selected.extend(
        _weighted_indices(
            remaining_indices,
            target_count - len(selected),
            population_size=len(entries),
            recency_halflife=buffer.recency_halflife,
            rng=rng,
        )
    )
    return tuple(entries[index] for index in selected)


def _weighted_indices(
    indices: Sequence[int],
    count: int,
    *,
    population_size: int,
    recency_halflife: float,
    rng: Random,
) -> list[int]:
    if count <= 0 or not indices:
        return []
    keys: list[tuple[float, int]] = []
    for index in indices:
        age = population_size - 1 - int(index)
        weight = exp(-log(2.0) * age / max(float(recency_halflife), 1.0e-6))
        keys.append((-log(max(1.0e-12, rng.random())) / max(weight, 1.0e-12), int(index)))
    keys.sort(key=lambda item: item[0])
    return [index for _key, index in keys[: min(count, len(keys))]]


def _summarize_sample_records(records: Sequence[CompressedSample]) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    players: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    values: Counter[str] = Counter()
    for sample in records:
        data = sample.decode() if isinstance(sample, CompressedSample) else sample
        sources[_sample_source(data)] += 1
        players[str(data.current_player)] += 1
        phases[str(data.phase)] += 1
        if data.value > 0.0:
            values["positive"] += 1
        elif data.value < 0.0:
            values["negative"] += 1
        else:
            values["zero"] += 1
    return {
        "source_counts": _counter_dict(sources),
        "current_player_counts": _counter_dict(players),
        "phase_counts": _counter_dict(phases),
        "value_sign_counts": _counter_dict(values),
    }


def _sample_source(data: Any) -> str:
    metadata = dict(getattr(data, "metadata", {}) or {})
    source = metadata.get("sample_source") or metadata.get("bootstrap") or metadata.get("source")
    if source is None:
        game_id = str(getattr(data, "game_id", ""))
        if game_id.startswith("classical-"):
            source = "classical"
        elif game_id.startswith("selfplay-"):
            source = "mcts"
        else:
            source = "unknown"
    return str(source)


def _is_classical_source(source: str) -> bool:
    normalized = source.lower()
    return normalized.startswith("classical") or "bootstrap" in normalized


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _record_metric(
    metrics: dict[str, dict[str, float]],
    label: str,
    top1: bool,
    top3: bool,
    target_mass: float,
    value_sign_match: bool,
    value_mae: float,
) -> None:
    bucket = metrics.setdefault(
        label,
        {
            "count": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "target_mass": 0.0,
            "value_sign_match": 0.0,
            "value_mae": 0.0,
        },
    )
    bucket["count"] += 1.0
    bucket["top1"] += 1.0 if top1 else 0.0
    bucket["top3"] += 1.0 if top3 else 0.0
    bucket["target_mass"] += float(target_mass)
    bucket["value_sign_match"] += 1.0 if value_sign_match else 0.0
    bucket["value_mae"] += float(value_mae)


def _finalize_metric_bucket(bucket: Mapping[str, float]) -> dict[str, float | int]:
    count = int(bucket.get("count", 0.0))
    denominator = max(1, count)
    return {
        "count": count,
        "top1_accuracy": float(bucket.get("top1", 0.0) / denominator),
        "top3_accuracy": float(bucket.get("top3", 0.0) / denominator),
        "mean_target_mass": float(bucket.get("target_mass", 0.0) / denominator),
        "value_sign_accuracy": float(bucket.get("value_sign_match", 0.0) / denominator),
        "value_mae": float(bucket.get("value_mae", 0.0) / denominator),
    }
