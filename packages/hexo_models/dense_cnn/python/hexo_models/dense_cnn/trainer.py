"""Optimizer-backed training for dense CNN samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from .config import Model1Config
from .d6 import D6Symmetry
from .losses import model1_loss
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
        self.inference_batch_size = 1
        self.selfplay_batch_size = config.selfplay.active_games
        self.mcts_virtual_batch_size: int | None = None
        self.mcts_progressive_widening_initial_actions = config.selfplay.progressive_widening_initial_actions
        self.mcts_progressive_widening_child_initial_actions = config.selfplay.progressive_widening_child_initial_actions
        self.mcts_progressive_widening_candidate_actions = config.selfplay.progressive_widening_candidate_actions
        self.mcts_progressive_widening_growth_interval = config.selfplay.progressive_widening_growth_interval
        self.mcts_progressive_widening_growth_base = config.selfplay.progressive_widening_growth_base
        self.mcts_root_dirichlet_noise_enabled = config.selfplay.root_dirichlet_noise_enabled
        self.mcts_root_dirichlet_noise_fraction = config.selfplay.root_dirichlet_noise_fraction
        self.mcts_root_dirichlet_alpha = config.selfplay.root_dirichlet_alpha
        self.mcts_hidden_prior_mass = config.selfplay.hidden_prior_mass
        self.mcts_fpu_reduction = config.selfplay.fpu_reduction
        self.mcts_virtual_loss = config.selfplay.virtual_loss
        self.mcts_active_root_limit = config.selfplay.mcts_active_root_limit
        self.training_batch_size = config.training.batch_size
        self.search_visits = config.selfplay.search_visits

    @property
    def sample_count(self) -> int:
        return self.buffer.sample_count

    def select_training_samples(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        requested = ctx.config.samples.train_sample_count or self.config.samples.train_sample_count
        records = self.buffer.sample(requested, seed=(ctx.config.run.seed or 0) + epoch)
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
        steps = 0
        started = perf_counter()
        records = tuple(sample_window.records)
        policy_preview: list[dict[str, Any]] = []

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
                    decoded = expand_sample(sample, symmetry=symmetry)
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
                steps += 1

        elapsed = perf_counter() - started
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
            "policy_target_path": str(policy_target_path) if policy_target_path is not None else None,
            "elapsed_seconds": elapsed,
            "samples_per_second": (len(records) * max(1, int(passes))) / max(elapsed, 1.0e-9),
        }


def _policy_preview(policy: Sequence[tuple[int, float]], limit: int = 16) -> list[dict[str, float | int]]:
    return [
        {"action_id": int(action_id), "weight": float(weight)}
        for action_id, weight in tuple(policy)[:limit]
    ]
