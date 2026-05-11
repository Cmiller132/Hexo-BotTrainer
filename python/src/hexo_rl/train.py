"""PyTorch training loop skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from models_common.model_api import ModelPlugin, load_model_plugin
from models_common.replay import ReplayBuffer

from .config import HexoConfig, resolve_path
from .metrics import MetricLogger


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    plugin: ModelPlugin
    device: torch.device


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    checkpoint_path: Path
    metrics: dict[str, float]
    status: str


def select_device(config: HexoConfig) -> torch.device:
    requested = config.inference.device
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_model(config: HexoConfig) -> LoadedModel:
    plugin = load_model_plugin(config.model.package)
    model = plugin.build_model(config.game.__dict__, config.model.__dict__)
    device = select_device(config)
    model.to(device)
    checkpoint_path = resolve_path(config, config.checkpointing.latest_path)
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
    return LoadedModel(model=model, plugin=plugin, device=device)


def make_optimizer(model: torch.nn.Module, config: HexoConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_checkpoint(
    model: torch.nn.Module,
    config: HexoConfig,
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = resolve_path(config, config.checkpointing.latest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "extra": extra or {},
        },
        path,
    )
    return path


def train_one_cycle(config: HexoConfig) -> TrainingResult:
    replay_path = resolve_path(config, config.paths.replay_latest)
    checkpoint_path = resolve_path(config, config.checkpointing.latest_path)
    if not replay_path.exists() or replay_path.stat().st_size == 0:
        return TrainingResult(
            steps=0,
            checkpoint_path=checkpoint_path,
            metrics={},
            status=f"skipped: replay file is empty or missing at {replay_path}",
        )

    loaded = load_model(config)
    optimizer = make_optimizer(loaded.model, config)
    replay = ReplayBuffer.from_path(
        replay_path,
        limit=config.training.replay_window_samples,
    )
    logger = MetricLogger(resolve_path(config, config.paths.metrics_log))
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp and loaded.device.type == "cuda")
    loaded.model.train()

    for step in range(config.training.steps_per_cycle):
        batch = replay.sample_batch(config.training.batch_size)
        batch = loaded.plugin.augment_batch(batch)
        batch = _move_batch(dict(batch), loaded.device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=loaded.device.type,
            dtype=torch.float16,
            enabled=config.training.amp and loaded.device.type == "cuda",
        ):
            outputs = loaded.plugin.forward_inference(loaded.model, batch)
            loss = loaded.plugin.loss(outputs, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(loaded.model.parameters(), config.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        logger.log("train.loss", float(loss.detach().cpu().item()), step=step)

    checkpoint = save_checkpoint(
        loaded.model,
        config,
        step=config.training.steps_per_cycle,
        extra={"metrics": logger.averages()},
    )
    return TrainingResult(
        steps=config.training.steps_per_cycle,
        checkpoint_path=checkpoint,
        metrics=logger.averages(),
        status="completed",
    )
