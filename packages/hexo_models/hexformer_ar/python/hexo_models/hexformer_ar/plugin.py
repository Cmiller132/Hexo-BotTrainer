"""Training plugin for Hexformer AR."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from hexo_train.components import ComponentOverrides

from .architecture import HexformerAR
from .checkpoints import HexformerCheckpointLoader, HexformerCheckpointSaver
from .config import parse_hexformer_config
from .performance import calibrate_hexformer
from .samples import HexformerReplayBuffer
from .samples_finalizer import HexformerSampleFinalizer
from .trainer import HexformerTrainer


class HexformerARPlugin:
    name = "hexo_models.hexformer_ar"

    def build_model(self, game_spec: Mapping[str, Any], config: Mapping[str, Any]) -> torch.nn.Module:
        _ = game_spec
        parsed = parse_hexformer_config(config)
        return HexformerAR(parsed.architecture)

    def training_component_overrides(
        self,
        *,
        defaults: Any,
        config: Mapping[str, Any],
        shared: Any,
        model: torch.nn.Module | None,
    ) -> ComponentOverrides:
        _ = defaults, shared
        if model is None:
            raise ValueError("HexformerARPlugin requires build_model() to run first")
        parsed = parse_hexformer_config(config)
        buffer = HexformerReplayBuffer(
            capacity=max(1, parsed.samples.train_sample_count * 50),
            recency_halflife=parsed.samples.recency_halflife,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parsed.training.learning_rate,
            weight_decay=parsed.training.weight_decay,
        )
        trainer = HexformerTrainer(model=model, config=parsed, buffer=buffer, optimizer=optimizer)
        return ComponentOverrides(
            sample_finalizer=HexformerSampleFinalizer(),
            trainer=trainer,
            optimizer=optimizer,
            checkpoint_loader=HexformerCheckpointLoader(),
            checkpoint_saver=HexformerCheckpointSaver(),
            extra={
                "model_family": "hexformer_ar",
                "sample_capacity": buffer.capacity,
                "train_sample_count": parsed.samples.train_sample_count,
                "selfplay_samples_per_epoch": parsed.selfplay.samples_per_epoch,
                "evaluation_games_per_epoch": parsed.evaluation.games_per_epoch,
            },
        )

    def generate_selfplay(self, *, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
        from .selfplay import generate_selfplay_epoch

        return generate_selfplay_epoch(ctx=ctx, components=components, epoch=epoch, games_per_epoch=games_per_epoch)

    def evaluate_epoch(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        from .evaluation import evaluate_epoch

        return evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    def calibrate_performance(self, *, ctx: Any, components: Any) -> dict[str, Any]:
        result = calibrate_hexformer(ctx=ctx, components=components)
        if result.get("status") == "completed":
            trainer = components.model.trainer
            trainer.training_batch_size = int(result.get("selected_training_batch_size", trainer.training_batch_size))
        return result


plugin = HexformerARPlugin()


def get_plugin() -> HexformerARPlugin:
    return plugin
