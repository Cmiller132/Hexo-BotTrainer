"""Training plugin for the standalone dense CNN model family."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch

from hexo_train.components import ComponentOverrides

from .architecture import Model1Network
from .checkpoints import DenseCNNCheckpointLoader, DenseCNNCheckpointSaver
from .config import parse_model1_config
from .evaluation import evaluate_epoch
from .performance import calibrate_dense_cnn
from .samples import SampleBuffer
from .samples_finalizer import DenseCNNSampleFinalizer
from .selfplay import generate_selfplay_epoch
from .trainer import DenseCNNTrainer


class DenseCNNPlugin:
    name = "hexo_models.dense_cnn"

    def build_model(self, game_spec: Mapping[str, Any], config: Mapping[str, Any]) -> torch.nn.Module:
        _ = game_spec
        parsed = parse_model1_config(config)
        arch = parsed.architecture
        return Model1Network(
            in_channels=arch.input_channels,
            channels=arch.channels,
            blocks=arch.residual_blocks,
            dropout=arch.dropout,
            lookahead_horizons=arch.lookahead_horizons,
        )

    def training_component_overrides(
        self,
        *,
        defaults: Any,
        config: Mapping[str, Any],
        shared: Any,
        model: torch.nn.Module | None,
    ) -> ComponentOverrides:
        _ = (defaults, shared)
        if model is None:
            raise ValueError("DenseCNNPlugin requires build_model() to run first")
        parsed = parse_model1_config(config)
        buffer = SampleBuffer(
            capacity=parsed.samples.capacity,
            recency_halflife=parsed.samples.recency_halflife,
            compression_level=parsed.samples.compression_level,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parsed.training.learning_rate,
            weight_decay=parsed.training.weight_decay,
        )
        trainer = DenseCNNTrainer(
            model=model,
            config=parsed,
            buffer=buffer,
            optimizer=optimizer,
        )
        return ComponentOverrides(
            sample_finalizer=DenseCNNSampleFinalizer(buffer),
            symmetry_selector=DenseCNNRandomExpansionSymmetrySelector(),
            trainer=trainer,
            optimizer=optimizer,
            checkpoint_loader=DenseCNNCheckpointLoader(),
            checkpoint_saver=DenseCNNCheckpointSaver(),
            extra={
                "model_family": "dense_cnn",
                "sample_capacity": parsed.samples.capacity,
                "train_sample_count": parsed.samples.train_sample_count,
                "selfplay_samples_per_epoch": parsed.selfplay.samples_per_epoch,
                "evaluation_games_per_epoch": parsed.evaluation.games_per_epoch,
            },
        )

    def generate_selfplay(self, *, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
        return generate_selfplay_epoch(
            ctx=ctx,
            components=components,
            epoch=epoch,
            games_per_epoch=games_per_epoch,
        )

    def evaluate_epoch(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        return evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    def cleanup_after_epoch(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        _ = ctx
        trainer = components.model.trainer
        if hasattr(trainer, "release_epoch_resources"):
            return trainer.release_epoch_resources(epoch=epoch)
        return {
            "status": "skipped",
            "epoch": epoch,
            "reason": "dense_cnn trainer has no release_epoch_resources hook",
        }

    def calibrate_performance(self, *, ctx: Any, components: Any) -> dict[str, Any]:
        trainer = components.model.trainer
        result = calibrate_dense_cnn(
            model=components.model.model,
            config=trainer.config,
            optimizer=components.model.optimizer,
            ctx=ctx,
        )
        if result.get("status") == "completed":
            trainer.inference_batch_size = int(result["selected_inference_batch_size"])
            trainer.selfplay_batch_size = int(result.get("selected_selfplay_batch_size", trainer.selfplay_batch_size))
            selected_virtual = int(result.get("selected_mcts_virtual_batch_size", 0))
            trainer.mcts_virtual_batch_size = selected_virtual if selected_virtual > 0 else None
            trainer.training_batch_size = int(result["selected_training_batch_size"])
        return result


class DenseCNNRandomExpansionSymmetrySelector:
    """Declare that dense_cnn chooses D6 randomly when compact samples expand."""

    def select_for_window(self, sample_window: object, *, seed: int | None, epoch: int = 0) -> object:
        count = int(getattr(sample_window, "window_size", 0) or 0)
        return SimpleNamespace(
            symmetries=(),
            seed=int(seed or 0),
            epoch=int(epoch),
            metadata={
                "sample_count": count,
                "mode": "random_per_training_expansion",
                "d6_group_size": 12,
                "note": "dense_cnn trainer samples a fresh random D6 transform when each compact sample is expanded",
            },
        )


plugin = DenseCNNPlugin()


def get_plugin() -> DenseCNNPlugin:
    return plugin
