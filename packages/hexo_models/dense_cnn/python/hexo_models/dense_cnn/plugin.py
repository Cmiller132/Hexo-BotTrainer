"""Training plugin for the standalone dense CNN model family.

`hexo_train` discovers this plugin and calls it to build model-specific
components. The plugin is the composition boundary: it wires config parsing,
network construction, optimizer setup, checkpoint IO, self-play, NPZ training,
performance calibration, NPZ replay, and evaluation into the
generic training pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from hexo_train.components import ComponentOverrides

from .architecture import Model1Network
from .checkpoints import DenseCNNCheckpointLoader, DenseCNNCheckpointSaver
from .config import parse_model1_config
from .evaluation import evaluate_epoch
from .performance import calibrate_dense_cnn
from .selfplay import generate_selfplay_epoch
from .trainer import DenseCNNTrainer


class DenseCNNPlugin:
    """Model plugin object consumed by `hexo_train.registry`."""

    name = "hexo_models.dense_cnn"

    def build_model(self, game_spec: Mapping[str, Any], config: Mapping[str, Any]) -> torch.nn.Module:
        """Build the PyTorch network from model-specific config."""

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
        """Create dense_cnn-owned components for the generic training loop."""

        _ = (defaults, shared)
        if model is None:
            raise ValueError("DenseCNNPlugin requires build_model() to run first")
        parsed = parse_model1_config(config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parsed.training.learning_rate,
            weight_decay=parsed.training.weight_decay,
        )
        trainer = DenseCNNTrainer(
            model=model,
            config=parsed,
            optimizer=optimizer,
        )
        return ComponentOverrides(
            trainer=trainer,
            optimizer=optimizer,
            checkpoint_loader=DenseCNNCheckpointLoader(),
            checkpoint_saver=DenseCNNCheckpointSaver(),
            uses_shared_sample_store=False,
            extra={
                "model_family": "dense_cnn",
                "train_samples_per_epoch": parsed.training.train_samples_per_epoch,
                "shuffle_min_rows": parsed.samples.shuffle_min_rows,
                "selfplay_mode": "game_driven_all_mcts",
                "selfplay_active_games": parsed.selfplay.active_games,
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

    def calibrate_performance(self, *, ctx: Any, components: Any) -> dict[str, Any]:
        """Run calibration and copy selected settings onto the trainer."""

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

plugin = DenseCNNPlugin()


def get_plugin() -> DenseCNNPlugin:
    return plugin
