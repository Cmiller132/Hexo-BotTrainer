"""Training plugin for the hexgt (Model 2) dynamic-graph model family.

`hexo_train` discovers this plugin (entry point `hexgt` in the
`hexo_train.models` group) and calls it to build model-specific components. The
plugin is the composition boundary, mirroring `dense_cnn/plugin.py`.

Phase 0 wires the minimum the pipeline can start from: `name`, `build_model`,
and `training_component_overrides` returning the checkpoint loader/saver and an
optimizer. The self-play / evaluation / calibration / trainer hooks land in the
GPU phases (5+) and Phase 4 (trainer); they are intentionally absent here so the
package builds and resolves before those exist.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from hexo_train.components import ComponentOverrides

from .architecture import HexgtNetwork
from .checkpoints import HexgtCheckpointLoader, HexgtCheckpointSaver
from .config import parse_hexgt_config


class HexgtPlugin:
    """Model plugin object consumed by `hexo_train.registry`."""

    name = "hexo_models.hexgt"

    def build_model(self, game_spec: Mapping[str, Any], config: Mapping[str, Any]) -> torch.nn.Module:
        """Build the dynamic GNN + transformer network from model-specific config."""

        _ = game_spec
        parsed = parse_hexgt_config(config)
        arch = parsed.architecture
        return HexgtNetwork(
            node_feature_dim=arch.node_feature_dim,
            token_dim=arch.token_dim,
            gnn_layers=arch.gnn_layers,
            ctx_layers=arch.ctx_layers,
            attention_heads=arch.attention_heads,
            ffn_dim=arch.ffn_dim,
            dropout=arch.dropout,
            short_term_value_horizons=arch.short_term_value_horizons,
        )

    def training_component_overrides(
        self,
        *,
        defaults: Any,
        config: Mapping[str, Any],
        shared: Any,
        model: torch.nn.Module | None,
    ) -> ComponentOverrides:
        """Create hexgt-owned components for the generic training loop."""

        _ = (defaults, shared)
        if model is None:
            raise ValueError("HexgtPlugin requires build_model() to run first")
        parsed = parse_hexgt_config(config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parsed.training.learning_rate,
            weight_decay=parsed.training.weight_decay,
        )
        # The trainer is added in Phase 4 (HexgtTrainer); until then the pipeline
        # can still load/save checkpoints and hold the optimizer.
        return ComponentOverrides(
            trainer=None,
            optimizer=optimizer,
            checkpoint_loader=HexgtCheckpointLoader(),
            checkpoint_saver=HexgtCheckpointSaver(),
            uses_shared_sample_store=False,
            extra={
                "model_family": "hexgt",
                "train_samples_per_epoch": parsed.training.train_samples_per_epoch,
                "shuffle_min_rows": parsed.samples.shuffle_min_rows,
                "selfplay_mode": "game_driven_all_mcts",
                "selfplay_active_games": parsed.selfplay.active_games,
                "evaluation_games_per_epoch": parsed.evaluation.games_per_epoch,
                "candidate_radius": parsed.architecture.candidate_radius,
            },
        )


plugin = HexgtPlugin()


def get_plugin() -> HexgtPlugin:
    return plugin
