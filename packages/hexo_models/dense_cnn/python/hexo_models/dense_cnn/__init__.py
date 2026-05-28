"""Dense CNN Model 1 public Python surface.

This package is intentionally model-owned. Shared packages such as
`hexo_engine`, `hexo_runner`, and `hexo_train` provide game truth, game-loop
contracts, and orchestration, but they do not know how Model 1 tensors, losses,
MCTS payloads, or replay samples are represented.

Only stable user-facing building blocks are re-exported here. Lower-level
production boundaries such as `mcts`, `rust_bridge`, `samples`, and
`selfplay` stay in their modules so callers have to opt into those specific
contracts.
"""

from .architecture import (
    GatedResBlock,
    HexConv2d,
    Model1Network,
    PolicyHead,
    ValueBinnedHead,
)
from .config import Model1Config, parse_model1_config
from .constants import BOARD_AREA, BOARD_SIZE, INPUT_CHANNELS, VALUE_BINS
from .d6 import D6_SIZE, D6Symmetry, inverse_index, transform_action_id, transform_coord
from .losses import (
    binned_value_loss,
    decode_binned_value,
    model1_loss,
    scalar_to_binned_target,
    soft_cross_entropy,
)
from .inference import DenseCNNInference, InferenceResult
from .performance import (
    build_benchmark_report,
    calibrate_dense_cnn,
)

__all__ = [
    "BOARD_AREA",
    "BOARD_SIZE",
    "D6Symmetry",
    "D6_SIZE",
    "GatedResBlock",
    "HexConv2d",
    "INPUT_CHANNELS",
    "InferenceResult",
    "Model1Config",
    "Model1Network",
    "PolicyHead",
    "VALUE_BINS",
    "ValueBinnedHead",
    "binned_value_loss",
    "build_benchmark_report",
    "calibrate_dense_cnn",
    "decode_binned_value",
    "DenseCNNInference",
    "inverse_index",
    "model1_loss",
    "parse_model1_config",
    "scalar_to_binned_target",
    "soft_cross_entropy",
    "transform_action_id",
    "transform_coord",
]
