"""dense_cnn_restnet public Python surface -- the ACTIVE ResTNet model lineage.

This package is a pure-Python fork of `packages/hexo_models/dense_cnn` ("Model
1") that swaps the gated-residual trunk for a faithful ResTNet (interleaved
residual + transformer, arXiv:2410.05347). It carries no Rust of its own: the
featurizer, MCTS session, and sample facts come read-only from
`hexo_models._rust.dense_cnn` via `rust_bridge.py` (see pyproject.toml). Module
names still say "Model 1" / `model1_*` because the tensor/search contracts are
identical to the parent lineage.

The package is intentionally model-owned. Shared packages such as
`hexo_engine`, `hexo_runner`, and `hexo_train` provide game truth, game-loop
contracts, and orchestration, but they do not know how Model 1 tensors, losses,
MCTS payloads, or replay samples are represented. `hexo_train` reaches this
package only through `plugin.py` (entry point "dense_cnn_restnet"); the
dashboard debug worker (`hexo_frontend/debug_infer.py`) imports architecture/
inference/losses/mcts/rust_bridge directly for checkpoint forensics.

Only stable user-facing building blocks are re-exported here. Lower-level
production boundaries such as `mcts`, `rust_bridge`, `samples`, and
`selfplay` stay in their modules so callers have to opt into those specific
contracts.
"""

from .architecture import (
    HexConv2d,
    PolicyHead,
    RelPosMHSA,
    ResidualBlock,
    RestnetNetwork,
    TransformerBlock,
    ValueReduction,
    parse_blocks_type,
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
    "HexConv2d",
    "INPUT_CHANNELS",
    "InferenceResult",
    "Model1Config",
    "PolicyHead",
    "RelPosMHSA",
    "ResidualBlock",
    "RestnetNetwork",
    "TransformerBlock",
    "VALUE_BINS",
    "ValueReduction",
    "binned_value_loss",
    "build_benchmark_report",
    "calibrate_dense_cnn",
    "decode_binned_value",
    "DenseCNNInference",
    "inverse_index",
    "model1_loss",
    "parse_blocks_type",
    "parse_model1_config",
    "scalar_to_binned_target",
    "soft_cross_entropy",
    "transform_action_id",
    "transform_coord",
]
