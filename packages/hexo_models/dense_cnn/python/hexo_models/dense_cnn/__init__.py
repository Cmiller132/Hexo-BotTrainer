"""Dense CNN Model 1 implementation.

This subpackage owns its architecture, sample encoding, losses, training
helpers, and plugin wiring so future model families can live beside it without
sharing implementation modules.
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
    benchmark_report,
    build_benchmark_report,
    calibrate_dense_cnn,
    calibrate_dense_cnn_performance,
    calibrate_model1_performance,
    calibrate_performance,
    report_performance_benchmark,
)
from .samples import encode_compact_sample

HexoModel = Model1Network
HexoModelConfig = Model1Config
decode_value = decode_binned_value
binned_soft_cross_entropy = binned_value_loss

__all__ = [
    "BOARD_AREA",
    "BOARD_SIZE",
    "D6Symmetry",
    "D6_SIZE",
    "GatedResBlock",
    "HexConv2d",
    "HexoModel",
    "HexoModelConfig",
    "INPUT_CHANNELS",
    "InferenceResult",
    "Model1Config",
    "Model1Network",
    "PolicyHead",
    "VALUE_BINS",
    "ValueBinnedHead",
    "binned_soft_cross_entropy",
    "binned_value_loss",
    "benchmark_report",
    "build_benchmark_report",
    "calibrate_dense_cnn",
    "calibrate_dense_cnn_performance",
    "calibrate_model1_performance",
    "calibrate_performance",
    "decode_binned_value",
    "decode_value",
    "DenseCNNInference",
    "encode_compact_sample",
    "inverse_index",
    "model1_loss",
    "parse_model1_config",
    "report_performance_benchmark",
    "scalar_to_binned_target",
    "soft_cross_entropy",
    "transform_action_id",
    "transform_coord",
]
