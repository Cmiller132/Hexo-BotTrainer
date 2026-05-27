"""Hexformer autoregressive sparse policy-value model family.

The package intentionally owns the sparse tensor contract, candidate frontier,
window-token features, losses, trainer, inference, and plugin wiring.  Shared
training code should see it only through the normal Hexo model plugin contract.
"""

from .architecture import HexformerAR, HexformerOutputs
from .augmentation import transform_sparse_input
from .benchmarks import benchmark_plan
from .config import HexformerConfig, parse_hexformer_config
from .curriculum import generate_tactical_pretraining_records
from .input import SparseDecisionInput, build_sparse_input, collate_sparse_inputs
from .losses import hexformer_loss
from .samples import (
    compressed_sample_from_training_record,
    training_record_from_sample,
    training_record_from_sparse_input,
)

__all__ = [
    "HexformerAR",
    "HexformerConfig",
    "HexformerOutputs",
    "SparseDecisionInput",
    "build_sparse_input",
    "benchmark_plan",
    "collate_sparse_inputs",
    "compressed_sample_from_training_record",
    "generate_tactical_pretraining_records",
    "hexformer_loss",
    "parse_hexformer_config",
    "training_record_from_sample",
    "training_record_from_sparse_input",
    "transform_sparse_input",
]
