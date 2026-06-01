"""hexgt (Model 2) public Python surface — dynamic GNN + transformer.

Like dense_cnn, this package is model-owned: shared packages (`hexo_engine`,
`hexo_runner`, `hexo_train`) provide game truth and orchestration but do not know
how hexgt's dynamic graph, losses, MCTS payloads, or replay samples are
represented.

Model 2 is a *truly dynamic* typed heterogeneous GNN → transformer hybrid: a
per-candidate (variable-length) policy, a 65-bin value head, and a packed-graph
batch contract. It slots into the existing training / MCTS / replay / eval
pipeline (drop-in PIPELINE compatibility, not tensor-shape matching). See
`docs/analysis/HEXFORMER_REWRITE_PLAN.md` and `HEXGT_DECISIONS.md`.
"""

from .architecture import HexgtNetwork
from .collate import collate_graphs
from .config import HexgtConfig, parse_hexgt_config
from .constants import (
    DEFAULT_CANDIDATE_RADIUS,
    EDGE_ATTR_DIM,
    NODE_FEATURE_DIM,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    VALUE_BINS,
)
from .d6 import D6_SIZE, transform_coord
from .features import GraphTensors, build_graph_tensors
from .losses import (
    binned_value_loss,
    decode_binned_value,
    hexgt_loss,
    scalar_to_binned_target,
    segment_softmax_cross_entropy,
)

__all__ = [
    "D6_SIZE",
    "DEFAULT_CANDIDATE_RADIUS",
    "EDGE_ATTR_DIM",
    "GraphTensors",
    "HexgtConfig",
    "HexgtNetwork",
    "NODE_FEATURE_DIM",
    "NUM_EDGE_TYPES",
    "NUM_NODE_TYPES",
    "VALUE_BINS",
    "binned_value_loss",
    "build_graph_tensors",
    "collate_graphs",
    "decode_binned_value",
    "hexgt_loss",
    "parse_hexgt_config",
    "scalar_to_binned_target",
    "segment_softmax_cross_entropy",
    "transform_coord",
]
