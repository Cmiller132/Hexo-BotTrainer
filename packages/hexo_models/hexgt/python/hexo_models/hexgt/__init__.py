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

STATUS (2026-06-12): the hexgt training lineage is HALTED — run `hexgt_rl_main3`
was permanently stopped by the owner at epoch 40 (2026-06-05, see HANDOFF.md);
the active lineage is `packages/dense_cnn_restnet`. The code remains live
infrastructure, NOT dead: the parked `packages/hexgnn` package is a fork that
compiles into the same `hexo_models._rust` native module, the dashboard debug
worker (`packages/hexo_frontend/python/hexo_frontend/debug_infer.py`) loads
hexgt checkpoints through this package, `dense_cnn_restnet` copied its PCR
design, and ~30 `tests/test_hexgt_*.py` files gate the Rust/Python contracts.

NOTE on doc citations: the design documents referenced throughout this package
(HEXFORMER_REWRITE_PLAN.md, HEXGT_DECISIONS.md, HEXGT_PMA_VALUE_HEAD_PLAN.md,
HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md, HEXGT_PCR_KATAGO_MAPPING.md and the section
numbers like §4/§6.3) were removed from the working tree by commit b50e92a
("chore: wipe all documentation files"); they survive only in git history. The
citations are kept because they identify WHICH decision a knob implements.
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
