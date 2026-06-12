"""hexgnn public Python surface — a stripped-down hexgt: GNN trunk only.

hexgnn is a deliberate simplification of hexgt (Model 2/3): the same relational
typed-GNN message-passing trunk, but with the context transformer and the
short-term-value (STV) heads REMOVED. Heads are policy + 65-bin value + opponent
policy only. It is a *truly dynamic* typed heterogeneous graph model (variable-
length per-candidate policy, packed-graph batch contract) and stays D6-invariant
by construction.

It is ADDITIVE and model-owned: shared packages (`hexo_engine`, `hexo_runner`,
`hexo_train`) provide game truth and orchestration. The native accelerator is
this package's OWN Rust crate (`packages/hexgnn/rust/src`, a fork of hexgt's),
compiled into the hexo_models cdylib as `hexo_models._rust.hexgnn` via the
`#[path]` include in `packages/hexo_models/rust/src/lib.rs` (the original
"reuses _rust.hexgt read-only" plan was superseded). See this package's
README.md.

STATUS (2026-06-12): PARKED lineage — HANDOFF.md marks hexgnn "explored and set
aside (not the active path)"; the active line is packages/dense_cnn_restnet.
Kept buildable: its Rust still compiles into every hexo_models native build, the
driver scripts (scripts/_rl_train_hexgnn.py, _pretrain_hexgnn.py) and the
tests/test_hexgnn_*.py suite still import it.

Entry points: the RL driver scripts above import modules directly (PYTHONPATH);
the dormant config-CLI path goes through plugin.py (entry point "hexgnn" in the
`hexo_train.models` group, configs/hexgnn_model.toml).
"""

from .architecture import HexgnnNetwork
from .collate import collate_graphs
from .config import HexgnnConfig, parse_hexgnn_config
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
    hexgnn_loss,
    scalar_to_binned_target,
    segment_softmax_cross_entropy,
)

__all__ = [
    "D6_SIZE",
    "DEFAULT_CANDIDATE_RADIUS",
    "EDGE_ATTR_DIM",
    "GraphTensors",
    "HexgnnConfig",
    "HexgnnNetwork",
    "NODE_FEATURE_DIM",
    "NUM_EDGE_TYPES",
    "NUM_NODE_TYPES",
    "VALUE_BINS",
    "binned_value_loss",
    "build_graph_tensors",
    "collate_graphs",
    "decode_binned_value",
    "hexgnn_loss",
    "parse_hexgnn_config",
    "scalar_to_binned_target",
    "segment_softmax_cross_entropy",
    "transform_coord",
]
