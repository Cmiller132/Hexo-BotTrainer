"""hexo_strix — a dependency-light port of the SootyOwl/hexo-strix GNN bot.

Runs a hexo-strix HeXONet checkpoint as an evaluation opponent inside this
repo, using this repo's game engine and runner. No torch_geometric and no
hexo-strix Rust extension required: the GINE trunk and the "axis" graph builder
are reimplemented in pure PyTorch / Python.

Typical use (see scripts/_strix_h2h.py):

    from hexo_strix import make_strix_factory
    factory = make_strix_factory("checkpoint_00237000.pt")

The factory plugs straight into ``run_head_to_head`` from any model package's
evaluation module.
"""

from .batched_infer import (
    RoundBatch,
    batched_eval,
    batched_eval_round,
    build_axis_graph_tensors,
    build_axis_round,
)
from .graph import AxisGraph, build_axis_graph, node_threat_features
from .loader import StrixCheckpoint, load_strix_checkpoint
from .model import HeXONet, build_from_model_config

# NOTE: player.py imports hexo_runner. It is imported LAZILY (via __getattr__)
# so that the model + graph builder can be used (and unit-tested) in an
# environment where hexo_runner is not on sys.path — the RunnerPlayer symbols
# resolve only when actually requested.


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    # player.py needs hexo_runner; mcts_player.py additionally needs the
    # compiled hexo_rs extension. Import lazily so the model + graph builder
    # stay usable where those deps are absent.
    if name in ("StrixPlayer", "make_strix_factory"):
        from . import player

        return getattr(player, name)
    if name in ("StrixMctsPlayer", "make_strix_mcts_factory"):
        from . import mcts_player

        return getattr(mcts_player, name)
    if name == "StrixBatchServer":
        from . import batch_server

        return batch_server.StrixBatchServer
    if name == "run_threaded_matches":
        from . import match_runner

        return match_runner.run_threaded_matches
    if name == "play_strix_match":
        # The lock-step GPU-batched hexfield-vs-strix arena physically lives in
        # the hexfield package (it reuses that package's result-dict builder,
        # .hxr writer, and Rust session helpers). Re-exported here lazily for
        # discoverability; importing it pulls in hexfield + torch, so it stays
        # behind __getattr__ like the other heavy symbols.
        from hexfield.eval_arena import play_strix_match

        return play_strix_match
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AxisGraph",
    "build_axis_graph",
    "node_threat_features",
    "StrixCheckpoint",
    "load_strix_checkpoint",
    "HeXONet",
    "build_from_model_config",
    "batched_eval",
    "build_axis_graph_tensors",
    "RoundBatch",
    "batched_eval_round",
    "build_axis_round",
    "StrixPlayer",
    "make_strix_factory",
    "StrixMctsPlayer",
    "make_strix_mcts_factory",
    "StrixBatchServer",
    "run_threaded_matches",
    "play_strix_match",
]
