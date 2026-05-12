"""Shared search utilities.

This package is intentionally focused on MCTS. Runner modes and model players
can use it for search mechanics while keeping policy interpretation inside the
model package and legality inside the engine.
"""

from .mcts import Evaluator, MCTSConfig, MCTSSearcher, SearchRequest, SearchResult

__all__ = [
    "Evaluator",
    "MCTSConfig",
    "MCTSSearcher",
    "SearchRequest",
    "SearchResult",
]
