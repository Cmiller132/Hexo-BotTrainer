"""Reusable MCTS search boundary.

The search utility owns tree mechanics, visit accounting, and search result
shape. The engine supplies legal actions and state transitions; a model or other
evaluator supplies policy/value estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Search knobs shared by model-backed and non-model-backed players."""

    simulations: int
    exploration_weight: float = 1.4
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Inputs needed to start search from an engine state."""

    state: object
    legal_actions: Sequence[object]
    budget: object | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Action choice and diagnostics produced by MCTS."""

    action: object
    visits: Mapping[str, int]
    value: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    """Policy/value provider consumed by reusable MCTS."""

    def evaluate(self, state: object, legal_actions: Sequence[object]) -> Mapping[str, Any]:
        """Return priors and value estimates for the supplied state."""


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by shared MCTS machinery.")


class MCTSSearcher:
    """Facade for the shared MCTS implementation."""

    def __init__(self, config: MCTSConfig, evaluator: Evaluator) -> None:
        self.config = config
        self.evaluator = evaluator

    def search(self, request: SearchRequest) -> SearchResult:
        """Run MCTS from the requested engine state."""

        _not_implemented("MCTSSearcher.search")
