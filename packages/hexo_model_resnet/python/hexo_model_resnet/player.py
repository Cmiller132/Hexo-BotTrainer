"""Runner player adapter for the ResNet model family.

This adapter will receive a cloned runner engine state, build ResNet inputs
from public `hexo_engine` queries, run inference and optional search, then
return one legal action to the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from hexo_runner.player import PlayerIdentity


@dataclass(slots=True)
class ResNetPlayer:
    """Model-backed participant for `hexo_runner`."""

    identity: PlayerIdentity
    inference: object
    search: object | None = None

    def setup_worker(self, context: object) -> None:
        """Prepare long-lived model resources for a runner worker."""

    def start_game(self, context: object) -> None:
        """Reset per-game model/search state."""

    def decide(self, state: object) -> object:
        """Choose a legal action from a cloned engine state."""

        _not_implemented("ResNetPlayer.decide")

    def observe_transition(self, transition: object) -> None:
        """Observe accepted engine transitions for diagnostics or stateful search."""

    def finish_game(self, final_summary: object) -> None:
        """Observe the final runner result for this game."""

    def close(self) -> None:
        """Release any long-lived model or search resources."""


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by the ResNet player adapter.")
