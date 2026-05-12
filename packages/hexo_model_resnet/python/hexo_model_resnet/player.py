"""Runner player adapter for the ResNet model family.

This adapter will receive runner decision requests, build ResNet inputs from
engine state, run inference and optional search, then return one legal action to
the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn


@dataclass(slots=True)
class ResNetPlayer:
    """Model-backed participant for `hexo_runner`."""

    player_id: str
    inference: object
    search: object | None = None

    def initialize(self, session_context: object) -> None:
        """Prepare model resources for a runner session."""

    def decide(self, request: object) -> object:
        """Choose a legal action from a runner decision request."""

        _not_implemented("ResNetPlayer.decide")

    def observe_transition(self, transition: object) -> None:
        """Observe accepted engine transitions for diagnostics or stateful search."""

    def close(self, final_summary: object) -> None:
        """Release any model or search resources."""


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by the ResNet player adapter.")
