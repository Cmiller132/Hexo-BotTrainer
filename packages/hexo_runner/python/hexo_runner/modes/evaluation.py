"""Evaluation runner mode.

Evaluation mode runs fixed comparisons across seeds, scenarios, checkpoints, or
opponent pools. It should aggregate runner results without knowing model tensor
layouts or training targets.
"""

from __future__ import annotations


def run_evaluation(config: object) -> object:
    """Run an evaluation suite once batch and player loading are wired."""

    raise NotImplementedError("evaluation mode will be built on runner sessions.")
