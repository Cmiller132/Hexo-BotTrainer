"""Evaluation runner mode.

Evaluation mode builds a fixed series of matches between configured opponents.
It should call batch mode under the hood, mark sessions as evaluation runs so
players can reduce exploration/noise, and own evaluation-specific analysis,
score keeping, and future strength estimates.
"""

from __future__ import annotations


def run_evaluation(config: object) -> object:
    """Run an evaluation suite by preparing eval match configs for batch mode."""

    raise NotImplementedError("evaluation mode will be built on runner sessions.")
