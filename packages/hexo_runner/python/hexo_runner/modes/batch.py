"""Batch runner mode.

Batch mode is the public entry point for running many games. It should schedule
independent match configs, run them in parallel when configured to do so, and
aggregate their `GameResult` values without duplicating the single-game loop.
"""

from __future__ import annotations


def run_batch(config: object) -> object:
    """Run many games by scheduling calls to match mode."""

    raise NotImplementedError("batch mode will be built on the shared runner loop.")
