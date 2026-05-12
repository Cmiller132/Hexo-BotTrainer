"""Batch runner mode.

Batch mode coordinates many independent games with concurrency limits and
aggregate summaries. It shares the normal engine application path; concurrency
policy belongs here, not in the engine or model packages.
"""

from __future__ import annotations


def run_batch(config: object) -> object:
    """Run a batch of games once scheduler and replay sinks are wired."""

    raise NotImplementedError("batch mode will be built on the shared runner loop.")
