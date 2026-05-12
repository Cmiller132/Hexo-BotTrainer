"""Pseudo-code config boundary for the future Python runner.

The redesign should make config ownership explicit:

1. Experiment config: cycle counts, paths, checkpoint policy, logging.
2. Game config: Rust-owned game limits and encoding options.
3. Search config: Rust-owned MCTS/search options.
4. Model config: Python-owned model package, device, precision, batching.
5. Training config: Python-owned optimizer and replay consumption options.

The final implementation should reject unknown or unsupported fields before a
self-play cycle starts.
"""

from __future__ import annotations


class RunnerConfig:
    """Design placeholder, not a production config object."""

    pass
