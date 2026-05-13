"""Pseudo-code config boundary for the future Python runner.

The redesign should make config ownership explicit:

1. Experiment config: cycle counts, paths, checkpoint policy, logging.
2. Game config: engine-owned rules, scenario, and seed options.
3. Player config: already-built or configured runner participants.
4. Record config: detached game record destinations and metadata.

The final implementation should reject unknown or unsupported fields before a
self-play cycle starts.
"""

from __future__ import annotations


class RunnerConfig:
    """Design placeholder, not a production config object."""

    pass
