"""Direct match runner mode.

Match mode starts a single match and returns the result.
This is the public entry point for running one game. It should build a session,
call the shared loop, and return a compact `GameResult`.
"""

from __future__ import annotations


def run_match(config: object) -> object:
    """Run one game through session setup and the shared loop."""

    raise NotImplementedError("match mode will be built on the shared runner loop.")
