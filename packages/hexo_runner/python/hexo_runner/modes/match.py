"""Direct match runner mode.

Match mode runs one game or a small explicit series between configured players.
It uses the same player contract and engine transition path as every other
mode.
"""

from __future__ import annotations


def run_match(config: object) -> object:
    """Run a direct match once session, loop, and engine wiring exist."""

    raise NotImplementedError("match mode will be built on the shared runner loop.")
