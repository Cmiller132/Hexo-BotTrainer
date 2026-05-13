"""Pseudo-code CLI boundary for a future Python runner.

Desired shape:

1. Load a rigorously typed experiment config.
2. Build configured runner players.
3. Run match, batch, evaluation, or self-play modes through the shared loop.
4. Write detached core game records for analysis and audit.

This module is intentionally non-operational while the Python layer is being
redesigned.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "The Python runner is currently a redesign skeleton. "
        "Use the Rust engine crate as the source of rule truth."
    )
