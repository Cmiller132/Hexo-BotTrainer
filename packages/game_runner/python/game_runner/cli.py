"""Pseudo-code CLI boundary for a future Python runner.

Desired shape:

1. Load a rigorously typed experiment config.
2. Load or initialize the requested model family.
3. Create an evaluator object with one explicit inference contract.
4. Ask Rust to generate self-play data.
5. Validate the Rust manifest and replay files.
6. Train from validated replay.
7. Write checkpoints, metrics, and cycle metadata.

This module is intentionally non-operational while the Python layer is being
redesigned.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "The Python runner has been stripped to pseudo-code for redesign. "
        "Use the Rust crates as the source of truth."
    )
