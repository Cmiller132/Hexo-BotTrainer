"""Pseudo-code CLI boundary for a future Python runner.

Desired shape:

1. Load a rigorously typed experiment config.
2. Load or initialize the requested model family.
3. Create model-backed players through one explicit `InferenceAdapter`.
4. Run match, batch, evaluation, or self-play modes through the shared loop.
5. Write core game records and training replay references.
6. Hand replay to model-owned training code.
7. Write checkpoints, metrics, and cycle metadata.

This module is intentionally non-operational while the Python layer is being
redesigned.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "The Python runner is currently a redesign skeleton. "
        "Use the Rust engine crate as the source of rule truth."
    )
