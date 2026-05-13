"""Pseudo-code bridge boundary for future Python access to Rust.

The redesigned Python layer should expose only narrow, typed calls into the
Rust extension. It should not duplicate game logic, sample semantics, or MCTS.
"""

from __future__ import annotations


def import_rust_module() -> object:
    """Design placeholder for loading the compiled PyO3 extension."""

    raise NotImplementedError("Rust extension loading will be redesigned")
