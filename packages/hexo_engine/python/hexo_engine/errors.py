"""Python-facing engine error types.

The Rust engine owns the authoritative error details for illegal moves,
snapshot loading, and incompatible state data. These Python exceptions are the
host-side boundary that API wrappers should raise after translating Rust errors.
"""

from __future__ import annotations


class HexoEngineError(Exception):
    """Base class for Python errors raised by the engine package."""


class EngineUnavailableError(HexoEngineError):
    """Raised when the Rust engine binding has not been wired yet."""


class IllegalActionError(HexoEngineError):
    """Raised when an action is rejected by the Rust rules authority."""


class SnapshotError(HexoEngineError):
    """Raised when a snapshot cannot be serialized or loaded."""


class IncompatibleSnapshotError(SnapshotError):
    """Raised when snapshot data was produced by an incompatible engine version."""
