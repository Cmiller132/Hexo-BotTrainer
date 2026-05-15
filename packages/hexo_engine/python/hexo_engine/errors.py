"""Python-facing engine error types."""

from __future__ import annotations


class HexoEngineError(Exception):
    """Base class for Python errors raised by the engine package."""


class EngineUnavailableError(HexoEngineError):
    """Raised when the Rust engine binding has not been wired yet."""


class IllegalActionError(HexoEngineError):
    """Raised when an action is rejected by the Rust rules authority."""
