"""Runner config re-exports."""

from __future__ import annotations

from .session import BatchSpec, GameSpec


RunnerConfig = BatchSpec

__all__ = ["BatchSpec", "GameSpec", "RunnerConfig"]
