"""Runner config re-exports.

Compatibility shim only: re-exports the spec dataclasses from
`hexo_runner/session.py` under an older naming. Nothing in the repo imports
this module (see UNUSED marker below).
"""

from __future__ import annotations

from .session import BatchSpec, GameSpec


# UNUSED(2026-06-12): no importers of hexo_runner.config or RunnerConfig found
# in packages/tests/scripts (excl. scripts/archive); callers use
# session.BatchSpec/GameSpec directly. Kept as a compatibility shim.
RunnerConfig = BatchSpec

__all__ = ["BatchSpec", "GameSpec", "RunnerConfig"]
