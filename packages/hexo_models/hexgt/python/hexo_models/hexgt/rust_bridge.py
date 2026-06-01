"""Thin Python import/call boundary for hexgt Rust acceleration.

All native acceleration lives in `hexo_models._rust.hexgt`, registered from
`hexgt/rust/src`. This module keeps the import error message readable and gives
Python code named functions for native calls. Phase 0 exposes only
`capabilities()`; the candidate/window/graph builders (Phase 1) and the MCTS
session search (Phase 5) are added to this boundary as they land.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    from hexo_models import _rust
except ImportError as exc:  # pragma: no cover - native extension is required on use.
    _rust = None
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


def capabilities() -> Mapping[str, Any]:
    """Return the Rust accelerator capability payload."""

    return _hexgt_module().capabilities()


def _hexgt_module() -> Any:
    """Return the loaded native hexgt module or raise a clear error."""

    module = getattr(_rust, "hexgt", None) if _rust is not None else None
    if module is None:
        raise RuntimeError(f"hexgt Rust accelerator is unavailable: {_IMPORT_ERROR}")
    return module
