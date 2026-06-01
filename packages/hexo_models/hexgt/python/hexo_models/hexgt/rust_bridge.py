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


def candidate_ids(state: object, n: int) -> list[int]:
    """Packed candidate action ids for a live engine state at radius `n` (§4)."""

    return list(_hexgt_module().hexgt_candidate_ids(state, int(n)))


def graph_facts(state: object, n: int) -> Mapping[str, Any]:
    """Full bounded-graph facts for a live engine state at radius `n`.

    Returns candidate ids/order, count-3/4/5 window tokens, typed node/edge
    arrays, and per-type counts (the §4.6 no-explosion gate). This is the shared
    construction reused by the Phase-4 expand step and (later) live MCTS.
    """

    return _hexgt_module().hexgt_graph_facts(state, int(n))


def _hexgt_module() -> Any:
    """Return the loaded native hexgt module or raise a clear error."""

    module = getattr(_rust, "hexgt", None) if _rust is not None else None
    if module is None:
        raise RuntimeError(f"hexgt Rust accelerator is unavailable: {_IMPORT_ERROR}")
    return module
