"""Thin Python import/call boundary for dense CNN Rust acceleration.

All production acceleration lives in `hexo_models._rust.dense_cnn`, registered
from `rust/src`. This module keeps the import error message readable and gives
Python code named functions for native calls.

It intentionally does not duplicate Rust MCTS scalar validation. The native
session is the actual search boundary; Python forwards values and lets PyO3/Rust
raise clear errors for invalid native search configuration.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    from hexo_models import _rust
except ImportError as exc:  # pragma: no cover - native extension is required on use.
    _rust = None
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


def capabilities() -> Mapping[str, Any]:
    """Return the Rust accelerator capability payload."""

    return _dense_cnn_module().capabilities()


def model1_batch_inputs(states: Sequence[object]) -> Mapping[str, Any]:
    """Encode live engine states through the dense-cnn-owned Rust accelerator."""

    return _dense_cnn_module().model1_batch_inputs(tuple(states))


def model1_new_mcts_session(*, max_states: int | None = None) -> object:
    """Create a native MCTS session that reuses selected subtrees across moves."""

    return _dense_cnn_module().Model1MctsSession(max_states)


def model1_mcts_session_search(
    session: object,
    game_keys: Sequence[int],
    states: Sequence[object],
    *,
    visits: int,
    c_puct: float,
    temperature: float,
    seed: int,
    evaluator: object,
    virtual_batch_size: int | None = None,
    active_root_limit: int | None = None,
    root_dirichlet_total_alpha: float | None = None,
    root_dirichlet_noise_fraction: float | None = None,
    root_policy_temperature: float | None = None,
    fpu_reduction: float | None = None,
    virtual_loss: float | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Search through a native MCTS session, preserving chosen subtrees.

    Arguments are forwarded in the PyO3 signature order expected by
    `rust/src/mcts.rs`. The session searches every legal move at each node; there
    is no progressive widening or candidate cap.
    """

    return tuple(
        session.search(
            tuple(int(item) for item in game_keys),
            tuple(states),
            visits,
            c_puct,
            temperature,
            int(seed),
            evaluator,
            virtual_batch_size,
            active_root_limit,
            root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction,
            root_policy_temperature,
            fpu_reduction,
            virtual_loss,
        )
    )


def model1_sample_from_state(
    state: object,
    *,
    game_id: str,
    turn_index: int,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build one compact sample's state-derived facts from a live engine state."""

    return _dense_cnn_module().model1_sample_from_state(
        state,
        str(game_id),
        int(turn_index),
        dict(metadata or {}),
    )


def _dense_cnn_module() -> Any:
    """Return the loaded native dense_cnn module or raise a clear error."""

    module = getattr(_rust, "dense_cnn", None) if _rust is not None else None
    if module is None:
        raise RuntimeError(f"dense_cnn Rust accelerator is unavailable: {_IMPORT_ERROR}")
    return module
