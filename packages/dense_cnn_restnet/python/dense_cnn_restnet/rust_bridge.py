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
    widening_policy_mass: float | None = None,
    widening_max_children: int | None = None,
    widening_min_children: int | None = None,
    forced_playout_k: float | None = None,
    move_temperatures: Sequence[float] | None = None,
    root_policy_temperatures: Sequence[float] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Search through a native MCTS session, preserving chosen subtrees.

    Arguments are forwarded in the PyO3 signature order expected by
    `rust/src/mcts.rs`. Each node materializes at most a policy-nucleus subset of
    its legal moves (top-p widening).
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
            widening_policy_mass,
            widening_max_children,
            widening_min_children,
            forced_playout_k,
            None if move_temperatures is None else [float(t) for t in move_temperatures],
            None
            if root_policy_temperatures is None
            else [float(t) for t in root_policy_temperatures],
        )
    )


def model1_mcts_session_run_continuous(
    session: object,
    game_keys: Sequence[int],
    states: Sequence[object],
    *,
    evaluator: object,
    on_move: object,
    visits: int,
    c_puct: float,
    base_seed: int,
    virtual_batch_size: int,
    flush_target: int,
    active_root_limit: int,
    temperature_by_ply: Sequence[float],
    root_dirichlet_total_alpha: float | None = None,
    root_dirichlet_noise_fraction: float | None = None,
    root_policy_temperature: float | None = None,
    fpu_reduction: float | None = None,
    virtual_loss: float | None = None,
    widening_policy_mass: float | None = None,
    widening_max_children: int | None = None,
    widening_min_children: int | None = None,
    forced_playout_k: float | None = None,
    root_policy_temperature_early: float | None = None,
    root_policy_temperature_halflife: float | None = None,
    pcr_full_proportion: float | None = None,
    pcr_fast_visits: int | None = None,
    policy_init_fraction: float | None = None,
    policy_init_avg_plies: float | None = None,
    policy_init_max_plies: int | None = None,
    policy_init_temperature: float | None = None,
) -> Mapping[str, Any]:
    """Run the native continuous per-slot scheduler to epoch completion.

    Restnet-only entry point into `Model1MctsSession.run_continuous`
    (rust/src/mcts.rs in hexo_models/dense_cnn): one call drives EVERY game to
    its end, invoking the Python `on_move(game_key, payload)` callback per
    decided move; the callback returns ("advance", state), ("replace", key,
    state), or None to retire the slot. PCR coins, policy-init draws, and the
    root-temperature ramp are resolved natively per slot. Arguments are
    forwarded in the PyO3 signature order. Returns the scheduler's epoch-wide
    diagnostics dict (flush counts, move-class tallies, mcts_batch_diagnostics).
    """

    return session.run_continuous(
        tuple(int(item) for item in game_keys),
        tuple(states),
        evaluator,
        on_move,
        int(visits),
        float(c_puct),
        int(base_seed),
        int(virtual_batch_size),
        int(flush_target),
        int(active_root_limit),
        [float(item) for item in temperature_by_ply],
        root_dirichlet_total_alpha,
        root_dirichlet_noise_fraction,
        root_policy_temperature,
        fpu_reduction,
        virtual_loss,
        widening_policy_mass,
        widening_max_children,
        widening_min_children,
        forced_playout_k,
        root_policy_temperature_early,
        root_policy_temperature_halflife,
        pcr_full_proportion,
        pcr_fast_visits,
        policy_init_fraction,
        policy_init_avg_plies,
        policy_init_max_plies,
        policy_init_temperature,
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
