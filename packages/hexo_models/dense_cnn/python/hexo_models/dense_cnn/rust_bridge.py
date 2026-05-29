"""Required Rust acceleration owned by the dense CNN model package."""

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


def model1_batched_mcts(
    states: Sequence[object],
    *,
    visits: int,
    c_puct: float,
    temperature: float,
    seed: int,
    evaluator: object,
    virtual_batch_size: int | None = None,
    progressive_widening_initial_actions: int | None = None,
    progressive_widening_child_initial_actions: int | None = None,
    progressive_widening_candidate_actions: int | None = None,
    progressive_widening_growth_interval: float | None = None,
    progressive_widening_growth_base: float | None = None,
    root_dirichlet_alpha: float | None = None,
    root_exploration_fraction: float | None = None,
    evaluation_cache: object | None = None,
    active_root_limit: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Run dense-cnn Rust MCTS from live engine states."""

    return tuple(
        _dense_cnn_module().model1_batched_mcts(
            tuple(states),
            int(visits),
            float(c_puct),
            float(temperature),
            int(seed),
            evaluator,
            None if virtual_batch_size is None else max(1, int(virtual_batch_size)),
            None if progressive_widening_initial_actions is None else max(1, int(progressive_widening_initial_actions)),
            None if progressive_widening_child_initial_actions is None else max(1, int(progressive_widening_child_initial_actions)),
            None if progressive_widening_growth_interval is None else max(1.0, float(progressive_widening_growth_interval)),
            None if progressive_widening_growth_base is None else max(1.000001, float(progressive_widening_growth_base)),
            None if progressive_widening_candidate_actions is None else max(1, int(progressive_widening_candidate_actions)),
            None if root_dirichlet_alpha is None else max(0.0, float(root_dirichlet_alpha)),
            None if root_exploration_fraction is None else min(1.0, max(0.0, float(root_exploration_fraction))),
            evaluation_cache,
            None if active_root_limit is None else max(1, int(active_root_limit)),
        )
    )


def model1_new_mcts_evaluation_cache(*, max_states: int | None = None) -> object:
    """Create a native scoped MCTS evaluation cache for one model-weight snapshot."""

    return _dense_cnn_module().Model1MctsEvaluationCache(
        None if max_states is None else max(1, int(max_states))
    )


def model1_new_mcts_session(*, max_states: int | None = None) -> object:
    """Create a native MCTS session that reuses selected subtrees across moves."""

    return _dense_cnn_module().Model1MctsSession(
        None if max_states is None else max(1, int(max_states))
    )


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
    progressive_widening_initial_actions: int | None = None,
    progressive_widening_child_initial_actions: int | None = None,
    progressive_widening_candidate_actions: int | None = None,
    progressive_widening_growth_interval: float | None = None,
    progressive_widening_growth_base: float | None = None,
    root_dirichlet_alpha: float | None = None,
    root_exploration_fraction: float | None = None,
    active_root_limit: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Search through a native MCTS session, preserving chosen subtrees."""

    return tuple(
        session.search(
            tuple(int(item) for item in game_keys),
            tuple(states),
            int(visits),
            float(c_puct),
            float(temperature),
            int(seed),
            evaluator,
            None if virtual_batch_size is None else max(1, int(virtual_batch_size)),
            None if progressive_widening_initial_actions is None else max(1, int(progressive_widening_initial_actions)),
            None if progressive_widening_child_initial_actions is None else max(1, int(progressive_widening_child_initial_actions)),
            None if progressive_widening_growth_interval is None else max(1.0, float(progressive_widening_growth_interval)),
            None if progressive_widening_growth_base is None else max(1.000001, float(progressive_widening_growth_base)),
            None if progressive_widening_candidate_actions is None else max(1, int(progressive_widening_candidate_actions)),
            None if root_dirichlet_alpha is None else max(0.0, float(root_dirichlet_alpha)),
            None if root_exploration_fraction is None else min(1.0, max(0.0, float(root_exploration_fraction))),
            None if active_root_limit is None else max(1, int(active_root_limit)),
        )
    )


def model1_sample_from_state(
    state: object,
    *,
    game_id: str,
    turn_index: int,
    policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    value: float = 0.0,
    opp_policy: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    lookahead: Mapping[int, float] | Sequence[tuple[int, float]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build one compact sample from a live engine state in Rust."""

    return _dense_cnn_module().model1_sample_from_state(
        state,
        str(game_id),
        int(turn_index),
        policy,
        float(value),
        opp_policy,
        lookahead,
        dict(metadata or {}),
    )


def model1_finalize_game_samples(
    pending: Sequence[tuple[str, Mapping[str, Any], float]],
    *,
    winner: str | None,
    horizons: Sequence[int],
    truncated: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Finalize compact game samples in Rust."""

    return tuple(
        _dense_cnn_module().model1_finalize_game_samples(
            tuple(pending),
            winner,
            tuple(int(item) for item in horizons),
            bool(truncated),
        )
    )


def _dense_cnn_module() -> Any:
    module = getattr(_rust, "dense_cnn", None) if _rust is not None else None
    if module is None:
        raise RuntimeError(f"dense_cnn Rust accelerator is unavailable: {_IMPORT_ERROR}")
    return module
