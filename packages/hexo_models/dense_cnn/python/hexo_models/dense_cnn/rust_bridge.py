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
