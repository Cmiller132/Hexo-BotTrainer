"""Optional Rust acceleration owned by the dense CNN model package."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import hexo_engine as engine
from hexo_engine.types import pack_coord_id

try:
    from hexo_models import _rust
except ImportError as exc:  # pragma: no cover - source checkouts may not build the extension.
    _rust = None
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


def is_available() -> bool:
    """Return whether the private dense-cnn Rust accelerator is importable."""

    return _dense_cnn_module(required=False) is not None


def capabilities() -> Mapping[str, Any]:
    """Return the Rust accelerator capability payload."""

    return _dense_cnn_module().capabilities()


def model1_batch_inputs(states: Sequence[object]) -> Mapping[str, Any]:
    """Encode live engine states through the dense-cnn-owned Rust accelerator."""

    return _dense_cnn_module().model1_batch_inputs(history_rows_from_states(states))


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
    """Run dense-cnn Rust MCTS from model-local reconstructed states."""

    return tuple(
        _dense_cnn_module().model1_batched_mcts(
            history_rows_from_states(states),
            int(visits),
            float(c_puct),
            float(temperature),
            int(seed),
            evaluator,
            None if virtual_batch_size is None else max(1, int(virtual_batch_size)),
        )
    )


def history_rows_from_states(states: Sequence[object]) -> tuple[tuple[int, ...], ...]:
    """Return packed action-history rows for live engine states."""

    rows: list[tuple[int, ...]] = []
    for state in states:
        python_state = engine.to_python_state(state)
        rows.append(
            tuple(
                int(pack_coord_id(record.coord))
                for record in python_state.placement_history
            )
        )
    return tuple(rows)


def import_error() -> BaseException | None:
    return _IMPORT_ERROR


def _dense_cnn_module(*, required: bool = True) -> Any:
    module = getattr(_rust, "dense_cnn", None) if _rust is not None else None
    if module is None and required:
        raise RuntimeError(f"dense_cnn Rust accelerator is unavailable: {_IMPORT_ERROR}")
    return module
