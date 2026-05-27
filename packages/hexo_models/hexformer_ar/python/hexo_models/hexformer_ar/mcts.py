"""Required Rust MCTS boundary for Hexformer candidate priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import hexo_engine as engine
from hexo_engine.types import pack_coord_id

from .inference import HexformerInference

try:
    from hexo_models import _rust
except ImportError as exc:  # pragma: no cover - source checkout before Rust is built.
    _rust = None
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_id: int
    visit_policy: Mapping[int, float]
    root_value: float
    visits: int


def run_mcts(
    root_state: object,
    inference: HexformerInference,
    *,
    visits: int,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    seed: int | None = None,
) -> SearchResult:
    return run_batched_mcts(
        (root_state,),
        inference,
        visits=visits,
        c_puct=c_puct,
        temperature=temperature,
        seed=seed,
    )[0]


def run_batched_mcts(
    root_states: Sequence[object],
    inference: HexformerInference,
    *,
    visits: int,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    seed: int | None = None,
    virtual_batch_size: int | None = None,
) -> list[SearchResult]:
    """Run Rust-owned PUCT searches with batched Hexformer leaf evaluation."""

    if not root_states:
        return []
    resolved_virtual_batch_size = (
        max(1, int(virtual_batch_size))
        if virtual_batch_size is not None
        else max(1, int(visits))
    )
    try:
        payloads = _hexformer_ar_module().hexformer_ar_batched_mcts(
            history_rows_from_states(root_states),
            max(1, int(visits)),
            float(c_puct),
            float(temperature),
            0 if seed is None else int(seed),
            inference.evaluate_mcts_payload,
            resolved_virtual_batch_size,
        )
    except ValueError as exc:
        if "Hexformer MCTS root has no legal candidate actions" in str(exc):
            raise RuntimeError("Hexformer MCTS root has no legal candidate actions") from exc
        raise

    return [_search_result_from_payload(payload) for payload in payloads]


def history_rows_from_states(states: Sequence[object]) -> tuple[tuple[int, ...], ...]:
    """Return packed placement-history rows for Rust MCTS reconstruction."""

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


def _search_result_from_payload(payload: Mapping[str, Any]) -> SearchResult:
    return SearchResult(
        action_id=int(payload["action_id"]),
        visit_policy={
            int(action_id): float(weight)
            for action_id, weight in payload["visit_policy"]
        },
        root_value=float(payload["root_value"]),
        visits=int(payload["visits"]),
    )


def _hexformer_ar_module() -> Any:
    module = getattr(_rust, "hexformer_ar", None) if _rust is not None else None
    if module is None:
        raise RuntimeError(f"hexformer_ar Rust MCTS accelerator is unavailable: {_IMPORT_ERROR}")
    return module
