"""Required-Rust MCTS boundary for the dense CNN self-play path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import rust_bridge
from .inference import DenseCNNInference


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_id: int
    visit_policy: Mapping[int, float]
    root_value: float
    visits: int


def run_mcts(
    root_state: object,
    inference: DenseCNNInference,
    *,
    visits: int,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    seed: int | None = None,
) -> SearchResult:
    """Run a single-root dense CNN MCTS search in Rust."""

    return run_batched_mcts(
        [root_state],
        inference,
        visits=visits,
        c_puct=c_puct,
        temperature=temperature,
        seed=seed,
    )[0]


def run_batched_mcts(
    root_states: Sequence[object],
    inference: DenseCNNInference,
    *,
    visits: int,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    seed: int | None = None,
    virtual_batch_size: int | None = None,
) -> list[SearchResult]:
    """Run dense CNN root searches through the required Rust accelerator."""

    if not root_states:
        return []
    target_visits = max(1, int(visits))
    payloads = rust_bridge.model1_batched_mcts(
        root_states,
        visits=target_visits,
        c_puct=float(c_puct),
        temperature=float(temperature),
        seed=0 if seed is None else int(seed),
        evaluator=inference.evaluate_model1_payload,
        virtual_batch_size=_resolve_virtual_batch_size(
            root_count=len(root_states),
            visits=target_visits,
            virtual_batch_size=virtual_batch_size,
        ),
    )
    return [_result_from_payload(payload) for payload in payloads]


def _resolve_virtual_batch_size(
    *,
    root_count: int,
    visits: int,
    virtual_batch_size: int | None,
) -> int:
    if virtual_batch_size is not None:
        return max(1, int(virtual_batch_size))
    return max(1, min(max(1, int(visits)), 8192 // max(1, int(root_count))))


def _result_from_payload(payload: Mapping[str, Any]) -> SearchResult:
    return SearchResult(
        action_id=int(payload["action_id"]),
        visit_policy={
            int(action_id): float(weight)
            for action_id, weight in payload["visit_policy"]
        },
        root_value=float(payload["root_value"]),
        visits=int(payload["visits"]),
    )
