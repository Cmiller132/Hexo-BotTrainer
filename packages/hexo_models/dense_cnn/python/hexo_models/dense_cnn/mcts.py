"""Required-Rust MCTS boundary for the dense CNN self-play path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import rust_bridge
from .inference import DenseCNNInference


def new_mcts_evaluation_cache(*, max_states: int = 131_072) -> object:
    """Create a native cache scoped to a single model-weight snapshot."""

    return rust_bridge.model1_new_mcts_evaluation_cache(max_states=max_states)


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_id: int
    visit_policy: Mapping[int, float]
    root_value: float
    visits: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def run_mcts(
    root_state: object,
    inference: DenseCNNInference,
    *,
    visits: int,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    seed: int | None = None,
    progressive_widening_initial_actions: int | None = 128,
    progressive_widening_child_initial_actions: int | None = 32,
    progressive_widening_candidate_actions: int | None = 192,
    progressive_widening_growth_interval: float | None = 40.0,
    progressive_widening_growth_base: float | None = 1.3,
    evaluation_cache: object | None = None,
    active_root_limit: int | None = None,
) -> SearchResult:
    """Run a single-root dense CNN MCTS search in Rust."""

    return run_batched_mcts(
        [root_state],
        inference,
        visits=visits,
        c_puct=c_puct,
        temperature=temperature,
        seed=seed,
        progressive_widening_initial_actions=progressive_widening_initial_actions,
        progressive_widening_child_initial_actions=progressive_widening_child_initial_actions,
        progressive_widening_candidate_actions=progressive_widening_candidate_actions,
        progressive_widening_growth_interval=progressive_widening_growth_interval,
        progressive_widening_growth_base=progressive_widening_growth_base,
        evaluation_cache=evaluation_cache,
        active_root_limit=active_root_limit,
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
    progressive_widening_initial_actions: int | None = 128,
    progressive_widening_child_initial_actions: int | None = 32,
    progressive_widening_candidate_actions: int | None = 192,
    progressive_widening_growth_interval: float | None = 40.0,
    progressive_widening_growth_base: float | None = 1.3,
    evaluation_cache: object | None = None,
    active_root_limit: int | None = None,
) -> list[SearchResult]:
    """Run dense CNN root searches through the required Rust accelerator."""

    if not root_states:
        return []
    target_visits = max(1, int(visits))
    root_limit = max(1, int(active_root_limit or len(root_states)))
    if len(root_states) > root_limit:
        results: list[SearchResult] = []
        for start in range(0, len(root_states), root_limit):
            chunk = root_states[start : start + root_limit]
            results.extend(
                run_batched_mcts(
                    chunk,
                    inference,
                    visits=target_visits,
                    c_puct=c_puct,
                    temperature=temperature,
                    seed=(0 if seed is None else int(seed)) + start,
                    virtual_batch_size=virtual_batch_size,
                    progressive_widening_initial_actions=progressive_widening_initial_actions,
                    progressive_widening_child_initial_actions=progressive_widening_child_initial_actions,
                    progressive_widening_candidate_actions=progressive_widening_candidate_actions,
                    progressive_widening_growth_interval=progressive_widening_growth_interval,
                    progressive_widening_growth_base=progressive_widening_growth_base,
                    evaluation_cache=evaluation_cache,
                    active_root_limit=root_limit,
                )
            )
        return results
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
        progressive_widening_initial_actions=progressive_widening_initial_actions,
        progressive_widening_child_initial_actions=progressive_widening_child_initial_actions,
        progressive_widening_candidate_actions=progressive_widening_candidate_actions,
        progressive_widening_growth_interval=progressive_widening_growth_interval,
        progressive_widening_growth_base=progressive_widening_growth_base,
        evaluation_cache=evaluation_cache,
    )
    return [_result_from_payload(payload) for payload in payloads]


def _resolve_virtual_batch_size(
    *,
    root_count: int,
    visits: int,
    virtual_batch_size: int | None,
) -> int:
    max_total_virtual_leaves = 8192
    max_virtual_per_root = max(1, max_total_virtual_leaves // max(1, int(root_count)))
    if virtual_batch_size is not None:
        return max(1, min(int(virtual_batch_size), int(visits), max_virtual_per_root))
    return max(1, min(max(1, int(visits)), max_virtual_per_root))


def _result_from_payload(payload: Mapping[str, Any]) -> SearchResult:
    return SearchResult(
        action_id=int(payload["action_id"]),
        visit_policy={
            int(action_id): float(weight)
            for action_id, weight in payload["visit_policy"]
        },
        root_value=float(payload["root_value"]),
        visits=int(payload["visits"]),
        diagnostics=dict(payload.get("diagnostics", {})),
    )
