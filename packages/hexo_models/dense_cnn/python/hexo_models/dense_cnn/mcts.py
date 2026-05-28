"""Required-Rust MCTS boundary for the dense CNN self-play path."""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import rust_bridge
from .inference import DenseCNNInference


DEFAULT_ACTIVE_ROOT_LIMIT = 1024
DEFAULT_EVAL_CHUNK_STATES = 1024


def new_mcts_evaluation_cache(*, max_states: int = 1_048_576) -> object:
    """Create a native cache scoped to a single model-weight snapshot."""

    return rust_bridge.model1_new_mcts_evaluation_cache(max_states=max_states)


def new_mcts_session(*, max_states: int = 1_048_576) -> "BatchedMctsSession":
    """Create a native search session that keeps selected subtrees per game."""

    return BatchedMctsSession(max_states=max_states)


class BatchedMctsSession:
    """KataGo-style selected-subtree reuse for batched self-play games."""

    def __init__(self, *, max_states: int = 1_048_576) -> None:
        self._session = rust_bridge.model1_new_mcts_session(max_states=max_states)

    def clear(self) -> None:
        self._session.clear()

    def discard(self, game_key: int) -> None:
        self._session.discard(int(game_key))

    def __len__(self) -> int:
        return int(self._session.len())

    def run(
        self,
        game_keys: Sequence[int],
        root_states: Sequence[object],
        inference: DenseCNNInference,
        *,
        visits: int,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        seed: int | None = None,
        virtual_batch_size: int | None = None,
        progressive_widening_initial_actions: int | None = 8,
        progressive_widening_child_initial_actions: int | None = 4,
        progressive_widening_candidate_actions: int | None = 128,
        progressive_widening_growth_interval: float | None = 256.0,
        progressive_widening_growth_base: float | None = 1.3,
        active_root_limit: int | None = None,
    ) -> list["SearchResult"]:
        if not root_states:
            return []
        if len(game_keys) != len(root_states):
            raise ValueError(f"received {len(game_keys)} game keys for {len(root_states)} root states")
        target_visits = max(1, int(visits))
        root_limit = max(1, int(active_root_limit or DEFAULT_ACTIVE_ROOT_LIMIT))
        if len(root_states) > root_limit:
            results: list[SearchResult] = []
            for start in range(0, len(root_states), root_limit):
                results.extend(
                    self.run(
                        game_keys[start : start + root_limit],
                        root_states[start : start + root_limit],
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
                        active_root_limit=root_limit,
                    )
                )
            return results
        payloads = rust_bridge.model1_mcts_session_search(
            self._session,
            game_keys,
            root_states,
            visits=target_visits,
            c_puct=c_puct,
            temperature=temperature,
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
            active_root_limit=root_limit,
        )
        return [_result_from_payload(payload) for payload in payloads]


@dataclass(frozen=True, slots=True)
class CompactVisitPolicy(Sequence[tuple[int, float]]):
    action_ids_bytes: bytes
    weights_bytes: bytes
    count: int

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[tuple[int, float]]:
        for index in range(self.count):
            yield self[index]

    def __getitem__(self, index: int) -> tuple[int, float]:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return (
            int(struct.unpack_from("I", self.action_ids_bytes, index * 4)[0]),
            float(struct.unpack_from("f", self.weights_bytes, index * 4)[0]),
        )

    def items(self) -> Iterator[tuple[int, float]]:
        return iter(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_id: int
    visit_policy: Sequence[tuple[int, float]]
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
    progressive_widening_initial_actions: int | None = 8,
    progressive_widening_child_initial_actions: int | None = 4,
    progressive_widening_candidate_actions: int | None = 128,
    progressive_widening_growth_interval: float | None = 256.0,
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
    progressive_widening_initial_actions: int | None = 8,
    progressive_widening_child_initial_actions: int | None = 4,
    progressive_widening_candidate_actions: int | None = 128,
    progressive_widening_growth_interval: float | None = 256.0,
    progressive_widening_growth_base: float | None = 1.3,
    evaluation_cache: object | None = None,
    active_root_limit: int | None = None,
) -> list[SearchResult]:
    """Run dense CNN root searches through the required Rust accelerator."""

    if not root_states:
        return []
    target_visits = max(1, int(visits))
    root_limit = max(1, int(active_root_limit or DEFAULT_ACTIVE_ROOT_LIMIT))
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
        active_root_limit=root_limit,
    )
    return [_result_from_payload(payload) for payload in payloads]


def _resolve_virtual_batch_size(
    *,
    root_count: int,
    visits: int,
    virtual_batch_size: int | None,
) -> int:
    max_total_virtual_leaves = DEFAULT_EVAL_CHUNK_STATES
    max_virtual_per_root = max(1, max_total_virtual_leaves // max(1, int(root_count)))
    if virtual_batch_size is not None:
        return max(1, min(int(virtual_batch_size), int(visits), max_virtual_per_root))
    return max(1, min(max(1, int(visits)), max_virtual_per_root))


def _result_from_payload(payload: Mapping[str, Any]) -> SearchResult:
    return SearchResult(
        action_id=int(payload["action_id"]),
        visit_policy=_visit_policy_from_payload(payload),
        root_value=float(payload["root_value"]),
        visits=int(payload["visits"]),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def _visit_policy_from_payload(payload: Mapping[str, Any]) -> Sequence[tuple[int, float]]:
    if "visit_policy_action_ids_bytes" in payload:
        count = int(payload.get("visit_policy_count", 0))
        return CompactVisitPolicy(
            action_ids_bytes=bytes(payload["visit_policy_action_ids_bytes"]),
            weights_bytes=bytes(payload["visit_policy_weights_bytes"]),
            count=count,
        )
    return tuple(
        (int(action_id), float(weight))
        for action_id, weight in payload["visit_policy"]
    )
