"""Required Rust MCTS boundary for dense CNN search.

This module is deliberately thin. Python owns the user-facing session object and
turns byte-backed native results into Python dataclasses, while the Rust session
owns state cloning, argument validation, tree reuse, PUCT search, evaluator
payload construction, and action selection.

The only production search entry point is `BatchedMctsSession.run`: callers pass
game keys, live `hexo_engine.HexoState` roots, and a `DenseCNNInference`
instance. There are no one-shot Python search wrappers or Python-side state
payload fallbacks.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import rust_bridge
from .inference import DenseCNNInference


def new_mcts_session(*, max_states: int = 1_048_576) -> "BatchedMctsSession":
    """Create a native search session that keeps selected subtrees per game."""

    return BatchedMctsSession(max_states=max_states)


class BatchedMctsSession:
    """KataGo-style selected-subtree reuse for batched self-play games.

    `game_keys` identify independent games across turns. Rust promotes the
    selected child after each search, then keeps that subtree under the same key
    until the caller discards it or sends a state whose hash no longer matches.
    """

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
        root_dirichlet_alpha: float | None = None,
        root_dirichlet_noise_fraction: float | None = None,
        hidden_prior_mass: float | None = 0.05,
        fpu_reduction: float | None = 0.20,
        virtual_loss: float | None = 1.0,
        active_root_limit: int | None = None,
    ) -> list["SearchResult"]:
        """Search live root states through the native dense-cnn MCTS session.

        The Python side supplies the `DenseCNNInference` callback. Every other
        search detail, including cloning engine states, validating numeric
        search settings, batching leaves, parsing evaluator bytes, and selecting
        the returned action, belongs to Rust.
        """

        if not root_states:
            return []
        payloads = rust_bridge.model1_mcts_session_search(
            self._session,
            game_keys,
            root_states,
            visits=visits,
            c_puct=c_puct,
            temperature=temperature,
            seed=0 if seed is None else int(seed),
            evaluator=inference.evaluate_model1_payload,
            virtual_batch_size=virtual_batch_size,
            progressive_widening_initial_actions=progressive_widening_initial_actions,
            progressive_widening_child_initial_actions=progressive_widening_child_initial_actions,
            progressive_widening_candidate_actions=progressive_widening_candidate_actions,
            progressive_widening_growth_interval=progressive_widening_growth_interval,
            progressive_widening_growth_base=progressive_widening_growth_base,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_dirichlet_noise_fraction=root_dirichlet_noise_fraction,
            hidden_prior_mass=hidden_prior_mass,
            fpu_reduction=fpu_reduction,
            virtual_loss=virtual_loss,
            active_root_limit=active_root_limit,
        )
        return [_result_from_payload(payload) for payload in payloads]


@dataclass(frozen=True, slots=True)
class CompactVisitPolicy(Sequence[tuple[int, float]]):
    """Byte-backed visit policy returned by Rust.

    Rust serializes action ids and weights into two contiguous buffers so large
    policies do not allocate thousands of Python tuples during search. This
    wrapper decodes lazily when callers iterate, index, or pass it to sample
    generation.
    """

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
    """One searched root result returned to self-play or runner players."""

    action_id: int
    visit_policy: Sequence[tuple[int, float]]
    root_value: float
    visits: int
    root_prior_policy: Sequence[tuple[int, float]]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _result_from_payload(payload: Mapping[str, Any]) -> SearchResult:
    """Convert the native result dict into the small Python result dataclass."""

    diagnostics = dict(payload.get("diagnostics", {}))
    if "action_selection" in payload:
        diagnostics["action_selection"] = str(payload["action_selection"])
    return SearchResult(
        action_id=int(payload["action_id"]),
        visit_policy=_policy_from_payload(payload, prefix="visit_policy"),
        root_value=float(payload["root_value"]),
        visits=int(payload["visits"]),
        root_prior_policy=_policy_from_payload(payload, prefix="root_prior_policy"),
        diagnostics=diagnostics,
    )


def _policy_from_payload(
    payload: Mapping[str, Any],
    *,
    prefix: str,
) -> Sequence[tuple[int, float]]:
    """Read a strict byte-only action/weight policy payload produced by Rust."""

    required = (
        f"{prefix}_action_ids_bytes",
        f"{prefix}_weights_bytes",
        f"{prefix}_count",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"MCTS result payload missing required byte field(s): {', '.join(missing)}")
    count = int(payload[f"{prefix}_count"])
    action_ids_bytes = bytes(payload[f"{prefix}_action_ids_bytes"])
    weights_bytes = bytes(payload[f"{prefix}_weights_bytes"])
    expected = count * 4
    if len(action_ids_bytes) != expected or len(weights_bytes) != expected:
        raise ValueError(
            f"MCTS result {prefix} byte lengths must match {prefix}_count"
        )
    return CompactVisitPolicy(
        action_ids_bytes=action_ids_bytes,
        weights_bytes=weights_bytes,
        count=count,
    )
