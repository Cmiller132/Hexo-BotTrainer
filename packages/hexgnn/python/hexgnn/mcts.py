"""Python boundary for the native hexgnn MCTS session.

Thin, like dense_cnn's `mcts.py`: Python owns the session object and decodes
byte-backed native results into dataclasses, while Rust owns state cloning,
validation, tree reuse, PUCT + nucleus widening, evaluator payload construction,
and action selection. The only difference from dense_cnn is the evaluator
callback (`HexgnnInference.evaluate_graph_facts`, a graph payload + per-candidate
priors) and the candidate-radius `n` threaded into the session.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import rust_bridge
from .constants import DEFAULT_CANDIDATE_RADIUS
from .inference import HexgnnInference


def new_mcts_session(
    *, max_states: int = 1_048_576, n: int = DEFAULT_CANDIDATE_RADIUS
) -> "HexgnnMctsSession":
    """Create a native search session that keeps selected subtrees per game."""

    return HexgnnMctsSession(max_states=max_states, n=n)


class HexgnnMctsSession:
    """KataGo-style selected-subtree reuse for batched self-play games.

    `game_keys` identify independent games across turns. Rust promotes the
    selected child after each search, then keeps that subtree under the same key
    until the caller discards it or sends a state whose hash no longer matches.
    """

    def __init__(self, *, max_states: int = 1_048_576, n: int = DEFAULT_CANDIDATE_RADIUS) -> None:
        self.n = int(n)
        self._session = rust_bridge.new_mcts_session(max_states=max_states, n=self.n)

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
        inference: HexgnnInference,
        *,
        visits: int,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        seed: int | None = None,
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
        per_root_visits: Sequence[int] | None = None,
        per_root_forced_playout_k: Sequence[float] | None = None,
        per_root_noise: Sequence[bool] | None = None,
    ) -> list["SearchResult"]:
        """Search live root states through the native hexgnn MCTS session.

        The ``per_root_*`` overrides (one entry per game key) let a SINGLE batched
        call run roots with heterogeneous visit caps / forced-playouts / root noise —
        coalescing KataGo PCR's full+fast move mix into one full-width forward stream.
        Each None falls back to the scalar value broadcast to every root.
        """

        if not root_states:
            return []
        payloads = rust_bridge.mcts_session_search(
            self._session,
            game_keys,
            root_states,
            visits=visits,
            c_puct=c_puct,
            temperature=temperature,
            seed=0 if seed is None else int(seed),
            evaluator=inference.evaluate_featurized_batch,
            virtual_batch_size=virtual_batch_size,
            active_root_limit=active_root_limit,
            root_dirichlet_total_alpha=root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction=root_dirichlet_noise_fraction,
            root_policy_temperature=root_policy_temperature,
            fpu_reduction=fpu_reduction,
            virtual_loss=virtual_loss,
            widening_policy_mass=widening_policy_mass,
            widening_max_children=widening_max_children,
            widening_min_children=widening_min_children,
            forced_playout_k=forced_playout_k,
            move_temperatures=move_temperatures,
            per_root_visits=per_root_visits,
            per_root_forced_playout_k=per_root_forced_playout_k,
            per_root_noise=per_root_noise,
        )
        return [_result_from_payload(payload) for payload in payloads]


@dataclass(frozen=True, slots=True)
class CompactVisitPolicy(Sequence[tuple[int, float]]):
    """Byte-backed visit policy returned by Rust (decoded lazily on iteration)."""

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


def _policy_from_payload(payload: Mapping[str, Any], *, prefix: str) -> Sequence[tuple[int, float]]:
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
        raise ValueError(f"MCTS result {prefix} byte lengths must match {prefix}_count")
    return CompactVisitPolicy(
        action_ids_bytes=action_ids_bytes,
        weights_bytes=weights_bytes,
        count=count,
    )
