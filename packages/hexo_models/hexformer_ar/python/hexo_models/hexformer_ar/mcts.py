"""Python PUCT search over Hexformer candidate priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import Random
from typing import Mapping

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from .inference import HexformerInference


@dataclass(slots=True)
class Edge:
    action_id: int
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "Node | None" = None

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass(slots=True)
class Node:
    player: str
    value: float
    edges: list[Edge] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value if self.visits == 0 else self.value_sum / self.visits


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
    root = _expand(root_state, inference)
    if not root.edges:
        raise RuntimeError("Hexformer MCTS root has no legal candidate actions")
    for _ in range(max(1, int(visits))):
        state = engine.clone_state(root_state)
        node = root
        path: list[tuple[Node, Edge]] = []
        while node.edges:
            edge = _select_edge(node, c_puct)
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(edge.action_id)))
            path.append((node, edge))
            if edge.child is None:
                terminal = engine.terminal(state)
                if terminal is None:
                    edge.child = _expand(state, inference)
                    leaf_player = edge.child.player
                    leaf_value = edge.child.value
                else:
                    leaf_player = _player_label(engine.current_player(state))
                    leaf_value = _terminal_value(terminal, leaf_player)
                _backup(path, leaf_player, leaf_value)
                break
            node = edge.child
        else:
            _backup(path, node.player, node.mean_value)
    total = sum(edge.visits for edge in root.edges)
    policy = {
        edge.action_id: (edge.visits / total if total > 0 else edge.prior)
        for edge in root.edges
    }
    return SearchResult(
        action_id=_select_root_action(root, temperature=temperature, seed=seed),
        visit_policy=policy,
        root_value=root.mean_value,
        visits=total,
    )


def _expand(state: object, inference: HexformerInference) -> Node:
    result = inference.infer_state(state)
    node = Node(player=_player_label(engine.current_player(state)), value=result.value)
    node.edges = [Edge(action_id=action_id, prior=max(0.0, float(prior))) for action_id, prior in result.legal_priors.items()]
    _normalize_edges(node.edges)
    return node


def _select_edge(node: Node, c_puct: float) -> Edge:
    total = max(1, node.visits)

    def score(edge: Edge) -> tuple[float, int]:
        q = edge.value
        u = float(c_puct) * edge.prior * sqrt(total) / (1 + edge.visits)
        return (q + u, -edge.action_id)

    return max(node.edges, key=score)


def _backup(path: list[tuple[Node, Edge]], leaf_player: str, leaf_value: float) -> None:
    for node, edge in reversed(path):
        value = leaf_value if node.player == leaf_player else -leaf_value
        node.visits += 1
        node.value_sum += value
        edge.visits += 1
        edge.value_sum += value


def _select_root_action(node: Node, *, temperature: float, seed: int | None) -> int:
    if temperature <= 1.0e-6:
        return max(node.edges, key=lambda edge: (edge.visits, -edge.action_id)).action_id
    rng = Random(seed)
    weights = [max(1, edge.visits) ** (1.0 / max(1.0e-3, temperature)) for edge in node.edges]
    total = sum(weights)
    threshold = rng.random() * total
    for edge, weight in zip(node.edges, weights):
        threshold -= weight
        if threshold <= 0:
            return edge.action_id
    return node.edges[-1].action_id


def _normalize_edges(edges: list[Edge]) -> None:
    total = sum(max(0.0, edge.prior) for edge in edges)
    if total <= 0.0 and edges:
        prior = 1.0 / len(edges)
        for edge in edges:
            edge.prior = prior
        return
    for edge in edges:
        edge.prior = max(0.0, edge.prior) / total


def _terminal_value(terminal: object, player: str) -> float:
    winner = getattr(terminal, "winner", None)
    if winner is None:
        return 0.0
    return 1.0 if _player_label(winner) == player else -1.0


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
