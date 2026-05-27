"""Small Python PUCT search used by the dense CNN self-play path."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import Random
from typing import Mapping, Sequence

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from .inference import DenseCNNInference


@dataclass(slots=True)
class Edge:
    action_id: int
    prior: float
    action: engine.PlacementAction | None = None
    visits: int = 0
    value_sum: float = 0.0
    pending: int = 0
    child: "Node | None" = None

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass(slots=True)
class Node:
    player: str
    edges: list[Edge] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


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
    root = _expand(root_state, inference)
    if not root.edges:
        raise RuntimeError("MCTS root has no legal actions")

    for _ in range(max(1, int(visits))):
        state = engine.clone_state(root_state)
        path: list[tuple[Node, Edge]] = []
        node = root

        while node.edges:
            edge = _select_edge(node, c_puct)
            engine.apply_action(state, _edge_action(edge))
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
            leaf_player = node.player
            _backup(path, leaf_player, node.value)

    policy_total = sum(edge.visits for edge in root.edges)
    if policy_total <= 0:
        policy = {edge.action_id: edge.prior for edge in root.edges}
    else:
        policy = {
            edge.action_id: edge.visits / policy_total
            for edge in root.edges
        }
    selected = _select_root_action(root, temperature=temperature, seed=seed)
    return SearchResult(
        action_id=selected,
        visit_policy=policy,
        root_value=root.value,
        visits=policy_total,
    )


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
    """Run root searches with batched neural evaluation at roots and leaves."""

    if not root_states:
        return []
    if hasattr(engine, "model1_batched_mcts"):
        resolved_virtual_batch_size = virtual_batch_size
        if resolved_virtual_batch_size is None:
            resolved_virtual_batch_size = max(1, min(max(1, int(visits)), 8192 // max(1, len(root_states))))
        payloads = engine.model1_batched_mcts(
            root_states,
            visits=max(1, int(visits)),
            c_puct=float(c_puct),
            temperature=float(temperature),
            seed=0 if seed is None else int(seed),
            evaluator=inference.evaluate_model1_payload,
            virtual_batch_size=resolved_virtual_batch_size,
        )
        return [
            SearchResult(
                action_id=int(payload["action_id"]),
                visit_policy={int(action_id): float(weight) for action_id, weight in payload["visit_policy"]},
                root_value=float(payload["root_value"]),
                visits=int(payload["visits"]),
            )
            for payload in payloads
        ]
    root_evals = inference.infer_states(root_states)
    roots = [_node_from_eval(state, evaluation) for state, evaluation in zip(root_states, root_evals)]
    target_visits = max(1, int(visits))
    leaf_batch_per_root = max(1, int(virtual_batch_size or target_visits))
    completed = [0 for _root in roots]

    while any(count < target_visits for count in completed):
        leaves: list[tuple[object, Edge, list[tuple[Node, Edge]]]] = []
        immediate: list[tuple[list[tuple[Node, Edge]], str, float]] = []
        made_progress = False
        for root_index, (root_state, root) in enumerate(zip(root_states, roots)):
            if completed[root_index] >= target_visits or not root.edges:
                continue
            root_budget = min(leaf_batch_per_root, target_visits - completed[root_index])
            for _ in range(root_budget):
                selected = _select_pending_leaf(root_state, root, c_puct)
                if selected is None:
                    break
                path, state, edge, terminal, leaf_node = selected
                _apply_virtual_visit(path)
                completed[root_index] += 1
                made_progress = True
                if terminal is not None:
                    leaf_player = _player_label(engine.current_player(state))
                    immediate.append((path, leaf_player, _terminal_value(terminal, leaf_player)))
                elif leaf_node is not None:
                    immediate.append((path, leaf_node.player, leaf_node.value))
                else:
                    edge.pending += 1
                    leaves.append((state, edge, path))

        for path, leaf_player, leaf_value in immediate:
            _backup_virtual(path, leaf_player, leaf_value)

        if leaves:
            evals = inference.infer_states([leaf[0] for leaf in leaves])
            for (state, edge, path), evaluation in zip(leaves, evals):
                child = _node_from_eval(state, evaluation)
                edge.child = child
                edge.pending = max(0, edge.pending - 1)
                _backup_virtual(path, child.player, child.value)

        if not made_progress:
            break

    results: list[SearchResult] = []
    for offset, root in enumerate(roots):
        if not root.edges:
            raise RuntimeError("MCTS root has no legal actions")
        policy_total = sum(edge.visits for edge in root.edges)
        policy = {
            edge.action_id: (edge.visits / policy_total if policy_total > 0 else edge.prior)
            for edge in root.edges
        }
        selected = _select_root_action(root, temperature=temperature, seed=None if seed is None else seed + offset)
        results.append(
            SearchResult(
                action_id=selected,
                visit_policy=policy,
                root_value=root.value,
                visits=policy_total,
            )
        )
    return results


def _select_pending_leaf(
    root_state: object,
    root: Node,
    c_puct: float,
) -> tuple[list[tuple[Node, Edge]], object, Edge, object | None, Node | None] | None:
    state = engine.clone_state(root_state)
    node = root
    path: list[tuple[Node, Edge]] = []
    last_edge: Edge | None = None
    while node.edges:
        edge = _select_edge(node, c_puct)
        if edge.pending > 0 and edge.child is None:
            return None
        engine.apply_action(state, _edge_action(edge))
        path.append((node, edge))
        last_edge = edge
        if edge.child is None:
            return path, state, edge, engine.terminal(state), None
        node = edge.child
    if last_edge is None:
        return None
    return path, state, last_edge, None, node


def _expand(state: object, inference: DenseCNNInference) -> Node:
    result = inference.infer_state(state)
    return _node_from_eval(state, result)


def _node_from_eval(state: object, result: object) -> Node:
    player = _player_label(engine.current_player(state))
    node = Node(player=player, value_sum=result.value, visits=1)
    for action_id, prior in result.legal_priors.items():
        action_id = int(action_id)
        node.edges.append(
            Edge(
                action_id=action_id,
                prior=max(0.0, float(prior)),
            )
        )
    _normalize_priors(node)
    return node


def _select_edge(node: Node, c_puct: float) -> Edge:
    exploration_scale = c_puct * sqrt(max(1, node.visits))
    available_edges = [edge for edge in node.edges if edge.pending == 0 or edge.child is not None]
    if not available_edges:
        available_edges = node.edges
    return max(
        available_edges,
        key=lambda edge: (
            edge.value + edge.prior * exploration_scale / (1 + edge.visits),
            -edge.visits,
            -edge.action_id,
        ),
    )


def _select_root_action(node: Node, *, temperature: float, seed: int | None) -> int:
    if temperature <= 1.0e-6:
        return max(node.edges, key=lambda edge: (edge.visits, -edge.action_id)).action_id
    rng = Random(seed)
    inv_temperature = 1.0 / max(temperature, 1.0e-3)
    weights = [(edge.visits or 1) ** inv_temperature for edge in node.edges]
    total = sum(weights)
    threshold = rng.random() * total
    for edge, weight in zip(node.edges, weights):
        threshold -= weight
        if threshold <= 0:
            return edge.action_id
    return node.edges[-1].action_id


def _edge_action(edge: Edge) -> engine.PlacementAction:
    action = edge.action
    if action is None:
        action = engine.PlacementAction(unpack_coord_id(edge.action_id))
        edge.action = action
    return action


def _backup(path: list[tuple[Node, Edge]], leaf_player: str, leaf_value: float) -> None:
    for node, edge in path:
        value = leaf_value if node.player == leaf_player else -leaf_value
        edge.visits += 1
        edge.value_sum += value
        node.visits += 1
        node.value_sum += value


def _apply_virtual_visit(path: list[tuple[Node, Edge]]) -> None:
    for node, edge in path:
        edge.visits += 1
        node.visits += 1


def _backup_virtual(path: list[tuple[Node, Edge]], leaf_player: str, leaf_value: float) -> None:
    for node, edge in path:
        value = leaf_value if node.player == leaf_player else -leaf_value
        edge.value_sum += value
        node.value_sum += value


def _normalize_priors(node: Node) -> None:
    total = sum(max(0.0, edge.prior) for edge in node.edges)
    if total <= 0.0:
        prior = 1.0 / max(1, len(node.edges))
        for edge in node.edges:
            edge.prior = prior
        return
    for edge in node.edges:
        edge.prior = max(0.0, edge.prior) / total


def _terminal_value(terminal: object | None, player: str) -> float:
    winner = getattr(terminal, "winner", None)
    if winner is None:
        return 0.0
    return 1.0 if _player_label(winner) == player else -1.0


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
