"""Expand-time graph construction (the MVP recompute-at-expand path, §7).

Rebuilds the candidate set / window tokens / typed graph + targets per epoch from
the COMPACT raw-fact rows (`dense_cnn.compact_io.read_compact_shard` — the same
representation-agnostic shard format; no new SCHEMA_VERSION). Each row's position
is reconstructed by replaying its placement history through the engine, then the
SHARED Rust builder produces the graph (so training inputs == search inputs).

Targets reuse dense_cnn's finalized facts (value / opp_policy were already
constructed per per-placement turn). hexgnn has NO short-term-value heads, so STV
targets are not built here. The only remapping is the
policy/opp visit distribution from action-id space onto the candidate nodes
(CSR order). At candidate_radius = 8 the candidate set is the full legal set, so
dropped visit mass is ~0; at smaller n we drop out-of-candidate visits,
renormalize, and report the dropped mass (open-q #4).

D6 augmentation is intentionally NOT applied: the model is D6-INVARIANT by
construction (features.py / the equivariance test), so rotating the data would
produce identical training signal — augmentation is redundant here.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import hexo_engine.api as engine
from hexo_engine.types import AxialCoord, PlacementAction

from .collate import collate_graphs
from .constants import DEFAULT_CANDIDATE_RADIUS
from .features import GraphTensors, build_graph_tensors
from . import rust_bridge


class ExpandedRow:
    """One expanded position: graph tensors + per-graph training targets."""

    __slots__ = (
        "graph", "policy", "opp_policy", "value",
        "dropped_policy_mass", "policy_total",
    )

    def __init__(
        self,
        graph: GraphTensors,
        policy: np.ndarray,
        opp_policy: np.ndarray,
        value: float,
        dropped_policy_mass: float,
        policy_total: float,
    ) -> None:
        """Bundle the graph tensors and the per-graph training targets that
        trainer / selfplay / evaluation read directly off this row.

        Args:
            graph: the candidate/window typed graph for this position.
            policy: (C,) float32 visit weights over the C candidates (CSR order).
            opp_policy: (C,) float32 opponent visit weights, same order.
            value: scalar game-result target in [-1, 1].
            dropped_policy_mass: visit mass that fell outside the candidate set.
            policy_total: total recorded visit mass (in- and out-of-set).
        """
        self.graph = graph
        self.policy = policy
        self.opp_policy = opp_policy
        self.value = value
        self.dropped_policy_mass = dropped_policy_mass
        self.policy_total = policy_total

    @property
    def dropped_policy_fraction(self) -> float:
        return self.dropped_policy_mass / self.policy_total if self.policy_total > 0 else 0.0

    def should_prune(self, max_dropped_mass: float) -> bool:
        """Prune if no candidate carries policy mass, or too much mass fell outside."""

        in_set = self.policy_total - self.dropped_policy_mass
        if in_set <= 0.0:
            return True
        return self.dropped_policy_fraction > float(max_dropped_mass)


def reconstruct_state(placement_history: Sequence[Any]) -> object:
    """Replay an idx-ordered placement history into a fresh engine state."""

    ordered = sorted(placement_history, key=lambda h: h[4])  # entry[4] = placement_index
    state = engine.new_game()
    for entry in ordered:
        q, r = int(entry[0]), int(entry[1])
        engine.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    return state


def _candidate_targets(candidate_ids: np.ndarray, pairs: Sequence[tuple[int, float]]) -> tuple[np.ndarray, float]:
    """Map a visit distribution (action_id -> weight) onto candidate CSR order.

    Returns (per-candidate weights, dropped mass not covered by any candidate).
    """

    weights = {int(a): float(w) for a, w in pairs}
    cand = candidate_ids.tolist()
    cand_set = set(cand)
    out = np.array([weights.get(int(a), 0.0) for a in cand], dtype=np.float32)
    dropped = sum(w for a, w in weights.items() if a not in cand_set)
    return out, float(dropped)


def expand_row_to_graph(row: Any, n: int = DEFAULT_CANDIDATE_RADIUS) -> ExpandedRow:
    """Expand one compact raw-fact row into graph tensors + targets."""

    state = reconstruct_state(row.placement_history)
    facts = rust_bridge.graph_facts(state, n)
    graph = build_graph_tensors(facts)

    policy, dropped = _candidate_targets(graph.candidate_ids, row.policy)
    opp_policy, _ = _candidate_targets(graph.candidate_ids, getattr(row, "opp_policy", ()))

    return ExpandedRow(
        graph=graph,
        policy=policy,
        opp_policy=opp_policy,
        value=float(row.value),
        dropped_policy_mass=dropped,
        policy_total=float(sum(float(w) for _a, w in row.policy)),
    )


def build_training_batch(
    rows: Sequence[Any],
    *,
    n: int = DEFAULT_CANDIDATE_RADIUS,
    prune_max_dropped_mass: float = 1.0,
) -> tuple[dict, dict] | tuple[None, None]:
    """Expand + collate a list of compact rows into a (batch, targets) pair.

    `batch` is the model forward input; `targets` carries the per-candidate
    `policy`/`opp_policy` and the per-graph `value` for `hexgnn_loss` (hexgnn has
    no STV heads). DATASET PRUNING (candidate_radius decision): rows whose policy
    visit-mass falls outside the candidate set beyond `prune_max_dropped_mass`
    (or with zero in-set mass) are dropped; `targets["_pruned"]/["_kept"]` report
    the rate. Surviving rows renormalize over in-set candidates (the segmented CE
    normalizes per graph). Returns (None, None) if every row is pruned.
    """

    import torch

    all_expanded = [expand_row_to_graph(r, n=n) for r in rows]
    expanded = [e for e in all_expanded if not e.should_prune(prune_max_dropped_mass)]
    n_pruned = len(all_expanded) - len(expanded)
    if not expanded:
        return None, None
    graphs = [e.graph for e in expanded]
    batch = collate_graphs(graphs)

    policy = np.concatenate([e.policy for e in expanded]) if expanded else np.zeros(0, np.float32)
    opp = np.concatenate([e.opp_policy for e in expanded]) if expanded else np.zeros(0, np.float32)
    value = np.array([e.value for e in expanded], dtype=np.float32)

    targets: dict[str, Any] = {
        "candidate_graph": batch["candidate_graph"],
        "num_graphs": batch["num_graphs"],
        "policy": torch.from_numpy(policy),
        "opp_policy": torch.from_numpy(opp),
        "value": torch.from_numpy(value),
    }
    targets["_dropped_policy_mass"] = float(sum(e.dropped_policy_mass for e in expanded))
    targets["_pruned"] = int(n_pruned)
    targets["_kept"] = int(len(expanded))
    return batch, targets
