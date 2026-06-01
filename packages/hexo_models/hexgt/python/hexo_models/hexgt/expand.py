"""Expand-time graph construction (the MVP recompute-at-expand path, §7).

Rebuilds the candidate set / window tokens / typed graph + targets per epoch from
the COMPACT raw-fact rows (`dense_cnn.compact_io.read_compact_shard` — the same
representation-agnostic shard format; no new SCHEMA_VERSION). Each row's position
is reconstructed by replaying its placement history through the engine, then the
SHARED Rust builder produces the graph (so training inputs == search inputs).

Targets reuse dense_cnn's finalized facts (value / opp_policy / short-term value
were already constructed per per-placement turn). The only remapping is the
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

    __slots__ = ("graph", "policy", "opp_policy", "value", "stvalue", "stvalue_mask", "dropped_policy_mass")

    def __init__(self, graph, policy, opp_policy, value, stvalue, stvalue_mask, dropped_policy_mass):
        self.graph = graph
        self.policy = policy            # (C,) float32 visit weights over candidates
        self.opp_policy = opp_policy    # (C,) float32
        self.value = value              # scalar in [-1, 1]
        self.stvalue = stvalue          # dict horizon -> scalar
        self.stvalue_mask = stvalue_mask  # dict horizon -> 0/1
        self.dropped_policy_mass = dropped_policy_mass


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


def expand_row_to_graph(row: Any, n: int = DEFAULT_CANDIDATE_RADIUS, horizons: Sequence[int] = ()) -> ExpandedRow:
    """Expand one compact raw-fact row into graph tensors + targets."""

    state = reconstruct_state(row.placement_history)
    facts = rust_bridge.graph_facts(state, n)
    graph = build_graph_tensors(facts)

    policy, dropped = _candidate_targets(graph.candidate_ids, row.policy)
    opp_policy, _ = _candidate_targets(graph.candidate_ids, getattr(row, "opp_policy", ()))

    stv = {int(h): float(v) for h, v in getattr(row, "short_term_value", ())}
    stvalue = {int(h): stv.get(int(h), 0.0) for h in horizons}
    stvalue_mask = {int(h): (1.0 if int(h) in stv else 0.0) for h in horizons}

    return ExpandedRow(
        graph=graph,
        policy=policy,
        opp_policy=opp_policy,
        value=float(row.value),
        stvalue=stvalue,
        stvalue_mask=stvalue_mask,
        dropped_policy_mass=dropped,
    )


def build_training_batch(
    rows: Sequence[Any],
    *,
    n: int = DEFAULT_CANDIDATE_RADIUS,
    horizons: Sequence[int] = (),
    import_torch: bool = True,
) -> tuple[dict, dict]:
    """Expand + collate a list of compact rows into a (batch, targets) pair.

    `batch` is the model forward input; `targets` carries the per-candidate
    `policy`/`opp_policy` and the per-graph `value`/`stvalue_*`(+mask) for
    `hexgt_loss`. Returns torch tensors.
    """

    import torch

    expanded = [expand_row_to_graph(r, n=n, horizons=horizons) for r in rows]
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
    for h in horizons:
        targets[f"stvalue_{h}"] = torch.from_numpy(
            np.array([e.stvalue[int(h)] for e in expanded], dtype=np.float32)
        )
        targets[f"stvalue_{h}_mask"] = torch.from_numpy(
            np.array([e.stvalue_mask[int(h)] for e in expanded], dtype=np.float32)
        )
    targets["_dropped_policy_mass"] = float(sum(e.dropped_policy_mass for e in expanded))
    return batch, targets
