"""Live-state -> packed-graph batch glue.

Ties the shared Rust builder (`rust_bridge.graph_facts`), the featurizer
(`features.build_graph_tensors`), and the collator (`collate.collate_graphs`)
into one path — guaranteeing training inputs == search inputs. Used by tests and
`inference.evaluate_states` (the Python-driven eval path). Live MCTS does NOT
come through here: it featurizes leaf states entirely in Rust
(`rust/src/features.rs::featurize_collate_states`) and hands the collated
payload straight to `inference.evaluate_featurized_batch`.
"""

from __future__ import annotations

from typing import Sequence

from .collate import collate_graphs
from .constants import DEFAULT_CANDIDATE_RADIUS
from .features import GraphTensors, build_graph_tensors
from . import rust_bridge


def graph_tensors_from_state(state: object, n: int = DEFAULT_CANDIDATE_RADIUS) -> GraphTensors:
    """Build one position's model tensors from a live engine state."""

    facts = rust_bridge.graph_facts(state, n)
    return build_graph_tensors(facts)


def batch_from_states(states: Sequence[object], n: int = DEFAULT_CANDIDATE_RADIUS) -> dict:
    """Build a packed-graph batch from a sequence of live engine states."""

    graphs = [graph_tensors_from_state(s, n) for s in states]
    return collate_graphs(graphs)
