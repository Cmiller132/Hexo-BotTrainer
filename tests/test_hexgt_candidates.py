"""Phase-1 invariant gates for the shared hexgt candidate/graph Rust builder.

These run on engine-generated games (no recorded-data dependency) and assert the
structural guarantees the design relies on:
- candidate_set ⊆ legal moves (never proposes an illegal move);
- the bounded edge construction stays LINEAR in #nodes (no same-axis clique /
  no-explosion gate, §6.3/§4.6): edges-per-node is bounded by a small constant;
- window tokens are only count-3/4/5 active windows (§5);
- candidate ids are unique and deterministically ordered.

The full coverage sweep vs dense_cnn's recorded games lives in
scripts/_validate_hexgt_candidates.py (needs recorded shards).
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _engine():
    return importlib.import_module("hexo_engine.api")


def _bridge():
    pytest.importorskip("hexo_models._rust")
    return importlib.import_module("hexo_models.hexgt.rust_bridge")


def _play_positions(seed: int, n_games: int = 8, max_moves: int = 80):
    """Yield (state, legal_ids) for engine-random self-play positions."""

    eng = _engine()
    rng = random.Random(seed)
    for g in range(n_games):
        state = eng.new_game(seed=seed + g)
        moves = 0
        while eng.terminal(state) is None and moves < max_moves:
            acts = list(eng.legal_actions(state))
            if not acts:
                break
            legal_ids = {eng.action_id(a) for a in acts}
            yield state, legal_ids
            eng.apply_action(state, rng.choice(acts))
            moves += 1


def test_candidate_subset_of_legal_all_radii() -> None:
    bridge = _bridge()
    for state, legal_ids in _play_positions(seed=1):
        for n in (2, 4, 8):
            cand = set(bridge.candidate_ids(state, n))
            assert cand.issubset(legal_ids), f"candidate not subset of legal at n={n}"


def test_edges_linear_no_explosion() -> None:
    """No-explosion gate: edges-per-node bounded by a small constant for all
    positions (no same-axis clique would keep this from growing with board size).
    """
    bridge = _bridge()
    worst_ratio = 0.0
    worst_nodes = 0
    for state, _legal in _play_positions(seed=7, n_games=12, max_moves=120):
        facts = bridge.graph_facts(state, 8)  # widest radius = most candidates
        nc = facts["node_counts"]
        ec = facts["edge_counts"]
        nodes = int(nc["total_nodes"])
        edges = int(ec["total_edges"])
        worst_nodes = max(worst_nodes, nodes)
        if nodes:
            worst_ratio = max(worst_ratio, edges / nodes)
    # Adjacency<=6/node + context(2/node hub) + bounded membership/recency.
    # A same-axis clique would blow this past any constant as boards fill.
    assert worst_ratio < 14.0, f"edges/node={worst_ratio:.1f} suggests super-linear growth"
    assert worst_nodes > 50, "expected to reach large positions to exercise the gate"


def test_window_tokens_are_count_345_only() -> None:
    bridge = _bridge()
    seen_counts = set()
    for state, _legal in _play_positions(seed=3, n_games=10, max_moves=120):
        facts = bridge.graph_facts(state, 2)
        for owner_is_current, count, axis_idx, _aq, _ar, empty in facts["window_tokens"]:
            assert count in (3, 4, 5), f"window token count {count} not in 3/4/5"
            assert axis_idx in (0, 1, 2)
            assert 0 <= empty <= 6
            seen_counts.add(int(count))
    assert seen_counts, "expected to observe some active-window tokens"


def test_candidate_ids_unique_and_sorted() -> None:
    bridge = _bridge()
    for state, _legal in _play_positions(seed=5, n_games=6):
        ids = bridge.candidate_ids(state, 2)
        assert len(ids) == len(set(ids)), "candidate ids must be unique"
        # Packed ids sort like (q, r); the builder emits (q,r)-sorted order.
        assert ids == sorted(ids), "candidate ids must be deterministically ordered"
