"""End-to-end tests for the native hexgt MCTS session (Phase 5).

Exercises the Rust batched-PUCT search driving the torch GNN evaluator over a
graph-facts payload: legal action selection, the transposition cache, and
subtree reuse across played moves. Runs on CPU (small model, few visits) so it
is fast and GPU-independent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session


def _midgame_state(seed: int, plies: int = 10):
    s = eng.new_game(seed=seed)
    for _ in range(plies):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    return s


def _inference():
    model = HexgtNetwork(short_term_value_horizons=()).to("cpu").eval()
    return HexgtInference(model, device="cpu", fp16=False)


def _search_kwargs(visits, vbatch):
    return dict(
        visits=visits, c_puct=1.5, seed=1, virtual_batch_size=vbatch,
        widening_policy_mass=0.95, widening_max_children=8, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.1,
    )


def test_search_selects_legal_action_and_completes():
    inf = _inference()
    states = [_midgame_state(1), _midgame_state(2)]
    sess = new_mcts_session(max_states=4096, n=3)
    results = sess.run([0, 1], states, inf, **_search_kwargs(visits=16, vbatch=4))
    assert len(results) == 2
    for s, res in zip(states, results):
        act = eng.PlacementAction(unpack_coord_id(res.action_id))
        assert act in list(eng.legal_actions(s)), "MCTS selected an illegal action"
        assert res.visits > 0
        assert -1.0 <= res.root_value <= 1.0
        assert len(res.visit_policy) > 0


def test_evaluator_value_near_zero_untrained():
    # An untrained net should give roughly neutral values (sanity on value sign/range).
    inf = _inference()
    sess = new_mcts_session(max_states=4096, n=3)
    res = sess.run([0], [_midgame_state(7)], inf, **_search_kwargs(visits=16, vbatch=8))
    assert abs(res[0].root_value) < 0.5


def test_eval_cache_accounting_and_persistence():
    inf = _inference()
    sess = new_mcts_session(max_states=1 << 16, n=3)
    state = _midgame_state(3)
    # First search builds the tree + fills the eval cache.
    res1 = sess.run([0], [state], inf, **_search_kwargs(visits=64, vbatch=4))[0]
    ev1 = dict(dict(res1.diagnostics).get("batch", {}).get("evaluation", {}))
    # Cache accounting must balance: every requested leaf is a miss (unique),
    # an in-cache hit, or an in-batch duplicate.
    req = ev1.get("requested_states", 0)
    assert req > 0
    assert ev1.get("unique_states", 0) + ev1.get("cache_hits", 0) + ev1.get("duplicate_hits", 0) == req
    assert ev1.get("cache_size", 0) == ev1.get("unique_states", 0)
    # Search the SAME state under a DIFFERENT game key: key 1 has no promoted
    # tree, so its root is evaluated as a "missing root" — and that root state
    # was cached by the key-0 search, so it MUST be a transposition cache hit.
    res2 = sess.run([1], [state], inf, **_search_kwargs(visits=4, vbatch=4))[0]
    ev2 = dict(dict(res2.diagnostics).get("batch", {}).get("evaluation", {}))
    assert ev2.get("cache_size", 0) >= ev1.get("cache_size", 0)  # cache persists
    assert ev2.get("cache_hits", 0) > 0  # the shared root was served from cache
    act = eng.PlacementAction(unpack_coord_id(res2.action_id))
    assert act in list(eng.legal_actions(state))


def test_candidate_priors_are_legal_normalized():
    # The Rust eval intersects candidates with engine legality and normalizes;
    # the visit policy actions must all be legal placements.
    inf = _inference()
    state = _midgame_state(5)
    sess = new_mcts_session(max_states=4096, n=3)
    res = sess.run([0], [state], inf, **_search_kwargs(visits=16, vbatch=8))[0]
    legal = list(eng.legal_actions(state))
    for action_id, weight in res.visit_policy:
        assert eng.PlacementAction(unpack_coord_id(action_id)) in legal
        assert weight >= 0.0
