"""Phase 4 — tactical-candidate injection at expansion (the load-bearing TSS
fix). At a node with an active >=4 threat, the tactical cells (own winning
completions + every opponent >=4-window empty) are force-materialized as root
children, bypassing the policy-nucleus widening cap, so an urgent block/win with
a LOW prior is still searched.

This drives the REAL native MCTS session over the torch GNN evaluator: with the
widening cap pinned to 2, a threat position's root children are exactly the
injected tactical cells (an untrained diffuse policy would otherwise put two
arbitrary high-prior cells there, essentially never the two specific block
cells). The common (no-threat) path is left to the existing mcts suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session


def _inference():
    model = HexgtNetwork(short_term_value_horizons=()).to("cpu").eval()
    return HexgtInference(model, device="cpu", fp16=False)


def _play(coords):
    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def _action_id(q, r):
    return int(eng.action_id(PlacementAction(AxialCoord(q=int(q), r=int(r)))))


def _tight_kwargs(visits=48, vbatch=8):
    # Pin the nucleus cap to 2: without injection only the top-2 priors are ever
    # materialized as root children.
    return dict(
        visits=visits, c_puct=1.5, seed=1, virtual_batch_size=vbatch,
        widening_policy_mass=0.95, widening_max_children=2, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.0,
    )


def test_low_prior_opponent_block_is_injected_and_searched():
    # TEST C: P1 owns a simple four {(0,4),(1,4),(4,4),(5,4)} gaps {(2,4),(3,4)};
    # P0 (defender) to move. The two block cells are the tactical set.
    import hexo_models.hexgt.rust_bridge as rb

    state = _play([(0, 0), (0, 4), (1, 4), (0, 2), (0, -2), (4, 4), (5, 4)])
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    block = {tuple(c) for c in info["tactical_cells"]}
    assert block == {(2, 4), (3, 4)}, f"unexpected tactical set {block}"
    assert info["own_win_now"] is False and info["opp_threat_count"] >= 1

    sess = new_mcts_session(max_states=4096, n=3)
    res = sess.run([0], [state], _inference(), **_tight_kwargs())[0]
    searched = {int(aid) for aid, _visits in res.visit_policy}

    # Both low-prior block cells were materialized and searched despite the cap=2.
    for (q, r) in block:
        assert _action_id(q, r) in searched, f"block cell {(q, r)} not injected/searched"
    # The chosen move is one of the legal blocks (or another legal move) — never
    # illegal — and the search completed normally.
    act = PlacementAction(unpack_coord_id(int(res.action_id)))
    assert act in list(eng.legal_actions(state))
    assert res.visits > 0


def test_own_winning_completion_is_injected_and_searched():
    # TEST G FirstStone: P0 owns a count-4; its empties are an own win-now (B==2)
    # and must be injected as searched root children.
    import hexo_models.hexgt.rust_bridge as rb

    state = _play([(0, 0), (0, 7), (2, 7), (1, 0), (4, 0), (4, 7), (6, 7), (5, 0), (0, -2), (0, 8), (2, 8)])
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert info["own_win_now"] is True
    tactical = {tuple(c) for c in info["tactical_cells"]}
    assert tactical, "expected own winning cells"

    sess = new_mcts_session(max_states=4096, n=3)
    res = sess.run([0], [state], _inference(), **_tight_kwargs())[0]
    searched = {int(aid) for aid, _visits in res.visit_policy}
    for (q, r) in tactical:
        assert _action_id(q, r) in searched, f"own win cell {(q, r)} not injected/searched"


def test_no_threat_search_is_normal_and_capped():
    # A threat-free midgame position: no injection, the nucleus cap of 2 holds, so
    # the root has at most 2 searched children (the common, perf-neutral path).
    import hexo_models.hexgt.rust_bridge as rb

    s = eng.new_game(seed=5)
    for _ in range(9):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    assert rb._hexgt_module().hexgt_threat_analysis(s)["tactical_cells"] == []

    sess = new_mcts_session(max_states=4096, n=3)
    res = sess.run([0], [s], _inference(), **_tight_kwargs())[0]
    searched = {int(aid) for aid, _visits in res.visit_policy}
    assert 0 < len(searched) <= 2, f"no-threat root should respect the cap of 2, got {len(searched)}"
    act = PlacementAction(unpack_coord_id(int(res.action_id)))
    assert act in list(eng.legal_actions(s))
