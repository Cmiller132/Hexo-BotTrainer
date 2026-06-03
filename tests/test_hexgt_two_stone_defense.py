"""Owner feedback item 3 (and the forced-visit fix it depends on): the search must
FIND multi-placement defenses where NO single stone neutralizes the threat -- a
FirstStone defender (B=2) hitting one of two independent fours per stone, and a
SecondStone defender (B=1) completing a block.

This is the empirical evidence that the FirstStone injection set is NOT too
narrow: injection seeds EVERY opponent >=4-window empty as a child, the forced
first-visit (mcts_tree: RustEdge.forced) ensures each is actually evaluated rather
than starved by FPU under a losing parent value, the SecondStone node re-injects
the remaining block, and the B-aware override never falsely calls the position
lost. Before the forced-visit fix these positions pinned at root_value == -1
(the search never tried the defense); they now resolve well above the loss floor.
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
    torch.manual_seed(0)
    return HexgtInference(HexgtNetwork(short_term_value_horizons=()).to("cpu").eval(), device="cpu", fp16=False)


def _play(flat):
    s = eng.new_game()
    for q, r in flat:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def _search(inf, state, visits):
    kw = dict(
        visits=visits, c_puct=1.5, seed=0, virtual_batch_size=8,
        widening_policy_mass=0.95, widening_max_children=16, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.0, temperature=0.0,
    )
    return new_mcts_session(max_states=1 << 16, n=3).run([0], [state], inf, **kw)[0]


def test_first_stone_two_stone_defense_is_found():
    # TEST D at FirstStone: P0 owns TWO INDEPENDENT simple fours; P1 (B=2) must hit
    # BOTH -- one stone per four. NEITHER stone alone neutralizes both. The search
    # must NOT conclude a loss and must play a block as its first stone.
    import hexo_models.hexgt.rust_bridge as rb

    P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0), 8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7), 9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    state = _play(flat)
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert eng.to_python_state(state).phase.name == "FIRST_STONE"
    assert info["forced_loss"] is False and info["min_hitting_set"] == 2
    blocks = {tuple(c) for c in info["tactical_cells"]}
    assert len(blocks) == 4  # both fours' empties injected -> NOT a narrow set

    res = _search(_inference(), state, visits=320)
    # the defense is FOUND: value is well above the proven-loss floor (was -1.0
    # before the forced-visit fix) ...
    assert res.root_value > -0.85, f"search failed to find the 2-stone defense: {res.root_value}"
    # ... and the played first stone is one of the block cells (a partial cover the
    # SecondStone node then completes).
    sel = unpack_coord_id(int(res.action_id))
    assert (sel.q, sel.r) in blocks, f"first stone {(sel.q, sel.r)} is not a block"


def test_second_stone_single_block_defense_is_found():
    # SecondStone (B=1): P0 has placed its first stone; P1 owns a live four; the one
    # remaining placement must block it. The injected block must be visited (not
    # starved) so the search holds instead of pinning at -1.
    import hexo_models.hexgt.rust_bridge as rb

    state = _play([(0, 0), (0, 4), (1, 4), (0, 2), (0, -2), (4, 4), (5, 4)])
    eng.apply_action(state, PlacementAction(AxialCoord(q=0, r=-3)))  # waste first stone
    assert eng.to_python_state(state).phase.name == "SECOND_STONE"
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert info["forced_loss"] is False and info["opp_threat_count"] >= 1

    res = _search(_inference(), state, visits=96)
    assert res.root_value > -0.85, f"search failed to find the block: {res.root_value}"
    sel = unpack_coord_id(int(res.action_id))
    assert (sel.q, sel.r) in {(2, 4), (3, 4)}, f"did not play a block: {(sel.q, sel.r)}"
