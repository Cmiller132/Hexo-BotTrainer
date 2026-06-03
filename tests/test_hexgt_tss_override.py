"""Phase 5 — phase-aware hitting-set leaf value override. At a fresh leaf the
search backs up a HARD +-1 (skipping the net eval) when the side to move has a
proven 1-ply win, or a proven loss (opponent >=4 windows whose minimum hitting
set exceeds the placements remaining). The MUST-ANSWER case (hittable threat, no
own win) is NOT overridden — Phase-4 injection guarantees the defense is searched.

Driven through the REAL native MCTS + untrained GNN evaluator. An untrained net
gives ~0 values, so a root that converges to ~-1 demonstrates the override is
propagating the proven loss (the direct cure for "value +0.8 right before a
forced loss"); the refuted set-cover position must NOT collapse to a loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction
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


def _kwargs(visits=96, vbatch=8):
    return dict(
        visits=visits, c_puct=1.5, seed=1, virtual_batch_size=vbatch,
        widening_policy_mass=0.95, widening_max_children=16, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.0,
    )


def _root_value(state, visits=96):
    sess = new_mcts_session(max_states=1 << 16, n=3)
    return sess.run([0], [state], _inference(), **_kwargs(visits=visits))[0].root_value


def test_forced_loss_root_value_collapses_negative():
    # TEST F: two independent OPEN fours owned by P0; P1 (side to move) is a
    # genuine 1-ply forced loss. Every line reaches an opponent win-now leaf the
    # override stamps -1, so the root value must converge strongly negative.
    import hexo_models.hexgt.rust_bridge as rb

    P0 = {0: (0, 0), 3: (1, 0), 4: (2, 0), 7: (3, 0),
          8: (0, 3), 11: (1, 3), 12: (2, 3), 15: (3, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    state = _play(flat)
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert info["forced_loss"] is True  # the analysis agrees it's lost

    v = _root_value(state)
    assert v < -0.85, f"forced-loss root value did not collapse negative: {v}"


def test_defensible_two_fours_is_not_a_depth1_loss():
    # TEST D: two INDEPENDENT simple fours; P1 (side to move) can hit both with its
    # 2 placements (hitting set 2 == B). The override must NOT stamp a depth-1 loss
    # here -- this is the refuted set-cover case. The authoritative guard is the
    # verdict itself (a deeper search may still find the position lost in LATER
    # play, which is legitimate and not an override over-fire).
    import hexo_models.hexgt.rust_bridge as rb

    P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0),
          8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    state = _play(flat)
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert info["forced_loss"] is False, "two independent simple fours wrongly called a forced loss"
    assert info["verdict"] is None, "depth-1 override must not fire on a defensible position"
    assert info["min_hitting_set"] == 2  # defended by 2 stones, not > B


def test_override_short_circuits_before_terminal_with_few_visits():
    # The override proves a loss at the leaf where the OPPONENT merely HAS the
    # winning >=4 threat (depth ~1-2), without searching all the way to the 6-in-
    # line terminal. So even with very few visits the forced-loss root value is
    # already strongly negative -- a pure net+terminal search couldn't reach the
    # terminal wins that shallow.
    P0 = {0: (0, 0), 3: (1, 0), 4: (2, 0), 7: (3, 0),
          8: (0, 3), 11: (1, 3), 12: (2, 3), 15: (3, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    state = _play(flat)
    v = _root_value(state, visits=12)
    assert v < -0.85, f"override did not short-circuit the forced loss at low visits: {v}"


def test_no_threat_root_value_is_near_zero_untrained():
    # Sanity: with no threats the override never fires, so an untrained net leaves
    # the root value near neutral (the common path is unchanged).
    s = eng.new_game(seed=7)
    for _ in range(8):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    v = _root_value(s, visits=48)
    assert abs(v) < 0.5, f"no-threat untrained root value not neutral: {v}"
