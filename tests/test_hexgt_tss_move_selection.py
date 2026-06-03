"""Owner feedback item 2: tactical-aware final move selection. The visit-policy
sampler (temperature / Dirichlet) must never PLAY a move the search has proven
losing when a non-losing move exists, and must play a proven win. The 1-ply guard
masks the selection weights using the same phase-aware threat analysis as the leaf
override.

Driven through the real native MCTS + untrained GNN, sampling across many seeds at
HIGH temperature (where, without the guard, forced-playout visits on a losing move
would be sampled with nonzero probability).
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


def _play(coords):
    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def _aid(q, r):
    return int(eng.action_id(PlacementAction(AxialCoord(q=q, r=r))))


def _phase(state):
    return eng.to_python_state(state).phase.name


def _run_once(inf, state, seed, temperature):
    kw = dict(
        visits=96, c_puct=1.5, seed=seed, virtual_batch_size=8,
        widening_policy_mass=0.95, widening_max_children=16, widening_min_children=2,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.0,
        temperature=temperature,
    )
    return new_mcts_session(max_states=1 << 16, n=3).run([0], [state], inf, **kw)[0]


def test_proven_win_is_always_played():
    # TEST G FirstStone: P0 owns a count-4 (a proven win-now). No opponent threat,
    # so the winning completion cells are the only proven wins; the guard must play
    # one of them at EVERY temperature/seed, never a non-winning move.
    import hexo_models.hexgt.rust_bridge as rb

    state = _play([(0, 0), (0, 7), (2, 7), (1, 0), (4, 0), (4, 7), (6, 7), (5, 0), (0, -2), (0, 8), (2, 8)])
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert info["own_win_now"] is True
    win_ids = {_aid(q, r) for (q, r) in {tuple(c) for c in info["tactical_cells"]}}

    inf = _inference()
    for seed in range(40):
        res = _run_once(inf, state, seed=seed, temperature=1.5)
        assert int(res.action_id) in win_ids, (
            f"seed {seed}: played non-winning move {res.action_id} when a proven win existed"
        )


def test_proven_losing_move_is_never_selected():
    # Construct a SecondStone position where P0 has ONE placement left and P1 owns a
    # live four: the only safe moves are the two blocks; every other move hands P1
    # the win next turn (a proven 1-ply loss). The guard must never play a loser.
    import hexo_models.hexgt.rust_bridge as rb

    # TEST C (P0 FirstStone, P1 four), then P0 wastes its first stone -> SecondStone.
    state = _play([(0, 0), (0, 4), (1, 4), (0, 2), (0, -2), (4, 4), (5, 4)])
    eng.apply_action(state, PlacementAction(AxialCoord(q=0, r=-3)))  # non-defensive first stone
    assert _phase(state) == "SECOND_STONE"
    info = rb._hexgt_module().hexgt_threat_analysis(state)
    assert info["own_win_now"] is False and info["opp_threat_count"] >= 1
    safe_ids = {_aid(2, 4), _aid(3, 4)}  # the only neutralizing (safe) cells

    inf = _inference()
    chosen = set()
    for seed in range(60):
        res = _run_once(inf, state, seed=seed, temperature=1.5)
        a = int(res.action_id)
        assert a in safe_ids, f"seed {seed}: played a proven-losing move {a} (safe set {safe_ids})"
        chosen.add(a)
    # sanity: selection actually happened (and could pick either block)
    assert chosen <= safe_ids and chosen


def test_quiet_position_selection_unchanged():
    # No threat / no own win: the guard is a no-op and a legal move is still played.
    inf = _inference()
    s = eng.new_game(seed=5)
    for _ in range(8):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    res = _run_once(inf, s, seed=1, temperature=1.0)
    act = PlacementAction(unpack_coord_id(int(res.action_id)))
    assert act in list(eng.legal_actions(s))
