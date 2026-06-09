"""Threat-Space Search for the dense_cnn (Model 1) lineage — full parity with the
hexgt TSS, ported onto dense_cnn's fixed 41x41 crop. Covers all four components:

  1. shared threat index + analysis (parity vs hexgt; hitting-set for small B),
  2. tactical INJECTION at expansion (forced low-prior block/win children),
  3. phase-aware hitting-set leaf value OVERRIDE (proven +-1 leaves),
  4. tactical move-selection GUARD (play a proven win, never a proven loss),

plus the dense_cnn-specific concerns the feasibility report flags:
  * crop coverage — every n=8 tactical cell is in-crop in normal play,
  * the training policy (exported visit policy) is left UNTOUCHED by the guard.

Driven through the REAL native dense_cnn MCTS over an untrained CPU CNN evaluator
(CPU-only by construction; no GPU contention with any live run). The threat
analysis itself comes from `crate::threats_shared`, the single module both
lineages share, so `dense_cnn_threat_analysis` and `hexgt_threat_analysis` are
equal by construction — `test_threat_analysis_parity_*` proves it on live states.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id

import hexo_models._rust as _rust
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.constants import INPUT_CHANNELS
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session


# --- helpers -----------------------------------------------------------------

def _inference():
    torch.manual_seed(0)
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=16, blocks=2).eval()
    return DenseCNNInference(net, device="cpu", amp=False, optimize_for_inference=False)


def _play(coords):
    s = eng.new_game()
    for q, r in coords:
        if q is None:
            continue
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def _aid(q, r):
    return int(eng.action_id(PlacementAction(AxialCoord(q=int(q), r=int(r)))))


def _phase(state):
    return eng.to_python_state(state).phase.name


def _dc_analysis(state):
    return _rust.dense_cnn.dense_cnn_threat_analysis(state)


def _hg_analysis(state):
    return _rust.hexgt.hexgt_threat_analysis(state)


_SHARED_KEYS = (
    "b",
    "own_win_now",
    "opp_threat_count",
    "min_hitting_set",
    "forced_loss",
    "verdict",
    "tactical_cells",
)


def _kwargs(*, visits=96, temperature=1.0, cap=16, seed=1):
    return dict(
        visits=visits,
        c_puct=1.5,
        seed=seed,
        temperature=temperature,
        virtual_batch_size=8,
        widening_policy_mass=0.95,
        widening_max_children=cap,
        widening_min_children=2,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        root_policy_temperature=1.0,
    )


def _run(inf, state, **over):
    kw = _kwargs(**over)
    return new_mcts_session(max_states=1 << 16).run([0], [state], inf, **kw)[0]


# Engine-level fixtures (lineage-agnostic positions), shared with the hexgt suite.

def _forced_loss_state():
    # Two independent OPEN fours owned by P0; P1 (side to move) is a genuine 1-ply
    # forced loss (every line reaches an opponent win-now leaf).
    P0 = {0: (0, 0), 3: (1, 0), 4: (2, 0), 7: (3, 0),
          8: (0, 3), 11: (1, 3), 12: (2, 3), 15: (3, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    return _play(flat)


def _defensible_two_fours_state():
    # Two INDEPENDENT simple fours; P1 (side to move) can hit both with its 2
    # placements (hitting set 2 == B). Not a depth-1 loss.
    P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0),
          8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    return _play(flat)


def _opp_four_block_state():
    # P1 owns a simple four {(0,4),(1,4),(4,4),(5,4)} gaps {(2,4),(3,4)}; P0
    # (defender) to move. The two block cells are the tactical set.
    return _play([(0, 0), (0, 4), (1, 4), (0, 2), (0, -2), (4, 4), (5, 4)])


def _own_win_now_state():
    # FirstStone: P0 owns a count-4; its empties are an own win-now (B == 2).
    return _play([(0, 0), (0, 7), (2, 7), (1, 0), (4, 0), (4, 7), (6, 7), (5, 0), (0, -2), (0, 8), (2, 8)])


def _quiet_state():
    s = eng.new_game(seed=5)
    for _ in range(8):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    return s


# --- 1. shared threat analysis: parity + hitting-set correctness -------------

@pytest.mark.parametrize(
    "factory",
    [_forced_loss_state, _defensible_two_fours_state, _opp_four_block_state,
     _own_win_now_state, _quiet_state],
    ids=["forced_loss", "defensible_two_fours", "opp_four_block", "own_win_now", "quiet"],
)
def test_threat_analysis_parity_with_hexgt(factory):
    # dense_cnn and hexgt funnel through crate::threats_shared, so the shared
    # analysis dict must be identical on every live state.
    state = factory()
    dca = _dc_analysis(state)
    hga = _hg_analysis(state)
    for key in _SHARED_KEYS:
        assert dca[key] == hga[key], f"{key} differs: dense_cnn={dca[key]} hexgt={hga[key]}"


def test_hitting_set_forced_loss_exceeds_budget():
    # Two independent OPEN fours owned by the opponent generate 6 threat windows;
    # with B == 2 placements the minimum hitting set still exceeds the budget, so
    # the side to move is a 1-ply forced loss (min_hitting_set encoded as -1=None).
    info = _dc_analysis(_forced_loss_state())
    assert info["b"] == 2
    assert info["own_win_now"] is False
    assert info["opp_threat_count"] == 6
    assert info["min_hitting_set"] == -1  # None encoded as -1: needs more than B
    assert info["forced_loss"] is True
    assert info["verdict"] == -1.0


def test_hitting_set_two_fours_defensible_is_two():
    # Two INDEPENDENT simple fours, B == 2 (FirstStone): each is blockable by one
    # stone, total 2 == B, so the exact small-B hitting set is 2 and the position is
    # defensible — the k=2 pair-scan must return 2 (not None), and the override must
    # NOT fire (verdict None: a must-answer, not a proof).
    info = _dc_analysis(_defensible_two_fours_state())
    assert info["b"] == 2
    assert info["own_win_now"] is False
    assert info["opp_threat_count"] == 2
    assert info["min_hitting_set"] == 2
    assert info["forced_loss"] is False
    assert info["verdict"] is None


def test_hitting_set_single_block_is_one():
    # One opponent four with gap cells {(2,4),(3,4)} sharing the same window: a
    # single stone in either gap kills it -> min hitting set == 1, defensible.
    info = _dc_analysis(_opp_four_block_state())
    assert info["own_win_now"] is False
    assert info["opp_threat_count"] >= 1
    assert info["min_hitting_set"] == 1
    assert info["forced_loss"] is False
    assert info["verdict"] is None  # must-answer, not a proof either way


def test_own_win_now_is_a_proven_win():
    info = _dc_analysis(_own_win_now_state())
    assert info["own_win_now"] is True
    assert info["verdict"] == 1.0


# --- crop coverage (feasibility report item 4) -------------------------------

@pytest.mark.parametrize(
    "factory",
    [_forced_loss_state, _defensible_two_fours_state, _opp_four_block_state, _own_win_now_state],
    ids=["forced_loss", "defensible_two_fours", "opp_four_block", "own_win_now"],
)
def test_all_tactical_cells_are_in_crop(factory):
    # Threat cells live within a length-6 window of >=4 stones (hex distance <= 5),
    # well inside both the n=8 placement range and the 41x41 crop centered on the
    # occupied-stone centroid. So in normal play EVERY tactical cell is injectable;
    # the out-of-crop residual must be empty.
    info = _dc_analysis(factory())
    tactical = {tuple(c) for c in info["tactical_cells"]}
    in_crop = {tuple(c) for c in info["tactical_cells_in_crop"]}
    out_of_crop = {tuple(c) for c in info["tactical_cells_out_of_crop"]}
    assert tactical, "fixture should carry tactical cells"
    assert out_of_crop == set(), f"unexpected out-of-crop tactical cells {out_of_crop}"
    assert in_crop == tactical


# --- 2. tactical injection at expansion --------------------------------------

def test_low_prior_opponent_block_is_injected_and_searched():
    # Pin the nucleus cap to 2; without injection only the top-2 priors would ever
    # be materialized as root children. The two (low-prior) block cells must still
    # be force-materialized and searched.
    state = _opp_four_block_state()
    info = _dc_analysis(state)
    block = {tuple(c) for c in info["tactical_cells"]}
    assert block == {(2, 4), (3, 4)}, f"unexpected tactical set {block}"

    inf = _inference()
    res = _run(inf, state, cap=2, visits=64)
    searched = {int(aid) for aid, _ in res.visit_policy}
    for (q, r) in block:
        assert _aid(q, r) in searched, f"block cell {(q, r)} not injected/searched under cap=2"
    act = PlacementAction(unpack_coord_id(int(res.action_id)))
    assert act in list(eng.legal_actions(state))
    assert res.visits > 0


def test_own_winning_completion_is_injected_and_found():
    state = _own_win_now_state()
    info = _dc_analysis(state)
    assert info["own_win_now"] is True
    win_ids = {_aid(q, r) for (q, r) in {tuple(c) for c in info["tactical_cells"]}}
    assert win_ids

    inf = _inference()
    res = _run(inf, state, cap=2, visits=64)
    assert int(res.action_id) in win_ids, "search did not pick an injected winning cell"
    assert res.root_value > 0.85, f"own win not resolved to a win: {res.root_value}"


def test_no_threat_search_is_normal_and_capped():
    # Threat-free position: no injection, the nucleus cap of 2 holds.
    state = _quiet_state()
    assert _dc_analysis(state)["tactical_cells"] == []
    inf = _inference()
    res = _run(inf, state, cap=2, visits=64)
    searched = {int(aid) for aid, _ in res.visit_policy}
    assert 0 < len(searched) <= 2, f"no-threat root should respect cap=2, got {len(searched)}"
    act = PlacementAction(unpack_coord_id(int(res.action_id)))
    assert act in list(eng.legal_actions(state))


# --- 3. phase-aware hitting-set leaf override --------------------------------

def test_forced_loss_root_value_collapses_negative():
    res = _run(_inference(), _forced_loss_state(), visits=96)
    assert res.root_value < -0.85, f"forced-loss root value did not collapse: {res.root_value}"


def test_override_short_circuits_at_low_visits():
    # The override proves the loss where the opponent merely HAS the winning >=4
    # threat (depth ~1-2), so even very few visits already collapse the root.
    res = _run(_inference(), _forced_loss_state(), visits=12)
    assert res.root_value < -0.85, f"override did not short-circuit at low visits: {res.root_value}"


def test_no_threat_root_value_is_near_zero_untrained():
    # Common path unchanged: with no threats the override never fires, so an
    # untrained net leaves the root value near neutral.
    res = _run(_inference(), _quiet_state(), visits=48)
    assert abs(res.root_value) < 0.5, f"no-threat untrained root value not neutral: {res.root_value}"


# --- 4. tactical move-selection guard ----------------------------------------

def test_proven_win_is_always_played():
    state = _own_win_now_state()
    info = _dc_analysis(state)
    assert info["own_win_now"] is True
    win_ids = {_aid(q, r) for (q, r) in {tuple(c) for c in info["tactical_cells"]}}
    inf = _inference()
    for seed in range(24):
        res = _run(inf, state, temperature=1.5, seed=seed)
        assert int(res.action_id) in win_ids, (
            f"seed {seed}: played non-winning move {res.action_id} when a proven win existed"
        )


def test_proven_losing_move_is_never_selected():
    # SecondStone: P0 has ONE placement left and P1 owns a live four; the only safe
    # moves are the two blocks. Every other move is a proven 1-ply loss.
    state = _opp_four_block_state()
    eng.apply_action(state, PlacementAction(AxialCoord(q=0, r=-3)))  # waste first stone
    assert _phase(state) == "SECOND_STONE"
    info = _dc_analysis(state)
    assert info["own_win_now"] is False and info["opp_threat_count"] >= 1
    safe_ids = {_aid(2, 4), _aid(3, 4)}
    inf = _inference()
    chosen = set()
    for seed in range(40):
        res = _run(inf, state, temperature=1.5, seed=seed)
        a = int(res.action_id)
        assert a in safe_ids, f"seed {seed}: played proven-losing move {a} (safe {safe_ids})"
        chosen.add(a)
    assert chosen and chosen <= safe_ids


def test_quiet_position_selection_unchanged():
    res = _run(_inference(), _quiet_state(), seed=1, temperature=1.0)
    act = PlacementAction(unpack_coord_id(int(res.action_id)))
    assert act in list(eng.legal_actions(_quiet_state()))


# --- training policy is UNTOUCHED by the guard -------------------------------

def test_guard_does_not_collapse_training_policy():
    # The guard masks the PLAYED move only; the exported visit policy (the training
    # target) must still describe what the search actually explored. In the proven-
    # loss position the played action is forced into the safe set, yet the exported
    # policy must still contain visited NON-safe moves (proof the export was not
    # masked / collapsed onto the guarded selection).
    state = _opp_four_block_state()
    eng.apply_action(state, PlacementAction(AxialCoord(q=0, r=-3)))
    assert _phase(state) == "SECOND_STONE"
    safe_ids = {_aid(2, 4), _aid(3, 4)}
    res = _run(_inference(), state, temperature=1.5, seed=3)
    export_ids = {int(aid) for aid, w in res.visit_policy if w > 0.0}
    assert int(res.action_id) in safe_ids, "guard must still force a safe played move"
    assert export_ids - safe_ids, (
        "exported training policy was collapsed onto the safe set — the guard must "
        "not touch the training target"
    )
