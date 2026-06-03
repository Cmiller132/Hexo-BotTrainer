"""Phase-aware threat-analysis core (Rust threats.rs) — regression fixtures from
the verified harness scripts/_tss_verify*.py. These pin the threat MECHANICS the
TSS injection (Phase 4) and leaf override (Phase 5) depend on:

  * TEST G  — a count-4 is win-now only with two placements left (FirstStone);
              at SecondStone it is NOT.
  * TEST C  — one defender stone neutralises a simple four (hitting set 1).
  * TEST D  — TWO INDEPENDENT simple fours are DEFENSIBLE with 2 stones (the
              refuted "set-cover => loss"): must NOT be called a forced loss.
  * TEST F  — TWO INDEPENDENT OPEN fours ARE a genuine forced loss (control).
  * TEST H  — an open four is 3 threat windows but a 2-cell hitting set defends
              it (">=3 windows = unstoppable" is false).

All positions are built by legal alternating self-play through the real engine,
so owner/phase parity is exactly what the engine enforces.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _eng():
    return importlib.import_module("hexo_engine.api")


def _analyze(state):
    rb = pytest.importorskip(
        "hexo_models.hexgt.rust_bridge", reason="hexgt native module not built"
    )
    return rb._hexgt_module().hexgt_threat_analysis(state)


def _play(coords):
    eng = _eng()
    from hexo_engine.types import AxialCoord, PlacementAction

    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def _phase(state):
    return _eng().to_python_state(state).phase.name


# ---------------------------------------------------------------- TEST G
def test_count4_is_win_now_only_at_first_stone() -> None:
    eng = _eng()
    from hexo_engine.types import AxialCoord, PlacementAction

    coords = [(0, 0), (0, 7), (2, 7), (1, 0), (4, 0), (4, 7), (6, 7), (5, 0), (0, -2), (0, 8), (2, 8)]
    s = _play(coords)
    assert _phase(s) == "FIRST_STONE"
    a = _analyze(s)
    assert a["b"] == 2
    assert a["own_win_now"] is True          # own count-4 + two placements => win-now
    assert a["verdict"] == 1.0
    assert a["forced_loss"] is False
    # the two empties of the count-4 are injected as own winning completions
    assert len(a["tactical_cells"]) >= 2

    # waste the first stone -> SecondStone; the count-4 is no longer win-now (TEST G)
    eng.apply_action(s, PlacementAction(AxialCoord(q=0, r=2)))
    assert _phase(s) == "SECOND_STONE"
    a2 = _analyze(s)
    assert a2["b"] == 1
    assert a2["own_win_now"] is False
    assert a2["verdict"] is None             # no own win, no opp threat
    assert a2["forced_loss"] is False


# ---------------------------------------------------------------- TEST C
def test_simple_opponent_four_is_one_stone_hitting_set() -> None:
    # From scripts/_tss_verify_c.py: P1 owns a simple four; P0 (defender) to move.
    coords = [(0, 0), (0, 4), (1, 4), (0, 2), (0, -2), (4, 4), (5, 4)]
    s = _play(coords)
    a = _analyze(s)
    assert a["opp_threat_count"] >= 1
    assert a["min_hitting_set"] == 1         # one stone in either empty kills it
    assert a["own_win_now"] is False
    assert a["forced_loss"] is False         # must-answer, NOT a forced loss
    assert a["verdict"] is None
    # the block cells are injected
    assert (2, 4) in [tuple(c) for c in a["tactical_cells"]]
    assert (3, 4) in [tuple(c) for c in a["tactical_cells"]]


# ---------------------------------------------------------------- TEST D (headline)
def test_two_independent_simple_fours_are_defensible_not_loss() -> None:
    # From scripts/_tss_verify.py TEST D. P0 owns two independent simple fours;
    # P1 (defender) to move with B=2. Set-cover would call this a loss (4 gap
    # cells > 2); the hitting set is 2 (one stone per four) -> DEFENSIBLE.
    P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0),
          8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    s = _play(flat)
    a = _analyze(s)
    assert a["opp_threat_count"] == 2
    assert a["min_hitting_set"] == 2         # NOT > B=2
    assert a["forced_loss"] is False, "two independent simple fours must NOT be a forced loss"
    assert a["verdict"] is None


# ---------------------------------------------------------------- TEST F (control)
def test_two_independent_open_fours_are_a_forced_loss() -> None:
    # From scripts/_tss_verify.py TEST F. Two independent OPEN fours = 6 threat
    # windows needing 4 distinct blocks > B=2 -> genuine forced loss.
    P0 = {0: (0, 0), 3: (1, 0), 4: (2, 0), 7: (3, 0),
          8: (0, 3), 11: (1, 3), 12: (2, 3), 15: (3, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
          9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    s = _play(flat)
    a = _analyze(s)
    assert a["opp_threat_count"] >= 4        # each open four is 3 overlapping windows
    assert a["min_hitting_set"] == -1        # not hittable within B=2
    assert a["forced_loss"] is True
    assert a["verdict"] == -1.0


# ---------------------------------------------------------------- TEST H
def test_open_four_is_three_windows_but_two_cell_hitting_set() -> None:
    # From scripts/_tss_verify2.py TEST H. P1 open four (3 windows); P0 to move B=2.
    coords = [(0, 0), (0, 6), (1, 6), (0, 2), (0, -2), (2, 6), (3, 6)]
    s = _play(coords)
    a = _analyze(s)
    assert a["opp_threat_count"] == 3        # 3 overlapping count-4 windows
    assert a["min_hitting_set"] == 2         # defended by 2 stones (the ends)
    assert a["forced_loss"] is False, "'>=3 windows = unstoppable' is FALSE"
    assert a["verdict"] is None


def test_no_threats_is_clean() -> None:
    s = _play([(0, 0)])  # opening only
    a = _analyze(s)
    assert a["opp_threat_count"] == 0
    assert a["min_hitting_set"] == 0
    assert a["own_win_now"] is False
    assert a["forced_loss"] is False
    assert a["verdict"] is None
    assert a["tactical_cells"] == []
