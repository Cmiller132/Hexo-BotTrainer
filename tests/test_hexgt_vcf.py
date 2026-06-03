"""Phase 6 — exploratory VCF/VCDT forcing-solver prototype (hexgt_vcf_solve).

This solver is NOT wired into search (the plan defers the deep solver); these
tests pin its sanity behavior + that it stays bounded by the node/time caps, so
the benchmark artifact (scripts/_vcf_bench.py) doesn't bit-rot. The solver is
forcing-restricted and therefore NOT a sound prover (it can over-claim a win on a
position refutable by a defender counter-threat) -- see scripts/_vcf_bench.py and
the Phase-6 recommendation; do not treat its `win` as ground truth.
"""

from __future__ import annotations

import pytest

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction


def _vcf():
    rb = pytest.importorskip("hexo_models.hexgt.rust_bridge", reason="hexgt native module not built")
    return rb._hexgt_module().hexgt_vcf_solve


def _play(coords):
    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def test_finds_immediate_forced_win():
    # TEST G FirstStone: side to move owns a count-4 (win-now this turn).
    vcf = _vcf()
    g = _play([(0, 0), (0, 7), (2, 7), (1, 0), (4, 0), (4, 7), (6, 7), (5, 0), (0, -2), (0, 8), (2, 8)])
    r = vcf(g, 8, 2_000_000, 200)
    assert r["win"] is True
    assert r["nodes"] >= 1


def test_no_forced_win_when_side_to_move_has_no_attack():
    # TEST D: P1 to move has no threats of its own -> no forcing win to find.
    vcf = _vcf()
    P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0), 8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3), 16: (0, -2)}
    P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7), 9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
    flat = [None] * 17
    for i, c in {**P0, **P1}.items():
        flat[i] = c
    r = vcf(_play(flat), 8, 2_000_000, 200)
    assert r["win"] is False


def test_opening_has_no_forcing_moves():
    vcf = _vcf()
    r = vcf(_play([(0, 0)]), 8, 2_000_000, 200)
    assert r["win"] is False
    assert r["nodes"] <= 2


def test_respects_node_cap():
    # A tiny node cap must bound the search (and flag hit_limit) rather than run away.
    vcf = _vcf()
    s = eng.new_game(seed=1)
    for _ in range(60):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    r = vcf(s, 16, 50, 200)  # node_cap=50
    assert r["nodes"] <= 51
