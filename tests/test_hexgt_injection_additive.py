"""Owner feedback item 1: Phase-4 tactical injection must be ADDITIVE -- the
normal top-prior (nucleus) children are preserved and the tactical cells are
added ON TOP, never displacing top-prior children.

To exercise the case injection EXISTS for -- a LOW-prior urgent block -- this uses
a stub evaluator that assigns deterministically LOW prior to every opponent-threat
block cell (identified by the v3 F_CAND_OPP_THREAT feature) and distinct higher
priors to the rest. So the block (tactical) cells are forced OUT of the nucleus.

Distinguishing assertion: every top-`K` nucleus prior cell is still a materialized
root child. Under the old `cap = max(nucleus, |T|)` bug, the 2 out-of-nucleus
block cells each consumed a nucleus slot (cap stays K=4 -> 2 forced blocks + only 2
top-prior children), so 2 of the top-4 nucleus children are DISPLACED/MISSING.
Under the additive fix (cap = K + |out-of-nucleus tactical| = 6) the full top-4
nucleus is preserved and the 2 blocks are added on top.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_models.hexgt.constants import F_CAND_OPP_THREAT
from hexo_models.hexgt.mcts import new_mcts_session


class _LowPriorBlockEvaluator:
    """Evaluator stub: opponent-threat block cells get a tiny prior; all other
    candidates get distinct descending priors by packed index. Values are 0. This
    is the adversarial low-prior-block case the injection is designed to rescue."""

    def evaluate_featurized_batch(self, payload: dict) -> dict:
        num_graphs = int(payload["num_graphs"])
        tn = int(payload["total_nodes"])
        fd = int(payload["feat_dim"])
        nc = int(payload["total_candidates"])
        node_feat = np.frombuffer(payload["node_feat"], dtype=np.float32).reshape(tn, fd)
        cand_index = np.frombuffer(payload["candidate_index"], dtype=np.int64)
        priors = np.empty(nc, dtype=np.float32)
        for i in range(nc):
            is_block = node_feat[cand_index[i], F_CAND_OPP_THREAT] > 0.0
            priors[i] = 1.0e-4 if is_block else (1.0 / (i + 1.0))  # distinct, all > 1e-4
        values = np.zeros(num_graphs, dtype=np.float32)
        return {
            "values_bytes": np.ascontiguousarray(values).tobytes(),
            "priors_bytes": np.ascontiguousarray(priors).tobytes(),
        }


def _play(coords):
    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def _aid(q, r):
    return int(eng.action_id(PlacementAction(AxialCoord(q=q, r=r))))


def _nucleus_count(priors_desc, mass, min_c, max_c):
    """Exact replica of Rust nucleus_count_values."""
    total = len(priors_desc)
    if total == 0:
        return 0
    lo = min(max(min_c, 1), total)
    hi = min(max(max_c, lo), total)
    if lo >= hi:
        return hi
    cum, count = 0.0, 0
    for p in priors_desc:
        cum += p
        count += 1
        if cum >= mass:
            break
    return max(lo, min(count, hi))


def test_injection_is_additive_preserves_full_nucleus():
    import hexo_models.hexgt.rust_bridge as rb

    # TEST C: P1 owns a simple four; gaps {(2,4),(3,4)}; P0 (defender) to move.
    state = _play([(0, 0), (0, 4), (1, 4), (0, 2), (0, -2), (4, 4), (5, 4)])
    tactical = {tuple(c) for c in rb._hexgt_module().hexgt_threat_analysis(state)["tactical_cells"]}
    assert tactical == {(2, 4), (3, 4)}
    tactical_ids = {_aid(q, r) for (q, r) in tactical}

    K, mass, min_c = 4, 0.95, 2
    kw = dict(
        visits=256, c_puct=1.5, seed=1, virtual_batch_size=8,
        widening_policy_mass=mass, widening_max_children=K, widening_min_children=min_c,
        fpu_reduction=0.2, virtual_loss=1.0, root_policy_temperature=1.0,
    )
    res = new_mcts_session(max_states=1 << 16, n=3).run([0], [state], _LowPriorBlockEvaluator(), **kw)[0]

    prior_pairs = sorted(res.root_prior_policy, key=lambda kv: kv[1], reverse=True)
    nuc = _nucleus_count([p for _, p in prior_pairs], mass, min_c, K)
    nucleus_ids = {aid for aid, _ in prior_pairs[:nuc]}
    children = {int(a) for a, _ in res.visit_policy}

    out_of_nucleus = tactical_ids - nucleus_ids
    assert out_of_nucleus == tactical_ids, "stub should have forced BOTH blocks out of the nucleus"
    # there must be genuine top-prior (non-tactical) nucleus children to preserve
    assert nucleus_ids - tactical_ids, "no non-tactical nucleus child to test displacement against"

    # ADDITIVE (the fix): full top-nucleus preserved...
    missing = nucleus_ids - children
    assert not missing, f"top-prior nucleus children displaced by injection: {missing}"
    # ...AND the low-prior blocks injected on top.
    assert tactical_ids <= children, "low-prior block cells not injected"
    # net: additive union exceeds the cap K (proves cap was lifted, not consumed).
    assert len(children) >= nuc + len(out_of_nucleus)
    assert len(children) > K, f"expected additive lift beyond cap {K}, got {len(children)} children"
