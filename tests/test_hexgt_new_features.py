"""Tactical feature-set v2: correctness, invariance helpers, and the zero-init
layer-expansion identical-output guarantee.

Covers the four feature additions (HEXGT feature schema v2):
  - 6 candidate window-count splits ``candidate_{own,opp}_count{3,4,5}_windows``
  - candidate ``distance_to_current_turn_first_stone``
  - side-token ``is_second_placement`` flag

Byte-parity (Rust == Python) for ALL features, including these, is gated by
``test_hexgt_feature_buffer``; D6-equivariance WITH them by ``test_hexgt_d6``.
This file adds (a) value-correctness checks the parity test can't make (parity
only proves the two halves agree, not that they're right) and (b) the zero-init
expansion property that lets the live checkpoint absorb the new features with an
identical first-step output.
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


def _np():
    return pytest.importorskip("numpy")


def _torch():
    return pytest.importorskip("torch")


def _eng():
    return importlib.import_module("hexo_engine.api")


def _hexdist(q0, r0, q1, r1):
    dq, dr = q0 - q1, r0 - r1
    return max(abs(dq), abs(dr), abs(-dq - dr))


# --- zero-init layer-expansion (pure torch; no native build needed) -----------

def test_zero_init_expansion_is_identical_output() -> None:
    """A model trained with the new slots ALWAYS ZERO (old checkpoint) must, after
    zeroing the new node_in columns, produce IDENTICAL output once those slots
    carry real nonzero values — the property that lets the live run absorb the
    features without a cold start."""

    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork, zero_init_expanded_feature_columns
    from hexo_models.hexgt.collate import collate_graphs
    from hexo_models.hexgt.constants import NEW_FEATURE_SLOTS_V2
    from hexo_models.hexgt.graph_build import graph_tensors_from_state

    eng = _eng()
    import random

    rng = random.Random(5)
    s = eng.new_game(seed=5)
    for _ in range(24):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        if not acts:
            break
        eng.apply_action(s, rng.choice(acts))
    graph = graph_tensors_from_state(s, 3)
    batch = collate_graphs([graph])

    torch.manual_seed(0)
    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, attention_heads=4, ffn_dim=64)
    model.eval()

    # "old data": new slots are zero (as in every pre-v2 sample).
    batch_old = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in batch.items()}
    batch_old["node_feat"][:, list(NEW_FEATURE_SLOTS_V2)] = 0.0
    with torch.no_grad():
        out_old = model(batch_old)

    # "new data": fill the new slots with arbitrary nonzero values.
    batch_new = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in batch.items()}
    for j, slot in enumerate(NEW_FEATURE_SLOTS_V2):
        batch_new["node_feat"][:, slot] = 0.1 * (j + 1)
    with torch.no_grad():
        out_pert = model(batch_new)

    # before zeroing, the random columns make the new features perturb the output
    assert not torch.allclose(out_pert["policy"], out_old["policy"], atol=1e-6), (
        "new feature columns were already ~0 — test cannot prove the expansion"
    )

    # zero-init the new columns, then the SAME nonzero-feature batch must match old
    zeroed = zero_init_expanded_feature_columns(model)
    assert sorted(zeroed) == sorted(NEW_FEATURE_SLOTS_V2)
    with torch.no_grad():
        out_new = model(batch_new)
    assert torch.allclose(out_new["policy"], out_old["policy"], atol=1e-6), "policy not identical after zero-init"
    assert torch.allclose(out_new["value"], out_old["value"], atol=1e-6), "value not identical after zero-init"


def test_zero_init_only_touches_new_columns() -> None:
    """Zeroing must leave every trained (pre-v2) input column untouched."""

    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork, zero_init_expanded_feature_columns
    from hexo_models.hexgt.constants import NEW_FEATURE_SLOTS_V2, NODE_FEATURE_DIM

    torch.manual_seed(1)
    model = HexgtNetwork(token_dim=32, gnn_layers=1, ctx_layers=1, attention_heads=4, ffn_dim=48)
    before = model.node_in[0].weight.detach().clone()
    zero_init_expanded_feature_columns(model)
    after = model.node_in[0].weight.detach()
    kept = [c for c in range(NODE_FEATURE_DIM) if c not in NEW_FEATURE_SLOTS_V2]
    assert torch.equal(before[:, kept], after[:, kept]), "a trained column was modified"
    assert torch.count_nonzero(after[:, list(NEW_FEATURE_SLOTS_V2)]) == 0, "new columns not zeroed"


# --- feature value-correctness (needs the native build) -----------------------

def _facts_for(state, n=3):
    rb = pytest.importorskip(
        "hexo_models.hexgt.rust_bridge",
        reason="hexgt native module not built",
    )
    return rb.graph_facts(state, n)


def test_window_count_splits_sum_to_nwin() -> None:
    """The three own count-{3,4,5} bins must sum to NWIN_OWN per candidate (same
    for opp) — they are exactly its per-count breakdown."""

    np = _np()
    from hexo_models.hexgt.constants import (
        F_CAND_NWIN_OPP, F_CAND_NWIN_OWN,
        F_CAND_OPP_WIN3, F_CAND_OPP_WIN4, F_CAND_OPP_WIN5,
        F_CAND_OWN_WIN3, F_CAND_OWN_WIN4, F_CAND_OWN_WIN5,
        NODE_TYPE_CANDIDATE,
    )
    from hexo_models.hexgt.features import build_graph_tensors

    eng = _eng()
    import random

    rng = random.Random(17)
    saw_window = False
    for trial in range(8):
        s = eng.new_game(seed=100 + trial)
        for _ in range(rng.randint(20, 70)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is not None:
            continue
        g = build_graph_tensors(_facts_for(s))
        feat = g.node_feat
        cand = g.node_type == NODE_TYPE_CANDIDATE
        own_sum = feat[:, F_CAND_OWN_WIN3] + feat[:, F_CAND_OWN_WIN4] + feat[:, F_CAND_OWN_WIN5]
        opp_sum = feat[:, F_CAND_OPP_WIN3] + feat[:, F_CAND_OPP_WIN4] + feat[:, F_CAND_OPP_WIN5]
        assert np.allclose(own_sum[cand], feat[cand, F_CAND_NWIN_OWN], atol=1e-6)
        assert np.allclose(opp_sum[cand], feat[cand, F_CAND_NWIN_OPP], atol=1e-6)
        if feat[cand, F_CAND_NWIN_OWN].sum() + feat[cand, F_CAND_NWIN_OPP].sum() > 0:
            saw_window = True
    assert saw_window, "no active-window candidates exercised — test vacuous"


def test_distance_to_first_stone_on_second_placement() -> None:
    """On a SECOND-stone position, every candidate's F_CAND_DIST_FIRST equals the
    hex-distance to this turn's first stone / COORD_SCALE; on a FIRST placement it
    is zero."""

    np = _np()
    from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
    from hexo_models.hexgt.constants import (
        COORD_SCALE, F_CAND_DIST_FIRST, NODE_TYPE_CANDIDATE,
    )
    from hexo_models.hexgt.features import build_graph_tensors

    eng = _eng()

    # Opening (P0 @ origin) -> P1 first stone -> now P1 is on its SECOND placement,
    # with first stone == the just-placed coord.
    s = eng.new_game()
    eng.apply_action(s, PlacementAction(AxialCoord(q=0, r=0)))
    first = (4, -2)
    eng.apply_action(s, PlacementAction(AxialCoord(q=first[0], r=first[1])))

    g = build_graph_tensors(_facts_for(s))
    feat = g.node_feat
    assert g.candidate_index.shape[0] > 0
    # candidate_index (node rows) is aligned 1:1 with candidate_ids (action ids).
    for node_idx, cid in zip(g.candidate_index.tolist(), g.candidate_ids.tolist()):
        c = unpack_coord_id(int(cid))
        expected = _hexdist(int(c.q), int(c.r), first[0], first[1]) / COORD_SCALE
        assert abs(float(feat[node_idx, F_CAND_DIST_FIRST]) - expected) < 1e-6, (
            f"distance mismatch at candidate {(c.q, c.r)}"
        )


def test_side_is_second_flag() -> None:
    """The side-token flag is 1 exactly on the turn's second placement."""

    np = _np()
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexo_models.hexgt.constants import F_SIDE_IS_SECOND, NODE_TYPE_SIDE
    from hexo_models.hexgt.features import build_graph_tensors

    eng = _eng()

    def side_flag(state):
        g = build_graph_tensors(_facts_for(state))
        side = np.nonzero(g.node_type == NODE_TYPE_SIDE)[0]
        return float(g.node_feat[side[0], F_SIDE_IS_SECOND])

    s = eng.new_game()
    eng.apply_action(s, PlacementAction(AxialCoord(q=0, r=0)))  # opening done -> P1 FirstStone
    assert side_flag(s) == 0.0, "first placement should have is_second=0"
    eng.apply_action(s, PlacementAction(AxialCoord(q=3, r=0)))  # P1 first stone -> now SecondStone
    assert side_flag(s) == 1.0, "second placement should have is_second=1"
    eng.apply_action(s, PlacementAction(AxialCoord(q=-3, r=1)))  # completes turn -> P0 FirstStone
    assert side_flag(s) == 0.0, "back to a first placement"
