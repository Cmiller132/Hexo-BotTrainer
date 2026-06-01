"""Hex D6 group laws + the non-negotiable model D6-equivariance test (§6.5).

hexgt is D6-INVARIANT by construction (invariant node/edge features +
permutation-equivariant message passing/attention). So for a genuinely rotated
position, the per-candidate policy must match the un-rotated policy under the
cell correspondence c -> d(c), for ALL 12 group elements, within fp tolerance.
This is the test that prevents subtly poisoning the model (the dense_cnn D6
lesson). It rebuilds the rotated graph through the SHARED Rust path, so it
exercises Rust build + featurizer + model end-to-end, not just the model.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch():
    return pytest.importorskip("torch")


def _eng():
    return importlib.import_module("hexo_engine.api")


def _d6():
    return importlib.import_module("hexo_models.hexgt.d6")


# --- group laws ---------------------------------------------------------------

def test_d6_group_laws() -> None:
    d6 = _d6()
    probes = [(1, 0), (0, 1), (2, -1), (-3, 2)]
    # identity is element 0
    for q, r in probes:
        assert d6.transform_coord(0, q, r) == (q, r)
    # all 12 elements are distinct permutations on the probe set
    sigs = {tuple(d6.transform_coord(e, q, r) for q, r in probes) for e in range(12)}
    assert len(sigs) == 12
    # closure + inverse
    for a in range(12):
        inv = d6.inverse_index(a)
        for q, r in probes:
            assert d6.transform_coord(inv, *d6.transform_coord(a, q, r)) == (q, r)
        for b in range(12):
            c = d6.compose_index(a, b)  # raises if not closed
            assert 0 <= c < 12


def test_d6_preserves_hex_distance() -> None:
    d6 = _d6()
    pts = [(0, 0), (1, 0), (3, -2), (-4, 1), (2, 2)]
    for e in range(12):
        for (q0, r0) in pts:
            for (q1, r1) in pts:
                a = (q0 - q1, r0 - r1)
                da = max(abs(a[0]), abs(a[1]), abs(-a[0] - a[1]))
                tq0, tr0 = d6.transform_coord(e, q0, r0)
                tq1, tr1 = d6.transform_coord(e, q1, r1)
                b = (tq0 - tq1, tr0 - tr1)
                db = max(abs(b[0]), abs(b[1]), abs(-b[0] - b[1]))
                assert da == db


# --- model equivariance -------------------------------------------------------

def _random_game_coords(seed: int, length: int) -> list[tuple[int, int]]:
    """A legal random game's placement coords (coords[0] is the forced opening)."""

    eng = _eng()
    from hexo_engine.types import unpack_coord_id

    rng = random.Random(seed)
    state = eng.new_game(seed=seed)
    coords: list[tuple[int, int]] = []
    for _ in range(length):
        if eng.terminal(state) is not None:
            break
        acts = list(eng.legal_actions(state))
        if not acts:
            break
        a = rng.choice(acts)
        cid = eng.action_id(a)
        c = unpack_coord_id(int(cid))
        coords.append((int(c.q), int(c.r)))
        eng.apply_action(state, a)
    return coords


def _replay(coords: list[tuple[int, int]]):
    eng = _eng()
    from hexo_engine.types import AxialCoord, PlacementAction

    state = eng.new_game()
    for q, r in coords:
        eng.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return state


def test_model_is_d6_equivariant() -> None:
    torch = _torch()
    d6 = _d6()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.graph_build import batch_from_states

    torch.manual_seed(0)
    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, attention_heads=4, ffn_dim=64,
                         short_term_value_horizons=(1,))
    model.eval()

    # modest positions keep fp accumulation small; n=4 keeps graphs reasonable
    for seed in (11, 23, 37):
        coords = _random_game_coords(seed, length=14)
        if len(coords) < 6:
            continue
        base = _replay(coords)
        base_batch = batch_from_states([base], n=4)
        with torch.no_grad():
            base_out = model(base_batch)
        base_ids = base_batch["candidate_ids"].tolist()
        base_pol = {cid: float(base_out["policy"][i]) for i, cid in enumerate(base_ids)}
        base_val = base_out["value"][0]

        from hexo_engine.types import unpack_coord_id

        for e in range(12):
            rot = [d6.transform_coord(e, q, r) for q, r in coords]
            rstate = _replay(rot)
            rbatch = batch_from_states([rstate], n=4)
            with torch.no_grad():
                rout = model(rbatch)
            rids = rbatch["candidate_ids"].tolist()
            # map rotated candidate id -> its coord
            rpol = {}
            for i, cid in enumerate(rids):
                c = unpack_coord_id(int(cid))
                rpol[(int(c.q), int(c.r))] = float(rout["policy"][i])

            # every base candidate's coord, rotated, must exist in rotated graph
            # with matching policy logit (within fp tolerance)
            for cid, p in base_pol.items():
                c = unpack_coord_id(int(cid))
                rc = d6.transform_coord(e, int(c.q), int(c.r))
                assert rc in rpol, f"element {e}: rotated candidate {rc} missing"
                assert abs(rpol[rc] - p) < 2e-3, (
                    f"element {e} seed {seed}: policy mismatch at {rc}: {rpol[rc]} vs {p}"
                )
            # value is a per-graph invariant
            assert torch.allclose(rout["value"][0], base_val, atol=2e-3), f"value not invariant under element {e}"
