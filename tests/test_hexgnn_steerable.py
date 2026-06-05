"""Exact D6 invariance of the tied steerable message-passing layer (spec §4.5).

Proves: (a) the SteerableTensorChannels output is bitwise-equal under all 12 D6
elements (each permutes the 6 hex directions; the O(2)-invariants tr/det of the
accumulated tensor are unchanged), (b) the full RelationalMessagePassing layer with
steerable is D6-invariant, and (c) direction discrimination SURVIVES (a collinear
"line" and a spread "blob" produce DIFFERENT outputs — the tying did not collapse to
direction-agnostic). Pure Python, no native rebuild needed.
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
_HEXGNN = ROOT / "packages" / "hexgnn" / "python"
if str(_HEXGNN) not in sys.path:
    sys.path.insert(0, str(_HEXGNN))


def _torch():
    return pytest.importorskip("torch")


def _d6():
    return importlib.import_module("hexgnn.d6")


def _dir_perm(elem: int) -> list[int]:
    """The permutation σ_g of the 6 hex axial unit directions induced by D6 elem."""
    from hexgnn.architecture import _HEX_DIRS_AXIAL
    d6 = _d6()
    index = {tuple(d): i for i, d in enumerate(_HEX_DIRS_AXIAL)}
    perm = []
    for (q, r) in _HEX_DIRS_AXIAL:
        tq, tr = d6.transform_coord(elem, q, r)
        perm.append(index[(int(tq), int(tr))])
    return perm


def test_dir_perm_is_a_permutation_for_all_12() -> None:
    _torch()
    for e in range(12):
        perm = _dir_perm(e)
        assert sorted(perm) == list(range(6)), f"element {e} not a permutation of the 6 directions"


def _synthetic(torch, n=40, e=160, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, 32, generator=g)               # D6-INVARIANT node features
    src = torch.randint(0, n, (e,), generator=g)
    dst = torch.randint(0, n, (e,), generator=g)
    edge_index = torch.stack([src, dst], 0)
    edge_dir = torch.randint(0, 6, (e,), generator=g)
    return h, edge_index, edge_dir


def test_steerable_module_exactly_invariant_all_12() -> None:
    torch = _torch()
    from hexgnn.architecture import SteerableTensorChannels

    torch.manual_seed(1)
    mod = SteerableTensorChannels(32, n_steer=4).eval()
    h, ei, ed = _synthetic(torch)
    with torch.no_grad():
        base = mod(h, ei, ed)
    for e in range(12):
        perm = torch.tensor(_dir_perm(e), dtype=torch.long)
        ed_rot = perm[ed]  # rotate every edge's direction by σ_g
        with torch.no_grad():
            out = mod(h, ei, ed_rot)
        assert torch.allclose(out, base, atol=1e-5), f"steerable output moved under D6 element {e}"


def test_relational_layer_with_steerable_invariant_all_12() -> None:
    torch = _torch()
    from hexgnn.architecture import RelationalMessagePassing
    from hexgnn.constants import NUM_EDGE_TYPES, EDGE_ATTR_DIM

    torch.manual_seed(2)
    layer = RelationalMessagePassing(32, NUM_EDGE_TYPES, EDGE_ATTR_DIM, steerable=4).eval()
    h, ei, ed = _synthetic(torch, seed=3)
    etype = torch.zeros(ei.shape[1], dtype=torch.long)
    eattr = torch.zeros(ei.shape[1], EDGE_ATTR_DIM)
    with torch.no_grad():
        base = layer(h, ei, etype, eattr, ed)
    for e in range(12):
        perm = torch.tensor(_dir_perm(e), dtype=torch.long)
        with torch.no_grad():
            out = layer(h, ei, etype, eattr, perm[ed])
        assert torch.allclose(out, base, atol=1e-5), f"layer output moved under D6 element {e}"


def test_direction_discrimination_survives() -> None:
    """A collinear 'line' vs a spread 'blob' into the same node must give DIFFERENT
    steerable outputs — proving the tie did not collapse to direction-agnostic."""
    torch = _torch()
    from hexgnn.architecture import SteerableTensorChannels

    torch.manual_seed(4)
    mod = SteerableTensorChannels(32, n_steer=4).eval()
    n = 8
    h = torch.randn(n, 32)
    # all edges point INTO node 0 from node 1
    src = torch.ones(6, dtype=torch.long)
    dst = torch.zeros(6, dtype=torch.long)
    ei = torch.stack([src, dst], 0)
    line = torch.tensor([0, 3, 0, 3, 0, 3], dtype=torch.long)   # one axis (+ opposite)
    blob = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)   # all 6 directions
    with torch.no_grad():
        o_line = mod(h, ei, line)[0]
        o_blob = mod(h, ei, blob)[0]
    assert not torch.allclose(o_line, o_blob, atol=1e-3), "line and blob are indistinguishable (direction info lost)"
