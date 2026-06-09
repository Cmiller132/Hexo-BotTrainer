"""hexgnn loss surface: policy + value + opponent-policy ONLY (no STV terms)."""

from __future__ import annotations

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


def _toy():
    torch = _torch()
    # two graphs: 2 and 3 candidates
    cg = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    ctot = cg.numel()
    g = 2
    policy = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    targets = {
        "candidate_graph": cg,
        "num_graphs": g,
        "policy": policy,
        "opp_policy": policy.clone(),
        "value": torch.tensor([0.5, -0.5]),
    }
    return ctot, g, targets


def test_loss_components_are_policy_value_opp_only() -> None:
    torch = _torch()
    from hexgnn.losses import hexgnn_loss

    ctot, g, targets = _toy()
    torch.manual_seed(0)
    outputs = {
        "policy": torch.randn(ctot, requires_grad=True),
        "opp_policy": torch.randn(ctot, requires_grad=True),
        "value": torch.randn(g, 65, requires_grad=True),
    }
    total, comp = hexgnn_loss(outputs, targets)
    assert set(comp.keys()) == {"policy", "value", "opp_policy", "total"}
    assert torch.isfinite(total)
    total.backward()


def test_loss_ignores_any_stvalue_keys() -> None:
    torch = _torch()
    from hexgnn.losses import hexgnn_loss

    ctot, g, targets = _toy()
    # Even if a stray stvalue_* sneaks into BOTH dicts, hexgnn_loss has no STV term.
    targets = {**targets, "stvalue_4": torch.zeros(g, 65), "stvalue_4_mask": torch.ones(g)}
    outputs = {
        "policy": torch.randn(ctot),
        "opp_policy": torch.randn(ctot),
        "value": torch.randn(g, 65),
        "stvalue_4": torch.randn(g, 65),
    }
    total, comp = hexgnn_loss(outputs, targets)
    assert not any(k.startswith("stvalue") for k in comp)
    assert set(comp.keys()) == {"policy", "value", "opp_policy", "total"}


def test_loss_signature_has_no_stv_weight() -> None:
    import inspect
    from hexgnn.losses import hexgnn_loss

    params = set(inspect.signature(hexgnn_loss).parameters)
    assert "short_term_value_weight" not in params
    assert {"policy_weight", "value_weight", "opp_policy_weight"} <= params
