"""hexgnn model gates: param budget (vs model-3), real-graph forward (policy/
value/opp only — NO stvalue), overfit a tiny batch, and the no-transformer/no-STV
structural guarantees that define this lineage."""

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
_HEXGNN = ROOT / "packages" / "hexgnn" / "python"
if str(_HEXGNN) not in sys.path:
    sys.path.insert(0, str(_HEXGNN))


# model-3 (the live hexgt lineage) parameter count, for the reduction assertion.
MODEL3_PARAMS = 2_584_774


def _torch():
    return pytest.importorskip("torch")


def _eng():
    return importlib.import_module("hexo_engine.api")


def _states(n_states: int, n_radius: int = 4, seed: int = 0):
    eng = _eng()
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n_states and tries < 4000:
        tries += 1
        s = eng.new_game(seed=rng.randint(0, 10**6))
        for _ in range(rng.randint(4, 30)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def test_default_param_count_and_reduction_vs_model3() -> None:
    _torch()
    from hexgnn.architecture import HexgnnNetwork

    model = HexgnnNetwork()  # defaults: token_dim 168, gnn_layers 4, PMA k=2, +SIDE
    params = sum(p.numel() for p in model.parameters())
    # Fixed default architecture -> deterministic count. Removing model-3's context
    # transformer (3 layers) + STV heads (3) drops it from 2,584,774 to 931,459
    # (~36% of model-3 / a ~64% reduction). Asserted exactly as a regression gate.
    assert params == 931_459, f"unexpected default param count {params}"
    assert params < 0.45 * MODEL3_PARAMS, "expected a large reduction vs model-3"


def test_no_transformer_and_no_stv_heads() -> None:
    _torch()
    from hexgnn.architecture import HexgnnNetwork

    model = HexgnnNetwork(token_dim=32, gnn_layers=2)
    # No context transformer stack and no STV heads survive in this lineage.
    assert not hasattr(model, "transformer")
    assert not hasattr(model, "short_term_value_heads")
    # The only learnable submodules: shared input proj, GNN trunk, three heads + PMA.
    child_names = {name for name, _ in model.named_children()}
    assert child_names == {"node_in", "gnn", "policy_head", "opp_policy_head", "value_pool", "value_head"}
    assert len(model.gnn) == 2


def test_forward_on_real_packed_graphs() -> None:
    torch = _torch()
    from hexgnn.architecture import HexgnnNetwork
    from hexgnn.graph_build import batch_from_states
    from hexgnn import VALUE_BINS

    states = _states(5, n_radius=4, seed=2)
    batch = batch_from_states(states, n=4)
    model = HexgnnNetwork(token_dim=48, gnn_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    ctot = int(batch["candidate_index"].shape[0])
    assert out["policy"].shape == (ctot,)
    assert out["opp_policy"].shape == (ctot,)
    assert out["value"].shape == (len(states), VALUE_BINS)
    # NO short-term-value heads in this lineage.
    assert not any(k.startswith("stvalue") for k in out)
    assert torch.isfinite(out["policy"]).all()
    assert torch.isfinite(out["value"]).all()

    # The inference-only path returns policy + value only (no opp).
    with torch.no_grad():
        pv = model.forward_policy_value(batch)
    assert set(pv.keys()) == {"policy", "value"}


def test_overfit_tiny_batch() -> None:
    torch = _torch()
    from hexgnn.architecture import HexgnnNetwork
    from hexgnn.graph_build import batch_from_states
    from hexgnn.losses import hexgnn_loss

    states = _states(2, n_radius=2, seed=9)
    batch = batch_from_states(states, n=2)
    ctot = int(batch["candidate_index"].shape[0])
    g = int(batch["num_graphs"])
    cg = batch["candidate_graph"]

    torch.manual_seed(0)
    pol_t = torch.zeros(ctot)
    for gid in range(g):
        first_idx = int((cg == gid).nonzero(as_tuple=True)[0][0])
        pol_t[first_idx] = 1.0
    targets = {
        "candidate_graph": cg,
        "num_graphs": g,
        "policy": pol_t,
        "opp_policy": pol_t.clone(),
        "value": torch.linspace(-0.8, 0.8, g),
    }

    model = HexgnnNetwork(token_dim=48, gnn_layers=2)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = last = None
    for step in range(250):
        opt.zero_grad()
        out = model(batch)
        loss, _ = hexgnn_loss(out, targets)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        if step == 0:
            first = val
        last = val
    assert last < first * 0.5, f"loss did not decrease enough: {first} -> {last}"
