"""Phase-3 model gates: param budget, real-graph forward, overfit a tiny batch."""

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


def test_param_budget_within_10pct_of_2_1M() -> None:
    _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.constants import (
        DEFAULT_TOKEN_DIM, DEFAULT_GNN_LAYERS, DEFAULT_CTX_LAYERS, DEFAULT_FFN_DIM, DEFAULT_ATTENTION_HEADS,
    )

    model = HexgtNetwork(
        token_dim=DEFAULT_TOKEN_DIM, gnn_layers=DEFAULT_GNN_LAYERS, ctx_layers=DEFAULT_CTX_LAYERS,
        attention_heads=DEFAULT_ATTENTION_HEADS, ffn_dim=DEFAULT_FFN_DIM,
        short_term_value_horizons=(1, 4, 8),
    )
    params = sum(p.numel() for p in model.parameters())
    assert 1.89e6 <= params <= 2.31e6, f"param count {params} not within ~10% of 2.1M"


def test_forward_on_real_packed_graphs() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.graph_build import batch_from_states
    from hexo_models.hexgt import VALUE_BINS

    states = _states(5, n_radius=4, seed=2)
    batch = batch_from_states(states, n=4)
    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64,
                         short_term_value_horizons=(1, 4))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    ctot = int(batch["candidate_index"].shape[0])
    assert out["policy"].shape == (ctot,)
    assert out["opp_policy"].shape == (ctot,)
    assert out["value"].shape == (len(states), VALUE_BINS)
    assert out["stvalue_1"].shape == (len(states), VALUE_BINS)
    assert torch.isfinite(out["policy"]).all()
    assert torch.isfinite(out["value"]).all()


def test_overfit_tiny_batch() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.graph_build import batch_from_states
    from hexo_models.hexgt.losses import hexgt_loss

    # small positions => small candidate sets => a tiny batch is overfittable
    states = _states(2, n_radius=2, seed=9)
    batch = batch_from_states(states, n=2)
    ctot = int(batch["candidate_index"].shape[0])
    g = int(batch["num_graphs"])
    cg = batch["candidate_graph"]

    torch.manual_seed(0)
    # PEAKED policy targets: all mass on the first candidate of each graph.
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

    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    last = None
    for step in range(250):
        opt.zero_grad()
        out = model(batch)
        loss, _ = hexgt_loss(out, targets)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        if step == 0:
            first = val
        last = val
    assert last < first * 0.5, f"loss did not decrease enough: {first} -> {last}"
