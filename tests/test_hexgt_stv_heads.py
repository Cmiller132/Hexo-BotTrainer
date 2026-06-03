"""Short-term-value (KataGo EMA) auxiliary heads at horizons [4,12,24].

Confirms: the 3 heads build, training forward emits stvalue_4/12/24 (search forward
does not), hexgt_loss picks up the stvalue_* terms, the EMA target matches the
KataGo decay (lambda=h/(h+1)), and — the graft proof — loading a pre-STV
checkpoint (strict=False) leaves policy+value output BYTE-IDENTICAL while adding
the fresh aux heads. STV is training-only aux: it cannot change search/play.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    p = ROOT / "packages" / package / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

HORIZONS = (4, 12, 24)


def _torch():
    return pytest.importorskip("torch")


def _eng():
    return importlib.import_module("hexo_engine.api")


def _net(horizons):
    from hexo_models.hexgt.architecture import HexgtNetwork
    return HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, attention_heads=4,
                        ffn_dim=64, short_term_value_horizons=horizons)


def _batch(seed=5, plies=24):
    from hexo_models.hexgt.graph_build import batch_from_states
    eng = _eng()
    rng = random.Random(seed)
    s = eng.new_game(seed=seed)
    for _ in range(plies):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        if not acts:
            break
        eng.apply_action(s, rng.choice(acts))
    return batch_from_states([s], n=3)


# --- heads build + emit -------------------------------------------------------

def test_stv_heads_build_and_emit() -> None:
    torch = _torch()
    from hexo_models.hexgt.constants import VALUE_BINS
    net = _net(HORIZONS)
    net.eval()
    assert set(net.short_term_value_heads.keys()) == {"4", "12", "24"}
    batch = _batch()
    with torch.no_grad():
        out = net(batch)                       # training forward (with_aux)
        sv = net.forward_policy_value(batch)   # search forward (no aux)
    g = int(batch["num_graphs"])
    for h in HORIZONS:
        assert out[f"stvalue_{h}"].shape == (g, VALUE_BINS)
        assert f"stvalue_{h}" not in sv         # search/play never emits stv
    assert "stvalue_4" not in sv and "opp_policy" not in sv


# --- loss wiring --------------------------------------------------------------

def test_loss_picks_up_stvalue_terms() -> None:
    torch = _torch()
    from hexo_models.hexgt.losses import hexgt_loss
    net = _net(HORIZONS)
    batch = _batch()
    out = net(batch)
    g = int(batch["num_graphs"])
    targets = {
        "candidate_graph": batch["candidate_graph"],
        "num_graphs": g,
        "policy": torch.ones_like(out["policy"]),
        "opp_policy": torch.ones_like(out["opp_policy"]),
        "value": torch.zeros(g),
    }
    for h in HORIZONS:
        targets[f"stvalue_{h}"] = torch.zeros(g)
        targets[f"stvalue_{h}_mask"] = torch.ones(g)
    total, comps = hexgt_loss(out, targets)
    for h in HORIZONS:
        assert f"stvalue_{h}" in comps and torch.isfinite(comps[f"stvalue_{h}"])


# --- EMA target correctness (KataGo lambda = h/(h+1)) -------------------------

def test_ema_target_matches_katago_decay() -> None:
    from hexo_models.dense_cnn.samples import _short_term_value_targets
    # decisions: (player, sample, root_value); same player so no sign flip.
    vals = [0.0, 1.0, -1.0, 0.5, 0.2, -0.3]
    decisions = [("player0", None, v) for v in vals]
    got = dict(_short_term_value_targets(decisions, 0, "player0", HORIZONS))
    future = vals[1:]
    for h in HORIZONS:
        decay = h / (h + 1.0)
        ws = wt = 0.0; w = 1.0
        for v in future:
            ws += w * v; wt += w; w *= decay
        assert got[h] == pytest.approx(ws / wt, abs=1e-9)
    # perspective flip: opponent's future root values are negated
    flipped = dict(_short_term_value_targets(
        [("player0", None, vals[0])] + [("player1", None, v) for v in future], 0, "player0", (4,)))
    decay = 4 / 5.0
    ws = wt = 0.0; w = 1.0
    for v in future:
        ws += w * (-v); wt += w; w *= decay
    assert flipped[4] == pytest.approx(ws / wt, abs=1e-9)


# --- GRAFT proof: policy+value identical after strict=False load --------------

def test_graft_preserves_policy_and_value_output() -> None:
    torch = _torch()
    import _rl_train

    torch.manual_seed(0)
    old = _net(())          # pre-STV checkpoint model (no aux heads)
    old.eval()
    sd = old.state_dict()
    assert not any(k.startswith("short_term_value_heads") for k in sd)

    torch.manual_seed(123)  # different init -> proves the load, not luck
    new = _net(HORIZONS)
    info = new.load_state_dict(sd, strict=False)
    # the validator the driver uses must accept this load
    _rl_train._validate_stv_resume_load(info)
    assert info.unexpected_keys == []
    assert all(k.startswith("short_term_value_heads") for k in info.missing_keys)
    assert info.missing_keys, "expected the new stv heads to be fresh-init"
    new.eval()

    batch = _batch(seed=7)
    with torch.no_grad():
        a, b = old(batch), new(batch)
    assert torch.equal(a["policy"], b["policy"]), "policy changed by the graft"
    assert torch.equal(a["value"], b["value"]), "value changed by the graft"


def test_validator_rejects_real_mismatch() -> None:
    import _rl_train

    class FakeInfo:
        def __init__(self, missing, unexpected):
            self.missing_keys = missing; self.unexpected_keys = unexpected
    # ok: only stv heads missing
    _rl_train._validate_stv_resume_load(FakeInfo(["short_term_value_heads.4.0.weight"], []))
    # bad: a trunk key missing, or an unexpected key present
    with pytest.raises(RuntimeError):
        _rl_train._validate_stv_resume_load(FakeInfo(["gnn.0.weight"], []))
    with pytest.raises(RuntimeError):
        _rl_train._validate_stv_resume_load(FakeInfo([], ["stale.key"]))
