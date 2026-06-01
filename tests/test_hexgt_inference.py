"""Phase-5 inference contract: per-candidate priors + scalar value from states."""

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


def _states(n_states, seed=0):
    eng = importlib.import_module("hexo_engine.api")
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n_states and tries < 2000:
        tries += 1
        s = eng.new_game(seed=rng.randint(0, 10**6))
        for _ in range(rng.randint(4, 24)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def test_inference_priors_and_value_contract() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.inference import HexgtInference
    from hexo_models.hexgt import rust_bridge

    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64)
    infer = HexgtInference(model, device="cpu", fp16=False)
    states = _states(5, seed=3)
    results = infer.evaluate_states(states, n=3)

    assert len(results) == len(states)
    for state, res in zip(states, results):
        # candidate ids match the shared Rust builder, in CSR order
        assert list(res["candidate_ids"]) == list(rust_bridge.candidate_ids(state, 3))
        priors = res["priors"]
        assert priors.shape[0] == res["candidate_ids"].shape[0]
        assert abs(float(priors.sum()) - 1.0) < 1e-4  # per-state softmax
        assert (priors >= 0).all()
        assert -1.0 <= res["value"] <= 1.0
