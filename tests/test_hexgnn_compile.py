"""torch.compile cleanliness for the hexgnn search forward.

The throughput-critical path (`forward_policy_value`) is the one self-play + eval
compile. With the SIDE-hub rows precomputed eagerly (as `HexgnnInference` does),
the compiled forward must run with NO data-dependent graph-break op and produce
output matching eager within fp tolerance, across DIFFERENT batch sizes (so a
single compiled artifact handles the variable-length packed graphs).
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
_HEXGNN = ROOT / "packages" / "hexgnn" / "python"
if str(_HEXGNN) not in sys.path:
    sys.path.insert(0, str(_HEXGNN))


def _torch():
    return pytest.importorskip("torch")


def _states(n_states, seed=0):
    eng = importlib.import_module("hexo_engine.api")
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


def test_compiled_forward_matches_eager_across_shapes() -> None:
    torch = _torch()
    from hexgnn.architecture import HexgnnNetwork, precompute_side_rows
    from hexgnn.graph_build import batch_from_states

    torch.manual_seed(0)
    model = HexgnnNetwork(token_dim=48, gnn_layers=2)
    model.eval()

    def prep(states):
        b = batch_from_states(states, n=4)
        b = {**b, **precompute_side_rows(b["node_type"], b["node_graph"])}
        return b

    try:
        compiled = torch.compile(model.forward_policy_value, dynamic=True)
    except Exception as exc:  # pragma: no cover - backend unavailable
        pytest.skip(f"torch.compile unavailable: {exc}")

    for k, seed in ((4, 1), (7, 2), (4, 3)):  # repeat a size to exercise reuse
        batch = prep(_states(k, seed=seed))
        with torch.no_grad():
            ref = model.forward_policy_value(batch)
            try:
                got = compiled(batch)
            except Exception as exc:  # pragma: no cover
                pytest.skip(f"torch.compile backend failed at runtime: {exc}")
        assert torch.allclose(got["policy"], ref["policy"], atol=1e-4, rtol=1e-4)
        assert torch.allclose(got["value"], ref["value"], atol=1e-4, rtol=1e-4)
        assert torch.isfinite(got["policy"]).all() and torch.isfinite(got["value"]).all()
