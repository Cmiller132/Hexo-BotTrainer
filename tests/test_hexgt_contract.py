"""Phase-2/3 contract conformance: candidate↔priors CSR ordering, checkpoints."""

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


def _state(seed: int, steps: int):
    eng = _eng()
    rng = random.Random(seed)
    s = eng.new_game(seed=seed)
    for _ in range(steps):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        if not acts:
            break
        eng.apply_action(s, rng.choice(acts))
    return s


def test_candidate_csr_ordering_matches_rust() -> None:
    """candidate_ids/candidate_index in the packed batch preserve the Rust
    per-position CSR order, contiguously per graph, aligned with the policy."""
    torch = _torch()
    from hexo_models.hexgt import rust_bridge
    from hexo_models.hexgt.graph_build import batch_from_states
    from hexo_models.hexgt.constants import NODE_TYPE_CANDIDATE

    s0 = _state(1, 12)
    s1 = _state(2, 20)
    ids0 = rust_bridge.candidate_ids(s0, 4)
    ids1 = rust_bridge.candidate_ids(s1, 4)
    batch = batch_from_states([s0, s1], n=4)

    packed_ids = batch["candidate_ids"].tolist()
    assert packed_ids == list(ids0) + list(ids1), "CSR candidate order not preserved across collation"

    cg = batch["candidate_graph"].tolist()
    assert cg == [0] * len(ids0) + [1] * len(ids1), "candidate_graph segments must be contiguous per graph"

    # candidate_index must point at CANDIDATE-typed nodes
    cidx = batch["candidate_index"]
    types = batch["node_type"].index_select(0, cidx)
    assert bool((types == NODE_TYPE_CANDIDATE).all().item())

    # policy length == number of candidates, aligned 1:1 with candidate_ids
    from hexo_models.hexgt.architecture import HexgtNetwork

    model = HexgtNetwork(token_dim=32, gnn_layers=1, ctx_layers=1, ffn_dim=48)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["policy"].shape[0] == len(packed_ids)


def test_checkpoint_round_trip_and_incompat() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.checkpoints import HexgtCheckpointLoader

    model = HexgtNetwork(token_dim=32, gnn_layers=1, ctx_layers=1, ffn_dim=48,
                         short_term_value_horizons=(1,))
    payload = {
        "model": "hexo_models.hexgt",
        "model_state": model.state_dict(),
        "optimizer_state": None,
        "train_state": None,
        "epoch": 5,
        "metadata": {},
    }
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "epoch_000005.pt"
        torch.save(payload, ckpt)

        # matching architecture -> loaded, weights restored
        fresh = HexgtNetwork(token_dim=32, gnn_layers=1, ctx_layers=1, ffn_dim=48,
                             short_term_value_horizons=(1,))

        class _M:
            pass

        comp = _M()
        comp.model = _M()
        comp.model.model = fresh
        comp.model.optimizer = None
        comp.model.trainer = None
        res = HexgtCheckpointLoader().load(str(ckpt), ctx=None, components=comp)
        assert res["status"] == "loaded", res
        for k, v in model.state_dict().items():
            assert torch.equal(v, fresh.state_dict()[k])

        # mismatched architecture -> initialized with incompatibilities
        wrong = HexgtNetwork(token_dim=48, gnn_layers=1, ctx_layers=1, ffn_dim=48)
        comp2 = _M()
        comp2.model = _M()
        comp2.model.model = wrong
        comp2.model.optimizer = None
        comp2.model.trainer = None
        res2 = HexgtCheckpointLoader().load(str(ckpt), ctx=None, components=comp2)
        assert res2["status"] == "initialized"
        assert "incompatible" in res2["reason"]


def test_value_decode_in_range() -> None:
    torch = _torch()
    from hexo_models.hexgt.losses import decode_binned_value

    logits = torch.randn(7, 65)
    decoded = decode_binned_value(logits)
    assert decoded.shape == (7,)
    assert bool((decoded >= -1.0001).all().item()) and bool((decoded <= 1.0001).all().item())
