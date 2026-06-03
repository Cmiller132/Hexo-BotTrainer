"""Chunked (compressed) forward == single forward, bit-for-bit.

The VRAM-compression path splits a large leaf batch into candidate-count-sorted,
budget-bounded sub-batches so each chunk pads only to its local max. Graphs are
independent (within-graph edges + per-graph attention; the only cross-graph op is
the final per-graph softmax done after reassembly), so chunking must NOT change
the per-candidate policy logits or per-graph value. This pins that equality and
verifies the chunk planner actually splits.
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


def _states(n_states, seed=0):
    eng = importlib.import_module("hexo_engine.api")
    rng = random.Random(seed)
    out = []
    tries = 0
    # Vary move depth so candidate counts differ (the padding-waste driver).
    while len(out) < n_states and tries < 4000:
        tries += 1
        s = eng.new_game(seed=rng.randint(0, 10**6))
        for _ in range(rng.randint(4, 40)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def test_chunked_forward_matches_single() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.graph_build import batch_from_states
    from hexo_models.hexgt.inference import HexgtInference, _plan_chunks

    torch.manual_seed(0)
    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64).cpu().eval()
    states = _states(12, seed=5)
    batch = batch_from_states(states, 3)

    num_graphs = int(batch["num_graphs"])
    counts = torch.bincount(batch["candidate_graph"], minlength=num_graphs)

    # Single forward (budget huge -> no chunking).
    single = HexgtInference(model, device="cpu", fp16=False, forward_pad_budget=10**12)
    out_single, _ = single.forward_batch({k: (v.clone() if hasattr(v, "clone") else v) for k, v in batch.items()})

    # Small budget -> forces multiple chunks.
    budget = int(counts.max()) * 3  # only a few graphs per chunk
    chunks = _plan_chunks(counts, budget)
    assert len(chunks) > 1, "test did not actually exercise chunking"
    chunked = HexgtInference(model, device="cpu", fp16=False, forward_pad_budget=budget)
    out_chunk, _ = chunked.forward_batch({k: (v.clone() if hasattr(v, "clone") else v) for k, v in batch.items()})

    assert torch.allclose(out_single["policy"].float(), out_chunk["policy"].float(), atol=1e-5), \
        "chunked policy logits differ from single forward"
    assert torch.allclose(out_single["value"].float(), out_chunk["value"].float(), atol=1e-5), \
        "chunked value differs from single forward"


def test_chunk_planner_respects_budget() -> None:
    torch = _torch()
    from hexo_models.hexgt.inference import _plan_chunks

    counts = torch.tensor([5, 5, 1000, 5, 5, 5, 5, 5], dtype=torch.long)
    chunks = _plan_chunks(counts, budget=40)
    # every chunk must satisfy len * max_count <= budget (except a lone oversized graph)
    for ch in chunks:
        c = counts[ch]
        if ch.numel() > 1:
            assert int(ch.numel()) * int(c.max()) <= 40
    # all graph ids covered exactly once
    allids = torch.cat([ch for ch in chunks]).sort().values
    assert torch.equal(allids, torch.arange(8))
