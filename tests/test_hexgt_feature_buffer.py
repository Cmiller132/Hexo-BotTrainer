"""Byte-identity gate for the Rust featurizer + zero-copy collation.

`features.rs` (the parallel Rust replacement for the per-graph Python
`build_graph_tensors` + `collate_graphs`, the ~88% self-play bottleneck) MUST
produce the SAME packed batch as the Python path — a mismatch silently poisons
the model (the dense_cnn D6-augmentation bug class), and it must hold because
search inputs == training inputs. This gate compares the Rust collated batch
(zero-copy buffers from `hexgt_featurize_states`) against `batch_from_states`
on real-ish random positions, and checks the buffers really are zero-copy
(read-only views).
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


def _np():
    return pytest.importorskip("numpy")


def _torch():
    return pytest.importorskip("torch")


def _states(k, seed=3):
    eng = importlib.import_module("hexo_engine.api")
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        s = eng.new_game(seed=rng.randint(0, 10**7))
        for _ in range(rng.randint(0, 90)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def _rust_batch(states, n):
    np = _np()
    from hexo_models.hexgt import rust_bridge

    d = rust_bridge._hexgt_module().hexgt_featurize_states(tuple(states), n)
    tn, te, tc = int(d["total_nodes"]), int(d["total_edges"]), int(d["total_candidates"])
    fd, ad = int(d["feat_dim"]), int(d["attr_dim"])
    ei = np.frombuffer(d["edge_index"], dtype=np.int64).reshape(2, te)
    return dict(
        node_feat=np.frombuffer(d["node_feat"], dtype=np.float32).reshape(tn, fd),
        node_type=np.frombuffer(d["node_type"], dtype=np.int64),
        node_graph=np.frombuffer(d["node_graph"], dtype=np.int64),
        edge_index=ei,
        edge_type=np.frombuffer(d["edge_type"], dtype=np.int64),
        edge_attr=np.frombuffer(d["edge_attr"], dtype=np.float32).reshape(te, ad),
        candidate_index=np.frombuffer(d["candidate_index"], dtype=np.int64),
        candidate_graph=np.frombuffer(d["candidate_graph"], dtype=np.int64),
        candidate_ids=np.frombuffer(d["candidate_ids"], dtype=np.int64),
        num_graphs=int(d["num_graphs"]),
    )


def test_rust_featurizer_matches_python_collate() -> None:
    np = _np()
    _torch()
    from hexo_models.hexgt.graph_build import batch_from_states

    n = 3
    states = _states(48, seed=3)
    rb = _rust_batch(states, n)
    pb = batch_from_states(states, n)
    py = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in pb.items()}

    assert rb["num_graphs"] == py["num_graphs"] == len(states)
    # int arrays must match EXACTLY (indices, segment ids, action ids, types)
    for key in (
        "node_type",
        "node_graph",
        "edge_type",
        "candidate_index",
        "candidate_graph",
        "candidate_ids",
    ):
        assert np.array_equal(rb[key], py[key]), f"{key} mismatch"
    assert np.array_equal(rb["edge_index"], py["edge_index"]), "edge_index mismatch"
    # float arrays: byte-identical in practice; gate at the 1e-6 the model can't feel
    assert rb["node_feat"].shape == py["node_feat"].shape
    assert np.abs(rb["node_feat"] - py["node_feat"]).max() < 1e-6
    assert np.abs(rb["edge_attr"] - py["edge_attr"]).max() < 1e-6


def test_feature_buffers_are_readonly_zero_copy() -> None:
    np = _np()
    from hexo_models.hexgt import rust_bridge

    states = _states(4, seed=11)
    d = rust_bridge._hexgt_module().hexgt_featurize_states(tuple(states), 3)
    arr = np.frombuffer(d["node_feat"], dtype=np.float32)
    # zero-copy view: numpy must see it as non-writable (read-only buffer protocol)
    assert not arr.flags.writeable
