"""Byte-identity gate for the Rust featurizer vs the Python collate, through the
hexgnn package.

hexgnn REUSES hexgt's already-built native accelerator (`hexo_models._rust.hexgt`)
read-only — the feature/candidate/graph layout is unchanged, so parity carries
over. This gate confirms it end-to-end: the Rust collated batch
(`hexgnn_featurize_states`, accessed via hexgnn.rust_bridge) must equal
`hexgnn.graph_build.batch_from_states` (the Python path) on real-ish positions,
and the zero-copy buffers must be read-only views.
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


def _np():
    return pytest.importorskip("numpy")


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
    from hexgnn import rust_bridge

    # native fn name stays `hexgnn_featurize_states` (the shared native module).
    d = rust_bridge._hexgnn_module().hexgnn_featurize_states(tuple(states), n)
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
        edge_dir=np.frombuffer(d["edge_dir"], dtype=np.int64),
        num_graphs=int(d["num_graphs"]),
    )


def test_rust_featurizer_matches_python_collate() -> None:
    np = _np()
    pytest.importorskip("torch")
    from hexgnn.graph_build import batch_from_states

    n = 3
    states = _states(48, seed=3)
    rb = _rust_batch(states, n)
    pb = batch_from_states(states, n)
    py = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in pb.items()}

    assert rb["num_graphs"] == py["num_graphs"] == len(states)
    for key in (
        "node_type",
        "node_graph",
        "edge_type",
        "edge_dir",
        "candidate_index",
        "candidate_graph",
        "candidate_ids",
    ):
        assert np.array_equal(rb[key], py[key]), f"{key} mismatch"
    assert np.array_equal(rb["edge_index"], py["edge_index"]), "edge_index mismatch"
    assert rb["node_feat"].shape == py["node_feat"].shape
    assert np.abs(rb["node_feat"] - py["node_feat"]).max() < 1e-6
    assert np.abs(rb["edge_attr"] - py["edge_attr"]).max() < 1e-6


def test_feature_buffers_are_readonly_zero_copy() -> None:
    np = _np()
    from hexgnn import rust_bridge

    states = _states(4, seed=11)
    d = rust_bridge._hexgnn_module().hexgnn_featurize_states(tuple(states), 3)
    arr = np.frombuffer(d["node_feat"], dtype=np.float32)
    assert not arr.flags.writeable
