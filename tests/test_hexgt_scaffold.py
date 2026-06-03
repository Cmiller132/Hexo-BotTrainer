"""Phase-0 contract-conformance tests for the hexgt (Model 2) scaffold.

Gate (HEXFORMER_REWRITE_PLAN.md Phase 0): the package installs editable,
`load_model_plugin` resolves `hexgt`, and a forward on a packed-graph batch
returns the right output keys/shapes — `policy` (dynamic per-candidate), `value`
(G, 65), `opp_policy` (dynamic), and optional `stvalue_*`. Also covers config
unknown-key rejection, checkpoint `sample_buffer` rejection, the dynamic policy
segment CE, and the native `hexgt` Rust submodule capability payload.

Phase 2 expands these into the full dense_cnn-style test set.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch() -> Any:
    return pytest.importorskip("torch")


def _hexgt() -> Any:
    return importlib.import_module("hexo_models.hexgt")


def _hexgt_plugin() -> Any:
    return importlib.import_module("hexo_models.hexgt.plugin").get_plugin()


def _small_model_config() -> dict[str, Any]:
    return {
        "device": "cpu",
        "architecture": {
            "node_feature_dim": 32,
            "token_dim": 16,
            "gnn_layers": 1,
            "ctx_layers": 1,
            "attention_heads": 2,
            "ffn_dim": 32,
            "short_term_value_horizons": [1, 4],
            "candidate_radius": 2,
        },
        "training": {"batch_size": 2, "learning_rate": 3.0e-4, "amp": False},
    }


def _synthetic_batch(torch: Any, *, graphs: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    """Build a packed-graph batch.

    `graphs` is a tuple of (num_nodes, num_candidates) per graph. Each graph gets
    one SIDE node (index 0 within the graph) plus filler nodes; `num_candidates`
    of the nodes are marked as candidates.
    """

    node_feat_rows = []
    node_type_rows = []
    node_graph_rows = []
    candidate_index = []
    candidate_graph = []
    edge_src = []
    edge_dst = []
    edge_type = []
    offset = 0
    feat_dim = 32
    for gid, (n_nodes, n_cands) in enumerate(graphs):
        assert n_nodes >= 1 and 1 <= n_cands <= n_nodes
        for local in range(n_nodes):
            node_feat_rows.append([float((local + gid) % 5) for _ in range(feat_dim)])
            node_type_rows.append(0 if local == 0 else (2 if local <= n_cands else 1))
            node_graph_rows.append(gid)
        # candidates: nodes [1 .. n_cands]
        for c in range(1, n_cands + 1):
            candidate_index.append(offset + c)
            candidate_graph.append(gid)
        # a couple of context edges from SIDE hub to other nodes
        for local in range(1, n_nodes):
            edge_src.append(offset + 0)
            edge_dst.append(offset + local)
            edge_type.append(4)
        offset += n_nodes

    return {
        "node_feat": torch.tensor(node_feat_rows, dtype=torch.float32),
        "node_type": torch.tensor(node_type_rows, dtype=torch.long),
        "node_graph": torch.tensor(node_graph_rows, dtype=torch.long),
        "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.long),
        "edge_type": torch.tensor(edge_type, dtype=torch.long),
        "candidate_index": torch.tensor(candidate_index, dtype=torch.long),
        "candidate_graph": torch.tensor(candidate_graph, dtype=torch.long),
        "num_graphs": len(graphs),
    }


# --- plugin / registry --------------------------------------------------------

def test_plugin_resolves_by_module_and_entry_point() -> None:
    from hexo_train.config import ModelConfig
    from hexo_train.registry import load_model_plugin

    by_module = load_model_plugin(ModelConfig(name="hexgt", module="hexo_models.hexgt.plugin"))
    assert by_module.name == "hexo_models.hexgt"
    # Entry-point / name resolution requires the package to be installed with its
    # entry point registered (the build venv); skip cleanly if not discoverable.
    try:
        by_name = load_model_plugin(ModelConfig(name="hexgt"))
    except LookupError:
        pytest.skip("hexgt entry point not registered in this interpreter")
    assert by_name.name == "hexo_models.hexgt"


# --- forward contract ---------------------------------------------------------

def test_forward_returns_expected_keys_and_shapes() -> None:
    torch = _torch()
    plugin = _hexgt_plugin()
    model = plugin.build_model({}, _small_model_config())
    model.eval()
    graphs = ((6, 3), (4, 2), (8, 5))  # (nodes, candidates) per graph
    batch = _synthetic_batch(torch, graphs=graphs)
    total_cands = sum(c for _, c in graphs)

    with torch.no_grad():
        out = model(batch)

    assert set(out) >= {"policy", "value", "opp_policy", "stvalue_1", "stvalue_4"}
    assert out["policy"].shape == (total_cands,)
    assert out["opp_policy"].shape == (total_cands,)
    assert out["value"].shape == (len(graphs), _hexgt().VALUE_BINS)
    assert out["stvalue_1"].shape == (len(graphs), _hexgt().VALUE_BINS)
    assert torch.isfinite(out["policy"]).all()
    assert torch.isfinite(out["value"]).all()


def test_forward_policy_value_skips_aux() -> None:
    torch = _torch()
    plugin = _hexgt_plugin()
    model = plugin.build_model({}, _small_model_config())
    model.eval()
    batch = _synthetic_batch(torch, graphs=((5, 2),))
    with torch.no_grad():
        out = model.forward_policy_value(batch)
    assert set(out) == {"policy", "value"}


# --- dynamic policy segment CE ------------------------------------------------

def test_segment_policy_ce_matches_dense_per_graph() -> None:
    torch = _torch()
    from hexo_models.hexgt.losses import segment_softmax_cross_entropy

    # Two graphs: graph0 has 2 candidates, graph1 has 3.
    logits = torch.tensor([0.5, -0.5, 1.0, 0.0, -1.0])
    target = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0])
    seg = torch.tensor([0, 0, 1, 1, 1])
    loss = segment_softmax_cross_entropy(logits, target, seg, 2)

    # Reference: dense per-graph CE averaged over the 2 graphs.
    import torch.nn.functional as F

    g0 = -(torch.tensor([1.0, 0.0]) * F.log_softmax(logits[:2], dim=0)).sum()
    g1_t = torch.tensor([0.0, 1.0, 0.0])  # normalized graph1 target
    g1 = -(g1_t * F.log_softmax(logits[2:], dim=0)).sum()
    expected = (g0 + g1) / 2
    assert torch.allclose(loss, expected, atol=1e-6)


def test_policy_loss_rejects_zero_mass_graph() -> None:
    torch = _torch()
    from hexo_models.hexgt.losses import segment_softmax_cross_entropy

    logits = torch.tensor([0.1, 0.2, 0.3])
    target = torch.tensor([0.0, 0.0, 0.0])
    seg = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError):
        segment_softmax_cross_entropy(logits, target, seg, 2)


# --- config -------------------------------------------------------------------

def test_config_rejects_unknown_section_key() -> None:
    from hexo_models.hexgt.config import parse_hexgt_config

    with pytest.raises(ValueError):
        parse_hexgt_config({"architecture": {"not_a_field": 1}})
    with pytest.raises(ValueError):
        parse_hexgt_config({"totally_unknown_section": {}})


def test_config_defaults_and_candidate_radius() -> None:
    from hexo_models.hexgt.config import parse_hexgt_config

    cfg = parse_hexgt_config({"architecture": {"candidate_radius": 4}})
    assert cfg.architecture.candidate_radius == 4
    assert cfg.architecture.token_dim == 168
    assert cfg.training.warmup_steps == 1000


# --- checkpoint ---------------------------------------------------------------

def test_checkpoint_rejects_sample_buffer(tmp_path: Path) -> None:
    torch = _torch()
    from hexo_models.hexgt.checkpoints import HexgtCheckpointLoader

    ckpt = tmp_path / "legacy.pt"
    torch.save({"model": "hexo_models.hexgt", "sample_buffer": {"rows": 1}, "epoch": 3}, ckpt)

    class _M:
        model = None
        optimizer = None
        trainer = None

    class _Comp:
        model = _M()

    result = HexgtCheckpointLoader().load(str(ckpt), ctx=None, components=_Comp())
    assert result["status"] == "initialized"
    assert "sample_buffer" in result["reason"]


# --- rust submodule -----------------------------------------------------------

def test_rust_capabilities_reports_hexgt() -> None:
    pytest.importorskip("hexo_models._rust")
    from hexo_models.hexgt import rust_bridge

    caps = rust_bridge.capabilities()
    assert caps["model_family"] == "hexgt"
    assert caps["policy"] == "dynamic_per_candidate"
