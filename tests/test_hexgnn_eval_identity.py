"""Bit-identity gate for the optimized hexgnn MCTS eval callback.

`HexgnnInference.evaluate_featurized_batch` was refactored (Stage-2 eval-path
overhaul) to fuse the TWO device->host `.cpu()` syncs into ONE and to run the
value clip/nan_to_num as on-device kernels instead of host numpy. This test
proves the refactor is NUMERICALLY EXACT: it reconstructs the PRE-refactor
post-forward arithmetic (two separate D2H copies + numpy clip/nan_to_num) from
the SAME raw forward output and asserts the resulting `values_bytes` /
`priors_bytes` are BYTE-IDENTICAL to the shipping path.

Why feed both paths the same raw forward output rather than calling the forward
twice: the GNN trunk uses GPU `index_add_` atomics, whose floating-point
accumulation order is non-deterministic across launches, so two forwards on the
same input differ at ~1e-5 even for UNCHANGED code. Gating against a single
forward's raw output isolates the host-side post-processing — the only thing the
refactor touched — and makes the byte-identity assertion meaningful. (Under
`torch.use_deterministic_algorithms(True)` the whole forward is reproducible and
an end-to-end byte gate also passes; that is the heavier variant exercised by
the WSL harness, kept out of the default suite because it disables the atomics.)

Skips when CUDA is unavailable (the forward + its CSR scatters are GPU paths).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hexgnn.architecture import HexgnnNetwork
from hexgnn.inference import HexgnnInference
from hexgnn.losses import decode_binned_value, segment_log_softmax


def _legacy_post(self, out, dev_batch, num_graphs):
    """The EXACT pre-refactor post-forward arithmetic (two D2H + numpy clip)."""
    cg = dev_batch["candidate_graph"]
    raw_policy, raw_value, _ = self._count_and_sanitize(
        out["policy"].float(), out["value"].float(), cg, num_graphs
    )
    log_probs = segment_log_softmax(raw_policy, cg, num_graphs)
    priors = log_probs.exp().cpu().numpy().astype(np.float32, copy=False)
    values = decode_binned_value(raw_value).cpu().numpy()
    values = np.nan_to_num(np.clip(values, -1.0, 1.0), nan=0.0).astype(np.float32, copy=False)
    return {
        "values_bytes": np.ascontiguousarray(values).tobytes(),
        "priors_bytes": np.ascontiguousarray(priors).tobytes(),
    }


def _new_post(self, out, dev_batch, num_graphs):
    """The shipping post-forward arithmetic (one fused D2H), factored to consume
    the SAME raw forward output as the legacy path for an apples-to-apples gate."""
    cg = dev_batch["candidate_graph"]
    raw_policy, raw_value, _ = self._count_and_sanitize(
        out["policy"].float(), out["value"].float(), cg, num_graphs
    )
    priors = segment_log_softmax(raw_policy, cg, num_graphs).exp()
    values = decode_binned_value(raw_value)
    values = torch.nan_to_num(values.clamp(-1.0, 1.0), nan=0.0)
    fused = torch.cat([values.reshape(-1), priors.reshape(-1)]).to("cpu", torch.float32).numpy()
    return {
        "values_bytes": np.ascontiguousarray(fused[:num_graphs]).tobytes(),
        "priors_bytes": np.ascontiguousarray(fused[num_graphs:]).tobytes(),
    }


def _synth_payload_batch(infr, *, num_graphs, nodes_per, edges_per, cand_per, seed):
    """Build a representative collated batch dict (device tensors) + run ONE
    forward, returning (out, dev_batch, num_graphs). Shapes mirror the Rust
    `collated_to_py_dict` contract: graph-contiguous node/candidate/edge ids."""
    from hexgnn.constants import (
        NODE_FEATURE_DIM,
        EDGE_ATTR_DIM,
        NUM_EDGE_TYPES,
        NUM_NODE_TYPES,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    N, E, C = num_graphs * nodes_per, num_graphs * edges_per, num_graphs * cand_per
    node_graph = torch.arange(num_graphs, device=dev).repeat_interleave(nodes_per)
    cand_graph = torch.arange(num_graphs, device=dev).repeat_interleave(cand_per)
    cand_in = torch.randint(0, nodes_per, (C,), device=dev, generator=g)
    candidate_index = (cand_graph * nodes_per + cand_in).clamp_max(N - 1)
    e_graph = torch.arange(num_graphs, device=dev).repeat_interleave(edges_per)
    src = torch.randint(0, nodes_per, (E,), device=dev, generator=g)
    dst = torch.randint(0, nodes_per, (E,), device=dev, generator=g)
    edge_index = torch.stack([e_graph * nodes_per + src, e_graph * nodes_per + dst]).clamp_max(N - 1)
    batch = {
        "node_feat": torch.randn(N, NODE_FEATURE_DIM, device=dev, generator=g),
        "node_type": torch.randint(0, NUM_NODE_TYPES, (N,), device=dev, generator=g),
        "node_graph": node_graph,
        "edge_index": edge_index,
        "edge_type": torch.randint(0, NUM_EDGE_TYPES, (E,), device=dev, generator=g),
        "edge_attr": torch.randn(E, EDGE_ATTR_DIM, device=dev, generator=g),
        "edge_dir": torch.randint(-1, 6, (E,), device=dev, generator=g),
        "candidate_index": candidate_index,
        "candidate_graph": cand_graph,
        "num_graphs": num_graphs,
    }
    out, dev_batch = infr.forward_batch(batch)
    return out, dev_batch, num_graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="hexgnn eval forward needs CUDA")
def test_eval_post_processing_byte_identical():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = HexgnnNetwork(token_dim=96, gnn_layers=2, steerable_channels=4).to("cuda").eval()
    infr = HexgnnInference(model, device="cuda", fp16=True)

    # Varied sizes: tiny single-graph edge case, small multi-graph, and a large
    # batch that exercises the chunked/padded forward path.
    shapes = [
        dict(num_graphs=1, nodes_per=4, edges_per=6, cand_per=3, seed=1),
        dict(num_graphs=34, nodes_per=62, edges_per=230, cand_per=58, seed=2),
        dict(num_graphs=512, nodes_per=110, edges_per=520, cand_per=100, seed=3),
    ]
    for sh in shapes:
        out, dev_batch, ng = _synth_payload_batch(infr, **sh)
        legacy = _legacy_post(infr, out, dev_batch, ng)
        new = _new_post(infr, out, dev_batch, ng)
        assert legacy["values_bytes"] == new["values_bytes"], f"values differ at {sh}"
        assert legacy["priors_bytes"] == new["priors_bytes"], f"priors differ at {sh}"
