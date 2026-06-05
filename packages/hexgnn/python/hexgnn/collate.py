"""Variable-size graph collation (the packed-graph batch contract, gap I).

Packs G per-position `GraphTensors` into ONE disjoint graph (PyG-style): node
arrays concatenated with a per-node `graph_id`, edges and candidate indices
offset into the packed node space, and a per-candidate `graph_id` for the
per-graph policy softmax. Deterministic and byte-stable (no Python set/dict
ordering) so training batches are reproducible.

The output dict is exactly the `HexgnnNetwork.forward` input contract
(HEXGT_DECISIONS.md): `node_feat`, `node_type`, `node_graph`, `edge_index`,
`edge_type`, `edge_attr`, `candidate_index`, `candidate_graph`, `num_graphs`,
plus `candidate_ids` (CSR mapping back to engine action ids for priors).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .features import GraphTensors


def collate_graphs(graphs: Sequence[GraphTensors]) -> dict[str, torch.Tensor | int]:
    """Pack per-graph tensors into a single disjoint-graph batch dict."""

    if not graphs:
        raise ValueError("collate_graphs requires at least one graph")

    node_feat = []
    node_type = []
    node_graph = []
    edge_src = []
    edge_dst = []
    edge_type = []
    edge_attr = []
    candidate_index = []
    candidate_graph = []
    candidate_ids = []

    node_offset = 0
    for gid, g in enumerate(graphs):
        n = g.num_nodes
        node_feat.append(g.node_feat)
        node_type.append(g.node_type)
        node_graph.append(np.full(n, gid, dtype=np.int64))
        if g.edge_index.shape[1] > 0:
            edge_src.append(g.edge_index[0] + node_offset)
            edge_dst.append(g.edge_index[1] + node_offset)
            edge_type.append(g.edge_type)
            edge_attr.append(g.edge_attr)
        c = int(g.candidate_index.shape[0])
        if c > 0:
            candidate_index.append(g.candidate_index + node_offset)
            candidate_graph.append(np.full(c, gid, dtype=np.int64))
            candidate_ids.append(g.candidate_ids)
        node_offset += n

    def _cat(parts, dim=0, dtype=None, empty_shape=None):
        """Concatenate `parts`, coercing to `dtype`; an empty list yields a
        zero-length array of `empty_shape`. `dtype` is enforced on BOTH branches
        so callers never need a trailing `.astype()`."""
        if parts:
            out = np.concatenate(parts, axis=dim)
            return out.astype(dtype) if dtype is not None else out
        return np.zeros(empty_shape, dtype=dtype)

    ei_src = _cat(edge_src, dtype=np.int64, empty_shape=(0,))
    ei_dst = _cat(edge_dst, dtype=np.int64, empty_shape=(0,))
    edge_index = np.stack([ei_src, ei_dst], axis=0)

    feat_dim = graphs[0].node_feat.shape[1]
    attr_dim = graphs[0].edge_attr.shape[1]

    batch = {
        "node_feat": torch.from_numpy(np.concatenate(node_feat, axis=0).astype(np.float32)),
        "node_type": torch.from_numpy(np.concatenate(node_type, axis=0).astype(np.int64)),
        "node_graph": torch.from_numpy(np.concatenate(node_graph, axis=0).astype(np.int64)),
        "edge_index": torch.from_numpy(edge_index),
        "edge_type": torch.from_numpy(_cat(edge_type, dtype=np.int64, empty_shape=(0,))),
        "edge_attr": torch.from_numpy(_cat(edge_attr, dtype=np.float32, empty_shape=(0, attr_dim))),
        "candidate_index": torch.from_numpy(_cat(candidate_index, dtype=np.int64, empty_shape=(0,))),
        "candidate_graph": torch.from_numpy(_cat(candidate_graph, dtype=np.int64, empty_shape=(0,))),
        "candidate_ids": torch.from_numpy(_cat(candidate_ids, dtype=np.int64, empty_shape=(0,))),
        "num_graphs": len(graphs),
        "_feat_dim": feat_dim,
    }
    return batch
