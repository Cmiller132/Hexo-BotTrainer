"""PyTorch architecture for hexgt (Model 2) — dynamic typed GNN + transformer.

Consumes the packed-graph batch (HEXGT_DECISIONS.md / `collate.py`) and returns
the training heads:
- `policy`: one logit per CANDIDATE node, dynamic length (Ctot,).
- `value`: a 65-bin scalar value distribution per graph (G, 65).
- `opp_policy`: per-candidate aux logits (Ctot,).
- `stvalue_<h>`: short-term value heads per graph (G, 65).

Body (§6.3): typed RELATIONAL message passing (line/co-linearity already routed
through window-hub nodes by the Rust builder — no same-axis cliques), then a
per-graph context transformer (context self-attention over {side, stone, window}
+ candidate→context cross-attention). All inputs are D6-INVARIANT (features.py)
and all ops are permutation-equivariant, so the model is D6-invariant by
construction (the equivariance test passes exactly; no augmentation).

Attention is computed per graph (graphs occupy contiguous slices in the packed
batch) to keep cost at O(#context^2 + #candidates·#context) per graph rather than
O(N^2) over the whole batch. The Python per-graph loop is the throughput lever to
vectorize in Phase 5; for the CPU MVP it is correct and clear.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .constants import (
    EDGE_ATTR_DIM,
    NODE_FEATURE_DIM,
    NODE_TYPE_CANDIDATE,
    NODE_TYPE_SIDE,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    VALUE_BINS,
)


def _graph_slices(node_graph: torch.Tensor, num_graphs: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) node slice per graph (collate keeps graphs in order)."""

    counts = torch.bincount(node_graph, minlength=num_graphs)
    slices = []
    start = 0
    for c in counts.tolist():
        slices.append((start, start + c))
        start += c
    return slices


class RelationalMessagePassing(nn.Module):
    """Typed message passing: m_{j->i} = relu(W_{type} h_j + edge_proj(attr)).

    Per-edge-type linear transform via a (num_edge_types, dim, dim) weight,
    applied through an einsum (no per-edge dense materialization). Aggregated by
    mean over incoming edges, residual + LayerNorm update.
    """

    def __init__(self, dim: int, num_edge_types: int, edge_attr_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_edge_types = num_edge_types
        self.weight = nn.Parameter(torch.empty(num_edge_types, dim, dim))
        self.bias = nn.Parameter(torch.zeros(num_edge_types, dim))
        self.edge_proj = nn.Linear(edge_attr_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        n = h.shape[0]
        if edge_index.shape[1] == 0:
            return self.norm(h)
        src = edge_index[0]
        dst = edge_index[1]
        # project[t] = h @ W_t^T  -> (num_edge_types, N, dim)
        project = torch.einsum("nf,tfg->tng", h, self.weight) + self.bias.unsqueeze(1)
        et = edge_type.clamp(0, self.num_edge_types - 1)
        m = project[et, src] + self.edge_proj(edge_attr)
        m = torch.relu(m)
        agg = h.new_zeros(n, self.dim).index_add_(0, dst, m)
        counts = h.new_zeros(n).index_add_(0, dst, h.new_ones(dst.shape[0]))
        agg = agg / counts.clamp_min(1.0).unsqueeze(-1)
        return self.norm(h + self.out_proj(agg))


class GraphTransformerLayer(nn.Module):
    """Per-graph context self-attention + candidate→context cross-attention.

    Context tokens = {side, stone, window} nodes; candidate tokens = candidates.
    Both updated with a residual FFN. Runs per graph over contiguous slices.
    """

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ctx_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cand_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ctx1 = nn.LayerNorm(dim)
        self.norm_ctx2 = nn.LayerNorm(dim)
        self.norm_cand1 = nn.LayerNorm(dim)
        self.norm_cand2 = nn.LayerNorm(dim)
        self.ffn_ctx = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, dim))
        self.ffn_cand = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, dim))

    def forward(
        self,
        h: torch.Tensor,
        slices: list[tuple[int, int]],
        is_candidate: torch.Tensor,
    ) -> torch.Tensor:
        out = h.clone()
        for start, end in slices:
            if end <= start:
                continue
            idx = torch.arange(start, end, device=h.device)
            cand_mask = is_candidate[start:end]
            ctx_local = idx[~cand_mask]
            cand_local = idx[cand_mask]
            if ctx_local.numel() == 0:
                continue
            ctx = h.index_select(0, ctx_local).unsqueeze(0)  # (1, M_ctx, D)
            # context self-attention
            a, _ = self.ctx_attn(ctx, ctx, ctx, need_weights=False)
            ctx = self.norm_ctx1(ctx + a)
            ctx = self.norm_ctx2(ctx + self.ffn_ctx(ctx))
            out.index_copy_(0, ctx_local, ctx.squeeze(0))
            # candidate -> context cross-attention
            if cand_local.numel() > 0:
                cand = h.index_select(0, cand_local).unsqueeze(0)  # (1, M_cand, D)
                a2, _ = self.cand_attn(cand, ctx, ctx, need_weights=False)
                cand = self.norm_cand1(cand + a2)
                cand = self.norm_cand2(cand + self.ffn_cand(cand))
                out.index_copy_(0, cand_local, cand.squeeze(0))
        return out


class HexgtNetwork(nn.Module):
    """Dynamic typed GNN + transformer producing the hexgt training heads."""

    def __init__(
        self,
        *,
        node_feature_dim: int = NODE_FEATURE_DIM,
        token_dim: int = 168,
        gnn_layers: int = 3,
        ctx_layers: int = 3,
        attention_heads: int = 4,
        ffn_dim: int = 336,
        dropout: float = 0.0,
        short_term_value_horizons: tuple[int, ...] = (),
        edge_attr_dim: int = EDGE_ATTR_DIM,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.token_dim = int(token_dim)
        self.gnn_layers = int(gnn_layers)
        self.ctx_layers = int(ctx_layers)
        self.attention_heads = int(attention_heads)
        self.ffn_dim = int(ffn_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.short_term_value_horizons = tuple(int(h) for h in short_term_value_horizons)

        self.node_in = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.gnn = nn.ModuleList(
            [RelationalMessagePassing(self.token_dim, NUM_EDGE_TYPES, self.edge_attr_dim) for _ in range(self.gnn_layers)]
        )
        self.transformer = nn.ModuleList(
            [GraphTransformerLayer(self.token_dim, self.attention_heads, self.ffn_dim, dropout) for _ in range(self.ctx_layers)]
        )

        self.policy_head = nn.Linear(self.token_dim, 1)
        self.opp_policy_head = nn.Linear(self.token_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim), nn.ReLU(inplace=True), nn.Linear(self.token_dim, VALUE_BINS)
        )
        self.short_term_value_heads = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.Linear(self.token_dim, self.token_dim), nn.ReLU(inplace=True), nn.Linear(self.token_dim, VALUE_BINS)
                )
                for h in self.short_term_value_horizons
            }
        )

    # -- internals -------------------------------------------------------------

    def _encode_nodes(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self._validate_batch(batch)
        h = self.node_in(batch["node_feat"])
        edge_index = batch["edge_index"]
        edge_type = batch.get("edge_type")
        edge_attr = batch.get("edge_attr")
        if edge_type is None:
            edge_type = h.new_zeros(edge_index.shape[1], dtype=torch.long)
        if edge_attr is None:
            edge_attr = h.new_zeros(edge_index.shape[1], self.edge_attr_dim)
        for layer in self.gnn:
            h = layer(h, edge_index, edge_type, edge_attr)

        num_graphs = int(batch["num_graphs"])
        slices = _graph_slices(batch["node_graph"], num_graphs)
        is_candidate = batch["node_type"] == NODE_TYPE_CANDIDATE
        for layer in self.transformer:
            h = layer(h, slices, is_candidate)
        return h

    def _graph_readout(self, batch: Mapping[str, torch.Tensor], node_emb: torch.Tensor) -> torch.Tensor:
        """Per-graph readout from the SIDE hub node (one per graph)."""

        num_graphs = int(batch["num_graphs"])
        node_type = batch["node_type"]
        node_graph = batch["node_graph"]
        side_rows = (node_type == NODE_TYPE_SIDE).nonzero(as_tuple=True)[0]
        readout = node_emb.new_zeros(num_graphs, self.token_dim)
        # last side node per graph (there is exactly one)
        readout[node_graph[side_rows]] = node_emb.index_select(0, side_rows)
        return readout

    def _heads(self, batch: Mapping[str, torch.Tensor], node_emb: torch.Tensor, *, with_aux: bool) -> dict[str, torch.Tensor]:
        candidate_index = batch["candidate_index"].to(dtype=torch.long)
        cand_emb = node_emb.index_select(0, candidate_index)
        graph_emb = self._graph_readout(batch, node_emb)
        outputs: dict[str, torch.Tensor] = {
            "policy": self.policy_head(cand_emb).squeeze(-1),
            "value": self.value_head(graph_emb),
        }
        if with_aux:
            outputs["opp_policy"] = self.opp_policy_head(cand_emb).squeeze(-1)
            for horizon, head in self.short_term_value_heads.items():
                outputs[f"stvalue_{horizon}"] = head(graph_emb)
        return outputs

    # -- public forward --------------------------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._heads(batch, self._encode_nodes(batch), with_aux=True)

    def forward_policy_value(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inference-only forward path for search (policy + value only)."""

        return self._heads(batch, self._encode_nodes(batch), with_aux=False)

    def _validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        node_feat = batch["node_feat"]
        if node_feat.ndim != 2 or node_feat.shape[1] != self.node_feature_dim:
            raise ValueError(
                f"node_feat must be (Ntot, {self.node_feature_dim}), got {tuple(node_feat.shape)}"
            )
        node_type = batch.get("node_type")
        if node_type is not None and node_type.numel() and bool(((node_type < 0) | (node_type >= NUM_NODE_TYPES)).any().item()):
            raise ValueError("node_type out of range")
