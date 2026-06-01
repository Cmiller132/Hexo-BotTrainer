"""PyTorch architecture for hexgt (Model 2) — Phase-0 SCAFFOLD STUB.

This module defines the *interface* of the dynamic GNN + transformer model: it
consumes a packed-graph batch (the contract in `HEXGT_DECISIONS.md`) and returns
the exact heads training and search consume:

- `policy`: one logit per **candidate** node, dynamic length (Ctot,).
- `value`: a 65-bin scalar value distribution in `[-1, 1]`, one per graph (G, 65).
- `opp_policy`: an auxiliary per-candidate policy target, same dynamic shape.
- `stvalue_<horizon>`: optional short-term value heads, per graph (G, 65).

The BODY here is a deliberately minimal placeholder (a per-node MLP + segment
pooling) so the package installs, the plugin resolves, and contract tests can run
on random weights (Phase 0/2 gate). The real typed message-passing GNN + context
transformer + cross-attention body, sized to the ~2.1M-param budget, lands in
Phase 3 — it will replace `_encode_nodes` while keeping this forward contract.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .constants import (
    NODE_FEATURE_DIM,
    NUM_NODE_TYPES,
    VALUE_BINS,
)


def _segment_mean(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Mean-pool (Ntot, D) node features into (num_segments, D) by segment id."""

    seg = segment_ids.to(dtype=torch.long)
    dim = values.shape[-1]
    summed = values.new_zeros(num_segments, dim).index_add(0, seg, values)
    counts = values.new_zeros(num_segments).index_add(0, seg, values.new_ones(values.shape[0]))
    return summed / counts.clamp_min(1.0).unsqueeze(-1)


class HexgtNetwork(nn.Module):
    """Dynamic graph model interface with a placeholder body (Phase 0 stub).

    `forward` takes the packed-graph batch dict and returns all training heads.
    `forward_policy_value` returns only the two outputs Rust search needs.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int = NODE_FEATURE_DIM,
        token_dim: int = 128,
        gnn_layers: int = 3,
        ctx_layers: int = 3,
        attention_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        short_term_value_horizons: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.token_dim = int(token_dim)
        self.gnn_layers = int(gnn_layers)
        self.ctx_layers = int(ctx_layers)
        self.attention_heads = int(attention_heads)
        self.ffn_dim = int(ffn_dim)
        self.short_term_value_horizons = tuple(int(h) for h in short_term_value_horizons)

        # Placeholder per-node encoder (Phase 3 replaces with typed GNN + ctx tf).
        self.node_in = nn.Linear(self.node_feature_dim, self.token_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        self.policy_head = nn.Linear(self.token_dim, 1)
        self.opp_policy_head = nn.Linear(self.token_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, VALUE_BINS),
        )
        self.short_term_value_heads = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.Linear(self.token_dim, self.token_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.token_dim, VALUE_BINS),
                )
                for h in self.short_term_value_horizons
            }
        )

    # -- internals -------------------------------------------------------------

    def _encode_nodes(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return (Ntot, token_dim) node embeddings. Phase-3 swaps the body here."""

        node_feat = batch["node_feat"]
        self._validate_batch(batch)
        h = self.node_in(node_feat)
        return h + self.encoder(h)

    def _heads(
        self, batch: Mapping[str, torch.Tensor], node_emb: torch.Tensor, *, with_aux: bool
    ) -> dict[str, torch.Tensor]:
        num_graphs = int(batch["num_graphs"])
        candidate_index = batch["candidate_index"].to(dtype=torch.long)
        cand_emb = node_emb.index_select(0, candidate_index)
        graph_emb = _segment_mean(node_emb, batch["node_graph"], num_graphs)

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
        node_emb = self._encode_nodes(batch)
        return self._heads(batch, node_emb, with_aux=True)

    def forward_policy_value(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inference-only forward path for search (policy + value only)."""

        node_emb = self._encode_nodes(batch)
        return self._heads(batch, node_emb, with_aux=False)

    def _validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        node_feat = batch["node_feat"]
        if node_feat.ndim != 2 or node_feat.shape[1] != self.node_feature_dim:
            raise ValueError(
                f"node_feat must be (Ntot, {self.node_feature_dim}), got {tuple(node_feat.shape)}"
            )
        node_type = batch.get("node_type")
        if node_type is not None and bool(((node_type < 0) | (node_type >= NUM_NODE_TYPES)).any().item()):
            raise ValueError("node_type out of range")
