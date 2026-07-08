"""Pure-PyTorch reimplementation of hexo-strix's HeXONet (GINE + JK-cat).

This is a faithful, dependency-light port of the ``HeXONet`` graph network from
https://github.com/SootyOwl/hexo-strix (``hexo-a0/src/hexo_a0/model.py``). It
exists so a hexo-strix checkpoint can be evaluated as an opponent inside this
repo WITHOUT installing ``torch_geometric`` (unavailable on this machine's
Python 3.14 / torch 2.10) or compiling hexo-strix's Rust extension.

Only the configuration used by the shipped checkpoint is supported:
``conv_type="gine"``, ``graph_type="axis"``, ``pre_norm=True``,
``use_jk=True``, ``jk_mode="cat"``, ``use_layer_scale=False``,
``axis_relational=False`` (the legacy 5-dim edge_attr schema). Unsupported
combinations raise rather than silently mis-load.

Module + parameter names deliberately match the upstream state dict exactly
(``representation.input_proj``, ``representation.edge_proj``,
``representation.convs.{i}.nn.{0,2}``, ``representation.convs.{i}.lin``,
``representation.convs.{i}.eps``, ``representation.norms.{i}``,
``representation.final_norm``, ``policy_head.mlp.{0,2}``,
``value_head.mlp.{0,2}``) so ``load_state_dict(strict=True)`` succeeds after
stripping any ``_orig_mod.`` torch.compile prefix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def node_feature_dim(*, relative_stones: bool, threat_features: bool) -> int:
    """Input node-feature width for the legacy (non-lean) axis layout.

    Base is 8 (absolute: p1,p2,empty,to_move,moves,norm_q,norm_r,inv_dist) or 7
    (relative: own,opp,empty,moves,norm_q,norm_r,inv_dist; to_move dropped), plus
    4 threat dims when enabled. Matches ``hexo_a0.config.node_feature_dim``.
    """
    base = 7 if relative_stones else 8
    return base + (4 if threat_features else 0)


class GINEConvLite(nn.Module):
    """Minimal GINEConv matching torch_geometric's semantics for this net.

    Reproduces ``torch_geometric.nn.GINEConv(nn=Sequential(Linear, ReLU,
    Linear), edge_dim=hidden)`` with ``train_eps=False`` and the default add
    aggregation:

        m_e   = ReLU(x[src_e] + lin(edge_attr_e))
        agg_i = sum_{e: dst_e == i} m_e
        out_i = nn((1 + eps) * x_i + agg_i)

    ``edge_index`` is ``[src, dst]`` (PyG convention: messages flow src->dst,
    aggregated at dst), matching hexo-strix's ``edge_src``/``edge_dst`` arrays.
    ``eps`` is a persistent buffer of shape (1,) so the checkpoint's
    ``convs.{i}.eps`` tensor loads directly.
    """

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.lin = nn.Linear(edge_dim, hidden)
        self.register_buffer("eps", torch.zeros(1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        messages = (x.index_select(0, src) + self.lin(edge_attr)).relu()
        agg = x.new_zeros(x.shape)
        agg.index_add_(0, dst, messages)
        out = agg + (1.0 + self.eps) * x
        return self.nn(out)


class RepresentationNetwork(nn.Module):
    """GINE trunk with pre-norm residual blocks and JK-cat aggregation.

    Faithful to hexo-strix ``RepresentationNetwork`` for the checkpoint's
    config. ``edge_proj`` projects the 5-dim edge_attr to ``hidden`` once and
    reuses it across layers; each GINEConv additionally applies its own
    ``lin`` (128->128) inside message construction (the intentional double edge
    projection). With ``jk_mode="cat"`` the shared ``final_norm`` (LayerNorm of
    width ``hidden``) is applied to EACH per-layer output before concatenation,
    so the head input width is ``num_layers * hidden``.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        in_dim: int,
        edge_in_dim: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GINEConvLite(hidden_dim, edge_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        # jk_mode="cat": heads see num_layers * hidden_dim.
        self.output_dim = num_layers * hidden_dim

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.input_proj(x)
        projected_edge_attr = self.edge_proj(edge_attr)
        hs: list[Tensor] = []
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = norm(x)  # pre_norm
            x = conv(x, edge_index, projected_edge_attr)
            x = x + residual
            x = self.activation(x)
            hs.append(x)
        # jk_mode="cat": final_norm applied to each layer output, then concat.
        hs = [self.final_norm(h) for h in hs]
        return torch.cat(hs, dim=-1)  # (N, num_layers * hidden_dim)


class PolicyHead(nn.Module):
    """MLP scoring legal-move nodes -> one logit per legal node."""

    def __init__(self, in_dim: int, policy_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, policy_hidden), nn.ReLU(), nn.Linear(policy_hidden, 1)
        )

    def forward(self, embeddings: Tensor, legal_mask: Tensor) -> Tensor:
        legal = embeddings[legal_mask]
        return self.mlp(legal).squeeze(-1)


class ValueHead(nn.Module):
    """Mean-pool over stone nodes, then MLP+Tanh -> scalar in [-1, 1]."""

    def __init__(self, in_dim: int, value_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, embeddings: Tensor, stone_mask: Tensor | None = None) -> Tensor:
        if stone_mask is not None and bool(stone_mask.any()):
            pooled = embeddings[stone_mask].mean(dim=0)
        else:
            pooled = embeddings.mean(dim=0)
        return self.mlp(pooled).squeeze(-1)


class HeXONet(nn.Module):
    """Full HeXO AlphaZero net: representation + policy + value heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 4,
        policy_hidden: int = 128,
        value_hidden: int = 32,
        relative_stones: bool = True,
        threat_features: bool = True,
    ) -> None:
        super().__init__()
        in_dim = node_feature_dim(
            relative_stones=relative_stones, threat_features=threat_features
        )
        self.representation = RepresentationNetwork(
            hidden_dim=hidden_dim, num_layers=num_layers, in_dim=in_dim
        )
        head_in = self.representation.output_dim
        self.policy_head = PolicyHead(head_in, policy_hidden)
        self.value_head = ValueHead(head_in, value_hidden)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor | None = None,
        *,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        embeddings = self.representation(x, edge_index, edge_attr)
        logits = self.policy_head(embeddings, legal_mask)
        value = self.value_head(embeddings, stone_mask)
        return logits, value


_SUPPORTED = {
    "conv_type": "gine",
    "graph_type": "axis",
    "jk_mode": "cat",
}


def build_from_model_config(model_config: dict) -> HeXONet:
    """Construct a HeXONet from a checkpoint's embedded ``model_config`` dict.

    Validates that the config matches the supported (checkpoint) architecture
    and raises ValueError otherwise, rather than silently mis-loading.
    """
    for key, expected in _SUPPORTED.items():
        got = model_config.get(key)
        if got != expected:
            raise ValueError(
                f"hexo_strix port only supports {key}={expected!r}, got {got!r}. "
                "This checkpoint uses a different architecture variant."
            )
    if not model_config.get("use_jk", False):
        raise ValueError("hexo_strix port requires use_jk=True (jk_mode='cat').")
    if not model_config.get("pre_norm", False):
        raise ValueError("hexo_strix port requires pre_norm=True.")
    if model_config.get("use_layer_scale", False):
        raise ValueError("hexo_strix port does not support use_layer_scale=True.")
    if model_config.get("axis_relational", False):
        raise ValueError("hexo_strix port does not support axis_relational=True.")
    return HeXONet(
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        policy_hidden=int(model_config.get("policy_hidden", model_config["hidden_dim"])),
        value_hidden=int(model_config.get("value_hidden", 32)),
        relative_stones=bool(model_config.get("relative_stone_encoding", True)),
        threat_features=bool(model_config.get("threat_features", True)),
    )
