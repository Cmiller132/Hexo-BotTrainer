"""PyTorch architecture for hexgnn — a stripped-down hexgt: typed GNN trunk only.

hexgnn is a deliberate simplification of hexgt (Model 2/3): it keeps hexgt's
relational message-passing GNN trunk and produces the SAME packed-graph batch
contract (`collate.py`), but DROPS the per-graph context transformer
(`GraphTransformerLayer`) and the short-term-value (STV) lookahead heads. The GNN
message-passing trunk feeds the heads directly. Heads are policy, value, and
opponent policy ONLY:

- ``policy``: one logit per CANDIDATE node, dynamic length (Ctot,).
- ``value``: a 65-bin scalar value distribution per graph (G, 65).
- ``opp_policy``: per-candidate aux logits (Ctot,).

Body (§6.3): typed RELATIONAL message passing over the sparse-rewrite graph
(constants.py): only ADJACENCY (+ per-edge `edge_dir`) and RECENCY edges exist —
the window-hub nodes/edges were removed and line/co-linearity information now
rides on per-node window-count FEATURES (the F_CAND_*/F_STONE_* slots) built by
the Rust candidates.rs builder. All inputs are D6-INVARIANT (features.py) and
all ops are permutation-equivariant, so the model stays D6-invariant by
construction (the equivariance test passes exactly; no augmentation), exactly
as in hexgt.

VALUE READOUT DECISION (transformer-free) — KEEP the PMA pool.
Without the context transformer there is no attention mixing into the SIDE hub,
so the value readout MUST pool globally over the post-GNN node embeddings rather
than read the SIDE token alone. The choice was PMA (Set-Transformer k-seed
scatter-softmax pool over all nodes) vs a fixed mean/max pool. We KEEP the PMA
pool, for these reasons:
  * It IS the global node pool the value head needs — it operates on the post-GNN
    node set and never depended on the transformer, so it transfers unchanged.
  * It is already proven, tested, D6-invariant, and torch.compile-clean in this
    codebase (varlen segment-softmax, no ``.item()`` graph-break; ~5x faster than
    the padded MHA path it replaced — HEXGT_PMA_VALUE_HEAD_PLAN.md).
  * It is a strict generalization of mean+max (a seed can learn uniform ~= mean
    or a sharp ~= soft-max attention), so it can only help the value head's
    defensive calibration; the cost is one small self-contained module.
The "simplicity" the owner wants is in the OVERALL architecture (no transformer,
no STV) — the PMA is a single ~k·D-wide module, not a complexity burden, and
swapping it for mean/max would under-power the ONLY global readout the value head
has. The SIDE hub is still concatenated (``value_head_use_side`` default True),
but NOTE (sparse rewrite): EDGE_TYPE_CONTEXT was RETIRED (constants.py) and the
SIDE node is now EDGE-ISOLATED — message passing no longer mixes board state into
it. Its embedding is just norm(node_in(side_features)), i.e. a learned projection
of the global scalars (phase / move number / stone counts) riding on the SIDE
node's own features; the original "connected to every node via the CONTEXT hub"
rationale no longer holds, the global mixing now comes solely from the PMA pool.

Callers: plugin.build_model, scripts/_rl_train_hexgnn.py / _pretrain_hexgnn.py
(direct construction + the zero-init resume grafts at the bottom of this file),
inference.py (forward_policy_value + precompute_side_rows), and the
tests/test_hexgnn_{model,steerable,value_readout,compile}.py suite.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


# --- D6 steerable direction basis (spec §4.5) --------------------------------
# The 6 hex axial unit directions and their 2x2 outer products e_d e_d^T, stored
# as the 3 independent symmetric components [xx, yy, xy]. These are the ONLY place
# absolute direction enters; under a D6 element the directions permute and each
# outer product maps e_d e_d^T -> R_g (e_d e_d^T) R_g^T, so any O(2)-invariant of
# the accumulated tensor T (trace, det) is unchanged => exact D6-invariance with
# NO per-direction weights and NO group lifting (constant tensors, ~zero cost).
_HEX_DIRS_AXIAL = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def _axial_to_xy(q: float, r: float) -> tuple[float, float]:
    # axial -> cartesian on the hex plane (each of the 6 unit steps has length 1,
    # 60 degrees apart), so D6 acts as a genuine O(2) rotation/reflection.
    return (q + r / 2.0, r * math.sqrt(3.0) / 2.0)


_DIR_XY = [_axial_to_xy(q, r) for (q, r) in _HEX_DIRS_AXIAL]
# (6, 3): [e_x*e_x, e_y*e_y, e_x*e_y] per direction.
_DIR_OUTER = [[x * x, y * y, x * y] for (x, y) in _DIR_XY]


class SteerableTensorChannels(nn.Module):
    """Tied-weight steerable edge feature (spec §4.5, owner option B).

    Accumulates a per-node 2x2 symmetric tensor ``T_v = Σ_{u->v} γ(h_u) · e_d e_d^T``
    over the n_steer channels, where γ is a learned SCALAR gate on the (D6-invariant)
    source features and ``e_d`` is the constant hex-direction unit vector of the edge
    (``edge_dir`` in 0..5; <0 = no direction => no contribution). Returns a
    token_dim vector built ONLY from the O(2)-INVARIANTS of T (trace, det), so the
    whole module is exactly D6-invariant for all 12 group elements while preserving
    line-vs-blob discrimination (T is rank-1/anisotropic for collinear neighbours,
    isotropic for a blob; a 2nd moment does not cancel for a straight line). No
    per-direction matrices, no group lifting -> ~zero forward cost vs untied weights.
    """

    def __init__(self, dim: int, n_steer: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_steer = int(n_steer)
        self.gate = nn.Linear(self.dim, self.n_steer)
        self.out = nn.Linear(2 * self.n_steer, self.dim)  # (tr, det) per channel
        self.register_buffer("dir_outer", torch.tensor(_DIR_OUTER, dtype=torch.float32))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_dir: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        if edge_index.shape[1] == 0:
            return h.new_zeros(n, self.dim)
        src, dst = edge_index[0], edge_index[1]
        gate = self.gate(h)                       # (N, n_steer) from invariant feats
        g_e = gate.index_select(0, src)           # (E, n_steer)
        ed = edge_dir.to(torch.long)
        valid = (ed >= 0).to(h.dtype)             # (E,)
        outer = self.dir_outer.to(h.dtype).index_select(0, ed.clamp_min(0))  # (E, 3)
        contrib = (g_e * valid.unsqueeze(-1)).unsqueeze(-1) * outer.unsqueeze(1)  # (E, n_steer, 3)
        T = h.new_zeros(n, self.n_steer, 3).index_add_(0, dst, contrib.to(h.dtype))
        txx, tyy, txy = T[..., 0], T[..., 1], T[..., 2]
        tr = txx + tyy
        det = txx * tyy - txy * txy
        inv = torch.stack([tr, det], dim=-1).reshape(n, 2 * self.n_steer)  # O(2)-invariants
        return self.out(inv)

from .constants import (
    DEFAULT_VALUE_PMA_SEEDS,
    EDGE_ATTR_DIM,
    NEW_FEATURE_SLOTS_V2,
    NODE_FEATURE_DIM,
    NODE_TYPE_SIDE,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    VALUE_BINS,
)


# Value readout = [SIDE hub | PMA_k pooled vectors], each token_dim wide, so the
# value MLP first Linear reads (1 + value_pma_seeds) * token_dim (with the SIDE
# block FIRST). With k=2 this is 3*token_dim.
VALUE_READOUT_MULT = 1 + DEFAULT_VALUE_PMA_SEEDS  # == 3 for k=2 (back-compat alias)


class RelationalMessagePassing(nn.Module):
    """Typed message passing: m_{j->i} = relu(W_{type} h_j + edge_proj(attr)).

    Per-edge-type linear transform via a (num_edge_types, dim, dim) weight,
    applied through an einsum (no per-edge dense materialization). Aggregated by
    mean over incoming edges, residual + LayerNorm update. Copied verbatim from
    hexgt — this is the shared GNN trunk.
    """

    def __init__(self, dim: int, num_edge_types: int, edge_attr_dim: int, *, steerable: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.num_edge_types = num_edge_types
        self.weight = nn.Parameter(torch.empty(num_edge_types, dim, dim))
        self.bias = nn.Parameter(torch.zeros(num_edge_types, dim))
        self.edge_proj = nn.Linear(edge_attr_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.weight)
        # Optional D6 steerable direction channels (spec §4.5). Active only when the
        # batch carries `edge_dir`; their output is built from O(2)-invariants of the
        # accumulated tensor, so the layer stays exactly D6-invariant.
        self.steerable = SteerableTensorChannels(dim, steerable) if steerable > 0 else None

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_dir: torch.Tensor | None = None,
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
        # Under AMP autocast the matmuls emit half while `h` may be float; keep the
        # scatter dtype-consistent (index_add_ requires matching scalar types).
        m = m.to(h.dtype)
        agg = h.new_zeros(n, self.dim).index_add_(0, dst, m)
        counts = h.new_zeros(n).index_add_(0, dst, h.new_ones(dst.shape[0]))
        agg = agg / counts.clamp_min(1.0).unsqueeze(-1)
        update = self.out_proj(agg)
        if self.steerable is not None and edge_dir is not None:
            update = update + self.steerable(h, edge_index, edge_dir).to(h.dtype)
        return self.norm(h + update)


# Key under which the PRECOMPUTED SIDE-hub rows travel in the batch dict. The
# SIDE-row lookup's only data-dependent op (`nonzero` to find the per-graph hub
# rows) graph-breaks `torch.compile`. The eager inference wrapper computes the
# rows ONCE (outside the compiled region) via `precompute_side_rows` and attaches
# them under these keys; `_graph_readout` then consumes the ready tensors so the
# compiled forward has zero such breaks. The uncompiled training / direct
# `forward()` path leaves them absent and computes them inline (unchanged).
#
# NOTE: hexgt also precomputed a padded ATTENTION layout here for its context
# transformer; hexgnn has no transformer, so only the SIDE-hub rows remain.
SIDE_ROW_BATCH_KEYS = ("side_rows", "side_graph")


def precompute_side_rows(node_type: torch.Tensor, node_graph: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the data-dependent SIDE-hub index tensors the compiled forward needs.

    Both are ``nonzero``/``index_select``-based (graph-breaking under
    torch.compile); computing them EAGERLY here and passing them in leaves the
    compiled forward free of every such op. Call OUTSIDE torch.compile."""

    side_rows = (node_type == NODE_TYPE_SIDE).nonzero(as_tuple=True)[0]
    return {
        "side_rows": side_rows,
        "side_graph": node_graph.index_select(0, side_rows),
    }


class PMAValuePool(nn.Module):
    """Set-Transformer PMA pooling for the value readout (HEXGT_PMA_VALUE_HEAD_PLAN).

    ``num_seeds`` learnable seed queries attend (``heads``-head) over ALL of a
    graph's post-GNN node embeddings (keys/values), pooling the set into
    ``num_seeds`` summary vectors: ``PMA_k(Z) = MAB(S, Z)`` (Lee et al. 2019,
    arXiv:1810.00825). A learnable, cross-node / cross-channel soft pooling — a
    strict generalization of a fixed mean+max pool (a seed can learn uniform
    attention ~= mean or a sharp attention ~= soft-max).

    Permutation-INVARIANT over the node set (no positional encoding; the seeds are
    graph-independent learned constants and D6 acts on nodes, not seeds), hence
    D6-invariant — the property the value readout requires.

    Implemented as a VARLEN segment-softmax attention over the packed nodes (no
    padding, no masked-softmax kernel): each graph's `k` seeds attend over exactly
    that graph's nodes, with the softmax reduced WITHIN each graph segment
    (`index_reduce` amax for stability + `index_add` for the exp/normalize). No
    `Tensor.item()` graph-break (so no torch.compile recompiles). Segment
    reductions are order-free, so the pool stays permutation- (hence D6-)
    invariant. Copied verbatim from hexgt.
    """

    def __init__(self, dim: int, heads: int, num_seeds: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.num_seeds = int(num_seeds)
        # seeds ~ small init (mirrors attention query scale); learned, role-free.
        self.seeds = nn.Parameter(torch.randn(self.num_seeds, self.dim) * (self.dim ** -0.5))
        # `attn` holds the q/k/v in-projection + out-projection weights (its
        # forward is not used; the varlen attention below applies them directly).
        self.attn = nn.MultiheadAttention(self.dim, heads, batch_first=True)

    def _project_qkv(self, node_emb: torch.Tensor):
        """Apply nn.MultiheadAttention's packed in-projection to q (seeds), k and v
        (nodes). q,k,v share embed_dim, so `in_proj_weight` is one (3*dim, dim)
        block; bias may be None."""
        w = self.attn.in_proj_weight
        b = self.attn.in_proj_bias
        d = self.dim
        wq, wk, wv = w[:d], w[d : 2 * d], w[2 * d :]
        if b is None:
            bq = bk = bv = None
        else:
            bq, bk, bv = b[:d], b[d : 2 * d], b[2 * d :]
        seeds = self.seeds.to(node_emb.dtype)
        q = F.linear(seeds, wq, bq)        # (k, dim)
        k = F.linear(node_emb, wk, bk)     # (N, dim)
        v = F.linear(node_emb, wv, bv)     # (N, dim)
        return q, k, v

    def forward(self, node_emb: torch.Tensor, node_graph: torch.Tensor, num_graphs: int) -> torch.Tensor:
        h, d = self.heads, self.dim
        dh = d // h
        k = self.num_seeds
        n = node_emb.shape[0]
        g = node_graph.to(torch.long)
        q, kk, vv = self._project_qkv(node_emb)
        qh = q.view(k, h, dh)         # (k, H, dh)
        kh = kk.view(n, h, dh)        # (N, H, dh)
        vh = vv.view(n, h, dh)        # (N, H, dh)
        scale = dh ** -0.5
        # per-node logits against each seed/head: (N, H, k)
        scores = torch.einsum("nhd,khd->nhk", kh, qh) * scale
        # segment-softmax within each graph (stable): subtract per-(g,h,k) max.
        seg_max = scores.new_full((num_graphs, h, k), float("-inf"))
        seg_max.index_reduce_(0, g, scores, "amax", include_self=True)
        ex = torch.exp(scores - seg_max.index_select(0, g))            # (N, H, k)
        denom = ex.new_zeros((num_graphs, h, k)).index_add_(0, g, ex)  # (G, H, k) >= 1
        attn = ex / denom.index_select(0, g)                           # (N, H, k)
        # value-weighted segment sum -> (G, H, k, dh)
        wv = attn.unsqueeze(-1) * vh.unsqueeze(2)                      # (N, H, k, dh)
        out = wv.new_zeros((num_graphs, h, k, dh)).index_add_(0, g, wv)
        out = out.permute(0, 2, 1, 3).reshape(num_graphs, k, d)        # (G, k, dim)
        out = self.attn.out_proj(out)                                  # (G, k, dim)
        return out.reshape(num_graphs, k * d)


class HexgnnNetwork(nn.Module):
    """Dynamic typed GNN trunk (NO transformer) producing policy/value/opp heads."""

    def __init__(
        self,
        *,
        node_feature_dim: int = NODE_FEATURE_DIM,
        token_dim: int = 168,
        gnn_layers: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.0,
        edge_attr_dim: int = EDGE_ATTR_DIM,
        value_pma_seeds: int = DEFAULT_VALUE_PMA_SEEDS,
        value_head_use_side: bool = True,
        steerable_channels: int = 0,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.token_dim = int(token_dim)
        self.gnn_layers = int(gnn_layers)
        self.attention_heads = int(attention_heads)
        self.edge_attr_dim = int(edge_attr_dim)
        self.value_pma_seeds = int(value_pma_seeds)
        self.value_head_use_side = bool(value_head_use_side)
        self.dropout = float(dropout)
        # D6 steerable direction channels (spec §4.5). 0 = off (current behavior);
        # >0 builds each GNN layer with the tied steerable tensor channels, used when
        # the batch carries `edge_dir`.
        self.steerable_channels = int(steerable_channels)

        self.node_in = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.gnn = nn.ModuleList(
            [RelationalMessagePassing(self.token_dim, NUM_EDGE_TYPES, self.edge_attr_dim,
                                      steerable=self.steerable_channels) for _ in range(self.gnn_layers)]
        )

        self.policy_head = nn.Linear(self.token_dim, 1)
        self.opp_policy_head = nn.Linear(self.token_dim, 1)
        # The value head reads a whole-board readout: a Set-Transformer PMA pool
        # (k = value_pma_seeds learned seed queries, attention_heads-head) over ALL
        # post-GNN node embeddings, OPTIONALLY prefixed with the SIDE hub embedding.
        # With `value_head_use_side` (default) the readout is [SIDE | PMA_k], width
        # (1 + k) * token_dim (SIDE block FIRST); when False it is PMA-only, width
        # k * token_dim. See the module docstring for the keep-PMA decision.
        self.value_pool = PMAValuePool(self.token_dim, self.attention_heads, self.value_pma_seeds)
        value_readout_blocks = (1 if self.value_head_use_side else 0) + self.value_pma_seeds
        self.value_readout_blocks = value_readout_blocks
        self.value_head = nn.Sequential(
            nn.Linear(value_readout_blocks * self.token_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, VALUE_BINS),
        )

    # -- internals -------------------------------------------------------------

    def _encode_nodes(self, batch: Mapping[str, torch.Tensor], *, validate_range: bool = True) -> torch.Tensor:
        self._validate_batch(batch, validate_range=validate_range)
        h = self.node_in(batch["node_feat"])
        edge_index = batch["edge_index"]
        edge_type = batch.get("edge_type")
        edge_attr = batch.get("edge_attr")
        if edge_type is None:
            edge_type = h.new_zeros(edge_index.shape[1], dtype=torch.long)
        if edge_attr is None:
            edge_attr = h.new_zeros(edge_index.shape[1], self.edge_attr_dim)
        edge_dir = batch.get("edge_dir")  # (E,) hex-direction index in 0..5, <0 = none
        for layer in self.gnn:
            h = layer(h, edge_index, edge_type, edge_attr, edge_dir)
        return h

    def _graph_readout(self, batch: Mapping[str, torch.Tensor], node_emb: torch.Tensor) -> torch.Tensor:
        """Per-graph readout from the SIDE hub node (one per graph)."""

        num_graphs = int(batch["num_graphs"])
        # Use the PRECOMPUTED side-hub rows (attached eagerly by the inference
        # wrapper) when present — so the compiled forward has no nonzero here — else
        # compute them inline (training / direct forward()).
        if "side_rows" in batch:
            side_rows = batch["side_rows"]
            side_graph = batch["side_graph"]
        else:
            side_rows = (batch["node_type"] == NODE_TYPE_SIDE).nonzero(as_tuple=True)[0]
            side_graph = batch["node_graph"].index_select(0, side_rows)
        readout = node_emb.new_zeros(num_graphs, self.token_dim)
        # last side node per graph (there is exactly one)
        readout[side_graph] = node_emb.index_select(0, side_rows)
        return readout

    def _value_readout(self, batch: Mapping[str, torch.Tensor], node_emb: torch.Tensor) -> torch.Tensor:
        """Whole-board value readout: [SIDE hub | PMA_k pool] (G, (1+k)*D), or
        PMA-only (G, k*D) when ``value_head_use_side`` is False."""

        num_graphs = int(batch["num_graphs"])
        pooled = self.value_pool(node_emb, batch["node_graph"], num_graphs)
        if not self.value_head_use_side:
            return pooled
        side = self._graph_readout(batch, node_emb)
        return torch.cat([side, pooled], dim=-1)

    def _heads(self, batch: Mapping[str, torch.Tensor], node_emb: torch.Tensor, *, with_aux: bool) -> dict[str, torch.Tensor]:
        candidate_index = batch["candidate_index"].to(dtype=torch.long)
        cand_emb = node_emb.index_select(0, candidate_index)
        value_emb = self._value_readout(batch, node_emb)
        outputs: dict[str, torch.Tensor] = {
            "policy": self.policy_head(cand_emb).squeeze(-1),
            "value": self.value_head(value_emb),
        }
        if with_aux:
            outputs["opp_policy"] = self.opp_policy_head(cand_emb).squeeze(-1)
        return outputs

    # -- public forward --------------------------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._heads(batch, self._encode_nodes(batch), with_aux=True)

    def forward_policy_value(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inference-only forward path for search (policy + value only).

        Skips the node-type RANGE validation (an ``.any().item()`` GPU->CPU sync
        that graph-breaks ``torch.compile`` and serializes every search forward).
        The cheap shape check is kept; the range check is redundant in search —
        the Rust featurizer always emits valid node types, and the trainer's
        ``forward`` still runs the full check. This is the throughput-critical
        path (self-play + eval), so the sync is removed here only."""

        return self._heads(batch, self._encode_nodes(batch, validate_range=False), with_aux=False)

    def _validate_batch(self, batch: Mapping[str, torch.Tensor], *, validate_range: bool = True) -> None:
        node_feat = batch["node_feat"]
        if node_feat.ndim != 2 or node_feat.shape[1] != self.node_feature_dim:
            raise ValueError(
                f"node_feat must be (Ntot, {self.node_feature_dim}), got {tuple(node_feat.shape)}"
            )
        node_type = batch.get("node_type")
        if validate_range and node_type is not None and node_type.numel() and bool(((node_type < 0) | (node_type >= NUM_NODE_TYPES)).any().item()):
            raise ValueError("node_type out of range")


def zero_init_expanded_feature_columns(
    model: HexgnnNetwork, slots: tuple[int, ...] = NEW_FEATURE_SLOTS_V2
) -> list[int]:
    """ZERO-INIT LAYER-EXPANSION for newly-activated node-feature slots.

    hexgnn routes every node type through ONE shared input projection
    (``model.node_in``); a feature lives in a fixed COLUMN of that projection's
    first ``Linear``. When a previously-always-zero slot is activated (feature
    schema bump), the column's weights still hold their original random init.
    Zeroing exactly those columns makes the projection — and therefore the whole
    forward — produce output IDENTICAL to the pre-expansion checkpoint on the
    first step after resume, while leaving every trained weight untouched.

    Call this ONCE, right after loading a pre-expansion checkpoint's weights.
    Returns the zeroed column indices (for logging / tests). Identical to hexgt.
    """

    first_linear = model.node_in[0]
    in_features = first_linear.weight.shape[1]
    valid = [int(s) for s in slots if 0 <= int(s) < in_features]
    with torch.no_grad():
        for slot in valid:
            first_linear.weight[:, slot].zero_()
    return valid


def expand_value_readout_columns(
    model: HexgnnNetwork,
    state_dict: dict,
    *,
    key: str = "value_head.0.weight",
) -> bool:
    """ZERO-INIT EXPANSION for the global-pooled value readout.

    A pre-expansion checkpoint's value-head first ``Linear`` read ONLY the SIDE
    hub: weight shape ``(token_dim, token_dim)``. The current head reads
    ``[SIDE | PMA_k]``: ``((1+k) * token_dim)``. This rewrites the checkpoint
    tensor IN PLACE in ``state_dict`` into the wider shape — the old weight in the
    leading SIDE block ``[:, :token_dim]`` and ZEROS in the PMA blocks — so the
    loaded head produces output IDENTICAL to the pre-expansion checkpoint on the
    first step (the pooled features contribute 0), while training then learns the
    pooled signal from zero.

    No-op (returns ``False``) when the checkpoint already carries the wide shape.
    Call BEFORE ``load_state_dict(..., strict=False)``. Returns ``True`` if it
    expanded. (Retained from hexgt for forward-compatibility / tests; a fresh
    hexgnn lineage will normally never trip it.)
    """

    w = state_dict.get(key)
    if w is None:
        return False
    target = model.value_head[0].weight
    if tuple(w.shape) == tuple(target.shape):
        return False
    d = model.token_dim
    if tuple(w.shape) != (int(target.shape[0]), d):
        raise ValueError(
            f"cannot expand {key}: checkpoint shape {tuple(w.shape)} is neither the "
            f"pre-pool SIDE-only shape ({int(target.shape[0])}, {d}) nor the target "
            f"{tuple(target.shape)}"
        )
    new_w = w.new_zeros(tuple(target.shape))
    new_w[:, :d] = w  # SIDE-hub block first; PMA blocks stay zero
    state_dict[key] = new_w
    return True
