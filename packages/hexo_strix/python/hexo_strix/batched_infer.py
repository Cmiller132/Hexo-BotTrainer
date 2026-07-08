"""Batched inference for the ported HeXONet over a list of axis graphs.

The Rust Gumbel MCTS hands the network callback (``eval_fn``) up to ``m_actions``
leaves per call (the sequential-halving rounds), and a cross-game batch server
(:mod:`hexo_strix.batch_server`) coalesces leaves across concurrent games. Both
want ONE GPU forward over many graphs instead of a Python loop of single-graph
forwards.

:func:`batched_eval` builds the disconnected union of the graphs (concatenate
node features, offset each graph's ``edge_index`` by its node base, tag every
node with its graph via a ``batch`` vector), runs the shared representation once,
then splits the per-graph outputs back out:

  * policy — ``policy_head.mlp`` over the legal nodes, split per graph by each
    graph's legal-node count, preserving the ``legal_moves()`` order the Rust
    MCTS expects.
  * value — a per-graph mean over that graph's STONE nodes (segment mean via
    ``index_add_`` on the ``batch`` vector), then ``value_head.mlp`` + tanh —
    exactly :class:`hexo_strix.model.ValueHead` done G-at-once. A graph with no
    stone nodes falls back to a mean over all its nodes (matching the
    single-graph ``else`` branch).

The result is numerically equal to a per-graph forward up to floating-point
reduction order (the batched matmul / cuBLAS kernel differs from the single-graph
one); moves are effectively identical but not bit-guaranteed. That is acceptable
for an eval opponent (the model-level faithfulness to upstream is proven
separately in the port validation).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import numpy as np
import torch
from torch import Tensor

# One graph as CPU tensors, in the shape hexo_strix.mcts_player builds:
# (x, edge_index, edge_attr, legal_mask, stone_mask).
GraphTensors = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def build_axis_graph_tensors(
    hexo_rs: Any,
    state: Any,
    *,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> GraphTensors:
    """Build one axis graph (CPU tensors) from a ``hexo_rs`` GameState.

    Uses hexo-strix's COMPILED Rust builder (``game_to_axis_graph_raw``); output
    is numerically identical to the pure-Python ``hexo_strix.graph`` builder and
    the legal-node order equals ``state.legal_moves()``. Extracted from
    ``StrixMctsPlayer._graph_from_gamestate`` so both the serial and batched
    paths build graphs identically.

    The raw fields come back as flat Python lists (features/edge_attr:
    ``float32``, edge_src/edge_dst: ``int``, masks: ``bool``). We go through
    ``np.asarray(dtype=...)`` + ``torch.from_numpy`` instead of ``torch.tensor``:
    the list -> buffer conversion runs in one C loop with the GIL released,
    instead of ``torch.tensor`` walking each element under the GIL — this build
    was the GIL-bound hot spot (~134 ms/decide at 256 sims), serializing against
    the other games sharing the batch server. ``np.asarray`` on a list always
    allocates a fresh writable array this function owns, so ``from_numpy`` shares
    it with no "array is not writable" warning; the ``.copy()`` guard only trips
    if a future ``hexo_rs`` returns a read-only buffer (then the shared tensor
    would be a write hazard).
    """

    raw = hexo_rs.game_to_axis_graph_raw(
        state,
        prune_empty_edges=prune_empty_edges,
        threat_features=threat_features,
        relative_stones=relative_stones,
    )

    def _arr(seq: Any, dtype: Any) -> np.ndarray:
        a = np.asarray(seq, dtype=dtype)
        # Only copies if hexo_rs handed back a read-only buffer; the list case
        # already produced a fresh writable array, so this is a no-op there.
        return a if a.flags.writeable else a.copy()

    n = raw["num_nodes"]
    x = torch.from_numpy(_arr(raw["features"], np.float32)).reshape(n, -1)
    esrc, edst = raw["edge_src"], raw["edge_dst"]
    if esrc:
        # Stack the two index rows into (2, E) int64, matching torch.tensor([esrc, edst]).
        edge_index = torch.from_numpy(
            np.stack((_arr(esrc, np.int64), _arr(edst, np.int64)))
        )
        edge_attr = torch.from_numpy(_arr(raw["edge_attr"], np.float32)).reshape(len(esrc), 5)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
    legal_mask = torch.from_numpy(_arr(raw["legal_mask"], np.bool_))
    stone_mask = torch.from_numpy(_arr(raw["stone_mask"], np.bool_))
    return x, edge_index, edge_attr, legal_mask, stone_mask


@torch.no_grad()
def batched_eval(
    model: Any, graphs: list[GraphTensors], device: str | torch.device
) -> tuple[list[list[float]], list[float]]:
    """Run ``model`` over ``graphs`` in one forward; return per-graph outputs.

    Returns ``(logits_per_graph, values)`` where ``logits_per_graph[i]`` is the
    list of legal-move logits for graph ``i`` (in its ``legal_moves()`` order)
    and ``values[i]`` its scalar value in [-1, 1] — the exact shape the Rust
    ``eval_fn`` contract expects. Empty ``graphs`` -> ``([], [])``.
    """

    if not graphs:
        return [], []

    xs, eis, eas, legals, stones = zip(*graphs)
    node_counts = [int(x.shape[0]) for x in xs]
    legal_counts = [int(lm.sum().item()) for lm in legals]
    g = len(graphs)

    x = torch.cat(xs, 0).to(device, non_blocking=True)
    edge_attr = torch.cat(eas, 0).to(device, non_blocking=True)

    # Offset each graph's edges by its node base so the union stays disconnected.
    bases = [0, *accumulate(node_counts)][:g]
    ei_parts = [ei + base for ei, base in zip(eis, bases) if ei.numel()]
    if ei_parts:
        edge_index = torch.cat(ei_parts, 1).to(device, non_blocking=True)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)

    legal_mask = torch.cat(legals, 0).to(device, non_blocking=True)
    stone_mask = torch.cat(stones, 0).to(device, non_blocking=True)
    # Node -> graph id.
    batch = torch.repeat_interleave(
        torch.arange(g, device=device),
        torch.tensor(node_counts, device=device),
    )

    emb = model.representation(x, edge_index, edge_attr)  # (sum_N, out_dim)
    out_dim = emb.shape[1]

    # --- policy: mlp over legal nodes, split per graph ---
    logits_flat = model.policy_head.mlp(emb[legal_mask]).squeeze(-1)  # (sum_legal,)
    logits_flat = logits_flat.float().cpu()

    # --- value: per-graph mean over stone nodes, then mlp + tanh ---
    stone_batch = batch[stone_mask]
    sums = emb.new_zeros(g, out_dim).index_add_(0, stone_batch, emb[stone_mask])
    counts = emb.new_zeros(g).index_add_(
        0, stone_batch, torch.ones(stone_batch.shape[0], device=device, dtype=emb.dtype)
    )
    # Graphs with no stone nodes fall back to a mean over ALL their nodes (the
    # single-graph ValueHead ``else`` branch).
    no_stone = counts == 0
    if bool(no_stone.any()):
        all_sums = emb.new_zeros(g, out_dim).index_add_(0, batch, emb)
        all_counts = emb.new_zeros(g).index_add_(
            0, batch, torch.ones(emb.shape[0], device=device, dtype=emb.dtype)
        )
        sums[no_stone] = all_sums[no_stone]
        counts[no_stone] = all_counts[no_stone]
    pooled = sums / counts.clamp(min=1).unsqueeze(1)
    values = model.value_head.mlp(pooled).squeeze(-1).float().cpu()  # (g,)

    logits_per_graph: list[list[float]] = []
    pos = 0
    for lc in legal_counts:
        logits_per_graph.append(logits_flat[pos : pos + lc].tolist())
        pos += lc
    return logits_per_graph, values.tolist()


# ===========================================================================
# Fast path: build the whole round's disconnected union in ONE Rust call.
# ===========================================================================
#
# ``batched_eval`` above builds one graph per leaf (``build_axis_graph_tensors``)
# and re-assembles the union in Python. Profiling the eval hot path showed that
# per-leaf build dominates: the Rust ``game_to_axis_graph_raw`` call boxes ~13k
# Python float/int objects per leaf (a 327-node / 1442-edge graph), and the numpy
# re-parse walks them again. ``hexo_rs.game_states_to_axis_batch_bytes`` collates
# the WHOLE round (rayon-parallel) into flat native-endian byte buffers already in
# disconnected-union layout (offset ``edge_index``, a per-node ``batch`` vector,
# ``legal_idx``/``stone_idx`` and ``legal_counts`` split metadata), ready for
# zero-copy ``torch.frombuffer`` — no Python-list materialization on either side.
#
# The graphs it produces are byte-identical to the per-leaf builder (verified:
# features / edge_attr / edge sets / masks max|diff| = 0), and the resulting
# moves are identical, so this is a pure speedup (2.6-6.6x on the build alone,
# ~2x end-to-end per round) with no change to Strix's play.


@dataclass
class RoundBatch:
    """A round's leaves as one pre-batched disconnected-union graph (CPU tensors).

    All index tensors are already offset into the concatenated node space, so a
    ``RoundBatch`` is fed straight to :func:`batched_eval_round` with no further
    assembly. ``legal_idx`` lists the legal nodes in each graph's
    ``legal_moves()`` order (NOT ascending node order), split per graph by
    ``legal_counts``; ``stone_idx`` / ``stone_batch`` drive the value pooling.
    """

    x: Tensor            # (Ntot, n_feat) f32
    edge_index: Tensor   # (2, Etot) i64, union-offset
    edge_attr: Tensor    # (Etot, 5) f32
    legal_idx: Tensor    # (Ltot,) i64, node indices in legal_moves() order
    legal_counts: list[int]
    stone_idx: Tensor    # (Stot,) i64, stone-node indices
    stone_batch: Tensor  # (Stot,) i64, graph id per stone node
    batch: Tensor        # (Ntot,) i64, graph id per node (no-stone value fallback)
    num_graphs: int
    num_nodes: int


_EMPTY_ROUND = RoundBatch(
    x=torch.zeros((0, 0), dtype=torch.float32),
    edge_index=torch.zeros((2, 0), dtype=torch.int64),
    edge_attr=torch.zeros((0, 5), dtype=torch.float32),
    legal_idx=torch.zeros((0,), dtype=torch.int64),
    legal_counts=[],
    stone_idx=torch.zeros((0,), dtype=torch.int64),
    stone_batch=torch.zeros((0,), dtype=torch.int64),
    batch=torch.zeros((0,), dtype=torch.int64),
    num_graphs=0,
    num_nodes=0,
)


def _frombuf(b: Any, dtype: torch.dtype) -> Tensor:
    # ``bytearray`` gives a writable buffer so ``frombuffer`` shares it with no
    # "array is not writable" warning; the copy is a single C memcpy (~µs for the
    # ~100 KB/round buffers), negligible against the per-element boxing it avoids.
    return torch.frombuffer(bytearray(b), dtype=dtype)


def build_axis_round(
    hexo_rs: Any,
    states: list[Any],
    *,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> RoundBatch:
    """Build a whole round's leaves as one :class:`RoundBatch` (zero-copy).

    One ``hexo_rs.game_states_to_axis_batch_bytes`` call collates every leaf's
    axis graph into flat byte buffers already in disconnected-union layout; we
    wrap them with ``torch.frombuffer``. Numerically identical to a per-leaf
    :func:`build_axis_graph_tensors` loop, but far cheaper (see module note).
    """

    if not states:
        return _EMPTY_ROUND
    d = hexo_rs.game_states_to_axis_batch_bytes(
        states,
        prune_empty_edges=prune_empty_edges,
        threat_features=threat_features,
        relative_stones=relative_stones,
    )
    nf = int(d["n_feat"])
    g = int(d["num_graphs"])
    x = _frombuf(d["x"], torch.float32).reshape(-1, nf)
    esrc = _frombuf(d["edge_src"], torch.int64)
    if esrc.numel():
        edge_index = torch.stack((esrc, _frombuf(d["edge_dst"], torch.int64)))
        edge_attr = _frombuf(d["edge_attr"], torch.float32).reshape(-1, 5)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
    return RoundBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        legal_idx=_frombuf(d["legal_idx"], torch.int64),
        legal_counts=_frombuf(d["legal_counts"], torch.int64).tolist(),
        stone_idx=_frombuf(d["stone_idx"], torch.int64),
        stone_batch=_frombuf(d["stone_batch"], torch.int64),
        batch=_frombuf(d["batch"], torch.int64),
        num_graphs=g,
        num_nodes=int(x.shape[0]),
    )


def round_from_graph_tensors(graphs: list[GraphTensors]) -> RoundBatch:
    """Assemble a :class:`RoundBatch` from per-graph :data:`GraphTensors`.

    The fallback the batch server uses when handed the legacy per-graph tuples
    (e.g. from tests) instead of a Rust-built round; produces the same union
    layout ``batched_eval_round`` expects.
    """

    if not graphs:
        return _EMPTY_ROUND
    xs, eis, eas, legals, stones = zip(*graphs)
    node_counts = [int(x.shape[0]) for x in xs]
    bases = [0, *accumulate(node_counts)]
    x = torch.cat(xs, 0)
    ei_parts = [ei + b for ei, b in zip(eis, bases) if ei.numel()]
    edge_index = torch.cat(ei_parts, 1) if ei_parts else torch.zeros((2, 0), dtype=torch.int64)
    edge_attr = torch.cat(eas, 0) if any(e.numel() for e in eas) else torch.zeros((0, 5), dtype=torch.float32)
    legal_idx_parts, stone_idx_parts, stone_batch_parts, batch_parts = [], [], [], []
    legal_counts: list[int] = []
    for i, (lm, sm, base) in enumerate(zip(legals, stones, bases)):
        legal_nodes = torch.nonzero(lm, as_tuple=False).flatten()
        legal_idx_parts.append(legal_nodes + base)
        legal_counts.append(int(legal_nodes.numel()))
        stone_nodes = torch.nonzero(sm, as_tuple=False).flatten()
        stone_idx_parts.append(stone_nodes + base)
        stone_batch_parts.append(torch.full((stone_nodes.numel(),), i, dtype=torch.int64))
        batch_parts.append(torch.full((node_counts[i],), i, dtype=torch.int64))
    return RoundBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        legal_idx=torch.cat(legal_idx_parts) if legal_idx_parts else torch.zeros((0,), dtype=torch.int64),
        legal_counts=legal_counts,
        stone_idx=torch.cat(stone_idx_parts) if stone_idx_parts else torch.zeros((0,), dtype=torch.int64),
        stone_batch=torch.cat(stone_batch_parts) if stone_batch_parts else torch.zeros((0,), dtype=torch.int64),
        batch=torch.cat(batch_parts) if batch_parts else torch.zeros((0,), dtype=torch.int64),
        num_graphs=len(graphs),
        num_nodes=int(x.shape[0]),
    )


def concat_rounds(rounds: list[RoundBatch]) -> RoundBatch:
    """Coalesce several :class:`RoundBatch` into one (the batch server's job).

    Offsets each round's node-space indices by the running node base and its
    graph ids by the running graph base, so the union stays disconnected and
    per-graph splits stay correct across games.
    """

    rounds = [r for r in rounds if r.num_graphs]
    if not rounds:
        return _EMPTY_ROUND
    if len(rounds) == 1:
        return rounds[0]
    xs, ei_parts, ea_parts = [], [], []
    legal_idx_parts, stone_idx_parts, stone_batch_parts, batch_parts = [], [], [], []
    legal_counts: list[int] = []
    node_base = 0
    graph_base = 0
    for r in rounds:
        xs.append(r.x)
        if r.edge_index.numel():
            ei_parts.append(r.edge_index + node_base)
            ea_parts.append(r.edge_attr)
        legal_idx_parts.append(r.legal_idx + node_base)
        legal_counts.extend(r.legal_counts)
        stone_idx_parts.append(r.stone_idx + node_base)
        stone_batch_parts.append(r.stone_batch + graph_base)
        batch_parts.append(r.batch + graph_base)
        node_base += r.num_nodes
        graph_base += r.num_graphs
    return RoundBatch(
        x=torch.cat(xs, 0),
        edge_index=torch.cat(ei_parts, 1) if ei_parts else torch.zeros((2, 0), dtype=torch.int64),
        edge_attr=torch.cat(ea_parts, 0) if ea_parts else torch.zeros((0, 5), dtype=torch.float32),
        legal_idx=torch.cat(legal_idx_parts),
        legal_counts=legal_counts,
        stone_idx=torch.cat(stone_idx_parts),
        stone_batch=torch.cat(stone_batch_parts),
        batch=torch.cat(batch_parts),
        num_graphs=graph_base,
        num_nodes=node_base,
    )


@torch.no_grad()
def batched_eval_round(
    model: Any, r: RoundBatch, device: str | torch.device
) -> tuple[list[list[float]], list[float]]:
    """Run ``model`` over a pre-batched :class:`RoundBatch` in one forward.

    Same contract as :func:`batched_eval` — returns ``(logits_per_graph,
    values)`` — but consumes the Rust-built union directly: no Python union
    re-assembly, and the per-graph splits come from ``legal_counts`` /
    ``stone_idx`` instead of a ``lm.sum().item()`` sync per graph.
    """

    g = r.num_graphs
    if g == 0:
        return [], []

    x = r.x.to(device, non_blocking=True)
    edge_index = r.edge_index.to(device, non_blocking=True)
    edge_attr = r.edge_attr.to(device, non_blocking=True)
    legal_idx = r.legal_idx.to(device, non_blocking=True)
    stone_idx = r.stone_idx.to(device, non_blocking=True)
    stone_batch = r.stone_batch.to(device, non_blocking=True)

    emb = model.representation(x, edge_index, edge_attr)
    out_dim = emb.shape[1]

    # policy: mlp over legal nodes (in legal_moves() order), split by legal_counts.
    logits_flat = model.policy_head.mlp(emb[legal_idx]).squeeze(-1).float().cpu()

    # value: per-graph mean over stone nodes, then mlp + tanh.
    sums = emb.new_zeros(g, out_dim).index_add_(0, stone_batch, emb[stone_idx])
    counts = emb.new_zeros(g).index_add_(
        0, stone_batch, torch.ones(stone_batch.shape[0], device=device, dtype=emb.dtype)
    )
    no_stone = counts == 0
    if bool(no_stone.any()):
        batch = r.batch.to(device, non_blocking=True)
        all_sums = emb.new_zeros(g, out_dim).index_add_(0, batch, emb)
        all_counts = emb.new_zeros(g).index_add_(
            0, batch, torch.ones(emb.shape[0], device=device, dtype=emb.dtype)
        )
        sums[no_stone] = all_sums[no_stone]
        counts[no_stone] = all_counts[no_stone]
    pooled = sums / counts.clamp(min=1).unsqueeze(1)
    values = model.value_head.mlp(pooled).squeeze(-1).float().cpu()

    logits_per_graph: list[list[float]] = []
    pos = 0
    for lc in r.legal_counts:
        logits_per_graph.append(logits_flat[pos : pos + lc].tolist())
        pos += lc
    return logits_per_graph, values.tolist()
