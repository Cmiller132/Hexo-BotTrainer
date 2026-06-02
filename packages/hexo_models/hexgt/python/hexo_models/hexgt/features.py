"""Build D6-invariant model tensors from the Rust graph facts (one position).

`rust_bridge.graph_facts(state, n)` returns the bounded typed graph as plain
Python arrays (node types/coords/owners/window-meta + typed edges). This module
turns that into the model's per-graph tensors:

- `node_feat` (N, NODE_FEATURE_DIM): the D6-INVARIANT node features (constants.py
  layout) — no raw coords / axis labels; geometry rides on edges + structure.
- `edge_index` (2, E), `edge_type` (E,), `edge_attr` (E, EDGE_ATTR_DIM): typed
  edges with an invariant per-edge hex-distance.
- `candidate_index` (C,), `candidate_ids` (C,): candidate node rows in CSR/legal
  order (the priors order).

This is the SINGLE featurizer shared by the model collation (`collate.py`) and
the Phase-4 expand step, so training inputs == search inputs.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .constants import (
    COORD_SCALE,
    COUNT_SCALE,
    EDGE_ATTR_DIM,
    EDGE_ATTR_DIST,
    EDGE_TYPE_CANDIDATE_WINDOW,
    F_CAND_COMPLETE_OPP,
    F_CAND_COMPLETE_OWN,
    F_CAND_DIST_FIRST,
    F_CAND_NWIN_OPP,
    F_CAND_NWIN_OWN,
    F_CAND_OPP_WIN3,
    F_CAND_OPP_WIN4,
    F_CAND_OPP_WIN5,
    F_CAND_OWN_WIN3,
    F_CAND_OWN_WIN4,
    F_CAND_OWN_WIN5,
    F_CENTER_DISTANCE,
    F_OWNER_OPP,
    F_OWNER_OWN,
    F_SIDE_IS_SECOND,
    F_SIDE_MOVE_NUMBER,
    F_SIDE_PHASE_ONEHOT,
    F_SIDE_STONES_OPP,
    F_SIDE_STONES_OWN,
    F_STONE_RECENCY,
    F_TYPE_ONEHOT,
    F_WIN_COUNT_ONEHOT,
    F_WIN_EMPTY_CELLS,
    NODE_FEATURE_DIM,
    NODE_TYPE_CANDIDATE,
    NODE_TYPE_SIDE,
    NODE_TYPE_STONE,
    NODE_TYPE_WINDOW,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    WINDOW_LEN,
)


class GraphTensors:
    """Per-graph model tensors (numpy; torch conversion happens in collate)."""

    __slots__ = (
        "node_feat",
        "node_type",
        "edge_index",
        "edge_type",
        "edge_attr",
        "candidate_index",
        "candidate_ids",
        "num_nodes",
    )

    def __init__(self, node_feat, node_type, edge_index, edge_type, edge_attr, candidate_index, candidate_ids):
        self.node_feat = node_feat
        self.node_type = node_type
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_attr = edge_attr
        self.candidate_index = candidate_index
        self.candidate_ids = candidate_ids
        self.num_nodes = int(node_feat.shape[0])


def _ints(x: Any) -> np.ndarray:
    """Coerce a Rust array to int64. PyO3 maps Vec<u8> -> Python bytes (so the
    u8 columns node_type/node_wcount/node_wempty/edge_type arrive as bytes);
    everything else arrives as list[int]."""

    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(bytes(x), dtype=np.uint8).astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def _hex_distance(q0: int, r0: int, q1: int, r1: int) -> int:
    dq = q0 - q1
    dr = r0 - r1
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))


def build_graph_tensors(facts: Mapping[str, Any]) -> GraphTensors:
    """Featurize one position's Rust `graph_facts` dict into model tensors."""

    nodes = facts["nodes"]
    node_type = _ints(nodes["node_type"])
    nq = _ints(nodes["node_q"])
    nr = _ints(nodes["node_r"])
    owner = _ints(nodes["node_owner"])
    recency = _ints(nodes["node_recency"])
    wcount = _ints(nodes["node_wcount"])
    wempty = _ints(nodes["node_wempty"])
    n = int(node_type.shape[0])

    meta = facts.get("meta", {})
    placements = int(meta.get("placements", max(1, int(recency.max()) if n else 1)))
    phase_idx = int(meta.get("phase", 0))
    # This turn's first-stone coord on the SECOND placement, else None (v2).
    first_stone = meta.get("first_stone")
    denom_recency = float(max(1, placements))

    feat = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)

    # type one-hot
    valid = (node_type >= 0) & (node_type < NUM_NODE_TYPES)
    feat[np.arange(n)[valid], F_TYPE_ONEHOT + node_type[valid]] = 1.0
    # owner
    feat[owner == 0, F_OWNER_OWN] = 1.0
    feat[owner == 1, F_OWNER_OPP] = 1.0
    # center distance (D6-invariant): max-norm from origin
    ns = -nq - nr
    center_dist = np.maximum(np.maximum(np.abs(nq), np.abs(nr)), np.abs(ns)).astype(np.float32)
    feat[:, F_CENTER_DISTANCE] = center_dist / COORD_SCALE

    # stone recency
    is_stone = node_type == NODE_TYPE_STONE
    feat[is_stone, F_STONE_RECENCY] = np.clip(recency[is_stone], 0, None) / denom_recency

    # window count one-hot (3,4,5) + empty
    is_win = node_type == NODE_TYPE_WINDOW
    win_idx = np.clip(wcount - 3, 0, 2)
    feat[np.arange(n)[is_win], F_WIN_COUNT_ONEHOT + win_idx[is_win]] = 1.0
    feat[is_win, F_WIN_EMPTY_CELLS] = wempty[is_win] / float(WINDOW_LEN)

    # candidate distance to this turn's first stone (v2; D6-invariant). Left 0
    # when there is no first stone yet (the turn's FIRST placement).
    is_cand = node_type == NODE_TYPE_CANDIDATE
    if first_stone is not None:
        fq, fr = int(first_stone[0]), int(first_stone[1])
        dq = nq - fq
        dr = nr - fr
        dd = -dq - dr
        cdist = np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(dd)).astype(np.float32)
        feat[is_cand, F_CAND_DIST_FIRST] = cdist[is_cand] / COORD_SCALE

    # side node (index 0 by construction)
    side_nodes = np.nonzero(node_type == NODE_TYPE_SIDE)[0]
    stones_own = int(np.count_nonzero(is_stone & (owner == 0)))
    stones_opp = int(np.count_nonzero(is_stone & (owner == 1)))
    for s in side_nodes:
        feat[s, F_SIDE_PHASE_ONEHOT + min(max(phase_idx, 0), 2)] = 1.0
        feat[s, F_SIDE_STONES_OWN] = stones_own / COUNT_SCALE
        feat[s, F_SIDE_STONES_OPP] = stones_opp / COUNT_SCALE
        feat[s, F_SIDE_MOVE_NUMBER] = placements / COUNT_SCALE
        # v2: 1 when the current placement is the turn's SECOND stone.
        feat[s, F_SIDE_IS_SECOND] = 1.0 if phase_idx == 2 else 0.0

    # edges
    edges = facts["edges"]
    esrc = _ints(edges["edge_src"])
    edst = _ints(edges["edge_dst"])
    etype = _ints(edges["edge_type"])
    e = int(esrc.shape[0])
    edge_index = np.stack([esrc, edst], axis=0) if e else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.zeros((e, EDGE_ATTR_DIM), dtype=np.float32)
    if e:
        etv = np.clip(etype, 0, NUM_EDGE_TYPES - 1)
        edge_attr[np.arange(e), etv] = 1.0
        # invariant per-edge hex-distance
        dq = nq[esrc] - nq[edst]
        dr = nr[esrc] - nr[edst]
        ds = -dq - dr
        dist = np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(ds)).astype(np.float32)
        edge_attr[:, EDGE_ATTR_DIST] = dist / COORD_SCALE

    # candidate tactical features from candidate<->window membership edges
    # (window->candidate direction: src is a WINDOW node).
    if e:
        cw = etype == EDGE_TYPE_CANDIDATE_WINDOW
        w_is_src = is_win[esrc] if e else np.zeros(0, dtype=bool)
        sel = cw & w_is_src
        for wsrc, cdst in zip(esrc[sel].tolist(), edst[sel].tolist()):
            own = owner[wsrc]
            cnt = wcount[wsrc]
            if own == 0:
                feat[cdst, F_CAND_NWIN_OWN] += 1.0
                # v2: per-count breakdown of the own active windows.
                if cnt == 3:
                    feat[cdst, F_CAND_OWN_WIN3] += 1.0
                elif cnt == 4:
                    feat[cdst, F_CAND_OWN_WIN4] += 1.0
                elif cnt == 5:
                    feat[cdst, F_CAND_OWN_WIN5] += 1.0
                if cnt == 5:
                    feat[cdst, F_CAND_COMPLETE_OWN] = 1.0
            elif own == 1:
                feat[cdst, F_CAND_NWIN_OPP] += 1.0
                # v2: per-count breakdown of the opp active windows.
                if cnt == 3:
                    feat[cdst, F_CAND_OPP_WIN3] += 1.0
                elif cnt == 4:
                    feat[cdst, F_CAND_OPP_WIN4] += 1.0
                elif cnt == 5:
                    feat[cdst, F_CAND_OPP_WIN5] += 1.0
                if cnt == 5:
                    feat[cdst, F_CAND_COMPLETE_OPP] = 1.0
        # normalize the nwin counts (+ the v2 per-count splits share the scale, so
        # the three own bins sum to NWIN_OWN and the three opp bins to NWIN_OPP).
        feat[:, F_CAND_NWIN_OWN] /= COORD_SCALE
        feat[:, F_CAND_NWIN_OPP] /= COORD_SCALE
        for _col in (F_CAND_OWN_WIN3, F_CAND_OWN_WIN4, F_CAND_OWN_WIN5,
                     F_CAND_OPP_WIN3, F_CAND_OPP_WIN4, F_CAND_OPP_WIN5):
            feat[:, _col] /= COORD_SCALE

    candidate_index = np.asarray(facts["candidate_nodes"], dtype=np.int64)
    candidate_ids = np.asarray(facts["candidate_ids"], dtype=np.int64)

    return GraphTensors(
        node_feat=feat,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=etype,
        edge_attr=edge_attr,
        candidate_index=candidate_index,
        candidate_ids=candidate_ids,
    )
