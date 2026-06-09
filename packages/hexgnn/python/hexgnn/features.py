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
    F_CAND_COMPLETE_OPP,
    F_CAND_COMPLETE_OWN,
    F_CAND_DIST_FIRST,
    F_CAND_NWIN_OPP,
    F_CAND_NWIN_OWN,
    F_CAND_OPP_WIN3,
    F_CAND_OPP_WIN4,
    F_CAND_OPP_WIN5,
    F_CAND_OPP_THREAT,
    F_CAND_OWN_WIN3,
    F_CAND_OWN_WIN4,
    F_CAND_OWN_WIN5,
    F_CAND_WIN_NOW_OWN,
    F_CENTER_DISTANCE,
    F_OWNER_OPP,
    F_OWNER_OWN,
    F_SIDE_IS_SECOND,
    F_SIDE_MOVE_NUMBER,
    F_SIDE_PHASE_ONEHOT,
    F_SIDE_STONES_OPP,
    F_SIDE_STONES_OWN,
    F_STONE_NWIN_OPP,
    F_STONE_NWIN_OWN,
    F_STONE_RECENCY,
    F_TYPE_ONEHOT,
    NODE_FEATURE_DIM,
    NODE_TYPE_CANDIDATE,
    NODE_TYPE_SIDE,
    NODE_TYPE_STONE,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    NUM_TURN_PHASES,
    PHASE_FIRST_STONE,
    PHASE_SECOND_STONE,
)


class GraphTensors:
    """Per-graph model tensors (numpy; torch conversion happens in collate)."""

    __slots__ = (
        "node_feat",
        "node_type",
        "edge_index",
        "edge_type",
        "edge_dir",
        "edge_attr",
        "candidate_index",
        "candidate_ids",
        "num_nodes",
    )

    def __init__(
        self,
        node_feat,
        node_type,
        edge_index,
        edge_type,
        edge_dir,
        edge_attr,
        candidate_index,
        candidate_ids,
    ):
        self.node_feat = node_feat
        self.node_type = node_type
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_dir = edge_dir
        self.edge_attr = edge_attr
        self.candidate_index = candidate_index
        self.candidate_ids = candidate_ids
        self.num_nodes = int(node_feat.shape[0])


def _ints(x: Any) -> np.ndarray:
    """Coerce a Rust array to int64. PyO3 maps Vec<u8> -> Python bytes (so the u8
    columns node_type/edge_type arrive as bytes); the per-node window-count columns
    are Vec<u16>/Vec<bool> and arrive as list[int]/list[bool]; everything else as
    list[int]. np.asarray coerces all the list forms to int64 directly."""

    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(bytes(x), dtype=np.uint8).astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def build_graph_tensors(facts: Mapping[str, Any]) -> GraphTensors:
    """Featurize one position's Rust `graph_facts` dict into model tensors."""

    nodes = facts["nodes"]
    node_type = _ints(nodes["node_type"])
    nq = _ints(nodes["node_q"])
    nr = _ints(nodes["node_r"])
    owner = _ints(nodes["node_owner"])
    recency = _ints(nodes["node_recency"])
    n = int(node_type.shape[0])

    # Per-node window-count facts (sparse rewrite): computed in Rust build_graph
    # from the active-window tokens, replacing the removed candidate<->window hub
    # edges. Candidate columns reproduce the old edge-loop accumulation EXACTLY;
    # the stone_nwin_* columns are the new per-stone window features. (u16 columns
    # arrive as list[int] from PyO3.)
    nwin_own = _ints(nodes["node_nwin_own"])
    nwin_opp = _ints(nodes["node_nwin_opp"])
    own_win3 = _ints(nodes["node_own_win3"])
    own_win4 = _ints(nodes["node_own_win4"])
    own_win5 = _ints(nodes["node_own_win5"])
    opp_win3 = _ints(nodes["node_opp_win3"])
    opp_win4 = _ints(nodes["node_opp_win4"])
    opp_win5 = _ints(nodes["node_opp_win5"])
    complete_own = _ints(nodes["node_complete_own"])
    complete_opp = _ints(nodes["node_complete_opp"])
    opp_threat = _ints(nodes["node_opp_threat"])
    stone_nwin_own = _ints(nodes["node_stone_nwin_own"])
    stone_nwin_opp = _ints(nodes["node_stone_nwin_opp"])

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

    # per-STONE window-count features (sparse rewrite): active windows through this
    # stone's cell, split by owner, normalized /COORD_SCALE like the candidate nwin.
    # Sourced from the per-node stone_nwin_* columns (folded from the removed
    # STONE_WINDOW hub edges). Zero on non-stone nodes (the columns are 0 there).
    feat[is_stone, F_STONE_NWIN_OWN] = stone_nwin_own[is_stone] / COORD_SCALE
    feat[is_stone, F_STONE_NWIN_OPP] = stone_nwin_opp[is_stone] / COORD_SCALE

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
        feat[s, F_SIDE_PHASE_ONEHOT + min(max(phase_idx, 0), NUM_TURN_PHASES - 1)] = 1.0
        feat[s, F_SIDE_STONES_OWN] = stones_own / COUNT_SCALE
        feat[s, F_SIDE_STONES_OPP] = stones_opp / COUNT_SCALE
        feat[s, F_SIDE_MOVE_NUMBER] = placements / COUNT_SCALE
        # v2: 1 when the current placement is the turn's SECOND stone.
        feat[s, F_SIDE_IS_SECOND] = 1.0 if phase_idx == PHASE_SECOND_STONE else 0.0

    # edges
    edges = facts["edges"]
    esrc = _ints(edges["edge_src"])
    edst = _ints(edges["edge_dst"])
    etype = _ints(edges["edge_type"])
    # Per-edge hex-direction index (adjacency 0..5, else -1). Empty -> int64[0].
    edir = _ints(edges["edge_dir"])
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

    # Candidate tactical window-count features (sparse rewrite): sourced from the
    # per-node window-count columns (computed in Rust build_graph from the active-
    # window tokens), replacing the removed candidate<->window hub edges. These
    # columns are populated by build_graph on CANDIDATE cells only (0 elsewhere), so
    # the whole-array writes below leave non-candidate rows at 0 — identical to the
    # old edge-driven values. Counts share the /COORD_SCALE normalization (the three
    # own bins sum to NWIN_OWN, the three opp bins to NWIN_OPP).
    feat[:, F_CAND_NWIN_OWN] = nwin_own / COORD_SCALE
    feat[:, F_CAND_NWIN_OPP] = nwin_opp / COORD_SCALE
    feat[:, F_CAND_OWN_WIN3] = own_win3 / COORD_SCALE
    feat[:, F_CAND_OWN_WIN4] = own_win4 / COORD_SCALE
    feat[:, F_CAND_OWN_WIN5] = own_win5 / COORD_SCALE
    feat[:, F_CAND_OPP_WIN3] = opp_win3 / COORD_SCALE
    feat[:, F_CAND_OPP_WIN4] = opp_win4 / COORD_SCALE
    feat[:, F_CAND_OPP_WIN5] = opp_win5 / COORD_SCALE
    # complete / threat flags (any own/opp count-5; any opp count>=4 -> threat).
    feat[complete_own != 0, F_CAND_COMPLETE_OWN] = 1.0
    feat[complete_opp != 0, F_CAND_COMPLETE_OPP] = 1.0
    feat[opp_threat != 0, F_CAND_OPP_THREAT] = 1.0
    # v3 phase-aware own win-now: own count-5 (any B) OR own count-4 only at
    # FirstStone (phase == FirstStone). Reconstructed from the per-count own columns.
    win_now = (own_win5 > 0)
    if phase_idx == PHASE_FIRST_STONE:
        win_now = win_now | (own_win4 > 0)
    feat[win_now, F_CAND_WIN_NOW_OWN] = 1.0

    candidate_index = np.asarray(facts["candidate_nodes"], dtype=np.int64)
    candidate_ids = np.asarray(facts["candidate_ids"], dtype=np.int64)

    return GraphTensors(
        node_feat=feat,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=etype,
        edge_dir=edir,
        edge_attr=edge_attr,
        candidate_index=candidate_index,
        candidate_ids=candidate_ids,
    )
