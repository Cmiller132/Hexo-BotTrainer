"""Pure-Python port of hexo-strix's "axis" graph builder.

Faithful reimplementation of the Rust ``build_axis_graph``
(``hexo-rs/hexo-mcts/src/axis_graph.rs``), ``node_threat_features``
(``hexo-rs/hexo-engine/src/threat.rs``), ``legal_moves``
(``hexo-rs/hexo-engine/src/legal_moves.rs``) and ``hex_distance``
(``hexo-rs/hexo-engine/src/hex.rs``) from https://github.com/SootyOwl/hexo-strix.

Engine-agnostic on purpose: players are plain ints (+1 = P1, -1 = P2), so the
featurizer can be unit-tested against the upstream Rust test vectors without
this repo's engine. See ``player.py`` for the engine-state -> graph mapping.

Only the legacy schema this checkpoint uses is implemented:
``compact_stone_onehot=False``, ``node_coords=True``, ``moves_scope=Node``,
``axis_relational=False`` (5-dim edge_attr).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The 3 win-detection axes (one direction per axis) — types.rs WIN_AXES.
WIN_AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))

P1 = 1
P2 = -1


def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Axial hex distance: max(|dq|, |dr|, |dq+dr|) — hex.rs hex_distance."""
    dq = abs(b[0] - a[0])
    dr = abs(b[1] - a[1])
    ds = abs(b[0] - a[0] + b[1] - a[1])
    return max(dq, dr, ds)


def hex_offsets(radius: int) -> list[tuple[int, int]]:
    """All (dq, dr) within hex-distance <= radius of the origin — hex.rs."""
    offsets: list[tuple[int, int]] = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                offsets.append((dq, dr))
    return offsets


def legal_moves(
    stones: dict[tuple[int, int], int], radius: int
) -> list[tuple[int, int]]:
    """Empty cells within hex-distance <= radius of any stone, sorted (q, r).

    Mirrors legal_moves.rs: lexicographic ascending order, which matches this
    repo's engine action-id order (pack_coord is (q, r) ascending).
    """
    offsets = hex_offsets(radius)
    candidates: set[tuple[int, int]] = set()
    for (sq, sr) in stones:
        for (dq, dr) in offsets:
            cell = (sq + dq, sr + dr)
            if cell not in stones:
                candidates.add(cell)
    return sorted(candidates)


def node_threat_features(
    stones: dict[tuple[int, int], int],
    coord: tuple[int, int],
    to_move: int,
    win_length: int,
) -> list[float]:
    """[own_max_line, opp_max_line, own_threat_axes, opp_threat_axes].

    Verbatim port of threat.rs ``node_threat_features``. "own" == side to move.
    Normalised by win_length (lines) and 3 (axes). A clean window (free of the
    other player's stones) with >= win_length-2 stones marks a threat axis.
    """
    wl = int(win_length)
    n_cells = 2 * wl - 1
    opp = -to_move
    own_max = 0
    opp_max = 0
    own_axes = 0
    opp_axes = 0
    cq, cr = coord
    for (dq, dr) in WIN_AXES:
        cells: list[int | None] = []
        for k in range(-(wl - 1), wl):  # k in [-(wl-1), wl-1], 2*wl-1 values
            cells.append(stones.get((cq + k * dq, cr + k * dr)))
        assert len(cells) == n_cells
        axis_own = 0
        axis_opp = 0
        for start in range(wl):  # wl sliding windows of width wl
            window = cells[start : start + wl]
            own_n = sum(1 for c in window if c == to_move)
            opp_n = sum(1 for c in window if c == opp)
            if opp_n == 0:
                axis_own = max(axis_own, own_n)
            if own_n == 0:
                axis_opp = max(axis_opp, opp_n)
        own_max = max(own_max, axis_own)
        opp_max = max(opp_max, axis_opp)
        if axis_own >= wl - 2:
            own_axes += 1
        if axis_opp >= wl - 2:
            opp_axes += 1
    return [own_max / wl, opp_max / wl, own_axes / 3.0, opp_axes / 3.0]


@dataclass
class _Layout:
    own: int
    opp: int
    empty: int | None
    to_move: int | None
    moves: int | None
    norm_q: int | None
    norm_r: int | None
    inv_dist: int
    base_dim: int


def _node_layout(relative_stones: bool) -> _Layout:
    """Replicates axis_graph.rs NodeLayout::new for the legacy default lean opts
    (empty one-hot present, node_coords present, moves_scope=Node)."""
    idx = 0

    def take() -> int:
        nonlocal idx
        c = idx
        idx += 1
        return c

    own = take()
    opp = take()
    empty = take()  # not compact_stone_onehot
    to_move = None if relative_stones else take()
    moves = take()  # moves_scope=Node
    norm_q = take()
    norm_r = take()
    inv_dist = take()
    return _Layout(own, opp, empty, to_move, moves, norm_q, norm_r, inv_dist, idx)


@dataclass
class AxisGraph:
    """Torch tensors for a single position, ready for HeXONet.forward."""

    x: torch.Tensor  # (N, fdim) float32
    edge_index: torch.Tensor  # (2, E) int64  [src, dst]
    edge_attr: torch.Tensor  # (E, 5) float32
    legal_mask: torch.Tensor  # (N,) bool
    stone_mask: torch.Tensor  # (N,) bool
    coords: torch.Tensor  # (N, 2) int32
    legal_coords: list[tuple[int, int]]  # coords of legal nodes, in node order


def build_axis_graph(
    stones: list[tuple[tuple[int, int], int]],
    *,
    to_move: int,
    moves_remaining: int,
    win_length: int,
    placement_radius: int,
    prune_empty_edges: bool = True,
    threat_features: bool = True,
    relative_stones: bool = True,
) -> AxisGraph:
    """Build the axis-window graph for one position.

    ``stones`` is a list of ((q, r), player) with player in {+1 (P1), -1 (P2)};
    it need not be sorted. Node order is: stones sorted by (q, r), then legal
    empties sorted by (q, r), then one all-but-scalars dummy/global node.
    """
    stones_sorted = sorted(stones, key=lambda item: item[0])
    stones_map: dict[tuple[int, int], int] = {c: p for c, p in stones_sorted}
    legal = legal_moves(stones_map, placement_radius)

    layout = _node_layout(relative_stones)
    fdim = layout.base_dim + (4 if threat_features else 0)
    player_feat = float(to_move)
    own_is_p1 = player_feat > 0.0
    moves_feat = moves_remaining / 2.0

    n_stones = len(stones_sorted)
    n_legal = len(legal)
    n_real = n_stones + n_legal
    n = n_real + 1
    dummy_idx = n_real
    window = win_length - 1

    # --- coord arrays / lookup / node kinds ---
    coords: list[tuple[int, int]] = []
    coord_to_idx: dict[tuple[int, int], int] = {}
    # node_kind: player int for stones, 0 for empty
    node_player: list[int] = []
    node_is_stone: list[bool] = []
    for i, (coord, player) in enumerate(stones_sorted):
        coords.append(coord)
        coord_to_idx[coord] = i
        node_player.append(player)
        node_is_stone.append(True)
    for j, coord in enumerate(legal):
        coords.append(coord)
        coord_to_idx[coord] = n_stones + j
        node_player.append(0)
        node_is_stone.append(False)
    coords.append((0, 0))  # dummy

    # --- centroid & spread over stones ---
    if n_stones > 0:
        sum_q = sum(c[0] for c, _ in stones_sorted)
        sum_r = sum(c[1] for c, _ in stones_sorted)
        cq = sum_q / n_stones
        cr = sum_r / n_stones
        max_dev = 0.0
        for (q, r), _ in stones_sorted:
            max_dev = max(max_dev, max(abs(q - cq), abs(r - cr)))
        spread = max(max_dev, 1.0)
    else:
        cq, cr, spread = 0.0, 0.0, 1.0

    # --- node features ---
    features = [[0.0] * fdim for _ in range(n)]

    def set_norm_coords(row: list[float], q: int, r: int) -> None:
        if layout.norm_q is not None:
            row[layout.norm_q] = float((q - cq) / spread)
        if layout.norm_r is not None:
            row[layout.norm_r] = float((r - cr) / spread)

    stone_coords = [c for c, _ in stones_sorted]

    for i, (coord, player) in enumerate(stones_sorted):
        row = features[i]
        if relative_stones:
            col = layout.opp if ((player == P1) != own_is_p1) else layout.own
        else:
            col = layout.own if player == P1 else layout.opp
        row[col] = 1.0
        if layout.to_move is not None:
            row[layout.to_move] = player_feat
        if layout.moves is not None:
            row[layout.moves] = moves_feat
        set_norm_coords(row, coord[0], coord[1])
        # inv_dist stays 0 for stones

    for j, coord in enumerate(legal):
        row = features[n_stones + j]
        if layout.empty is not None:
            row[layout.empty] = 1.0
        if layout.to_move is not None:
            row[layout.to_move] = player_feat
        if layout.moves is not None:
            row[layout.moves] = moves_feat
        set_norm_coords(row, coord[0], coord[1])
        min_d = min((hex_distance(coord, sc) for sc in stone_coords), default=1)
        row[layout.inv_dist] = 1.0 / max(min_d, 1)

    # dummy: only global scalars
    if layout.to_move is not None:
        features[dummy_idx][layout.to_move] = player_feat
    if layout.moves is not None:
        features[dummy_idx][layout.moves] = moves_feat

    # --- threat features (real nodes only) ---
    if threat_features:
        toff = layout.base_dim
        for i in range(n_real):
            tf = node_threat_features(stones_map, coords[i], to_move, win_length)
            row = features[i]
            row[toff + 0] = tf[0]
            row[toff + 1] = tf[1]
            row[toff + 2] = tf[2]
            row[toff + 3] = tf[3]

    # --- masks ---
    legal_mask = [False] * n
    stone_mask = [False] * n
    for i in range(n_stones):
        stone_mask[i] = True
    for j in range(n_legal):
        legal_mask[n_stones + j] = True

    # --- axis-window edges ---
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attr: list[list[float]] = []

    for i in range(n_real):
        iq, ir = coords[i]
        i_is_stone = node_is_stone[i]
        i_player = node_player[i]
        src_player_i = float(i_player) if i_is_stone else 0.0
        for axis_idx, (dq, dr) in enumerate(WIN_AXES):
            for sign in (1, -1):
                sdq = dq * sign
                sdr = dr * sign
                for d in range(1, window + 1):
                    tq = iq + sdq * d
                    tr = ir + sdr * d
                    j = coord_to_idx.get((tq, tr))
                    if j is None:
                        break  # off-graph: stop this ray
                    j_is_stone = node_is_stone[j]
                    j_player = node_player[j]
                    both_empty = (not i_is_stone) and (not j_is_stone)
                    if not (prune_empty_edges and both_empty):
                        src_player_j = float(j_player) if j_is_stone else 0.0
                        signed_dist = float(d * sign)
                        # forward i -> j
                        attr_fwd = [0.0] * 5
                        attr_fwd[axis_idx] = 1.0
                        attr_fwd[3] = signed_dist
                        attr_fwd[4] = src_player_i
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_attr.append(attr_fwd)
                        # reverse j -> i
                        attr_rev = [0.0] * 5
                        attr_rev[axis_idx] = 1.0
                        attr_rev[3] = -signed_dist
                        attr_rev[4] = src_player_j
                        edge_src.append(j)
                        edge_dst.append(i)
                        edge_attr.append(attr_rev)
                    # walk stopping
                    if i_is_stone:
                        should_stop = j_is_stone and (j_player != i_player)
                    else:
                        should_stop = j_is_stone
                    if should_stop:
                        break

    # --- dedup on (src, dst, axis) keeping first occurrence ---
    if edge_src:
        seen: set[tuple[int, int, int]] = set()
        ksrc: list[int] = []
        kdst: list[int] = []
        kattr: list[list[float]] = []
        for e in range(len(edge_src)):
            a = edge_attr[e]
            axis = 0 if a[0] > 0.5 else (1 if a[1] > 0.5 else 2)
            key = (edge_src[e], edge_dst[e], axis)
            if key not in seen:
                seen.add(key)
                ksrc.append(edge_src[e])
                kdst.append(edge_dst[e])
                kattr.append(a)
        edge_src, edge_dst, edge_attr = ksrc, kdst, kattr

    # --- legacy dummy edges: bidirectional to all real nodes, zero attr ---
    for i in range(n_real):
        edge_src.append(dummy_idx)
        edge_dst.append(i)
        edge_attr.append([0.0] * 5)
        edge_src.append(i)
        edge_dst.append(dummy_idx)
        edge_attr.append([0.0] * 5)

    if edge_src:
        edge_index_t = torch.tensor([edge_src, edge_dst], dtype=torch.int64)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index_t = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr_t = torch.zeros((0, 5), dtype=torch.float32)

    return AxisGraph(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        legal_mask=torch.tensor(legal_mask, dtype=torch.bool),
        stone_mask=torch.tensor(stone_mask, dtype=torch.bool),
        coords=torch.tensor(coords, dtype=torch.int32),
        legal_coords=list(legal),
    )
