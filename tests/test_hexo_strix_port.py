"""Tests for the hexo-strix port (packages/hexo_strix).

The threat-feature tests mirror the upstream Rust unit tests verbatim
(hexo-rs/hexo-engine/src/threat.rs), which is the tightest available check that
the pure-Python featurizer reproduces the trained model's inputs. Checkpoint /
engine-dependent tests are skipped if the checkpoint or engine is unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# hexo_runner's editable install is broken on this machine (points at a sibling
# checkout); prepend this worktree's source so the RunnerPlayer imports resolve.
# hexo_strix is not always pip-installed in the test venv (the eval scripts add
# it to sys.path); add it here too so the tests import from this worktree.
_PKG_ROOT = Path(__file__).resolve().parents[1] / "packages"
for _pkg in ("hexo_runner", "hexo_strix"):
    _src = _PKG_ROOT / _pkg / "python"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from hexo_strix.graph import P1, P2, build_axis_graph, node_threat_features

# The checkpoint lives outside the repo; override with STRIX_CKPT if needed.
_DEFAULT_CKPT = Path(os.path.expanduser("~/Downloads/checkpoint_00237000.pt"))
_CKPT = Path(os.environ.get("STRIX_CKPT", _DEFAULT_CKPT))


def _stones(items):
    return {c: p for c, p in items}


def test_threat_empty_board():
    assert node_threat_features({}, (0, 0), P1, 4) == [0.0, 0.0, 0.0, 0.0]


def test_threat_three_own_in_row_wl4():
    s = _stones([((1, 0), P1), ((2, 0), P1), ((3, 0), P1)])
    assert node_threat_features(s, (0, 0), P1, 4) == [3 / 4, 0.0, 1 / 3, 0.0]


def test_threat_p2_perspective_swaps_own_opp():
    s = _stones([((1, 0), P1), ((2, 0), P1), ((3, 0), P1)])
    assert node_threat_features(s, (0, 0), P2, 4) == [0.0, 3 / 4, 0.0, 1 / 3]


def test_threat_opponent_stone_blocks_window():
    s = _stones([((1, 0), P1), ((3, 0), P1), ((2, 0), P2)])
    assert node_threat_features(s, (0, 0), P1, 4) == [1 / 4, 0.0, 0.0, 0.0]


def test_threat_stone_node_counts_itself():
    s = _stones([((0, 0), P1), ((1, 0), P1)])
    f = node_threat_features(s, (0, 0), P1, 4)
    assert f[0] == 2 / 4
    assert f[2] == 1 / 3


def test_threat_mixed_window_counts_for_neither():
    s = _stones([((0, 0), P1), ((1, 0), P1), ((2, 0), P2)])
    assert node_threat_features(s, (0, 0), P1, 4) == [2 / 4, 0.0, 1 / 3, 0.0]


def test_threat_multi_axis_max_and_axis_count():
    s = _stones(
        [((1, 0), P1), ((2, 0), P1), ((3, 0), P1), ((0, 1), P1), ((0, 2), P1)]
    )
    assert node_threat_features(s, (0, 0), P1, 4) == [3 / 4, 0.0, 2 / 3, 0.0]


def test_threat_near_saturation_wl6():
    s = _stones([((1, 0), P1), ((2, 0), P1), ((3, 0), P1), ((4, 0), P1), ((5, 0), P1)])
    assert node_threat_features(s, (0, 0), P1, 6) == [5 / 6, 0.0, 1 / 3, 0.0]


def test_build_axis_graph_shapes_and_ordering():
    # Two stones; graph = stones (sorted) + radius-2 empties (sorted) + dummy.
    stones = [((0, 0), P1), ((1, 0), P2)]
    g = build_axis_graph(
        stones,
        to_move=P1,
        moves_remaining=2,
        win_length=6,
        placement_radius=2,
        prune_empty_edges=True,
        threat_features=True,
        relative_stones=True,
    )
    n = g.x.shape[0]
    assert g.x.shape[1] == 11  # 7 base + 4 threat
    assert g.legal_mask.sum().item() == len(g.legal_coords)
    assert g.stone_mask.sum().item() == 2
    # dummy node is last, in neither mask
    assert not bool(g.legal_mask[-1]) and not bool(g.stone_mask[-1])
    # legal_coords aligns with the legal_mask nodes' coords, in order
    legal_from_mask = [
        (int(g.coords[i, 0]), int(g.coords[i, 1]))
        for i in range(n)
        if bool(g.legal_mask[i])
    ]
    assert legal_from_mask == g.legal_coords
    assert g.legal_coords == sorted(g.legal_coords)


@pytest.mark.skipif(not _CKPT.exists(), reason=f"checkpoint not found at {_CKPT}")
def test_checkpoint_loads_strictly():
    import torch

    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    assert ck.game_config["win_length"] == 6
    # A forward pass on a simple 2-stone position yields aligned logits + value.
    g = build_axis_graph(
        [((0, 0), P1), ((0, 1), P2)],
        to_move=P1,
        moves_remaining=2,
        win_length=ck.game_config["win_length"],
        placement_radius=ck.game_config["placement_radius"],
    )
    with torch.no_grad():
        logits, value = ck.model(
            g.x, g.edge_index, g.legal_mask, g.stone_mask, edge_attr=g.edge_attr
        )
    assert logits.shape[0] == len(g.legal_coords)
    assert -1.0 <= float(value.item()) <= 1.0


@pytest.mark.skipif(not _CKPT.exists(), reason=f"checkpoint not found at {_CKPT}")
def test_bot_completes_own_five_in_a_row():
    """Side to move (P1) has 5 in a row with both ends open -> must win now.

    A competent trained net should place the 6th stone to complete the line.
    This exercises the full featurize->forward->argmax path and asserts the
    move is sensible (one of the two winning completions).
    """
    import torch

    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    wl = ck.game_config["win_length"]
    # P1 stones at (0,0)..(4,0); winning completions are (-1,0) and (5,0).
    stones = [((k, 0), P1) for k in range(5)]
    # give P2 a couple of unrelated stones so it's a realistic mid-game board
    stones += [((0, 3), P2), ((1, 3), P2)]
    g = build_axis_graph(
        stones,
        to_move=P1,
        moves_remaining=2,
        win_length=wl,
        placement_radius=ck.game_config["placement_radius"],
    )
    with torch.no_grad():
        logits, _ = ck.model(
            g.x, g.edge_index, g.legal_mask, g.stone_mask, edge_attr=g.edge_attr
        )
    best = g.legal_coords[int(torch.argmax(logits).item())]
    assert best in {(-1, 0), (5, 0)}, f"expected a winning completion, got {best}"


# --- Real Gumbel MCTS path (hexo_rs) ---------------------------------------

_HAS_HEXO_RS = False
try:  # the compiled Rust extension may not be installed
    import hexo_rs as _hexo_rs  # noqa: F401

    _HAS_HEXO_RS = True
except Exception:
    pass

_skip_mcts = pytest.mark.skipif(
    not (_HAS_HEXO_RS and _CKPT.exists()),
    reason="requires hexo_rs extension and the checkpoint",
)


@_skip_mcts
def test_mcts_completes_own_five():
    """Real Gumbel MCTS (256 sims, noise off) must complete an open five."""
    import hexo_rs

    from hexo_strix.loader import load_strix_checkpoint
    from hexo_strix.mcts_player import StrixMctsPlayer

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    p = StrixMctsPlayer(checkpoint=ck, device="cpu", sims=256, disable_gumbel_noise=True)
    gc = hexo_rs.GameConfig(ck.game_config["win_length"], ck.game_config["placement_radius"], 300)
    # P1 to move, five in a row (0,0)..(4,0); (5,0) or (-1,0) wins immediately.
    stones = [((k, 0), "P1") for k in range(5)] + [((0, 3), "P2"), ((2, 3), "P2")]
    game = hexo_rs.GameState.from_state(stones, "P1", 2, gc)
    (_a, improved, *_rest) = hexo_rs.gumbel_mcts_with_diagnostics(
        game, p._eval_fn, p._mcts_config, seed=None
    )
    lm = game.legal_moves()
    move = lm[max(range(len(improved)), key=lambda i: improved[i])]
    assert tuple(move) in {(-1, 0), (5, 0)}, f"MCTS missed the win: {move}"


@_skip_mcts
def test_mcts_deterministic_when_noise_off():
    """disable_gumbel_noise=True => identical move across repeated searches."""
    import hexo_rs

    from hexo_strix.loader import load_strix_checkpoint
    from hexo_strix.mcts_player import StrixMctsPlayer

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    p = StrixMctsPlayer(checkpoint=ck, device="cpu", sims=128, disable_gumbel_noise=True)
    gc = hexo_rs.GameConfig(ck.game_config["win_length"], ck.game_config["placement_radius"], 300)
    stones = [((0, 0), "P1"), ((1, 0), "P1"), ((2, 0), "P1"), ((0, 2), "P2"), ((1, 2), "P2")]

    def run():
        game = hexo_rs.GameState.from_state(stones, "P2", 2, gc)
        (_a, improved, *_r) = hexo_rs.gumbel_mcts_with_diagnostics(
            game, p._eval_fn, p._mcts_config, seed=None
        )
        lm = game.legal_moves()
        return tuple(lm[max(range(len(improved)), key=lambda i: improved[i])])

    assert run() == run()


# --- Batched forward + cross-game batch server -----------------------------


def _sample_flags(ck):
    mcfg = ck.model_config
    return dict(
        prune_empty_edges=bool(mcfg.get("prune_empty_edges", True)),
        threat_features=bool(mcfg.get("threat_features", True)),
        relative_stones=bool(mcfg.get("relative_stone_encoding", True)),
    )


def _sample_states(hexo_rs, ck, n_positions=6):
    """A few varied mid-game hexo_rs GameStates."""
    gc = ck.game_config
    game_config = hexo_rs.GameConfig(
        int(gc["win_length"]), int(gc["placement_radius"]), int(gc.get("max_moves", 300))
    )
    boards = [
        [((0, 0), "P1"), ((1, 0), "P2")],
        [((0, 0), "P1"), ((1, 0), "P2"), ((-1, 1), "P1")],
        [((0, 0), "P1"), ((1, 0), "P2"), ((-1, 1), "P1"), ((2, -1), "P2")],
        [((0, 0), "P1"), ((1, 0), "P2"), ((-1, 1), "P1"), ((2, -1), "P2"), ((0, 1), "P1")],
        [((k, 0), "P1") for k in range(3)] + [((0, 2), "P2"), ((1, 2), "P2")],
        [((0, 0), "P1"), ((2, 0), "P2"), ((-1, 2), "P1"), ((3, -1), "P2"), ((0, 2), "P1"), ((1, 1), "P2")],
    ][:n_positions]
    states = []
    for b in boards:
        to_move = "P1" if len(b) % 2 == 0 else "P2"
        states.append(hexo_rs.GameState.from_state(b, to_move, 2, game_config))
    return states


def _sample_graphs(hexo_rs, ck, n_positions=6):
    """A few varied mid-game axis graphs (as CPU tensor tuples)."""
    from hexo_strix.batched_infer import build_axis_graph_tensors

    flags = _sample_flags(ck)
    return [
        build_axis_graph_tensors(hexo_rs, s, **flags)
        for s in _sample_states(hexo_rs, ck, n_positions)
    ]


@_skip_mcts
def test_batched_eval_matches_serial_forward():
    """batched_eval over N graphs == per-graph model forward (up to fp)."""
    import hexo_rs
    import torch

    from hexo_strix.batched_infer import batched_eval
    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    model = ck.model.eval()
    graphs = _sample_graphs(hexo_rs, ck)

    lp_b, v_b = batched_eval(model, graphs, "cpu")
    # Per-graph reference via the model's own single-graph forward.
    lp_s, v_s = [], []
    for (x, ei, ea, lm, sm) in graphs:
        with torch.no_grad():
            logits, value = model(x, ei, lm, sm, edge_attr=ea)
        lp_s.append(logits.tolist())
        v_s.append(float(value.item()))

    assert [len(a) for a in lp_b] == [len(a) for a in lp_s]  # legal counts line up
    dlog = max((abs(a - b) for A, B in zip(lp_b, lp_s) for a, b in zip(A, B)), default=0.0)
    dval = max((abs(a - b) for a, b in zip(v_b, v_s)), default=0.0)
    assert dlog < 1e-4, f"policy drift too large: {dlog}"
    assert dval < 1e-4, f"value drift too large: {dval}"


@_skip_mcts
def test_build_axis_round_matches_per_leaf_build():
    """The one-call zero-copy round builder (game_states_to_axis_batch_bytes)
    produces the SAME logits/values as the per-leaf build + batched_eval — a
    pure speedup, byte-identical graphs."""
    import hexo_rs

    from hexo_strix.batched_infer import (
        batched_eval,
        batched_eval_round,
        build_axis_round,
    )
    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    model = ck.model.eval()
    states = _sample_states(hexo_rs, ck)
    flags = _sample_flags(ck)

    # reference: per-leaf build + batched_eval
    graphs = _sample_graphs(hexo_rs, ck)
    lp_ref, v_ref = batched_eval(model, graphs, "cpu")

    # fast path: one Rust call -> RoundBatch -> batched_eval_round
    rb = build_axis_round(hexo_rs, states, **flags)
    assert rb.num_graphs == len(states)
    lp_fast, v_fast = batched_eval_round(model, rb, "cpu")

    assert [len(a) for a in lp_fast] == [len(a) for a in lp_ref]
    dlog = max((abs(a - b) for A, B in zip(lp_fast, lp_ref) for a, b in zip(A, B)), default=0.0)
    dval = max((abs(a - b) for a, b in zip(v_fast, v_ref)), default=0.0)
    assert dlog < 1e-4, f"round policy drift too large: {dlog}"
    assert dval < 1e-4, f"round value drift too large: {dval}"


@_skip_mcts
def test_concat_rounds_matches_separate_rounds():
    """Coalescing several RoundBatches (the batch server's cross-game step) gives
    per-graph results identical to scoring each round on its own."""
    import hexo_rs

    from hexo_strix.batched_infer import (
        batched_eval_round,
        build_axis_round,
        concat_rounds,
    )
    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    model = ck.model.eval()
    flags = _sample_flags(ck)
    states = _sample_states(hexo_rs, ck)
    # split into 3 uneven "rounds" as different games would submit
    rounds = [
        build_axis_round(hexo_rs, states[0:1], **flags),
        build_axis_round(hexo_rs, states[1:4], **flags),
        build_axis_round(hexo_rs, states[4:6], **flags),
    ]
    per = [batched_eval_round(model, r, "cpu") for r in rounds]

    merged = concat_rounds(rounds)
    assert merged.num_graphs == sum(r.num_graphs for r in rounds)
    lp_m, v_m = batched_eval_round(model, merged, "cpu")

    # flatten the separate results and compare to the coalesced split
    lp_sep = [a for lp, _ in per for a in lp]
    v_sep = [x for _, v in per for x in v]
    assert [len(a) for a in lp_m] == [len(a) for a in lp_sep]
    dlog = max((abs(a - b) for A, B in zip(lp_m, lp_sep) for a, b in zip(A, B)), default=0.0)
    dval = max((abs(a - b) for a, b in zip(v_m, v_sep)), default=0.0)
    assert dlog < 1e-4 and dval < 1e-4


@_skip_mcts
def test_batch_server_roundbatch_matches():
    """The server scores submitted RoundBatches (the production path) identically
    to a standalone batched_eval_round, under concurrent submission."""
    import threading

    import hexo_rs

    from hexo_strix.batch_server import StrixBatchServer
    from hexo_strix.batched_infer import batched_eval_round, build_axis_round
    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    flags = _sample_flags(ck)
    states = _sample_states(hexo_rs, ck)
    rounds = [build_axis_round(hexo_rs, [s], **flags) for s in states]
    ref = [batched_eval_round(ck.model.eval(), r, "cpu") for r in rounds]

    server = StrixBatchServer(ck.model, device="cpu", linger_s=0.005)
    results: dict[int, tuple] = {}

    def worker(i):
        results[i] = server.evaluate(rounds[i])

    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(rounds))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        server.close()

    assert server.n_forwards >= 1
    for i in range(len(rounds)):
        (lp, v) = results[i]
        (rlp, rv) = ref[i]
        assert [len(a) for a in lp] == [len(a) for a in rlp]
        dlog = max((abs(a - b) for A, B in zip(lp, rlp) for a, b in zip(A, B)), default=0.0)
        assert dlog < 1e-4, f"server round result diverged for {i}: {dlog}"
        assert abs(v[0] - rv[0]) < 1e-4


@_skip_mcts
def test_batch_server_matches_batched_eval_multithread():
    """The cross-game server's per-request results equal a direct batched_eval,
    even when many threads submit concurrently."""
    import threading

    import hexo_rs

    from hexo_strix.batch_server import StrixBatchServer
    from hexo_strix.batched_infer import batched_eval
    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    graphs = _sample_graphs(hexo_rs, ck)
    # Reference: each single graph scored on its own.
    ref = [batched_eval(ck.model.eval(), [g], "cpu") for g in graphs]

    server = StrixBatchServer(ck.model, device="cpu", linger_s=0.005)
    results: dict[int, tuple] = {}

    def worker(i):
        # Each thread submits one graph repeatedly to force coalescing pressure.
        results[i] = server.evaluate([graphs[i]])

    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(graphs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        server.close()

    assert server.n_forwards >= 1
    for i in range(len(graphs)):
        (lp, v) = results[i]
        (rlp, rv) = ref[i]
        assert [len(a) for a in lp] == [len(a) for a in rlp]
        dlog = max((abs(a - b) for A, B in zip(lp, rlp) for a, b in zip(A, B)), default=0.0)
        assert dlog < 1e-4, f"server result diverged for graph {i}: {dlog}"
        assert abs(v[0] - rv[0]) < 1e-4


@_skip_mcts
def test_batch_server_rejects_after_close_does_not_hang():
    """After close(), evaluate() raises promptly instead of blocking forever on
    its event (the fail-open contract play_strix_match relies on)."""
    import hexo_rs

    from hexo_strix.batch_server import StrixBatchServer
    from hexo_strix.loader import load_strix_checkpoint

    ck = load_strix_checkpoint(_CKPT, device="cpu")
    graphs = _sample_graphs(hexo_rs, ck, n_positions=1)
    server = StrixBatchServer(ck.model, device="cpu")
    server.close()
    with pytest.raises(RuntimeError):
        server.evaluate([graphs[0]])
