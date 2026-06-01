"""Read-only microbenchmark for the dense_cnn native MCTS.

Builds a randomly-initialized 64ch/4block Model1Network on the (free) GPU, wraps
it in the production DenseCNNInference, and drives the real native
BatchedMctsSession exactly as self-play does. It measures, per search() call:

  * wall time (perf_counter around mcts_session.run)
  * Rust-reported encode + evaluator seconds (from the batch diagnostics)
  * node_count / active_edge_bytes / hidden_prior_bytes / eval cache size

Three experiments:
  A) sims scaling at fixed root count (how does cost grow 128 -> 800 visits?)
  B) root-count scaling at fixed sims (does root parallelism amortize?)
  C) per-move timeline of one long game (subtree reuse / advance_root behaviour)

Nothing here touches run state, configs, checkpoints, or the supervisor. Random
weights are fine: we are measuring search *mechanics/throughput*, not strength.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234


def build_inference() -> DenseCNNInference:
    torch.manual_seed(SEED)
    model = Model1Network(channels=64, blocks=4)
    inf = DenseCNNInference(
        model,
        device=DEVICE,
        amp=False,
        return_logits=False,
        max_batch_size=1024,
        optimize_for_inference=True,
    )
    # The native search sends a DIFFERENT batch size almost every pass (number of
    # uncached unique leaves varies). With cudnn.benchmark=True that triggers a
    # fresh autotune per novel shape -> minutes of warmup thrash. Disable it so we
    # measure steady search mechanics, not autotuning. (This itself is finding F-AUTOTUNE.)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = False
    return inf


def fresh_games(n: int, *, base_seed: int = 0) -> list[dict]:
    return [
        {"key": i, "state": engine.new_game(seed=base_seed + i)}
        for i in range(n)
    ]


def playable(games: list[dict], max_actions: int = 1024) -> list[dict]:
    out = []
    for g in games:
        if engine.terminal(g["state"]) is None and g.get("plies", 0) < max_actions:
            out.append(g)
    return out


def run_search(session, games, inference, *, visits, vbatch=4):
    pg = playable(games)
    if not pg:
        return None, []
    t0 = time.perf_counter()
    results = session.run(
        [g["key"] for g in pg],
        [g["state"] for g in pg],
        inference,
        visits=visits,
        c_puct=1.5,
        temperature=1.0,
        seed=SEED,
        virtual_batch_size=vbatch,
        active_root_limit=max(256, len(pg)),
        root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1,
        fpu_reduction=0.20,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=32,
        widening_min_children=2,
    )
    wall = time.perf_counter() - t0
    return wall, list(zip(pg, results))


def advance(games_results):
    for g, res in games_results:
        g["plies"] = g.get("plies", 0) + 1
        engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(res.action_id)))


def batch_diag(results):
    if not results:
        return {}
    d = results[0][1].diagnostics
    return dict(d.get("batch", {}))


def warm(games, session, inference, *, plies=4, visits=32):
    # advance every game a fixed number of plies so positions have realistic
    # branching/encode cost before we time anything.
    for _ in range(plies):
        wall, gr = run_search(session, games, inference, visits=visits)
        if not gr:
            break
        advance(gr)


def exp_sims_scaling(inference) -> list[dict]:
    rows = []
    for visits in (128, 256, 400, 800):
        session = new_mcts_session(max_states=262144)
        games = fresh_games(128, base_seed=visits * 7)
        warm(games, session, inference)
        # time 2 consecutive searched moves, average
        walls, evs, encs, nodes = [], [], [], []
        for _ in range(2):
            wall, gr = run_search(session, games, inference, visits=visits)
            if not gr:
                break
            bd = batch_diag(gr)
            ev = bd.get("evaluation", {})
            tr = bd.get("tree", {})
            walls.append(wall)
            evs.append(ev.get("evaluator_seconds", 0.0))
            encs.append(ev.get("encoding_seconds", 0.0))
            nodes.append(tr.get("node_count", 0))
            advance(gr)
        n = max(len(walls), 1)
        rows.append({
            "visits": visits,
            "roots": 256,
            "wall_s": sum(walls) / n,
            "eval_s": sum(evs) / n,
            "encode_s": sum(encs) / n,
            "tree+orch_s": (sum(walls) - sum(evs) - sum(encs)) / n,
            "node_count": int(sum(nodes) / n) if nodes else 0,
        })
        print(f"[sims] visits={visits}: {rows[-1]}", flush=True)
    return rows


def exp_sims_single_root(inference) -> list[dict]:
    # The eval / actual-play path searches ONE root (player.decide). With one root
    # there is zero rayon parallelism, so this isolates pure single-tree latency
    # vs sims -- the budget-limited path relevant to "raise sims under 50ms".
    rows = []
    for visits in (128, 256, 400, 800):
        session = new_mcts_session(max_states=262144)
        games = fresh_games(1, base_seed=visits * 31 + 5)
        warm(games, session, inference, plies=6, visits=64)
        walls = []
        for _ in range(3):
            wall, gr = run_search(session, games, inference, visits=visits)
            if not gr:
                break
            walls.append(wall)
            advance(gr)
        n = max(len(walls), 1)
        rows.append({"visits": visits, "roots": 1, "wall_ms": 1000.0 * sum(walls) / n})
        print(f"[1root] visits={visits}: {rows[-1]}", flush=True)
    return rows


def exp_root_scaling(inference) -> list[dict]:
    rows = []
    for roots in (1, 8, 32, 128, 256):
        session = new_mcts_session(max_states=262144)
        games = fresh_games(roots, base_seed=roots * 13 + 1)
        warm(games, session, inference)
        walls = []
        for _ in range(2):
            wall, gr = run_search(session, games, inference, visits=128)
            if not gr:
                break
            walls.append(wall)
            advance(gr)
        n = max(len(walls), 1)
        avg = sum(walls) / n
        rows.append({
            "roots": roots,
            "visits": 128,
            "wall_s": avg,
            "wall_per_root_ms": 1000.0 * avg / roots,
        })
        print(f"[roots] roots={roots}: {rows[-1]}", flush=True)
    return rows


def exp_game_timeline(inference) -> list[dict]:
    session = new_mcts_session(max_states=262144)
    games = fresh_games(32, base_seed=99)
    rows = []
    for move in range(30):
        wall, gr = run_search(session, games, inference, visits=256)
        if not gr:
            break
        bd = batch_diag(gr)
        tr = bd.get("tree", {})
        rows.append({
            "move": move,
            "active_roots": len(gr),
            "wall_s": wall,
            "node_count": tr.get("node_count", 0),
            "active_edge_bytes": tr.get("active_edge_bytes", 0),
            "hidden_prior_bytes": tr.get("hidden_prior_bytes", 0),
            "cache_size": bd.get("evaluation", {}).get("cache_size", 0),
            "max_nodes_per_root": tr.get("max_nodes_per_root", 0),
        })
        advance(gr)
        if move % 8 == 0:
            print(f"[timeline] move={move}: {rows[-1]}", flush=True)
    return rows


def main() -> None:
    print(f"device={DEVICE} torch={torch.__version__}", flush=True)
    inference = build_inference()
    # ramp GPU clocks on a couple of fixed shapes so the first timed config is warm.
    if DEVICE == "cuda":
        for _ in range(5):
            inference.evaluate_model1_payload  # noqa: B018  (attribute touch; real warm below)
        gw = fresh_games(64, base_seed=7)
        s = new_mcts_session(max_states=262144)
        warm(gw, s, inference, plies=3, visits=32)
    out = {
        "device": DEVICE,
        "torch": torch.__version__,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if DEVICE == "cuda" else None,
        "sims_scaling_256roots": exp_sims_scaling(inference),
        "sims_scaling_1root": exp_sims_single_root(inference),
        "root_scaling": exp_root_scaling(inference),
        "game_timeline": exp_game_timeline(inference),
    }
    path = Path(__file__).with_name("mcts_microbench_summary.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
