"""Phase 6 — VCF/VCDT forcing-solver exploration: correctness sanity + cost
benchmark. Drives the exploratory Rust prototype (hexgt_vcf_solve) which is NOT
wired into search. Run under the WSL hexgt-build venv (only hexo_engine needed
for positions; torch only for the forward-cost comparison)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for p in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))

import random

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction
import hexo_models.hexgt.rust_bridge as rb

VCF = rb._hexgt_module().hexgt_vcf_solve
ANALYZE = rb._hexgt_module().hexgt_threat_analysis


def play(coords):
    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------- correctness
banner("CORRECTNESS SANITY (prototype is forcing-restricted, NOT a sound prover)")

# Own count-4 at FirstStone (TEST G): side to move wins this turn.
g = play([(0, 0), (0, 7), (2, 7), (1, 0), (4, 0), (4, 7), (6, 7), (5, 0), (0, -2), (0, 8), (2, 8)])
r = VCF(g, 8, 2_000_000, 200)
print(f"TEST G (own count-4, FirstStone) -> win={r['win']} nodes={r['nodes']} (expect win=True)")

# Defensible TEST D: side to move (P1) has NO attack of its own -> no forced win.
P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0), 8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3), 16: (0, -2)}
P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7), 9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
flat = [None] * 17
for i, c in {**P0, **P1}.items():
    flat[i] = c
d = play(flat)
r = VCF(d, 8, 2_000_000, 200)
print(f"TEST D (P1 to move, no own attack) -> win={r['win']} nodes={r['nodes']} (expect win=False)")

# No-threat opening: instant, no forcing moves.
r = VCF(play([(0, 0)]), 8, 2_000_000, 200)
print(f"opening only -> win={r['win']} nodes={r['nodes']} (expect win=False, ~0 nodes)")


# ---------------------------------------------------------------- positions
def gen_positions(n, seed=0, min_plies=20, max_plies=90):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = eng.new_game(seed=rng.randint(0, 10**7))
        for _ in range(rng.randint(min_plies, max_plies)):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out


def bench(states, ply_budget, node_cap=2_000_000, time_limit_ms=100):
    micros, nodes, wins, limits, threated = [], [], 0, 0, 0
    for s in states:
        a = ANALYZE(s)
        has_threat = a["opp_threat_count"] > 0 or a["own_win_now"] or a["tactical_cells"]
        r = VCF(s, ply_budget, node_cap, time_limit_ms)
        micros.append(r["micros"])
        nodes.append(r["nodes"])
        wins += int(r["win"])
        limits += int(r["hit_limit"])
        threated += int(bool(has_threat))
    return micros, nodes, wins, limits, threated


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


banner("COST BENCHMARK (representative random midgame positions)")
states = gen_positions(300, seed=7)
print(f"{len(states)} positions; measuring forcing-solver cost per call\n")
for pb in (4, 8, 12):
    micros, nodes, wins, limits, threated = bench(states, pb, time_limit_ms=100)
    print(f"ply_budget={pb:2d}: "
          f"mean={statistics.mean(micros):8.1f}us p50={pct(micros,50):7d}us "
          f"p90={pct(micros,90):8d}us max={max(micros):9d}us | "
          f"nodes mean={statistics.mean(nodes):8.0f} max={max(nodes):9d} | "
          f"wins={wins} hit_limit={limits} threat_positions={threated}")

# Subset that actually has threats (where the solver does real work).
threat_states = [s for s in states if (ANALYZE(s)["opp_threat_count"] > 0 or ANALYZE(s)["own_win_now"])]
banner(f"COST ON THREAT-BEARING POSITIONS ONLY ({len(threat_states)} of {len(states)})")
if threat_states:
    for pb in (8, 12):
        micros, nodes, wins, limits, _ = bench(threat_states, pb, time_limit_ms=100)
        print(f"ply_budget={pb:2d}: mean={statistics.mean(micros):8.1f}us "
              f"p90={pct(micros,90):8d}us max={max(micros):9d}us | "
              f"nodes mean={statistics.mean(nodes):8.0f} max={max(nodes):9d} | "
              f"wins={wins} hit_limit={limits}")

# ---------------------------------------------------------------- comparison
banner("COST CONTEXT: one GNN forward + one MCTS move, for reference")
try:
    import torch
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.inference import HexgtInference
    from hexo_models.hexgt.mcts import new_mcts_session

    inf = HexgtInference(HexgtNetwork(short_term_value_horizons=()).eval(), device="cpu", fp16=False)
    probe = states[: 8]
    # one full search move (visits=128) per state
    kw = dict(visits=128, c_puct=1.5, seed=1, virtual_batch_size=16, widening_policy_mass=0.95,
              widening_max_children=96, widening_min_children=2, fpu_reduction=0.2,
              virtual_loss=1.0, root_policy_temperature=1.0)
    sess = new_mcts_session(max_states=1 << 16, n=3)
    t0 = time.perf_counter()
    for s in probe:
        sess.run([0], [s], inf, **kw)
    move_us = (time.perf_counter() - t0) / len(probe) * 1e6
    print(f"one MCTS move (visits=128, cpu): ~{move_us:,.0f} us")
    m8, _, _, _, _ = bench(probe, 8, time_limit_ms=100)
    print(f"one root-only VCF call (ply=8):  ~{statistics.mean(m8):,.0f} us  "
          f"=> ~{statistics.mean(m8) / move_us * 100:.2f}% of a move if run root-only")
except Exception as exc:  # torch missing or other
    print("forward-cost comparison skipped:", exc)
