"""REPRESENTATIVE throughput bench — matches the LIVE run's regime AND per-decision cost.

Two things make a naive bench over-state pos/s vs the live trainer (~3.3 pos/s on
96 mid-game games):
  1) Phase: starting 96 games from an empty board spends the window in the cheap
     early-game ramp. FIX: seed each game from a real mid/late position (replay a
     random-length prefix of a random real .hxr game) so the support distribution
     matches the live run from t=0.
  2) Per-decision host work: the live ContinuousDriver.__call__ runs, ON THE SEARCH
     THREAD, window_scan(tuple(records), ...) + HexfieldSampleData construction over
     the game's whole GROWING record list every FULL (PCR) decision — O(game length),
     which the levers do NOT speed up. FIX: replicate that exact per-decision work
     here (file WRITING stays off, as in prod it is on the background writer thread).
A FIXED seed makes the 96 initial games + every refill identical run-to-run so pos/s
deltas between code versions are comparable at low variance.

  python _hexfield_repr_bench.py <ckpt> <seconds> <active_games> <out.json> <refill> [overhead]
     refill = "real" | "fresh"   (real = refill from a fresh real mid-game seed)
     overhead = "full" (default, replicate driver per-decision cost) | "none"
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile

from hexfield import _rust
from hexfield.engine_facts import player_int
from hexfield.features import record_phase, record_player, window_scan
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet
from hexfield.samples import HexfieldSampleData

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPT = sys.argv[1]
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
ACTIVE = int(sys.argv[3]) if len(sys.argv) > 3 else 96
OUT = sys.argv[4] if len(sys.argv) > 4 else "/tmp/hexfield_repr_bench.json"
REFILL_MODE = sys.argv[5] if len(sys.argv) > 5 else "real"
OVERHEAD = sys.argv[6] if len(sys.argv) > 6 else "full"
SIZE_FLOOR = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # min support per seeded game (0 = mix)
SEED = 20260616
FLUSH_TARGET = 1024
CACHE = 262144
VISITS = 512


def load_real_action_seqs(max_games=4000):
    seqs = []
    for hxr in sorted(glob.glob(f"{RUN}/selfplay/*.hxr"), key=os.path.getmtime, reverse=True):
        try:
            rf = HexoRecordFile.open(hxr)
        except Exception:
            continue
        with rf:
            for rec in rf.iter_records():
                aids = [int(a) for a in rec.action_ids]
                if len(aids) >= 4:
                    seqs.append(aids)
                if len(seqs) >= max_games:
                    return seqs
    return seqs


SEQS = load_real_action_seqs()
assert SEQS, "no real .hxr games found to seed from"
# When seeding LARGE games, draw from the LONG games (deep plies -> large support).
LONG_SEQS = [s for s in SEQS if len(s) >= 120] or SEQS


def _replay(seq, target):
    """Replay seq[:target] -> (state, records, ply) or None if it went terminal."""
    st = api.new_game()
    records: list[tuple[int, int, int, int]] = []
    for i in range(min(target, len(seq))):
        q, r = unpack_action_id(int(seq[i]))
        cur = record_player(i)
        res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
        records.append((q, r, cur, i + 1))
        if res.terminal or api.terminal(st) is not None:
            return None
    if api.terminal(st) is not None:
        return None
    return st, records, len(records)


def seed_game(rng):
    """Seed a real position. If SIZE_FLOOR>0, keep drawing DEEP prefixes of LONG
    games until the legal-move support is >= SIZE_FLOOR (so all 96 games are large),
    else a random-ply mix. records carries the (q,r,player,ply) history so the
    window_scan per-decision cost matches a real deep-game decision."""
    pool = LONG_SEQS if SIZE_FLOOR > 0 else SEQS
    best = None
    best_sup = -1
    for _ in range(40):
        seq = rng.choice(pool)
        if SIZE_FLOOR > 0:
            target = rng.randint(max(1, int(len(seq) * 0.55)), len(seq) - 1)
        else:
            target = rng.randint(1, len(seq) - 1)
        out = _replay(seq, target)
        if out is None:
            continue
        sup = len(api.legal_action_ids(out[0]))
        if sup > best_sup:
            best, best_sup = out, sup
        if sup >= SIZE_FLOOR:
            return out
    return best if best is not None else (api.new_game(), [], 0)


device = torch.device("cuda")
model = HexfieldNet()
p = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(p.get("model", p), strict=True)
evaluator = HexfieldEvaluator(model, device=device)
print(f"loaded {CKPT}; window={SECONDS}s active={ACTIVE} refill={REFILL_MODE} overhead={OVERHEAD} "
      f"seeded from {len(SEQS)} real games", flush=True)

rng = random.Random(SEED)
states, records, plies = {}, {}, {}
for k in range(ACTIVE):
    st, recs, ply = seed_game(rng)
    states[k], records[k], plies[k] = st, recs, ply
ss = np.array(sorted(len(api.legal_action_ids(s)) for s in states.values()))
print(f"INITIAL seed support N: mean {ss.mean():.0f} p10 {np.percentile(ss,10):.0f} "
      f"p50 {np.percentile(ss,50):.0f} p90 {np.percentile(ss,90):.0f} max {ss.max()}  "
      f"ply mean {np.mean(list(plies.values())):.0f}", flush=True)

n = {"d": 0, "f": 0, "next_key": ACTIVE}
support_hist: dict[int, int] = {}
pending: dict[int, list] = {k: [] for k in range(ACTIVE)}
deadline = time.time() + SECONDS
MAX_PLIES = 256


def on_move(game_key, payload):
    n["d"] += 1
    st = states[game_key]
    recs = records[game_key]
    ply = plies[game_key]
    s = len(api.legal_action_ids(st))
    support_hist[((s // 256) + 1) * 256] = support_hist.get(((s // 256) + 1) * 256, 0) + 1

    # --- replicate ContinuousDriver.__call__ per-decision host work ---
    if OVERHEAD == "full":
        current = record_player(ply)
        full = bool(payload.get("pcr_full", False))
        init = bool(payload.get("policy_init", False))
        if full and not init:
            n["f"] += 1
            ids = np.frombuffer(bytes(payload["visit_policy_action_ids_bytes"]), dtype=np.uint32)
            weights = np.frombuffer(bytes(payload["visit_policy_weights_bytes"]), dtype=np.float32)
            phase = record_phase(ply)
            first_stone = (recs[-1][0], recs[-1][1]) if (phase == "SecondStone" and recs) else None
            own_hot, opp_hot, own_win, opp_win = window_scan(tuple(recs), current, len(recs))
            sample = HexfieldSampleData(
                game_id=str(game_key), turn_index=ply, current_player=current, phase=phase,
                records=tuple(recs), first_stone=first_stone, own_hot=own_hot, opp_hot=opp_hot,
                own_win=own_win, opp_win=opp_win,
                policy=tuple(zip((int(a) for a in ids), (float(w) for w in weights))),
                metadata={"pcr_full": True})
            pending[game_key].append((current, sample, float(payload.get("root_value", 0.0))))
        else:
            sample = HexfieldSampleData(
                game_id=str(game_key), turn_index=ply, current_player=current,
                phase=record_phase(ply), records=tuple(recs), first_stone=None,
                own_hot=(), opp_hot=(), own_win=(), opp_win=(), policy=(),
                metadata={"pcr_full": False, "policy_init": init})
            pending[game_key].append((current, sample, float(payload.get("root_value", 0.0))))

    q, r = unpack_action_id(int(payload["action_id"]))
    cur = record_player(ply)
    res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
    recs.append((q, r, cur, ply + 1))
    plies[game_key] = ply + 1
    ended = res.terminal or plies[game_key] >= MAX_PLIES
    if not ended:
        if time.time() > deadline:
            return None
        return ("advance", st)
    if time.time() <= deadline:
        nk = n["next_key"]; n["next_key"] += 1
        if REFILL_MODE == "real":
            states[nk], records[nk], plies[nk] = seed_game(rng)
        else:
            states[nk], records[nk], plies[nk] = api.new_game(), [], 0
        pending[nk] = []
        return ("replace", nk, states[nk])
    return None


# GPU-util sampler thread (reliable in-process measurement of saturation).
import subprocess
import threading
_gpu_samples: list[float] = []
_gpu_stop = threading.Event()


def _gpu_sampler():
    while not _gpu_stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            _gpu_samples.append(float(out.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        _gpu_stop.wait(0.5)


session = _rust.HexfieldMctsSession(max_states=CACHE)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
_sampler = threading.Thread(target=_gpu_sampler, daemon=True)
_sampler.start()
t0 = time.time()
stats = session.run_continuous(
    list(range(ACTIVE)), tuple(states.values()), evaluator=evaluator, on_move=on_move,
    visits=VISITS, c_puct=1.5, base_seed=7, virtual_batch_size=4,
    flush_target=FLUSH_TARGET, active_root_limit=ACTIVE,
    temperature_by_ply=[1.0] * 8 + [0.3] * 400,
    forced_playout_k=2.0, widening_policy_mass=0.95, widening_max_children=96,
    widening_min_children=2, root_dirichlet_total_alpha=10.83,
    root_dirichlet_noise_fraction=0.25, pcr_full_proportion=0.33, pcr_fast_visits=128,
    policy_init_fraction=0.25, policy_init_avg_plies=4.0, policy_init_max_plies=8,
    policy_init_temperature=1.4,
    tss_enabled=True, fpu_reduction=0.2,  # match the live config (§5.1)
)
torch.cuda.synchronize()
dt = time.time() - t0
_gpu_stop.set()
peak = torch.cuda.max_memory_allocated() / 2**30
import numpy as _np
_g = _np.array(_gpu_samples) if _gpu_samples else _np.array([0.0])
print(f"  GPU util: avg {_g.mean():.0f}% max {_g.max():.0f}% min {_g.min():.0f}%  "
      f">=90%:{int((_g>=90).sum())}/{len(_g)} <=70%:{int((_g<=70).sum())}/{len(_g)}", flush=True)
rec = {
    "active_games": ACTIVE, "seconds": round(dt, 1), "decisions": n["d"], "full_decisions": n["f"],
    "pos_per_s": round(n["d"] / dt, 2),
    "evals_per_s": round(stats.get("flushed_states", 0) / dt, 0),
    "mean_flush": round(stats.get("mean_flush_states", 0), 1),
    "peak_vram_gib": round(peak, 2), "refill": REFILL_MODE, "overhead": OVERHEAD,
    "support_hist_256": dict(sorted(support_hist.items())),
}
Path(OUT).write_text(json.dumps([rec], indent=2), encoding="utf-8")
print(f"  pos/s={rec['pos_per_s']}  evals/s={rec['evals_per_s']:.0f}  mean_flush={rec['mean_flush']}  "
      f"peak_vram={rec['peak_vram_gib']}GiB  full_dec={n['f']}", flush=True)
print(f"  support_hist(256-buckets)={rec['support_hist_256']}", flush=True)
print(f"wrote {OUT}", flush=True)
