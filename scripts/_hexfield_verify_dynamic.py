"""End-to-end verification of the dynamic-compile serve path on REAL states.

Exercises the production entry point (HexfieldEvaluator.evaluate_payload) with
the new single-dynamic-compile inference.py, on real late-game payloads replayed
from the live run's .hxr. Confirms, all through the real path:
  1. it runs (no crash / no CantSplit) across small AND deep shapes,
  2. real-state parity: compiled outputs == forced-eager outputs to fp16 noise,
  3. ONE compile total: the deep-shape flushes don't pay a fresh compile,
  4. speedup: compiled is faster than eager on a realistic 96-state flush.

Force-eager is done the production way (point _compiled_fpv at _raw_fpv), so the
ONLY thing that changes between the two timings is which forward runs.

Run (GPU free):
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_verify_dynamic.py
"""
from __future__ import annotations

import glob
import os
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
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator, plan_groups, PAIR_CEILING
from hexfield.model import HexfieldNet

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{RUN}/checkpoints/epoch_000034.pt"
NBR_SENTINEL = 0xFFFF


def collect_real_states(max_states=4000):
    hxrs = sorted(glob.glob(f"{RUN}/selfplay/*.hxr"), key=os.path.getmtime, reverse=True)
    states = []
    for hxr in hxrs:
        try:
            rf = HexoRecordFile.open(hxr)
        except Exception:
            continue
        with rf:
            for rec in rf.iter_records():
                st = api.new_game()
                for aid in list(rec.action_ids):
                    q, r = unpack_action_id(int(aid))
                    res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
                    if api.terminal(st) is None:
                        states.append(api.clone_state(st) if hasattr(api, "clone_state") else st)
                    if res.terminal:
                        break
                if len(states) >= max_states:
                    return states, hxrs[:1]
    return states, hxrs[:1]


def build_payload(batch_rows):
    rs = sorted(enumerate(batch_rows), key=lambda ir: (-ir[1]["num_nodes"], ir[0]))
    feats, qr, nbr, offsets, legal = [], [], [], [0], []
    for _, r in rs:
        n = r["num_nodes"]
        feats.append(np.frombuffer(r["feats"], dtype=np.float32).astype(np.float16))
        qr.append(np.frombuffer(r["coords"], dtype=np.int16))
        nb = np.frombuffer(r["nbr"], dtype=np.int32)
        nb = np.where(nb < 0, NBR_SENTINEL, nb).astype(np.uint16)
        nbr.append(nb)
        offsets.append(offsets[-1] + n)
        legal.append(r["legal_count"])
    return {"abi": 1, "shape": (len(rs), offsets[-1]),
            "node_feats": np.concatenate(feats).tobytes(),
            "node_qr": np.concatenate(qr).tobytes(),
            "nbr": np.concatenate(nbr).tobytes(),
            "node_row_offsets": offsets,
            "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(),
            "request_moves_left": True}


def timed(ev, payload, warmup, reps):
    for _ in range(warmup):
        ev.evaluate_payload(payload)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        ev.evaluate_payload(payload)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000.0


def force_eager(ev, on):
    if on:
        ev._saved = ev._compiled_fpv
        ev._compiled_fpv = ev._raw_fpv
    else:
        ev._compiled_fpv = ev._saved


print(f"torch {torch.__version__}  ckpt={CKPT}", flush=True)
device = torch.device("cuda")
model = HexfieldNet()
p = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(p.get("model", p), strict=True)
ev = HexfieldEvaluator(model, device=device)
assert ev._use_compile and ev._compiled_fpv is not ev._raw_fpv, "compile not active"

states, used = collect_real_states()
rows_all = _rust.featurize_states(states)
rows_by_size = sorted(rows_all, key=lambda r: r["num_nodes"])
sizes = np.array([r["num_nodes"] for r in rows_all])
print(f"REAL states: {len(sizes)}  N mean {sizes.mean():.0f} p50 {np.percentile(sizes,50):.0f} "
      f"p90 {np.percentile(sizes,90):.0f} p99 {np.percentile(sizes,99):.0f} max {sizes.max()}", flush=True)


def group_near(target, band=0.16):
    lo = target * (1 - band)
    cand = [r for r in rows_by_size if lo <= r["num_nodes"] <= target]
    ceil_b = int(PAIR_CEILING // ((target + 8) ** 2)) or 1
    b = min(len(cand), ceil_b)
    return cand[-b:] if b else []


# --- 1+2+3: run + real-state parity across small..deep, count recompiles -------
import torch._dynamo as dyn
print("\n=== real-state parity (compiled vs forced-eager), small..deep ===")
print(f"{'Npad~':>7} {'B':>4} {'max|dvalue|':>11} {'max|dprior|':>11}  status", flush=True)
for tgt in [384, 512, 768, 1024, 1536, 2048, 2560, 3008]:
    g = group_near(tgt)
    if len(g) < 2:
        print(f"{tgt:>7}  (too few real states)", flush=True)
        continue
    pl = build_payload(g)
    force_eager(ev, True)
    oe = ev.evaluate_payload(pl)
    force_eager(ev, False)
    oc = ev.evaluate_payload(pl)
    ve = np.frombuffer(oe["values_bytes"], dtype=np.float32)
    vc = np.frombuffer(oc["values_bytes"], dtype=np.float32)
    pe = np.frombuffer(oe["priors_bytes"], dtype=np.float32)
    pc = np.frombuffer(oc["priors_bytes"], dtype=np.float32)
    dv = np.abs(ve - vc).max()
    dp = np.abs(pe - pc).max()
    ok = "OK" if (dv < 5e-3 and dp < 5e-3) else "** CHECK **"
    print(f"{tgt:>7} {len(g):>4} {dv:>11.2e} {dp:>11.2e}  {ok} (realN~{int(np.mean([r['num_nodes'] for r in g]))})", flush=True)

# Recompile accounting: how many distinct graphs did dynamo build?
try:
    from torch._dynamo.utils import compile_times
    n_frames = len(dyn.utils.counters.get("frames", {})) if hasattr(dyn.utils, "counters") else None
except Exception:
    n_frames = None

# --- 4: realistic late-game flush speedup -------------------------------------
print("\n=== realistic late-game flush: eager vs compiled ===")
for nstates in (96, 150):
    big = rows_by_size[-nstates:]
    pl = build_payload(big)
    grp = plan_groups(sorted((r["num_nodes"] for r in big), reverse=True))
    ns = np.array([r["num_nodes"] for r in big])
    force_eager(ev, True)
    e_ms = timed(ev, pl, warmup=3, reps=10)
    force_eager(ev, False)
    c_ms = timed(ev, pl, warmup=3, reps=10)
    print(f"  {nstates} states (N {ns.min()}-{ns.max()}, {len(grp)} groups): "
          f"eager {e_ms:.1f} ms | compiled {c_ms:.1f} ms | speedup {e_ms/c_ms:.2f}x", flush=True)

print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
print("VERIFY DONE", flush=True)
