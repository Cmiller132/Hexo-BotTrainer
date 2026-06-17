"""Does torch.compile help the LARGE-Npad serve forward, or only small shapes?

The serve path gates compile behind Npad <= HEXFIELD_COMPILE_MAX_NPAD (default
512); large late-game groups run eager. This bench answers, on REAL late-game
states (replayed from the live run's .hxr), whether compiling large shapes is a
win, a wash, or a loss -- BEFORE anyone flips the cutoff on the hot run.

Method: one evaluator, real payloads. We flip ev._compile_max_npad between -1
(force eager) and 1e9 (force compiled) so the ONLY difference is which forward
runs -- identical host pack / H2D / D2H. We warm up each shape (compile happens
on first call per static Npad), then time steady-state. Reports per-Npad ms and
the compiled/eager ratio (>1 = compile is SLOWER), plus a realistic 96-state
late-game flush, plus a numeric parity check.

Run (GPU should be free):
    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_compile_largeN_bench.py
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

# Bump the dynamo recompile ceiling so the many distinct large Npad in the
# late-mix flush all compile (this is exactly the cost a real cutoff-bump pays).
torch._dynamo.config.cache_size_limit = 512


def collect_real_states(max_states=4000):
    """Replay real self-play games, keep a non-terminal state at every ply."""
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
    """batch_rows: list of featurized _rust rows. Size-desc, CSR flat-concat."""
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


def timed(ev, payload, warmup=5, reps=20):
    for _ in range(warmup):
        ev.evaluate_payload(payload)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        ev.evaluate_payload(payload)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000.0


print(f"loading {CKPT}", flush=True)
states, used = collect_real_states()
rows_all = _rust.featurize_states(states)
sizes = np.array([r["num_nodes"] for r in rows_all])
print(f"REAL states: {len(sizes)} from {used[0].split('/')[-1]} ...")
print(f"support N: mean {sizes.mean():.0f}  p50 {np.percentile(sizes,50):.0f}  "
      f"p90 {np.percentile(sizes,90):.0f}  p99 {np.percentile(sizes,99):.0f}  max {sizes.max()}", flush=True)

device = torch.device("cuda")
model = HexfieldNet()
p = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(p.get("model", p), strict=True)
ev = HexfieldEvaluator(model, device=device)
assert ev._use_compile and ev._compiled_fpv is not ev._raw_fpv, "compile not active (need CUDA, no HEXFIELD_NO_COMPILE)"

rows_by_size = sorted(rows_all, key=lambda r: r["num_nodes"])


def group_near(target, band=0.16, max_b=None):
    """Pick real rows whose N is in [target*(1-band), target] so plan_groups
    pads them to ONE 64-quant anchor; B capped to the pair ceiling."""
    lo = target * (1 - band)
    cand = [r for r in rows_by_size if lo <= r["num_nodes"] <= target]
    ceil_b = int(PAIR_CEILING // ((target + 8) ** 2)) or 1
    b = min(len(cand), ceil_b if max_b is None else min(ceil_b, max_b))
    return cand[-b:] if b else []


print("\n=== per-Npad: real states, eager vs compiled (full evaluate_payload) ===")
print(f"{'Npad~':>7} {'B':>4} {'groups':>6} {'eager_ms':>9} {'compiled_ms':>11} {'ratio':>6}  verdict")
TARGETS = [384, 512, 768, 1024, 1536, 2048, 2560, 3008]
for tgt in TARGETS:
    g = group_near(tgt)
    if len(g) < 2:
        print(f"{tgt:>7} {'--':>4}  (too few real states near this size)")
        continue
    pl = build_payload(g)
    ng = len(plan_groups(sorted((r["num_nodes"] for r in g), reverse=True)))
    ev._compile_max_npad = -1          # force eager
    e_ms = timed(ev, pl)
    ev._compile_max_npad = 10 ** 9     # force compiled (pays warmup inside timed())
    c_ms = timed(ev, pl)
    ratio = c_ms / e_ms
    verdict = "compile WINS" if ratio < 0.91 else ("WASH" if ratio <= 1.05 else "compile LOSES")
    realN = int(np.mean([r["num_nodes"] for r in g]))
    print(f"{tgt:>7} {len(g):>4} {ng:>6} {e_ms:>9.2f} {c_ms:>11.2f} {ratio:>6.2f}  {verdict} (realN~{realN})", flush=True)

# Realistic late-game flush: the 96 + 150 largest real states (mirrors a late
# ac96 flush where finished slots have drained into big supports).
print("\n=== realistic late-game flush (largest real states) ===")
for nstates in (96, 150):
    big = rows_by_size[-nstates:]
    pl = build_payload(big)
    grp = plan_groups(sorted((r["num_nodes"] for r in big), reverse=True))
    ns = np.array([r["num_nodes"] for r in big])
    ev._compile_max_npad = -1
    e_ms = timed(ev, pl, reps=12)
    ev._compile_max_npad = 10 ** 9
    c_ms = timed(ev, pl, reps=12)
    print(f"  {nstates} states (N {ns.min()}-{ns.max()}, {len(grp)} groups): "
          f"eager {e_ms:.1f} ms | compiled {c_ms:.1f} ms | ratio {c_ms/e_ms:.2f}", flush=True)

# Numeric parity: compile must not change outputs.
pl = build_payload(rows_by_size[-32:])
ev._compile_max_npad = -1
out_e = ev.evaluate_payload(pl)
ev._compile_max_npad = 10 ** 9
out_c = ev.evaluate_payload(pl)
ve = np.frombuffer(out_e["values_bytes"], dtype=np.float32)
vc = np.frombuffer(out_c["values_bytes"], dtype=np.float32)
pe = np.frombuffer(out_e["priors_bytes"], dtype=np.float32)
pc = np.frombuffer(out_c["priors_bytes"], dtype=np.float32)
print(f"\nparity (eager vs compiled): max|dvalue|={np.abs(ve-vc).max():.2e}  "
      f"max|dprior|={np.abs(pe-pc).max():.2e}")
print(f"peak VRAM this process: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
