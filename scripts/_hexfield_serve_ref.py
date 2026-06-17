"""Deterministic serve-output parity harness for the throughput levers.

Builds a FIXED set of real-state payloads (incl. size-1 deep groups) and either
SAVES the evaluate_payload outputs as a baseline, or CHECKS the current code's
outputs against that baseline. Used to gate every lever change (sync-fix must be
bit-identical; decode-fuse must be fp16-close).

  save:  python _hexfield_serve_ref.py save  /tmp/ref.npz
  check: python _hexfield_serve_ref.py check /tmp/ref.npz <tol>   # prints PASS/FAIL
"""
from __future__ import annotations

import glob
import os
import sys
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
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPT = os.environ.get("HEXFIELD_REF_CKPT", f"{RUN}/checkpoints/epoch_000034.pt")
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
                    return states
    return states


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


def fixed_payloads():
    """A deterministic battery: several size bands (each padded to one anchor ->
    multi-row groups) PLUS the largest singletons (size-1 groups), PLUS a big
    realistic late-game flush. Sorted/seeded so it is identical run-to-run."""
    rows = _rust.featurize_states(collect_real_states())
    rows_by_size = sorted(rows, key=lambda r: r["num_nodes"])
    pls = []
    # narrow same-size bands (multi-row groups), deterministic slices
    for lo, hi, take in [(300, 360, 24), (480, 540, 24), (740, 800, 16),
                         (1000, 1080, 8), (1480, 1580, 6)]:
        band = [r for r in rows_by_size if lo <= r["num_nodes"] <= hi][:take]
        if len(band) >= 2:
            pls.append(build_payload(band))
    # the largest singletons -> size-1 groups (pad-batch path)
    for r in rows_by_size[-3:]:
        pls.append(build_payload([r]))
    # realistic 96-state late-game flush
    pls.append(build_payload(rows_by_size[-96:]))
    return pls


def run_all(ev, pls):
    outs = []
    for pl in pls:
        o = ev.evaluate_payload(pl)
        outs.append((np.frombuffer(o["values_bytes"], dtype=np.float32).copy(),
                     np.frombuffer(o["priors_bytes"], dtype=np.float32).copy(),
                     np.frombuffer(o.get("moves_left_bytes", b""), dtype=np.float32).copy()))
    return outs


def main():
    mode, path = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    device = torch.device("cuda")
    model = HexfieldNet()
    p = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(p.get("model", p), strict=True)
    ev = HexfieldEvaluator(model, device=device)
    pls = fixed_payloads()
    outs = run_all(ev, pls)
    if mode == "save":
        flat = {}
        for i, (v, pr, ml) in enumerate(outs):
            flat[f"v{i}"] = v; flat[f"p{i}"] = pr; flat[f"m{i}"] = ml
        np.savez(path, n=len(outs), **flat)
        print(f"SAVED {len(outs)} payload outputs -> {path}", flush=True)
        return
    ref = np.load(path)
    dv = dp = dm = 0.0
    for i, (v, pr, ml) in enumerate(outs):
        dv = max(dv, float(np.abs(v - ref[f"v{i}"]).max()) if v.size else 0.0)
        dp = max(dp, float(np.abs(pr - ref[f"p{i}"]).max()) if pr.size else 0.0)
        dm = max(dm, float(np.abs(ml - ref[f"m{i}"]).max()) if ml.size else 0.0)
    ok = dv <= tol and dp <= tol and dm <= max(tol, 1e-3) * 512  # ml is in [0,512]
    print(f"PARITY max|dvalue|={dv:.3e} max|dprior|={dp:.3e} max|dml|={dm:.3e} tol={tol:.1e} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
