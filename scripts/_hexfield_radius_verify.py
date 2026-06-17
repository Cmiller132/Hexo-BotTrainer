"""Verify HEXFIELD_SUPPORT_RADIUS: does the Rust featurizer shrink the support?

Featurizes real deep states via _rust.featurize_states (which reads
HEXFIELD_SUPPORT_RADIUS once) and reports num_nodes / legal_count. Run with the
env unset (default 8 == today) and =4 to compare. Also confirms a serve forward
runs on the radius-4 support (checkpoint loads, no crash).
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import numpy as np
from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexfield.geometry import unpack_action_id
from hexfield import _rust

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"


def collect(n=60, min_ply=110):
    states = []
    for hxr in sorted(glob.glob(f"{RUN}/selfplay/*.hxr"), key=os.path.getmtime, reverse=True):
        try:
            rf = HexoRecordFile.open(hxr)
        except Exception:
            continue
        with rf:
            for rec in rf.iter_records():
                aids = [int(a) for a in rec.action_ids]
                if len(aids) < min_ply:
                    continue
                st = api.new_game()
                for a in aids[: len(aids) - 4]:
                    q, r = unpack_action_id(a)
                    api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
                if api.terminal(st) is None:
                    states.append(st)
                if len(states) >= n:
                    return states
    return states


R = os.environ.get("HEXFIELD_SUPPORT_RADIUS", "unset(8)")
states = collect()
rows = _rust.featurize_states(states)
N = np.array([r["num_nodes"] for r in rows])
L = np.array([r["legal_count"] for r in rows])
print(f"HEXFIELD_SUPPORT_RADIUS={R}: rust featurize over {len(states)} deep states -> "
      f"support N mean {N.mean():.0f} p50 {np.median(N):.0f} max {N.max()} | legal mean {L.mean():.0f}",
      flush=True)

if os.environ.get("RUN_FWD") == "1":
    import torch
    from hexfield.inference import HexfieldEvaluator
    from hexfield.model import HexfieldNet
    NBR_SENTINEL = 0xFFFF
    m = HexfieldNet()
    p = torch.load(f"{RUN}/checkpoints/epoch_000034.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(p.get("model", p), strict=True)
    ev = HexfieldEvaluator(m, device=torch.device("cuda"))
    rs = sorted(rows, key=lambda r: -r["num_nodes"])[:32]
    feats, qr, nbr, offs, legal = [], [], [], [0], []
    for r in rs:
        n = r["num_nodes"]
        feats.append(np.frombuffer(r["feats"], dtype=np.float16))
        qr.append(np.frombuffer(r["coords"], dtype=np.int16))
        nb = np.frombuffer(r["nbr"], dtype=np.int32)
        nbr.append(np.where(nb < 0, NBR_SENTINEL, nb).astype(np.uint16))
        offs.append(offs[-1] + n); legal.append(r["legal_count"])
    payload = {"abi": 1, "shape": (len(rs), offs[-1]),
               "node_feats": np.concatenate(feats).tobytes(),
               "node_qr": np.concatenate(qr).tobytes(),
               "nbr": np.concatenate(nbr).tobytes(),
               "node_row_offsets": offs,
               "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(),
               "request_moves_left": True}
    out = ev.evaluate_payload(payload)
    v = np.frombuffer(out["values_bytes"], dtype=np.float32)
    pr = np.frombuffer(out["priors_bytes"], dtype=np.float32)
    print(f"  serve forward OK: values {len(v)} (finite={np.isfinite(v).all()}), "
          f"priors {len(pr)} (sum~{pr.sum():.0f}, finite={np.isfinite(pr).all()})", flush=True)
print("VERIFY DONE", flush=True)
