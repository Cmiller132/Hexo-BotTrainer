"""Measure REAL self-play support-size distribution + eval-only throughput at
those sizes (vs the random-playout benchmark which over-spreads stones). Replays
actual self-play games from a .hxr, collects mid/late-game states, featurizes,
and times HexfieldEvaluator.evaluate_payload. Resolves model-cost vs scheduler-
cost by comparing eval-only evals/s to the self-play evals/s (~373)."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import numpy as np, torch
from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexfield import _rust
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

HXR = sys.argv[1]
CKPT = sys.argv[2] if len(sys.argv) > 2 else None

# Replay real games; collect a state at every ply (the real eval distribution).
states = []
with HexoRecordFile.open(HXR) as rf:
    for rec in rf.iter_records():
        st = api.new_game()
        for aid in list(rec.action_ids):
            q, r = unpack_action_id(int(aid))
            res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
            if api.terminal(st) is None:
                states.append(api.clone_state(st) if hasattr(api, "clone_state") else st)
            if res.terminal:
                break
        if len(states) >= 1500:
            break

# Support-size distribution over real states.
rows = _rust.featurize_states(states)
sizes = np.array([r["num_nodes"] for r in rows])
print(f"REAL self-play states: {len(sizes)}")
print(f"support N  mean {sizes.mean():.0f}  p50 {np.percentile(sizes,50):.0f}  "
      f"p90 {np.percentile(sizes,90):.0f}  p99 {np.percentile(sizes,99):.0f}  max {sizes.max()}")

# Eval-only throughput at the REAL distribution (flush 256, size-desc packed).
def build_payload(batch):
    rs = _rust.featurize_states(batch)
    rs = sorted(enumerate(rs), key=lambda ir: (-ir[1]["num_nodes"], ir[0]))
    feats, qr, nbr, offsets, legal = [], [], [], [0], []
    for _, r in rs:
        n = r["num_nodes"]
        feats.append(np.frombuffer(r["feats"], dtype=np.float32).astype(np.float16))
        qr.append(np.frombuffer(r["coords"], dtype=np.int16))
        nb = np.frombuffer(r["nbr"], dtype=np.int32); nb = np.where(nb < 0, 0xFFFF, nb).astype(np.uint16)
        nbr.append(nb); offsets.append(offsets[-1] + n); legal.append(r["legal_count"])
    return {"abi": 1, "shape": (len(rs), offsets[-1]),
            "node_feats": np.concatenate(feats).tobytes(), "node_qr": np.concatenate(qr).tobytes(),
            "nbr": np.concatenate(nbr).tobytes(), "node_row_offsets": offsets,
            "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(), "request_moves_left": True}

device = torch.device("cuda")
model = HexfieldNet()
if CKPT:
    p = torch.load(CKPT, map_location="cpu", weights_only=False); model.load_state_dict(p.get("model", p), strict=True)
ev = HexfieldEvaluator(model, device=device)
sample = states[:512]
payloads = [build_payload(sample[i:i+256]) for i in range(0, len(sample), 256)]
ev.evaluate_payload(payloads[0]); torch.cuda.synchronize()
t0 = time.time(); reps = 3
for _ in range(reps):
    for pl in payloads: ev.evaluate_payload(pl)
torch.cuda.synchronize()
dt = (time.time() - t0) / reps
n_eval = len(sample)
print(f"EVAL-ONLY at real support: {n_eval/dt:.0f} evals/s  (flush seconds {dt:.3f})")
print(f"  vs production self-play ~373 evals/s -> ratio {(n_eval/dt)/373:.1f}x")
print(f"  (eval-only >> 373 => SCHEDULER-bound; ~= 373 => MODEL-bound)")
