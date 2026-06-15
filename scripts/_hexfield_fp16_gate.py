"""M8 fp16 adopt-and-gate: max value/policy divergence fp16-autocast vs fp32
on real mid-game states (seconds of GPU). Spec §9: fp16 eval weights behind
the adopt-and-gate; LN/softmax/CE stay fp32."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexfield import _rust
from hexfield.model import HexfieldNet
from _hexfield_perf import midgame_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HexfieldNet()
if len(sys.argv) > 1:
    payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
model.to(device).eval()

states = midgame_states(24, seed=5)
payloads = _rust.featurize_states(states)
vdiffs, pdiffs = [], []
with torch.no_grad():
    for row in payloads:
        n = row["num_nodes"]
        feats = np.frombuffer(row["feats"], dtype=np.float32).reshape(n, -1)
        qr = np.frombuffer(row["coords"], dtype=np.int16).reshape(n, 2)
        nbr = np.frombuffer(row["nbr"], dtype=np.int32).reshape(n, 6)
        f = torch.from_numpy(feats).unsqueeze(0).to(device)
        c = torch.from_numpy(qr.astype(np.int64)).unsqueeze(0).to(device)
        nb = torch.from_numpy(np.where(nbr < 0, n, nbr).astype(np.int64)).unsqueeze(0).to(device)
        m = torch.ones(1, n, dtype=torch.bool, device=device)
        o32 = model.forward_policy_value(f, nb, m, c)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            o16 = model.forward_policy_value(f, nb, m, c)
        legal = int(row["legal_count"])
        v32 = torch.softmax(o32["value"].float(), -1)
        v16 = torch.softmax(o16["value"].float(), -1)
        vdiffs.append(float((v32 - v16).abs().max()))
        p32 = torch.softmax(o32["policy"][0, :legal].float(), -1)
        p16 = torch.softmax(o16["policy"][0, :legal].float(), -1)
        pdiffs.append(float((p32 - p16).abs().max()))

print(f"device: {device}")
print(f"fp16-vs-fp32 value-prob max diff:  {max(vdiffs):.5f} (mean {np.mean(vdiffs):.5f})")
print(f"fp16-vs-fp32 policy-prob max diff: {max(pdiffs):.5f} (mean {np.mean(pdiffs):.5f})")
print("GATE: adopt fp16 if both maxima are small (~<0.02); else fall back to fp32 eval")
