"""Run the moves-left head L0 audit on a checkpoint (the §5.4.4 heal-gate).

Decodes the model's moves-left head on recent self-play shards and scores its
calibration (within-game conversion-zone Spearman, monotonicity, near-end MAE)
vs the true remaining-decisions target the shards carry. A flood-damaged head
(chance-level ordering) would steer the MLH search lever BACKWARD, so this is
the gate for trusting the lever ON. Read-only; one model forward, no MCTS.

Usage: python _hexfield_run_head_audit.py CKPT.pt SAMPLES_DIR [MAX_GAMES] [OUT_JSON]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "packages" / "dense_cnn_restnet" / "python"))

import torch  # noqa: E402

torch.set_grad_enabled(False)

from hexfield.head_audit import audit_moves_left_head  # noqa: E402
from hexfield.model import HexfieldNet  # noqa: E402

CKPT = Path(sys.argv[1])
SAMPLES = Path(sys.argv[2])
MAX_GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = HexfieldNet()
payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
model.load_state_dict(payload["model"], strict=True)

shards = sorted(SAMPLES.glob("epoch_*/game_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
print(f"ckpt={CKPT.name} device={DEVICE} shards_available={len(shards)} max_games={MAX_GAMES}", flush=True)

result = audit_moves_left_head(model, shards, device=DEVICE, max_games=MAX_GAMES)
print(json.dumps(result, indent=2), flush=True)
verdict = "PASS — head is calibrated, MLH lever safe to trust" if result.get("passed") else "FAIL — head miscalibrated, lever should stay disabled until it heals"
print(f"\nL0 VERDICT: {verdict}", flush=True)
if OUT:
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
