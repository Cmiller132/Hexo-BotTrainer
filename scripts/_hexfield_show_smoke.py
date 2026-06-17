"""Summarize a hexfield smoke run's epoch result + checkpoint validity."""

import json
import sys
from pathlib import Path

import torch

run = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/e/Hexo-BotTrainer-hexgt/configs/runs/hexfield_smoke_tiny")

ep = json.loads((run / "diagnostics" / "epoch_000001.json").read_text())
res = ep["payload"]["metadata"]["result"] if "payload" in ep else ep
sp = res.get("selfplay", {})
tr = res.get("training", {})
print("selfplay:", {k: sp.get(k) for k in
                     ("status", "games_finished", "truncated_games", "rows_written",
                      "full_decisions", "mean_game_length")})
print("training:", {k: tr.get(k) for k in
                    ("status", "window_rows", "steps", "loss_total", "loss_policy", "loss_value")})
print("checkpoint:", res.get("checkpoint", {}).get("checkpoint_path"))

ckpt = torch.load(run / "checkpoints" / "epoch_000001.pt", map_location="cpu", weights_only=False)
print("ckpt keys:", sorted(ckpt.keys()), "meta:", ckpt.get("meta"))
print("ckpt model params:", sum(v.numel() for v in ckpt["model"].values()))
