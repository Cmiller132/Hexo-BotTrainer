#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python -u - <<'PY'
import sys, torch
from pathlib import Path
sys.path.insert(0, "scripts")
import _head_to_head as h
# CPU construction only (no games, no GPU) to validate wiring while BC trains on GPU.
dev = torch.device("cpu")
ck = sorted(Path("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc").glob("hexgt_bc_step*.pt"))
hexgt_ck = str(ck[-1]) if ck else "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_latest.pt"
print("hexgt ckpt:", hexgt_ck)
mh = h.build_hexgt(hexgt_ck, 8, dev, compile_model=False)
md = h.build_densecnn("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/checkpoints/epoch_000024.pt",
                      "configs/dense_cnn_model1_target_96x8.toml", 8, dev)
pa = mh(1); pb = md(1)
print("hexgt player:", type(pa).__name__, "| dense player:", type(pb).__name__)
print("CONSTRUCTION OK")
PY
