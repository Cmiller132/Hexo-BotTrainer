"""Retrain epoch 11 from the SAVED self-play games (no self-play regen).

Loads epoch_000010.pt (model+optimizer, the real resume path: load on CPU), then
runs HexfieldTrainer.train_passes over the saved shards (the 32k recent-shard
window) and writes epoch_000011.pt. This both (a) uses the already-saved 511
self-play games instead of regenerating ~45 min of self-play, and (b) exercises
the train_passes optimizer-device fix on REAL data. After this the supervisor
resumes from epoch_000011.pt and continues at epoch 12.
"""
import sys, tomllib
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "packages" / "dense_cnn_restnet" / "python"))

import torch
from hexfield.config import parse_hexfield_config
from hexfield.model import HexfieldNet
from hexfield.checkpoints import load_into, save_checkpoint
from hexfield.trainer import HexfieldTrainer

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1")
CKPT_IN = RUN / "checkpoints" / "epoch_000010.pt"
CKPT_OUT = RUN / "checkpoints" / "epoch_000011.pt"
CONFIG = REPO / "configs" / "hexfield_main_1.toml"

cfg = parse_hexfield_config(tomllib.loads(CONFIG.read_text(encoding="utf-8"))["model"]["config"])
print("device", cfg.device, "batch_rows", cfg.training.batch_rows,
      "keep_rows", cfg.training.shuffle_keep_target_rows)

# Build model + optimizer; load epoch 10 (model on CPU at load -> the real path).
# Optimizer MUST match the plugin's 2-group decay/no_decay split or
# load_state_dict rejects the saved state ("different number of parameter groups").
model = HexfieldNet()
decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim >= 2 and "bias_table" not in name and name != "tokens":
        decay.append(param)
    else:
        no_decay.append(param)
optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": cfg.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=cfg.training.learning_rate,
)
payload = torch.load(str(CKPT_IN), map_location="cpu", weights_only=False)
meta = load_into(model, payload, optimizer=optimizer)
print("loaded epoch", meta.get("epoch"), "optimizer state groups", len(optimizer.state))

trainer = HexfieldTrainer(model=model, config=cfg, optimizer=optimizer)
if not hasattr(trainer, "global_step"):
    trainer.global_step = 0

ctx = SimpleNamespace(samples_dir=RUN / "samples", diagnostics_dir=RUN / "diagnostics")
n_shards = len(list((RUN / "samples").glob("epoch_*/game_*.npz")))
print("saved shards available:", n_shards)

result = trainer.train_passes(passes=1, sample_window=None, sample_symmetries=None,
                              ctx=ctx, components=None, epoch=11)
print("TRAIN RESULT:", {k: result[k] for k in
      ("status", "window_rows", "steps", "seconds", "loss_total", "loss_policy", "loss_value")
      if k in result})

save_checkpoint(CKPT_OUT, model=model, optimizer=optimizer, epoch=11,
                extra={"run": cfg.__dict__.get("run", "hexfield_main_1"),
                       "note": "retrained from saved self-play games (no regen)"})
print("WROTE", CKPT_OUT, "exists:", CKPT_OUT.exists())
