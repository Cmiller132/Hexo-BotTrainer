"""Rebuild diagnostics/epoch_000011.json so the dashboard shows epoch 11's
loss + self-play (it currently shows "failed": the original epoch crashed in
training before the framework wrote a result, and the retrain only wrote the
trainer-level hexfield.training.epoch_000011.json, not the framework result
the dashboard reads from).

The dashboard's epoch table (_epoch_history in web.py) reads training loss from
diagnostics/epoch_NNNNNN.json -> metadata.result.training, NOT from
hexfield.training.epoch_NNNNNN.json. So we assemble the real self-play +
training (+ checkpoint) blocks into the framework-shaped result file, mirroring
a normal epoch_000010.json. Evaluation is intentionally OMITTED: epoch 11 never
played eval games (crashed before eval; retrain was training-only), so the eval
ladder honestly skips 11 -> 12 rather than carrying a fabricated win rate. The
original failed stub is backed up to epoch_000011.json.crashed.bak.
"""
import json
from pathlib import Path

DIAG = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/diagnostics")
CKPT = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000011.pt"
TARGET = DIAG / "epoch_000011.json"

selfplay = json.loads((DIAG / "hexfield.selfplay.epoch_000011.json").read_text(encoding="utf-8"))
training = json.loads((DIAG / "hexfield.training.epoch_000011.json").read_text(encoding="utf-8"))

# Back up the crash stub (once) for the record.
if TARGET.exists():
    bak = DIAG / "epoch_000011.json.crashed.bak"
    if not bak.exists():
        bak.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
        print("backed up crash stub ->", bak.name)

result = {
    "epoch": 11,
    "selfplay": selfplay,
    "samples": {
        "finalize": {
            "status": "skipped",
            "epoch": 11,
            "reason": "model finalizes samples during self-play",
        },
        "selection": {
            "epoch": 11,
            "shard_count": None,
            "keep_target_rows": training.get("window_rows"),
        },
    },
    "training": training,
    "checkpoint": {
        "checkpoint_path": CKPT,
        "name": "epoch_000011",
        "epoch": 11,
        "kind": "epoch",
    },
    # NOTE: no "evaluation" key on purpose -- eval did not run for epoch 11
    # (original crashed pre-eval; recovery was a training-only retrain on the
    # already-saved self-play games). The eval ladder resumes at epoch 12.
    "recovery_note": "epoch_000011.json reconstructed from saved self-play + "
    "retrain training diagnostics after the optimizer-device crash; "
    "training redone from the 511 saved .npz shards, eval skipped.",
}

elapsed = round(float(selfplay.get("elapsed_seconds", 0.0)) + float(training.get("seconds", 0.0)), 2)
payload = {
    "stage": "epoch_000011",
    "status": "completed",
    "elapsed_seconds": elapsed,
    "metadata": {"result": result},
}

TARGET.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("WROTE", TARGET.name, f"({TARGET.stat().st_size} bytes)")
print("  selfplay games_finished:", selfplay.get("games_finished"),
      "rows_written:", selfplay.get("rows_written"))
print("  training loss_total:", training.get("loss_total"),
      "loss_policy:", training.get("loss_policy"),
      "loss_value:", training.get("loss_value"))
print("  elapsed_seconds:", elapsed, "| evaluation: OMITTED (did not run)")
