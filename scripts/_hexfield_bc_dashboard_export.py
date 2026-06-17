"""Mirror the BC prefit into a dashboard-visible run dir (idempotent).

The :8080 dashboard scans run dirs under its cwd (/mnt/e/Hexo-BotTrainer/runs)
for manifest.json, diagnostics/events.jsonl, and checkpoints/epoch_*.pt. The
prefit is a standalone trainer, so this exporter rebuilds a minimal compatible
mirror from runs/hexfield_bc_1/{diagnostics.jsonl,checkpoint_epoch*.pt} every
time it runs. Safe to re-run any time (it rewrites the mirror from scratch);
the production hexfield run will NOT need this — hexo_train writes the full
layout natively once configs/hexfield_main_1.toml points output_dir at the
dashboard mount.

Run (WSL): /root/.venvs/hexgt-build/bin/python scripts/_hexfield_bc_dashboard_export.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC = Path("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1")
DST = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_bc_1")


def main() -> None:
    diag_path = SRC / "diagnostics.jsonl"
    if not diag_path.is_file():
        print("no diagnostics yet; nothing to export")
        return
    rows = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    (DST / "diagnostics").mkdir(parents=True, exist_ok=True)
    (DST / "checkpoints").mkdir(parents=True, exist_ok=True)

    manifest = {
        "run": {"name": "hexfield_bc_1", "output_dir": str(DST), "seed": 1},
        "model": {
            "name": "hexfield",
            "module": "hexfield.plugin",
            "config": {
                "device": "cuda",
                "architecture": {
                    "input_channels": 15,
                    "channels": 96,
                    "trunk": "C C C A C C A C A",
                    "attention_heads": 4,
                    "short_term_value_horizons": [2, 6, 16],
                    "moves_left_head": True,
                },
                "training": {
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "phase": "BC prefit (HF human corpus, 431k rows)",
                },
            },
        },
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    events = [
        {"event": "stage_started", "payload": {"stage": "initialize_run"}},
        {
            "event": "stage_finished",
            "payload": {"stage": "initialize_run", "status": "completed", "elapsed_seconds": 0.0,
                        "metadata": {"result": {"manifest": str(DST / "manifest.json")}}},
        },
    ]
    for row in rows:
        epoch = int(row["epoch"])
        stage = f"epoch_{epoch:06d}"
        events.append({"event": "stage_started", "payload": {"stage": stage}})
        events.append(
            {
                "event": "stage_finished",
                "payload": {
                    "stage": stage,
                    "status": "completed",
                    "elapsed_seconds": float(row.get("seconds", 0.0)),
                    "metadata": {
                        "result": {
                            "epoch": epoch,
                            "training": {
                                "status": "completed",
                                "epoch": epoch,
                                "steps": row.get("steps"),
                                "loss": row.get("train_total"),
                                "policy_loss": row.get("train_policy"),
                                "value_loss": row.get("train_value"),
                                "val_top1": row.get("top1"),
                                "val_value_ece": row.get("value_ece"),
                                "val_value_optimism": row.get("value_optimism"),
                                "probe_policy_kl_prev": row.get("probe_policy_kl_prev"),
                                "probe_d6_kl": row.get("probe_d6_kl"),
                            },
                        }
                    },
                },
            }
        )
        diag = DST / "diagnostics" / f"hexfield.prefit.epoch_{epoch:06d}.json"
        diag.write_text(json.dumps(row, indent=2), encoding="utf-8")
        src_ckpt = SRC / f"checkpoint_epoch{epoch}.pt"
        dst_ckpt = DST / "checkpoints" / f"epoch_{epoch:06d}.pt"
        if src_ckpt.is_file() and not dst_ckpt.is_file():
            shutil.copyfile(src_ckpt, dst_ckpt)
    (DST / "diagnostics" / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    print(f"exported {len(rows)} epochs to {DST}")


if __name__ == "__main__":
    main()
