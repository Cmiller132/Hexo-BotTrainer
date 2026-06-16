#!/usr/bin/env bash
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "=== diagnostics files mentioning train/loss ==="
ls -1t "$RUNDIR"/diagnostics/ 2>/dev/null | grep -iE 'train|epoch' | head -10
echo "=== any 'loss' in diagnostics (latest) ==="
grep -rilE 'loss' "$RUNDIR"/diagnostics/ 2>/dev/null | head -5 | while read f; do
  echo "--- $f ---"; tr -d '\n' < "$f" | grep -oE '"[a-z_]*loss[a-z_]*":[ ]*[0-9.eE+-]+' | head -8
done
echo "=== events.jsonl: train/shuffle/loss lines ==="
grep -iE 'train|shuffle|loss|skip' "$RUNDIR/events.jsonl" 2>/dev/null | tail -8 | cut -c1-220
echo "=== epoch_000002.pt metadata (loss?) ==="
/root/.venvs/hexgt-build/bin/python - "$RUNDIR/checkpoints/epoch_000002.pt" <<'PY'
import torch,sys
try:
    d=torch.load(sys.argv[1], map_location="cpu")
    md=d.get("metadata",{})
    print("epoch:", d.get("epoch"), "| metadata keys:", list(md.keys()))
    print("metadata:", {k:v for k,v in md.items() if not isinstance(v,(dict,list)) or k in ("sample_count",)})
except Exception as e:
    print("err:", e)
PY
