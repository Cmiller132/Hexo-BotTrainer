#!/usr/bin/env bash
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "=== epoch_000002.json (full) ==="
/root/.venvs/hexgt-build/bin/python -c "import json; print(json.dumps(json.load(open('$RUNDIR/diagnostics/epoch_000002.json')), indent=1))" 2>/dev/null | head -80
echo "=== train phase in epoch_000002.json (train/loss/skip/steps) ==="
grep -oE '"(status|steps|loss|losses|skipped|train[a-z_]*|warmup|rows|window)"[ ]*:[ ]*("?[A-Za-z0-9_.-]+"?)' "$RUNDIR/diagnostics/epoch_000002.json" 2>/dev/null
echo "=== train log: train/loss/skip/warmup/shuffle lines (latest log) ==="
LOG=$(ls -1t "$RUNDIR"/train.*.out.log | head -1)
grep -inE 'loss|train pass|trained|skip|warmup|shuffle|insufficient|min_rows' "$LOG" 2>/dev/null | tail -20
