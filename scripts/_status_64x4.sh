#!/usr/bin/env bash
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
DI="$RD/diagnostics"
echo "NOW: $(date -u +%FT%TZ)"
echo "=== procs ==="; ps -eo pid,etime,cmd | grep -E 'supervise_target_64x4|train_model .*64x4' | grep -v grep || echo NONE
echo "=== supervisor log (tail) ==="; tail -6 "$DI/supervisor_wsl.log" 2>/dev/null || echo "(none)"
echo "=== newest trainer out log (tail) ==="; o=$(ls -t "$DI"/trainer.*.out.log 2>/dev/null | head -1); [ -n "$o" ] && { echo "($(basename "$o"))"; tail -12 "$o"; } || echo "(no trainer out log yet)"
echo "=== trainer err (tail, if any) ==="; e=$(ls -t "$DI"/trainer.*.err.log 2>/dev/null | head -1); [ -n "$e" ] && tail -6 "$e" || echo "(none)"
echo "=== events (last 3) ==="; tail -3 "$DI/events.jsonl" 2>/dev/null || echo "(none yet)"
echo "=== TRT adopt? ==="; [ -n "$o" ] && grep -h 'adopted TRT FP16' "$o" 2>/dev/null | tail -1 || echo "(none yet)"
echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
