#!/usr/bin/env bash
# Clean manual stop of the 96x6 run (user-approved: hand GPU to 64x4). Kill the
# SUPERVISOR first so it cannot relaunch when the trainer exits, then the trainer.
# Checkpoints are left intact under runs/.../checkpoints so 96x6 can be resumed.
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
DI="$RD/diagnostics"
echo "=== 96x6 final state before stop ==="
ls -t "$RD"/checkpoints/epoch_*.pt 2>/dev/null | head -1 | xargs -r stat -c 'latest_ckpt %n (mtime %y)'
tail -1 "$DI/events.jsonl" 2>/dev/null

echo "=== killing supervisor (+watchdog), then trainer ==="
sup=$(pgrep -f 'supervise_target_96x6_wsl.sh' | tr '\n' ' ')
echo "supervisor pids: ${sup:-none}"
for p in $sup; do kill "$p" 2>/dev/null && echo "TERM supervisor $p"; done
sleep 1
tp=$(pgrep -f 'train_model .*dense_cnn_model1_target_96x6' | tr '\n' ' ')
echo "trainer pids: ${tp:-none}"
for p in $tp; do kill "$p" 2>/dev/null && echo "TERM trainer $p"; done

# Wait up to 30s for graceful exit, then SIGKILL stragglers.
for i in $(seq 1 15); do
  sleep 2
  alive=$(pgrep -f 'supervise_target_96x6_wsl.sh|train_model .*dense_cnn_model1_target_96x6' | tr '\n' ' ')
  [ -z "$alive" ] && break
done
alive=$(pgrep -f 'supervise_target_96x6_wsl.sh|train_model .*dense_cnn_model1_target_96x6' | tr '\n' ' ')
if [ -n "$alive" ]; then echo "still alive ($alive) -> SIGKILL"; for p in $alive; do kill -9 "$p" 2>/dev/null; done; sleep 2; fi

echo "=== post-stop ==="
echo "remaining 96x6 procs: $(pgrep -f 'supervise_target_96x6_wsl.sh|train_model .*dense_cnn_model1_target_96x6' | tr '\n' ' ' || echo none)"
rm -f "$DI/supervisor_wsl.lock" 2>/dev/null && echo "removed stale supervisor lock"
echo "=== GPU free check ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable in WSL shell)"
