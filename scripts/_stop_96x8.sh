#!/usr/bin/env bash
# Clean stop of the 96x8 run. Kill the SUPERVISOR first so it cannot relaunch when
# the trainer exits, then the trainer. Mirrors _stop_64x4.sh.
set -uo pipefail
echo "=== killing supervisor (+watchdog), then trainer ==="
sup=$(pgrep -f 'supervise_target_96x8_wsl.sh' | tr '\n' ' ')
echo "supervisor pids: ${sup:-none}"
for p in $sup; do kill "$p" 2>/dev/null && echo "TERM supervisor $p"; done
sleep 1
tp=$(pgrep -f 'train_model .*dense_cnn_model1_target_96x8' | tr '\n' ' ')
echo "trainer pids: ${tp:-none}"
for p in $tp; do kill "$p" 2>/dev/null && echo "TERM trainer $p"; done
for i in $(seq 1 15); do
  sleep 2
  alive=$(pgrep -f 'supervise_target_96x8_wsl.sh|train_model .*dense_cnn_model1_target_96x8' | tr '\n' ' ')
  [ -z "$alive" ] && break
done
alive=$(pgrep -f 'supervise_target_96x8_wsl.sh|train_model .*dense_cnn_model1_target_96x8' | tr '\n' ' ')
if [ -n "$alive" ]; then echo "still alive ($alive) -> SIGKILL"; for p in $alive; do kill -9 "$p" 2>/dev/null; done; sleep 2; fi
echo "remaining 96x8 procs: $(pgrep -f 'supervise_target_96x8_wsl.sh|train_model .*dense_cnn_model1_target_96x8' | tr '\n' ' ' || echo none)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "(no compute apps)"
