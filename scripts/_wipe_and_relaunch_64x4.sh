#!/usr/bin/env bash
# Clean restart of the 64x4 run at the new sim count: wipe the stale generated data
# (512-sim shards + calibration + epoch/eval diagnostics) so the from-scratch curve is
# pure, then relaunch the supervisor detached. Refuses to run if a trainer is alive.
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
if pgrep -f 'supervise_target_64x4_wsl.sh|train_model .*dense_cnn_model1_target_64x4' >/dev/null 2>&1; then
  echo "ABORT: a 64x4 process is still alive: $(pgrep -af 'supervise_target_64x4_wsl.sh|train_model .*dense_cnn_model1_target_64x4')"; exit 1
fi
echo "=== pre-wipe contents ==="
echo "  selfplay shards: $(ls "$RD"/selfplay/*.npz 2>/dev/null | wc -l)"
echo "  checkpoints:     $(ls "$RD"/checkpoints/*.pt 2>/dev/null | wc -l)"
# Remove the whole run dir (no checkpoints worth keeping — cold start, no epoch done).
rm -rf "$RD"
mkdir -p "$RD/diagnostics" "$RD/checkpoints"
echo "=== wiped; relaunching ==="
bash /mnt/e/Hexo-BotTrainer/scripts/_launch_verify_64x4.sh
