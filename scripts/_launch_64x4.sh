#!/usr/bin/env bash
# Launch the 64x4 supervisor detached so it survives this shell / session.
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
mkdir -p "$RD/diagnostics"
# Refuse to double-launch if a supervisor is already up for this run.
if pgrep -f 'supervise_target_64x4_wsl.sh' >/dev/null 2>&1; then
  echo "ALREADY RUNNING: $(pgrep -af supervise_target_64x4_wsl.sh)"; exit 0
fi
nohup bash /mnt/e/Hexo-BotTrainer/scripts/supervise_target_64x4_wsl.sh \
  > "$RD/diagnostics/supervisor_nohup.out" 2>&1 &
disown
echo "launched supervisor pid $!"
