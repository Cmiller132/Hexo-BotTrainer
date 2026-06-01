#!/usr/bin/env bash
# Launch the supervisor detached (new session via setsid) and verify it is still
# alive 12s later WITHIN THIS SAME wsl session, so we can tell a startup crash apart
# from a WSL-teardown-after-exit. Keeps WSL anchored for the duration of this call.
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8
DI="$RD/diagnostics"
mkdir -p "$DI"
if pgrep -f 'supervise_target_96x8_wsl.sh' >/dev/null 2>&1; then
  echo "ALREADY RUNNING: $(pgrep -af supervise_target_96x8_wsl.sh)"; exit 0
fi
setsid nohup bash /mnt/e/Hexo-BotTrainer/scripts/supervise_target_96x8_wsl.sh \
  > "$DI/supervisor_nohup.out" 2>&1 < /dev/null &
echo "launched pid $!"
sleep 12
echo "=== +12s alive? ==="
pgrep -af 'supervise_target_96x8_wsl.sh|train_model .*96x8' || echo "  NOT ALIVE"
echo "=== supervisor log ==="; tail -8 "$DI/supervisor_wsl.log" 2>/dev/null || echo "  (no log)"
echo "=== nohup.out ==="; tail -8 "$DI/supervisor_nohup.out" 2>/dev/null || echo "  (empty)"
