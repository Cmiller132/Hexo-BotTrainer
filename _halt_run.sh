#!/usr/bin/env bash
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== state before stop ==="
tail -1 "$RUN/rl_train.log"
echo "latest ckpt:"; ls -la --time-style=+%H:%M "$RUN/checkpoints/hexgt_rl_latest.pt" 2>/dev/null
echo "=== stop driver + supervisor + bridge + telemetry (TERM then KILL by pattern) ==="
for pat in _rl_run_fg.sh _rl_supervise.sh _rl_train.py _dashboard_bridge.py "query-gpu=timestamp,temperature"; do
  for p in $(pgrep -f "$pat"); do echo "TERM [$pat] $p"; kill "$p" 2>/dev/null; done
done
sleep 5
for pat in _rl_run_fg.sh _rl_supervise.sh _rl_train.py _dashboard_bridge.py "query-gpu=timestamp,temperature"; do
  for p in $(pgrep -f "$pat"); do echo "KILL [$pat] $p"; kill -9 "$p" 2>/dev/null; done
done
sleep 2
echo "=== write HALT flag so nothing auto-relaunches the run ==="
echo "halted for deep defense analysis $(date -u +%FT%TZ)" > "$RUN/supervisor_halted.flag"
echo "HALT flag: $RUN/supervisor_halted.flag"
echo "=== survivors (should be none for training) ==="
pgrep -af "_rl_train.py|_rl_supervise|_rl_run_fg|query-gpu=timestamp" | grep -v pgrep || echo "training STOPPED"
echo "=== GPU (should be free) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "=== checkpoints preserved ==="
ls -1 "$RUN/checkpoints"/hexgt_rl_epoch*.pt 2>/dev/null | tail -3; echo "latest:"; ls -la --time-style=+%H:%M "$RUN/checkpoints/hexgt_rl_latest.pt"
