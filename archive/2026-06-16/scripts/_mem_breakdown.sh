#!/usr/bin/env bash
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
tp=$(pgrep -f 'train_model .*64x4' | head -1)
echo "trainer pid=${tp:-NONE}"
if [ -z "$tp" ]; then echo "TRAINER NOT ALIVE — likely OOM-killed"; dmesg 2>/dev/null | grep -iE 'killed process|out of memory|oom' | tail -5; exit 0; fi
echo "=== smaps_rollup ==="
grep -E '^(Rss|Pss|Anonymous|Shared_|Private_|Swap):' /proc/$tp/smaps_rollup 2>/dev/null
echo "=== free now / 30s / 60s ==="
for s in 0 30 30; do
  [ "$s" -gt 0 ] && sleep "$s"
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  rss=$(awk '/VmRSS/{print $2}' /proc/$tp/status 2>/dev/null)
  sw=$(awk '/VmSwap/{print $2}' /proc/$tp/status 2>/dev/null)
  echo "  MemAvailable=$((a/1024))MB  trainerRSS=$((rss/1024))MB  trainerSwap=$((sw/1024))MB"
done
echo "=== live ==="
python3 -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('finished=%s/%s active=%s elapsed=%.0fs pos/s=%.1f'%(d.get('games_finished'),d.get('requested_games'),d.get('active_games'),d.get('elapsed_seconds',0),d.get('search_positions_per_second',0)))"
echo "=== dmesg oom (if any) ==="
dmesg 2>/dev/null | grep -iE 'killed process|out of memory' | tail -3 || echo "(dmesg unavailable / none)"