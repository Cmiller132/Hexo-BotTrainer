#!/usr/bin/env bash
# Distinguish a real leak from benign page cache. MemAvailable (reclaimable-aware)
# is the number that matters, not MemFree. Also show trainer RSS + cache size.
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
echo "=== /proc/meminfo (key lines) ==="
grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Buffers|SwapTotal|SwapFree):' /proc/meminfo
echo "=== free -h ==="
free -h
tp=$(pgrep -f 'train_model .*64x4' | head -1)
echo "=== trainer pid=$tp RSS ==="
[ -n "$tp" ] && grep -E '^Vm(RSS|Swap|HWM):' /proc/$tp/status
echo "=== live self-play ==="
python3 -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('epoch=%s status=%s pos/s=%.1f finished=%s/%s active=%s elapsed=%.0fs'%(d.get('epoch'),d.get('status'),d.get('search_positions_per_second',0),d.get('games_finished'),d.get('requested_games'),d.get('active_games'),d.get('elapsed_seconds',0)))" 2>/dev/null
echo "=== finished game lengths (turns, from shard sample counts) ==="
ls "$RD"/selfplay/epoch_000001_game_*.npz 2>/dev/null | wc -l | xargs echo "  npz shards:"
echo "=== watchdog RAM trend (last 8) ==="
tail -8 "$RD/diagnostics/watch_wsl.jsonl" 2>/dev/null
