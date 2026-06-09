RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
for i in $(seq 1 60); do
  gf=$($V -c "import json;print(json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json')).get('games_finished',0))" 2>/dev/null)
  hxr=$(stat -c %s "$RD/selfplay/epoch_000001.hxr" 2>/dev/null)
  if [ "${gf:-0}" -ge 1 ]; then echo "FIRST GAMES DONE: games_finished=$gf (.hxr size=$hxr B at $(date '+%H:%M:%S'))"; break; fi
  sleep 15
done
echo "=== final: live status + .hxr ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('games_finished','completed_games','active_games','searched_positions','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo ".hxr size: $(stat -c %s "$RD/selfplay/epoch_000001.hxr" 2>/dev/null) B"
