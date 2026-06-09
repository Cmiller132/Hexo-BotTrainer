RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
# how many epoch-30 shards are NEW (mtime after the resume at ~00:37)?
new=$(find "$RD/selfplay" -name 'epoch_000030_game_*.npz' -newermt '2026-06-06 00:38:00' 2>/dev/null | wc -l)
echo "epoch-30 shards written since resume (new 1024-visit games done): $new / 512"
echo "=== recompilation thrashing? (count in err log since resume) ==="
le=$(ls -t "$RD"/driver.20260606_0436*.err.log 2>/dev/null | head -1)
grep -icE "recompil|Recompiling|guard|cache_size_limit|torch._dynamo" "$le" 2>/dev/null | sed 's/^/dynamo-recompile-ish lines: /'
grep -iE "recompil|cache_size_limit|exceeded" "$le" 2>/dev/null | tail -3
# poll a few min for the epoch-30 selfplay SUMMARY (the definitive 1024 pos/s)
echo "=== waiting up to 4 min for epoch-30 (1024) selfplay summary line ==="
for i in $(seq 1 20); do
  n=$(grep -cE "epoch 30 selfplay:" "$RD/rl_train.log" 2>/dev/null)
  if [ "${n:-0}" -ge 2 ]; then break; fi
  sleep 12
done
grep -E "epoch 30 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "now: $(date '+%H:%M:%S'); newest shard: $(ls -t "$RD/selfplay"/epoch_000030_game_*.npz 2>/dev/null | head -1 | xargs -n1 stat -c '%y' 2>/dev/null | cut -c12-19)"
