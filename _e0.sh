RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
# wait up to ~4min for epoch-0 self-play pos/s line (or report live progress)
for i in $(seq 1 40); do
  if grep -q "epoch 0 selfplay" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 6
done
echo "=== epoch-0 self-play result ==="
grep "epoch 0 selfplay" "$RD/rl_train.log" 2>/dev/null | tail -1 || {
  echo "(epoch 0 still running) shards so far: $(ls "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | wc -l)/512"
  # estimate pos/s from shard rows over wall since self-play start
}
echo "=== train line (if epoch 0 trained) ==="; grep "epoch 0 train" "$RD/rl_train.log" 2>/dev/null | tail -1
