RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
# Poll up to ~9 min for epoch-30 FULL completion (checkpoint epoch 30 written = clean boundary).
for i in $(seq 1 36); do
  if [ -f "$RD/checkpoints/hexgnn_rl_epoch000030.pt" ]; then echo "EPOCH30_CHECKPOINT_READY"; break; fi
  sleep 15
done
echo "=== epoch-30 (1024-visit) self-play number [ISOLATED 1024] ==="
grep -E "epoch 30 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "=== epoch 30 done + checkpoints ==="
grep -E "epoch 30 done" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -2 | xargs -n1 basename 2>/dev/null
echo "=== current tail (what phase now) ==="
tail -1 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
