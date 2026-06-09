RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 40); do
  if [ -f "$RD/checkpoints/hexgnn_rl_epoch000050.pt" ]; then echo "READY: epoch 50 checkpointed"; break; fi
  sleep 12
done
echo "=== final hexgnn epoch-50 self-play (no-PCR) + checkpoint ==="
grep -E "epoch 50 (selfplay|done):" "$RD/rl_train.log" 2>/dev/null | tail -2 | sed -E 's/\[[0-9: -]+\] //'
ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | xargs -n1 basename
