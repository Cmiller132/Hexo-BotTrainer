RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 24); do
  if [ -f "$RD/checkpoints/hexgnn_rl_epoch000030.pt" ]; then echo "READY: epoch-30 checkpoint written"; break; fi
  sleep 10
done
echo "=== epoch 30 done + latest checkpoints ==="
grep -E "epoch 30 (done|train:)" "$RD/rl_train.log" 2>/dev/null | tail -2 | sed -E 's/\[[0-9: -]+\] //'
ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -2 | xargs -n1 basename 2>/dev/null
echo "=== current phase (epoch 31 starting?) ==="
tail -1 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
