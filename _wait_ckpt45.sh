RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 30); do
  if [ -f "$RD/checkpoints/hexgnn_rl_epoch000045.pt" ]; then echo "READY: epoch-45 checkpoint written"; break; fi
  sleep 10
done
grep -E "epoch 45 done" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | xargs -n1 basename
tail -1 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
