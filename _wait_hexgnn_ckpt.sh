RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
# current latest epoch
cur=$(ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | sed -E 's/.*epoch0*([0-9]+)\.pt/\1/')
echo "current latest epoch checkpoint: $cur (waiting for epoch $((cur+1)) checkpoint = current epoch finishing)"
target=$((cur+1))
for i in $(seq 1 50); do
  if [ -f "$RD/checkpoints/hexgnn_rl_epoch$(printf %06d $target).pt" ]; then echo "READY: epoch $target checkpointed"; break; fi
  sleep 15
done
echo "=== final hexgnn state ==="
grep -E "epoch $target (selfplay|done):|epoch $cur (selfplay|done):" "$RD/rl_train.log" 2>/dev/null | tail -3 | sed -E 's/\[[0-9: -]+\] //'
ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | xargs -n1 basename
tail -1 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
