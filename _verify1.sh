RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "=== driver startup (params/config/from-scratch) ==="
sed -n '1,12p' "$RD/rl_train.log" 2>/dev/null
echo "=== procs ==="
ps -eo pid,etime,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn|_dashboard_bridge_hexgnn" | grep -v grep | head
echo "=== driver err tail (any crash?) ==="
ls -t "$RD"/driver.*.err.log 2>/dev/null | head -1 | xargs tail -3 2>/dev/null
