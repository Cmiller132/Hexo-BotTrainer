RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "=== driver/supervisor alive? ==="
ps -eo pid,pgid,etime,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn" | grep -v grep | grep -oE "pid|[0-9]+:[0-9]+ .*(_rl_train_hexgnn|supervise).*" | head
ps -eo pid,pgid,etime,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn" | grep -v grep | sed -E 's/(--out-dir).*//' | head
echo "=== epoch-30 selfplay PROGRESS (shards being written now?) ==="
echo "epoch-30 shards: $(ls "$RD/selfplay"/epoch_000030_game_*.npz 2>/dev/null | wc -l)/512"
echo "newest epoch-30 shard mtime: $(ls -t "$RD/selfplay"/epoch_000030_game_*.npz 2>/dev/null | head -1 | xargs -n1 stat -c '%y' 2>/dev/null)"
echo "now: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
echo "=== driver err/out log tails (latest driver.*.log) ==="
le=$(ls -t "$RD"/driver.*.err.log 2>/dev/null | head -1); echo "ERR: $le"; tail -6 "$le" 2>/dev/null
lo=$(ls -t "$RD"/driver.*.out.log 2>/dev/null | head -1); echo "OUT: $lo"; tail -4 "$lo" 2>/dev/null
echo "=== supervisor.log tail (relaunches/halts?) ==="
tail -5 "$RD/supervisor.log" 2>/dev/null
