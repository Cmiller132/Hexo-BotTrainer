RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 500
echo "=== health ==="
ps -eo cmd | grep -q "[_]rl_train_hexgnn" && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
echo "=== epoch-46 (PCR) summary if landed (shows pcr_str: recorded/fast/mean-visits) ==="
grep -E "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "=== early pos/s from epoch-46 shards + fresh-shard hard-z ==="
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000046_game_*.npz"), key=os.path.getmtime)
if len(sh)<2: print("epoch-46 shards so far:",len(sh)); raise SystemExit
t0=os.path.getmtime(sh[0]); t1=os.path.getmtime(sh[-1]); rows=0
for f in sh:
    try:
        with np.load(f) as z:
            vk=[k for k in z.files if 'value' in k.lower()][0]; rows+=len(z[vk])
    except: pass
w=t1-t0
print(f"epoch-46 PCR shards={len(sh)}/512 rows={rows} wall={w:.0f}s -> EARLY ~{rows/w:.1f} pos/s (recorded rows only; searched pos/s ~2x)")
f=sh[-1]
with np.load(f) as z:
    vk=[k for k in z.files if 'value' in k.lower()][0]; v=z[vk]
    print(f"newest shard: recorded_rows={len(v)} all|val|>=0.99: {bool(np.all(np.abs(v)>=0.99))} min/max {v.min():.2f}/{v.max():.2f}")
PY
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
