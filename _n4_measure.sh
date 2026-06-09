RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 500
echo "=== health: driver alive + relaunches/halts ==="
ps -eo pid,etime,cmd | grep "[_]rl_train_hexgnn" | grep -v grep >/dev/null && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
echo "=== epoch-42 (n=4) pos/s: cumulative + definitive summary if present ==="
grep -E "epoch 42 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted([f for f in glob.glob(RD+"/selfplay/epoch_000042_game_*.npz")], key=os.path.getmtime)
if len(sh)<2: print("epoch-42 shards so far:",len(sh)); raise SystemExit
t0=os.path.getmtime(sh[0]); t1=os.path.getmtime(sh[-1]); rows=0; cand_rows=[]
for f in sh:
    try:
        with np.load(f) as z:
            vk=[k for k in z.files if 'value' in k.lower()][0]; rows+=len(z[vk])
    except: pass
w=t1-t0
print(f"epoch-42 n=4 shards={len(sh)}/512 rows={rows} wall={w:.0f}s -> CUMULATIVE ~{rows/w:.1f} pos/s (incl ~2min warmup)")
# fresh shard hard-z + candidate count
f=sh[-1]
with np.load(f) as z:
    vk=[k for k in z.files if 'value' in k.lower()][0]; v=z[vk]
    print(f"newest shard: rows={len(v)} all|val|>=0.99: {bool(np.all(np.abs(v)>=0.99))} min/max {v.min():.2f}/{v.max():.2f}")
    for k in z.files:
        if 'candidate' in k.lower() and ('count' in k.lower() or 'graph' in k.lower()):
            a=z[k]; print(f"  {k}: shape={a.shape}")
PY
echo "=== GPU/VRAM ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
