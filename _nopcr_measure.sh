RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 500
echo "=== health ==="
ps -eo cmd | grep -q "[_]rl_train_hexgnn" && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
echo "=== epoch-50 summary if landed (should have NO '| PCR' suffix = PCR off, 100% recorded) ==="
grep -E "epoch 50 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "=== early: epoch-50 shards, pos/s, avg rows/shard (vs PCR ~half), hard-z ==="
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000050_game_*.npz"), key=os.path.getmtime)
if len(sh)<2: print("epoch-50 shards:",len(sh)); raise SystemExit
t0=os.path.getmtime(sh[0]); t1=os.path.getmtime(sh[-1]); rows=0; per=[]
for f in sh:
    with np.load(f) as z:
        vk=[k for k in z.files if 'value' in k.lower()][0]; n=len(z[vk]); rows+=n; per.append(n)
w=t1-t0
print(f"epoch-50 shards={len(sh)}/512 rows={rows} wall={w:.0f}s -> EARLY ~{rows/w:.1f} pos/s")
print(f"avg rows/shard={np.mean(per):.1f} (no-PCR: ~full game length; PCR epochs were ~half)")
f=sh[-1]
with np.load(f) as z:
    vk=[k for k in z.files if 'value' in k.lower()][0]; v=z[vk]
    print(f"newest shard rows={len(v)} all|val|>=0.99: {bool(np.all(np.abs(v)>=0.99))} min/max {v.min():.2f}/{v.max():.2f}")
PY
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
