RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
# wait until >=25 NEW epoch-31 shards exist (mtime after resume 01:39) or ~8 min
for i in $(seq 1 40); do
  n=$(find "$RD/selfplay" -name 'epoch_000031_game_*.npz' -newermt '2026-06-06 01:40:00' 2>/dev/null | wc -l)
  if [ "${n:-0}" -ge 25 ]; then break; fi
  # crash check: is the driver still alive?
  if ! ps -eo cmd | grep -q "[_]rl_train_hexgnn"; then echo "WARNING: driver not running!"; fi
  sleep 12
done
echo "=== no-crash check: driver alive + supervisor relaunches ==="
ps -eo pid,etime,cmd | grep "[_]rl_train_hexgnn" | grep -oE "[0-9]+:[0-9]+" | head -1 | sed 's/^/driver etime: /'
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/supervisor RELAUNCH\/HALT\/EXC count: /'
echo "=== epoch-31 (n=5) shard production + early pos/s ==="
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=[f for f in glob.glob(RD+"/selfplay/epoch_000031_game_*.npz") if os.path.getmtime(f)>1749000000]
sh=sorted([f for f in glob.glob(RD+"/selfplay/epoch_000031_game_*.npz")], key=os.path.getmtime)
# keep only those written after ~01:40 (new n=5)
import time
new=[f for f in sh if os.path.getmtime(f) > os.path.getmtime(sh[0])-1] if sh else []
if len(sh)<2: print("too few shards yet:",len(sh)); raise SystemExit
t0=os.path.getmtime(sh[0]); t1=os.path.getmtime(sh[-1]); rows=0; cands=[]
for f in sh:
    try:
        with np.load(f) as z:
            vk=[k for k in z.files if 'value' in k.lower()][0]; rows+=len(z[vk])
            ck=[k for k in z.files if 'cand' in k.lower() and 'count' in k.lower()] or [k for k in z.files if k.lower().endswith('candidate_counts')]
    except: pass
w=t1-t0
print(f"epoch-31 n=5 shards={len(sh)} rows={rows} wall={w:.0f}s -> EARLY ~{rows/w:.1f} pos/s" if w>0 else "wall=0")
# hard-z + candidate count on newest shard
f=sh[-1]
with np.load(f) as z:
    vk=[k for k in z.files if 'value' in k.lower()][0]; v=z[vk]
    print(f"newest shard {os.path.basename(f)}: rows={len(v)} all|val|>=0.99: {bool(np.all(np.abs(v)>=0.99))} min/max {v.min():.2f}/{v.max():.2f}")
    # candidate count per row
    for key in z.files:
        if 'cand' in key.lower():
            a=z[key]; print(f"  {key}: shape={a.shape} {'mean='+str(round(float(a.mean()),1)) if a.ndim==1 else ''}")
PY
echo "=== GPU/VRAM ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
