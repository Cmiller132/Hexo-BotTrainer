RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 510   # ~8.5 min more of self-play to let the ply mix mature
echo "=== still healthy? ==="
ps -eo pid,etime,cmd | grep "[_]rl_train_hexgnn" | grep -oE "^ *[0-9]+ +[0-9:]+" | head -1 | sed 's/^/driver pid+etime: /'
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
echo "=== epoch-31 (n=5) settled pos/s: cumulative + recent window + ply ==="
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000031_game_*.npz"), key=os.path.getmtime)
def rate(files,label):
    if len(files)<2: print(label,"too few"); return
    t0=os.path.getmtime(files[0]); t1=os.path.getmtime(files[-1]); rows=0; lens=[]
    for f in files:
        try:
            with np.load(f) as z:
                vk=[k for k in z.files if 'value' in k.lower()][0]; n=len(z[vk]); rows+=n; lens.append(n)
        except: pass
    w=t1-t0
    print(f"{label}: shards={len(files)} rows={rows} wall={w:.0f}s -> {rows/w:.1f} pos/s | len med/max={int(np.median(lens))}/{max(lens)}")
rate(sh, "CUMULATIVE")
rate(sh[len(sh)//2:], "RECENT half")
rate(sh[-60:], "LAST 60 shards")
print("total epoch-31 shards so far:", len(sh))
PY
echo "=== GPU/VRAM ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
