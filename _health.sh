RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "===== PROCESS STATE ====="
ps -eo pid,etime,rss,cmd | grep -iE "_rl_train_hexgnn|_rl_supervise_hexgnn|dashboard_bridge_hexgnn|hexo_frontend.web" | grep -v grep
echo "===== EPOCH-0 PROGRESS / WALL ====="
ls "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | wc -l
F=$(ls -t "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | tail -1)
L=$(ls -t "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | head -1)
echo "first shard mtime: $(stat -c %y "$F" 2>/dev/null)"
echo "latest shard mtime: $(stat -c %y "$L" 2>/dev/null)"
echo "run start (log): $(head -1 "$RD/rl_train.log" 2>/dev/null)"
echo "===== GAME LENGTH / DECISIVENESS (epoch 0) ====="
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000000_game_*.npz"))
lens=[]; finalvals=[]; cand=[]
for f in sh:
    try:
        with np.load(f) as z:
            vk=[x for x in z.files if 'value' in x.lower()][0]
            v=z[vk]; lens.append(len(v))
            finalvals.append(float(v[-1]))
            # candidate count if present
            ck=[x for x in z.files if 'cand' in x.lower() or 'policy' in x.lower()]
            if ck:
                a=z[ck[0]]
                cand.append(a.shape[1] if a.ndim>1 else len(a))
    except Exception as e: pass
import statistics as st
L=np.array(lens)
print(f"games={len(L)}  len min/median/mean/max = {L.min()}/{int(np.median(L))}/{L.mean():.1f}/{L.max()}")
print(f"  len pctiles 10/50/90/99 = {np.percentile(L,10):.0f}/{np.percentile(L,50):.0f}/{np.percentile(L,90):.0f}/{np.percentile(L,99):.0f}")
print(f"  games at/over 500 (max_actions cap): {(L>=500).sum()} ({100*(L>=500).mean():.1f}%)")
fv=np.array(finalvals)
print(f"final-move value sign: +1={ (fv>0).sum() }  -1={ (fv<0).sum() }  0={ (np.abs(fv)<1e-6).sum() }  (decisiveness)")
print(f"all |final|==1 hard-z: {bool(np.all(np.abs(fv)>0.99))}")
if cand: print(f"candidates/row mean={np.mean(cand):.1f} min/max={min(cand)}/{max(cand)}")
PY
echo "===== LR / WARMUP behaviour (log) ====="
grep -iE "warmup|lr=|learning rate| lr | step " "$RD/rl_train.log" 2>/dev/null | tail -6
echo "===== VRAM / files ====="
du -sh "$RD/selfplay" 2>/dev/null
ls "$RD/checkpoints"/*.pt 2>/dev/null | wc -l
