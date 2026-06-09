V=/root/.venvs/hexgt-build/bin/python
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob, os, numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000000_game_*.npz"), key=os.path.getmtime)
def measure(sub,label):
    if len(sub)<2: print(label,"too few"); return
    t0=os.path.getmtime(sub[0]); t1=os.path.getmtime(sub[-1]); rows=0
    for f in sub:
        try:
            with np.load(f) as z:
                k=[x for x in z.files if 'value' in x.lower()] or z.files
                rows+=len(z[k[0]])
        except: pass
    w=t1-t0
    print(f"{label}: shards={len(sub)} rows={rows} wall={w:.1f}s -> {rows/w:.1f} pos/s, {len(sub)/w:.2f} games/s")
print("total shards:",len(sh))
measure(sh, "ALL (incl compile warmup)")
measure(sh[len(sh)//2:], "2nd HALF (steady)")
measure(sh[-60:], "LAST 60 (steady-state)")
measure(sh[-30:], "LAST 30 (steady-state)")
PY
