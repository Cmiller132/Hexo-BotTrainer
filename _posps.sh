RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob, os, numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000000_game_*.npz"), key=os.path.getmtime)
if not sh:
    print("no shards"); raise SystemExit
t0=os.path.getmtime(sh[0]); t1=os.path.getmtime(sh[-1])
rows=0
for f in sh:
    try:
        with np.load(f) as z:
            k=[x for x in z.files if 'value' in x.lower()] or z.files
            rows+=len(z[k[0]])
    except Exception: pass
wall=t1-t0
print(f"epoch-0 shards={len(sh)}  rows(searched positions)={rows}")
print(f"wall(first->last shard)={wall:.1f}s")
print(f"measured pos/s (active=256, visits=512) = {rows/wall:.1f}" if wall>0 else "wall=0")
print(f"games/sec={len(sh)/wall:.2f}  avg rows/game={rows/len(sh):.1f}" if wall>0 else "")
PY
