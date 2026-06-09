RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "now: $(date '+%H:%M:%S')"
echo "epoch-46 shards: $(ls "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | wc -l)/512"
echo "newest mtime: $(ls -t "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | head -1 | xargs -n1 stat -c '%y' 2>/dev/null | cut -c12-19)"
echo "rl_train.log tail:"; tail -2 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
echo "=== recorded pos/s over full epoch-46 window so far ==="
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
sh=sorted(glob.glob(RD+"/selfplay/epoch_000046_game_*.npz"), key=os.path.getmtime)
if len(sh)<2: print("shards:",len(sh)); raise SystemExit
t0=os.path.getmtime(sh[0]); t1=os.path.getmtime(sh[-1]); rows=sum(len(np.load(f)[[k for k in np.load(f).files if 'value' in k.lower()][0]]) for f in sh[:0]) if False else 0
for f in sh:
    with np.load(f) as z:
        vk=[k for k in z.files if 'value' in k.lower()][0]; rows+=len(z[vk])
w=t1-t0
print(f"shards={len(sh)}/512 recorded_rows={rows} wall={w:.0f}s -> recorded ~{rows/w:.1f} pos/s (searched ~{2*rows/w:.1f})")
PY
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
