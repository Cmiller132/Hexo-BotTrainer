RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
# only NEW shards (written after the no-PCR bounce ~18:11 UTC / 14:11 local)
echo "now: $(date '+%H:%M:%S')"
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
bounce=os.path.getmtime(RD+"/checkpoints/hexgnn_rl_epoch000049.pt")  # ~bounce time ref
allsh=glob.glob(RD+"/selfplay/epoch_000050_game_*.npz")
new=sorted([f for f in allsh if os.path.getmtime(f) > bounce+60], key=os.path.getmtime)
print(f"epoch-50 shards: total={len(allsh)} new(no-PCR)={len(new)}")
if len(new)<5: print("too few new shards yet"); raise SystemExit
t0=os.path.getmtime(new[0]); t1=os.path.getmtime(new[-1]); rows=0; per=[]
for f in new:
    with np.load(f) as z:
        vk=[k for k in z.files if 'value' in k.lower()][0]; n=len(z[vk]); rows+=n; per.append(n)
w=t1-t0
print(f"NEW no-PCR: shards={len(new)} rows={rows} wall={w:.0f}s -> ~{rows/w:.1f} pos/s | avg rows/shard={np.mean(per):.1f}")
f=new[-1]
with np.load(f) as z:
    vk=[k for k in z.files if 'value' in k.lower()][0]; v=z[vk]
    print(f"newest new shard rows={len(v)} hard±1: {bool(np.all(np.abs(v)>=0.99))} min/max {v.min():.2f}/{v.max():.2f}")
PY
echo "=== epoch-50 summary if landed (no '| PCR' = confirmed off) ==="
grep -E "epoch 50 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
ps -eo cmd | grep -q "[_]rl_train_hexgnn" && echo "driver ALIVE" || echo "driver GONE"
