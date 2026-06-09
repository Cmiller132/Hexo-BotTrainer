RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
CUT=$(date -d "2026-06-06 18:15:30 UTC" +%s)   # after the no-PCR relaunch compile
echo "now: $(date '+%H:%M:%S %Z')  cutoff(UTC 18:15:30)=$CUT"
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" CUT=$CUT $V - <<'PY'
import glob,os,numpy as np
RD="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"; cut=float(os.environ["CUT"])
new=sorted([f for f in glob.glob(RD+"/selfplay/epoch_000050_game_*.npz") if os.path.getmtime(f)>cut], key=os.path.getmtime)
print(f"epoch-50 shards strictly after cutoff (clean no-PCR): {len(new)}")
if len(new)<5: print("too few yet — will rely on the epoch-50 summary"); raise SystemExit
t0=os.path.getmtime(new[0]); t1=os.path.getmtime(new[-1]); rows=0; per=[]
for f in new:
    with np.load(f) as z:
        vk=[k for k in z.files if 'value' in k.lower()][0]; n=len(z[vk]); rows+=n; per.append(n)
w=t1-t0
print(f"NEW no-PCR: shards={len(new)} rows={rows} wall={w:.0f}s -> ~{rows/w:.1f} pos/s | avg rows/shard={np.mean(per):.1f} (=full game len; PCR was ~half)")
PY
echo "=== epoch-50 summary if landed (no '| PCR' suffix = confirmed off) ==="
grep -E "epoch 50 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
