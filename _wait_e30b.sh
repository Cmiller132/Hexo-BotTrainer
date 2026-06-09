RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
# wait until a SECOND "epoch 30 selfplay" line appears (the new 1024-visit one)
for i in $(seq 1 320); do
  n=$(grep -cE "epoch 30 selfplay:" "$RD/rl_train.log" 2>/dev/null)
  if [ "${n:-0}" -ge 2 ]; then break; fi
  sleep 12
done
echo "=== epoch 30 selfplay (NEW = 1024 visits) ==="
grep -E "epoch 30 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1
echo "--- v/move + GPU mem (new epoch 30) ---"
grep -E "~[0-9]+v/move|GPU mem \[epoch 30 selfplay\]" "$RD/rl_train.log" 2>/dev/null | tail -2
echo "--- fresh epoch-30 shard hard-z check ---"
F=$(ls -t "$RD/selfplay"/epoch_000030_game_*.npz 2>/dev/null | head -1)
PYTHONPATH="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python" $V - "$F" <<'PY'
import sys,numpy as np
with np.load(sys.argv[1]) as z:
    vk=[k for k in z.files if 'value' in k.lower()][0]; v=z[vk]
    print(f"shard {sys.argv[1].split('/')[-1]} rows={len(v)} all|val|==1: {bool(np.all(np.abs(v)>0.99))} min/max {v.min():.2f}/{v.max():.2f}")
PY
