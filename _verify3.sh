RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
# wait up to ~4min for first shards
for i in $(seq 1 40); do
  n=$(ls "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 5 ]; then break; fi
  sleep 6
done
echo "=== shards so far: $(ls "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | wc -l) ==="
ls -t "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | head -2
echo "=== inspect a shard (hard +-1 value targets, candidate sets) ==="
/root/.venvs/hexgt-build/bin/python - "$RD" <<'PY'
import sys,glob,numpy as np
shards=sorted(glob.glob(sys.argv[1]+"/selfplay/epoch_000000_game_*.npz"))
if not shards: print("no shards yet"); sys.exit()
import os
sys.path.insert(0,"/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python")
from hexo_models.dense_cnn.compact_io import read_compact_shard
from pathlib import Path
rows=read_compact_shard(Path(shards[0]))
print("shard:",os.path.basename(shards[0]),"rows:",len(rows))
vals=[float(r.value) for r in rows[:50]]
print("value targets (first 12):",[round(v,2) for v in vals[:12]])
print("all |value|==1 (hard-z)?:", all(abs(v)==1.0 for v in vals), " min/max:",min(vals),max(vals))
print("policy candidates (first row):", len(list(rows[0].policy)), " stv targets:", len(list(getattr(rows[0],'short_term_value',()))))
PY
