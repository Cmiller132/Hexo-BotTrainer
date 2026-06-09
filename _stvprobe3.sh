#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_models/python"
SP="$WT/runs/hexgt_rl_main2/selfplay"
echo "=== newest 3 ep12 shards (mtime) ==="
ls -lat --time-style=+%H:%M "$SP"/epoch_000012_game_*.npz 2>/dev/null | head -3
echo "=== STV in the NEWEST ep12 shard (new driver if mtime>18:21) ==="
f=$(ls -t "$SP"/epoch_000012_game_*.npz 2>/dev/null | head -1)
echo "newest: $f  mtime=$(date -r "$f" +%H:%M:%S 2>/dev/null)"
python3 -c "
from hexo_models.dense_cnn.compact_io import read_compact_shard
r=read_compact_shard('$f')
stv=[x.short_term_value for x in r if x.short_term_value]
print('rows=%d rows_with_stv=%d'%(len(r),len(stv)))
if stv: print('horizons=',sorted(h for h,_ in stv[0]),'sample=',[(h,round(v,3)) for h,v in stv[0]])
"
