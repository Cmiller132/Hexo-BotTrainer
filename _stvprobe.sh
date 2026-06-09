#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_models/python"
echo "=== driver ==="
p=$(pgrep -f _rl_train.py | head -1); echo "driver=$p start=$(ps -o lstart= -p "$p" 2>/dev/null)"
awk '/MemAvailable/{printf "RAM free %.1f GB\n", $2/1048576}' /proc/meminfo
echo "=== STV targets in a NEW (post-graft) epoch-12 shard ==="
f=$(ls "$WT"/runs/hexgt_rl_main2/selfplay/epoch_000012_game_*.npz 2>/dev/null | head -1)
echo "shard: $f"
python3 -c "
from hexo_models.dense_cnn.compact_io import read_compact_shard
r=read_compact_shard('$f')
stv=[x.short_term_value for x in r if x.short_term_value]
print('rows:',len(r),'| rows WITH stv targets:',len(stv))
if stv:
    print('horizons present:', sorted(h for h,_ in stv[0]))
    print('sample stv:', [(h,round(v,3)) for h,v in stv[0]])
"
