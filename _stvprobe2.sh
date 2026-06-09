#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_models/python"
SP="$WT/runs/hexgt_rl_main2/selfplay"
echo "=== ep12 shard timestamps (new driver started 18:21) ==="
ls -la --time-style=+%H:%M "$SP"/epoch_000012_game_000000.npz "$SP"/epoch_000012_game_000100.npz "$SP"/epoch_000012_game_000130.npz 2>/dev/null
echo "=== STV in a HIGH-numbered (definitely new-driver) ep12 shard ==="
for g in 000130 000120 000100; do
  f="$SP/epoch_000012_game_$g.npz"
  [ -f "$f" ] || continue
  python3 -c "
from hexo_models.dense_cnn.compact_io import read_compact_shard
r=read_compact_shard('$f')
stv=[x.short_term_value for x in r if x.short_term_value]
print('$g: rows=%d rows_with_stv=%d'%(len(r),len(stv)), ('horizons='+str(sorted(h for h,_ in stv[0]))+' sample='+str([(h,round(v,3)) for h,v in stv[0]])) if stv else '(none)')
"
  break
done
echo "=== also re-check game_000000 timestamp vs the 18:21 relaunch ==="
echo "driver started: $(ps -o lstart= -p $(pgrep -f _rl_train.py|head -1) 2>/dev/null)"
