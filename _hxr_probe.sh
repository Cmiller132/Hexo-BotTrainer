#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
from hexo_runner.records import HexoRecordFile
import glob
f=sorted(glob.glob('/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval/epoch_000039_vs_dense/*.hxr'))[0]
print("file:", f)
rf=HexoRecordFile.open(f)
print("rf attrs:", [a for a in dir(rf) if not a.startswith('_')])
recs=list(rf.iter_records())
print("n records:", len(recs))
r=recs[0]
print("record attrs:", [a for a in dir(r) if not a.startswith('_')])
for a in ('game_id','winner','players','seed','result','outcome'):
    if hasattr(r,a):
        try: print(f"  r.{a} =", getattr(r,a))
        except Exception as e: print(f"  r.{a} err {e}")
try: print("  len(action_ids) =", len(list(r.action_ids)))
except Exception as e: print("  action_ids err", e)
# metadata / players source
for a in ('metadata','engine_metadata','header'):
    if hasattr(rf,a):
        try: print(f"rf.{a} =", getattr(rf,a))
        except Exception as e: print(f"rf.{a} err {e}")
PY
