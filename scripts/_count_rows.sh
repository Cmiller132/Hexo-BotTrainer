#!/usr/bin/env bash
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python
/root/.venvs/hexgt-build/bin/python - "$RUNDIR" <<'PY'
import glob, os, re, sys
from pathlib import Path
from dense_cnn_restnet.replay import npz_row_count
rd = sys.argv[1]
shards = sorted(glob.glob(os.path.join(rd, "selfplay", "**", "*.npz"), recursive=True))
by_epoch = {}
for f in shards:
    m = re.search(r"epoch_0*([0-9]+)", os.path.basename(f))
    ep = int(m.group(1)) if m else -1
    by_epoch[ep] = by_epoch.get(ep, 0) + npz_row_count(Path(f))
for ep in sorted(by_epoch):
    print(f"epoch {ep}: {by_epoch[ep]} recorded rows")
print("TOTAL:", sum(by_epoch.values()), "rows across", len(shards), "shards")
PY
