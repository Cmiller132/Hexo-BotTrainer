#!/usr/bin/env bash
set -uo pipefail
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3
VENV=/root/.venvs/hexgt-build
# pick an OLDER fully-written shard (not the newest few)
F=$(ls -t "$RUN"/selfplay/epoch_000000_game_*.npz 2>/dev/null | sed -n '8p')
echo "probing older shard: $F"
"$VENV/bin/python" - "$F" <<'PY'
import sys, numpy as np
try:
    z = np.load(sys.argv[1])
    for k in z.files:
        a = z[k]
        print(f"  {k}: shape={a.shape} dtype={a.dtype}")
except Exception as e:
    print("LOAD FAILED:", repr(e))
PY
