#!/usr/bin/env bash
set -uo pipefail
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3
VENV=/root/.venvs/hexgt-build
F=$(ls -t "$RUN"/selfplay/epoch_000000_game_*.npz 2>/dev/null | head -1)
echo "probing: $F"
"$VENV/bin/python" - "$F" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1])
for k in z.files:
    a = z[k]
    print(f"  {k}: shape={getattr(a,'shape',None)} dtype={getattr(a,'dtype',None)}")
PY
echo "=== driver epoch/throughput lines ==="
grep -iE 'epoch|pos/s|positions|self-?play|games|throughput' "$RUN/driver.20260604_023004.out.log" | tail -10
