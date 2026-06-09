#!/usr/bin/env bash
# Windowed pos/s: sum num_rows across complete epoch-0 shards at two snapshots.
set -uo pipefail
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3
VENV=/root/.venvs/hexgt-build
WINDOW=120
count_pos() {
  "$VENV/bin/python" - "$RUN/selfplay" <<'PY'
import sys, glob, os, numpy as np
d = sys.argv[1]
tot = 0; n = 0
for f in glob.glob(os.path.join(d, "epoch_000000_game_*.npz")):
    try:
        with np.load(f, allow_pickle=False) as z:
            tot += int(z["num_rows"]); n += 1
    except Exception:
        pass   # partially-written / in-flight shard
print(f"{n} {tot}")
PY
}
read G0 P0 < <(count_pos); T0=$EPOCHSECONDS
sleep "$WINDOW"
read G1 P1 < <(count_pos); T1=$EPOCHSECONDS
DT=$((T1 - T0))
echo "snapshot0: games=$G0 positions=$P0"
echo "snapshot1: games=$G1 positions=$P1  (elapsed ${DT}s)"
python3 - "$P0" "$P1" "$DT" "$G0" "$G1" <<'PY'
import sys
p0,p1,dt,g0,g1 = (float(x) for x in sys.argv[1:6])
dp=p1-p0; dg=g1-g0
print(f"=> {dp:.0f} positions over {dt:.0f}s = {dp/dt:.2f} pos/s | {dg:.0f} games = {dg/dt*60:.1f} games/min | {dp/max(dg,1):.1f} pos/game")
PY
