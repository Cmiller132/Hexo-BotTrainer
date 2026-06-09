#!/usr/bin/env bash
# Robust steady-state pos/s: total positions across all complete epoch-0 shards
# divided by (latest mtime - earliest mtime). Averages out bursty completion and
# the brief pipeline-fill ramp. Also reports the tail-half rate (second half of
# the shards) to confirm steady state.
set -uo pipefail
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3
VENV=/root/.venvs/hexgt-build
"$VENV/bin/python" - "$RUN/selfplay" <<'PY'
import sys, glob, os, numpy as np
d = sys.argv[1]
recs = []
for f in glob.glob(os.path.join(d, "epoch_000000_game_*.npz")):
    try:
        m = os.path.getmtime(f)
        with np.load(f, allow_pickle=False) as z:
            recs.append((m, int(z["num_rows"])))
    except Exception:
        pass
recs.sort()
n = len(recs)
if n < 3:
    print("not enough shards yet:", n); raise SystemExit
t0, tN = recs[0][0], recs[-1][0]
span = tN - t0
tot = sum(r[1] for r in recs)
print(f"complete shards: {n}")
print(f"span (first->last mtime): {span:.0f}s")
print(f"total positions: {tot}")
print(f"=> cumulative {tot/span:.2f} pos/s  ({n/span*60:.1f} games/min, {tot/n:.1f} pos/game)")
# tail-half (steady state, excludes ramp)
half = recs[n//2:]
ht0, htN = half[0][0], half[-1][0]; hspan = htN-ht0; htot = sum(r[1] for r in half)
if hspan > 0:
    print(f"=> tail-half {htot/hspan:.2f} pos/s  (over {hspan:.0f}s, {len(half)} games)")
PY
echo "=== driver alive? ==="; kill -0 54299 2>/dev/null && echo "ALIVE" || echo "GONE"
