#!/usr/bin/env bash
echo "=== driver pid/start (is it the 16:00 bundle process? crash-relaunch?) ==="
for p in $(pgrep -f _rl_train.py); do
  echo "pid=$p start=$(ps -o lstart= -p $p) etimes=$(ps -o etimes= -p $p|tr -d ' ')s"
done
echo "supervisor:"; pgrep -af "_rl_supervise|_rl_run_fg" | grep -v pgrep | head
echo
echo "=== driver count of LAUNCH lines (relaunches) in bg/driver logs ==="
ls -1 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/driver.*.out.log 2>/dev/null | tail -3
echo
echo "=== npz shard keys + per-game length for epoch 7 (old=240 cap) vs epoch 8 (new=512) ==="
python3 - <<'PY'
import glob, numpy as np
for ep in (7, 8):
    fs = sorted(glob.glob(f'/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_{ep:06d}_game_*.npz'))
    if not fs:
        print(f'epoch {ep}: no shards'); continue
    z0 = np.load(fs[0])
    keys = list(z0.keys())
    # row count = number of positions in the game = game length
    def rows(f):
        z = np.load(f)
        return max(int(z[k].shape[0]) for k in z.keys() if hasattr(z[k],'shape') and z[k].ndim>=1)
    lens = []
    for f in fs:
        try: lens.append(rows(f))
        except Exception as e: pass
    lens.sort()
    print(f'epoch {ep}: shards={len(fs)} keys={keys[:6]}')
    if lens:
        import statistics as st
        print(f'   len: min={lens[0]} med={st.median(lens):.0f} max={lens[-1]} | #>240={sum(1 for L in lens if L>240)} | #==240={sum(1 for L in lens if L==240)}')
PY
echo
echo "=== bridge: display-epoch mapping (_disp) ==="
grep -nE "_disp|rl_epoch.*\+.*1|current_epoch" /mnt/e/Hexo-BotTrainer-hexgt/_dashboard_bridge.py | head -6
