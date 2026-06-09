#!/usr/bin/env bash
echo "=== 1. LIVE DRIVER: pid / start / argv ==="
p=$(pgrep -f _rl_train.py | head -1)
echo "driver pid=$p  start=$(ps -o lstart= -p "$p" 2>/dev/null)  elapsed=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')s"
echo "argv:"; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE -- "--max-actions [0-9]+|--widening-max-children [0-9]+|--visits [0-9]+|--active [0-9]+|--eval-every [0-9]+"
echo
echo "=== 2. widening default in the RUNNING code tree ==="
grep -E "widening-max-children" /mnt/e/Hexo-BotTrainer-hexgt/scripts/_rl_train.py | grep default
grep -nE "cc4748e|honor widening" /dev/null 2>/dev/null
echo "  (commit caa7bdf/cc4748e/7456f96/7502bc0 = bundle; HEAD below)"
git -C /mnt/e/Hexo-BotTrainer-hexgt log --oneline -1
echo
echo "=== 3. SUPERVISOR relaunch history since 16:00 ==="
grep -E "SUPERVISOR start|LAUNCH|relaunch|driver exited|breaker" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/supervise*.log 2>/dev/null | tail -8
echo "(if no supervise log, the supervisor logs to stdout/the bg task)"
echo
echo "=== 4. rl_train.log: current epoch + recent self-play len stats ==="
grep -E "RESUME|ZERO-INIT|epoch [0-9]+ selfplay|epoch [0-9]+ done" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/rl_train.log | tail -10
echo
echo "=== 5. per-epoch game_length_max from selfplay JSON (the truncation tell) ==="
for e in 5 6 7 8; do
  f=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval/epoch_$(printf %06d $e)_selfplay.json
  if [ -f "$f" ]; then
    python3 -c "import json;d=json.load(open('$f'));print('rl_epoch $e: max=%s mean=%.1f med=%s C/T=%s/%s'%(d.get('game_length_max'),d.get('game_length_mean',0),d.get('game_length_median'),d.get('completed_games'),d.get('truncated_games')))"
  else
    echo "rl_epoch $e: (no selfplay.json yet)"
  fi
done
echo
echo "=== 6. epoch-8 in-progress: any .hxr game > 240 moves? (max_actions=512 proof) ==="
ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_000008_game_*.npz 2>/dev/null | wc -l
echo "(npz rows == game length; checking max rows across epoch-8 shards)"
python3 - <<'PY'
import glob, numpy as np
fs = sorted(glob.glob('/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_000008_game_*.npz'))
mx = 0; n = 0
for f in fs:
    try:
        z = np.load(f)
        k = list(z.keys())[0]
        L = z[k].shape[0]
        mx = max(mx, L); n += 1
    except Exception:
        pass
print(f"epoch-8 shards read={n}  max rows(=len) so far={mx}  (>240 proves 512 cap is live)")
PY
