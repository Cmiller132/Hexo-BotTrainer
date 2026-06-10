#!/usr/bin/env bash
# Bounce dense_cnn_restnet_main1 to pick up the select<->eval overlap rebuild.
#
# Safe order: rebuild + scheduler tests run FIRST while the live run keeps going
# on its already-loaded .so (Linux keeps the old inode after maturin replaces the
# file). Only if both pass do we stop the supervisor and relaunch — so a failed
# build/test leaves the run untouched.
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
VENV=/root/.venvs/hexgt-build

echo "=== [1/4] rebuild (release), run still live ==="
bash "$ROOT/scripts/_rebuild_hexo_models_hexgt.sh" || { echo "REBUILD FAILED — run untouched"; exit 1; }

echo "=== [2/4] native scheduler tests (CPU, won't fight the GPU run) ==="
source "$VENV/bin/activate"
cd "$ROOT"
python -m pytest tests/test_dense_cnn_continuous_scheduler.py -x -q 2>&1 | tail -8
rc=${PIPESTATUS[0]}
[[ $rc -ne 0 ]] && { echo "SCHEDULER TESTS FAILED (rc=$rc) — run untouched, NOT relaunching"; exit 1; }

echo "=== [3/4] stop supervisor session ==="
SUP=$(cat "$RD/supervisor.lock" 2>/dev/null || true)
if [[ -n "${SUP:-}" ]] && kill -0 "$SUP" 2>/dev/null; then
  PGID=$(ps -o pgid= -p "$SUP" | tr -d ' ')
  echo "supervisor=$SUP pgid=$PGID -> TERM group (stops supervisor + driver, no relaunch)"
  kill -TERM -- "-$PGID" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SUP" 2>/dev/null || break; sleep 1; done
  kill -0 "$SUP" 2>/dev/null && { echo "supervisor alive; KILL group"; kill -KILL -- "-$PGID" 2>/dev/null; sleep 2; }
else
  echo "supervisor not running (lock pid=${SUP:-none})"
fi
if pgrep -f "train_model.*_resume_config" >/dev/null 2>&1; then
  echo "stray driver remains; aborting before relaunch"; pgrep -af "train_model.*_resume_config"; exit 1
fi
rm -f "$RD/supervisor.lock"
# Drop epoch-31's partial self-play artifacts (it had only just started); the
# resume redoes epoch 31 from epoch_000030.pt and rewrites these cleanly.
rm -f "$RD"/selfplay/epoch_000031_game_*.npz "$RD"/selfplay/epoch_000031_game_*.json "$RD"/selfplay/epoch_000031.hxr 2>/dev/null || true

echo "=== [4/4] relaunch ==="
bash "$ROOT/scripts/_dc_restnet_launch_main1.sh"
