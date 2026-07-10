#!/usr/bin/env bash
# Launcher for the autonomous hexfield_eq prefit-ladder runner
# (scripts/eq_ladder_runner.py). Run from WSL. Detaches the runner with
# nohup+setsid so it survives the launching (orchestrator) session, prints the
# PID, and points at the two status surfaces.
#
#   EQ_LADDER_LIMIT_STEPS=<calibrated cap> \
#   EQ_LADDER_DEADLINE_TS=$(( $(date +%s) + 6*3600 )) \
#   bash /mnt/e/Hexo-BotTrainer-hexgt/scripts/run_eq_ladder.sh
#
# Any extra args are forwarded to eq_ladder_runner.py (e.g. --dry-run
# --mock-root /tmp/eq_mock, --deadline-in-minutes N). The deadline-regime
# calibration knobs (EQ_LADDER_BATCH_ROWS / _LR / _WARMUP_STEPS /
# _PAIR_BUDGET_CA / _PAIR_BUDGET_L / _EVAL_GAMES) are env vars read by the
# runner at start. Docs: docs/AUTONOMOUS_LADDER_RUNNER.md.
set -euo pipefail

ROOT="${EQ_LADDER_REPO:-/mnt/e/Hexo-BotTrainer-hexgt}"
VENV_PY="${EQ_LADDER_VENV_PY:-/root/.venvs/hexgt-build/bin/python}"
LADDER_ROOT="${EQ_LADDER_ROOT:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit}"
DATA_DIR="${EQ_LADDER_DATA:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11}"
RUNNER="$ROOT/scripts/eq_ladder_runner.py"

fail() { echo "FATAL: $*" >&2; exit 1; }

[[ -x "$VENV_PY" ]] || fail "venv python not found/executable: $VENV_PY"
[[ -f "$RUNNER" ]] || fail "runner not found: $RUNNER"
[[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] \
  || fail "converted corpus missing train/ + val/ under $DATA_DIR (run the converter first)"
# Arm set: EQ_LADDER_ARMS (comma list of name[:l] tokens, e.g. the ray-tap
# wave-1 "raytap_a0:l,...") or the default R/L ladder.
ARM_NAMES="${EQ_LADDER_ARMS:-arm1_vanilla,arm2_reglane,arm3_tokread,arm4_raylayout,arm4c_georay}"
IFS=',' read -ra ARM_TOKS <<< "$ARM_NAMES"
for tok in "${ARM_TOKS[@]}"; do
  arm="${tok%%:*}"
  arm="$(echo "$arm" | xargs)"
  [[ -z "$arm" ]] && continue
  [[ -f "$ROOT/scripts/prefit_env/hexfield_eq_${arm}.env" ]] \
    || fail "missing arm env file: scripts/prefit_env/hexfield_eq_${arm}.env"
done

mkdir -p "$LADDER_ROOT"
LOG="$LADDER_ROOT/runner.log"

# Refuse a duplicate launch (the runner also holds its own pidfile lock).
if [[ -f "$LADDER_ROOT/ladder_runner.lock" ]]; then
  OLD_PID="$(cat "$LADDER_ROOT/ladder_runner.lock" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    fail "ladder runner already alive (pid $OLD_PID). Status: tail $LADDER_ROOT/LADDER_STATUS.md"
  fi
fi

nohup setsid "$VENV_PY" -u "$RUNNER" "$@" >> "$LOG" 2>&1 &
PID=$!
sleep 3
# setsid may fork (pid changes when we were a process-group leader): trust the
# runner's own lock file over $PID before declaring a failed launch.
if ! kill -0 "$PID" 2>/dev/null; then
  LOCK_PID="$(cat "$LADDER_ROOT/ladder_runner.lock" 2>/dev/null || true)"
  if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
    PID="$LOCK_PID"
  else
    echo "runner exited immediately — last log lines:" >&2
    tail -n 20 "$LOG" >&2 || true
    exit 1
  fi
fi
echo "eq_ladder_runner launched: pid=$PID"
echo "  runner log : tail -f $LOG"
echo "  status     : tail -f $LADDER_ROOT/LADDER_STATUS.md"
echo "  state json : cat $LADDER_ROOT/ladder_state.json"
