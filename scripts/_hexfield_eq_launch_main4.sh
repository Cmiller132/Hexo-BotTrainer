#!/usr/bin/env bash
# Launch hexfield_eq_main_4 — the clean fresh-start run WEIGHTS-ONLY warm-started
# from main_3 epoch_000090.pt (owner ruling 2026-07-20). Same A5 additive
# ray-tap arch as main_3 (a5_lut2 env), so the ep90 weights load shape-for-shape.
# Fresh optimizer state, fresh replay window, epoch counter restarts at 0
# (initialize_from, NOT resume_from — the supervisor injects resume_from only
# once main_4 checkpoints exist).
#
# Deltas vs _hexfield_eq_launch_soak_main3.sh: CONFIG = hexfield_eq_main_4.toml,
# RUNDIR = ..._main_4, prefit check = the main4_prefit soak_init. ROOT resolves
# to THIS tree (the merged claude/main4-integration worktree — the dual-pass +
# G2 + ordering solver code lives only here; build it into the venv from WSL
# via maturin release FIRST, see docs/MAIN4_LAUNCH_PLAN.md).
# Holds the session a few seconds after the detach: launched via a bare
# `wsl -e`, the WSL teardown otherwise races the child's daemonization.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname "$SCRIPT_DIR")}"
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env"
ROOT="$ROOT"
VENV=/root/.venvs/hexgt-build
CONFIG="$ROOT/configs/hexfield_eq_main_4.toml"
RUNDIR="$RUNDIR"
SEALBOT_PATH=/mnt/e/SealBot
HEXFIELD_ANCHOR_ROOTS="$ROOT"
MALLOC_TRIM_THRESHOLD_=536870912
MALLOC_MMAP_THRESHOLD_=536870912
MALLOC_TOP_PAD_=134217728
# Raytap-known-good microbucket budget (the 4.0e7 materialized-path crash is
# still root-cause-OPEN; see RAYTAP_RESULTS_WAVE1.md anomaly 1).
HEXFIELD_TRAIN_PAIR_BUDGET=1.6e7
# Cap the inductor fork-worker pool (nproc=32 workers OOM-crashed the VM twice
# on 2026-07-10 during compile bursts).
TORCHINDUCTOR_COMPILE_THREADS=8
set +a

if [[ ! -f /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main4_prefit/main3ep90/soak_init.pt ]]; then
  echo "ABORT: main_4 prefit soak_init.pt not found — run" >&2
  echo "  scripts/_main4_repackage.sh   first." >&2
  exit 1
fi

mkdir -p "$RUNDIR"
rm -f "$RUNDIR/supervisor_halted.flag" "$RUNDIR/supervisor.lock"
{ echo; echo "===== main_4 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) (ROOT=$ROOT) ====="; } >> "$RUNDIR/supervisor_nohup.log"
nohup setsid bash "$ROOT/scripts/_hexfield_eq_supervise_main1.sh" >> "$RUNDIR/supervisor_nohup.log" 2>&1 &
PID=$!
echo "supervisor launched pid=$PID"
# Hold the session so the detached child survives `wsl -e` teardown.
sleep 8
if kill -0 "$PID" 2>/dev/null; then
  echo "supervisor alive after 8s"
else
  echo "WARNING: supervisor pid gone — check $RUNDIR/supervisor_nohup.log" >&2
fi
tail -4 "$RUNDIR/supervisor.log" 2>/dev/null || true
