#!/usr/bin/env bash
# Relaunch hexfield_eq_main_1 soak with the fast serve profile (2026-07-09).
# Mirrors eq_ladder_runner.launch_soak: arm-4 env + operational env, then
# nohup setsid the supervise script (which now defaults the fast profile).
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_arm4_raylayout.env"
ROOT="$ROOT"
VENV=/root/.venvs/hexgt-build
CONFIG=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/hexfield_eq_main_1.launch.toml
RUNDIR="$RUNDIR"
SEALBOT_PATH=/mnt/e/SealBot
HEXFIELD_ANCHOR_ROOTS="$ROOT"
MALLOC_TRIM_THRESHOLD_=536870912
MALLOC_MMAP_THRESHOLD_=536870912
MALLOC_TOP_PAD_=134217728
set +a

rm -f "$RUNDIR/supervisor_halted.flag" "$RUNDIR/supervisor.lock"
{ echo; echo "===== fast-serve-profile relaunch $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="; } >> "$RUNDIR/supervisor_nohup.log"
nohup setsid bash "$ROOT/scripts/_hexfield_eq_supervise_main1.sh" >> "$RUNDIR/supervisor_nohup.log" 2>&1 &
echo "supervisor launched pid=$!"
