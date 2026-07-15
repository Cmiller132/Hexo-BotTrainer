#!/usr/bin/env bash
# Launch the hexfield_eq_main_2 soak (ray-tap wave-1 winner arch, A5:
# feature-v2 + RAYTAP=both + CCACCACA). Mirrors the main_1 relaunch script
# but: ROOT = this branch's tree (the main tree cannot build this net),
# arch env = hexfield_eq_raytap_a5.env, CONFIG = configs/hexfield_eq_main_2.toml.
# Holds the session a few seconds after the detach: launched via a bare
# `wsl -e`, the WSL teardown otherwise races the child's daemonization
# (observed on the 2026-07-10 main_1 relaunch).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname "$SCRIPT_DIR")}"
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_2

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_raytap_a5.env"
ROOT="$ROOT"
VENV=/root/.venvs/hexgt-build
CONFIG="$ROOT/configs/hexfield_eq_main_2.toml"
RUNDIR="$RUNDIR"
SEALBOT_PATH=/mnt/e/SealBot
HEXFIELD_ANCHOR_ROOTS="$ROOT"
MALLOC_TRIM_THRESHOLD_=536870912
MALLOC_MMAP_THRESHOLD_=536870912
MALLOC_TOP_PAD_=134217728
# Pin the trainer's microbucket class to the raytap-known-good budget: the
# 4.0e7 materialized-path crash (RAYTAP_RESULTS_WAVE1.md anomaly 1) is
# root-cause-OPEN, so main_2's first flight avoids larger buckets even on the
# flex path. Costs a little step-packing efficiency; lift after the trace.
HEXFIELD_TRAIN_PAIR_BUDGET=1.6e7
# Cap the inductor fork-worker pool (defaults to nproc=32; each worker holds
# hundreds of MB during compile bursts — a multi-GB guest-RAM spike on a VM
# that OOM-crashed twice on 2026-07-10). 8 is plenty for warmup compiles.
TORCHINDUCTOR_COMPILE_THREADS=8
set +a

mkdir -p "$RUNDIR"
rm -f "$RUNDIR/supervisor_halted.flag" "$RUNDIR/supervisor.lock"
{ echo; echo "===== main_2 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) (ROOT=$ROOT) ====="; } >> "$RUNDIR/supervisor_nohup.log"
nohup setsid bash "$ROOT/scripts/_hexfield_eq_supervise_main1.sh" >> "$RUNDIR/supervisor_nohup.log" 2>&1 &
PID=$!
echo "supervisor launched pid=$PID"
# Hold the session so the detached child survives `wsl -e` teardown, then
# report liveness + the first supervisor lines.
sleep 8
if kill -0 "$PID" 2>/dev/null; then
  echo "supervisor alive after 8s"
else
  echo "WARNING: supervisor pid gone — check $RUNDIR/supervisor_nohup.log" >&2
fi
tail -4 "$RUNDIR/supervisor.log" 2>/dev/null || true
