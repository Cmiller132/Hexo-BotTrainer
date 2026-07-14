#!/usr/bin/env bash
# Launch the hexfield_eq_main_3 soak — the ADDITIVE reach-conditioned ray-tap
# arm (ray7lut2, Wave-2 selected): A5 arch + HEXFIELD_EQ_RAYTAP_LUT=additive,
# warm-started from the main_2 ep72 prefit soak_init. Single-variable test vs
# main_2: config is a verbatim copy except run names + initialize_from, so the
# strength delta attributes to the additive tables alone.
#
# Mirrors _hexfield_eq_launch_soak_main2.sh but: arch env = a5_lut2 (adds the
# LUT knob), CONFIG = configs/hexfield_eq_main_3.toml, RUNDIR = ..._main_3.
# ROOT = this branch's tree (the additive mechanism exists only on
# claude/resume-run-crash-fdef2b). Doubles as the relaunch script: the
# supervisor injects resume_from when checkpoints exist.
# Holds the session a few seconds after the detach: launched via a bare
# `wsl -e`, the WSL teardown otherwise races the child's daemonization.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname "$SCRIPT_DIR")}"
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_3

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env"
ROOT="$ROOT"
VENV=/root/.venvs/hexgt-build
CONFIG="$ROOT/configs/hexfield_eq_main_3.toml"
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

if [[ ! -f /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main3_prefit/additive/soak_init.pt ]]; then
  echo "ABORT: main_3 prefit soak_init.pt not found — run" >&2
  echo "  scripts/_hexfield_eq_main3_prefit.sh   first." >&2
  exit 1
fi

mkdir -p "$RUNDIR"
rm -f "$RUNDIR/supervisor_halted.flag" "$RUNDIR/supervisor.lock"
{ echo; echo "===== main_3 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) (ROOT=$ROOT) ====="; } >> "$RUNDIR/supervisor_nohup.log"
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
