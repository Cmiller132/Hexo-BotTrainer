#!/usr/bin/env bash
# Launch hexfield_eq_main_5 — the 15-block CCAx5 capacity line, WEIGHTS-ONLY
# warm-started from the main_5 BC prefit (initialize_from = the epoch-2 prefit
# checkpoint per owner ruling 2026-07-22, NOT resume_from; the supervisor
# injects resume_from only once main_5 checkpoints exist). Fresh optimizer
# state, fresh replay window, epoch counter restarts at 0.
#
# Arch = hexfield_eq_main5_cca15.env (TRUNK=CCACCACCACCACCA, CELL_Q=0); the
# prefit checkpoint's arch_meta must match this env exactly (self-describing
# {meta,model}; the supervisor's HEXFIELD_EQ_* block below is the launch-time
# authority and is what the load asserts against).
#
# Deltas vs _hexfield_eq_launch_main4.sh: CONFIG = hexfield_eq_main_5.toml,
# RUNDIR = ..._main_5, arch env = main5_cca15, prefit check = the epoch-2
# checkpoint. ROOT resolves to THIS tree (the main5-prep worktree — its
# hexfield_eq package + freshly-built _rust .so). INFRA packages resolve from
# INFRA_TREE (same split main_4 used).
# Holds the session a few seconds after the detach so the WSL teardown does not
# race the child's daemonization.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname "$SCRIPT_DIR")}"
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_5
PREFIT_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main5_prefit/cca15/checkpoint_epoch1.pt

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_main5_cca15.env"
ROOT="$ROOT"
VENV=/root/.venvs/hexgt-build
CONFIG="$ROOT/configs/hexfield_eq_main_5.toml"
RUNDIR="$RUNDIR"
SEALBOT_PATH=/mnt/e/SealBot
HEXFIELD_ANCHOR_ROOTS="$ROOT"
MALLOC_TRIM_THRESHOLD_=536870912
MALLOC_MMAP_THRESHOLD_=536870912
MALLOC_TOP_PAD_=134217728
# Raytap-known-good microbucket budget (the 4.0e7 materialized-path crash is
# still root-cause-OPEN; see RAYTAP_RESULTS_WAVE1.md anomaly 1). The 15-block
# trunk roughly doubles activation memory vs main_4's 8 blocks; if selfplay or
# train OOMs at the 12 GB WDDM line, lower this before anything else.
HEXFIELD_TRAIN_PAIR_BUDGET=1.6e7
TORCHINDUCTOR_COMPILE_THREADS=8
set +a

if [[ ! -f "$PREFIT_CKPT" ]]; then
  echo "ABORT: main_5 prefit checkpoint not found:" >&2
  echo "  $PREFIT_CKPT" >&2
  echo "  (run scripts/_hexfield_eq_main5_prefit.sh to epoch 1 first)" >&2
  exit 1
fi

mkdir -p "$RUNDIR"
rm -f "$RUNDIR/supervisor_halted.flag" "$RUNDIR/supervisor.lock"
{ echo; echo "===== main_5 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) (ROOT=$ROOT) ====="; } >> "$RUNDIR/supervisor_nohup.log"
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
