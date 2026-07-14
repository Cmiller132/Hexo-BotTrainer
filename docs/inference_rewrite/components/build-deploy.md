# build-deploy

The Bash tool ran but the venv/run listing came back empty (sandboxed bash may not see /root or /mnt mounts the same way). The key facts I need are all confirmed from the scripts I already read: the live venv is `/root/.venvs/hexgt-build` (has torch), the dev/build venv for maturin is `/root/.venvs/hexfield-dev`, the run dir is `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1`, supervisor controls are `supervisor.lock` / `supervisor_halted.flag` / `driver.pid`.

I have everything needed. Here is the deliverable.

---

# HEXFIELD INFERENCE-REWRITE — BUILD / DEPLOY / ROLLBACK RUNBOOK

This is the operator procedure for assembling the accepted rewrite components into the **separate worktree** `E:\Hexo-BotTrainer-hexgt-rewrite` (branch `claude/inference-rewrite`), building the Rust extension, validating against the parity gates, cutting it over for the deliberate GPU pause, and rolling back. It touches the live tree **read-only** and is consistent with the existing `scripts/_rebuild_hexfield.sh`, `scripts/_hexfield_supervise_main1.sh`, and `configs/hexfield_main_1.toml`.

Design facts this runbook is wired against (from the live tree, read-only):
- Live torch venv (has torch, runs the trainer/parity harnesses): `/root/.venvs/hexgt-build` — `_hexfield_supervise_main1.sh:14`.
- Rust build venv (maturin target; **no torch**): `/root/.venvs/hexfield-dev` — `_rebuild_hexfield.sh:10`.
- Run dir + control files: `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1` with `supervisor.lock`, `supervisor_halted.flag`, `driver.pid` — `_hexfield_supervise_main1.sh:16-20,75`.
- Rebuild copies the built `_rust*.so` back into the source tree at `packages/hexfield/python/hexfield/` — `_rebuild_hexfield.sh:20-25`.
- The trainer imports hexfield from source via `PYTHONPATH=$ROOT/packages/hexfield/python:...` — `_hexfield_supervise_main1.sh:36`. **So the live process loads the `.so` and `.py` from whatever tree `$ROOT` points at.** Cutover = pointing `$ROOT` at the rewrite worktree. Rollback = pointing it back.
- New gating env flags introduced by the rewrite: `HEXFIELD_ATTN_IMPL` (default `sdpa`), `HEXFIELD_LARGE_NPAD` (default = `HEXFIELD_COMPILE_MAX_NPAD` = 512), and the v2-ABI selector carried in-payload (`"abi": 2`). Default values keep the live path byte-identical to today.

---

## File 1 — `scripts/_rewrite_preflight.sh`

Read-only assertion that the live run is paused and the GPU is free, plus a copy of the live checkpoint pointer (recoverability). Run this **first**. It writes nothing to the live tree.

```bash
#!/usr/bin/env bash
# Pause-window preflight for the inference-rewrite cutover. READ-ONLY against the
# live tree. Verifies: (1) the hexfield supervisor + trainer are STOPPED, (2) the
# GPU is free, (3) the resume checkpoint exists, and records the current live
# git SHA + run pointer so rollback is mechanical. Touches no run files.
set -uo pipefail

LIVE="/mnt/e/Hexo-BotTrainer-hexgt"
RUNDIR="/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
CKPTS="$RUNDIR/checkpoints"
OUT="/mnt/e/Hexo-BotTrainer-hexgt-rewrite/_preflight.$(date -u +%Y%m%d_%H%M%S).txt"

say(){ echo "$*" | tee -a "$OUT" >&2; }
fail=0

mkdir -p "$(dirname "$OUT")"
say "== rewrite preflight $(date -u +%FT%TZ) =="

# 1. supervisor + trainer must be DOWN (the supervisor relaunches the trainer, so
#    it must die first — same ordering rationale as _restart_supervisor.sh).
say "-- live hexfield processes (expect NONE) --"
PROCS="$(pgrep -af 'hexfield_supervise_main1|hexo_train.cli.train_model .*hexfield' | grep -v pgrep || true)"
if [[ -n "$PROCS" ]]; then say "FAIL: live hexfield procs still running:"; say "$PROCS"; fail=1
else say "ok: no hexfield supervisor/trainer running"; fi

# 2. lock / halt state
[[ -f "$RUNDIR/supervisor.lock" ]] && { say "FAIL: supervisor.lock present (pid $(cat "$RUNDIR/supervisor.lock" 2>/dev/null)) — supervisor not cleanly down"; fail=1; } \
  || say "ok: no supervisor.lock"

# 3. GPU free (no compute apps). Mirrors _check_gpu_state.sh.
say "-- GPU compute apps (expect none) --"
APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "$APPS" ]]; then say "WARN: GPU has compute apps (confirm they are not the live run):"; say "$APPS"
else say "ok: GPU compute-app list empty"; fi
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1 | tee -a "$OUT" >&2

# 4. resume target exists (the supervisor resumes from the latest epoch ckpt).
LC="$(ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "$LC" ]]; then say "ok: resume checkpoint = $LC"; else say "WARN: no epoch_*.pt — would FIRST-LAUNCH from BC prefit"; fi

# 5. record live git SHA for the diff base + rollback reference (read-only).
say "-- live tree git SHA (rollback base) --"
git -C "$LIVE" rev-parse HEAD 2>/dev/null | tee -a "$OUT" >&2 || say "WARN: live tree SHA unavailable"

say "== preflight $([[ $fail -eq 0 ]] && echo PASS || echo FAIL) =="
exit $fail
```

---

## File 2 — `scripts/_rewrite_build.sh`

Creates/refreshes the worktree, builds the Rust extension in the dev venv, and mirrors the `.so` into the worktree source — the exact pattern of `_rebuild_hexfield.sh` but pointed at the **rewrite worktree**, never the live tree. Idempotent.

```bash
#!/usr/bin/env bash
# Build the inference-rewrite worktree. Mirrors scripts/_rebuild_hexfield.sh but
# targets E:\Hexo-BotTrainer-hexgt-rewrite (branch claude/inference-rewrite).
# NEVER builds into the live tree. Run while the live run is PAUSED (maturin uses
# the GPU-free toolchain only; the .so build is CPU, but the parity step that
# follows is GPU and needs the pause).
set -euo pipefail

LIVE="/mnt/e/Hexo-BotTrainer-hexgt"
WT="/mnt/e/Hexo-BotTrainer-hexgt-rewrite"
BRANCH="claude/inference-rewrite"
DEV_VENV="/root/.venvs/hexfield-dev"

# 0. Create the worktree from the live tree's git repo if it doesn't exist.
#    (The operator has already assembled accepted code onto $BRANCH; this just
#     ensures the checkout exists. If it exists, leave it — do NOT reset operator work.)
if [[ ! -d "$WT/.git" && ! -f "$WT/.git" ]]; then
  echo "creating worktree $WT on $BRANCH"
  git -C "$LIVE" worktree add "$WT" "$BRANCH" 2>/dev/null \
    || git -C "$LIVE" worktree add -b "$BRANCH" "$WT" HEAD
fi
echo "worktree at $WT, branch: $(git -C "$WT" rev-parse --abbrev-ref HEAD)"

cd "$WT"
source "$DEV_VENV/bin/activate"
export PATH="/root/.cargo/bin:$PATH"   # rustup toolchain (lockfile v4), not apt cargo

# --release is mandatory (debug featurizer/search ~10x slower) — same as live rebuild.
maturin develop --release -m packages/hexfield/Cargo.toml

# Mirror the built cdylib into the worktree source tree so the live torch venv
# (hexgt-build) imports it via PYTHONPATH at serve time. Identical to
# _rebuild_hexfield.sh:20-25 except the destination is the WORKTREE, not live.
SO=$(ls "$DEV_VENV"/lib/python3.12/site-packages/hexfield/_rust*.so 2>/dev/null | head -1)
if [[ -n "${SO:-}" ]]; then
  cp "$SO" "$WT/packages/hexfield/python/hexfield/"
  echo "copied $(basename "$SO") into $WT/packages/hexfield/python/hexfield/"
else
  echo "ERROR: no _rust*.so produced by maturin"; exit 1
fi
ls -la "$WT"/packages/hexfield/python/hexfield/_rust*.so
echo "BUILD OK — worktree built, .so mirrored into source."
```

---

## File 3 — `scripts/_rewrite_validate.sh`

Runs the three parity tiers in order, from the **rewrite worktree**, using the **live torch venv** (the parity harnesses need torch, which lives only in `hexgt-build`). Tier 1 is GPU-free; Tiers 2/3 need the pause. This is the gate: nothing cuts over until this exits 0.

```bash
#!/usr/bin/env bash
# Parity validation for the inference rewrite, run from the worktree against the
# LIVE torch venv (hexgt-build has torch; hexfield-dev does NOT). Tier 1 is
# GPU-free (statically-certain bias-index oracle); Tiers 2/3 need the GPU pause.
# Reuses the existing harnesses + thresholds — invents no new tolerances.
set -uo pipefail

WT="/mnt/e/Hexo-BotTrainer-hexgt-rewrite"
PY="/root/.venvs/hexgt-build/bin/python"
export PYTHONPATH="$WT/packages/hexfield/python:$WT/packages/dense_cnn_restnet/python"
cd "$WT"
rc=0

echo "===== TIER 1 (GPU-FREE): bias-index + math oracles ====="
# Includes the new hexflash/flex bias-index equality test (test_pair_index_*),
# the SDPA-vs-materialized oracle (CPU fp32 portion), and the kernel-constants
# single-source assertion (constants.py owner). torch.equal integer math — certain.
"$PY" -m pytest -q packages/hexfield/python/tests/test_hexfield_model.py \
  -k "pair_index or sdpa_equals or hexflash_bias or constants_single_source" \
  || { echo "TIER 1 FAIL"; rc=1; }

echo "===== TIER 2 (GPU): fp16 output oracle (hexflash + flex) ====="
# Extends test_sdpa_equals_materialized_fp16_cuda to assert impl=hexflash and
# impl=flex vs materialized + sdpa, SAME budget already in the file
# (fp16 diff<=2e-3, fp32<=1e-4). Padded row exercises the pad-key mask.
"$PY" -m pytest -q packages/hexfield/python/tests/test_hexfield_model.py \
  -k "fp16_cuda" \
  || { echo "TIER 2 FAIL — hexflash blocked; fall back to HEXFIELD_ATTN_IMPL=flex"; rc=1; }

echo "===== TIER 3a (GPU): compile + async parity, large-S band ====="
# COMPILE-PARITY TOL=3e-3 on values/priors/moves_left; ASYNC maxabsdiff==0.0.
# The harness 'cases' now include large-S sizes (1024/2048/3300). Run with each
# impl so the new large-S routing is actually covered.
for impl in sdpa hexflash flex; do
  echo "--- compile/overlap @ HEXFIELD_ATTN_IMPL=$impl ---"
  HEXFIELD_NO_COMPILE= HEXFIELD_ATTN_IMPL="$impl" "$PY" \
    scripts/_hexfield_compile_overlap_test.py \
    || { echo "TIER 3a FAIL (impl=$impl)"; rc=1; }
done

echo "===== TIER 3b (GPU): v2-ABI byte parity + depth-2 action parity ====="
# v2 dense-scatter vs v1 numpy loop must be torch.equal (byte-exact: same fp16
# feats, same gather idx, same coords). Depth-2 pipeline must preserve the
# action sequence (overlap only moves sync points; FIFO drain keeps cache order).
HEXFIELD_NO_COMPILE= "$PY" scripts/_hexfield_async_parity.py \
  || { echo "TIER 3b FAIL"; rc=1; }

echo "===== VALIDATION $([[ $rc -eq 0 ]] && echo PASS || echo FAIL) ====="
exit $rc
```

Notes for the operator:
- If **Tier 2 fails for hexflash but passes for flex**, the rewrite is still shippable: deploy with `HEXFIELD_ATTN_IMPL=flex` (the spec's gated fallback). The cutover supervisor (File 4) reads this flag, so no code change is needed — just export `flex` instead of `hexflash`.
- Tier 1 is the only statically-certain gate. Treat a Tier 1 failure as a **blocker** — it means the kernel's bias-row index math diverged from `build_attn_bias`, which would silently corrupt the learned function.

---

## File 4 — `scripts/_rewrite_supervise_main1.sh`

The cutover supervisor. This is a near-verbatim copy of `_hexfield_supervise_main1.sh` with three changes: (1) `ROOT` points at the **worktree**, (2) it exports the new gating env flags, (3) the resume/breaker/lock logic is unchanged so the run resumes from the same `epoch_*.pt`. The defaults keep the live path byte-identical; flip `HEXFIELD_ATTN_IMPL` to `hexflash` (or `flex`) only after validation passes.

```bash
#!/usr/bin/env bash
# Inference-REWRITE supervisor for hexfield_main_1. Identical control flow to
# scripts/_hexfield_supervise_main1.sh (auto-relaunch, circuit breaker, single-
# instance lock, halt flag, resume_from injection) — the ONLY differences are
# ROOT pointing at the rewrite worktree and the new gating env flags. Same RUNDIR,
# so it resumes from the same epoch checkpoint and renders on the :8080 dashboard.
set -uo pipefail

# >>> CUTOVER: ROOT is the worktree, not the live tree. <<<
ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt-rewrite}"
VENV="${VENV:-/root/.venvs/hexgt-build}"
CONFIG="${CONFIG:-$ROOT/configs/hexfield_main_1.toml}"
RUNDIR="${RUNDIR:-/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1}"

CKPTS="$RUNDIR/checkpoints"
SUPLOG="$RUNDIR/supervisor.log"; LOCK="$RUNDIR/supervisor.lock"
HALT="$RUNDIR/supervisor_halted.flag"; DONE="$RUNDIR/supervisor_completed.flag"
PY="$VENV/bin/python"
FAST_CRASH_SECONDS=300; MAX_CONSEC_FAST=3; MAX_PER_HOUR=8

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SEALBOT_PATH="${SEALBOT_PATH:-/mnt/e/SealBot}"
export PYTHONPATH="$ROOT/packages/hexfield/python:$ROOT/packages/dense_cnn_restnet/python"
export HEXFIELD_ASYNC_EVAL="${HEXFIELD_ASYNC_EVAL:-1}"

# ---- NEW gating flags (rewrite). Defaults reproduce the live path bit-for-bit:
#      sdpa attention everywhere, v1 ABI. Flip to hexflash/flex ONLY after
#      _rewrite_validate.sh PASSes; large-S routing kicks in above HEXFIELD_LARGE_NPAD.
export HEXFIELD_ATTN_IMPL="${HEXFIELD_ATTN_IMPL:-sdpa}"        # sdpa | hexflash | flex
export HEXFIELD_LARGE_NPAD="${HEXFIELD_LARGE_NPAD:-512}"        # cutover Npad for the new kernel
export HEXFIELD_COMPILE_MAX_NPAD="${HEXFIELD_COMPILE_MAX_NPAD:-512}"  # small-S compile band (Layer C, unchanged)
# HEXFIELD_ABI: 2 enables the Rust-owned pinned v2 pack (byte-exact vs v1).
# Default 1 keeps the deployed numpy path until B is validated.
export HEXFIELD_ABI="${HEXFIELD_ABI:-1}"                        # 1 | 2

mkdir -p "$RUNDIR" "$CKPTS"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another hexfield supervisor running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; rm -f "$LOCK"; exit 1; }
rm -f "$DONE"
trap 'rm -f "$LOCK"' EXIT

latest_ckpt(){ ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1; }

log "REWRITE SUPERVISOR start (pid=$$) ROOT=$ROOT run=$RUNDIR"
log "flags: ATTN_IMPL=$HEXFIELD_ATTN_IMPL LARGE_NPAD=$HEXFIELD_LARGE_NPAD ABI=$HEXFIELD_ABI ASYNC=$HEXFIELD_ASYNC_EVAL"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr -> halt"

declare -a crash_times=(); consec_fast=0
while :; do
  lc="$(latest_ckpt)"
  if [[ -n "$lc" ]]; then
    USE="$RUNDIR/_resume_config.toml"
    awk -v c="$lc" '/^\[checkpoint\]/{print; print "resume_from = \"" c "\""; next} {print}' "$CONFIG" > "$USE"
    log "RESUME from $(basename "$lc")"
  else
    USE="$CONFIG"; log "FIRST LAUNCH (init per config: initialize_from BC prefit)"
  fi
  stamp="$(date -u +%Y%m%d_%H%M%S)"; t0=$(date +%s)
  log "LAUNCH out=$RUNDIR/train.$stamp.out.log"
  "$PY" -u -m hexo_train.cli.train_model "$USE" >"$RUNDIR/train.$stamp.out.log" 2>&1 &
  cpid=$!; echo "$cpid" > "$RUNDIR/driver.pid"
  wait "$cpid"; code=$?; t1=$(date +%s); up=$((t1-t0))
  log "EXIT pid=$cpid code=$code uptime=${up}s"
  if (( code == 0 )); then echo "exit 0 at $(date -u +%FT%TZ)" > "$DONE"; log "DONE (exit 0)"; break; fi
  crash_times+=("$t1"); now=$(date +%s); kept=(); for ct in "${crash_times[@]}"; do (( now-ct < 3600 )) && kept+=("$ct"); done; crash_times=("${kept[@]}")
  if (( up < FAST_CRASH_SECONDS )); then consec_fast=$((consec_fast+1)); else consec_fast=0; fi
  log "breaker: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}"
  if (( consec_fast >= MAX_CONSEC_FAST || ${#crash_times[@]} > MAX_PER_HOUR )); then
    echo "halt: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}" > "$HALT"
    log "HALT: breaker tripped. Wrote $HALT. Not relaunching."; break
  fi
  log "RELAUNCH (resume from latest) in 3s"; sleep 3
done
log "REWRITE SUPERVISOR exit."
```

---

## File 5 — `scripts/_rewrite_rollback.sh`

One-command revert to the live tree. Because cutover is purely "which `$ROOT` the supervisor uses," rollback stops the rewrite supervisor and relaunches the original `_hexfield_supervise_main1.sh` (live `$ROOT`). The run resumes from the same checkpoint with the deployed code. Same kill ordering as `_restart_supervisor.sh` (supervisor first, then trainer).

```bash
#!/usr/bin/env bash
# Roll back the inference-rewrite cutover: stop the rewrite supervisor + its
# trainer, then relaunch the LIVE supervisor (live $ROOT, deployed code). Resumes
# from the same epoch checkpoint. Kill the supervisor FIRST so it can't relaunch
# the trainer (same ordering as scripts/_restart_supervisor.sh).
set -uo pipefail

LIVE="/mnt/e/Hexo-BotTrainer-hexgt"
RUNDIR="/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"

echo "BEFORE:"; pgrep -af 'rewrite_supervise_main1|hexo_train.cli.train_model .*hexfield' | grep -v pgrep || echo "(none)"

# 1. stop the rewrite supervisor (it owns supervisor.lock; killing it lets the
#    EXIT trap remove the lock). Then stop its trainer via driver.pid.
pkill -f '_rewrite_supervise_main1.sh' && echo "SIGTERM rewrite supervisor" || echo "no rewrite supervisor"
sleep 3
DPID="$(cat "$RUNDIR/driver.pid" 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$DPID" ]] && kill -0 "$DPID" 2>/dev/null; then
  kill -TERM "$DPID" 2>/dev/null; echo "SIGTERM trainer $DPID"
  for i in $(seq 1 20); do kill -0 "$DPID" 2>/dev/null || { echo "trainer exited after ${i}s"; break; }; sleep 1; done
  kill -0 "$DPID" 2>/dev/null && { kill -KILL "$DPID"; echo "SIGKILL $DPID"; }
fi
# hard-kill survivors
pkill -9 -f '_rewrite_supervise_main1.sh' 2>/dev/null
pkill -9 -f 'hexo_train.cli.train_model .*hexfield' 2>/dev/null
sleep 1
rm -f "$RUNDIR/supervisor.lock"; echo "lock removed"
# clear any breaker halt the rewrite may have tripped, so the live supervisor starts
rm -f "$RUNDIR/supervisor_halted.flag"; echo "halt flag cleared"

echo "AFTER STOP:"; pgrep -af 'rewrite_supervise_main1|hexo_train.cli.train_model .*hexfield' | grep -v pgrep || echo "ALL STOPPED"

# 2. relaunch the LIVE supervisor (deployed code, live $ROOT). nohup + detach.
echo "RELAUNCHING LIVE supervisor ($LIVE/scripts/_hexfield_supervise_main1.sh)"
nohup bash "$LIVE/scripts/_hexfield_supervise_main1.sh" >/dev/null 2>&1 &
sleep 5
echo "LIVE supervisor pid(s):"; pgrep -af '_hexfield_supervise_main1.sh' | grep -v pgrep || echo "WARN: not detected — check $RUNDIR/supervisor.log"
echo "ROLLBACK DONE — live code resumes from $(ls -1 "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | sort -V | tail -1)"
```

---

## Operator runbook — exact pause-window sequence

All steps run as the WSL user that owns the live run. The live run must already be stopped before step 0 (use the existing bounce/stop tooling; this runbook assumes the pause is in effect).

**Phase 0 — pause + preflight (GPU-free)**
1. Confirm the live run is paused. Stop the live supervisor first, then the trainer (existing pattern in `_restart_supervisor.sh` / `_bounce_trainer.sh`):
   ```bash
   pkill -f '_hexfield_supervise_main1.sh'; sleep 3
   kill -TERM "$(cat /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/driver.pid)"; sleep 5
   rm -f /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/supervisor.lock
   ```
2. `bash scripts/_rewrite_preflight.sh` — must print `PASS` (no live procs, no lock, GPU free, resume ckpt present). Records the live git SHA for rollback.

**Phase 1 — build (CPU)**
3. Operator assembles the accepted rewrite components onto branch `claude/inference-rewrite` in `E:\Hexo-BotTrainer-hexgt-rewrite` (do this before the build; the build script will create the worktree if missing but will not overwrite assembled code).
4. `bash scripts/_rewrite_build.sh` — maturin `--release` in the `hexfield-dev` venv, mirrors `_rust*.so` into the worktree source. Must print `BUILD OK`.

**Phase 2 — validate (GPU; this is the gate)**
5. `bash scripts/_rewrite_validate.sh` — Tiers 1→3 in the live torch venv against the worktree. Must print `VALIDATION PASS`.
   - Tier 1 fail → **blocker** (bias-index divergence). Do not proceed.
   - Tier 2 hexflash fail / flex pass → proceed but set `HEXFIELD_ATTN_IMPL=flex` in step 6.
   - Any Tier 3 fail → do not cut over; investigate.

**Phase 3 — cutover (GPU)**
6. Launch the rewrite supervisor. **Start conservative** (defaults reproduce the live path bit-for-bit), confirm a clean resume, then escalate the flags:
   ```bash
   # 6a. neutral cutover — sdpa everywhere, v1 ABI: proves the worktree resumes cleanly.
   nohup bash scripts/_rewrite_supervise_main1.sh >/dev/null 2>&1 &
   # watch one epoch boundary in the run log:
   tail -f /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/supervisor.log
   ```
   After one clean epoch, stop (`_rewrite_rollback.sh` then relaunch) OR — preferred — relaunch the rewrite supervisor with the validated flags exported:
   ```bash
   # 6b. enable the new large-S kernel (or 'flex' if Tier 2 selected the fallback)
   HEXFIELD_ATTN_IMPL=hexflash HEXFIELD_LARGE_NPAD=512 \
     nohup bash scripts/_rewrite_supervise_main1.sh >/dev/null 2>&1 &
   ```
   Then, only after the attention kernel is stable across an epoch, enable Layer B:
   ```bash
   # 6c. enable the v2 pinned ABI (byte-exact vs v1; Tier 3b gated it)
   HEXFIELD_ATTN_IMPL=hexflash HEXFIELD_LARGE_NPAD=512 HEXFIELD_ABI=2 \
     nohup bash scripts/_rewrite_supervise_main1.sh >/dev/null 2>&1 &
   ```
7. Spot-check throughput / VRAM during the first rewrite epoch (this is the only place the speedup + VRAM-headroom claims become real — they are not statically certain):
   ```bash
   nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
   ```
   Expect VRAM at or below the live ~3.4 GiB transient (no `(B,4,S,S)` bias materialization on the hexflash path) and large-S forward time down ~1.2–1.4x (Amdahl-capped).

**Phase 4 — rollback (any time)**
8. If anything regresses: `bash scripts/_rewrite_rollback.sh`. Stops the rewrite supervisor + trainer, clears lock/halt, relaunches the **live** supervisor against the live tree. The run resumes from the same checkpoint on deployed code. No checkpoint surgery — the rewrite never changed weights or the reply ABI, so checkpoints are interchangeable.

---

## Why this is safe / consistent (the load-bearing reasoning)

- **The live tree is never written.** Build and cutover happen entirely in the worktree. The only shared artifact is the run dir (`/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1`), and both supervisors use the identical lock/halt/resume/breaker protocol, so they cannot run concurrently (single-instance `supervisor.lock`, `_hexfield_supervise_main1.sh:46-49`) and resume from the same `epoch_*.pt`.
- **Cutover = `$ROOT` swap.** The live trainer already loads hexfield purely from `PYTHONPATH=$ROOT/packages/hexfield/python` (`_hexfield_supervise_main1.sh:36`). Pointing `$ROOT` at the worktree is the entire deployment; pointing it back is the entire rollback. No installs, no `.so` copies into the live venv (`maturin develop` targets `hexfield-dev` only, never the live `hexgt-build` — `_rebuild_hexfield.sh:4-7`).
- **Default flags = today's behavior.** `HEXFIELD_ATTN_IMPL=sdpa`, `HEXFIELD_ABI=1`, `HEXFIELD_COMPILE_MAX_NPAD=512` reproduce the deployed forward exactly, so step 6a is a pure resume-correctness check before any numeric-path change is enabled. Every escalation (6b, 6c) is independently parity-gated by `_rewrite_validate.sh`.
- **Checkpoint interchangeability.** The rewrite changes only the forward-compute kernel and serve plumbing — not weights, not the reply ABI (`parse_chunk_reply`/`finalize_priors` reused verbatim per the spec). So rollback needs no checkpoint conversion: the same `epoch_*.pt` loads on either tree.

Relevant live-tree files this runbook is consistent with (read-only references):
- `E:\Hexo-BotTrainer-hexgt\scripts\_rebuild_hexfield.sh`
- `E:\Hexo-BotTrainer-hexgt\scripts\_hexfield_supervise_main1.sh`
- `E:\Hexo-BotTrainer-hexgt\scripts\_restart_supervisor.sh`
- `E:\Hexo-BotTrainer-hexgt\scripts\_bounce_trainer.sh`
- `E:\Hexo-BotTrainer-hexgt\scripts\_check_gpu_state.sh`
- `E:\Hexo-BotTrainer-hexgt\configs\hexfield_main_1.toml`
- `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\inference.py` (env-flag conventions: `HEXFIELD_NO_COMPILE`, `HEXFIELD_COMPILE_MAX_NPAD`)
- `E:\Hexo-BotTrainer-hexgt\scripts\_hexfield_compile_overlap_test.py` (Tier-3 harness invocation contract)

**Validation honesty:** Tier 1 (bias-index oracle) is the only step certain without a GPU. Tiers 2–3, the throughput (~1.2–1.4x large-S), and the VRAM headroom are confirmed only in the live pause by running Files 3’s `_rewrite_validate.sh` and step 7. The runbook itself runs no GPU workload and writes nothing to the live tree.