# NOTES — dense_cnn Model 1 (current state)

Tight current-state memory for future sessions. Full history is in
[`NOTES_archive.md`](NOTES_archive.md) (1385 lines — the scratch_64 overnight log + the
optimization/model-change work). Keep THIS file short.

## What we're running now

The active target is **Model 1 dense_cnn at 96 channels × 6 blocks, fully-conv policy
head (P7), 512 MCTS sims** — a fresh run that replaces the old `scratch_64` (64×4, FC
policy head, 128 sims). The two architectures are checkpoint-incompatible, so this is a
fresh run, not a resume.

- **Branch:** `impl/scratch64-phase1-opt` (pushed to origin; PR not yet opened).
- **Config:** `configs/dense_cnn_model1_target_96x6.toml`
- **Run dir:** `runs/dense_cnn_model1_target_96x6/`
- **Supervisor:** `scripts/supervise_target_96x6.ps1` (copy of the scratch_64 supervisor,
  retargeted: config + `scratch_64`→`target_96x6` process-match; same guardrails).
- **Bootstrap:** fresh SealBot prefit for THIS arch via
  `scripts/bootstrap_dense_cnn_sealbot.py` →
  `runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt`, wired as the
  config's `initialize_from`. The old 64×4/FC bootstrap is shape-incompatible and is NOT
  reused — **the bootstrap must be regenerated whenever the architecture changes.**
- **Smoke config:** `configs/dense_cnn_model1_target_96x6_smoke.toml` (scaled cycle test).
- **Dashboard (fixed build):** serves ALL runs by auto-discovering `cwd/runs`; a run appears
  once it has a `diagnostics/` or `selfplay/` dir. Start from the repo root so it serves the
  worktree's fixed static (`hexo_frontend/static/`, bounded-viewer + mobile-zoom) and finds the
  runs. Bind `0.0.0.0:8080` for LAN (http://192.168.68.62:8080). One command:
  ```
  cd E:/Hexo-BotTrainer
  $env:PYTHONPATH="E:/Hexo-BotTrainer/packages/hexo_frontend/python;E:/Hexo-BotTrainer/packages/hexo_engine/python;E:/Hexo-BotTrainer/packages/hexo_runner/python;E:/Hexo-BotTrainer/packages/hexo_utils/python;E:/Hexo-BotTrainer/packages/hexo_models/python;E:/Hexo-BotTrainer/packages/hexo_train/python"
  python -m hexo_frontend.web --host 0.0.0.0 --port 8080 --sealbot-path E:/SealBot
  ```

The old `scratch_64` run is intentionally stopped at `epoch_000022.pt` and is being
**abandoned** (superseded by this fresh run). Don't resume it.

**RUN STATUS — LIVE (launched 2026-05-29 ~14:59).** Supervisor `supervise_target_96x6.ps1`
(self.pid 54612) launched trainer pid 28292; bootstrapped from `bootstrap_sealbot_prefit.pt`
(load_checkpoint=loaded, epoch 0 → starts at epoch 1), arch 96×6 P7 confirmed. Watchdog armed
(free-RAM floor 4 GB). Backstop monitor: watch `runs/dense_cnn_model1_target_96x6/diagnostics/`
(supervisor.log, events.jsonl, epoch_*.pt, supervisor_halted.flag / _completed.flag) — same
decision tree as the old scratch_64 backstop. Dashboard live at http://192.168.68.62:8080 showing
this run. (PIDs are point-in-time; re-derive from supervisor.log / process list each session.)

## Stability guardrails (carry these into every run — hard-won from scratch_64 crash-loops)

- Shuffle: `shuffle_worker_group_size = 8000`, `approx_rows_per_out_file = 8000` (bigger
  groups spiked RAM and crash-looped the epoch-16 shuffle).
- Replay window: `shuffle_keep_target_rows = 300000` (600k spiked host RAM; re-widen only
  with a watchdog check).
- Watchdog (`scripts/watch_model1_resources.ps1`, launched by `start_model1_training.ps1`):
  `MinFreeRamGb = 4`, trainer-private cap 18 GB. Config-agnostic (run-name-targeted).
- Supervisor circuit breaker: 3 consecutive crashes <180 s, OR >6 crashes/60 min, OR
  `MaxNoProgressRelaunches = 3` (no new epoch checkpoint) → halt flag.
- Eval-cache cap `mcts_session_cache_max_states = 131072` (host RAM, not VRAM).
- Supervisor injects `resume_from = <latest epoch_*.pt>` on each relaunch; first launch (no
  checkpoint) uses `initialize_from` (the bootstrap).

## Key decisions / reframings (measured — don't re-litigate)

- **Self-play MCTS is GPU-forward-bound, NOT CPU/selection-bound** within `session.run`.
  The "29% GPU duty" figure was the whole self-play *phase* (incl. Python orchestration
  outside search), not the overlappable fraction.
- **Real wins (shipped):** P0 NPZ load-once (57× data-prep, byte-identical); A7 cuDNN
  batch-bucketing (kills ~830 s cold-epoch autotune); **parse parallelization** (Rust prior
  parse ~1650 ms→75 ms, ~18% self-play, byte-identical); FP16/autocast is already on
  (2.4× forward, numerically safe at 96×6).
- **Measured NOT worth building (don't):** A1 select↔eval pipeline gave ~0 throughput
  (selection is ~2-3% of move) — KEPT only as structure + M2 cache `Arc<Mutex>`; §4.2
  shared-tree atomics (dominated by the eval `virtual_batch_size` knob, same quality
  tradeoff); root parallelism (splits sims → shallow trees → worse). A3 select-replay and
  A4 marshal are each ~2% — not worth it.
- **Single-game / eval latency lever** = single-tree `virtual_batch_size` (eval-only config
  knob shipped; default 0). Raising it ~3.4× cuts eval latency at a small quality cost;
  validate strength before trusting eval win-rates.
- **VRAM:** 96×6 + P7 train step at bs256 ≈ 32% of 12 GB — R1 risk is a non-issue;
  `calibrate=true` OOM-guard auto-falls-back regardless.
- **Goal #4:** hold our own vs SealBot best-50ms. scratch_64 plateaued at 2–6 wins/64
  (diffuse FC policy + too-few sims) — the motivation for the 96×6 + P7 + 512-sim change.

## Environment gotcha

`hexo_models` is a PEP-420 namespace whose installed copy is STALE. To test worktree code,
set `PYTHONPATH=E:/Hexo-BotTrainer/packages/hexo_models/python` (what
`start_model1_training.ps1` does). Rust: `cargo build --release --manifest-path
packages/hexo_models/Cargo.toml --features python`, then copy `target/release/hexo_models.dll`
→ `packages/hexo_models/python/hexo_models/_rust.cp314-win_amd64.pyd` (no venv for `maturin
develop`).

## LOG (supervisor backstop watcher)

New entries on top. Written for the next watcher run. NOTE: the scheduled-task prompt

### 2026-05-30 ~01:04 UTC (21:04 EDT) — **RUN ADVANCING NORMALLY (WSL+TRT live, epoch 1 selfplay in flight)** — NO ACTION

**Verdict:** the WSL+TRT run launched ~13 min ago (00:50:53Z) is **healthy and advancing**. TRT FP16
is genuinely engaged (not torch fallback), selfplay is actively writing epoch-1 shards, RAM is fine,
zero crash signatures. First clean datapoint for the 96×6 arch is now imminent. I took NO action
(advancing branch) — only logged this note.

**State found / how verified (cross-checked WSL procs + supervisor log + events + shard mtimes +
err/out logs + RAM sampler; no single signal trusted):**
- **WSL liveness (CONFIRMED live):** `pgrep` shows supervisor `supervise_target_96x6_wsl.sh` **pid
  410** (+ child 427) AND trainer **pid 446** (`hexo_train.cli.train_model ...target_96x6.toml`) at
  **311% CPU, 19.9% MEM** — top process in WSL. These are WSL procs (invisible to Get-Process); this
  is why a native `Get-Process` would show nothing. Matches the 00:51 entry's pids exactly (410/446)
  → same launch, no relaunch yet.
- **supervisor_wsl.log:** a SINGLE `LAUNCH pid=446` at 00:50:53Z, bootstrapped via `initialize_from`
  (SealBot prefit, no epoch checkpoint to resume). NO EXIT/RELAUNCH/CAPTURE/HALT/COMPLETED after it
  → clean first launch, **zero crashes** this run. (The old PS `supervisor.log` is stale — ignore it.)
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present → not halted,
  not completed. No `crashdumps/` dir.
- **TRT CONFIRMED ADOPTED (not fallback):** out.log line `[trt_backend] adopted TRT FP16 (build
  50.8s, argmax_match=0.9922, value_err=0.0078)` — the per-epoch engine built in ~51s and passed the
  quality gate (99.22% argmax match, value err 0.0078, both within the validated tolerances from the
  18:5x/19:2x bench entries). Config is fail-loud (no silent torch fallback), so a build failure
  would have crashed — it didn't. So the 2.31× TRT path is actually live this run, unlike every
  prior native-torch attempt. (The only other out.log line is a benign TRT default-stream perf
  warning.)
- **events.jsonl:** `load_checkpoint`=loaded (epoch 0, arch 96×6 P7 confirmed) → `calibrate_
  performance` finished (207s, **meets_target=false @ 10.4 pos/s** — note this is the CALIBRATION
  probe metric, the known-low full-pipeline number; NOT the live TRT selfplay rate, which the bench
  measured at ~84 pos/s) → `run_epochs` → `stage_started epoch_000001`. No `stage_finished` for
  epoch 1 yet (in progress).
- **Selfplay ADVANCING (freshest signal):** 79 epoch-1 shards, newest `epoch_000001_game_000188.npz`
  stamped **21:03:26** vs my check at **21:03:31** — i.e. written ~5 s before I looked. Game indices
  run up to ~253 (rolling replenishment, `games_per_epoch=512`, so shard count < max index is
  normal). Definitively NOT a stall.
- **No crash:** newest err.log `trainer.20260530_005053.err.log` (last write 20:54:29, startup
  warnings only) — fault-sig scan (Fatal Python error|Current thread|panicked|stack backtrace|
  Traceback|0xc0000005|access violation|STATUS_|SIGSEGV|SIGABRT) = **0 hits**. Tail = only the two
  known-benign warnings (inference.py non-writable-buffer + architecture.py TracerWarning).
- **RAM healthy:** `watch_wsl.jsonl` actively sampling (latest 01:04:09Z), **free_ram_gb ~21.8,
  flat** — nowhere near the 4 GB floor. No pressure (host protected by the 28 GB WSL2 cap).
- **No checkpoints/eval yet:** checkpoints/ holds ONLY `bootstrap_sealbot_prefit.pt` (epoch 0). No
  `epoch_*.pt`, no `dense_cnn.evaluation.*.json`. Expected — epoch 1 hasn't finished.

**Why no action:** run is up, TRT engaged, selfplay producing shards in real time, RAM fine, no flag,
no crash, no stall. Nothing to fix, nothing to relaunch (the supervisor owns relaunch anyway).

**Still open / next-step instructions for next watcher:**
1. **Re-derive liveness via WSL, NOT native procs.** Run `wsl.exe -e bash -lc "pgrep -af
   'supervise_target_96x6_wsl|train_model'"`. Expect supervisor (was pid 410) + trainer (was 446) —
   **pids are point-in-time; a different trainer pid + a new `RELAUNCH`/`LAUNCH` line in
   supervisor_wsl.log = the supervisor restarted after a crash** (then root-cause from the new
   err.log + any crash_artifacts/ before assuming healthy). A momentary process gap during a
   relaunch is NORMAL — confirm "crash" via the halt flag + supervisor_wsl.log, not a momentary
   absence.
2. **FIRST Goal-#4 milestone is imminent — watch for it:** `stage_finished epoch_000001` in
   events.jsonl + the first `checkpoints/epoch_000001.pt` + first
   `diagnostics/dense_cnn.evaluation.epoch_000001.json`. Report wins/losses/mean_turns. **Baseline to
   beat = scratch_64's 2–6 wins/64 vs SealBot best-50ms** — the 96×6 + P7 + 512-sim + TRT change
   exists to clear that plateau. This will be the FIRST-EVER eval datapoint for the 96×6 arch.
3. **TRT re-checks each epoch:** the engine rebuilds per epoch (~51s). Confirm each new epoch's
   out.log still shows `[trt_backend] adopted TRT FP16 (... argmax_match≈0.99 ...)` — if an epoch's
   build ever FAILS, fail-loud will crash the trainer (supervisor relaunches; watch the breaker).
   The 20:0x-entry blockers (unreliable in-process build, eval-rebuilds-per-game) were fixed before
   this launch (subprocess-isolated build + selfplay-only TRT per the recent commits 0fde413/320032a
   /26821ac/867a8de/bc17400), and the 50.8s clean build here confirms the fix held for epoch 1.
4. **Throughput sanity (optional):** if you want a live pos/s, compute it from epoch-1 shard mtime
   spread or wait for `epoch_000001`'s `stage_finished` elapsed_seconds. Don't trust the calibration's
   10.4 pos/s as the live rate — TRT selfplay should be materially faster (~84 pos/s bench).
5. **Decision tree unchanged:** advancing → log only (as here). Halted (flag) → root-cause + maybe
   fix + clear flag + restart supervisor. Completed (flag) → report final eval, ask re raising
   loop.epochs. Stalled (live trainer but no new shard / no events progress >25 min, no flag) →
   capture err/events tails + flag a hang.

### 2026-05-30 ~00:51 UTC (20:51 EDT) — **RUN RE-LAUNCHED (LIVE) under the WSL supervisor with TRT FP16** — user-authorized

The training run is **LIVE again** (user explicitly authorized the launch). It now runs **in WSL**
(so TensorRT FP16 engages) under a NEW supervisor: **`scripts/supervise_target_96x6_wsl.sh`**
(the Windows `supervise_target_96x6.ps1` CANNOT drive a WSL trainer — WSL procs are invisible to
Win32_Process). Same guardrails (resume-from-latest, fast-crash×3 / >6/hr / no-progress×5 breaker,
crash artifacts, RAM-floor sampler; host protected by the 28GB WSL2 cap).
- **Optimized config**: TRT FP16 ON (`inference_use_tensorrt=true`, **fail-loud, NO silent torch
  fallback** — fallback is opt-in via `inference_trt_allow_torch_fallback`/`HEXO_TRT_ALLOW_FALLBACK`),
  bucketing (mult-16), rolling replenishment (`games_per_epoch=512`). Eval is torch (selfplay-only TRT).
- **PIDs (point-in-time, WSL — re-derive!):** supervisor bash **pid 410**, trainer **pid 446**.
  Bootstrapped from the SealBot prefit (`initialize_from`, NOT random) — log confirms.
- **Logs:** `runs/dense_cnn_model1_target_96x6/diagnostics/supervisor_wsl.log` (WSL supervisor),
  `trainer.<stamp>.out/err.log`, `watch_wsl.jsonl` (RAM). The OLD `supervisor.log` (PS) is stale.
- **Backstop watcher note:** the trainer is a **WSL** process (invisible to `Get-Process`/Win32_Process)
  — check liveness via `wsl ... pgrep -af supervise_target_96x6_wsl|train_model` and the WSL supervisor
  log, NOT the native process list. Do NOT relaunch the PS supervisor. TRT builds per epoch (~46s).
- Validation done first (8/8 TRT build reliability, fail-loud unit test, 2-epoch WSL smoke adopted 2/2
  fail=0 + completed). Measured TRT end-to-end ~84 search-pos/s @256 (2.3x). Verifying live pos/s now.

### 2026-05-29 ~20:0x EDT — TRT PHASE FINISHED (launch ABORTED); GPU now IDLE; run STILL DOWN by design; NO ACTION

**Verdict:** training run still intentionally stopped (NOT a crash, NOT a stall, NOT a breaker
halt). The inference-opt / TRT phase that owned the GPU has now **FINISHED** — and unlike every
prior entry today, **the GPU is IDLE and the WSL bench agent is gone.** The bench agent's final
act was to **abort the TRT launch** (2 build blockers) and commit it; it did NOT start a training
run. User has NOT relaunched. I took **NO action** (decision-tree branch #3, deliberate stop). The
new wrinkle vs prior entries: the GPU is now free and the bench phase is done, so this is the
"down + benchmark FINISHED + user hasn't relaunched, awaiting their launch decision" case.

**State found / how verified (cross-checked flags + files + git + native procs + WSL procs + GPU;
no single signal trusted):**
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present → not a
  breaker halt, not a clean completion. Down for the deliberate (external) reason.
- **supervisor.log:** UNCHANGED — still ONLY `LAUNCH pid=28292` at 14:59:45, no
  EXIT/RELAUNCH/CAPTURE/HALT/COMPLETED after it. User has NOT relaunched. Pidfiles still hold the
  ORIGINAL launch pids (supervisor.self.pid=54612, supervisor.pid=28292) — **both DEAD** (absent
  from `Get-CimInstance Win32_Process`). STALE; re-derive every time.
- **Native procs:** NO live trainer/supervisor/watchdog. Only relevant native python = pid
  **52864** = dashboard (`hexo_frontend.web` 0.0.0.0:8080, up by design, started 14:50).
  powershell pid 19844 (created 20:04) is MY OWN tool shell — ignore (self-artifact).
- **GPU = 0% / 733 MiB / 41 °C — IDLE (CHANGE).** First time since ~15:1x the GPU is free. The
  WSL `ps` top is now only systemd housekeeping (systemd/snapd/udevd at <15% CPU) — **NO WSL
  python**. So the TRT/bench agent that held the GPU at 27–100% in the 16:0x–19:0x entries has
  EXITED. (`wsl.exe` prints a benign "131072x1 screen size is bogus" warning — cosmetic, ignore.)
- **Bench agent's FINAL state (freshest signal in the system):** HEAD of
  `bench/inference-backends-wsl` is `bc17400 WSL supervisor + pre-flight smoke: ABORT TRT launch
  (2 build blockers found)` stamped **20:02:26** — only ~1 min before this check (20:03). The two
  newest commits are `4b78b99` (19:25, NOTES: TRT validated+enabled) and `bc17400` (20:02, the
  abort). So the TRT journey is fully recorded and the agent CHOSE NOT to launch a run (see the
  ~20:0x entry directly below this one for the 2 blockers: unreliable in-process TRT build +
  eval rebuilding TRT per game). It went idle after committing — no run pending.
- **No progress, no crash:** only `bootstrap_sealbot_prefit.pt` (epoch 0, 96×6 P7, 25.6 MB) in
  checkpoints/, NO `epoch_*.pt`. events.jsonl ends `stage_started epoch_000001` (calibrate done:
  meets_target=false @ 12.8 pos/s, the known 96×6/512-sim profile). 99 epoch-1 selfplay shards,
  newest `epoch_000001_game_000015.npz` stamped **15:31** (~4.5 h cold) — NOT a stall (no live
  trainer to stall; deliberately-stopped run sitting idle). No `crashdumps/` dir. Newest err.log
  `trainer.20260529_145945.err.log` last write **15:00:05** (the stop), fault-signature scan
  (Fatal Python error|Current thread|panicked|stack backtrace|Traceback|0xc0000005|access
  violation|STATUS_|SIGSEGV|SIGABRT) = **0 hits**. No `dense_cnn.evaluation.*.json`.

**Why no action:** run is OFF by deliberate choice; the bench phase that justified the stop is now
complete and the GPU is free, but the bench agent deliberately ABORTED the launch (the optimized
WSL+TRT path has 2 unresolved blockers). Relaunching is a USER decision with a real fork (see
next steps), and the hard rule forbids me auto-relaunching a deliberate stop. No halt flag, no
crash, no stall. Nothing to fix in the training/Rust code, nothing to relaunch.

**Still open / next-step instructions for next watcher:**
1. **First re-check whether the user relaunched:** NEW `LAUNCH` line in supervisor.log dated AFTER
   14:59:45 AND a live `supervise_target_96x6.ps1` (supervisor.self.pid) + a live NATIVE python
   trainer with the config arg. If present → switch to the normal advancing/halted/stalled tree
   (flags → events.jsonl last stage → selfplay shard mtimes vs now → Get-Process on pidfiles →
   watchdog tail). All PIDs here are STALE — re-derive every time.
2. **GPU is now idle + WSL bench gone (NEW since 16:0x–19:0x):** unlike all of today's earlier
   entries, the GPU is no longer held by a WSL bench python. If you find the GPU busy again with
   NO native trainer, still check WSL (`wsl.exe -e bash -lc "ps -eo pid,pcpu,comm --sort=-pcpu |
   head"`) before concluding anything — but as of now there's no WSL python. Confirm a relaunch by
   supervisor.log + a NATIVE python with the config arg, NOT GPU% alone.
3. **THE OPEN DECISION (user's, not the watcher's) — which launch path:** the bench agent left two
   launch-ready options (do NOT auto-pick; report and wait):
   - **(a) Cheap path, native Windows, launch-ready NOW:** TRT off → torch FP16 + bucketing +
     replenishment, measured ~36–41 pos/s (already > the 32 target). No blockers. This is the
     resume command in the "RUN INTENTIONALLY STOPPED ~15:1x" entry (supervisor `-ValidateOnly`
     then detached `Start-Process` of `supervise_target_96x6.ps1`). Caveat: if the first
     post-resume shuffle errors on a truncated final `selfplay/*.npz`, delete the newest shard.
     NOTE: `dense_cnn_model1_target_96x6.toml` currently has `inference_use_tensorrt=true`, which
     **only engages under WSL** — on native Windows it silently falls back to torch FP16, so the
     cheap path "just works" natively without a config edit (you lose only the 2.31× TRT win).
   - **(b) Optimized WSL+TRT path, ~84 pos/s (2.31×), BLOCKED:** needs the 2 fixes from the ~20:0x
     entry below first — (i) make TRT selfplay-ONLY (player.py/eval → torch; eval strength ==
     TRT per the regret test) so the engine builds once/epoch not once/eval-game; (ii) reliable
     TRT build (isolate per-build state / single global logger / build-once+REFIT / subprocess).
     Then re-smoke under WSL and launch via `scripts/supervise_target_96x6_wsl.sh`.
4. **Still NO Goal-#4 datapoint for the 96×6 arch** — no `epoch_*.pt`, no eval JSON. First
   milestone once training resumes: `epoch_000001` finishing in events.jsonl + first
   `dense_cnn.evaluation.epoch_000001.json` (wins/losses/mean_turns; scratch_64 baseline to beat =
   2–6 wins/64 vs SealBot best-50ms). At ~12.8 pos/s (native torch path) epochs are SLOW — judge
   stalls by "no new selfplay shard / no events progress >25 min WHILE a trainer is live", not by
   wall-clock. The WSL+TRT path (~84 pos/s) would make epochs ~6× faster if (b) is unblocked.

### 2026-05-29 ~19:2x EDT — TRT FP16 VALIDATED + ENABLED (strength-equivalent); optimized config launch-ready

TRT FP16 is now **enabled** in `dense_cnn_model1_target_96x6.toml`
(`inference_use_tensorrt=true`, gated + torch fallback). Journey: the earlier
"NaN" was a runner stream-race (fixed), NOT fp16 overflow. FP16 beat BF16 on speed
AND fidelity. Strength validated by a low-variance paired per-decision value-regret
(tv5): **mean regret -0.002 ± 0.0035 win-prob over 400 512-sim decisions** (flips
13 TRT-better / 10 torch-better) = strength-equivalent. **Measured end-to-end
self-play @256 concurrency: baseline ~36 → +TRT+bucketing ~84 full pos/s (2.31×,
~2.6× the 32 target).** Engages under WSL only (native Windows falls back to torch
FP16). Branch `bench/inference-backends-wsl` (HEAD f8e60e4). Run still DOWN by
design; no training run started. NOTE for env: built SealBot `minimax_cpp` for
WSL (`E:/SealBot/best/minimax_cpp.cpython-312-...so`) for the strength A/B harness.

### 2026-05-29 ~20:0x EDT — LAUNCH ABORTED: WSL pre-flight smoke found 2 TRT blockers; run still DOWN (NOT launched)

Attempted to start the real run in WSL (TRT). Pre-flight WSL smoke (optimized
config, 2 epochs, TRT on, SealBot eval) COMPLETED and proved the full WSL pipeline
works (selfplay→shuffle→train→checkpoint→SealBot-eval, both epochs, run.completed)
— BUT only via torch fallback, exposing two TRT blockers, so I did NOT launch:
 1. **TRT build unreliable**: 2 of 6 in-process builds failed
    (`IOptimizationProfile::isValid Err4 MIN<=OPT<=MAX` on a valid (1,128,1024)
    profile; "logger differs from one already registered" warnings) → TRT
    global-state corruption across repeated in-process builds → TRT engages
    inconsistently (~33% silently fall back to torch → lose the 2.4x unpredictably).
 2. **Eval rebuilds TRT per game**: `player.py` builds DenseCNNInference per
    DenseCNNPlayer = per eval game; at eval games_per_epoch=64 that's ~64x42s≈45
    min/epoch of build overhead. Prohibitive.
FIX NEEDED before a TRT launch: (a) TRT selfplay-ONLY (player.py/eval → torch;
eval strength == TRT per the regret test, and eval is a benchmark not training
data) so it's built once/epoch amortized over 512 games; (b) reliable build —
isolate TRT state per build (del+gc, single global logger) or build-once+REFIT
weights per epoch, or build in a subprocess. Then re-smoke + launch.
Cheap path (TRT off: torch FP16 + bucketing + replenishment, ~36-41 pos/s > 32)
is validated + launch-ready if an immediate non-TRT start is wanted. Branch
bench/inference-backends-wsl. WSL supervisor written (scripts/supervise_target_96x6_wsl.sh).

### 2026-05-29 19:03 EDT — RUN STILL DOWN BY DESIGN; TRT bench agent ACTIVE (now BF16 cmp + SealBot A/B); NO ACTION

**Verdict:** training run intentionally stopped (NOT a crash, NOT a stall, NOT a breaker halt).
The inference-opt/TRT implementation phase is STILL running and STILL owns the GPU via WSL.
User has NOT relaunched the trainer. I took **NO action** (decision-tree branch #3, same as the
16:03/17:03/18:03 entries). Nothing to fix, nothing to relaunch.

**State found / how verified (cross-checked flags + files + native procs + WSL procs + GPU +
git; no single signal trusted):**
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present → not a
  breaker halt, not a clean completion. Down for the deliberate (external) reason.
- **supervisor.log:** UNCHANGED — still ONLY `LAUNCH pid=28292` at 14:59:45, no
  EXIT/RELAUNCH/CAPTURE/HALT after it. User has NOT relaunched; it was the clean stop, not a
  crash-loop. Pidfiles still hold the ORIGINAL launch PIDs (supervisor.pid=28292,
  supervisor.self.pid=54612) — **both confirmed DEAD via Get-Process**. STALE; re-derive.
- **Native procs:** NO live trainer/supervisor (`supervise_target_96x6.ps1`)/watchdog. Only
  relevant native python = pid **52864** = dashboard (`hexo_frontend.web` 0.0.0.0:8080, up by
  design, started 14:50). PowerShell pids 51484/44756 spawned 18:59–19:03 are MY OWN tool
  shells (self-artifact caveat) — ignore.
- **GPU = 27% / 4230 MiB / 49 °C — BUSY, but NOT the trainer.** Consumer is **WSL python PID
  401 @ 713% CPU** (confirmed via `wsl.exe ... ps`). WSL procs are invisible to
  `Win32_Process`/`Get-Process` — exactly why the GPU is busy with no native trainer. This is
  the TRT bench agent (do NOT kill). Lower GPU% than the 84–100% seen earlier = the
  comparison/A-B harness phase, not a heavy sweep.
- **Bench agent ACTIVELY iterating (freshest signal in the system):** HEAD of
  `bench/inference-backends-wsl` is `cc36471 TRT FP16-vs-BF16-vs-torch comparison + SealBot A/B
  harness` stamped **19:01:06** — only ~2.5 min before now (19:03:43). Directly above this
  entry, the ~18:5x note said "BF16 comparison running. Then pick winner + decide whether to
  enable inference_use_tensorrt" — cc36471 IS that BF16 comparison + the SealBot A/B strength
  harness (the OPEN Goal-#4 TRT re-validation). So the TRT work has advanced past 6b84075
  (the 18:46 stream-race fix). Workspace-cleanup branch also advanced (2aa245f/6d65d09 @ 18:32,
  isolated worktree — does not touch our files).
- **No progress, no crash:** only `bootstrap_sealbot_prefit.pt` (epoch 0) in checkpoints/, NO
  `epoch_*.pt`. events.jsonl ends `stage_started epoch_000001` (calibrate done: meets_target=
  false @ 12.8 pos/s, the known 96×6/512-sim profile). 99 epoch-1 selfplay shards, newest
  stamped **15:31:59** (~3.5 h cold) — NOT a stall (no live trainer to stall; deliberately-
  stopped run idle). No `crashdumps/` dir. Newest err.log (`trainer.20260529_145945.err.log`,
  last write **15:00:05**, unchanged since the stop) — prior entries scanned it clean (only the
  two benign warnings: Triton cosmetic + inference.py:214 non-writable-buffer; 0 fault-sig
  hits). No `dense_cnn.evaluation.*.json`.

**Why no action:** run is OFF by deliberate choice for the (still-active) TRT phase, which owns
the GPU via WSL and just committed 2.5 min ago. No halt flag, no crash, no stall. Relaunching
would (a) override a deliberate stop and (b) contend for the GPU with the live bench agent. Per
the hard rules I do NOT auto-relaunch.

**Still open / next-step instructions for next watcher:**
1. **First re-check whether the user relaunched:** NEW `LAUNCH` line in supervisor.log dated
   AFTER 14:59:45 AND a live `supervise_target_96x6.ps1` (supervisor.self.pid) + a live NATIVE
   python trainer with the config arg. If present → switch to the normal advancing/halted/
   stalled tree (flags → events.jsonl last stage → selfplay shard mtimes vs now → Get-Process on
   pidfiles → watchdog tail). All PIDs here are STALE — re-derive every time.
2. **Liveness gotcha (recurring, important):** a busy GPU with NO native trainer does NOT mean a
   relaunch — check WSL (`wsl.exe -e bash -lc "ps -eo pid,pcpu,comm --sort=-pcpu | head"`). The
   TRT bench agent runs python IN WSL, invisible to Win32_Process. Confirm "trainer relaunched"
   by supervisor.log + a NATIVE python with the config arg, NOT GPU% alone. Also cross-check the
   `bench/inference-backends-wsl` HEAD commit time — a very-recent commit = bench agent still live.
3. **If still down + bench agent active (as now):** log a note, take NO action, do NOT relaunch
   (deliberate stop + GPU contention). Exact resume command is in the "RUN INTENTIONALLY STOPPED
   ~15:1x" entry below (supervisor `-ValidateOnly` then detached `Start-Process`). Caveat: if the
   first post-resume shuffle errors on a truncated final `selfplay/*.npz`, delete the newest shard.
4. **OPEN — TRT FP16 quality gate / Goal-#4 re-validation:** cc36471 added a "SealBot A/B harness"
   — likely the long-OPEN TRT-vs-torch strength re-validation over 512 sims (per-forward gate !=
   search-outcome equivalence; 18:0x noted ~3% per-leaf top-1 flips, 18:5x noted 93.75% search
   move-agreement / 6.25% flip on TRT fp16). Check for a NEW validation artifact / NOTES entry +
   the final TRT-on/off + gate-threshold decision before the TRT flag is trusted for real data.
5. **Still NO Goal-#4 datapoint for the 96×6 arch** — no `epoch_*.pt`, no eval JSON. First
   milestone once training resumes: `epoch_000001` finishing in events.jsonl + first
   `dense_cnn.evaluation.epoch_000001.json` (wins/losses/mean_turns; scratch_64 baseline to beat
   = 2–6 wins/64 vs SealBot best-50ms). At ~12.8 pos/s epochs are SLOW — judge stalls by "no new
   selfplay shard / no events progress >25 min WHILE a trainer is live", not by wall-clock.

### 2026-05-29 ~18:5x EDT — TRT NaN ROOT-CAUSED + FIXED (was a runner stream race, NOT fp16 overflow)

The earlier "TRT FP16 NaN" was a **stream-ordering race in my TRT runner** (input
copied on the default stream; TRT enqueued on a separate stream w/o cross-stream
sync → read-before-ready → garbage/NaN), NOT an fp16 overflow (pinpoint: pure-fp16
torch max activation ~47 vs fp16 max 65504, no NaN). Fixed: run copy+enqueue+read
on one stream + zeroed output buffers (committed 6b84075). With the fix, **FP16 TRT
is NaN-free over 80×512-sim searches**: forward **2.35–2.67× over torch FP16**
(bs128 7043→16567, bs256 6440→17195 fwd/s), move-agreement vs torch **93.75%**
(6.25% flip; torch-vs-torch is 100% deterministic here so the flip is purely TRT
fp16 numerics), value decoded-err ~5e-5. BF16 comparison running. Then pick winner
+ decide whether to enable `inference_use_tensorrt`. Run still DOWN by design.

### 2026-05-29 ~18:3x EDT — VALIDATION PHASE (bench/inference-backends-wsl); games_per_epoch=512; TRT strength test running

Config now `games_per_epoch = 512` (middle ground, was 1024). Validating before any
real run. TRT FP16 correctness so far: decoded-value err ~4.6e-5; per-forward
policy-argmax match on REAL positions = **96.9%** (3% of leaves flip top-1, fp16
logit err ~0.04). The per-forward build gate (thresh 0.99) currently makes TRT
FALL BACK to torch in production — SAFE but no speedup. Running the search-OUTCOME
move-agreement test (512-sim searches, TRT vs torch, forced-on) to decide whether
3% per-leaf flips change the chosen move; the gate threshold + TRT-on/off decision
follow it. Supervisor finding: NO wall-clock breaker (`proc.WaitForExit`;
no-progress guard counts relaunches, not time) → a longer epoch can't false-trip
it on time. GPU intermittently busy (WSL bench); training run still DOWN by design.

### 2026-05-29 EDT — WORKSPACE-CLEANUP AGENT ACTIVE (separate worktree, branch `chore/workspace-cleanup`) — does NOT touch your files

A workspace cleanup / code-quality agent is running in an **isolated worktree** at
`E:/Hexo-BotTrainer-cleanup` on branch **`chore/workspace-cleanup`** (branched off the latest
committed state `033df19`). It is removing genuinely-dead code, stale docs, obsolete one-off
scripts, and unused tests — test-gated (full suite green before/after). It will **NOT** touch
the active validation/optimization paths: `configs/dense_cnn_model1_target_96x6*.toml`, the
supervisor/watch/start scripts, the inference/evaluator/self-play hot path
(`dense_cnn .../inference.py|trt_backend.py|selfplay.py|player.py|config.py` + Rust MCTS/engine),
or the live `analysis/throughput_understanding/*` + `analysis/inference_backends/*` + `tu*.py`
benchmark scripts. It does NOT run the trainer or touch the GPU. Reconcile/merge its branch
**after** the validation work lands — it is intentionally not merged into the active branches.

### 2026-05-29 18:03 EDT — RUN STILL DOWN BY DESIGN; WSL bench/TRT agent ACTIVE on the GPU; NO ACTION

**Verdict:** training run intentionally stopped (NOT a crash, NOT a stall, NOT a breaker
halt). The inference-opt / TRT implementation phase is STILL running — now in WSL — and owns
the GPU. The user has NOT relaunched the trainer. I took **NO action** (decision-tree branch
#3, same as the 17:03 and 16:03 entries). Nothing to fix, nothing to relaunch.

**State found / how verified (cross-checked flags + files + native procs + WSL procs + GPU;
no single signal trusted):**
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present → not a
  breaker halt, not a clean all-epochs completion. Down for the external (deliberate) reason.
- **supervisor.log:** UNCHANGED — still ONLY `LAUNCH pid=28292` at 14:59:45, no
  EXIT/RELAUNCH/CAPTURE/HALT after it. So (a) user has NOT relaunched, (b) it was the clean
  stop, not a crash-loop. Pidfiles still hold the original launch PIDs (supervisor.self.pid=
  54612, supervisor.pid=28292) — **both DEAD** (not in the live process list). STALE; re-derive.
- **Native procs:** NO live trainer/supervisor/watchdog. Only relevant native python = pid
  **52864** = dashboard (`hexo_frontend.web` on 0.0.0.0:8080, up by design). PowerShell pids
  43016/47392 spawned 18:02–18:03 are MY OWN tool shells (self-artifact caveat) — ignore.
- **GPU = 84% / 2067 MiB / 67 °C — BUSY, but NOT the trainer.** Consumer is **WSL python
  PID 306 at 403% CPU** (confirmed via `wsl.exe ... ps`). WSL procs do NOT appear in
  `Win32_Process`, which is exactly why the GPU is busy with no native trainer visible. This
  is the inference-opt bench agent (the 17:1x heads-up predicted intermittent WSL GPU usage;
  do NOT kill it). nvidia-smi compute-apps is the usual WDDM `[N/A]`-memory noise — ignore.
- **Bench agent is ACTIVELY iterating on TRT:** newest commits on `bench/inference-backends-wsl`
  are `4fb8c6b Fix TRT ONNX export: force NCHW (contiguous) layout` and `a01d8d6 Adopt
  quality-safe inference combo` — `4fb8c6b` is NEWER than the 18:0x NOTES entry (which cited
  a01d8d6), so the TRT work has advanced since. `analysis/_results_*.json` top out at ~16:23
  (trt/verify/callback_attr), but the live WSL python at 403% CPU = work in flight that may
  not write those JSONs (likely TRT engine build / the OPEN SealBot-strength re-validation).
- **No progress, no crash:** only `bootstrap_sealbot_prefit.pt` (epoch 0) in checkpoints/, NO
  `epoch_*.pt`. 99 epoch-1 selfplay shards, newest stamped **15:31:59** (~2.5 h cold) — NOT a
  stall (no live trainer to stall; deliberately-stopped run sitting idle). No `crashdumps/`
  dir. Newest err.log (`trainer.20260529_145945.err.log`, last write 15:00:05) tail = ONLY the
  two known-benign warnings (Triton cosmetic "Failed to find CUDA"; inference.py:214
  non-writable-buffer). Fault-signature scan (Fatal Python error|panicked|stack backtrace|
  Traceback|0xc0000005|access violation|STATUS_) = **0 hits**. No `dense_cnn.evaluation.*.json`.

**Why no action:** run is OFF by deliberate choice for the (still-active) inference-opt/TRT
phase, which currently owns the GPU via WSL. No halt flag, no crash, no stall. Relaunching now
would (a) override a deliberate stop and (b) contend for the GPU with the live WSL bench agent.
Per the hard rules I do NOT auto-relaunch.

**Still open / next-step instructions for next watcher:**
1. **First re-check whether the user relaunched:** NEW `LAUNCH` line in supervisor.log dated
   AFTER 14:59:45 AND a live `supervise_target_96x6.ps1` (supervisor.self.pid) + a live native
   python trainer. If present → switch to the normal advancing/halted/stalled tree (flags →
   events.jsonl last stage → selfplay shard mtimes vs now → Get-Process pidfiles → watchdog
   tail). All PIDs here are STALE — re-derive every time.
2. **Liveness gotcha (NEW, important):** a busy GPU with NO native trainer proc does NOT mean a
   relaunch — check WSL (`wsl.exe -e bash -lc "ps -eo pid,pcpu,comm --sort=-pcpu | head"`). The
   bench/TRT agent runs python IN WSL, invisible to `Win32_Process`/`Get-Process`. Confirm
   "trainer relaunched" by supervisor.log + a NATIVE python with the config arg, not GPU% alone.
3. **If still down + bench agent active (as now):** log a note, take NO action, do NOT
   relaunch (deliberate stop + GPU contention). Exact resume command is in the "RUN
   INTENTIONALLY STOPPED ~15:1x" entry below (supervisor `-ValidateOnly` then detached
   `Start-Process`). Caveat: if the first post-resume shuffle errors on a truncated final
   `selfplay/*.npz`, delete the newest shard and relaunch.
4. **OPEN — TRT FP16 quality gate (from the 18:0x entry):** before the TRT flag is trusted for
   real training data it needs a SealBot best-50ms strength re-validation over 512 sims (logit
   error can compound; per-forward gate != search-outcome equivalence). The live WSL work may
   be exactly this — check for a new validation artifact / NOTES entry from the bench agent.
5. **Still NO Goal-#4 datapoint for the 96×6 arch** — no `epoch_*.pt`, no eval JSON. First
   milestone once training resumes: `epoch_000001` finishing in events.jsonl + first
   `dense_cnn.evaluation.epoch_000001.json` (wins/losses/mean_turns; scratch_64 baseline to
   beat = 2–6 wins/64 vs SealBot best-50ms). At ~12.8 pos/s (calibration meets_target=false,
   heavy 96×6/512-sim profile) epochs are SLOW — judge stalls by "no new selfplay shard / no
   events progress in >25 min WHILE a trainer is live", not by wall-clock.

### 2026-05-29 ~18:0x EDT — ADOPTED quality-safe inference combo into target_96x6 config (launch-ready; NOT yet launched)

Config `dense_cnn_model1_target_96x6.toml` now defaults to the chosen combo
(committed on branch `bench/inference-backends-wsl`, a01d8d6):
- `[selfplay] games_per_epoch = 1024` (was 256) — rolling replenishment keeps the
  256-game pool full; epoch ~4x longer (4x samples), tail cut to the final cohort.
  **Confirm the supervisor no-progress window tolerates the longer epoch before a
  long run.**
- `[model.config.performance] inference_use_tensorrt = true` — TRT FP16, gated +
  torch fallback. **Only engages under WSL** (no native py3.14 TRT wheel); native
  Windows falls back to torch (verified). So the ~2.4x forward needs a WSL launch.
- `[model.config.performance] inference_bucket_pad_multiple = 16` — equivalence-
  preserving padding fix.
EXCLUDED raising virtual_batch_size (search-quality cost). Did NOT launch a run.
Measured pos/s @256 concurrency: baseline ~37-39 search (~35-36 full) — already
>32; +bucketing ~41 (~38). TRT self-play row measuring now.
**OPEN: TRT FP16 needs a SealBot best-50ms strength re-validation over 512 sims
before trusting it for real training data (logit error can compound; per-forward
gate != search-outcome equivalence). Recommend gating the TRT flag on that check.**

### 2026-05-29 ~17:4x EDT — WSL Rust .so rebuilt (was stale) for the impl phase — note for env

Rebuilt `hexo_models` Linux extension for WSL (the existing
`packages/hexo_models/python/hexo_models/_rust.cpython-312-x86_64-linux-gnu.so` was
STALE — old `Model1MctsSession.search()` signature, 16 args vs the current 17, so
WSL self-play errored). Rebuilt from current source via
`CARGO_TARGET_DIR=/root/hexobuild cargo build --release --manifest-path
packages/hexo_models/Cargo.toml --features python` and copied over the worktree .so.
Native `_rust.cp314-win_amd64.pyd` is UNTOUCHED (Windows self-play / dashboard
unaffected). This only affects WSL imports. (hexo_engine/hexo_utils WSL .so were
compatible — only hexo_models needed rebuild.)

### 2026-05-29 ~17:1x EDT — GPU INTENTIONALLY BUSY AGAIN (throughput-understanding + impl phase) — NOT a relaunch; NO ACTION

**Heads-up for the backstop watcher:** the training run is STILL down by design. The GPU
being busy now is NOT the training run relaunching — it is a follow-up
**throughput-understanding + inference-optimization implementation** phase (a coding agent on
branch `bench/inference-backends-wsl`). Expect intermittent `C:\Python314\python.exe analysis\...`
and WSL `python` GPU usage (self-play probes, batch sweeps, TRT). Do NOT relaunch the trainer;
do NOT kill these analysis processes. supervisor.log will show NO new LAUNCH (still the 14:59:45
one). Decision tree unchanged: down until the user re-launches. Work: GPU-occupancy / batch-tail
study + bucketing fix + TensorRT-in-self-play + 4-config pos/s table. Dashboard stays up.

### 2026-05-29 ~17:03 EDT — RUN STILL DOWN BY DESIGN; benchmark cycle FINISHED; awaiting user re-launch — NO ACTION

**Verdict:** training run intentionally stopped (NOT a crash); the inference-opt benchmark
cycle has now FINISHED and the GPU is idle; the user has NOT yet relaunched. No halt flag,
no crash, no stall. I took **NO action** — per the standing decision, the deliberate stop is
the user's to undo. This is decision-tree branch #3 from the 16:03 entry.

**State found / how verified (cross-checked files + processes + GPU; no single signal trusted):**
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present → not a
  breaker halt, not a clean all-epochs completion. Down for the external (deliberate) reason.
- **supervisor.log:** UNCHANGED — still only `LAUNCH pid=28292` at 14:59:45, with NO new
  LAUNCH/EXIT/RELAUNCH/CAPTURE/HALT line after it. So (a) the user has NOT relaunched (no new
  LAUNCH dated after 14:59:45), and (b) it was the clean stop, not a crash-loop.
- **Processes:** NO live trainer/supervisor/watchdog. Only relevant live proc = pid **52864** =
  dashboard (`C:\Python314\python.exe -m hexo_frontend.web --host 0.0.0.0 --port 8080`,
  confirmed via cmdline), left up by design. CAUTION/self-artifact: a `Get-CimInstance ... -match
  'supervise_target|...'` filter will match my OWN tool shell (the regex words are in its command
  line) — pid 32952 started at the current minute was exactly that, NOT a supervisor. Verify any
  "supervisor" hit's StartTime (~14:59) and cmdline before believing it.
- **Benchmark cycle FINISHED:** no `analysis\*.py` python proc alive; GPU **15% / 1233 MiB / 48 °C**
  = idle (dashboard polling only) — contrast the 100% / 4.7 GB it showed mid-cycle at 16:03. All
  `analysis/_results_*.json` present + stable; full report committed at
  `analysis/inference_backend_benchmarks.md` (see the 16:25 entry below for headline results:
  keep FP16, reject BF16, bucketing fix is the free win, TensorRT FP16 = 2.4–2.7× max win in WSL).
  (nvidia-smi compute-apps listed many pids with `[N/A]` memory — that's the Windows/WDDM
  per-process-attribution quirk, not real GPU consumers; util 15% is the truth.)
- **No progress, no crash:** only `bootstrap_sealbot_prefit.pt` (epoch 0) in `checkpoints/`, NO
  `epoch_*.pt`. events.jsonl ends at `stage_started epoch_000001` (never finished). Newest selfplay
  shard `epoch_000001_game_000015.npz` stamped **15:31:59**; now 17:03 (~1.5 h cold). That is NOT a
  stall — there is no live trainer to stall; it's the deliberately-stopped run sitting idle. 99
  epoch-1 selfplay shards on disk. No `crashdumps/` dir. No `dense_cnn.evaluation.epoch_*.json`.
  newest err.log unchanged (only the two benign warnings; no Fatal/Traceback/panic/0xc0000005).

**Why no action:** the run is OFF by deliberate choice for the (now-complete) inference-opt cycle.
No halt flag, no crash, no stall-to-investigate, and the GPU is free. Relaunching is the user's
call (the stop was intentional), so per the hard rules I do NOT auto-relaunch.

**Still open / next-step instructions for next watcher:**
1. **First re-check whether the user relaunched:** look for a NEW `LAUNCH` line in supervisor.log
   dated AFTER 14:59:45 AND a live `supervise_target_96x6.ps1` (supervisor.self.pid) + a live
   python trainer. If present → switch to the normal advancing/halted/stalled tree (flags →
   events.jsonl last stage → selfplay shard mtimes vs now → Get-Process on the pidfiles →
   watchdog tail). All PIDs in this log are STALE — re-derive every time.
2. **If still down + benchmark done (as now) and user hasn't relaunched:** same as this entry —
   log a note, take NO action, do NOT auto-relaunch. The exact resume command (supervisor
   `-ValidateOnly` then detached `Start-Process`) is in the "RUN INTENTIONALLY STOPPED ~15:1x"
   entry below. Caveat from that entry: if the first post-resume shuffle errors on a truncated
   final `selfplay/*.npz`, delete the newest shard and relaunch.
3. **Still NO Goal-#4 datapoint for the 96×6 arch** — no `epoch_*.pt`, no eval JSON. The first
   milestone to watch once training resumes: `epoch_000001` finishing in events.jsonl + the first
   `dense_cnn.evaluation.epoch_000001.json` (report wins/losses/mean_turns; scratch_64 baseline to
   beat = 2–6 wins/64 vs SealBot best-50ms). At ~12.8 pos/s (calibration meets_target=false, the
   known heavy 96×6/512-sim profile) epochs are SLOW — judge stalls by "no new selfplay shard /
   no events progress in >25 min WHILE a trainer is live", not by wall-clock expectation.
4. Self-artifact reminder (see Processes above): don't mistake your own PowerShell/CIM query shell
   for a supervisor; check StartTime ≈ 14:59 and the real `-File ...supervise_target_96x6.ps1` arg.

### 2026-05-29 ~16:25 EDT — INFERENCE-BACKEND BENCHMARK CYCLE COMPLETE (results below)

**This is the benchmark cycle the run was stopped for. Run still down BY DESIGN — do
not relaunch; that is the user's call.** GPU returned to clean idle (0% / ~660 MiB =
dashboard only), all benchmark processes exited, large temp artifacts (ONNX/caches)
deleted. Dashboard still up on :8080.

Full report: [`analysis/inference_backend_benchmarks.md`](analysis/inference_backend_benchmarks.md)
(+ scripts `analysis/01..09_*.py`, raw `analysis/_results_*.json`). Headlines, all
measured at the **real** production batch (MCTS leaf batch is mean≈99 / p50 70 →
bucket **128**, NOT 1024 — bs1024 is a near-dead bucket here), verified two ways
(CUDA-event + wall-clock, fresh process, <0.31% agreement):

- **BF16: reject.** ~5% slower than FP16 on Ada (shared tensor-core throughput) AND
  ~9× worse numerically (decoded-value err 0.079 vs FP16 0.009). Both autocast + TRT.
- **FP16/AMP (current production): keep as default.** bs128 7023 fwd/s, decoded-value
  err 0.009, 2.5× over FP32. Correct + already shipped.
- **torch.compile FP16 (WSL): 1.36× (bs128) / 1.44× (bs256), correctness PASS**, +7×
  lower single-eval latency (cudagraphs). BUT does NOT run on native-Windows torch
  2.10 (Triton/Inductor) — WSL only. Moderate effort.
- **TensorRT 11 FP16 (WSL): 2.39× (bs128) / 2.66× (bs256), correctness PASS** (decoded-
  value err 0.011). Biggest forward win. Needs strongly-typed ONNX (TRT11 dropped the
  FP16 builder flag); per-epoch engine rebuild (~44 s) / refit; engine is platform+
  version-specific (won't load on native Windows as-is). Highest integration cost.
- **Attribution (measured):** evaluator callback = 78% of search wall; callback is
  ~90% forward-compute at bs128/256. ⇒ ~70% of search wall is speedup-able. **Est.**
  end-to-end search: FP16 58.8 pos/s → compile ~73 → TRT ~100.
- **Highest-ROI, zero-risk lever (separate from backend):** the evaluator pads p50=70
  → bucket 128, so ~23–45% of every forward is zeros. Tighter buckets recover much of
  that with NO dtype change and NO correctness risk. Stacks with any backend.
- **Premise note:** warm search-only is **58.8 pos/s** here, vs the live calibration's
  12.8 / "1287 fwd/s" — that figure was cold-clock and/or full-pipeline (sample-finalize
  + NPZ write, which this probe omits). The forward-opt decision rests on the
  microbenchmarks + attribution, which are solid regardless.

**Recommendation:** keep FP16, reject BF16; do the bucketing fix first (free); adopt
TensorRT FP16 for the max win IF willing to pay integration + re-validate SealBot
strength; torch.compile is the lower-effort middle option but only viable if self-play
moves to WSL. **Env side effect:** the WSL smoke venv (`/root/.venvs/hexo-bottrainer-wsl`)
now has `onnx`, `onnxscript`, `tensorrt` 11.0 added (for the compile/TRT legs) — harmless,
remove if undesired.

### 2026-05-29 16:03 EDT — RUN STILL DOWN BY DESIGN (benchmark cycle ACTIVE); NO ACTION

**Verdict:** training run intentionally stopped (NOT a crash), benchmark cycle is actively
running on the GPU. I took **NO action** — do not relaunch per the standing decision below.
Backstop hard rule held: the run is down until the user re-launches after the inference-opt
benchmark cycle.

**State found / how verified (cross-checked files + processes + GPU, didn't trust one signal):**
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present — so
  NOT a circuit-breaker halt and NOT a clean all-epochs completion. (Down for an external
  reason = the deliberate stop.)
- **Processes:** trainer pid 28292, supervisor pid 54612, watchdog pid 24104 ALL DEAD
  (Get-Process). supervisor.log shows a SINGLE `LAUNCH pid=28292` at 14:59:45 and NO
  EXIT/RELAUNCH/CAPTURE/HALT line after it — i.e. the supervisor itself was killed (didn't
  relaunch), consistent with the clean stop-order in the prior entry, NOT a crash-loop.
- **events.jsonl:** last lines are `calibrate_performance` finished (177 s, meets_target=false
  @ 12.8 pos/s — the known heavy 96×6/512-sim profile) → `run_epochs` → `stage_started
  epoch_000001`. Epoch 1 never wrote a `stage_finished`; no `epoch_*.pt` exists (only the
  bootstrap prefit). So at relaunch the run restarts epoch 1 from `bootstrap_sealbot_prefit.pt`
  (epoch 0), exactly as the prior entry's RESUME POINT says. No trained progress to lose.
- **Crash check (clean):** newest trainer err.log (`trainer.20260529_145945.err.log`) tail =
  ONLY the two known-benign warnings (Triton "Failed to find CUDA" cosmetic; inference.py:214
  non-writable-buffer). NO Fatal Python error / Traceback / panicked / backtrace / 0xc0000005.
  No `crashdumps/` dir. The trainer ended by deliberate kill, not a fault.
- **Timestamp reconciliation (resolved a seeming contradiction):** the prior "RUN
  INTENTIONALLY STOPPED ~15:1x" entry's time is approximate — selfplay shards are actually
  stamped up to **15:31** (`epoch_000001_game_000235.npz` etc.). So the trainer ran epoch-1
  selfplay from ~15:03 until it was killed ~15:31. Not a relaunch, not a second run — just a
  looser timestamp in the note. Newest selfplay shard 15:31, now 16:03.
- **GPU / live python (the benchmark cycle):** GPU **100% util, 4783/12282 MiB, 63 °C** —
  actively busy. Two live python procs: pid **38756** = `analysis\06_native_batchsweep.py`
  (the GPU consumer; appears in nvidia-smi compute-apps), pid **52864** = the dashboard
  (`hexo_frontend.web` on 0.0.0.0:8080, left up by design). The benchmark agent is producing
  results: `analysis/_results_baseline.json` (15:53), `_results_bf16.json` (15:54),
  `_results_attribution.json` (15:57), `_results_selfplay_attribution.json` (15:58), and
  `06_native_batchsweep.py` running now. So the GPU is OWNED by the benchmark cycle — another
  reason not to relaunch the trainer (would contend for the GPU and corrupt both).

**Why no action:** this watcher backstops the *training* run. That run is intentionally OFF
for the inference-opt benchmark cycle (BF16 / native batch sweep / torch.compile / TensorRT),
and the cycle is mid-flight. No halt flag, no crash, no stall-to-investigate. Nothing to fix,
nothing to relaunch.

**Still open / next-step instructions for next watcher:**
1. **First check whether the user has re-launched.** Re-read supervisor.log for a NEW
   `LAUNCH` line dated AFTER 14:59:45 and check for a live `supervise_target_96x6.ps1`
   (supervisor.self.pid) + a live python trainer. If those exist → switch back to the normal
   "advancing / halted / stalled" decision tree (flags → events.jsonl last stage → selfplay
   shard mtimes vs now → Get-Process on the pidfiles → watchdog tail). PIDs above (28292/
   54612/24104) are STALE — re-derive.
2. **If still down + benchmark still running** (GPU busy, `analysis\0*.py` python proc alive,
   `analysis/_results_*.json` mtimes advancing): same as now — log a progress note, take NO
   action, do NOT relaunch (GPU contention).
3. **If down + benchmark FINISHED** (no `analysis` python proc, GPU idle, all `_results_*.json`
   present and stable) and the user has NOT relaunched: the run is simply awaiting the user's
   re-launch decision. Do NOT auto-relaunch (the stop was deliberate, not a crash). The exact
   resume command is in the "RUN INTENTIONALLY STOPPED" entry below (supervisor `-ValidateOnly`
   then detached `Start-Process`). Caveat from that entry: a truncated final selfplay shard may
   break the first post-resume shuffle — delete the newest `selfplay/*.npz` and relaunch if so.
4. There is STILL no `epoch_*.pt` and no `dense_cnn.evaluation.epoch_*.json` — so no Goal-#4
   datapoint yet for the 96×6 arch. The first one to watch for, once training resumes, is
   `epoch_000001` finishing + `dense_cnn.evaluation.epoch_000001.json` (report wins/losses/
   mean_turns; scratch_64 baseline to beat = 2–6 wins/64 vs SealBot best-50ms).

### 2026-05-29 — INFERENCE-BACKEND BENCH HARNESS + BF16 variant (bench agent, BUILD + smoke only)

Built the SHARED, reusable benchmark + correctness scaffolding for the inference-opt
cycle, plus the BF16 variant. Branch `bench/inference-backends-bf16` (pushed to origin;
commit 6032321). Files (committed, nothing else touched):
- `analysis/inference_backends/bench_harness.py` — model load (96x6+P7 from
  `bootstrap_sealbot_prefit.pt`, built via `configs/dense_cnn_model1_target_96x6.toml`,
  `model_state` loaded STRICT), `make_inputs` (REPRESENTATIVE inputs — not zeros — at
  production shapes bs=1 and bs=1024, channels_last), pluggable `Variant` (`fp32`
  reference / `fp16` AMP = current production / `bf16` AMP = this deliverable; all run on
  the production folded inference clone so only the autocast dtype varies), `time_variant`
  (full warmup = cuDNN autotune + clock ramp, then many iters reporting mean/stdev/p50/p95
  + fwd/s + pos/s), `compare_to_reference` (max-abs-error of policy logits AND value logits
  vs FP32, plus policy-argmax match fraction).
- `analysis/inference_backends/__init__.py`.

Invoke (production PYTHONPATH required — see start_model1_training.ps1):
  `python -m analysis.inference_backends.bench_harness --smoke`   (light; default)
  `python -m analysis.inference_backends.bench_harness --full`    (full sweep — verification agent)
  `... --variant bf16 --batches 1 1024 --iters 200`               (single variant)

SMOKE (bs=8, verified by me): model loads (2.122M params, epoch 0), fp32/fp16/bf16 forwards
all OK. Parity vs FP32: FP16 policy_max_abs_err=6.52e-02 / value=2.76e-01;
BF16 policy_max_abs_err=4.62e-01 / value=1.07e+00; BOTH policy_argmax_match=100%. (BF16's
larger raw error is expected — fewer mantissa bits than FP16 — but top move unchanged.)

LEFT FOR THE VERIFICATION AGENT (NOT run, to avoid GPU contention): the full timing sweep
(`--full`) at production shapes bs=1 and bs=1024 with long warmup/many iters, and parity at
those real shapes (bs=8 was smoke-only; rerun correctness at bs=1024). torch.compile/TensorRT
variants can be added by registering another `Variant` in VARIANTS. Dashboard left UP.

still describes the *old* scratch_64 run (supervise_scratch64.ps1, etc.). We have since
moved to the fresh **target_96x6** run — apply the same decision tree to
`runs/dense_cnn_model1_target_96x6/` and `scripts/supervise_target_96x6.ps1`. ORIENTATION
= the "What we're running now" + "Stability guardrails" sections at the top of this file.

### 2026-05-29 ~15:1x EDT — RUN INTENTIONALLY STOPPED (clean) for the inference-opt benchmark cycle

**NOT a crash — deliberate clean shutdown** so the GPU is free for a BF16 / torch.compile /
TensorRT inference-optimization benchmark cycle. Backstop: do NOT relaunch; the run is
down BY DESIGN until the benchmark cycle finishes and the user re-launches.

**Stop order (clean):** supervisor (`supervise_target_96x6.ps1`, pid 54612) killed FIRST so it
could not relaunch, THEN trainer (pid 28292), THEN watchdog (`watch_model1_resources.ps1`,
pid 24104). Verified all three = 0 alive (PIDs gone), supervisor.log shows NO new LAUNCH after
the original 14:59:45, GPU = 0% / 570 MiB (VRAM released). Dashboard left UP (http://192.168.68.62:8080).

**RESUME POINT — latest saved 96×6 checkpoint:** the run was stopped mid-epoch-1 (in selfplay),
so NO `epoch_*.pt` was written yet. The only saved 96×6 checkpoint is the SealBot prefit:
`runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt` (**epoch 0**, 96×6 + P7,
2.12 M params, 25.6 MB). Resuming therefore restarts at epoch 1 from the bootstrap (no trained
progress lost — epoch 1 never reached the train stage; its selfplay shards persist under
`selfplay/` and feed the next shuffle).

**EXACT RESUME COMMAND** (relaunch the supervisor; with no `epoch_*.pt` it uses `initialize_from`
= the bootstrap, exactly like the first launch — validate first):
```powershell
# sanity check (no side effects):
powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\Hexo-BotTrainer\scripts\supervise_target_96x6.ps1 -ValidateOnly
# launch (detached):
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','E:\Hexo-BotTrainer\scripts\supervise_target_96x6.ps1' -WindowStyle Hidden
```
(Caveat: a shard from the game in flight at kill time MAY be truncated; if the first post-resume
shuffle errors on a bad `selfplay/*.npz`, delete the newest one and relaunch.)

**For the benchmark agent — a real 96×6+P7 model to test on:**
- Prefit / current resume checkpoint: `runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt`
- (Same file is the bootstrap prefit — there is no separate later checkpoint yet.)
- Build the model with `configs/dense_cnn_model1_target_96x6.toml` (channels 96, blocks 6, P7) and
  load `model_state` strict. Steady self-play was ~12.8 pos/s (evaluator/forward-bound, ~1287
  forwards/s cap) — that forward is the BF16/compile/TensorRT target.

### 2026-05-29 15:03 EDT — ADVANCING NORMALLY (fresh run, epoch 1 in progress)

**Verdict:** healthy, no action taken. The 96×6/P7/512-sim run just started and is
mid-epoch-1. Nothing to fix, no flags, no restart.

**State found / how verified (cross-checked 3 ways, didn't trust any single signal):**
- **Flags:** neither `supervisor_halted.flag` nor `supervisor_completed.flag` present.
- **Supervisor.log:** clean start at 14:59:45 (pid=54612), epochs=60, breaker armed,
  "no checkpoint found; first launch will use initialize_from" → `LAUNCH pid=28292`.
  No EXIT/RELAUNCH/CAPTURE/HALT lines. This is the *first* launch — opening-diversity
  eval fix + epochs=60 are baked into THIS config from the start (no relaunch needed to
  activate them, unlike the scratch_64 history).
- **events.jsonl:** initialize_run ✓ → load_checkpoint ✓ (loaded bootstrap
  `bootstrap_sealbot_prefit.pt`, epoch 0, arch channels=96/blocks=6/policy=fully_conv_P7,
  prefit 8 epochs losses 6.01→1.63) → calibrate_performance ✓ (177 s) → run_epochs →
  `stage_started epoch_000001` (last line, no finish yet).
- **Liveness (3 signals agree):** (1) `Get-Process` confirms BOTH pid 54612 (powershell
  supervisor) and pid 28292 (python trainer) ALIVE; (2) watchdog jsonl updating in real
  time (last sample 19:03:39Z == wall-clock "now" 15:03:39 EDT), trainer cpu_seconds
  climbing 3076→3183→3291 across samples; (3) GPU 73–75% util, 65 °C, used 2.5/12 GB.
  Not a momentary relaunch gap — genuinely working.
- **Single supervisor:** `supervisor.self.pid` = 54612 = the live supervisor. No duplicate.
- **Resources OK:** watchdog status "ok", critical=[], free RAM ~14 GB, trainer private
  5.95 GB (well under the 18 GB cap), GPU free 9.2 GB. Guardrails confirmed active:
  `mcts_session_cache_max_states=131072`, epochs=60.
- **Crash signatures:** err.log has ONLY benign warnings — Triton "Failed to find CUDA"
  (cosmetic; torch uses CUDA directly, GPU is clearly working at 75%) and the known
  non-writable-buffer warning from `inference.py:214` (frombuffer; harmless, suppressed
  after first). No Fatal Python error / panicked / backtrace / 0xc0000005. No .dmp.
- **SealBot eval:** no `dense_cnn.evaluation.epoch_*.json` yet — expected, eval runs at
  epoch end and epoch 1 hasn't finished. No Goal-#4 trend to report this run.

**One thing to be aware of (NOT a problem):** calibration `meets_target=false` —
measured 12.8 selfplay pos/s vs the 64 target. This is the KNOWN heavier 96×6 + 512-sim
profile (self-play is GPU-forward-bound per the reframings above; 512 sims × 96×6 is far
heavier than scratch_64's 128 sims × 64×4). Calibration is informational and does NOT
halt. Selected knobs: inference bs=1024, selfplay bs=256, train bs=256, virtual_batch=4.
Implication for the watcher: epochs will be SLOW. At ~12.8 pos/s a full epoch of
self-play games will take a while — do NOT mistake a long-but-progressing epoch for a
stall. Use the "no new selfplay shard / no events progress in >25 min" rule, not a
wall-clock expectation.

**Still open / next-step instructions for next watcher:**
1. Re-verify the same way: flags first, then events.jsonl last stage, then
   selfplay/`epoch_*.npz` (or `.hxr`) mtimes vs now, then `Get-Process` on the pids in
   supervisor.pid (trainer) + supervisor.self.pid (supervisor), then watchdog tail.
   The PIDs WILL change on any relaunch — re-read them from the pidfiles, don't reuse
   54612/28292.
2. **First real milestone to look for:** `epoch_000001` finishing in events.jsonl + the
   first `dense_cnn.evaluation.epoch_000001.json`. Report wins/losses/mean_turns — this
   is the first Goal-#4 datapoint for the new architecture. scratch_64 baseline to beat:
   2–6 wins/64 vs SealBot best-50ms.
3. If you find a HALT flag: root-cause from the flag + newest trainer.*.err.log +
   crash_artifacts/ + any .dmp BEFORE touching anything. The most likely first-crash
   suspects for a brand-new run are bootstrap/shape issues or the heavier MCTS memory
   footprint — but the bootstrap loaded cleanly here, so a crash would more likely be in
   selfplay/shuffle. Capture artifacts, write the diagnosis, only then (if safe) fix +
   rebuild (maturin/cargo per Environment gotcha) + clear flag + restart supervisor.
4. Do NOT start a second supervisor; do NOT relaunch/kill the trainer yourself.

---

## 2026-05-30 — Live self-play pos/s on the dashboard (feature)

User asked the frontend to show the *proper* pos/s and have it be live. Problem: during
an active self-play epoch (the dominant phase) the dashboard had NO fresh pos/s — the
"Speed" card showed `calibration.selfplay_pos_s` (measured once at run start; for this
run a misleading **10.4 pos/s**) and the per-epoch `search_positions_per_second` is only
written when the whole epoch *finishes*. (`dense_cnn.training_progress.epoch_*.json` is
read by the dashboard but written by nothing — dead path.)

**Fix (producer → consumer → frontend):**
- `dense_cnn/.../selfplay.py`: writes `diagnostics/dense_cnn.selfplay.live.json` every
  ~2 s during the epoch loop (status `running`) + a final `completed` snapshot at epoch
  end. Carries the SAME authoritative metric the completed-epoch summary/calibration use
  (`search_positions_per_second = searched_positions / mcts_search_elapsed`), plus a
  wall-clock `timestamp` for staleness. Tiny JSON; negligible overhead.
- `hexo_frontend/web.py`: `_training_live_status` reads it; `_selfplay_live_summary`
  marks `live = (status==running AND age<=20s)` and exposes `search_pos_s` etc.
- `static/app.js`: Speed card prefers the live search-pos/s (`● LIVE · e{n} · {done}/{req}
  games`) while running, shows `(done)` for the last epoch, else falls back to calibration.

**Verified:** py_compile clean; consumer unit test (running/stale/completed) passes;
dashboard API returns the new `selfplay_live` field.

**Activation (user chose "bounce at epoch boundary"):**
- Dashboard on :8080 RESTARTED now (old pid 52864 → new pid 35520) so the Python consumer
  + new app.js are served. Same launch: `C:\Python314\python -m hexo_frontend.web --host
  0.0.0.0 --port 8080 --sealbot-path E:/SealBot`, PYTHONPATH=worktree pkgs. Log:
  `diagnostics/dashboard_8080.out.log`.
- Trainer (pid 446) still runs OLD selfplay.py (one process / all 60 epochs), so it won't
  emit the live file until restarted. Plan: wait for `checkpoints/epoch_000001.pt` to
  stabilize, then bounce the trainer ONCE so the supervisor resumes epoch 2 with the new
  code (near-zero lost work). Background watcher: `scripts/_wait_epoch1_boundary.sh`.
  Bounce = kill the pid in `diagnostics/trainer_wsl.pid`; uptime>180s so NOT a fast-crash,
  and the new epoch-1 ckpt resets no-progress — safe vs the breaker.
- This is the ONE authorized trainer bounce (supersedes the older "don't relaunch/kill the
  trainer yourself" note, for this single epoch-boundary restart only).
