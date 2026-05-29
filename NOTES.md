# NOTES — dense_cnn Model 1 (scratch_64) overnight autonomy log

This file is the running memory for the **recurring backstop routine** watching the
supervised `scratch_64` training run. Read `HANDOFF.md` first (project/build/run
conventions and crash history), then read the **most recent LOG entry below** for
continuity. Append a new dated entry every run. Be detailed, and **verify** claims
before recording them as fact (see "Verification principles").

---

## ORIENTATION (stable — re-read each run)

**What is running.** A host-side PowerShell supervisor keeps the `scratch_64`
training run advancing overnight without depending on any chat/agent session.
Architecture (added 2026-05-29):

- **Supervisor:** `scripts/supervise_scratch64.ps1`, launched detached. It *adopts*
  the live trainer, waits for it to exit, then on each exit: freezes crash artifacts
  → bumps `[checkpoint] resume_from` in the config to the newest `epoch_*.pt` → relaunches
  with fault-handler env vars. **The supervisor — not you — owns relaunching.**
- **Resume math (verified):** the checkpoint loader reads top-level `payload['epoch']`
  (NOT `metadata.epoch`, which is `None`), and start epoch = that + 1. So
  `resume_from = epoch_000NNN.pt` resumes at epoch NNN+1. Relaunches ADVANCE.
- **Circuit breaker:** halts (writes `diagnostics/supervisor_halted.flag`, stops
  relaunching) on ANY of: **3 consecutive crashes <180 s apart**, OR **>6 crashes in 60 min**,
  OR **`MaxNoProgressRelaunches` (=3) relaunches with no new epoch checkpoint** (the
  slow-loop / no-progress guard, added 2026-05-29 08:08 — closes the gap where ~20-min
  watchdog kills looped 7 h without ever advancing the epoch).
- **Clean finish:** when `latest_epoch+1 > loop.epochs` (currently `epochs=60`), the
  supervisor writes `diagnostics/supervisor_completed.flag` and stops (this is success,
  not a crash).
- **Eval diversity fix (activates on next relaunch):** eval games were deterministic
  (all 64 collapsed to ~3 trajectories). Config now sets `[model.config.evaluation]`
  `opening_temperature=0.6, opening_moves=8` so the dense player samples its opening.
  Until a relaunch happens, eval JSONs may still show identical `mean_turns`.

**Goal context.** Goal #4: train until the model holds its own vs SealBot best-50ms.
Watch the **per-epoch SealBot eval** (`dense_cnn.evaluation.epoch_*.json`: `wins`,
`losses`, `mean_turns`) trend — early on expect 0 wins; game **length** rising is the
first sign of progress.

**Key paths** (run root `E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64`):
- `diagnostics\supervisor.log` — lifecycle (ADOPT/LAUNCH/EXIT/RELAUNCH/CAPTURE/HALT/COMPLETED)
- `diagnostics\supervisor.pid` / `supervisor.self.pid` — current child trainer PID / supervisor PID
- `diagnostics\supervisor_halted.flag` / `supervisor_completed.flag` — terminal states
- `diagnostics\crashlog.md` — one signature block per exit
- `diagnostics\crash_artifacts\<ts>\` — frozen logs + dumps per exit
- `diagnostics\crashdumps\*.dmp` — WER minidumps (only if elevated `scripts/setup_python_minidumps.ps1` was run)
- `diagnostics\trainer.<stamp>.err.log` — newest one is the live trainer's stderr (stamp changes per relaunch)
- `diagnostics\events.jsonl` — per-stage progress; `checkpoints\epoch_*.pt`; `selfplay\epoch_*_game_*.npz`
- Config: `configs\dense_cnn_model1_scratch_64.toml`. Pointer to latest ckpt:
  `data\checkpoints\dense_cnn_model1_scratch_64_latest.txt`.

**What to do each run (decision tree):**
1. **Advancing normally?** Newest `epoch_*.pt` / selfplay shard mtime within ~15 min,
   no halt/completed flag → log a one-line progress note (current epoch, eval trend)
   and stop. Nothing to fix.
2. **`supervisor_completed.flag` exists?** Run finished all epochs. Report final eval
   trend. Decide with the user whether to raise `loop.epochs` and restart the supervisor.
3. **`supervisor_halted.flag` exists?** Circuit breaker tripped. ROOT-CAUSE: read the
   flag, `crashlog.md`, the newest `crash_artifacts\<ts>\`, the `.err.log` (look for
   `Fatal Python error`, `panicked`, `Traceback`, `0xc0000005`/access violation), and
   any `.dmp`. If it's a fixable bug in the Python worktrees or Rust MCTS/inference/engine,
   write the diagnosis + proposed fix into NOTES. If the fix is clearly safe, apply it
   (rebuild via maturin if Rust — see HANDOFF build note), then **delete the halt flag**
   and restart the supervisor (see "How to (re)start").
4. **Stalled?** Trainer process appears up but no new shard/checkpoint/events for
   >~25 min and no halt/completed flag and supervisor.log shows no recent EXIT/RELAUNCH
   → likely a hang. Capture the err.log tail + events tail into NOTES, and flag it.
5. **New `.dmp` or new fault signature but no halt yet?** Note it and begin root-cause;
   the supervisor may still be mid-relaunch (a brief gap is NORMAL — do not call it a
   crash on its own).

**Hard constraints:**
- **Do NOT relaunch the trainer yourself** — the supervisor does that. **Do NOT kill the
  live trainer.** **Do NOT start a second supervisor** if one is already alive (it uses a
  pidfile lock at `supervisor.self.pid`; starting another just aborts, but don't rely on it).
- **Capture before you change anything.** Never delete/overwrite logs or dumps.
- Don't blindly trust a single signal — verify (below).

**How to (re)start the supervisor** (only if it is NOT already running — check
`supervisor.self.pid` is a live powershell, and `supervisor.log` has no newer instance):
```powershell
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',`
  'E:\Hexo-BotTrainer\scripts\supervise_scratch64.ps1' -WindowStyle Hidden
```
Validate-only (safe, no side effects): add `-ValidateOnly`.

---

## VERIFICATION PRINCIPLES (don't treat findings as gospel)

- **Liveness:** confirm with ≥2 signals — process query if available
  (`Get-Process -Id <pid>`), AND file freshness (newest shard/checkpoint mtime vs now),
  AND supervisor.log. A pidfile alone can be stale (PID reuse). CPU-seconds rising over
  two samples a few seconds apart is the strongest "actually working" signal.
- **"Crashed" vs "relaunching" vs "completed":** distinguish via the FLAGS and
  supervisor.log, not by a momentary process absence. A short gap during relaunch is normal.
- **A crash signature in an OLD `.err.log` is history**, not a live fault — check the
  timestamp and which stamp is current.
- **Eval numbers:** identical `mean_turns` across epochs is a *symptom of deterministic
  eval* (known), not necessarily real stagnation — note whether the opening-diversity fix
  has activated yet (i.e., has a relaunch happened since 2026-05-29 00:36?).
- Re-derive epoch numbers from filenames; cross-check against `events.jsonl` last stage.

---

## LOG (newest entry first — prepend new entries here)

### 2026-05-29 ~12:08 — PROFILING: DEEP PASS (thoroughness-first); two self-corrections; all 4 phases measured
**TL;DR:** Per user direction (prioritize thoroughness, don't publish until the picture is solid),
I went well past the first cut. All four phases now have a defensible per-phase budget and a
measured binding constraint. Report rewritten: [`analysis/performance_profiling.md`](analysis/performance_profiling.md).
New re-runnable probes (all committed): `parse_selfplay_diag.py`, `reconstruct_epoch_timeline.py`,
`evaluator_microbench.py`, `train_microbench.py`, `train_step_reconcile.py`,
`train_step_pipeline.py`, `train_step_components.py`, `shuffle_mem_probe.py` (+ `*_summary.json`).

**Per-phase budget (epoch 21, reconciled to within 2 s of the instrumented epoch total):**
training **41.9%** (479 s), self-play **32.6%** (373 s), eval **15.2%** (173 s), shuffle **10.2%**
(116 s). selfplay+training instrumented; shuffle+eval from an mtime timeline summing to 1141 s vs
total 1143 s.

**Binding constraints, now pinned with evidence:**
- **Self-play is CPU/Python-bound, NOT GPU-bound.** Per searched position (35.2 ms): orchestration
  35%, NN-eval 29% (92–95% of which is the GPU forward), Rust tree 23%, encode 13%. GPU duty ~29%,
  raw forward ~16%. Cache hit ~10% → ~113 NN evals/position.
- **"11 pos/s" = cuDNN `benchmark=True` autotune**, PROVEN: first forward on a never-seen batch
  shape ~925 ms vs 32 ms steady; 0 ms with `benchmark=False`. ~925 ms × ~900 shapes ≈ 830 s ≈ the
  observed 842 s cold-epoch penalty. Fix = `cudnn.benchmark=False` (or bucket batch sizes).
- **Training step = ~465 ms GPU compute** (trunk fwd+bwd), 4 concordant measurements; latency ≈
  pipelined throughput (no sync win). + NPZ re-decompress bug ~244 ms/step (~95 s/epoch, 20–23×).
- **Shuffle is CPU-zlib-COMPRESS-bound** (~122 s of ~154 s modeled); peak RAM 0.87 GB/8000-row
  group over a 32.8 GB window. Memory is NOT a current constraint.

**Two self-corrections from repeated measurement:** (1) my first cut's 260 ms training step was a
non-reproducible transient — real value ~465 ms (caught by re-running + 3 other methods); (2) my
first cut said "~half the evaluator is Python marshaling" — measured it is 92–95% GPU forward.
Also CONFIRMED in BOTH inference and training that the FC policy head is **speed-neutral**
(quality-only; 472↔473 ms with a tiny conv head).

**Top fixes (measured impact):** P0 NPZ load-once (−~95 s/epoch, free); PB `cudnn.benchmark=False`
(−~830 s on every cold/relaunch epoch); P3 uncompressed shuffle scratch (−~40 s). Self-play CPU
buckets (P4–P6) are next; P7 fully-conv head = quality only.

**Cross-ref:** the companion `mcts_code_review.md` (entry below) MEASURES sims-scaling and surfaces
the key Goal-#4 finding my pass missed — MCTS is **root-parallel only**, so the **single-game eval/
play path is fully serial** (184 ms/128-sim, 576 ms/400-sim move). Cited in report §9.

**Still-needed (flagged):** in-situ `cProfile` over one real train epoch to size the cold-shard
disk-IO part of the training residual; direct `build_katago_shuffle` instrumentation; measured
SealBot-eval timing. No fixes applied — report only; run stays intentionally stopped.

### 2026-05-29 ~12:03 — BACKSTOP: run still DOWN by design (investigation stage); NO action taken
**TL;DR:** Confirmed the run remains intentionally stopped (policy-diffuseness / optimization stage),
unchanged since the deliberate ~10:17 stop. NOT a crash/halt/stall/completion. No state-changing
action taken; did NOT restart the supervisor (same reasoning as the ~11:03 entry — this is not a
breaker halt with a fixable bug, and the GPU is intentionally freed for the investigation).

**How verified (cross-checked, multiple signals, at 12:03):**
- **Flags:** `supervisor_halted.flag` and `supervisor_completed.flag` BOTH absent.
- **No live procs:** the only `train_model|supervise_scratch64|watch_model1` match was my own
  NonInteractive tool shell (pid 46140). No trainer / supervisor / watchdog alive.
- **Pidfiles stale & dead:** `supervisor.self.pid=45320`→alive=False, `supervisor.pid=53664`→alive=False
  (both mtime 08:08:38 — the supervisor killed at the stop). Unchanged from the ~11:03 cycle.
- **supervisor.log tail:** last line still `[08:08:38] ADOPT existing trainer pid=53664`. NO
  EXIT/RELAUNCH/CAPTURE after it → killed before any relaunch (documented stop order).
- **No crash:** newest crash_artifacts dir is `20260529_074646` (morning watchdog-kill history);
  crashdumps\ = none; newest err.log = `trainer.20260529_074646.err.log` (829 B, mtime 07:47:22) —
  fault-signature scan (Fatal Python error/panicked/Traceback/0xc0000005/access violation/STATUS_/
  SIGSEGV/SIGABRT) = NONE. No new err.log after the stop → external kill, not a fault.
- **Last activity:** `epoch_000022.pt` @ 10:07:41; newest selfplay shard `epoch_000023_game_062.npz`
  @ 10:16:37 (from the killed epoch-23 selfplay, regenerated on resume — harmless). So nothing has
  advanced in ~1h47m, consistent with "stopped," not "stalled" (no process is up to stall).
- **GPU free:** `nvidia-smi` 0% / 726 MiB used / 11269 MiB free. One python proc alive (pid 41584) is
  the **`hexo_frontend.web` dashboard** (port 8080, up since 5/28 20:21) — NOT the trainer; left alone.

**RESUME POINT (unchanged):** `checkpoints\epoch_000022.pt` (146,588,369 B, 10:07:41) → resumes at
**epoch 23**. Pointer `data\checkpoints\...latest.txt` agrees. CAVEAT (still true): config's literal
`[checkpoint] resume_from` says `epoch_000015.pt`; the supervisor overwrites it with the latest on
launch, but a MANUAL trainer launch would redo from epoch 15 — set it first. If the investigation
changes architecture (channels/blocks), epoch_000022.pt won't load (fresh run, not a resume).

**SealBot eval trend (frozen at epoch 22, best-50ms wins/64):** 17=6, 18=6, 19=2, 20=2, 21=4, 22=4 —
bouncing 2–6/64, no upward slope; mean_turns falling 41→31. This plateau is exactly what motivated the
stop + the [[scratch64-policy-bottleneck]] investigation (search budget=128 too low for the 400–1400
action space + diffuse FC policy head). Grinding more epochs of this config won't move the win rate.

**Context — investigation work has continued (read-only, no run-state change):** since the stop, prior
cycles produced [`analysis/performance_profiling.md`](analysis/performance_profiling.md) (~11:30) and
[`analysis/mcts_code_review.md`](analysis/mcts_code_review.md) (~12:00), both using the free GPU then
releasing it. None applied any fix or touched the supervisor/config/checkpoint. The proposed fixes
(raise sims ≥400 + widening, rebalance/shrink the FC policy head, NPZ load-once loader, then test
128ch/8block) remain NOT applied — gated on the user.

**NEXT CYCLE, do (in order):**
1. **First check whether the run is back up.** If a new trainer/supervisor is alive and advancing
   (newest `epoch_*.pt`/shard mtime within ~15 min, no flags) → user resumed; revert to the normal
   decision tree, log progress + new eval rows (epoch 23+). If a config/architecture change landed,
   epoch_000022.pt may not load (fresh run) — note it.
2. **If still down (expected):** re-confirm the same terminal signals (no procs, dead pidfiles, no
   flags, no new crash_artifacts/dumps, benign err.log) and log a one-line "still intentionally down".
   Do NOT restart the supervisor.
3. Only resume if the **user explicitly asks**: `-ValidateOnly` first, confirm no live supervisor
   (pidfile + supervisor.log), then launch (auto-injects `resume_from`=latest epoch_*.pt).

**Open items (report-don't-act, unchanged):** (i) ~19 stale `shuffleddata\*epoch_000016*` .tmp dirs —
safe to delete (well past epoch 16). (ii) WER minidumps still not enabled (irrelevant — zero native
crashes since the morning shuffle-RAM fix). (iii) policy-diffuseness fixes proposed but not applied.

### 2026-05-29 ~12:00 — MCTS CODE REVIEW (read-only; GPU used then freed; NO run-state change)
**TL;DR:** Wrote a full code-quality + performance + multithreading + memory review of the dense_cnn
MCTS → [`analysis/mcts_code_review.md`](analysis/mcts_code_review.md). **Read-only** w.r.t. the run:
no config/checkpoint/supervisor change, no training launched, run remains intentionally stopped
(resume point `epoch_000022.pt` → epoch 23, unchanged). I used the free GPU for a read-only
microbenchmark, then **freed it again** (verified 0% / 722 MiB at finish).

**What I ran:** [`analysis/mcts_microbench.py`](analysis/mcts_microbench.py) → 
[`analysis/mcts_microbench_summary.json`](analysis/mcts_microbench_summary.json) — random-init
64ch/4block net + production `DenseCNNInference` + real native `BatchedMctsSession`, driving fresh
`hexo_engine` games like self-play. Measures search mechanics/throughput (NOT strength).

**Key measured findings:**
- Search cost is **linear in sims** through 400 (128/256/400 = 1.0/2.0/3.2×); 800 showed an eval
  blow-up that is an autotune-off regime artifact, not tree behavior.
- **Single-root (eval/play) latency:** 128 sims = **184 ms/move**, 400 = 576 ms — single-game
  high-sims does NOT fit a 50 ms SealBot budget; the `player.decide` path is fully serial (1 root).
- **Root parallelism amortizes 3.3× (185→57 ms/root) then saturates at ~8 roots** on the serial
  GPU-eval stage → motivates pipelining select↔eval.
- Live tree is cheap/bounded (edges ~0.35 MB/32 trees); the dominant RAM pools are the **eval cache**
  (fills ~190k/262k in 24 moves) and a large **staged-prior pool** (24→87 MB, flagged to investigate).

**Two stuck processes cleaned up:** my FIRST benchmark attempt hung in cuDNN autotune thrash
(variable batch shapes + `cudnn.benchmark=True`); I killed that python (pid 53688) + its bash waiter
and re-ran with autotune disabled. Both were MY benchmark processes, not the trainer/supervisor —
verified no trainer/supervisor/watchdog was alive before or after (run was already stopped by design).

**No backstop action taken / needed:** still no halt/completed flags, no crash artifacts, run down by
design. This entry is just to record GPU use + the new analysis artifacts; do not treat the benchmark
as run activity.

### 2026-05-29 ~11:30 — PERFORMANCE PROFILING COMPLETE (epoch-cycle deep-dive; GPU micro-benchmarks run)
**TL;DR:** Took over the stuck profiling session. Killed a **hung GPU probe** (orphan
`python -` from the prior session, PID 33056, pegging the GPU at 100% / 11.7 GB VRAM — the
thing the previous session was blocked on); GPU then free (0%, 11.3 GB). Finished the per-phase
profiling **with real GPU micro-benchmarks** (the earlier draft could only estimate them).
Full report: [`analysis/performance_profiling.md`](analysis/performance_profiling.md). Re-runnable:
[`analysis/gpu_microbench.py`](analysis/gpu_microbench.py) (+ `gpu_microbench_summary.json`),
[`analysis/parse_epoch_timings.py`](analysis/parse_epoch_timings.py) (+ `epoch_timings.json`).
**Read-only** w.r.t. run state — no checkpoint/config/supervisor changes, no training launched.

**What is COMPLETE (measured):**
- **Per-phase epoch budget** (steady epochs 17–21, from per-epoch JSON): training ~42%,
  self-play ~33%, SealBot eval ~15%, shuffle ~10%. (~19 min/epoch.)
- **"11 pos/s" SOLVED:** it is the **cold first-epoch-per-process** number (epochs 9 & 16). The
  GPU evaluator is ~661 µs/state cold vs ~100–147 µs/state warm (≈4.5×) → cuDNN-autotune +
  clock-ramp tax on a fresh process. Steady self-play is ~28–34 pos/s, NOT 11.
- **Training is GPU-compute-bound at a 260 ms/step floor** (measured fwd 92 + bwd/opt 168 ms;
  ≈102 s/epoch irreducible at bs256) **plus a CPU input-pipeline tax** that grows with the
  replay window.
- **NPZ data-loader bug confirmed** (`trainer.py:_batch_from_npz` re-decompresses each shard
  per batch): measured **19.9×** slower than load-once, ≈**90 s/epoch** wasted, ~5-line fix.
- **Two prior-draft claims REFUTED by measurement:** (a) the per-step `.cpu().item()` sync /
  grad-clip / memory-pinning are **non-factors** (258.7 vs 257.9 ms; H2D 2.1 ms); (b) the FC
  policy head is **negligible for speed** (~0.26 ms/batch, ~2%) — it is a *quality* problem
  (see [[scratch64-policy-bottleneck]]), not the GPU cost. Shrinking it will NOT make more MCTS
  sims affordable.
- **Self-play is CPU/Python-bound:** GPU ~29% duty, raw forward only ~16% of the phase;
  dominated by Python orchestration (129 s), Rust MCTS tree (86 s), encode (50 s), and
  evaluator-side Python marshaling (~48 s).
- **Prioritized fixes** with impact in §9: P0 NPZ load-once (free, −90 s/epoch), P1 background
  prefetch/overlap loader (training ~400→~110–150 s), P3 uncompressed shuffle scratch, P4 cut
  evaluator marshaling, P5 trim self-play orchestration.

**What is STILL NEEDED (flagged in report §8):**
- An **in-situ `cProfile`/`torch.profiler` pass over one real train epoch** to split the
  ~250–380 s/epoch training residual into decompress vs disk-IO vs compute (my micro-bench used
  one hot shard, so it can't size cold-shard IO). Safe to run now (GPU free).
- A **cold-start probe** (first vs later evaluator batches, `cudnn.benchmark` on/off, watch
  `nvidia-smi` clocks) to separate autotune from clock-ramp in §7.
- A **measured "raise sims" cost** at visits ∈ {256, 400} (currently projected: epoch ≈ doubles).
- None of the §9 fixes are **applied** — report only; they are the next deliberate effort, gated
  on the user. The run remains intentionally stopped (resume point `epoch_000022.pt` → epoch 23).

### 2026-05-29 ~11:03 — BACKSTOP: run still DOWN by design (investigation stage); no action taken
**TL;DR:** Verified the run is intentionally stopped and stays that way. Terminal state =
"deliberately stopped for the policy-diffuseness investigation," NOT crash/halt/stall. No
processes, dead pidfiles, no flags, no relaunch in supervisor.log, no new crash artifacts/dumps.
The investigation is committed (`git 1776bef "Add policy-diffuseness investigation"`, branch
`analysis/policy-diffuseness`). I took **NO** state-changing action and did **NOT** restart the
supervisor — see "Why not restart" below.

**How verified (cross-checked, multiple signals):**
- **Processes:** the only match for `train_model|supervise_scratch64|watch_model1` was my own
  NonInteractive tool shell (pid 34184). No trainer / supervisor / watchdog alive.
- **Pidfiles stale & dead:** `supervisor.self.pid=45320` → alive=False; `supervisor.pid=53664`
  → alive=False (both mtime 08:08:38, left by the supervisor that was killed at the stop).
- **Flags:** `supervisor_halted.flag` and `supervisor_completed.flag` BOTH absent.
- **supervisor.log tail:** last line is `[08:08:38] ADOPT existing trainer pid=53664`. **No
  EXIT/RELAUNCH/CAPTURE after it** → the supervisor was killed before it could relaunch (matches
  the documented stop order: supervisor FIRST so the trainer kill couldn't trigger a relaunch).
- **No crash:** newest crash_artifacts dir is `20260529_074646` (the morning watchdog-kill loop,
  history). crashdumps\ = none. Newest err.log = `trainer.20260529_074646.err.log` (829 B, benign
  Triton/torch warnings only) — **no new err.log after the stop**, i.e. external kill, not a fault.
- **Last activity / actual stop time:** `epoch_000022.pt` @ 10:07:41, eval epoch22 @ 10:10:17,
  newest selfplay shard `epoch_000023_game_000062.npz` @ 10:16:37. So the trainer was actually
  killed **~10:17–10:20**, mid epoch-23 selfplay. **NOTE for next self:** the entry below headed
  "~09:35 RUN INTENTIONALLY STOPPED" is **mislabeled** — epoch_000022 didn't exist until 10:07, so
  that stop happened ~10:17, not 09:35. The 10:04 entry (also below) predates the stop. The LOG is
  not in strict chronological order; trust file mtimes over the headings.

**RESUME POINT (unchanged):** latest checkpoint `checkpoints\epoch_000022.pt` (146,588,369 B,
10:07:41); pointer `data\checkpoints\dense_cnn_model1_scratch_64_latest.txt` agrees. Resuming
from it starts **epoch 23** (loader = top-level `payload['epoch']`+1). The `epoch_000023_game_*`
shards on disk are from the killed epoch-23 selfplay; they're regenerated on resume — harmless.

**SealBot eval trend through epoch 22 (best-50ms, wins/64):**
| epoch | wins | losses | mean_turns |
|------:|-----:|-------:|-----------:|
| 17 | 6 | 58 | 40.50 |
| 18 | 6 | 58 | 41.31 |
| 19 | 2 | 62 | 35.44 |
| 20 | 2 | 62 | 34.75 |
| 21 | 4 | 60 | 34.38 |
| 22 | 4 | 60 | 31.25 |
Wins still oscillate in a 2–6/64 band with no upward slope; mean_turns keeps falling (41→31).
This plateau is exactly what motivated the stop + the [[scratch64-policy-bottleneck]] investigation.

**Why NOT restart the supervisor (deliberate):** (1) This is not a crash/halt — the backstop only
owns relaunching after a *circuit-breaker halt with a fixable bug*; here there's no fault. (2) The
GPU was intentionally freed for the investigation, which is now committed. (3) The investigation's
own conclusion (analysis/policy_diffuseness_investigation.md + the ~10:15 entry below) is that the
bottleneck is SEARCH BUDGET (128 sims, far too low for a 400–1400 action space) + POLICY-HEAD
architecture — i.e. **grinding more epochs of this exact config will not move the win rate**; the
recommendation is to fix sims/widening + rebalance the policy head *before* resuming/scaling.
Auto-resuming the grind would directly contradict that. → user decision, not the backstop's.

**NEXT CYCLE, do (in order):**
1. **First check whether the run is back up.** If a new trainer/supervisor is alive and advancing
   (newest `epoch_*.pt`/shard mtime within ~15 min, no flags), the user resumed it → revert to the
   normal decision tree and just log progress + the new eval rows (epoch 23+). Watch whether any
   config change (sims↑, policy-head rebalance, bigger model) landed — if architecture changed,
   epoch_000022.pt won't load (fresh run, not a resume).
2. **If still down (expected):** confirm the same terminal signals (no procs, dead pidfiles, no
   flags, no new crash artifacts) and log a one-line "still intentionally down" note. Do NOT restart.
3. Only resume the supervisor if the **user explicitly asks**; then `-ValidateOnly` first, confirm
   no live supervisor (pidfile + supervisor.log), then launch (it auto-injects `resume_from`=latest
   epoch_*.pt). Reminder: if launched MANUALLY bypassing the supervisor, the config's literal
   `[checkpoint] resume_from` still says `epoch_000015.pt` — set it to the latest first.

**Open items (unchanged, report-don't-act):** (i) ~19 stale `shuffleddata\*epoch_000016*` .tmp dirs
from the pre-fix killed shuffles — safe to delete, run is well past epoch 16. (ii) WER minidumps
still not enabled (irrelevant — zero native crashes since the morning shuffle-RAM fix). (iii) The
policy-diffuseness fixes (raise sims ≥400, rebalance policy head, then test 128ch/8block) are
proposed but NOT applied — that's the next deliberate effort, gated on the user.

### 2026-05-29 ~09:35 — RUN INTENTIONALLY STOPPED for the optimization/investigation stage
- **Not a crash — a deliberate clean shutdown** so the GPU is free for a profiling task.
  Stopped in the correct order: **supervisor (PID 45320) FIRST** (so the trainer kill could
  NOT trigger a relaunch), **then trainer (PID 53664)**, then the **watchdog (PID 36800)**.
  Verified all three down, no remaining train_model/supervisor/watchdog processes, and
  supervisor.log shows NO relaunch after the stop. Real-time monitors: none armed (already
  ended). GPU confirmed free of any python/trainer process (compute-app list was all desktop
  apps; 0% util, ~8.7 GB free).
- **RESUME POINT — latest saved checkpoint: `epoch_000022.pt`**
  `E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\checkpoints\epoch_000022.pt`
  (pointer `data\checkpoints\dense_cnn_model1_scratch_64_latest.txt` agrees). Resuming from it
  starts **epoch 23** (loader uses top-level `payload['epoch']`+1).
- **TO RESUME LATER** (only after the profiling task releases the GPU): just relaunch the
  supervisor — it auto-injects `resume_from = <latest epoch_*.pt>` and launches, so it picks
  up from epoch_000022.pt automatically:
  ```powershell
  Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',`
    'E:\Hexo-BotTrainer\scripts\supervise_scratch64.ps1' -WindowStyle Hidden
  ```
  (Sanity-check first with `... supervise_scratch64.ps1 -ValidateOnly`.) All guardrails are
  baked into the config/scripts already: shuffle `group=8000`/`bucket=8000`/`window=300000`,
  watchdog `MinFreeRamGb=4`, breaker fast+hourly+**no-progress(3)**.
  CAVEAT: the config's `[checkpoint] resume_from` line still literally says `epoch_000015.pt`
  (last written by the 07:46 relaunch); the supervisor OVERWRITES it with the latest on launch,
  so the supervisor path is correct. If you instead launch the trainer MANUALLY (bypassing the
  supervisor), first set `resume_from` to the latest checkpoint or it will redo from epoch 15.
  ALSO: if the optimization stage changes the **architecture** (channels/blocks/etc.),
  `epoch_000022.pt` will NOT load into the new shape — that's a fresh run, not a resume.
- **Final eval state at stop (SealBot best-50ms, wins/64):** 15=0 → 16=3 → 17=6 → 18=6 →
  19=2 → 20=2 (low single digits, bouncing; opening diversity verified at epoch 16). The
  motivation for stopping = move to investigating the [[scratch64-policy-bottleneck]] (diffuse
  policy head) rather than keep grinding epochs.
- **Housekeeping still pending (deferred, not done):** ~19 stale `shuffleddata\*epoch_000016*`
  .tmp dirs from the pre-fix killed shuffles are still on disk — safe to delete now that the run
  is stopped and past epoch 16.


### 2026-05-29 ~10:04 — HEALTHY: clean to epoch 21, but wins DIPPED (6→2→2→4) & mean_turns keeps falling — watch (backstop)
**TL;DR:** Still nothing to fix. Same trainer pid 53664 (the loop-breaker from 07:46:46, both fixes
active) is STILL live — ~2h17m uptime, no relaunch since the 08:08:38 ADOPT. Produced epochs 16→21
clean at ~18 min/epoch and is now in epoch-22 selfplay. No halt/completed flags, no faults, no dumps.
Took **NO** state-changing action — observe + log only. **One thing to watch (not act on yet):** the
SealBot win count is NOT climbing — it peaked at 6 (epochs 17-18) then dropped to 2,2,4 (epochs 19-21),
while mean_turns falls monotonically (41→35→34). Could be sharper play OR the diversified-opening lines
ending in quicker losses. Needs a few more epochs to call.

**How verified (cross-checked, 3+ signals):**
- Flags: `supervisor_halted.flag` / `supervisor_completed.flag` BOTH absent.
- Liveness: `Get-Process 53664` ALIVE (python, **CPU 57,068 s** — up from 32,183 s at the 09:03 cycle,
  ~+25k CPU-s in one hour = actively crunching; WS 4.31 GB). Supervisor pid 45320 ALIVE (powershell,
  CPU 1 s — correctly blocked in WaitForExit).
- Advancement: checkpoints `017`@08:34:29 → `018`@08:52:06 → `019`@09:09:56 → `020`@09:27:53 →
  `021`@09:46:55 (~18 min cadence, sizes still creeping 146,579,409 → …586,641 B). Newest selfplay
  shard `epoch_000022_game_*.npz` @ 09:58:11 (~5 min before now 10:03:21). No `epoch_000022.pt` yet.
- supervisor.log: last line still `[08:08:38] ADOPT existing trainer pid=53664`. NO EXIT/RELAUNCH/
  CAPTURE after it → 53664 has run selfplay→shuffle→train→ckpt→eval for SIX epochs without dying.
- Faults: live err.log (`trainer.20260529_074646.err.log`, 829 B, mtime 07:47:22 — unchanged) = only
  benign Triton "Failed to find CUDA" + torch non-writable-buffer warnings. No Fatal Python error /
  panicked / Traceback / 0xc0000005 / STATUS_ / SIGSEGV. crashdumps\ → NONE.
- RAM: resource_watchdog last `status=ok`, trainer private 8.37 GB (well under the 18 GB ceiling),
  stop.json frozen at 07:46:45 (no new watchdog kill since the old-config pid). The 8000-row shuffle
  bound is holding the peak flat as the replay window grows — confirmed through 6 more epochs.

**SealBot eval trend — wins PEAKED then dipped; games keep shortening:**
| epoch | wins | losses | mean_turns | note |
|------:|-----:|-------:|-----------:|------|
| 16 | 3 | 61 | 41.47 | first epoch with both fixes |
| 17 | **6** | 58 | 40.5  | wins doubled |
| 18 | **6** | 58 | 41.31 | held |
| 19 | 2 | 62 | 35.44 | wins dropped; mean_turns falls |
| 20 | 2 | 62 | 34.75 | |
| 21 | 4 | 60 | 34.38 | partial recovery |
So the 09:03 entry's "sustained trend 3→6→6" did NOT continue climbing — it reverted to 2-4. Over 6
epochs the win rate is bouncing in a 2–6/64 band with no clear upward slope, and mean_turns has dropped
~7 turns (41→34). Interpretation is ambiguous (sharper vs. quicker losses); **do not over-read either
direction** — this is still very early Goal-#4 territory (0 wins was the baseline through epoch 15).

**NEXT CYCLE, do (in order):** (1) confirm tools work (note: `$pid` is a PowerShell read-only
automatic var — DON'T use it as a loop var, it errors; use `$id`). (2) normal decision tree — flags
first, then newest ckpt/shard mtimes + supervisor.log tail. Expect epoch ≥22 advancing ~every 18 min;
`epochs=60` so ~38 epochs × 18 min ≈ **~11 h of runway** left. (3) Append epoch 22+ eval rows and
**judge the win/mean_turns trend over a wider window** — if wins keep oscillating 2–6 with falling
mean_turns for several more epochs, that's a plateau worth flagging to the user (maybe lr/temperature/
eval-opponent question), not a bug. (4) Keep an eye on RAM (private should stay ~8-9 GB; free stays
healthy). (5) If `supervisor_completed.flag` appears (epoch 60), report final eval + ask about raising
`loop.epochs`.

**Open items:** (i) WER minidumps still not enabled (irrelevant — zero native crashes since the fix).
(ii) Stale `shuffleddata\*-epoch_000016*.tmp` dirs from the old killed shuffles may still be on disk —
report to user, safe to delete, don't act unsolicited. (iii) Win-rate plateau question (above) — track,
don't act.

### 2026-05-29 ~09:03 — HEALTHY: run advancing clean to epoch 19, wins holding 3→6→6 (backstop)
**TL;DR:** Nothing wrong. pid 53664 (the loop-breaker from 07:46:46, first run with BOTH the
shuffle-RAM fix and the opening-diversity fix) is STILL the live trainer — ~1h17m uptime, no
relaunch since the 08:08:38 ADOPT. It has produced epochs 16, 17, 18 cleanly and is now in
epoch-19 selfplay. The 7-hour epoch-16 crash loop is decisively broken. I took **NO**
state-changing action — observe + log only.

**How verified (cross-checked, 3+ signals):**
- Flags: `supervisor_halted.flag` / `supervisor_completed.flag` BOTH absent.
- Liveness: `Get-Process 53664` ALIVE (python, CPU 32,183 s, WS 4.48 GB); supervisor pid 45320
  ALIVE (powershell, CPU ~1 s — correctly blocked in WaitForExit). File freshness: newest
  selfplay shard `epoch_000019_game_*.npz` @ 09:01:28 (~2 min before now 09:03:21).
- Advancement: checkpoints `epoch_000016.pt`@08:17:59 → `017`@08:34:29 → `018`@08:52:06,
  ~17-min cadence (and growing slightly in size each epoch: 146,577,553 → …579,409 → …581,201 B).
  No `epoch_000019.pt` yet (epoch 19 still in selfplay).
- supervisor.log: last line is `[08:08:38] ADOPT existing trainer pid=53664`. NO EXIT/RELAUNCH/
  CAPTURE after it → 53664 ran selfplay→shuffle→train→ckpt→eval for 3 epochs without dying.
- Faults: live err.log (`trainer.20260529_074646.err.log`, 53664's) = only benign Triton
  "Failed to find CUDA" + torch non-writable-buffer warnings. Fault-signature scan
  (Fatal Python error / panicked / Traceback / 0xc0000005 / access violation / STATUS_ /
  SIGSEGV / SIGABRT) → NONE. crashdumps\ → NONE.

**SealBot eval trend — wins HOLDING, both fixes validated across multiple epochs:**
| epoch | wins | losses | mean_turns | note |
|------:|-----:|-------:|-----------:|------|
| 13 | 0 | 64 | 59.625 | pre-fix, deterministic |
| 14 | 0 | 64 | 57.75  | pre-fix, deterministic |
| 15 | 0 | 64 | 56.5   | pre-fix, deterministic (3 distinct trajectories) |
| 16 | 3 | 61 | 41.47  | FIRST epoch with both fixes; 26 distinct games |
| 17 | **6** | 58 | 40.5   | wins doubled |
| 18 | **6** | 58 | 41.31  | wins held |
The 08:24 entry's caveat ("3 wins on epoch 16 might be opening variance, not strength") is now
answered: wins 3→6→6 over three consecutive epochs is a **sustained trend, not a one-off blip**.
mean_turns stabilized ~41 (down from ~57 pre-diversity). Real Goal-#4 progress.

**NEXT CYCLE, do (in order):** (1) confirm tools work. (2) normal decision tree — flags first,
then newest ckpt/shard mtimes + supervisor.log tail. Expect epoch ≥19 advancing ~every 17-18 min;
`epochs=60` so ~12 h of runway left at this cadence. (3) Append the epoch 19+ eval rows; watch
whether wins keep climbing (>6) or plateau. (4) Sanity-check RAM headroom held through the bigger
replay window (resource_watchdog.jsonl free_ram should stay >4 GB — the 8000-row shuffle bound
keeps the peak flat regardless of window growth). (5) If `supervisor_completed.flag` appears
(epoch 60 done), report final eval and ask the user whether to raise `loop.epochs`. (6) Stale
`shuffleddata\*-epoch_000016*.tmp` dirs from the old killed shuffles may still be on disk —
report to user, safe to delete, but don't act unsolicited.

**Open items:** (i) WER minidumps still not enabled (irrelevant while no native crashes).
(ii) Old scratch probe files from the 08:24 cycle (`_state_probe.txt` etc.) — re-confirm they're
gone if convenient. (iii) ORIENTATION breaker text now DOES include the no-progress guard (08:08
entry) — that stale note is resolved.

### 2026-05-29 ~08:24 — FIX CONFIRMED: epoch 16 cleared, crash-loop BROKEN, run advanced to epoch 17 (backstop)
**TL;DR:** The 8000-row shuffle fix (07:33 entry) + the watchdog 8→4 GB relax & no-progress
guard (08:08 entry, by a concurrent session) WORKED. `epoch_000016.pt` was written **08:17:59**
— the FIRST epoch-16 checkpoint ever, after ~7 h of watchdog-kill looping. Epoch-16 SealBot
eval ran (`dense_cnn.evaluation.epoch_000016.json`, 08:21:23) and the run is now into
**epoch 17 selfplay** (`epoch_000017_*` shards present). No halt/completed flags. I took NO
state-changing action — only observed and logged.

**How verified (cross-checked):**
- `checkpoints\epoch_000016.pt` exists, 146,577,553 B, mtime 08:17:59 (vs epoch_000015.pt @
  00:53). `epoch_000017.pt` not yet (epoch 17 in selfplay).
- `dense_cnn.evaluation.epoch_000016.json` mtime 08:21:23 — first eval with the
  opening-diversity fix active.
- supervisor.log: NO new EXIT/RELAUNCH/CAPTURE after the 07:46:46 RELAUNCH of pid=53664.
  So pid=53664 (FIRST fixed-config run) ran selfplay→shuffle→train→checkpoint→eval→epoch17
  without dying. The four prior loops (06:21/06:42/07:03/07:23) each died ~20 min in, at the
  shuffle. Loop broken.
- watchdog stop.json frozen at 07:46:45 (the OLD-config pid=48140 kill: free_ram 2.10 GB,
  private 18.48 GB) — no new kill since. Confirms the relaxed `free_ram_gb<4` threshold still
  caught that genuine spike but the 8000-row shuffle no longer spikes.
- Flags: `supervisor_halted.flag`/`supervisor_completed.flag` both absent. Supervisor pid=45320
  (restarted 08:08:38) alive, adopted pid=53664.

**Note on a confusing earlier signal this cycle:** at cycle start a muddled, partly-cancelled
tool batch + a stretch where the shell/Glob/Read tools returned EMPTY output made me briefly
(and WRONGLY) think the run was absent and the .md files corrupted. They were NOT — HANDOFF.md
/NOTES.md are intact and the run was alive the whole time. Lesson for next self: when
enumeration tools return empty even for files you KNOW exist (e.g. `configs/*.toml`), the
HARNESS is degraded — do NOT read "empty" as "absent." Re-test with a known file first.

**Eval trend — FIRST WINS + opening-diversity CONFIRMED (both fixes validated):**
| epoch | wins | losses | mean_turns | eval games (distinct .hxr sizes) |
|------:|-----:|-------:|-----------:|----------------------------------|
| 13 | 0 | 64 | 59.625 | (deterministic) |
| 14 | 0 | 64 | 57.75  | (deterministic) |
| 15 | 0 | 64 | 56.5   | **3** distinct (32×226B, 20×555B, 12×395B — collapsed) |
| 16 | **3** | 61 | **41.47** | **26** distinct (broad spread 201–523B) |

Two things landed at once on epoch 16 (the first epoch reached after the relaunch with both
fixes active):
1. **Opening-diversity fix CONFIRMED.** Eval games went from **3 distinct trajectories
   (epoch 15) → 26 distinct (epoch 16)** out of 64. `opening_temperature=0.6, opening_moves=8`
   works; the 64 games are no longer near-identical. (Memory `scratch64-eval-opening-diversity`
   can be marked validated.)
2. **First-ever wins:** 3/64 vs SealBot best-50ms (epochs 9–15 were all 0). CAVEAT: don't
   over-read this — diversified openings mean the dense player now sometimes leaves "book," so
   part of the 0→3 jump may be opening variance rather than pure strength, and `mean_turns`
   dropping 56.5→41.5 means games are also ending FASTER (could be sharper play OR quicker
   losses in off-book lines). Watch the wins trend over epochs 17–20 to tell signal from noise.

**Cleanup:** I left three scratch probe files I created this cycle and tried to delete:
`_state_probe.txt` (repo root), `runs\...\_backstop_probe.txt`, `runs\...\_eval_probe.txt`.
If any remain, delete them (harmless, gitignored except the repo-root one — remove that so it
doesn't clutter `git status`).

**NEXT CYCLE, do (in order):** (1) confirm tools work (read CLAUDE.md). (2) normal decision
tree — flags, newest checkpoint/shard mtimes, supervisor.log tail. Expect epoch ≥17 advancing
~every 20–25 min. (3) READ the epoch-16 (and later) eval JSONs + check `.hxr` opening
diversity — this validates the opening-diversity fix, still pending. (4) Watch RAM headroom
holds as the replay window keeps growing past epoch 16 (the 8000-row bound should keep the
shuffle peak flat regardless of window size, but verify free_ram stays >4 GB in
resource_watchdog.jsonl). (5) If it ever crash-loops again, see 07:33 "NEXT RUN" fallbacks.

### 2026-05-29 08:08 — no-progress breaker guard added; supervisor restarted (PID 45320)
- **Implemented the slow-loop guard** in `scripts/supervise_scratch64.ps1`: new param
  `MaxNoProgressRelaunches = 3`. The loop tracks the highest epoch checkpoint seen; each exit
  that does NOT advance it increments `noProgress`; at 3 it HALTS (writes
  `supervisor_halted.flag`, reason "no checkpoint progress across N relaunches"). This closes
  the gap that let the epoch-16 watchdog-kill loop run 7 h unflagged (those kills were ~20 min
  apart, so the fast-crash rule never fired). The `breaker state:` log line now also prints
  `noProgress=k/3 (latest epoch N)`.
- **Supervisor restarted to load it:** stopped old PID 14920, started **PID 45320**, which
  ADOPTED the live trainer **53664** (latest checkpoint epoch 15). One supervisor only;
  pidfile lock intact. `noProgress` baseline = 15. Script syntax-validated via `-ValidateOnly`
  before restart.
- **COORDINATION (I now own this):** NOTES.md is the single source of truth. Current applied
  fixes, reconciled across both agents:
  * config `[model.config.samples]`: `shuffle_keep_target_rows=300000`,
    `shuffle_worker_group_size=8000`, `approx_rows_per_out_file=8000` (group+bucket = routine's
    07:33 fix; window = session's 01:33 fix).
  * watchdog free-RAM floor `MinFreeRamGb=4` — LIVE watchdog (currently PID 36800) AND the
    durable default in `start_model1_training.ps1`. **Do NOT re-relax it; the 07:33 fallback
    (c) is satisfied.**
  * breaker: fast-crash + hourly + **no-progress(3)** guards.
  Routine: if you change any of these, reconcile HERE and don't undo the others.
- **PENDING VERIFICATION (report target):** 53664 (started 07:46, first run with the COMPLETE
  memory fix) is at its epoch-16 shuffle now. Expect `epoch_000016.pt` to finally save →
  then check epoch-16 eval (`evaluation\epoch_000016\*.hxr` size clusters; >3 = opening
  diversity working) and the eval win/mean_turns. If instead it gets killed again, the
  no-progress guard will now halt after 3 tries instead of looping.


### 2026-05-29 08:02 — reconciling two concurrent fixes; 53664 is the verification run
- **Heads-up: two agents acted on this run.** The interactive session AND this routine both
  made changes around 07:25–07:33. They are COMPLEMENTARY (no conflict), but record both so
  nothing gets re-done:
  - **Interactive session (~07:26):** lowered the resource-watchdog free-RAM floor
    **`MinFreeRamGb` 8 → 4** — both on the LIVE watchdog (now PID 36800) AND the durable
    default in `scripts/start_model1_training.ps1`. So the 07:33 entry's "fallback (c) relax
    watchdog 8→5" is ALREADY DONE (at 4). Do NOT relax it again.
  - **Routine (07:33):** `shuffle_worker_group_size 40000→8000`, `approx_rows_per_out_file
    70000→8000` (bounds BOTH shuffle phases). This was the decisive fix.
- **Why earlier 300k/40k still died, confirmed by the kill records:** with watchdog@8, runs
  died in shuffle **phase 1** (group load, ~13.6 GB priv / ~5–6 GB free, only `free_ram<8`
  fired). After I lowered the floor to 4, PID 48140 SURVIVED phase 1 but then hit shuffle
  **phase 2** (the 70k-row bucket) → priv **18.5 GB / free 2.1 GB**, tripping all three
  (`free_ram<4, free_virtual<12, private>18`) at 07:46. So phase 2 (`approx_rows_per_out_file`)
  was the second, bigger spike — exactly what the routine's bucket→8000 change targets.
- **53664 (started 07:46:46) is the FIRST run with BOTH fixes** (group=8000, bucket=8000,
  watchdog floor=4). It should clear the epoch-16 shuffle (~08:08–10) and finally save
  `epoch_000016.pt`. Config verified no-BOM (`5B 6D 6F`), knobs confirmed in file.
- **STILL OPEN / not yet fixed:** the circuit breaker's blind spot — it only catches *fast*
  (<180 s) crashes, so this ~21-min slow loop ran **7+ hours** unflagged. A **no-progress
  guard** (halt if latest checkpoint epoch doesn't advance over N relaunches) is NEEDED in
  `supervise_scratch64.ps1` (requires editing it + restarting the supervisor). Not done yet.
- **Stale in this file:** ORIENTATION's breaker description (lines ~24–25) still states only
  the fast/hourly thresholds — it does NOT mention the slow-loop gap. Treat that as known-bad
  until the no-progress guard lands.
- **Disk:** 19 stale `shuffleddata\*epoch_000016*` .tmp dirs from the killed shuffles. Safe to
  delete once epoch 16 completes; left in place for now.


### 2026-05-29 07:33 — FOUND IT: epoch-16 crash-LOOP = shuffle phase-1 RAM spike; applied a real fix
**TL;DR:** The run has been stuck in a silent crash-loop on **epoch 16 for ~6.5 h
(17 relaunches, 01:50→07:23), never advancing past epoch 15.** The "crashes" are
NOT native faults — the **resource watchdog is killing the trainer** because the
two-phase shuffle's phase-1 group load spikes RAM and drops system free RAM below
the watchdog's `free_ram_gb < 8` threshold. The circuit breaker never trips because
the kills are ~20 min apart (>180 s) and ≤6/hr. I applied a config fix that the
supervisor will pick up on its **next natural relaunch** (no kill/relaunch by me).
**MUST VERIFY next run that the fix worked** (see "NEXT RUN" below).

**How I verified the state (cross-checked, not one signal):**
- No `supervisor_halted.flag`, no `supervisor_completed.flag`.
- Newest checkpoint = `epoch_000015.pt` @ 00:53 (≈6.6 h stale). NO `epoch_000016.pt`
  has EVER been written. But `epoch_000016_game_*.npz` shards are fresh (age <1 min)
  and number **256** (all selfplay games complete). → selfplay finishes; epoch never does.
- `supervisor.log`: a monotonous loop — every ~20 min `EXIT pid=… code=-1 uptime≈1220-1330s`
  → `resume_from -> epoch_000015.pt (start epoch 16)` → `RELAUNCH`. ~17 cycles since 01:50.
  CAPTURE sig each time: "no fault text (clean or external stop)".
- Newest `trainer.*.err.log` (829 B): only benign Triton "Failed to find CUDA" +
  torch non-writable-buffer warnings. No `Fatal Python error`, no `panicked`, no traceback.
- **Smoking gun:** `diagnostics\resource_watchdog.stop.json` @ 07:23:30 (matches the
  07:23:31 EXIT) → `"status":"stopping_trainer"`, `"critical":["free_ram_gb < 8"]`,
  trainer working_set 11.66 GB / private 13.5 GB, system free_ram 5.13 GB.
- **RAM trajectory** (`resource_watchdog.jsonl`, the 4 min before the kill): trainer
  FLAT at ws ~5.5 GB / priv ~7.5 GB, free RAM steady ~11 GB, `status: ok` … then in
  ONE ~6 s step (11:23:24→11:23:30 UTC) ws 6.2→11.66, priv 8.0→13.5, free 10.7→5.13 →
  kill. A discrete +5.5 GB spike, NOT a gradual leak.
- Only **2 python.exe** processes exist (live trainer + a tiny 0.08 GB helper) → **no
  orphan/leak accumulation across cycles**; consistent ~20-min uptime confirms each
  fresh process independently hits the spike. Baseline non-trainer RAM ≈ 17 GB (31 GB
  box), so only ~11 GB is free for the trainer to grow into.
- **18 leftover `shuffleddata\*-epoch_000016.tmp` dirs** (one per killed shuffle) +
  20 `epoch_000016` stage_starts in events.jsonl, 0 finishes → **dies IN the shuffle.**

**ROOT CAUSE (code-level, `dense_cnn/python/.../replay.py`):** `_build_split_outputs`
is a two-phase on-disk shuffle. **Phase 1** (`replay.py:738`) iterates `_worker_groups`,
each up to `shuffle_worker_group_size` rows, and `_load_group_kept_arrays` →
`np.concatenate` loads a whole group into RAM. Each dense row ≈ **110 KB** (13×41×41
f32 input plane + several 41×41 policy planes), so a **40000-row group ≈ 4.4 GB resident
+ concat transient ≈ the observed ~5.5 GB spike.** Phase 1 runs FIRST, so it's the killer.
**Phase 2** (`replay.py:766`) later loads a whole output bucket (~`approx_rows_per_out_file`
= 70000 rows ≈ even bigger) — would also trip if phase 1 didn't.

**Why the prior fix (the 01:33 entry below) failed:** it cut `shuffle_keep_target_rows`
600k→300k and `shuffle_worker_group_size` 80k→40k, but (a) 40k still spikes ~5.5 GB —
not under the ~3 GB the `<8 GB free` threshold needs — and (b) it left
`approx_rows_per_out_file = 70000` untouched (phase-2 peak). The window size
(`keep_target_rows`) barely matters here — the peak is per-GROUP/per-BUCKET, not the
whole window.

**FIX APPLIED (config only, correctness-neutral):** in
`configs\dense_cnn_model1_scratch_64.toml` `[model.config.samples]`:
`shuffle_worker_group_size 40000 → 8000` and `approx_rows_per_out_file 70000 → 8000`.
At 8000 rows each phase peak ≈ 0.9 GB resident / ~1.7 GB transient → free RAM should
stay ~9+ GB (above the 8 GB kill threshold) with margin. The scatter→gather two-phase
shuffle is **correctness-neutral at any group/bucket size** (this is exactly how KataGo
bounds RAM); the only cost is more, smaller scratch/output files + a bit more I/O.
I rewrote the config comment block to record this root cause. **No Rust change, no
rebuild needed** (pure config; the trainer reads it at startup).

**Why this was safe to apply autonomously & how it activates:** I did NOT kill or
relaunch the trainer, did NOT touch `resume_from`, did NOT start a 2nd supervisor.
The supervisor (pid 14920, alive, blocked in WaitForExit) relaunches via
`start_model1_training.ps1 -ConfigPath <this config>` and only rewrites
`[checkpoint] resume_from` — it leaves `[model.config.samples]` alone. So: the CURRENT
trainer (pid 48140, started 07:23:31) will still die in the shuffle ~07:43 with the OLD
in-memory config; the supervisor then relaunches reading the EDITED config, and that
relaunch is the first one with the fix. **Expected first proof: an `epoch_000016.pt`
checkpoint + an epoch_000016 eval JSON appear, and the run advances to epoch 17.**

**SealBot eval trend (frozen at epoch 15 — epoch 16 never completes, so no new eval):**
| epoch | wins | losses | mean_turns |
|------:|-----:|-------:|-----------:|
| 9 | 0 | 64 | 54.0 |
| 10 | 0 | 64 | 58.375 |
| 11 | 0 | 64 | 58.375 |
| 12 | 0 | 64 | 60.875 |
| 13 | 0 | 64 | 59.625 |
| 14 | 0 | 64 | 57.75 |
| 15 | 0 | 64 | 56.5 |
Still 0 wins (expected this early). NOTE: the opening-diversity eval fix only affects
epoch 16+ eval, which we've never reached — so it remains UN-validated. Can't judge it
until an epoch ≥16 completes.

**NEXT RUN, do (in order):**
1. **Verify the fix landed and worked.** Check: does `checkpoints\epoch_000016.pt` now
   exist? Is there a `trainer.<stamp>.err.log` with a stamp AFTER ~07:43 whose run
   survived the shuffle (uptime in supervisor.log > ~1400 s, or a clean epoch finish)?
   Confirm via `events.jsonl` that an `epoch_000016` reached past selfplay/shuffle into
   `train`/`evaluate`/checkpoint. If yes → **fix confirmed**; log new epoch + eval trend
   (watch for the now-diversified epoch-16+ `mean_turns`) and stop.
2. **If it STILL crash-loops on epoch 16** (new `*-epoch_000016.tmp` dirs keep appearing,
   resume_from still epoch_000015, watchdog stop.json still `free_ram_gb < 8`): my RAM
   estimate was off or there's another peak. FALLBACK OPTIONS, in order of preference:
   (a) cut `shuffle_worker_group_size`/`approx_rows_per_out_file` further (e.g. 4000);
   (b) trim selfplay residency so more RAM is free at shuffle time
   (`mcts_session_cache_max_states` 262144→131072, and/or `active_games` 256→192);
   (c) LAST RESORT — relax the watchdog `free_ram_gb` critical threshold from 8→5 in
   `scripts\watch_model1_resources.ps1` (find the threshold; verify it's not masking a
   real OOM — commit/virtual headroom looked fine: free_virtual ~17 GB at the spike).
   Don't do (c) blindly; (a)/(b) attack the cause, (c) just widens the guardrail.
3. **Housekeeping (report, don't act unless asked):** there are now ~18+ stale
   `shuffleddata\*-epoch_000016.tmp` dirs from the killed shuffles eating disk. The fix
   stops new ones; once epoch 16 completes cleanly they can be safely deleted, but I left
   them in place (capture-before-change). Mention to the user.
4. If the run has cleanly advanced several epochs, also recheck whether
   `epochs=60` completion (`supervisor_completed.flag`) is near and whether to raise it.

**Open items:** (i) fix unverified until an epoch-16 relaunch completes — VERIFY next run.
(ii) opening-diversity eval fix still unvalidated (needs epoch ≥16 to finish).
(iii) WER minidumps still not enabled (irrelevant here — this was never a native crash).

### 2026-05-29 01:33 — epoch-16 watchdog KILL from shuffle memory spike; bounded the window
- **What happened:** after the relaunch, trainer 52416 was **killed by the resource
  watchdog** at 01:29 (`status=stopping_trainer`, `free_ram=0.84GB`, `trainer_private=19.9GB`,
  triggers: free_ram<8, free_virtual<12, private>18). Exit code -1 = TerminateProcess (watchdog),
  NOT a Python/native crash. My eval-code change is NOT implicated (it died after selfplay,
  before eval).
- **Verified root cause (memory timeline from resource_watchdog.jsonl):** 52416 climbed gently
  to 7.4GB during epoch-16 selfplay, then **18s after the last selfplay shard** (entering the
  finalize/SHUFFLE phase) jumped to 19.9GB / 0.84GB free. Left a 0-byte
  `shuffleddata\...epoch_000016.tmp` (killed mid-shuffle). Cross-checked: prior trainers
  10672/47300/23156 peaked only 10–12.4GB through epoch 15 — so this is the **two-phase shuffle's
  peak RAM growing with the replay window** (2048 shards by epoch 16), tipping over the edge. It
  will WORSEN each epoch.
- **Why the breaker did NOT stop it:** ~23-min uptime per kill is not a "fast crash" (<180s) and
  <6/hour, so the breaker correctly let it relaunch. BUT this is a *slow, no-progress* loop
  (dies before saving epoch 16) — a real gap: the breaker won't catch it for ~2.3h.
- **FIX applied (config only, takes effect on next relaunch):**
  `shuffle_keep_target_rows 600000->300000`, `shuffle_worker_group_size 80000->40000` in
  `configs\dense_cnn_model1_scratch_64.toml`. Verified parses + no BOM. Watchdog left UNCHANGED
  on purpose, so the next run's shuffle peak cleanly shows if window reduction alone is enough.
  TRADE-OFF: smaller replay window = less data diversity per epoch (mildly slower learning) —
  flag for user to retune once RAM is understood.
- **NEXT / WHAT TO WATCH:** 53624 (old config) will die at its shuffle; supervisor relaunches
  with the reduced config. Then watch resource_watchdog.jsonl `trainer.private_gb` /
  `free_ram_gb` during the new run's shuffle (~20 min after relaunch, right after its 256
  selfplay shards finish):
    * SUCCESS = `epoch_000016.pt` checkpoint appears (survived shuffle+train) and free_ram stayed
      >8GB. Diversity verification (the original goal) can then proceed on the epoch-16 eval.
    * STILL KILLED = window cut alone insufficient. Then ALSO relax the watchdog: lower
      `MinFreeRamGb` 8->5 (it kills before the dangerous 0.8GB) by having the supervisor pass
      `-RestartWatchdog -MinFreeRamGb 5 -MaxTrainerPrivateGb 24` to the launcher (edit
      `Launch-Trainer` in supervise_scratch64.ps1, then RESTART the supervisor so it reloads),
      and/or cut `shuffle_keep_target_rows` further (e.g. 200000).
- **Eval-diversity verification is STILL pending** — blocked behind getting a clean epoch past
  the shuffle. Once epoch 16 completes, cluster `evaluation\epoch_000016\*.hxr` by byte size
  (old deterministic = ~3 clusters).


### 2026-05-29 01:06 — controlled relaunch to activate eval diversity; hit + fixed a BOM bug; breaker worked
- **Why:** user asked to relaunch once epoch 15 finished so the opening-diversity eval fix
  (`opening_temperature=0.6, opening_moves=8`) and `epochs=60` (which only load on a fresh
  process) take effect. Epoch 15 was complete (its eval had run). Killed trainer PID 10672
  at 01:02 so the supervisor would capture → bump `resume_from` → relaunch. (Resuming from
  `epoch_000015.pt` regenerates epoch 16 selfplay cleanly, so killing mid-epoch-16 was safe.)
- **BUG FOUND (fixed):** every relaunch insta-crashed (exit 1, uptime 0s) with
  `tomllib.TOMLDecodeError: Invalid statement (at line 1, column 1)`. Root cause: the
  supervisor's `Set-ResumeFrom` wrote the config with PowerShell 5.1 `Set-Content -Encoding
  UTF8`, which **prepends a UTF-8 BOM (EF BB BF)**; tomllib rejects a BOM. Verified by reading
  the config's first bytes (were `EF BB BF`).
- **Circuit breaker WORKED:** after 3 consecutive fast crashes it wrote `supervisor_halted.flag`
  and STOPPED relaunching — exactly the design, no overnight burn. (The earlier kill of 10672
  had uptime 6220s so it correctly did NOT count as a fast crash.)
- **FIX applied:** added `Write-Utf8NoBom` (uses `UTF8Encoding($false)`) in
  `scripts/supervise_scratch64.ps1` and routed the config + pidfile + flag writes through it.
  Repaired the live config (stripped BOM; first bytes now `5B 6D 6F`). **Verified the config
  parses through the real `load_training_config`** (epochs=60, resume_from=epoch_000015.pt,
  eval 0.6/8) — the check I should have run originally (my ValidateOnly only grep-checked the
  injected lines, never tomllib-parsed them).
- **State now:** halt flag cleared; supervisor restarted **PID 14920**; trainer relaunched
  **PID 52416** resuming **epoch 16** with new code, alive past startup (config no longer
  BOMmed). Old PIDs 10672/44472 are gone.
- **STILL PENDING — verify eval diversity:** epoch 16 eval (~01:22) is the first with the
  diversity fix active. NEXT: cluster its eval `.hxr` records
  (`runs\...\evaluation\epoch_000016\*.hxr`) by byte size. Old deterministic eval collapsed to
  ~3 sizes (30/19/11). If still collapsed → `opening_temperature=0.6` is too low; raise it
  (e.g. 0.9–1.1) and/or `opening_moves`, then it re-activates on the next relaunch. If many
  distinct sizes → diversity confirmed.
- **GOTCHA for future:** NEVER write the config (or any file tomllib/`[int]` parses) with PS
  `Set-Content -Encoding UTF8` — it adds a BOM. Use `Write-Utf8NoBom` / `UTF8Encoding($false)`.


### 2026-05-29 00:41 — seed entry (from the interactive Claude session)
- **State (verified):** trainer **PID 10672** alive (CPU ~57,100 s and rising),
  supervisor **PID 44472** alive (CPU ~0 — correctly blocked in WaitForExit). **Epoch 14
  in progress**: `epoch_000014.pt` written 00:40:10, selfplay shard 00:40:09 (fresh).
  No `supervisor_halted.flag`, no `supervisor_completed.flag`, **0** crash dumps, **0**
  fault signatures in `trainer.20260528_231854.err.log` (only benign Triton/torch warnings).
- **Supervisor adopted the run at 00:36:48** (see supervisor.log) — it has NOT yet had to
  relaunch (the original trainer from 23:18:54 is still the live one). Therefore the
  **opening-diversity eval fix and the bumped `epochs=60` have NOT activated yet** — they
  take effect on the first relaunch (crash or clean stop). The original process loaded
  `epochs=30` in memory and will run to epoch 30, exit cleanly; the supervisor will then
  relaunch toward 60 (now with diverse eval).
- **SealBot eval trend so far** (all best-50ms, 64 games, deterministic until fix activates):
  | epoch | wins | losses | mean_turns |
  |------:|-----:|-------:|-----------:|
  | 9 | 0 | 64 | 54.0 |
  | 10 | 0 | 64 | 58.375 |
  | 11 | 0 | 64 | 58.375 |
  | 12 | 0 | 64 | 60.875 |
  | 13 | 0 | 64 | 59.625 |
  Note epochs 10 & 11 are byte-identical (58.375) — **corroborates** the deterministic-eval
  diagnosis. 0 wins is expected this early; watch game length and (post-fix) win rate.
- **Crash history (from HANDOFF):** two prior native self-terminations (epoch 9→10 boundary;
  mid-epoch-11) with no Python traceback/dump. This instrumented build has since cleared BOTH
  points (ran 9→14 clean), so the fault looks intermittent/non-deterministic, not a hard
  deterministic bug. If it recurs, the supervisor will capture artifacts; root-cause from those.
- **NEXT RUN, do:** (1) read newest LOG entry; (2) check flags first; (3) check newest
  checkpoint/shard mtime to confirm advancement + record current epoch; (4) record the eval
  trend (append new epochs); (5) confirm whether a relaunch has occurred (supervisor.log has
  a RELAUNCH line, OR a new `trainer.<stamp>.err.log` appeared) — if so, note that the eval
  fix + epochs=60 are now live and watch for diversified `mean_turns`. (6) If halted/stalled,
  follow the decision tree above and write a full diagnosis + proposed fix here.
- **Open item:** WER minidumps require an elevated one-time run of
  `scripts/setup_python_minidumps.ps1` (HKLM write). Until done, crashes capture logs +
  PYTHONFAULTHANDLER stderr but NO faulting-module dump. Remind the user if a crash recurs
  without a usable signature.


### 2026-05-29 ~10:15 — Investigation: late-game "random" play = SEARCH BUDGET + POLICY HEAD (read-only analysis)
**Full write-up + re-runnable scripts:** `analysis/policy_diffuseness_investigation.md`,
`analysis/phase{1..4}_*.py`. **Read-only:** no config/model/supervisor/checkpoint changes,
no training launched; CPU inference on deleted *copies* of epoch_000009/epoch_000021 .pt.

**Question:** does scratch_64 (64ch/4block) "play randomly as games lengthen" because of
model size, training, or environment? **Answer: it is a coupled SEARCH-BUDGET + POLICY-HEAD
problem; value head and representation are fine.** These must be improved **before** launching
a bigger-model training run — a bigger net on the current 128-sim search would waste capacity.

**Evidence (all verified from logged selfplay NPZ + light CPU inference):**
- **Value head is NOT the problem:** it gets *more* accurate late-game (sign-acc 0.96–1.0,
  corr ~0.95 past move 40) and clearly learned (coin-flip @ep9 → near-perfect @ep21). So the
  trunk's RF/capacity suffices to *judge* positions; representation/env is not the cause.
- **Post-MCTS visit target is SHARP and sharpens late** (eff~3, top1~0.7) — but this is an
  **under-exploration artifact**, not resolution (see search budget below).
- **Raw policy head is diffuse and underfits its own sharp target, worst in long games**
  (move 100–200: KL(target‖pred)=3.1, best move falls to raw rank 3, eff~85 predicted vs ~4
  target). So the head can't represent sharp targets — a policy-pathway capacity limit.
- **MCTS search budget is far too low for the action space (first-class cause):** legal moves
  grow 388→1426 with move#, but search **visits only ~6–12 distinct moves (<1% of legal)**
  late-game; sims/legal ≈ 0.09 (≈1 sim per 11 legal moves); visited-set is *below* the 32
  widening cap, so 128 sims (+ PUCT exploitation) is the active limit. The visit target is a
  sharpened echo of the (diffuse) prior — search rarely overrides it (best move = raw #1–3),
  so it can't generate targets that *improve* the policy. KataGo/AlphaZero use ~400–1600 sims.

**Architecture imbalance (param audit):** 12.2M params but trunk (the actual reasoner) is only
451K; policy head `Conv(64→2,1x1)+Linear(3362→1681)` = 5.65M (2-ch bottleneck + single linear,
no nonlinearity) plus a duplicate 5.65M `opp_policy_head`. Mis-allocated, not "too small."

**Cause ranking:** (1 co-primary) MCTS sims too low for action space; (1 co-primary, coupled)
policy-pathway capacity/architecture; (3) trunk depth/RF — secondary; (4) env/representation —
ruled out by the value head. **#1 and #2 bootstrap each other — fix both before scaling model.**

**Suggested order for a future (deliberate) effort — NOT done here:** raise sims to ≥400
(800–1600) and re-tune widening/Dirichlet; rebalance the policy head to fully-convolutional
(`3x3 conv→ReLU→1x1 conv→1 logit/cell`, cheaper + more expressive); THEN test a trunk bump
(128ch/8block). Discriminate with late-game KL(target‖pred), best-move rank, and %legal-visited.
