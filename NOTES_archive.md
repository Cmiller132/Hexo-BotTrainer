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

### 2026-05-29 ~18:?? — STARTING THE FRESH 96×6 RUN: committed/pushed, SealBot-bootstrapping for the NEW arch, configs+supervisor ready
**TL;DR:** User approved committing, pushing, and STARTING the fresh 96×6+P7+512-sim run. In progress — this entry
will be finalized with PIDs/paths once the run is live. **The scratch_64 (64×4) run is being abandoned by design**
(the new arch is shape-incompatible); a backstop monitor should now watch the NEW run dir, NOT scratch_64.

**Step 1 — COMMIT + PUSH (done):** all implementation work committed on `impl/scratch64-phase1-opt`
(`ddaab60`) and pushed to origin. Branch:
https://github.com/Cmiller132/Hexo-BotTrainer/tree/impl/scratch64-phase1-opt ; PR-create:
https://github.com/Cmiller132/Hexo-BotTrainer/pull/new/impl/scratch64-phase1-opt . Data excluded per .gitignore.

**Step 2 — FRESH SealBot bootstrap for 96×6+P7 (in progress):** the old `bootstrap_sealbot_..._converted.pt` is
64×4/FC — shape-incompatible, NOT reusable. The original `scripts/bootstrap_dense_cnn_classical.py` was deleted
and used a since-rewritten API (`SampleBuffer`/`CompressedSample`/`DenseSampleWindow` all gone). I wrote a
current-API replacement: **`scripts/bootstrap_dense_cnn_sealbot.py`** — plays SealBot(best)-vs-SealBot games
(SealBot confirmed live at `E:\SealBot`, ~2 s/game, 0.03 s/move), records one-hot policy on SealBot's move +
outcome value via the canonical `sample_from_state`→`finalize_game_samples`→`write_selfplay_npz` chain (so D6/
schema handling is exactly self-play's — no poisoning risk), then supervised-prefits the 96×6+P7 model with the
production `DenseCNNTrainer` step and saves a `{model_state,optimizer_state,epoch}` checkpoint. **Smoke-proven**
end-to-end (305 samples → loss 12.6 → checkpoint loads strict into a fresh 96×6+P7 model). Full bootstrap running
in background: 50000 samples, 8 prefit epochs → `runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt`
(scratch shards under `runs/_bootstrap_work_96x6/`, transient). Note: the P7 checkpoint is ~25 MB (vs the old
146 MB) because P7 deleted the ~11.3 M FC-head params.

**Step 3 — configs (done):** `configs/dense_cnn_model1_target_96x6.toml` reviewed against scratch_64; ALL
stability guardrails carried: shuffle group/bucket 8000/8000, replay window 300000, P8 cache cap 131072,
policy-surprise weights, opening-diversity eval (temp 0.6/8 moves), `calibrate=true` (OOM-guarded batch select).
Changes flagged: `initialize_from` → the new bootstrap; `batch_size` 128→256 (measured to fit at 32% VRAM;
calibration still picks from [64,128,192,256]); lr kept 0.001 (no scheduler exists; fixed Adam — proven value);
eval `virtual_batch_size` left 0 (trustworthy first-run SealBot strength; raise to 8–16 later if eval wall is
slow — the Phase-3 lever). The 64×4/FC checkpoints will NOT load (fresh run).

**Step 3b — supervisor (done):** `scripts/supervise_target_96x6.ps1` = copy of `supervise_scratch64.ps1` with
default config → target, process-match `scratch_64`→`target_96x6`, run-name default updated. Same guardrails
(crash-artifact freeze, resume-from-latest, circuit breaker, no-progress(3) guard). Launcher + watchdog are
config-agnostic (run-name-targeted, `MinFreeRamGb=4` floor) so they carry over unchanged. PS-parse verified.

**Steps 4–5 (pending bootstrap completion):** smoke-test one short cycle via
`configs/dense_cnn_model1_target_96x6_smoke.toml` (scaled: 64 sims, 6 games, lowered shuffle gate, 2 eval games)
to prove bootstrap-load→selfplay→shuffle→train→checkpoint→eval, THEN launch
`supervise_target_96x6.ps1`. Will record supervisor PID, trainer PID, run dir, config path, bootstrap-confirmation,
and first-shard advancement here.

### 2026-05-29 (backstop wakeup, host clock read 14:03) — run still DOWN by design; Phase-4 gate now COMMITTED; NO action
**TL;DR:** Backstop check. Run remains intentionally stopped at `epoch_000022.pt` (optimization stage) —
NOT a crash/halt/stall/completion. Nothing has advanced since the ~10:17 stop. No state-changing action;
did NOT restart the supervisor (same reasoning as every cycle since ~11:03 — this is not a breaker halt
with a fixable bug; the GPU is intentionally freed and the structural/model-change work lives on a branch,
not activated on the run). **New since last cycle:** the Phase-4 model-change-gate work is now *committed*
locally as `ddaab60` on branch `impl/scratch64-phase1-opt` (was uncommitted at the ~13:04 cycle).

**How verified (cross-checked, multiple signals):**
- **Flags:** `supervisor_halted.flag` and `supervisor_completed.flag` BOTH absent.
- **Liveness (2+ signals):** pidfile PIDs `supervisor.self.pid=45320`, `supervisor.pid=53664` (and prior
  child `48140`) all confirmed **DEAD** via `Get-Process`. Process scan shows NO trainer/supervisor/watchdog
  alive — only powershell tool shells + python pid **41584** (= the long-running `hexo_frontend.web`
  dashboard, up since 5/28 20:21, CPU ~19764 s; NOT the trainer — left alone).
- **Pidfiles stale & dead:** both mtime **08:08:38** — left by the supervisor killed at the stop. Unchanged.
- **supervisor.log tail:** last line still `[08:08:38] ADOPT existing trainer pid=53664 ... NOT modifying
  its config/process`. NO EXIT/RELAUNCH/CAPTURE after it → supervisor killed before any relaunch (documented
  stop order: supervisor first). Prior lines are the morning 07:23/07:46 watchdog-kill relaunch history.
- **No advancement (expected — nothing is up):** newest checkpoint `epoch_000022.pt` @ **10:07:41**
  (146,588,369 B); newest selfplay shard `epoch_000023_game_*.npz` @ **10:16:37** (leftover from the killed
  epoch-23 selfplay — regenerated on resume, harmless). ~3h47m idle = "stopped," not "stalled."
- **No crash:** newest `crash_artifacts/` dir = `20260529_074646` (morning history); `crashdumps/` EMPTY;
  newest err.log = `trainer.20260529_074646.err.log` (829 B, 07:47:22) — content confirmed benign (Triton
  "Failed to find CUDA" UserWarning + the non-writable-buffer torch warning; no Fatal/panic/Traceback/
  0xc0000005/STATUS_). No new err.log after the stop → external kill, not a fault.
- **Eval frozen at epoch 22:** newest `dense_cnn.evaluation.epoch_000022.json` @ 10:10:17, no epoch 23+.
  Trend (best-50ms wins/64, from prior cycles) 17=6,18=6,19=2,20=2,21=4,22=4 — plateau 2–6/64, mean_turns
  41→31. Unchanged. The opening-diversity fix (`opening_temperature=0.6`) has STILL not activated (needs a
  relaunch, which the supervisor — currently dead — would do).
- **Git:** branch `impl/scratch64-phase1-opt`, HEAD `ddaab60` = "dense_cnn perf + model-change gate:
  P0/A7/P8, M2+A1, parse-parallel, P7+96x6 target". Matches the Phase-4 LOG entry below.

**RESUME / NEXT-RUN POINT (unchanged):** Two divergent paths now exist, user's choice:
  (a) **Continue 64×4 scratch_64 lineage:** `checkpoints\epoch_000022.pt` (10:07:41) → resumes at **epoch
  23**. CAVEAT: config literal `[checkpoint] resume_from` = `epoch_000015.pt`; the supervisor overwrites it
  with the latest on launch, but a MANUAL launch would redo from 15 — set it first. Grinding more 64×4
  epochs will NOT move the win-rate plateau (see [[scratch64-policy-bottleneck]]).
  (b) **Fresh 96×6 / P7 / 512-sim run (the strength fix):** point launcher/supervisor at
  `configs\dense_cnn_model1_target_96x6.toml`. This is a FRESH random init — `epoch_000022.pt` is
  shape-incompatible and will NOT load. This is the recommended path per the optimization plan.

**Clock note:** host clock read 14:03 this wakeup, yet the LOG entries below carry ~14–17 timestamps for
work whose artifacts/commit already exist on disk. The wall clock on this host/scheduled-task env is
unreliable — I trust file mtimes + git + log contents over the absolute clock. State is unambiguous from
files regardless.

**NEXT CYCLE, do (in order):**
1. **First check whether the run is back up.** If a new trainer/supervisor is alive and advancing (newest
   `epoch_*.pt`/shard mtime within ~15 min, no flags) → user resumed; revert to the normal decision tree,
   log progress + new eval rows. Determine WHICH lineage: if `runs\dense_cnn_model1_target_96x6\` (or
   similar fresh output_dir) is being written, the 96×6 gate run started (fresh, epoch 1+); if
   `scratch_64` epoch 23+ appears, the 64×4 lineage resumed. Check the live config for cache cap 131072 +
   bucketing (PB/A7) + P7 head.
2. **If still down (expected):** re-confirm the same terminal signals (no procs, dead pidfiles, no flags,
   no new crash_artifacts/dumps, benign err.log) and log a one-line "still intentionally down." Do NOT
   restart the supervisor.
3. Only resume if the **user explicitly asks**: `-ValidateOnly` first, confirm no live supervisor (pidfile
   + supervisor.log), then launch (auto-injects `resume_from`=latest epoch_*.pt for the 64×4 path; the
   96×6 path is a fresh run with no resume).

**Open items (report-don't-act, unchanged):** (i) ~19 stale `shuffleddata\*epoch_000016*` .tmp dirs — safe
to delete (well past epoch 16). (ii) WER minidumps still not enabled (irrelevant — zero native crashes since
the morning shuffle-RAM fix). (iii) The decision between lineage (a) and (b) above is gated on the user; the
96×6 gate work is committed (`ddaab60`) and proven-ready (see Phase-4 entry) but no run launched.

### 2026-05-29 ~17:?? — PHASE 4 — MODEL-CHANGE GATE proven READY (P7 + 96×6 + 512 sims); fresh config written; NO run started
**TL;DR:** Landed the coupled model change as one coherent unit and proved it's sound without starting a run.
**P7 fully-conv policy head** replaces the FC head (`architecture.py`) — drop-in: outputs `(N, BOARD_AREA)` so
loss / inference flat-index / Rust contracts are unchanged; removes the ~11.3M FC-head params (64×4 model:
12M→973K; 96×6: **2.12M**). New fresh-run config `configs/dense_cnn_model1_target_96x6.toml` (96ch, 6block,
search_visits 512). **Verified:** parses; `plugin.build_model` builds it; P7 heads confirmed fully-conv (no
`Linear`); inference-optimized (conv-folded) head matches training head to ~1e-8; a **512-sim self-play search
runs end-to-end** (4 roots, exactly 512 visits each, valid actions); a **forward+AMP+optimizer train step fits
in 12 GB VRAM at bs256 = 3875 MiB (32%)**, bs128 = 1992 MiB; FP16 forward safe at 96×6 (Phase 3.5). Full suite
**154 passed**. Run still STOPPED at `epoch_000022.pt` — **no training launched** (the new config is a fresh run;
the 64×4/FC checkpoints are shape-incompatible and won't load).

**R1 (VRAM) is a non-issue, better than the audit feared:** bs256 at 96×6 uses only ~32% of the 12 GB card
(the small P7 head + modest trunk keep activations low), so the bs256→128 calibration fallback exists but isn't
needed. Verify the calibration log on the first epoch as a formality.

**P7 details:** `PolicyHead` is now `HexConv2d(C→C,3×3) → ReLU → Conv2d(C→1,1×1) → flatten` (per the plan's
"3×3 Conv→ReLU→1×1 Conv→1 logit/cell"). Both `policy_head` and `opp_policy_head` use it. `HexConv2d` keeps the
trunk's hex adjacency; `_replace_remaining_hex_convs` folds it to a plain conv for CUDA inference. The spatial
training target `policyTargetsNCHW (N,1,41,41)` flattens to the same `(N,1681)` the head emits — no replay/D6/
schema change. **D6 augmentation untouched** (it operates on the compact→dense expansion, head-agnostic).

**Config notes:** fresh `output_dir`/`name` (does not clobber scratch_64); no `resume_from`/`initialize_from`
(fresh random init); kept the RAM-safe shuffle window (300k) and P8 cache cap (131072); eval
`virtual_batch_size` left at 0 (default) with a commented hint to enable the Phase-3 eval-latency lever after a
SealBot-strength validation. `calibrate=true` retains the OOM-guarded batch auto-select.

**To start the fresh run (user decision — NOT done here):** point the supervisor/launcher at
`configs/dense_cnn_model1_target_96x6.toml`. This abandons the 64×4 scratch_64 lineage by design.

### 2026-05-29 ~16:?? — PHASE 3.5 — FP16 confirmed (already on, safe @96×6); A3/A4 measured NOT worth it; **parse parallelization = real 18% self-play win**
**TL;DR:** Measured each Phase-3.5 target before building. **FP16 is already enabled** (`config.training.amp=true`
→ self-play + eval inference) and gives a **2.3–2.9× forward speedup**, numerically safe at both 64×4 and the
96×6 target (softmax max-diff ~1e-7 / ~5e-7). **A3 (state-on-node) is NOT worth building** — select-replay is
~2% of a move even at 512 sims (widening keeps trees bushy not deep). **A4 (on-GPU gather) is marginal** — the
gather/softmax is already on-device. The real main-thread cost (once the forward is FP16) was the **Rust prior
parse (~1650 ms/move, ~39%)**; I **parallelized it across rayon** (per-row, independent) → **~1650 ms → 75 ms
(22×)**, cutting the self-play move **7100 ms → 5849 ms (~18%)** at amp=False, **byte-identical** (equiv digest
unchanged `d9d0aa9a`, 0 invariant failures). Full suite **154 passed**. Run still STOPPED at `epoch_000022.pt`.

**Important correction to earlier Phase-2/3 numbers:** those benches used `amp=False`, so they *overstated* the
production forward by ~2.4×. Production self-play (amp=true) forward ≈ 2000 ms/move, not ~4900 ms. With FP16 on +
this parse fix, a production move ≈ forward 2000 + encode 350 + parse 75 + tree/orch ~500 ≈ **~2.9 s/move** vs the
amp=False sequential-parse ~7.1 s — and the parse fix is the part this session newly captured.

**Measurements:**
- *FP16* ([`analysis/forward_fp_bench.py`](analysis/forward_fp_bench.py)): forward fp32→fp16 — 64×4: 2.3–2.5×;
  96×6: 2.5–2.9×. Numerics (max abs diff, b256): 64×4 value 1.5e-4 / policy-softmax 2.4e-7; 96×6 value 2.7e-4 /
  softmax 5.1e-7. **FP16 is on and safe for the Phase-4 96×6 model** — no code change needed.
- *A3 sizing*: at 512 sims/64 roots the select (incl. O(depth) replay) is ~90–177 ms/move (~2%) — A3 would save
  ~2% even at the target. **Not built.**
- *Parse*: per-row decode + `finalize_model_priors` sort over ~32k leaves/move was ~1650 ms serial on the GIL
  thread. Parallelized with `par_iter` (precomputed per-row prior offsets; rows are independent and output stays
  row-ordered → identical bytes). New `parse_seconds` stat exposed in batch diagnostics.

**Shipped:** parse parallelization in `evaluate_model1_state_refs` (`mcts_eval.rs`) + `parse_seconds` stat. A4 and
A3 deliberately not built (measured low-value). **Net Phase-3.5 self-play win: ~18% (parse) on top of the
already-present 2.4× FP16 forward.** New artifact: `forward_fp_bench.py`.

### 2026-05-29 ~15:?? — PHASE 3 (§4.2) — MEASURED the lever; shared-tree atomics & root-parallel are BOTH the WRONG lever; shipped a safe vbatch knob
**TL;DR:** Applied the measurement-first discipline to §4.2 before building the high-risk shared-tree atomics —
and the data says **don't build them.** Single-game (eval/play) latency is bound by the NUMBER of NN forwards,
not by selection. The right lever is simply the single-tree **virtual_batch_size** (fewer, fatter forwards):
measured **~3.4× latency cut** (512 sims: 317 ms→94 ms/move from vb4→vb32). Both alternatives the plan named are
**measured inferior**: (a) root parallelism (N shallow trees, sum visits) collapses quality — splitting sims
across independent shallow trees loses depth; (b) §4.2 shared-tree atomics produce the *identical* leaves/forward
and *identical* virtual-loss window as raising vbatch, just selecting them on T threads (selection is cheap), so
they're strictly dominated by the vbatch knob at zero concurrency risk. **Shipped** an opt-in eval-only
`virtual_batch_size` (default 0 = unchanged) instead of the atomics. Run still STOPPED at `epoch_000022.pt`.

**Measurements (CUDA, 64×4 net; fresh positions, real forwards):**
- *Single-game latency vs vbatch* ([`analysis/single_game_latency_bench.py`](analysis/single_game_latency_bench.py)):
  128 sims: vb4 80 ms → vb32 24 ms (3.3×); 512 sims: vb4 317 ms → vb32 94 ms → vb64 88 ms (3.4×). avg forward
  batch tracks vbatch exactly (a single tree fills the batch — no root-blocking, Q3 not limiting here).
  *(First pass was contaminated by re-searching one cached position → fixed to advance the game so each move is a
  fresh position.)*
- *Quality vs lever at equal latency* ([`analysis/root_parallel_bench.py`](analysis/root_parallel_bench.py),
  agreement of the played move with a strong 1024-sim/vb4 reference): single vb4/512s = 100%; single vb32/512s =
  73% @214 ms; **root4 (4×128s) = 27% @233 ms; root8 (8×64s) = 9% @100 ms.** Root parallelism is both slower-per-
  quality AND far worse — confirms "scales worse per core." (Numbers use a *random* net, which exaggerates argmax
  sensitivity; a trained net's sharper policy is more stable, so these are a pessimistic quality bound — validate
  the chosen vbatch on the real net via a SealBot eval.)

**Why §4.2 shared-tree atomics are dominated (the key argument):** T threads each selecting `vbatch` leaves from
one shared tree = T·vbatch leaves/forward with a T·vbatch virtual-loss window — *exactly* what single-tree
`virtual_batch_size = T·vbatch` produces, with the same quality. The only thing §4.2 adds is parallel *selection*,
which Phase-2 measured at ~2-3% of move time. So §4.2 buys ~nothing over the vbatch knob while costing a high-risk
shared-tree concurrency model (atomic stats, false-sharing, virtual-loss-cancellation correctness). **Not built —
recommend not building it.** Root parallelism likewise not built (measured worse).

**Shipped (safe, opt-in):** `Model1EvalConfig.virtual_batch_size` (config.py) + plumb in `player.decide`
(player.py) — eval/play uses it when >0, else the calibrated self-play value (default 0 → **behavior
unchanged**). This exposes the measured latency lever so eval-phase wall (64 games played serially, ~4× at 512
sims) can be cut with an explicit, validated speed/quality trade. Config parses (default 0; override 16 OK); full
suite **154 passed**.

**Recommendation:** to cut eval-phase wall time at the 512-sim target, set `[evaluation] virtual_batch_size` to a
small value (8–16 conservative, 32 for max speed) and **validate strength with a real SealBot eval** before
trusting the win-rate (the quality cost is real but likely smaller on the trained net than the random-net bound
above). Do NOT invest in §4.2 atomics or root parallelism. **Deploy 50 ms budget:** even vb64 at 512 sims is
~88 ms on this net, so the plan's "decouple deploy-sims from train-sims" remains the right call for real-time play.

**No deviation harm:** I implemented the *measured* right lever instead of the planned §4.2 atomics, per the
"flag if the plan premise doesn't hold" guidance. New analysis artifacts: `single_game_latency_bench.py`,
`root_parallel_bench.py`.

### 2026-05-29 ~14:?? — PHASE 2 (M2 + A1) IMPLEMENTED & MEASURED — A1 CORRECT but **NO throughput gain** (key finding); STOP for checkpoint
**TL;DR:** Implemented M2 (thread-safe eval cache) and A1 (select↔eval pipeline) on `impl/scratch64-phase1-opt`.
Both build clean, full suite **154 passed**, A1 is **correct, deterministic, and search-quality-neutral**. BUT a
direct serial-vs-A1 benchmark shows **A1 yields ~0 throughput gain** (35.4 → 35.1 pos/s) at the current
64×4 / 128-visit / 256-root config. Root cause measured: the work A1 overlaps (leaf *selection*) is only
**~2-3% of move time**; the move is dominated by the GPU forward + Python marshal + Rust encode/parse, none of
which A1 overlaps. The plan's "~29% GPU duty → fill it" was a self-play-*phase* metric (incl. Python
orchestration *outside* `session.run`), NOT the overlappable fraction inside search. **Stopping at the phase
boundary as instructed — recommend the user reconsider A1's priority before Phase 3.** Run still STOPPED at
`epoch_000022.pt`; nothing run-state touched. Rust `.pyd` rebuilt into the worktree (built via `cargo build
--release` + copy, since no venv for `maturin develop`).

**M2 — eval cache + stats `Rc<RefCell>` → `Arc<Mutex>` (`mcts_eval.rs`, `mcts.rs`):** mechanical, behavior-
preserving (uncontended `Mutex` == `RefCell` in single-thread use). Added `lock_cache`/`lock_stats` helpers;
converted all 14 borrow sites (the `ENCODE_SCRATCH` thread-local RefCell is unrelated, left as-is). Now
`Send + Sync`, satisfying A1's prerequisite. Single mutex is fine because cache access is at eval boundaries
only (never the hot select loop); a sharded lock is deferred until profiling shows contention. **Verified:**
full suite 154 passed; the MCTS trajectory fingerprint
([`analysis/mcts_equiv_harness.py`](analysis/mcts_equiv_harness.py)) is deterministic with 0 invariant failures
(`mcts_baseline_pre_a1.json`, digest `9c78bcc3`).

**A1 — select↔eval pipeline (`mcts.rs run_searches_to_targets`):** rewrote the serial barrier into a 2-stage
software pipeline via `std::thread::scope`: the current leaf batch is evaluated on the GIL thread while the
*next* batch is selected on a scoped worker (internally rayon). Trees have a single mutator at all times
(select during the scope; backup after join), so **no tree lock**; virtual loss (already applied at selection)
is the sync primitive. Extracted `select_leaf_batch` + `apply_eval_backups` helpers.
- **Correctness (verified):** deterministic for a fixed seed (digest `d9d0aa9a`, identical across runs); exactly
  128 visits per root every move; `visit_policy` sums to 1, no negative weights → 0 invariant failures; 154
  tests pass. NOT bit-identical to the serial barrier (expected — the next batch is selected before the current
  is backed up, extending the virtual-loss window by ~1 batch). **Search-quality-neutral, proven:** on *aligned*
  positions ([`analysis/mcts_aligned_harness.py`](analysis/mcts_aligned_harness.py), advance by NN-prior argmax
  so both builds traverse identical positions) serial-vs-A1 = **75.8% visit-argmax agreement, TV-dist mean 0.16**
  — *identical* to the natural vbatch window sensitivity within one build (vb4-vs-vb1 = 75.0%, vb4-vs-vb2 = 75.8%).
  I.e. A1's perturbation = the perturbation the project already accepts by running vbatch=4 vs true-sequential.
  It shrinks at higher (target 512) visits where the search is more converged.
- **Bug found & fixed during verification:** my first loop keyed termination off the prefetch making progress;
  on a *narrow* tree (move 0 has a single legal action) the prefetch is blocked by the in-flight pending edge,
  returns empty, and the loop terminated early (visits=1 not 128). Fixed: terminate off `needs_visits`; when the
  prefetch is starved by a narrow tree, fall back to a synchronous select after backup (overlap is best-effort,
  correctness is not). Re-verified all moves hit 128 visits.
- **THROUGHPUT (the key negative result):** serial **35.4 pos/s / 7224 ms-move** vs A1 **35.1 pos/s / 7291 ms-move**
  ([`analysis/a1_throughput_bench.py`](analysis/a1_throughput_bench.py), 256 roots, 128 visits, vbatch 4, CUDA,
  apples-to-apples — only the Rust loop differs). Env-gated `HEXO_MCTS_TRACE` instrumentation in
  `run_searches_to_targets` shows per timed move: **eval ≈ 7000 ms, select ≈ 170 ms (overlapped, hidden),
  backup ≈ 80 ms**, and `scope_wall ≈ eval` (overlap works — select fully hides). So the overlap is real but the
  overlappable slice (select) is ~2-3% of move time. The ~7000 ms "eval" is forward+marshal (~4900 ms,
  `evaluator.call1`) **+ Rust encode/parse/finalize (~2000 ms, on the main thread inside the eval call — NOT
  overlapped by A1).**
- **Why this matters for the plan:** the real self-play levers are (1) the **GPU forward** (→ FP16/TensorRT — the
  plan's §4.6 "only remaining lever"), and (2) the **Rust encode + payload parse (~2000 ms/move)**, which a
  *finer* pipeline (encode/parse batch i+1 while forward batch i) or A4 (on-GPU marshal) would overlap — NOT leaf
  selection. Note A3 (state-on-node) would *shrink* select further, making A1's overlap even less valuable. So
  A1 as specified is not the throughput lever the plan assumed at either the current or the target config.

**Recommendation for the checkpoint:** A1 is correct, safe (no measurable slowdown), quality-neutral, and is the
structural prerequisite §4.2 tree-parallelism reuses — so **keep it**, but **re-prioritize**: the measured
bottleneck says the next high-ROI work is (a) FP16/TensorRT on the eval forward and (b) overlapping/cutting the
Rust encode+parse, ahead of more MCTS-CPU pipelining. M2 stands on its own (needed for any threaded eval work).
Awaiting greenlight for Phase 3 (§4.2) or a pivot.

**Build/test status:** `cargo check` + `cargo build --release` clean (6 pre-existing dead-code warnings); full
`tests/` suite **154 passed**; MCTS equivalence harness deterministic, 0 invariant failures. New analysis
artifacts: `mcts_equiv_harness.py`, `mcts_aligned_harness.py`, `mcts_aligned_diff.py`, `a1_throughput_bench.py`
(+ `mcts_baseline_pre_a1.json`, `mcts_post_a1_fixed.json`, `aligned_serial.json`, `aligned_a1.json`).
**Deviation from plan:** Q3 deliberately NOT folded into A1 — analysis showed the budget-loop `break` only fires
on full *root* blockage (rare at vbatch=4), the deep-collision case backs up an existing node inline (no abort),
and a naive break→continue risks an infinite loop; Q3's real value is for A5 (large vbatch), where it belongs.

### 2026-05-29 ~13:04 — BACKSTOP: run still DOWN by design (optimization stage); NO action taken
**TL;DR:** Confirmed the run remains intentionally stopped — unchanged since the deliberate ~10:17 stop
for the optimization/investigation stage. NOT a crash/halt/stall/completion. No state-changing action
taken; did NOT restart the supervisor (same reasoning as every cycle since ~11:03: this is not a breaker
halt with a fixable bug — the GPU is intentionally freed and Phase-1 opt work is on a branch, not yet
activated on the run). Resume point still `epoch_000022.pt` → epoch 23.

**How verified (cross-checked, multiple signals, at 13:03–13:04):**
- **Flags:** `supervisor_halted.flag` and `supervisor_completed.flag` BOTH absent.
- **Liveness (2 signals):** `Get-Process` → both pidfile PIDs **DEAD** (supervisor 45320, trainer 53664).
  Process scan for `train_model|supervise_scratch64|watch_model1|resource_watchdog` → only my own
  NonInteractive tool shell (pid 3032). No trainer / supervisor / watchdog alive.
- **Pidfiles stale & dead:** `supervisor.self.pid=45320`, `supervisor.pid=53664` (note BOM in the file
  → shows as `﻿53664`), both mtime **08:08:38** — left by the supervisor killed at the stop. Unchanged.
- **supervisor.log tail:** last line still `[08:08:38] ADOPT existing trainer pid=53664`. NO
  EXIT/RELAUNCH/CAPTURE after it → killed before any relaunch (documented stop order: supervisor first).
- **No advancement (expected — nothing is up):** newest checkpoint `epoch_000022.pt` @ **10:07:41**
  (146,588,369 B); newest selfplay shard `epoch_000023_game_*.npz` @ **10:16:37** (from the killed
  epoch-23 selfplay — regenerated on resume, harmless). ~2h47m since last activity = "stopped," not
  "stalled."
- **No crash:** newest `crash_artifacts/` dir = `20260529_074646` (morning watchdog-kill history);
  `crashdumps/` = EMPTY; newest err.log = `trainer.20260529_074646.err.log` (829 B, mtime **07:47:22**,
  unchanged from prior cycles that confirmed it benign Triton/torch warnings). No new err.log after the
  stop → external kill, not a fault.

**RESUME POINT (unchanged):** `checkpoints\epoch_000022.pt` (10:07:41) → resumes at **epoch 23**
(loader = top-level `payload['epoch']`+1). Pointer `data\checkpoints\...latest.txt` agrees. CAVEATS
(still true): (a) config literal `[checkpoint] resume_from` says `epoch_000015.pt`; the supervisor
overwrites it with the latest on launch, but a MANUAL launch would redo from 15 — set it first. (b) If
Phase-1/structural work changes architecture (channels/blocks), `epoch_000022.pt` won't load → fresh run.

**What changed since last cycle:** nothing on the RUN. The ~13:?? entry below records that Phase-1
optimizations (P0 NPZ load-once, PB/A7 cuDNN bucketing, P8 cache cap, Q5 comment fix) were IMPLEMENTED
on branch `impl/scratch64-phase1-opt` (current git branch), build+tests green, but **not committed to the
shared review branch and NOT activated on the run** — the config change (P8 cache cap → 131072) only
takes effect on the next supervisor relaunch, which the user controls. So the run state is identical to
the ~12:03 cycle; only the worktree carries new (uncommitted) opt code.

**SealBot eval trend (frozen at epoch 22, best-50ms wins/64):** 17=6, 18=6, 19=2, 20=2, 21=4, 22=4 —
bouncing 2–6/64, no upward slope; mean_turns falling 41→31. This plateau is exactly what motivated the
stop + the [[scratch64-policy-bottleneck]] investigation. Grinding more epochs of THIS config won't move
the win rate — the Phase-1 opts are speed/foundation work; the strength fix is the gated MODEL-CHANGE
GATE (P7 fully-conv head + 96×6 trunk + 512 sims) per `analysis/optimization_plan.md`.

**NEXT CYCLE, do (in order):**
1. **First check whether the run is back up.** If a new trainer/supervisor is alive and advancing
   (newest `epoch_*.pt`/shard mtime within ~15 min, no flags) → user resumed; revert to the normal
   decision tree, log progress + new eval rows (epoch 23+). If architecture changed (Phase 1/gate
   landed), `epoch_000022.pt` may not load (fresh run) — note it. Also check the live config for the
   P8 cache cap (131072) and whether `pad_to_buckets` / bucketing (PB/A7) is active.
2. **If still down (expected):** re-confirm the same terminal signals (no procs, dead pidfiles, no
   flags, no new crash_artifacts/dumps, benign err.log) and log a one-line "still intentionally down."
   Do NOT restart the supervisor.
3. Only resume if the **user explicitly asks**: `-ValidateOnly` first, confirm no live supervisor
   (pidfile + supervisor.log), then launch (auto-injects `resume_from`=latest epoch_*.pt).

**Open items (report-don't-act, unchanged):** (i) ~19 stale `shuffleddata\*epoch_000016*` .tmp dirs —
safe to delete (well past epoch 16). (ii) WER minidumps still not enabled (irrelevant — zero native
crashes since the morning shuffle-RAM fix). (iii) Phase-1 opt branch `impl/scratch64-phase1-opt` is
uncommitted to the shared review branch and unactivated; the structural/strength work (A1, §4.2,
MODEL-CHANGE GATE) is gated on the user.

### 2026-05-29 ~13:?? — PHASE 1 IMPLEMENTED (P0, PB/A7, P8, Q5) on branch `impl/scratch64-phase1-opt`; run still STOPPED
**TL;DR:** Implemented the low-risk Phase-1 foundation from
[`analysis/optimization_plan.md`](analysis/optimization_plan.md) on a NEW branch
`impl/scratch64-phase1-opt` (cut from `review/scratch64-optimization-stage`, carrying its
working-tree changes). Build + tests green; measured before/after where feasible. **STOPPED before
all structural work** (A1, M2, §4.2, the MODEL-CHANGE GATE / 96×6 / P7) as instructed. **Did NOT
touch the live run** — resume point still `epoch_000022.pt`; supervisor not started; no checkpoint
changed. NOT committed to the shared review branch (kept on the impl branch).

**Environment gotcha (important for any future test run):** the installed `hexo_models` is a PEP-420
namespace package whose `__init__.py` (in site-packages) hardcodes the dense_cnn source root relative
to ITS OWN location → `import hexo_models.dense_cnn` resolves to a STALE COPY at
`site-packages\dense_cnn\python\…`, NOT the worktree. A plain `pytest` tests the stale copy. To test
worktree edits you MUST prepend the worktree pkg dir:
`PYTHONPATH=E:/Hexo-BotTrainer/packages/hexo_models/python` (the worktree `__init__.py` then points its
relative roots back at `packages/hexo_models/dense_cnn/python/...`). This mirrors what
`start_model1_training.ps1` does. All test/bench numbers below were taken with that PYTHONPATH set.

**P0 — NPZ load-once (`trainer.py`):** added `_materialize_npz()` (decompress each of the 6 batch
arrays once per shard) and switched both `train_passes` and `_run_validation` to slice the in-RAM dict
instead of `data[KEY][start:stop]` per batch (NpzFile re-decompresses the whole array on every access).
`_batch_from_npz` is unchanged. **Measured** ([`analysis/p0_loadonce_microbench.py`](analysis/p0_loadonce_microbench.py)
on real shard `…epoch_000016/train/data00000.npz`, 7936 rows, bs128): data-prep **13475 ms → 234 ms =
57.7×**; batches verified **byte-identical** (`torch.equal` over every batch & key) ⇒ training numerics
unchanged. Confirms the plan's −~95 s/epoch (model-independent).

**PB / A7 — cuDNN cold-start fix (`inference.py`):** chose the production A7 form (NOT a bare
`benchmark=False`): keep `cudnn.benchmark=True`, but pad every forward batch up to a power-of-two
bucket (`_bucket_batch_size`, `_pad_to_bucket` at the single `_forward_device_inputs` chokepoint), so
the evaluator presents ≤11 shapes (1,2,…,1024) instead of the measured ~900. Padded rows are sliced
off; correctness rests on per-sample independence (conv / **eval-mode** BatchNorm running stats / FC —
no cross-batch op), **proven** by `test_padding_does_not_leak_into_real_rows` (byte-identical real rows
regardless of padding content) + on/off equivalence + bucket-math tests
([`tests/test_dense_cnn_inference_bucketing.py`](tests/test_dense_cnn_inference_bucketing.py), 4 pass).
**Measured** cold autotune ([`analysis/a7_autotune_bench.py`](analysis/a7_autotune_bench.py), 64×4,
32 distinct input shapes, fresh process each): OFF **26.85 s** (~0.84 s/shape) vs ON **5.13 s** (32
inputs → 6 buckets). Extrapolated to the production ~900 shapes that's ~755 s → matches the plan's
~830 s tax; bucketing caps it at ~9 s/process regardless of shape variety, killing the recurring
cold/relaunch thrash AND the >10-min-hang risk. *Tradeoff:* ≤2× compute on under-filled forwards in
steady state; at the current 64×4 config self-play is CPU-bound (GPU ~29% duty) so this hides, and it
is the right form for the 96×6 target (bigger kernels prefer autotune). Not measured: steady-state
self-play wall-clock delta (needs a full epoch). If a regression shows there, `benchmark=False` is the
fallback (set `pad_to_buckets=False` + flip the flag).

**P8 — eval-cache cap (`configs/dense_cnn_model1_scratch_64.toml`):** `mcts_session_cache_max_states`
262144 → 131072 with the **audit-corrected** justification (HOST RAM, ~−340 MB; NOT VRAM — it's an
`Rc<RefCell<HashMap>>` of `Arc` priors on CPU). Throughput-neutral (cache fills ~190k/262k and inserts
== unique states). I did **NOT** re-widen the replay window (`shuffle_keep_target_rows` stays 300000) —
that re-grows the shuffle peak that crash-looped epoch 16, so it's a separate higher-risk decision; the
freed RAM merely enables it later. Verified the TOML still parses via `parse_model1_config` → value
131072. (Config change takes effect only on the next supervisor relaunch, which the user controls.)

**Q5 (done) / Q3 (skipped):** fixed two stale `Rc`→`Arc` comments that mis-describe the threading
contract A1/§4.2 depend on — `mcts_eval.rs:341` and `mcts_tree.rs:144` (the live code already uses
`Arc<RustEvaluation>`; only the comments lagged). Comment-only, zero behavioral change; `cargo check
--manifest-path packages/hexo_models/Cargo.toml --features python` clean (6 pre-existing dead-code
warnings, unrelated), no `.pyd` rebuild needed. **Q3 deliberately SKIPPED** — "don't abort a root's
batch on a pending collision" is a *behavioral* change to MCTS batch-fill and is coupled to A5/A1;
per the plan it belongs in the A1 PR, not this low-risk pass.

**Build/test status:** full `tests/` suite **154 passed** (worktree PYTHONPATH). `cargo check` clean.
New artifacts (untracked): `analysis/p0_loadonce_microbench.py`, `analysis/a7_autotune_bench.py`,
`tests/test_dense_cnn_inference_bucketing.py`.

**Deviations from plan:** (1) PB implemented as the A7 bucketing form (per instruction "production-safe
form, not a hack"), not `benchmark=False`. (2) P0 measured 57.7× data-prep (vs plan's 20–23×) because
the microbench isolates pure data-prep on a real shard. (3) Discovered the stale-copy import gotcha
above — prior pytest baselines in this repo may have tested the installed copy, not the worktree.

**STOPPED before structural work** (A1 select↔eval, M2 cache-Send, §4.2 tree parallelism, P7/96×6/512)
as instructed — checkpoint with the user before starting those.

### 2026-05-29 ~12:35 — FINAL CHECK of the optimization plan: **GO**; P7 folded into mandatory; 1 error fixed (doc-only)
**TL;DR:** Rigorously audited [`analysis/optimization_plan.md`](analysis/optimization_plan.md) against
the profiling, MCTS review (+ microbench JSON), and diffuseness reports before implementation. **Verdict:
GO.** The feasibility math is arithmetically sound (every number re-derived below), the dependency
ordering is correct, and the mandatory set is well-motivated. **Doc-only changes** — NO production code,
config, model, supervisor, or checkpoint touched; run still stopped at `epoch_000022.pt`.

**Task 1 — P7 folded into MANDATORY (user decision):** moved the fully-conv policy head from "out of
scope / separate workstream" into the mandatory set and a new **Group F**, tied explicitly to the
model/sims change. It now lands as **one fresh run** with channels 64→96, blocks 4→6, and sims 128→512
(a "MODEL-CHANGE GATE" added to the §2 sequence). Rationale per diffuseness §7/§8: head ⊕ trunk ⊕ sims
are coupled — a 96×6 trunk feeding the current FC head still trains on prior-echoing targets, wasting the
capacity. P7 is **speed-neutral** (472↔473 ms [P]) so it changes **none** of the §4 feasibility seconds;
schema-compatible (target `policyTargetsNCHW` is already spatial); it also deletes the ~11.3 M FC-head
param bloat. Updated §1, §2, §5, and the bottom line.

**Task 2 — audit results:**
- **Feasibility math: VERIFIED, no arithmetic errors.** Re-derived: model FLOPs `(6·96²)/(4·64²)=3.375×`
  ✓; sims 4× ✓; self-play serial `816+192+344+200+129=1681 s` ✓ → pipelined `max(816, 566)≈850 s` ✓;
  training `619+0+150≈770 s` ✓; epoch totals baseline 1141≈1143 ✓, naive 3913 (~65 min) ✓, optimized
  1966 (~33 min, 1.74×) ✓; deploy `184·4·2.5/8≈240 ms` (~5× over 50 ms) ✓; current-workload cumulative
  `384+250+60+116=810 s` (1.4×) ✓.
- **Dependency ordering: CORRECT.** A1 → §4.2 (reuses A1's virtual-loss/leaf machinery, no conflict —
  A1 uses per-tree ownership, §4.2 adds shared-tree atomics on top). Q3 → A5. A3 must precede 512-live
  (steepest-growing CPU term). **M2 is a genuine HARD prerequisite for A1** — confirmed the eval cache is
  `Rc<RefCell<…>>` (`mcts_eval.rs:110`), not even `Send`, so A1's worker threads literally cannot share
  it as-is. Nothing in Phase 1/2 depends on anything later.
- **ERROR FOUND + FIXED (doc-only): P8 conflated RAM with VRAM.** The plan claimed lowering the eval
  cache "buys headroom for the 96×6 activations / VRAM." It does **not** — the cache is host RAM (CPU
  `Rc<RefCell<HashMap>>` of `Arc<RustEvaluation>` priors), so it's irrelevant to the 12 GB VRAM risk.
  Corrected P8's justification to host-RAM-for-replay-window-rewiden (the window was cut 600k→300k for
  RAM), and fixed two §2 references.
- **VRAM risk R1: addressed, and BETTER than the plan claimed.** Verified `calibrate=true` +
  `performance.py:194/241` wrap every candidate batch (training, inference, self-play) in a
  `try/except → _is_oom` guard that drops OOM candidates and selects the largest viable batch. So the
  bs256→bs128 fallback at 96×6 is **automatic**, covering the self-play `inference_batch_candidates`
  (incl. 1024) too. Documented this in R1.
- **Eval phase: §4.2 mandatory confirmed.** Verified SealBot eval plays its 64 games **serially**
  (`evaluation.py:90` `for game_index in range(...)`) on the single-root `player.decide` path — so the
  ~4× sims blow-up is real and tree parallelism is genuinely needed for the eval phase, not only deploy.
- **Minor (left as-is):** PB's "−830 s / identical steady speed" is [M] at 64×4, an extrapolation to
  96×6 — the plan correctly hedges by making A7 (bucketed shapes + benchmark=True) the production form.

**Task 3 — VERDICT: GO.** **First implementation step:** ship **Phase 1** — P0 (NPZ load-once,
`trainer.py`), PB (`cudnn.benchmark=False`, `inference.py:77`), P8 (cache cap → 131072, config). All
free/near-free, no dependencies, individually testable, and they expose the GPU floor before the model
grows. **Concretely start with P0** (measured −95 s/epoch, ~5-line correctness-neutral change, covered by
`test_dense_cnn_pipeline.py`/`performance.py`). **Mandatory ordered sequence:**
1. P0, PB, P8 (Phase 1, days, independent)
2. Q3 + Q5 → **A1/P5** (keystone) ⊕ **M2** (cache shard, prereq) → A2 + A3 + A4 + A5 + A7 (CPU-cut tuning under the A1 PR)
3. **§4.2** tree parallelism (reuses A1 machinery; root-parallel eval as the zero-risk first step)
4. **MODEL-CHANGE GATE:** P7 + channels 96 + blocks 6 + sims 512 as one fresh run (checkpoint shape
   changes → `epoch_000022.pt` won't load)
**Plan corrections made (doc-only):** P7 → mandatory (Group F + gate + §5 + bottom line); P8 RAM/VRAM
fix; R1 auto-fallback note. Nothing applied to code/config/run — implementation is the next, gated step.

### 2026-05-29 ~12:20 — OPTIMIZATION PLAN written (planning only; NO production code changed, run still down by design)
**TL;DR:** Consolidated the profiling fix list (P0–P8) + the MCTS review (A1–A7, §4.2 tree parallelism,
M1–M4) into one dependency-ordered plan of attack oriented on ONE goal: **make 512 MCTS sims feasible
alongside a 96ch×6block model** (current: 128 sims, 64ch×4block). Written to
[`analysis/optimization_plan.md`](analysis/optimization_plan.md). **Read-only w.r.t. the run** — no
config/checkpoint/supervisor change, no training launched; resume point still `epoch_000022.pt` → epoch
23. The ONE tiny code touch (allowed by the user): a `TODO(P3, deferred)` comment at
[`replay.py:753`](packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/replay.py:753) marking the
uncompressed-scratch optimization as intentionally deferred (P3 itself NOT implemented).

**The two compute axes, quantified:**
- sims 128→512 = **4×** on sim-scaling buckets (evaluator/tree/encode); orchestration is sim-independent.
- model 64ch×4b→96ch×6b = **3.375×** trunk FLOPs `(6·96²)/(4·64²)` → forward + training step ×3.4;
  ~2.25× activation memory (VRAM risk at bs256 on the 12 GB card). Worst single component (eval forward)
  = 4×3.4 = **13.6×** raw.

**Feasibility verdict (the headline math):**
| epoch | training | self-play | eval | shuffle | total |
|---|---:|---:|---:|---:|---:|
| baseline 128/64×4 [P] | 479 | 373 | 173 | 116 | **1143 s (19 min)** |
| target 512/96×6 NAIVE | 916 | 1681 | ~1200 | 116 | **~3913 s (~65 min)** |
| target 512/96×6 OPTIMIZED | 770 | 850 | ~230 | 116 | **~1966 s (~33 min, ~1.7×)** |

- **A1 (pipeline select↔eval) is the keystone** — converts self-play from serial `sum(GPU+CPU)=1681 s`
  to `max(GPU 816, CPU 566)≈850 s`. Motivated by the measured root-saturation at ~8 roots [M].
- After opts the **critical path is raw 96×6 GPU compute** (~1435 s of the ~1966 s epoch is the trunk
  fwd/bwd in self-play+training). Only further lever: FP16/TensorRT + torch.compile (unmeasured).
- **Still falls short on the 50 ms DEPLOY budget**: even with §4.2 tree parallelism (~8×), one 512-sim
  96×6 move ≈ ~240 ms ⇒ ~5× over 50 ms. Recommendation: **decouple training-sims (512) from
  deployment-sims** (standard AZ/KataGo) — high sims only needed at train time to generate good targets.

**Mandatory set for the goal:** P0, PB/A7, A1/P5, A2+A3, §4.2 tree parallelism, M2 (cache shard) + Q3/Q5
(concurrency prereqs). **Headline cumulative steady speedup on TODAY's workload ≈ 1.4× (1143→~810 s)
plus −830 s on every cold/relaunch epoch [M].** Full per-item effort/deps/risk + ordered sequence in the
plan. Nothing applied — gated on the user; this is the plan, not the implementation.

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


===== ARCHIVED 2026-05-30 (target_96x6 TRT/forced-playouts/frontend session) =====

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

### 2026-05-30 ~04:04 UTC (00:04 EDT) — **RUN ADVANCING; bounce cycle has SETTLED; forced-playouts k=2 fix is LIVE; first 96×6 epoch_000002 checkpoint + first SealBot eval are IMMINENT (epoch-2 selfplay 511/512 done)** — NO ACTION

**Verdict:** run is **advancing normally and the bounce cycle from the 03:08 entry has settled.** Trainer **pid
32750** (the one the concurrent agent launched at the 02:55Z bounce to deploy the opening-diversity fix) has now
run **uninterrupted for ~1h08m** with NO further bounce, is at the **very end of epoch-2 selfplay (511/512
games)**, and the **first-ever 96×6 `epoch_000002.pt` checkpoint + first SealBot eval JSON are imminent.** No
flag, no crash, no stall, RAM fine. I took **NO action** (advancing branch). The 03:08 worry about the bounce
cycle pushing `no_progress` to 5/5 is **defused for now** — see #2.

**State found / how verified (cross-checked WSL pgrep + file freshness + supervisor_wsl.log + events.jsonl +
err-log fault scan + RAM sampler + live config + git; no single signal trusted):**
- **WSL liveness CONFIRMED:** `pgrep` shows supervisor `supervise_target_96x6_wsl.sh` **pid 307** (+child 395,
  the FIXED set_resume code from ba6cd09) AND trainer **pid 32750** (`...train_model ...target_96x6.toml`). These
  are the SAME pids the 03:08 entry reported — **no relaunch since 02:55Z**, so the bounce cycle has paused.
- **Advancing (freshest signal):** newest shard `epoch_000002_game_000441.npz` mtime **04:03:55Z** vs my check
  **04:04:08Z** — **~13 s old.** Definitively NOT a stall. Epoch-2 selfplay: **481 shards on disk, max game
  index 511** (games_per_epoch=512) → selfplay is essentially DONE; shuffle+train+checkpoint next.
- **supervisor_wsl.log:** last lines are `EXIT pid=420 code=143 uptime=2946s` (02:55:11Z) → `noProgress=1/5
  (latest epoch 1)` → `RELAUNCH` → `resume_from -> epoch_000001.pt (start epoch 2)` → `LAUNCH pid=32750`
  (02:55:14Z). **NO EXIT line after 02:55:14Z** → 32750 alive the whole time. (The 420 code=143 EXIT is the same
  benign concurrent-agent bounce the 03:08 entry root-caused — `_watch_fp_epoch2.sh`/`_wait_k2_games.sh` deploying
  the opening-diversity fix. NOT a fault.)
- **CORRECT RESUME (ba6cd09 holds):** events.jsonl `load_checkpoint` → `epoch: 1` from `epoch_000001.pt` with a
  real `train_state` (NOT bootstrap/epoch 0) → `calibrate_performance` finished (294s, meets_target=false @ 7.2
  pos/s — the known calibration probe metric, NOT live TRT rate) → `run_epochs` → `stage_started epoch_000002`.
- **TRT FP16 adopted** for 32750: out.log `[trt_backend] adopted TRT FP16 (build 51.2s, argmax_match=0.9688,
  value_err=0.0219)`. argmax 0.9688 is a touch below the ~0.99 of the cleanest builds but within per-build fp16
  variance; fail-loud would have crashed on a gate failure, so it's live (not torch fallback).
- **FORCED-PLAYOUTS k=2 FIX IS LIVE:** live config `configs/dense_cnn_model1_target_96x6.toml` line 89
  `forced_playout_k = 2.0` (+ the explanatory block lines 83-88: forced visits guarantee each materialized root
  child ≥k visits, policy-target pruning removes them from the trained target, the PLAYED move samples un-pruned
  visits → openings diversify). This is the `target96x6-selfplay-opening-collapse` memory fix, committed as
  `424d186` (Rust MCTS forced playouts + policy-target pruning) + `cff52b9` (enable k=2 on the run, 03:16:54Z).
  Trainer 32750 launched 02:55:14Z by the bounce whose explicit purpose (per 03:08 root-cause) was deploying this
  fix → **32750 has k=2 ACTIVE.** Epoch 2 is the first epoch with diversified openings.
- **No crash / no flag / RAM fine:** no `supervisor_halted.flag` / `_completed.flag`; no `crashdumps/` dir;
  newest err log `trainer.20260530_025514.err.log` fault-sig scan (Fatal Python error|Current thread|panicked|
  stack backtrace|Traceback|0xc0000005|access violation|STATUS_|SIGSEGV|SIGABRT|CUDA OOM|CUDA error) = **0
  hits**, tail = only the 2 known-benign warnings (inference.py:270 non-writable-buffer + architecture.py:225
  TracerWarning). `watch_wsl.jsonl` latest 04:04:55Z **free_ram_gb 21.3, flat.**
- **Concurrent agent has gone QUIET:** newest `scripts/_*.sh` mtime is `_wait_k2_games.sh` 03:04Z; newest commit
  on `bench/inference-backends-wsl` is `cff52b9` 03:16:54Z. Both are >45 min stale vs now (04:04Z), and no bounce
  since 02:55Z → the opening-diversity deploy is DONE and the agent is letting epoch 2 run to completion.

**Why no action:** up, resuming correctly, TRT engaged, k=2 fix live, selfplay 511/512 done, RAM fine, no flag,
no crash, no stall. Nothing to fix/relaunch (supervisor owns relaunch; hard rules forbid me relaunching/killing
/2nd-supervisor anyway).

**Still open / next-step instructions for next watcher:**
1. **THE milestone is now ONE step away — go check for it FIRST:** `checkpoints/epoch_000002.pt` (currently only
   `epoch_000001.pt` + bootstrap exist) AND the **first-ever** `diagnostics/dense_cnn.evaluation.epoch_*.json`.
   When they appear: report **wins/losses/mean_turns** — this is the first 96×6 strength number AND the first
   with diversified openings (k=2). **Baseline to beat = scratch_64's 2-6 wins/64 vs SealBot best-50ms.** If
   epoch_000002.pt exists, the breaker `no_progress` will have reset to 0/5 (good).
2. **no_progress watch (the one tension, now LOW risk):** breaker at **1/5** as of the 02:55 EXIT. It only climbs
   on a relaunch that yields NO new epoch checkpoint. With the concurrent agent quiet and 32750 at 511/512
   selfplay, epoch 2 should checkpoint BEFORE any further bounce → no_progress resets to 0. Only worry if you
   find no_progress at 3-4/5 AND the agent has resumed bouncing AND still no epoch_000002.pt (then it's a bounce
   landing before each checkpoint — heads-up note only, don't act). At 5/5 the supervisor writes
   `supervisor_halted.flag` even though nothing crashed; if THAT happens, it's a benign breaker trip (clear flag
   + restart supervisor only if you're confident no real fault — check err log fault-sig scan first).
3. **Re-derive liveness via WSL** (wsl.exe is at `C:\WINDOWS\system32\wsl.exe`; `command -v wsl.exe` returns it —
   the `/mnt/c/Windows/System32/wsl.exe` path FAILED from git-bash, use the lowercase-system32 one or `command
   -v`). Cmd: `wsl.exe -e bash -lc "pgrep -af 'supervise_target_96x6_wsl|train_model'"`. Current: sup **307**
   (+395), trainer **32750** (point-in-time). A NEW trainer pid + a fresh `RELAUNCH`/`resume_from ->` in
   supervisor_wsl.log = a normal bounce/crash relaunch — verify the new events.jsonl `load_checkpoint` shows
   `epoch: N` (NOT 0/bootstrap); if it EVER shows epoch 0 the resume regressed.
4. **code=143 is STILL benign-by-default on this run** (intentional `_bounce_trainer.sh`/`_restart_supervisor.sh`
   bounces). ONLY treat a 143 as a real fault if there's NO matching `scripts/_*.sh` edit near the EXIT time AND a
   fault signature appears in the err log (zero so far). A clean SIGTERM (143) with healthy RAM and no fault-sig
   is a bounce, not a crash.
5. **Decision tree unchanged:** advancing → log only (as here). Halted (flag) → FIRST check it wasn't a benign
   no_progress=5/5 breaker trip from bounces (see #2) before deep root-causing; else root-cause from flag +
   crash_artifacts + err.log + any .dmp, fix if safe, clear flag, restart supervisor. Completed (flag) → report
   final eval, ask re raising loop.epochs. Stalled (live trainer, no new shard/events >25 min, no flag) → capture
   err/events tails + flag a hang. Do NOT start a 2nd supervisor (307 is live), do NOT kill/relaunch the trainer.

### 2026-05-30 ~03:08 UTC (23:08 EDT) — **RUN ADVANCING; the recurring code=143 is INTENTIONAL trainer bounces by a CONCURRENT agent (opening-diversity work), NOT a fault** — NO ACTION

**Verdict:** the run is **advancing normally** (trainer pid 32750 live, TRT FP16 adopted, epoch-2 selfplay writing
shards in real time, RAM flat ~23 GB free, no flag, no crash, no stall). The big thing I resolved this run: the
**code=143 (SIGTERM) deaths the 02:09 entry flagged to "watch for recurrence" ARE recurring — but they are
DELIBERATE trainer bounces by a concurrent agent/user actively iterating on this run, NOT an ~hourly native
fault.** I took **NO action** (advancing branch + a concurrent agent is mid-work — do not interfere). The prior
worry ("a real ~hourly native/WSL fault to root-cause") is **closed: benign**.

**ROOT CAUSE of the code=143 SIGTERMs (cross-checked exit codes + uptimes + helper-script mtimes + script bodies
+ WSL boot history + dmesg; no single signal trusted):**
- The trainer is killed by hand-rolled bounce scripts in `scripts/_*.sh` (all UNTRACKED, created live this
  session). `_bounce_trainer.sh` does `kill -TERM "$(cat trainer_wsl.pid)"` (SIGTERM → 128+15 = **143**, no
  Python traceback, no fault dump — exactly the clean signature we saw). `_restart_supervisor.sh` `pkill`s the
  supervisor+trainer.
- **mtime ⇄ death correlation is exact:**
  - `_bounce_trainer.sh` mtime **01:51** ⇄ trainer **446 EXIT 01:51:25 code=143**. Script comment: bounce at the
    epoch-1 boundary "to pick up the new live-pos/s selfplay code." → intentional redeploy, not a crash.
  - `_restart_supervisor.sh` mtime **02:04** ⇄ the **02:06 supervisor restart** (sup 410→307) the 02:09 entry
    already attributed to the resume-fix reload. Confirmed: WSL `last` shows distro systemd restarts at
    22:05–22:06 local; the VM kernel itself stayed up since 20:49 (uptime 2:18 vs `last` boots — normal WSL2:
    systemd session restarts under a persistent shared kernel).
  - trainer **420 EXIT 02:55:11 code=143** ⇄ `_watch_fp_epoch2.sh` (02:57) + `_wait_k2_games.sh` (03:04): a
    bounce to deploy an **opening-move-diversity fix**. `_wait_k2_games.sh` explicitly waits for epoch-2 games to
    measure "realized opening-move diversity (.hxr)" — i.e. the `target96x6-selfplay-opening-collapse` work is
    LIVE right now.
- **Ruled OUT as causes:** OOM (RAM 22.9–23.6 GB free, flat; Linux OOM = SIGKILL/137 anyway), the supervisor
  (it only `wait`s on the trainer (line 98) and its RAM watchdog only LOGS, never kills (line 82)), WSL
  cron/at/oomd/pkill-loop (none exist — checked `crontab -l`, `/etc/cron.d`, `atq`, `pgrep oomd`), and a Windows
  scheduled task / native killer (none; only native python is the dashboard pid 53832). Fault-sig scan
  (Fatal Python error|Current thread|panicked|stack backtrace|Traceback|0xc0000005|access violation|STATUS_|
  SIGSEGV|SIGABRT|CUDA out of memory) across every `trainer.*.err.log` = **0 hits**.

**CONCURRENT AGENT IS ACTIVE — do NOT interfere.** A second worker is driving this run from Windows: an
interactive WSL login shell (pid 363) + `_wait_k2_games.sh` (pid 36388) parented to `/init` (a `wsl.exe ...`
call from the Windows side, started ~03:04). It is bouncing the trainer to deploy fixes (live-pos/s selfplay,
then opening-diversity). The hard rules already forbid me relaunching/killing/2nd-supervisor; the active
concurrent work makes that doubly true. My role this run = confirm state + log only.

**Current LIVE state (verified 03:03–03:08 UTC):**
- **Supervisor pid 307** (+child 395) LIVE (running the FIXED set_resume code). **Trainer pid 32750** LIVE:
  755% CPU, etimes 802s (matches launch 02:55:14Z), TRT FP16 adopted (`[trt_backend] adopted TRT FP16 (build
  51.2s, argmax_match=0.9688, value_err=0.0219)` — argmax 0.9688 is a touch below the ~0.99 of earlier builds
  but per-build fp16 numerics vary; not a gate failure since fail-loud would have crashed).
- **Resume is CORRECT** (the ba6cd09 fix holds): events.jsonl `load_checkpoint` → `epoch: 1` from
  `epoch_000001.pt` with a real `train_state` (NOT bootstrap/epoch 0). Then calibrate (the resumed launch
  re-ran calibrate ~294s) → `run_epochs` → `stage_started epoch_000002`.
- **Selfplay ADVANCING:** newest `epoch_000002_game_000153.npz` stamped 03:06:32Z vs my 03:08:36Z check (~2 min
  fresh). Definitively not a stall.
- **No flag / no crash / RAM fine:** no `supervisor_halted.flag` / `_completed.flag`; no `crashdumps/`; only the
  2 benign warnings (inference.py non-writable-buffer + architecture.py TracerWarning) in err logs;
  `watch_wsl.jsonl` ~22.9–23.6 GB free, flat.

**Still open / next-step instructions for next watcher:**
1. **Do NOT re-investigate code=143 as a fault.** On THIS run it = an intentional `_bounce_trainer.sh` /
   `_restart_supervisor.sh` bounce by the concurrent agent to redeploy code. Confirm by matching the EXIT
   timestamp to a `scripts/_*.sh` mtime and reading the script's comment. ONLY treat a 143 as suspicious if
   there is NO corresponding helper-script edit AND a fault signature appears in the err log (there were none).
2. **no_progress vs the bounce cycle (the one real tension to watch):** the breaker increments `no_progress`
   on every relaunch that yields no NEW epoch checkpoint; it's at **1/5** (last EXIT 02:55 logged
   `noProgress=1/5`). Each bounce + the supervisor re-running calibrate (~5 min) + epochs taking ~50 min means a
   bounce landed before the epoch-2 checkpoint keeps `no_progress` climbing. **If it reaches 5/5 the supervisor
   writes `supervisor_halted.flag` and stops** — even though nothing actually crashed. If you see no_progress at
   3–4/5 with the concurrent agent still bouncing, that's worth a heads-up note (the agent may want to let one
   epoch fully checkpoint, or the bounces should pause). Don't act on it yourself.
3. **STILL no Goal-#4 datapoint for the 96×6 arch.** `epoch_000001.pt` exists but there is NO
   `dense_cnn.evaluation.epoch_*.json` and NO `epoch_000002.pt` — epoch-1's SealBot eval was almost certainly
   cut off by the 01:51 bounce. First milestone remains: a clean `epoch_000002` checkpoint + the first
   `dense_cnn.evaluation.epoch_*.json` (report wins/losses/mean_turns; **baseline to beat = scratch_64's 2–6
   wins/64 vs SealBot best-50ms**). Until the bounce cycle settles, epoch 2 may not checkpoint.
4. **Re-derive liveness via WSL** (`wsl.exe -e bash -lc "pgrep -af 'supervise_target_96x6_wsl|train_model'"` +
   `last -x reboot`). Current: sup **307**, trainer **32750** (point-in-time — a new trainer pid + a fresh
   `RELAUNCH`/`resume_from -> ...` in `supervisor_wsl.log` = a normal bounce relaunch; verify the new launch's
   events.jsonl `load_checkpoint` shows `epoch: N` (NOT 0/bootstrap) — if it EVER shows epoch 0 again the resume
   regressed; re-check the config still has a real `resume_from` line + sup 307 is the live one).
5. **Decision tree unchanged:** advancing → log only (as here). Halted (flag) → root-cause from
   flag + crash_artifacts + err.log + any .dmp (but FIRST check it wasn't a bounce hitting the breaker — see #2),
   fix if safe, clear flag, restart supervisor. Completed (flag) → report final eval, ask re raising loop.epochs.
   Stalled (live trainer, no new shard/events >25 min, no flag) → capture err/events tails + flag a hang. Do NOT
   start a 2nd supervisor (307 is live), do NOT kill/relaunch the trainer, and be aware a concurrent agent may be
   actively bouncing it.

### 2026-05-30 ~02:09 UTC (22:09 EDT) — **RUN HEALTHY & CORRECTLY RESUMING after a resume-bug fix + supervisor restart (NEW sup pid 307 / trainer pid 420)** — NO ACTION

**Verdict:** the run is **advancing normally again** and — critically — now **resumes correctly** (loads the
latest epoch checkpoint, not the bootstrap). Between the last entry (01:04Z) and now, the prior agent found
and FIXED a real resume bug, then RESTARTED the supervisor to load the fix. I verified the fix is live, took
**NO action** (advancing branch), and logged this. The earlier "epoch 1 selfplay in flight" has become "epoch 1
trained + checkpointed; now resumed and starting epoch 2."

**What happened (full reconstruction — cross-checked supervisor_wsl.log + events.jsonl + normalized config +
git log + WSL pgrep + CPU/etimes + RAM; no single signal trusted):**
- **Trainer 446** (orig supervisor pid 410, launched 00:50:53Z) ran epoch 1, wrote the **FIRST 96×6 checkpoint
  `checkpoints/epoch_000001.pt` at 01:47:32Z**, then **EXITED code=143 (SIGTERM) at uptime 3632s (~60 min)**.
  Cause of the 143 is UNEXPLAINED (RAM was healthy ~22 GB; the supervisor has NO wall-clock breaker; watchdog
  only logs, never kills). Isolated so far (breaker logged crashesLastHour=1). **Watch whether code=143 recurs
  at ~60 min** — if it does, it's a real ~hourly native/WSL fault to root-cause; if not, it was likely the
  manual stop that accompanied the supervisor restart below.
- **THE RESUME BUG (now FIXED in git, commit `ba6cd09`, HEAD of `bench/inference-backends-wsl`, 02:02:16Z):**
  the WSL supervisor's `set_resume` guarded on `if "resume_from" in t:` — but the config's *comments* contain
  that substring, so it always took the "replace existing assignment" branch whose regex matched **no real
  assignment line** → it injected **nothing**. Every relaunch therefore restarted from the **bootstrap (epoch
  1)** instead of resuming. I confirmed the symptom directly: trainer **121867** (relaunched 01:51:27Z by the
  buggy supervisor 410) loaded `bootstrap_sealbot_prefit.pt` epoch 0 (`config.normalized.json` mtime 01:51:33Z
  showed `resume_from: null`) and was redoing epoch 1.
- **The fix landed two ways, both now in force:** (a) commit `ba6cd09` rewrote `set_resume` to use
  `re.subn`'s replacement **count** (`n==0` → insert after `[checkpoint]`, else replace); (b) the config
  `configs/dense_cnn_model1_target_96x6.toml` now carries a **real `resume_from = ".../epoch_000001.pt"` line**
  (line 157, shows as a dirty `M` in git — this is BY DESIGN; the supervisor rewrites it each relaunch). The
  real line alone neutralizes even the OLD buggy code (its regex now matches a real assignment).
- **Supervisor was RESTARTED to load the fix:** a NEW **supervisor pid 307** started **02:06:05Z** (bash parses
  functions once at startup, so the still-running old pid 410 would have kept using the buggy `set_resume` in
  memory regardless of the on-disk fix — restarting was necessary and correct). Supervisor 307 immediately
  logged `resume_from -> epoch_000001.pt (start epoch 2)` and launched **trainer pid 420**. (121867's ~02:05
  death = the supervisor restart killing the old tree; NOT a crash, and it has no EXIT line because old sup 410
  stopped logging when killed.)

**Current LIVE state (verified 02:06–02:09Z):**
- **Supervisor pid 307** (+ child 395) LIVE, running the FIXED code. **Trainer pid 420** LIVE: **205% CPU, 12.4%
  MEM, etimes 209s** (matches the 02:06:05Z launch).
- **CORRECT RESUME CONFIRMED:** events.jsonl (mtime 02:06:11Z) `load_checkpoint` → `status: loaded,
  checkpoint_ref=.../epoch_000001.pt, epoch: 1` **with a real `train_state`** (not the bootstrap, not epoch 0).
  Then `calibrate_performance` started — at ~209s calibrate it's just finishing; epoch-2 selfplay is imminent.
- **TRT FP16 still engaging** (trainer 121867's out.log showed `[trt_backend] adopted TRT FP16 (build 49.6s,
  argmax_match=0.9922, value_err=0.0232)` — note value_err 0.0232 is higher than 446's 0.0078 but within the
  ~0.04 fp16 tolerance from the bench entries; per-build numerics vary, acceptable). 420's out.log has no TRT
  line yet (engine builds on the first selfplay batch, after calibrate).
- **No crash / no flag / RAM fine:** 0 fault-signature hits in either new err log; no `supervisor_halted.flag`
  / `supervisor_completed.flag`; no `crashdumps/`; `watch_wsl.jsonl` ~21.9 GB free, flat.
- **512 epoch-1 selfplay shards on disk** (newest game_000218 mtime 02:04:31Z = 121867's last write before the
  restart). These will be overwritten/extended as trainer 420 proceeds.

**Why no action:** run is up, resuming correctly, TRT engaged, RAM fine, no flag, no crash, no stall. The resume
bug was already fixed AND the supervisor already restarted to load it — there is nothing left to fix or
relaunch, and the hard rules forbid me relaunching/killing the trainer or starting a second supervisor anyway.

**Still open / next-step instructions for next watcher:**
1. **Re-derive liveness via WSL** (`wsl.exe -e bash -lc "pgrep -af 'supervise_target_96x6_wsl|train_model'"`).
   Expect **supervisor pid 307** + a trainer (was **pid 420**). PIDs are point-in-time. The CURRENT supervisor
   is **pid 307 launched 02:06:05Z** — if you see a DIFFERENT supervisor start line after that, someone
   restarted it again (check why). A new trainer pid + a fresh `RELAUNCH`/`resume_from -> ...` line in
   `supervisor_wsl.log` = a normal supervised relaunch; **now that the fix is live, every relaunch should log
   `resume_from -> epoch_0000NN.pt (start epoch NN+1)` AND the new process's events.jsonl `load_checkpoint`
   should show `epoch: NN` (NOT epoch 0 / bootstrap).** If a relaunch EVER shows `epoch: 0` again, the resume
   regressed — re-check the config still has a real `resume_from` line and supervisor 307 is the live one.
2. **FIRST Goal-#4 milestone is STILL pending — no eval datapoint yet.** `epoch_000001.pt` exists but there is
   **NO `diagnostics/dense_cnn.evaluation.epoch_*.json`**. Either SealBot eval runs every N>1 epochs, or epoch
   1's eval didn't run / was cut off by 446's code-143 exit. **Next watcher: confirm eval cadence** (grep the
   config `[evaluation]` for an interval; check whether epoch 2 produces an eval JSON). The first eval is the
   first-ever 96×6 strength number — report wins/losses/mean_turns; **scratch_64 baseline to beat = 2–6 wins/64
   vs SealBot best-50ms**.
3. **Watch the code=143 question:** 446 died with SIGTERM at ~60 min. If trainer 420 (or a successor) also
   exits 143 around the ~60-min mark with no flag and a healthy RAM trace, that's a recurring native/WSL fault
   worth root-causing (read the relevant `trainer.*.err.log` tail + any `crash_artifacts/`/`crash.*.txt`). If
   420 simply runs long and completes epochs, treat 446's 143 as the one-off stop that triggered the
   supervisor restart.
4. **Throughput / epoch length:** epoch 1 took ~53 min of selfplay+train (446: calibrate done ~00:54Z →
   checkpoint 01:47Z). At that cadence judge stalls by "no new selfplay shard / no events progress >25 min
   WHILE trainer 420 is live", NOT by wall-clock. Calibration meets_target=false @ 10.3 pos/s is the known
   probe metric — live TRT selfplay is faster (~84 pos/s bench); don't treat 10.3 as the live rate.
5. **Decision tree unchanged:** advancing → log only (as here). Halted (flag) → root-cause from
   flag + crash_artifacts + err.log + any .dmp, fix if safe, clear flag, restart supervisor. Completed (flag) →
   report final eval, ask re raising loop.epochs. Stalled (live trainer, no new shard/events >25 min, no flag)
   → capture err/events tails + flag a hang. Do NOT start a 2nd supervisor (307 is live), do NOT kill/relaunch
   the trainer yourself.

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

### 2026-05-30 — Frontend review + replay-controls relocation (commit b90127b)

Reviewed the dashboard frontend with three parallel agents (web.py / app.js / html+css)
and landed a focused, browser-verified improvement pass. Headline UX change: the game
replay controls (|< < Play > >| + move slider) moved from the bottom dock to directly
UNDER the board (new `.replay-bar` inside `.board-panel`); dock now holds only Move
History. All element ids + the `.replay-buttons` wrapper preserved, so app.js (binds by
id) was untouched by the move. Also: web.py `_safe_stat` (no more 500s when selfplay
shards rotate mid-poll), `_read_json_file` one-retry (kills `selfplay.live.json` flicker),
static-file backslash path-traversal guard; app.js poll/replay lifecycle on screen
switch + error backoff + binding safety net + formatter/unit cleanup; CSS/a11y cleanups.

DASHBOARD PID CHANGED: the :8080 dashboard was restarted to serve the new code and is now
**pid 60696** (was 35520, was 52864). Same launch line / worktree PYTHONPATH. Verified in
a headless browser: controls render under the board, History screen loads, zero console
errors. `.claude/launch.json` (untracked) points the preview tool at the running :8080.

The two changes deliberately deferred (need live browser interaction to verify): app.js
`applyState` freshness-gate rewrite (C1) and board render-diffing on no-change polls (H1).
Flagged for a follow-up; both are real wins but risky to land blind on the live dashboard.

### 2026-05-30 — Dashboard QoL fixes + supervisor resume bug (commits 6c1bae3, ba6cd09)

QoL sweep (3 parallel agents, browser-verified on the live dashboard):
- web.py KEY DRIFT fix: `_selfplay_epoch_summary` read keys the producer never emits
  (samples_added/games/winner_counts/lengths/mcts_sims_per_searched_position) -> all
  null -> "null samples" in Epoch Progress. Now mapped to the real selfplay.py keys
  (samples_added<-effective_samples, games<-games_finished, sims_per derived). Output
  key names unchanged (app.js contract). Audited training/eval/calibration mappers too.
- app.js: null-safe display everywhere (asFinite treats null/""/undefined as missing but
  keeps real 0); humanized stage labels (epoch_000001 -> "Epoch 1", calibrate_performance
  -> "Calibrating", etc.); title= tooltips on truncated paths/ids.
- styles.css: metric values (Stage, Resources, learning-health, eval-trend, epoch-progress,
  state-summary, bot, training-summary) now WRAP instead of nowrap+ellipsis clip; long
  path/id fields keep ellipsis + hover title. min-width:0 where needed.
- Verified live: 0 null/undefined/NaN tokens, 0 clipped cards, "48867 samples | 17.3 pos/s".

CRITICAL supervisor bug found + fixed (ba6cd09): `set_resume` guarded on
`if "resume_from" in t` -- but the config COMMENTS contain that substring, so it always
took the substitute-existing-assignment branch, whose regex matched no real assignment
line, and silently injected NOTHING. => every relaunch restarted from the bootstrap
(epoch 1) instead of resuming. My epoch-boundary bounce was this run's first real
relaunch, which exposed it: the resumed trainer redid epoch 1. Fixed with re.subn (use
the replacement count, not a substring check; insert after [checkpoint] if count==0).
Verified by dry-run.

RECOVERY (user-approved): stopped the buggy-in-memory supervisor (410) + epoch-1-redo
trainer (121867), relaunched the FIXED supervisor. It correctly injected
resume_from=epoch_000001.pt; new trainer (pid 420 under supervisor 307/395) RESUMES
epoch 2 (normalized config confirms resume_from set) -- original epoch 1 salvaged, resume
now works for all future relaunches.

NOTE on throughput: epoch-1 real selfplay measured **17.3 search-pos/s** (512/512 games,
48972 positions, ~52 min) -- FAR below the tu8 microbench's ~84. TRT engaged (adopt line
present) but full-epoch 512-sim x 96x6 is much slower than the warm microbench predicted;
only marginally above the 10.4 calibration. The live dashboard now shows this TRUE number.
The 17.3-vs-84 gap is a real perf question worth a separate look (TRT actually helping at
scale? cache/replenishment-tail? CPU-bound sample finalization between searches?).

OPS: durable standalone dashboard is now **pid 53832** on :8080 (worktree PYTHONPATH).
Supervisor relaunched as background task; trainer pid 420. Avoid the Bash tool for `wsl`
(git-bash mangles /mnt/e -> exit 127); use PowerShell `wsl ...` or run_in_background.
