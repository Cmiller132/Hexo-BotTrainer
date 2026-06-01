# NOTES — dense_cnn Model 1 (target_64x4 cold-start) live run

Tight current-state / orientation doc. Verbose history is in `NOTES_archive.md`.
Keep THIS file tight. Backstop logs to `runs/dense_cnn_model1_target_64x4/diagnostics/backstop_log.md`.

## Last action
2026-05-30 ~19:54 UTC: **relaunched the 64×4 COLD-START at 256 sims on the
optimized build** (wiped the partial pre-opt epoch-1 shards for a clean curve).
Self-play is now **~2.2× faster (~62 pos/s in production**, was ~24–29) after two
byte-identical eval-path fixes — see "Self-play throughput optimizations" below.
96×6 stays **paused** (`epoch_000011.pt` preserved). Supervisor owns relaunch/resume.

## Self-play throughput optimizations (2026-05-30, branch bench/inference-backends-wsl)
Profiled with `scripts/_profile_selfplay.py`; full writeup `runs/selfplay_profile_findings.md`.
1. **Zero-copy plane buffer** (1.85×): the Rust→Python `inputs` handoff was a serial ~89 MB
   `PyBytes` memcpy/pass (was 46% of wall). Replaced with a buffer-protocol `#[pyclass]
   PlaneBuffer` viewed zero-copy by `torch.frombuffer`. `mcts_eval.rs`.
2. **f16 input transport** (~1.06×): `PlaneBuffer` now carries **f16** planes; Python upcasts
   f16→f32 **on-device** after the halved H2D, so TRT/forward are byte-for-byte unchanged.
   Gated byte-identical (`scripts/_fp16_input_gate.py`: 320/320 argmax, 0 err — TRT already
   downcasts its input to f16). Added `half` crate to hexo_models.
Self-play is now **forward-bound** (callback ≈89% NN forward). Next levers are NOT in the
pipeline: smaller net / INT8, or fewer sims (revisit post-epoch-1). 59 dense_cnn tests pass.

## What's running (ACTIVE)
Cold-start (random init, **no bootstrap**) self-play RL of **dense_cnn Model 1**:
- Architecture **64 ch × 4 blocks + P7 fully-conv policy head**, **256 MCTS sims/move** (settled at 256 on 2026-05-30 ~17:30 UTC: tried 128 but a random cold-start net + weak search made games meander ~3× longer — active games hit ~170+ plies, cancelling the pos/s win; 256 is strong enough to terminate games while ~2× faster than the 512 baseline. Sims is the real speed lever, not model size — see findings below).
- **Self-play temperature DECAY**: 1.0 at the opening → linearly to **0.2 by ply 30**, held 0.2
  after (per-game, per-ply; new `move_temperatures` vector through native MCTS). Opening still
  explores (temp 1.0 + Dirichlet + forced playouts); endgames are now played decisively.
- **forced_playout_k = 2** (KataGo forced playouts + policy-target pruning).
- **TensorRT FP16** forward, self-play only, fail-loud; engages in WSL only.
- **validation_fraction = 0.02** → per-epoch HELD-OUT loss; both train & val now log a
  **per-component breakdown** (`loss_components`: policy / value / opp_policy / stvalue_*).
- Bucketing (pad mult-16) + rolling replenishment (`games_per_epoch=512` > `active_games=256`). 60 epochs.

## Paths / branch / dashboard
- Run dir: `runs/dense_cnn_model1_target_64x4/` — `selfplay/ checkpoints/ diagnostics/`
- Config: `configs/dense_cnn_model1_target_64x4.toml` (cold start — no `initialize_from`)
- WSL supervisor: `scripts/supervise_target_64x4_wsl.sh`; **launch detached** with
  `scripts/_launch_verify_64x4.sh` via `wsl -d Ubuntu-24.04 -- bash /mnt/e/.../_launch_verify_64x4.sh`.
  GOTCHA: a plain `nohup ... & disown` dies on WSL session exit — must use **`setsid nohup ... < /dev/null &`**
  (the launch script does). Status: `scripts/_status_64x4.sh`.
- Native build: rebuild in WSL with **rustup cargo** (`export PATH=/root/.cargo/bin:$PATH`; the apt cargo
  is too old for lockfile v4). If a stale-rlib **E0461 ICE** appears, run a **full `cargo clean`** (pyo3/rayon
  rlibs from an older rustc must be rebuilt too). Reusable: `scripts/_rebuild_hexo_models_clean.sh`.
- Branch: **bench/inference-backends-wsl**. Dashboard: http://localhost:8080
- Diagnostic tool (run on demand): `scripts/_loss_decomp.py` — scores any checkpoints on a FIXED batch
  (the decisive "is the model actually learning" test). `scripts/_target_trend.py` — per-epoch target entropy + game length.

## Bootstrap / resume
- COLD START: first launch has neither `resume_from` nor `initialize_from` → `load(None)` → random init
  (verified: load_checkpoint status "initialized", checkpoint_ref null).
- Relaunch: supervisor injects `[checkpoint].resume_from = <newest checkpoints/epoch_*.pt>`.
  (The supervisor's "uses initialize_from → bootstrapped" log line is cosmetic; there is no prefit here.)

## Stability guardrails (WSL supervisor owns relaunching)
- Auto-relaunch on trainer exit + resume-from-latest. Single-instance lock `diagnostics/supervisor_wsl.lock`.
- Circuit breaker → `diagnostics/supervisor_halted.flag` on: 3 consecutive crashes <180 s, OR >6/hr, OR
  5 relaunches with no new epoch checkpoint. Completion → `supervisor_completed.flag`. Crash tails → `crash_artifacts/`.
- RAM watchdog → `diagnostics/watch_wsl.jsonl`; WSL 28 GB cap + Linux OOM are the hard backstop.

## Backstop decision tree
1. Terminal flags first: `supervisor_halted.flag`, `supervisor_completed.flag`.
2. Advancement: newest `checkpoints/epoch_*.pt` + `selfplay/epoch_*_game_*.npz` mtimes vs now;
   `dense_cnn.selfplay.live.json` (live pos/s); `events.jsonl` last stage.
3. `supervisor_wsl.log` tail (LAUNCH/EXIT/RELAUNCH/breaker/halt). WSL procs are invisible to Windows
   Get-Process — check liveness via `wsl -d Ubuntu-24.04 -- pgrep -af train_model` AND file freshness.
4. TRT engaged: grep `[trt_backend] adopted TRT FP16` in newest `trainer.*.out.log`.
5. **Halted** → root-cause (flag + newest `trainer.*.err.log` + `crash_artifacts/`). Safe fix → rebuild
   (full clean if ICE), clear flag, relaunch via `_launch_verify_64x4.sh`.
6. Learning trend: `epoch_*.json` `loss_components` (val especially) + `dense_cnn.evaluation.epoch_*.json`
   (wins/losses/mean_turns vs SealBot best-50ms).
- HARD RULES: don't kill/relaunch the trainer yourself (supervisor owns it); don't start a 2nd supervisor;
  capture artifacts before changing anything; a brief process gap during a relaunch is normal.

## Key findings (from the 96×6 investigation that motivated this run)
- **"Rising loss" was a MEASUREMENT ARTIFACT, not divergence.** On a FIXED held-out batch every later 96×6
  checkpoint was strictly better (policy CE 3.99→3.16, value CE 0.81→0.29). The reported epoch-loss rose
  because (a) forced-playout widening raised the policy-target entropy (0.22→1.27 nats → higher CE floor)
  and (b) policy-surprise resampling oversamples high-CE rows. → fixed by logging `loss_components` + a
  held-out val loss.
- **Flat SealBot winrate (~1/64) is GENUINE but early.** Eval is a fair greedy eval (argmax after 8 opening
  moves, no noise/forced-playouts). Limiter = **diffuse policy head** (model policy CE ≫ target entropy);
  value head is strong. Watch whether policy CE keeps dropping.
- **"Weird" self-play games = no temperature decay** (96×6 played temp=1.0 the whole game). → fixed here
  with the 1.0→0.2-by-ply-30 schedule.
- Self-play is forward/overhead-bound (GPU SM ~38%); **k=2 ~halves throughput**; **vbatch=4** is the
  quality/throughput sweet spot. TRT FP16 ≈2.4× forward, strength-neutral.
