# HANDOFF — dense_cnn Model 1 (target_64x4) cold-start run

Date: 2026-05-30. Branch: **`bench/inference-backends-wsl`** (all work below is uncommitted on
this branch). Read `CLAUDE.md` first for repo layout/build; this doc is the live state + recent
work. Tight orientation also in `NOTES.md`; deep perf writeup in
`runs/selfplay_profile_findings.md`. (A prior `rust-rebuild`/scratch_64 crash-hunt handoff was
here before — it is superseded by this doc.)

## 1. What this project is
Single-machine reinforcement-learning trainer for the board game **Hexo** (Ryzen 7950X + one
CUDA GPU, RTX 4070 Ti 12 GB). Self-play → train → checkpoint → eval, AlphaZero/KataGo-style.
Production target is **Model 1**, a dense CNN in the `hexo_models.dense_cnn` package. Six
installable packages (Rust+PyO3 for engine/utils/models, pure-Python for runner/train/frontend).
Board is fixed 41×41, 13 input planes, 65-bin value head, `ActionId = u32_i16_pair`. Training is
driven by the config CLI: `python -m hexo_train.cli.train_model <config.toml>`; lifecycle lives
in `hexo_train.pipeline.TrainingPipeline`.

## 2. The model (dense_cnn "Model 1")
- **Architecture** (`architecture.py`): conv-in (13→C) + N gated residual blocks (HexConv2d, a
  3×3 conv with the two non-hex corners masked) + heads: **policy** (fully-conv P7, one logit per
  41×41 crop cell), **value** (65-bin KataGo-style), **opp_policy**, short-term-value heads
  (horizons 1/4/8). `forward_policy_value` is the search-only path (policy+value).
  `optimized_model1_for_inference` folds HexConv/BN into plain convs for CUDA.
- **Current run config** (`configs/dense_cnn_model1_target_64x4.toml`): **64 channels × 4
  blocks** (small trunk for fast iteration), **256 MCTS sims/move**, `active_games=256`,
  `forced_playout_k=2`, temperature decay 1.0→0.2 by ply 30, **cold start from random init (no
  SealBot bootstrap — `initialize_from`/`resume_from` are absent; the supervisor's "uses
  initialize_from → bootstrapped" log line is COSMETIC)**, TensorRT FP16 self-play eval, 60
  epochs, `games_per_epoch=512`, `max_actions=1024`.
- **Python/Rust split:** Python owns PyTorch, config, self-play control, sample finalization, NPZ
  replay/shuffle, training, checkpoints. Rust owns live `HexoState` intake, dense tensor
  encoding, batched PUCT MCTS (tree reuse via `advance_root`, policy-nucleus widening
  max_children=32), and state-derived sample facts. Native calls go through
  `hexo_models._rust.dense_cnn` — built INTO the `hexo_models` package via `#[path]` includes, so
  editing `dense_cnn/rust/src/*.rs` requires rebuilding **`hexo_models`**, not a separate crate.

## 3. What's going on right now (THIS session's work)
The run was relaunched after a **self-play performance + memory effort**. All changes are
**byte-identical / functionally verified** (§5). Three landed fixes:

1. **Zero-copy plane buffer (≈1.85× self-play)** — `mcts_eval.rs`. The Rust→Python `inputs`
   handoff was a serial ~89 MB `PyBytes` memcpy *every forward pass* (was 46% of self-play
   wall). Replaced with a buffer-protocol `#[pyclass] PlaneBuffer` that `torch.frombuffer` views
   zero-copy. (Diagnosed with `scripts/_profile_selfplay.py`: forwards were already full 1024,
   eval cache 0% hit mid-game — the copy, not the GPU, was the bottleneck.)
2. **f16 input transport (≈1.06×, halves H2D)** — `PlaneBuffer` now carries **f16** planes;
   `inference.py` reads `dtype=float16` and upcasts f16→f32 **on-device** after the halved
   host→device copy, so the TRT engine + forward are unchanged. Added `half` crate to
   `hexo_models/Cargo.toml`. Gated byte-identical (TRT already downcasts its f32 input to f16).
3. **Sample compaction (~13× smaller pending samples — fixes an OOM)** — `samples.py`, pure
   Python. See §4.

Net: self-play **~29 → ~58 pos/s**, and the pending-sample OOM is gone.

## 4. The OOM that was fixed (key context)
On the first optimized relaunch the run climbed to **27.7 GB RSS and OOM'd mid-epoch**.
Root-caused (NOT the perf fixes — proven via a leak probe = +0 KB/call; and MCTS trees are
bounded): **cold-start games barely terminate** (random net), so 256 concurrent games run ~426
plies deep toward `max_actions=1024`, each hoarding a per-ply `pending` policy-sample list. Each
sample was **~320 KB** — ~75% Python-object overhead on packable arrays (`root_prior_policy`
stored all ~1471 legal-move priors as Python `(int,float)` tuples; `legal_action_ids` as Python
ints; `stones`/`placement_history` grew O(ply) as tuples-of-tuples).

Fix (`samples.py`): policies → byte-backed `CompactVisitPolicy` (search already returns this);
`legal_action_ids` → `array("q")`; `stones`/`placement_history` → columnar
`_PackedStones`/`_PackedHistory` (int16 coords + 1-byte owner) that **iterate to the identical
tuples the encoder expects** (so `input.py build_input_planes` is unchanged), dropping 3
encoder-unused history fields. **Sample 312 KB → ~24 KB.** In-memory only — NPZ schema unchanged.

## 5. Verification done (all green)
- **Full test suite: 154 passed** (`scripts/_run_all_tests.sh`). Updated 3 tests + 1 stale fake
  to the compact/f16 representations.
- **Functional identity PROVEN** (`scripts/_equiv_check.py`): expanding a compacted sample vs a
  tuple-form sample of identical data through `expand_sample` under all 12 D6 symmetries →
  **4284/4284 training tensors `torch.equal`**. Training data is bit-identical.
- **Production RAM verified plateaus** (`scripts/_ram_monitor.sh`): over ~13.5 min / 29 games
  trainer RSS held ~5.5→6.5 GB with ~21.5 GB free and a *flattening* slope (pre-fix hit 27 GB and
  OOM'd by ~30 min). pos/s steady ~58, TRT argmax-match 1.0, search-exact.

## 6. Current state / what's running
- **Supervised cold-start epoch RUNNING** (relaunched 2026-05-30 ~21:42 UTC). Supervisor PID
  ~3381, trainer ~3412 (WSL). Epoch 1 self-play in progress (~30 of 512 games done at writing),
  RAM healthy. **No checkpoint yet** — epoch 1 self-play must finish before train+checkpoint+eval.
- WSL venv `/root/.venvs/hexo-bottrainer-wsl` (py3.12, torch, tensorrt). The supervisor sets
  `PYTHONPATH` to the worktree `packages/*/python` so it uses the freshly-built native module.

## 7. How to operate (WSL; invoke via PowerShell: `wsl -d Ubuntu-24.04 -- bash <script-path>`)
- **Status:** `scripts/_status_64x4.sh` · `scripts/_ram_check.sh` (RSS/MemAvailable/live pos/s).
- **Stop:** `scripts/_stop_64x4.sh` (kills supervisor FIRST so it can't relaunch, then trainer).
- **Relaunch clean:** `scripts/_wipe_and_relaunch_64x4.sh` (refuses if procs alive; wipes run dir;
  detaches via `setsid nohup … < /dev/null &` — a plain `disown` dies on WSL session exit).
- **Wait for self-play:** `scripts/_wait_selfplay_64x4.sh`.
- **Rebuild native after Rust edits:** `scripts/_rebuild_and_profile.sh` (sets `VIRTUAL_ENV` +
  rustup cargo on PATH — apt cargo is too old for lockfile v4; full `cargo clean` only on an
  E0461 ICE). **Pure-Python edits (e.g. `samples.py`) need NO rebuild.**
- **Diagnostic probes (on-demand, each has a `_run_*.sh` wrapper):** `_profile_selfplay.py`
  (time split + forward floor + batch histogram), `_sample_size_probe.py` (per-field bytes),
  `_tree_growth_probe.py` (`PROBE_PENDING=1 PROBE_PLIES=N` → RAM-vs-plies), `_leak_probe.py`
  (eval-path leak), `_equiv_check.py` (functional identity), `_mem_breakdown.sh`.
- **GOTCHAS:** inline `bash -c "...$(...)..."` via the PowerShell→wsl bridge mangles quoting —
  always run a **script file**. Native Windows GPU memory (~3 GB) is the Windows desktop sharing
  the GPU, not our process. Run dir: `runs/dense_cnn_model1_target_64x4/`
  (`selfplay/`, `checkpoints/`, `diagnostics/`). Dashboard: http://localhost:8080. Supervisor
  guardrails: auto-relaunch + resume-from-latest; circuit breaker (3 crashes <180s OR >6/hr OR 5
  no-progress → `diagnostics/supervisor_halted.flag`); RAM watchdog → `diagnostics/watch_wsl.jsonl`.

## 8. Open items / watch next
- **First epoch result:** when epoch 1 self-play finishes, expect a checkpoint + a training step
  (`loss_components`: policy/value/opp/stvalue, train + held-out val in `diagnostics/
  epoch_000001.json`) + a SealBot best-50ms eval (`dense_cnn.evaluation.epoch_*.json`). Report
  whether **policy CE drops** and value CE / winrate / mean game-length move — the real "is it
  learning" signal. (Prior 96×6 lesson: a *rising* epoch-loss was a measurement artifact, not
  divergence; the real limiter was a diffuse policy head — so trust the fixed-holdout val + a
  `_loss_decomp.py`-style check over checkpoints, not the raw epoch loss.)
- **Cold-start games are long** (don't terminate at random init) → epoch 1 self-play is the
  slowest; later epochs speed up as games shorten. Expected, not a bug.
- **Uncommitted:** all of §3-4 (Rust + Python + `half` dep + config + test updates + scripts) is
  on `bench/inference-backends-wsl`. Commit when the user asks.
- **Further memory headroom (NOT needed — OOM solved):** `array('q')→'I'` on legal ids (8→4 B);
  shrink the eval cache (`mcts_session_cache_max_states` 131072, ~0% hit ≈ 2 GB); or recompute
  board facts from the move history at expand-time (O(N) vs O(N²), trades CPU for memory).
- **96×6 run is PAUSED, not deleted** — `runs/dense_cnn_model1_target_96x6/checkpoints/
  epoch_000011.pt` preserved; resume with `scripts/supervise_target_96x6_wsl.sh`. Never run both
  (one GPU).

## 9. Key file map (dense_cnn)
- Self-play loop: `dense_cnn/python/.../selfplay.py` · MCTS Python boundary + `CompactVisitPolicy`:
  `mcts.py` · Torch evaluator/inference (+ f16 upcast): `inference.py` · **samples (compaction):
  `samples.py`** · tensor expansion: `input.py` · trainer/losses: `trainer.py`/`losses.py` ·
  replay/NPZ + policy-surprise: `replay.py` · calibration: `performance.py` · plugin/registry:
  `plugin.py`, `hexo_train/registry.py`.
- Rust: `dense_cnn/rust/src/mcts.rs` (PUCT, result payloads), **`mcts_eval.rs` (PlaneBuffer +
  eval cache + encode→Python)**, `encoding.rs` (planes), `sample_gen.rs` (state facts).
- When changing the representation, update BOTH language halves together (CLAUDE.md lists the
  exact file pairs).
