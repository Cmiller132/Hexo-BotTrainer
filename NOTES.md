# hexgt (Model 2) — state notes

_Continuity snapshot for the hexgt dynamic-GNN rewrite on branch `hexgt-rewrite`.
Written the night the overnight RL run was launched. (The previous dense_cnn
Model-1 notes live on branch `bench/inference-backends-wsl`.)_

---

## LATEST STATE — Model 3 / `hexgt_rl_main3` (2026-06-03, LIVE)

_This supersedes the "The overnight run (LIVE)" section below, which describes the
earlier Model-2 `hexgt_rl_main` run. The stack has since advanced to **Model 3**
(heavier 4-GNN + PMA value head) and a new run dir `runs/hexgt_rl_main3/`._

### Live run health (verified 2026-06-03 ~23:24 EDT)
- **ALIVE & healthy.** Supervisor (WSL pid 57835, `_rl_supervise.sh`) + driver
  (pid 57883, `_rl_train.py`, ~426% CPU) both up; GPU ~54% util / 6.5 GB used of
  12.3 GB; free system RAM ~19 GB. Driver err log shows only benign warnings
  (non-writable `frombuffer`, `index_reduce` beta) — no crashes.
- **Phase: epoch 0, self-play.** Pre-RL baseline already logged this launch
  (vs SealBot 0W/40L = 0.0%, expected pre-RL; vs-dense_cnn no-op'd, see below);
  **136 / 256** epoch-0 self-play games produced and advancing in real time
  (newest shard `epoch_000000_game_000082.npz` written within the last minute).
- **Note:** the keep-online supervisor has relaunched ~hourly (last launch
  22:52 EDT). These are NOT crash-loops (far above the `fast<180s ×3` breaker);
  each relaunch auto-resumes and re-runs the pre-RL baseline before continuing
  epoch-0 self-play. Worth watching that epoch 0 actually completes & checkpoints
  (`checkpoints/` still empty as of the first checkpoint at epoch end).

### Model-3 stack (what changed vs Model 2)
- **4 GNN layers** (was 3), **PMA value head** (k=2 seeds, scatter-softmax
  varlen — replaces mean+max pooling), `value_head_use_side` default **true**
  (a no-SIDE A/B toggle exists), **soft-Z value targets** (λ=0.5).
- **Policy-surprise weighting** enabled (KataGo-style row-duplication via
  `materialize_policy_surprise_rows`, KL(visits‖prior)).
- **STV heads [4,12,24] @ weight 0.10**, **count-4 threat / hot-token features**.
- **TSS always-on**: tactical injection + phase-aware hitting-set leaf override +
  tactical-aware move-selection guard; engine threat lookup is **incremental**.
- **AMP fp16 GradScaler fix** applied.
- **Non-finite-logit sanitization is now audited** (counters + per-epoch/per-run
  surfacing + dashboard); affected positions are **EXCLUDED from training shards**
  (round-level).

### Run config (driver args, verified)
- 256 games/epoch, **512 sims** (self-play + eval), **65,536 train samples/epoch**
  (512 steps × batch 128), **500k replay pool**, recency **0.9/epoch**, replay
  window 8 ep, widening **96**, max-actions 512, **eval-every 3**, lr 2e-4,
  warmup 200, active=64, **vbatch=128**. BC/seed = `runs/hexgt_rl_main3/pretrain/
  hexgt_model3_pretrain.pt` (step=354, rl_epoch=0; 2,415,430 params).
- **Compiled + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — a startup
  log line confirms both (`GPU mem config: compile=True expandable_segments=ON`),
  warns loudly if not.

### Eval
- **dense_cnn e24 checkpoint was LOST** (see integrity event) — eval-vs-dense_cnn
  is try/except-guarded and **no-ops** (logs "vs-dense_cnn SKIPPED … missing").
  **SealBot is the strength benchmark now.**

### Performance / VRAM findings
- **Self-play ~6–8 pos/s** (old 3-layer/mean+max hexgt got ~12). Bottleneck = the
  **NN forward (~78% of wall)** of the heavier 4-GNN+PMA model; GPU only ~45% util.
  TSS threat scans are ~1% of wall (NOT the bottleneck — an incremental-threats +
  short-circuit fix gave only ~1.05×, kept as a small bit-identical win).
- Behavior-preserving "safe wins" (featurize‖forward overlap + dtype/host-overhead)
  ≈1.3×; **vbatch 64→128 ≈1.23×** but is behavior-changing (fewer NN feedback
  rounds per move: 4 vs 8 at 512 visits) and tops the throughput knobs. True 2×+
  needs a **lighter trunk** (gnn_layers 4→3 / smaller token_dim) = a model change
  needing re-pretrain.
- **FlexAttention trunk** prototyped (correct, 0 recompiles) but **DEFERRED** (net
  pos/s regression from per-forward mask build; no real VRAM win vs compiled-padded).
- **VRAM is fine**: live peak ~1.4 GB compiled (model weights ~5 MB); the earlier
  "11.8 GB / 96%" was allocator fragmentation under eager+default, fully reclaimed
  by compile + expandable_segments (~2.4 GB envelope).

### Integrity event + safe routines (IMPORTANT)
- `E:\Hexo-BotTrainer` (the **main clone**) lost its `.git` and many files
  (docs/configs/scripts/most packages) in a logical recursive deletion (drive
  healthy — not us, not hardware). **All code + history is safe on origin**
  (github Cmiller132/Hexo-BotTrainer); the `-hexgt` worktree is intact; the
  dense_cnn eval checkpoint is gone (accepted). Restoring the main clone is a
  re-clone-from-origin when desired (**deferred**).
- The two old scheduled routines were disabled. A **SAFE 30-min keep-online
  monitor** (task `hexo-epoch1-watch`) is re-enabled targeting `hexgt_rl_main3`:
  it only health-checks and gently relaunches the supervisor if the run died,
  with hard constraints against any delete/git/force-kill/destructive recovery.
  The stale `hexo-bottrainer-overnight-monitor` (old 96×6/dense_cnn run) stays
  **DISABLED**.
- `origin/main` carries the consolidated stack (PMA varlen value head, hoisted-
  layout padded trunk, TSS + soft-Z + policy-surprise + threat features,
  GradScaler, logit audit, drop-SIDE flag, mem logging) up through the latest
  commits.

---

## What hexgt is

Model 2 for the Hexo RL trainer: a **dynamic GNN + graph-transformer** that scores
a per-position **candidate set** (active-windows ∪ n-radius, `candidate_radius=3`)
instead of a dense 41×41 plane (dense_cnn = Model 1). It is **D6-invariant by
construction** (no augmentation needed), ~2.0M params (token_dim 168, gnn×3,
ctx×3), and produces a per-candidate policy + 65-bin distributional value +
opp-policy + short-term-value heads. Package: `packages/hexo_models/hexgt/`
(Python) + `hexgt/rust/` (candidate/window/graph builder, featurizer, MCTS).

## What's built (all on `hexgt-rewrite`, pushed)

- **Model** (`architecture.py`): relational message passing (per-edge-type einsum)
  ×3 + graph-transformer (context self-attn + candidate→context cross-attn) ×3,
  vectorized padded attention. D6-equivariance gated by tests.
- **MCTS** (`rust/src/mcts*.rs`, `mcts.py`): dense_cnn's tree+session copied
  verbatim (nucleus widening, forced playouts, virtual-loss select↔eval, subtree
  reuse); only the eval boundary differs (graph payload + per-candidate priors).
  The synchronous batched session already coalesces all concurrent games' leaves
  into one forward/round (= the "async batcher" throughput property); a true async
  Rust batcher was deliberately NOT built (the sync path already beats dense_cnn).
- **Featurizer** (`rust/src/features.rs`): Rust+rayon, zero-copy buffers → Python
  `frombuffer` + forward. Byte-identical to the Python path (gated).
- **Trainer** (`trainer.py`): AdamW + warmup + AMP, `hexgt_loss`, recompute-at-
  expand from compact shards (`expand.py`). Reuses dense_cnn's compact IO/replay.
- **Self-play** (`selfplay.py`): game-driven loop → dense_cnn COMPACT shards (so
  the trainer reads them unchanged). Emits Q1–Q5 self-play data-quality metrics.
- **Inference** (`inference.py`): torch FP16; **chunked forward** (sorted, budget-
  bounded sub-batches) caps search-forward VRAM ~3–4 GB and is bit-identical;
  fp16-overflow NaN guard.
- **Eval harness** (`evaluation.py`): `run_head_to_head_parallel` (many games
  batched, deterministic per-game) + optional opening variety (decorrelates the
  win-rate estimate, still repeatable). `HexgtPlayer` (deterministic runner
  player). SealBot leg wired (`SEALBOT_PATH=/mnt/e/SealBot`).
- **Drivers**: `scripts/_rl_train.py` (resumable main RL run), `scripts/_rl_ablate.py`
  (short exploration ablations), `scripts/_rl_supervise.sh` (supervisor), the
  converged BC trainer `scripts/_bc_train.py`. Suite: **196 tests green**.

## Key results / anchors

- **BC seed** `runs/hexgt_bc/hexgt_bc_step006009.pt`: converged behavioral clone of
  dense_cnn e24 (held-out top-1 33.5%). Starting head-to-head anchors (deterministic,
  visits=200, opening-variety eval): **45.8% vs dense_cnn epoch-24** (on par), **0%
  vs SealBot best-50ms** (expected pre-RL — dense_cnn only beat SealBot after RL).
- **Throughput**: ~28–62 self-play pos/s (64 concurrent games + vbatch 64 + compile,
  after the VRAM-compression fix removed host-RAM spilling). Beats dense_cnn ~23.
- **Architecture finding**: the GNN's raw policy PRIOR is diffuse (entropy ~3.3–3.6)
  but MCTS sharpens it hard into decisive play; RL further sharpens the prior itself.
- **Ablation (C1/C2/C3)**: chose **C1** (derived baseline) — `total_alpha=6.6,
  eps=0.25, root_policy_temperature=1.0, c_puct=1.5, temp 1.0→0.2@30, forced_k=2`.
  Best "diverse AND decisive" data profile; details in HEXGT_DECISIONS.md Phase 11.

## The overnight run (Model 2 — `hexgt_rl_main`) — SUPERSEDED

_Historical: this was the original Model-2 run. The live run is now Model 3 /
`hexgt_rl_main3` — see "LATEST STATE" at the top._

- **Run dir**: `runs/hexgt_rl_main/` (worktree `E:\Hexo-BotTrainer-hexgt`).
- **Config**: C1, BC-seeded, 60-epoch cap, visits=128, 96 games/epoch (active=64,
  refilled), train 300 steps/epoch, lr 2e-4, eval every 3 epochs (vs dense_cnn e24
  + SealBot + holdout + Q-metrics). Under the supervisor (auto-relaunch + RAM
  watchdog + circuit breaker), so it advances unattended through crashes.
- **Check status**: `cat runs/hexgt_rl_main/rl_train.log` (per-epoch self-play
  pos/s + Q1–Q5; eval lines `>>> ... EVAL`), `tail runs/hexgt_rl_main/supervisor.log`,
  `runs/hexgt_rl_main/eval/epoch_*_eval.json`. Checkpoints:
  `runs/hexgt_rl_main/checkpoints/hexgt_rl_{epochNNNNNN,latest}.pt`.
- **Stop cleanly**: `touch runs/hexgt_rl_main/supervisor_halted.flag` (stops the
  supervisor from relaunching), then `pkill -9 -f _rl_train.py` (and the supervisor
  bash). The foreground launch is a Bash background task; the halt flag is the clean
  brake.
- **Resume**: relaunch `bash _rl_run_fg.sh` — the driver auto-resumes from
  `hexgt_rl_latest.pt` (model + optimizer + train-state + rl_epoch).

## Watch items / known caveats

- **The key open question**: the 5-epoch ablation showed all configs sharpen fast
  with a lumpy/declining L1-vs-frozen and a rising L2-holdout-CE (model moving off
  the dense_cnn imitation target). Whether 60 epochs RECOVERS and pushes past the
  45.8% baseline (vs over-commits/collapses) is exactly what this run answers.
  Judge by the **head-to-head vs dense_cnn e24 + SealBot** trend + Q-metric
  diversity retention, NOT raw epoch loss.
- **VRAM compile-cache growth**: torch.compile caches sub-batch shape variants;
  reserved VRAM grows from ~3 GB to ~8 GB over epochs but plateaus (well under 12
  GB; `expandable_segments` on). If it ever approaches the ceiling, lower `--vbatch`.
- **fp16-overflow NaNs** are sanitized (logged to `driver.*.err` as "sanitized N
  non-finite"); frequency scales with exploration. If excessive, run eval in fp32.
- **candidate_radius=3** drops ~10% of far-spread moves (pruned in training); a
  deliberate move-vocabulary choice (HEXGT_DECISIONS.md candidate-set decision).
- **Isolation**: everything runs in the `E:\Hexo-BotTrainer-hexgt` worktree with its
  own WSL venv `/root/.venvs/hexgt-build`. Do NOT touch the live tree
  `/mnt/e/Hexo-BotTrainer`, its 96x8 checkpoints (read-only for eval), or the
  dashboard (`hexo_frontend.web` :8080, separate venv). Commits via Windows git.
  Bash tool = Git-Bash/MINGW (paths `/e/...`); run WSL via
  `wsl.exe -- bash -lc "tr -d '\r' < /mnt/e/.../script.sh | bash"`.

## Open items (post-overnight)

- Read the morning trajectory; decide continue / adjust lr / switch to C3 if C1
  destabilized. Pull example games to characterize play style vs the BC seed.
- If throughput-starved later: the true async Rust leaf batcher (deferred).
- If NaN frequency bites: fp32 eval forward, or clamp attention logits at the root.
- Port `expand.py` training featurization to the Rust path if BC/RL step time bites.
