# hexgt (Model 2) — state notes

_Continuity snapshot for the hexgt dynamic-GNN rewrite on branch `hexgt-rewrite`.
Written the night the overnight RL run was launched. (The previous dense_cnn
Model-1 notes live on branch `bench/inference-backends-wsl`.)_

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

## The overnight run (LIVE)

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
