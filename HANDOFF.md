# Handoff

Single-machine RL training prototype for Hex (Connect6-style rules: opening single placement,
then two placements per turn). Python orchestration + Rust/PyO3 (maturin) for the hot paths
(engine, MCTS, sample generation). One GPU (~13 GB). Runs execute under WSL
(`/root/.venvs/hexgt-build`); run dirs live under `/mnt/e/Hexo-BotTrainer/runs/` (note: a
different mount root than this repo).

## Where the run is (as of 2026-06-10, ~00:30 UTC)

**Active line: `dense_cnn_restnet`, run `dense_cnn_restnet_main1` — LIVE at epoch 24.**
Launched 2026-06-09 18:50 from the HF-prefit warm start; supervised by
`scripts/_dc_restnet_supervise_main1.sh` (auto-resume + circuit breaker).

- SealBot eval trend (64 games, every 3rd epoch): 3 -> 3 -> 13 -> 9 -> **42 (ep 15)** -> 16 (ep 18).
  The 42->16 swing is mostly eval-opening correlation from the old fixed-seed bug (fixed at the
  bounce below); treat post-bounce evals as the new baseline. Next eval: epoch 27.
- Self-play games shortened 88 -> ~31 decisions over epochs 1-19 (offense outpacing defense —
  motivated the temperature change below). ~8.4k rows / 256-game epoch.
- **heads_v2 bounce EXECUTED 2026-06-10 00:00 UTC**: stop -> checkpoint migration
  (`scripts/_restnet_migrate_heads_v2.py`, resumed from migrated `epoch_000023.pt`, original kept
  as `.preheadsv2.bak`, value head function-preserved, optimizer moments dropped/rebuilt) ->
  relaunch. Verified live: 512 sims/position exactly, ~8.4 pos/s (was ~22 at 256 visits),
  calibration completed, train_state carried over (167,994 global step samples).
- Epoch 24's first post-bounce TRAIN pass had not completed at handoff time. Expected component
  picture: policy CE a level lower (legal-masked support), stvalue_2/6/16 near the uniform-bin
  floor (~4.17) training on new rows only, a moves_left component ramping in, value continuing
  from ~0.38.

## 2026-06-09/10 session changes (all in `packages/dense_cnn_restnet` + its config/scripts; the
parent `hexo_models.dense_cnn` package was NOT touched)

Training loss & data path:
- Policy CE is masked to the position's in-disk LEGAL cells (`losses.soft_cross_entropy(mask=...)`,
  fp32-upcast before the -1e9 fill for AMP safety; fail-loud if target mass falls outside the
  mask). Matches the serve contract (the Rust evaluator consumes per-legal-cell priors only).
  `opp_policy` is deliberately NOT masked (phase legality differs across turns). `legal_mask` is
  derived at expansion from stored legal ids — no shard migration was needed.
- `train_samples_per_epoch` 8000 -> 32000 (~3.8x the measured ~8.4k generated rows/epoch; KataGo-
  range reuse; training ~4 min vs ~16 min self-play at 512 visits).
- The shuffle writes the UNION of input-shard stvalue horizons with per-row masks
  (`replay._build_compact_split`), so mid-run horizon changes are lossless: old {1,4,8} rows train
  the new heads on nothing.

Search & exploration:
- Per-move search seeds (`selfplay.py` round counter; same in `evaluation.py`): root Dirichlet
  noise and the move-sampling quantile were previously frozen per (epoch, batch-slot) — one noise
  realization and one sampling quantile for a game's entire life, shared across games in a slot.
  Now i.i.d. per move, still deterministic in (run seed, epoch, round).
- Adaptive temperature: exponential decay with half-life = `temperature_halflife_fraction` (0.25)
  x expected game length, where the expectation is an EMA of measured mean decisions/game
  (`selfplay/length_ema.json`, seeded by `temperature_length_prior`, telemetry in the selfplay
  epoch summary under `temperature_control`). Replaces the absolute-move anchor schedule that the
  game-length collapse had reduced to "every move at temp >= 0.5".
- `search_visits` 256 -> 512; `mcts_session_cache_max_states` 131072 -> 262144 (was saturating).

Heads & architecture (heads_v2; params 1,827,775 -> 1,506,256 incl. buffers):
- One shared `ValueReduction` (conv1x1 -> Linear 1681->64) feeds per-head 64->65 Linear tops
  (value, stvalue_<h>, moves_left) — replaces four private flatten-Linear heads.
- Lookahead heads turn-aligned: the EMA steps over FULL TURNS (even decision offsets, per-step
  decay `(m-1)/(m+1)` preserving each horizon's effective mean lookahead;
  `samples._short_term_value_targets`), horizons `[2, 6, 16]` decisions = 1/3/8 turns (no
  horizon-1 head), EVEN horizons enforced at config parse, weight 0.25 -> 0.1 (they had grown to
  ~47% of the weighted gradient while diverging).
- New `moves_left` auxiliary head: decisions remaining (masked for truncated games), normalized
  onto the existing 65-bin support via `constants.MOVES_LEFT_CAP` (80); weight 0.1. Targets flow
  finalize -> shard (back-compat: old shards read as masked) -> expansion -> `binned_value_loss`.

Optimizer & checkpoints:
- AdamW param groups: weight decay only on matrix weights; biases, norm gains, and the learned
  relative-position bias tables get zero decay (`plugin.py`).
- `initialize_from` is weights-only (fresh optimizer/train state); only `resume_from` restores
  them (`checkpoints._is_initialize_only`). The first launch predated this and did import the BC
  Adam moments — harmless by now.
- The loader RAISES on a resume/architecture mismatch instead of silently restarting from random.
  Any future head/layout change needs `scripts/_restnet_migrate_heads_v2.py` (or a successor)
  run against the run dir BEFORE relaunch (script refuses while driver.pid is alive; keeps a
  backup; verifies value-head function preservation).
- `scripts/bootstrap_dense_cnn_restnet_hf.py` fixed for FUTURE prefits: BC replay has no search,
  so stvalue targets are now omitted (masked) instead of trained toward constant 0; moves_left
  targets come free. The existing prefit was NOT redone (owner call).

Tests: 71 passing across `tests/test_dense_cnn_restnet*.py`, including two new files —
`test_dense_cnn_restnet_policy_mask.py` (masked CE semantics, legal_mask plumbing, weights-only
init) and `test_dense_cnn_restnet_heads_v2.py` (half-life decay, turn-aligned EMA math, horizon
union, moves_left plumbing + back-compat, head layout).

The prefit (`runs/dense_cnn_restnet_main1_prefit/restnet_hf_prefit.pt` — 22 MB, 1 epoch,
loss ≈ 4.10) predates the Spec A disk-contract / Spec C-D scope code and the bootstrap fixes;
owner decided NOT to redo it — the run is progressing well.

**Prior line: `hexgt_rl_main3` — permanently halted** by owner at end of epoch 40 (2026-06-05).
`runs/hexgt_rl_main3/supervisor_halted.flag` is intentionally left in place; do not relaunch it
without the owner's say-so. State at halt: rl_epoch=40, step≈21525, ~2.58M params, 512 games/epoch,
1024 visits, PCR enabled (p_full=0.5).

**`hexgnn`** was explored and set aside (not the active path).

## Codebase structure

`packages/<pkg>/python/<pkg>/` (Python) and `packages/<pkg>/rust/` (Rust crate, where present):

- **hexo_engine** (py+rust) — core game engine: board, rules, tactics/threats
  (`threats::analyze` is an exact 1-ply oracle, also used by the TSS move-selection guard).
- **hexo_models** (py+rust) — the model zoo. Two architectures live here:
  - `dense_cnn/` — CNN policy/value net (+ rust: encoding, mcts, sample_gen). The Rust crate
    carries the radius-20 hex-disk crop contract (rebuilt 2026-06-09) and is reused read-only by
    the restnet fork.
  - `hexgt/` — graph/transformer net (+ rust mcts, threats).
- **dense_cnn_restnet** (py) — the ACTIVE model: faithful ResTNet trunk (interleaved residual +
  transformer blocks, disk-scope attention; Spec A–D) on the dense_cnn pipeline, now with the
  heads_v2 surface (shared ValueReduction; policy / value / opp_policy / stvalue_{2,6,16} /
  moves_left). Full Python fork of dense_cnn — its trainer/losses/replay/selfplay diverge from
  the parent as of this session.
- **hexgnn** (py+rust) — GNN experiment (parked).
- **hexo_train** (py) — training harness (pipeline, plugin registry, epoch loop, checkpoints).
- **hexo_runner** (py) — run/process orchestration & supervision.
- **hexo_frontend** (py) — Flask web dashboard (`web.py`, `static/app.js`); served via
  `_dashboard_bridge.py`. Reads run logs/telemetry under `runs/`.
- **hexo_utils** (py+rust) — shared helpers.

Top level: `scripts/` holds launch + train entrypoints (for the active line:
`_dc_restnet_launch_main1.sh`, `_dc_restnet_supervise_main1.sh`, `_restnet_migrate_heads_v2.py`,
`bootstrap_dense_cnn_restnet_hf.py`). `runs/` holds per-run logs, checkpoints, selfplay data,
GPU telemetry (gitignored artifacts, not source). `tests/` covers the dense_cnn / restnet / hexgt
pipelines.

## Workflow notes

- The live run imports `.py` from this tree (`E:`) via PYTHONPATH and holds its `.so` in memory.
  Edit here, but commit/push from a separate clone so you don't reset the working tree under a
  running job. Sync `E:` to a new commit only at a clean epoch boundary / run bounce.
- **All of this session's changes are UNCOMMITTED working-tree edits on `E:`** (restnet package,
  config, scripts, tests, this file) — fold them into the next sync.
- Rust changes require a maturin rebuild before they take effect (last rebuild 2026-06-09 16:09,
  disk-crop contract; the in-tree `.so` is what the WSL venv loads via editable `.pth`).
- Architecture/head changes additionally require the checkpoint migration before relaunch (see
  above) — the loader now fail-louds rather than silently restarting a run.
- `main` and `chore/hexgt-consolidation` are both at the cleaned tip (docs/memory wiped). `E:`'s
  local branch is intentionally older than the remote so the live run's files stay put.
