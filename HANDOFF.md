# Handoff

Single-machine RL training prototype for Hex. Python orchestration + Rust/PyO3 (maturin) for the
hot paths (engine, MCTS, sample generation). One GPU (~13 GB).

## Where the run is (as of 2026-06-09)

**Active line: `dense_cnn_restnet`.** The behavior-cloning prefit just finished
(`runs/dense_cnn_restnet_main1_prefit/`):
- `restnet_hf_prefit.pt` — 22 MB, 1,825,266 params, epoch 0 `status=completed`, 13342 steps,
  loss ≈ 4.10. Verified it loads `strict` into a fresh restnet.
- This is the BC bootstrap on the HF corpus; next step is to seed RL self-play from this checkpoint.
- Nothing is training right now (no live python process).

**Prior line: `hexgt_rl_main3` — permanently halted** by owner at end of epoch 40 (2026-06-05).
`runs/hexgt_rl_main3/supervisor_halted.flag` is intentionally left in place; do not relaunch it
without the owner's say-so. State at halt: rl_epoch=40, step≈21525, ~2.58M params, 512 games/epoch,
1024 visits, PCR enabled (p_full=0.5).

**`hexgnn`** was explored and set aside (not the active path).

## Codebase structure

`packages/<pkg>/python/<pkg>/` (Python) and `packages/<pkg>/rust/` (Rust crate, where present):

- **hexo_engine** (py+rust) — core Hex game engine: board, rules, tactics/threats.
- **hexo_models** (py+rust) — the model zoo. Two architectures live here:
  - `dense_cnn/` — CNN policy/value net (+ rust: encoding, mcts, sample_gen).
  - `hexgt/` — graph/transformer net (+ rust mcts, threats).
- **dense_cnn_restnet** (py) — residual variant of dense_cnn; the current active model + Spec A–D
  disk-attention work.
- **hexgnn** (py+rust) — GNN experiment (parked).
- **hexo_train** (py) — training harness (loss, optimizer, replay).
- **hexo_runner** (py) — run/process orchestration & supervision.
- **hexo_frontend** (py) — Flask web dashboard (`web.py`, `static/app.js`); served via
  `_dashboard_bridge.py`. Reads run logs/telemetry under `runs/`.
- **hexo_utils** (py+rust) — shared helpers.

Top level: `scripts/` holds launch + train entrypoints (e.g. `_rl_train.py`,
`_rl_launch_main3.sh`, `_rl_supervise.sh`). `runs/` holds per-run logs, checkpoints, selfplay data,
GPU telemetry (gitignored artifacts, not source). `tests/` covers the dense_cnn / restnet / hexgt
pipelines.

## Workflow notes

- The live run imports `.py` from this tree (`E:`) via PYTHONPATH and holds its `.so` in memory.
  Edit here, but commit/push from a separate clone so you don't reset the working tree under a
  running job. Sync `E:` to a new commit only at a clean epoch boundary / run bounce.
- Rust changes require a maturin rebuild before they take effect.
- `main` and `chore/hexgt-consolidation` are both at the cleaned tip (docs/memory wiped). `E:`'s local
  branch is intentionally older than the remote so the live run's files stay put.
