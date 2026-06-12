# Hexo-BotTrainer

Reinforcement-learning training system for **Hexo**, a Connect6-style game played
on an unbounded hexagonal grid (place stones, win with 6 in a row along any hex
axis; two stones per turn after the opening). The repo contains an authoritative
Rust rules engine, several neural model lineages (AlphaZero-style: self-play MCTS
-> replay buffer -> supervised updates), a model-neutral training pipeline, a
match runner with an external minimax baseline opponent (SealBot), and a web
dashboard for monitoring runs and debugging checkpoints. The currently active
lineage is `dense_cnn_restnet` (a ResTNet: interleaved residual + transformer
trunk), trained via the `main_4` run.

> NOTE: a live training run may be executing from this working tree under WSL.
> Treat the tree as hot: do not casually restart processes, edit behavior of
> imported modules, or touch run directories.

## Packages

| Path | Role | Status |
|---|---|---|
| `packages/hexo_engine` | Authoritative rules engine (Rust + PyO3 `hexo_engine._rust`): board, turn phases, legality, incremental 6-cell win/threat windows, packed action IDs | Active |
| `packages/hexo_utils` | Shared utilities: `.hxr` binary game-record codec (Rust), `state_hash` for MCTS eval caches; generic JSON sample store (bypassed in production) | Active (records/hash); samples scaffolding unused |
| `packages/hexo_runner` | Model-agnostic game execution: player contracts, match loop, `.hxr` records, SealBot subprocess adapter (the fixed eval opponent) | Active (core); batch/evaluation modes unused |
| `packages/hexo_train` | Config-driven training orchestration: loads `configs/*.toml`, discovers model plugins via the `hexo_train.models` entry-point group, runs the epoch loop (selfplay -> train -> checkpoint -> eval) | Active |
| `packages/hexo_models` | Umbrella crate/wheel: builds all native accelerators into one PyO3 module `hexo_models._rust` plus shared threat-space-search core (`rust/src/threats_shared.rs`) | Active (build host) |
| `packages/hexo_models/dense_cnn` | Original "Model 1" dense-CNN lineage. Its **Rust half is fully active** (the shared featurizer/MCTS used by dense_cnn_restnet); the Python half is legacy but loadable | Rust active / Python legacy |
| `packages/hexo_models/hexgt` | "Model 2/3" typed-GNN + transformer lineage (own Rust vertical). Run `hexgt_rl_main3` permanently halted 2026-06-05 | Legacy (halted) |
| `packages/hexgnn` | Stripped-down GNN fork of hexgt (own Rust crate, compiled into `hexo_models._rust.hexgnn`) | Legacy (parked) |
| `packages/dense_cnn_restnet` | **ACTIVE lineage**: ResTNet fork of dense_cnn (pure Python/PyTorch; reuses `hexo_models._rust.dense_cnn` read-only). Owns config, architecture, self-play schedulers, KataGo-style replay, trainer, SealBot eval | Active (live main_4 run) |
| `packages/hexo_frontend` | Web dashboard (stdlib HTTP server, no framework): Match arena, training History, and a Debug workbench backed by a CPU-only torch worker subprocess | Active (under heavy development) |

Cross-package contracts worth knowing: the (N,13,41,41) tensor byte protocol
between Rust MCTS and the Python evaluator; the compact `.npz` shard format
(`compact_io.py`); the `.hxr` record format; the packed action-ID encoding
duplicated in `hexo_engine` Rust and Python; and the run-dir diagnostics JSON
files the dashboard reads.

## Training-run workflow

1. **Config**: each run is a heavily-annotated TOML in `configs/` (the file
   header doubles as the run's evidence dossier). Live: `configs/dense_cnn_restnet_main_4.toml`.
   Earlier runs of the same lineage: `main1`, `main_2`, `main_3` (collapsed/halted).
2. **Launch/supervise** (WSL shell scripts in `scripts/`):
   `scripts/_wf_r4_launch_main4.sh` setsid-detaches the generic supervisor
   `scripts/_dc_restnet_supervise_main1.sh` (env-overridable CONFIG/RUNDIR --
   despite the "main1" name it supervises main_2/3/4 too), which loops
   `python -m hexo_train.cli.train_model <config>`. Mid-run config edits are
   applied with `scripts/_wf_r4_bounce_main4.sh` (driver-only bounce).
3. **Run directories** live on a *different* mount than the repo:
   `/mnt/e/Hexo-BotTrainer/runs/<run_name>/` (Windows: `E:\Hexo-BotTrainer\runs\`),
   containing `checkpoints/`, `selfplay/` npz shards, `evaluation/` `.hxr` records,
   `diagnostics/*.json` + `events.jsonl`, `manifest.json`, supervisor lock/halt
   flags, and `_resume_config.toml`.
4. **Monitoring**: the dashboard (`packages/hexo_frontend`) runs in WSL on
   **:8080** (launched by `scripts/_dashboard_launch.sh`, cwd at the run mount
   root, reachable from Windows via a netsh portproxy). Babysitting gates:
   `scripts/_wf_r4_health.py`, `scripts/_wf_r4_m4_gates.py`.

Warm starts come from `scripts/bootstrap_dense_cnn_restnet_hf.py` (behavioral-
cloning prefit on a human-games corpus), consumed via `[checkpoint].initialize_from`.

## Environment

- **Repo on Windows, training in WSL.** The repo lives at `E:\Hexo-BotTrainer-hexgt`
  and is mounted in WSL at `/mnt/e/Hexo-BotTrainer-hexgt`. All training, tests,
  and the dashboard run inside WSL.
- **Python venv (WSL)**: `/root/.venvs/hexgt-build` -- the authoritative
  interpreter for everything (training, pytest, dashboard, maturin).
- **Native builds**: Rust workspace (root `Cargo.toml`: hexo_engine, hexo_models,
  hexo_utils) built with maturin via `scripts/_rebuild_hexo_models_hexgt.sh`
  (`--release`, in the WSL venv). Rust edits are inert until this script runs.
  Beware: the older `_rebuild_hexo_models.sh` builds a *sibling* checkout.
- **Hardware**: one 12GB GPU (RTX 4070 Ti) shared by self-play inference and
  training; batch sizes in configs are measured against it.
- **External dependency**: SealBot (C++ minimax baseline) at `E:\SealBot`
  (`/mnt/e/SealBot`), put on PYTHONPATH by supervisor and dashboard scripts.

## Tests

Run pytest **inside the WSL venv** -- that is the authoritative environment
(it has torch + the compiled native modules):

```
wsl -e bash -c 'cd /mnt/e/Hexo-BotTrainer-hexgt && /root/.venvs/hexgt-build/bin/python -m pytest tests/ -q'
```

Collecting on Windows works but torch/native-dependent tests skip cleanly
(`pytest.importorskip` guards), so a green Windows run proves little. All ~115
test files live flat in `tests/`; ownership is by filename prefix
(`test_dense_cnn_restnet*` = active lineage, `test_hexgt_*`/`test_hexgnn_*` =
legacy lineages, `test_frontend_*`/`test_hexo_*` = infrastructure). Rust-side:
`cargo test` per crate.

## Docs index

| Doc | What it is |
|---|---|
| `docs/intro_to_hexo.md` | The game: rules, coordinates, terminology |
| `docs/ARCHITECTURE.md` | Detailed system architecture |
| `docs/analysis/` | Run forensics and post-mortems (e.g. `MAIN4_RECOMMENDATION.md` -- the design dossier behind the live run; `MAIN1_DIVERGENCE_FORENSICS.md`; superseded audits under `archive/`) |
| `docs/specs/` | Binding contracts for the dashboard rewrites (debug/history/match screen v2 specs) |
| `HANDOFF.md` | Running engineering log / session-to-session handoff. **The canonical onboarding doc, but trailing reality**: newest section predates the main_3 collapse and the live main_4 run -- current state lives in config headers and `docs/analysis/` |
