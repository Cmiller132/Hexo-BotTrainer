# Hexo-BotTrainer

Reinforcement-learning training system for **Hexo**, a Connect6-style game played
on an unbounded hexagonal grid (place stones, win with 6 in a row along any hex
axis; two stones per turn after the opening). The repo contains an authoritative
Rust rules engine, self-improving neural bots (AlphaZero-style: self-play search
-> replay buffer -> supervised updates), a model-neutral training pipeline, a
match runner with an external minimax baseline opponent (SealBot), a ported GNN
eval opponent (Strix), and a web dashboard for monitoring runs and debugging
checkpoints. The active bot lineage is `hexfield_eq` (a D6-equivariant rewrite:
25-plane graded window features, an x12 regular-representation equivariant trunk,
register-lane / ray attention), whose first self-play run is gated behind a
BC-prefit arm ladder; the currently *running* self-play training is the
`hexfield` lineage's `main_9` run.

> NOTE: a live training run may be executing from this working tree under WSL.
> Treat the tree as hot: do not casually restart processes, edit behavior of
> imported modules, rebuild native crates, or touch run directories. The
> supervisors relaunch the driver between epochs and pick up whatever `.so` and
> Python source are on disk.

## Packages

| Path | Role | Status |
|---|---|---|
| `packages/hexfield_eq` | **Active bot lineage.** D6-equivariant rewrite: graded per-axis window features (25 node planes), an x12 regular-rep tied trunk, orbit/joint-tied bias, register-lane / ray attention. Own Rust cdylib (`hexfield_eq._rust`: featurizer, support build, native search, evaluator ABI). Config `configs/hexfield_eq_main_1.toml` | Active (current bot) |
| `packages/hexfield` | **Active run.** `main_9` lineage (variable-geometry hex-lattice model; c=128, CCACCACCA trunk, Gumbel/PUCT search). Own Rust cdylib (`hexfield._rust`). Config `configs/hexfield_main_9.toml`; `configs/hexfield_main_11.toml` kept as a revival candidate | Active (live `main_9`) |
| `packages/hexo_engine` | Authoritative rules engine (Rust + PyO3 `hexo_engine._rust`): board, turn phases, legality, incremental 6-cell win/threat windows, packed action IDs. Used by everything | Active |
| `packages/hexo_utils` | Shared utilities: `.hxr` binary game-record codec (Rust `records.rs`, re-exported via `hexo_runner.records`), `state_hash` for eval caches; generic JSON sample store (bypassed in production) | Active |
| `packages/hexo_runner` | Model-agnostic game execution: player contracts, match loop, `.hxr` records, SealBot subprocess adapter (the fixed minimax eval opponent) | Active |
| `packages/hexo_train` | Config-driven training orchestration: loads `configs/*.toml`, discovers model plugins via the `hexo_train.models` entry-point group (or a config `module = "<pkg>.plugin"` path), runs the epoch loop (selfplay -> train -> checkpoint -> eval) | Active |
| `packages/hexo_strix` | Ported SootyOwl/hexo-strix GNN bot (pure-torch HeXONet + axis graph) used as a fixed eval anchor opponent; weights ship in-repo | Active (eval opponent) |
| `packages/hexo_frontend` | Web dashboard (stdlib HTTP server, no framework): Match arena, training History, and a Debug workbench backed by a CPU-only torch worker subprocess. Runs as WSL systemd unit `hexfield-dashboard.service` on **:8080** | Active |
| `packages/hexo_models` | Umbrella crate/wheel: builds the legacy `dense_cnn` + `hexgt` native accelerators into one PyO3 module `hexo_models._rust` plus the shared threat-space-search core. Parked, but still loaded by the dashboard debug worker for legacy checkpoints | Parked |
| `packages/dense_cnn_restnet` | Former ResTNet lineage (pure Python/PyTorch; reuses `hexo_models._rust.dense_cnn`). Parked; kept on the dashboard PYTHONPATH as the legacy-shard oracle adapter and checkpoint loader. Its data flow is the reference model in `docs/ARCHITECTURE.md` | Parked |
| `packages/hexgnn` | Stripped-down GNN fork of `hexgt` (own Rust crate, compiled into `hexo_models._rust`) | Parked |

The Rust workspace (root `Cargo.toml`) has five members: `hexo_engine`,
`hexo_models`, `hexo_utils`, `hexfield`, `hexfield_eq` (the `hexgnn` crate is
compiled inside `hexo_models`; `dense_cnn_restnet` ships no Rust).

Cross-package contracts worth knowing (spot-verified against the tree):

- **Evaluator payload ABI** between the Rust search and the Python GPU evaluator
  (`packages/hexfield_eq/rust/src/payload.rs`, `ABI_VERSION = 1`): a per-flush
  CSR flat-concat of node features (F=25) over each row's support set, replied to
  with `values_bytes` / `priors_bytes` (f32, positional over each row's legal
  prefix) plus optional `moves_left_bytes` / `priors_logits_bytes`. This is the
  hexfield-lineage successor to the old dense-plane byte protocol.
- **`.hxr` game-record format** — the codec is Rust (`hexo_utils/rust/src/records.rs`),
  re-exported through `hexo_utils.records` -> `hexo_runner.records`.
- **Compact training shards** — one columnar `hexfield_compact_v1` `.npz` plus a
  JSON sidecar per game (`hexfield/shards.py`, sidecar written last as the commit
  marker).
- **Packed action-ID encoding** `(q+2^15)<<16 | (r+2^15)` — implemented in
  `hexo_engine` Rust and Python and again client-side in the dashboard `app.js`;
  persisted in shards, records, and URLs, so it must never diverge.
- **Run-dir diagnostics** — `diagnostics/*.json` + `events.jsonl` that the
  dashboard reads read-only.

## Training-run workflow

1. **Config**: each run is a heavily-annotated TOML in `configs/` whose header
   doubles as the run's evidence dossier. Live: `configs/hexfield_main_9.toml`
   (the `main_9` run) and `configs/hexfield_eq_main_1.toml` (the `hexfield_eq`
   soak — a pre-launch scaffold until its BC-prefit ladder has a winner and the
   deployment gates are green). Smoke/soak fixtures:
   `configs/hexfield_smoke.toml`, `configs/hexfield_smoke_tiny.toml`,
   `configs/hexfield_soak.toml`.
2. **Launch / supervise** (WSL systemd units in `scripts/systemd/`):
   `hexfield-supervisor-9.service`, `hexfield-supervisor-11.service`, and
   `hexfield-eq-supervisor-1.service`. All hexfield units drive the shared
   `scripts/_hexfield_supervise_main1.sh` (CONFIG/RUNDIR env-overridable) except
   the eq run, which uses its own copy `scripts/_hexfield_eq_supervise_main1.sh`
   (dedicated because its PYTHONPATH and serve flags differ). Each supervisor
   auto-relaunches with a circuit breaker + single-instance lock + halt flag,
   and **resumes from the latest checkpoint** by injecting `resume_from` right
   after `[checkpoint]` into a generated `_resume_config.toml`. The architecture
   env block (`HEXFIELD_*` / `HEXFIELD_EQ_*`) lives in the systemd unit and is
   **load-bearing** — it selects the network shape and must match the warm-start
   checkpoint's `arch_meta` exactly.
3. **Warm starts / prefit**: a run cold-starts from a behavioral-cloning prefit
   via `[checkpoint].initialize_from`. `main_9`'s BC prefit:
   `scripts/_main9_prefit_launch.sh` -> `scripts/_hexfield_prefit.py`. HF-corpus
   bootstraps: `scripts/bootstrap_hexfield_hf.py` and, for the legacy lineage,
   `scripts/bootstrap_dense_cnn_restnet_hf.py`.
4. **`hexfield_eq` prefit arm ladder**: `scripts/eq_ladder_runner.py` (launcher
   `scripts/run_eq_ladder.sh`) runs detached in WSL, chains the arm env files in
   `scripts/prefit_env/` on the single GPU under a hard deadline, strength-ranks
   them vs SealBot, picks a winner, and launches the `hexfield_eq_main_1` soak
   from it — no human in the loop (see `docs/AUTONOMOUS_LADDER_RUNNER.md`).
5. **Run directories** live on a *different* mount than the repo:
   `/mnt/e/Hexo-BotTrainer/runs/<run_name>/` (Windows: `E:\Hexo-BotTrainer\runs\`),
   containing `checkpoints/`, `selfplay/` npz shards, `evaluation/` `.hxr` records,
   `diagnostics/*.json` + `events.jsonl`, `manifest.json`, supervisor lock/halt
   flags, and `_resume_config.toml`.
6. **Monitoring**: the dashboard (`hexfield-dashboard.service`) runs in WSL on
   **:8080** with cwd at the run mount root, scanning `runs/<name>/` read-only;
   reachable from Windows via a netsh portproxy.

## Environment

- **Repo on Windows, training in WSL.** The repo lives at `E:\Hexo-BotTrainer-hexgt`
  and is mounted in WSL at `/mnt/e/Hexo-BotTrainer-hexgt`. All training, tests,
  and the dashboard run inside WSL.
- **Python venv (WSL)**: `/root/.venvs/hexgt-build` -- the authoritative
  interpreter for training, pytest, dashboard, and maturin. Note the hexfield
  packages are *not* pip-installed there; the supervisor scripts add them to
  `PYTHONPATH` from the source tree.
- **Native builds**: Rust workspace (root `Cargo.toml`) built with maturin.
  `scripts/_rebuild_hexfield.sh` rebuilds the `hexfield` crate in the *isolated*
  `hexfield-dev` venv (`--release`) and mirrors the built `.so` into the tree so
  the live `hexgt-build` supervisor never picks up a half-built extension; the
  `hexfield_eq` crate builds the same way. `scripts/_rebuild_hexo_models_hexgt.sh`
  builds the `hexo_models`/`hexo_engine`/`hexo_utils` wheels the dashboard loads.
  Rust edits are inert until the relevant rebuild runs.
- **Hardware**: one 12GB GPU (RTX 4070 Ti) shared by self-play inference and
  training; batch sizes in configs are measured against it.
- **External dependency**: SealBot (C++ minimax baseline) at `E:\SealBot`
  (`/mnt/e/SealBot`), resolved via `$SEALBOT_PATH` by the supervisor and
  dashboard. Strix eval-opponent weights ship in-repo under `packages/hexo_strix`.

## Tests

Run pytest **inside the WSL venv** -- that is the authoritative environment
(it has torch + the compiled native modules):

```
wsl -e bash -c 'cd /mnt/e/Hexo-BotTrainer-hexgt && /root/.venvs/hexgt-build/bin/python -m pytest tests/ -q'
```

Collecting on Windows works but torch/native-dependent tests skip cleanly
(`pytest.importorskip` guards), so a green Windows run proves little. Tests live
flat in `tests/` (plus `tests/eval_dashboard/` and `tests/katago_buffer/`);
ownership is by filename prefix: `test_hexfield_eq_*` = current bot,
`test_hexfield_*` = the `main_9` lineage + shared eval infra,
`test_hexo_*` / `test_frontend_*` / `test_sealbot_*` = infrastructure. Rust-side:
`cargo test` per crate.

## Docs index

| Doc | What it is |
|---|---|
| `docs/intro_to_hexo.md` | The game: rules, coordinates, terminology (derived from the `hexo_engine` Rust source) |
| `docs/hexfield_blueprint.md` | Plain-language "how it learns" onboarding — no board-game or ML background assumed |
| `docs/ARCHITECTURE.md` | Detailed cross-package system architecture and end-to-end data flow |
| `docs/specs/` | Binding contracts: `hexfield_model_spec.md`, `hexfield_eval_v2_spec.md`, and the dashboard `debug` / `history` / `match` screen v2 specs |
| `docs/PLAN_D6_EQUIVARIANT_REWRITE.md` | The `hexfield_eq` rewrite plan (decisions locked) |
| `docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md` | Group-theory derivation of exact D6-equivariant attention on the regular-rep fiber |
| `docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md` + `docs/SPEC_REGISTER_LANE_RAY_ATTENTION.md` | Register-lane / ray-attention plan and its implementation contract |
| `docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md` | Prefit-ladder -> self-play soak go/no-go gates for `hexfield_eq` |
| `docs/AUTONOMOUS_LADDER_RUNNER.md` | The autonomous `hexfield_eq` prefit-ladder runner (deadline regime) |
| `docs/BUGS_FOUND.md` | Open-bugs list for the landed `hexfield_eq` changes |
| `docs/quotient_reps/` | Next-direction spec bundle: `CONTEXT.md` + Phase A CPU proof + Phase B implementation spec |
| `todo.txt` | Deferred pre-KataGo FPU dead-code removal plan (kept for the parity harness) |
