# hexo_models

Umbrella package for the Hexo model zoo. It is **one Cargo crate and one
maturin wheel** that physically hosts every model lineage's native (Rust/PyO3)
accelerator under a single extension module, `hexo_models._rust`, plus the
shared Threat-Space-Search (TSS) core used by all lineages' MCTS.

Built via maturin (see `pyproject.toml`, module-name `hexo_models._rust`);
canonical rebuild script: `scripts/_rebuild_hexo_models_hexgt.sh` (WSL
`hexgt-build` venv, `--release`). Rust edits are inert until that script runs.

## What lives here

| Path | What it is |
|---|---|
| `python/hexo_models/__init__.py` | Namespace shim: grafts `dense_cnn/python/hexo_models` and `hexgt/python/hexo_models` onto `__path__` so `hexo_models.dense_cnn` / `hexo_models.hexgt` import from their subdirectories. Re-exports nothing itself. |
| `rust/src/lib.rs` | Crate root. `#[path]`-includes the lineage Rust trees (below) and registers them as submodules `hexo_models._rust.{dense_cnn,hexgt,hexgnn}`. |
| `rust/src/threats_shared.rs` | Shared TSS core (threat / win-now / forced-loss analysis over the engine `WindowStore`), reached as `crate::threats_shared` by the dense_cnn and hexgt Rust. (The hexgnn crate carries its own forked copy.) |
| `dense_cnn/` | The original "Model 1" dense-CNN lineage: Python package + Rust accelerator. See [`dense_cnn/README.md`](dense_cnn/README.md). |
| `hexgt/` | The "Model 2/3" GNN + transformer (hexgt) lineage: Python package + Rust accelerator. See [`hexgt/README.md`](hexgt/README.md). |
| `pyproject.toml` | maturin manifest; registers the `hexo_train.models` entry points `dense_cnn` and `hexgt`; bundles both lineages' Python trees into the wheel. |
| `Cargo.toml` | Workspace-member crate manifest (`rlib` + `cdylib`, pyo3 behind the optional `python` feature; depends on `hexo_engine` and `hexo_utils`). |

Note: a third lineage, **hexgnn** (`packages/hexgnn`, a fork of hexgt), lives in
a *different* package but its Rust crate is `#[path]`-included into this crate
and compiled as `hexo_models._rust.hexgnn` (see `rust/src/lib.rs` and the
`../hexgnn/rust/**/*` sdist include in `pyproject.toml`).

## Lineage status

| Lineage | Python status | Rust status |
|---|---|---|
| `dense_cnn` | **Legacy** as a training line (superseded by `packages/dense_cnn_restnet`), but still loadable: old checkpoints, ~14 test files, the dashboard debug worker, and hexgnn's `compact_io` dependency keep it alive. | **Active production.** The live `dense_cnn_restnet` runs (main_3/main_4) ship no Rust of their own and drive `hexo_models._rust.dense_cnn` (encoding, MCTS incl. the continuous scheduler, sample facts) read-only. |
| `hexgt` | **Halted lineage** (run `hexgt_rl_main3` permanently halted at epoch 40, 2026-06-05 per `HANDOFF.md`). Still live infrastructure: the frontend debug worker loads hexgt checkpoints, hexgnn is a fork of it, ~30 test files gate Rust/Python parity. | Compiled into every build; used by the parked hexgnn line and tests. |
| `hexgnn` (external, `packages/hexgnn`) | **Parked** experiment (explored and set aside). | Compiled into every build as `_rust.hexgnn`. |

The ACTIVE training lineage, `packages/dense_cnn_restnet`, is a full Python
fork of `hexo_models.dense_cnn` whose only native path is this package's crate.
Rebuilding the crate therefore changes search semantics for *both* the active
restnet line and the legacy dense_cnn line (intended, but easy to forget).

## How consumers reach this package

- `from hexo_models import _rust` then `_rust.dense_cnn` / `_rust.hexgt` /
  `_rust.hexgnn` -- via each lineage's `rust_bridge.py` (including
  `dense_cnn_restnet/python/dense_cnn_restnet/rust_bridge.py`).
- `hexo_train` plugin discovery: entry-point group `hexo_train.models`
  (`dense_cnn`, `hexgt` declared in `pyproject.toml`; resolved by
  `packages/hexo_train/python/hexo_train/registry.py`).
- `hexo_frontend/python/hexo_frontend/debug_infer.py` imports
  `hexo_models.dense_cnn.*` and `hexo_models.hexgt.*` to serve checkpoints in
  the dashboard debug screen / Match Arena.
- `cargo test -p hexo_models` runs the `threats_shared.rs` unit tests (no
  `python` feature needed).

## Caveats

- The built Linux `.so` (`python/hexo_models/_rust.cpython-*.so`) is an
  untracked maturin artifact in the source tree; on Windows-side Python the
  `_rust` module is simply absent (surfaced lazily by `rust_bridge` import
  guards). Re-run the rebuild script in the WSL venv after any Rust change.
- The `#[path]` include of `../../../hexgnn/rust/src/lib.rs` reaches outside
  this package directory -- the build breaks if `packages/hexgnn` moves.
- `threats_shared.rs` claims to be the single TSS definition; that is true for
  dense_cnn and hexgt, but hexgnn's included crate carries a duplicated fork.
