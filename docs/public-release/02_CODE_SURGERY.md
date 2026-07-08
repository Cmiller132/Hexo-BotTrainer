# 02 — Code Surgery

Source edits required so the kept subset builds and runs standalone. All edits
happen in the staged copy at `E:\hexo-bot` — never in this repo (the live main_7
run executes from this tree).

Scope note: this doc covers the edits needed to **build and run** the subset.
The deeper dead-path removal (PUCT knobs, A/B perf branches, dead subsystems)
is a separate pass — [06_CODE_SLIMMING.md](06_CODE_SLIMMING.md) — applied after
the Phase 3 build gate is green.

## 1. Vendor `threats_shared.rs` into hexfield

**Problem:** `packages/hexfield/rust/src/lib.rs:20` includes a file from the
cut `hexo_models` crate:

```rust
#[path = "../../../hexo_models/rust/src/threats_shared.rs"]
mod threats_shared;
```

**Fix:** copy `packages/hexo_models/rust/src/threats_shared.rs` →
`packages/hexfield/rust/src/threats_shared.rs` and change the include to a plain
`mod threats_shared;`. The file is self-contained (depends only on
`hexo_engine::{HexCoord, HexoState, TurnPhase}`), so nothing else moves.

**Verify:** `cargo check -p hexfield` in the staged repo.

## 2. Rust workspace trim

Root `Cargo.toml`:
- `members`: keep `packages/hexo_engine`, `packages/hexo_utils`, `packages/hexfield`;
  remove `packages/hexo_models`.
- Remove any workspace-dep entries that only served hexo_models.
- Regenerate `Cargo.lock` (`cargo update --workspace` or delete + rebuild), then
  `cargo test` per kept crate.

## 3. Python packaging note (document, don't "fix")

hexfield is intentionally **not pip-installed** — it's imported via PYTHONPATH
(spec §5.1 packaging discipline; the frontend's `debug_infer.py` injects
`packages/hexfield/python` onto `sys.path`). Keep this pattern but document it
prominently in the public README (a `PYTHONPATH` line or a small `env.sh`),
since it will surprise outside readers. The native module is built with
`maturin develop --release` into the active venv (`scripts/build_native.sh`).

## 4. Frontend: prune legacy lineage branches

`packages/hexo_frontend/python/hexo_frontend/debug_infer.py`:
- Delete the `_hexgt()` (line ~609) and `_dense()` lazy loaders and their
  checkpoint-loading branches; keep `_hexfield()` (line ~862) and its input
  builders (`_load_hexfield_checkpoint`, `_hexfield_inputs`).
- Simplify `_detect_lineage` to hexfield-or-error with a clear message.
- Trim `tests/test_debug_infer.py` cases that exercise the deleted branches.

`static/app.js` / `index.html` / `styles.css`: **leave as-is.** The lineage
strings there are defensive conditional UI branches that degrade gracefully;
ripping out ~115 references risks UI regressions for zero functional gain.
Optionally revisit after release.

## 5. Public config: `configs/hexfield_main_7.toml`

Edits (consider renaming to `configs/hexfield_main.toml` as "the config"):

| Line (current) | Edit |
|---|---|
| `run.output_dir = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7"` (~152) | → relative `runs/hexfield_main_7` (in-repo, gitignored) |
| `checkpoint.initialize_from = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7_prefit/checkpoint_epoch3.pt"` (~168) | → `models/hexfield_main7_prefit.pt` if we ship it, else comment out with a note pointing at `scripts/prefit_launch.sh` |
| `multi_stage_eval.opponents.permanent_anchors` (~147) — absolute paths to main_5/main_6 checkpoints | → remove (public users have no anchors) or point at the shipped `models/*.pt`; document what anchors do |
| `sealbot_enabled = true`, `sealbot_share = 0.25` (~136–144) | → `sealbot_enabled = false` default, comment explaining how to enable with an external SealBot build + `$SEALBOT_PATH` |
| Header comments referencing `docs/PLAN_MAIN7_PERF_ROADMAP.md`, private run history, main_6 comparisons | → rewrite header as a clean annotated "this is the recipe" dossier (the annotation style is a feature — keep the *why*, drop the private *history*) |
| Env-driven arch (`HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 HEXFIELD_TRUNK=CCACCACCACCACCA`) — currently set in the systemd unit | → document in the config header AND set as defaults in the templated systemd unit / launch script, so a bare `python -m hexo_train.cli.train_model` matches the shipped weights |

Also scrub `configs/hexfield_smoke*.toml` the same way.

## 6. Scripts cleanup

General rules for every kept script:
- Rename per the manifest table (drop `_` prefixes, drop `main7` naming).
- Replace `/root/.venvs/hexgt-build`, `/root/.venvs/hexfield-dev` with a
  `$HEXO_VENV` env var (default: `.venv` at repo root).
- Replace `/mnt/e/Hexo-BotTrainer-hexgt` / `E:\...` with repo-relative paths
  (`$(dirname "$0")/..`).
- `scripts/dashboard.sh` (from `_dashboard_launch.sh`): remove hardcoded
  `--sealbot-path /mnt/e/SealBot` (line 21) → honor `$SEALBOT_PATH` if set;
  remove `packages/hexo_models/python` and `packages/dense_cnn_restnet/python`
  from PYTHONPATH (line 16).
- `scripts/supervise.sh` (from `_hexfield_supervise_main1.sh`): parameterize
  CONFIG/RUNDIR via env as it already supports; strip main_1–6 defaults.
- systemd units: template `WorkingDirectory`, venv path, and env vars with
  `%h`/placeholder comments; present as an optional Linux convenience, not a
  requirement — the supervisor script alone must be sufficient.

## 7. NEW: `scripts/export_weights.py`

Small script that loads a training checkpoint (`epoch_XXXXXX.pt`), drops
optimizer/scheduler/EMA-bookkeeping state, and saves a weights-only `.pt` with
an embedded arch metadata dict (channels, heads, trunk string, epoch, run name).
Used to produce `models/hexfield_main7_infer.pt`. Verify the export loads in
both the Debug workbench and the Match arena.

## 8. Repo scaffolding (new files)

- `LICENSE` — MIT, current year, your name.
- `.gitattributes` — `models/*.pt filter=lfs diff=lfs merge=lfs -text`.
- `.gitignore` — fresh: `runs/`, `target/`, `__pycache__/`, `.pytest_cache/`,
  `*.so`, `*.pyd`, `.venv/`, `data/*` caches (keep `.gitkeep`s), `*.log`.
- Optional but recommended: `.github/workflows/ci.yml` — `cargo test` for the
  three crates + the CPU-safe pytest subset (torch-dependent tests already
  `importorskip`-guard). Keeps the public repo verifiably green.
