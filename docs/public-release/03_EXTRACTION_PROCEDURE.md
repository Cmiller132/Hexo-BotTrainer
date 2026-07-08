# 03 — Extraction Procedure

Ordered phases. Each phase ends with a checkable gate. Total ≈ two focused days.

## Phase 1 — Stage the skeleton (~30 min)

1. `git init E:\hexo-bot`; `git lfs install` and commit `.gitattributes` first
   (LFS patterns must exist **before** any `.pt` is added, or the weights land
   as regular blobs).
2. Create the new scaffolding files: `LICENSE` (MIT), `.gitignore`,
   `.gitattributes`, empty `models/`, `runs/.gitkeep`.

**Gate:** `git lfs track` lists `models/*.pt`.

## Phase 2 — Copy the keep-set (~30 min)

Copy per [01_KEEP_CUT_MANIFEST.md](01_KEEP_CUT_MANIFEST.md) — **allowlist copy**
(script the copy from the manifest; do not copy-everything-then-delete):

- 6 packages (engine, utils, runner, train, frontend, hexfield)
- 3 configs (main_7 + two smoke)
- ~10 scripts + 2 systemd templates (with renames)
- kept tests + `conftest.py` + `eval_dashboard/` + `katago_buffer/`
- kept docs
- root `Cargo.toml` / `Cargo.lock`

Copy with symlink-dereference **disabled** and verify no symlinks came along
(`data/hexfield_main5_prefit` symlinks are cut anyway, but check).

**Gate:** `git status` in the staged repo shows only intended paths; no
`__pycache__`, no `.so`, no symlinks (`find -type l` empty).

## Phase 3 — Code surgery (~2–3 h)

Apply [02_CODE_SURGERY.md](02_CODE_SURGERY.md) in order: vendor
`threats_shared.rs` → workspace trim → `debug_infer.py` pruning → config
parameterization → script renames/path scrubbing → `export_weights.py`.

**Gate:** in WSL, in a **fresh venv** (not `hexgt-build` — that's the point):
```
python -m venv .venv && . .venv/bin/activate
pip install maturin torch numpy pytest
scripts/build_native.sh          # maturin develop --release for all 3 crates
cargo test -p hexo_engine -p hexo_utils -p hexfield
```
Rust builds green with `hexo_models` absent.

## Phase 3b — Code slimming (~4–6 h)

Apply [06_CODE_SLIMMING.md](06_CODE_SLIMMING.md) in its §7 order, **only after
the Phase 3 gate is green** (slim a building repo, not a broken one). One commit
per step in the staging repo so a bad cut is cheap to bisect:

1. Rewrite smoke configs as Gumbel mini-main_7 (gate: CPU epoch completes).
2. Python subsystem cuts — runner/train/utils/frontend dead code (gate: kept
   pytest subset green).
3. Rust A/B env-branch cuts — depth2, gate-complete, no-prefetch, serial
   backup/complete, fp8 (gate: `cargo test` + gumbel smoke).
4. Deep PUCT-knob strip — Dirichlet noise, forced playouts, dynamic/visit-scaled
   c_puct, fpu-under-noise (gate: parity suite + gumbel smoke + `cargo test`).
5. Eval simplification — legacy_model_v2, foreign-PUCT anchor path, regime
   labels (gate: `test_hexfield_eval_*` green, dashboard eval panel works).

**Final gate:** behavioral A/B — same seed, same tiny config, main_7 search
profile, identical move choices and visit distributions pre/post-strip. The
strip removes only off-branches; any behavioral drift is a bug.

## Phase 4 — Weights + LFS (~1 h)

1. Pick the latest **gated-good** main_7 epoch at publish time (check the run's
   eval diagnostics, not just the newest file — the run is live).
2. `scripts/export_weights.py <epoch.pt> models/hexfield_main7_infer.pt`
3. Copy the same epoch as `models/hexfield_main7_full.pt`.
4. Decide on `models/hexfield_main7_prefit.pt` (ship = fully reproducible
   recipe; skip = document warm start as optional).
5. Write `models/MODEL_CARD.md` (arch env vars, epoch, params, eval strength,
   provenance, corpus link).

**Gate:** `git lfs ls-files` shows the `.pt` files; fresh clone + `git lfs pull`
round-trips them; the Debug workbench loads the inference export.

## Phase 5 — Docs rewrite (~2–3 h)

Per [05_PUBLIC_DOCS_PLAN.md](05_PUBLIC_DOCS_PLAN.md): new root README, trimmed
ARCHITECTURE.md, path-scrubbed specs, reviewed package READMEs, model card.

**Gate:** grep the staged tree for legacy lineage names in docs
(`dense_cnn|hexgt|hexgnn|restnet`) — remaining hits are intentional
(e.g. "history" footnote) or code comments explicitly kept.

## Phase 6 — End-to-end verification in WSL (~1–2 h)

All in the fresh venv, from a **fresh clone** of the staged repo (clone catches
missing-file and LFS mistakes that the working tree hides):

1. `pytest tests/ -q` — kept subset green (torch tests run in WSL; guard-skips
   count as failure to investigate, not success).
2. Smoke training: `python -m hexo_train.cli.train_model configs/hexfield_smoke_tiny.toml`
   — completes an epoch loop end-to-end (now the Gumbel rewrite of the config,
   so this also re-proves the slimmed search path on CPU).
3. Dashboard: `scripts/dashboard.sh`, load the shipped inference weights in the
   Debug workbench, play a Match arena game vs the bot.
4. Prefit path: `scripts/fetch_corpus.py` downloads the HF corpus;
   `scripts/prefit_launch.sh` starts (can abort after first steps).
5. Optional: clone SealBot from <https://github.com/Ramora0/SealBot>, build it,
   set `$SEALBOT_PATH`, confirm the arena offers it and eval works with
   `sealbot_enabled = true`.

**Gate:** all five pass from the fresh clone.

## Phase 7 — Sanitize + publish (~30 min)

1. Run every sweep in [04_SANITIZATION_CHECKLIST.md](04_SANITIZATION_CHECKLIST.md).
2. Squash to the single initial commit (`git checkout --orphan` or re-init;
   verify `git log` shows exactly one commit).
3. Create the GitHub repo `hexo-bot` (public), enable LFS, push.
4. Post-publish: confirm GitHub renders the README, LFS files download, and a
   clean `git clone` + Phase 6 steps 1–3 work on a machine/user without access
   to any `E:\` private paths.

## Publish-day dependency

main_7 is mid-run. The plan freezes **code now, weights at publish time** —
re-run Phase 4 with the final chosen epoch just before pushing, and update the
model card's epoch/strength numbers to match.
