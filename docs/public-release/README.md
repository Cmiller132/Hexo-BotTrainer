# Public Release Plan: `hexo-bot`

Plan for extracting a clean, minimal, public repository from `Hexo-BotTrainer-hexgt`
containing the game engine, frontend, runner, trainer, and the **hexfield main_7**
bot — for other people to study.

## Locked decisions

| Decision | Choice |
|---|---|
| Public repo name | **hexo-bot**, staged locally at `E:\hexo-bot` |
| Git history | **Fresh single-commit repo** (no filtered history — zero leak risk) |
| Weights | **Both** in-repo via **Git LFS**: inference-only export (~35 MB) + full training checkpoint (~98 MB). Latest good main_7 epoch at publish time. |
| SealBot | **Optional external dependency** — adapter code kept, eval defaults off, docs link to <https://github.com/Ramora0/SealBot> |
| License | **MIT** |
| Model lineage | **hexfield only** (dense_cnn_restnet, hexgt, hexgnn all cut) |
| Legacy-path slimming | **Deep strip in the public copy only**: main_7 is the baseline; smoke configs rewritten as Gumbel; classic-PUCT auxiliary knobs, A/B perf env branches, and dead subsystems removed. Parity harnesses kept as the correctness net. |

## Plan documents

| Doc | Contents |
|---|---|
| [01_KEEP_CUT_MANIFEST.md](01_KEEP_CUT_MANIFEST.md) | Exact file/directory keep-vs-cut list with reasons |
| [02_CODE_SURGERY.md](02_CODE_SURGERY.md) | Source edits required for the kept subset to build & run standalone (Rust vendoring, config parameterization, script cleanup, frontend pruning) |
| [03_EXTRACTION_PROCEDURE.md](03_EXTRACTION_PROCEDURE.md) | Step-by-step phases: stage → copy → surgery → weights/LFS → verify → publish |
| [04_SANITIZATION_CHECKLIST.md](04_SANITIZATION_CHECKLIST.md) | Pre-publish audit: private paths, secrets, symlinks, final grep sweeps |
| [05_PUBLIC_DOCS_PLAN.md](05_PUBLIC_DOCS_PLAN.md) | What the public README/docs/model-card should say |
| [06_CODE_SLIMMING.md](06_CODE_SLIMMING.md) | Dead/legacy paths inside the kept packages to strip (PUCT knobs, A/B perf branches, dead subsystems), with the load-bearing "looks dead but isn't" list |

## Key findings that shaped the plan

1. **hexfield is nearly self-contained.** Pure Python/PyTorch/Triton plus its own
   Rust crate (`hexfield._rust`). It imports only `hexo_engine`, `hexo_runner.records`,
   and `hexo_train.components` from the rest of the repo. No dependency on the
   legacy `hexo_models._rust` umbrella — **except one hidden build-time include**:
   `packages/hexfield/rust/src/lib.rs:20` pulls in
   `packages/hexo_models/rust/src/threats_shared.rs` via `#[path]`. The fix is to
   vendor that single file into hexfield (see 02).
2. **Plugin discovery already bypasses entry points.** `hexfield_main_7.toml` sets
   `module = "hexfield.plugin"`, so `hexo_train` loads hexfield by direct import.
   The trainer works with only hexfield present; legacy lineage mentions in
   `hexo_train` are comments only.
3. **The frontend is model-agnostic at package level.** Only
   `hexo_frontend/debug_infer.py` references legacy lineages, and lazily — those
   branches get pruned for the public build.
4. **SealBot is already optional at runtime** (`$SEALBOT_PATH` / `--sealbot-path`;
   graceful "unavailable" reporting). The only default-on wiring is in the main_7
   eval config (`sealbot_enabled = true`) and a hardcoded path in
   `scripts/_dashboard_launch.sh` — both get flipped.
5. **The main_7 config leaks private absolute paths** (`/mnt/e/Hexo-BotTrainer/runs/...`
   for `output_dir`, `initialize_from`, eval anchors). The public config gets
   relative paths and optional warm-start.
6. **No secrets found** in the tree (token/key/password sweep clean). The risks are
   personal absolute paths, internal handoff/forensics docs, and scratch files —
   all handled by the fresh-history + manifest approach.
7. **PUCT is the substrate, not a legacy path.** Gumbel search is layered on the
   PUCT machinery (`c_puct` feeds the completed-Q math), so the core stays. What
   *is* removable — after rewriting the smoke configs to Gumbel — are the
   classic-PUCT auxiliary knobs (Dirichlet noise, forced playouts, dynamic/visit-scaled
   c_puct), six default-off A/B perf env branches in the Rust search, the fp8
   conv kernel, and ~10 dead subsystems across runner/train/utils/frontend. The
   CPU eager kernel fallbacks look legacy but are load-bearing (the frontend
   debug worker is CPU-only). Full inventory in 06.

## Effort estimate

Roughly two focused days: staging + copy (~1 h), code surgery (~2–3 h),
**code slimming per 06 (~4–6 h — the deep PUCT-knob strip touches the Rust
search core and needs the parity/A-B gates)**, weights export + LFS (~1 h),
docs rewrite (~2–3 h), verification in WSL (build, tests, dashboard, play a
match) (~1–2 h), sanitization sweep (~30 min).

## Safety note

The live main_7 run executes **from this working tree** under WSL. The extraction
is copy-based (nothing in this repo is moved, deleted, or edited), so it cannot
disturb the run. Do all surgery in `E:\hexo-bot`, never in place.
