# 01 — Keep / Cut Manifest

Exact inventory for the `hexo-bot` public repo. "Keep" means copy into
`E:\hexo-bot` (possibly with edits per [02_CODE_SURGERY.md](02_CODE_SURGERY.md)).
Anything not listed as Keep is cut — the extraction is **allowlist-based**, so
new scratch files can never leak by accident.

Kept packages additionally get **internal slimming** per
[06_CODE_SLIMMING.md](06_CODE_SLIMMING.md) (dead PUCT knobs, A/B perf branches,
dead subsystems) — this manifest covers whole files/dirs; 06 covers paths
inside them.

## Packages

| Path | Verdict | Notes |
|---|---|---|
| `packages/hexo_engine/` | **KEEP** | Authoritative Rust rules engine + PyO3 bridge. Self-contained. |
| `packages/hexo_utils/` | **KEEP** | `.hxr` codec, state_hash, encoding/symmetry, sample store. Needed by runner (Python), train (Python), and hexfield (Rust workspace dep). |
| `packages/hexo_runner/` | **KEEP** | Match loop, player contracts, records, SealBot adapter (kept — optional at runtime). |
| `packages/hexo_train/` | **KEEP** | Config-driven orchestration. Loads hexfield via `module = "hexfield.plugin"` — no entry-point or legacy dependency. Legacy lineage mentions are comments only. |
| `packages/hexo_frontend/` | **KEEP** | Dashboard + Match arena + Debug workbench. `debug_infer.py` needs legacy-branch pruning (02 §4). |
| `packages/hexfield/` | **KEEP** | The bot. Own Rust crate (`hexfield._rust`) + Python/PyTorch/Triton. Needs `threats_shared.rs` vendored in (02 §1). |
| `packages/hexo_models/` | **CUT** | Legacy umbrella crate — **except** `rust/src/threats_shared.rs`, which is vendored into `packages/hexfield/rust/src/` (02 §1). Nothing else survives. |
| `packages/dense_cnn_restnet/` | CUT | Legacy lineage (main_1–4 era). |
| `packages/hexgnn/` | CUT | Legacy parked lineage. |

## Root files

| Path | Verdict | Notes |
|---|---|---|
| `Cargo.toml` | **KEEP (edited)** | Workspace members become `hexo_engine`, `hexo_utils`, `hexfield` only (02 §2). |
| `Cargo.lock` | **KEEP (regenerated)** | Regenerate after workspace edit. |
| `README.md` | **REWRITE** | Public version per [05_PUBLIC_DOCS_PLAN.md](05_PUBLIC_DOCS_PLAN.md). Current one describes dense_cnn_restnet/main_4 and private environment. |
| `HANDOFF.md` | CUT | Internal engineering log — private paths, run forensics, worktree names. |
| `.gitignore` | **REWRITE** | Fresh minimal version (runs/, __pycache__/, target/, data caches, *.so). |
| `LICENSE` | **NEW** | MIT. |
| `.gitattributes` | **NEW** | LFS patterns for `models/*.pt`. |
| `_arena_ep20.log`, `_dashboard_bridge*.py`, `_hexgt_bc*.pt`, `_optimism_main3.py`, `_rl_run_fg.sh`, `_tmp_*.py` | CUT | Root scratch files. |
| `__pycache__/`, `.pytest_cache/`, `target/` (1.8 GB), `.claude/` | CUT | Build output / caches / private tooling (`.claude/settings.local.json` references private sibling repos). |
| `runs/` (in-repo), `archive/`, `analysis/` | CUT | Legacy run outputs, investigation archives, scratch analysis. |

## Configs

| Path | Verdict | Notes |
|---|---|---|
| `configs/hexfield_main_7.toml` | **KEEP (edited)** | The flagship config. Parameterize paths, default SealBot off, document env-driven arch (02 §5). |
| `configs/hexfield_smoke.toml`, `configs/hexfield_smoke_tiny.toml` | **KEEP (rewritten — Gumbel)** | Quick-start configs so people can verify the pipeline in minutes without a GPU-day. Rewritten as mini-main_7 Gumbel profiles on CPU (06 §1) — the current classic-PUCT smoke profile would keep legacy knobs alive. Same path scrub. |
| All other `configs/*.toml` (dense_cnn*, hexgt*, hexgnn*, hexfield_main_1–6, soak, eval scratch) | CUT | Legacy/experimental. main_1–6 configs reference private run dirs and tell a private-history story. |
| `configs/runs/` | CUT | Run-local artifacts. |

## Scripts

Public scripts get renamed without the `_` scratch prefix (mapping in 02 §6).

| Source | Verdict | Public name |
|---|---|---|
| `scripts/_hexfield_supervise_main1.sh` | **KEEP (edited)** | `scripts/supervise.sh` — the training supervisor loop |
| `scripts/_main7_launch.sh` | **KEEP (edited)** | `scripts/launch_training.sh` |
| `scripts/_rebuild_hexfield.sh` | **KEEP (edited)** | `scripts/build_native.sh` — maturin release build |
| `scripts/_dashboard_launch.sh` | **KEEP (edited)** | `scripts/dashboard.sh` — drop hardcoded `--sealbot-path`, legacy PYTHONPATH entries |
| `scripts/_hexfield_prefit.py` | **KEEP (edited)** | `scripts/prefit.py` — behavioral-cloning warm start |
| `scripts/_main7_prefit_data.sh`, `_main7_prefit_launch.sh` | **KEEP (merged/edited)** | `scripts/prefit_launch.sh` — fold into one documented script |
| `scripts/_hexfield_fetch_corpus.py` | **KEEP (edited)** | `scripts/fetch_corpus.py` — pulls the public HF bootstrap corpus |
| `scripts/bootstrap_hexfield_hf.py` | **KEEP (edited)** | `scripts/bootstrap_from_corpus.py` |
| `scripts/systemd/hexfield-supervisor-7.service` | **KEEP (templated)** | `scripts/systemd/hexo-bot-supervisor.service` — placeholder paths, documented as optional Linux convenience |
| `scripts/systemd/hexfield-dashboard.service` | **KEEP (templated)** | `scripts/systemd/hexo-bot-dashboard.service` |
| Everything else in `scripts/` (~200+ files: `_wf_*`, `_main6_*`, `_deploy_*`, probes, logs, `remote-control.ps1`, `scripts/archive/`, `__pycache__`) | CUT | Private workflow/scratch. |
| **NEW**: `scripts/export_weights.py` | **NEW** | Strips optimizer state from a training checkpoint → inference-only `.pt` (02 §7). |

## Tests

| Pattern | Verdict |
|---|---|
| `tests/test_hexfield_*` (23 files), `tests/hexfield_eval_kit.py`, `tests/hexfield_testkit.py` | **KEEP** |
| `tests/eval_dashboard/` | **KEEP** (hexfield eval-dashboard verifiers; check for SealBot-path assumptions) |
| `tests/katago_buffer/` | **KEEP minus `test_p6_pool.py`** (replay-buffer port used by hexfield; `test_p6_pool.py` pins the dead `pool` expand backend — cut with it, 06 §4) |
| `tests/test_hexfield_{search_parity,continuous_parity,divergence_properties,rust_parity}.py` | **KEEP (parity net)** — updated when the dead knobs they pass are stripped (06 §2) |
| `tests/test_hexo_engine_rust_bridge.py`, `test_sealbot_adapter.py` | **KEEP** |
| `tests/test_hexo_runner_match_mode.py` | **KEEP (trimmed)** — its `run_batch` cases go with `run_batch` (06 §4); `run_match` cases stay |
| `tests/test_frontend_training_{artifacts,epoch,live}.py` | **KEEP** |
| `tests/test_debug_infer.py` | **REWRITE** — currently written entirely against the hexgt lineage; replace with a small hexfield-lineage test of the debug worker (06 §5) |
| `tests/conftest.py` | **KEEP (review)** — check for legacy fixtures |
| `tests/test_hexo_utils_sample_store.py` | CUT — pins the dead `hexo_utils/samples` store (06 §4) |
| `tests/test_dense_cnn_*` (~24), `test_hexgt_*` (~35), `test_hexgnn_*` (~9), `test_restnet_*` (4), `test_hexo_models_*` (2), `test_pipeline_depth2_determinism.py` (pins dead depth2 pipeline), `test_training_pipeline_simplification.py` | CUT |

## Docs

| Path | Verdict |
|---|---|
| `docs/intro_to_hexo.md` | **KEEP** — the game rules; ideal for the study audience |
| `docs/ARCHITECTURE.md` | **KEEP (rewritten)** — trim legacy lineages, describe the hexfield-only system |
| `docs/hexfield_blueprint.md`, `docs/specs/hexfield_model_spec.md`, `hexfield_eval_v2_spec.md`, `hexfield_v2_fixes.md`, `hexfield_v2_synthesis.md` | **KEEP (path-scrubbed)** — the bot design docs; core study material |
| `docs/specs/{match,debug,history}_screen_v2*.md` | **KEEP (path-scrubbed)** — frontend contracts |
| `packages/*/README.md` for kept packages | **KEEP (reviewed)** — scrub legacy-lineage references (esp. `hexo_models` mentions) |
| `docs/analysis/`, `docs/investigation/`, `docs/HANDOFF_*.md`, `docs/EVAL_DASHBOARD_FIXES.md`, `docs/INVESTIGATION_*.md`, `docs/IMPROVEMENT_BACKLOG.md`, `docs/PLAN_katago_replay_buffer_port.md`, `docs/hexfield_cleanup_*.md`, `docs/PLAN_MAIN7_PERF_ROADMAP.md` | CUT — internal forensics/planning with private paths. (Perf roadmap is referenced by main_7 config comments; strip those comment references in the public config.) |
| `docs/public-release/` (this folder) | CUT — the plan itself stays private |

## Data & weights

| Path | Verdict | Notes |
|---|---|---|
| `data/hexo-bootstrap-corpus/` | CUT (fetchable) | It's the public HF dataset `timmyburn/hexo-bootstrap-corpus`; `scripts/fetch_corpus.py` re-downloads it. Keep only its `README.md`/`SCHEMA.md` if useful, or link the HF card. |
| `data/hexfield_main5_prefit/`, `data/hexfield_bootstrap*` | CUT | Private prefit shard symlinks/caches. **Symlink targets leak absolute paths — never copy.** |
| `data/{checkpoints,replay,selfplay}/.gitkeep` | KEEP | Empty scaffolding dirs recreated fresh. |
| **NEW** `models/hexfield_main7_infer.pt` | **NEW (LFS)** | Inference-only export of the latest good main_7 epoch (~35 MB). |
| **NEW** `models/hexfield_main7_full.pt` | **NEW (LFS)** | Full training checkpoint (~98 MB) for training resume. |
| **NEW** `models/MODEL_CARD.md` | **NEW** | Arch env vars (`HEXFIELD_CHANNELS=192`, `HEXFIELD_ATTENTION_HEADS=3`, `HEXFIELD_TRUNK=CCACCACCACCACCA`), training provenance, eval strength. |
| Optional: `models/hexfield_main7_prefit.pt` | **DECIDE AT PUBLISH** | The warm-start checkpoint `initialize_from` points at. Shipping it makes the full training recipe reproducible; otherwise document warm start as optional. |
