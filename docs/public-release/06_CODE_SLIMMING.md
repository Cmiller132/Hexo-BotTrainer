# 06 — Code Slimming (main_7 as the baseline)

Strip unused and legacy code paths *inside* the kept packages so the public repo
contains only what the main_7 configuration actually exercises (plus the
CPU/debug/smoke paths that keep it usable, and the parity harness as the
correctness net).

**Locked decisions:**
- Slimming happens in the **public copy only** (`E:\hexo-bot`). The dev repo is
  untouched — the live run executes from it.
- **Parity harnesses stay**: `search_parity_mode`, the Rust `parity()`
  divergence profile, and `tests/test_hexfield_{search_parity,continuous_parity,divergence_properties,rust_parity}.py`.
- **Smoke configs are rewritten as Gumbel** (mini-main_7 on CPU), which makes the
  classic-PUCT auxiliary knobs globally dead → **deep strip** of those knobs.

Baseline definition: `configs/hexfield_main_7.toml` + the env set by
`scripts/systemd/hexfield-supervisor-7.service` (`HEXFIELD_ASYNC_EVAL=1`,
`DEFER_DECODE=1`, `SERVE_FLEX=1`, full Triton stack; **no** `PIPELINE_DEPTH2`,
`GATE_COMPLETE`, `NO_PREFETCH`, `SERIAL_BACKUP`, `SERIAL_COMPLETE`, `CONV_FP8`).

## 0. What is explicitly NOT removable (looks dead, isn't)

| Item | Why it stays |
|---|---|
| Core PUCT machinery (`c_puct`, `select_or_materialize_edge`, σ/completed-Q math in `tree.rs`/`search.rs`) | Gumbel is layered on it: `Divergences::gumbel()` = `production()` + 4 bools; `c_puct` is a required arg on every search entry point and feeds completed-Q even under Gumbel |
| Eager (non-Triton) attention/conv paths in `model.py` | All fast kernels are guarded by `x.is_cuda`; the frontend debug worker is CPU-only by construction (`debug_infer.py:26-29`) and always takes eager |
| Sync (`!async_eval`) branch in `run_continuous` (`search.rs:~1167`) | Default fallback; CPU/tests use it |
| `serial` expand backend | Default; the (rewritten) smoke configs and CPU tests use it. Only `pool` is dead (see §4) |
| `virtual_loss`, `moves_left_utility` + `ml_weight/ml_scale/ml_q_gate/ml_final_pick` defaults | Live defaults main_7 relies on |
| Pure-Python debug PUCT tree (`search_tree_position`, `_select_puct`, `debug_infer.py:1659-1853`; `/api/debug/search_tree`) | Live Tree Explorer frontend feature, not a legacy search path |
| `run_match` (`hexo_runner/modes/match.py`) | Live: Match arena (`web.py:627`) |
| `hexo_utils/records.py` + `hexo_utils/encoding/` | Live `.hxr` codec; live D6 symmetry via `hexo_train/symmetry.py:19`. (Fix the `hexo_utils/__init__.py` docstring that mislabels `encoding` as LEGACY.) |
| Parity harness (`search_parity_mode`, `parity()` profile, 4 parity tests) | Kept by decision — reference-vs-native bug-localization net + study material |

## 1. Smoke configs → Gumbel rewrite (prerequisite for §2)

Rewrite `hexfield_smoke.toml` / `hexfield_smoke_tiny.toml` as mini-main_7:
Gumbel block on (`gumbel_root_enabled`, `gumbel_sequential_halving`,
`gumbel_nonroot_select`, `gumbel_target_enabled`), `forced_playout_k = 0.0`,
`root_dirichlet_noise_fraction = 0.0`, `c_scale = 0.0`, tiny arch env
(`HEXFIELD_CHANNELS`/`TRUNK` small), `device = "cpu"`, `expand_backend = "serial"`,
few games/epochs. Model on `tests/test_hexfield_gumbel_smoke.py`, which proves
Gumbel-on-CPU works.

**Gate:** smoke config completes an epoch on CPU in a fresh venv.

## 2. Deep strip: classic-PUCT auxiliary knobs (hexfield config + Rust)

Now globally dead (main_7 forces them off; rewritten smoke no longer revives
them; parity profile has them off by definition):

| Knob / path | Locations | Action |
|---|---|---|
| Dirichlet root noise machinery | `config.py` (`root_dirichlet_noise_fraction`, `root_dirichlet_total_alpha`, `dirichlet_shaped`); `tree.rs:94,153,180`; noise application at root in `search.rs` | Remove knobs + noise-application code; `Divergences` loses the noise fields |
| Forced playouts + dynamic c_puct | `config.py` (`forced_playout_k`, `pruned_dynamic_cpuct`); `tree.rs:97,1005`; `search.rs:2671,2855` (forced-playout pruning) | Remove |
| Visit-scaled c_puct | `config.py` (`visit_scaled_c_puct`, `c_base`, `c_scale`); `tree.rs:59-61,1015` (`c_for` scaling) | Remove scaling; `c_for` collapses to static `c_puct` |
| FPU-zero-under-noise | `config.py:61`; `tree.rs:77-81,158`; `search.rs:548,646,981` | Remove (only the parity tests mention it — update them to stop passing the field) |
| `ml_auto_disabled` heal-gate | `config.py:90,394` (`ML_AUTO_DISABLED_FLAG`, `build_divergence_overrides:415`); `eval_arena._resolve_eval_overrides` | Remove (run-dir flag mechanism; no kept config or ops tooling in the public repo writes it) |

**Order:** do this AFTER the repo builds green post-extraction (see 03 Phase 3b).
Update `Divergences::{parity,production,gumbel}()` constructors and the parity
tests together; the parity suite is the net proving search behavior is unchanged
for the live profile.

**Gate:** `cargo test -p hexfield` + all 4 parity tests + `test_hexfield_gumbel_smoke.py` green.

## 3. Env-gated Rust A/B perf branches (all default-off, bench-only)

| Branch | Location | Action |
|---|---|---|
| `HEXFIELD_PIPELINE_DEPTH2` (`run_continuous_pipeline_depth2`) | `search.rs:1085-1114,1407` | Remove fn + pyfunction export; cut `tests/test_pipeline_depth2_determinism.py` |
| `HEXFIELD_PIPELINE_COMPLETE_OVERLAP` | `search.rs:1441` (inside depth2) | Dies with depth2 |
| `HEXFIELD_GATE_COMPLETE` | `search.rs:1123` | Remove (decision-identical A/B toggle; only `_main6_*` bench scripts used it) |
| `HEXFIELD_NO_PREFETCH` | `search.rs:1077,1180` | Remove (keep the prefetch path unconditionally) |
| `HEXFIELD_SERIAL_BACKUP` | `search.rs:2097-2117` | Remove — keep the parallel `par_iter_mut` backup (2119-2152) unconditionally |
| `HEXFIELD_SERIAL_COMPLETE` | `search.rs:2198` | Remove — keep parallel Phase-A completion unconditionally |
| `HEXFIELD_CONV_FP8` | `model.py:126-133,363-414`; `_triton_conv.hex_conv_ln_fp8` | Remove branch + kernel (commented out in prod with a value-deviation note; never shipped) |
| `debug_plan_groups` pyfunction | `lib.rs` export; `serve_pack` | Remove export (only caller was a non-kept diagnostic script). Keep `build_serve_groups` (live) |

## 4. Dead subsystems in hexo_runner / hexo_train / hexo_utils

Most carry in-tree `UNUSED(2026-06-12)` markers already:

| Item | Evidence | Action |
|---|---|---|
| `hexo_runner/modes/evaluation.py` | Never-implemented stub (raises `NotImplementedError`) | Remove + `modes/__init__.py` re-export |
| `hexo_runner/modes/batch.py::run_batch` | Only caller is a test | Remove; trim the `run_batch` cases from `tests/test_hexo_runner_match_mode.py` (keep its `run_match` cases) |
| `hexo_runner/cli.py` + `hexo-rl` entry point | Only ref is the pyproject script entry | Remove both |
| `hexo_runner/config.py` (`RunnerConfig`) | No importers | Remove |
| `hexo_runner/session.py` legacy aliases | Pre-GameSpec naming, no callers (`session.py:46`) | Remove aliases |
| `hexo_train/config.py` YAML loader | No `.yaml` config exists (`config.py:261`) | Remove + drop PyYAML dep |
| `hexo_train/epoch/selfplay.py::build_selfplay_request` branch | No plugin implements it — unreachable (`selfplay.py:52-56`) | Remove dispatch branch |
| `hexo_train/registry.py` entry-point/name modes | main_7 uses module mode (`module = "hexfield.plugin"`); the `hexo_train.models` entry-point group is populated only by cut packages | Remove `_load_from_entry_point` / `_load_by_name`; keep module mode. Update registry docstring |
| `hexo_utils/samples/` subtree | Bypassed in production (`uses_shared_sample_store=False` everywhere); target helpers "functionally inert" (`defaults.py:33-38`) | Remove subtree + the inert wiring in `hexo_train/defaults.py` + the shared-store path in `hexo_train/epoch/samples.py:38,103`; cut `tests/test_hexo_utils_sample_store.py` (and `test_training_pipeline_simplification.py`, already cut) |
| `pool` expand backend | Dead in production (main_7: `rust`; smoke: `serial`); pinned only by `tests/katago_buffer/test_p6_pool.py` | Remove backend + cut `test_p6_pool.py`; keep `serial` and `rust` |

## 5. Frontend dead surface

| Item | Evidence | Action |
|---|---|---|
| `debug_infer.py` legacy lineage branches (`_dense_pkg`, `_load_dense_checkpoint`, `_search_dense`, `_load_hexgt_checkpoint`, `_search_hexgt`, `_build_dense_model`, dense plane tables) | Serve only cut packages | Remove (was already planned in 02 §4; scope confirmed) |
| `web.py` `STATIC_MAX_AGE_SECONDS` | `web.py:127`, marked UNUSED | Remove |
| `web.py` `/api/training/file` route + `_send_training_file` | `web.py:1280,1547` — no app.js fetch, no tests | Remove |
| `web.py` `_training_histories` | `web.py:3810` — pre-paging builder, superseded | Remove |
| `tests/test_debug_infer.py` | Written entirely against the **hexgt** lineage (`hexgt_rl_main3` fixtures) | **Rewrite** as a small hexfield-lineage test (the debug worker is a kept feature and deserves coverage); use a smoke-size hexfield checkpoint fixture |
| `dashboard.py` | Grep found no lineage-specific code, but not read end-to-end | Verify-read during Phase 3b; no planned cut |

## 6. Eval: consequences of dropping private anchors

The public config ships **no main_5/main_6 anchors** (02 §5), which kills two
code paths that are load-bearing only for foreign PUCT-era checkpoints:

| Item | Evidence | Action |
|---|---|---|
| `legacy_model_v2.py` (old 6-conv-block checkpoint loader) | `eval_arena.py:220-226` fallback; only needed for legacy-arch anchor checkpoints — all shipped checkpoints are main_7 arch | Remove |
| `puct_eval_overrides()` + `_foreign_opponent_overrides` PUCT-anchor path | `eval_arena.py:265`, `multistage_eval.py:1265` — served main5/main6 anchors | Remove the foreign-PUCT profile path; keep foreign-opponent plumbing only if the shipped-checkpoint anchors need it (they're same-arch/same-profile — verify, then simplify) |
| Zero-point regime provenance labels (`puct era` / `budget-starved` / `budget-calibrated`, `multistage_eval.py:1066-1068`) | Labels of private run history; the public pool starts fresh in the calibrated-gumbel regime | Collapse to the single current regime (small edit) |

## 7. Suggested execution order (slots into 03 as Phase 3b)

1. §1 smoke rewrite (gate: CPU epoch completes)
2. §4 + §5 Python subsystem cuts (gate: kept pytest subset green)
3. §3 Rust env-branch cuts (gate: `cargo test`, gumbel smoke, bench sanity)
4. §2 deep PUCT-knob strip (gate: parity suite + gumbel smoke + `cargo test`)
5. §6 eval simplification (gate: `test_hexfield_eval_*` green, dashboard eval panel works)

Each step is a separate commit in the staging repo (history gets squashed at
publish anyway) so a bad cut is cheap to bisect and revert.

## 8. Verification additions

- After §2, run one **behavioral A/B**: same seed, same tiny config, search the
  same positions pre/post-strip with the main_7 profile — move choices and visit
  distributions must be identical (the strip only removes off-branches).
- Sweep for orphaned references after each step:
  `grep -rE "PIPELINE_DEPTH2|GATE_COMPLETE|NO_PREFETCH|SERIAL_BACKUP|SERIAL_COMPLETE|CONV_FP8|forced_playout|dirichlet|c_base|visit_scaled|fpu_zero|ml_auto_disabled|run_batch|RunnerConfig|hexo-rl|build_selfplay_request|uses_shared_sample_store|legacy_model_v2|puct_eval_overrides"`
  — remaining hits must be intentional (parity-test comments, docs explaining
  the Gumbel-vs-PUCT design choice).
