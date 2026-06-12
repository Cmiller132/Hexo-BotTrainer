# Improvement Backlog

Compiled 2026-06-12 from a full-repo audit (11 code mappers + 10 comment-pass agents).
Section 1 lists code that was **marked in-source** with `UNUSED(2026-06-12)` / `DEPRECATED(2026-06-12)`
comments during this pass; Section 2 lists structural rewrite candidates; Section 3 lists smaller cleanups.
Top claims were spot-checked against the working tree before writing (see the note at the end).

Recommended actions: **delete** (no consumer, safe to remove), **keep-ref** (keep as documented
reference / API mirror until a lineage decision), **extract** (live logic in the wrong place),
**confirm** (needs owner sign-off first).

---

## 1. Deprecated / unused code (marked in-source this pass)

### 1a. Marked with UNUSED/DEPRECATED comments

#### dense_cnn_restnet

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 1 | `packages/dense_cnn_restnet/python/dense_cnn_restnet/player.py` — entire module (`DenseCNNPlayer`) | Zero imports of `dense_cnn_restnet.player` repo-wide; all live `DenseCNNPlayer` users (`scripts/_head_to_head.py`, `scripts/_rl_train.py`, `scripts/_rl_train_hexgnn.py`, `tests/test_dense_cnn_pipeline.py`) import `hexo_models.dense_cnn.player`. Restnet eval/selfplay drive `mcts.new_mcts_session` directly. | delete |
| 2 | `.../dense_cnn_restnet/debug_artifacts.py` — entire module (`render_preview_game_actions`) | Zero references anywhere; the only test of this renderer imports `hexo_models.dense_cnn.debug_artifacts`. The frontend renders boards client-side via `debug_infer.py`. | delete |
| 3 | `.../dense_cnn_restnet/d6.py` — `compose_indices()` | Only the definition matches repo-wide (the hexo_models sibling has its own copy); no restnet caller or test. | delete |
| 4 | `.../dense_cnn_restnet/performance.py` — `build_benchmark_report` (restnet copy) | Exported via `__init__.py` but no caller; the only test exercises the `hexo_models.dense_cnn` copy. `calibrate_dense_cnn` in the same module IS live. | delete (or keep-ref for API symmetry with parent lineage) |
| 5 | `.../dense_cnn_restnet/samples.py` — `_optional_int()` | Only the definition at samples.py:452 matches anywhere. | delete |
| 6 | `.../dense_cnn_restnet/replay.py` — `INPUT_KEY`..`NPZ_KEYS` dense-tensor key constants (DEPRECATED) | Restnet shards use the compact columnar schema (`compact_io.py`); no importer of `dense_cnn_restnet.replay` references these constants; users of the names all resolve to `hexo_models.dense_cnn.replay`. | delete |

#### hexgnn (whole package is parked — see Section 2 item R2)

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 7 | `packages/hexgnn/python/hexgnn/rust_bridge.py` — `capabilities()` and `candidate_ids()`, plus their Rust exports (`rust/src/lib.rs capabilities()`, `rust/src/candidates.rs hexgnn_candidate_ids`) | All callers of these names resolve to the hexgt/dense_cnn bridges; zero callers of the hexgnn copies. Consumers get candidate ids through `graph_facts`. API-surface mirrors of hexgt. | keep-ref until retire decision, then delete with the crate |
| 8 | `packages/hexgnn/python/hexgnn/evaluation.py` — `BatchedSearcher` Protocol, `HexgnnBatchedSearcher`, `run_head_to_head_parallel()` | Only definitions exist; live callers (`scripts/_rl_ablate.py`, `tests/test_hexgt_parallel_eval.py`) use the `hexo_models.hexgt` twins. The hexgnn RL driver imports only the sequential `run_head_to_head`. | delete |
| 9 | `packages/hexgnn/python/hexgnn/mcts.py` — `per_root_visits`/`per_root_forced_playout_k`/`per_root_noise` override path | No caller ever passes non-None (hexgnn selfplay ships PCR as two separate full/fast `run()` calls); tests exercise the hexgt twin only. | delete (mirror of the hexgt decision, item 22) |
| 10 | `packages/hexgnn/rust/src/threats.rs` — `hexgnn_threat_analysis` `#[pyfunction]` export only | Only definition + register line repo-wide; internal `analyze()`/`tactical_cells()` ARE live in mcts.rs/mcts_tree.rs. TSS fixtures/probes drive the hexgt copy. | delete the Python export, keep internals |
| 11 | `packages/hexgnn/rust/src/vcf.rs` — `hexgnn_vcf_solve` `#[pyfunction]` | Self-documented as a benchmark artifact "NOT wired into the live MCTS"; `scripts/_vcf_bench.py` and `tests/test_hexgt_vcf.py` drive `hexgt_vcf_solve` only. | delete |

#### hexo_engine

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 12 | `packages/hexo_engine/python/hexo_engine/types.py:102` — `format_coord_id()` | Only the definition matches repo-wide; not exported in `__init__`, never called. | delete |
| 13 | `types.py:~152` — `PythonTerminal` dataclass | Never constructed: `api.to_python_state` assigns a `TerminalResult` to `PythonHexoState.terminal` despite the `PythonTerminal \| None` annotation. Spot-checked: confirmed, README documents the same. | delete + fix the annotation (the type lie is item S-E1) |
| 14 | `packages/hexo_engine/rust/src/board.rs:143` — `Board::bounds()` | Zero `.bounds()` callers in any crate; encoders compute their own crop from `occupied_cells()`. | delete |
| 15 | `packages/hexo_engine/rust/src/legal.rs:60` — `LegalMoveStore::version()` accessor | No callers anywhere; advertised "for cache users" but none exist. The underlying counter is still maintained/restored on every placement and undo. | delete accessor; decide on the counter + delta field together (S-E2) |

#### hexo_frontend

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 16 | `web.py` — `STATIC_MAX_AGE_SECONDS` (line 117) | Only the definition exists; `_send_static` now serves index.html no-store and app.js/styles.css no-cache, so no header reads it. Spot-checked: confirmed. | delete |
| 17 | `web.py` — `GET /api/training/file` route + `_send_training_file` | No app.js fetch, no test, no doc beyond a README mention. Plausibly a manual raw-artifact download URL (json/jsonl/png/hxr). | confirm with owner, then delete or document as a manual endpoint |
| 18 | `web.py` — `_training_histories()` (~line 3120) | Only the definition; superseded by the streaming pipeline `_history_files_for_runs` + `_history_rows_for_file` + `_hxr_base_rows`. | delete |
| 19 | `static/app.js` — `loadMoreArtifacts()` (~line 744) | Former callers `renderTraining`/`trainingArtifactRow` no longer exist anywhere in app.js (removed by Match-v2 rewrite). It was the ONLY client of `GET /api/training/artifacts-page`, which is now tests-only — consider retiring that endpoint + `tests/test_frontend_training_artifacts.py` together. | delete (and decide on the endpoint) |
| 20 | `static/app.js` — `summaryMetric()` (~2849), `historyEpochs()` (~3005), `averageHistoryLength()` (~3061) | Only the definitions match repo-wide; orphaned by the Match-v2/History-v2 rewrites. `docs/specs/history_screen_v2_spec.md` §3 still lists them as load-bearing — amend the spec (S-F1). | delete |

#### hexo_models / dense_cnn (parent lineage)

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 21 | `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/debug_artifacts.py` — entire module | Only reference is its own test `tests/test_dense_cnn_debug_artifacts.py`; docstring self-declares "not part of the production self-play or training path". | delete with its test, or keep-ref if PNG previews are wanted later |
| 22 | `.../dense_cnn/performance.py` — `build_benchmark_report` (~line 152) + `__init__` re-export | Only callers are `tests/test_dense_cnn_performance.py:369-373` and the re-export; `plugin.py` calls `calibrate_dense_cnn` only. Same pattern dead in the restnet fork (item 4). | delete in both lineages together |
| 23 | `.../dense_cnn/d6.py` — `compose_indices` (~line 115) | Exactly two hits repo-wide: this definition and the restnet fork's (item 3). `inverse_index` in the same file IS live. | delete in both lineages together |

#### hexo_models / hexgt

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 24 | `packages/hexo_models/hexgt/rust/src/candidates.rs` — `PositionGraph.node_waxis` field + `nodes['node_waxis']` export | Spot-checked: writers only inside candidates.rs; neither featurizer (features.py / features.rs) reads it (axis labels deliberately excluded for D6 invariance); the hexgnn fork already DROPPED the column. Serialized into every `graph_facts` payload for nothing. | delete (saves a Vec build + clone + PyO3 serialization per featurized position) |
| 25 | `per_root_visits`/`per_root_forced_playout_k`/`per_root_noise` override transport — marked at all three layers (`mcts.py` run signature, `rust_bridge.py mcts_session_search`, `rust/src/mcts.rs search()`) | Built for selfplay PCR ("coalescing full+fast into one batched call") but selfplay.py instead issues two separate `run()` calls per subset and never passes `per_root_*`. Exercised only by `tests/test_hexgt_mcts_per_root.py` + `test_hexgt_pcr.py`. | decide: delete end-to-end (incl. ~60 lines of rust validation + the two tests + hexgnn mirror, item 9) OR migrate PCR onto it to regain full-width leaf batching |

#### hexo_runner

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 26 | `packages/hexo_runner/python/hexo_runner/config.py` — `RunnerConfig` alias (whole module is a shim) | No importer of `hexo_runner.config` anywhere. | delete |
| 27 | `.../hexo_runner/cli.py` — `main()` + the `hexo-rl` script entry in pyproject.toml | `main()` unconditionally raises SystemExit (deliberate placeholder); no caller. | delete module + script entry |
| 28 | `.../hexo_runner/session.py:34-35` — `SessionSpec`/`SessionContext` aliases (DEPRECATED) | Only the definitions match; all callers use `GameSpec`/`GameContext` directly. | delete |
| 29 | `.../hexo_runner/modes/evaluation.py` — `run_evaluation` | Body is `raise NotImplementedError`; only the definition + re-export exist. Every model package built its own eval harness instead (see R6). | delete (fold into the R6 decision) |
| 30 | `.../hexo_runner/loop.py` — early `result = GameResult(..., status=ABORTED)` in `run_match_loop` | Verified by reading the function: never read before the unconditional reassignment after the try/except. | delete the assignment |

#### hexo_train

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 31 | `packages/hexo_train/python/hexo_train/epoch/selfplay.py:47-59` — the `build_selfplay_request` transitional branch (+ the unreachable placeholder branch below it) | Spot-checked: no plugin — real or test fake — implements the hook; all four registered plugins implement `generate_selfplay`, so the first branch always wins. | delete both branches |
| 32 | `.../hexo_train/config.py:246-255` — YAML branch of `_load_raw_config` + the PyYAML dependency | configs/ contains only .toml; no test or script uses YAML training configs; advertised in the CLI docstring but never called. | delete branch + drop PyYAML from pyproject (removes a live-venv runtime dep) |
| 33 | `.../hexo_train/defaults.py` — `ScalarValueTargetHelper()`/`LegalPolicyTargetHelper()` default instantiations | Wired into `ModelComponents` on every pipeline run but no plugin, trainer, or test ever reads the handles back. | delete with the hexo_utils targets pair (item 37) as one change |

#### hexo_utils

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 34 | `packages/hexo_utils/rust/src/pybridge.rs:439` — `capabilities()` pyfunction | Every `capabilities()` caller resolves to a model package's own rust_bridge; nothing calls `hexo_utils._rust.capabilities` despite its docstring claiming smoke-test use. | delete |
| 35 | `packages/hexo_utils/rust/src/records.rs:55` — `RecordError::WriteOnlyFile` variant | Never constructed — `iter_records()` on a Write-mode file reopens the path for reading instead; only the declaration, Display arm, and pybridge mapping arm exist. | delete variant + both arms |
| 36 | `.../hexo_utils/samples/buffer.py` — `iter_sample_records()` | Only the definition, `__init__` re-export, and `__all__`; tests and hexo_train use `read_sample_records` / the window API. | delete |
| 37 | `.../hexo_utils/samples/targets.py` — `LegalPolicyValueTarget` + `build_legal_policy_value_target` (DEPRECATED) | Only consumer is `tests/test_training_pipeline_simplification.py`; every model package builds its own targets. | delete with item 33; or fold into the R7 museum-vertical trim |

### 1b. Dead files / build outputs not markable in-source

| # | Item | Evidence | Action |
|---|------|----------|--------|
| 38 | `configs/_smoke_optimized_test.toml` | `git grep '_smoke_optimized'` returns nothing; derives from a retired config line. | delete |
| 39 | `scripts/goal_benchmark.py` | No wrapper, doc, or config invokes it; self-contained argparse CLI so possibly manual-use. | confirm with owner, then archive |
| 40 | `tests/{engine,integration,models,runner,utils}/` — .gitkeep-only directories | All 115 test files live flat in tests/; no conftest or doc references the layout. | delete the placeholder dirs |
| 41 | `scripts/setup_python_minidumps.ps1` | Hardcodes the crashdump folder of the retired `dense_cnn_model1_scratch_64` run; active line runs under WSL where WER minidumps do not apply. | delete |
| 42 | `packages/hexo_models/Cargo.toml` — `crate-type = ["rlib", ...]` | No crate under packages/ depends on hexo_models; cargo builds a test rlib automatically anyway. | drop `"rlib"` (marginal) |

---

## 2. Rewrite candidates

Grouped and deduplicated across the per-package findings. Effort: S (hours), M (days), L (multi-day / needs a design decision).

**R1. Lineage fork drift: `dense_cnn_restnet` vs `hexo_models/dense_cnn` (and `hexgnn` vs `hexo_models/hexgt`) — L.**
Most of the restnet package (d6, geometry, input, losses, mcts, rust_bridge, player, debug_artifacts, performance, compact_io, trt_backend) is a copy of the parent with the same "Model1" naming; fixes must land twice, and the dead modules in Section 1 are drift symptoms. Worse, the relationship is inverted: the ACTIVE lineage's hot search path (run_continuous, PCR, policy-init, root-noise) lives in the nominally-legacy `hexo_models/dense_cnn` Rust crate, so rebuilding for restnet silently changes parent search semantics (HANDOFF: "intended", but unguarded). The hexgnn/hexgt pair has the same disease with identical public names polluting greps. Decide on a shared-library extraction or a formal "parent is frozen" policy; until then every fix needs a double-landing checklist.

**R2. hexgnn retire-or-extract decision — M.**
The package is parked (HANDOFF.md) yet its Rust crate is `#[path]`-compiled into every hexo_models native build (`packages/hexo_models/rust/src/lib.rs:22`) — ongoing compile-time/binary cost for the active line, plus a cross-package `../../../hexgnn/rust/src/lib.rs` include that breaks if hexgnn moves and forces an sdist special case. Its packaging story is stale ("carries NO Rust of its own" in pyproject/config/scripts is false), its checkpoint format is split-brained (plugin saves `{'model':'hexgnn','model_state':...}` vs the driver's `{'model': state_dict, 'arch': meta}`, and `debug_infer.py`'s sniffer would misclassify the plugin shape as DENSE_RESTNET), and `threats.rs` is an unparity-tested fork of `threats_shared.rs` compiled into the same crate (contradicting threats_shared's "single source" claim). True retirement = remove the `#[path]` include + sdist glob + entry point; revival = diff-audit against hexgt first.

**R3. Selfplay monoliths — M per package.**
(a) `dense_cnn_restnet/selfplay.py` (~1650 lines): `_generate_selfplay_epoch_lockstep` and `_generate_selfplay_epoch_continuous` duplicate ~200 lines (inference construction, ~20 counters, the finalize → `_rows_to_write` → materialize → `write_selfplay_npz` chain, live-progress writers, length-EMA update, summary dict). A shared per-game-finalization helper + summary builder would roughly halve the file. (b) `hexo_models/hexgt/selfplay.py run_selfplay_games` (~460 lines, lines 334-801) and its hexgnn twin mix search orchestration, PCR bucketing, sanitization-taint bookkeeping, record/shard IO, and ~25 metric accumulators; the round-level taint (one sanitized logit discards every position decided that round) is acknowledged-too-coarse pending a Rust per-leaf sanitized flag.

**R4. hexo_frontend split: app.js (8.5k lines) and web.py (4.2k lines) — L / M.**
app.js holds three largely independent screens, ~80 module-level mutable state vars, and module-level DOM lookups; splitting needs an ES-modules-or-concat decision first (no build step exists). web.py mixes five concerns (HTTP routing, threaded ManualMatchController, training-run scanner/cache, history paging/cursors, debug glue); the training-scan side (`_training_run`/`_epoch_history`/`_learning_health`/`_training_live_status`) is a clean first extraction (M).

**R5. Restnet temperature control consolidation — M.**
Four stacked schemes (linear decay, anchor schedule, adaptive halflife EMA, opening floor) plus the KataGo root-policy ramp, with precedence encoded in `_move_temperature`'s branch order (selfplay.py:47-95) and duplicated as prose in config.py:176-222. An enum-dispatched scheme object would be much harder to misconfigure. Related: PCR/policy-init determinism exists twice (Python `_splitmix64_unit` for lockstep vs native `mix_seed` for continuous) with the contract living only in comments.

**R6. hexo_runner "modes" layer: build it or retire it — M.**
Four near-identical SealBot eval harnesses exist downstream (hexgt, hexgnn, dense_cnn, dense_cnn_restnet `evaluation.py` — identical line numbers betray copy-paste), exactly the duplication `modes/evaluation.py` was meant to absorb; the stub was never built (item 29). Batch mode (`run_batch`/`BatchSpec`/`PlayerFactory`) is similarly exercised only by `tests/test_hexo_runner_match_mode.py` — every model package reimplements its own multiprocessing selfplay loop. Either build the shared eval mode and migrate the four callers, or retire modes/ and stop implying a layer that does not exist.

**R7. Shared sample-store museum vertical (hexo_train + hexo_utils) — M.**
All four production plugins set `uses_shared_sample_store=False`, so `pipeline._initialize_run` → `epoch/samples.prepare_sample_store` and the default branches of `select_training_samples`/`finalize_samples` are dead at runtime; the entire `hexo_utils/samples/` subpackage (buffer.py 836 lines + records.py + targets.py) has no production consumer. Kept alive only by FakePlugin tests (`tests/test_training_pipeline_simplification.py`, `test_hexo_utils_sample_store.py`). A coordinated trim deletes ~1,200 lines but requires test changes; alternatively document explicitly as "generic-pipeline reserve". Includes items 33/36/37 and the unadopted `encoding/symmetry.py` ActionSymmetryMapper (both dense_cnn d6.py files carry independent `transform_action_ids` copies).

**R8. hexo_train plugin contract: convention → types — M.**
`components.py` is almost entirely `Any` despite `py.typed`; `registry.ModelPlugin` Protocol covers 2 of ~7 hooks actually dispatched via `hasattr` (generate_selfplay, evaluate_epoch, calibrate_performance, trainer hooks, loader's `{'status':'loaded','epoch':N}` resume shape). The resume contract in `epoch/loop.py:147-155` is load-bearing for main_4's epoch-6 fast-forward yet exists only as an implicit dict shape. Promoting the hooks to typed Protocols would also let the placeholder branches (checkpoints.py no-loader/no-saver, `CheckpointStore.write_placeholder`) become hard errors.

**R9. Per-epoch wasted D6 work on the active lineage — S/M.**
`epoch/symmetry.select_epoch_symmetries` computes one blake2b digest per visible sample every epoch (hundreds of thousands of rows on restnet windows), but `dense_cnn_restnet/trainer.py:_aug_seed` consumes only `selection.seed` and re-draws its own per-row symmetries — the whole tuple is computed and discarded and the `symmetry_count` diagnostic is misleading. Add a seed-only fast path when the trainer declares it re-draws.

**R10. rust mcts split now that two schedulers coexist — M.**
`hexo_models/dense_cnn/rust/src/mcts.rs` is ~2,670 lines mixing the session object, the lockstep scheduler, the continuous per-slot scheduler, root-noise config, and four diagnostics builders; `mcts_tree.rs` adds 1,609. Section markers were added this pass as an interim map; an actual module split is the fix.

**R11. Radius-20 crop design limitation — L (design, not refactor).**
Root cause of the main_3 collapse (memory: restnet-crop-frozen-win-zugzwang): the disk crop intentionally excludes out-of-crop legal engine moves from policy/MCTS (`encoding.rs`), freezing out-of-rim wins for both players — inherited by every consumer of the shared Rust. Tracked here so the backlog reflects it; any fix is an encoding-contract change (the contract itself is hand-triplicated: constants.py/geometry.py/input.py vs constants.rs/encoding.rs, no generated source of truth).

**R12. Repo/ops hygiene pass — M.**
(a) HANDOFF.md newest section is 2026-06-10; main_3's collapse and live main_4 are reconstructable only from config headers + docs/analysis + auto-memory. (b) Docs and configs cite evidence files moved into gitignored `scripts/archive/` (`_wf_r4_RESULT_*.json`, `_wf_gd_RESULT_*.json`, `_wf_r4_ckpt5_load_out.txt`) — dangling for any fresh clone; hexgt's design-doc citations all point at files deleted by commit b50e92a. (c) `scripts/_run_all_tests.sh`/`_run_dense_cnn_tests.sh` hardcode the SIBLING checkout and the OLD venv and omit dense_cnn_restnet from PYTHONPATH. (d) Flat scripts/ mixes four generations of one-off tooling (~133 files) with no live/retired convention; `_dc_restnet_supervise_main1.sh` is the generic supervisor misleadingly named for main1; three near-duplicate rebuild scripts include a documented wrong-checkout footgun. (e) 10 of 15 configs are retired with no `archive/` split (docs/analysis got one); `dense_cnn_restnet_main_4.toml` is mutated in place mid-run (config-as-run-journal). (f) `tests/test_dense_cnn_compact_io.py` has 4 known stale failures asserting pre-disk-crop semantics.

---

## 3. Smaller cleanups

### dense_cnn_restnet
- `trainer.py`: extract a shared `_forward_loss(batch)` — the forward/loss/autocast block is duplicated between `_optimizer_step` (~393) and `_run_validation` (~440); loss-keyword lists kept in sync by hand. Same fix applies to the `hexo_models/dense_cnn` twin.
- Promote cross-module private imports to public names: selfplay.py imports `performance._extend_mcts_diagnostic_batches`/`_summarize_mcts_diagnostic_batches` and `mcts._result_from_payload` (also in the parent lineage).
- `trt_backend.py _serialize_engine` (~126): the intended docstring sits AFTER two executable clamp statements — a discarded expression, not a docstring. Move it up.
- `config.py Model1ArchitectureConfig.residual_blocks` is parsed but explicitly IGNORED by `build_model` — silent no-op in an otherwise fail-fast config boundary; warn or reject when it disagrees with the blocks_type-derived depth.
- `checkpoints._is_initialize_only` reconstructs resume-vs-initialize intent by string-comparing the ref against config fields, defaulting to full-restore on surprise; the pipeline should pass intent explicitly.
- `replay.build_katago_shuffle`: the finally-block `rmtree(tmp_dir)` is correct only because `tmp_dir.rename()` makes `exists()` false — use an explicit success flag (same in the parent lineage's copy).
- `trainer.py` uses `Sequence` in an annotation without importing it (harmless under `from __future__ import annotations`, breaks typing tools).
- Bucketing duplicated: `compile_backend.bucket_sizes` must agree with `inference._bucket_batch_size` power-of-two mode — extract one helper.
- Naming debt: everything is still Model1/`model1_*`/`dense_cnn.*` (incl. diagnostics filenames the dashboard special-cases by lineage tag); rename blocked by the shared Rust ABI and on-disk contracts — acknowledged in `__init__.py`, revisit at the next format break.

### hexo_engine
- S-E1: fix the `PythonHexoState.terminal` annotation when deleting `PythonTerminal` (item 13).
- S-E2: `new_game(seed=..., scenario=...)` silently discards both args (`pybridge.rs let _ = seed`) while hexo_runner/web.py pass seed as if meaningful — drop the params or reject non-None; also remove `GameSpec.scenario` (run_match_loop raises on non-None anyway) and the LegalMoveStore version counter + delta field if no cache user appears (item 15).
- `engine_metadata()` builds a fresh HexoState + snapshot just to read `rules_version` — expose `HEXO_STATE_SNAPSHOT_VERSION` directly.
- `apply_with_delta` clones `previous_last_turn` (heap Vec) on every placement purely for undo — SmallVec/inline `[HexCoord; 2]` removes a per-simulation allocation under the MCTS hot loop.
- Action-ID packing is triplicated by hand (rust legal.rs, python types.py, app.js offset 32768) with persisted IDs in .npz/.hxr; only one test cross-checks — add a tri-language parity test or generated table.
- Snapshot machinery (`snapshot.rs` + `load_state`) is public-but-dormant (test oracle only); make `pub(crate)` or document as reserved API.
- `to_python_state` materializes every stone/legal coord/window entry as Python dicts per call — O(18 × placements); first thing to slim if dashboard polling intensifies (pairs with the dashboard.py item below).
- Pre-existing pyo3 deprecation in `state_api_capsule` (`Py::from_owned_ptr` → `Bound::from_owned_ptr`); trivial at next rebuild window.

### hexo_frontend
- S-F1: amend `docs/specs/history_screen_v2_spec.md` §3 "DO NOT TOUCH" — it still lists `renderTraining`/`trainingArtifactRow`/`loadMoreArtifacts` and the orphaned helpers as load-bearing; anyone honoring the spec will resurrect dead code. Also fix HANDOFF.md:249 (server is not Flask; `_dashboard_bridge.py` is a data mirror, not the server).
- Cache-bust token lives in three places (index.html ×2, `APP_VERSION` in app.js) that must move in lockstep — have web.py inject one token at serve time (project memory already records a re-bump caveat).
- `dashboard.py` builds the full tactics payload (per-window masks, immediate_wins, must_blocks, summary) on EVERY `/api/state` poll and replay, but app.js consumes only `tactics.threats` — trim or gate behind a query flag.
- `_debug_resolve_record_npz`'s trailing game-index fallback is the known-provably-wrong path (project memory) — return a miss reason instead of silently attaching a shard by parsed index.
- Permanent phone-debug instrumentation (`__diagBar` z-index 2147483647 + tap-echo) overlays every screen — add a visibility toggle or query-param gate.
- `_normalize_player_setup` keeps the legacy `mode`+`human_player` config shape solely for `tests/test_sealbot_adapter.py` fixtures — migrate the fixtures, kill the branch.
- Repeated mtime-sort idiom (6+ sites) — add `_mtime_or_zero(path)`.
- `debug_service.py` hardcodes `DEFAULT_WSL_PYTHON`; env overrides documented only in code — log the resolved worker command at startup.
- `_send_json` sha1-ETags every response incl. multi-hundred-KB payloads behind the 3s cache — memoize the hash alongside the cached payload if poll rates rise.
- Note: the live :8080 WSL instance may still serve the pre-Match-v2/History-v2 build until restarted via `scripts/_dashboard_launch.sh`.

### hexo_models (dense_cnn / hexgt / shared)
- `replay.py npz_row_count` silently swallows a corrupt JSON sidecar and reopens the shard — add a counter or one-time log.
- `evaluation.py _run_games_concurrent` (~190 lines) mixes SealBot lifecycle, batched search, opening-temp seeding, per-game bookkeeping — let `_EvalGame` absorb the state transitions.
- hexgt: duplicate temperature plumbing — `HexgtMctsSession.run` takes both scalar `temperature` and `move_temperatures`; selfplay passes both though the per-root list fully overrides.
- hexgt: `run_head_to_head` builds a fresh HexgtPlayer (fresh `model.to(device)`) per game; same in the hexgnn twin — a shared-evaluator sequential driver removes the per-game model move (pair must stay result-equivalent per `test_hexgt_parallel_eval.py`).
- hexgt: `trainer.train_on_shards` loads every shard's rows into one Python list — memory hazard if pointed at a full replay window (lineage halted; document or fix with a chunked reader).
- hexgt: `expand_stv_readout_columns` relies on `scripts/_rl_train.py` dropping shape-drifted optimizer state — move the reconciliation next to the graft.
- hexgt: stale docs — `expand_value_readout_columns`/`VALUE_READOUT_MULT` still describe the removed `[side|mean|max]` readout; candidates.rs header says radius "default 2, range [2,8]" vs actual default 3, range [2,4]; the hexgt `mcts_tree.rs` header still claims "no progressive widening" (the dense_cnn twin was corrected this pass).
- hexgt: `candidates.rs` re-derives `coords_within_radius` locally — export it from hexo_engine (same duplication in the hexgnn fork). Also verify whether `mcts_eval.rs read_values`' legacy `values` list fallback (~468-491) has any remaining caller (not grep-verified this pass).
- hexgt: add a lineage-status note in `hexgt/__init__.py`/README — permanently halted (epoch 40, 2026-06-05) but a hard dependency of hexgnn, the debug worker, and the shared native module.
- shared: `lib.rs` `_rust` pymodule registration sequence is copy-pasted three times — extract a helper fn/macro before a fourth lineage drifts.
- shared: replace the four-candidate `__path__` probe in `python/hexo_models/__init__.py` with a PEP 420 namespace package (or maturin python-packages config) to remove the documented wheel-layout fragility.
- shared: re-point `hexgnn/rust/src/threats.rs` at `crate::threats_shared` (mechanical — same crate) or add a hexgnn leg to the TSS parity test; fix the threats_shared "single source" overclaim.
- shared: featureless `cargo check -p hexo_models` trips a rustc ICE while rendering expected dead-code warnings — `#[cfg_attr(not(feature = "python"), allow(dead_code))]` on threats_shared items dodges it.
- shared: built Linux `.so` sits untracked inside the Python source tree; Windows-side imports silently lack `_rust` until the lazy ImportError guards fire — document or guard at import.
- threats min_hitting_set/push_unique use O(n²) `Vec::contains` dedup (both copies) — fine at ≤~12 cells; switch to HashSet/SmallVec if threat counts grow.
- hexgnn (if ever revived): `run_searches_to_targets` carries an unresolved "INTEGRATOR: confirm cache race window" review note (~474); `MAX_CANDIDATE_RADIUS=4` is a self-described dead validation bound (config.py validates nothing) — enforce or delete; `_dashboard_bridge_hexgnn.py` still labels hexgnn_rl_main1 "the ACTIVE run"; pyproject `description` string still carries the stale "no Rust" claim (comments were fixed this pass, the string needs a code change).

### hexo_runner
- `loop.py _start_players` takes an `adapter` parameter it never uses — drop it.
- `loop.py` clones the full primary state once per observer per action (two `clone_state` calls per move just for `TransitionEvent.state`) — share one read-only clone or clone lazily.
- `sealbot.py _moves_left_in_turn` hardcodes turn-phase semantics, duplicating engine rules in the adapter — derive from engine state so a rules change cannot silently desync the SealBot payload.
- `_SealBotProcess` shares one response queue for ready/decide with no request-id correlation — add an id echo so a stray worker line fails loudly instead of mis-pairing.
- pyproject.toml description still reads "Placeholder Python runner package" for a load-bearing package.
- HANDOFF.md:248 overstates scope ("run/process orchestration & supervision" vs single-machine game loop + adapters).

### hexo_train / hexo_utils
- Pointer-publishing duplicated: `artifacts.publish_selfplay_checkpoint_pointer` vs `checkpoints._publish_epoch_checkpoint_pointer` — extract one writer or delete the feature (only legacy configs set `update_checkpoint_pointer=true`).
- Committed `__pycache__` (cpython-312 and -314 .pyc) under `python/hexo_train/` and `epoch/` — untrack.
- `hexo-engine`/`hexo-runner` are declared deps of hexo_train but never imported — move to an extras group.
- `ctx.section()`/`TrainingConfig.raw` escape hatch means most real configuration bypasses the typed boundary — document the boundary honestly or extend validation (overlaps R8).
- hexo_utils dependency contradiction: pyproject hard-requires lz4/zstandard while buffer.py imports them optionally with fallbacks and no config ever selects compression — demote to extras or delete the branches.
- `buffer.py read_sample_records`: vestigial `_ = store`; re-decompresses a whole chunk for any subset — footgun if the JSON store is revived.
- AbortRecord name collision: `hexo_utils._rust.AbortRecord` (PyO3) vs `hexo_runner.records.record.AbortRecord` (dataclass) work only via duck-typing — pick one canonical type or rename.
- `PyHexoRecord.replay()` forwards seed to `hexo_engine.new_game`, which discards it — fix belongs in hexo_engine (S-E2).
- hexo_utils has no rebuild script of its own — after editing rust/src the WSL venv silently keeps a stale `_rust` .so; add a sibling maturin one-liner.

---

## How this list was produced

Findings were collected on 2026-06-12 by 11 code-mapper agents (one per package/unit:
dense_cnn_restnet, hexgnn, hexo_engine, hexo_frontend, hexo_models dense_cnn/hexgt/shared,
hexo_runner, hexo_train, hexo_utils, repo_layout) and 10 comment-pass agents that added
`UNUSED(2026-06-12)` / `DEPRECATED(2026-06-12)` markers in-source. Evidence is grep-based
(packages/, tests/, scripts/ excluding archive, configs/, docs/) plus direct reads for
control-flow claims. During compilation, five representative claims were re-verified against
the working tree (dense_cnn_restnet/player.py orphan, hexgt node_waxis, hexo_engine
PythonTerminal, hexo_frontend STATIC_MAX_AGE_SECONDS, hexo_train build_selfplay_request
branch) — all confirmed, including the in-source markers. Overlapping mapper/commenter
findings were merged; speculative items without file evidence were dropped. Note that
hexo_frontend (and parts of dense_cnn_restnet) carry uncommitted Match-v2/History-v2
changes, so line numbers describe the working tree as of 2026-06-12, not the last commit.
