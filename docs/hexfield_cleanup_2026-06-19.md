# hexfield cleanup — branch `claude/hexfield-cleanup` (2026-06-19)

A reviewed, test-gated cleanup of the `hexfield` package (Python + Rust), its
scripts, and its test suite. Driven by a per-file opus review of every module
(28 Python + 11 Rust) plus a synthesized, dependency-ordered plan; applied in an
**isolated git worktree** so the live `hexfield_main_3` training run was never
touched. Net change: **107 files, +861 / −9,430 lines (≈ −8,600).**

> The thesis from the review: hexfield is **not** a tangled mess — the live
> kernels (search, expand, serve, train, eval) are healthy and well-tested. The
> bloat was concentrated in (1) comment/doc archaeology, (2) a handful of dead
> symbols, (3) ~68 one-off investigation scripts, and (4) duplicated eval tests.
> One real **correctness bug** was found and fixed along the way.

---

## 1. Headline: a live correctness bug (fixed here)

`replay_expand.rs::expand_shard_train` (the Rust train-read kernel the live run
uses, `expand_backend="rust"`) **never read the per-row `outcome_valid` column
nor emitted `value_mask`.** Once truncated-game training went live (ep46), every
Rust-expanded **truncated** row trained its outcome heads (value / short-term
value / cell-Q) against a masked-as-valid `0.0` target — silently corrupting the
value signal on truncated games.

**Fix:** the kernel now reads `outcome_valid` (u8) and, for `outcome_valid==0`,
sets `value_mask=0.0` and zeroes `stvalue_mask` + `cell_q_mask` — **byte-for-byte
identical to the serial Python oracle** (`samples.py` truncated path). Completed
rows (`outcome_valid==1`) are unchanged. `expand_backends._reassemble_rust_rows`
now consumes the emitted `value_mask`.

**Guard:** `tests/katago_buffer/test_p7_rust_parity.py` gained
`test_truncated_rows_rust_eq_serial`, which forces `outcome_valid=0` and asserts
the Rust backend equals the serial oracle across all 12 D6 symmetries (masks
zeroed + targets element-equal). It passes.

> **Live-run note:** the deployed run is still affected until this branch ships.
> To stop the bleeding immediately without adopting the branch, set
> `HEXFIELD_EXPAND=serial` on `hexfield-supervisor-3.service` (the serial oracle
> handles truncated rows correctly), or adopt this branch.

---

## 2. Verification (all green, isolated worktree)

- **Build:** `maturin develop --release` clean (7 pre-existing pyo3 deprecation
  warnings only). The `.so` is mirrored into the worktree, never the live tree.
- **Final gate:** `113 passed, 2 skipped` (the 2 skips are CUDA-fp16 and a
  native-ABI case that skip cleanly CPU-only). `PYTEST_EXIT=0`.
- Every stage was gated by an isolated rebuild + targeted pytest and **aborted
  on failure rather than forcing green**. No stage aborted.
- Suite covered: geometry, support, features, model, targets, rust_parity,
  search_parity, continuous_parity, seed_contract, divergence_properties,
  plugin, evaluate_epoch, eval_stats, and `katago_buffer/test_p7_rust_parity`
  (incl. the new truncated case).

Reproduce in the worktree:
```bash
# build (hexfield-dev venv)
source /root/.venvs/hexfield-dev/bin/activate && export PATH=/root/.cargo/bin:$PATH
maturin develop --release -m /mnt/e/hexgt-cleanup/packages/hexfield/Cargo.toml
# test (hexgt-build venv); fixtures: tests/katago_buffer/_scratch/p5/samples (from main_3 v3 samples)
cd /mnt/e/hexgt-cleanup && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
  PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
  /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexfield_*.py tests/katago_buffer/test_p7_rust_parity.py -q
```

---

## 3. What changed, by category

### 3a. Dead code removed (verified zero callers)
| Where | Removed |
|---|---|
| `features.py` | `facts_from_records()` |
| `geometry.py` | `neighbor()` (+ now-unused `DIRECTIONS` import) |
| `model.py` | `_exact_lut` buffer + its sole-use `disk_offsets` import |
| `evaluation.py` | dead `from . import _rust` import |
| `legacy_model_v2.py` | `_BiasGather`, dict-returning `forward()`, `set_attention_impl` + the `materialized` attention branch, `_exact_lut`, the training-only grad branch of `build_attn_bias` (frozen eval snapshot — **all head params kept** so strict load still works; serve-flex + inference path intact) |
| `state.rs` | `state_from_py_state` (single-state wrapper) |
| `search.rs` | `continuous_flush_decision_pub`, `RustSearch::clear`, `debug_dump`, `debug_no_advance` param+branch |
| `cache.rs` | `RustEvaluationCache::clear` (dead once `RustSearch::clear` gone) |

### 3b. Comment / doc trims (comment-only, no logic change)
Stripped PLAN-§/M-milestone archaeology, brittle `file.py:NNN` cross-citations
(many had drifted), "byte-identical-to-v1" migration narration, incident
post-mortems, and `dense_cnn` port references across the big files (`model.py`,
`trainer.py`, `window.py`, `inference.py`, `expand_backends.py`, `samples.py`,
and the Rust `search.rs`/`tree.rs`/`lib.rs`/`payload.rs`/`replay_expand.rs`).
**Load-bearing rationale was preserved** (lr-reapply footgun, masked-softmax
`-inf` requirement, seed/determinism contracts, virtual-loss/in-flight safety,
ABI byte-exactness, the `#![allow(dead_code)]` attribute and build-venv warning).
Fixed factual drift: `tree.rs` root-policy-temp comment (live = 1.07, not 1.0),
`replay_expand.rs` moves-left comment (`/MOVES_LEFT_CAP`, not `/512`),
`serve_pack.rs` reworded from "HOT-PATH ENTRY" to "flag-gated alt arm
(`HEXFIELD_RUST_PACK`, off)".

### 3c. Scripts: 68 orphans deleted, 31 kept
Deleted (each re-grepped for zero live references first): the throughput/
concurrency/compile/flex benches & sweeps, `lr_finder`/`lr_microsweep`, audit
inspectors, the soak cluster, epoch-specific retrain/migrate one-shots, and
spy/wait/stop scripts whose conclusions already live in project memory.
**Kept** (live/build/test/eval-tool/parity-oracle): `_hexfield_supervise_main1.sh`,
`_rebuild_hexfield.sh`, `_hexfield_run_multistage_eval.py`, `_hexfield_run_suite.sh`,
`_hexfield_run_smoke.sh`, `_hexfield_run_head_audit.py`, `_hexfield_buffer_convert.py`,
`_hexfield_learning_ladder.sh`, `_hexfield_systemd_reset.sh`, `bootstrap_hexfield_hf.py`,
and the shipped-comment parity oracles (`_hexfield_serve_ref.py`,
`_hexfield_plan_groups_parity.py`, `_hexfield_ref_puct.py`, `_hexfield_bias_check.py`,
`_hexfield_compile_diag.py`, `_hexfield_flex_probe.py`, `_hexfield_check_prod_config.py`, …).

### 3d. Tests: 7 eval files → 3 + shared kit; stale tests repaired
- **New** `tests/hexfield_eval_kit.py` — the engine/session fakes and arena/
  orchestrator stubs that were copy-pasted/cross-imported across 5 files.
- **Merged** `multistage_eval + eval_harden + eval_parts` → `test_hexfield_eval_orchestrator.py`
  (each behavior asserted once: SealBot fail-open, anchor pin/fallback,
  budget floor, SPRT label-mapping, pure-eval invariant, parts/resume).
- **Merged** `eval_arena_concurrent + eval_arena_native + eval_concurrent_multi` →
  `test_hexfield_eval_arena.py` (fake-engine oracle + real-ABI native CRN oracle
  + `play_multi == N-serial` equivalence; forced-opening-replay asserted once).
- **Repaired stale tests** (were red on `main`): `model.bias_table` → per-block
  `bias_tables` ParameterList; param-count `1,230,651 → 1,591,748` (current
  8 conv + 3 attn blocks, 3 per-block bias tables, cell_q head);
  `decode_moves_left_median` reconciled to the 65-bin `[0, MOVES_LEFT_CAP=209]`
  decode; plugin optimizer-split + `train_passes` API updated to the PackedWindow
  contract.

---

## 4. New structure

The **module set is unchanged** — no live module was removed; the package was
trimmed in place. Reachability map for the live `hexfield_main_3` path:

```
configs/hexfield_main_3.toml → hexfield.plugin (HexfieldPlugin)
  ├─ build_model            → model.HexfieldNet
  ├─ training overrides     → trainer.HexfieldTrainer, checkpoints, config, train_state
  ├─ generate_selfplay      → selfplay → inference.HexfieldEvaluator (serve)
  │                                    → _rust.HexfieldMctsSession.run_continuous   [search.rs]
  │                                    → features / samples / support / geometry / shards
  ├─ train step             → trainer → window.PackedWindow
  │                                    → expand_backends (backend="rust")           [replay_expand.rs]
  │                                    → batching → losses → model
  └─ evaluate_epoch (×5)    → evaluation → multistage_eval → eval_arena → eval_stats
                                         → head_audit; legacy_model_v2 (old-arch anchor loader)
```

**Live Rust → Python entry points (`lib.rs`):** `HexfieldMctsSession`
(`search.rs`), `expand_shard_train` (`replay_expand.rs`), `build_serve_groups`
(`serve_pack.rs`, gated by `HEXFIELD_RUST_PACK`). The `debug_*` / `featurize_states`
/ `mix_seed` exports are test-only.

### Deliberately KEPT (load-bearing — do not "clean up")
- **Parity oracles:** serial + pool expand backends, the Python CSR serve-pack,
  `read_compact_shard`, `debug_plan_groups`, the materialized-bias path — these
  guard the live Rust kernels via tests. Removing them removes the safety net.
- **`legacy_model_v2.py`:** frozen eval-only snapshot; the **only** way the eval
  arena loads the old-arch `main2_ep45` anchor. Not dead.
- **`prefit.py`:** off the main training path but is the BC-prefit reproduction
  recipe and is imported by kept prefit/bench scripts. Doc-trimmed only.

---

## 5. Isolation & safety

- All work occurred in worktree `E:/hexgt-cleanup` (`/mnt/e/hexgt-cleanup`); the
  live tree `E:/Hexo-BotTrainer-hexgt` stayed on `main` with zero tracked changes
  throughout. The live supervisor imports hexfield from `$ROOT/packages/hexfield/python`
  in the live tree, so this separation is what kept the run safe.
- Rust builds installed into the isolated `hexfield-dev` venv; the live run uses
  `hexgt-build`. Builds/tests were serialized (one at a time) to avoid contending
  for the box's RAM with the active GPU run.
- Nothing was merged or deployed. To adopt: review this branch, then either fast-
  forward `main` or rebuild + restart the supervisor onto it (and the
  `HEXFIELD_EXPAND` interim mitigation can be dropped once the Rust fix is live).
