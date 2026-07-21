# Hexo Bot Trainer Code Quality and Maintainability Review

Date: 2026-07-10

## Executive summary

The core has a solid test culture, but `hexfield_eq` and the MCTS implementations have accumulated duplicated code, global architecture assumptions, unsafe native boundaries, and oversized configuration surfaces. This review found three high-priority defects plus several subtler correctness and maintainability risks.

Recommended order of work:

1. Fix cross-width checkpoint loading, the `hexgnn` root-promotion sign error, and unsafe vector initialization.
2. Harden native input validation and centralize MCTS and serve-planner configuration.
3. Extract the shared MCTS core, split oversized modules, and replace global lint suppression with targeted annotations.

## High-priority findings

### 1. Foreign-width `hexfield_eq` checkpoint loading is broken

`infer_net_kwargs_from_state_dict()` reconstructs `channels`, heads, and trunk depth from a checkpoint, and the arena loader passes those values to `HexfieldNet` ([model.py](../packages/hexfield_eq/python/hexfield_eq/model.py#L1374), [eval_arena.py](../packages/hexfield_eq/python/hexfield_eq/eval_arena.py#L227)). However, several internals still use the import-time global `CHANNELS` and `C_ORBIT`, including normalization, layer scale, tokens, and equivariant matrices ([model.py](../packages/hexfield_eq/python/hexfield_eq/model.py#L807), [equivariant.py](../packages/hexfield_eq/python/hexfield_eq/equivariant.py#L28)).

This was reproduced by constructing a 192-channel model in the default 96-channel process. Construction succeeds, but forward evaluation fails in `einsum` with a 192-versus-96 dimension mismatch. The checkpoint metadata tests pass because they save and reload under the same imported architecture ([test_hexfield_eq_checkpoint_meta.py](../tests/test_hexfield_eq_checkpoint_meta.py#L47)).

Suggested fix:

- Make orbit width entirely instance-derived and pass it into every component; or
- Explicitly reject mismatched checkpoints and relaunch a process with metadata-derived environment settings.
- Add a subprocess test that creates a checkpoint under one width and loads it under another process default.

### 2. `hexgnn` MCTS flips values incorrectly when root promotion keeps the same player

Root promotion unconditionally negates the edge value in [`hexgnn`'s `mcts_tree.rs`](../packages/hexgnn/rust/src/mcts_tree.rs#L709). Connect6's first-to-second stone transition keeps the same player, so this sign flip is incorrect.

The equivalent `hexgt` implementation already compares the old and new players before flipping ([`hexgt`'s `mcts_tree.rs`](../packages/hexo_models/hexgt/rust/src/mcts_tree.rs#L730)). Git history indicates that the sign correction was applied to dense/hexgt but missed `hexgnn`.

The package is parked, but it still compiles and can be loaded through legacy tooling. It also lacks a same-player root-promotion regression test.

### 3. Legacy dense MCTS uses an unsafe uninitialized vector

[`mcts_eval.rs`](../packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs#L260) allocates vector capacity and calls `set_len(total)` before initializing the `f16` elements, then creates mutable slices over those elements. Clippy reports this as `uninit_vec`; it violates Rust's initialization contract.

Suggested fix:

- Use initialized storage such as `vec![f16::ZERO; total]`; or
- Use `MaybeUninit`/spare capacity and set the vector length only after every element has been written.

## Correctness and robustness findings

### 4. Native byte and CSR input validation is insufficient

The Rust serve packer trusts offset tails and casts signed offsets to `usize` without validating start-at-zero, monotonicity, or bounds ([serve_pack.rs](../packages/hexfield_eq/rust/src/serve_pack.rs#L239)). Replay expansion also reinterprets `PyBytes` through unsafe typed slices and trusts CSR metadata ([replay_expand.rs byte decoding](../packages/hexfield_eq/rust/src/replay_expand.rs#L835), [replay expansion](../packages/hexfield_eq/rust/src/replay_expand.rs#L964)).

The Python shard loader does not fully validate these invariants before native dispatch ([window.py](../packages/hexfield_eq/python/hexfield_eq/replay/window.py#L364)). A corrupt or truncated shard can therefore panic the native extension instead of producing a clean Python error.

Suggested improvements:

- Validate all offsets as nonnegative, monotonic, start-at-zero, and bounded by their corresponding arrays.
- Validate scalar shapes and CSR data lengths when loading a shard.
- Avoid assuming `PyBytes` alignment is suitable for arbitrary typed slices; use checked decoding or owned typed storage.
- Return `PyValueError` for malformed inputs before entering unsafe code.

### 5. Rust and Python serve-pack planners have drifted

Rust hardcodes `NUM_TOKENS = 8` and its packing thresholds ([serve_pack.rs](../packages/hexfield_eq/rust/src/serve_pack.rs#L31)). The current Python model has six tokens and uses configurable thresholds plus a row cap ([constants.py](../packages/hexfield_eq/python/hexfield_eq/constants.py#L184), [inference.py](../packages/hexfield_eq/python/hexfield_eq/inference.py#L47), [Python planner](../packages/hexfield_eq/python/hexfield_eq/inference.py#L202)).

The native path therefore ignores tuning variables and can choose a different packing plan from the Python path.

Suggested improvements:

- Define one planner configuration schema and pass it into the Rust planner.
- Derive token count from the actual model architecture.
- Add Python/Rust parity tests across boundary cases and environment overrides.

### 6. MCTS divergence overrides validate names but not numeric domains

Rust extracts the override values without validating their ranges or finiteness ([search.rs](../packages/hexfield_eq/rust/src/search.rs#L2730)). Python likewise forwards arbitrary numeric values ([config.py](../packages/hexfield_eq/python/hexfield_eq/config.py#L498)).

Zero, negative, infinity, or NaN values can reach logarithms and divisors such as visit-scaled PUCT ([tree.rs](../packages/hexfield_eq/rust/src/tree.rs#L1058)). Invalid combinations can silently disable selection mechanisms or propagate non-finite scores.

Suggested fix: resolve overrides into one typed, validated configuration with finite/range/dependency checks. Examples include positive `c_base` and moves-left scale, nonzero Gumbel candidate counts, bounded fractions, and explicit dependencies between Gumbel modes and root configuration.

### 7. Requested policy logits can disappear silently

Search requests policy logits for Gumbel behavior ([search.rs](../packages/hexfield_eq/rust/src/search.rs#L691)), but the payload parser accepts their absence ([payload.rs](../packages/hexfield_eq/rust/src/payload.rs#L210)). The Gumbel root can then partially degrade or become a no-op instead of reporting an evaluator contract violation.

Missing requested data should be an explicit error, or all dependent mechanisms should be deliberately disabled with a visible diagnostic.

### 8. Evaluator cache identity is implicit

Session evaluation-cache entries are keyed by state while the evaluator is supplied separately on later searches ([search.rs](../packages/hexfield_eq/rust/src/search.rs#L531), [cache.rs](../packages/hexfield_eq/rust/src/cache.rs#L52)). Reusing a session after changing a model can return priors and values from the previous network.

Suggested alternatives:

- Include an evaluator/model generation in the cache key.
- Make evaluator identity immutable and part of session construction.
- Expose and require a `clear_cache` operation when changing evaluators.

### 9. Gumbel force-stuck completion is reported as successful early stopping

The force-stuck safety escape returns `early=true` ([search.rs](../packages/hexfield_eq/rust/src/search.rs#L2333)). Later code increments early-stop and visits-saved telemetry ([search.rs](../packages/hexfield_eq/rust/src/search.rs#L2454)).

This can hide sequential-halving saturation from health monitoring. It should have a distinct completion reason and separate counters.

## Dead code and maintainability

### Crate-wide dead-code suppression

`hexfield_eq` applies `#![allow(dead_code)]` to the entire crate ([lib.rs](../packages/hexfield_eq/rust/src/lib.rs#L12)). Forced dead-code linting exposed unused utility constants, `RustEvaluation.legal_action_count`, `RowFacts.gumbel_present`, unused diagnostics fields, and test-only helpers.

Some legacy FPU branches are intentionally retained for parity, but those cases should be annotated individually. Removing the crate-wide suppression would make future drift much easier to detect.

### Large-scale source duplication

`hexfield` and `hexfield_eq` contain byte-identical copies of `search.rs`, `tree.rs`, `cache.rs`, and `state.rs`, representing at least 8,419 duplicated Rust lines. Broader exact Python and Rust duplication exceeds 10,000 lines.

Build and runtime isolation can remain while the shared implementation moves into a versioned search-core crate. Lineage-specific behavior should be represented through an explicit profile or typed configuration rather than source forks. The missed `hexgnn` sign correction is an example of the drift this consolidation would prevent.

### Oversized modules and APIs

[`search.rs`](../packages/hexfield_eq/rust/src/search.rs) is approximately 4,450 lines and [`tree.rs`](../packages/hexfield_eq/rust/src/tree.rs) approximately 3,750 lines. `run_continuous` exposes roughly 40 arguments, and divergence configuration repeats field definitions, key lists, extraction logic, and Python-side forwarding.

Suggested decomposition:

- Validated search configuration
- Root and Gumbel policy
- Continuous-search scheduling
- Evaluator request/reply contracts
- Payload conversion
- Tree arena and promotion
- Diagnostics and telemetry

Python orchestration would also benefit from splitting `model.py` and `multistage_eval.py` into architecture, serving, persistence, statistics, and opponent-adapter modules.

### Static-analysis cleanup

Ruff found 62 F-class issues across `packages`, mostly unused imports plus a missing `Sequence` annotation import in the legacy dense trainer. PyO3 deprecation warnings are also widespread. These are not currently catastrophic, but global lint suppression and mirrored source copies amplify their maintenance cost.

## Verification performed

- 224 Rust tests passed across the workspace with all features.
- 44 targeted `hexfield_eq` CPU/model/checkpoint tests passed.
- 13 CUDA-dependent cases were skipped.
- Python bytecode compilation passed.
- Ruff F-class checks and Rust Clippy/dead-code checks were run for static analysis.
- Full native Python integration and CUDA/Triton behavior could not be exercised on macOS because the authoritative native environment is the WSL build.

No source code was modified as part of this review.
