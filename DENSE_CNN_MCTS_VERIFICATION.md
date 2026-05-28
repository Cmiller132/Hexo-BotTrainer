# Dense-CNN MCTS Final Verification And Suggestions

## Scope

This document is the finalized dense-cnn MCTS suggestion document after reviewing
the updated `Review.md` against the current working tree.

The current code has moved the dense-cnn search path to one production route:

```text
Python self-play / player / benchmark code
  -> hexo_models.dense_cnn.mcts.BatchedMctsSession
  -> hexo_models.dense_cnn.rust_bridge.model1_mcts_session_search
  -> hexo_models._rust.dense_cnn.Model1MctsSession.search
```

The removed direct Rust call path, standalone Python-visible MCTS evaluation cache,
history-row MCTS path, and Python search fallback should stay removed.

Verification here is static source verification plus focused Python tests. Rust
compile/tests could not be completed on this machine because the Rust toolchain is
not available here.

## Review.md Findings Rechecked

The updated `Review.md` identified a good set of dense-cnn MCTS issues. Several of
those issues are now fixed in the working tree:

- Reused-session action selection now uses the same delta visit policy that is
  returned for training targets.
- Root Dirichlet noise is threaded through config, Python self-play, the Rust bridge,
  and Rust root priors.
- Tactical candidate protection force-includes immediate wins and immediate blocks
  inside dense-cnn Rust.
- Root and child nodes now use the full legal action count from evaluation instead
  of treating child legal space as only the returned candidate list.
- Hidden legal moves now have explicit hidden-prior mass accounting.
- FPU/reduced-FPU is implemented for unvisited edges.
- Virtual loss is implemented for virtual batched leaf selection.
- Already-ranked selected priors are deduplicated before truncation, and selected
  flat-index priors are checked against legal action IDs.
- `c_puct` and temperature are validated at the Rust MCTS boundary.
- Diagnostics are attached to every result payload, not only the first result.
- The old `mcts_evaluation_cache_max_states` config name was replaced with
  `mcts_session_cache_max_states` to match the single session-owned cache.

## Current Verified Strengths

### Direct Engine-State Search

Dense-cnn accepts live `hexo_engine.HexoState` objects at the Python boundary and
clones them into Rust-owned model search state. MCTS does not replay history rows
and does not mutate the live game. Only the final selected action is applied by
self-play after search returns.

This is the preferred architecture because `hexo_engine` stays generic and
authoritative, while dense-cnn owns model-specific encoding, search, caching,
candidate handling, and sample generation.

### Single MCTS API Surface

The preferred public Python surface is now session based. `run_mcts` and
`run_batched_mcts` remain convenience functions, but they create/use the same
native session path. They are not alternate search implementations.

The Rust bridge exposes:

```text
model1_new_mcts_session(...)
model1_mcts_session_search(...)
```

It no longer exposes:

```text
model1_batched_mcts(...)
model1_new_mcts_evaluation_cache(...)
```

### Search-Quality Improvements

The dense-cnn Rust tree now has the main search-quality features that were missing
from the reviewed version:

- progressive widening;
- session tree reuse;
- exact evaluator cache inside the native session;
- root Dirichlet noise;
- tactical candidate protection;
- explicit hidden-prior mass;
- FPU/reduced-FPU;
- virtual loss for virtual batching;
- legal validation and dedupe for ranked priors.

## Remaining Suggestions

### 1. Run Rust Compile And Rust Unit Tests

Priority: highest verification item.

The working tree should be checked on a machine with `cargo` and `rustfmt`:

```powershell
cargo fmt --check
cargo check -p hexo_models --features python
cargo test -p hexo_models --features python dense_cnn
```

Why this matters:

The most important remaining risk is not conceptual; it is Rust compile and
behavioral verification for the new MCTS tree code.

### 2. Add More Rust-Level Regression Tests

Priority: high.

Python boundary tests are useful, but the new behavior mostly lives in Rust. Add
focused Rust tests for:

- delta action selection when cumulative and delta visits disagree;
- root Dirichlet noise determinism by seed;
- root noise applied only to the current root;
- tactical win/block candidate insertion and dedupe;
- hidden-prior mass staying stable as hidden actions materialize;
- FPU affecting only unvisited edge values;
- virtual loss reducing duplicate virtual path selection;
- selected-flat illegal candidates being dropped.

### 3. Track Forced-Candidate Diagnostics

Priority: medium-high.

Tactical candidate protection is now present, but diagnostics should make it easy
to see when it mattered.

Recommended root diagnostics:

```text
tactical_win_candidates
tactical_block_candidates
tactical_candidates_added
tactical_candidates_deduped
hidden_edges_materialized
visible_prior_mass
hidden_prior_mass
```

This will make self-play review much easier when a move was selected because Rust
overrode a weak model top-k prior list.

### 4. Add A Model Snapshot Guard For Long-Lived Sessions

Priority: medium.

The native session owns an evaluator cache keyed by state identity and candidate
limit semantics, not by neural network weights. This is correct if sessions are
created per model snapshot and discarded when weights change.

If any future training loop keeps a session across checkpoint swaps or model weight
updates, add a `weights_id` or `model_generation` guard and clear/replace the native
session when it changes.

### 5. Tune Root Noise And Hidden-Prior Values

Priority: medium.

The default root Dirichlet settings are sensible starting values:

```toml
root_dirichlet_noise_enabled = true
root_dirichlet_noise_fraction = 0.25
root_dirichlet_alpha = 0.03
hidden_prior_mass = 0.05
fpu_reduction = 0.20
virtual_loss = 1.0
```

These should be treated as tunables. In particular, `alpha = 0.03` is common for
large action spaces, but dense-cnn usually searches a candidate frontier rather
than the entire legal action set. It is worth comparing `0.03`, `0.05`, `0.10`,
and `0.15` once self-play throughput is stable.

### 6. Keep Compact Input Naming Honest

Priority: low.

The compact MCTS input path stores both own and opponent stone planes. It is a
compact transfer format, not a lossy board representation. Avoid describing it as
"half input" in user-facing docs unless the nearby comment clearly says it means
compact `u16`/byte transfer.

Better names:

```text
compact_input
packed_input
u16_input
```

### 7. Keep Cumulative Policy Diagnostic-Only

Priority: low.

The production self-play invariant should remain:

```text
played action and stored policy target come from the same visit basis
```

For the session path, that basis is delta visits. If cumulative policy is useful
later, expose it only as diagnostics or an explicit analysis payload. Do not add
another production action-selection route.

## Things To Avoid

- Do not restore history-row MCTS.
- Do not restore `model1_batched_mcts`.
- Do not restore `Model1MctsEvaluationCache` as a Python-visible object.
- Do not add Python MCTS fallback logic.
- Do not move dense-cnn tactical/search logic into `hexo_engine`.
- Do not key the evaluator cache by state hash alone if request shape can differ.
- Do not add unseeded self-play randomness.

## Final Recommendation

The dense-cnn MCTS architecture is now in the right shape: direct engine-state
handoff, model-owned Rust search, session-owned tree reuse/cache, and one preferred
production API. The remaining work should focus on Rust compile verification,
targeted Rust tests, and tuning/diagnostics for the new search-quality features.
