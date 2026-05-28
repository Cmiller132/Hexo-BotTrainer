# Dense-CNN MCTS Verification And Final Recommendations

## Purpose

This is the finalized dense-cnn MCTS suggestion document. It incorporates the updated
`Review.md` analysis and keeps only claims that were verified against the current
codebase or are explicitly marked as design recommendations.

The current dense-cnn Rust MCTS is a good production baseline. It does not need a full
rewrite. The right next step is to fix a small number of training-target correctness
and search-quality issues while preserving the direct engine-state architecture.

## Verification Scope

This review was static source verification only. No model training, benchmark run, or
full pytest run was performed.

Primary files reviewed:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`
- `packages/hexo_models/dense_cnn/rust/src/encoding.rs`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/mcts.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/rust_bridge.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/selfplay.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/config.py`
- `configs/dense_cnn_model1.toml`
- `Review.md`

## Current Verified Strengths

### Direct engine-state handoff

Dense-cnn search accepts live engine states at the Python boundary, clones them through
the Rust bridge, and searches on Rust-owned state. The live Python game state is only
mutated after MCTS returns a selected action.

This is the right production architecture:

- `hexo_engine` remains the authoritative rules implementation.
- `dense_cnn` owns model-specific encoding, MCTS, sample generation, and search policy.
- Search does not depend on action-history replay.
- Search does not mutate the live Python state.

### Compact Rust search tree

The Rust tree stores nodes, edges, priors, visits, value sums, pending counts, and child
links. It does not store full board copies in every node. Traversal reconstructs the
selected path by applying placements to a scratch state cloned from the root.

This is memory efficient and easier to debug than storing one full state per edge.

### Batched evaluator callback

Rust selects leaves, builds dense-cnn evaluator payloads, calls Python/PyTorch once per
batch, then expands nodes and backs values up in Rust. This is the right split because
search/tree logic belongs in Rust while neural inference belongs in PyTorch.

### Evaluator cache

The evaluator cache deduplicates exact states and is aware of candidate-limit semantics.
That is important because the same state can appear repeatedly during batched self-play,
but the returned prior payload changes when the candidate limit changes.

### Progressive widening

Progressive widening is appropriate for Hexo because the legal action count can be
large. The current implementation starts from high-prior candidates and lazily
materializes more edges as visits grow.

Actual default settings are:

```toml
progressive_widening_initial_actions = 8
progressive_widening_child_initial_actions = 4
progressive_widening_candidate_actions = 128
progressive_widening_growth_interval = 256.0
progressive_widening_growth_base = 1.3
```

## Final Priority List

1. Fix reused-session action/policy mismatch.
2. Add self-play root Dirichlet noise.
3. Add tactical candidate protection.
4. Improve hidden legal action and hidden prior accounting.
5. Add FPU or reduced-FPU for unvisited edges.
6. Add virtual-loss or duplicate-path diagnostics for batched search.
7. Harden evaluator inputs and boundary validation.
8. Clean up documentation/naming around compact input and deterministic ties.

## Verified Issues And Recommendations

### 1. Reused-session action/policy mismatch

Status: verified issue.

Priority: highest.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/selfplay.py`

What happens:

In session mode, the Rust MCTS session keeps a root tree across moves. Before adding
new visits, it records a baseline map of root edge visits:

```rust
let baselines: Vec<HashMap<PackedCoord, u32>> = searches
    .iter()
    .map(|search| search.root_edge_visits().into_iter().collect())
    .collect();
```

After more search, the result payload subtracts that baseline when constructing the
returned `visit_policy`:

```rust
let visits = edge.visits.saturating_sub(before);
```

That means the returned training policy is a delta policy from the current MCTS call,
not the cumulative tree policy.

However, `action_id` is selected from the full cumulative root tree:

```rust
let selected = select_root_action(root, temperature, seed.wrapping_add(index as u64));
```

Python self-play then stores `search.visit_policy` as the training target but applies
`search.action_id` to the live game state.

Result:

```text
played action = sampled from cumulative root visits
training target = delta visits from only this search call
```

Why this matters:

With reused trees and nonzero temperature, the selected action can disagree with the
returned policy target. That creates inconsistent self-play samples: the game follows
one distribution while training records another.

Recommended production invariant:

Use delta visits for both action selection and training policy in self-play session
mode.

Rationale:

- The self-play code already expects `search.visits` to equal the configured visits for
  the current decision.
- The stored sample policy should describe the search that produced this decision.
- Delta policy avoids old root visits dominating the action while not appearing in the
  training target.

Implementation notes:

- Add a Rust helper that selects from baseline-subtracted visits:

  ```rust
  select_root_action_from_deltas(root, baseline, temperature, seed)
  ```

- Use the same helper to build the returned `action_id` in session mode.
- Keep cumulative selection available only if a later explicit feature needs it, but do
  not use it for self-play targets.
- Add diagnostics:

  ```text
  action_selection_policy = "delta_visits"
  cumulative_root_visits
  delta_root_visits
  ```

- If you want the option to compare behavior, return both:

  ```text
  cumulative_visit_policy
  delta_visit_policy
  action_selection_policy
  ```

  But only one should control self-play action selection.

Tests to add:

- Construct a reused root where cumulative visits prefer action A and new delta visits
  prefer action B.
- Assert session MCTS returns action B when configured for delta self-play.
- Assert `visit_policy` and `action_id` come from the same visit basis.
- Assert non-session MCTS remains unchanged.
- Assert tree advance uses the same selected action returned to Python.

### 2. Missing root Dirichlet noise for self-play

Status: verified gap.

Priority: high.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/config.py`
- `configs/dense_cnn_model1.toml`

What happens:

The MCTS boundary accepts visits, `c_puct`, temperature, seed, virtual batch size,
progressive widening settings, candidate limit, cache, and active root limit. It does
not expose root-noise parameters.

Temperature is used only during final root action selection from visit counts. It does
not perturb root priors before search.

Why this matters:

Self-play can collapse too quickly onto the current network prior. That is especially
risky early in training, when the policy head can be confidently wrong. Root Dirichlet
noise is a standard way to force root exploration while keeping the rest of search
deterministic and value-guided.

Recommendation:

Add root-only Dirichlet noise for self-play. This is a good idea and should be the
first search-quality improvement after the action/policy mismatch is fixed.

Suggested config:

```toml
root_dirichlet_noise_enabled = true
root_dirichlet_noise_fraction = 0.25
root_dirichlet_alpha = 0.03
```

Implementation notes:

- Apply noise only at the current root before search visits are added.
- Do not apply noise to child nodes.
- Disable noise for deterministic evaluation/play by default.
- Seed noise from the existing MCTS/self-play seed.
- Record noise settings in diagnostics.
- Use the standard mixture:

  ```text
  noisy_prior = (1.0 - epsilon) * model_prior + epsilon * dirichlet_sample
  ```

- With tree reuse, a promoted child becomes the root on the next self-play decision.
  That new root should receive root noise once for that decision. Avoid repeatedly
  applying noise to the same root within the same call.
- With progressive widening and top-k candidates, start by applying noise to the root
  candidate set that is actually searchable. If hidden-prior accounting is improved
  later, revisit whether part of the noise should be allocated to hidden legal mass.

Alpha value:

- `0.03` is a reasonable starting point.
- Because this model usually searches up to `128` root candidates, experiments should
  also try values around `0.05`, `0.10`, and `0.15`.
- Keep alpha configurable rather than baking in a fixed value.

Tests to add:

- Root priors change when noise is enabled.
- Root priors do not change when noise is disabled.
- Noise is deterministic for the same seed.
- Noise differs for different seeds.
- Noise is applied only to root nodes.
- Noisy root priors remain normalized over represented root candidates.

### 3. Candidate pruning can miss immediate tactical moves

Status: verified risk.

Priority: high.

Files:

- `packages/hexo_models/dense_cnn/rust/src/encoding.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py`
- `configs/dense_cnn_model1.toml`

What happens:

The default candidate limit is `128`. When candidate limiting is active, Rust sends a
compact evaluator payload with:

```text
max_prior_candidates
legal_mask_from_inputs = true
```

Python then chooses top-k priors from the legal plane embedded in the input tensor.

The legal plane only represents legal moves that fit inside the dense-cnn crop. Legal
moves outside the crop are counted in `all_legal_action_count`, but they are not marked
in the legal plane and are not included in the crop-local flat-index lists.

The Rust tree can later discover hidden legal moves by lazy fallback, but those moves
may be reached too late under normal finite visit budgets.

Why this matters:

If an immediate winning move or mandatory block is outside the represented crop or
outside the model's top-k prior list, search may not examine it soon enough. That can
produce bad self-play moves and bad policy targets even though the engine rules would
make the tactic easy to verify.

Recommendation:

Add dense-cnn-owned tactical candidate protection in Rust.

Do not move this into `hexo_engine`. The engine should remain generic. Dense-cnn can use
engine state clones to test tactical consequences.

Implementation notes:

- Before final candidate truncation, force include:
  - immediate winning placements for the side to move,
  - immediate opponent winning placements that the side to move can block,
  - optionally high-priority threat continuations already aligned with dense-cnn hot
    cell features.
- Immediate own wins can be detected by cloning the state, applying each legal
  placement, and checking terminal outcome.
- Block detection should be deterministic and conservative. A practical first version:
  - enumerate opponent immediate-win cells under the current board,
  - force include legal current-player moves that occupy those cells,
  - expand later for two-stone-turn tactical patterns if needed.
- If a forced tactical action already has a model prior, preserve it.
- If it is absent from the top-k model payload, assign a deterministic tactical
  fallback prior and renormalize the represented candidates.
- Tactical forced candidates should be kept separate from ordinary hidden fallback
  moves in diagnostics.

Tests to add:

- A one-placement winning move outside model top-k is still searched at root.
- A required block outside model top-k is still searched at root.
- Forced candidates dedupe against model candidates.
- Forced candidates preserve deterministic ordering.
- Forced candidates appear in diagnostics.

### 4. Non-root hidden legal actions are delayed by candidate exhaustion

Status: verified nuance.

Priority: medium-high.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`

What happens:

Root nodes preserve the full legal action count:

```rust
evaluation.legal_action_count.max(candidates.len())
```

Non-root nodes initially use only:

```rust
candidates.len()
```

There is a refresh path:

```rust
refresh_total_legal_actions
```

That refresh can update a node's total legal count from `state.legal_move_count()`, but
only after:

```text
node.unexpanded_priors is empty
node.edges.len() >= node.total_legal_actions
```

Correct conclusion:

Non-root omitted legal moves are not permanently hidden. They can become visible later.
However, they are delayed until the returned candidate list is exhausted enough for the
refresh path to run. Under finite visit budgets, a critical move outside the returned
candidate list can be practically unreachable.

Recommendation:

Represent full legal space explicitly for every node.

Implementation notes:

- Split the current overloaded count into clearer fields:

  ```rust
  full_legal_action_count
  ranked_candidate_count
  hidden_legal_count
  ```

- Use the full legal action count for root and non-root nodes.
- Let progressive widening decide when hidden moves are exposed.
- Do not encode "unknown legal space" as `candidates.len()`.
- Keep the refresh path only as a safety repair, not the main way non-root nodes learn
  their legal-space size.

Tests to add:

- Non-root nodes created from top-k evaluator payloads retain full legal count.
- Hidden legal count is nonzero when candidates are fewer than legal moves.
- Lazy fallback can materialize non-root hidden moves before all returned candidates
  dominate the search.

### 5. Hidden lazy-move prior mass has no explicit invariant

Status: verified design weakness.

Priority: medium-high.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`

What happens:

`finalize_model_priors` renormalizes selected candidate priors. Hidden legal moves can
later materialize with a fallback prior:

```text
min_positive_represented_prior * 0.01
```

or a uniform prior if no positive represented prior exists.

That means the visible candidate priors start normalized, and hidden moves later enter
with extra fallback prior that was not part of a single visible-plus-hidden prior-mass
model.

Why this matters:

PUCT behavior becomes harder to reason about and tune. The total effective prior mass
can drift as hidden moves materialize.

Recommendation:

Add explicit hidden-prior mass accounting after tactical candidate protection.

Implementation notes:

- Add a config field:

  ```toml
  hidden_prior_mass = 0.05
  ```

- Normalize visible priors to:

  ```text
  1.0 - hidden_prior_mass
  ```

- Allocate hidden mass deterministically as hidden moves materialize:

  ```text
  hidden_prior = remaining_hidden_prior_mass / remaining_hidden_action_count
  ```

- Track diagnostics:

  ```text
  visible_prior_mass
  hidden_prior_mass
  materialized_hidden_prior_mass
  hidden_edges_materialized
  ```

Tests to add:

- Visible plus hidden prior mass remains approximately constant.
- Materializing hidden moves does not inflate total prior mass.
- Root and child nodes follow the same prior-mass rule.
- Tactical forced moves are not treated as ordinary hidden fallback moves.

### 6. Virtual batching increments visits but does not apply value virtual loss

Status: verified behavior and improvement opportunity.

Priority: medium.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`

What happens:

During batched selection, Rust applies virtual visits immediately:

```rust
search.apply_virtual_visit(&selected.path);
```

This increments node and edge visits, plus completed visit accounting. It does not
change `value_sum`. Real value backup happens later after terminal resolution,
existing-node lookup, or evaluator callback.

The selection code skips a pending childless edge:

```rust
edge.pending > 0 && edge.child.is_none()
```

Once a child exists, multiple virtual selections can still flow into the same subtree
before fresh values are backed up.

Why this matters:

Visit-only virtual batching reduces exact duplicate leaf evaluation, but it is weaker
than virtual loss or WU-UCT-style handling. It can still over-concentrate a virtual
batch in the same promising subtree.

Recommendation:

Add diagnostics first. Add virtual loss only if duplicate virtual paths are common.

Implementation notes:

- Track:

  ```text
  duplicate_virtual_paths
  max_pending_per_edge
  repeated_leaf_hashes_per_batch
  subtree_reentry_count
  ```

- If diagnostics show concentration, add one of:
  - virtual loss,
  - virtual mean,
  - WU-UCT-style unobserved visit handling.
- Keep the first implementation configurable:

  ```toml
  virtual_loss_enabled = false
  virtual_loss_value = 1.0
  ```

Tests to add:

- Pending childless edges are skipped.
- Existing child subtrees can be re-entered under virtual batching.
- Diagnostics count duplicate virtual paths.
- Enabling virtual loss changes selection away from repeated paths in a controlled
  fixture.

### 7. No FPU or reduced-FPU for unvisited edges

Status: verified gap.

Priority: medium.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`

What happens:

Unvisited edge value is neutral:

```rust
if self.visits == 0 { 0.0 }
```

Selection then scores the edge as:

```text
Q + c_puct * prior * sqrt(parent_visits) / (1 + edge_visits)
```

For unvisited edges, Q contributes `0.0`.

Why this matters:

Neutral unvisited values are simple, but they make early edge choice heavily dependent
on policy prior. FPU-style logic gives unvisited edges a parent-aware initial value,
usually improving stability.

Recommendation:

Add configurable FPU after root Dirichlet noise.

Suggested config:

```toml
fpu_enabled = true
fpu_reduction = 0.20
```

Implementation notes:

- Use a simple first version:

  ```text
  fpu_value = parent_value - fpu_reduction
  ```

- Make sure value perspective matches the existing player-relative backup convention.
- Keep FPU disabled or fixed in tests that assert existing exact search behavior.

Tests to add:

- With FPU disabled, edge scoring matches current behavior.
- With FPU enabled, unvisited edge Q uses parent-relative FPU.
- FPU does not affect visited edge Q.
- FPU sign/perspective is correct across player turns.

### 8. Already-ranked priors are truncated before dedupe and legality validation

Status: mostly verified issue.

Priority: medium.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`

What happens:

When `already_ranked` is true, `finalize_model_priors` does:

```rust
priors.truncate(limit);
renormalize_priors(priors);
return;
```

Later, `node_from_evaluation` sorts and deduplicates candidates. Because truncation
happens before dedupe, duplicate returned candidates can reduce the effective candidate
count.

The selected-ordinal path is relatively safe because it maps ordinals through
`row.legal_action_ids`. The selected-flat path maps flats back to coordinates and packs
action IDs, but does not verify those action IDs against the engine legal set before
finalizing priors.

Why this matters:

The normal Python top-k path should be producing legal crop flats, so this is not
necessarily a current gameplay bug. It is still brittle at the Rust boundary and can
make evaluator bugs harder to detect.

Recommendation:

Harden ranked-prior intake.

Implementation notes:

- For already-ranked priors:
  - dedupe before truncation,
  - drop illegal action IDs,
  - renormalize after filtering,
  - emit diagnostics for dropped duplicates or illegal candidates.
- For selected-flat results, validate packed action IDs against
  `state.write_legal_action_ids`.
- Keep validation cheap by using a hash set only for the small returned candidate list.

Tests to add:

- Duplicate ranked candidates do not reduce effective candidate count.
- Illegal selected-flat candidates are dropped.
- Ranked candidates are renormalized after filtering.
- Diagnostics report dropped candidates.

### 9. `c_puct` is not validated at the Rust boundary

Status: verified issue.

Priority: low-medium.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`

What happens:

`c_puct` is accepted as `f32` and passed directly into search scoring. There is no
explicit rejection for negative, zero, NaN, or infinite values.

Why this matters:

Invalid `c_puct` values can produce degenerate or undefined-feeling search behavior.
This should fail loudly at the Python/Rust boundary.

Recommendation:

Validate `c_puct` in both single-call and session MCTS entrypoints:

```rust
if !c_puct.is_finite() || c_puct <= 0.0 {
    return Err(PyValueError::new_err("c_puct must be finite and > 0"));
}
```

Tests to add:

- `c_puct = 0.0` raises.
- negative `c_puct` raises.
- NaN `c_puct` raises.
- infinite `c_puct` raises.
- valid positive `c_puct` still works.

### 10. Deterministic tie behavior is biased toward smaller action IDs

Status: verified behavior, not a bug.

Priority: low.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/selfplay.py`

What happens:

Temperature-zero root selection chooses max visits and breaks ties toward smaller
packed action IDs. Edge-score tie comparison also has deterministic action-ID behavior.
The Python policy rollout helper uses the same smaller-ID tie convention.

Why this matters:

Determinism is good for tests and reproducibility. The only downside is a coordinate
bias when visits or policy weights tie exactly. Root Dirichlet noise should reduce this
during self-play.

Recommendation:

Keep the behavior, document it, and test it.

Tests to add:

- Equal-visit deterministic selection chooses the smaller action ID.
- Python rollout and Rust MCTS agree on deterministic tie convention.
- With root noise and nonzero temperature, seeded self-play can break ties without
  losing reproducibility.

### 11. Diagnostics are attached only to the first result

Status: verified behavior and improvement opportunity.

Priority: low.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`

What happens:

`build_search_result_payloads` attaches diagnostics only when:

```rust
index == 0
```

Batch diagnostics aggregate across all searches, but per-root pathologies can be hard
to inspect when the problematic root is not the first item.

Recommendation:

Add optional debug diagnostics modes:

```text
diagnostics_mode = "first" | "all" | "worst" | "sampled"
```

Default can remain `"first"` to avoid bloating normal payloads.

Tests to add:

- Default diagnostics behavior remains first-result only.
- `all` attaches diagnostics to every result.
- `worst` attaches diagnostics to the root with the highest hidden/candidate pressure.

### 12. Candidate-limit cache invalidation is correct but coarse

Status: verified improvement opportunity.

Priority: low.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`

What happens:

The evaluation cache tracks candidate limit and clears when the limit changes.

This is correct. A cache keyed only by state hash would be wrong because the returned
prior payload can change with candidate limit.

Recommendation:

Leave this as-is unless performance profiling shows cache churn from changing candidate
limits.

If needed later, key cache entries by:

```text
state_hash
candidate_limit
payload_format_version
model_generation
```

Tests to add:

- Same state and same candidate limit reuse cache.
- Same state and different candidate limit do not reuse incompatible payloads.
- If multi-key caching is added, both candidate limits can coexist safely.

### 13. Evaluation cache and MCTS session must stay scoped to one model snapshot

Status: verified invariant and guardrail.

Priority: low-medium.

Files:

- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/rust_bridge.py`
- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`

What happens:

The Python wrapper documents the native MCTS evaluation cache as scoped to one
model-weight snapshot. The Rust cache key does not include model identity.

This is acceptable if sessions and caches are recreated whenever model weights change.
It is unsafe if a cache/session can survive across checkpoint swaps.

Recommendation:

Keep the current behavior if the lifecycle is guaranteed. Add a guard if sessions can
ever live across model changes.

Implementation options:

- Add a Python-side `weights_id`.
- Add a Rust-side `model_generation` field on session/cache.
- Clear session/cache explicitly after checkpoint load or weight update.

Tests to add:

- Cache/session is recreated or cleared after a model snapshot change.
- Reusing a cache with a mismatched `weights_id` raises.

### 14. "Half input" naming is misleading

Status: verified documentation/naming issue.

Priority: low.

Files:

- `packages/hexo_models/dense_cnn/rust/src/encoding.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`

What happens:

The compact path writes both own-stone and opponent-stone planes. It is not masking
opponent stones and is not removing board information. "Half" refers to compact
`u16`/float16-style transfer representation.

Recommendation:

Rename or document the path.

Preferred names:

```text
compact_input
u16_input
packed_input
```

If renaming is too noisy, add comments near the encoder and payload builder:

```text
Compact MCTS payload. Stores the same model planes in u16 form for transfer efficiency.
This does not remove opponent-stone planes.
```

Tests to add:

- Compact input contains own stones.
- Compact input contains opponent stones.
- Compact input contains legal plane data for crop-local legal moves.

### 15. Fixed visit targets have no early confidence stopping

Status: verified behavior, not a bug.

Priority: low.

Files:

- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`

What happens:

Search runs until each root reaches its target visit count. There is no early stop when
the leading move can no longer be overtaken.

Why this is acceptable:

Fixed visits are simple and produce stable self-play targets. That is usually what you
want for training.

Recommendation:

Do not prioritize this before root noise, tactical candidate protection, FPU, and
virtual batching diagnostics.

If added later, make it evaluation-only by default:

```text
best_visits > second_best_visits + remaining_visits
```

Tests to add:

- Early stopping does not change selected action compared with completing all visits in
  deterministic fixtures.
- Self-play still emits exact configured visit counts when early stopping is disabled.

## Claims Corrected From Earlier Reviews

### Non-root legal moves are not permanently hidden

Correct wording:

```text
Non-root omitted legal moves can be delayed until returned candidate priors are
exhausted enough for legal-count refresh. Under normal visit budgets, this can make
them practically unreachable, but not permanently impossible.
```

### Compact input does not mask opponent stones

Correct wording:

```text
The compact MCTS input path includes both own and opponent stone planes. It is a compact
transfer format, not a lossy board representation.
```

### Cache should not be keyed by state hash alone

Correct wording:

```text
The cache must include request-shape semantics such as candidate limit. A state-only
key can reuse the wrong prior payload.
```

## Recommended Implementation Plan

### Phase 1: Fix reused-session training-target consistency

Goal:

Make the action that self-play applies and the policy target that self-play stores come
from the same visit basis.

Work:

- Add delta-visit root action selection.
- Use delta action selection in session self-play mode.
- Add diagnostics for action-selection basis.
- Add mismatch regression tests.

Acceptance:

- A fixture where cumulative and delta visits disagree returns the delta-selected action
  and delta policy together.
- Non-session MCTS behavior remains unchanged.

### Phase 2: Add root Dirichlet noise

Goal:

Improve self-play exploration without changing deterministic evaluation.

Work:

- Add config fields.
- Add Rust root-prior noise application.
- Thread settings through Python self-play and Rust bridge.
- Seed noise from existing self-play seed.
- Add diagnostics.

Acceptance:

- Noise is root-only, deterministic by seed, disabled for deterministic evaluation, and
  normalized over represented root candidates.

### Phase 3: Add tactical candidate protection

Goal:

Prevent obvious wins and blocks from being missed due to crop/top-k candidate pruning.

Work:

- Add dense-cnn Rust tactical candidate detector.
- Force include immediate wins and conservative immediate blocks.
- Assign deterministic tactical fallback priors for absent top-k candidates.
- Add diagnostics.

Acceptance:

- Immediate wins/blocks outside top-k are included and searchable.

### Phase 4: Represent hidden legal space explicitly

Goal:

Make root and non-root progressive widening consistent and easier to reason about.

Work:

- Split full legal count, ranked candidate count, and hidden legal count.
- Use full legal count for all nodes.
- Keep refresh as a safety repair path.

Acceptance:

- Non-root nodes know their full hidden legal space at creation time.

### Phase 5: Add explicit hidden-prior mass

Goal:

Avoid implicit prior inflation when hidden moves are lazily materialized.

Work:

- Add hidden prior mass config.
- Normalize visible priors to `1.0 - hidden_prior_mass`.
- Allocate hidden prior mass deterministically.
- Keep tactical forced candidates separate.

Acceptance:

- Visible plus hidden prior mass remains stable as edges materialize.

### Phase 6: Improve edge scoring and batched selection

Goal:

Reduce overdependence on raw policy priors and improve virtual batch diversity.

Work:

- Add configurable FPU.
- Add virtual-path diagnostics.
- Add virtual loss only if diagnostics show duplicate subtree concentration.

Acceptance:

- FPU behavior is testable and disabled/enabled explicitly.
- Virtual batching diagnostics make duplicate path behavior visible.

### Phase 7: Harden boundaries and diagnostics

Goal:

Make invalid inputs fail loudly and make debugging easier.

Work:

- Validate `c_puct`.
- Dedupe and legality-check already-ranked priors.
- Add optional per-root diagnostics.
- Add model snapshot guard for cache/session if needed.
- Document deterministic tie behavior.
- Rename or document compact input.

Acceptance:

- Invalid `c_puct` raises.
- Illegal ranked priors are dropped or rejected.
- Debug diagnostics can identify pathological roots.

## Things Not To Do

### Do not move these changes into `hexo_engine`

These are model-search behaviors. They belong under `hexo_models/dense_cnn`.

### Do not restore history-row MCTS

The direct engine-state handoff is cleaner and should remain the single production
path.

### Do not add Python MCTS fallback logic

The production MCTS path should remain Rust-owned. Python should orchestrate and run
PyTorch inference, not duplicate search algorithms.

### Do not key the evaluator cache by state hash alone

Candidate limit and payload format matter. A state-only cache key is not sufficient.

### Do not add unseeded randomness

Dirichlet noise should be stochastic but reproducible from the self-play/MCTS seed.

## Final Recommendation

Treat the dense-cnn Rust MCTS architecture as sound. The most important correctness fix
is the reused-session action/policy mismatch. After that, implement root Dirichlet
noise and tactical candidate protection. Those three changes should materially improve
self-play data quality without changing the overall architecture.

The short version:

```text
1. Make played action and stored policy target use the same visit basis.
2. Add seeded root Dirichlet noise for self-play.
3. Force immediate tactical wins/blocks into the candidate set.
4. Then improve hidden-prior accounting, FPU, and virtual batching.
```
