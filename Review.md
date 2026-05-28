# Final verification summary

I reviewed the newest uploaded pass and compared it against the current dense-CNN MCTS source-level evidence. The uploaded document is broadly accurate and is a better consolidation than the earlier review, especially because it avoids overstating the non-root legal-action issue and correctly treats the “half-input” path as a compact transfer format, not a lossy board representation. 

My final view is:

1. The dense-CNN MCTS is a solid baseline architecture.
2. It does **not** need a rewrite.
3. The highest-value fixes are targeted search-quality and self-play-data improvements.
4. One important issue from the prior review should still be kept on the list: **session reuse selects actions from cumulative visits while returning delta visits as the policy target**.

Below is the most accurate verified issue list I would act on.

---

# Highest-priority verified issues

## 1. Session reuse action/policy mismatch

**Status: verified.**
**Priority: high.**

In session mode, the code records baseline root visits before adding more search, then returns a visit policy using only the **delta visits** after that baseline. The returned policy weights subtract `before` visits from each edge and normalize by the delta `policy_total`.  

However, selected actions are computed from the full cumulative root tree:

```rust
select_root_action(search.root(), temperature, seed...)
```

before the tree is advanced. 

Self-play then stores `search.visit_policy` as the training target but applies `search.action_id` as the actual played move.  

This means:

```text
played action = sampled from cumulative visits
training target = delta visit distribution
```

That may be intentional, but it is not documented and can make the policy target disagree with the actual move, especially with nonzero temperature and reused trees.

**Recommended fix:** choose one invariant and make it explicit:

```text
Option A: choose from cumulative visits and train on cumulative visits.
Option B: choose from delta visits and train on delta visits.
Option C: return both cumulative_policy and delta_policy, and record which controlled action selection.
```

For self-play consistency, I would choose **Option B** unless there is a deliberate reason to preserve cumulative action selection.

---

## 2. No root Dirichlet noise for self-play

**Status: verified gap.**
**Priority: high.**

The dense-CNN MCTS boundary exposes visits, `c_puct`, temperature, seed, virtual batch size, progressive widening settings, candidate limit, cache, and active-root limit, but there is no root-noise parameter. 

The default self-play config similarly has search visits, active games, progressive widening, cache size, max actions, temperature, and worker count, but no root Dirichlet/noise fields. 

Temperature is only used during final action selection from visit counts, not to perturb root priors before search. The root action selector samples from `visits^(1/temperature)` or chooses max visits when temperature is near zero. 

**Impact:** self-play can collapse too quickly onto the current network prior, especially early in training.

**Recommended fix:** add root-only Dirichlet noise for self-play:

```toml
root_dirichlet_noise_enabled = true
root_dirichlet_noise_fraction = 0.25
root_dirichlet_alpha = 0.03
```

Keep it disabled for deterministic evaluation. Seed it from the existing MCTS seed and record noise settings in diagnostics.

---

## 3. Candidate pruning can miss immediate tactical moves

**Status: verified risk.**
**Priority: high.**

The default candidate limit is 128 in the dense-CNN self-play config.  When progressive-widening candidate limiting is active, Rust sends `max_prior_candidates` and asks Python to select priors using the legal mask from the input tensor. 

The dense crop only marks legal actions that map inside the crop. In the half-input MCTS path, legal moves outside the crop are not represented in the legal plane.  In the full path, legal action IDs and flat indices are also only pushed if the move maps into the crop. 

So if a winning move or mandatory block is outside the crop or outside the model’s top-k candidate list, the search may only reach it through lazy fallback, possibly too late.

**Recommended fix:** add dense-CNN-owned tactical candidate protection in Rust. Do not move this into `hexo_engine`.

Force include:

```text
immediate wins for side to move
immediate opponent wins that must be blocked
optionally high-priority threat continuations
```

If a forced move is missing from the network top-k, assign a deterministic tactical fallback prior and renormalize.

---

## 4. Hidden legal moves are not permanently hidden, but they are discovered too late

**Status: corrected and verified nuance.**
**Priority: medium-high.**

The earlier claim that progressive widening/top-k can “permanently hide” non-root legal moves is too strong. Non-root nodes initially set:

```rust
total_legal_actions = candidates.len()
```

while root nodes preserve:

```rust
evaluation.legal_action_count.max(candidates.len())
```



But the code has a refresh path. Once a node has no unexpanded priors and its edge count has caught up to `total_legal_actions`, `refresh_total_legal_actions` can update the count from `state.legal_move_count()`. 

So the accurate statement is:

```text
Non-root omitted legal moves are not strictly permanent-hidden.
They are delayed until the returned candidate list is exhausted enough for refresh_total_legal_actions to run.
```

That is still a real problem under finite visit budgets. A critical move outside the returned priors can be practically unreachable.

**Recommended fix:** separate these concepts:

```rust
full_legal_action_count
ranked_candidate_count
hidden_legal_count
```

Use the full legal count for all nodes, not only roots. Let progressive widening decide when to expose hidden legal moves, but do not encode “unknown legal space” as `candidates.len()`.

---

## 5. Hidden lazy-move prior mass is not explicitly accounted for

**Status: verified design weakness.**
**Priority: medium-high.**

`finalize_model_priors` renormalizes the selected candidate priors.  Hidden legal moves can later materialize with `fallback_prior`, which is the minimum positive represented prior times `0.01`, or uniform if no positive prior exists. 

That means the represented candidate priors start as a normalized distribution, and hidden moves later enter with additional fallback priors that were not part of a single explicit prior-mass invariant.

**Recommended fix:** add explicit hidden-prior mass accounting:

```toml
hidden_prior_mass = 0.05
```

Then normalize visible priors to:

```text
1.0 - hidden_prior_mass
```

and allocate hidden prior mass deterministically as hidden moves materialize.

This should come after tactical candidate protection, because forced tactical moves should not be treated as ordinary hidden moves.

---

# Medium-priority verified issues

## 6. Virtual batching increments visits but does not apply value virtual loss

**Status: verified.**
**Priority: medium.**

During virtual batching, the search increments visits along the selected path:

```rust
self.completed_visits += 1
node.visits += 1
edge.visits += 1
```

but it does not change `value_sum`. 

The actual value backup happens later through `backup_virtual`. 

The pending check only skips an edge when:

```rust
edge.pending > 0 && edge.child.is_none()
```



Once a child exists, multiple virtual selections can still flow into the same subtree before fresh values are backed up.

**Recommended fix:** implement either:

```text
virtual loss
virtual mean / WU-UCT-style unobserved visits
duplicate-path diagnostics
```

I would start with diagnostics first, then add virtual loss if duplicate virtual paths are common.

---

## 7. No FPU / reduced-FPU for unvisited edges

**Status: verified gap.**
**Priority: medium.**

Unvisited edge value is neutral:

```rust
if self.visits == 0 { 0.0 }
```



So unvisited moves are scored as neutral value plus prior exploration. This is simple, but it makes search heavily dependent on policy priors and does not use parent value to set a more informed first-play estimate.

**Recommended fix:** add configurable FPU after root noise:

```toml
fpu_enabled = true
fpu_reduction = 0.20
```

A simple first version:

```text
fpu_value = parent_value - fpu_reduction
```

Make sure the value perspective matches the existing player-relative backup convention.

---

## 8. Already-ranked priors are truncated before deduplication and legal validation

**Status: mostly verified.**
**Priority: medium.**

When `already_ranked` is true, `finalize_model_priors` simply truncates and renormalizes:

```rust
priors.truncate(limit);
renormalize_priors(priors);
return;
```



In the selected-flat branch, Rust maps returned flat indices back to coordinates and packs them into action IDs, but does not verify those action IDs against the engine’s legal-action set. 

The selected-ordinal path is safer because it indexes into `row.legal_action_ids`. 

**Recommended fix:** for already-ranked paths, at least in debug mode:

```text
deduplicate returned candidates
verify against state.write_legal_action_ids()
drop illegal candidates before node creation
```

The top-k candidate lists are small, so the cost should be acceptable.

---

## 9. `c_puct` is unvalidated

**Status: verified.**
**Priority: low-medium.**

Rust accepts `c_puct: f32` directly.  It is passed into search and used directly in edge scoring through the exploration scale.  

Negative, NaN, or infinite values can create degenerate selection behavior.

**Recommended fix:** reject invalid values at the Rust boundary:

```rust
if !c_puct.is_finite() || c_puct <= 0.0 {
    return Err(PyValueError::new_err("c_puct must be finite and > 0"));
}
```

---

# Lower-priority verified improvements

## 10. Deterministic action-ID tie bias

**Status: verified behavior.**
**Priority: low.**

Temperature-zero selection chooses the max visit count, with ties biased toward smaller action IDs via `Reverse(edge.action_id)`. 

Edge-score comparison also has deterministic tie behavior. 

This is not a bug; deterministic tie-breaking is useful for reproducibility. But it should be documented, and root noise should reduce tie bias during self-play.

**Recommended fix:** document and test it.

---

## 11. Diagnostics are only attached to the first result

**Status: verified.**
**Priority: low.**

`build_search_result_payloads` only attaches diagnostics when `index == 0`. 

Batch diagnostics aggregate across all searches, but a pathological root can be hard to debug if it is not the first root. 

**Recommended fix:** add a debug flag for:

```text
per-root diagnostics
worst-root diagnostics
sampled root diagnostics
```

---

## 12. Candidate-limit cache invalidation is correct but coarse

**Status: verified improvement opportunity.**
**Priority: low.**

The evaluation cache tracks candidate limit and clears when it changes. 

That is correct and should not be weakened. A state-only cache key would be wrong because the returned prior payload changes with candidate limit.

**Recommended fix:** leave as-is for now. If performance becomes an issue, key cache entries by:

```text
state_hash
candidate_limit
payload_format_version
future evaluator/model generation
```

---

## 13. Evaluation cache/session must stay scoped to one model snapshot

**Status: verified invariant.**
**Priority: low-medium guardrail.**

The Python wrapper explicitly documents the evaluation cache as scoped to a single model-weight snapshot.  The Rust cache key uses state hash and candidate-limit semantics, but not model identity. 

Current self-play appears to create a fresh inference object and session per generated epoch, so this is not obviously broken in the normal path.  

**Recommended fix:** add a `model_generation` or `weights_id` guard if sessions can ever live across checkpoint swaps.

---

## 14. “Half-input” naming is misleading, but not a gameplay bug

**Status: verified documentation issue.**
**Priority: low.**

The compact/half path still writes both own and opponent stone planes. It is not removing opponent stones; it is storing planes as `u16`/compact payload data for transfer efficiency. 

**Recommended fix:** rename internally to:

```text
compact_input
u16_input
packed_input
```

or add a clear comment saying it is compact transfer, not reduced board information.

---

## 15. Fixed visit targets without early confidence stopping

**Status: verified behavior.**
**Priority: low.**

Search runs until `completed_visits < target_visits` becomes false. 

That is simple and appropriate for stable self-play targets. Early stopping could help evaluation latency, but I would not prioritize it over root noise, FPU, and tactical candidate protection.

**Recommended fix:** optional evaluation-only early stopping:

```text
best_visits > second_best_visits + remaining_visits
```

Do not enable by default in self-play unless you intentionally change the training target semantics.

---

# Claims I would correct or soften

## “Non-root legal moves are permanently hidden”

I would **not** keep this exact wording. The code can refresh non-root legal counts later. The accurate wording is:

```text
Non-root omitted legal moves can be delayed until returned candidate priors are exhausted enough for legal-count refresh, which can make them practically unreachable under normal visit budgets.
```

## “Half-input masks opponent stones”

This is false for the current code. The compact path writes both own and opponent stone planes. 

## “Key cache only by state hash”

This is incomplete. The cache is not model-version-aware, but it does track candidate-limit semantics and clears when the candidate limit changes. 

---

# Final prioritized implementation list

## Phase 1 — Fix self-play data consistency

1. Resolve cumulative-action vs delta-policy mismatch in reused-session search.
2. Add diagnostics showing whether action selection used cumulative or delta visits.
3. Add tests where cumulative and delta policies intentionally disagree.

This is the most concrete training-data correctness issue.

## Phase 2 — Add root exploration noise

1. Add self-play-only root Dirichlet noise.
2. Seed it deterministically.
3. Store noise settings and maybe root prior summaries in diagnostics.

This is the highest-value exploration improvement.

## Phase 3 — Add tactical candidate protection

1. Detect immediate wins and required blocks inside dense-CNN Rust.
2. Force them into candidate priors before truncation.
3. Assign deterministic tactical fallback priors when absent from model top-k.

This reduces obvious tactical failures under weak priors.

## Phase 4 — Improve edge scoring

1. Add FPU / reduced-FPU for unvisited edges.
2. Add virtual loss or WU-UCT-style virtual mean for batched selection.
3. Add duplicate virtual-path diagnostics.

This improves search stability and batched search quality.

## Phase 5 — Clean up hidden-prior accounting

1. Track visible prior mass and hidden prior mass explicitly.
2. Avoid implicit prior inflation when lazy hidden moves materialize.
3. Make root and child behavior consistent.

## Phase 6 — Hardening and documentation

1. Validate `c_puct`.
2. Deduplicate and legality-check already-ranked priors.
3. Add optional per-root diagnostics.
4. Document deterministic tie behavior.
5. Rename or document the compact “half-input” path.

# Bottom line

The current dense-CNN MCTS is a good production baseline, but I would **not** treat self-play data as fully trustworthy until two things are fixed:

1. **the reused-tree action/policy mismatch**, and
2. **candidate protection for immediate tactical wins/blocks**.

After that, root Dirichlet noise, FPU, and virtual-loss improvements should make the search much more robust without changing the overall architecture.
