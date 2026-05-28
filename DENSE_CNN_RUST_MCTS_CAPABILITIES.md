# Dense-CNN Rust MCTS Capabilities

This document explains the dense-cnn Rust MCTS capabilities currently exposed by
`hexo_models._rust.dense_cnn`. It focuses on what each capability is, how it is
implemented, what the intended performance or search benefit is, and how it
changes model behavior.

Relevant files:

- `packages/hexo_models/dense_cnn/rust/src/lib.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs`
- `packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs`
- `packages/hexo_models/dense_cnn/rust/src/encoding.rs`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/mcts.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py`
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/rust_bridge.py`

The advertised Rust capability payload includes:

```text
model1_mcts_progressive_widening = true
model1_mcts_tree_reuse_session = true
model1_mcts_session_search = true
model1_mcts_lazy_staged_edges = true
model1_mcts_root_dirichlet_noise = true
model1_mcts_hidden_prior_mass = true
model1_mcts_tactical_candidate_protection = true
model1_mcts_first_play_urgency = true
model1_mcts_virtual_loss = true
```

The dense-cnn MCTS path is intentionally model-owned. Python passes live engine
states to dense-cnn Rust. Dense-cnn Rust clones those states, owns search-local
state mutation, asks Python/PyTorch only for neural evaluation, and returns a
compact search result payload.

## End-To-End Search Shape

At a high level, a search does this:

1. Python creates a `Model1MctsSession` and calls `Model1MctsSession.search(...)`
   through `dense_cnn.rust_bridge`.
2. Rust clones each live `hexo_engine.HexoState` into a model-owned
   `RustHexoState`.
3. Rust evaluates each root state through the Python callback
   `DenseCNNInference.evaluate_model1_payload`.
4. Rust creates one `RustSearch` per root.
5. Rust repeatedly selects leaves from each root tree, applies virtual visits,
   batches unique uncached leaf states, evaluates those states, expands nodes,
   and backs values up the selected paths.
6. Rust returns one payload per root:
   - `action_id`
   - compact visit-policy action-id bytes
   - compact visit-policy weight bytes
   - `root_value`
   - exact new visit count
   - diagnostics for each root

The important ownership rule is that the live Python game is not mutated by
MCTS. Search mutates cloned states and tree-local state only.

## Progressive Widening

### What It Is

Progressive widening, also called progressive unpruning, limits how many child
moves are active at a node early in search. Instead of allocating and exploring
every legal move immediately, MCTS starts with a small high-prior frontier and
opens more moves as the node gets more visits.

In this implementation, the neural policy prior is the heuristic used to rank
which moves become visible first.

### Intent

Dense-cnn boards can have a large legal move count. Full-width MCTS spends memory
and selection time on many low-prior moves before the search knows whether the
position is worth exploring. Progressive widening is intended to:

- reduce root and child branching cost;
- focus early visits on the moves the network thinks are most plausible;
- keep tree memory lower;
- make a fixed visit budget reach deeper lines sooner;
- still allow more moves to appear as evidence accumulates.

### Implementation Details

The configuration lives in `ProgressiveWideningConfig` in `mcts_tree.rs`:

```text
root_initial_actions
child_initial_actions
growth_interval
growth_base
```

The edge limit is:

```text
if visits < growth_interval:
    edge_limit = initial_actions
else:
    edge_limit = initial_actions + floor(log(visits / growth_interval, growth_base))
```

Then the limit is clamped by the node's known legal action count.

The current config path forwards:

```text
progressive_widening_initial_actions
progressive_widening_child_initial_actions
progressive_widening_growth_interval
progressive_widening_growth_base
```

from Python to Rust through `rust_bridge.py`.

When a node is created from a neural evaluation, its policy candidates are sorted
by prior. They are stored in `unexpanded_priors` so the best candidate can be
popped when the widening rules allow another edge to become active.

### How Selection Uses It

`select_or_materialize_edge(...)` compares two kinds of options:

- existing active edges, scored with PUCT;
- the next hidden prior candidate, scored as a new child with no visits.

If the hidden candidate has the best score and the widening limit allows another
child, Rust materializes that candidate into a real `RustEdge`.

This means a legal move can exist in the neural evaluation payload but not yet
exist as a tree edge. It becomes an edge only when search pressure justifies it.

### How It Changes Model Behavior

Without progressive widening, the MCTS policy can spread visits across many legal
moves early. With progressive widening, early visits are concentrated into a
smaller set of high-prior moves. The returned policy target will usually be
sharper early in the search, and low-prior moves need enough parent visits before
they can compete.

This makes the neural prior more influential. That is the point, but it also
means bad priors can hide tactical moves longer unless candidate generation or
widening parameters are generous enough.

### Current Legal-Space Behavior

Root and non-root nodes both preserve the evaluator's full legal action count.
The ranked model candidates form the staged frontier, while remaining legal moves
stay hidden and can be materialized lazily as widening allows. Tactical wins and
blocks are force-included before candidate normalization so obvious one-ply moves
are not dependent on the model top-k list.

## Evaluator Cache

### What It Is

The evaluator cache stores neural evaluations keyed by an exact state hash. If
the same model-relevant state appears again, Rust can reuse the policy/value
evaluation instead of calling Python/PyTorch again.

The cache stores `RustEvaluation`:

```text
value
legal_action_count
priors: Vec<(PackedCoord, prior)>
```

### Intent

Neural evaluation is the expensive part of MCTS. The cache is intended to:

- avoid repeated PyTorch calls for duplicate states;
- deduplicate identical leaf states within the same virtual batch;
- share evaluations across roots in one batched self-play step;
- share evaluations across moves when using a tree reuse session;
- make repeated openings and transpositions cheaper.

### Implementation Details

The cache implementation is in `mcts_eval.rs`:

```text
RustEvaluationCache
  entries: HashMap<StateHash, RustEvaluation>
  insertion_order: VecDeque<StateHash>
  candidate_limit_initialized: bool
  candidate_limit: Option<usize>
```

The cache is owned by the native `Model1MctsSession`. There is no separate public
Python-visible cache object.

The cached key is `hexo_utils::hash_state(state)`. That hash is intended to be
history-sensitive, not just board-equivalent. This matters because dense-cnn
inputs include recency planes and other history-dependent features. Two positions
with the same stones but different placement order may be different model inputs,
so they should not be merged by a board-only hash.

The main function is:

```text
evaluate_model1_state_refs_cached(...)
```

It performs three passes:

1. For each requested state, check the cache.
2. For uncached states, deduplicate identical hashes within the current request
   batch.
3. Evaluate only unique uncached states, insert them into the bounded cache, and
   fill all result slots.

The cache is bounded by `max_states`. Eviction is insertion-order based: when the
cache is full, old keys are popped from the front of `insertion_order`.

The cache also tracks the `prior_candidate_limit`. If that limit changes, the
cache is cleared. This is necessary because a cached evaluation with top 32
priors is not equivalent to a cached evaluation with top 128 priors.

### How It Changes Model Behavior

The cache should not change the mathematical result for the same model weights
and same candidate limit. It should only reduce duplicate work.

However, it does make the cache lifetime important. Cached evaluations are only
valid for one model-weight snapshot. If model weights change, old policy/value
outputs are stale. The code is structured so self-play sessions create caches
under the current inference object and should be scoped to that generation pass.

### Diagnostics

The search result diagnostics include:

```text
requested_states
cache_hits
duplicate_hits
unique_states
cache_inserts
cache_size
cache_size_peak
encoding_seconds
evaluator_seconds
```

These are useful for judging whether the cache is actually reducing evaluator
load.

## Tree Reuse Session

### What It Is

A tree reuse session keeps selected MCTS subtrees alive across moves in self-play.
This is similar to the common KataGo-style pattern: after the engine plays the
chosen move, promote the chosen child subtree to become the next root instead of
starting from an empty tree.

The Python boundary is:

```text
new_mcts_session(...)
BatchedMctsSession.run(...)
```

The Rust class is:

```text
Model1MctsSession
```

### Intent

Self-play repeatedly searches consecutive states in the same game. A lot of work
from the previous search is still relevant after the selected move. Reusing the
subtree is intended to:

- preserve useful statistics for the next turn;
- reduce repeated expansion/evaluation;
- make search more stable across consecutive moves;
- make long self-play games cheaper.

### Implementation Details

`Model1MctsSession` stores:

```text
searches: HashMap<u64, RustSearch>
evaluation_cache: SharedEvaluationCache
cache_max_states: usize
```

Python passes a `game_key` per active game. During a session search:

1. Rust clones the current root states.
2. For each `(game_key, root_state)`, Rust computes the root state hash.
3. If a stored search exists and its `root_hash` matches the passed state hash,
   Rust reuses that tree and calls `set_additional_visits(...)`.
4. If no matching search exists, Rust evaluates the root and creates a new
   `RustSearch`.
5. After search, Rust picks the selected root action.
6. `advance_root(action_id)` attempts to promote that action's child node.
7. If promotion succeeds and the new root is nonterminal, the session stores the
   promoted search under the same `game_key`.

`advance_root(...)` clones the selected subtree into a fresh node arena, rebuilds
the node table, applies the selected move to the stored root state, and recomputes
accounting.

### Baseline-Subtracted Visit Policy

When a session reuses a tree, the root may already have visits before the current
search call starts. The implementation records a per-action baseline before the
new visits run:

```text
baselines: Vec<HashMap<PackedCoord, u32>>
```

When building the result payload, the visit policy uses:

```text
edge.visits - baseline_visits_for_edge
```

This makes the returned `visits` and policy represent the current search call's
new simulations, not all historical simulations stored in the reused tree.

### How It Changes Model Behavior

Tree reuse makes consecutive self-play decisions less independent. The next move
starts with information from the previous search, so the model can effectively
spend more search effort on lines that stay relevant after the played move.

For training data, the baseline subtraction is meant to keep each decision's
policy target tied to the new visit budget, not the entire accumulated session
history.

### Current Caveat

The current implementation selects `action_id` from total accumulated root visits
but returns a baseline-subtracted visit policy. That can make the played action
reflect older visits while the policy target reflects only new visits. The cleaner
behavior is to select from the same baseline-subtracted visit counts used to
build the returned policy.

## Lazy And Staged Root/Child Edges

### What It Is

Lazy staged edges mean a node does not allocate all children immediately. A node
stores two child-related collections:

```text
edges: Vec<RustEdge>
unexpanded_priors: Vec<RustPriorCandidate>
```

`edges` are active tree edges with visits, value sums, pending counts, and an
optional child node id.

`unexpanded_priors` are policy-ranked candidate moves that do not yet have an
edge object.

### Intent

The intent is to avoid paying tree memory and traversal cost for children that
may never matter. This is especially valuable when:

- legal move count is large;
- the model returns many legal priors;
- progressive widening is active;
- many root searches are batched at once.

It also keeps node structure closer to a staged-child design: first keep policy
candidates compact, then instantiate real child edges only when selection needs
them.

### Implementation Details

When a node is created:

1. Rust receives a `RustEvaluation`.
2. Priors are sanitized and sorted.
3. Duplicate action ids are removed.
4. The candidate list is reversed so `.last()` is the highest-prior hidden
   candidate.
5. `edges` starts empty.

During selection:

1. Existing active edges are scored with PUCT:

   ```text
   score = edge.value + edge.prior * c_puct * sqrt(node.visits) / (1 + edge.visits)
   ```

2. If widening allows a new move, the best hidden candidate is scored as a new
   child with no visits.
3. If the hidden candidate wins, it is popped from `unexpanded_priors` and pushed
   into `edges`.
4. If there are no hidden priors left but the node still believes hidden legal
   moves exist, Rust can ask the engine state for the next legal action id and
   materialize it with a small fallback prior.

The pending count supports virtual batching. If an edge has a pending unevaluated
child, another selection pass will avoid selecting that same unevaluated child
again.

### How It Changes Model Behavior

Lazy staged edges make the active tree smaller and more policy-driven. Early
search behavior is more selective. Low-prior moves are not removed outright, but
they are delayed until widening and PUCT pressure justify materializing them.

Compared with allocating every child immediately, this reduces search noise and
CPU work. The tradeoff is that tactics outside the staged frontier can be delayed
or, at non-root nodes today, omitted if they are not in the returned candidate
set.

### Diagnostics

Tree diagnostics expose:

```text
active_edge_count
hidden_prior_count
root_active_edges
root_hidden_priors
max_active_edges_per_node
max_hidden_priors_per_node
widened_edges_total
```

These fields tell you how much of the tree is active versus still staged.

## Top-K Legal Prior Payloads

### What It Is

Top-k legal prior payloads are a reduced evaluator contract between Rust MCTS and
Python/PyTorch. Instead of asking Python to return priors for every legal move,
Rust can ask for only the top `k` legal priors per state.

The `k` value comes from:

```text
progressive_widening_candidate_actions
```

### Intent

For MCTS with progressive widening, Rust usually does not need a full legal
policy vector. It only needs enough high-prior candidates to seed the staged
frontier. Returning priors for every legal move can be expensive because it
requires:

- listing all legal flat indices;
- softmaxing over every legal action;
- transferring all legal priors back to Rust;
- storing a large prior list per expanded node.

The top-k payload reduces Python/Rust transfer size and keeps node prior storage
small.

### Implementation Details

When `prior_candidate_limit` is set, `mcts_eval.rs` changes the evaluator payload:

```text
input_dtype = "float16"
legal_mask_from_inputs = true
max_prior_candidates = k
```

Instead of sending a full `legal_flat_indices_bytes` list, Rust encodes the legal
mask into the dense input tensor's legal plane. Python then recovers legal cells
directly from the input tensor.

`DenseCNNInference.evaluate_model1_payload(...)` sees:

```text
legal_mask_from_inputs = true
max_prior_candidates > 0
```

and calls:

```text
_topk_legal_priors_from_input_mask(...)
```

That helper:

1. Reads the legal plane from the input tensor.
2. Masks non-legal logits to `-inf`.
3. Runs `torch.topk` over the policy logits.
4. Applies a softmax over the selected top-k logits.
5. Returns:

   ```text
   values_bytes
   priors_bytes
   selected_flat_indices_bytes
   selected_row_offsets
   ```

Rust then converts selected flat indices back into packed coordinate ids using
the crop center from the encoded state. It normalizes and truncates priors again
as a safety check.

### Alternative Full-Prior Mode

If no candidate limit is set, Rust sends:

```text
legal_flat_indices_bytes
legal_row_offsets
```

Python returns priors for all legal moves. This mode is wider and more complete,
but heavier.

### How It Changes Model Behavior

Top-k legal priors make the search more explicitly candidate-based. MCTS begins
from the highest-scoring legal actions under the dense-cnn policy head. This can
greatly reduce work when the legal set is large.

The behavior change is that very low-ranked legal moves may not appear in the
neural candidate list at all. At the root, the lazy fallback path can eventually
materialize legal moves outside the returned top-k list. At non-root nodes, the
current `total_legal_actions = candidates.len()` behavior means the returned
top-k list is effectively the complete child set unless that is changed.

## Rayon Parallel Root Leaf Selection

### What It Is

Rayon is used to select pending leaves from multiple root searches in parallel.
Each active root is an independent `RustSearch`, so root-level leaf selection can
run across CPU workers before the Python/PyTorch evaluator is called.

### Intent

Self-play often searches many active games at the same time. Each root tree can
be traversed independently. Parallel root selection is intended to:

- use CPU cores while preparing the next neural evaluation batch;
- reduce wall-clock time spent in Rust tree traversal;
- keep Python/PyTorch calls batched;
- avoid shared mutable tree state between threads.

### Implementation Details

`run_searches_to_targets(...)` uses:

```text
searches.par_iter_mut().enumerate()
```

Each Rayon worker owns one mutable `RustSearch` at a time. For each root, it
selects up to:

```text
leaf_batch_per_root
```

leaves, where `leaf_batch_per_root` is the resolved `virtual_batch_size`.

Within each root:

1. Rust calls `select_pending_leaf(...)`.
2. It applies a virtual visit immediately.
3. Terminal leaves are backed up immediately.
4. Already-expanded leaves are backed up immediately.
5. New unevaluated leaves are marked pending and returned to the coordinator.

After all workers finish, the coordinator merges all returned leaves into one
list. Only then does Rust touch the shared evaluator cache and call Python. This
keeps shared cache mutation out of the Rayon section.

`mcts_eval.rs` also uses Rayon during encoding:

- encode states in parallel;
- copy encoded planes into contiguous payload buffers in parallel.

### How It Changes Model Behavior

Rayon parallelism should not change the intended search result for fixed seeds,
fixed visit counts, and fixed virtual batching semantics. It changes how quickly
the pending leaf batch is prepared.

There is one behavioral nuance: virtual batching itself changes search dynamics
relative to strictly serial MCTS. Because several leaves can be selected before
their neural values are known, the algorithm uses pending counts and virtual
visits to discourage selecting the same unevaluated path repeatedly. This is a
standard throughput tradeoff: larger virtual batches improve evaluator
utilization, but they make selection slightly less sequentially informed.

## How These Features Work Together

These capabilities are designed to reinforce each other:

- top-k prior payloads reduce evaluator output size;
- lazy staged edges avoid instantiating unused top-k candidates;
- progressive widening controls when hidden candidates become active;
- virtual batching gathers enough leaves for efficient PyTorch calls;
- Rayon makes root traversal and encoding faster;
- evaluator caching avoids repeated neural calls;
- tree reuse carries useful search work across consecutive self-play moves.

The production intent is a KataGo-inspired shape:

```text
Search owns root state and tree
Traversal mutates search-local state only
PyTorch is a batched evaluator callback
Chosen child subtree can be promoted after a move
```

## Important Review Notes

These are not part of the intended capability design, but they are important when
reviewing the current implementation:

1. Session reuse currently returns a baseline-subtracted policy but selects the
   played action from total accumulated visits. Those should probably use the
   same visit basis.
2. Non-root nodes currently use `candidates.len()` as their legal action count.
   That makes non-root expansion candidate-pruned. Decide whether that is
   desired or change it to use full legal counts.
3. The engine state capsule currently passes a raw cloned Rust state pointer
   between extension modules. That should be replaced with a stable versioned
   wire payload or a stricter C ABI state representation.
4. Dense-cnn still has some Python wrapper/alternate-route code around session
   MCTS. If the goal is one production path, those wrapper checks should be
   trimmed.
