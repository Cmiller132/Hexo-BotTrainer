# Revised plan: Hexformer receives cloned engine states, not move-history rows

I agree with your correction. For this codebase, the cleanest model boundary is:

```text
Python engine HexoState
â†’ engine.clone_state(...)
â†’ model-specific Rust clones/owns the Rust state
â†’ Hexformer builds sparse payloads directly from that state
â†’ Python/PyTorch runs the network
```

The model should **not** use packed move history as its main live inference or self-play interface. Packed history can remain useful for record files, debugging, reproducibility, and compatibility tests, but the production model path should operate on **cloned engine state snapshots**.

This also fits the current repository. `hexo_engine.api` already exposes `clone_state(state)` as a public Python function, returning an independent mutable Rust state clone.  The PyO3 bridge also exposes `clone_state` directly and implements it as `state.state.clone()`.  The Hexformer self-play code already stores `engine.clone_state(state)` in each pending decision before applying the selected action, so the pipeline is already conceptually moving in this direction.

---

# 1. Core ownership rule

## 1.1 Keep `hexo_engine` authoritative and small

`hexo_engine` still owns:

```text
rules
legality
state transitions
terminal detection
turn phase
legal action IDs
board windows
placement history
clone_state
apply_action
to_python_state
```

Do **not** move Hexformer-specific logic into `hexo_engine`.

Hexformer-specific logic belongs in:

```text
packages/hexo_models/hexformer_ar/
```

The repository already establishes this package boundary. The README says `hexo_engine` owns canonical rules and state transitions, while `hexo_models` owns standalone production model families.  The engine Rust crate also explicitly says model, search, and sample code live outside the rules crate so the rules layer stays small, deterministic, and auditable.

## 1.2 Updated state-source rule

Replace this current Hexformer Rust capability:

```text
state_source = "packed_history_rows"
```

with:

```text
state_source = "engine_state_clone"
```

The current Hexformer Rust capability still reports `state_source = "packed_history_rows"`, so that should be updated once the clone-state path lands.

---

# 2. Why cloned engine state is better than history rows

The history-row path reconstructs state by replaying all placements. Current Hexformer Python converts a state into a tuple of packed coordinate IDs using `history_row_from_state()`.  The current Hexformer Rust `state_from_history_row()` then starts from `HexoState::new()` and reapplies every placement.

That works, but it has drawbacks:

```text
It repeats work already done by the engine.
It risks mismatch if runtime state gains fields not recoverable from history.
It makes every model call depend on serialization/replay.
It is less direct for MCTS, where leaf states already exist as Rust states.
It obscures the actual contract: the model wants a state snapshot, not a log.
```

A cloned engine state is better because:

```text
It preserves the exact board, legal store, window store, phase, terminal state, and caches.
It avoids replaying history just to recover a state already available.
It gives model-specific Rust direct read access to the same state structure used by rules/search.
It matches self-play, which already snapshots pending states by cloning.
It simplifies live inference and MCTS evaluator design.
```

---

# 3. Revised data path

## 3.1 Direct inference path

Old live path:

```text
HexoState
â†’ to_python_state(...)
â†’ placement_history
â†’ packed history row
â†’ Rust reconstructs HexoState by replay
â†’ sparse payload
â†’ model
```

New live path:

```text
HexoState
â†’ engine.clone_state(...)
â†’ _rust.hexformer_ar.sparse_input_payload_from_state(...)
â†’ model-specific Rust clones/owns RustHexoState
â†’ sparse payload
â†’ SparseDecisionInput
â†’ PyTorch HexformerAR
```

## 3.2 Self-play path

Old self-play sample finalization:

```text
PendingDecision.state clone
â†’ build_selfplay_sample_payloads(...)
â†’ history_row_from_state(...)
â†’ Rust replay history
â†’ sparse payload
```

New self-play sample finalization:

```text
PendingDecision.state clone
â†’ build_selfplay_sample_payloads_from_states(...)
â†’ Rust receives cloned state handles
â†’ Rust clones/owns RustHexoState through engine state API
â†’ sparse payload
```

The pending self-play state should remain exactly as it is now: `PendingDecision(state=engine.clone_state(state), ...)`.  The change is that `_finalize_pending()` should stop converting those states into history rows before handing them to Rust.

## 3.3 MCTS path

Old MCTS evaluator path:

```text
Rust MCTS leaf state
â†’ convert leaf state to placement history row
â†’ Python evaluator rebuilds sparse input from history row
â†’ model eval
```

New MCTS evaluator path:

```text
Rust MCTS leaf state
â†’ build sparse payload directly in hexformer_ar Rust
â†’ Python evaluator receives sparse payloads
â†’ SparseDecisionInput.from_payload(...)
â†’ model eval
```

This is important. Inside model-specific Rust MCTS, leaf states are already `RustHexoState` values. There is no need to serialize them into histories and replay them again.

---

# 4. New model-state bridge design

## 4.1 Use the existing engine state API capsule

`hexo_engine` already exposes a C-style state API capsule with:

```text
clone_state
free_state
version
```

The capsule struct and version are defined in the Rust bridge.  The Python-visible `state_api_capsule()` function returns the capsule.  The capsule clone function allocates a cloned `RustHexoState`, and the free function releases it.

Hexformer should use this existing capsule. Do not add a new engine API unless absolutely necessary.

## 4.2 Add model-side Rust wrapper

Add:

```text
packages/hexo_models/hexformer_ar/rust/src/engine_state.rs
```

Responsibilities:

```text
load hexo_engine._rust.state_api_capsule()
validate API version
clone Python HexoState into owned RustHexoState
free temporary capsule-owned pointer
return model-owned RustHexoState
```

Suggested Rust shape:

```rust
pub(crate) struct EngineStateApi {
    version: u32,
    clone_state: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    free_state: unsafe extern "C" fn(*mut c_void),
}

pub(crate) fn clone_py_engine_state(
    py: Python<'_>,
    state_obj: &Bound<'_, PyAny>,
) -> PyResult<RustHexoState> {
    let api = load_state_api(py)?;
    let mut raw: *mut c_void = std::ptr::null_mut();

    let code = unsafe {
        (api.clone_state)(state_obj.as_ptr().cast::<c_void>(), &mut raw)
    };

    if code != 0 || raw.is_null() {
        return Err(PyValueError::new_err(format!(
            "failed to clone HexoState through state API: code={code}"
        )));
    }

    let cloned_ref: &RustHexoState = unsafe { &*(raw.cast::<RustHexoState>()) };
    let owned = cloned_ref.clone();

    unsafe {
        (api.free_state)(raw);
    }

    Ok(owned)
}
```

The wrapper must not store borrowed engine pointers beyond the call. It should return a model-owned `RustHexoState`.

## 4.3 Safety rules

Strict rules:

```text
Never mutate caller-owned Python HexoState.
Always clone at the boundary.
Never store raw capsule pointers after the function returns.
Always free capsule-owned cloned pointers.
Always validate state API version.
Fail fast on capsule version mismatch.
```

Recommended metadata:

```text
state_source = "engine_state_clone"
state_api_version = 2
```

The current engine bridge reports `STATE_API_VERSION = 2`, so Hexformer should require version 2 for this plan.

---

# 5. Revised Rust module layout

Update `hexformer_ar/rust/src/lib.rs` from:

```rust
mod constants;
mod state;
mod mcts_eval;
mod mcts_tree;
mod mcts;
mod sample_gen;
```

to:

```rust
mod constants;
mod engine_state;
mod mcts_eval;
mod mcts_tree;
mod mcts;
mod sample_gen;
```

Optional compatibility module:

```rust
mod history_state_compat;
```

Only keep the history-row path under a compatibility module if tests or old sample records still need it.

The current `lib.rs` is already the correct PyO3 export map and should stay that way. It currently registers capabilities, MCTS functions, and sample generation.

---

# 6. Revised Rust public functions

## 6.1 Replace history-row payload APIs

Current Rust API:

```rust
sparse_input_payload(history_row, architecture, candidates, ...)
sparse_input_payloads(history_rows, architecture, candidates)
selfplay_sample_payloads(game_id, history_rows, ...)
```

New Rust API:

```rust
sparse_input_payload_from_state(state, architecture, candidates, ...)
sparse_input_payloads_from_states(states, architecture, candidates)
selfplay_sample_payloads_from_states(game_id, states, ...)
```

Python-visible signatures:

```python
_rust.hexformer_ar.sparse_input_payload_from_state(
    state,
    architecture,
    candidates,
    policy,
    opp_policy,
    value,
    distance,
    lookahead,
    metadata,
)
```

```python
_rust.hexformer_ar.sparse_input_payloads_from_states(
    states,
    architecture,
    candidates,
)
```

```python
_rust.hexformer_ar.selfplay_sample_payloads_from_states(
    game_id,
    states,
    players,
    turn_indices,
    visit_policies,
    root_values,
    search_visits,
    selected_action_ids,
    winner,
    architecture,
    candidates,
)
```

## 6.2 Internal implementation

The internal pipeline stays the same after state acquisition.

```rust
let state = clone_py_engine_state(py, state_obj)?;
let payload = build_sparse_payload(&state, &arch, &candidate_cfg, &targets);
```

The existing `build_sparse_payload()` function should remain the core assembly function. It already does the correct pipeline: legal actions, tactical scan, candidate frontier, anchor selection, candidate/stone/window features, local windows, relation edges, global features, and targets.

---

# 7. Revised Python input API

## 7.1 Update `input.py`

Current `build_sparse_input()` does:

```python
payload = _hexformer_ar_rust().sparse_input_payload(
    history_row_from_state(state),
    ...
)
```

That should become:

```python
state_clone = engine.clone_state(state)
payload = _hexformer_ar_rust().sparse_input_payload_from_state(
    state_clone,
    _config_mapping(arch),
    _config_mapping(candidate_cfg),
    _policy_items(policy),
    _policy_items(opp_policy),
    None if value is None else float(value),
    None if distance is None else float(distance),
    _lookahead_items(lookahead),
    dict(metadata or {}),
)
```

Keep the public Python function name:

```python
build_sparse_input(state, ...)
```

This preserves call sites while changing the internal state source.

## 7.2 Update batch input builder

Replace:

```python
build_sparse_inputs_from_history_rows(...)
```

with:

```python
build_sparse_inputs_from_states(states, ...)
```

Implementation:

```python
def build_sparse_inputs_from_states(
    states: Sequence[object],
    *,
    architecture: HexformerArchitectureConfig | None = None,
    candidates: HexformerCandidateConfig | None = None,
) -> tuple[SparseDecisionInput, ...]:
    arch = architecture or HexformerArchitectureConfig()
    candidate_cfg = candidates or HexformerCandidateConfig(max_candidates=arch.max_candidates)
    clones = tuple(engine.clone_state(state) for state in states)
    payloads = _hexformer_ar_rust().sparse_input_payloads_from_states(
        clones,
        _config_mapping(arch),
        _config_mapping(candidate_cfg),
    )
    return tuple(_sparse_input_from_payload(payload) for payload in payloads)
```

## 7.3 Deprecate history helpers

Deprecate, but do not immediately delete:

```python
history_row_from_state
build_sparse_inputs_from_history_rows
```

Use warnings:

```python
DeprecationWarning:
"history-row Hexformer input is compatibility-only; use cloned engine states."
```

This makes migration safer.

---

# 8. Revised inference API

## 8.1 Direct inference

Current `HexformerInference.infer_states()` calls `build_sparse_input(state, ...)`.  Once `build_sparse_input()` is changed internally, this API remains valid.

No external change needed:

```python
inference.infer_state(state)
inference.infer_states(states)
```

But internally it now uses cloned engine state, not history rows.

## 8.2 MCTS evaluator callback

Current evaluator callback expects:

```text
payload["history_rows"]
```

and rebuilds sparse inputs from history rows.

Replace with:

```text
payload["sparse_payloads"]
```

New implementation:

```python
@torch.no_grad()
def evaluate_mcts_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
    if "sparse_payloads" in payload:
        sparse = tuple(
            sparse_input_from_payload(item)
            for item in payload["sparse_payloads"]
        )
    else:
        # temporary compatibility only
        sparse = build_sparse_inputs_from_states(
            payload["states"],
            architecture=self.config.architecture,
            candidates=self.config.candidates,
        )

    results = self.infer_sparse(sparse)

    values = torch.tensor([r.value for r in results], dtype=torch.float32).contiguous()
    candidate_rows = tuple(tuple(r.legal_action_ids) for r in results)
    priors = torch.tensor(
        [
            float(r.legal_priors.get(action_id, 0.0))
            for r in results
            for action_id in r.legal_action_ids
        ],
        dtype=torch.float32,
    ).contiguous()

    return {
        "values_bytes": values.numpy().tobytes(),
        "candidate_action_ids": candidate_rows,
        "priors_bytes": priors.numpy().tobytes(),
    }
```

The important design change is that Rust MCTS should send already-built sparse payloads, not histories.

---

# 9. Revised MCTS design

## 9.1 Python `mcts.py`

The current Python `run_batched_mcts()` already passes `tuple(root_states)` into `_rust.hexformer_ar.hexformer_ar_batched_mcts(...)`. It also passes architecture and candidate config mappings.

Keep that Python shape.

The Rust MCTS signature should be updated to match it.

## 9.2 Rust `mcts.rs`

Current Rust `hexformer_ar_batched_mcts()` still names its first argument `history_rows` and reconstructs roots with `states_from_history_rows(...)`.

Replace that with:

```rust
#[pyfunction(signature = (
    root_states,
    visits,
    c_puct,
    temperature,
    seed,
    evaluator,
    architecture,
    candidates,
    virtual_batch_size=None
))]
pub fn hexformer_ar_batched_mcts(
    py: Python<'_>,
    root_states: &Bound<'_, PyAny>,
    visits: u32,
    c_puct: f32,
    temperature: f32,
    seed: u64,
    evaluator: &Bound<'_, PyAny>,
    architecture: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    virtual_batch_size: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let roots = clone_engine_states(py, root_states)?;
    ...
}
```

## 9.3 Rust MCTS leaf evaluation

Current `mcts_eval.rs` serializes leaf states into `history_rows` and `legal_action_ids` before calling Python.

Replace with:

```rust
fn evaluate_states(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[RustHexoState],
    architecture: &ArchitectureConfig,
    candidates: &CandidateConfig,
) -> PyResult<Vec<RustEvaluation>> {
    let sparse_payloads = PyList::empty(py);

    for state in states {
        let targets = SparseTargets::default();
        let sparse = build_sparse_payload(state, architecture, candidates, &targets);
        sparse_payloads.append(sparse_payload_to_py(py, &sparse, empty_metadata)?)?;
    }

    let payload = PyDict::new(py);
    payload.set_item("sparse_payloads", sparse_payloads)?;
    payload.set_item("state_source", "engine_state_clone")?;

    let output = evaluator.call1((payload,))?;
    parse_evaluation_output(&output, states)
}
```

This means:

```text
Rust MCTS owns Rust states.
Rust sample_gen builds sparse payloads from Rust states.
Python/PyTorch only evaluates sparse tensors.
No history replay inside MCTS evaluation.
```

## 9.4 Cache keys

Current `mcts_eval.rs` uses placement-history packed coordinates as cache keys.

This can remain for now as an internal cache key:

```text
state_cache_key = placement_history coordinates
```

This is not the model input path; it is only a stable hash-like identifier for memoization. Later, replace it with a native state hash if `hexo_engine` exposes one.

---

# 10. Revised sample finalization

## 10.1 Keep `PendingDecision.state`

Self-play already does the right thing:

```python
PendingDecision(
    state=engine.clone_state(state),
    player=...,
    turn_index=...,
    search=...
)
```

Keep it.

## 10.2 Change `_finalize_pending()`

Current `_finalize_pending()` calls `build_selfplay_sample_payloads(states=...)`, but that function currently converts those states into rows internally.

Keep the Python call shape:

```python
build_selfplay_sample_payloads(states=tuple(decision.state for decision in pending), ...)
```

Change only the implementation inside `input.py`:

```python
return tuple(
    _hexformer_ar_rust().selfplay_sample_payloads_from_states(
        game_id,
        tuple(engine.clone_state(state) for state in states),
        ...
    )
)
```

Do not convert to `history_row_from_state()`.

---

# 11. Revised sparse payload contract

The `SparseDecisionInput` dataclass remains correct.

Keep:

```text
candidate_action_ids
candidate_features
candidate_coords
candidate_mask
stone_features
stone_coords
stone_mask
window_features
window_coords
window_mask
local_input
local_inputs
local_window_coords
local_window_mask
rel_edge_index
rel_edge_features
rel_edge_mask
global_features
policy_target
opp_policy_target
wdl_target
distance_target
threat_target
relevance_target
lookahead_targets
metadata
```

The current dataclass already includes these fields.  The current collator already pads sparse candidates, stones, windows, local windows, relative edges, global features, and targets.

Add metadata fields:

```text
state_source = "engine_state_clone"
state_api_version = 2
placements_made
phase
current_player
legal_action_count
candidate_count
```

---

# 12. Revised candidate and tactical generation

No conceptual change here.

Candidate frontier remains model-specific Rust.

Current `sample_gen.rs` already owns:

```text
tactical windows
candidate frontiers
local crops
relation edges
target tensors
```

and explicitly says it does not rely on engine-side model accelerators.

Keep:

```rust
build_tactical_summary(&state, &legal_action_ids)
build_candidate_frontier(&state, &legal_action_ids, &tactical, &candidate_cfg)
build_local_windows(...)
build_rel_edges(...)
build_global_features(...)
```

The only change is that `state` now comes from a cloned engine state instead of replayed history.

---

# 13. Revised public API map

## 13.1 Python public functions

Keep stable:

```python
build_sparse_input(state, ...)
build_selfplay_sample_payloads(states=..., ...)
HexformerInference.infer_state(state)
HexformerInference.infer_states(states)
run_mcts(root_state, ...)
run_batched_mcts(root_states, ...)
```

Change internals only.

## 13.2 Rust PyO3 functions

Add:

```rust
sparse_input_payload_from_state
sparse_input_payloads_from_states
selfplay_sample_payloads_from_states
hexformer_ar_mcts
hexformer_ar_batched_mcts
```

`hexformer_ar_mcts` should accept a root state object, not a history row.

## 13.3 Compatibility functions

Keep temporarily:

```rust
sparse_input_payload_from_history_row
sparse_input_payloads_from_history_rows
```

Mark as compatibility-only.

Remove once:

```text
tests updated
self-play updated
inference updated
MCTS updated
sample finalization updated
```

---

# 14. File-by-file implementation instructions

## 14.1 `hexformer_ar/rust/src/lib.rs`

Update capabilities:

```rust
dict.set_item("state_source", "engine_state_clone")?;
dict.set_item("engine_state_clone", true)?;
dict.set_item("history_row_compat", true)?; // temporary
```

Keep:

```rust
mcts::register_pybridge(module)?;
sample_gen::register_pybridge(module)?;
```

Current registration pattern is already correct.

## 14.2 `hexformer_ar/rust/src/engine_state.rs`

New file.

Implement:

```rust
load_state_api(py) -> PyResult<EngineStateApi>
clone_py_engine_state(py, state_obj) -> PyResult<RustHexoState>
clone_py_engine_states(py, states_obj) -> PyResult<Vec<RustHexoState>>
```

Use existing `hexo_engine._rust.state_api_capsule()`.

## 14.3 `hexformer_ar/rust/src/sample_gen.rs`

Change entry functions from history rows to states.

Before:

```rust
let state = state_from_history_row(history_row)?;
```

After:

```rust
let state = clone_py_engine_state(py, state_obj)?;
```

Before:

```rust
let states = states_from_history_rows(history_rows)?;
```

After:

```rust
let states = clone_py_engine_states(py, states_obj)?;
```

Keep `build_sparse_payload(&state, ...)` unchanged except for metadata additions.

## 14.4 `hexformer_ar/python/hexo_models/hexformer_ar/input.py`

Remove production use of:

```python
history_row_from_state(state)
```

Change `build_sparse_input()` and batch builders to pass cloned engine states to Rust.

Current `build_sparse_input()` already hides the Rust call behind a Python function, so call sites do not need to change.

## 14.5 `hexformer_ar/python/hexo_models/hexformer_ar/inference.py`

Change `evaluate_mcts_payload()` so it consumes:

```text
sparse_payloads
```

instead of:

```text
history_rows
```

The rest of inference can stay the same: `infer_sparse()` collates, runs the model, softmaxes policy logits, maps priors back to candidate IDs, and returns values.

## 14.6 `hexformer_ar/rust/src/mcts.rs`

Change root input from `history_rows` to `root_states`.

Python already calls Rust MCTS with `tuple(root_states)`, architecture config, and candidate config.  Align Rust with that Python boundary.

## 14.7 `hexformer_ar/rust/src/mcts_eval.rs`

Replace history-row evaluator payload with sparse payload evaluator payload.

Before:

```rust
payload.set_item("history_rows", history_rows)?;
payload.set_item("legal_action_ids", legal_rows)?;
```

After:

```rust
payload.set_item("sparse_payloads", sparse_payloads)?;
payload.set_item("state_source", "engine_state_clone")?;
```

Keep `parse_evaluation_output()` mostly as-is because Python still returns candidate action rows, priors, and values.

## 14.8 `hexformer_ar/rust/src/mcts_tree.rs`

No required structural rewrite.

The tree already owns `RustHexoState` values and applies placements internally.

Later optimization:

```text
replace repeated root_state.clone() in select_pending_leaf with scratch state + apply_with_delta/undo
```

Do this only after clone-state ingestion is working.

---

# 15. Updated tests

## 15.1 New Rust tests

Add to model-specific Rust tests:

```text
clone_py_engine_state_rejects_non_state
clone_py_engine_state_preserves_phase
clone_py_engine_state_preserves_legal_count
sparse_payload_from_state_matches_history_compat_payload
sparse_payloads_from_states_batch_order_is_stable
selfplay_sample_payloads_from_states_preserve_turn_indices
mcts_accepts_root_state_objects
mcts_does_not_mutate_root_state
mcts_eval_sends_sparse_payloads_not_history_rows
```

## 15.2 Python tests

Add under:

```text
tests/models/hexformer_ar/
```

Required tests:

```python
def test_build_sparse_input_uses_clone_state_not_history_rows(): ...
def test_build_sparse_inputs_from_states_batch(): ...
def test_evaluate_mcts_payload_accepts_sparse_payloads(): ...
def test_run_mcts_accepts_engine_state_object(): ...
def test_selfplay_finalizer_passes_cloned_states(): ...
```

## 15.3 Compatibility tests

While both paths exist:

```text
history_row payload and cloned-state payload should match for simple positions
history_row path emits DeprecationWarning
```

Once the clone path is stable, remove history-row compatibility tests.

---

# 16. Updated acceptance criteria

The refactor is accepted when:

```text
build_sparse_input(state) no longer calls history_row_from_state in production
selfplay sample finalization no longer converts states to history rows
Rust MCTS root input is Python HexoState objects
Rust MCTS evaluator sends sparse_payloads to Python
Hexformer capabilities report state_source = engine_state_clone
dense_cnn remains unchanged
hexo_engine has no major new model-specific APIs
```

Functional acceptance:

```text
Hexformer direct inference works from engine.new_game()
Hexformer direct inference works after several applied actions
Hexformer self-play completes at least one game
Hexformer samples round-trip through replay buffer
Hexformer MCTS returns legal action IDs
Root state is unchanged after run_mcts()
Sparse payloads from clone-state and history-compat match on simple fixtures
```

Performance acceptance:

```text
Clone-state sparse payload generation is no slower than history replay for midgame states.
MCTS no longer spends avoidable time rebuilding states from history for every evaluator call.
Candidate counts and policy rows match previous behaviour.
```

---

# 17. Updated implementation sequence

## Phase 1 â€” Add model-side state clone wrapper

Files:

```text
hexformer_ar/rust/src/engine_state.rs
hexformer_ar/rust/src/lib.rs
```

Tasks:

```text
Load hexo_engine state API capsule.
Validate version.
Clone Python HexoState into RustHexoState.
Free capsule pointer safely.
Expose capability state_source = engine_state_clone.
```

Exit criteria:

```text
Rust test can clone engine.new_game() state.
Rust test can clone a non-opening state.
No root state mutation.
```

## Phase 2 â€” Change sparse input generation

Files:

```text
hexformer_ar/rust/src/sample_gen.rs
hexformer_ar/python/hexo_models/hexformer_ar/input.py
```

Tasks:

```text
Add sparse_input_payload_from_state.
Add sparse_input_payloads_from_states.
Update build_sparse_input.
Update build_sparse_inputs_from_states.
Deprecate history-row helpers.
```

Exit criteria:

```text
build_sparse_input(state) works without history replay.
SparseDecisionInput shapes unchanged.
Payload metadata says engine_state_clone.
```

## Phase 3 â€” Change self-play finalization

Files:

```text
hexformer_ar/python/hexo_models/hexformer_ar/selfplay.py
hexformer_ar/python/hexo_models/hexformer_ar/input.py
hexformer_ar/rust/src/sample_gen.rs
```

Tasks:

```text
Keep PendingDecision.state = engine.clone_state(state).
Update build_selfplay_sample_payloads to call Rust from cloned states.
Remove history row conversion from finalizer path.
```

Exit criteria:

```text
self-play produces HexformerSample objects.
Training records still append to sample store.
Stored payload schema unchanged except metadata.
```

## Phase 4 â€” Change MCTS evaluator path

Files:

```text
hexformer_ar/rust/src/mcts.rs
hexformer_ar/rust/src/mcts_eval.rs
hexformer_ar/python/hexo_models/hexformer_ar/inference.py
hexformer_ar/python/hexo_models/hexformer_ar/mcts.py
```

Tasks:

```text
Make Rust MCTS accept root state objects.
Clone root states via engine_state.rs.
Build sparse payloads from Rust leaf states.
Send sparse_payloads to Python evaluator.
Parse values/candidate IDs/priors as before.
```

Exit criteria:

```text
run_mcts(state, inference, ...) works.
run_batched_mcts((state1, state2), ...) works.
Evaluator no longer receives history_rows in production.
```

## Phase 5 â€” Cleanup

Tasks:

```text
Mark history-row functions compatibility-only.
Update docs.
Update diagnostics.
Remove stale capability state_source = packed_history_rows.
Add tests.
```

Exit criteria:

```text
No production Hexformer code path depends on move-history replay.
No major hexo_engine changes.
dense_cnn still passes existing tests.
```

---

# 18. Final revised spec

Use this as the updated implementation contract:

```text
HexformerAR state interface:
  Input object:
    cloned Python HexoState

  Python entry:
    build_sparse_input(state, ...)
    run_mcts(state, ...)
    build_selfplay_sample_payloads(states=...)

  Rust state acquisition:
    hexformer_ar/rust/src/engine_state.rs
    clone via hexo_engine._rust.state_api_capsule()

  Rust sparse generation:
    build_sparse_payload(&RustHexoState, ...)

  Rust MCTS:
    root states cloned from Python HexoState objects
    leaf states remain native RustHexoState values
    evaluator receives sparse_payloads, not history_rows

  Durable records:
    .hxr / replay may still store action history for audit and reproduction

  Production rule:
    no live model inference, sample finalization, or MCTS evaluation may rebuild state by replaying move history
```

This keeps `hexo_engine` clean, uses its existing clone-state API, and makes Hexformerâ€™s model-specific Rust code operate on actual engine state snapshots instead of serialized history rows.
