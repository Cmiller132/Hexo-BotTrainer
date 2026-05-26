# Verified Review Fix Plan

This file contains implementation-ready requirements for review items 2 through 6.
Item 1, the record/sample architecture question, is intentionally not specified
here because it needs a separate design decision about replay records versus
training samples.

Recommended implementation order:

1. Standardize action IDs.
2. Add the max-action guard.
3. Add invariant validation tests.
4. Thin `HexoEngineAdapter`.
5. Add apply/undo deltas for MCTS search.

## 2. Add A Max-Action Guard

### Problem

`hexo_runner.loop.run_match_loop` currently runs until
`HexoEngineAdapter.terminal(primary_state)` returns a terminal result. The
current rules are no-draw, but the board is unbounded and legal play can keep
expanding. A rule bug, model loop, adapter bug, or intentionally evasive player
could keep one worker occupied for an unbounded number of placements.

### Requirements

- Add `max_actions: int = 1024` to `hexo_runner.session.GameSpec`.
- The value must be positive. Reject `max_actions <= 0` before a game starts.
- The runner must abort a game before requesting another decision once the
  number of accepted actions has reached `max_actions`.
- A game that becomes terminal exactly on the `max_actions`th accepted action
  must be recorded as completed, not aborted.
- The abort must be structured and durable in the `.hxr` record:
  - `stage`: `runner.max_actions`
  - `exception_type`: `MaxActionsExceeded`
  - `message`: include the game id and configured limit.
- The aborted `.hxr` record must keep all accepted action IDs before the abort.
- The guard must apply equally through `run_match`, `run_batch`, and direct
  `run_match_loop` calls because they all share the same loop.

### Implementation Notes

- File: `packages/hexo_runner/python/hexo_runner/session.py`
  - Add the field to `GameSpec`.
  - Implement validation in `__post_init__`. Because the dataclass is frozen,
    validation should only inspect values.
- File: `packages/hexo_runner/python/hexo_runner/loop.py`
  - Check the action count at the top of the loop, after `record_writer` is
    available and before cloning the state for `decide`.
  - Raise `RunnerAbort(AbortRecord(...))` so normal abort finalization writes
    the record.
  - Use `record_writer.action_count` as the authoritative accepted-action count.
- Do not add a separate timeout-based guard in this pass. Timeouts are a
  different policy problem and should be handled at player/adapter boundaries.

### Validation

- Add a runner test with scripted legal players and `GameSpec(max_actions=1)`.
  The game should record one accepted action, then abort before the next
  decision.
- Add a test where the winning move is exactly at `max_actions`; it should
  complete.
- Add a test that `GameSpec(max_actions=0)` raises `ValueError`.
- Verify the `.hxr` aborted record contains accepted actions plus abort metadata.
- Run:
  - `python -m unittest tests.test_hexo_runner_match_mode`
  - `python -m pytest`

## 3. Thin `HexoEngineAdapter`

### Problem

`HexoEngineAdapter` is useful as a runner boundary, but it currently contains
unused payload conversion helpers and still exposes string action IDs through
`action_id`. The runner should keep the adapter as a centralized engine facade
while avoiding Python-side game semantics that Rust already owns.

### Requirements

- Keep `HexoEngineAdapter` as the runner's engine boundary.
- Remove unused methods:
  - `action_payload`
  - `transition_payload`
- Keep pass-through methods that centralize direct engine calls:
  - `metadata`
  - `new_game`
  - `clone_state`
  - `current_player`
  - `apply_action`
  - `terminal`
  - `terminal_payload`
- Keep `player_index` and `player_role` unless a cleaner local helper is added
  in the runner loop. Do not scatter that mapping throughout the codebase.
- After item 5 lands, `HexoEngineAdapter.action_id` must return packed `int`
  action IDs, not `"q,r"` strings.
- Remove dead imports made unnecessary by deleting payload helpers.

### Implementation Notes

- File: `packages/hexo_runner/python/hexo_runner/engine.py`
  - Delete unused methods and rerun `rg "action_payload|transition_payload"`.
  - Keep `_jsonable` if still used by `metadata` or `terminal_payload`.
- File: `packages/hexo_runner/python/hexo_runner/loop.py`
  - No behavior change should be required other than the action ID type update
    from item 5.
- This cleanup should be done after action ID standardization if both are in the
  same implementation pass, so type signatures are updated once.

### Validation

- `rg "action_payload|transition_payload" packages tests scripts` returns no
  usages.
- Existing runner tests still pass.
- Run:
  - `python -m unittest tests.test_hexo_runner_match_mode`
  - `python -m pytest`

## 4. Add Apply/Undo Deltas For MCTS

### Problem

MCTS currently clones the full `HexoState` at the start of every simulation:
`SearchPosition::from_root(root_state)` clones `HexoState`, and
`run_simulation` calls it for every visit. That is simple and correct, but it
becomes expensive as visits, active games, and resident search trees grow.

`HexoState` clone cost includes:

- `Board.stones`
- `Board.occupied`
- `Board.windows`
- `Board.legal`
- `placement_history`
- `last_turn`
- phase, player, placement count, terminal state

### Requirements

- Add an undo-capable apply path in Rust without changing the existing public
  behavior of `apply_placement`.
- Introduce an explicit `ApplyDelta` that stores enough information to restore
  the exact pre-move state.
- Introduce a board-level delta that stores all board cache changes needed for
  undo.
- `HexoState::apply_with_delta` must:
  - validate the placement using the same rule path as `apply_placement`;
  - apply the move through the authoritative engine logic;
  - return both the current `ApplyResult` and an `ApplyDelta`.
- `HexoState::undo(delta)` must restore:
  - board stone map;
  - occupied list;
  - legal move store membership/order/version;
  - window masks for all changed windows;
  - current player;
  - turn phase;
  - placement count;
  - terminal outcome;
  - last turn;
  - placement history length/content.
- Keep the existing free function:
  - `apply_placement(state, placement) -> Result<ApplyResult, MoveError>`
  - It can internally call the new method and discard the delta.
- Update MCTS search to avoid cloning the root state for every simulation.
  `SearchPosition` should walk one mutable state down the tree and undo back up
  after the simulation.
- Do not add inference batching, evaluator queues, or new search algorithms in
  this pass.

### Implementation Notes

- Likely files:
  - `packages/hexo_engine/rust/src/state.rs`
  - `packages/hexo_engine/rust/src/board.rs`
  - `packages/hexo_engine/rust/src/legal.rs`
  - `packages/hexo_engine/rust/src/tactics.rs`
  - `packages/hexo_utils/rust/src/position.rs`
  - `packages/hexo_utils/rust/src/mcts/search.rs`
- `BoardDelta` should avoid full map clones. It should capture only:
  - placed coordinate and previous stone state;
  - previous occupied length;
  - legal store insertions/removals/version before mutation;
  - previous masks for touched window keys.
- `WindowStore` needs helper methods to apply a placement while returning prior
  masks and to restore those masks.
- `LegalMoveStore` needs an undo-friendly update path or a delta that records
  inserted/removed packed IDs and previous version.
- `SearchPosition` can maintain a `Vec<ApplyDelta>` for the current path.
  During one simulation, push deltas as actions are applied, then pop and undo
  before returning.
- Be careful with terminal moves. Undo must restore the non-terminal pre-move
  state exactly.

### Validation

- Add Rust engine tests:
  - applying then undoing one opening placement restores a fresh state;
  - applying then undoing a first-stone placement restores phase/player/legal
    moves;
  - applying then undoing a second-stone placement restores turn ownership and
    `last_turn`;
  - applying then undoing a winning placement restores terminal to `None`;
  - repeated apply/undo over random legal games returns to a byte-for-byte
    equivalent public state at every step.
- Add MCTS tests:
  - running MCTS does not mutate the root state;
  - search result matches legal root actions;
  - repeated searches from the same root remain deterministic with a fixed
    evaluator/config if deterministic before this change.
- Run:
  - `cargo fmt`
  - `cargo test -p hexo_engine`
  - `cargo test -p hexo_utils`
  - `cargo test --workspace`

## 5. Standardize Packed Action IDs

### Problem

Python currently has two action identity systems:

- `ActionId = str`
- `LegalActionId = int`

The Rust engine already uses compact packed `u32` IDs for legal actions and
`.hxr` records. The Python bridge returns packed integer legal IDs, but
`hexo_engine.action_id()` returns strings like `"q,r"`. Runner events and
training sample contracts still reference string action IDs. This is a real
schema inconsistency.

### Requirements

- Use packed integer action IDs everywhere an action identity is passed between
  engine, runner, records, samples, policy targets, model inputs, and adapters.
- Change Python type aliases:
  - `ActionId = int`
  - `LegalActionId = int`
- Change Rust PyO3 `action_id(q, r)` to return `u32` from `pack_coord`.
- Change `hexo_engine.api.action_id()` to return an integer.
- Change `hexo_runner.player.TransitionEvent.action_id` to `int`.
- Change `HexoEngineAdapter.action_id()` to return `int`.
- Change sample/training/model contracts that currently use string action IDs:
  - `hexo_utils.samples.records.PolicyOutputRecord.selected_action_id`
  - `hexo_utils.samples.records.TrainingSampleRecord.legal_action_ids`
  - `hexo_utils.samples.targets.LegalPolicyValueTarget`
  - `hexo_utils.encoding.symmetry.ActionSymmetryMapper`
  - `hexo_utils.encoding.masks`
  - `hexo_model_resnet.input.ResNetInput.action_ids`
- Keep string formatting only as display/logging helpers, never as identity.
- Do not add backwards-compatible string aliases. This should be a hard cleanup.

### Implementation Notes

- File: `packages/hexo_engine/rust/src/pybridge.rs`
  - Import/use `pack_coord`.
  - Return `u32` from `action_id`.
- File: `packages/hexo_engine/python/hexo_engine/types.py`
  - Change aliases and any type annotations.
  - Keep `pack_coord_id` and `unpack_coord_id`.
  - Consider adding `format_coord_id(action_id: int) -> str` only for display.
- File: `packages/hexo_runner/python/hexo_runner/player.py`
  - Update `TransitionEvent.action_id`.
- File: `packages/hexo_runner/python/hexo_runner/engine.py`
  - Update return annotation.
- Files under `packages/hexo_utils/python/hexo_utils/samples` and
  `packages/hexo_utils/python/hexo_utils/encoding`
  - Update type annotations and tests from string IDs to ints.
- Files under `packages/hexo_model_resnet/python`
  - Update action ID type annotations to ints.
- Tests that currently create sample records with IDs like `"a"` and `"b"`
  should use packed ints. If the test does not care about real coordinates,
  use small integer constants and name them clearly.

### Validation

- `rg "ActionId = str|tuple\\[str, \\.\\.\\.\\]|selected_action_id: str|action_id: str" packages tests`
  should show no identity-bearing action ID annotations. Review any remaining
  string hits manually to confirm they are display text only.
- Add/adjust Python bridge tests:
  - `engine.action_id(PlacementAction(AxialCoord(0, 0)))` equals
    `pack_coord_id(AxialCoord(0, 0))`.
  - `legal_action_ids(state)` and `LegalActions.action_ids` remain packed ints.
- Add/adjust runner tests:
  - `TransitionEvent.action_id` is an `int`.
  - `.hxr` action IDs match event action IDs.
- Run:
  - `cargo test -p hexo_engine --features python`
  - `python -m unittest tests.test_hexo_engine_rust_bridge`
  - `python -m unittest tests.test_hexo_runner_match_mode`
  - `python -m pytest`

## 6. Add Engine Invariant Validation Tests

### Problem

The engine now uses incremental caches for legal moves and six-cell windows.
That is much faster than scanning from scratch, but correctness depends on
those caches staying synchronized with authoritative stones and history.
Current tests cover important examples, but they do not stress random legal
games or validate every cache invariant after every move.

### Requirements

- Add deterministic property-style Rust tests that replay random legal games.
- Tests must validate after every accepted placement:
  - `Board.stones` and `Board.occupied` agree.
  - Every occupied coordinate is not legal.
  - Every legal coordinate is empty.
  - Every stored `WindowStore` mask matches a slow scan over board stones.
  - Every accepted placement's `WindowUpdate.changed.len()` equals
    `WINDOWS_PER_PLACEMENT`.
  - `StateSnapshot -> load_state` produces the same public state.
  - Terminal snapshots reject any additional post-win placement.
- Tests must be reproducible with fixed seeds.
- These tests should live in Rust where private cache fields can be inspected
  under `#[cfg(test)]`.

### Implementation Notes

- Likely files:
  - `packages/hexo_engine/rust/src/board.rs`
  - `packages/hexo_engine/rust/src/tactics.rs`
  - `packages/hexo_engine/rust/src/state.rs`
- Add test-only helper methods or helper modules instead of making internal
  fields public in production APIs.
- Slow window validation algorithm:
  - For each `WindowEntry` in `state.board().windows().entries()`, iterate the
    six cells from `entry.cells()`.
  - For each player, build an expected mask by checking `Board::get(coord)`.
  - Assert expected masks equal `entry.mask(player)`.
- Legal validation algorithm:
  - Build a set of occupied coords from `occupied_cells()`.
  - Assert every occupied coord has `Board::get(coord).is_some()`.
  - Assert every `Board::get(coord)` entry appears in occupied cells.
  - Assert no legal action decodes to an occupied coord.
  - Optionally recompute legal candidates around occupied coords using
    `coords_within_radius` and compare against `LegalMoveStore`.
- Random game generation:
  - Use a fixed seed and a small number of games, for example 32 games.
  - Cap each game at a reasonable action count, for example 256 actions, to
    avoid very long random tests.
  - Pick from engine-provided legal action IDs only.
  - Stop when terminal.
- Terminal snapshot rejection:
  - Once a game is terminal, append any coordinate to the snapshot placements.
  - `load_state` should reject it with `StateLoadError::IllegalPlacement`
    containing `MoveError::TerminalState`.

### Validation

- New invariant tests pass repeatedly without flakes.
- Existing example tests still pass.
- Run:
  - `cargo fmt`
  - `cargo test -p hexo_engine`
  - `cargo test --workspace`



7.
Updated [Review.md](E:/Hexo-BotTrainer/Review.md) with implementation-ready plans for items 2 through 6 only, including requirements, implementation notes, and validation commands.

For item 1, I would use a two-layer episode design:

**Replay core**
Keep `.hxr` as the runner-defined authoritative replay file. It should stay compact and durable: game id, seed, players, accepted packed action IDs, status, winner/placements, abort metadata. This is the audit log and replay source.

**Training samples**
Add a separate append-only sample file keyed back to `.hxr`, probably one per worker next to the replay file. Each sample row should have a runner-defined core:

```text
record_path
game_id
action_index
player_role
current_player
phase
selected_action_id
legal_action_ids
terminal_value_after_game
```

Then allow player-defined payloads as namespaced extensions:

```text
namespace = "hexo.mcts.policy"
schema_version = 1
payload = {
  visit_policy: [(action_id, visits), ...],
  root_value: float,
  temperature: float,
  search_visits: int,
  model_id: str,
  search_config_id: str
}
```

For CNN/network players, the extension could be:

```text
namespace = "hexo.model.policy"
schema_version = 1
payload = {
  model_id: str,
  value: float,
  logits_ref: ...,
  input_config_id: str
}
```

I would not store arbitrary `DecisionResult.diagnostics` directly. Keep diagnostics for UI/debugging, and add a separate `training_payloads` or `sample_payloads` field/protocol with versioned, validated schemas. That gives you durable replay plus extensible training data without turning `.hxr` into a large, schema-loose catchall.