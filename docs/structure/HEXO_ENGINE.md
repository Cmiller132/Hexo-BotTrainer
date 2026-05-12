# HEXO_ENGINE

## Purpose

`hexo-engine` is the rules and state authority for Hexo. It defines the
canonical game state, legal transitions, terminal detection, replayable history,
and rules-derived tactical facts.

Every other package consumes engine facts instead of duplicating game logic.

## Owns

- Canonical game state.
- Player0/Player1 to act and current turn placement slot.
- Legal action generation.
- Move validation and state transitions.
- Terminal detection and winner.
- Move history and replayable snapshots.
- Rules-derived tactical facts, such as threats or immediate wins.
- Stable state/action identity for caches, replay checks, search tables, and
  diagnostics.
- Clear errors for illegal, stale, malformed, or incompatible inputs.

## Does Not Own

- Neural network architecture.
- Tensor, graph, or token construction.
- Training targets, losses, optimizers, or checkpoints.
- Runner lifecycle, player orchestration, or run results.
- Model-specific search policy.
- Dashboards, experiment management, or storage backends.

## Package Layout

```text
packages/hexo_engine/
  pyproject.toml
  Cargo.toml
  python/
    hexo_engine/
      __init__.py
      api.py
      types.py
      errors.py
  rust/
    src/
      lib.rs
      board.rs
      coord.rs
      rules.rs
      state.rs
      tactics.rs
      identity.rs
      snapshot.rs
      error.rs
```

`packages/hexo_engine` is the single ownership boundary for engine code. The
Python package is the host-facing API; the Rust code is the rules authority and
performance core. The Python side should not contain a parallel rules
implementation.

## Core API

```text
new_game() -> state
load_snapshot(snapshot) -> state
snapshot(state) -> snapshot
current_player(state) -> Player0 | Player1
turn_placement(state) -> PLACEMENT_0 | PLACEMENT_1
legal_actions(state) -> list[action]
validate_action(state, action) -> ok | legality_error
apply_action(state, action) -> transition_result | legality_error
terminal(state) -> terminal_result | none
tactics(state) -> tactical_summary
state_id(state) -> stable identity
action_id(action) -> stable identity
```

## Interfaces

To the runner:

- initial or scenario state,
- active player,
- legal action context,
- action validation and application,
- transition result,
- terminal result,
- replayable history and state snapshots.

The primary action is a single placement. A pair action may exist as a
host-facing convenience for normal two-placement turns, but the engine boundary
resolves it into deterministic single placements, records the resolved order,
and discards the second placement if the first placement wins.

To models:

- canonical state context,
- legal actions and action identities,
- optional tactical summaries,
- terminal result and replay history for target construction.

To utilities:

- state and legal action APIs for search, encoding, replay validation, and
  test harnesses.

## Tests

Engine tests should focus on rule correctness and determinism:

- legality and illegal move errors,
- turn placement transitions,
- terminal detection,
- snapshot round trips,
- replay from action history,
- state/action identity stability,
- tactical payload consistency with legal actions.
