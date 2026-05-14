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
      py.typed
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
      pybridge.rs
```

`packages/hexo_engine` is the single ownership boundary for engine code. The
Python package is the host-facing API; the Rust code is the rules authority and
performance core. The Python side should not contain a parallel rules
implementation.

The clean bridge choice is PyO3 built with maturin. The Python package exposes
friendly typed functions while `hexo_engine._rust` owns the narrow compiled
boundary into Rust. The bridge should pass opaque state handles and simple
transport types, not reimplement rules in Python.

Current status: the Rust engine skeleton contains real rule/state modules, but
the Python API and PyO3 bridge are still scaffolding. The Python functions
define the intended host-facing surface and currently raise engine-unavailable
errors until the bridge is wired.

## File Responsibilities

| File | Role |
| --- | --- |
| `pyproject.toml` | Python package metadata and maturin bridge settings. |
| `Cargo.toml` | Rust crate metadata for the engine package. |
| `python/hexo_engine/__init__.py` | Public Python export surface for engine API, errors, and transport types. |
| `python/hexo_engine/api.py` | Intended Python API for state creation, legal actions, transitions, snapshots, tactics, and identities. |
| `python/hexo_engine/types.py` | Python dataclasses and aliases for engine-facing transport values. |
| `python/hexo_engine/errors.py` | Python exception types for unavailable engine, illegal actions, and snapshot errors. |
| `python/hexo_engine/py.typed` | Marker that the package ships type information. |
| `rust/src/lib.rs` | Rust crate root and public export surface. |
| `rust/src/board.rs` | Sparse unlimited-board storage and board-level helpers. |
| `rust/src/coord.rs` | Axial coordinate math and distance helpers. |
| `rust/src/rules.rs` | Rule constants and legality helpers. |
| `rust/src/state.rs` | Canonical game state, turn phase, transitions, terminal checks, and tests. |
| `rust/src/tactics.rs` | Threat-window and tactical-summary logic. |
| `rust/src/identity.rs` | Placeholder home for stable state/action identity helpers. |
| `rust/src/snapshot.rs` | Replayable snapshot DTOs and metadata. |
| `rust/src/error.rs` | Rust engine error types. |
| `rust/src/pybridge.rs` | Minimal PyO3 module scaffold for the compiled Python bridge. |

## Game Rules

Hexo is played on an unlimited hex grid. Player 0 opens at `(0, 0)`.
After the opening, players place two stones per turn, one placement at a time.
Each stone must be empty and within 8 hex steps of at least one existing stone.

A player wins immediately by making six connected stones in a straight line
along any of the three hex axes. There is no normal draw rule in the current
design.

A threat is a six-cell window containing at least four stones for one player
and no opponent stones. Because normal turns place two stones, many open
four-in-a-row and five-in-a-row threats can be decisive if they cannot be
blocked with the opponent's two placements.

## Core API

```text
new_game(seed?, scenario?) -> state
load_snapshot(snapshot) -> state
snapshot(state) -> snapshot
current_player(state) -> Player0 | Player1
turn_phase(state) -> Opening | FirstStone | SecondStone
legal_actions(state) -> list[action]
validate_action(state, action) -> ok | legality_error
apply_action(state, action) -> transition_result | legality_error
terminal(state) -> terminal_result | none
tactics(state) -> tactical_summary
state_id(state) -> stable identity
action_id(action) -> stable identity
```

The Rust skeleton names phases `Opening`, `FirstStone`, and `SecondStone`.
The Python skeleton still exposes `turn_placement()` for the normal placement
slot; the final bridge should make the opening phase explicit rather than
pretending every turn has only placement 0 or placement 1.

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
