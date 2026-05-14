# HEXO_ENGINE

## Purpose

`hexo-engine` is the rules and state authority for Hexo. It defines the
canonical game state, legal transitions, terminal detection, move history, and
the tactical window data maintained by the rules engine.

Every other package consumes engine facts instead of duplicating game logic.

## Owns

- Canonical game state.
- Player0/Player1 to act and current turn placement slot.
- Legal action generation.
- Move validation and state transitions.
- Terminal detection and winner.
- Move history.
- A native `HexoState` handle as the only public Python state object.
- `to_python_state()` conversion for slow inspection, frontend, and debugging.
- Stable action identity for diagnostics.
- Clear errors for illegal, stale, malformed, or incompatible inputs.

## Does Not Own

- Neural network architecture.
- Tensor, graph, or token construction.
- Training targets, losses, optimizers, or checkpoints.
- Runner lifecycle, player orchestration, or run results.
- Model-specific search policy.
- Dashboards, experiment management, or storage backends.
- UI/dashboard tactical interpretation, sorting, filtering, labels, summaries,
  or derived helper facts such as immediate wins and must-block moves.

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
the PyO3 bridge is still scaffolding. The Python functions define the intended
host-facing surface and temporarily provide a small fallback implementation so
callers can use the engine boundary until the Rust API is bound.

The bridge should expose the native engine state as an opaque `HexoState`.
Slow clients can call `to_python_state()` to get a read-only Python mirror of
Rust `HexoState`, including `Board.windows`. Dashboard packages may derive
display-oriented facts from that mirror, but the engine API should not reshape
state into UI summaries.

## File Responsibilities

| File | Role |
| --- | --- |
| `pyproject.toml` | Python package metadata and maturin bridge settings. |
| `Cargo.toml` | Rust crate metadata for the engine package. |
| `python/hexo_engine/__init__.py` | Public Python export surface for engine API, errors, and transport types. |
| `python/hexo_engine/api.py` | Python API for state creation, legal actions, transitions, Python-state conversion, and action identity. |
| `python/hexo_engine/types.py` | Python dataclasses and aliases for engine-facing actions, results, and read-only state mirrors. |
| `python/hexo_engine/errors.py` | Python exception types for unavailable engine and illegal actions. |
| `python/hexo_engine/py.typed` | Marker that the package ships type information. |
| `rust/src/lib.rs` | Rust crate root and public export surface. |
| `rust/src/board.rs` | Sparse unlimited-board storage and board-level helpers. |
| `rust/src/coord.rs` | Axial coordinate math and distance helpers. |
| `rust/src/rules.rs` | Rule constants and legality helpers. |
| `rust/src/state.rs` | Canonical game state, turn phase, transitions, terminal checks, and tests. |
| `rust/src/tactics.rs` | Incremental raw six-cell window tracking and window update data. |
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
current_player(state) -> Player0 | Player1
turn_placement(state) -> Placement0 | Placement1
legal_actions(state) -> list[action]
validate_action(state, action) -> ok | legality_error
apply_action(state, action) -> transition_result | legality_error
terminal(state) -> terminal_result | none
to_python_state(state) -> PythonHexoState
action_id(action) -> stable identity
```

`to_python_state()` exposes the Rust-like phase names `Opening`, `FirstStone`,
and `SecondStone` on the read-only Python mirror.

## Interfaces

To the runner:

- initial or scenario state,
- active player,
- legal action context,
- action validation and application,
- transition result,
- terminal result,
- Python state mirror for clients that need slow inspection/export.

The primary action is a single placement. A pair action may exist as a
host-facing convenience for normal two-placement turns, but the engine boundary
resolves it into deterministic single placements, records the resolved order,
and discards the second placement if the first placement wins.

To models:

- canonical state context,
- legal actions and action identities,
- optional Python state mirror,
- terminal result and history for target construction.

To utilities:

- state and legal action APIs for search, encoding, and test harnesses.

## Tests

Engine tests should focus on rule correctness and determinism:

- legality and illegal move errors,
- turn placement transitions,
- terminal detection,
- replay from action history,
- action identity stability,
- Python state mirror consistency with engine window state.
