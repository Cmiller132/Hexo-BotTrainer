# HEXO_RUNNER

## Purpose

`hexo-runner` is the headless execution layer. It creates games, initializes
players, asks the active participant for decisions, applies accepted actions
through the engine, and records what happened.

Its central abstraction is:

```text
participant receives decision request
participant returns action | error
runner applies accepted actions through the engine
```

## Owns

- Session setup from players, seeds, scenarios, and run options.
- Engine state handle ownership for one game session.
- The game loop.
- Player lifecycle.
- Core game recording and event emission.
- Result summaries.
- Self-play, evaluation, direct match, and batch run modes.

## Does Not Own

- Game legality or terminal detection.
- Dashboard-specific game-state shaping or tactical interpretation.
- Model architecture or tensors.
- Search internals.
- Training targets or losses.
- Long-term storage semantics.
- UI rendering.

## Package Layout

```text
packages/hexo_runner/
  pyproject.toml
  python/
    hexo_runner/
      __init__.py
      cli.py
      config.py
      player.py
      session.py
      loop.py
      py.typed
      records/
        __init__.py
        events.py
        record.py
        results.py
      modes/
        __init__.py
        match.py
        selfplay.py
        evaluation.py
        batch.py
```

## Current Status

This document describes the runner boundary. The CLI and non-match modes are
still redesign scaffolds, but match mode now runs a generic player-versus-player
game through the shared loop. The runner owns the primary engine state, builds a
typed decision view for the active player, applies accepted actions through the
public engine API, writes core transition records, and returns a compact
`GameResult`.

## File Responsibilities

| File | Role |
| --- | --- |
| `pyproject.toml` | Python package metadata and `hexo-rl` console entry point. |
| `python/hexo_runner/__init__.py` | Package description and version export. |
| `python/hexo_runner/cli.py` | Placeholder CLI entry point for the runner package. |
| `python/hexo_runner/config.py` | Placeholder runner config dataclass for future session/run options. |
| `python/hexo_runner/player.py` | Shared player identity, decision request/result, and runner-player protocol. |
| `python/hexo_runner/session.py` | Game spec, session context, and engine-backed session creation. |
| `python/hexo_runner/loop.py` | Generic single-game player/engine loop. |
| `python/hexo_runner/py.typed` | Marker that the package ships type information. |
| `records/__init__.py` | Public exports for runner record/result types. |
| `records/events.py` | Live transition/record event shapes. |
| `records/record.py` | Durable detached game-record dataclasses. |
| `records/results.py` | Compact match, batch, evaluation, and self-play result dataclasses. |
| `modes/__init__.py` | Public exports for available runner modes. |
| `modes/match.py` | One-game match mode built on the generic player/engine loop. |
| `modes/batch.py` | Future many-game mode built from match jobs. |
| `modes/evaluation.py` | Future fixed-opponent evaluation mode. |
| `modes/selfplay.py` | Future self-play execution mode using model-owned players and sample writers. |

## Player Contract

All participants implement the same contract:

```text
identity -> player identity
initialize(session_context)
decide(decision_request) -> decision_result
observe_transition(transition_event)
close(final_summary)
```

A player may be model-backed, scripted, human-controlled, remote, search-based,
or random. The runner only depends on the contract, and every player self
reports the same `identity` field.

## Decision Request

A request contains:

- game id and turn index,
- current player,
- typed engine decision view,
- legal actions,
- evaluation flag,
- optional raw tactical/window-store data,
- seed and provenance metadata.

A response contains:

- chosen action,
- optional opaque diagnostics.

Diagnostics are transported by the runner but owned by the player/model that
produced them.

The decision view contains a replay snapshot, stable state id, raw game state,
raw tactical/window-store data, legal actions, terminal status, and a cloned
`EngineStateRef`. The clone is built from `snapshot(primary)` followed by
`load_snapshot(snapshot)`, so players can mutate it for search without touching
the primary game state held by the runner.

## Runtime Flow

```text
create session
initialize players
create or load engine state
store EngineStateRef on SessionContext
while not terminal:
    ask engine for current context
    ask active player for a decision
    handle player error if needed
    submit action to engine
    emit events and core game records
close players
emit final summary
```

## Records And Results

- `events`: live, ephemeral notifications for logging and observers.
- `record`: durable core game records written as the game runs. These contain
  position history, accepted actions, players, seeds, terminal state, and run
  metadata.
- `results`: compact return summaries for match, batch, evaluation, and
  self-play calls.

Records are detached from training. They can be analyzed after a game to
produce derived summaries, but the original game record should remain
append-only and replayable. Model-specific training payloads are written by the
model's self-play sample writer, not into the runner game record.

## Run Modes

- `match`: the public entry point for one game; creates a session and calls the
  shared player/engine loop.
- `batch`: the public entry point for many games; schedules independent match
  configs and aggregates results.
- `evaluation`: builds fixed opponent comparisons on top of batch, marks
  sessions as evaluation runs, and owns score/analysis summaries.
- `selfplay`: receives model-backed `RunnerPlayer`s from the caller, calls
  batch, writes detached core game records, and lets those model-owned players
  maintain their own training-sample writers while games run.

All modes share the same player contract and engine application path.
