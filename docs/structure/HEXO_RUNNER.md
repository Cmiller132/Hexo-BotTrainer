# HEXO_RUNNER

## Purpose

`hexo-runner` is the headless execution layer. It creates games, initializes
players, asks the active participant for decisions, applies accepted actions
through the engine, and records what happened.

Its central abstraction is:

```text
participant receives decision request
participant returns action | refusal | error
runner applies accepted actions through the engine
```

## Owns

- Session setup from players, seeds, scenarios, and run options.
- Engine state handle ownership for one game session.
- The game loop.
- Player lifecycle.
- Core game recording and event emission.
- Result summaries.
- Self-play, evaluation, direct match, smoke test, and batch run modes.

## Does Not Own

- Game legality or terminal detection.
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
      config.py
      player.py
      session.py
      loop.py
      records/
        __init__.py
        events.py
        record.py
        results.py
      modes/
        match.py
        selfplay.py
        evaluation.py
        batch.py
```

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
- engine snapshot or state reference,
- legal actions,
- evaluation flag,
- optional tactical summary,
- seed and provenance metadata.

A response contains:

- chosen action, or a controlled failure,
- optional opaque diagnostics.

Diagnostics are transported by the runner but owned by the player/model that
produced them.

## Runtime Flow

```text
create session
initialize players
create or load engine state
store EngineStateRef on SessionContext
while not terminal:
    ask engine for current context
    ask active player for a decision
    handle player refusal or error if needed
    submit action to engine
    emit events and replay records
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

Records can be analyzed after a game to produce derived summaries, but the
original game record should remain append-only and replayable. Model-specific
training payloads belong in replay extensions, not in the core game record.

## Run Modes

- `match`: the public entry point for one game; creates a session and calls the
  shared loop.
- `batch`: the public entry point for many games; schedules independent match
  configs and aggregates results.
- `evaluation`: builds fixed opponent comparisons on top of batch, marks
  sessions as evaluation runs, and owns score/analysis summaries.
- `selfplay`: creates model-backed players through an `InferenceAdapter`, calls
  batch, and writes records for training to consume.

All modes share the same player contract and engine application path.
