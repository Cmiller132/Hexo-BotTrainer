# hexo_runner

Model-agnostic, headless game-execution layer for the Hexo RL project. It owns
the authoritative engine state for a single game, mediates between two opaque
`RunnerPlayer` adapters and the Rust `hexo_engine`, and emits durable `.hxr`
game records plus structured `GameResult` / `BatchResult` summaries. It also
ships the **SealBot adapter** -- a subprocess bridge to an external C++ minimax
baseline bot used as the fixed evaluation opponent across all model packages
and the frontend Match/Arena screen.

## Status

| Part | Usage |
| --- | --- |
| Player contracts (`player.py`), game loop (`loop.py`), match mode, records facade, SealBot adapter | Used by all four model packages, the frontend dashboard, and tests |
| Batch mode (`modes/batch.py`) | Exercised by `tests/test_hexo_runner_match_mode.py`; the model packages run their own multiprocessing self-play with batched GPU inference and import only the records facade |
| Evaluation mode (`modes/evaluation.py`) | Not implemented (`run_evaluation` raises `NotImplementedError`); each model package provides its own SealBot eval harness |
| CLI (`cli.py`, `hexo-rl` script) | `main()` exits with guidance to use the programmatic API; the package surface is Python-first by design |

## Modules

All paths relative to `packages/hexo_runner/python/hexo_runner/`.

| File | Role |
| --- | --- |
| `__init__.py` | Package facade re-exporting player contracts, record types, and specs |
| `player.py` | Core contracts: `PlayerIdentity`, `WorkerContext`, `GameContext`, `DecisionResult`, `TransitionEvent`, `FinalSummary`, and the `RunnerPlayer` / `PlayerFactory` protocols. Implemented by every model package's player adapter and by the frontend's bot wrappers. Lifecycle: `setup_worker -> start_game -> decide / observe_transition* -> finish_game -> close` |
| `loop.py` | `run_match_loop`: single-game synchronous loop. Owns the one authoritative `HexoState`, hands players cloned states, applies actions, writes `.hxr` actions, and stages every player/engine call through `_run_stage` so failures become structured `AbortRecord`s |
| `engine.py` | `HexoEngineAdapter`: the single point where the runner touches the `hexo_engine` public API (`new_game`, `clone_state`, `apply_action`, `terminal`, JSON-able terminal payloads) |
| `session.py` | `GameSpec` (game_id / seed / mode / max_actions; `seed` is persisted in the `.hxr` header and exposed via `GameContext.seed` -- the engine's `new_game` does not consume it; recorded games require `scenario=None`, validated by the loop) and `BatchSpec` dataclasses; `SessionSpec` / `SessionContext` are compatibility aliases |
| `modes/match.py` | `run_match`: one game -> one `{game_id}.hxr` file via `run_match_loop` |
| `modes/batch.py` | `run_batch`: local multiprocessing (spawn pool, round-robin chunk assignment, per-worker `.hxr` file, reusable players via `PlayerFactory`) |
| `modes/evaluation.py` | `run_evaluation` placeholder (raises `NotImplementedError`) |
| `records/record.py` | Re-exports the Rust-backed `.hxr` record types from `hexo_utils.records`; defines the Python `AbortRecord` dataclass for runner abort metadata |
| `records/results.py` | `GameStatus` enum, `GameResult` and `BatchResult` summary dataclasses |
| `records/__init__.py` | Records facade -- the most-imported path of the package; all model self-play/eval imports `AbortRecord` / `HexoRecordFile` / `HexoRecordPlayer` from here |
| `adapters/sealbot.py` | `SealBotPlayer` (a `RunnerPlayer` over the external SealBot minimax), `SealBotConfig` (path via `SEALBOT_PATH` env or `--sealbot-path`, variant, time limit), `_SealBotProcess` (JSON-line subprocess manager with reader threads and timeouts), `discover_sealbot_adapters` (availability metadata for the frontend) |
| `adapters/_sealbot_worker.py` | Standalone subprocess script spawned by `sealbot.py` (overridable via `SealBotConfig.worker_script` for tests): imports the SealBot checkout's `game.py` + one variant's compiled `minimax_cpp` pybind extension, rebuilds the game from the JSON state payload, returns moves + diagnostics over stdout JSON lines |
| `timing.py` | `Timer` (perf_counter ms helper) used by loop and batch mode |
| `cli.py` | `hexo-rl` console entry point; exits with guidance to use the programmatic API |
| `config.py` | Compatibility re-export of the spec dataclasses (`RunnerConfig = BatchSpec`) |

## Design notes

- **One authoritative state.** `run_match_loop` owns the only mutable
  `HexoState`. Players receive a fresh clone for every `decide` and
  `observe_transition`, so player code cannot mutate the official game. Seat
  order is fixed: `players[0]` is `player0` and moves first.
- **Staged failure handling.** Every player/engine/record call runs through
  `_run_stage(name, fn)`; any exception becomes an `AbortRecord` (stage,
  exception type, message) persisted in the `.hxr` record and the game ends
  ABORTED instead of raising. The record entry is always finalized;
  `finish_game`/`close` errors never change a decided result. Stage names
  persist verbatim in records, so keep them stable for abort triage.
- **One subprocess per SealBot variant.** The two variants (`current`/`best`)
  export identical pybind module names, so each `SealBotPlayer` lazily spawns
  a dedicated `_sealbot_worker.py` subprocess on the first `decide`. The
  protocol is strictly request-response JSON lines over stdin/stdout: a ready
  handshake at startup, then one `{"type": "decide", "state": ...}` request
  in flight at a time, answered by exactly one `{ok, moves, diagnostics}`
  line; `{"type": "close"}` ends the loop. Worker stdout carries protocol
  JSON only; stderr is tailed into a bounded buffer for error reporting.
  Timeouts/worker exits surface as `SealBotUnavailableError`/`TimeoutError`.
- **Two-stone turn buffering.** SealBot answers with its full turn (1-2
  moves) while the runner requests one action at a time, so `SealBotPlayer`
  buffers the second stone and replays it on the next `decide` (tagged
  `diagnostics["buffered_move"]`). Moves are legality-checked before being
  returned; an empty or illegal response raises, aborting via the loop.
- **State payload contract.** `_state_payload` sends the worker the current
  player, phase, `moves_left_in_turn` (from `TurnPhase`: opening or second
  stone -> 1, otherwise 2), placements made, terminal winner, and the stone
  list; runner `player0`/`player1` map to SealBot `Player.A`/`B`.

## Connections to other packages

Imports out:

- **hexo_engine** -- `engine.py` wraps `new_game` / `clone_state` /
  `apply_action` / `terminal`; `adapters/sealbot.py` uses `to_python_state`,
  `PlacementAction`, `is_legal_action`, `TurnPhase` to translate engine states
  into the SealBot JSON payload.
- **hexo_utils** -- `records/record.py` re-exports the Rust-backed `.hxr`
  binary codec (`HexoRecordFile`, `HexoRecordGameWriter`, `HexoRecordPlayer`,
  magic/schema constants). This re-export is the path through which all
  production `.hxr` IO flows.

Imports in (who depends on hexo_runner):

- **hexfield / hexfield_eq** (active lineages): the eval path
  (`eval_driver.py` / `evaluation.py` / `eval_arena.py`) writes `.hxr` via
  `hexo_runner.records` and uses `SealBotPlayer` as an eval opponent, driving
  games with their own batched loop.
- **dense_cnn_restnet**, **hexo_models/dense_cnn**, **hexo_models/hexgt**
  (parked/halted lineages): same pattern; hexgt
  evaluation additionally calls `run_match` + `GameSpec` for SealBot gating.
  Each package's `player.py` implements the `RunnerPlayer` protocol.
- **hexo_frontend**: `web.py` imports the SealBot adapter, `run_match`, the
  player contracts, `GameResult`/`HexoRecordFile`, and `GameSpec` -- Match-v2
  Arena games run through this runner; `/api` serves adapter discovery.

Protocols / shared formats owned or relayed here:

- `RunnerPlayer` protocol (`player.py`) -- the cross-package player contract.
- `.hxr` game-record format (defined in `hexo_utils`, consumed through
  `hexo_runner.records` by every writer and the dashboard reader).
- SealBot subprocess protocol (see Design notes); the worker imports the
  external checkout at `$SEALBOT_PATH` (repo-external, typically `E:\SealBot`
  / `/mnt/e/SealBot`) with per-variant dirs `current`/`best`.

## Entry points / how it gets exercised

- Programmatic API: `hexo_runner.modes.run_match(spec, players, output_dir)`
  and `run_batch(BatchSpec)`; lower-level `hexo_runner.loop.run_match_loop`.
- Frontend HTTP, indirectly: `hexo_frontend` Arena/match endpoints construct
  `GameSpec` + players and call `run_match`.
- Subprocess: `python _sealbot_worker.py --root --variant --time-limit`,
  spawned by `_SealBotProcess` (not run by hand).
- `hexo-rl` console script -- exits with guidance to use the programmatic API.
- Tests: `tests/test_hexo_runner_match_mode.py` (loop, match, batch, abort)
  and `tests/test_sealbot_adapter.py` (discovery, buffering, worker-script
  override), run under the WSL venv.
