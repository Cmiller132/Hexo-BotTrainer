# hexo_runner

Model-agnostic, headless game-execution layer for the Hexo RL project. It owns
the authoritative engine state for a single game, mediates between two opaque
`RunnerPlayer` adapters and the Rust `hexo_engine`, and emits durable `.hxr`
game records plus structured `GameResult` / `BatchResult` summaries. It also
ships the **SealBot adapter** -- a subprocess bridge to an external C++ minimax
baseline bot used as the fixed evaluation opponent across all model packages
and the frontend Match/Arena screen.

## Status

| Part | Status |
| --- | --- |
| Player contracts (`player.py`), game loop (`loop.py`), match mode, records facade, SealBot adapter | **Active** -- used by all four model packages, the frontend dashboard, and tests |
| Batch mode (`modes/batch.py`) | Legacy-but-referenced: exercised only by `tests/test_hexo_runner_match_mode.py`; every model package reimplements its own multiprocessing self-play with batched GPU inference |
| Evaluation mode (`modes/evaluation.py`) | Stub -- `run_evaluation` unconditionally raises `NotImplementedError`; each model package built its own SealBot eval harness instead |
| CLI (`cli.py`, `hexo-rl` script) | Placeholder -- `main()` raises `SystemExit` directing callers to the programmatic API |

Note: `pyproject.toml` still describes the package as a "Placeholder" -- it is
in fact load-bearing across the repo.

## Modules

All paths relative to `packages/hexo_runner/python/hexo_runner/`.

| File | Role |
| --- | --- |
| `__init__.py` | Package facade re-exporting player contracts, record types, and specs |
| `player.py` | Core contracts: `PlayerIdentity`, `WorkerContext`, `GameContext`, `DecisionResult`, `TransitionEvent`, `FinalSummary`, and the `RunnerPlayer` / `PlayerFactory` protocols. Implemented by every model package's player adapter and by the frontend's bot wrappers |
| `loop.py` | `run_match_loop`: single-game synchronous loop. Owns the one authoritative `HexoState`, hands players cloned states, applies actions, writes `.hxr` actions, and stages every player/engine call through `_run_stage` so failures become structured `AbortRecord`s |
| `engine.py` | `HexoEngineAdapter`: thin centralizing wrapper over the `hexo_engine` public API (`new_game`, `clone_state`, `apply_action`, `terminal`, JSON-able terminal payloads) |
| `session.py` | `GameSpec` (game_id/seed/mode/max_actions; `scenario` must be `None`) and `BatchSpec` dataclasses; `SessionSpec`/`SessionContext` are unused legacy aliases |
| `modes/match.py` | `run_match`: one game -> one `{game_id}.hxr` file via `run_match_loop` |
| `modes/batch.py` | `run_batch`: local multiprocessing (spawn pool, round-robin chunk assignment, per-worker `.hxr` file, reusable players via `PlayerFactory`). Test-only in practice |
| `modes/evaluation.py` | `run_evaluation` stub (raises `NotImplementedError`) |
| `records/record.py` | Re-exports the Rust-backed `.hxr` record types from `hexo_utils.records`; defines the Python `AbortRecord` dataclass for runner abort metadata |
| `records/results.py` | `GameStatus` enum, `GameResult` and `BatchResult` summary dataclasses |
| `records/__init__.py` | Records facade -- the most-imported path of the package; all model self-play/eval imports `AbortRecord` / `HexoRecordFile` / `HexoRecordPlayer` from here |
| `adapters/sealbot.py` | `SealBotPlayer` (a `RunnerPlayer` over the external SealBot minimax), `SealBotConfig` (path via `SEALBOT_PATH` env or `--sealbot-path`, variant, time limit), `_SealBotProcess` (JSON-line subprocess manager with reader threads and timeouts), `discover_sealbot_adapters` (availability metadata for the frontend). Handles move buffering for two-stone turns and illegal-move validation |
| `adapters/_sealbot_worker.py` | Standalone subprocess script spawned by `sealbot.py` (overridable via `SealBotConfig.worker_script` for tests): imports one SealBot variant's `game.py` + compiled `minimax_cpp` pybind extension (the variants share module names, so they cannot coexist in one process), rebuilds the game from the JSON state payload, returns moves + diagnostics over stdout JSON lines |
| `timing.py` | `Timer` (perf_counter ms helper) used by loop and batch mode |
| `cli.py` | Placeholder `hexo-rl` console entry point |
| `config.py` | `RunnerConfig = BatchSpec` alias shim; no external importers |

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

- **dense_cnn_restnet** (active lineage): `selfplay.py` and `evaluation.py`
  write `.hxr` records via `hexo_runner.records` and use `SealBotPlayer` as
  the per-epoch eval opponent (driving games with their own batched-inference
  loop rather than `run_match`).
- **hexo_models/dense_cnn**, **hexo_models/hexgt**, **hexgnn** (legacy/parked
  lineages): same pattern; hexgt and hexgnn evaluation additionally call
  `run_match` + `GameSpec` for SealBot gating. Each package's `player.py`
  implements the `RunnerPlayer` protocol.
- **hexo_frontend**: `web.py` imports the SealBot adapter, `run_match`, the
  player contracts, `GameResult` / `HexoRecordFile`, and `GameSpec` -- the
  Match-v2 Arena screen plays live games through this runner, and the
  `/api` adapters endpoint serves `discover_sealbot_adapters` output.
- **scripts/goal_benchmark.py** imports `HexoEngineAdapter` and
  `run_match_loop` directly.

Protocols / shared formats owned or relayed here:

- `RunnerPlayer` protocol (`player.py`) -- the cross-package player contract.
- `.hxr` game-record format (defined in `hexo_utils`, consumed through
  `hexo_runner.records` by every writer and the dashboard reader).
- SealBot subprocess protocol: JSON lines over stdin/stdout between
  `_SealBotProcess` and `_sealbot_worker.py` (`{type: "decide", state}` ->
  `{ok, moves, diagnostics}`; ready handshake; close). The worker imports the
  external SealBot checkout at `$SEALBOT_PATH` (repo-external, typically
  `E:\SealBot` / `/mnt/e/SealBot`), with per-variant dirs `current`/`best`.

## Entry points / how it gets exercised

- Programmatic API: `hexo_runner.modes.run_match(spec, players, output_dir)`
  and `run_batch(BatchSpec)`; lower-level `hexo_runner.loop.run_match_loop`.
- Frontend HTTP, indirectly: `hexo_frontend` Arena/match endpoints construct
  `GameSpec` + players and call `run_match`.
- Subprocess: `python _sealbot_worker.py --root --variant --time-limit`,
  spawned by `_SealBotProcess` (never run by hand).
- `hexo-rl` console script -- registered but a pure error-message placeholder.
- Tests: `tests/test_hexo_runner_match_mode.py` (loop, match, batch, abort
  paths) and `tests/test_sealbot_adapter.py` (discovery, move buffering,
  worker-script override). Tests are authoritative in the WSL venv.

## Gotchas

- **Batch/evaluation modes never became the shared orchestration layer** their
  docstrings describe. Four near-identical SealBot eval harnesses exist
  downstream (one per model package); only `records/` and the player protocol
  are the truly shared surface.
- `adapters/sealbot.py` `_moves_left_in_turn` hardcodes turn-phase semantics
  (OPENING/SECOND_STONE -> 1 move, else 2), duplicating engine rules in the
  adapter; a rules change in `hexo_engine` would silently desync the SealBot
  state payload.
- `GameSpec.scenario` is vestigial: `run_match_loop` raises if it is non-None.
- The two SealBot variants cannot be imported into one process (shared module
  names) -- hence the one-subprocess-per-variant design.
- `_SealBotProcess` has no request-id correlation on its response queue; the
  strictly request-response protocol makes this safe today, but a stray extra
  stdout line from the worker would mis-pair responses silently.
- `loop.py` contains a dead `result = GameResult(...)` assignment that is
  unconditionally overwritten later in the function.
- `tests/test_hexo_runner_match_mode.py` and `tests/test_sealbot_adapter.py`
  currently carry uncommitted modifications alongside the frontend
  Match-screen-v2 work; keep them in sync when touching the adapter.
