# hexo_frontend

Web dashboard for the Hexo RL training project. A single stdlib
`ThreadingHTTPServer` (no Flask, no framework -- older notes in HANDOFF.md
saying "Flask" are stale) serves one static SPA bundle with three screens:

- **Match** (`#match`) -- live Arena: manual / SealBot / checkpoint bots play
  real games through `hexo_runner`.
- **History** (`#history`) -- training-run status: live status band (2.5s
  poll), per-epoch trends/table/inspector, paged game history.
- **Debug** (`#debug?run=..&ply=..`) -- single-position model forensics
  (policy/value heads, MCTS search, training-row targets, per-ply sweeps),
  backed by a CPU-only torch worker subprocess so the HTTP server itself never
  imports torch or touches the training GPU.

**Status: ACTIVE, under heavy development.** The production instance runs in
WSL on :8080 (launched by `scripts/_dashboard_launch.sh`, cwd at the run-mount
root `/mnt/e/Hexo-BotTrainer`, NOT this repo). The working tree currently
carries large uncommitted Match-v2 / History-v2 changes; the live :8080
instance may be serving the pre-rewrite build until restarted.

## Modules

| File | Role |
| --- | --- |
| `python/hexo_frontend/web.py` | ~4.2k-line core: HTTP route table (`do_GET`/`do_POST`), `ManualMatchController` (threaded live match/series bridge between browser clicks and `hexo_runner` players, incl. checkpoint bots played via the debug worker), training-run scanning/caching for `/api/training/*`, and glue for the `/api/debug/*` endpoints. Entry points `run()` / `main()`, default `127.0.0.1:8765`. |
| `python/hexo_frontend/dashboard.py` | Pure shaping layer: `PythonHexoState` mirror -> browser JSON payload (placements, legal moves, winner, window-tactics block). Called only from web.py. |
| `python/hexo_frontend/debug_service.py` | Server-side manager for the Debug worker: lazily spawns `debug_worker` as a child process, serializes requests behind a lock (NDJSON, timeouts, auto-restart on transport failure, LRU result cache). Module singleton via `get_worker()`. |
| `python/hexo_frontend/debug_worker.py` | Child-process main loop: one JSON request per stdin line, dispatches ops (`ping`/`info`/`analyze`/`search`/`search_tree`/`record_row`/`game_eval`/`reeval`) to debug_infer, one JSON response per stdout line. 3-model LRU checkpoint cache; Windows->WSL path translation. Run as `python -m hexo_frontend.debug_worker` -- never by hand. |
| `python/hexo_frontend/debug_infer.py` | CPU-only, lineage-aware inference library (~1.3k lines): detects checkpoint lineage (hexgt graph / dense_cnn_restnet / plain dense_cnn), rebuilds the network from state-dict + run manifest, replays action-id sequences into engine states, returns a uniform debug schema (priors, distributional value, aux heads, fresh MCTS, pure-Python PUCT debug tree, .npz training-row decode, per-ply game-eval sweeps). Lazy per-lineage imports; only importer is debug_worker (+ tests). |
| `python/hexo_frontend/static/app.js` | ~8.5k-line single-file SPA holding all three screens (`mt*` Match, `hist*` History, `dbg*` Debug prefixes) plus the top diag/error bar and `APP_VERSION`. |
| `python/hexo_frontend/static/index.html` | Single page hosting all three screens; references `styles.css` and `app.js` with the `?v=` cache-bust token. |
| `python/hexo_frontend/static/styles.css` | ~3.2k-line dark-theme stylesheet for all three screens. |
| `python/hexo_frontend/__init__.py` | Version stub only. |
| `pyproject.toml` | Deps `hexo-engine` + `hexo-runner`; registers the `hexo-play` console script -> `hexo_frontend.web:main`. |

## Connections to other packages

**Imports out:**

- `hexo_runner` -- the Match screen plays real games through the production
  runner: `SealBotPlayer`/`SealBotConfig`/`discover_sealbot_adapters`
  (adapters.sealbot), `run_match` (modes.match), the player protocol types
  (`DecisionResult`, `PlayerIdentity`, `WorkerContext`, ...),
  `GameResult`/`HexoRecordFile` (records), `GameSpec` (session).
- `hexo_engine` -- web.py and debug_infer use `HexoState`, `PlacementAction`,
  `is_legal_action`, `to_python_state`, and `pack_coord_id`/`unpack_coord_id`.
  app.js re-implements the same coord<->action-id packing client-side
  (`DBG_COORD_OFFSET` 32768) -- a pinned cross-language contract.
- Model packages (lazily, inside the debug worker only):
  `dense_cnn_restnet.*`, `hexo_models.dense_cnn.*`, `hexo_models.hexgt.*` --
  selected by sniffing the checkpoint payload (`ck["model"]` /
  `"model_state"` / `"arch"` shapes).

**Imports in:** none -- no package imports hexo_frontend. It is exercised by
the browser, by tests, and by supervisor scripts that put
`packages/hexo_frontend/python` on `PYTHONPATH`.

**Shared on-disk formats (read-only, from cwd/`runs` and
`HEXO_DEBUG_RUN_ROOT`):** produced by hexo_train + the model packages:

- `manifest.json` (lineage + architecture), `diagnostics/*.json`
  (`dense_cnn.selfplay.live.json`, `dense_cnn.selfplay.epoch_*.json`,
  `dense_cnn.evaluation.epoch_*.json`), `events.jsonl` tails (live status).
- `checkpoints/*.pt` (epoch filename regex `epoch_?NNN.pt`).
- `.hxr` game records via `hexo_runner.records.HexoRecordFile`.
- Self-play per-game `.npz` training-row shards (Targets tab decode).

**Protocols:**

- debug_service <-> debug_worker: newline-delimited JSON over stdin/stdout,
  `{id, op, ...}` -> `{id, ok, result|error}`. stdout must carry ONLY protocol
  JSON (diagnostics go to stderr). On win32 the child runs under WSL
  (`wsl.exe bash -lc`, venv default `/root/.venvs/hexgt-build/bin/python`);
  both sides translate `E:\...` <-> `/mnt/e/...` paths. Env overrides:
  `HEXO_DEBUG_WORKER_CMD`, `HEXO_DEBUG_WSL_PYTHON`, `HEXO_DEBUG_USE_WSL`,
  `HEXO_DEBUG_RUN_ROOT`.
- Error taxonomy: `DebugWorkerError`/timeout = 500 (retryable, transport);
  `DebugRequestError` = 400 (deterministic, bad request).

## Entry points / how it gets exercised

- `python -m hexo_frontend.web --host --port --sealbot-path` (default
  `127.0.0.1:8765`). Production: `scripts/_dashboard_launch.sh` launches it
  detached in WSL on :8080 behind a netsh portproxy, cwd = run mount, with
  `--sealbot-path /mnt/e/SealBot`.
- `hexo-play` console script (same `main()`).
- Browser: `/` serves index.html; screens are addressed by URL hash
  (`#match` / `#history` / `#debug?...` -- the debug hash is a full
  deep-link/back-forward navigation state).
- HTTP routes (verified against web.py): GET `/api/state`, `/api/adapters`,
  `/api/training/{runs,run,live,epoch,history-page,history-count,
  artifacts-page,file,history}`, `/api/debug/{checkpoints,games,trajectory,
  position,ckpt_info,record_row,game_eval}`; POST `/api/new`,
  `/api/match/stop`, `/api/move`, `/api/debug/{analyze,search,search_tree}`.
- Tests (via `tests/conftest.py` sys.path injection; authoritative only in the
  WSL `hexgt-build` venv per project convention):
  `tests/test_hexo_runner_match_mode.py`, `tests/test_sealbot_adapter.py`
  (ManualMatchController / player-spec normalization),
  `tests/test_frontend_training_{artifacts,epoch,live}.py` (training scan
  functions called directly), `tests/test_debug_infer.py` (debug_service +
  debug_infer + web debug endpoints).

## Gotchas

- **Cache-bust token in three places.** index.html is served `no-store` and
  `app.js`/`styles.css` are served `no-cache` + ETag (web.py `_send_static`;
  the old 300s max-age policy is gone — `STATIC_MAX_AGE_SECONDS` is now
  unused). The `?v=` token (currently `20260611-match1`) and `APP_VERSION` in
  app.js should still be bumped in lockstep on every static change: they
  defeat intermediary caches and keep the on-screen version tag honest.
- **The debug worker is the only torch path.** web.py never imports torch;
  everything model-related rides the subprocess. If the worker venv path
  (`/root/.venvs/hexgt-build/bin/python`, hardcoded default in
  debug_service.py) is wrong for your setup, override via env vars above.
- **Run scanning is cwd-relative.** Training endpoints scan `runs/` under the
  server's cwd; production deliberately runs with cwd at the run mount, not
  the repo. Debug endpoints additionally honor `HEXO_DEBUG_RUN_ROOT`.
- **Known-wrong fallback:** the `.npz` record-index fallback in the debug
  record resolution path (`_debug_resolve_record_npz`, web.py) is provably
  wrong when game_id matching fails -- it degrades silently rather than
  reporting a miss.
- **Lineage sniffing is heuristic.** debug_infer classifies checkpoints by
  payload shape; a plugin-saved hexgnn checkpoint would be misclassified as
  dense_cnn_restnet (only the hexgnn RL driver's `{model, arch}` format maps
  to the graph lineage).
- **Spec drift:** `docs/specs/history_screen_v2_spec.md` still lists helpers
  (`renderTraining`, `trainingArtifactRow`, `loadMoreArtifacts`, ...) that the
  Match-v2 rewrite removed or orphaned; do not treat its "DO NOT TOUCH" list
  as current. `GET /api/training/artifacts-page` now has no client caller
  (tests only), and `GET /api/training/file` is a manual raw-download URL with
  no client or test coverage.
- **Always-on diag bar.** The top diag/error bar and tap-echo overlays in
  app.js are permanent instrumentation from the phone-debug episode, present
  on every screen.
- **Uncommitted state.** web.py, app.js, index.html, styles.css carry
  substantial uncommitted Match-v2 + History-v2 work; this README describes
  the working-tree state.
