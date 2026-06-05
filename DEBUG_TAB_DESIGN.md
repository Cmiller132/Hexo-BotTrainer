# Dashboard DEBUG tab — design

_Phase-0 design doc. Goal: a feature-complete Debug tab letting the owner open
any position/game (from the Match/History tabs or any epoch's self-play games)
and inspect what the **hexgt** model thinks — policy prior, search, value (full
65-bin distribution), opponent-policy head, STV lookahead heads, threats — and
run **fresh CPU MCTS** on demand against any checkpoint. Must NOT disturb the
live `hexgt_rl_main3` RL run or contend for the GPU._

## Hard constraints (from the run topology)

- A live RL run owns the GPU. **All debug inference is CPU-only**, isolated in a
  dedicated **worker subprocess** launched with `CUDA_VISIBLE_DEVICES=""`. The
  dashboard HTTP process never imports torch, so live-status polling is never
  blocked and the GPU is never touched.
- Dashboard server: stdlib `ThreadingHTTPServer` (`hexo_frontend/web.py`), system
  Python `C:\Python314` (has torch 2.10 cpu-capable + numpy), launched with a
  PYTHONPATH pointing at the worktree packages. `_training_roots()` is cwd-derived
  (`cwd/runs`). Run dir: `runs/hexgt_rl_main3/`.
- Commits only from the `C:\Hexo-consolidate2` clone (branch
  `chore/hexgt-consolidation`). Never commit from this worktree.

## What recorded data exists per game/position

- **`selfplay/epoch_NNNNNN.hxr`** — full game records (256 games/epoch). Stores
  the **move sequence** (`action_ids`) + winner + length. No per-move model
  outputs. Replayable to any ply via `engine.apply_action` (the existing
  `_training_history` endpoint already does exactly this → board state JSON).
- **`selfplay/epoch_NNNNNN_game_MMMMMM.npz`** — compact training shards. Per
  searched position: **MCTS visit policy** (`pol_act`,`pol_w`), opponent-policy
  target, value target, STV targets, board stones, legal ids. **Raw network
  priors are intentionally dropped at write time** — so priors must be recomputed.
- **`eval/epoch_NNNNNN_examples.json`** — rich per-move trace for a few example
  games: `root_value`, `visits`, `visit_entropy`, `prior_entropy`,
  `top_visit_fraction`, `temperature`, `candidates`.
- **`checkpoints/hexgt_rl_epochNNNNNN.pt`** (+ `hexgt_rl_latest.pt`) — 20 epochs.
  Payload keys: `model` (state_dict), `arch` (token_dim/gnn_layers/ctx_layers/
  ffn_dim/attention_heads/value_pma_seeds/short_term_value_horizons),
  `optimizer`, `rl_epoch`, `step`, `feature_schema_version`. **STV graft at
  epoch 7**: epochs 0–6 are pre-graft (29 MB, STV/value heads SIDE-only width);
  7+ post-graft (31 MB, `[SIDE|PMA]` width). The debug loader must mirror the
  driver's resume recipe (build with STV horizons on → `expand_value_readout_columns`
  → `expand_stv_readout_columns` → `load_state_dict(strict=False)` →
  `zero_init_expanded_feature_columns` if `feature_schema_version` is behind).

Because priors aren't recorded, the **canonical source of model outputs is fresh
CPU inference** — which is also better (any checkpoint, any position). Recorded
data is used for the value-trajectory overlay and recorded-vs-fresh comparison.

## Model interfaces (verified)

- `engine` replay: `new_game(seed)`, `apply_action(state, PlacementAction(unpack_coord_id(aid)))`,
  `legal_action_ids`, `current_player`, `to_python_state`, `dashboard_state`.
- Featurize + all heads: `batch_from_states([state], n=3)` →
  `model.forward(batch)` → `{policy:(C,), value:(G,65), opp_policy:(C,),
  stvalue_4/12/24:(G,65)}`; `batch["candidate_ids"]` maps policy index → action_id.
  Scalar value: `decode_binned_value(value)` (softmax · linspace(-1,1,65)).
- Search: `new_mcts_session(n=3).run([key],[state], HexgtInference(model,
  device="cpu", fp16=False), visits=…, c_puct=1.5, …)` → `SearchResult` with
  `visit_policy`, `root_prior_policy`, `root_value`, `visits`.

## Architecture: data flow

```
browser (Debug tab)
   │  fetch /api/debug/*
   ▼
HTTP server (web.py, ThreadingHTTPServer, NO torch)
   │  • reconstructs board at ply N from .hxr (reuse _training_history logic)
   │  • reads recorded npz visits / examples.json
   │  • request queue + LRU result cache keyed (checkpoint, position-hash, mode)
   │  • forwards inference requests as line-delimited JSON
   ▼
debug worker subprocess  (sys.executable -m hexo_frontend.debug_worker,
   │                       env CUDA_VISIBLE_DEVICES="", single-threaded)
   │  • LRU cache of loaded models per checkpoint (graft-aware loader)
   │  • forward (all heads) | mcts search | trajectory re-eval
   ▼  JSON results (per-candidate priors, value dist, opp, stv, visits)
```

The worker is single-threaded → requests serialize naturally (the "small request
queue"). The server caches results so re-opening the same (checkpoint, position)
is instant and never re-hits the worker. The worker is spawned lazily on first
debug request and reaped on server shutdown; if it dies, the server restarts it.

## Endpoints (added to web.py)

- `GET /api/debug/checkpoints?run=` → `[{epoch, name, path, size, graft:"pre|post", latest:bool}]` + worker status.
- `GET /api/debug/games?run=&source=selfplay|evaluation&epoch=` → list of `.hxr` files + per-file record summaries (game_id, length, winner) for the picker.
- `GET /api/debug/position?run=&path=&record=&ply=` → board state at ply N (reuses replay), legal moves, last move, recorded per-move data (root_value/visits from npz+examples if present), plus the full move list for the scrubber.
- `POST /api/debug/analyze` `{run, action_ids:[…prefix], checkpoint, both_perspectives?}` → policy priors (per candidate action_id), value scalar+65-bin dist, opp-policy, STV heads, and (optional) the same from the opponent's perspective. Cached.
- `POST /api/debug/search` `{run, action_ids, checkpoint, visits=512, c_puct}` → fresh MCTS visit distribution + root prior + root_value + visits actually run. Cached by (checkpoint, position, visits).
- `GET /api/debug/trajectory?run=&path=&record=&checkpoint=` → per-ply recorded root_value (from examples/npz) + the checkpoint's re-evaluated scalar value (lazy, cached) for the value-trajectory chart.

All keep the existing `_send_json` (ETag/gzip) discipline and `_resolve_run_path`
safety. None touch the live-status `/api/state` long-poll.

## UI layout (third `<main id="debugScreen">`, nav button `data-screen="debug"`)

Two-column, matching `.top-grid` / `.panel-card` conventions and the dark theme:

- **Left — board**: self-contained SVG board (reuses pure geometry helpers
  `center`/`path`/`playerColor`/`HEX`), with toggleable overlays:
  policy-prior heatmap · search-visit heatmap · opponent-policy heatmap ·
  threat windows (≥4-count) · last move + move numbers. A ply scrubber/navigation
  bar (start/prev/next/end + slider) like the match replay bar.
- **Right — panels**:
  1. *Source*: run + game picker (selfplay epoch / eval / deep-linked game), checkpoint selector (default latest) with optional A/B second checkpoint, "Open in Debug" deep-link target.
  2. *Value*: scalar value, 65-bin distribution bar chart, STV 4/12/24 readouts, both-perspectives (optimism probe) readout.
  3. *Top-K moves*: candidate cells ranked by prior, with visits (recorded + fresh), sampled-vs-best at recorded temperature.
  4. *Search-on-demand*: visits input (default 512) + Run button; shows fresh visits vs recorded vs raw prior.
  5. *Value-trajectory chart*: recorded vs re-evaluated value across the whole game; flips/divergences visible.
  6. *Import*: paste move list / pick from game (freeform editor only if cheap).

## Milestones (each independently shippable + testable)

- **M1** — Worker subprocess + graft-aware CPU loader; `/api/debug/checkpoints`,
  `/api/debug/analyze`; Debug tab shell with board + policy/value panel for a
  pasted/loaded position. pytest for the loader (pre & post graft) + endpoint.
- **M2** — Game loading + ply scrubber: `/api/debug/games`, `/api/debug/position`;
  navigate any self-play/eval game; deep-link from History ("Open in Debug").
- **M3** — `/api/debug/search` fresh CPU MCTS + Top-K table (fresh vs recorded vs prior).
- **M4** — Overlays (policy / visits / opp-policy / threats / move numbers) +
  both-perspectives + STV panel.
- **M5** — Checkpoint A/B compare + value-trajectory chart + import/editor + polish,
  error/loading states, regression check of existing tabs.

## Risks / mitigations

- *GPU contention*: eliminated by the `CUDA_VISIBLE_DEVICES=""` subprocess.
- *Checkpoint shape drift*: handled by mirroring the driver's expander recipe;
  tested against a pre-graft (≤e6) and a post-graft (≥e7) checkpoint.
- *Worker latency / blocking polling*: torch stays out of the HTTP process;
  requests serialize in the worker; results cached. A first cold forward loads a
  checkpoint (~1–2 s) — surfaced as a loading state, then cached.
- *Live-data races*: the run writes new shards/checkpoints continuously; the
  debug tab reads completed files only and tolerates missing/in-progress ones.
