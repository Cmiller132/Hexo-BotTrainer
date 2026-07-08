# 01 — Architecture

## Services (one compose stack)

```
[browser] ──HTTPS──> Cloudflare edge ──tunnel──> cloudflared ──> app (FastAPI)
                                                                  │
                                                    SQLite volume ┤ models volume
```

- **app** — single container: FastAPI + uvicorn serving (a) the static frontend,
  (b) the game/analysis API, (c) a small in-process worker pool for search.
  One container keeps state simple (SQLite + in-memory sessions); scale-out is
  not a launch goal.
- **cloudflared** — official image, tunnel token via env. No other ingress.

## Reused from the repo (this is the point of putting it in `apps/`)

| Piece | Reused for |
|---|---|
| `hexo_engine` (PyO3) | authoritative rules, legality, win detection |
| `hexfield` (`HexfieldMctsSession.search`) | bot moves — same CPU path the dashboard's Match arena uses (proven end-to-end in the public-repo verification) |
| `hexfield` featurizer + `HexfieldNet` forward | analysis tab: policy/value/top-k per position |
| `hexo_utils` records codec (`.hxr`) | compact move-record blobs in SQLite |
| `hexo_frontend/debug_infer.py` loader patterns | checkpoint loading with arch auto-detect (do NOT import the frontend package; lift the minimal loader into the showcase server) |

The showcase server imports `hexo_engine`, `hexfield`, `hexo_utils` only — it
must not depend on `hexo_train`/`hexo_frontend`.

## Game session lifecycle

1. `POST /api/game` {bot_id, human_color, nickname?} → creates a session
   (UUID token in an httpOnly cookie), inserts a `games` row (status=active).
2. `POST /api/game/{id}/move` {q, r} → engine validates, applies; if it's the
   bot's turn afterwards, the move request enqueues a search job and long-polls
   (or the client polls `GET /api/game/{id}` — simpler, matches the dev arena
   pattern). Bot plays two stones per turn post-opening, same as humans; the
   session tracks turn phase via the engine.
3. Terminal (six-in-line / resign / abandonment timeout ~10 min idle) →
   `games` row finalized (result, termination, ply_count, duration,
   `.hxr` blob), session evicted.
4. `GET /api/game/{id}/analysis?ply=N` → on-demand model eval for the position
   after ply N: policy over legal cells, value, top-k with visit-free network
   opinion; optionally `?search=1` for a small searched eval (capped sims,
   rate-limited). Results cached in `analysis_cache`.

## Bot ladder

Config file `bots.toml` (mounted, not baked into the image):

```toml
[[bot]]
id = "ep2-fast"      # early epoch, low sims
checkpoint = "models/ladder/main7_ep2.pt"
visits = 16
label = "Novice (epoch 2, 16 sims)"

# ... epochs {2, 10, latest} x visits {16, 64, 256, 1024}, curated to ~6-8
# entries rather than the full 12 — a 16-sim latest and a 1024-sim ep2
# overlap in strength; pick a monotone-feeling ladder after calibration.
```

All ladder checkpoints are inference-only exports (`export_weights.py`), loaded
once at startup and kept resident. `HEXFIELD_SUPPORT_RADIUS=4` and the arch
env are set by the container entrypoint — all ladder checkpoints must share the
main_7 arch (the public loader is current-arch-only by design).

## Worker pool / concurrency

- `N_WORKERS` processes (default 4) each own the loaded ladder and serve search
  jobs from a queue (multiprocessing; model tensors are small enough to load
  per-worker). A 256-sim move ≈ 1–3 s; 1024-sim ≈ 4–10 s (set client
  expectations in the UI with a thinking indicator).
- Global cap: `MAX_ACTIVE_GAMES` (8). Per-IP cap: 2 active games + token
  bucket on move/analysis endpoints. 429 with a friendly message beyond caps.
- Search determinism doesn't matter here; each job seeds from os.urandom.

## Frontend (static, no build step)

- `web/index.html` + `app.js` + `board.js` (SVG hex board with click-to-place,
  pan/zoom for the unbounded grid) + `style.css`. Vanilla JS deliberately —
  matches the repo's no-framework frontend philosophy and keeps the public
  artifact easy to study.
- **Play tab**: bot picker (ladder with labels + short blurbs), the board,
  move list, thinking indicator, resign button, optional nickname field at
  game end ("save this game as …").
- **Analysis tab**: game picker (your games via cookie; any finished game by
  id later), ply stepper, policy heatmap overlay (opacity ∝ prior/top-k),
  value/win-prob line chart across plies (lazy: computed as you step, then
  cached server-side), top-5 candidate list with values.
- Mobile-first layout; the board is the screen.

## API surface (complete)

```
POST /api/game                    create game
GET  /api/game/{id}               state (poll)
POST /api/game/{id}/move          human move
POST /api/game/{id}/resign
POST /api/game/{id}/nickname      set/curate nickname on the finished game
GET  /api/game/{id}/analysis      per-ply model insight (cached)
GET  /api/games                   recent finished PUBLIC games feed (paginated)
GET  /api/bots                    catalogue metadata (checkpoints + allowed sims)
GET  /api/stats                   public aggregates (see 03)
GET  /healthz                     liveness (also used by the tunnel)
```

No admin endpoints in the public app. Checkpoint refresh and DB maintenance
happen via CLI scripts run in the container (`docker compose exec`).

## Decisions locked 2026-07-05 (design round with the owner)

- **Visual language**: dark polished-stone "Instrument" direction; hexagonal
  beveled stones keyed to player index like the dev dashboard (p0 blue
  #4f93ff, p1 red #ff5650); warm-white interactive accent; mosaic grid;
  last-two-stone marks; winning six outlined as the union boundary of the
  six stone tiles. Mockup of record: scratchpad design/v3.html (copy into
  apps/showcase/web notes when building).
- **Bot selection**: checkpoint catalogue list × sims segmented control
  (16/64/256/512, any combination). `bots.toml` becomes a catalogue of
  checkpoints; visits move to a per-request parameter validated against an
  allowed set. No flavor text.
- **Analysis**: mirrors the play view's chrome; policy overlay in the dev
  debug-screen style (all legal cells shaded, 0.10+0.90×max-norm opacity,
  best cell outlined, hover readout with exact %, tint = to-move player's
  color) with a single opacity slider — no log scale, no search overlay.
  Panel shows value + a "horizon" block (short-term value + moves-left from
  the model's heads) + value/ply chart + a click-to-jump move list (arrow
  keys step). The analysis payload therefore adds `stv` and `moves_left`.
- **Game discovery**: public recent-games feed — finished games are public
  by default and load by id (shareable URLs fall out of this for free).
- **Final round (v4 mockup)**: policy overlay uses a paler hue-shifted tone
  family distinct from the solid stone fills (still keyed to the side to
  move); analysis gains a ply scrubber slider, autoplay (~1 ply/s), chart
  click-to-seek, and a copy-game-link button; finished play games offer an
  "analyze this game" jump; boards get MANUAL pan/zoom only (wheel-at-cursor
  + drag with click-preserving threshold, pinch + one-finger pan on touch,
  double-click/tap reset, no auto-fit); touch placement is tap-to-stage then
  tap-to-confirm, mouse stays direct-place.

## Explicitly out of scope at launch

Human-vs-human play, accounts, ELO for humans, spectating live games,
a stats page/tab (the /api/stats endpoint ships; a page for it is a
post-launch candidate), XPU/GPU (phase 2), search-based analysis overlays.
