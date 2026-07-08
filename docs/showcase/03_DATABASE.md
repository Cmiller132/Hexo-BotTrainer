# 03 — Database (SQLite)

Single file `/data/showcase.db`, WAL mode (concurrent readers + one writer fits
the app's write volume easily). Compact by design: a full game is one row plus
a small binary blob.

## Schema

```sql
-- The bot ladder over time. Games reference a bots row, so refreshing
-- "latest" creates a NEW row and old games keep their true identity.
CREATE TABLE bots (
  id          INTEGER PRIMARY KEY,
  slug        TEXT NOT NULL,            -- "ep10-256" (from bots.toml)
  label       TEXT NOT NULL,
  run         TEXT NOT NULL,            -- "hexfield_main_7"
  epoch       INTEGER NOT NULL,
  visits      INTEGER NOT NULL,
  weights_sha TEXT NOT NULL,            -- content hash of the .pt
  active_from TEXT NOT NULL,            -- ISO time
  UNIQUE (slug, weights_sha)
);

CREATE TABLE games (
  id           TEXT PRIMARY KEY,        -- UUID (public, used in URLs)
  bot_id       INTEGER NOT NULL REFERENCES bots(id),
  human_color  INTEGER NOT NULL,        -- 0/1
  started_at   TEXT NOT NULL,
  finished_at  TEXT,
  status       TEXT NOT NULL,           -- active|finished|abandoned
  result       INTEGER,                 -- +1 human win, -1 bot win, 0 draw/none
  termination  TEXT,                    -- six_in_line|resign|timeout
  ply_count    INTEGER,
  duration_s   REAL,
  nickname     TEXT,                    -- optional, sanitized, NULL default
  client_hash  TEXT,                    -- salted hash of CF-Connecting-IP
  record       BLOB                     -- .hxr bytes (hexo_utils codec)
);
CREATE INDEX games_bot_time ON games (bot_id, finished_at);
CREATE INDEX games_status   ON games (status);

-- On-demand analysis results, cached forever (positions are immutable).
CREATE TABLE analysis_cache (
  game_id  TEXT NOT NULL REFERENCES games(id),
  ply      INTEGER NOT NULL,
  bot_id   INTEGER NOT NULL REFERENCES bots(id),  -- which net produced it
  payload  BLOB NOT NULL,               -- msgpack/json.gz: value, top-k, policy sparse
  PRIMARY KEY (game_id, ply, bot_id)
);
```

Why `.hxr` in the `record` blob: it's the repo's existing compact binary game
codec (Rust, tested), the dev tools can open showcase games directly, and a
typical game is well under a kilobyte. The moves are also replayable through
the engine for any future re-analysis, so nothing else needs storing.

## Statistics (views, exposed read-only via GET /api/stats)

```sql
-- Win rate per bot (the headline table)
CREATE VIEW v_bot_stats AS
SELECT b.slug, b.label, b.epoch, b.visits,
       COUNT(*)                                   AS games,
       AVG(g.result = -1)                         AS bot_winrate,
       AVG(g.ply_count)                           AS avg_plies,
       AVG(g.duration_s)                          AS avg_duration_s
FROM games g JOIN bots b ON b.id = g.bot_id
WHERE g.status = 'finished'
GROUP BY g.bot_id;

-- Daily activity
CREATE VIEW v_daily AS
SELECT date(started_at) AS day, COUNT(*) AS games,
       SUM(status = 'finished') AS finished
FROM games GROUP BY day;

-- Human-win hall of fame (nicknamed wins vs the strongest bots)
CREATE VIEW v_hall_of_fame AS
SELECT g.nickname, b.label, g.ply_count, g.finished_at
FROM games g JOIN bots b ON b.id = g.bot_id
WHERE g.result = +1 AND g.nickname IS NOT NULL
ORDER BY b.visits DESC, g.ply_count ASC;
```

Opening statistics (popular first placements, human vs bot) are derived by
decoding `record` blobs — a maintenance script materializes
`openings(first_cell, count, human_winrate)` nightly rather than at request
time.

## Retention & privacy

- Store no raw IPs, no user agents, no emails. `client_hash` (salted, salt in
  `.env`) exists only for abuse analysis and per-IP caps; the salt can be
  rotated to orphan history.
- Nicknames are the only user-provided text; sanitized at write, only shown
  in stats views.
- Abandoned games auto-finalize after the idle timeout and keep their record
  (they're data too).
- The whole DB is small (thousands of games ≈ a few MB); no pruning needed.
  Nightly backup per 02.

## Analysis workflow it enables (your side, private)

Pull the DB file (or query over SSH) and analyze in the dev repo: decode
`.hxr` records with `hexo_utils`, replay through the engine, run any
checkpoint over the positions — e.g. "where do humans beat ep-latest",
"which openings do 1024-sim games punish", blunder detection by value-drop.
Nothing in the showcase constrains this; the DB is the interface.
