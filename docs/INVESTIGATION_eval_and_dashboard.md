# Investigation: hexfield in-run EVAL + dashboard game-history

Authoritative synthesis of four deep-dives, verified against the canonical tree
`/mnt/e/Hexo-BotTrainer-hexgt` and the LIVE run dir
`/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2`. Read-only. Every claim below is
anchored to `file:line` and was checked against code AND run-dir artifacts.

**Date:** 2026-06-17 · **Run:** hexfield_main_2 (radius-4) · **Status:** live (do not disturb)

> Naming note: all `file:line` are in the canonical tree
> `/mnt/e/Hexo-BotTrainer-hexgt/packages/...`. CRITICAL: the *live trainer process*
> (pid 84567) imports hexfield from `PYTHONPATH=/mnt/e/hexgt-katago/packages/hexfield/python`,
> a DIFFERENT worktree. The eval functions are byte-identical across the two trees,
> but the differing parent dir is the direct root cause of Issue #2 (anchor drop).

---

## 1. How the eval works (mechanism)

**Dispatch.** The runner calls `plugin.py:67 evaluate_epoch` → `plugin.py:68` →
`evaluation.py:69 evaluate_epoch`. The eval therefore runs **inside the training
process**, per epoch. This is load-bearing for the radius confound (§3.1).

**Cadence / gating** (`evaluation.py:89-99`):
- `every = max(int(mse_cfg.every_n_epochs), 1)` → run config **5**
  (`_resume_config.toml:81 every_n_epochs = 5`).
- `not enabled` → `"disabled"`; `epoch < 2` → `"skipped: no opponent yet"`;
  `epoch % every != 0` → `"skipped: every_n_epochs=5"`; missing
  `cand_path = ctx.checkpoint_dir / f"epoch_{epoch:06d}.pt"` (`evaluation.py:90`) → skipped.
- So it fires on epochs **5,10,15,…**. Confirmed: detail JSONs exist exactly for
  epochs 5,10,15,20,25,30,35.
- The whole block is `try/except Exception` fail-soft (`evaluation.py:100,125`):
  any error is recorded as `{"status":"error"}` and never kills the epoch.

**Same function also runs the moves-left-head audit / heal-gate**
(`evaluation.py:137-153`, `audit_moves_left_head`), which CAN write/unlink the run-dir
flag `ml_auto_disabled.flag`. That audit is the ONLY part of `evaluate_epoch` that
mutates the run; the multistage eval itself is pure. The combined record is written
to `hexfield.evaluation.epoch_NNNNNN.json` (`evaluation.py:156-157`). A legacy
`_play_pair` 16-game arena (`evaluation.py:23-66`) is dead in this path.

**Roster** (`multistage_eval.py:349-399 select_opponents`, 4 roles, label-deduped):
- **SealBot** — cross-lineage zero-point, `ckpt=None`, only if `sealbot_enabled`
  (`_resume_config.toml:92 = true`).
- **Permanent anchors** (`multistage_eval.py:352-359`) from
  `cfg.opponents.permanent_anchors` = `(("bc_prefit","runs/hexfield_bc_1/checkpoint_epoch2.pt"),("ep5","epoch_000005.pt"))`
  (`config.py:133-136`), resolved by `_resolve_anchor_path` (`multistage_eval.py:210-275`).
  **A missing anchor is silently skipped with a bare `continue`** (`multistage_eval.py:354-356`)
  — see Issue #2.
- **Sliding bracket** (`multistage_eval.py:361-370`) — nearest `bracket_size`
  log-grid rungs strictly below the candidate. `bracket_size = 1`
  (`_resume_config.toml:93`) → exactly ONE rung.
- **Champion** = verdict reference (`multistage_eval.py:372-398`): highest epoch
  `≤ cand_epoch - verdict_reference_lag` (lag = 5, `_resume_config.toml:85`); falls
  back to nearest-prior when none old enough (`eligible = [...] or prior_epochs`,
  `multistage_eval.py:387`). This is why ep5's champion is ep4 (fallback).

**Budget** (`multistage_eval.py:431 allocate_budget`, called at `:2153`):
`games_budget = 64` (`_resume_config.toml:80`), `sealbot_share = 0.5` (`:84`) →
SealBot 32 unpaired games, remaining 32 split across the N checkpoint opponents
(`per = 32 // N`, floor ≥2). For 3 ckpt opponents → 10 each. Advisory, not a hard cap.

**Search / CRN** (`config.py:215-226`, `eval_arena.py` docstring 20-88):
`full_search_visits = 512` (`_resume_config.toml:82`) used for BOTH nets;
`eval_virtual_batch_size = 32` (`:83`, a run override over the code default 16);
`opening_plies=8`, `opening_temperature=1.0`. **Checkpoint matches are PAIRED with
common random numbers** (`eval_arena.py:41-86, 919-959`): 2 seat-swapped games per
pair share `pair_seed`; game 0 LEADS (searches sampled opening), game 1 is the
seat-swapped FOLLOWER replaying the recorded opening ply-for-ply; after the opening
both play GREEDY (temp 0). The candidate is net A in ONE persistent batched session
across all opponents → wall-clock = MAX over opponents, bit-identical to N serial
matches. **SealBot is UNPAIRED** (`eval_arena.py:1490-1518`): its minimax depth
varies under load, so seats merely alternate and a Wilson CI is taken on the raw
winrate; SealBot is time-bounded (`sealbot_time_limit`), so it is NOT visit-matched
to the 512-visit hexfield side.

**Outputs:**
1. `diagnostics/eval_pool.json` — append-only Bradley-Terry pool
   (`config.py:251 pool_path`, saved `multistage_eval.py:2255`). **29 edges, 16
   players** across epochs 5-35. The compounding artifact.
2. `diagnostics/hexfield.multistage_eval.epoch_NNNNNN.json` — per-eval detail
   (`multistage_eval.py:2257`): meta, roster, stages, full BT ratings table, edges,
   `sealbot_winrate_ci95`, verdict. One per eval epoch (7 exist).
3. `diagnostics/hexfield.evaluation.epoch_NNNNNN.json` — wrapper, one per EVERY
   epoch 1-35 (`evaluation.py:156`); holds the multistage summary + the MLH audit.
4. `diagnostics/epoch_NNNNNN.json` — full runner record (multistage nested under
   `result.evaluation.multistage`).
5. `<run>/evaluation/epoch_NNNNNN/<a>_vs_<b>.hxr` — eval-game replay records
   (`_write_eval_hxr`, `eval_arena.py:245-302`). **LIVE FILES ARE EMPTY — see §3 & §5.**

**Verdict** (`multistage_eval.py:1475-1522`, `eval_stats.py:745-768`): the ONE
primary hypothesis is `r_candidate - r_champion` from the pooled BT difference CI.
**PROMOTE** if CI-lo > `promote_elo_threshold` (0.0); **REGRESS** if CI-hi <
`regress_elo_threshold` (0.0); else **INCONCLUSIVE**. Bonferroni is wired but inert
(one gating edge, `_n_gating_edges` returns 1). The label **gates nothing**:
`eval_gating_enabled`/`eval_promotion_enabled` are False and `_assert_no_run_mutation`
(`multistage_eval.py:703-722`) is asserted at entry (`:2132`). The detail JSON's own
`single_epoch_se_elo_note` concedes the candidate is a FRESH BT node each epoch, so
the candidate-vs-champion verdict NEVER compounds (SE ≈ 120-140 Elo, resolving only
~250-300 Elo): it is a **gross-regression tripwire, not a fine-edge test**.

---

## 2. What it actually emitted for main_2

DOES IT RUN? **Yes.** All 7 eval epochs (5,10,15,20,25,30,35) produced
`status: completed`, BT `converged: true` (`max_grad ≈ 6.3e-12`). ep35 ran
post-resume (16:11, `elapsed ≈ 342.6s`). No eval errors in the train log (only a
benign non-writable-numpy `UserWarning` and an unrelated inductor "Not enough SMs"
warning).

| epoch | verdict | champion | elo_diff | diff CI95 | se_elo | SealBot wr (decided) | champ edge decided/wr |
|---|---|---|---|---|---|---|---|
| 5  | **REGRESS** | ep4 (fallback) | -374.5 | [-722, -27] | 177 | 0.281 (32) | 10 / 0.10 |
| 10 | INCONCLUSIVE | ep5  | -30.1 | [-178, +118] | 76  | 0.419 (31) | 16 / 0.375 |
| 15 | INCONCLUSIVE | ep10 | -1.3  | [-215, +213] | 109 | 0.344 (32) | 10 / 0.50 |
| 20 | INCONCLUSIVE | ep15 | -0.3  | [-239, +239] | 122 | 0.500 (30) | 8  / 0.50 |
| 25 | INCONCLUSIVE | ep20 | +143.9| [-142, +430] | 146 | 0.594 (32) | 10 / 0.70 |
| 30 | INCONCLUSIVE | ep25 | +87.7 | [-159, +334] | 126 | 0.438 (32) | 8  / 0.625 |
| 35 | INCONCLUSIVE | ep30 | -0.6  | [-270, +269] | 138 | 0.533 (30) | 10 / 0.50 |

Reads: ep5 REGRESS is a genuine startup artifact (weakest net). Every epoch since is
INCONCLUSIVE — consistent with the ~250-300 Elo resolution floor. The real progress
signal is the **descriptive SealBot zero-point winrate** climbing 0.28 (ep5) →
~0.5-0.6 (ep25-35). Pool at ep35: 29 edges / 16 players, converged, SealBot pinned
at 0 Elo; candidate ladder cand_ep5 (-84) → cand_ep20 (-9.5) → cand_ep30 (+37.5) →
cand_ep35 (-31.8); bc_prefit -299.6.

**Eval-game replay artifacts are EMPTY.** `evaluation/epoch_000035/` holds 3 files
(`cand_ep35_vs_ep5/ep20/ep30.hxr`), each ~99-101 bytes. Decoded with the real codec:
`players=[seat0, seat1]`, **`num_records = 0`** (verified live). Header-only stubs.

---

## 3. PRIORITIZED ISSUES (severity-ranked)

### SEV-1 — radius-4 confound: REAL and ACTIVE (but narrowly scoped)

**Verdict: REAL.** The legal/support radius is a **process-global env read once per
process**: `support.py:40 _SUPPORT_RADIUS = int(os.environ.get("HEXFIELD_SUPPORT_RADIUS", LEGAL_RADIUS))`
(Python mirror) and `rust/src/support.rs:24-33` `support_radius()` is a `OnceLock`,
used at `support.rs:110` `legal.retain(|c| … d <= radius)` to filter the candidate-move
set for EVERY net's MCTS. The live process has `HEXFIELD_SUPPORT_RADIUS=4`
(`_resume_config.toml:17`; systemd unit). The eval runs in-process (§1), so **every
opponent net is featurized at radius-4**. The opponent loader has NO per-net radius
override: `eval_arena.py:201-221 _load_hexfield_net` → `:421-424 HexfieldEvaluator(...)`;
grep for `support_radius`/`SUPPORT_RADIUS` in `eval_arena.py`/`multistage_eval.py`/
`inference.py` is empty.

Effect: a radius-8-trained net (bc_prefit, from `runs/hexfield_bc_1/checkpoint_epoch2.pt`,
a BC prefit with no radius override) is forced OOD at radius-4 → plays weaker than its
true strength → **inflates the candidate's relative Elo** against it. Consistent with
bc_prefit's pool rating of **-299.6 Elo** (cand_ep30 beat it 8-0). The candidate,
native radius-4, is not penalized — an asymmetric, candidate-favoring bias in exactly
the flagged direction.

**Honest scoping (load-bearing):** the candidate-vs-CHAMPION **primary verdict**
compares two SAME-lineage radius-4 nets (e.g. cand_ep35 vs ep30), so that ONE number
is NOT directly radius-confounded. The confound poisons the **descriptive fixed-anchor
curve** (bc_prefit, and the partially-radius-8-initialized early epochs) — i.e. the
only signal the eval explicitly claims compounds — and any cross-lineage read.

**Fix:** featurize each opponent at ITS OWN training radius. Currently impossible
(OnceLock + module-global). Minimal options, in order:
(a) record training radius in the checkpoint `meta` (NOT present today — `epoch_000035.pt["meta"]`
has only lineage/epoch/run/train_state) and thread it per-net into the featurizer;
(b) exclude radius-mismatched opponents from the cross-lineage curve;
(c) at minimum, annotate every radius-8-era opponent edge as "featurized OOD" so it
is not read as a strength signal (and surface it in the dashboard — see §4 #6).

### SEV-2 — bc_prefit, a PERMANENT anchor, is silently dropped from the live roster

**Real, happening now.** ep30 roster = `[bc_prefit(anchor), ep5, ep20, ep25]`;
ep35 roster = `[ep5(anchor), ep20, ep30]` — **bc_prefit GONE** (verified by loading
both detail JSONs; config identical). Root cause: `multistage_eval.py:354-356` — a
permanent anchor failing `is_file()` is skipped with a bare `continue`, no warning.
`_resolve_anchor_path` (`:210-275`) tries the run-data tree, then the repo tree via
`Path(__file__).resolve().parents[4]` (`:259`), then `run_dir.parent.parent`. In the
LIVE process the importing file is `/mnt/e/hexgt-katago/.../multistage_eval.py`, so
`parents[4] = /mnt/e/hexgt-katago`, and
`/mnt/e/hexgt-katago/runs/hexfield_bc_1/checkpoint_epoch2.pt` does NOT exist; the
run-data path `/mnt/e/Hexo-BotTrainer/runs/hexfield_bc_1/...` does NOT exist either.
The file exists ONLY in the canonical tree
`/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1/checkpoint_epoch2.pt`, which is not
on the live process's search roots.

Consequence: the eval loses one of only ~3 stable anchors of its own compounding
curve; bc_prefit's pooled rating now rests on STALE edges (cand_ep5…cand_ep30; none
from cand_ep35), and the per-epoch graph is inconsistent (some epochs have the anchor,
some don't).

**Fix:** (a) make a missing PERMANENT anchor LOUD — record a `dropped_anchors` field
in the stage detail + meta instead of a silent `continue` (`multistage_eval.py:354-356`);
(b) store anchors as ABSOLUTE paths in config, or add the actual run-data sibling so
the live PYTHONPATH tree can reach the bc checkpoint.

### SEV-2 — SealBot fail-open silently re-anchors the entire Elo scale

`multistage_eval.py:1181-1193` catches `Exception` across the whole SealBot boundary
and returns it as a soft "unavailable"; `:1294-1338` then drops just that edge, keeps
Stage C "completed", and `_choose_anchor` re-pins the zero-point to bc_prefit → else
the lowest checkpoint. SealBot is the ONLY cross-lineage zero-point and its own
docstring admits its worker can die mid-match under load. If it dies, every reported
Elo shifts and the verdict label is STILL emitted with no `degraded`/`anchor_changed`
flag. In THIS run SealBot never dropped (all 7 JSONs have a ~30-decided SealBot edge),
so this is a latent confound, not an active one. Secondary: the in-trainer eval has no
explicit `sealbot_path` (config has none; `:1174` passes `cfg.opponents.sealbot_path = None`,
relying on the adapter default) — the dashboard's `--sealbot-path /mnt/e/SealBot` is a
DIFFERENT process.

**Fix:** when the anchor changes from SealBot, tag the verdict block as
`anchor_substituted` + status `degraded`; pin `sealbot_path` explicitly in the eval config.

### SEV-2 — `learning_health` (dashboard) is blind to the real eval and reports falsehoods

`_training_run` builds health from the DEAD path:
`web.py:1622 "learning_health": _learning_health(epoch_history, evaluation_history, live_status)`
— it passes `evaluation_history`, NOT `multistage_eval_history`. `_learning_health`
(`web.py:3719-3823`) consequently emits literal `"No SealBot evaluation result yet
for the completed epochs."` (`web.py:3789`) and `"D6 augmentation preview is missing
for the latest epoch."` (`web.py:3823`) while 7 full BT reports exist. The
run-overview health pill / status-band "eval" chip therefore show `--` / "Watch · no
eval" for the live run. (Detail in §4.)

### SEV-3 — statistical power / identifiability

- **Tiny decided counts; permanently single-epoch-limited verdict.** ep35 checkpoint
  edges = `decided: 10` each (5 CRN pairs); pentanomial `[2,0,1,0,2]` → `n_eff: 6.25`
  (`eval_stats.py:227-264`). Primary `elo_diff -0.6, ci95 [-270, +269], se 137.7`. The
  note concedes this never tightens. Every epoch 10-35 is INCONCLUSIVE. **Fix:** if the
  per-epoch label is meant to gate, raise `games_budget` or concentrate it on the single
  primary edge; otherwise stop emitting a per-epoch PROMOTE/REGRESS and report only the
  descriptive curve.
- **Each candidate epoch is a SEPARATE BT node from its identically-named opponent.**
  The pool has 16 players for 29 edges; `epoch_000030.pt` is rated TWICE — cand_ep30
  (+37.5) vs opponent ep30 (-31.2), a 68.7-Elo gap for the SAME file. Candidates connect
  only to shared opponents, never to each other → the difference CI cannot tighten and
  ratings are wasteful. **Fix (highest-leverage statistical lever):** unify candidate and
  opponent into ONE node keyed by checkpoint identity so the graph connects across epochs
  and the descriptive curve actually compounds.
- **Scale pinned by a down-weighted, depth-varying SealBot edge.** SealBot is anchored at
  0 while its edges carry `weight 0.5` (`sealbot_overdispersion`, `multistage_eval.py:604/1062`).
  Cross-checkpoint DIFFERENCES are clean; ABSOLUTE placements vs SealBot are noisy by
  construction and should not be read as calibrated.

### SEV-4 — hygiene

- **Config `eval_visits = 128` is misleading** (`_resume_config.toml:75`): actual play is
  512 (`_eval_visits` returns `full_search_visits` when set, `multistage_eval.py:137-139`;
  every edge provenance shows `eval_visits: 512`). Both nets use 512, so within-eval
  comparability is fine — but the config line mis-states the protocol.
- **Broad `except Exception` everywhere.** `_write_eval_hxr` swallows all
  (`eval_arena.py:301-302`), as do the anchor/SealBot boundaries — a pattern that hides
  integration breakage. Worth one audit pass.
- **Bonferroni inert** (`multistage_eval.py:1484-1490`): wired but only one edge gates;
  must receive k>1 if multi-edge gating is ever enabled.

---

## 4. Dashboard current state + candid assessment

**Location.** NOT pip-installed (`/root/.venvs/hexgt-build/.../hexo_frontend` absent).
Live code is the repo package
`/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python/hexo_frontend/`: `web.py`
(4743 lines, server + all readers), `static/index.html` (514), `static/app.js` (8887),
`static/styles.css`. Hand-rolled `ThreadingHTTPServer` + `HexoPlayHandler`
(`web.py:1095-1102`), route dispatch in `do_GET` (`web.py:1105`)/`do_POST` (`:1307`).
Three SPA screens by URL hash: `#matchScreen / #historyScreen / #debugScreen`.

**Reads a run dir** via `_training_run` (`web.py:1563-1624`): `manifest.json` →
lineage; `_diag_prefix` → `"hexfield"`; `diagnostics/epoch_*.json` → `_epoch_history`;
`hexfield.evaluation.*` → `_evaluation_history` (`web.py:3388`);
`hexfield.multistage_eval.*` → `_multistage_eval_history` (`web.py:3448`, 7 reports
live); `diagnostics/eval_pool.json` → `_eval_pool_summary` (`web.py:3513`, 29 edges);
`selfplay/*.hxr` + `evaluation/*.hxr` → game history/replay.

**WORKING eval surface (the only one):** the multistage block, keyed off
`run.multistage_eval_history`:
- `renderHistRatingTable` (`app.js:3815-3904`, `#histRatingTable` `index.html:191`):
  pooled-Elo ladder (anchor ⚓), latest verdict chip (`msVerdictClass` `app.js:3547`),
  candidate Elo+CI, Δ-Elo, verdict note, pure-eval tag, headline edges (vs champion /
  SealBot / bc_prefit).
- `renderHistTrends` (`app.js:3563`): T7 "Eval Elo (anchor-pinned)" Elo+CI over epochs
  (`app.js:3729-3764`); T8 "Win-rate vs bc_prefit / vs ep5" frozen-anchor curves
  (`app.js:3766-3802`).

**BROKEN / DEAD / MISSING — needs major improvement:**
1. **Two parallel eval paths, one fully dead.** `evaluation_history` is 35 ALL-NULL
   rows: the hexfield wrapper schema only POINTS to the multistage report and carries
   none of the keys `_evaluation_history` reads (`games/wins/mean_turns` →
   `web.py:3408-3427`). The legacy T4 "SealBot eval" chart (`app.js:3695-3728`) never
   draws (`histSeriesCount<2`).
2. **`learning_health` is blind to the real eval** (SEV-2 above): driven only by the
   dead `evaluation_history` (`web.py:1622, 3719-3823`), so the run-overview verdict and
   pill are wrong/misleading ("no eval", false "D6 missing") for every hexfield run.
3. **No BT ladder / head-to-head UI despite the data shipping.** `eval_pool` (29 edges
   with `wins_a/wins_b/weight/kind`) and `sealbot_winrate_ci95` (`web.py:3504,3513-3551`)
   are both transmitted and have ZERO app.js consumers — dead payload weight. A
   pairing-matrix / pooled-ladder-over-time view is the obvious missing high-value piece.
4. **Coarse cadence not signposted.** Multistage runs every 5 epochs while
   `epoch_history` is per-epoch; the Elo/winrate charts silently have far fewer points
   with no UI cue (`app.js:3729-3802`).
5. **Verdict noise under-communicated.** The report flags the verdict as permanently
   single-epoch-limited; the UI shows the chip + note but gives the wide CI ([-270,+269]
   live) no visual treatment, so a single INCONCLUSIVE can be over-read.
6. **Radius confound is INVISIBLE.** Nothing in the payload or UI surfaces the featurizer
   radius per opponent, so an OOD-inflated candidate looks legitimate on the ladder. No
   `web.py` reader or `app.js` renderer exposes radius provenance.
7. **Hard-coded headline allowlist duplicated.** `web.py:3435
   _MULTISTAGE_HEADLINE_OPPONENTS = ("sealbot","bc_prefit","ep5")` and the matching
   `["bc_prefit","ep5"]` chart list (`app.js:3771`) — new anchors require editing both.
8. **Charts are tiny hand-rolled SVG sparklines** (280×110, `app.js:3279-3284`): no zoom,
   no shared epoch cursor, single delegated tooltip — low-resolution for a multi-opponent
   ladder trajectory.

---

## 5. THE GAME-HISTORY PLAN — make EVAL games appear in History

**Key finding: the dashboard is ALREADY fully wired for `evaluation/` games.** The gap
is UPSTREAM — the live eval writes header-only `.hxr` files with **zero records**.

**The pipeline already in place (no dashboard change needed for it to work):**
- Enumeration: `_iter_history_artifact_files` walks BOTH `run_dir/"selfplay"` and
  `run_dir/"evaluation"` (`web.py:2999-3002`), recursing subdirs (`:3008-3023`);
  `_is_loadable_history_path` whitelists exactly `{"selfplay","evaluation"}`
  (`:3154-3155, 3275`); the History "source" filter already offers `evaluation`.
- Parse → rows: `_hxr_base_rows` (`web.py:3076-3128`) opens each file, reads
  `record_file.players` + `iter_records()`, emits one row per record with
  `game_id/status/winner/length/actions/epoch/source/seed/players/abort`; memoized by
  `(mtime_ns,size)`. **A file with zero records yields zero rows.**
- Replay: `_training_history` (`web.py:1809-1839`) re-applies `record.action_ids`
  through the engine; **raises "Game history artifact contains no games" when not records**
  (`:1827-1828`). An empty file is therefore invisible.

**The .hxr schema contract** (Rust codec `packages/hexo_utils/rust/src/records.rs`):
magic `HEXOREC1` (`:32`), schema v1 (`:34`); `HexoRecord{game_id, seed, status,
action_ids: Vec<u32>, winner, placements, abort}` (`:212-219`); players
`HexoRecordPlayer{role,...}` (`:137`); file API `create/open/begin_game/iter_records`
(`:295,328,364,405`); per-game `HexoRecordGameWriter` closed via
`finish_completed`/`finish_aborted` (`:439`). Self-play already satisfies this
(`selfplay.py:217-240, 384-410` — real multi-game files on disk).

**Why eval files are empty** (verified live: `cand_ep35_vs_ep5.hxr` →
`players=[seat0,seat1]`, `num_records=0`): `_write_eval_hxr` (`eval_arena.py:245-302`)
skips every game whose `.actions` is falsy (`:279 if not getattr(g,"actions",None): continue`)
and returns `None` when `n==0` (`:301`). Every `g` passed had empty `.actions`, so only
the header was written. Corroboration that the LIVE build is a PRE-FIX variant: the live
`hexfield.multistage_eval.epoch_000035.json` has **no `hxr_record` key** anywhere, even
though the canonical `play_checkpoint_match` threads `"hxr_record": hxr_path` into `meta`
(`eval_arena.py:855`) and the round-robin at `:1420`. The on-disk canonical writer is
correct; the running bytes are stale (the epoch_000035 files were written 16:11 by the
earlier launch). `_Game.actions` IS populated in canonical code (`eval_arena.py:580/594,
1146/1155`); callers `:846, 1412`; `multistage_eval.py:794, 971, 1228`.

**Minimal change set (do NOT implement here):**
1. **Make the running eval build actually persist actions.** (Re)deploy the current
   `eval_arena.py` so `_write_eval_hxr` runs against `_Game` objects whose `.actions` are
   populated. Harden: assert/surface a `games_written` counter so a 0-record file is LOUD,
   not silently swallowed by the `:279` guard + `except Exception` (`:301-302`).
2. **No dashboard change required** for enumerate/parse/replay — they already accept
   `evaluation/`. Fixed files appear in History automatically.
3. **Tagging (already mostly present — make it land):**
   - File path already encodes matchup + epoch
     (`evaluation/epoch_NNNNNN/<cand>_vs_<opp>.hxr`); `_history_source` keys it
     `"evaluation"` and `_epoch_from_artifact_path` recovers the epoch.
   - `game_id` already encodes opponent + candidate seat
     (`ep{ep}-{a}-vs-{b}-g{i}-{candP0|candP1}`, `eval_arena.py:283`). Keep.
   - **Result** is captured per game via `finish_completed("player{seat_w}", plies)` /
     `finish_aborted(stage="evaluation", ...)` (`eval_arena.py:288-298`).
   - Optional cosmetic: set player `role`/name to the real labels (candidate vs opponent)
     per seat instead of generic seat0/seat1 (`eval_arena.py:272-275`) so the UI shows
     which side is the candidate. Records render either way.
4. **Radius note:** replay re-applies raw `action_id` `(q,r)` coord packs
   (`unpack_action_id` `eval_arena.py:286`, `unpack_coord_id` `web.py:1837`) through the
   engine, independent of `HEXFIELD_SUPPORT_RADIUS`. The radius confound affects *who won*,
   not whether the game renders — no schema/radius mismatch in the history pipeline.

**Phased checklist:**
- [ ] **P0** Redeploy canonical `eval_arena.py` to the running eval build so `_Game.actions`
      populate; assert `games_written > 0`; surface the counter (make 0-record loud).
- [ ] **P0** Verify: next eval epoch's `evaluation/epoch_NNNNNN/*.hxr` decode with
      `num_records > 0` AND the detail JSON gains a `hxr_record` path per match.
- [ ] **P1** Confirm History screen lists the eval games (source=`evaluation`) and replay
      renders board + winner.
- [ ] **P2** (cosmetic) Per-seat candidate/opponent player labels.
- [ ] **P2** (dashboard, separate) Tag/filter eval games by opponent + result + epoch in
      the History list UI (data is already in `game_id`/path/`winner`).

---

## 6. Prioritized recommendations

**Quick wins (small, high-value):**
1. **Fix empty eval `.hxr` (P0, §5):** redeploy canonical `eval_arena.py`; assert/surface
   `games_written`. Lights up eval games in History with zero frontend change.
2. **Re-target `learning_health` to `multistage_eval_history` (`web.py:1622, 3719-3823`):**
   removes the false "no eval / D6 missing" health verdict on every hexfield run.
3. **Delete or retarget the dead `evaluation_history` path** (`web.py:3388-3429`,
   T4 chart `app.js:3695-3728`): 35 null rows + a never-drawing chart.
4. **Make missing PERMANENT anchors LOUD** (`multistage_eval.py:354-356`): record
   `dropped_anchors` in meta — surfaces the silent bc_prefit drop (SEV-2).
5. **Store anchor paths as ABSOLUTE in config** (or add the run-data sibling) so the live
   PYTHONPATH tree resolves bc_prefit again (SEV-2).
6. **Fix the misleading `eval_visits=128` config line** to reflect the real 512
   (`_resume_config.toml:75`).

**Medium (statistics / correctness):**
7. **Unify candidate ↔ opponent BT nodes by checkpoint identity** (§3 SEV-3): the single
   biggest lever to make the descriptive curve compound and connect the graph.
8. **Tag the radius-8-era opponent edges as "featurized OOD"** and stop reading the
   cross-lineage curve as a strength signal until per-net radius is honored (SEV-1).
9. **Flag SealBot anchor substitution as `degraded`/`anchor_substituted`** and pin
   `sealbot_path` explicitly (SEV-2).
10. **Either raise `games_budget` / concentrate it on the primary edge, OR stop emitting a
    per-epoch PROMOTE/REGRESS label** and report only the descriptive curve (SEV-3).

**Larger dashboard rework:**
11. **Add a BT ladder / pairing-matrix view** consuming the already-shipped `eval_pool`
    (29 edges) and `sealbot_winrate_ci95` (`web.py:3504,3513-3551`) — currently dead payload.
12. **Surface per-opponent featurizer radius / lineage provenance** in the eval payload + UI
    so an OOD-inflated candidate is visible (SEV-1, §4 #6).
13. **Signpost eval cadence** (every-5 vs per-epoch) and give the wide verdict CI visual
    treatment (§4 #4, #5).
14. **De-duplicate the headline-opponent allowlist** (`web.py:3435` / `app.js:3771`) — make
    it data-driven from the roster.
15. **Upgrade the SVG sparklines** (zoom, shared epoch cursor) for the multi-opponent ladder
    (§4 #8).
