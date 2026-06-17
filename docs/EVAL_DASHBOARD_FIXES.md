# Eval + Dashboard fixes — changes & deploy

Branch `claude/eval-dashboard-fixes` (worktree `/mnt/e/hexgt-evaldash`). Fixes the
issues in `docs/INVESTIGATION_eval_and_dashboard.md`. Verified on CPU against the
LIVE run `hexfield_main_2` read-only (the live run + `:8080` dashboard were never
touched). `file:line` below are in this worktree.

## What changed, per issue

### Eval (hexfield)

- **E1 — empty eval `.hxr` is silent (§5).**
  `eval_arena.py:251-341 _write_eval_hxr` — a 0-record write (every game had falsy
  `.actions`) now emits a LOUD `logging.WARNING` and the write exception is logged
  instead of blanket-swallowed; a `games_written`/`games_skipped` counter is filled
  via the new `stats=` kwarg and threaded into match meta as `hxr_games_written`
  (`eval_arena.py:882-896, 1453-1465`). NOTE: this makes a future regression
  machine-visible; the *actual* records appear once the corrected `eval_arena.py`
  (whose `_Game.actions` populate) is the running build.

- **E2 — permanent anchor (bc_prefit) silently dropped (SEV-2).**
  `multistage_eval.py:408-428 select_opponents` records each unresolved permanent
  anchor in `roster.dropped_anchors` (new `Roster` field, `:239-244`) and logs a
  WARNING (was a bare `continue`). Surfaced in `_stage_a_bridge` problems
  (`:989`), the Stage-A detail (`:992`), and `_roster_summary` (`:2707-2714`) so it
  lands in every per-epoch JSON. `_resolve_anchor_path` (`:287-300`) gains the
  `HEXFIELD_ANCHOR_ROOTS` (os.pathsep-separated) search-root override, tried FIRST;
  config (`config.py:146-158`) documents absolute-path anchors.

- **E3 — SealBot fail-open silently re-anchors the Elo scale (SEV-2).**
  `multistage_eval.py:1531-1707 _stage_d_pool` gains `sealbot_expected_but_unavailable=`;
  when SealBot was config-ENABLED but the anchor re-pinned off it, Stage-D is marked
  `degraded` and the verdict gets `anchor_substituted`/`substituted_to`/
  `degraded_note`. Wired from all three drivers: `run_multistage_eval` (`:902-910`),
  `run_multistage_eval_concurrent` (`:2425-2436`), and the parts path
  `aggregate_pool` (`:1985-2000`) via the new `_epoch_has_sealbot_edge` (`:2039-2061`).

- **E4 — radius-4 confound invisible (SEV-1).**
  `multistage_eval.py` reads the live featurize radius read-only
  (`_live_featurize_radius`, `:131-150`) and tags radius-8-era opponents
  `featurized_ood` on each edge + provenance (`_build_checkpoint_edge_from_match`,
  `:1186-1235`). Such opponents are EXCLUDED from the pinned BT zero-point
  (`_choose_anchor` new `ood_labels=` kwarg, `:2489-2552`) and flagged on the
  Stage-D fit + verdict (`ood_opponents`/`ood_note`, `:1589-1607, 1667-1690`). New
  config `MultiStageEvalOpponents.radius8_opponents = ("bc_prefit",)`
  (`config.py:160-172`). Descriptive only — the same-lineage primary verdict is
  unaffected.

### Dashboard (hexo_frontend)

- **D1 — `learning_health` blind to the real eval (SEV-2).**
  `web.py:1622-1624` passes `multistage_eval_history`; `_learning_health`
  (`web.py:3766-3896`) drives the hexfield eval-health branch off it (verdict +
  candidate Elo + descriptive SealBot winrate), suppresses the false "No SealBot
  evaluation result yet" / "D6 augmentation preview is missing" messages
  (`:3997-4002`), and adds `latest_verdict`/`latest_cand_elo`/
  `latest_sealbot_winrate`/`latest_eval_epoch` to the payload (`:4022-4030`). New
  helpers `_ms_candidate_elo`/`_ms_sealbot_winrate` (`web.py:3762-3808`).

- **D2 — eval_pool/sealbot CI shipped but dead; W-L matrix misreports counts.**
  `_eval_pool_summary` (`web.py:3558-3593`) keeps a SLIM `raw` with only the integer
  `physical_wins_*` so the matrix renders the TRUE head-to-head, not the n_eff-
  weighted fractional `wins_a/wins_b`. `app.js renderHistEvalPool` (`:4090-4230`,
  bound `:222`, container `index.html:192`) draws the checkpoint-unified BT ladder,
  the per-opponent W-L matrix and a verdict-history strip from the previously dead
  `eval_pool` payload.

- **D3 — roster/verdict readability.**
  `_multistage_eval_history` (`web.py:3490-3526`) threads a compact `roster` incl.
  `permanent_anchors`. `app.js`: dropped-anchor muted pills (`msDroppedAnchors`
  `:3598-3608`, used `:4072-4074`), cadence subtitle (`msEvalCadenceSubtitle`
  `:3614-3621`), the Δ-Elo CI error bar (`:3996-4012`), and verdict chips that now
  surface E3 `DEGRADED`/anchor-substituted and E4 `OOD` opponents
  (`:3979-3998`). Supporting CSS in `static/styles.css`.

## How to DEPLOY to the live run/dashboard

The fixes only take effect once the running builds import the corrected code.
Nothing here mutates the live run.

1. **Eval (`eval_arena.py` / `multistage_eval.py` / `config.py`).** The LIVE trainer
   imports hexfield from `PYTHONPATH=/mnt/e/hexgt-katago/packages/hexfield/python`.
   Deploy by updating that worktree to this branch's hexfield package (merge/cherry-
   pick `claude/eval-dashboard-fixes`, or sync the three files), then the change
   takes effect at the NEXT trainer (re)start — it is read at process import.
   - To restore bc_prefit on the live process WITHOUT moving the checkpoint, set
     `HEXFIELD_ANCHOR_ROOTS=/mnt/e/Hexo-BotTrainer-hexgt` in the trainer's systemd
     unit (the canonical tree that holds `runs/hexfield_bc_1/checkpoint_epoch2.pt`),
     or change the config anchor to an absolute path.
   - **The 7 existing empty `.hxr` will NOT backfill** — only FUTURE eval epochs
     (5,10,15… after redeploy) write populated replay records and the new
     anchor/SealBot/radius flags.

2. **Dashboard (`web.py` / `static/*`).** The `:8080` dashboard imports
   `hexo_frontend` from the canonical tree `/mnt/e/Hexo-BotTrainer-hexgt` (editable
   install). Deploy by syncing the four frontend files to that tree and restarting
   the dashboard service (`systemctl --user restart` the dashboard unit, or its
   systemd equivalent). The readers are backward-compatible: they render the
   existing 7 reports immediately (D1/D2/D3 light up without an eval rerun); the new
   `dropped_anchors`/`ood_opponents`/`anchor_substituted` chips appear once an eval
   epoch is produced by the redeployed eval build.

## How to VERIFY after deploy

- **Eval:** after the next eval epoch, decode `<run>/evaluation/epoch_NNNNNN/*.hxr`
  and assert `num_records > 0`; the detail JSON gains `hxr_games_written > 0` per
  match. If bc_prefit was unresolvable, `roster.dropped_anchors` records it loudly;
  once the anchor root is fixed, bc_prefit reappears in the roster. A SealBot death
  now shows `verdict.degraded == true`. Radius-8 opponents show
  `verdict.ood_opponents`.
- **Dashboard:** History → Evaluation region shows the BT ladder + per-opponent
  W-L matrix (counts now match the physical head-to-head), the verdict strip, and
  the run-overview health pill reads a real verdict instead of "no eval". Eval games
  appear in History (source=`evaluation`) once the populated `.hxr` land.
- **Tests:** `bash tests/eval_dashboard/_run_all.sh` (eval) and
  `bash tests/eval_dashboard/_run_dashboard.sh` (dashboard) — both `*_ALL_OK=1`.
