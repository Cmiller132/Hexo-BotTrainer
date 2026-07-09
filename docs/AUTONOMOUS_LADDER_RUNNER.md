# AUTONOMOUS PREFIT-LADDER RUNNER — hexfield_eq (deadline regime)

`scripts/eq_ladder_runner.py` + launcher `scripts/run_eq_ladder.sh`. Runs
detached in WSL, chains the prefit arms on the single GPU **under a hard
deadline**, strength-ranks them vs **SealBot**, picks a winner by the owner's
rules, and launches the `hexfield_eq_main_1` self-play soak from the winner —
no human in the loop. Companion to `docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md`.

## What it does (stages)

1. **Prefit** — sequential: `arm1_vanilla → arm2_reglane → arm3_tokread →
   arm4_raylayout → (arm4c_georay, CONDITIONAL)`. DEADLINE REGIME (owner,
   2026-07-09): **1 epoch per arm** (arm 2's 6-epoch plan is dead), all arms
   identical: `python -m hexfield_eq.prefit --data <main11 corpus> --out <arm
   dir> --epochs 1 --workers 10 --seed 1 --policy-target gumbel --batch-rows
   256 --lr 2.8e-3 --warmup-steps 200 [--limit-steps <calibrated>]` with that
   arm's `scripts/prefit_env/hexfield_eq_<arm>.env` sourced plus
   `HEXFIELD_EQ_PAIR_BUDGET=4.0e7` (C/A arms 1–3) / `1.6e7` (L arms 4/4c).
   All regime numbers live in one `REGIME` dict at the top of the runner and
   are env-overridable (`EQ_LADDER_BATCH_ROWS`, `EQ_LADDER_LR`,
   `EQ_LADDER_WARMUP_STEPS`, `EQ_LADDER_LIMIT_STEPS`,
   `EQ_LADDER_PAIR_BUDGET_CA/_L`, `EQ_LADDER_WORKERS`, `EQ_LADDER_EPOCHS`) —
   the orchestrator's calibration wins. **`EQ_LADDER_LIMIT_STEPS` must be set
   by the orchestrator** (the runner warns loudly if it is not).
   Idempotent: a COMPLETE arm (final-epoch checkpoint + one real diagnostics
   row per epoch, `steps ≥ min(200, cap)`) is skipped; a PARTIAL arm resumes
   via `--resume <latest ckpt>`; an arm ALREADY RUNNING (e.g. started directly
   by the orchestrator) is detected via `/proc` and monitored, never
   double-launched; `--limit-steps ~30` smoke artifacts are quarantined.
   Crash ⇒ retry with resume (3 attempts); stall (no progress 1 h) ⇒ kill +
   retry.
   **Arm 4c is conditional**: it runs only if, after arm 4, the remaining time
   covers its projected duration plus a 50-min reserve
   (`EQ_LADDER_4C_RESERVE_SECONDS`); otherwise it is skipped and blockers-on
   (arm 4) is the default ray mode.
2. **Health** — newest real `diagnostics.jsonl` row. Catastrophic = NaN death
   / no checkpoint / `value_ece_ema > 0.2`. Everything else is a recorded
   warning. `ema_*` metrics are **recorded only** (see 3).
3. **Strength eval** — every non-catastrophic arm's checkpoint, repackaged to
   its **RAW weights** (`"model"` key — owner: at ~2–4k optimizer steps the
   EMA twin lags most of the run), plays an identical
   `eval_arena.play_sealbot_match` (**60 games**, unpaired — SealBot's
   time-limited minimax makes pairs unmatched, so score = decided win rate and
   SE = binomial). Evaluated in decision-priority order (arm4, 4c, 3, 2, 1).
   Results in `<arm>/eval_sealbot.json` (+`_full.json`); idempotent.
4. **Decision** — owner's rules (below), full ranking + reasoning recorded.
5. **Strix baseline (record-only)** — ONE small `play_strix_match` (~60
   games, paired) for the WINNER only; a strength baseline in the status file.
   No decision weight; skipped on any error or deadline pressure.
6. **Soak launch — ALWAYS happens before the deadline** — build
   `<winner>/soak_init.pt` = `{"meta": <arch meta>, "model": <RAW state
   dict>}` (the shape `hexfield_eq.checkpoints.HexfieldCheckpointLoader`
   warm-starts from); write `<ladder root>/hexfield_eq_main_1.launch.toml`
   (copy of `configs/hexfield_eq_main_1.toml` with `[checkpoint]
   initialize_from` pointed at it); source the WINNER arm's env file (arch env
   == winner's checkpoint meta; arm4c brings `HEXFIELD_EQ_RAY_BLOCKERS=0`
   automatically; the prefit-only `HEXFIELD_EQ_PAIR_BUDGET` is stripped) and
   launch `scripts/_hexfield_eq_supervise_main1.sh` detached
   (`CONFIG=<launch toml>`). After ~2 min it verifies the supervisor is alive
   and the first log lines are sane, and records PID + log paths.

## Deadline governor

`--deadline-ts <unix>` (or `EQ_LADDER_DEADLINE_TS`, or `--deadline-in-minutes
N`). Every stage is projected from measured history (median; priors until
data exists: prefit 40 min, eval 20 s/game, strix 15 min, final reserve
15 min) and `TIMELINE <stage>: projected Xm, actual Ym, remaining` lines make
every decision auditable in LADDER_STATUS.md. On projected overrun it degrades
in the owner's order:

1. skip arm 4c (blockers-on is the default ray mode);
2. remaining SealBot matches drop 60 → 40 games;
3. skip the record-only Strix match;
4. last resort: stop remaining prefits/evals and decide from the arms
   completed so far.

The soak launch itself is never skipped (the only hard stop remains "no arm
produced a loadable checkpoint").

## Decision rules (verbatim intent, as implemented in `decide()`)

- Ranking opponent is **SealBot** (owner, 2026-07-08): Strix is much too
  strong for prefit-level checkpoints — all arms would score ≈0. The SealBot
  adapter is **UNPAIRED by design** ⇒ score = decided win rate, **SE =
  binomial** `sqrt(p(1−p)/n)`. 60 games/arm under the deadline
  (`EQ_LADDER_EVAL_GAMES`); the coarser SE biases toward keeping arms, which
  matches the owner's preference.
- **Winner is strength-based, not top-1.** "Only remove the arms that are
  unambiguously negative. Prefer the run to have all mechanisms if possible."
- Preference order, fullest-stack-first: **arm4** — replaced by **arm4c** only
  if 4c beats 4 by **> 1·SE_of_difference** on their SealBot scores (the soak
  then runs `RAY_BLOCKERS=0`) — **> arm3 > arm2 > arm1**. Walk down; select
  the FIRST arm NOT unambiguously negative.
- **Unambiguously negative** = SealBot score < (best arm's score −
  **2·SE_of_difference**) OR catastrophic health (NaN death / no checkpoint /
  `value_ece_ema > 0.2`). SE_of_difference = `sqrt(se_a² + se_b²)`. Mild
  prefit-metric regressions do NOT disqualify.
- Failure doctrine: proceed with best available. Unscored-but-healthy arms are
  skipped in the primary walk; if no scored arm survives, the first
  non-catastrophic arm in order wins (loud warning); if ALL arms are
  catastrophic, the least-bad arm with a checkpoint wins (louder warning).
  The ONLY hard stop: no arm produced a loadable checkpoint.

## Launch (what the orchestrator runs)

```bash
# WSL — detaches, survives the session. Set the calibrated cap + deadline!
EQ_LADDER_LIMIT_STEPS=<calibrated cap> \
EQ_LADDER_DEADLINE_TS=$(( $(date +%s) + 6*3600 )) \
bash /mnt/e/Hexo-BotTrainer-hexgt/scripts/run_eq_ladder.sh
# (further calibration overrides: EQ_LADDER_BATCH_ROWS / EQ_LADDER_LR /
#  EQ_LADDER_WARMUP_STEPS / EQ_LADDER_PAIR_BUDGET_CA / EQ_LADDER_PAIR_BUDGET_L)
```

## Checking status

```bash
tail -f /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/LADDER_STATUS.md   # human log
cat  /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/ladder_state.json     # machine state
tail -f /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/runner.log         # raw runner stdout
tail -f /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/<arm>/prefit.log   # live prefit
tail -f /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/<arm>/eval_sealbot.log
```

`ladder_state.json.stage` ∈ `init | prefit:<arm> | eval:<arm> | decision |
strix-baseline | soak | done | fatal`; `timeline` and `degradation` arrays
carry the deadline audit; heartbeats update `heartbeat_utc` every ~5 min.

## Stopping / resuming the RUNNER safely

- Stop: `kill $(cat /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/ladder_runner.lock)`.
  NOTE: a prefit/eval child runs in its own session and **keeps running** —
  deliberate (progress is never thrown away). To stop the child too:
  `pkill -f hexfield_eq.prefit` (or the `--eval-arm` process).
- Resume: relaunch `run_eq_ladder.sh` (with the same env). The runner
  re-assesses everything (completed arms skip, partial arms resume, a
  still-running child is re-attached via `/proc`), reuses fresh eval results,
  and continues. Remember the deadline env is re-read at start.
- Other knobs: `EQ_LADDER_EVAL_GAMES` (60) / `EQ_LADDER_DEGRADED_EVAL_GAMES`
  (40), `EQ_LADDER_EVAL_MAX_WALL` (3600 s/arm), `EQ_LADDER_STALL_SECONDS`
  (3600), `EQ_LADDER_MAX_ATTEMPTS` (3), `EQ_LADDER_EVAL_SEED_BASE` (990001),
  `EQ_LADDER_STRIX_BASELINE_GAMES` (60), `EQ_LADDER_MIN_EPOCH_STEPS` (200),
  `EQ_LADDER_FINAL_RESERVE` (900 s), `EQ_LADDER_4C_RESERVE_SECONDS` (3000 s),
  `EQ_LADDER_WEIGHTS` (raw|ema, default raw).

## How the soak was launched / stopping THAT

The runner does **not** install the systemd unit
(`scripts/systemd/hexfield-eq-supervisor-1.service` remains the manual
alternative — copy to `/etc/systemd/system/`, set the winner's arch env in it,
`systemctl daemon-reload && systemctl start hexfield-eq-supervisor-1`).
Instead it launched, detached:

```
set -a; source scripts/prefit_env/hexfield_eq_<winner>.env; set +a
CONFIG=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/hexfield_eq_main_1.launch.toml \
ROOT=/mnt/e/Hexo-BotTrainer-hexgt RUNDIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1 \
nohup setsid bash scripts/_hexfield_eq_supervise_main1.sh \
  >> /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1/supervisor_nohup.log 2>&1 &
```

(+ `HEXFIELD_ANCHOR_ROOTS`, `SEALBOT_PATH`, malloc tunables — mirroring the
unit.) PID + logs are recorded under `ladder_state.json.soak` and in
LADDER_STATUS.md.

To stop the soak: `touch /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1/supervisor_halted.flag`
then kill the supervisor (`kill $(cat .../supervisor.lock)`) and the trainer
(`kill $(cat .../driver.pid)`). Clear the flag to allow a restart. The
supervisor auto-resumes from the latest epoch checkpoint and has its own crash
circuit-breaker.

## Dry-run rehearsal (no GPU, nothing executed)

```bash
V=/root/.venvs/hexgt-build/bin/python; R=/mnt/e/Hexo-BotTrainer-hexgt
$V $R/scripts/eq_ladder_runner.py --make-mock /tmp/eq_mock_happy --scenario happy
$V $R/scripts/eq_ladder_runner.py --dry-run --mock-root /tmp/eq_mock_happy
$V $R/scripts/eq_ladder_runner.py --make-mock /tmp/eq_mock_sick --scenario arm3sick
$V $R/scripts/eq_ladder_runner.py --dry-run --mock-root /tmp/eq_mock_sick
$V $R/scripts/eq_ladder_runner.py --make-mock /tmp/eq_mock_deadline --scenario deadline
$V $R/scripts/eq_ladder_runner.py --dry-run --mock-root /tmp/eq_mock_deadline --deadline-in-minutes 45
```

`happy`: all arms healthy, arm4 wins. `arm3sick`: arm3 catastrophic
(`ema_value_ece 0.35`), arm4c beats arm4 by >1 SE ⇒ winner arm4c with
`RAY_BLOCKERS=0`. `deadline`: 45 min left ⇒ 4c skipped, eval plan degrades to
40 games + no strix + partial evals, winner from what completed, soak still
launched. In dry-run a virtual clock charges each stage its projection so the
degradations replay deterministically. Always pass `--mock-root` so status
files stay out of the real ladder root.

## Caveats

- The ranking and the soak init use the **RAW** `"model"` weights
  (`soak_init.pt`); the EMA twin stays inside the prefit checkpoint and its
  `ema_*` metrics are recorded in LADDER_STATUS.md for reference.
- "Identical seeds across arms" means identical match configuration and seed
  streams; SealBot's minimax depth also varies with machine load, which the
  binomial SE absorbs but does not remove.
- Eval wall-clock is capped (`EQ_LADDER_EVAL_MAX_WALL`); games aborted at the
  cap are excluded from `decided` (score stays unbiased, SE grows). If the
  60→40-game degradation triggers mid-stage, earlier arms keep their larger
  matches — SE_of_difference handles unequal ns.
- The runner assumes the orchestrator's patched prefit CLI
  (`--batch-rows/--lr/--warmup-steps` — verified present 2026-07-09).
- `eval_arena._load_hexfield_net` now passes checkpoint `meta` into
  `infer_net_kwargs_from_state_dict` (fixed by the ladder work): the
  `ray_blockers` toggle is meta-only, so an arm4c checkpoint loaded without
  meta would silently rebuild with blocker semantics from env.
