# Ray-tap conv — wave-1 prefit-ladder results

Date: 2026-07-10. Spec: `SPEC_RAYTAP_CONV.md` §6.3 (wave 1). Branch:
`raytap-phase-r` (Phase R implementation + this wave's tooling). Runner:
`scripts/eq_ladder_runner.py` with the wave extensions (`EQ_LADDER_ARMS`,
`EQ_LADDER_EVAL_PRIORITY`, `EQ_LADDER_NO_SOAK`); ladder root
`/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_raytap_wave1`.

## Procedure (spec §6.3 compliance)

- **Regime** (all arms identical): 1 epoch on the main11 BC corpus,
  `--limit-steps 600` (calibrated to the owner's 8-hour GPU window; the
  original R/L ladder ran 1200 — cross-wave levels are therefore NOT
  comparable, wave-internal deltas are), batch-rows 256, lr 2.8e-3, warmup
  200, seed 1, gumbel policy target, `HEXFIELD_EQ_PAIR_BUDGET=1.6e7` (all
  arms — see anomaly 1), workers 10.
- **Full arch env pinned per arm** (`scripts/prefit_env/hexfield_eq_raytap_a*.env`):
  CHANNELS=192, GROUP_ORDER=12, C_ORBIT=16, HEADS=3, SUPPORT_RADIUS=4,
  REG_LANE=1, REG_TOK_READ=0 everywhere; only FEATURE_VERSION / RAYTAP /
  TRUNK differ per §6.3.
- **Search budgets visit-based everywhere**: arm search at 512 visits
  (vbs 32) in every match; no time budgets on any arm side. (SealBot itself
  is a time-limited minimax anchor — identical across arms, absorbed by the
  binomial SE per the standing ladder doctrine.)
- **Ranking**: SealBot-anchored, 60 games/arm, unpaired, RAW weights,
  seed_base 990001. Strix is record-only.
- The live `hexfield_eq_main_1` soak was stopped for the wave window
  (owner-authorized) and relaunched afterwards; the ladder ran in
  `EQ_LADDER_NO_SOAK=1` mode (no soak relaunch from a wave winner).

## Arms and results

| arm | features | raytap | layout | blocks | BC top-1 | policy CE† | value CE | value ECE (raw / EMA-gate) | SealBot score ± SE | prefit wall |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 (control) | v1 (25) | 0 | CCLACCLACLA | 11 | — | — | — | — | — (prefit skipped, deadline governor) | — |
| A1 | v2 (46) | 0 | CCLACCLACLA | 11 | 0.4298 | 2.1863 | 0.5792 | 0.1355 / 0.0752 | 0.2167 ± 0.0532 | 133 min‡ |
| A2 | v2 | both | CCLACCLACLA | 11 | 0.4407 | 2.1734 | 0.6061 | 0.1047 / 0.0632 | 0.3333 ± 0.0609 | 108 min |
| **A5** | v2 | both | **CCACCACA** (L removed) | **8** | **0.4421** | **2.1228** | 0.5917 | 0.1595 / **0.0235** | **0.4667 ± 0.0644** | **36 min** |

† Held-out CE against the (gumbel) soft policy target — equals the
soft-policy KL up to the targets' entropy constant, which is identical
across arms (same corpus rows, same targets), so deltas read as KL deltas.
‡ A1's wall-clock includes ~20 min of GPU contention from an operator error
(a concurrent debug run, killed); its trained steps/seed are identical to
the other arms — metrics unaffected, timing not comparable.

Ladder BT rank (SealBot score order): **A5 > A2 > A1**. A1 is
*unambiguously negative* under the standing rule (gap to best 0.25 >
2·SE_diff 0.167). The formal ladder "winner" is A2 by the
fullest-stack-first preference walk (A2 within band of best); the ranking
above is the wave read.

Strix (record-only, 60 games, paired):

| arm | Strix score ± SE | pentanomial |
|---|---|---|
| A2 (formal winner) | 0.0667 ± 0.0322 | {0: 26, 1: 4, 2: 0} |
| A5 (top scorer, extra record) | 0.0333 ± 0.0232 | {0: 28, 1: 2, 2: 0} |

(Strix is far above prefit level by design; both figures are the standing
record-only baseline, no decision weight. The A2/A5 ordering inverts the
SealBot read at ~1σ — 60 paired games against a much stronger opponent
carry little signal at this level.)

Serve throughput (pos/s, full fast serve profile incl. CUDA graphs +
half-serve, 512 visits, 16 games, ply-cap 20 — `raytap_serve_throughput.py`;
path label per spec §2.4). Short-probe numbers are warmup/compile-dominated
(the live soak's steady state is ~10× these) — they are comparable to EACH
OTHER, not to the soak:

| arm | serve path | pos/s | vs A1 |
|---|---|---|---|
| A1 | baseline kernels (no equipped convs) | 1.88 | — |
| A2 | fused-K1 (per-shape fallback on compile failure) | 1.61 | −14% |
| **A5** | fused-K1 (per-shape fallback on compile failure) | **2.20** | **+17%** |

Serve read: ray-tap costs ~14% on the unchanged layout (A2 vs A1), and
removing the 3 L blocks more than pays it back (A5 is +37% vs A2, +17% vs
the no-ray-tap baseline).

## The two pre-registered comparisons

**A2 − A1 (operator effect, same layout, same features):**

- SealBot: **+0.117** (0.333 vs 0.217; SE_diff 0.081 → 1.45σ)
- BC top-1: +0.011; policy CE: −0.013; value CE: +0.027 (worse);
  value ECE: −0.031 raw / −0.012 EMA (better)
- Read: the ray-tap operator adds strength on top of the v2 features at
  matched budget. Direction consistent across strength and most BC
  metrics; not individually conclusive at 60 games.

**A5 − A2 (L-subsumption — the load-bearing comparison):**

- SealBot: **+0.133** (0.467 vs 0.333; SE_diff 0.089 → 1.51σ)
- BC top-1: +0.001 (parity); policy CE: −0.051 (better); value CE: −0.014
  (better); value ECE: +0.055 raw (worse) / −0.040 EMA (better)
- Architecture: **3 L blocks removed** (11 → 8 blocks, −24 C² of L-block
  MACs per cell), prefit wall-clock 36 min vs 108 min (~3×), serve +37%
  (pos/s table above).
- Read, stated plainly: **at this scale ray-tap does not merely subsume the
  ray-attention blocks — removing them cost nothing and the point estimate
  says it helped (+13pp SealBot), while training 3× faster on a 27%
  smaller trunk.** The spec pre-registered "equal strength, smaller layout,
  faster serve" as a recordable outcome; wave 1 met or beat that on every
  axis measured. The strength edge alone is 1.5σ — direction, not proof;
  the conjunction (strength ≥, policy CE better, smaller, faster) is the
  finding.

## Anomalies (all recorded, none decision-blocking)

1. **A5 pair-budget crash.** At the C/A-class training pair budget
   (`HEXFIELD_EQ_PAIR_BUDGET=4.0e7`) A5's prefit crashed deterministically
   (3/3 attempts) at the first backward with `CUDA driver error: device not
   ready` surfacing in `_BiasGather.backward` — i.e. a deferred async fault
   from earlier in the step. At the L-class budget (1.6e7) the identical
   config trains cleanly; A5 was rerun at 1.6e7. Pair-budget microbucketing
   is gradient-neutral (step-global denominators), and all wave arms ended
   up at 1.6e7, so comparability is intact. Root cause is OPEN: yesterday's
   arms 1–3 ran 4.0e7 without ray-tap; the suspect is a large-microbucket
   transient interacting with the ray-tap graph under `torch.compile`.
   Follow-up: a `CUDA_LAUNCH_BLOCKING=1` trace at 4.0e7 (repro assets:
   the failing config is exactly `hexfield_eq_raytap_a5.env` + prefit at
   4.0e7).
2. **A0 skipped.** The deadline governor dropped the control arm's prefit
   (112 min remaining < 133 min projected + reserve) per the pre-agreed
   degradation order (arms were queued A2, A5, A1, A0 so pressure drops the
   control first). Consequence: the features-alone-vs-v1 delta (A1 − A0)
   is NOT measured this wave. Yesterday's arm4 (same config as A0 at 1200
   steps) scored 0.66 vs SealBot but is not step-matched. A0 can be
   backfilled in wave 2 at 600 steps if the delta is wanted.
3. **A1 timing contamination** (see ‡ above). Timing only.
4. **K1 bench gate is shape-bistable**: −44.9% overhead at B=96/Npad=256
   vs +150% at B=38/Npad=640 (both B·Npad ≈ 24k, C=192; parity 3.9e-3 at
   both). §2.4 fallback recorded; end-to-end pos/s below is the honest
   serve number; conv2 promotion remains available for the candidate
   config if `both` serve cost disappoints at scale.
5. **Strength SEs are 60-game binomial** — the wave separates arms
   directionally, not at 2σ pairwise. The standing 60-game regime was the
   owner's deadline trade-off.

## Recommendations

- **Wave 2 (A2c / A6):** run **A6 first** (winner − 1 A block on the A5
  base, i.e. attention-budget subtraction from CCACCACA) — A5's win makes
  the attention budget the live question. A2c (conv2 attribution) is now
  lower value: `both` won cleanly and K2 removed the training-memory
  motive for conv2; keep A2c only if the K1 serve picture at scale forces
  the conv2 fallback. Backfill **A0** in the same wave (36–110 min) to
  close the features-alone delta, and consider a 1200-step A5-vs-A2
  confirmation at ≥150 games if a 2σ strength read is wanted before big
  commitments.
- **Quotient Phase B:** the A5 answer it consumes is: **ray-tap subsumes
  the L blocks at prefit scale — plan against the L-free CCACCACA-class
  layout.** That removes the L-block head-split (6-head, orbit-halves)
  from the rep-typing surface Phase B must carry, and shrinks the trunk to
  C-conv + A-attention + lane, all of which Phase B already types. Treat
  as a qualified input (600-step/60-game evidence): if Phase B's layout
  choice is expensive to reverse, gate it on the wave-2 confirmation
  above.

## Reproduction

Launch (idempotent; resumes/skips completed arms):

    EQ_LADDER_REPO=<raytap-phase-r worktree> \
    EQ_LADDER_ROOT=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_raytap_wave1 \
    EQ_LADDER_ARMS="raytap_a2:l,raytap_a5:l,raytap_a1:l" \
    EQ_LADDER_EVAL_PRIORITY="raytap_a2,raytap_a5,raytap_a1" \
    EQ_LADDER_NO_SOAK=1 EQ_LADDER_LIMIT_STEPS=600 \
    EQ_LADDER_DEADLINE_TS=<unix> \
    bash <worktree>/scripts/run_eq_ladder.sh

Artifacts per arm under the ladder root: `prefit.log`,
`diagnostics.jsonl`, `checkpoint_epoch0.pt`, `eval_sealbot.json`,
`soak_init.pt` (repackaged RAW weights), `strix_baseline.json` (A2, A5),
`LADDER_STATUS.md` + `ladder_state.json` (full timeline incl. the A5
crash/rerun and the A0 skip).
