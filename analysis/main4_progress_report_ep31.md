# hexfield_main_4 — ep31 RE-ANALYSIS Report

**Run:** `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4` (KataGo-faithful, POST-FIX)
**Window analyzed:** ep1–ep31 (post-fix fresh relaunch from prefit)
**Prior snapshot:** ep17 (`analysis/main4_progress_report.md`, grade A−)
**Date:** 2026-06-20
**Live config:** `E:/Hexo-BotTrainer-hexgt/configs/hexfield_main_4.toml`

---

## 1. OVERALL VERDICT

**Grade: A− (held from ep17; trending healthy / delta = IMPROVED).**

**Headline — did main_4 show FURTHER improvement past ep17? YES.** It did not plateau or regress. Every load-bearing loss (total/policy/value/soft_policy) sits at its **global minimum at ep31, the latest epoch**; strength set a **point-estimate new high at ep30 (BT Elo 215.0, SealBot wr 0.794)**; the value-Q saturation fix held with zero realized relapse; and 6 of 7 heads net-improved. The predicted ep25–40 plateau materialized only as a **soft, diminishing-returns deceleration**, not a hard floor.

Two things moved the wrong way, both isolated and low-impact: (1) a **latent** value_head.bias edge-mass runaway (0.27→0.81), benign only because it is near-symmetric and cancels in the scalar value; and (2) the three tiny-weight **stvalue heads** (+0.065 to +0.082), most likely a game-length-shortening distribution-shift artifact.

---

## 2. STRENGTH TRAJECTORY (all 6 evals)

Authoritative ep30-pool common-scale BT fit (SealBot pinned 0; converged, 14 players / 25 edges):

| Eval | ep5 | ep10 | ep15 | ep20 | ep25 | ep30 |
|---|---|---|---|---|---|---|
| Cand BT Elo | 76.6 | 5.4 (dip) | 193.6 | 161.5 | 142.3 | **215.0** [114.3, 315.7] |
| SealBot raw wr | .625 | .453 | .726 | .781 | .683 | **.794** (50–13) |

- **ep30 is the best of all six on every fixed anchor.** All four post-dip evals (ep15/20/25/30) have CI95 **fully positive** vs SealBot.
- **ep20 ("the decisive test") CONFIRMED:** wr .781, common-scale 161.5 with CI fully positive.
- **ep15→ep30 delta:** +21.4 common-scale Elo (193.6→215.0) plus the SealBot edge hitting an all-time high (.726→.794). The prior snapshot's latest point was the ep15 self-fit (+180.4); the re-analysis adds ep20/25/30.
- **ep30 vs ep10 dip:** provably up — common-scale +209.6 (z≈3.1), SealBot wr +0.341 (z=3.96).
- **HONEST CAVEAT:** the increment *past ep15 specifically* is NOT statistically resolvable at current eval power (ep30 vs ep15: +21.4 Elo, z=0.30 ns; SealBot wr +0.068, z=0.89 ns; CIs overlap). ep30 is **at-or-above, not provably-above, the ep15 peak**. The ep20→25 wobble (161.5→142.3) is within per-node SE (~50 Elo). Net: continued gain / no regression CONFIRMED; fine progress past ep15 awaits ep35/ep40 evals.

Files: `diagnostics/hexfield.multistage_eval.epoch_{5,10,15,20,25,30}.json`, `eval_pool.json`.

---

## 3. PER-HEAD SCORECARD (ep17 → ep31)

| Head | Δ | Score | loss ep17 → ep31 | Note |
|---|---|---|---|---|
| **policy** | improved | 8.5 | 2.350 → **2.227** (global min) | still falling; early-25% top1 stayed soft (0.350→0.323) |
| **soft_policy** | improved | 8.5 | 2.739 → **2.634** (global min) | benign ~36% share, NOT creeping to domination |
| **opp_policy** | improved | 8.5 | 1.163 → **1.127** (global min) | re-engaged after ep6–17 flat |
| **cell_q** | improved | 8.5 | per-epoch noisy; realized CE 2.20 → **2.04** | apparent rise = target-difficulty artifact, head improved |
| **value (scalar)** | improved* | 7.5 | 0.598 → **0.567** (global min) | *realized honest; latent bias drift (see §6) |
| **moves_left** | improved | 8.0 | 3.540 → **3.468** (global min) | ml-utility ON all run; conv_spearman 0.775→0.689 watch |
| **stvalue 2/6/16** | **REGRESSED** | 5.5 | +0.080 / +0.082 / +0.065 | only wrong-way family; ACCELERATING ep27–31 |

Deceleration is real and uniform (~3–3.5x slower descent ep17→31 vs ep1→17) but every core head is still descending at ep31 — diminishing returns, not a stall.

Files: `diagnostics/hexfield.training.epoch_0000NN.json`, `diagnostics/epoch_0000NN.json` (nested losses), `samples/epoch_0000NN/*.npz`.

---

## 4. EXPLORATION

Intact and plateau-soft; no greedy collapse.

- **Root entropy stable:** 1.563 (ep17) → 1.517 (ep31); oscillates 1.33–1.93 with no trend toward 0.
- **Shaped Dirichlet (α=10.83, noise 0.20) working:** turn-0 is a single FORCED opener (support=1 always — NOT a model signal). Dirichlet bites at turn 1, where visits spread at **95–97% of theoretical max entropy**; turn-1 most-common-opening share only 0.082→0.091; effective-distinct-openings 31.5→25.8 (mild plateau softening, not degeneracy).
- **root_policy_temperature ramp** still aids opening diversity; unsound-seeding probe NEGATIVE (rare openings do not lose more than common ones).
- **PCR/forced-playout economy rock-solid:** Full-PCR share 32.75% (sd 0.34%) hits the 0.33 target every epoch; early-stop saves ~18.8% of visit budget; lcb-override RATE flat (~0.61, prior "decline" was a cohort-size artifact); truncations fell 7.4%→2.3%; rows_skipped_off_legal=0 every epoch.
- **Temperature schedule** (halflife 45, floor 0.15) well-matched to ~100-ply games; opening exploratory, mid/endgame committal. Mild benign endgame committal-softening as games shorten (frac games reaching the deep floor 0.332→0.250).
- **Widening** (lazy=false, cap 96/0.95) appropriately focused; cap binds <0.4% and decreasing — breadth set by search agreement, not the cap. Mild healthy narrowing (median support 14→10) driven by more resolved endgames; top1 mass FLAT at 0.545. ep31 is a length/cohort trough; trough-corrected support is flat ~18.

---

## 5. TRAINING HEALTH (incl. grad-norm & buffer)

- **Gradients:** the ep17 grad_norm-climb watch RESOLVED BENIGN. Slope collapsed ~17–36x; peaked ep21 (5.736), then a flat 5.53–5.74 band (ep25→31 slope NEGATIVE). Residual creep is healthy reallocation into trunk_attn (+0.22) while grad_norm_heads DECLINES (1.483→1.403). clip_fraction flat 1.8–3.2%.
- **AMP/fp:** healthy, essentially unchanged. Scale in {8192, 16384, 32768} all run; ZERO NaN/inf/overflow events; one transient ep18–19 backoff to 8192 self-recovered in 1 epoch.
- **Loss balance:** loss_total reconstructs EXACTLY from the 9 weighted components (±0.000000) at ep17 and ep31 — config weights confirmed, no hidden terms. No head dominates; soft_policy (35.6%) and policy (30.1%) co-equal leaders (post-fix weight 1.0, not the pre-fix 4.0). Every weighted share moved ≤0.67pp.
- **LR/optim:** flat 3e-4 AdamW still productive; the plateau is a soft step-change at the ep13 fixed-budget lock, not a too-high-LR floor bounce. LR decay NOT indicated yet (re-probe ep45–50).
- **Buffer/governor:** clean steady state. reuse_ratio in a tight 4.7–6.0 band (~5x target), keep_prob=1.0 every epoch (no subsampling), no over-reuse, no starvation. NOTE (corrects handoff): the KataGo taper window is NOT pinned — used_rows GREW 56,784 (ep17) → 78,479 (ep31), +38%; the 452k figure is the governor CREDIT bucket, a throttle, not the replay window.
- **Overfit:** ruled out (inferentially — no held-out val set, validation_fraction=0.0). Decisive: value loss at a NEW GLOBAL MIN while independent eval strength set new highs — the exact opposite of a memorization signature.

---

## 6. THE TWO ep17 WATCH ITEMS — RESOLVED?

### (B) Value-bias-drift relapse — verdict: **NO realized relapse; latent prior DID worsen (inert).**

- **The realized guardrail held cleanly.** turn-0 mean|Q|: 0.054 → 0.067 → 0.076 (+0.022 over 14 epochs, ~10x below the broken run's 0.79). **turn-0 saturation fraction (|q|≥0.85) = exactly 0.0 at all 31 epochs.** Early-25% sat ~0. Live root_value_mean flat (-0.011→-0.018). loss_value at a new global min (0.598→0.567). Realized calibration IMPROVED (Brier 0.193→0.173, sign-acc 0.667→0.712; error direction is mild UNDER-confidence, not over).
- **The latent prior worsened exactly as the tripwire feared.** value_head.bias extreme-edge-bin softmax mass 0.267 → 0.609 → **0.811**, crossing 0.5 at ~ep23 (earlier than the extrapolated ep29) and STILL accelerating (the edge-mass deceleration is only the softmax 1.0 ceiling, not a fix). Mechanism: interior bins pushed to a deepening floor (-2.43→-4.90) while extreme bins stay flat. bias-only entropy collapsed 3.80→1.83. value_head.weight inflates even FASTER than the bias.
- **Why still inert:** near-symmetric (sm[0]/sm[-1]≈1.06; bias-implied E[v] only -0.009→-0.025), so it cancels in the scalar value and has NOT leaked into balanced-position confidence. **De-risk:** the identical 65-bin head in known-good main_3 drifts the same way (0.165→0.801→0.983 by ep50) and main_3 trained fine — so this is NOT a main_4-specific pathology. It is the run's #1 latent watch item (the exact mechanism of the pre-fix blowup, currently disarmed by symmetry, with no LN/weight-decay counterbalance).

### (C) Predicted plateau — verdict: **SOFT plateau, did NOT bite by ep31.**

- All four load-bearing losses at GLOBAL MINIMUM at ep31 → incompatible with a hard plateau.
- Deceleration is real (loss_total 0.0658/ep → 0.0196/ep, ~3.4x slower) but a **step-change at the ep13 fixed-budget lock** (1500 steps, 452k governor bucket, ~5x reuse), NOT a continuing curve-over: late-window slope is steady-linear (ep17–24 −0.0205/ep ≈ ep24–31 −0.0187/ep). Still descending at the latest epoch.
- Visible plateau-onset signature: games shortened (mean 114→99.7, p90 230→173), entropy ticked down slightly — but exploration intact.

---

## 7. VALUE-Q SATURATION GUARDRAIL THROUGH ep31

**Status: HELD. No relapse.** FPU config verified live (root_fpu_reduction=0.2, lazy_widening=false, capped widening 96/0.95). Decoder self-validated (ep1 |Q|=0.17415, ep17=0.05394 reproduce prior anchors exactly).

- turn-0 sat-frac = 0.0 at every epoch 1..31; frac≥0.5 and frac≥0.3 also 0.0.
- turn-0 mean|Q| 0.054→0.076 (small upward slope past ep23, benign at ~10x below broken run).
- Confidence correctly confined to resolved endgames (late-25% sat-frac 0.381→0.484; late mean|Q| 0.642→0.712).
- Three honesty cross-checks agree: value loss falling (not collapsing toward 0), root_value_mean pinned near zero, games stayed long.

---

## 8. RANKED CONCERNS

1. **value_head.bias / weight magnitude inflation (LATENT, accelerating) — score 6, REGRESSED.** Edge mass 0.27→0.81, crossed 0.5 ~ep23, still accelerating; weight L2 inflating even faster. Benign ONLY via symmetry. The exact latent mechanism of the pre-fix overconfidence loop, with no normalization/weight-decay counterbalance. **Highest-priority watch.**
2. **stvalue heads (stv2/6/16) late reversal — score 5.5, REGRESSED.** All up ep17→31, ACCELERATING ep27–31 (stv2 slope ~84x its ep18–26 rate); within ~0.02 of ep1 worst-ever. Low impact (weight 0.1 each); most likely game-length-shortening distribution shift; decoupled from the healthy scalar value head.
3. **conv_spearman (moves_left) soft drift — score 8, minor.** 0.775→0.689, still ~0.19 above the 0.5 heal-gate (ml-utility never disabled); noise-dominated; watch through ep40.
4. **Persistent ~11pt second-player advantage — score 7, low urgency.** FPWR plateaued at ~0.444 since ep11 (P1 wr ~0.556, z=-6.5). A genuine learned second-mover edge under no-swap ruleset, NOT saturation; zero draws.
5. **Soft-plateau onset (diminishing returns).** Future per-epoch gains will be small; soft_policy is already at its floor (asymptote ~2.62, gap ~0.01).

---

## 9. RECOMMENDATIONS

| # | Action | Trigger | Expected payoff |
|---|---|---|---|
| 1 | **Continue unchanged through ep35–40** | now | Zero cost. ep35/ep40 evals are the ONLY way to resolve whether strength truly climbs past the ep15 peak (current z=0.30 ns) — the single biggest open question. |
| 2 | **Forward-pass value-VARIANCE/entropy probe on balanced (turn-0 / early-25%) positions** | now (cheap) | Closes the one real gap. Bias-only analysis + realized turn-0 \|Q\| both say no leak, but a direct per-position variance measurement would SETTLE whether the bimodal bias prior is widening the value distribution even with E[v]=0. Arms-or-disarms the latent time-bomb. |
| 3 | **Trim soft_policy weight (1.0 → ~0.5)** | only if policy progress stalls (re-probe ep45–50) | soft_policy is at its floor and yields near-zero marginal signal while consuming ~36% of the gradient budget — the single highest-leverage trim to free trunk capacity for policy/value under a capacity squeeze. |
| 4 | **Modest LR decay (3e-4 → ~2e-4)** | PREMATURE now; re-probe ep45–50 | Load-bearing losses still descending (not LR-floor bouncing), so decay is not yet indicated. Reassess when global-min epochs stop being the latest epoch. |
| 5 | **Add weight decay or a LayerNorm before value_head** | only if the bias drift breaks symmetry (E[v] departs ±0.05) | The structural fix for the only latent relapse mechanism — caps the inflating logit magnitude. |
| 6 | **PCR 192-vs-128 fast-visit A/B** | only if throughput becomes the binding constraint | The 192 bump adds ~16 extra visits/avg move (~13% throughput cost inferred, never A/B'd in main_4). Most of the 1.36x/move vs main_3 is KataGo-divergence host overhead, NOT the bump — so an A/B would quantify how much of the slowdown is recoverable. Low priority while strength is climbing. |
| 7 | **Pin canonical tripwires** (free) | now | Use early-25% top1 (not aggregate top1) as the policy-sharpening alarm, and realized-CE / cross-vintage control (not raw per-epoch loss) as the cell_q/value quality signal — both aggregate metrics are confounded by the bimodal game-length sawtooth. |

---

## INCONCLUSIVE ITEM + RESOLVING EXPERIMENT

**The one unresolved question:** whether the latent value_head.bias bimodal drift (edge mass 0.811, bias-only entropy 1.83) is widening the value-distribution VARIANCE on real balanced positions — even though E[v]≈0 and realized turn-0 |Q| is flat. The no-leak verdict currently rests on (a) the near-symmetry of the bias prior and (b) realized search-averaged Q (q_pol_q), which agree, but a direct forward-pass measurement was deferred for cost.

**Experiment to resolve (Recommendation #2):** load the ep31 checkpoint in the WSL venv, encode a batch of real turn-0 / early-25% balanced positions, run a forward pass, and measure the value-head output-distribution variance/entropy (not just E[v]). If variance stays narrow → fully disarmed. If it is going bimodal on the two extreme bins → the latent prior IS reaching the live forward pass and Recommendation #5 (LN/weight-decay) becomes actionable before symmetry can break.

---

## KEY FILE REFS

All under `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4/`:
- Per-epoch losses/saturation: `diagnostics/hexfield.training.epoch_0000NN.json`, `diagnostics/hexfield.selfplay.epoch_0000NN.json`, `diagnostics/epoch_0000NN.json`
- Strength: `diagnostics/hexfield.multistage_eval.epoch_{5,10,15,20,25,30}.json`, `eval_pool.json`, `events.jsonl`
- Decoded rows / value-bias probe: `samples/epoch_0000NN/*.npz`, `checkpoints/epoch_0000{05,17,25,31}.pt`
- Buffer/governor: `diagnostics/hexfield.select.epoch_0000NN.json`
- Live config: `E:/Hexo-BotTrainer-hexgt/configs/hexfield_main_4.toml`
- Prior snapshot: `analysis/main4_progress_report.md`
- Baselines: `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3`, `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4_prefit/checkpoint_epoch5.pt`
