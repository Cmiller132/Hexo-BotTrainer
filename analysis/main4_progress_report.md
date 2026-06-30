# hexfield_main_4 — Definitive Progress Report (post-fix, through ep18; ep19 self-play live)

**Date:** 2026-06-20  **Run dir:** `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4`  **Config:** `E:/Hexo-BotTrainer-hexgt/configs/hexfield_main_4.toml`
**Broken-run archive (contrast):** `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4_pre_fix_saturated_ep19`  **Known-good baseline:** `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3`

---

## 1. OVERALL VERDICT

**Health grade: A− (healthy, learning, strengthening — past its steepest phase).**

main_4 is **healthy and genuinely learning.** The value-overconfidence feedback loop that destroyed the original launch is **decisively fixed and durable through 18 epochs** with large margin. Load-bearing losses fall and stay honest (policy 2.819→2.291 min, value in a tight honest band 0.632→**0.584 new run-low @ep18** vs the broken run's collapse to 0.31), strength clears noise by ep15 (first PROMOTE, +180 BT Elo, only candidate with a fully-positive CI), and every one of the seven heads net-improved with none dead, detached, or regressing. The ep14–17 multi-head loss reversal that briefly looked like a forward risk **reversed back down at ep18**, reclassifying it as replay-window/step-budget oscillation rather than divergence. The real story is no longer health — it is **diminishing returns**: the loss curve is bending toward a soft plateau (base case ep25–40) under a fixed-budget regime (1500-step cap + reuse ~5x + train_bucket ~452k pinned since ep13), and eval cadence/throughput cap how fast the next improvement can be confirmed.

---

## 2. STRENGTH / LEARNING TRAJECTORY

**Is it learning? YES.** Headline number: **value loss hit a new run-low 0.5838 @ep18 while strength reached +180.4 BT Elo @ep15** — honest losses + rising strength is the textbook anti-overfit signature, the exact inverse of the broken run.

**Strength (joint Bradley-Terry, SealBot pinned at 0 Elo):**
- cand_ep5 **+59.5** → cand_ep10 **−14.1** → cand_ep15 **+180.4** Elo.
- cand_ep15 is the **only candidate with a fully-positive CI95 [73.8, 286.9]** (z=3.32 vs 0, p≈0.001); first PROMOTE.
- Corroborated across three independent fixed anchors: SealBot winrate 0.625→0.726; main2_ep45 0.70→0.75; ep5-self 0.40→0.75 — not one noisy opponent.
- The ep10 dip is single-epoch eval noise (SE ~120–140 Elo over ~20–31 decided games; root_value_mean clean −0.008), fully overtaken at ep15.
- vs broken: cand_ep15 sits **+280.8 Elo above the broken ep15 candidate** on the same scale.

**Loss trajectory:**
- loss_policy 2.819 → 2.291 (min ep14) → 2.332 (ep18, resumed falling). Early slope −0.42/epoch (ep1→10), late −0.046 (ep10→17) — decelerating but still down.
- loss_value 0.632 → 0.584 (run-low ep18), honest band, no collapse.
- loss_total 8.720 → 7.512 (min ep14) → 7.613 (ep18, resumed falling).

**Caveat (the real limiter): eval power, not direction.** Verified climb is through **ep15 only**; ep16–18 have no strength signal yet. Next test is the **ep20 multistage eval** — the single most important upcoming data point.

---

## 3. PER-HEAD SCORECARD

| Head | Status | Score | One-line |
|---|---|---|---|
| **policy** | healthy | 8.5 | 2.819→2.291 (min ep14)→2.332@ep18; floor ~0.7 below broken, below no-soft main_3's best-ever (2.513); trunk no longer starved. |
| **value** | healthy | 8.5 | Honest masked CE 0.584–0.632, no collapse; bias-prior edge-bin mass growing/accelerating (0.064→0.290) but stays SYMMETRIC (E[v]=−0.009) = legit endgame sharpening, not false confidence. |
| **opp_policy** | healthy | 8.0 | 1.215→1.152; the one head where main_4 lags main_3 (~1.16 vs 1.05, soft-head competition); magnitude diluted ~3x by ~68% zero-target masking. |
| **soft_policy** | healthy | 8.5 | 4.0→1.0 down-weight achieved intent: stable ~36% weighted share (vs 69–75% broken); tracks main policy, doesn't starve trunk. |
| **stvalue_2/6/16** | healthy | 8.5 | Most diagnostic confirmation of the fix — regressed in broken while value collapsed; post-fix all net-DECREASE and stay below main_3; decoded early-game stvalue stays low/stable 0.04–0.14. |
| **moves_left** | healthy | 8.5 | Near-flat loss (3.602→3.531) but correct (high-entropy 65-bin CE on ~120-ply games); audit PASSES every epoch (conv_spearman 0.62–0.80, ml_auto_disabled=False); functional, not dead. |
| **cell_q** | healthy | 8.0 | Fastest learner (3.003→1.975 min ep10), reversed up to 2.279@ep17, **back down 2.248@ep18**; targets verified clean (rust kernel byte-identical to serial oracle, truncated rows masked). |

**All seven heads healthy. None broken, dead, or regressing.**

---

## 4. EXPLORATION ASSESSMENT — well-calibrated

- **Opening diversity not degenerate.** Move-1 is deterministically the canonical center (board translational symmetry, not a defect); real diversity begins at move-2 (25–42 distinct second moves over ~65–79 games, choice-entropy ~3.0–3.6 nats, top-share 0.06–0.20). Opening-ply (1–4) policy-target entropy is rock-stable ~3.0–3.7 nats across all epochs, matching known-good main_3 — Dirichlet is demonstrably firing.
- **Explore→exploit transition correctly ordered every epoch:** monotone entropy decline openings (~3.4) → mid (~1.9–2.2) → endgame (~1.2). Aggregate root_policy_entropy_mean stays in the 1.45–1.93 band — never the broken run's stuck ~2.5 (diffuse), never greedy collapse.
- **Mechanism verified live:** shaped Dirichlet (frac 0.20, ~51% mass on top-10 candidates), lazy_widening=false (cap 96 binds <1% of rows), root_fpu_reduction=0.2, forced_playout_k=1.0, temp halflife 45/floor 0.15, root_policy_temp 1.1/early 1.15, clean_root_prior_cache=true (prevents cross-ply compounding). PCR split locked at 33% full; exploration budget spent exactly where data is recorded.
- **One genuine nuance (watch):** midgame is meaningfully more committal than main_3 (greedy-fraction top1>0.9 or support≤2 ≈0.21 @ep18 vs main_3 ~0.10, ~2.2x, rising mildly). By design (halflife 60→45, forced_playout_k 2→1 for conversion), not a collapse (79% of midgame moves still carry multi-move exploration; openings stay diverse), co-occurs with rising strength + honest value.

---

## 5. TRAINING HEALTH ASSESSMENT — numerically sound

- **No instability.** amp_scale = two isolated single backoffs (32768→16384 @ep5, 16384→**8192 @ep18**), never repeated intra-epoch halving; 0 non-finite tensors across ep16/17/18 checkpoints; max-abs weight 6.79 (fp16 ceiling 65504).
- **Gradients clean.** grad_norm_mean rose 4.10→5.59 then **plateaued ep12–18** (NOT accelerating; trunk_conv flat 4.71→4.64 @ep17). p95/mean flat 1.45–1.60 (no spiking tail). clip_fraction bounded 0.012–0.063; adaptive clip tracks 1.75× EMA. Trunk grads grow while heads decline = trunk building representations, heads settling.
- **Buffer/overfit guarded.** reuse_ratio bounded 4.1–5.6 (~5.06 @ep18), inside main_3's healthy 5.0–6.3 band; train_bucket pinned 452k (never throttled, never starved); keep_prob=1.0; status=completed every epoch.
- **Weights/optimizer.** AdamW lr=3e-4 fixed (no scheduler), wd=1e-4 on matrices only; trunk drifts ~2.4x less than heads from prefit (trunk relL2 0.40 cos 0.93; heads relL2 0.95 cos 0.82) = stable representation with specializing heads, no prefit-forgetting, no thrashing.
- **loss_total reconstructs EXACTLY** from the 9 weighted components every epoch → configured weights are live, no silent weighting bug.

---

## 6. VALUE-Q SATURATION GUARDRAIL — DOES THE FIX HOLD? **YES, decisively.**

The guardrail is balanced-position confidence: turn-0 |Q| and early-game (first 25% of plies) mean |q_chosen| / saturation fraction. Saturation must stay near zero on balanced positions and live only in resolved endgames.

| ep | turn-0 \|Q\| | early mean\|q\| | early sat(\|q\|≥0.85) | late mean\|q\| | loss_value |
|----|----|----|----|----|----|
| 13 | 0.0141 | 0.055 | 0.0004 | 0.572 | 0.6049 |
| 15 | 0.0319 | 0.077 | 0.0000 | 0.626 | 0.5963 |
| 17 | 0.0539 | 0.103 | 0.0004 | 0.647 | 0.5979 |
| 18 | 0.0366 | 0.109 | 0.0004 | 0.681 | **0.5838** (run-low) |
| 19 (partial) | 0.012 | 0.033 | 0.0000 | 0.917 | — |

Three independent confirmations:
1. **Balanced-position saturation stays ~0** (early sat ~0.0004 every epoch; broken hit 0.037 by ep13 — ~90x higher). No balanced position scored as decided.
2. **Saturation correctly LOCATED in resolved endgames** (late mean|q| 0.57→0.68); the early/late gradient is textbook-correct — the broken signature was FLAT across phases.
3. **Value loss stays HONEST** (run-low 0.5838 @ep18) — a re-saturating head would show loss *falling toward 0.31* as it overfits saturated ±1 targets; it does the opposite. root_value_mean pinned −0.006…−0.012 all epochs.

Magnitude contrast @ep17: post-fix early|q| 0.103 vs broken 0.768 (~7.5×); turn-0 |Q| 0.054 vs 0.762 (~14×). ep18 turn-0 |Q| **dropped** (0.054→0.037) and ep19-partial dropped further (0.012) — refutes any monotone climb. **The fix holds.**

---

## 7. RANKED CONCERNS / RISKS (what to watch)

1. **(MEDIUM, forward-looking) Plateau shape.** policy/soft_policy/loss_total bottomed ep10–14 with sharply decelerating slopes; steps pinned at 1500 cap + window saturated from ep13 = fixed-budget passes. Base case: soft plateau ~ep25–40. *Not a defect — the ceiling of the current config.*
2. **(MEDIUM, structural) Eval power/cadence.** Full eval only every 5 epochs; single-epoch champion edge SE ~120–140 Elo. The ep15 +240 point rests on 20 games — the reliable claim is the joint-fit +180.4 [73.8,286.9]. A plateau/reversal could hide up to ~5 epochs.
3. **(MEDIUM, cadence) Throughput ceiling.** ~21 epochs/day, selfplay ~85% of wall-clock, ~1.85× slower than main_3 (pcr_fast_visits 192 + KataGo host overhead). The fast=192-vs-128 knob (~14% of the cost) is an unverified data-quality bet (fast moves are never recorded as rows).
4. **(LOW, standing relapse tripwire) Faint late value re-sharpening.** turn-0 |Q| 0.014(ep13)→0.056(ep16); early|q| 0.049(ep9 trough)→0.109(ep18); value_head bias edge-mass growing/accelerating (0.064→0.290). All ~14× below broken, early-sat ~0, non-monotone (ep18 turn-0 dropped). Tripwire = early|q| past ~0.30 or value loss starting a sustained fall.
5. **(LOW–MEDIUM) Midgame more committal than main_3** (greedy-fraction ~0.21 vs ~0.10, mildly rising). By design for conversion; confirm it asymptotes.
6. **(LOW, by design) soft_policy out-weighs primary policy** (~36% vs ~31% weighted, every epoch). Not harmful, but not "fully balanced."
7. **(LOW, chronic) moves_left near-flat learner** (Δ−0.06 over 18 epochs, largest single raw component) — passes audit, functional, same in main_3.
8. **(LOW, benign) First-mover-advantage erosion** (decisive P0 win early 0.537 → late 0.470, toward symmetry). Watch it doesn't deepen past ~0.43.

---

## 8. RECOMMENDATIONS

**Immediate (no action, just observe):**
- **The ep20 multistage eval is the decisive next test** of whether strength continues past replay-window/step-budget saturation. If it confirms continued climb, no action needed.
- **Re-decode the saturation guardrail at ep20–25** (turn-0 |Q|, early-game mean|q|) to confirm the faint late re-sharpening stays bounded.

**If strength plateaus at/after ep20 (levers in order of leverage):**
1. **Small lr decay toward 2e-4** — aligns with main_3's held-out lower-is-better finding; the natural late-run annealing the constant 3e-4 currently lacks.
2. **Raise train_samples_per_epoch** to cut reuse below ~5x and feed more fresh-row coverage per epoch (addresses the fixed-budget plateau directly).
3. **Trim soft_policy_weight toward 0.7–0.8** so the primary policy head leads the objective.

**Throughput tradeoff (the one experiment that resolves an inconclusive question):**
- **The fast=192-vs-128 (pcr_fast_visits) bet is INCONCLUSIVE.** It costs ~14% throughput, but its claimed trajectory-quality benefit cannot be measured because fast moves are never recorded as training rows. **Resolving experiment:** a one-time A/B running a few epochs at pcr_fast_visits=128, measuring **Elo per wall-clock-hour**. If 128 matches 192's strength-per-hour, drop to 128 and reclaim ~14% throughput (lifting epoch cadence past ~21/day and enabling faster lever iteration).

**Cheap observability adds (process, not health — none block the verdict):**
- Emit per-epoch turn-0 |Q| / early-game mean|q| to the selfplay diagnostics JSON (the saturation guardrail currently requires manual npz decode; root_value_mean alone is non-discriminating — it stayed ~0 in BOTH the broken and fixed runs).
- Add per-seat win-rate / draw-rate to the selfplay block (first-mover erosion currently only visible via decode).
- Startup assertion `'value_mask' in rust_result` to prevent a silent stale-.so regression to the np.ones fallback (the cell_q/value masking correctness depends entirely on the FIXED rust kernel).

---

*Key files: per-epoch metrics `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4/diagnostics/hexfield.{training,selfplay}.epoch_0000NN.json`; live `…/hexfield.selfplay.live.json` (ep19); guardrail decode over `…/samples/epoch_0000NN/*.npz`; loss reconstruction `packages/hexfield/python/hexfield/losses.py:282`; weight pass-through `trainer.py:555-577`; value-mask fix `replay_expand.rs:645-655`; broken-run archive `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4_pre_fix_saturated_ep19`.*
