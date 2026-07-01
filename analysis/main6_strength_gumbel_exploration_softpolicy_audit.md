# main_6 Gumbel-AlphaZero Deep Audit: Strength, Gumbel Benefit, Exploration, Soft-Policy Head

**Date:** 2026-07-01 | **Run:** hexfield main_6 (full Gumbel AZ) @ ~ep21-22 (~14h) | **Net:** c=128 (~2.9M params), radius 4 | **Warm-start:** main_5 epoch_000105

Scope: synthesis of four independently-verified audit dimensions. Every quantitative claim below was reproduced by a verifier (fresh probes, source re-reads, or diagnostics recomputation); where a verifier *refuted* or downgraded an analyst claim, that is stated explicitly. Per-point pooled-BT `se_elo ~= 75` is the dominant statistical constraint on every strength conclusion.

---

## 1. Bottom line

- **main_6 is healthy and clearly stronger than main_4**, but "stronger than its warm-start parent main_5" is **currently unmeasured** — no eval file through ep20 contains a main_5 player. The direct, drift-immune h2h vs main4_ep60 is a genuine reversal (main_6 76.1%, +202 Elo, vs main_5's 34.6% losing record), and value/entropy/length probes show **no pathology** (root_value_mean ~-0.016, no saturation; truncated_games ~0).
- **Gumbel is NOT yet earning its keep as a variance-reducer.** The central hypothesis that motivated the whole run — "the completedQ target is lower-variance than the visit target" — **resolves NO at 1024 visits**: the visit-target argmax is already stable (bootstrap-flip ~4%, not the ~50% the AZ plan assumed from main_5's mixed 512/1024 budget). The Gumbel target is *sharper* (H~0.37 vs visit ~1.2) but **no better aligned with the net** (top1-agree indistinguishable, ~0.45-0.52), so sharpness buys extraction ease, not correctness.
- **The pooled +225-over-main4 lead is almost entirely inherited** from the main5_ep105 warm start; drift-corrected earned rise over ep5->20 is only **~+21 Elo (< 1/3 of one se_elo)** — statistically indistinguishable from flat, with the raw h2h trend even sloping slightly *down* (noise-dominated).
- **Two live-config facts confound every Gumbel-benefit attribution and must be settled first:** the AZ-plan Stage-0 exploration preconditions (dirichlet 0.25->0.20, root_fpu 0.2->0.0) were **never applied** in main_6, and dynamic c_puct is **inert** (c_scale=0.0). One nuance the analyst missed: dirichlet 0.20 *was* live in main_5, so main_6's 0.25 is a **regression from the parent**, not merely an un-applied intent.
- **Single highest-value change:** a dedicated **high-budget (200+ paired-game) gauntlet of cand_ep20 vs main5_ep105 directly** — it is the only way to separate inherited from earned strength and it is the load-bearing gap in the entire run's evaluation. Everything else (the soft-head A/B, the c_scale sweep) is secondary and underpowered at current eval SE.

---

## 2. Strength — improving or inheriting?

**Verdict: healthy, decisively above main_4, but earned improvement is indistinguishable from flat and the parent comparison is missing.**

### What is solid
- **Drift-immune h2h vs main4_ep60 = 67-21 of 88 decided = 76.1%** (Wilson [0.663, 0.838]), implied +201.5 Elo. Per-epoch: ep5 15-5 (75%), ep10 27-5 (84%), ep15 16-4 (80%), ep20 9-7 (56%). This is a **clean reversal** of main_5's 120/349 = 34.6% (Wilson [0.296, 0.395], ~-112 Elo). *Caveat:* this h2h runs at eval_visits=128, not the 1024 self-play budget, and each edge is only 16-32 games — "decisively" rests on the **aggregate**, not any single epoch.
- **No collapse.** Self-play proxies are flat/slightly declining, not degenerate: mean_game_length slope -0.30/ep (t=-1.83, band 66-88, ep20 dip 66.1 rebounding to 79.8 @ep21); root_policy_entropy slope -0.0098/ep (t=-4.58, 1.42->1.21, a mild stabilizing decline, not narrowing); root_value_mean slope +0.0001/ep (t=0.79, flat ~-0.016, **no saturation** — contrast main_4's 0.79); truncated_games ~0-1/256.

### The confounds (all CONFIRMED)
- **The lead is inherited.** cand_ep5 already sat at pooled **380.6 Elo**. Re-rating the *same* ep5 checkpoint against the growing pool drifts it 380.6 -> 420.8 -> 444.7 -> 444.7 (**+64.1 purely from pool composition**); fixed main4_ep60 drifts 188.2 -> 173.0 -> 199.3 -> 241.1 (**+52.9**). cand_ep20 = 465.8. **Drift-corrected earned rise = 465.8 - 444.7 = +21.1 Elo** (or +32.3 using the main4_ep60 drift reference). Both are **well under se_elo ~75** => indistinguishable from flat.
- **Trend actually points slightly down.** h2h logit slope = **-0.057/ep**. This half is *real in the point estimate but noise-consistent*: it is dominated by the ep20 point (56% on only 16 decided games after the budget dropped from 20/32; Wilson [0.332, 0.769] overlaps all prior epochs).
- **No parent comparison exists.** grep of all four multistage_eval files returns **zero main_5 players**. main5_ep105 is configured (TOML:465) but inactive through ep20; it enters the pool only at effective ep22+, and then only as a low-power pooled anchor (~16-32 games/edge, se~75), **not** the decisive gauntlet needed.
- **Anchor pool is noisy.** A *fixed* checkpoint swings ~64-68 Elo purely from pool composition (main4_ep60 range 68.1; cand_ep5 range 64.1). Report drift-corrected, fixed-reference Elo, never raw pooled.
- **SealBot is saturated** (58/60/61/60 wins of 64; CI lower bound >=0.81 from ep5) — no discriminating power; it only pins the 0 scale.
- **Every per-eval verdict is INCONCLUSIVE** at games_budget=128 (elo_diff_ci95 straddles 0 every epoch; the primary candidate-vs-champion verdict is *structurally* single-epoch-limited per the in-file note).

**Net:** main_6 is a real, healthy run that has clearly surpassed main_4. Whether it has *learned* anything since warm-start, or merely inherited main_5's strength and held it, is **not answerable from the current evidence.**

---

## 3. Gumbel benefit — does it help, and how?

**Verdict: implementation is faithful and safe; the variance-reduction rationale does NOT hold at 1024 visits; benefit is unproven and its mechanisms are largely off-axis for this regime.**

### The target-variance question is resolved: NO (at 1024 visits)
The AZ-plan premise ("completedQ removes ~50% target-argmax variance") came from main_5's *mixed 512/1024* budget. At main_6's flat 1024:
- **visit-target argmax bootstrap-flip = 4.0% (ep6) / 2.0% (ep12) / 4.1% (ep20)** — the visit target is already argmax-stable, so completedQ has almost nothing to remove.
- Controlled for the fast-PCR confound (fast rows are 192-visit): restricting to gumbel-valid Full rows gives 3.5%/3.8% — **essentially identical**, because ~99% of stored non-empty-policy rows are Full.
- Median top1-top2 margin = 0.42; near-ties (margin<0.05) only 19-23%.

**The single OPEN question from the established facts is therefore closed: the "lower-variance teacher" is not delivered by Gumbel here.**

### What Gumbel actually changes
- **Sharper, not better-aligned.** Gumbel target H = 0.37 vs visit H ~1.2; gumbel top1-mass 0.87 vs visit 0.56-0.63. But **top1-agree(target vs net argmax) is statistically indistinguishable**: gumbel 0.44-0.48 vs visit 0.48-0.52. Sharpness != correctness.
- **Easier to fit; plateaus immediately.** Per-row CE(net, gumbel) < CE(net, visit) (direction robust across normalizations). loss_policy = 1.87(ep5) -> 1.78(ep10) -> 1.76(ep15) -> 1.77(ep20), **flat after ep10**. CE decomp H(gumbel)0.38 + KL~1.4 ~= live 1.77 checks out.
- **Distinct signal touches ~2% of moves.** Gumbel overrides the visit argmax to a *different* move 67-72% of the time, but only on the ~7% of Full rows where the visit target is jittery (flip>0.15) — and Full rows are only ~33% of moves. => ~7% x 33% ~= **2% of moves carry Gumbel's distinct contribution**, and eval se~75 cannot see it.
- **Mechanism load-split:** SH concentrates visits only *modestly* (visit_nnz 9-17 of m=32); **completedQ does the real concentration** (gumbel_nnz well under m either way). The low-sim tools #1 (Gumbel-Top-k root sampling) and #2 (Sequential Halving) are **off-axis / weak fit** for a 1024-visit budget per the suitability doc.

### Implementation & safety (strongest finding)
Faithful to Danihelka, verified in source: `gumbel_sigma=(c_visit+max_n)*c_scale*q` (tree.rs:2226-2227); `v_mix=(v_node+n*visited_avg)/(1+n)` (tree.rs:2305-2311); target = softmax(logit + sigma(completedQ)) with degenerate fallback (search.rs:2914-2950); SH top-m by logit+g with TSS force-include added to budget, keep=(n+1)/2 (tree.rs:751-940); force_stuck_gumbel + fallback_root_action guards present. **f1916863 SH bugs leave no residual; value-head saturation (risk #3) has NOT triggered** (root_value_mean -0.014..-0.022 across ep5-21).

### Is target-only viable?
**Yes as an experiment, moderate confidence, not as a proven simplification.** #1/#2 are low-value at 1024 visits and the visit target is already argmax-stable, so a target-only fork (keep gumbel_target=1, revert search to plain PUCT) is the right way to isolate #3. Two caveats: eval se~75 means any small #1/#2 benefit is invisible unless the fork runs long with pinned anchors; and the warm-start confound already prevents attributing current strength to #1/#2/#3 individually.

---

## 4. Exploration — config intent vs live, and diversity

**Verdict: Dirichlet is inert on every recorded policy target; two intended exploration levers were never applied; measured diversity is healthy and not collapsing (though the specific opening-diversity magnitudes are unverified).**

### Config intent-vs-live discrepancies (state plainly)
| Lever | Comment/intent | LIVE (manifest) | Applied? |
|---|---|---|---|
| root_dirichlet_noise_fraction | 0.25 -> 0.20 (AZ-plan Stage-0) | **0.25** | **NO** — and 0.20 *was* live in main_5, so 0.25 is a **regression from parent** |
| root_fpu_reduction | -> 0.0 (AZ-plan Stage-0) | **0.2** | **NO** |
| gumbel_c_scale (dynamic c_puct) | intended dynamic | **0.0** (inert), c_puct 1.5 flat | dynamic c_puct **OFF** |
| gumbel_m | dataclass default 16 | **32** (toml:320 explicit override) | **intentional** (Hexo branching 337->~1000 grows late-game; m=32 so SH never starves) |
| pcr_full_proportion | — | **0.33** | live |
| soft_policy_weight | 4.0 -> 1.0 (5:1-domination fix) | **1.0** | applied |

**Nuance the exploration-dimension verifier surfaced that the strength/gumbel verifiers agree on:** one reading (config-comment-final-note) says the dated `2026-06-29: 0.20 -> 0.25` note means 0.25 is *intended* and nothing was silently dropped; the AZ-plan reading says these are Stage-0 preconditions that were supposed to precede judging Gumbel. **Both are true and not contradictory:** the *comment* matches live, but the *AZ-plan Stage-0 sequencing* was not run. Net practical fact: **you are judging Gumbel's benefit while its intended exploration baseline (main_5's 0.20 dirichlet, 0.0 fpu) is not set.**

### Dirichlet is dead as a policy-target influence (CONFIRMED, code-level)
- On **Full moves** the Gumbel-Top-k/SH root reads **raw un-noised logits** (root_logits "never temperature-shaped, normalized, or Dirichlet-noised", tree.rs:322; gumbel-root branch tree.rs:1110-1111 -> SH allocation tree.rs:1364-1403 by visit-deficit only, never edge.prior). The intermediate PUCT-among-survivors branch that *would* read prior is **dead** (gumbel_sequential_halving=true live).
- On **Fast moves** no Dirichlet is applied (noise_for=None, search.rs:144-149).
- **Recorded rows @ep20:** 16746 total, policy_valid=1 = 5494 (32.8% Full), Fast = 11252 (67.2%, zeroed in policy CE via `_pol_weight = policy_ce_weight * policy_valid`, losses.py:244-246); gumbel_valid = 5466 (99.5% of Full rows carry a gumbel target). **=> Dirichlet touches ZERO recorded policy/gumbel targets.** The "dirichlet inflates target variance" hypothesis does not apply to main_6.
- *File-citation correction:* the analyst's `search.rs:1106-1112` was wrong (unrelated async code); the real branch is tree.rs:1110-1111. Conclusion unaffected.

### Diversity: not collapsing (but magnitudes unverified)
- **Not-collapsing half is solid:** truncated_games=0 every epoch, mean_game_length wanders 66-84, root_policy_entropy 1.42->1.21 (mild decline, stabilizes ~1.2).
- **Specific opening-diversity numbers are UNVERIFIED:** the "244/250 unique first-4 / entropy 5.45" figure is not reproducible from any obvious field; an independent argmax-of-visit-policy proxy gave only 81/256 unique / 3.31 nats. Treat the *magnitude* as unconfirmed; the *direction* (not collapsing) stands.
- **Caps are slack:** widening_max_children=96 binds in only 0.53% of Full rows (median Full support 8, p90 32). gumbel_m=32 mostly non-binding but *closer to saturating than the analyst said* — median gumbel target support 8 (not ~4), fraction>=30 = 10.85% (not 6-10%).
- **Residual policy loss (1.77) is a target-extraction property, not an exploration deficit.** Gumbel KL-from-prior 1.43 > visit KL 0.85, but gumbel H 0.36 << visit 1.19. *Correction:* the analyst's finding-#7 quoted KLs (visit ~0.48, gumbel ~1.06-1.16) are **inaccurate**; the correct, reproduced values match the established probe (visit 0.71-0.85, gumbel 1.4-1.5, top1 ~0.50/0.41-0.45). The conclusion is right; those specific cited numbers were wrong.

---

## 5. Soft-policy head — help or hurt?

**Verdict: mildly helpful-to-neutral, NOT a capacity conflict. The head's core "directional conflict" framing is REFUTED by the gradient data.**

### The structural mismatch is real...
- Soft target is derived **purely from the VISIT policy** (batching.py:164-173: `soft = p.pow(0.5)` on visited support), **never** the Gumbel target. With policy_target='gumbel', the main head trains toward gumbel (losses.py:253-267) while the soft head trains toward visit-softness (losses.py:324-331). So the two heads target *different* distributions: **KL(soft_target || gumbel_target) ~= 6.9 (ep20) to 9.0 (ep5)**, vs only ~0.18-0.21 from the visit target it is built from. H_soft 1.6-1.8 vs H_gumbel 0.34-0.37.

### ...but it does NOT produce a trunk-gradient conflict (this is the key finding)
- **cos(g_soft_trunk, g_gumbel_trunk) = +0.78 (STRONGLY ALIGNED)**; projection of the soft gradient onto the main gumbel descent direction = **+1.35 (POSITIVE = reinforces)**. For comparison cos(g_opp, g_gumbel) = +0.14. The large *target-distribution* KL (6.9) does **not** translate into opposing trunk gradients — both losses push the trunk to raise logits on the same top-region moves.
- **=> The analyst's core framing ("first-order capacity/direction conflict", "pulling the trunk toward stale visit-softness while the main head is pulled to a sharp Gumbel target", "not a mild regularizer") is REFUTED.** The soft gradient is a largely-aligned, moderate-magnitude aux gradient.
- Magnitudes (reproduced, but re-interpreted): trunk grad-norm soft_w1.0 = 1.736 = 72% of policy_gumbel 2.415. That is *magnitude*, not *conflict* — and because it is +0.78 aligned, most of it reinforces the main objective.

### Also established
- **LIVE soft_policy_weight = 1.0**, not 4.0 (the brief was stale). Already trimmed from 4.0 with the documented 5:1-domination rationale (toml:394), matching main_5. At the old 4.0 the trunk grad-norm would have been 6.945 (2.9x main head) — *that* would have been domination; 1.0 is not.
- Head is **learning, not saturated:** loss_soft 2.55(ep1) -> 2.24(ep19) -> 2.27(ep21), tracking loss_policy; residual KL to floor ~0.69.
- **Not a redundant copy** distributionally (H_softhead 2.5-2.7, flatter than the sharp main head) — BUT the analyst's argmax-divergence claim is **REFUTED**: top1(soft_head vs main_head argmax) = **0.87/0.84**, not the claimed 0.56/0.52. The heads agree at the argmax ~85% of the time; only the tails differ.
- grad_norm_heads (~1.13) lumps all heads (trainer.py:169-190) and cannot indicate soft-head-specific harm — the per-loss trunk-grad probe is the correct instrument.

**Net:** the soft head is a mild, mostly-aligned regularizer at w=1.0. There is no evidence it is hurting. Any A/B should be run as **hygiene/curiosity**, not as resolution of a conflict the gradient data says is not there.

---

## 6. Ranked evidence-backed tweaks

Ordered by (evidence_strength x expected_impact). REFUTED tweaks dropped (see "Rejected" below).

### Do now — high confidence, no run risk
| # | Change | Measured evidence | Expected effect | Risk | How to validate |
|---|---|---|---|---|---|
| 1 | **Report drift-corrected candidate Elo** (fixed-reference), never raw pooled | 64.1 of the 85.2 raw ep5->20 "rise" is same-checkpoint cand_ep5 re-rating up; fixed main4_ep60 drifts +52.9 | Removes an artifact that overstates learning ~3-4x | None; reporting-only | Recompute curve vs a pinned reference; **state the reference explicitly** (cand_ep5 -> +21; main4_ep60 -> +32) |
| 2 | **Set the deferred Stage-0 exploration levers intentionally**: dirichlet 0.25->0.20, root_fpu 0.2->0.0 (between epochs, not mid-epoch) | manifest dirichlet 0.25 / fpu 0.2 vs AZ-plan §69/§244 Stage-0 = 0.20/0.0; **0.20 was live in main_5** => 0.25 is a regression | Restores parent baseline so Gumbel-benefit attribution is clean | Small perturbations, effect itself within se~75 (may be unresolvable) | Fork; compare loss_policy floor + pooled Elo over 10-15 ep vs control |

### A/B on a fork — moderate confidence
| # | Change | Measured evidence | Expected effect | Risk | How to A/B |
|---|---|---|---|---|---|
| 3 | **Target-only Gumbel A/B**: keep gumbel_target=1, disable gumbel_root/SH/nonroot_select (plain PUCT search) | #1/#2 off-axis at 1024 visits (suitability §5); SH concentrates only modestly (visit_nnz 9-17/32); visit target argmax-stable (flip ~4%) | Isolates whether ANY strength comes from #1/#2 vs #3; likely simplifies with no strength loss | eval se~75 hides small #1/#2 benefit unless fork runs long with pinned anchors; warm-start confound already blocks per-mechanism attribution | Common checkpoint; ~10-15 ep; pinned anchors incl. main5_ep105; metric = pooled BT + loss_policy floor |
| 4 | **Raise pcr_full_proportion 0.33 -> 0.50** | only 32.8% of moves carry a weighted policy target; other 67.2% zeroed in policy CE | Denser Gumbel-target coverage (the actual policy-learning channel) | Higher self-play cost (1024 Full vs 192 Fast), cuts value-row volume; residual loss driven by target *variance/KL* which more targets won't reduce; **throughput-gated** | Fork; hold wall-clock/throughput fixed; watch loss_policy floor + value-row count |
| 5 | **Soft-head A/B/C** (w=1.0 / 0.25 / 0.0) as hygiene | grad-norm soft 1.736 (72% of main) BUT cos(g_soft,g_gumbel)=+0.78 aligned | Confirms soft head is neutral/mildly-helpful; not expected to lower loss_policy floor | Rationale ("conflict") is refuted; underpowered at se~75; run as curiosity not resolution | Common checkpoint; ~10-15 ep; metric = pooled BT + loss_policy floor; guardrail = train entropy |

### Measure first — the load-bearing gap (do this before trusting any strength number)
| # | Change | Measured evidence | Expected effect | Risk | How to run |
|---|---|---|---|---|---|
| 0 | **High-budget gauntlet cand_ep20 vs main5_ep105, 200+ paired games** | 0 of 4 eval files contain a main5 player; "stronger than parent" is UNMEASURED | Separates inherited from earned strength; paired => ~2x lower variance than pooled BT | Compute cost; pick a fixed eval_visits (ideally test both 128 and 1024) | Direct paired match; report win% + Wilson + implied Elo; repeat at each milestone |
| 6 | **Raise eval games_budget 128 -> 256-320** (per-anchor ~48-64) OR **enable SPRT** (currently {enabled:false, elo1:-50, max_games:64}) | all 4 verdicts INCONCLUSIVE; ep20 edge collapsed to 16 decided (drove the "trends down" read) | Halves pooled-difference SE (~75->~53) on the descriptive/anchor curve | Primary champion verdict is structurally single-epoch-limited => budget helps the anchor curve more than the primary verdict; sqrt-N gain depends on edge allocation | Raise budget in eval config; or flip sprt.enabled |
| 7 | **c_scale sweep 1.0 / 0.3** (flatten sigma so gumbel target less near-one-hot) | gumbel H 0.38, top1 0.87; loss_policy plateau by ep10; AZ-plan §217 | *Unknown direction* — if residual is irreducible positional entropy (~1.25 = the visit target's H), flattening just walks the gumbel target back toward the already-available visit target, recovering nothing | Genuinely a sweep-to-find-out; may collapse to prior | Grid 1.0/0.3 on a fork; watch top1-agree(target vs net) — the metric stuck ~0.45-0.52 |

---

## 7. Open questions / what to measure next

1. **Is main_6 beating its own parent?** (Tweak 0.) The single unanswered question. Direct 200+ paired-game gauntlet vs main5_ep105 at both 128 and 1024 eval_visits. Until this runs, "main_6 improved on main_5" is unsupported.
2. **Does the "trends down" h2h slope survive more games?** It rests on the 16-decided-game ep20 point. Re-run ep15-22 at budget 256-320 and recheck the logit slope sign.
3. **Prove or kill the lower-variance-teacher hypothesis directly.** The bootstrap-flip test (visit ~4%) already suggests NO at 1024 visits. Add a D6 self-consistency / rollout-variance test *on the completedQ target itself* to close it definitively (the established "entropy != variance" caveat).
4. **Does any strength come from #1/#2 (root sampling + SH) at all?** Target-only fork (Tweak 3). If Elo is indistinguishable, simplify to target-only PUCT and reclaim self-play compute.
5. **Reconcile the exploration-target discrepancy against parent.** dirichlet 0.20 was live in main_5, 0.25 in main_6 — intentional 2026-06-29 change or drift? Confirm with the owner, then set Stage-0 levers deliberately (Tweak 2).
6. **Is the top1-agree plateau (~0.45-0.52, stuck like main_5's 0.54) capacity-invariant irreducible positional entropy?** If yes, no target-sharpening lever (c_scale, m) can move it, and the bottleneck is fundamental to the position distribution, not the extraction method. The c_scale sweep (Tweak 7) is the cheapest probe.
7. **Verify the opening-diversity magnitudes.** The "244/250 unique / entropy 5.45" figure is unreproducible; an independent proxy gave 81/256 / 3.31 nats. Locate the authoritative first-move field before citing opening diversity in any decision.

### Rejected (REFUTED by verification — do not pursue on the stated rationale)
- **Rebuild the soft target from the gumbel target (gumbel^0.5, support-only)** — premise ("eliminate directional conflict") is false: cos(g_soft, g_gumbel)=+0.78, no conflict exists. Gumbel is near-one-hot (H~0.37, median-0 single-support on many rows), so a support-only gumbel^0.5 target would be near-degenerate and likely *destroy* the entropy-floor regularizer role. Requires an out-of-scope batching.py change for a non-existent benefit.
- **Lower gumbel_c_scale toward 0.3 as a *fix*** — kept above only as a *measure-first sweep* (Tweak 7), NOT an evidence-backed fix: if the residual is irreducible positional entropy, flattening recovers nothing and may collapse to prior.
- **Reduce gumbel_m 32 -> 16** — m=32 is a *deliberate, reasoned* toml override (Hexo branching 337->~1000 grows late-game; SH must not starve), not the "accidental doubling of the config default" the analyst framed. At fixed 1024 budget, larger m already risks thinner per-candidate visits => *more* completedQ noise, working against the variance goal. Weak evidence, mildly mischaracterized premise.

---
*Confounds threaded throughout: (a) warm-start — cand_ep5 already 380.6 Elo, so most of the pooled lead is inherited; (b) per-point se_elo ~75 — no 10-15-epoch Elo readout can resolve a mild aux head or small exploration perturbation; (c) games_budget=128 => every per-eval verdict INCONCLUSIVE; (d) two intended Stage-0 exploration levers never applied + dynamic c_puct inert, so no Gumbel-benefit attribution is currently clean.*
