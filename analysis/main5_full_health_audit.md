# hexfield_main_5 — Full Health Audit (@ ~ep102)

Date: 2026-06-29. Status: LIVE (~epoch 102), c=128 (2,916,133 params), BC warm-started from a c=128 prefit on main_4 ep67-77. CPU-only read-only analysis over all 102 epoch diagnostics JSONs, 20 multistage_eval files, and CPU-loaded checkpoints. Per-point multistage SE is large (~48-90 Elo); single-epoch deltas are treated as noise and only trends are load-bearing.

Every claim below survived independent adversarial verification. Refuted/corrected claims are marked inline and their numbers are restated to the verified value, not the original.

---

## 1. Executive summary

**Overall verdict: HEALTHY but PLATEAUED. Not capacity-bound, not pathological — strength-limited upstream of the model.**

main_5 is structurally and optimization-wise healthy across all 102 epochs: zero failed steps, clean gradient flow, calibrated value head, live/growing residual branches, and a complete absence of main_4's value-Q saturation death mode (root_value_mean stays in [-0.0189, -0.0003], abs-max 0.0189 @ep40, vs main_4's runaway to |Q|~0.5-0.79). It learned strongly early (candidate pooled-BT Elo +5.37/ep over ep5-30) then flattened (-0.84 Elo/ep over ep60-100, t=-0.79, indistinguishable from zero).

**The single most important finding: the 1.76x capacity bump did not pay off, and the run has plateaued at roughly parity with — not above — its predecessor main4_ep60.** Aggregate direct head-to-head over all 20 evals is 120/349 = 34.6% (Wilson CI [0.296,0.395], implied Elo ~-112), i.e. main_5 is over the full run significantly weaker than main4_ep60. There is genuine within-run convergence (h2h 22.3% @ep5-35 → 53.1% @ep85-100, a statistical tie), but no breakout. The train-loss floor barely moved for +1.76x params (loss_policy floor 1.998 @ep66 vs main_4's 2.086 @ep87 — only ~0.09 nats), and the wider net leaves proportionally MORE width idle (conv1 effective rank 78.6/128 = 61% vs main_4 69.3/96 = 72%). This is the signature of a net NOT bound by raw width.

**Top evidence-backed levers (in order):**
1. **Improve TARGET QUALITY, not row count (`pcr_full_proportion` 0.33 → ~0.5).** With train losses floored/U-shaped and strength flat, the binding constraint is the information content of the targets, not how many times they are trained. Raising the fraction of full-search (vs 192-visit fast) rows gives sharper policy/value targets per row. Config-only; the cheapest quality lever.
2. **Do NOT widen further.** +33% width bought only +13% effective dimensions and lowered the policy floor by ~0.05-0.09 nats; a wider net would idle more for no gain.
3. **Consider trimming soft_policy_weight.** loss_soft_policy is floored (~2.35, ~3.6% drop) and is the second-largest term in loss_total; with core strength stalled it is a low-yield head slot. (Inherited from prior audit; consistent with the floored-loss evidence.)

> **CORRECTION (2026-06-29, post-audit, owner-flagged + code-verified).** An earlier draft of this report named "raise `train_samples_per_epoch` (96k→142k) for data coverage" as lever #1. **That was wrong** and has been removed. Traced through `window.py`/`trainer.py`: the distinct-data pool is the recent-data **taper window** (`compute_katago_window_rows` → `desired_rows` ≈ 537k @ep100), set by the window config (`expand_window_per_row`, `taper_exponent`, `keep_target`) and self-play volume — **`tspe` is not an input to it.** Each epoch trains `effective_rows` = 96,000 **distinct rows, one pass** (`passes_per_epoch=1`; verified `steps=375 × batch_rows=256 = 96,000`), drawn as a *fresh random shard subset* (re-seeded `seed + epoch*65537`) of that window. `reuse_ratio = effective_rows / new_rows_this_epoch = 96000/22023 = 4.36` = the lifetime average number of times each *generated* row is pulled into training over its ~24-epoch window residence (~24 epochs × ~18% per-epoch draw ≈ 4.3). The "404k" cited earlier is `train_bucket_level`, a credit/throttle accumulator (`8 × new_rows`, capped 500k) — **not** the data pool. **Therefore raising `tspe` raises REUSE (4.36→6.45 at 142k), not coverage** — same fixed pool, more gradient passes. Overfit risk from that extra reuse is *low* (D6 augmentation, `trainer.py:442`, presents each row under a fresh random one of 12 orientations every epoch, so the exact input tensor is rarely repeated), but it adds **no new information**; whether more passes help at all depends on whether the net is underfitting the current pool's signal, which is unresolved. The genuine distinct-data levers are **self-play volume** (`selfplay.games_per_epoch`, GPU-bound) and the **window config**; the genuine quality lever is **`pcr_full_proportion`** (now lever #1).

---

## 2. Per-dimension findings

### Exploration — verdict: mostly-healthy
**Bottom line: exploration is structurally healthy and is not the strength bottleneck.**

- **No value-Q saturation.** root_value_mean over ep1-102 (ex-resume ep82): min -0.018914 @ep40, max -0.000294 @ep102, abs-max 0.0189, mean -0.0088, full-run slope -2.9e-5/ep (flat, t=-2.40). main_4's runaway to |Q|~0.5-0.79 is entirely absent.
- **Entropy narrows only slowly; no mode collapse.** Root-policy (post-search visit-count) entropy full-run slope -0.00177 nats/ep (t=-6.31, n=101); band [1.097,1.600]; total ~-0.18 nats over 95 epochs. ep5=1.600, ep50=1.305, ep85=1.413, ep100=1.225. Robust to dropping partial epochs (-0.00171, t=-6.01).
- **PCR mix rock-stable** at 33.03% full (min 32.10%, max 34.04%, std 0.32%), matching config pcr_full_proportion=0.33.
- **Early-stopping steady**, saving ~35-37% of full-search budget (ep1-30 mean 0.348, ep91-100 0.370), no runaway termination; lcb_overrides 0.53/full move.
- **Noise bump effect is unmeasurable.** The root_dirichlet_noise_fraction 0.20→0.25 bump and the search_visits 512→1024 doubling were committed together (_resume_config.toml mtime 2026-06-29 08:25:52; both first take effect ep101). Entropy did NOT rise after the bump — it fell to the run's two lowest values (ep101=1.097, ep102=1.142) — but this is dominated by the simultaneous sims doubling, which sharpens visit-count entropy. The noise effect cannot be isolated from these data.

**Corrections:**
- *[length-decline] PARTIALLY CORRECTED:* full-run mean_game_length slope is **+0.021/ep** (verified), NOT the claimed +0.054. Both are ~flat-positive; conclusion (mild recent decline, no collapse) unchanged. Confirmed: peak mgl 110.9 @ep75, ep100=88.1, ep102=92.9, last-20 slope -0.44/ep; p90 last-20 slope -0.84/ep; truncated last-5 mean 2.4/256.
- *[resume-artifacts] REFUTED on count:* there are **FIVE** non-256-game epochs (ep3=216, ep81=242, ep82=0, ep83=233, ep101=99), not two. Only 97 epochs have exactly 256 games. The analyst missed ep3/81/83. Trend conclusions unaffected (entropy slope robust at -0.00171).
- *[exploration-not-bottleneck] window figure CORRECTED:* the replay window (window_rows) is ~96k, not ~481k. The ~18-20% coverage figure is effective_rows (~96000) / desired_rows (~537000), i.e. fraction of the training-draw budget, not "20% of a 481k window." Conclusion (exploration not binding) stands.

---

### Learning / strength trajectory — verdict: mixed
**Bottom line: learned strongly through ~ep55, then a genuine (eval-noise-aware) plateau at roughly parity with main4_ep60.**

- **Fast early rise, then flat.** Candidate pooled-BT Elo slope +5.37/ep (ep5-30, mean 136.4) → -0.84/ep (ep60-100, mean 281.2, OLS se 1.07, t=-0.79 = indistinguishable from zero). ep55-100 mean 278.6 ± point std 39.0 (mean per-point se ~50.8).
- **Peak ep65 = 349.9 Elo; ep100 = 229.5 (se 49.5).** Peak-to-current drop 120 Elo < 2× combined SE — within the plateau band.
- **Has NOT decisively beaten main4_ep60 over the full run.** Aggregate direct h2h over 20 evals = 120/349 (physical wins) = 34.6% (Wilson CI [0.296,0.395], implied Elo ~-112); 50% outside CI. Pooled-BT cand−m4 gap negative at every eval except ep65 (+10.1, z=+0.16); max z = +0.16, never significantly positive.
- **But genuine within-run convergence to parity.** h2h: ep5-35 31/139=22.3% → ep40-65 38/99=38.4% → ep70-100 51/111=45.9% → ep85-100 34/64=53.1% (CI [0.411,0.648], 50% inside = statistical tie). Slope +0.37 pct/ep, corr(ep,wr)=0.667.
- **Training-side health stable, no pathology.** loss_policy 2.039→2.051, loss_value 0.571→0.572, loss_total 6.927→6.911 (ep45-55 vs ep90-102); grad_norm_mean ~3.4; mean_game_length 89→94.6. No main_4-style collapse.
- **Sealbot winrate corroborates.** corr(sealbot mid, cand Elo)=0.869; mid rose 0.659 (ep5-30) → 0.790 (ep60-100); last-5 evals [0.84,0.81,0.81,0.81,0.74] (flat/slightly declining).
- **main4_ep60 anchor drift (360→314) is a pooling/SE artifact**, not main_4 weakening (a fixed checkpoint's strength is constant; early SE ~124 vs late ~27). The artifact-free direct h2h still shows the convergence.

**Caveats / inferred:**
- *[BC-basin] PARTIALLY-CONFIRMED:* main_5 IS BC-warm-started on main_4 (config initialize_from = hexfield_main_5_prefit/checkpoint_epoch3.pt). The data is consistent with asymptoting to parity, but the *causal* claim that the warm-start anchored it near main_4 strength is an inference, not measured.
- Minor: the "121/349" figure is winrate-field-weighted; raw physical_wins is 120/349 (both 0.346). ep82 is a no-op epoch (0 games, mgl 0) excluded from all windows.

---

### Training / optimization — verdict: healthy
**Bottom line: optimization is stable and well-behaved; losses are floored and not the binding constraint.**

- **All 102 epochs completed, no instability.** status='completed' for all (top-level and training block); steps ramp 79→375 over ep1-8 then hold at 375; amp_scale uniq={32768,65536,131072} — no low-value NaN backoff.
- **Healthy gradient flow.** grad_norm_mean 3.248 (min 2.117, max 3.625); grad_norm_p95 ~5.01; clip_fraction mean 0.0202 (max 0.056, well below pathology). Per-group: trunk_conv 2.581, trunk_attn 1.755, heads 0.801 (heads/conv 0.31, stable ep1 0.805 → ep102 0.803 = near-converged smaller heads, not starvation).
- **Moving-target self-play signature, not under/overfit.** Fixed-anchor strength rose ~+180 Elo (cand_ep5=50.9 → cand_ep100=229.5) while absolute losses stayed flat; residual loss drops correlate negatively with Elo (corr: policy -0.780, value -0.639, total -0.839). (Caveat: cand_ep100=229.5 is a downward noise dip; the +180 endpoint is noise-sensitive but the upward trend holds.)
- **Reuse ~4x is not overfitting.** reuse_ratio ramps to ~4.0 by ep8-9 (mean 4.02); train_bucket_level steady 404000; train losses flat AND strength rises = the opposite of an overfit signature.
- **moves_left head never auto-disabled** (ml_auto_disabled=False, passed=True all 102/102); conv_spearman 0.789 @ep50 → 0.816 @ep102; overall_mae 36.94 @ep1 → 28.76 @ep102.

**Corrections:**
- *[loss-floored-flat-from-ep1] CORRECTED:* major losses follow a **U-shape**, not a flat floor from ep1-3. They RISE to a peak ~ep10 (policy 2.04→2.15, total 7.06→7.26), DECLINE to a minimum ~ep66 (policy 1.998, total 6.801), then RISE again to ep102. Endpoint deltas (policy 1.7%, value 1.0%, opp_policy 0.34%) are real but mask the dynamics.
- *[stvalue/cell_q still learning late] CORRECTED:* stvalue_2/6/16 and cell_q are the largest-improving heads over the FULL run, but they peaked (stvalue @ep85, cell_q @ep25) and REGRESS by ep102 — they are NOT improving late. (stvalue_2 2.412@ep85 → 2.509@ep102; cell_q 1.759@ep25 → 1.865@ep102.)
- *[dead-unit attribution] REFUTED:* near-zero Adam second moment is concentrated in **WEIGHT tensors** (conv trunk, reductions, head matrices), NOT in bias/LN/gamma vectors as claimed. Coordinate-weighted, ~100% of near-zero coords (493141 vs 1030) live in multi-D weights; high-near-zero tensors are conv_blocks.*.conv{1,2}.weight (30-59% near-zero), reduction weights (50-64%), head weights. This is plausibly STRUCTURED sparsity from board geometry (the model keeps improving, so not collapse) — but the attribution was backwards. The cited update-ratio numbers also do not reproduce: per-tensor median is **0.122** (not 0.15), min **0.0** (not 0.002).

---

### Architecture — verdict: healthy
**Bottom line: the c=128 trunk is structurally healthy and still learning; the plateau is data/capacity-bound, not an architecture defect.**

- **c=128, 2,916,133 params** (verified): stem + 8 ConvBlocks + 3 AttnBlocks (4 heads, head_dim=32) + 3 per-block rel-pos bias tables (237×4) + 8 heads. tokens (8,128).
- **Every LayerScale branch is live and growing.** All 14 gammas monotonically increase ep20<ep60<ep100 and sit 282×-2037× their 1e-4 init. ep100 mean|g|: conv blocks 0.071-0.186, attn ls_mlp 0.105-0.204, ls_attn 0.028(attn0)-0.098(attn1). Optimizer exp_avg_sq nonzero for all (frac<1e-12=0.000).
- **Weakest branch is attn_block.0.ls_attn** (mean|g| 0.0282, 282× init, still actively updating) — first attention block underweighted but not dead.
- **No dead/saturated conv or q/k/v units.** Conv out-channel L2 frac<0.1=0.000 across stem/conv0/conv4/conv7/policy_conv; attn q/v_proj per-unit frac<0.1=0.000 all 3 blocks; LayerNorm gammas frac|<0.05|=0.000.
- **Residual sparsity = specialization, not collapse.** ep100 fracDead(<1e-3) of |ln2.gamma|·|ls.gamma|: conv2=0.492, conv5=0.367, conv7=0.359, conv6=0.328 vs conv1=0.094. Surviving channels strengthen (conv2 mean gain 0.049→0.056→0.067; conv6 0.075→0.097→0.121).
- **Weights maturing, not frozen.** drift(20-60) > drift(60-100) for every trunk layer (e.g. conv0.conv1 0.258>0.220; attn2.fc2 0.405>0.301; value_reduction highest late 0.436). No explosion.
- **Head norms sane** and scale with target type: value_head 131.7 (65-bin softmax, expected), stvalue 38-42, moves_left 44.7, cell_q 31.2; policy_head 0.94 / opp_policy 1.79 / soft_policy 0.95 (1-output readouts).

**Corrections:**
- *[grad-balance / amp_scale "steady 65536"] CORRECTED:* amp_scale is NOT pinned at 65536 — it spans 32768 (ep10) to 131072 (max), settling at 65536 only in the second half. Normal AMP loss-scaling adaptation, not a pathology; all grad-flow numbers otherwise confirmed.
- *[residual-sparsity "STABLE"] CORRECTED:* the dead-channel set is high-OVERLAP but mildly GROWING, not perfectly stable (conv2 54→63, conv5 35→47, conv7 36→46 from ep20→100; overlap ~83-98%). "Specialization not collapse" still holds (overlap dominates, mean gain rises).
- Minor: policy_conv min out-channel norm is 0.634 (slightly below the cited 1.9-5.1 range) but still frac<0.1=0; late Elo band is ~230-350 (ep65=350), wider than the cited 230-310; policy loss bottomed 2.031 @ep50 and ticked up to 2.052 @ep100.

---

### Value head — verdict: healthy
**Bottom line: the value family is calibrated, near its honest aleatoric floor, and is NOT a strength bottleneck.**

- **loss_value flat and honest.** first(ep1)=0.5860, min 0.5526 @ep60, last(ep102)=0.5680; full-run slope -0.000154/ep; ep51-102 slope +0.000132/ep. exp(-0.568)=0.567 prob mass on the correct outcome bin vs no-info ln(65)=4.17. Target is hard z=±1 (soft_z_lambda=0) → theoretical min CE = 0, so the ~0.55 sit is genuine aleatoric positional uncertainty, not a saturated/smoothed signature.
- **No value-Q saturation** (same data as Exploration): root_value_mean in [-0.0189 @ep40, -0.0003 @ep102], abs-max 0.0189; never near main_4's |Q|~0.5-0.79.
- **moves_left head healthy and improving, never auto-disabled** (102/102 pass): full_spearman 0.728→0.901, conv_spearman 0.528→0.816, near_end_mae_0_5 11.07→2.22, overall_mae 36.94→28.76; all gates cleared.
- **stvalue heads still actively learning in the back half** (value family not frozen): ep51-102 slopes stv2 -0.00308, stv6 -0.00280, stv16 -0.00235 — steeper than full-run.
- **cell_q fit early, modest contribution.** loss_cell_q 1.975→1.865, min 1.759 @ep25; weighted (×0.1) only 0.187 of loss_total ~6.85.
- **value_head weights still moving** (not frozen): value_head.weight norm 94.87→131.89, bias 25.39→52.40, value_reduction.weight 18.26→23.96; per-epoch reldelta slowing (ep80-100 0.00092) but nonzero.
- **value_mask plumbed end-to-end and active** (samples.py:304 → batching.py:185-186/242 → shards.py → losses.py:263-266; rust replay_expand.rs:166 emits value_mask:f32). Truncated rows are few so masking matters little but is applied.

**Corrections (neither affects verdict):**
- Max loss_value is at **ep22** (early), not "after ep60"; value 0.5979 correct.
- Overall truncated fraction is **1.58%** (not 1.56%); max per-epoch is **5.86% @ep75** (not 5.54%). Both low.

---

### Capacity — verdict: mixed
**Bottom line: the 1.76x capacity bump did NOT pay off; capacity is demonstrably NOT the binding constraint.**

- **Capacity bump real and confirmed.** main_5 2,916,133 params (c=128) vs main_4 1,656,453 (c=96) = 1.76×, at 1.33× width.
- **Train-loss floor barely improved.** With identical loss weights (policy=value=soft=1.0 in both → loss_policy directly comparable): main_5 loss_policy min 1.998 @ep66 vs main_4 2.086 @ep87 = only 0.088 nats. Like-for-like last-30 tail: 2.051 (std 0.010) vs 2.099 (std 0.008) = 0.048 nats. Both hard flat underfit floors.
- **Did NOT match main_4's strength ceiling.** Within main_5's own BT pool (directly comparable), cand_epNNN − main4_ep60 averages -68.9 Elo over ep45-100; only ep65 nominally positive (+10.1, within ~52 Elo single-rating SE). Slope ep45-100 ~+0.9 Elo/ep (flat). main4_ep60 anchor 314-360.
- **Wider net leaves proportionally MORE width idle.** conv1 participation-ratio effective rank: main_5 78.6/128 = 61% vs main_4 69.3/96 = 72%; absolute eff-dim grew only 69.3→78.6 (+13%) for +33% width. LayerScale gamma<0.01 fraction: main_5 0.254 vs main_4 0.173. Signature of a net NOT bound by raw capacity.
- **No dead conv channels** — idle capacity is graceful underuse, not BN/init/dead-ReLU failure: zero channels below 0.001×mean across all 8 conv blocks.
- **Lower data coverage in main_5.** tspe 96000 (vs main_4 142000); ~24% of the 404k bucket trained per epoch at healthy ~4x reuse — coverage, not capacity, is the more binding lever.
- **Underfit read is INFERRED, not measured** — there is no held-out/validation loss in the diagnostics (regex 'val_' returns nothing); the read rests on flat train-loss floors + healthy reuse without divergence.
- **Confounds honestly flagged:** main_5 differs from main_4 in lr (4e-4 vs 5e-4) AND tspe (96k vs 142k), so the pure-capacity attribution is not perfectly clean.

**Correction:**
- *[no-dead-conv-channels] magnitudes CORRECTED:* the cited norms (21.4-26.2, min/mean 0.85-0.97) are mislabeled per-TAP Frobenius norms. True per-output-channel norms are ~4.0-6.0 with min/mean ratios 0.70-0.85. The load-bearing conclusion (zero dead channels) is correct.

---

## 3. Bottleneck ranking (what actually limits strength)

1. **Target quality / signal content (PRIMARY).** Train losses are floored and U-shaped (loss_policy ~2.05, min 1.998 @ep66, rising after) while the net is healthy and ~39% of attn-block and ~25% of overall gammas are idle. A flat ~2.0-nat policy loss across two architectures/widths points to the *information content of the targets* as the limit, not the model. **This is now CONFIRMED by direct forward-pass measurement (see §4):** the policy loss = H(target) 1.25 + KL(target‖prior) 0.67, where the 0.67-nat gap is irreducible target-sampling noise (10× the net's D6 self-noise floor, capacity-invariant, untrainable over 40 epochs). The lever is a **stronger, lower-variance teacher** — `search_visits`↑ (live), `c_puct`↓, then **`pcr_full_proportion` 0.33→0.5** (more full-search rows) — NOT a sharper net (that fits noise). **NOTE:** raising `train_samples_per_epoch` does **not** address this — it raises reuse (4.36→6.45), not distinct-data coverage or target quality; the pool is the fixed ~537k taper window, untouched by `tspe` (see the §1 correction). It is also low-overfit-risk thanks to per-epoch D6 augmentation, but adds no new signal.
2. **Self-play volume / window (SECONDARY, distinct-data).** The only ways to feed genuinely *new* information are more fresh rows per epoch (`selfplay.games_per_epoch`, GPU/self-play-bound, lowers reuse) or a larger distinct pool (`expand_window_per_row`, `keep_target`). The sims 512→1024 bump (ep101) raises per-row search quality but feeds a net already underfitting, so extra depth may yield little.
3. **Capacity (NOT binding).** Demonstrated: +33% width → +13% effective dimensions, +0.05-0.09 nats policy floor, strength tie-at-best vs main_4. Do not widen.
4. **Exploration (NOT binding).** Stable PCR 33%, near-zero value band, gentle entropy decline, low truncation, steady early-stops. Adequate.
5. **Optimization / value head (NOT binding).** Stable grads, no instability, calibrated value head near its honest floor.

---

## 4. CONFIRMED bottleneck — forward-pass measurement & the fix (2026-06-29)

The §3 "target-signal" read was **confirmed directly** with a CPU-only forward pass of the trained net on 600 stored full-PCR self-play positions from ep99 (`_scratch_klgap_main5.py`). The policy loss decomposes as `loss_policy = H(target) + KL(target‖prior)`; we measured `KL(target‖prior)` for the ep100 and ep60 nets on the *same* positions:

| metric (600 ep99 full rows) | ep100 net | ep60 net | reading |
|---|---|---|---|
| H(target) | 1.248 | 1.248 | legitimate positional entropy (irreducible) |
| **KL(target‖prior)** | **0.672** | 0.775 | the *fittable* gap — confirms the ~0.7-nat proxy |
| — diffuseness over visited support | 0.404 | 0.467 | net more spread than the noisy realization |
| — leak onto unvisited-legal cells | 0.186 mass | 0.210 mass | net mass on cells search hard-pruned (96-child cap) |
| CE (≈ loss_policy) | 1.920 | 2.023 | reconciles with the logged loss |
| **top-1 agreement (prior vs search argmax)** | **0.568** | 0.548 | net's best move ≠ search's ~43% of the time |
| **D6 self-KL (net's own noise floor)** | **0.071** | 0.065 | net is highly self-consistent |

**The gap is irreducible noise, not underfit — four independent reads:**
1. **10× the net's own noise floor.** D6 self-KL = 0.07 (the net gives near-identical priors to symmetric positions), vs the 0.67 gap. The architecture is *not* the noise source.
2. **Capacity-invariant.** main_4 (c=96) gap 0.70 vs main_5 (c=128) 0.73 — 1.76× the parameters moved it the wrong way.
3. **top-1 agreement is stuck at ~0.56 for 40 epochs** (0.548 @ep60 → 0.568 @ep100). The 43% prior-vs-search argmax disagreement **cannot be trained away** — the hallmark of noise. If it were learnable signal, agreement would climb.
4. **Median support is 7 moves at H≈1.03 nats.** With several near-tied moves and 512 visits, *which* top move wins the visit count is substantially random; the net correctly learns the stable average and refuses to chase the per-search winner.

**Verdict (now high-confidence): the bottleneck is the variance of the 512-visit policy targets on near-tied positions.** It is *not* that search rubber-stamps the prior (it disagrees 43% of the time) — it is that those disagreements are dominated by sampling noise the net can't and shouldn't fit. Width cannot help; only a better/lower-noise teacher can.

**Important framing for the fix:** the goal is **not to minimize the KL gap** (you can do that by making search more exploitative, which just sharpens onto a possibly-wrong move). The goal is to make the **teacher (MCTS) a stronger, lower-variance player**, so its targets encode genuinely better moves. Two non-exclusive hypotheses remain for the *strength* plateau: **H1 = target-noise-limited** (what we measured for the loss floor) and **H2 = on-policy state-space entrenchment / BC-warm-start basin**. They imply *opposite* exploration moves, so do not crank everything at once.

**Knobs, ranked (the fix):**
1. **`search_visits` 512→1024 — already live since ep101.** This is the principled primary fix: more visits lower the visit-estimate variance *and* make search a stronger player, so targets are both less noisy and better. Shrinks the dominant 0.40 diffuseness term. **Verify it:** re-run `_scratch_klgap_main5.py` on ep115–120 rows — the gap should drop. If it does *and* strength rises → H1 confirmed, keep 1024.
2. **`c_puct` 1.5→1.1 (KataGo self-play value) — config-only A/B on a fork.** Lower c_puct makes PUCT exploit Q instead of spraying visits across Hexo's flat opening prior, concentrating the visit distribution → sharper, lower-variance targets. main_5 kept 1.5 explicitly "pending a c_puct decision" (it favors extra exploration); this is that decision. Potent here because the prior is flat. Caveat: it slightly reduces move discovery, so pair it with #1, don't substitute.
3. **Resolve the Dirichlet tension.** The ep101 `noise_fraction` 0.20→0.25 *injects* root noise that **inflates** target variance — it fights #1 and #2 on this exact metric (forced-playout target-pruning removes only part of it). If H1 holds (our evidence), revert to ≤0.20. If the gap fails to drop after ep101 despite 1024 visits, this bump is the prime suspect.
4. **`pcr_full_proportion` 0.33→0.5.** More full-search (low-noise) targets per epoch — cheaper than global 1024, complements #1.
5. *(minor)* **`root_policy_temperature` 1.1/1.15→1.0/1.05** — a sharper prior into search yields sharper targets; second-order.

**NOT the fix:** move-selection `temperature` (changes which move is *played* and thus state coverage — relevant to H2/conversion, **not** the per-position target variance); sharpening the *net* (higher `policy_weight` / lower train temperature → fits noise); `tspe`/width/depth (shown inert in §1/§3).

**Early-stopping: keep it (verified `search.rs:267-322`).** It fires only when the leader is mathematically locked (`best − second > remaining`) and **not before a 75% visit floor on recorded rows** (`full_visit_floor`), so it never changes the target's argmax and withholds ≤25% of visits only on already-decided positions. It does *not* touch the gap — the gap lives on near-tied positions, which don't early-stop (`best − second` is small there). Disabling it would cost ~12–16% self-play throughput to marginally sharpen already-won targets; that compute is far better spent on #1.

---

## 5. Disagreements / unresolved

**Refuted/corrected claims (laundered numbers restated above, not here):**
- Dead-unit attribution (Training C8): REFUTED — near-zero Adam second moment lives in WEIGHT tensors, not bias/LN/gamma; update-ratio median 0.122 not 0.15, min 0.0 not 0.002.
- Resume-artifact count (Exploration): REFUTED — 5 non-256-game epochs, not 2.
- "Floored flat from ep1-3" and "stvalue/cell_q still learning late" (Training C3/C5): CORRECTED to a U-shape with late regression.
- amp_scale "steady 65536" (Architecture): CORRECTED — spans 32768-131072, settles late.
- Conv-channel norm magnitudes (Capacity C5): mislabeled per-tap; corrected to ~4-6 per-output-channel (no-dead conclusion intact).
- mean_game_length slope +0.054 (Exploration): CORRECTED to +0.021.
- Coverage "20% of 481k window" (Exploration/Capacity): CORRECTED to effective ~96k / desired ~537k ≈ 18%.

**Unverifiable from current artifacts (and what would settle them):**
- **Did the noise 0.20→0.25 bump raise pre-search exploration?** Unanswerable: noise_fraction is not logged per-epoch, and the only entropy metric is post-search visit entropy, which the simultaneous sims doubling drives down. NEEDS pre-search prior-entropy logging or a sims-held A/B. (Also: the pre-bump value 0.20 is inferred from prompt+config-comment, not run data; config comments confusingly narrate "0.25 → 0.20" while the live value is 0.25.)
- **Is the ep85-100 parity a true c=128 ceiling, or would sharper/fresher data resume the climb?** Flat losses since ~ep45 suggest a data-signal limit, not optimization failure or raw capacity. (Note: more `tspe` is NOT the test — it only adds reuse over the same pool; the real distinct-data/quality levers are `pcr_full_proportion`, `selfplay.games_per_epoch`, and the window config.) A dedicated higher-game gauntlet (200+ games cand_ep100 vs main4_ep60) is needed to resolve the parity tie at lower variance than the pooled BT (per-point SE ~48-90 Elo permanently makes single-epoch verdicts INCONCLUSIVE).
- **Is there any train/val generalization gap?** No held-out/validation loss exists in diagnostics; the underfit read is inferred. A read-only held-out-shard scoring check approaches a forward pass and is out of scope under CPU/light constraints.
- ~~**Is the ~2.0-nat policy floor irreducible target entropy vs fixable underfit?**~~ **RESOLVED (§4):** forward-pass measurement shows the floor = H(target) 1.25 + an irreducible KL-noise gap 0.67; capacity-invariant and untrainable. Remaining open piece: does shrinking that gap (via #1/#2 in §4) actually *raise strength* (H1) or is strength additionally capped by state-space entrenchment (H2)? The ep101 1024-visit change is the live A/B.
- **Does the pooled BT systematically under-rate the candidate vs its direct h2h?** At ep100 the pooled gap is -84 Elo but that eval's direct edge was ~+44 Elo / 56% WR — the two disagree in sign. Re-pinning the pool or a dedicated gauntlet would settle it.
- **Is the idle effective-rank (61%) a genuinely low-rank task vs a training artifact?** Would need an activation-level forward-pass probe (out of scope). If low-rank, a NARROWER+faster net would match strength.
