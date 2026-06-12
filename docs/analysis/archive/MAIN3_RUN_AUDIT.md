# dense_cnn_restnet_main_3 — Run Audit

**Audit date:** 2026-06-11 (evening), ~5.2 h into the run
**Run dir:** `E:\Hexo-BotTrainer\runs\dense_cnn_restnet_main_3` (read-only)
**Scope:** completed epochs 1–6 (checkpoint + diagnostics `status=completed`), with an epoch-7 addendum (ep7 completed at 19:20:58 local, mid-audit, and was re-measured by the completeness pass). Epoch 8 is in flight and untouched.
**Verification protocol:** every finding below was adversarially re-derived by an independent verifier on disjoint samples and/or independent data paths before inclusion. Labels: **[confirmed]** survived as stated; **[modified]** direction held but a quantitative element was corrected (corrected claim shown); refuted claims appear only in the Corrected Record.

---

## Executive summary

**Verdict: healthy, with watch items.** main_3 at epochs 1–7 shows none of the main_2 collapse signatures and is at-or-above main1's healthy trajectory on every comparable metric.

The single most important number per dimension:

| Dimension | Headline number | Reading |
|---|---|---|
| Mechanics | 0 relaunches, 0 errors, 18/18 fp16+compile gates passed | clean infrastructure |
| Policy — sharpening | search−prior top1 gap **+0.212** at ep6, positive in all 4 ply buckets | no main_2-style inversion (main_2 ep5: +0.100 and falling; main1 ep11: +0.171) |
| Policy — early plies | ply[5,16) prior top1 **0.268** at ep6 (0.248→0.271→0.268) | the exact axis that died in main_2 (0.219→0.027) is holding |
| Policy CE | 2.835 at ep6 (+0.52 since ep2) → **2.663 at ep7** (reversed) | drift was almost entirely target-entropy mechanical; ep7 undid the residual |
| Value | **+0.062 nats** over best-constant hedge at ep5, sign-acc 0.604; ep7 train value CE **0.6843** — first epoch below the ln(2)=0.6931 hedge bound | learning real signal, mild optimism bias to watch |
| Strength | sealbot **106/128 (82.8 %)** at ep5 | vs main_2's 33/128 at ep5; single data point, next at ep10 |
| Throughput | ~2,670 s/epoch, 16.3–16.9 pos/s; 60-epoch ETA **~2026-06-13 13:00 local** | flat and stable (ep7's 3,010 s came from a pos/s dip to 14.76, plausibly the audit's own CPU probes) |

The two excursions the audit flagged at ep6 — the P0 winner skew (58.9 %) and the visit-target softening — **both resolved in the healthy direction at ep7** (winner split 193/191; full-population visit-target top1 0.576→0.651). The remaining genuine watch items are the unexplained ep5→ep6 policy-CE residual (+0.196 nats, reversed at ep7 but unexplained), the value head's optimism bias trajectory, two never-audited auxiliary heads with rising losses, and the structurally unobservable 14.2 % of full-search rows dropped by surprise weighting.

---

## 1. Mechanics

### 1.1 Supervisor: single launch, zero relaunches, zero errors **[confirmed]**

Since start (2026-06-11 18:08:53 Z) the supervisor launched the driver exactly once (pid 897382 → child 897404). `supervisor.log` is 4 lines (start, breaker policy, FIRST LAUNCH, LAUNCH); the single train log (33 lines) has zero Traceback/RuntimeError/CUDA-error/OOM hits even under a 15-pattern case-insensitive grep. Verifier's strongest independent evidence: supervisor and driver show **identical etime (04:54:52)** in the live WSL process tree — a relaunch would necessarily make the child younger. `diagnostics/events.jsonl` (a third path) has zero error/halt/breaker events. Only stderr noise: one benign `torch.frombuffer` UserWarning; all 11 compact_io crop-guard lines read "0 dropped".

### 1.2 Wall time and ETA **[confirmed]**

6 epochs in 15,715 s (14:08:53→18:30:48 local). `search_positions_per_second` flat: 16.36 / 17.36 / 16.25 / 16.47 / 16.84 / 16.93. Epoch wall grew ep1→ep4 only because decisions/game grew 85.2→108.8 (+28 %) and then plateaued (108.5 / 108.4 / 108.8) — verified by an independent count of `len(action_ids)` over all 384 raw .hxr games per epoch (ep1: 32,731 moves; ep6: 41,794; exact match to diagnostics counters). Checkpoint-mtime deltas: 2054 / 2595 / 2690 / 2640 / 3426 s — ep6's delta includes ep5's 754 s sealbot eval (2667+754=3421 ✓), expected accounting, not a stall.

**Projection:** 54 remaining epochs × ~2,670 s + 11 evals × ~754 s ≈ 152,500 s (~42.4 h) → completion ~**2026-06-13 13:00 local**, conditioned on the ~109-ply plateau holding. Known risk (disclosed in the claim): main1's mature EMA was ~150 decisions/game, so later game-length growth would push this later.

**ep7 addendum:** ep7 took 3,010 s — 13 % over projection — but dec/game stayed at 107.5; the cause was a pos/s dip to 14.76, plausibly contention from this audit's own WSL CPU probes running during ep7. Re-check pace at ep8 before adjusting the ETA.

### 1.3 Continuous scheduler batching improving **[confirmed]**

`mean_flush_states` 152.6 → 327.9 over ep1–6 (+115 %; verifier recomputed as flushed_states/flush_count — exact match all epochs, and cross-validated byte-identical against the independent `dense_cnn.selfplay.epoch_NN.json` serialization). No-progress flush ratio fell 83.1 % → 61.9 %. The 1024-state bucket is modal from ep2 onward (8,975/24,393 = 36.8 % of flushes at ep6, bounded 69–84 % of the 8.00 M flushed states). flushed/queued = 94.1–96.3 % with no decline — no starvation trend. Note: no main1/main_2 baseline exists for the no-progress counter; the **trend** (falling) and rising flush size are the meaningful signals, not the absolute level.

### 1.4 PCR and policy-init conformance **[confirmed]**

full/(full+fast) = 0.2464–0.2491 each epoch vs configured 0.25 (max deviation 0.36 pp); ep7: 0.2511. Accounting identities hold exactly at the diagnostics level: `raw_samples == full_search_count`, and `fast_rows_excluded − fast_search_count == policy_init.moves` every epoch. Verifier's census: summing per-game action counts across all 2,304 .hxr games equals full+fast+policy_init **exactly (diff 0)** in all 6 epochs. Truncated games 3/2,304 (0.13 %), independently confirmed by max game length hitting the 1024 cap in exactly epochs 1/3/5.

*Caveat (verifier):* on-disk npz row totals differ from `full_search_count` by −0.9 % to +0.4 % (both directions); the identity is exact only at the counters, approximate for literal disk rows.

Policy-init moves: 319/282/322/315/314/250 (mean 300.3, consistent with ~288 expected if the forced-origin ply-0 doesn't count). The ep6 dip to 250 **resolved at ep7 (350 moves)** — binomial noise, as suspected.

### 1.5 Temperature controller EMA converging **[confirmed]**

EMA: 150.0 (seed) → 133.8 → 122.4 → 118.1 → 115.7 → 113.9 (ep7: 112.6); halflife = 0.12×EMA **exactly** every epoch (18.00→13.66). Verifier solved the recurrence: α = 0.7500 exactly at all five transitions (prediction error 0.0), so the gap to actual (64.8 plies at ep1 → 5.0 at ep6) closes at exactly 0.75×/epoch → within ~2 plies by ep10. The 150-seed (from main_2's measurement) made early epochs run a temperature halflife longer than intended — ep1 +38 %, ep2 +23 %, ep3 +12 % (the original "~25–30 %" understated ep1) — self-correcting and nearly done.

### 1.6 ep6 winner skew **[confirmed, resolved at ep7]**

ep6: P0 won 226/384 (58.9 %; z=3.47, two-sided p≈5e-4, ~0.003 after Bonferroni over 6 epochs) vs balanced 49.0–52.1 % in ep1–5. Verified on two independent data paths (.hxr winner fields and npz value-target sign × current_player — identical 226/158), uniform across game-index quartiles (55/57/56/58 per 96-game block), so epoch-wide, not a worker burst. **ep7 re-measurement: 193/191 (50.3 %) — did not persist.** Training-row value-target means by current_player re-balanced from ep6's +0.175/−0.126 to +0.030/+0.041 at ep7. Keep an eye on ep8–10; a *persistent* P0 skew would bias hard-z value targets.

### Mechanics anomalies (noted, non-findings)

- ep5 top-level `elapsed_seconds` (3398.5) exceeds its checkpoint-mtime delta (2640 s) because the sealbot eval runs after the checkpoint save but inside the stage timer — expected.
- `_main3_health.py` prints `loss=nan` because it reads `d['train']` but the diagnostics key is `'training'` — **script bug**, training data is present. (Internal-consistency note: the critic findings cite `metadata.result.train.loss_components`; the actual key, verified directly, is `training`. The numbers themselves agree.)
- compact_io spill-beyond-hex-dist-20 counts are stones/history channels only (never policy/legal) — informational, not data loss.

---

## 2. Policy

### 2.1 Search still sharpens the prior at every ply bucket **[confirmed]**

ep6: search_top1 − prior_top1 = **+0.212 overall**; by bucket +0.208 [0,5) / +0.173 [5,16) / +0.196 [16,48) / +0.231 [48,∞). Verifier reproduced on a fully **disjoint** sample (last 60 games vs first 60): +0.223 overall, all buckets positive, and per-position paired analysis shows 80–88 % of individual positions sharpened (median positive in all buckets) — the means aren't hiding an inverted subpopulation. Zero rows missing visit weights; no load warnings.

**Cross-run:** main_2 ep5 was already down to +0.100 en route to collapse; main1 healthy ep11 was +0.171. main_3 sits *above* the healthy reference.

### 2.2 Early-ply prior confidence rising/holding **[confirmed]**

Ply [5,16) prior top1: 0.248 (ep3) → 0.271 (ep5) → 0.268 (ep6); ply [0,5): 0.064 → 0.136 → 0.133. Verifier's disjoint sample (games 60–355, stride 5) reproduced direction everywhere, with the [0,5) rise *larger* (3.1×) on his sample. main_2 fell 0.219 → 0.027 on this exact metric — main_3 shows the opposite. Overall prior top1 stable at 0.33–0.35, H_raw 2.8–2.9 nats. Minor: verifier's sample shows a small ep5→ep6 dip in [5,16) (0.308→0.265) — "holding" is fair but recheck at ep8–9.

### 2.3 Policy CE drift **[modified — corrected claim]**

Raw series correct: CE 2.659 / 2.316 / 2.711 / 2.638 / 2.686 / 2.835 (ep1–6); ep2→ep6 +0.519 nats. Far milder than main_2 (+1.387 over the same window, 5.36 by ep7) and opposite to main1's healthy decline (3.36→2.71).

**Corrected attribution** (the original said "about half mechanical"): measured over **all 384 games/epoch**, the ep2→ep6 drift is almost **entirely** mechanical — full-population visit-target entropy rose +0.499 (1.043→1.542) against the +0.519 CE rise; CE−H is roughly flat (1.27→1.29). Game lengthening (88→109 dec/game) and midgame composition shift (rows at ply≤20: 22.5 %→17.2 %; median branching 530→645) drive it. **However**, the ep5→ep6 step (+0.149) is *not* explained by fresh-data target entropy, which actually **fell** (1.589→1.542) — the previously reported 1.628→1.690 rise was a first-N-games sampling artifact. The fresh-data CE−entropy gap widened +0.196 in that step, only partly attributable to the tapered replay window (48k sampled rows/epoch, ~2-epoch lookback) aging out low-entropy ep1–2 rows. **Watch the ep5→ep6-style residual, not the headline drift.**

**ep7 addendum:** CE reversed to **2.663**, back below the ep1 level — the residual did not compound.

### 2.4 Opening diversity flat — no monoculture **[confirmed]**

Distinct ply-1 moves per 384-game epoch: 70/75/92/87/100/84; top ply-1 share fell 0.549 (ep1) into a stable 0.43–0.47 band; distinct 2-ply prefixes 147–166, top share 0.34–0.38. Verifier added measures: ply-1 Shannon entropy *rose* H=2.19 (eff. 8.9 moves) → 2.5–2.76 (eff. 12–16); top-5 mass fell 0.76→0.65–0.70 — mild **de-concentration**, opposite of main1's failure mode. The dominant ply-1 move was a different action each of ep1–6.

*Critic correction:* the "rotates every epoch" rider is already **stale** — ep7's modal reply is the *same* action as ep6's (2147188736, share 0.448→0.440), and its P0-win rate moved 0.570→0.479, i.e., as a P1 reply it now performs near-fair — consistent with convergence on a genuinely good reply, not noise churn. Also: the ep6 winner skew was **not opening-mediated** (P0 win rate 0.570 on the modal reply vs 0.604 on all other replies).

### 2.5 Visit-target sharpness eased slightly **[confirmed as "within noise", resolved at ep7]**

First-60-game probe: top1 0.577 (ep3) → 0.558 (ep5) → 0.544 (ep6). Verifier's analysis shows the monotone decline is **sample-dependent**: on a disjoint last-100-games sample ep6 *rebounds* above ep5 (0.581 vs 0.562), per-epoch deltas are ~1–2 SE (per-game std 0.086–0.115 → 60-game SE ~0.011–0.015), and first-60 vs last-100 within ep6 differ by 0.037 — more than any epoch-to-epoch delta. Full-population ep6 value: **0.576 / H 1.542**.

**Internal-consistency note:** three sampling schemes give three ep6 numbers (first-60: 0.544/H 1.690; last-100: 0.581/H 1.534; full population: 0.576/H 1.542). The first-N-games scheme used by `_m2_probe.py` is **systematically biased toward diffuser targets**; full-population numbers are authoritative and are what the trend assessments below use. All readings remain far from main_2's ep5 collapse (0.433/H 2.14) and at/above main1 ep11 (0.569).

**ep7 addendum:** full-population stored-target top1 jumped 0.576 → **0.651**, H 1.542 → 1.197. The audit's escalation trigger was a drop to ~0.51; it moved +0.075 the other way.

### 2.6 ep6 winner skew (policy-side view) **[confirmed]**

Same excursion as §1.6, independently confirmed from npz targets (hard ±1, confirming soft_z=0 as configured). Possibly-related rider: probe value bias mean_v−mean_z rose 0.089→0.111 and sign-acc dipped 0.604→0.578 ep5→ep6 — plausibly confounded by the skew itself (see §3).

### Policy anomalies (noted)

- `_m2_probe.py` hardcodes main_2-era knobs (T_base 1.1, eps 0.25) that do **not** match main_3's config (T 1.05 flat, eps 0.20). This contaminates only the synthetic decomposition fields (H_temp_*, H_mix, frac_from_*) in `_wf_m3_policy_ep*.json`; the raw prior, visit-target, and sharpening numbers used above are knob-independent.
- main_3 diagnostics carry no per-epoch training `rows` field; row counts come from `selfplay.raw_samples` (7,986–10,297/epoch, rising with game length).

---

## 3. Value head

*Value was not this audit's primary scope; the ep5 autopsy remains the reference, supplemented by ep6/ep7 cross-measurements made during the policy and critic passes.*

- **Reference (ep5, established):** +0.062 nats over the best-constant hedge, sign-acc 0.604 with a strong horizon gradient (0.97 near game end → 0.50 far), bias mean_v−mean_z = +0.089. Healthy and comparable to main1's mature ep35 head (+0.061 nats, sign-acc 0.66) — at epoch **5** rather than 35.
- **ep6 drift (anomaly, watch):** bias +0.089 → +0.111, sign-acc 0.604 → 0.578 (a second probe with a different sample: bias 0.128, sign-acc 0.574 — same direction, within noise). Small, and plausibly **confounded by the ep6 winner skew**.
- **Window-lag hypothesis (critic, [info], testable):** the ep6 checkpoint trained almost entirely on balanced-z rows (ep4/ep5 current-player z means −0.003/+0.058) and was then evaluated on ep6 rows where P1-to-move z crashed to −0.126 — a head predicting the window's balanced base rate mechanically shows positive P1 "bias" on those rows without any head pathology. **Prediction:** since ep7 targets re-balanced (+0.030/+0.041), optimism measured on ep7/ep8 rows should fall back toward ~+0.07. If it instead stays ≥ +0.15 on balanced rows, the surprise-weighting z-shift (§4) and a real head bias become the leading suspects.
- **ep7 train value CE = 0.6843** — the first epoch below the ln(2) = 0.6931 constant-hedge bound (prior minimum 0.695). The training loss itself now certifies above-hedge signal.
- **Standing reminder (established):** the owner-swap zero-sum probe is off-distribution-confounded and must not be read as optimism; use mean(v_pred)−mean(z). The main1 epoch≤23 checkpoint load trap (silent random value head) does not apply to main_3 checkpoints (current heads keys, clean loads verified in every probe).

---

## 4. Training data & surprise weighting (critic pass)

### 4.1 Surprise upweighting does NOT concentrate on mush **[confirmed via gap-run; watch for side effects]**

The feared failure mode — duplicated training rows being high-entropy "mush" — is **refuted**: corr(copies, policy-target entropy) is *negative* every epoch (−0.183/−0.168/−0.140 for ep5/6/7); 2×-copied rows have entropy 0.99–1.27 vs 1.30–1.75 for singletons; 3+-copy rows 0.35–0.59 with frac(H>3) ≤ 1.4 % vs 13–15 % in singletons. Surprise weighting concentrates on **sharp** targets.

Two un-flagged side effects, same sign all 3 epochs measured:

- **(a) Standing positive z-shift:** duplicated rows skew toward current-player wins, lifting the materialized value-target mean above the unique-position mean by +0.0200 (ep5), +0.0072 (ep6), +0.0068 (ep7) — the same direction as the observed optimism bias.
- **(b) Late-ply tilt:** 3+-copy rows sit at ply_mean 91.6–120.3 vs 70.6–84.5 for singletons (children 6.5–12.7 vs 25–34) — training emphasis tilted toward near-terminal rows, the region where sign-acc is already strong; consistent with (not proven to cause) the contracting accuracy horizon.

### 4.2 Dropped surprise rows are unobservable **[watch — structural gap]**

1,456–1,478 full-search positions per epoch (14.2 % of raw rows) receive 0 copies and never appear in any npz; the search prior is not persisted per row, so the drop set's composition **cannot be audited post hoc**. Given copies anti-correlate with entropy and positively correlate with current-player wins, the dropped set plausibly skews toward prior-confirmed, current-player-LOSS rows — systematic under-training of "I am losing and search agrees" positions, which would push the value head optimistic. *Hypothesis, not measurement.* Closing it requires logging surprise weights (or priors) at selfplay time for one epoch.

---

## 5. Strength / evaluation

- **Sealbot ep5: 106/128 wins (82.8 %)**, mean 76.3 turns, 754 s. Compare main_2's collapse trace (33/128 at ep5, 6/128 at ep10) and main1 (50/64 at ep30). One data point; next eval at ep10 (~3 h away at current pace).
- **Gap [info]:** strength evidence is a *single point*. No checkpoint-vs-checkpoint games (e.g., ep7 vs ep2) exist anywhere in the run dir, so "strength rising" between evals is inferred only from proxy metrics — and main_2 proved proxies can look mixed while strength collapses. Cheap to close with a small CPU h2h after ep10.
- **Gap [info]:** the eval P0 funnel line (0,0),(0,2),(−2,2),(−1,2) — 61/64 P0 eval games — was never decoded against selfplay's modal opening ids (opaque ints like 2147188736), so whether sealbot steers the model into a heavily-trained or barely-trained line is unknown. This conditions how much the 82.8 % generalizes.
- **Standing reminder (established):** opp_policy CE reads ~4× low due to PCR mask dilution (coverage 0.236) — not anomalous.

---

## 6. Corrected record

Claims from the audit that verification refuted or materially modified:

1. **Policy-CE attribution (modified):** "about half the recent rise is mechanical" → over ep2→ep6 it is almost **entirely** mechanical (full-pop target entropy +0.499 vs CE +0.519, CE−H flat); but the ep5→ep6 step is **not** entropy-driven — the cited H_visits rise 1.628→1.690 was a first-60-games sampling artifact; full-population fresh-data entropy *fell* 1.589→1.542 in that step.
2. **First-N-games probe sampling bias (new systematic):** `_m2_probe.py`'s `shards[:max_games]` selection biases visit-target entropy high / top1 low (ep6: first-60 H 1.682 vs full-pop 1.542). All single-sample probe trend deltas of ~0.01–0.04 should be treated as ~1–2 SE noise.
3. **"Dominant opening rotates every epoch" (stale):** ep7 retained ep6's modal reply — the rotation pattern broke at the first epoch after the audit snapshot.
4. **Visit-target monotone decline ep5→ep6 (sampling noise):** disjoint sample shows ep6 *above* ep5; pooled ep6 ≈ flat vs ep5; fully reversed at ep7 anyway.
5. **"~25–30 % longer halflife in early epochs" (understated for ep1):** ep1 was +38 %, ep2 +23 %, ep3 +12 %.
6. **`_main3_health.py` loss=nan (script bug, not missing data):** reads `d['train']`; the key is `'training'`.
7. **ep6 policy_init dip to 250 (resolved):** ep7 = 350 moves; binomial noise as hypothesized.
8. **ep6 winner-skew p-value (overstated precision):** quoted p≈0.0005 is the max over 6 epochs; Bonferroni-corrected p≈0.003 — still significant.

---

## 7. Watch items

| # | Signal | Current value | Trigger threshold | Action |
|---|---|---|---|---|
| 1 | Policy CE − full-pop target entropy (residual) | gap 1.29 at ep6 (+0.196 step ep5→ep6); CE reversed to 2.663 at ep7 | residual gap rises >0.3 nats cumulatively over 3 epochs | full-pop entropy decomposition on the offending epochs; check replay-window composition |
| 2 | Full-population visit-target top1 | 0.651 at ep7 (0.576 ep6) | < 0.51 on **full-pop or ≥100-game** measurement (never a single first-60 probe) | escalate: rerun `_m2_probe.py` on 2 disjoint subsets + grid-check exploration knobs |
| 3 | Winner balance (P0 share) | 0.503 at ep7 (excursion 0.589 at ep6) | > 0.55 for 2 consecutive epochs | check value-target balance by current_player; consider z-rebalancing before it feeds both heads |
| 4 | Value optimism bias mean_v − mean_z | +0.089 (ep5) → +0.111/+0.128 (ep6, skew-confounded) | ≥ +0.15 measured on a *balanced* epoch's rows | run `_value_autopsy.py` ep8 ckpt on ep8 rows (CPU, ≤60 games); if high, suspect surprise-weighting z-shift (§4.1a) |
| 5 | Window-lag prediction (test of §3 hypothesis) | untested | optimism on ep7/ep8 rows does NOT fall back toward ~+0.07 | promotes watch item 4 to active investigation |
| 6 | Aux heads: moves_left, stvalue_2 losses | 3.153→3.179→3.203 and 2.741→2.822 (ep5–7), rising while primary heads improved | monotone rise through ep10 | one-off CPU calibration probe for both heads (never measured; share the trunk) |
| 7 | Sealbot eval at ep10 | 106/128 at ep5 | < ~106/128 at ep10 | direct strength regression: h2h ep10-vs-ep5 games; do not rely on proxies (main_2 lesson) |
| 8 | Epoch wall time / pos/s | ep7: 3,010 s at 14.76 pos/s (vs 16.3–16.9 baseline) | pos/s < 16 at ep8 with no audit probes running | profile evaluator; rule out external CPU contention first |
| 9 | Early-ply prior top1 [5,16) | 0.268 at ep6 (one sample showed 0.308→0.265 ep5→ep6) | < 0.20 at ep8–9 | this is the main_2 death axis — immediate probe on 2 disjoint samples |
| 10 | Surprise-dropped rows (unobservable) | 14.2 %/epoch invisible | n/a (structural) | log per-row surprise weights or priors for one epoch to audit drop-set composition |

---

## 8. Gaps & critic additions

**Resolved by the critic's gap run (ep7 completed mid-audit):**
- ep6 winner skew → 193/191 at ep7 (resolved; watch item 3 retained at low priority).
- Policy CE drift → 2.663 at ep7, below ep1 level (resolved; residual mechanism still unexplained, watch item 1).
- Visit-target softening → 0.651 full-pop at ep7 (resolved, moved opposite to the feared direction).
- Bonus: ep7 train value CE 0.6843, first sub-hedge epoch.

**Open gaps (not closable from existing artifacts):**
- **Dropped surprise rows** (§4.2) — needs one epoch of selfplay-time weight logging.
- **Aux heads never audited** (§ watch 6) — no prediction-side measurement of moves_left/stvalue exists in any artifact; losses drift opposite to primary heads.
- **Single-point strength evidence** (§5) — no h2h between main_3 checkpoints; next direct evidence ep10.
- **Eval-vs-selfplay opening overlap** (§5) — requires decoding action ids to axial coordinates (scheme in `hexo_utils/records.py`).
- **Window-lag prediction** (§3) — deliberately deferred to stay within CPU budget; one `_value_autopsy.py` run on ep7/ep8 rows settles it.

---

## 9. Methodology appendix

**Epoch admission rule.** An epoch counts only if `checkpoints/epoch_0000NN.pt` exists AND `diagnostics/epoch_0000NN.json` has `status=completed`. Audit body: epochs 1–6. Epoch 7 completed at 19:20:58 during the audit and is included only via the critic's explicit re-measurements. Epoch 8 in-flight, untouched. All in-flight partials (ep7 at 296–346 games during the main pass) were excluded from conclusions.

**Compute discipline.** All torch probes CPU-only (`CUDA_VISIBLE_DEVICES=` empty) in the WSL venv `/root/.venvs/hexgt-build`; probe samples ≤60 games except where full-population npz/.hxr scans (no model forward) allowed a complete census. The GPU was never touched; the ep7 pos/s dip (16.9→14.8) is plausibly CPU contention from these probes.

**Measurements and sample sizes.**
- *Diagnostics counters:* full-population per-epoch JSON (`metadata.result.selfplay.*`, `metadata.result.training.loss_components`) — no sampling.
- *Policy probes:* `_m2_probe.py` (CPU, hard-target mode, λ=0.0) on 60-game samples → 1,449–1,898 positions/epoch; raw prior, visit-target, and sharpening fields are exploration-knob-independent (the hardcoded main_2 knobs contaminate only synthetic decomposition fields).
- *Full-population target stats:* direct npz `pol_w`/`pol_off` and value-target reads over all 384 games/epoch (7.9k–10.3k rows), no model load — immune to the checkpoint-load trap.
- *.hxr census:* `hexo_utils.records.HexoRecordFile` over all 384 games/epoch for winners, openings, game lengths.
- *Value reference:* `_value_autopsy.py` ep5 (CE-vs-best-constant-hedge, sign-acc by horizon, conv-bottleneck health).

**Verification protocol.** Every finding was adversarially verified before inclusion: independent re-derivation on **disjoint samples** (e.g., last-60/last-100/stride-5 games vs the auditor's first-60) and/or **independent data paths** (npz targets vs .hxr records; alternate JSON serializations; live process-tree etimes vs logs; recomputed means vs stored fields), with explicit confound checklists (in-flight contamination, denominator dilution, first-N sampling bias, checkpoint-load key traps, composition shifts, multiple comparisons). Findings whose quantitative content failed re-derivation were downgraded to [modified] with a corrected claim (§2.3) or moved to the Corrected Record (§6). Where two measurements disagree (visit-target sharpness across sampling schemes, §2.5), both numbers are reported and the discrepancy attributed, never averaged.

**Cross-run baseline hygiene.** All main_2/main1 comparisons use the identical tool and field (e.g., `loss_components.policy`, `_m2_probe.py` outputs), so cross-run deltas are protocol-consistent. main1 epoch≤23 checkpoints are never used for value comparisons (silent random-value-head load trap).

**Key artifacts** (under `E:\Hexo-BotTrainer-hexgt\scripts\`): `_wf_m3_policy_ep3.json`, `_wf_m3_policy_ep6.json`, `_m3_probe_ep5.json`, `_m3_autopsy_ep5.json`, `_wf_m3_mech_health.txt`, `_wf_m3_health_out.txt`, `_wf_m3_critic_skew_out2.txt`, `_wf_m3_critic_surprise_out.txt`, plus per-finding verifier outputs `_wf_m3_v_<finding-id>_*`. Baselines: `_m2_probe_m2_ep5.json`, `_m2_probe_m2_ep11.json`, `_value_autopsy_out.json`, `_grid_ep{2,5,11}.json`.
