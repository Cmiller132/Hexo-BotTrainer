# hexfield_main_5 — Health, Strength & Levers (synthesis @ ep85→86)

**Run:** `hexfield_main_5` (c=128, 2,916,133 params — confirmed from `checkpoints/epoch_000082.pt`), live at epoch 86 on a single RTX 4070 Ti. Warm-started from a BC prefit on main_4 ep67-77, launched 2026-06-25. Self-play = 512 visits.

This report synthesizes six domain investigations (health, eval, search-sims, strategy, recipe, capacity-infra) and 11 adversarial verdicts. Claims the verdicts **refuted** are downweighted and flagged; **supported** high-impact claims are foregrounded.

---

## 1. Bottom line

- **The run is HEALTHY.** None of main_4's three failure modes are present: no value-Q saturation (`root_value_mean` stays in [-0.019, -0.003] across all ep1-85 vs main_4's runaway to |Q|≈0.5-0.79), no game-length collapse (length 81→103 plies, *rising*, vs main_4's monotone 72→40), no policy hardening (`root_policy_entropy` steady 1.27-1.47 nats). Losses fall *coupled with* rising strength — the exact opposite of main_4's "loss down / Elo down" decoupling.
- **The model is genuinely STRONGER than its own lineage.** Fixed-anchor pooled Bradley-Terry Elo (vs SealBot=0) rose `cand_ep5` ≈ 51-63 → `cand_ep85` = 312 (~+260 Elo). SealBot raw win-rate climbed 0.59 → 0.83 (ep85: 53-11). This is the authoritative, compounding signal.
- **The recurring "INCONCLUSIVE" verdict is a by-design artifact**, not a stall. The candidate-vs-champion test is a single-epoch, ~10-pair tripwire with SE ≈ 86-140 Elo that resolves only ~250-300 Elo gross regressions. It is structurally permanent (16/17 evals INCONCLUSIVE) and gates nothing. **Ignore it for trajectory; track the fixed-anchor curve.**
- **Strength is in a soft plateau, not divergence.** Training losses bottomed ~ep66 (`loss_total` 6.80, `loss_policy` 1.998) and crept up slightly to ep85 (6.91 / 2.05); the fixed-anchor candidate curve is flat-within-noise since ~ep55-65. This is diminishing-returns at a config/data ceiling with all health indicators green — **not** the main_4 pathology.
- **main_5 has NOT yet decisively beaten the model it was built to surpass.** In the pooled BT fit, `main4_ep60` = 326.9 ±29 sits *nominally above* `cand_ep85` = 312.0 ±55 (diff −14.9 ±62, z=−0.24). The head-to-head edge oscillates (ep70 5-10, ep85 11-9) with CIs spanning zero. **Verdict-corrected:** statistical tie, not a positive edge.
- **The single biggest lever is DATA COVERAGE per epoch, not net size or search depth.** Capacity is *not* binding (2.9M params, flat *train* loss, 59% VRAM with ~5GB headroom). The taper window holds ~481k rows but `train_samples_per_epoch=96000` trains only ~20% of it per epoch. Raising tspe is config-only, self-play-cost-free, and directly attacks the plateau.
- **Do NOT raise self-play sims 512→1024.** Net (not search) is the binding constraint; expected gain is only +15-40 Elo one-time (low confidence) at ~36-50% throughput loss on the scarce GPU. See §4.

---

## 2. Run health & trajectory

**Verdict: healthy and learning across ep1-85; soft plateau approaching from ~ep55-66; zero main_4-style pathology.**

### Trajectory (numbers)
| Metric | ep~5 | ep~40 | ep~66 | ep85 | Read |
|---|---|---|---|---|---|
| Fixed-anchor cand Elo (BT, SealBot=0) | 51-63 | 251-264 | 327 (peak) | 312 | +260 then flat-within-SE |
| SealBot raw win-rate | 0.59 | ~0.80 | 0.91 | 0.828 (53-11) | strong, plateaued ~0.80-0.85 |
| `loss_total` (train) | 7.42 | 6.97 | **6.80 (min)** | 6.91 | bottomed ep66, mild uptick |
| `loss_policy` (train) | 2.24 | 2.058 | **1.998 (min)** | 2.051 | bottomed ep66, +0.05 nats |
| `loss_value` (train) | 0.586 | 0.57 | 0.553 | 0.570 | flat-to-down; honest (not collapsing to ~0.31) |
| `root_value_mean` (selfplay) | ~0 | -0.019 | ~-0.01 | -0.003 | **no saturation** (band [-0.019, -0.003]) |
| mean game length | 81-87 | 83 | 95 | 103 (p90 185) | **no collapse**, slight rise |
| `root_policy_entropy` | 1.41-1.60 | 1.41 | 1.19 | 1.41 | stable, no mode collapse |
| `grad_norm_mean` / p95 | 2.2 | ~3.2 | 3.5 | 3.5 / 5.5 | rose then plateaued ~ep40; benign |
| `clip_fraction` | — | ~0.02 | ~0.02 | ~0.01-0.04 | clipping rarely binds |
| `amp_scale` | 65536 | 65536 | 65536 | 65536 | stable, no NaN cascade |
| `reuse_ratio` | 3.0-3.4 | 4.6 | 4.0 | 3.76 | healthy ~4x |

### main_4 pathology check (all ABSENT — verdict: strongly supported, high confidence)
- **Value-Q saturation loop:** `root_value_mean` never leaves ±0.019. main_4 ran to |v|≈0.65 / turn-0 |Q|≈0.51. Corroborated by honest `loss_value` (~0.57, refuses to fall toward the 0.31 saturated-target signature). The adversarial verdict notes `root_value_mean` *alone* was non-discriminating in main_4 (stayed ~0 in both runs); the *combination* of near-zero mean + honest value loss + stable length is what makes the no-saturation conclusion safe.
- **Game-length collapse:** length is flat-to-*rising* (the inverse of main_4). No defensive lock-up; truncated_games <2.5% (4/256 at ep85).
- **Loss/strength decoupling:** losses fall *with* rising Elo — the strongest single structural-health signal.

### The soft-plateau read (verdict: partially-supported, high confidence)
The "second-half slope softening" claim is **directionally right but leaned on the wrong metric.** The per-epoch `candElo` series (ep45=232, ep65=350, ep75=230, ep85=312) is the single-epoch, non-compounding measure the eval itself flags as non-diagnostic for slope (SE ~50-140; the 230-350 swing is ~2 SE of noise). The *fixed-anchor* curve actually shows a steep ep60-80 segment (+9.28 Elo/ep, t=3.17, peak ep80=455.7 on that scale) — so the plateau **has not clearly bound yet**, and main_5 sits at/above the main4_ep60 ceiling and was still climbing into ep80. The corroborating plateau signal is the flat *train* loss post-ep66.

**Note on the ep66-82 loss reversal:** one investigation attributed it to serve-recompile throughput starvation (~3 pos/s starving the buffer). **This causal claim is REFUTED (high confidence):** per-epoch self-play throughput stayed 13.2-15.7 pos/s through the whole window (same band as ep40-65), "recompile" appears 0× in `events.jsonl`, no ~3 pos/s figure exists, reuse_ratio held ~3.8-4.9, and length *rose*. The reversal is a genuine diminishing-returns ceiling, not buffer starvation. (`ep82` selfplay shows 0 games — a resume artifact, not a stall.)

---

## 3. Model strength — what we actually know

**Eval methodology is statistically honest but underpowered for fine edges.** Converged pooled BT (ep85: 78 edges, 36 players, max_grad 1.9e-8), proper CRN pairing, SealBot down-weighted 0.5 and excluded from difference inference, explicit SE disclaimers. Strength is measured every 5 epochs at a 128-game budget (64 SealBot + 20×3 checkpoints).

### What is solid
- **+260 Elo lineage gain** (`cand_ep5`→`cand_ep85`), monotone-ish through the noise. This compounds and is the real signal.
- **SealBot dominance:** 0.83 win-rate (53-11) at ep85, up from 0.59. Independent external opponent, unambiguous.

### What the INCONCLUSIVE verdict means (verdict: SPRT-disabled claim SUPPORTED, high confidence)
- The candidate-vs-champion primary edge rests on **10 pairs** (ep85: pentanomial [5,0,4,0,1], n_eff=19.09, SE_elo=86.1, CI [-312.5, 25.0]). It is **permanently single-epoch-limited** — a fresh candidate node enters each epoch and never compounds, so SE never tightens with more epochs. It is a gross-regression tripwire (resolves ~250-300 Elo), nothing more.
- **SPRT is implemented but disabled** (`sprt.enabled=false`; class default is `True`, so it was deliberately turned off). The live `run_concurrent` mode excludes Stage B entirely (stages = [C_deep, D_pool]). Eval **gates nothing** — the deployed/serving net is always the latest checkpoint, no rollback. Between 5-epoch evals the run is unguarded for a mid-run collapse, though the absence of any saturation signal makes that low-probability.

### Corrected claims (downweighted)
- **"main_5 edges main4_ep60 11-9 (+35 Elo)"** — **REFUTED in direction (high confidence).** The +34.9 is cherry-picked from a noisy head-to-head (ep70 was 5-10, *losing*; CI spans zero every epoch). The stable pooled BT fit puts `main4_ep60`=326.9 ±29 *nominally above* `cand_ep85`=312.0 ±55. Correct statement: **statistical tie / nominally slightly behind**. main_5 has not yet demonstrably surpassed the predecessor it was built to beat.
- **"Primary edge starved because seats always agree (n_eff≈n_pairs)"** — the *starvation fact* is supported, but the *mechanism is refuted (high confidence)*: empty (1,3) pentanomial classes mean games were *decided* (Hex has no draws), not that seats agree. The 1-1 split class (class 2) is heavily populated (~40-44%), so n_eff≈2×n_pairs, not n_pairs. Pairing buys little *because* within-pair correlation is already low.
- **"Re-weight the 128-game budget toward the champion edge to cut SE 86→55"** — **REFUTED (high confidence).** The primary SE is dominated by the *frozen champion node's own* marginal SE (Cov_BB = 101.7² ≈ 10,343 >> 55²=3,025), which no candidate-side reallocation can touch. The candidate is already the *more* precise side (54.6 vs 101.7). Do not expect SE≈55 from re-weighting.
- **"Permanent anchors lack a strong external ceiling"** — **REFUTED on its load-bearing point (high confidence).** `main4_ep60` *is* a pinned, external (different run, different arch), strong, stable anchor (se 29, held BT rank 1 ep5-60, 17 compounding edges). It is the cleanest ceiling signal present and is already doing the job the claim said was missing. (The roster *is* thin — that half stands.)

---

## 4. The 1024-sims question

**Verdict: NOT WORTH IT for self-play. Modest-at-best one-time gain, real throughput cost, on a net that is the actual bottleneck.**

| Dimension | Finding |
|---|---|
| Expected Elo from 512→1024 (self-play strength) | **+15 to +40 Elo, one-time, logarithmic** |
| Confidence | **LOW** — extrapolated from AZ/Leela/KataGo priors, not measured for this net; single-epoch eval SE (~86-140 Elo) cannot even cleanly measure +30 |
| Throughput cost | **~36-50% drop in pos/s** (17.3 → ~8.4-12). Verdict-corrected: the "~halved" framing is overstated; re-derived ~1.57× NN work → −36% (only 33% of moves are full-search; fast moves at 192 and ~20% early-stop savings cushion it). Decisive either way. |
| Why net, not search, is the bottleneck | Policy loss bottomed ep66 and is flat/regressing; value loss flat; search at 512 is *healthy* (no saturation, length 103, ~30%/1.57M visits already early-stopped on easy positions — extra budget would be disproportionately absorbed). Verdict: the *facts* hold; the strong "net not search" *causal* framing is partially-supported (loss is non-stationary, search-vs-net was never empirically isolated). But the *decision* is robust. |

### Concrete cheap A/B (zero live disruption)
The multistage eval **already brackets sims**: the candidate plays at `eval_visits=128` while `full_search_visits=512` (a 4× gap on disk across 17 epochs). **Mine `hexfield.multistage_eval.epoch_*.json` to estimate this net's realized Elo-per-sims-doubling slope from existing edges.** If 128→512 (two doublings) already shows a shallow slope, 512→1024 is shallower still — confirming the verdict empirically with no GPU contention.

> ⚠️ **Open question to resolve before trusting eval-as-A/B:** one investigation noted SealBot edge provenance reports `eval_visits=512` while config lists `128` — confirm the actual eval search budget before using the 128-vs-512 bracket as ground truth.

**If sims are ever wanted, raise them at EVAL/deploy time only** — captures the log-scaling bump for the served bot with zero feedback into (and zero throughput cost on) the training loop.

---

## 5. How to push the model stronger (prioritized)

Best-first by impact × (1/effort) × (1/risk). **CONFIG-ONLY** levers are safe and need no rebuild; apply them one at a time so the plateau diagnosis isn't confounded. **All require waiting until GPU/run is not mid-epoch.**

### CONFIG-ONLY (safe, no rebuild)

| # | Lever | Concrete change | Impact | Effort | Risk | Rationale |
|---|---|---|---|---|---|---|
| **1** | **Raise `train_samples_per_epoch`** | `96000` → `192000` (then consider `288000`) in `_resume_config.toml`/toml | **Medium-High** | Low | Low | Window holds ~481k rows (`select` ep85: desired/used=480,783) but only 96k (~20%) trains per epoch. *Verdict correction:* the per-epoch draw is a **uniform random sample over the full window** (`window.py:663` shuffles all candidates, re-seeded `seed+epoch*65537`), **not** a "thin recent slice" — so the real effect is **lower per-step gradient variance + higher coverage per epoch**, not rescuing an ignored region. Self-play (the bottleneck) is unchanged; only the cheap ~6-10 min train phase grows ~2×. Check `max_train_bucket_size=500k` headroom first (level pinned ~404k). Fully reversible. |
| **2** | **Cut `soft_policy_weight`** | `1.0` → `0.25-0.5` (config.normalized.json:67) | **Medium** | Low | Medium | **Verdict SUPPORTED (high confidence):** the train-only soft head is the *single largest* weighted objective term — 2.400 (34.7% of `loss_total`=6.910) > policy 2.051 (29.7%). Its target is a flattened (`visit_policy^0.5`) duplicate of the policy target the main head already fits. Reallocating that objective mass is the most reducible at plateau. Risk is medium because the *gradient*-dominance claim is inferred, not measured (no per-head trunk-grad attribution exists). Ideally gate on probe `e2_aux_head_utility` when GPU is free; if alignment is low, the cut is near-free. |
| **3** | **Decay `learning_rate`** (only after #1/#2 land) | `5e-4` → step to `3e-4` then `2e-4`, or revert to sweep-favored `3e-4` | Low-Medium | Low | Low | lr is flat 5e-4 with no schedule; it drove gains to ~ep55 but not beyond — classic late-run decay candidate. main_3's held-out sweep was strictly lower-is-better across [5e-4, 1e-3]. *Do this separately* from #1/#2 to avoid confounding. Note: the toml comments are contradictory (a 5e-4→3e-4 "revert" block is itself overridden by a later 3e-4→5e-4 line); the live value is 5e-4. Trivially reversible. |
| **4** | **Add a strong in-run permanent eval anchor** | Pin `main_5 ep55` or `ep65` into `permanent_anchors` | Low | Low | Low | Makes future eval deltas resolve fine-edge (~15-20 Elo) gains/regressions instead of only gross ones. Eval-only, zero training risk. (Roster is thin; `main4_ep60` already supplies an external ceiling, so add an *in-lineage* strong anchor, not another external one.) |
| **5** | **Enable SPRT triage** | `sprt.enabled=true` (elo0=0, elo1=-50, max_games=64) | Low-Medium | Low | Low | Free gross-regression tripwire against any future saturation collapse. *Caveat (verdict):* the live `run_concurrent` path excludes Stage B, so this only fires on a path that actually runs Stage B — confirm wiring before relying on it. |
| **6** | **Keep self-play at 512 sims** | no change | — | — | — | Affirmative non-action; see §4. Spend GPU on data, not depth. |
| **7** | **Do NOT widen the net / add blocks** | no change | — | — | — | Capacity is not binding (flat *train* loss = net near its fit on current data; 59% VRAM). A bigger net slows *both* phases for no clear gain. Bank the ~5GB + GPU-slack headroom. |

### HEAVIER (needs GPU-free window or code; defer while live)

| # | Lever | What | Impact | Effort | Risk | Rationale |
|---|---|---|---|---|---|---|
| **8** | **Improve self-play batching** | Raise `active_games`/`virtual_batch_size`/`flush_target` — `mean_flush_states=217` << `flush_target=1024` | **High** (if real) | Medium | Medium | Self-play is 80-84% of wall-clock and host-bound at 73% GPU util; the batcher fires under-full, so the GPU is starved by host scheduling, not FLOPs. Lifting the ~3000 states/s ceiling shrinks the ~1900s self-play phase at no quality cost. Highest-leverage *throughput* move — but tuning concurrency on a VRAM-tight live box is risky; do it on a free GPU. |
| **9** | **Run the deferred `improve_probes` battery** | `e1` trunk linear probe (capacity vs head/data ceiling), `e2` aux-head grad-cosine, `a2`/`a3` search & strength scaling, `d1`/`d3` self-play overfit/coverage | High (insight) | Medium | Low | Written for the identical-arch main_4; localizes the plateau ceiling before any structural bet. `e1` is the decisive test of whether c=128 is itself the ceiling. GPU-gated — cannot run against the live run. |
| **10** | **Larger paired match vs main4_ep60** | >20 pairs in one eval cycle | Medium | Medium | Low | The project's entire premise (beat capacity-bound main_4) is currently a statistical tie (§3). A tighter measurement settles whether the goal is met. |
| **11** | **Lower instrumentation** (suggest now, implement later) | Log turn-0/first-25%-ply `mean(|q_chosen|)`, per-phase entropy, opening 2nd-move diversity into `events.jsonl` | Low | Medium | Low | Cheap early-warning for saturation (the direct main_4 metric isn't logged; `root_value_mean` is only a proxy) and closes the opening-diversity gap this audit couldn't resolve from JSON. |

---

## 6. Caveats & open questions

**Low-confidence / unresolved:**
- **Plateau vs eval-noise:** is strength truly flat ep65-85 or just eval-power-limited? The fixed-anchor curve was still climbing steeply into ep80 on the joint-fit scale; the per-epoch cand curve is too noisy to tell. A denser low-variance ladder (probe `a3`, GPU) or several more epochs of the fixed-anchor curve would decide. **Watch the next ~15-20 epochs.**
- **Direct saturation metric not verified:** `root_value_mean` (±0.02) is a strong proxy, but turn-0 / first-25%-ply `mean(|q_chosen|)` — the exact main_4 metric — is not logged. Would require decoding `samples/.../*.npz` (deferred to avoid IO/GPU contention).
- **Opening diversity unconfirmed for main_5:** no first-move histogram is logged; opening health is *inferred* from entropy + length spread + low truncation. The persistently empty pentanomial (1,3) classes raise a (now-explained: no draws) flag, but per-move opening decorrelation was not directly measured.
- **Is c=128 itself the ceiling?** Unresolved without the GPU-gated `e1` trunk linear probe. Flat *train* loss argues the net is near its fit on *current data* (favoring data levers #1, not more width), but this is inference.
- **tspe governor headroom:** at tspe=192k the per-epoch debit may exceed new-data credit; re-derive steady-state reuse and `max_train_bucket_size=500k` headroom before raising tspe far. reuse_ratio is already ~3.8-4.2; doubling tspe roughly doubles per-row reuse (staleness/overfit trade-off).
- **Eval search budget ambiguity:** SealBot edge provenance reports `eval_visits=512` while config lists `128` — confirm before trusting the 128-vs-512 sims A/B (§4).
- **Why does main_5 only tie main4_ep60 despite ~2× capacity?** Possibly the BC warm-start (main_4 ep67-77) anchors its policy too close to the predecessor, capping the achievable margin. Open.

**What to measure next (cheap, read-only, no GPU):**
1. Mine the existing 128-vs-512 eval edges for this net's Elo/sims slope (§4 A/B).
2. Track a 3-point rolling mean of the fixed-anchor cand-vs-SealBot Elo as the **headline** trajectory metric (not the verdict label) over ep86-105.
3. Re-check `loss_total`/`loss_policy` at ep90-95 against the ep66 minima to confirm whether the plateau is firm.
4. Watch `mean_game_length` drift (97→103 in 3 epochs) — unbounded growth toward `max_game_plies=256` silently erodes epochs/day.

---

*Synthesis grounded in: `E:\Hexo-BotTrainer\runs\hexfield_main_5\diagnostics\` (events.jsonl, hexfield.training/selfplay/multistage_eval/select.epoch_*.json), `checkpoints\epoch_000082.pt`, `_resume_config.toml`, `config.normalized.json`; code in `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\` (trainer.py, window.py, batching.py, losses.py, model.py, eval_stats.py, multistage_eval.py, config.py) and `rust\src\tree.rs`. Adversarial verdicts applied: divergence-vs-plateau (partially-supported), primary-edge starvation mechanism (refuted), SPRT disabled (supported), anchor ceiling (refuted), budget re-weight (refuted), net-not-search (partially-supported), ~halved throughput (partially-supported, ~−36%), data-quality redirect (partially-supported), main4_ep60 margin (refuted-direction), loss-reversal cause (refuted), tspe waste mechanism (partially-supported), soft-head dominance (supported).*
