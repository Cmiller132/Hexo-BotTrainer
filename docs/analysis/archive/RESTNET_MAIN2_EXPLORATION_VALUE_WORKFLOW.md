# RestNet Exploration & Value-Learning Workflow — main_2 (live, epoch 11)

## Executive summary

Two owner hypotheses about the struggling `dense_cnn_restnet_main_2` run were tested CPU-only, read-only, against checkpoints + self-play shards, with a **matched-epoch A/B vs `main1`** and an adversarial multi-agent adjudication. **Both hypotheses are largely refuted *as causes* of the run's trouble, though each yields a defensible minor tuning.**

- **H1 — "set `soft_z_lambda` to 0 (hard targets)": REFUTED as a cause; do NOT zero it as a fix.** soft-z is live and correct on main_2 (`value = 0.9·z + 0.1·root_value`, persisted), so the real MCTS `root_value` is recoverable exactly (no proxy). The decisive control: at the only matched epoch (ep11), the **hard-target** `main1` had an *equal-or-worse* value head (mean|v|=**0.000**, sign-acc 0.520) than soft `main_2` (mean|v|=**0.072**), and `main1` still recovered to mean|v|=0.286 / 80% eval by ep35. Near-zero value at ep11 is **early-training immaturity common to both target modes**, not a soft-z artifact, and the per-label soft perturbation is only ≈`0.1·0.186 = 0.019` on a ±1 label — far too small to drive the eval drop. *Recommend keep 0.1* (optionally taper to 0.05 after ~ep10); root_value did carry signal early (corr 0.37 at ep5) but decays to near-noise (corr **0.11**) by ep11.
- **H2 — "`root_policy_temperature` made the policy much too diffuse": PARTIALLY supported in mechanism, REFUTED as the dominant cause.** The **raw** net prior is already maximally diffuse *before any knob* (top1 **0.108**, H 5.58 nats over a median **1010** legal moves); the ramp temperature only moves selection top1 0.108→0.083. Of the modest +0.65-nat controllable over-flattening, **Dirichlet noise is actually the slightly *larger* contributor (Δ=0.330 nats, 51%) than temperature (Δ=0.316 nats, 49%)** — so temperature is not "the" culprit. The genuinely anomalous pathology is a **diffuse policy + de-sharpening prior** (see below), dominated by a measured **2.1× branching confound** (n_legal 1010 vs main1's 475) plus an under-training feedback spiral.
- **Real diagnosis:** the disease is a **diffuse-policy / large-branching feedback loop on an immature net**, not value miscalibration. main_2's *recorded visit-policy training target* is far flatter than main1 at matched age (visits top1 **0.270 vs 0.569**) and the raw prior is **de-sharpening over its own training** (top1 0.333 ep5 → 0.108 ep11). A flat visit target starves MCTS of guidance and trains an even flatter prior next epoch.
- **Proposed (recommend, not applied):** the load-bearing lever is **trim Dirichlet `root_dirichlet_noise_fraction` 0.25 → 0.15–0.20**, paired with a smaller **early-ramp trim `root_policy_temperature_early` 1.25 → 1.15 (or 1.1)**. Keep `soft_z_lambda=0.1`, hold base temperature 1.1, and verify the branching/reuse regime. Apply **one lever at a time** — no clean ablation exists (≈7 levers changed vs main1 at once).

## Run state & method

- **Run:** `dense_cnn_restnet_main_2`, **epoch 11**, supervisor-halted 2026-06-11 10:38 by a separate "Codex value-head experiment." Treated **READ-ONLY**; no training launched, no run/config/code modified; CPU-only enforced (`CUDA_VISIBLE_DEVICES=""` before torch import, `map_location="cpu"`) to avoid GPU contention.
- **Compute path:** WSL venv `/root/.venvs/hexgt-build/bin/python`; run data under `/mnt/e/Hexo-BotTrainer/runs/`, repo under `/mnt/e/Hexo-BotTrainer-hexgt`.
- **Tooling (new):** `scripts/_m2_probe.py` — one forward/position, computing value calibration (vs hard `z=sign(stored)` and vs the soft target), exact `root_value` recovery, and a policy-diffuseness decomposition with the **ply-dependent** KataGo ramp `T(ply)=1.1+(1.25−1.1)·0.5^(ply/16)`. Auto-detects soft vs hard targets, so it runs identically on main_2 (soft) and main1 (hard). Summary: `scripts/_m2_summary.txt`. Per-target JSON: `scripts/_m2_probe_{m2_ep11,m2_ep5,m1_ep11,m1_ep35}.json`.
- **Datasets:** main_2 ep11 (4,746 positions / 60 games), main_2 ep5 (893), and the matched **A/B**: main1 ep11 (1,649) + main1 ep35 (8,964). Probe **validated** by reproducing the prior main1 ep35 value figure (mean|v|=0.286 here vs 0.297 in the [main1 review](RESTNET_EXPLORATION_VALUE_WORKFLOW.md)).
- **Adjudication:** a multi-agent workflow (`scripts/_wf_main2_exploration_value.mjs`) — independent forensic + data audit, then prosecute/defend on each hypothesis (diverse lenses), then synthesis. It **corrected** two of this analyst's initial framings (Dirichlet > temperature as the controllable lever; soft-z exonerated) and hedged the "collapse" language; those corrections are folded in below.
- **Builds on:** [RESTNET_EXPLORATION_VALUE_WORKFLOW.md](RESTNET_EXPLORATION_VALUE_WORKFLOW.md) (main1), [RESTNET_EXPLORATION_KNOBS.md](RESTNET_EXPLORATION_KNOBS.md), [RESTNET_OPENING_DIVERSITY.md](RESTNET_OPENING_DIVERSITY.md).

## Matched-epoch comparison (the load-bearing evidence)

| metric | main1 ep11 (hard, temp≈off) | main1 ep35 (mature) | **main_2 ep5** | **main_2 ep11** |
|---|---|---|---|---|
| value mean\|v\| (vs ±1) | 0.000 | 0.286 | 0.119 | **0.072** |
| value sign-acc (decisive) | 0.520 | 0.650 | 0.508 | **0.500** |
| value bias (mean_v−mean_z) | −0.073 | −0.021 | −0.097 | −0.073 |
| **raw prior top1** | 0.398 | 0.398 | 0.333 | **0.108** |
| raw prior H (nats) | 2.55 | 2.50 | 3.43 | 5.58 |
| **recorded visit top1** | 0.569 | 0.688 | 0.433 | **0.270** |
| n_legal median | 475 | 744 | 602 | **1010** |
| Δ_temp / Δ_dir (nats) | 0.632 / 1.011 | 0.453 / 1.234 | 0.486 / 0.825 | **0.316 / 0.330** |
| root_value corr w/ outcome | — (hard) | — | 0.367 | **0.109** |

**Eval vs SealBot** (128 games/pt on main_2; 64 on main1): main_2 ep5 = 33W/95L (**26%**), ep10 = 6W/122L (**4.7%**), mean turns 76.9→51.8. main1: ep3 = 4.7%, ep6 = 4.7%, ep9 = 20%, ep12 = 14%, ep15 = **66%**, ep30 = 78%, ep33 = 80%.

Two reads follow directly:
1. **Value-head deadness at ep11 is normal immaturity, not a main_2 regression** — main1 was *worse* (mean|v| 0.000) at the same age and recovered. The head is **under-confident, not biased** (mean|v|≈0.07 vs ±1; bias −0.073 identical to main1) — the defect `soft_z=0` would *not* fix.
2. **The genuinely anomalous signal is policy diffuseness** — main_2's prior (0.108) is 3.7× flatter than main1's (0.398) at matched age and *de-sharpens over its own training* (0.333→0.108), and its recorded visit target (0.270) is ~2× flatter than main1's (0.569). This is the one signature whose schedule overlaps the eval weakness.

> **Caveat on "collapse":** main1 itself sat at 4.7–20% across ep3–12 before reaching 66% (ep15) / 80% (ep33). main_2's ep10 = 4.7% is *within* that early-volatility band, and the eval is ep10 while the probe is ep11 (one self-play epoch apart). Treat the absolute early win-rate as weak evidence; the diffuse **target** and de-sharpening **prior** are the solid signals.

## Hypothesis 1 — `soft_z_lambda = 0`

**Live value: `soft_z_lambda = 0.1` (applied).** Plumbed end-to-end: `config.samples.soft_z_lambda` → `finalize_game_samples(...)` at `selfplay.py:762,1056`; `value = (1−λ)·z + λ·root_value` (`samples.py:218–228`). Stored shard `value` is the *blend*: mean|target|=**0.903**, only **3.0%** exactly at ±1 (vs 100% on hard main1). Because the blend is persisted, `root_value` is recovered **exactly** (`rv=(stored−0.9·hard_z)/0.1`; recovered-in-range fraction 1.0 — no net proxy, fixing main1's caveat).

**What the recovered `root_value` shows:** it carries real signal early then decays to near-noise — corr with the eventual outcome **0.367 (ep5) → 0.109 (ep11)**, sign-agreement **0.605 → 0.526**, mean|rv| only 0.186, std 0.31. So by ep11 the blended-in 0.1·`root_value` is ≈0.019 of low-information nudge on a ±1 label.

**λ sweep on realized targets (ep11):** mean|target| = 1.00 (λ0) · 0.95 (0.05) · 0.90 (0.10) · 0.76 (0.25) · 0.52 (0.50); the decisiveness cliff is at 0.5, and λ=0.1 keeps essentially all decisive signal.

**Verdict — REFUTED as a cause; keep 0.1 (do not zero).** The matched control is decisive: soft main_2's value head is *equal-or-better* than hard main1's at ep11, and the per-label perturbation (0.019) cannot explain the eval weakness. soft-z is the wrong variable. This **reverses neither** the [main1 recommendation](RESTNET_EXPLORATION_VALUE_WORKFLOW.md) (set 0→0.1) — it endorses staying at 0.1 — but adds the nuance that the blend's *benefit* is now early-only (the head is too immature for `root_value` to be a good target). A principled refinement is an **early-only taper toward 0.05 after ~ep10**, not a hard zero. *If the owner still prefers 0 for signal cleanliness, it is a near-wash — it will not fix the run.*

## Hypothesis 2 — `root_policy_temperature` too strong / too diffuse

**Live and now actually active.** `root_policy_temperature=1.1`, `…_early=1.25`, `…_halflife=16` — confirmed in config and in `diagnostics/dense_cnn.selfplay.epoch_000011.json` → `root_policy_temperature_control {base 1.1, early 1.25, halflife 16}`. The reuse-root bug that silently skipped temperature on main1 is fixed, so unlike main1 it **applies at every searched root**, with the ramp front-loaded on the opening (ply 0–5 mean T≈1.235).

**Decomposition (ep11, ply-dependent ramp):** total controllable prior over-flattening H(P_mix)−H(P_raw) ≈ **+0.646 nats**, split:

| stage | top1 | H (nats) |
|---|---|---|
| raw prior P_raw | 0.108 | 5.575 |
| + ramp temp (P_temp) | 0.083 | 5.891 |
| (fixed T=1.1) | 0.084 | 5.880 |
| (fixed T=1.25) | 0.058 | 6.200 |
| + Dirichlet mean (P_mix) | 0.063 | 6.221 |
| Dirichlet only, no temp | 0.081 | 6.011 |
| **realized search visits** | **0.270** | **2.546** |
| uniform over legal | — | 6.856 |

- **Temperature is *not* the largest controllable lever:** Δ_temp = +0.316 nats (49%) vs **Δ_dir = +0.330 nats (51%)** — Dirichlet noise edges it out, and the gap widens in deeper buckets (frac_dir 0.52–0.67). Removing the ramp recovers only ≈0.02–0.05 of selection top1.
- **Temperature flattens *selection*, not the *target*:** the 512-visit search re-sharpens by **+0.162 top1 / −3.03 nats** over the (temp+noise) prior, so the recorded visit target (0.270) is largely *upstream* of temperature.
- **The dominant drivers are elsewhere:** (1) an under-trained, **de-sharpening raw net** (top1 0.333→0.108 ep5→ep11 — temperature is fixed across epochs, so it cannot cause this trend); (2) a **2.1× branching confound** (n_legal 1010 vs 475; +0.75 nats of uniform entropy), itself partly a *chosen* consequence of lengthening games (`temperature_length_prior` 32→150, PCR, 384 games/epoch → 84% of recorded rows are deep `moves_left ≥ 48` positions).
- **The one legitimate residual:** the **T=1.25 early ramp** is the most aggressive flattening, newly-live, front-loaded on the most-collapsed opening plies (ply 0–5 prior top1 just 0.017 vs main1's 0.082), and the only A/B available (main1 with temperature effectively *off* reaching 80%) favors trimming it.

**Verdict — REFUTED as the dominant cause; the early ramp is a fair minor trim.** The literal claim ("temperature made the policy much too diffuse") fails on magnitude: the prior is already maximally diffuse before any knob, temperature is the *smaller* of the two controllable flatteners, and search re-sharpens the target it produces. Reduce the early ramp as cheap, well-isolated tuning — but bill it as minor, and **trim the Dirichlet fraction first** (the larger lever).

## Proposed settings

Recommendations only — **do not auto-apply**, and apply **one at a time** (no clean ablation exists). Each cites a measured number.

| Knob | Current (main_2) | Proposed | Evidence | Predicted effect | Confidence |
|---|---|---|---|---|---|
| `root_dirichlet_noise_fraction` | 0.25 | **0.15–0.20** (primary lever) | Dirichlet is the *larger* controllable flattener: Δ_dir 0.330 nats / 51% (> Δ_temp 0.316 / 49%); noise-only top1 0.081 vs raw 0.108; root-FPU now zeroed under noise on main_2 amplifies it; spread scales with n_legal=1010 | peaks the sampled policy and (downstream) the visit target more than temperature does — attacks the feedback spiral at its larger, branching-sensitive lever | Medium |
| `root_policy_temperature_early` | 1.25 | **1.15** (or 1.1 = ramp off) | ramp is the most aggressive flattening, front-loaded on ply 0–5 (T≈1.235, prior top1 0.017 vs main1 0.082); top1@T1.25 0.058 vs @T1.1 0.084; only A/B (main1 temp-off) hit 80% | recovers ~0.02–0.05 opening-ply selection top1; small, off-the-gradient (search re-sharpens −3.03 nats); will *not* fix collapse alone | Medium |
| `root_policy_temperature` (base) | 1.1 | **hold 1.1** | Δ at T=1.1 is small (top1 0.084 vs raw 0.108); temp is ≈49% of a *minority* over-flattening, dominated by branching; search re-sharpens +0.162 | preserves the opening diversity the ramp was added for; lowering now would mask value-immaturity, not fix it | Medium |
| `root_policy_temperature_halflife` | 16.0 | **keep 16.0** | Tply_mean 1.111 overall → asymptote dominates beyond opening; ramp effect concentrated in ply 0–16 | no material standalone effect; leave fixed to reduce changed-lever count | Medium |
| `soft_z_lambda` | 0.1 | **keep 0.1** (optional taper → 0.05 after ~ep10) | matched-epoch value head equal-or-better under soft (0.072) than hard main1 (0.000); bias identical −0.073; per-label perturbation 0.019; rv_corr 0.37(ep5)→0.11(ep11) | not the fix (wrong variable); 0.1 retains the early-epoch regularizer; a taper removes the now-near-noise blend without touching the (immature) head | High (that it is *not* the fix) |
| `forced_playout_k` | 2.0 | **keep 2.0** | visit target already very flat (0.270 vs main1 0.569) despite 512 visits; k=2 injects guaranteed exploratory visits that widen the recorded target; KataGo's value is 2.0 | holding avoids further flattening the target; revisit only if it stays flat after noise/temp trims | Low |
| games/reuse (`games_per_epoch` / `train_samples_per_epoch`) | 384 / 48000 | **keep; recompute reuse from *measured* yield** | config assumes ~150-ply / ~2.9× reuse, but 84% of ep11 rows are deep (`moves_left ≥ 48`), n_legal hit 1010, ~78 recorded decisions/game; main1's post-ep25 collapse was blamed on ~7.6× over-reuse | keeps reuse in the 1–4× KataGo band as games lengthen; no change unless measured reuse exits the band | Low |
| `policy_init_fraction` | 0.25 | **keep 0.25** (do not raise) | injects raw-prior opening diversity that compounds the early ramp on the same opening plies (ply 0–5 prior top1 0.017) | holding avoids double-paying opening diffuseness while primary trims are isolated | Low |

## Caveats & what would falsify this

- **No clean ablation.** ≈7 levers changed vs main1 at once (PCR, policy-init, soft-z, root-temp reuse-fix + ramp, heads_v3 split, root-FPU-under-noise, game-length prior 32→150). All recommendations are measured-cited proposals, not a confident root-cause fix.
- **Branching is a confound *and* a choice.** n_legal 1010 vs 475 mechanically flattens prior + visits, but it stems partly from deliberately longer games — shortening games would reduce it independent of the exploration knobs.
- **Eval-epoch ≠ probe-epoch.** The 4.7% reading is ep10; the policy/value probe is ep11. Near-terminal value slices (main_2 n=50–63, sign-acc ~0.60) are within ~1 SE of chance and are *not* load-bearing; only main1 ep35 (n=217, 0.97) is statistically solid. The matched ep11 comparison is the evidence.
- **Falsifiers:**
  1. A clean `soft_z_lambda=0` arm (else fixed) that still shows near-zero ep11 value + collapsing eval → soft-z fully exonerated (predicted). If instead it lifts mean|v| above ~0.15 or ep10 eval above ~15%, H1 is partially vindicated.
  2. main_2 run past ep25–35 recovering to mean|v|~0.28 + eval >30% (as main1 did) → ep10=4.7% was normal immaturity, and the "feedback spiral" framing is weakened.
  3. Trimming Dirichlet to 0.15–0.20 (alone) *not* raising visits top1 toward ~0.4 → noise is not the larger driver despite frac 0.51, and branching dominates even more.
  4. A run with branching held to main1's ~475 regime recovering prior/visit top1 *without* touching temp/noise/soft-z → branching confound confirmed dominant; all knob recs are second-order.
  5. Disabling the early ramp moving overall selection top1 by ≫0.045 → the P_raw decomposition is wrong and temperature is a larger driver than claimed.
  6. Prior continuing to de-sharpen past ep20 *with* a recovered value head and trimmed noise/temp → cause is the trainer/target pipeline (e.g. `policy_surprise_uniform_fraction=0.5`, PCR target quality), not an exploration knob.

## Reproduce

- Probes: `scripts/_m2_probe.py RUN_NAME CKPT_EPOCH SELFPLAY_EPOCHS MAX_GAMES OUT_JSON [LAMBDA]` under the WSL venv with `CUDA_VISIBLE_DEVICES=""`; e.g. `… dense_cnn_restnet_main_2 11 11 60 out.json 0.1`. Summary table: `scripts/_m2_summarize.py` → `scripts/_m2_summary.txt`.
- Forensics: `configs/dense_cnn_restnet_main_2.toml` vs `dense_cnn_restnet_main1.toml`; `samples.py:218–228`, `selfplay.py:762,1056`; `diagnostics/dense_cnn.selfplay.epoch_000011.json` (root-temp/PCR/policy-init control blocks); eval JSONs `dense_cnn.evaluation.epoch_000005.json`/`…010.json`.
- Adjudication workflow: `scripts/_wf_main2_exploration_value.mjs`.
