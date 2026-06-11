# Restnet Exploration & Value-Head Workflow: Soft-Z and Root-Policy-Temperature Analysis

## Executive summary

Two owner hypotheses about the halted `dense_cnn_restnet_main1` run (epoch 35) were tested against real CPU measurements on the run's own checkpoints and self-play shards. **Both hypotheses are essentially refuted by the data, but in instructive ways.** (1) "Set `soft_z=0`" is a **no-op**: the live run already trains its value head on the pure hard ±1 outcome (`soft_z_lambda` defaults to 0.0 and is unset in config, `_resume_config.toml`, and `manifest.json`). The value head is **under-confident, not biased** (`mean_v − mean_z = −0.026` ≈ 0; `mean_abs_v = 0.298` vs targets of ±1, i.e. ~3.4× compression toward zero). Soft-z would *lower* target magnitude further, so it does not fix the actual defect and a heavy lambda would aggravate it. (2) `root_policy_temperature=1.1` is **not "much too strong."** On 1,992 real epoch-35 root positions it accounts for only **0.385 nats (23%)** of the +1.65-nat root-prior flattening; Dirichlet (eps=0.25) supplies **1.263 nats (77%)**. Moreover a confirmed **reused-root bug** meant T=1.1 only applied to each game's *first* searched root for all of main1 — the intended flattening was largely **inactive**. The realized 512-visit search distribution (which is the policy *target*) is actually *sharper* than the raw prior (1.05 vs 2.46 nats), so root diffuseness never reaches the trained policy.

**Headline proposed settings (recommend, do not apply):** keep `root_policy_temperature=1.1` and adopt the main_2 **reused-root fix** (the real lever — it activates flattening at every searched root); adopt main_2's mild early ramp `1.25→1.1` (halflife 16) as low-risk; keep `soft_z_lambda=0` on main1 (nothing to change) and carry `0.1` (not 0.5) into the next generation as a light variance regularizer; treat **Dirichlet eps** (77% of flattening) and **widening_max_children** (search expands <14% of ~715 legal moves) as the dominant breadth/coverage levers if more or less root diffuseness is genuinely wanted.

## Run state & method

- **Run:** `dense_cnn_restnet_main1`, halted manually 2026-06-10 after epoch 35 (`supervisor_halted.flag`). All reads were **read-only**; no training launched, no run files modified.
- **Compute:** CPU-only enforced (`CUDA_VISIBLE_DEVICES=""` before torch import, `map_location="cpu"`). No GPU use.
- **Checkpoints:** `epoch_000035.pt` (primary), trend across `epoch_0000{24,28,30,32,33,34,35}.pt`.
- **Self-play shards:** compact `.npz` from selfplay epochs 34/35/36. Compact shards store the search visit distribution (`pol_act`/`pol_w`, the policy target) and hard z (`value`), but **not** the raw net prior or the MCTS `search.root_value` (both dropped at write). Prior and root-value were recomputed by forwarding the net.
- **Sample sizes:** value calibration n=32,364 decisive positions; soft-z lambda sweep n=35,482 positions (220 shards); policy diffuseness n=1,992 root positions (2,000 sampled, 8 dropped for <2 legal moves).
- **Tooling reused:** `scripts/_value_head_review_ep35.py` (calibration harness), `scripts/_softz_value_probe_cpu.py` (soft-z counterfactual), `scripts/_policy_diffuseness_probe.py` (entropy decomposition), plus a config/code forensics pass over `samples.py`, `config.py`, `mcts.rs`, `mcts_tree.rs`, `selfplay.py`, the run manifest and resume config.
- **Caveats baked in:** `epoch_000035.pt` is missing `aux_value_reduction.{conv,mlp}` weights, so the **moves_left and short-term-value aux heads read a randomly-initialized layer and are unreliable** (this fully explains their ~49% sign-acc / corr~0.07). The **main value head loads cleanly**, so value calibration is trustworthy. The soft-z probe uses the net's own decoded value as a proxy for MCTS `root_value` (the true value is not persisted); the real 512-visit root value would be sharper, so reported softening is a mild upper bound.

## Hypothesis 1 — "soft_z should be 0"

**Effective current value: `soft_z_lambda = 0.0` already.** Verified three independent ways: the key is **absent** from `configs/dense_cnn_restnet_main1.toml`, **absent** from the run's effective `_resume_config.toml`, and **absent** from `manifest.json` `model.config.samples`; `config.py` default is `0.0`. So the live value head is already trained on the pure ±1 outcome with no MCTS-root regularization.

**Mechanism** (`samples.py:182–230`): `value_target = (1−λ)·hard_z + λ·root_value`, gated `if λ>0 … else value_target = hard_z`. `hard_z = _winner_value(...)` is the pure +1/−1/0 outcome in the side-to-move perspective; `root_value` is the MCTS root value at the decision (no sign flip needed). A convex blend stays in [−1,1] and flows unchanged through the 65-bin `scalar_to_binned_target` mapping — **no head/schema change** for any λ.

**Lineage**

| Config | soft_z_lambda | Meaning |
|---|---|---|
| `configs/dense_cnn_restnet_main1.toml` (LIVE) | **0 (unset → default)** | Pure hard ±1, no MCTS-root regularization |
| `configs/hexgnn_model.toml` | 0 (explicit) | Pure hard z |
| `configs/dense_cnn_restnet_main_2.toml` | 0.1 (explicit, commit 18bcad0) | Light regularizer: 0.9·z + 0.1·root_value |
| `configs/hexgt_model3.toml` | 0.5 (explicit) | Heavy 50/50 blend |

**Target-distribution lambda sweep (n=35,482, net-value-as-root-value proxy)**

| λ | target mean_abs | frac at exactly ±1 | frac decisive (|t|>0.5) | shift from hard |
|---|---|---|---|---|
| 0 (live) | 1.000 | 100% | 100% | 0.000 |
| 0.1 (main_2) | 0.915 | ~0.9% | 100% | 0.085 |
| 0.25 | 0.786 | ~0.9% | 100% | 0.214 |
| 0.5 (hexgt_model3) | 0.573 | ~0.8% | 68.5% | 0.427 |

Soft-z's dominant measured effect is **reducing target variance / pulling labels off the ±1 endpoints**, not correcting any directional bias. Any λ>0 collapses the mass sitting exactly at ±1 from 100% to <1%.

**Value-calibration diagnosis (epoch 35 on epoch-35 self-play, n=32,364 decisive):**
- **Bias is negligible:** `mean_v − mean_z = −0.0275 − (−0.0017) = −0.026` ≈ 0 (a hair pessimistic; no meaningful optimism/pessimism).
- **Under-confidence is the real problem:** `mean_abs_v = 0.298` vs `mean|z| = 1.0` → predictions compressed ~3.4× toward zero. Reliability bins show monotonic mid-range under-shoot (pred [+0.3,+0.6) → mean_v 0.391 vs mean_z 0.569) while extremes are well-calibrated (pred [+0.6,+1.0) → 0.895 vs 0.898).
- **Concentrated early/long-horizon:** sign-acc falls 93.4% ([0,5) moves-left) → 61.7% ([80,1000)), mean_abs_v 0.704 → 0.260; 80% of mass sits at moves_left≥40.
- **Still improving:** epochs 24→35 sign-acc rises 52.1%→65.4%, value_CE falls 0.823→0.637; under-confidence is persistent across training, not a late artifact.

**The owner-swap optimism number is excluded from the verdict.** The vP0+vP1 probe reads +0.395 at ep35 and swings +0.75→−0.37 across the trend, but it is **off-distribution-confounded (FirstStone parity)** and MUST NOT be read as optimism. The per-color spread (mean_z player0=−0.355, player1=+0.365) is a real structural game asymmetry the head reproduces (compressed), not a bias.

**Verdict: REFUTED (as moot / pointing the opposite way).** "Set soft_z to 0" is a no-op — main1 is already at λ=0. If the *intent* is "hard z is optimal, keep soft-z off," the data refutes that as the best setting: the head is under-confident, not biased, and softening targets toward its own already-shrunk values does not cure (and at λ=0.5 would aggravate) under-confidence. A **small λ=0.1** is mildly preferable for the next generation as a low-risk variance regularizer (frac_decisive stays 1.0 through λ=0.25), but the larger lever for under-confidence is **value-loss weighting / target sharpness**, not soft-z magnitude.

## Hypothesis 2 — "root_policy_temperature too strong / too diffuse"

**Entropy decomposition on 1,992 real epoch-35 root positions** (P_raw = net prior; P_temp = after T=1.1; P_mix = after T + Dirichlet; P_visits = realized 512-visit search = the policy target):

| Distribution | Entropy (nats) | top-1 |
|---|---|---|
| Raw prior | 2.46 | 0.405 |
| + Temperature (T=1.1) | 2.844 | 0.360 |
| + Temp + Dirichlet (full stack) | 4.108 | 0.270 |
| Dirichlet-only counterfactual (T=1.0) | 3.861 | 0.304 |
| **Search visits (policy TARGET)** | **1.052** | **0.668** |
| Uniform over legal | 6.488 | — |

- **Temperature is the minor contributor.** Of the +1.648-nat total flattening, Δ_temp = **+0.385 nats (23%)** vs Δ_dir = **+1.263 nats (77%)** — Dirichlet does ~3.3× more flattening. Stable across phases (temp share 24%/23%/20% early/mid/late).
- **Removing temperature entirely barely sharpens the root:** Dirichlet-only drops H from 4.108→3.861 (only 0.246 nats recovered; top-1 0.270→0.304). T=1.1 is a mild softmax temperature (1/T=0.909) and does **not** "stack" strongly with Dirichlet.
- **Search SHARPENS — diffuseness never reaches the target.** P_visits has entropy **1.05 nats / top-1 0.668**, *sharper than even the raw prior* (2.46 / 0.405). The 512 visits + PUCT wash out injected Dirichlet noise: search top-1 exceeds prior top-1 by +0.263 and search entropy is 1.41 nats below the prior. Diffuseness lives entirely in the **root prior used to seed exploration**, not in the played move or the trained policy target.
- **Widening dominates coverage.** Legal sets are ~715 median (p90 887, max 976) across all phases, but `widening_max_children=96` caps the tree to ≤96 children — search physically expands **<14% of legal moves**. This branching-vs-widening gap is a larger structural diffuseness/coverage constraint than temperature or Dirichlet.
- **Realized Dirichlet draws are stable:** mean H_mix over 8 real seeded draws = 3.334 nats, within-position std 0.049 — a realized draw concentrates more than E[Dir] (3.33 vs 4.11) but the per-draw spread is tiny.

**The decisive finding — the reused-root bug.** Forensics on `packages/hexo_models/dense_cnn/rust/src/mcts.rs` confirm that in the lockstep `search()` reuse branch, a promoted/reused root re-applied `set_additional_visits`, `set_forced_playout_k`, and Dirichlet noise but **did NOT call `apply_root_policy_temperature()`**. Fresh roots get the temperature inside `RustSearch::new`; reused (subtree-promoted) roots carried the **raw eval priors at effective T=1.0**. So `root_policy_temperature=1.1` only ever applied to **each game's FIRST searched root**; every subsequent reused root searched the un-flattened, sharper prior. The fix lives in commit **18bcad0 (2026-06-10 21:32 UTC)**, ~4h **after** the run's last launch (17:46 UTC), and no relaunch occurred before the halt — so **the buggy binary ran for all of epochs 1–35**. The pre-fix `search()` pyo3 signature didn't even accept a per-root temperature vector and `selfplay.py` passed only the scalar. **Dirichlet root noise was applied correctly at every searched root** (including reused), so root exploration noise was intact; only prior-flattening temperature was skipped. Direction in code is correct and matches KataGo (P′ ∝ P^(1/T), T>1 flattens) — the only defect was the skip.

**Verdict: REFUTED / PARTIAL.** `root_policy_temperature=1.1` is **not** "much too strong": it supplies only 0.385 nats (23%) of root flattening, vanishes entirely from the played move and the policy target (search is *sharper* than the prior at 1.05 nats), and — critically — was **largely inactive on main1 anyway** because the reused-root bug skipped it on all but each game's first root. If anything, main1 explored *less* at the root than the config nominally specified. The dominant flattening knob is **Dirichlet eps (77%)**; the dominant coverage constraint is **widening_max_children**.

## Proposed settings

Recommendations only — not instructions to apply. Each row cites a measured number.

| Knob | Current (main1) | Proposed | Evidence | Predicted effect | Confidence |
|---|---|---|---|---|---|
| reused-root temperature fix | buggy (skipped on reused roots) | **adopt fix (commit 18bcad0)** | T=1.1 applied only at each game's 1st root for all of epochs 1–35; reused roots ran at T=1.0 | Activates the intended ~1.1× opening-prior flattening at *every* searched root — the real lever behind H2 | High |
| `root_policy_temperature` | 1.1 | **keep 1.1** | Only 0.385 nats / 23% of root flattening; removing it entirely recovers just 0.246 nats; gone from target | Mild root breadth; not "too strong" — no change warranted | High |
| `root_policy_temperature_early` / `_halflife` | unset (no ramp) | **adopt 1.25 → 1.1, halflife 16** (main_2) | Whole temp contribution is 0.385 nats; a 1.25→1.1 ramp is a small, KataGo-aligned early-breadth nudge | Slightly more opening diversity at low risk; aimed at post-ep25 opening monoculture | Medium |
| `soft_z_lambda` (main1) | 0 (unset) | **keep 0** | "Set to 0" is a no-op; bias = −0.026 ≈ 0 | No change | High |
| `soft_z_lambda` (next gen) | — | **0.1, not 0.5** | At λ=0.5 frac_decisive drops to 68.5% / mean_abs 0.573; head already under-confident (mean_abs_v 0.298) | Light variance regularization without erasing decisiveness; avoids worsening under-confidence | Medium |
| value-loss weighting / target sharpness | (current) | **investigate as the real calibration lever** | mean_abs_v 0.298 vs ±1 (~3.4× compression); soft-z moves the *wrong* way | Address under-confidence directly rather than via soft-z | Medium |
| Dirichlet `eps` (noise fraction) | 0.25 | **primary knob if root diffuseness must change** | 77% (1.263 nats) of all root flattening | Lower eps → sharper root seeding (and vice-versa); ~3.3× the leverage of temperature | High |
| `root_dirichlet_total_alpha` | 10.83 | leave (de-prioritize) | Prior memory: dead lever at 300–800 branching; realized-draw spread std 0.049 nats | Negligible at this branching factor | Medium |
| `widening_max_children` | 96 | **the structural coverage lever** | Legal sets median 715 (max 976); search expands <14% of legal moves | Raising widens tree breadth far more than any temperature/noise change; costs compute | Medium |
| played-move / opening temperature | temp 1.0 adaptive; opening 1.4 over 5 plies | leave | Search visits (target) entropy 1.05 nats / top-1 0.668 — already sharp; diffuseness is at the prior, not the played move | No change indicated by this data | Low–Medium |

## Caveats & what would falsify this

- **Owner-swap optimism trap:** the vP0+vP1 probe (+0.395 at ep35, swinging +0.75→−0.37) is off-distribution-confounded by FirstStone parity / opponent-last-turn ablation and is **excluded** from the bias verdict. Anyone reading it as optimism would mis-diagnose the head.
- **Soft-z root_value proxy:** the lambda sweep uses the net's decoded value (MCTS `root_value` isn't persisted). The true 512-visit root value is sharper, so real soft-z targets soften *less* than shown — the bimodality reductions are a mild upper bound. A probe that re-runs MCTS to capture true root_value would tighten these numbers.
- **Aux-head numbers are artifacts:** `epoch_000035.pt` is missing `aux_value_reduction.{conv,mlp}`; moves_left (corr 0.07, over-predicts +73 decisions) and stvalue (sign-acc ~49%, mean_pred −0.45) feed a randomly-initialized layer and must NOT be read as trained-head defects. The main value head loads cleanly.
- **Root-FPU under noise not separately verified:** main_2 claims root FPU is auto-zeroed under Dirichlet (`rootFpuReductionMax=0`) as code behavior; this was not confirmed for the main1 binary. Any hypothesis leaning on root-FPU should verify whether the pre-fix binary zeroed it or applied the full 0.20.
- **Halted-run staleness:** all numbers are epoch-35 snapshots of a halted run; behavior on a resumed/fixed binary (especially with the reused-root fix active) may differ — that is precisely why the fix is the highest-confidence recommendation.
- **Sample limits:** diffuseness from 1,992 root positions (epochs 34–36); calibration/soft-z from ~32–35k positions of the same epochs — representative of late-main1 self-play, not earlier regimes.

## Reproduce

All scripts force CPU (`CUDA_VISIBLE_DEVICES=""`, `map_location="cpu"`); run from `E:/Hexo-BotTrainer-hexgt` with the heads_v3-compatible Python/torch noted in each header.

- **Value calibration:** `python scripts/_value_head_review_ep35.py` → `scripts/_value_head_review_ep35_out.json`, `_value_head_review_ep35.log` (~35s/ckpt; trend uses epochs 24–35).
- **Soft-z lambda sweep:** `python scripts/_softz_value_probe_cpu.py` → `scripts/_softz_value_probe_cpu_out.json` (n=35,482, 220 shards).
- **Policy diffuseness decomposition:** `python scripts/_policy_diffuseness_probe.py` → `scripts/_policy_diffuseness_probe_out.json` (n=1,992 root positions).
- **Forensics sources:** `packages/dense_cnn_restnet/python/dense_cnn_restnet/samples.py:182–230`; `config.py:121,499–503`; `packages/hexo_models/dense_cnn/rust/src/mcts.rs` (reuse branch ~343–352 pre-fix, 527–541 post-fix; `mcts.rs:185–197` `root_temp_for`); `mcts_tree.rs:1079–1092` & `342–377` (temperature transform); `git show 18bcad0` / `18bcad0^` for the reused-root fix; run `manifest.json` and `_resume_config.toml`.