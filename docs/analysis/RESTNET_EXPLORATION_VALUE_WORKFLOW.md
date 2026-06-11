# RestNet Exploration & Value-Head Workflow — main1 Forensics

## Executive summary

Two owner hypotheses about the halted `dense_cnn_restnet_main1` run (epoch 35) were tested CPU-only against checkpoints and self-play shards. **Both fail as stated, for different reasons.** Hypothesis 1 ("set `soft_z_lambda` to 0") is a **no-op**: the live run already trains value on pure hard `z` (lambda=0, triple-confirmed by config, effective `_resume_config.toml`, and measured labels that are 100% at exactly ±1). Worse, the value-head diagnosis points the *opposite* direction — the head is **under-confident, not biased** (mean|v|=0.297 vs ±1 targets; bias mean_v−mean_z = −0.026), so a *small positive* lambda is the evidence-backed move, not 0. Hypothesis 2 ("`root_policy_temperature=1.1` is much too strong / makes the root too diffuse") is **refuted**: temperature contributes only **23% (+0.385 nats, −0.045 top1)** of prior flattening vs Dirichlet's 77%, and — decisively — a **reuse bug in the binary main1 actually executed silently skipped temperature on every reused root**, so on lockstep self-play the 1.1 flattening barely ran at all. The 512-visit search re-sharpens the played distribution well past the prior (top1 0.405→0.668), so root-prior diffuseness is largely washed out. **Proposed: leave `root_policy_temperature=1.1` (now that the reuse bug is fixed in tree); set `soft_z_lambda=0.1` to regularize the under-confident head; do not raise it past 0.25.**

## Run state & method

- **Run:** `dense_cnn_restnet_main1`, supervisor-halted 2026-06-10 at **epoch 35** for main_2 prep. Treated **READ-ONLY**; no training launched, no run files modified.
- **Compute:** CPU-only enforced (`CUDA_VISIBLE_DEVICES=""` before torch import, `map_location="cpu"`).
- **Checkpoints:** `E:/Hexo-BotTrainer/runs/dense_cnn_restnet_main1/checkpoints/epoch_000035.pt` (policy/value forwarding).
- **Shards:** compact self-play `.npz` shards; **220 shards / 35,482 positions** for the soft-z target distribution; **1,992 positions** (epoch_000035) for the policy-diffuseness decomposition.
- **What was measured:**
  1. **Forensics** — config/code lineage of `soft_z_lambda` and `root_policy_temperature`, plus the reused-root temperature bug in the exact binary main1 ran.
  2. **Policy diffuseness decomposition** — per-position prior entropy/top1 split into Δ_temp (T=1.1) vs Δ_dir (Dirichlet eps=0.25) vs realized search visits, by game phase.
  3. **Soft-z counterfactual** — measured target-label distribution and a lambda sweep {0, 0.1, 0.25, 0.5}, plus net value calibration vs hard z.
- **Reused tooling:** `scripts/_value_head_review.py`, `scripts/_softz_value_probe.py`, `scripts/_policy_ce_decompose.py` (extended to `scripts/_policy_diffuseness_probe.py`), and the soft-z counterfactual probe over the compact shards.
- **Note:** the standalone VALUE CALIBRATION analysis returned null; the equivalent calibration numbers were recovered from the soft-z probe's `net_value_calibration_vs_hard_z` block (same checkpoint, 35,482 positions).

## Hypothesis 1 — `soft_z=0`

**Effective current value: `soft_z_lambda = 0` (already).** Triple-confirmed: (a) `configs/dense_cnn_restnet_main1.toml` has no `samples.soft_z_lambda` key; (b) the effective `E:/Hexo-BotTrainer/runs/dense_cnn_restnet_main1/_resume_config.toml` has none; (c) code default `Model1SampleConfig.soft_z_lambda = 0.0`. A fourth confirmation: the pre-fix selfplay.py did not even plumb `soft_z_lambda` into `_finalize_game_samples`, so main1 could not have applied a nonzero blend.

**Mechanism** (`packages/dense_cnn_restnet/python/dense_cnn_restnet/samples.py:176-244`):
`value_target = (1 − lambda)·hard_z + lambda·root_value`, where `hard_z ∈ {+1, −1, 0}` is the decisive game outcome (side-to-move perspective) and `root_value` is the MCTS `search.root_value`. lambda=0 ⇒ pure ±1; lambda>0 convex-blends toward the search value, reducing label variance. Range-validated to [0,1]; blend stays in [−1,1] and flows through the unchanged 65-bin scalar target with no schema change.

**Lineage:**

| Config | soft_z_lambda | Notes |
|---|---|---|
| `dense_cnn_restnet_main1.toml` (live) | **0** (unset → default) | pure hard z |
| `_resume_config.toml` (effective) | **0** (unset) | byte-identical samples section |
| `config.py` default | 0.0 | code default |
| `hexgnn_model.toml` | 0.0 (explicit) | different family |
| `dense_cnn_restnet_main_2.toml` | **0.1** | owner next-gen target |
| `hexgt_model3.toml` | 0.5 | different family (hexgt) |

**Measured target distribution (35,482 positions, epoch 35):** hard_z labels are pure bimodal — mean −0.003, **mean_abs = 1.0, 100% at exactly ±1** — directly confirming lambda=0. Lambda sweep (de-bimodalization is monotone; mean stays ~0, so no bias is injected):

| lambda | target mean | target mean_abs | frac at ±1 | per-label shift |
|---|---|---|---|---|
| 0 | −0.003 | 1.000 | 1.00 | 0 |
| 0.1 | −0.005 | 0.915 | 0.009 | 0.085 |
| 0.25 | −0.009 | 0.786 | 0.009 | 0.214 |
| 0.5 | −0.016 | 0.573 | 0.008 | 0.427 |

**Value-head diagnosis (vs hard z):** mean_v_pred = −0.029 ≈ mean_z = −0.003 ⇒ **bias = −0.026 (NOT biased)**; **mean_abs_v = 0.297 vs ±1 targets ⇒ strongly UNDER-CONFIDENT** (consistent with prior ~0.36 memory, even lower at ep35). MAE 0.854 is variance off the bimodal labels, not directional bias. Decisive sign-accuracy 0.685.

**Verdict — REFUTED (as an action).** "Set it to 0" is a no-op: the run is already at 0. As a value judgment, the data argues *against* exactly 0 and *for* a small positive lambda: the head is under-confident (not biased), so the bias-amplification risk of soft-z does not apply, and label-smoothing toward calibrated search values is exactly the right regularizer. **Recommend lambda = 0.1** (softens the hardest spikes, mean_abs 1.0→0.915, per-label shift only 0.085, keeps essentially all decisive signal; matches main_2). Do not exceed 0.25 — at 0.5 only ~68% of labels stay decisive against a value proxy right only ~68% of the time. *Caveat:* `root_value` here is approximated by the net's own decoded value (mean|v|=0.297) because the raw MCTS root_value is not persisted in compact shards; real soft targets sit somewhat closer to ±1, so these mean_abs figures over-state the softening.

## Hypothesis 2 — `root_policy_temperature` too strong / too diffuse

**Decomposition (1,992 positions, epoch_000035, CPU).** Total prior→mixed flattening H(P_mix) − H(P_raw) = **+1.648 nats**, split against P_temp:

| Stage | H (nats) | top1 |
|---|---|---|
| Raw prior P_raw | 2.460 | 0.405 |
| + T=1.1 (P_temp) | 2.844 | 0.360 |
| + Dirichlet mean-noise (P_mix) | 4.108 | 0.270 |
| + Dirichlet real draws | 3.334 | — |
| **Realized search visits** | **1.052** | **0.668** |
| (uniform over legal) | 6.488 | — |

- **Δ_temp (T=1.1 alone) = +0.385 nats = 23% of flattening**, top1 0.405→0.360 (−0.045). Small, mild.
- **Δ_dir (Dirichlet on top) = +1.263 nats = 77%**, top1 0.360→0.270. Dirichlet dominates ~3:1.
- **Counterfactual T=1.0, Dirichlet on:** H=3.861, top1=0.304 — removing temperature recovers only +0.045 top1 (dH −0.246) because Dirichlet already dominates.
- **Search washes it out:** visits top1 0.668 ≫ prior 0.405 (+0.263); H 1.05 ≪ 2.46 (−1.41). The 512 visits + widening re-sharpen the played distribution far past the prior.
- **Phase-stable:** early/mid/late all give frac_from_temp 20–24%, frac_from_dir 76–80%, search-sharpens top1 by +0.22 to +0.27.
- **n_legal / widening context:** median n_legal 715 (p90 887, max 976). The raw prior is already concentrated (top1 0.40, H 2.46 vs uniform 6.49) over hundreds of legal cells. `widening_max_children=96` caps search to <14% of legal moves — a structural diffuseness lever on the **target**, independent of the prior.

**The decisive forensic finding — the reused-root bug.** The binary main1 actually executed (pre-commit 18bcad0) applied `root_policy_temperature` **only when a root was freshly constructed**. The lockstep `search()` reuse branch (`mcts.rs`) re-used the promoted native root by hash and called only `set_additional_visits` + `set_forced_playout_k` + `apply_root_dirichlet_noise` — **no `apply_root_policy_temperature`**. main1 ran `scheduler='lockstep'` with a persistent per-game search tree reused move-to-move within an epoch, so **every move after a game's first searched root used the raw, untempered prior**. The 1.1 flattening was effectively absent for nearly every position. Verified pre-fix at `18bcad0^:packages/hexo_models/dense_cnn/rust/src/mcts.rs`; fixed in 18bcad0 (working tree `mcts.rs:536` now applies it in the reuse branch). The temperature math itself is correct KataGo direction (`prior^(1/T)` renormalized; T>1 flattens; unit test `root_policy_temperature_flattens_priors` passes) — the defect was purely the missing call. A **secondary** change in the same commit: root FPU now auto-zeroes under Dirichlet noise; main1 (pre-fix) applied the 0.20 interior `fpu_reduction` at the root too, making main1's root exploration *narrower*, not broader.

**Verdict — REFUTED.** `root_policy_temperature=1.1` is a minor root-flattener: 23% of prior flattening, +0.385 nats, −0.045 top1, ~3× weaker than Dirichlet, and re-sharpened away by the 512-visit search. Moreover it **barely ran on main1** due to the reuse bug. The opening monoculture observed on main1 is consistent with searching the *sharp raw prior* (and a non-zeroed root FPU), i.e. *too little* exploration, not the over-flattening the hypothesis assumes. Lowering T would push in the wrong direction.

## Proposed settings

Recommendations only — do not auto-apply. Each cites a measured number.

| Knob | Current (main1) | Proposed | Evidence | Predicted effect | Confidence |
|---|---|---|---|---|---|
| `soft_z_lambda` | 0 (effective) | **0.1** | head under-confident: mean_abs_v=0.297 vs ±1, bias only −0.026; lambda=0.1 → target mean_abs 1.0→0.915, per-label shift 0.085, mean stays ~0 | gentler target for the under-confident head, near-zero signal loss; matches main_2 | Medium |
| `root_policy_temperature` | 1.1 (mostly unapplied via reuse bug) | **keep 1.1** (now bug-fixed) | Δ_temp = +0.385 nats = 23% of flattening, top1 −0.045; removing it recovers only +0.045 top1 | once correctly applied, a mild, benign flattener; lowering would sharpen an already-sharp prior | Medium-High |
| `root_policy_temperature_early` ramp | absent on main1 | optional 1.25 / halflife 16 (main_2 design) | main1 searched the raw sharp prior (reuse bug) → opening monoculture; early ramp adds opening breadth | modest extra opening diversity; low risk | Low-Medium |
| root FPU under noise | 0.20 interior applied at root too | **auto-zero at root** (18bcad0) | main1 suppressed unvisited root children with 0.20 fpu at root | broader root exploration, addresses opening narrowness directly | Medium |
| `root_dirichlet_noise_fraction` (eps) | 0.25 | leave; primary lever if more exploration wanted | Dirichlet = 77% of flattening (Δ_dir +1.263 nats) vs temp 23% | eps is the dominant root-exploration knob, not T | Medium |
| `root_dirichlet_total_alpha` | 10.83 | leave | prior memory: near-dead lever at 300–800 branching; per-action alpha tiny | negligible | Medium |
| `widening_max_children` | 96 | leave / monitor | caps search to <14% of median n_legal 715 — structural target-diffuseness lever | raising widens search target coverage; not implicated by the hypotheses | Low |
| `opening_temperature` / `opening_moves` | 1.4 / 5 | leave | anchors opening diversity at the played-move stage (separate from root prior) | already addresses opening diversity at move-selection | Low |

## Caveats & what would falsify this

- **root_value proxy:** the soft-z counterfactual approximates MCTS `search.root_value` with the net's own decoded value (not persisted in compact shards). This makes soft targets mildly self-referential and *over-states* softening; real targets sit closer to ±1. Falsifier: re-run with persisted `root_value` (requires re-instrumenting self-play) and recompute mean_abs per lambda.
- **Owner-swap optimism trap:** do **not** read the P0/P1 zero-sum (vP0+vP1) probe as value-head optimism — it is off-distribution-confounded by FirstStone parity. The valid metric is mean_v − mean_z = −0.026 (not biased); the head is under-confident, not optimistic.
- **Calibration upside is inferred, not trained:** the lambda=0.1 benefit is reasoned from the under-confident/not-biased diagnosis, not from a training run. A prior soft-control experiment (~lambda 0.9) did not move value MAE in 80 CPU steps — treat 0.1 as low-risk, not a guaranteed win. Falsifier: a controlled lambda=0 vs 0.1 training arm showing no calibration improvement (mean_abs_v unchanged).
- **Halted-run staleness:** all numbers are epoch-35 snapshots; the prior continues to sharpen with training, so the temp/Dirichlet split could drift. Falsifier: the decomposition at a later epoch showing Δ_temp ≥ Δ_dir.
- **Sample sizes:** 1,992 positions (diffuseness) and 35,482 (soft-z) are adequate but single-checkpoint; phase splits down to n=255 (late) are noisier.
- **Bug scope:** the reuse-bug finding is verified by code inspection at `18bcad0^` and the fix commit message, not by a behavioral diff of two binaries. Falsifier: a self-play trace showing temperature applied on reused roots in the pre-fix binary.

## Reproduce

- Forensics: read `configs/dense_cnn_restnet_main1.toml`, `E:/Hexo-BotTrainer/runs/dense_cnn_restnet_main1/_resume_config.toml`, `packages/dense_cnn_restnet/python/dense_cnn_restnet/{samples.py,config.py}`; `git show 18bcad0^:packages/hexo_models/dense_cnn/rust/src/mcts.rs` and `git show 18bcad0:...` for the reuse-fix diff (`mcts.rs:536`, `mcts_tree.rs:342-377,1079-1092`).
- Policy diffuseness: `E:/Hexo-BotTrainer-hexgt/scripts/_policy_diffuseness_probe.py` (extends `scripts/_policy_ce_decompose.py`) on `epoch_000035.pt` over epoch-35 compact shards. Set `CUDA_VISIBLE_DEVICES=""` before importing torch; `map_location="cpu"`.
- Soft-z counterfactual: `scripts/_softz_value_probe.py` over 220 epoch-35 shards (35,482 positions); reports hard-z distribution, lambda sweep, and `net_value_calibration_vs_hard_z`.
- Value calibration: `scripts/_value_head_review.py` (~35s/checkpoint CPU) for MAE / sign_acc / mean_v / mean_abs_v / reliability bins.