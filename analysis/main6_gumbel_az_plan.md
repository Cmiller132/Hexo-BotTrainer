# main_6: Gumbel AlphaZero — Definitive Implementation Plan

**Author:** lead engineer (hexfield)
**Date:** 2026-06-30
**Status:** PLANNING — no code written yet. This document is the build spec.
**Scope:** A fresh run `main_6` that trials Gumbel AlphaZero (Danihelka et al., ICLR 2022, *"Policy Improvement by Planning with Gumbel"*) on the hexfield system, staged so each step is independently measurable and reversible.

---

## 0. TL;DR verdict (read this first)

Gumbel AlphaZero has **three separable mechanisms**: (#1) Gumbel-Top-k root sampling, (#2) deterministic non-root selection via `argmax[logits + σ(completedQ)]`, and (#3) the improved low-variance training target `π'(a) = softmax(logits(a) + σ(completedQ(a)))`. The diagnosed main_5 bottleneck is **target-estimator variance** (`KL(target‖prior) = 0.672 nats`, top-1 prior-vs-search agreement stuck at `0.568` over 40 epochs, median visited support 7 where ±1 visit flips the argmax ~50% of the time). **Mechanism #3 lands directly on that bottleneck; mechanisms #1/#2 buy low-visit sample-efficiency, which main_5 — running 1024-visit full search — is NOT bound by.**

**Therefore the plan is staged and #3-first:**
- **Stage 0** — offline feasibility + instrumentation gate. We CANNOT compute the full `softmax(logits + σ(completedQ))` target offline because **raw policy logits are not logged** (only softmaxed `root_prior_policy` weights). So Stage 0 (a) measures a *degenerate* completedQ-blend target on existing shards to estimate variance reduction, and (b) specifies the exact new export column (raw logits) needed. This gates everything.
- **Stage 1** — completed-Q target only, PUCT search **unchanged**. The faithful target `softmax(logit + σ(completedQ))` requires raw policy logits, which **do not exist anywhere in Rust today** (the Rust tree only ever sees softmaxed/tempered/noised `priors`; see §0.1 and Appendix A). Stage 1 therefore has TWO load-bearing prerequisites that Stage 0 does not: (i) plumb raw logits **Python→Rust** (new `priors_logits_bytes` evaluator-reply field; §5.0.4), and (ii) compute the target either online in Rust once logits arrive, OR offline in Python from the Stage-0 raw-logit export column. Low risk to search semantics, attacks the bottleneck. Flag-gated, default-off.
- **Stage 2** — Gumbel-Top-k root + Sequential Halving + deterministic non-root (the throughput play), reconciled with the async continuous scheduler via **root-only SH** (not a full per-root round barrier). Only pursued if Stage 1 proves out AND throughput is the new constraint.

The skeptic's strongest points are accepted and designed around (see §1.3): the offline logit gap is real (→ Stage 0 export spec AND a Python→Rust logit-plumbing spec, §5.0.4), the value-Q saturation re-entry risk through the completedQ fallback is real (→ visit-weighted node-value fallback + min-visit floor + saturation kill-switch), the eval apparatus may not resolve the expected delta (→ A/B on the KL gap and top-1 agreement, which ARE resolvable, not on raw Elo alone), and the SH async-port cost is severe (→ deferred to Stage 2, root-only).

### 0.1 CRITICAL FACT: there are NO raw logits in Rust (load-bearing for Stage 1)

Verified against source (2026-06-30):
- The network policy **logits** are produced and softmaxed entirely in **Python**: `inference.py:466` `logits = out['policy'].float()` → `inference.py:477` `priors = torch.softmax(masked, dim=1)`. Only the softmaxed `priors[legal]` is serialized as `priors_bytes`.
- Across the boundary, `payload.rs` (`parse_chunk_reply`, ~175-181) parses `priors_bytes` into `RustEvaluation.priors: Vec<(PackedCoord, f32)>` (**probabilities**, `cache.rs:18-24`).
- `owned_root_from_evaluation()` (tree.rs:1248-1317) builds root candidates **directly from `evaluation.priors` probabilities** (tree.rs:1258-1272), then immediately applies `apply_root_policy_temperature_to` (powf) + normalize + Dirichlet. **There are no logits anywhere in the Rust tree.**

Consequence: the Stage-1 claim "compute `softmax(logit + σ(Q))` online in Rust, logits in hand" is **false as written** — the Rust tree cannot reconstruct logits from tempered/noised probabilities faithfully. Stage 1 must EITHER (a) add a new Python→Rust `priors_logits_bytes` field so Rust receives raw logits (the "online in Rust" path, §5.0.4), OR (b) compute the target in Python from the Stage-0 raw-logit export column (the "offline" path). This is a different and larger change than the Stage-0 Rust→Python export column; both are now specified explicitly below.

---

## Table of contents

1. [Executive summary & suitability verdict](#1-executive-summary--suitability-verdict)
2. [Background: the main_5 bottleneck & whether Gumbel addresses it](#2-background-the-main_5-bottleneck--whether-gumbel-addresses-it)
3. [Factors considered (search-theory / systems-perf / game-model-fit) + upside/drawback tables](#3-factors-considered)
4. [Design decisions specific to THIS game/model](#4-design-decisions-specific-to-this-gamemodel)
5. [Staged rollout (Stage 0 / 1 / 2) with exact files, functions, config knobs, go/no-go](#5-staged-rollout)
6. [Performance plan: scheduler vs Sequential-Halving tension](#6-performance-plan)
7. [Verification passes](#7-verification-passes)
8. [Risks, fallbacks, kill-criteria per stage](#8-risks-fallbacks-kill-criteria)
9. [Concrete config sketch: configs/hexfield_main_6.toml](#9-concrete-config-sketch)

---

## 1. Executive summary & suitability verdict

### 1.1 The three mechanisms, restated precisely

| # | Name | What it replaces | Where in code | Risk | Lands on bottleneck? |
|---|------|------------------|---------------|------|----------------------|
| 1 | **Gumbel-Top-k root** sampling of m candidates (draw `g(a)~Gumbel(0)`, take top-m of `logits(a)+g(a)`) | Root Dirichlet noise + move-temperature sampling | `apply_root_dirichlet_noise()` tree.rs:582; `owned_root_from_evaluation()` tree.rs:1248; root-noise seed stream `SEED_STREAM_ROOT_NOISE` search.rs:43; requires logits in Rust (§0.1/§5.0.4) | HIGH (scheduler) | NO (low-visit lever) |
| 2 | **Deterministic non-root** `argmax[logits + σ(completedQ)]` — no c_puct/FPU/widening | PUCT `select_or_materialize_edge()` | tree.rs:831 (score formula ~849-855) | HIGH (re-saturation) | NO (low-visit lever) |
| 3 | **Improved target** `π'(a)=softmax(logits+σ(completedQ))` | Visit-normalized policy target | export site `build_search_result_payloads()` search.rs:1886 (calls `visit_policy()` search.rs:2330); consumed `q_policy`→`cell_q` samples.py:266-275 | MEDIUM | **YES (variance)** |

> **Function/line attribution note (verified 2026-06-30):** the per-decision *export* region (where `visit_policy_q_bytes` and `root_prior_policy_*` are set, search.rs:1950-1960) lives in **`build_search_result_payloads()` (search.rs:1886-1996)**, which *calls* `visit_policy()`. The actual weight-computation logic is `visit_policy()` (search.rs:2330) and `pruned_visit_policy()` (search.rs:2370). Earlier drafts cited the export region as "`visit_policy()` search.rs:1886-1960" — both the function name and the upper bound were wrong. All export-column work targets `build_search_result_payloads()` (search.rs:1886); weight math targets `visit_policy()`/`pruned_visit_policy()`.

The durable, high-visit win is **#3**. #1/#2 are sample-efficiency at low sims.

### 1.2 Synthesized suitability verdict across the four lenses

| Lens | Verdict | One-line synthesis |
|------|---------|--------------------|
| search-theory | suitable-with-caveats | #3 swaps a discontinuous multinomial-count estimator (the ±1-visit argmax flip) for a smooth monotone function of backed-up Q means → genuine variance reduction at our 1024 budget. |
| systems-performance | conditional | #3's *training* cost is ~0%; its *self-play* cost is a new per-Full-root softmax over visited support on the host per-decision path (O(legal), the post-flex host bottleneck — measure, §6.3/§7.5). canonical SH (#1/#2) is a multi-week barrier rewrite that *regresses* a FLOP-bound forward. |
| game-model-fit | suitable-with-caveats | Q∈[-1,1] is ideal for σ; late-game branching *grows* (337→~1000) so SH does not starve; but only ~33% Full moves benefit and σ/m/fallback are all new untuned knobs. |
| adversarial skeptic | conditional | Bottleneck is teacher *strength*/finite-visit variance addressed by free config levers; offline #3 is logit-blocked; completedQ fallback re-opens main_4's value→policy coupling; expected Elo may sit below eval SE. |

**Consensus:** Adopt **#3 incrementally and flag-gated; defer #1/#2.** Run the **free config levers first** (`pcr_full_proportion 0.33→0.5`, `c_puct 1.5→1.1`, `dirichlet 0.25→0.20`) and re-measure the 0.67-nat gap — if they close much of it, #3's marginal ROI shrinks and we may stop there.

### 1.3 The skeptic's strongest points and how this plan addresses each

1. **"The bottleneck is teacher strength / finite-visit variance, not target representation."** — Partly true. The free config levers (sims, c_puct, pcr_full) add real visits and are sequenced **first** (Stage 0 precondition). But #3 still reduces the *extraction* variance of the same statistics: on a 7-way near-tie, `N(a)/N` is a discontinuous count ratio while `softmax(logit+σ(Q))` is a continuous function of Q means. We commit to A/B-measuring the residual KL gap *after* the config levers, so #3 is only built if a residual variance gap survives.

2. **"Offline #3 is blocked — only softmaxed priors are logged, and raw logits do not exist in Rust either."** — Accepted as a hard fact (confirmed: `selfplay.py:162-190` reads `visit_policy_q_bytes` and `root_prior_policy_weights_bytes`, never raw logits; `samples.py:46` stores `policy_surprise` from softmaxed weights; and per §0.1 the Rust tree only holds softmaxed `priors`, so "logits in hand" inside Rust is **not true today**). Stage 0 therefore does NOT attempt a faithful offline target — it measures a degenerate Q-blend (feasible from `q_policy`) and adds the Rust→Python **raw-logit export column** (§5.0.3). The faithful target is then computed by ONE of two routes, both fully specified: **(a) offline in Python** from the exported raw-logit column, or **(b) online in Rust** *after* plumbing logits Python→Rust via a new `priors_logits_bytes` evaluator-reply field (§5.0.4). Route (a) is the lower-risk default for Stage 1's A/B; route (b) is required only if/when Stage 2's Gumbel-root sampler needs logits in the tree.

3. **"completedQ fallback re-imports main_4 value-Q saturation."** — Accepted. The fallback for unvisited actions is the channel through which an overconfident value head writes policy targets on balanced nodes — exactly main_4's failure surface. Mitigations are first-class (§4.5): visit-weighted **node-value blend** (not a bare child `eval_value` read), a **min-visit floor** before a candidate contributes target mass, keep `root_fpu_reduction=0.2` / no `lazy_widening+root_fpu=0` stack, and a **turn-0 |Q| saturation kill-switch** ported from the main_4 probe.

4. **"Expected Elo win is below the eval SE (~48-90 Elo)."** — Accepted for raw Elo. So the **primary** Stage-1 go/no-go is the *resolvable* metrics: KL(target‖prior) gap and top-1 prior-vs-search agreement on the fixed forward-pass probe (`_scratch_klgap_main5.py`), plus root_value saturation. Elo vs the `main2_ep45` anchor is a *secondary, directional* check with raised eval power, not the gate.

5. **"SH async-port cost is severe for the least-needed win."** — Accepted. #1/#2 are **deferred to Stage 2** and, when built, use **root-only SH** (no per-root non-root round barrier) to avoid rewriting the global-flush / rayon `par_iter_mut` hot path.

---

## 2. Background: the main_5 bottleneck & whether Gumbel addresses it

### 2.1 The diagnosed bottleneck (ep100 audit)

From `analysis/main5_full_health_audit.md` / `_scratch_klgap_main5.py`:

```
loss_policy = 1.920 = H(target) 1.248  +  KL(target‖prior) 0.672
                       └ irreducible      └ THE GAP (target-estimator variance)
                         positional entropy
```

- `KL(target‖prior) = 0.672 nats` ≈ **10×** the net's own D6 self-KL floor (`0.071`), and **capacity-invariant** (0.70 main_4 → 0.73 main_5 at 1.76× params).
- Top-1 agreement (prior argmax vs search argmax) stuck **0.548 (ep60) → 0.568 (ep100)** — 43% disagreement that has not been trainable away over 40 epochs (noise signature, not learnable signal).
- Median visited support ≈ **7 moves** at `H≈1.03 nats`; on these near-tied roots, ±1 visit on second place flips the count argmax ~**50%** of the time.
- Search is **healthy**: no value-Q saturation (root_value_mean ∈ [-0.019, -0.000] vs main_4's |Q|→0.79), length rising 81→103, loss–Elo coupling correct.
- Capacity is **not** binding: +33% width bought only +13% eff-rank; net is 61% idle.

**Conclusion:** the limit is **target-signal quality (variance), not bias, not capacity, not exploration.**

### 2.2 Why #3 is on-axis and #1/#2 are off-axis

- **#3 (target).** `π_visit(a)=N(a)/N` is a multinomial-count estimator whose top-mass is a **discontinuous** function of the counts (the ±1-visit flip). `Var[N(a)/N] ≈ p(1-p)/N`, but the discontinuity is what makes the argmax noisy on near-ties. `softmax(logit + σ(completedQ))` replaces it with a **smooth, monotone** function of backed-up Q means (`Q(a)=value_sum/visits`, variance `≈Var(v)/n_a`), so the relative ordering is preserved with far less Monte-Carlo jitter on exactly the near-tied/diffuse positions that dominate the 0.67-nat gap. **This is the load-bearing fix.**
- **#1/#2 (root sampler + non-root selection).** These buy sample-efficiency at *low* sims (the paper shows wins at ~2-200 sims). main_5 runs **1024 full / 192 fast** — past that regime — and search health says coverage/exploration is not the binding constraint. They are largely orthogonal to THIS bottleneck.

### 2.3 Honest caveat

If the residual after the free config levers turns out to be the `H(target)=1.248` term (true multi-move-equivalence positional entropy, irreducible), **#3 cannot help that** — it only compresses the 0.67-nat variance term. Stage 0 exists to estimate how much of the 0.672 is compressible before we commit engineering.

---

## 3. Factors considered

### 3.1 Search-theory

**Upsides**

| Upside | Why it holds here |
|--------|-------------------|
| #3 attacks the *exact* estimator variance the audit named | smooth Q-mean function vs discontinuous count ratio on 7-move near-ties |
| Required Q data already plumbed end-to-end | `visit_policy_q_bytes`→`q_policy`→`cell_q`→`collate_training`→`losses.py`, validated finite/[-1,1] (`samples.py:268-271`) |
| We're at the high-visit regime (#3's durable-win regime) | 1024 full visits → low-variance Q per surviving action |
| Orthogonal to & stackable with queued config levers | attacks the *residual* the levers can't remove (they keep the count estimator) |
| No serve-path parity break | training-target-only; parity tests don't load it |

**Drawbacks**

| Drawback | Mitigation |
|----------|-----------|
| Offline adoption BLOCKED — only softmaxed priors logged; `root_policy_temperature 1.1/1.15` + remaining_priors padding make `log(weight)` inversion unfaithful; AND raw logits are not in Rust either (§0.1) | add Rust→Python raw-logit export column (§5.0.3) and compute target offline in Python; OR plumb logits Python→Rust (§5.0.4) and compute online |
| completedQ fallback = raw value-head read → main_4 re-saturation channel | visit-weighted node-value blend + min-visit floor (§4.5) |
| σ() parameterization must be re-tuned for Q∈[-1,1] (not KataGo's scale) | calibrate against `Q_UTILITY_WIDTH=2.0` (§4.2) |
| If residual is `H(target)` not the KL gap, #3 can't help | Stage 0 estimates compressible fraction first |

### 3.2 Systems-performance

**Upsides:** completed-Q-only target is **near-zero scheduler cost** (Python collate/loss change; Q already exported). #1/#2 are root-only/selection-only and don't break golden-vector serve identity. A **playout-cap SH variant** (fixed per-root budget, pruning at export via existing `pruned_visit_policy`) captures much of the benefit while leaving the async monolithic-target scheduler intact.

**Drawbacks:** Canonical SH porting is **large and invasive** — per-slot state must expand `(target, completed, in_flight)` → `(root_candidates, current_round, round_budget, completed_per_candidate)`; a **new global round barrier** must gate `par_iter_mut` (`select_continuous_pass` search.rs:1412, `.par_iter_mut()` at search.rs:1423); flush_target must scale per-round; the async "stale prefetch discarded" guarantee (search.rs:979-990) is **unproven** under round barriers. The barrier serializes the slowest slot per round and small early-round flushes **starve a FLOP-bound forward** → real pos/s regression — for a lever that targets a non-binding constraint.

### 3.3 Game-model-fit

**Upsides:** branching **grows** late-game (337→~1000) — the opposite of SH-starvation; Q∈[-1,1] makes σ numerically clean (no Q rescale); `cell_q` head already trains on per-action searched Q so the pipeline is half-built; Gumbel-Top-k replaces **both** Dirichlet AND temperature with one per-root draw, removing several un-instrumented knobs.

**Drawbacks:** SH fights the per-root-async scheduler; logits not in buffer (offline blocked); #2 re-opens value-Q saturation if fallback reads child eval; only ~33% Full moves benefit; `m`, σ's `c_visit/c_scale`, and the fallback are all new untuned knobs and tuning iterations cost epochs; SH's m-candidate discovery is **redundant** with the 96-child nucleus + `forced_playout_k=1.0` (must disable one).

---

## 4. Design decisions specific to THIS game/model

### 4.1 Candidate count m vs branching factor

- Opening forced to `(0,0)`; early Nlegal ≈ 337-777; mid/late ≈ 700-1000 (`LEGAL_RADIUS=8` halo grows). Branching **does not collapse** late, so SH does not starve.
- **Decision: m = 16** for Stage 2 SH. With budget 1024 over `log2(16)=4` halving rounds, equal-per-round gives ~256 visits/round; the final 2 candidates get ~256+ visits each — deep enough for the tactical conversion the run needs. `m=8` wastes the 1024 budget given ~700-1000 legal; `m=32` (5 rounds, ~205/round) is the upper viable bound. Make `m` a config knob; default 16.
- m applies **only to Full moves**. Fast (192) and Init (1) skip SH entirely.

### 4.2 The σ transform given value scale [-1,1]

- Q is already bounded to [-1,1] (`losses.py linspace(-1,1)`, `VALUE_BINS=65`, pure win/loss, `Q_UTILITY_WIDTH=2.0` tree.rs:49). No Q re-normalization needed.
- **Decision: σ(q) = (c_visit + max_visit_N) · c_scale_gumbel · q**, the Danihelka monotone form, with:
  - `c_visit ≈ 50` (start), `c_scale_gumbel` tuned so `σ(Q)` is **commensurate with the prior-logit spread at median-support-7 positions** (i.e. σ·Q ≈ same order as the gap between the top few prior logits there).
  - **Do NOT reuse KataGo's σ scale** — it assumes a different Q normalization.
  - Calibration guard: too-flat σ ⇒ target collapses to the prior (no improvement); too-sharp σ ⇒ over-sharpens to a noisy Q argmax (re-imports variance through Q). Calibrate offline (Stage 0) on a held-out shard batch before committing.

### 4.3 completedQ definition & the value-fallback choice

- Visited actions: `completedQ(a) = Q(a) = value_sum/visits` (tree.rs:191-197), already exported as `q_policy`.
- **Unvisited actions (the dangerous term): use a VISIT-WEIGHTED NODE-VALUE BLEND, not a bare child `eval_value` read.** Concretely the parent node's current mixed value estimate (visit-weighted mean of visited-child Q at the root) — NOT the child's raw value-head output. This denies the main_4 value-Q saturation loop a re-entry path through the target.
- Two-player zero-sum: the parent's current value estimate is the natural fallback (paper uses "the value function" for unvisited).

### 4.4 How exploration replaces Dirichlet + temperature

- **Stage 1 keeps Dirichlet + temperature** (search semantics unchanged; only the recorded *target* changes). This is deliberate: Stage 1 isolates the target change from any search-distribution change so the A/B attributes the KL-gap movement to #3 alone.
- **Stage 2 (if reached):** the per-root **Gumbel draw is the sole diversity source**. On Gumbel/SH moves, **disable** `root_dirichlet_*` (10.83/0.20), `temperature` (1.0/floor 0.15/halflife 45), AND `root_policy_temperature` (1.1/1.15). Leaving them on double-counts exploration. Also turn **OFF** the 96-child nucleus widening + `forced_playout_k` discovery machinery on Gumbel moves — SH's m-candidate set IS the discovery mechanism; running both double-budgets the 1024 visits.

### 4.5 Avoiding main_4 value-Q saturation (first-class guards)

main_4 died from `root_fpu_reduction=0 + new_child_fpu + lazy_widening + const c_puct=1.5` letting an overconfident value head dominate Q on balanced nodes (|Q| 0.17→0.79). Guards baked into every Gumbel stage:

1. **Fallback = visit-weighted node-value blend** (§4.3), never a bare child eval read.
2. **Min-visit floor:** a candidate contributes target mass only if `visits ≥ floor` (e.g. `gumbel_target_min_visits = 1` initially; raise if saturation appears). Below the floor, the action is **excluded from the softmax support** so the value head never writes the target on un-searched balanced cells.
   - **Implementation note (NOT a `cell_q_mask` drop-in).** `cell_q_mask` (samples.py:210, replay_expand.rs:600-609) is a **per-action presence mask for the `cell_q` Q-regression head**, not a mask on the main policy CE target. The main policy target is the **dense `policy (B,L)` tensor** (batching.py:129-135) and there is no existing per-action mask that zeroes entries of the main policy distribution. So the floor is implemented **Rust-side at target-construction time**: "which actions enter the `softmax(logit+σ(Q))`" is decided when the gumbel weights are built (the exported weights are already a normalized softmax over the surviving support, §5.1.1), and the renormalization happens there. The Python side just consumes the already-normalized `gumbel_policy (B,L)` array. We do NOT claim `cell_q_mask` machinery covers this; the support selection + renormalization is a new code path co-located with the gumbel-weight build.
3. **Run config keeps the saturation-breaking settings:** `root_fpu_reduction=0.2`, NO `lazy_widening + root_fpu=0` stack.
4. **Saturation kill-switch:** automate the main_4 fixed-balanced-opening turn-0 |Q| probe (40 fixed openings) as a per-epoch gate; abort/flag if signed-mean |Q| climbs monotonically.

### 4.6 Auxiliary heads stay decoupled

`soft_policy` (p^0.5 support-only, weight 1.0, separate `HexNodeConv` model.py:383-384), `moves_left` (0.2), `cell_q` (0.1) remain **independent** of the Gumbel main-policy target. `cell_q` already trains on exactly the `q_policy` data #3 needs, so the data plumbing partially exists; do **not** unify them (out of scope/risk). Re-check their relative weights only if the sharpened main target shifts trunk-gradient balance (main_4 showed an aux head at 74-76% of loss can starve the trunk). **The §8 trunk-starvation kill-criterion needs an instrument:** §7.4 logs a per-epoch per-head loss-share and trunk-gradient-norm breakdown so the "value/stvalue losses regress +0.28-0.52" trip is actually observable rather than inferred after the fact.

### 4.7 PCR interaction

Only ~33% of moves are Full (recorded, 1024-visit); ~67% are Fast (192) with `policy_valid=0` (policy heads masked), and Init (1). #3 naturally applies only to **Full** rows — exactly the high-variance rows where it matters. SH (#1/#2) also gates to Full only.

---

## 5. Staged rollout

> **Divergence-flag pattern (repo convention):** every new behavior is gated by a flag that resolves to `Divergences::production()` in production self-play and `Divergences::parity()` (byte-identical golden vectors) in the M5/M6 parity tests. New flags default OFF until a stage's go/no-go passes. Mirror the existing flags (`nucleus_f64`, `new_child_fpu`, `lazy_widening`, `clean_root_prior_cache`, `dirichlet_shaped`, `pruned_dynamic_cpuct`).

### Stage 0 — OFFLINE feasibility + instrumentation gate (GATES EVERYTHING)

**Goal:** before any production code, (a) estimate how much of the 0.672-nat KL gap is compressible by a Q-informed target, and (b) lock the export schema needed for the faithful online target.

**5.0.1 What is feasible offline (no new logging).**
`q_policy` (searched child Q) IS logged and plumbed to `cell_q` (`samples.py:266-275`, validated finite/[-1,1]). So we can compute a **degenerate completedQ-blend target** on existing main_5 shards:
- For each Full row, build `π_blend(a) = softmax( log(max(prior_w(a), eps)) + σ(Q(a)) )` over the **visited support only** (mask unvisited via the existing `cell_q_mask`), using the **logged softmaxed prior** as a stand-in for logits.
- **This is NOT the faithful target** (log(softmax_weight) ≠ logit up to a non-constant offset because `root_policy_temperature` 1.1/1.15 and remaining_priors padding break the inversion). It is a *variance-direction estimate only.*

**5.0.2 Offline metrics to compute** (new scratch script `analysis/_gumbel_offline_main5.py`, modeled on `_scratch_klgap_main5.py`):
- Per-row **target variance proxy:** bootstrap-resample the visit counts (multinomial with the observed N) and measure argmax-flip rate of `π_visit` vs the stability of `π_blend` under the same resampling. Expect `π_blend` flip-rate ≪ `π_visit` flip-rate on median-support-7 rows if #3 is real.
- **KL(π_blend ‖ prior)** vs **KL(π_visit ‖ prior)** distributional shift.
- **Top-1(π_blend) vs top-1(π_visit)** agreement with the eventual game outcome / with search argmax.
- σ **calibration sweep:** grid `c_scale_gumbel ∈ {0.1, 0.3, 1.0, 3.0}` × `c_visit ∈ {0, 50}`; pick the (flattest σ that still moves top-1 stability) operating point.

**5.0.3 Instrumentation spec (the Rust→Python export column the faithful target needs).**
Add a **raw-policy-logits export column** at decision time so the faithful target can later be computed offline AND audited.

> **CRITICAL prerequisite (§0.1):** the raw logits do **not** exist in Rust — `owned_root_from_evaluation` only receives softmaxed `priors`. The export column below therefore *also* depends on first getting logits across the Python→Rust boundary (§5.0.4). The two pieces are: (5.0.4) Python→Rust logit plumbing so Rust *has* the logits, then (5.0.3) Rust→Python export so the *training pipeline* sees them per row. If, for Stage 0 only, the goal is merely the offline audit, the logits can instead be exported **directly from Python** at `inference.py` decision time (where `logits` is already in hand at inference.py:466) and joined to the row by state hash — sidestepping the Rust round-trip. Pick the Python-direct export for Stage 0 (cheapest), and add §5.0.4's Rust-side logits only when Stage 1's online-in-Rust path or Stage 2's Gumbel sampler needs them.

Export column spec (Rust round-trip variant):
- **Rust:** in the per-decision payload build (**`build_search_result_payloads()` search.rs:1886-1996**, alongside the `root_prior_policy_*` exports at search.rs:1950-1960), export `root_prior_logits_bytes` = the pre-softmax network policy logits for the legal set, BEFORE `apply_root_policy_temperature()` (tree.rs:470) and BEFORE Dirichlet. (Requires the logits to be present on the node via §5.0.4.) Add the field to the `lib.rs` payload dict.
- **Python ingest:** `selfplay.py` (~line 162, next to `visit_policy_q_bytes`) reads the new bytes into `HexfieldSampleData` (new optional field `prior_logits: tuple[(action_id, float), ...]` in `samples.py:42`-region).
- **Shard schema:** `shards.py` adds a per-action `prior_logit` array **parallel to `pol_act`** (mirror `q_pol_q` everywhere it appears: write at shards.py:191, read at shards.py:268-270 with a legacy-absent guard like the existing `if "q_pol_q" in arrays`). Bump `SCHEMA_VERSION` (shards.py:29, currently `1`). Forward-compatible reader: old shards → field absent → fallback (offline/blend).
- **window.py (MUST update — else the column is written but never read through the window/expand path, exactly the q_pol_q reader bug fixed at shards.py:263-267):** add `prior_logit` to the `CSR_GROUPS` pol-offset group (window.py:83 `("pol_off", ("pol_act","pol_w","q_pol_q"), False)` → add `prior_logit`); add the field to the `PackedWindow` dataclass view (window.py:142-144 region, mirroring `q_pol_q` at window.py:144), the slice in `_view_row` (window.py:280-282 region), and the dtype map (window.py:314-316 region).
- **expand_backends.py (MUST update):** add `prior_logit` to the CSR-data column tuple (expand_backends.py:81-83 region, next to `q_pol_q`) so `_window_columns_as_bytes` (expand_backends.py:281) packs it for the Rust expand kernel.
- **replay_expand.rs:** project the new column. **Naming caveat:** the Rust struct field for the existing Q column is **`q_policy`** (replay_expand.rs:123, 601), while the Python shard/window/expand array name is **`q_pol_q`** — the same data has two names by layer. The new column is analogously `prior_logits` (Rust struct field) / `prior_logit` (Python shard array). Project it in BOTH `replay_expand.rs` AND the serial (non-Rust) expand path so both backends agree. This Rust change is **export/projection-only** (no scheduler touch, no pos/s impact).

**Files touched (Stage 0):** `analysis/_gumbel_offline_main5.py` (new); for the export column: `packages/hexfield/python/hexfield/inference.py` (Python-direct logit export — Stage-0 default), and for the Rust round-trip variant additionally `packages/hexfield/rust/src/search.rs`, `lib.rs`, `replay_expand.rs`, plus `selfplay.py`, `samples.py`, `shards.py`, **`window.py` (CSR_GROUPS + dataclass view + dtype map)**, **`expand_backends.py` (CSR-data column tuple)**.

**5.0.4 Python→Rust logit plumbing spec (required for the "online in Rust" path; required by Stage 2's Gumbel-root sampler).**
There is currently no channel that carries raw logits into the Rust tree (§0.1). To make `softmax(logit+σ(Q))` computable inside Rust (and to let the Gumbel-Top-k root sampler in Stage 2 draw over `logit(a)+g(a)`), add a parallel logits payload to the evaluator reply:
- **inference.py:** `_decode_group` already has `logits = out['policy'].float()` in hand at inference.py:466. Gather `logits[legal]` exactly as `priors[legal]` is gathered (inference.py:477-478) and serialize it as a new reply field `priors_logits_bytes` alongside `priors_bytes` (same fp32 positional layout; the doc header at inference.py:6-7 must be updated to mention the new field).
- **payload.rs:** in `parse_chunk_reply` (~payload.rs:162-181), parse `priors_logits_bytes` with the same length validation as `priors_bytes` (`require_exact_bytes`) into a new `Vec<(PackedCoord, f32)>`. Make it **optional** (absent → `None`) so parity replies / older evaluators that omit it still load.
- **cache.rs:** add `logits: Option<Vec<(PackedCoord, f32)>>` to `RustEvaluation` (cache.rs:18-28).
- **tree.rs:** carry `evaluation.logits` through `owned_root_from_evaluation` (tree.rs:1248-1317) onto the root node (a new optional field on `RustNode`, populated next to `eval_value`), so the target-build site and the Gumbel sampler can read pre-softmax logits aligned to the candidate set.

This is a **Stage-1 (online path) / Stage-2 prerequisite**, distinct from the Stage-0 Rust→Python export column, and is added to Stage 1's touched-files list (§5.1).

**Go/No-Go (Stage 0):**
- **GO to Stage 1** iff the offline blend shows a *materially* lower argmax-flip rate (target: **≥30% relative reduction** on median-support-7 rows, measured over **≥50k Full rows** with a **bootstrap 95% CI** on the relative-reduction estimate whose lower bound clears 30% — i.e. the gate is the CI lower bound, not the point estimate) AND a calibratable σ exists that doesn't collapse to the prior or over-sharpen. **AND** the free config levers (`pcr_full_proportion 0.33→0.5`, `c_puct 1.5→1.1`, `dirichlet 0.25→0.20`) have been run on main_5 ep115-120 and a residual KL gap (>~0.4 nat) survives.
- **NO-GO / STOP** if the config levers alone close most of the gap (then ship those; skip Gumbel), or if the compressible fraction is small (residual is `H(target)`).

### Stage 1 — completed-Q TARGET only, PUCT search UNCHANGED (low risk, on-bottleneck)

**Goal:** replace the recorded Full-move policy target with `softmax(logits + σ(completedQ))`, leaving all search selection (root Dirichlet, PUCT, temperature) untouched. A/B against the live visit-count target on identical self-play.

> **Where the target is computed — two routes (§0.1).** Raw logits are NOT in the Rust tree by default. Either:
> - **Route A (offline, lower-risk default):** export the raw logits (Stage-0 column, §5.0.3) and the searched Q per row, then compute `softmax(logit+σ(Q))` in **Python** in `samples.py`/`batching.py` at shard-expand time. No new Rust math; the Rust change is the export column only. This is the recommended Stage-1 path because the A/B is then a pure training-side flip with no self-play re-run.
> - **Route B (online in Rust):** plumb logits Python→Rust (§5.0.4) so the Rust tree holds pre-softmax logits, then compute the target in `build_search_result_payloads()`. Required only if the target must influence self-play (it does not in Stage 1) or to share code with Stage 2's Gumbel sampler. Costs the full §5.0.4 plumbing.
>
> Stage 1 ships **Route A**. §5.0.4 plumbing is built when Stage 2 begins.

**5.1.1 Rust changes (Route A — export only):**
- In **`build_search_result_payloads()` (search.rs:1886-1996)**, alongside the existing `root_prior_policy_*` exports (search.rs:1950-1960): export the raw logits column (§5.0.3) so Python can build the target. No softmax/σ math in Rust on Route A.
- (Route B only, deferred) compute `weight'(a) = softmax_a( logit(a) + σ(completedQ(a)) )` over the floored support, where `logit(a)` comes from the §5.0.4 logits field carried onto the root node, `completedQ(a)` = `edge.value()` for visited / visit-weighted node-value fallback (§4.3) for unvisited-but-floored, and export `gumbel_target_weights_bytes`. σ constants (`c_visit`, `c_scale_gumbel`) read from config.
- **Support floor (both routes):** actions below `gumbel_target_min_visits` are **excluded from the softmax support** at target-build time (Route A: in the Python build; Route B: in the Rust build). This is NOT the `cell_q_mask` head-presence mask (§4.5 guard 2); it is the choice of which actions enter the `softmax`, followed by renormalization over the surviving support.

**5.1.1b Python target build (Route A):**
- `samples.py expand_sample` / `batching.py`: from the per-row `prior_logit` array + the `q_policy` (searched Q) array, build `gumbel_policy (B,L)` = `softmax(logit + σ(Q))` over the floored visited support, renormalized. This is a **new dense-policy code path** parallel to the existing `policy (B,L)` build (batching.py:129-135); it is NOT covered by existing `cell_q_mask` machinery.

**5.1.2 Python changes:**
- `selfplay.py`: ingest `gumbel_target_weights_bytes` into a new `HexfieldSampleData` field (`gumbel_policy`).
- `samples.py` `expand_sample`: when present, build a second policy target array (parallel to `policy (L,)`).
- `batching.py` `collate_training`: emit `gumbel_policy (B,L)` tensor.
- `losses.py` `hexfield_loss`: a config switch `policy_target ∈ {"visit","gumbel"}` selects which array drives the main policy CE; default `"visit"`. (A/B = flip this offline on the SAME shards.)

**5.1.2b Config-section ownership (REQUIRED — strict unknown-key guard).** `config.py:_merge` (config.py:377-381) raises `ValueError` on **any** toml key not present in the target dataclass's fields. So every new knob must be added to the correct dataclass, and placed under the matching toml table, or the run fails to load. Assignment:
- **`SelfplayConfig` (config.py:16) + `[model.config.selfplay]`** — search/export/sampler knobs that must also reach Rust via `build_divergence_overrides` (config.py:407): `gumbel_target_enabled`, `gumbel_c_visit`, `gumbel_c_scale`, `gumbel_target_min_visits`, `export_root_prior_logits`, and (Stage 2) `gumbel_root_enabled`, `gumbel_m`, `gumbel_sequential_halving`, `gumbel_disable_dirichlet`, `gumbel_disable_widening`.
- **`TrainingSection` (config.py:97) + `[model.config.training]`** — the training-side target selector consumed in the loss path: **`policy_target`** (read by `losses.py hexfield_loss`, which sources from `TrainingSection`, NOT `SelfplayConfig`). Putting `policy_target` under `[model.config.selfplay]` (as an earlier draft's §9 sketch did) is a cross-section mismatch and will either ValueError or land in the wrong dataclass. It moves to `[model.config.training]`.
- Any selfplay knob that must reach Rust also needs a line in `build_divergence_overrides` (config.py:407+) and a matching parameter in `run_continuous`'s pyo3 signature (search.rs:763).

**5.1.3 Saturation guards active:** `root_fpu_reduction=0.2`, no `lazy_widening+root_fpu=0`, the turn-0 |Q| kill-switch probe automated per-epoch.

**Files touched (Stage 1):**
- *Route A (default):* `search.rs` (`build_search_result_payloads` raw-logit export), `lib.rs` (payload), `replay_expand.rs` + serial expand path (project `prior_logit`), `selfplay.py`, `samples.py`, `shards.py`, **`window.py`** (CSR_GROUPS + dataclass view + dtype), **`expand_backends.py`** (CSR-data column tuple), `batching.py` (build `gumbel_policy`), `losses.py` (`policy_target` switch), `config.py` (knobs split across `SelfplayConfig` / `TrainingSection` per §5.1.2b).
- *Route B (if/when online-in-Rust is built):* additionally `inference.py` (`priors_logits_bytes`), `payload.rs` (parse), `cache.rs` (`RustEvaluation.logits`), `tree.rs` (carry logits onto root + node-value fallback).

**New config knobs (Stage 1):**
```
# [model.config.selfplay]  (SelfplayConfig + build_divergence_overrides → Rust)
gumbel_target_enabled = true     # divergence flag, production-only, default false in parity
gumbel_c_visit = 50.0
gumbel_c_scale = <calibrated in Stage 0>
gumbel_target_min_visits = 1
export_root_prior_logits = true

# [model.config.training]  (TrainingSection; consumed by losses.py)
policy_target = "gumbel"         # {"visit","gumbel"}; A/B switch (training side)
```

**Go/No-Go (Stage 1):**
- **GO to Stage 2 (or DECLARE WIN and stop)** iff, on an A/B (same self-play, `policy_target` flipped): KL(target‖prior) gap drops measurably (target: residual gap < main_5's 0.672 by a resolvable margin), top-1 prior-vs-search agreement rises off 0.568, **AND** root_value saturation probe stays flat (no |Q| climb), **AND** strength vs `main2_ep45` anchor (raised eval power) is non-regressing.
- **NO-GO / ROLLBACK** (flip `policy_target="visit"`, flag off) if: saturation probe trips, OR KL gap unchanged (σ mis-calibrated or residual is entropy not variance), OR Elo regresses beyond eval SE in the wrong direction.

**If Stage 1 wins, #1/#2 may not be worth building** — #3 is the durable high-visit win and main_5 is not throughput-bound. Stage 2 is conditional on a *new* constraint (throughput) emerging.

### Stage 2 — Gumbel-root + Sequential Halving + deterministic non-root (the throughput play)

**Only if** Stage 1 wins AND we want to **drop self-play visits** (e.g. 1024→256) to recover throughput while preserving target quality via SH's policy-improvement guarantee.

**5.2.1 Root-only SH (the scheduler-friendly variant — see §6):** keep non-root selection as PUCT (do NOT adopt #2's `argmax[logit+σ(Q)]` at non-root initially — it re-opens saturation and demands the deepest refactor). Implement Gumbel-Top-k + Sequential Halving **at the root only**, as a per-root budget allocator over m candidates, with **playout-cap semantics** (fixed per-candidate budget, halving applied at decision time over the accumulated visits) rather than a strict global round barrier.

**5.2.2 Rust changes:**
- After RootInit backup (`backup_continuous_items` search.rs:1455-…), BEFORE leaf selection: sample `g(a)~Gumbel(0)` for all root edges (reuse `SEED_STREAM_ROOT_NOISE` stream 0, search.rs:43), select top-m by `logit(a)+g(a)` (requires logits on the root node, §5.0.4), store `slot.root_candidates`, `current_round=0`.
- Per-slot state expansion: `root_candidates: Vec<PackedCoord>`, `current_round: u32`, `round_budget: u32` (playout-cap, not strict *global* barrier).
- **Intra-slot barrier (REQUIRED for the SH guarantee).** Halving within a slot fires **only when ALL surviving candidates in that slot have reached the per-candidate round cap** — not when any single candidate fills. This enforces SH's equal-allocation-per-round *within* each root, which is what the policy-improvement argument needs. Async leaf generation can let candidates accumulate unequal visits between flushes; the intra-slot barrier prevents an early-arriving candidate from triggering a premature halving against under-visited rivals. There is still **no cross-slot barrier** — slot A may be in round 2 while slot B is in round 0 — so the global flush keeps batching across roots and the async scheduler is untouched. (This is the §6.2 option-2 design, now made precise: independent *across* slots, equal-allocation *within* a slot.)
- Halving: once a slot's survivors all reach the round cap, rank them by `g(a)+logit(a)+σ(completedQ(a))`, keep `ceil(survivors/2)`, advance round.
- The Gumbel-Top-k sampling-without-replacement equivalence (top-m of `logit+g` ≡ m draws w/o replacement from `softmax(logit)`) is verified in §7.2.
- Disable Dirichlet/temperature/root_policy_temperature on Gumbel moves (§4.4); disable nucleus widening + forced_playout on Gumbel moves.
- Final target = `softmax(logit + σ(completedQ))` over the m supports (Stage-1 machinery, now over the SH-concentrated visits).

**5.2.3 Files touched (Stage 2):** `search.rs` (run_continuous slot state, SH allocation, Gumbel sampling, intra-slot barrier), `tree.rs` (candidate bookkeeping + carry logits from §5.0.4), `lib.rs`, `inference.py` + `payload.rs` + `cache.rs` (§5.0.4 logit plumbing, if not already built in Stage 1), `selfplay.py`, `config.py`.

> **Flush is a DESIGN CONSTRAINT, not a maybe (§6.2).** The entire justification for root-only independent-per-slot halving is to **avoid touching** the global-flush / per-round scheduler. Stage 2 therefore commits to **"no per-round flush change"**: `continuous_flush_decision` (search.rs:235-247) and `flush_target` semantics are unmodified, and a regression assert (§7.5) confirms per-flush batch behavior is unchanged at the same visit count. The earlier "`inference.py` only if flush tuning per-round is needed" conditional is **removed** — if SH is found to require per-round flush tuning, that is a Stage-2 **NO-GO** (it would mean we failed to keep the scheduler intact), handled by the §8 throughput kill-criterion, not by quietly editing the flush path.

**New config knobs (Stage 2):**
```
gumbel_root_enabled = true       # divergence flag, default false in parity
gumbel_m = 16
gumbel_sequential_halving = true
gumbel_disable_dirichlet = true  # on Gumbel moves
gumbel_disable_widening = true   # on Gumbel moves
```

**Go/No-Go (Stage 2):** GO iff pos/s at reduced visits (e.g. 256) ≥ current ~9 pos/s AND target quality (KL gap, top-1) ≥ Stage-1-at-1024. NO-GO if SH bookkeeping regresses throughput or the independent-halving concession degrades the policy-improvement guarantee (measure: does m=16 SH at 256 visits match 1024 PUCT target quality?).

---

## 6. Performance plan: scheduler vs Sequential-Halving tension

### 6.1 The tension (root cause)

The continuous scheduler is **per-root ASYNCHRONOUS** with a **GLOBAL queue flush** (`continuous_flush_decision` search.rs:235-247, flush when `queue_len ≥ flush_target`). **`flush_target` is the CAP, set to `1024` in main_5 (`hexfield_main_5.toml:118`); `≈213` is the *observed mean effective batch*, not the cap** — flushes also fire on no-further-progress, so the realized batch averages ~213 even though the trigger is 1024. `select_continuous_pass` fans slots across cores via **rayon `par_iter_mut`** (`select_continuous_pass` search.rs:1412, `.par_iter_mut()` search.rs:1423) with **no inter-slot barrier**. Completion is per-slot independent (`continuous_completion_ready`: `completed_visits ≥ target && in_flight == 0`, search.rs:258-259). Throughput is **GPU-FLOP-bound** on deep games; the async overlap exists to keep the GPU saturated by amortizing each forward over the ~213-mean states/flush across ~512 roots.

Canonical SH wants **per-root synchronous rounds** (allocate `budget_r`, eval ALL survivors, halve, advance). A global round barrier serializes the slowest slot per round and **shrinks early-round flushes** (few candidates per root) → **GPU starvation** on a FLOP-bound forward.

### 6.2 Resolution options (ranked)

1. **Stage 1 (chosen first): NO scheduler change.** #3 target is computed at decision time; the existing async overlap, rayon slot-parallelism, and global flush batching are preserved **verbatim**. **~0% pos/s impact.** This captures the load-bearing win.
2. **Stage 2 root-only SH with playout-cap, INTRA-slot barrier, independent ACROSS slots (chosen for #1/#2).** No *global* barrier — each slot advances rounds on its own schedule, so slots stay independent and the global flush still batches across roots. But **within** a slot, halving fires only when all survivors reach the round cap (§5.2.2 intra-slot barrier), preserving SH's equal-allocation-per-round — without an intra-slot barrier, async leaf generation lets candidates accumulate unequal visits and erodes the guarantee. Net: equal-allocation within a root, full async across roots. Expected impact: small (single-digit %) pos/s change at equal visits, plus the late-round batch-occupancy effect quantified in §6.3.
3. **(Rejected) Canonical global-barrier SH.** Multi-week rewrite of `run_continuous` + `select_continuous_pass` + `backup_continuous_items` + completion, re-proving the "stale prefetch discarded" guarantee (search.rs:979-990) under round barriers, with a real pos/s regression. Not worth it for a non-binding constraint.

### 6.3 Throughput math & the visit-drop lever

- Current: ~9 pos/s, ~40min epochs, 1024 full / 192 fast.
- **Stage 1 alone does NOT let visits drop** — it changes the *target*, not search depth; dropping visits would reduce the Q-estimate quality #3 consumes. Stage 1 keeps 1024.
- **Stage 1 host cost is NOT exactly zero.** Route A adds a per-Full-root `softmax(logit+σ(Q))` over the visited support on the **host per-decision path** — `O(legal)` per Full decision. Since post-FlexAttention the bottleneck has shifted toward host per-decision work (MEMORY: hexfield-flexattention-serve), this is the relevant cost, not GPU FLOPs. Expected negligible (a few hundred floats per Full row, ~33% of rows) but **measured in §7.5**, not assumed.
- **Stage 2 is the visit-drop play — but throughput is NOT visit-linear.** The naive "256 visits ⇒ 4× pos/s" assumes throughput scales linearly with visits; it does **not**, because the forward is FLOP-bound and SH's **late rounds shrink the per-flush batch** (m candidates → … → 1, so a slot in its final round contributes far fewer in-flight states). Fewer in-flight states per flush ⇒ smaller effective batch ⇒ worse GPU amortization — the exact starvation mechanism §6.1 warns about, now arising from SH's candidate decay rather than a barrier. **Therefore: (a)** include a **batch-occupancy estimate** (expected mean in-flight states/flush under m=16 SH at 256 visits vs the current ~213) in the Stage-2 design review, and **(b)** the go/no-go is a **direct pos/s measurement at 256-visit SH**, not an extrapolation from the visit count. If yes (pos/s ≥ current ~9 AND target quality ≥ Stage-1-at-1024), real throughput win; if no, Stage 2 is not worth its cost.
- **Regression guard:** any Stage-2 build that drops pos/s below current at the SAME visit count is rejected (the flush-batching amortization must be preserved; per §5.2.3 the flush path itself is unmodified).

---

## 7. Verification passes

### 7.1 Offline metric checks (Stage 0)
- `analysis/_gumbel_offline_main5.py`: argmax-flip rate `π_blend` vs `π_visit` under multinomial resampling; KL(·‖prior) shift; top-1 agreement; σ calibration sweep. Gate: ≥30% relative flip-rate reduction on median-support-7 rows, over **≥50k Full rows** with a **bootstrap 95% CI** whose lower bound clears 30% (§7.4).
- Re-run `_scratch_klgap_main5.py` on ep115-120 AFTER the free config levers to confirm a residual gap survives.

### 7.2 Rust unit tests (Stage 1 & 2) — in `packages/hexfield/rust/src` test modules
- **completedQ math:** for a hand-built tree, assert `completedQ(visited)=value_sum/visits` and `completedQ(unvisited)=visit-weighted node-value` (NOT child eval); assert min-visit-floored actions are masked.
- **σ transform:** assert `σ(q)=(c_visit+max_N)·c_scale·q` monotone in q on [-1,1]; boundary q=±1.
- **Gumbel-Top-k = sampling-without-replacement (Stage 2) — with a concrete gate:** over **≥100 random logit vectors** (varied length/entropy), draw the top-m of `logit+g` for **≥10,000 independent Gumbel seeds** each, and compare the empirical **first-pick** selection frequencies against the analytic `softmax(logit)` distribution (the m=1 marginal of Gumbel-max sampling-without-replacement). **Gate: per-vector chi-squared goodness-of-fit p > 0.01**, and across the ≥100 vectors the **fraction failing at p<0.01 must be ≤ 5%** (the expected false-positive rate at α=0.01 is 1%; ≤5% tolerates seed noise). For the full top-m ordering, additionally assert the second-pick conditional frequencies match `softmax` renormalized over the remaining support (same p>0.01 / ≤5%-fail criterion). A bare "KS/χ² test" with no threshold is a no-op gate; this fixes that.
- **SH allocation correctness (Stage 2):** assert per-round budget = `budget_total/2^r` per surviving candidate; survivors = `ceil(prev/2)`; final round 1 survivor; total visits ≤ budget. Assert the **intra-slot barrier**: a halving event in a slot occurs only after every survivor reached the round cap (construct a slot where one candidate fills early and verify no premature halving).
- **Independent-vs-canonical SH regret simulation (Stage 2 — PRE-BUILD gate, NOT deferred to the full build):** on **synthetic bandit roots with known per-arm Q** (a grid of gap structures: clear-best, near-tie, multi-way), simulate three allocators at budget 256, m=16 — **(i) independent-per-slot halving (our concession), (ii) canonical synchronous-round SH, (iii) uniform allocation** — over many noise seeds. **Gate: the independent variant's top-1-recovery rate and simple-regret must stay within a stated tolerance of canonical** (target: top-1-recovery within **3 percentage points (absolute)** and mean simple-regret within **15% (relative)** of canonical, across all gap structures), and must strictly beat uniform. This converts the §8 Stage-2 "independent-halving concession breaks the improvement guarantee" risk from a build-then-discover failure into a **pre-build gate** runnable as a standalone simulation (`analysis/_gumbel_sh_sim.py`) before any self-play build.
- **Target softmax:** assert exported `gumbel_target_weights` sum to 1 over the support and equal `softmax(logit+σ(Q))` (Route A: assert the Python build matches a reference; Route B: assert the Rust export matches).

### 7.3 Parity / divergence-flag tests
- M5/M6 golden-vector `Divergences::parity()` must remain **byte-identical** with all new flags default-OFF. Add a parity case that explicitly asserts `gumbel_target_enabled=false`, `gumbel_root_enabled=false` in parity mode.
- A production-mode test asserting the new export column is present and well-formed (logits finite, target weights normalized).
- **Mixed old/new shard transition check (operational gap — main_6 warm-starts from main_5 ep120 and reads a MIX of pre-bump and post-bump shards):** load a batch containing BOTH old shards (no `prior_logit` array, `SCHEMA_VERSION` old) and new shards (with `prior_logit`, bumped version) through `collate_training`/the window/expand path and assert: (a) the absent-field fallback executes without crashing (mirrors the `if "q_pol_q" in arrays` guard at shards.py:268-270, and the `CSR_GROUPS`/dtype handling in window.py for a column that is sometimes absent); (b) old rows fall back to `policy_target="visit"` (no gumbel target available) while new rows can use `"gumbel"`; (c) the **bumped `SCHEMA_VERSION` is accepted by the live supervisor/dashboard shard reader** (confirm the reader's version guard, shards.py:223-224, is updated to accept the new version rather than raising "unsupported schema"). This must pass before main_6 launches, since the persistent replay window spans the version boundary during the transition epochs.

### 7.4 Self-play A/B protocol (main_6 vs main_5 anchor)
- **Setup:** identical self-play config; flip `policy_target` (Stage 1) on the SAME shards so the target change is the only variable.
- **Primary (resolvable) metrics:** KL(target‖prior) gap (expect drop); top-1 prior-vs-search agreement (expect rise off 0.568); target argmax-flip stability.
- **Saturation guard:** `root_value_mean` per epoch ∈ main_5's healthy band; the 40-fixed-opening turn-0 |Q| probe flat (NOT climbing 0.17→0.79).
- **Per-head loss-share / trunk-gradient instrument (feeds the §8 trunk-starvation kill-criterion):** log, **per epoch**, each head's share of total loss (main policy, soft_policy, moves_left, cell_q, value, stvalue) AND the per-head gradient L2-norm contribution at the trunk. Without this, the §8 "value/stvalue losses regress +0.28-0.52" and "aux head at 74-76% of loss starves the trunk" criteria have no instrument feeding them. Trip alert if any aux head's loss-share rises >~70% or value/stvalue loss climbs beyond the §8 band over 2 consecutive epochs.
- **Flip-rate gate row count:** the Stage-0/Stage-1 argmax-flip-rate metric (§7.1) is computed over **≥50k Full rows** with a **bootstrap 95% CI**; the gate is the CI lower bound clearing the 30% relative-reduction threshold (not the point estimate).
- **Secondary (directional) metrics:** Elo vs `main2_ep45` BT anchor with **raised eval power (>20 pairs/edge)** — directional only (point Elo SE ~48-90); game length (expect stable/rising, NOT collapsing toward 40); loss–Elo coupling sign (rising Elo ↔ falling loss).

### 7.5 CUDA / throughput regression checks
- pos/s before/after on a representative deep-game bench (the existing throughput bench from MEMORY hexfield-throughput-gpu-bound). Stage 1: assert within noise of current ~9 pos/s, **and explicitly measure the new per-Full-root `O(legal)` host softmax cost** (§6.3) so the "~0% impact" claim is verified, not assumed. Stage 2: assert ≥ current at same visits; **directly measure 256-visit SH pos/s** (NOT extrapolated from the visit count, §6.3) for the visit-drop decision, and report the **mean in-flight states/flush (batch occupancy)** under m=16 SH vs the current ~213 to expose late-round starvation.
- **Stage-2 flush-invariance regression assert (§5.2.3 design constraint):** at the SAME visit count, assert `continuous_flush_decision` behavior and the per-flush batch-size distribution are statistically unchanged vs the pre-Stage-2 build (the flush path must be untouched; any required per-round flush tuning is a Stage-2 NO-GO, §8).
- Confirm serve-flex recompile cap (512) is not tripped by any new shape (MEMORY hexfield-flex-recompile-eager-fallback).

---

## 8. Risks, fallbacks, kill-criteria per stage

| Stage | Risk | Fallback | Kill-criterion |
|-------|------|----------|----------------|
| 0 | Offline blend is unfaithful (logit gap) and mis-estimates the win | Treat as direction-only; rely on the export column for the faithful audit | Compressible fraction < ~15% of the 0.67 gap → STOP (residual is entropy); ship config levers only |
| 0 | Free config levers already close the gap | Ship `pcr_full 0.5`, `c_puct 1.1`, `dirichlet 0.20`; skip Gumbel | Residual KL gap < ~0.4 nat → no Gumbel |
| 1 | completedQ fallback re-imports value-Q saturation | Node-value blend + min-visit floor; if still climbing, mask ALL unvisited (visited-support-only target) | **Two-tier trip:** (a) **within-epoch absolute band** — if turn-0 |Q| signed-mean exceeds an absolute threshold (e.g. >0.4, well below main_4's 0.79) at ANY epoch, trip immediately and **quarantine** the shards written that epoch (do not let them enter the persistent replay window); (b) the slower 3-epoch-monotonic-climb trip → ROLLBACK (`policy_target="visit"`, flag off). The within-epoch band closes the gap where ~3 epochs of corrupted targets could otherwise enter the buffer before the monotonic trip fires. |
| 1 | σ mis-calibrated (collapse-to-prior or over-sharpen) | Re-sweep `c_scale` offline; ship the calibrated value | KL gap unchanged AND top-1 flat after σ sweep → declare #3 ineffective, stop |
| 1 | Sharper main target starves trunk (main_4 aux-head lesson) | Re-tune aux head weights (soft 1.0, ml 0.2, cell_q 0.1) | value/stvalue losses regress +0.28-0.52 (target-corruption signature), observed via the **§7.4 per-head loss-share / trunk-gradient log** → rollback |
| 2 | SH bookkeeping regresses pos/s | Root-only playout-cap, intra-slot barrier, independent across slots (no global barrier) | pos/s < current at same visits (measured directly, §7.5), OR SH requires per-round flush tuning (§5.2.3) → reject build |
| 2 | Independent-halving concession breaks the improvement guarantee | Fall back to canonical barrier ONLY if measurably better AND affordable | **PRE-BUILD gate (§7.2 `_gumbel_sh_sim.py`):** independent-variant top-1-recovery >3pp worse OR simple-regret >15% worse than canonical on synthetic bandit roots → fix the barrier design BEFORE building. **Post-build:** m=16 SH @256 visits < 1024-PUCT target quality → keep 1024, drop SH |
| 2 | Non-root #2 (`argmax[logit+σQ]`) re-opens saturation | Keep PUCT at non-root (don't adopt #2) | any saturation re-entry → revert non-root to PUCT |
| all | Eval can't resolve the Elo delta | Gate on resolvable KL/top-1 metrics, not raw Elo | n/a (designed around) |

---

## 9. Concrete config sketch: configs/hexfield_main_6.toml

A fresh run copied from `hexfield_main_5.toml`, with the Gumbel knobs default-OFF (Stage 0/1 ship behind them). Warm-start from a strong main_5 checkpoint (post-config-lever ep, e.g. ep120) so #3 is measured against a healthy, saturation-free baseline — NOT from a fresh prefit (avoids confounding the target change with cold-start dynamics).

```toml
# hexfield production run 6 (GUMBEL TRIAL). FRESH run, copied from hexfield_main_5.toml.
# Stage 1 = completed-Q TARGET only (PUCT search unchanged); Stage 2 = Gumbel-root+SH (deferred).
# All Gumbel flags resolve to Divergences::production() in self-play and parity() (byte-identical)
# in the M5/M6 golden-vector tests. Default OFF until each stage's go/no-go passes.

[run]
name = "hexfield_main_6"
# output_dir -> .../runs/hexfield_main_6

[checkpoint]
# Warm-start from a HEALTHY, saturation-free main_5 checkpoint AFTER the free config
# levers (do NOT cold-prefit; isolate the target change from cold-start dynamics).
initialize_from = ".../runs/hexfield_main_5/checkpoints/ep120.pt"   # tolerant/non-strict load
warmup_steps = 0

[model]
name = "hexfield"
module = "hexfield.plugin"

[model.config]
device = "cuda"

[model.config.selfplay]
# --- Free config levers FIRST (orthogonal to Gumbel; sequence + measure before #3) ---
search_visits = 1024
pcr_full_proportion = 0.50          # 0.33 -> 0.50 (more low-noise Full rows; PRIMARY config lever)
pcr_fast_visits = 192
c_puct = 1.1                        # 1.5 -> 1.1 (concentrate visits vs flat prior)
active_games = 96
virtual_batch_size = 4
flush_target = 1024
active_root_limit = 192
root_dirichlet_total_alpha = 10.83
root_dirichlet_noise_fraction = 0.20  # 0.25 -> 0.20 (exploration not the bottleneck)
root_dirichlet_shaped = true
root_policy_temperature = 1.1
root_policy_temperature_early = 1.15
root_fpu_reduction = 0.2            # GUARD: never 0.0 (main_4 saturation breaker)
fpu_reduction = 0.2
lazy_widening = false               # GUARD: never stack lazy_widening + root_fpu=0
new_child_fpu = true
nucleus_f64 = true
clean_root_prior_cache = true
pruned_dynamic_cpuct = true
forced_playout_k = 1.0
temperature = 1.0
temperature_floor = 0.15
temperature_halflife_plies = 45
max_game_plies = 256

# --- STAGE 0/1: completed-Q TARGET (divergence-flagged, default OFF) ---
# NOTE (§5.1.2b): these are SELFPLAY-side knobs -> add to SelfplayConfig (config.py:16)
# AND to build_divergence_overrides (config.py:407) so they reach Rust. The strict
# unknown-key guard (config.py:377-381) rejects the toml otherwise. `policy_target`
# is NOT here — it is a TRAINING-side switch (see [model.config.training] below).
gumbel_target_enabled = false       # Stage 1: flip true after Stage-0 GO
gumbel_c_visit = 50.0
gumbel_c_scale = 0.30               # PLACEHOLDER: set from Stage-0 σ calibration sweep
gumbel_target_min_visits = 1        # value head never writes target below this floor
export_root_prior_logits = true     # Stage-0 instrumentation: raw pre-softmax/pre-temp logits

# --- STAGE 2: Gumbel-root + Sequential Halving (deferred; default OFF) ---
# (also SelfplayConfig + build_divergence_overrides)
gumbel_root_enabled = false
gumbel_m = 16
gumbel_sequential_halving = false
gumbel_disable_dirichlet = true     # on Gumbel moves only
gumbel_disable_widening = true      # on Gumbel moves only

[model.config.training]
# policy_target lives HERE (TrainingSection, config.py:97), consumed by losses.py
# hexfield_loss — NOT in [model.config.selfplay] (§5.1.2b cross-section fix).
policy_target = "visit"            # {"visit","gumbel"}; A/B switch (training side)
soft_policy_weight = 1.0           # keep decoupled from the Gumbel main target
# moves_left_weight 0.2, cell_q (Q_HEAD_WEIGHT) 0.1 unchanged
train_samples_per_epoch = 48000    # do NOT raise (adds reuse, no signal)
expand_backend = "rust"
```

**Launch discipline:** bring up main_6 via a systemd drop-in from the worktree branch (mirror the main_2/main_5 pattern in MEMORY), NOT via `wsl.exe` background tasks. Keep the supervisor/dashboard on systemctl.

---

## Appendix A — confirmed-against-source facts (re-verified 2026-06-30)

- `q_policy` (searched child Q) is logged: `selfplay.py:162` reads `visit_policy_q_bytes`, `:190` builds `q_policy`; `samples.py:42` field; `:268-275` projects to `cell_q` with finite/[-1,1] validation. **(faithful target's Q side is ready.)**
- Raw policy **logits are NOT logged AND do not exist in Rust** — `selfplay.py:165-168` reads only `root_prior_policy_action_ids_bytes` + `root_prior_policy_weights_bytes` (softmaxed); `samples.py:46` `policy_surprise` is computed from those softmaxed weights (`_policy_surprise_kl`, samples.py:68-87). On the Rust side, logits are softmaxed in Python (`inference.py:466` → `:477`); only `priors_bytes` cross (`payload.rs:175-181`); `RustEvaluation.priors` are **probabilities** (`cache.rs:18-24`); `owned_root_from_evaluation` builds candidates from those probabilities (tree.rs:1258-1272). **(offline faithful target blocked → Stage-0 Rust→Python export column §5.0.3; online-in-Rust blocked → Python→Rust logit plumbing §5.0.4.)**
- **Function/line attributions (verified):** export site = `build_search_result_payloads()` search.rs:1886-1996 (sets `visit_policy_q_bytes`/`root_prior_policy_*` at search.rs:1950-1960); weight math = `visit_policy()` search.rs:2330, `pruned_visit_policy()` search.rs:2370. `apply_root_policy_temperature` tree.rs:470; `owned_root_from_evaluation` tree.rs:1248; `apply_root_dirichlet_noise` tree.rs:582; `select_or_materialize_edge` tree.rs:831. Scheduler: `continuous_flush_decision` search.rs:235-247; `continuous_completion_ready` search.rs:258-259; `select_continuous_pass` search.rs:1412 (`.par_iter_mut()` :1423); `backup_continuous_items` search.rs:1455; stale-prefetch discard search.rs:979-990; `SEED_STREAM_ROOT_NOISE` search.rs:43; `run_continuous` pyo3 signature search.rs:763. Shard column `q_pol_q`: shards.py:191 (write) / :268-270 (read, legacy-guarded), window.py:83 (CSR_GROUPS) / :144 (view) / :282 (slice) / :316 (dtype), expand_backends.py:81-83 (column tuple); Rust struct field name is `q_policy` (replay_expand.rs:123, :601). `SCHEMA_VERSION=1` shards.py:29; version guard shards.py:223-224. Config: `_merge` unknown-key `ValueError` config.py:377-381; `SelfplayConfig` config.py:16; `TrainingSection` config.py:97; `build_divergence_overrides` config.py:407. `cell_q_mask` is a per-action `cell_q`-head presence mask (samples.py:210, replay_expand.rs:600-609), NOT a main-policy-target mask; the main policy target is the dense `policy (B,L)` tensor (batching.py:129-135).
- main_5 live config: `search_visits=1024` (raised 2026-06-29), `pcr_full_proportion=0.33`, `c_puct=1.5`, `root_dirichlet 10.83/0.20`, `root_fpu_reduction=0.2`, `lazy_widening=false`, `soft_policy_weight=1.0`, `flush_target=1024` (CAP; ~213 is the observed mean batch) (`hexfield_main_5.toml:104-120` + header).
- main_5 health: no value-Q saturation, length rising, capacity unbound — bottleneck is target variance (`KL=0.672`, top-1 `0.568`).
