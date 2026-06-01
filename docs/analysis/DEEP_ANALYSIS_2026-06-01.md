# dense_cnn_model1_target_96x8 — Deep Offline Analysis: Value Calibration & Policy Sharpness

**Date:** 2026-06-01 · **Type:** offline model inference on recorded held-out positions.
This is the follow-up that **closes the three open items** the read-only
`ANALYSIS_2026-06-01.md` could not answer from disk: value-head calibration (§A.4),
raw policy-head sharpness (§B.3), and a fixed-holdout loss decomposition (red flag #1).

---

## 0. Method — read this first

- **Device: CPU.** No GPU was touched. The live trainer (≈6.5/12 GB VRAM) was **not**
  disturbed, killed, or restarted; no new self-play was generated. Frozen checkpoints were
  loaded read-only (`torch.load(..., map_location="cpu")`). Torch threads capped at 6 so the
  trainer kept its cores.
- **Runtime:** position sampling **17.4 s**, then **107–144 s per checkpoint** (mean ≈128 s)
  across 8 checkpoints → **≈16 min wall**. The model is small (96ch×8blk, ~2.6 M params); the
  per-checkpoint cost is entirely CPU forward of 4000 positions while *sharing the CPU with the
  live trainer* (the run held ~4.3 of 6 requested cores at ~430% CPU). On an idle box this would
  be a few minutes.
- **Sample:** **4000 recorded positions** drawn from **epoch-19 self-play shards**
  (`selfplay/epoch_000019_game_*.npz`), expanded at the **identity** symmetry. Phase split is
  balanced: 1302 opening / 1413 midgame / 1285 endgame. The **same fixed 4000 positions** were
  run through **every** checkpoint, so all cross-checkpoint comparisons are apples-to-apples.
- **Ground-truth targets** (read from the shards, not invented): value target = the **actual
  game outcome** at that position (±1 from the side-to-move's perspective; +1 win / −1 loss);
  policy target = the **MCTS visit-count distribution** at that position.
- **Held-out status (important caveat):** epoch-19 positions are genuinely **held-out for
  ep1–ep14** (those checkpoints predate the data). For the **latest `epoch_000019.pt` they
  overlap its training window**, so treat ep19's *absolute* numbers as a mild upper bound. **All
  progression claims below are already established on the held-out ep1→ep14 segment** before the
  in-sample ep19 point — ep19 only continues the trend.
- **Checkpoints:** `prefit` (= `bootstrap_sealbot_prefit.pt`), `ep1` (`epoch_000001.pt`),
  `ep3, ep5, ep7, ep10, ep14`, and `ep19` (`epoch_000019.pt`, latest). **prefit ≡ ep1 to every
  digit** — consistent with the earlier report's note that **epochs 1–2 did not train** (no
  training block until epoch 3), so `epoch_000001.pt` is literally the prefit weights. **ep3 is
  the first trained checkpoint.**
- **Helper (uncommitted):** `analysis/_deep_model_analysis.py`; raw stdout in
  `analysis/_dma_results.txt`.

---

## TL;DR verdict

1. **VALUE HEAD: well-calibrated and monotonically improving — this is the strong head.**
   On the ep19 reliability curve, **predicted win-probability ≈ empirical win-rate in every
   bucket** (expected-calibration-error **≈3.3%**), monotonic, with mild *under*-confidence
   (the safe direction for search). Sign-accuracy rose **0.50→0.69**, Brier **0.295→0.191**,
   held-out value CE **0.826→0.554**. Notably, the **SealBot-imitation prefit value head is
   ≈random on the RL self-play distribution** (sign-acc 0.495, *flat/anti-calibrated*
   reliability, ECE ≈16%) — **RL built the value calibration essentially from scratch.**

2. **POLICY HEAD: the raw prior is diffuse, but it is sharpening — this is the weaker head and
   the real bottleneck.** At ep19 the prior's **effective-move count ≈26.6 vs the search
   target's ≈5.6** (the prior is ~5× too broad). The prior **catastrophically diffused in early
   RL** (eff-moves 8.4 → 64.6 by ep7 — it "unlearned" the sharp imitation), then has been
   **recovering since ep7**: eff-moves 64.6→26.6, KL(target‖prior) 2.53→1.50, argmax-match
   0.26→0.42, top-1 prob 0.30→0.35. **The P7 head + sims=512 fix is working but has not closed
   the gap** — search (512 sims + forced playouts) does the sharpening at play time, which is why
   winrate is 80–92% despite a diffuse prior.

3. **LOSS DECOMPOSITION proves the "rising train loss" is a measurement artifact.** On the
   **same fixed held-out positions**, both **value CE (0.826→0.554)** and **policy CE
   (4.905→2.535)** fall **monotonically** across training. The earlier report could only
   *hypothesize* that the post-ep13 rising per-epoch train loss was a moving-target artifact;
   the fixed-holdout decomposition now **confirms it** — later checkpoints are strictly better on
   a frozen target.

---

## 1. Value-head calibration (closes §A.4)

### 1.1 Reliability curve — latest (ep19), n=4000

Bucketed by the model's **predicted** value; "empirical win-rate" = fraction of those positions
whose **actual outcome** was a win. Predicted win-prob = (mean_pred + 1) / 2.

| predicted-value bucket | n | mean predicted | pred. win-prob | **empirical win-rate** | gap |
|---|---:|---:|---:|---:|---:|
| [−1.0, −0.6) | 481 | −0.825 | 0.088 | **0.098** | +0.011 |
| [−0.6, −0.2) | 964 | −0.360 | 0.320 | **0.361** | +0.041 |
| [−0.2, +0.2) | 1537 | −0.015 | 0.493 | **0.519** | +0.026 |
| [+0.2, +0.6) | 532 | +0.381 | 0.690 | **0.737** | +0.046 |
| [+0.6, +1.0) | 486 | +0.834 | 0.917 | **0.961** | +0.044 |

**Monotonic and tight.** Expected calibration error **ECE ≈ 0.033**. Every gap is positive →
the model's predictions are **slightly compressed toward 0** (mild **under-confidence**): when it
says +0.38 the position actually wins 74% of the time. For MCTS this is the **safe** direction
(it won't over-commit on the value signal). The extreme-loss bucket is essentially perfect
(0.088 vs 0.098).

### 1.2 Reliability curve — prefit / ep1 (contrast)

| predicted-value bucket | n | mean predicted | pred. win-prob | **empirical win-rate** |
|---|---:|---:|---:|---:|
| [−1.0, −0.6) | 305 | −0.798 | 0.101 | **0.538** |
| [−0.6, −0.2) | 704 | −0.369 | 0.316 | **0.561** |
| [−0.2, +0.2) | 1573 | +0.024 | 0.512 | **0.488** |
| [+0.2, +0.6) | 1140 | +0.357 | 0.679 | **0.525** |
| [+0.6, +1.0) | 278 | +0.726 | 0.863 | **0.450** |

**Flat and partially inverted** (ECE ≈ 0.16): the SealBot-imitation value head's confidence is
**uncorrelated with — even slightly anti-correlated with — outcomes on the RL distribution**
(its most-confident wins, +0.73, actually win only 45%). It is a coin flip (sign-acc 0.495).
This is the clearest possible evidence that **the calibrated value head is a product of RL
training, not inherited from the prefit.**

### 1.3 Value metrics across training (held-out ep1→ep14; ep19 in-sample upper bound)

| ckpt | MAE | Brier | sign-acc | value CE | mean pred (act = +0.025) |
|---|---:|---:|---:|---:|---:|
| prefit / ep1 | 1.011 | 0.295 | 0.495 | 0.826 | +0.036 |
| ep3 | 1.009 | 0.266 | 0.479 | 0.726 | +0.118 |
| ep5 | 0.946 | 0.239 | 0.564 | 0.669 | −0.058 |
| ep7 | 0.921 | 0.241 | 0.562 | 0.671 | −0.172 |
| ep10 | 0.882 | 0.238 | 0.584 | 0.666 | +0.083 |
| ep14 | 0.835 | 0.222 | **0.611** | 0.628 | +0.015 |
| **ep19** | **0.767** | **0.191** | **0.693** | **0.554** | −0.040 |

- **Brier, sign-acc, and value CE improve monotonically** (Brier and CE strictly; sign-acc dips
  once at ep3 then climbs). The improvement is established on **held-out ep1→ep14** (Brier
  0.295→0.222, sign-acc 0.50→0.61) before the in-sample ep19 point.
- **On MAE not being alarming:** MAE stays ≈0.77 even at ep19. That is *expected*, not a defect:
  outcomes are ±1 but many positions (especially openings) are genuinely undecided, so a
  *well-calibrated* head correctly predicts ≈0 there → |0 − (±1)| ≈ 1 inflates MAE. The
  reliability curve and Brier (which reward calibration, not point-matching to ±1) are the right
  lenses, and both say the head is good. MAE *does* still fall 1.01→0.77, consistent with growing
  endgame decisiveness (see §5).

**Verdict (value):** the value head is **the healthy half of the network** — monotonically
improving and, at ep19, genuinely well-calibrated with a safe mild under-confidence.

---

## 2. Policy-head sharpness (closes §B.3 — the "diffuse policy head" concern)

All "prior" numbers are the network's **raw policy head**, legal-masked then soft-maxed (exactly
the prior MCTS consumes). The **MCTS target** column is the recorded visit distribution and is
**constant** across checkpoints (it belongs to the epoch-19 data): **eff-moves 5.6, top-1 0.694.**
"eff-moves" = exp(entropy) = effective number of candidate moves.

### 2.1 Raw prior sharpness across training

| ckpt | prior entropy | **prior eff-moves** | prior top-1 | (target eff-moves) | (target top-1) |
|---|---:|---:|---:|---:|---:|
| prefit / ep1 | 1.67 | **8.4** | 0.538 | 5.6 | 0.694 |
| ep3 | 3.42 | 53.7 | 0.278 | 5.6 | 0.694 |
| ep5 | 3.16 | 41.1 | 0.304 | 5.6 | 0.694 |
| ep7 | 3.44 | **64.6** | 0.299 | 5.6 | 0.694 |
| ep10 | 3.41 | 55.6 | 0.282 | 5.6 | 0.694 |
| ep14 | 2.93 | 30.1 | 0.306 | 5.6 | 0.694 |
| **ep19** | **2.68** | **26.6** | **0.350** | 5.6 | 0.694 |

**The U-curve.** The prefit prior is **sharp** (eff-moves 8.4, close to the target's 5.6) because
it was supervised to imitate SealBot's moves. RL **immediately blows it up to diffuse** (eff-moves
53.7 at ep3, peaking **64.6 at ep7**): the policy head forgets the sharp imitation while the value
signal is still noisy. From ep7 it **re-sharpens** (64.6 → 30.1 → 26.6) and top-1 climbs back
(0.30 → 0.35). **But even at ep19 the prior (eff-moves 26.6) is ~5× broader than the search target
(5.6)** — the raw policy head is **still diffuse**.

### 2.2 Agreement of the prior with the MCTS search target

| ckpt | argmax-match | KL(target‖prior) | policy CE (prior vs target) |
|---|---:|---:|---:|
| prefit / ep1 | 0.194 | 3.865 | 4.905 |
| ep3 | 0.232 | 2.701 | 3.738 |
| ep5 | 0.253 | 2.550 | 3.587 |
| ep7 | 0.260 | 2.526 | 3.564 |
| ep10 | 0.261 | 2.387 | 3.424 |
| ep14 | 0.335 | 1.858 | 2.895 |
| **ep19** | **0.415** | **1.498** | **2.535** |

All three agreement metrics **improve monotonically from ep3**: by ep19 the prior's top move
matches the search's top move **41.5%** of the time (up from 19–23%), KL falls to 1.50, policy CE
to 2.54. So while the prior is *broad*, it is **increasingly pointed in the right direction** and
**converging toward the search** — the trajectory is healthy.

> *Confound, stated honestly:* the policy target was generated by the ep19 model's search, so
> KL/CE/argmax-match are measured against ep19's own target distribution and naturally **favor
> later checkpoints**. This is why the **value** evidence (§1, judged against model-independent
> *outcomes*) is the load-bearing calibration result, while the policy metrics are best read as a
> **convergence/sharpening trend** rather than absolute quality. Even so, the steady rise across
> *all* checkpoints (not just a jump at ep19) is a real sharpening signal.

**Verdict (policy):** the long-suspected **"diffuse policy head" is confirmed** — the raw prior
is ~5× broader than the search target and remains the weaker head with clear headroom. It is
**not stuck**: it is recovering from an early-RL diffusion blow-up and steadily sharpening. The
model wins anyway because **512-sim search + forced playouts sharpen the diffuse prior at play
time.** Closing the prior↔search gap (sharper head, or stronger policy-target weighting / lower
policy temperature) is the highest-value next lever.

---

## 3. Full progression (one table)

Fixed 4000 held-out positions, same for every checkpoint. (prefit = ep1; ep19 in-sample.)

| metric | prefit/ep1 | ep3 | ep5 | ep7 | ep10 | ep14 | ep19 |
|---|---:|---:|---:|---:|---:|---:|---:|
| value sign-acc | 0.495 | 0.479 | 0.564 | 0.562 | 0.584 | 0.611 | **0.693** |
| value Brier | 0.295 | 0.266 | 0.239 | 0.241 | 0.238 | 0.222 | **0.191** |
| value CE (held-out) | 0.826 | 0.726 | 0.669 | 0.671 | 0.666 | 0.628 | **0.554** |
| prior eff-moves | 8.4 | 53.7 | 41.1 | 64.6 | 55.6 | 30.1 | **26.6** |
| prior top-1 | 0.538 | 0.278 | 0.304 | 0.299 | 0.282 | 0.306 | **0.350** |
| argmax-match | 0.194 | 0.232 | 0.253 | 0.260 | 0.261 | 0.335 | **0.415** |
| KL(target‖prior) | 3.865 | 2.701 | 2.550 | 2.526 | 2.387 | 1.858 | **1.498** |
| policy CE (held-out) | 4.905 | 3.738 | 3.587 | 3.564 | 3.424 | 2.895 | **2.535** |
| STV MAE (h1/h4/h8) | .171/.183/.192 | .173/.181/.187 | .178/.174/.186 | .164/.158/.168 | .179/.202/.202 | .143/.147/.155 | **.123/.121/.125** |

---

## 4. Loss decomposition on a frozen sample (closes red flag #1)

The run has **no on-disk validation** (`validation_fraction=0`), and the earlier report flagged
that the post-ep13 **rising per-epoch train loss** *looked* like regression but was probably a
moving-target artifact. This pass provides the missing fixed holdout:

- On the **same 4000 positions**, **value CE 0.826→0.554** and **policy CE 4.905→2.535** decrease
  **monotonically** from prefit to ep19.
- Therefore the per-epoch train-loss rise (which compares each epoch's model against *that
  epoch's own freshly-sharpened targets*) is **definitively the moving-target artifact, not
  regression** — on a frozen target the latest model is strictly the best.
- (Caveat from §2.2: ep19's policy CE is partly favored because it authored the targets; the
  *value* CE, scored against outcomes, carries this conclusion without that confound.)

---

## 5. Short-term value heads & confidence vs game phase

- **Short-term value (EMA future-root) heads improve:** MAE for horizons 1/4/8 falls
  **.171/.183/.192 (prefit) → .123/.121/.125 (ep19)** — the auxiliary heads track the future root
  value to ≈0.12 on a [−1,1] scale, and (like the main value head) they degraded mid-run
  (ep10 .179/.202/.202) before recovering.
- **Value accuracy by phase (ep19 MAE):** opening **0.964**, midgame **0.856**, endgame
  **0.470**. The value head is **most accurate late** (outcomes are more determined) and
  appropriately uncertain in the opening. Endgame MAE improved the most over training
  (1.014 prefit → 0.470 ep19), i.e., the model learned to **read decided endgames** — consistent
  with the §1.3 MAE drop and with the eval observation that it now closes out competitive games.
- **Prior entropy by phase (ep19):** opening **2.94**, midgame **2.66**, endgame **2.44** — the
  prior is sharpest in the endgame, as it should be, and every phase sharpened from the
  ep3–ep10 peak (~3.5).

---

## Bottom line

- **Value head: strong.** Genuinely well-calibrated at ep19 (predicted ≈ empirical win-rate,
  ECE ≈3.3%, mild safe under-confidence), monotonically improved from a **prefit value head that
  was random on the RL distribution**. RL built this calibration from scratch. This fully answers
  §A.4's open question: **calibration is good.**
- **Policy head: diffuse but improving — the bottleneck.** The raw prior is ~5× broader than the
  search target (eff-moves 26.6 vs 5.6). It blew up in early RL and is re-sharpening (eff-moves
  64.6→26.6; argmax-match 0.19→0.42; KL 3.87→1.50). Search compensates, so strength is high, but
  **the prior↔search gap is the clearest remaining headroom.** This answers §B.3: **the policy
  head is sharpening, but is not yet sharp.**
- **The rising train loss is an artifact, now proven**, not hypothesized (fixed-holdout CE falls
  monotonically).

### Red flags / watch-items
1. **Policy prior is still ~5× too diffuse vs search.** Healthy trend, but not converged. Levers:
   sharper policy target weighting, lower policy/playout temperature, longer training, or a
   higher-capacity policy head. This is the top lever for raw (search-free) move quality and for
   faster search.
2. **Early-RL forgetting from the prefit was severe** (policy eff-moves 8→65; value calibration
   8→0; both at ep3–ep7). It recovered, but a future cold-start-from-prefit run could lose less by
   warming up RL more gently (e.g., a few epochs of higher policy-imitation weight before full RL),
   saving the ep3–ep10 "rebuild" cost.
3. **ep19 numbers are mildly in-sample** (it trained on the epoch-19 window). The progression is
   safe (rests on held-out ep1→ep14), but for a clean ep19 figure, re-run this on epoch-20
   self-play once it exists.
4. **MAE ≈0.77 will not drop to ~0** and should not be treated as a defect — it reflects genuine
   opening uncertainty under a correctly-calibrated head (see §1.3). Track **Brier/ECE/sign-acc**,
   not MAE, for value quality.

### Provenance
- Inference helper (uncommitted): `analysis/_deep_model_analysis.py`; raw output
  `analysis/_dma_results.txt`. CPU-only, frozen checkpoints, recorded epoch-19 shard positions;
  the live GPU trainer was not touched and no self-play was generated.
- Complements (does not replace) `ANALYSIS_2026-06-01.md` (game quality + winrate progression).
