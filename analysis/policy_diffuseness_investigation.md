# Policy diffuseness investigation — dense_cnn Model 1 "scratch_64"

**Date:** 2026-05-29 · **Branch:** `rust-rebuild` · **Run:** `runs/dense_cnn_model1_scratch_64`
**Question:** as games get longer the model appears to play randomly — is this (a) model capacity, (b) training, or (c) environment/representation?

> **Constraint honored:** this is a read-only analysis. No training files, configs, checkpoints, or the live supervisor were touched. All model inference was done **on CPU** against **copies** of `epoch_000009.pt` / `epoch_000021.pt` (`analysis/_ckpt_*.pt`), not the live files, to avoid GPU contention or lock contention.

---

## TL;DR (headline finding)

The surface hypothesis — *"the policy distribution gets too diffuse as moves climb"* — is **half right, but the mechanism is the opposite of what it looks like, and it is not primarily a whole-model-size problem.**

- The **post-MCTS visit distribution** (the move actually chosen, and the training *target*) is **sharp and gets sharper** late-game (effective ≈ 3 moves, top-1 ≈ 0.7). It is *not* diffuse. **[verified from logged NPZ]**
- The **network's raw policy head** *is* persistently diffuse (effective ≈ 35–85 moves, top-1 ≈ 0.4) and is **most diffuse and least accurate in the longest games** (move 100–200: KL(target‖pred)=3.1, best move falls to rank 3). **[verified via CPU inference]**
- The **value head is excellent and accurate late-game** (sign-accuracy 0.96–1.0, corr ≈ 0.95 past move 40), and clearly learned over training. **[verified]**

**Hypothesis (high confidence):** Two coupled bottlenecks, both of which must be fixed *before* scaling to a bigger model:

- **(I) MCTS search budget is far too low for the action space.** 128 visits over ≤32 widened children, against **300→1700 legal moves**, means search visits only **~6–12 distinct moves (<1% of legal moves) late-game** (§6). The "sharp" visit target is therefore an **artifact of under-exploration** — search rubber-stamps the prior's top move after a shallow value check on a handful of candidates — *not* evidence of a well-resolved position. Low-sim search cannot generate targets that improve on a weak prior, so the whole system self-limits.
- **(II) The policy pathway can't represent a sharp, well-ranked policy** over the large action space (§4–5): a 64→2-channel + single-linear policy head on top of a 451K-param trunk. The raw prior is diffuse (top-1 ≈ 0.4) and its top-32 captures only ~70% of mass, so the best move can fall outside the set search even considers.

The value head is fine — it *judges* positions accurately and improves late-game — so this is **not** an environment/representation problem and **not** a "value capacity" problem. Late-game the agent *knows who is winning* but *cannot identify which move is good*, because search explores almost none of the legal moves and the prior that picks the candidates is diffuse. → confident value, near-arbitrary move → "plays randomly."

The net has 12.2M params but they are badly allocated (§4); raw size is not the issue. **Search budget (I) and policy-pathway capacity (II) are co-primary and coupled** — in AlphaZero-style training they bootstrap each other, so raising one without the other yields little.

---

## 1. What the logged data contains

Every selfplay game writes a per-game `.npz` (`runs/.../selfplay/epoch_*_game_*.npz`) with one row per recorded decision:

| array | meaning |
|---|---|
| `inputNCHW` (N,13,41,41) | encoded board state (the model input) |
| `policyTargetsNCHW` (N,1,41,41) | **post-MCTS visit distribution** (training target) |
| `rootPolicyNCHW` (N,1,41,41) | root prior used by search — **NOT raw network**: includes `root_policy_temperature=1.1` **and** Dirichlet noise (`mcts.rs` mutates edge priors in place) |
| `legalMaskNCHW` (N,1,41,41) | legal moves |
| `valueTargetsN` (N,) | game outcome z for that position's player |
| `metadataInputNC` (N,4) | `[turn_index, policy_surprise, frequency_weight, search_visits]` |

Games reach **~180 moves**; legal-move count grows to **~1700** by late game (sparse infinite board, stones spread). Because the recorded `rootPolicy` is noised+tempered, **isolating the raw network policy required forward passes on `inputNCHW`** (§3).

Scripts (re-runnable): `analysis/phase1_logged_entropy.py`, `phase2_raw_head_inference.py`, `phase3_target_vs_pred.py`.

---

## 2. Phase 1 — post-MCTS visit policy is SHARP and sharpens late (logged data, no inference)

Epoch 21, by move number (`effVIS` = effective #moves = exp(entropy); `HnormV` = entropy ÷ log(legal) where 1.0 = uniform):

| move# | legalN | effVIS | top1V | HnormV |
|---|---|---|---|---|
| 0–10 | 388 | 6.23 | 0.543 | 0.256 |
| 20–30 | 719 | **2.82** | **0.730** | 0.117 |
| 40–60 | 1019 | 2.91 | 0.738 | 0.109 |
| 100–130 | 1409 | 3.09 | 0.700 | 0.114 |

`corr(effVIS, move#) = −0.20`. **The training target concentrates as the game grows** — it does not approach uniform. (Bounded anyway by 128 visits over ≤32 children.) So the *label* the policy head learns from is sharp, and the *move played* in selfplay is decisive, not random.

---

## 3. Phase 2 — raw network heads (CPU inference on `inputNCHW`)

Epoch 21, by move number. `effRAW`/`top1`/`Hnorm` = raw policy head (masked-softmax over legal); `top32m` = prior mass on the top-32 moves (= the widening cap MCTS can even expand); value columns vs game outcome:

| move# | legalN | effRAW | top1 | top32m | \|vpred\| | vSignAcc | vCorr |
|---|---|---|---|---|---|---|---|
| 0–10 | 390 | 65.5 | 0.281 | 0.718 | 0.255 | 0.591 | 0.291 |
| 20–30 | 729 | 35.4 | 0.432 | 0.834 | 0.722 | 0.847 | 0.748 |
| 40–60 | 1011 | 49.9 | 0.418 | 0.791 | 0.903 | **0.960** | **0.940** |
| 60–80 | 1228 | 59.8 | 0.397 | 0.765 | 0.931 | **0.980** | **0.954** |
| 100–130 | 1386 | 101.8 | 0.289 | 0.668 | 0.986 | **1.000** | **0.998** |

Two clean, contrasting facts:

1. **Value head is NOT saturating.** It is *more* accurate and confident as the game grows — exactly the opposite of representation/capacity saturation. The trunk evidently has enough receptive field/capacity to encode the global late-game position **for the purpose of judging who is winning.**
2. **Raw policy head stays diffuse throughout** (top-1 ≈ 0.4, effective 35–100 moves) and **the top-32 prior captures only ~70–85% of mass** — so ~15–30% of probability sits on moves MCTS will never expand (widening cap = 32).

**Learning check (epoch 9 vs 21):** at epoch 9 the value head was a coin-flip (sign-acc ≈ 0.45–0.52, corr ≈ 0); by epoch 21 it is near-perfect late-game. So the model *is* learning and capacity is clearly sufficient for the value task. *(Caveat: a SealBot-distilled "bootstrap" re-init happened around epoch 15→16 — see `runs/.../_pre_bootstrap_*` and the config `initialize_from` — so epoch 9 vs 21 is partly a lineage change; the within-epoch-21 conclusions do not depend on it.)*

---

## 4. Phase 3 — the head underfits its OWN sharp target, worst in long games

The decisive measurement. Epoch 21, raw policy (pred) vs the game's own MCTS visit target:

| move# | tgtTop1 | predTop1 | tgtEff | predEff | polCE | KL(t‖p) | pred@best | bestRank |
|---|---|---|---|---|---|---|---|---|
| 0–10 | 0.535 | 0.281 | 6.31 | 65.5 | 3.52 | 2.00 | 0.201 | 2 |
| 20–40 | **0.735** | 0.441 | 2.78 | 34.6 | 2.48 | 1.73 | 0.336 | 1 |
| 40–60 | 0.738 | 0.418 | 2.90 | 49.9 | 2.59 | 1.85 | 0.333 | 1 |
| 100–200 | 0.604 | 0.279 | 4.28 | **84.5** | **4.27** | **3.14** | **0.175** | **3** |

Reading this:

- The **target is sharp** (top-1 0.6–0.74) but the **prediction is diffuse** (top-1 0.28–0.44). The gap is a large **KL of 1.7–3.1 nats** — the head is **underfitting sharp labels**, not being trained on diffuse labels.
- The gap is **worst in the longest games** (move 100–200): the head is most diffuse there, commits only 17.5% mass to the best move, and the best move slips to **rank 3**. This is the direct fingerprint of the user's observation: *late-game move quality degrades the most.*
- Mid-game (20–60 moves) is the sweet spot (best move ranked #1) — consistent with eval games being lost around move ~34.

Because the **target is sharp**, "the policy is diffuse because many moves are genuinely equivalent" is **ruled out**: search, given the same position, concentrates ~70% of visits on one move, so a best move exists — the head just cannot predict it.

---

## 5. Why — the architecture is badly balanced (parameter audit)

`Model1Network(channels=64, blocks=4, horizons=[1,4,8])` → **12,205,874 params**, allocated:

| component | params | note |
|---|---|---|
| trunk (`conv_in` + 4 `GatedResBlock`) | **451,456** | the entire shared "reasoning" capacity — tiny |
| `policy_head` | **5,653,333** | 93% of it is one `Linear(3362→1681)` |
| `opp_policy_head` (auxiliary) | **5,653,333** | a *second* giant FC head, weight 0.25 |
| `value_head` | 111,938 | small MLP, with a hidden nonlinearity |
| short-term value heads | 335,814 | |

The policy head is `Conv2d(64→**2**, 1×1) → flatten(2·1681) → Linear(3362→1681)`:

- It **squeezes the 64-channel trunk features down to 2 channels per cell** before the output map — a severe information bottleneck for a 1681-way spatial decision.
- The final map is a **single linear layer with no hidden nonlinearity** — a near-linear function of a 2-channel feature map. Compare the value head, which has a genuine `Linear→ReLU→Linear` MLP (and works great).
- It is simultaneously **over-parameterized** (5.65M) **and under-expressive** (2-ch bottleneck, linear).

So **~92% of the network's parameters are in two giant, weak policy FC layers; the part that actually builds representations (the trunk) has only 451K.** Receptive field of the trunk is ~19×19 (9 stacked 3×3 convs; hex-masked corners don't shrink the ±1 reach) on a **41×41** board — local features, with global mixing happening only at the linear policy layer.

This explains the value/policy asymmetry: "who is winning" is a coarse global scalar the small trunk + value MLP can learn; "which of ~1000 legal cells is best" needs sharp, spatially-structured, nonlinear policy features that the 64→2-channel linear head cannot represent — and the deficiency grows with the action space, i.e. with game length.

---

## 6. MCTS simulation budget vs action-space growth (first-class cause)

The recorded visit target is the **delta visit distribution for that turn**, normalized over the ~128 visits added that search call (`mcts.rs::visit_policy` with a baseline; zero-delta moves are dropped — verified, weight rows sum to exactly 1.0). So the count of nonzero cells = **how many distinct moves actually received a visit**, and `weight × 128` ≈ that move's visit count. Epoch 21, by move number (`analysis/phase4_sim_budget.py`):

| move# | legalN | moves visited | %legal visited | sims/legal | sims/visited | top1 | effVIS |
|---|---|---|---|---|---|---|---|
| 0–10 | 388 | 14.5 | 3.72% | 0.33 | 8.9 | 0.543 | 6.23 |
| 20–30 | 719 | 6.3 | 0.87% | 0.18 | 20.4 | 0.730 | 2.82 |
| 40–60 | 1019 | 6.3 | 0.62% | 0.13 | 20.3 | 0.738 | 2.91 |
| 80–100 | 1389 | 6.6 | **0.48%** | 0.09 | 19.4 | 0.724 | 2.86 |
| 130–200 | 1426 | 7.5 | **0.52%** | **0.09** | 17.1 | 0.617 | 4.25 |

The decisive facts:

- **Legal moves grow 388→1426 with move number** (corr +0.80), but **the number of moves search actually visits *shrinks* to ~6–7** (corr −0.24) and is **far below the 32-child widening cap** — so it is the *128-sim budget*, plus PUCT exploitation of a low-entropy value, that bounds exploration, not the cap.
- **Search explores <1% of legal moves late-game** (0.5%). 99.5% of legal moves never receive a single visit. `sims/legal ≈ 0.09` — roughly one simulation per *eleven* legal moves.
- This **reframes the "sharp visit target" of §2**: the target looks confident not because the position is resolved but because search only ever touched ~6–7 prior-favoured moves and committed to one. It is a **sharpened echo of the (diffuse, possibly wrong) prior**, not an independent signal. (Corroborated by §3: the visit-target's best move is the raw policy's #1–3 — search rarely overrides the prior, because at 128 sims it can't.)
- **Trend (epoch 9 → 21):** at epoch 9 search visited ~11–12 moves with effVIS ≈ 6–7; by epoch 21 it visits ~6–7 with effVIS ≈ 3. As value sharpens, PUCT exploits harder and explores *even fewer* moves — coverage of the legal set stays <1.5% and worsens. So the system increasingly trains on narrowly-explored targets.

Why this is a genuine training bottleneck, not just a curiosity: in AlphaZero/KataGo the policy improves because **search discovers moves the prior under-rated and writes them into the target**. With 128 sims over a 1000+ action space, search has almost no power to discover anything beyond the prior's top few — so the policy target ≈ prior, the head learns ≈ its own diffuse prior, and late-game errors are *reinforced* rather than corrected. The value head escapes this trap because "who is winning" is learnable from outcomes regardless of search width. **KataGo/AlphaZero train at ~400–1600 sims; 128 over this action space is very low.**

Aggravating factor — **Dirichlet noise** `fraction=0.25`, `total_alpha=10.83` → per-move α ≈ 0.02 over hundreds of legal moves (spiky), mixed at 25% — adds prior noise and is why the *logged* `rootPolicy` looks far more diffuse than the true raw head.

---

## 7. Verdict

Ranked by how much each binds, with the two co-primary causes coupled:

| Rank | Candidate cause | Verdict | Evidence |
|---|---|---|---|
| **1 (co-primary)** | **MCTS search budget too low for the action space** (128 sims, ≤32 children) | **Primary, and the first thing to fix for a training run.** | Search visits only ~6–12 of 300–1700 legal moves (<1% late-game); sims/legal ≈ 0.09 late; the "sharp" target is an under-exploration artifact that echoes the prior (best move = raw #1–3), so search can't generate targets that improve the policy. |
| **1 (co-primary)** | **Policy-pathway capacity / architecture** (mis-allocated) | **Primary, coupled with #1.** | Head underfits its own sharp targets (KL up to 3.1, worst in long games); 64→2-ch + single-linear policy head; 451K-param trunk vs 11.3M in two policy FCs; value (easy global task) learns where policy (hard spatial task) does not. |
| 3 | **Trunk depth / receptive field** | Secondary. | RF ≈ 19 < 41 board; but value works late-game, so the trunk is *adequate for judging* positions — precise move ranking may still benefit from more depth once #1/#2 are fixed. |
| 4 | **Environment / representation** | **Unlikely primary.** | Value head is highly accurate late-game → the encoding carries late-game signal; no plane scales pathologically with move count. (Minor: no explicit game-phase/move-count broadcast plane.) |

**Why #1 and #2 are co-primary and must be fixed together:** they bootstrap each other. More sims with the current weak head still feeds a diffuse prior into search (best move may sit outside the top-32 candidates). A better head with 128 sims still can't have its over-/under-rated moves corrected by search. **Scaling to a bigger model while leaving 128 sims in place would likely waste the extra capacity** — the bigger policy head would still be trained on narrowly-explored, prior-echoing targets.

**Confidence:** high that search budget and the policy pathway (not value, not representation) are the binding constraints; high that the visit-target sharpness is an under-exploration/underfitting artifact rather than genuine resolution or correct equivalent-move diffuseness; medium on the *exact* relative split between sims, policy-head architecture, and trunk depth — that needs the controlled experiments below.

---

## 8. Recommended next directions (NOT performed here — see §10)

> Per the user's instruction, **no fixes were applied** and **no training was launched.** These are documented for a future, deliberate effort that should land **before** scaling to a bigger model. Use **late-game KL(target‖pred) + best-move rank (Phase 3) and % legal-moves-visited (Phase 4) on a held-out set**, plus SealBot win-rate, as discriminators.

**Search side (address first — it gates target quality):**
1. **Raise simulations** to the AlphaZero/KataGo range (≥400, ideally 800–1600) so search explores meaningfully more than <1% of legal moves and can override a weak prior. This is the lever that lets the policy *improve* during training rather than echo its prior. Re-measure % legal visited and whether the visit-target's best move starts diverging from the raw prior's top-1 (sign that search is adding information).
2. **Reconsider widening + noise for the action space:** the 32-child cap is not the active limit at 128 sims, but becomes relevant once sims rise; tune `widening_max_children`/`widening_policy_mass` alongside sims, and lower `root_dirichlet_noise_fraction` (0.25→~0.10) to reduce prior noise.

**Model side (address together with search):**
3. **Fix the policy head:** the `Conv(64→2)+Linear(3362→1681)` head is both over-parameterized (5.65M) and under-expressive (2-channel bottleneck, no nonlinearity). A fully-convolutional head (e.g. `3×3 Conv→ReLU→1×1 Conv→1 logit/cell`) is cheaper and more expressive; same for/​drop `opp_policy_head`. This is an *architecture rebalance*, not "a bigger model."
4. **Then test trunk capacity:** `channels 64→128`, `blocks 4→8` (RF → ~35, near board size) — but only meaningful once #1 and #3 are in place, else the extra capacity trains on weak targets.

The single most informative comparison: with identical data, vary **sims (128 vs ≥400)** and **policy head (FC vs fully-conv)** in a 2×2 and watch late-game KL + % legal visited — that names how much each of the two co-primary causes binds.

---

## 9. What was verified vs inferred

**Verified (measured):** visit-target sharpness vs move#/legal-count; raw-policy diffuseness vs move#/legal-count; value-head accuracy vs move# and across epochs; the target-vs-prediction KL/rank gap and its worsening in long games; **legal-move growth (388→1426) vs the number of moves search actually visits (~6–12, <1% of legal) and sims/legal (~0.09 late)**; that the recorded visit weights are per-turn deltas summing to 1.0; exact head architecture and per-component parameter counts; SealBot win-rate/turn-count trend; that the recorded `rootPolicy` is tempered+noised (so it is not the raw head).

**Inferred (not directly measured):** that policy diffuseness + under-exploration is the *proximate cause* of the visually "random" late play (strongly supported, but no per-move strength oracle was used); the *relative* contribution of sims vs head-architecture vs trunk-depth (needs the §8 ablations); receptive-field number (~19) is analytic, not profiled; that more sims would let search override the prior (mechanistically expected from PUCT, not yet demonstrated on this run).

## 10. Process note

No fixes were applied. No configs, model code, the supervisor, or checkpoints were modified, and no training was started. All inference ran on CPU against deleted *copies* of the checkpoints. Conclusions are mirrored into `NOTES.md` as the shared source of truth.

*Artifacts:* analysis scripts + `phase1_summary.json`, `phase2_summary.json` in `analysis/`. Checkpoint copies `analysis/_ckpt_*.pt` can be deleted (293 MB).
