# hexfield_main_4 Divergence — Definitive Root-Cause Analysis

**Date:** 2026-06-20
**Run:** hexfield_main_4 (KataGo-faithful build), warm-started from `runs/hexfield_main_4_prefit/checkpoint_epoch5.pt`.
**Baseline (known-good):** hexfield_main_3.
**Verdict in one line:** main_4 trained the **value head into a self-reinforcing overconfidence feedback loop**. The search divergence stack (new_child_fpu + lazy_widening + root_fpu_reduction=0 + c_scale=0/c_puct=1.5) lets an already-overconfident value head dominate Q on balanced positions; self-play then exports near-±1 value targets on roughly-even openings; the next epoch fits those targets and gets *more* overconfident. Loss falls because the net fits its own corrupted targets; strength falls because the targets are wrong.

---

## 1. The core mechanism (what actually broke)

A single, monotone, model-side-confirmed pathology drives every symptom:

**VALUE-Q SATURATION ON BALANCED POSITIONS.**
- Most-visited-action |Q| at the **opening move (turn 0)** climbs monotonically over training:
  `0.17 (ep1) → 0.46 (ep3) → 0.51 (ep5) → 0.73 (ep10) → 0.79 (ep13) → 0.75 (ep19)`.
  main_3 stays correctly uncertain at the opening: `0.016 (ep40) … 0.099 (ep50) … 0.034 (ep55)`.
- **Model-side proof (cleanest single evidence):** running prefit vs ep1 vs ep19 on the SAME 40 fixed balanced openings, value-head `|v|` grows `0.141 (prefit) → 0.294 (ep1) → 0.648 (ep19)`. Signed mean stays ~0 (`-0.006 / 0.059 / -0.031`), so it is **two-sided overconfidence**, not first-player bias. The network *itself* learns to assign large |value| to near-even positions.
- The exported policy target is **diffuse** the whole time (entropy ~2.5 nats, top1 ~0.4, support median ~48–61) — search never sharpens to match the saturated value. Diffuse visits + saturated per-action Q = a value-overconfidence loop, **not** crisp tactical conversion.

**Downstream symptoms, all explained by the loop:**
- **Game-length collapse** (mean plies 71.7→40; full-decisions/game 22.5→13): positions get declared near-won earlier each epoch, one side commits to a shallow "winning-looking" line, games end fast. 100% of games end |value[-1]|==1 (decisive-in-outcome, degenerate-in-cause). The collapse is **epoch-progressive and monotone** — a fixed config cannot produce that; only the learned network changes between epochs.
- **stvalue_2/6/16 are the ONLY losses that regress** (+0.28..+0.52 ep1→ep5) — the classic target-corruption signature: short-horizon value can't fit a distribution that is collapsing under it.
- **All other losses fall** — the net is fitting its own increasingly-saturated targets (and shorter/easier games), so loss-space looks healthy while absolute strength sinks: candidate-vs-SealBot raw winrate `0.406 → 0.344` (never >0.5); candidate vs the prefit's own target main2_ep45 stuck at **20–25%** (~150–190 Elo BELOW where it warm-started).

---

## 2. Ranked root causes (most likely first)

### #1 — Value-saturation feedback loop, *enabled by the search-divergence selection stack* — CONFIDENCE: HIGH
**This is the culprit.** The mechanism: a slightly-overconfident value head + a selection stack that lets that value dominate Q on unexplored/early nodes → search reports near-±1 Q on balanced openings → those become the recorded value targets → next epoch the head fits them and saturates harder.

The enabling stack (all in `Divergences::production()`, `tree.rs:152-169`, all ON in self-play):
- **`root_fpu_reduction = 0.0`** (`search.rs:146-153`, `root_fpu_for` returns it for EVERY move class): the root FPU baseline becomes `parent_value` for all new/unvisited root children. Unexplored moves inherit the (overconfident) parent value instead of being pessimistically discounted.
- **`new_child_fpu`** (`tree.rs:881-887`, `new_child_score = (parent_value - fpu_reduction) + prior*scale`): a fresh interior child now inherits `V - 0.2` instead of the tiny legacy U-only prior. At winning interior nodes (V>0) that baseline is high, so the value head's optimism propagates into newly-opened branches.
- **`lazy_widening`** (`tree.rs:873-877`): drops the frozen `max_eligible_children` cap entirely; FPU is now the *sole* broadening gate. With FPU baselines pinned near `parent_value`, breadth is bounded only by how optimistic the value head is — which keeps rising.
- **`c_scale 0.45→0.0` ⇒ c_puct constant 1.5** (`tree.rs:841` selection uses `c_for(N)=1.5`), **kept at 1.5 not the KataGo-faithful 1.1** (self-acknowledged in-config as over-exploration that "compounds with the other main_4 exploration changes").

These four together reshape *which* nodes get visited and *what Q* they report, with no change to the loss surface — the textbook "strength falls while loss improves" signature.

Evidence: turn-0 |Q| 0.17→0.79 monotone (search-behavior-probe, confirmed twice); model-side |v| 0.14→0.65 on fixed openings (data-target-checkpoint-integrity); code loci above; rust tests confirm `new_child_score_matches_existing_zero_visit_edge` and `lazy_widening` drops the cap.

### #2 — soft-policy aux head (weight 4.0) starving/diluting the shared trunk — CONFIDENCE: MEDIUM (secondary)
The new train-only KataGo soft-policy head is **~74–76% of the weighted objective EVERY epoch** (`losses.py:34` SOFT_POLICY_WEIGHT=4.0 × raw ~4.0 vs main policy 1.0 × ~3.1; weighted total reproduces logged `loss_total` to the digit: ep1 23.479=23.479, ep19 21.043=21.043). It backprops into the shared trunk (`model.py` `soft_policy_conv` off the shared LN_final output). Main-policy raw loss improves ~2x SLOWER than main_3 (m4 3.52→3.07 over 19ep vs m3 3.32→2.51 by ep10). **But:** its target is `(visit_policy)^0.5` over the same support, so its gradient is *aligned* with (a flatter version of) the hard policy — a dilution/regularizer, **not** an antagonistic objective, and it is absent from serve so it cannot directly shape search or the value targets. Real but secondary. (Correction to the brief: the prefit DOES contain soft_policy_* weights — warm-started pretrained, not zero-init, so no early-epoch gradient shock.)

### #3 — Hot tuning bundle amplifying the loop (lr 5e-4, moves_left_weight 0.2, temp/halflife) — CONFIDENCE: MEDIUM (amplifier, not cause)
`learning_rate 3e-4→5e-4` makes the net fit the corrupted targets *faster* (optimizer is healthy: grad_norm steady, clip_fraction collapses, amp stable — not catastrophic forgetting). `moves_left_weight 0.1→0.2` doubles a head that decisiveness-steers and adds trunk pressure toward shorter games. `temperature_halflife 60→45` + `root_policy_temperature` ramp lower the *asymptotic* length floor (intended). None of these can produce the *progressive* epoch-over-epoch collapse (they are fixed functions of ply); they only nudge/amplify the #1 loop.

---

## 3. Confirmed bugs vs harmful-but-intended vs red herrings

### (a) CONFIRMED BUGS (wrong code)
**NONE.** Every suspect was traced end-to-end and found correctly implemented. 22/22 rust tests pass (incl. all six divergence flag-pin + shaped-Dirichlet + nucleus_f64 + new_child_fpu + clean-cache tests); 8/8 soft-policy tests pass; the cleanup merge 86195592 is benign (comment/docstring/dead-code only, plus a genuine *fix* to replay_expand value_mask which is INERT here since truncated_games=0). The warm-start loaded clean (loaded:168, missing:[], unexpected:[]; trunk cos≥0.99). **This is not a code regression — it is a tuning/design regression.**

### (b) HARMFUL-BUT-INTENDED (correct code, bad-for-Hexo choices)
1. **`root_fpu_reduction=0.0`** + **`new_child_fpu`** + **`lazy_widening`** — KataGo-faithful, but on Hexo's flat ~337–777-move prior they let an overconfident value head propagate ±1 Q into unexplored breadth with no cap. **#1 root cause.**
2. **`c_scale=0` with `c_puct` left at 1.5** (not 1.1) — self-acknowledged over-exploration stacked on top of the above.
3. **`soft_policy_weight=4.0`** — 75% of the objective on an aux regularizer; trunk dilution. **#2.**
4. **`learning_rate=5e-4`** (vs main_3's own sweep finding lower-is-better) + **`moves_left_weight=0.2`** — amplifiers. **#3.**

### (c) RED HERRINGS (ruled out)
- **shaped Dirichlet + clean_root_prior_cache** — byte-faithful to KataGo; shaped+fraction-0.20 makes noise *more* focused (opposite direction of the pathology); the cache *fixes* reuse-compounding. Cannot produce value saturation (it only perturbs root priors, never leaf value). VERDICT: correct, no bug, cause=low.
- **dynamic-cpuct target export (`pruned_dynamic_cpuct`)** — provably INERT at c_scale=0 (`c_for(N)=1.5` for all N); exported target is identical to static-c. Export faithfully records whatever visits the search produced. VERDICT: no-op, cause=none.
- **forced_playout_k 2.0→1.0** — at 337 legal moves both k=1 and k=2 force exactly 1 visit/child at the opening (NULL change there); where it differs k=1 forces FEWER (less spray). VERDICT: cause=low.
- **rust-expand value_mask / truncation masking** — INERT: `truncated_games=0` every epoch ⇒ outcome_valid all 1 ⇒ value_mask all 1. Cannot have altered targets here.
- **broken warm-start / corrupted-or-mis-masked targets / catastrophic forgetting / stale .so** — all refuted: targets are well-formed pure ±1, trunk loaded faithfully (cos≥0.99), policy head drifts moderately (not wiped), .so built 2026-06-19 16:13 *after* source.
- **"search uniformly sprays visits too thin everywhere"** — REFUTED at matched turns: opening branching is similar to main_3 (~26–42), and main_4 is actually *sharper* in mid-game (perp 8.0 vs 12.0). The aggregate over-broadening was a game-length composition artifact. The unambiguous signal is value-Q saturation, not search breadth.

---

## 4. "Something broke when it was working before" — what introduced the regression

**It is NOT a single broken commit. It is an accumulation of intended-but-Hexo-hostile tuning, concentrated in two commits.** No sign-flip, dropped mask, wrong flag-default, or lost early-return exists anywhere in the change-set.

- **Primary introducer: commit `8714bb55`** ("feat KataGo-faithful + soft-policy"). This adds the six search divergences (the #1-enabling selection stack: new_child_fpu, lazy_widening, root_fpu_reduction first-class) **and** the soft-policy aux head (#2). This is where the value-saturation enabling mechanism enters.
- **Co-conspirator: commit `85808c52`** (c_scale 0.45→0, root_policy_temperature ramp) and **`58465399`** (lr 5e-4, moves_left_weight 0.2, halflife 45). These tune the system into the over-exploration + hot-fitting regime that turns the latent loop into a runaway (#3).
- **Exonerated: cleanup merge `86195592`** — benign; if anything it *fixed* a real value_mask bug (inert here).

The honest framing for the user: main_3 worked because its conservative FPU/widening/c_puct kept the value head's optimism in check; main_4's KataGo-faithful descent, valid on Go, is mis-matched to Hexo's flat huge-branching prior and lets value overconfidence compound.

---

## 5. Prioritized remediation plan

### FIRST — smallest reversible change, biggest expected effect
**Restore FPU discipline at the root.** In `configs/hexfield_main_4.toml`, set:
- **`root_fpu_reduction = 0.2`** (match interior FPU; stop unexplored root children inheriting the overconfident `parent_value`). This is the single lever most directly upstream of turn-0 Q saturation — it makes balanced openings score near 0 again, breaking the feedback loop at its source. (`search.rs:146-153`.)

Do this together with the cheapest two config reverts (same file, zero rebuild):
- **`c_puct: 1.5 → 1.1`** (KataGo self-play-faithful; removes the self-acknowledged over-exploration). (`tree.rs:841` selection.)
- **`learning_rate: 5e-4 → 3e-4`** (stop fitting bad targets fast; matches main_3's own lower-is-better sweep). (`plugin.py:41-46`, constant-LR AdamW.)

These three are pure-config, reversible, no recompile.

### SECOND — relieve trunk dilution
- **`soft_policy_weight: 4.0 → 1.0`** (or 0.0 to fully detach for one diagnostic epoch). (`losses.py:34` default 8.0 / config line 250.) Restores the main policy/value gradient share.
- **`moves_left_weight: 0.2 → 0.1`** (revert to main_3). (config.)

### THIRD — if Q saturation persists after the above
Disable the breadth-uncapping divergences (requires the Rust path but no source change — flags only):
- **`lazy_widening = false`** (restore the frozen-cap widening). Then, if needed, **`new_child_fpu = false`**. Keep `nucleus_f64`, `clean_root_prior_cache`, `dirichlet_shaped`, `pruned_dynamic_cpuct` ON (verified harmless/beneficial).

### Re-run & validation
1. Relaunch a **short 5-epoch run from the same prefit** with FIRST-tier changes only.
2. **Primary success metric:** turn-0 most-visited |Q| stays **≤ ~0.2** through ep5 (vs the broken 0.17→0.51). Decode `samples/epoch_*/game_*.npz` `q_pol_q` exactly as the probe did. Also check model-side |v| on the 40 fixed balanced openings stays ≤ ~0.2.
3. **Secondary:** mean game length stops collapsing (holds ≥ ~60 plies, ideally grows like main_3); candidate-vs-SealBot winrate ≥ 0.5 by ep5.
4. If turn-0 |Q| is controlled but strength still lags, layer in SECOND-tier (soft-head relief), then THIRD-tier (widening).

### Guardrail to add
Add a **self-play assertion / diagnostic alarm** on exported value targets: log per-epoch `mean(|q_chosen|)` restricted to the **first 25% of plies**, and **fail-fast / alert** if it exceeds a threshold (e.g. 0.35) — this is the earliest, cleanest leading indicator of the feedback loop (it crossed 0.5 by ep5, long before Elo confirmed the decline). Wire it into `events.jsonl` alongside `root_value_mean`/`root_policy_entropy_mean`.

---

## 6. Inconclusive items & the one experiment to resolve each

- **Relative contribution of the soft-policy head (#2) vs the search stack (#1).** The soft head's trunk-gradient *magnitude* dominates loss, but its *direction* is aligned (regularizer), so its true causal share is inferred, not measured. **Resolving experiment:** one diagnostic epoch with `soft_policy_weight=0` (head detached), all else fixed. Prediction: main-policy raw loss falls faster (confirms dilution) **but turn-0 |Q| saturation and length collapse persist** (confirms the soft head is not the value-saturation driver, only a trunk diluter).
- **Which single search flag is load-bearing for the loop.** #1 names a *stack*; the relative weights of root_fpu=0 vs new_child_fpu vs lazy_widening vs c_puct=1.5 are not individually isolated. **Resolving experiment:** the FIRST-tier `root_fpu_reduction=0.2` lesion alone — if turn-0 |Q| recovers, root FPU is the dominant lever and the others are amplifiers; if not, escalate to the THIRD-tier widening lesions one at a time.
- **Absolute Elo magnitudes are noisy** (BT anchor main2_ep45 drifts 173.8→74.1; ~20 decided games/edge). The *sign/direction* (flat-to-down, below warm-start) is robust; the *magnitude* is not. **Resolving experiment:** raise eval power (more games/anchor) on the re-run rather than trusting point Elo.
