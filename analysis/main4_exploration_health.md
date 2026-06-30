# main_4 Exploration Health Report

**Date:** 2026-06-20 | **Run:** hexfield_main_4 (ep52 latest full; ep53 partial) | **Verdict:** HEALTHY — committal but still exploring; no urgent change.

---

## TL;DR

main_4 is **NOT exploiting too much**. The headline entropy decline (root_policy_entropy 1.66@ep5 → 1.358@ep52) is **benign explore→exploit convergence of a strengthening net** (+311 Elo, beats SealBot ~85%, healthy ~108-ply games), not a discovery-killing collapse. It **can still find new strategies** — every independent novelty signal is flat-to-rising. The recent lr/batch sharper-prior changes **did NOT measurably cut exploration**; if anything the search-vs-prior discovery gap grew. Recommendation: **no urgent fix**, switch monitoring off raw entropy onto support-normalized entropy + policy_surprise.

---

## Q1 — Is exploration HEALTHY or EXPLOITING TOO MUCH? → HEALTHY

The raw entropy number is misleading because the stored policy support is the **MCTS-expanded children set (~14–20 moves)**, NOT the 337–777 legal set. Judge 1.358 nats against ln(support)≈2.68, not ln(700)≈6.5.

| Metric | ep5 | ep30 | ep52 | Read |
|---|---|---|---|---|
| raw root_policy_entropy (nats) | 1.66 | 1.478 | 1.358 | falling |
| mean expanded-children support | 19.5 | 17.0 | 14.6 | falling **by design** (capped widening 96 + dynamic-c_puct pruning + root_fpu_reduction=0.2) |
| **H / ln(support)** | **0.615** | **0.586** | **0.575** | **FLAT** — near-constant fraction of achievable max |
| top1 visit-mass | 0.49 | — | 0.56 | far from ~0.9 collapse signature |
| effective move count exp(H) | 10.3 | 9.2 | 7.6 (FirstStone) / 3.9 (full-PCR) | still multi-move |

**The decline is ~90% a candidate-set shrink, not a distribution collapse.** Support-normalized entropy is essentially flat. The trend is **decelerating, not accelerating**: slope/ep = −0.0121 (ep5–20) → −0.0056 (ep21–40) → −0.0064 (ep41–50). ep52=1.358 is a single-epoch noise trough (epoch-to-epoch swings ±0.10–0.15 all run; **ep53 already rebounded to 1.50–1.53**).

**Cross-run clincher:** main_3 sits HIGHER (~1.8–2.2 nats) but is the defensive-lock / conversion-failure pathology (150–176-ply games). main_3 contested-midgame runs eff~13 over 55 candidates with top1 stuck at 0.36 — it cannot commit, hence cannot convert. main_4 commits to a tight ~15-candidate set at top1~0.55, which is the **source** of its strength. Higher raw entropy ≠ healthier.

## Q2 — Will it FIND NEW STRATEGIES? → YES

Four independent novelty signals all agree discovery is intact:

1. **Search-discovery channel (decisive):** policy_surprise = KL(visit‖prior) is **FLAT across all 52 epochs** — mean ~0.60, frac KL>0.5 = 0.382@ep5 → 0.380@ep52 (unchanged). Search overrides the raw prior just as much late as early.
2. **Move-universe still growing:** cumulative considered-move universe keeps expanding at the latest epochs (late 2018→2075→2169→2233 ep45→50→52→53); new ≥1%-mass moves still enter every epoch (1–6% novel-vs-history even ep45–53). A frozen space would flatten — it does not.
3. **Top-move turnover stable:** dominant-argmax Jaccard holds ~0.65–0.72 consecutive epochs end-to-end; ~17–20% new-dominant-move churn every epoch through ep52→53 (same rate as ep5–20). No convergence toward 1.0.
4. **No degenerate line / local optimum:** most-dominant single move share tiny and flat (early 0.02–0.03, mid 0.01–0.02, late ≤0.01). Transient opening concentration dips at ep20/30/35 **fully reverted** by ep40+ — the opening re-broadened on its own.

**Opening (the discovery frontier) is wide and NOT freezing:** ply1 raw prior near-uniform (effN ~47 of 60, top1~0.025); turn-1 canonical effN ~4.2–4.4 (better than the ep30 dip of 3.2); k6 distinct openings ~226–233/256 with ~85% singletons. Shaped Dirichlet (alpha=10.83, frac=0.20) fully dominates the flat opening prior. Strategy space is **concentrating compute onto stronger lines (intended), not freezing.**

## Q3 — Did lr/batch changes hurt exploration? → NO

Across the b32(ep34–40) → b128(ep41–50) → b256/lr5e-4(ep51–53) bands:

| Metric | b32 | b128 | b256/lr5e-4 | Read |
|---|---|---|---|---|
| policy_surprise mean (KL gap) | 0.589 | 0.617 | 0.603 | **GREW** — sharper prior ⇒ search disagrees MORE |
| frac KL>0.5 | 0.363 | 0.394 | 0.381 | up |
| q_pol_q value spread | 0.253 | — | 0.271 | up |
| top1-mass | 0.526 | 0.551 | 0.545 | +2pp (tiny) |
| eff-moves | 9.0 | 8.0 | 8.1 | −1 |
| H/ln(support) | 0.605 | 0.586 | 0.591 | −2% (noise) |
| opening entropy (turn≤4) | 2.73 | — | 2.67 | −2% (untouched) |

A cleaner gradient sharpened the per-move distribution by a couple percent but **left the discovery channel and opening diversity intact** — and the discovery gap moved the *opposite* way. Caveat: only 2 epochs exist after the ep51 change; re-confirm the slope after ep56+.

### Recommendation: NO CHANGE NEEDED (exploration is fine)

If you want a small margin as the prior keeps sharpening, ranked config-only nudges (target the **midgame**, where the decline lives — NOT the opening, which is healthy):

| # | Change | Expected effect | Risk | Alters learning? |
|---|---|---|---|---|
| 1 | **temperature_halflife_plies 45 → 60** | holds sampling temp higher into the ~108-ply midgame (halflife 45 collapses temp to near-floor by mid-game) | low | No (self-play sampling only) |
| 2 | **pcr_full_proportion 0.33 → ~0.40** (coverage — highest structural leverage) | only ~33% of moves get ANY noise/forced-playouts; the other 67% run near-greedy. Raising coverage is higher-leverage than noise magnitude | low–med (more full searches = slower self-play) | No (search-time only) |
| 3 | **root_dirichlet_noise_fraction 0.20 → 0.25** OR **total_alpha 10.83 → ~8** | spreads more noise at the root; shaped-dirichlet so it lands on more candidates | low | No (search-time only) |

**Do NOT:**
- lower temperature_floor below 0.15 (it never binds in a typical game anyway — first hit at ply124 > mean length 108)
- touch forced_playout_k=1.0 (the dominant, healthy, epoch-stable discovery floor)
- raise widening_max_children above 96 (the candidate-set narrowing IS the designed strength gain)
- raise opening Dirichlet (opening is the least at-risk part)

**Monitoring (replace raw entropy):**
- Alert if **policy_surprise mean < 0.40** sustained 3+ epochs
- Alert if **H/ln(support) < 0.50** sustained
- Alert if **contested-midgame (ply21–60, |q|<0.15) eff-moves < 3.0 with top1 > 0.7** sustained 3+ epochs
- Alert if mid/midlate **novel-considered-move rate < 1%** sustained 4+ epochs AND cumulative universe flattens

### Separate confound (not exploration, but watch)
The rust-expand value_mask bug (per MEMORY, live since ep46) trains truncated-game value/stvalue/cell_q heads unmasked. Not an exploration metric, but it biases the value signal that drives Q-based pruning of the candidate set — keep in view.
