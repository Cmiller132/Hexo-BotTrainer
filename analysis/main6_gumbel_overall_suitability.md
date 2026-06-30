# Gumbel AlphaZero — Overall Suitability for the hexfield Project

## 1. Bottom-line verdict

Full Gumbel AlphaZero is the **wrong paradigm choice** for hexfield — a weak overall fit. Two of its three mechanisms (Gumbel-Top-k root sampling + Sequential Halving, and deterministic non-root selection) are *low-simulation sample-efficiency* tools imported into a high-visit (1024-full / 192-fast) run that does not occupy that regime, and the one piece that would cut compute (Sequential Halving) structurally fights the async streaming scheduler that keeps this project's FLOP-bound GPU saturated. The only defensible adoption is mechanism #3 alone — the completed-Q training target — flag-gated and reversible, leaving PUCT search fully intact; the rest is a lateral move at best and adds integration risk for no measured strength gain.

## 2. The question, framed correctly

This is **not** "does Gumbel fix the KL=0.67 target-variance issue." That issue has a set of free config levers proposed against it (`pcr_full` 0.33→0.5, `c_puct` 1.5→1.1, `dirichlet` 0.25→0.20) that are *not yet run or measured* — the companion plan sequences them as Stage-0 preconditions — so it is set aside here as motivation, not because the gap is known to be closed.

The real question is a **paradigm-foundation** one: *if we were choosing the search + training paradigm for a self-play hex bot from scratch today, would full Gumbel AlphaZero be a better foundation than the current PUCT + Dirichlet + dynamic-c_puct + LCB stack* — judged on the project's actual long-term goals:

- a **stronger** bot,
- trained **cheaper** (single RTX 4070 Ti, ~9 pos/s, ~40-min epochs),
- more **robustly** (long unattended supervised runs),
- more **maintainably** (effectively solo-maintained, large existing KataGo-faithful PUCT codebase).

The distinction matters because a paradigm can be elegant in the abstract yet a poor fit for a specific operating point, hardware budget, codebase, and maintenance model. Gumbel must be judged on its *general* merits and demerits for *this* project across all those axes simultaneously — not on whether one mechanism nudges one metric. The honest read is asymmetric across the three mechanisms, so a single thumbs-up/down on "Gumbel" obscures the answer; the scorecard below separates them.

## 3. Dimension-by-dimension scorecard

| Dimension | Fit rating | One-line reason |
|---|---|---|
| **Strength ceiling / policy quality** | Neutral (wash) | Completed-Q target (#3) *plausibly* lowers extraction variance at 1024 visits (unverified — see §6); #1/#2 buy ~nothing at high sims and risk *lowering* the ceiling via value-Q coupling. KataGo stays PUCT at high visits. |
| **Throughput / compute** | Weak fit | Throughput is GPU-FLOP-bound, not visit-linear; the "256 visits ⇒ 4× epochs" math is false here, and Sequential Halving's late-round candidate decay starves the batch occupancy the scheduler depends on (the ~213 figure is the *observed mean* effective batch, vs the `flush_target=1024` cap). |
| **Robustness / long-run stability** | Weak fit | Current visit-count target is value-head-*independent*; the completed-Q target deliberately couples the target to the value head, re-opening main_4's value-Q saturation death. #3 requires four guards to not regress, raising integration risk. |
| **Simplicity / maintainability** | Neutral (cancels) | In principle collapses a large coupled knob surface (real win); in practice means a multi-week rewrite of the repo's most fragile, KataGo-faithful, parity-tested code — and the shippable slice (#3) *adds* knobs while removing none. |
| **Hex game / model fit** | Good fit | Pure win/loss Q in [-1,1] is an ideal rescale-free sigma input; late-growing branching (337→~1000) never starves SH; cell_q head already plumbs the per-action Q the target consumes. (Mechanism #2 is the worst fit here — many balanced midgame nodes.) |
| **Scalability / future-proofing** | Weak fit | The async per-root streaming scheduler is the right single-GPU batching paradigm and stays better as the net widens; SH's round structure fights it harder at scale. #3 is the only net-size-invariant, decoupled, reversible piece. |

## 4. Where Gumbel helps / where it doesn't

**Where it genuinely helps this project:**

- **Completed-Q target estimator quality (#3).** `softmax(logits + σ(completedQ))` is a smooth, monotone function of backed-up Q means rather than a discontinuous `N(a)/N` visit-count ratio. On the ~7-move near-ties that dominate hex midgames, that *plausibly* lowers extraction variance at the *same* 1024 budget — unverified, and conditional on the residual gap being variance rather than positional entropy (see §6).
- **Game/model match of #3.** Hex's pure win/loss Q in [-1,1] (`Q_UTILITY_WIDTH=2.0`) is the ideal, rescale-free input to the sigma transform; the `cell_q` head already trains on exactly the per-action Q the target needs. The training-target half of the pipeline is genuinely half-built.
- **Knob-surface reduction in principle.** Full Gumbel collapses `c_puct / c_scale / c_base / FPU / root_fpu / Dirichlet / widening / LCB / multiple temperatures` — a surface whose interactions have *already* caused real failures here (main_4's `lazy_widening` + `root_fpu` stacking). Fewer coupled knobs is a real maintainability win for a solo maintainer who tunes by config sweep.
- **A principled low-sim option for the future.** If the net ever grows large enough that 1024 full visits become unaffordable per epoch, Gumbel's provable-improvement-at-low-sims is the principled way to spend fewer, better-allocated sims — a lever PUCT structurally lacks.

**Where it doesn't help or actively hurts:**

- **Throughput.** Wall-clock is GPU-FLOP-bound on deep games with a host-side per-decision tail, not visit-linear; halving visits does not halve epoch time. Worse, Sequential Halving's candidate decay (m→1) shrinks per-flush batch occupancy below the ~213 *observed-mean* effective batch the scheduler relies on (a measured/MEMORY figure, distinct from the `flush_target=1024` cap) — the lever meant to buy throughput is in direct tension with what currently delivers it. Addressable fraction is bounded to the ~33% Full moves anyway.
- **Robustness.** The current target is value-head-*independent* (Q laundered through discrete visit counts), so a transiently overconfident value head cannot directly write the policy target. The completed-Q target couples them and gives the value head a direct write path via the unvisited-action fallback — re-opening exactly main_4's value-Q saturation (|Q| 0.17→0.79).
- **Near-term simplicity.** The realistically-shippable slice (#3 only) *adds* knobs (sigma `c_visit`/`c_scale`, m, min-visit floor) and code paths across the Rust/Python schema boundary while removing *zero* PUCT machinery.
- **Maintainability of the rewrite.** #1/#2 require porting Sequential Halving into a per-root-async, `par_iter_mut`, global-flush Rust core and re-proving the stale-prefetch-discard guarantee — the largest, least-reversible change in the repo, for a constraint (visit budget) that is not currently binding.

## 5. The KataGo question

The project's explicit north star is KataGo — the strongest open AZ-lineage engine, with full access to *both* paradigms. KataGo **stays on PUCT + Dirichlet + dynamic-c_puct + LCB at high sims.** That is the behavior you'd expect if Gumbel were a low-sim convenience rather than a high-visit *strength* frontier. We are aware of no major published high-visit AZ engine that runs full-Gumbel as its strength frontier. (Honest caveat: KataGo's PUCT choice could be partly legacy/path-dependence — it predates Gumbel-AZ and carries years of accreted PUCT tuning — rather than a clean, informed rejection of the paradigm. That weakens the "considered rejection" reading but not the practical signal: the strongest reference engine, given the option, has not found a reason to switch.)

This is decisive for several reasons, and a reasonable person should weight it heavily:

- The project spent main_2 → main_5 **aligning to that reference**, with byte-identical golden-vector parity tests (`Divergences::production` / `parity`) and a deliberately KataGo-faithful Rust MCTS core.
- That faithfulness is itself an asset: it is an external, debuggable correctness anchor and a reservoir of KataGo's years of tuning lore that the maintainer can reason by analogy to.
- Abandoning it forfeits both — for a paradigm whose end-goal benefit (a stronger bot) the project's own evaluation apparatus **cannot even resolve** above its ~48–90 Elo standard error.

**Implication:** the north-star stays PUCT. Any Gumbel adoption must be incremental, divergence-flag-gated, parity-safe (default-off), and reversible — never a from-scratch swap of the working PUCT stack. The burden of proof sits on Gumbel to show *net-positive* benefit, not mere non-regression.

## 6. Conditions for adoption

The verdict is tightly tied to one fork: **does the project ever want to drop visit counts for throughput?**

**Triggers that would make full Gumbel (#1/#2) the right call:**

- A future, substantially larger net makes 1024 full visits unaffordable per epoch, *and*
- the GPU forward stops being the dominant bound (host/visit work becomes binding), *and*
- the §7.2 synthetic-bandit regret simulation shows m=16 SH at 256 visits matches 1024-PUCT target quality, *and*
- a **direct 256-visit-SH pos/s measurement** beats the current ~9 pos/s while matching 1024-PUCT target quality.

Only with *all* of these does the multi-week rewrite of #1/#2 earn its keep. Absent that measured win, #1/#2 are the wrong call.

**Triggers that would make mechanism #3 (completed-Q target) the right call:**

- The free config levers (`pcr_full` 0.33→0.5, `c_puct` 1.5→1.1, `dirichlet` 0.25→0.20 — the live value is genuinely 0.25 despite a stale header comment, and these are *proposed-but-not-yet-measured* Stage-0 preconditions) are run and exhausted on ep115-120, *and*
- Stage-0 confirms the residual gap is **variance** (KL/variance > ~0.4 nat survives the levers), *not* irreducible positional entropy `H(target)≈1.25` (which #3 cannot touch), *and*
- it is gated behind the saturation kill-switch (two-tier turn-0 |Q| trip) plus a visit-weighted, min-visit-floored node-value fallback (never a bare child-eval read), *and*
- main_6 is warm-started from a **healthy, saturation-free** post-config-lever checkpoint so the value head is honest before the target depends on it.

**Triggers that mean "don't bother":**

- The free config levers close most of the gap → ship those, skip Gumbel.
- The residual is positional entropy, not variance → #3 cannot help; stay PUCT.
- A Stage-2 build drops pos/s at equal visits, or requires per-round flush-path tuning.
- **Hard NO-GO:** turn-0 |Q| climbs toward main_4's 0.79 despite the guards.

## 7. Holistic recommendation

**Partial, not paradigm. Not now for #1/#2.**

Do **not** pursue full Gumbel AlphaZero. The full paradigm targets a low-sim regime the project does not occupy, fights the async scheduler that makes the project cheap, re-opens a robustness failure the project already paid to escape, and forfeits the KataGo external anchor — all for an end-goal benefit the project's evaluation cannot resolve.

Pursue **mechanism #3 (the completed-Q training target) only** — flag-gated, reversible, PUCT search left fully intact — and *only after* the free config levers are exhausted and Stage-0 confirms the residual is variance (not entropy). Gate it hard behind the saturation kill-switch and a visit-weighted, min-visit-floored node-value fallback. If turn-0 |Q| climbs toward main_4's 0.79, it is a NO-GO.

That captures essentially the entire plausible upside — a plausibly-real, well-matched estimator improvement on hex's clean bounded Q (conditional on the residual being variance) — at a fraction of the risk. The honest one-line summary: **Gumbel offers this project one cheap, plausibly-good training-target idea, not a better foundation.** Adopting #1/#2 is a lateral move at best, and a costly one that re-opens a robustness failure for no measured gain at worst.

## 8. Pointer

This document is the **suitability decision** (is Gumbel a good paradigm fit overall). The detailed staged **implementation plan** — for use *if* the project decides to proceed with the #3 slice (and the gated path to #1/#2) — already lives at:

`E:/Hexo-BotTrainer-hexgt/analysis/main6_gumbel_az_plan.md`

That doc is the *how*; this one is the *whether*.
