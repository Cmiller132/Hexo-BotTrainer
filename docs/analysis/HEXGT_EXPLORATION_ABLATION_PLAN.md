# HEXGT Exploration-Constant Ablation Plan

Status: **PLAN — awaiting review. No sweep launched.** Graph-model-centric:
dense_cnn is *not* the benchmark (it appears only as an optional external yardstick).

## 0. Corrected target (read this first)

The exploration constants — Dirichlet noise (`total_alpha`, `epsilon`),
move-temperature schedule, `c_puct`, `root_policy_temperature`, nucleus widening
mass — do **not** matter because they change the static strength of a *frozen*
net. They matter because they shape the **self-play DATA DISTRIBUTION**, which is
what drives **LEARNING**. So we ablate by measuring **learning trajectory +
self-play data quality** over a few **short BC-seeded RL runs**, not by ranking a
fixed checkpoint's search strength.

Consequence for methodology:
- `c_puct`, `root_policy_temperature`, widening **do** affect the deterministic
  (greedy, no-noise) eval search → they show up in the per-epoch learning eval.
- Dirichlet `alpha`/`epsilon` and the temperature schedule are **self-play-only**
  (eval is greedy + noiseless) → they have **zero** effect on deterministic
  head-to-head and **cannot** be ranked by eval win rate. They are *derived*
  (alpha) / *chosen* (epsilon, temp) and validated only through the self-play
  data-quality metrics + the learning deltas across the short runs.

## 1. Fixed reference for per-epoch eval

The **frozen BC seed** `runs/hexgt_bc/hexgt_bc_step006009.pt`. Every config's
per-epoch eval pits the *current* checkpoint against this *frozen* net,
deterministic (greedy, no noise, fixed per-game seeds), matched visits. Win-rate
>50% and rising ⇒ that config's RL is improving over the seed; the **slope +
stability** of the trajectory is the cross-config comparison signal. (Secondary,
optional external anchor: SealBot best-50ms, for absolute scale — the BC seed is
currently 0% vs SealBot, so any SealBot wins are a strong signal. Primary remains
the frozen seed.)

## 2. Measured candidate-count distribution (drives the derived alpha)

Measured at `candidate_radius=3` over **1200 recorded 96x8 positions** (read-only),
via the exact shared Rust builder self-play uses:

| stat | candidates |
|---|---|
| p05 | 81 |
| p25 | 166 |
| **median (p50)** | **220** |
| p75 | 291 |
| p95 | 467 |
| p99 | 674 |
| mean | 239 |

By phase (median / p95): **opening 141 / 237**, **midgame 241 / 396**,
**endgame 397 / 704**.

### Derived Dirichlet `total_alpha`

KataGo/dense_cnn parameterize noise as `alpha_i = total_alpha / count` (per
position, so it auto-adapts to the local candidate count). The AlphaZero heuristic
sets per-move `alpha_i ≈ 10 / (typical moves)` ≈ **0.03** for Go-scale branching
(~250 moves) — and hexgt's median 220 candidates *is* Go-scale.

- Target `alpha_i = 0.03` at median 220 ⇒ **`total_alpha = 6.6`**.
- dense_cnn's inherited `total_alpha = 10.83` would give hexgt `alpha_i = 0.049` —
  ~1.6× flatter (more-uniform) noise than the AlphaZero heuristic, because hexgt's
  candidate set is smaller. **This is exactly why copying 10.83 is wrong.**
- With `alpha_i = total_alpha/count`, `total_alpha=6.6` auto-adapts per phase:
  opening 6.6/141 = **0.047**, midgame 0.027, endgame 6.6/397 = **0.017** — a
  sensible spiky-in-opening / smooth-in-endgame profile.

**Derived baseline: `total_alpha = 6.6`, `epsilon = 0.25`** (standard mixing
fraction). The ablation tests ±exploration around this.

## 3. Configs (4 — justified, not a grid)

Shared across all: `candidate_radius=3`, `forced_playout_k=2` (anti
opening-collapse, validated on dense_cnn target runs), nucleus `widening_policy_mass=0.95`,
`widening_max_children=32`. They differ **only** in the exploration knobs:

| Config | `total_alpha` (α_i@220) | `eps` | `root_policy_temp` | `c_puct` | temperature schedule |
|---|---|---|---|---|---|
| **C1 derived-baseline** | 6.6 (0.030) | 0.25 | 1.0 | 1.5 | 1.0→0.2 over 30 plies, floor 0.1 |
| **C2 higher-exploration** | 9.0 (0.041) | 0.35 | 1.0 | 2.0 | 1.2→0.3 over 45 plies, floor 0.15 |
| **C3 lower-exploration** | 4.5 (0.020) | 0.15 | 1.0 | 1.0 | 1.0→0.1 over 20 plies, floor 0.05 |
| **C4 prior-flatten** | 6.6 (0.030) | 0.25 | **1.15** | 1.5 | 1.0→0.2 over 30 plies, floor 0.1 |

Rationale:
- **C1 derived-baseline** — `total_alpha` derived above; `root_policy_temp=1.0`
  (no prior flattening) because hexgt's per-candidate pointer policy over the
  *pruned* n=3 set is plausibly sharper than dense_cnn's radius-8 policy, which was
  flattened with 1.1 specifically to fix a ~5×-too-diffuse prior — a defect hexgt
  may not share. Inherits `c_puct=1.5` + the temperature schedule as the neutral
  anchor.
- **C2 higher-exploration** — more + flatter noise, hotter/longer temperature,
  higher `c_puct` (broader tree exploration). Hypothesis: more diverse data →
  faster/more-robust learning; risk: noisier targets, dragging games.
- **C3 lower-exploration** — less + spikier noise, sharper/shorter temperature,
  lower `c_puct` (exploit the prior). Hypothesis: crisper targets → faster early
  learning; risk: premature convergence, low diversity, mode collapse.
- **C4 prior-flatten** — C1 with `root_policy_temp=1.15`. Isolates the single
  "does hexgt need dense_cnn's prior flattening?" question — broader search prior
  vs C1's sharp prior, holding noise/temperature fixed.

These four span the explore↔exploit axis (C3 ← C1 → C2) plus the orthogonal
prior-flattening lever (C4), which is the minimum to *learn the shape* of the
response without a full grid.

## 4. Metrics

### Learning (primary)
- **L1 — per-epoch improvement vs frozen BC seed.** Each epoch: parallel
  deterministic head-to-head (current ckpt vs frozen seed), **~60 games**, matched
  `visits=96`, fixed seeds, alternating colors. Report the win-rate **trajectory**
  (does it rise, how fast, is it stable or does it diverge?).
- **L2 — fixed-holdout loss.** Each epoch's checkpoint scored on a **fixed**
  held-out set (96x8 `epoch_000022` shards, never trained on — the BC held-out
  set): policy CE, value loss, top-1 visit agreement. Trust this over raw epoch
  train-loss (the dense_cnn "rising-loss-is-an-artifact" lesson).

### Self-play data quality (what the constants actually control)
- **Q1 — decisiveness:** fraction of games reaching a real terminal vs
  `max_actions` truncation; game-length median/p95. (Over-exploration ⇒ dragging
  / non-terminating games.)
- **Q2 — opening/move diversity:** Shannon entropy of the first-10 move choices
  across the epoch's games + count of unique first-6-ply openings. (Healthy
  exploration = diverse but not random.)
- **Q3 — policy-target sharpening:** mean MCTS visit-distribution entropy per
  epoch (already instrumented) — do targets crisp up as the model learns?
- **Q4 — value-target balance:** win/loss/draw fractions + mean |value| of
  self-play outcomes (degeneracy/skew check).
- **Q5 — tactical incidence:** rate of active-window / forcing positions
  (immediate-win / must-block / forcing-threat) arising in generated games — proxy
  for whether the data contains learnable tactics.

## 5. Per-run budget + concurrency

Short runs, each ≈ 40–50 min:

| knob | value | note |
|---|---|---|
| games/epoch | 48 | |
| epochs | 5 | enough to see a slope |
| search visits | 96 | (RL self-play; eval L1 matched at 96) |
| train-steps/epoch | 200 | |
| batch | 128 | |
| **self-play concurrency (`active_games`)** | **64** | **the throughput lever** |
| `vbatch` | 64 | leaf-batch width |
| compile | on | warmup amortized over the run |

**Concurrency note (critical):** an earlier probe showed **8 concurrent games =
1.5 pos/s** (GPU-starved) vs **64 games + vbatch 64 + compile = ~13 pos/s** (≈19
steady minus the one-time ~90 s compile warmup). All short runs use **active=64**
so they are *not* crippled. (With games/epoch=48 < active=64, all 48 games run
concurrently — full batches, no drain tail.)

Time per epoch ≈ self-play ~5–6 min + train ~1 min + **L1 parallel eval ~1.5 min**
(vs ~9 min sequential — the parallel-eval prerequisite is what makes this
affordable) + L2 holdout ~0.5 min ≈ **~9 min/epoch** → **~45 min/config** →
**~3 hours for all 4 configs** sequentially on the one GPU. The 32-thread box +
GPU stay saturated via parallel self-play (Rust featurization across threads) +
parallel eval.

## 6. Sequencing + budget tradeoff

1. **Prerequisite harness** (cheap, needed regardless; build on approval):
   (a) **parallel eval** — many concurrent games through the leaf batcher,
   deterministic per-game (fixed seed + greedy ⇒ result invariant to async batch
   composition); **verify it reproduces the sequential result** on a small set.
   (b) **self-play data-quality instrumentation** (Q1–Q5) in the self-play summary.
   (c) hexgt-OWN exploration config fields (already present in `config.py`'s
   `[selfplay]`; expose `total_alpha`/`eps`/`root_policy_temperature` per-config).
2. Run **C1** first → reference trajectory + end-to-end harness validation.
3. Run **C2, C3, C4**.
4. Compare L1 slopes + L2 + Q1–Q5; promote the config whose *data* yields the
   best, most-stable learning to the full RL run.

**Time/info tradeoff:** ~3 h of GPU buys a 4-point map of the explore/exploit +
prior-flattening response on the metric that matters (learning), instead of
guessing or running one long under-tuned RL run. Cheaper alternative if 3 h is too
much: drop C4 and run C1/C2/C3 (~2.2 h); richer alternative: add a 5th midpoint
config (~3.7 h). Recommended: the 4 above.

## 7. Starting-point anchors (already measured, this session)

BC seed step-6009, pre-RL, at the run's eval config (24 games, `visits=200`,
`max_actions=1024`, fixed seed, deterministic):
- **vs dense_cnn e24: 11W / 13L / 0D = 45.8%** (consistent with the documented
  55% @ 40 games within the 24-game CI).
- **vs SealBot best-50ms: 0W / 24L / 0D = 0.0%** (expected — dense_cnn only beat
  SealBot after RL; this is the strongest improvement signal to watch).
- Self-play throughput: ~13 pos/s @ 64 games + vbatch 64 + compile.
- Play-style: the GNN's raw policy **prior is diffuse** (entropy ~3.3 over ~190
  candidates) but **MCTS sharpens it hard** (visit entropy ~2.07, top-move ~0.38;
  individual moves collapse prior_H 3.6 → visit_H 0.3). Whether RL sharpens the
  *prior itself* is tracked by Q3 + the L2 policy CE.

> The in-flight 20-epoch run that produced these anchors used **inherited**
> dense_cnn exploration constants (`total_alpha=10.83`, `root_policy_temp=1.1`) —
> NOT the derived C1 baseline — and was stopped before any RL epoch completed, so
> there is no reusable config-#1 RL data; C1 will be run fresh with the derived
> settings. The baseline-eval anchor above is preserved.
