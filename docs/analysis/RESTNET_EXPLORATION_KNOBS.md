# ResTNet self-play — exploration / search-knob investigation

**Run:** `dense_cnn_restnet_main1` (live). ResTNet 96ch × 8-block `R_R_R_T_R_R_T_R`
(~1.51M params), `attention_scope=disk`, fp16 inference, continuous scheduler,
`search_visits=512`, `games_per_epoch=256`, `active_games=128`, init from HF
human-corpus prefit, TSS active. Analysis at **epoch 30** (latest complete
checkpoint), with self-play samples from epochs 27–31.

**Scope:** exploration / search knobs only. Move-selection *temperature* is treated
as out of scope per owner (believed fine) and is used only as context. Note that
`root_policy_temperature` is a *search* knob (it reshapes the prior before search),
not move-selection temperature, so it **is** in scope.

**Bottom line:** the run is **healthily-to-generously explored, not starved.** The
two things the owner specifically worried about are both fine — `forced_playout_k`
matches KataGo exactly, and `root_policy_temperature`+Dirichlet together is KataGo's
*intended* design (not a double-count). The single most important empirical result
is that **`root_dirichlet_total_alpha` is a dead lever at this game's branching
factor** — sweeping it 6 → 21 changes realized root-prior flattening by < 0.001.
The exploration improvements worth making are about the *shape* and *placement* of
noise, not its magnitude.

> **Status: analysis only. No run/config/code was modified.** All probes were
> CPU-only and read-only; the live GPU training was not disturbed. Recommendations
> below are proposals, not applied changes.

---

## How this was measured

- **Stage 1 — realized exploration** (`scripts/_explore_stage1.py`): parsed every
  per-game `.npz` shard (the pruned visit-policy training target, legal sets, game
  outcome) plus `.json` sidecars for epochs 27–31. Per ply-band: visit-policy
  entropy, top-1 mass, effective move count `exp(H)`, support size, legal-move
  count, played-vs-top-visit deviation, opening diversity, game length.
- **Stage 2 — per-knob analysis** (`scripts/_explore_stage2.py`): loaded
  `epoch_000030.pt` **on CPU**, forwarded ~3,140 real self-play positions to get the
  raw net prior, then replicated the **exact** Rust math (`mcts_tree.rs` /
  `mcts.rs`) for Dirichlet noise, root-policy-temperature, policy-nucleus widening,
  and KataGo forced playouts. This also recovered the prior↔search agreement that
  the v4 shard intentionally drops (`root_prior_policy=()`).
- **Stage 3 — KataGo comparison**: point-by-point against the KataGo paper
  (Wu 2019, arXiv:1902.10565), `KataGoMethods.md`, and the self-play config
  `selfplay8b20.cfg` (its self-play values, not the C++ struct defaults).
- **Code reading**: the noise / forced-playout / widening / PUCT implementations
  in `packages/hexo_models/dense_cnn/rust/src/{mcts.rs,mcts_tree.rs}`.

---

## Stage 1 — realized exploration (epoch 30, latest complete)

| Ply band | visit-H (nats) | top-1 mass | eff. moves | support | legal moves | played≠top |
|---|---|---|---|---|---|---|
| 00–09 | 1.42 | 0.58 | 9.5 | 17.8 | 337 | 0.42 |
| 10–19 | 0.88 | 0.72 | 3.9 | 9.8 | 451 | 0.22 |
| 20–39 | 0.89 | 0.73 | 4.5 | 12.2 | 519 | 0.14 |
| 40–69 | 0.78 | 0.77 | 3.7 | 12.1 | 595 | 0.06 |
| 70+   | 0.96 | 0.73 | 5.2 | 18.6 | 777 | 0.03 |

- **Openings are richly diverse, not under-explored.** 78–106 *unique* 2nd moves per
  256 games (entropy ≈ 4.3 nats ≈ 78 effective distinct lines), ~200 unique 3-move
  opening sequences out of 256 games. First move is always center (canonical).
- **Two structural facts that reshape the whole analysis:**
  1. **Branching is large and *grows* over the game (337 → 777 legal moves).** The
     disk crop expands as stones spread, so Hexo's branching factor *exceeds* Go's
     361 throughout. This is the opposite of the usual "branching shrinks as the
     board fills" intuition, and it is what makes flat Dirichlet noise inefficient
     (see Stage 2).
  2. **Every game is decisive (`|value| = 1.0`, 100%).** Connection game → no draws,
     so "decisiveness trend" carries no exploration signal here.
- **Search visits concentrate on only ~10–18 moves** out of 300–800 legal. So
  `widening_max_children=96` is **not binding on realized visits** in most positions
  (confirmed by epoch-30 diagnostics: `tree_max_active_edges_per_n=96`,
  `tree_widened_edges_total=0`).
- **Played-move deviation tracks the temperature half-life** (0.42 opening → 0.03
  endgame), i.e. it mostly reflects move-selection temperature (out of scope), not
  search breadth.
- **`policy_surprise_mean` is trending down** across epochs (1.48 → 1.36 → 1.21 →
  1.13 for epochs 28→31): the net is increasingly predicting its own search target.
  Healthy convergence; no sign of exploration collapse or stagnation.

---

## Stage 2 — per-knob analysis on the live checkpoint (CPU, ~3,140 real positions)

### Prior vs. search — search is doing real work

| Ply band | raw-prior top-1 | search top-1 | top-1 agree | KL(visit‖prior) |
|---|---|---|---|---|
| 00–09 | 0.371 | 0.594 | 0.43 | 0.91 |
| 10–19 | 0.421 | 0.714 | 0.46 | 1.07 |
| 20–39 | 0.432 | 0.729 | 0.51 | 1.16 |
| 40–69 | 0.435 | 0.755 | 0.48 | 1.30 |
| 70+   | 0.405 | 0.733 | 0.50 | 1.48 |

- The **raw net prior is flat** (top move only 37–44%) and **search sharpens it
  substantially** (top move 59–76%). Crucially, search's top move **differs from the
  prior's top move ~50–58% of the time** (agreement ≈ 0.42–0.51), with KL ≈ 0.9–1.5
  nats. Search is *not* rubber-stamping the prior — if exploration were too low,
  agreement would be near 1.0 and KL near 0. This is direct evidence that the search
  knobs are exploring enough to override a still-immature prior.

### Dirichlet noise — `total_alpha` is a dead lever; only `eps` and *shape* matter

Mean noised-prior top-1 mass (lower = more flattening), averaged over 24 seeds ×
real positions:

| band | α=6,eps.25 | **α=10.83,eps.25** | α=16,eps.25 | α=21,eps.25 | α=10.83,eps.15 | α=10.83,eps.35 |
|---|---|---|---|---|---|---|
| 20–39 | 0.3248 | **0.3246** | 0.3244 | 0.3245 | 0.3674 | 0.2819 |
| 70+   | 0.3042 | **0.3039** | 0.3038 | 0.3038 | 0.3440 | 0.2641 |

- **Sweeping `total_alpha` from 6 to 21 (3.5×) moves realized flattening by < 0.001.**
  Reason: with N=300–800 legal moves, per-child α = `total_alpha/N` ≈ 0.008–0.07 —
  always deep in the sparse-Dirichlet regime — so the *normalized* draw's effect on
  the top move is governed almost entirely by the mix fraction `eps`, not by the
  concentration. **Tuning `total_alpha` will produce no measurable change.** The
  owner's instinct to sweep it would (correctly) find nothing.
- **`eps` is the only magnitude lever that does anything**: 0.15 → 0.25 → 0.35 maps
  to top-1 ≈ 0.367 → 0.325 → 0.282. Current `eps=0.25` matches AlphaZero/KataGo and
  is a sensible middle. No change recommended on its own.
- **The flat noise is unfocused.** The Dirichlet draw spreads its mass over
  **~18 effective (essentially random) cells** out of 300–800 legal, *regardless of
  ply* (`noise_eff_moves ≈ 18` in every band). Because the prior is also flat
  (top-1 ≈ 0.4), a non-trivial slice of the 512-visit budget is steered toward
  arbitrary moves rather than plausible ones. This is the real inefficiency, and it
  is *worse* at Hexo's branching than at Go's — which is exactly the regime KataGo's
  **shaped noise** (50% uniform + 50% concentrated on high-prior moves) was built
  for. The common assumption that "Hexo has fewer moves than Go so flat noise is
  fine" is false here.

### Forced playouts (`forced_playout_k=2.0`) — firing correctly, healthy budget share

| band | forced % of 512-visit budget | children with ≥1 forced visit |
|---|---|---|
| 00–09 | 17.6% | 18.7 |
| 10–19 | 14.5% | 11.0 |
| 20–39 | 15.2% | 15.4 |
| 40–69 | 15.7% | 16.3 |
| 70+   | 17.8% | 27.0 |

- Forced playouts consume a **modest ~15–18% of the visit budget** — they keep root
  noise alive without dominating the tree, exactly as intended. The implementation
  (`prune_forced_delta_counts`) is a **faithful KataGo** policy-target subtraction:
  `n_forced = floor(sqrt(k·prior·root_visits))`, root-only, never prunes the
  most-visited child, PUCT-aware early stop. That ~11–27 children get forced visits
  while the *pruned* training target has only ~10–18 support (Stage 1) confirms the
  pruning is removing the forced visits from the target as designed. **Keep k=2.0**
  (= KataGo `rootDesiredPerChildVisitsCoeff`).

### Widening / policy-nucleus — cap binds only where it least matters

| band | nucleus mean | nucleus median | fraction at the 96 cap |
|---|---|---|---|
| 00–09 | 26.2 | 7 | 2.5% |
| 10–19 | 19.0 | 8 | 8.5% |
| 20–39 | 30.0 | 11 | 20.2% |
| 40–69 | 31.9 | 12 | 23.5% |
| 70+   | 56.8 | 90 | **49.6%** |

- The 95%-mass nucleus cap (`widening_max_children=96`) **binds ~50% of the time in
  the endgame (70+) but only 2.5% at the opening.** Since (a) endgame positions are
  already decided (`|value|=1`) and (b) realized visits concentrate on ~10–18 moves
  regardless, the cap is throttling *low-value endgame breadth* and is essentially
  never the constraint where exploration matters (the opening). **Raising
  `widening_max_children` would add endgame compute for negligible strength.** Leave
  it at 96.
- **Code check (corrects a Stage-3 worry):** Dirichlet noise is applied in
  `apply_root_dirichlet_noise` over **all** root candidates (materialized edges *and*
  the unexpanded prior list) *before* widening selects, so a noise-boosted tail move
  can still be materialized. The nucleus does **not** silently truncate the noise to
  the un-noised top-p set; it only caps the *count* of children ever materialized.

### Root policy temperature — current 1.1 is mild; KataGo also ramps the opening

Raw-prior top-1 under `prior^(1/T)`:

| band | T=1.0 | **T=1.1 (current)** | T=1.25 |
|---|---|---|---|
| 00–09 | 0.371 | 0.345 | 0.306 |
| 20–39 | 0.432 | 0.392 | 0.336 |
| 70+   | 0.405 | 0.356 | 0.288 |

- `T=1.1` flattens the prior ~9% (relative top-1); `T=1.25` ~22%. KataGo uses **1.25
  early decaying to 1.1**; we use a constant 1.1, i.e. we under-soften the opening
  vs KataGo. But the opening is already the *most*-explored band, so the marginal
  value of adding the early ramp is small. Low priority, KataGo-aligned, cheap.

### c_puct / FPU

- `c_puct=1.5` is **higher** than KataGo's self-play `1.1` → we explore *more* per
  move, not less. Combined with the healthy prior→search disagreement, this confirms
  the run is not under-exploring on this axis.
- **`fpu_reduction=0.20` is applied at the root too.** KataGo **zeroes root FPU when
  Dirichlet noise is on** (`rootFpuReductionMax=0`) so all root children look
  attractive and the noise isn't fought by FPU suppression. We partially fight our
  own noise here; forced playouts paper over it, but zeroing root FPU is the
  KataGo-correct, cheap, low-risk change.

---

## Stage 3 — point-by-point vs. KataGo

| # | Knob | Ours | KataGo (self-play cfg) | Verdict |
|---|------|------|------------------------|---------|
| 1 | Forced-playout k + policy-target pruning | k=2.0, root-only, subtract-back | `rootDesiredPerChildVisitsCoeff=2`, identical pruning | **MATCH** |
| 2 | Dirichlet α / eps | flat, α=10.83/legal, eps=0.25 | **shaped 50/50**, α=10.83/legal, eps=0.25 | **MINOR** (flat vs shaped) |
| 3 | Root policy temp | const 1.1 | 1.25 early → 1.1 | **MINOR** (no early ramp) |
| 4 | Widening / nucleus | top-p 95%, clamp [2,96] | none (lazy expand + FPU) | **MINOR** (cap binds only in endgame) |
| 5 | c_puct / FPU | 1.5; fpu 0.20 incl. root | 1.1; fpu 0.2, **root fpu = 0** under noise | **MINOR** |
| 6 | Playout-cap randomization (PCR) | none | p≈0.25, record only full-search turns | **MAJOR (missing)** |

- α=10.83 = KataGo's `0.03 × 361` total Go concentration, divided by legal count —
  exactly KataGo's framing. Forced-playout k and the temp→noise ordering match.
- The only genuine deviations worth attention: **shaped noise (2), root FPU under
  noise (5), and PCR (6).** PCR is the highest-leverage item but is an
  efficiency/value-target change as much as an exploration knob (see below).

---

## Stage 4 — ranked recommendations

Ranked by expected value, with the measured evidence, predicted effect, the
tradeoff, and confidence. **All are proposals; none have been applied.**

### 1. Switch to KataGo *shaped* Dirichlet noise (50% uniform + 50% prior-concentrated)
- **Evidence:** flat noise spreads over ~18 *random* cells out of 300–800 (Stage 2,
  `noise_eff_moves≈18` in every band) while the prior is flat (top-1≈0.4), so a real
  slice of the 15–18% forced-playout budget is spent on arbitrary moves. Branching
  here *exceeds* Go's 361, the exact regime shaped noise targets.
- **Predicted effect:** the same exploration budget is focused on *plausible* moves
  → less wasted search on junk, modestly better self-play move quality and
  value-target signal, especially mid/late game.
- **Tradeoff:** slightly less "wild" exploration of long-shot moves; small
  implementation change in the noise routine.
- **Confidence: medium.** It's a documented KataGo refinement and the inefficiency
  is directly measured, but I did not A/B the strength delta on this run.

### 2. Zero the root FPU reduction while Dirichlet noise is on (`rootFpuReductionMax=0`)
- **Evidence:** we apply `fpu_reduction=0.20` at the root; KataGo zeroes it under
  noise so unvisited (incl. noise-boosted) root children aren't suppressed below
  `parent_value−0.20` before they're tried.
- **Predicted effect:** root noise/forced playouts act without FPU fighting them;
  marginally broader, KataGo-correct root exploration.
- **Tradeoff:** very low risk; near-zero compute cost. Note it overlaps with forced
  playouts (which already force those children), so the realized effect may be
  small — but it's the correct default.
- **Confidence: medium-low** on magnitude, **high** on correctness/safety.

### 3. (Adjacent, highest *throughput* leverage) Add playout-cap randomization (PCR)
- **Evidence:** missing vs KataGo; memory note records the evaluator at ~84% of
  self-play wall time with avg GPU batch ~54. PCR runs most moves at a small visit
  cap (noise off, not recorded) and only a fraction at full 512 visits (noise +
  forced playouts, recorded).
- **Predicted effect:** large self-play throughput gain *and* higher-quality value
  targets (recorded positions all get full searches). More/better data per GPU-hour.
- **Tradeoff:** this is a **structural self-play change**, not a one-line knob, and
  is more about efficiency/value-quality than exploration per se — flagged here
  because it's the biggest available win and interacts with the noise/forced-playout
  machinery. Recommend scoping it as its own change, not bundled with knob tweaks.
- **Confidence: high** that it helps throughput; **medium** on net strength-per-hour
  without a trial.

### 4. (Optional, KataGo-aligned) Ramp `root_policy_temperature` 1.25 → 1.1 early-game
- **Evidence:** const 1.1 under-softens the opening vs KataGo's 1.25→1.1; T=1.25
  flattens prior top-1 ~22% vs ~9% at 1.1 (Stage 2).
- **Predicted effect:** slightly broader opening book. **But openings are already the
  most diverse band**, so marginal.
- **Tradeoff:** cheap; low expected upside. **Confidence: low** it moves strength.

### 5. (Test-only) Consider `c_puct` 1.5 → ~1.1 to match KataGo
- **Evidence:** ours (1.5) is 36% above KataGo self-play (1.1); search already
  disagrees with the prior ~50% and sharpens well, so we're not exploration-starved.
- **Predicted effect:** *less* exploration / more focused search at 512 visits —
  could improve move quality, or could reduce useful breadth. Genuinely ambiguous.
- **Tradeoff:** **A/B only, do not blind-change.** 1.5 may have been tuned for this
  value-net's scale. **Confidence: low.**

### Do NOT bother changing
- **`root_dirichlet_total_alpha` (10.83):** measured dead lever (6→21 ≈ no effect).
  Leave it; it equals KataGo's canonical value anyway. *(High confidence.)*
- **`forced_playout_k=2.0`:** matches KataGo exactly, healthy 15–18% budget share,
  pruning verified. Keep.
- **`widening_max_children=96` / `widening_policy_mass=0.95`:** cap binds only in the
  already-decided endgame; raising it buys endgame compute for negligible strength.
- **`eps=0.25`, `virtual_loss=1.0`:** standard, sensible.
- **policy-surprise reweighting (`uniform_fraction=0.5`, `max_weight=8.0`):** active
  and behaving — per-game `frequency_weight_mean≈1.13`, row-drop ≈ 0, and
  `policy_surprise_mean` trending down across epochs (1.48→1.13). Working as intended.

---

## Cross-checks & honest caveats

- **SealBot eval:** the run produces eval games vs SealBot as raw `.hxr` records per
  3-epoch checkpoint, but **no aggregate win-rate is persisted** in `diagnostics/` or
  the logs, so I could not plot an eval trend without parsing the binary records.
  The available learning-health signals — `policy_surprise_mean` falling 1.48→1.13
  and the strong, stable prior→search sharpening (KL≈1, agreement≈0.5) — are
  consistent with a net that is still improving and a search that is contributing
  real signal. If an eval win-rate cross-check is wanted, the next step is to parse
  `evaluation/epoch_*/eval-*.hxr` for outcomes.
- **Value calibration** was not computed (would need an extra forward pass scoring
  predicted value vs. the ±1 outcome); the exploration findings stand independently
  of it.
- **Method limits:** Stage 2 replicates the search math *analytically* on the raw
  prior rather than running a full 512-visit CPU tree (to avoid disturbing the live
  GPU run). Forced-playout budget share and nucleus counts are therefore
  prior-derived estimates, corroborated by the live epoch-30 tree diagnostics
  (`max_active_edges_per_n=96`, `widened_edges_total=0`) and Stage-1 realized support
  (~10–18). A full CPU search probe would tighten the forced-playout/widening numbers
  but is unlikely to change any recommendation.
- Played-move deviation in Stage 1 is influenced by move-selection temperature
  (out of scope) and is reported as context, not as a search-exploration metric.

*Probes: `scripts/_explore_stage1.py`, `scripts/_explore_stage2.py`,
`scripts/_explore_run.sh` (CPU wrapper). Checkpoint: `epoch_000030.pt`. KataGo refs:
arXiv:1902.10565; `lightvector/KataGo` `docs/KataGoMethods.md`,
`cpp/configs/training/selfplay8b20.cfg`, `cpp/search/searchexplorehelpers.cpp`.*

---

## Postscript — config change applied live (2026-06-10, owner directive)

Two owner-directed changes were applied to `dense_cnn_restnet_main1` via a clean
boundary bounce (halt → relaunch resuming from `epoch_000031.pt`, throughput knobs
`scheduler=lockstep` / `torch.compile` / `attention_kv_gather` preserved):

1. **Self-play opening-temperature anchor** (new code hook): `opening_temperature`
   /`opening_moves` added to `Model1SelfPlayConfig` + parser, and a floor in
   `selfplay._move_temperature` (`max(opening_temperature, adaptive_base)` for the
   first `opening_moves` decisions). Set to **`opening_temperature=1.4`,
   `opening_moves=8`**. Resulting opening curve at expected length ~97 (half-life
   ~24 plies): **plies 0–7 held flat at 1.4**, then the existing adaptive decay
   resumes (~0.79 @ ply 8 → 0.50 @ ply 24 → 0.10 floor). Chosen per the
   opening-diversity analysis (recommended 1.3–1.5 over plies 0–6; the adaptive
   scheme alone left the opening at only ~0.8–1.0, too low to diversify).
2. **`train_samples_per_epoch` 32000 → 64000.**

**Git: held uncommitted** (option A) — the commit clone is stale relative to the
live run (origin lacks the throughput knobs, the adaptive-temperature block, and
the 32000 setting; `selfplay.py`/`config.py` are +391/+34 lines ahead uncommitted),
so a clean non-regressive commit isn't possible without the parallel session's
coordination. The edits live in the `E:\Hexo-BotTrainer-hexgt` worktree like the
rest of the run's uncommitted state. This doc is the record.

**Honest caveat (opening diversity).** The opening collapse ("origin + n=8-edge")
is **prior-driven** — the net's raw policy puts ~96% mass on the edge ring (argmax
dist-8 in 140/140 probed boards). A *move-selection* temperature anchor acts
**downstream of search**: it spreads selection across the moves the search already
visited (mostly edge-ring), so it will increase **direction / which-edge-cell**
diversity and give the occasional inner move more weight — but it is **unlikely to
break the radius/shape stereotypy on its own**, because temperature cannot make the
search visit moves the concentrated prior never surfaces. The higher-leverage levers
for genuine opening-*shape* diversity act **before/inside search**: raise
`root_policy_temperature` at the opening (flattens the prior pre-search so search
actually explores non-edge moves), shaped/boosted root Dirichlet (see
recommendation 1 above; the opening-diversity note suggests opening-only `eps→0.35`),
or ultimately the prior/training side (the 96%-edge prior is learned and only shifts
as the value signal teaches the net that inner openings are viable). **Assessment:
1.4 is a reasonable, low-risk first touch that will measurably diversify the played
opening, but treat it as necessary-not-sufficient — if opening *shape* doesn't
broaden within a few epochs, the fix is on the prior/search side, not temperature.**
