# hexgnn_rl_main1 — Exploration (Dirichlet + temperature) analysis

**Question (owner):** is the live `hexgnn_rl_main1` run doing *too much* exploration?

**Verdict (one line):** **About right, leaning high on the *mid/late-game* move
temperature — not harmfully, but with one measured cost and one clean tightening.**
Exploration is not throwing games away (1‑ply value cost of played deviations ≈ 0 in every
ply band; catastrophic single‑move outcome‑flips ≈ 0.02 %). The opening is *maximally*
diverse and essentially free, and the model is still 0–5 % vs SealBot, so the opening
exploration should be **left alone**. The single substantiated concern is **value‑target
hygiene at λ=0**: `corr(v,z)` climbed to a 0.80 peak at epoch 35 then **softened to ~0.66 at
epochs 37–41** as games lengthened — the played‑move temperature, which at λ=0 hard‑z writes
±1 z‑labels and (unlike hexgt main3) is **applied to every move because PCR is off**, is the
prime suspect. The cleanest levers are to **reduce mid/late temperature exposure** (shorten
halflife and/or lower the floor — see the owner caveat in §7) or to **re‑enable PCR**. Leave
Dirichlet (α=6.6, ε=0.30) and the opening temperature as‑is.

**Analysis only.** No config, model, run, or launch script was modified. All compute was
**CPU and read‑only**; the live GPU run was never contended. Another agent concurrently owns
this run's config bounces (notably candidate radius `n`); this touched only read‑only shards,
checkpoints, and logs.

> This doc consolidates two same‑day read‑only passes: the realized‑metrics / Dirichlet‑sweep
> pass and a second pass adding the forced‑playout‑pruning mechanism, controlled native‑MCTS
> attribution at the live 1024 visits, and the per‑epoch value‑calibration trend.

---

## 0. Confirmed live configuration

The config‑driven TOML is **not** read by the live driver (`scripts/_rl_train_hexgnn.py`) —
the live values come from [`scripts/_rl_launch_hexgnn.sh`](../../scripts/_rl_launch_hexgnn.sh)
→ `EXTRA_ARGS`, verified against the running process and the startup banner
(`rl_train.log`, resume at epoch 41; run had reached epoch 43 self‑play by end of analysis).
**These knobs are inherited verbatim from the deliberately‑tuned hexgt `main3` config**
([`scripts/_rl_launch_main3.sh`](../../scripts/_rl_launch_main3.sh)).

| knob | live value | channel |
|---|---|---|
| candidate radius `n` | **4** (2 ep0–29, 5 ep30–41, 4 from ep42) | search support |
| search visits / active | **1024 / 256** (512 ep0–28) | search |
| Dirichlet `total_alpha` | **6.6** (root only) | search → **policy target** + played move |
| Dirichlet `eps` | **0.30** | " |
| `root_policy_temperature` | 1.0 | search |
| `c_puct` | 1.5 | search |
| move `temperature` | start **1.0**, floor **0.3**, halflife **33** | **played move only** (z‑labels) |
| `forced_playout_k` | 2.0 | search; **pruned out of the policy target** |
| widening | mass 0.95, max **96**, min 2 | search breadth |
| `soft_z_lambda` | **0.0 (hard‑z)** | value targets — **undamped** ±1 |
| PCR | **OFF** | (hexgt main3 had it **ON**) |
| policy‑surprise | ON | row‑duplication by KL(visits‖prior) |
| params | 200,139 (td96 / gnn2 / heads4 / pma2 / steerable4), from scratch | |

Played‑move temperature curve (`_move_temperature`, halflife branch):
`temp(ply) = 0.3 + 0.7·2^(−ply/33)`

| ply | 0 | 10 | 20 | 30 | 40 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|
| temp | 1.00 | 0.87 | 0.76 | 0.67 | 0.60 | 0.55 | 0.46 | 0.39 |

At the **median game length (~43 plies)** the played temperature is still **≈0.59**; it only
asymptotes to the 0.3 floor — **it never reaches greedy/argmax.**

**Differences vs hexgt main3:** n=4 (was 3), visits 1024 (was 512+PCR), **PCR removed**,
smaller model, from scratch. The exploration knobs are identical; **PCR removal is the change
that most increases temperature/z‑label exposure** — main3 played a fraction of moves greedily
and unrecorded; hexgnn temperature‑samples and records *every* move.

---

## 1. Method

CPU‑only (`CUDA_VISIBLE_DEVICES=""`), read‑only, on the WSL build venv, the **live e41
checkpoint** (driver schema `model`+`arch`; 200,139 params; loaded `strict=True`, no
missing/unexpected) as the value/policy oracle.

- **Phase 1** — realized metrics + 1‑ply value lookahead over recorded self‑play positions
  (visit policy from compact shards, played move from the per‑epoch `.hxr`). Reported at 240
  games / 7,360 positions (epoch 41) here; a wider 17,750‑position random sample across epochs
  36/38/40/41 reproduces the same numbers.
- **Phase 2** — attribution. Temperature is a **post‑hoc transform** on the stored visit
  policy (so swept analytically); **Dirichlet ε requires re‑searching**, so it is swept on the
  **actual native MCTS** (`HexgnnMctsSession.run`) over fixed real positions with fixed seed.
- **Phase 3** — full‑population per‑epoch diagnostics + per‑epoch value calibration
  recomputed by evaluating each epoch's own model on its own positions.

**Caveats carried throughout:** the e41 value head is moderately calibrated
(`corr(v,z)` ≈ 0.25 / 0.48 / 0.37 by band, ~0.57 overall), so per‑move ΔV is indicative, not
precise — the verdict does not rest on it. `.hxr` records are stored in *finish* order, so a
head‑slice skews toward shorter games; band means over thousands of positions are stable and
cross‑epoch trends use a constant skew, but absolute late‑band counts are under‑sampled.

---

## 2. The two‑channel mechanism (verified in Rust)

From [`packages/hexgnn/rust/src/mcts.rs`](../../packages/hexgnn/rust/src/mcts.rs):

- **Dirichlet noise** is mixed into the **root prior** used by PUCT → it perturbs the *visit
  distribution*, hence **both** the recorded policy target **and** the played move.
- **Forced playouts** add visits during search but are **pruned back out of the exported
  policy target** (`pruned_visit_policy`, `mcts.rs:789`, KataGo‑style) → they do **not** flatten
  the training target.
- **Move temperature** is applied **only at action selection**; the recorded policy weights are
  the raw (forced‑pruned) visit fractions. **Temperature never touches the policy target** — it
  changes only which move is *played*, i.e. the trajectory and the ±1 z‑labels.

| channel | driven by | what it corrupts at λ=0 |
|---|---|---|
| **policy‑target** spread/sharpness | Dirichlet ε, widening | nothing toxic — a slightly flatter visit target |
| **played‑move / z‑label** | **temperature** (+Dirichlet) | the ±1 outcome label *iff* a deviation flips the result |

This is exactly the separation the owner's curve comment was designed around ("separate policy
targets/search — no z‑noise at λ=0 — from played‑move temperature, which writes ±1 outcomes").
§3 confirms it empirically.

---

## 3. Phase 1 — realized exploration level (e41 oracle)

"Deviation" = played ≠ visit‑argmax. "Outcome‑flip" = played move's 1‑ply value sign opposite
to the top move's.

| band | n | temp@band | **played‑dev** | mean ΔV(dev) | P(ΔV>0.1) | P(ΔV>0.3) | search H | top1‑visit | corr(v,z) |
|---|---|---|---|---|---|---|---|---|---|
| 0–20 | 3791 | 0.87 | **54 %** | −0.010 | 0.04 | 0.00 | 1.87 | 0.47 | +0.25 |
| 20–50 | 3237 | 0.68 | **29 %** | −0.012 | 0.04 | 0.01 | 1.00 | 0.67 | +0.48 |
| 50+ | 332 | 0.52 | **22 %** | −0.009 | 0.03 | 0.01 | 0.90 | 0.68 | +0.37 |

- **Value cost of deviations ≈ 0** in every band (mean ΔV marginally *negative* — sampled moves
  are value‑equal to the top‑visit move on the model's own head). Decisive deviations
  (P(ΔV>0.3) ≈ 0). **Catastrophic single‑move outcome‑flips** (top clearly winning V>0.25 →
  played clearly losing V<−0.25): **~0.02 %**. Deviations are between **near‑equal moves** — the
  healthy regime.
- **Deviation is comparable‑to‑lower than the predecessor hexgt** (58/36/35 %), *despite* the
  hotter schedule, because hexgnn's search is **sharper** (top1‑visit 0.47/0.67/0.68 vs hexgt
  0.43/0.39/0.36) — the sharper search absorbs the hotter temperature.

---

## 4. Phase 2 — attribution: noise vs temperature

### 4a. Temperature sweep (post‑hoc over the real visit policies)

| band | T | E[dev] | E[ΔV] | P(Δ>0.1) | P(Δ>0.3) | sampling H |
|---|---|---|---|---|---|---|
| 0–20 | 0.3 / 0.6 / 1.0 | 0.25 / 0.38 / 0.48 | −0.00 / −0.00 / −0.00 | 0.02 / 0.03 / 0.04 | 0.00 | 0.67 / 1.01 / 1.26 |
| 20–50 | 0.3 / 0.6 / 1.0 | 0.13 / 0.23 / 0.31 | −0.00 / −0.00 / −0.01 | 0.02 / 0.03 / 0.04 | 0.00–0.01 | 0.33 / 0.57 / 0.81 |
| 50+ | 0.3 / 0.6 / 1.0 | 0.15 / 0.23 / 0.31 | −0.00 / −0.00 / −0.01 | 0.01 / 0.02 / 0.03 | 0.00–0.01 | 0.34 / 0.56 / 0.77 |

**Temperature is "free" in per‑move value terms** (E[ΔV] ≈ 0 at every T). Its only effect is on
the **deviation rate / selection entropy** — i.e. how stochastic the *played* line is, hence
z‑label variance at λ=0.

### 4b. Dirichlet ε sweep — **live native MCTS, the live 1024 visits**, fixed seed

`H` / `top1` = recorded policy‑target spread / sharpness; `dev` at the live curve temperature.

| band | ε=0.00 (H / top1 / dev) | ε=0.15 | **ε=0.30 (live)** | ε=0.45 |
|---|---|---|---|---|
| 0–20 | 2.22 / 0.39 / 0.55 | 2.29 / 0.38 / 0.60 | 2.33 / **0.37** / 0.60 | 2.34 / 0.36 / 0.60 |
| 20–50 | 0.36 / 0.83 / 0.00 | 0.39 / 0.82 / 0.00 | 0.44 / **0.82** / 0.00 | 0.50 / 0.81 / 0.00 |
| 50+ | 0.68 / 0.73 / 0.25 | 0.85 / 0.70 / 0.25 | 1.01 / **0.67** / 0.25 | 1.13 / 0.63 / 0.38 |

### 4c. Temperature does **not** touch the policy target — empirical proof

Same native MCTS, **Dirichlet held at the live ε=0.30**, only move‑temperature varied:

| band | T=0.0 (H / top1 / dev) | T=0.3 | T=0.6 | T=1.0 |
|---|---|---|---|---|
| 0–20 | 2.33 / 0.37 / **0.00** | 2.33 / 0.37 / 0.20 | 2.33 / 0.37 / 0.55 | 2.33 / 0.37 / **0.65** |
| 20–50 | 0.44 / 0.82 / 0.00 | 0.44 / 0.82 / 0.00 | 0.44 / 0.82 / 0.00 | 0.44 / 0.82 / 0.00 |
| 50+ | 1.01 / 0.67 / 0.00 | 1.01 / 0.67 / 0.25 | 1.01 / 0.67 / 0.25 | 1.01 / 0.67 / 0.38 |

`H` and `top1` are **byte‑constant across T** while opening deviation goes **0 → 65 %** — the
clean demonstration that **temperature drives only the played move / z‑labels**, and Dirichlet
drives the policy‑target spread.

**Attribution conclusion:**
1. **Temperature is the dominant played‑line driver** (opening dev 0→65 % across T) and adds
   **zero** to the policy target — at λ=0 it is the sole knob feeding z‑label variance.
2. **Dirichlet's job is modest policy‑target flattening** at the live ε=0.30 (opening top1
   0.39→0.37, **endgame top1 0.73→0.67**); it changes the searched top move ~27 % of the time
   in the opening (where diversity is most valuable) but only ~6 % mid‑game (never overriding
   the decisive move). **Healthy and well‑targeted** — ε=0.30 is slightly above the 0.25
   baseline but in the healthy band.

---

## 5. Phase 3 — cross‑epoch statistics

### Opening diversity (full population, 512 games/epoch)

| ep | len med / mean | %<30 | uniq m1 / m2 / 3‑move | open‑H (3‑move) | top‑3 share |
|---|---|---|---|---|---|
| 0  | 73 / 89.6 | 1 %  | 1 / 34 / 440 | 6.04 | 2 % |
| 8  | 35 / 41.2 | 28 % | 1 / 34 / 425 | 5.98 | 2 % |
| 20 | 33 / 41.0 | 33 % | 1 / 34 / 344 | 5.71 | 3 % |
| 28 | 35 / 59.7 | 26 % | 1 / 34 / 392 | 5.89 | 2 % |
| 32 | 41 / 65.4 | 20 % | 1 / 81 / 450 | 6.06 | 2 % |
| 36 | 53 / 76.5 | 14 % | 1 / 75 / 425 | 5.98 | 2 % |
| 41 | 43 / 58.4 | 16 % | 1 / 71 / 364 | 5.73 | 5 % |

- **Opening diversity is near‑maximal and stable** — 3‑move openings 344–450 unique of 512
  (67–88 %), entropy ≈ 5.6–6.1 nats, top‑3 lines only 2–5 %. The 6‑ply `opening_entropy` is
  pinned at **6.24 = ln(512)** every epoch (literally *all* 512 games have a unique 6‑ply
  opening). **No opening collapse, ever** — temperature is not *needed* to manufacture opening
  variety; the candidate structure + Dirichlet already saturate it.
- Move‑1 is structurally forced (uniq=1, symmetry). The move‑2 jump 34→70–81 at ep32 is the
  **`n` candidate‑radius increase (2→5→4)**, *not* an exploration‑knob effect.
- **Game length is recovering, not degenerating:** 73 → 33–35 → 43–53; %<30‑ply fell 33→16 %.
  Decisiveness ~100 %, draws ~0 %. The recovery tracks the visits 512→1024 bump (ep29/30) and
  the `n` changes — games getting *longer and more decisive* is the opposite of an
  over‑exploration pathology.

### Value calibration `corr(v,z)` by epoch (model = that epoch, 70 games / ~1,650 positions)

| epoch | 25 | 29 | 33 | **35** | 37 | 39 | 41 |
|---|---|---|---|---|---|---|---|
| corr(v,z) | +0.73 | +0.74 | +0.76 | **+0.80** | +0.61 | +0.69 | +0.66 |
| mean\|v\| | 0.53 | 0.62 | 0.54 | 0.55 | 0.45 | 0.46 | 0.40 |

**This is the one *measured* cost.** Calibration climbed to a 0.80 peak at epoch 35 then
**dropped ~0.13–0.19** at epochs 37–41, and the value head got *less confident* (mean|v|
0.55→0.40). The drop coincides with games lengthening (median 33→43+). The visits 512→1024
bump would *improve* targets (more search), so it does **not** explain a *drop* — the
played‑move/z‑label channel does: at **λ=0 + no PCR**, every move is temperature‑sampled and
recorded, so a longer game accrues more cumulative temperature label‑noise per ±1 outcome.
(Two caveats: 30–70‑game samples carry ±0.05; and the finish‑order skew is constant across
epochs so the *trend* holds. Worth re‑checking at epoch ~48–50.)

### Exploration vs strength

SealBot win rate is flat at **0–5 %** with no trend (SealBot mean_turns rises 24→72 — the
model *survives longer* but still loses), and vs‑dense_cnn is skipped. **SealBot is too strong
to provide any usable per‑epoch correlate** for exploration at this stage. Weakness is
attributable to model size + from‑scratch + early epochs, **not** to the exploration settings.

---

## 6. Comparison to norms and to prior hexgt

| knob | hexgnn live | hexgt main3 (inherited‑from) | hexgt C1 ("about right, leaning high") | AlphaZero | KataGo |
|---|---|---|---|---|---|
| Dirichlet total_alpha | 6.6 (α_i≈0.026 @ ~250 cand) | 6.6 | 6.6 | α≈0.03 (Go) | ~10.8 / avg‑moves |
| Dirichlet eps | **0.30** | 0.30 | 0.25 | 0.25 | 0.25 |
| temperature start | 1.0 | 1.0 | 1.0 | 1.0 (first 30 plies) | ~0.8 |
| temperature decay | exp, halflife 33 | exp, halflife 33 | linear → 0.2 by ply 30 | argmax after ply 30 | exp, hl ~19 |
| temperature floor | **0.3** | 0.3 (owner: "don't lower") | 0.2 | ~0 (argmax) | ~0.2 |
| PCR | **OFF** | **ON** | — | — | yes |
| value target | **λ=0 hard‑z** | λ=0 hard‑z | λ=0.5 soft‑z | hard‑z | soft (utility) |

The Dirichlet mass is calibrated to be **AlphaZero‑equivalent** at the typical candidate count.
The departure from the whole reference lineage is the **persistent 0.30 temperature floor
(never greedy) at λ=0 with no PCR**: AlphaZero plays the endgame *greedily*; KataGo damps the
late temperature and uses soft targets + PCR; hexgt main3 used PCR to keep a fraction of moves
greedy/unrecorded. **hexgnn keeps full temperature exposure on every recorded move, all game** —
the theoretically riskiest corner, and §5 is the first place it shows.

---

## 7. Verdict & recommendation (recommend only — do NOT apply)

**Over‑exploring? Not harmfully — but the *mid/late* move temperature is hotter than is
justified at λ=0, and it now has a measured cost. "About right, leaning high," with one clean
tightening.**

Evidence it is **not** broadly over‑exploring: per‑move value cost ≈ 0; catastrophic flips
≈ 0.02 %; realized deviation ≤ predecessor; opening diversity near‑maximal and stable; game
length recovering, decisiveness ~100 %; Dirichlet rarely overrides the decisive move (~6 %).

Evidence the **mid/late temperature is hotter than warranted**: it is ≈0.59 at the median game
length and never reaches argmax (AZ is greedy by ply 30, KataGo floor ~0.2); its only payoff is
played‑line diversity, which is **already saturated** by candidate structure + Dirichlet; it
runs at **λ=0 with no PCR** (every move recorded), where played‑move stochasticity is the
direct z‑label‑noise driver — and the epoch‑35→41 calibration softening (§5) is consistent with
that noise biting as games lengthen.

It is **not under‑exploring** anywhere.

**Recommendations, in order (each is a config bounce for the owner/config agent to decide):**

1. **Re‑enable PCR.** The most faithful fix and the cleanest revert: hexgt main3 had it,
   hexgnn dropped it. PCR plays a fraction of moves greedily and *does not record* them,
   directly cutting per‑game temperature/z‑label exposure **without touching the opening or
   the floor**. *Predicted:* cleaner long‑game z‑labels → calibration recovers; throughput
   improves (fast moves are cheap). *Tradeoff:* fewer recorded rows per game.
2. **Cool the mid/late temperature** toward the KataGo/C1 profile — shorten
   `temperature_halflife` **33 → ~18–20** (and/or lower `temperature_floor` **0.30 → 0.15–0.20**)
   so played temperature reaches its floor near the median game length instead of
   asymptotically. *Predicted, from §4a:* mid‑band dev 29 % → ~18–22 %, late‑band 22 % → ~10–13 %;
   **opening dev barely moves** (early temp stays ~0.85–0.9, so the diversity asset is preserved);
   per‑move value cost stays ≈ 0 (no self‑play strength loss); cleaner z‑labels + sharper
   policy targets exactly where positions are more forced. **Caveat:** lowering the floor
   conflicts with the owner's standing directive *"Floor kept at 0.3 (owner: don't lower noise
   floor)"* in `_rl_launch_main3.sh`. Surfaced for the owner to weigh; if the floor is to stay,
   shortening the **halflife** alone (or option 1) captures most of the benefit without touching
   it. A single conservative step — **halflife 33 → 22**, floor untouched — is hard to argue
   against.
3. **(Optional, minor) Dirichlet ε 0.30 → 0.25** to match every reference and marginally sharpen
   the endgame policy target (top1 0.67→~0.70). Low priority; not worth a bounce on its own.

**Do NOT** reduce opening temperature, reduce `total_alpha`, or shrink widening — opening
diversity and broad policy targets are assets here, and for a still‑weak from‑scratch model
keeping opening/state coverage high is correct.

---

## 8. Reproducibility

Read‑only CPU scripts under `_expl/` (untracked); JSON outputs under `_expl/out/`
(`phase1.json`, `phase2.json` @1024 visits, `phase3.json`). Model from
`checkpoints/hexgnn_rl_epoch000041.pt`. Phase 1/3 read recorded visit policy from compact
shards + played move from the per‑epoch `.hxr`; Phase 2 drives the actual native
`HexgnnMctsSession.run` with controlled `root_dirichlet_noise_fraction` / `move_temperatures`
/ fixed `seed`. Generated 2026‑06‑06 from `runs/hexgnn_rl_main1` (e41; shards/`.hxr` epochs
25–41; full‑population opening stats epochs 0–41). No run/model/training/launch state modified.
