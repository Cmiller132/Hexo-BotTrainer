# hexgnn_rl_main1 — exploration analysis (Dirichlet noise + temperature)

**Question (owner):** is the live `hexgnn_rl_main1` run doing *too much* exploration?

**Verdict (one line): No measurable harm, but the move-temperature schedule is
hotter than both the proven predecessor (hexgt C1) and AlphaZero/KataGo norms, and
it runs at λ=0 hard-z where played-move stochasticity directly adds z-label
variance. Exploration is "about right, leaning high" — with ONE low-risk tightening
available: cool the temperature floor/halflife toward the C1/KataGo profile. The
Dirichlet noise is healthy and should be left alone (at most a minor eps 0.30→0.25).**

This is **analysis only**. No config, model, run, or launch script was modified.
Recommendations are for the owner to decide. Another agent is concurrently managing
config bounces (notably the candidate radius `n`); this analysis touched only
read-only shards, checkpoints, and logs.

---

## 0. Confirmed live configuration (not the TOML — the driver args)

The config-driven TOML (`configs/hexgnn_model.toml`) is **not** read by the live
driver (`scripts/_rl_train_hexgnn.py`) — same documented trap as the hexgt driver.
The live values come from `scripts/_rl_launch_hexgnn.sh` → `EXTRA_ARGS`, verified
against the running process command line (`ps`) and the most recent startup banner
(`rl_train.log`, resume at epoch 41, 2026-06-06 08:30 local):

| knob | live value | source |
|---|---|---|
| candidate radius `n` | **4** (was 2 ep0–29, 5 ep30–41, 4 from ep42) | ps + startup |
| search visits | **1024** (512 ep0–28) | startup |
| active games | 256 | startup |
| Dirichlet `total_alpha` | **6.6** | ps |
| Dirichlet `eps` (noise fraction) | **0.30** | ps |
| `root_policy_temperature` | 1.0 | ps |
| `c_puct` | 1.5 | ps |
| move `temperature` (start) | **1.0** | ps |
| `temperature_floor` | **0.3** | ps |
| `temperature_halflife` | **33** | ps |
| `forced_playout_k` | 2.0 | ps |
| `widening_max_children` | 96 (`policy_mass` 0.95) | ps |
| `soft_z_lambda` | **0.0 (hard-z)** | ps |
| params | 200,139 (td96 / gnn2 / heads4 / pma2 / steerable4) | startup |

The played-move temperature curve (driver `_move_temperature`, halflife branch):

```
temp(ply) = floor + (init - floor) · 2^(-ply / halflife)
          = 0.3 + 0.7 · 2^(-ply / 33)
```

| ply | 0 | 10 | 20 | 30 | 40 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|
| temp | 1.00 | 0.87 | 0.76 | 0.67 | 0.60 | 0.55 | 0.46 | 0.39 |

At the **median game length (~43 plies)** the played temperature is still **≈0.59**.
It only asymptotes to the 0.3 floor — it never reaches greedy/argmax.

---

## 1. Method

CPU, read-only, on the WSL build venv. Reuses the prior hexgt `_temp_cost.py`
methodology, adapted to hexgnn (`HexgnnInference.evaluate_states`, `n=4`, the live
**epoch-41 checkpoint** as the value/policy oracle). Scripts: `_hexgnn_explore.py`
(realized metrics + temperature sweep + cross-epoch stats) and `_hexgnn_dirichlet.py`
(live CPU MCTS noise sweep). The GPU training run was never contended — all
evaluations and the MCTS sweep ran on CPU.

- **Phase 1 / 2a** — 17,750 recorded self-play positions, randomly sampled across
  epochs 36/38/40/41 (random sampling is essential: `.hxr` records are stored in
  *finish* order, so a head-slice biases toward the shortest games and starves the
  late ply bands).
- **Phase 2b** — 96 fixed real positions (36 opening / 36 mid / 24 late), each
  searched at **512 visits × 4 noise seeds** with the *only* varied knob being
  Dirichlet `eps ∈ {0, 0.15, 0.30, 0.45}` (everything else at live values, greedy
  move selection so we isolate the effect on the *searched* visit distribution =
  the policy target).
- Ply bands: **0–20 / 20–50 / 50+**.

Caveat carried throughout: the e41 value head is a 200k-param model trained
from scratch, only **moderately** calibrated (corr(v,z) = 0.25 / 0.48 / 0.37 by
band). So the per-move "value cost" numbers are indicative, not precise. The
verdict does **not** rest on them — it rests on the comparison to proven baselines,
the λ=0 sensitivity, and the already-saturated opening diversity, none of which
need a precise value head.

---

## 2. Phase 1 — realized exploration level (17,750 positions, e41 oracle)

| band | n | temp@band | **played-dev%** | mean ΔV(dev) | P(ΔV>0.1) | P(ΔV>0.3) | outcome-flip% | search H | temp-sel H | top1-visit | corr(v,z) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–20 | 4982 | 0.87 | **50%** | −0.006 | 0.14 | 0.01 | 0% | 1.76 | 1.60 | 0.50 | +0.25 |
| 20–50 | 5757 | 0.65 | **30%** | −0.005 | 0.12 | 0.03 | 0% | 1.21 | 0.82 | 0.63 | +0.48 |
| 50+ | 7011 | 0.46 | **18%** | +0.000 | 0.09 | 0.02 | 0% | 1.21 | 0.49 | 0.64 | +0.37 |

- **Played-move deviation (played ≠ visit-argmax): 50% / 30% / 18%.** This is
  *comparable-to-slightly-lower* than the predecessor hexgt run (58% / 36% / 35%),
  **not** higher — despite hexgnn's hotter temperature schedule. The reason: hexgnn's
  search is **sharper** (top1-visit 0.50/0.63/0.64 vs hexgt's 0.43/0.39/0.36), so
  there is simply less off-top mass for temperature to sample. The sharper search
  partially absorbs the hotter temperature.
- **Value cost of deviations ≈ zero.** Mean ΔV of deviating moves is ≈0 (even
  marginally negative — sampled moves are, on average, value-equal to the top-visit
  move). P(ΔV>0.3) ≤ 3% in every band. **Catastrophic outcome-flips** (top-visit
  clearly winning V>0.25 → played move clearly losing V<−0.25): **4 of 17,750 =
  0.02%.** Exploration is essentially never turning a won position into a lost one
  within a single move.
- Interpretation: deviations are between **near-equal candidate moves**. That is the
  healthy regime — the model is trying alternative *reasonable* moves, not blundering.

---

## 3. Phase 2 — attribution: noise vs temperature

The two knobs act on different things:
- **Dirichlet noise** diversifies what gets **searched** → shapes the **policy
  target** and root exploration. Effect measured by re-running MCTS.
- **Temperature** diversifies what gets **played** → writes the ±1 z outcomes at
  λ=0. Effect is a post-hoc transform on the stored visit policy (no search needed).

### 3a. Temperature sweep (predicted; post-hoc over the visit policies)

E[ΔV] is the expected per-move value cost of sampling at temperature T over the
top-k; "samplingH" is the entropy of the temperature-applied selection distribution.

| band | T | E[dev] | E[ΔV] | P(Δ>0.1) | P(Δ>0.3) | samplingH |
|---|---|---|---|---|---|---|
| 0–20 | 0.3 | 0.22 | −0.000 | 0.02 | 0.00 | 0.66 |
| 0–20 | 0.6 | 0.35 | −0.001 | 0.03 | 0.00 | 1.19 |
| 0–20 | 1.0 | 0.45 | −0.001 | 0.04 | 0.00 | 1.76 |
| 20–50 | 0.3 | 0.13 | −0.000 | 0.01 | 0.00 | 0.37 |
| 20–50 | 0.6 | 0.24 | −0.001 | 0.02 | 0.00 | 0.75 |
| 20–50 | 1.0 | 0.34 | −0.002 | 0.04 | 0.01 | 1.21 |
| 50+ | 0.3 | 0.13 | −0.000 | 0.01 | 0.00 | 0.36 |
| 50+ | 0.6 | 0.23 | −0.001 | 0.02 | 0.00 | 0.73 |
| 50+ | 1.0 | 0.33 | −0.002 | 0.04 | 0.01 | 1.21 |

**Temperature is "free" in per-move value terms** (E[ΔV] ≈ 0 at every T up to 1.0,
because the top-k moves are value-near-equal). Its real effect is purely on the
**deviation rate and selection entropy** — i.e. on how stochastic the *played* line
is, hence on **z-label variance** at λ=0. Cooling the mid/late temperature buys
sharper, less-noisy targets at negligible per-move value cost.

### 3b. Dirichlet `eps` sweep (live MCTS, 512 visits × 4 seeds, total_alpha=6.6)

Effect on the **searched** visit distribution (= the policy training target);
"P(top-move ≠ eps0)" = how often noise changes which move the search would pick.

| band | eps | visit H | effN | top1-visit | P(top-move ≠ eps0) |
|---|---|---|---|---|---|
| 0–20 | 0.00 | 2.29 | 16.6 | 0.36 | — |
| 0–20 | 0.15 | 2.40 | 18.1 | 0.34 | 0.19 |
| 0–20 | **0.30** | 2.46 | 18.9 | 0.33 | **0.27** |
| 0–20 | 0.45 | 2.48 | 18.8 | 0.33 | 0.42 |
| 20–50 | 0.00 | 1.48 | 8.0 | 0.58 | — |
| 20–50 | **0.30** | 1.74 | 10.3 | 0.54 | **0.06** |
| 20–50 | 0.45 | 1.80 | 10.8 | 0.52 | 0.10 |
| 50+ | 0.00 | 1.89 | 11.8 | 0.47 | — |
| 50+ | **0.30** | 2.01 | 13.1 | 0.45 | **0.20** |
| 50+ | 0.45 | 2.01 | 12.7 | 0.46 | 0.25 |

**Dirichlet is doing healthy work and is well-targeted:**
- It changes the searched top move **27%** of the time in the opening (the place
  diversity is most valuable), but only **6%** in the mid-game — i.e. it broadens
  the policy target everywhere (effN up ~15–30%) **without overriding the decisive
  mid-game move**. The late-game 20% is on diffuse near-terminal positions and is
  benign.
- This is *less* disruptive to the chosen move than the prior hexgt BC-seed
  decomposition (40% / 20% / 44% top-move flips) — because hexgnn's prior/visit is
  sharper. eps=0.30 is slightly above the proven 0.25 baseline but squarely in the
  healthy range.

**Attribution conclusion:** the Dirichlet knob is fine and well-shaped. The
**temperature** knob is the one carrying the bulk of the *played-line*
stochasticity (50/30/18% deviation), and that stochasticity is what feeds z-label
variance at λ=0 — for a benefit (opening line diversity) that is already saturated
by other means (see §4).

---

## 4. Phase 3 — cross-epoch statistics

### Opening diversity (full population, 512 games/epoch)

| ep | len med / mean | %<30 | uniq m1 / m2 / 3-move | open-H (3-move, nats) | top-3 share |
|---|---|---|---|---|---|
| 0  | 73 / 89.6 | 1%  | 1 / 34 / 440 | 6.04 | 2% |
| 8  | 35 / 41.2 | 28% | 1 / 34 / 425 | 5.98 | 2% |
| 20 | 33 / 41.0 | 33% | 1 / 34 / 344 | 5.71 | 3% |
| 28 | 35 / 59.7 | 26% | 1 / 34 / 392 | 5.89 | 2% |
| 32 | 41 / 65.4 | 20% | 1 / 81 / 450 | 6.06 | 2% |
| 36 | 53 / 76.5 | 14% | 1 / 75 / 425 | 5.98 | 2% |
| 41 | 43 / 58.4 | 16% | 1 / 71 / 364 | 5.73 | 5% |

- **Opening diversity is near-maximal and stable across the whole run.** 3-move
  openings: 344–450 unique out of 512 games (67–88% unique), entropy ≈ 5.6–6.1 nats
  (max for ~450 classes is ~6.1), top-3 lines only 2–5% of games. **No opening
  collapse, ever.** Temperature is *not* needed to manufacture opening variety — the
  candidate structure + Dirichlet already saturate it.
- Move-1 is always unique=1 (Hexo's first move is structurally forced — same as
  hexgt). Move-2 jumped 34→70–81 at ep32, which is the **`n` candidate-radius
  increase** (2→5→4) widening the move-2 option set, *not* an exploration-knob
  effect.
- **Game length is recovering, not degenerating:** 73 (ep0) → 33–35 (ep8–28) →
  43–53 (ep32–41); %<30-ply games fell 33%→16%. The recovery tracks the visits
  512→1024 bump (ep29) and the `n` changes. Games are getting *longer and more
  decisive* as the model learns — the opposite of an over-exploration pathology.

### z-label noise from temperature (λ=0)

Across the 17,750 positions: 31% of played moves deviate from the visit-argmax, but
only **0.02%** are catastrophic single-move outcome-flips. The subtler cumulative
effect (many small value-neutral deviations tilting a close game's eventual winner)
is not directly measurable, but it is **bounded** by the moderate corr(v,z)
(0.25–0.48): the value head still extracts a learnable signal from the z-labels, so
the temperature noise is *tolerable*, not *ruinous*. It is, however, a real and
avoidable tax that scales with the temperature floor/halflife.

### Exploration vs strength

The SealBot eval is currently uninformative (0–2.5% win rate; the model is a 200k
from-scratch net still early in training) and the vs-dense_cnn eval is skipped
(checkpoint missing). So there is no clean per-epoch eval signal to correlate
against. The available learning signals — falling loss, recovering/lengthening
games, moderate value calibration, saturated opening diversity — all point to a run
that is **learning and exploring adequately**; weakness is attributable to model
size + from-scratch + early epochs, **not** to the exploration settings.

---

## 5. Comparison to norms and to the prior hexgt run

| knob | hexgnn live | hexgt **C1** (proven, "about right, leaning high") | AlphaZero | KataGo |
|---|---|---|---|---|
| Dirichlet total_alpha | 6.6 (α_i≈0.02 @ ~270 cand) | 6.6 | α≈0.03 (Go) | ~10.8/avg-moves |
| Dirichlet eps | **0.30** | 0.25 | 0.25 | 0.25 |
| temperature start | 1.0 | 1.0 | 1.0 (first 30 plies) | ~0.8 |
| temperature decay | **exp, halflife 33** | **linear → 0.2 by ply 30** | argmax after ply 30 | exp, halflife ~19 |
| temperature floor | **0.3** | 0.2 | ~0 (argmax) | 0.2 |
| value target | **λ=0 hard-z** | λ=0.5 soft-z | hard-z | soft (utility) |

The standouts: hexgnn's **temperature stays hotter for longer than every reference**
(it is ≈0.67 at ply 30 where C1 is 0.2, AZ is argmax, and KataGo is ~0.4→0.2), its
**eps is slightly above** the 0.25 that all three references use, and it does this at
**λ=0 hard-z** — the most label-noise-sensitive of the four. The prior hexgt analysis
already flagged its own (cooler) settings as "leaning high" and explicitly advised "a
*slightly sharper* temperature for cleaner targets, NOT more noise." hexgnn currently
sits on the hotter side of that advice.

---

## 6. Verdict

**Over-exploring? Not harmfully — but the temperature schedule is hotter than is
justified, especially at λ=0. Call it "about right, leaning high," with one clean
tightening available.**

Evidence it is **not** over-exploring in any damaging way:
- value cost of deviations ≈ 0; catastrophic outcome-flips 0.02%;
- realized deviation is comparable-to-lower than the predecessor;
- opening diversity is near-maximal and stable (no collapse);
- game length is recovering and decisiveness rising;
- Dirichlet rarely overrides the decisive mid-game move (6%).

Evidence the temperature is **hotter than warranted**:
- played temperature ≈0.59 at the median game length; never reaches argmax;
- hotter than hexgt C1 (→0.2 by ply 30), AZ (argmax) and KataGo (floor 0.2);
- its only real payoff is played-line/opening diversity, which is **already
  saturated** by the candidate structure + Dirichlet;
- it runs at λ=0, where played-move stochasticity is the direct driver of z-label
  variance — so it pays the highest price for the least marginal benefit.

It is **not under-exploring** anywhere (opening near-max diverse; mid/late deviation
non-trivial; Dirichlet broadens all targets).

---

## 7. Recommendation (NOT applied — owner decides)

**Primary — cool the move temperature toward the proven C1 / KataGo profile.**
Lower `temperature_floor` **0.3 → 0.15–0.20** and shorten `temperature_halflife`
**33 → ~18–20**, so played temperature reaches its floor near the median game length
(~40 plies) instead of asymptotically.

Predicted, measurable effects (read off the §3a sweep):
- mid-band played-deviation **30% → ~18–22%**, late-band **18% → ~10–13%**;
- per-move value cost stays **≈0** (top-k moves are value-near-equal) → no self-play
  strength loss;
- **opening deviation barely moves** (early temp stays ~0.85–0.9) → opening diversity
  (the asset) is preserved;
- z-label variance and policy-target softness drop in the mid/late game → cleaner
  hard-z value targets and sharper policy targets exactly where positions are more
  forced and value-decided.

**Secondary / minor — `eps` 0.30 → 0.25** to match the proven baseline and all three
references. Predicted: opening searched-top-move-change 27% → ~22%, marginally
sharper opening targets. Low priority; current value is within the healthy band.

**Do NOT** reduce Dirichlet further, lower `total_alpha`, or shrink widening —
opening diversity and broad policy targets are assets here, and `forced_playout_k` /
widening are fine.

**Tradeoff to weigh:** cooling temperature reduces self-play *state-space coverage*.
For a from-scratch, still-weak model one could argue for keeping coverage high.
Counter: the change leaves the **opening** (where novel-state coverage matters most)
essentially untouched — early temperature stays ~0.9 — and only sharpens the
**mid/late** game, where positions are more forced and where λ=0 most rewards clean
outcomes. The expected downside is small and the target-quality upside is concrete.
If the owner prefers a single conservative step, **floor 0.3 → 0.2** alone (leave
halflife at 33) captures most of the benefit and is hard to argue against.

---

*Generated read-only on 2026-06-06 from `runs/hexgnn_rl_main1` (checkpoint e41,
shards/`.hxr` epochs 36–41, full-population opening stats epochs 0–41). Scripts:
`_hexgnn_explore.py`, `_hexgnn_dirichlet.py`. No run/model/training/launch state was
modified.*
