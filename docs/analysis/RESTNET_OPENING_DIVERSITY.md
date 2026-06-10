# ResTNet self-play — opening diversity investigation

**Run:** `dense_cnn_restnet_main1` (ResTNet 96ch `R_R_R_T_R_R_T_R`, ~1.51M params,
`attention_scope=disk`, `search_visits=512`, 256 games/epoch, warm-started from an
HF human-corpus prefit). **Analysed at epoch 30 trained** (epoch 31 self-play in
progress). Read-only; CPU-only; the live run was not disturbed.

Code/config: `E:\Hexo-BotTrainer-hexgt`. Run data: `E:\Hexo-BotTrainer\runs\dense_cnn_restnet_main1`.

---

## TL;DR

The owner's observation — *"one stone near the origin and another near/at max
placement distance, almost always"* — is **real and quantified**, but the precise
picture is more specific than "the model's opening collapsed":

1. **The first stone of every game is engine-forced to the origin** `(0,0)` —
   it is a single, deterministic placement, not a choice. (256/256 games, every
   epoch.) So "a stone near origin" is partly a **rule**, not a learned habit.
2. The pattern the owner sees is **player1's first full (2-stone) turn**: by epoch
   30, player1 plays its **FirstStone on the n=8 legal edge (hex-dist 7–8 from
   origin) in ~79% of games**, then its **SecondStone adjacent to origin
   (dist 1) in ~81%**. The "one near, one at the edge" signature rose from
   **7.8% (ep1) → 85.9% (ep29)**.
3. **Player0's first full turn is the opposite**: both stones clamp tightly onto
   the origin (dist ≤3 in 94–100% of games); its FirstStone reached **0.00 bits of
   D6-canonical entropy at ep29** (literally one canonical cell).
4. This is **primarily a learned policy/value strategy, not a search artifact, not
   inherited from the prefit, and not an obvious featurization bug.** The network's
   *raw policy prior* already concentrates ~96% of its mass on the edge ring for
   player1's FirstStone (argmax dist = 8 in 140/140 probed boards); MCTS search
   actually *softens* it. The pattern **emerged during RL** (epoch 1 openings were
   diffuse) and is **role-asymmetric** (player1 spreads, player0 consolidates),
   which is the signature of strategy rather than an encoding pull toward the edge.
5. **Diversity is in direction, not shape.** Openings still rotate freely around
   the origin (D6-canonical distinct FirstStone cells ≈ 22 at ep30); what is
   stereotyped is the **radius and the role-pattern**, which sharpened over
   training (diversity peaked at ep5–10, then narrowed).
6. **It has not hurt strength.** Eval win-rate vs SealBot is at its **peak at
   ep30 (50/64 = 78%)**, up from 31→39→50 over ep24/27/30.

**Verdict:** a **partial, shape-level diversity narrowing** — a genuine collapse
for a few specific decisions (player0 FirstStone), but layered on preserved
directional diversity and, so far, *effective* play. The most likely cause is the
**learned value/policy** (search follows the prior), amplified by an opening that
is still **drifting/re-stabilizing** (player1's SecondStone prior flipped
edge→origin between ep25 and ep30; net value on identical boards swung −0.87→+0.22).
It is **not** an exploration-floor bug: opening moves are already sampled near
temperature 1.0 and direction is diverse. If broader opening coverage is wanted
for training robustness, the lever is **opening-specific temperature/Dirichlet or
an opening randomizer**, not a change to the architecture or value head.

---

## 1. Turn structure (a prerequisite the observation depends on)

Hexo plays **two stone placements per turn**, but the **opening turn is a single
forced stone**. Verified by aligning the `.hxr` realized action sequence with the
per-game `.npz` compact shard (`turn_index`, `phase`, `current_player`,
`first_stone`):

| ply (action idx) | player | phase | meaning |
|---|---|---|---|
| `action[0]` | player0 | — | **forced origin `(0,0)`** (single stone; not a recorded decision) |
| `action[1]` | player1 | FirstStone | player1's first **free** placement |
| `action[2]` | player1 | SecondStone | player1's second placement (legal within n=8 of either stone) |
| `action[3]` | player0 | FirstStone | player0's first full-turn placement |
| `action[4]` | player0 | SecondStone | … |

Consequence: **legal moves are the hex-disk of radius n=8 around existing stones.**
For player1's FirstStone the only stone is the origin, so the legal set is the
radius-8 disk = **216 cells**, and **"max placement distance" = the dist-8 edge of
that disk.** So the owner's "near or at max placement distance" is, precisely,
**the n=8 legal frontier**.

Data source: per-epoch `epoch_NNNNNN.hxr` records store the **full 256 games** (not
a subset) — verified for ep5/15/25/29/30. Diversity is reported over all 256.

---

## 2. Quantified opening distribution (epoch 30, n=256 games)

Distances are hex-distance from origin. "A" = FirstStone of that turn, "B" =
SecondStone.

### Player1's first full turn (`action[1]`, `action[2]`) — the owner's pattern

| metric | value |
|---|---|
| FirstStone (A) dist from origin | mean **6.97**, median 7, **dist 7–8 in 202/256 (79%)** |
| FirstStone dist histogram | `{1:4, 3:5, 4:11, 5:14, 6:20, 7:85, 8:117}` |
| SecondStone (B) dist from origin | mean **2.13**, median 1, **dist 1 in 208/256 (81%)** |
| dist between the two stones | mean 7.15, median 7 |
| **"one near (≤3) + one at edge (≥7)"** | **177/256 (69%)** |
| A far (≥6) & B near (≤3) | **192/256 (75%)** |
| distinct FirstStone cells | raw **106**, D6-canonical **22** |
| distinct (A,B) pairs | raw 197, D6-canonical 118 |
| most-common D6-canonical (A,B) | shared by 10/256 (3.9%) |

→ Player1's opening is **"FirstStone on the n=8 edge, SecondStone back at the
origin."** The direction varies (22 canonical first cells; entropy 3.84 bits), the
**radius does not**.

### Player0's first full turn (`action[3]`, `action[4]`)

| metric | value |
|---|---|
| FirstStone (A) dist | mean **1.42**, **dist 1 in 226/256 (88%)** |
| SecondStone (B) dist | mean **1.83**, dist ≤2 in 232/256 |
| both stones near origin (≤3) | **240/256 (94%)** |
| distinct FirstStone cells | raw 21, D6-canonical 14 (ep30); **D6-canon = 1 at ep29** |
| most-common D6-canonical (A,B) | `((-1,0),(-1,1))` shared by 82/256 (32%); 54% at ep29 |

→ Player0 **consolidates around the origin** — the opposite spatial behaviour to
player1, on a nearly identical board. This role-asymmetry is the key evidence that
the edge preference is **strategic**, not a blanket "edge is attractive" artifact.

---

## 3. Diversity trend across epochs (collapse check)

`p1_dA/dB` = player1 First/Second-stone mean dist; `NE%` = "one-near-one-edge"
fraction; `canonAB` = distinct D6-canonical (A,B) pairs; `top%` = share of the
single most common canonical opening.

```
 ep  p1_dA  p1_dB  p1_NE%  p1_canonAB  p1_HAB  p1_top% | p0_dA  p0_dB  p0_NE%  p0_canonAB  p0_top%
  1   3.44   3.60    7.8%        69     4.84    30.5%  |  4.00   3.93    9.4%       72      28.9%
  5   5.68   5.31   21.1%       153     6.90     8.2%  |  6.31   6.15   19.1%      215       8.2%   <- diversity peak
 10   5.90   5.27   16.0%       179     7.16     6.2%  |  4.60   5.15   20.3%      184       7.8%
 15   6.87   6.80    2.7%        77     5.61     7.4%  |  3.06   5.25   27.7%      162      12.5%
 20   7.55   7.55    9.0%       139     6.35    11.3%  |  1.49   2.83   12.5%       62      38.3%
 25   7.74   6.08   29.7%       182     7.28     2.0%  |  1.02   1.52    1.2%       12      57.4%   <- player0 collapse
 29   7.59   2.02   85.9%        85     6.00     3.9%  |  1.00   1.29    0.0%        7      54.3%
 30   6.97   2.13   69.1%       118     6.45     3.9%  |  1.42   1.83    1.6%       32      32.0%
```

Reading:
- **Diversity peaked around ep5–10** (player0 had 215 distinct canonical openings),
  then **narrowed** — classic RL self-play sharpening.
- **Player1's opening shape changed over time, not just narrowed:** early-mid
  training it pushed *both* stones outward (`p1_dB` rose to ~7.5 at ep20 →
  "both-far"); then **between ep25 and ep29 the SecondStone snapped back to the
  origin** (`p1_dB` 6.08 → 2.02), producing the current "edge + origin" signature
  (`NE%` 29.7% → 85.9%). The opening theory is **still moving**.
- **Player0's FirstStone genuinely collapsed**: D6-canonical entropy → **0.00 bits
  at ep29** (one canonical cell). This is the one decision that is fully
  stereotyped.

---

## 4. Root cause — hypothesis tests

Each hypothesis was tested against data. For the prior-vs-search test, the ep25 and
ep30 checkpoints were run forward **on CPU** over real ep30 opening boards
(reconstructed via `expand_sample`), comparing the **net policy prior** to the
**stored MCTS visit policy** (`pol_act/pol_w`). "edge mass" = prior/visit mass on
legal cells at dist ≥7; "eff#moves" = `exp(entropy)`.

### H1 — Search/exploration artifact (prior diffuse, search concentrates)? → **Rejected.** The prior already carries the pattern.

| ep30 net, player1 FirstStone | PRIOR | VISIT |
|---|---|---|
| argmax dist-from-origin | **8.0 (140/140)** | 5.6 |
| edge mass (dist ≥7) | **0.958** | 0.771 |
| eff #moves | 97.1 | 50.6 |
| top-1 mass | 0.021 | 0.059 |

The **raw policy prior** puts ~96% of its mass on the radius-8 edge ring and its
argmax is *always* dist 8. The prior is high-entropy in absolute terms (eff ≈ 97)
because the edge ring has ~90 cells and the prior is roughly uniform *over the
ring* — i.e. "play somewhere on the frontier, direction free." **Search does not
create the edge preference; it slightly pulls in toward dist 4–6** (visit argmax
mean 5.6). So this is a **policy/value property**, with MCTS following (and mildly
softening) it.

### H2 — Inherited from the HF prefit? → **Rejected.** The pattern emerged during RL.

- Epoch-1 self-play (init directly from the prefit) is **diffuse**: player1
  `dA`/`dB` ≈ 3.4/3.6, "one-near-one-edge" only **7.8%** (vs 85.9% at ep29). The
  edge-spread habit is *not* present right after the prefit; it **grows over the RL
  epochs**.
- (The exact HF-prefit checkpoint referenced in the config,
  `…_prefit/restnet_hf_prefit.pt`, is no longer on disk, so ep1 self-play is used
  as the post-prefit behavioural proxy.)

### H3 — A legitimate, value-driven learned strategy? → **Largely supported, with caveats.**

- The prior is edge-locked **and value-positive** (ep30 net value for the player1
  opening ≈ +0.22 to +0.41, current-player perspective).
- It is **role-asymmetric and context-dependent**: player1 (moving into space far
  from player0's lone origin stone) maximises frontier reach; player0 (already
  owning the origin) consolidates around it. A blanket encoding bias toward the
  edge could not produce opposite behaviour on near-identical boards.
- **Caveat — it is not yet converged.** The player1 SecondStone prior **flipped**
  from edge (ep25 argmax dist 6.4, edge mass 0.60) to origin (ep30 argmax dist 1.0,
  edge mass 0.08). On identical boards, net value swung from **−0.87 (ep25 net) to
  +0.22 (ep30 net)** — a large recalibration, consistent with the ep23 value-head
  rework (`ValueReduction` + moves-left head) still settling. The strategy is
  **effective but drifting**, not a stable solved opening.

### H4 — Crop/featurization bias toward the n=8 edge? → **Unlikely as the driver.**

"Max placement distance" *is* the n=8 legal frontier, and the input does carry a
normalised center-distance plane and a legal-move plane that make the frontier
salient. But H3's role-asymmetry argues against the encoding *causing* the edge
attraction: player0 sees the same frontier features and does the opposite. The
frontier is the *substrate* of the strategy (you can only extend reach by playing
the legal edge), not an artifactual magnet.

### H5 — Exploration too low to escape it? → **No (for direction); partially (for radius).**

- Opening sampling temperature is **≈1.0** at plies 1–4: `temperature=1.0` with an
  adaptive half-life of `0.25 × expected_game_length` (~25 plies at current ~100-ply
  games), so the opening is barely decayed. Direction is therefore **already
  diverse** (106 raw / 22 canonical FirstStone cells). Raising temperature will not
  "uncollapse" direction — it is not collapsed.
- Root Dirichlet: `noise_fraction=0.25`, `total_alpha=10.83` → **per-move
  α ≈ 10.83/216 = 0.05** at the opening (spiky noise on a few random cells). This
  injects exploration but, over 512 sims, **value backups dominate and pull visits
  back to the edge**, so the *radius* stays locked. Exploration is doing its job on
  direction; it is not strong enough (by design) to override the value signal on
  radius/shape.

---

## 5. Verdict

- **Is the owner's observation real?** Yes. The dominant visible opening is
  **origin stone (forced) + a stone on the n=8 edge**, plus player1's
  origin-adjacent SecondStone — present in ~70–86% of recent games. "Almost always"
  is accurate for the *shape*.
- **Is it a pathology or a learned opening?** **Mostly a learned, value-driven
  opening**, with a **partial genuine collapse** for specific decisions (player0
  FirstStone → 0-bit canonical entropy) and an **unstable/drifting** quality
  (player1 SecondStone theory flipped ep25→30; large value recalibration). It is
  **not** a search artifact, **not** prefit-inherited, **not** an exploration-floor
  bug, and **not** clearly a featurization bug.
- **Is it currently harmful?** **No measurable harm to strength** — eval vs SealBot
  is at its peak (78%) at ep30. The risk is **latent**: training on an increasingly
  narrow opening distribution can reduce robustness to off-distribution openings and
  can self-reinforce, and the directional-only diversity may mask a thinning of
  strategic variety.

---

## 6. Recommendations (proposed — not applied)

Apply **only if broader opening coverage / training robustness is a goal**; strength
does not currently demand it. All are self-play-data levers; none touch the
architecture, value head, or training targets.

1. **Opening-specific temperature anchor (lowest risk, recommended first).** Add a
   temperature schedule that holds a higher value (e.g. **1.3–1.5**) for plies
   **0–6**, decaying to the current schedule afterward. The player1 FirstStone
   *visit* policy already has real mass at dist 4–6 (`{4:24,5:22,6:36,...}`), so
   higher opening temperature will sample those instead of the pure edge.
   *Predicted:* FirstStone edge fraction drops from ~79% toward ~50–60%, SecondStone
   less origin-locked; negligible strength risk (only the played-move sampling
   changes — search and targets are untouched).

2. **Opening-specific Dirichlet boost.** Raise `root_dirichlet_noise_fraction` to
   **~0.35–0.40** (and/or raise per-move α above the current 0.05) for the first
   few plies only. *Predicted:* more radial spread enters the root visit
   distribution; moderate effect (value backups will still recover some edge
   preference), best combined with (1).

3. **Opening randomizer / mini-book (most reliable coverage).** For a fraction of
   self-play games (e.g. **25%**), force the first 1–2 free placements to be sampled
   from a deliberately widened distribution (temperature-2 over the prior, or a
   curated set of opening shapes). *Predicted:* guarantees the net trains against
   diverse openings → more robust responses, at the cost of some games spent on
   weaker openings. Use this if the concern is *training-data coverage* rather than
   *live-play variety*.

4. **Monitor, don't intervene yet (also defensible).** Given strength is improving
   and the opening is mid-transition, an equally valid choice is to **re-measure at
   ep ~35–40** (re-run §2–§3). If diversity keeps narrowing *and* eval win-rate
   stalls or regresses, escalate to (1)+(3); if strength keeps climbing, the
   stereotype is an effective learned opening and needs no fix.

---

## Appendix — method & reproducibility

- **Opening distribution / trend (§2–3):** parsed `epoch_NNNNNN.hxr` via
  `hexo_utils.records.HexoRecordFile` + `hexo_engine.types.unpack_coord_id` (full
  256 games/epoch). Distances are hex-distance; diversity folded under the order-12
  **D6** symmetry about the origin (rotation+reflection) to remove the ~12×
  inflation from free board rotation — essential, since raw cell counts otherwise
  hide a fixed *shape* behind varying *direction*.
- **Prior-vs-visit (§4 H1, H3):** `dense_cnn_restnet` checkpoints ep25/ep30 loaded
  on CPU into `RestnetNetwork(**config)`; real opening boards rebuilt from the
  per-game `.npz` compact shards via `expand_sample(symmetry=0)`; `forward_policy_value`
  prior compared to the stored MCTS visit policy on the same boards. Net value
  decoded from the 65-bin head. `policy_keys_missing=0` for all loads (clean).
- **Strength trend (§ TL;DR/5):** `diagnostics/dense_cnn.evaluation.epoch_*.json`
  (`wins`/`games`, 64 games vs SealBot every 3 epochs).
- **Config (§1, §4 H5):** `runs/dense_cnn_restnet_main1/manifest.json`.
- Analysis was read-only and CPU-only; the live training run was not disturbed. A
  parallel read-only investigation of the search/exploration knobs is documented in
  [`RESTNET_EXPLORATION_KNOBS.md`](RESTNET_EXPLORATION_KNOBS.md).
