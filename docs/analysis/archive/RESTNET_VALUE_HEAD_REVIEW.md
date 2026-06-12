# ResTNet value head — calibration & P0/P1 zero-sum review

**Run:** `dense_cnn_restnet_main1` (live). ResTNet 96ch × 8-block
`R_R_R_T_R_R_T_R` (~1.5M params), `attention_scope=disk`, 65-bin distributional
value head over `[-1, 1]`, shared `ValueReduction` feeding `value` +
`stvalue_{2,6,16}` + `moves_left` tops (heads-v2). Self-play `search_visits=512`,
`games_per_epoch=256`, hard-outcome value target (`soft_z_lambda=0`).

**Model under review:** `checkpoints/epoch_000033.pt` (latest), evaluated on the
epoch-33 self-play positions (256 games, **0 truncated**, 29 807 sampled
decisions). Trend uses checkpoints 24→33.

**Scope:** the value head only — does it predict game state, and is it zero-sum
consistent between the two players (the "optimism" pathology prior lineages hit)?
Policy/search are out of scope.

**Bottom line:** **the value head is healthy and well-calibrated on the positions
it actually sees — it is NOT optimistic.** Aggregate predicted value tracks the
realized outcome to within +0.019; the reliability curve is monotonic; confidence
sharpens cleanly toward the end of the game (sign-accuracy 92% within 5 decisions
of terminal, ~61% at 80+ decisions out, which is correct — the winner genuinely
is near-undecided that early). The headline P0/P1 zero-sum probe returns
`mean(v(P0)+v(P1)) = +0.50`, which **looks** like optimism — **but that number is
an out-of-distribution artifact of the owner-swap method, not a property of the
deployed head.** The swap creates board states that cannot occur in real play (it
flips a hard stone-count parity invariant), and the entire +0.50 comes from the
head's evaluation of those *impossible* boards. On the real perspective the head
is, if anything, mildly *pessimistic*. Two genuine, actionable findings remain: a
real first/second-player asymmetry the head correctly (slightly under-)tracks, and
a freshly-added **`moves_left` head that is not yet learning** (correlation ≈ 0).

> **Status: analysis only. No run/config/code/checkpoint was modified.** Every
> probe was **CPU-only** (`CUDA_VISIBLE_DEVICES=""`) and **read-only**; the live
> GPU training under WSL was not touched and shared no GPU. Recommendations are
> proposals, not applied changes.

---

## How this was measured

- **Harness:** `scripts/_value_head_review.py` (CPU, read-only). It loads a
  checkpoint into a freshly-constructed `RestnetNetwork` (architecture taken from
  the run `manifest.json`), and reads the per-game compact `.npz` shards with the
  production `compact_io.read_compact_shard`. Each position is expanded to the
  dense 13×41×41 input by the **same** Python encoder training uses
  (`input.build_input_planes`), and the value is decoded with the **same**
  `losses.decode_binned_value` (`softmax(logits) · linspace(-1,1,65)`).
- **Ground truth:** the stored target `value` is the **hard game outcome**
  `z ∈ {−1,+1}` from the side-to-move's perspective (`soft_z_lambda = 0`,
  confirmed in `selfplay.finalize_game_samples` and by the data — every decisive
  row is exactly ±1). So "does the value head predict game state" reduces to:
  does decoded `v_pred` match the eventual winner `z`?
- **P0/P1 zero-sum probe** (the requested check): at **FirstStone** positions
  (between-turns, the only phase where "the other player to move" is well-defined)
  the *identical board* is evaluated from both perspectives via a facts-level
  **owner swap** — flip `current_player` and swap `own_hot`↔`opponent_hot`; the
  stones / recency / colour planes re-derive their own/opp assignment from the
  absolute owner vs `current_player`, so they flip automatically.
  `opponent_last_turn` is perspective-relative and its swapped replacement is not
  stored, so it is ablated on **both** sides to keep the comparison symmetric.
  This mirrors the prior hexgt/hexgnn optimism tooling (`_optimism_main3.py`,
  `scripts/_optimism.sh`), adapted to the dense encoder.
- **Cross-check:** the recomputed value cross-entropy on epoch-33 data (0.576)
  sits right next to the trainer's own logged `loss_components.value` for epoch 33
  (0.633 — measured over D6-augmented, policy-surprise-duplicated training
  batches), confirming the pipeline reproduces the training objective.

---

## 1. Calibration — does the value head predict game state? **Yes.**

Epoch-33 checkpoint on 29 807 epoch-33 decisions (all decisive):

| metric | value | reading |
|---|---|---|
| sign accuracy (predicts eventual winner) | **69.6 %** | over *all* plies, incl. near-opening |
| MAE \|v_pred − z\| | 0.800 | z is ±1, so 0.80 ≈ a 0.60-confident average call |
| RMSE | 0.889 | |
| value cross-entropy | 0.576 | ≈ trainer-logged 0.633 |
| mean v_pred | **+0.030** | vs mean z **+0.012** → aggregate bias **+0.018** |
| mean \|v_pred\| | 0.364 | cautious/hedged on average (correct — see below) |
| mean distribution entropy | 0.59 nats | vs 4.17 uniform → distributions are peaked |

**Reliability is monotonic and close to the diagonal** (predicted bucket →
realized mean outcome):

| predicted v bucket | n | mean v_pred | realized mean z |
|---|---:|---:|---:|
| [−1.0,−0.6) | 2 891 | −0.696 | **−0.772** |
| [−0.6,−0.3) | 4 745 | −0.451 | −0.413 |
| [−0.3,−0.1) | 3 620 | −0.191 | −0.303 |
| [−0.1,+0.1) | 5 076 | +0.001 | −0.099 |
| [+0.1,+0.3) | 3 959 | +0.203 | +0.237 |
| [+0.3,+0.6) | 7 548 | +0.437 | +0.459 |
| [+0.6,+1.0) | 1 968 | +0.835 | **+0.886** |

The tails are mildly **under-confident** (it says −0.70 / +0.84 when reality is
−0.77 / +0.89) — the safe direction to err. There is no bucket where the head
claims a win and loses on net.

**Confidence sharpens correctly toward the end of the game** — the single
clearest sign of a healthy value head:

| decisions-to-end | n | MAE | mean \|v\| | sign acc |
|---|---:|---:|---:|---:|
| 0–5 | 1 136 | **0.302** | 0.740 | **92.3 %** |
| 5–15 | 2 747 | 0.566 | 0.529 | 83.1 % |
| 15–40 | 6 791 | 0.745 | 0.388 | 76.1 % |
| 40–80 | 8 412 | 0.844 | 0.320 | 68.3 % |
| 80+ | 10 721 | 0.914 | 0.300 | **60.6 %** |

These self-play games average ~117 decisions, so the 80+ band is essentially the
opening/early-middlegame. 60.6 % sign-accuracy and a hedged \|v\|≈0.30 there is
**the right behaviour** — the eventual winner is genuinely close to a coin-flip
that far out — and 92 % / \|v\|≈0.74 near terminal shows the head commits when the
result is actually knowable.

---

## 2. P0/P1 zero-sum probe — the +0.50 is an out-of-distribution artifact

The requested check: evaluate the identical FirstStone board from both players'
perspectives and test `v(P0)+v(P1) ≈ 0`. **Raw result (epoch 33, 6 000 FirstStone
positions):**

```
mean(v(P0)+v(P1)) = +0.4985   median +0.3714   stdev 0.465
both perspectives predict a win (v>0.05): 1428/6000 = 24%
|sum| > 0.5: 2386/6000
```

Taken at face value that is "optimism" (both sides lean toward winning). **It is
not.** Decomposing the sum by which evaluation is on- vs off-distribution:

| quantity | value | what it is |
|---|---:|---|
| `mean v(real perspective)` | **−0.154** | the actual side-to-move — *on-distribution* |
| `mean v(swapped perspective)` | **+0.652** | the other player re-encoded — *off-distribution* |
| sum | +0.498 | = −0.154 + 0.652 |

**The entire +0.50 comes from the swapped (off-distribution) evaluation. On the
real perspective the head is slightly *negative*, not optimistic.**

Why the swap is off-distribution — a hard invariant, not a soft skew: at a
FirstStone position the side to move **always has exactly one fewer stone than the
opponent**. Measured directly on the data:

```
player0-to-move FirstStone: n=1123  own−opp stones = −1 for 1123/1123 (100%)
player1-to-move FirstStone: n=1039  own−opp stones = −1 for 1039/1039 (100%)
```

(This is forced by Hexo's two-placements-per-turn structure.) The owner-swap keeps
the stones and relabels owners, so the swapped board shows the side to move **one
stone *ahead*** — a material configuration that occurs with probability **0** in
real FirstStone play. The value head has correctly learned, on-distribution, that
"more stones than the opponent → ahead," so when handed these impossible
boards it returns inflated values (+0.65). That inflation *is* the +0.50.

**The owner-swap zero-sum test is therefore structurally confounded for this
dense_cnn encoding** (the prior hexgt/hexgnn lineage's references — old head
+0.82, pretrain seed −0.058 — carried the same confound and are not
apples-to-apples). The trustworthy zero-sum statement is the on-distribution one:
**aggregate `mean(v_pred) − mean(z) = +0.019`** over 29 807 real positions — i.e.
**no measurable optimism on the boards the head is actually asked to evaluate.**

---

## 3. Genuine finding — a real first/second-player asymmetry (head tracks it)

Splitting the calibration set by side to move:

| side to move | n | mean v_pred | mean z (realized) | sign acc |
|---|---:|---:|---:|---:|
| player0 | 15 373 | −0.243 | **−0.331** | 67.8 % |
| player1 | 14 434 | +0.322 | **+0.377** | 71.5 % |

Player1 wins materially more often in current self-play, and the head **correctly
encodes the direction** (negative for P0, positive for P1). It slightly
**regresses both toward zero** (predicts −0.243 / +0.322 where reality is
−0.331 / +0.377) — the same mild under-confidence seen in the reliability tails,
not a bias. This asymmetry is real signal, not a value-head defect; it lines up
with the role-asymmetric opening structure noted in prior opening-diversity work.

There is also a small **within-turn phase bias**: FirstStone evaluations average
−0.084 while SecondStone average +0.149 (both against realized z ≈ 0). The head
runs ~0.15 more optimistic right after you place the first stone of your turn,
before the opponent's reply is accounted for — a small, intuitively-explicable
skew worth a glance but not alarming.

---

## 4. Auxiliary heads

- **`moves_left` head — NOT learning yet (actionable).** Decoded remaining-decisions
  vs the true count: **correlation −0.067**, MAE 68.9 decisions, mean prediction
  113 vs true 75. It is emitting a near-constant, slightly-high value with no real
  dependence on the position. This head was only just added (commit *"Add
  moves_left head and shared ValueReduction"*) and its cap was widened to 512
  **today**; at weight 0.1 it is early, but right now it contributes nothing
  useful. Worth re-checking in a few epochs; if the correlation stays ≈ 0, the cap
  (512 vs measured median ~80) is the first suspect — most targets land in the
  bottom ~15 % of the `[0,512]→[−1,1]` support, so resolution is poor.
- **`stvalue_{2,6,16}` (short-term value) — fine.** MAE 0.167 / 0.171 / 0.189,
  sign-accuracy 66–68 %, mean prediction +0.025…+0.035 against targets ≈ 0. A
  faint positive lean (the same mild optimism echo) but well-behaved; low MAE is
  expected since the game rarely resolves within 2–16 plies so these targets
  cluster near the bootstrap value.

---

## 5. Trend across training (post-migration checkpoints)

The heads-v2 migration rebuilt the value stack at **epoch 23** (per-head
`ValueBinnedHead` → shared `ValueReduction`), so checkpoints ≤ 22 load with a
randomly-initialised value head and are **not comparable** (they score at chance —
CE ≈ ln 65, \|v\| ≈ 0 — and are excluded below). On a fixed 4 258-position /
2 000-FirstStone sample:

| epoch | MAE | sign acc | value CE | mean \|v\| | swap-sum (OOD) | first-player bias |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.886 | 57.7 % | 0.759 | 0.487 | −0.261 | −0.07 |
| 28 | 0.779 | 71.6 % | 0.551 | 0.359 | +0.759 | −0.36 |
| 32 | 0.830 | 75.5 % | 0.569 | 0.255 | +0.338 | −0.22 |
| 33 | 0.745 | **77.6 %** | **0.522** | 0.377 | +0.505 | −0.44 |

**Calibration is improving** since the migration (sign-accuracy 58 → 78 %,
value-CE 0.76 → 0.52). The swap-sum oscillates (−0.26 → +0.76 → +0.34 → +0.50) and
the first-player bias grows in magnitude — both expected: the swap-sum is the
off-distribution artifact from §2 (so it tracks how confidently the head reads
material, which is *increasing* as it calibrates), and the bias grows because
player1's real edge sharpens as self-play matures. Neither is a regression.

---

## Findings & recommendations

1. **Value head is healthy — no optimism pathology.** On real positions it is
   well-calibrated, monotonic, and aggregate-unbiased (+0.019). No action needed;
   this is the opposite of the broken +0.82 head from the earlier lineage.
2. **Retire / reinterpret the owner-swap zero-sum probe for dense_cnn.** Because
   FirstStone stone-count parity is a hard function of whose turn it is, the swap
   is intrinsically off-distribution and its `v(P0)+v(P1)` cannot be read as
   optimism. Use the **on-distribution** `mean(v_pred) − mean(z)` (and the
   reliability curve) as the zero-sum/optimism gauge instead. If a true
   same-board two-perspective test is wanted, it needs an engine-level
   construction that preserves a legal parity, not an owner relabel.
3. **Watch `moves_left`.** It is not learning (corr ≈ 0). Re-evaluate in a few
   epochs; if flat, reconsider `MOVES_LEFT_CAP=512` (median target ~80 → most mass
   in the bottom bins) or the 0.1 weight.
4. **The first/second-player gap is real and growing.** Not a value-head problem,
   but worth tracking — if P1's edge keeps widening it speaks to opening balance,
   not calibration.

## Caveats

- All numbers are epoch-33 self-play positions (the head's own distribution); this
  measures calibration on-policy, not against an external oracle.
- The swap probe ablates `opponent_last_turn` on both sides (symmetric) and, per
  §2, is off-distribution by construction — its absolute value is not meaningful,
  only used here to *demonstrate* the artifact.
- Reproduce with `CUDA_VISIBLE_DEVICES="" python scripts/_value_head_review.py`
  (writes `scripts/_value_head_review_out.json`).
